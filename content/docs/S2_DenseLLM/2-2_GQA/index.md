---
title: "2-2. GQA / MQA"
weight: 2
---

# 2-2. Grouped-Query Attention (GQA / MQA)

## 개념 / 동기

S1의 MHA는 Q, K, V가 각각 $n_h$개 head를 가진다.
**KV cache 크기가 $n_h \times d_h$ per token per layer에 비례**하여 메모리 병목이 됨.

### 계보

| Variant | Q heads | KV heads | Ratio |
|---------|---------|----------|-------|
| MHA | $n_h$ | $n_h$ | 1:1 |
| **MQA** (Shazeer 2019) | $n_h$ | 1 | $n_h$:1 |
| **GQA** (Ainslie 2023) | $n_h$ | $n_{kv}$ | $n_h/n_{kv}$:1 (= group size $g$) |

MQA는 극단적으로 KV head 1개만 사용 → 품질 저하.
GQA는 그 중간지점을 찾은 것 — Llama-2부터 표준.

## 수식

$n_h$ query heads, $n_{kv}$ KV heads, group size $g = n_h / n_{kv}$.

**Step 1: Projection (weight shape가 달라짐)**
$$
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{W}_Q \in \mathbb{R}^{d_m \times n_h d_h}
$$
$$
\mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{W}_K \in \mathbb{R}^{d_m \times n_{kv} d_h}
$$
$$
\mathbf{V} = \mathbf{X} \mathbf{W}_V, \quad \mathbf{W}_V \in \mathbb{R}^{d_m \times n_{kv} d_h}
$$

**Step 2: Head Mapping**

Query head $i$ ($0 \le i < n_h$)는 KV head $\lfloor i / g \rfloor$와 attention을 계산:
$$
\text{Attn}_i = \text{softmax}\!\left(\frac{\mathbf{Q}_i \mathbf{K}_{\lfloor i/g \rfloor}^\top}{\sqrt{d_h}}\right) \mathbf{V}_{\lfloor i/g \rfloor}
$$

즉, 같은 group 내의 $g$개 query heads가 **하나의 KV head를 공유**.

## Llama-3-8B 예시

```
d_model = 4096
num_q_heads = 32, num_kv_heads = 8, head_dim = 128
group_size g = 32 / 8 = 4

Weight shapes:
  W_Q: [4096, 32*128] = [4096, 4096]
  W_K: [4096, 8*128]  = [4096, 1024]   ← MHA 대비 1/4
  W_V: [4096, 8*128]  = [4096, 1024]   ← MHA 대비 1/4
  W_QKV (fused): [4096, 4096 + 1024 + 1024] = [4096, 6144]

KV cache per token per layer:
  2 × 8 × 128 × 2B = 4 KB   (MHA면 16 KB였음)
```

## 연산 분해

### QKV GEMM

```
Single fused GEMM:
  Input: X [num_tokens, d_model]
  Weight: W_QKV [d_model, d_q + d_k + d_v] = [4096, 6144]
  Output: [num_tokens, 6144]
    → split into Q [num_tokens, 32*128], K [num_tokens, 8*128], V [num_tokens, 8*128]
```

MHA 대비:
- QKV GEMM FLOPs: **50% 감소** (weight가 12288 → 6144)
- K, V는 이미 작게 생성되므로 RoPE, cache append도 그만큼 가볍다

### Attention

**Prefill**: 각 query token이 같은 KV stream과 attention 계산 → 기존과 동일한 kernel로 처리 가능 (단, head mapping 정보 전달)

**Decode**: KV cache read 양이 MHA 대비 $1/g$배
- A100 HBM BW bound인 decode에서 **$g$배 속도 향상** 가능
- 또는 같은 시간에 $g$배 큰 batch 가능

## GPU 커널 구현 — 두 가지 전략

### 전략 A: K, V replication (naive)

KV head를 $g$배 복제하여 MHA kernel에 그대로 통과:
```
K_replicated = K.repeat_interleave(g, dim=heads)  # [seq, n_kv*g, d_h] = [seq, n_h, d_h]
V_replicated = V.repeat_interleave(g, dim=heads)
→ 기존 MHA kernel 사용
```
**문제**: HBM에 실제로 $g$배 쓰므로 메모리 절감 이득을 잃음.

### 전략 B: Head mapping in kernel (실제 구현)

Attention kernel이 `num_q_heads, num_kv_heads`를 직접 받아 내부에서 index 매핑:
```cuda
// pseudo
int q_head_id = blockIdx.y;
int kv_head_id = q_head_id / group_size;  // group_size = n_q / n_kv

// K, V load는 kv_head_id 사용
float* K_ptr = K_cache + kv_head_id * stride_head;
// Q load는 q_head_id 사용
float* Q_ptr = Q + q_head_id * stride_head;
```
→ KV는 HBM에 1회만 저장, read도 $1/g$배. **이것이 FlashInfer/FlashAttention의 실제 구현.**

## FlashInfer on A100 — GQA 커널

### API는 MHA와 동일, 파라미터만 다름

```python
import torch
import flashinfer

# Llama-3-8B GQA: num_q_heads=32, num_kv_heads=8
num_q_heads = 32
num_kv_heads = 8
head_dim = 128

workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace, kv_layout="NHD", backend="auto"
)

# Paged KV cache — num_kv_heads로 할당 (MHA 대비 1/4 메모리)
k_cache = torch.zeros(1024, 16, num_kv_heads, head_dim,
                       dtype=torch.bfloat16, device="cuda")
v_cache = torch.zeros_like(k_cache)

prefill_wrapper.plan(
    qo_indptr=qo_indptr,
    paged_kv_indptr=kv_indptr,
    paged_kv_indices=kv_indices,
    paged_kv_last_page_len=kv_last_page_len,
    num_qo_heads=num_q_heads,      # ← 32
    num_kv_heads=num_kv_heads,     # ← 8  (여기서 GQA가 선언됨)
    head_dim_qk=head_dim,
    page_size=16,
    causal=True,
    sm_scale=1.0 / (head_dim ** 0.5),
    q_data_type="bfloat16",
)

q = torch.randn(total_tokens, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
#                             ^^^^^^^^^^^^ Q는 32 heads
attn_out = prefill_wrapper.run(q, (k_cache, v_cache))
# k_cache, v_cache는 8 heads — FlashInfer가 내부에서 group mapping 처리
```

### 내부 CUDA kernel의 동작

FlashInfer는 `num_q_heads, num_kv_heads` 값에 따라 적절한 kernel specialization을 선택:

```
n_q == n_kv:         MHA path
n_kv == 1:           MQA path (K,V broadcast)
1 < n_kv < n_q:      GQA path
  - Thread block이 (batch, q_head_group) 단위로 매핑
  - 같은 group의 q_heads는 동일 K,V tile을 shared memory에 공유
  - → KV read를 group 내에서 amortize
```

### 왜 GQA가 A100에서 특히 효과적인가

```
A100 decode에서의 bottleneck 분석:

MHA:  KV cache read per token = n_q × d_h × 2B × 2(K+V)
                              = 32 × 128 × 2 × 2 = 16 KB
      A100 HBM 2039 GB/s → 이론 최대 decode 속도 ∝ 1/16KB

GQA (g=4):  KV cache read = n_kv × d_h × 2B × 2
                          = 8 × 128 × 2 × 2 = 4 KB
      → decode 속도 4배, 또는 같은 속도에 batch 4배
```

실제로 Llama-3-8B가 A100 80GB에서 batch=32+ 로 serving 가능한 핵심 이유.

## SGLang에서의 호출 경로

```
LlamaAttention.forward()
  → qkv = linear(hidden, W_qkv)    # fused, weight shape [d_model, d_q+d_k+d_v]
  → q, k, v = qkv.split([d_q, d_k, d_v])   # k,v는 이미 small
  → q, k = apply_rotary_pos_emb(q, k)
  → radix_attention.forward(q, k, v, layer)
    → FlashInferAttnBackend.forward_extend()
      → prefill_wrapper.run(q, (k_cache, v_cache))
        # num_qo_heads=32, num_kv_heads=8이 plan() 시점에 지정됨
        → [CUDA] fa2_prefill_paged_run<GQA specialization>
```

## 성능 Trade-off — g 값의 선택

| Model | $n_q$ | $n_{kv}$ | $g$ | KV 절감 | 품질 영향 |
|-------|-------|---------|-----|---------|-----------|
| Llama-2-70B (MHA) | 64 | 64 | 1 | — | — |
| Llama-3-8B/70B | 32/64 | 8/8 | 4/8 | 4/8× | 거의 없음 |
| Mistral 7B | 32 | 8 | 4 | 4× | 거의 없음 |
| Qwen 2.5 | 28 | 4 | 7 | 7× | 거의 없음 |

$g=4 \sim 8$이 sweet spot으로 정착. $g > 16$부터 품질 저하 발생 (경험적).

## Examples

{{< hint info >}}
TODO: `examples/gqa_kv_cache_comparison.py` — MHA vs GQA KV cache 메모리 측정
TODO: `examples/gqa_decode_bandwidth.py` — decode 시 HBM 사용량 비교
{{< /hint >}}
