---
title: "1-2. Self-Attention"
weight: 2
---

# 1-2. Self-Attention Mechanism

## 개념 / 동기

Self-Attention은 시퀀스 내 모든 위치 간의 관계를 학습하는 메커니즘이다.
RNN과 달리 병렬 처리가 가능하며, 장거리 의존성을 직접 모델링할 수 있다.

## 수식

### Multi-Head Attention (MHA)

입력 $\mathbf{X} \in \mathbb{R}^{S \times d_{\text{model}}}$ (시퀀스 길이 $S$)에 대해:

**Step 1: QKV Projection**

$$
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V
$$

여기서 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$

**Step 2: Per-Head Split**

$\mathbf{Q}, \mathbf{K}, \mathbf{V}$를 $n_h$개 head로 분할:
$\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i \in \mathbb{R}^{S \times d_h}$, where $d_h = d_{\text{model}} / n_h$

**Step 3: Scaled Dot-Product Attention (per head)**

$$
\text{Attn}_i = \text{softmax}\!\left(\frac{\mathbf{Q}_i \mathbf{K}_i^\top}{\sqrt{d_h}}\right) \mathbf{V}_i
$$

**Step 4: Concat + Output Projection**

$$
\text{MHA}(\mathbf{X}) = \text{Concat}(\text{Attn}_1, \dots, \text{Attn}_{n_h}) \mathbf{W}_O
$$

## 연산 분해

| Step | 연산 | Shape | FLOPs (per head) |
|------|------|-------|-------------------|
| QKV Proj | GEMM | $(S, d_m) \times (d_m, d_m)$ | $3 \times 2 S d_m^2$ |
| $\mathbf{Q}\mathbf{K}^\top$ | GEMM (batched) | $(S, d_h) \times (d_h, S)$ → $(S, S)$ | $2 S^2 d_h$ |
| Softmax | Reduction + Exp | $(S, S)$ | $O(S^2)$ |
| Score $\times$ V | GEMM (batched) | $(S, S) \times (S, d_h)$ → $(S, d_h)$ | $2 S^2 d_h$ |
| Output Proj | GEMM | $(S, d_m) \times (d_m, d_m)$ | $2 S d_m^2$ |

**총 FLOPs** (all heads): $\approx 8 S d_m^2 + 4 S^2 d_m$

- $S \ll d_m$ (짧은 시퀀스): **GEMM 지배** (QKV/Output projection)
- $S \gg d_m$ (긴 시퀀스): **Attention score 지배** ($O(S^2)$ 항)

## GPU 커널 매핑

### QKV Projection & Output Projection → cuBLASLt GEMM

```
cublasGemmEx() 또는 cublasLtMatmul()
  - A: activation [S, d_model], B: weight [d_model, d_model]
  - 보통 QKV를 하나로 fuse: W_QKV ∈ R^{d_model × 3*d_model} → single GEMM
  - SGLang dense path는 cuBLASLt via gemm_and_bias (**1-8 참고**)
```

### Attention Core → FlashAttention / FlashInfer Fused Kernel

```
Q @ K^T → softmax → @ V 를 하나의 fused kernel로 실행
  - FlashInfer: flashinfer.prefill_with_paged_kv_cache() 등
  - O(S) memory (S×S score matrix를 materialization하지 않음)
  - Tiling: Q를 block 단위로 처리, online softmax
  - GPU에서: shared memory에 Q tile 로드 → K,V tile streaming → accumulate
```

### 왜 Fused Attention이 필요한가

Naive 구현시:
1. `Q @ K^T` → $(S, S)$ matrix를 HBM에 write → **$O(S^2)$ memory**
2. Softmax → HBM read/write
3. `Score @ V` → HBM read

FlashAttention: tiling + online softmax로 중간 결과를 SRAM(shared memory)에 유지
→ HBM access $O(S^2 d_h / M)$ (M = SRAM size), memory 사용 $O(S)$

## FlashInfer on A100 — Attention Kernel 상세

### Prefill Attention

```python
import torch
import flashinfer

# ── Setup (서버 시작 시) ──
workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace, kv_layout="NHD", backend="auto"  # A100 → fa2 (CUTLASS-based)
)

# ── Paged KV cache 구조 ──
# page_size=16일 때, 시퀀스 길이 100 → ceil(100/16) = 7 pages 사용
num_pages = 1024
page_size = 16
num_kv_heads = 32   # MHA: num_q_heads == num_kv_heads
head_dim = 128

k_cache = torch.zeros(num_pages, page_size, num_kv_heads, head_dim,
                       dtype=torch.bfloat16, device="cuda")
v_cache = torch.zeros_like(k_cache)

# ── 배치 2개: seq_len=[100, 200] ──
batch_size = 2
qo_indptr = torch.tensor([0, 100, 300], dtype=torch.int32, device="cuda")  # Q token 범위
kv_indptr = torch.tensor([0, 7, 20], dtype=torch.int32, device="cuda")     # page 범위
kv_indices = torch.arange(20, dtype=torch.int32, device="cuda")             # page IDs
kv_last_page_len = torch.tensor([100 - 6*16, 200 - 12*16],                 # = [4, 8]
                                 dtype=torch.int32, device="cuda")

# ── plan(): scheduling 결정 ──
prefill_wrapper.plan(
    qo_indptr=qo_indptr,
    paged_kv_indptr=kv_indptr,
    paged_kv_indices=kv_indices,
    paged_kv_last_page_len=kv_last_page_len,
    num_qo_heads=32,
    num_kv_heads=32,
    head_dim_qk=128,
    page_size=page_size,
    causal=True,              # autoregressive masking
    sm_scale=1.0 / (128 ** 0.5),  # 1/sqrt(d_h)
    q_data_type="bfloat16",
)

# ── run(): fused attention kernel launch ──
q = torch.randn(300, 32, 128, dtype=torch.bfloat16, device="cuda")  # [total_tokens, heads, dim]
attn_out = prefill_wrapper.run(q, (k_cache, v_cache))
# attn_out shape: [300, 32, 128] — 각 token에 대한 attention output

# 내부에서 launch되는 CUDA kernel:
#   flashinfer::fa2_prefill_paged_run<bfloat16, 128, /*causal=*/true>
#   - Q를 tile (e.g., 64 tokens) 단위로 분할
#   - 각 tile이 KV pages를 streaming하며 online softmax 수행
#   - Shared memory에 Q tile 상주, K/V는 global memory에서 stream
```

### Decode Attention

```python
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    workspace, kv_layout="NHD", backend="auto"
)

# Decode: 배치 4개, 각각 이전 context 길이가 다름
batch_size = 4
kv_indptr = torch.tensor([0, 7, 20, 25, 30], dtype=torch.int32, device="cuda")
kv_indices = torch.arange(30, dtype=torch.int32, device="cuda")
kv_last_page_len = torch.tensor([4, 8, 10, 3], dtype=torch.int32, device="cuda")

decode_wrapper.plan(
    indptr=kv_indptr,
    indices=kv_indices,
    last_page_len=kv_last_page_len,
    num_qo_heads=32,
    num_kv_heads=32,
    head_dim=128,
    page_size=16,
    sm_scale=1.0 / (128 ** 0.5),
    q_data_type="bfloat16",
)

# Decode: q는 batch당 1 token
q = torch.randn(4, 32, 128, dtype=torch.bfloat16, device="cuda")  # [batch, heads, dim]
attn_out = decode_wrapper.run(q, (k_cache, v_cache))
# attn_out shape: [4, 32, 128]

# 내부 CUDA kernel:
#   flashinfer::BatchDecodeWithPagedKVCache<bfloat16, 128>
#   - 각 request의 q (1 token)가 전체 KV cache pages를 scan
#   - Parallel reduction: 여러 thread block이 KV pages를 분담 → partial results merge
#   - Memory-bound: q는 작고, KV cache read가 dominant
#   - A100 HBM BW 2039 GB/s가 bottleneck
```

### Prefill vs Decode — 커널 동작 차이 요약

| | Prefill | Decode |
|---|---------|--------|
| **Q shape** | `[total_tokens, heads, dim]` | `[batch, heads, dim]` |
| **FlashInfer wrapper** | `BatchPrefillWithPagedKVCacheWrapper` | `BatchDecodeWithPagedKVCacheWrapper` |
| **내부 커널** | `fa2_prefill_paged_run` | `BatchDecodeWithPagedKVCache` |
| **Tiling 전략** | Q tiles × KV streaming | KV pages parallel reduction |
| **Bottleneck** | Compute (HMMA on A100 SM) | Memory BW (HBM read) |

### SGLang에서의 실제 호출 경로

```
SGLang model forward
  → RadixAttention.forward()
    → FlashInferAttnBackend.forward_extend()   # prefill path
      → kv_pool.set_kv_buffer(layer, loc, k, v)  # cache write
      → prefill_wrapper.forward(q, kv_buffer, causal=True, sm_scale=...)
        → [CUDA] fa2_prefill_paged_run kernel

    → FlashInferAttnBackend.forward_decode()   # decode path
      → kv_pool.set_kv_buffer(layer, loc, k, v)
      → decode_wrapper.forward(q, kv_buffer, sm_scale=...)
        → [CUDA] BatchDecodeWithPagedKVCache kernel
```

## Examples

> [!NOTE]
> TODO: `examples/attention_flops_calculator.py` — shape별 FLOPs 계산
> TODO: `examples/naive_vs_flash_attention.py` — memory/compute 비교

