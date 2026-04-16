---
title: "2-3. RoPE (Rotary Position Embedding)"
weight: 3
---

# 2-3. Rotary Position Embedding (RoPE)

## 개념 / 동기

Vanilla Transformer의 sinusoidal PE는 **입력에 더하는** 방식:
$\mathbf{x}_t' = \mathbf{x}_t + \text{PE}(t)$.
이는 다음 한계가 있다:

1. 학습 중 본 position 범위를 벗어나면 외삽 품질 급락
2. Relative position 정보가 attention score에 간접적으로만 전달

**RoPE** (Su et al., 2021): Q와 K에 **회전 변환**을 적용.
두 토큰 간 attention score $\mathbf{q}_m^\top \mathbf{k}_n$이
자동으로 **상대 위치 $m-n$**에만 의존하게 된다.

Llama, Mistral, Qwen 등 거의 모든 현대 LLM의 표준.

## 수식

### 복소수 표현 (이해용)

2D 회전은 복소수 곱으로 표현: $\text{rotate}(\mathbf{x}, \theta) = \mathbf{x} \cdot e^{i\theta}$.

Head dim $d_h$를 2씩 묶어 $d_h/2$쌍의 2D 벡터로 봄.
각 쌍 $(x_{2i}, x_{2i+1})$에 다른 주파수 $\theta_i$로 회전:

$$
\theta_i = \text{base}^{-2i/d_h}, \quad \text{base} = 10000 \text{ (보통)}
$$

Position $m$에서:
$$
\text{RoPE}(\mathbf{x}, m)_{[2i, 2i+1]} =
\begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

### 핵심 성질 (증명 생략)

Attention score 계산 시:
$$
\text{RoPE}(\mathbf{q}, m)^\top \text{RoPE}(\mathbf{k}, n) = \mathbf{q}^\top \mathbf{R}_{m-n} \mathbf{k}
$$

즉, **절대 위치 $m, n$이 상대 위치 $m-n$으로 자동 변환**됨.
→ 외삽 성능 개선, KV cache의 K가 **position-aware**이지만 재계산 불필요.

## 실제 구현 — Real-valued Form

GPU에서는 복소수 연산 대신 real-valued form으로 구현:

두 가지 convention이 있음:

### Interleaved (원 논문)
$(x_0, x_1), (x_2, x_3), \dots$ 쌍을 회전:
```
x_new[2i]   = x[2i] * cos(mθ_i) - x[2i+1] * sin(mθ_i)
x_new[2i+1] = x[2i] * sin(mθ_i) + x[2i+1] * cos(mθ_i)
```

### Half-split (Llama/HuggingFace 표준)
앞 절반 $x_{[:d/2]}$과 뒤 절반 $x_{[d/2:]}$를 쌍으로:
```
x_new[:d/2] = x[:d/2] * cos(mθ) - x[d/2:] * sin(mθ)
x_new[d/2:] = x[:d/2] * sin(mθ) + x[d/2:] * cos(mθ)
```

수학적으로 동일 (basis 재배열), 하지만 메모리 접근 패턴이 다름.
**GPU에서는 half-split이 더 효율적** (연속 메모리 접근).

## 연산 분해

```
Input: Q [num_tokens, n_q_heads, d_h], K [num_tokens, n_kv_heads, d_h]
       positions [num_tokens]
       cos_cache, sin_cache [max_pos, d_h/2]

For each (token, head, i=0..d_h/2-1):
  x1 = q[token, head, i]
  x2 = q[token, head, i + d_h/2]
  c  = cos_cache[positions[token], i]
  s  = sin_cache[positions[token], i]
  q[token, head, i]        = x1 * c - x2 * s
  q[token, head, i + d_h/2] = x1 * s + x2 * c
```

- 순수 elementwise + broadcast cos/sin
- Arithmetic: 4 FMA per element
- Memory: read Q + read cos/sin + write Q
- **매우 memory-bound**

## 왜 cos/sin을 precompute하는가

$\theta_i$는 고정이므로 $\cos(m\theta_i), \sin(m\theta_i)$를 미리 계산해 cache:
```
max_position_embeddings = 8192  # 또는 그 이상
cos_cache = cos(positions[:, None] * theta[None, :])  # [max_pos, d_h/2]
sin_cache = sin(positions[:, None] * theta[None, :])
```
→ 매 forward에서 trig 연산 불필요, lookup만.

Llama-3는 `rope_theta = 500000` (긴 context), NTK scaling 등 변형 있음.

## FlashInfer on A100 — RoPE 커널

### 단독 API

```python
import torch
import flashinfer

num_tokens = 512
n_q_heads = 32
n_kv_heads = 8
head_dim = 128

q = torch.randn(num_tokens, n_q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
k = torch.randn(num_tokens, n_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
positions = torch.arange(num_tokens, dtype=torch.int32, device="cuda")

# Precomputed rope cache
rope_theta = 500000.0  # Llama-3
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
cos_cache = torch.cos(positions.float()[:, None] * inv_freq[None, :]).to(torch.bfloat16)
sin_cache = torch.sin(positions.float()[:, None] * inv_freq[None, :]).to(torch.bfloat16)

# In-place RoPE application
flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
    positions=positions,
    query=q,
    key=k,
    head_size=head_dim,
    cos_sin_cache=torch.cat([cos_cache, sin_cache], dim=-1),
    is_neox=True,   # Half-split (Llama/Neox) convention
)
# q, k가 in-place로 회전됨
# 내부 CUDA kernel: flashinfer::apply_rope
#   - thread: 1 element pair (i, i+d/2) of one (token, head)
#   - 매우 lightweight, 거의 memory-bound
```

### Attention kernel에 integration된 버전

FlashInfer는 RoPE를 **attention kernel 내부에서 fuse**할 수도 있다
(Q/K를 HBM에 다시 쓰지 않고 attention 계산에 바로 사용):

```python
prefill_wrapper.plan(
    ...,
    pos_encoding_mode="ROPE_LLAMA",   # ← attention 내부에서 RoPE 적용
    rope_scale=1.0,
    rope_theta=500000.0,
)
# q, k에 RoPE 미적용 상태로 넘겨도 내부에서 적용 후 attention 수행
```

그러나 KV cache에 **RoPE가 적용된 K를 저장**해야 decode에서 재사용 가능하므로,
실제로는 **cache append 전에 RoPE를 먼저 적용**하는 방식이 일반적:

```
pipeline:
  QKV linear → q, k, v
  → apply_rope(q, k)               ← RoPE 먼저
  → kv_pool.set_kv_buffer(k, v)    ← RoPE 적용된 k 저장
  → attention(q, k_cache, v_cache)
```

## SGLang에서의 호출 경로

```
LlamaAttention.forward()
  → qkv = linear(hidden, W_qkv)
  → q, k, v = qkv.split(...)
  → rotary_emb(positions, q, k)                          ← RoPE
    → vllm/sglang의 RotaryEmbedding 모듈
      → [CUDA] rotary_embedding_kernel (SGLang custom or FlashInfer)
  → kv_pool.set_kv_buffer(layer, cache_loc, k, v)        ← RoPE된 K 저장
  → flashinfer wrapper run (pos_encoding_mode="NONE")    ← 이미 적용됨
```

## 성능 특성 on A100

```
Llama-3-8B one layer, S=2048:
  Input: Q [2048, 32, 128] + K [2048, 8, 128] in bf16
  Total bytes: 2048 × (32 + 8) × 128 × 2 = 20 MB read/write

  RoPE kernel: ~30-50 μs
  Attention kernel: ~500-1000 μs

→ RoPE는 전체의 3-5% 정도. 최적화 우선순위는 낮지만 kernel launch 수가 많아
  (매 layer × 2회 QKV 처리) CUDA graph capture에 포함시키는 게 중요.
```

## 변형들 — Long Context 지원

Llama-3, Llama-3.1의 긴 context (128K+) 대응:

- **NTK-aware scaling**: 고주파 dim은 scale 줄이고, 저주파는 그대로
- **YaRN** (Peng et al., 2023): frequency별 다른 scaling + attention temperature
- **Llama 3.1 scaling**: $\theta$ 자체를 늘리는 방식 (500K → 더 큰 값)

이런 변형은 **cos/sin cache를 다시 계산**하는 것만으로 적용 가능 — 커널 레벨에서는 변화 없음.

## Examples

{{< hint info >}}
TODO: `examples/rope_convention_check.py` — interleaved vs half-split 동치성 검증
TODO: `examples/rope_extrapolation.py` — context 길이별 품질 곡선
{{< /hint >}}
