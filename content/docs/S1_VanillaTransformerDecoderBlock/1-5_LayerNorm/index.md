---
title: "1-5. LayerNorm"
weight: 5
---

# 1-5. Layer Normalization

## 개념 / 동기

Deep network의 학습 안정화를 위한 normalization.
BatchNorm과 달리 **단일 샘플의 feature dimension**에 대해 정규화하므로
autoregressive 생성에 적합.

## 수식

### LayerNorm (Original)

$$
\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$

where $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$, $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$

- $\gamma, \beta \in \mathbb{R}^{d}$: learnable scale, shift

### RMSNorm (현대 LLM 표준 — Llama, Mistral 등)

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma
$$

- Mean 계산 제거 ($\beta = 0$, center 안 함) → 연산 절약
- 실측에서 성능 차이 미미하면서 속도 이득

## 연산 분해

| Step | 연산 | 복잡도 |
|------|------|--------|
| Mean (LN only) | Reduction over $d$ | $O(d)$ |
| Variance / RMS | Reduction over $d$ | $O(d)$ |
| Normalize + Scale (+Shift) | Elementwise | $O(d)$ |

$d = 4096$ 기준 매우 작은 연산량이지만, **매 layer에 2번** (Pre-LN) 호출되므로
kernel launch overhead가 중요해짐.

## GPU 커널 매핑

### Fused LayerNorm / RMSNorm Kernel

```
1개의 CUDA kernel에서:
  1. Warp-level reduction으로 mean/variance(또는 RMS) 계산
  2. 같은 kernel에서 normalize + scale 적용
  3. Thread block 하나가 하나의 token(= d차원 벡터)을 처리

- d ≤ 1024: single warp reduction
- d > 1024: multi-warp reduction (shared memory 사용)
```

### Residual + LayerNorm Fusion

```
많은 구현에서 residual add를 LayerNorm kernel에 fuse:
  x' = x + attn_out          ← residual
  ln_out = RMSNorm(x')       ← normalize

→ HBM read/write 1회 절약 (x'를 HBM에 쓰지 않고 register에서 바로 normalize)
SGLang에서: fused_add_rmsnorm kernel
```

## FlashInfer on A100 — RMSNorm / Fused Kernels

### FlashInfer의 RMSNorm API

```python
import torch
import flashinfer

# Llama-3-8B: d_model=4096, bf16
num_tokens = 512
d_model = 4096

hidden = torch.randn(num_tokens, d_model, dtype=torch.bfloat16, device="cuda")
weight = torch.ones(d_model, dtype=torch.bfloat16, device="cuda")  # γ parameter

# ── 단순 RMSNorm ──
out = flashinfer.norm.rmsnorm(hidden, weight, eps=1e-6)
# 내부 CUDA kernel: flashinfer::rmsnorm
#   - 1 thread block = 1 token (d_model차원 전체 처리)
#   - d=4096 → 4 warps (128 threads), warp-level reduction으로 RMS 계산
#   - 1 pass: load x → compute sum(x²) → rsqrt → x * rsqrt * γ → write

# ── Fused Residual + RMSNorm (핵심 최적화) ──
residual = torch.randn(num_tokens, d_model, dtype=torch.bfloat16, device="cuda")
attn_out = torch.randn(num_tokens, d_model, dtype=torch.bfloat16, device="cuda")

# In-place: residual += attn_out, then RMSNorm(residual)
out = flashinfer.norm.fused_add_rmsnorm(attn_out, residual, weight, eps=1e-6)
# residual이 in-place로 업데이트됨 (residual = residual + attn_out)
# out = RMSNorm(updated residual)
#
# 내부 CUDA kernel: flashinfer::fused_add_rmsnorm
#   Without fusion (2 kernels):
#     [1] elementwise_add: read residual + attn_out → write residual' → 3 × num_tokens × d × 2B
#     [2] rmsnorm: read residual' → write out → 2 × num_tokens × d × 2B
#     Total HBM traffic: 5 × 512 × 4096 × 2 = 20 MB
#
#   With fusion (1 kernel):
#     read residual + attn_out → add in register → rmsnorm → write residual' + out
#     Total HBM traffic: 4 × 512 × 4096 × 2 = 16 MB  (20% 절약)
#     + kernel launch overhead 1회 절감
```

### A100에서의 성능 특성

```
RMSNorm은 순수 memory-bound:
  d=4096, bf16: 8KB per token (read) + 8KB (write) = 16KB per token
  Arithmetic: ~4096 mul + 4096 add + 1 rsqrt + 4096 fma ≈ 12K FLOPs
  AI = 12K / 16KB ≈ 0.75 FLOPs/Byte → 극도로 memory-bound

  512 tokens: 16KB * 512 = 8 MB transfer
  A100 HBM 2039 GB/s → 이론 ~4 μs
  실측: ~10-20 μs (kernel launch overhead 포함)
```

### SGLang에서의 호출 경로

```
LlamaDecoderLayer.forward():
  # Attention 전 RMSNorm
  hidden, residual = fused_add_rmsnorm(hidden, residual, attn_norm_weight)
  #   → flashinfer.norm.fused_add_rmsnorm()
  #     → [CUDA] flashinfer::fused_add_rmsnorm kernel

  ... (attention) ...

  # FFN 전 RMSNorm
  hidden, residual = fused_add_rmsnorm(attn_out, residual, ffn_norm_weight)

  ... (FFN) ...
```

> Decoder block당 fused_add_rmsnorm이 **2회** 호출됨.
> 32 layers → 64회. 개별 시간은 작지만 누적하면 전체의 ~3-5% 차지.

## Examples

> [!NOTE]
> TODO: `examples/layernorm_vs_rmsnorm.py` — 수식 검증 및 성능 비교

