---
title: "1-4. FFN (Feed-Forward Network)"
weight: 4
---

# 1-4. Feed-Forward Network (FFN)

## 개념 / 동기

Attention이 토큰 간 관계를 모델링한다면,
FFN은 각 토큰의 representation을 **독립적으로** 비선형 변환한다.
Position-wise이므로 토큰 간 interaction 없음.

## 수식

### Vanilla FFN (Original Transformer)

$$
\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2
$$

- $\mathbf{W}_1 \in \mathbb{R}^{d_m \times d_{ff}}$, $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_m}$
- $d_{ff} = 4 d_m$ (전통적)

### Gated FFN (SwiGLU — Llama, PaLM 등)

$$
\text{FFN}(\mathbf{x}) = (\text{SiLU}(\mathbf{x} \mathbf{W}_{\text{gate}}) \odot (\mathbf{x} \mathbf{W}_{\text{up}})) \mathbf{W}_{\text{down}}
$$

- $\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}} \in \mathbb{R}^{d_m \times d_{ff}}$, $\mathbf{W}_{\text{down}} \in \mathbb{R}^{d_{ff} \times d_m}$
- $d_{ff} = \frac{8}{3} d_m$ (파라미터 수 맞추기 위해)
- Gate와 Up을 하나의 fused GEMM으로 처리 가능: $\mathbf{W}_{gate\_up} \in \mathbb{R}^{d_m \times 2d_{ff}}$

## 연산 분해

| Variant | Step | Shape | FLOPs |
|---------|------|-------|-------|
| Vanilla | Up proj | $(S, d_m) \times (d_m, d_{ff})$ | $2 S d_m d_{ff}$ |
| | Activation | $(S, d_{ff})$ | $O(S d_{ff})$ |
| | Down proj | $(S, d_{ff}) \times (d_{ff}, d_m)$ | $2 S d_m d_{ff}$ |
| **Total** | | | $4 S d_m d_{ff}$ |
| SwiGLU | Gate+Up proj | $(S, d_m) \times (d_m, 2d_{ff})$ | $4 S d_m d_{ff}$ |
| | SiLU + Hadamard | $(S, d_{ff})$ | $O(S d_{ff})$ |
| | Down proj | $(S, d_{ff}) \times (d_{ff}, d_m)$ | $2 S d_m d_{ff}$ |
| **Total** | | | $6 S d_m d_{ff}$ |

FFN의 GEMM들은 **전체 decoder block FLOPs의 ~2/3**를 차지 (attention이 ~1/3).

## GPU 커널 매핑

### GEMM (Up/Gate/Down Projection)

```
cuBLAS: cublasGemmEx() / cublasLtMatmul()
  - Prefill: 큰 M (= S) → compute-bound, 높은 GPU utilization
  - Decode: M=1 (single token) → GEMV에 가까움, memory-bound
```

### Activation Fusion

```
SiLU + Hadamard product는 보통 별도 elementwise kernel로 launch
  - 또는 Gate+Up GEMM의 epilogue로 fuse (CUTLASS epilogue fusion)
  - SGLang/FlashInfer: 별도 custom CUDA kernel (silu_and_mul)
```

## FlashInfer on A100 — FFN 커널 경로

### GEMM: cuBLAS 호출 경로

SGLang에서 FFN projection은 PyTorch `torch.mm()` / `F.linear()`를 통해
최종적으로 cuBLAS를 호출한다.

```python
import torch
import torch.nn.functional as F

# Llama-3-8B SwiGLU FFN 파라미터
d_model = 4096
d_ff = 14336  # = 8/3 * 4096, rounded

# Gate + Up을 하나의 fused weight로 저장 (SGLang/vLLM 공통)
W_gate_up = torch.randn(d_model, 2 * d_ff, dtype=torch.bfloat16, device="cuda")
W_down    = torch.randn(d_ff, d_model, dtype=torch.bfloat16, device="cuda")

# ── Prefill (S=512 tokens) ──
hidden = torch.randn(512, d_model, dtype=torch.bfloat16, device="cuda")

# [1] Gate+Up Projection — single cuBLAS GEMM
gate_up = F.linear(hidden, W_gate_up.T)   # [512, 2*d_ff] = [512, 28672]
# 내부: cublasLtMatmul(M=512, N=28672, K=4096)
#   A100: Ampere HMMA (bf16 tensor core), tile 128x256x32 등
#   FLOPs: 2 * 512 * 4096 * 28672 = 120.3 GFLOPS → compute-bound

# [2] SiLU + Hadamard — elementwise kernel
gate, up = gate_up.chunk(2, dim=-1)        # 각 [512, d_ff]
ffn_mid = F.silu(gate) * up                # [512, d_ff]
# SGLang에서는 fused 'silu_and_mul' CUDA kernel 1회로 처리
# → memory-bound (단순 read → silu → multiply → write)

# [3] Down Projection — cuBLAS GEMM
output = F.linear(ffn_mid, W_down.T)       # [512, d_model]
# cublasLtMatmul(M=512, N=4096, K=14336)

# ── Decode (B=1 token) ──
hidden_1 = torch.randn(1, d_model, dtype=torch.bfloat16, device="cuda")
gate_up_1 = F.linear(hidden_1, W_gate_up.T)   # [1, 28672]
# cublasLtMatmul(M=1, N=28672, K=4096)
#   → 사실상 GEMV, memory-bound
#   → Weight 전체를 HBM에서 읽어야 함: 4096 * 28672 * 2B = 224 MB
#   → A100 HBM 2039 GB/s 기준 이론 ~0.11ms
```

### A100에서 FFN GEMM의 Compute vs Memory 분석

| Phase | GEMM | M | N | K | FLOPs | Weight Read | AI | Bound |
|-------|------|---|---|---|-------|-------------|-----|-------|
| Prefill (S=512) | Gate+Up | 512 | 28672 | 4096 | 120G | 224MB | 536 | **Compute** |
| Prefill (S=512) | Down | 512 | 4096 | 14336 | 60G | 112MB | 536 | **Compute** |
| Decode (B=1) | Gate+Up | 1 | 28672 | 4096 | 235M | 224MB | ~1 | **Memory** |
| Decode (B=1) | Down | 1 | 4096 | 14336 | 117M | 112MB | ~1 | **Memory** |

> AI (Arithmetic Intensity) = FLOPs / Bytes. A100 ridge point ≈ 153.
> Decode에서 batch size를 키우면 AI가 linear하게 증가 → B≥153이면 compute-bound 전환.

### SGLang에서의 FFN 구현 경로

```
LlamaDecoderLayer.forward()
  → LlamaMLP.forward()
    → gate_up_proj = MergedColumnParallelLinear(W_gate, W_up)  # 하나의 GEMM
      → [cuBLAS] cublasLtMatmul
    → silu_and_mul(gate_up_proj)
      → [CUDA] silu_and_mul_kernel  # SGLang custom elementwise
    → down_proj = RowParallelLinear(W_down)
      → [cuBLAS] cublasLtMatmul
```

## Examples

{{< hint info >}}
TODO: `examples/ffn_flops_comparison.py` — Vanilla vs SwiGLU FLOPs 비교
{{< /hint >}}
