---
title: "2-4. SwiGLU Deep Dive"
weight: 4
---

# 2-4. SwiGLU — Gated FFN Deep Dive

## 개념 / 동기

Vanilla FFN: $\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x}\mathbf{W}_1)\mathbf{W}_2$

**SwiGLU** (Shazeer, 2020 — "GLU Variants Improve Transformer"):
$$
\text{SwiGLU}(\mathbf{x}) = (\text{SiLU}(\mathbf{x}\mathbf{W}_{\text{gate}}) \odot (\mathbf{x}\mathbf{W}_{\text{up}})) \mathbf{W}_{\text{down}}
$$

- **SiLU** (= Swish, $\beta=1$): $\text{SiLU}(x) = x \cdot \sigma(x) = x / (1 + e^{-x})$
- **Gated Linear Unit** family — gating mechanism으로 비선형성을 "곱"으로 주입

실증적으로 Perplexity가 GELU FFN 대비 일관되게 개선.
Llama, Mistral, PaLM, Qwen 모두 채택.

## 수식 — 왜 Gating인가

단순 MLP의 비선형성은 ReLU/GELU처럼 **한 입력에 한 함수**:
$$
y_i = f(\mathbf{w}_i^\top \mathbf{x})
$$

GLU는 **두 linear projection의 곱**:
$$
y_i = f(\mathbf{w}_{\text{gate},i}^\top \mathbf{x}) \cdot (\mathbf{w}_{\text{up},i}^\top \mathbf{x})
$$

→ $x$의 서로 다른 선형 projection이 서로를 "검열"하는 구조.
표현력 측면에서 더 강력 (bilinear form 표현 가능).

## 파라미터 Budget 조정

Vanilla FFN (2 linears): $d_{\text{ff}} = 4 d_m$
- 파라미터: $d_m \cdot 4 d_m + 4 d_m \cdot d_m = 8 d_m^2$

SwiGLU (3 linears): 파라미터를 맞추려면 $d_{\text{ff}}$를 조정
- 파라미터: $d_m \cdot d_{\text{ff}} \times 2 + d_{\text{ff}} \cdot d_m = 3 d_m d_{\text{ff}}$
- $3 d_m d_{\text{ff}} = 8 d_m^2 \Rightarrow d_{\text{ff}} = \frac{8}{3} d_m$

실제로는 hardware alignment 때문에 반올림:
```
Llama-3-8B: d_m = 4096, d_ff = 14336   (8/3 × 4096 = 10922.67 → 14336, 128 배수)
Llama-3-70B: d_m = 8192, d_ff = 28672  (8/3 × 8192 = 21845.33 → 28672)
```

## 연산 분해

```
Input: x [S, d_m]

[1] Gate projection:   gate = x @ W_gate^T    shape [S, d_ff]
[2] Up projection:     up   = x @ W_up^T      shape [S, d_ff]
[3] SiLU:              g    = silu(gate)      shape [S, d_ff]
[4] Hadamard:          mid  = g * up          shape [S, d_ff]
[5] Down projection:   out  = mid @ W_down^T  shape [S, d_m]
```

FLOPs (per token):
- Gate: $2 d_m d_{ff}$
- Up: $2 d_m d_{ff}$
- SiLU: $\sim 3 d_{ff}$ (exp approx)
- Hadamard: $d_{ff}$
- Down: $2 d_m d_{ff}$
- **Total: $\sim 6 d_m d_{ff}$** (vanilla FFN의 $4 d_m d_{ff}$ 대비 1.5배이지만, $d_{ff}$가 $\frac{2}{3}$이라 실제 총 FLOPs는 거의 동일)

## 핵심 최적화 — Gate와 Up의 GEMM Fusion

두 GEMM `x @ W_gate^T`와 `x @ W_up^T`는 같은 input $x$를 읽는다.
**Weight를 concat하여 하나의 GEMM으로 실행**:

```python
# W_gate_up: concat된 weight
W_gate_up = torch.cat([W_gate, W_up], dim=0)  # [2 * d_ff, d_m]

gate_up = x @ W_gate_up.T  # [S, 2 * d_ff]
gate, up = gate_up.chunk(2, dim=-1)
```

**이점**:
- Input $x$ HBM read 1회로 감소 (2회 → 1회)
- GEMM kernel launch 1회 절감
- Tensor core utilization 개선 (더 큰 GEMM)

이를 **MergedColumnParallelLinear** (vLLM/SGLang 용어)라 부른다.

## silu_and_mul — Fused Elementwise Kernel

Naive 구현:
```
g = silu(gate)           # read gate, write g  → 2 × S × d_ff 바이트
out = g * up             # read g, read up, write out → 3 × S × d_ff 바이트
Total: 5 × S × d_ff HBM traffic + 2 kernel launches
```

Fused `silu_and_mul`:
```cuda
// pseudo
__global__ void silu_and_mul(T* out, const T* gate, const T* up, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  T g = gate[i];
  T u = up[i];
  T silu_g = g / (1 + expf(-g));
  out[i] = silu_g * u;
}
// Read gate, read up, write out: 3 × S × d_ff bytes (40% 절감)
// 1 kernel launch
```

A100에서 `d_ff = 14336`, `S = 512`:
- Naive: $5 \times 512 \times 14336 \times 2B = 70$ MB HBM
- Fused: $3 \times 512 \times 14336 \times 2B = 42$ MB HBM
- HBM 2039 GB/s → 이론적으로 각 34.3 μs vs 20.6 μs

## FlashInfer on A100 — SwiGLU FFN 경로

```python
import torch
import torch.nn.functional as F
import flashinfer

d_model = 4096
d_ff = 14336
num_tokens = 512

hidden = torch.randn(num_tokens, d_model, dtype=torch.bfloat16, device="cuda")
W_gate_up = torch.randn(2 * d_ff, d_model, dtype=torch.bfloat16, device="cuda")
W_down = torch.randn(d_model, d_ff, dtype=torch.bfloat16, device="cuda")

# ── [1] Gate+Up fused GEMM (cuBLAS) ──
gate_up = F.linear(hidden, W_gate_up)   # [512, 2*14336 = 28672]
# cuBLAS: cublasLtMatmul(M=512, N=28672, K=4096)
#   A100 bf16 tensor core (HMMA), tile 128×256×32
#   Compute-bound: AI = K/2 = 2048 >> 153 (ridge)

# ── [2] silu_and_mul (FlashInfer fused kernel) ──
ffn_mid = flashinfer.activation.silu_and_mul(gate_up)
# 입력: gate_up [512, 28672]
# 내부 split: gate = gate_up[:, :14336], up = gate_up[:, 14336:]
# 출력: [512, 14336]
# 내부 CUDA kernel: flashinfer::silu_and_mul
#   - Memory-bound: 3 × 512 × 14336 × 2B = 42 MB HBM

# ── [3] Down projection (cuBLAS) ──
output = F.linear(ffn_mid, W_down)      # [512, 4096]
# cublasLtMatmul(M=512, N=4096, K=14336)
```

### SGLang에서

```
LlamaMLP.forward():
  → gate_up = MergedColumnParallelLinear(hidden, W_gate_up)
    → [cuBLAS] cublasLtMatmul — fused Gate+Up GEMM
  → mid = silu_and_mul(gate_up)
    → [CUDA] flashinfer::silu_and_mul OR sglang custom silu_and_mul_kernel
  → out = RowParallelLinear(mid, W_down)
    → [cuBLAS] cublasLtMatmul
```

## A100에서의 성능 Breakdown

Llama-3-8B, 1 layer, **prefill S=2048**:
```
Gate+Up GEMM:    2×2048×14336×4096 FLOPs = 241 GFLOPS → ~0.77 ms (312 TFLOPS 이론)
silu_and_mul:    2048 × 28672 × 2B × 3 = 352 MB HBM → ~0.17 ms
Down GEMM:       2×2048×4096×14336 = 241 GFLOPS → ~0.77 ms
Total FFN: ~1.7 ms
```

**Decode (B=1)**:
```
Gate+Up GEMM:    M=1, weight 235 MB → memory-bound ~115 μs
silu_and_mul:    28672 × 2B × 3 = 172 KB → ~0.1 μs (launch overhead 지배)
Down GEMM:       M=1, weight 117 MB → ~58 μs
Total FFN: ~175 μs (대부분 weight read)
```

→ **Decode에서 FFN weight read가 가장 큰 HBM traffic.**
이것이 Llama-3 같은 Dense LLM의 decode throughput 상한을 결정.
MoE (S3)가 이걸 해결하려는 동기가 됨.

## 변형 비교

| 이름 | $f$ | 채택 모델 |
|------|------|-----------|
| ReGLU | ReLU | — |
| GEGLU | GELU | GPT-NeoX |
| **SwiGLU** | SiLU | **Llama, Mistral, Qwen, PaLM** |
| Bilinear | Identity (gating only) | — |

SwiGLU가 실증적으로 가장 안정적.

## Examples

{{< hint info >}}
TODO: `examples/swiglu_vs_gelu.py` — 수식 검증 + PPL 비교 (작은 모델)
TODO: `examples/profile_silu_and_mul.py` — fused vs non-fused 성능
{{< /hint >}}
