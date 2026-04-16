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

## Examples

{{< hint info >}}
TODO: `examples/ffn_flops_comparison.py` — Vanilla vs SwiGLU FLOPs 비교
{{< /hint >}}
