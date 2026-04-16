---
title: "1-1. Overall Architecture"
weight: 1
---

# 1-1. Vanilla Transformer Decoder Block — Overall Architecture

## 개념

Transformer decoder block은 autoregressive language model의 기본 단위이다.
하나의 decoder block은 다음 두 sub-layer로 구성된다:

```
Input (hidden states)
  │
  ├─→ [LayerNorm] → [Self-Attention] → [+ Residual] ─→
  │                                                     │
  └─────────────────────────────────────────────────────┘
  │
  ├─→ [LayerNorm] → [FFN] → [+ Residual] ─→
  │                                          │
  └──────────────────────────────────────────┘
  │
Output (hidden states)
```

> 위 그림은 **Pre-LN** (LayerNorm을 sub-layer 앞에 배치) 구조이며,
> 현대 LLM (GPT-2 이후)의 표준이다. Original Transformer는 Post-LN이었다.

## Data Flow (Single Token, Single Layer)

입력: $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$ (하나의 토큰의 hidden state)

1. **LayerNorm**: $\hat{\mathbf{x}} = \text{LN}(\mathbf{x})$
2. **Self-Attention**: $\mathbf{a} = \text{Attn}(\hat{\mathbf{x}})$
3. **Residual**: $\mathbf{x}' = \mathbf{x} + \mathbf{a}$
4. **LayerNorm**: $\hat{\mathbf{x}}' = \text{LN}(\mathbf{x}')$
5. **FFN**: $\mathbf{f} = \text{FFN}(\hat{\mathbf{x}}')$
6. **Residual**: $\mathbf{x}'' = \mathbf{x}' + \mathbf{f}$

## 주요 파라미터

| Symbol | Meaning | Typical Values |
|--------|---------|----------------|
| $d_{\text{model}}$ | Hidden dimension | 4096, 5120, 8192 |
| $n_{\text{heads}}$ | Number of attention heads | 32, 40, 64 |
| $d_{\text{head}}$ | Per-head dimension ($d_{\text{model}} / n_{\text{heads}}$) | 128 |
| $d_{\text{ff}}$ | FFN intermediate dimension | $4 \times d_{\text{model}}$ 또는 $\frac{8}{3} \times d_{\text{model}}$ (SwiGLU) |
| $L$ | Number of layers | 32, 40, 80 |

## GPU 커널 관점 Preview

하나의 decoder block forward pass에서 launch되는 주요 커널 유형:

| 연산 | 커널 유형 | 대표 라이브러리 |
|------|-----------|-----------------|
| QKV / Output / FFN projection | GEMM | cuBLAS, CUTLASS |
| Attention score + softmax + value aggregation | Fused Attention | FlashInfer, FlashAttention (Triton) |
| LayerNorm | Reduction + Elementwise | Custom CUDA kernel |
| Residual add | Elementwise | 보통 fused (LayerNorm과 합쳐짐) |
| Activation (GELU/SiLU) | Elementwise | 보통 fused (FFN GEMM과 합쳐짐) |

> 각 연산의 상세는 subsection 1-2 ~ 1-6에서 다룬다.

## Examples

{{< hint info >}}
TODO: 간단한 PyTorch 참조 구현 — `examples/vanilla_decoder_block.py`
{{< /hint >}}
