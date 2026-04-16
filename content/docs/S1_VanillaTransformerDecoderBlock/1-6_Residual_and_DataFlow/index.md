---
title: "1-6. Residual & Data Flow"
weight: 6
---

# 1-6. Residual Connection & Full Data Flow

## 개념 / 동기

Residual connection (He et al., 2016)은 gradient vanishing 문제를 해결하여
수십~수백 layer를 쌓을 수 있게 한다.
$\mathbf{x}' = \mathbf{x} + f(\mathbf{x})$ 형태로, gradient가 identity path를 통해 직접 전파.

## 전체 Forward Pass — 커널 Launch 순서

Pre-LN Decoder Block 1개의 forward에서 실제로 launch되는 커널 시퀀스:

```
[1] fused_add_rmsnorm      ← residual(이전 layer) + RMSNorm
[2] qkv_gemm               ← W_QKV projection (cuBLAS)
[3] rope_kernel             ← Rotary Position Embedding (elementwise)
[4] kv_cache_append         ← K,V를 cache에 저장
[5] flash_attention         ← FlashInfer fused attention kernel
[6] output_proj_gemm        ← W_O projection (cuBLAS)
[7] fused_add_rmsnorm      ← residual + RMSNorm
[8] gate_up_gemm            ← W_gate_up projection (cuBLAS)
[9] silu_and_mul            ← SiLU activation + hadamard (elementwise)
[10] down_proj_gemm         ← W_down projection (cuBLAS)
```

> 위는 Llama-style (Pre-LN + SwiGLU + RoPE + GQA)의 예시.
> Vanilla Transformer는 [3] RoPE 없음, [8-9]가 단일 up_proj + GELU.

## 커널별 시간 비율 (대략적)

| 커널 | Prefill (S=2048) | Decode (B=1) |
|------|:---:|:---:|
| GEMM (QKV + O + FFN) | ~60-70% | ~70-80% |
| Flash Attention | ~15-25% | ~5-10% |
| LayerNorm / RMSNorm | ~2-5% | ~5-10% |
| Elementwise (activation, RoPE) | ~2-3% | ~3-5% |
| KV cache append | <1% | ~2-5% |

→ **GEMM이 항상 지배적**이나, Decode에서는 memory-bound GEMV로 바뀌면서
상대적으로 다른 커널들의 비중이 올라감.

## Memory Bandwidth 관점

각 커널의 HBM access:

| 커널 | Read | Write | Arithmetic Intensity |
|------|------|-------|----------------------|
| GEMM (large M) | Weight + Activation | Output | **High** (compute-bound) |
| GEMM (M=1) | Weight | Output | **Low** (memory-bound) |
| Flash Attention (prefill) | Q, K, V | O | Medium-High |
| Flash Attention (decode) | q, K_cache, V_cache | o | **Low** (memory-bound) |
| RMSNorm | Input | Output | Very Low |
| Elementwise | Input | Output | Very Low |

## Examples

{{< hint info >}}
TODO: `examples/kernel_timeline_diagram.py` — nsys trace를 시각화
{{< /hint >}}
