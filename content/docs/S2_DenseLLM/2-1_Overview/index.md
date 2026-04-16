---
title: "2-1. Overview & What Changed"
weight: 1
---

# 2-1. Dense LLM Overview — What Changed from Vanilla

## "Dense LLM"의 정의

본 study에서 **Dense LLM**은 MoE가 아닌, 모든 파라미터가 매 forward에 활성화되는
현대 Llama-style 아키텍처를 지칭한다. 대표 모델:

- **Llama 1/2/3** (Meta)
- **Mistral 7B**, **Mixtral의 dense 버전들**
- **Qwen 2/2.5** (Alibaba)
- **Gemma** (Google)

이들은 공통적으로 다음 "Llama recipe"를 공유한다:
**Pre-LN + RMSNorm + RoPE + GQA + SwiGLU + no-bias**

## 전체 Decoder Block 구조

```
x_in ─┐
      ├─→ RMSNorm ─→ GQA Attention ─→ + ─→ RMSNorm ─→ SwiGLU FFN ─→ + ─→ x_out
      │                               │                             │
      └───── residual ────────────────┘─── residual ────────────────┘
```

Vanilla (S1)과 비교하면:

| Step | Vanilla | Llama-style |
|------|---------|-------------|
| 1 | `LayerNorm(x)` | `RMSNorm(x)` — bias/mean 연산 제거 |
| 2 | QKV proj (with bias) | QKV proj (no bias), **Q head 수 > KV head 수 (GQA)** |
| 3 | (no positional op here) | **RoPE applied to Q, K** — position을 연산 중 주입 |
| 4 | Softmax attn | Softmax attn (동일) |
| 5 | Output proj (with bias) | Output proj (no bias) |
| 6 | `LayerNorm(x')` | `RMSNorm(x')` |
| 7 | FFN: Up → GELU → Down | **SwiGLU**: Gate, Up → SiLU×Hadamard → Down |

## 수치로 보는 영향 (Llama-3-8B 기준)

```
Vanilla MHA 기준 (가상):
  num_q_heads = 32, num_kv_heads = 32, head_dim = 128
  KV cache per token per layer = 2 × 32 × 128 × 2B = 16 KB
  32 layers, seq_len=4096: 16KB × 4096 × 32 = 2 GB per request

Llama-3-8B (GQA):
  num_q_heads = 32, num_kv_heads = 8, head_dim = 128   ← KV heads 4배 축소
  KV cache per token per layer = 2 × 8 × 128 × 2B = 4 KB
  32 layers, seq_len=4096: 4KB × 4096 × 32 = 512 MB per request

→ 메모리 4배 절감 = batch size 4배 가능 = throughput 극대화
```

## 커널 레벨에서 생기는 차이

Vanilla → Llama-style로 바뀌면서 **추가**되는 커널:
- `apply_rotary_pos_emb` (RoPE) — Q, K 각각에 적용
- `silu_and_mul` (SwiGLU 중간 단계) — gate ⊙ silu(up) 계산

**변형**되는 커널:
- Attention kernel은 같은 `fa2_*` family지만,
  **`num_q_heads ≠ num_kv_heads`** case에 해당하는 variant로 dispatch
- QKV GEMM의 weight shape: `[d_model, d_q + d_k + d_v]` 에서
  **`d_k = d_v = d_model × num_kv_heads / num_q_heads`** 로 축소

**제거**되는 커널:
- LayerNorm의 mean reduction path (RMSNorm은 RMS만 계산)
- Bias add (epilogue에 포함되던 것 제거)

## 왜 이 조합이 정답이 되었나

Llama recipe는 개별 변경이 모두 **inference-friendly**하다:

| 변경 | Compute 영향 | Memory 영향 |
|------|-------------|-------------|
| LayerNorm → RMSNorm | −5% | 동일 |
| Absolute PE → RoPE | +소량 (elementwise) | 동일 |
| MHA → GQA | Attention FLOPs ±0 | **−75% (KV cache)** |
| FFN → SwiGLU | +33% (weight 3개) | +33% |
| Bias 제거 | −elementwise | −small |

**핵심**: GQA가 unlocking technique.
KV cache가 줄어야 batch size를 키울 수 있고, batch가 커야 decode의 memory-bound를 완화.
나머지 변경은 학습 안정성/표현력 개선 + 소폭의 연산 절감.

## Subsections Preview

- **2-2 GQA**: head 수 mismatch를 attention 커널이 어떻게 처리하는가
- **2-3 RoPE**: 복소수 회전을 real-valued CUDA로 구현하는 방법
- **2-4 SwiGLU**: gated FFN의 fused kernel 설계
- **2-5 Kernel Summary**: 전체 경로 차이를 한눈에
