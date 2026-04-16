---
title: "S2. Dense LLM (Llama-style)"
weight: 2
bookCollapseSection: false
---

# S2. Dense LLM — Llama-style Architecture

Vanilla Transformer (S1)에서 현대 Dense LLM(Llama, Mistral, Qwen 등)로 넘어오면서
**정확도 손실 없이 inference 효율성을 극대화**하기 위한 여러 변형이 누적되었다.
S2에서는 그 **delta**들만 집중적으로 다룬다.

## Deltas from Vanilla

| 영역 | Vanilla | Llama-style | 동기 |
|------|---------|-------------|------|
| Normalization | LayerNorm (Post-LN) | **RMSNorm** + **Pre-LN** | 연산 절감, 학습 안정성 |
| Position | Absolute / Sinusoidal | **RoPE** (Rotary) | 외삽 성능, cache 친화성 |
| Attention | MHA | **GQA/MQA** | KV cache 메모리 절감 |
| FFN | GELU + 2 linears | **SwiGLU** + 3 linears | 성능 개선 |
| Bias | 있음 | 대부분 **없음** | 파라미터/연산 절감 |

## Subsections

| # | Topic | Description |
|---|-------|-------------|
| 2-1 | [Overview & What Changed]({{< relref "2-1_Overview" >}}) | Llama-style이 묶은 기법들, 전체 영향 |
| 2-2 | [GQA / MQA]({{< relref "2-2_GQA" >}}) | Grouped-Query Attention — KV cache 절감의 핵심 |
| 2-3 | [RoPE]({{< relref "2-3_RoPE" >}}) | Rotary Position Embedding — 수식과 fused 커널 |
| 2-4 | [SwiGLU Deep Dive]({{< relref "2-4_SwiGLU" >}}) | Gated FFN의 기하학적 의미와 커널 최적화 |
| 2-5 | [Kernel-Level Summary]({{< relref "2-5_KernelSummary" >}}) | vanilla 대비 FlashInfer/SGLang 경로 차이 총정리 |
