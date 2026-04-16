---
title: "S1. Vanilla Transformer Decoder Block"
weight: 1
bookCollapseSection: false
---

# S1. Vanilla Transformer Decoder Block

"Attention Is All You Need" (Vaswani et al., 2017)에서 제안된 Transformer decoder block의 inference 연산을
수식부터 GPU 커널까지 분해하여 학습한다.

## Subsections

| # | Topic | Description |
|---|-------|-------------|
| 1-1 | [Overall Architecture]({{< relref "1-1_OverallArchitecture" >}}) | Decoder block 전체 구조와 data flow |
| 1-2 | [Self-Attention]({{< relref "1-2_SelfAttention" >}}) | QKV projection → Score → Softmax → Output — GEMM 커널 매핑 |
| 1-3 | [KV Cache]({{< relref "1-3_KVCache" >}}) | Prefill/Decode에서의 KV cache 역할, 메모리 레이아웃 |
| 1-4 | [FFN]({{< relref "1-4_FFN" >}}) | Feed-Forward Network — Linear, Activation, GEMM 커널 |
| 1-5 | [LayerNorm]({{< relref "1-5_LayerNorm" >}}) | Normalization 수식과 reduction 커널 |
| 1-6 | [Residual & Data Flow]({{< relref "1-6_Residual_and_DataFlow" >}}) | Residual connection, 전체 forward의 커널 launch 순서 |
| 1-7 | [Prefill vs Decode]({{< relref "1-7_PrefillVsDecode" >}}) | 같은 수식, 다른 커널 — compute-bound vs memory-bound |
| 1-8 | [Dense GEMM Call Path]({{< relref "1-8_DenseGEMMCallPath" >}}) | SGLang→cuBLASLt 실호출 경로 검증 (file:line) |
