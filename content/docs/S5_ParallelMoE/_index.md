---
title: "S5. Parallel MoE (DeepSeek-V3)"
weight: 5
bookCollapseSection: false
---

# S5. Parallel MoE — DeepSeek-V3

DeepSeek-V3 (2024-12)는 지금까지 학습한 기법들의 **총집합** + 추가 혁신:

| 기법 | 출처 | 이 study의 section |
|------|------|-------------------|
| GQA와 비교되는 attention 압축 | DeepSeek-V2 | S4 (MLA) |
| MoE FFN | Mixtral/DeepSeekMoE | S3 |
| **Shared + Fine-grained Routed experts** | DeepSeek-V2/V3 | **S5** (this) |
| **Parallel attention & MoE** | V3 | **S5** (this) |
| **Auxiliary-loss-free load balancing** | V3 | **S5** (this) |
| **Multi-Token Prediction** | V3 | **S5** (this) |
| **FP8 training & inference** | V3 | **S5** (this) |

## Scale

```
DeepSeek-V3: 
  Total params: 671B
  Active per token: 37B
  Training: 2.788M H800 GPU hours (~$6M cost)
  Context: 128K
  Inference (API): 20 tokens/s, $2/M input
```

**Mixtral 8x7B 대비 15배 더 큰 capacity, 약 3배 active params.**
동급 품질의 Dense model(GPT-4급)과 비교하면 훨씬 효율적 inference.

## 이 section이 다루는 것

V3의 여러 혁신 중 **architecture & inference kernel** 관점에 집중.
Training details (FP8, MoE balance, distillation)은 간략만.

## Subsections

| # | Topic | Description |
|---|-------|-------------|
| 5-1 | [DeepSeek-V3 Overview]({{< relref "5-1_Overview" >}}) | V3 전체 구조, V2와의 차이 |
| 5-2 | [Fine-grained MoE + Shared Expert]({{< relref "5-2_FineGrainedMoE" >}}) | 256+1 experts, routing 전략 |
| 5-3 | [Auxiliary-Loss-Free Balancing]({{< relref "5-3_LossFreeBalancing" >}}) | Bias-based load balancing 기법 |
| 5-4 | [Parallel Attention / MoE]({{< relref "5-4_ParallelMoE" >}}) | 의미와 실제 구현, GPU kernel scheduling |
| 5-5 | [Multi-Token Prediction]({{< relref "5-5_MultiTokenPrediction" >}}) | MTP로 decode 병목 완화 |
| 5-6 | [FP8 Inference & DeepGEMM]({{< relref "5-6_FP8" >}}) | FP8 경로, Hopper 전용이지만 A100 관점도 |
| 5-7 | [Full Kernel Summary]({{< relref "5-7_FullSummary" >}}) | V3 decoder block 전체 커널 시퀀스 |
