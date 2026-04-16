---
title: "S3. MoE (Mixtral-style)"
weight: 3
bookCollapseSection: false
---

# S3. Mixture of Experts (Mixtral-style)

Dense LLM (S2)의 **FFN을 sparse하게** 만들어,
- 모델 capacity(parameter count)는 유지하거나 늘리면서
- per-token inference compute/memory는 줄이는 기법.

대표 모델:
- **Mixtral 8x7B / 8x22B** (Mistral AI)
- **DeepSeek-MoE** (DeepSeek)
- **Qwen2-57B-MoE**
- **Grok-1**

본 section에서는 Mixtral-style (사전 연구인 Switch/GShard의 후예)을 중심으로 다룬다.
DeepSeek/V3-style (많은 작은 expert + shared expert)은 **S4, S5**에서.

## 핵심 아이디어

```
Dense FFN:    모든 토큰이 모든 parameter를 읽음
              → 토큰당 weight read = full FFN weight

MoE FFN:      N개 expert 중 top-k만 activate
              → 토큰당 weight read = k/N × full FFN weight  (예: 2/8 = 25%)
              → 대신 router + dispatching overhead 추가
```

## Subsections

| # | Topic | Description |
|---|-------|-------------|
| 3-1 | [MoE Overview & Motivation]({{< relref "3-1_Overview" >}}) | Dense vs MoE, 왜 MoE가 등장했나 |
| 3-2 | [Router / Gating]({{< relref "3-2_Router" >}}) | Top-k gating, softmax, noise/aux loss |
| 3-3 | [Expert FFN + Grouped GEMM]({{< relref "3-3_GroupedGEMM" >}}) | Token-to-expert dispatch, CUTLASS grouped GEMM |
| 3-4 | [Load Balancing & Capacity]({{< relref "3-4_LoadBalancing" >}}) | Expert load imbalance, capacity factor |
| 3-5 | [Kernel-Level Summary]({{< relref "3-5_KernelSummary" >}}) | Mixtral 8x7B on A100 — 전체 경로 정리 |
