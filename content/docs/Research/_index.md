---
title: "Research Notes"
weight: 10
bookCollapseSection: false
---

# Research Notes

Study 과정에서 도출된 고찰, 논의, 연구 방향에 대한 기록.

S1~S5의 학습 내용을 기반으로, LLM inference 최적화의 현재 landscape와
향후 연구 기회를 정리한다.

## Articles

| # | Topic | 핵심 주제 |
|---|-------|-----------|
| R-1 | [GPU Kernel 최적화의 현재와 미래]({{< relref "R1_KernelOptimizationLandscape" >}}) | Regular → Irregular workload 전환, kernel 전문성의 가치 변화 |
| R-2 | [Decoder Block 수준 Profiling 설계]({{< relref "R2_BlockLevelProfilingDesign" >}}) | Inter-kernel dead time, KV cache L2 locality 정량화 실험 설계 |
