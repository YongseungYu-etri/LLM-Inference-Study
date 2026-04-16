---
title: "5-1. DeepSeek-V3 Overview"
weight: 1
---

# 5-1. DeepSeek-V3 Overview

## 블록 구조 — 한눈에

```
x ─→ RMSNorm ─→ MLA Attention ─→ +residual ─→
  ─→ RMSNorm ─→ Sparse MoE FFN ─→ +residual ─→ x_out
```

"Parallel MoE"라는 section 이름은 오해의 소지가 있음 — V3의 attention과 MoE FFN은
**sequential** (둘 다 병렬이 아님). "Parallel"은 다음 두 의미:
1. MoE 내부의 여러 expert가 **병렬 실행** (그건 모든 MoE가 동일)
2. 일부 구현/논문에서 "parallel transformer block" (attention ∥ FFN) 용어와 혼동 가능

명확히 하면: **V3는 sequential (attention → FFN)**. 
단지 MoE FFN 내부에 shared expert(전체 토큰 공통)와 routed experts(top-k 선택)가 **병렬**로 기여.

## V3 주요 사양

| | 값 |
|---|---|
| Layers (total) | 61 (3 dense + 58 MoE) |
| $d_m$ | 7168 |
| MoE experts | 1 shared + 256 routed |
| Routed top-k | 8 |
| Expert hidden dim | 2048 |
| Attention | MLA (n_h=128, d_c=512, d_c'=1536, d_h^R=64) |
| Parameter (total / active) | 671B / 37B |
| Training dtype | BF16 + FP8 mixed |
| Context | 128K (YaRN extended) |

## Dense Layer + MoE Layer 분리

V3는 **앞 3개 layer는 dense FFN**, 나머지 58개는 MoE:

```
Layer 1-3:    Attn → Dense FFN (SwiGLU, full expert size)
Layer 4-61:   Attn → MoE FFN (1 shared + 256 routed)
```

동기:
- 초반 layer는 low-level feature 추출, expert specialization이 불필요
- Dense가 stability 좋음
- MoE는 해석/의미 추상화 layer에서 효과적

## V2 → V3 핵심 차이

| | V2 | V3 |
|---|---|---|
| Experts | 160 routed + 2 shared, k=6 | 256 routed + 1 shared, k=8 |
| Expert size | larger | smaller (fine-grained) |
| Load balance | aux loss | **loss-free (bias adjust)** |
| MTP | — | **Multi-Token Prediction** |
| Training dtype | BF16 | **FP8 mixed** |
| Context | 128K (YaRN) | 128K |
| Multi-head Latent Attn | yes | yes (동일) |

## "Fine-grained MoE"의 의미

Mixtral: 8 large experts, k=2  
DeepSeek-V3: 256 small experts, k=8

같은 active param count를 더 많은 작은 expert의 조합으로 구성.
**표현력 증가 + 세분화된 specialization 가능**.

단점: router/dispatch overhead 상대적 증가 (N 커질수록).
이를 위한 kernel 최적화가 필수 (3-3, 5-2에서).

## Active Parameter Breakdown (V3, per token)

```
Attention (MLA, per layer): ~200M params
  - Q proj, KV proj (compressed), Output
Dense FFN (layers 1-3): 3 × ~180M = 540M
MoE FFN (layers 4-61, active):
  Shared expert: 58 × ~180M = 10.4B
  Routed experts: 58 × 8 × ~18M = 8.3B  (256개 중 8개 active)
Attention total: 61 × 200M = 12.2B
---
Total active: ~37B
```

## A100 x2 Perspective

V3 Full은 A100 80GB x2에 **안 올라감** (671B BF16 = 1342 GB).
연구 목적으로 관련 기법을 맛보려면:
- **DeepSeek-V2-Lite** (16B, MLA + small MoE): A100 80GB 1개에 올라감
- **DeepSeek-V3 FP8** (671B, 336 GB): 8x H200 이상 필요
- **DeepSeek-V3 AWQ 4bit** (~170 GB): 4x A100 80GB 클래스

본 study의 A100 x2 환경에서는 **V3 기법의 이해**에 집중.
실험은 V2-Lite 또는 (관련 기법만 담은) 작은 모델로 proxy.

## Subsection 가이드

- **5-2 Fine-grained MoE**: 256+1 expert의 의미, routing 구조
- **5-3 Loss-free balancing**: aux loss를 안 쓰고 balance하는 방법
- **5-4 Parallel MoE**: shared expert와 routed expert의 병렬 실행
- **5-5 MTP**: decode 성능을 직접 2x 늘리는 기법
- **5-6 FP8**: DeepGEMM의 세계 (H100/H200 전용 요소 포함)
- **5-7 Full Summary**: 전체 decoder block의 kernel 시퀀스

## 공부 순서 권장

시간 제약이 있다면: 5-2 → 5-4 → 5-7 순서로 먼저.
5-3, 5-5, 5-6는 주제별 심화라 독립적.
