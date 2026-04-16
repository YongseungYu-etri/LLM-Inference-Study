---
title: "3-1. MoE Overview & Motivation"
weight: 1
---

# 3-1. MoE — Why Sparse FFN?

## 동기 — Dense LLM의 한계

S2의 Llama-3-8B, 70B를 분석하면:
- **Parameter 대부분이 FFN**: 8B의 경우 전체 8B 중 ~6B가 FFN weight
- **Decode는 memory-bound**: per-token에 FFN weight 전체를 HBM에서 읽음
- **품질 ∝ parameter count** (일정 범위에서)

따라서 품질을 더 높이려면 parameter를 늘려야 하는데, 이는:
- Training compute $\propto$ params
- Inference memory $\propto$ params
- Inference latency $\propto$ params (decode 기준)

**근본적 질문**: 파라미터를 늘리되, per-token에서는 그중 일부만 쓰면 어떨까?

## MoE의 아이디어

FFN을 $N$개의 **expert**로 복제하고, 각 토큰이 router 판단에 따라 $k$개($k \ll N$)만 activate:

```
Dense FFN:              x ──→ [FFN] ──→ y
Parameters: W

MoE FFN (top-k):        x ──→ [Router] ──→ top-k experts
                         └──→ [Expert_1] ──┐
                         └──→ [Expert_2] ──┤
                         └──→ ...         │ weighted sum ──→ y
                         └──→ [Expert_N] ──┘
Parameters: N × W   (capacity up N배)
Active per token: k × W   (compute down k/N배)
```

Mixtral 8x7B를 예로 들면:
- N=8 experts, k=2 top-k
- Dense 대비 capacity 8배, per-token active compute 2배
- 하지만 **per-token weight read는 2× dense FFN**만 필요 (= dense 대비 $1/4$ if 같은 capacity)

## 용어 정리

| 용어 | 의미 |
|------|------|
| Expert | 각각의 FFN. 전체 N개 |
| Router / Gate | 어느 expert를 activate할지 결정하는 작은 네트워크 |
| Top-k | 각 토큰당 선택하는 expert 수 (Mixtral: k=2) |
| Active parameters | 토큰당 실제 연산에 참여하는 params |
| Sparse / Dense FFN | MoE FFN 내부가 sparse, 각 expert 자체는 dense |
| Fine-grained vs Coarse-grained | DeepSeek은 64+ 작은 expert, Mixtral은 8개 큰 expert |

## Mixtral 8x7B 수치

```
Total parameters:    47B     (not 56B — attention은 공유이므로)
Active per token:    13B     (attention 공유 + 2 experts' FFN)
Dense equivalent:    ~13B 모델의 inference cost
                     ~47B 모델의 capacity

FFN expert 구조 (per expert):
  d_model = 4096
  d_ff = 14336   (SwiGLU)
  gate, up, down:  each 4096 × 14336 = 58.7M params × 3 = 176M per expert
  × 8 experts = 1.4B FFN params per layer

Layers = 32 → 45B FFN params + 2B attention params ≈ 47B total
```

## 왜 Mixtral 이전에 못 했나 — MoE의 난점

MoE는 사실 옛 아이디어 (Shazeer 2017, GShard 2020, Switch 2021).
인프라 + training 기법이 성숙해야 실용적이 됨:

1. **Load imbalance**: 어떤 expert만 많이 쓰이면 나머지는 낭비
   → auxiliary loss로 balance 강제
2. **Sparse dispatch overhead**: 토큰을 expert별로 재정렬하는 비용
   → CUTLASS Grouped GEMM이 성숙 (2023)
3. **Distributed training**: expert를 여러 GPU/node에 샤딩 → all-to-all 통신
   → NVLink/NVSwitch + fast interconnect 필요
4. **Generalization**: 작은 데이터에서 sparse model이 underfit하기 쉬움
   → large-scale pretraining으로 완화

## Inference 관점 — MoE가 좋은 점과 나쁜 점

### 좋은 점 (decode 관점)
- Per-token active weight read가 dense 대비 $k/N$
- Memory-bound decode에서 직접적 속도 향상
- Memory bandwidth를 effective하게 쓸 수 있음

### 나쁜 점
- **Total weight는 모두 GPU에 올라가 있어야 함** (batch마다 다른 expert가 활성화되므로)
  → HBM capacity 요구량은 dense model(여러 배 큰)과 동일
- **Load imbalance**: 같은 batch 내 토큰들이 같은 expert에 몰리면 GEMM shape이 불규칙
- **Router latency**: 작지만 무시 못함 (매 layer 추가)
- **Dispatching overhead**: token permute/unpermute 커널 추가

## Mixtral vs Dense Llama 비교 — A100 decode

```
Llama-3-13B (Dense):
  FFN weight read / token / layer = 6 × 4096 × ~14336 × 2B ≈ 700 MB
  (gate + up + down)
  × 40 layers = 28 GB per token
  A100 HBM 2039 GB/s → 이론 ~14 ms/token (~70 tok/s)

Mixtral 8x7B (MoE):
  FFN weight read / token / layer = 2/8 × 700 MB = 175 MB
  × 32 layers = 5.6 GB per token
  이론 ~2.7 ms/token (~370 tok/s)
  + router, dispatching overhead ~20%
  실제 ~3.3 ms/token (~300 tok/s)

→ Mixtral은 capacity는 더 큰데 decode 속도는 더 빠름.
```

## 다음 subsection Preview

- **3-2 Router**: 어떤 expert를 고르는가 — top-k softmax, auxiliary loss
- **3-3 Grouped GEMM**: 선택된 expert들을 효율적으로 실행하는 방법
- **3-4 Load Balancing**: GPU 상에서 load imbalance 처리
- **3-5 Kernel Summary**: 전체 경로 + FlashInfer/SGLang 구현
