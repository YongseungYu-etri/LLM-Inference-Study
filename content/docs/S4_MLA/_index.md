---
title: "S4. MLA (DeepSeek-V2)"
weight: 4
bookCollapseSection: false
---

# S4. Multi-Head Latent Attention (MLA)

DeepSeek-V2가 도입한 attention 변형.
**KV cache를 low-rank latent representation으로 압축**하여 메모리를 극단적으로 절감.

GQA(S2)와 목적은 같지만 접근이 다름:
- **GQA**: KV head 수 자체를 줄임
- **MLA**: KV를 작은 latent vector로 압축, per-head KV는 "on-the-fly" 복원

## 왜 MLA가 필요한가

```
Llama-3-8B GQA: per-token KV cache = 2 × 8 × 128 × 2B = 4 KB
MQA 극단: per-token KV cache = 2 × 1 × 128 × 2B = 0.5 KB (품질 저하)

MLA (DeepSeek-V2): per-token KV cache = ~576 bytes (4x smaller than GQA)
                   동시에 품질은 MHA 수준 유지
```

MLA는 GQA보다 더 공격적 압축 + 품질 보존을 **동시에** 달성.
그 비결은 **학습 가능한 down/up projection + 수학적 구조 활용**.

## Subsections

| # | Topic | Description |
|---|-------|-------------|
| 4-1 | [MLA Motivation & Concept]({{< relref "4-1_Motivation" >}}) | 왜 low-rank compression인가, DeepSeek-V2 수치 |
| 4-2 | [MLA Math]({{< relref "4-2_Math" >}}) | Down/up projection 수식, compressed latent |
| 4-3 | [Matrix Absorption Trick]({{< relref "4-3_Absorption" >}}) | Q/K,V projection을 합쳐 compressed KV에 직접 attention |
| 4-4 | [Decoupled RoPE]({{< relref "4-4_DecoupledRoPE" >}}) | RoPE가 low-rank와 충돌하는 문제 + 해결 |
| 4-5 | [FlashInfer MLA on A100]({{< relref "4-5_FlashInferMLA" >}}) | MLA 전용 API, kernel 구조, SGLang 경로 |
