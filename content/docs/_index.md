---
title: "LLM Inference Study"
weight: 1
bookToc: false
---

# LLM Inference Study

Transformer decoder-block 기반 LLM에서 최신 variants (Dense, MoE, MLA, Parallel MoE)까지,
**inference 연산을 compute/memory 커널 레벨에서 체계적으로 학습**하는 프로젝트.

## Study Roadmap

| Section | Topic | 핵심 주제 |
|---------|-------|-----------|
| **S1** | [Vanilla Transformer Decoder Block]({{< relref "S1_VanillaTransformerDecoderBlock" >}}) | MHA, FFN, LayerNorm, Residual, KV cache, prefill vs decode |
| **S2** | [Dense LLM (Llama-style)]({{< relref "S2_DenseLLM" >}}) | RMSNorm, Pre-LN, RoPE, **GQA**, SwiGLU — Llama recipe |
| **S3** | [MoE (Mixtral-style)]({{< relref "S3_MoE" >}}) | Expert routing, **Grouped GEMM**, token dispatch, load balancing |
| **S4** | [MLA (DeepSeek-V2)]({{< relref "S4_MLA" >}}) | Low-rank KV compression, **matrix absorption**, decoupled RoPE |
| **S5** | [Parallel MoE (DeepSeek-V3)]({{< relref "S5_ParallelMoE" >}}) | Fine-grained MoE (256+1), loss-free balancing, **MTP**, FP8 |

## Baseline Stack

- **Serving platform**: SGLang
- **GPU kernel backend**: FlashInfer 0.6.3 (+ cuBLASLt for dense GEMM, cuDNN, Triton-FA, CUTLASS)
- **Hardware**: NVIDIA A100 80GB PCIe x2 (no NVLink, cross-NUMA SYS connection)

## Methodology

각 variant를 4단계로 분석:

1. **개념 / 동기** — 왜 이 변형이 필요한가
2. **수식** — 연산의 수학적 정의
3. **연산 분해** — 어떤 GEMM/elementwise/reduction이 필요한지
4. **GPU 커널 매핑** — 실제 FlashInfer/cuBLAS/CUTLASS/Triton 함수 호출

각 subsection에 **FlashInfer on A100** 코드 스니펫 + SGLang 호출 경로 + A100 성능 분석 포함.

## Study의 서사 — 한 줄 요약

```
S1 (Vanilla Transformer) — 기본 연산을 커널까지 분해
  └→ S2 (Dense LLM) — inference-friendly하게 세부 연산 최적화
     └→ S3 (MoE) — FFN을 sparse하게 (active weight ↓)
        └→ S4 (MLA) — Attention KV를 low-rank로 (cache ↓)
           └→ S5 (V3)  — 둘 다 적용 + MTP + FP8 (모든 기법 총집합)
```

각 단계에서 공격 지점: **compute/memory bottleneck**.
