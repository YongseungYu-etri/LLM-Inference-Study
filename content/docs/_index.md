---
title: "LLM Inference Study"
weight: 1
bookToc: false
---

# LLM Inference Study

Transformer decoder-block 기반 LLM에서 최신 variants(Dense, MoE, MLA, Parallel MoE 등)까지,
**inference 연산을 compute/memory 커널 레벨에서 체계적으로 학습**하는 프로젝트.

## Study Roadmap

| Section | Topic | Focus |
|---------|-------|-------|
| **S1** | [Vanilla Transformer Decoder Block]({{< relref "S1_VanillaTransformerDecoderBlock" >}}) | 기본 구조, 각 연산 블록의 수식 → 커널 매핑 |
| **S2** | Dense LLM (Llama-style) | GQA, RoPE, RMSNorm, SwiGLU — Dense 최적화 기법들 |
| **S3** | MoE (Mixtral-style) | Expert routing, Grouped GEMM, sparse dispatch |
| **S4** | MLA (DeepSeek-V2) | Low-rank KV compression, latent attention |
| **S5** | Parallel MoE (DeepSeek-V3) | Attention ∥ MoE FFN, 커널 동시성 |

## Baseline Stack

- **Serving platform**: SGLang
- **GPU kernel backend**: FlashInfer (cuBLAS, cuDNN, Triton-FA, CUTLASS)
- **Hardware**: NVIDIA A100 80GB PCIe x2

## Methodology

각 variant마다 4단계로 분석:
1. **개념/동기** — 왜 이 변형이 필요한가
2. **수식** — 연산의 수학적 정의
3. **연산 분해** — 어떤 GEMM/elementwise/reduction이 필요한지
4. **GPU 커널 매핑** — 실제 FlashInfer/cuBLAS/CUTLASS/Triton 함수 호출
