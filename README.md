# LLM Inference Study

Transformer decoder-block 기반 LLM에서 최신 variants (Dense, MoE, MLA, Parallel MoE)까지,
**inference 연산을 compute/memory 커널 레벨에서 체계적으로 학습**하는 프로젝트.

**Site**: https://yongseungyu-etri.github.io/LLM-Inference-Study/

## Study Sections

| # | Topic | 핵심 주제 |
|---|-------|-----------|
| **S1** | Vanilla Transformer Decoder Block | MHA, FFN, LayerNorm, KV cache, prefill vs decode |
| **S2** | Dense LLM (Llama-style) | RMSNorm, RoPE, GQA, SwiGLU |
| **S3** | MoE (Mixtral-style) | Expert routing, Grouped GEMM, load balancing |
| **S4** | MLA (DeepSeek-V2) | Low-rank KV compression, matrix absorption |
| **S5** | Parallel MoE (DeepSeek-V3) | Fine-grained MoE, MTP, FP8 |

## Baseline Stack

- **Serving**: SGLang
- **Kernel backend**: FlashInfer 0.6.3 (cuBLAS, cuDNN, Triton-FA, CUTLASS)
- **Hardware**: NVIDIA A100 80GB PCIe x2 (no NVLink)

## Methodology (per subsection)

1. 개념 / 동기
2. 수식
3. 연산 분해
4. GPU 커널 매핑 + **FlashInfer on A100** 코드 스니펫 + SGLang 호출 경로

## Local Development

```bash
hugo server --buildDrafts
# → http://localhost:1313/LLM-Inference-Study/
```

Site auto-deployed to GitHub Pages on push to `main` (via `.github/workflows/deploy.yml`).

## Repository Structure

```
content/docs/
├── _index.md                          # Home / roadmap
├── S1_VanillaTransformerDecoderBlock/
│   ├── _index.md                      # Section overview
│   ├── 1-1_OverallArchitecture/index.md
│   ├── 1-2_SelfAttention/index.md
│   ├── ... (7 subsections)
├── S2_DenseLLM/                       # 5 subsections
├── S3_MoE/                            # 5 subsections
├── S4_MLA/                            # 5 subsections
└── S5_ParallelMoE/                    # 7 subsections
```
