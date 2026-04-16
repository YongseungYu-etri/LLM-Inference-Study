# LLM Inference Study

Transformer decoder-block 기반 LLM에서 최신 variants (Dense, MoE, MLA, Parallel MoE)까지,
**inference 연산을 compute/memory 커널 레벨에서 체계적으로 학습**하는 프로젝트.

## Study Sections

| Section | Topic |
|---------|-------|
| S1 | Vanilla Transformer Decoder Block |
| S2 | Dense LLM (Llama-style: GQA, RoPE, RMSNorm, SwiGLU) |
| S3 | MoE (Mixtral-style: Expert routing, Grouped GEMM) |
| S4 | MLA (DeepSeek-V2: Low-rank KV compression) |
| S5 | Parallel MoE (DeepSeek-V3: Attention ∥ MoE FFN) |

## Baseline Stack

- **Serving**: SGLang
- **Kernel backend**: FlashInfer (cuBLAS, cuDNN, Triton-FA, CUTLASS)
- **Hardware**: NVIDIA A100 80GB PCIe

## Local Development

```bash
hugo server --buildDrafts
```

Site auto-deployed to GitHub Pages on push to `main`.
