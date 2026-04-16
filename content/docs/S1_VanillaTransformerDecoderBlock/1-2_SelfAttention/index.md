---
title: "1-2. Self-Attention"
weight: 2
---

# 1-2. Self-Attention Mechanism

## 개념 / 동기

Self-Attention은 시퀀스 내 모든 위치 간의 관계를 학습하는 메커니즘이다.
RNN과 달리 병렬 처리가 가능하며, 장거리 의존성을 직접 모델링할 수 있다.

## 수식

### Multi-Head Attention (MHA)

입력 $\mathbf{X} \in \mathbb{R}^{S \times d_{\text{model}}}$ (시퀀스 길이 $S$)에 대해:

**Step 1: QKV Projection**

$$
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V
$$

여기서 $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$

**Step 2: Per-Head Split**

$\mathbf{Q}, \mathbf{K}, \mathbf{V}$를 $n_h$개 head로 분할:
$\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i \in \mathbb{R}^{S \times d_h}$, where $d_h = d_{\text{model}} / n_h$

**Step 3: Scaled Dot-Product Attention (per head)**

$$
\text{Attn}_i = \text{softmax}\!\left(\frac{\mathbf{Q}_i \mathbf{K}_i^\top}{\sqrt{d_h}}\right) \mathbf{V}_i
$$

**Step 4: Concat + Output Projection**

$$
\text{MHA}(\mathbf{X}) = \text{Concat}(\text{Attn}_1, \dots, \text{Attn}_{n_h}) \mathbf{W}_O
$$

## 연산 분해

| Step | 연산 | Shape | FLOPs (per head) |
|------|------|-------|-------------------|
| QKV Proj | GEMM | $(S, d_m) \times (d_m, d_m)$ | $3 \times 2 S d_m^2$ |
| $\mathbf{Q}\mathbf{K}^\top$ | GEMM (batched) | $(S, d_h) \times (d_h, S)$ → $(S, S)$ | $2 S^2 d_h$ |
| Softmax | Reduction + Exp | $(S, S)$ | $O(S^2)$ |
| Score $\times$ V | GEMM (batched) | $(S, S) \times (S, d_h)$ → $(S, d_h)$ | $2 S^2 d_h$ |
| Output Proj | GEMM | $(S, d_m) \times (d_m, d_m)$ | $2 S d_m^2$ |

**총 FLOPs** (all heads): $\approx 8 S d_m^2 + 4 S^2 d_m$

- $S \ll d_m$ (짧은 시퀀스): **GEMM 지배** (QKV/Output projection)
- $S \gg d_m$ (긴 시퀀스): **Attention score 지배** ($O(S^2)$ 항)

## GPU 커널 매핑

### QKV Projection & Output Projection → cuBLAS GEMM

```
cublasGemmEx() 또는 cublasLtMatmul()
  - A: activation [S, d_model], B: weight [d_model, d_model]
  - 보통 QKV를 하나로 fuse: W_QKV ∈ R^{d_model × 3*d_model} → single GEMM
  - FlashInfer/SGLang에서는 cuBLAS 또는 CUTLASS GEMM 호출
```

### Attention Core → FlashAttention / FlashInfer Fused Kernel

```
Q @ K^T → softmax → @ V 를 하나의 fused kernel로 실행
  - FlashInfer: flashinfer.prefill_with_paged_kv_cache() 등
  - O(S) memory (S×S score matrix를 materialization하지 않음)
  - Tiling: Q를 block 단위로 처리, online softmax
  - GPU에서: shared memory에 Q tile 로드 → K,V tile streaming → accumulate
```

### 왜 Fused Attention이 필요한가

Naive 구현시:
1. `Q @ K^T` → $(S, S)$ matrix를 HBM에 write → **$O(S^2)$ memory**
2. Softmax → HBM read/write
3. `Score @ V` → HBM read

FlashAttention: tiling + online softmax로 중간 결과를 SRAM(shared memory)에 유지
→ HBM access $O(S^2 d_h / M)$ (M = SRAM size), memory 사용 $O(S)$

## Examples

{{< hint info >}}
TODO: `examples/attention_flops_calculator.py` — shape별 FLOPs 계산
TODO: `examples/naive_vs_flash_attention.py` — memory/compute 비교
{{< /hint >}}
