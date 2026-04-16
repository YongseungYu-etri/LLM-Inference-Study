---
title: "3-2. Router / Gating"
weight: 2
---

# 3-2. Router / Gating Mechanism

## 개념

Router는 각 토큰이 어느 expert로 갈지 결정하는 작은 네트워크.
본질적으로는 **단순 linear classifier + top-k selection**이다.

```
x (hidden) ──→ [Linear W_g] ──→ logits ──→ Softmax ──→ Top-k ──→ (expert_ids, weights)
```

## 수식

입력 토큰 $\mathbf{x} \in \mathbb{R}^{d_m}$.
Router weight $\mathbf{W}_g \in \mathbb{R}^{d_m \times N}$ ($N$ = expert 수).

### Step 1: Gating logits
$$
\mathbf{s} = \mathbf{x} \mathbf{W}_g \in \mathbb{R}^{N}
$$

### Step 2: Top-k selection
$$
\mathcal{T} = \text{TopK}(\mathbf{s}, k) \subset \{1, \dots, N\}
$$

### Step 3: Weight normalization
Softmax를 **top-k 선택 이후**에 적용 (Mixtral convention):
$$
g_i = \frac{\exp(s_i)}{\sum_{j \in \mathcal{T}} \exp(s_j)}, \quad i \in \mathcal{T}
$$

또는 **softmax 먼저, 그 다음 top-k** (일부 모델).

### Step 4: MoE output
$$
\text{MoE}(\mathbf{x}) = \sum_{i \in \mathcal{T}} g_i \cdot \text{Expert}_i(\mathbf{x})
$$

Mixtral의 경우: $k=2$, $N=8$.

## 직관 — 왜 잘 작동하는가

- 서로 다른 expert가 서로 다른 **input pattern에 특화**
- Training 중 router gradient는 선택된 expert를 통해서만 흐름
  → 자연스럽게 "specialization"이 emerge
- 관찰된 specialization 예:
  - 어떤 expert는 code-heavy token에 active
  - 어떤 expert는 mathematical token에 active
  - 명확한 "topic"이 아닐 수도 있음 (학습된 특성)

## Top-k의 의미

| k | 효과 |
|---|------|
| k=1 | Switch Transformer — 극단적 sparsity, 불안정 |
| **k=2** | **Mixtral/GShard 표준** — 안정성 + sparsity 균형 |
| k=4+ | DeepSeek-MoE (fine-grained 64 experts 중 6-8) |

k가 크면:
- Weight read 증가 → sparsity 이득 감소
- 여러 expert의 조합 → 표현력 증가
- Router decision의 stochasticity 흡수

## 연산 분해

```
Input: x [num_tokens, d_model]

[1] Router GEMM:     logits = x @ W_g       shape [num_tokens, N]
[2] Top-k:           (topk_scores, topk_idx) = topk(logits, k=2)
[3] Softmax (top-k): topk_weights = softmax(topk_scores)
[4] (dispatch)       토큰을 expert별로 재정렬 → Grouped GEMM (3-3에서)
```

### 크기 감각 (Mixtral 8x7B, S=2048)

```
Router GEMM:  [2048, 4096] × [4096, 8] = [2048, 8]
  FLOPs: 2 × 2048 × 4096 × 8 = 134 MFLOPs  ← 매우 작음
  HBM: weight 64 KB, output 32 KB

Top-k:        per token 8개 중 2개 뽑기
  → radix sort 또는 comparator network, ~ns per token

전체 router overhead: ~10-20 μs on A100 (매우 가벼움)
```

## Auxiliary Loss (Training only, inference 무관)

학습 중 expert 사용 빈도를 균등화하기 위한 loss:
$$
\mathcal{L}_{\text{aux}} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

- $f_i$: expert $i$에 할당된 토큰 비율
- $P_i$: expert $i$에 대한 평균 router probability

Inference에는 없음. 다만 이 training trick 덕분에 inference 시 특정 expert에 극단적으로 몰리지는 않음.

## GPU 커널 구현

### Router GEMM — 너무 작아서 overhead 지배

```
M = num_tokens, N = 8, K = d_model = 4096
→ compute: 2*M*N*K = 6.7 MFLOPs per 100 tokens
→ A100에서 ~1 μs 연산, 하지만 kernel launch ~5 μs
→ CUDA graph에 포함 필수
```

### Top-k — 전용 kernel

PyTorch `torch.topk`는 sort 기반이라 N이 작을 때 비효율.
FlashInfer/SGLang은 전용 커널 사용:

```
FlashInfer: sampling.top_k_top_p_sampling_from_logits (for sampling)
SGLang/vLLM MoE: custom moe_topk_softmax kernel
  - warp-level reduction으로 top-k 찾기
  - softmax도 같은 kernel에서 처리
  - ~1-2 μs on A100
```

### Fused Gate + Softmax + Top-k

이론적으로는 한 kernel로 fuse 가능:
```
input: x [num_tokens, d_model], W_g [d_model, N]
output: topk_indices [num_tokens, k], topk_weights [num_tokens, k]

for token in num_tokens:
  1. dot product x[token] · W_g → logits[N]   (N은 작음, register에 유지)
  2. top-k of logits → k pairs
  3. softmax over k values
  4. write output

→ HBM read: x + W_g 1회, write: 가벼움
```

실제로는 router GEMM(cuBLAS)과 top-k+softmax(custom kernel)로 분리된 경우가 많음.

## SGLang/vLLM Router 구현 확인

```python
# vllm/model_executor/layers/fused_moe/fused_moe.py 참조
def fused_topk(hidden_states, gating_output, topk, renormalize):
    """
    gating_output: [num_tokens, num_experts]
    returns: topk_weights [num_tokens, topk], topk_ids [num_tokens, topk]
    """
    # SGLang/vLLM 표준 구현:
    #   1. topk over gating_output
    #   2. (Mixtral) softmax over topk scores only
    #   3. (normalize=True일 때) sum to 1
```

## Router Health Check — Inference 시점에서

Router가 건강한지 확인할 수 있는 지표:

```python
# 한 batch 내 expert 선택 분포
expert_counts = torch.zeros(N)
for token in batch:
    for expert_id in topk_ids[token]:
        expert_counts[expert_id] += 1

# Ideal: k × num_tokens / N per expert
# Metric: Coefficient of Variation
expected = k * num_tokens / N
cv = expert_counts.std() / expected
# cv < 0.2: 건강
# cv > 0.5: imbalance 심각 → 3-4에서 다룸
```

## FlashInfer / SGLang 지원 상태 (0.6.3 기준)

```
FlashInfer 0.6.3에는 MoE 전용 router/dispatch API가 없음
→ SGLang/vLLM의 custom kernel을 사용:
   - sglang.srt.layers.moe.fused_moe.fused_topk()
   - sglang.srt.layers.moe.topk.grouped_topk() (DeepSeek V2/V3용)
```

다음 subsection에서 실제 expert FFN 실행 부분을 다룬다.

## Examples

> [!NOTE]
> TODO: `examples/router_profile.py` — Mixtral router latency 측정
> TODO: `examples/expert_distribution.py` — expert 선택 분포 시각화

