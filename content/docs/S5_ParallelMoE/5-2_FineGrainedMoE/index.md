---
title: "5-2. Fine-grained MoE + Shared Expert"
weight: 2
---

# 5-2. Fine-grained MoE + Shared Expert

## 문제 — Mixtral-style MoE의 한계

Mixtral: 8 experts, k=2
- 한 expert가 받을 수 있는 "specialization 공간"이 넓어짐
- 하지만 각 expert가 "여러 기능을 겸임"하게 되어 redundancy 발생
- 또한 k=2로 2 expert 조합 수 = C(8,2) = 28가지만 표현 가능

## DeepSeek의 진단

"Expert를 잘게 쪼개고 많이 활성화시키면:"
1. 각 expert가 더 좁은 domain에 특화
2. 여러 expert의 조합으로 풍부한 표현
3. **Redundancy 감소 → 같은 active param으로 더 나은 품질**

## Fine-grained 구성

### DeepSeek-V2
- 160 routed experts + 2 shared experts, k=6

### DeepSeek-V3
- 256 routed experts + **1 shared expert**, k=8
- Active experts per token: **1 shared + 8 routed = 9**

### Shared Expert의 역할

```
Shared expert: 모든 토큰이 공통으로 거침
  → 일반 지식, common pattern 학습
Routed experts: 특정 패턴/domain에 특화
  → specialization, 도메인 지식
```

모든 토큰에게 shared의 출력 + top-8 routed의 가중합 → FFN 출력.

수식:
$$
\text{MoEFFN}(\mathbf{x}) = \text{Shared}(\mathbf{x}) + \sum_{i \in \mathcal{T}} g_i \cdot \text{Routed}_i(\mathbf{x})
$$

## 왜 이게 효과적인가 — 직관

Mixtral에서는 "common knowledge"를 모든 expert가 중복 학습:
```
Expert 1: common + specialty_A
Expert 2: common + specialty_B
...
→ common 부분이 8번 중복 (parameter 낭비)
```

Shared expert를 분리하면:
```
Shared: common knowledge
Expert 1..256: pure specialty only
→ parameter 효율 극대화
```

## Routing Math — V3

### Score 계산

$$
s_i = \sigma(\mathbf{x}^\top \mathbf{e}_i) + b_i
$$

- $\mathbf{e}_i$: expert $i$의 **centroid vector** (학습됨)
- $\sigma$: sigmoid (V2는 softmax였으나 V3는 sigmoid)
- $b_i$: **bias** (loss-free balancing, 5-3에서 상세)

### Grouped Top-k

V3는 256 expert를 8 groups × 32 experts로 나누고 (group 당 32 experts),
**먼저 top-4 group을 고른 뒤, 그 안에서 top-8 expert** 선택:

```python
# V3 grouped top-k pseudocode
scores = sigmoid(x @ W_gate)  # [num_tokens, 256]

# Step 1: group scores = top-2 expert scores in each group
group_scores = scores.view(-1, 8, 32).topk(2, dim=-1).values.sum(dim=-1)  # [tokens, 8]

# Step 2: choose top-4 groups
top_groups = group_scores.topk(4, dim=-1).indices

# Step 3: mask non-top-4 group experts
group_mask = zeros_like(scores)
for t in range(num_tokens):
    for g in top_groups[t]:
        group_mask[t, g*32:(g+1)*32] = 1

masked_scores = scores * group_mask

# Step 4: top-8 from masked scores
topk_weights, topk_ids = masked_scores.topk(8, dim=-1)
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
```

왜 grouped?
- **Expert parallelism과 align**: 같은 group = 같은 GPU에 배치 → all-to-all 통신 절감
- 품질 저하 거의 없음

## GPU Kernel 관점 — 256 experts는 다루기 힘든가?

### 문제
- Grouped GEMM의 `problem_count` 파라미터: expert 수 $N$에 비례
- $N=256$: kernel launch overhead, scheduling overhead 증가

### 실제 구현 (SGLang, vLLM)
Triton 기반 `fused_moe_kernel`이 grouped GEMM을 generic하게 처리:
- Program ID를 (expert_idx, token_block, feature_block)로 매핑
- Tile이 자기 expert의 활성 토큰만 처리

256 experts의 경우 실측:
- Launch overhead: 50-100 μs (CUDA graph 시)
- Grouped GEMM 자체 시간은 active tokens 수에 의존

## A100에서 DeepSeek-V2-Lite (fine-grained proxy)

V2-Lite: 16B total, 2.4B active, MLA + 64 routed + 2 shared experts, k=6.

```
Per layer FFN forward (T=2048):
  Shared expert FFN: standard SwiGLU, ~3ms on A100
  Routed experts (64 expert, 6 active per token):
    Active tokens: 2048 × 6 = 12288
    Per expert avg: 12288 / 64 ≈ 192 tokens
    Grouped GEMM 1 (Gate+Up): 64 GEMMs, each M=192, N=4352, K=2048
    → ~2.5 ms
    Grouped GEMM 2 (Down): similar, ~2.5 ms
    silu_and_mul: ~0.5 ms

  Total FFN: ~8.5 ms per layer
```

Decode 시 (per token):
```
Shared expert: M=1 GEMV, weight read 100MB → ~50μs
Routed experts (6 active): M=1 each
  For 6 experts, only 6 × weight_size bytes needed
  → Weight read ~30MB, ~15μs
Total: ~65μs per layer FFN
```

## SGLang 구현 snippet — V3-style fine-grained

```python
# sglang.srt.layers.moe.topk.grouped_topk
from sglang.srt.layers.moe.topk import grouped_topk
from sglang.srt.layers.moe.fused_moe import fused_experts

# Router
router_logits = F.linear(hidden, layer.W_router)  # [T, 256]

topk_weights, topk_ids = grouped_topk(
    hidden_states=hidden,
    gating_output=router_logits,
    topk=8,
    num_expert_group=8,        # 8 groups
    topk_group=4,              # top-4 groups
    renormalize=True,
    scoring_func="sigmoid",    # V3는 sigmoid
    # loss-free balancing bias는 router layer에 이미 포함
)

# MoE FFN execution
routed_out = fused_experts(
    hidden, w1_routed, w2_routed, topk_weights, topk_ids
)

# Shared expert (별도 linear layer)
shared_out = shared_expert_ffn(hidden)  # standard SwiGLU

# Final combine
moe_out = routed_out + shared_out
```

## 요약

| Aspect | Mixtral | DeepSeek-V3 |
|--------|---------|-------------|
| # experts | 8 | 256 + 1 shared |
| top-k | 2 | 8 |
| Specialization | 넓은 영역 겸임 | 세분화 전문 |
| Redundancy | 높음 | 낮음 (shared 분리) |
| Router overhead | 낮음 | 중간 (grouped top-k로 완화) |
| Grouped GEMM complexity | 낮음 | 높음 (problem_count 많음) |
| Quality per active param | baseline | 더 우수 |

## 다음 — Load Balancing

256 expert를 efficient하게 분배하려면 load balancing이 critical.
V3가 aux loss 없이 balance하는 방법은 5-3에서.
