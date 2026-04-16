---
title: "3-4. Load Balancing & Capacity"
weight: 4
---

# 3-4. Load Balancing & Capacity in MoE Inference

## 문제 — 왜 Load Balancing이 Inference에서도 중요한가

Training 시 aux loss로 expert load balance를 강제하지만,
**inference runtime**에서도 imbalance는 문제:

```
Batch의 어떤 토큰들이 특정 주제/패턴이면 → 몇몇 expert에 집중
→ Grouped GEMM에서 M_i가 불균일
→ 전체 kernel 실행 시간 = max(M_i) 기준 (straggler problem)
```

### 예시

Mixtral 8x7B, T=2048 tokens, k=2:
```
Balanced (이상적):
  각 expert에 4096/8 = 512 tokens
  Grouped GEMM 시간 ∝ 512

Imbalanced (최악의 경우):
  expert 3에 4096 tokens 전부, 나머지는 0
  Grouped GEMM 시간 ∝ 4096 (8배 느림!)
```

실제로는 그 중간 — 보통 CV(coefficient of variation) 0.2~0.4 정도 발생.

## 해결 전략 1 — Capacity Factor (Training + Inference)

각 expert가 받을 수 있는 토큰 수에 상한 설정:
$$
C = \text{capacity\_factor} \cdot \frac{T \cdot k}{N}
$$

- $C$개 초과한 토큰은 **drop** (attention 통과시키지 않음)
  → "token dropping MoE"
- 또는 second-choice expert로 **reassign**
  → "expert choice" approach

### Inference에서의 의미
- Token dropping을 하면 품질 저하
- Mixtral은 **token drop 없음** (capacity factor = ∞)
- 실제 inference는 imbalance를 그대로 수용하고 grouped GEMM의 straggler 감수

## 해결 전략 2 — Expert Parallelism (Distributed)

여러 GPU에 expert를 분산:
```
EP=4: 8 experts를 4 GPU에 2개씩 배치
  각 토큰 → 자기 위치의 2 expert 중 top-k가 있을 수도, 없을 수도
  → all-to-all communication: 토큰을 해당 expert가 있는 GPU로 전송
```

### A100 x2 (본 환경)에서의 한계

```
NVLink 없는 A100 x2:
  All-to-all을 PCIe로 수행 → 극도로 느림 (16 GB/s)
  Mixtral 8x7B EP=2: per-layer all-to-all 2회 × 32 layers = 64회
  한 step당 통신 overhead가 compute보다 훨씬 큼

→ 결론: A100 x2 (no NVLink)에서는 EP 비효율
→ 단일 GPU에 모든 expert 올리는 것이 현실적 (Mixtral 8x7B: 94GB in bf16,
  A100 80GB 1개로는 OOM → quantization 필요 or TP)
```

## 해결 전략 3 — Drop-and-Pad / Fused Dispatch

Triton/CUTLASS의 MoE kernel들이 취하는 실용적 접근:

### Block Padding
각 expert의 tokens을 `BLOCK_M` (e.g., 64) 배수로 padding:
```
expert 0: 47 tokens → padded to 64  (17 padding)
expert 1: 312 tokens → padded to 320 (8 padding)
...
```
- 모든 expert가 같은 kernel에서 같은 tile size로 처리됨
- Padding은 mask로 무시
- Grouped GEMM의 straggler 문제 완화 (tile boundary로 정규화)

### Fused Permute + GEMM
SGLang/vLLM의 `fused_moe_kernel`은 permute와 GEMM을 하나로:
```
indirect indexing으로 각 tile이 자기 expert의 토큰을 직접 로드
→ 별도 permute buffer 없이 처리
→ 단, unpermute는 여전히 별도 kernel
```

## 해결 전략 4 — Expert-Choice vs Token-Choice (FYI)

| | Token-Choice (Mixtral) | Expert-Choice (Google 2022) |
|---|------------------------|---------------------------|
| 주체 | 토큰이 expert를 고름 | Expert가 토큰을 고름 |
| 결과 | 각 토큰이 정확히 k개 expert | 각 expert가 정확히 C개 token |
| Load balance | Aux loss 필요 | **자동 balanced** |
| Token drop | 없음 (Mixtral) | 있음 (C개만 받음) |

Expert-choice는 inference에서 load balance 자동 보장이 장점이지만, 
autoregressive decode에서 구현이 복잡 (다음 토큰이 어느 expert에 갈지 미리 정하기 어려움).
**Mixtral/DeepSeek은 모두 token-choice.**

## 실측 — Inference Imbalance 측정

```python
import torch
from collections import Counter

# 한 forward step에서 router 결과 분석
def analyze_moe_balance(topk_ids: torch.Tensor, num_experts: int):
    """
    topk_ids: [num_tokens, topk]
    """
    flat = topk_ids.flatten().cpu().numpy()
    counts = Counter(flat.tolist())
    total = len(flat)
    expected = total / num_experts
    
    max_load = max(counts.values())
    min_load = min(counts.values()) if len(counts) == num_experts else 0
    cv = (max_load / expected)  # 1.0이 완벽 균형
    
    return {
        "expected_per_expert": expected,
        "max_load": max_load,
        "min_load": min_load,
        "max_over_expected": max_load / expected,
        "imbalance_cost": max_load / expected,  # 실제 GEMM time multiplier
    }

# Mixtral 8x7B 실측치 (대략)
# - random prompts: max_over_expected ~ 1.2-1.3
# - domain-specific (code, math): 1.4-1.7 
```

## Capacity Factor의 영향 — A100 실측 관점

```
Mixtral은 capacity factor = ∞ (drop 없음)
→ max(expert_load)에 따라 grouped GEMM 시간 결정

A100 decode (T=1, k=2): 활성 2 tokens
  expert 분포: 2 experts가 1 token씩 받거나, 1 expert가 2 tokens
  → M이 어차피 매우 작아서 imbalance 영향 미미

A100 prefill (T=2048):
  Balanced: 512 per expert, Grouped GEMM ~3 ms
  Imbalanced (1.5x): max expert 768, ~4.5 ms (50% slower)
```

## Mixtral 8x7B가 A100에서 잘 도는 이유 요약

1. **k=2 작은 top-k**: 복잡한 dispatching 없음
2. **N=8 작은 expert 수**: grouped GEMM의 problem count 낮음 → overhead ↓
3. **Token-choice**: autoregressive와 잘 맞음
4. **Aux loss로 pretrained balanced**: inference imbalance도 제한적

DeepSeek-V2/V3 스타일 (수십~수백 experts)은 A100에서도 잘 도나,
imbalance 영향이 크고 dispatching 비용이 올라감. **S4, S5**에서 다룸.

## Examples

{{< hint info >}}
TODO: `examples/moe_imbalance_benchmark.py` — imbalance 정도별 latency 곡선
TODO: `examples/capacity_factor_simulation.py` — token drop 시 품질 영향
{{< /hint >}}
