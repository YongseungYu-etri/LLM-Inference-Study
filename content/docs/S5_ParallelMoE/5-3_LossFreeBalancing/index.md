---
title: "5-3. Auxiliary-Loss-Free Load Balancing"
weight: 3
---

# 5-3. Loss-Free Load Balancing

## 문제 — Aux Loss의 부작용

기존 MoE(Switch, GShard, Mixtral)는 auxiliary loss로 load balance 강제:
$$
\mathcal{L} = \mathcal{L}_{\text{main}} + \alpha \cdot \mathcal{L}_{\text{aux}}
$$

부작용:
- 과도한 $\alpha$: main task 품질 저하 (expert 사용이 "억지로" 균형 → routing decision 왜곡)
- 적은 $\alpha$: balance 안 됨 → inference kernel 효율 ↓
- $\alpha$ 튜닝 자체가 복잡

## V3 아이디어 — Bias Adjustment

Loss로 강제하는 대신, **router 출력에 per-expert bias**를 넣고 runtime에 동적 조정:

$$
s_i = \sigma(\mathbf{x}^\top \mathbf{e}_i) + b_i
$$

- $b_i$: expert $i$의 bias. 학습 가능한 parameter 아니고, **training 중 동적 업데이트**.
- $b_i$를 증가시키면 expert $i$가 더 자주 선택됨 (top-k 진입 쉬워짐)

## Update Rule (Training only)

매 step 후, expert $i$의 실제 load를 측정:
```
actual_load_i = (number of tokens routed to expert i) / total_tokens
expected_load = k / N
```

Update:
```python
error = expected_load - actual_load_i
b_i += gamma * sign(error)   # small step, only sign matters
```

- $\gamma$: small step size (e.g., 0.001)
- Under-used expert → bias 증가 → 더 자주 선택
- Over-used expert → bias 감소 → 덜 선택

### 핵심 특징

- $b_i$는 **score에만 영향**, gradient에는 영향 없음
- Routing decision은 bias-adjusted score로, gradient backprop은 raw score로
- **주 task gradient가 왜곡되지 않음**

## Inference 시점

학습이 끝나면 $b_i$는 고정된 상수:
```python
b_i_final = b_values.pt   # checkpoint에 저장됨
```

Inference에서는 그대로 사용:
```
s_i = sigmoid(x @ e_i) + b_i_final
```

즉, inference kernel의 변화는 없음. **Router GEMM output에 bias를 더하는 elementwise 1 step**뿐.

### SGLang 구현

```python
# sglang/srt/layers/moe/topk.py (simplified)
def biased_router(hidden_states, router_weight, router_bias):
    logits = F.linear(hidden_states, router_weight)   # [T, N]
    logits = logits + router_bias                      # elementwise add [N] broadcasting
    scores = torch.sigmoid(logits)
    # ... grouped top-k ...
```

거의 0 overhead. Router GEMM 뒤에 한 줄 더 추가될 뿐.

## Training/Inference의 일관성

일부 기법은 training과 inference의 behavior가 다름 (bias dropping, capacity factor 등).
Loss-free balancing은:
- Training 중 bias가 동적으로 바뀌지만, 매 step 값이 checkpoint에 저장 가능
- Inference는 final bias 값만 사용 → deterministic
- **distribution shift 없음**

## 효과 정량화 (V3 논문)

```
Aux loss MoE (baseline):
  Load imbalance CV: 0.15-0.25
  Expert utilization variance: 큼
  Downstream task loss: baseline

Loss-free balancing (V3):
  Load imbalance CV: 0.05-0.10  (더 작음)
  Expert utilization: 더 균등
  Downstream task loss: ↓ (개선됨)
```

즉, **main task 품질이 더 좋아지고 balance도 더 잘 됨**.

## GPU Kernel 관점

### Runtime overhead
- Router GEMM: 기존과 동일 (M=T, N=256, K=7168)
- Bias add: [T, 256]에 [256] broadcast add — few μs
- Sigmoid: [T, 256] elementwise — few μs
- Grouped top-k: ~5-10 μs

**Total router overhead for V3 (256 experts): ~30-50 μs on A100**.
16.0 ms 정도인 full forward에서 0.2-0.3% 수준.

### 저장 공간
- Bias: 256 × 4B (fp32) = 1 KB per layer
- 58 MoE layers: 58 KB total → 무시 가능

## 관련 다른 기법 비교

| 기법 | 장단점 |
|------|--------|
| Aux Loss (Mixtral) | 단순, $\alpha$ 튜닝 필요, main loss 간섭 |
| **Loss-free Bias** (V3) | 자동 조정, 간섭 없음, 동적 update 필요 |
| Switch Transformer's noise | Randomness로 exploration, balance 효과는 간접 |
| Expert-choice | Balance 자동, 하지만 inference 복잡 |

V3가 결국 택한 loss-free가 **inference kernel을 가장 깔끔하게 유지**하면서 balance를 달성.

## 요약

- Router 출력에 learnable bias $b_i$ 추가
- Training 중 dynamic update (gradient 흐르지 않음)
- Inference 시 고정값 → runtime kernel 변화 없음
- 품질 + balance 모두 개선

## 다음 — Parallel Attn/MoE

5-4에서는 V3의 "parallel" 측면 (shared + routed experts의 병렬 실행)을
GPU kernel 스케줄링 관점에서 본다.
