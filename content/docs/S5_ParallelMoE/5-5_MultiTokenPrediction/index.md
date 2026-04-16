---
title: "5-5. Multi-Token Prediction (MTP)"
weight: 5
---

# 5-5. Multi-Token Prediction — Decode 가속의 또 다른 축

## 문제 — Decode의 Sequential 한계

Autoregressive decode는 근본적으로 sequential:
- Token $t$를 생성하려면 token $t-1$이 필요
- 한 forward = 1 token
- Throughput ≈ 1 / per_step_latency

Per-step latency는 이미 memory-bandwidth가 상한.
**근본적으로 더 빠르려면 "한 번에 여러 token 예측"이 필요.**

기존 해결책들:
- **Speculative decoding**: 작은 draft model이 여러 token 제안 → 큰 model이 verify
- **Medusa, EAGLE**: verification 구조를 model 내부에 내장
- **Multi-Token Prediction** (V3): 학습 시점부터 N개 token 동시 예측

## MTP Core Idea

Training 중 **추가 output head**를 훈련시켜, token $t, t+1, t+2, \dots$를 동시에 예측.

```
Standard LM training:
  Input: x_{1..n}
  Output: predict x_{2..n+1} from each position
  Loss: Σ -log p(x_{t+1} | x_{1..t})

MTP training:
  Input: x_{1..n}
  Output from each position t:
    - Main head: predict x_{t+1}
    - MTP head 1: predict x_{t+2}
    - MTP head 2: predict x_{t+3}
    ...
  Loss: main + λ × Σ(MTP heads)
```

V3는 **1 additional MTP module** (shallow transformer) 추가.
2 token per step 예측 능력.

## 구조

```
Main model:
  [L=61 layers of attention+FFN] → final hidden → [LM head] → token_{t+1}

MTP module:
  [1 extra layer] takes:
    - hidden state from main
    - embedding of predicted token_{t+1}
  → [LM head (shared)] → token_{t+2}
```

Pseudocode:
```python
# Training forward
h = main_model(x)                            # [seq, d_m]
logits_main = lm_head(h)                     # → predict x[t+1]

# MTP part
mtp_input = concat([h, embed(x[t+1])], dim=-1)  # [seq, 2*d_m]
mtp_input = mtp_linear(mtp_input)               # [seq, d_m]
mtp_h = mtp_transformer_layer(mtp_input)        # 1 transformer block
logits_mtp = lm_head(mtp_h)                     # → predict x[t+2]

loss = CE(logits_main, x[t+1:t+1+seq]) + λ * CE(logits_mtp, x[t+2:t+2+seq])
```

## Inference 활용 — Speculative Style

학습된 MTP는 inference에서 **speculative decoding**처럼 활용:

```
Step 1: Main model로 token_{t+1} 예측
Step 2: MTP head로 token_{t+2} 예측 (speculative)
Step 3: 이 두 token을 다음 step input으로 → verify
         - Main model이 position t+2에서 실제 예측한 token과 비교
         - MTP 예측이 맞으면: 2 tokens in 1 step ✓
         - 틀리면: main 예측만 accept, MTP 예측 버림
```

**기대값 (V3 보고):**
- Acceptance rate: ~85%
- 평균 tokens per step: ~1.85
- Decode throughput: **~1.8x** (overhead 고려)

## GPU Kernel 관점 — MTP의 비용과 이득

### 추가 비용 (per step)
- 1 extra transformer block forward
- Main output의 position t+1에 대한 MTP path
- ~ main forward의 1/61 비용 (V3 61 layers)

### 이득
- 성공하면 2 tokens per step
- 총 effective throughput ≈ main_throughput × 1.8 ÷ 1.02 ≈ 1.76x

### 실패 경우
- 1 extra token 위해 1 extra MTP forward
- 실패하면 그 forward는 낭비
- Failure rate × overhead < acceptance rate × gain → 항상 이득

## Kernel Implementation in SGLang (예상)

SGLang 0.5.9에는 MTP/speculative decoding 지원:

```python
# sglang/srt/speculative/ — speculative decoding 모듈
# Medusa, EAGLE, MTP 등 여러 방식 지원

# 대략적 구조:
class DeepSeekV3SpeculativeDecoder:
    def generate_step(self, state):
        # 1. Main model forward → get next token + MTP head hidden
        main_logits, mtp_hidden = self.main_forward(state)
        next_token = sample(main_logits)
        
        # 2. MTP forward → next-next token
        next_next_logits = self.mtp_forward(mtp_hidden, embed(next_token))
        next_next_token = sample(next_next_logits)
        
        # 3. Speculative batch: [next_token, next_next_token]
        # Next step: main model processes both, verify next_next_token
        return [next_token, next_next_token], (verify_state)
    
    def verify_step(self, spec_tokens, state):
        # Verify next_next_token matches main model's prediction at that position
        ...
```

## A100에서 실제 이득

```
DeepSeek-V2-Lite (MLA + MoE, no MTP):
  Decode: ~30 ms/token (single A100 80GB)

Hypothetical V2-Lite + MTP (MTP layer 1개 추가):
  Per-step: ~31 ms (3% overhead)
  Effective: 1.85 tokens/step → 16.8 ms/token
  
→ ~1.8x throughput improvement on A100
```

본 study의 A100 x2 환경에서도 MTP는 그대로 작동 (특별한 통신 요구 없음).

## MTP의 광의적 의미

MTP는 "LLM architecture"의 연장선에 있는 기법이지만,
inference 성능 관점에서는 **새로운 axis**:
- 기존: per-token compute/memory 최적화
- MTP: **per-step에서 몇 token을 얻는가**

이는 speculative decoding family의 evolution:
| 기법 | Draft 방식 | 추가 train 필요 |
|------|-----------|----------------|
| Spec decoding | 별도 작은 model | no |
| Medusa | 병렬 head (독립) | 소량 |
| EAGLE | 별도 작은 model (tree) | 별도 train |
| **MTP** (V3) | **main과 같이 train** | yes (하지만 통합) |

V3는 MTP를 pretrain부터 통합 → 품질 저하 없이 acceptance rate 높음.

## 정리

- MTP: training 시 N-step ahead prediction 추가 훈련
- Inference: 1 step에 2 token 얻음 (speculative verify)
- 약 1.8x decode throughput 이득
- A100/A100x2 환경에서도 동일 이득
- Kernel 관점: 1 extra block forward per step, 실패 시만 낭비

## 다음 — FP8 Inference

5-6에서는 V3가 FP8을 어떻게 활용하는지, 그리고 A100에서는 어떻게 할 수 있는지 (제한적).
