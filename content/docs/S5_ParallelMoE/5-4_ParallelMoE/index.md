---
title: "5-4. Parallel MoE — Shared ∥ Routed Experts"
weight: 4
---

# 5-4. Parallel MoE — 무엇이 "Parallel"인가

## 용어 정리

본 study에서 "**Parallel MoE**"의 의미:

DeepSeek-V3의 **FFN block 내에서**:
- Shared expert와 routed experts가 **병렬로 실행 가능**
- 최종 output은 둘의 합

이는 attention과 FFN의 병렬(다른 아키텍처인 PaLM/Parallel Transformer)과는 다른 개념.
V3는 attention → FFN 순차 구조 유지.

```
hidden → attention → residual → norm ─┬─→ Shared Expert ──→
                                       │                    + → output
                                       └─→ Router ──→ Routed FFN ──→
```

## 왜 "병렬"이 GPU 관점에서 중요한가

Shared expert와 routed FFN은 **서로 다른 kernel**로 실행:
- Shared: 큰 SwiGLU (M = num_tokens)
- Routed: fused_moe_kernel (grouped GEMM)

두 kernel을 **별도 CUDA stream에서 동시 실행**하면 GPU resource 경쟁 없이 활용 가능.

### 이론적 이득

A100 SM 개수: 108. Shared FFN이 SM 50개로 충분하면, 나머지 58개는 유휴.
두 kernel을 병렬 실행하면 유휴 SM 활용 → throughput 증가.

### 현실

- Shared expert 자체가 이미 full wave로 GPU 점유 → 동시 실행 이득 제한적
- 단, decode시 (M=1) shared FFN도 작아서 이득 존재할 수 있음
- CUDA stream 간 sync overhead 고려 필요

## 수식

$$
\text{FFN}^{V3}(\mathbf{x}) = \underbrace{\text{SharedFFN}(\mathbf{x})}_{\text{모든 토큰 공통}} + \underbrace{\sum_{i=1}^{K} g_i \cdot \text{RoutedFFN}_i(\mathbf{x})}_{\text{선택된 K개 expert}}
$$

Shared와 Routed는 **완전히 독립적**. 순서 무관, 병렬화 가능.

## 파라미터 비율

```
V3 per MoE layer:
  Shared expert:   1 × SwiGLU(d_m=7168, d_ff=2048 × n_shared)
                   Actually V3 shared expert has d_inter = 2048 * 1 (one shared)
                   Params per shared: 3 × 7168 × 2048 ≈ 44M
  
  Routed experts:  256 × SwiGLU(d_m=7168, d_ff=2048)
                   Params per expert: 3 × 7168 × 2048 ≈ 44M
                   Total: 256 × 44M ≈ 11.3B

  Per layer MoE: ~11.35B
  58 layers: ~658B    ← 이것이 V3의 671B 중 대부분
```

Shared 1개 + Routed 256개 중 top-8 = active 9개.
Per token FFN active = 9 × 44M ≈ 400M.

## 구현 — 두 가지 스케줄링 전략

### Strategy 1: Sequential (단순)

```python
def moe_ffn_v3(hidden, layer):
    shared_out = layer.shared_expert(hidden)          # kernel 1
    
    topk_weights, topk_ids = grouped_topk(hidden, layer.router_weight, ...)
    routed_out = fused_experts(hidden, layer.w1, layer.w2, topk_weights, topk_ids)  # kernel 2
    
    return shared_out + routed_out                     # kernel 3 (elementwise add)
```

장점: 단순, 디버깅 쉬움  
단점: GPU utilization 낮을 수 있음

### Strategy 2: Concurrent (병렬)

```python
def moe_ffn_v3_parallel(hidden, layer):
    # 두 stream 생성
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    # 병렬 실행
    with torch.cuda.stream(stream1):
        shared_out = layer.shared_expert(hidden)
    
    with torch.cuda.stream(stream2):
        topk_weights, topk_ids = grouped_topk(hidden, layer.router_weight, ...)
        routed_out = fused_experts(hidden, layer.w1, layer.w2, topk_weights, topk_ids)
    
    # Sync
    torch.cuda.synchronize()  # 또는 event-based
    
    return shared_out + routed_out
```

단점: 
- Sync overhead (~10-50 μs)
- CUDA Graph capture가 복잡해짐
- Stream 경합으로 예상한 이득이 안 나올 수 있음

실제 SGLang/vLLM은 Strategy 1 (sequential)을 기본으로 사용 — 이유는 안정성과 CUDA graph 친화성.

## CUDA Graph 관점

CUDA graph는 stream 간 병렬을 capture하기 어려움.
Single-stream sequential 실행이 더 graph-friendly.
V3 serving 시 CUDA graph 활용이 중요하므로 sequential이 일반적.

## A100 실측 감각 — V2-Lite로 추론

V2-Lite (64 routed + 2 shared, k=6):
```
Decode per layer FFN (M=1):
  Shared (2 experts 합산): 2 × ~30 μs = 60 μs
  Routed (6 active): fused_moe_kernel ~50 μs
  Sequential total: ~110 μs
  Concurrent (이론): max(60, 50) = 60 μs + overhead 20 μs = ~80 μs
  
→ Concurrent 이득 ~25%, 하지만 실제로는 overhead로 줄어듦
```

V3 (1 shared + 256 routed, k=8):
```
Decode per layer FFN:
  Shared (1 expert): ~50 μs
  Routed (8 active): fused_moe ~80 μs
  Sequential: ~130 μs
  Concurrent: 80 μs + overhead = ~100 μs (~23% 이득 예상)
```

현재 공개 구현은 sequential을 선호. 진짜 parallelism은 향후 구현 여지.

## SGLang 구현 (실제)

```python
# sglang/srt/models/deepseek_v2.py — V3/V2 공통 MoE layer
class DeepseekV2MoE(nn.Module):
    def __init__(self, config):
        ...
        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            ...
        )
        if config.n_shared_experts:
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
                # Shared expert는 SwiGLU (= DeepseekV2MLP)
                hidden_act=config.hidden_act,
            )
    
    def forward(self, hidden_states):
        identity = hidden_states
        
        # Router
        router_logits = self.gate(hidden_states)
        
        # Routed experts (FusedMoE 내부 fused_experts 호출)
        routed_output = self.experts(hidden_states, router_logits)
        
        # Shared expert
        if hasattr(self, 'shared_experts'):
            routed_output = routed_output + self.shared_experts(identity)
        
        return routed_output
```

순차 실행. 향후 multi-stream 최적화 여지 있음.

## Expert Parallelism과의 관계

V3 inference를 distributed(EP=8)로 돌릴 때:
- 256 routed experts를 8 GPU에 32개씩 배치
- 매 token → 해당 expert가 있는 GPU로 **all-to-all**
- Shared expert는 **모든 GPU에 복제** (duplicate) → shared는 local 계산

```
GPU 0: [routed 0-31]   + [shared expert copy]
GPU 1: [routed 32-63]  + [shared expert copy]
...
GPU 7: [routed 224-255] + [shared expert copy]

All-to-all per layer:
  Pre-FFN: 토큰을 해당 expert GPU로
  Post-FFN: 결과를 원래 토큰 위치로
```

**A100 x2 (NVLink 없음)**에서는 all-to-all이 PCIe로 → 매우 비효율.
V3 스타일은 NVLink 필수 또는 H200 node (NVSwitch) 기준 설계.

## 요약

- V3 FFN = Shared FFN + Routed FFN (수학적 합)
- 이 둘은 이론적으로 병렬 실행 가능
- 실제 구현은 대부분 sequential (안정성 + CUDA graph)
- Parallel 실행은 future optimization 여지
- "Parallel MoE"는 **구조가 병렬화 친화적**임을 의미

## 다음 — MTP

5-5에서는 V3의 또 다른 중요한 혁신 — **Multi-Token Prediction** —
을 살펴본다. Attention 스타일이 바뀌진 않지만, decode 효율을 직접 2x 늘릴 수 있다.
