---
title: "1-3. KV Cache"
weight: 3
---

# 1-3. KV Cache

## 개념 / 동기

Autoregressive generation에서 token $t$를 생성할 때,
이전 token $1 \dots t-1$의 Key/Value는 이미 계산되어 있다.
매번 다시 계산하는 것은 $O(t \cdot d_m^2)$의 불필요한 연산이므로,
**KV Cache**에 저장하고 재사용한다.

## Prefill vs Decode

| Phase | Input | KV Cache 동작 | 연산 특성 |
|-------|-------|---------------|-----------|
| **Prefill** | 전체 prompt $[x_1, \dots, x_S]$ | 모든 K,V를 한 번에 계산하여 cache에 저장 | Compute-bound (large GEMM) |
| **Decode** | 새 토큰 $x_t$ 하나 | $k_t, v_t$ 하나만 계산하여 cache에 append, 전체 cache를 읽어 attention | Memory-bound (KV cache read) |

## 수식 (Decode step)

Token $t$에서:

$$
\mathbf{q}_t = \mathbf{x}_t \mathbf{W}_Q \in \mathbb{R}^{1 \times d_m}
$$
$$
\mathbf{k}_t = \mathbf{x}_t \mathbf{W}_K, \quad \mathbf{v}_t = \mathbf{x}_t \mathbf{W}_V
$$

Cache update: $\mathbf{K}_{1:t} = [\mathbf{K}_{1:t-1}; \mathbf{k}_t]$, $\mathbf{V}_{1:t} = [\mathbf{V}_{1:t-1}; \mathbf{v}_t]$

Attention: $\text{attn}_t = \text{softmax}\!\left(\frac{\mathbf{q}_t \mathbf{K}_{1:t}^\top}{\sqrt{d_h}}\right) \mathbf{V}_{1:t}$

## 메모리 사용량

단일 레이어, 단일 시퀀스:
$$
\text{KV cache size} = 2 \times n_h \times t \times d_h \times \text{sizeof(dtype)}
$$

전체 모델 ($L$ layers, batch $B$):
$$
\text{Total} = 2 \times L \times B \times n_h \times t \times d_h \times \text{sizeof(dtype)}
$$

예: Llama-3-8B ($L=32, n_h=32, d_h=128$, bf16), $B=1, t=4096$:
$$
2 \times 32 \times 1 \times 32 \times 4096 \times 128 \times 2 = 2\text{ GB}
$$

→ KV cache는 batch/sequence가 커질수록 **메모리 병목**이 됨.
이것이 GQA(S2), MLA(S4) 등 variant가 등장하는 직접적 동기.

## GPU 커널 관점

### Cache Append
- Decode시 $k_t, v_t$를 cache에 write: **memory copy / scatter kernel**
- PagedAttention: page table 기반으로 non-contiguous memory에 저장

### Decode Attention
- $\mathbf{q}_t \in \mathbb{R}^{1 \times d_h}$ vs $\mathbf{K} \in \mathbb{R}^{t \times d_h}$
- Batch GEMV에 가까움 → **memory-bandwidth bound**
- FlashInfer: `decode_with_paged_kv_cache()` — KV page들을 streaming read

## Examples

{{< hint info >}}
TODO: `examples/kv_cache_memory_calculator.py` — 모델별 KV cache 크기 계산
{{< /hint >}}
