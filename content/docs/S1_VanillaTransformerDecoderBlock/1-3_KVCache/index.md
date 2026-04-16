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

## FlashInfer on A100 — Paged KV Cache 관리

### KV Cache 메모리 레이아웃

```python
# FlashInfer Paged KV Cache 구조 (NHD layout)
#
# 전체 GPU 메모리에서 page pool을 미리 할당:
#   k_cache: [max_num_pages, page_size, num_kv_heads, head_dim]
#   v_cache: [max_num_pages, page_size, num_kv_heads, head_dim]
#
# 예: Llama-3-8B on A100 80GB
#   num_kv_heads=32 (MHA), head_dim=128, page_size=16, dtype=bf16
#   page당 메모리: 16 * 32 * 128 * 2bytes = 128KB (K만), K+V = 256KB
#   max_num_pages=10000 → KV pool = 2.56 GB

import torch

max_num_pages = 10000
page_size = 16
num_kv_heads = 32
head_dim = 128

k_cache = torch.zeros(max_num_pages, page_size, num_kv_heads, head_dim,
                       dtype=torch.bfloat16, device="cuda")
v_cache = torch.zeros_like(k_cache)
```

### Page Table 구조 — indptr/indices/last_page_len

```python
# 3개 요청이 동시 serving 중:
#   req0: seq_len=50  → ceil(50/16) = 4 pages, last_page_len = 50 - 3*16 = 2
#   req1: seq_len=200 → ceil(200/16) = 13 pages, last_page_len = 200 - 12*16 = 8
#   req2: seq_len=33  → ceil(33/16) = 3 pages, last_page_len = 33 - 2*16 = 1

kv_indptr = torch.tensor([0, 4, 17, 20], dtype=torch.int32, device="cuda")
#           req0: pages[0:4], req1: pages[4:17], req2: pages[17:20]

kv_indices = torch.tensor([
    42, 7, 91, 3,                          # req0의 4 pages (비연속!)
    10, 11, 55, 56, 57, 100, 101, 102,     # req1의 13 pages
    103, 200, 201, 202, 203,
    80, 81, 82,                            # req2의 3 pages
], dtype=torch.int32, device="cuda")

kv_last_page_len = torch.tensor([2, 8, 1], dtype=torch.int32, device="cuda")

# 핵심: kv_indices가 비연속적 → "Paged" 장점
#   - 요청 간 메모리 단편화 없이 page 단위로 할당/해제
#   - Prefix caching: 공통 prefix의 page를 여러 요청이 공유 가능
```

### KV Cache Append — 새 토큰의 K,V 저장

```python
from flashinfer.page import append_paged_kv_cache, get_batch_indices_positions

# Decode step: 3개 요청에 각각 1 token씩 생성
append_indptr = torch.tensor([0, 1, 1, 1], dtype=torch.int32, device="cuda")  # 각 1 token
seq_lens = torch.tensor([51, 201, 34], dtype=torch.int32, device="cuda")       # 기존+1

batch_indices, positions = get_batch_indices_positions(
    append_indptr, seq_lens, nnz=3
)
# batch_indices: [0, 1, 2]  — 어떤 요청의 토큰인지
# positions: [50, 200, 33]  — 시퀀스 내 position

new_k = torch.randn(3, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
new_v = torch.randn(3, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")

append_paged_kv_cache(
    append_key=new_k,
    append_value=new_v,
    batch_indices=batch_indices,
    positions=positions,
    paged_kv_cache=(k_cache, v_cache),
    kv_indices=kv_indices,
    kv_indptr=kv_indptr,
    kv_last_page_len=kv_last_page_len,
    kv_layout="NHD",
)
# 내부 CUDA kernel: flashinfer::append_paged_kv_cache
#   - scatter write: position에 따라 올바른 page의 올바른 slot에 k,v 기록
#   - 매우 가벼운 memory-bound kernel (토큰 수 × head × dim만큼 write)
```

### SGLang에서의 KV Cache 관리

```
SGLang TokenToKVPool
  ├── k_buffer: [max_num_pages, page_size, num_kv_heads, head_dim]  per layer
  ├── v_buffer: [max_num_pages, page_size, num_kv_heads, head_dim]  per layer
  ├── set_kv_buffer(layer_id, cache_loc, k, v)   → scatter write
  └── get_kv_buffer(layer_id) → (k_buffer, v_buffer)  → FlashInfer에 전달

Page 할당/해제는 RadixCache가 관리:
  - 새 요청 → free pages에서 할당
  - 요청 완료 → pages 반환 (또는 prefix cache로 유지)
  - LRU eviction으로 메모리 pressure 관리
```

## Examples

> [!NOTE]
> TODO: `examples/kv_cache_memory_calculator.py` — 모델별 KV cache 크기 계산

