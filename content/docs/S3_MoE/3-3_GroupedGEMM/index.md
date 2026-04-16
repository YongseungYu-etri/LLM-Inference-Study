---
title: "3-3. Expert FFN + Grouped GEMM"
weight: 3
---

# 3-3. Expert FFN — Grouped GEMM의 세계

## 개념 / 문제 정의

Router가 각 토큰에 대해 `top-k` expert를 선택했다.
이제 선택된 토큰들을 해당 expert의 FFN에 통과시켜야 한다.

**Naive 구현의 문제**:
```
for expert_id in range(N):
    tokens_for_this_expert = tokens[topk_ids == expert_id]
    output = Expert[expert_id].forward(tokens_for_this_expert)
# N = 8 → 최대 8개 small GEMM kernel launches
# 각 GEMM은 tokens_for_expert_i 크기만큼 → 매우 작고 unbalanced
```

Tensor core 활용률이 낮고 kernel launch overhead 지배.

## 해결책: Grouped GEMM

N개 expert의 FFN GEMM들을 **하나의 kernel launch**로 묶어 실행.
CUTLASS의 **Grouped GEMM** 프리미티브가 이를 제공.

```
입력: 각 expert 그룹별 (M_i, N_i, K_i) 정의된 여러 GEMM 문제
       A_i ∈ R^{M_i × K_i}, B_i ∈ R^{K_i × N_i}
출력: C_i ∈ R^{M_i × N_i} for each i

CUTLASS Grouped GEMM:
  - 각 GEMM을 thread block 단위로 스케줄링
  - 다른 M_i가 있어도 하나의 kernel launch에서 처리
  - Tile 배치를 M_i 크기에 맞춰 동적으로 결정
```

## MoE FFN 전체 파이프라인

```
Input x [T, d_m], topk_ids [T, k], topk_weights [T, k]
(T = num_tokens)

[1] Permute / Dispatch:
    sort tokens by expert_id
    → x_permuted [T*k, d_m]
    → expert_boundaries [N+1]  (prefix sum, 각 expert가 받은 token 수)

[2] Grouped GEMM 1 — Gate+Up:
    For each expert i: x_permuted[boundary_i:boundary_{i+1}] @ W_gate_up[i]
    출력: gate_up [T*k, 2*d_ff]

[3] silu_and_mul:
    mid = silu(gate) * up    → [T*k, d_ff]

[4] Grouped GEMM 2 — Down:
    For each expert i: mid[...] @ W_down[i]
    출력: expert_out [T*k, d_m]

[5] Unpermute + Weighted sum:
    토큰별로 k개 expert output을 topk_weights로 가중합
    → output [T, d_m]
```

## Permute (Dispatch) 커널

```cuda
// pseudo: each token i has topk[i] = (e0, e1), weights = (w0, w1)

// 1. Count tokens per expert
atomicAdd(&expert_counts[e0], 1);
atomicAdd(&expert_counts[e1], 1);

// 2. Prefix sum → expert_offsets

// 3. Scatter tokens to permuted buffer
//    Each token is copied TWICE (once per expert in topk)
for each token i, each expert idx ∈ topk[i]:
    pos = atomicAdd(&expert_offsets[idx], 1)
    permuted[pos] = x[i]
    permuted_weights[pos] = weights[...]
    permuted_src_token[pos] = i     // for unpermute
```

이는 SGLang/vLLM에서 `fused_moe.py`의 `moe_align_block_size` + `moe_sum_reduce` 또는
**triton 기반 커스텀 kernel**로 구현됨.

## Grouped GEMM 상세

### CUTLASS Grouped GEMM Kernel 시그니처

```cpp
// CUTLASS GroupedGemm (간략화)
template <typename Gemm>
struct GroupedGemm {
    struct Arguments {
        cutlass::gemm::GemmCoord* problem_sizes;  // 각 GEMM의 MNK
        ElementA** ptr_A;                          // 각 GEMM의 A pointer
        ElementB** ptr_B;
        ElementC** ptr_C;
        int problem_count;                         // N (expert 수)
    };
    // 실행 시 각 thread block이 어느 expert의 어느 tile을 처리할지 dynamic 스케줄링
};
```

### Mixtral 8x7B의 경우 (T=2048, k=2, N=8)

```
총 활성 토큰: T*k = 4096
평균 per-expert: 4096 / 8 = 512 tokens (balanced 가정)

GEMM group 1 (Gate+Up):
  expert 0: [M_0 ≈ 512, N=28672, K=4096]
  expert 1: [M_1 ≈ 512, N=28672, K=4096]
  ...
  expert 7: [M_7 ≈ 512, N=28672, K=4096]

  → Grouped GEMM 1회 실행 → 8개 GEMM 전부 처리
  총 FLOPs: 8 × 2 × 512 × 28672 × 4096 = 960 GFLOPs
  vs Dense equiv FFN: 2 × 2048 × 28672 × 4096 = 480 GFLOPs
  → Mixtral은 k=2이므로 FLOPs가 dense의 2배
```

**Decode (T=1, k=2)**: 활성 토큰 2개 — per-expert 0 or 1 token → 매우 unbalanced, **GEMV**에 가까움.
이 경우 grouped GEMM 대신 효율적 인 별도 kernel (e.g., Marlin, ExLlamaV2, DeepGEMM) 사용.

## FlashInfer on A100 — MoE 지원 현황

FlashInfer 0.6.3은 **MoE attention은 지원하지 않음** (attention은 dense와 동일).
MoE FFN은 별도 라이브러리가 담당:

### SGLang의 MoE 구현 경로

```python
# sglang.srt.layers.moe.fused_moe.py의 핵심 함수
def fused_experts(
    hidden_states: torch.Tensor,    # [num_tokens, d_model]
    w1: torch.Tensor,               # gate_up weights [N, 2*d_ff, d_model]
    w2: torch.Tensor,               # down weights [N, d_model, d_ff]
    topk_weights: torch.Tensor,     # [num_tokens, topk]
    topk_ids: torch.Tensor,         # [num_tokens, topk]
    inplace: bool = False,
) -> torch.Tensor:
    """
    Fused MoE expert computation.
    """
    # 실제 구현은 backend에 따라 다름:
    # - CUTLASS grouped GEMM (A100/H100)
    # - Triton fused_moe kernel (GPU generic)
    # - DeepGEMM (H100, fp8)
```

### Triton fused_moe Kernel (SGLang 기본, A100 호환)

```python
# sglang.srt.layers.moe.fused_moe_triton 에 있는 kernel 구조
@triton.jit
def fused_moe_kernel(
    A, B, C,                        # input, weights, output pointers
    topk_ids, expert_offsets,       # dispatch info
    M, N, K, num_experts,
    BLOCK_M, BLOCK_N, BLOCK_K,
    GROUP_SIZE_M,                   # how many token blocks per expert
    ...
):
    # 각 pid가 (expert_id, block_m, block_n) 담당
    pid = tl.program_id(0)
    expert_id = ...
    # expert_id의 토큰 범위 내에서 tile 계산
    # A의 token rows는 permuted, B는 expert_id의 weight matrix
    ...
```

### 사용 예시

```python
import torch
from sglang.srt.layers.moe.fused_moe import fused_experts

# Mixtral 8x7B 설정
num_tokens = 512
d_model = 4096
d_ff = 14336
num_experts = 8
topk = 2

hidden = torch.randn(num_tokens, d_model, dtype=torch.bfloat16, device="cuda")

# Expert weights (보통 nn.Parameter로 저장, 여기선 simple)
# w1: gate+up fused per expert [N, 2*d_ff, d_model]
# w2: down per expert [N, d_model, d_ff]
w1 = torch.randn(num_experts, 2 * d_ff, d_model, dtype=torch.bfloat16, device="cuda")
w2 = torch.randn(num_experts, d_model, d_ff, dtype=torch.bfloat16, device="cuda")

# Router 결과 (3-2에서 얻은 것)
topk_weights = torch.rand(num_tokens, topk, dtype=torch.bfloat16, device="cuda")
topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda")

# Fused MoE FFN 실행
output = fused_experts(
    hidden, w1, w2,
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    inplace=False,
)
# output shape: [num_tokens, d_model]

# 내부 CUDA kernels (triton backend):
#   1. align_block_size: token-to-expert alignment
#   2. fused_moe_kernel: Gate+Up grouped GEMM
#   3. silu_and_mul
#   4. fused_moe_kernel: Down grouped GEMM
#   5. sum_reduce: topk weighted sum + unpermute
```

## 중요한 최적화 기법들

### 1. Block alignment (padding)

Grouped GEMM이 효율적으로 돌려면 각 expert의 M을 **block 크기의 배수**로 padding:

```
if tokens_for_expert_i = 37, BLOCK_M = 16:
  padded to 48 (3 blocks)
  11 padding tokens는 mask out

Padding overhead는 보통 10-30%
```

SGLang의 `moe_align_block_size` kernel이 이 역할.

### 2. Permute-free Grouped GEMM

Token을 permute하지 않고, **indirection**으로 각 expert가 자기 토큰을 직접 접근:

```
Standard:    permute → GEMM → unpermute
Indirect:    each thread block reads topk_ids and knows which tokens to process
```

DeepGEMM (H100 전용)이 이 방식을 완전히 활용. A100에서는 permute 기반이 일반적.

### 3. Weight Layout 최적화

```
Option A: expert-first  [N, 2*d_ff, d_model]  ← 각 expert의 weight가 연속
Option B: feature-first [2*d_ff, N, d_model]  ← feature 차원이 먼저

A가 grouped GEMM에 유리 (expert별 access가 연속적).
SGLang/vLLM 표준: Option A.
```

## A100에서의 현실 — Mixtral 8x7B Decode

```
Prefill (T=2048, k=2):
  Gate+Up grouped GEMM: 8 GEMMs, avg M=512 each
    → Tensor core 활용 가능, ~3 ms
  silu_and_mul: ~0.3 ms
  Down grouped GEMM: similar, ~3 ms
  Permute/Unpermute: ~0.2 ms
  Total FFN per layer: ~7 ms
  × 32 layers: 225 ms for prefill FFN

Decode (T=1, k=2):
  활성: 2 tokens × 각 다른 expert
    → 각 GEMM의 M=1 → GEMV 성격
    → Grouped GEMM보다는 batched GEMV kernel이 나을 수도
    → SGLang은 fused_moe_kernel을 사용하지만 throughput 낮음
  Total FFN per layer: ~0.8 ms (8 expert 중 2개 weight 읽기)
  × 32 layers: 25 ms per token
  vs Dense 13B: 더 빠름 (weight 읽기량이 ~1/3)
```

## Examples

{{< hint info >}}
TODO: `examples/grouped_gemm_benchmark.py` — CUTLASS grouped GEMM vs 개별 GEMM
TODO: `examples/moe_permute_cost.py` — permute/unpermute overhead 측정
{{< /hint >}}
