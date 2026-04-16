---
title: "3-5. Kernel-Level Summary"
weight: 5
---

# 3-5. Kernel-Level Summary — Mixtral 8x7B on A100

## 전체 Decoder Block 커널 시퀀스

Mixtral 8x7B는 Llama-style architecture + MoE FFN.
Attention은 S2 Llama와 동일. FFN만 MoE로 교체.

```
[1]  fused_add_rmsnorm         ← S2와 동일
[2]  qkv_gemm                  ← S2와 동일 (GQA, n_kv=8)
[3]  apply_rope                ← S2와 동일
[4]  kv_cache_append           ← S2와 동일
[5]  prefill_attention         ← S2와 동일 (flashinfer)
[6]  output_proj_gemm          ← S2와 동일
[7]  fused_add_rmsnorm         ← S2와 동일

--- 여기서부터 MoE FFN ---
[8]  router_gemm               ← 새 커널: x @ W_g → [T, N=8]
[9]  topk_softmax              ← 새 커널: top-2 + weight normalize
[10] moe_align_block_size      ← 새 커널: dispatch 준비 (expert별 token count)
[11] fused_moe_kernel (Gate+Up) ← Grouped GEMM 1 (Triton or CUTLASS)
[12] silu_and_mul              ← S2와 동일 (각 expert의 중간값에 적용)
[13] fused_moe_kernel (Down)   ← Grouped GEMM 2 + weighted sum unpermute
```

## 전체 파이프라인 코드 — SGLang + FlashInfer + fused_moe

```python
import torch
import torch.nn.functional as F
import flashinfer
from sglang.srt.layers.moe.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import select_experts

# ── 서버 시작 ──
workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
decode_wrapper  = flashinfer.BatchDecodeWithPagedKVCacheWrapper (workspace, kv_layout="NHD")

# Mixtral 8x7B 파라미터
d_model = 4096
d_ff = 14336
n_q_heads, n_kv_heads, head_dim = 32, 8, 128
num_experts = 8
topk = 2

# ── Forward (per layer) ──
for layer in model.layers:
    # [1-7] S2 Llama와 동일한 attention path
    hidden, residual = flashinfer.norm.fused_add_rmsnorm(
        hidden, residual, layer.attn_norm_weight, eps=1e-5
    )

    qkv = F.linear(hidden, layer.W_qkv)
    q, k, v = qkv.split([32*128, 8*128, 8*128], dim=-1)
    q = q.view(-1, 32, 128); k = k.view(-1, 8, 128); v = v.view(-1, 8, 128)

    flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
        positions, q, k, head_size=128, cos_sin_cache=rope_cache, is_neox=True,
    )

    kv_pool.set_kv_buffer(layer.idx, cache_loc, k, v)
    attn_out = prefill_wrapper.run(q, kv_pool.get_kv_buffer(layer.idx))
    attn_out = F.linear(attn_out.view(-1, d_model), layer.W_o)

    hidden, residual = flashinfer.norm.fused_add_rmsnorm(
        attn_out, residual, layer.ffn_norm_weight, eps=1e-5
    )

    # ── [8-9] Router ──
    router_logits = F.linear(hidden, layer.W_router)   # [T, num_experts=8]
    #   cuBLAS: very small GEMM (M=T, N=8, K=4096)

    topk_weights, topk_ids = select_experts(
        hidden_states=hidden,
        router_logits=router_logits,
        top_k=topk,
        use_grouped_topk=False,       # Mixtral: simple top-k
        renormalize=True,             # softmax normalize
    )
    # topk_weights: [T, topk], topk_ids: [T, topk]

    # ── [10-13] MoE Expert FFN ──
    # w1: [num_experts, 2*d_ff, d_model] — gate+up fused per expert
    # w2: [num_experts, d_model, d_ff]   — down per expert
    ffn_out = fused_experts(
        hidden_states=hidden,
        w1=layer.moe_w1,
        w2=layer.moe_w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
    )
    #   내부:
    #   [10] moe_align_block_size (SGLang kernel)
    #   [11] fused_moe_kernel (Triton grouped GEMM: Gate+Up)
    #   [12] silu_and_mul (Triton or flashinfer)
    #   [13] fused_moe_kernel (Triton grouped GEMM: Down) + weighted sum

    hidden = ffn_out   # residual은 다음 layer [1]에서 합산
```

## 커널별 역할과 비용 — Prefill (T=2048, A100)

| # | 커널 | Library | Count | Per call | Per layer | × 32 |
|---|------|---------|:-----:|---------:|----------:|-----:|
| 1,7 | fused_add_rmsnorm | flashinfer | 2 | 20μs | 40μs | 1.3ms |
| 2 | qkv_gemm | cuBLAS | 1 | 230μs | 230μs | 7.4ms |
| 3 | apply_rope | flashinfer | 1 | 30μs | 30μs | 1.0ms |
| 4 | kv_cache_append | flashinfer | 1 | 10μs | 10μs | 0.3ms |
| 5 | prefill_attn (GQA) | flashinfer | 1 | 420μs | 420μs | 13.4ms |
| 6 | output_gemm | cuBLAS | 1 | 240μs | 240μs | 7.7ms |
| 8 | router_gemm | cuBLAS | 1 | 15μs | 15μs | 0.5ms |
| 9 | topk_softmax | SGLang | 1 | 5μs | 5μs | 0.16ms |
| 10 | moe_align | SGLang | 1 | 10μs | 10μs | 0.3ms |
| 11 | fused_moe (Gate+Up) | Triton | 1 | 2.8ms | 2.8ms | **89.6ms** |
| 12 | silu_and_mul | Triton | 1 | 0.3ms | 0.3ms | 9.6ms |
| 13 | fused_moe (Down) | Triton | 1 | 2.8ms | 2.8ms | **89.6ms** |
| | **Total** | | | ~7ms | | **~220ms** |

**비교 — Llama-3-13B Dense**: prefill에서 FFN이 ~70% 차지, MoE는 MoE kernel이 ~80%.
MoE는 FLOPs 자체는 많지만 (k=2 여서 2× active) Mixtral 총 parameter가 더 크므로 품질이 높음.

## 커널별 역할과 비용 — Decode (T=1, A100)

| # | 커널 | Per layer | × 32 | Bound |
|---|------|----------:|-----:|-------|
| 1,7 | fused_add_rmsnorm | 30μs | 0.96ms | Memory |
| 2 | qkv_gemm (M=1) | 30μs | 0.96ms | Memory |
| 3-5 | rope/append/attn (GQA) | 50μs | 1.6ms | Memory |
| 6 | output_gemm (M=1) | 25μs | 0.8ms | Memory |
| 8 | router_gemm | 5μs | 0.16ms | Launch |
| 9 | topk | 2μs | 0.06ms | Launch |
| 10 | moe_align | 5μs | 0.16ms | Launch |
| 11 | fused_moe Gate+Up (M=1×2 experts) | 140μs | 4.48ms | Memory (weight) |
| 12 | silu_and_mul | 5μs | 0.16ms | Launch |
| 13 | fused_moe Down (M=1×2 experts) | 70μs | 2.24ms | Memory (weight) |
| | **Total** | **~365μs** | **~12ms** | |

**vs Llama-3-8B Dense decode: ~12ms** — 거의 비슷.
Mixtral이 47B total이지만 **active 13B**이므로 Llama-13B와 유사한 decode latency.
품질은 Mixtral이 더 좋음 (capacity 우위).

## A100 x2에서 Mixtral 8x7B 운영 옵션

Mixtral 8x7B bf16 = **94 GB** → 단일 A100 80GB에 안 올라감.

### 옵션 1: Quantization (권장)
- Mixtral-8x7B-Instruct-v0.1-GPTQ 4bit: ~25 GB
- 단일 A100 80GB에 올라감, decode 속도 개선

### 옵션 2: Tensor Parallelism (TP=2)
- Attention weight, FFN expert weight를 dim으로 split
- **NVLink 없음 → All-Reduce가 PCIe**: per-layer ~5-10ms 추가
- Prefill은 수용 가능, decode는 2x-3x 느려질 수 있음

### 옵션 3: Expert Parallelism (EP=2)
- Expert 4개씩 분산
- 매 layer all-to-all 2회 → PCIe로는 극도로 느림
- **NVLink 없는 A100 x2에서는 비추천**

### 옵션 4: CPU Offload + GPU 1개
- Expert 일부를 CPU RAM에 두고 필요 시 fetch
- Throughput 매우 낮음, 학습용만

## 현실적 권장

**A100 x2 (no NVLink), Mixtral 계열 study용**:
- Mixtral-8x7B AWQ/GPTQ 4bit on **single A100 80GB** → 가장 쉬운 환경
- 원한다면 두 번째 A100은 **다른 모델** (Llama-3-8B 등) 병행 실험
- TP는 NVLink 있는 환경에서 비교 실험용

## 다음 Section Preview

S4 (MLA)는 다른 방향의 진화:
- Mixtral style: FFN을 sparse (MoE)
- DeepSeek-V2 style: **KV cache 자체를 low-rank compress** (MLA)

둘 다 "active memory를 줄이는" 철학이지만 공격 지점이 다름.
DeepSeek-V3 (S5)는 **둘 다** 적용 + parallel attention/MoE.

## Examples

> [!NOTE]
> TODO: `examples/mixtral_kernel_profile.py` — nsys로 MoE 커널들 시각화
> TODO: `examples/mixtral_vs_llama_comparison.py` — 같은 prompt에서 비교

