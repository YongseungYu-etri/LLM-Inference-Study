---
title: "5-7. Full Kernel Summary — V3 Decoder Block"
weight: 7
---

# 5-7. DeepSeek-V3 Decoder Block — Full Kernel Summary

## 전체 커널 시퀀스 (MoE Layer, 즉 layer 4~61)

```
[1]  fused_add_rmsnorm          ← residual + RMSNorm (attention 전)

── MLA Attention path ──
[2]  q_down_proj                ← x @ W_DQ → c_q [T, d_c'=1536]
[3]  q_rmsnorm                  ← c_q에 RMSNorm 
[4]  q_up_proj                  ← c_q @ W_UQ → q_full [T, 128 × (128+64)]
[5]  kv_down_proj               ← x @ W_DKV → c_kv [T, d_c=512]
[6]  kpe_proj                   ← x @ W_KR → k_pe [T, d_h^R=64]
[7]  apply_rope                 ← q의 RoPE part와 k_pe에 rotation
[8]  mla_kv_cache_append        ← (c_kv, k_pe)를 cache에 저장
[9]  mla_attention              ← FlashInfer MLA kernel
[10] output_proj_absorbed       ← attn_out @ W̃_O → [T, d_m]

[11] fused_add_rmsnorm          ← FFN 전

── MoE FFN path ──
[12] router_gemm                ← x @ W_router → [T, 256]
[13] router_bias_sigmoid        ← + bias, sigmoid (loss-free balancing)
[14] grouped_topk               ← group-wise top-4 groups → top-8 experts
[15] moe_align_block_size       ← dispatch 준비
[16] fused_moe_gate_up          ← Grouped GEMM (256 experts, 8 active per token)
[17] silu_and_mul               ← SwiGLU activation
[18] fused_moe_down             ← Grouped GEMM (down proj) + weighted sum
[19] shared_expert_gate_up      ← Shared expert Gate+Up GEMM
[20] shared_expert_silu         ← Shared expert silu_and_mul
[21] shared_expert_down         ← Shared expert Down GEMM
[22] add_routed_and_shared      ← routed_out + shared_out

    (residual은 다음 layer의 [1]에서 합산)
```

## 비교 표 — 모든 Section

| 단계 | Vanilla (S1) | Llama (S2) | Mixtral (S3) | V2 MLA (S4) | V3 (S5) |
|------|:---:|:---:|:---:|:---:|:---:|
| Norm | LayerNorm | **fused_add_rmsnorm** | fused_add_rmsnorm | fused_add_rmsnorm | fused_add_rmsnorm |
| Position | Abs PE | **RoPE** | RoPE | **Decoupled RoPE** | Decoupled RoPE |
| Attention | MHA | **GQA** | GQA | **MLA** | MLA |
| QKV | 3 GEMM | fused GEMM (small K,V) | fused | **down+up paths** | down+up paths |
| FFN | 2 linears | **SwiGLU (3 linears)** | **MoE (8 expert)** | MoE (V2) | **Fine-grained MoE + shared (256+1)** |
| Load balance | n/a | n/a | aux loss | aux loss | **loss-free bias** |
| Precision | fp32/bf16 | bf16 | bf16 | bf16 | **fp8** (H100) |
| Speculative | no | no | no | no | **MTP** |

## Code — V3 Layer Forward (한 번에 조감)

```python
import torch
import torch.nn.functional as F
import flashinfer
from flashinfer.norm import fused_add_rmsnorm, rmsnorm
from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
from flashinfer.page import append_paged_mla_kv_cache
from flashinfer.mla import BatchMLAPagedAttentionWrapper
from sglang.srt.layers.moe.topk import grouped_topk
from sglang.srt.layers.moe.fused_moe import fused_experts

def v3_layer_forward(
    x, residual, layer, positions,
    mla_wrapper, mla_pool,
):
    # ── [1] Pre-Attention RMSNorm ──
    hidden, residual = fused_add_rmsnorm(x, residual, layer.attn_norm_w, eps=1e-6)
    
    # ── MLA Attention ──
    # [2] Q down
    c_q = F.linear(hidden, layer.W_DQ)                  # [T, 1536]
    # [3] Q RMSNorm
    c_q = rmsnorm(c_q, layer.q_a_layernorm, eps=1e-6)
    # [4] Q up + split
    q_full = F.linear(c_q, layer.W_UQ)                  # [T, 128 × (128+64)]
    q_full = q_full.view(-1, 128, 192)
    q_nope, q_pe = q_full[..., :128], q_full[..., 128:]
    
    # [5] KV down
    c_kv_with_pe = F.linear(hidden, layer.W_DKV_combined)  # [T, d_c + d_h^R]
    c_kv, k_pe = c_kv_with_pe.split([512, 64], dim=-1)
    c_kv = rmsnorm(c_kv, layer.kv_a_layernorm, eps=1e-6)
    
    # [7] RoPE on q_pe, k_pe
    apply_rope_with_cos_sin_cache_inplace(
        positions, q_pe, k_pe.unsqueeze(1), head_size=64,
        cos_sin_cache=layer.rope_cache, is_neox=True,
    )
    
    # [8] Append MLA KV cache
    mla_pool.append(layer.idx, c_kv.unsqueeze(1), k_pe.unsqueeze(1))
    
    # [9] MLA Attention
    mla_wrapper.plan(...)  # once per step, before layer loop
    attn_out = mla_wrapper.run(
        q_nope=q_nope, q_pe=q_pe,
        ckv_cache=mla_pool.get_ckv(layer.idx),
        kpe_cache=mla_pool.get_kpe(layer.idx),
    )
    # attn_out: [T, 128, 512] (compressed)
    
    # [10] Absorbed output projection
    attn_out = attn_out.view(-1, 128 * 512)
    attn_proj = F.linear(attn_out, layer.W_O_absorbed)   # [T, d_m]
    
    # ── [11] Pre-FFN RMSNorm ──
    hidden, residual = fused_add_rmsnorm(
        attn_proj, residual, layer.ffn_norm_w, eps=1e-6
    )
    
    # ── MoE FFN ──
    # [12-14] Router + grouped top-k
    router_logits = F.linear(hidden, layer.W_router)    # [T, 256]
    topk_weights, topk_ids = grouped_topk(
        hidden_states=hidden,
        gating_output=router_logits,
        topk=8,
        num_expert_group=8,
        topk_group=4,
        renormalize=True,
        scoring_func="sigmoid",
    )
    # (bias는 layer.W_router에 이미 포함 or 별도 add)
    
    # [15-18] Routed experts
    routed_out = fused_experts(
        hidden_states=hidden,
        w1=layer.routed_w1,         # [256, 2*d_ff, d_m]
        w2=layer.routed_w2,         # [256, d_m, d_ff]
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    # 내부 kernel sequence:
    #   moe_align_block_size (dispatch)
    #   fused_moe_kernel (Gate+Up grouped GEMM)
    #   silu_and_mul
    #   fused_moe_kernel (Down grouped GEMM + weighted sum)
    
    # [19-21] Shared expert (standard SwiGLU)
    shared_gate_up = F.linear(hidden, layer.shared_W_gate_up)
    shared_mid = flashinfer.activation.silu_and_mul(shared_gate_up)
    shared_out = F.linear(shared_mid, layer.shared_W_down)
    
    # [22] Combine
    ffn_out = routed_out + shared_out
    
    return ffn_out, residual
```

## A100 Timing Breakdown (V2-Lite로 추정, V3 proxy)

V2-Lite (16B MoE, MLA, 27 layers of MoE + attention):

### Prefill (T=2048, single A100 80GB)

| 단계 | Per call | Per layer | Total (27 layers) |
|------|---------:|----------:|-------------------:|
| fused_add_rmsnorm ×2 | 20μs | 40μs | 1.08 ms |
| MLA Q/KV downs + up (4 GEMMs) | 200μs | 200μs | 5.4 ms |
| RoPE | 30μs | 30μs | 0.81 ms |
| MLA append cache | 15μs | 15μs | 0.4 ms |
| **MLA attention** | **400μs** | **400μs** | **10.8 ms** |
| Output proj | 150μs | 150μs | 4.05 ms |
| Router + topk | 25μs | 25μs | 0.68 ms |
| **Routed MoE (fused_moe, 64 expert)** | **3.5 ms** | **3.5 ms** | **94.5 ms** |
| Shared expert FFN | 2 × 200μs | 400μs | 10.8 ms |
| **Total** | | ~4.75 ms | **~128 ms** |

### Decode (T=1, context 4K)

| 단계 | Per layer | × 27 |
|------|----------:|-----:|
| fused_add_rmsnorm ×2 | 30 μs | 0.8 ms |
| MLA projections (4 GEMMs, M=1) | 120 μs | 3.2 ms |
| RoPE | 5 μs | 0.14 ms |
| MLA append cache | 5 μs | 0.14 ms |
| MLA attention (context=4K) | 80 μs | 2.16 ms |
| Output proj (M=1) | 50 μs | 1.35 ms |
| Router | 10 μs | 0.27 ms |
| Routed MoE (6 active, M=1 each) | 80 μs | 2.16 ms |
| Shared (M=1) | 60 μs | 1.62 ms |
| **Total** | **~440 μs** | **~11.8 ms** |

→ ~85 tok/s on single A100.

## 요약 — 이 study 전체 흐름

```
S1 Vanilla Transformer
  └→ 기본 연산 (MHA, FFN, LayerNorm, Residual, KV cache)
     ↓ 최적화
S2 Dense LLM (Llama)
  └→ +RMSNorm, +RoPE, +GQA, +SwiGLU, +no-bias
     ↓ Sparse로 확장
S3 MoE (Mixtral)
  └→ FFN을 8 experts로 분리, top-2 routing, Grouped GEMM
     ↓ Attention도 압축
S4 MLA (DeepSeek-V2)
  └→ KV cache를 low-rank로 압축, decoupled RoPE, matrix absorption
     ↓ 결합 + 더 fine-grained + 추가 혁신
S5 V3 (DeepSeek-V3)
  └→ Fine-grained MoE (256+1), loss-free balancing, MTP, FP8
```

각 step에서 **compute/memory efficiency**를 개선하면서 품질은 유지.
Inference 관점에서 가장 큰 wins:
- **GQA/MQA/MLA**: KV cache 절감 → batch 확대 → decode throughput ↑
- **MoE**: active param ↓ → decode weight-bound 완화
- **MTP**: per-step token 수 ↑ → throughput ↑
- **FP8**: weight memory ↓ → bandwidth-bound 완화 (H100)

## 본 study 이후 방향

1. **실측 연결**: tbd-project에서 각 variant의 실제 커널 launch 매핑 (study → 데이터)
2. **최적화 탐색**: 특정 커널(e.g., MLA attention)의 tuning/custom 구현
3. **Architecture 비교 실험**: 같은 조건에서 Dense vs MoE, GQA vs MLA latency/throughput
4. **본 study의 A100 x2 환경에 맞는 모델**: V2-Lite, Mixtral AWQ-4bit 등으로 실습

## 다음 — 전체 검토

모든 section이 완성되었다.
다음 task는 **S1-S5 전체 일관성 검토** 및 **cross-reference 정리**.

## Examples

> [!NOTE]
> TODO: `examples/v3_layer_forward.py` — V3 layer를 PyTorch mock으로 구현 (shape 확인)
> TODO: `examples/v3_kernel_profile.py` — V2-Lite의 실제 nsys profile (V3 proxy)

