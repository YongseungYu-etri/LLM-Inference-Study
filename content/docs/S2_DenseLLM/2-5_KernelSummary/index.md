---
title: "2-5. Kernel-Level Summary"
weight: 5
---

# 2-5. Kernel-Level Summary — Vanilla → Llama-style

## 커널 시퀀스 — Side-by-Side

### Vanilla Decoder Block (S1)

```
[1]  layernorm             ← mean + var reduction + normalize + scale + shift
[2]  qkv_gemm              ← cuBLAS, weight [d_m, 3*d_m]
[3]  kv_cache_append
[4]  flash_attention       ← n_kv = n_q
[5]  output_proj_gemm
[6]  residual_add          ← 별도 kernel (LN과 fuse 안 됨 일 수도)
[7]  layernorm
[8]  up_proj_gemm          ← weight [d_m, 4*d_m]
[9]  gelu                  ← elementwise
[10] down_proj_gemm        ← weight [4*d_m, d_m]
[11] residual_add
```

### Llama-style Decoder Block (S2)

```
[1]  fused_add_rmsnorm     ← residual + RMSNorm (1 kernel)
[2]  qkv_gemm              ← weight [d_m, d_q + d_k + d_v] — n_kv < n_q
[3]  apply_rope            ← Q, K에 회전 (elementwise)
[4]  kv_cache_append
[5]  flash_attention       ← GQA specialization
[6]  output_proj_gemm
[7]  fused_add_rmsnorm
[8]  gate_up_proj_gemm     ← MergedColumnParallelLinear, weight [d_m, 2*d_ff]
[9]  silu_and_mul          ← SiLU + Hadamard fused
[10] down_proj_gemm
```

## 커널별 차이 정리

| 커널 | Vanilla | Llama-style | 영향 |
|------|---------|-------------|------|
| Norm | `layer_norm` (mean+var+bias) | `fused_add_rmsnorm` (residual fused, no bias) | ~30% 빠름, HBM 1회 절감 |
| QKV GEMM | Q,K,V 같은 크기 | K,V 1/4 크기 (GQA) | weight 50% 감소 |
| PosEncoding | Input에 한 번 더함 (사전) | 매 layer에 RoPE | 커널 추가지만 outweighed by 이득 |
| Attention | MHA | GQA specialization | KV cache read $1/g$ |
| Activation | `gelu` 별도 | `silu_and_mul` fused | HBM 40% 절감 |
| FFN GEMMs | Up + Down (2개) | Gate+Up fused + Down (2개) | count 동일, shape 다름 |
| Residual | 별도 add | norm에 fuse | kernel 하나 줄음 |

## SGLang + FlashInfer 전체 경로 (Llama-3-8B)

```python
# 서버 시작 (1회)
workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
decode_wrapper  = flashinfer.BatchDecodeWithPagedKVCacheWrapper (workspace, kv_layout="NHD")

# 매 forward step마다 (모든 layer에서 같은 wrapper 재사용)
prefill_wrapper.plan(
    qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
    num_qo_heads=32, num_kv_heads=8,       # ← GQA
    head_dim_qk=128, page_size=16, causal=True,
    sm_scale=1/sqrt(128),
    q_data_type="bfloat16",
)

for layer in model.layers:  # 32 layers
    # [1] Fused Add + RMSNorm
    hidden, residual = flashinfer.norm.fused_add_rmsnorm(
        hidden, residual, layer.attn_norm_weight, eps=1e-6
    )

    # [2] QKV fused GEMM (n_kv=8)
    qkv = F.linear(hidden, layer.W_qkv)
    q, k, v = qkv.split([32*128, 8*128, 8*128], dim=-1)
    q = q.view(-1, 32, 128); k = k.view(-1, 8, 128); v = v.view(-1, 8, 128)

    # [3] RoPE (in-place on q, k)
    flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
        positions, q, k, head_size=128,
        cos_sin_cache=rope_cache, is_neox=True,
    )

    # [4] KV cache append (paged)
    kv_pool.set_kv_buffer(layer.idx, cache_loc, k, v)

    # [5] Prefill attention (GQA)
    attn_out = prefill_wrapper.run(q, kv_pool.get_kv_buffer(layer.idx))
    # → [CUDA] fa2_prefill_paged_run<GQA>

    # [6] Output projection
    attn_out = F.linear(attn_out.view(-1, 4096), layer.W_o)

    # [7] Fused Add + RMSNorm
    hidden, residual = flashinfer.norm.fused_add_rmsnorm(
        attn_out, residual, layer.ffn_norm_weight, eps=1e-6
    )

    # [8] Gate+Up fused GEMM
    gate_up = F.linear(hidden, layer.W_gate_up)   # [S, 2*14336]

    # [9] silu_and_mul fused
    ffn_mid = flashinfer.activation.silu_and_mul(gate_up)  # [S, 14336]

    # [10] Down projection
    hidden = F.linear(ffn_mid, layer.W_down)

# 최종 residual + 마지막 RMSNorm + lm_head
```

## A100 Performance Breakdown — Llama-3-8B

### Prefill (S=2048, batch=1)

| 커널 | Count/layer | Per call | Per layer | 32 layers | % |
|------|:-----------:|---------:|----------:|----------:|--:|
| fused_add_rmsnorm | 2 | 20 μs | 40 μs | 1.28 ms | 3% |
| QKV GEMM | 1 | 230 μs | 230 μs | 7.4 ms | 19% |
| apply_rope | 1 | 30 μs | 30 μs | 0.96 ms | 2% |
| kv_cache_append | 1 | 10 μs | 10 μs | 0.32 ms | 1% |
| prefill_attention (GQA) | 1 | 420 μs | 420 μs | 13.4 ms | 34% |
| output GEMM | 1 | 240 μs | 240 μs | 7.7 ms | 20% |
| gate_up GEMM | 1 | 250 μs | 250 μs | 8.0 ms | 20% |
| silu_and_mul | 1 | 40 μs | 40 μs | 1.28 ms | 3% |
| down GEMM | 1 | 250 μs | 250 μs | 8.0 ms | 20% |
| **Total** | | | ~1.5 ms | **~48 ms** | |

### Decode (B=1)

| 커널 | Per layer | 32 layers | % | Bound |
|------|----------:|----------:|--:|-------|
| fused_add_rmsnorm | 15 μs × 2 | 0.96 ms | 8% | Memory |
| QKV GEMM (M=1) | 30 μs | 0.96 ms | 8% | Memory |
| apply_rope | 3 μs | 0.1 ms | 1% | Memory |
| decode_attention (GQA) | 40 μs | 1.28 ms | 11% | Memory |
| output GEMM (M=1) | 25 μs | 0.8 ms | 7% | Memory |
| gate_up GEMM (M=1) | 140 μs | 4.48 ms | 38% | **Memory (weight dominates)** |
| silu_and_mul | 5 μs | 0.16 ms | 1% | Launch bound |
| down GEMM (M=1) | 70 μs | 2.24 ms | 19% | Memory |
| **Total** | | **~11.7 ms** (~85 tok/s) | | |

> **Decode는 거의 전부 memory-bound.**
> Batch size를 키워 Mb를 키워야 compute side에 여유가 생김.
> Continuous batching이 핵심 이유.

## A100 x2 (본 환경) — TP=2 고려사항

A100 x2 서버에 Llama-3-8B를 TP=2로 올리면:

```
Weight shard:
  W_qkv, W_o: column/row split → GEMM 절반씩
  W_gate_up: column split → 절반씩
  W_down: row split → 절반씩

Collective:
  Attention output → All-Reduce (또는 Reduce-Scatter + All-Gather)
  FFN down projection → All-Reduce

NVLink 없음 → PCIe + cross-NUMA QPI로 All-Reduce
  → Bandwidth ~16 GB/s (NVLink 300-600 GB/s 대비 수십 배 느림)
  → TP latency 크게 증가
```

**결론**: A100 x2 (NVLink 없음)에서는 TP=2보다 **단일 GPU로 각각 다른 request serving**하는
Data Parallel이 훨씬 유리. 또는 PP=2 (파이프라인 병렬, collective 최소화) 고려.

## Llama-3-8B vs Llama-3-70B — 같은 아키텍처, 다른 scale

| | 8B | 70B |
|---|---|---|
| $L$ (layers) | 32 | 80 |
| $d_m$ | 4096 | 8192 |
| $n_q$ / $n_{kv}$ | 32 / 8 | 64 / 8 |
| $d_{ff}$ | 14336 | 28672 |
| Single-token decode (A100) | ~12 ms | ~50 ms (OOM on single 80GB) |

70B는 단일 A100 80GB에 안 올라감 (BF16 = 140 GB) → TP 또는 quantization 필수.

## 다음 section Preview

S3 (MoE)에서는 이 Dense FFN을 **sparse하게** 바꿔,
decode에서 **active weight read**를 줄이는 방법을 다룬다.
→ 같은 "capacity"를 유지하면서 per-token compute/memory를 낮추는 전략.
