---
title: "1-1. Overall Architecture"
weight: 1
---

# 1-1. Vanilla Transformer Decoder Block — Overall Architecture

## 개념

Transformer decoder block은 autoregressive language model의 기본 단위이다.
하나의 decoder block은 다음 두 sub-layer로 구성된다:

```
Input (hidden states)
  │
  ├─→ [LayerNorm] → [Self-Attention] → [+ Residual] ─→
  │                                                     │
  └─────────────────────────────────────────────────────┘
  │
  ├─→ [LayerNorm] → [FFN] → [+ Residual] ─→
  │                                          │
  └──────────────────────────────────────────┘
  │
Output (hidden states)
```

> 위 그림은 **Pre-LN** (LayerNorm을 sub-layer 앞에 배치) 구조이며,
> 현대 LLM (GPT-2 이후)의 표준이다. Original Transformer는 Post-LN이었다.

## Data Flow (Single Token, Single Layer)

입력: $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$ (하나의 토큰의 hidden state)

1. **LayerNorm**: $\hat{\mathbf{x}} = \text{LN}(\mathbf{x})$
2. **Self-Attention**: $\mathbf{a} = \text{Attn}(\hat{\mathbf{x}})$
3. **Residual**: $\mathbf{x}' = \mathbf{x} + \mathbf{a}$
4. **LayerNorm**: $\hat{\mathbf{x}}' = \text{LN}(\mathbf{x}')$
5. **FFN**: $\mathbf{f} = \text{FFN}(\hat{\mathbf{x}}')$
6. **Residual**: $\mathbf{x}'' = \mathbf{x}' + \mathbf{f}$

## 주요 파라미터

| Symbol | Meaning | Typical Values |
|--------|---------|----------------|
| $d_{\text{model}}$ | Hidden dimension | 4096, 5120, 8192 |
| $n_{\text{heads}}$ | Number of attention heads | 32, 40, 64 |
| $d_{\text{head}}$ | Per-head dimension ($d_{\text{model}} / n_{\text{heads}}$) | 128 |
| $d_{\text{ff}}$ | FFN intermediate dimension | $4 \times d_{\text{model}}$ 또는 $\frac{8}{3} \times d_{\text{model}}$ (SwiGLU) |
| $L$ | Number of layers | 32, 40, 80 |

## GPU 커널 관점 Preview

하나의 decoder block forward pass에서 launch되는 주요 커널 유형:

| 연산 | 커널 유형 | 대표 라이브러리 |
|------|-----------|-----------------|
| QKV / Output / FFN projection | GEMM | cuBLASLt (→ libcublasLt의 CUTLASS templates) |
| Attention score + softmax + value aggregation | Fused Attention | FlashInfer, FlashAttention (Triton) |
| LayerNorm | Reduction + Elementwise | Custom CUDA kernel |
| Residual add | Elementwise | 보통 fused (LayerNorm과 합쳐짐) |
| Activation (GELU/SiLU) | Elementwise | 보통 fused (FFN GEMM과 합쳐짐) |

> 각 연산의 상세는 subsection 1-2 ~ 1-6에서 다룬다.

## FlashInfer on A100 — Full Decoder Block Pipeline

SGLang + FlashInfer 스택에서 하나의 decoder block forward가 실행될 때,
실제 Python-level 호출 경로는 다음과 같다.

### 전체 흐름 (Pseudocode — SGLang model runner 기준)

```python
import torch
import flashinfer

# ── 서버 시작 시 1회: workspace 할당 + wrapper 생성 ──
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")  # 128MB

prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer, kv_layout="NHD", backend="auto"  # A100: fa2 backend 선택됨
)
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    workspace_buffer, kv_layout="NHD", backend="auto"
)

# ── 매 forward step마다: 각 layer를 순회 ──
for layer_idx in range(num_layers):
    # [1] RMSNorm + Residual (fused CUDA kernel)
    #     → flashinfer.rmsnorm() 또는 SGLang custom fused_add_rmsnorm
    hidden = fused_add_rmsnorm(residual, hidden, weight=rmsnorm_weight)

    # [2] QKV Projection (cuBLASLt GEMM via gemm_and_bias)
    qkv = torch.mm(hidden, W_qkv)            # shape: [num_tokens, 3 * d_model]
    q, k, v = qkv.split([d_q, d_k, d_v], dim=-1)

    # [3] RoPE (elementwise CUDA kernel)
    q, k = apply_rotary_pos_emb(q, k, positions)

    # [4] KV Cache Append + Attention (FlashInfer)
    #     → 아래 "Prefill path" 또는 "Decode path" 분기
    if is_prefill:
        kv_pool.set_kv_buffer(layer_idx, cache_loc, k, v)  # cache에 write
        prefill_wrapper.plan(...)                            # plan: index 준비
        attn_out = prefill_wrapper.run(                     # fused attention kernel
            q.view(-1, num_q_heads, head_dim),
            kv_pool.get_kv_buffer(layer_idx),               # (k_cache, v_cache) tuple
            causal=True, sm_scale=1/sqrt(d_h),
        )
    else:  # decode
        kv_pool.set_kv_buffer(layer_idx, cache_loc, k, v)
        attn_out = decode_wrapper.run(                      # fused decode kernel
            q.view(-1, num_q_heads, head_dim),
            kv_pool.get_kv_buffer(layer_idx),
            sm_scale=1/sqrt(d_h),
        )

    # [5] Output Projection (cuBLASLt GEMM)
    attn_out = torch.mm(attn_out.view(-1, d_model), W_o)

    # [6] Residual + RMSNorm (fused)
    hidden = fused_add_rmsnorm(residual, attn_out, weight=rmsnorm_weight_2)

    # [7] FFN: Gate+Up Projection (cuBLASLt GEMM)
    gate_up = torch.mm(hidden, W_gate_up)     # [num_tokens, 2 * d_ff]

    # [8] SiLU + Hadamard (elementwise CUDA kernel)
    ffn_out = silu_and_mul(gate_up)           # [num_tokens, d_ff]

    # [9] Down Projection (cuBLASLt GEMM)
    ffn_out = torch.mm(ffn_out, W_down)       # [num_tokens, d_model]

    # residual은 다음 layer의 [1]에서 합산
```

### 커널 ↔ 라이브러리 매핑 (A100 기준 — SGLang 0.5.9 venv에서 검증)

| # | 연산 | 실제 커널 | 경로 | Bound |
|---|------|----------|------|-------|
| 1,6 | RMSNorm + Residual | `fused_add_rmsnorm` | FlashInfer / SGLang custom | Memory |
| 2,5,7,9 | Linear Projection | **`ampere_bf16_s16816gemm_*`** | **cuBLASLt** via PyTorch `gemm_and_bias` wrapper | Compute (prefill) / Memory (decode) |
| 3 | RoPE | `rotary_embedding_kernel` | SGLang custom (vLLM kernels 경로) | Memory |
| 4 (prefill) | Attention | `flashinfer::fa2_*_paged_run` | FlashInfer FA2 (CUTLASS) | Compute |
| 4 (decode) | Attention | `flashinfer::BatchDecodeWithPagedKVCache` | FlashInfer FA2 | Memory |
| 8 | SiLU × Gate | `silu_and_mul_kernel` | SGLang custom | Memory |

> **A100에서 FlashInfer backend="auto"는 `fa2` (Flash Attention v2, CUTLASS 기반)를 선택.**
> H100 이상에서는 `fa3` 또는 `cudnn`이 선택될 수 있음.

> **Dense GEMM 실호출 경로는 cuBLAS가 아니라 cuBLASLt다.**
> `F.linear` → `at::native::linear` → `at::matmul` → `at::mm` → `addmm_out_cuda_impl` →
> `gemm_and_bias<BFloat16>` → `cublasLtMatmul` → libcublasLt.so의 precompiled CUTLASS template
> (`ampere_bf16_s16816gemm_*` 또는 fallback `cutlass::Kernel2<cutlass_80_tensorop_*_align8>`).
> 전체 file:line 경로는 **[1-8 Dense GEMM Call Path]({{< relref "1-8_DenseGEMMCallPath" >}})**.

### 왜 plan() → run() 2-phase인가?

```
plan(): KV page table의 indptr/indices로부터 각 request의 attention 범위를 계산하고,
        split-KV 전략 (긴 시퀀스를 여러 블록에 분산) 등의 scheduling을 결정.
        → CPU-side 연산 + 소량의 GPU buffer write

run():  실제 fused attention CUDA kernel을 launch.
        plan() 결과를 참조하여 각 thread block이 어떤 Q/K/V 범위를 담당할지 결정됨.
```

이 2-phase 설계 덕분에 CUDA graph capture가 가능하고,
같은 plan을 여러 layer에서 재사용할 수 있다 (layer간 KV page 구조가 동일하므로).

## Examples

> [!NOTE]
> TODO: 간단한 PyTorch 참조 구현 — `examples/vanilla_decoder_block.py`

