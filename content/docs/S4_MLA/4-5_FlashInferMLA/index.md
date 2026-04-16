---
title: "4-5. FlashInfer MLA on A100"
weight: 5
---

# 4-5. FlashInfer MLA — Kernel & API

## FlashInfer의 MLA 지원

FlashInfer 0.6.3은 MLA 전용 API와 kernel을 제공:
- `BatchMLAPagedAttentionWrapper` / variant wrappers
- `append_paged_mla_kv_cache` — MLA KV cache append

MLA는 구조가 달라 기존 attention kernel을 재사용 불가.
FlashInfer는 별도 kernel family를 구현해 놓음.

## MLA KV Cache 레이아웃

DeepSeek-V2/V3의 MLA cache는 두 tensor로 구성:

```
ckv_cache:  [max_num_pages, page_size, 1, d_c]      
            = [P, page_size, 1, 512]
            per-page: page_size × 1 × 512 × 2B = page_size × 1024 B
            
kpe_cache:  [max_num_pages, page_size, 1, d_h^R]
            = [P, page_size, 1, 64]
            per-page: page_size × 1 × 64 × 2B = page_size × 128 B
```

- `ckv` = compressed KV ($c^{KV}$)
- `kpe` = K positional embedding part ($k^R$) — shared across heads
- `heads` 차원은 1로 저장 (head별로 다르지 않으니까)

## Cache Append API

```python
import torch
from flashinfer.page import append_paged_mla_kv_cache, get_batch_indices_positions

# DeepSeek-V2 setup
d_c = 512       # compressed KV dim
d_h_rope = 64   # RoPE part dim

# Cache buffers
num_pages = 10000
page_size = 16
ckv_cache = torch.zeros(num_pages, page_size, 1, d_c, dtype=torch.bfloat16, device="cuda")
kpe_cache = torch.zeros(num_pages, page_size, 1, d_h_rope, dtype=torch.bfloat16, device="cuda")

# Decode step: 3 requests, 1 new token each
append_indptr = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
seq_lens = torch.tensor([51, 201, 34], dtype=torch.int32, device="cuda")

batch_indices, positions = get_batch_indices_positions(
    append_indptr, seq_lens, nnz=3
)

# 새 token의 압축된 c^KV와 k^R 생성 (상위에서 GEMM으로 계산됨)
new_ckv = torch.randn(3, 1, d_c, dtype=torch.bfloat16, device="cuda")     # [nnz, 1, d_c]
new_kpe = torch.randn(3, 1, d_h_rope, dtype=torch.bfloat16, device="cuda") # [nnz, 1, d_h^R]

kv_indptr = torch.tensor([0, 4, 17, 20], dtype=torch.int32, device="cuda")
kv_indices = torch.arange(20, dtype=torch.int32, device="cuda")
kv_last_page_len = torch.tensor([2, 8, 1], dtype=torch.int32, device="cuda")

append_paged_mla_kv_cache(
    append_ckv=new_ckv,
    append_kpe=new_kpe,
    batch_indices=batch_indices,
    positions=positions,
    ckv_cache=ckv_cache,
    kpe_cache=kpe_cache,
    kv_indices=kv_indices,
    kv_indptr=kv_indptr,
    kv_last_page_len=kv_last_page_len,
)
# 내부 CUDA kernel: flashinfer::append_paged_mla_kv_cache
#   - ckv, kpe를 각각 올바른 page/position에 scatter write
```

## MLA Attention API

FlashInfer의 MLA prefill/decode wrapper (0.6.3 기준):

```python
# SGLang이 사용하는 MLA wrapper
# (이름과 API는 SGLang의 wrapper를 통해 접근)
from sglang.srt.layers.attention.flashinfer_mla_backend import FlashInferMLAAttnBackend

# 또는 flashinfer에서 직접:
from flashinfer.mla import BatchMLAPagedAttentionWrapper

workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
mla_wrapper = BatchMLAPagedAttentionWrapper(
    float_workspace_buffer=workspace,
    backend="auto",    # A100: fa2 family
)
```

### Plan & Run — MLA decode

```python
# DeepSeek-V2 configuration
num_heads = 128         # Q heads (MLA에선 kv heads 개념 없음)
head_dim_ckv = 512     # compressed KV head dim
head_dim_kpe = 64      # RoPE part head dim  
page_size = 16

# Plan with MLA-specific parameters
mla_wrapper.plan(
    qo_indptr=qo_indptr,
    kv_indptr=kv_indptr,
    kv_indices=kv_indices,
    kv_last_page_len=kv_last_page_len,
    num_heads=num_heads,
    head_dim_ckv=head_dim_ckv,
    head_dim_kpe=head_dim_kpe,
    page_size=page_size,
    causal=False,      # decode에선 causal 불필요
    sm_scale=1.0 / (head_dim_ckv + head_dim_kpe) ** 0.5,
    q_data_type="bfloat16",
)

# Run
# q_nope: Q compressed part (absorbed) [total_q_tokens, num_heads, d_c]
# q_pe:   Q RoPE part [total_q_tokens, num_heads, d_h^R]
q_nope = torch.randn(batch, num_heads, head_dim_ckv, dtype=torch.bfloat16, device="cuda")
q_pe   = torch.randn(batch, num_heads, head_dim_kpe, dtype=torch.bfloat16, device="cuda")

attn_out = mla_wrapper.run(
    q_nope=q_nope,
    q_pe=q_pe,
    ckv_cache=ckv_cache,
    kpe_cache=kpe_cache,
)
# attn_out shape: [batch, num_heads, head_dim_ckv]
# 주의: 이 output은 아직 compressed space. W̃_O를 곱해야 final output.

# 내부 CUDA kernel: flashinfer::BatchMLAPagedAttention<fa2_mla, ...>
#   - q_nope · c^KV^T: compressed part score
#   - q_pe · k^R^T: RoPE part score  
#   - 합산 후 softmax
#   - softmax · c^KV: output (compressed space)
#   - 모든 head가 같은 c^KV, k^R cache를 공유 → memory read 대폭 절약
```

## Kernel 구현 관점 — FlashInfer MLA가 어떻게 동작하는가

```cuda
// Pseudo CUDA kernel for MLA decode attention
// Thread block: one (request, q_head_group) pair

for kv_tile in kv_cache_pages:
  // Load c^KV and k^R tiles into shared memory
  load c_KV_tile [tile_size, d_c = 512] into smem
  load kpe_tile  [tile_size, d_h^R = 64] into smem
  
  for q_head in assigned_heads:
    // Compute score for this tile
    // 각 head가 같은 c_KV_tile, kpe_tile을 shared memory에서 read
    score_C = dot(q_nope[q_head], c_KV_tile)    // [tile_size]
    score_R = dot(q_pe[q_head],   kpe_tile)      // [tile_size]
    score = (score_C + score_R) * scale
    
    // Online softmax update
    update m, l, acc using score, c_KV_tile
  
// Final: for each head, acc is in compressed space [d_c]
write attn_out[q_head] = acc
```

**핵심**:
- K, V를 복원하지 않고 $c^{KV}, k^R$만 사용
- 모든 head가 shared memory의 $c^{KV}$를 공유 → HBM read 효율적
- Head-parallelism은 q, output에만 있음 (kv read는 공유)

## SGLang의 MLA Backend

```python
# sglang/srt/layers/attention/flashinfer_mla_backend.py
from sglang.srt.layers.attention import AttentionBackend
from flashinfer.mla import BatchMLAPagedAttentionWrapper

class FlashInferMLAAttnBackend(AttentionBackend):
    def init_forward_metadata(self, forward_batch):
        # plan() 호출
        self.mla_wrapper.plan(
            qo_indptr=forward_batch.qo_indptr,
            kv_indptr=forward_batch.kv_indptr,
            ...
        )
    
    def forward_extend(self, q_nope, q_pe, ckv, kpe, layer, forward_batch):
        # KV cache append
        forward_batch.token_to_mla_kv_pool.set_mla_buffer(
            layer, cache_loc, ckv, kpe
        )
        # Attention
        return self.mla_wrapper.run(
            q_nope, q_pe,
            forward_batch.token_to_mla_kv_pool.get_ckv_buffer(layer.layer_id),
            forward_batch.token_to_mla_kv_pool.get_kpe_buffer(layer.layer_id),
        )
    
    def forward_decode(self, q_nope, q_pe, ckv, kpe, layer, forward_batch):
        # 동일 구조, decode wrapper 사용
```

## 전체 MLA Decoder Block — DeepSeek-V2 one layer

```python
import torch
import torch.nn.functional as F
import flashinfer
from flashinfer.page import append_paged_mla_kv_cache
from flashinfer.mla import BatchMLAPagedAttentionWrapper

# Per-layer forward
def mla_layer_forward(x, residual, layer, positions, mla_wrapper, mla_pool):
    # 1. Fused RMSNorm
    hidden, residual = flashinfer.norm.fused_add_rmsnorm(
        x, residual, layer.attn_norm_weight, eps=1e-6
    )
    
    # 2. Down projections
    c_q = F.linear(hidden, layer.W_DQ)         # [T, d_c' = 1536]
    c_q = flashinfer.norm.rmsnorm(c_q, layer.q_norm, eps=1e-6)
    
    c_kv = F.linear(hidden, layer.W_DKV)        # [T, d_c = 512]
    # Note: c_kv는 normalize 안 함 (또는 small RMSNorm)
    
    # 3. Q up-projection + split into nope/pe
    q_full = F.linear(c_q, layer.W_UQ)          # [T, n_heads × (d_h + d_h^R)]
    q_full = q_full.view(-1, 128, 192)          # [T, n_heads=128, 192]
    q_nope = q_full[..., :128]                   # compressed part
    q_pe   = q_full[..., 128:]                   # RoPE part
    
    # 4. K RoPE part (shared across heads)
    k_pe = F.linear(hidden, layer.W_KR)          # [T, d_h^R = 64]
    
    # 5. Apply RoPE to q_pe and k_pe
    q_pe = apply_rope(q_pe, positions, ...)
    k_pe = apply_rope(k_pe.unsqueeze(1), positions, ...).squeeze(1)  # [T, d_h^R]
    
    # 6. Absorption: q_nope already 'in compressed space' via W_UQ
    #    But real implementation: W̃_Q = W_UQ @ W_UK^T
    #    q_absorbed = hidden @ W̃_Q instead of hidden @ W_UQ @ W_UK^T
    #    FlashInfer MLA handles both conventions
    
    # 7. Append to MLA KV cache
    mla_pool.append(layer.idx, c_kv.unsqueeze(1), k_pe.unsqueeze(1))
    # c_kv: [T, 1, d_c], k_pe: [T, 1, d_h^R]
    
    # 8. MLA attention
    attn_out = mla_wrapper.run(
        q_nope=q_nope, q_pe=q_pe,
        ckv_cache=mla_pool.get_ckv(layer.idx),
        kpe_cache=mla_pool.get_kpe(layer.idx),
    )
    # attn_out: [T, 128, d_c = 512]  (compressed output)
    
    # 9. Output absorption: apply W̃_O (which is W_UV^T @ W_O absorbed)
    attn_out = attn_out.view(-1, 128 * 512)
    output = F.linear(attn_out, layer.W_O_absorbed)   # [T, d_model]
    
    # 10. Residual + RMSNorm for FFN (standard)
    hidden, residual = flashinfer.norm.fused_add_rmsnorm(
        output, residual, layer.ffn_norm_weight, eps=1e-6
    )
    
    # 11. FFN (DeepSeek-V2 has MoE FFN — 12가지 subsection이 S5에서 본격)
    # ... MoE FFN ...
    
    return hidden, residual
```

## A100에서의 성능 특성

```
DeepSeek-V2 full (236B params, MoE)는 A100 80GB 1개엔 안 올라감.
그러나 'DeepSeek-V2-Lite' (16B)는 올라감.

MLA attention per layer:
  Prefill (T=2048):
    MLA kernel: ~300-500 μs
    vs GQA equivalent: ~300-400 μs
    → 약간 느리지만 KV cache 메모리 7x 절감
  
  Decode (T=1, context=4K):
    MLA kernel: ~80 μs (context-dependent)
    vs GQA: ~100 μs (KV read가 더 많음)
    → MLA가 더 빠름 (memory-bound win)

A100 80GB single:
  DeepSeek-V2-Lite (MLA) + KV cache: batch 50+ at 4K context
  GQA equivalent: batch 10-15 at 4K context
```

## A100 x2 관점

```
DeepSeek-V2-Lite (16B MoE): 
  BF16 = 32 GB → 단일 A100 80GB 여유롭게 탑재
  두 번째 A100은 다른 모델 또는 DP 증가용

DeepSeek-V2 Full (236B):
  BF16 = 472 GB → A100 80GB x2 합쳐도 부족
  → quantization (FP8, INT4) + TP/EP 필요
  → NVLink 없는 A100 x2에선 심하게 병목
  → 이 모델은 H100 node 환경에서만 실용적

→ 본 study의 A100 x2 환경에서 MLA 실험은 DeepSeek-V2-Lite 또는 DeepSeek-V2 quantized로 진행 가능
```

## 다음 Section — S5에서 다룰 것

DeepSeek-V3는 다음을 전부 가짐:
- MLA (S4에서 학습)
- 더 많은 작은 experts (DeepSeek-MoE의 확장)
- **Parallel Attention + MoE** (새로운 구조)
- FP8 training + inference
- Multi-Token Prediction

S5에서 이것들을 종합.

## Examples

{{< hint info >}}
TODO: `examples/mla_vs_gqa_cache.py` — 같은 설정에서 KV cache 크기 비교
TODO: `examples/mla_decode_bench.py` — A100에서 MLA decode latency
{{< /hint >}}
