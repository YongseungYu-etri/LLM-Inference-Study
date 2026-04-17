---
title: "1-8. Dense GEMM Call Path (SGLang → cuBLASLt)"
weight: 8
---

# 1-8. Dense GEMM Call Path — SGLang → cuBLASLt (검증)

> **Verified in**: SGLang 0.5.9 + PyTorch 2.x (CUDA 12.x build) + libcublasLt.so.12 @ A100.
> 출처: `tbd-project/results/CALL_STACK_F_LINEAR_TO_CUBLASLT.md` (file:line까지 직접 검증).

앞의 1-1 ~ 1-7에서 "QKV projection → cuBLAS GEMM"이라고 축약해 썼지만,
실제 SGLang default path에서 dense Linear는 **항상 cuBLASLt** 경로로 떨어진다.
**cuBLAS는 fallback에만 등장.** 이 subsection에서 그 실체를 file:line 수준으로 본다.

## TL;DR — 정확한 경로

```
SGLang ColumnParallelLinear.forward
  → UnquantizedLinearMethod.apply              ← dispatch 분기
    → F.linear (= torch._C._nn.linear 바인딩)
      → at::_ops::linear::call
        → at::native::linear                    ← composite implicit autograd
          → at::matmul → at::mm
            → structured_mm_out_cuda::impl      ← β=0으로 addmm에 위임
              → addmm_out_cuda_impl             ← ★ backend 분기
                ├── gemm_and_bias<BFloat16>     ← cuBLASLt 경로 (default)
                │     → cublasLtMatmulAlgoGetHeuristic
                │     → cublasLtMatmul
                │       └── libcublasLt.so.12 내부
                │            └── ampere_bf16_s16816gemm_*  (사전 컴파일 CUTLASS)
                │                또는 cutlass::Kernel2<cutlass_80_tensorop_*_align8>
                └── gemm_internal<BFloat16>     ← fallback (일부 조건)
                      → cublasGemmEx / cublasGemmStridedBatchedEx
                        └── 같은 CUTLASS kernel pool
```

> **실측 결론**: BF16 + Ampere + 우리 shape들 모두 cuBLASLt 적격 → **모든 dense GEMM이 위쪽 경로**로 흐름.
> cuBLAS 경로는 실제로 hit되지 않음.

## Stage별 상세 (요약)

### Stage 1 — SGLang Linear family

**File**: `sglang/srt/layers/linear.py`

| Class | Role | Llama-3 관점 |
|-------|------|--------------|
| `LinearBase` (L140) | 추상 부모, `quant_method` 보유 | — |
| `ColumnParallelLinear` (L277) | TP column split | QKV, gate_up proj의 베이스 |
| `MergedColumnParallelLinear` (L469) | 2+ weight concat | `gate_up_proj` (SwiGLU) |
| `QKVParallelLinear` (L834) | Q/K/V 합친 weight | `qkv_proj` (GQA 반영) |
| `RowParallelLinear` (L1280) | TP row split | `o_proj`, `down_proj` |

```python
# linear.py:446 (ColumnParallelLinear.forward)
output_parallel = self.quant_method.apply(self, input_, bias)
```

### Stage 2 — Dispatch 분기

**File**: `sglang/srt/layers/quantization/unquant.py`

```python
# unquant.py:151 (UnquantizedLinearMethod.apply)
def apply(self, layer, x, bias=None):
    if _USE_FLASHINFER_GEMM:            # env FLASHINFER_USE_CUTLASS_GEMM=1
        return seg_gemm.run(...)        # PACT26 patch path
    if use_intel_amx_backend(layer):
        return torch.ops.sgl_kernel.weight_packed_linear(...)
    return F.linear(x, layer.weight, bias)   # ★ default (우리 경로)
```

**Knob**: `FLASHINFER_USE_CUTLASS_GEMM=1` 환경변수로 FlashInfer CUTLASS segment GEMM으로 우회 가능.
이 flag는 proja-ysyu의 PACT26 실험에서 사용됨. 본 study의 default는 OFF.

### Stage 3-4 — PyTorch F.linear → ATen

**File**: `torch/nn/functional.py:2302` — `F.linear`는 **순수 docstring wrapper**:
```python
linear = _add_docstr(torch._C._nn.linear, """...""")
```

실체는 C++ pybind: `torch._C._nn.linear` → `aten::linear` op → **`at::native::linear`**.

```cpp
// at::native::linear (aten/src/ATen/native/Linear.cpp)
Tensor linear(const Tensor& input, const Tensor& weight,
              const std::optional<Tensor>& bias_opt) {
  if (input.dim() == 2 && bias_opt.has_value()) {
    return at::addmm(*bias_opt, input, weight.t());    // 2D + bias
  }
  if (input.dim() == 3 && bias_opt.has_value() && input.is_contiguous()) {
    return at::_unsafe_view(                            // 3D + bias (reshape)
        at::addmm(*bias_opt, input.view({-1, input.size(-1)}), weight.t()),
        {input.size(0), input.size(1), -1});
  }
  auto output = at::matmul(input, weight.t());          // bias 없음
  if (bias_opt.has_value()) output.add_(*bias_opt);
  return output;
}
```

**Llama-style은 no-bias 모델**이므로 항상 `at::matmul` 경로.

### Stage 5 — matmul → mm (혹은 bmm)

**File**: `aten/src/ATen/native/LinearAlgebra.cpp`

```cpp
Tensor matmul(const Tensor& t1, const Tensor& t2) {
  if (t1.dim() == 2 && t2.dim() == 2) return t1.mm(t2);      // [M,K]×[K,N]
  if (t1.dim() == 3 && t2.dim() == 2)
      return t1.view({-1, t1.size(-1)}).mm(t2).view(...);    // flatten to 2D then mm
  if (t1.dim() >= 1 && t2.dim() >= 1) return at::bmm(...);   // batched
}
```

Llama forward는 대부분 `[B, K] @ [K, N]` 또는 `[B, S, K] @ [K, N]` → 결국 **2D mm**로 수렴.

### Stage 6 — CUDA dispatch (mm도 addmm로 수렴)

**File**: `aten/src/ATen/native/cuda/Blas.cpp` (libtorch_cuda.so)

```cpp
// structured_mm_out_cuda::impl
TORCH_IMPL_FUNC(mm_out_cuda)(const Tensor& self, const Tensor& mat2, const Tensor& result) {
  addmm_out_cuda_impl(const_cast<Tensor&>(result), result,
                      self, mat2, /*beta=*/0, /*alpha=*/1);
  // β=0 → C에 기록된 이전 값 무시 → 실질적으로 mm과 동일
}
```

→ **`mm`은 `addmm`의 special case**. 모든 Linear의 커널은 결국 `addmm_out_cuda_impl`로 수렴.

### Stage 7 — cuBLASLt 분기

```cpp
// addmm_out_cuda_impl 핵심 (요약)
auto preferred_backend = at::globalContext().blasPreferredBackend();
bool use_lt = (preferred_backend == BlasBackend::Cublaslt) ||
              (preferred_backend == BlasBackend::Default && lt_heuristic_ok());

if (use_lt && scalar_type_supports_lt(mat1.scalar_type())) {
  bool ok = at::cuda::blas::gemm_and_bias<at::BFloat16>(
      transpose_mat1, transpose_mat2, m, n, k, alpha,
      mat1_ptr, lda, mat2_ptr, ldb,
      bias_ptr, result_ptr, ldc, activation);
  if (ok) return result;                              // ★ cuBLASLt 경로
}
at::cuda::blas::gemm_internal<at::BFloat16>(...);     // fallback: cublasGemmEx
```

**Knob**: `torch.backends.cuda.preferred_blas_library("cublas" | "cublaslt")`로 강제 가능.
현재 PyTorch default는 `Default` (heuristic-based).

### Stage 8 — `gemm_and_bias<BFloat16>` → libcublasLt

**File**: `aten/src/ATen/cuda/CUDABlas.cpp`

핵심 두 API:
```cpp
// 1. Best algo 1개 선택 (M,N,K,dtype,layout,sm_arch 기반)
cublasLtMatmulAlgoGetHeuristic(lt, computeDesc, Adesc, Bdesc, Cdesc, Cdesc,
                                pref, /*request=*/1, &heuristic, &returned);

// 2. 선택된 algo로 kernel launch
cublasLtMatmul(lt, computeDesc,
               &alpha, A_ptr, Adesc, B_ptr, Bdesc,
               &beta, C_ptr, Cdesc, C_ptr, Cdesc,
               &heuristic.algo, workspace, ws_size,
               getCurrentCUDAStream());
```

**Symbol 존재 확인** (본 환경 venv에서 검증):
```
U cublasLtMatmul@libcublasLt.so.12
U cublasLtMatmulAlgoGetHeuristic@libcublasLt.so.12
U cublasLtMatmulDescCreate / SetAttribute / Destroy@libcublasLt.so.12
U cublasLtMatrixLayoutCreate@libcublasLt.so.12
```

### Stage 9 — libcublasLt 내부 (closed-source)

`cublasLtMatmul`이 내부에서:
1. `(M,N,K,dtype,layout,sm80)` → **사전 컴파일된 CUTLASS template catalog**에서 best tile 선택
2. `cudaLaunchKernel`로 띄움
3. nsys trace에 다음 family로 잡힘:

| Family | nsys 이름 패턴 | 비고 |
|--------|----------------|------|
| **cuBLAS-precompiled CUTLASS** | `ampere_bf16_s16816gemm_bf16_{N}x{M}[_sliced1x2]_ldg8[_relu]_f2f_stages_{K}x{ST}_tn` | 대다수 |
| **CUTLASS-Kernel2 fallback** | `void cutlass::Kernel2<cutlass_80_tensorop_*_align8>(T1::Params)` | 일부 (M,N,K) |
| Warmup only | `wmma_16x16_f16_nn` | 모델당 정확히 84회 |

**관찰 (D2/D3 sweep, Llama/Mistral/Qwen decode)**:
- `ampere_bf16_s16816gemm_*`가 압도적 다수
- `cutlass::Kernel2<>` fallback은 Llama-3-8B 2237회, Mistral 2242회, Qwen2.5 6659회

## 왜 "cuBLAS"가 아니라 "cuBLASLt"인가

혼동 주의: "cuBLAS"는 legacy API (`cublasGemmEx` 등), "cuBLASLt"는 별도 library (`libcublasLt.so`).

| | cuBLAS (legacy) | **cuBLASLt** |
|---|:---:|:---:|
| Library | libcublas.so | **libcublasLt.so** |
| Main API | `cublasGemmEx`, `cublasGemmStridedBatchedEx` | **`cublasLtMatmul`** |
| Epilogue fusion | 제한적 | **bias, GELU, ReLU 등 fuse 가능** |
| Workspace | 암묵적 | **명시적 management** |
| Algo 선택 | 내부 fixed | **`AlgoGetHeuristic` 명시 선택** |
| 최신 SM 최적화 | 보수적 | **공격적 (새 Hopper WGMMA 등 우선 활용)** |

PyTorch 2.x부터 BF16/FP16 GEMM은 **기본적으로 cuBLASLt로 dispatch**.
cuBLAS 경로는 거의 쓰이지 않음 (우리 환경의 실측도 이를 확인).

## 커널 이름 공식적 분해

`ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_32x3_tn`의 의미:

| 부분 | 뜻 |
|------|-----|
| `ampere` | SM80 (A100) target |
| `bf16` | A/B input dtype |
| `s16816` | tensor core MMA 명령 `m16n8k16` (Ampere BF16) |
| `gemm` | 연산 종류 |
| `bf16` | C/D output dtype |
| `128x256` | threadblock-level tile (M×N) |
| `ldg8` | 128-bit vectorized loads |
| `f2f` | format-to-format (dtype conversion path) |
| `stages_32x3` | K tile size × pipeline stage 수 |
| `tn` | transpose ops: A^T × B (no-trans-B) |

같은 GEMM이라도 (M, N, K)에 따라 다른 tile이 선택됨 → CUDA graph에 들어가면
그 run에서는 pinned된 algo 사용 (재 heuristic lookup X).

## 왜 이 지식이 optimization 전략에 중요한가

이 매핑이 안 정확하면 다음 실수를 하게 됨:

1. **"cuBLAS tile tuning"으로 접근** → 존재하지 않는 경로 (실제는 cuBLASLt + CUTLASS 내부 카탈로그)
2. **CUTLASS grouped GEMM을 직접 호출하려 시도** → SGLang이 그 path를 쓰지 않음 (env flag 필요)
3. **nsys kernel 이름을 못 알아봄** → `sm80_xmma_gemm_*` (가짜) vs `ampere_bf16_s16816gemm_*` (실제)
4. **Optimization headroom 과대평가** → cuBLASLt 카탈로그는 closed-source + hard-coded per SM
   → PyTorch 외부에서 직접 kernel 선택 불가 (env-level hint만 가능)

**실측 결론 (tbd-project D2 finding)**: 
A100에서 HBM roofline 70-86% 활용 중 → dense GEMM에서 짜낼 수 있는 여지는 ~20% 이내.

## Knob 총정리

| Layer | Env / API | 효과 |
|-------|-----------|------|
| Stage 2 | `FLASHINFER_USE_CUTLASS_GEMM=1` | cuBLASLt 대신 FlashInfer cutlass_segment_gemm |
| Stage 6 | `torch.backends.cuda.preferred_blas_library()` | Lt ON/OFF 강제 |
| Stage 7 | `CUBLASLT_LOG_LEVEL=5` | 선택된 algo 로그 (D1 finding 참조) |
| Stage 7 | `CUBLASLT_WORKSPACE_SIZE` | algo 후보 풀 영향 |
| Stage 8 | — | libcublasLt 내부 카탈로그는 직접 건드릴 수 없음 |

## Cross-References

- **1-1 Overall Architecture**: kernel 매핑 표의 "cuBLAS" → 실제는 "cuBLASLt"
- **1-2 Self-Attention**: QKV/Output proj도 동일 경로 (GQA에서도 불변)
- **1-4 FFN**: Gate+Up / Down proj GEMM도 동일 경로
- **2-5 Kernel Summary**: QKV/FFN GEMM이 모두 이 경로
- **tbd-project** `CALL_STACK_F_LINEAR_TO_CUBLASLT.md`: 원본 ground truth
- **tbd-project** `FINDINGS_D1_CUBLASLT_HEURISTIC.md`: `CUBLASLT_LOG_LEVEL=5` 결과
- **tbd-project** `FINDINGS_D2_KERNEL_LEVEL.md`: kernel-level 시간 분포 / roofline

## Re-verification Commands

```bash
# SGLang Linear 구조
grep -n -E "^class (LinearBase|ColumnParallelLinear|MergedColumnParallelLinear|QKVParallelLinear|RowParallelLinear)\b" \
  /home/ysyu/sglang-venv/lib/python3.10/site-packages/sglang/srt/layers/linear.py

# cuBLASLt symbols
nm -D --demangle /home/ysyu/sglang-venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so 2>/dev/null \
  | grep -E "(cublasLt|gemm_and_bias)" | head -20

# Backend 확인 (runtime)
python -c "import torch; print(torch.backends.cuda.preferred_blas_library())"
```

## 정리 — 모든 S1-S5 section에 적용되는 교정 rule

| 제가 어딘가에 쓴 표현 | 정확한 표현 |
|-----|-----|
| "cuBLAS GEMM" | **"cuBLASLt GEMM (PyTorch `gemm_and_bias` wrapper)"** |
| `sm80_xmma_gemm_*` | **`ampere_bf16_s16816gemm_*`** |
| "cublasGemmEx() 또는 cublasLtMatmul()" | **"cublasLtMatmul (via ATen addmm_out_cuda_impl)"** |
| "cuBLAS (Ampere HMMA)" | **"cuBLASLt → libcublasLt.so 내부 CUTLASS template"** |

이 subsection은 S2-S5에서도 **모든 dense GEMM**에 동일하게 적용된다 (QKV/output/FFN/gate_up/down).
MoE grouped GEMM은 다른 경로 (Triton fused_moe or CUTLASS grouped GEMM) — **S3-3에서 이미 정확히 다룸**.
MLA attention은 FlashInfer MLA wrapper 경로 — **S4-5에서 이미 정확히 다룸**.

---

# Part 2: Decoder Block 전체 구조 — Before / Loop / After

> **Verified**: 2026-04-17, 동일 환경 (SGLang 0.5.9 + FlashInfer 0.6.3, A100 SM80).
> 출처: `tbd-project/results/CALL_STACK_F_LINEAR_TO_CUBLASLT.md` Part 2.

## 전체 Forward Pass 구조

```
LlamaForCausalLM.forward                          (llama.py:574)
 │
 ├── [BEFORE LOOP] ─────────────── 1회
 │    embed_tokens(input_ids)      ← embedding lookup (GEMM 아님, index_select)
 │    residual = None
 │
 ├── [LOOP ×32] ────────────────── num_hidden_layers 반복
 │    for i in range(start_layer, end_layer):
 │        hidden_states, residual = layer(positions, hidden_states, forward_batch, residual)
 │
 └── [AFTER LOOP] ──────────────── 1회
      norm(hidden_states, residual)  ← fused_add_rmsnorm (loop 내와 동일)
      logits_processor(...)
        └── torch.matmul(hidden, lm_head.weight.T)   ★ F.linear 아님, torch.matmul 직접
            [B, 4096] @ [4096, 128256] → [B, 128256]
```

### lm_head는 SGLang Linear 경로를 안 탄다

| 항목 | 값 |
|------|-----|
| 진입점 | `torch.matmul` 직접 호출 (logits_processor.py:881) |
| SGLang `LinearBase` 계열? | **아님**. `ParallelLMHead(VocabParallelEmbedding)` |
| PACT26 패치(`_USE_FLASHINFER_GEMM`) 적용? | **안 됨**. `UnquantizedLinearMethod.apply`를 경유 안 함 |
| cuBLASLt 경로? | **예**. `torch.matmul` → `at::matmul` → `at::mm` → cuBLASLt (Stage 5부터 동일) |

## Decoder Block 내부: LlamaDecoderLayer (×32)

**File**: `llama.py:309-390` — **정확한 kernel 순서 (검증)**:

```
LlamaDecoderLayer.forward(positions, hidden_states, forward_batch, residual)
 │
 ├─[1] input_layernorm(hidden_states, residual)
 │      layer 0: rmsnorm(x)  /  layer 1~31: fused_add_rmsnorm(x, residual)
 │
 ├─[2] self_attn
 │   ├─[2a] qkv_proj(x)          GEMM #1 — [B,4096]@[4096,6144] → cuBLASLt
 │   │      → split → q[B,4096], k[B,1024], v[B,1024]
 │   │
 │   ├─[2b] rotary_emb(pos, q, k) RoPE — FlashInfer BatchQKApplyRotaryPosIdsCosSinCacheKernel
 │   │
 │   ├─[2c] attn(q, k, v)         Attention — KV write + FlashInfer kernel (아래 상세)
 │   │
 │   └─[2d] o_proj(attn_out)      GEMM #2 — [B,4096]@[4096,4096] → cuBLASLt
 │
 ├─[3] post_attention_layernorm    fused_add_rmsnorm
 │
 └─[4] mlp
     ├─[4a] gate_up_proj(x)       GEMM #3 — [B,4096]@[4096,28672] → cuBLASLt
     ├─[4b] silu_and_mul           FlashInfer act_and_mul_kernel
     └─[4c] down_proj(x)          GEMM #4 — [B,14336]@[14336,4096] → cuBLASLt
```

## KV Cache — Write와 Read의 비대칭성

| 동작 | 별도 kernel? | 설명 |
|------|:-----------:|------|
| **Write** | **Yes** — `sgl_kernel.store_kvcache` | `set_kv_buffer()` → 새 token의 k,v를 layer별 buffer에 scatter write |
| **Read** | **No** — attention kernel 내부 implicit | `get_kv_buffer(layer_id)` → tensor reference만 반환. 실제 HBM read는 attention kernel의 `cp_async` prefetch |

```python
# flashinfer_backend.py:882-895 (decode path)
if save_kv_cache:
    forward_batch.token_to_kv_pool.set_kv_buffer(    # ← [WRITE] 별도 kernel launch
        layer, cache_loc, k, v, ...)
o = decode_wrapper.forward(
    q, forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),  # ← [READ] 포인터만
    sm_scale=..., ...)
```

## FlashInfer Attention Kernel — "FA2"의 실체 (교정)

### 핵심 교정: FlashInfer는 외부 라이브러리를 사용하지 않는다

**기존 study에서의 오류**: "FlashInfer FA2 (CUTLASS)", "`flashinfer::fa2_*_paged_run`"

**실제**: FlashInfer는 **자체 CUDA kernel을 JIT 컴파일**하여 사용. "FA2"는 Flash Attention 2의 **tile-based softmax recomputation 알고리즘**을 차용한 것이지, Tri Dao의 FA2 라이브러리/CUTLASS/Triton-FA/cuDNN을 호출하는 것이 아님.

### Backend 선택 로직 (A100 SM80)

```
should_use_tensor_core(kv_dtype=bf16, n_q_heads, n_kv_heads):
    gqa_group = n_q_heads / n_kv_heads
    return (gqa_group >= 4)    # BF16 기준
```

| 모델 | Q heads | KV heads | GQA group | use_tensor_cores | Kernel |
|------|---------|----------|:---------:|:----------------:|--------|
| Llama-3-8B | 32 | 8 | 4 | **True** | `BatchPrefillWithPagedKVCacheKernel` |
| Mistral-7B | 32 | 8 | 4 | **True** | `BatchPrefillWithPagedKVCacheKernel` |
| Qwen2.5-7B | 28 | 4 | 7 | **True** | `BatchPrefillWithPagedKVCacheKernel` |
| Qwen-MoE | 16 | 16 | 1 | **False** | `BatchDecodeWithPagedKVCacheKernel` |
| DeepSeek-V2 | MLA | MLA | — | — | `BatchMLAPagedAttentionKernel` |

### nsys에서 보이는 실제 kernel 이름

| 조건 | Kernel | nsys 이름 |
|------|--------|-----------|
| GQA≥4 (tensor core) | FlashInfer JIT prefill | `BatchPrefillWithPagedKVCacheKernel<...>` |
| GQA<4 (standard) | FlashInfer JIT decode | `BatchDecodeWithPagedKVCacheKernel<...>` |
| MLA | FlashInfer JIT MLA | `BatchMLAPagedAttentionKernel<...>` |

### JIT 컴파일 경로

```
FlashInfer wrapper.plan() or first run
  → get_batch_prefill_module("fa2", ...)           (decode.py:1051)
    → flashinfer.jit.attention.modules.gen_batch_prefill_module()
      → Jinja2 template rendering
        → CUDA source generation
          → nvcc JIT compile → .so
            → pybind11 module load
```

### Kernel 구현 위치

```
flashinfer/data/include/flashinfer/attention/
  decode.cuh:613    BatchDecodeWithPagedKVCacheKernel (template)
  prefill.cuh       BatchPrefillWithPagedKVCacheKernel (tensor core path)
  variants.cuh:31   DefaultAttention struct (softmax variant)
```

자체 작성 CUDA `__global__` kernel. Shared memory pipelining (`cp_async`), split-K reduction, online softmax 등 구현.

## 전체 Kernel Count (1 decode step, Llama-3-8B)

```
1회만 (loop 밖):
  embed_tokens         ── embedding index_select              1회
  final RMSNorm        ── sgl_kernel fused_add_rmsnorm        1회
  lm_head              ── GEMM via cuBLASLt (torch.matmul)    1회

32회 (× num_layers):
  input_layernorm      ── sgl_kernel fused_add_rmsnorm       32회
  qkv_proj             ── GEMM via cuBLASLt                  32회
  rotary_emb           ── FlashInfer RoPE kernel             32회
  KV cache write       ── sgl_kernel store_kvcache           32회
  attention            ── FlashInfer BatchPrefill/Decode*    32회
  o_proj               ── GEMM via cuBLASLt                  32회
  post_attn_layernorm  ── sgl_kernel fused_add_rmsnorm       32회
  gate_up_proj         ── GEMM via cuBLASLt                  32회
  SiLU activation      ── FlashInfer act_and_mul_kernel      32회
  down_proj            ── GEMM via cuBLASLt                  32회
─────────────────────────────────────────────────────────────
Total CUDA kernel launches:    ~323회/step

내역:
  GEMM:       4 × 32 + 1 (lm_head)  = 129회
  RMSNorm:    2 × 32 + 1 (final)    =  65회
  Attention:  1 × 32                 =  32회
  RoPE:       1 × 32                 =  32회
  KV write:   1 × 32                 =  32회
  SiLU:       1 × 32                 =  32회
  Embedding:                            1회
```

## Layer 간 데이터 흐름

| 데이터 | 32개 layer 걸쳐 변화? | 상세 |
|--------|:--------------------:|------|
| `hidden_states` | **매 layer 갱신** | layer output → 다음 input. shape `[B, H]` 유지 |
| `residual` | **매 layer 갱신** | fused_add_rmsnorm이 in-place 갱신 |
| `positions` | **불변** | 동일한 position 벡터가 32개 layer RoPE에 전달 |
| `forward_batch` | **불변** (구조체) | batch metadata (cache_loc, seq_lens 등) |
| weight | **layer마다 고유** | `layers[i]` 각각 독립 weight set |
| KV cache | **layer별 독립 축적** | `k_buffer[layer_id]` — 매 token마다 1행 append |

## 교정 rule 추가 (Part 2 기반)

| 기존 study 표현 | 정확한 표현 |
|-----|-----|
| "FlashInfer FA2 (CUTLASS)" | **"FlashInfer 자체 CUDA kernel (JIT compiled, FA2 algorithm style)"** |
| `flashinfer::fa2_*_paged_run` | **`BatchPrefillWithPagedKVCacheKernel<...>` / `BatchDecodeWithPagedKVCacheKernel<...>`** |
| "FlashInfer FA2" = FA2 library | **FlashInfer는 FA2/Triton-FA/cuDNN/CUTLASS 어느 것도 호출하지 않음. 자체 구현.** |
| `append_paged_kv_cache` (FlashInfer API) | **SGLang에서는 `sgl_kernel.store_kvcache` 사용 (FlashInfer page API와 다른 경로)** |
