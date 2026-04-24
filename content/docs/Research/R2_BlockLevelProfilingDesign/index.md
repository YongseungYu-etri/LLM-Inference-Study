---
title: "R-2. Decoder Block 수준 Profiling 설계"
weight: 2
---

# R-2. Decoder Block 수준 Workload Profiling — 실험 설계

> 본 문서는 R-1("GPU Kernel 최적화의 현재와 미래")의 4절에서 제안한
> Systematic Profiling Framework를 **구체적 실험 설계**로 발전시킨 것이다.
> R-1이 "무엇을 해야 하는가"를 제시했다면, R-2는 "어떻게 측정할 것인가"를 설계한다.
>
> 작성일: 2026-04-24

---

## 1. 동기: 왜 Decoder Block 수준인가

### 1.1 R-1의 미완성 부분

R-1 4절에서 세 가지 연구 방향을 제안했다:

1. **Systematic Profiling Framework** — GEMM shape 분포, memory access pattern, SM utilization 자동 수집
2. **Roofline Gap Analysis** — 이론적 roofline 대비 실제 달성률의 gap 정량화
3. **Irregular Kernel Design Pattern** — 재사용 가능한 kernel 설계 패턴

그러나 이 제안은 여전히 **개별 kernel 관점**에 머물러 있었다.
"이 GEMM kernel의 roofline 달성률"을 아무리 정밀하게 측정해도,
**decoder block 한 바퀴의 wall-clock time**을 설명하지 못하는 요소들이 존재한다.

### 1.2 Single Kernel Roofline이 놓치는 것

개별 kernel의 roofline 분석은 "이 kernel이 **발사된 동안**" 의 효율만 보여준다.
그러나 실제 inference latency는:

$$T_\text{block} = T_\text{compute} + T_\text{dead} + T_\text{memory\_stall}$$

여기서:
- $T_\text{compute}$: kernel들의 순수 실행 시간 합 (roofline 분석 대상)
- $T_\text{dead}$: kernel 간 빈 시간 (launch overhead, host dispatch, synchronization)
- $T_\text{memory\_stall}$: cache miss로 인한 추가 대기 시간 (L2 miss → HBM 접근)

**$T_\text{dead}$와 $T_\text{memory\_stall}$은 어떤 single kernel의 roofline 위에도 나타나지 않는다.**

### 1.3 프로파일링 관점의 전환

```
기존:  "이 kernel의 arithmetic intensity가 X이고, roofline 대비 Y% 달성"
       → 개별 kernel의 효율은 알 수 있지만, 전체 그림이 안 보임

전환:  "이 decoder block 한 바퀀의 wall time 중,
        실제 GPU가 연산한 시간은 몇 %이고, 나머지는 왜 비는가"
       → 최적화 투자 대비 수익이 가장 큰 곳을 식별
```

이 전환은 프로파일링의 **단위(unit)**를 바꾸는 것이다:
개별 kernel → **decoder block (attn + GEMM + non-linear의 반복 단위)**.


---

## 2. Decoder Block 구조 복습 — Variant별 Kernel 시퀀스

분석 대상을 명확히 하기 위해, S1~S5 각 variant의 decoder block을 kernel 수준으로 나열한다.
(상세는 각 section의 Kernel Summary 참조)

### 2.1 S2 Dense (Llama-3-8B) — 10 kernels/layer

```
[1]  fused_add_rmsnorm          ← residual + RMSNorm
[2]  qkv_gemm                   ← cuBLASLt, [B, 4096] @ [4096, 6144]
[3]  apply_rope                 ← FlashInfer RoPE
[4]  kv_cache_write             ← sgl_kernel.store_kvcache (scatter write)
[5]  attention                  ← FlashInfer GQA (prefill or decode variant)
[6]  output_proj_gemm           ← cuBLASLt, [B, 4096] @ [4096, 4096]
[7]  fused_add_rmsnorm          ← FFN 전
[8]  gate_up_proj_gemm          ← cuBLASLt, [B, 4096] @ [4096, 28672]
[9]  silu_and_mul               ← FlashInfer fused activation
[10] down_proj_gemm             ← cuBLASLt, [B, 14336] @ [14336, 4096]
```

### 2.2 S3 MoE (Mixtral 8x7B) — 13 kernels/layer

```
[1-7]   S2와 동일한 attention path (7 kernels)
[8]     router_gemm              ← cuBLASLt, [T, 4096] @ [4096, 8] (매우 작은 GEMM)
[9]     topk_softmax             ← SGLang custom kernel
[10]    moe_align_block_size     ← dispatch 준비 (expert별 token count 정렬)
[11]    fused_moe_gate_up        ← Triton grouped GEMM (8 experts, top-2 active)
[12]    silu_and_mul             ← activation
[13]    fused_moe_down           ← Triton grouped GEMM + weighted sum
```

### 2.3 S4/S5 MLA + Fine-grained MoE (DeepSeek-V3) — 22 kernels/layer

```
[1]     fused_add_rmsnorm
[2]     q_down_proj              ← [T, d_m] @ [d_m, 1536]
[3]     q_rmsnorm                ← latent space norm
[4]     q_up_proj                ← [T, 1536] @ [1536, 128×192]
[5]     kv_down_proj             ← [T, d_m] @ [d_m, 512+64]
[6]     kv_rmsnorm               ← KV latent norm
[7]     apply_rope               ← decoupled RoPE (q_pe, k_pe만)
[8]     mla_kv_cache_append      ← (c_kv[512], k_pe[64]) 저장
[9]     mla_attention            ← FlashInfer MLA kernel
[10]    output_proj_absorbed     ← absorbed W̃_O projection
[11]    fused_add_rmsnorm
[12]    router_gemm              ← [T, d_m] @ [d_m, 256]
[13]    router_bias_sigmoid      ← loss-free balancing bias
[14]    grouped_topk             ← group-wise top-k (8 groups, top-4 groups, top-8 experts)
[15]    moe_align_block_size     ← dispatch
[16]    fused_moe_gate_up        ← 256 experts grouped GEMM
[17]    silu_and_mul
[18]    fused_moe_down           ← grouped GEMM + weighted sum
[19]    shared_expert_gate_up    ← shared expert GEMM
[20]    shared_expert_silu
[21]    shared_expert_down       ← shared expert GEMM
[22]    add_routed_and_shared    ← elementwise add
```

### 2.4 Kernel 수 비교

| Variant | Kernels/layer | ×Layers | Total launches/step |
|---------|:---:|:---:|:---:|
| S2 Dense (Llama-3-8B) | 10 | 32 | ~323 |
| S3 MoE (Mixtral 8x7B) | 13 | 32 | ~419 |
| S5 V3 (DeepSeek-V3) | 22 | 61 (MoE layers) | ~1,345 |

**관찰**: 아키텍처가 진화할수록 layer당 kernel 수가 2배 이상 증가한다.
이는 개별 kernel의 효율이 동일하더라도, **inter-kernel overhead가 variant별로 크게 달라짐**을 의미한다.


---

## 3. 논점 1 — Inter-Kernel Dead Time

### 3.1 문제 정의

Decoder block 내 연속된 kernel 사이에는 GPU가 idle한 시간이 존재한다.

```
    kernel[i]                    kernel[i+1]
    ┌─────────┐                  ┌─────────┐
    │ compute │                  │ compute │
────┴─────────┴──── gap[i] ─────┴─────────┴────→ time
                    ↑
                    T_dead contribution
```

이 gap은 다음 요소들의 합산이다:

| 요소 | 원인 | 크기 (추정) |
|------|------|:---:|
| CUDA kernel launch overhead | `cudaLaunchKernel` API 호출 | ~3-5 μs |
| Host dispatch | Python → PyTorch → ATen → cuBLASLt 경유 | ~3-8 μs |
| Implicit synchronization | 이전 kernel 완료 대기 (same stream) | 0 (비동기) 또는 variable |
| Driver scheduling | CUDA driver의 kernel 스케줄링 | ~1-2 μs |

**총 추정**: kernel당 ~8-15 μs의 gap.

### 3.2 Decode Phase에서의 심각성

Decode phase에서 이 문제가 특히 심각한 이유는, **kernel 자체가 매우 짧기 때문이다**:

```
S2 Llama-3-8B Decode (B=1):
  Kernel duration range: 3 μs (apply_rope) ~ 140 μs (gate_up_gemm)
  평균: ~33 μs
  
  추정 dead time per layer:
    10 gaps × ~10 μs/gap = ~100 μs
  
  Per-layer wall time: ~365 μs
  Dead time ratio: 100 / 365 ≈ 27%
```

같은 분석을 variant별로 확장하면:

| Variant | Kernels/layer | 추정 dead time/layer | Per-layer wall | Dead time ratio |
|---------|:---:|---:|---:|:---:|
| S2 Llama (B=1) | 10 | ~100 μs | ~365 μs | **~27%** |
| S3 Mixtral (B=1) | 13 | ~130 μs | ~365 μs | **~36%** |
| S5 V3 (B=1, proxy) | 22 | ~220 μs | ~440 μs | **~50%** |

**V3의 경우, decoder block wall time의 절반이 dead time일 수 있다.**
이 수치가 정확한지 실측으로 검증해야 한다.

### 3.3 CUDA Graph의 효과와 한계

CUDA graph는 kernel launch sequence를 사전 capture하여 host dispatch overhead를 제거한다.

```
Without CUDA Graph:
  Python dispatch → kernel launch → Python dispatch → kernel launch → ...
  (host-side overhead 매번 발생)

With CUDA Graph:
  cudaGraphLaunch(graph)
  → 사전 capture된 kernel sequence가 GPU-side에서 연쇄 실행
  → host dispatch overhead 제거
```

**그러나 CUDA graph에도 한계가 있다:**

1. **MoE routing의 dynamic shape**: Expert별 token 수가 batch마다 변동하므로,
   SGLang은 **padded allocation**으로 우회한다 (max expert tokens로 capture).
   이 padding은 불필요한 연산을 포함하므로, **dead time은 줄지만 T_compute가 증가**할 수 있다.

2. **Graph 내부의 implicit barrier**: Same-stream kernel들은 순차 실행이 보장되지만,
   kernel 간 GPU-side scheduling overhead가 여전히 존재할 수 있다.

3. **CUDA graph 미적용 영역**: Prefill phase, variable-length batch의 일부 path 등은
   CUDA graph capture가 어렵거나 비효율적이다.

따라서 실험에서는 **CUDA graph ON/OFF 모두** 측정하여:
- Graph OFF: host dispatch + GPU-side gap의 총합
- Graph ON: GPU-side gap만 (host dispatch 제거된 상태)
- 차이 = host dispatch의 기여분

을 분리해야 한다.

### 3.4 Batch Size에 따른 Dead Time Ratio 변화

Batch size가 커지면 각 kernel의 duration이 길어지므로 (GEMM의 M이 증가),
**고정 크기인 gap의 상대적 비중이 줄어든다**:

```
B=1:   kernel ~30 μs + gap ~10 μs → ratio 25%
B=8:   kernel ~60 μs + gap ~10 μs → ratio 14%
B=32:  kernel ~200 μs + gap ~10 μs → ratio 5%
B=128: kernel ~800 μs + gap ~10 μs → ratio 1%
```

이 전환이 **smooth한지, 아니면 특정 B에서 cliff가 있는지**가 중요하다.
cuBLASLt의 tile heuristic이 M에 따라 불연속적으로 바뀌므로
(예: M=1에서 split-K → M=8에서 standard tile),
cliff 형태의 전환이 있을 수 있다.


---

## 4. 논점 2 — KV Cache Write/Read의 L2 Locality

### 4.1 문제 정의

매 decoder layer에서 KV cache에 대해 두 가지 접근이 발생한다:

```
1. Write: 새 token의 K, V를 cache에 저장 (scatter write to paged buffer)
2. Read: Attention kernel이 전체 KV cache를 읽어서 dot product 수행
```

이 두 접근의 **memory access pattern**이 실제 L2 cache 활용에 어떤 영향을 미치는지가 핵심이다.

### 4.2 Paged KV Cache의 Scatter Write

SGLang/FlashInfer는 paged memory management를 사용한다:
- KV cache가 고정 크기 page들로 분할
- 각 request의 KV가 비연속적인 page들에 분산 저장
- 새 token 추가 시 해당 request의 마지막 page에 append (또는 새 page 할당)

```
store_kvcache (sgl_kernel):
  token_to_kv_pool.set_kv_buffer(layer_id, cache_loc, k, v)
  
  cache_loc은 각 token이 저장될 page 내 slot을 지정.
  → 여러 request가 batch로 들어오면, 각각 다른 page에 scatter write.
  → spatial locality가 request 수에 비례하여 감소.
```

### 4.3 L2 Cache Fit 분석 — Variant별

A100의 L2 cache 크기는 **40 MB**이다.
Context length에 따라 KV cache가 L2에 fit하는지 계산한다:

**S2 Llama-3-8B (Standard Paged KV)**

```
Per-token KV size: n_kv_heads × head_dim × 2 (K+V) × 2B (BF16)
                 = 8 × 128 × 2 × 2 = 4,096 B/token/layer

Context별 per-layer KV cache size:
  512 tokens:   512 × 4,096  =   2 MB  → L2에 ~20 layer분 fit
  2,048 tokens: 2,048 × 4,096 =  8 MB  → L2에 ~5 layer분 fit
  8,192 tokens: 8,192 × 4,096 = 32 MB  → L2에 ~1.25 layer분만 fit
  32K tokens:   32K × 4,096   = 128 MB  → L2에 전혀 안 fit
```

**S4/S5 MLA (DeepSeek-V2/V3)**

```
Per-token KV size: (d_c + d_h^R) × 2B
                 = (512 + 64) × 2 = 1,152 B/token/layer
                   (Llama 대비 3.6x 작음)

Context별 per-layer KV cache size:
  512 tokens:   512 × 1,152  = 0.6 MB  → L2에 ~67 layer분 fit (전체 모델 가능)
  2,048 tokens: 2,048 × 1,152 = 2.4 MB → L2에 ~16 layer분 fit
  8,192 tokens: 8,192 × 1,152 = 9.4 MB → L2에 ~4 layer분 fit
  32K tokens:   32K × 1,152  = 37.5 MB  → L2에 ~1 layer분 fit
```

**핵심 관찰**:
MLA의 KV cache 압축이 **L2 fit 가능한 layer 수를 3~4배 증가**시킨다.
이것이 이론적으로는 알려져 있지만, **실제 L2 hit rate로 정량화한 데이터는 거의 없다.**

### 4.4 Layer 간 L2 Eviction

Decoder block은 32개(또는 61개) layer를 순차 처리한다.
Layer $i$의 attention이 사용한 KV cache가 L2에 남아 있을 때,
Layer $i+1$의 QKV projection GEMM이 weight를 읽으면서 L2를 오염시킬 수 있다.

```
Layer i:   attention(KV_cache_i)     → KV_cache_i가 L2에 load됨
           output_gemm(W_o)          → W_o가 L2에 load됨 → KV_cache_i 일부 evict
           rmsnorm                   → 작은 접근
           gate_up_gemm(W_gate_up)   → W_gate_up이 L2에 load됨 → 추가 evict
           ...
Layer i+1: attention(KV_cache_{i+1}) → KV_cache_{i+1} load 시, KV_cache_i는 이미 evict?
```

**만약 KV cache가 L2에 fit하는 상황이라면**:
- Layer간 weight read가 KV cache를 evict하는 속도가 핵심 변수
- 이 속도는 weight size에 비례 → Dense(Llama)에서 더 심각, MLA에서 덜 심각

**이 가설을 검증하려면**:
같은 모델에서 Layer 0, Layer 16, Layer 31의 attention kernel L2 hit rate를 비교해야 한다.
앞쪽 layer일수록 "아직 다른 layer의 접근으로 오염되지 않은" 상태이므로 hit rate가 높을 것으로 예상.

### 4.5 KV Write Locality의 실제 중요도

위 분석에서 드러나는 중요한 통찰:

```
KV write (store_kvcache)가 L2에 남아서 직후 attention read에서 hit → 이론적으로 가능

그러나 attention kernel은 "새로 쓴 1 token"만 읽는 것이 아니라,
"과거 N-1 token 전체"를 읽는다.

따라서:
  방금 쓴 1 token의 L2 hit 이득:  1 / N (context length에 반비례)
  과거 N-1 token의 HBM read:      (N-1) / N (지배적)
```

**즉, KV write의 L2 locality보다 "전체 KV cache의 L2 fit 여부"가 attention 성능을 지배한다.**

이것이 실험의 초점을 재설정한다:
- **주 측정**: context length별 attention kernel의 전체 L2 hit rate
- **부 측정**: KV write kernel 자체의 L2 write pattern (scatter 정도)
- **핵심 질문**: "어떤 context length까지 L2에 fit하는가" = serving 최적화의 핵심 parameter


---

## 5. Variant별 고유 Profiling Edge

위 두 논점에 더해, 각 variant에서 **특별히 프로파일링이 필요한 지점**을 정리한다.

### 5.1 S2 Dense — Batch Size에 따른 Regime 전환

```
B=1:  GEMM shape (1, d_m) × (d_m, d_m) → GEMV → memory-bound
B=64: GEMM shape (64, d_m) × (d_m, d_m) → regular GEMM → compute-bound

이 전환 지점에서 cuBLASLt의 heuristic이 tile config를 변경:
  Small M: split-K strategy, 작은 tile
  Large M: 128×256 tile, 높은 compute throughput

전환이 smooth한지 cliff인지가 serving의 batch scheduling에 직접 영향.
```

**계측 필요**: Batch size 1→128 sweep에서 per-kernel throughput (TFLOPS) 곡선의 형태.

### 5.2 S3 MoE — Expert Load 분포의 동적 변동

```
Router가 매 batch마다 token→expert 매핑을 결정.
이 매핑의 불균형이 fused_moe_kernel의 실행 시간을 결정.

최악 case: 한 expert에 token 집중
  → 그 expert의 GEMM이 크고, 나머지 SM은 idle
  → padding (CUDA graph용)이 이 불균형을 더 악화

최선 case: token이 expert에 균등 분포
  → grouped GEMM이 balanced → SM utilization 최대
```

**계측 필요**: 다양한 prompt에서 expert별 token count 분포의 variance 수집.
`topk_ids`를 로깅하여 expert load histogram을 생성.

### 5.3 S4 MLA — Projection Chain의 Sequential Overhead

```
S2 QKV: 1 merged GEMM → [B, 4096] @ [4096, 6144]
S4 MLA Q path: W_DQ → RMSNorm → W_UQ → split  (4 sequential kernels)
S4 MLA KV path: W_DKV → split → RMSNorm        (3 sequential kernels)

총 7 kernels vs S2의 1 kernel.
각 kernel이 decode B=1에서 ~10-30 μs이면,
chain 전체: 7 × ~20 μs = ~140 μs (kernel time)
                      + 6 × ~10 μs = ~60 μs (gap time)
                      = ~200 μs

S2의 QKV GEMM 1회: ~30 μs (kernel) + 0 gap = ~30 μs
```

**MLA의 QKV projection이 S2 대비 ~7x wall time을 소비할 수 있다.**
이는 MLA의 KV cache 절감 이득을 일부 상쇄한다.

**계측 필요**: MLA Q/KV projection chain의 실제 wall time과 이론적 compute time의 ratio.

**향후 가능성 (3번 논점, 본 문서에서는 기록만):**
이 chain의 일부를 fuse할 수 있는가?
예: `W_DQ → RMSNorm → W_UQ`를 하나의 fused kernel로 만들면,
intermediate tensor의 HBM write/read를 제거하고 gap 2개를 절감할 수 있다.
이 fuse가 가능하다면, MLA에서 가장 큰 단일 최적화 기회일 수 있다.

### 5.4 S5 V3 — Shared Expert와 Routed Expert의 Overlap

```
현재 SGLang 구현 (sequential):
  routed_out = fused_experts(hidden, w1, w2, topk_weights, topk_ids)  → T_routed
  shared_out = shared_expert_forward(hidden)                          → T_shared
  out = routed_out + shared_out

이론적 가능성 (concurrent, 2 streams):
  stream1: shared_out = shared_expert_forward(hidden)
  stream2: routed_out = fused_experts(hidden, w1, w2, ...)
  sync()
  out = routed_out + shared_out
  → wall time = max(T_routed, T_shared) + sync_overhead

실측 추정 (V2-Lite):
  T_shared ≈ 60 μs, T_routed ≈ 80 μs
  Sequential: 140 μs
  Concurrent: 80 + 20 (sync) = 100 μs → ~29% 이득
```

그러나 현재 구현이 sequential인 이유:
- CUDA graph capture가 multi-stream을 지원하기 어려움
- SM 경합으로 실제 이득이 이론보다 줄어들 수 있음

**계측 필요**: Shared expert와 routed expert의 SM 사용량을 개별 측정하여,
overlap 시 SM 경합이 발생하는지 확인.


---

## 6. 계측 프레임워크 — 3계층 설계

위 논의를 종합하여, LLM inference의 workload profiling을 3개 계층으로 구조화한다.

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Request-Level Profiling                           │
│  ─────────────────────────────────────────────────────────  │
│  대상: Prefill/decode interleave, continuous batching,      │
│        batch composition 변화, P99 tail latency             │
│  도구: Serving framework 내 instrumentation                 │
│  상태: 향후 확장 (본 문서의 범위 밖)                         │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Block-Level Profiling  ★ 본 문서의 핵심           │
│  ─────────────────────────────────────────────────────────  │
│  대상: Decoder block 1회 wall time 분해                     │
│        Inter-kernel dead time, KV cache L2 locality,        │
│        Projection chain overhead, Expert load distribution  │
│  도구: nsys timeline + ncu selective metrics + custom marker │
│  상태: 실험 설계 완료 (아래 7, 8절), 실측 미수행             │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Kernel-Level Profiling                            │
│  ─────────────────────────────────────────────────────────  │
│  대상: 개별 kernel의 roofline 달성률, occupancy              │
│  도구: ncu roofline analysis                                │
│  상태: 기존 연구에서 이미 성숙 (S1-S5 study에서 정리 완료)   │
└─────────────────────────────────────────────────────────────┘
```

**Layer 2가 가장 비어있고, 가장 actionable한 insight를 줄 수 있는 영역이다.**


---

## 7. Experiment 1 — Inter-Kernel Dead Time 정량화

### 7.1 목표

Decoder block 1회(1 layer)의 wall time을 `T_compute`와 `T_dead`로 분해하고,
이를 variant × batch size × CUDA graph on/off로 비교한다.

### 7.2 측정 정의

```
Per decoder block (layer i):
  kernels = [k_1, k_2, ..., k_n]   (시간순 정렬)
  
  T_wall    = k_n.end - k_1.start
  T_compute = Σ (k_j.end - k_j.start)  for j = 1..n
  T_dead    = T_wall - T_compute
  R_dead    = T_dead / T_wall
  
  Per-gap:
  gap[j]    = k_{j+1}.start - k_j.end   for j = 1..n-1
```

### 7.3 도구: nsys → SQLite → Python 분석

**Step 1: nsys profile 수집**

```bash
# CUDA graph OFF
nsys profile \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --output=llama8b_decode_B{batch}_nograph \
  python -m sglang.bench_one_batch \
    --model meta-llama/Meta-Llama-3-8B \
    --batch-size {batch} \
    --input-len 2048 \
    --output-len 32 \
    --disable-cuda-graph

# CUDA graph ON (default)
nsys profile \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --output=llama8b_decode_B{batch}_graph \
  python -m sglang.bench_one_batch \
    --model meta-llama/Meta-Llama-3-8B \
    --batch-size {batch} \
    --input-len 2048 \
    --output-len 32
```

**Step 2: SQLite export**

```bash
nsys export --type=sqlite llama8b_decode_B1_nograph.nsys-rep
```

**Step 3: Python 분석**

```python
import sqlite3
import pandas as pd

def analyze_block_dead_time(db_path):
    conn = sqlite3.connect(db_path)
    
    # CUPTI_ACTIVITY_KIND_KERNEL 테이블에서 kernel timeline 추출
    kernels = pd.read_sql("""
        SELECT start, end, demangledName as name
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        ORDER BY start
    """, conn)
    
    # Decoder block boundary 식별 —
    # fused_add_rmsnorm의 출현 패턴으로 layer 시작점을 찾음
    # (구체적 패턴 매칭은 model variant에 따라 조정 필요)
    
    layer_boundaries = identify_layer_boundaries(kernels)
    
    results = []
    for layer_id, (start_idx, end_idx) in enumerate(layer_boundaries):
        layer_kernels = kernels.iloc[start_idx:end_idx+1]
        
        T_wall = layer_kernels.iloc[-1]['end'] - layer_kernels.iloc[0]['start']
        T_compute = (layer_kernels['end'] - layer_kernels['start']).sum()
        T_dead = T_wall - T_compute
        
        # 개별 gap 기록
        gaps = []
        for i in range(len(layer_kernels) - 1):
            gap_ns = layer_kernels.iloc[i+1]['start'] - layer_kernels.iloc[i]['end']
            gaps.append({
                'from': layer_kernels.iloc[i]['name'],
                'to': layer_kernels.iloc[i+1]['name'],
                'gap_ns': gap_ns
            })
        
        results.append({
            'layer_id': layer_id,
            'T_wall_us': T_wall / 1000,
            'T_compute_us': T_compute / 1000,
            'T_dead_us': T_dead / 1000,
            'R_dead': T_dead / T_wall,
            'n_kernels': len(layer_kernels),
            'gaps': gaps,
            'max_gap_us': max(g['gap_ns'] for g in gaps) / 1000 if gaps else 0,
        })
    
    return pd.DataFrame(results)
```

### 7.4 실험 매트릭스

| 차원 | 값 | 근거 |
|------|-----|------|
| **Model** | Llama-3-8B, Mixtral-8x7B-AWQ, DeepSeek-V2-Lite | S2/S3/S4+S5 variant 대표 |
| **Phase** | Decode only | Dead time은 decode에서 지배적 (prefill kernel은 길어서 ratio 낮음) |
| **Batch size** | 1, 8, 32, 128 | GEMV → regular GEMM 전환점 탐색 |
| **CUDA graph** | ON / OFF | `--disable-cuda-graph` flag |
| **Context length** | 2048 (고정) | Exp 1에서는 KV size가 부차적, 변수 격리 |

**총 실험 수**: 3 models × 4 batch sizes × 2 graph modes = **24 runs**

### 7.5 기대 출력

**Table A: Dead Time Ratio 전체 비교**

```
                  CUDA Graph OFF                    CUDA Graph ON
Model        B=1    B=8    B=32   B=128      B=1    B=8    B=32   B=128
─────────────────────────────────────────────────────────────────────────
Llama-8B     ??%    ??%    ??%    ??%        ??%    ??%    ??%    ??%
Mixtral-AWQ  ??%    ??%    ??%    ??%        ??%    ??%    ??%    ??%
V2-Lite      ??%    ??%    ??%    ??%        ??%    ??%    ??%    ??%
```

**Table B: Variant별 Top-5 Largest Gaps (B=1, Graph OFF)**

```
Rank  From kernel         → To kernel            Gap (μs)   Cause
──────────────────────────────────────────────────────────────────
1     ???                  → ???                  ???        ???
2     ...
```

### 7.6 기대되는 핵심 답변

| 질문 | 기대 형태 |
|------|-----------|
| Dead time이 실제로 27-50%인가? | 정량적 검증 또는 반증 |
| CUDA graph가 dead time을 얼마나 줄이는가? | Graph ON vs OFF 비교로 host dispatch 기여분 분리 |
| Graph ON에서도 남는 dead time은 무엇인가? | GPU-side scheduling overhead, implicit barrier 식별 |
| 어느 batch size에서 dead time이 무의미해지는가? | B sweep curve에서 plateau 지점 |
| Variant별 가장 큰 gap은 어디인가? | Fuse 기회의 직접적 지표 |


---

## 8. Experiment 2 — KV Cache L2 Locality 정량화

### 8.1 목표

Context length별 attention kernel의 L2 cache hit rate를 측정하여,
"KV cache의 L2 fit 여부가 attention 성능에 미치는 영향"을 정량화한다.
특히 Standard Paged KV (S2)와 MLA Paged KV (S4)의 차이를 실측한다.

### 8.2 측정 대상

| Metric | ncu 이름 | 의미 |
|--------|----------|------|
| L2 read hit sectors | `lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum` | L2에서 hit한 sector 수 |
| L2 read miss sectors | `lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum` | L2에서 miss한 sector 수 |
| HBM read bytes | `dram__bytes_read.sum` | 실제 HBM에서 읽은 byte 수 |
| HBM write bytes | `dram__bytes_write.sum` | HBM에 쓴 byte 수 |

```
L2 hit rate = hit_sectors / (hit_sectors + miss_sectors)
```

### 8.3 측정 전략: 2단계 (nsys로 식별 → ncu로 정밀 측정)

ncu는 kernel replay 방식이므로 전수 조사는 비현실적이다.
nsys로 target kernel의 launch 순번을 먼저 식별한 후, ncu로 선택적 측정한다.

**Phase A: nsys로 target kernel launch 순번 확인**

```bash
nsys profile --trace=cuda --output=identify_kernels \
  python -m sglang.bench_one_batch \
    --model meta-llama/Meta-Llama-3-8B \
    --batch-size 1 --input-len {context_len} --output-len 8
```

nsys SQLite에서 `store_kvcache`와 `BatchDecodeWithPaged` (또는 `BatchPrefill`)의
launch index를 추출 → ncu의 `--launch-skip`, `--launch-count` 파라미터로 전달.

**Phase B: ncu로 선택적 L2 metric 수집**

```bash
# Layer 0의 KV write + attention 2개 kernel만 프로파일링
ncu --metrics \
  lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,\
  lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,\
  dram__bytes_read.sum,\
  dram__bytes_write.sum \
  --kernel-name-base demangled \
  --kernel-name "store_kvcache|BatchDecodeWithPaged|BatchPrefillWithPaged|BatchMLAPaged" \
  --launch-skip {layer0_skip} --launch-count 2 \
  python -m sglang.bench_one_batch \
    --model meta-llama/Meta-Llama-3-8B \
    --batch-size 1 --input-len {context_len} --output-len 8

# Layer 16, Layer 31도 동일하게 (launch-skip 값만 변경)
```

### 8.4 실험 매트릭스

| 차원 | 값 | 근거 |
|------|-----|------|
| **Model** | Llama-3-8B, DeepSeek-V2-Lite | Standard KV vs MLA KV |
| **Context length** | 512, 2048, 8192 | L2 fully fit → partial → no fit 전환 |
| **Batch size** | 1, 32 | 단일 요청 vs batch (L2 contention) |
| **Target layer** | Layer 0, Layer 16, Layer 31 | 앞/중/뒤 layer의 L2 eviction 패턴 |

**총 ncu 실험**: 2 models × 3 contexts × 2 batches × 3 layers = **36 kernel pairs**

### 8.5 기대 출력

**Table C: Attention Kernel L2 Hit Rate (B=1)**

```
                Context 512         Context 2048        Context 8192
Model/Layer     hit%   HBM(MB)      hit%   HBM(MB)      hit%   HBM(MB)
──────────────────────────────────────────────────────────────────────
Llama-8B
  Layer 0       ??%    ??           ??%    ??           ??%    ??
  Layer 16      ??%    ??           ??%    ??           ??%    ??
  Layer 31      ??%    ??           ??%    ??           ??%    ??
V2-Lite (MLA)
  Layer 0       ??%    ??           ??%    ??           ??%    ??
  Layer 13      ??%    ??           ??%    ??           ??%    ??
  Layer 26      ??%    ??           ??%    ??           ??%    ??
```

### 8.6 기대되는 핵심 답변

| 질문 | 기대 형태 |
|------|-----------|
| MLA의 KV 압축이 실제 L2 hit rate 향상으로 연결되는가? | Llama vs V2-Lite 동일 context에서의 L2 hit rate 비교 |
| L2 cliff는 어느 context length에서 발생하는가? | Context sweep에서 hit rate가 급락하는 지점 |
| Layer 간 L2 eviction이 실제로 관측되는가? | 같은 context에서 Layer 0 vs Layer 31의 hit rate 차이 |
| Batch size가 L2 contention을 유발하는가? | B=1 vs B=32 비교 |
| "어떤 context length까지 L2에 fit하는가"가 serving parameter로 유효한가? | 이론적 계산과 실측의 일치 여부 |


---

## 9. 통합 분석 계획

Experiment 1과 2의 결과를 결합하면 decoder block의 wall time을 3요소로 분해할 수 있다:

$$T_\text{block} = T_\text{compute} + T_\text{dead} + T_\text{memory\_stall}$$

| 요소 | 측정 출처 | 최적화 방향 |
|------|-----------|-------------|
| $T_\text{compute}$ | Exp 1의 `T_compute` | Kernel 내부 최적화 (Layer 1, 기존 연구) |
| $T_\text{dead}$ | Exp 1의 `T_dead` | Kernel fusion, CUDA graph, launch 최적화 |
| $T_\text{memory\_stall}$ | Exp 2의 L2 miss → HBM latency로 간접 추정 | KV cache 압축 (MLA), page size tuning |

이 분해를 variant별로 비교하면, **각 variant에서 최적화 투자 대비 수익이 가장 큰 곳**이 명확해진다:

```
Dense (S2): T_dead이 지배적이면 → kernel fusion이 답
MoE (S3):   T_dead + expert load variance가 크면 → dispatch 최적화가 답
MLA (S4/S5): T_memory_stall이 작으면 → MLA의 L2 이득 실증 (알고리즘 논문 보완)
```


---

## 10. 실행 계획

### Phase 1: 분석 도구 준비 (랩탑에서 수행 가능)

| 단계 | 내용 | 산출물 |
|------|------|--------|
| 1a | nsys SQLite 파싱 스크립트 작성 | `analyze_block_dead_time.py` |
| 1b | Layer boundary 패턴 매칭 로직 (variant별) | 위 스크립트에 포함 |
| 1c | ncu metric 추출 스크립트 작성 | `analyze_l2_locality.py` |
| 1d | 결과 시각화 코드 (matplotlib/plotly) | `plot_results.py` |

### Phase 2: Experiment 1 수행 (A100 서버 필요)

| 단계 | 내용 | 소요 시간 추정 |
|------|------|:---:|
| 2a | Llama-3-8B 모델 로드 확인 | ~5분 |
| 2b | nsys 24 runs (3 models × 4 batch × 2 graph) | 각 ~3분 → ~72분 |
| 2c | SQLite export | ~10분 |
| 2d | 분석 스크립트 실행 | ~5분 |

### Phase 3: Experiment 2 수행 (A100 서버 필요)

| 단계 | 내용 | 소요 시간 추정 |
|------|------|:---:|
| 3a | Phase A: nsys로 launch 순번 식별 | 6 runs × ~3분 = ~18분 |
| 3b | Phase B: ncu 36 kernel pairs | 각 ~5-10분 → ~4-6시간 |
| 3c | 분석 스크립트 실행 | ~10분 |

> **주의**: ncu는 kernel replay 방식이므로 매우 느리다.
> Phase 3b의 소요 시간이 가장 길다.
> 필요 시 가장 흥미로운 조건만 선별하여 실험 수를 줄일 수 있다
> (예: Llama-8B + V2-Lite만, context 2048만 → 12 kernel pairs → ~1.5시간).

### Phase 4: 통합 분석 및 문서화

| 단계 | 내용 | 산출물 |
|------|------|--------|
| 4a | Exp 1 + Exp 2 결과 통합 | $T_\text{compute}$ / $T_\text{dead}$ / $T_\text{memory\_stall}$ 분해 표 |
| 4b | Variant별 최적화 기회 순위 | 정량적 근거 기반 |
| 4c | R-2 문서에 실측 결과 추가 | 본 문서 업데이트 |
| 4d | 필요 시 R-3 (실측 결과 + 해석) 분리 작성 | |


---

## 11. 향후 확장 — 기록해 둘 논점

본 문서에서 설계한 실험으로 답변할 수 없지만, 결과에 따라 후속으로 탐구할 수 있는 논점들:

### 11.1 MLA Projection Chain Fusion 가능성

MLA의 `W_DQ → RMSNorm → W_UQ` chain을 하나의 fused kernel로 만들 수 있는가?
- Intermediate tensor의 HBM write/read 제거
- Inter-kernel gap 2개 절감
- RMSNorm이 중간에 있어 단순 GEMM fusion은 불가 → custom kernel 필요
- Exp 1에서 이 chain의 dead time 비중이 크다면, 투자 가치가 높을 것

### 11.2 Irregular Workload 간의 상호작용

Chunked prefill + MoE + continuous batching이 겹치면:
- Grouped GEMM의 expert별 token 분포가 chunk boundary에 의해 추가 왜곡
- Prefill chunk와 decode batch가 같은 layer를 통과할 때 KV cache 접근 패턴 간섭
- 이 상호작용은 Layer 3 (Request-Level) profiling에서 다뤄야 할 주제

### 11.3 Compiler/Autotuner의 한계선

MoE의 dynamic routing은 compile-time에 shape을 알 수 없으므로,
Triton/XLA의 JIT compilation이 근본적 한계를 가진다.
- 런타임 shape에 따른 adaptive tile 선택이 가능한가?
- cuBLASLt의 heuristic과 비교했을 때 Triton의 autotune overhead는?
- 이 문제는 Exp 1의 MoE variant 결과 (CUDA graph padding overhead) 분석 후 구체화


---

## 12. 관련 Cross-References

| 참조 | 관련 내용 |
|------|-----------|
| [R-1 GPU Kernel 최적화의 현재와 미래]({{< relref "R1_KernelOptimizationLandscape" >}}) | 본 문서의 직접적 선행 논의 (4절의 구체화) |
| [1-7 Prefill vs Decode]({{< relref "docs/S1_VanillaTransformerDecoderBlock/1-7_PrefillVsDecode" >}}) | Roofline model, compute/memory bound 분석 |
| [1-8 Dense GEMM Call Path]({{< relref "docs/S1_VanillaTransformerDecoderBlock/1-8_DenseGEMMCallPath" >}}) | Decoder block 전체 구조, kernel 수, KV write/read 비대칭 |
| [2-5 Kernel Summary]({{< relref "docs/S2_DenseLLM/2-5_KernelSummary" >}}) | S2 Dense timing breakdown |
| [3-5 Kernel Summary]({{< relref "docs/S3_MoE/3-5_KernelSummary" >}}) | S3 MoE timing breakdown |
| [5-7 Full Summary]({{< relref "docs/S5_ParallelMoE/5-7_FullSummary" >}}) | S5 V3 timing breakdown |
| [5-4 Parallel MoE]({{< relref "docs/S5_ParallelMoE/5-4_ParallelMoE" >}}) | CUDA graph 한계, shared/routed expert overlap |
