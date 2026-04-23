---
title: "R-1. GPU Kernel 최적화의 현재와 미래"
weight: 1
---

# R-1. GPU Kernel 최적화의 현재와 미래: Regular에서 Irregular Workload로의 전환

> 본 문서는 S1~S5 학습 과정에서 도출된 고찰을 정리한 것이다.
> LLM inference의 GPU kernel 최적화가 어떤 전환점에 있으며,
> 향후 kernel 전문성이 어디에서 차별적 가치를 가질 수 있는지를 논의한다.

---

## 1. 배경: Regular Workload Kernel 최적화의 성숙

### Vendor Library의 수렴

Regular workload — 큰 M×N×K의 dense GEMM, 고정 길이 batch, 표준 decoder block 연산 — 에 대한
GPU kernel 최적화는 이미 hardware theoretical peak의 85~95%에 도달한 상태이다.

- **cuBLAS/cuBLASLt**: Dense GEMM의 tile config, split-K, warp scheduling이 자동 최적화
- **CUTLASS 3.x**: Tensor Core (HMMA m16n8k16 등) 활용의 template 기반 코드 생성
- **cuDNN**: Fused attention (FlashAttention 계열)이 라이브러리로 흡수
- **FlashInfer**: Paged KV cache 기반 fused attention의 사실상 표준

이 영역에서 register 배치, shared memory bank conflict 해소, warp-level 스케줄링 등의
수작업 tuning으로 얻는 추가 이득은 1~3% 수준이며, 다음 라이브러리 버전에 흡수된다.

### FlashAttention의 사례

FlashAttention은 단순한 "kernel tuning"이 아니라 **memory hierarchy를 고려한 알고리즘 재설계**였다.
Q를 tiling하고 online softmax를 통해 $S \times S$ score matrix의 HBM materialization을 회피하는 접근은
커널 수준의 혁신이었지만, FlashInfer/cuDNN에 흡수되면서 "fused attention kernel을 직접 작성하는 것"의
차별적 가치는 급격히 줄어들었다.

### 연구 무게중심의 이동

결과적으로 연구의 무게중심이 이동했다:

```
"커널을 빠르게 하자" (kernel-level tuning)
  → "커널이 해야 할 일을 줄이자" (algorithmic reduction)
```

MLA(KV cache 차원 압축), NSA(sparse attention), KV cache pruning/quantization 등이
모두 이 방향에 해당한다. 이들은 GPU 커널의 실행 효율이 아니라,
**커널에 전달되는 workload 자체의 크기를 줄이는 것**을 목표로 한다.


---

## 2. 새로운 알고리즘이 새로운 Irregular Workload를 만든다

### 순환 구조

Kernel 최적화의 대상이 소멸한 것이 아니라, **공략 대상이 이동**하고 있다.
새로운 알고리즘들이 기존 vendor library가 최적화하지 않은 irregular workload를 끊임없이 생성한다:

```
새 알고리즘 등장
  → Irregular workload 발생
    → Kernel 최적화 필요
      → 성숙
        → Vendor library 흡수
          → 새 알고리즘 등장 (반복)
```

### 구체적 사례

| 알고리즘 혁신 | 발생하는 Irregular Kernel 문제 | 현재 상태 (2025 기준) |
|---|---|---|
| **MoE (Mixtral, DeepSeek)** | Grouped GEMM — expert별 token 수 불균일, ragged batch | CUTLASS 3.x에 grouped GEMM 추가되었으나, dynamic load balancing은 연구 진행 중 |
| **Speculative Decoding** | Variable-length verification batch, tree attention | 표준화된 kernel 없음, 각 serving framework이 자체 구현 |
| **NSA (Sparse Attention)** | Block-sparse score 계산 + top-k selection + multi-path merge | 완전히 새로운 kernel 필요, DeepSeek 자체 구현 |
| **Continuous Batching** | Request 단위 가변 길이 → ragged tensor GEMM | FlashInfer의 ragged prefill이 대응 중이나 진행 중 |
| **KV Cache Quantization** | Mixed-precision dequant + attention fused (2-3bit) | KIVI 수준의 2-bit은 mature kernel 부재 |
| **MLA Absorption** | 비표준 GEMM shape (absorbed Q projection) | FlashInfer에 MLA 전용 kernel 추가, 최적화 진행 중 |

이 중 어느 것도 "cuBLAS를 호출하면 끝"이 되지 않는다.


---

## 3. Workload Characterization의 중요성

### Production vs Benchmark의 괴리

대부분의 kernel 최적화는 **고정 batch, 고정 seq_len, single model** 기준으로 벤치마킹된다.
그러나 실제 serving 환경에서는:

- Request별 input/output 길이가 10x 이상 차이
- Prefill과 decode가 동시에 GPU를 공유 (chunked prefill)
- KV cache 크기가 request마다 상이
- MoE에서 expert load가 batch마다 변동
- "평균 throughput"이 아닌 **P99 latency**가 SLO의 핵심

이 **tail latency를 유발하는 irregular case**를 식별하는 것이 workload characterization의 핵심이며,
여기서 kernel 수준의 profiling/분석 전문성이 차별화된다.

### 알고리즘 연구자가 놓치는 Gap

알고리즘 논문은 이론적 연산량/메모리 절감을 보고하지만, 실제 kernel 실행에서는 gap이 존재한다:

**예시: MLA**
- 논문: "KV cache byte가 5~13x 줄었다"
- 실제: Absorption으로 변형된 Q projection의 GEMM shape이 cuBLAS의 sweet spot에서 벗어남
- 실제: Latent vector 차원($d_c$)이 tensor core tile size($16 \times 16$)에 align되지 않을 수 있음
- 결과: 이론적 절감량이 실제 속도 향상으로 1:1 전환되지 않음

**예시: MoE Grouped GEMM**
- 논문: "Active parameter가 1/8로 줄었다"
- 실제: Expert 간 token 수 불균형으로 GPU SM utilization이 50% 이하로 떨어지는 case 존재
- 실제: Small expert GEMM이 tensor core의 minimum tile보다 작을 수 있음

**예시: Speculative Decoding**
- 논문: "Draft model로 k개 token을 한 번에 검증"
- 실제: Acceptance rate에 따라 verification batch 크기가 매 step 변동
- 실제: Tree attention의 irregular mask pattern이 표준 causal mask 최적화와 맞지 않음

이 **"이론적 연산량 절감"과 "실제 kernel 실행 시간 절감" 사이의 gap**을 메우는 것이
kernel 전문가의 핵심 역할이며, 이 gap은 irregular workload에서 더 크다.


---

## 4. Kernel 전문성의 가치 변화

### 과거: Regular Kernel Tuning

```
대상: Dense GEMM, standard attention
방법: Register tuning, shared memory 최적화, warp scheduling
결과: Vendor library에 흡수 → 차별적 가치 소멸
```

### 현재: Irregular Workload의 Bridge 역할

```
대상: MoE, sparse attention, variable-length batching, mixed-precision KV
방법: Workload profiling → bottleneck 진단 → algorithm-aware kernel 설계
가치: 알고리즘 혁신과 실제 GPU 실행 사이의 gap을 bridge
      → 현재 가장 희소한 역량
```

### 미래: Compiler/Autotuner의 확장

```
Triton, XLA 등의 compiler와 autotuner가 irregular case도 점진적으로 흡수할 가능성
그러나 "무엇을 fuse하고 무엇을 분리할지", "어떤 memory hierarchy를 활용할지"의
설계 결정(design decision)은 여전히 사람의 architectural insight에 의존
```

### 연구 방향 제언

특정 커널을 수작업으로 tuning하는 방향보다는,
**새 알고리즘이 만드는 irregular case를 체계적으로 characterize하고,
그 bottleneck을 kernel 수준에서 해소하는 methodology**를 구축하는 것이
더 차별화된 기여가 될 수 있다.

구체적으로:

1. **Systematic Profiling Framework**: 새 알고리즘이 생성하는 실제 GEMM shape 분포,
   memory access pattern, SM utilization을 자동으로 수집·분석하는 체계
2. **Roofline Gap Analysis**: 이론적 roofline 대비 실제 달성률의 gap을 알고리즘별로 정량화하고,
   gap의 원인(tile misalignment, load imbalance, synchronization overhead 등)을 분류
3. **Irregular Kernel Design Pattern**: MoE grouped GEMM, sparse attention, variable-length batch 등
   반복적으로 등장하는 irregular pattern에 대한 재사용 가능한 kernel 설계 패턴 정립


---

## 5. 관련 연구 흐름 — MLA 이후 KV Cache 최적화 (2024~2025)

MLA가 KV entry의 **차원(width)**을 줄인 이후, 나머지 축을 공략하는 연구들이 활발하다:

| 축 | 의미 | 대표 기법 | 연도 |
|---|---|---|---|
| **Width** | Token당 KV byte 수 축소 | MLA (DeepSeek-V2) | 2024 |
| **Depth** | Layer간 KV 공유 | CLA, YOCO, xKV | 2024~2025 |
| **Precision** | 수치 정밀도 축소 | KIVI (2-bit), KVQuant (3-bit) | 2024 |
| **Length** | 비중요 token 제거 | H2O, SnapKV, PyramidKV | 2023~2024 |
| **Sparsity** | 선택적 KV read | NSA (DeepSeek) | 2025 |

이들은 서로 직교적이며 조합 가능하다 (예: xKV가 MLA 위에서 추가 압축 달성을 검증).

이 각각의 알고리즘적 혁신이 **새로운 irregular kernel workload를 발생시킨다**는 점에서,
위 4절의 순환 구조가 계속 적용된다.


---

## 6. 결론

"Regular workload의 GPU kernel 최적화는 red ocean"이라는 진단은 타당하다.
그러나 **kernel 전문성 자체**가 red ocean인 것은 아니다.

알고리즘 혁신이 가속될수록 irregular workload는 더 빈번하게, 더 다양한 형태로 발생하며,
이를 **체계적으로 characterize하고 실제 hardware에서의 gap을 bridge하는 역량**은
오히려 더 희소해지고 있다.

핵심 전환:

```
"이 커널을 어떻게 빠르게 만들까" (diminishing returns)
  → "이 알고리즘이 실제 GPU에서 어디서 stall하는가" (growing demand)
```
