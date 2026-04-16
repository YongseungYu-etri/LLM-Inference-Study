---
title: "4-3. Matrix Absorption Trick"
weight: 3
---

# 4-3. Matrix Absorption — MLA의 핵심 성능 트릭

## 문제 Recap

MLA의 문제점: decode 시 매 step마다 전체 context의 $K, V$를 $c^{KV}$에서 복원해야 함.
$$
\mathbf{k}_i = \mathbf{c}^{KV} \mathbf{W}^{UK}_i, \quad \mathbf{v}_i = \mathbf{c}^{KV} \mathbf{W}^{UV}_i
$$

이를 naive하게 수행하면 $K, V$ 복원에 drop-in time이 소비되어 이득이 사라짐.

## 해결 — 수학적 rearrangement

Attention score 식:
$$
\mathbf{q}_i \mathbf{k}_i^\top = \mathbf{q}_i (\mathbf{c}^{KV} \mathbf{W}^{UK}_i)^\top = \mathbf{q}_i (\mathbf{W}^{UK}_i)^\top (\mathbf{c}^{KV})^\top
$$

정의:
$$
\mathbf{q}_i' := \mathbf{q}_i (\mathbf{W}^{UK}_i)^\top \in \mathbb{R}^{d_c}
$$

그러면:
$$
\mathbf{q}_i \mathbf{k}_i^\top = \mathbf{q}_i' (\mathbf{c}^{KV})^\top
$$

**핵심**: $\mathbf{q}_i'$는 **현재 토큰의 Q만으로** 계산 가능 (과거 토큰 개수와 무관).
과거 토큰들은 $c^{KV}$ 하나씩만 있으면 되고, $K$를 복원할 필요 없음.

### V 쪽도 동일

$$
\text{output}_i = \text{softmax}(...)_j \mathbf{v}_{i,j} = \text{softmax}(...)_j \mathbf{c}^{KV}_j \mathbf{W}^{UV}_i
$$

Softmax attention 결과에 $\mathbf{W}^{UV}_i$를 나중에 곱해도 같음 (linear):
$$
\text{output}_i = \left(\sum_j \text{softmax}(...)_j \mathbf{c}^{KV}_j \right) \mathbf{W}^{UV}_i
$$

즉, attention을 $c^{KV}$ 위에서 바로 계산하고, **마지막에** $\mathbf{W}^{UV}_i$를 곱함.

## Absorbed Attention — 새로운 수식

결합하면:

### Query absorption
$$
\mathbf{q}_i' = \mathbf{q}_i (\mathbf{W}^{UK}_i)^\top = \mathbf{x} \mathbf{W}^{UQ}_i (\mathbf{W}^{UK}_i)^\top = \mathbf{x} \tilde{\mathbf{W}}^{Q}_i
$$

여기서 $\tilde{\mathbf{W}}^{Q}_i = \mathbf{W}^{UQ}_i (\mathbf{W}^{UK}_i)^\top \in \mathbb{R}^{d_c' \times d_c}$.

### Output absorption
$$
\text{final\_out}_i = \text{attn\_out}_i \cdot \mathbf{W}^{UV}_i = \text{attn\_out}_i \cdot \mathbf{W}^{UV}_i
$$

이걸 $\mathbf{W}^O$와 합치면:
$$
\text{final\_output} = \sum_i \text{attn\_out}_i \cdot \mathbf{W}^{UV}_i \cdot \mathbf{W}^O_i = \sum_i \text{attn\_out}_i \cdot \tilde{\mathbf{W}}^O_i
$$

### 결과: 새 attention 경로

```
1. x → q'_i = x · W̃_Q_i          (각 head i, per-token)
       shape: d_c (압축 차원!)

2. x → c_KV = x · W_DKV            (현재 토큰의 KV down)
       c_KV를 cache에 append

3. Attention on compressed dim:
   attn_out_i = softmax(q'_i · c_KV^T / scale) · c_KV

4. Final projection:
   output = Σ_i attn_out_i · W̃_O_i
```

### 요점

- **K, V를 per-head로 복원하지 않음** — 모든 attention 연산이 $d_c$ 차원에서 일어남
- Cache에서 읽는 건 $c^{KV}$뿐 (작음)
- **Attention FLOPs는 MHA보다 오히려 적을 수 있음** ($d_c < n_h \times d_h$일 때)

## 새로운 GPU 커널 요구사항

일반 Flash Attention은 $q_i \in \mathbb{R}^{d_h}$, $k_j \in \mathbb{R}^{d_h}$ 입력을 가정.
MLA absorbed form은:
- $q'_i \in \mathbb{R}^{d_c}$ (예: 512)
- $c^{KV}_j \in \mathbb{R}^{d_c}$
- 모든 head $i$가 **같은** $c^{KV}$를 공유

→ 기존 FA 커널을 그대로 못 씀. **MLA 전용 attention kernel**이 필요.
이것이 FlashInfer의 `BatchPrefillWithPagedMLAKVCacheWrapper` 등 (4-5에서).

## 구체적 숫자 — DeepSeek-V2 Absorbed Attention

```
d_c = 512 (+ 64 RoPE)
d_c' = 1536

Per token compute (Q):
  x · W̃_Q_i for i in 128 heads: 5120 × 512 × 128 GEMM = 335 MFLOPs

Per token compute (KV):
  x · W_DKV: 5120 × 576 = 2.95M params, FLOPs = 5.9M

Attention (absorbed, per head):
  q'_i (512) · c_KV (seq × 512)^T: seq × 512 × 2 FLOPs per head
  128 heads × seq × 1024 FLOPs

Output projection (absorbed):
  attn_out (seq × 512) · W̃_O_i → (seq × d_m)

→ Total FLOPs comparable to standard MHA, but memory 7x less
```

## Absorption의 Trade-off

### 장점
- KV cache 크기 dramatic 감소
- Decode memory-bound bottleneck 완화
- 긴 context 지원 용이

### 단점
- Absorption 후 weight matrix가 커짐: $\tilde{\mathbf{W}}^Q \in \mathbb{R}^{d_c' \times d_c}$
  - 원래 $\mathbf{W}^{UQ}$ ($d_c' \times d_h$) 보다 $d_c/d_h$배 큼 → weight memory ↑
  - 4배 증가 (512/128) 정도
- 새로운 attention kernel 필요 (FlashInfer MLA variants)
- Training과 inference의 수식이 달라 debugging 어려움

## 흥미로운 비교 — MLA vs MQA

```
MQA: n_kv = 1 → K,V 1 head만 저장
     per-token KV: 2 × 1 × 128 × 2B = 512 B
     품질 저하 (경험적)

MLA absorbed: c_KV 저장, per-token KV = 512 × 2B = 1024 B
             품질 유지, overhead는 absorbed projection
```

공간적으로 MQA는 MLA와 비슷하지만,
MLA는 학습 가능한 projection이 있어 **품질 손실 없이** 같은 압축을 달성.

## 실제 효과 측정 (DeepSeek-V2 논문)

```
Long context (128K) benchmark:
  Llama-3-GQA model: needle-in-haystack 정답률 급락 (KV cache OOM)
  DeepSeek-V2 MLA:   128K 전체에서 90%+ 유지

A100 80GB single GPU:
  Llama-3-70B GQA:   실질적 batch=1 at 4K context
  DeepSeek-V2 MLA:   batch=10+ at 4K context
```

## 다음 — RoPE가 이 우아한 그림을 깨뜨린다

문제: RoPE는 $K$에 position-dependent 회전을 적용.
하지만 absorbed form에서는 K를 명시적으로 복원하지 않으므로 RoPE 적용 위치가 애매.

→ DeepSeek이 도입한 **Decoupled RoPE**로 해결 (4-4).
