---
title: "4-2. MLA Math"
weight: 2
---

# 4-2. MLA Math — Down/Up Projection 상세

## 기본 설정

DeepSeek-V2/V3의 MLA 차원:

| Symbol | Meaning | V2 | V3 |
|--------|---------|:---:|:---:|
| $d_m$ | Hidden dim | 5120 | 7168 |
| $n_h$ | # Q heads | 128 | 128 |
| $d_h$ | per-head dim (no RoPE) | 128 | 128 |
| $d_h^R$ | per-head dim (with RoPE) | 64 | 64 |
| $d_c$ | KV compression dim | 512 | 512 |
| $d_c'$ | Q compression dim | 1536 | 1536 |

## 수식 — Pure MLA (RoPE 무시한 간소화)

우선 RoPE를 빼고 핵심 구조만.

### Query 경로 (V2는 Q도 부분 압축)

$$
\mathbf{c}^Q = \mathbf{x} \mathbf{W}^{DQ}, \quad \mathbf{W}^{DQ} \in \mathbb{R}^{d_m \times d_c'}
$$
$$
\mathbf{q}_i = \text{RMSNorm}(\mathbf{c}^Q) \mathbf{W}^{UQ}_i, \quad \mathbf{W}^{UQ}_i \in \mathbb{R}^{d_c' \times d_h}
$$

- Q도 먼저 $d_c' = 1536$으로 down, 그 후 per-head로 up
- Q는 cache 안 함이라 메모리 절감은 아니고, 파라미터 절감 효과

### KV 경로 (핵심)

$$
\mathbf{c}^{KV} = \mathbf{x} \mathbf{W}^{DKV}, \quad \mathbf{W}^{DKV} \in \mathbb{R}^{d_m \times d_c}
$$

**$c^{KV}$ (512-dim)만 cache에 저장**. per-head K, V는 on-demand:
$$
\mathbf{k}_i = \text{RMSNorm}(\mathbf{c}^{KV}) \mathbf{W}^{UK}_i, \quad \mathbf{W}^{UK}_i \in \mathbb{R}^{d_c \times d_h}
$$
$$
\mathbf{v}_i = \text{RMSNorm}(\mathbf{c}^{KV}) \mathbf{W}^{UV}_i, \quad \mathbf{W}^{UV}_i \in \mathbb{R}^{d_c \times d_h}
$$

### Attention

$$
\text{attn}_i = \text{softmax}\!\left(\frac{\mathbf{q}_i \mathbf{k}_i^\top}{\sqrt{d_h}}\right) \mathbf{v}_i
$$

## 메모리 분석

### Standard MHA 기준 (DeepSeek-V2 scale로)

```
Per-token cache (standard MHA style, n_h=128, d_h=128):
  2 × 128 × 128 × 2B = 64 KB per token per layer   ← 엄청남
  60 layers, 4K tokens = 15 GB per request         ← 불가능
```

### GQA 시도 (n_kv=16)

```
2 × 16 × 128 × 2B = 8 KB per token per layer
60 layers, 4K tokens = 2 GB per request            ← 가능하지만 여전히 큰 편
```

### MLA

```
c_KV: d_c × 2B = 512 × 2 = 1024 bytes             ← 단일 벡터 저장
(RoPE 부분은 4-4에서 추가로 포함: +64 × 2B = 128 bytes)
Per token per layer: ~1.15 KB
60 layers, 4K tokens = ~280 MB per request         ← 7x 절감!
```

**Batch 10명 = 2.8 GB** — A100 80GB에서 아주 편함.

## FLOPs 분석

MLA는 compute에서 **더 많은** 연산이 필요.

```
MHA (per token):
  Q: [d_m × n_h d_h] weight 
  K,V: [d_m × 2 × n_h d_h] weight 
  O: [n_h d_h × d_m]

MLA (per token):
  W^DQ: [d_m × d_c'] weight                                   ← Q down
  W^UQ: [d_c' × n_h d_h] weight                               ← Q up
  W^DKV: [d_m × d_c] weight                                   ← KV down
  W^UK: [d_c × n_h d_h] weight                                ← K up (on-demand)
  W^UV: [d_c × n_h d_h] weight                                ← V up (on-demand)
```

실제 측정하면 compute는 MHA 대비 비슷하거나 소폭 증가 (10-20%).
Decode에서는 이 compute가 memory-bound wasted cycle을 활용 → net win.

## 그러나 순진한 구현은 문제

Decode 시 매 step:
1. Cache에서 $c^{KV}$ 읽기 (작음 — 좋음)
2. 모든 과거 토큰에 대해 $K_i = c^{KV} \mathbf{W}^{UK}_i$, $V_i$ 계산 (비쌈!)
3. Attention

매 step마다 전체 context 길이 × n_h × d_h 크기의 K,V를 재계산 → 오히려 느려짐.

→ **Matrix Absorption** 트릭으로 해결 (4-3).

## 왜 Down/Up이 학습 가능한가 — 직관

$d_c = 512 \ll n_h \times d_h = 128 \times 128 = 16384$이므로
원칙적으로 K, V의 정보가 "압축 가능"해야 함.

경험적 근거:
- Transformer 중간 activation은 낮은 effective rank를 가진다 (연구 다수)
- $K, V$는 서로 correlated (같은 $x$에서 출발)
- Attention score가 dominant한 몇 개의 pattern에 집중

이 구조를 **학습 가능한 projection으로 명시화**한 것이 MLA.

## 전체 파라미터 Count (DeepSeek-V2)

per layer attention params:
```
W^DQ:   5120 × 1536 = 7.86M
W^UQ:   1536 × 128 × 128 = 25.1M  (여기서 128*128 = n_h * (d_h + d_h^R) 실제로는 192)
W^DKV:  5120 × 576 = 2.95M  (512 + 64 RoPE part)
W^UK:   512 × 128 × 128 = 8.39M
W^UV:   512 × 128 × 128 = 8.39M
W^O:    128 × 128 × 5120 = 83.9M
Total per layer: ~136M

60 layers: ~8.2B attention params
```

## 다음 — Matrix Absorption

순진한 MLA는 decode에서 느리다.
4-3에서 *실제로는 K, V를 복원하지 않고도* attention을 계산하는 방법을 본다.
