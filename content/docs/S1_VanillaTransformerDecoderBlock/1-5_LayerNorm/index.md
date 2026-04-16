---
title: "1-5. LayerNorm"
weight: 5
---

# 1-5. Layer Normalization

## 개념 / 동기

Deep network의 학습 안정화를 위한 normalization.
BatchNorm과 달리 **단일 샘플의 feature dimension**에 대해 정규화하므로
autoregressive 생성에 적합.

## 수식

### LayerNorm (Original)

$$
\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$

where $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$, $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$

- $\gamma, \beta \in \mathbb{R}^{d}$: learnable scale, shift

### RMSNorm (현대 LLM 표준 — Llama, Mistral 등)

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma
$$

- Mean 계산 제거 ($\beta = 0$, center 안 함) → 연산 절약
- 실측에서 성능 차이 미미하면서 속도 이득

## 연산 분해

| Step | 연산 | 복잡도 |
|------|------|--------|
| Mean (LN only) | Reduction over $d$ | $O(d)$ |
| Variance / RMS | Reduction over $d$ | $O(d)$ |
| Normalize + Scale (+Shift) | Elementwise | $O(d)$ |

$d = 4096$ 기준 매우 작은 연산량이지만, **매 layer에 2번** (Pre-LN) 호출되므로
kernel launch overhead가 중요해짐.

## GPU 커널 매핑

### Fused LayerNorm / RMSNorm Kernel

```
1개의 CUDA kernel에서:
  1. Warp-level reduction으로 mean/variance(또는 RMS) 계산
  2. 같은 kernel에서 normalize + scale 적용
  3. Thread block 하나가 하나의 token(= d차원 벡터)을 처리

- d ≤ 1024: single warp reduction
- d > 1024: multi-warp reduction (shared memory 사용)
```

### Residual + LayerNorm Fusion

```
많은 구현에서 residual add를 LayerNorm kernel에 fuse:
  x' = x + attn_out          ← residual
  ln_out = RMSNorm(x')       ← normalize

→ HBM read/write 1회 절약 (x'를 HBM에 쓰지 않고 register에서 바로 normalize)
SGLang에서: fused_add_rmsnorm kernel
```

## Examples

{{< hint info >}}
TODO: `examples/layernorm_vs_rmsnorm.py` — 수식 검증 및 성능 비교
{{< /hint >}}
