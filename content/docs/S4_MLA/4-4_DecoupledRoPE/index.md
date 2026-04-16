---
title: "4-4. Decoupled RoPE"
weight: 4
---

# 4-4. Decoupled RoPE — MLA와 RoPE의 충돌 해결

## 문제 정의

MLA absorption은 다음 식에 의존:
$$
\mathbf{q}_i \mathbf{k}_j^\top = \mathbf{q}_i (\mathbf{c}^{KV}_j \mathbf{W}^{UK}_i)^\top = (\mathbf{q}_i (\mathbf{W}^{UK}_i)^\top) (\mathbf{c}^{KV}_j)^\top
$$

이게 성립하려면 **$\mathbf{W}^{UK}_i$가 $\mathbf{q}_i \mathbf{k}_j^\top$ 계산 중 constant**여야 함.

### RoPE가 개입하면?

RoPE는 position $m$에 따라 Q와 K에 회전:
$$
\mathbf{k}_j^{RoPE} = \mathbf{R}_j \mathbf{k}_j = \mathbf{R}_j (\mathbf{c}^{KV}_j \mathbf{W}^{UK}_i)
$$

이제:
$$
\mathbf{q}_i^{RoPE} \cdot (\mathbf{k}_j^{RoPE})^\top = \mathbf{R}_i \mathbf{q}_i \cdot \mathbf{R}_j \mathbf{c}^{KV}_j \mathbf{W}^{UK}_i
$$

$\mathbf{R}_i, \mathbf{R}_j$가 position별로 다르므로 **절대 $\mathbf{W}^{UK}_i$를 $\mathbf{W}^{UQ}$에 흡수할 수 없음**.
→ Absorption trick 붕괴, 원래 속도 이득 소실.

## Decoupled RoPE — 해결책

아이디어: **RoPE는 별도의 작은 sub-vector에만 적용**.
Compressed part는 RoPE 없이 그대로, RoPE part는 별도 channel로.

### 수식

각 head를 두 부분으로 분할:
```
q_i = [q_i^C (no RoPE, d_h=128) ; q_i^R (with RoPE, d_h^R=64)]
k_j = [k_j^C (no RoPE, d_h=128) ; k_j^R (with RoPE, d_h^R=64)]
```

전체 head dimension: $d_h + d_h^R = 128 + 64 = 192$.

### Compressed part (absorption 적용 가능)

$$
\mathbf{q}_i^C = \mathbf{x} \mathbf{W}^{UQ,C}_i, \quad
\mathbf{k}_j^C = \mathbf{c}^{KV}_j \mathbf{W}^{UK,C}_i
$$

여기서 absorption:
$$
\mathbf{q}_i^C \cdot (\mathbf{k}_j^C)^\top = (\mathbf{q}_i^C \mathbf{W}^{UK,C\top}_i) \cdot (\mathbf{c}^{KV}_j)^\top
$$

→ 기존 대로 흡수 가능.

### RoPE part (별도 처리)

$$
\mathbf{q}_i^R = \text{RoPE}_{pos_i}(\mathbf{x} \mathbf{W}^{QR}_i)
$$

K의 RoPE part는 **head마다 다르지 않고 shared** (중요):
$$
\mathbf{k}_j^R = \text{RoPE}_{pos_j}(\mathbf{x} \mathbf{W}^{KR})
$$

- $\mathbf{W}^{KR} \in \mathbb{R}^{d_m \times d_h^R}$ — 단일 공유 matrix (per-head 아님)
- → $\mathbf{k}^R$은 **모든 head가 공유** (1 head의 $d_h^R$ 크기)
- Cache에 $\mathbf{k}^R$ ($d_h^R = 64$)과 $\mathbf{c}^{KV}$ ($d_c = 512$) 저장

### Attention 결합

$$
\text{score}_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_h + d_h^R}} = \frac{\mathbf{q}_i^C (\mathbf{k}_j^C)^\top + \mathbf{q}_i^R (\mathbf{k}_j^R)^\top}{\sqrt{192}}
$$

즉, score를 두 부분으로 나눠 계산 후 합산.

### V는 어떻게

V는 RoPE 적용 안 됨 (원래부터). MLA absorption 그대로:
$$
\mathbf{v}_j^i = \mathbf{c}^{KV}_j \mathbf{W}^{UV}_i
$$

Attention output 쪽도 $\mathbf{W}^{UV}$ 흡수 가능 (4-3).

## Cache 구조

```
Per token per layer:
  c_KV  ∈ R^{d_c} = R^{512}           ← 2B × 512 = 1024 bytes
  k^R   ∈ R^{d_h^R} = R^{64}          ← 2B × 64 = 128 bytes
  
  Total: ~1152 bytes per token per layer
  (60 layers, 4K context = ~283 MB per request)
```

기존 GQA 대비 ~7배 절감 유지.

## 전체 Attention 계산 흐름 (Decode)

```
# 현재 token x_t에서:

# 1. Compressed path
c^KV_t = x_t · W^DKV              # [d_c]
cache.append(c^KV_t)              # stored

# 2. RoPE path for K (shared across heads)
k^R_t = RoPE_t(x_t · W^KR)        # [d_h^R]
cache.append(k^R_t)

# 3. Query for each head
for i in num_heads:
    # compressed Q
    q^C_i = x_t · W̃_Q_i            # [d_c] — absorbed form
    
    # RoPE Q (per-head)
    q^R_i = RoPE_t(x_t · W^QR_i)    # [d_h^R]

# 4. Attention (all history)
for i in num_heads:
    # Score from compressed part (using c^KV cache)
    score_C = q^C_i · (all c^KV)^T      # [1, seq]
    
    # Score from RoPE part (using k^R cache — shared!)
    score_R = q^R_i · (all k^R)^T       # [1, seq]
    
    score = (score_C + score_R) / sqrt(192)
    attn_weights = softmax(score)
    
    # Attn output still on compressed space
    attn_out_i = attn_weights · (all c^KV)    # [d_c]

# 5. Absorbed output projection
output = Σ_i attn_out_i · W̃_O_i    # [d_m]
```

## Memory Traffic (Decode, context L tokens)

```
Per step:
  Read c^KV cache: L × 512 × 2B = L × 1024 B
  Read k^R cache:  L × 64 × 2B  = L × 128 B
  (Read q, W matrices: 작음, O(d_m × d_c))
  
Total ~ L × 1152 B per layer

Llama-3-70B GQA 대조:
  Read K,V cache: L × 8 × 128 × 2B × 2 = L × 4096 B per layer
  
→ 약 3.5x 적은 HBM read per decode step
```

## Training과 Inference의 수식 관계

**Training**:
- Expanded form으로 학습 (numerically stable)
- $K_i = c^{KV} \mathbf{W}^{UK}_i$를 실제로 계산하여 attention

**Inference**:
- Absorbed form으로 수행
- $\tilde{\mathbf{W}}^Q_i = \mathbf{W}^{UQ}_i (\mathbf{W}^{UK}_i)^\top$를 사전 계산 (weight merging)
- Decoupled RoPE part는 어차피 별도

### Weight 변환

Model loading 시 absorb할 수 있음:
```python
# Expanded form (학습된 weights)
W_UQ = [...]   # [d_c', d_h] per head
W_UK = [...]   # [d_c, d_h] per head

# Absorbed form for inference
W_Q_absorbed = W_UQ @ W_UK.T   # [d_c', d_c] per head
# 이 weight를 runtime에 사용
```

단, 이러면 per-head weight matrix가 커져 total param memory가 증가.
실제 구현은 absorption을 runtime에 dynamic하게 수행 (fused kernel)하는 경우도 있음.

## A100에서 한눈에

```
DeepSeek-V2 attention decode per layer (single token, context 4K):
  c^KV read: 4096 × 1024 B = 4 MB
  k^R read:  4096 × 128 B = 0.5 MB
  Q compute: ~ 150 MFLOPs per token
  Score compute: 128 × 4096 × 576 = 300 MFLOPs
  Attention: 128 × 4096 × 512 = 268 MFLOPs
  Output proj: 300 MFLOPs

Total HBM: ~5-10 MB per token per layer
Total compute: ~1 GFLOP

A100: HBM 2039 GB/s, FP16 312 TFLOPS
  Latency ≈ max(10/2039, 0.001/312) = 5 μs (memory-bound)

60 layers × 5 μs = 300 μs attention only. 
Throughput of attention >> Llama with GQA.
```

## 다음 — 실제 FlashInfer API

4-5에서 FlashInfer의 MLA 전용 wrapper와 실제 A100에서의 실행 경로를 본다.
