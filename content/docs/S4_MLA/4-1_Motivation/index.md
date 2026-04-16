---
title: "4-1. MLA Motivation & Concept"
weight: 1
---

# 4-1. MLA Motivation & Concept

## 다시 보는 KV Cache 문제

S1-S3에서 반복된 관찰:
- Decode는 memory-bound
- KV cache read가 매 step 메모리 트래픽의 상당 부분
- Batch를 키울수록 KV cache가 HBM을 잠식

### 숫자로 다시
Llama-3-70B (GQA, n_kv=8, d_h=128), seq_len=4096:
```
KV cache = 2 × 80 (layers) × 8 × 128 × 4096 × 2B = 2.6 GB per request
```

Batch 10개 → 26 GB 하나의 A100 80GB에서 단일 request 하면 weight에 다 잡혀있고 KV cache에 배드먼 50GB여도 부족.

→ 더 압축하고 싶다. 그러나 GQA를 더 밀어붙이면 (n_kv=1 즉 MQA) 품질 저하.

## MLA의 핵심 아이디어

"KV를 하나의 작은 latent vector로 압축하고, 필요할 때 per-head K,V를 **projection으로 복원**한다."

```
Standard MHA/GQA:
  x → W_K → K [n_kv, d_h]   → cache에 저장
  x → W_V → V [n_kv, d_h]   → cache에 저장
  Cache size per token: 2 × n_kv × d_h

MLA:
  x → W_DK → c_KV [d_c]     ← compressed latent (작음!)
  Cache size per token: d_c  (+ 작은 RoPE 부분, 4-4에서)
  
  사용 시:
  K_i = c_KV · W_UK_i          ← i번째 head의 K를 on-the-fly 복원
  V_i = c_KV · W_UV_i          ← i번째 head의 V를 on-the-fly 복원
```

즉, cache에 **압축된 $c_{KV}$만 저장**하고, attention 계산 시 per-head로 **up-project**.

## 수치 비교 — DeepSeek-V2

DeepSeek-V2의 MLA 파라미터:
```
d_model = 5120
n_q_heads = 128           (매우 많음, 기존 모델 대비)
d_h per head = 128
d_c (KV compression dim) = 512    ← 이게 핵심
d_c' (Q compression dim) = 1536   ← Q도 일부 압축

KV cache per token per layer:
  GQA-equivalent (n_kv=8, d_h=128):  2 × 8 × 128 × 2B = 4 KB
  MLA (compressed):                     512 × 2B + RoPE 64 × 2B ≈ 576 + 128 = 704 bytes  (실제 ~576 bytes in fp16/bf16 + 위 RoPE 부분)
  
→ ~7x 압축 vs GQA, 품질은 MHA 수준 유지
```

**결과**:
- DeepSeek-V2 총 236B params (MoE 포함)
- KV cache는 GQA 모델의 7x 작음
- → 훨씬 큰 batch / 긴 context 가능

## MLA vs GQA — 수치 비교 표

| | MHA (Llama-2-70B) | GQA (Llama-3-70B) | MLA (DS-V2) |
|---|:---:|:---:|:---:|
| n_q heads | 64 | 64 | 128 |
| n_kv heads | 64 | 8 | n/a (compressed) |
| per-token KV | ~16 KB | ~2 KB | ~576 B |
| 품질 | baseline | baseline급 | baseline급 |
| Attn FLOPs | baseline | baseline | baseline |
| Extra compute | 0 | 0 | up-projection 추가 |

MLA는 "KV 저장 용량 ↓ 대신 compute 조금 ↑" trade-off.
Memory-bound decode에선 compute 여유가 있으므로 유리.

## 왜 일반 low-rank compression이 아닌가

$K, V$를 단순히 low-rank로 근사 $K \approx U_K S_K V_K^\top$하면:
- Approximation 손실로 품질 저하
- Runtime에 SVD 같은 것 불가능

MLA는 이와 달리:
- **학습 가능한 projection**으로 down/up
- Training 시점에 loss가 perplexity 기반으로 최적화
- Inference 시 projection 연산은 저렴 (GEMM)

## MLA의 "공짜 점심" — Matrix Absorption

단순히 보면:
- Store $c_{KV}$ (작음) → save memory ✓
- Every attention 계산 시 K, V 복원 → compute 추가 ✗

그러나 **수학적 구조 덕분에 compute 증가도 피할 수 있음**:
- $\mathbf{Q} \mathbf{K}^\top = \mathbf{Q} (c_{KV} \mathbf{W}_{UK})^\top = (\mathbf{Q} \mathbf{W}_{UK}^\top) c_{KV}^\top$
- $\mathbf{W}_{UK}$를 $\mathbf{W}_Q$ 뒤에 **흡수**시키면, runtime에는 $c_{KV}$와 직접 내적

자세한 설명은 4-3 Absorption Trick에서.

## 왜 이제서야 등장했나

1. **학습 안정화 기법 성숙**: Pre-LN + RMSNorm으로 low-rank 구조 학습 가능
2. **RoPE 해결**: 순수 low-rank는 RoPE와 충돌 — 4-4에서 설명
3. **Dataset + compute scale**: small-scale에서는 ablation 효과 미미

DeepSeek-V2 (2024)가 이를 검증하고 mainstream화.

## Subsection 미리보기

- **4-2 Math**: 정확한 수식 (down/up projection dimension)
- **4-3 Absorption**: runtime에 K,V를 실제로 복원 안 하는 트릭
- **4-4 Decoupled RoPE**: RoPE를 MLA에 어떻게 결합했나
- **4-5 FlashInfer MLA**: 실제 API와 CUDA kernel
