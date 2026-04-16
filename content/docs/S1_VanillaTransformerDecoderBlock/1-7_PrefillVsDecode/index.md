---
title: "1-7. Prefill vs Decode"
weight: 7
---

# 1-7. Prefill vs Decode — 같은 수식, 다른 커널

## 개념 / 동기

LLM inference는 두 가지 phase로 나뉘며,
**동일한 모델 아키텍처**가 phase에 따라 완전히 다른 연산 특성을 보인다.
이 차이가 모든 LLM serving 최적화의 출발점이다.

## 비교

| | Prefill | Decode |
|---|---------|--------|
| **입력** | 전체 prompt ($S$ tokens) | 새 token 1개 |
| **GEMM shape** | $(S, d_m) \times (d_m, d_m)$ — large M | $(1, d_m) \times (d_m, d_m)$ — M=1 |
| **Attention** | $\mathbf{Q} \in \mathbb{R}^{S \times d_h}$ vs $\mathbf{K,V} \in \mathbb{R}^{S \times d_h}$ | $\mathbf{q} \in \mathbb{R}^{1 \times d_h}$ vs $\mathbf{K,V} \in \mathbb{R}^{t \times d_h}$ |
| **Bottleneck** | **Compute** (FLOPs) | **Memory bandwidth** (weight read + KV cache read) |
| **GPU utilization** | High (SMs saturated) | Low (arithmetic intensity 낮음) |
| **Batching 효과** | 이미 large GEMM → 효과 제한적 | B개 요청을 batch → GEMM shape $(B, d_m)$ → utilization 개선 |

## 수식은 같지만 커널이 다르다

### GEMM

```
Prefill:  cublasGemmEx(M=S, N=d_m, K=d_m)   → compute-bound
Decode:   cublasGemmEx(M=1, N=d_m, K=d_m)    → memory-bound (= GEMV)
          또는 batched: M=B (batch size)
```

cuBLAS/CUTLASS는 M 크기에 따라 **다른 tile configuration**을 선택.
- Large M: 큰 tile (128×256), 높은 compute throughput
- Small M: 작은 tile, 또는 split-K strategy

### Attention

```
Prefill:  flashinfer.prefill_with_paged_kv_cache()
          - Q block과 KV block을 tile 단위로 처리
          - Compute-bound (S가 클 때)

Decode:   flashinfer.decode_with_paged_kv_cache()
          - q가 1개 → KV cache를 streaming read하면서 dot product
          - Memory-bound (KV cache read가 bottleneck)
          - 최적화: page 단위 parallel reduction
```

## Serving 관점의 의미

이 prefill/decode 특성 차이가 다음 최적화 기법들의 동기:

| 기법 | 해결하려는 문제 |
|------|----------------|
| **Continuous Batching** | Decode의 low utilization → 여러 요청을 batch로 묶어 GEMM M 키우기 |
| **Chunked Prefill** | 긴 prefill이 decode를 blocking → prefill을 chunk로 쪼개 interleave |
| **GQA / MQA** (→ S2) | KV cache 메모리 절약 → 더 큰 batch 가능 → decode utilization 개선 |
| **MLA** (→ S4) | KV cache를 low-rank로 압축 → 극단적 메모리 절약 |
| **Speculative Decoding** | Decode의 sequential 특성 회피 → draft model로 병렬 생성 시도 |

## Roofline Model로 보기

```
Arithmetic Intensity (AI) = FLOPs / Bytes

Prefill GEMM:  AI ≈ d_m / 2  (large M일 때)
               d_m=4096 → AI ≈ 2048 → compute-bound on A100

Decode GEMM:   AI ≈ 1         (M=1일 때)
               → memory-bound on A100 (peak BW ~2TB/s)

A100 80GB:
  - Peak FP16 compute: 312 TFLOPS
  - Peak HBM BW: 2039 GB/s
  - Ridge point: 312T / 2039G ≈ 153 FLOPs/Byte
```

## Examples

> [!NOTE]
> TODO: `examples/roofline_prefill_decode.py` — A100에서의 roofline 분석
> TODO: `examples/profile_prefill_vs_decode.sh` — nsys로 실측 비교

