---
title: "5-6. FP8 Inference & DeepGEMM"
weight: 6
---

# 5-6. FP8 Inference & DeepGEMM

## 왜 FP8인가

LLM weight는 bf16/fp16에서 이미 품질 충분.
**FP8로 가면**:
- 메모리 2배 절감 (per-weight 1 byte instead of 2)
- Memory-bound decode에서 latency 2배 가까이 개선
- H100 Hopper는 FP8 tensor core hardware 지원 (full speed)

DeepSeek-V3는 FP8 mixed precision으로 **training 자체**도 수행 ($3M 절감).
Inference는 더 쉬움 — weight를 FP8로 저장만 하면 됨.

## FP8 Formats — Hopper

NVIDIA Hopper가 지원하는 FP8은 두 종류:
- **E4M3**: 4 exponent + 3 mantissa bits → range 작음, precision 높음, **weight용**
- **E5M2**: 5 exponent + 2 mantissa bits → range 넓음, precision 낮음, **activation/gradient용**

```
          Exponent  Mantissa  Max value   Min normal
FP16      5         10        65504       6.10e-5
BF16      8         7         ~3.4e38     1.18e-38
E4M3      4         3         448         2^-9 ≈ 0.002
E5M2      5         2         57344       2^-16 ≈ 1.5e-5
```

## FP8 Scaling

FP8은 range가 좁아 per-tensor scale factor 필요:
$$
x_{\text{fp8}} = \text{round}(x / s) \text{ clamped to FP8 range}
$$
$$
x_{\text{restored}} \approx s \cdot x_{\text{fp8}}
$$

Scale 결정:
- **Per-tensor**: 한 tensor 전체에 하나의 scale
- **Per-channel / per-block**: 작은 block마다 다른 scale (더 정확, overhead)

V3는 **fine-grained per-block scaling** (128x128 블록 등) 사용.

## FP8 GEMM

GEMM $C = A \times B$가 FP8일 때:
- $A$는 E4M3 (또는 E5M2), $B$는 E4M3, 누적은 FP32
- Per-tensor/per-block scale로 dequantize 후 누적

Hopper tensor core:
```
FP8 A (E4M3) × FP8 B (E4M3) → FP32 accumulate
Throughput: 1978 TFLOPS on H100 (BF16의 2배)
```

A100 (Ampere) tensor core:
```
FP8 hardware 지원 없음
→ FP8 weight를 BF16으로 dequantize 후 BF16 tensor core 사용
→ Throughput은 BF16과 동일 (312 TFLOPS)
→ 이득은 '메모리 읽기 절감' 만
```

## A100에서 FP8의 의미

A100에서 FP8 weight 사용의 **실제 이득**:
- **Weight memory read 2배 절감**
- Weight가 HBM에서 읽히는 decode path에서 latency 개선
- Compute speed는 그대로 (BF16 dequant + BF16 MMA)

```
Llama-3-8B decode on A100:
  BF16 weight: FFN gate_up weight read 235 MB → 115 μs
  FP8 weight:  FFN gate_up weight read 117 MB → 58 μs
  → 2x decode throughput 가능 (weight-bound 구간에서)
```

이것이 **GPTQ-4bit, AWQ-4bit** 같은 quantization이 A100에서 잘 돌아가는 이유.
FP8은 그 중간 — 4bit만큼 공격적이진 않지만 quality loss는 더 작음.

## DeepGEMM — H100 전용 FP8 MoE kernel

DeepSeek이 자체 개발한 오픈소스: **DeepGEMM** (2024-12).

### 특징
- FP8 MoE grouped GEMM 전용
- **Hopper WGMMA 명령어** 활용 (A100에는 없음)
- Per-block scaling 최적화
- CUTLASS Grouped GEMM보다 MoE에 특화

### 지원 연산
- `gemm_fp8_fp8_bf16_nt`: FP8×FP8 → BF16 output
- `gemm_fp8_fp8_bf16_nt_masked`: masked version (일부 토큰 무시)
- `grouped_gemm`: MoE용 grouped GEMM

### 예시 API

```python
import deep_gemm

# Grouped FP8 GEMM for MoE
deep_gemm.gemm_fp8_fp8_bf16_nt(
    lhs=(hidden_fp8, hidden_scale),          # (tensor, scale)
    rhs=(expert_weight_fp8, weight_scale),
    out=output,
)
```

A100에서는 **직접 사용 불가** (WGMMA 없음).
해당 연산을 A100에서 하려면:
- Weight는 FP8로 저장 (메모리 절감)
- Dequantize → BF16 GEMM (cuBLAS 또는 Triton)
- 또는 INT8/INT4 (AWQ, GPTQ) 경로

## V3 FP8 Inference 구조

```
Model load:
  Weight files: FP8 (E4M3) + per-block scale (FP32)
  
Forward:
  For each layer:
    - Hidden activation in BF16
    - Weight read from HBM in FP8 (2x less bandwidth)
    - H100: FP8 GEMM directly (DeepGEMM or cuBLAS FP8)
    - A100: dequantize FP8 → BF16 → BF16 GEMM
```

## SGLang에서 V3 serving

SGLang 0.5.x는 DeepSeek-V3 FP8을 지원 (Hopper 기준):

```python
# 모델 로드
from sglang import launch_server

launch_server(
    model_path="deepseek-ai/DeepSeek-V3",
    tp_size=8,           # 8-way TP for 671B
    enable_mla=True,     # MLA backend
    enable_dp_attention=True,
    quantization="fp8",  # Hopper FP8 path
)
```

A100에서는 quantization="awq"/"gptq" 같은 INT quantization을 사용:
```python
launch_server(
    model_path="deepseek-ai/DeepSeek-V2.5-AWQ",  # 4-bit
    tp_size=2,
    enable_mla=True,
    quantization="awq",
)
```

## A100 x2 Perspective

DeepSeek-V3 full (671B FP8 = 336 GB): **A100 x2 (160GB 합)로는 불가능**.
가능한 경로:
1. **V3 AWQ-4bit** (~170 GB): A100 x4 이상 필요, x2로는 여전히 부족
2. **V2-Lite (16B)** FP8/AWQ: 단일 A100 80GB에 올라감
3. **학습 목적으로 관련 기법만 확인**: V2-Lite로 MLA + MoE + MTP + FP8 (A100 dequant) 모두 가능

## A100에서 FP8 활용 예시 (실전)

```python
# PyTorch에서 FP8 weight 저장/로드
# (A100에서는 compute 이득 없이 메모리 이득만)

# Weight quantization (보통 한 번만 실행, save)
def quantize_to_fp8(weight_bf16):
    # per-tensor scaling
    abs_max = weight_bf16.abs().max()
    scale = abs_max / 448  # E4M3 max = 448
    weight_fp8 = (weight_bf16 / scale).clamp(-448, 448)
    weight_fp8 = weight_fp8.to(torch.float8_e4m3fn)
    return weight_fp8, scale

# Runtime dequantization (A100 경로)
def matmul_fp8_weight_a100(x_bf16, weight_fp8, scale):
    weight_bf16 = weight_fp8.to(torch.bfloat16) * scale
    return x_bf16 @ weight_bf16.T
    # 내부: cuBLAS GEMM (BF16)
    # FP8 weight 읽기 덕분에 HBM traffic 절반
```

## 정리 — A100에서 본 FP8

| Aspect | H100 FP8 | A100 FP8 (dequant path) |
|--------|:--------:|:-----------------------:|
| Weight memory | 2x 절감 | 2x 절감 (같음) |
| Compute throughput | 2x (hardware) | 1x (same BF16) |
| GEMM latency (weight-bound) | 2x faster | ~1.5-2x faster |
| GEMM latency (compute-bound) | 2x faster | 동일 |
| Implementation complexity | DeepGEMM/cuBLAS FP8 | manual dequant 필요 |

**A100 결론**: FP8은 "weight 압축"으로만 의미. Decode 같은 memory-bound에서 대부분 이득.
Compute-bound prefill에서는 이득 제한적.

## 다음 — 전체 Kernel Summary

5-7에서 V3 decoder block의 **전체 커널 시퀀스**를 side-by-side로 본다 (S2 Llama와 비교).
