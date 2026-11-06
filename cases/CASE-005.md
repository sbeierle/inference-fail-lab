# CASE-005: FlashInfer AllReduce Fusion Dispatch & Eager Execution Skew on GB300 NVL

## Metadata
- **Target Hardware:** NVIDIA GB300 NVL (SM 10.3), 4-rank TP4 (Single Node / MNNVL)
- **Software Stack:** CUDA 13.0, FlashInfer 0.7.1rc5, vLLM GB300 Serving Recipes
- **Upstream Issue:** [flashinfer-ai/flashinfer#6558](https://github.com/flashinfer-ai/flashinfer/issues/6558)
- **Related Issue:** [flashinfer-ai/flashinfer#7883](https://github.com/flashinfer-ai/flashinfer/issues/7883) (MNNVL Multi-Node Hang)
- **Status:** 🟢 Closed / Benchmark Artifact Refined

## 1. Problem Statement
In Issue #6558 wurde behauptet, dass der `AUTO`-Pfad von FlashInfers `allreduce_fusion` auf GB300 NVL (TP4) signifikanten Durchsatz verschenkt (1.28x–3.10x hinter der besten Form), weil er bereits bei ~1 MiB auf Two-Shot umschaltet, obwohl One-Shot bis 12–21 MB überlegen sei.

## 2. Dispatch Code Inspection
Eine forensische Prüfung von `cake_gemm_allreduce_binding.cu` zeigte folgende Registrierung in der TP4-Binding-Tabelle:

```cpp
{4, 16, 0, reinterpret_cast<const void*>(&kernel_cake_blackwell_gemm_allreduce_oneshot_n16_bf16_ws4)},
{4, 32, 1, reinterpret_cast<const void*>(&kernel_cake_blackwell_gemm_allreduce_twoshot_n32_bf16_ws4)},
{4, 64, 1, reinterpret_cast<const void*>(&kernel_cake_blackwell_gemm_allreduce_twoshot_n64_bf16_ws4)},
{4, 128, 1, reinterpret_cast<const void*>(&kernel_cake_blackwell_gemm_allreduce_twoshot_n128_bf16_ws4)},
