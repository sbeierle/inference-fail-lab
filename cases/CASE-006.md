# CASE-006: vLLM `#67110` — ROCm `wvSplitK` VGPR Register Spills at Odd Skinny Batches ($N \in \{3, 5\}$)

- **Target Hardware:** AMD Instinct MI250X (`gfx90a`, 110 CUs per GCD, LUMI-G / HPE Cray EX) & AMD Instinct MI355X (`gfx950`, 256 CUs)
- **Subsystem:** vLLM ROCm Unquantized Linear Dispatch (`rocm_unquantized_gemm_impl` in `vllm/model_executor/layers/utils.py`) $\rightarrow$ `wvSplitK` (`csrc/rocm/skinny_gemms_wvsplitk.cu`, `WVSPLIT_TILE_GENERIC`)
- **Upstream References:**
  - Issue: [`vllm-project/vllm#67110`](https://github.com/vllm-project/vllm/issues/67110)
  - Upstream Fix PR (CDNA3/3.5 `N=5`): [`vllm-project/vllm#66444`](https://github.com/vllm-project/vllm/pull/66444) (superseding `#63015`)
- **Status:** 🟡 Partially Resolved (`#66444` addresses `bf16` `N=5` on `gfx942`/`gfx950`; `gfx90a` at `N=3` and `N=5` remains open)

---

## 1. Environment & Measured State

- **Cluster / OS:** LUMI-G (HPE Cray EX), SLES 15 SP6 (`6.4.0` `cray_shasta_c`), Singularity container
- **Software Stack:** `vllm 0.30.0+rocm723`, `ROCm 7.2.3`, `torch 2.12.0+git6bbd260` (`HIP 7.2.53211`), `triton 3.7.1`, `amd-aiter 0.1.21.post2`
- **Precision:** `bfloat16` (`fp16` not measured on `gfx90a`; verified separately by AMD on `gfx950`)
- **Numerical Verification:** Relative error $1.65 \times 10^{-3}$ to $1.68 \times 10^{-3}$ vs. FP32 reference across all tested shapes (within tolerance; numerical accuracy unaffected).

---

## 2. Observed Symptom (End-to-End Serving Regression)

Serving a 27B hybrid model (`--dtype bfloat16 --tensor-parallel-size 2 --max-model-len 65536 --gpu-memory-utilization 0.88`) on two MI250X GCDs under closed-loop streaming completions (`7.7k` prompt tokens, greedy, `96` output tokens, rotated `A B B A` runs):

| Concurrent Requests ($N$) | Default (`ms / decode step`) | `VLLM_ROCM_USE_SKINNY_GEMM=0` (`ms`) | Ratio (`default / off`) |
| :--- | :--- | :--- | :--- |
| **1** | `57.92`, `58.58` | `69.20`, `69.65` | `0.84x` (16% faster) |
| **2** | `67.62`, `68.74` | `69.83`, `69.46` | `0.98x` |
| **4** | `63.98`, `63.31` | `69.02`, `69.49` | `0.92x` (8% faster) |
| **5** | **`79.56`, `79.89`** | **`69.33`, `70.22`** | **`1.14x` (14% slower)** |
| **6** | `70.09`, `69.38` | `69.28`, `69.85` | `1.00x` |
| **8** | `69.71`, `70.06` | `69.95`, `69.67` | `1.00x` |

**Graph-Padding Masking Effect:** With default CUDA/HIP graph capture buckets `(1, 2, 4, 8, ...)`, transformer layers at $N=5$ concurrent requests are padded to $N=8$ and bypass `wvSplitK`. The uncaptured LM head (`124160 x 5120`) executes outside the graph with the true $N=5$ rows (`11.37 ms` vs. `1.39 ms` in `hipBLASLt`), directly accounting for the `+10 ms` per-step penalty.

---

## 3. Standalone Kernel Isolation (`repro_wvsplitk.py`)

Measured across 200 calls between synchronizations (median of 7 rotated rounds on MI250X `gfx90a`):

| Weight Shape ($M \times K$, `bf16`) | Activation Rows ($N$) | `wvSplitK` (`ms`) | `F.linear` / `hipBLASLt` (`ms`) | Slowdown Ratio |
| :--- | :--- | :--- | :--- | :--- |
| `17408 x 5120` | `1` / `4` | `0.153` / `0.160` | `0.189` / `0.190` | `0.81x` / `0.84x` |
| `17408 x 5120` | `2` | `0.223` | `0.189` | `1.18x` |
| `17408 x 5120` | **`3`** | **`0.619`** | `0.190` | **`3.27x`** |
| `17408 x 5120` | **`5`** | **`1.566`** | `0.191` | **`8.22x`** |
| `5120 x 8704` | `1` / `2` / `3` / `4` | `0.070–0.088` | `0.102–0.103` | `0.68x–0.86x` |
| `5120 x 8704` | **`5`** | **`0.324`** | `0.103` | **`3.14x`** |
| `124160 x 5120` (LM head) | `1` / `4` | `1.128` / `1.135` | `1.379` / `1.386` | `0.82x` / `0.82x` |
| `124160 x 5120` (LM head) | **`3`** | **`4.623`** | `1.385` | **`3.34x`** |
| `124160 x 5120` (LM head) | **`5`** | **`11.371`** | `1.392` | **`8.17x`** |

---

## 4. Root-Cause Analysis (`WVSPLIT_TILE_GENERIC`)

Inspection of `csrc/rocm/skinny_gemms_wvsplitk.cu` (lines 73–86) isolated the register-allocation cliff:

1. **Asymmetric Tile Guard:** For large $M$ (`sYT = ceil(M / (CuCount * 4)) > 12`, e.g., `17408` and `124160` on a 110-CU MI250X GCD), dispatch selects `YTILE = 4`. `__N == 4` had an explicit guard lowering `UNRL` from `2` to `1` to stay within the `gfx90a` VGPR budget, whereas `__N == 3` and `__N == 5` fell through to `YTILE = 4, UNRL = 2`.
2. **Mid-Size Spill at $N=5$:** At $M = 5120$ (`sYT <= 12`), dispatch selects `YTILE = 3, UNRL = 2` (9 FP32 accumulators at $N=3$, which fits; 15 FP32 accumulators at $N=5$, which spills).
3. **BF16 Widening Pressure:** On CDNA (`gfx9`), `bf16` dot products widen both operands to `fp32` accumulators (`YTILE * N` accumulators plus double-buffered loads), overflowing VGPRs into HBM scratch memory (`fp16` uses packed dot instructions and does not spill on `gfx942`/`gfx950`).

---

## 5. Upstream Resolution & Open Scope

- **Confirmed on MI355X (`gfx950`):** In PR `#66444`, AMD engineers confirmed via `-Rpass-analysis=kernel-resource-usage` that `bf16` $N=5$ on `wave64` spilled `80–136 B/lane` (`YTILE=3`) and `248–272 B/lane` (`YTILE=4`) on `gfx950`/`gfx942`. Pinning `bf16` $N=5$ to `YTILE=2, UNRL=2` eliminated spills (`0 B/lane`) and yielded **1.9x–5.7x** bit-identical speedups (critical for single-request speculative decoding with 4 draft tokens).
- **Unresolved Items:**
  1. `gfx90a` (`MI250X`) still spills even at `YTILE=2, UNRL=2` (`148 / 20 / 176 B/lane`) and remains regressed at both $N=3$ and $N=5$.
  2. Secondary uninvestigated `1.5x` discrepancy at `5120 x 3072` for $6 \le N \le 32$ (`0.067 ms` vs. `0.045 ms` in `F.linear`).
