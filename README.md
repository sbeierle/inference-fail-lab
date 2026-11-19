# 🔬 Inference Fail-Lab

Edge-case verification, failure telemetry, and kernel-level checks for LLM serving engines (vLLM, SGLang, TensorRT-LLM). Including the cases that turned out to be our own mistakes.

## ⚡ What is the Inference Fail-Lab?

Modern LLM serving engines sit between dynamic scheduling (Paged Attention, Chunked Prefill, GQA) and accelerator hardware (NVIDIA CUDA, AMD ROCm). When these layers collide, they rarely fail with a clear error. Typical outcomes:
1. **Silent wrong output:** attention results are off, nothing crashes.
2. **Compiler aborts:** JIT kernels refuse to compile for unusual shapes.
3. **Memory faults and deadlocks.**

The lab reproduces such cases on bare-metal hardware and compares against an FP32 PyTorch reference. Every entry states what was measured, what was not, and under which versions.

## 📏 Ground rules (learned the hard way, see CASE-003)

1. **Record the exact state:** engine/kernel commit and git status (local patches), ROCm / CUDA / PyTorch / Triton versions.
2. **Mirror the real caller.** Call the launcher, or copy every launcher parameter (tile sizes, padded dims, grid, cache layout). A kernel called with different arguments is a different experiment.
3. **Every harness needs a positive control:** a case that is known to fail must fail, and a deliberately corrupted input must be flagged.
4. **Compare against FP32 with an explicit tolerance.** Say `"within tolerance"`, never `"bit-identical"` unless it is.
5. **Unmeasured claims are labelled as such.**

---

## 📋 Cases Registry

| Case | Target | Subsystem | What was observed | Status |
|---|---|---|---|---|
| [**CASE-008**](cases/CASE-008.md) | NVIDIA Blackwell (`GB10` / SM121) | FlashInfer `[b12x]` (`preparation/session.py`, `program_cache.py`, `preparation/types.py`), issue `#9670` | **Planless Kernel Eviction & `cuGraphLaunch` Segfault (SM121 Harness Verified):** Reporter ran our child-process harness `repro_9670.py` on unpatched GB10 (`removed_total=0`, `plan_a_in_keep=false`), proving `_Prepared.retained` (`retained_program_keys()`) protects plan-bound kernels across sessions and confirming our Hypothesis #3: the `cuModuleUnload` segfault exclusively hits planless, cache-only kernels (`W4A16FusedMoeKernel`, `W4A16TopKSumKernel`, `_pack_topk_routes_sort_kernel`) captured in CUDA graphs. Reporter's per-job `PreparationJob.__init__` snapshot patch passed on GB10 (5 requests, 37 first-use preps, zero crashes); upstream contract decision pending. | 🟡 Open / Scope Narrowed on SM121 |
| [**CASE-007**](cases/CASE-007.md) | AMD CDNA3.5 (`gfx950` / MI355X) | SGLang + ROCm AITER (`aiter_backend.py`, `forward_mla_rocm.py`, `aiter/mla.py`, `asm_mla.cu`), issue `#51278` | **FP8 MLA Decode Split-Cap Bypass under CUDA Graph (Static Audit):** Verified `DeepSeek-V2-Lite` (`gqa=16`) routes to `mla_a8w8_qh16_qseqlen1_gqaratio16.co` (`ps=0`). Identified `cuda_graph_kv_indices` (`max_bs * max_context_len`) bypassing `get_meta_param`'s fp8 split cap, causing Stage 2 (`mgc=64`) to reduce uninitialized graph-pool buffers (`logits`/`attn_lse`); added `PoisonEmpty` op reproducer (`v2`) and row-0 zeroing isolation tests. | 🟡 Open / Static Audit (Unmeasured on MI355X) |
| [**CASE-006**](cases/CASE-006.md) | AMD CDNA2 (`gfx90a` / MI250X) & CDNA3/3.5 (`gfx942`/`gfx950`) | vLLM `wvSplitK` (`skinny_gemms_wvsplitk.cu`, `WVSPLIT_TILE_GENERIC`), issue `#67110` | **BF16 Register-Spill Regression at `N∈{3,5}`:** Isolated a 14% end-to-end decode slowdown at 5 concurrent requests on MI250X to unpadded LM-head dispatch (`124160x5120`, 8.17x slower than hipBLASLt). Source inspection traced the regression to missing VGPR budget guards in `WVSPLIT_TILE_GENERIC` (`YTILE * __N >= 12`), causing bf16 fp32-widening accumulators to spill into HBM scratch memory. Upstream PR `#66444` adopted the `YTILE=2` fix for wave64 `N=5` (1.9–5.7x speedup on MI355X); `gfx90a` (`N=3, N=5`) remains open. | 🟡 Partially Resolved (`#66444` open for `gfx942`/`gfx950`; `gfx90a` pending) |
| [**CASE-005**](cases/CASE-005.md) | NVIDIA Blackwell (`GB300 NVL` / SM 10.3) | FlashInfer `allreduce_fusion` / `cake_gemm_allreduce_binding.cu`, issue `#6558` | **Eager CUPTI Skew Artifact vs. Graphed Parity:** C++ binding inspection isolated missing one-shot registrations for `n32–n128`. Requested per-call launch traces; upstream rerun under CUDA Graphs + PDL disproved alleged 1.3–3.1x regression as eager host launch skew spun away in collective waits. Crossover restated to 3–4 MB (±1 µs parity). Spin-off: MNNVL multi-node hang isolated (`#7883`). | 🟢 Closed / Benchmark Artifact Refined |
| [**CASE-004**](cases/CASE-004.md) | AMD RDNA3 (`gfx1100`) vs. RDNA4 (`gfx1201`) | vLLM `kernel_paged_attention_2d`, issue `#62697` | **Confirmed Upstream Resolution:** 80/80 runs on `gfx1100` established the clean baseline. Upstream maintainers confirmed Pattern A is isolated to `gfx1201` Triton/LLVM buffer-ops (`< 2 GiB`) and mitigated via PR `#63735` (`num_stages=1`). | 🟢 Closed / Verified by Upstream |
| [**CASE-003**](cases/CASE-003.md) | AMD RDNA3 (`gfx1100`) | Triton decode kernel, direct call | Compile aborts, traced to our harness, not to the vLLM launcher | ⚪ Withdrawn as a vLLM bug |
| **CASE-002** | NVIDIA Blackwell (`GB300` / SM 10.3) | vLLM `TRITON_MLA` | Batch invariance violation via chunked LSE rescaling | 🟡 Triage (no standalone report file yet) |
| [**CASE-001**](cases/CASE-001.md) | NVIDIA Grace-Blackwell (`GB10`) | Chunked prefill scheduler | Deadlock on `FULL_AND_PIECEWISE` attention | 🟡 Benchmarking, not re-reviewed in this revision |

---

## 🔍 CASE-008: FlashInfer `#9670` — `[b12x]` Planless Kernel Eviction & `cuGraphLaunch` Segfault on SM121 (Static Audit + Reporter SM121 Run)

**Background.** `flashinfer-ai/flashinfer#9670` reported that on NVIDIA GB10 (DGX Spark, SM121), serving `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` (NVFP4 MoE, DFlash speculative decoding) via vLLM nightly + FlashInfer `0.7.4` (`77039051`) warms b12x kernels at startup (`row counts 1, 2, 4, 8, 16, 2048`) and captures CUDA graphs, but segfaults inside `at::cuda::CUDAGraph::replay()` $\rightarrow$ `cuGraphLaunch` on the first request hitting an unwarmed capacity (`num_tokens=64`).

**Independent Source-Level Audit & Ownership Isolation (`flashinfer@77039051`).**
- **Session-Local `keep` vs. Global Sweep:** At the end of a preparation job (`session.py:2220–2225`), `keep` is collected solely from `self.session._plans`, whereas `evict_unretained(keep)` (`program_cache.py:166–181`) sweeps all process-global caches and `_NATIVE_JITS`.
- **Ownership Boundary Hypothesis (#3):** Because plan-bound states retain strong references to their executables, we hypothesized that cross-session eviction cannot unload plan-owned modules and that the `cuModuleUnload` $\rightarrow$ `cuGraphLaunch` segfault must target cache-only, planless kernels.
- **Diagnostic Harness (`harness/repro_9670.py`):** Built a child-process reproducer (`bf16_gemv` plan + CUDA graph capture, followed by `prepare_default()` or a second `PreparationSession`) to isolate native `SIGSEGV` failures and log `_RESIDENT_PROGRAMS` retention across `gc.collect()`.

**Bare-Metal SM121 Verification (Executed by Reporter `@thomaspons` on DGX Spark).**
- **Plan-Bound Kernels Confirmed Safe:** Running `repro_9670.py` on the unpatched GB10 image in both `lazy` and `second-session` modes yielded `plan_a_in_keep=false`, `removed_total=0`, `plan_a_programs_collected=[]`, and `output_matches=true` (`max_abs_diff=0.0`). `_Prepared.retained` (`preparation/types.py`) feeds `retained_program_keys()` into `evict_unretained()`, keeping plan-owned programs resident across sessions.
- **Scope Narrowed to Planless Kernels:** This ruled out "plans of other owners" from the original bug report and narrowed the crash exclusively to planless, cache-only kernels captured in CUDA graphs (`W4A16FusedMoeKernel`, `W4A16TopKSumKernel`, `_pack_topk_routes_sort_kernel`).
- **Patch & Contract Status:** The reporter's per-job snapshot in `PreparationJob.__init__` (`self._preexisting_programs = frozenset(_RESIDENT_PROGRAMS)`) passed on GB10 (4 text + 1 image request, 37 first-use preparations including `num_tokens=64`, zero crashes; memory overhead and load-test behavior unmeasured). Upstream vLLM PRs `#66277` and `#64766` also pass the same check. Awaiting FlashInfer maintainer decision on whether callers must hold `retain_compiled_programs()` for planless graph kernels or whether `b12x` should preserve preexisting programs by default.

**Forensic Conclusion.** Pure static ownership auditing plus a targeted child-process harness disproved an overly broad assumption on hardware we do not own, narrowing the upstream issue to a clean binary contract decision for maintainers. See full case report: [`cases/CASE-008.md`](cases/CASE-008.md).

---

## 🔍 CASE-007: SGLang `#51278` — AITER MLA Decode FP8 Split-Cap Bypass under CUDA Graph on MI355X (`gfx950`, Static Audit)

**Background.** `sgl-project/sglang#51278` reported that on AMD MI355X (`gfx950`), `DeepSeek-V2-Lite-Chat` (`num_head=16`, `gqa_ratio=16`) with `--attention-backend aiter`, `--kv-cache-dtype fp8_e4m3`, CUDA Graph enabled, and `SGLANG_AITER_MLA_PERSIST=0` answers Request 1 cleanly but degrades into repetitive garbage from Request 2 onward (`0/50` on sequential GSM8K), whereas `PERSIST=1` or `bf16` KV cache works normally.

**Independent Cross-Repo Root-Cause Audit (`sglang@21dae9b1` / `aiter@285579b6`, unmeasured on MI355X).**
- **Exact Kernel Resolution:** `get_heuristic_kernel_mla` (`asm_mla.cu:164–219`) aborts hard on lookup failure, while `forward_mla_rocm.py:1116, 472–480` quantizes `q` to `fp8` for `num_head != 12`. Non-persistent decode (`ps=0`) therefore hits `mla_a8w8_qh16_qseqlen1_gqaratio16.co` (`hsa/gfx950/mla/mla_asm.csv` row 14).
- **Candidate 1 (CUDA Graph Capacity Buffer Bypasses FP8 Split Cap):** With `PERSIST=0` (`num_kv_splits=None`, `aiter_backend.py:2979`), `get_meta_param` (`aiter/mla.py:1212`) derives split counts from `total_kv = kv_indices.shape[0]`. Under CUDA Graph replay, `kv_indices` points to `cuda_graph_kv_indices` (`aiter_backend.py:3995`, sized `max_bs * max_context_len` at `3802`), rendering the fp8 split cap (`mla.py:367–378`) a no-op and fixing splits at capture. Stage 2 (`mla.py:224–226`) reduces `min(splits, cdiv(seq_len, 64))` splits; if the `qh16` fp8 ASM kernel leaves trailing splits unwritten, Stage 2 reduces uninitialized `logits` / `attn_lse` buffers from the graph pool (`mla.py:1280, 1287`). Notably, `mla_decode_fwd_v4_nm` (`mla.py:2372–2381, 2643–2647`) already documents this exact capacity-sized index / `mgc` tiling hazard for V4.
- **Candidate 2 (Unzeroed KV Row 0):** `k_buffer_flat[0].zero_()` (`aiter_backend.py:2296–2306`) is gated behind `_use_mla_ps_kernel`; whether the non-persistent `a8w8` decode kernel reads row `0` for masked lanes remains unverified from code alone.
- **Op-Level Reproducer (`harness/mla_qh16_fp8_split_repro.py`):** Added an unexecuted `v2` reproducer using `TorchFunctionMode` (`PoisonEmpty`) to poison every float `torch.empty()` inside `mla_decode_fwd` with `NaN` (avoiding PyTorch small-pool vs. large-pool allocator separation) across exact vs. padded indices, forced splits, `bf16` control, and row-0 `NaN` injection.

**Forensic Conclusion.** Cross-repo static dispatch tracing isolated a silent interaction between CUDA Graph capacity buffers and Stage-2 split reduction granularity without requiring `gfx950` hardware access. Upstream diagnostics and a standalone op reproducer were posted for bare-metal verification. See full case report: [`cases/CASE-007.md`](cases/CASE-007.md).

---

## 🔍 CASE-006: vLLM `#67110` — ROCm wvSplitK VGPR Register Spills at Odd Skinny Batches (`N∈3,5`)

**Background.** On ROCm, `rocm_unquantized_gemm_impl` routes unquantized linear projections with `1≤N≤5` activation rows to the custom `wvSplitK` kernel. During multi-stream serving benchmarks on AMD Instinct MI250X (`gfx90a`, 110 CUs per GCD, LUMI-G), decode step latency degraded by 14% at exactly `N=5` concurrent requests (79.56 ms vs. 69.33 ms with `VLLM_ROCM_USE_SKINNY_GEMM=0`), whereas `N∈1,2,4,6,8` showed no comparable regression.

**Independent Isolation & Source-Level Root-Cause Audit.**
- **End-to-End vs. Layer Isolation:** Under default CUDA/HIP graph capture buckets (`1, 2, 4, 8, ...`), a 5-request decode batch pads transformer layers to `N=8` (bypassing `wvSplitK`), while the uncaptured LM head (`124160 x 5120`, bf16) executes at true `N=5` (11.37 ms vs. 1.39 ms in hipBLASLt / `F.linear` — an 8.17x slowdown matching the observed ~10 ms step penalty).
- **Standalone Kernel Matrix (`repro_wvsplitk.py`):** Across 200 calls (median of 7 rotated runs, relative error `1.65×10−3` vs. FP32 reference), `wvSplitK` beat hipBLASLt at `N∈1,4` (`0.81x–0.84x`), but regressed sharply at `N=3` (`3.27x–3.34x`) and `N=5` (`3.14x–8.22x`).
- **Dispatch Macro Forensics (`csrc/rocm/skinny_gemms_wvsplitk.cu`):** Inspection of `WVSPLIT_TILE_GENERIC` revealed an asymmetric guard: `__N == 4` explicitly drops unrolling (`UNRL = 1`) on `YTILE = 4` tiles (`sYT > 12`) to fit within the `gfx90a` VGPR budget, whereas `__N == 3` and `__N == 5` fell through to `YTILE = 4, UNRL = 2` (and `YTILE = 3, UNRL = 2` at `M = 5120`). Because bf16 dot products on CDNA widen operands to fp32 accumulators (`4 * N` accumulators plus double-buffered loads), register pressure overflows into HBM scratch memory.

**Upstream Adoption & Scope Boundary.**
- **Upstream PR `#66444` (`gfx942` / `gfx950`):** AMD engineers verified the identical bf16 `N=5` register spill on MI355X (`gfx950` — triggered at single-request speculative decoding with 4 draft tokens), confirming 248–272 B/lane scratch spills via `-Rpass-analysis=kernel-resource-usage` vs. 0 B/lane when pinned to `YTILE=2, UNRL=2` (achieving 1.9x–5.7x bit-identical speedups).
- **Remaining Open Surface (`gfx90a`):** PR `#66444` restricts its scope to `N=5` on `gfx942`/`gfx950`, leaving `gfx90a` (`N=3` and `N=5`) and a secondary 1.5x discrepancy at `5120 x 3072` (`6≤N≤32`) open for follow-up.

**Forensic Conclusion.** Pairing black-box serving step telemetry with targeted kernel-dispatch source inspection exposed a latent register-allocation cliff masked by CUDA-graph bucket padding. Publishing minimal, reproducible isolation harnesses lets hardware vendors own the C++ kernel maintenance while preserving an auditable trail of cross-layer root-cause attribution. See full case report: [`cases/CASE-006.md`](cases/CASE-006.md).

---

## 🔍 CASE-005: FlashInfer `#6558` Eager CUPTI Spin-Wait Skew vs. CUDA Graph Replay on GB300 NVL

**Background.** `flashinfer-ai/flashinfer#6558` reported that on NVIDIA GB300 NVL (TP4, SM103), FlashInfer's `allreduce_fusion` AUTO heuristic picked two-shot all-reduce ~16x below the apparent crossover point (~1 MiB vs. 12–21 MB), allegedly leaving 1.3x–3.1x throughput uncollected across major serving recipes (DeepSeek-V4.1, GLM-5.3, Nemotron-3-Ultra).

**Independent Dispatch & Binding Audit.** Inspection of the underlying C++ registration table in `cake_gemm_allreduce_binding.cu` revealed an architectural routing disparity:
- Tile `n16` was registered for one-shot (`ws4`), while `n32`, `n64`, and `n128` were registered exclusively as two-shot.
- Highlighting the 42-token anomaly (AUTO running at 192.4 µs vs. 100.7 µs forced two-shot and 44.4 µs forced one-shot), we called for a per-call kernel launch trace to separate CPU launch latency from true device execution, while explicitly cautioning against a blanket `force_oneshot_support=True` override due to two-shot dominance in the prefill tail (218.4 µs vs. 307.2 µs at 46.4 MB).

**Upstream Root-Cause Resolution & Trace Parity.** Subsequent re-benchmarking by FlashInfer maintainers using production serving harness semantics (`bench_gpu_time`, cold L2, `launch_with_pdl=True`, CUDA Graph capture) revealed the original 3x gap was an eager-execution measurement artifact:
1. **Eager Launch Skew:** Under back-to-back eager CUPTI timing, host-side launch skew across 4 ranks was spun away inside the collective's spin-wait synchronization loops, artificially penalizing two-shot due to its two synchronization barriers.
2. **CUDA Graph Parity:** Under CUDA Graph replay with Programmatic Dependent Launch (PDL), the decode-size gap collapsed to `<= 1.2x` on 15–25 µs kernels, with AUTO matching the optimal fixed form within `±1 µs`.
3. **Crossover Restatement:** The true physical crossover restated from 12–21 MB down to ~3–4 MB aggregate (48–64 tokens). End-to-end serving recipes showed threshold tuning remained neutral within `±5–7%` system noise bands.
4. **Isolated Upstream Bug (`#7883`):** Validating the one-shot boundary across topologies isolated a deterministic CUDA Graph hang at multi-MB payloads on 2-node MNNVL, tracked independently as issue `#7883`.

**Forensic Conclusion.** Static binding analysis combined with execution-trace scrutiny prevented an ungrounded heuristic retune in upstream FlashInfer. Issue closed cleanly as not planned to preserve tracker integrity. See full case report: [`cases/CASE-005.md`](cases/CASE-005.md).

---

## 🔍 CASE-004: vLLM `#62697` "Pattern A" & "Pattern B" on RDNA (`gfx1100` vs. `gfx1201`)

**Background.** `vllm-project/vllm#62697` reported that on RDNA4 (`gfx1201`), ROCM_ATTN's Triton decode kernel `kernel_paged_attention_2d` produced numerical mismatches and GPU memory faults for block sizes `>= 32` with head size `>= 128` (Pattern A), alongside a separate cache-writer corruption at block 16 / MHA / head 192/256 (Pattern B).

**Independent Baseline Audit (measured 2026-10-24 on `gfx1100`).** On AMD Radeon RX 7900 XTX (`gfx1100`), Pattern A did NOT reproduce under clean Triton execution:
- 80 of 80 runs passed within numerical tolerance (`|y - ref| <= 0.001` vs FP32, bitwise deterministic across calls, zero GPU faults).
- Test matrix: Head sizes 64–256, block sizes 16–528, GQA 1/2/4/8, context up to 32k tokens, including ragged multi-sequence inputs.
- Positive control: Deliberate block-table corruption was flagged in 80/80 runs.

**Upstream Root-Cause Confirmation & Resolution (2026-10-26).** Subsequent verification by AMD engineers confirmed the partition between the two patterns:
1. **Pattern B (Cache Writer):**
  - Confirmed on datacenter hardware (MI355X / `gfx950`) by `@giuseppegrossi`.
  - Root cause: Hardcoded 512-thread cap in `csrc/cache_kernels.cu` (`reshape_and_cache_kernel`) without grid-stride loop. Chunks past thread 511 are never scheduled, leaving trailing KV heads (16–31) unwritten. Escaped CI because `test_cache.py` only evaluated `NUM_HEADS = [8]`.
2. **Pattern A (Decoder Memory Faults & Mismatches):**
  - Confirmed by `@lijipeng787` to be specific to RDNA4 (`gfx1201`) with ROCm 10 / Triton 3.8.0, and absent on `gfx1100` (consistent with our 80/80 run matrix).
  - Trigger: Triton AMD buffer-ops codegen failure when KV-cache allocation is `< 2 GiB` under multi-stage pipelining (`num_stages > 1`).
  - Upstream Fix: PR `#63735` forces `num_stages=1` and a tile-based warp config for RDNA architectures.

**Forensic Conclusion.** The `gfx1100` baseline in this lab accurately demonstrated that Pattern A was not an inherent RDNA paged-attention algorithmic failure, but a hardware/compiler-codegen edge case on RDNA4. Case closed with upstream fix PR `#63735` (Pattern A) and pending writer PR (Pattern B). See full case documentation: [`cases/CASE-004.md`](cases/CASE-004.md).

---

## 🧯 CASE-003: Withdrawn as a vLLM bug

An earlier version described Triton compile aborts for head sizes 96 / 192 and block size 528 as an upstream bug on RDNA3. All three symptoms traced back to harness invocation arguments (missing tile power-of-2 clamping and disabled constexpr guards). Kept in repository history for forensic discipline: [`cases/CASE-003.md`](cases/CASE-003.md).

---

## 🛠️ Repository layout

```text
├── cases/             # Per-case reports (CASE-001, CASE-003 through CASE-008)
├── docs/assets/       # Terminal logs and before/after forensics
├── harness/           # Reproducible test harnesses (repro_a_v2.py, mla_qh16_fp8_split_repro.py, repro_9670.py)
├── results/           # Raw logs of 80/80 runs on gfx1100
├── telemetry/         # Run logs and GPU fault monitors
└── setup_env.sh       # Environment configuration

👤 Author
Stefan Beierle, independent AI systems researcher and failure analyst

Focus: mechanistic LLM activation steering, air-gapped serving infrastructure, accelerator telemetry
