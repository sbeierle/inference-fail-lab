# 🔬 Inference Fail-Lab

Edge-case verification, failure telemetry, and kernel-level checks for LLM serving engines (vLLM, SGLang, TensorRT-LLM). Including the cases that turned out to be our own mistakes.

## ⚡ What is the Inference Fail-Lab?
Modern LLM serving engines sit between dynamic scheduling (Paged Attention, Chunked Prefill, GQA) and accelerator hardware (NVIDIA CUDA, AMD ROCm). When these layers collide, they rarely fail with a clear error. Typical outcomes:
1. Silent wrong output: attention results are off, nothing crashes.
2. Compiler aborts: JIT kernels refuse to compile for unusual shapes.
3. Memory faults and deadlocks.

The lab reproduces such cases on bare-metal hardware and compares against an FP32 PyTorch reference. Every entry states what was measured, what was not, and under which versions.

## 📏 Ground rules (learned the hard way, see CASE-003)
1. Record the exact state: vLLM commit and git status (local patches), ROCm / PyTorch / Triton versions.
2. Mirror the real caller. Call the launcher, or copy every launcher parameter (tile sizes, padded dims, grid, cache layout). A kernel called with different arguments is a different experiment.
3. Every harness needs a positive control: a case that is known to fail must fail, and a deliberately corrupted input must be flagged.
4. Compare against FP32 with an explicit tolerance. Say "within tolerance", never "bit-identical" unless it is.
5. Unmeasured claims are labelled as such.

## 📋 Cases Registry

| Case | Target | Subsystem | What was observed | Status |
| :--- | :--- | :--- | :--- | :--- |
| [**CASE-005**](cases/CASE-005.md) | NVIDIA Blackwell (`GB300 NVL` / `SM 10.3`) | FlashInfer `allreduce_fusion` / `cake_gemm_allreduce_binding.cu`, issue #6558 | **Eager CUPTI Skew Artifact vs. Graphed Parity:** C++ binding inspection isolated missing one-shot registrations for n32–n128. Requested per-call launch traces; upstream rerun under CUDA Graphs + PDL disproved alleged 1.3–3.1x regression as eager host launch skew spun away in collective waits. Crossover restated to 3–4 MB (±1 µs parity). Spin-off: MNNVL multi-node hang isolated (#7883). | 🟢 **Closed / Benchmark Artifact Refined** |
| [**CASE-004**](cases/CASE-004.md) | AMD RDNA3 (`gfx1100`) vs. RDNA4 (`gfx1201`) | vLLM `kernel_paged_attention_2d`, issue #62697 | **Confirmed Upstream Resolution:** 80/80 runs on gfx1100 established the clean baseline. Upstream maintainers confirmed Pattern A is isolated to gfx1201 Triton/LLVM buffer-ops (< 2 GiB) and mitigated via PR #63735 (`num_stages=1`). | 🟢 **Closed / Verified by Upstream** |
| [**CASE-003**](cases/CASE-003.md) | AMD RDNA3 (`gfx1100`) | Triton decode kernel, direct call | Compile aborts, traced to our harness, not to the vLLM launcher | ⚪ Withdrawn as a vLLM bug |
| [**CASE-002**](cases/CASE-002.md) | NVIDIA Blackwell (GB300 / SM 10.3) | vLLM TRITON_MLA | Batch invariance violation via chunked LSE rescaling | 🟡 Triage, not re-reviewed in this revision |
| [**CASE-001**](cases/CASE-001.md) | NVIDIA Grace-Blackwell (GB10) | Chunked prefill scheduler | Deadlock on FULL_AND_PIECEWISE attention | 🟡 Benchmarking, not re-reviewed in this revision |

---

## 🔍 [CASE-005](cases/CASE-005.md): FlashInfer #6558 Eager CUPTI Spin-Wait Skew vs. CUDA Graph Replay on GB300 NVL

**Background.** flashinfer-ai/flashinfer#6558 reported that on NVIDIA GB300 NVL (TP4, SM103), FlashInfer's `allreduce_fusion` AUTO heuristic picked two-shot all-reduce ~16x below the apparent crossover point (~1 MiB vs. 12–21 MB), allegedly leaving 1.3x–3.1x throughput uncollected across major serving recipes (DeepSeek-V4.1, GLM-5.3, Nemotron-3-Ultra).

**Independent Dispatch & Binding Audit.**
Inspection of the underlying C++ registration table in `cake_gemm_allreduce_binding.cu` revealed an architectural routing disparity:
- Tile `n16` was registered for one-shot (`ws4`), while `n32`, `n64`, and `n128` were registered exclusively as two-shot.
- Highlighting the 42-token anomaly (AUTO running at 192.4 µs vs. 100.7 µs forced two-shot and 44.4 µs forced one-shot), we called for a per-call kernel launch trace to separate CPU launch latency from true device execution, while explicitly cautioning against a blanket `force_oneshot_support=True` override due to two-shot dominance in the prefill tail (218.4 µs vs. 307.2 µs at 46.4 MB).

**Upstream Root-Cause Resolution & Trace Parity.**
Subsequent re-benchmarking by FlashInfer maintainers using production serving harness semantics (`bench_gpu_time`, cold L2, `launch_with_pdl=True`, CUDA Graph capture) revealed the original 3x gap was an eager-execution measurement artifact:
1. **Eager Launch Skew:** Under back-to-back eager CUPTI timing, host-side launch skew across 4 ranks was spun away inside the collective's spin-wait synchronization loops, artificially penalizing two-shot due to its two synchronization barriers.
2. **CUDA Graph Parity:** Under CUDA Graph replay with Programmatic Dependent Launch (PDL), the decode-size gap collapsed to <= 1.2x on 15–25 µs kernels, with AUTO matching the optimal fixed form within ±1 µs.
3. **Crossover Restatement:** The true physical crossover restated from 12–21 MB down to ~3–4 MB aggregate (48–64 tokens). End-to-end serving recipes showed threshold tuning remained neutral within ±5–7% system noise bands.
4. **Isolated Upstream Bug (#7883):** Validating the one-shot boundary across topologies isolated a deterministic CUDA Graph hang at multi-MB payloads on 2-node MNNVL, tracked independently as issue #7883.

**Forensic Conclusion.**
Static binding analysis combined with execution-trace scrutiny prevented an ungrounded heuristic retune in upstream FlashInfer. Issue closed cleanly as *not planned* to preserve tracker integrity. See full case report: [`cases/CASE-005.md`](cases/CASE-005.md).

---

## 🔍 [CASE-004](cases/CASE-004.md): vLLM #62697 "Pattern A" & "Pattern B" on RDNA (gfx1100 vs. gfx1201)

**Background.** vllm-project/vllm#62697 reported that on RDNA4 (`gfx1201`), `ROCM_ATTN`'s Triton decode kernel `kernel_paged_attention_2d` produced numerical mismatches and GPU memory faults for block sizes >= 32 with head size >= 128 (Pattern A), alongside a separate cache-writer corruption at block 16 / MHA / head 192/256 (Pattern B).

**Independent Baseline Audit (measured 2026-10-24 on gfx1100).**
On AMD Radeon RX 7900 XTX (`gfx1100`), Pattern A did NOT reproduce under clean Triton execution:
- 80 of 80 runs passed within numerical tolerance (|y - ref| <= 0.001 vs FP32, bitwise deterministic across calls, zero GPU faults).
- Test matrix: Head sizes 64–256, block sizes 16–528, GQA 1/2/4/8, context up to 32k tokens, including ragged multi-sequence inputs.
- Positive control: Deliberate block-table corruption was flagged in 80/80 runs.

**Upstream Root-Cause Confirmation & Resolution (2026-10-26).**
Subsequent verification by AMD engineers confirmed the partition between the two patterns:
1. **Pattern B (Cache Writer):**
   - Confirmed on datacenter hardware (MI355X / gfx950) by @giuseppegrossi.
   - Root cause: Hardcoded 512-thread cap in `csrc/cache_kernels.cu` (`reshape_and_cache_kernel`) without grid-stride loop. Chunks past thread 511 are never scheduled, leaving trailing KV heads (16–31) unwritten. Escaped CI because `test_cache.py` only evaluated `NUM_HEADS = [8]`.
2. **Pattern A (Decoder Memory Faults & Mismatches):**
   - Confirmed by @lijipeng787 to be specific to RDNA4 (`gfx1201`) with ROCm 10 / Triton 3.8.0, and absent on `gfx1100` (consistent with our 80/80 run matrix).
   - Trigger: Triton AMD buffer-ops codegen failure when KV-cache allocation is < 2 GiB under multi-stage pipelining (`num_stages > 1`).
   - Upstream Fix: PR #63735 forces `num_stages=1` and a tile-based warp config for RDNA architectures.

**Forensic Conclusion.**
The `gfx1100` baseline in this lab accurately demonstrated that Pattern A was not an inherent RDNA paged-attention algorithmic failure, but a hardware/compiler-codegen edge case on RDNA4. Case closed with upstream fix PR #63735 (Pattern A) and pending writer PR (Pattern B). See full case documentation: [`cases/CASE-004.md`](cases/CASE-004.md).

---

## 🧯 [CASE-003](cases/CASE-003.md): Withdrawn as a vLLM bug
An earlier version described Triton compile aborts for head sizes 96 / 192 and block size 528 as an upstream bug on RDNA3. All three symptoms traced back to harness invocation arguments (missing tile power-of-2 clamping and disabled constexpr guards). Kept in repository history for forensic discipline: [`cases/CASE-003.md`](cases/CASE-003.md).

---

## 🛠️ Repository layout
```text
├── cases/             # Per-case reports (CASE-001, CASE-002, CASE-003, CASE-004, CASE-005)
├── docs/assets/       # Terminal logs and before/after forensics
├── harness/           # Reproducible test harnesses (repro_a_v2.py, matrix runners)
├── results/           # Raw logs of 80/80 runs on gfx1100
├── telemetry/         # Run logs and GPU fault monitors
└── setup_env.sh       # Environment configuration

👤 Author
Stefan Beierle, independent AI systems researcher and failure analyst

Focus: mechanistic LLM activation steering, air-gapped serving infrastructure, accelerator telemetry
