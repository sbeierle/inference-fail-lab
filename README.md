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
| **CASE-004** | AMD RDNA3 (`gfx1100`) vs. RDNA4 (`gfx1201`) | vLLM `kernel_paged_attention_2d`, issue #62697 | **Confirmed Upstream Resolution:** 80/80 runs on gfx1100 established the clean baseline. Upstream maintainers confirmed Pattern A is isolated to gfx1201 Triton/LLVM buffer-ops (< 2 GiB) and mitigated via PR #63735 (`num_stages=1`). | 🟢 **Closed / Verified by Upstream** |
| **CASE-003** | AMD RDNA3 (`gfx1100`) | Triton decode kernel, direct call | Compile aborts, traced to our harness, not to the vLLM launcher | ⚪ Withdrawn as a vLLM bug |
| **CASE-002** | NVIDIA Blackwell (GB300 / SM 10.3) | vLLM TRITON_MLA | Batch invariance violation via chunked LSE rescaling | 🟡 Triage, not re-reviewed in this revision |
| **CASE-001** | NVIDIA Grace-Blackwell (GB10) | Chunked prefill scheduler | Deadlock on FULL_AND_PIECEWISE attention | 🟡 Benchmarking, not re-reviewed in this revision |

---

## 🔍 CASE-004: vLLM #62697 "Pattern A" & "Pattern B" on RDNA (gfx1100 vs. gfx1201)

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
The `gfx1100` baseline in this lab accurately demonstrated that Pattern A was not an inherent RDNA paged-attention algorithmic failure, but a hardware/compiler-codegen edge case on RDNA4. Case closed with upstream fix PR #63735 (Pattern A) and pending writer PR (Pattern B).

---

## 🧯 CASE-003: Withdrawn as a vLLM bug
An earlier version described Triton compile aborts for head sizes 96 / 192 and block size 528 as an upstream bug on RDNA3. All three symptoms traced back to harness invocation arguments (missing tile power-of-2 clamping and disabled constexpr guards). Kept in repository history for forensic discipline.

---

## 🛠️ Repository layout
```text
├── cases/             # Per-case reports (CASE-003, CASE-004)
├── docs/assets/       # Terminal logs and before/after forensics
├── harness/           # Reproducible test harnesses (repro_a_v2.py, matrix runners)
├── results/           # Raw logs of 80/80 runs on gfx1100
├── telemetry/         # Run logs and GPU fault monitors
└── setup_env.sh       # Environment configuration

👤 Author
Stefan Beierle, independent AI systems researcher and failure analyst

Focus: mechanistic LLM activation steering, air-gapped serving infrastructure, accelerator telemetry
