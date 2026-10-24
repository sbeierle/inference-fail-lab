# 🔬 Inference Fail-Lab

> **Edge-case verification, failure telemetry, and kernel-level checks for LLM serving engines (vLLM, SGLang, TensorRT-LLM). Including the cases that turned out to be our own mistakes.**

---

## ⚡ What is the Inference Fail-Lab?

Modern LLM serving engines sit between dynamic scheduling (Paged Attention, Chunked Prefill, GQA) and accelerator hardware (NVIDIA CUDA, AMD ROCm). When these layers collide, they rarely fail with a clear error. Typical outcomes:

1. **Silent wrong output:** attention results are off, nothing crashes.
2. **Compiler aborts:** JIT kernels refuse to compile for unusual shapes.
3. **Memory faults and deadlocks.**

The lab reproduces such cases on bare-metal hardware and compares against an FP32 PyTorch reference. Every entry states what was **measured**, what was **not**, and under which versions.

---

## 📏 Ground rules (learned the hard way, see CASE-003)

1. **Record the exact state:** vLLM commit **and** `git status` (local patches), ROCm / PyTorch / Triton versions.
2. **Mirror the real caller.** Call the launcher, or copy every launcher parameter (tile sizes, padded dims, grid, cache layout). A kernel called with different arguments is a different experiment.
3. **Every harness needs a positive control:** a case that is known to fail must fail, and a deliberately corrupted input must be flagged.
4. **Compare against FP32** with an explicit tolerance. Say "within tolerance", never "bit-identical" unless it is.
5. **Unmeasured claims are labelled as such.**

---

## 📋 Cases Registry

| Case | Target | Subsystem | What was observed | Status |
| :--- | :--- | :--- | :--- | :--- |
| [**CASE-004**](cases/CASE-004.md) | AMD RDNA3 (`gfx1100`) | vLLM `kernel_paged_attention_2d`, issue #62697 "Pattern A" | **Not reproduced**: 80/80 runs within tolerance in a launcher-mirroring harness with positive control (toolchain differs from the report) | 🟢 **Negative result (limited scope)** |
| [**CASE-003**](cases/CASE-003.md) | AMD RDNA3 (`gfx1100`) | Triton decode kernel, direct call | Compile aborts, traced to **our harness**, not to the vLLM launcher | ⚪ **Withdrawn as a vLLM bug** |
| **CASE-002** | NVIDIA Blackwell (GB300 / SM 10.3) | vLLM `TRITON_MLA` | Batch invariance violation via chunked LSE rescaling | 🟡 Triage, *not re-reviewed in this revision* |
| **CASE-001** | NVIDIA Grace-Blackwell (GB10) | Chunked prefill scheduler | Deadlock on `FULL_AND_PIECEWISE` attention | 🟡 Benchmarking, *not re-reviewed in this revision* |

---

## 🔍 CASE-004: vLLM #62697 "Pattern A" on RDNA3

**Background.** [vllm-project/vllm#62697](https://github.com/vllm-project/vllm/issues/62697) reports, on RDNA4 (`gfx1201`), that `ROCM_ATTN`'s Triton decode kernel `kernel_paged_attention_2d` returns wrong output for block size ≥ 32 with padded head size ≥ 128 (Pattern A), and a separate cache **write** bug for block 16 / MHA / head 192 / 256 (Pattern B). Per the maintainer (2026-10-23), Pattern B is in the HIP `reshape_and_cache` kernel and reproduces on MI355X; Pattern A did not reproduce there.

**Result (measured 2026-10-24).** On RX 7900 XTX (`gfx1100`), Pattern A did **not** reproduce: **80 of 80 runs** passed (max abs error vs FP32 between 0.000013 and 0.001091, against an allowed ~0.02; the issue reports ~0.33 at 1k context). All runs bitwise identical between first and second call, no NaN, no GPU fault.

| Matrix | Runs | What varied |
| :--- | :--- | :--- |
| `run_matrix_v2.sh` | 38 | Head 64/96/192/256 (+128 fp16), block 16 and 528, GQA 4 and 8, context 527/528/529/1k/8k/32k, 8 sequences, ragged lengths |
| `run_pow2.sh` | 42 | Head 128/192/256, blocks 32/64/128/256 (tile 32/64/128), context 1k/8k/32k, plus MHA (8/8) and GQA 2 |

**Harness (v2, `harness/repro_a_v2.py`).** Mirrors the launcher: grid `(num_seqs, num_kv_heads)`, `num_queries_per_kv_padded = max(next_pow2(G), 16)`, `filter_by_query_len=True`, issue-style V layout `[blocks, kv_heads, head, block]` (`INTERLEAVED_V_PACK_FACTOR=0`), shuffled block table, unused slots garbage-filled. Every run first checks the cache layout against the kernel's address formulas, then corrupts one cache block on purpose and requires that the harness flags it (it did, 80/80).

**Environment.** Ubuntu 25.10, kernel 6.17.0-41-generic; ROCm 6.4.43484-123eb5128; PyTorch 2.9.1+rocm6.4; Triton 3.5.1; vLLM checkout `be2de711989bb273e108c41f95d868122b154f9d` (not installed, imported via `sys.path`) with a local change in `vllm/triton_utils/importing.py` forcing `HAS_TRITON = True`.

**What this does not show.**
- It does not show the issue is absent on `gfx1201`. The report used Triton 3.8.0 / ROCm 10.2 on RDNA4; we differ in architecture **and** toolchain, so a difference in outcome cannot be attributed to either one. This is an open hypothesis, not a finding.
- The checkout is newer than the issue's `dde9aca` (the launcher file differs by 10 lines).
- Kernel called directly with launcher-equivalent arguments, **not** through `chunked_prefill_paged_decode`, the KV-cache allocator or a real model.
- Pattern B is not covered (writer bug; this harness never runs the writer).
- The `HAS_TRITON` override is a local workaround. A later `import triton.backends` printed no error, so we have not determined why the guard failed originally, or whether the override is needed.
- Ragged and multi-sequence cases were run for two configurations only; fp16 for one.

**Lesson kept from this case.** An earlier harness version showed errors of 0.18 to 0.30 on *control* cases, the same size as Pattern A. It was a cache-layout mistake in our harness. Hence the self-check and the deliberate-corruption check in v2.

**Next (optional).** The same matrix on a `gfx1201` card or with Triton 3.8.0 would separate architecture from toolchain. We have neither.

---

## 🧯 CASE-003: withdrawn as a vLLM bug

An earlier version of this README described Triton compile aborts for head sizes 96 / 192 and block size 528 as an upstream bug on RDNA3, with a power-of-2 padding fix. The logs we can currently back up show three symptoms, and all three trace to how we called the kernel:

| Observed message | What caused it |
| :--- | :--- |
| `arange's arguments must be of type tl.constexpr` | vLLM's import guard disabled Triton (`triton.backends` import failed in our environment), so the placeholder decorators stripped `constexpr`. Environment artifact. |
| `arange's range must be a power of 2` at `tl.arange(0, BLOCK_SIZE)` | We passed the **physical** block size 528 as the Triton tile. The launcher uses `min(block_size, 128)` for power-of-2 blocks and **32 otherwise**, and it already passes `HEAD_SIZE_PADDED = next_power_of_2(head_size)`. |
| `out of resource: shared memory, Required: 262144, Hardware limit: 65536` | Message observed. The tile size passed in that run is not confirmed in our logs; a tile padded to 1024 would be consistent with it. Not a launcher path either way. |

The earlier matrix with head-size crashes (96 / 192) is **not supported by the logs we have** and is removed. If it reproduces through the real launcher on a stated vLLM commit, it returns as a new case.

---

## 🛠️ Repository layout

```text
├── cases/             # Per-case reports
├── harness/
│   ├── repro_rocm_attn_ref.py   # v1, NOT equivalent to the issue's setup (superseded)
│   ├── repro_a_v2.py            # v2, launcher-mirroring, with layout self-check and positive control
│   ├── run_matrix_v2.sh         # 38-run matrix
│   └── run_pow2.sh              # 42-run matrix (power-of-two blocks)
├── telemetry/         # dmesg ring buffer and GPU fault monitors
├── analysis/          # Numerical drift calculators and baseline comparators
├── requirements.txt   # Pinned dependencies
└── setup_env.sh       # Hardware pinning script (gfx1100)
```

---

## 👤 Author

**Stefan Beierle**, independent AI systems researcher and failure analyst
*Focus:* mechanistic LLM activation steering, air-gapped serving infrastructure, accelerator telemetry
