# CASE-008: FlashInfer `#9670` — `[b12x]` Planless Kernel Eviction & `cuGraphLaunch` Segfault on SM121

- **Upstream Issue:** `flashinfer-ai/flashinfer#9670`
- **Target Hardware:** NVIDIA Blackwell (`GB10` / DGX Spark, `SM121`)
- **Subsystem:** FlashInfer `[b12x]` (`preparation/session.py`, `program_cache.py`, `preparation/types.py`, `compile_plan.py`)
- **Pinned State Audited:** `flashinfer 0.7.4` (`77039051`), vLLM nightly Docker image, `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` (NVFP4 MoE + DFlash speculative decoding)
- **Status:** `🟡 Open / Scope Narrowed on SM121 (Harness Verified by Reporter)`
- **Measurement Scope:** Static source audit and harness authoring performed without local SM121 hardware; bare-metal verification of `harness/repro_9670.py` and patch check executed by issue reporter `@thomaspons` on DGX Spark (`GB10`).

---

## 1. Observed Failure

On NVIDIA DGX Spark (`GB10` / `SM121`), serving `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` (NVFP4 MoE with DFlash speculative decoding) on vLLM nightly warms b12x kernels at startup (`row counts 1, 2, 4, 8, 16, 2048`) and captures CUDA graphs.

On the first inference request hitting an unwarmed capacity (`num_tokens=64`), FlashInfer triggers a first-use preparation (`prepare_default()`), and the server crashes with `SIGSEGV` inside `at::cuda::CUDAGraph::replay()` $\rightarrow$ `cuGraphLaunch`.

---

## 2. Independent Source-Level Audit (`flashinfer@77039051`)

Static inspection of the `[b12x]` preparation and program-cache lifecycle identified three structural points:

1. **Session-Local `keep` vs. Process-Global Eviction:**  
   At the end of every preparation job (`preparation/session.py:2220–2225`), `keep` is collected solely from `self.session._plans`, whereas `evict_unretained(keep)` (`program_cache.py:166–181`) sweeps all process-global caches and `_NATIVE_JITS`.
2. **Per-Job Snapshot Scope (`_LAZY_SESSIONS` Reuse):**  
   Because `_LAZY_SESSIONS[detected.ordinal]` reuses a single `PreparationSession` across `prepare_default()` calls (`session.py:2303–2309`), a resident-program snapshot must be taken per job (`PreparationJob.__init__`) rather than once per session, evicting only `programs_created_in_job - keep`. *(Note: the reporter later clarified their proposed diff already placed the snapshot in `PreparationJob.__init__`.)*
3. **Ownership & Unload Boundary (Hypothesis #3):**  
   Plan-bound execution states hold strong references to their compiled executables. Cache eviction alone therefore should not unload plan-owned modules; the `cuModuleUnload` $\rightarrow$ `cuGraphLaunch` segfault should specifically affect **planless, cache-only kernels** that lack an owning plan and were captured into CUDA graphs.

---

## 3. Diagnostic Harness (`harness/repro_9670.py`)

To test whether cross-session eviction actually unloads plan-bound kernels captured in CUDA graphs, we authored a standalone child-process harness (`harness/repro_9670.py`) supporting two modes (`--mode lazy` and `--mode second-session`):

- Prepares a plan-bound `bf16_gemv` kernel (`plan_a`) and captures it inside a CUDA graph.
- Triggers a subsequent preparation via `prepare_default()` (`lazy`) or a separate `PreparationSession` (`second-session`).
- Instruments `evict_unretained`, tracks `_RESIDENT_PROGRAMS` and weak references across `gc.collect()`, and replays the captured graph inside an isolated child process to catch native `SIGSEGV` exits cleanly.

---

## 4. Bare-Metal SM121 Verification (Reporter Run on DGX Spark)

`@thomaspons` executed `harness/repro_9670.py` on the unpatched GB10 / SM121 image in both `--mode lazy` and `--mode second-session`. Neither mode crashed, and replayed outputs matched:

```text
===== scenario: lazy =====            (second-session gives the same numbers)
  plan_a: programs ["cute:gemm.bf16_projection.selected"], resident 1, cache_entries 2
  evictions: caller "_run", keep_size 1, plan_a_in_keep false,
             plan_a_entries_before 2, plan_a_entries_after 2, removed_total 0
  plan_a_programs_collected: []
  replay_done: output_matches true, max_abs_diff 0.0
  VERDICT: no eviction of plan A observed 
  ```

  ### Confirmed Findings

- **Plan-bound kernels are protected across sessions:** Although `plan_a` is absent from the evicting session's `keep` (`plan_a_in_keep: false`), `_Prepared.retained` (`preparation/types.py`, documented as *"keeps those programs resident for graphs"*) injects `retained_program_keys()` into `evict_unretained()`. Plan-owned programs therefore remain resident regardless of which session triggers eviction.
- **Hypothesis #3 confirmed on hardware:** The reporter narrowed the original issue scope (retracting *"plans of other owners"* as victims). The actual `cuModuleUnload` → `cuGraphLaunch` crash exclusively affects **planless, cache-only kernels** launched outside any `PreparationSession` plan and captured into CUDA graphs—matching the unpatched server logs where `W4A16FusedMoeKernel`, `W4A16TopKSumKernel` (CuTe), and `_pack_topk_routes_sort_kernel` (Triton) were recompiled immediately after the first-use preparation despite startup warmup.
- **Patch & Workaround Verification (Reporter Check):**
  - Patching `PreparationJob.__init__` in `preparation/session.py` with `self._preexisting_programs = frozenset(_RESIDENT_PROGRAMS)` resolved the crash on GB10 across 4 text requests and 1 image request (37 first-use preparations, including `num_tokens=64`, zero crashes).
  - For comparison, vLLM-side PRs `vllm-project/vllm#66277` and `vllm-project/vllm#64766` also passed the same 5-request check on the same image (`#64766` reduced first-use preparations from 37 to 16).
  - **Known limits of reporter check:** Single sequence, 5 requests, no load test; additional memory retained by the patch has not yet been measured.

---

## 5. Open Upstream Contract Question

With plan-bound kernels ruled out, the issue reduces to a clean architectural contract decision for FlashInfer maintainers:

1. **Caller-managed retention:** Must callers launching `b12x` kernels outside a plan explicitly hold a `retain_compiled_programs()` scope for the lifetime of their CUDA graphs (covered by vLLM PRs `#66277` / `#64766`)?
2. **Library-side safety by default:** Or should `PreparationJob` / `prepare_default()` avoid evicting preexisting resident programs it did not create?

*(A follow-up harness targeting a planless kernel captured in a CUDA graph—such as fused MoE or `_pack_topk_routes_sort_kernel`—would directly reproduce the `SIGSEGV` for `tests/experimental/` if needed.)*
