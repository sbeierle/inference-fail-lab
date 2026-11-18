# CASE-008: `flashinfer#9670` — `[b12x]` Cross-Session Program Eviction & `cuModuleUnload` Segfault in `cuGraphLaunch` (SM121 / GB10)

- **Upstream Issue:** [`flashinfer-ai/flashinfer#9670`](https://github.com/flashinfer-ai/flashinfer/issues/9670)
- **Target Hardware:** NVIDIA GB10 (DGX Spark, SM121, 48 SMs)
- **Inspected Versions / Commits:** `flashinfer-python 0.7.4` (`77039051b0be921788dc2784ec41423b558878ec`), `vllm 0.34.0rc1.dev776+g7c3a9302f`, CUDA 13.2 / `torch 2.14.0+cu130`
- **Subsystem:** `flashinfer/experimental/b12x/preparation/session.py`, `_lib/program_cache.py`, `_lib/compile_plan.py`
- **Verification Scope:** Static source-level audit + unexecuted child-process diagnostic harness (`harness/repro_9670.py`). **Not executed on SM120/SM121 bare-metal hardware.**
- **Status:** `🟡 Open / Static Audit (Unmeasured on SM121)`

---

## 1. Background & Reported Symptom

In `flashinfer-ai/flashinfer#9670`, serving `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` (with DFlash speculative decoding) on NVIDIA GB10 (SM121) via vLLM (`--linear-backend b12x --moe-backend b12x`) warms up b12x kernels at startup (`row counts 1, 2, 4, 8, 16, 2048`) and captures CUDA graphs. On the first inference request hitting an unwarmed row capacity (`num_tokens=64`), b12x triggers a lazy first-use plan preparation (`prepare_default`), immediately re-triggers JIT compilation for previously warmed MoE/routing kernels, and crashes with a segmentation fault inside `at::cuda::CUDAGraph::replay()` $\rightarrow$ `cuGraphLaunch`.

---

## 2. Independent Source-Level Audit (`flashinfer@77039051`)

Inspection of `session.py`, `program_cache.py`, and `compile_plan.py` identified three structural issues in the eviction lifecycle:

1. **Session-Local `keep` vs. Process-Global Eviction:**
   - At the end of each preparation job (`session.py:2220–2225`), `keep` is constructed strictly from `self.session._plans`.
   - `evict_unretained(keep)` (`program_cache.py:166–181`) sweeps all process-global registered caches and `_NATIVE_JITS`. Consequently, *any* `PreparationSession` other than the one that originally prepared the captured programs — both the lazy `_LAZY_SESSIONS` instance via `prepare_default()` and a second explicit `PreparationSession` — evicts cache entries belonging to other sessions. Gating eviction on `autotune` alone leaves the cross-session case exposed.

2. **Snapshot Placement (`__init__` vs. Per-Job):**
   - `_LAZY_SESSIONS[detected.ordinal]` reuses the same `PreparationSession` across `prepare_default()` calls (`session.py:2303–2309`).
   - Taking a `_RESIDENT_PROGRAMS` snapshot in `PreparationSession.__init__` goes stale whenever another kernel is loaded between two lazy calls. Taking the snapshot per job and evicting only `programs_created_in_job - keep` aligns with the stated intent at `session.py:2251` (`"losing candidates were launched"`) across both lazy and explicit sessions. (`_RESIDENT_PROGRAMS` also tracks native Triton kernels via `compile_plan.py:401, 559`.)

3. **Executable Ownership & Module Unloading Hypothesis (Unverified on Hardware):**
   - Plan-bound execution states hold a strong reference to their compiled executable, so cache eviction alone should not trigger `cuModuleUnload` for plan-owned kernels.
   - The `cuModuleUnload` $\rightarrow$ `cuGraphLaunch` segfault likely requires a kernel whose sole owner is a cache entry (looked up per call without an owning plan — potentially `_pack_topk_routes_*` if routed through a b12x cache). Additionally, `retain_compiled_programs()` (`compile_plan.py:318`) is currently only entered internally at `session.py:1482`, leaving external graph-capturing callers unprotected unless explicitly wrapped.

---

## 3. Diagnostic Harness (`harness/repro_9670.py`)

Constructed an unexecuted standalone harness (`bf16_gemv`, explicit session + CUDA Graph capture, followed by `prepare_default()` or a second explicit session on an unseen shape):
- Logs whether `keep` covers the captured plan's programs, whether their cache entries are evicted, whether they drop out of `_RESIDENT_PROGRAMS` after `gc.collect()`, and whether `CUDAGraph.replay()` survives.
- Executes each scenario (`--mode lazy` / `--mode second-session`) in an isolated child process so a native `SIGSEGV` in `cuGraphLaunch` is captured and reported cleanly.

---

## 4. Forensic Conclusion

Source-level ownership and cache-lifetime auditing refined the upstream fix proposal from a static `__init__` resident snapshot to a per-job delta (`programs_created_in_job - keep`), closing both the lazy first-use path and cross-session eviction while isolating why non-plan cache-only kernels unload under active CUDA graphs.
