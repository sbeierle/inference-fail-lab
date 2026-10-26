# CASE-001: SGLang Issue #45345 – Startup OOM with 32 Tokenizer Workers & CUDA VMM on Multi-Node GB300

## Status: Resolved Upstream / Verified via Hardware A/B (2026-10-21)

### 1. Incident Overview
On a 2-node NVIDIA GB300 (MNNVL / IMEX, 8x GPUs total) serving Kimi-K3 with SGLang v0.5.22, launching with `--tokenizer-worker-num 32` and auto-selected `cuda_vmm` multimodal feature transport consistently triggered a startup `CUDA_ERROR_OUT_OF_MEMORY` before receiving any inference requests.

- **Target Architecture:** NVIDIA Blackwell GB300 (MNNVL, IMEX channels)
- **Engine / Model:** SGLang v0.5.22 / Kimi-K3 (Multimodal VLM, TP8 / DCP8)
- **Memory Boundary:** GPU capacity 276,620 MiB; peak memory collapsed at 282,299 MiB during startup.

### 2. Forensic Analysis & Root-Cause Deconstruction

The failure was initially suspected to be caused by the 1024 MiB feature-pool budget. Forensic analysis separated two independent drivers:

1. **Hidden CUDA Context & Driver Overhead (Lifecycle Inefficiency):**
   - Eager initialization in `CudaVmmFeatureTransport.__init__` constructed driver runtime pools across all 32 independent worker processes on base GPU 0 before any multimodal request arrived.
   - Mitigated via lazy initialization in PR #45384: pool allocation deferred until the first actual multimodal dispatch.
   - *Limitation:* Lazy init alone still produced OOM during default startup if multimodal warmup was active.

2. **Thundering-Herd Activation Spike (Concurrency Bottleneck):**
   - In multi-tokenizer mode, each worker process spawned an independent, concurrent 448x448 VLM warmup execution via `mm_processor.process_mm_data_async`.
   - Running 32 concurrent vision warmups generated simultaneous activation spikes in the remaining memory headroom (~10% buffer), exceeding physical VRAM.

### 3. Empirical Hardware Verification & Concurrency Sweep

A scaling profile varying `tokenizer_worker_num = N` on the 2-node GB300 cluster demonstrated clear monotonic memory pressure from concurrent worker warmups:

| Workers (N) | Warmups Observed | Peak GPU 0 Used (MiB) | Scheduler Peak (MiB) | Tokenizer-Side Peak (MiB) | Result |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1 | 261,425 | 259,504 | N/A | 🟢 READY |
| **2** | 2 | 262,357 | 259,504 | 2,816 | 🟢 READY |
| **4** | 4 | 264,590 | 259,924 | 4,608 | 🟢 READY |
| **8** | 8 | 266,767 | 260,544 | 6,144 | 🟢 READY |
| **16** | 16 | 269,325 | 261,544 | 7,680 | 🟢 READY |
| **32** | 32 | **282,299** | 261,304 | **17,766** | 🔴 **OOM** |

*Scheduler memory remained essentially flat (~259.5–261.5 GiB), isolating memory growth entirely to concurrent tokenizer worker allocations.*

### 4. Controlled A/B Serialization Test (N=32)

To validate the thundering-herd hypothesis, worker warmups were wrapped with an advisory lock (`flock`) to serialize execution while preserving individual worker initialization and GC-freeze semantics:

- **Arm A (Concurrent Warmups):** 32 workers concurrent $\rightarrow$ Peak 282,299 MiB $\rightarrow$ **OOM** (capacity exceeded by 5.6 GiB).
- **Arm B (Serialized Warmups):** 32 workers sequential $\rightarrow$ Peak 263,275 MiB $\rightarrow$ **🟢 READY** (~19 GiB headroom reclaimed).

### 5. Resolution & Upstream PRs
- **PR #45384:** `[Perf] multimodal: Lazily initialize CUDA VMM feature transport` (Lifecycle improvement).
- **PR #45584:** `[Fix] server: Serialize multi-tokenizer general warmups` (Resolves Issue #45345 by serializing general warmups across workers).
