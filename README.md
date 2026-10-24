# 🔬 Inference Fail-Lab

> **Edge-case verification, failure telemetry, and kernel-level invariant checks for LLM serving engines (vLLM, SGLang, TensorRT-LLM).**

---

## ⚡ What is the Inference Fail-Lab?

Modern LLM serving engines sit at the intersection of dynamic scheduling (Paged Attention, Chunked Prefill, GQA) and accelerator hardware (NVIDIA CUDA, AMD ROCm). When these layers collide, they rarely fail with a clear error message. Typical outcomes:

1. **Silent data corruption:** attention outputs are wrong, nothing crashes, nobody notices.
2. **Hard compiler aborts:** JIT kernels refuse to compile for unusual tensor shapes.
3. **Memory faults and deadlocks:** out-of-bounds access, GPU page faults, stuck schedulers.

The Fail-Lab isolates and reproduces such cases on bare-metal hardware. Cases marked *verified* have a standalone harness and a comparison against an FP32 PyTorch reference. Cases marked *triage* or *benchmarking* do not yet.

---

## 🧭 Where the failures happen

```text
[ LLM engine request ] (e.g. Qwen / DeepSeek / Llama)
│
▼
┌─────────────────────────────────────────────────────────────┐
│ KERNEL / HARDWARE BOUNDARY                                  │
│   • Head sizes that are not a power of 2 (e.g. 96, 192)     │
│   • Large or odd KV block sizes (e.g. 528 vs 16/32/64)      │
│   • Triton compile-time rule: tl.arange needs a power of 2  │
│   • Cache write / read layout mismatches                    │
└─────────────────────────────────────────────────────────────┘
│
├── unpadded or mis-sized path ──► compiler abort OR silent wrong output
└── padded + masked path       ──► output within tolerance of FP32 reference
```

---

## 📋 Cases Registry

| Case ID | Target | Engine / Subsystem | Failure signature | Status |
| :--- | :--- | :--- | :--- | :--- |
| [**CASE-003**](cases/CASE-003.md) | AMD RDNA3 (`gfx1100`, RX 7900 XTX) | vLLM Triton paged-attention decode path | JIT compiler abort (`arange's range must be a power of 2`) on vLLM `<VERSION / COMMIT>` | 🟡 **Patched in standalone harness; not yet re-verified on current vLLM `main`** |
| **CASE-002** | NVIDIA Blackwell (GB300 / SM 10.3) | vLLM `TRITON_MLA` attention | Batch invariance violation via chunked LSE rescaling | 🟡 *In triage* |
| **CASE-001** | NVIDIA Grace-Blackwell (GB10) | Chunked prefill scheduler | Engine deadlock on `FULL_AND_PIECEWISE` attention | 🟡 *Benchmarking* |

---

## 🔍 Deep-Dive: CASE-003

### 1. What broke (plain English)

> **Triton rule:** `tl.arange(0, N)` requires `N` to be a power of 2 (16, 32, 64, 128, 256, ...).

Some models use head sizes like 96 or 192, and some hybrid configurations use KV block sizes like 528. If such a number is passed straight into `tl.arange`, the Triton JIT compiler stops with:

```text
[-] CRASH: arange's range must be a power of 2
```

**Scope of this observation:** it was seen on vLLM `<VERSION / COMMIT>`, kernel `<KERNEL NAME AS IN TRACEBACK>`, file `<PATH:LINE>`, with ROCm 6.4 / PyTorch 2.9.1 / Triton 3.5.1.

> ⚠️ **Check against current upstream before relying on this.** The upstream `kernel_paged_attention_2d` already takes `HEAD_SIZE_PADDED = triton.next_power_of_2(head_size)` and masks with `tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE` (visible in the vLLM API docs since v0.9.0). Whether the crash still reproduces on `main` is open. Until it is re-run there, CASE-003 should be read as a historical or version-specific finding.

### 2. The workaround: pad to the next power of 2 and mask

```python
# Allocate register range at the next power of 2
offs_d = tl.arange(0, HEAD_SIZE_P2)
mask_d = offs_d < HEAD_SIZE

# Load without reading past the real head size
q = tl.load(q_ptr + offs_d, mask=mask_d, other=0.0)
```

The same idea applies to the token dimension for non-power-of-2 block sizes.

### 3. Verification matrix (standalone harness, RX 7900 XTX)

Setup: AMD Radeon RX 7900 XTX (`gfx1100`), ROCm 6.4, PyTorch 2.9.1, Triton 3.5.1. The patched kernel is compared against an FP32 PyTorch reference. **This tests the kernel in a standalone harness, not vLLM's full path** (allocator, cache writer, metadata builder).

| Head dim | KV block | Unpatched kernel | Patched kernel | Max abs. delta vs FP32 |
| :--- | :--- | :--- | :--- | :--- |
| 96 | 16 | ❌ compile abort | ✅ runs | 0.000977 |
| 96 | 64 | ❌ compile abort | ✅ runs | 0.000366 |
| 96 | 528 | ❌ compile abort | ✅ runs | 0.000244 |
| 128 | 528 | ❌ compile abort | ✅ runs | 0.000214 |
| 192 | 16 / 64 / 528 | ❌ compile abort | ✅ runs | < 0.000980 |
| 256 | 16 / 64 / 528 | ✅ runs | ✅ runs | < 0.001950 |

Deltas are within normal bf16/fp16 rounding tolerance. They are **not** bit-identical results, and this README does not claim that.

---

## ⚠️ Related but distinct: vLLM issue #62697

[vllm-project/vllm#62697](https://github.com/vllm-project/vllm/issues/62697) (reported on RDNA4 `gfx1201`, vLLM `main` @ `dde9aca`) describes a **different failure mode**: `ROCM_ATTN` compiles and runs but returns **silently wrong attention** for some head size / block size combinations, and sometimes causes GPU memory faults. According to that thread:

- **Pattern B** (block 16, MHA, head 192/256) is a cache **write** bug in the HIP `reshape_and_cache` kernel (thread cap of 512). A maintainer reproduced it on an MI355X, so it is not RDNA-specific.
- **Pattern A** (block ≥ 32, padded head ≥ 128, decode reads) did not reproduce on MI355X and is still open.

The power-of-2 padding described in CASE-003 does **not** address either pattern. They are separate bugs.

---

## 🚀 Reproduction harness

The harness is standalone; no full vLLM install is needed.

```bash
# 1. Clone repository
git clone https://github.com/<YOUR-USERNAME>/inference-fail-lab.git
cd inference-fail-lab

# 2. Source RDNA3 environment
source setup_env.sh

# 3. Run the standalone kernel check
python harness/case_003_triton_standalone.py
```

---

## 🛠️ Repository layout

```text
├── cases/             # Per-case reports
│   └── CASE-003.md    # Triton power-of-2 compile abort on ROCm
├── harness/           # Minimal working examples reproducing each case
│   └── case_003_triton_standalone.py
├── telemetry/         # dmesg ring buffer and GPU fault monitors
│   └── watch_amdgpu.sh
├── analysis/          # Numerical drift calculators and baseline comparators
├── requirements.txt   # Pinned dependencies (ROCm PyTorch / Triton)
└── setup_env.sh       # Hardware pinning script (gfx1100)
```

---

## 👤 Author

**Stefan Beierle**, independent AI systems researcher and failure analyst
*Focus:* mechanistic LLM activation steering, air-gapped serving infrastructure, accelerator telemetry
