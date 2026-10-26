# CASE-004: vLLM Issue #62697 – Pattern A & Pattern B Breakdown on RDNA (gfx1100 vs. gfx1201)

## Status: Resolved Upstream / Verified by Lab Baseline (2026-10-26)

### 1. Summary of Upstream Resolution
vllm-project/vllm#62697 originally grouped two distinct failure modes under ROCM_ATTN:
- **Pattern A:** Numerical mismatches (max abs error ~0.33) and GPU memory faults during decode on large blocks (e.g. block 528) and head dimensions >= 128.
- **Pattern B:** KV-cache writer corruption for block size 16, MHA (32 heads), and head dimensions 192/256.

Subsequent isolation by AMD engineers confirmed our independent baseline findings:
1. **Pattern B (C++ Kernel Bug):**
   - Confirmed on MI355X (gfx950) by @giuseppegrossi.
   - Root cause: `reshape_and_cache_kernel` in `csrc/cache_kernels.cu` capped thread blocks at 512 threads without a grid-stride loop. For 32 KV heads x 256 head size (1024 chunks), chunks past 511 were never scheduled, leaving trailing heads 16–31 unwritten.
   - Slipped through CI because `tests/kernels/attention/test_cache.py` only evaluated `NUM_HEADS = [8]`.
2. **Pattern A (Compiler / Hardware Codegen):**
   - Confirmed by @lijipeng787 to reproduce on RDNA4 (`gfx1201`) with ROCm 10 / Triton 3.8.0, and absent on `gfx1100` (consistent with our 80/80 matrix runs).
   - Trigger: Triton AMD buffer-ops codegen failure when KV cache is < 2 GiB under multi-stage pipelining (`num_stages > 1`).
   - Mitigated via PR #63735 (`num_stages=1` and tile-based warp config on RDNA).

### 2. Empirical Verification on gfx1100 (RX 7900 XTX)
- **Runs:** 80/80 within numerical tolerance (|y - ref| <= 0.001 vs. FP32 PyTorch reference).
- **Bitwise Determinism:** 100% across repeated runs, zero GPU faults.
- **Positive Control:** Deliberate block corruption caught in 80/80 cases.

### 3. Conclusion
The gfx1100 standalone harness correctly isolated that Pattern A was not an algorithmic failure of the Triton paged-attention layout, but a toolchain/codegen edge case on RDNA4.
