# CASE-007: `sglang#51278` — AITER MLA Decode FP8 Split-Cap Bypass under CUDA Graph (`SGLANG_AITER_MLA_PERSIST=0`, MI355X / `gfx950`)

- **Upstream Issue:** [`sgl-project/sglang#51278`](https://github.com/sgl-project/sglang/issues/51278)
- **Target Hardware:** AMD Instinct MI355X (`gfx950`)
- **Inspected Commits:** `sgl-project/sglang@21dae9b17ef1` / `ROCm/aiter@285579b6211b`
- **Subsystem:** `python/sglang/srt/layers/attention/aiter_backend.py`, `forward_mla_rocm.py`, `aiter/mla.py`, `csrc/py_itfs_cu/asm_mla.cu`
- **Verification Scope:** Static cross-repo source audit + unexecuted op-level reproducer (`harness/mla_qh16_fp8_split_repro.py`). **Not measured on MI355X bare-metal hardware.**
- **Status:** `🟡 Open / Static Audit (Unmeasured on MI355X)`

---

## 1. Background & Reported Symptom

In `sgl-project/sglang#51278`, running `DeepSeek-V2-Lite-Chat` (`num_head = 16`, `gqa_ratio = 16`, `page_size = 1`) on AMD MI355X (`gfx950`) with `--attention-backend aiter`, `--kv-cache-dtype fp8_e4m3`, CUDA Graph enabled (default), and `SGLANG_AITER_MLA_PERSIST=0`:
- The first request produces valid output.
- Subsequent requests progressively degrade into repetitive garbage (`0/50` on sequential GSM8K).
- Enabling persistent mode (`SGLANG_AITER_MLA_PERSIST=1`, default) or switching to `bf16` KV cache eliminates the corruption.
- No runtime exception or kernel-lookup abort occurs.

---

## 2. Independent Source-Level Root-Cause Audit

### A. Exact Kernel Resolution (`fp8` Q + `fp8` KV, `ps=0`)
- `get_heuristic_kernel_mla` (`asm_mla.cu:164–219`) aborts hard with `AITER_CHECK(false, ... "cannot get heuristic kernel!")` when no exact catalog match exists, ruling out a missing-kernel failure.
- On `gfx95` with `--attention-backend aiter`, `DeepSeek-V2-Lite` executes `_fused_rope_cat_and_cache` (`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py:1116`, gate at `1342–1346`), where `q_out_dtype` matches the KV cache dtype (`fp8`) for `num_head != 12` (`472–480`).
- With `SGLANG_AITER_MLA_PERSIST=0` (`_use_mla_ps_kernel = False`, `aiter_backend.py:189`), `work_meta_data=None` routes `mla_decode_fwd` (`aiter/mla.py:1170, 1307, 1371`) to the non-persistent two-stage path, resolving to `mla_a8w8_qh16_qseqlen1_gqaratio16.co` (`hsa/gfx950/mla/mla_asm.csv` row 14).

### B. Primary Candidate (H1): CUDA Graph Capacity Buffer Bypasses FP8 Split Cap
1. With `PERSIST=0`, SGLang passes `num_kv_splits=None` (`aiter_backend.py:2979`), causing `get_meta_param` (`aiter/mla.py:1212`) to compute split counts from `total_kv = kv_indices.shape[0]` (`mla.py:1168`).
2. `get_meta_param` (`mla.py:367–378`) enforces an fp8-specific cap (`splits <= ceil(avg_kv / 128)` for `nhead * qlen = 16`). In eager mode (`kv_indices` length = `seq_lens_sum`), this prevents empty splits.
3. Under CUDA Graph replay, `kv_indices` points to `cuda_graph_kv_indices` (`aiter_backend.py:3995`), pre-allocated to `max_bs * max_context_len` (`3802`). The fp8 cap becomes a no-op and fixes the split count at capture time (typically `16` at `bs=1`).
4. Stage 2 (`mla.py:224–226`) reduces `min(splits, cdiv(seq_len, 64))` splits (`valid_split_count` is not written by the `gfx950` kernel). If the `qh16` fp8 ASM kernel leaves trailing splits unwritten, Stage 2 reduces uninitialized `logits` / `attn_lse` buffers from the graph memory pool (`mla.py:1280, 1287`), causing step-to-step state leakage.
5. **Precedent in AITER (`v4_nm`):** `mla_decode_fwd_v4_nm` explicitly instructs CUDA-graph callers with capacity-sized indices to pass explicit split counts (`mla.py:2372–2381`) and documents a prior `mgc` / kernel-tile mismatch (`mla.py:2643–2647`). The V3 `qh16` fp8 path lacks both safeguards.

### C. Secondary Candidate (H2): Unzeroed KV Row 0
- `k_buffer_flat[0].zero_()` (`aiter_backend.py:2296–2306`) is gated behind `_use_mla_ps_kernel`. While the prefill path avoids index `0` noting `"the 0 location will be made nan in cuda graph capture mode"` (`aiter_backend.py:6124–6152`), whether the non-persistent `a8w8` decode kernel reads row `0` for masked lanes is unverified from source alone.

---

## 3. Diagnostic Harness (`harness/mla_qh16_fp8_split_repro.py`)

Added an unexecuted op-level reproducer (`v2`) targeting `mla_decode_fwd` directly without starting an SGLang server:
- Uses a `TorchFunctionMode` (`PoisonEmpty`) to fill every float `torch.empty()` inside `mla_decode_fwd` with `NaN` (avoiding PyTorch caching allocator small-pool vs. large-pool isolation).
- Evaluates four arms: **(A)** `fp8` exact vs. capacity-padded `kv_indices`, **(B)** `fp8` forced `num_kv_splits` + `num_kv_splits_indptr`, **(C)** `bf16` control, and **(D)** `NaN` injected into unused KV row `0`.

---

## 4. Forensic Conclusion

Cross-repo static inspection traced a silent interaction across CUDA Graph metadata buffers (SGLang), split-count heuristics (AITER Python), and binary ASM kernel tiling (ROCm AITER) without requiring `gfx950` hardware access. Isolated diagnostic tests and an op-level reproducer were posted upstream for hardware verification by the issue author.
