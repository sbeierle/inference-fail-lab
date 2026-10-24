# CASE-004: vLLM #62697 "Pattern A" on RDNA3

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
