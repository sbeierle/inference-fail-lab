**Data point: Pattern A does not reproduce on RDNA3 (`gfx1100`) in a launcher-mirroring harness (80/80 runs). Toolchain differs from the report, so this does not exclude `gfx1201`.**

**Environment**
- AMD Radeon RX 7900 XTX (`gfx1100`, 24 GB), Ubuntu 25.10, kernel 6.17.0-41-generic
- ROCm 6.4.43484-123eb5128, PyTorch 2.9.1+rocm6.4, Triton 3.5.1
- vLLM source checkout at `be2de711989bb273e108c41f95d868122b154f9d`, not installed (imported via `sys.path`). This is newer than `dde9aca`; `chunked_prefill_paged_decode.py` differs by 10 lines.
- Local change: `HAS_TRITON` forced to `True` in `vllm/triton_utils/importing.py` (the guard fell back to the Triton placeholder in my environment). I did not determine why; a later `import triton.backends` printed no error.

**Method**
- Calls `kernel_paged_attention_2d` directly with the launcher's arguments: grid `(num_seqs, num_kv_heads)`, `num_queries_per_kv_padded = max(next_pow2(G), 16)`, `filter_by_query_len=True`, tile `min(block, 128)` for power-of-two blocks else 32, `HEAD_SIZE_PADDED = next_pow2(head)`. It does **not** go through `chunked_prefill_paged_decode`, the cache allocator, or a model.
- Cache layout as in the report: K `[blocks, kv_heads, head//x, block, x]`, V `[blocks, kv_heads, head, block]` (`INTERLEAVED_V_PACK_FACTOR=0`). Shuffled block table, unused slots filled with garbage.
- Reference: FP32 PyTorch attention. Criterion as in this issue: `|y - ref| <= 0.02 + 0.02·|ref|`.
- Each run first checks the cache against the kernel's address formulas, then corrupts one cache block on purpose and requires the harness to flag it (it did in all 80 runs). First and second call are compared bitwise.

**Results (bf16 unless noted): 80/80 within tolerance, all bitwise identical between calls, no NaN, no GPU fault**

| Heads (q/kv) | Head size | Block (tile) | Context | Max abs err |
|---|---|---|---|---|
| 32/8 | 64 | 528 (32) | 527, 528, 529, 1k, 8k, 32k | 0.00012–0.00109 |
| 32/8 | 256 | 16 (16) | same | 0.00013–0.00090 |
| 32/8 | 96 | 528 (32) | same | 0.00011–0.00099 |
| 32/8 | 192 | 528 (32) | same | 0.00012–0.00102 |
| 16/2 | 256 | 528 (32) | same | 0.00009–0.00101 |
| 32/8 (fp16) | 128 | 528 (32) | same | 0.00001–0.00012 |
| 32/8 | 128, 192, 256 | 32, 64, 128, 256 (32/64/128/128) | 1k, 8k, 32k | 0.00009–0.00087 |
| 8/8 (MHA) | 256 | 64 (64) | 1k, 8k, 32k | 0.00012–0.00058 |
| 16/8 (GQA 2) | 128 | 64 (64) | 1k, 8k, 32k | 0.00009–0.00057 |
| 16/2, 8 seqs | 256 | 528 (32) | 8k | 0.00027 |
| 32/8, 8 seqs, ragged | 96 | 528 (32) | 8k | 0.00026 |

For comparison, the report shows ~0.33 at 1k context. Errors here shrink with context (about 0.001 at 1k to 0.0001 at 32k).

**What this does and does not say**
- On `gfx1100` with Triton 3.5.1 / ROCm 6.4, I could not reproduce Pattern A for these shapes. I have not tried `gfx1201`, Triton 3.8.0 or ROCm 10.2, so I cannot say whether the difference comes from the architecture or from the toolchain.
- Not covered: the full launcher path, a real model, Pattern B (writer bug), the HIP paged-attention path.
- Reproduction script and logs: `<LINK TO REPO / GIST>` (`harness/repro_a_v2.py`, `run_matrix_v2.sh`, `run_pow2.sh`, `gfx1100_matrix_v2.txt`, `gfx1100_matrix_pow2.txt`).

If it helps narrowing this down: the same matrix on a `gfx1201` machine, or with Triton 3.8.0 on `gfx1100`, would separate architecture from toolchain.
