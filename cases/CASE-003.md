# CASE-003: Triton decode compile aborts (withdrawn as a vLLM bug)

**Status: ⚪ Withdrawn.** The aborts came from our own harness, not from vLLM's launcher.

| Symptom we saw | What it actually was |
| :--- | :--- |
| `arange's arguments must be tl.constexpr` | Triton placeholder, active when vLLM's `HAS_TRITON` import guard fails (`vllm/triton_utils/importing.py`); strips `constexpr`. |
| `arange's range must be a power of 2` at `tl.arange(0, BLOCK_SIZE)` | Harness passed the raw physical block size (528) as the tile. The launcher passes `min(block_size, 128)` for power-of-two blocks and 32 otherwise; head size is padded via `HEAD_SIZE_PADDED = next_power_of_2(head_size)`. |
| `out of resource: shared memory 262144 > 65536` | Cause not confirmed; consistent with a tile of 1024 from the same wrong argument. |

The earlier README claims (crash for head 96/192, "bit-identical", root cause `tl.arange(0, HEAD_SIZE)`) were contradicted by the upstream launcher code and are withdrawn. See CASE-004 for the corrected, launcher-mirroring test.
