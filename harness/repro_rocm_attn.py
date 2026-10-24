import argparse
import math
import sys
import torch
import triton

sys.path.insert(0, "/home/predator/inference-fail-lab/vllm_upstream")
from vllm.v1.attention.ops.chunked_prefill_paged_decode import kernel_paged_attention_2d

CASES = {
    "ok-d64-bs528":       {"head_size": 64,  "block_size": 528, "dtype": torch.bfloat16},
    "p1-d96-bs528":       {"head_size": 96,  "block_size": 528, "dtype": torch.bfloat16},
    "p1-d128-bs528-fp16": {"head_size": 128, "block_size": 528, "dtype": torch.float16},
    "p1-d192-bs528":      {"head_size": 192, "block_size": 528, "dtype": torch.bfloat16},
    "p1-d256-bs528":      {"head_size": 256, "block_size": 528, "dtype": torch.bfloat16},
    "ok-d256-bs16-gqa4":  {"head_size": 256, "block_size": 16,  "dtype": torch.bfloat16},
    "p2-d256-bs16-gqa1":  {"head_size": 256, "block_size": 16,  "dtype": torch.bfloat16},
}

def run_case(name):
    cfg = CASES.get(name)
    if not cfg:
        print(f"[-] Unbekannter Fall: {name}")
        sys.exit(1)

    head_size = cfg["head_size"]
    block_size = cfg["block_size"]
    dtype = cfg["dtype"]
    device = torch.device("cuda:0")

    print(f"[*] Starte Hardware-Test: {name}")
    print(f"    Target: {torch.cuda.get_device_name(0)} (ROCm {torch.version.hip})")
    print(f"    Config: Head={head_size}, Block={block_size}, Dtype={dtype}")

    head_size_padded = triton.next_power_of_2(head_size)
    
    # Triton Tile-Size: Maximal 64, damit shared memory < 64KB bleibt
    triton_tile_size = 32 if block_size >= 32 else block_size

    num_seqs = 1
    num_query_heads = 1
    num_kv_heads = 1
    seq_len = 1024
    num_blocks = (seq_len + block_size - 1) // block_size

    q = torch.randn((1, num_query_heads, head_size), dtype=dtype, device=device)
    out = torch.empty_like(q)

    x = 16 // torch.tensor([], dtype=dtype).element_size()
    if head_size % x != 0:
        x = 1

    k_cache = torch.randn((num_blocks, num_kv_heads, head_size // x, block_size, x), dtype=dtype, device=device)
    v_cache = torch.randn((num_blocks, num_kv_heads, head_size, block_size), dtype=dtype, device=device)

    sink = torch.empty(0, dtype=dtype, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    alibi_slopes = torch.empty(0, dtype=torch.float32, device=device)

    sm_scale = 1.0 / math.sqrt(head_size)
    grid = (num_seqs, num_query_heads, 1)

    try:
        kernel_paged_attention_2d[grid](
            out,
            q,
            k_cache,
            v_cache,
            sink,
            block_table,
            seq_lens,
            alibi_slopes,
            sm_scale,
            1.0,  # k_scale
            1.0,  # v_scale
            1.0,  # out_scale_inv
            num_query_heads=num_query_heads,
            num_queries_per_kv=1,
            num_queries_per_kv_padded=1,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            BLOCK_SIZE=triton_tile_size,
            PHYSICAL_BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            USE_ALIBI_SLOPES=False,
            SLIDING_WINDOW=0,
            x=x,
            stride_k_cache_0=k_cache.stride(0),
            stride_k_cache_1=k_cache.stride(1),
            stride_k_cache_2=k_cache.stride(2),
            stride_k_cache_3=k_cache.stride(3),
            stride_k_cache_4=k_cache.stride(4),
            stride_v_cache_0=v_cache.stride(0),
            stride_v_cache_1=v_cache.stride(1),
            stride_v_cache_2=v_cache.stride(2),
            stride_v_cache_3=v_cache.stride(3),
            filter_by_query_len=False,
            query_start_len_ptr=torch.zeros(num_seqs + 1, dtype=torch.int32, device=device),
            USE_SINKS=False,
            USE_FP8=False,
            INTERLEAVED_V_PACK_FACTOR=1,
            num_warps=4,
            num_stages=1,
        )
        torch.cuda.synchronize()
        has_nan = torch.isnan(out).any().item(); has_inf = torch.isinf(out).any().item(); print(f"[+] CASE {name}: SUCCESS | NaNs: {has_nan} | Infs: {has_inf} | Mean: {out.float().mean().item():.4f}\n")
    except Exception as e:
        print(f"[-] CASE {name}: CRASH / EXCEPTION -> {e}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=list(CASES.keys()))
    args = parser.parse_args()
    run_case(args.case)
