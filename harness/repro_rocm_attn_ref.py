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

def is_power_of_two(n):
    return (n > 0) and (n & (n - 1) == 0)

def run_attention_ref(q, k, v, sm_scale):
    # Pure PyTorch FP32 Reference Implementation
    # q: [1, 1, head_size]
    # k, v: [seq_len, head_size]
    q_f32 = q.float().unsqueeze(1)                   # [1, 1, 1, head_size]
    k_f32 = k.float().unsqueeze(0).unsqueeze(0)       # [1, 1, seq_len, head_size]
    v_f32 = v.float().unsqueeze(0).unsqueeze(0)       # [1, 1, seq_len, head_size]
    
    scores = torch.matmul(q_f32, k_f32.transpose(-1, -2)) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    out_ref = torch.matmul(probs, v_f32).squeeze(0).squeeze(1)
    return out_ref

def execute_kernel(cfg, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    head_size = cfg["head_size"]
    block_size = cfg["block_size"]
    dtype = cfg["dtype"]
    device = torch.device("cuda:0")

    is_pow2 = is_power_of_two(block_size)
    triton_block_size = min(block_size, 128) if is_pow2 else 32
    head_size_padded = triton.next_power_of_2(head_size)

    num_seqs = 1
    num_query_heads = 1
    num_kv_heads = 1
    seq_len = 1024
    num_blocks = (seq_len + block_size - 1) // block_size

    # Q & Ground-Truth Sequenzen
    q = torch.randn((1, num_query_heads, head_size), dtype=dtype, device=device)
    k_full = torch.randn((seq_len, head_size), dtype=dtype, device=device)
    v_full = torch.randn((seq_len, head_size), dtype=dtype, device=device)

    # 16-Byte Alignment Pack Factor x
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    if head_size % x != 0:
        x = 1

    # K-Cache: [num_blocks, num_kv_heads, head_size // x, block_size, x]
    k_cache = torch.zeros((num_blocks, num_kv_heads, head_size // x, block_size, x), dtype=dtype, device=device)
    
    # V-Cache: [num_blocks, num_kv_heads, block_size, head_size] (für INTERLEAVED_V_PACK_FACTOR=1)
    v_cache = torch.zeros((num_blocks, num_kv_heads, block_size, head_size), dtype=dtype, device=device)

    # Cache deterministisch befüllen
    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, seq_len)
        cur_len = end - start
        if cur_len > 0:
            # V-Cache: Tokens direkt in [cur_len, head_size]
            v_cache[b, 0, :cur_len, :] = v_full[start:end, :]
            
            # K-Cache: [cur_len, head_size // x, x] -> permute(1, 0, 2) -> [head_size // x, cur_len, x]
            chunk_k = k_full[start:end, :].view(cur_len, head_size // x, x).permute(1, 0, 2)
            k_cache[b, 0, :, :cur_len, :] = chunk_k

    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    alibi_slopes = torch.empty(0, dtype=torch.float32, device=device)
    sink = torch.empty(0, dtype=dtype, device=device)
    out = torch.empty_like(q)

    sm_scale = 1.0 / math.sqrt(head_size)
    grid = (num_seqs, num_query_heads, 1)

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
        1.0, 1.0, 1.0,
        num_query_heads=num_query_heads,
        num_queries_per_kv=1,
        num_queries_per_kv_padded=1,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=triton_block_size,
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

    ref = run_attention_ref(q, k_full, v_full, sm_scale)
    return out.squeeze(0), ref

def evaluate_case(name):
    cfg = CASES.get(name)
    if not cfg:
        print(f"[-] Unbekannter Fall: {name}")
        sys.exit(1)

    print(f"[*] Teste Fall: {name} (Head={cfg['head_size']}, Block={cfg['block_size']}, Dtype={cfg['dtype']})")

    # Run 1 & Run 2 (Determinismus-Prüfung)
    out1, ref = execute_kernel(cfg, seed=42)
    out2, _   = execute_kernel(cfg, seed=42)

    diff_runs = (out1 - out2).abs().max().item()
    is_deterministic = (diff_runs == 0.0)

    # Issue #62697 Toleranzgrenze: tol = 0.02 + 0.02 * abs(ref)
    out1_f32 = out1.float()
    abs_diff = (out1_f32 - ref).abs()
    tol = 0.02 + 0.02 * ref.abs()
    violations = (abs_diff > tol).sum().item()
    max_err = abs_diff.max().item()
    max_tol_at_err = tol[abs_diff == max_err].max().item() if max_err > 0 else 0.0

    print(f"    Determinismus (Run1 vs Run2): {'PASS (bitgenau 0.0)' if is_deterministic else f'FAIL (Diff: {diff_runs})'}")
    print(f"    Max Diff vs FP32 Ref: {max_err:.6f} (Erlaubte Tol: {max_tol_at_err:.6f})")
    
    if violations == 0:
        print(f"[+] RESULT {name}: PASS (Mathematisch innerhalb der Issue-Toleranz)\n")
    else:
        print(f"[-] RESULT {name}: FAIL (Pattern A REPRODUZIERT! {violations} Werte ausserhalb Toleranz)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=list(CASES.keys()))
    args = parser.parse_args()
    evaluate_case(args.case)
