import torch
import triton
import triton.language as tl
import math

def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

@triton.jit
def _paged_attn_decode_kernel_patched(
    Out,
    Q,
    K_Cache,
    V_Cache,
    Block_Tables,
    Context_Lens,
    sm_scale,
    stride_out_b, stride_out_h,
    stride_q_b, stride_q_h,
    stride_k_b, stride_k_bl, stride_k_h,
    stride_v_b, stride_v_bl, stride_v_h,
    stride_bt_b,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_P2: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_P2: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_context_len = tl.load(Context_Lens + cur_batch)
    if cur_context_len <= 0:
        return

    # Head-Offsets gepuffert auf 2^n mit striktem Masking
    offs_d = tl.arange(0, HEAD_SIZE_P2)
    mask_d = offs_d < HEAD_SIZE

    q_ptr = Q + cur_batch * stride_q_b + cur_head * stride_q_h
    q = tl.load(q_ptr + offs_d, mask=mask_d, other=0.0)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_SIZE_P2], dtype=tl.float32)

    num_blocks = (cur_context_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    for block_idx in range(num_blocks):
        phys_block = tl.load(Block_Tables + cur_batch * stride_bt_b + block_idx)

        # Block-Offsets gepuffert auf 2^n mit striktem Masking
        offs_tokens = tl.arange(0, BLOCK_SIZE_P2)
        valid_block_mask = offs_tokens < BLOCK_SIZE
        token_mask = valid_block_mask & ((block_idx * BLOCK_SIZE + offs_tokens) < cur_context_len)

        # K-Pointer mit 2D Masking (Tokens x Head)
        k_ptr = K_Cache + phys_block * stride_k_b + offs_tokens[:, None] * stride_k_bl + cur_head * stride_k_h + offs_d[None, :]
        k = tl.load(k_ptr, mask=token_mask[:, None] & mask_d[None, :], other=0.0)

        # QK Scaled Dot Product
        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
        qk = tl.where(token_mask, qk, -float("inf"))

        # Numerisch stabiles Online-Softmax
        m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
        p = tl.exp(qk - m_ij)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        # V-Pointer mit 2D Masking
        v_ptr = V_Cache + phys_block * stride_v_b + offs_tokens[:, None] * stride_v_bl + cur_head * stride_v_h + offs_d[None, :]
        v = tl.load(v_ptr, mask=token_mask[:, None] & mask_d[None, :], other=0.0)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_ij

    acc = acc / l_i
    out_ptr = Out + cur_batch * stride_out_b + cur_head * stride_out_h + offs_d
    tl.store(out_ptr, acc.to(Out.dtype.element_ty), mask=mask_d)


def run_pytorch_reference(q, k_cache, v_cache, block_tables, context_lens, scale, block_size):
    batch_size, num_heads, head_size = q.shape
    ref_out = torch.empty_like(q)

    for b in range(batch_size):
        c_len = context_lens[b].item()
        num_b = (c_len + block_size - 1) // block_size
        k_list, v_list = [], []
        for i in range(num_b):
            phys_b = block_tables[b, i].item()
            k_list.append(k_cache[phys_b, :block_size])
            v_list.append(v_cache[phys_b, :block_size])
            
        k_all = torch.cat(k_list, dim=0)[:c_len]
        v_all = torch.cat(v_list, dim=0)[:c_len]

        for h in range(num_heads):
            qh = q[b, h].float()
            kh = k_all[:, h].float()
            vh = v_all[:, h].float()

            scores = torch.matmul(kh, qh) * scale
            weights = torch.softmax(scores, dim=0)
            res = torch.matmul(weights, vh)
            ref_out[b, h] = res.to(torch.float16)

    return ref_out


def test_block_size(head_size=128, block_size=16):
    batch_size = 2
    num_heads = 4
    context_len = block_size * 2
    scale = 1.0 / (head_size ** 0.5)

    block_size_p2 = next_power_of_2(block_size)
    head_size_p2 = next_power_of_2(head_size)

    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, head_size, dtype=torch.float16, device="cuda")
    k_cache = torch.randn(64, block_size, num_heads, head_size, dtype=torch.float16, device="cuda")
    v_cache = torch.randn(64, block_size, num_heads, head_size, dtype=torch.float16, device="cuda")
    
    block_tables = torch.randint(0, 64, (batch_size, 4), dtype=torch.int32, device="cuda")
    context_lens = torch.tensor([context_len] * batch_size, dtype=torch.int32, device="cuda")
    out_triton = torch.empty_like(q)

    grid = (batch_size, num_heads)
    
    try:
        _paged_attn_decode_kernel_patched[grid](
            out_triton, q, k_cache, v_cache,
            block_tables, context_lens, scale,
            out_triton.stride(0), out_triton.stride(1),
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            block_tables.stride(0),
            BLOCK_SIZE=block_size,
            BLOCK_SIZE_P2=block_size_p2,
            HEAD_SIZE=head_size,
            HEAD_SIZE_P2=head_size_p2,
        )
        torch.cuda.synchronize()

        out_ref = run_pytorch_reference(q, k_cache, v_cache, block_tables, context_lens, scale, block_size)
        delta = (out_triton - out_ref).abs().max().item()

        status = "OK (Bit-Ident / In Toleranz)" if delta < 0.02 else f"DIVERGENCE ({delta:.4f})"
        print(f"[*] Head {head_size:<3} (P2: {head_size_p2:<3}) | Block {block_size:<3} (P2: {block_size_p2:<3}) | Max Delta: {delta:<10.6f} | {status}")

    except Exception as e:
        print(f"[-] Head {head_size:<3} | Block {block_size:<3} | CRASH: {e}")

if __name__ == "__main__":
    print("\n" + "="*85)
    print("VERIFIKATION NACH DEM PATCH: ALLE HEAD- & BLOCK-GRÖSSEN")
    print("="*85)
    for h in [96, 128, 192, 256]:
        for b in [16, 64, 528]:
            test_block_size(head_size=h, block_size=b)
    print("="*85)
