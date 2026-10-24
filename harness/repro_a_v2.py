#!/usr/bin/env python3
"""
repro_a_v2.py - check for vLLM issue #62697 "Pattern A" (decode read path), mirrors the launcher call.

What is different from v1 (repro_rocm_attn_ref.py):
  * grid = (num_seqs, num_kv_heads), GQA (e.g. 32 q / 8 kv heads), num_queries_per_kv_padded >= 16
  * V cache layout [blocks, kv_heads, head_size, block_size], INTERLEAVED_V_PACK_FACTOR=0 (as in the issue)
  * filter_by_query_len=True with a real query_start_loc, sink/alibi = None
  * NO num_warps / num_stages override (the launcher does not pass them)
  * shuffled block table, context length from the command line, optional ragged lengths
  * unused cache slots hold garbage (finite, or NaN with --nan-fill), output pre-filled with NaN
  * layout self-check (cache vs the kernel's own address formulas) BEFORE the kernel runs
  * sensitivity check: one cache block is corrupted on purpose, the harness MUST flag it
  * first call vs second call compared bitwise (covers first-call non-determinism)

Exit codes: 0 PASS | 10 MISMATCH vs fp32 | 11 harness not sensitive | 12 layout self-check failed
            13 non-deterministic | anything else: crash (e.g. GPU fault)
"""
import argparse
import math
import subprocess
import sys

import torch

CASES = {
    "ok-d64-bs528":       dict(head=64,  block=528, dtype=torch.bfloat16, hq=32, hkv=8),
    "ok-d256-bs16-gqa4":  dict(head=256, block=16,  dtype=torch.bfloat16, hq=32, hkv=8),
    "p1-d96-bs528":       dict(head=96,  block=528, dtype=torch.bfloat16, hq=32, hkv=8),
    "p1-d192-bs528":      dict(head=192, block=528, dtype=torch.bfloat16, hq=32, hkv=8),
    "p1-d256-bs528":      dict(head=256, block=528, dtype=torch.bfloat16, hq=16, hkv=2),
    "p1-d128-bs528-fp16": dict(head=128, block=528, dtype=torch.float16,  hq=32, hkv=8),
}
# power-of-two physical blocks >= 32 with padded head >= 128 (tile = min(block,128), i.e. 32/64/128)
for _d in (128, 192, 256):
    for _b in (32, 64, 128, 256):
        CASES[f"pw2-d{_d}-bs{_b}"] = dict(head=_d, block=_b, dtype=torch.bfloat16, hq=32, hkv=8)
CASES["pw2-d256-bs64-mha"] = dict(head=256, block=64, dtype=torch.bfloat16, hq=8, hkv=8)
CASES["pw2-d128-bs64-gqa2"] = dict(head=128, block=64, dtype=torch.bfloat16, hq=16, hkv=8)

TOL_ABS = 0.02
TOL_REL = 0.02


def p(msg):
    print(f"[A2] {msg}", flush=True)


def is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


def git_info(path):
    def run(*a):
        try:
            return subprocess.check_output(["git", "-C", path, *a], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    return run("rev-parse", "HEAD"), run("status", "--short")


def make_problem(cfg, ctx, seqs, identity, ragged, nan_fill, device):
    D, B, dtype, Hq, Hkv = cfg["head"], cfg["block"], cfg["dtype"], cfg["hq"], cfg["hkv"]
    x = 16 // torch.empty((), dtype=dtype).element_size()
    assert D % x == 0, "head size must be a multiple of x"

    lens = [max(1, ctx - 13 * i) if ragged else ctx for i in range(seqs)]
    nblk = [(L + B - 1) // B for L in lens]
    max_blk = max(nblk)
    total = sum(nblk)
    pool = total + 3
    perm = list(range(pool)) if identity else torch.randperm(pool).tolist()
    ids = perm[:total]

    bt = torch.zeros((seqs, max_blk), dtype=torch.int32)
    pos = 0
    for s in range(seqs):
        for j in range(nblk[s]):
            bt[s, j] = ids[pos]
            pos += 1

    k_shape = (pool, Hkv, D // x, B, x)
    v_shape = (pool, Hkv, D, B)
    if nan_fill:
        k_cache = torch.full(k_shape, float("nan"), dtype=dtype, device=device)
        v_cache = torch.full(v_shape, float("nan"), dtype=dtype, device=device)
    else:
        k_cache = torch.randn(k_shape, dtype=dtype, device=device) * 10
        v_cache = torch.randn(v_shape, dtype=dtype, device=device) * 10

    k_full = torch.randn((seqs, max(lens), Hkv, D), dtype=dtype, device=device)
    v_full = torch.randn((seqs, max(lens), Hkv, D), dtype=dtype, device=device)
    q = torch.randn((seqs, Hq, D), dtype=dtype, device=device)

    for s in range(seqs):
        for j in range(nblk[s]):
            phys = int(bt[s, j])
            t0 = j * B
            t1 = min(t0 + B, lens[s])
            n = t1 - t0
            k_cache[phys, :, :, :n, :] = k_full[s, t0:t1].reshape(n, Hkv, D // x, x).permute(1, 2, 0, 3)
            v_cache[phys, :, :, :n] = v_full[s, t0:t1].permute(1, 2, 0)

    return dict(q=q, k_full=k_full, v_full=v_full, k_cache=k_cache, v_cache=v_cache,
                bt=bt, lens=lens, nblk=nblk, x=x)


def layout_selfcheck(pr, cfg, device, n=2048):
    """Read the cache with the kernel's own address formulas and compare with the dense data."""
    D, B, Hkv = cfg["head"], cfg["block"], cfg["hkv"]
    kc, vc, x = pr["k_cache"], pr["v_cache"], pr["x"]
    lens, bt = pr["lens"], pr["bt"]
    seqs = len(lens)
    g = torch.Generator().manual_seed(1)
    s = torch.randint(0, seqs, (n,), generator=g)
    lens_t = torch.tensor(lens)
    t = (torch.rand(n, generator=g) * lens_t[s]).long()
    h = torch.randint(0, Hkv, (n,), generator=g)
    d = torch.randint(0, D, (n,), generator=g)
    phys = bt[s, t // B].long()
    internal = t % B
    sk, sv = kc.stride(), vc.stride()
    k_off = phys * sk[0] + h * sk[1] + (d // x) * sk[2] + internal * sk[3] + (d % x) * sk[4]
    v_off = phys * sv[0] + h * sv[1] + d * sv[2] + internal * sv[3]
    got_k = kc.reshape(-1)[k_off.to(device)]
    got_v = vc.reshape(-1)[v_off.to(device)]
    sd, td, hd, dd = s.to(device), t.to(device), h.to(device), d.to(device)
    exp_k = pr["k_full"][sd, td, hd, dd]
    exp_v = pr["v_full"][sd, td, hd, dd]
    return bool(torch.equal(got_k, exp_k) and torch.equal(got_v, exp_v))


def reference(q, k_full, v_full, lens, Hkv, sm_scale):
    seqs, Hq, D = q.shape
    G = Hq // Hkv
    outs = []
    for s in range(seqs):
        L = lens[s]
        K = k_full[s, :L].float()
        V = v_full[s, :L].float()
        qs = q[s].float().reshape(Hkv, G, D)
        scores = torch.einsum("kgd,lkd->kgl", qs, K) * sm_scale
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("kgl,lkd->kgd", probs, V).reshape(Hq, D))
    return torch.stack(outs)


def launch(kernel, triton, out, pr, cfg, qsl, seq_lens, sm_scale):
    D, B, Hq, Hkv = cfg["head"], cfg["block"], cfg["hq"], cfg["hkv"]
    G = Hq // Hkv
    q, kc, vc, bt = pr["q"], pr["k_cache"], pr["v_cache"], pr["bt_dev"]
    tile = min(B, 128) if is_pow2(B) else 32
    kernel[(q.shape[0], Hkv)](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=kc,
        value_cache_ptr=vc,
        sink_ptr=None,
        block_tables_ptr=bt,
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=None,
        scale=sm_scale,
        k_scale=1.0,
        v_scale=1.0,
        out_scale_inv=1.0,
        num_query_heads=Hq,
        num_queries_per_kv=G,
        num_queries_per_kv_padded=max(triton.next_power_of_2(G), 16),
        block_table_stride=bt.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=tile,
        PHYSICAL_BLOCK_SIZE=vc.shape[3],
        HEAD_SIZE=D,
        HEAD_SIZE_PADDED=triton.next_power_of_2(D),
        USE_ALIBI_SLOPES=False,
        SLIDING_WINDOW=0,
        x=kc.shape[4],
        stride_k_cache_0=kc.stride(0),
        stride_k_cache_1=kc.stride(1),
        stride_k_cache_2=kc.stride(2),
        stride_k_cache_3=kc.stride(3),
        stride_k_cache_4=kc.stride(4),
        stride_v_cache_0=vc.stride(0),
        stride_v_cache_1=vc.stride(1),
        stride_v_cache_2=vc.stride(2),
        stride_v_cache_3=vc.stride(3),
        filter_by_query_len=True,
        query_start_len_ptr=qsl,
        USE_SINKS=False,
        USE_FP8=False,
        INTERLEAVED_V_PACK_FACTOR=0,
    )
    torch.cuda.synchronize()
    return tile


def violations(out, ref):
    err = (out.float() - ref).abs()
    tol = TOL_ABS + TOL_REL * ref.abs()
    bad = ~(err <= tol)  # NaN counts as a violation
    err_inf = torch.where(torch.isnan(err), torch.full_like(err, float("inf")), err)
    return bad, err_inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=list(CASES.keys()))
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--seqs", type=int, default=1)
    ap.add_argument("--identity", action="store_true", help="identity block table (default: shuffled)")
    ap.add_argument("--ragged", action="store_true", help="slightly different length per sequence")
    ap.add_argument("--nan-fill", action="store_true", help="fill unused cache slots with NaN")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vllm-path", default="/home/predator/inference-fail-lab/vllm_upstream")
    args = ap.parse_args()

    sys.path.insert(0, args.vllm_path)
    import triton
    from vllm.v1.attention.ops.chunked_prefill_paged_decode import kernel_paged_attention_2d

    cfg = CASES[args.case]
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    commit, status = git_info(args.vllm_path)
    p(f"env: torch {torch.__version__} | triton {triton.__version__} | hip {torch.version.hip} | {torch.cuda.get_device_name(0)}")
    p(f"vllm commit {commit} | local changes: {status.replace(chr(10), '; ') or 'none'}")
    p(f"case={args.case} ctx={args.ctx} seqs={args.seqs} q/kv heads={cfg['hq']}/{cfg['hkv']} "
      f"head={cfg['head']} block={cfg['block']} dtype={cfg['dtype']} "
      f"table={'identity' if args.identity else 'shuffled'} ragged={args.ragged} nan_fill={args.nan_fill}")

    pr = make_problem(cfg, args.ctx, args.seqs, args.identity, args.ragged, args.nan_fill, device)
    pr["bt_dev"] = pr["bt"].to(device)
    seq_lens = torch.tensor(pr["lens"], dtype=torch.int32, device=device)
    qsl = torch.arange(args.seqs + 1, dtype=torch.int32, device=device)
    sm_scale = 1.0 / math.sqrt(cfg["head"])

    ok_layout = layout_selfcheck(pr, cfg, device)
    p(f"layout self-check (cache vs kernel address formulas): {'PASS' if ok_layout else 'FAIL'}")
    if not ok_layout:
        p("RESULT HARNESS-BUG: cache layout does not match the kernel's addressing. Not a kernel finding.")
        sys.exit(12)

    ref = reference(pr["q"], pr["k_full"], pr["v_full"], pr["lens"], cfg["hkv"], sm_scale)

    out1 = torch.full_like(pr["q"], float("nan"))
    tile = launch(kernel_paged_attention_2d, triton, out1, pr, cfg, qsl, seq_lens, sm_scale)
    out2 = torch.full_like(pr["q"], float("nan"))
    launch(kernel_paged_attention_2d, triton, out2, pr, cfg, qsl, seq_lens, sm_scale)
    p(f"launch: grid=({args.seqs},{cfg['hkv']}) tile(BLOCK_SIZE)={tile} PHYSICAL_BLOCK_SIZE={cfg['block']} "
      f"HEAD_SIZE_PADDED={triton.next_power_of_2(cfg['head'])} "
      f"q_padded={max(triton.next_power_of_2(cfg['hq'] // cfg['hkv']), 16)} V=[blocks,kv_heads,head,block] interleave=0")

    deterministic = torch.equal(out1.view(torch.int16), out2.view(torch.int16))
    nan_out = int(torch.isnan(out1).sum().item())
    bad, err = violations(out1, ref)
    n_bad = int(bad.sum().item())
    max_err = float(err.max().item())
    s_i, h_i, d_i = [int(v) for v in torch.unravel_index(torch.argmax(err), err.shape)]
    bad_heads = sorted(set(bad.any(dim=2).nonzero()[:, 1].tolist())) if n_bad else []
    p(f"first call vs second call bitwise identical: {deterministic}")
    p(f"NaN in output: {nan_out} | max abs err vs fp32: {max_err:.6f} at (seq {s_i}, q-head {h_i}) | "
      f"values outside tolerance: {n_bad}/{bad.numel()}")
    if bad_heads:
        p(f"q-heads with violations: {bad_heads}")

    # Sensitivity check: corrupt one cache block of sequence 0 on purpose. The harness must notice.
    jb = min(1, pr["nblk"][0] - 1)
    phys = int(pr["bt"][0, jb])
    pr["k_cache"][phys] *= 4
    pr["v_cache"][phys] += 2
    out3 = torch.full_like(pr["q"], float("nan"))
    launch(kernel_paged_attention_2d, triton, out3, pr, cfg, qsl, seq_lens, sm_scale)
    bad3, err3 = violations(out3, ref)
    sensitive = bool(bad3.any().item())
    p(f"sensitivity check (one corrupted cache block): harness {'FLAGS it' if sensitive else 'DOES NOT FLAG it'} "
      f"(max err {float(err3.max().item()):.4f})")

    if not sensitive:
        p("RESULT HARNESS-NOT-SENSITIVE: this configuration cannot detect corruption. Do not interpret.")
        sys.exit(11)
    if not deterministic:
        p("RESULT NON-DETERMINISTIC: first and second call differ.")
        sys.exit(13)
    if n_bad or nan_out:
        p("RESULT MISMATCH: kernel output outside the issue's tolerance (harness passed its own checks).")
        sys.exit(10)
    p("RESULT PASS: within tolerance, deterministic, harness sensitive.")


if __name__ == "__main__":
    main()
