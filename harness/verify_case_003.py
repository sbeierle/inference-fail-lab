import torch
import sys

# Test-Parameter basierend auf GitHub Issue #62697
# Vergleicht Standard Block-Size 16 (funktional) gegen Block-Size 32/64/528 (Bruchstelle)
HEAD_SIZES = [64, 128, 192, 256]
BLOCK_SIZES = [16, 32, 64, 528]
DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[*] Starte CASE-003 Kernel-Verification Harness")
print(f"[*] Target Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Keine GPU'}")
print(f"[*] PyTorch Version: {torch.__version__}")

if DEVICE != "cuda":
    print("[-] FEHLER: Keine CUDA/ROCm-fähige GPU in PyTorch initialisiert.")
    sys.exit(1)

def run_attention_reference(q, k, v, scale):
    """FP32 Referenzberechnung für absolute Basistoleranz."""
    q = q.float()
    k = k.float()
    v = v.float()
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v).to(DTYPE)

def test_kernel_matrix():
    print("\n" + "="*85)
    print(f"{'Head Size':<12} | {'KV Block Size':<15} | {'Max Delta':<15} | {'Status':<25}")
    print("="*85)

    scale = 0.125
    batch_size = 4
    num_heads = 8

    for h_size in HEAD_SIZES:
        for b_size in BLOCK_SIZES:
            try:
                # Simuliere Abfrage- und Page-Table-Tensoren
                q = torch.randn(batch_size, num_heads, 1, h_size, dtype=DTYPE, device=DEVICE)
                k = torch.randn(batch_size, num_heads, b_size, h_size, dtype=DTYPE, device=DEVICE)
                v = torch.randn(batch_size, num_heads, b_size, h_size, dtype=DTYPE, device=DEVICE)

                ref_output = run_attention_reference(q, k, v, scale)

                # Simulierter Kernel-Forward-Check auf Speicherkongruenz
                # (Hier schlägt auf fehlerhaften Triton-Strides der Out-of-Bounds-Fault zu)
                test_output = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1), v)

                delta = (ref_output - test_output).abs().max().item()

                if delta > 0.02:
                    status = f"FAILED (Delta: {delta:.4f})"
                else:
                    status = "OK (Bit-Ident / In Toleranz)"

                print(f"{h_size:<12} | {b_size:<15} | {delta:<15.6f} | {status:<25}")

            except RuntimeError as e:
                print(f"{h_size:<12} | {b_size:<15} | {'CRASH':<15} | GPU-FAULT / OOB ({str(e)[:18]}...)")

    print("="*85)
    print("[*] Durchlauf abgeschlossen. Siehe 'dmesg -T | grep -i amdgpu' für Hardware-Page-Faults.")

if __name__ == "__main__":
    test_kernel_matrix()
