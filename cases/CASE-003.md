# CASE-003: Triton Fallback Kernel Compiler Crash on Non-Power-of-2 Head Sizes & KV Blocks

## 1. Environment & Pinning
- **Engine:** vLLM `ROCM_ATTN` Triton Fallback (`_paged_attn_decode_kernel`)
- **ROCm / PyTorch:** ROCm 6.4 / PyTorch 2.9.1 / Triton 3.5.1
- **Hardware:** AMD Radeon RX 7900 XTX (RDNA3 / `gfx1100`)
- **Architektur-Invariante:** Triton Vektor-Register erzwingen strikt `range must be a power of 2`.

## 2. Failure Signature & Observability
- **Crash-Typ:** Hard JIT Compiler Exception (`arange's range must be a power of 2`).
- **Betroffene Topologien:**
  - `HEAD_SIZE = 96` (Crash bei allen Block-Größen)
  - `HEAD_SIZE = 192` (Crash bei allen Block-Größen)
  - `BLOCK_SIZE = 528` (Crash im unkorrigierten Upstream-Kernel)
- **Status:** **[gemessen: RX 7900 XTX / ROCm 6.4 / Triton 3.5.1]**

## 3. Gemessene Testmatrix

| Head Size | Block Size | Status | Ursache |
| :--- | :--- | :--- | :--- |
| **96** | 16 / 64 / 528 | ❌ **CRASH** | `tl.arange(0, 96)` ist keine 2^n |
| **128** | 16 / 64 / 528 | ✅ **OK** | 128 = 2^7 (gepadded mit P2-Masking) |
| **192** | 16 / 64 / 528 | ❌ **CRASH** | `tl.arange(0, 192)` ist keine 2^n |
| **256** | 16 / 64 / 528 | ✅ **OK** | 256 = 2^8 (native 2^n) |

## 4. Remediation & Upstream Patch
`HEAD_SIZE` und `BLOCK_SIZE` müssen im Kernel über `next_power_of_2()` gepuffert und unvollständige Vektoren über boolesche Masken (`tl.load(..., mask=...)`) entkoppelt werden.
