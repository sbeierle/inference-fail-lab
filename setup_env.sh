#!/bin/bash
# Zwingt ROCm/PyTorch dazu, RDNA3 (gfx1100) wie eine unterstützte Architektur zu behandeln
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Optional: Triton Cache isolieren, um verunreinigte Kernel-Artefakte auszuschließen
export TRITON_CACHE_DIR=~/inference-fail-lab/.triton_cache

echo "[+] RDNA3 Environment aktiviert (HSA_OVERRIDE_GFX_VERSION=11.0.0)"
rocminfo | grep -m 1 "gfx1100" || echo "[!] Warnung: gfx1100 nicht direkt via rocminfo erkannt. Prüfe dmesg/rocm-Installation."
