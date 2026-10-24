#!/bin/bash
echo "[*] Überwache Linux-Kernelringpuffer auf AMDGPU Faults (Strg+C zum Beenden)..."
sudo dmesg -wT | grep -iE "amdgpu|page fault|gpu reset|ring gfx"
