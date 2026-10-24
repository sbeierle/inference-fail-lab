#!/usr/bin/env bash
# Zweite Matrix: Zweierpotenz-Blockgroessen 32/64/128/256 (Tile 32/64/128) mit Head 128/192/256, plus MHA und GQA2.
export OUTNAME="gfx1100_matrix_pow2.txt"
export SKIP_BATCHES=1
export CTXS="1024 8192 32768"
export CASES="$(python3 -I - <<'PY'
d=[f"pw2-d{h}-bs{b}" for h in (128,192,256) for b in (32,64,128,256)]
print(" ".join(d+["pw2-d256-bs64-mha","pw2-d128-bs64-gqa2"]))
PY
)"
exec bash "${LAB:-$HOME/inference-fail-lab}/harness/run_matrix_v2.sh"
