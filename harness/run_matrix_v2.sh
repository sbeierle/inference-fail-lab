#!/usr/bin/env bash
# Runs the whole matrix. Each run is its own process, so a GPU fault only kills one run.
# Usage: bash ~/inference-fail-lab/harness/run_matrix_v2.sh
LAB="${LAB:-$HOME/inference-fail-lab}"
if [ ! -f "$LAB/harness/repro_a_v2.py" ]; then
  echo "FEHLER: $LAB/harness/repro_a_v2.py nicht gefunden. Beide Dateien nach $LAB/harness/ kopieren." >&2
  exit 1
fi
cd "$LAB" || exit 1

OUT="$LAB/${OUTNAME:-gfx1100_matrix_v2.txt}"
LOG="/tmp/a2_run.log"
: > "$OUT"

CASES="${CASES:-ok-d64-bs528 ok-d256-bs16-gqa4 p1-d96-bs528 p1-d192-bs528 p1-d256-bs528 p1-d128-bs528-fp16}"
CTXS="${CTXS:-527 528 529 1024 8192 32768}"

run_one() {
  echo "### $*" | tee -a "$OUT"
  HIP_VISIBLE_DEVICES=0 python3 harness/repro_a_v2.py "$@" > "$LOG" 2>&1
  rc=$?
  grep '^\[A2\]' "$LOG" | tee -a "$OUT"
  echo "[A2] exit code $rc" | tee -a "$OUT"
  if [ "$rc" -ne 0 ] && { [ "$rc" -lt 10 ] || [ "$rc" -gt 13 ]; }; then
    echo "[A2] crash, last log lines:" | tee -a "$OUT"
    tail -n 6 "$LOG" | tee -a "$OUT"
  fi
  echo | tee -a "$OUT"
}

for c in $CASES; do
  for n in $CTXS; do
    run_one --case "$c" --ctx "$n"
  done
done

# issue-like batches: 8 sequences at 8192 tokens
if [ -z "$SKIP_BATCHES" ]; then
run_one --case p1-d256-bs528 --seqs 8 --ctx 8192
run_one --case p1-d96-bs528 --seqs 8 --ctx 8192 --ragged
fi

echo "=================== SUMMARY ===================" | tee -a "$OUT"
grep -E '^###|\[A2\] RESULT|\[A2\] exit code' "$OUT" | paste - - - | sed 's/\[A2\] //g' | tee -a "$OUT"
echo
echo "Full log: $(pwd)/$OUT"
