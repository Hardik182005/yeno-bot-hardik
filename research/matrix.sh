#!/bin/bash
# Robustness matrix on all real days. Usage: research/matrix.sh BOT OUTFILE
BOT=${1:-bot/yeno_bot.py}; OUT=${2:-research/results/matrix.txt}
: > "$OUT"
run() { echo "### $*" >> "$OUT"; ../.venv/bin/python sim/run.py "$BOT" "$@" > "$OUT.tmp" 2>&1; cat "$OUT.tmp" >> "$OUT"; }
run --scenario "bnf@2.0"
run --scenario "bnf@0"
run --scenario "bnf@1.0"
run --scenario "cl@0"
run --scenario "bnf_k@0"
run --scenario "bnf@2.0" --adverse 0.02
run --scenario "bnf@3.0"
run --scenario "cl@2.0"
echo DONE >> "$OUT"; rm -f "$OUT.tmp"
