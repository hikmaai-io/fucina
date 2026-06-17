#!/usr/bin/env bash
# scripts/bench.sh — fucina correctness + performance smoke (GB10).
# Invoked by `make bench` (builds fucina first). Runs the engine's own
# self-tests for correctness and a few one-shot generations for throughput.
#
#   make bench                         # uses ./model.gguf
#   make bench MODEL=/path/to.gguf
#
# Correctness gates are HARD: any failure exits non-zero so the target fails.
# Performance numbers are reported, not asserted (they vary with the box).
set -u

FUCINA="${FUCINA:-./fucina}"
MODEL="${MODEL:-model.gguf}"
fail=0

line() { printf '%s\n' "──────────────────────────────────────────────────────────────"; }
hdr()  { line; printf '  %s\n' "$1"; line; }

if [ ! -x "$FUCINA" ]; then echo "bench: $FUCINA not built"; exit 1; fi
if [ ! -f "$MODEL" ]; then echo "bench: model '$MODEL' not found (set MODEL=...)"; exit 1; fi

hdr "CORRECTNESS 1 — multi-seq batch == single decodes + sampling (engine self-test)"
out=$(FUCINA_PAGED_KV=1 FUCINA_BATCH_SELFTEST=1 "$FUCINA" -m "$MODEL" --prompt "x" --predict 1 2>&1)
echo "$out" | grep -iE "self-test seq|self-test PASSED|sampling .* PASSED" || true
echo "$out" | grep -qi "batch self-test PASSED" || { echo "  !! batch self-test FAILED"; fail=1; }
echo "$out" | grep -qi "sampling .* PASSED"     || { echo "  !! sampling self-test FAILED"; fail=1; }

hdr "CORRECTNESS 2 — greedy byte-identical: default vs continuous-batch path"
A=$("$FUCINA" -m "$MODEL" --prompt "The capital of France is" --predict 12 --temp 0 2>/dev/null | grep -oE "France is.*" | head -1)
B=$(FUCINA_PAGED_KV=1 FUCINA_BATCH=1 "$FUCINA" -m "$MODEL" --prompt "The capital of France is" --predict 12 --temp 0 2>/dev/null | grep -oE "France is.*" | head -1)
printf '  default: %s\n  batch  : %s\n' "$A" "$B"
[ -n "$A" ] && [ "$A" = "$B" ] && echo "  OK (byte-identical)" || { echo "  !! MISMATCH"; fail=1; }

hdr "PERF 1 — prefill throughput (long prompt)"
LONG=$(printf 'The quick brown fox jumps over the lazy dog. Compute and data drive progress. %.0s' $(seq 1 40))
"$FUCINA" -m "$MODEL" --prompt "$LONG" --predict 1 --temp 0 --spec=false 2>&1 \
  | grep -oE "prefill [0-9]+ tokens in [0-9.]+s \([0-9.]+ tok/s\)" | sed 's/^/  /'

hdr "PERF 2 — decode throughput (256 tokens)"
printf '  spec OFF: '; "$FUCINA" -m "$MODEL" --prompt "Once upon a time" --predict 256 --temp 0 --spec=false 2>&1 \
  | grep -oE "generated 256 tokens in [0-9.]+s \([0-9.]+ tok/s\)"
printf '  spec ON : '; "$FUCINA" -m "$MODEL" --prompt "Once upon a time" --predict 256 --temp 0 2>&1 \
  | grep -oE "generated 256 tokens in [0-9.]+s \([0-9.]+ tok/s\), [0-9]+ drafts accepted \(avg [0-9.]+ tokens/step"

line
if [ "$fail" -ne 0 ]; then echo "  BENCH: correctness FAILED"; exit 1; fi
echo "  BENCH: correctness PASSED"
