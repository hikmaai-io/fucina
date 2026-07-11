#!/usr/bin/env bash
# ABOUTME: Long-session restart gate — save a session in one fucina process,
# ABOUTME: restart, /load, and continue with ZERO re-prefill of the saved
# ABOUTME: prefix (asserted from the REPL's own prefill counters).
#
# Drives the paged Qwen3.5 hybrid REPL end-to-end: the session snapshot must
# carry the GDN recurrent state + conv rings or the resumed turn would emit
# garbage / cold-prefill. PASS requires, on the post-restart turn:
#   prefill T tokens (R from session, N new)  with R ≥ MIN_RESTORED and
#   N ≤ MAX_NEW (only the new user turn), R + N == T.
#
# Usage: scripts/test_session_restart.sh [MODEL_DIR]
set -euo pipefail

MODEL=${1:-/opt/spark/models/models--Qwen--Qwen3.5-9B-FP8}
BIN=${BIN:-./fucina}
MIN_RESTORED=${MIN_RESTORED:-400}
MAX_NEW=${MAX_NEW:-120}

WORK=$(mktemp -d /tmp/fucina-session-restart.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
SESS="$WORK/long.fcsess"
LOG1="$WORK/run1.log" LOG2="$WORK/run2.log"

# A long first turn so the saved session comfortably clears MIN_RESTORED
# tokens: a filler paragraph the model is asked to acknowledge briefly.
FILLER=$(printf 'The quick brown fox jumps over the lazy dog near the riverbank at dawn. %.0s' $(seq 1 40))
TURN1="Here is a passage: $FILLER Reply with the single word OK."

echo "== run 1: converse + /save (model: $MODEL)"
printf '%s\n/save %s\n/quit\n' "$TURN1" "$SESS" \
    | "$BIN" -m "$MODEL" --interactive --temp 0 -n 48 >"$WORK/out1.txt" 2>"$LOG1" \
    || { echo "FAIL — run 1 exited nonzero"; tail -20 "$LOG1"; exit 1; }
grep -q "session saved to $SESS" "$LOG1" \
    || { echo "FAIL — /save did not report success"; tail -20 "$LOG1"; exit 1; }
[ -s "$SESS" ] || { echo "FAIL — session file missing/empty"; exit 1; }
echo "   session file: $(stat -c%s "$SESS") bytes"

echo "== run 2: restart + /load + continue"
printf '/load %s\nNow reply with the single word DONE.\n/quit\n' "$SESS" \
    | "$BIN" -m "$MODEL" --interactive --temp 0 -n 48 >"$WORK/out2.txt" 2>"$LOG2" \
    || { echo "FAIL — run 2 exited nonzero"; tail -20 "$LOG2"; exit 1; }
grep -q "session loaded from $SESS" "$LOG2" \
    || { echo "FAIL — /load did not report success"; tail -20 "$LOG2"; exit 1; }

# The post-restart turn's prefill counter line:
#   fucina: prefill T tokens (R from session, N new) ...
LINE=$(grep -o 'prefill [0-9]* tokens ([0-9]* from session, [0-9]* new)' "$LOG2" | tail -1)
[ -n "$LINE" ] || { echo "FAIL — no session prefill counter line"; tail -20 "$LOG2"; exit 1; }
T=$(echo "$LINE" | awk '{print $2}')
R=$(echo "$LINE" | sed 's/.*(\([0-9]*\) from session.*/\1/')
N=$(echo "$LINE" | sed 's/.*, \([0-9]*\) new).*/\1/')
echo "   $LINE"

[ "$((R + N))" -eq "$T" ] || { echo "FAIL — counters inconsistent ($R + $N != $T)"; exit 1; }
[ "$R" -ge "$MIN_RESTORED" ] || { echo "FAIL — only $R tokens restored (< $MIN_RESTORED): session did not resume"; exit 1; }
[ "$N" -le "$MAX_NEW" ] || { echo "FAIL — $N tokens re-prefilled (> $MAX_NEW): saved prefix was not free"; exit 1; }

echo "PASS — restart resumed $R tokens with zero re-prefill of the saved prefix ($N new-turn tokens only)"
