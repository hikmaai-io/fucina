#!/usr/bin/env bash
# ABOUTME: Qwen HTTP-session restart gate: save through /v1/completions,
# ABOUTME: restart the server, restore the named slot, and prefill only a suffix.
set -euo pipefail

MODEL=${1:-/opt/spark/models/models--Qwen--Qwen3.5-9B-FP8}
BIN=${BIN:-./fucina}
PORT=${PORT:-18091}
WORK=$(mktemp -d /tmp/fucina-qwen-http-session.XXXXXX)
SESS_DIR="$WORK/sessions"
mkdir -p "$SESS_DIR"
PID=
cleanup() {
    if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
        kill -TERM "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

wait_ready() {
    local deadline=$((SECONDS + 600))
    until curl -fsS "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1; do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "FAIL — server exited during startup"
            tail -80 "$1"
            exit 1
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "FAIL — server did not become ready in 600s"
            tail -80 "$1"
            exit 1
        fi
        sleep 1
    done
}

start_server() {
    local log=$1
    "$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
        --session-dir "$SESS_DIR" --temp 0 >"$WORK/server.out" 2>"$log" &
    PID=$!
    wait_ready "$log"
}

stop_server() {
    kill -TERM "$PID"
    wait "$PID"
    PID=
}

# The Qwen special token creates a tokenizer-stable boundary: appending the
# second suffix cannot retokenize the saved prefix. max_tokens=1 means the first
# snapshot frontier is exactly the first prompt (the sampled token is one ahead).
PROMPT1='HTTP session restart gate: preserve this exact prefix.<|im_end|>'
PROMPT2="$PROMPT1
A new suffix after restart."
python3 - "$WORK/req1.json" "$WORK/req2.json" "$PROMPT1" "$PROMPT2" <<'PY'
import json, sys
_, out1, out2, prompt1, prompt2 = sys.argv
for path, prompt in ((out1, prompt1), (out2, prompt2)):
    with open(path, "w") as f:
        json.dump({"prompt": prompt, "session": "restart", "max_tokens": 1}, f)
PY

LOG1="$WORK/run1.log"
echo "== run 1: create HTTP Qwen session"
start_server "$LOG1"
curl -fsS -H 'Content-Type: application/json' --data-binary @"$WORK/req1.json" \
    "http://127.0.0.1:$PORT/v1/completions" >"$WORK/resp1.json"
python3 -m json.tool "$WORK/resp1.json" >/dev/null
[ -s "$SESS_DIR/restart.fcsess" ] || { echo "FAIL — first request did not create session"; exit 1; }
stop_server

LOG2="$WORK/run2.log"
echo "== run 2: restart and extend HTTP Qwen session"
start_server "$LOG2"
curl -fsS -H 'Content-Type: application/json' --data-binary @"$WORK/req2.json" \
    "http://127.0.0.1:$PORT/v1/completions" >"$WORK/resp2.json"
python3 -m json.tool "$WORK/resp2.json" >/dev/null
stop_server

LINE=$(grep -E 'batch: disk session restored \([0-9]+ cached, [0-9]+ new prompt tokens\)' "$LOG2" | tail -1 || true)
[ -n "$LINE" ] || { echo "FAIL — restart did not report disk-session restore"; tail -80 "$LOG2"; exit 1; }
R=$(echo "$LINE" | sed -E 's/.*\(([0-9]+) cached, ([0-9]+) new.*/\1/')
N=$(echo "$LINE" | sed -E 's/.*\(([0-9]+) cached, ([0-9]+) new.*/\2/')
[ "$R" -gt 0 ] || { echo "FAIL — restored token count is zero"; exit 1; }
[ "$N" -gt 0 ] || { echo "FAIL — appended suffix was not prefilled"; exit 1; }
grep -q 'session restart.fcsess saved' "$LOG2" \
    || { echo "FAIL — resumed session was not saved back"; tail -80 "$LOG2"; exit 1; }

echo "PASS — HTTP restart restored $R cached tokens and prefilled only $N new tokens"
