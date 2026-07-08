#!/usr/bin/env bash
# scripts/tool_eval_bench.sh — run the agentic tool-call benchmark against fucina.
#
# WHY max-turns 12 (not the upstream default 8): two scenarios in the v2.0.4 suite
# need a 9th turn to finish their correct tool chain (TC-46 "Deep Multi-Turn
# Research", TC-62 "6-Turn Research Chain"). At --max-turns 8 the model selects the
# right tool on every turn and is simply cut off mid-chain (empty final answer),
# losing points for a harness limit rather than a model error. 12 gives headroom
# without masking genuine runaway loops.
#
# Usage:
#   scripts/tool_eval_bench.sh                       # launch ./fucina (single-flight) + bench
#   BATCH=1 scripts/tool_eval_bench.sh --perf        # launch with continuous batching (--batch)
#   MODEL=/path/to.gguf scripts/tool_eval_bench.sh   # pick the model
#   BASE_URL=http://host:8080/v1 scripts/tool_eval_bench.sh   # use a RUNNING server (no launch)
#
# BATCH=1 routes /v1/* through the per-step scheduler (paged multi-sequence KV) so
# concurrent throughput (--perf c2/c4 columns, TTFT) stops scaling with client count.
# Leave it off for single-stream runs: the batch path has no MTP spec decode yet and
# pays a ~10% split-K tax, so default-off is faster for non-concurrent traffic.
#
# Any extra args after the script are passed straight through to tool-eval-bench
# (e.g. --perf, --short, --scenarios TC-46 TC-62, --parallel 4).
set -u

FUCINA="${FUCINA:-./fucina}"
MODEL="${MODEL:-model.gguf}"
MAX_TURNS="${MAX_TURNS:-12}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
BASE_URL="${BASE_URL:-}"
BATCH="${BATCH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/}"

srv_pid=""
cleanup() { [ -n "$srv_pid" ] && kill "$srv_pid" 2>/dev/null; }
trap cleanup EXIT

if [ -z "$BASE_URL" ]; then
  [ -x "$FUCINA" ] || { echo "tool-bench: $FUCINA not built (run: make fucina)"; exit 1; }
  [ -e "$MODEL" ]  || { echo "tool-bench: model '$MODEL' not found (set MODEL=...)"; exit 1; }

  srv_args=( -m "$MODEL" --host "$HOST" --port "$PORT" )
  if [ -n "$BATCH" ]; then
    srv_args+=( --batch )
    echo "tool-bench: continuous batching ON (--batch / paged multi-sequence KV)"
  fi
  echo "tool-bench: launching $FUCINA ${srv_args[*]}"
  "$FUCINA" "${srv_args[@]}" >/tmp/fucina_toolbench.log 2>&1 &
  srv_pid=$!
  BASE_URL="http://${HOST}:${PORT}/v1"

  printf 'tool-bench: waiting for server readiness'
  for _ in $(seq 1 600); do
    if curl -fsS "http://${HOST}:${PORT}/readyz" >/dev/null 2>&1; then printf ' ready\n'; break; fi
    if ! kill -0 "$srv_pid" 2>/dev/null; then
      printf ' FAILED\n'; echo "tool-bench: server exited early — see /tmp/fucina_toolbench.log"; exit 1
    fi
    printf '.'; sleep 1
  done
  curl -fsS "http://${HOST}:${PORT}/readyz" >/dev/null 2>&1 || {
    echo "tool-bench: server not ready after timeout — see /tmp/fucina_toolbench.log"; exit 1; }
fi

echo "tool-bench: tool-eval-bench --max-turns $MAX_TURNS --base-url $BASE_URL $*"
# Run (not exec): the EXIT trap must still fire to tear down a launched server.
tool-eval-bench --max-turns "$MAX_TURNS" --base-url "$BASE_URL" --output-dir "$OUTPUT_DIR" "$@"
