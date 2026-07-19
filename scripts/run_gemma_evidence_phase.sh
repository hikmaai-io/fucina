#!/usr/bin/env bash
# Run one serialized three-start Gemma evidence phase. No runtime source is changed.
set -euo pipefail

MODE=${1:?usage: run_gemma_evidence_phase.sh MODE [STARTS] [START_ID_BEGIN]}
STARTS=${2:-3}
START_ID_BEGIN=${3:-1}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/benchmark-evidence/results/2026-07-20-gemma-gb10/raw"
mkdir -p "$OUT"
DENSE_Q4=/opt/spark/models/hub/models--google--gemma-4-12B-it-qat-q4_0-gguf/snapshots/f6e7774e6148da3b7f201e42ba37cf084c1db35f/gemma-4-12b-it-qat-q4_0.gguf
DENSE_NVFP4=/opt/spark/models/hub/models--RedHatAI--gemma-4-12B-it-NVFP4/snapshots/a1d2478a9b99cc444bc9f64839609d3a82ca9195
DENSE_BF16=/opt/spark/models/hub/models--google--gemma-4-12B-IT/snapshots/5926caa4ec0cac5cbfadaf4077420520de1d5205
DENSE_MTP=/opt/spark/models/assistants/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf
E4B_BF16=/opt/spark/models/hub/models--google--gemma-4-E4B-it/snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187
IMAGE=vllm/vllm-openai@sha256:9c719fc0c869092c7d0533f8357d6985a38d5ff03b20ffb6a4620c2b4806dd4b
export GPU_CLOCK_MAX=2400

case "$MODE" in
  fucina-dense-q4-plain)
    ENGINE=fucina; MODEL_PATH=$DENSE_Q4; MODEL_ID=gemma-4-12b-it-qat-q4_0; PORT=18080; PARALLEL=32
    FUCINA_ENV=(FUCINA_NO_BATCH_SPEC=1); FUCINA_EXTRA=(--batch --spec=false)
    ;;
  fucina-dense-q4-mtp)
    ENGINE=fucina; MODEL_PATH=$DENSE_Q4; MODEL_ID=gemma-4-12b-it-qat-q4_0; PORT=18080; PARALLEL=32
    FUCINA_ENV=(); FUCINA_EXTRA=(--batch --assistant "$DENSE_MTP")
    ;;
  fucina-dense-nvfp4-plain)
    ENGINE=fucina; MODEL_PATH=$DENSE_NVFP4; MODEL_ID=gemma-4-12B-it-NVFP4; PORT=18080; PARALLEL=32
    FUCINA_ENV=(FUCINA_NO_BATCH_SPEC=1); FUCINA_EXTRA=(--batch --spec=false)
    ;;
  fucina-dense-nvfp4-mtp)
    ENGINE=fucina; MODEL_PATH=$DENSE_NVFP4; MODEL_ID=gemma-4-12B-it-NVFP4; PORT=18080; PARALLEL=32
    FUCINA_ENV=(); FUCINA_EXTRA=(--batch --assistant "$DENSE_MTP")
    ;;
  fucina-e4b-bf16-plain)
    ENGINE=fucina; MODEL_PATH=$E4B_BF16; MODEL_ID=gemma-4-E4B-it; PORT=18080; PARALLEL=8
    FUCINA_ENV=(FUCINA_NO_BATCH_SPEC=1); FUCINA_EXTRA=(--batch --spec=false)
    ;;
  vllm-dense-bf16-plain)
    ENGINE=vllm; MODEL_PATH=$DENSE_BF16; MODEL_ID=gemma-4-12B-IT; PORT=18081; PARALLEL=32; VLLM_KV=fp8
    VLLM_EXTRA=()
    ;;
  vllm-dense-nvfp4-plain)
    ENGINE=vllm; MODEL_PATH=$DENSE_NVFP4; MODEL_ID=gemma-4-12B-it-NVFP4; PORT=18081; PARALLEL=32; VLLM_KV=fp8
    VLLM_EXTRA=()
    ;;
  vllm-e4b-bf16-plain)
    ENGINE=vllm; MODEL_PATH=$E4B_BF16; MODEL_ID=gemma-4-E4B-it; PORT=18081; PARALLEL=8; VLLM_KV=auto
    VLLM_EXTRA=(--kv-sharing-fast-prefill)
    ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

quiescent() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ -n "$apps" ]]; then
    echo "GPU NOT QUIESCENT:" >&2; echo "$apps" >&2; return 1
  fi
  if pgrep -af '(/fucina( |$)|vllm( +serve|\.entrypoints)|api_server)' | grep -v -E "run_gemma_evidence_phase|pgrep"; then
    echo "inference process exists" >&2; return 1
  fi
}

wait_ready() {
  local url=$1 process_pid=$2 deadline=$((SECONDS+1800))
  until curl -fsS "$url/readyz" >/dev/null 2>&1 || curl -fsS "$url/health" >/dev/null 2>&1; do
    kill -0 "$process_pid" 2>/dev/null || return 1
    if (( SECONDS >= deadline )); then return 1; fi
    sleep 1
  done
}

run_bench() {
  local start=$1 startup_ms=$2 rss=$3 avail=$4
  python3 "$ROOT/scripts/gemma_evidence_bench.py" \
    --base-url "http://127.0.0.1:$PORT" --model "$MODEL_ID" \
    --engine "$ENGINE" --mode "$MODE" --start-id "$start" --suite all \
    --concurrency 1,2,4,8,16,32 --max-tokens 128 --long-words 2570 \
    --long-reps 2 --burst-reps 1 --startup-ms "$startup_ms" \
    --first-ready-rss-bytes "$rss" --physical-available-bytes "$avail" \
    --out "$OUT/$MODE-start${start}.json"
}

run_fucina_start() {
  local start=$1 log="$OUT/$MODE-start${start}.server.log"
  quiescent
  # Hold the repository-wide fucina GPU lock for load, benchmark, and teardown.
  exec 9>/tmp/fucina_gpu.lock
  flock -w 1200 9
  local t0 t1 pid rss avail rc=0
  t0=$(date +%s%N)
  env FUCINA_PAGED_MAXSEQS=$PARALLEL "${FUCINA_ENV[@]}" "$ROOT/fucina" -m "$MODEL_PATH" \
    --ctx 16384 --parallel "$PARALLEL" --max-concurrent 64 --gpu-mem-util 0.90 \
    --temp 0 --host 127.0.0.1 --port "$PORT" "${FUCINA_EXTRA[@]}" >"$log" 2>&1 &
  pid=$!
  trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; flock -u 9' RETURN
  wait_ready "http://127.0.0.1:$PORT" "$pid" || { tail -200 "$log"; return 1; }
  t1=$(date +%s%N)
  rss=$(awk '/VmRSS:/ {print $2*1024}' "/proc/$pid/status")
  avail=$(awk '/MemAvailable:/ {print $2*1024}' /proc/meminfo)
  nvidia-smi -q -d CLOCK,PERFORMANCE > "$OUT/$MODE-start${start}.nvidia-smi.txt"
  run_bench "$start" "$(( (t1-t0)/1000000 ))" "$rss" "$avail" || rc=$?
  curl -fsS "http://127.0.0.1:$PORT/metrics" > "$OUT/$MODE-start${start}.metrics.final" || true
  kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
  flock -u 9
  trap - RETURN
  sleep 3
  quiescent
  return "$rc"
}

run_vllm_start() {
  local start=$1 name="gemma-evidence-${MODE}-${start}" log="$OUT/$MODE-start${start}.server.log"
  quiescent
  docker rm -f "$name" >/dev/null 2>&1 || true
  local t0 t1 hostpid rss avail rc=0
  t0=$(date +%s%N)
  docker run --name "$name" --gpus all --ipc=host --network host \
    -e GPU_CLOCK_MAX=2400 -v /opt/spark/models:/opt/spark/models:ro \
    "$IMAGE" "$MODEL_PATH" --language-model-only \
    --host 127.0.0.1 --port "$PORT" --served-model-name "$MODEL_ID" \
    --max-model-len 16384 --max-num-seqs "$PARALLEL" --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.90 --enable-prefix-caching --kv-cache-dtype "$VLLM_KV" \
    --attention-backend TRITON_ATTN "${VLLM_EXTRA[@]}" >"$log" 2>&1 &
  local docker_cli=$!
  trap 'docker rm -f "$name" >/dev/null 2>&1 || true; wait "$docker_cli" 2>/dev/null || true' RETURN
  wait_ready "http://127.0.0.1:$PORT" "$docker_cli" || { tail -300 "$log"; return 1; }
  t1=$(date +%s%N)
  hostpid=$(docker inspect -f '{{.State.Pid}}' "$name")
  rss=$(awk '/VmRSS:/ {print $2*1024}' "/proc/$hostpid/status")
  avail=$(awk '/MemAvailable:/ {print $2*1024}' /proc/meminfo)
  nvidia-smi -q -d CLOCK,PERFORMANCE > "$OUT/$MODE-start${start}.nvidia-smi.txt"
  run_bench "$start" "$(( (t1-t0)/1000000 ))" "$rss" "$avail" || rc=$?
  curl -fsS "http://127.0.0.1:$PORT/metrics" > "$OUT/$MODE-start${start}.metrics.final" || true
  docker rm -f "$name" >/dev/null 2>&1 || true
  wait "$docker_cli" 2>/dev/null || true
  trap - RETURN
  sleep 5
  quiescent
  return "$rc"
}

cd "$ROOT"
START_ID_END=$((START_ID_BEGIN + STARTS - 1))
for start in $(seq "$START_ID_BEGIN" "$START_ID_END"); do
  echo "[$(date -Is)] $MODE independent start $start ($START_ID_BEGIN..$START_ID_END)"
  if [[ $ENGINE == fucina ]]; then run_fucina_start "$start"; else run_vllm_start "$start"; fi
done
