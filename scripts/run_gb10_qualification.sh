#!/usr/bin/env bash
# Run one benchmark claim against exactly three independently started servers.
# The benchmark command must write a top-level JSON array to $FUCINA_RAW_RESULTS.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/run_gb10_qualification.sh \
  --claim NAME --model PATH [--model PATH ...] \
  --server-command COMMAND --benchmark-command COMMAND \
  --ready-url URL [--ready-timeout SECONDS] [--allow-dirty]

Each command receives FUCINA_RUN_INDEX, FUCINA_RUN_DIR, FUCINA_RAW_RESULTS,
and FUCINA_SOURCE_COMMIT. The server is terminated after each benchmark, then
started as a new process for the next run. Evidence defaults to
benchmark-evidence/qualification/; override with EVIDENCE_ROOT for testing.
USAGE
}

claim=""
server_command=""
benchmark_command=""
ready_url=""
ready_timeout=300
allow_dirty=0
models=()
while (($#)); do
    case "$1" in
        --claim) claim=${2:?missing claim}; shift 2 ;;
        --model) models+=("${2:?missing model path}"); shift 2 ;;
        --server-command) server_command=${2:?missing server command}; shift 2 ;;
        --benchmark-command) benchmark_command=${2:?missing benchmark command}; shift 2 ;;
        --ready-url) ready_url=${2:?missing ready URL}; shift 2 ;;
        --ready-timeout) ready_timeout=${2:?missing timeout}; shift 2 ;;
        --allow-dirty) allow_dirty=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$claim" || -z "$server_command" || -z "$benchmark_command" || -z "$ready_url" || ${#models[@]} -eq 0 ]]; then
    usage >&2
    exit 2
fi
if [[ ! "$claim" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "claim must contain only letters, digits, dot, underscore, or dash" >&2
    exit 2
fi
if [[ ! "$ready_timeout" =~ ^[1-9][0-9]*$ ]]; then
    echo "ready timeout must be a positive integer" >&2
    exit 2
fi
for model in "${models[@]}"; do
    [[ -e "$model" ]] || { echo "model path does not exist: $model" >&2; exit 2; }
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
source_commit=$(git rev-parse HEAD)
source_short=$(git rev-parse --short=12 HEAD)
dirty=false
if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
    dirty=true
    if ((allow_dirty == 0)); then
        echo "refusing qualification from a dirty source tree (use --allow-dirty for harness testing only)" >&2
        exit 2
    fi
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
evidence_root=${EVIDENCE_ROOT:-"$repo_root/benchmark-evidence/qualification"}
run_root="$evidence_root/${stamp}-${claim}-${source_short}-$$"
mkdir -p "$run_root"
printf '%s\n' "$source_commit" > "$run_root/source-commit.txt"
git diff --binary HEAD > "$run_root/source-working-tree.patch"

# Hash full model contents once. The aggregate includes relative paths and file
# digests, so sharding or metadata changes alter the model fingerprint.
python3 - "$run_root/model-fingerprints.json" "$run_root/model-files.sha256" "${models[@]}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

json_path, files_path, *roots = sys.argv[1:]
records = []
lines = []
for root_arg in roots:
    root = Path(root_arg).resolve()
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                total_bytes += len(chunk)
        hex_digest = digest.hexdigest()
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(hex_digest.encode("ascii"))
        aggregate.update(b"\n")
        lines.append(f"{hex_digest}  {root_arg}/{relative}")
    records.append({
        "path": root_arg,
        "resolved_path": str(root),
        "sha256": aggregate.hexdigest(),
        "file_count": len(files),
        "bytes": total_bytes,
    })
Path(json_path).write_text(json.dumps(records, indent=2) + "\n")
Path(files_path).write_text("\n".join(lines) + "\n")
PY

capture() {
    local output=$1
    shift
    if command -v "$1" >/dev/null 2>&1; then
        "$@" > "$output" 2>&1 || true
    else
        printf 'unavailable: %s\n' "$1" > "$output"
    fi
}
capture "$run_root/nvidia-smi.txt" nvidia-smi
capture "$run_root/nvidia-smi-query.csv" nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv
capture "$run_root/nvcc-version.txt" nvcc --version
capture "$run_root/uname.txt" uname -a
if [[ -r /usr/local/cuda/version.json ]]; then
    cp /usr/local/cuda/version.json "$run_root/cuda-version.json"
else
    printf '{"status":"unavailable"}\n' > "$run_root/cuda-version.json"
fi

echo "$server_command" > "$run_root/server-command.txt"
echo "$benchmark_command" > "$run_root/benchmark-command.txt"
server_pid=""
stop_server() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        for _ in {1..50}; do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 0.1
        done
        kill -KILL "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=""
}
trap 'stop_server' EXIT INT TERM

for run_index in 1 2 3; do
    run_dir="$run_root/run-$run_index"
    mkdir -p "$run_dir"
    raw_results="$run_dir/raw-results.json"
    export FUCINA_RUN_INDEX=$run_index
    export FUCINA_RUN_DIR=$run_dir
    export FUCINA_RAW_RESULTS=$raw_results
    export FUCINA_SOURCE_COMMIT=$source_commit

    started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '%s\n' "$started_at" > "$run_dir/server-started-at.txt"
    bash -c "exec $server_command" > "$run_dir/server.stdout.log" 2> "$run_dir/server.stderr.log" &
    server_pid=$!
    printf '%s\n' "$server_pid" > "$run_dir/server.pid"

    deadline=$((SECONDS + ready_timeout))
    until curl --fail --silent --show-error "$ready_url" > "$run_dir/ready-response.txt" 2> "$run_dir/ready-errors.txt"; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "server exited before readiness on run $run_index" >&2
            exit 1
        fi
        if ((SECONDS >= deadline)); then
            echo "server readiness timed out on run $run_index" >&2
            exit 1
        fi
        sleep 0.2
    done
    date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/server-ready-at.txt"

    rm -f "$raw_results"
    bash -c "$benchmark_command" > "$run_dir/benchmark.stdout.log" 2> "$run_dir/benchmark.stderr.log"
    [[ -f "$raw_results" ]] || { echo "benchmark did not write $raw_results" >&2; exit 1; }
    python3 - "$raw_results" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
value = json.loads(path.read_text())
if not isinstance(value, list):
    raise SystemExit(f"{path}: raw benchmark result must be a top-level JSON array")
PY
    date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/benchmark-finished-at.txt"
    stop_server
    echo "completed independent server start $run_index/3"
done

export RUN_ROOT="$run_root" CLAIM="$claim" SOURCE_COMMIT="$source_commit" SOURCE_DIRTY="$dirty" READY_URL="$ready_url"
python3 <<'PY'
import json
import os
import platform
from pathlib import Path
root = Path(os.environ["RUN_ROOT"])
manifest = {
    "schema_version": 1,
    "claim": os.environ["CLAIM"],
    "source_commit": os.environ["SOURCE_COMMIT"],
    "source_dirty": os.environ["SOURCE_DIRTY"] == "true",
    "ready_url": os.environ["READY_URL"],
    "independent_server_starts": 3,
    "model_fingerprints": json.loads((root / "model-fingerprints.json").read_text()),
    "raw_result_arrays": [str(Path(f"run-{i}/raw-results.json")) for i in range(1, 4)],
    "host": {"platform": platform.platform(), "machine": platform.machine()},
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

printf 'qualification evidence: %s\n' "$run_root"
