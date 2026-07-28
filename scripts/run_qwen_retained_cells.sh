#!/usr/bin/env bash
# Re-run the 12 retained Qwen serving cells with three independent starts/model.
set -euo pipefail

config=${1:-benchmark-evidence/configs/qwen-retained-12.json}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
[[ -f "$config" ]] || { echo "missing config: $config" >&2; exit 2; }
[[ -x ./fucina ]] || { echo "missing ./fucina; build first with: make lib fucina" >&2; exit 2; }

# One lock covers both three-start claims; vLLM and other Fucina qualification
# jobs must use the same GB10 lock.
exec 9>/tmp/fucina_gpu.lock
flock 9

mapfile -t model_rows < <(python3 - "$config" <<'PY'
import json, os, sys
cfg=json.load(open(sys.argv[1]))
bench=cfg["bench"]
for name, model in cfg["models"].items():
    override=os.environ.get(f"FUCINA_{name.upper()}_MODEL", "")
    path=override or model["model_path"]
    print("\t".join(map(str, [name,path,model["served_model"],model["port"],bench["max_tokens"],bench["long_tokens"],bench["verify_sample"]])))
PY
)

check_args=()
roots=()
for row in "${model_rows[@]}"; do
    IFS=$'\t' read -r name model served port max_tokens long_tokens verify_sample <<<"$row"
    [[ -e "$model" ]] || { echo "$name model missing: $model" >&2; exit 2; }
    server_command="./fucina -m '$model' --port '$port' --parallel 32 --max-concurrent 64"
    benchmark_command="python3 scripts/bench_serving.py --base-url 'http://127.0.0.1:$port' --model '$served' --label 'sol-05-$name' --max-tokens '$max_tokens' --long-tokens '$long_tokens' --ignore-eos --conc 1,2,4,8,16,32 --diverse --verify-sample '$verify_sample' --out \"\$FUCINA_RUN_DIR/serving.json\" && python3 scripts/qwen_retained_cells.py wrap --input \"\$FUCINA_RUN_DIR/serving.json\" --output \"\$FUCINA_RAW_RESULTS\""
    output=$(scripts/run_gb10_qualification.sh \
        --claim "sol-05-qwen-$name" \
        --model "$model" \
        --server-command "$server_command" \
        --benchmark-command "$benchmark_command" \
        --ready-url "http://127.0.0.1:$port/health" \
        --ready-timeout 600)
    printf '%s\n' "$output"
    root=${output##*qualification evidence: }
    [[ -f "$root/manifest.json" ]] || { echo "qualification root not found: $root" >&2; exit 1; }
    roots+=("$root")
    for run in 1 2 3; do
        check_args+=(--candidate "$name=$root/run-$run/raw-results.json")
    done
done

summary_root=${EVIDENCE_ROOT:-benchmark-evidence/qualification}
summary="$summary_root/sol-05-qwen-retained-$(date -u +%Y%m%dT%H%M%SZ).json"
python3 scripts/qwen_retained_cells.py check --config "$config" "${check_args[@]}" --json-out "$summary"
printf 'retained-cell summary: %s\n' "$summary"
printf 'qualification roots:\n'; printf '  %s\n' "${roots[@]}"
