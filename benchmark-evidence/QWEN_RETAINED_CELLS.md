# SOL-05 Qwen retained-cell qualification

This guard re-runs the retained Qwen3.5 serving matrix: dense 9B and MoE
35B-A3B at concurrency 1/2/4/8/16/32 (12 cells total). The eleven historical
winning cells fail if median aggregate decode throughput across three independent
server starts drops by more than 2%. Dense N=32 remains a reported target cell.

## GB10 command

Both configured checkpoints are present on the qualification GB10. From a clean
SOL-05 checkout, serialize with the shared GPU lock and run:

```bash
make lib fucina
scripts/run_qwen_retained_cells.sh \
  benchmark-evidence/configs/qwen-retained-12.json
```

The wrapper calls `scripts/run_gb10_qualification.sh` separately for dense and
MoE, starts a fresh server three times for each checkpoint, runs:

```bash
python3 scripts/bench_serving.py \
  --base-url http://127.0.0.1:18080 \
  --model qwen35 --max-tokens 128 --long-tokens 3500 \
  --ignore-eos --conc 1,2,4,8,16,32 --diverse --verify-sample 4
```

and archives source/model fingerprints, CUDA/driver metadata, logs, and raw
arrays under `benchmark-evidence/qualification/`. Override model locations only
when mirroring the exact checkpoints:

```bash
FUCINA_DENSE_MODEL=/path/to/models--Qwen--Qwen3.5-9B-FP8 \
FUCINA_MOE_MODEL=/path/to/models--Qwen--Qwen3.5-35B-A3B-FP8 \
  scripts/run_qwen_retained_cells.sh \
  benchmark-evidence/configs/qwen-retained-12.json
```

## Offline gate

Any three protocol-matched JSON results per model can be checked directly:

```bash
python3 scripts/qwen_retained_cells.py check \
  --config benchmark-evidence/configs/qwen-retained-12.json \
  --candidate dense=dense-run-1.json \
  --candidate dense=dense-run-2.json \
  --candidate dense=dense-run-3.json \
  --candidate moe=moe-run-1.json \
  --candidate moe=moe-run-2.json \
  --candidate moe=moe-run-3.json \
  --json-out retained-summary.json
```
