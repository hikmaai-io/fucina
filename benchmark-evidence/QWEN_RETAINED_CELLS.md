# SOL-05 Qwen retained-cell qualification

This guard re-runs the retained Qwen3.5 serving matrix: dense 9B and MoE
35B-A3B at concurrency 1/2/4/8/16/32 (12 cells total). The eleven retained
winning cells fail if median aggregate decode throughput across three independent
server starts drops by more than 2%. Dense N=32 remains a reported target cell,
but its exact-length three-start median is also frozen and reported.

## Refrozen baseline and provenance

The gate owner countersigned the Phase-1 FAILED-with-attribution waiver on
2026-07-28. The canonical config now uses all 12 `candidate` medians from
`results/2026-07-28-integration-p1-qwen/retained-gate-summary.json` as its
Fucina baselines. Dense N=32 is included even though it remains target-only.
The fresh July vLLM files remain separate competitive references; they were not
substituted for or folded into the Fucina baseline.

The machine-checkable audit record is
`results/2026-07-28-integration-p1-qwen/baseline-provenance.json`. It records the
`c7af618` length-contract source, `a591fac` evidence commit, merged main
`915eaca`, both qualification roots, source config/harness/prompt/model/result
hashes, and the exact protocol. Baseline generation used three independent
server starts and a cellwise median with `max_tokens=128`, `ignore_eos=true`,
`min_tokens=128`, temperature zero, and diverse prompts. Thus every request has
128 completion tokens (127 timed decode intervals).

The old 2026-07-18 Fucina files remain unchanged as historical evidence, but are
no longer canonical gate targets. In particular, the old MoE baseline came from
a broken-EOS, pre-router-correctness window: it could reproduce full-length
numbers without an explicit length contract and predates the router-corruption
fixes, so it is not a correctness-qualified comparison for the fixed engine.
The re-freeze does not erase the Phase-1 failure or its attribution.

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

Validation fails closed if the exact-length contract, dated countersign,
qualification manifests, required hashes, or baseline provenance are absent or
do not match:

```bash
python3 scripts/qwen_retained_cells.py validate \
  --config benchmark-evidence/configs/qwen-retained-12.json
```

Any three protocol-matched JSON results per model can then be checked directly:

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
