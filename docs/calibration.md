# Calibration and quality gates

`fucina-calibrate` is the first Phase-B calibration primitive for sparse Qwen3.5/3.6 MoE
checkpoints. It profiles the actual CUDA router during prefill and writes a versioned
`.imatrix`-style JSON sidecar.

```bash
make fucina-calibrate
./fucina-calibrate \
  -m /path/to/Qwen3.5-35B-A3B-NVFP4 \
  --corpus corpus/agentic-code-redteam.jsonl \
  --max-tokens 3000000 \
  --ctx 8192 \
  --out qwen35-agentic.imatrix.json
```

The corpus may be plain text (one document per line) or JSONL. JSON records may contain `text`,
`prompt`, `content`, or OpenAI-style `messages`. Keep corpus generation and provenance beside the
result; the intended mix is agentic coding/tool traces plus AI red-team and security-report tasks.

The sidecar contains, for every layer and expert:

- selected-route count and frequency;
- mean selected top-k router probability;
- a normalized heat score projected onto the expert gate/up/down tensor names;
- shared-expert tensor priority fixed at 1.0.

Profiling is explicit and has no normal-serving kernel overhead. The current v1 score is a
**routing/residency signal**, not yet a complete activation-sensitivity matrix. Attention,
GatedDeltaNet, norm, and shared-expert activation instrumentation remains the next B1 increment
before the sidecar can drive mixed precision.

## Quality baseline and gate

Record quality before changing a precision or residency policy, then rerun the identical suite.
The repository wrapper handles server startup and uses the 12-turn limit required by long chains:

```bash
PORT=18080 MODEL=/path/to/model BATCH=1 \
  OUTPUT_DIR=./runs/calibration-baseline \
  scripts/tool_eval_bench.sh --short
```

Baseline captured on 2026-07-11 for `unsloth/Qwen3.6-35B-A3B-NVFP4` with
`tool-eval-bench v2.0.4 --short`: **100/100 quality, 15/15 scenarios, 30/30 points**. Report:
`runs/calibration-baseline/2026/07/2026-07-11T12-57-32.616593Z_ab52d4ae.md`.
The short suite is a fast regression gate; a candidate precision policy must also pass the full
coding/red-team battery before becoming a default.
