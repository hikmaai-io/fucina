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
- RMS and maximum activation magnitude for each layer's mixer input, mixer output-projection
  input, MoE input, routed-expert down input, and shared-expert down input;
- a normalized activation score projected onto attention, GatedDeltaNet, router, shared-expert,
  and routed-expert tensor names;
- norms pinned at priority 1.0 and shared experts kept at a minimum 0.9 priority.

Profiling is explicit and has no normal-serving kernel overhead. Tensor names sharing the same
input activation (for example Q/K/V projections) intentionally share a magnitude score. Expert
precision combines routing heat with the corresponding gate/up or down activation score. This is
sufficient for a first measured precision/residency policy; per-channel sensitivity and
perturbation-based error attribution are possible later refinements, not prerequisites for B1.

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

After activation instrumentation landed, the same gate remained **100/100, 15/15, 30/30** with
normal profiling disabled, confirming the calibration branches do not perturb serving numerics.
Report: `runs/calibration-b1-activation/2026/07/2026-07-11T13-10-01.460992Z_2f135eb8.md`.
