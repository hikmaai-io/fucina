# Phase B: calibration-driven precision policy

Phase B is an evidence pipeline, not a direct “make everything smaller” switch:

1. build a provenance-carrying workload corpus;
2. profile routing and activation sensitivity;
3. derive a codec policy constrained by kernels fucina actually has;
4. apply the policy in a converter/loader experiment;
5. accept it only through the quality gate.

## B2 — reproducible corpus

Edit `calibration/corpus-recipe.json` and add local source descriptors:

```json
{"path":"inputs/tool-traces.jsonl","category":"tool_calling",
 "provenance":"internal tool traces, 2026-07 export","license":"internal-eval"}
```

Then build and profile:

```bash
python3 scripts/build_calibration_corpus.py \
  --recipe calibration/corpus-recipe.json --out /tmp/fucina-calibration.jsonl
make fucina-calibrate
./fucina-calibrate -m /path/to/model --corpus /tmp/fucina-calibration.jsonl \
  --max-tokens 3000000 --out /tmp/model.imatrix.json
```

The builder deduplicates content by SHA-256, deterministically shuffles each category, preserves
source/license/line provenance, emits a corpus hash manifest, and targets the agreed mix:
agentic coding/tool use 50%, red-team/security analysis 35%, math/long documents 15%.
Its token estimate is only for weighted assembly; `fucina-calibrate` applies the exact model
Tokenizer and hard three-million-token ceiling.

## B3 — capability-gated policy

```bash
python3 scripts/derive_precision_policy.py /tmp/model.imatrix.json \
  --out /tmp/model.precision-policy.json
```

The policy keeps attention, GatedDeltaNet, router, shared experts, and norms at
`fp8_or_bf16`; hot/warm routed experts use NVFP4. Cold experts also remain NVFP4 because fucina
has no accepted INT2 kernel. `--sub4-kernel` exists only for the future kernel experiment and must
not be used merely to claim a memory reduction. Every policy records the source sidecar hash,
thresholds, capabilities, reasons, and exact per-tensor decisions.

The output is currently **declarative**. The next B3 increment is checkpoint conversion/direct
loader dispatch that applies multiple codecs. Uniform NVFP4 remains the runnable baseline until
that lands and passes B4.

## B4 — quality gate

Run the baseline and candidate with the same `tool-eval-bench` arguments, coding battery, red-team
battery, seed, and prompt set. The fast tool-call report gate is:

```bash
python3 scripts/quality_gate.py baseline.md candidate.md \
  --max-quality-drop 1 --max-score-drop 1 --max-error-increase 0
```

This gate intentionally fails closed on missing report fields. Tool-call quality alone is not a
complete release gate; record coding and red-team task scores alongside it before changing a
shipping default.

Unit tests:

```bash
PYTHONPATH=scripts python3 -m unittest scripts/test_phase_b.py
```
