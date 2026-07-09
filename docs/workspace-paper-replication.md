# Anthropic Global-Workspace Replication with Fucina

This reproduces one qualitative experiment from Anthropic's 2026 paper
[“Verbalizable Representations Form a Global Workspace in Language Models”](https://transformer-circuits.pub/2026/workspace/index.html):
Figure 9's **directed modulation while copying**.

## Experiment

The target sentence is identical in both conditions:

> The old painting hung crookedly on the wall.

- **Control:** copy the sentence exactly.
- **Focus:** copy it while concentrating on citrus fruits, without mentioning the side task.

The paper reads the J-Lens at the output token containing `ook` in `crookedly`. Since Qwen uses a
different tokenizer, the harness maps the character span back to whichever Qwen source token
covers the middle of `ook` (for Qwen3.5-9B this is typically `oked`). It then compares fitted-layer
top words for `citrus`, `orange`, `lemon`, `lime`, `fruit`, `grapefruit`, and `tangerine`.

## End-to-end procedure

### Preferred: published Qwen3.5-9B lens

Neuronpedia publishes the paper-compatible `Qwen/Qwen3.5-9B-Base` lens in
`neuronpedia/jacobian-lens`. Its metadata reports 458 WikiText prompts, BF16 fitting, sequence
length 128, and convergence stopping at mean relative change 0.00198491. For an official
Qwen3.5-9B FP8 checkpoint, this is preferable to a small local fit: FP8 block scales represent a
quantized version of the same base weights.

```sh
huggingface-cli download neuronpedia/jacobian-lens \
  qwen3.5-9b-pt/jlens/Salesforce-wikitext/Qwen3.5-9B-Base_jacobian_lens.pt \
  --local-dir jlens-download

python scripts/convert_jlens.py \
  jlens-download/qwen3.5-9b-pt/jlens/Salesforce-wikitext/Qwen3.5-9B-Base_jacobian_lens.pt \
  qwen35-9b-base.fjls --model-layers 32

python scripts/workspace_citrus_example.py \
  --model /models/Qwen3.5-9B-FP8 \
  --lens qwen35-9b-base.fjls \
  --top-k 20 --output-dir workspace-citrus-run
```

### Fit locally when no matching lens exists

```sh
# Python fitting environment only
pip install torch transformers accelerate
pip install git+https://github.com/anthropics/jacobian-lens.git

# Inference-only FP8 operators do not implement backward. Build an offline BF16 fitting copy.
python scripts/dequantize_fp8_hf.py \
  /models/Qwen3.5-9B-FP8 \
  /models/Qwen3.5-9B-BF16-for-JLens

# Smoke lens: use one prompt first. For evidence, use at least the default 8 prompts and
# preferably a larger held-out corpus; fitting is resumable.
python scripts/fit_fucina_jlens.py \
  --model /models/Qwen3.5-9B-BF16-for-JLens \
  --output /models/qwen35-9b-jlens.pt \
  --layers 4,8,12,16,20,24,28 \
  --n-prompts 8

# Native fucina inference still uses the original FP8 model.
python scripts/workspace_citrus_example.py \
  --model /models/Qwen3.5-9B-FP8 \
  --lens /models/qwen35-9b-jlens.fjls \
  --top-k 20 \
  --output-dir workspace-citrus-run
```

The run writes `control.jsonl`, `focus.jsonl`, stdout/stderr for both conditions, and a structured
`summary.json`. A positive qualitative result requires more fitted layers containing citrus words
in the focus condition than in control. Failure is reported as failure/inconclusive, never silently
promoted to a replication.

## Causal trace convention

For an autoregressive model, the residual at token `oked` predicts the following token, such as
`ly`. Every JSONL event therefore records both:

```json
{
  "source_token": "oked",
  "sampled_token": "ly",
  "layers": []
}
```

Workspace interpretation uses `source_token`; labeling the same hidden state as `ly` would be an
off-by-one error. The first event in a turn reads the final prompt token and predicts the first
completion token.

## Scientific scope

- This is a **cross-model qualitative replication** on Qwen3.5, not a reproduction of Sonnet 4.5
  effect sizes.
- The lens must be fitted from the same underlying checkpoint used by fucina. The BF16 fitting copy
  materializes the FP8 block scales solely to make autograd available.
- One-prompt lenses are smoke tests. In the initial exact-Qwen3.5-9B smoke run (`n_prompts=1`,
  layers 4/8/12/16/20/24/28), neither condition put a citrus word in the top-20 at `oked`; this is
  correctly reported as **no replication**, not treated as evidence against the paper. Multi-prompt
  averaging and multiple instruction phrasings are required before drawing a conclusion.
- `--jspace` is intentionally diagnostic and slow; it is excluded from production serving paths.
