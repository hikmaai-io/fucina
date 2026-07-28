# Gemma-4 qualification protocol

`../../scripts/gemma_qualification.py` is the SOL-08a release-gate harness for
Gemma-4 12B and 31B across Q4_0-QAT, Q8_0, and native NVFP4. It launches Fucina
and the equivalent vLLM artifact serially, uses the shared OpenAI completions
API, and writes `fucina.gemma-qualification.v1` evidence.

## Controlled-run rules

- Run on an otherwise quiescent GB10. Do not run Fucina and vLLM performance
  phases together.
- Bind the Fucina and vLLM artifacts to the same upstream release, tokenizer,
  context limit, prompt, quantization intent, and greedy generation policy.
  `artifact` and optional `vllm_artifact` are fingerprinted independently.
- Keep `independent_starts >= 3`. Startup-to-ready is measured from process
  creation until the configured readiness route first returns 2xx.
- Cold prompts carry a unique leading nonce. Warm-prefix requests extend the
  exact cold prompt text, never re-tokenized model output. Only server-reported
  execution skips count as cached; a logical LCP estimate is never substituted.
- The runner asserts on every response:
  `0 <= cached_tokens <= prompt_tokens`,
  `new_prefill_tokens = prompt_tokens - cached_tokens`, and
  `total_tokens = prompt_tokens + completion_tokens`.
- TTFT is request send to the first observed token event. ITL is emitted only
  when final usage proves one observed event per token. Per-stream throughput excludes the first token;
  aggregate throughput uses all completed tokens divided by whole-burst wall
  time. Raw event offsets and arrays are retained alongside p50/p95/p99.
- Memory sampling covers the launched process tree: RSS/HWM from `/proc` and GPU
  process memory from `nvidia-smi` when available. This matters on unified-memory
  GB10 systems where one source alone may be incomplete.

## Matrix run

Copy and edit `gemma-qualification.example.json`. Replace paths and every
fingerprint placeholder. When vLLM requires a converted but equivalent artifact,
set `vllm_artifact` and `vllm_artifact_sha256`, then use `{vllm_artifact}` in its
command. Commands are argument arrays and are never run through a shell.

```sh
python3 scripts/gemma_qualification.py validate \
  --config benchmark-evidence/gemma-qualification.json

python3 scripts/gemma_qualification.py run \
  --config benchmark-evidence/gemma-qualification.json \
  --out-dir benchmark-evidence/results/$(date +%F)-gemma-qualification
```

For the SOL-08a acceptance smoke, select 12B Q4_0-QAT while retaining the same
three-start policy:

```sh
python3 scripts/gemma_qualification.py run \
  --config benchmark-evidence/gemma-qualification.json \
  --subject gemma-4-12b-q4_0-qat --allow-partial-matrix \
  --out-dir benchmark-evidence/results/$(date +%F)-gemma-12b-q4-smoke
```

The archive contains a root `manifest.json`, a per-subject `manifest.json`, and
raw server logs. The manifests retain per-start raw requests and memory samples,
pooled p50/p95/p99 summaries, and explicit Fucina/vLLM p50 ratios. A ratio is
always `fucina / vllm`; therefore lower is better for latency/memory and higher
is better for throughput.

## Batch-MTP activation and losslessness

Launch two otherwise identical Fucina servers with continuous batching and the
same MTP assistant. Disable only the plain server's batch speculative path with
`FUCINA_NO_BATCH_SPEC=1`. Distinct ports are required. Then run:

```sh
python3 scripts/gemma_qualification.py mtp-probe \
  --mtp-url http://127.0.0.1:18080 \
  --plain-url http://127.0.0.1:18081 \
  --model gemma-4-12b-q4_0-qat --batch 4 --max-tokens 32 \
  --allow-text-fallback \
  --out benchmark-evidence/results/mtp-probe.json
```

The gate is not allowed to pass by silently falling back to plain decode. The
MTP server must increase `verify_forwards`, `drafted`, and `accepted` under a
real concurrent batch, while plain must show zero verify/draft work. Every stream
must exactly equal plain greedy decode. The harness compares like-for-like SSE
token IDs first, then standard `logprobs.tokens`. Without either it fails closed
unless `--allow-text-fallback` is explicit; that fallback requires one event per
usage token on both sides, equal token counts, and equal concatenated text.
Coalesced traces cannot pass. This mirrors
the non-triviality and token-equality checks in
`internal/server/batch/spec_lossless_test.go`.

## Trusted-oracle hook

An oracle artifact is JSON with full first-token logits and at least 32 greedy
reference token IDs:

```json
{
  "first_token_logits": [0.125, -1.5],
  "greedy_token_ids": [123, 456, 789]
}
```

The example is abbreviated; real `first_token_logits` must have the full vocab
width and `greedy_token_ids` must contain at least 32 IDs. Produce the trusted
artifact with the upstream reference implementation from the exact same raw
prompt/token IDs. Produce the candidate with an offline Fucina probe. Compare:

```sh
python3 scripts/gemma_qualification.py oracle \
  --reference trusted-gemma-oracle.json \
  --candidate fucina-gemma-oracle.json \
  --atol 1e-3 --rtol 1e-3 \
  --out benchmark-evidence/results/oracle-comparison.json
```

A matrix subject may instead declare `oracle.reference` plus either
`oracle.candidate` or an argument-array `oracle.candidate_command`. Placeholders
`{artifact}`, `{vllm_artifact}`, `{model}`, and `{subject}` are expanded. The
hook rejects NaN/Inf, width mismatch, logits outside tolerance, or any mismatch
among the first 32 greedy IDs. Reference and candidate file SHA-256 values are
archived.

## Archival contract

`gemma-qualification.schema.json` defines version 1. Every evidence document has:

- immutable `schema_version`, `kind`, UTC creation time, source commit/dirty bit,
  host/driver/CUDA metadata, and config SHA-256;
- independently fingerprinted Fucina and vLLM artifacts;
- at least three independent starts per benchmark claim;
- raw TTFT, ITL, throughput, usage/cache, memory, metrics, and server logs;
- derived p50/p95/p99 summaries without discarding raw arrays;
- separate MTP activation/losslessness and trusted-oracle gate documents.

Never hand-edit a generated manifest. If a run is invalid, retain it with a note
outside the manifest and run a new UUID archive.
