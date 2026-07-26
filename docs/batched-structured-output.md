# Batched structured output

## Status

`response_format: {"type":"json_object"}` and OpenAI-style
`response_format: {"type":"json_schema", ...}` are supported in continuous batching, including
the mandatory Qwen3.5/Qwen3.6 serving path. The schema subset covers typed objects, required
properties, nested objects, arrays, array items, enums, strings, numbers, integers, booleans,
null, and `additionalProperties:false` behavior.

Adapters must expose exact one-token host-input decode plus full logits. The CUDA Qwen/Gemma
adapter does; adapters without it retain the HTTP 501 fail-closed guard.

## Correctness design

- Every scheduler sequence owns a fresh `grammar.Constraint` and host RNG. The FSM is advanced
  only by that slot's accepted token; no grammar state is indexed by batch row or shared globally.
- If any active row is constrained, the shared step uses `StepBatchExact`. Speculative decoding is
  disabled for that step, so no draft tail can outrun the grammar and no rollback is needed.
- Qwen exact decode bypasses both speculative verification and the CUDA-graph GPU token splice.
  The host-resampled grammar token is therefore the authoritative next input. Ordinary graph and
  speculative paths are unchanged when no constrained row is active.
- Prefill and exact decode leave full logits resident. The adapter copies compacted rows to the
  host; each constrained row applies its own mask and sampling parameters. B=1's approximate Q8
  head is recomputed with the exact BF16 head before constrained sampling.
- Prefix caching and session restore/export keep the ordinary committed-token frontier. A forced
  response suffix is replayed as prompt suffix on a later restored turn rather than pretending it
  was committed to device state.
- Optional tool declarations may accompany `response_format`, but constrained JSON is treated as
  assistant content and is not passed through legacy Qwen/Gemma tool-call recovery. Required or
  specific `tool_choice` is rejected with HTTP 400 because its forced XML/marker continuation and
  a JSON-first grammar are mutually incompatible.
- At a token/KV cap, `Close()` synthesizes a schema-valid suffix, including unfinished enums,
  nested containers, and missing required properties. Strict JSON-number and escape states prevent
  malformed partial scalars. Generation remains bounded and returns `finish_reason: "length"`.

This path is correctness-first: full-logit D2H and, for B=1 FP8, exact-head recomputation cost more
than ordinary on-device sampling. Unconstrained traffic pays neither cost.

## CPU coverage

`internal/server/batch/constraint_test.go` covers:

- two simultaneous slots with different schemas (state isolation);
- nested objects and arrays of objects;
- enum masking against an illegal model preference;
- max-token force-close with nested required fields and enum completion;
- speculative-engine capability present but never called for constrained rows;
- session restore/export through the exact grammar path.

`internal/grammar/schema_test.go` additionally covers strict numbers, required-key synthesis, and
nested schema-valid closure. `internal/server/response_format_test.go` drives the HTTP batch route.

## Measured before/after — GB10, 2026-07-26

Hardware smoke used the official FP8 checkpoint
`Qwen/Qwen3.6-35B-A3B-FP8` (snapshot `95a723d...`), port 8090, `--ctx 8192 --parallel 2`, with the
server and all GPU gates serialized by `flock /tmp/fucina_gpu.lock`.

| Request style | Before | After | Wall time | Validation |
|---|---:|---:|---:|---|
| TC-64 typed object | HTTP 501 | HTTP 200 | 0.725 s | required string + integer |
| TC-65 nested object | HTTP 501 | HTTP 200 | 1.589 s | nested required object/types |
| TC-66 enum | HTTP 501 | HTTP 200 | 0.401 s | value in `pending/running/done` |
| TC-67 array of objects | HTTP 501 | HTTP 200 | 2.489 s | every item has typed required fields |
| TC-69 `json_object` | HTTP 501 | HTTP 200 | 0.781 s | parsed top-level object |
| hard cap (`max_tokens: 8`) | HTTP 501 | HTTP 200, `length` | — | force-closed schema-valid `{"rows":[]}` |

Result: **0/5 available before, 5/5 schema-valid after**. Validation parsed the assistant content
with Python `json.loads` and recursively checked required keys, object/array/scalar types, enums,
and additional-property exclusion.

Regression evidence from the same source/build:

- `make gpu-gates`: PASS (dense/MoE parity, batch, state, shard, chunk, multiseq prefill,
  clean-GDN, exact-head, and MoE engine gates).
- `go test ./...`: PASS.
- `make check`: PASS (vet, lint, and race-enabled Go suite).
