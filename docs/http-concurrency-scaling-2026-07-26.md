# Qwen3.6-35B-A3B-FP8 HTTP concurrency scaling — GB10

Date: 2026-07-26  
Baseline: `984d4a1`  
Candidate: this change

## Root cause

The CUDA batched decoder was not the bottleneck. The HTTP scheduler lost concurrent long-prompt requests before they could join the decode batch:

1. `BatchAdapter.MaxFusedRows` inferred fused-prefill support from broad Qwen-family detection. Qwen3.5/3.6 is in that family for serving control, but its hybrid runtime's `gemma4_engine_step_batch_fused` deliberately returns `-2` (unsupported).
2. The scheduler treated that optional-capability result as a request failure. Every long request arriving beside an active stream was evicted, then the incumbent stream was decoded alone. This produced the repeated `avgB 1.0` telemetry and `StepBatchFused ... ret -2` errors.
3. Idle admission was decided from the live held-slot count after each admission. Consequently, request 2+ in one initially idle burst looked like a busy-batch arrival after request 1 took a slot and was diverted to chunked prefill. Requests arriving while a blocking prefill ran were not re-drained before the first decode step. First tokens were also published before peer admission finished, so client-side generation time included the remaining prefills.
4. `SeqAddMultiseq` failure itself was not intended to be permanent, but the fallback policy undermined it: only one pass was considered idle, and long prompts were excluded from tile-bounded grouped admission.

The engine mutex is not a per-request single-flight bottleneck: the one scheduler goroutine owns it and submits all active rows in one `StepBatch`. The memory plan was also not the limiting factor. This run requested eight slots and selected `slots=8/32`, `slotctx=8192`, `maxctx=25280`; the earlier default `slots=4` was the configured admission target and was sufficient for C4.

## Fix

- Added an authoritative CUDA fused-prefill capability query. The current hybrid runtime advertises zero rows instead of being guessed from model family.
- Mapped runtime `-2` to an optional-capability sentinel. A stale optimistic probe now disables fusion once and continues with exact `PrefillChunk` + `StepBatch` interleaving without evicting the request.
- Snapshot idle state at admission entry, re-drain requests that arrive during blocking prefill, and complete an initially idle capacity wave before its first decode.
- Complete idle long-prompt bursts as one admission wave. Existing multisequence-eligible prompts may use repeated bounded groups; long prompts preserve their established serial one-shot numerical path. A transient multisequence failure falls back to serial prefill in the same idle wave.
- Hold the wave's already-sampled first tokens until peer admission completes. This changes delivery timing only; token IDs, KV state, and sampling order are unchanged.
- Keep chunked prefill interleaved with decode for genuine arrivals to an already-running batch.

## Reproduction

Server:

```bash
flock /tmp/fucina_gpu.lock ./fucina \
  -m /opt/spark/models/hub/models--Qwen--Qwen3.6-35B-A3B-FP8 \
  --tokenizer /opt/spark/models/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989/tokenizer.json \
  --port 8091 --parallel 8 --max-concurrent 16
```

Client (`llama-benchy 0.3.8`, one run, uncached 2K prompts, 128 generated tokens):

```bash
llama-benchy \
  --base-url http://127.0.0.1:8091/v1 \
  --model Qwen/Qwen3.6-35B-A3B-FP8 \
  --served-model-name models--Qwen--Qwen3.6-35B-A3B-FP8 \
  --tokenizer /opt/spark/models/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989 \
  --pp 2048 --tg 128 --runs 1 --no-warmup --skip-coherence \
  --no-adapt-prompt --latency-mode none --concurrency 1 2 4 8 --no-cache
```

## Result

`tg` is aggregate generated-token throughput reported by llama-benchy.

| concurrency | before tg tok/s | after tg tok/s | after per-request tok/s | observed steady decode B |
|---:|---:|---:|---:|---:|
| 1 | 46.11 | 45.11 | 45.11 | 1 |
| 2 | 46.01 | 75.63 | 37.82 | 2 |
| 4 | 46.06 | **117.98** | 29.50 | 4 |
| 8 | 45.90 | **171.42** | 21.44 | 8 |

Before the fix, telemetry stayed at `avgB 1.0` for concurrent 2K-prompt runs and logged `StepBatchFused failed ... ret -2`. After the fix, steady decode telemetry reached `avgB 2.0`, `4.0`, and `8.0`; no unsupported fused call was made. C4 aggregate improved 2.56x and cleared the requested 100 tok/s floor. C8 improved 3.73x. No B=1 decode kernel or sampling arithmetic changed.

## Gates

- `make gpu-gates`: PASS, including dense/MoE parity, graph row-independence, state/session boundary, chunk parity, multisequence prefill, exact head, and the Qwen3.6 MoE 8/8 oracle.
- `go test ./...`: PASS.
- `make check`: PASS (`go vet`; race suite; lint skipped because `golangci-lint` is not installed).

Scheduler tests specifically cover:

- requests arriving during the first blocking prefill joining the initial decode batch;
- idle long-burst admission and full first-step occupancy;
- transient `SeqAddMultiseq` failure falling back without serializing decode;
- unsupported fused prefill falling back without eviction;
- concurrent chunked-prefill losslessness and eventual full batch occupancy.
