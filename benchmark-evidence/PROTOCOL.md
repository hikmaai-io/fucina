# Qwen3.5-35B serving benchmark protocol

This is the reproducible protocol used for the cold-TTFT/C32 optimization gates. Keep these inputs fixed when comparing commits or servers.

## Hardware and checkpoint

- NVIDIA GB10 unified-memory system, CUDA 13, `sm_121a` build.
- Checkpoint: `Qwen/Qwen3.5-35B-A3B-FP8`, snapshot `0b2752837483aa34b3db6e83e151b150c0e00e49`.
- Context: 25,280 tokens.
- Temperature: 0.
- Diverse prompts are mandatory for concurrency tests; identical prompts create convergent MoE routing and can conceal row-mixing bugs.
- Run with no other inference server active. Record shared-box memory pressure from startup logs rather than silently assuming the machine is quiescent.

## Build and launch

```sh
make lib
make fucina
go test ./...

MODEL=/opt/spark/models/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/0b2752837483aa34b3db6e83e151b150c0e00e49
./fucina -m "$MODEL" \
  --ctx 25280 \
  --parallel 32 \
  --max-concurrent 64 \
  --gpu-mem-util 0.90 \
  --port 18080
```

Wait for `/readyz` before timing. Do not enable `--timings` during throughput measurements: Qwen prefill phase telemetry deliberately inserts CUDA synchronization boundaries.

## Decode/concurrency sweep

```sh
python3 scripts/bench_serving.py \
  --base-url http://127.0.0.1:18080 \
  --model qwen35 \
  --label "$(git rev-parse --short HEAD)" \
  --max-tokens 128 \
  --long-tokens 3500 \
  --ignore-eos \
  --conc 1,2,4,8,16,32 \
  --diverse \
  --verify-sample 4 \
  --out result.json
```

`agg_decode_tps` is `sum(completion_tokens - 1) / whole_burst_wall_time`; it includes burst admission and TTFT in the denominator. This is the served-throughput metric, not an engine-only kernel rate.

Protection baselines from the original run are 32.2 / 58.2 / 101.7 / 154.6 / 206.0 tok/s at N=1/2/4/8/16. Changes must remain within the established 5–8% noise allowance. The old N=32 result was 212.6 tok/s.

## Cold and warm TTFT

- Cold means an uncached roughly 2,000-token prompt on an already-ready process. Use a unique prompt each repetition so neither the prefix cache nor hybrid state snapshot can satisfy it.
- Warm turn-2 means a second request extending the exact completed first-turn history. Finish and drain turn 1 before starting turn 2.
- Report median of at least three repetitions.
- Historical protection baselines: cold 3,506 ms; warm turn-2 76.5 ms.

```sh
python3 scripts/turn_ttft.py \
  --base-url http://127.0.0.1:18080 \
  --model qwen35 \
  --reps 3 \
  --prompt-words 1500 \
  --turn-tokens 16 \
  --out turn-ttft.json
```

The script gives every turn-1 prompt a unique nonce, fully drains it, reconstructs the assistant
message from the SSE stream, then measures only the first meaningful token of the extending turn.

## Correctness and pressure gates

Before accepting runtime changes:

```sh
make qwen35-batch-test
make qwen35-state-test
make qwen35-moe-fp8-engine-test
make qwen35-longctx-test
```

Also run a diverse C32 HTTP burst and several simultaneous prompts longer than the default 8,192-token slot reservation. Require zero HTTP 503s, no CUDA allocation errors, and no output corruption. Record `/metrics` and the memory-plan startup lines with the result.
