# Unsloth Qwen3.6-35B-A3B-NVFP4-Fast — first fucina validation

Date: 2026-07-10  
Branch: `feat/qwen36-unsloth-nvfp4`  
Checkpoint: `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` (23.63 GB compressed-tensors)  
Protocol: [`../PROTOCOL.md`](../PROTOCOL.md)

Raw results:

- [`2026-07-10-unsloth-nvfp4-fast-serving.json`](2026-07-10-unsloth-nvfp4-fast-serving.json)
- [`2026-07-10-unsloth-nvfp4-fast-ttft.json`](2026-07-10-unsloth-nvfp4-fast-ttft.json)

## Serving results

| Metric | Result |
|---|---:|
| Short single-stream decode | 60.99 tok/s |
| ~3.5K single-stream decode | 52.21 tok/s |
| N=1 served | 35.45 tok/s |
| N=2 served | 60.69 tok/s |
| N=4 served | 118.09 tok/s |
| N=8 served | 166.12 tok/s |
| N=16 served | 233.46 tok/s |
| N=32 served | **319.00 tok/s** |
| Cold ~2K turn-1 TTFT median | 1,614.2 ms |
| Warm turn-2 TTFT median | 70.6 ms |

Startup admitted 32/32 slots at 25,280 maximum context with the normal 8,192-token reservation.
There were no HTTP 503s, allocation failures, CUDA errors, or malformed outputs in the diverse
C32 sweep.

Opt-in phase telemetry on a warm 1,922-token prefill measured 1,444.85 ms: 355.08 ms expert slab
dequant, 44.68 ms routing, 489.19 ms grouped experts, 25.76 ms shared expert, and 530.14 ms other.
This confirms the existing GB10 policy still applies to this checkpoint: grouped BF16 cuBLAS is
the correct wide-prefill path, while resident NVFP4 CUTLASS is the decode path.

## Correctness gates

`make qwen36-unsloth-nvfp4-test` uses the production continuous-batching engine and passed:

- Three diverse rows: B=3 graph versus B=1, 24/24 each.
- CUDA graph on versus off: 24/24 each.
- Batched versus single-forward oracle: 8/8.
- Sampling and row independence: PASS.

The first loader smoke test exposed a producer-specific bug: Unsloth FP8 scales are BF16
`[out,1]`. Broadcasting row zero as a scalar generated repeated `is` tokens. Per-row conversion
fixed it; the same request then returned exactly `READY`.
