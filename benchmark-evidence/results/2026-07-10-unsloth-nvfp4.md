# Unsloth Qwen3.6-35B-A3B-NVFP4 — fucina validation

Date: 2026-07-10 (server log uses the host's UTC+ timezone and rolled into July 11)  
Branch: `feat/qwen36-unsloth-nvfp4`  
Checkpoint: `unsloth/Qwen3.6-35B-A3B-NVFP4` (26.47 GB compressed-tensors)  
Protocol: [`../PROTOCOL.md`](../PROTOCOL.md)

Raw artifacts:

- [`2026-07-10-unsloth-nvfp4-serving.json`](2026-07-10-unsloth-nvfp4-serving.json)
- [`2026-07-10-unsloth-nvfp4-ttft.json`](2026-07-10-unsloth-nvfp4-ttft.json)
- [`2026-07-10-unsloth-nvfp4-metrics.json`](2026-07-10-unsloth-nvfp4-metrics.json)
- `2026-07-10-unsloth-nvfp4-server.log`
- `2026-07-10-unsloth-nvfp4-timings.log` (diagnostic run only)

`--timings` was not enabled for any throughput or TTFT result.

## Results

| Metric | Accurate NVFP4 | Fast NVFP4 |
|---|---:|---:|
| Short single-stream decode | **60.87 tok/s** | 60.99 tok/s |
| ~3.5K single-stream decode | **53.10 tok/s** | 52.21 tok/s |
| N=1 served | **56.63 tok/s** | 35.45 tok/s |
| N=2 served | **94.03 tok/s** | 60.69 tok/s |
| N=4 served | **142.13 tok/s** | 118.09 tok/s |
| N=8 served | **181.48 tok/s** | 166.12 tok/s |
| N=16 served | **244.01 tok/s** | 233.46 tok/s |
| N=32 served | **333.07 tok/s** | 319.00 tok/s |
| Cold ~2K turn-1 TTFT median | **1,515.2 ms** | 1,614.2 ms |
| Warm turn-2 TTFT median | **69.9 ms** | 70.6 ms |

The served aggregate is completion-token throughput over each diverse burst, so lower-concurrency
comparisons are affected by the two checkpoints choosing different stop lengths. The fixed-length
single-stream tests are the cleaner decode comparison; both variants are effectively tied there.
At C32 the accurate checkpoint delivered 333.07 tok/s, 13.4% above the prior fucina FP8 rerun
(293.72 tok/s) and 23.4% above the recorded vLLM reference (269.9 tok/s).

The server admitted **32/32** slots at `--ctx 25280`, with zero 503 responses, CUDA errors, or
allocation failures.

## Tensor-core prefill/decode review

The accurate checkpoint upcasts the final eight expert layers to per-output-channel FP8. Fucina
normalizes those layers and the native NVFP4 layers into one resident grouped-NVFP4 expert format,
so decode remains one CUTLASS grouped gate/up GEMM plus one grouped down GEMM per layer.

A separate diagnostic run measured a warm 1,922-token prefill at **1,458.50 ms**:

- Expert slab dequantization: 353.43 ms
- Router/route/gather: 43.42 ms
- Grouped experts: 496.91 ms
- Shared expert: 26.42 ms
- Remaining work: 538.31 ms

This reproduces the Fast-checkpoint profile. The existing policy remains appropriate on GB10:
wide prefill dequantizes each resident expert slab once and uses grouped BF16 cuBLAS; decode-sized
continuous batches consume the resident NVFP4 slabs directly through CUTLASS. There is no
multi-device tensor parallelism on the single-GB10 target, and adding TP synchronization would not
improve this path.

## Correctness

`make qwen36-unsloth-nvfp4-test` passed on this checkpoint:

- B=3 graph versus per-row B=1: 24/24 for all three rows.
- CUDA graph enabled versus disabled: 24/24 for all rows.
- Batched versus single-forward oracle: 8/8.
- Sampling and row independence: PASS.

Sparse-MoE MTP remains disabled by policy; the model is served entirely through continuous
batching without speculative decoding.
