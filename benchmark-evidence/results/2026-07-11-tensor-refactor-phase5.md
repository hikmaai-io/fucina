# Tensor ownership refactor — Phase 5 checkpoint

Date: 2026-07-11  
Checkpoint: measured worktree based on `f583b93` (typed workspace checkpoint)  
Model: `unsloth/Qwen3.6-35B-A3B-NVFP4` (accurate)  
Serving flags: `--ctx 25280 --parallel 32 --max-concurrent 64 --gpu-mem-util 0.90`

## Correctness and ownership gates

- Official FP8, Unsloth Fast, and Unsloth accurate engine oracle: PASS (8/8).
- M4 row independence, graph on/off, M3 parity, and sampling: PASS.
- Qwen state continuation: PASS (16/16).
- Q4_K expert compatibility mode (`FUCINA_MOE_Q4K=1`): PASS (8/8).
- Host model-plan and transactional allocation fault-injection tests: PASS.
- `go test ./...` and `git diff --check`: PASS.
- Exact model ledger remains **23.13 GiB**; admission remains **32/32**.

## Serving KPI

| Row | Baseline | Phase 5 | Delta | Final target |
|---|---:|---:|---:|---:|
| short fixed decode | 60.87 | 60.86 tok/s | -0.02% | >64 |
| 2 streams aggregate | 94.03 | 95.90 tok/s | +1.99% | >105 |
| 4 streams aggregate | 142.13 | 144.49 tok/s | +1.66% | >150 |
| 32 streams aggregate | 333.07 | 320.67 tok/s | -3.72% | protection row |
| long-context decode | 53.10 | 53.26 tok/s | +0.31% | protection row |

The primary 1/2/4-stream phase gate passes. Final throughput targets remain open. C32 remains noisy
and below the pinned publication baseline, so it continues to be tracked as a protection row.

Raw artifacts:

- `2026-07-11-tensor-refactor-phase5-serving.json`
- `2026-07-11-tensor-refactor-phase5-startup.log`
