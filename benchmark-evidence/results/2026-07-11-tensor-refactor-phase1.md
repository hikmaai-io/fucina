# Tensor descriptor refactor — phase 1 checkpoint

Date: 2026-07-11  
Base commit: `e10820a`  
Checkpoint: `unsloth/Qwen3.6-35B-A3B-NVFP4` accurate  
Protocol: [`../PROTOCOL.md`](../PROTOCOL.md), without `--timings`

Raw artifacts:

- [`2026-07-11-tensor-refactor-phase1-serving.json`](2026-07-11-tensor-refactor-phase1-serving.json)
- [`2026-07-11-tensor-refactor-phase1-ttft.json`](2026-07-11-tensor-refactor-phase1-ttft.json)
- [`2026-07-11-tensor-refactor-phase1-startup.log`](2026-07-11-tensor-refactor-phase1-startup.log)

## KPI gate

Requested KPI: **>64 / >105 / >150 tok/s at 1/2/4 streams**.

| Metric | Pre-refactor validated | Descriptor checkpoint | Delta | Target |
|---|---:|---:|---:|---:|
| N=1 fixed decode | 60.87 tok/s | 60.36 tok/s | -0.84% | >64 |
| N=2 served | 94.03 tok/s | 93.48 tok/s | -0.59% | >105 |
| N=4 served | 142.13 tok/s | 142.01 tok/s | -0.08% | >150 |
| N=1 served | 56.63 tok/s | 56.64 tok/s | +0.02% | protection only |
| N=32 served | 333.07 tok/s | 317.06 tok/s | -4.81% | protection only |
| Cold TTFT median | 1,515.2 ms | 1,598.9 ms | +5.52% | protection only |
| Warm turn-2 TTFT median | 69.9 ms | 72.2 ms | +3.22% | protection only |

The migrated 1/2/4 rows remain within the phase-1 `<1%` throughput-regression gate. The requested
performance KPI is not yet met; this ownership-only phase intentionally changes no kernel arithmetic.
The C32 and TTFT differences remain inside the publication protocol's protection/noise allowance.

## Correctness and residency

- Official FP8 MoE engine: oracle 8/8; row independence, graph on/off, and sampling passed.
- Unsloth Fast and accurate: oracle 8/8; row independence, graph on/off, and sampling passed.
- Qwen state snapshot: 16/16 bit-identical continuation.
- Qwen long context: 40/40 at both ~1K and ~4K.
- `make nvfp4-test`, `go test ./...`, and `git diff --check` passed.
- Exact allocation ledger stayed **23.13 GiB** and admission stayed **32/32**.
- No HTTP errors, CUDA allocation errors, or output corruption were observed.
