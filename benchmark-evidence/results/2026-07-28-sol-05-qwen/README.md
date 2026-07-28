# SOL-05 Qwen GB10 qualification — 2026-07-28

Hardware: GB10 `sm_121a`, CUDA 13. Source scheduler commits: `acc5e2f` and
`ed7a6b9`; corrected MoE snapshot config: `1b90bf7`. Every model has three
independent server starts captured by `scripts/run_gb10_qualification.sh`.
Raw one-object arrays and manifests are retained in this directory; full logs,
model fingerprints, driver metadata, and commands remain under the qualification
roots named by the manifests.

## Three-start medians

| model | N | aggregate tok/s | median TTFT ms | p95 TTFT ms | retained delta | gate |
|---|---:|---:|---:|---:|---:|---|
| dense | 1 | 33.44 | 89.58 | 89.58 | +0.01% | PASS |
| dense | 2 | 58.27 | 100.39 | 100.43 | -1.69% | PASS |
| dense | 4 | 115.51 | 137.96 | 155.23 | -1.50% | PASS |
| dense | 8 | 201.39 | 185.37 | 185.73 | -1.57% | PASS |
| dense | 16 | 317.46 | 236.55 | 240.29 | +1.39% | PASS |
| dense | 32 | 443.06 | 383.11 | 390.18 | +0.97% | TARGET |
| MoE | 1 | 39.51 | 58.70 | 58.70 | -32.99% | FAIL |
| MoE | 2 | 61.69 | 87.40 | 108.46 | -39.15% | FAIL |
| MoE | 4 | 107.94 | 123.05 | 148.30 | -19.43% | FAIL |
| MoE | 8 | 167.91 | 182.59 | 216.48 | -26.92% | FAIL |
| MoE | 16 | 231.32 | 277.34 | 318.32 | -27.74% | FAIL |
| MoE | 32 | 376.90 | 455.18 | 519.52 | -20.22% | FAIL |

## TTFT result

N=32 TTFT improved materially against the retained 2026-07-18 Fucina runs:

- dense: 479.23 -> 383.11 ms median (**-20.1%**), 485.58 -> 390.18 ms p95 (**-19.6%**);
- MoE: 670.08 -> 455.18 ms median (**-32.1%**), 722.41 -> 519.52 ms p95 (**-28.1%**).

Against the retained vLLM N=32 medians, the ratio narrowed from 2.25x to 1.80x
for dense and from 2.15x to 1.46x for MoE. Dense aggregate throughput rose
438.80 -> 443.06 tok/s, narrowing its vLLM gap from 15.90% to 15.09%.

## Throughput gate failure

The strict eleven-win aggregate gate is **red**: all five protected dense cells
pass the 2% floor, while all six MoE cells fail. This result is not hidden or
re-labeled. The current server ignores the harness's `ignore_eos`/`min_tokens`
fields (those fields are absent from the server request contract). Consequently
the current MoE N=32 run generated 2,876 decode tokens per burst, versus 3,561
in the frozen run; wall time stayed comparable (7.63 s versus 7.54 s), so the
output-length-sensitive aggregate metric drops even though median per-stream
decode rates remain close (N=32 17.63 vs 18.51 tok/s). Resolving that protocol
mismatch/API gap is outside SOL-05 scheduler ownership. The acceptance criterion
"zero regression on 11 winning aggregate cells" is therefore **not met** by the
physical rerun and remains an explicit integration blocker.
