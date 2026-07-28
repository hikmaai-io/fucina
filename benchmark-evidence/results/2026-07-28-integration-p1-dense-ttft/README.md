# Integration Phase-1 — dense N=32 TTFT three-start evidence

Branch `integration/phase1` (scheduler identical to sol-05's `acc5e2f`; no
scheduler modification was made during integration). Two independent
three-start qualification sets on GB10, GPU flock held, default clocks,
length contract enforced (`ignore_eos` + `min_tokens` now server-side, every
request decodes exactly 128 tokens):

| set | run | N=32 TTFT median ms | p95 ms | agg tok/s |
|---|---|---:|---:|---:|
| step-5 retained (c7af618, 17:42Z) | 1 | 382.60 | 389.73 | 430.09 |
| | 2 | 404.20 | 410.11 | 436.50 |
| | 3 | 396.93 | 404.50 | 436.49 |
| step-6 dedicated (a591fac, 18:40Z) | 1 | 403.50 | 409.72 | 417.99 |
| | 2 | 398.90 | 404.54 | 428.58 |
| | 3 | 398.84 | 402.42 | 379.86 |

- Three-start medians: 396.93 ms (step-5 set), 398.90 ms (step-6 set);
  six-start median 398.87 ms.
- Retained (pre-SOL-05) baseline: 479.23 ms → **−16.7%** preserved TTFT win.
- Fresh vLLM reference: 213.09 ms → ratio 1.87× (baseline was 2.25×).
- sol-05's own capture measured 383.11 ms earlier in the day on the same
  scheduler code; today's later-session numbers sit 3–4% above it after
  ~2 h of continuous GPU benchmarking (environmental drift; the scheduler
  binary path is byte-identical in behavior — see the step-5 A/B parity
  diagnostics).

Full provenance: `benchmark-evidence/qualification/20260728T174257Z-…-2169047`
and `…20260728T184013Z-integration-p1-dense-ttft-a591fac2a81c-2396631`.
Summary: `summary.json`.
