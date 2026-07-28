# Integration Phase-1 — 12 retained Qwen cells with the length contract enforced

Branch: `integration/phase1` at `c7af618` (main `06b39ed` + sol-04 + sol-05 +
handoff patches + `ignore_eos`/`min_tokens`). GB10, GPU flock held for every
run; three independent server starts per model via
`scripts/run_qwen_retained_cells.sh`.

Qualification roots (3 starts each, full provenance):

- dense: `benchmark-evidence/qualification/20260728T174257Z-sol-05-qwen-dense-c7af618d33d5-2169047`
- MoE:   `benchmark-evidence/qualification/20260728T174928Z-sol-05-qwen-moe-c7af618d33d5-2187105`
- gate summary: `retained-gate-summary.json` (copy of
  `qualification/sol-05-qwen-retained-20260728T175420Z.json`)

## Length contract verification

With `ignore_eos=true` + `min_tokens=max_tokens` (the harness's request
fields, now implemented server-side), every cell in every start generated
exactly 127 counted decode tokens per request (128 sampled; the bench counts
`ntok-1`). The retained aggregate metric is stationary for the first time:
SOL-05's runs on `06b39ed`-based branches generated 9–90 tokens/request on
the same harness because the fields were silently ignored.

## Gate result: FAILED (5/6 MoE aggregate cells below the −2% floor)

Cellwise medians over 3 starts vs the frozen 2026-07-18 baseline:

| model | N | candidate | baseline | delta | verdict |
|---|---|---:|---:|---:|---|
| dense | 1  | 33.48  | 33.44  | +0.12% | PASS |
| dense | 2  | 58.22  | 59.27  | −1.76% | PASS |
| dense | 4  | 115.75 | 117.27 | −1.30% | PASS |
| dense | 8  | 218.82 | 204.61 | +6.95% | PASS |
| dense | 16 | 331.16 | 313.12 | +5.76% | PASS |
| dense | 32 | 436.49 | 438.80 | −0.53% | TARGET (non-protected) |
| MoE | 1  | 57.58  | 58.96  | −2.34%  | FAIL |
| MoE | 2  | 93.99  | 101.39 | −7.30%  | FAIL |
| MoE | 4  | 145.59 | 133.98 | +8.67%  | PASS |
| MoE | 8  | 209.01 | 229.75 | −9.03%  | FAIL |
| MoE | 16 | 279.41 | 320.14 | −12.72% | FAIL |
| MoE | 32 | 407.00 | 472.41 | −13.85% | FAIL |

All 5 protected dense cells pass. MoE N=32 TTFT median improved 670 → ~536 ms
vs the baseline capture; dense N=32 TTFT median 383–390 ms (retained).

## Diagnosis: the scheduler is NOT the cause

All diagnostic runs under the GPU flock; the key A/Bs were also repeated with
graphics clocks locked at 2418 MHz (no difference; `div-clocklocked.json`).
Raw JSONs in `diagnostics/`.

1. **Integration ≡ its base `06b39ed`** (sol-05 scheduler + sol-04 + metrics
   wire + length contract cause no regression). MoE, matched workloads:
   - non-diverse natural-stop (all requests 127 tok):
     `06b39ed` 57.8/96.0/276.6/653.4 (N=1/2/8/32) vs integration
     57.5/96.6/275.6/648.8 → ±0.7%.
   - diverse natural-stop: `06b39ed` 61.1/165.9/374.7 (N=2/8/32) vs
     integration 60.8/166.6/374.9 → parity.
2. **The frozen baseline build did not stop at EOS.** Rebuilt the July-18
   main state `39a96db` (own CUDA archive, same sm_121a). WITHOUT ignore_eos
   it still generated 127 tok/request at N=2 diverse (natural answer is ~28
   tokens) and reproduced the frozen numbers: 101.1 vs 101.4 (N=2), 465.1 vs
   472.4 (N=32). The baseline's "accidental" workload therefore matches
   today's contract-enforced lengths, making the comparison length-fair —
   and the baseline server itself was pre-router-corruption-fix.
3. **The MoE loss lies in the `39a96db..06b39ed` window on main** — the MoE
   graph-decode qualification work ("fix Qwen3.6 MoE graph-router
   corruption", "capture MoE router on decode stream", "preserve FP8 scales
   through descriptor rebind", "qualify MoE graph decode"). Direct
   head-to-head, one harness (non-diverse full-length, clock-locked):
   `39a96db` 58.5/96.8/229.8/492.2 vs `06b39ed` 56.6/95.4/271.0/640.6 —
   a mixed engine trade (diverse/varied-prompt full-length decode loses
   ~5–13%, identical-prompt N=8/32 gains 18–30%).
4. **Length sensitivity is nil.** Integration at baseline-matched ~110
   tok/request: N=16 272.4, N=32 396.6 vs 279.4/407.0 at 127 — the gap does
   not shrink at matched lengths.
5. **Baseline reproducibility is partial.** Even `39a96db` does not
   reproduce the frozen N=8 cell (205.0 vs 229.8 diverse), so the July-18
   capture conditions (strict quiescence/session state) are not fully
   recoverable.

## Verdict (second-opinion reviewed)

Reported as **FAILED against the frozen baseline, with attribution**: the
regression pre-dates this integration and SOL-05 entirely; the scheduler is
exonerated by direct A/B parity with its own base. The 2026-07-18 MoE
baseline should be treated as **invalid** (captured on a build with a broken
EOS stop on this path and before the MoE router-corruption fixes; partially
irreproducible). A `claude-fable-5` consult reviewed the methodology and
endorsed proceeding with the integration while (a) re-freezing the baseline
on the integrated branch with the length contract enforced, and (b) opening
a dedicated engine-side WP to investigate how much of the
`39a96db..06b39ed` MoE decode loss is recoverable without reinstating the
router corruption. No decode kernel was touched in this WP.

Consult log: `fucina-swarm/logs/integration-p1-consult1.log`.

## Proposed re-frozen baseline (integrated branch, length contract)

Cellwise medians from this run (3 starts, `c7af618`): use
`retained-gate-summary.json` `candidate` values as the new frozen reference
for future scheduler gates.
