# Qwen3.5-only: prune + beat vLLM on every configuration

Status: ACTIVE (2026-07-11, supersedes the sweep-planning in qwen35-beat-vllm-plan.md)
Evidence: fucina-vs-vLLM head-to-head on 41a1b99 (post-tensor-refactor-merge), both
Qwen3.5-35B-A3B-FP8 and Qwen3.5-9B-FP8, concs 1–32, results in the claude-fucina job dir
(to be copied into benchmark-evidence/).

## Measured state (the scoreboard to flip)

**Qwen3.5-35B-A3B (MoE) — agg decode tok/s / TTFT ms:**

| N | fucina | vLLM | fucina TTFT | vLLM TTFT |
|---|---|---|---|---|
| 4 | 101.8 | 105.0 | 264 | 417 |
| 8 | 154.7 | 146.5 | 502 | 669 |
| 16 | 208.4 | 204.8 | 974 | 549 |
| 32 | 291.7 | 302.8 | 1923 | 664 |

**Qwen3.5-9B (dense) — agg decode tok/s:** fucina 56.6/109.6/179.3/236.1/260.5 vs
vLLM 42.9/83.9/161.7/296.1/501.9 @ N=2/4/8/16/32.

**Won:** single-stream decode (+28% MoE), N≤8 everything, single long-prompt TTFT
(4.6 s vs 6.8 s @3.5k), warm/state-cache TTFT.
**Lost:** (L1) TTFT under concurrency — MoE 2.9× worse at N=32; (L2) dense aggregate
N≥16 — −93% at N=32; (L3) MoE N=32 aggregate — −4%, within noise but not a win.

**Root cause found (code-confirmed):** Stage-18 fused prefill+decode was only ever wired
for `GEMMA4_IS_QWEN3_FAMILY = QWEN3 || QWEN3MOE`. Qwen3.5 loads the separate M4 batched
engine and the fused ABI returns -2 → `FUCINA_NO_FUSED_PREFILL` is a structural no-op on
Qwen3.5. Every fusion benefit measured to date (~13× decode-during-ingestion) applies
only to the architecture we are deleting. **L1 is unaddressed engineering, not physics.**

**Merge check:** 41a1b99 matches PROTOCOL gate baselines within 1–2%, N=32 improved
(291.7 vs 212.6). Tensor-refactor merge is clean.

## Part 1 — prune everything not adding value

Per decision "Qwen3.5 only" (see legacy-qwen3-removal-plan.md, now unblocked in a
reordered form):

- **R1. Legacy Qwen3/Qwen3MoE support** — arch-detect, Q5_K expert requant, qwen3moe
  router path, all `test_qwen3_*`/`test_qwen3moe_*` gates. The Stage-18 losslessness
  gate for legacy models is deleted WITH the legacy engines; its Qwen3.5 replacement is
  P1's gate (below), which must land in the same series — never merge a state with no
  fused-path gate at all.
- **R2. Dead/debunked lever code** — anything env-gated off-by-default that was measured
  a loss and never enabled: int8 Q4_K-MMQ prefill (dense-only, dense-8B is deleted),
  LM-head batched quant experiment remnants. Audit before deleting: keep what Gemma/e4b
  still uses.
- **R3. Legacy models out of the bench matrix** — PROTOCOL and gate scripts reference
  only Qwen3.5 checkpoints (35B-A3B FP8 primary, 9B FP8 dense secondary).
- **Keep:** Gemma/diffusion/e4b engines (different product surface), `dg_fp4_moe`
  including `_mapped` (Phase-C SSD streaming), shared mmvq kernels (call-graph audit
  first), all `qwen35_*`, internal/dist, session persistence.
- **Acceptance:** full make + all remaining gates green; Qwen3.5 bench numbers unchanged
  pre/post-prune (same binary protocol); `rg 'qwen3moe'` only in comments/history docs.

## Part 2 — beat vLLM on every configuration (ranked by measured gap)

- **P1. Port fused prefill+decode + chunked prefill into the qwen35 batched engine**
  (closes L1, the 2.9× TTFT-under-load loss — the single biggest lever).
  - Chunk arriving prefills (vLLM-style chunked prefill) and co-batch each chunk with
    active decode rows in one `decode_multiseq_forward`-equivalent on the M4 engine.
    The GDN recurrent state makes this different from the legacy port: a prefill chunk
    advances deltanet state sequentially per-sequence — chunk boundaries are natural
    (the GDN kernels are already chunked scans).
  - TDD gate first: Qwen3.5 fused == standalone byte-identical (decode rows AND
    prefilled seq's first token + 20-token continuation), on 35B-A3B and 9B.
  - Exit: N=32 TTFT ≤ vLLM's ~670 ms band while N=1–16 throughput stays in the
    protection band; expected side effect: closes L3 (fusion removes the decode
    stall that also costs aggregate).
- **P2. Dense N≥16 aggregate scaling** (closes L2). Diagnose before building: profile
  a dense N=32 step — if decode is bandwidth-saturated like the MoE, the gap is
  scheduler/stall time, and P1's fusion likely closes most of it for free. Only if a
  real kernel gap remains after P1, scope it separately. Do not tune kernels first —
  three prior kernel levers were debunked (roadmap history).
- **P3. Protection-band CI gate.** One script: both models, N=1–32, TTFT + decode +
  aggregate, asserting no metric regresses >5% vs the recorded baseline and printing
  the vLLM delta. Run before every merge to main (this is how "every configuration"
  stays won once flipped).
- **P4. Copy the head-to-head evidence** from the job dir into
  `benchmark-evidence/results/2026-07-11-qwen35-vs-vllm/` with the run protocol, so
  the scoreboard above is reproducible.

## Sequencing

P4 (evidence, minutes) → P1 gate + implementation (the lever) → P3 (lock it in) →
R1–R3 prune (fast follow, protected by P3) → P2 (measure-first, only if a gap survives P1).

Rationale: prune AFTER P1 lands — the legacy fused-path code is the reference
implementation for the port; deleting it first throws away the working example.

## Non-goals

- No decode-kernel tuning on the MoE grouped GEMM / mixer / LM head — measured at
  hardware floors (~81% of LPDDR5X peak), three consecutive debunks.
- No spec-on-MoE until P1 lands and acceptance is measured (deferred, unchanged).
- Distributed (Phase E) continues independently; not a lever for single-node vLLM parity.
