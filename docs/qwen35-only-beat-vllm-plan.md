# Qwen3.5-only: prune + beat vLLM on every configuration

Status: ACTIVE, rev 2 (2026-07-11, supersedes the sweep-planning in qwen35-beat-vllm-plan.md)
Rev 2: folds in adversarial review (gpt-5.6-sol): profile-before-implement for P1, P3
moved ahead of P1, dual-sided gate, expanded correctness matrix, causality demoted to
hypothesis. Original rev 1 verdict was NO-GO as written.
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

**Code-confirmed fact:** Stage-18 fused prefill+decode was only ever wired for
`GEMMA4_IS_QWEN3_FAMILY = QWEN3 || QWEN3MOE`. Qwen3.5 loads the separate M4 batched
engine and the fused ABI returns -2 → `FUCINA_NO_FUSED_PREFILL` is a structural no-op on
Qwen3.5.

**Hypothesis (NOT yet proven causality — review finding #1):** that the missing fusion
explains the N=32 TTFT loss. The legacy 13× win was decode-during-ingestion; if 32
requests arrive together with no decode active, co-batching gives no initial benefit,
and N=32 TTFT may instead be dominated by serial prompt admission, lack of batched
prefill, launch overhead, or GDN scan behavior. **P0 profiling must attribute the 1923 ms
before any implementation.** Similarly, "decode is bandwidth-saturated" (proven only for
the grouped-expert GEMM at ~81% of peak) does not preclude losses from batch shapes,
state movement, kernel gaps, or synchronization — the dense N=32 −93% needs a timeline,
not a bandwidth assertion.

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
- **Acceptance (review-strengthened):** full make + all remaining gates green,
  explicitly including the Gemma, diffusion, and e4b suites (legacy-qwen3 test deletion
  must not drop coverage of shared mmvq/routing/cache paths those engines use — verify
  via call-graph/symbol audit, not grep); Qwen3.5 bench numbers unchanged pre/post-prune
  (same binary protocol, isolated benchmarked series); grep is a smoke check only.

## Part 2 — beat vLLM on every configuration (re-sequenced per review)

- **P4. Evidence archive** into `benchmark-evidence/results/2026-07-11-qwen35-vs-vllm/`:
  raw logs, exact commits/flags, metric definitions (TTFT statistic: report median AND
  p95; synchronized vs staggered arrivals stated per table). Must resolve the N=32
  baseline provenance question (291.7 current vs 212.6 in the older gate doc — protocol
  or commit mismatch, identify which).
- **P3. Protection gate — BEFORE P1, dual-sided.** One script, both models, N=1–32,
  TTFT(median+p95) + decode + aggregate. Two assertions per cell: (a) absolute floor —
  no metric regresses >5% vs frozen raw baseline; (b) competitive margin — in cells we
  claim as wins, fucina must beat a contemporaneous protocol-matched vLLM run, not a
  historical number. Runs before every merge to main.
- **P0. Profile N=32 TTFT + dense N≥16 decode (NEW, gates P1's design).**
  CPU/GPU timeline of a 32-burst arrival: prefill admission order, active decode rows
  during ingestion, kernel gaps, per-request TTFT distribution; dense: bytes/token and
  batch-shape utilization vs vLLM. Output: attribution of the 1923 ms into
  admission-serialization vs missing-fusion vs launch overhead vs GDN scan — this
  decides whether P1 needs chunked scheduling, batched prefill, fusion, or all three.
- **P1. Implement what P0 proves** (likely some combination of chunked prefill
  scheduling + fused prefill/decode on the M4 engine, behind a rollback env toggle).
  - Correctness matrix (review-expanded, replaces the too-narrow byte-identical pair):
    chunk sizes {1, odd, prompt_len−1, default}, heterogeneous batches, repeated
    prefill/decode interleaving, GDN recurrent+conv state and KV snapshot/restore
    across chunk boundaries, cancellation mid-prefill, session save/load interaction,
    numerical comparison vs an independent standalone reference. Exact byte equality
    asserted only where identical operation order is genuinely promised; tolerance
    bounds elsewhere (changed batching legitimately reorders FP reductions).
  - Exit: N=32 TTFT(median) into the vLLM band measured contemporaneously (not the
    historical 664 ms), p95 reported, ALL of N=1–32 inside the P3 gate on both models.
- **P2. Dense N≥16 residual gap.** Diagnosis starts in P0 (not after P1). Kernel
  changes remain NO-GO until profiling names a specific residual bottleneck.

## Sequencing (rev 2)

P4 (evidence + baseline-provenance fix) → P3 (harness + frozen baselines) →
P0 (profiling/attribution) → P1 (implement what P0 proves, gated by P3) →
P2 (residual dense gap) → R1–R3 prune (separately benchmarked series, protected by P3).

Rationale: the guardrail exists before the lever is pulled; the lever is designed from
attribution, not assumption; prune last — the legacy fused path is the port's reference
implementation AND pruning mid-push would complicate bisects.

## Non-goals

- No decode-kernel tuning on the MoE grouped-expert GEMM — measured at ~81% of LPDDR5X
  bandwidth peak. Mixer/LM-head are deprioritized on weaker evidence (one failed
  experiment each, not a measured floor — review finding); they may re-enter only via
  P0 profiling naming them.
- No spec-on-MoE until P1 lands and acceptance is measured (deferred, unchanged).
- Distributed (Phase E) continues independently; not a lever for single-node vLLM parity.
