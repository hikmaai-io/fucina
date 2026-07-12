# SOTA GB10 Blackwell vertical inference for Qwen3.5 — mission plan

Status: ACTIVE (2026-07-12). Supersedes the tactical parts of
`qwen35-only-beat-vllm-plan.md` (rev 2) and folds in the vLLM subsystem analysis
(`/tmp/vllm-analysis-report.md`, vLLM @ 5f8e73cb).

## Mission

Be the state-of-the-art inference engine for Qwen3.5 on a single NVIDIA GB10
(DGX Spark, 48 SMs, sm_121a, ~273 GB/s LPDDR5X unified memory, 128 GB, native
FP4/FP8) — beating vLLM on **every** concurrency/model configuration while
keeping fucina's differentiators: byte-identical determinism, session
persistence, prefix-reuse KV, and a Go+CUDA host with no interpreter tax.

## Where we are (measured, post-P1)

**Qwen3.5-35B-A3B FP8 (MoE)** — fucina wins every aggregate cell and every TTFT
cell except N=32 (866 vs 664 ms). Single-stream +32%.
**Qwen3.5-9B FP8 (dense)** — fucina wins N≤8; loses N≥16 aggregate (vLLM +93% at
N=32). Root cause **measured** (P2, `1cc35ed`): dense decode re-reads the mixer
weight set 3× per step at B=30 (chunked 16+8+6) and the LM head 2×, at 16.6%
occupancy — 37% of peak is weight-re-read + register pressure, NOT launch gaps
(GPU busy 84.3/84.7 ms) and NOT physics.

Remaining losses to close:
- **L-dense**: dense N≥16 aggregate (the big one; P2 owns it, fixes designed).
- **L-moe-ttft**: MoE N=32 TTFT 866 vs 664 ms (last TTFT hold-out).
- **L-moe-lowc**: MoE N=2/4 cells (small-batch efficiency).

## Finishing the in-flight work — DONE (2026-07-12, merged to main 27b686f)

- **P2 F1+F2 — SHIPPED** (`f6c5634` head, `0ea4604` mixer). Measured on
  Qwen3.5-9B-FP8 vs the 1cc35ed baseline, 96-step apples-to-apples:
  **B=30 (the served avgB, 3-pass ladder case): 330.6 → 425.0 tok/s (+28.6%)**,
  step 91.6 → 70.6 ms; B=32 +4.4%, B=16 +2.0%. Both fixes bitwise-identical by
  construction; losslessness gate PASS (dense logit rel ≤0.0029 unchanged). This
  closes most of L-dense at the batch that matters for serving.
- **Prune acceptance — GREEN, merged** (`chore/prune-legacy-qwen3`, 4 commits):
  diffusion kernel gates + Gemma e2e + qwen35 determinism all PASS; the two
  suite flags (dg-bf16 transient contention, e4b-MTP-checkpoint detector) proven
  model-availability not code. fucina is now Qwen3.5-only.
- Merged main 27b686f: full build green, Go tests green, multiseq-prefill gate
  PASS, pushed. Baselines re-freeze pending a fresh serving sweep.

## The SOTA levers (from the vLLM analysis, ranked by measured leverage)

The analysis' central finding: on GB10 unified memory, ~half of vLLM's MRV2
cleverness solves a PCIe/Python problem fucina doesn't have. Three patterns
transfer; the rest is explicitly rejected.

### S1 — DFlash-style parallel speculative decoding (HIGHEST value, decode ceiling)
Draft N tokens in ONE non-causal forward of a small draft stack, uniform
fixed shape (1+N tokens/req), verify with a single (1+N)-token target pass +
rejection sampling. This attacks the one thing GB10 cannot be tuned around —
the 273 GB/s bandwidth ceiling — by materializing ~4–6.5 target tokens per
target weight-read.
- **Why now applicable**: a Qwen3.5-9B draft checkpoint exists
  (`z-lab/Qwen3.5-9B-DFlash`); a GDN-hybrid sibling proves the pattern works on
  Qwen3-Next-style hybrids (`z-lab/Qwen3-Coder-Next-DFlash`).
- **fucina fit**: uniform shapes slot into existing CUDA-graph decode; the
  stateless `(seed, position)` Gumbel scheme keeps byte-identical determinism
  THROUGH rejection sampling (draft & verifier derive identical keys
  independently).
- **Decomposition (derisked, dense-first)**:
  - S1a (M): dense-9B DFlash — context-KV precompute (one fused GEMM over
    stacked per-layer KV weights + grouped K-norm + batched RoPE), fixed
    (1+N)-token draft forward, rejection sampler. Greedy first (deterministic
    outright), then probabilistic via shared Gumbel keys.
  - S1b (M): GDN recurrent-state snapshot/rewind to the last accepted token —
    the single hardest piece, shared by any speculative method on the hybrid;
    extends S1a to 35B-A3B. **Correctness prerequisite** (rejected drafts must
    not corrupt in-place GDN state).
  - S1c (S): scheduler N+1 lookahead KV slots (trivial on prefix-reuse cache).
  - S1d (S): concurrency gate — spec decode goes net-NEGATIVE past ~B=8 on 48
    SMs (verification inflates compute); enable only at low batch, empirically
    tuned (vLLM's dynamic-SD `[start_bs,end_bs,K]` table is the reference).
- **Expected**: dense-9B single-stream ~30 → ~150 tok/s (accept_len≈4 chat,
  ≈6.5 code); indirectly lifts **L-moe-lowc** (a verify pass at N=2 looks like a
  2×(1+N) batch to the grouped GEMM, moving those cells up the efficiency curve).
- **Risk**: checkpoint conversion into fucina's format (uniform-RoPE assumption
  across draft layers); GDN rollback correctness vs the determinism guarantee;
  net-negative past the critical batch (hence S1d).

### S2 — MRV2 selective adoption (S/M, enables S1 + future-proofs graphs)
Adopt exactly three ideas; reject the PCIe machinery.
- S2a (M): **GPU-native input splicing** — one CUDA kernel derives next-step
  `input_ids`/positions from persistent state on-GPU (splice last sampled/
  accepted token device-side). This is what lets the WHOLE decode step
  (including varying accepted-token counts) be a replayable CUDA graph — a hard
  prerequisite for S1's zero-sync draft loop. Determinism unaffected (pure
  function of state).
- S2b (S): **CUDA-graph key scheme** `(num_tokens, num_reqs,
  uniform_token_count)` with dominance dispatch + decode-first batch ordering —
  fucina needs `uniform_token_count` the day S1 lands; retrofitting is painful.
- S2c (S, conditional): permanent-slot request state IF fucina still
  compacts/reorders batch state on admit/exit.
- **Explicitly NOT adopted**: `StagedWriteTensor`, `UvaBufferPool`,
  pinned-staging pools, UVA-vs-GPU residency policy, execute/sample RPC split,
  FlashMLA/MLA backends, whole-hog MRV2 rewrite. All are PCIe/Python-tax
  workarounds moot on unified memory.

### S3 — DSpark Markov head (DEFER)
DFlash + a low-rank Markov transition bias for intra-block dependency. Small
delta on top of S1, but **no Qwen3.5 Markov-head checkpoint exists** — training
one is out of scope. Revisit only if S1 acceptance on real workloads is
insufficient at N=7–16.

## The gating discriminator (do BEFORE committing S1/S2 effort)

The analysis' own adversarial self-check: if the dense N≥16 gap were
launch-gap/host-serialization, MRV2 async+full-graph would jump to #1. **P2
already ruled this out** (GPU busy 84.3/84.7 ms, 0.45 ms gap) — the gap is
kernel weight-re-read. So: **P2 F1+F2 is the correct first lever for L-dense,
not MRV2 async.** S1/S2 are the next frontier AFTER the measured kernel fixes
land — they attack a different regime (low-concurrency ceiling), not the N≥16
gap.

## Sequencing

1. P2 F1+F2 (finish in-flight; closes L-dense) — gated, pushed.
2. Prune e4b/diffusion gate → merge `chore/prune-legacy-qwen3`.
3. Merge `perf/qwen35-fused-prefill` + `perf/qwen35-dense-decode` to main;
   re-freeze protection-gate baselines.
4. S2a+S2b (graph-replayable full decode step + graph-key scheme) — the S1
   substrate.
5. S1a (dense-9B DFlash, greedy → probabilistic) behind a rollback toggle,
   concurrency-gated (S1d). Gate: determinism + protection band + a NEW
   spec-acceptance gate (accept_len vs the z-lab reference, ≥95%).
6. S1b (GDN state rewind) → extend S1 to 35B-A3B.
7. Re-benchmark all configs vs a contemporaneous vLLM; update the scoreboard.
8. Re-assess L-moe-ttft and MoE N=2/4 against the post-S1 numbers.

## Non-goals / rejected (measured or architecturally moot)
- MoE grouped-expert GEMM tuning (81% of peak, floored — three debunks).
- MoE mixer/LM-head decode-kernel tuning beyond P2's weight-re-read fix.
- Any PCIe-era MRV2 machinery (see S2 reject list).
- FlashMLA (no MLA layers in Qwen3.5).
- Spec decode at high concurrency (net-negative past ~B=8 on 48 SMs).

## Correctness invariants (non-negotiable, every lever)
- Byte-identical run-to-run determinism preserved (the differentiator).
- TDD: parity/acceptance gate written before the kernel it guards.
- Protection gate (absolute 5% floor + contemporaneous vLLM margin, median+p95
  TTFT) green on both models before any merge to main.
- C-style per-token hot paths; config-not-constants; conventional commits.
