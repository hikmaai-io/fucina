# SOTA GB10 Blackwell vertical inference for Qwen3.5 — mission plan

Status: **CONCLUDED (2026-07-18) — see the Official Position below.**
Supersedes the tactical parts of `qwen35-only-beat-vllm-plan.md` (rev 2) and
folds in the vLLM subsystem analysis (vLLM @ 5f8e73cb).

---

## OFFICIAL POSITION (enshrined 2026-07-18, main @ 82a6392)

**fucina is the state-of-the-art Qwen3.5 inference engine on the NVIDIA GB10,
winning 11 of 12 concurrency cells against fresh contemporaneous vLLM
(2026-07-18) while providing byte-identical run-to-run determinism that vLLM
does not offer. The 12th cell is the measured, proven price of that guarantee
— and we choose the guarantee.**

### The scoreboard (fresh contemporaneous vLLM, 2026-07-18, quiescent GB10)

**Qwen3.5-35B-A3B-FP8 (MoE) — fucina sweeps all 6 cells:**

| N | fucina | vLLM | margin |
|---|---|---|---|
| 1 | 59.0 | 14.0* | — |
| 2 | 101.4 | 74.1 | +37% |
| 4 | 134.0 | 111.7 | +20% |
| 8 | 229.8 | 155.2 | +48% |
| 16 | 320.1 | 207.2 | +55% |
| 32 | 472.4 | 321.3 | +47% |

**Qwen3.5-9B-FP8 (dense) — fucina wins 5 of 6:**

| N | fucina | vLLM | margin |
|---|---|---|---|
| 2 | 59.3 | 44.2 | +34% |
| 4 | 117.3 | 85.8 | +37% |
| 8 | 204.6 | 164.4 | +24% |
| 16 | 313.1 | 280.8 | +12% |
| 32 | 438.8 | 521.8 | **−16% (the accepted trade-off)** |

Plus: MoE N=32 TTFT 641/647 ms (med/p95) vs vLLM 664; single-stream, long-prompt
TTFT, and warm/state-cache TTFT all won. Evidence:
`benchmark-evidence/results/2026-07-18-d32b/`, `docs/qwen35-d32b.md`.

### Why the 12th cell is a principled boundary, not a tuning gap

Triply proven (D32 `docs/qwen35-dense32.md` §5, D32B `docs/qwen35-d32b.md`
§§1-5b):

1. **Not bytes**: fucina reads 44% FEWER weight bytes (Q4_K 3.65 GiB vs FP8
   ~6.5 GiB). If the gap were bytes, fucina would win.
2. **Not bandwidth**: both engines run ~2× above the 785–974 tok/s memory
   floor; the mixer is cache-latency-bound (L1/L2 hit ~90%), not DRAM-bound.
3. **Not tuning**: seven ILP levers systematically measured (DPSPLIT, PIPE
   depth, 2/4-rows-per-warp, `__ldg`, cp.async, NWARPS, full BIGCHUNK×MINBLK
   cross). The one winner (+9.1%, occupancy 53→86%) is shipped. The register
   file is the quantitatively proven binding constraint (exactly 4 blocks/SM at
   the 64-reg minimum the bit-identical arithmetic requires); every pipe is
   <15% utilized; the residual stall is irreducible dependency latency.
4. **The boundary**: closing the last −16% requires a tensor-core mixer whose
   MMA reduction order ≠ the warp-serial dp4a order → cannot be
   bitwise-identical → would break the byte-identity gate. The gate is the
   product; the cell is the price. **Decision: keep the gate.**

### What the determinism guarantee buys (what vLLM cannot claim)

- Byte-identical run-to-run outputs (FNV stream-hash-verified per kernel
  change; golden hashes at B=1/8/16/32).
- Reproducible sessions: save/restore + prefix-reuse KV with zero re-prefill
  and identical continuations.
- Auditable serving: every perf lever shipped under a losslessness gate
  (unchanged logit bounds, 0 first-token flips outside documented MoE
  expert-flip cells).

### Standing rules going forward

1. The byte-identity gate is not to be weakened by default-path changes. Any
   future tensor-core mixer must be a new, explicitly sanctioned, default-off
   mode with its own measured bound — a mission amendment, not an optimization.
2. The protection gate (absolute floors + contemporaneous vLLM margins,
   re-frozen 2026-07-18) guards all 11 winning cells on every merge.
3. The scoreboard claim must always cite fresh contemporaneous vLLM numbers,
   never carried-forward columns.

---

## UPDATE 2026-07-13: S1 DFlash verdict + pivot

**S1 spec decode (DFlash) is CORRECT but NOT a net perf win on GB10 — shelved as a
SOTA lever.** Branch `feat/qwen35-dflash` @ `1413cea` (58 commits, pushed, NOT
merged): lossless GDN rollback + deterministic RNG + loader all certified;
greedy byte-identical to greedy decode (3.556 accept/step), probabilistic
TV=0.0015; L1 draft-head tensor-core opt kept (435→408 ms/step, lossless).

**The measured ceiling (why we pivot):** B=1 greedy DFlash is *structurally* ≥
plain decode because the lossless commit re-decodes the accepted prefix as j
sequential single-token decodes (293 of 408 ms/step); at accept 9.2 that is
≥31.8 ms/emitted-token even with a free draft+verify, vs 29 ms plain. The
batched (T=17) verify argmax diverges from single-token decode on 2.0% of rows,
forcing the re-decode. And at B>1, **plain batched decode already amortizes
weights** (B=8: 4.35 ms/token), so batched DFlash only ~matches it. This confirms
the vLLM-analysis prediction: on a 273 GB/s bandwidth-bound engine, batching
already captures spec-decode's weight-amortization win; spec decode has a low
ceiling here.

**Decision: pivot the SOTA effort to the remaining MEASURED vLLM gaps** rather
than chase spec decode. Only surviving DFlash lever (deferred, not scheduled): a
bit-identical `(1+K)` batched verify enabling re-decode-free fast-commit (~1.2×
at B=1 interactive) — a large separate workstream, low priority.

### New priority order (evidence-ranked)
1. ~~**M-TTFT**~~ **DONE (merged 60b109a)**: N=32 TTFT 866→641/647 ms serving-confirmed,
   below vLLM 664 on median AND p95. All 6 MoE cells now won.
2. ~~MoE N=2/4~~ **CLOSED (merged 7496820)**: already won (106.8/161.5 vs 71.1/105.0);
   kernel path measured at floor (NVFP4 grouped 80–85% peak; Q8 head debunked).
3. ~~Re-baseline~~ **DONE (merged d9333ee)**: 2026-07-13 sweep archived. Caveat:
   vLLM column carried forward from 07-11 — rerun the container before freezing
   the official claim.
4. **D32 (ACTIVE)**: dense N=32 aggregate — THE ONLY LOSING CELL LEFT
   (392.4 vs vLLM 501.9; was 303 pre-P2 — gap halved, not closed).
   CORRECTION (code-verified): the F2 multi-chunk dispatch ALREADY covers
   K=32 (switch K≤8, default → multi_kernel<8>, ceil(K/8) chunks, weight
   read once) — yet B=32 gained only +4.4% from F2 vs +28.6% at B=30. The
   remaining ~2×-above-floor step time (~80 ms measured vs ~28–35 ms
   theoretical for 3.89 GB mixer + 2.03 GB head + 1.5 GB GDN state at
   273 GB/s) has an UNKNOWN dominant cost — requires fresh ncu/nsys
   attribution at B=32 on the post-F1/F2 build before any fix. No assumed
   lever. Branch `perf/qwen35-dense32`.
5. Phase E distributed (parked).

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
