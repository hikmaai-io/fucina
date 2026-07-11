# P0 attribution + P1 design proposal (Qwen3.5, plan rev 2)

Status: PROPOSAL — awaiting review before any P1 implementation (per rev-2 sequencing).
Branch: `perf/qwen35-fused-prefill`. Evidence: `benchmark-evidence/results/2026-07-11-qwen35-vs-vllm/`.

## P0 — attribution of the losses (profile-first, as rev 2 demanded)

### Method
Fresh P3 baseline sweeps on the built branch (`bench_serving.py`, canonical protocol:
16 cycled SHORT diverse prompts ~10–15 tok, `--conc 1,2,4,8,16,32`, `--long-tokens 3500`
probe, `--max-tokens 128`). Enhanced harness now records p95 + full per-request TTFT
arrays. Scheduler telemetry captured per run (`*.telemetry.log`). Box caveat: harbor
CPU testers were active (no GPU compute); admission timing is CPU-sensitive, but the
dominant term below is GPU engine time, robust to that.

### L1 — MoE N=32 TTFT (1892 ms median / 1957 ms p95 vs vLLM ~664 ms). ATTRIBUTED.
**Cause: serial single-sequence admission-prefill, NOT missing fusion.**

Evidence:
- Per-request TTFT staircase at N=32 (sorted ms): `217, 536, 764, 1033, 1367, 1424,`
  then a cluster of ~26 at `1882–1959`. Not 32 evenly-spaced round-trips — a serial
  ramp that saturates.
- Telemetry, single-seq window: `avgB 1.0, engine 19.2 ms/step, admit 52.07 ms
  (eng 51.5)`. **One `AddSeq` of a ~15-token prompt through the 35B costs ~52 ms** —
  it re-pays the full MoE weight-dequant / per-layer fixed cost for a tiny forward.
- `admit()` (scheduler.go:869-872) admits an idle burst in ONE pass (`oneShotCap =
  Capacity`), but the loop calls `s.engine.AddSeq` **sequentially** (line 922) — 32
  short prompts ⇒ 32 back-to-back single-seq prefill forwards ≈ 32×~52-59 ms ≈ 1.9 s.
  `chunkMin=256` routes short prompts to this one-shot path (line 897 condition false).
- During this window there is **no active decode** (avgB ramps 1→1.5→4→9.6 only as
  prefills finish), so co-batching a prefill with decode (Stage-18 fusion) has nothing
  to fuse with. **The rev-1 "fusion explains TTFT" hypothesis is falsified for this
  workload.**
- vLLM keeps TTFT ~664 ms because it batches all pending prefills into one forward
  (`max_num_batched_tokens=2048` ≥ 32×15) — one batched prefill, not 32 serial ones.

### L2 — dense 9B N≥16 aggregate (260 tok/s vs vLLM 502 @ N=32). PARTIALLY ATTRIBUTED.
- Dense N=32 also shows the same TTFT serialization (2898 ms — worse per-prefill, the
  9B dense forward is heavier than MoE's 3B-active).
- Decode side: telemetry at avgB≈30 shows `engine 88–90 ms/step`. A B=30 decode reads
  the ~9 GB FP8 weight set once ⇒ 9 GB / 88 ms ≈ **102 GB/s effective ≈ 37% of LPDDR5X
  peak** — i.e. real headroom (contrast the MoE expert GEMM at ~81% of peak, a floor).
  So dense has BOTH a TTFT-serialization component (L1-style) AND a genuine decode
  batch-shape/efficiency gap. Naming the region is P0's job; the precise ncu (bytes/token,
  occupancy, kernel breakdown) is the P2-entry diagnosis before any kernel change.

### L3 — MoE N=32 aggregate (−4%, within noise). Not a standalone lever; expected to
move with L1 (removing the 1.9 s serial-prefill stall recovers admission wall-time that
sits in the agg_decode_tps denominator).

### What P0 rules OUT
- Missing GDN fusion as the L1 cause (no decode active during burst ingestion).
- Admission round-trips / scheduler chattiness (burst admits in one pass already).
- MoE expert-GEMM decode kernel (at ~81% BW floor; unchanged, non-goal).

## P1 design proposal (implement what P0 proved — pending review)

**Primary lever: batched multi-sequence PREFILL admission** (closes L1; the measured gap).
Replace the serial `for … AddSeq` in the idle-burst branch with a single batched prefill
that ingests up to K pending prompts' tokens in ONE forward, amortizing the MoE
weight-dequant across all rows — the qwen35 analog of vLLM chunked prefill.

- Mechanism: a `qwen35_step_prefill_multiseq(slots[], tokens[][], lens[])` that lays out
  `Σ lenᵢ` rows across sequences, runs the projection/FFN/FULL-attn GEMMs batched over all
  rows (already ragged-position capable), and runs the **GDN/conv chunked scan per
  sequence** (grid over sequences; each sequence's tokens are one contiguous 64-token
  chunked run — the exact §4 constraint the exploration surfaced). Sample each sequence's
  first token from its last row. This is the SAME "split the recurrence, batch the
  projections" primitive the fused port needs — build it here first, all-prefill (no
  decode rows), which is strictly simpler (no rng/lockstep-with-decode concerns).
- Scope: batch only same-ish-length short prompts per pass (K bounded by row budget /
  `M2` geometry); long prompts keep the chunked path.

**Secondary lever (only if a decode-during-long-prompt workload is a target): GDN fused
prefill+decode** — the original P1. Deprioritized: the canonical benchmark is short-prompt
bursts where it does nothing. Worth it only if the product needs low decode-jitter while a
long prompt streams in mid-conversation (the staggered case). Same recurrence-split
primitive, plus decode rows + `rng_off=NULL` losslessness (legacy reference maps cleanly).

**Dense L2 residual (P2, not P1):** after batched prefill lands, re-measure dense N≥16; if
a decode gap remains, ncu the B=30 dense step (the 37%-of-peak finding) before any kernel
change.

### Correctness matrix for the primary lever (rev-2 expanded)
Batched multi-seq prefill == per-sequence standalone prefill:
- first token + ≥20-tok greedy continuation per sequence byte-identical to standalone
  `seq_add`/`seq_prefill_chunk` (byte-equality valid: per-seq GDN scan order is identical
  to standalone; only cross-seq batching of the shared projection GEMMs changes, and those
  are per-row independent — assert byte-identical, fall to tolerance only if a shared
  reduction reorders).
- chunk/prompt lengths {1, odd, `M2_CHUNK`−1, `M2_CHUNK`, `M2_CHUNK`+1, 256}, heterogeneous
  lengths in one batch, K = {2, 8, row-budget}.
- GDN recurrent + conv-ring + FULL-attn KV state at each sequence's final position ==
  standalone (via `q35_state_save` snapshot compare).
- cancellation of one sequence mid-batch leaves the others bit-identical.
- session save/load of a batch-prefilled sequence == save/load of a standalone-prefilled one.

### Exit criteria (unchanged intent, measured contemporaneously)
- MoE N=32 median TTFT into vLLM's contemporaneous band (~660 ms), p95 reported.
- ALL of N=1–32 pass the P3 dual-sided gate on both models (no >5% floor regression; no
  loss of the claimed-win cells: MoE N=8, dense N=2/4/8).
- Behind a rollback env toggle; full gate suite green before merge.
