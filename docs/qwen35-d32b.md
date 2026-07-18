<!-- ABOUTME: D32B — mixer ILP restructure (latency-hiding, byte-identity-preserving) + fresh
     ABOUTME: contemporaneous vLLM re-baseline for the last losing cell (dense Qwen3.5-9B N=32). -->
# D32B — dense Qwen3.5-9B-FP8 N=32: mixer ILP + fresh vLLM re-baseline

Status: RESOLVED — **D32B_BLOCKED** (2026-07-18). Branch `perf/qwen35-mixer-ilp`,
worktree `/home/mauromedda/hack/fucina-d32b`, from main `5bca236`. Shipped a mixer
ILP occupancy lever (**BIGCHUNK=12 + MINBLK=4**, bit-identical, +9.1% bench / +8%
serving on the last cell, zero regression) and captured the **fresh contemporaneous
vLLM numbers** (the mission's real target). Dense N=32 gap narrowed from **−22% →
−16%** but is **NOT flipped**: the dp4a warp-per-row GEMV is now occupancy-saturated
and the residual is the same tensor-core-GEMM-vs-dp4a **kernel-class** deficit D32
identified — closing it needs a tensor-core mixer that breaks the byte-identity gate
(forbidden). Every other cell (all MoE, dense N=1..16) is a fucina WIN vs fresh vLLM.

## 0. Foundation (from D32_BLOCKED, docs/qwen35-dense32.md §5)

D32 proved the dense N=32 residual is **NOT bytes** (fucina reads 44% FEWER weight
bytes: Q4_K 3.65 GiB vs FP8 ~6.5 GiB) and **NOT bandwidth** (both engines ~2× above
the 785–974 tok/s memory floor). The mixer `mmvq_q4_k_packedT_multi_kernel` is
**latency-bound**: one warp per output row, 32 tokens accumulated via serial `__dp4a`.
D32B's mandate: latency-hiding restructures that PRESERVE the per-(row,token)
ascending-b accumulation order (LEGAL, byte-identity verifiable) — no tensor-core
mixer (FORBIDDEN, different MMA reduction order breaks the gate).

## 1. Attribution — the mixer stall is MEMORY-LOAD LATENCY, not the ALU chain (ncu, MEASURED)

Baseline (D32 NK=16 mixer, `<16,3>`) at B=32, steady-state decode, GB10 sm_121a,
CUDA 13, ncu under sudo (`RmProfilingAdminOnly=1`):

| metric | baseline `<16,3>` | meaning |
|---|---|---|
| inst/cycle (issued) | **0.20–0.26** | SM issue starved |
| warp latency / inst issued | 23–28 cyc | each issue waits ~a load latency |
| **stall: long_scoreboard** | **16.7–20.4 cyc/issue** | **dominant — memory-load latency** |
| stall: wait (ALU dep) | 2.4 cyc/issue | ALU dependency chain is NOT the bottleneck |
| stall: short_scoreboard | 1.4 cyc/issue | — |
| sm__warps_active | 53–57% | register-capped at **77 regs** |
| sm__throughput | 22–32% | not compute-bound |
| L1 hit / L2 hit | 90% / 85–89% | not DRAM-bound (loads served from cache) |

**The binding constraint is load-issue latency the occupancy (53%) cannot hide** —
not the dp4a serial chain (wait=2.4), not DRAM bandwidth (L1/L2 hit ~90%).

## 2. ILP variants — each measured independently (bit-identity MEASURED, not assumed)

Golden byte-identity hashes (FNV-1a greedy token stream, `test_qwen35_dense32_byteident`,
24 steps), captured on the BASE build, verified UNCHANGED for EVERY variant below:

| B | streamhash |
|---|---|
| 32 | **c6ab45eab1f2751c** (canonical D32 hash) |
| 16 | f12bef42220457ea |
| 8  | f14062748dfbf4cd |
| 1  | c9d58638316db93a |

All variants are config-selectable via env (measured-best defaults baked in):
`FUCINA_Q4K_{BIGCHUNK,MINBLK,DPSPLIT,PIPE,NWARPS,2ROW}`.

### (1) DPSPLIT — split the 8-deep serial `__dp4a` chain — MARGINAL (+0.6%), confirms the diagnosis

The inner loop threads one `sumi` through 8 serial `__dp4a` (an 8-long ALU dep chain).
DPSPLIT={2,4} breaks it into 2/4 INDEPENDENT int32 partials summed at the end.
Bit-identical by construction (two's-complement add is associative & commutative — the
final `sumi` is the sum of the SAME 8 dot4 terms). Hash-verified all B.

| DPSPLIT | B=32 tok/s |
|---|---|
| 1 (serial) | 427.3 |
| 2 | 429.9 |
| 4 | 429.7 |

Only +0.6% — **confirms the ncu finding**: the stall is memory latency (long_scoreboard),
NOT the ALU chain (wait=2.4). Kept DPSPLIT=2 as a free bit-identical default.

### (2) PIPE — deeper superblock staging — REGRESSION, rejected

More in-flight LDGs per warp via PIPE=3/4 (was constexpr 2, now template arg).

| PIPE | B=32 tok/s |
|---|---|
| 2 | 430.3 |
| 3 | 340.7 (register spill) |
| 4 | 408.0 |

PIPE≥3 spills (matches the D32 doc's rejected PIPE=4 note). Kept PIPE=2.

**PIPE × MINBLK cross (BIGCHUNK=12) — lower MINBLK does NOT rescue deeper staging:**
The hypothesis that PIPE=3/4 spilled only because MINBLK=4 starved it of registers was
tested across the full cross — disproven. Peak is PIPE=2/MINBLK=4 at every point.

| B=32 tok/s | MINBLK=2 | MINBLK=3 | MINBLK=4 |
|---|---|---|---|
| PIPE=2 | 421 | 421 | **457** |
| PIPE=3 | 343 | 361 | 412 |
| PIPE=4 | 347 | 408 | 397 |

Deeper staging's extra `uint4 hh/hq × PIPE` registers crush occupancy faster than the
added in-flight loads help — no MINBLK point recovers it. All PIPE variants bit-identical
(hash c6ab45eab1f2751c). PIPE=2 confirmed optimal across the whole 2D sweep.

### (3) NWARPS — block size — 8 is best

| NWARPS | B=32 tok/s |
|---|---|
| 4 | 460.8 |
| 8 | **465.3** |
| 16 | FAIL (exceeds `__launch_bounds__(256)` = 8 warps) |

### (4) BIGCHUNK × MINBLK — THE WIN (occupancy lever)

MINBLK (the `__launch_bounds__(256, MINBLK)` min-blocks/SM target) forces the compiler
to fit more resident blocks by spilling fewer registers → higher occupancy → more warps
to hide the load latency. D32 shipped MINBLK=3; D32B finds **MINBLK=4** is the lever:

| config | B=32 tok/s | regs | warps_active | inst/cycle |
|---|---|---|---|---|
| BASE `<16,3>` (D32) | 426 | 77 | 53–57% | 0.20–0.26 |
| `<16,4>` | 449 | — | — | — |
| **`<12,4>` (D32B best)** | **465** | **64** | **70–86%** | **0.29–0.31** |
| `<8,4>` (4 chunks) | 398 | — | — | — |

**Complete BIGCHUNK curve (best MINBLK per tile) — NK=12 is the GLOBAL optimum:**
The full width sweep including the wide end (NK=24 = 2 chunks, NK=32 = 1 chunk = weights
dequantized exactly ONCE across all 32 tokens) confirms a single interior peak. All
bit-identical (hash c6ab45eab1f2751c).

| NK (chunks @ B=32) | best B=32 tok/s |
|---|---|
| 8 (4 chunks) | 398 |
| **12 (3 chunks)** | **465** ← peak |
| 16 (2 chunks) | 449 |
| 24 (2 chunks) | 416 |
| 32 (1 chunk, dequant-once) | 411 |

The wide tiles' acc[24]/acc[32] register pressure crushes occupancy faster than the
dequant-reuse benefit helps — even NK=32 (weights dequantized only once) loses to NK=12.
The register-vs-reuse tradeoff has one interior optimum at NK=12; candidate #4 exhausted.

The NK=12 tile (3 chunks @ B=32) at MINBLK=4 drops registers 77→64, lifts occupancy
53%→70–86%, and issue rate 0.20→0.31 → **B=32 426→465 (+9.1%)**, bit-identical.
NK=16 was register-crippled; NK=8's 4 chunks cost too much redundant L2/dequant.
MINBLK≥5 and NWARPS variations plateau (occupancy saturated at MINBLK=4).

### (5) 2-ROWS-PER-WARP (the doc's named candidate #1) — MEASURED DEAD (register cliff)

One warp owns 2 output rows, issuing the activation loads (A,B — which depend only on
token+b, NOT row) ONCE and feeding 2 independent weight streams, to amortize the
long-scoreboard latency. Implemented as `mmvq_q4_k_packedT_multi2row_kernel`, behind
`FUCINA_Q4K_2ROW=1`, bit-identical (all hashes match; per-row order untouched).

| config | B=32 tok/s | regs | warps_active | long_scoreboard |
|---|---|---|---|---|
| 2ROW `<8,3>` (NK=8, acc[16]) | 358 | 80 | **45%** | **17.4 (unchanged)** |
| 2ROW `<4,·>` (NK=4, acc[8], 8 chunks) | 185 | — | — | chunk-count dominated |
| 1-row `<12,4>` | 465 | 64 | 70–86% | 16.6 |

**Measured dead** (not merely asserted, as D32 did): doubling `acc[]` (2×NK) costs 80
regs → occupancy 45% (WORSE), and the load sharing does NOT cut the stall (L1 already
serves the duplicate at 90% hit — the latency is per-LDG-issue, not per-DRAM-fetch).
Dropping to NK=4 to relieve the register cliff (acc[8]) is far WORSE (185) — 8 chunks at
B=32 means 8× redundant weight re-reads + dequant, exactly the chunk-count cost D32's
BIGCHUNK analysis quantified. Neither 2-row corner beats the 1-row occupancy lever.
**Occupancy is the binding constraint; trading it for load-sharing loses.** This is the
key negative result: the latency wall is occupancy-bound, not load-count-bound.

### (6) `__ldg` on activation loads (read-only data cache) — MEASURED NEUTRAL

The activation loads (A,B) are the `long_scoreboard` stall source and were plain global
loads. Routing them through `__ldg` (read-only data cache) is bit-identical (same bytes).

| activation load path | B=32 tok/s | hash |
|---|---|---|
| plain `*(const uint4*)` (default) | 456–465 | c6ab45eab1f2751c |
| `__ldg` | 455.5 | c6ab45eab1f2751c |

**Neutral** (within run noise): the activations are already L1-served at 90% hit, so the
read-only path does not change the per-issue load latency. Confirms candidate #3 (wider
vectorized loads) is already satisfied — the kernel uses `uint4` 128-bit loads for BOTH
weights (hh/hq) and activations (A/B); there is no wider legal load. Reverted (no-op).

### (7) cp.async deep-prefetch (global→smem, register-bypass) — MEASURED COUNTERPRODUCTIVE

The canonical CUTLASS/Marlin latency-hiding technique, and the one lever that could beat
the PIPE spill: `__pipeline_memcpy_async` (cp.async) stages weights global→**shared**
DIRECTLY, bypassing registers, so prefetch depth decouples from register pressure. New
kernel `mmvq_q4_k_packedT_cpasync_kernel` (STAGES-deep smem ring), `FUCINA_Q4K_CPASYNC=1`,
`FUCINA_Q4K_STAGES={2,3,4,6}`. **Bit-identical** — hashes match all B (c6ab45eab1f2751c
@ B=32): cp.async only moves the SAME bytes earlier; the dp4a order is untouched.

| STAGES | B=32 tok/s | vs register-staged 465 |
|---|---|---|
| 2 | 350.6 | −25% |
| 3 | 343.6 | −26% |
| 4 | 308.5 | −34% |
| 6 | 274.0 | −41% |

**Slower, and worse with depth.** ncu (STAGES=2, `<12,4,2,2>`): long_scoreboard **30**
(vs 16 register-staged), inst/cycle **0.19-0.22** (vs 0.31), 64 regs, 66% occ. Root
cause: on GB10 the weights are already L1/L2-hot (89-90% hit) — the kernel is
**cache-latency-bound, not DRAM-bound**, so there is no DRAM latency for cp.async to
hide. `__pipeline_wait_prior` instead SERIALIZES the warp on the smem-arrival scoreboard
(a blocking wait counted as long_scoreboard) and the extra global→smem→register
round-trip adds latency the direct L1-hot load did not have. This is the definitive
negative result: **the very technique GEMMs use to hide DRAM latency backfires when the
bottleneck is cache-hit-latency**, confirming the D32 diagnosis that the mixer sits
~2× above the memory floor. Kept env-gated (default off) as evidence.

## 3. Final mixer config (shipped defaults) + zero-regression sweep

Defaults: **BIGCHUNK=12, MINBLK=4, DPSPLIT=2, PIPE=2, NWARPS=8** (K>16 path only;
K≤16 keeps the untouched NK=8 winning path). Batch bench, B=1..32, quiescent GB10:

| B | BASE (D32 default) | D32B default | Δ |
|---|---|---|---|
| 1  | 34.48 | 34.42 | ~0 (NK=1 path untouched) |
| 2  | 60.04 | 60.91 | ~0 |
| 4  | 120.04 | 121.79 | ~0 |
| 8  | 230.50 | 233.30 | ~0 |
| 16 | 352.94 | 347.55 | ~0 (NK=8 2-chunk path untouched) |
| 32 | 425.79 | **455–465** | **+9.1%** |

Zero regression N=1..16 (they route through the untouched NK≤8 path). B=32 hash
`c6ab45eab1f2751c` unchanged.

## 4. TASK B — FRESH contemporaneous vLLM (2026-07-18, the REAL target)

The mission's 501.9 was 07-11 (3 fucina generations old). Re-ran the canonical image
`hellohal2064/vllm-qwen3.5-gb10:latest` per the 07-11 recipe (MAX_MODEL_LEN 25280,
GPU_MEMORY_UTIL 0.88, ATTENTION_BACKEND FLASHINFER, GPU_CLOCK_MAX 2400, whole-repo
mount for the blobs symlink), `bench_serving.py` conc 1..32, diverse. **vLLM and fucina
runs strictly serialized** (docker stop + nvidia-smi quiescence check between phases —
vLLM ignores `/tmp/fucina_gpu.lock`).

**Fresh vLLM 2026-07-18 (agg_decode_tps | median/p95 TTFT ms):**

| N | dense 9B | MoE 35B |
|---|---|---|
| 1  | 10.2* | 14.0* (\*cold-start artifact) |
| 2  | 44.2 \| 79/79 | 74.1 \| 106/106 |
| 4  | 85.8 \| 114/135 | 111.7 \| 147/147 |
| 8  | 164.4 \| 176/176 | 155.2 \| 217/218 |
| 16 | 280.8 \| 208/210 | 207.2 \| 245/356 |
| 32 | **521.8** \| 213/218 | 321.3 \| 312/316 |

**Fresh vLLM dense N=32 = 521.8** — the target got HARDER than the mission's 501.9
(+4%). MoE N=32 = 321.3.

## 5. Head-to-head — fucina D32B vs FRESH vLLM (serving sweep, quiescent, 2026-07-18)

**Dense Qwen3.5-9B-FP8 (agg_decode_tps):**

| N | fucina D32B | fresh vLLM | verdict |
|---|---|---|---|
| 2  | 59.3  | 44.2  | **WIN +34%** |
| 4  | 117.3 | 85.8  | **WIN +37%** |
| 8  | 204.6 | 164.4 | **WIN +24%** |
| 16 | 313.1 | 280.8 | **WIN +12%** |
| 32 | **438.8** (rep2 433.3) | **521.8** | **LOSS −16%** |

**MoE Qwen3.5-35B-A3B-FP8 (agg_decode_tps):**

| N | fucina D32B | fresh vLLM | verdict |
|---|---|---|---|
| 1  | 59.0  | 14.0* | WIN |
| 2  | 101.4 | 74.1  | **WIN +37%** |
| 4  | 134.0 | 111.7 | **WIN +20%** |
| 8  | 229.8 | 155.2 | **WIN +48%** |
| 16 | 320.1 | 207.2 | **WIN +55%** |
| 32 | **472.4** | 321.3 | **WIN +47%** |

fucina sweeps ALL MoE cells and dense N=1..16. **Dense N=32 is the sole losing cell.**
Serving 438 vs bench 465 (the ~27 tok/s gap is admission/scheduler overhead in the
`agg_decode_tps` denominator — consistent with D32's bench-vs-serving gap).

Protection gate (`scripts/protection_gate.py`): **GATE PASS both models** — all
claimed-win cells beat the fresh contemporaneous vLLM; no floor regressions.

## 6. Gates — all green

- **Byte-identity**: FNV hash `c6ab45eab1f2751c` (B=32/24) unchanged for every variant;
  extended to B=1/8/16 golden set — all unchanged.
- **`make qwen35-multiseq-prefill-test`**: PASS both models. Dense 9B logit rel ≤0.0029
  (identical to the D32 bound), byte-identical determinism, 0 token flips; MoE only
  documented expert-flips.
- **`scripts/protection_gate.py`**: GATE PASS both models (§5).
- **No regression on any winning cell**: dense N=1..16 unchanged (NK≤8 path untouched);
  MoE B=32 decode 583.8 (≥ D32's 576.2 — experts use FP8 block GEMM, not the Q4_K mixer).
- **`go test ./...`**: PASS.
- **`make lib libdg fucina`**: clean (sm_121a cubin verified).

## 7. FINAL VERDICT — D32B_BLOCKED (improved, but the kernel-class wall stands)

D32B shipped a mixer ILP occupancy lever (**BIGCHUNK=12 + MINBLK=4**): dense N=32
bench **426→465 (+9.1%)**, serving **~406→438 (+8%)**, bit-identical, zero regression,
gates green. The gap vs the FRESH vLLM N=32 (521.8) narrowed from **−22% (07-13) →
−16%**.

**Why still blocked (measured, not assumed):**

| lever | result | reason |
|---|---|---|
| DPSPLIT (ALU chain) | +0.6% | stall is memory latency, not ALU (ncu: long_scoreboard 17 vs wait 2.4) |
| PIPE 3/4 | regression | register spill |
| NWARPS | 8 best | launch_bounds caps at 8 warps |
| **BIGCHUNK=12 + MINBLK=4** | **+9.1%** | **occupancy 53→70-86%, regs 77→64 — the win** |
| 2-rows-per-warp | −23% | register cliff (80 regs, occ 45%); load-sharing doesn't cut the stall |
| PIPE×MINBLK full cross | no better point | deeper staging unrescuable by lower MINBLK |
| `__ldg` activations | neutral | activations already L1-hot (90%) |
| cp.async deep-prefetch | −25 to −41% | cache-latency-bound, not DRAM-bound: smem round-trip + wait_prior backfire (long_scoreboard 16→30) |

After the occupancy lever the mixer sits at **SM 37%, warps 70–86%, inst/cycle 0.31**,
with the residual **long_scoreboard ~16 cyc/issue irreducible** for the dp4a
warp-per-row GEMV kernel class.

**The occupancy ceiling is REGISTER-BOUND, quantitatively proven** (ncu
`launch__occupancy_limit_*`, `<12,4,2,2>`): the resident-block limiter is **registers = 4
blocks/SM**, below warps (6) and block-count (24). At 64 regs/thread × 256 threads =
16,384 regs/block, the SM's 65,536-reg file admits exactly 4 blocks. To raise occupancy
further needs <64 regs/thread, but MINBLK=4 already cut 77→64 and lower forces spills
(measured regressions). The live set — `acc[NK=12]` + `wv[8]` + the Q4_K dequant
temporaries — is all essential to the bit-identical arithmetic; **no legal register
reduction remains**. The wall is structural, not a tuning gap. The
remaining −16% is the SAME tensor-core-GEMM-vs-dp4a-GEMV kernel-class deficit D32
identified: vLLM's B=32 tensor-core MMA computes all 32 tile-rows in one hardware pass;
the dp4a GEMV issues one warp per row and is bound by per-LDG-issue latency that
occupancy alone cannot fully hide at B=32. Closing it requires a tensor-core INT4/FP8
mixer whose MMA reduction order ≠ the warp-serial dp4a order → **cannot be
bitwise-identical** to the byte-identity gate → out of scope under the no-weakened-gate
rule.

The exact remaining gap: **fucina dense N=32 serving 438.8 vs fresh vLLM 521.8 = −16%**
(bench 465 vs 522 = −11%). Every other cell (all 6 MoE + dense N=1..16) is a fucina win
vs fresh contemporaneous vLLM. **D32B_BLOCKED** — the floor is not the constraint; the
byte-identity gate is, exactly as D32 concluded, now with the last legal ILP lever
(+9.1%) banked and the vLLM target freshly re-measured (and harder: 521.8, not 501.9).
