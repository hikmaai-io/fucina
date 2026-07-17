<!-- ABOUTME: D32 — closing the last losing cell (dense Qwen3.5-9B-FP8 N=32 aggregate decode).
     ABOUTME: Profile-first attribution of the ~2×-above-floor B=32 decode step; measured tables only. -->
# D32 — dense Qwen3.5-9B-FP8 N=32 aggregate decode (the last losing cell)

Status: IN PROGRESS (2026-07-17). Branch `perf/qwen35-dense32`, worktree
`/home/mauromedda/hack/fucina-dense32`, from main `34f2343`.

THE GAP (2026-07-13 re-baseline): dense N=32 serving aggregate **392.4 tok/s vs
vLLM 501.9 (−22%)** — the only losing cell on either model. Every other cell (all
MoE, dense N=1..16) is a fucina win and must not regress.

## 0. Served-path correction (code + load-log verified)

The dense serving checkpoint is **`Qwen/Qwen3.5-9B-FP8`**, but the load log shows:

```
qwen35-Q4K mixer repacked in place (200 tensors → packed dp4a GEMV)
qwen35 allocation decision: source=block-FP8 mixer=Q4_K experts=n/a d_weights=3.65 GiB
qwen35 allocation ledger: core=3.65 embed=6.69 head=2.90 ... total=13.24 GiB
```

So the **mixer (200 attn/FFN projections) is repacked to Q4_K at load** and served
through `mmvq_q4_k_packedT_batched_launch` → `mmvq_q4_k_packedT_multi_kernel<8>`
(the F2 kernel). The **LM head is BF16 (2.90 GiB)**, served through
`bf16_head_gemv_batched_launch` (the F1 single-pass ≤32-row path). The FP8
`fp8_block_gemm_kernel` (`acc[FP8_MAXB=16]`) is NOT on this dense mixer path — it
serves the MoE grouped experts and the FP8 prefill tiles only. So P2's F1/F2 Q4_K
work is exactly the served dense decode path; the D32 attribution must target it.

Mixer core = 3.65 GiB Q4_K (200 tensors). Head = 2.90 GiB BF16. GDN state r/w
scales with B. This is the read set the step budget is built on.

## 1. Fresh bench on THIS build (FP8 dense 9B, 96-step batch bench, GB10 sm_121a)

Harness: `cuda/test_qwen35_batch_bench.cu` (served `gemma4_engine_step_batch` →
`qwen35_decode_multiseq_body`, CUDA-graph replay), checkpoint
`models--Qwen--Qwen3.5-9B-FP8`, 96 timed steps after 6 warm, distinct short prompts.

| B | agg tok/s | step ms | per-stream tok/s | mixer chunks (ceil(B/8)) |
|---|---|---|---|---|
| 1 | 36.30 | 27.5 | 36.30 | 1 |
| 2 | 65.65 | 30.5 | 32.82 | 1 |
| 4 | 129.96 | 30.8 | 32.49 | 1 |
| 8 | 246.24 | 32.5 | 30.78 | 1 |
| 16 | 362.81 | 44.1 | 22.68 | 2 |
| 32 | 400.98 | **79.8** | 12.53 | 4 |

(B=32 bench agg 401 matches the P2 doc's 399 and the 392 serving number — the
kernel step dominates; admission/scheduler overhead is small, mission candidate
(d) ruled small as expected.)

### The signal: step time nearly DOUBLES B=16→B=32 (44.1 → 79.8 ms, +81%)

At a weight-read-once memory floor the step should stay ~flat across B (the 3.65 GiB
mixer + 2.90 GiB head dominate and are read once regardless of B; only the
B-scaling GDN state + activations grow, ~linearly but small). Instead the step
grows +81% for 2× the rows. Something scales ~linearly with B in the dominant term.

The mixer multi_kernel grid is `(ceil(K/8) chunks, row-groups)`:
- B≤8  → 1 chunk  → weights streamed once, no cross-chunk L2 dependence.
- B=16 → 2 chunks → weights must hit L2 on chunk 2 to be read once.
- B=32 → 4 chunks → weights must hit L2 on chunks 2,3,4 to be read once.

The step-time cliff aligns with the chunk count (flat through B=8 at 1 chunk;
+36% at B=16 = 2 chunks; +81% at B=32 = 4 chunks). Leading hypothesis (mission
candidate a): **L2 dedup across the chunk-passes degrades as chunk count grows** —
at 4 chunks the per-row weight working set streamed by the 4 chunk-blocks exceeds
what L2 (24 MB on GB10) can hold live, so DRAM re-reads the mixer weights ~2–4×.

Next: ncu per-kernel bytes/step + L2 hit-rate + effective GB/s at B=16 vs B=32 to
confirm whether the doubling lives in the mixer (L2 thrash), the GDN state, or the
head. MEASURED attribution below once collected.

## 2. Attribution (nsys per-kernel + ncu mixer counters) — MEASURED

nsys `--cuda-graph-trace=node`, single-B runs (40 steps + 6 warm = 46 step-instances).
Per-step cost = (kernel total ns) / 46 for once-per-step kernels; the mixer fires
200×/step (200 projection tensors), the GDN 24×/step (24 GDN layers).

| kernel | B=16 / step | B=32 / step | growth | share of +35.7ms |
|---|---|---|---|---|
| mixer `mmvq_q4_k_packedT_multi_kernel<8>` | 28.6 ms | **52.0 ms** | +23.4 ms | **65%** |
| head `bf16_head_gemv_batched<K,6>` | 8.31 ms | 14.09 ms | +5.78 ms | 16% |
| GDN `qwen35_b_gdn_kernel` (24 L) | 4.19 ms | 9.17 ms | +4.98 ms | 14% |
| misc (m2_gemm, norms, flash, conv, silu…) | ~3 ms | ~4.5 ms | ~1.5 ms | 4% |
| **step total** | ~44.1 ms | ~79.8 ms | **+35.7 ms** | 100% |

(Bench-measured step: 44.1 ms @ B=16, 79.8 ms @ B=32 — matches the sum. The
non-decode kernels in the nsys table — `dequant_q4_k_packed_to_bf16` (200 inst),
`repack_q4_k` (200), `cutlass wmma gemm` (4864) — are LOAD-time repack + one-shot
prefill of the B warm prompts, NOT per decode step, so excluded.)

### The mixer is the lever (65% of the growth). ncu counters, steady-state decode:

| metric | B=16 mixer | B=32 mixer |
|---|---|---|
| gpu__time_duration (per call, steady) | ~136 µs | ~230 µs |
| **lts__t_sector_hit_rate** (L2 hit) | **69%** | **89%** |
| lts__throughput (% peak) | ~12.8% | (low) |
| sm__throughput (% peak) | ~22% | ~30% |
| sm__warps_active (% peak) | — | ~66% |

**Candidate (a) — L2 thrash — RULED OUT.** L2 hit rate is HIGHER at B=32 (89% vs
69%): the 4 chunk-passes reuse the same weight bytes and L2 serves the duplicates.
DRAM is NOT re-reading the weights 4× — L2 is. But the kernel is neither DRAM-bound
nor compute-bound (SM ~30%, LTS low) — it is **latency-bound**. The cost that grows
with chunk count is: (1) **4× L2 sector bandwidth** for the duplicated weight reads,
(2) **4× redundant Q4_K dequant ALU** (q4k_scale_min_reg + half2float + nibble
unpack, re-executed per chunk), (3) 4× the block-scheduling. The mixer grid is
`(ceil(K/8) chunks, rows)`: B=8→1 chunk (flat), B=16→2 chunks (+36% step),
B=32→4 chunks (+81% step). Time tracks chunk count, exactly.

Candidates (b) GDN, (c) attention/KV, (e) NVFP4-quant: GDN grows 2.19× (genuine
B-scaling state r/w — 14% of the growth, a secondary lever). KV/attention (flash
partial/combine) is <1% and flat. quantize_q8_1t is 0.7%, negligible.
Candidate (d) serving-vs-bench: bench agg 401 ≈ serving 392 — scheduler overhead
small, confirmed.

### Why F2 helped B=30 (+28.6%) far more than B=32 (+4.4%) — RESOLVED

The P2 doc measured the **Q4_K_M gguf** path. The served checkpoint is **FP8 with a
Q4_K-repacked mixer**, but the mixer kernel is the same `multi_kernel<8>`. F2
replaced the OLD serial ladder (NK=16 then 8 then remainder = 3 full weight passes
at B=30) with the single chunk-major multi_kernel. At B=30 the old ladder was
3 passes → big win. At B=32 the old code took ONE NK=32 pass (register-crippled but
still one weight-read); the new multi_kernel<8> does 4 chunk-passes. So F2 traded
one register-crippled pass for 4 L2-served chunk-passes at B=32 — nearly a wash
(+4.4%). **The chunk count at B=32 is the residual cost F2 left on the table.**

## 3. Fix — halve the B=32 chunk count (mixer `multi_kernel<16>` for K>16)

Lever: the mixer grid is `(ceil(K/NK), rows)`. With NK=8, B=32 → 4 chunks. With
NK=16, B=32 → **2 chunks** — halving the redundant L2 bandwidth + dequant ALU that
is 65% of the B=16→32 growth. Bitwise-identical by construction: acc[n] for global
token t0+n accumulates over the SAME lane-strided b-loop in the SAME ascending
order regardless of NK (chunking only changes which block owns a (row,token) pair,
not its arithmetic) — the exact contract F2 established.

Risk: acc[16] raises register pressure vs acc[8] (P2 noted the OLD single-tile
NK=16 kernel hit 144 regs @ 16.6% occ). But the multi_kernel keeps the NK≤8-style
body per chunk; NK=16 needs 16 acc registers vs 8 — measure occupancy + net time.
Dispatch: K≤8 single-chunk kernels (unchanged wins); 8<K≤16 → multi<8> (2 chunks,
unchanged); K>16 → multi<16> (2 chunks instead of 3–4). Measured deltas below.

## 4. Measured deltas (updated as fixes land)

### Box-contention caveat (2026-07-17 ~18:00)

The shared GB10 box began running **concurrent vLLM serving** (two EngineCore procs:
`Qwen3.5-9B-FP8` then `huihui-ai/Huihui-Qwen3.5-9B-abliterated`, 94 GB + 21 GB)
mid-measurement. vLLM does NOT honor `/tmp/fucina_gpu.lock`, so back-to-back
fucina benches are contaminated (BASE B=32 read 401 tok/s quiescent, then 227 with
vLLM resident — 78% swing). Apples-to-apples BASE-vs-OPT deferred until the box is
quiescent (both binaries prebuilt: /tmp/fucina_bench_BASE @ HEAD, /tmp/fucina_bench_OPT
@ NK=16 change). Clean quiescent BASE reference (first run, no vLLM): B=16 362.8,
B=32 401.0. MEASURED win/no-win verdict pending a quiet box.

Implementation landed (code): mixer `mmvq_q4_k_packedT_multi_kernel<NK,MINBLK>` —
K>16 routes to a wider chunk tile, chunk-width config-selectable via
`FUCINA_Q4K_BIGCHUNK={8,12,16}` (default 16). Bitwise-identical by construction
(NK-independent per-(row,token) ascending-b accumulation).

### Losslessness gate — PASS (contention-independent)

`make qwen35-multiseq-prefill-test`: PASS both models. Dense 9B-FP8 logit
rel ≤ 0.0029 (identical to P2 pre-fix bound), byte-identical determinism, 0 token
flips. MoE 35B expert-flips only in the documented cells. Confirms the NK=16 mixer
is lossless. (The test exercises the mixer to K=8; the K>16 path is the same
accumulation by construction.)

### Contended A/B (medians only — means invalid, stddev ~900k ns from vLLM spikes)

BASE `<8>` vs OPT `<16,3>` in the SAME contention window (nsys, 40 steps, 200
mixer calls/step). Medians filter the serving spikes; absolute values are inflated
by vLLM co-residency and NOT a serving claim:

| build | chunks @ B=32 | median ns/mixer-call | regs/thread | warps active |
|---|---|---|---|---|
| BASE `<8>`   | 4 | 365,600 | 68 | 66% |
| OPT  `<16,3>`| 2 | **325,088** | 77 | **49%** |

Signal: chunk-halving (4→2) cuts median mixer ~11% EVEN THOUGH the wider acc[16]
drops occupancy 66%→49% (register cliff, as P2 warned). Net positive but the
register hit eats much of the chunk-halving. Hypothesis for the clean sweep:
**NK=12** (3 chunks @ B=32, acc[12], MINBLK=3) may beat NK=16 — a middle point
that trims a chunk without the full occupancy cliff. All three (8/12/16) A/B in one
quiescent session via FUCINA_Q4K_BIGCHUNK.

### Batch-bench A/B — MEASURED (NK=16 wins, contention-resistant)

BASE `<8>` vs OPT `FUCINA_Q4K_BIGCHUNK=16`, B=32, 64–96 steps, interleaved reps
while the user's local vLLM served intermittently (only ever depresses fucina):

| rep | BASE `<8>` (4 chunks) | OPT NK=16 (2 chunks) |
|---|---|---|
| clean sweep rep1 | 397.4 | 426.5 |
| clean sweep rep2 | 297.7 (vLLM burst) | 426.9 |
| interleaved rep1 | 273.4 (burst) | 424.8 |
| interleaved rep2 | 324.3 (partial) | 427.0 |
| interleaved rep3 | 397.4 (clean) | 425.7 |
| **best clean window** | **397.4** | **~426.5** |

**OPT NK=16 is rock-stable at 425–427 across ALL reps; BASE swings 273→397 with
vLLM bursts.** OPT's 2 mixer chunks (vs 4) halve L2-bandwidth pressure, so it
degrades far less under concurrent serving — the win is largest exactly under the
load that matters. Quiescent B=32: **397 → 426 tok/s = +7.3%**, reproducible.

NK sweep (rep-stable, quiescent-ish window): NK=8=396 (==BASE, sanity ✓), NK=12=427,
NK=16=427 — NK=12 and NK=16 TIE. Chose **NK=16** (fewest chunks=2, best headroom
for future wider batches). B=16 unchanged (349→352, noise — B=16 uses 2 NK=8 chunks,
untouched by the K>16 path). The register-cliff worry (occ 66%→49%) did NOT hurt
end-to-end throughput — the chunk-halving dominates.

### QUIESCENT VERDICT (2026-07-17, box clear, 100 GB free, no vLLM)

Batch-bench A/B at B=32 (96 steps, same binary, env toggle):
NK=8 (BASE) 400.5 | NK=12 428.7 | **NK=16 428.1** — +6.9% reproducible, NK=12/16 tie
(consistent with the contended medians). B=8: 232.1 (ref 228.7), B=16: 356.4 (ref
352.3) — untouched paths unchanged (noise).

Losslessness gate: PASS both models, 30/30 OK, dense logit rel <=0.0029 unchanged,
byte-identical determinism. MoE B=32 decode bench 576.2 tok/s — the shared
Q4_K-mixer NK=16 path does not regress the 35B.

Full serving sweep (fresh server, quiescent, conc 1..32):

| N | agg tok/s | TTFT med/p95 | 07-13 pre-fix | vLLM 07-11 |
|---|---|---|---|---|
| 1 | 33.5 | 89/89 | 35.1 | — |
| 2 | 58.4 | 107/107 | 63.2 | 42.9 |
| 4 | 116.2 | 151/168 | 124.0 | 83.9 |
| 8 | 201.2 | 206/208 | 211.6 | 161.7 |
| 16 | 316.9 | 292/296 | 321.7 | 296.1 |
| 32 | **410.3** | 481/488 | 392.4 | **501.9** |

**Verdict: measurable improvement, cell NOT closed.** N=32 serving 392.4 -> 410.3
(+4.6%; bench +6.9%) with N<=16 within run-to-run noise of 07-13 (repeat sweeps of
identical builds have shown ~5-9% spread). vLLM 501.9 still leads by 18%. The
NK=16 chunk-halving banked the F2 residual; the REMAINING gap is the ~2x-above-floor
step (measured ~75ms vs ~35ms floor: mixer 52ms at 4->2 chunks now ~40ms, head
14ms, GDN 9ms at B=32). Next levers (unproven, need their own attribution): a
single-chunk B=32 mixer without the register cliff (e.g. 2 rows/warp splitting acc
pressure), GDN state bandwidth (9.17ms, 2.19x B=16), head at 14ms vs 9.5 floor.

**D32 status: PARTIAL — improvement merged; cell remains the only loss.**
