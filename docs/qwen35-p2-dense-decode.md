# P2 — Qwen3.5-9B-FP8 dense decode at N≥16 (profile → attribution → fixes)

Status: IN PROGRESS (2026-07-12). Branch `perf/qwen35-dense-decode`.
Baseline scoreboard (2026-07-11 head-to-head): dense agg decode fucina 236.1 @ N=16,
260.5 @ N=32 vs vLLM 296.1 / 501.9. P0 telemetry: 88–90 ms/step at avgB≈30
(~102 GB/s effective on a ~9 GB read-set = 37% of LPDDR5X peak).

## 1. Profile (B=30/32 dense decode step, GB10, CUDA 13, sm_121a)

Harness: `cuda/test_qwen35_batch_bench.cu` (served `gemma4_engine_step_batch` path,
CUDA-graph replay), Qwen3.5-9B-FP8 checkpoint. nsys `--cuda-graph-trace=node`;
ncu `--set basic` (needed sudo: `RmProfilingAdminOnly=1`).

Reproduced step time: **84.7 ms/step wall at B=32** (91.6 at B=30 with per-B chunk
inefficiency, see below). Bench aggregate: 228 @ B=8, 345 @ B=16, 380 @ B=32 tok/s
(decode-only, no admission — the serving numbers sit below these).

**Launch gaps are a non-issue: GPU busy 84.29 ms of 84.74 ms wall (gap 0.45 ms/step).**
The CUDA graph is captured and replayed (`qwen35 M4 batch graph captured (B=30/32)`).
Category (c) of the attribution matrix is ruled out.

Per-kernel table, B=30 step (nsys, 30 steps; ms/step):

| kernel | ms/step | share | bytes/step | eff. BW |
|---|---|---|---|---|
| `mmvq_q4_k_packedT_batched<16>` (200 proj) | 30.5 | 33% | 3.89 GB | 128 GB/s |
| `mmvq_q4_k_packedT_batched<8>` (200 proj) | 14.3 | 16% | 3.89 GB (L2-assisted) | 272 GB/s |
| `mmvq_q4_k_packedT_batched<6>` (200 proj) | 14.4 | 16% | 3.89 GB (L2-assisted) | 270 GB/s |
| `bf16_head_gemv_batched<14,6>` | 9.6 | 10% | 2.03 GB | 212 GB/s |
| `bf16_head_gemv_batched<16,6>` | 8.9 | 10% | 2.03 GB | 227 GB/s |
| `qwen35_b_gdn_kernel` (24 L) | 8.6 | 9% | ~1.5 GB state | 176 GB/s |
| `m2_gemm` (in_a/in_b f32) + small ops | ~4.9 | 5% | — | — |
| **total** | **91.2** | 100% | | |

(The dense mixer is served as **Q4_K repacked at load** — "qwen35-Q4K mixer repacked
in place (200 tensors → packed dp4a GEMV)" — 6.91 B params ⇒ 3.89 GB; the LM head is
BF16 248320×4096 = 2.03 GB.)

Theoretical floor: mixer 3.89 + head 2.03 + GDN state 1.51 + KV/activations ~0.3
≈ **7.7 GB ⇒ 28 ms @ 273 GB/s** (35 ms at a realistic 220). Measured 85–92 ms.

## 2. Attribution

- **(a) weights read more than once per step — CONFIRMED, dominant.**
  - Mixer: `gemv_batched_w` Q4_K dispatch chunks K>8 as NK=16, then 8s, then
    remainder. B=30 ⇒ THREE passes (16+8+6) over the full 3.89 GB mixer weight set
    per step = 59.2 ms. B=32 ⇒ one `NK=32` pass, but that kernel is
    register-crippled (see (b)) and takes 52 ms alone — barely better.
  - LM head: `qwen35_decode_multiseq_body` reads the 2.03 GB BF16 head
    `ceil(B/16)` = 2× per step at B>16 (18.5 ms; the code comment even names it:
    "the 1 GB head is read ceil(B/16)× per step").
- **(b) register-pressure / batch-shape inefficiency — CONFIRMED (ncu).**
  `mmvq_q4_k_packedT_batched<16>`: 144 regs/thread, achieved occupancy **16.6%**,
  memory throughput 21% ⇒ 128 GB/s (the <8>/<6> passes ride partial L2 reuse of
  pass 1 at ~270). `<32>` at B=32 is worse (75 GB/s effective).
- **(c) launch gaps/syncs — RULED OUT** (0.45 ms/step; graph replay works).
- **(d) non-weight traffic — minor.** GDN state r/w is 1.5 GB at B=30
  (unavoidable, 176 GB/s — improvable but only ~3 ms of upside), KV negligible at
  short contexts, activations ~MBs.

Gap accounting at B=30: mixer excess ≈ 45 ms (59.2 vs ~14.3 for one full-BW pass),
head excess ≈ 9 ms (18.5 vs ~9.3 one pass), GDN + small ops ≈ 13 ms. That is the
whole 88 ms − 33 ms story. **The dense gap is kernel weight-re-read + register
pressure, exactly where P0 pointed (37% of peak = headroom, not floor).**

## 3. Fixes (cheapest first, each gated)

### F1 — LM head: one weight pass for B ≤ 32 (bf16_head_gemv_batched K≤32)
`bf16_head_gemv_batched_kernel<K,R>` exists for K≤16 (R=6 for K 9–16). Extend the
dispatch to K≤32 with R=3 (same ~96 acc registers/warp; identical per-row k-order
⇒ bitwise-identical logits), and widen the head loop in
`qwen35_decode_multiseq_body` from 16-row to 32-row chunks.
Expected: 18.5 → ~9.5 ms/step at B=30 (head read once).
**Measured: see §4.**

### F2 — mixer Q4_K batched GEMV: weight-read-once for 8 < K ≤ 32
New kernel variant: block = 2 output rows × ceil(K/8) chunk-warps; each warp runs
the EXACT NK≤8 per-row loop for its (row, 8-activation-chunk) pair. Warps of the
same row issue identical weight addresses ⇒ L1/L2 serve the duplicates and DRAM
reads the 3.89 GB once per step (vs 3×). Per-(row,chunk) accumulation order is
identical to the current chunked launches ⇒ bitwise-identical outputs.
Expected: 59 → ~16–20 ms/step at B=30.
**Measured: see §4.**

### Non-goals (this pass)
- GDN scan tuning (8.6 ms, ~3 ms upside — revisit only if F1+F2 land clean).
- Any MoE decode-kernel change (debunked levers; different model).

## 4. Measured deltas (updated as fixes land)

| change | B=16 agg | B=30 step | B=32 agg | gates |
|---|---|---|---|---|
| baseline (028c8b2) | 345 tok/s | 91.6 ms | 380 tok/s | green |
