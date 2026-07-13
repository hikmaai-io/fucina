# Qwen3.5 MoE N=32 TTFT — the last vLLM-win cell (L-moe-ttft)

Status: ACTIVE (2026-07-13). Branch `perf/qwen35-moe-ttft` from main. Sole owner.
Closes the ONLY remaining cell vLLM beats fucina on: Qwen3.5-35B-A3B-FP8, N=32
time-to-first-token. Everything else (single-stream, N≤16 all, N=32 aggregate)
fucina already wins; do NOT regress any winning cell.

## The gap (measured on THIS build, 2026-07-13)

Reproduced per `benchmark-evidence/PROTOCOL.md`: fucina server (`--parallel 32
--max-concurrent 64 --gpu-mem-util 0.90 --ctx 25280`), `bench_serving.py`
synchronized-burst harness, diverse prompts, `--conc 1,2,4,8,16,32`. Under
`flock /tmp/fucina_gpu.lock`. Raw: `benchmark-evidence/results/2026-07-13-moe-ttft/`.

| N | median TTFT ms | p95 TTFT ms | agg tok/s | vLLM TTFT (2026-07-11) |
|---|---|---|---|---|
| 1 | 58.2 | 58.2 | 59.9 | — |
| 2 | 100.7 | 110.4 | 61.1 | — |
| 4 | 162.5 | 163.3 | 132.1 | 417 |
| 8 | 276.0 | 276.5 | 226.1 | 669 |
| 16 | 480.5 | 483.0 | 307.3 | 549 |
| 32 | **870.9** | **874.5** | **448.5** | **664** |

fucina wins TTFT at N≤16 outright and wins N=32 aggregate (448.5 vs vLLM 302.8).
The single hold-out is **N=32 TTFT: 870.9 median / 874.5 p95 vs vLLM 664**. (This
build is ~post-S2; the 866 in the P1 doc is confirmed essentially unchanged.)

## Attribution — profile-driven, no speculative tuning

### Isolating the admission-prefill GPU cost (microbench)

`cuda/bench_qwen35_moe_ttft.cu` times `gemma4_engine_seq_add_multiseq` alone (the
N=32 TTFT critical path) via CUDA events — excluding HTTP/scheduler/decode wall-
time. Diverse ~15-tok prompts, 12 reps, median. Under the GPU flock.

| M | median ms | ms/seq |
|---|---|---|
| 1 | 53.75 | 53.7 |
| 2 | 76.58 | 38.3 |
| 4 | 121.37 | 30.3 |
| 8 | 198.22 | 24.8 |
| 16 | 346.71 | 21.7 |
| 32 | **653.79** | 20.4 |

**Linear fit: `ms(M) = 40.0 + 19.21·M`** (R²≈1.000). The M=32 batched-prefill GPU
cost is **653.8 ms** — that IS essentially the whole 870 ms TTFT (the ~217 ms
residual is HTTP + scheduler + first decode-step interleave wall-time).

**The batched prefill is NOT flat in M.** A perfectly weight-amortized batched
forward should cost ≈`fixed + tiny·M`. Instead there is a **19.2 ms/seq serial
slope** → 596 ms of serial-per-sequence work at M=32 (91% of the 654 ms). vLLM
hits 664 ms because its single batched prefill (`max_num_batched_tokens≥480`) has
NO per-sequence serial tail — one forward over all rows.

### Per-phase kernel attribution (nsys `--cuda-graph-trace=node`, M=32)

`/tmp/moe_ttft_m32.nsys-rep`, `cuda_gpu_kern_sum`. Per-rep (proportions robust;
nsys inflates absolutes ~1.4× vs the CUDA-event wall). % of GPU kernel time:

| kernel | % | launches/rep | what / why serial |
|---|---|---|---|
| `qwen35_b_gdn_chunk_kernel` | **23.9** | 960 = 32 seq × 30 SSM layers | **per-sequence serial loop** (`for i in M`), each `<<<NVH=32,512>>>` = 32 blocks on 48 SMs → underfilled AND serialized across seqs |
| `bf16_head_gemv_kernel` | **18.0** | 32 (serial) | **per-row serial LM-head loop**, each re-reads the ~1.0 GB BF16 head (248320×2048×2B). 32× = 32.5 GB / 273 GB/s ≈ 119 ms of redundant weight traffic |
| `dequant_fp8_expert_slab_bf16` | 10.1 | 20 | MoE expert path — **already batched** over T=480 rows (amortized, floored per plan) |
| `nvfp4_amax_bf16` | 8.5 | 20 | MoE expert quant — batched |
| cutlass grouped GEMM | 7.4 | 80 | MoE grouped-expert GEMM — batched, ~81% BW (floored per plan) |
| `nvfp4_quant_bf16` | 7.4 | 5120 | MoE expert quant — batched |
| `fp8_block_gemm_dual` | 7.3 | 1200 | projection GEMMs — batched over all T rows |

The two **serial-per-sequence tails** (`gdn_chunk` 23.9% + `bf16_head` 18.0%) =
**~42% of prefill GPU time** and account for essentially the entire 19.2 ms/seq
slope. The MoE grouped-expert GEMM (the plan's floored 81%-BW kernel) is already
batched over all 480 rows and is NOT the bottleneck — confirming the plan's
"MoE grouped-GEMM tuning is a non-goal" verdict. **The TTFT gap is serialization
in two per-sequence loops in `qwen35_prefill_multiseq_body`, not kernel physics.**

### What the profile rules OUT
- MoE expert grouped-GEMM efficiency (batched already, ~81% BW floor — non-goal).
- Admission SCHEDULING / KV allocation host overhead (the 654 ms is pure GPU
  kernel time inside `seq_add_multiseq`, measured by CUDA events).
- Router counting-sort / expert residency/staging (dequant+quant are batched over
  all rows; small % and flat in M).
- First-token sampling (greedy argmax, single `argmax_rows_kernel<<<M>>>`, µs).

## Fixes (implement ONLY what the profile proves)

### F1 — batched LM head (weight-read-ONCE), byte-identical
Replace the `for i in M: bf16_head_gemv_launch(...)` serial loop with the SAME
`nvfp4_xT_launch` + `bf16_head_gemv_batched_launch` recipe P2 already shipped for
dense decode (weight read ONCE for ≤32 rows, ascending-k order preserved ⇒ logits
bitwise-identical to the per-row loop). Expected: 118 ms → ~10–16 ms at H=2048.

### F2 — batched GDN chunk scan across sequences
Fuse the `for i in M: qwen35_b_gdn_chunk_kernel<<<NVH,512>>>` serial loop into ONE
`<<<NVH·M,512>>>` launch; `blockIdx.x` decodes `(seq, vh)`, each block reads its
own slot state `S[l][slots[seq]]` and its own row-offset q/k/v. Per-sequence GDN
recurrence order is UNCHANGED (each block still scans one sequence's contiguous
64-token chunks carrying that slot's state) ⇒ byte-identical; only the launch is
coalesced. Fills 32·M blocks across 48 SMs (was 32) — removes the serial tail AND
the underfill. Expected: 156 ms → occupancy-bound floor.

Gates (each change): determinism byte-identical; `qwen35-multiseq-prefill-test`
PASS with UNCHANGED logit bounds (MoE ≤0.0946, dense ≤0.0029); `protection_gate.py`
green BOTH models, no regression on any winning cell; full build + Go tests + all
Qwen3.5/Gemma/diffusion gates. Re-measure N=1..32 MoE TTFT + aggregate after each.

## Results log

### F1 — batched LM head (LANDED)

Microbench (`gemma4_engine_seq_add_multiseq`, CUDA events, 12 reps median):

| M | pre-F1 ms | F1 ms | Δ |
|---|---|---|---|
| 1 | 53.75 | 53.76 | 0 |
| 8 | 198.22 | 161.65 | −37 |
| 16 | 346.71 | 269.64 | −77 |
| 32 | **653.79** | **503.75** | **−150** |

Linear fit `ms = 43.4 + 14.37·M` (was `40.0 + 19.21·M`) — the 18% head serial tail
is gone (−4.84 ms/seq slope; −150 ms at M=32, matching the predicted 32×→1× head
read). Gate: `qwen35-multiseq-prefill-test` PASS with UNCHANGED bounds (MoE
≤0.0946, dense ≤0.0029) — bitwise-identical by construction.

### F2 — batched GDN chunk scan across sequences (pending)
