# Qwen3.5-MoE low-concurrency decode (L-moe-lowc): profile → attribution

Status: IN PROGRESS (2026-07-15). Branch `perf/qwen35-moe-lowc` from main `60b109a`.
Target: the N=2 / N=4 MoE aggregate-throughput cells, historically vLLM's strongest
MoE cells vs fucina (frozen 2026-07-11 baseline: N=2 fucina 58.3 vs vLLM **71.1**;
N=4 102.0 vs **105.0**). Sole owner of this branch/worktree.

## TL;DR of the attribution (measured so far)

The task's hypothesized lever — "the FP8 grouped GEMM at B=2/4 wastes acc-registers /
underfills expert slots (FP8_GMAXB), fixed per-expert overhead dominates" — **does not
hold on this build**, for two independent reasons:

1. **Wrong kernel.** The default FP8-MoE serving path does NOT use the FP8 grouped
   kernel (`fp8_block_gemm_grouped_kernel`, FP8_GMAXB) at all. Loader defaults
   (`cuda/qwen35_backend.cuh:1062,1073`): `q4k_mode = !FUCINA_MOE_FP8` (on) and
   `fp4_mode = q4k_mode && !FUCINA_MOE_Q4K` (on) ⇒ **experts are served as NVFP4**
   (`moe_experts_fp4=1`), decoded by the CUTLASS sm120 grouped block-scaled NVFP4
   GEMM (`dg_fp4_moe_grouped`, `cuda/gemma4_kernels.cu:9613/9630`). The FP8 grouped
   and Q4_K grouped kernels are escape-hatch paths (`FUCINA_MOE_FP8` / `FUCINA_MOE_Q4K`).

2. **That NVFP4 grouped GEMM is already at the bandwidth floor at 1 token/expert**, and
   its cost scales *linearly* with the number of active experts — i.e. each active
   expert's weights are read exactly once and there is no fixed-overhead / MMA-waste
   pathology to remove at low B. Isolated microbench below.

Combined with (a) the P2 head/mixer weight-read-once fixes already covering B≤16, and
(b) the whole decode step being a replayed CUDA graph (launch overhead hidden), the
low-B MoE decode step is **bandwidth-bound at its floor**: expert GEMM (linear in active
experts, ~80–85% peak BW) + mixer + LM head + shared-expert, each read once per step.

## The served decode path (default FP8-MoE checkpoint)

Per-layer (`qwen35_decode_multiseq_body`, `cuda/qwen35_runtime.cuh:390–458`), captured
into a per-B keyed CUDA graph (`qwen35_ms_graph_ensure:519`, replayed every step; only
disabled under SSD expert streaming, not default):

- attn_norm → mixer: FULL-attn layers (q/k/v GEMV, RoPE, KV write, flash-decode
  partial+combine, o-proj) OR GDN layers (in_qkv/in_z/in_a/in_b, conv, split, L2-norm,
  decay, `qwen35_b_gdn_kernel`, gated-norm, out-proj) → residual.
- ffn_norm → **`moe_ffn`** (`cuda/gemma4_kernels.cu:9366`): router (cublasSgemm at
  decode under `moe_unify`), `dg_softmax_topk`, `dg_moe_route_inv` (single-block
  counting sort), `dg_gather_cols`, NVFP4 quant of gathered activations
  (`q35fp4_row_gs` / `q35fp4_quant_grp`), **`dg_fp4_moe_grouped` (gate|up)** →
  `q35fp4_gu_silu_mul` → requant → **`dg_fp4_moe_grouped` (down)** → `q35fp4_dn_scale`
  → `dg_moe_reduce`, then the FP8-block **shared expert** (`fp8_block_gemm_*`).
- residual → next layer. After L layers: output_norm → LM head (B=1: Q8_0 two-pass
  greedy head; B∈[2,16]: single-pass `bf16_head_gemv_batched`, weight read ONCE — P2 F1).

Geometry: H=2048, EFFN=512, E=256, topk=8, ~40 layers. At decode with diverse routing,
each active expert holds ~1 token, so `n_slot = B·8` active-expert groups of ~1 row.

## Measured: NVFP4 grouped expert-GEMM floor (isolated microbench)

`cuda/bench_moe_lowc_fp4.cu` (`make bench-moe-lowc-fp4`) — the exact CUTLASS sm120
grouped block-scaled NVFP4 GEMM `dg_fp4_moe_grouped` uses, at the Qwen MoE projection
shapes, sweeping active-expert count (= B·8) at 1 and 2 tokens/expert. 300 timed iters,
CUDA events, under `flock /tmp/fucina_gpu.lock`, GB10 (vLLM-27B idle-resident on box).

gate|up (K=in=2048, N=out=1024), **1 token/expert**:

| active experts | B | ms/iter | weight MB | eff GB/s |
|---|---|---|---|---|
| 8 | 1 | 0.0389 | 9.4 | 242 |
| 16 | 2 | 0.0574 | 18.9 | 329* |
| 32 | 4 | 0.1609 | 37.8 | 235 |
| 64 | 8 | 0.3594 | 75.5 | 210 |
| 128 | 16 | 0.7051 | 151.0 | 214 |
| 256 | 32 | 1.3639 | 302.0 | 221 |

down (K=in=512, N=out=2048), **1 token/expert**:

| active experts | B | ms/iter | weight MB | eff GB/s |
|---|---|---|---|---|
| 8 | 1 | 0.0272 | 4.7 | 174 |
| 16 | 2 | 0.0472 | 9.4 | 200 |
| 32 | 4 | 0.0812 | 18.9 | 232 |
| 64 | 8 | 0.1641 | 37.8 | 230 |
| 128 | 16 | 0.3423 | 75.5 | 221 |
| 256 | 32 | 0.6607 | 151.0 | 229 |

(* the B=2 gu row reads >peak because at 18.9 MB the weight set is partly L2-resident;
the settled large-N figure ~210–235 GB/s is the true DRAM floor. LPDDR5X peak ≈ 273 GB/s.)

**Reading:** time scales ~linearly with active experts (gu 8→256 groups: 0.039→1.364 ms
= 35× for 32× the experts; dn 24×), at **~80–85% of peak DRAM bandwidth** even at 1
token/expert. Each active expert's weights are read once; there is NO fixed per-group /
launch / MMA-tile-underfill overhead to reclaim at low B. 2-tokens/expert (collision
case, B≥8) shows the same floor at ~2× TFLOP/s (weights amortised over 2 rows).

**Per-step expert-GEMM budget at B=1:** gu 0.039 + dn 0.027 ≈ 0.066 ms/layer × ~40
layers ≈ **2.6 ms**, i.e. only ~12% of the ~21.7 ms single-stream step. **The low-B MoE
decode step is dominated by the NON-expert path** (mixer GDN/attn, LM head, shared
expert) + the small per-layer MoE glue kernels — all read-once / graph-replayed.

## What the profile rules OUT as low-B levers

- **Expert grouped GEMM tuning** (FP8_GMAXB, acc-registers, slot packing): the default
  path is CUTLASS NVFP4 at ~80–85% BW, linear in active experts — floored (confirms the
  mission plan's non-goal, now extended from prefill to the low-B decode regime).
- **Head / mixer weight-re-read** (P2's dense lever): the multi-chunk re-read only
  appears at B≥17 (ceil(B/16)≥2 / NK ladder). At B=2/4 the head is a single
  `bf16_head_gemv_batched` pass and the mixer is a single chunk — already read once.
- **Per-step launch / host overhead**: the whole decode step is a replayed CUDA graph
  (`qwen35_ms_graph_ensure`); launch gaps are hidden (same result P2 measured for dense:
  GPU busy ≈ wall). Router uses cublasSgemm on-device; `dg_moe_route_inv` is one small
  block; no per-step host sync on the non-SSD fp4 path.

## Open (GPU-contended — an 85 GiB vLLM-27B service holds the box; ~20 GiB free)

Still to measure on a quiescent box (harnesses built and staged):
1. **Serving sweep on THIS build** (`scripts/bench_serving.py`, diverse, N=1..32) — the
   gate question: is N=2/4 aggregate already a win vs vLLM 71.1/105.0? Memory
   `moe-vllm-rebaseline-jul8` (current source of truth, post-F1F2F3) says fucina already
   wins agg N=1..16 — the frozen "vLLM wins N=2/4" premise predates F1/F2/F3. Confirm.
2. **Full-step nsys `--cuda-graph-trace=node`** at B=2/4 (`/tmp/fucina_moe_decode_bench`)
   for the empirical mixer / head / shared-expert / expert-GEMM split.

Harnesses: `make bench-moe-lowc-fp4` (done, above), `make qwen35-moe-decode-bench`
(built `/tmp/fucina_moe_decode_bench`).
