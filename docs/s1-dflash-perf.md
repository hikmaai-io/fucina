# S1a-PERF: DFlash net-speedup profiling + attribution

ABOUTME: Profile-driven perf work to make greedy DFlash a NET speedup at B=1 on GB10,
ABOUTME: not just lossless. Measured numbers only; losslessness gates stay green + unweakened.

## Baseline (MEASURED, real FP8 target + real z-lab draft, B=1, K=16 / T=17)

`/tmp/rstep` on `/opt/spark/models/models--Qwen--Qwen3.5-9B-FP8` + z-lab draft:
- target plain greedy decode: **29.2 ms/tok**
- DFlash greedy step: **435 ms/step**, emitting **9.2 tok/step** = **47.3 ms/emitted-token**
- => DFlash is **~1.6x SLOWER per token**. Break-even accept would need the step under
  9.2 * 29.2 = 269 ms; it is 435 ms.

## Per-kernel attribution (nsys, 10 DFlash steps ONLY, cudaProfilerApi-scoped)

Per step (total / 10):

| kernel | ms/step | %step | what |
|---|---|---|---|
| q35df_head_argmax_batched | 38.5 | 23.9 | DRAFT greedy LM head (my warp kernel) |
| q35df_matmul | 32.2 | 20.0 | DRAFT dense projections (my warp kernel) |
| nvjet_sm121 mma 64x144x64 | 32.2 | 20.0 | TARGET verify GEMMs (tensor core) |
| cutlass bf16 32x32_128x1 align8 | 25.9 | 16.0 | TARGET verify GEMMs (tensor core) |
| cutlass bf16 32x32_128x1 align2 | 10.3 | 6.4 | TARGET verify GEMMs |
| bf16_head_gemv_batched<17,6> | 10.1 | 6.3 | TARGET LM head over 17 rows |
| qwen35_b_gdn_chunk_kernel | 5.5 | 3.4 | TARGET GDN recurrence |
| (rmsnorm/rope/attn/silu/misc) | ~8 | ~5 | draft + target elementwise |

Draft-forward subtotal (head+matmul+draft rmsnorm/rope/attn) ≈ **~90 ms/step** — the
DOMINANT cost, MORE than the whole target verify (~78 ms/step incl. LM head).

## Root cause (NOT the target verify amortization)

The target (1+K) verify forward already rides tensor-core GEMMs (nvjet/cutlass) and
the weight-read-once batched LM head (`bf16_head_gemv_batched<17,6>`, 10 ms for 17
rows) — it is NOT re-reading weights per row. The amortization is fine on the target
side. The REAL excess is the DRAFT forward's own kernels:

1. **Draft LM head** `q35df_head_argmax_batched` = **38.5 ms** for 16 rows, vs the
   target's `bf16_head_gemv_batched` = **10.1 ms** for 17 rows on the SAME [vocab,H]
   — the draft head is **~3.8x slower** because it is a hand-rolled warp-per-token
   kernel, not the tuned transpose + weight-read-once batched head.
2. **Draft projections** `q35df_matmul` = **32.2 ms**, a hand-rolled warp GEMM, vs
   the target's cutlass/nvjet tensor-core GEMMs for the same class of matmul.

Both are safe to optimize: the DRAFT is NOT required to be bit-identical. Greedy
losslessness comes from the TARGET verify + decode-body commit (a wrong/most-drafts
only changes accepted length j, never the emitted tokens). So the draft head+matmul
can be swapped to the engine's tuned tensor-core kernels without touching the
losslessness contract.

## Levers (in priority order, each gated by the existing losslessness + determinism gates)

- L1: draft LM head -> reuse the engine's `bf16_head_gemv_batched_launch` + argmax
  (target-parity kernel). Expected ~38.5 -> ~10 ms/step (-28 ms).
- L2: draft dense projections -> route through the tensor-core `gemm_bf16` path
  instead of `q35df_matmul`. Expected ~32 -> ~? ms/step.
- Re-measure end-to-end after each; report real tok/s + break-even accept length.

## Landed / measured

- **L1 (landed, commit e34c74d)**: draft LM head -> engine tuned batched head.
  nsys: draft head 38.5 -> 9.2 ms/step. End-to-end 435 -> 408 ms/step (47.3 ->
  44.3 ms/emitted-token). Lossless + determinism gates green, same 3.556 accept.
- **L2 attempt #1 (REVERTED)**: a register-blocked weight-read-once batched draft
  matmul (`acc[K]` per warp) REGRESSED to 488 ms/step. K=17 register array +
  wide out_dim (I=12288) => heavy register pressure / low occupancy, worse than the
  memory-bound `q35df_matmul`. Weight-read-once alone is not enough; the draft
  projections need REAL tensor cores (cutlass/nvjet), like the target verify GEMMs.
  Note: a stale-build hazard was found + fixed here (the Makefile does not list
  qwen35_dflash_forward.cuh as a dep of gemma4_kernels.o, so header-only edits
  require `rm -f cuda/gemma4_kernels.o` before `make lib` to actually rebuild the
  engine path -- earlier "no-change" measurements were stale).
- **L2 attempt #2 (next)**: route the draft's big projections (gate/up/down, q/k/v/o)
  through the engine's tensor-core `gemm_bf16` (cutlass) with a bf16 activation
  transpose, exactly as the target verify does -- needs the draft head done in the
  engine like L1, or a draft-forward variant that takes an engine cuBLAS/gemm hook.

## DECISIVE CEILING (MEASURED /tmp/breakdown, B=1)

DFlash step (408 ms) decomposes as:
- draft forward ~76 ms
- verify_block(T=17) **89.3 ms**
- commit replay of the accepted prefix **292.8 ms** (== ~10 x single_decode 29.3 ms)

The commit replays the j accepted tokens as **j SEQUENTIAL single-token target
decodes** (q35_gdn_commit -> qwen35_step_batch per token), because exact-lossless
accept requires the decode-body argmax (the batched verify argmax diverges from
single-token decode at ~10/40 interior positions -- see f348ec7; the fast-commit
primitive q35_gdn_commit_fast gives byte-identical STATE in ~1 pass but its argmax
is chunk-body, not decode-identical).

**Ceiling math**: at accept j=9.2, the commit alone is ~10 x 29.3 = 293 ms => the
DFlash step emits 9.2 tokens for >=293 ms = **>= 31.8 ms/emitted-token EVEN IF
draft and verify were free** (both are not). Plain decode is 29.3 ms/token. So
**greedy DFlash at B=1 is STRUCTURALLY >= plain decode with this commit**: the
commit re-runs the target once per accepted token (exactly plain decode's work)
PLUS draft+verify overhead. This is not a kernel-tuning gap; it is the exact-
lossless commit design on a per-token-decode engine.

The ONLY B=1 win requires the commit to advance state over the accepted prefix
WITHOUT j sequential target decodes -- i.e. a BATCHED (T-row) target forward that
is BIT-IDENTICAL to single-token decode so its argmax can be trusted for accept +
emit. That is a deep engine change (make batched GEMM/attention numerics equal to
the gemv single-token path) OUT OF S1a-PERF scope and risking the proven
losslessness. Pivot: the smallest B>1 concurrency-batched path, where the verify +
commit amortize the target weight read across B requests so the per-request commit
cost is shared -- the regime where spec decode wins on this engine.
