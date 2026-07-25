<!-- ABOUTME: Code-verified Qwen3.6-35B-A3B B=1 decode byte roofline and GB10 target analysis. -->
<!-- ABOUTME: Ranks exact-greedy speed levers and defines the proof gate for each one. -->
# Qwen3.6-35B-A3B-FP8 single-stream decode on GB10

Status: **analysis complete; no GPU was touched**. The rows-per-warp Q8-head experiment is staged,
default-off, and still lacks a GB10 A/B. The production baseline is **54–59 tok/s**, or **17–18
ms/step**. Reaching 80 tok/s requires 12.5 ms/step (a 4.5–5.5 ms saving); 100 tok/s requires 10.0
ms/step (a 7–8 ms saving).

## Executive answer

- The short-context, compulsory-byte floor is **2.155 GB/token**, or **7.894 ms at the GB10's
  advertised 273 GB/s**: a purely mathematical **126.7 tok/s** roof. At 3,500 prior tokens it is
  2.227 GB, 8.156 ms, and 122.6 tok/s if the eight GQA query heads obtain perfect cache reuse of
  their two physical KV heads.
- Sustaining a more credible 80% of peak lowers those ceilings to **101.3 tok/s at short context**
  and **98.1 tok/s at 3.5k**. This is a memory-only ceiling, not a prediction: Q4_K unpack/dp4a,
  NVFP4 scale handling, softmax, recurrence, synchronization, and graph-node issue all consume
  time not represented by `bytes / 273 GB/s`.
- **80 tok/s is physically allowed without reducing bytes/token.** The counted traffic requires
  172.4 GB/s effective at short context and 178.1 GB/s at 3.5k; unlike the current 17–18 ms path,
  this leaves some peak-bandwidth room for imperfect kernels. It is nevertheless a stretch
  engineering target.
- **100 tok/s is not forbidden by the ideal short-context byte count, but is not an honest
  byte-preserving target.** It needs 215.5–222.7 GB/s at short/3.5k *for all counted traffic*, with
  almost no allowance for cache misses or non-memory work. At 3.5k the kernel-requested KV traffic
  (no inter-query-head L2 reuse) brings the ledger to 2.729 GB and exactly consumes the 2.730
  GB/token peak-bandwidth budget. At 4k it already implies only 96.6 tok/s at peak. Treat 100 as a
  physics-edge diagnostic, not an attainable commitment without fewer bytes/token or accepted-token
  amortization.
- The honest outcome of the listed local, byte-preserving kernel work is **about 65–78 tok/s** if
  the Q8 head rewrite succeeds; 80 is the upper edge. A certified smaller proxy head can plausibly
  move that to **roughly 72–85 tok/s** while preserving exact greedy output, but it does reduce
  bytes/token. No non-speculative plan here credibly sustains 100 tok/s.

## What was verified from code

The `qwen35` implementation namespace serves the Qwen3.6 checkpoint variants. The checked-in
runtime plans for official FP8, ModelOpt NVFP4, and both Unsloth Qwen3.6 variants have the same
finalized serving totals. The host-only calculator consumes the accurate-Qwen3.6 plan rather than
inferring the resident representation from checkpoint marketing names:

```sh
python3 scripts/qwen36_decode_roofline.py
```

It asserts the finalized tensor plan and the following production geometry:

- 40 layers: **30 LINEAR/GDN + 10 FULL** (`full_attention_interval=4`), not the stale 24/8 geometry
  in some older comments;
- `H=2048`, `VOC=248320`, `NQ=16`, `NKV=2`, head dim 256;
- GDN `NVH=32`, state dim 128, inner 4096, convolution width 8192 and kernel 4;
- 256 experts, top-8, routed/shared intermediate 512.

Relevant dispatch is `qwen35_decode_layer_range_body` in `cuda/qwen35_runtime.cuh`. Mixer weights
are load-time FP8-to-Q4_K projections, routed experts are grouped NVFP4, the shared expert remains
FP8-block, GDN state is BF16, FULL KV is FP16, and B=1 greedy uses the Q8_0 proxy plus BF16 rows in
`q8_head_*` (`cuda/gemma4_kernels.cu`). `moe_ffn` contains both NVFP4 activation-quant passes and
both grouped expert GEMMs.

The startup log's **0.88 GiB core arena is not all Q4_K mixer bytes**. It is 719.585 MB of Q4_K
matrices plus router, shared expert, norms, convolution, and GDN metadata. The distinction matters
for any proposed Q4_K reduction.

## Complete per-token byte ledger

All units below are decimal. Immutable tensor bytes are exact plan totals. Dynamic state/scratch
bytes are formulas matching the kernel data types. “Charged” means the roofline pessimistically
charges the logical global transfer to LPDDR even when a small producer/consumer buffer can remain
in cache. Conversely, the KV “unique” row assumes ideal GQA cache reuse; the context table also
shows the no-reuse request volume.

| Source | Exact bytes/token | Derivation / code path |
|---|---:|---|
| **Q4_K mixer matrices** | **719,585,280** | 130 plan entries, 144 B/256 weights: 30×(in_qkv + in_z + out) and 10×(q + k + v + o). Each B=1 GEMV reads its matrix once. |
| Router weights | 83,886,080 | 40×256×2048×F32; `cublasSgemm` in `moe_ffn`. |
| Shared-expert gate vector | 327,680 | 40×2048×F32. |
| **Shared expert** | **125,844,480** | 40×3×512×2048 FP8 bytes plus 15,360 BF16 block-scale bytes. |
| Norm/GDN/conv static metadata | 20,367,872 | Everything else in the exact 949,996,032-byte core plan: norms, GDN a/b matrices, conv weights, A/dt vectors. |
| **NVFP4 routed experts** | **566,231,040** | Per layer, top-8 reads 8.389 MB gate\|up packed + 1.049 MB scales + 4.194 MB down packed + 0.524 MB scales = 14.156 MB; ×40. Top-k indices are unique. |
| **GDN recurrent state** | **62,914,560** | 30×load+store of `[32,128,128]` BF16. The kernel's F32 shared copy is on-chip. |
| **Conv recurrent ring** | **5,898,240** | 30×load+store of `8192×(4−1)` F32. |
| **FULL-attn KV, unique** | **20,480×(p+2)** | At prior context `p`: 10 layers×K/V×2 KV heads×256×FP16, one current write plus `p+1` positions read. This is 0.041 MB at p=0, 71.721 MB at p=3500. |
| FULL flash-decode partials | 16,512,000 | At configured maxctx 25,280, `S=50`: write then read `m`, `l`, and 256-F32 output for 16 heads×10 layers×50 splits. |
| **Q8_0 head scan** | **540,344,320** | `248320×(2048/32)×34`. This is the mandatory proxy-weight read, not the 1.017 GB BF16 resident head. |
| Head logits/candidate scan scratch | 2,979,840 | One F32 proxy-logit write and two full reads by `q8_head_candidates_kernel`. Candidate ids/count are below 1 KiB at B=1. |
| **BF16 exact rescore rows** | **0…262,144** | `C×2048×2`, `C≤64`. The table charges the maximum. |
| **Q4 activation-quant scratch** | **2,263,040** | For all 130 mixer GEMVs: one F32 input read, Q8+scale+sum write, and one compulsory scratch read. Repeated row reads hit cache and are not LPDDR-compulsory. |
| **NVFP4 expert activation-quant scratch** | **7,475,200** | Across 40 layers, top-8 rows, both H=2048 gate\|up and K=512 down passes: row-amax read, quant read, E2M1 write/read, and one E4M3 scale/16 values write/read. |
| Embedding row + expert global scales | 4,416 | One 2048-element BF16 embedding row plus 80 F32 scales. |
| **Fixed subtotal before KV** | **2,154,896,192** | Includes the 16.512 MB FULL-attention partial workspace and max 64-row rescore. |

Small FP32 residuals, routing indices, RoPE values, normalized activations, and expert intermediates
are global-memory instructions but have no compulsory LPDDR read when their immediate consumer
finds them in L2. Charging all such requests as DRAM would be an implementation-traffic upper
bound, not a physical memory floor. The table explicitly charges the requested quant and head
scratch named in the question. Its 2.155 GB subtotal is therefore a reproducible lower roofline,
not a claim that Nsight will report exactly 2.155 GB of DRAM traffic.

Two important request-vs-DRAM qualifications:

1. The Q8 head requests about 2.034 GB of activation loads (`VOC×H×F32`) in addition to its Q8
   weights, but all rows reuse the same 8 KiB activation. Those loads should be L1/L2, not LPDDR.
   Their scalar load/convert/FMA issue cost is exactly why the measured 540 MB scan takes 5.04 ms
   (107 GB/s by weight bytes) instead of its 1.98 ms raw-memory floor.
2. Each of 16 GQA query-head blocks independently requests its mapped K/V. The unique KV formula
   assumes the eight blocks mapped to one physical KV head reuse cache lines. With no such reuse,
   multiply the KV *read* by `NQ/NKV=8`; do not multiply the current write.

## Roofline at 273 GB/s

`p` is prior context length. “80% ceiling” is still optimistic: it applies one sustained bandwidth
to unlike kernels and gives zero debit for arithmetic. “GQA-request” is the opposite cache bound,
where every query head's KV requests reach LPDDR.

| p | unique bytes | peak floor | peak ceiling | 80% ceiling | GQA-request bytes | GQA-request peak ceiling |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.155 GB | 7.894 ms | 126.7 tok/s | 101.3 | 2.155 GB | 126.7 |
| 1,000 | 2.175 GB | 7.969 ms | 125.5 | 100.4 | 2.319 GB | 117.7 |
| 3,500 | 2.227 GB | 8.156 ms | 122.6 | 98.1 | 2.729 GB | 100.1 |
| 4,096 | 2.239 GB | 8.201 ms | 121.9 | 97.6 | 2.826 GB | 96.6 |
| 8,192 | 2.323 GB | 8.508 ms | 117.5 | 94.0 | 3.497 GB | 78.1 |
| 25,280 | 2.673 GB | 9.790 ms | 102.1 | 81.7 | 6.297 GB | 43.4 |

The target budgets are 3.4125 GB/token at 80 tok/s and 2.730 GB/token at 100 tok/s, if every byte
moves at peak. This is why 80 has physical room and 100 has almost none. Context and actual GQA
cache behavior must accompany any target claim.

## Reconciliation with the 17–18 ms baseline

Historical measurements were collected in separate runs and must not be summed as a simultaneous
profile:

- Q8 head: **5.04 ms** isolated versus its 1.98 ms byte floor; BF16 was 5.96 ms. This is the
  largest instruction/ILP gap.
- Routed grouped NVFP4 GEMMs: **0.039 + 0.027 ms/layer**, about **2.6 ms/step**, versus 2.074 ms
  for their 566 MB at peak. They already deliver roughly 80–85% of GB10 bandwidth.
- The remaining immutable core is 950 MB, not “0.88 GiB of Q4_K”, and the state/attention/glue
  above adds context-dependent work. It occupies the remaining roughly 9–10 ms in serving.
- The whole greedy path is one replayed CUDA graph and host delivery is about 0.001–0.002 ms.
  Graph replay removes host launch gaps; it does **not** make the graph's device nodes free.

A useful hard bound for the *current* major kernels is: keep the measured 5.04 ms Q8 scan and 2.6
ms experts, run every other short-context byte at impossible peak bandwidth, and the result is
still about **11.4–11.6 ms (86–88 tok/s)** before Q4 unpack, recurrence, softmax, or node issue.
Thus rows-per-warp alone cannot produce 80–100, and 100 requires changing the head kernel and then
nearly perfecting everything else.

## Ranked concrete levers

Ranges are engineering estimates anchored to measured component times and byte floors, not GPU
results. They overlap and are **not additive**. “Lossless” below means exact target greedy output,
not merely a good validation score.

| Rank | Lever | Expected step saving | Confidence | Risk | Why / exact lossless gate |
|---:|---|---:|---|---|---|
| 1 | **Vectorized `uint4` loads + int8 `dp4a` Q8-head scan** | **1.5–2.5 ms** | Medium | Medium–high | The current scan is 5.04 ms for 540 MB because it performs scalar int8→F32 loads and 32 FMA per block. Vector loads fix transactions; dp4a needs a Q8 activation and changes proxy logits. It is lossless only with a certified candidate algorithm: precompute per-row BF16-vs-proxy error norms, use `U_i=z_i+||e_i||₂||x||₂`, exactly rescore every row whose upper bound can beat the exact incumbent, and fall back to a full BF16 scan on certificate failure/candidate overflow. Bit-equal proxy logits are sufficient for a vector-load-only rewrite; corpus argmax agreement alone is not proof for dp4a. |
| 2 | **Smaller proxy head: Q6_K, Q4_K, or certified two-level search** | **1.5–3.0 ms** (occasionally 3.5) | Low–medium | High | Q6_K is 417.178 MB (0.45 ms less raw floor); Q4_K is 286.065 MB (0.93 ms less). A hierarchical search may avoid more rows, but loose residual bounds can erase the win. Preserve exact greedy with the same upper-bound certificate + BF16 rescore + unconditional full-BF16 fallback. The current fixed 1.5 margin and 64 cap are empirical: if `cnt>64`, the kernel clamps and can omit a winner, so they are not a global mathematical proof. |
| 3 | **Rows-per-warp Q8 head (already staged)** | **0.3–1.2 ms** | Medium | Low–medium | Rows=2/4 reuse each activation load and expose independent accumulators while retaining each row's b/j/reduction order. It cannot remove the 540 MB read. Gate rows=1 vs 2/4 by bitwise equality of all 248,320 proxy logits, candidate set/count, final token, and recurrent/KV state over production-graph streams; no spill in `sm_121a` compile; then require repeated alternating timing outside noise. `FUCINA_QWEN35_Q8_HEAD_ROWS=1` is rollback. |
| 4 | **Fuse/reuse norm + activation quant + GEMV glue** | **0.3–0.8 ms** | Medium–low | Medium | FULL q/k/v currently requantize the same normalized vector three times; GDN qkv/z do it twice. Reuse one exact Q8_1 result, fuse safe norm/quant boundaries, and remove materializations/nodes. Only 2.263 MB is LPDDR-floor traffic, so the win is issue/synchronization, not bandwidth. Gate every quant byte (`q`, scale, sum), projection output, layer residual, and final state bitwise against the incumbent. Do not change RMS reduction or quant rounding order. |
| 5 | **Reduce CUDA-graph nodes per step** | **0.1–0.5 ms** | Low | Medium–high | Forty layers contain hundreds of explicit launches plus library-internal nodes; graph replay removes host launch cost but device dispatch/dependency cost remains. The exact node count must be obtained with `cudaGraphGetNodes` or `nsys --cuda-graph-trace=node`; source counting is not authoritative. Coalesce only dependency-compatible elementwise/routing nodes. Gate graph-on/off outputs and all GDN/conv/KV state bitwise, and require an actual node-count reduction plus end-to-end ms reduction. It cannot save weight bytes. |
| 6 | **MoE grouped-GEMM tile/BK retune** | **0.02–0.04 ms** | High that upside is tiny | Medium | Existing sweep found BK=256 improves only down by 2–3%, gate\|up ~0%; 2–3% of the ~1.1 ms down budget is ~0.03 ms. NVFP4 scale swizzle prevents a useful small-M tile. A different BK may reorder MMA accumulation; call it lossless only if grouped outputs are bitwise equal for every route/count shape and complete greedy/state streams match. Otherwise it is a numerical change, not an exact lever. |
| 7 | **Current Qwen MoE speculation/rollback path** | **≤0 ms today; likely negative** | High | High | Snapshot/restore plus replay is correct, but replaying accepted tokens means target work is at least plain decode plus draft/verify. The production loader also does not bind the official MTP into the continuous-batch MoE path, and `SpecWorthwhile()` rejects MoE. Do not enable it for a speed target. A future versioned-state verifier can qualify only under the acceptance/economics gate below. |

### Lossless speculation option and real acceptance math

The only credible B=1-safe option is **deterministic prompt-lookup/n-gram drafting** (zero learned
model cost) or a loaded MTP, followed by target verification that stores/selects the candidate GDN,
conv, and FULL-KV state for every prefix. It must commit the selected state directly; restore plus
B=1 replay cannot win. MoE expert bytes still scale approximately with verify width because rows
route differently, so dense-model “one weight read for K tokens” economics do not transfer intact.

For a depth-`K` greedy draft with independent conditional acceptance `a`, the expected emitted
length per verify is

`E[T] = 1 + a + a² + … + a^K = (1-a^(K+1))/(1-a)`.

For `K=4`, this is 1.938 at `a=0.5`, 2.306 at 0.6, and 2.773 at 0.7. If an exact versioned target
verify really costs 25 ms/round, break-even against a 17 ms baseline needs `E[T]>25/17=1.47`; 80
tok/s needs `E[T]≥2.00`; 100 tok/s needs `E[T]≥2.50`. Those correspond to roughly `a≥0.52` and
`a≥0.65` for K=4 before draft, state-copy, and rejection overhead. The historical “1.4 accepted
drafts/round” is not enough evidence: its prompt mix, verify cost, and direct-state-commit path were
not production measurements.

The exact gate is algorithmic and runtime-visible:

1. target-verify argmax/rejection uses the same deterministic tie/RNG contract as plain decode;
2. candidate GDN/conv/FULL-KV state after every possible accepted length `j=0…K` is byte-identical
   to `j` trusted B=1 steps—**without replay in the timed path**;
3. emitted token streams and next-step state are identical on long diverse greedy runs; sampled
   mode separately proves the standard rejection distribution;
4. measure the full acceptance-length histogram, `C_verify(K)`, draft cost, commit cost, and route
   collisions, then require `E[T]/C_round` above 80 or 100 tok/s with confidence bounds.

Until all four pass, the expected saving is zero and speculation stays off.

## What cannot reach the target

- Rows-per-warp alone: at most about 1.2 ms versus the 4.5–5.5 ms needed for 80.
- Fusion alone, fewer graph nodes alone, or both: they do not alter the 2.0+ GB dominant read and
  cannot close even half the 80-tok/s gap.
- Expert BK retuning: below 0.04 ms versus multi-ms gaps.
- Any combination that leaves the scalar 5.04 ms Q8 scan intact has an optimistic major-kernel
  ceiling around 86–88 tok/s and a much lower realizable result.
- Current restore-and-replay speculation cannot beat plain B=1 by construction.
- No listed local lever, including a smaller exact-proxy head, honestly supplies the 7–8 ms needed
  for sustained 100 tok/s. That target needs certified byte reduction plus near-peak execution, or
  a versioned-state speculative path with measured `E[T]` high enough to amortize a verify round.

## Measurement gates when the GPU is available

Do not run these as part of this host-only analysis:

```sh
MODEL=/opt/spark/models/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/0b2752837483aa34b3db6e83e151b150c0e00e49

make qwen35-head-rows-test QWEN35_MOE_FP8_MODEL="$MODEL"
make gpu-gates QWEN35_MOE_FP8_MODEL="$MODEL"
make qwen35-longctx-test qwen35-gdn-rollback-test

# Alternate fresh-engine A/B order; retain exact stream hashes and context.
make qwen35-moe-decode-bench
for REP in 1 2 3; do
  if [ $((REP % 2)) -eq 1 ]; then ORDER="1 4"; else ORDER="4 1"; fi
  for ROWS in $ORDER; do
    flock -w 1800 /tmp/fucina_gpu.lock -c \
      "FUCINA_QWEN35_Q8_HEAD_ROWS=$ROWS /tmp/fucina_moe_decode_bench '$MODEL' 256 32" \
      | tee "/tmp/q36-q8-head-r${ROWS}-rep${REP}.log"
  done
done

flock /tmp/fucina_gpu.lock -c \
  "FUCINA_QWEN35_Q8_HEAD_ROWS=1 nsys profile --trace=cuda --cuda-graph-trace=node \
   -o /tmp/q36-b1-nodes --force-overwrite=true \
   /tmp/fucina_moe_decode_bench '$MODEL' 128 32"
```

Report median and dispersion over alternating runs, engine ms/step, exact output/state hashes,
prior context, clocks, DRAM counters, L2 hit rate for GQA KV and Q8 activations, graph node count,
and contention. An isolated kernel result is not served tok/s.
