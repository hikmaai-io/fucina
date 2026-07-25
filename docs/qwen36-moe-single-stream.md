<!-- ABOUTME: Evidence-based Qwen3.6-35B-A3B-FP8 B=1 decode attribution and exact speedup gate. -->
<!-- ABOUTME: Records why speculation, rollback, lossy heads, and expert retuning are not the keeper. -->
# Qwen3.6-35B-A3B-FP8 single-stream decode on GB10

Status: **implementation staged and default-off; GPU A/B pending**. The roadmap target is
**80–100 tok/s**, but no target attainment or speedup claim is made. Target hardware is DGX Spark
GB10 (48 SM, compute capability 12.1, `sm_121a`, about 273 GB/s LPDDR5X). The observed production
continuous-batch B=1 baseline is **54–59 tok/s**, or roughly 17–18 ms per generated token.

## Findings and claim boundary

- The rows-per-warp Q8 head is a credible ILP experiment, not an 80–100 tok/s result. It remains
  opt-in until exact production-graph parity and a repeatable hardware speed A/B pass.
- The existing Qwen MTP MoE serving proposal remains **unsafe/unprofitable for production until its
  rollback, exactness, and acceptance-rate/profitability gates pass**. A tested rollback primitive
  alone does not make the unintegrated production path safe or worthwhile.
- Do not overclaim: report measured commands, model snapshot, context, and distributions; do not
  turn an isolated kernel delta into served tok/s or target attainment.

## Actual serving path and 17 ms attribution

The Go scheduler calls `BatchAdapter.StepBatch`, which reaches
`gemma4_engine_step_batch` -> `qwen35_step_batch` -> `qwen35_ms_run`. Greedy B=1 is a
self-contained CUDA graph: device slot splice -> 40 hybrid layers -> exact greedy head -> slot
writeback. The only required host synchronization is the final four-byte token readback. Historical
telemetry records 16.54–17.93 ms engine steps at average B close to one, while Go delivery is about
0.001–0.002 ms; scheduler work is not the lever.

The current evidence accounts for the step as follows. Values should not be added as if they came
from one simultaneous profile; the component microbenches and serving traces were collected in
separate historical runs.

- **Exact LM head: 5.04 ms isolated (about 29% of a 17.2 ms step).** B=1 already avoids the
  1.02 GB BF16 full scan. It scans a 0.54 GB Q8_0 approximation, selects nearby candidates, and
  exactly rescans their BF16 rows. The BF16 incumbent measured 5.96 ms. Thus “quantize the head”
  has already been done in the only form that retained the observed greedy stream.
- **Routed expert GEMMs: about 2.6 ms/step isolated.** Top-8 routing reads about 9.4 MB for fused
  gate|up and 4.7 MB for down per layer at one token/expert. Across 40 layers that is about 564 MB.
  There are two grouped CUTLASS launches/layer (80 graph nodes/step). They measured about
  0.039+0.027 ms/layer and 80–85% of GB10 DRAM bandwidth. The graph hides host launch latency;
  bytes, not launches or empty MMA tiles, are the floor.
- **Remaining roughly 9–10 ms:** the 0.88 GiB packed-Q4_K mixer store, shared expert, routing and
  activation quantization/glue, GDN/conv, and eight FULL-attention layers. Historical B=16 node
  attribution found mixer, GDN, shared expert, and activation quantization material; it did not
  identify a B=1 host bubble. A fresh B=1 `nsys` run is required before assigning finer percentages.

This budget makes 80 tok/s (12.5 ms) an aggressive target and 100 tok/s (10 ms) more aggressive.
The change below attacks the largest isolated component but cannot credibly close the full gap by
itself. Even the unattainable best case of reading the 0.54 GB Q8 head at the full 273 GB/s device
rate is about 2.0 ms, saving only about 3.0 ms from the measured 5.04 ms. Applied to the 54–59 tok/s
baseline, that mathematical ceiling is only about 65–72 tok/s before real instruction, cache, and
occupancy costs. A realistic result is therefore a smaller several-tok/s gain; only the A/B below
may establish its size.

## Explicit alternatives

### Existing one-layer MTP head and `SpecWorthwhile`

The official FP8 checkpoint contains `mtp.*` tensors for one full-attention decoder layer. Fucina's
M6 standalone FP8 oracle loads all 22 tensors and implements a lossless sequential verifier. It is
not the production `gemma4_engine_t` continuous-batch drafter: the production loader does not bind
that head into `eng->mtp`, and Qwen serving cannot exercise the Gemma `--assistant` head.
Chain-local MTP history measured only about 1.4 accepted drafts/round; persistent history was an
unimplemented estimate.

More importantly, `BatchAdapter.SpecWorthwhile()` deliberately returns false whenever
`NExperts()>0`. A K-row target verify generally routes to different top-8 experts per row, so routed
expert bytes approach K times the B=1 bytes instead of one amortized dense-weight pass. The Go
scheduler therefore does not install `StepBatchSpec` for this MoE. This is a profitability gate,
not a quality weakening: external prompt-lookup and learned drafts are both losslessly target
verified where enabled. Keep production MTP MoE disabled until integrated rollback/exactness gates
and a measured acceptance-rate profitability gate all pass.

### Safe GDN/KV rollback

Safe rollback exists and is tested. `q35_gdn_snapshot` copies the fixed recurrent GDN+conv slab;
commit restores it and replays exactly the accepted tokens through ordinary B=1 decode. FULL-layer
KV is absolute-position indexed: replay overwrites accepted positions and stale later positions are
never read. The j=0..6 gate proves byte-identical state and continuation.

Rollback solves correctness, not B=1 throughput. Historical DFlash work measured that batched
verify differs from trusted B=1 decode at interior positions because projection/FULL-attention and
head numerics differ. Exact commit therefore had to replay accepted tokens, making target work at
least plain-decode work plus draft/verify overhead. Using stale GDN state, truncating only KV, or
trusting the divergent batched argmax is not an acceptable shortcut.

### LM-head quantization/fusion

Lossy per-row FP8 head generation flipped argmax and remains off. Extending the exact Q8 candidate
path to B>1 regressed B=16 throughput 2.2x because its scalar candidate scans underfilled GB10;
the weight-read-once BF16 batched head is correct there. B=1's Q8 scan itself reaches only about
107 GB/s because the scalar int8-to-float dot is latency/ILP limited. This is the best credible B=1
kernel target.

### MoE expert bytes and launches

The serving checkpoint's nominal FP8 experts are requantized at load to grouped NVFP4. At B=1 the
CUTLASS kernel already reads each selected expert once and runs near the memory floor. BK=256 gave
only 2–3% on down and approximately zero on dominant gate|up (<1% whole-step estimate), while the
NVFP4 scale-factor swizzle prevents a useful smaller-M tile. Fusing the two dependent projections
is not generally possible across SiLU and requantization. The 80 expert GEMM graph nodes are not
80 host launches. Expert retuning is lower confidence and lower upside than the head change.

## Implemented keeper candidate: register-blocked exact Q8 search

`q8_head_gemv_rows_kernel<ROWS>` evaluates 2 or 4 independent vocabulary rows per warp and reuses
each activation load across those rows. For every row it preserves the legacy block order, j order,
FP32 operations, and shuffle reduction. Candidate selection and exact BF16 rescoring are untouched.
It therefore changes instruction-level parallelism only, unlike another lossy quantization. The
host-only `sm_121a` compile reports 40/40/72 registers per thread for rows=1/2/4 respectively and
no local-memory spill; rows=4's extra register pressure is another reason to expect less than the
idealized bandwidth gain.

Hardware/config gate:

- rows=1 remains the default while the hardware gate is pending;
- on the target GB10 MoE, `FUCINA_QWEN35_Q8_HEAD_ROWS=2` or `4` enables the experiment and `1` is
  immediate rollback;
- malformed values fail closed to rows=1;
- the choice is fixed before graph capture; the test-only setter synchronizes, destroys every
  cached Qwen graph executable, and forces recapture before changing it, so no graph replays a
  kernel captured with a different row count;
- production session and distributed lifecycle are otherwise unchanged.

CPU dispatch tests are mandatory in normal `go-test`. The staged GPU gate runs 128 production graph
tokens with rows=1 and rows=4 and requires exact token equality. Full model gates remain required.
Because the GPU is currently busy, neither parity nor speed has been measured on this commit yet.

## Commands to run when the GPU is available

```sh
MODEL=/opt/spark/models/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/0b2752837483aa34b3db6e83e151b150c0e00e49

# Exactness first.
make qwen35-head-rows-test QWEN35_MOE_FP8_MODEL="$MODEL"
make gpu-gates QWEN35_MOE_FP8_MODEL="$MODEL"
make qwen35-longctx-test qwen35-gdn-rollback-test

# Exact hardware A/B: build once, then alternate three fresh-engine runs under the shared lock.
make qwen35-moe-decode-bench
for REP in 1 2 3; do
  if [ $((REP % 2)) -eq 1 ]; then ORDER="1 4"; else ORDER="4 1"; fi
  for ROWS in $ORDER; do
    flock -w 1800 /tmp/fucina_gpu.lock -c \
      "FUCINA_QWEN35_Q8_HEAD_ROWS=$ROWS /tmp/fucina_moe_decode_bench '$MODEL' 256 32" \
      | tee "/tmp/q36-q8-head-r${ROWS}-rep${REP}.log"
  done
done

# Node attribution (repeat rows=4 with a different output name).
flock /tmp/fucina_gpu.lock -c \
  "FUCINA_QWEN35_Q8_HEAD_ROWS=1 nsys profile --trace=cuda --cuda-graph-trace=node \
   -o /tmp/q36-b1-head-r1 --force-overwrite=true \
   /tmp/fucina_moe_decode_bench '$MODEL' 128 32"
```

Then run the exact served protocol in `benchmark-evidence/PROTOCOL.md` in fresh server starts,
alternating rows=1 and rows=4. Record median of at least three runs, exact generated text/hash,
engine ms/step, context length, clocks, and contention. Promote rows=4 from opt-in only if the full
production gate is bit-exact and the B=1 improvement is outside noise; otherwise leave rows=1 as
the default. Do not report 80–100 tok/s unless that served benchmark actually measures it.
