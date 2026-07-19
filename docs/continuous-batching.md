# Continuous batching + paged KV — design & plan

> **Update (post-Qwen):** this doc was written when continuous batching was a Gemma-4-only,
> opt-in experiment. That mechanism is unchanged, but it is no longer opt-in for every model:
> **every Qwen3/Qwen3.5/Qwen3.6 checkpoint is now served exclusively through this path**, because
> Qwen has no working single-flight prefill entry point (`cmd/fucina/main.go` detects
> `eng.IsQwen3Family()` and force-enables `--batch`/`--paged-kv`; startup fails fast if the paged
> pool can't allocate). The "why default-off" reasoning below still applies to **Gemma-4 only**.
> For the Qwen-specific serving behavior (mandatory batching, dense-only spec decode inside the
> batch path, the `response_format` 501 gap), see the README's
> [Continuous batching & paged KV](../README.md#-continuous-batching--paged-kv) section and
> [docs/qwen-models.md](qwen-models.md). The design/internals below are still accurate for both.

Status: **FUNCTIONAL** (on `main`). Enable for Gemma-4 with the `--batch` CLI flag (implies
`--paged-kv`), or the legacy `FUCINA_PAGED_KV=1 FUCINA_BATCH=1` env pair — both converge on the
same path. Default path for Gemma-4 is unchanged (single-flight); Qwen3 is always on this path.
Concurrent smoke: 4 parallel requests served concurrently (4.43s vs 5.56s sequential), correct
outputs, no serialization. The current source also has Gemma-4 paged batched MTP and per-B CUDA
graphs; the missing work is fresh served evidence and product-default calibration, not those
kernels. See [docs/gemma-gb10-competitive.md](gemma-gb10-competitive.md) for the 2026-07 source and
measurement audit, and the scoreboard in
[docs/qwen35-beat-vllm-plan.md](qwen35-beat-vllm-plan.md) for Qwen.

Why default-off for Gemma-4: this is the **current product default**, not evidence that the batch
path lacks MTP. With `--batch --assistant <head.gguf>`, `BatchAdapter.StepBatch` routes to the
paged B-row MTP drafter and one lossless target verify; without an assistant, dense Gemma uses the
model-agnostic prompt-lookup verifier unless `FUCINA_NO_BATCH_SPEC=1`. The default remains off
because the fresh three-start Q4_0 matrix is shape-dependent: learned MTP is +99.6% at N=1 and
+37.0% at N=2, but −22.8% at N=4 and −23.2% at N=8 before tapering to plain at N≥16; separate
paged/batch GPU gates also fail. Use `--batch` for concurrent Gemma only after checking the
checkpoint and traffic shape you actually deploy. `--spec=false` controls the single-flight
generator; use `FUCINA_NO_BATCH_SPEC=1` to obtain a genuinely plain continuous-batch baseline.

How it works end to end:
- `FUCINA_PAGED_KV` allocates the block pools (capacity-sized; `paged_cap = min(MAX_SEQS,
  max_seqs+1)` bounds concurrency to what the pool backs).
- `FUCINA_BATCH` builds a `batch.Scheduler` (single owner goroutine) over a cgo `BatchAdapter`
  implementing `BatchEngine` (AddSeq/StepBatch/RemoveSeq/Capacity → `gemma4_engine_seq_*`).
  `serveCompletions` routes to `serveBatch` (Submit) instead of the per-request `s.kv.Lock()`.
- C `gemma4_engine_step_batch` runs ONE `decode_multiseq_forward` over B independent slots
  (per-row positions + per-row paged block tables; `paged_attn_decode_batched` for attention),
  samples one greedy token per row. Per-row admission: a slot that can't grow its KV is marked
  `-1` and excluded; the scheduler stops just that sequence (never the whole batch).

Paged batched attention is implemented as SPLIT-K and is **intended** to be bit-identical to the
contiguous split-K decode (`paged_sliding_attn_splitk_batched` /
`paged_global_attn_splitk_batched` +
`paged_flash_decode_combine_batched` in cuda/paged_kv_device.cuh): per (head,seq,split)
online-softmax partials reading K/V through each row's block table, merged in the same
split order — so the per-row n_splits/per/scan/combine all match the contiguous rows
kernels, and only the K/V *address* (block-table lookup vs ring modulo) differs. Historical runs
reported 64/64 agreement. The fresh 2026-07-19 gate does **not** validate the unqualified claim:
its global case differed from the contiguous oracle by 0.0112 (while paged-vs-host was 1.19e-07);
sliding and recycled-sliding passed. Treat global bit-identity as unresolved until the oracle/path
mismatch is root-caused. Tradeoff: at
batch decode the attention is not the bottleneck (weight GEMV bandwidth is), so split-K's extra
combine launch costs ~10% aggregate tok/s vs the old sequential kernel at short/medium contexts
(measured 4×512 long-gen: 57.0 → ~51 tok/s on the GB10). Kept because bit-identity to the
contiguous path is structurally impossible with a single-scan kernel (different summation order).

The greedy multi-seq decode step is now a CUDA graph captured PER batch size B (1..MAX_SEQS),
`multiseq_graph_ensure` / `decode_multiseq_body` in cuda/gemma4_kernels.cu. Per-row positions
(`d_ms_pos`), per-row paged views (`d_ms_views_slid/glob`, which carry each row's block-table
device pointer + length) and per-row tokens (`d_sb[0]`) are device-resident and refreshed each
step OUTSIDE the capture (same trick as `d_specpos`), so one captured graph replays across steps
at any position; attention launches at the full split grid (`GEMMA4_GLOBAL_MAX_SPLITS`, each row
tail-returns past its own n_splits → bit-identical to the per-kernel split-K path). Logs
"multiseq batch graph captured (B=%d)" once per distinct B. Temperature rows (any `temp>0`),
capture failure, or `FUCINA_NO_BATCHED_GRAPH` fall back to the per-kernel body (which also runs
the per-row sampler; the same env var also disables the spec-verify `batched_graph`). Historical
runs reported a 32/32 batch self-test and correct B=1/3/4 replay. The fresh evidence run reported
`seq_add(batch) failed`, so those historical results are not promoted to a current passing gate.

Known limitations in current source:
- Batched Gemma MTP self-drafting is greedy-only. Temperature rows are target-sampled losslessly
  but are not eligible for the learned MTP draft round.
- Verify scratch is capped at 32 total rows (`B + Σdrafts`); default depth is capped at 6 and
  speculation auto-disables when fewer than 2 draft rows per slot fit. MTP therefore tapers out at
  high concurrency by design.
- `/metrics` has speculation and request/TTFT fields, but the fresh batched-MTP run left the
  speculation counters and average TTFT at zero while MTP was active. It also lacks per-request
  p95/p99 and accepted-length histograms. Batch speculation telemetry is therefore not usable for
  a production SLO decision yet.
- Fresh dense prompts that fit the supported tile use single-pass paged prefill; prefix/chunk
  suffixes use batched chunks of up to 32 rows. Unsupported/long shapes retain a correct
  token-by-token fallback, so long-prompt performance must be measured rather than inferred.
- `response_format`/`json_schema` remains unavailable under continuous batching.

---
Original plan (branch `perf/continuous-batching-paged-kv`).

## Motivation

fucina today is **single-flight**: `s.kv.Lock()` (internal/server/server.go:967) is held for
the *entire* request (prefill → end of generation). Measured on the GB10 (tool-eval-bench,
runs/2026/06/...): per-stream `tg t/s` stays ~40–50 under concurrency, but **TTFT scales
linearly** with the number of clients (d0: 923ms→2.67s→6.6s at c1/c2/c4; d8192: 5.0s→9.0s→17.0s).
Aggregate throughput does **not** grow with concurrency — concurrency is pure queue cost.

Goal: real continuous batching of independent sequences in one batched forward pass, with a
paged KV cache and per-step (not per-request) serialization.

## Key facts about the current engine (cuda/gemma4_kernels.cu)

- KV cache is **already FP8 E4M3** (`typedef __nv_fp8_storage_t kv_t;` line 242). Storage
  "quantization" is largely done; remaining quant work is accuracy/memory refinement (Phase 6).
- Model: 48 layers; 40 sliding (window 1024, nkv=8, head_dim=256, RoPE θ=10k) + 8 global
  (full ctx, nkv=1, head_dim=512, RoPE θ=1M). `GEMMA4_HEADS=16`, `GEMMA4_HIDDEN=3840`,
  `GEMMA4_VOCAB=262144`, `GEMMA4_SPEC_MAX=16`.
- Sliding KV is a **ring buffer** cap `FUCINA_SLIDING_RING` (default 8192): slot = pos % cap.
  Global KV is flat: index = pos. Both per-layer contiguous, one physical sequence.
- Kernels are **already multi-row** (`decode_batched_forward`, `*_attn_splitk_rows_kernel`,
  `gemv_batched_w`): row dim = `blockIdx.y`. BUT today the K rows are *consecutive positions of
  one sequence* sharing one KV cache (causal: row r attends pos+r). Independent-sequence batching
  needs per-row position, per-row length, per-row KV (block table).
- CUDA graphs are **position-independent** via device-resident pos (`d_decpos[2]`, `d_specpos[2]`),
  captured **one graph per K**. Generalize to one graph per active batch size B.
- Single implicit position counter `global_n_tokens` (line 2704) — must become per-sequence.
- Go engine is single-sequence: one `*C.gemma4_engine_t`, one `logitsBuf` reused per call
  (bridge.go:72). Lock order KVCache.mu → Engine.mu.

## Decisions (2026-06-16, with user)

1. **Max concurrency: dynamic** — block pool sized at runtime to free VRAM after weights.
2. **KV layout: paged** (block table + free-list), not fixed contiguous slots.
3. **Spec decode: kept per-sequence inside the batch** (per-row MTP draft+verify), not disabled.

## Plan (each phase compiles + verifies on the GB10 before the next)

1. **Per-sequence state in C engine.** `seq_t` handle: own pos/len/KV region. Execution still
   serial (1 seq/step) → behavior-identical. ABI: seq_create/prefill/decode_step/destroy.
   Verify: single-stream t/s + tool-bench identical.
2. **Paged KV allocator.** Shared block pool (~256-tok blocks) + per-seq block table + free-list,
   sized to free VRAM. 1 seq ≡ contiguous. Verify: single-stream correct, memory accounting.
3. **Batch-aware attention w/ block-table indirection.** rows kernels: row r = independent seq,
   per-row d_pos[r]/d_seqlen[r]/block_table[r]. Verify: batched == B separate forwards.
4. **Go continuous-batching scheduler.** Engine-owning goroutine; admission, chunked interleaved
   prefill, per-step batch build, on-device per-row sample + scatter, per-seq evict + free.
   Replace per-request lock with per-step. Verify: c2/c4/c8 aggregate scales, TTFT flat.
5. **Graphs per batch size + per-seq spec in batch.** Verify: hit-rate, perf.
6. **(opt) KV quant refinement.** per-head/block FP8 scales or NVFP4 global. Gate on tool-bench.

## Landed so far (branch state)

- `cuda/paged_kv.h` (+ host test `make paged-kv-test`): block pool, per-seq block tables,
  budget-sizing, sliding-window recycling. Verified bound: a 10k-token sliding seq holds ≤5
  blocks.
- `cuda/paged_kv_device.cuh` (+ GPU test `make paged-kv-device-test`): device kernels
  `paged_kv_write` / `paged_attn_gather` + POD descriptors `PagedSeqView`/`PagedWriteBatch`.
  Verified: block-table indirection is BIT-IDENTICAL to contiguous on the read path
  (`readback=0 EXACT`), attention numerically correct (1e-7 vs host), incl. recycled `base>0`.
- Phase 1 increment 1: `gemma4_seq` struct + `eng->cur`, position migrated
  (`global_n_tokens`→`cur.n_tokens`). Compiles sm_121a, smoke output byte-identical to baseline.
- `internal/server/batch/` (Phase 4 scheduler skeleton, `make go-test`): single-owner-goroutine
  loop, admission/eviction/backpressure, tested with a mock. Race-clean.

### The BatchEngine contract (CUDA side must implement)

```go
type BatchEngine interface {
    AddSeq(prompt []int32, params SeqParams) (slot int, first int32, err error)
    StepBatch(active []int32, inputs []int32) (out []int32, err error)
    RemoveSeq(slot int) error
    Capacity() int   // dynamic — polled each admission pass
}
```
On-device sampling both for prefill (returns first token) and each step (one token/slot).

### Historical snapshot: spec decode interface before the CUDA kernel landed

> **Historical, not current capability.** The text below records the intermediate branch state.
> Current source resolves the listed CUDA gaps with per-slot recurrent state,
> `mtp_forward_paged_batched` / `mtp_draft_paged_batched`, ragged paged target verification, and
> lossless per-slot commit in `step_batch_spec_impl` (`cuda/gemma4_kernels.cu`).

The scheduler interface now carries a token RUN per slot: `StepBatch(active, inputs) (out
[][]int32, err error)`. The scheduler's `step` walks each row's run in order, calling `deliver`
per token, and stops emitting the instant a token evicts the row (stop token / budget / cancel /
the -1 KV-exhausted sentinel) — so drafted tokens past the boundary are dropped. The non-spec C
path still samples exactly one token per slot; `BatchAdapter.StepBatch` wraps each into a length-1
run, so behavior is unchanged (batch self-test 32/32, regression byte-identical). Two unit tests
cover multi-token runs (`TestSpeculativeRunsDeliverEveryToken`) and a stop landing mid-run
(`TestSpeculativeRunStopsMidRun`).

STILL PENDING **at that historical snapshot** (the hard CUDA half): the MTP drafter (`mtp_forward`) was structurally single-
sequence — it reads K/V from the CONTIGUOUS single-seq cache at fixed layer offsets
(`d_sliding_k + (MAX_LAYERS-2)*lstride`, `d_global_k + global_slot[MAX_LAYERS-1]`), its recurrent
state `d_mtp_h` is one `[3840]` buffer, and its attention launches at a single-row grid. Per-slot
spec-in-batch needs (1) per-slot `d_mtp_h[MAX_SEQS][3840]`, (2) a PAGED + BATCHED drafter attention
reading each slot's block table (the contiguous cache is not where batch KV lives), and (3) a
variable-K, per-slot, 2D-batched verify (slot × draft-position) over each slot's paged KV — a new
kernel geometry that does not yet exist. The Go interface above is the prerequisite that unblocks
that work; until the C side lands, runs stay length-1 in the batch path.

## Verification harness

Reuse the existing bench in runs/ (spec-bench + tool-eval-bench). After each phase:
single-stream must be unchanged (t/s within noise, tool-bench score identical / deterministic
seed 42). Concurrency wins are validated at Phase 4+ via the c2/c4 columns and TTFT.
