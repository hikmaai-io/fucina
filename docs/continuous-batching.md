# Continuous batching + paged KV — design & plan

Status: **FUNCTIONAL** (on `main`). Behind `FUCINA_PAGED_KV=1 FUCINA_BATCH=1`; default
path unchanged. Concurrent smoke: 4 parallel requests served concurrently (4.43s vs 5.56s
sequential), correct outputs, no serialization. Remaining: perf (split-K paged attention),
per-seq spec decode, sampling params in the batch kernels, CUDA graphs per batch size.

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

Paged batched attention is now SPLIT-K and BIT-IDENTICAL to the contiguous split-K decode
(`paged_sliding_attn_splitk_batched` / `paged_global_attn_splitk_batched` +
`paged_flash_decode_combine_batched` in cuda/paged_kv_device.cuh): per (head,seq,split)
online-softmax partials reading K/V through each row's block table, merged in the same
split order — so the per-row n_splits/per/scan/combine all match the contiguous rows
kernels, and only the K/V *address* (block-table lookup vs ring modulo) differs. The single-
seq paged_read path uses the same kernels (B=1 view), so FUCINA_PAGED_E2E_SELFTEST now agrees
on all 64/64 tokens (was "numerically equivalent", a few top-2 near-tie flips). Tradeoff: at
batch decode the attention is not the bottleneck (weight GEMV bandwidth is), so split-K's extra
combine launch costs ~10% aggregate tok/s vs the old sequential kernel at short/medium contexts
(measured 4×512 long-gen: 57.0 → ~51 tok/s on the GB10). Kept because bit-identity to the
contiguous path is structurally impossible with a single-scan kernel (different summation order).

Known limitations: no spec decode / TTFT metrics in the batch path. (Per-sequence sampling
params are now honored; see the sampling-params phase note.)

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

### KNOWN GAP — spec decode in the batch (decision #3)

`StepBatch` returns ONE token per slot. But per-sequence spec decode (MTP draft+verify) emits a
VARIABLE number of accepted tokens per slot per step. Before Phase 5 the interface must grow to
e.g. `StepBatch(...) (out [][]int32, err error)` (a token run per slot), and the scheduler's
`deliver` loop must emit each run with stop/budget checks mid-run. Tracked on task #5. Phase 4's
1-token base case is correct as-is for B>1 non-spec batching.

## Verification harness

Reuse the existing bench in runs/ (spec-bench + tool-eval-bench). After each phase:
single-stream must be unchanged (t/s within noise, tool-bench score identical / deterministic
seed 42). Concurrency wins are validated at Phase 4+ via the c2/c4 columns and TTFT.
