# Continuous batching + paged KV — design & plan

Status: **in progress** (branch `perf/continuous-batching-paged-kv`).

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
