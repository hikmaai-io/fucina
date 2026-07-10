# GB10 inference platform improvement plan

_Status: active · 2026-07-09 · primary target: Qwen3.5/Qwen3.6 hybrid dense + MoE on NVIDIA GB10 (sm_121a, unified LPDDR5X)_

## Objective

Build a fast, memory-predictable inference server that scales from one interactive coding session to 16 concurrent requests without changing model results or failing late under memory pressure.

The governing GB10 constraints are:

1. **Decode is usually memory-bandwidth bound.** Reduce bytes moved before adding arithmetic or launches.
2. **Qwen hybrid state is not ordinary KV.** Each sequence owns fixed Gated-DeltaNet (GDN) recurrent state and conv history plus length-proportional K/V on FULL-attention layers.
3. **Prefill and decode need different kernels.** Wide prefill should use tensor cores and tiled online attention; decode should use weight-reuse GEMV/grouped GEMM and split-KV attention.
4. **Unified memory is shared.** Admission must be based on allocations the engine can guarantee, not a compile-time slot count or stale free-memory sample.
5. **Every optimization needs three gates:** architecture/oracle parity, batch/sequence independence, and measured end-to-end throughput/TTFT.

## Review summary

### Already strong

- Runtime Qwen3.5/3.6 geometry and FULL/GDN layer dispatch.
- CUDA-graph decode per batch width.
- FP16 FULL-layer KV and BF16 GDN state arenas.
- Split-KV flash decoding for long-context decode.
- Wide tensor-core prefill projections.
- Chunked GDN scan, chunked continuation prefill, and state snapshots.
- Packed Q4_K mixer kernels and grouped NVFP4 MoE experts.
- Per-row activation scaling for batch-invariant grouped expert execution.
- Long-context, graph, state-restore, diverse-burst, and oracle parity gates.

### Highest-value gaps

| Priority | Gap | Impact |
|---|---|---|
| P0 | Hybrid arenas were always allocated for 16 slots and legacy Gemma KV was allocated too | Several to tens of GiB wasted; poor coexistence and late OOM risk |
| P0 | Qwen hybrid batching silently ignored temperature/top-k/top-p/min-p | Incorrect server semantics; all requests were greedy |
| P0 | Main had a merged GQA paged-attention template regression and did not build | Platform blocker |
| P1 | Qwen memory is allocated as monolithic per-layer slabs, eagerly for every slot | Capacity cannot grow/shrink with load; fragmentation and startup pressure |
| P1 | Memory budget does not yet reserve every lazy Qwen allocation (attention scratch/weight cache) | A request can cross the configured budget after load |
| P1 | Fresh FULL-attention prefill materializes `NQ × N × N` scores/probabilities | O(N²) memory; ~6.4 GiB at N=8192/NQ=16 before Q/K/V buffers |
| P1 | Scalar shared-score fallback caps Qwen context at device max shared memory | Artificial ~25K-class context ceiling despite split-KV decode |
| P1 | GDN prefill uses a hand-written 64-token TF32 WMMA scan | Good correctness, but below FlashQLA-class fusion/occupancy |
| P2 | Model weights, immutable metadata, scratch, and per-sequence state live in one 20K-line engine object/TU | Hard lifetime reasoning, hard memory accounting, slow iteration |
| P2 | Prefill admission and decode are interleaved at scheduler level, not in one GPU execution plan | Head-of-line latency and lost overlap under mixed traffic |
| P2 | Split count/tile choice is static per engine | Leaves occupancy or combine overhead on the table as B/context changes |

## Work completed in this pass

1. **Restored the GQA-aware paged global-attention template** (`NH,NKV,HD,MAX_SPLITS`) and the runtime KV-head offset. This fixes the main build and preserves 31B GQA correctness.
2. **Made CUDA header dependencies complete in the Makefile.** Editing FP8/Qwen/detection/loader headers now rebuilds the CUDA archive instead of silently linking stale objects.
3. **Removed duplicate Make targets** that were overriding Qwen/FP8 test recipes.
4. **Added runtime Qwen state capacity.** `--parallel`, `--max-concurrent`, and one-shot mode now map to the number of hybrid state arenas actually allocated, capped at 16. Direct C tests retain 16 unless configured.
5. **Stopped allocating legacy contiguous Gemma KV for Qwen3.5.** Qwen uses only its hybrid GDN/FULL arenas; the memory budget and allocation path now agree.
6. **Added real per-row sampling to Qwen hybrid serving.** Temperature, top-k, top-p, min-p and deterministic per-sequence RNG now work for prefill and decode. Mixed greedy/sampled batches are sequence-independent; all-greedy batches retain CUDA graphs.
7. **Rejected duplicate slot IDs in one Qwen batch.** Advancing the same recurrent state twice in one launch is now an error instead of a GDN/conv/KV race.
8. **Added a sampling batch-independence gate** to the Qwen self-test and eager hybrid-arena memory telemetry.
9. **Split the Qwen implementation out of the generic monolith.** `gemma4_kernels.cu` now includes focused internal modules: `qwen35_kernels.cuh` (device kernels/oracle), `qwen35_runtime.cuh` (serving/state/prefill/decode), `qwen35_backend.cuh` (FP8/NVFP4/MTP/MoE), and the strictly opt-in `qwen35_jspace.cuh` (J-Lens diagnostics/interventions).
10. **Separated the Qwen runtime data model.** `qwen35_state.cuh` owns hybrid state, workspace, graphs, caches, and named memory accounting under `engine.q35` rather than mixing hundreds of Qwen fields into the generic engine.
11. **Added a pre-allocation memory plan and transactional rollback.** Qwen capacity is reduced to a physically and policy-guaranteed value before state allocation; partial workspace failures release every Qwen-owned allocation and can be retried.
12. **Added public C/Go memory statistics.** Workspace, recurrent/KV bytes per slot, committed/reserved/peak bytes, allocated/configured slots, capacity, and effective context are now queryable and covered by the batch gate.
13. **Made Qwen sequence state lazy and pool-backed.** Stable device pointer tables keep captured graphs valid while recurrent/KV storage is allocated transactionally only when a slot is first admitted, then retained across slot reuse. At three configured 4K slots, initial committed memory is now the 1.95 GiB workspace instead of the full 2.40 GiB reservation.
14. **Made FULL-layer KV block-grown.** Each admitted slot starts with 256 tokens and grows geometrically in 256-token-aligned generations. Prefixes are copied before atomically publishing new pointers; old generations are released afterward. A dedicated gate validates 256→512 growth followed by replay of an already captured decode graph, and the 1K/4K oracle gate remains 40/40.
15. **Coalesced each slot's fixed GDN/conv state into one aligned slab.** The 24 LINEAR layers now use non-owning aligned views into one transactional allocation instead of 48 independent allocations. This reduces allocator fragmentation and makes fixed recurrent admission all-or-nothing while preserving snapshot and captured-graph pointer-table semantics.
16. **Root-caused Qwen model-residency/slot-count startup variance.** The old `free_at_create - free_after_load` value mixed true CUDA allocations with Linux file-cache growth on GB10 unified memory. A cold safetensors mmap read charged roughly 24.5 GiB of newly populated page cache as “weights+scratch”; warm restarts reported 24.8–25.8 GiB. The new exact allocation ledger is stable at 23.13 GiB (16.88 expert weights + 0.88 core + 3.35 embedding copies + 1.45 head + 0.58 MoE scratch), and logs CUDA checkpoints, `/proc/meminfo` free/available/cache, Q4_K/NVFP4 decisions, weight-cache fit inputs, and memory-plan inputs. Five sequential restart probes reproduced the cold/warm split and identify the variance source rather than merely observing it. Behavior is unchanged; using reclaimable memory for admission remains a later capacity-policy change.

Measured gate after these changes on Qwen3.5-35B-A3B-FP8 with three allocated slots:

- Oracle parity: **8/8**
- B=3 row independence: **24/24 for all rows**
- Graph on/off: **24/24 for all rows**
- Mixed greedy/sampled solo-vs-batch: **PASS**
- Runtime arena capacity: **3/16**, legacy Qwen KV allocation: **0 MiB**

## Execution plan

### Phase 1 — memory model and guaranteed admission (next)

#### 1.1 Split the runtime data model

Introduce explicit ownership types:

- `ModelWeights`: immutable tensor metadata, quantized stores, scale tables, embedding/head.
- `EngineWorkspace`: streams, handles, graph cache, projection/attention/GDN scratch.
- `SequencePool`: slot allocator and per-sequence state ownership.
- `QwenHybridState`: GDN state, conv ring, FULL-layer KV and token/RNG counters.
- `MemoryPlan`: named allocations, eager/lazy class, bytes, budget and lifetime.

Keep the C ABI stable while migrating internals. This makes double allocation, aliasing, and teardown auditable.

#### 1.2 Make hybrid state lazy and pool-backed — implemented

- [x] Allocate sequence state on slot admission, not for every configured slot at first request.
- [x] Reuse freed state blocks from a device pool; do not `cudaFree` on every request.
- [x] Keep compute scratch separate from sequence-state capacity.
- [x] Grow FULL-layer KV geometrically in 256-token-aligned generations rather than reserving `maxctx` per slot.
- [x] Coalesce fixed GDN/conv state into one aligned slab per sequence to reduce allocation fragmentation. Snapshot copies remain layer-ordered to preserve the stable cache format.

**Acceptance:** configured capacity is admitted only when its worst-case block reservation fits; no request fails mid-decode because another slot consumed its promised memory.

#### 1.3 Account for all lazy memory

Before first request, reserve/budget:

- graph instances and graph-private allocations;
- hybrid fixed state and initial KV blocks;
- wide-prefill activation/projection scratch;
- FULL-attention scratch;
- optional resident BF16/NVFP4 weight caches;
- grouped-MoE scratch;
- host-pinned snapshot pool.

Add `/metrics` counters for committed, reserved, peak, state, KV, scratch, weight-cache and snapshot bytes.

#### 1.4 Transactional allocation

Replace partial `cudaMalloc` chains with an allocation transaction that frees every successful allocation on failure. Use a CUDA memory pool / `cudaMallocAsync` where capture and toolchain behavior are validated; otherwise use a typed reusable arena.

### Phase 2 — SOTA attention management

#### 2.1 Streaming FULL-attention prefill

Replace `NQ × N × N` score/probability materialization with a tiled online-softmax kernel:

- query tiles × KV tiles;
- GQA KV loaded once per KV head and reused by its query-head group;
- FP32 online `(m,l,o)` accumulation;
- BF16/FP16 tensor-core QK and PV where parity permits;
- causal masking and base-offset continuation in the same kernel;
- no broadcast K/V buffers.

Target O(N × hidden + tile scratch) memory and one implementation for fresh and continuation prefill.

#### 2.2 Remove the shared-memory context cap

Delete the full-context shared-score fallback. All decode and continuation paths must use tiled online softmax or split-KV partial/combine. Context should then be bounded by configured memory, not per-block shared memory.

#### 2.3 Dynamic split planner

Choose decode splits from `(batch, head count, context length, SM count)`:

- fewer splits at short context/high B;
- enough `(B × NQ × splits)` CTAs to fill 48 SMs;
- cap combine traffic;
- cache graph variants by split class, not exact position.

Benchmark p50/p95 latency at B=1/4/8/16 and context 1K/4K/16K/64K.

### Phase 3 — GDN vector/state pipeline

Use FlashQLA as the performance reference (Qwen reports roughly 2–3× forward speedup over FLA in supported scenarios): <https://github.com/QwenLM/FlashQLA>.

#### 3.1 Prefill

- Sweep chunk 64/128/256 with GB10-specific occupancy measurements.
- Fuse decay/beta preparation, normalization and WY/UT staging where it reduces HBM traffic.
- Split long sequences across CTAs and merge carried states to fill the GPU at B=1.
- Evaluate CuTe-DSL/FlashInfer algorithm structure but keep fucina's standalone compiled CUDA deployment.

#### 3.2 Decode

- Retain vectorized BF16 state load/store.
- Fuse conv → split → L2 norm → decay/beta where register pressure permits.
- Fuse GDN output normalization/gate before the output projection.
- Keep recurrence math FP32; state storage BF16 remains parity-gated.

**Acceptance:** long-context oracle parity, state snapshot parity, graph on/off and batch independence all pass; at least 10% GDN phase reduction or the change is reverted.

### Phase 4 — vector/projection and MoE management

1. Preserve current Q4_K packed mixer and grouped NVFP4 expert paths as baselines.
2. Add shape-class autotuning at load time for B=1/2/4/8/16; cache the choice.
3. Fuse router logits, top-k, counting sort and active-expert list generation where deterministic ordering is preserved.
4. Reuse activation quantization across compatible projections; avoid repeated F32→quantized passes.
5. Investigate fused grouped gate/up/SiLU/down only after node tracing proves intermediate traffic, not weights, is limiting.
6. Keep per-row scales; never use batch-global activation scales because they violate row independence.

### Phase 5 — token prefill/decode scheduling

- Merge prefill chunks and active decode rows into one execution plan when shapes permit.
- Reserve a decode latency budget per scheduler pass; adapt prefill chunk width to current B and queue age.
- Use state snapshots for exact multi-turn Qwen reuse and radix/paged KV for ordinary transformers.
- Add cancellation checkpoints between layers/chunks without synchronizing the device every token.
- Keep speculative decode off sparse MoE until a GDN/conv sandbox makes rejected drafts state-safe and measured acceptance pays for expert work.

### Phase 6 — benchmark and release gates

Every optimization must run:

1. Build + host detection.
2. Standalone kernel numerical tests.
3. Qwen dense and MoE oracle parity.
4. Graph on/off and B=1-vs-batch row independence.
5. Diverse-prompt burst admission.
6. State save/restore across slots.
7. 1K/4K and new 16K/64K long-context gates.
8. Sampling reproducibility in solo, homogeneous batch and mixed batch.
9. OOM/fault-injection tests for every lazy allocation.
10. Benchmarks: TTFT, inter-token latency, aggregate tok/s, peak memory and bytes/token.

Primary scoreboard:

- B=1 decode tok/s at 1K/4K/16K context.
- Aggregate decode tok/s at B=4/8/16 with diverse prompts/routing.
- Cold TTFT at 128/512/2K/8K/32K.
- Warm turn TTFT with state/prefix reuse.
- Peak/reserved memory per configured slot and per context block.
- p95 queue + prefill + decode latency under mixed arrivals.

## Definition of done

The GB10 server is ready when it:

- honors all sampling and structured-output semantics;
- has no compile-time-capacity allocation hidden behind a smaller runtime admission limit;
- guarantees admitted sequence memory through its configured context;
- performs FULL attention without O(N²) score storage or full-context shared memory;
- keeps Qwen hybrid state isolated, restorable, and race-free;
- exposes enough phase/memory metrics to explain regressions;
- passes dense/MoE, long-context, stochastic and concurrency gates;
- improves measured end-to-end metrics, not only isolated kernel throughput.
