# NVIDIA GB10 competitive assessment: fucina main vs DS4 `80ebbc39` vs vLLM `5f8e73cb`

**Assessment type:** analysis only  
**fucina revision:** `900de794cc66f2bdf7f9ea2c761ee74ca0fb0c22` (`main`)  
**DS4 revision:** `80ebbc39`  
**vLLM revision:** `5f8e73cb8b8d41f7a2a5168cddf5b772888fa991`  
**Hardware in the contemporaneous Qwen evidence:** NVIDIA GB10, `sm_121a`, CUDA 13, 128 GiB coherent LPDDR5X  
**Scope:** Qwen3.5 dense 9B and MoE 35B-A3B; Gemma models actually represented in fucina source/docs; DS4 only as technique evidence because it is a DeepSeek-V4-specific runtime.

---

## A. Executive verdict and confidence

### Verdict

1. **For the two measured Qwen3.5 checkpoints, fucina is the best GB10 throughput engine in this evidence set, but not the best burst-latency engine.** On the valid contemporaneous served-throughput cells, fucina wins all five MoE concurrency cells and dense N=2/4/8/16; it loses dense N=32 by 15.9%. Counting the valid N=1 `single_short.decode_tps` measurements produces the repository's **11/12 aggregate-throughput** position. This is not an 11/12 TTFT claim.
2. **vLLM is the stronger general serving system.** It has a unified token-budget scheduler, preemption/recompute, hybrid-state cache management, broad CUDA-graph dispatch, structured output, multiple speculative methods, and TP/PP/distributed support. Its decisive current GB10 advantage is synchronized short-prompt burst admission: at MoE N=32, 312/316 ms median/p95 TTFT versus fucina 670/722 ms; at dense N=32, about 213/218 ms versus 479/486 ms.
3. **fucina's burst TTFT is the highest-value measured Qwen deficit if interactive synchronized bursts are a product SLO.** It is not the highest-value issue for every deployment: fucina already wins long MoE prefill (4.367 s versus 5.837 s), all measured MoE aggregate cells, and most dense aggregate cells. The correct next work is profile-driven short clean-prefix admission, not another broad kernel rewrite.
4. **There is no credible byte-identical path from the current dense Q4_K mixer to 12/12 today.** D32/D32B established that the dense-N32 residual is a kernel-class deficit: fucina's occupancy-saturated Q4_K/dp4a mixer versus vLLM's tensor-core FP8 GEMM. The obvious tensor-core replacement changes reduction order and violates the current byte-identity contract. The recommended decision is to **accept dense N=32 as a documented exact-mode exception** while maintaining a floor and continuing only profile-proven, exact-preserving work. A separate tensor-core mode may be explored only under a new, explicitly qualified output contract; it is not an “exact” completion of 12/12.
5. **Gemma cannot be competitively ranked from the current evidence.** fucina has substantial current-source support—dense Gemma-4, native NVFP4, paged continuous batching, prefix reuse, and current-source batched MTP—but the available performance claims are historical/local, not a fresh same-hardware, same-checkpoint vLLM comparison. Some documentation is stale: it says batched MTP is missing, while current source implements a batched paged MTP drafter and lossless batched verification. E4B remains a separate engine/product path and does not have equivalent server evidence.
6. **DS4 is not a third benchmark competitor here.** It targets DeepSeek V4 with different model geometry, quantization, kernels, platforms, and a mostly serialized serving loop. Its useful contributions are bounded policy mechanisms: expert-route profiling and capacity replay, durable checkpoint scoring, bounded distributed-prefill metadata, and fixed-capacity reusable buffers. Its model numbers, ROCm/Strix/Metal kernels, HMM behavior, SSD streaming results, and whole-session “disk KV” are not apples-to-apples Qwen/Gemma evidence.

### Confidence

| Conclusion | Confidence | Basis / limitation |
|---|---:|---|
| Qwen aggregate ranking at N=2..32 | **High** | Fresh serialized fucina/vLLM runs, fixed protocol, raw JSON, dense repeat. |
| Qwen burst-TTFT ranking | **High** | Raw per-request TTFT arrays; large gaps at N=32. |
| Long-prefill MoE ranking | **Medium-high** | One fresh long probe per engine; no p95 across repeats. |
| Dense N=32 root cause | **High** | D32/D32B ncu, microbench, bit-identity hashes, served repeat. |
| Global “fucina deterministic” claim | **Low unless scoped** | Dense/reference gates are strong, but MoE grouped-GEMM B>1 has a known pre-existing run-to-run self-consistency failure. |
| Quality-normalized MoE speed ranking | **Low** | Raw fucina MoE samples include obvious repetition/wrong answers; no quality-normalized checkpoint parity or oracle score accompanies throughput. |
| Gemma performance ranking | **Low / unavailable** | No fresh raw fucina-vLLM Gemma matrix. |
| DS4 technique transfer | **Medium** for policy, **low** for performance | Source mechanisms are clear; performance is model/platform-specific. |

### Evidence hygiene warnings

- **Two vLLM N=1 cold-start artifacts must be excluded from competitive claims.** Dense concurrency N=1 reports 6.534 s TTFT and 10.24 aggregate tok/s even though `single_short` immediately around it is 65.3 ms TTFT and 21.67 decode tok/s. MoE concurrency N=1 similarly reports 6.341 s and 14.02 tok/s versus `single_short` 83.9 ms and 46.65 tok/s. These are cold/startup anomalies, not representative single-stream service. For the “11/12” count, use the valid `single_short.decode_tps` N=1 evidence, then aggregate N=2..32.
- **The fucina MoE outputs are suspiciously low quality.** Examples in `fucina-q35moe-d32b.json` include repeated “capital of France” loops, repeated “photosynthesis” fragments, `137 + 265 = 400`, and an N=8 prime-number sample degenerating into repeated `20`. The benchmark still measures the serving work done by those exact checkpoints/configurations, but it does **not** establish quality-normalized parity. Do not advertise the throughput lead as a same-quality lead until exact checkpoint/quantization provenance and oracle quality are normalized.
- **Determinism is scoped, not global.** Dense state/graph/reference gates and many single-sequence paths are bit-identical. The MoE FP8 checkpoint's default transformed NVFP4 grouped-GEMM path at B>1 has a known run-to-run self-consistency failure on GB10, present before the latest TTFT work. Oracle token parity can pass while graph-on/off or self-chain bytes differ. “Exact” below means exact for the named path and gate, never a blanket promise for all concurrent MoE serving.

---

## B. Qwen3.5 scorecard and raw evidence

### B.1 Protocol and comparability

`benchmark-evidence/PROTOCOL.md:1-63` fixes the machine, checkpoint identity, server configuration, prompt set, concurrency `1,2,4,8,16,32`, 128 generated tokens, and synchronized-burst measurement. `agg_decode_tps` is completion tokens divided by the whole burst wall time, so admission and TTFT are in the denominator. The 2026-07-18 runs were serialized between engines with the GPU quiescent (`benchmark-evidence/results/2026-07-18-d32b/README.md:1-15`).

The scorecard uses:

- fucina dense: `fucina-q35dense-d32b.json` plus `fucina-q35dense-d32b-rep2.json`;
- fucina MoE: `fucina-q35moe-d32b.json`;
- vLLM dense/MoE: `vllm-q35dense-fresh.json`, `vllm-q35moe-fresh.json`;
- frozen protection references only as regression gates, not as substitutes for the fresh vLLM run.

### B.2 Served aggregate throughput: raw scorecard

| Model / N | fucina agg tok/s | vLLM agg tok/s | fucina delta | Winner |
|---|---:|---:|---:|---|
| Dense 9B, valid N=1 `single_short.decode_tps` | 34.69 | 21.67 | +60.1% | fucina |
| Dense 9B, N=2 | 59.27 | 44.23 | +34.0% | fucina |
| Dense 9B, N=4 | 117.27 | 85.75 | +36.8% | fucina |
| Dense 9B, N=8 | 204.61 | 164.40 | +24.5% | fucina |
| Dense 9B, N=16 | 313.1 | 280.8 | +11.5% | fucina |
| Dense 9B, N=32 | **438.8** | **521.8** | **−15.9%** | vLLM |
| MoE 35B-A3B, valid N=1 `single_short.decode_tps` | 61.01 | 46.65 | +30.8% | fucina |
| MoE, N=2 | 101.39 | 74.07 | +36.9% | fucina |
| MoE, N=4 | 133.98 | 111.74 | +19.9% | fucina |
| MoE, N=8 | 229.75 | 155.21 | +48.0% | fucina |
| MoE, N=16 | 320.1 | 207.2 | +54.5% | fucina |
| MoE, N=32 | 472.4 | 321.3 | +47.0% | fucina |

Dense repeat evidence is adverse to wishful interpretation: fucina N=32 repeated at 433.3 tok/s, so the 438.8 result is not a one-off low vLLM/high fucina inversion. Dense N=32 is the sole valid aggregate loss.

### B.3 TTFT, prefill, and interactivity

| Workload | fucina median / p95 TTFT | vLLM median / p95 TTFT | Interpretation |
|---|---:|---:|---|
| Dense N=32 short burst | ~479 / 486 ms | ~213 / 218 ms | vLLM ≈2.2× lower TTFT. |
| MoE N=32 short burst | **670 / 722 ms** | **312 / 316 ms** | vLLM ≈2.1× lower median and ≈2.3× lower p95. |
| MoE ~3,500-token prompt | 4,367 ms | 5,837 ms | fucina 25.2% lower TTFT on this one long probe. |
| Dense ~3,500-token prompt | 3,611 ms | 5,720 ms | fucina lower on this one long probe. |

The short-burst result and long-prompt result are not contradictory. fucina's layer-major, weight-amortized long prefill is strong; its short multi-sequence admission still pays shape padding, per-sequence launch work, and a less general packing/scheduling path. vLLM's unified token budget and packed hybrid-state execution are especially effective when 32 short requests arrive together.

**Priority judgment:** burst TTFT is the highest-value *measured Qwen gap* for interactive agents because users feel the first-token delay and the MoE gap is >350 ms at median and >400 ms at p95. It does not justify sacrificing fucina's long-prefill and aggregate wins. Every change must therefore be default-off until it beats the incumbent dual-sided gate.

### B.4 Capability scorecard (5 = strongest; “?” = not measured)

| Dimension | fucina main | vLLM `5f8e73cb` | DS4 `80ebbc39` | Raw basis |
|---|---:|---:|---:|---|
| Dense Qwen served throughput | **4.5** | 4.0 | N/A | fucina wins N=1..16; vLLM wins N=32. |
| MoE served throughput | **5.0** | 3.5 | N/A | fucina wins every valid cell, but quality parity is unproven. |
| Short-burst TTFT | 2.0 | **5.0** | N/A | Fresh N=32 raw arrays. |
| Long-prompt prefill | **4.5** | 3.5 | N/A | One fresh 3.5k probe/model; needs repeats. |
| Continuous batching/scheduling breadth | 3.5 | **5.0** | 1.0 | fucina model-specific scheduler; vLLM unified priorities/preemption/spec/prefix; DS4 primarily one-session loop. |
| Hybrid GDN state handling | **4.0** | **4.5** | N/A | Both have explicit recurrent state; vLLM integrates it into generic hybrid cache/scheduler machinery. |
| Prefix/session reuse | **4.5** | 4.5 | 3.0 | fucina persists exact KV+GDN+conv; vLLM has paged prefix manager; DS4 persists whole-session checkpoints keyed by rendered prefix. |
| Speculative decode | 3.5 | **5.0** | 2.5 | fucina prompt lookup/MTP/DFlash primitives; vLLM broad spec stack and Qwen/Gemma integrations; DS4 MTP state exists but serving is narrow. |
| Structured output | 2.5 | **5.0** | 1.0 | vLLM scheduler carries grammar state and masks spec tokens; fucina lacks equivalent broad documented coverage. |
| Distributed serving | 2.5 | **5.0** | 2.5 technique-only | fucina phase-E prototype; vLLM TP/PP/DP ecosystem; DS4 has bounded layer-pipeline prefill but different model/platform. |
| Memory/residency controls | **4.5** | 4.5 | 4.0 technique-only | fucina owned-byte/physical-headroom accounting and expert SSD mode; vLLM paged block manager/offload; DS4 SSD/HMM/arena paths. No fresh GB10 memory table. |
| Determinism | 3.0 scoped | ? | ? | fucina dense/single-path gates strong; concurrent MoE B>1 caveat. No equivalent fresh repeat matrix for vLLM/DS4. |
| Startup/model load | ? | ? | ? | Source mechanisms exist; no comparable raw startup/warmup timings. |
| Operational breadth | 3.0 | **5.0** | 2.0 | fucina is compact but model-specific/env-heavy; vLLM is broad but dependency-heavy; DS4 is narrow and model-specialized. |

### B.5 Dense N=32: exact-gate decision

D32/D32B's evidence is unusually strong:

- fucina's mixer reads approximately 3.65 GiB of Q4_K weights versus roughly 6.5 GiB FP8 for vLLM; the loss is not explained by reading more bytes;
- ncu shows the Q4_K warp-per-row kernel dominated by long-scoreboard latency, not DRAM bandwidth or the dp4a ALU chain (`docs/qwen35-d32b.md:15-40`);
- BIGCHUNK=12/MINBLK=4 improved the cell while preserving all recorded stream hashes, then saturated the legal occupancy lever;
- DPSPLIT was only +0.6%; deeper PIPE variants regressed/spilled; further legal variants were measured and rejected (`docs/qwen35-d32b.md:42-100` and continuation);
- the remaining competitor advantage is tensor-core FP8 GEMM, whose reduction order cannot satisfy the current byte-identical Q4_K output gate.

**Decision:** accept dense N=32 at 438.8 tok/s as the current exact-mode result, with these conditions:

1. protection floor: no merge may reduce the repeat median below 95% of the frozen/current result;
2. continue to publish it as a loss, not “within noise”;
3. do not revive D32/D32B measured-dead variants;
4. a tensor-core experiment must be a separately named, default-off mode with a new deterministic-reference contract and quality evaluation; it cannot be counted as exact relative to the incumbent;
5. revisit only if a profile reveals a new non-mixer bottleneck or CUDA/CUTLASS introduces a primitive that can reproduce the established accumulation order.

This is more credible than promising a nonexistent exact path to 12/12.

---

## C. Gemma scorecard, current-source correction, and required evidence

### C.1 What fucina actually supports now

Current source is ahead of parts of the documentation:

- Dense Gemma-4 has paged multi-sequence decode, per-B CUDA graphs, single-pass/fallback prefill, prefix reuse, native NVFP4 safetensors, and Q4_0/Q8_0 GGUF paths (`cuda/gemma4_kernels.cu:11954-12485`).
- **Batched MTP is present in current source.** `cuda/gemma4_kernels.cu:13983-14160` implements a B-row paged MTP drafter; `cuda/gemma4_kernels.cu:12532-12747` uses batched drafting and target verification; `internal/engine/cuda/bridge.go:1099-1200,1594-1638` exposes lossless `StepBatchSpec`; the scheduler invokes the speculative batch interface at `internal/server/batch/scheduler.go:1340-1430`.
- Therefore statements in `docs/continuous-batching.md` that batched MTP is absent are stale and must not be repeated as current fact. The real gap is **fresh end-to-end evidence, default-product enablement, and documentation alignment**, not absence of the code.
- Gemma-4-E4B exists as a separate runtime with PLE/KV-sharing and an assistant loader (`internal/engine/e4b/bridge.go:196-204`), but it does not have the same demonstrated server/API integration or fresh comparative benchmark.
- Diffusion Gemma is a separate experimental generation path and is not comparable to ordinary autoregressive vLLM serving; exclude it from the primary score.

### C.2 vLLM Gemma breadth

At `5f8e73cb`, vLLM's Gemma4 model has dense and MoE components, explicit KV-sharing/fast-prefill paths, Eagle3/MTP integration, LoRA, PP, and the generic scheduler/structured-output stack (`/tmp/vllm/vllm/model_executor/models/gemma4.py:218-368,459-501,790-963,1108-1295,1508-1575`; `/tmp/vllm/vllm/model_executor/models/gemma4_mtp.py`; `/tmp/vllm/vllm/v1/spec_decode/gemma4.py`). This is source capability evidence, not proof that each path is fast on GB10.

### C.3 Gemma scorecard

| Dimension | fucina main | vLLM `5f8e73cb` | Confidence / raw evidence |
|---|---:|---:|---|
| Dense Gemma-4 basic generation | 4.5 | 4.5 | Both source-complete; fucina local tests/historical claims, no fresh head-to-head. |
| Native NVFP4 GB10 specialization | **5.0** | 4.0 | fucina single-store NVFP4 and fused decode kernels are explicit; no same-checkpoint performance matrix. |
| Continuous batching | 4.0 | **5.0** | fucina source has paged batching/graphs; vLLM has generic production scheduler. User-facing defaults/docs need reconciliation. |
| Batched MTP/speculation | 4.0 | **5.0** | Both have source support. fucina current source corrects stale docs but lacks fresh served evidence. |
| Prefix reuse | 4.0 | **4.5** | fucina radix/full-block prefix and paged tables; vLLM generic KV/prefix manager. |
| E4B/PLE/KV sharing | 3.0 | **4.5** | fucina standalone E4B engine/tests; vLLM integrated Gemma4 self/cross/fast-prefill model path. |
| Structured outputs/API breadth | 2.5 | **5.0** | vLLM generic grammar and OpenAI ecosystem; fucina narrower. |
| Distributed/LoRA/multimodal breadth | 2.0 | **5.0** | vLLM source protocols; fucina focuses on local GB10 kernels. |
| Performance | ? | ? | **No fresh comparable raw evidence.** |
| Quality-normalized parity | ? | ? | No current same-checkpoint oracle matrix. |
| Startup/memory | ? | ? | No comparable raw measurements. |

**Gemma verdict:** vLLM leads capability breadth; fucina may lead selected native-NVFP4 GB10 kernels, but that is a hypothesis until measured. No honest overall performance winner can be named.

### C.4 Exact benchmark matrix required before any Gemma claim

Run all cells on the same quiescent GB10, with serialized engine phases and three independent server starts:

1. **Artifacts:**
   - dense Gemma-4 12B exact source revision and SHA-256;
   - the fucina-supported native-NVFP4 artifact, with exact source checkpoint and conversion provenance;
   - E4B exact artifact and assistant hash, only if both engines can load semantically equivalent weights;
   - label GGUF-vs-HF or different quantization as a *format/system comparison*, never kernel parity.
2. **Modes:** fucina/vLLM × plain decode/MTP; prefix cold/warm; continuous batch on; E4B fast-prefill/KV-sharing on and off where supported.
3. **Traffic:** valid N=1 single request plus synchronized N=2/4/8/16/32; 16+ diverse short prompts; one 3,500-token probe repeated at least five times; a 4k/16k long-context matrix; 128 generated tokens; back-to-back bursts; mixed long-prefill plus active decode.
4. **Metrics:** TTFT p50/p95/p99, inter-token latency p50/p95/p99, aggregate completion tok/s, per-stream tok/s, request failures, preempt/recompute count, accepted draft length, prefix hit blocks, startup-to-ready, first-request warmup, steady RSS/device-owned bytes, and physical available memory.
5. **Quality/exactness:** greedy token hash against a fixed BF16/HF oracle corpus; logit error bounds; repetition, arithmetic, code, multilingual, and long-context evaluations; repeated-run hashes by batch shape; sampled-distribution test separately. Throughput is reported only beside the quality result.
6. **Acceptance:** no performance claim from a single run; coefficient of variation and all raw arrays retained; any vLLM N=1 warmup artifact rerun rather than averaged into the result.

---

## D. File:line architecture comparison

### D.1 Scheduling and batching

**fucina**

- `internal/server/batch/scheduler.go:1-220` defines a compact single-goroutine engine contract with optional speculative, chunked-prefill, fused-prefill, batched-admit, prefix, and state-snapshot interfaces.
- `internal/server/batch/scheduler.go:879-881` uses fixed default coalescing values of 3 ms window, 12 ms quiet, and 150 ms maximum; these are env-overridable.
- `internal/server/batch/scheduler.go:904-1009` handles idle and busy admission; `:920-934` routes eligible bursts to multi-sequence admission.
- `internal/server/batch/scheduler.go:1014-1084` bounds batched admission to 4,096 prompt tokens and free slots.
- `internal/server/batch/scheduler.go:1095-1230` interleaves and, where supported, fuses prefill with decode.
- Strength: understandable, bounded, model-aware behavior. Weakness: more special-case product logic and less general policy than vLLM; Qwen batched admission groups only a leading run of short prompts.

**vLLM**

- `/tmp/vllm/vllm/v1/core/sched/scheduler.py:448-626` schedules running requests against a global token budget and allocates cache blocks, preempting when needed.
- `:673-1052` admits waiting requests under the same budget, including chunked prefills and blocked queues.
- `:1190-1212` implements preemption by freeing state and requeueing for recomputation; `:1526-1548` integrates grammar masks.
- Strength: one generic policy for decode, chunked prefill, speculation, priorities, preemption, remote KV, and structured output. Cost: more dependency and operational complexity.

**DS4**

- `ds4.c:10300-10500` owns one fixed release graph with persistent single-session tensors plus a separate prefill set.
- It does not provide comparable multi-tenant continuous serving. Its bounded fixed buffers are useful; its serving topology is not a template to copy.

### D.2 Qwen3.5 hybrid GDN and attention state

**fucina**

- `cuda/qwen35_kernels.cuh:530-599` implements token-sequential GDN recurrence, updating per-slot recurrent state and conv rings.
- `cuda/qwen35_kernels.cuh:806-903` implements the chunk scan: WMMA prefix products, lower-triangular solve, state interaction, output, and final state.
- `cuda/qwen35_runtime.cuh:806-887` runs flattened projection GEMMs but still has per-sequence loops for conv/ring handling (`:819-827`) and full attention (`:831-845`), then a multi-sequence GDN scan (`:848-864`).
- `cuda/qwen35_backend.cuh:867-1013` owns per-slot full-attention KV, BF16 GDN state, and conv rings, accounted as physical engine memory.
- `docs/session-persistence.md:18-92` and `cuda/qwen35_state.cuh` persist/restore KV, GDN state, and conv rings together.

**vLLM**

- `/tmp/vllm/vllm/model_executor/models/qwen3_5.py:280-398,564-650` exposes dense/MoE Qwen3.5 under generic PP/Eagle/hybrid model protocols.
- `/tmp/vllm/vllm/v1/attention/backends/gdn_attn.py` represents GDN as a hybrid attention backend with per-request state indices and accepted-token handling.
- `/tmp/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` selects Blackwell/CUDA-13 prefill implementations: FlashInfer under supported constraints, CuteDSL opt-in, and Triton fallback.
- `/tmp/vllm/vllm/v1/worker/gpu/model_states/mamba_hybrid.py` binds recurrent state to the generic hybrid cache lifecycle.
- Advantage: packed, scheduler-integrated hybrid state and multiple maintained kernel backends. Trade-off: less model-specific simplicity and no byte-identity claim against fucina.

**Exact-clean-prefix GDN limit:** A fresh slot has exactly zero GDN matrix and empty conv ring. The first chunk can therefore omit only operations whose input is provably zero and can shrink only all-zero padded work if byte tests prove identical. This helps a short one-chunk prompt; it does **not** eliminate recurrence, does not apply unchanged to a restored/warm prefix, and saves only the first chunk of a long prompt. It cannot explain or solve dense N=32 steady decode.

### D.3 Weight formats and kernels

- fucina's Qwen default transforms dense mixer/expert paths to compact Q4_K/NVFP4 and uses model-specific GEMV/grouped GEMM (`cuda/qwen35_backend.cuh:1020-1110`; `cuda/gemma4_kernels.cu:9366-9680`). This wins many LPDDR-bound shapes.
- vLLM uses ecosystem FP8/tensor-core GEMMs and maintained FlashInfer/CUTLASS/Triton backends. Dense N=32 demonstrates the tensor-core advantage at a sufficiently wide batch.
- DS4's IQ2/Q2/Q4 and ROCm/Metal kernels are bound to DeepSeek-V4 geometry and cannot be treated as Qwen/Gemma kernel evidence.

### D.4 KV/cache allocation, memory, residency, and startup

**fucina**

- `cuda/qwen35_backend.cuh:867-1013` computes owned bytes and uses physical-availability headroom rather than treating reclaimable unified-memory file cache as free.
- `cuda/qwen35_backend.cuh:457-630` supports optional SSD expert backing, global slots, checksums, and deterministic residency-plan seeding; SSD mode incurs host count readback/synchronization and disables the normal all-resident graph assumptions.
- Gemma's paged pools and per-sequence block tables are at `cuda/gemma4_kernels.cu:4015-4060,11954-12485`.
- Startup can include load-time format conversion and graph capture. No fresh comparable startup table was retained, so no winner is claimed.

**vLLM**

- `/tmp/vllm/vllm/v1/core/kv_cache_manager.py` owns block allocation, prefix hits, freeing, and request block lifetimes; the scheduler can preempt and recompute.
- CUDA graphs are shape-dispatched by `/tmp/vllm/vllm/v1/cudagraph_dispatcher.py` rather than per-model scheduler switches.
- vLLM has broader offload/distributed connectors, but no fresh GB10 memory/residency numbers in this evidence bundle.

**DS4**

- `ds4_cuda.cu:870-950` implements CUDA HMM advice/prefetch; `:1000-1230` has pinned staging and direct reads; `:1190-1230` sets a 96 GiB default local weight-cache limit.
- These are useful experiments, not transferable performance claims. fucina should retain its stricter physical-headroom accounting if it tries HMM.

### D.5 Prefix caching and persistence

- fucina's Gemma radix cache adopts immutable full blocks; Qwen hybrid sessions instead snapshot exact recurrent/KV state. This architecture correctly avoids pretending recurrent GDN state is a truncatable KV block.
- vLLM uses generic paged prefixes plus hybrid-state cache metadata; preemption may discard and recompute state.
- DS4 `ds4_kvstore.c:1190-1350` finds the longest exact byte prefix but restores the exact stored token history before tokenizing only the suffix. Its durable scoring at `ds4_kvstore.c:500-620` decays hits and scores useful tokens per byte. This is checkpoint persistence, not token-level paged KV sharing.

### D.6 Speculation, structured output, and distributed execution

- fucina Gemma current source has batched MTP drafting and lossless target verification (`cuda/gemma4_kernels.cu:12532-12747,13983-14160`; bridge `:1099-1200,1594-1638`). Qwen DFlash primitives include recurrent-state rollback but the real draft checkpoint path remains gated.
- vLLM model classes advertise Eagle/PP and integrate accepted-token hybrid state; generic scheduler grammar validation filters speculative tokens (`scheduler.py:1235-1243,1691-1708,2018-2050`).
- fucina phase-E and DS4 both have distributed-prefill prototypes. DS4 `ds4_distributed.c:3070-3520` is specifically useful for fixed slot rings, ACK-only intermediate chunks, bounded flow windows, and prefix/result hashes. vLLM remains much broader operationally.

---

## E. Quantified improvement options

Ranges below are **evidence-bounded, not promises**. A `0` lower bound is intentional where no direct experiment exists. “Exact” means byte identity against the named incumbent path and state, not global MoE determinism.

### E1. Profile and fuse Qwen multi-sequence conv/ring launches

- **Bottleneck:** short Qwen burst prefill still loops sequences for convolution and ring update in each GDN layer (`qwen35_runtime.cuh:819-827`) even though projection and GDN scan work are multi-sequence.
- **Mechanism:** add one metadata-driven kernel over flattened `(sequence,row,channel)` conv work and one deterministic ring-update kernel keyed by slot and exact position. Preserve each row's multiply/add order; never share state between slots.
- **Benefit range:** **0 to the measured conv/ring launch share** of N=32 admission. No percentage is claimed before nsys; acceptance requires at least 5% median-TTFT or 8% p95-TTFT reduction. The mathematical upper bound is removal of that phase only.
- **Cost:** M. **Determinism:** exact if state/output bytes match standalone for all length/slot cases.
- **Dependencies:** fresh nsys with graph-node tracing; heterogeneous-length metadata; state snapshot comparator.
- **Falsify/stop:** conv/ring <5% of GPU admission time, or fused bytes differ.
- **Rollback:** `FUCINA_QWEN35_FUSED_CONV=0`; delete path if it misses gate.

### E2. Exact clean-prefix, short-tile GDN specialization

- **Bottleneck:** a short fresh prompt enters a 64-token chunk scan with zero initial state and padded rows.
- **Mechanism:** template first-chunk kernels by deterministic length class (for example 16/32/48/64), and omit only `K·S`/`Q·S` terms proven zero for a fresh slot. Keep causal order for real rows and final state exactly equal to the 64-row kernel. Dispatch from prompt length and `state_is_clean`, never from timings.
- **Benefit range:** **0 to the first-chunk GDN fraction** of short admission; **approximately zero for long prompts** beyond amortizing one of many chunks, and zero for restored/warm prefixes. Earlier profiling put the pre-F2 GDN scan at 23.9% of admission kernel time, but F2 already removed sequence serialization, so 23.9% is an obsolete ceiling, not a forecast. Require ≥5% N=32 TTFT improvement.
- **Cost:** M. **Determinism:** exact only after bitwise output, GDN-state, conv-ring, KV, first-token, and continuation gates for every length 1..64 and mixed batch.
- **Dependencies:** explicit clean-state bit; signed-zero/NaN adversarial vectors; no use on snapshots.
- **Falsify/stop:** any byte mismatch or current GDN share below 5%.
- **Rollback:** default-off dispatch flag and incumbent CS=64 kernel retained.

### E3. Pack/group Qwen full-attention prefill calls across sequences

- **Bottleneck:** `qwen35_runtime.cuh:831-845` invokes full attention per sequence while vLLM uses packed request metadata.
- **Mechanism:** first profile full-attention share. If material, use grouped/strided batched calls with per-sequence causal bounds and independent KV destinations; preserve the exact existing per-sequence inner reduction order rather than switching algorithm silently.
- **Benefit range:** **0 to the measured full-attention phase share**. Require ≥5% TTFT improvement; no aggregate/decode claim.
- **Cost:** L. **Determinism:** qualified until byte/state gate passes; grouped library algorithms may change reduction order.
- **Dependencies:** ragged offsets, grouped GEMM algorithm pinning, all prompt lengths, cancellation tests.
- **Falsify/stop:** phase <5%, library cannot pin exact math, or any token/state mismatch.
- **Rollback:** environment gate; incumbent per-sequence loop remains authoritative.

### E4. Scheduler burst-shape telemetry and fixed deterministic bucketing

- **Bottleneck:** fucina MoE N=32 TTFT spans roughly 595–727 ms while vLLM is tightly clustered near 304–316 ms; some rows may be split into later admissions. Existing fixed 3/12/150 ms policy lacks per-burst shape evidence.
- **Mechanism:** add counters only: arrival-to-batch delay, batch count, rows/tokens per admission, engine time per admission, first-decode delay. Sweep existing fixed env knobs; if needed, bucket a leading run by prompt-length class and a fixed token budget. Do not use latency feedback to choose arithmetic policy.
- **Benefit range:** **0 to the measured split-admission component**. The raw MoE p95-minus-median spread is 52 ms; eliminating spread cannot close the 358 ms median gap by itself. Require p95 reduction without >5 ms N=1 cost.
- **Cost:** S telemetry; M policy. **Determinism:** scheduling-deterministic for a fixed arrival trace, but concurrent MoE arithmetic remains subject to its existing B>1 caveat.
- **Dependencies:** replayable timestamp trace; busy and idle bursts; no new wait on lone requests.
- **Falsify/stop:** all 32 rows already enter one admission, or wider quiet worsens p95/aggregate.
- **Rollback:** existing env defaults and policy.

### E5. Resolve the MoE B>1 determinism contract

- **Bottleneck:** a known grouped-GEMM run-to-run self-consistency failure prevents a global exact-serving claim even when oracle tokens pass.
- **Mechanism:** isolate whether nondeterminism originates in CUTLASS grouped NVFP4 GEMM, route grouping, scratch reuse, or graph replay. Build a fixed-order reference for B=2/4/8/16/32 and compare every intermediate. If CUTLASS cannot be made stable, expose a separately benchmarked deterministic fallback rather than relabeling tolerance as exact.
- **Benefit range:** correctness **FAIL→PASS**; performance range **unknown and must be measured**. No positive throughput benefit is claimed. Existing Q4_K/NVFP4 comparisons are not sufficient to price the fallback.
- **Cost:** L. **Determinism:** this is the determinism gate.
- **Dependencies:** intermediate hashes, racecheck/sanitizer where feasible, repeated graph-on/off runs, exact routing inputs.
- **Falsify/stop:** if only a slower fallback is possible, ship it as opt-in “strict” and keep default claim scoped.
- **Rollback:** default grouped path unchanged; strict mode explicit.

### E6. Merge `feat/ds4-expert-policy` only after real held-out SSD proof

- **Bottleneck:** SSD expert mode has no workload-derived capacity/hotlist policy; misses create SSD and unified-memory contention.
- **Mechanism:** the branch records bounded route events using the host counts already copied by SSD mode, atomically emits deterministic JSON, replays global LRU capacities, and emits the existing `fucina-expert-residency-v1` loader format. Ranking is fixed by selection events, rows, layer, expert—not timing.
- **Benefit range:** **0 to 100% of latency currently attributable to avoidable SSD misses**; zero for full-resident serving. This is a bound, not an estimate. No merge claim until actual miss-attributable latency and held-out p95/p99 are measured.
- **Cost:** code M; validation M. **Determinism:** policy bytes deterministic; inference arithmetic unchanged when profiling; concurrent MoE caveat remains.
- **Dependencies/pre-merge requirements:** see section F.2; representative and held-out traces; SSD endurance/flush review.
- **Falsify/stop:** no held-out p95/p99 gain, overfit hotlist, dropped trace too large, or default-off path changes any CUDA graph/allocation.
- **Rollback:** unset `FUCINA_EXPERT_PROFILE_OUT` and `FUCINA_EXPERT_RESIDENCY_PLAN`; no-plan loader behavior remains.

### E7. Adopt DS4-style durable checkpoint eviction scoring for fucina sessions

- **Bottleneck:** a byte-budgeted disk session cache can retain large, cold, superseded checkpoints while evicting smaller reusable prefixes.
- **Mechanism:** score exact fucina snapshots by decayed hits × reusable tokens / physical bytes, with reason weights and superseded-prefix penalties, following the policy shape at `ds4_kvstore.c:500-620`. Retain fucina's stronger model/quant/state schema checks.
- **Benefit range:** **0 to 100% of avoidable re-prefill time on cache evictions**; zero on no-reuse traffic. Measure hit-token-hours and TTFT saved, not raw file-hit count.
- **Cost:** M. **Determinism:** exact; eviction policy does not alter a loaded snapshot's bytes.
- **Dependencies:** schema version, monotonic access metadata, crash-safe atomic updates, quota tests.
- **Falsify/stop:** score does not beat LRU on held-out replay or metadata writes dominate.
- **Rollback:** policy enum back to LRU; snapshot format unchanged.

### E8. Gemma current-source benchmark/protection gate and docs repair

- **Bottleneck:** no fresh competitive evidence; stale docs incorrectly say batch MTP is absent.
- **Mechanism:** implement section C.4's matrix in CI-manifested scripts; record exact artifact hashes; add a protection gate for dense Gemma plain/MTP and E4B separately; update docs to match current source.
- **Benefit range:** direct runtime benefit **0**; decision benefit is eliminating an unbounded evidence gap. This is prerequisite work, not a speed claim.
- **Cost:** M. **Determinism:** measurement only.
- **Dependencies:** model artifacts, vLLM image pinned to commit, GPU time.
- **Falsify/stop:** none; if artifacts cannot be normalized, publish format-qualified results rather than forcing parity.
- **Rollback:** scripts/docs only; no runtime change.

### E9. Tune and productize existing Gemma batched MTP, rather than reimplement it

- **Bottleneck:** current source has batched MTP but no fresh GB10 served acceptance/throughput/SLO matrix and docs/defaults are inconsistent.
- **Mechanism:** instrument draft calls, drafted/accepted tokens, verify rows, rejection and fallback; choose fixed per-batch draft depth from offline evidence; verify target-committed token/state equivalence to plain decode.
- **Benefit range:** **0 to the measured single-stream ceiling**. Historical fucina claims roughly 28→57 tok/s for one Gemma mode, so +104% is an evidence-derived *single-stream upper reference*, not a batch forecast. Merge gate: ≥15% aggregate gain in at least one target concurrency band, no >5% regression elsewhere.
- **Cost:** M. **Determinism:** target output exact/lossless; draft logits need not match because target re-verifies.
- **Dependencies:** exact assistant hash, accepted-length telemetry, plain-vs-spec committed state gate.
- **Falsify/stop:** acceptance too low, verify cost exceeds saved steps, or batch p95 regresses.
- **Rollback:** unload assistant/disable spec; plain batch path unchanged.

### E10. Bring E4B to server/API parity only after the Gemma matrix

- **Bottleneck:** E4B's standalone engine, PLE/KV-sharing, assistant, and tests are not equivalent to a supported multi-tenant server product.
- **Mechanism:** reuse the existing `BatchEngine` interfaces rather than a second scheduler; implement slot lifecycle, cancellation, bounded paged state, MTP, metrics, and exact model detection in the E4B bridge.
- **Benefit range:** runtime **unknown (0 until measured)**; capability moves from local engine to served support. Do not infer vLLM parity from local tests.
- **Cost:** L. **Determinism:** qualified until per-slot/graph/plain/spec gates pass.
- **Dependencies:** E4B benchmark results, memory accounting, API model identity.
- **Falsify/stop:** memory cannot sustain target concurrency or vLLM is decisively better with no product need.
- **Rollback:** separate E4B server enable flag; standalone engine remains.

### E11. Bounded distributed prefill using DS4's flow-control ideas

- **Bottleneck:** long-prefill compute or model residency may exceed one GB10; fucina's distributed phase remains experimental.
- **Mechanism:** fixed slot ring, bounded window, ACK-only intermediate chunks, prefix/result hashes, and preallocated activation buffers, following `ds4_distributed.c:3070-3520`; retain fucina model/state ownership and explicit failure recovery.
- **Benefit range:** **0 to the single-device prefill portion moved off-device, minus activation/network overhead**. DS4's own speedups are not reused as a forecast. Require ≥1.2× long-prefill speedup and no decode-SLO regression to continue.
- **Cost:** L/XL. **Determinism:** qualified; activation compression and partitioned reduction must pass exact/tolerance contract explicitly.
- **Dependencies:** second GB10, topology measurement, failure injection, security/authentication.
- **Falsify/stop:** network/serialization consumes ≥80% of moved compute or recovery cannot preserve session state.
- **Rollback:** single-node default; distributed route disabled.

### E12. Default-off HMM/model-prefetch startup experiment

- **Bottleneck:** model loading and transformed-weight residency may cause startup delay or page-fault tails, but no comparable startup evidence exists.
- **Mechanism:** test read-mostly/preferred-location/prefetch on a nonblocking stream, inspired by `ds4_cuda.cu:870-950`, under fucina's physical-headroom accounting. Record startup-to-ready, first-token cold tail, resident/available memory, and later eviction behavior.
- **Benefit range:** **0 to measured page-fault/staging time**; no steady-state throughput claim. It may regress under unified-memory contention.
- **Cost:** M. **Determinism:** exact arithmetic; operational timing only.
- **Dependencies:** cold-cache protocol, memory-pressure scenarios, no implicit oversubscription.
- **Falsify/stop:** no repeatable cold-start gain or p95 decode/page-fault regression.
- **Rollback:** default off; current explicit load path retained.

---

## F. Ranked roadmap and branch disposition

### F.1 Ranked roadmap

| Rank | Work | Why now | Exit gate |
|---:|---|---|---|
| 1 | **Qwen N=32 admission nsys + E1/E2 decision** | Largest measured interactive deficit; exact-preserving candidates exist. | Attribute ≥90% of admission GPU/wall time; implement only a phase ≥5%. |
| 2 | **Gemma fresh matrix + docs repair (E8)** | Current competitive position is unknowable and docs contradict source. | Raw matrix, artifact hashes, quality/determinism results, protection gate. |
| 3 | **MoE B>1 determinism root-cause (E5)** | Necessary to make “exact serving” honest. | Repeated byte/state pass or explicitly scoped strict fallback. |
| 4 | **`feat/ds4-expert-policy` held-out validation and merge decision (E6)** | Good bounded policy, but useful only in SSD mode and currently lacks real serving proof. | All pre-merge requirements below. |
| 5 | **Gemma existing batched-MTP productization (E9)** | Code exists; evidence/defaults are the gap. | ≥15% target-band gain, lossless target state, no >5% protected regression. |
| 6 | **Qwen full-attention packing only if profile proves it (E3)** | Plausible short-burst residual, higher exactness risk. | Phase ≥5%, exact byte/state gate. |
| 7 | **Durable snapshot scoring (E7)** | Low arithmetic risk, useful for coding-agent multi-turn workloads. | Held-out replay beats LRU and real TTFT-saved telemetry. |
| 8 | **E4B server parity (E10)** | Capability gap, but should follow evidence. | Product need + memory/performance acceptance. |
| 9 | **HMM startup experiment (E12)** | Unknown opportunity; default-off and bounded. | Repeatable startup/first-request win without steady-state tail. |
| 10 | **Distributed prefill (E11)** | Strategic but high cost; local long prefill already competitive. | ≥1.2× and robust recovery on real second-node topology. |

**Dense N=32 is not on this roadmap as another dp4a tuning project.** It is accepted under the exact-mode exception in B.5. Reopen only with new evidence or a new explicit arithmetic contract.

### F.2 `feat/ds4-expert-policy`: concrete pre-merge requirements

Branch head observed: `4f42856b1d23acd5fc37b4fdc03f2f1ec0d67e48`; two commits ahead of merge base `10f364d`, while main is one documentation commit ahead (`900de79`). The branch adds 1,350 lines across 11 files. Host tests have been reported passing, the producer/consumer size contract is now proven at 46,206,976 bytes below the 64 MiB parser cap, and `git diff --check` is clean. That is necessary but not sufficient.

Before merge:

1. **Rebase/merge current main** and resolve the one documentation commit so the final statement still says 11/12 aggregate, not TTFT.
2. **Run the complete authoritative gates on the rebased commit:** `make expert-policy-test`, Go tests/race, build/link, Qwen dense/MoE engine tests, SSD streaming test, state/chunk/multiseq tests, Gemma/E4B/diffusion tests. Host-only tests do not prove CUDA integration.
3. **Prove default-off zero effect** with binary-level/runtime evidence: unset profile env must allocate no recorder, copy no extra bytes, add no synchronization, preserve CUDA graph behavior, output hashes, engine memory, and N=1..32 protection metrics.
4. **Exercise real SSD mode on GB10** with checksum failure, short read, invalid plan, unwritable output, full disk, interrupted shutdown, oversized/pathological env values, and abnormal process termination. Profile write failure must remain nonfatal to completed inference.
5. **Collect representative training traces and separate held-out traces** across languages, code, chat, long agent sessions, and tenant mixtures. Report dropped-event ratio, uniqueness, adjacency, actual cache hits/misses, SSD reads/bytes, and checksum failures.
6. **Validate capacity curves against real runs.** Simulated capacities must predict actual hit/miss direction; startup seeding and OS page-cache effects must be measured separately.
7. **Require a held-out serving win:** statistically repeatable p95 or p99 TTFT/ITL improvement in a memory-constrained SSD configuration, with no quality/output change and no >5% aggregate regression. There is no resident-mode performance requirement because the feature should not run there.
8. **Overfit defense:** compare hotlist trained on workload A against held-out A and distinct workload B; if B regresses, document profile versioning and a no-plan fallback rather than auto-refreshing from timing.
9. **Schema/identity hardening:** bind plan/profile operationally to model checkpoint hash, transformed expert-store hash, geometry, and quantization. The current `source_sha256` authenticates profile bytes, not necessarily the model/store identity.
10. **Privacy review:** routes are less sensitive than tokens but can fingerprint workload/model behavior; document profile retention, permissions (`0600` is good), and operator responsibility.
11. **Operational docs:** shutdown flush can take seconds (synthetic 9.1 MiB flush measured 1.36 s); document graceful-stop behavior, trace cap, dropped events, and disk-space sizing.
12. **Keep it deterministic and offline.** No timing-derived online capacity selection, eviction order, or arithmetic dispatch. The plan is generated offline and applied on restart.

**Merge verdict:** **conditionally merge after these gates**, not before. The implementation shape is sound and faithful to the highest-value DS4 policy idea, but current evidence establishes host determinism/safety—not a GB10 SSD serving benefit.

### F.3 DS4 techniques to adopt, adapt, or reject

| DS4 technique | Decision | fucina adaptation |
|---|---|---|
| Route histograms, adjacent overlap, simulated capacities | **Adopt** | `feat/ds4-expert-policy`, offline deterministic hotlist, held-out validation. |
| Durable checkpoint score (decayed hits × tokens/bytes) | **Adapt** | Apply to exact fucina session snapshots; preserve model/state schema checks. |
| Fixed graph buffers and bounded rings | **Adopt principle** | Reuse buffers in hot paths; retain model-generic ownership where already present. |
| ACK-only pipelined prefill and bounded flow window | **Adapt later** | Phase-E distributed work, with hashes and failure injection. |
| HMM read-mostly/prefetch | **Experiment only** | Default-off, physical-headroom gate, cold/startup evidence. |
| DS4 SSD/ROCm/Metal kernel numbers | **Reject as evidence** | Different model, quantization, and platform. |
| DS4 serialized serving topology | **Reject** | fucina should keep continuous batching. |
| Whole-session “disk KV” as paged-prefix equivalent | **Reject terminology** | Treat as durable exact checkpoint, not token-granular KV cache. |

---

## G. Top-three coding-agent briefs

### Brief 1 — Qwen short-burst exact admission attribution and conv/GDN prototype

**Objective:** reduce Qwen3.5 dense and MoE N=32 short-burst TTFT without changing any committed model/state byte or regressing aggregate cells.

**Owned files:** `cuda/qwen35_runtime.cuh`, `cuda/qwen35_kernels.cuh`, one new microbench/test file, benchmark manifest/docs. Do not touch dense mixer D32 code.

**Tasks:**

1. Add phase timers/nvtx ranges around projection, conv/ring, GDN scan, full attention, FFN, LM head, scheduler wait, H2D/sync.
2. Capture nsys for M=1/2/4/8/16/32 with the exact fresh dense/MoE checkpoints and diverse short prompts.
3. If conv/ring ≥5%, implement E1 behind `FUCINA_QWEN35_FUSED_CONV`.
4. If first-chunk GDN ≥5%, implement E2 behind `FUCINA_QWEN35_CLEAN_GDN`; dispatch only when every relevant slot state is explicitly clean.
5. Run byte comparisons for logits, first token, ≥32-token continuation, full KV, GDN matrix, conv rings, graph on/off, cancellation, snapshot restore, mixed lengths 1..65, and M=1/2/4/8/16/32.
6. Re-run canonical served matrix and long probes; retain all raw JSON.

**Acceptance:** exact named-path bytes; ≥5% median or ≥8% p95 N=32 TTFT improvement; no >5% aggregate regression; no long-prefill regression >5%; no new allocation/sync in decode. **Stop immediately** if attribution is below threshold.

### Brief 2 — Fresh Gemma-vLLM GB10 evidence and source/docs reconciliation

**Objective:** produce the first honest current-main Gemma competitive matrix and align documentation with current-source batched MTP.

**Owned files:** benchmark scripts/manifests, Gemma docs, protection gate; runtime changes forbidden in this brief.

**Tasks:**

1. Pin fucina `900de79` and vLLM `5f8e73cb`; record image/toolchain/model/assistant hashes.
2. Execute section C.4 for dense Gemma-4 12B and supported E4B only where artifacts are semantically comparable.
3. Capture cold/warm startup, N=1 valid reruns, N=2..32, long/mixed prefill, plain/MTP, prefix cold/warm, memory, acceptance, and raw outputs.
4. Run greedy oracle/quality corpus; flag quantization/checkpoint differences explicitly.
5. Update `docs/continuous-batching.md` and related docs: batch MTP exists in current source; describe its actual default/product state and limitations.
6. Add a non-flaky protection report, not a claimed-win gate, until three-run variance is known.

**Acceptance:** complete raw artifacts, no N=1 cold artifact reported as normal, quality beside speed, docs match source, no runtime modification.

### Brief 3 — MoE grouped-GEMM determinism root cause

**Objective:** replace the current ambiguous “oracle passes but self-consistency fails” state with a precise component-level contract and, if feasible, a deterministic strict path.

**Owned files:** MoE route/grouped kernels and tests only; no scheduler or prefill tuning.

**Tasks:**

1. Pin identical router inputs/counts/offsets and hash after router, gather, activation quant, GU GEMM, SiLU, down GEMM, reduction, residual.
2. Repeat eager/graph and B=1/2/4/8/16/32 at least 100 times; identify first divergent tensor.
3. Check scratch initialization, group descriptor lifetime, atomics, split-K/reduction mode, and graph capture aliasing.
4. Build a fixed-order reference for the divergent stage. Compare tokens, logits, bytes, and speed.
5. If a performant fix exists, gate it across all Qwen tests and fresh serving. If not, expose an opt-in strict path and rewrite claims to scope default MoE determinism honestly.

**Acceptance:** either 100/100 byte-identical all shapes with ≤5% protected throughput regression, or a documented strict-mode trade-off plus explicit default limitation. No tolerance-only relabeling as exact.

---

## H. Rejected work and anti-roadmap

The following ideas are rejected because they are measured dead, violate exactness, or transfer invalid evidence. They must not be revived without genuinely new hardware/library/profile evidence.

1. **More D32/D32B dp4a mixer parameter sweeps**—DPSPLIT was marginal; PIPE≥3 spilled/regressed; PIPE×MINBLK and other occupancy variants were measured. The residual is kernel class.
2. **A tensor-core dense mixer presented as byte-identical to current Q4_K**—different reduction/quantization arithmetic. It may be a new qualified mode, not an exact replacement.
3. **Q8 batched LM-head half-read for Qwen low B**—measured slower than BF16 at B≥2 (9.01 vs 5.30 ms at B=2; 34.69 vs 5.24 ms at B=8).
4. **Low-concurrency expert grouped-GEMM tuning as the MoE lever**—the default NVFP4 grouped expert GEMM measured roughly 80–85% of peak bandwidth and scaled with active experts; N=2/4 already win fresh serving.
5. **More prefetch chunk sweeps/PIPE depth/routing slot packing without a new profile**—existing attribution ruled them out or found a floor.
6. **Claiming scheduler tuning alone can close MoE burst median TTFT**—the raw p95-minus-median spread is only 52 ms versus a 358 ms median gap; engine work must be profiled.
7. **Treating clean-prefix GDN as a general GDN bypass**—only zero initial state and the first chunk qualify. Warm/restored state and later chunks still require exact recurrence.
8. **Using DS4 DeepSeek-V4/ROCm/Metal performance numbers as expected Qwen/Gemma gains**—not comparable.
9. **Copying DS4's serialized serving loop**—would discard one of fucina's strongest competitive capabilities.
10. **Calling whole-session disk snapshots “prefix KV caching” without qualification**—snapshots are coarse durable checkpoints; paged prefix sharing is token/block-granular.
11. **Enabling timing-derived online expert policy**—would be workload-noisy, harder to reproduce, and outside the branch's deterministic offline design.
12. **Advertising 11/12 as TTFT or same-quality victory**—it is aggregate served throughput; MoE quality parity is not normalized.
13. **Advertising vLLM's two N=1 cold artifacts as representative losses**—they contradict adjacent valid single-short measurements.
14. **Reimplementing Gemma batch MTP because stale docs say it is missing**—current source already has it; benchmark and productize it instead.
15. **Writing Qwen DFlash P3/P4 without the exact real draft checkpoint**—existing work correctly stops at the real-weights boundary; untested kernels and acceptance claims are not evidence.

---

## Appendix 1 — Raw evidence inventory and caveats

### Revisions and trees

- fucina main: `900de794cc66f2bdf7f9ea2c761ee74ca0fb0c22`.
- expert-policy branch: `4f42856b1d23acd5fc37b4fdc03f2f1ec0d67e48`; merge base `10f364d`; main ahead by the documentation-only `900de79`; branch ahead by `5cbe48e` and `4f42856`.
- DS4 source: `/tmp/ds4` at `80ebbc39`.
- vLLM source: `/tmp/vllm` at exact `5f8e73cb8b8d41f7a2a5168cddf5b772888fa991`.

### Primary benchmark files

- `benchmark-evidence/PROTOCOL.md`
- `benchmark-evidence/results/2026-07-18-d32b/README.md`
- `fucina-q35dense-d32b.json`
- `fucina-q35dense-d32b-rep2.json`
- `fucina-q35moe-d32b.json`
- `vllm-q35dense-fresh.json`
- `vllm-q35moe-fresh.json`
- `baseline-dense-d32b.json`, `baseline-moe-d32b.json` (protection only)

### Key raw anomalies

- vLLM dense concurrency N=1: 6,533.7 ms TTFT, 10.24 agg tok/s; adjacent valid `single_short`: 65.3 ms, 21.67 decode tok/s.
- vLLM MoE concurrency N=1: 6,340.6 ms TTFT, 14.02 agg tok/s; adjacent valid `single_short`: 83.9 ms, 46.65 decode tok/s.
- fucina MoE outputs contain repetition and factual/arithmetic failure. This may reflect checkpoint/quantization/template behavior rather than the serving engine alone, which is exactly why quality normalization is required.
- Long TTFT is one probe per engine in these JSON files; use as directional evidence, not a p95 claim.

### Quantization/comparability caveat

The benchmark compares deployable systems/checkpoints as configured, not identical arithmetic kernels. fucina's load-time transformations and vLLM's FP8/tensor-core execution can produce different logits and quality. A systems throughput comparison remains useful, but a model-quality claim requires a common oracle and artifact provenance.

---

## Appendix 2 — DS4 source evidence and transfer boundaries

- Expert profiler geometry, capacity candidates, histograms, weighted counts, adjacent overlap, and deterministic sorting: `/tmp/ds4/ds4.c:751-845` and following profiler functions.
- Fixed release graph, persistent KV/compressed-state frontiers, speculative scratch, and separate batched-prefill buffers: `/tmp/ds4/ds4.c:10300-10500`.
- Selected expert page-in, layer read-ahead/pread/madvise, and selected-byte profiling: `/tmp/ds4/ds4.c:11500-12210`.
- Durable checkpoint eviction score and prefix supersession: `/tmp/ds4/ds4_kvstore.c:500-620`.
- Exact text-prefix lookup that restores exact stored tokens then tokenizes only suffix: `/tmp/ds4/ds4_kvstore.c:1190-1350`.
- CUDA HMM advice/prefetch, pinned staging, direct reads, and bounded device arenas: `/tmp/ds4/ds4_cuda.cu:870-1230`.
- Pipelined distributed prefill, fixed slots, ACK-only intermediate chunks, hashes, and bounded window: `/tmp/ds4/ds4_distributed.c:3070-3520`.

These mechanisms are technique evidence. DeepSeek-V4 layer counts, compression, expert geometry, checkpoint sizes, and DS4's Metal/ROCm/CUDA timings do not become Qwen3.5 or Gemma evidence by analogy.

---

## Appendix 3 — Determinism contract recommended for publication

Use four separate labels:

1. **Reference-exact:** token/logit/state bytes match a named standalone/reference path.
2. **Run-deterministic:** repeated runs of the same engine/build/shape produce identical bytes.
3. **Lossless speculation:** target-committed output/state equals plain target decode; draft internals may differ.
4. **Quality-qualified:** output differs by design due to format/batching, but passes named logit/token/quality bounds.

For current fucina:

- dense Qwen state restore, graph-on/off, and D32 hashes support strong named-path exactness;
- Qwen clean-prefix optimizations must earn reference-exact status independently;
- Gemma MTP should be described as lossless target verification after its served gate;
- concurrent MoE B>1 must remain explicitly outside a global run-deterministic claim until E5 passes;
- multisequence MoE prefill's documented tolerance/batch dependence must not be called byte-identical standalone inference.

---

## Appendix 4 — Final competitive position in one sentence

**Choose fucina today for measured GB10 Qwen3.5 throughput, long prefill, compact local operation, and exact hybrid session snapshots; choose vLLM for the lowest measured short-burst TTFT and the broadest production serving feature set; treat Gemma performance as unranked until the fresh matrix, and mine DS4 only for bounded policy/orchestration ideas.**

GB10_COMPETITIVE_ASSESSMENT_COMPLETE