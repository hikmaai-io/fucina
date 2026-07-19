# DS4 technique-mining report for fucina

**Scope.** Analysis of antirez's DwarfStar (`ds4`) at commit `80ebbc39` (2026-06-17), with cascade assessment for fucina on NVIDIA GB10. This is a source/design review, not a proposal to replace fucina's measured fast paths. No source under `/tmp/ds4` was modified.

## Executive verdict

DS4's strongest transferable ideas are **policy and orchestration**, not kernel transplants:

1. profile routed-expert locality and use the profile to seed/score an already-existing SSD expert cache;
2. score durable prefix checkpoints by decayed reuse value per byte, and place checkpoints at stable, naturally reached boundaries;
3. carry rolling prefix identity and topology-neutral state through a bounded distributed-prefill pipeline;
4. make all prefill storage fixed-capacity and reusable, then tune chunk size as a correctness-qualified kernel parameter rather than an API batching parameter.

The source is also valuable as negative evidence. DS4 has **no online SSD attention-KV tier**: its “disk KV cache” is whole-session checkpoint/resume. Its SSD streaming is chiefly **quantized MoE-weight streaming**, which fucina already has. Its server serializes inference and does not cross-request batch. Its decode is not CUDA-graph replayed. Its 2-bit/dp4a and ROCm WMMA kernels target another model and another kernel regime. Its Strix-Halo and Metal assumptions do not automatically carry to GB10 merely because all three systems have unified physical memory.

For fucina, the ranked recommendation is therefore: **adopt small policy refinements; prototype bounded distributed-prefill and deterministic GDN chunk schedules; reject wholesale kernel, paging, serving-loop, radix-tree, and SSD-tier ports.**

---

# A. What DS4 is

## A.1 Architecture

DwarfStar is a deliberately narrow, self-contained DeepSeek V4 Flash/PRO runtime rather than a generic GGUF executor (`README.md:1-31`; `AGENT.md:1-14`). Its major pieces are:

- `ds4.c`: model loading, tokenizer, CPU reference, session state, graph scheduling, GPU-independent orchestration, and session payload serialization (`AGENT.md:35-42`).
- `ds4_metal.m` plus `metal/*.metal`: the primary production backend.
- `ds4_cuda.cu`: CUDA backend, including a GB10/Spark-specific model-memory path, quantized kernels, and CUDA SSD expert staging.
- `ds4_rocm.cu` plus `rocm/*.cuh`: Strix-Halo backend. The large ROCm implementation is split by operation—MoE, attention, matmul, indexer, compressor, norm/rope, router, output—rather than concentrated in the top-level translation unit.
- `ds4_kvstore.c/.h`: durable session/prefix files and their retention policy.
- `ds4_distributed.c/.h`: layer-pipeline distribution hidden behind the ordinary session API.
- `ds4_server.c`, `ds4_agent.c`, and front ends: stateful serving and model-specific agent integration.
- `rax.c/.h`: Redis's compact radix tree, used in the server for protocol replay metadata, not tensor scheduling.

The graph is explicitly bifurcated into one-token decode and batched prefill. Persistent KV tensors live beside reusable per-layer scratch; batched prefill has a separately allocated fixed-capacity tensor set (`ds4.c:10306-10499`). This is a hand-described model execution plan, not a generic operator graph.

## A.2 Target hardware and design philosophy

The primary target is Metal on unified-memory Macs; CUDA/DGX Spark and ROCm/Strix Halo are supported production GPU paths, while CPU is reference/debug only (`README.md:13-22,58-69`; `AGENT.md:8-14`). The philosophy is “one model at a time,” readable C, known tensor names/layouts, correctness against official vectors, and end-to-end session/agent behavior. `AGENT.md:16-32` explicitly prioritizes correctness, small APIs, no unexplained drift, and no permanent semantic variants.

This matches fucina in several important ways:

- narrow vertical specialization rather than generic-framework overhead;
- hand-owned memory layouts and kernels;
- long-context and serving behavior treated as product features;
- output drift treated as a defect, not merely a benchmark tradeoff.

It diverges in equally important ways:

- DS4's model is DeepSeek V4 with compressed attention/indexer KV and highly quantized routed experts; fucina is Qwen3.5-only.
- DS4's primary optimization intuition comes from Apple/AMD APUs and very large models near or beyond RAM capacity. Fucina's winning paths are CUDA-specific, graph-replayed, concurrent, and measured on GB10.
- DS4's current server has one mutable graph and serializes independent requests (`README.md`, Server section, approximately `README.md:800-821`). Fucina already wins 11/12 concurrency cells with batched multisequence admission-prefill.
- DS4 generally optimizes capacity and simplicity first. Fucina already operates near kernel-class boundaries: Q4_K dp4a mixer at its measured register/latency wall, and NVFP4 grouped MoE at 80–85% peak bandwidth.

The correct use of DS4 here is thus **pattern extraction**, not code reuse.

---

# B. Findings by focus area

## B.1 Prefill

### B.1.1 Layer-major, fixed-capacity chunked prefill

DS4 allocates one persistent set of decode tensors and one persistent set of batched-prefill tensors. A prompt chunk goes through every layer in layer-major order while updating the same KV state later used by decode (`ds4.c:10306-10499`, especially `10427-10499`). Session prefill defaults to 4096-token chunks; a run can select another chunk or one whole batch. The range-capable graph is reused for every chunk rather than rebuilt per layer/chunk (`README.md`, Benchmarking section, approximately `README.md:730-758`).

**Generalized pattern:** make prefill chunking an internal fixed-shape execution schedule. Allocate at a maximum cap, reuse buffers, and pass a live token range. This avoids allocation and graph-construction noise while bounding scratch memory.

**Important correctness caveat:** DS4 explicitly documents that changing chunk size changes its KV checkpoint/logit path, because compressor/indexer frontier finalization is chunk-sensitive. Its official-vector path pins 2048 (`README.md`, Benchmarking and Test Vectors sections, approximately `README.md:744-758,1190-1210`). Chunking is therefore not semantically free even in DS4.

### B.1.2 Batch projections, then stream state updates

The CPU reference makes the intended ordering unusually clear. Batched attention projects Q and KV for all tokens, then streams token positions through raw/compressed cache updates and prefix attention (`ds4.c:9330-9449`). Its prefill Q8 path quantizes the activation batch and then scans weights (`ds4.c:5221` onward). The default FFN path batches hyperconnection/shared-expert work while treating routed work specially (`ds4.c:8115-8230`).

**Generalized pattern:** separate high-arithmetic-intensity batch transforms from serial/stateful scan frontiers. Batch the projections and shared work; retain explicit ordering only where KV compression or causal scan demands it.

For fucina, this pattern is relevant only to the known GDN chunk-scan occupancy headroom. It is not evidence that fucina should replace already-fast LM-head, mixer, or grouped-expert kernels.

### B.1.3 Specialized prefix-attention branch

DS4 has a fresh-prefix branch that can process attention rows in parallel only when the raw cache is empty and `pos0 == 0`; otherwise it follows the incremental path. It conditionally batches rope work and allocates masks/scratch according to compression ratio (`ds4.c:9411-9449`).

**Generalized pattern:** distinguish a clean, contiguous first-prefix scan from continuation-prefill. The former can use stronger parallel assumptions; the latter must preserve existing frontier state.

This is potentially useful in fucina's GDN implementation if its current kernel treats clean-prefix and continuation chunks identically. Any alternate path must land on the exact same state bytes at each boundary.

### B.1.4 Prefill-specific routed-expert selection and compaction

For SSD mode, DS4 can gather only selected experts into compact gate/up/down storage and pass a slot-remapped selected tensor. On Metal this path is guarded by model format, six experts per token, cache capacity, quality mode, and a token range; default automatic maxima are about 760 tokens for Flash and 800 for PRO (`ds4.c:11563-11635`). CUDA has an analogous selected-address path for Q4_K or IQ2/Q2 expert layouts (`ds4.c:11635-11693`). DS4 also profiles selected-expert uniqueness and selected-vs-full bytes per layer (`ds4.c:11918-12079`).

**Generalized pattern:** use routing output as an I/O plan. For small/medium prefill chunks, compact only the distinct experts used by the chunk; for large chunks, where distinct experts approach the whole layer, switch to sequential full-layer reads.

This is clever in a capacity path, but it is not automatically useful for fucina's resident NVFP4 grouped MoE. At batch 32 or long chunks, expert coverage can saturate, turning compaction into extra synchronization and copies.

### B.1.5 Bounded distributed-prefill pipeline

Distributed prefill cuts the prompt into microbatches. The coordinator computes its layer slice for chunk N+1 while downstream workers process chunk N. Intermediate chunks return ACKs; only the final chunk needs hidden state/logits (`ds4_distributed.c:3096-3109`). A reader enforces a bounded flow-control window and validates prefix hashes (`ds4_distributed.c:3290-3415`). Chunk/window validation and a hard window limit appear at `ds4_distributed.c:3421-3500`.

**Generalized pattern:** pipeline only the naturally batch-parallel phase, use bounded in-flight work, and keep autoregressive decode synchronous.

**Measured claim:** on two M5 Max 128 GB Macs over Thunderbolt 5 with a 4096-token chunk, DS4 reports 1.38× at 9,421 prompt tokens, 1.66× at 28,684, and 1.85× at 63,819. Decode falls from 30.59 to 24.67 t/s, a 19.4% loss (`README.md`, Distributed Inference, approximately `README.md:286-330`). These are pipeline-parallel, model/host/link-specific measurements—not evidence for a GB10 speedup without real CUDA sharding.

### B.1.6 What DS4 does *not* provide

- The server does not batch independent requests; one graph worker serializes inference.
- The local single-GPU prefill path does not demonstrate a general overlap of consecutive chunks through the same layers; the explicit N/N+1 overlap is distributed pipeline parallelism.
- There is no evidence that DS4's chunk choice is byte-identical across configurations; documentation says the opposite.
- DS4's measured GB10 prefill is good but below fucina's supplied long-context result in the relevant product engine, and it says nothing about fucina's N=32 admission performance.

### B.1.7 Measured prefill baseline

`speed-bench/gb10.csv:2-33` reports incremental 2048-token prefill on Spark dropping from **402.88 t/s at 2,048 context** to **287.44 t/s at 65,536 context**. The README's 7,047-token sample reports **343.81 t/s prefill** and **13.75 t/s generation**. On Metal, the headline long-prompt numbers range from roughly 250 t/s on M3 Max to 463–468 t/s on M5 Max/M3 Ultra (`README.md`, Speed table, approximately `README.md:145-175`). These numbers validate the layer-major chunk path; they do not show a concurrency or TTFT advantage over fucina.

---

## B.2 KV cache, SSD tiers, eviction, prefetch, and compression

DS4 uses “cache” for two substantially different systems. They should not be conflated.

### B.2.1 Online SSD streaming is for routed MoE *weights*

In capacity mode, non-routed weights remain resident while routed experts are loaded from the GGUF into a bounded cache. `ds4_ssd.c` itself is mostly CLI budget parsing, an 80%-of-recommended-memory planner, and a chunked `mmap`/touch/`mlock` diagnostic (`ds4_ssd.c:14-111,113-181`). The actual expert cache, asynchronous reads, and backend storage are in `ds4.c`, `ds4_metal.m`, `ds4_cuda.cu`, and ROCm helpers.

That distinction matters: **`ds4_ssd.c` is not an SSD KV pager and does not contain the main streaming algorithm.**

The cache unit is a complete routed expert's gate/up/down weights. The CUDA cache stores contiguous gate/up/down slabs plus metadata slots keyed by model, layer, and expert, with an age counter (`ds4_cuda.cu:132-184`). A separate selected cache packs a batch's distinct experts and remaps selected IDs (`ds4_cuda.cu:132-155`).

**Generalized pattern:** cache in the unit consumed by the kernel, not in arbitrary byte pages. Keep compact metadata and large contiguous quantized slabs.

### B.2.2 Profile-driven preload and cache simulation

DS4's locality profiler records per-layer expert histograms, route-weight histograms, adjacent-token overlap/Jaccard, and simulated latest-N cache hits for capacities from 1 to 384 (`ds4.c:750-829`). `ds4_streaming_hotlist.inc` is generated in descending hit/weight order and supplies default preload pairs. This makes preload policy empirical rather than based on expert number or uniformity.

**Generalized pattern:** collect workload traces without changing the serving path, replay them through candidate cache capacities/policies, and generate a deterministic preload list.

This is one of the strongest cascade candidates because fucina already has the expensive capability—SSD expert streaming. The incremental opportunity is better policy and telemetry, not another streamer.

### B.2.3 Hide I/O under useful compute

`AGENT.md:8-14` states the intended policy explicitly: load missing experts while shared experts and resident routed experts execute, and pre-load the next layer during current-layer prefill. CUDA uses four pinned staging buffers with events and a nonblocking upload stream (`ds4_cuda.cu:1014-1053`), `pread` loops with EINTR handling (`ds4_cuda.cu:1055-1069`), and optional aligned `O_DIRECT` reads with ordinary `pread` fallback (`ds4_cuda.cu:1070-1123`). The same four-way staging concept is used for selected experts (`ds4_cuda.cu:218-222,1692` onward).

For large ROCm prefills, DS4 can choose full-layer expert loading rather than per-selected-expert reads, start that load in a pthread, and seed the decode cache from a bounded number of selected rows after prefill (`ds4.c:11696-11916`).

**Generalized pattern:** choose I/O granularity from expected expert coverage, then double-/multi-buffer reads and uploads behind shared/resident compute. Preserve a small prefill-to-decode handoff so decode does not begin cold.

### B.2.4 No extra streaming compression stage

The streamed bytes are already quantized GGUF experts. DS4 does not add a new lossless/lossy SSD compression codec in the online path. Its model quants are asymmetric: routed gate/up IQ2_XXS and down Q2_K in the 2-bit profile, while shared experts, projections, routing, and output stay at higher precision (`README.md`, Model Weights, approximately `README.md:105-135`).

**Adversarial conclusion:** “compression” here is offline model quantization, not a novel disk-tier compressor. Fucina's NVFP4/Q4_K decisions have their own measured quality and kernel constraints; DS4's quant mix is not directly transferable.

### B.2.5 The disk KV store is checkpoint/resume, not online attention paging

`ds4_kvstore` writes a complete model-specific session payload: exact token IDs, next-token logits, per-layer row counts, raw sliding-window rows, compressed attention/indexer rows, and compressor frontier state. Files are keyed by SHA-1 of rendered prefix bytes, while the payload's exact tokens remain authoritative (`ds4_kvstore.h:29-60,171-218`; `README.md`, Disk KV Cache format, approximately `README.md:1063-1135`). Ordinary `read`/`write` is intentional; the store avoids adding mappings to an already mapping-heavy process.

On a hit, DS4 validates the rendered byte prefix, loads exact graph state, and tokenizes only the textual suffix. This handles BPE boundary changes where identical rendered bytes do not imply the same token split (`ds4_kvstore.c:1240-1338`). Files are temp-written and atomically renamed (`ds4_kvstore.c:930-1085`).

**Generalized pattern:** use rendered bytes to discover reusable client-visible prefixes, but restore exact token/model state and continue from that exact state. The textual key and computational identity have different jobs.

**What it is not:** attention kernels never demand-page arbitrary historical KV rows from SSD. There is no per-page online eviction/prefetch loop for attention. Calling this an SSD KV “tier” in the vLLM-style serving sense would overstate it.

### B.2.6 Stable checkpoint placement

Defaults are:

- minimum 512 tokens;
- trim 32 tokens from a cold boundary;
- align down to 2048;
- continued checkpoint interval around 10,000 tokens, rounded to alignment;
- cold prompt maximum 30,000 tokens (`ds4_kvstore.c:30-49,151-169,676-746`).

The trim avoids tokenizer merges at an append boundary; the alignment matches the prefill/compressor schedule. Continued snapshots are written only when the live graph naturally reaches an absolute frontier (`ds4_kvstore.c:709-746`).

**Generalized pattern:** do not persist every final token. Persist a stable prefix behind the mutable tail, at an execution boundary whose state is reproducible.

### B.2.7 Value-density eviction

DS4's disk-cache score is approximately:

`(decayed_hits + 1) * tokens / file_size`

Hits decay with a six-hour half-life. Cold/evict/shutdown “anchor” reasons receive a 2× prior. A continued checkpoint that is a strict prefix of the incoming longer checkpoint is strongly discounted, especially if never hit (`ds4_kvstore.h:13-15`; `ds4_kvstore.c:30-49,484-570`). Lowest score is evicted, with last-use as a tie-break (`ds4_kvstore.c:550-603`).

**Generalized pattern:** retain saved compute per byte, corrected for recency and checkpoint role, rather than plain LRU.

### B.2.8 Compressed in-memory KV is model-native

The live graph stores a raw sliding-window ring and per-layer compressed caches. Ratio-4 indexer layers and ratio-128 attention-compressed layers have different capacities, and frontier tensors are part of checkpoint identity (`ds4.c:10328-10379`). This is DeepSeek V4 architecture support, not a generic post-hoc KV quantizer.

**Adversarial conclusion:** fucina cannot cascade the ratio-4/ratio-128 representation onto Qwen3.5 without changing model semantics. Only the implementation principle—serialize frontier state with compressed rows, and size each layer from its actual ratio—transfers.

---

## B.3 Efficiency: memory layout, quantization, kernels, allocation, and rax

### B.3.1 Explicit stage layout and lifetime allocation

DS4 names every persistent and temporary tensor in `ds4_gpu_graph`, allocates batched buffers from `prefill_cap`, reuses per-layer scratch, and sizes compressed caches by each layer's true compression ratio (`ds4.c:10306-10499,11090-11290`). This sacrifices genericity for predictable allocation and easy lifetime reasoning.

**Generalized pattern:** allocate execution-plan storage once per session/shape key; alias only when lifetime is locally obvious. This is compatible with CUDA graphs and deterministic execution.

Fucina already has keyed graph decode and purpose-built kernels, so the relevant audit is narrow: ensure GDN prefill uses stable arenas and has no per-chunk allocation or size-dependent graph construction. A port of DS4's tensor catalog would add no value.

### B.3.2 CUDA model residency has multiple fallbacks

DS4's CUDA path supports:

1. ATS/HMM access to the mmap-backed model, with `ReadMostly`, preferred-device location, and optional asynchronous prefetch (`ds4_cuda.cu:874-952`);
2. host-register/mapped ranges (`ds4_cuda.cu:284-359`);
3. explicit device copies by model range;
4. large CUDA allocation arenas, defaulting to 1792 MiB chunks and 256-byte suballocation alignment (`ds4_cuda.cu:1145-1212`);
5. optional Q8-to-F16/F32 caches under explicit budgets and reserve headroom (`ds4_cuda.cu:485-790`).

After explicit copies it can discard source VM/file pages with `posix_madvise`/`posix_fadvise` (`ds4_cuda.cu:967-995`). A direct host pointer is accepted only when CUDA owns/registers/prefetches the mapping, avoiding delayed illegal accesses (`ds4_cuda.cu:1203-1212`).

**Generalized pattern:** treat model residency as a policy ladder, but make the selected steady-state explicit and instrumented. Pool large weight allocations to avoid allocator fragmentation.

**GB10 caveat:** HMM accessibility is not proof that demand access is fast. Hot fucina weights should stay resident and predictable. HMM/prefetch is useful as a startup/capacity experiment, not a replacement for the measured resident fast path.

### B.3.3 Quantized kernel structure

The ROCm MoE code performs IQ2/Q2 products with packed integer operations and `dp4a`-style primitives (`rocm/ds4_rocm_moe.cuh:1-80`). Batch paths also use rocWMMA; the indexer has a 16×16×16 WMMA score kernel (`rocm/ds4_rocm_indexer.cuh:128-213`), and matmul launch policy selects a four-wave WMMA batch kernel (`rocm/ds4_rocm_matmul.cuh:285-295`). Routed MoE bucket construction explicitly preserves pair order before WMMA kernels (`rocm/ds4_rocm_moe.cuh:1032` onward).

The source pattern is sensible:

- low-batch/decode quantized dot kernels use packed integer instructions;
- sufficiently batched work crosses to matrix-instruction kernels;
- routing is bucketed/compacted before expert GEMM;
- ordering constraints are documented where bucket construction affects results.

But fucina already knows this boundary empirically. Its Q4_K dp4a mixer is at a proven latency/register wall, and its NVFP4 grouped experts are at 80–85% peak bandwidth. DS4 does not provide a missing CUDA kernel class for dense N=32, nor evidence that its ROCm WMMA tuning wins on Blackwell `sm_121a`.

### B.3.4 Quantization quality strategy

DS4 quantizes the huge routed-expert majority aggressively while retaining higher precision for shared experts, routing, projections, and output. It validates against official continuations and logits (`README.md`, Model Weights and Test Vectors). The transferable idea is sensitivity-weighted byte allocation, not the literal IQ2/Q2 formats.

Fucina's quant formats and quality gates are already product decisions. Requantizing Qwen3.5 to mimic DS4 would entail model-quality work and new kernels, with no supplied throughput case.

### B.3.5 Rax radix tree usage is outside inference

`ds4_server.c` uses two `rax` trees: ID → replay entry and exact sampled DSML block → replay block (`ds4_server.c:7670-7934`). This deduplicates and retrieves variable-length protocol strings for exact tool-call replay. It is not used for KV pages, expert residency, batching, or graph keys.

**Generalized pattern:** a compressed radix tree is appropriate for many shared-prefix byte strings and deterministic lexicographic iteration.

**Cascade verdict:** Go's map/string and existing prefix structures are likely simpler. Do not transplant `rax` into fucina's hot inference path. Consider a trie only if profiling shows large protocol-key memory or prefix-query cost.

### B.3.6 Decode and serving loop

DS4's native agent keeps inference in-process, treats live KV as session truth, and persists sessions. The server maintains one mutable checkpoint, exact-prefix reuse, exact DSML replay, and a single graph worker. This is elegant for a one-user local model but is dominated by fucina's keyed CUDA-graph decode, GPU-native input splice, prefix reuse, persistence, and measured concurrent admission.

DS4's experimental MTP speculative path is documented as at most a slight speedup (`README.md`, model/build and CLI sections). It does not justify disturbing fucina's byte-identity gate.

---

## B.4 Distributed execution, serving transferables, and unified memory

### B.4.1 Distributed architecture

Distributed execution is a backend behind the normal session API (`ds4_distributed.h:9-17,65-124`). Workers register model identity, quant profile, context capacity, and a layer range; a route covers all layers. Activations move worker-to-worker rather than being relayed through the coordinator. Each worker owns KV for its layers.

This is pipeline/layer parallelism, not tensor parallelism or expert parallelism. There are no collectives and no intra-layer CUDA sharding.

### B.4.2 Consistency and recovery

Work carries session/request IDs, positions, and rolling token-prefix hashes. A restarted or stale worker cannot silently accept work at the wrong position; hash/KV mismatch triggers prefix replay, while transport failure removes the route. Snapshots gather worker-owned rows into one ordinary layer-ordered DSV4 payload and redistribute them on load, so files are topology-neutral (`ds4_distributed.h:92-112`; README Distributed Protocol section).

**Generalized patterns:**

- content identity must accompany positional work;
- distinguish semantic state mismatch from transport failure;
- persist logical model state, not physical topology;
- drain in-flight work cooperatively before cancellation to avoid split KV timelines.

These patterns map well to fucina's parked FCNDIST1 protocol even before real CUDA sharding exists.

### B.4.3 Activation transport precision

DS4 defaults to FP32 activation transport and offers FP16/FP8. Its README says reduced activation size did not yield significant improvement and may be removed. FP8 is explicitly approximate (`README.md`, distributed tuning section, approximately `README.md:400-435`).

**Adversarial conclusion:** do not adopt transport quantization as a default. It changes values, can change token output, and did not demonstrate a worthwhile gain even in DS4's slower-link experiments.

### B.4.4 Link sensitivity

For the same two-Mac setup and 8,192-token prompt, DS4 reports:

- Thunderbolt 5, 0.45 ms ping: 582.99 prefill t/s, 25.09 generation t/s;
- Wi-Fi, 77.2 ms: 250.70 / 10.70;
- Internet/VPN, 152.1 ms: 114.88 / 3.63.

This confirms the expected shape: long prefill can fill a layer pipeline; decode pays link latency every token. It is capacity-oriented outside low-latency links.

### B.4.5 Strix Halo unified-memory lessons

`STRIXHALO.md:1-16` targets a 128 GB Radeon 8060S (`gfx1151`). The machine initially exposes only about 62 GB to the GPU; DS4 instructs users to disable the IOMMU and configure a roughly 126,976 MB GTT aperture so an 80.76 GiB model plus buffers is visible (`STRIXHALO.md:45-88`). It warns that mixed larger quants can provoke system OOM (`STRIXHALO.md:100-113`).

The lesson is mostly operational: “unified physical RAM” does not mean the accelerator can safely or efficiently address all of it. Aperture limits, driver accounting, pinned/lockable budgets, and OS paging remain first-class.

### B.4.6 GB10 is similar physically, different operationally

GB10 and Strix Halo both attach CPU and GPU to LPDDR rather than a discrete GDDR card, but the software and access paths differ:

- GB10's CUDA ATS/HMM, pageable-memory access, host registration, managed allocations, and prefetch semantics are NVIDIA-specific (`ds4_cuda.cu:874-952,2365-2395`).
- Strix Halo requires AMD GTT aperture tuning; that prescription does not apply to CUDA.
- Apple's SSD/page-cache behavior and Metal's unified allocations are not a model for Linux CUDA direct I/O.
- CUDA “zero copy” or HMM can avoid an explicit copy yet still incur page faults, migration/accounting overhead, weaker access behavior, and run-to-run latency variance.
- With ~273 GB/s shared LPDDR bandwidth, CPU reads, SSD staging, and GPU weight reads contend for one memory fabric. Overlap is not free; it can reduce rather than hide latency when the resident kernel is bandwidth-bound.

DS4's own CUDA backend responds by offering explicit range copies, large arenas, pinned staging, and source-page discard—not by assuming all unified access is equally fast. That is the relevant GB10 lesson.

### B.4.7 GB10 measured behavior in DS4

`speed-bench/gb10.csv:2-33` shows:

- prefill: 402.88 t/s at 2,048 context, declining to 287.44 t/s at 65,536;
- generation: 14.20 t/s at 2,048, declining to 12.08 t/s at 65,536;
- serialized KV footprint: about 52.2 MB at 2,048 and 926.0 MB at 65,536.

These figures support DS4's compressed-KV capacity claim. They do not beat or invalidate fucina's supplied 4.3 s single-3,500-token TTFT or 641 ms MoE N=32 admission-prefill, because model, token workload, quantization, and benchmark methodology differ.

---

# C. Cascade assessment onto fucina

Costs: **S** = isolated policy/instrumentation; **M** = one subsystem or kernel experiment; **L** = new kernel class, distributed runtime, or model-format work.

Determinism labels:

- **Compatible:** arithmetic and reduction order can remain unchanged.
- **Qualified:** possible only with fixed scheduling and byte/state equivalence tests.
- **Default-off:** inherently approximate or likely to change reduction order.

| DS4 technique | Concrete fucina mapping | Expected benefit against supplied state | Cost | Determinism gate | GB10 / adversarial caveat |
|---|---|---|---:|---|---|
| Expert locality profiler + capacity simulator (`ds4.c:750-829`) | Instrument fucina's existing SSD expert streamer: per-layer hit/weight histograms, adjacent-token overlap, replayed hit curves for candidate capacities; emit a deterministic preload table. | Better cold-start and cache-budget decisions; likely tail-latency benefit, not resident decode throughput. Most useful only when SSD expert tier is active. | S | Compatible; profiling must be observational and policy tie-breaks fixed. | Do not claim a new streaming capability. Shared LPDDR contention may make excessive preloading harmful. |
| Hotlist preload (`ds4_streaming_hotlist.inc`) | Seed fucina's existing expert residency from production traces, split by model/quant/workload; age versions with telemetry. | Reduces first-token cold misses and avoids uniform preload. Benefit workload-dependent. | S | Compatible if cache content cannot alter math. | Static hotlists can overfit; keep fallback and measure p95/p99, not just average. |
| Selected-expert compaction for short prefill (`ds4.c:11563-11693`) | In SSD-capacity mode only, compact distinct routed experts after routing for a short chunk; switch to full-layer sequential load after measured coverage threshold. | Potentially less SSD traffic for short prompts; little/no benefit for resident NVFP4 or long/N=32 batches that touch most experts. | M | Qualified; preserve token/expert bucket order and exact kernel path. | GPU→host routing readback would destroy TTFT; selection and I/O plan must remain device-native or asynchronously mirrored. |
| Next-layer / shared-compute I/O hiding (`AGENT.md:8-14`) | Audit whether fucina's existing streamer overlaps layer L+1 reads/uploads with layer L shared/resident work, using bounded pinned buffers/events. Add only missing overlap. | May reduce SSD-capacity stalls. No gain when weights are resident; can hurt bandwidth-bound NVFP4. | M | Compatible with fixed event dependencies. | GB10 SSD staging and GPU consume shared memory bandwidth; “overlap” must be measured under real kernels. |
| Multi-buffered pinned/O_DIRECT staging (`ds4_cuda.cu:1014-1123`) | Borrow robustness details—not code—if fucina lacks aligned direct-read fallback, EINTR-safe pread, event-owned staging slots, or adaptive granularity. | Better streaming stability and fewer bubbles. | S–M | Compatible. | GPUDirect Storage/CUDA APIs may dominate CPU `pread`; benchmark the existing fucina path before changing it. |
| Stable checkpoint trim/alignment (`ds4_kvstore.c:676-746`) | Refine existing session persistence: save behind mutable BPE tail and only at a known GDN/prefill frontier. | Fewer unusable snapshots and exact continuation rebuilds; storage/write reduction. | S | Compatible if load reproduces current exact state. | Alignment must match fucina's own scan semantics, not DS4's 2048. |
| Decayed saved-compute-per-byte eviction (`ds4_kvstore.c:484-603`) | If fucina currently uses FIFO/LRU for persistent prefix/session files, rank by recompute tokens or measured recompute ms per byte, decayed hits, and anchor role. | Better disk budget utility for agent workloads; no kernel speedup. | S | Compatible. | Use measured fucina prefill cost, not raw token count, because GDN/attention cost varies with position and prefix type. |
| Rendered-byte discovery + exact-token authority (`ds4_kvstore.c:1240-1338`) | Audit current prefix reuse at API/tool boundaries: discover by canonical rendered bytes, but restore exact token IDs, KV/frontiers, and logits. | Avoid BPE boundary misses without changing model state. | M if absent | Compatible and supportive of the product gate. | Do not weaken cryptographic/model-version identity; SHA-1 is adequate for accidental lookup but not adversarial trust. |
| Whole-session SSD KV snapshots | Compare to fucina's existing session persistence; adopt only missing atomic-write/topology metadata. | Fucina already has session persistence. Duplicate implementation offers little. | S audit / M duplicate | Compatible. | This is not online KV streaming and should not be sold as such. Large snapshots can add latency/write amplification. |
| Layer-major fixed-cap prefill workspace (`ds4.c:10306-10499`) | Audit GDN prefill for per-chunk allocation, dynamic graph creation, or unnecessarily live scratch; preallocate by keyed chunk cap. | Possible TTFT variance reduction and modest overhead reduction; unlikely to solve occupancy by itself. | S–M | Compatible. | Fucina likely already does much of this due to CUDA graphs and hand-written kernels. Avoid cargo-cult tensor duplication. |
| Fresh-prefix specialized scan (`ds4.c:9411-9449`) | Prototype a clean-prefix GDN kernel/schedule distinct from continuation-prefill, while writing the exact same frontier format. | Directly targets known GDN occupancy headroom for long initial prompts. Potentially material if current clean-prefix kernel is constrained by continuation logic. | M–L | Qualified; byte-compare every frontier/logit across chunk sizes. | A faster tree/parallel scan can change floating reduction order. Default-off unless exact, or use a mathematically/order-equivalent schedule. |
| Chunk-size sweep as an execution key | Add deterministic benchmark matrix over fixed GDN chunk classes, register occupancy, TTFT, and state equality; choose a fixed key from request shape, not timing. | May improve the known GDN occupancy headroom without touching already-floored MoE kernels. | M | Qualified. DS4 itself warns chunk changes output path. | Dynamic, timing-driven chunking can violate run-to-run identity. Fixed shape keys only. |
| Batch shared/projection work before state scan (`ds4.c:9330-9449`) | Keep/extend batching around GDN projections while isolating ordered scan. | Potential occupancy gain if fucina currently interleaves too much serial frontier work. | M | Qualified; projections are safe, scan ordering is not. | Requires Qwen3.5-specific analysis; DS4's compressed attention structure is not portable. |
| Bounded distributed-prefill window (`ds4_distributed.c:3096-3500`) | Extend parked FCNDIST1 tests/spec with fixed chunk sequence numbers, bounded credits, ACK-only intermediate chunks, final-result ownership, and cancellation drain. | Makes Phase E protocol robust and benchmarkable. No current single-GB10 speedup; enables future multi-node capacity/prefill. | M protocol; L real CUDA | Compatible in FP32 if each layer uses the same deterministic kernel/order; cross-device equivalence still must be tested. | Pipeline alone is not CUDA sharding. Network/serialization can dominate, and decode slows. |
| Rolling prefix hashes and mismatch replay | Put pre/post token-history hash and expected position in every FCNDIST1 work/result frame; distinguish mismatch from disconnect. | Prevents silent remote KV corruption and improves recovery. | S–M | Compatible; strengthens determinism. | Use a modern fast keyed/unkeyed hash as appropriate; hash is identity, not security unless authenticated. |
| Topology-neutral distributed snapshots | Serialize logical layer-order KV/frontier state and redistribute under the current shard map. | Sessions survive route changes; valuable if Phase E resumes. | M | Compatible if format is canonical. | Large gather/scatter pauses can be substantial. Version model/kernel/frontier ABI strictly. |
| FP16/FP8 activation transport | None by default. Potential opt-in lossy WAN capacity mode only. | DS4 observed no significant win; risks token drift. | M | **Default-off.** Not byte-identical. | Reject for product default. Even FP16 changes values from FP32. |
| ATS/HMM whole-model demand access (`ds4_cuda.cu:874-952`) | Benchmark-only capacity fallback against fucina's resident weights; possibly prefetch cold ranges at startup. | Might reduce startup copies or enable oversubscription; unlikely to improve steady-state inference. | M | Arithmetic compatible, latency nondeterministic; fixed admission needed. | Page faults and shared-LPDDR contention can destroy tail latency. Do not displace resident fast path. |
| Large CUDA weight arenas (`ds4_cuda.cu:1145-1212`) | Audit model-weight allocations for fragmentation and map overhead; pool only if current allocator shows a problem. | Startup/reliability improvement, not tokens/s. | S–M | Compatible. | 1.75 GiB is DS4-specific. Size from fucina topology and CUDA 13 behavior. |
| IQ2/Q2 dp4a kernels | None. Keep as external evidence for low-batch packed-dot design. | Does not break fucina's accepted Q4_K latency/register wall or dense N=32 kernel-class block. | L | New reductions/quantization: **default-off** until full quality and byte baseline exists. | Different model, format, backend tuning, and quality allocation. Reject transplant. |
| ROCm WMMA batch kernels | Concept only: retain fucina's existing matrix-core crossover and grouped NVFP4 work. | Existing NVFP4 experts already 80–85% peak BW; no credible incremental benefit. | L | Likely changes reduction order; default-off. | rocWMMA wave/fragment tuning is not Blackwell CUDA tuning. |
| Rax for protocol maps (`ds4_server.c:7670-7934`) | Continue using Go maps/current prefix index unless profile proves otherwise. | Negligible inference benefit. | M to port | Compatible but irrelevant. | C radix-tree cache locality does not translate automatically to Go or GPU hot paths. |
| Single mutable graph worker | Do not map. | Would regress the 11/12 concurrency-cell lead. | — | Deterministic but throughput-regressive. | Explicit reject. |
| Native in-process agent loop | Keep only model/session identity lessons; fucina already has prefix/session persistence and a serving architecture. | No measured kernel benefit; likely product coupling. | L | Potentially compatible. | DS4's one-user local regime differs from fucina's concurrent serving. |
| MTP speculative decode | Do not prioritize. | DS4 reports only slight/no meaningful speedup; jeopardizes byte identity and graph simplicity. | L | Default-off unless target verification reproduces baseline exactly. | Fucina's graph-replayed decode is already stronger evidence. |

## C.1 Explicit non-duplication against fucina's current state

The following are **not recommendations**, because fucina already has them or a stronger measured version:

- multisequence admission-prefill and N=32 batching;
- prefix-reuse KV and session persistence;
- SSD streaming for MoE experts;
- grouped MoE execution;
- weight-read-once LM head;
- keyed CUDA-graph decode and GPU-native input splice;
- low-bit mixer kernels;
- a distributed protocol/test pipeline.

Where DS4 is relevant, it is as a refinement: cache profiling for the existing expert streamer, retention policy for existing persistence, safety fields/windowing for FCNDIST1, or an exact clean-prefix GDN prototype.

## C.2 Determinism rules for any cascade

1. **Never select chunk size or distributed window from runtime timing.** Shape/configuration may choose a fixed key; timing must not.
2. **Preserve token and expert pair order.** Stable buckets alone are insufficient if the downstream grouped GEMM uses nondeterministic atomics or split-K reductions.
3. **Byte-compare frontier state, not only sampled output.** A temporary equal token can hide later drift.
4. **Memory policy may vary only if it cannot select an arithmetic kernel variant.** Resident/SSD hits must feed identical quantized bytes to the same launch order.
5. **Transport compression remains default-off.** FP16/FP8 is not byte-identical to FP32.
6. **Do not use the pre-existing grouped-GEMM B>1 nondeterminism as permission to add more.** New work should isolate or eliminate it, not normalize it.

---

# D. Ranked adopt / prototype / reject list

## ADOPT

### 1. Expert-streaming policy profiler and deterministic preload generation — **S**

**Why #1:** fucina already paid for SSD expert streaming; this extracts more value without another kernel class. Capture per-layer frequency, route-weight mass, adjacent overlap, miss latency, bytes, and simulated capacities. Generate versioned hotlists and validate on held-out agent traces. Arithmetic remains unchanged.

### 2. Value-density retention for existing persistent prefixes/sessions — **S**

Use **measured recompute milliseconds saved per stored byte**, decayed hits, and anchor role. DS4's token/byte score is a good starting pattern, but fucina can do better because it knows actual prefill cost. This is deterministic and improves disk budget quality rather than adding a duplicate persistence feature.

### 3. Stable-boundary checkpoint policy audit — **S**

Ensure snapshots sit behind the tokenization-mutating tail and exactly on fucina's own GDN/KV frontier. Save atomically and retain exact token IDs/logits/frontier metadata. This strengthens, rather than relaxes, the byte-identity product gate.

### 4. FCNDIST1 prefix identity, bounded credits, and cancellation drain — **M**

These protocol fields are useful even while Phase E remains parked. Add pre/post prefix hashes, expected position, fixed sequence IDs, bounded in-flight chunks, mismatch-vs-transport error separation, and topology-neutral snapshot versioning to tests/specification. This is not a claim that real CUDA sharding is implemented.

## PROTOTYPE

### 5. Clean-prefix GDN prefill schedule with exact frontier equivalence — **M–L**

This is the one kernel-oriented DS4 lesson aimed at a known fucina weakness rather than a solved path. Prototype a clean-prefix specialization and/or projection-batch/state-scan split. Gate success on byte-identical state and logits for all chunk boundaries. If exactness requires the existing reduction order and eliminates the speedup, reject it.

### 6. Fixed GDN chunk-class sweep and reusable workspace audit — **M**

Benchmark fixed chunk classes for occupancy, registers, scratch, TTFT, and frontier identity. Do not use DS4's 2048/4096 values blindly. The output should be one deterministic shape-key policy, not adaptive scheduling.

### 7. Coverage-threshold I/O mode for the existing expert streamer — **M**

For SSD mode only, compare selected-expert reads with sequential full-layer reads as distinct expert coverage rises; preserve a decode-hot seed. Prototype only if telemetry says fucina currently issues fragmented reads for near-full coverage.

### 8. Real distributed-prefill pipeline when Phase E resumes — **L**

Implement actual CUDA layer ownership first, then bounded pipeline prefill over low-latency links. Expect capacity and long-prompt wins, not faster decode. Keep FP32 transport for the deterministic product path. DS4's 1.38–1.85× numbers establish plausibility, not a forecast.

### 9. ATS/HMM startup/capacity experiment — **M**, low priority

Measure prefetch-resident, range-copy-resident, and HMM-demand paths on GB10 with p50/p99 TTFT and memory-fabric counters. This is a capacity/startup experiment only. The resident path remains default unless HMM wins under the real concurrent workload without latency variance.

## REJECT / DO NOT PRIORITIZE

### 10. Porting DS4's IQ2/Q2 dp4a or ROCm WMMA kernels

Different model, quantization, wave/tensor-core behavior, and quality budget. It neither fixes dense N=32 nor advances the already-floored NVFP4 expert path.

### 11. Replacing resident hot weights with unified-memory demand paging

Unified physical RAM is not uniform performance. Page faults, shared-fabric contention, and unpredictable admission latency conflict with fucina's measured deterministic serving lead.

### 12. Calling whole-session files an online SSD KV tier

DS4 has excellent durable checkpoints, but no attention-time SSD KV pager. Fucina should not invest based on a capability DS4 does not demonstrate.

### 13. FP16/FP8 distributed activation transport in the default path

DS4 reports no significant gain; it violates byte identity. Keep only as an explicitly approximate future capacity mode, if ever.

### 14. Distributed decode for speed

DS4 measures a 19.4% local-to-distributed decode regression even on Thunderbolt 5. Use distribution for fit/capacity and prefill, not autoregressive speed.

### 15. DS4's single-graph-worker serving model

This would directly surrender fucina's 11/12 concurrency-cell advantage. DS4's simplicity is appropriate to its local one-live-session regime, not fucina's measured product target.

### 16. Rax in inference or scheduling

DS4 uses it for exact DSML protocol maps only. No evidence supports placing a radix tree in fucina's GPU, KV-page, graph-key, or expert hot paths.

### 17. MTP/speculative decoding as a near-term project

DS4 itself reports only a slight speedup. Fucina's graph replay and byte-identical gate make the opportunity cost especially poor.

### 18. Reopening accepted/floored kernel work without new evidence

DS4 provides no new kernel class for dense N=32, no escape from Q4_K dp4a's latency/register wall, and no credible improvement over 80–85% peak-BW NVFP4 grouped experts. Elegance is not counter-evidence to measurement.

---

# E. Evidence appendix

## E.1 Architecture and philosophy

- `README.md:1-31` — narrow DeepSeek V4 engine, supported Metal/CUDA/ROCm backends.
- `README.md:34-69` — one-model vertical, official-vector validation, disk/session goals, CPU as diagnostics.
- `AGENT.md:1-14` — narrow implementation and production-path goals.
- `AGENT.md:16-32` — correctness before speed, no unexplained drift, small/readable C.
- `AGENT.md:35-47` — source layout and backend responsibilities.

## E.2 Prefill

- `ds4.c:5221` onward — batched Q8 prefill: quantize token activations, then scan weights.
- `ds4.c:8115-8230` — default prefill FFN organization and profiling breakdown.
- `ds4.c:9330-9449` — batch Q/KV projection, streaming cache updates, fresh-prefix parallel branch.
- `ds4.c:9913-10063` — CPU layer-major prefill and compressor finalization.
- `ds4.c:10306-10499` — separate persistent decode and batched-prefill tensors; per-layer compressed capacities.
- `ds4.c:11090-11290` — prefill-cap-based allocation and graph readiness.
- `ds4.c:11563-11635` — Metal selected-address eligibility and token thresholds.
- `ds4.c:11635-11693` — CUDA selected-address eligibility for Q4/IQ2 routed experts.
- `ds4.c:11918-12079` — selected expert uniqueness/byte profiling.
- `ds4_distributed.c:3096-3109` — N/N+1 pipelined prefill and ACK-only intermediate chunks.
- `ds4_distributed.c:3290-3415` — flow control, prefix hash validation, final result handling.
- `ds4_distributed.c:3421-3500` — pipeline eligibility, chunk cap, bounded window validation.
- `speed-bench/gb10.csv:2-33` — 2K–65K incremental prefill and decode measurements.

## E.3 SSD expert streaming and CUDA memory

- `ds4_ssd.c:14-111` — argument parsing and automatic 80% memory-budget plan.
- `ds4_ssd.c:113-181` — bounded touch/`mlock` diagnostic; not the expert streamer itself.
- `ds4.c:750-829` — expert histograms, adjacency, and simulated capacities.
- `ds4_streaming_hotlist.inc:1` onward — generated hit/weight-sorted layer/expert preload list.
- `ds4.c:11514-11561` — prefill page-in/readahead/pread/madvise controls.
- `ds4.c:11696-11916` — ROCm full-layer prefill load and decode-cache seeding.
- `ds4_cuda.cu:132-184` — selected cache and age-tagged complete-expert slots.
- `ds4_cuda.cu:218-222` — four selected-expert staging buffers/events and upload stream.
- `ds4_cuda.cu:874-952` — ATS/HMM memory advice and asynchronous prefetch.
- `ds4_cuda.cu:967-995` — source/file-page discard after copies.
- `ds4_cuda.cu:1014-1053` — four pinned model staging slots and events.
- `ds4_cuda.cu:1055-1123` — robust pread and aligned direct-I/O fallback.
- `ds4_cuda.cu:1145-1212` — CUDA model arena sizing/suballocation and safe direct-pointer rule.

## E.4 Durable KV/session store

- `ds4_kvstore.h:13-15` — default budget and six-hour hit half-life.
- `ds4_kvstore.h:29-60` — entry identity, reason, model/quant/context metadata.
- `ds4_kvstore.h:118-170` — byte-prefix and exact-token APIs, eviction/load/store entry points.
- `ds4_kvstore.c:30-49` — stable-boundary defaults and eviction constants.
- `ds4_kvstore.c:151-169` — default options.
- `ds4_kvstore.c:484-603` — decayed utility-density score, anchor prior, superseded-prefix penalty, victim loop.
- `ds4_kvstore.c:676-746` — trim/alignment and naturally reached continued checkpoints.
- `ds4_kvstore.c:806-1085` — staged payload, size/budget check, temp write, atomic rename.
- `ds4_kvstore.c:1190-1239` — longest rendered-byte-prefix lookup under model/quant/context constraints.
- `ds4_kvstore.c:1240-1338` — exact payload restore plus suffix tokenization and hit accounting.
- `ds4.c:10328-10379` — raw ring, compressed caches, and frontier state that snapshots must retain.

## E.5 Kernels and allocation

- `rocm/ds4_rocm_moe.cuh:1-80` — packed IQ2/Q2 integer dot helpers.
- `rocm/ds4_rocm_moe.cuh:1032` onward — stable pair ordering in expert buckets.
- `rocm/ds4_rocm_moe.cuh:3535-3600` — routed MoE rocWMMA kernel.
- `rocm/ds4_rocm_indexer.cuh:128-213` — 16×16×16 WMMA indexer scores.
- `rocm/ds4_rocm_indexer.cuh:794-801` — indexer WMMA launch.
- `rocm/ds4_rocm_matmul.cuh:285-295` — four-wave batch WMMA launch choice.
- `ds4_cuda.cu:485-790` — budgeted Q8→F16/F32 caches and failure fallback.
- `ds4_cuda.cu:2365-2395` — managed runtime tensor allocation rationale on unified-memory systems.
- `ds4_server.c:7670-7934` — `rax` exact-ID and exact-DSML-block maps; no inference use.

## E.6 Distributed and unified-memory evidence

- `ds4_distributed.h:9-17` — distribution is an engine backend behind normal session calls.
- `ds4_distributed.h:65-124` — route readiness, sync/eval, topology-neutral payload save/load.
- `README.md`, Distributed Inference section (approximately `README.md:260-460`) — topology, 4096-token microbatch pipeline, measured 1.38–1.85× prefill, 19.4% decode loss, link comparison, FP16/FP8 transport result.
- `STRIXHALO.md:1-16` — gfx1151/128 GB target and ROCm setup.
- `STRIXHALO.md:45-88` — GPU-visible aperture enlargement to roughly 126,976 MB.
- `STRIXHALO.md:100-113` — recommended quant and OOM warning.
- `ds4_cuda.cu:874-952` — GB10-relevant pageable-memory/HMM prefetch path.
- `speed-bench/gb10.csv:2-33` — measured GB10 prefill, generation, and KV footprint scaling.

## Final assessment

DS4 is elegant because it makes model semantics, storage, and session behavior explicit. That elegance is most valuable to fucina where it exposes **policy invariants**: stable boundaries, exact computational identity, bounded pipelines, topology-neutral state, and empirically driven expert residency. It is least valuable where it reflects a different machine/model regime: IQ2/Q2 kernels, ROCm wave tuning, demand-access unified memory, serial serving, or capacity-first SSD behavior.

Fucina should preserve its measured CUDA vertical. Adopt DS4's low-risk policy ideas, use its distributed invariants to harden Phase E, and allow exactly one kernel experiment—the clean-prefix GDN schedule—to proceed only behind frontier-by-frontier byte equivalence. Everything else requires new evidence stronger than fucina's current measurements.
