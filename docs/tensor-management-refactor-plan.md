# Tensor management refactor plan

Status: in progress  
Scope: CUDA model loading, resident weight representation, scratch ownership, and dispatch  
Primary target: Qwen3.5/3.6 on one GB10; design must remain usable by Gemma and diffusion

## Executive summary

Tensor management is now the main structural risk in the CUDA backend. The runtime is fast and its
correctness gates are strong, but adding a producer or quantization layout currently requires edits
across checkpoint naming, host conversion, a monolithic engine struct, offset assignment, pointer
lookup, kernel dispatch, memory accounting, and teardown.

The refactor should introduce one canonical `WeightRef`/`TensorSpec`, one host-side `ModelPlan`, and
one transactional allocation registry. It must **not** rewrite kernels or alter hot-path arithmetic.
Migrate Qwen first, projection family by projection family, while keeping the existing benchmark and
bit-identity gates green.

## Current-state findings

### 1. A tensor is represented differently at every stage

Today a logical projection may be represented as:

- a safetensors name plus `st::Tensor` mmap view;
- a GGUF absolute byte offset;
- a `uint64_t` offset in `eng->tensors`;
- an optional `uint8_t fmt_*` field;
- a raw pointer into `d_weights`;
- a pointer in a separate allocation/override table;
- an FP8 scale found by binary-searching a pointer-to-scale table;
- NVFP4 packed weights plus separate linear/swizzled scales and global multipliers;
- a persistent BF16 prefill cache of the same logical weight.

`weight_fp8()`, `use_packed_q4*()`, `wscale_fp8()`, `gemv_w()`, and `gemv_batched_w()` reconstruct
this metadata from engine-global format, pointer ranges, offset arithmetic, and side tables. The
representation is implicit rather than carried by the tensor itself.

### 2. Whole-model format and per-tensor format are conflated

`tensor_format_t` contains both model-level identities and per-weight encodings. Qwen mixed
checkpoints set `eng->format = FORMAT_FP8_BLOCK` even when their mixers are Q4_K, experts are
NVFP4, norms are F32, embeddings are BF16/F32, and the head has BF16 plus Q8_0 representations.
Per-projection `fmt_*` fields partially repair this, but experts, embeddings, and heads use separate
flags and pointer conventions.

This makes every new mixed checkpoint a special-case exercise and allows an omitted override to
silently route a tensor through the wrong kernel.

### 3. Ownership is spread across the engine and teardown code

`gemma4_kernels.cu` is over 15,000 lines and has hundreds of allocation/free sites. Qwen loading has
additional allocations and host temporaries in `qwen35_backend.cuh`. Ownership rules are encoded in
comments and conditions such as:

- the LM head may alias the embedding;
- overrides do not live in `d_weights`;
- linear and swizzled NVFP4 scales have different consumers;
- some resources are freed only when `fp4_ready` is set;
- Qwen slot state is transactional, while much of model loading is procedural.

The exact memory ledger is useful, but it is independently reconstructed after allocation rather
than emitted by the allocation owner. That permits drift between actual ownership and accounting.

### 4. Loading mixes schema, transformation, allocation, and upload

`qwen35_fp8_fill_engine()` currently performs all of the following in one procedure:

1. resolve producer-specific names;
2. validate tensor presence and dimensions;
3. convert BF16/F32/FP8/NVFP4 on the host;
4. decide serving formats;
5. create expert slabs through GPU work;
6. build descriptors for a bulk device blob;
7. allocate and upload;
8. create side tables and alternate head/embedding copies;
9. produce telemetry and the model ledger.

A failure late in the process occurs after substantial work and relies on outer teardown to recover.
There is no inspectable dry-run plan showing every source tensor, transformation, destination,
consumer, and byte count before CUDA allocation begins.

### 5. Scratch is indexed rather than typed

The Qwen runtime improved ownership by moving state into `qwen35_runtime_state`, but shared scratch
is still exposed as `sb[24]`, `wc[layer][12]`, and `fp4_w[layer][12]`. Meaning comes from comments
and call-site index constants. This is compact but makes lifetime overlap and capacity assumptions
hard to audit.

### 6. File boundaries follow history, not ownership

Qwen state/runtime/backend fragments are included into `gemma4_kernels.cu` because they need the
private engine definition and static kernel helpers. The split improves navigation but does not
provide module boundaries. Checkpoint adapters, allocation policy, tensor metadata, runtime
workspace, and kernel launch dispatch remain coupled to one translation unit.

## Target architecture

### A. Canonical tensor metadata

Introduce an internal, trivially copyable descriptor used directly by runtime dispatch:

```cpp
enum class WeightEncoding : uint8_t {
    F32, BF16, Q8_0, Q4_0, Q4_K, Q6_K,
    FP8_BLOCK_128, FP8_ROW, NVFP4_LINEAR, NVFP4_SWIZZLED
};

enum class TensorLayout : uint8_t {
    ROW_MAJOR, GGML_NATIVE, Q4K_PACKED, NVFP4_SCALE_LINEAR, NVFP4_SCALE_SWIZZLED
};

struct WeightRef {
    const uint8_t *data;
    const void    *scale;
    const float   *global_scale;
    int32_t out_dim, in_dim;
    WeightEncoding encoding;
    TensorLayout layout;
    uint16_t flags;       // tied, packed, primary, cache, etc.
};
```

Requirements:

- `WeightRef` contains everything needed to select a kernel; no pointer-to-scale lookup.
- Dimensions and layout travel with the pointer.
- A model-level format may remain for API reporting, but runtime dispatch must use `WeightRef`.
- Expert slabs use a sibling `ExpertWeightRef` with expert count and per-expert strides rather than
  pretending to be ordinary offsets.
- Embedding and LM-head references use the same ownership/alias model as projections.

The hot path still reads raw pointers from a fixed descriptor. This adds no allocation, map lookup,
or virtual dispatch inside captured graphs.

### B. Producer adapters output a canonical host specification

Keep container parsing (`st::Model`, GGUF parsing) separate from model semantics. Add adapters:

- `QwenOfficialFp8Adapter`
- `QwenModelOptAdapter`
- `QwenCompressedTensorsAdapter`
- existing GGUF adapter

Each adapter resolves names and returns a `SourceTensor` plus `QuantSpec`:

```cpp
struct SourceTensor {
    std::string logical_name;
    const void *data;
    size_t bytes;
    DType dtype;
    Shape shape;
    QuantSpec quant;
};
```

Adapters normalize producer conventions, including reciprocal global scales and scalar versus
per-row FP8 scales. They do **not** allocate CUDA memory or choose a runtime kernel.

### C. Build a complete `ModelPlan` before allocating

A planner consumes canonical source tensors, model geometry, hardware capabilities, and memory
policy. It emits immutable entries such as:

```cpp
struct PlannedTensor {
    TensorId id;
    SourceTensor source;
    Transform transform;       // COPY, BF16_TO_F32, FP8_TO_Q4K, NVFP4_REBASE, ...
    WeightEncoding destination;
    AllocationClass arena;
    size_t bytes, alignment;
    TensorId aliases;
};
```

The plan must:

- validate all required tensors, shapes, dtypes, scale conventions, and group sizes first;
- identify primary and alternate representations explicitly;
- compute exact model, scale, head, embedding, expert, and cache bytes;
- reject incompatible mixed layers before any `cudaMalloc`;
- be printable as deterministic JSON/text for support and regression tests;
- provide the authoritative input to capacity planning and the allocation ledger.

### D. Transactional allocation registry

Add a small internal registry around CUDA allocations:

```cpp
class DeviceAllocationSet {
  DeviceBuffer allocate(size_t bytes, AllocationTag tag);
  DeviceArena  arena(size_t bytes, AllocationTag tag);
  void commit();
  ~DeviceAllocationSet(); // rolls back unless committed
};
```

Properties:

- every allocation records pointer, bytes, tag, and ownership;
- aliases are non-owning handles and cannot be double-freed;
- failed transforms/uploads automatically roll back;
- teardown iterates the committed registry instead of reproducing allocation conditions;
- memory telemetry and ledgers are generated from the same records;
- fault-injection tests can fail the Nth allocation and prove complete rollback.

Use a few large aligned arenas where lifetime is uniform:

1. immutable core weights;
2. quantization scales/metadata;
3. expert slabs;
4. embedding/head stores;
5. persistent prefill caches.

Keep standalone allocations only where independent lazy growth or eviction is required.

### E. Explicit representation sets and cache policy

A logical tensor may have several physical forms. Represent that directly:

```cpp
struct WeightSet {
    WeightRef primary;
    optional<WeightRef> decode;
    optional<WeightRef> prefill;
};
```

Examples:

- source FP8, packed Q4_K decode, BF16 prefill cache;
- native NVFP4 packed values, linear scales for GEMV, swizzled scales for CUTLASS;
- BF16 head plus Q8_0 approximate-search index;
- tied embedding/head alias.

Each representation records its provenance and consumer. Cache construction becomes a planner
choice with exact bytes, not a collection of readiness flags and unrelated pointer arrays.

### F. Typed workspaces

Replace numeric scratch arrays incrementally with named views:

```cpp
struct QwenDecodeWorkspace { float *x, *norm, *q, *k, *v, ...; };
struct QwenPrefillWorkspace { ... };
struct MoeWorkspace { ... };
```

Back them with one arena per lifetime/capture domain. The planner computes maximum sizes from model
geometry and configured tile/capacity. Named views make overlap intentional and auditable while
preserving the current low allocation count.

### G. Module boundaries

After descriptors and ownership are stable, split by responsibility:

- `tensor_types.h` — encoding/layout/descriptor types;
- `device_alloc.{h,cu}` — registry, arenas, accounting;
- `model_plan.{h,cc}` — host-only plan and validation;
- `qwen_checkpoint_adapter.{h,cc}` — naming and producer normalization;
- `qwen_weight_builder.cu` — transforms and upload;
- `qwen_workspace.{h,cu}` — typed runtime arenas;
- existing kernel files — math only;
- `gemma4_engine.cu` — orchestration and public ABI.

Do not start by physically splitting the 15K-line translation unit. First remove hidden dependencies
through descriptors and ownership APIs; moving code then becomes mechanical and reviewable.

## Execution status

- **Phase 0 — evidence:** partial. Publication KPI, TTFT, allocation-ledger, and oracle artifacts are
  pinned under `benchmark-evidence/results/`; deterministic per-producer tensor snapshots and the
  malformed-checkpoint matrix remain open.
- **Phase 1 — descriptor foundation:** complete. Canonical descriptor types are present and all
  Qwen attention, GDN, and dense-FFN projection paths use descriptors. Official FP8, GGUF, and both
  Unsloth variants pass oracle, graph, state, and long-context gates.
- **Phase 2 — host model planner:** in progress. The immutable host-only `ModelPlan`, validation,
  alias handling, exact aligned arena totals, and deterministic JSON serialization are implemented.
  Official FP8, ModelOpt, and compressed-tensors source conventions now pass a complete host-only
  shape/dtype/scale preflight before engine CUDA allocations; making plan entries authoritative for
  upload descriptors and exact ledger generation remains open.
- **Phases 3–6:** not started.

The performance KPI remains **>64 / >105 / >150 tok/s at 1/2/4 streams**. Descriptor-only changes
must first preserve the `<1%` phase gate; kernel optimization remains sequenced after ownership work.

## Phased execution plan

### Phase 0 — Freeze behavior and evidence

- Add a machine-readable tensor-plan snapshot for official FP8, ModelOpt, Unsloth Fast, and
  Unsloth accurate checkpoints.
- Pin current 1/2/4/32 throughput, TTFT, allocation ledger, and oracle outputs.
- Add startup-failure tests for missing scales, wrong shape, wrong dtype, and mixed expert formats.

Exit gate: no refactor yet; snapshots reproduce current residency and dispatch decisions.

### Phase 1 — Descriptor foundation

- Introduce `WeightEncoding`, `TensorLayout`, `WeightRef`, and `ExpertWeightRef`.
- Populate descriptors alongside existing offsets and format bytes.
- Convert `gemv_w()` and `gemv_batched_w()` to descriptor overloads.
- Migrate one low-risk family first: attention/GDN mixer projections.
- Keep compatibility wrappers so token outputs remain bit-identical.

Exit gates:

- no hot-path pointer lookup for migrated projections;
- graph capture remains valid;
- all Qwen oracle/row-independence/long-context gates pass;
- throughput regression below 1%.

### Phase 2 — Host-only model planner

- Move producer naming and quant normalization behind adapters.
- Build and validate a complete Qwen `ModelPlan` before CUDA allocation.
- Emit deterministic plan JSON containing source, transform, destination encoding, dimensions,
  bytes, and consumer.
- Replace ad-hoc `D.push_back` construction with plan entries.

Exit gates:

- malformed checkpoints fail before the first CUDA allocation;
- Fast and accurate Unsloth plans differ only where their tensor groups actually differ;
- planned bytes equal the current exact ledger.

### Phase 3 — Transactional ownership and accounting

- Implement `DeviceAllocationSet` and aligned arenas.
- Migrate immutable core, scales, and host temporary ownership.
- Generate ledgers from allocation records.
- Add Nth-allocation and Nth-upload fault injection.
- Migrate engine teardown to registry ownership, retaining explicit destruction only for CUDA
  handles/events/graphs.

Exit gates:

- every injected failure leaves no owned CUDA allocation;
- tied embedding/head aliases cannot double-free;
- actual committed bytes equal recorded bytes by allocation class;
- C32 admission uses the registry ledger unchanged.

### Phase 4 — Experts, heads, and alternate representations

- Migrate expert slabs to `ExpertWeightRef`.
- Represent native NVFP4 linear/swizzled scales as named views of one logical weight.
- Represent BF16/Q8 head and embedding aliases as `WeightSet`s.
- Move prefill-cache decisions into the plan and expose cache bytes/consumers in telemetry.
- Remove `fp8_scale_tab`, weight override scans, and pointer-range format inference once their last
  users migrate.

Exit gates:

- no binary search or linear override scan in decode;
- no whole-model format branch selects a mixed projection kernel;
- all representation duplicates are visible in the plan and ledger.

### Phase 5 — Typed workspaces and file decomposition

- Replace `q35.sb[24]` and projection-number arrays with named workspace views.
- Consolidate eager workspace allocation into transactional arenas.
- Establish internal headers for the engine-private tensor and workspace APIs.
- Split checkpoint planning, weight building, and runtime orchestration out of
  `gemma4_kernels.cu`.

Exit gates:

- kernel files do not parse checkpoint names or own model allocations;
- loader files do not select launch geometry;
- numeric scratch indices are gone from Qwen production paths;
- compile time and binary size do not materially regress.

### Phase 6 — Cleanup

Delete compatibility paths only after all callers migrate:

- per-layer raw offsets for migrated Qwen tensors;
- `fmt_*` bytes superseded by descriptors;
- `fp8_scale_tab`;
- weight override pointer scans;
- duplicated conditional teardown;
- obsolete readiness flags whose state is represented by a valid `WeightRef`/`WeightSet`.

No dead compatibility code should remain.

## Performance work after the refactor

The requested KPIs are **>64 / >105 / >150 tok/s at 1/2/4 streams**. Current validated results are
about 60.9 / 94.0 / 142.1 tok/s under the publication protocol.

Three plausible-looking changes were measured and rejected rather than shipped:

- serving the native NVFP4 shared expert directly reduced fixed single-stream decode to 57.5 tok/s;
- a weight-read-once batched Q8 approximate head reduced N=2/N=4 to roughly 78/98 tok/s;
- compacting CUTLASS expert groups from 256 to `B*topk` reduced N=2/N=4 to roughly 81/127 tok/s,
  indicating the extra groups were helping occupancy/scheduling despite zero-M work.

These results argue against format-driven rewrites without whole-step profiling. After descriptor
migration, add per-component decode CUDA-event telemetry (head, mixer, routed experts, shared
expert, attention/state, other) under the existing opt-in timing policy. Optimize only the measured
largest term, and keep every experiment behind compile-time development code until it beats all
three KPI rows and passes bit-identity gates. Candidate investigations are:

1. CUTLASS grouped expert tile/cluster scheduling at 8/16/32 assignments while retaining enough
   groups for occupancy;
2. shared-expert FP8 gate/up fusion and launch geometry without changing its stored numerics;
3. Q4_K mixer batched kernels for K=2 and K=4, using fixed synthetic decode to separate scheduling
   from prompt/stop-length effects;
4. LM-head candidate search using a genuinely vectorized DP4A Q8 kernel, not scalar Q8 FMA;
5. overlap opportunities only after event telemetry proves independent work exists.

## Required validation matrix

Every phase must run:

- `make nvfp4-test`
- `make qwen35-moe-fp8-engine-test`
- `make qwen36-unsloth-nvfp4-test` on Fast and accurate checkpoints
- `make qwen35-state-test`
- `make qwen35-longctx-test`
- `go test ./...`
- `git diff --check`

Before merging a runtime phase, rerun `benchmark-evidence/PROTOCOL.md` without `--timings` and
compare 1/2/4/32 throughput, cold/warm TTFT, model ledger, admitted slots, and errors. Diagnostic
timing runs are separate artifacts because their synchronization changes throughput.

## Non-goals

- No multi-device tensor parallelism for the single-GB10 target.
- No speculative decoding for sparse MoE.
- No new runtime flags as the final delivery mechanism.
- No kernel arithmetic changes in the ownership/organization phases.
- No big-bang rewrite of the engine or checkpoint loaders.
