# vLLM Subsystem Analysis for fucina (GB10 / DGX Spark)

**Source tree:** `/tmp/vllm` @ `5f8e73cb` ("[Bugfix] Guard mixed-dtype allreduce RMSNorm quant fusions (#48330)"), analyzed 2026-07-12.
**Target of applicability assessment:** fucina — hand-written Go+CUDA single-node engine, NVIDIA GB10 (48 SMs, sm121, ~273 GB/s LPDDR5X unified memory, 128 GB, native FP4/FP8), serving Qwen3.5 (hybrid 3×GatedDeltaNet + 1×full-attn per block; 35B-A3B MoE and 9B dense) with FP8/NVFP4 weights.

**Name resolution up front (per instructions):**

| Requested name | Exists literally? | Resolved to |
|---|---|---|
| "runner v2" | **Yes** — `VLLM_USE_V2_MODEL_RUNNER`, Model Runner V2 ("MRV2") | `vllm/v1/worker/gpu/` (entire package) + `docs/design/model_runner_v2.md` |
| "dspark" / "DSSpark" | **Yes** — `method="dspark"` speculative decoding | `vllm/v1/worker/gpu/spec_decode/dspark/`, `vllm/model_executor/models/qwen3_dspark.py`, `vllm/models/deepseek_v4/{nvidia,amd}/dspark.py` |
| "dsflash" | **No literal match** (`rg -ic "dsflash|ds_flash"` → 0 hits anywhere in the tree) | Closest match by intent: **DFlash** (`method="dflash"`), the parallel-drafting speculative-decoding subsystem that DSpark itself subclasses: `vllm/v1/worker/gpu/spec_decode/dflash/`, `vllm/v1/spec_decode/dflash.py`, `vllm/model_executor/models/qwen3_dflash.py`. Evidence for this resolution in §A.3. The other candidate, FlashMLA / DeepSeek MLA flash backends (`vllm/v1/attention/backends/mla/flashmla.py`), is assessed and dismissed there. |

---

## A. Per-Subsystem Analysis

### A.1 "runner v2" — Model Runner V2 (MRV2)

**What it is.** A ground-up rewrite of vLLM's GPU model runner, replacing the 7,751-line `vllm/v1/worker/gpu_model_runner.py` (V1) with a modular package. It is opt-in/auto-selected via `VLLM_USE_V2_MODEL_RUNNER` (`vllm/envs.py:271,1908`) and the `use_v2_model_runner` property (`vllm/config/vllm.py:531`). Certain features (DSpark, mixed-KV-group DFlash, diffusion models) *force* V2 (`vllm/config/vllm.py:536-552`).

**Where it lives.**

- Design doc: `docs/design/model_runner_v2.md`
- Core: `vllm/v1/worker/gpu/model_runner.py` (1,632 lines — vs 7,751 for V1)
- Persistent request state: `vllm/v1/worker/gpu/states.py` (`RequestState`)
- Input batch/buffers + Triton input-prep kernels: `vllm/v1/worker/gpu/input_batch.py`
- Staged-write / UVA buffers: `vllm/v1/worker/gpu/buffer_utils.py` (`StagedWriteTensor:114`, `UvaBuffer:44`, `UvaBufferPool:53`, `UvaBackedTensor:90`)
- Block tables + slot-mapping kernels: `vllm/v1/worker/gpu/block_table.py`
- Explicit CUDA-graph management: `vllm/v1/worker/gpu/cudagraph_utils.py` (`CudaGraphManager:112`, `BatchExecutionDescriptor:53`)
- Triton-native sampler: `vllm/v1/worker/gpu/sample/` (`gumbel.py`, `sampler.py`, `logprob.py`, …)
- Model-family-specific state: `vllm/v1/worker/gpu/model_states/` (notably `mamba_hybrid.py` for GDN/Mamba hybrids — directly relevant to Qwen3.5)
- Spec-decode integration: `vllm/v1/worker/gpu/spec_decode/`

**The generalized design pattern.** MRV2 is a bundle of ~8 orthogonal ideas (from the design doc and code):

1. **Decoupled persistent state vs. per-step inputs.** Each request gets a *permanent row index* in fixed-size `[max_num_reqs, …]` state tensors for its lifetime (`states.py:RequestState`, `free_indices` slot allocator at `states.py:28,97`). Per-step input tensors are *gathered* from persistent state on-GPU given the step's `idx_mapping` (batch_idx → req_state_idx, `input_batch.py:43`). No tensor-wide compaction/reordering when requests join/finish; preemption is treated as completion. This removes V1's `CachedRequestState` backup copies.
2. **Async-first, zero-sync hot loop.** The execution loop is "a CUDA stream with no CPU synchronization points"; the CPU only queues work, preparing step N+1 while step N runs. Races between CPU writes and in-flight async H2D copies are *eliminated structurally* rather than guarded by barriers: the CPU-side source-of-truth is unpinned; each step copies through a round-robin pool of pinned/UVA staging buffers sized to the number of in-flight steps (`UvaBufferPool`, `_DEFAULT_MAX_CONCURRENCY = 2`, `buffer_utils.py:18`).
3. **StagedWriteTensor: diff-based GPU state updates.** Large state (block tables, `all_token_ids`, `num_computed_tokens`) lives on GPU (or UVA); the CPU stages ragged row-diffs (`stage_write(row, start, values)`), packs them contiguously, and one Triton kernel applies all diffs (`buffer_utils.py:114-206`). One H2D copy + one kernel launch per step regardless of how many requests changed.
4. **GPU-native input preparation.** `input_ids`, `positions`, `query_start_loc`, `seq_lens`, slot mappings, and logits indices are computed *by Triton kernels on the GPU* from persistent state (`input_batch.py:186-612`: `prepare_prefill_inputs`, `prepare_pos_seq_lens`, `combine_sampled_and_draft_tokens`, `expand_idx_mapping`; `block_table.py:200+`: `_gather_block_tables_kernel`, `_compute_slot_mappings_kernel`). Crucially, the GPU can derive values the CPU *doesn't know yet* (last-step sampled/accepted tokens under async scheduling and spec decode) — the CPU only maintains an "optimistic upper bound" mirror (`states.py:60-62`).
5. **UVA for huge cold state.** `all_token_ids` (`max_num_reqs × max_model_len`, potentially GBs) is CPU-resident but GPU-addressable via UVA, so prefill-token gathers read host memory directly without a resident GPU copy (`states.py:30-38`).
6. **Triton-native sampling.** Gumbel-max sampling with stateless counter-based RNG keyed on `(seed, position)` — no softmax materialization, deterministic per (request, position) (`sample/gumbel.py:215`, `tl_rand64/tl_rand32` at :62-83). Top-k logprobs computed from logits *after* top-k selection to avoid full-vocab logprob materialization. Per-logit → per-request state indirection via `idx_mapping` inside kernels instead of tensor expansion.
7. **Explicit CUDA-graph management.** `CudaGraphManager` (`cudagraph_utils.py:112`) captures FULL graphs keyed by `BatchExecutionDescriptor(cg_mode, num_tokens, num_reqs, uniform_token_count, num_active_loras)` (:53). Dispatch is a compatibility lookup (`_is_compatible:76`: captured shape must dominate the runtime shape). `get_uniform_token_count` (:96) lets FULL graphs also serve uniform *multi-token* decode batches (spec decode: every request has exactly 1+K tokens). Capture goes through the *same* `execute_model` path as real steps (design doc §8: "no abuse of `dummy_run`"). Batch ordering is normalized before dispatch: `sort_batch_req_ids` (`model_runner.py:1626-1632`) orders decode → short-extend → prefill so uniform decodes lead.
8. **Split execute/sample for pipelined output processing.** `execute_model` (`model_runner.py:1129`) enqueues the forward and stashes an `ExecuteModelState` (:1617); `sample_tokens` (:1373) is a separate call, letting the engine overlap grammar/structured-output CPU work with the forward.

**What problem it solves.** Python/CPU overhead per step, CPU–GPU sync stalls, bookkeeping complexity of a persistent batch that doubles as model input, and implicit/fragile CUDA-graph state. It is fundamentally a *host-overhead amortization* architecture for a Python host driving a fast discrete GPU.

**Measured/claimed wins.** No quantified perf numbers are documented in-tree. The design doc claims qualitative wins ("substantial improvement", lower CPU overhead, better async overlap, lower peak memory for logprobs). The most concrete claim is structural: async scheduling with PP is fully supported only under V2 (`vllm/config/vllm.py:493-503`).

---

### A.2 "dspark" — DSpark semi-autoregressive block drafting

**What it is.** A speculative-decoding method (`method="dspark"`, `vllm/config/speculative.py:60,304-305`) that drafts a whole *block* of `num_speculative_tokens` tokens in **one parallel forward pass** of a draft model, then injects intra-block token dependency with a lightweight **sequential Markov head** — a low-rank `V×r / r×V` transition bias sampled left-to-right. It is a DeepSeek technique (checkpoints `deepseek-ai/DeepSeek-V4-Pro-DSpark`, `deepseek-ai/dspark_qwen3_8b_block7` — `tests/models/registry.py:1435-1446`) and is implemented **only in MRV2** (`vllm/config/vllm.py:536-544` forces V2 for dspark).

**Where it lives.**

- Speculator (execution): `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` (170 lines; subclasses `DFlashSpeculator`)
- Draft-model loading / weight sharing with target: `vllm/v1/worker/gpu/spec_decode/dspark/utils.py`
- Qwen3 draft architecture + `DSparkMarkovHead`: `vllm/model_executor/models/qwen3_dspark.py`
- DeepSeek-V4 variant (draft reuses target architecture, MTP-style, weights ship inside the target checkpoint): `vllm/models/deepseek_v4/nvidia/dspark.py`, `vllm/models/deepseek_v4/amd/dspark.py`
- Config plumbing/validation: `vllm/config/speculative.py:674-680` (weights from target checkpoint), :901-911 (architecture rewrite), :945-967 (hard requirement `num_speculative_tokens >= dspark_block_size`, e.g. 7)
- E2E test with measured acceptance stats: `tests/v1/e2e/spec_decode/test_spec_decode.py:1422-1470`

**The generalized design pattern.** DSpark = DFlash's parallel block drafting (§A.3) + a **cheap sequentializer**:

1. The parallel backbone emits hidden states for N query positions in one non-causal forward (context-KV precompute + query-block forward, inherited from DFlash). "Anchor-as-first-prediction": each request contributes exactly N query tokens (anchor + N−1 noise/mask tokens); every position is a prediction (`dspark/speculator.py:9-22,47-54`).
2. Base draft logits for all N positions come from one LM-head GEMM (`_sample_sequential`, `speculator.py:101-150`).
3. A **Markov head** — `markov_w1: [vocab, r]` embedding of the previously sampled token, `markov_w2: [r, draft_vocab]` projecting to a logit bias (`qwen3_dspark.py:36-67`) — is applied *sequentially* left-to-right: `logits_i += markov_bias(prev)`. The loop is N tiny ops (embedding lookup + skinny GEMM + Gumbel argmax per step), all captured **inside one FULL CUDA graph together with the backbone forward** (`speculator.py:24-26`).
4. Optional reduced draft vocabulary with a `d2t` (draft→target) id map; for probabilistic rejection sampling, draft logits are scattered into target-vocab columns (`speculator.py:76-99,120-135`).
5. Deterministic verification coupling: draft sampling uses the same stateless Gumbel keys `(seed, position)` the target verifier uses, offset so the draft's sample at position Q is verified with the predecessor's key (`speculator.py:126-136`).

Generalized: **"parallel draft for bandwidth, sequential micro-model for accuracy."** The expensive backbone runs once per block at batch-GEMM efficiency; the intra-block dependency that pure parallel drafting loses is recovered by a first-order Markov correction whose cost is negligible and whose sequential loop is fixed-shape (CUDA-graph-friendly).

**Measured/claimed wins (in-tree).** `tests/v1/e2e/spec_decode/test_spec_decode.py:1444-1462`: Qwen3-4B-FP8 target + `dspark_qwen3_4b_block7` draft on GSM8K at temperature 1.0, mean over 12 runs: **acceptance_rate ≈ 0.428, acceptance_len ≈ 3.99**, GSM8K accuracy preserved (0.801). I.e. ~4 target tokens materialized per target forward pass with a *single* draft forward pass per step.

---

### A.3 "dsflash" — resolved to **DFlash** (parallel-drafting / block-diffusion speculative decoding)

**Literal-name status.** The string `dsflash` / `ds_flash` does **not** appear anywhere in the tree (case-insensitive search → 0 files). This must be stated explicitly: there is no subsystem named dsflash.

**Why DFlash is the intended subsystem.** (a) "dsflash" is one transposition from "dflash", a first-class speculation method (`method="dflash"`, `vllm/config/speculative.py:59,848-849`); (b) the query grouped it with "dspark", and in this tree DSpark literally subclasses DFlash (`class DSparkSpeculator(DFlashSpeculator)`, `dspark/speculator.py:37`) — they are two members of one family; (c) the query hint "DeepSeek flash attention variants (e.g. MLA/flashmla)" points at the alternative candidate **FlashMLA** (`vllm/v1/attention/backends/mla/flashmla.py`) — but FlashMLA is an MLA-specific decode kernel for DeepSeek-architecture KV compression. fucina's Qwen3.5 has **no MLA layers** (GDN + standard full attention), and nothing in FlashMLA transfers to GDN. I therefore assess FlashMLA as not the intended subsystem and not applicable, and analyze DFlash.

**What it is.** DFlash ("block diffusion" drafting; paper arXiv:2602.06036, referenced at `tests/v1/e2e/spec_decode/test_spec_decode.py:1367`) drafts N tokens in **one non-causal forward pass** of a small draft transformer. Draft checkpoints exist for fucina's exact model family: **`z-lab/Qwen3.5-9B-DFlash`** (`vllm/model_executor/models/qwen3_dflash.py:77`) and `z-lab/Qwen3-Coder-Next-DFlash` for the GDN-hybrid Qwen3-Next (`tests/models/registry.py:1427-1434`).

**Where it lives.**

- MRV2 speculator: `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` (638 lines), `cudagraph.py` (111), `utils.py` (72)
- V1 proposer: `vllm/v1/spec_decode/dflash.py` (309 lines)
- Draft model with fused context-KV precompute: `vllm/model_executor/models/qwen3_dflash.py` (853 lines; also `laguna_dflash.py`)
- Scheduler interaction (lookahead block reservation): `vllm/v1/core/sched/scheduler.py:234-247`; `tests/v1/spec_decode/test_dflash_lookahead.py:98-129`

**The generalized design pattern.** Per decode step:

1. **Context-KV precompute (cross-attention trick).** The draft never re-runs its layers over the context. Instead, the *target model's hidden states* for this step's tokens are projected into the draft's K/V for **all draft layers at once**: KV-projection weights of every draft layer are stacked into one `[L·2·kv, hidden]` matrix at load time (`_build_fused_kv_buffers` / `_build_context_kv_buffers`, `qwen3_dflash.py:413-463`); then one fused GEMM, grouped per-layer K-norm, one batched RoPE over `L·num_ctx` rows, and per-layer KV-cache inserts (`precompute_and_store_context_kv`, `qwen3_dflash.py:521-593`). Runs **eagerly, outside the CUDA graph**, because context length varies per step.
2. **Fixed-shape query forward.** Each request contributes exactly `1 + N` query tokens (bonus token + N mask tokens embedded via a learned mask embedding / vocab row; `get_parallel_drafting_token_id`, `spec_decode/utils.py:55-76`). The draft attends non-causally from these queries to the precomputed context KV. Because every request has an identical query count, the batch is *uniform* and the whole draft forward **plus sampling** is captured as one FULL CUDA graph keyed by request count (`DFlashCudaGraphManager`, `dflash/cudagraph.py:62`; `dispatch_cg_and_sync_dp(..., uniform_token_count=num_query_per_req)`, `dflash/speculator.py:389-397`).
3. **One fused Triton input-prep kernel** builds *everything* for the draft step from the target batch on-GPU: query input_ids (splicing the last sampled token or next chunked-prefill token GPU-side — the CPU never knows which), positions, per-request seq_lens, query/context slot mappings from the block table, sample indices, and CUDA-graph-safe padding of every buffer to max shapes (`_prepare_dflash_inputs_kernel`, `dflash/speculator.py:424-570`). Rejected-token counts are consumed on-GPU (`num_rejected`), so draft prep never syncs with the host.
4. **Parallel sampling + rejection verification.** All N positions sampled in one Gumbel kernel with position-keyed seeds shared with the target's rejection sampler (`_generate_draft`, `speculator.py:198-231`; `rejection_sampler.py:43-160`), preserving distribution-correct speculative decoding (probabilistic mode) or greedy.
5. **Scheduler contract:** N+1 lookahead KV slots reserved per request so draft queries have cache slots (`scheduler.py:234-247`).

Generalized: **turn K sequential draft forwards (EAGLE-style) into 1 parallel forward with a constant, uniform shape** — trading some acceptance length for (a) K× fewer draft launches, (b) a fully CUDA-graphable draft step including sampling, and (c) reuse of the *target's* compute for draft context via cross-projection instead of running draft layers over the context.

**Measured/claimed wins (in-tree).** `tests/v1/e2e/spec_decode/test_spec_decode.py:1352-1420` asserts ≥95% of the paper's Table-1 acceptance lengths for Qwen3-8B + `z-lab/Qwen3-8B-DFlash-b16` at N=16: **acceptance_len 4.24 (MT-Bench), 6.50 (HumanEval), 6.54 (GSM8K)**. On a bandwidth-bound decode that is roughly a 4–6.5× reduction in target weight traffic per generated token, minus draft overhead.

---

## B. Applicability to fucina

Framing constants for GB10 that shape everything below — and where vLLM's assumptions break:

- **Decode is bandwidth-bound with a low ceiling (~273 GB/s).** Anything that reduces *bytes moved per generated token* (speculation, verification batching) is worth ~1:1 in tokens/s. Anything that reduces *host overhead per step* is worth far less than in vLLM: fucina is Go+CUDA (no Python interpreter tax), and GB10 step times are long relative to host prep (a 9B FP8 dense forward at B=1 is ~33 ms of weight traffic at peak; MRV2 fights per-step host costs that are large *relative to H100/B200 step times of 5–10 ms*, not GB10's).
- **Unified memory kills the H2D-copy problem class.** MRV2's UVA buffers, pinned-staging pools, and `StagedWriteTensor` H2D-diff machinery exist because discrete-GPU PCIe copies are expensive and asynchronous copies race with CPU writes. On GB10, CPU and GPU share LPDDR5X: "copy to GPU" can be a pointer pass, and `all_token_ids`-style UVA residency is simply *the default*. Roughly half of MRV2's cleverness solves a problem fucina does not have.
- **48 SMs still reward fixed-shape CUDA graphs and fewer/bigger kernels** — the fusion and graph-keying patterns transfer even where the copy patterns don't.

### B.1 MRV2 → fucina

| MRV2 idea | fucina mapping | Verdict |
|---|---|---|
| Permanent-row persistent state + per-step gather via `idx_mapping` | Slot-based request table in Go; per-step gather kernel producing packed inputs | **Adopt the pattern** if fucina still compacts/reorders batch state on admission/exit; it simplifies the batched-admission prefill path it just shipped and composes cleanly with per-request determinism (row identity stable for the request's lifetime). If fucina already does slot-stable state: nothing to do. |
| Async-first, zero-sync loop (CPU prepares step N+1 during step N) | Scheduler goroutine prepares next step's metadata while step N's graph runs | **Adopt the intent, skip the mechanism.** Overlap is nearly free in Go and protects TTFT under load. The race-elimination machinery (pinned pools, round-robin `UvaBufferPool`) reduces on GB10 to double-buffering two host-visible buffers plus a visibility fence. |
| `StagedWriteTensor` diff application | N/A as designed | **Skip.** Its purpose is minimizing PCIe H2D traffic + launch count for CPU-authored diffs. On unified memory the CPU writes the (host-visible) block table directly. Adopting it would be cargo-culting a PCIe workaround. |
| GPU-native input prep (kernels deriving inputs the CPU doesn't know yet) | One CUDA kernel that splices last-sampled/accepted tokens into next step's `input_ids` inside the decode graph | **Adopt narrowly — the key enabler for full CUDA-graph decode coverage and for spec decode.** fucina has graph decode "for some paths"; the blocker for graphing everything (including varying accepted-token counts once spec decode exists) is exactly this: inputs derived on-GPU from persistent state so replay needs no host knowledge of per-step outcomes. Determinism unaffected (pure function of state). |
| UVA for `all_token_ids` | Free on GB10 | Implied by unified memory. No action. |
| Stateless `(seed, position)` Gumbel sampler | CUDA kernel; counter-based RNG (Philox-style) keyed by (request seed, position) | **Adopt if/when spec decode lands.** Cleanest known way to keep byte-identical determinism *through* rejection sampling: draft and verifier derive identical Gumbel keys independently; no RNG state to serialize; replay-stable. |
| Explicit graph manager keyed by `(num_tokens, num_reqs, uniform_token_count)` with dominance dispatch | Go-side graph cache with the same key structure | **Adopt the keying scheme.** `uniform_token_count` is what distinguishes 1-token decode from (1+K)-token spec-decode batches; retrofitting later is painful. Also copy `sort_batch_req_ids`' decode-first ordering for mixed decode+admission steps. Cost small. |
| execute/sample split | Overlap sampling/detokenize with next forward | Optional; matters at high concurrency only. |

**Expected benefit against known losses.**

- **Dense aggregate N≥16 (vLLM +93% at N=32; 37% of peak at B=30):** MRV2 patterns will **not** close this — *unless* the profile shows inter-launch gaps or host prep serialized with the GPU. 37%-of-peak at B=30 on a 273 GB/s part is a kernel/graph problem (the ncu investigation is the right tool). One check first: if the B=30 dense path is not fully CUDA-graphed ("some paths"), extending graph coverage via GPU-native input prep is plausibly worth 10–30% there. If already graphed, MRV2 offers ~nothing for this loss.
- **MoE N=32 TTFT (866 vs 664 ms):** Prefill is compute/bandwidth bound; MRV2 contributes at most overlap of admission bookkeeping with the running batch — single-digit % unless admission currently stalls the GPU.
- **MoE N=2/4 cells:** Slot-stable state + graph-key dispatch reduce small-batch step overhead; modest.

**Cost: M** for selective adoption (GPU input-splice kernel + graph-key scheme + async prep ≈ a few weeks). A full MRV2-style rewrite: **L, not justified.**

**Risks.** (1) Over-adoption: importing PCIe-era buffer machinery onto unified memory adds complexity for zero win. (2) GPU-derived inputs hurt debuggability (state lives only on-device mid-step) — add a debug mode mirroring to host. (3) Async prep introduces exactly the CPU/GPU aliasing races MRV2's doc warns about; on unified memory *every* buffer is shared, so a disciplined double-buffer convention is mandatory or determinism bugs will be miserable.

### B.2 DSpark → fucina

**Concrete mapping.** DSpark is not adoptable standalone — it is DFlash + Markov head, and it requires a trained draft checkpoint with Markov weights. **No Qwen3.5 DSpark checkpoint exists in-tree** (DSpark checkpoints: DeepSeek-V4-Pro and Qwen3-8B/4B — `tests/models/registry.py:1435-1446`, `tests/v1/e2e/spec_decode/test_spec_decode.py:1425`). Training one is out of scope for an inference engine.

What *is* adoptable is the pattern, contingent on DFlash first:

- The Markov head is ~`2·V·r` extra parameters and an N-step loop of (embedding lookup + `[B,r]×[r,V]` GEMM + argmax) — trivially implementable inside fucina's CUDA-graph decode: fixed shapes, deterministic.
- The tradeoff it buys (acceptance_rate 0.43 at block 7 with *one* draft forward) is tuned for bandwidth-starved decode — exactly GB10's regime.
- **Hybrid-model caveat, in fucina's favor:** GatedDeltaNet state makes tree/multi-branch speculation awkward (recurrent state can't be cheaply forked), but DSpark/DFlash speculation is *linear* (one chain of N tokens) and verification is a single (1+N)-token pass. vLLM already runs GDN with spec decode via `num_accepted_tokens`-aware metadata (`gdn_attn.py:58-66,162-172`) and Mamba-state align/rollback (`model_states/mamba_hybrid.py:167-296`). fucina would need the same GDN checkpoint/rewind — the single hardest piece, shared with any speculative method.

**Expected benefit against known losses.** Speculation attacks a loss category *not on the list*: low-concurrency decode tokens/s. It does not help TTFT, N=32 aggregate, or the dense N≥16 gap — the DSD doc (`docs/features/speculative_decoding/dynamic_speculative_decoding.md:5`) explicitly warns SD goes net-negative beyond a critical batch size (verification multiplies effective batch by K). On a 48-SM, 273 GB/s part that critical batch is *low*: spec decode on GB10 is a B≤8 feature.

**Cost: L** (requires DFlash infrastructure + GDN rollback + a trained/converted draft; the Markov head itself is S on top of DFlash). **Risks:** no Qwen3.5 checkpoint; acceptance is workload-dependent; the block-size floor (`speculative.py:945-967`) prevents dialing K down gracefully at higher batch — you disable instead.

### B.3 DFlash → fucina

**Concrete mapping.** Best-aligned of the three: **a draft checkpoint for fucina's exact dense model exists** (`z-lab/Qwen3.5-9B-DFlash`, `qwen3_dflash.py:77`), and a GDN-hybrid sibling proves the pattern works with Qwen3-Next-style hybrids (`z-lab/Qwen3-Coder-Next-DFlash`, `tests/models/registry.py:1427-1434`).

Implementation shape in fucina:

1. Keep the last step's target hidden states for accepted tokens (fucina controls its forward — no aux-hidden plumbing needed if the checkpoint uses last-layer states; check `use_aux_hidden_state` in the draft config, `qwen3_dflash.py:337-342`).
2. Port `precompute_and_store_context_kv`: one fused GEMM `[num_ctx, hidden] × [hidden, L·2·kv]`, grouped K-norm, one batched RoPE, L cache inserts (`qwen3_dflash.py:521-593`). In Go+CUDA: ~3 kernels + one CUTLASS/cuBLASLt call. FP8 applies; an FP4 DFlash draft already exists for MiMo (`qwen3_dflash.py:75`), so NVFP4 drafts are plausible later.
3. Fixed `(1+N)`-token-per-request query forward over a small draft stack, non-causal attention against context KV. Uniform shape → one CUDA graph per batch bucket, *including sampling* — matches fucina's graph-decode philosophy exactly.
4. Verification: (1+N)-token target pass + rejection sampling. For GDN layers, add **state rollback**: snapshot recurrent state at step start, advance only through accepted tokens (vLLM reference: `gdn_attn.py`, `mamba_hybrid.py`).
5. KV admission: reserve N+1 lookahead slots per request (`scheduler.py:234-247`) — trivial in fucina's prefix-reuse cache.
6. Determinism: compatible. Greedy drafting is deterministic outright; probabilistic drafting stays byte-identical with stateless `(seed, position)` Gumbel keys (§B.1).

**Expected benefit against known losses.**

- Not the listed losses (same caveat as DSpark) — this is new upside, not gap-closing.
- **Where it wins:** low-concurrency decode. Back-of-envelope, dense 9B FP8 on GB10: ~9 GB weights/step → ~33 ms/step floor → ~30 tok/s ceiling at B=1. With acceptance_len ≈ 6.5 (code-like workloads) and draft ≈ 5–8% of target cost, effective ceiling → ~150+ tok/s. For chat-like workloads use ~4.2 (MT-Bench), still ~3.5×.
- For MoE-35B-A3B the per-token win is smaller (fewer active bytes/token), but verification batches expert traffic across N+1 tokens — which *also* moves the grouped NVFP4 GEMM up its efficiency curve at low concurrency, indirectly relevant to the **MoE N=2/4 cells** loss.
- Biggest available multiplier on interactive-session UX (fucina has session persistence + prefix reuse: interactive chat is a core workload).

**Cost: L overall, but decomposable:** (M) draft forward + context-KV precompute + rejection sampler for **dense 9B** (full-attn-only draft); (+M) GDN state rollback to extend to the 35B hybrid; (S) scheduler/KV lookahead. Dense-9B-first derisks everything.

**Risks.** (1) Checkpoint dependency: acceptance numbers are from z-lab checkpoints; conversion into fucina's weight format and exact matching of norms/RoPE is fiddly (the fused precompute asserts uniform RoPE params across draft layers, `qwen3_dflash.py:453-458`). (2) GDN rollback correctness threatens the byte-identical guarantee if saved-state vs recompute approaches are mixed inconsistently. (3) At B ≳ 8–16 on 48 SMs, verification inflates compute and speculation goes net-negative — needs a concurrency gate (vLLM's dynamic-SD `[start_bs, end_bs, K]` table is the reference design). (4) None of this touches the 866→664 ms TTFT or dense N≥16 gaps — don't let it displace the ncu work.

---

## C. Ranked Recommendation

1. **DFlash pattern (dense 9B first) — ADOPT, highest value.** Targets fucina's bandwidth ceiling — the one thing GB10 cannot be tuned around; existing Qwen3.5-9B draft checkpoint; fixed uniform shapes that slot directly into fucina's CUDA-graph decode; determinism-compatible sampling. Sequence: dense 9B greedy → probabilistic (stateless Gumbel) → GDN rollback for the 35B hybrid. Gate on concurrency (disable above ~B=8; tune empirically).
2. **MRV2, selectively — ADOPT three pieces (S/M):** (a) GPU-native input splicing so the *entire* decode step, including next-step input construction, is graph-replayable — also a prerequisite for DFlash's zero-sync draft loop; (b) the CUDA-graph key scheme with `uniform_token_count` + dominance dispatch — future-proofs the graph cache for spec decode; (c) permanent-slot request state if fucina doesn't already have it. Adopt *async-first* as a design rule, not the buffer machinery.
3. **DSpark — DEFER.** Adopt only as an increment on DFlash *if* a Qwen3.5 Markov-head checkpoint appears (or DFlash acceptance proves insufficient on real workloads at N=7–16). The Markov head itself is a small delta; the checkpoint is the blocker.

**Explicitly do NOT adopt:**

- **`StagedWriteTensor` / `UvaBufferPool` / pinned-staging machinery** — PCIe workarounds; unified memory makes direct host writes + a visibility fence strictly simpler. The clearest case of a vLLM assumption (discrete GPU, expensive H2D) breaking on GB10.
- **UVA-vs-GPU residency policy for token state** — a meaningless distinction on GB10.
- **Python-overhead-driven micro-architecture as an end in itself** (execute/sample RPC split, Triton-because-Python-is-slow input prep): fucina's Go host loop doesn't pay the interpreter tax MRV2 amortizes. Adopt GPU input prep only where it enables graph replay.
- **FlashMLA / MLA backends** (`vllm/v1/attention/backends/mla/`) — DeepSeek-architecture KV compression; Qwen3.5 has no MLA layers; irrelevant.
- **Whole-hog MRV2 rewrite** — fucina's listed losses (TTFT N=32, dense N≥16 at 37% of peak, MoE N=2/4) are kernel/scheduling problems, not host-architecture problems; the ncu investigation stays first in line.

**Adversarial self-check (where this analysis could be wrong):** If fucina's dense N≥16 gap is actually *launch-gap/host-serialization* (visible as inter-kernel bubbles in nsys), then MRV2's async-first + full-graph coverage jumps to #1 and would partially explain vLLM's +93% at N=32 — vLLM runs those batches under captured graphs with prep overlap. Spend one nsys trace to discriminate *before* starting DFlash. Also: DFlash acceptance lengths were measured on code/math benchmarks; free-form chat gives ~4.2 — size the speedup model on ~4, not 6.5. Finally, vLLM's in-tree numbers are acceptance stats, not end-to-end GB10 tokens/s; the bandwidth-model extrapolations in §B.3 are mine, not vLLM's claims.

---

## D. Evidence Appendix

**MRV2**

- `docs/design/model_runner_v2.md` — full design rationale (persistent batch, async-first, race elimination, StagedWriteTensor, UVA, Triton sampler, explicit CG management, dummy_run cleanup)
- `vllm/v1/worker/gpu/README.md:1-4` — "[Experimental] Model Runner V2"
- `vllm/v1/worker/gpu/model_runner.py:120` `GPUModelRunner`; `:856` `prepare_inputs`; `:1036` `prepare_attn`; `:1129` `execute_model`; `:1373` `sample_tokens`; `:1617` `ExecuteModelState`; `:1626-1632` `sort_batch_req_ids`; `:318-321` `decode_query_len = num_speculative_steps + new_sampled_per_step`
- `vllm/v1/worker/gpu/states.py:9-131` `RequestState` — slot allocator `free_indices` (`:28,97`), `all_token_ids` as UVA-backed `StagedWriteTensor` (`:30-38`), optimistic CPU mirror `num_computed_tokens_np` (`:60-62`)
- `vllm/v1/worker/gpu/buffer_utils.py:18` `_DEFAULT_MAX_CONCURRENCY=2`; `:26-41` `async_copy_to_gpu`; `:44-50` `UvaBuffer`; `:53-88` `UvaBufferPool` (round-robin race elimination); `:90-113` `UvaBackedTensor`; `:114-206` `StagedWriteTensor.stage_write/apply_write`; `:210+` `FusedStagedWriter`
- `vllm/v1/worker/gpu/input_batch.py:12-34` `InputBuffers`; `:37-102` `InputBatch` (`idx_mapping`, `expanded_idx_mapping`); `:186-612` Triton kernels: `prepare_prefill_inputs`, `prepare_pos_seq_lens`, `combine_sampled_and_draft_tokens`, `post_update`, `expand_idx_mapping`
- `vllm/v1/worker/gpu/block_table.py:107-199` staged writes + `gather_block_tables` / `compute_slot_mappings`; `:200,240` Triton kernels
- `vllm/v1/worker/gpu/cudagraph_utils.py:53-62` `BatchExecutionDescriptor`; `:76-93` `_is_compatible` (dominance dispatch); `:96-110` `get_uniform_token_count`; `:112` `CudaGraphManager`; `:428` `ModelCudaGraphManager`
- `vllm/v1/worker/gpu/sample/gumbel.py:62-83` stateless `tl_rand64/tl_rand32`; `:215` `gumbel_sample(logits, idx_mapping, temperature, seeds, pos, …)`
- `vllm/v1/worker/gpu/model_states/mamba_hybrid.py:35-66` `MambaHybridAttnMetadata` (`num_accepted_tokens`); `:167-296` preprocess/prepare_attn state align
- `vllm/envs.py:271,1908-1909` `VLLM_USE_V2_MODEL_RUNNER`; `vllm/config/vllm.py:531-563` auto-selection (dspark forces V2 `:540-544`; multi-KV-group DFlash `:546-549`); `:493-503` async+PP fully supported only on V2
- V1 contrast: `vllm/v1/worker/gpu_model_runner.py` — 7,751 lines monolith

**DSpark**

- `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:1-27` design docstring (anchor-as-first-prediction; sequential Markov sampling; FULL CUDA graph over backbone + sampling); `:37` `class DSparkSpeculator(DFlashSpeculator)`; `:76-99` reduced-vocab d2t scatter; `:101-150` `_sample_sequential`; `:152-170` `_generate_draft`
- `vllm/v1/worker/gpu/spec_decode/dspark/utils.py:15-68` `load_dspark_model` (non-causal draft config; embed/lm_head sharing with target)
- `vllm/model_executor/models/qwen3_dspark.py:36-67` `DSparkMarkovHead` (low-rank V×r / r×V transition bias); `:70-93` backbone = DFlash Qwen3 + Markov head; `:133-141` `map_draft_to_target`
- `vllm/models/deepseek_v4/nvidia/dspark.py:1-10` DSV4 variant (non-causal via sparse-attention top-k inclusion of future queries); `:71` multi-layer MTP-style draft
- `vllm/config/speculative.py:674-680` weights ship in target checkpoint; `:850-854` name-based method inference; `:901-911` DSV4 architecture rewrite; `:912-913` `parallel_drafting = True`; `:945-967` `num_speculative_tokens >= dspark_block_size`
- Measured: `tests/v1/e2e/spec_decode/test_spec_decode.py:1444-1462` — GSM8K temp=1.0: acceptance_rate mean 0.428, acceptance_len mean 3.994, accuracy 0.801 (12 runs)
- Checkpoints: `tests/models/registry.py:1435-1446`

**DFlash (resolution of "dsflash")**

- Literal absence: `rg -ic "dsflash|ds_flash"` → 0 matches tree-wide
- `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:32` `DFlashSpeculator`; `:47-50` `num_query_per_req = 1 + N`; `:253-421` `propose` (eager context precompute / graphed query forward; `dispatch_cg_and_sync_dp` with `uniform_token_count` `:389-397`); `:424-570` `_prepare_dflash_inputs_kernel` (fused GPU input prep incl. GPU-side bonus-token splice `:474-479`, CG-safe padding `:520-568`)
- `vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py:62` `DFlashCudaGraphManager` (FULL graphs, own attention metadata)
- `vllm/v1/spec_decode/dflash.py:23-309` V1 `DFlashProposer` (same pattern; separate stable-address query buffers for CG `:44-52`)
- `vllm/model_executor/models/qwen3_dflash.py:55-77` checkpoint ecosystem incl. `z-lab/Qwen3.5-9B-DFlash` (`:77`) and FP4 MiMo draft (`:75`); `:353-362` mask-token embedding; `:413-463` fused KV buffer construction (uniform-RoPE assertion `:453-458`); `:521-593` `precompute_and_store_context_kv`; `:712` `get_draft_kv_cache_layer_names`
- `vllm/v1/worker/gpu/spec_decode/utils.py:55-76` `get_parallel_drafting_token_id`
- `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:43-160` rejection sampling with shared Gumbel keys; `rejection_sampler_utils.py:864` `rejection_sample`
- Scheduler lookahead: `vllm/v1/core/sched/scheduler.py:234-247`; `tests/v1/spec_decode/test_dflash_lookahead.py:98-129`
- Measured: `tests/v1/e2e/spec_decode/test_spec_decode.py:1352-1420` — acceptance_len 4.24 (MT-Bench), 6.50 (HumanEval), 6.54 (GSM8K) from arXiv:2602.06036 Table 1, asserted at ≥95%
- GDN-hybrid draft exists: `tests/models/registry.py:1427-1434` (`z-lab/Qwen3-Coder-Next-DFlash` on `Qwen/Qwen3-Coder-Next`)
- Dismissed alternative: `vllm/v1/attention/backends/mla/flashmla.py:47-119` (FlashMLA — MLA/DeepSeek-only; N/A to Qwen3.5/GDN)
- SD batch-size warning + dynamic-K schema: `docs/features/speculative_decoding/dynamic_speculative_decoding.md:5,14-33`

**fucina-relevant model support in-tree**

- Qwen3.5 registered: `vllm/model_executor/models/registry.py:558-561` (`Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration`); MTP variants `:631-632`; DFlash/DSpark drafts `:593-596`
- GDN attention backend with spec-decode support: `vllm/v1/attention/backends/gdn_attn.py:42-66` (`num_spec_decodes`, `spec_state_indices_tensor`, `num_accepted_tokens`), `:82-97` builder (`UNIFORM_BATCH` CG support)
