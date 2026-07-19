# Gemma on GB10: current-source capability and fresh evidence

Date: 2026-07-20 evidence directory (runs began 2026-07-19 local time)<br>
Hardware: NVIDIA GB10, 48 SM, `sm_121a`, CUDA 13, 128 GiB unified LPDDR<br>
Fucina tree: `39a96dbd4856f394821021efa10ef31848ad2581`<br>
Raw evidence: [`benchmark-evidence/results/2026-07-20-gemma-gb10/`](../benchmark-evidence/results/2026-07-20-gemma-gb10/)

## Publication rule

A number in the raw directory is not automatically a performance claim. A cell is publishable only
when its artifact/format class is explicit, the fixed greedy corpus passes, the relevant engine
gates pass, and three independent starts have acceptable variance. Different HF/GGUF or
quantization formats are **FORMAT/SYSTEM** comparisons, not kernel parity. A failed or missing
artifact is reported as `MISSING`; it is never replaced by a convenient checkpoint.

## Current-source correction

The old statement “Gemma batch mode has no MTP” is false for this tree.

- `cuda/gemma4_kernels.cu` implements paged multi-sequence target decode with per-B CUDA graphs,
  a B-row paged MTP drafter (`mtp_forward_paged_batched` / `mtp_draft_paged_batched`), and
  `step_batch_spec_impl`, which flattens each slot's anchor plus drafts into one ragged target
  verify, accepts only the longest target-matching prefix, and commits only accepted state.
- `internal/engine/cuda/bridge.go` exposes `StepBatchSpec`; `BatchAdapter.StepBatch` selects it when
  an assistant is loaded. The scheduler's `SpecBatchEngine` path handles variable-length committed
  runs and stops safely in the middle of a run.
- The verify row budget is 32 (`B + Σdrafts`). Learned MTP depth defaults to at most 6 and
  auto-disables when fewer than two draft rows per slot fit. High-concurrency batches therefore
  converge to plain decode. Learned MTP self-drafting is greedy-only; target sampling remains
  lossless but does not use that learned draft round.
- Dense prompt-lookup speculation is separately default-on in the batch scheduler. `--spec=false`
  does **not** create a plain batch baseline; `FUCINA_NO_BATCH_SPEC=1` is required.
- Gemma continuous batching is still opt-in. That is a product/default decision pending evidence,
  not evidence that paged MTP is absent.

The source guarantee is structurally lossless: every emitted token comes from the target verify,
and the committed KV position plus MTP recurrence are advanced only through the accepted row.
Unit tests in `internal/server/batch/spec_lossless_test.go` exercise the scheduler contract. This
is source/test evidence, not a substitute for the failing and passing GPU gates listed in the raw
result report.

## vLLM source capability

The audited `/tmp/vllm` tree is `5f8e73cb8b8d41f7a2a5168cddf5b772888fa991`. Its Gemma-4
implementation includes dense and MoE blocks, PLE, KV sharing, optional KV-sharing fast prefill,
LoRA/PP interfaces, and a Gemma-4 MTP model/proposer that maps Q-only draft attention to target KV
cache groups. The runnable local image is a **different revision**:
`vllm/vllm-openai@sha256:9c719f…`, vLLM commit
`74b5964f02c7e023fadd3004cfac8a61c52eef1f`. Source breadth at `5f8e73cb` is not used as proof of
runtime behavior for the image.

The image has no matching local HF assistant artifact. The local dense assistant is GGUF Q8_0,
which vLLM's audited tree does not load as a Gemma MTP model. vLLM MTP is therefore `MISSING`, not
silently compared with another assistant.

| Capability | fucina current source | audited vLLM source / pinned image |
|---|---|---|
| Dense / E4B / MoE | dense Gemma engine; separate E4B PLE engine | unified dense, PLE/KV-sharing E4B, and MoE blocks |
| Continuous batch | paged custom CUDA, per-B graphs, 32-row verify ceiling | V1 scheduler, chunked prefill, compile + CUDA graphs |
| Attention in measured launch | custom split-K paged kernels | forced `TRITON_ATTN`; FlashInfer used for sampling, not attention |
| KV precision | dense FP8; E4B BF16 | measured dense FP8; E4B plan would use BF16 for parity |
| Prefix | source scheduler/cache interfaces; fresh dense counters stayed zero | enabled; fresh run reused 4,736 tokens |
| Drafting | prompt lookup and dense paged MTP; E4B MTP single-sequence only | Gemma-4 MTP proposer/model, but no matching local HF assistant |
| Adapters/distribution | one GB10, no LoRA/TP/PP product path | LoRA and pipeline/tensor-parallel framework interfaces |
| Compile strategy | ahead-of-time nvcc custom kernels + eager graph capture | Torch/Inductor compile cache + full/piecewise CUDA graphs |

Source breadth is not measured speed. In particular, the image revision differs from the audited
checkout, and the exact launch log—not the source table—is authoritative for measured backend and
flags.

## Artifact and gate findings

- The intended exact dense artifact was local
  `RedHatAI/gemma-4-12B-it-NVFP4@a1d2478…`, SHA-256
  `2a476980…afaf27`. Fucina failed before readiness with
  `NVFP4 shape mismatch L5 P1 (packed 983040 vs 7864320, scale 122880 vs 983040)`.
  The exact dense head-to-head cell stopped there.
- The available fucina Q4_0-QAT GGUF and vLLM BF16 safetensors are different formats. Any numbers
  for them belong only in the FORMAT/SYSTEM table.
- `paged-kv-device-test` failed its global-attention paged-vs-contiguous check (0.0112 max error),
  although paged-vs-host was `1.19e-07` and sliding tests passed.
- The dense batch self-test reported `seq_add(batch) failed`, including a direct rerun with an
  explicit four-slot pool. The legacy `make bench` harness also failed its batch/sampling markers.
  Its one plain-vs-batch text probe happened to match, which does not override the failed gates.
- Full build (`make lib libdg fucina`) and `go test ./... -count=1` passed. Runtime directories
  remain unmodified on this branch.

## E4B maturity (separate from dense Gemma)

Current source is ahead of old README/runtime comments here too. E4B has an HTTP server bridge,
slot-based greedy continuous batching, BF16 and GGUF loaders, PLE/KV sharing, and a
single-sequence greedy MTP stream. It is not dense-product parity:

- the batch adapter is greedy and does not apply per-request sampling parameters;
- the engine is capped at eight total slots and reserves slot 0 for single-sequence APIs;
- the E4B assistant is not used by the batch adapter;
- server cancellation, memory at target concurrency, exact prefix behavior, and a three-start
  competitive matrix must pass before calling it a multi-tenant product.

Fresh gates make the maturity boundary concrete:

- config/PLE, BF16 load, Q4_0 GGUF load/forward sanity, and the hybrid NVFP4+FP8 kernel gate pass;
- continuous-batch parity **fails**: the third divergent sequence differs at 2 of 8 generated
  positions, so no E4B server throughput winner is reported;
- HF forward and generation oracle gates are `MISSING` because `/tmp/e4b_ref.bin` and
  `/tmp/e4b_gen_ref.bin` plus a repository dump producer are absent;
- the three assistant Makefile targets omit `libdg.a` and fail to link. Re-running the unchanged
  test sources with the already-built archive linked passes assistant load, 160-token plain-vs-MTP
  byte identity, and 160-token server-style stream byte identity. That isolates a gate-wiring bug,
  not an assistant arithmetic failure. The standalone gate measured 55.9 tok/s plain and
  167.6 tok/s MTP (3.00×), which is capability evidence only—not a multi-tenant server claim.

E4B results and gate outcomes are kept in a separate table in the result report. They do not repair
or stand in for the missing exact dense checkpoint cell.

## Fresh measured bottleneck

The one-start cross-engine row is FORMAT/SYSTEM only (Q4_0-QAT target versus BF16 target), so it
names no winner. It is still diagnostic: at N=1 fucina produced 17.887 aggregate completion tok/s
versus vLLM 7.618, while at N=32 fucina produced 97.303 versus vLLM 270.311. More importantly,
fucina TTFT was 37.30 s for a 3,501-token prompt and 261.89 s at 15,877 tokens; vLLM was 1.31 s
and 6.31 s. Fucina's identical 4,751-token prefix was 53.38 s cold and 53.37 s warm with zero
prefix counters, whereas vLLM was 1.73 s cold and 0.285 s warm with 4,736 cached prompt tokens.
The dominant measured runtime bottleneck is therefore the dense unsupported/long paged-prefill
fallback and absent effective batch-prefix reuse, not decode attention. Full raw arrays and the
comparability warning are in the result README.

## MTP productization gate

The acceptance rule is unchanged:

1. at least **+15% aggregate completion throughput** in one declared target concurrency band;
2. no protected cell regresses by more than **5%**;
3. target committed token/state remains lossless;
4. quality passes beside every speed number; and
5. three independent starts make the floor non-flaky.

Three-start same-target results are decisive: MTP changes aggregate throughput by +99.63% at N=1,
+37.01% at N=2, −22.75% at N=4, −23.24% at N=8, −0.46% at N=16, and −0.42% at N=32. CV is
0.08–0.42%, and all seven quality output hashes match plain across all three starts (token-event
hashes also match where the field was captured). MTP therefore meets the +15% target-band rule but
**fails productization** on the >5% N=4/N=8 regressions. The 32-row verify budget's intermediate-B
draft cost is the immediate scheduling bottleneck; learned MTP should decline those shapes rather
than run unconditionally.

This branch ships `scripts/gemma_protection_gate.py`: default mode is report-only and `--enforce`
fails when variance/starts are insufficient or a protected cell crosses the 5% floor. This data is
statistically enforceable and correctly fails N=4/N=8. It is not wired as a merge gate because the
broader GPU correctness gates are red. Batch `/metrics` also reports zero drafted/accepted counters
while MTP is active, so actual accepted-length/rate telemetry is `MISSING`, not reconstructed from
SSE timing. The missing exact dense artifact independently prevents a cross-engine merge claim.

## Next runtime brief (not implemented on this branch)

**Top blocker:** make the intended exact dense artifact load and restore the existing GPU
correctness gates before tuning throughput.

1. Reproduce the RedHat compressed-tensors L5/P1 shape mismatch with a tensor-name/shape manifest;
   determine whether the checkpoint's `attention_k_eq_v`/unified naming changed or the loader's
   expected packed geometry is stale. Add a loader-only shape gate for this exact SHA-256.
2. Root-cause the global paged-vs-contiguous 0.0112 discrepancy. Because paged-vs-host is accurate,
   first verify whether the contiguous test oracle still matches current proportional-RoPE/global
   geometry; do not weaken the tolerance.
3. Make `FUCINA_BATCH_SELFTEST` provision its required slots and fail the process on self-test
   failure. A logged failure with exit status zero is not a gate.
4. Only after 1–3 pass, rerun this exact manifest. Then profile MTP drafted/accepted length by B,
   verify-row utilization, drafter time, target-verify time, and long-prefill fallback. Tune
   `FUCINA_BATCH_DRAFT_K`/minimum depth only if the +15%/-5% productization rule passes.

No runtime change from this brief is included in the evidence branch.
