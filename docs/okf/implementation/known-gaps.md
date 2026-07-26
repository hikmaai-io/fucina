---
type: Risk Register
title: Known implementation and documentation gaps
description: Prioritized correctness, coverage, feature, architecture, and documentation gaps.
tags: [risks, gaps, technical-debt, documentation]
status: draft
stale_after: 2026-08-01
generated: { by: openai-codex/gpt-5.6-sol, at: 2026-07-25T17:28:46+02:00 }
snapshot_commit: 480f1b85722754d0e692321082890b6103fe56d8
sources:
  - id: ci
    resource: ../../../.github/workflows/ci.yml
    title: Hosted CI workflow
  - id: makefile
    resource: ../../../Makefile
    title: Local gates
  - id: remaining
    resource: ../../remaining-plans.md
    title: Remaining feature plans
  - id: tensor
    resource: ../../tensor-management-refactor-plan.md
    title: Tensor management refactor
  - id: sessions
    resource: ../../session-persistence.md
    title: Session persistence status
  - id: dist
    resource: ../../phase-e-distributed.md
    title: Distributed inference status
---

# P0 — quality gate

1. **Local and hosted gates have drifted.** The Makefile race suite includes `internal/server/batch`; hosted CI's `PKGS` omits it. Hosted CI also omits pure-Go `internal/grammar`, `internal/session`, and `internal/dist`, even though these carry structured-output, persistence, and protocol logic.[^ci]
2. **GPU evidence is not continuous CI.** Correctness depends on a GB10 and large local checkpoints. The Qwen HTTP restart and Phase-E frontier targets pass locally, but no automated public runner validates CUDA changes.

[^ci]: Hosted CI workflow

# P1 — user-visible feature gaps

* **Closed 2026-07-26:** `response_format` and `json_schema` now use independent per-slot grammars under continuous batching; TC-64/65/66/67/69-style Qwen3.6 FP8 hardware requests passed 5/5 schema validation. See [`../../batched-structured-output.md`](../../batched-structured-output.md).
* TC-60 cross-turn sleeper injection remains a safety-critical model-level failure in the final Qwen3.6 MoE qualification.
* Qwen base>0 continuation prefill uses the exact scalar attention path. The faster tensor-core candidate is quarantined after correct router replay exposed only 2/25 continuation agreement versus scalar/one-shot 25/25.
* E4B continuous batching is greedy and ignores per-request temperature/top-k/top-p/min-p/seed parameters. The historical short-prompt batch parity mismatch is resolved; see [`../../e4b-batch-parity.md`](../../e4b-batch-parity.md). Its HF forward/generation gates require `/tmp/e4b_ref.bin` and `/tmp/e4b_gen_ref.bin`, but no pinned producer or artifacts are in the repository.
* Gemma's MTP assistant is not implemented per slot in the continuous-batch path.
* Embeddings remain a stub that returns an empty data list.

# P2 — architecture completion

* Tensor metadata, alternate weight representations, ownership, and scratch remain partly implicit in a monolithic CUDA runtime; the canonical-plan migration is still in progress.[^tensor]
* Phase B has a complete policy pipeline but no accepted sub-4-bit kernel or default policy that improves on the incumbent.
* Phase C SSD expert streaming is a memory fallback with synchronous misses and advisory prefetch, not a default throughput path.
* Phase E now has a real CUDA shard runner, partial-forward ABI, CLI orchestration, and a same-engine split-layer parity gate. It still loads the full checkpoint on every process and lacks true two-host parity/throughput evidence, batched prefill, and RDMA.[^dist]

[^tensor]: Tensor management refactor
[^dist]: Distributed inference status

# P1 — documentation consistency

* README, changelog, and `docs/qwen-models.md` still advertise legacy Qwen3/Qwen3MoE support after their detector path was removed.
* `docs/remaining-plans.md` still presents E4B CUDA-graph and JSON-schema work as pending even though both landed; only the E4B batched-sampling portion remains pending.
* The `v0.1.0` tag exists and is hundreds of commits behind `main`, while `CHANGELOG.md` still labels `0.1.0` “unreleased.”
* Several older “ACTIVE” plan documents are superseded by the concluded Qwen mission but are not consistently marked as archival.
