---
type: Risk Register
title: Known implementation and documentation gaps
description: Prioritized correctness, coverage, feature, architecture, and documentation gaps.
tags: [risks, gaps, technical-debt, documentation]
status: draft
stale_after: 2026-08-01
generated: { by: openai-codex/gpt-5.6-sol, at: 2026-07-25T10:33:01+02:00 }
snapshot_commit: 39a96dbd4856f394821021efa10ef31848ad2581
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

1. **`make check` is red.** The batch cancellation test performs unlocked reads of locked mock counters. Fix the fixture and rerun the race suite.
2. **Local and hosted gates have drifted.** The Makefile race suite includes `internal/server/batch`; hosted CI's `PKGS` omits it. Hosted CI also omits pure-Go `internal/grammar`, `internal/session`, and `internal/dist`, even though these carry structured-output, persistence, and protocol logic.[^ci]
3. **GPU evidence is not continuous CI.** Correctness depends on a GB10 and large local checkpoints. Historical gates are extensive, but no automated public runner validates current CUDA changes.

[^ci]: Hosted CI workflow

# P1 — user-visible feature gaps

* `response_format` and `json_schema` return HTTP 501 under continuous batching; all supported Qwen3.5/3.6 serving uses that path.
* E4B continuous batching is greedy and ignores per-request temperature/top-k/top-p/min-p/seed parameters.
* HTTP disk sessions are single-flight only. Qwen hybrid persistence works in the paged REPL but is not threaded through the server scheduler.
* Gemma's MTP assistant is not implemented per slot in the continuous-batch path.
* Embeddings remain a stub that returns an empty data list.

# P2 — architecture completion

* Tensor metadata, alternate weight representations, ownership, and scratch remain partly implicit in a monolithic CUDA runtime; the canonical-plan migration is still in progress.[^tensor]
* Phase B has a complete policy pipeline but no accepted sub-4-bit kernel or default policy that improves on the incumbent.
* Phase C SSD expert streaming is a memory fallback with synchronous misses and advisory prefetch, not a default throughput path.
* Phase E lacks the real CUDA shard runner, partial-forward ABI, range loading, orchestration, and multi-node evidence.[^dist]

[^tensor]: Tensor management refactor
[^dist]: Distributed inference status

# P1 — documentation consistency

* README, changelog, and `docs/qwen-models.md` still advertise legacy Qwen3/Qwen3MoE support after their detector path was removed.
* `docs/remaining-plans.md` still presents E4B CUDA-graph and JSON-schema work as pending even though both landed; only the E4B batched-sampling portion remains pending.
* The `v0.1.0` tag exists and is hundreds of commits behind `main`, while `CHANGELOG.md` still labels `0.1.0` “unreleased.”
* Several older “ACTIVE” plan documents are superseded by the concluded Qwen mission but are not consistently marked as archival.
