---
type: Roadmap
title: Evidence-ranked fucina roadmap
description: Recommended next work ordered by correctness, product coverage, and implementation dependency.
tags: [roadmap, quality, serving, distributed]
status: draft
stale_after: 2026-08-25
generated: { by: openai-codex/gpt-5.6-sol, at: 2026-07-25T17:28:46+02:00 }
snapshot_commit: 480f1b85722754d0e692321082890b6103fe56d8
sources:
  - id: gaps
    resource: implementation/known-gaps.md
    title: Known implementation gaps
  - id: mission
    resource: ../sota-gb10-qwen3-mission-plan.md
    title: Concluded Qwen3.5 GB10 mission
  - id: tensor-plan
    resource: ../tensor-management-refactor-plan.md
    title: Tensor management refactor plan
  - id: phase-e
    resource: ../phase-e-distributed.md
    title: Distributed inference plan
---

# Priority order

1. **Close the Qwen HTTP-session hardware gate.** Run `make qwen35-http-session-restart-test` on the GB10 with the official 9B FP8 checkpoint and retain the exact restore/suffix evidence.
2. **Unify local and hosted pure-Go gates.** Include scheduler, grammar, session, and distributed packages from one shared package list so CI cannot silently omit them.
3. **Reconcile public documentation with code.** Remove legacy Qwen3/Qwen3MoE support claims, mark superseded plans, and reconcile the `v0.1.0` tag with the changelog's “unreleased” heading.
4. **Close mandatory-batching feature gaps.** Add constrained `response_format`/`json_schema` support to batching, or provide a correctness-preserving constrained route for Qwen3.5/3.6.
5. **Finish E4B per-sequence sampling.** Its batch adapter currently ignores request sampling parameters and remains greedy.
6. **Continue tensor-management refactoring.** Finish canonical tensor metadata, transactional ownership, and typed scratch without changing hot-path arithmetic.
7. **Productize Phase E.** Add range-filtered weight residency, true two-host parity/performance gates, then batched prefill and RDMA; the CUDA partial-forward runner and TCP CLI are now integrated.

# Performance policy

The single-node Qwen throughput mission is concluded with an explicit determinism trade-off.[^mission] New optimization work should begin only from fresh profiler evidence and must retain the byte-identity/protection gates unless introduced as an explicit, default-off mode.

[^mission]: Concluded Qwen3.5 GB10 mission

# Dependencies

The detailed evidence behind these priorities is in [known gaps](implementation/known-gaps.md), [DS4 pillars](capabilities/ds4-pillars.md), and [test gates](validation/test-gates.md).
