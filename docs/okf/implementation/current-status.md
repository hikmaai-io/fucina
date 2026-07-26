---
type: Implementation Status
title: Fucina implementation status at 480f1b8
description: Dated repository and validation assessment for the current main branch.
tags: [status, tests, git, maturity]
status: draft
stale_after: 2026-08-01
generated: { by: openai-codex/gpt-5.6-sol, at: 2026-07-25T17:28:46+02:00 }
snapshot_commit: 480f1b85722754d0e692321082890b6103fe56d8
sources:
  - id: git
    resource: https://github.com/hikmaai-io/fucina/commit/480f1b85722754d0e692321082890b6103fe56d8
    title: Inspected commit
  - id: makefile
    resource: ../../../Makefile
    title: Build and test gates
  - id: ci
    resource: ../../../.github/workflows/ci.yml
    title: Hosted CI workflow
  - id: test
    resource: ../../../internal/server/batch/scheduler_test.go
    title: Batch scheduler tests
---

# Repository state

* The implementation snapshot is local `main` at `480f1b8`, containing Qwen HTTP persistence and the integrated Phase-E CUDA/TCP boundary; `origin/main` remains at `39a96db`.
* Build outputs and benchmark databases present in the worktree are ignored by Git as intended.

# Validation executed on 2026-07-25

| Check | Result | Scope |
|---|---|---|
| `go test ./internal/server/... ./internal/session/...` | **PASS** | HTTP, scheduler/batch, and session format |
| `go test -race -count=1 ./internal/server/... ./internal/session/...` | **PASS** | Includes `internal/server/batch` |
| `go vet ./internal/server/... ./internal/session/...` | **PASS** | Changed pure-Go packages |
| Qwen HTTP restart GPU gate | **PASS** | Restored 11 cached tokens; prefilled only 7 new tokens |
| Phase-E split-layer CUDA gate | **PASS** | 8/8 byte-identical frontiers and 8/8 FP8 oracle tokens on Qwen3.5-9B |
| Phase-E separate-process TCP loopback | **PASS** | `[0,16)` coordinator plus `[16,32)` final worker generated ` Paris.` on one GB10 |
| Full Qwen GPU umbrella | **PASS** | Dense+MoE parity, state, shard, chunk, multiseq prefill, clean GDN, head, and MoE engine |
| MoE graph self-test | **PASS** | Every ragged row B=3 vs B=1 and graph-on/off 24/24; self-chain PASS; engine + standalone oracle 8/8 |
| Qwen3.6 MoE quality/performance | **80/100 router-only; 78/100 final** | 12/100 before; final decode median 59.0 tok/s, no loss versus 46–53 baseline |
| Gemma regression set | **No new signature** | paged device PASS; legacy dense bench and E4B batch retain their documented historical failures |
| Full Go and race gates | **PASS** | `go test ./... -count=1` and `make check`; local lint tool unavailable and skipped |

# Race-gate finding resolved

The unlocked `mockEngine` counter reads in `TestBatchedAdmissionCancellation` now use locked accessors. The cancellation test also joins deferred scheduler teardown before asserting no live slots. The targeted race suite, including `internal/server/batch`, is green.[^test]

[^test]: Batch scheduler tests

# Overall verdict

Fucina is far beyond a prototype in core Qwen/Gemma inference and GB10 optimization, but it remains an experimental lab engine rather than a release-clean product. The historical Qwen MoE self-test failure is now resolved on hardware and the router correctness fix retains decode performance. Release constraints remain: TC-60 fails, known Gemma/E4B gates remain red, and GPU validation is not continuous CI. Batched/Qwen structured output was closed on 2026-07-26 with 5/5 schema-valid Qwen3.6 FP8 hardware requests.
