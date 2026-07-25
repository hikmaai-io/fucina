---
type: Validation Playbook
title: Fucina test and quality gates
description: Layered validation from pure-Go tests through real-model GB10 parity and performance gates.
tags: [testing, ci, cuda, race, parity]
status: draft
stale_after: 2026-08-01
generated: { by: openai-codex/gpt-5.6-sol, at: 2026-07-25T10:33:01+02:00 }
snapshot_commit: 39a96dbd4856f394821021efa10ef31848ad2581
sources:
  - id: makefile
    resource: ../../../Makefile
    title: Test targets
  - id: ci
    resource: ../../../.github/workflows/ci.yml
    title: Hosted CI
  - id: evidence
    resource: ../../../benchmark-evidence/PROTOCOL.md
    title: Benchmark protocol
---

# Gate layers

| Layer | Representative command | Environment | Current inspection result |
|---|---|---|---|
| Go unit/integration | `go test ./...` | CPU plus existing cgo link artifacts | Passed 2026-07-25 |
| Static/race gate | `make check` | CPU; local lint tool optional | Failed on batch test-fixture race |
| Host CUDA-adjacent tests | `make paged-kv-test`, detector/plan/allocation tests | C++ compiler; no model for some targets | Defined, not rerun in this inspection |
| Build/smoke | `make all`, `make smoke` | CUDA 13 + GB10 + checkpoint | Not rerun |
| Qwen real-model correctness | `make gpu-gates` and specialized `qwen35-*` targets | GB10 + downloaded dense/MoE checkpoints | Historical pass evidence; not rerun |
| Performance protection | `scripts/protection_gate.py`, benchmark protocol | Quiescent GB10 and comparison runtime | Historical evidence; not rerun |

# Required interpretation

A normal `go test ./...` pass does not replace the race detector, and historical benchmark logs do not prove the current worktree was rebuilt from source. Conversely, the current race report points to a test fixture and does not by itself demonstrate a production data race.

# CI discrepancy

Hosted CI claims to mirror the pure-Go gate but omits `internal/server/batch` from `PKGS`, while the Makefile includes it. It also excludes grammar, session, and distributed packages. The package list should be defined once or checked for equality to prevent future drift.

# Recommended acceptance sequence

1. `gofmt` and `go vet` over all pure-Go packages.
2. `go test ./...` and a race-enabled pure-Go package set that includes scheduler, grammar, session, and dist.
3. Host C++ metadata/allocation/paged-cache tests.
4. Forced CUDA archive rebuild and cgo relink.
5. Relevant real-model correctness gates under a GPU lock.
6. Performance protection only after correctness is green.
