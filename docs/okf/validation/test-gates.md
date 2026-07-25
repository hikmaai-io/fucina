---
type: Validation Playbook
title: Fucina test and quality gates
description: Layered validation from pure-Go tests through real-model GB10 parity and performance gates.
tags: [testing, ci, cuda, race, parity]
status: draft
stale_after: 2026-08-01
generated: { by: openai-codex/gpt-5.6-sol, at: 2026-07-25T17:28:46+02:00 }
snapshot_commit: 480f1b85722754d0e692321082890b6103fe56d8
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
| Go unit/integration | `go test ./internal/server/... ./internal/session/...` | CPU | Passed 2026-07-25 |
| Targeted race gate | `go test -race -count=1 ./internal/server/... ./internal/session/...` | CPU | Passed, including batch scheduler |
| Host CUDA-adjacent tests | `make paged-kv-test`, detector/plan/allocation tests | C++ compiler; no model for some targets | Defined, not rerun in this inspection |
| Build/smoke | `make all`, `make smoke` | CUDA 13 + GB10 + checkpoint | Not rerun |
| Qwen HTTP persistence | `make qwen35-http-session-restart-test` | GB10 + official Qwen3.5-9B-FP8 | **PASS**: restart restored 11 cached tokens and prefilled only 7 new tokens |
| Phase-E layer frontier | `make qwen35-shard-test` | GB10 + official Qwen3.5-9B-FP8 | PASS: 8/8 byte-identical frontier steps, 8/8 oracle tokens |
| Phase-E TCP loopback | Separate coordinator/worker processes, ranges `0:16`/`16:32` | One shared GB10 | **PASS**: generated ` Paris.`; 21.7 tok/s prefill, 31.9 tok/s decode; not a two-host speed gate |
| Qwen real-model correctness | `make gpu-gates` and specialized `qwen35-*` targets | GB10 + downloaded dense/MoE checkpoints | Historical pass evidence; not rerun |
| Performance protection | `scripts/protection_gate.py`, benchmark protocol | Quiescent GB10 and comparison runtime | Historical evidence; not rerun |

# Required interpretation

A normal unit pass does not replace the race detector, and historical benchmark logs do not prove the current worktree was rebuilt from source. The prior test-fixture race is fixed and the targeted race command is green. Both the Phase-E CUDA layer frontier and Qwen HTTP process-restart path are validated on the official Qwen3.5 checkpoint.

# CI discrepancy

Hosted CI claims to mirror the pure-Go gate but omits `internal/server/batch` from `PKGS`, while the Makefile includes it. It also excludes grammar, session, and distributed packages. The package list should be defined once or checked for equality to prevent future drift.

# Recommended acceptance sequence

1. `gofmt` and `go vet` over all pure-Go packages.
2. `go test ./...` and a race-enabled pure-Go package set that includes scheduler, grammar, session, and dist.
3. Host C++ metadata/allocation/paged-cache tests.
4. Forced CUDA archive rebuild and cgo relink.
5. Relevant real-model correctness gates under a GPU lock.
6. Performance protection only after correctness is green.
