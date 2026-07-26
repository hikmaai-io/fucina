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
| Go unit/integration | `go test ./... -count=1` | CPU + cgo link | **PASS** 2026-07-25 |
| Targeted race gate | `make check` | CPU | **PASS**, including batch scheduler; local `golangci-lint` unavailable and explicitly skipped |
| Host CUDA-adjacent tests | `make paged-kv-test`, metadata helpers | C++ compiler | PASS through `make test` |
| Build/smoke | clean `make fucina`, `make test` | CUDA 13 + GB10 | PASS; note `--test-parser`/`--test-cuda` still print placeholder messages |
| Qwen HTTP persistence | `make qwen35-http-session-restart-test` | GB10 + official Qwen3.5-9B-FP8 | **PASS**: restart restored 11 cached tokens and prefilled only 7 new tokens |
| Phase-E layer frontier | `make qwen35-shard-test` | GB10 + official Qwen3.5-9B-FP8 | **PASS**: 8/8 byte-identical frontier steps, 8/8 oracle tokens |
| Qwen real-model correctness | `make gpu-gates` | GB10 + dense/MoE checkpoints | **PASS** on final branch, including exact continuation 25/25 |
| MoE engine + standalone oracle | `make qwen35-moe-fp8-engine-test qwen35-moe-fp8-test` | GB10 + official Qwen3.5-35B-A3B-FP8 | **PASS**: engine self-test PASS, both oracle paths 8/8 |
| MoE graph replay | engine self-test, three ragged rows | GB10 | **PASS**: B=3 vs B=1 and graph-on vs graph-off are 24/24 for every row; self-chain PASS |
| Gemma dense paged parity | `make paged-kv-device-test` | GB10 | **PASS**; global max error 0.000968, sliding exact |
| Gemma legacy bench | `make bench MODEL=gemma-4-12b-it-qat-q4_0.gguf` | GB10 | **Known historical FAIL unchanged**: self-test marker checks fail, while plain-vs-batch output remains byte-identical |
| E4B foundation/load/NVFP4 | targeted `e4b-*` gates | GB10 + E4B checkpoints | **PASS**, including ragged batch-vs-independent parity for lengths 1..5 (8/8 tokens per row) |
| E4B MTP | `e4b-mtp-load-test`, `e4b-spec-test`, `e4b-spec-stream-test` | GB10 + Q4_0 target/assistant | **PASS** after fixing Make target linkage; 160/160 byte-identical spec and stream |
| E4B HF oracle artifacts | `e4b-fwd-test`, `e4b-gen-test` | GB10 + BF16 checkpoint + `/tmp` refs | **Unavailable**: repository has no producer/artifacts for `e4b_ref.bin` and `e4b_gen_ref.bin` |
| Performance protection | 3×128-step Qwen3.6 MoE decode microbench | Quiescent GB10 under flock | B=1 **59.0 tok/s median**, above 46–53 tok/s baseline |

# Required interpretation

A normal unit pass does not replace the race detector, and historical benchmark logs do not prove the current worktree was rebuilt from source. This inspection used a clean CUDA rebuild and reran both. The historical MoE line `qwen-gates.log:417` (`oracle 8/8, self-test FAIL`, with 6/24 row/graph agreement) is **resolved** on the final branch: all three rows are 24/24, graph-on/off is 24/24, self-chain passes, and both oracle gates are 8/8. The retained legacy dense Gemma failure predates this Qwen fix and remains a known gap. The separate E4B batch mismatch has since been resolved by making the batched numerical path identical to independent decode.

# CI discrepancy

Hosted CI claims to mirror the pure-Go gate but omits `internal/server/batch` from `PKGS`, while the Makefile includes it. It also excludes grammar, session, and distributed packages. The package list should be defined once or checked for equality to prevent future drift.

# Recommended acceptance sequence

1. `gofmt` and `go vet` over all pure-Go packages.
2. `go test ./...` and a race-enabled pure-Go package set that includes scheduler, grammar, session, and dist.
3. Host C++ metadata/allocation/paged-cache tests.
4. Forced CUDA archive rebuild and cgo relink.
5. Relevant real-model correctness gates under a GPU lock.
6. Performance protection only after correctness is green.
