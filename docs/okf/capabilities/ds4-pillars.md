---
type: Capability Matrix
title: DS4-inspired implementation pillars
description: Maturity of the Qwen3.5 Phase A-E roadmap generalized from the DS4 workflow.
tags: [qwen, calibration, quantization, residency, sessions, distributed]
status: draft
stale_after: 2026-08-25
generated: { by: openai-codex/gpt-5.6-sol, at: 2026-07-25T17:28:46+02:00 }
snapshot_commit: 480f1b85722754d0e692321082890b6103fe56d8
sources:
  - id: calibration
    resource: ../../phase-b-smart-quant.md
    title: Phase B precision policy
  - id: residency
    resource: ../../phase-c-expert-residency.md
    title: Phase C expert residency
  - id: sessions
    resource: ../../session-persistence.md
    title: Phase D session persistence
  - id: distributed
    resource: ../../phase-e-distributed.md
    title: Phase E distributed inference
---

# Phase status

| Phase | Current state | What is implemented | What remains |
|---|---|---|---|
| **A — Qwen3.5 hybrid engine** | Complete for the supported experimental product | GDN and full-attention hybrid runtime, dense/MoE execution, loaders, tokenizer, server/REPL, continuous batching, correctness and performance gates | Ongoing compatibility and regression maintenance |
| **B — calibration and smart quant** | Functional, opt-in, no shipping win yet | `fucina-calibrate`, reproducible corpus recipe, activation/router sidecar, policy derivation, loader application, quality gate | A validated sub-4-bit kernel or another policy that beats the default on memory/latency/quality |
| **C — tiered expert residency** | Functional constrained-memory fallback | Calibration-derived placement, SSD backing file, compact device slot pool, checksums, LRU, deterministic overflow chunks, next-layer `posix_fadvise` hints | True overlapped direct-I/O/RDMA-class streaming and a throughput case strong enough for default use; host tier was correctly dropped on unified memory |
| **D — session persistence** | Implemented and hardware-validated | Versioned checked snapshot format; Gemma REPL/HTTP; Qwen paged REPL plus continuous-batching HTTP restore/export with GDN state; process restart restored 11 cached tokens and prefilled only the 7-token suffix | Preserve the restart gate in GB10 validation |
| **E — distributed inference** | Integrated experimental boundary | FCNDIST v2 framing/ACK lifecycle, real Qwen3.5 CUDA `[lo,hi)` forward, cgo `ShardRunner`, worker/coordinator CLI, TCP orchestration, CPU protocol tests, and a byte-identical GB10 split-layer gate | Range-filtered weight residency, true two-host parity/performance evidence, batched prefill, and RDMA |

# Interpretation

The original “80% there” architecture plan is now realized through the experimental distributed engine boundary. Remaining work is productization: advantageous precision policy, structured output under batching, range-filtered shard residency, and true two-host throughput evidence.
