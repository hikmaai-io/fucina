---
type: Capability Matrix
title: DS4-inspired implementation pillars
description: Maturity of the Qwen3.5 Phase A-E roadmap generalized from the DS4 workflow.
tags: [qwen, calibration, quantization, residency, sessions, distributed]
status: draft
stale_after: 2026-08-25
generated: { by: openai-codex/gpt-5.6-sol, at: 2026-07-25T10:33:01+02:00 }
snapshot_commit: 39a96dbd4856f394821021efa10ef31848ad2581
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
| **D — session persistence** | Partial by serving surface | Versioned checked snapshot format; Gemma REPL and HTTP sessions; Qwen paged REPL save/load with GDN state | Qwen/continuous-batching HTTP session integration |
| **E — distributed inference** | Scaffold only | Versioned checksummed protocol, handshake, TCP hop/worker/pipeline abstractions, fake-runner tests | CUDA partial-forward entry point, range-filtered loading, CLI/orchestration, real multi-node parity and speed gates, RDMA |

# Interpretation

The original “80% there” architecture plan is now largely realized for one-node Qwen execution. The remaining distance is not the core model forward: it is proving an advantageous precision policy, broadening persistence/structured-output across mandatory batching, and building the real engine boundary for distributed execution.
