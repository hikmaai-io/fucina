---
type: System
title: Fucina inference engine
description: A hardware-specific Go and CUDA inference engine optimized for NVIDIA DGX Spark GB10.
resource: https://github.com/hikmaai-io/fucina
tags: [inference, cuda, blackwell, gb10, gemma, qwen]
status: draft
stale_after: 2026-08-25
generated: { by: openai-codex/gpt-5.6-sol, at: 2026-07-25T17:28:46+02:00 }
snapshot_commit: 480f1b85722754d0e692321082890b6103fe56d8
sources:
  - id: readme
    resource: ../../README.md
    title: Fucina README
  - id: makefile
    resource: ../../Makefile
    title: Build and validation targets
  - id: okf
    resource: https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf
    title: Open Knowledge Format v0.2
---

# Purpose

Fucina is a from-scratch inference runtime specialized for one accelerator: NVIDIA DGX Spark GB10 (`sm_121a`) with CUDA 13. It combines a Go API/scheduler layer with CUDA C++ model loaders, kernels, state management, quantization, and graph replay.[^readme]

[^readme]: Fucina README

# Product surface

* OpenAI-shaped HTTP chat/completions API, streaming, tool calls, and reasoning channels.
* One-shot and interactive REPL execution.
* Gemma 4, Qwen3.5/3.6 hybrid dense and MoE engines, Gemma E4B, and an experimental DiffusionGemma path.
* Continuous batching, paged or per-slot state, prefix/state reuse, speculative paths, and disk session primitives.
* Calibration, precision-policy, and SSD expert-residency tooling.

See [model support](capabilities/model-support.md) for the code-verified support boundary and [CUDA runtime](architecture/cuda-runtime.md) for the backend structure.

# Maturity assessment

| Area | Assessment |
|---|---|
| Qwen3.5/3.6 core inference | Mature experimental implementation with extensive real-model gates and strong GB10 benchmark evidence. |
| Gemma 4 core inference | Mature experimental implementation; single-flight remains the richest path. |
| Continuous serving | Functional and performance-oriented; Qwen HTTP slot persistence is implemented, while constrained output and some speculation features do not span every batch path. |
| DS4-inspired features | Phases A-D substantially implemented with the Qwen HTTP restart gate passing; Phase E has an integrated CUDA/TCP execution boundary with range-residency and two-host gates still pending. |
| Release engineering | Not release-clean today: targeted unit/race gates pass, but CI coverage differs from the Makefile gate and the Qwen HTTP restart GPU gate remains to be recorded. |
| Portability | Intentionally absent; other GPUs and CUDA toolchains are unsupported. |

# Trust boundary

This bundle distinguishes current validation from historical evidence. The Phase-E Qwen3.5 split-layer CUDA gate was rerun on 2026-07-25 and passed 8/8 byte-identical frontiers plus 8/8 oracle tokens; the Qwen HTTP process-restart gate also passed; a true two-host distributed run remains pending. See [test gates](validation/test-gates.md).
