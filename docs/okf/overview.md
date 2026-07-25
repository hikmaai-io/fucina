---
type: System
title: Fucina inference engine
description: A hardware-specific Go and CUDA inference engine optimized for NVIDIA DGX Spark GB10.
resource: https://github.com/hikmaai-io/fucina
tags: [inference, cuda, blackwell, gb10, gemma, qwen]
status: draft
stale_after: 2026-08-25
generated: { by: openai-codex/gpt-5.6-sol, at: 2026-07-25T10:33:01+02:00 }
snapshot_commit: 39a96dbd4856f394821021efa10ef31848ad2581
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
| Continuous serving | Functional and performance-oriented; constrained output and some persistence/spec features do not span every batch path. |
| DS4-inspired single-node features | Phases A-C substantially implemented; Phase D is split by serving path; Phase E is a protocol scaffold. |
| Release engineering | Not release-clean today: the normal unit suite passes, but the documented race gate fails and CI coverage differs from the Makefile gate. |
| Portability | Intentionally absent; other GPUs and CUDA toolchains are unsupported. |

# Trust boundary

This bundle distinguishes code inspection and tests run in the current session from historical GPU evidence. GPU correctness and benchmark suites were not rerun during this inspection; their status is inherited from checked-in evidence and documentation. See [test gates](validation/test-gates.md).
