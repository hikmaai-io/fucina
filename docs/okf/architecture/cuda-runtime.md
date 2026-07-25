---
type: Architecture
title: CUDA runtime architecture
description: The GB10-specific model loading, quantization, execution, state, and CUDA-graph backend.
tags: [cuda, kernels, quantization, graphs, state]
status: draft
stale_after: 2026-08-25
generated: { by: openai-codex/gpt-5.6-sol, at: 2026-07-25T10:33:01+02:00 }
snapshot_commit: 39a96dbd4856f394821021efa10ef31848ad2581
sources:
  - id: runtime
    resource: ../../../cuda/gemma4_kernels.cu
    title: Main Gemma and Qwen CUDA runtime
  - id: q35-runtime
    resource: ../../../cuda/qwen35_runtime.cuh
    title: Qwen3.5 runtime state and execution
  - id: e4b
    resource: ../../../cuda/e4b_engine.cu
    title: Gemma E4B CUDA engine
  - id: tensor-plan
    resource: ../../tensor-management-refactor-plan.md
    title: Tensor management refactor analysis
---

# Runtime boundary

`cuda/libfucina.a` contains the autoregressive Gemma/Qwen runtime and E4B engine; `cuda/libdg.a` contains the separate DiffusionGemma engine. Go reaches these archives through cgo bridges in `internal/engine/*`.

# Main execution capabilities

* Runtime checkpoint detection and geometry loading for Gemma 4 and Qwen3.5-style hybrid models.
* GGUF, FP8-block safetensors, ModelOpt mixed FP8/NVFP4, and compressed-tensors variants where documented.
* Q4_0/Q6_K/Q8_0/Q4_K `dp4a`, FP8, BF16, and native NVFP4 execution paths.
* Full attention, sliding/paged attention, and Qwen Gated-DeltaNet recurrent state.
* Grouped MoE routing and expert GEMM, including bounded SSD-backed expert slots.
* Position-independent CUDA graphs for major decode paths, with eager fallbacks.
* Per-slot snapshot, rollback, and state-cache primitives used by warm turns, sessions, and speculative experiments.

# Structural risk

The main CUDA translation unit remains very large, and tensor identity, format, ownership, alternate representations, scratch indexing, and dispatch are distributed across several implicit conventions.[^tensor-plan] The landed `ModelPlan` and allocation-set foundations reduce risk, but the refactor document still marks the broader migration in progress.

[^tensor-plan]: Tensor management refactor analysis

# Build invariant

The Makefile deliberately forces a cgo relink after rebuilding static CUDA archives because Go does not hash archive contents. A plain `go build` can otherwise produce a binary linked against stale device code.
