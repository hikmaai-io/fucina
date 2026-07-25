---
type: Capability Matrix
title: Model and checkpoint support
description: Code-verified model families, formats, execution modes, and support caveats.
tags: [models, gemma, qwen, diffusion, gguf, safetensors]
status: draft
stale_after: 2026-08-25
generated: { by: openai-codex/gpt-5.6-sol, at: 2026-07-25T10:33:01+02:00 }
snapshot_commit: 39a96dbd4856f394821021efa10ef31848ad2581
sources:
  - id: detector
    resource: ../../../cuda/gemma4_detect.h
    title: Runtime architecture detector
  - id: loader
    resource: ../../../cuda/qwen35_fp8_loader.h
    title: Qwen3.5 safetensors loader
  - id: readme
    resource: ../../../README.md
    title: Public model matrix
  - id: prune
    resource: ../../legacy-qwen3-removal-plan.md
    title: Legacy Qwen3 removal plan
---

# Verified current boundary

| Family | Inputs | Serving path | Assessment |
|---|---|---|---|
| Gemma 4 12B | Q4_0/Q8_0 GGUF; supported NVFP4 safetensors variants | Single-flight by default; optional continuous batch | First-class experimental |
| Gemma 4 E4B | E4B GGUF/safetensors paths represented by the E4B loader | Dedicated E4B runtime | Functional, with batch feature gaps |
| Qwen3.5/3.6 hybrid dense | `qwen35` GGUF; official FP8-block safetensors | Mandatory continuous batching | First-class experimental |
| Qwen3.5/3.6 hybrid MoE | FP8-block, ModelOpt mixed NVFP4/FP8, supported compressed-tensors checkpoints | Mandatory continuous batching | First-class experimental and benchmarked |
| DiffusionGemma 26B-A4B | Q4_K_M GGUF through `-dm` | Separate diffusion runtime | Experimental |

# Important documentation mismatch

The current detector branches only on GGUF `general.architecture = "qwen35"`; legacy `qwen3` and `qwen3moe` detector functions were removed from `main`.[^detector] Public README/changelog/model-support prose still claims Qwen3 dense and MoE support. Until code or documentation is reconciled, those legacy Qwen3 claims must be treated as stale rather than supported.

[^detector]: Runtime architecture detector

# Platform boundary

All entries above target DGX Spark GB10 (`sm_121a`) and CUDA 13. Other Blackwell variants and pre-Blackwell GPUs are unverified and unsupported as built.
