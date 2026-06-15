---
name: Bug report
about: Report a problem on the supported hardware (DGX Spark GB10)
title: "[bug] "
labels: bug
---

> [!IMPORTANT]
> fucina is experimental and supports **only the NVIDIA DGX Spark GB10** (`sm_121a`, CUDA 13).
> Reports for other GPUs (RTX 30xx/40xx/50xx, A100, H100, B100/B200, …) are out of scope and will
> be closed — the build is pinned to `sm_121a` and the hot paths use GB10-class tensor-core
> features. See the README's *Hardware support* section.

## Hardware (required)
- GPU: <!-- must be DGX Spark GB10 -->
- CUDA version:
- Go version:
- fucina commit (`git rev-parse --short HEAD`):

## What happened

## What you expected

## Repro
<!-- exact command(s), model file + quant (Q4_0-QAT / Q8_0), prompt, flags -->

```sh
fucina -m ... 
```

## Logs / output
<!-- relevant stderr, /metrics, or compute-sanitizer output if a CUDA error -->
