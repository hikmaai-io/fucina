# Contributing to fucina

First, the honest expectations.

> [!IMPORTANT]
> **fucina is an experimental research/lab project from [hikmaai.io](https://hikmaai.io),
> provided with no support.** It targets exactly one accelerator — the NVIDIA DGX Spark GB10
> (Blackwell `sm_121a`, CUDA 13). Issues and pull requests are handled **best-effort, with no SLA**.
> We may decline contributions that don't fit the single-target scope.

## Before you open an issue or PR

- **Hardware.** Anything touching the CUDA engine can only be built and validated on a **DGX Spark
  GB10**. Bug reports for other GPUs (RTX 30xx/40xx/50xx, A100, H100, B100/B200, …) are **out of
  scope** today — the build is pinned to `sm_121a` and the hot paths use GB10-class FP8/NVFP4
  tensor-core features. See the README's *Hardware support* section.
- **Scope.** First-class targets are the dense **Gemma 4 12B** engine and the **Qwen3 / Qwen3.5 /
  Qwen3.6** family (dense and MoE, GGUF and safetensors/FP8/NVFP4) — see
  [docs/qwen-models.md](docs/qwen-models.md) for the Qwen detection/loader reference. DiffusionGemma
  and any multi-GPU port are roadmap, not current guarantees.

## What you can build & test without a GPU

The Go server, tokenizer, sampler, chat, and batch-scheduler packages are **pure Go** and run
anywhere:

```sh
go test ./internal/server/ ./internal/server/batch/ ./internal/tokenizer/ ./internal/sampler/ ./internal/chat/
go vet  ./internal/server/ ./internal/server/batch/ ./internal/tokenizer/ ./internal/sampler/ ./internal/chat/
gofmt -l .   # must print nothing
```

This is exactly what CI runs. The cgo engine package (`internal/engine/cuda`) and `cmd/...` link
against `libfucina.a` and require the CUDA toolchain — they are **not** in CI.

## Building the full engine (GB10 only)

```sh
make            # nvcc -arch=sm_121a -> cuda/libfucina.a, then the Go binary
make smoke      # quick end-to-end generate
make check      # go vet + pure-Go unit tests
```

## Correctness bar

The CUDA engine is held to **bit-exactness**: changes to kernels must keep greedy output
byte-identical to the reference path (or be provably reassociation-only), pass
`compute-sanitizer` memcheck cleanly, and keep `make check` green. Include the evidence in your PR
description (which harness, which prompts, before/after).

## Style

- Go: `gofmt` (CI enforces); match the surrounding code.
- CUDA: match the conventions already in `cuda/gemma4_kernels.cu` (all architectures share this
  file) and `cuda/qwen35_fp8_loader.h` (Qwen3.5/3.6 safetensors loading).
- Commits: clear, imperative subject; explain the *why*. (This repo keeps a clean, linear history.)

## DCO / licensing

By submitting a contribution you agree it is licensed under the project's
[Apache-2.0](LICENSE) license. Please sign off your commits (`git commit -s`).
