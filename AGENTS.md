# AGENTS.md

Operational notes for coding agents working on **fucina**. For architecture and the public
API see [README.md](README.md); this file is about *building, running, and testing* the engine.

## Hardware & toolchain

fucina builds and runs on **exactly one machine**: an NVIDIA **DGX Spark GB10** (Blackwell
`sm_121a`, CUDA 13). It does not compile on a machine without `nvcc` and the GB10 toolchain.

- CUDA: `/usr/local/cuda-13` (`nvcc` 13.0)
- Go: `/usr/local/go` (1.26+ — `go.mod` requires `go 1.26`; a system `go 1.22` will not build it)
- Add both to `PATH`: `export PATH=/usr/local/go/bin:/usr/local/cuda-13/bin:$PATH`

If you edit on a machine without CUDA, sync the changed files to the GB10 and build there.

## Build

```bash
make lib       # nvcc: cuda/*.cu -> cuda/libfucina.a  (the slow step; a few minutes)
make fucina    # cgo: link the Go binary against libfucina.a (forces -a relink)
```

`make lib` must run before `make fucina` whenever a `.cu`/`.cuh` changes (cgo does not hash the
archive contents). `make test` runs the Go suites plus the in-binary self-tests.

## Run

Server mode is the **default** (no `-p`/`-f`/`-i`): fucina serves an **OpenAI-compatible** API
on `--host:--port` (default `127.0.0.1:8080`), `/v1/chat/completions` and `/v1/completions`.

```bash
# Autoregressive Gemma-4 (12B or 31B — geometry is read from the GGUF):
./fucina -m /path/to/gemma-4-31B-it-Q4_0.gguf --ctx 8192 --port 8080

# One-shot (no server):
./fucina -m <model.gguf> -p "The capital of France is" -n 32
```

The model id reported to clients is the GGUF basename without `.gguf`. Auth is off on
localhost; set `--api-key` (or `FUCINA_API_KEY`) to require a bearer token.

## Agentic tool-call testing — `tool-eval-bench`

`tool-eval-bench` (installed at `~/.local/bin/tool-eval-bench` on the GB10) is an agentic
tool-calling benchmark that drives any OpenAI-compatible endpoint. Use it to check that a
model serves coherent multi-turn tool calls end-to-end.

1. Start fucina as a server with the model under test (leave it running):

   ```bash
   ./fucina -m /path/to/gemma-4-31B-it-Q4_0.gguf --ctx 16384 --port 8080 &
   ```

2. Point the bench at it. `--base-url` is the fucina endpoint, `--model` the served id
   (the GGUF basename), `--backend llamacpp` selects the OpenAI-compatible adapter:

   ```bash
   export PATH=$HOME/.local/bin:$PATH
   # Quick reachability check (exit 0 = server ready):
   tool-eval-bench --base-url http://127.0.0.1:8080/v1 --probe

   # Core 15-scenario run, deterministic:
   tool-eval-bench \
     --base-url http://127.0.0.1:8080/v1 \
     --model gemma-4-31B-it-Q4_0 \
     --backend llamacpp \
     --temperature 0 --short

   # Full suite incl. hard mode:
   tool-eval-bench --base-url http://127.0.0.1:8080/v1 --model gemma-4-31B-it-Q4_0 \
     --backend llamacpp --temperature 0 --hardmode
   ```

`--categories A B …`, `--hardmode-only`, `--trials N`, and `--json-file PATH` narrow or capture
a run. `--temperature 0` is the right default for a regression-style comparison.

## Conventions

- Commit messages: Conventional Commits (`feat:`/`fix:`/`refactor:`…).
- The attention kernels are templated on head counts on purpose — making model dimensions
  runtime spills registers to local memory (50-200× slower). New geometries are added as
  pre-compiled template instantiations dispatched at load by `eng->geom`, not as runtime
  parameters. See `cuda/gemma4_kernels.cu` (the note above `sliding_attn_splitk_kernel`).
