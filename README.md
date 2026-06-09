# gem4d

A **Gemma 4 12B** inference engine written in **Go + CUDA C++**, built specifically for the
**NVIDIA DGX Spark GB10** (Blackwell, compute capability **sm_121**, **CUDA 13.0**). It loads
**Q4_0 (QAT)** and **Q8_0** GGUF weights (auto-detected from the file), runs the full model on
the GPU with FP8 Tensor Core attention, and exposes an OpenAI-compatible HTTP API as well as
one-shot and interactive command-line modes. This is a hardware-specific, experimental project —
it targets exactly one accelerator and toolchain.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  cmd/gem4d            CLI: server / one-shot / interactive    │
├─────────────────────────────────────────────────────────────┤
│  internal/server      OpenAI-compatible HTTP API + KV cache   │
│  internal/tokenizer   SentencePiece / Unigram (Gemma 4 vocab) │
├─────────────────────────────────────────────────────────────┤
│  internal/engine/cuda CGO bridge (Go ⇄ C)                     │
├─────────────────────────────────────────────────────────────┤
│  cuda/                CUDA C++ kernels (sm_121, FP8 TC)        │
└─────────────────────────────────────────────────────────────┘
```

Bottom-up, each layer builds on the one below it:

- **`cuda/`** — CUDA C++ kernels for the Gemma 4 12B forward pass: 48 layers (40 sliding-window
  + 8 global attention), GeGLU FFN (`gelu_pytorch_tanh`, intermediate 15360), RoPE/p-RoPE,
  logit softcapping at 30.0, and GPU-side sampling. Q4_0 (QAT, with a Q6_K tied LM head) and
  Q8_0 GGML block formats are handled natively. Compiled to a static archive (`libgem4d.a`).
- **`internal/engine/cuda`** — the CGO bridge. Wraps the opaque CUDA engine state in a Go
  `Engine` type (`NewEngine`, `Prefill`, `Decode`, `DecodeNoCopy`, `SampleDevice`, `GenerateSpec`,
  `LoadAssistant`, …) so the rest of the program never touches C directly.
- **`internal/tokenizer`** — Gemma 4 SentencePiece/Unigram tokenizer (vocab size 262144),
  loaded from the GGUF's tokenizer section. Knows the Gemma-4 turn (`<|turn>`/`<turn|>`),
  channel (`<|channel>`/`<channel|>`), and tool-calling control tokens.
- **`internal/server`** — the OpenAI-compatible HTTP server, route registration, the gemma-4
  chat template renderer, tool-call parsing, the reasoning ("thinking") channel handling, and
  the prefix-reuse KV cache.
- **`cmd/gem4d`** — the CLI entry point: flag parsing (llama.cpp-style), and the three run modes.

### Key features

- **Prefix-reuse KV cache** — instead of re-prefilling the whole prompt every request (llama.cpp
  style), the server tracks the exact tokens already materialized in the single physical KV cache,
  computes the longest common prefix with the new prompt, rewinds to it, and prefills only the
  divergent suffix. Single logical sequence / single slot (`--n-slots 1`); requests are serialized
  under a mutex for the full prefill+generate span.
- **Speculative decoding** — prompt-lookup speculation is on by default (`--spec`), works for both
  greedy and sampling at the same output distribution; an optional **MTP assistant** (the official
  Gemma-4 draft head) can be loaded with `--assistant` to draft novel text.
- **GPU-side sampling** — when no repeat penalty is configured, the next token is selected on the
  device and decoded without copying the 262k-element logit vector back to the host.
- **OpenAI-compatible API** — `/v1/chat/completions` (streaming + non-streaming) with **tool
  calling** and the **gemma-4 thinking channel** (reasoning emitted as `reasoning_content`,
  controlled per-request via `reasoning_effort` / `thinking` / `enable_thinking`).

## Requirements

- **NVIDIA DGX Spark GB10** — Blackwell, compute capability **sm_121** (this exact arch).
- **CUDA 13.0**, installed at **`/usr/local/cuda-13`** (provides FP8 Tensor Core support).
- **Go 1.26** (the Makefile expects `/usr/local/go/bin/go`).
- A Gemma 4 12B GGUF in **Q4_0 (QAT)** or **Q8_0** format.

## Build

```sh
make
```

`make` runs two targets: `lib` (the CUDA static library) then `gem4d` (the Go binary).

1. **Device compile (`nvcc -dc`)** — `cuda/gemma4_kernels.cu` is compiled to relocatable device
   code (`gemma4_kernels.o`) for `-arch=sm_121`.
2. **Device link (`nvcc -dlink`)** — the device objects are linked into `gemma4_kernels_link.o`.
3. **Static archive** — both objects are bundled with `ar rcs` into `cuda/libgem4d.a`.
4. **CGO link** — `go build` links the Go code against `libgem4d.a` via the cgo `CGO_CFLAGS`/
   `CGO_LDFLAGS` (`-lcudart -lcublas -lcuda …`).

> **Stale-link verification.** cgo does **not** hash the contents of the `-lgem4d` static archive,
> so a plain `go build` can silently relink a stale binary against an updated `libgem4d.a` (this
> once caused weights to be read over unified memory — a multi-second cold page-fault charged to
> prefill). The Makefile defends against this: it removes the old binary, rebuilds with `go build -a`,
> and then runs `strings gem4d | grep "uploading … weights to device"` — the build **fails** if the
> device-upload path is not present in the binary.

## Usage

### Server mode (default)

```sh
gem4d -m gemma-4-12b-it.gguf --ctx 32768 --host 0.0.0.0 --port 8080
```

### One-shot prompt

```sh
gem4d -m gemma-4-12b-it.gguf -p "Hello" -n 100
gem4d -m gemma-4-12b-it.gguf --prompt "Test" --predict 32
```

### Interactive REPL

```sh
gem4d -m gemma-4-12b-it.gguf --interactive
```

The REPL is a multi-turn chat with prefix-reuse KV caching. Commands: `/reset` clears the
conversation, `/stats` shows the KV cache hit rate, `/quit` (or Ctrl-D) exits.

### Chat completion via curl

```sh
curl http://localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

### Important flags

| Flag          | Default     | Description                                                            |
|---------------|-------------|------------------------------------------------------------------------|
| `-m, --model` | (required)  | Path to the GGUF model (Q4_0-QAT or Q8_0; auto-detected)               |
| `--ctx`       | `4096`      | Context size in tokens (max 262144)                                    |
| `--temp`      | `1.0`       | Sampling temperature (gemma-4 default; `0` = greedy)                   |
| `--top-k`     | `64`        | Top-K sampling (gemma-4 default)                                       |
| `--top-p`     | `0.95`      | Top-P / nucleus sampling (gemma-4 default)                             |
| `--thinking`  | `off`       | Default reasoning channel: `off`/`on`/`low`/`mid`/`high`/`xhigh`       |
| `--assistant` | (none)      | Gemma-4 MTP assistant GGUF (official draft head) for speculation       |
| `--spec`      | `true`      | Prompt-lookup speculative decoding                                     |
| `--port`      | `8080`      | Server port                                                            |

Run `gem4d --help` for the full flag list.

## HTTP API

| Endpoint                | Description                                                              |
|-------------------------|--------------------------------------------------------------------------|
| `POST /v1/chat/completions` | Chat completions; streaming + non-streaming, tool calls, thinking channel |
| `POST /v1/completions`  | Legacy raw-prompt completions (handled by the chat path)                 |
| `GET  /v1/models`       | Lists the loaded model id                                                |
| `POST /v1/embeddings`   | Stub — returns an empty data list                                        |
| `GET  /health`          | Liveness + KV cache stats (hits, misses, hit rate, cached tokens)        |
| `GET  /metrics`         | KV/context utilization, prefix-cache hit rate, prefill/decode throughput |

## Testing

CPU-only tests (no GPU required) — server and tokenizer packages:

```sh
go test ./internal/server/ ./internal/tokenizer/
```

GPU smoke test (requires the DGX Spark GB10):

```sh
make smoke
```

`make smoke` builds the binary and runs `gem4d --prompt "Hello, world!" --predict 32 --temp 0`.

## License / status

**Experimental and hardware-specific.** gem4d targets exactly one platform — the NVIDIA DGX Spark
GB10 (Blackwell sm_121) with CUDA 13.0 — and is not portable to other GPUs or toolchains as-is.
Use at your own risk.
