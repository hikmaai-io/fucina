<div align="center">

# fucina

### Gemma 4, forged for the NVIDIA DGX Spark

***fucina*** *— Italian for **forge**: the smithy where raw Gemma 4 weights are hammered into a fast engine for one machine.*

A from-scratch **Gemma 4 12B** inference engine, hand-tuned for exactly one accelerator — the
**DGX Spark GB10** (Blackwell, `sm_121a`, CUDA 13). FP8 Tensor-Core attention, CUDA-graph decode,
MTP speculative decoding, and an OpenAI-compatible server, all in a single static binary.

[Features](#-features) · [Quick start](#-quick-start) · [Performance](#-performance) ·
[Models](#-models) · [Speculative decoding](#-speculative-decoding-mtp) · [HTTP API](#-http-api) ·
[DiffusionGemma](#-diffusiongemma)

![Platform](https://img.shields.io/badge/platform-DGX%20Spark%20GB10-76B900?logo=nvidia&logoColor=white)
![Arch](https://img.shields.io/badge/arch-sm__121a%20(Blackwell)-76B900)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)
![Go](https://img.shields.io/badge/Go-1.26-00ADD8?logo=go&logoColor=white)
![API](https://img.shields.io/badge/API-OpenAI--compatible-412991?logo=openai&logoColor=white)
![Status](https://img.shields.io/badge/status-experimental-orange)

</div>

> [!WARNING]
> **fucina is experimental, hardware-specific, and provided with no support.** It is built and
> tested for a single accelerator — the NVIDIA DGX Spark GB10 — with a toolchain expected at
> `/usr/local/cuda-13` and `/usr/local/go`. It is **not portable** to other GPUs as-is. Licensed
> **Apache-2.0** and shipped **as-is, with no warranty and no support**. See
> [Hardware support](#-hardware-support) and [Status](#-status).

---

## About

**fucina** runs Google's **Gemma 4 12B** entirely on the GPU and serves it over an
OpenAI-compatible HTTP API, plus one-shot and interactive CLI modes. It is a focused experiment in
*how fast a single Blackwell GB10 can drive a dense 12B model* — so instead of portability, it bets
everything on one architecture: FP8/NVFP4 Tensor Cores, position-independent CUDA-graph decode, an
on-GPU sampler, and MTP speculative decoding measured head-to-head against `llama.cpp`.

The same binary also runs **DiffusionGemma 26B-A4B** — a block text-diffusion MoE model — through a
separate CUDA engine selected with `-dm`. See [DiffusionGemma](#-diffusiongemma).

The model: 48 layers (40 sliding-window + 8 global attention), GeGLU FFN (`gelu_pytorch_tanh`,
intermediate 15360), RoPE / p-RoPE, logit softcap 30.0, vocab 262144, hidden 3840, and a Q6_K tied
LM head — loaded from **Q4_0 (QAT)** or **Q8_0** GGUF weights.

---

## ✨ Features

**fucina is fast with:**

- ⚡ **CUDA-graph decode** — single-token decode *and* the K-row batched speculative-verify forward
  are captured as **position-independent** graphs (device-resident position, KV writes *inside* the
  graph) and replayed each step, eliminating per-kernel launch overhead. Pre-captured at startup.
- 🎯 **MTP speculative decoding** — prompt-lookup speculation (free) plus an optional **MTP draft
  head** (`--assistant`) that drafts novel text; one batched weight pass verifies many tokens at the
  exact target distribution. **>2× dense decode** at typical acceptance.
- 🧮 **FP8 Tensor-Core attention** + **FP8 E4M3 KV cache** (1 byte/element) — half the KV bandwidth
  and a flat decode curve as context grows.
- 📦 **Native quantized GEMV/GEMM** — Q4_0 / Q6_K / Q8_0 read directly via `dp4a`, with an optional
  repacked-Q4_0 coalesced-load decode path. No BF16 materialize on the decode hot path.
- 🧠 **On-GPU sampling** — the next token is selected on the device; no 262k-element logit copy back
  to the host when no repeat penalty is set.

**fucina gives you:**

- 🔁 **Prefix-reuse KV cache** — instead of re-prefilling the whole prompt each request, the server
  rewinds the single physical KV cache to the longest common prefix and prefills only the divergent
  suffix — the difference between sub-second and multi-second agentic turns.
- 🌐 **OpenAI-compatible API** — `/v1/chat/completions` (streaming + non-streaming), `/v1/models`,
  `/health`, `/metrics`.
- 🛠️ **Tool calling** (gemma-4 format, OpenAI-shaped) and the **gemma-4 thinking channel**
  (reasoning as `reasoning_content`, controllable per request via `reasoning_effort`).
- 💻 **Three run modes** from one binary — server, one-shot prompt, interactive REPL.
- 📊 **Live observability** — `/metrics` exposes prefix-cache hit rate, prefill/decode throughput,
  and a `speculation` block (`accept_rate`, `tokens_per_forward`).

---

## 🚀 Quick start

### 1. Install dependencies

fucina runs on a **DGX Spark GB10** (see [Hardware support](#-hardware-support)). You need:

| Dependency | Version | How |
|---|---|---|
| **CUDA Toolkit** | **13.0** | Expected at `/usr/local/cuda-13`; else pass `NVCC=`/`CUDA_HOME=` to `make`. [NVIDIA CUDA downloads](https://developer.nvidia.com/cuda-downloads) |
| **Go** | **1.26** | Expected at `/usr/local/go`; else pass `GO=` to `make`. [go.dev/dl](https://go.dev/dl/) |
| **Build tools** | gcc/g++, make, binutils | `sudo apt-get install build-essential` (DGX OS / Ubuntu) |
| **huggingface_hub** | Python 3.10+ | To download GGUF weights: `pip install -U "huggingface_hub[cli]"` |
| **CUTLASS** *(DiffusionGemma only)* | sm_120-capable headers | `pip install flashinfer` vendors them, or `git clone https://github.com/NVIDIA/cutlass`; pass `CUTLASS_DIR=` to `make` |

> [!NOTE]
> The dense Gemma 4 engine needs only CUDA + Go + build tools. **CUTLASS is required only for the
> optional [DiffusionGemma](#-diffusiongemma) engine** — skip it if you only want the 12B.

### 2. Build

```sh
make
```

`make` compiles the CUDA static libraries (`nvcc -arch=sm_121a` → `cuda/libfucina.a` for the dense
engine and `cuda/libdg.a` for the [DiffusionGemma](#-diffusiongemma) engine) and then the Go binary,
verifying the cubin arch and the device-upload code path along the way.

> [!NOTE]
> Requires the [supported toolchain](#requirements): CUDA 13.0 at `/usr/local/cuda-13` and Go 1.26
> at `/usr/local/go`, on a DGX Spark GB10.

#### Build configuration (no hardcoded paths)

All machine-specific locations are overridable — nothing personal is baked in. Pass them as `make`
variables (or environment variables for the helper scripts):

| Setting | Used by | Default | Purpose |
|---------|---------|---------|---------|
| `NVCC`, `CUDA_HOME` | `make` | `/usr/local/cuda-13` | CUDA 13 toolchain location |
| `GO` | `make` | `/usr/local/go/bin/go` | Go 1.26 toolchain |
| `CUTLASS_DIR` | `make` (diffusion) | `/path/to/cutlass` | CUTLASS include dir for the NVFP4 MoE GEMM |
| `DG_GGUF` | `make` (diffusion targets) · `scripts/dg_dump_tensor.py` (env) | `./models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf` | DiffusionGemma GGUF |
| `DG_NVFP4_CKPT` | `scripts/dg_nvfp4_convert.py` (env / `--ckpt`) | — (required) | NVFP4 safetensors snapshot dir |
| `LLAMA_GGUF_PY` | `scripts/dg_dump_tensor.py` (env) | auto (only if `gguf` isn't importable) | path to llama.cpp's `gguf-py` |

```sh
# Example: build with your own CUTLASS and model locations
make CUTLASS_DIR=/opt/cutlass DG_GGUF=/data/diffusiongemma.gguf

# Helper scripts read env vars / flags — no editing required
DG_NVFP4_CKPT=/data/dg-nvfp4-snapshot python3 scripts/dg_nvfp4_convert.py --inspect
DG_GGUF=/data/dg.gguf python3 scripts/dg_dump_tensor.py /tmp/out
```

### 3. Download a model

```sh
# Gemma 4 12B — Q4_0 QAT (recommended; official Google QAT GGUF)
hf download google/gemma-4-12B-it-qat-q4_0-gguf \
  gemma-4-12b-it-qat-q4_0.gguf --local-dir ./models

# Optional: the MTP draft head for faster decode (see step 4 / MTP)
hf download unsloth/gemma-4-12b-it-GGUF \
  MTP/gemma-4-12b-it-Q8_0-MTP.gguf --local-dir ./models
```

See [Models](#-models) for the Q8_0 dense build and the DiffusionGemma weights.

### 4. Run

**As an OpenAI-compatible server:**

```sh
fucina -m ./models/gemma-4-12b-it-qat-q4_0.gguf --ctx 32768 --host 0.0.0.0 --port 8080
```

```sh
# then, from another shell:
curl http://localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

See [HTTP API](#-http-api) for every endpoint.

**As an interactive REPL:**

```sh
fucina -m ./models/gemma-4-12b-it-qat-q4_0.gguf --interactive
```

A multi-turn chat with prefix-reuse caching. Commands: `/reset` (clear the conversation),
`/stats` (KV-cache hit rate), `/quit` (or Ctrl-D).

**As a one-shot prompt:**

```sh
fucina -m ./models/gemma-4-12b-it-qat-q4_0.gguf -p "Write a haiku about CUDA." -n 100
```

**With MTP speculative decoding** — add the draft head for faster decode; works in *all three*
modes (here, the REPL):

```sh
fucina -m ./models/gemma-4-12b-it-qat-q4_0.gguf \
       --assistant ./models/MTP/gemma-4-12b-it-Q8_0-MTP.gguf --interactive
```

See [Speculative decoding (MTP)](#-speculative-decoding-mtp) for how it works and how to observe
acceptance live.

---

## 📈 Performance

Measured against `llama.cpp` (`llama-server`) on a fair side-by-side harness
([`scripts/pi_bench.py`](scripts/pi_bench.py): identical transcript, temperature 0, thinking on,
MTP draft head on both engines):

| Phase | Result vs llama.cpp |
|-------|---------------------|
| **Decode** | **Parity-to-ahead overall; +15–20% at high context (≥5k tokens)** — throughput *rises* with context as acceptance climbs and FP8-KV attention scales, the regime that dominates long agentic sessions. |
| **Prefill** | **Steady-state tied**; the residual gap is one-time cold turns (small-N GEMM efficiency), not steady throughput. |
| **Tool calling** | Matches on the easy suite, ahead on hard agentic scenarios, at equal-or-faster latency. |

The decode wins come from **CUDA-graph launch-bubble removal and speculation acceptance (τ)** — not
weight-load width. Single-token decode on GB10 is **bandwidth-bound on total weight bytes**, so
wider (128-bit) loads change instruction count but not bytes and do not help. Watch the
`speculation` block in `/metrics` (`accept_rate`, `tokens_per_forward`) to observe τ live.

---

## 🖥️ Hardware support

fucina is built and tested for **exactly one accelerator**: the **NVIDIA DGX Spark GB10**.

| | |
|---|---|
| **Supported GPU** | NVIDIA **DGX Spark GB10** (Grace-Blackwell) |
| **Compute capability** | **`sm_121`**, compiled as **`sm_121a`** (arch-specific GB10 features: FP8 / NVFP4 block-scaled MMA, `tcgen05`) |
| **Memory** | GB10's **128 GB unified LPDDR5X** (CPU+GPU shared) |
| **CUDA** | **13.0** at `/usr/local/cuda-13` |

**Other GPUs are not supported as built.** The Makefile pins `CUDA_ARCH := sm_121a` and the fast
paths depend on GB10-class tensor-core features:

- **Consumer Blackwell** (RTX 50-series, `sm_120`) and **datacenter Blackwell** (B100/B200,
  `sm_100`) are **not targeted or tested** — they would at minimum need a `CUDA_ARCH` change and
  FP8/NVFP4 re-validation. Treat as unverified.
- **Pre-Blackwell GPUs** (Hopper, Ada, Ampere, …) lack the FP8/NVFP4 block-scaled MMA paths the
  engine relies on and are out of scope.

> [!TIP]
> **Context vs. memory.** The FP8 KV cache (1 B/element) totals **~54 GiB at the full 262144
> context**, which fits GB10's 128 GB unified memory but not a smaller pool. Lower `--ctx` to save
> memory (e.g. `--ctx 131072` ≈ 27 GiB). The engine clamps `--ctx` to 262144.

### Requirements

- **NVIDIA DGX Spark GB10** — Blackwell, compute capability `sm_121`.
- **CUDA 13.0** at `/usr/local/cuda-13`.
- **Go 1.26** (the Makefile expects `/usr/local/go/bin/go`).
- A **Gemma 4 12B GGUF** in **Q4_0 (QAT)** or **Q8_0** ([download below](#-models)).
- *(Optional)* the **Gemma 4 MTP draft head** GGUF for [speculative decoding](#-speculative-decoding-mtp).

---

## 📥 Models

fucina loads **local GGUF files** and has **no model-download logic of its own** — fetch the weights
with `hf` (or `huggingface-cli download …`, equivalent) and pass the path with `-m`. Only **Q4_0
(QAT)** and **Q8_0** quantizations are accepted for the dense 12B; other formats are rejected at
load. *Repo ids and filenames below are verified against Hugging Face.*

```sh
pip install -U "huggingface_hub[cli]"     # provides the `hf` command
```

**1 · Dense Gemma 4 12B** *(required)* — Q4_0 **QAT** is the recommended path (the official
quantization-aware-trained weights: Q4_0 layers with a Q6_K tied LM head). Q8_0 also works.

```sh
# Q4_0 (QAT) — recommended (official Google QAT GGUF)
hf download google/gemma-4-12B-it-qat-q4_0-gguf \
  gemma-4-12b-it-qat-q4_0.gguf --local-dir ./models

# or Q8_0 (unsloth)
hf download unsloth/gemma-4-12b-it-GGUF \
  gemma-4-12b-it-Q8_0.gguf --local-dir ./models
```

**2 · MTP draft head** *(optional, for [speculation](#-speculative-decoding-mtp))* — a small
(~444 MB, **Q8_0**) separate GGUF. It is a GGUF build of Google's official assistant
**`google/gemma-4-12B-it-assistant`** (architecture `gemma4-assistant`, 4 layers), shipped inside
unsloth's GGUF repo under `MTP/`.

```sh
hf download unsloth/gemma-4-12b-it-GGUF \
  MTP/gemma-4-12b-it-Q8_0-MTP.gguf --local-dir ./models
# → ./models/MTP/gemma-4-12b-it-Q8_0-MTP.gguf
```

**3 · DiffusionGemma 26B-A4B** *(optional, for the [`-dm` engine](#-diffusiongemma))*

```sh
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF \
  diffusiongemma-26B-A4B-it-Q4_K_M.gguf --local-dir ./models
```

> [!NOTE]
> Google's `google/gemma-4-12B-it-assistant` is the canonical MTP head but is published as
> **safetensors**; fucina's `--assistant` needs a **GGUF** build (the unsloth file above).
> If you place a dense model at `./gemma-4-12b-it.gguf`, `./model.gguf`, or `./gguf/model.gguf`,
> fucina finds it automatically when `-m` is omitted. The reported model id is derived from the
> GGUF filename.

---

## 🎯 Speculative decoding (MTP)

Speculative decoding is **on by default** (`--spec`). Each step *drafts* candidate tokens cheaply,
then the 12B verifies them all in **one batched weight pass** and keeps the longest matching prefix
— so accepted drafts cost far less than one decode each. Every verified position is drawn from the
**target model's own distribution**, so the output is **identical to plain decoding** at the same
settings — drafting only changes speed, never the result.

Two drafters work together:

| Drafter | Needs | Best at |
|---------|-------|---------|
| **Prompt-lookup** | nothing (on with `--spec`) | repetitive / structured / code-like text — finds where recent context recurs earlier (consensus n-gram match). Free, host-side. |
| **MTP draft head** | `--assistant <head>.gguf` | **novel** text — a 4-layer multi-token-prediction head (Google's `gemma-4-12B-it-assistant`, ~444 MB Q8_0) running over the shared frozen KV cache. llama.cpp's `--spec-type draft-mtp` equivalent. |

When both are available the engine prefers MTP for novel text and only lets a strong prompt-lookup
draft displace it; draft length adapts per step from each drafter's running acceptance rate
(clamped to `[2, --draft-k]`).

```sh
# Enable MTP (server, one-shot, or REPL — just add --assistant)
fucina -m ./models/gemma-4-12b-it-qat-q4_0.gguf \
      --assistant ./models/MTP/gemma-4-12b-it-Q8_0-MTP.gguf \
      --ctx 32768 --host 0.0.0.0 --port 8080
```

- **Works at any temperature** — correct for both greedy (`--temp 0`) and sampling (`--temp > 0`).
- **Observe it live** — `/metrics` → `speculation` (`accept_rate`, `tokens_per_forward`); CLI runs
  print `[mtp]` / `[lookup]` stats at the end.
- **Falls back** to per-token decode for requests with text `stop` strings (host-side trimming).
- **Disable:** `--spec=false`. Graph escape hatches: `FUCINA_NO_DECODE_GRAPH=1`,
  `FUCINA_NO_BATCHED_GRAPH=1`. The draft head adds ~0.4 GB VRAM.

> The `--spec` flag's built-in help still reads *"greedy/temp=0 only"* — that wording is stale; the
> engine fully supports sampling, including on-GPU repeat-penalty at `--temp > 0`.

---

## 🌐 HTTP API

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions — streaming + non-streaming, tool calls, thinking channel |
| `POST /v1/completions` | Legacy raw-prompt completions (handled by the chat path) |
| `GET  /v1/models` | Lists the loaded model id |
| `POST /v1/embeddings` | Stub — returns an empty data list |
| `GET  /health`, `/healthz` | Liveness + KV-cache stats (hits, misses, hit rate, cached tokens) |
| `GET  /readyz` | Readiness — checks the tokenizer + engine are loaded; `503` when not serviceable |
| `GET  /metrics` | KV/context utilization, prefix-cache hit rate, prefill/decode throughput, `speculation`, `requests_detail` (total, errors, avg latency, avg TTFT), `saturation` (in-flight / max) |

`/v1/*` routes accept an optional `Authorization: Bearer <key>` (see `--api-key`); `/health`, `/healthz`, `/readyz`, `/metrics` are always open. Every response carries an `X-Request-Id` (echoed from the request when present) for log correlation.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  cmd/fucina           CLI: server / one-shot / interactive    │
├─────────────────────────────────────────────────────────────┤
│  internal/server      OpenAI-compatible HTTP API + KV cache   │
│  internal/tokenizer   SentencePiece / Unigram (Gemma 4 vocab) │
├─────────────────────────────────────────────────────────────┤
│  internal/engine/cuda CGO bridge (Go ⇄ C)                     │
├─────────────────────────────────────────────────────────────┤
│  cuda/                CUDA C++ kernels (sm_121a, FP8 TC)       │
└─────────────────────────────────────────────────────────────┘
```

- **`cuda/`** — the entire Gemma 4 12B forward pass: quantized GEMV/GEMM (Q4_0/Q6_K/Q8_0 `dp4a`),
  FP8 flash attention, RoPE, RMSNorm, GeGLU, GPU sampling, and CUDA-graph capture/replay. Compiled
  to `libfucina.a`.
- **`internal/engine/cuda`** — the CGO bridge, wrapping the opaque CUDA engine as a Go `Engine`
  (`NewEngine`, `Prefill`, `Decode`, `GenerateSpec`, `LoadAssistant`, …).
- **`internal/tokenizer`** — Gemma 4 SentencePiece/Unigram tokenizer (vocab 262144), loaded from
  the GGUF's tokenizer section; knows the turn / channel / tool-calling control tokens.
- **`internal/server`** — the HTTP server, chat-template renderer, tool-call parsing, thinking
  channel, and prefix-reuse KV cache.
- **`cmd/fucina`** — CLI entry point and the three run modes.

> [!IMPORTANT]
> cgo does **not** hash the contents of the `-lfucina` static archive, so a plain `go build` can
> silently relink a stale binary against an updated `libfucina.a`. The Makefile defends against this:
> it removes the old binary, rebuilds with `go build -a`, and asserts the device-upload path is
> present in the binary.

---

## ⚙️ Configuration

<details>
<summary><b>Important flags</b> (run <code>fucina --help</code> for the full list)</summary>

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | (required) | Path to the GGUF model (Q4_0-QAT or Q8_0; auto-detected) |
| `--ctx` | `262144` | Context size in tokens (max/default 262144; lower it to save memory) |
| `--temp` | `1.0` | Sampling temperature (gemma-4 default; `0` = greedy) |
| `--top-k` | `64` | Top-K sampling (gemma-4 default) |
| `--top-p` | `0.95` | Top-P / nucleus sampling (gemma-4 default) |
| `--thinking` | `off` | Default reasoning channel: `off`/`on`/`low`/`mid`/`high`/`xhigh` |
| `--assistant` | (none) | Gemma-4 MTP draft-head GGUF for speculation |
| `--spec` | `true` | Speculative decoding (prompt-lookup; MTP too when `--assistant` is set) |
| `--draft-k` | `6` | Max speculative draft length per step |
| `--host` | `127.0.0.1` | Server listen address |
| `--port` | `8080` | Server port |
| `--api-key` | (none) | Bearer token required on `/v1/*` (constant-time; reads `FUCINA_API_KEY` if unset). Empty = auth off (localhost dev) |
| `--max-concurrent` | `4` | Admission-queue depth (in-flight + waiting); excess requests get `503` |
| `--max-output-tokens` | `0` | Absolute per-request output-token ceiling (independent of context window); `0` = no extra cap |

</details>

<details>
<summary><b>Environment toggles</b></summary>

| Variable | Effect |
|----------|--------|
| `FUCINA_NO_DECODE_GRAPH=1` | Disable CUDA-graph capture for single-token decode |
| `FUCINA_NO_BATCHED_GRAPH=1` | Disable CUDA-graph capture for the K-row batched verify |
| `FUCINA_NO_PACKED=1` | Disable the repacked-Q4_0 coalesced-load decode GEMV |
| `FUCINA_NO_WARMUP_PASS=1` | Skip the one-time startup warmup pass |
| `FUCINA_DEBUG=1` | Dump request bodies + rendered prompts to `/tmp/fucina_debug.log` |

</details>

---

## 🌫️ DiffusionGemma

<details>
<summary><b>Running the 26B-A4B text-diffusion MoE model</b></summary>

DiffusionGemma is loaded with **`-dm`** (`--diffusion-model`) instead of `-m`. That routes to a
separate diffusion CUDA engine and enables the **NVFP4 Tensor-Core MoE experts**. All three run
modes and the OpenAI API work exactly as for the dense model.

```sh
# Server
fucina -dm ./models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --ctx 8192 --host 0.0.0.0 --port 8080

# One-shot
fucina -dm ./models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf -p "Write a haiku about the ocean."

# Faster generation, lower quality: fewer denoise steps per block (default 48)
fucina -dm ./models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --denoise-steps 16 -p "Explain hashing."
```

**How it differs from the dense model:**

- **Block diffusion, not autoregressive.** Output is generated by iteratively *denoising* a fixed
  **256-token canvas**, then chaining blocks until an end-of-turn/EOS or `max_tokens`.
- **No token streaming.** A whole block is denoised at once (REPL shows `denoising…`; the server
  emits the block as one SSE delta).
- **Two throughput figures** are reported: `tokens_per_second` (delivered) and
  `canvas_tokens_per_second` (raw denoising rate). Delivered is the apples-to-apples number.
- **Context auto-caps to GPU memory** — pass a large `--ctx`; the engine logs the capped `max_prompt`.

| Flag | Default | Description |
|------|---------|-------------|
| `-dm, --diffusion-model` | (required) | DiffusionGemma GGUF; routes to the diffusion engine + NVFP4 MoE |
| `--denoise-steps` | `48` | Denoise steps per block; **lower = faster, lower quality** (blocks converge ~13–16) |
| `--fp4-moe` | (implied) | NVFP4 MoE experts (on automatically with `-dm`; also usable with `-m`) |
| `--ctx` | `8192` | Context ceiling (auto-capped to free GPU memory) |

</details>

---

## 🧪 Testing

```sh
# CPU-only unit tests (no GPU required) — server, tokenizer, sampler, chat
go test ./internal/server/ ./internal/tokenizer/ ./internal/sampler/ ./internal/chat/

# GPU smoke test (requires the DGX Spark GB10)
make smoke      # builds, then: fucina --prompt "Hello, world!" --predict 32 --temp 0
```

The CUDA engine is validated for **bit-exactness** against reference paths (greedy byte-identical
output, `compute-sanitizer` memcheck clean) and benchmarked with the `scripts/` harnesses
([`pi_bench.py`](scripts/pi_bench.py), [`parity_bench.py`](scripts/parity_bench.py),
[`benchmark_gem4.py`](scripts/benchmark_gem4.py)).

---

## 📌 Status

**Experimental, hardware-specific, no support.** fucina targets exactly one platform — the NVIDIA
DGX Spark GB10 (Blackwell `sm_121a`) with CUDA 13.0 — and is not portable to other GPUs or
toolchains as-is. It is an open research/lab project from **[hikmaai.io](https://hikmaai.io)**,
provided **as-is with no warranty and no support commitment**; issues and PRs are handled
best-effort. The Gemma 4 weights you supply are governed by the
[Gemma license](https://ai.google.dev/gemma/docs/gemma_4_license).

- **Code:** [Apache-2.0](LICENSE) · **Third-party notices:** [NOTICE](NOTICE)
- **Roadmap:** harden the experimental DiffusionGemma path; an sm_120 (RTX 50-series) port to
  loosen the single-hardware constraint.

## 🙏 Acknowledgements

- The name **fucina** is Italian for *forge* — a smithy, and figuratively a *crucible of ideas*.
- **Google** for [Gemma 4](https://ai.google.dev/gemma) and the QAT GGUF / MTP assistant releases.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — the GGUF format, quantized `dp4a`
  kernels, and the `draft-mtp` speculation design that this project measures itself against.
- **[unsloth](https://huggingface.co/unsloth)** for the GGUF conversions used here.

<div align="center"><sub>Built by <a href="https://hikmaai.io">hikmaai.io</a> · formerly <code>gem4d</code></sub></div>
