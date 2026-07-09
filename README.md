<div align="center">

# fucina

### Gemma 4 and Qwen3 / Qwen3.5 / Qwen3.6, forged for the NVIDIA DGX Spark

***fucina*** *— Italian for **forge**: the smithy where raw model weights are hammered into a fast engine for one machine.*

A from-scratch inference engine, hand-tuned for exactly one accelerator — the **DGX Spark GB10**
(Blackwell, `sm_121a`, CUDA 13). It serves **Gemma 4 12B**, **Qwen3 / Qwen3.5 / Qwen3.6** (dense
and MoE, GGUF / FP8-safetensors / NVFP4), and the experimental **DiffusionGemma 26B-A4B** text-
diffusion MoE — FP8/NVFP4 Tensor-Core attention, CUDA-graph decode, continuous batching over a
paged KV cache, speculative decoding, and an OpenAI-compatible server, all in a single static
binary.

[Features](#-features) · [Quick start](#-quick-start) · [Model support](#-models) ·
[Performance](#-performance) · [Continuous batching](#-continuous-batching--paged-kv) ·
[Speculative decoding](#-speculative-decoding) · [HTTP API](#-http-api) ·
[DiffusionGemma](#-diffusiongemma)

![Platform](https://img.shields.io/badge/platform-DGX%20Spark%20GB10-76B900?logo=nvidia&logoColor=white)
![Arch](https://img.shields.io/badge/arch-sm__121a%20(Blackwell)-76B900)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)
![Go](https://img.shields.io/badge/Go-1.26-00ADD8?logo=go&logoColor=white)
![Models](https://img.shields.io/badge/models-Gemma%204%20%C2%B7%20Qwen3%2F3.5%2F3.6-412991)
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

**fucina** runs Google's **Gemma 4 12B** and Alibaba's **Qwen3 / Qwen3.5 / Qwen3.6** families
(dense and sparse-MoE) entirely on the GPU and serves them over an OpenAI-compatible HTTP API, plus
one-shot and interactive CLI modes. It started as a focused experiment in *how fast a single
Blackwell GB10 can drive a dense 12B model* and has since grown into a small multi-architecture
engine — but it still bets everything on one accelerator instead of portability: FP8/NVFP4 Tensor
Cores, position-independent CUDA-graph decode, an on-GPU sampler, continuous batching over a paged
KV cache, and speculative decoding measured head-to-head against `llama.cpp` and `vLLM`.

There is no model-selection flag: `-m <path>` is enough — the architecture (Gemma-4 / Qwen3 dense /
Qwen3 MoE / Qwen3.5-3.6 hybrid dense / Qwen3.5-3.6 MoE) and the weight format (GGUF quant, official
FP8-block safetensors, or NVFP4 — generic or NVIDIA ModelOpt) are detected from the file itself. See
[Models](#-models) for the full support matrix and exact download commands.

The same binary also runs **DiffusionGemma 26B-A4B** — a block text-diffusion MoE model — through a
separate CUDA engine selected with `-dm`. See [DiffusionGemma](#-diffusiongemma).

**Gemma 4 12B:** 48 layers (40 sliding-window + 8 global attention), GeGLU FFN
(`gelu_pytorch_tanh`, intermediate 15360), RoPE / p-RoPE, logit softcap 30.0, vocab 262144, hidden
3840, and a Q6_K tied LM head — loaded from **Q4_0 (QAT)** or **Q8_0** GGUF weights, or native
NVFP4 safetensors.

**Qwen3.5 / Qwen3.6 hybrid:** a Gated-DeltaNet (GDN) linear-attention mixer on most layers with a
periodic full softmax-GQA layer (`qwen35.full_attention_interval`), served dense (9B/27B) or as a
sparse MoE (35B-A3B, top-k routed experts), from GGUF or official FP8-block / NVIDIA ModelOpt
safetensors. "Qwen3.6" checkpoints reuse this same hybrid architecture and detector — there is no
separate Qwen3.6 code path.

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
  and a flat decode curve as context grows. The sliding-window layers use a **capped ring buffer**,
  so KV memory stays nearly context-independent (~1.5 GiB sliding regardless of `--ctx`). An optional
  **NVFP4 KV codec** (`FUCINA_KV_NVFP4=1`, native Blackwell FP4) trades a small accuracy cost for a
  1.78× smaller KV footprint — memory-only, opt-in (see [`docs/kv-quant-exploration.md`](docs/kv-quant-exploration.md)).
- 🧵 **Continuous batching over a paged KV cache** — independent sequences decode in one batched
  pass (vLLM-style block tables + free-list, split-K paged attention bit-identical to the contiguous
  path, CUDA graph per batch size). **Opt-in for Gemma-4** (`--batch`, implies `--paged-kv`) — the
  default single-flight path is unchanged there; **mandatory and auto-enabled for every Qwen3
  checkpoint** (Qwen has no single-flight path). See [`docs/continuous-batching.md`](docs/continuous-batching.md).
- 🧑‍🤝‍🧑 **Multi-architecture, no model-select flag** — Gemma-4, Qwen3 dense, Qwen3 MoE, and the
  Qwen3.5/3.6 Gated-DeltaNet hybrid (dense and MoE) are all detected from the GGUF metadata or
  `config.json`, and served through the same binary and (mostly) the same code paths.
- 📦 **Native quantized GEMV/GEMM** — Q4_0 / Q6_K / Q8_0 / Q4_K read directly via `dp4a`, with an
  optional repacked-Q4_0 coalesced-load decode path (Gemma) and a default Q4_K requant of the
  attention/GDN mixer + FFN/expert weights for Qwen3.5/3.6 dense **and** MoE checkpoints (smaller
  resident weight set, faster decode; `FUCINA_MOE_FP8=1` reverts to pure FP8). No BF16 materialize
  on the decode hot path.
- 🧠 **On-GPU sampling** — the next token is selected on the device; no full-vocab logit copy back
  to the host when no repeat penalty is set.

**fucina gives you:**

- 🔁 **Prefix-reuse KV cache** — instead of re-prefilling the whole prompt each request, the server
  rewinds the single physical KV cache to the longest common prefix and prefills only the divergent
  suffix — the difference between sub-second and multi-second agentic turns. Rewinds stay exact within
  the sliding ring's window (covers same-conversation turns and speculation); a deeper divergence
  falls back to a full re-prefill (see `FUCINA_SLIDING_RING`).
- 🌐 **OpenAI-compatible API** — `/v1/chat/completions` (streaming + non-streaming), `/v1/models`,
  `/health`, `/metrics`.
- 🛠️ **Tool calling**, OpenAI-shaped on the wire for both dialects — Gemma-4's native format and
  Qwen3-Coder's XML `<tool_call>` form (auto-selected with the chat dialect, no flag) — plus a
  **thinking/reasoning channel** on both (reasoning as `reasoning_content`, controllable per request
  via `reasoning_effort` or `thinking`).
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

# — or — Qwen3.6-35B-A3B, official FP8 (MoE; --local-dir gives a self-contained checkpoint dir,
# no --tokenizer needed — see "HuggingFace checkpoint paths" under Models)
hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir ./models/Qwen3.6-35B-A3B-FP8
```

See [Models](#-models) for the Q8_0 dense build, every Qwen3/3.5/3.6 checkpoint, and the
DiffusionGemma weights.

### 4. Run

**As an OpenAI-compatible server:**

```sh
fucina -m ./models/gemma-4-12b-it-qat-q4_0.gguf --ctx 32768 --host 0.0.0.0 --port 8080
# — or — a Qwen checkpoint (continuous batching auto-enables itself; no extra flags needed):
fucina -m ./models/Qwen3.6-35B-A3B-FP8 --host 0.0.0.0 --port 8080
```

```sh
# then, from another shell:
curl http://localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

See [HTTP API](#-http-api) for every endpoint and a tool-calling example.

**As an interactive REPL:**

```sh
fucina -m ./models/gemma-4-12b-it-qat-q4_0.gguf --interactive
```

A multi-turn chat with prefix-reuse caching. Commands: `/thinking LEVEL` (set the reasoning
channel — `off`/`on`/`low`/`medium`/`high`/`xhigh`; the thought channel renders dimmed and is
budget-bounded), `/reset` (clear the conversation), `/stats` (KV-cache hit rate), `/quit` (or
Ctrl-D). The REPL applies the chat template for the detected dialect (Gemma-4 or Qwen ChatML — no
flag) and honours `--thinking` / `--repeat-penalty`, matching the server. Qwen checkpoints route
through the same paged multi-sequence path as the server (single-flight is Gemma-only).

**As a one-shot prompt:**

```sh
fucina -m ./models/gemma-4-12b-it-qat-q4_0.gguf -p "Write a haiku about CUDA." -n 100
```

**With MTP speculative decoding** (Gemma-4 only) — add the draft head for faster decode; works in
*all three* modes (here, the REPL):

```sh
fucina -m ./models/gemma-4-12b-it-qat-q4_0.gguf \
       --assistant ./models/MTP/gemma-4-12b-it-Q8_0-MTP.gguf --interactive
```

Qwen checkpoints get free-standing **prompt-lookup** speculation automatically on dense models
(not MoE) — no `--assistant` head applies there. See [Speculative decoding](#-speculative-decoding)
for how it works and how to observe acceptance live.

---

## 📈 Performance

**Gemma-4 12B**, measured against `llama.cpp` (`llama-server`) on a fair side-by-side harness
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

**Qwen3.5-35B-A3B FP8 MoE**, measured against `vLLM` on the same checkpoint (`bench_serving.py`,
see [`docs/qwen35-beat-vllm-plan.md`](docs/qwen35-beat-vllm-plan.md) for the full record):

| Regime | fucina | vLLM | Status |
|---|---|---|---|
| Decode, single stream (short ctx) | 56–58 tok/s | 53.4 tok/s | ahead |
| Long-context single-stream decode | 49.5 tok/s @3.5k / 45.5 tok/s @6k | — | flash-decoding fixed this regime |
| Turn-2+ TTFT @2k (per-conversation state cache) | 0.107 s | weak/no equivalent APC | ahead |
| Cold turn-1 TTFT @2k | 2.6 s | 1.19 s | **behind, ~2.2×** |
| Aggregate throughput @conc-16 | ~203–274 tok/s (measurement noise ±8%) | ~449 in an identical-prompt bench (not a steady-state number; not directly comparable) | inconclusive, see the plan doc |

Net: fucina wins single-stream decode and warm-conversation TTFT; cold-prompt (first-turn) TTFT and
aggregate-throughput scaling at high concurrency are open gaps, not wins — don't read the aggregate
row as "fucina beats vLLM at scale."

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
> **Context vs. memory (Gemma-4).** The sliding-window KV cache (40 of 48 layers) is a capped
> **ring buffer**, so it does **not** grow with `--ctx`: ~**1.5 GiB** at the default ring size
> (8192 slots), tunable via `FUCINA_SLIDING_RING`. Only the 8 global-attention layers' KV scales
> with context (~2.0 GiB at the full 262144). Total FP8 KV cache (1 B/element, K+V): **~3.5 GiB at
> 262144**, **~2.5 GiB at 131072** — down from ~54 GiB before the ring. To shrink the *weights*
> footprint, set `FUCINA_NO_PACKED=1` to drop the repacked-Q4_0 decode copy (**−~7 GiB**, ~2–3%
> slower decode). The engine clamps `--ctx` to 262144. See [Environment toggles](#-http-api).

> [!TIP]
> **Memory sizing (Qwen).** There is no hardcoded minimum-VRAM figure for the 35B-A3B MoE
> checkpoints. `--gpu-mem-util` (default `0.90`) caps total device memory the engine will use as a
> fraction of the GPU's total; the paged-KV pool is auto-sized to whatever free memory remains
> after weights, and `--ctx` / the paged pool auto-cap to fit. If a Qwen checkpoint fails to start
> with a paged-KV allocation error, lower `--ctx` or raise `--gpu-mem-util` first before assuming
> the checkpoint doesn't fit at all. The 27B dense and 35B-A3B MoE FP8 checkpoints, plus the
> smaller NVFP4 ModelOpt build, are all default-Q4_K-requanted for the mixer/FFN/expert weights at
> load (see [Models](#-models)), which shrinks the resident set relative to their on-disk FP8 size.

### Requirements

- **NVIDIA DGX Spark GB10** — Blackwell, compute capability `sm_121`.
- **CUDA 13.0** at `/usr/local/cuda-13`.
- **Go 1.26** (the Makefile expects `/usr/local/go/bin/go`).
- A supported checkpoint ([download below](#-models)): **Gemma 4 12B** GGUF (**Q4_0 QAT** or
  **Q8_0**) or NVFP4 safetensors; or a **Qwen3 / Qwen3.5 / Qwen3.6** GGUF, official FP8-block
  safetensors, or NVIDIA ModelOpt NVFP4/FP8 checkpoint.
- *(Optional, Gemma-4 only)* the **Gemma 4 MTP draft head** GGUF for
  [speculative decoding](#-speculative-decoding).

---

## 📥 Models

fucina loads **local model files** (GGUF or safetensors) and has **no model-download logic of its
own** — fetch weights with `hf` (or `huggingface-cli download …`, equivalent) and pass the path
with `-m`. The architecture and quant format are **auto-detected from the file itself** — there is
no `--model-type`/`--arch` flag. *Repo ids and filenames below are verified against Hugging Face.*

```sh
pip install -U "huggingface_hub[cli]"     # provides the `hf` command
```

### Model support matrix

| Family | `-m` accepts | Detected via | Batching |
|---|---|---|---|
| Gemma 4 12B | Q4_0-QAT / Q8_0 GGUF; NVFP4 safetensors | GGUF falls through to Gemma when `general.architecture` isn't a `qwen*` key; NVFP4 dir/`config.json` | opt-in (`--batch`) |
| Qwen3 dense | GGUF | `general.architecture = "qwen3"` | mandatory (auto) |
| Qwen3 MoE | GGUF | `general.architecture = "qwen3moe"` | mandatory (auto) |
| Qwen3.5 / Qwen3.6 hybrid, dense | GGUF, or official FP8-block safetensors | GGUF `"qwen35"`; safetensors `config.json` `model_type` contains `"qwen3_5"` + `quant_method` `fp8`/`modelopt`, no expert tensors | mandatory (auto) |
| Qwen3.5 / Qwen3.6 hybrid, MoE (A3B) | official FP8-block safetensors, or NVIDIA ModelOpt mixed NVFP4/FP8 safetensors | same as above, plus an `mlp.experts.0.gate_proj.weight` tensor | mandatory (auto) |
| DiffusionGemma 26B-A4B | Q4_K_M GGUF (via `-dm`) | separate engine, explicit flag | N/A (block diffusion) |

**"Qwen3.6" is a checkpoint generation, not a separate code path** — `Qwen/Qwen3.6-27B-FP8`,
`Qwen/Qwen3.6-35B-A3B-FP8`, and `nvidia/Qwen3.6-35B-A3B-NVFP4` are all recognized and served through
the same Qwen3.5 hybrid detector and loader (identical `qwen3_5` architecture).

**Continuous batching is mandatory for every Qwen3/3.5/3.6 checkpoint** (no single-flight path
exists for Qwen); fucina detects this and turns `--batch`/`--paged-kv` on for you automatically — a
bare `fucina -m <qwen-checkpoint>` just works. See
[Continuous batching & paged KV](#-continuous-batching--paged-kv).

### HuggingFace checkpoint paths — the robust way

For any safetensors checkpoint (Qwen or Gemma NVFP4), **always download with `--local-dir` and
point `-m` at that directory**:

```sh
hf download <repo> --local-dir ./models/<name>
fucina -m ./models/<name> ...
```

This gives you a flat directory with `config.json` + weight shards + `tokenizer.json` at the top
level, so both the weight loader and the tokenizer auto-discovery (below) find everything with no
`--tokenizer` flag and no ambiguity.

If you instead used the bare `hf download <repo>` cache (no `--local-dir`), the files land under
`~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<hash>/` — pass **that snapshot
directory** to `-m`, not the `models--...` parent. The Qwen3.5/3.6 FP8/ModelOpt loader specifically
resolves a bare repo-root parent too (it globs `snapshots/*/model.safetensors.index.json`), but the
Go-side tokenizer auto-discovery does **not** do that glob and will fail to find `tokenizer.json`
if you point `-m` at the parent — you'd need an explicit `--tokenizer <snapshot-dir>/tokenizer.json`
in that case. The generic Gemma-4 NVFP4 loader has no repo-root resolution at all and needs the
real snapshot-shaped directory. Using `--local-dir` sidesteps all of this.

**Tokenizer auto-discovery, in short:** GGUF models never need `--tokenizer` (vocab is inline).
Safetensors/NVFP4/FP8 checkpoints need a `tokenizer.json`, found automatically as a sibling of `-m`
(per the path rule above) or passed explicitly via `--tokenizer <tokenizer.json>`.

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

**2 · MTP draft head** *(optional, for [speculation](#-speculative-decoding))* — a small
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

### Native NVFP4 (safetensors) — Gemma 4

fucina also runs a **natively NVFP4-quantized** Gemma 4 loaded straight from a HuggingFace
safetensors checkpoint (e.g. `RedHatAI/gemma-4-12B-it-NVFP4`) — a single FP4 weight store feeding
both the cuBLASLt block-scaled prefill and a fused FP4 decode GEMV, with the checkpoint's own
`tokenizer.json` read natively (no `--tokenizer`). Point `-m` at the checkpoint directory (see
[HuggingFace checkpoint paths](#huggingface-checkpoint-paths--the-robust-way) above — use
`--local-dir` so this "just the directory" usage works without a snapshot-path gotcha):

```bash
./fucina -m /path/to/gemma-4-12B-it-NVFP4 -p "Explain photosynthesis." -n 128
```

Both compressed-tensors (`RedHatAI/*`) and ModelOpt (`nvidia/*-FP4`) naming are auto-detected. See
[docs/nvfp4-safetensors.md](docs/nvfp4-safetensors.md) for the schema, architecture, performance,
and limits. **This is a different loader from the Qwen3.5/3.6 NVFP4 path below** — this one is
Gemma-4-only, single-expert (no MoE), and has no repo-root resolution (needs the real snapshot dir).

> [!TIP]
> **Speculative decoding with NVFP4.** The MTP draft head (`--assistant`) accepts NVFP4's tokens at
> a lower rate (~42%) than the original model (~89%): the assistant is matched to the original
> weights, not the NVFP4 checkpoint. For NVFP4, **prompt-lookup speculation** (model-agnostic,
> enabled by default) is the recommended speculative path. MTP still works, but the throughput gain
> is smaller (~28 tok/s vs ~57 with Q4_0+MTP). See
> [docs/nvfp4-safetensors.md](docs/nvfp4-safetensors.md) for details.

### 4 · Qwen3 / Qwen3.5 / Qwen3.6 (dense and MoE)

All Qwen checkpoints below are served through the same binary with **no model-select flag** —
just point `-m` at the file or directory. Continuous batching auto-enables itself (see
[Continuous batching & paged KV](#-continuous-batching--paged-kv)). For the full detection
reference, the official-FP8-vs-ModelOpt distinction, chat-dialect/tool-calling details, and a
diagnostics table, see [docs/qwen-models.md](docs/qwen-models.md).

**Qwen3 / Qwen3.5 GGUF (dense or MoE)** — any standard GGUF conversion works: the loader reads
`general.architecture` (`qwen3`, `qwen3moe`, or `qwen35`) directly from the file, and no
`--tokenizer` is needed (vocab is inline in the GGUF, same as Gemma).

```bash
./fucina -m ./models/<qwen-checkpoint>.gguf --host 0.0.0.0 --port 8080
```

**Qwen3.5 / Qwen3.6, official FP8-block safetensors** (DeepSeek-V3-style block quant) — dense or
MoE, distinguished automatically by the presence of expert tensors:

```sh
# Qwen3.5-9B (dense) / Qwen3.5-35B-A3B (MoE)
hf download Qwen/Qwen3.5-9B-FP8 --local-dir ./models/Qwen3.5-9B-FP8
hf download Qwen/Qwen3.5-35B-A3B-FP8 --local-dir ./models/Qwen3.5-35B-A3B-FP8

# Qwen3.6-27B (dense) / Qwen3.6-35B-A3B (MoE) — same architecture/loader as Qwen3.5 above
hf download Qwen/Qwen3.6-27B-FP8 --local-dir ./models/Qwen3.6-27B-FP8
hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir ./models/Qwen3.6-35B-A3B-FP8

fucina -m ./models/Qwen3.6-35B-A3B-FP8 --host 0.0.0.0 --port 8080
```

**Qwen3.6-35B-A3B, NVIDIA ModelOpt NVFP4/FP8 (MIXED_PRECISION)** — this is a **different
checkpoint and loader** from the official FP8 build above: NVIDIA's ModelOpt re-quantizes the
MoE experts, shared expert, and LM head to native **NVFP4**, while keeping attention and the
Gated-DeltaNet mixer in **per-tensor FP8** — smaller on-disk and resident footprint than the
official all-FP8 checkpoint, at a small accuracy cost from the 4-bit experts. Detected the same
way (`config.json` `model_type`/`quant_method`), with the FP8-vs-NVFP4 choice made **per tensor**
from which scale sibling is present — no flag selects it.

```sh
hf download nvidia/Qwen3.6-35B-A3B-NVFP4 --local-dir ./models/Qwen3.6-35B-A3B-NVFP4
fucina -m ./models/Qwen3.6-35B-A3B-NVFP4 --host 0.0.0.0 --port 8080
```

> [!NOTE]
> **Perf default:** dense **and** MoE Qwen3.5/3.6 checkpoints get their attention/GDN mixer and
> FFN/expert weights requanted to **Q4_K** at load time (smaller resident weight set, faster
> decode than the on-disk FP8). Opt back into pure FP8 with `FUCINA_MOE_FP8=1` if you need to
> compare against the unquantized checkpoint.

> [!NOTE]
> **Tool calling / thinking / structured output** work over the standard OpenAI wire shape for
> Qwen too — see [HTTP API](#-http-api). One caveat: `response_format`/`json_schema` (constrained
> JSON decoding) is currently rejected with HTTP 501 for every Qwen checkpoint, because it isn't
> supported under continuous batching and Qwen is always served through that path.

---

## 🧵 Continuous batching & paged KV

Independent requests are served through a per-step scheduler over a **paged, multi-sequence KV
cache** (vLLM-style block table + free-list) instead of one lock held for the whole request —
concurrent clients share each batched forward pass, and CUDA graphs are captured per active batch
size.

- **Gemma-4: opt-in.** Off by default (the default single-flight path is faster for one client at
  a time and has MTP speculation, which the batch path lacks). Turn it on with `--batch` (implies
  `--paged-kv`) — equivalent to the legacy `FUCINA_PAGED_KV=1 FUCINA_BATCH=1` env pair — when you
  are actually serving concurrent clients.
- **Qwen3 / Qwen3.5 / Qwen3.6: mandatory, auto-enabled.** There is no single-flight path for Qwen
  (its prefill entry points decline), so fucina detects the architecture and turns batching on for
  you — `fucina -m <qwen-checkpoint>` just works, with or without `--batch`. If the paged-KV pool
  fails to allocate, startup fails fast with a message telling you to lower `--ctx` or raise
  `--gpu-mem-util`, rather than serving requests that would 500.
- **Speculative decoding inside the batch path** is prompt-lookup only (n-gram, free) and applies
  automatically to **dense** checkpoints (Gemma or Qwen); it's disabled for **MoE** checkpoints.
  The Gemma-4 MTP draft head (`--assistant`) is not exercised in the batch path.
- **Burst-admission coalescing:** when the scheduler wakes from idle, it holds a short escalating
  window (a few ms, capped at 150 ms) so near-simultaneous requests land in the same batch instead
  of admitting one-by-one — unconditional scheduler behavior, no flag.
- **Known gap:** `response_format`/`json_schema` (constrained decoding) is rejected with HTTP 501
  under the batch path — it only works for Gemma-4 single-flight today.

See [`docs/continuous-batching.md`](docs/continuous-batching.md) for the full design (paged-KV
allocator, split-K paged attention, per-batch-size CUDA graphs) and
[`docs/qwen35-beat-vllm-plan.md`](docs/qwen35-beat-vllm-plan.md) for the Qwen-specific measurement
record.

---

## 🎯 Speculative decoding

Speculative decoding is **on by default** (`--spec`). Each step *drafts* candidate tokens cheaply,
then the model verifies them all in **one batched weight pass** and keeps the longest matching
prefix — so accepted drafts cost far less than one decode each. Every verified position is drawn
from the **target model's own distribution**, so the output is **identical to plain decoding** at
the same settings — drafting only changes speed, never the result.

This section describes the Gemma-4 single-flight path, where both drafters below are available.
Qwen checkpoints run through the continuous-batching path instead (see
[Continuous batching & paged KV](#-continuous-batching--paged-kv)): prompt-lookup speculation
applies automatically to **dense** Qwen checkpoints, is off for **MoE**, and the MTP draft head
below (`--assistant`) does not apply to Qwen at all.

Two drafters work together (Gemma-4):

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

The chat dialect (Gemma-4 native format, or Qwen ChatML + Qwen3-Coder XML tool calls) is picked
**automatically from the loaded vocab** — no flag. The wire format is OpenAI-shaped either way:

```sh
curl http://localhost:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "weather in Paris?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the weather",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "max_tokens": 64
}'
```

The response carries standard `choices[0].message.tool_calls[].function.{name,arguments}` and
`finish_reason: "tool_calls"` regardless of dialect — for Qwen, the server parses the model's raw
Qwen3-Coder XML output (`<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`)
back into that same shape. Force a specific tool with
`"tool_choice": {"type":"function","function":{"name":"get_weather"}}`.

Reasoning/thinking works the same for both dialects: `"reasoning_effort": "low"|"medium"|"high"` or
`"thinking": true/false` in the request; the model's reasoning appears in
`choices[0].message.reasoning_content`.

> [!IMPORTANT]
> `response_format`/`json_schema` (constrained JSON decoding) is implemented but is route-guarded
> off the continuous-batching path with HTTP `501 unsupported_under_batching` — since every Qwen
> checkpoint is served through that path, `response_format` currently **never works for Qwen**. It
> works for Gemma-4 in the default (non-`--batch`) single-flight mode.

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  cmd/fucina           CLI: server / one-shot / interactive          │
├───────────────────────────────────────────────────────────────────┤
│  internal/server      OpenAI-compatible HTTP API + KV cache          │
│  internal/server/batch continuous-batching scheduler (Qwen; opt-in   │
│                        for Gemma-4)                                  │
│  internal/chat        chat-template + tool-call dialects (Gemma/Qwen)│
│  internal/tokenizer   SentencePiece/Unigram (Gemma) + native HF BPE  │
│                        (Qwen tokenizer.json)                         │
├───────────────────────────────────────────────────────────────────┤
│  internal/engine/cuda CGO bridge (Go ⇄ C)                            │
├───────────────────────────────────────────────────────────────────┤
│  cuda/                CUDA C++ kernels: Gemma-4, Qwen3/3.5/3.6,      │
│                        GDN mixer, MoE grouped-expert GEMM (sm_121a)  │
└───────────────────────────────────────────────────────────────────┘
```

- **`cuda/`** — the forward pass for every architecture: quantized GEMV/GEMM (Q4_0/Q6_K/Q8_0/Q4_K
  `dp4a`), FP8/NVFP4 flash attention, the Qwen3.5/3.6 Gated-DeltaNet mixer, MoE grouped-expert GEMM,
  RoPE, RMSNorm, GPU sampling, CUDA-graph capture/replay, and paged multi-sequence batching.
  Compiled to `libfucina.a`. Key files: `gemma4_detect.h` (arch/format detection),
  `qwen35_fp8_loader.h` (Qwen3.5/3.6 FP8-block + ModelOpt NVFP4 safetensors loader),
  `nvfp4_loader.h`/`safetensors.h` (Gemma-4 NVFP4), `gemma4_kernels.cu` (everything else).
- **`internal/engine/cuda`** — the CGO bridge, wrapping the opaque CUDA engine as a Go `Engine`
  (`NewEngine`, `Prefill`, `Decode`, `GenerateSpec`, `LoadAssistant`, `IsQwen3Family`,
  `SeqAdd`/`StepBatch`, …).
- **`internal/tokenizer`** — Gemma SentencePiece/Unigram (vocab 262144, from the GGUF) and a native
  HuggingFace BPE reader (`tokenizer.json`, byte-fallback) for Qwen/NVFP4/FP8 checkpoints.
- **`internal/chat`** — the `Dialect` interface (`Gemma`, `Qwen`), auto-selected from the loaded
  vocab; chat-template rendering and tool-call parsing/formatting for each.
- **`internal/server`** — the HTTP server, thinking channel, `response_format`/JSON-schema
  constrained decoding (`internal/grammar`), and prefix-reuse KV cache (single-flight path).
- **`internal/server/batch`** — the continuous-batching scheduler: admission/burst-coalescing,
  per-step batch build, per-sequence spec gating (dense-only), eviction.
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
| `-m, --model` | (required) | GGUF file, or a safetensors file/directory (FP8-block, NVFP4, or ModelOpt mixed); architecture and quant auto-detected |
| `--tokenizer` | (auto) | `tokenizer.json` or `.gguf` to source vocab from; only needed if auto-discovery (sibling of `-m`) fails |
| `--ctx` | `262144` | Context size in tokens (max/default 262144; lower it to save memory) |
| `--temp` | `1.0` | Sampling temperature (gemma-4 default; `0` = greedy) |
| `--top-k` | `64` | Top-K sampling (gemma-4 default) |
| `--top-p` | `0.95` | Top-P / nucleus sampling (gemma-4 default) |
| `--thinking` | `off` | Default reasoning channel: `off`/`on`/`low`/`mid`/`high`/`xhigh` (Gemma and Qwen) |
| `--assistant` | (none) | Gemma-4 MTP draft-head GGUF for speculation (Gemma-4 only; no effect on Qwen) |
| `--spec` | `true` | Speculative decoding (prompt-lookup; MTP too when `--assistant` is set, Gemma-4 single-flight only) |
| `--draft-k` | `6` | Max speculative draft length per step |
| `--paged-kv` | `false` | Allocate the paged multi-sequence KV pools; auto-forced on for any Qwen3 checkpoint |
| `--batch` | `false` | Continuous batching over the paged engine (implies `--paged-kv`); auto-forced on for any Qwen3 checkpoint, opt-in for Gemma-4 |
| `--gpu-mem-util` | `0.90` | Fraction of total GPU memory the engine may use; caps `--ctx`/paged-pool sizing to fit |
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
| `FUCINA_NO_PACKED=1` | Drop the repacked-Q4_0 decode-GEMV weight copy (Gemma-4). **Frees ~7 GiB VRAM** at the cost of ~2–3% slower decode; output is bit-identical. The repacked copy is a second, coalesced-load layout of the Q4_0 projection weights kept resident only to speed the bandwidth-bound decode hot path. Recommended on memory-constrained hosts. |
| `FUCINA_SLIDING_RING=N` | Sliding-window KV ring capacity in tokens (default **8192**, floored at `window+spec_max`), Gemma-4 only. Caps the sliding cache so it stays ctx-independent (~1.5 GiB at 8192). `N` also bounds how far a prefix-reuse rewind stays exact — deeper rewinds (e.g. editing context older than `N-1024` tokens) fall back to a full re-prefill. With `--ctx ≤ N` behavior is identical to the old flat cache. Lower = less VRAM; higher = deeper exact rewind (~+190 MiB per +1024). |
| `FUCINA_PAGED_KV=1` | Allocate the paged KV pools (block table + free-list), capacity-sized to free VRAM. Prerequisite for continuous batching; equivalent to `--paged-kv`. Always on for Qwen3. |
| `FUCINA_BATCH=1` | Route the server through the continuous-batching scheduler (needs `FUCINA_PAGED_KV=1`): independent sequences share each batched forward instead of the per-request lock. Equivalent to `--batch`; always on for Qwen3. |
| `FUCINA_MOE_FP8=1` | Disable the default Q4_K requant of the Qwen3.5/3.6 attention/GDN mixer and FFN/expert weights; serve pure FP8 instead. |
| `FUCINA_QWEN35_FP4=1` | Opt-in NVFP4 activations for the Qwen3.5/3.6 hybrid path (in addition to the default weight quant). |
| `FUCINA_NO_BATCH_SPEC=1` | Disable prompt-lookup speculation inside the continuous-batching scheduler (Gemma or Qwen dense). |
| `FUCINA_PAGED_MAXCTX=N` | Override the paged-KV pool sizing context (default 32768, clamped to `--ctx`). |
| `FUCINA_PAGED_MAXSEQS=N` | Override the auto-computed max concurrent sequences the paged pool backs (auto range 1–64). |
| `FUCINA_KV_NVFP4=1` | Quantize the KV cache to **NVFP4** precision (native Blackwell FP4 E2M1 + per-16 E4M3 block scale). Memory-only (~1.78× smaller KV), small accuracy cost; default OFF keeps flat FP8 byte-identical. |
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
# CPU-only unit tests (no GPU, no cgo, no model file required) — this is what CI runs
go test ./internal/server/ ./internal/server/batch/ ./internal/tokenizer/ ./internal/sampler/ ./internal/chat/
make check      # go vet + the same pure-Go tests + gofmt check

# GPU smoke test (requires the DGX Spark GB10)
make smoke      # builds, then: fucina --prompt "Hello, world!" --predict 32 --temp 0

# GPU correctness + performance (requires the GB10 + a Gemma model; MODEL=… overridable)
make bench               # batch==single self-test + greedy byte-identity, then prefill/decode tok/s
make paged-kv-device-test  # paged KV reads bit-identical to the contiguous cache
make packed-kv-test        # packed 4.5-bit NVFP4 KV storage bit-identical to the fake-quant
make kv-quant-explore      # offline FP8 / NVFP4 / TurboQuant codec comparison (host-only)

# Qwen3/3.5/3.6 GPU correctness gates (require the GB10 AND real downloaded checkpoints at
# hardcoded /opt/spark/models/... paths — override with QWEN3_DENSE_MODEL=/QWEN3MOE_MODEL=/
# QWEN35_MODEL=/QWEN35_FP8_MODEL=/QWEN35_MOE_FP8_MODEL=. NOT runnable in CI or without hardware+weights.
make gpu-gates          # Qwen3 dense/MoE parity + spec + prefix/suffix regression gates
make qwen35-batch-test  # Qwen3.5/3.6 paged-batch + CUDA-graph decode gate
make qwen35-burst-test  # diverse-prompt burst-admission + prefill-determinism gate
```

The CUDA engine is validated for **bit-exactness** against reference paths (greedy byte-identical
output, `compute-sanitizer` memcheck clean) and benchmarked with the `scripts/` harnesses
([`pi_bench.py`](scripts/pi_bench.py), [`parity_bench.py`](scripts/parity_bench.py),
[`benchmark_gem4.py`](scripts/benchmark_gem4.py)) for Gemma, and against llama.cpp/vLLM/`torch`
oracles for Qwen (see the `qwen35-*-test` targets in the [Makefile](Makefile)).

---

## 📌 Status

**Experimental, hardware-specific, no support.** fucina targets exactly one platform — the NVIDIA
DGX Spark GB10 (Blackwell `sm_121a`) with CUDA 13.0 — and is not portable to other GPUs or
toolchains as-is. It is an open research/lab project from **[hikmaai.io](https://hikmaai.io)**,
provided **as-is with no warranty and no support commitment**; issues and PRs are handled
best-effort. Weights you supply are governed by their own upstream license: Gemma 4 by the
[Gemma license](https://ai.google.dev/gemma/docs/gemma_4_license); Qwen checkpoints by their
respective Qwen license terms on Hugging Face. fucina's own code is Apache-2.0.

- **Code:** [Apache-2.0](LICENSE) · **Third-party notices:** [NOTICE](NOTICE)
- **Model support:** Gemma 4 12B (dense, GGUF/NVFP4) and Qwen3/Qwen3.5/Qwen3.6 (dense and MoE,
  GGUF/FP8-block safetensors/NVFP4) are both first-class; continuous batching is mandatory for
  Qwen and opt-in for Gemma (see [Continuous batching & paged KV](#-continuous-batching--paged-kv)).
- **Roadmap:** for Qwen, the current known gaps vs `vLLM` are cold-prompt (turn-1) TTFT and
  aggregate throughput at high concurrency (see [Performance](#-performance) and
  [`docs/qwen35-beat-vllm-plan.md`](docs/qwen35-beat-vllm-plan.md)); `response_format`/`json_schema`
  does not work under continuous batching, so it's unavailable for Qwen today. For Gemma, per-slot
  spec decode inside the batch path is not yet built, and the NVFP4 KV codec / packed 4.5-bit KV
  storage are verified but still opt-in behind a quality gate
  ([`docs/kv-quant-exploration.md`](docs/kv-quant-exploration.md)). Also: harden the experimental
  DiffusionGemma path; an sm_120 (RTX 50-series) port to loosen the single-hardware constraint.

## 🙏 Acknowledgements

- The name **fucina** is Italian for *forge* — a smithy, and figuratively a *crucible of ideas*.
- **Google** for [Gemma 4](https://ai.google.dev/gemma) and the QAT GGUF / MTP assistant releases.
- **Alibaba / the Qwen team** for [Qwen3, Qwen3.5, and Qwen3.6](https://huggingface.co/Qwen) and
  their official FP8-block checkpoints.
- **NVIDIA** for [ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) and its NVFP4/FP8
  mixed-precision checkpoint format.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — the GGUF format, quantized `dp4a`
  kernels, and the `draft-mtp` speculation design that this project measures itself against.
- **[vLLM](https://github.com/vllm-project/vllm)** — the continuous-batching / paged-KV design this
  project's batching implementation follows, and the head-to-head benchmark target for Qwen.
- **[unsloth](https://huggingface.co/unsloth)** for the GGUF conversions used here.

<div align="center"><sub>Built by <a href="https://hikmaai.io">hikmaai.io</a> · formerly <code>gem4d</code></sub></div>
