# fucina v0.1.0

> **Draft, not yet published.** These are the release notes as drafted before the
> Qwen3/Qwen3.5/Qwen3.6 work landed on `qwen35-hybrid` — v0.1.0 has not been tagged (see
> [RELEASE_CHECKLIST.md](../../RELEASE_CHECKLIST.md)). They describe a Gemma-4-only snapshot;
> current README.md/CHANGELOG.md are the source of truth for what fucina supports today.

**Gemma 4 12B inference, forged for the NVIDIA DGX Spark GB10.** First public release
(formerly the internal `gem4d`).

> ⚠️ **Experimental · single-hardware · no support.** fucina is built and tested only for the
> DGX Spark GB10 (Blackwell `sm_121a`, CUDA 13). It is not portable to other GPUs as built.
> Apache-2.0, provided as-is.

## Highlights
- Full Gemma 4 12B forward pass on-GPU: **FP8 Tensor-Core attention + FP8 KV cache**.
- **Position-independent CUDA-graph decode** (single-token + batched speculative-verify).
- **MTP speculative decoding** — prompt-lookup + optional draft head (`--assistant`),
  distribution-exact at any temperature.
- **Prefix-reuse KV cache**, on-GPU sampling, native Q4_0 (QAT) / Q8_0 GGUF.
- **OpenAI-compatible server** (streaming, tool calls, thinking channel) + one-shot + REPL.
- Optional, experimental **DiffusionGemma 26B-A4B** engine (`-dm`).

## Performance
On a fair side-by-side vs `llama.cpp` (identical transcript, MTP on both): decode **parity-to-ahead
overall, +15–20% at high context**; prefill steady-state tied. Validated bit-exact.

## Getting started
Requires a DGX Spark GB10 + CUDA 13 + Go 1.26. See the [README](../../README.md):
`make`, download a Gemma 4 GGUF, `fucina -m model.gguf --port 8080`.

## Known limitations
GB10-only · single slot · server has no auth · no support SLA.
