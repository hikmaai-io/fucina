# Changelog

All notable changes to fucina are recorded here. This project is **experimental and pre-1.0** —
expect breaking changes between releases.

## [0.1.0] — unreleased

First public release as `github.com/hikmaai-io/fucina` (formerly the internal `gem4d`).

### Added
- **Gemma 4 12B inference engine for the NVIDIA DGX Spark GB10** (Blackwell `sm_121a`, CUDA 13):
  full forward pass on-GPU, FP8 Tensor-Core attention, FP8 E4M3 KV cache.
- **CUDA-graph decode** — position-independent graphs for single-token decode and the K-row
  batched speculative-verify forward, replayed each step; pre-captured at startup.
- **Speculative decoding** — prompt-lookup (free) plus an optional **MTP draft head**
  (`--assistant`); distribution-exact at any temperature.
- **Prefix-reuse KV cache**, **on-GPU sampling**, native **Q4_0 (QAT) / Q8_0** GGUF loading, and a
  packed-Q4_0 coalesced-load decode path.
- **OpenAI-compatible HTTP server** — `/v1/chat/completions` (streaming + non-streaming), tool
  calling, the gemma-4 thinking channel, `/v1/models`, `/health`, `/metrics`; plus one-shot and
  interactive CLI modes.
- **(Experimental, optional) DiffusionGemma 26B-A4B** engine via `-dm`.

### Performance
- Measured against `llama.cpp` on a fair side-by-side harness (`scripts/pi_bench.py`): decode at
  **parity-to-ahead overall and +15–20% at high context (≥5k tokens)**; prefill steady-state tied.
  All changes validated bit-exact (greedy byte-identical, `compute-sanitizer` clean).

### Changed
- **Sliding-window KV cache is now a capped ring buffer** instead of a flat per-position cache.
  Sliding-layer memory no longer scales with `--ctx`: ~1.5 GiB at the default ring size
  (`FUCINA_SLIDING_RING`, 8192 slots) vs ~21 GiB flat at 131072. Total FP8 KV cache drops to
  ~2.5 GiB at 131072 / ~3.5 GiB at 262144 (was ~27 / ~54 GiB). Decode/spec/MTP output is
  **bit-identical** to the flat cache (verified at 30k-token context with heavy ring wrap). Prefix-
  reuse stays exact within the ring window; a deeper rewind falls back to a full re-prefill. With
  `FUCINA_NO_PACKED=1` the engine fits in **~23 GiB at 131072** — hostable off the 128 GB GB10.

### Known limitations
- Runs **only** on the DGX Spark GB10; not portable to other GPUs as built.
- Single logical sequence / single slot; the server has **no authentication**.
- **No support commitment** — issues and PRs are handled best-effort.

[0.1.0]: https://github.com/hikmaai-io/fucina/releases/tag/v0.1.0
