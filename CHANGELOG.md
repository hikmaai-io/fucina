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
- **Continuous batching over a paged KV cache** (`FUCINA_PAGED_KV=1 FUCINA_BATCH=1`):
  vLLM-style block-pool allocator with free-list and per-sequence block tables,
  paged split-K attention bit-identical to the contiguous path, CUDA graphs per
  batch size (1..MAX_SEQS), Go scheduler with per-step batching (AddSeq/StepBatch/
  RemoveSeq/Capacity), and server routing (serveBatch). Opt-in; the default
  single-flight path is unchanged. See [`docs/continuous-batching.md`](docs/continuous-batching.md).
- **Per-sequence sampling params** in the batch path: temperature, top-k, top-p,
  min-p, and seed are stored per slot and applied on-device every sampled token
  (temp≤0 selects an exact greedy argmax, byte-identical to the single-seq path).
- **NVFP4-KV codec** (`FUCINA_KV_NVFP4=1`): native Blackwell FP4 (E2M1 + per-16
  E4M3 block scale) fake-quant at every KV store site. Prefill/decode throughput
  identical to FP8 within noise (memory-only); ~1.78× memory savings with the
  packed storage codec. Opt-in; default stays FP8.
- **Packed 4.5-bit NVFP4 KV storage codec** (`make packed-kv-test`): `E/2` FP4
  nibbles + `E/16` block scales = 0.5625 B/element. Proven bit-faithful to the
  fake-quant (0/8.4M mismatches). Staged behind a quality gate, not yet the default.
- **KV quant exploration** (`make kv-quant-explore`, `docs/kv-quant-exploration.md`):
  offline comparison of FP8, per-token FP8, NVFP4, and TurboQuant at fucina's
  operating point. Finding: FP8 is already near-lossless; TurboQuant declined
  (no win over NVFP4).
- **CLI `/thinking` command** (off/on/low/medium/high/xhigh): thought-channel
  budget with force-close; thought rendered dimmed, stripped from history.
- **REPL now applies the chat template** and honours `--thinking` and
  `--repeat-penalty`, matching the server behaviour.
- **`/help` slash command** (`/h`, `/?`, `/commands`) in both the dense and
  DiffusionGemma REPLs, plus unknown-command detection.
- **`make bench`**: correctness gates (batch==single self-test, greedy byte-identity)
  and prefill/decode throughput smoke.
- **`make paged-kv-device-test`**: GPU test proving paged KV reads are bit-identical
  to the contiguous cache.

### Fixed
- **Repeat-penalty silently disabled**: both the non-spec CLI loop and the one-shot
  path passed `pastTokens=nil` to the host sampler, making repeat-penalty a no-op.
  Now passes `kv.CurrentTokens()` so the penalty is applied.
- **Batch over-subscription** (high-sev): `seq_capacity()` reported free slots
  regardless of block-pool budget, allowing admission of more sequences than the
  pool could back. Now returns `paged_cap - used`, guaranteeing each admitted
  sequence its maxctx blocks.
- **Whole-batch eviction on KV exhaustion** (high-sev): a single row unable to
  grow its block table returned -1 for the whole batch. Now `step_batch` admits
  per-row: a failing row is excluded, the others proceed.

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
