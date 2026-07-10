# Qwen3 / Qwen3.5 / Qwen3.6 — model support, checkpoint paths, and serving

fucina serves the whole Qwen3 family — dense and sparse-MoE, GGUF and safetensors — through the
same binary as Gemma-4, with the architecture and quant format auto-detected from the checkpoint
itself. This doc is the deep-dive companion to the [README](../README.md) model matrix and
[llms.txt](../llms.txt): checkpoint-path gotchas, what the detector actually reads, and how to
diagnose a failed load or a serving quirk.

## Family tree and detection

| Generation | Shape | Detected from | Loader |
|---|---|---|---|
| Qwen3 | dense | GGUF `general.architecture = "qwen3"` (`cuda/gemma4_detect.h`, `gemma4_detect_qwen3`) | GGUF dp4a/Q4_K path |
| Qwen3 | MoE | GGUF `general.architecture = "qwen3moe"` (`gemma4_detect_qwen3moe`) | GGUF dp4a/Q4_K path, grouped experts |
| Qwen3.5 | hybrid dense | GGUF `general.architecture = "qwen35"` (`gemma4_detect_qwen3_5`), or safetensors `config.json.model_type` containing `"qwen3_5"` + `quant_method` `fp8`/`modelopt`, no expert tensors | `cuda/qwen35_fp8_loader.h` |
| Qwen3.5 | hybrid MoE | same safetensors detection, plus an `mlp.experts.0.gate_proj.weight` tensor (`qwen35_fp8_is_moe`) | `cuda/qwen35_fp8_loader.h` |
| **Qwen3.6** | dense or MoE | **identical detection to Qwen3.5** — same `model_type: "qwen3_5"`, same GGUF `"qwen35"` arch key | same loader |

"Qwen3.6" is a checkpoint generation name only. `Qwen/Qwen3.6-27B-FP8`, `Qwen/Qwen3.6-35B-A3B-FP8`,
and `nvidia/Qwen3.6-35B-A3B-NVFP4` all report `model_type: "qwen3_5"` in `config.json` and are
served by the exact same Gated-DeltaNet hybrid detector and loader as Qwen3.5. There is no
`"qwen36"` string anywhere in the detection code — don't grep for one.

The hybrid architecture itself: most layers are a Gated-DeltaNet (GDN) linear-attention mixer;
every `qwen35.full_attention_interval`-th layer is instead full softmax GQA. Dense checkpoints run
every layer over the full (dense) FFN; MoE checkpoints route each token through a top-k subset of
experts plus a shared expert.

## Two different FP4/FP8 checkpoint families — don't confuse them

| | Official Qwen FP8 (`Qwen/Qwen3.5-*-FP8`, `Qwen/Qwen3.6-*-FP8`) | NVIDIA ModelOpt (`nvidia/Qwen3.6-35B-A3B-NVFP4`) |
|---|---|---|
| Quant | DeepSeek-V3-style per-128 block FP8 across the bulk Linears | **Mixed precision**: NVFP4 (E2M1 + block scale) for experts/shared-expert/LM-head, per-tensor FP8 for attention/GDN |
| `config.json` | `quant_method: "fp8"` | `quant_method: "modelopt"` |
| Why pick it | Reference-accuracy FP8, matches vLLM's serving numerics most closely | Smaller on-disk and resident footprint (4-bit experts dominate MoE param count), small extra accuracy cost |
| fucina loader | `cuda/qwen35_fp8_loader.h` (same file) | `cuda/qwen35_fp8_loader.h` (same file, MIXED_PRECISION branch) |

Within the ModelOpt checkpoint, whether a given weight tensor is served as NVFP4 or FP8 is decided
**per tensor**, by which scale sibling is present next to it — not by a global flag:

- `<tensor>_scale_inv` (a grid) → native block-FP8.
- `<tensor>_scale` (scalar E4M3) + `<tensor>_scale_2` → native NVFP4.
- `<tensor>_scale` alone (scalar F8_E4M3) → per-tensor FP8.

## Checkpoint path resolution — read this before debugging a load failure

`-m` opens: a `.gguf` file; a single `.safetensors` file; or a directory containing
`model.safetensors.index.json` (sharded) or a lone `model.safetensors`, with a sibling
`config.json` (`cuda/safetensors.h: Model::open`). It does **not** generically descend into a
HuggingFace hub-cache's `snapshots/<hash>/` — with one exception:

- **Qwen3.5/3.6 FP8/ModelOpt weight loading** specifically globs
  `<dir>/snapshots/*/model.safetensors.index.json` (`q35moe_resolve_dir` in `cuda/gemma4_kernels.cu`),
  so pointing `-m` at a raw hub-cache **repo root** — `models--Qwen--Qwen3.6-35B-A3B-FP8/`, what
  `hf download <repo>` produces *without* `--local-dir` — works for the **weights**.
- **Tokenizer auto-discovery does not share that glob.** `cmd/fucina/main.go`'s
  `siblingTokenizerJSON` only looks for `tokenizer.json` directly inside `-m`'s directory. Pointed
  at a repo root, it won't find the `tokenizer.json` that actually lives under
  `snapshots/<hash>/`, and tokenizer init fails.
- **The generic (Gemma-4) NVFP4 loader has no repo-root resolution at all.** It needs the real
  snapshot-shaped directory unconditionally.

**Robust rule, for every safetensors checkpoint regardless of family:**

```sh
hf download <repo> --local-dir ./models/<name>
fucina -m ./models/<name> ...
```

`--local-dir` produces a flat directory with `config.json` + shards + `tokenizer.json` at the top
level. This satisfies the weight loader and the tokenizer auto-discovery identically for Qwen and
Gemma NVFP4 alike — no `--tokenizer` flag, no repo-root-vs-snapshot ambiguity to reason about.

If you must use the bare `hf download` cache layout, pass the actual
`~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<hash>/` path to `-m` (this is also the
convention this repo's own test fixtures and golden-generation scripts use, e.g.
`scripts/gen_qwen_golden.py`), not the `models--...` parent.

## Serving behavior

- **Continuous batching (paged KV + per-step scheduler) is mandatory and auto-enabled** for every
  Qwen3/3.5/3.6 checkpoint — `cmd/fucina/main.go` detects `eng.IsQwen3Family()` and turns batching
  on regardless of `--batch`/`FUCINA_BATCH`. If the paged-KV pool fails to allocate, startup fails
  fast: `fucina: Qwen3 requires paged KV but the pools are not active (allocation failed — lower
  --ctx or raise --gpu-mem-util)`.
- **Perf default:** dense and MoE Qwen3.5/3.6 checkpoints get the attention/GDN mixer and
  FFN/expert weights requanted to Q4_K at load (smaller resident set, faster decode than on-disk
  FP8). `FUCINA_MOE_FP8=1` reverts to pure FP8.
- **Speculative decoding inside the batch scheduler** is prompt-lookup only, and gated on
  `NExperts() == 0` — it runs automatically for dense Qwen checkpoints, is off for MoE. The Gemma-4
  MTP draft head (`--assistant`) is never exercised for Qwen (Qwen has no single-flight path to run
  it on).
- **Burst-admission coalescing** in the scheduler holds a short escalating window (few ms, capped
  at 150 ms) when it wakes from idle, so near-simultaneous requests land in the same batch instead
  of admitting one at a time. Unconditional, no flag.
- **Hybrid state capacity is lazy and context-proportional.** `--parallel` can request up to 32
  rows. The planner reserves 8192 context tokens per slot by default while FULL-attention KV grows
  transactionally to `--ctx`; `FUCINA_QWEN35_SLOT_CTX=N` changes the reservation or restores the
  old worst-case policy by setting it equal to `--ctx`.
- **`--timings` includes Qwen MoE prefill phases.** Wide prefills log expert dequant, router/route,
  grouped-expert, shared-expert, and remaining mixer/attention/head milliseconds. The diagnostic
  inserts CUDA synchronization boundaries and should not be used for normal serving benchmarks.
- **`response_format`/`json_schema` (constrained JSON decoding) is rejected under continuous
  batching** with HTTP `501 unsupported_under_batching` — since Qwen is always served through that
  path, structured output does not currently work against any Qwen checkpoint. It works for Gemma-4
  in the default (non-`--batch`) mode.

## Chat dialect and tool calling

The dialect (`internal/chat`: `Gemma` vs `Qwen`) is chosen automatically from the loaded vocab —
`<|im_start|>` present selects `Qwen`, no flag involved. The HTTP wire format is OpenAI-shaped for
both: `tools`/`tool_choice` in the request, `tool_calls` in the response. Internally, Qwen tool
calls are emitted by the model as Qwen3-Coder's XML form
(`<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`) and parsed
back into the standard `tool_calls` JSON shape — callers never see the XML.

One dialect-specific detail worth knowing if you inspect multi-turn message history: when a
historical assistant turn is re-rendered into a new prompt, the Qwen dialect always injects an
empty, pre-closed `<think>\n\n</think>\n\n` before that turn's content, regardless of what
`reasoning_content` you echo back for older turns — this keeps re-rendered prompts token-identical
to what was actually generated (needed for the prefix/state KV cache to keep matching across
turns). Only the reasoning of the assistant turn(s) *after* the last real user message is preserved
verbatim; anything older is always collapsed to the empty block. This is transparent to a normal
chat-completions caller — you don't need to insert `<think>` markers yourself.

## Diagnostics

| Symptom | Likely cause | Fix |
|---|---|---|
| Safetensors `-m` load fails with a missing-index/config error | Pointed at a HF hub-cache repo root that has no `config.json` at the top level | Use `--local-dir` on download, or point `-m` at the actual `snapshots/<hash>/` dir |
| Tokenizer init fails right after weights loaded fine | Same repo-root vs snapshot-dir mismatch — the Qwen weight loader resolved the repo root, but tokenizer auto-discovery didn't | Pass `--tokenizer <snapshot-dir>/tokenizer.json`, or re-download with `--local-dir` |
| `fucina: Qwen3 requires paged KV but the pools are not active ...` | Paged-KV pool allocation failed (usually OOM) | Lower `--ctx`, raise `--gpu-mem-util` toward `1.0`, or use a smaller checkpoint |
| `response_format`/`json_schema` request 501s | Expected — not supported under continuous batching, which every Qwen checkpoint uses | Not fixable today for Qwen; works for Gemma-4 single-flight |
| `--assistant` flag has no visible effect | It's Gemma-4-only; Qwen ignores it | Use prompt-lookup speculation (already automatic for dense Qwen) |

## Testing

The Qwen-specific GPU gates require a real GB10 and real downloaded checkpoints at hardcoded
`/opt/spark/models/...` paths (override with `QWEN3_DENSE_MODEL=`, `QWEN3MOE_MODEL=`,
`QWEN35_MODEL=`, `QWEN35_FP8_MODEL=`, `QWEN35_MOE_FP8_MODEL=`) — they are **not** runnable in CI or
without hardware and weights on disk:

```sh
make gpu-gates          # Qwen3 dense/MoE parity + spec + prefix/suffix regression gates
make qwen35-detect-test # arch detection from GGUF metadata (host-only, no GPU)
make qwen35-batch-test  # paged-batch + CUDA-graph decode gate
make qwen35-burst-test  # diverse-prompt burst-admission + prefill-determinism gate
```

The pure-Go dialect, tool-calling, and tokenizer tests run anywhere, no GPU required:

```sh
go test ./internal/chat/ ./internal/tokenizer/ ./internal/server/
```

See also: [docs/continuous-batching.md](continuous-batching.md) for the paged-KV/scheduler design,
[docs/qwen35-beat-vllm-plan.md](qwen35-beat-vllm-plan.md) for the measured performance record
against vLLM, and [docs/nvfp4-safetensors.md](nvfp4-safetensors.md) for the (separate) Gemma-4
NVFP4 loader.
