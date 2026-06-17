# NVFP4 safetensors — native FP4 Gemma 4 on the GB10

fucina runs a **natively NVFP4-quantized Gemma 4 12B** loaded straight from a HuggingFace
safetensors checkpoint, as a **single weight store** that serves both prefill (cuBLASLt
block-scaled FP4 GEMM) and decode (fused FP4 GEMV) — no GGUF, no Q4_0 copy, no NVFP4→BF16
materialization. The model loads, generates coherent text, and speculatively decodes on one GB10.

## Usage

```bash
# A directory containing model.safetensors + config.json + tokenizer.json (auto-detected),
# or the .safetensors file directly. No --tokenizer needed: the HF tokenizer.json is read natively.
./fucina -m /path/to/gemma-4-12B-it-NVFP4 -p "Explain photosynthesis." -n 128

# Optional MTP draft head (a Gemma-4 GGUF assistant) for speculative decoding:
./fucina -m /path/to/gemma-4-12B-it-NVFP4 --assistant gemma-4-mtp.gguf -p "..." -n 128
```

Verified checkpoint: `RedHatAI/gemma-4-12B-it-NVFP4` (compressed-tensors, 10.3 GB). NVIDIA
ModelOpt `nvidia/*-FP4` checkpoints use the same schema with different key names (see below).

## Why safetensors (not GGUF)

NVFP4 weights ship only as safetensors (NVIDIA ModelOpt or compressed-tensors / llm-compressor);
GGUF has no NVFP4 block-scaled type. Deriving NVFP4 from QAT-Q4_0 would push E2M1 rounding error
onto the whole model — loading native NVFP4 avoids that.

## On-disk schema (verified against the real file — two naming conventions)

**Dequant:** `real = e2m1(nib) · e4m3(block) · global_mul`, block size 16. The raw global differs
by producer — compressed-tensors stores the LARGE reciprocal `weight_global_scale = (6·448)/amax`
(q_proj = 7392) so `global_mul = 1/raw` (DIVIDE); ModelOpt stores the small `amax/(6·448)` so
`global_mul = raw`. `nvfp4ld::global_mul` normalizes both into one multiply.

| per quantized Linear `<p>` | ModelOpt (`nvidia/*-FP4`) | compressed-tensors (`RedHatAI/*`) |
|---|---|---|
| packed E2M1 `U8 [out,in/2]` (low nibble=even k) | `<p>.weight` | `<p>.weight_packed` |
| E4M3 block scale `[out,in/16]` linear | `<p>.weight_scale` | `<p>.weight_scale` |
| FP32 global | `<p>.weight_scale_2` | `<p>.weight_global_scale` |
| activation scale (ignored — dynamic) | `<p>.input_scale` | `<p>.input_global_scale` |

Layer prefix `model.language_model.layers.*` (Gemma 4 multimodal) or `model.layers.*`. embeddings,
lm_head (untied, ~2 GB BF16), all norms, `layer_scalar` (per-layer output scale — load-bearing:
omitting it blows the residual ~19× and saturates the 30.0 softcap), q/k_norm, pos_embedding and
biases stay BF16/F32. Vision/audio towers (the `ignore` list) are not loaded.

## Components

| File | Role |
|---|---|
| `cuda/safetensors.h` | container parser: u64+JSON header, sharded `index.json`, mmap (host C++) |
| `cuda/nvfp4.h` | dequant oracle: E2M1 LUT, software E4M3, reconstruct (host+device) |
| `cuda/nvfp4_loader.h` | name mapping (both conventions) + config.json parse (quant kind, tie, ignore-glob) |
| `cuda/nvfp4_gemv.cuh` | fused decode GEMV (register-blocked) + weight-read-once batched verify + BF16/FP8 head GEMVs |
| `cuda/nvfp4_inspect.cc` | opens a real checkpoint, detects, validates shapes, dequants a row |
| `internal/tokenizer/hf_bpe.go` | native HF `tokenizer.json` BPE encoder (byte-fallback, metaspace) |
| `cuda/gemma4_kernels.cu` | `FORMAT_NVFP4` create-fork, residency, decode + spec-verify routing |

C-vs-C++ rule (rooted in fucina's "lean where it counts"): the loader leans on STL but ONLY in the
load-once create/residency path; every per-token/decode path stays raw-pointer C-style — no STL
ever crosses into the hot loop. Unit tests: `make nvfp4-test` (parser, math, name-mapping, decode +
batched-verify kernel correctness/profitability) + `go test ./internal/tokenizer` (BPE vs the HF
`tokenizers` library, token-for-token).

## Architecture

- **Residency** (`nvfp4_load_from_safetensors`): packed E2M1 → `d_fp4_w`; E4M3 scales kept twice —
  swizzled `d_fp4_wsc` (cuBLASLt prefill) + linear `d_fp4_wsc_lin` (decode); global multiplier →
  `d_fp4_gsw`. embed/lm_head/norms → BF16. Single store: 6.81 GB (no Q4_0 duplicate).
- **Prefill** reuses the existing `gemm_nvfp4` (cuBLASLt block-scaled FP4 tensor-core GEMM).
- **Decode (N=1)** uses a register-blocked fused GEMV (`nvfp4_gemv`, `NVFP4_GEMV_ROWS=4`,
  152–193 GB/s). At N=1 it is a memory-bound GEMV — FP4 tensor cores cannot help; the lever is
  reading the 4.5-bit footprint once at full LPDDR5X bandwidth.
- **Spec-verify (K tokens)** uses a **weight-read-once batched** NVFP4 GEMV (transposed activation,
  register-blocked, `acc[ROWS][K]`) — each weight read once for all K rows (2.8–3.4× the per-row
  loop), CUDA-graph-capturable. This is what makes speculation profitable.

## Performance (gemma-4-12B-it-NVFP4 vs the Q4_0 GGUF, GB10, GPU at 96% util)

| | base (`--draft-k 0`) | + speculation |
|---|---|---|
| **NVFP4** | ~20 tok/s | ~28 (MTP) / ~28 (prompt-lookup on structured text) |
| Q4_0 (reference) | 22.5 tok/s | ~57 (MTP) |

NVFP4 base nearly matches Q4_0 base (same 4.5 bit/weight; decode is memory-bound). MTP went from
net-negative (11.2, slower than base) to profitable (~28, above base) once the spec-verify read
weights once.

## Known limits (investigated — fundamental, not bugs)

- **MTP draft-head gap (28 vs Q4_0's 57):** the Gemma-4 MTP assistant predicts NVFP4's next token
  only ~42% of the time (avg ~2.0 accepted/step) vs ~89% (avg ~5.3) for Q4_0 — the confidence
  threshold is already 0, so this is pure draft quality: the assistant is matched to the original
  model, not the NVFP4 checkpoint. Closing it needs an NVFP4-matched draft head (retraining).
  **Prompt-lookup speculation is model-agnostic and already efficient for NVFP4** (~28 tok/s on
  structured/repetitive text) — the better speculation source for agentic workloads.
- **FP8 LM head (OFF by default, `FUCINA_FP8_HEAD=1` opt-in):** quantizing the 2 GB untied head to
  per-row E4M3 would halve the per-token head read, but E4M3 (3 mantissa bits) flips the argmax over
  the 262144-vocab head and degrades greedy generation ("capital of France is France"). The kernels
  are correct — it is a fundamental precision limit, so the BF16 head stays resident.
