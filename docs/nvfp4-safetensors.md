# NVFP4 safetensors loading — design & status

Goal: feed the Blackwell FP4 tensor cores a **natively NVFP4-quantized** Gemma 4 (loaded from a
safetensors checkpoint) as a **single weight store** that serves both prefill (cuBLASLt
block-scaled GEMM) and decode (fused GEMV) — dropping the duplicate Q4_0 + persistent-NVFP4
copy (~6 GB on the 12B, the memory win for the RAM-constrained client target).

## Why safetensors (not GGUF)

NVFP4 weights ship only as safetensors (NVIDIA ModelOpt or compressed-tensors / llm-compressor);
GGUF has no NVFP4 block-scaled type. Deriving NVFP4 from QAT-Q4_0 would push E2M1 rounding error
onto the whole model — loading native NVFP4 avoids that. Real checkpoint: `RedHatAI/gemma-4-12B-it-NVFP4`.

## On-disk schema (verified — two naming conventions)

**Dequant (VERIFIED on the real file):** `real = e2m1(nib) · e4m3(block) · global_mul`, block 16.
The raw global differs by producer — compressed-tensors stores the LARGE reciprocal
`weight_global_scale = (6·448)/amax` (q_proj = 7392) so `global_mul = 1/raw` (DIVIDE); ModelOpt
stores the small `amax/(6·448)` so `global_mul = raw`. `nvfp4ld::global_mul` normalizes both.

Per quantized Linear `<p>` (block size 16):

| | ModelOpt (`nvidia/*-FP4`) | compressed-tensors (`RedHatAI/*`) |
|---|---|---|
| packed E2M1 `U8 [out,in/2]` | `<p>.weight` | `<p>.weight_packed` |
| E4M3 block scale `[out,in/16]` linear | `<p>.weight_scale` | `<p>.weight_scale` |
| FP32 global `amax/(6·448)` | `<p>.weight_scale_2` | `<p>.weight_global_scale` |
| activation scale (ignored) | `<p>.input_scale` | `<p>.input_global_scale` |

Layer prefix `model.language_model.layers.*` (Gemma 4 multimodal) or `model.layers.*`. embeddings,
lm_head (untied, ~2 GB BF16), norms, `layer_scalar`, q/k_norm, pos_embedding, biases stay BF16/F32.

## Components (all in `cuda/`, unit-tested via `make nvfp4-test`)

| File | Role | Status |
|---|---|---|
| `safetensors.h` | container parser: u64+JSON header, sharded `index.json`, mmap | ✅ tested |
| `nvfp4.h` | dequant oracle: E2M1 LUT, software E4M3, reconstruct, row dequant | ✅ tested |
| `nvfp4_loader.h` | name mapping + config.json parse (quant kind, tie, ignore-glob) | ✅ tested (both conventions) |
| `nvfp4_inspect.cc` | opens a REAL checkpoint, detects, validates shapes, dequants a row | ✅ passes on `RedHatAI/gemma-4-12B-it-NVFP4` |
| `nvfp4_gemv.cuh` | fused decode GEMV (warp-per-row) | ✅ correct (L2rel 1e-5%); ⚠️ 66 GB/s — needs tuning to dp4a parity |
| `gemma4_kernels.cu` | `FORMAT_NVFP4`, `nvfp4_load_from_safetensors()` residency, `nvfp4_decode_proj()` | ✅ compiles; ⏳ not yet wired into create |

`nvfp4_load_from_safetensors()` populates `d_fp4_w / d_fp4_wsc / d_fp4_gsw` exactly as
`build_fp4_weights` does (same cuBLASLt `fp4_desc` setup), so **`gemm_nvfp4` drives prefill
unchanged** — only the source of the weights differs (disk vs Q4_0 requant).

## Remaining work (the create-fork — needs the real build + 10 GB model to verify)

1. **Format detection** in `gemma4_engine_create` (`gemma4_kernels.cu:3199`): read the first 4 bytes;
   if not `"GGUF"` (0x46554747), treat as safetensors. Branch BEFORE the GGUF metadata parse.
2. **NVFP4 create path**: `st::Model::open(path)` → `nvfp4ld::detect()` → `nvfp4_load_from_safetensors()`.
   Architecture is hardcoded Gemma-4-12B (assert `layout.n_layers == 48`); read remaining config from
   `config.json` if needed.
3. **BF16 non-quant tensors** (new — GGUF path reads these from the blob; here they come from
   safetensors BF16): norms → `d_w_*_norm` (BF16→float convert, cf. `:3478` UPLOAD_NORM), embeddings →
   embed lookup (currently Q8_0 `d_token_embd`; add a BF16 path), LM head (untied BF16 GEMV).
4. **Decode residency + routing**: the decode GEMV needs LINEAR scales, but `d_fp4_wsc` is swizzled
   (for cuBLASLt). Keep a per-proj linear-scale copy (+~0.75 GB, still a net ~6 GB win) OR teach the
   kernel the swizzle inverse. Route `gemv_w`/`decode_layer` to `nvfp4_decode_proj()` when
   `format == FORMAT_NVFP4`. The decode CUDA-graph capture must include the new kernel.
5. **Go/CLI plumbing**: accept a safetensors path / dir; pass `FORMAT_NVFP4` hint (auto-detected
   anyway). `cmd/fucina` + `internal/engine/cuda/bridge.go`.
6. **Kernel perf**: tune `nvfp4_gemv.cuh` to ≥125 GB/s (ncu: split-K, vectorized x, software
   pipelining) before making the NVFP4 decode the default — else decode regresses ~2× (NVFP4 and Q4_0
   are both 4.5 bit, so there is no inherent decode speedup; the win is single-store memory).
