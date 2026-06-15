// diffusion_gemma_engine.h — C ABI for the DiffusionGemma block-diffusion engine.
//
// Standalone from the autoregressive gemma4 engine (separate libdg.a). Loads a
// diffusion-gemma GGUF, runs the block-diffusion denoising loop, returns committed token ids.
// Correctness-first (dequant-per-matmul + cuBLAS); see PLAN_DIFFUSION_GEMMA.md for the perf TODOs.

#ifndef DIFFUSION_GEMMA_ENGINE_H
#define DIFFUSION_GEMMA_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dg_engine dg_engine_t;

// Returns 1 if the GGUF's general.architecture == "diffusion-gemma", 0 otherwise, -1 on error.
int dg_gguf_is_diffusion(const char *gguf_path);

// Read diffusion.canvas_length from the GGUF (returns it, or -1 on error).
int dg_gguf_canvas_length(const char *gguf_path);

// Create an engine: parse + upload weights, size scratch for prompts up to max_prompt tokens.
// fp4_moe != 0 enables the NVFP4 MoE experts (CUTLASS grouped FP4, canvas-only). Returns NULL on failure.
dg_engine_t *dg_engine_create(const char *gguf_path, int max_prompt, int fp4_moe);
void dg_engine_free(dg_engine_t *eng);

int dg_engine_canvas_length(const dg_engine_t *eng);
int dg_engine_max_prompt(const dg_engine_t *eng); // actual prompt window after the GPU-memory cap

// Run one block of diffusion over a fresh 256-token canvas conditioned on `prompt`.
// Writes committed argmax token ids (trimmed at the first EOS) to out_ids (≤ max_out).
// Returns the number of output tokens, or -1 on error.
//   max_steps      : denoising step cap (e.g. 48)
//   t_min/t_max    : linear temperature schedule bounds (e.g. 0.4 / 0.8)
//   entropy_bound  : entropy-bound sampler acceptance budget (e.g. 0.1)
//   seed           : RNG seed (canvas init, sampling, renoise)
//   eot_id         : end-of-turn token that stops block chaining (besides EOS); <=0 to disable
// Chains 256-token diffusion blocks (appending each committed block to the K/V cache) until a
// block emits EOS/eot_id, out_ids fills (max_out), or the cache is full. Up to max_out tokens.
int dg_engine_generate(dg_engine_t *eng, const int32_t *prompt, int n_prompt,
                       int max_steps, float t_min, float t_max, float entropy_bound,
                       uint64_t seed, int eot_id, int32_t *out_ids, int max_out);

// Warm up at load time: builds the NVFP4 MoE weights + selects cuBLAS algos + loads kernels via a
// throwaway prefill + canvas pass, so the first real request doesn't stall mid-answer.
void dg_engine_warmup(dg_engine_t *eng);

// Timing of the most recent dg_engine_generate: prompt-prefill ms, denoise-loop ms, and the
// number of denoising steps actually run (any out pointer may be NULL).
void dg_engine_last_stats(const dg_engine_t *eng, float *prefill_ms, float *denoise_ms, int *steps);

#ifdef __cplusplus
}
#endif

#endif // DIFFUSION_GEMMA_ENGINE_H
