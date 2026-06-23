// e4b_engine.h — C ABI for the Gemma-4-E4B autoregressive engine.
//
// Standalone from the dense gemma4 engine: E4B has runtime dims (not the
// compile-time 12B/31B constants), Per-Layer Embeddings, and KV-cache sharing,
// so it gets its own translation unit. Correctness-first bring-up: BF16 weights
// resident on device, cuBLAS matmuls, FP8 Per-Layer-Embedding "index" (the big
// memory win). NVFP4 weight path is a later optimization.
//
// See gemma4_e4b.h for the architecture and gemma4-e4b-arch (memory) for specs.
#ifndef FUCINA_E4B_ENGINE_H
#define FUCINA_E4B_ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct e4b_engine e4b_engine_t;

// Returns 1 if the safetensors checkpoint at `path` (dir / .safetensors /
// .index.json with a sibling config.json) is a Gemma-4-E4B text model, 0 if
// not, -1 on error (can't open / no config.json).
int e4b_is_e4b_checkpoint(const char *path);

// Create an engine: detect + parse config, upload all language_model weights to
// device as BF16, and quantize the Per-Layer-Embedding table to FP8 E4M3 (the
// "index"). context_size caps the KV cache. device_id selects the GPU.
// Returns NULL on failure (message on stderr).
e4b_engine_t *e4b_engine_create(const char *model_path, uint32_t context_size, int device_id);

void e4b_engine_destroy(e4b_engine_t *eng);

// Diagnostics / accessors (available after create).
void e4b_engine_print_info(const e4b_engine_t *eng);
int  e4b_engine_n_layers(const e4b_engine_t *eng);
int  e4b_engine_hidden_size(const e4b_engine_t *eng);
int  e4b_engine_vocab_size(const e4b_engine_t *eng);
// Total device bytes resident (weights + FP8 PLE index + KV cache).
uint64_t e4b_engine_device_bytes(const e4b_engine_t *eng);

// ── Inference ─────────────────────────────────────────────────────────────

// Prefill the whole prompt in one pass into a FRESH KV cache (resets n_past) and
// write the last token's softcapped logits to logits_out[vocab] (or NULL).
int e4b_engine_prefill(e4b_engine_t *eng, const int32_t *tokens, int n_tokens, float *logits_out);

// Decode one token at the current cache position, advancing n_past by 1; writes
// the next-token softcapped logits to logits_out[vocab] (or NULL).
int e4b_engine_decode(e4b_engine_t *eng, int32_t token, float *logits_out);

// KV-cache sequence control.
void e4b_engine_reset(e4b_engine_t *eng);          // rewind to empty (n_past=0)
int  e4b_engine_n_past(const e4b_engine_t *eng);   // tokens currently in the cache

// Greedy generate: prefill prompt then argmax-decode up to max_new tokens, stopping
// at any id in stop_ids. Returns number of tokens written to out_tokens, -1 on error.
int e4b_engine_generate_greedy(e4b_engine_t *eng, const int32_t *prompt, int n_prompt,
                               int32_t *out_tokens, int max_new,
                               const int32_t *stop_ids, int n_stop);

// ── Continuous batching: multiple sequences decoded in one weight pass ──────
// seq_add: claim a free slot, prefill `prompt` into it, return the slot id (≥1) and
//   the greedy first token in *first_token_out. -1 if no slot / error.
// step_batch: ONE batched forward — feed in_tokens[i] to slots[i] (i<B), advance
//   each, write the greedy next token to out_tokens[i]. Weights read once for B tokens.
// seq_remove: release a slot (caches kept for reuse). seq_capacity: free slot count.
int  e4b_engine_seq_add(e4b_engine_t *eng, const int32_t *prompt, int n_prompt, int32_t *first_token_out);
int  e4b_engine_step_batch(e4b_engine_t *eng, const int *slots, const int32_t *in_tokens,
                           int B, int32_t *out_tokens);
void e4b_engine_seq_remove(e4b_engine_t *eng, int slot);
int  e4b_engine_seq_capacity(e4b_engine_t *eng);

// Debug forward: run prefill over `tokens` and (for any non-NULL pointer) copy
// the fp32 activations that the HF reference dumps, for numerical validation:
//   emb_out  [n*hidden]  scaled token embedding (pre-layers)
//   l0_out   [n*hidden]  hidden state after decoder layer 0
//   fin_out  [n*hidden]  hidden state after the final RMSNorm
//   logits_last_out [vocab] softcapped logits of the last token
// Returns 0 on success, -1 on error.
int e4b_engine_forward_debug(e4b_engine_t *eng, const int32_t *tokens, int n_tokens,
                             float *emb_out, float *l0_out, float *fin_out, float *logits_last_out);

#ifdef __cplusplus
}
#endif

#endif // FUCINA_E4B_ENGINE_H
