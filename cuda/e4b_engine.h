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
// "index"). context_size caps the KV cache; max_seqs is the desired number of
// concurrent sequences for continuous batching (clamped to [1,8]; <=0 ⇒ 8).
// device_id selects the GPU.
//
// MEMORY: after all weights/quant copies are resident, the engine queries free
// device memory (and the optional FUCINA_MEM_BUDGET_GB *total-device* cap) and
// AUTO-SHRINKS context_size and/or max_seqs so the KV cache provably fits —
// instead of cudaMalloc'ing until the kernel OOM-kills the process. Read the
// values actually chosen back with e4b_engine_max_ctx / e4b_engine_max_seqs.
// Returns NULL on failure (message on stderr).
e4b_engine_t *e4b_engine_create(const char *model_path, uint32_t context_size,
                                int max_seqs, int device_id);

void e4b_engine_destroy(e4b_engine_t *eng);

// Diagnostics / accessors (available after create).
void e4b_engine_print_info(const e4b_engine_t *eng);
int  e4b_engine_n_layers(const e4b_engine_t *eng);
int  e4b_engine_hidden_size(const e4b_engine_t *eng);
int  e4b_engine_vocab_size(const e4b_engine_t *eng);
// Total device bytes resident (weights + FP8 PLE index + KV cache).
uint64_t e4b_engine_device_bytes(const e4b_engine_t *eng);
// Effective KV-cache limits after the create-time memory fit (may be smaller
// than requested): max_ctx = per-sequence token capacity, max_seqs = concurrent
// sequence slots actually provisioned.
uint32_t e4b_engine_max_ctx(const e4b_engine_t *eng);
int      e4b_engine_max_seqs(const e4b_engine_t *eng);

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

// ── Prefix-cache reuse (server KVCache contract) ────────────────────────────
// Append `suffix` at the CURRENT n_past (set by a prior rewind) instead of resetting;
// writes the last token's logits. The server's KVCache reuses a shared prefix then
// prefills only the divergent suffix through this.
int  e4b_engine_prefill_append(e4b_engine_t *eng, const int32_t *suffix, int n, float *logits_out);
// Rewind slot 0 to n_keep tokens. Returns 1 if safe, 0 if the sliding ring already
// overwrote the window for n_keep (rewind depth > sliding_cap - window) — caller then
// falls back to reset + full re-prefill.
int  e4b_engine_rewind(e4b_engine_t *eng, int n_keep);

// ── MTP speculative decode (draft head) ─────────────────────────────────────
// Load the gemma4-assistant draft head GGUF (~78M, 4 Q-only layers) for ~2x decode via
// greedy speculation. Returns 0 on success, -1 on failure (engine still usable, plain
// decode). e4b_engine_has_assistant reports whether one is loaded.
int  e4b_engine_load_assistant(e4b_engine_t *eng, const char *path);
int  e4b_engine_has_assistant(const e4b_engine_t *eng);

// Debug: run ONE drafter-head forward on slot 0. h_io is the [hidden] recurrent state
// (in: target post-final-norm hidden of the preceding token; out: next recurrent h).
// `tok` is that token id, `pos` its absolute RoPE position (= n_past at the draft point).
// Writes the greedy-drafted next token id to *draft_id. Returns 0 on success, -1 on error.
// (Increment 2 standalone numeric check; the spec loop will call this internally.)
int  e4b_engine_mtp_forward_debug(e4b_engine_t *eng, int32_t tok, int pos,
                                  float *h_io, int32_t *draft_id);

// Greedy generate: prefill prompt then argmax-decode up to max_new tokens, stopping
// at any id in stop_ids. Returns number of tokens written to out_tokens, -1 on error.
int e4b_engine_generate_greedy(e4b_engine_t *eng, const int32_t *prompt, int n_prompt,
                               int32_t *out_tokens, int max_new,
                               const int32_t *stop_ids, int n_stop);

// Greedy SPECULATIVE generate via the MTP draft head (must be loaded first with
// e4b_engine_load_assistant). Drafts up to K tokens (default 4, FUCINA_E4B_DRAFT_K) per
// step, verifies them in one target forward, and accepts the longest greedy-matching
// prefix. Output is BIT-IDENTICAL to e4b_engine_generate_greedy (lossless). No assistant
// loaded ⇒ transparently falls back to plain greedy. Returns tokens written, -1 on error.
int e4b_engine_generate_spec_greedy(e4b_engine_t *eng, const int32_t *prompt, int n_prompt,
                                    int32_t *out_tokens, int max_new,
                                    const int32_t *stop_ids, int n_stop);

// Streaming greedy speculative decode (server path). CONTINUE variant: resumes from slot 0's
// CURRENT KV (caller must have prefilled `history`; n_hist == n_past) and `first_logits` (the
// last-token logits the caller captured). Invokes cb(tok, ud) for every committed token in
// order between verify rounds; cb returning non-zero stops after that token. out_tokens
// receives ALL committed tokens (callback-declined + accepted-tail of the final round) so the
// caller can reconcile its prefix cache with the engine KV (n_past advances by the count
// returned). Bit-identical to plain greedy. No assistant ⇒ plain greedy decode driving cb.
// Returns tokens written (>=0), -1 on error.
typedef int (*e4b_spec_token_cb)(int32_t tok, void *ud);
int e4b_engine_spec_stream(e4b_engine_t *eng, const int32_t *history, int n_hist,
                           const float *first_logits, int32_t *out_tokens, int max_new,
                           const int32_t *stop_ids, int n_stop,
                           e4b_spec_token_cb cb, void *ud);

// Cumulative speculative-decode acceptance counters for /metrics. τ = emitted/steps;
// acceptance = accepted/drafted. All zero until the spec path runs.
long e4b_engine_spec_steps(const e4b_engine_t *eng);
long e4b_engine_spec_drafted(const e4b_engine_t *eng);
long e4b_engine_spec_accepted(const e4b_engine_t *eng);
long e4b_engine_spec_emitted(const e4b_engine_t *eng);

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
