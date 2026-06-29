// gemma4_kernels.cuh - Gemma 4 12B CUDA inference kernels for DGX Spark GB10
// Blackwell sm_121 (GB10 specific), FP8 Tensor Core native via CUDA 13.0
//
// Architecture:
//   48 layers: 40 sliding window + 8 global attention
//   sliding: 8 KV heads, head_dim=256, RoPE theta=10000
//   global:  1 KV head,  head_dim=512, p-RoPE theta=1M, K=V unified
//   FFN: GeGLU (gelu_pytorch_tanh), intermediate=15360
//   Output: logit softcapping at 30.0
//
// Formats: Q4_0 (QAT, Q6_K tied head) and Q8_0 — auto-detected from the GGUF.

#ifndef GEMMA4_KERNELS_CUH
#define GEMMA4_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>

// ─── Compile-time constants ────────────────────────────────────────────

// Runtime model config (gemma4_model_config_t), capacity maxima (GEMMA4_CAP_*), and the constant
// head dims (GEMMA4_HEAD_DIM / GEMMA4_GLOBAL_HEAD_DIM). M0 migrates the hardcoded sizes below into
// fields of that struct, read from the GGUF/safetensors metadata. See docs/dense-31b-89tok-plan.md.
#include "gemma4_config.h"

// CURRENT HARDCODE (Gemma-4-12B) — to be replaced by gemma4_model_config_t reads in M0.
// These still drive every kernel today; the 31B runtime path is not wired yet.
#define GEMMA4_MAX_LAYERS        48
#define GEMMA4_HIDDEN_SIZE       3840
#define GEMMA4_INTERMEDIATE      15360   // 4× hidden
#define GEMMA4_HEADS             16
#define GEMMA4_KV_HEADS          8
#define GEMMA4_GLOBAL_KV_HEADS   1

#define GEMMA4_SLIDING_WINDOW    1024
#define GEMMA4_VOCAB_SIZE        262144
#define GEMMA4_MAX_CTX           262144
#define GEMMA4_SOFTCAP           30.0f
#define GEMMA4_RMS_EPS           1e-6f
#define GEMMA4_SPEC_MAX          16      // max draft length per batched-decode pass
// GQA-broadcast global flash-decode split-K (DECODE-30-35 Step 1): the global context is
// split into ≤MAX_SPLITS blocks of ~SPLIT_CHUNK timesteps so the single global KV head still
// saturates bandwidth across SMs while being read only ONCE per token (not n_heads× = 16×).
#define GEMMA4_GLOBAL_MAX_SPLITS 128
// 64-key chunks: at chat-typical context (~250 tok) the old 256 gave ONE split = one
// block = one SM for the whole global layer (nsys: 238 µs avg, 9.4% of generation).
// 64 → 4+ blocks at short ctx; long ctx still clamps at MAX_SPLITS.
#define GEMMA4_GLOBAL_SPLIT_CHUNK 64
// Sliding-decode split-K chunk: the window caps the attended range at ≤1024 keys, so 128-key
// chunks give ≤8 splits — enough blocks to hide latency once the per-key __syncthreads is gone
// (warp-per-KV-head kernel), while keeping the combine pass trivial. Shares the global path's
// d_fa_acc/m/l scratch (splits clamped ≤ GEMMA4_GLOBAL_MAX_SPLITS; see wrapper for sizing proof).
#define GEMMA4_SLIDING_SPLIT_CHUNK 64

// Special tokens
#define GEMMA4_BOS_ID  2
#define GEMMA4_EOS_ID  1
#define GEMMA4_PAD_ID  0

// ─── Tensor format flags ───────────────────────────────────────────────

typedef enum {
    FORMAT_Q8_0  = 1,  // GGML Q8_0 blocks
    FORMAT_Q4_0  = 2,  // GGML Q4_0 blocks (QAT 4-bit; layers Q4_0, token_embd→Q8_0)
    FORMAT_Q6_K  = 3,  // GGML Q6_K super-blocks (used only as a wfmt override for the native
                       // QAT tied LM head — DECODE-30-35 Step 8; not a whole-model format)
    FORMAT_NVFP4 = 4,  // NVFP4 from safetensors (ModelOpt/compressed-tensors): E2M1 + per-16
                       // E4M3 block scales + per-tensor FP32 global. Single weight store feeds
                       // both the cuBLASLt block-scaled prefill and the fused decode GEMV.
    FORMAT_Q4_K  = 5,  // GGML Q4_K super-blocks — wfmt override for the native Unsloth-UD tied LM
                       // head (reads raw 4.5-bit head vs the Q8_0 upconvert); not a whole-model format.
} tensor_format_t;

// ─── GGML Q8_0 block ──────────────────────────────────────────────────
//
// IMPORTANT: matches the on-disk GGML layout exactly.
//   d  = fp16 (half) scale  → 2 bytes
//   qs = 32 × int8          → 32 bytes
// Total = 34 bytes per block (NOT 36). The scale is stored as IEEE-754
// half precision, not float32.

typedef struct {
    uint16_t d;         // fp16 scale (half precision, raw bits)
    int8_t   qs[32];    // quantized values
} block_q8_0;           // sizeof == 34

// ─── Layer types ───────────────────────────────────────────────────────

typedef enum {
    LAYER_SLIDING = 0,
    LAYER_GLOBAL  = 1,
} layer_type_t;

// ─── Engine state (opaque, passed to Go via CGO) ──────────────────────

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gemma4_engine gemma4_engine_t;

// ─── Engine lifecycle ─────────────────────────────────────────────────

gemma4_engine_t* gemma4_engine_create(
    const char *model_path,     // path to GGUF file
    tensor_format_t format,     // hint only — the real format is auto-detected from the GGUF
    uint32_t context_size,      // max context tokens
    int device_id,              // CUDA device (0 for DGX Spark)
    double gpu_mem_util         // --gpu-mem-util: fraction of total device mem the engine may use
                                // (vLLM-style; e.g. 0.90). Budgets weights+KV+scratch+packed; the
                                // engine caps ctx / drops the packed-Q4_0 copy to satisfy it. <=0 → 0.90.
);

void gemma4_engine_destroy(gemma4_engine_t *eng);

// ─── Core inference ──────────────────────────────────────────────────

// Prefill: process n_tokens in sequence, filling KV cache.
// Returns 0 / -1 / -3 (aborted between chunks via gemma4_engine_abort_prefill).
int gemma4_engine_prefill(
    gemma4_engine_t *eng,
    const int32_t   *tokens,        // [n_tokens] input token IDs
    int              n_tokens,
    float           *logits_out     // [VOCAB_SIZE] logits of last token, or NULL
);

// Decode: process a single token, return logits
int gemma4_engine_decode(
    gemma4_engine_t *eng,
    int32_t          token,
    float           *logits_out     // [VOCAB_SIZE] output logits
);

// Batched decode: forward K tokens in ONE weight pass, continuing the sequence.
// Writes logits_out[K × VOCAB_SIZE] (row i = logits after token i), advances the
// cache by K. K ≤ GEMMA4_SPEC_MAX. Returns 0 ok, -2 defer (LoRA), -1 error.
int gemma4_engine_decode_batched(
    gemma4_engine_t *eng,
    const int32_t   *tokens,         // [K] tokens to forward
    int              K,
    float           *logits_out      // [K × VOCAB_SIZE]
);

// Greedy generation with prompt-lookup speculative decoding. Forwards [g,draft...]
// per step in one weight pass; (1+accepted) tokens per ~one token's bandwidth.
// Fills out_tokens (≤ max_new); returns count generated; *n_accepted_out (or NULL)
// gets the total drafts accepted. draft_k ≤ GEMMA4_SPEC_MAX-1.
int gemma4_engine_generate_spec(
    gemma4_engine_t *eng,
    const int32_t   *prompt, int n_prompt,
    int32_t         *out_tokens, int max_new,
    const int32_t   *stop_ids, int n_stop,
    int              draft_k,
    float temp, int top_k, float top_p, float min_p, float repeat_penalty,
    uint64_t seed,
    int             *n_accepted_out);

// Server path: speculative generation continuing from an ALREADY-prefilled engine.
// `history` = prompt tokens in the cache; `first_logits` = post-prefill logits.
int gemma4_engine_generate_spec_continue(
    gemma4_engine_t *eng,
    const int32_t   *history, int n_history,
    const float     *first_logits,
    int32_t         *out_tokens, int max_new,
    const int32_t   *stop_ids, int n_stop,
    int              draft_k,
    float temp, int top_k, float top_p, float min_p, float repeat_penalty,
    uint64_t seed,
    int             *n_accepted_out);

// Per-token streaming callback for the speculative generators: invoked once per
// emitted token, in emission order, from the calling thread between verify steps.
// Return nonzero to stop generation after this token (the token IS still counted
// in out_tokens). Tokens already committed to the KV by the verify pass that
// produced them stay committed — callers reconcile via gemma4_engine_n_tokens.
typedef int (*gemma4_token_cb)(int32_t token, void *user_data);

// gemma4_engine_generate_spec_continue with a per-token callback: the SSE/REPL
// streaming form of the speculative fast path (same exact-distribution guarantee;
// cb == NULL degrades to generate_spec_continue).
int gemma4_engine_generate_spec_stream(
    gemma4_engine_t *eng,
    const int32_t   *history, int n_history,
    const float     *first_logits,
    int32_t         *out_tokens, int max_new,
    const int32_t   *stop_ids, int n_stop,
    int              draft_k,
    float temp, int top_k, float top_p, float min_p, float repeat_penalty,
    uint64_t seed,
    int             *n_accepted_out,
    gemma4_token_cb  cb, void *cb_user_data);

// ─── Sampling ─────────────────────────────────────────────────────────

int gemma4_sample_argmax(const float *logits, int vocab_size);

// GPU-side sampling over the engine's resident logits (eng->d_logits): temp<=0 →
// argmax, else temperature → top-k → softmax → top-p → min-p → multinomial(rnd).
// Only the 4-byte token id crosses to host (no 262k logits D2H). rnd ∈ [0,1).
int gemma4_engine_sample_device(
    gemma4_engine_t *eng, float temp, int top_k, float top_p, float min_p, float rnd);

// Load the official Gemma-4 MTP assistant GGUF (~423M draft head, Q8_0). When loaded,
// the speculative loop drafts novel text with it (recursive multi-token prediction over
// the shared target KV cache) instead of falling back to single-token decode — the
// llama.cpp `--spec-type draft-mtp` equivalent (>2x dense decode at ~0.59 acceptance).
// Call once after create. Returns 0 on success; failure leaves drafting disabled.
int gemma4_engine_load_assistant(gemma4_engine_t *eng, const char *path);

// Batched prefill: process the whole prompt in one BF16 tensor-core pass instead
// of token-by-token GEMV. Requires a fresh sequence (n_tokens in KV cache == 0);
// returns -2 if not applicable (caller should fall back to gemma4_engine_prefill),
// 0 on success, -1 on error. Same logits_out contract as gemma4_engine_prefill.
int gemma4_engine_prefill_batched(
    gemma4_engine_t *eng, const int32_t *tokens, int n_tokens, float *logits_out);

// Chunked FLASH prefill: bounded per-chunk memory (O(chunk + KV), no [HEADS][N×N]
// score buffer), so it handles arbitrary context (256k+) where the batched GEMM path
// OOMs. BF16 tensor-core projections, online-softmax flash attention. UNLIKE
// prefill_batched it also accepts a NON-EMPTY cache (global_n_tokens > 0): the new
// tokens are processed as chunks attending the frozen history at absolute
// positions — the fast path for multi-turn agent SUFFIX prefills. Tiny suffixes
// (<32 tokens) defer (-2) to the chunked dp4a path, which is cheaper there.
// Returns 0 / -2 (defer) / -1 / -3 (aborted via gemma4_engine_abort_prefill).
int gemma4_engine_prefill_flash(
    gemma4_engine_t *eng, const int32_t *tokens, int n_tokens, float *logits_out);

// Eagerly allocate the lazy first-prefill state (persistent prefill scratch +
// BF16 dequant scratch); call once at server startup so request #1 doesn't pay
// ~0.5-2.1 s of one-time cudaMallocs inside its prefill timer. Idempotent.
int gemma4_engine_warmup(gemma4_engine_t *eng);

// Cooperative prefill abort, callable from another thread while a prefill is in
// flight: the chunked prefill loops poll it between chunks and return -3. The
// flag clears at the next prefill's entry.
void gemma4_engine_abort_prefill(gemma4_engine_t *eng);

// ─── Diagnostics ─────────────────────────────────────────────────────

void gemma4_engine_print_info(const gemma4_engine_t *eng);
void gemma4_engine_print_timing(const gemma4_engine_t *eng);
int  gemma4_engine_get_n_layers(const gemma4_engine_t *eng);
int  gemma4_engine_get_context_size(const gemma4_engine_t *eng);

// Timing accessors for speed logging
float gemma4_engine_prefill_ms(const gemma4_engine_t *eng);
int   gemma4_engine_prefill_tokens(const gemma4_engine_t *eng);
float gemma4_engine_decode_ms(const gemma4_engine_t *eng);
int   gemma4_engine_decode_tokens(const gemma4_engine_t *eng);

// Cumulative speculative-decode acceptance counters (across all spec calls), for /metrics:
// τ = emitted/steps tokens per verify forward; acceptance = accepted/drafted.
long  gemma4_engine_spec_steps(const gemma4_engine_t *eng);
long  gemma4_engine_spec_drafted(const gemma4_engine_t *eng);
long  gemma4_engine_spec_accepted(const gemma4_engine_t *eng);
long  gemma4_engine_spec_emitted(const gemma4_engine_t *eng);

// KV cache state management (for prefix reuse across requests)
int  gemma4_engine_n_tokens(const gemma4_engine_t *eng);  // tokens in KV cache
void gemma4_engine_reset(gemma4_engine_t *eng);           // rewind to empty
int  gemma4_engine_rewind(gemma4_engine_t *eng, int n_keep); // keep first n_keep

// KV sequence snapshot/restore (multi-conversation prefix cache). The flat
// per-position KV layout makes a sequence's state exactly the first n_tokens
// positions of each K/V buffer, so save/restore are four strided 2D copies.
// state_size returns the host-buffer size needed for n_tokens (0 on bad args);
// save/restore return 0 on success. Restore overwrites the live sequence and
// sets the engine token count to n_tokens.
size_t gemma4_engine_kv_state_size(const gemma4_engine_t *eng, int n_tokens);
int    gemma4_engine_kv_save(gemma4_engine_t *eng, void *buf, int n_tokens);
int    gemma4_engine_kv_restore(gemma4_engine_t *eng, const void *buf, int n_tokens);

// ─── Continuous batching (multi-sequence paged decode) ─────────────
// All require paged mode (engine created with FUCINA_PAGED_KV). They drive B
// independent sequences through ONE batched forward over the shared paged KV.
//
// seq_add: allocate a free slot, prefill `prompt` into that slot's paged KV, and
//   sample the first token into *first_token_out using the per-sequence params
//   (temp<=0 ⇒ greedy argmax; temp>0 ⇒ top_k/top_p/min_p with a reproducible RNG
//   stream seeded by `seed`). The params are stored on the slot and reused for
//   every subsequent step_batch token of this sequence. Returns slot id (>=0)
//   or -1 (no free slot / not paged / error).
// step_batch: ONE batched forward over the B given slots — feed in_tokens[i] to
//   slots[i] at its current position, advance each, sample one token/slot into
//   out_tokens[i] using THAT slot's stored params. Returns 0 / -1.
// seq_remove: free a slot's block tables back to the pools, mark it free.
// seq_capacity: number of free slots.
int  gemma4_engine_seq_add(gemma4_engine_t *eng, const int32_t *prompt, int n_prompt,
                           int32_t *first_token_out,
                           float temp, int top_k, float top_p, float min_p, uint64_t seed);
int  gemma4_engine_step_batch(gemma4_engine_t *eng, const int *slots,
                              const int32_t *in_tokens, int B, int32_t *out_tokens);
// MTP speculative batched step: per-slot draft + one batched verify. out_tokens is
// [B*GEMMA4_SPEC_MAX] (each row's emitted run), out_lens[B] the per-row run length
// (>=1) or -1 if the slot hit its KV limit. Output is byte-identical to step_batch.
int  gemma4_engine_step_batch_spec(gemma4_engine_t *eng, const int *slots,
                                   const int32_t *in_tokens, int B,
                                   int32_t *out_tokens, int *out_lens);
// As above but verifies EXTERNAL per-slot drafts (model-agnostic prompt-lookup, driven
// from Go): drafts is flat [B*GEMMA4_SPEC_MAX], dlens[B] the per-row draft count. No MTP
// head required. Output is byte-identical to step_batch / lossless w.r.t. greedy decode.
int  gemma4_engine_step_batch_spec_ext(gemma4_engine_t *eng, const int *slots,
                                       const int32_t *in_tokens, int B,
                                       int32_t *out_tokens, int *out_lens,
                                       const int32_t *drafts, const int *dlens);
void gemma4_engine_seq_remove(gemma4_engine_t *eng, int slot);
int  gemma4_engine_seq_capacity(gemma4_engine_t *eng);
// Cross-request prefix cache (RadixAttention). set: enable/disable (effective only
// on the full-attention single-pool geometry, n_layers_sliding==0; no-op for Gemma).
// stats: observability counters (all zero when disabled).
void gemma4_engine_set_prefix_cache(gemma4_engine_t *eng, int enable);
void gemma4_engine_prefix_cache_stats(const gemma4_engine_t *eng, uint64_t *lookups,
                                      uint64_t *hit_blocks, uint64_t *cached_blocks,
                                      uint64_t *evictions);
// Chunked prefill (interleave a long prompt's prefill with decode of other slots).
// seq_open: reserve a free slot with EMPTY KV and store the sampling params, WITHOUT
//   prefilling. Returns slot id (>=0) or -1 (no free slot / not paged / error).
// seq_prefill_chunk: append `n` tokens to an open slot's paged KV at its current
//   position (resumable suffix prefill), token-by-token — the SAME per-position forward
//   as the seq_add fallback, so after the final chunk the KV is position-for-position
//   identical to a one-shot seq_add of the whole prompt. When do_sample != 0 (final
//   chunk) it samples the first generated token into *first_token_out with the slot's
//   stored params. Returns 0 on success, -1 on error (caller frees the slot).
int  gemma4_engine_seq_open(gemma4_engine_t *eng,
                            float temp, int top_k, float top_p, float min_p, uint64_t seed);
int  gemma4_engine_seq_prefill_chunk(gemma4_engine_t *eng, int slot,
                                     const int32_t *tokens, int n,
                                     int do_sample, int32_t *first_token_out);

// ─── CUDA Graph support (experimental, off by default) ─────────────
// Call gemma4_engine_set_graph_mode(eng, 1) to enable. This allocates
// persistent prefill scratch and prepares graph capture infrastructure.
void gemma4_engine_set_graph_mode(gemma4_engine_t *eng, int mode);
void gemma4_engine_graph_stats(const gemma4_engine_t *eng,
    int *hits, int *misses, int *captures, int *launches);

#ifdef __cplusplus
}
#endif

#endif // GEMMA4_KERNELS_CUH
