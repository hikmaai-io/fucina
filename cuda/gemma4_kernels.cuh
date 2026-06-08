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
// Formats:      FP8 E4M3 (primary), Q8_0 (fallback)
// Adapters:     LoRA scaled (GGUF format)

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

#define GEMMA4_MAX_LAYERS        48
#define GEMMA4_HIDDEN_SIZE       3840
#define GEMMA4_INTERMEDIATE      15360   // 4× hidden
#define GEMMA4_HEADS             16
#define GEMMA4_KV_HEADS          8
#define GEMMA4_GLOBAL_KV_HEADS   1
#define GEMMA4_HEAD_DIM          256
#define GEMMA4_GLOBAL_HEAD_DIM   512
#define GEMMA4_SLIDING_WINDOW    1024
#define GEMMA4_VOCAB_SIZE        262144
#define GEMMA4_MAX_CTX           262144
#define GEMMA4_SOFTCAP           30.0f
#define GEMMA4_RMS_EPS           1e-6f
#define GEMMA4_MAX_LORA_RANK     64
#define GEMMA4_SPEC_MAX          16      // max draft length per batched-decode pass

// Special tokens
#define GEMMA4_BOS_ID  2
#define GEMMA4_EOS_ID  1
#define GEMMA4_PAD_ID  0

// ─── Tensor format flags ───────────────────────────────────────────────

typedef enum {
    FORMAT_FP8   = 0,  // CUDA_R_8F_E4M3 (primary, native Blackwell)
    FORMAT_Q8_0  = 1,  // GGML Q8_0 blocks (fallback)
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

// ─── LoRA adapter descriptor ──────────────────────────────────────────

typedef struct {
    float  *d_a;             // LoRA A matrix [in_dim × rank] on device
    float  *d_b;             // LoRA B matrix [rank × out_dim] on device
    float   scale;           // scale = alpha / rank
    int     rank;            // LoRA rank
    int     input_dim;       // input dimension
    int     output_dim;      // output dimension
    int     active;          // 1 if this adapter is loaded
} lora_adapter_t;

// ─── Per-weight LoRA set ──────────────────────────────────────────────

typedef struct {
    lora_adapter_t q;        // attention Q projection
    lora_adapter_t k;        // attention K projection
    lora_adapter_t v;        // attention V projection
    lora_adapter_t o;        // attention output projection
    lora_adapter_t gate;     // FFN gate projection
    lora_adapter_t up;       // FFN up projection
    lora_adapter_t down;     // FFN down projection
} layer_lora_set_t;

// ─── Engine state (opaque, passed to Go via CGO) ──────────────────────

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gemma4_engine gemma4_engine_t;

// ─── Engine lifecycle ─────────────────────────────────────────────────

gemma4_engine_t* gemma4_engine_create(
    const char *model_path,     // path to GGUF file
    tensor_format_t format,     // FP8 or Q8_0
    uint32_t context_size,      // max context tokens
    int device_id               // CUDA device (0 for DGX Spark)
);

void gemma4_engine_destroy(gemma4_engine_t *eng);

// ─── LoRA adapter loading ─────────────────────────────────────────────

// Load a LoRA scaled adapter from a GGUF file
// The GGUF contains lora.{layer}.{weight}.weight_a/b tensors
// and lora.{layer}.{weight}.alpha metadata
// Returns 0 on success, -1 on error
int gemma4_engine_load_lora(
    gemma4_engine_t *eng,
    const char      *lora_path,     // path to LoRA GGUF file
    float            scale          // additional scale multiplier (use 1.0f normally)
);

// Remove all loaded LoRA adapters (restore base model)
void gemma4_engine_unload_lora(gemma4_engine_t *eng);

// ─── Core inference ──────────────────────────────────────────────────

// Prefill: process n_tokens in sequence, filling KV cache
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
    int             *n_accepted_out);

// Speculative verify batch: verify K draft tokens in parallel
// Returns number of accepted tokens (0..K)
int gemma4_engine_verify_batch(
    gemma4_engine_t *eng,
    const int32_t   *draft_tokens,  // [K] draft tokens
    int              K,
    const int32_t   *positions,     // [K] positions in context
    float           *logits_out     // [K × VOCAB_SIZE] or NULL
);

// ─── Sampling ─────────────────────────────────────────────────────────

int gemma4_sample_argmax(const float *logits, int vocab_size);

// ─── Session save/restore ────────────────────────────────────────────

int gemma4_engine_save_session(
    gemma4_engine_t *eng,
    uint8_t         *buffer,     // output buffer, or NULL for size query
    uint64_t        *size        // in/out: buffer size / required size
);

int gemma4_engine_load_session(
    gemma4_engine_t *eng,
    const uint8_t   *buffer,
    uint64_t         size
);

// Batched prefill: process the whole prompt in one BF16 tensor-core pass instead
// of token-by-token GEMV. Requires a fresh sequence (n_tokens in KV cache == 0);
// returns -2 if not applicable (caller should fall back to gemma4_engine_prefill),
// 0 on success, -1 on error. Same logits_out contract as gemma4_engine_prefill.
int gemma4_engine_prefill_batched(
    gemma4_engine_t *eng, const int32_t *tokens, int n_tokens, float *logits_out);

// ─── Diagnostics ─────────────────────────────────────────────────────

void gemma4_engine_print_info(const gemma4_engine_t *eng);
void gemma4_engine_print_timing(const gemma4_engine_t *eng);
int  gemma4_engine_get_n_layers(const gemma4_engine_t *eng);
int  gemma4_engine_get_context_size(const gemma4_engine_t *eng);
int  gemma4_engine_has_lora(const gemma4_engine_t *eng);

// Timing accessors for speed logging
float gemma4_engine_prefill_ms(const gemma4_engine_t *eng);
int   gemma4_engine_prefill_tokens(const gemma4_engine_t *eng);
float gemma4_engine_decode_ms(const gemma4_engine_t *eng);
int   gemma4_engine_decode_tokens(const gemma4_engine_t *eng);

// KV cache state management (for prefix reuse across requests)
int  gemma4_engine_n_tokens(const gemma4_engine_t *eng);  // tokens in KV cache
void gemma4_engine_reset(gemma4_engine_t *eng);           // rewind to empty
int  gemma4_engine_rewind(gemma4_engine_t *eng, int n_keep); // keep first n_keep

#ifdef __cplusplus
}
#endif

#endif // GEMMA4_KERNELS_CUH
