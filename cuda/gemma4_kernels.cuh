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
#define GEMMA4_SPEC_MAX          32      // max draft length per batched-decode pass / max concurrent decode rows
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
    FORMAT_Q5_K  = 6,  // GGML Q5_K super-blocks — appears on a few Qwen3 UD bulk weights (mixed
                       // attention). No native Q5_K kernel: each Q5_K bulk tensor is requantized to
                       // Q8_0 at load (wt_override) and thereafter read as FORMAT_Q8_0.
    FORMAT_FP8_BLOCK = 7, // DeepSeek/Qwen3.5 block-FP8: F8_E4M3 weights [out][in] + per-128×128
                       // BF16 block scale (weight_scale_inv). gemv_batched_w dispatches to the
                       // fp8_block_gemm primitive; the scale ptr comes from the engine's ptr→scale table.
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

// ─── Qwen3.5 hybrid (qwen35) M2 per-layer kernel parity self-test ──────────────
// Reads the torch reference binary written by cuda/qwen35_layer_ref.py (dequantized
// fp32 weights + a fixed input hidden + the reference mixer outputs), runs the qwen35
// FULL gated-softmax-attention and GDN (gated-deltanet) mixer kernels in fp32, and
// asserts max-abs relative error < 1e-2 vs torch for BOTH layer kinds AND that the GDN
// chunked-scan output matches the single-step recurrence. Returns 0 on pass, non-zero
// on failure. CUDA-free of the engine: it allocates its own device buffers.
int qwen35_m2_layer_selftest(const char *ref_bin_path);

// ─── Qwen3.5 hybrid (qwen35) M3 single-seq hybrid greedy forward ───────────────
// Token-by-token forward over a loaded qwen35 engine that carries the GDN recurrent state +
// conv ring (LINEAR layers) and a per-FULL-layer KV cache across tokens, dispatching per layer
// off cfg.attn_kind[]. Fills out_ids[0..n_gen-1] with the greedy (argmax) continuation of the
// prompt in_ids[0..n_prompt-1]. Returns 0 on success, non-zero on error. The engine must have
// been created from a qwen35 GGUF (gemma4_engine_create auto-detects the arch).
int qwen35_forward_greedy(gemma4_engine_t *eng, const int32_t *in_ids, int n_prompt,
                          int32_t *out_ids, int n_gen);

// ─── Qwen3.5 hybrid (qwen35) M4 batched continuous-batching decode gate ────────
// Drives the qwen35 continuous-batching ABI (seq_add prefill + step_batch decode) and asserts:
// (1) B-row batched decode is BIT-IDENTICAL per row to that row run alone B=1 (row independence),
// (2) graph-ON == graph-OFF, (3) the batched path reproduces the M3 single-seq France->Paris 8/8.
// Returns 0 on PASS. The engine must have been created from a qwen35 GGUF.
int qwen35_batch_selftest(gemma4_engine_t *eng);

// ─── Qwen3.5 hybrid (qwen35) M5: FP8 block-quant safetensors path ──────────────
// Loads the OFFICIAL Qwen3.5-9B FP8 checkpoint (DeepSeek-V3 block-fp8: F8_E4M3 weights +
// weight_scale_inv BF16 [out/128,in/128] block scales; norms/embed/lm_head/conv1d/A_log/dt_bias/
// in_proj_a/b stay BF16/F32) and runs the same hybrid forward as qwen35_forward_greedy with the
// projections driven by the fp8_block decode GEMV. Text path only (model.language_model.*).
//   qwen35_fp8_load   — parse + upload; returns an opaque model handle (NULL on failure).
//   qwen35_fp8_forward_greedy — greedy argmax continuation (mirrors qwen35_forward_greedy).
//   qwen35_fp8_free   — release all device buffers.
void *qwen35_fp8_load(const char *path);
int   qwen35_fp8_forward_greedy(void *model, const int32_t *in_ids, int n_prompt,
                                int32_t *out_ids, int n_gen);
void  qwen35_fp8_free(void *model);

// ─── Qwen3.5 hybrid (qwen35) M6: single-MTP draft head + LOSSLESS speculative decode ──────
// Loads the 22 mtp.* tensors (FP8 checkpoint only; the GGUF drops the head) inside qwen35_fp8_load.
// qwen35_fp8_spec_greedy drives the MTP draft head + a sequential stop-at-first-reject verify on
// the stateful hybrid backbone: it emits the SAME out_ids[0..n_gen-1] as qwen35_fp8_forward_greedy
// (lossless), while *drafted_out/*accepted_out report the cumulative draft accept counts (>0).
// Returns 0 on success, -2 if the checkpoint has no MTP head. Depth via FUCINA_QWEN35_MTP_K (def 4).
int   qwen35_fp8_spec_greedy(void *model, const int32_t *in_ids, int n_prompt,
                             int32_t *out_ids, int n_gen, long *drafted_out, long *accepted_out);

// ─── Qwen3.5-35B-A3B MoE hybrid (qwen3_5_moe) P6: FP8 block-quant safetensors path ─────────────
// Loads the OFFICIAL Qwen3.5-35B-A3B-FP8 checkpoint (same DeepSeek-V3 block-fp8 schema as the 9B,
// text path model.language_model.*) and runs the hybrid forward with the dense SwiGLU MLP replaced
// by the Qwen3_5MoeSparseMoeBlock: a 256-expert top-8 softmax-renorm mixture + a sigmoid-gated
// shared expert (both moe_intermediate 512). Hidden 2048, 2 KV heads; GDN geometry is identical to
// the 9B path (kernels reused). Self-contained device buffers + handle separate from the 9B q35fp8
// path so the dense 9B forward stays byte-identical. Greedy argmax continuation, mirrors
// qwen35_fp8_forward_greedy. The 9B dense checkpoint (no mlp.experts.*) loads through qwen35_fp8_load
// instead; this entry rejects a non-MoE checkpoint.
void *qwen35_moe_fp8_load(const char *path);
int   qwen35_moe_fp8_forward_greedy(void *model, const int32_t *in_ids, int n_prompt,
                                    int32_t *out_ids, int n_gen);
void  qwen35_moe_fp8_free(void *model);

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
int  gemma4_engine_is_qwen3_family(const gemma4_engine_t *eng);  // 1 for Qwen3 / Qwen3-MoE
int  gemma4_engine_n_experts(const gemma4_engine_t *eng);        // >0 for sparse/MoE (spec gate)

// Calibration-only sparse-MoE routing profiler. start allocates and zeros an
// [n_layers,n_experts] device histogram; normal inference pays zero overhead until
// start is called. snapshot synchronizes the engine stream and copies assignment
// counts plus the sum of selected router probabilities into caller-owned arrays.
// The output capacity must be at least n_layers*n_experts elements.
int gemma4_engine_moe_profile_start(gemma4_engine_t *eng);
int gemma4_engine_moe_profile_shape(const gemma4_engine_t *eng,
                                    int *n_layers, int *n_experts, int *top_k);
int gemma4_engine_moe_profile_snapshot(gemma4_engine_t *eng,
                                       uint64_t *counts, double *weight_sums,
                                       size_t capacity);
// Five activation classes per layer: mixer projection input, mixer output-projection
// input, MoE/shared gate-up input, routed-expert down input, shared-expert down input.
// Each output array needs n_layers*5 elements.
int gemma4_engine_moe_profile_activation_snapshot(gemma4_engine_t *eng,
                                       double *sum_squares, uint64_t *elements,
                                       float *max_abs, size_t capacity);

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

// Qwen3.5 hybrid per-SLOT state snapshot (batched-engine conversation cache):
// fixed GDN state S + conv ring per LINEAR layer, plus the FULL-layer K/V
// prefix at n_tokens. The GDN recurrence pins the snapshot to EXACTLY
// n_tokens, so restore is valid only for prompts that EXTEND the snapshot's
// token sequence. save requires the slot live at exactly n_tokens; restore
// overwrites a freshly opened slot's state and sets its token count.
// seq_ntokens reports a live slot's committed token count (-1 if free).
size_t gemma4_engine_q35_state_size(gemma4_engine_t *eng, int n_tokens);
int    gemma4_engine_q35_state_save(gemma4_engine_t *eng, int slot, void *buf, int n_tokens);
int    gemma4_engine_q35_state_restore(gemma4_engine_t *eng, int slot, const void *buf, int n_tokens);
int    gemma4_engine_seq_ntokens(gemma4_engine_t *eng, int slot);

// P0 (S1a) lossless GDN snapshot / commit / rewind for DFlash (1+K) verification. snapshot copies
// the slot's GDN/conv recurrent state; commit(accepted,j,out_next) restores the snapshot and
// replays exactly j accepted tokens so the slot ends byte-identical to j sequential decodes
// (out_next[i] optionally receives each replay step's argmax); rewind restores with j=0. The
// FULL-layer K/V cache is absolute-position-indexed and needs no snapshot. Returns 0 on success.
int    gemma4_engine_q35_gdn_snapshot(gemma4_engine_t *eng, int slot);
int    gemma4_engine_q35_gdn_commit(gemma4_engine_t *eng, int slot,
                                    const int32_t *accepted, int j, int32_t *out_next);
int    gemma4_engine_q35_gdn_rewind(gemma4_engine_t *eng, int slot);

// S1a P3/P4: lazily load the resident DFlash draft model from FUCINA_QWEN35_DFLASH_PATH (config +
// safetensors). Returns 0 on success/already-loaded, non-zero on failure (validated + geometry-
// checked against the target). ready reports whether the draft substrate is resident.
int    gemma4_engine_q35_dflash_load(gemma4_engine_t *eng);
int    gemma4_engine_q35_dflash_ready(gemma4_engine_t *eng);

// Named device-memory accounting. Qwen fields are zero for other architectures.
typedef struct gemma4_memory_stats {
    uint64_t qwen_workspace_bytes;
    uint64_t qwen_recurrent_per_slot_bytes;
    uint64_t qwen_kv_per_slot_bytes;
    uint64_t qwen_committed_bytes;
    uint64_t qwen_reserved_bytes;
    uint64_t qwen_peak_bytes;
    int32_t  qwen_capacity;
    int32_t  qwen_allocated_slots;
    int32_t  qwen_max_context;
    int32_t  qwen_reserved_context;
} gemma4_memory_stats_t;
void gemma4_engine_memory_stats(const gemma4_engine_t *eng, gemma4_memory_stats_t *out);

// Debug-only Qwen3.5 Jacobian-lens support. `load` accepts the FJSPACE1 format produced by
// scripts/convert_jlens.py. snapshot writes [returned_layers × max_topk] token ids/probabilities.
// Steering injects the normalized J_l^T·unembed[token] direction at selected fitted layers;
// n_layers<=0 selects all fitted layers. Strength is clamped to [-1,1] residual norms.
int  gemma4_engine_jspace_load(gemma4_engine_t *eng, const char *path, int topk);
int  gemma4_engine_jspace_snapshot(gemma4_engine_t *eng, int max_layers, int max_topk,
                                   int *layers, int *token_ids, float *probs);
int  gemma4_engine_jspace_steer(gemma4_engine_t *eng, int token_id, float strength,
                                const int *layers, int n_layers);
void gemma4_engine_jspace_clear_steer(gemma4_engine_t *eng);

// PINNED host allocation for state-snapshot buffers: q35_state_copy issues ~2·L
// small async copies, and pageable memory forces a driver bounce-buffer sync per
// copy (~250 ms per 35 MB snapshot measured on GB10/CUDA-13); pinned buffers make
// them true async DMA (~ms). Returns NULL on failure; free ONLY with the matching
// gemma4_host_free.
void  *gemma4_host_alloc(size_t bytes);
void   gemma4_host_free(void *p);

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
// P1: batched multi-sequence admission-prefill (Qwen3.5). Returns M / 0 (unsupported) / -1.
int  gemma4_engine_seq_add_multiseq(gemma4_engine_t *eng, const int32_t *tokens_flat,
                           const int *lens, int M, const float *temps, const int *topks,
                           const float *topps, const float *minps, const uint64_t *seeds,
                           int *out_slots, int32_t *out_first);
// DEBUG (test-only): copy just-computed first-token logits (nrows==1: d_logits; nrows>1: d_sb[11]).
int  gemma4_engine_debug_logits(gemma4_engine_t *eng, float *out, int nrows);
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
// Stage 18 — FUSED prefill+decode (Qwen3 family only; returns -2 for other archs so the caller
// falls back to its own path). Mixes B_dec decode rows + pf_len prefill-chunk rows of slot pf_slot
// into ONE batched forward: the decode rows are byte-identical to step_batch (out_dec[B_dec] is the
// per-row sampled token, -1 = KV-exhausted with out_dec_lens[r] = -1; surviving rows report
// out_dec_lens[r] = 1), the prefill rows are byte-identical to seq_prefill_chunk, and neither
// perturbs the other (distinct block tables, NULL rng_off). pf_slot advances by pf_len; when
// pf_is_final != 0 the prompt's FIRST generated token is written to *pf_first_out. B_dec + pf_len
// must be <= GEMMA4_MAX_SEQS. Returns 0 / -1 (hard error or KV exhaustion) / -2 (arch unsupported).
int  gemma4_engine_step_batch_fused(gemma4_engine_t *eng,
                                    const int *dec_slots, const int32_t *dec_toks, int B_dec,
                                    int pf_slot, const int32_t *pf_chunk, int pf_len, int pf_is_final,
                                    int32_t *out_dec, int *out_dec_lens, int32_t *pf_first_out);
void gemma4_engine_seq_remove(gemma4_engine_t *eng, int slot);
int  gemma4_engine_seq_capacity(gemma4_engine_t *eng);
// Cross-request prefix cache (RadixAttention). set: enable/disable (effective only
// on the full-attention single-pool geometry, n_layers_sliding==0; no-op for Gemma).
// stats: observability counters (all zero when disabled).
void gemma4_engine_set_prefix_cache(gemma4_engine_t *eng, int enable);
void gemma4_engine_prefix_cache_stats(const gemma4_engine_t *eng, uint64_t *lookups,
                                      uint64_t *hit_blocks, uint64_t *cached_blocks,
                                      uint64_t *evictions);
// Register full blocks (prompt + generated) from a slot's committed token history so
// later requests can reuse generated text. Idempotent; call on 256-token boundaries.
void gemma4_engine_prefix_commit(gemma4_engine_t *eng, int slot,
                                 const int32_t *history, int n);
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
// As seq_open but adopts the longest cached prefix of `prompt` into the slot's KV and
// reports the number of already-satisfied prompt tokens in *shared_out, so the chunked
// path prefills only the divergent suffix (keeps the prefix-cache win on the interleave path).
int  gemma4_engine_seq_open_prefix(gemma4_engine_t *eng, const int32_t *prompt, int n_prompt,
                                   int *shared_out,
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
