// diffusion_gemma_kernels.cuh — DiffusionGemma 26B-A4B (4B active) CUDA engine for DGX Spark GB10
//
// DiffusionGemma is a discrete text-diffusion MoE built on the Gemma-4 backbone. It runs the
// SAME weights in two modes (HF: DiffusionGemmaEncoder/DecoderModel):
//   • encoder — causal attention, WRITES the KV cache (prompt prefill + per-block commit)
//   • decoder — bidirectional attention, READS the KV cache only (denoises a 256-tok canvas)
//
// Architecture (verified from the GGUF + HF transformers reference, 2026-06-13):
//   30 layers, 5:1 sliding:global (global at idx 5,11,17,23,29; last forced global)
//   hidden 2816, 16 query heads
//   sliding: 8 KV heads, head_dim 256, RoPE theta 10000 (full rotary)
//   global : 2 KV heads, head_dim 512, p-RoPE theta 1e6 (proportional, partial 0.25) + rope_freqs; V=K (no v_proj)
//   dense FFN: GeGLU (gelu_pytorch_tanh), intermediate 2112
//   MoE: 128 experts, 8 active, expert intermediate 704 — runs IN PARALLEL with the dense FFN
//   attention scale 1.0 (no 1/sqrt(d), no attn-logit softcap); final logit softcap 30.0
//   self-conditioning gated MLP on prev-step softmax→embedding
//   vocab 262144 (gemma4 tokenizer), canvas_length 256
//
// Quant formats present: Q4_K (most weights), Q6_K (token_embd, sliding attn_v), Q8_0
//   (ffn_down, ffn_down_exps), Q5_0 (self_cond_down), F32 (norms/scales/rope_freqs).
//
// This header is standalone (no dependency on gemma4_kernels.cuh) so the diffusion engine can
// be built and tested independently of the autoregressive Gemma-4 engine.

#ifndef DIFFUSION_GEMMA_KERNELS_CUH
#define DIFFUSION_GEMMA_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ─── Model constants ───────────────────────────────────────────────────
#define DG_MAX_LAYERS        30
#define DG_HIDDEN            2816
#define DG_FFN_INTERMEDIATE  2112
#define DG_HEADS             16
#define DG_KV_HEADS_SLIDING  8
#define DG_KV_HEADS_GLOBAL   2
#define DG_HEAD_DIM          256
#define DG_GLOBAL_HEAD_DIM   512
#define DG_SLIDING_WINDOW    1024
#define DG_N_EXPERTS         128
#define DG_N_EXPERTS_USED    8
#define DG_EXPERT_FFN        704
#define DG_VOCAB             262144
#define DG_CANVAS_LENGTH     256
#define DG_PREFILL_CHUNK     1024   // prefill runs chunk-by-chunk so activation scratch is CH-sized, not max_prompt-sized
#define DG_SOFTCAP           30.0f
#define DG_RMS_EPS           1e-6f
#define DG_ROPE_THETA_SLIDING 10000.0f
#define DG_ROPE_THETA_GLOBAL  1000000.0f

// Diffusion generation defaults (HF _get_default_generation_params)
#define DG_MAX_DENOISE_STEPS 48
#define DG_ENTROPY_BOUND     0.1f
#define DG_T_MIN             0.4f
#define DG_T_MAX             0.8f
#define DG_STABILITY_THRESH  1
#define DG_CONFIDENCE_THRESH 0.005f

// Special tokens
#define DG_BOS_ID  2
#define DG_EOS_ID  1
#define DG_PAD_ID  0
#define DG_MASK_ID 4

// ─── GGML quant type ids (subset present in this model) ─────────────────
typedef enum {
    DG_GGML_Q4_0 = 2,
    DG_GGML_Q5_0 = 6,
    DG_GGML_Q8_0 = 8,
    DG_GGML_Q4_K = 12,
    DG_GGML_Q6_K = 14,
    DG_GGML_F32  = 0,
    DG_GGML_F16  = 1,
} dg_ggml_type_t;

#define DG_QK   32     // legacy block size (Q4_0/Q5_0/Q8_0)
#define DG_QK_K 256    // super-block size (Q4_K/Q6_K)

// Fused-MMQ output-tile size (BM rows × BN cols). The grouped-expert tile descriptors built
// host-side in the engine MUST use these same values as the kernels.
#define DG_MMQ_BM 64
#define DG_MMQ_BN 64

// ─── On-disk block layouts (match ggml-common.h byte-for-byte) ──────────
// ggml_half == uint16_t IEEE-754 fp16 (raw bits).

typedef struct { uint16_t d; uint8_t qs[16]; } dg_block_q4_0;            // 18 B / 32 elem
typedef struct { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; } dg_block_q5_0; // 22 B / 32
typedef struct { uint16_t d; int8_t qs[32]; } dg_block_q8_0;            // 34 B / 32

// Q4_K: d, dmin (fp16), 12 packed 6-bit scale/min bytes, 128 nibble bytes → 144 B / 256 elem
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; } dg_block_q4_K;
// Q6_K: 128 low-nibble, 64 high-2bit, 16 int8 scales, fp16 d → 210 B / 256 elem
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; } dg_block_q6_K;

// Bytes per on-disk block for a given ggml type (0 = unsupported here).
static inline int dg_block_bytes(int ggml_type) {
    switch (ggml_type) {
        case DG_GGML_Q4_0: return 18;
        case DG_GGML_Q5_0: return 22;
        case DG_GGML_Q8_0: return 34;
        case DG_GGML_Q4_K: return 144;
        case DG_GGML_Q6_K: return 210;
        default: return 0;
    }
}
static inline int dg_block_nelem(int ggml_type) {
    return (ggml_type == DG_GGML_Q4_K || ggml_type == DG_GGML_Q6_K) ? DG_QK_K : DG_QK;
}

#ifdef __cplusplus
extern "C" {
#endif

// Dequantize a whole quantized tensor (raw on-disk bytes, device ptr) to fp32 (device ptr).
// n_elem must be a multiple of the format's block element count. Returns 0 on success.
int dg_dequant(int ggml_type, const void *raw_dev, int64_t n_elem, float *out_dev,
               cudaStream_t stream);

// ─── Forward-pass elementwise kernels (Phase 2) ─────────────────────────
// Activations are column-major [features, tokens]: element (f,t) at f + features*t.

// RMSNorm per token (column) over `feat` features. weight (or NULL) is a [feat] scale.
void dg_rmsnorm(float *out, const float *in, const float *weight, int feat, int tokens,
                float eps, cudaStream_t s);
// Per-head RMSNorm over head_dim, for [head_dim, n_head, tokens]. weight (or NULL) is [head_dim].
void dg_head_rmsnorm(float *x, const float *weight, int head_dim, int n_head, int tokens,
                     float eps, cudaStream_t s);
// NEOX rope on [head_dim, n_head, tokens] at absolute positions pos[tokens]; freq_factors or NULL.
void dg_rope(float *x, const int *pos, int head_dim, int n_head, int tokens,
             float theta_base, const float *freq_factors, cudaStream_t s);
void dg_gelu_mul(float *out, const float *gate, const float *up, int64_t n, cudaStream_t s); // gelu_tanh(gate)*up
void dg_silu_mul(float *out, const float *gate, const float *up, int64_t n, cudaStream_t s); // silu(gate)*up (Qwen3-MoE)
void dg_add(float *out, const float *a, const float *b, int64_t n, cudaStream_t s);          // out=a+b
void dg_scale(float *x, int64_t n, float s_scalar, cudaStream_t s);                          // x *= s
void dg_mul_vec_cols(float *x, const float *vec, int feat, int tokens, cudaStream_t s);      // x[:,t]*=vec (per feature)
void dg_mul_region_scalar(float *x, int feat, int t0, int t1, const float *scalar_dev, cudaStream_t s);
void dg_softmax_topk(const float *logits, int n_expert, int tokens, int topk,
                     int *out_idx, float *out_w, cudaStream_t s);   // per-token softmax→topk→normalize
void dg_softcap(float *x, int64_t n, float cap, cudaStream_t s);
void dg_embed_gather(float *dst, const float *tok_f32, const int *ids, int n_embd, int tokens,
                     float embed_scale, cudaStream_t s);
// region-aware bidirectional attention (scale=1.0), materialized scores. q/k/v: [head_dim,*,tokens].
void dg_attention(float *out, const float *q, const float *k, const float *v,
                  int head_dim, int n_head, int n_kv_head, int tokens, int n_prompt,
                  int n_swa, int is_sliding, cudaStream_t s);
// Canvas-only attention reading a cached prompt K/V (pk/pv, n_prompt cols) plus fresh canvas
// K/V (ck/cv, canvas cols). qc/out carry the canvas columns. Used by the prompt-KV-cached
// per-denoise-step forward so the prompt isn't recomputed every step.
void dg_attention_canvas(float *out, const float *qc, const __nv_bfloat16 *pk, const __nv_bfloat16 *pv,
                         const float *ck, const float *cv, int head_dim, int n_head, int n_kv_head,
                         int n_prompt, int canvas, int n_swa, int is_sliding, cudaStream_t s);
// Chunked causal prefill attention: chunk queries (CH) attend cached prefix (Pc, bf16) + fresh
// chunk K/V (CH, fp32), causally. Lets prefill run chunk-by-chunk with CH-sized activation buffers.
void dg_attention_chunk(float *out, const float *qc, const __nv_bfloat16 *pk, const __nv_bfloat16 *pv,
                        const float *ck, const float *cv, int head_dim, int n_head, int n_kv_head,
                        int n_prefix, int chunk, int n_swa, int is_sliding, cudaStream_t s);
void dg_gather_cols(float *dst, const float *src, const int *idx, int feat, int ncols, cudaStream_t s);
void dg_scatteradd_cols(float *dst, const float *src, const int *idx, const float *colscale,
                        int feat, int ncols, cudaStream_t s);
void dg_split_gelu_mul(float *out, const float *gateup, int half, int ncols, cudaStream_t s);
void dg_softmax_cols(const float *logits, float *probs, int vocab, int C, cudaStream_t s);
void dg_sample_step(const float *logits, const float *rnd, int *out_sample, int *out_argmax,
                    float *out_entropy, int vocab, int C, cudaStream_t s); // gelu(gu[0:half])*gu[half:2half]

// ─── Fused quantized matmul (dp4a) — replaces dequant-to-fp32 + cuBLAS sgemm ─────────────
// Activations are column-major [in_dim, tokens]; weights are raw on-disk GGUF quant blocks.
// Output is column-major [out_dim, tokens] (== token-major out[col*out_dim+row]), matching the
// engine's activation layout exactly (no transpose). in_dim must be a multiple of 256 for the
// k-quant formats / 32 for the legacy formats.

// fp32 → bf16 elementwise convert (for the bf16 tensor-core GEMM path).
void dg_f32_to_bf16(const float *in, __nv_bfloat16 *out, int64_t n, cudaStream_t s);

// Dequantize a quantized tensor straight to bf16 (no fp32 round-trip) for the bf16 GEMM path.
int dg_dequant_bf16(int ggml_type, const void *raw_dev, int64_t n_elem, __nv_bfloat16 *out_dev,
                    cudaStream_t stream);

// Symmetric per-32-block int8 quantization of the activation matrix [in_dim, tokens] → qx
// (int8, [in_dim*tokens]), per-block scale dx and per-block Σ sx (each [in_dim/32*tokens]).
void dg_quantize_q8_1(const float *x, int8_t *qx, float *dx, int *sx, int in_dim, int tokens,
                      cudaStream_t s);

// Tiled-MMQ GEMM, Y[out_dim×N] = W_q4_K · X_int8. qx/dx/sx come from dg_quantize_q8_1(x,...,N).
void dg_mmq_q4_K(float *out, const void *weight, const int8_t *qx, const float *dx, const int *sx,
                 int in_dim, int out_dim, int N, cudaStream_t s);

// Grouped expert matmul — all active experts in ONE launch. out/qx/dx/sx are the flattened
// gathered assignments [in_dim × total] (columns contiguous per expert). grid = (row-tiles,
// num_active); blockIdx.y selects an active expert via the tiny per-expert arrays aexp (expert id),
// acoloff (abs column base), ane (column count), each length num_active (≤128). wbase is the
// expert-weight base, slab_stride the bytes per expert slab. Q4_K = gate_up; Q8_0/Q5_0 = down.
// On-GPU MoE routing (counting-sort): fills count[e], coloff[e] (exclusive prefix), src[pos]
// (source token) and csc[pos] (router weight × per-expert scale), all grouped by expert. cursor is
// [n_expert] scratch. No host round-trip. (total assignments = n_tokens·n_used, known host-side.)
void dg_moe_route(const int *tki, const float *tkw, const float *pes, int n_tokens, int n_used,
                  int n_expert, int *count, int *coloff, int *cursor, int *src, float *csc,
                  cudaStream_t s);

// Grouped expert matmul — grid=(row-tiles, n_expert); blockIdx.y is the expert, reading its
// coloff[e]/count[e] (empty experts return early). qx/dx/sx/out are the flattened gathered
// assignments [in_dim × total] grouped per expert. Q4_K=gate_up; Q8_0/Q5_0=down.
void dg_mmq_q4_K_grouped(float *out, const void *wbase, int64_t slab_stride,
                         const int8_t *qx, const float *dx, const int *sx,
                         const int *coloff, const int *count, int n_expert,
                         int in_dim, int out_dim, cudaStream_t s);
void dg_mmq_q8_0_grouped(float *out, const void *wbase, int64_t slab_stride,
                         const int8_t *qx, const float *dx, const int *sx,
                         const int *coloff, const int *count, int n_expert,
                         int in_dim, int out_dim, cudaStream_t s);
void dg_mmq_q5_0_grouped(float *out, const void *wbase, int64_t slab_stride,
                         const int8_t *qx, const float *dx, const int *sx,
                         const int *coloff, const int *count, int n_expert,
                         int in_dim, int out_dim, cudaStream_t s);

#ifdef __cplusplus
}
#endif

#endif // DIFFUSION_GEMMA_KERNELS_CUH
