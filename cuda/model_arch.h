// model_arch.h — model-agnostic architecture descriptor for fucina's dense
// (gemma4-family) serving engine. ONE runtime descriptor parameterizes the
// generic transformer-stack builder so Gemma-4(/E4B), Qwen3-dense and Qwen3-MoE
// all run through the same f32 serving kernels (paged KV, continuous batching,
// CUDA-graph decode, in-graph sampler), with per-arch attention GEOMETRY still
// resolved at COMPILE time via template instantiation (never runtime-dim — that
// would regress the templated paged-attn / 1024-wide sampler kernels).
//
// Invariant: the gemma4 descriptor (populate_arch_gemma4) reproduces EXACTLY the
// current GEMMA4_* macro values, so de-hardcoding the stack to read these fields
// is bit-identical (verified by FP8-KV memcmp==0 + logit memcmp==0, not the 90%
// self-test). Every toggle must resolve to a launched-or-skipped kernel so the
// per-arch decode graph stays capturable.
//
// Pure C header (no CUDA / no <cuda_runtime.h> deps) so it is also includable
// from host-only unit tests and, later, the cgo bridge.
#ifndef FUCINA_MODEL_ARCH_H
#define FUCINA_MODEL_ARCH_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ARCH_GEMMA4   = 0,   // Gemma-4 / E4B (the original hardcoded path)
    ARCH_QWEN3    = 1,   // Qwen3 dense (and the llama/mistral/qwen2 family subset)
    ARCH_QWEN3MOE = 2,   // Qwen3-MoE (sparse top-k experts)
} arch_family_t;

// Normalization placement per layer.
typedef enum {
    NORM_PRE_ONLY          = 0,  // Qwen3/llama: input_layernorm + post_attention_layernorm
    NORM_PRE_POST_SANDWICH = 1,  // Gemma: extra post-attn + post-ffn norms (4 norms/layer)
} norm_style_t;

// FFN block kind.
typedef enum {
    FFN_DENSE_GEGLU = 0,  // Gemma: gelu-tanh gate
    FFN_DENSE_SWIGLU = 1, // Qwen3/llama: silu gate
    FFN_MOE_SWIGLU  = 2,  // Qwen3-MoE: router -> top-k experts (silu), weighted sum
} ffn_kind_t;

// Gated-FFN activation.
typedef enum {
    ACT_GELU_TANH = 0,    // Gemma
    ACT_SILU      = 1,    // Qwen3
} act_kind_t;

// Per-arch attention GEOMETRY family. The actual paged-attn kernels are
// template-instantiated on these (compile time); this enum keys the dispatch.
// Gemma uses TWO geometries alternating per layer (sliding vs global); Qwen3
// uses ONE (full GQA). The per-layer SLIDING/GLOBAL choice stays in the engine's
// layer_types[] array — this descriptor carries the geometry parameters.
typedef enum {
    GEOM_GEMMA4 = 0,      // sliding<16,8,256> + global<16,1,512>
    GEOM_QWEN3  = 1,      // full GQA <32,8,128>
} attn_geom_t;

typedef struct {
    arch_family_t family;
    attn_geom_t   geom;

    // Core dims.
    int   n_layers;
    int   hidden;
    int   intermediate;     // dense FFN width (per-expert width lives in moe_intermediate)
    int   n_heads;
    int   n_kv_heads;       // default/sliding KV heads (GQA)
    int   head_dim;         // default/sliding head dim
    int   vocab;
    int   max_ctx;
    float rms_eps;

    // Embeddings.
    int   embed_scale_sqrt_hidden;  // Gemma multiplies embeds by sqrt(hidden); Qwen3 does NOT
    int   tied_embeddings;          // tied LM head (Gemma=1, Qwen3-8B=0)

    // Norm.
    norm_style_t norm_style;        // sandwich (Gemma) vs pre-only (Qwen3)
    int   has_qk_norm;              // per-head QK-RMSNorm pre-RoPE (Gemma & Qwen3 = 1)
    int   has_v_norm;               // per-head V-RMSNorm (Gemma = 1, Qwen3 = 0)
    int   has_layer_out_scale;      // per-layer hidden output scale h_out_scale (Gemma = 1)

    // Attention geometry (Gemma's second/global geometry; ignored when geom is single-family).
    int   sliding_window;           // sliding window length (Gemma=1024; 0 = full attention everywhere)
    int   global_kv_heads;          // Gemma global layer KV heads (=1); unused for Qwen3
    int   global_head_dim;          // Gemma global layer head dim (=512); unused for Qwen3
    int   global_v_eq_k;            // Gemma global layers set V := K (no separate V proj path)

    // RoPE.
    float rope_theta;               // primary theta (Gemma sliding=1e4; Qwen3=1e6)
    float rope_theta_global;        // Gemma global theta (=1e6); unused for Qwen3
    int   rope_partial_global;      // Gemma global uses partial-rope freq factors; Qwen3=0 (full NEOX)

    // Attention QK softmax scale applied to the query before the (scale-1.0) attn dot.
    // Gemma-4 uses 1.0 (its query scaling is handled in its norm scheme); standard
    // archs (Qwen3/llama) need 1/sqrt(head_dim). Launch-or-skip: 1.0 → no scale kernel.
    float attn_scale;

    // Output.
    float logit_softcap;            // Gemma=30.0; Qwen3=0 (disabled)

    // FFN.
    ffn_kind_t ffn_kind;
    act_kind_t act;

    // MoE (FFN_MOE_SWIGLU only).
    int   n_experts;                // 0 for dense
    int   n_experts_used;           // top-k active experts
    int   moe_intermediate;         // per-expert FFN width
    int   moe_norm_topk_prob;       // renormalize the top-k router weights to sum 1

    // Special tokens (informational; tokenizer owns the source of truth).
    int   bos_id, eos_id, pad_id;
} model_arch_t;

// Populate the descriptor with the EXACT current Gemma-4 constants. Keep these in
// lockstep with the GEMMA4_* macros in gemma4_kernels.cuh — changing a macro must
// change here too. This is the bit-identity anchor for the de-hardcoding refactor.
static inline void populate_arch_gemma4(model_arch_t *a) {
    a->family   = ARCH_GEMMA4;
    a->geom     = GEOM_GEMMA4;
    a->n_layers     = 48;       // GEMMA4_MAX_LAYERS
    a->hidden       = 3840;     // GEMMA4_HIDDEN_SIZE
    a->intermediate = 15360;    // GEMMA4_INTERMEDIATE
    a->n_heads      = 16;       // GEMMA4_HEADS
    a->n_kv_heads   = 8;        // GEMMA4_KV_HEADS
    a->head_dim     = 256;      // GEMMA4_HEAD_DIM
    a->vocab        = 262144;   // GEMMA4_VOCAB_SIZE
    a->max_ctx      = 262144;   // GEMMA4_MAX_CTX
    a->rms_eps      = 1e-6f;    // GEMMA4_RMS_EPS
    a->embed_scale_sqrt_hidden = 1;
    a->tied_embeddings         = 1;
    a->norm_style       = NORM_PRE_POST_SANDWICH;
    a->has_qk_norm      = 1;
    a->has_v_norm       = 1;
    a->has_layer_out_scale = 1;
    a->sliding_window   = 1024;  // GEMMA4_SLIDING_WINDOW
    a->global_kv_heads  = 1;     // GEMMA4_GLOBAL_KV_HEADS
    a->global_head_dim  = 512;   // GEMMA4_GLOBAL_HEAD_DIM
    a->global_v_eq_k    = 1;
    a->rope_theta        = 10000.0f;
    a->rope_theta_global = 1000000.0f;
    a->rope_partial_global = 1;
    a->attn_scale        = 1.0f;  // Gemma-4: attention dot uses scale 1.0
    a->logit_softcap     = 30.0f; // GEMMA4_SOFTCAP
    a->ffn_kind = FFN_DENSE_GEGLU;
    a->act      = ACT_GELU_TANH;
    a->n_experts = 0;
    a->n_experts_used = 0;
    a->moe_intermediate = 0;
    a->moe_norm_topk_prob = 0;
    a->bos_id = 2;  // GEMMA4_BOS_ID
    a->eos_id = 1;  // GEMMA4_EOS_ID
    a->pad_id = 0;  // GEMMA4_PAD_ID
}

// Populate the descriptor for a Qwen3 DENSE checkpoint. Variable dims come from the
// GGUF metadata (qwen3.* keys); the toggles are the Qwen3-family constants: full
// attention everywhere (no sliding window), pre-norm only (no Gemma sandwich), per-head
// QK-RMSNorm but NO V-norm, NEOX RoPE single theta full rotation, SwiGLU/SiLU FFN, and
// NO embedding scale / NO logit softcap / NO per-layer out scale / NO KV-share. A single
// attention geometry (GEOM_QWEN3) → one KV pool; global_* mirror the sole geometry.
static inline void populate_arch_qwen3(model_arch_t *a,
    int n_layers, int hidden, int intermediate, int n_heads, int n_kv_heads,
    int head_dim, int vocab, int max_ctx, float rms_eps, float rope_theta,
    int tied_embeddings, int bos_id, int eos_id, int pad_id)
{
    a->family   = ARCH_QWEN3;
    a->geom     = GEOM_QWEN3;
    a->n_layers     = n_layers;
    a->hidden       = hidden;
    a->intermediate = intermediate;
    a->n_heads      = n_heads;
    a->n_kv_heads   = n_kv_heads;
    a->head_dim     = head_dim;
    a->vocab        = vocab;
    a->max_ctx      = max_ctx;
    a->rms_eps      = rms_eps;
    a->embed_scale_sqrt_hidden = 0;        // Qwen3 does NOT scale embeddings
    a->tied_embeddings         = tied_embeddings;
    a->norm_style       = NORM_PRE_ONLY;   // input_layernorm + post_attention_layernorm
    a->has_qk_norm      = 1;               // per-head QK-RMSNorm shape[head_dim] pre-RoPE
    a->has_v_norm       = 0;               // Qwen3 has NO V-norm (Gemma does)
    a->has_layer_out_scale = 0;
    a->sliding_window   = 0;               // full attention on every layer
    a->global_kv_heads  = n_kv_heads;      // single geometry → mirror
    a->global_head_dim  = head_dim;
    a->global_v_eq_k    = 0;               // separate V projection on every layer
    a->rope_theta        = rope_theta;     // 1e6 for Qwen3-8B
    a->rope_theta_global = rope_theta;
    a->rope_partial_global = 0;            // full NEOX rotation, no partial-rope factors
    a->attn_scale        = 1.0f / sqrtf((float)head_dim);  // standard 1/sqrt(head_dim)
    a->logit_softcap     = 0.0f;           // disabled (launch-or-skip → skipped)
    a->ffn_kind = FFN_DENSE_SWIGLU;
    a->act      = ACT_SILU;
    a->n_experts = 0;
    a->n_experts_used = 0;
    a->moe_intermediate = 0;
    a->moe_norm_topk_prob = 0;
    a->bos_id = bos_id;
    a->eos_id = eos_id;
    a->pad_id = pad_id;
}

#ifdef __cplusplus
}
#endif

#endif // FUCINA_MODEL_ARCH_H
