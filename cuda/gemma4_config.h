// gemma4_config.h — runtime Gemma-4 model configuration.
//
// fucina is ONE binary that identifies the model from the checkpoint itself: a single build runs
// any supported Gemma-4 size (12B, 31B, …) by reading the architecture from the model's own GGUF kv
// metadata or safetensors config.json at load time. There are NO per-model compiler flags and NO
// env-var switches. See docs/dense-31b-89tok-plan.md (M0) and the runtime-model-detection note.
//
// Migration (M0): the engine was born with these as compile-time #defines hardcoded to 12B. They
// are being promoted to fields of gemma4_model_config_t below, populated by the loader. Static
// arrays size to the GEMMA4_CAP_* capacity maxima; head-count-templated kernels are instantiated
// for each supported config and dispatched on the runtime value (head_dim is constant: 256 sliding
// / 512 global, so those template params stay fixed).

#ifndef GEMMA4_CONFIG_H
#define GEMMA4_CONFIG_H

#include <stdint.h>

// ── Capacity maxima (compile-time) — bound static arrays / shared-memory tiles ──────────────────
// Sized for the largest supported model with headroom. Actual counts come from gemma4_model_config_t.
#define GEMMA4_CAP_LAYERS   64    // ≥ 60 (31B); 12B uses 48
#define GEMMA4_CAP_HEADS    32    // ≥ 32 (31B); 12B uses 16
#define GEMMA4_CAP_KV_HEADS 16    // ≥ 16 (31B sliding); 12B uses 8
// Sparse-MoE (qwen3moe) caps. These MUST stay <= the hardcoded stack-array bound in the
// MoE top-k router kernel (dg_softmax_topk's `bool used[DG_N_EXPERTS]`, DG_N_EXPERTS=128,
// DG_N_EXPERTS_USED=8): the kernel indexes that array by the RUNTIME expert_count, so a
// checkpoint with more experts than this would walk off thread 0's stack. Detection caps
// expert_count/expert_used_count against these so an oversized MoE fails cleanly at load
// instead of silently corrupting the stack. The whole Qwen3-MoE family is 128/top-8.
#define GEMMA4_CAP_EXPERTS       128
#define GEMMA4_CAP_EXPERTS_USED  8

// Geometry shared by every Gemma-4 size (constant — safe to keep as template params).
#define GEMMA4_HEAD_DIM         256   // sliding-layer head dim (key_length_swa)
#define GEMMA4_GLOBAL_HEAD_DIM  512   // global-layer head dim (key_length)

// ── Model architecture family ───────────────────────────────────────────────────────────────────
// fucina is one binary that runs several arch families off the SAME engine. The differences (embed
// scale, sandwich norms, GLU activation, softcap, head_dim, V-sharing, attention scale) are gated on
// cfg.arch so Gemma-4 stays byte-identical. Detected from general.architecture in the GGUF.
enum {
    GEMMA4_ARCH_GEMMA4   = 0,   // Gemma-4 (sliding+global, V=K on global, geglu, softcap, baked attn scale)
    GEMMA4_ARCH_QWEN3    = 1,   // Qwen3 dense (full-causal all layers, separate V, silu-glu, no softcap)
    GEMMA4_ARCH_QWEN3MOE = 2,   // Qwen3 MoE (qwen3moe): IDENTICAL attention/norm/rope/KV to Qwen3 dense,
                                // but the dense FFN becomes a 128-expert top-8 SiLU-GLU mixture.
    GEMMA4_ARCH_QWEN3_5  = 3,   // Qwen3.5 hybrid (qwen35): per-layer mix of FULL softmax-GQA (output-gated,
                                // partial-RoPE, q/k norm) and LINEAR gated-deltanet (SSM) layers, period-4
                                // full at (i+1)%full_attention_interval==0; SwiGLU MLP, untied lm_head.
};

// Per-layer attention kind for the Qwen3.5 hybrid (cfg.attn_kind[]). FULL = softmax GQA
// (reuses the engine's "global" full-attention class); LINEAR = gated-deltanet recurrence.
enum {
    GEMMA4_ATTN_FULL   = 0,
    GEMMA4_ATTN_LINEAR = 1,
};

// ── Per-model runtime configuration ─────────────────────────────────────────────────────────────
// Populated at load time from GGUF kv (gemma4.block_count, embedding_length, feed_forward_length,
// attention.head_count, attention.head_count_kv[], attention.sliding_window_pattern[],
// final_logit_softcapping, rope.freq_base[_swa]) or safetensors config.json (already parsed in
// nvfp4_loader.h). One struct per loaded model; the engine reads it instead of the old #defines.
typedef struct gemma4_model_config_t {
    int   arch;              // GEMMA4_ARCH_* (default GEMMA4_ARCH_GEMMA4)
    int   head_dim;          // uniform head dim for single-head-dim archs (Qwen3=128); Gemma uses
                             // the per-class GEMMA4_HEAD_DIM/GEMMA4_GLOBAL_HEAD_DIM constants instead.
    int   n_layers;          // 48 (12B) / 60 (31B)
    int   hidden_size;       // 3840 / 5376
    int   intermediate;      // 15360 / 21504  (FFN)
    int   n_heads;           // 16 / 32        (query heads)
    int   n_kv_sliding;      // 8 / 16         (sliding-layer KV heads)
    int   n_kv_global;       // 1 / 4          (global-layer KV heads; 12B=broadcast, 31B=GQA)
    int   vocab_size;        // 262144 (both)
    float softcap;           // 30.0
    float rope_theta_global; // 1e6
    float rope_theta_sliding;// 1e4
    // Per-layer attention type, read from sliding_window_pattern[] (true=sliding, false=global).
    // is_global[i]=1 → global layer. Index into a CAP_LAYERS-sized array.
    uint8_t is_global[GEMMA4_CAP_LAYERS];
    int   n_global;          // count of global layers (8 / 10)

    // ── Sparse-MoE (GEMMA4_ARCH_QWEN3MOE) ───────────────────────────────────────────────────────
    // Zero for dense archs. n_experts = total expert count (128), n_experts_used = top-k routed
    // per token (8), expert_ffn = per-expert FFN intermediate (768). The router is a plain
    // hidden→n_experts GEMV (softmax over all experts, top-k, renormalize the k weights to sum 1).
    int   n_experts;         // 0 (dense) / 128 (qwen3moe)
    int   n_experts_used;    // 0 (dense) / 8   (qwen3moe top-k)
    int   expert_ffn;        // 0 (dense) / 768 (qwen3moe per-expert FFN intermediate)

    // ── Qwen3.5 hybrid (GEMMA4_ARCH_QWEN3_5) ────────────────────────────────────────────────────
    // Zero for non-hybrid archs. Per-layer attention kind (GEMMA4_ATTN_FULL / GEMMA4_ATTN_LINEAR):
    // full-attn iff (i+1)%full_attention_interval==0, else gated-deltanet (linear). The FULL layers
    // are also marked is_global[i]=1 so they route through the engine's existing global GQA class;
    // the LINEAR layers are dispatched off attn_kind[] by the (M-stage) gated-deltanet forward.
    uint8_t attn_kind[GEMMA4_CAP_LAYERS]; // GEMMA4_ATTN_FULL / GEMMA4_ATTN_LINEAR per layer
    int   full_attention_interval; // 4   (full layer iff (i+1)%interval==0); 0 for non-hybrid
    int   n_full;                  // 8   (count of FULL softmax-attention layers)
    int   rotary_dim;              // 64  (partial-RoPE width applied to the first rotary_dim of
                                   //      head_dim 256; the remaining dims pass through). 0 = full RoPE.
    // Gated-DeltaNet (SSM) geometry for the LINEAR layers, read from qwen35.ssm.*:
    int   ssm_state_size;          // 128 (per-head key/value state dim)
    int   ssm_conv_kernel;         // 4   (depthwise causal conv1d kernel over concat[q;k;v])
    int   ssm_inner_size;          // 4096 (value-path inner width = n_v_heads * state_size)
    int   ssm_group_count;         // 16  (key/query heads; value heads = repeat_interleave 16→32)
    int   ssm_time_step_rank;      // 32  (num value heads; A_log/dt_bias/b/a width)
} gemma4_model_config_t;

#endif // GEMMA4_CONFIG_H
