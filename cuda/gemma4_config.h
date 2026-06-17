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

// Geometry shared by every Gemma-4 size (constant — safe to keep as template params).
#define GEMMA4_HEAD_DIM         256   // sliding-layer head dim (key_length_swa)
#define GEMMA4_GLOBAL_HEAD_DIM  512   // global-layer head dim (key_length)

// ── Per-model runtime configuration ─────────────────────────────────────────────────────────────
// Populated at load time from GGUF kv (gemma4.block_count, embedding_length, feed_forward_length,
// attention.head_count, attention.head_count_kv[], attention.sliding_window_pattern[],
// final_logit_softcapping, rope.freq_base[_swa]) or safetensors config.json (already parsed in
// nvfp4_loader.h). One struct per loaded model; the engine reads it instead of the old #defines.
typedef struct gemma4_model_config_t {
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
} gemma4_model_config_t;

#endif // GEMMA4_CONFIG_H
