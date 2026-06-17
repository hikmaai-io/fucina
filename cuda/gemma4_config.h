// gemma4_config.h — selectable Gemma-4 architecture constant sets.
//
// The engine was born hardcoded for Gemma-4-12B. To also run Gemma-4-31B (and as the staging
// step toward a fully runtime model config — see docs/dense-31b-89tok-plan.md, milestone M0),
// the per-variant architecture constants live here behind a compile-time switch.
//
//   default            → 12B   (48 layers, hidden 3840, 16 heads, KV 8/1, FFN 15360)
//   -DGEMMA4_VARIANT_31B → 31B  (60 layers, hidden 5376, 32 heads, KV 16/4, FFN 21504)
//
// Geometry shared by both (head_dim 256 sliding / 512 global, vocab, ctx, softcap, RoPE) stays in
// gemma4_kernels.cuh. Only the size/shape constants that differ between checkpoints live here.
//
// NOTE (M0 structural item): 12B global attention is specialized for ONE global KV head
// (broadcast). 31B global is 4-KV-head GQA. Flipping GEMMA4_GLOBAL_KV_HEADS is necessary but NOT
// sufficient — the global_attn_splitk* kernel family must handle NKV=4. Tracked in the plan doc.

#ifndef GEMMA4_CONFIG_H
#define GEMMA4_CONFIG_H

#if defined(GEMMA4_VARIANT_31B)

  // ── Gemma-4-31B-it ──────────────────────────────────────────────────
  #define GEMMA4_MAX_LAYERS        60      // 50 sliding + 10 global (every 6th is global)
  #define GEMMA4_HIDDEN_SIZE       5376
  #define GEMMA4_INTERMEDIATE      21504   // 4× hidden
  #define GEMMA4_HEADS             32
  #define GEMMA4_KV_HEADS          16      // sliding-layer KV heads
  #define GEMMA4_GLOBAL_KV_HEADS   4       // global-layer KV heads (GQA, not broadcast)

#else

  // ── Gemma-4-12B-it (default) ────────────────────────────────────────
  #define GEMMA4_MAX_LAYERS        48      // 40 sliding + 8 global
  #define GEMMA4_HIDDEN_SIZE       3840
  #define GEMMA4_INTERMEDIATE      15360   // 4× hidden
  #define GEMMA4_HEADS             16
  #define GEMMA4_KV_HEADS          8       // sliding-layer KV heads
  #define GEMMA4_GLOBAL_KV_HEADS   1       // single global KV head (broadcast specialization)

#endif

// Shared geometry (identical across 12B / 31B)
#define GEMMA4_HEAD_DIM          256       // sliding-layer head dim
#define GEMMA4_GLOBAL_HEAD_DIM   512       // global-layer head dim

#endif // GEMMA4_CONFIG_H
