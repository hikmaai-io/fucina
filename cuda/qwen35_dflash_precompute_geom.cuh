// ABOUTME: Config-derived context-KV precompute geometry for the DFlash draft (P3 shape planning).
// ABOUTME: Pure host/device arithmetic; sizes the fused KV GEMM, grouped K-norm, RoPE, cache insert.
//
// DFlash never re-runs the draft layers over the context. Instead the TARGET model's hidden states
// for the step's context tokens are projected into the draft's K/V for ALL draft layers at once
// (qwen3_dflash.py precompute_and_store_context_kv):
//   1. RMSNorm the num_ctx context hidden rows (hidden_norm), [num_ctx, H].
//   2. One fused GEMM against the stacked per-layer KV projection weights
//      [L * 2 * kv_dim, H] -> [num_ctx, L * 2 * kv_dim], separated to K/V per layer.
//   3. Grouped per-layer K-RMSNorm using a stacked [L, HD] weight (one kernel over all layers).
//   4. Batched RoPE over L * num_ctx rows of kv_dim (K only).
//   5. Per-layer KV-cache insert at the context slot mapping.
// This header computes ALL element counts and strides from the loader Geometry, so P3 can size its
// device buffers exactly and assert the fused-GEMM shape before allocating. No kernel math, no
// weights: pure shape arithmetic, host/device identical, overflow-guarded via 64-bit.
#ifndef FUCINA_QWEN35_DFLASH_PRECOMPUTE_GEOM_CUH
#define FUCINA_QWEN35_DFLASH_PRECOMPUTE_GEOM_CUH

#include <cstdint>

#if defined(__CUDACC__)
#define Q35_DFLASH_PC_HD __host__ __device__
#else
#define Q35_DFLASH_PC_HD
#endif

// All sizes are element COUNTS (not bytes); the caller multiplies by the chosen dtype size. int64
// throughout so large num_ctx * L * kv_dim products never overflow.
struct q35_dflash_precompute_geom {
    int   L;              // draft layers
    int   H;              // hidden size
    int   HD;             // head dim
    int   NKV;            // draft KV heads
    int   kv_dim;         // NKV * HD (one of K or V per layer)
    int   num_ctx;        // context rows this step

    int64_t fused_kv_weight_rows;   // L * 2 * kv_dim  (rows of the stacked KV projection weight)
    int64_t fused_kv_weight_elems;  // fused_kv_weight_rows * H
    int64_t kv_proj_out_elems;      // num_ctx * L * 2 * kv_dim  (GEMM output, K and V interleaved)
    int64_t k_all_elems;            // L * num_ctx * kv_dim       (K across all layers)
    int64_t v_all_elems;            // L * num_ctx * kv_dim       (V across all layers)
    int64_t knorm_weight_elems;     // L * HD   (grouped K-norm weight stack)
    int64_t rope_rows;              // L * num_ctx  (batched RoPE row count over K)
    int64_t ctx_norm_elems;         // num_ctx * H  (normed context hidden rows)
};

// Compute the precompute geometry from config-derived dims. Returns false if any dim is
// non-positive or a product would overflow int64 (defensive; real configs are far below this).
Q35_DFLASH_PC_HD static inline bool q35_dflash_precompute_geometry(
        int L, int H, int HD, int NKV, int num_ctx, q35_dflash_precompute_geom *out) {
    if (!out) return false;
    if (L <= 0 || H <= 0 || HD <= 0 || NKV <= 0 || num_ctx <= 0) return false;
    const int64_t kv_dim = (int64_t)NKV * HD;
    // Guard the largest product (num_ctx * L * 2 * kv_dim) against int64 overflow.
    const int64_t LIMIT = (int64_t)1 << 60;
    int64_t two_kv = 2 * kv_dim;
    if (kv_dim <= 0 || two_kv / 2 != kv_dim) return false;
    int64_t rows = (int64_t)L * two_kv;
    if (L != 0 && rows / two_kv != (int64_t)L) return false;
    if (rows > LIMIT / (H > 0 ? H : 1)) return false;
    int64_t proj_out = (int64_t)num_ctx * rows;
    if (num_ctx != 0 && proj_out / rows != (int64_t)num_ctx) return false;
    if (proj_out > LIMIT) return false;

    out->L = L; out->H = H; out->HD = HD; out->NKV = NKV;
    out->kv_dim = (int)kv_dim; out->num_ctx = num_ctx;
    out->fused_kv_weight_rows  = rows;
    out->fused_kv_weight_elems = rows * (int64_t)H;
    out->kv_proj_out_elems     = proj_out;
    out->k_all_elems           = (int64_t)L * num_ctx * kv_dim;
    out->v_all_elems           = (int64_t)L * num_ctx * kv_dim;
    out->knorm_weight_elems    = (int64_t)L * HD;
    out->rope_rows             = (int64_t)L * num_ctx;
    out->ctx_norm_elems        = (int64_t)num_ctx * H;
    return true;
}

#endif // FUCINA_QWEN35_DFLASH_PRECOMPUTE_GEOM_CUH
