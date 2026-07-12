// ABOUTME: Host-testable DFlash shape/lookahead planner + enable/concurrency gate (P1/P4 of S1a).
// ABOUTME: Derives the (1+K) verify batch shape, S2 graph key, and KV lookahead from config only.
//
// This is pure integer planning shared by the scheduler and the verify path. It answers three
// questions with no CUDA and no weights:
//   1. What is the uniform per-request query count for a DFlash verify step?  -> 1 + K
//   2. What S2 graph key does an R-request verify batch use?                  -> spec key
//   3. How many KV lookahead slots must the scheduler reserve per request?    -> K + 1
//   4. Is DFlash enabled for this step, given the mode flag and batch size?   -> gate
//
// K (num speculative/draft tokens) is CONFIG-DERIVED (the draft's dflash_config.block_size),
// clamped to a safe planner maximum; it is never a hard-coded model constant. The enable flag
// mirrors FUCINA_QWEN35_DFLASH: OFF (0) is the default until real-checkpoint acceptance is proven,
// ON (1) forces it within the concurrency gate, AUTO (2) is reserved for a future measured policy
// and behaves like ON-within-gate. The concurrency gate disables DFlash at or above a conservative
// critical batch size because verification inflates compute and speculation goes net-negative on a
// 48-SM part past low batch (measured critical B is filled in from GB10 sweeps later).
#ifndef FUCINA_QWEN35_DFLASH_PLAN_CUH
#define FUCINA_QWEN35_DFLASH_PLAN_CUH

#include "qwen35_graph_key.cuh"

#if defined(__CUDACC__)
#define Q35_DFLASH_PLAN_HD __host__ __device__
#else
#define Q35_DFLASH_PLAN_HD
#endif

// Hard planner bounds. K is the number of drafted (speculative) tokens per step; the query block is
// 1 (bonus/anchor) + K. These bound device buffer sizing and are intentionally conservative.
enum {
    Q35_DFLASH_K_MAX = 16,           // max drafted tokens per step (config-clamped)
    Q35_DFLASH_MODE_OFF  = 0,
    Q35_DFLASH_MODE_ON   = 1,
    Q35_DFLASH_MODE_AUTO = 2,
    // Conservative default critical batch: DFlash disabled for num_reqs >= this until a GB10 sweep
    // establishes the real crossover. 8 is the analysis' upper bound; keep it as the cap.
    Q35_DFLASH_CRITICAL_BATCH_DEFAULT = 8,
};

struct q35_dflash_plan {
    int K;                    // drafted tokens per step (config-derived, clamped to K_MAX)
    int uniform_token_count;  // query rows per request in a verify step = 1 + K
    int lookahead_slots;      // KV slots to reserve per request = K + 1
    int enabled;              // 1 if DFlash should run for this step, else 0
};

// Clamp a config-derived K (e.g. dflash_config.block_size) to the safe planner range [1, K_MAX].
Q35_DFLASH_PLAN_HD static inline int q35_dflash_clamp_k(int cfg_k) {
    if (cfg_k < 1) return 1;
    if (cfg_k > Q35_DFLASH_K_MAX) return Q35_DFLASH_K_MAX;
    return cfg_k;
}

// Query rows per request for a (1+K) verify step.
Q35_DFLASH_PLAN_HD static inline int q35_dflash_uniform_tokens(int K) { return 1 + K; }

// KV lookahead per request. DFlash needs K+1: one query for the last sampled/bonus token plus K
// queries for the drafted tokens (matches vLLM scheduler.use_dflash() -> num_spec_tokens + 1).
Q35_DFLASH_PLAN_HD static inline int q35_dflash_lookahead(int K) { return K + 1; }

// S2 graph key for an R-request verify batch: each request contributes exactly (1+K) uniform query
// rows, so the key is the spec key (R*(1+K), R, 1+K). Distinct from any plain-decode key
// (uniform_token_count 1), so a verify graph never aliases a decode graph.
Q35_DFLASH_PLAN_HD static inline q35_graph_key q35_dflash_graph_key(int num_reqs, int K) {
    return q35_make_spec_key(num_reqs, q35_dflash_uniform_tokens(K));
}

// Enable/concurrency gate. `mode` is the FUCINA_QWEN35_DFLASH value; `num_reqs` is the batch's
// request count; `critical_batch` is the disable threshold (<=0 uses the conservative default).
// Returns 1 iff DFlash should run this step. OFF is always disabled; ON/AUTO run only for
// 0 < num_reqs < critical_batch.
Q35_DFLASH_PLAN_HD static inline int q35_dflash_gate(int mode, int num_reqs, int critical_batch) {
    if (mode == Q35_DFLASH_MODE_OFF) return 0;
    if (num_reqs <= 0) return 0;
    int cb = (critical_batch > 0) ? critical_batch : Q35_DFLASH_CRITICAL_BATCH_DEFAULT;
    return num_reqs < cb ? 1 : 0;
}

// Build the full plan for a step. cfg_k is config-derived; mode/num_reqs/critical_batch drive the
// gate. When disabled, enabled=0 and the caller MUST fall back to the plain decode path (which is
// byte-identical to current main).
Q35_DFLASH_PLAN_HD static inline q35_dflash_plan q35_dflash_make_plan(
        int cfg_k, int mode, int num_reqs, int critical_batch) {
    q35_dflash_plan p;
    p.K = q35_dflash_clamp_k(cfg_k);
    p.uniform_token_count = q35_dflash_uniform_tokens(p.K);
    p.lookahead_slots = q35_dflash_lookahead(p.K);
    p.enabled = q35_dflash_gate(mode, num_reqs, critical_batch);
    return p;
}

// Parse the FUCINA_QWEN35_DFLASH env value ("0"/"1"/"auto", default 0). Host helper; kept here so
// the runtime and tests agree on the mapping. Plain host function (no device call site), so it is
// safe in a device TU without a __CUDA_ARCH__ guard.
#include <cstdlib>
#include <cstring>
static inline int q35_dflash_mode_from_env(const char *val) {
    if (!val || !*val) return Q35_DFLASH_MODE_OFF;
    if (!strcmp(val, "1")) return Q35_DFLASH_MODE_ON;
    if (!strcmp(val, "auto") || !strcmp(val, "AUTO")) return Q35_DFLASH_MODE_AUTO;
    return Q35_DFLASH_MODE_OFF;   // "0" and anything unrecognized => OFF (safe default)
}

#endif // FUCINA_QWEN35_DFLASH_PLAN_CUH
