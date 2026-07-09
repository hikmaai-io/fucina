#pragma once

// Qwen3.5 engine-owned runtime state. This is deliberately separate from gemma4_engine:
// generic model weights/configuration stay in the parent engine, while hybrid recurrent state,
// attention KV, prefill workspace, graph cache and Qwen-specific quantization caches live here.
// The implementation remains in one CUDA translation unit for now; ownership is no longer mixed.
struct qwen35_runtime_state {
    int ready;
    int capacity;
    int maxctx;
    int graph_enabled;

    // Stable device pointer tables indexed by slot. Individual slot allocations are created on
    // first admission and retained for reuse; graphs dereference these tables at replay time.
    __nv_bfloat16 **S[GEMMA4_CAP_LAYERS];
    float         **ring[GEMMA4_CAP_LAYERS];
    __half        **Kc[GEMMA4_CAP_LAYERS];
    __half        **Vc[GEMMA4_CAP_LAYERS];
    // Host ownership mirrors for reset/snapshot/teardown. All fixed recurrent state for one slot
    // is one aligned allocation; S_slot/ring_slot are non-owning views into recurrent_slab.
    uint8_t       *recurrent_slab[GEMMA4_MAX_SEQS];
    __nv_bfloat16 *S_slot[GEMMA4_CAP_LAYERS][GEMMA4_MAX_SEQS];
    float         *ring_slot[GEMMA4_CAP_LAYERS][GEMMA4_MAX_SEQS];
    __half        *Kc_slot[GEMMA4_CAP_LAYERS][GEMMA4_MAX_SEQS];
    __half        *Vc_slot[GEMMA4_CAP_LAYERS][GEMMA4_MAX_SEQS];
    uint8_t        slot_allocated[GEMMA4_MAX_SEQS];
    int             kv_capacity[GEMMA4_MAX_SEQS];
    int             allocated_slots;

    // Shared decode/prefill workspace.
    int     *rowslot;
    float   *sb[24];
    float   *chunk_scr;
    int     *pf_pos;
    int32_t *pf_tok;
    int      attn_splits;
    int      attn_tile;
    float   *part_m;
    float   *part_l;
    float   *part_o;
    __nv_bfloat16 *wbf16[2];
    __nv_bfloat16 *xbf16;

    // Full-attention prefill workspace (lazy).
    __nv_bfloat16 *qb;
    __nv_bfloat16 *kb;
    __nv_bfloat16 *vb;
    __nv_bfloat16 *kbx;
    __nv_bfloat16 *vbx;
    __nv_bfloat16 *pb;
    float         *scores;
    int            attn_cap;
    __half        *qh;
    size_t         qh_cap;
    float         *cont_scores;
    __half        *cont_p;
    size_t         cont_cap;

    // Resident prefill weight caches.
    __nv_bfloat16 *wc[GEMMA4_CAP_LAYERS][12];
    int            wcache_on;
    uint8_t       *fp4_w[GEMMA4_CAP_LAYERS][12];
    uint8_t       *fp4_wsc[GEMMA4_CAP_LAYERS][12];
    float         *fp4_gsw;
    int            fp4_on;

    // FP8 weight-to-scale lookup.
    struct fp8_scent { const uint8_t *w; const __nv_bfloat16 *s; };
    fp8_scent     *fp8_scale_tab;
    int            fp8_scale_n;

    // Captured decode graphs.
    cudaGraphExec_t graph[GEMMA4_MAX_SEQS + 1];
    int             graph_failed;
    uint64_t        graph_logged;

    // J-Lens/J-space debugging (strictly opt-in; never allocated in production).
    int             jspace_enabled;
    int             jspace_topk;
    int             jspace_nlayers;
    uint8_t         jspace_layer_enabled[GEMMA4_CAP_LAYERS];
    __half          *jspace_J[GEMMA4_CAP_LAYERS];       // row-major [H,H], fitted J_l
    float           *jspace_hidden;                     // [L,H], latest post-layer residual
    float           *jspace_transport;                  // [H]
    float           *jspace_norm;                       // [H]
    float           *jspace_logits;                     // [vocab]
    float           *jspace_dirs;                       // [L,H], unit intervention directions
    int             *jspace_steer_mask;                 // [L], device runtime control
    float           *jspace_steer_strength;             // device scalar
    size_t           jspace_bytes;

    // Phase-1 accounting. committed includes eager workspace + currently allocated state;
    // reserved is the configured worst-case capacity used by admission decisions.
    size_t workspace_bytes;
    size_t per_slot_recurrent_bytes;
    size_t per_slot_kv_bytes;
    size_t committed_bytes;
    size_t reserved_bytes;
    size_t peak_bytes;
};
