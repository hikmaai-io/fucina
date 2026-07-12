#pragma once

// Linux GB10 exposes CUDA unified memory through system RAM. cudaMemGetInfo reports raw free
// pages, while MemAvailable includes reclaimable file cache. Keep both visible so admission can
// distinguish actual allocations from safetensors mmap cache growth.
struct q35_host_meminfo { size_t mem_free, mem_available, cached, sreclaimable; };
static q35_host_meminfo q35_read_host_meminfo() {
    q35_host_meminfo m{};
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return m;
    char line[256], key[64]; unsigned long long kib=0;
    while (fgets(line,sizeof(line),f)) {
        if (sscanf(line,"%63s %llu",key,&kib) != 2) continue;
        size_t bytes=(size_t)kib*1024;
        if (!strcmp(key,"MemFree:")) m.mem_free=bytes;
        else if (!strcmp(key,"MemAvailable:")) m.mem_available=bytes;
        else if (!strcmp(key,"Cached:")) m.cached=bytes;
        else if (!strcmp(key,"SReclaimable:")) m.sreclaimable=bytes;
    }
    fclose(f);
    return m;
}

static size_t q35_physical_available(size_t cuda_free) {
    q35_host_meminfo m=q35_read_host_meminfo();
    return m.mem_available > cuda_free ? m.mem_available : cuda_free;
}

// S2b — CUDA-graph cache key (shape triple + dominance dispatch); see the header for semantics.
#include "qwen35_graph_key.cuh"
struct q35_graph_entry {
    q35_graph_key   key;
    cudaGraphExec_t exec;
};
#define Q35_GRAPH_CACHE_CAP 64   // decode buckets (<=32) + headroom for (1+K) spec-decode keys

// Qwen3.5 engine-owned runtime state. This is deliberately separate from gemma4_engine:
// generic model weights/configuration stay in the parent engine, while hybrid recurrent state,
// attention KV, prefill workspace, graph cache and Qwen-specific quantization caches live here.
// The implementation remains in one CUDA translation unit for now; ownership is no longer mixed.
struct qwen35_runtime_state {
    int ready;
    int capacity;
    int maxctx;
    int reserved_context;
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
    WorkspaceRef   recurrent_workspace[GEMMA4_MAX_SEQS];
    __nv_bfloat16 *S_slot[GEMMA4_CAP_LAYERS][GEMMA4_MAX_SEQS];
    float         *ring_slot[GEMMA4_CAP_LAYERS][GEMMA4_MAX_SEQS];
    __half        *Kc_slot[GEMMA4_CAP_LAYERS][GEMMA4_MAX_SEQS];
    __half        *Vc_slot[GEMMA4_CAP_LAYERS][GEMMA4_MAX_SEQS];
    WorkspaceRef   kv_key_workspace[GEMMA4_CAP_LAYERS][GEMMA4_MAX_SEQS];
    WorkspaceRef   kv_value_workspace[GEMMA4_CAP_LAYERS][GEMMA4_MAX_SEQS];
    uint8_t        slot_allocated[GEMMA4_MAX_SEQS];
    int             kv_capacity[GEMMA4_MAX_SEQS];
    int             allocated_slots;

    // Shared decode/prefill workspace.
    int     *rowslot;
    WorkspaceRef routing_workspace;
    float   *sb[24];
    WorkspaceRef decode_workspace[24];
    float   *chunk_scr;
    WorkspaceRef gdn_workspace;
    int     *pf_pos;
    int32_t *pf_tok;
    WorkspaceRef prefill_position_workspace;
    WorkspaceRef prefill_token_workspace;
    int      attn_splits;
    int      attn_tile;
    float   *part_m;
    float   *part_l;
    float   *part_o;
    WorkspaceRef attention_workspace[3];
    __nv_bfloat16 *wbf16[2];
    __nv_bfloat16 *xbf16;
    WorkspaceRef   prefill_weight_workspace[2];
    WorkspaceRef   prefill_input_workspace;

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
    WeightRef      wc_ref[GEMMA4_CAP_LAYERS][12];
    int            wcache_on;
    uint8_t       *fp4_w[GEMMA4_CAP_LAYERS][12];
    uint8_t       *fp4_wsc[GEMMA4_CAP_LAYERS][12];
    WeightRef      fp4_ref[GEMMA4_CAP_LAYERS][12];
    float         *fp4_gsw;
    int            fp4_on;

    // FP8 weight-to-scale lookup.
    struct fp8_scent { const uint8_t *w; const __nv_bfloat16 *s; };
    fp8_scent     *fp8_scale_tab;
    int            fp8_scale_n;

    // Captured decode graphs, keyed by shape triple (S2b). Linear-probed small cache; entries
    // are exact-shape captures dispatched by q35_graph_dominates (exact match today, strict
    // dominance once S2a padding lands).
    q35_graph_entry graph_cache[Q35_GRAPH_CACHE_CAP];
    int             graph_count;
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
    // reserved covers the configured typical-context capacity; state may grow beyond it
    // transactionally up to maxctx, declining cleanly if physical/budget headroom disappears.
    size_t model_bytes;
    size_t workspace_bytes;
    size_t per_slot_recurrent_bytes;
    size_t per_slot_kv_bytes;
    size_t reserved_slot_kv_bytes;
    size_t committed_bytes;
    size_t reserved_bytes;
    size_t peak_bytes;

    // Opt-in prefill phase telemetry (--timings / FUCINA_QWEN35_PREFILL_TIMINGS=1).
    int    prefill_timing;
    double prefill_dequant_ms;
    double prefill_router_ms;
    double prefill_expert_ms;
    double prefill_shared_ms;
};
