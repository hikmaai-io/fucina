// gemma4_detect.h — runtime Gemma-4 architecture auto-detection.
//
// fucina is ONE binary: the model architecture (layer count, hidden size, FFN, head counts,
// per-layer attention pattern, softcap, rope thetas) is read from the checkpoint's OWN metadata
// at load time — GGUF kv for *.gguf, config.json for safetensors — and written into a
// gemma4_model_config_t. NO compiler flags, NO env vars select the model size. See M0 in
// docs/dense-31b-89tok-plan.md.
//
// This header is deliberately self-contained and CUDA-free: it carries its own minimal GGUF
// metadata reader (g4d_* prefix, distinct from the engine's gguf_* parser so it can be included
// into the engine TU without symbol clashes) and a tiny config.json scalar scanner. That lets a
// standalone host test fill + print the config WITHOUT building the CUDA engine — proving
// auto-detection in isolation.

#ifndef GEMMA4_DETECT_H
#define GEMMA4_DETECT_H

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "gemma4_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// ── Minimal, self-contained GGUF metadata reader (host, no CUDA) ────────────────────────────────
// Only what detection needs: scalar uint32/float32 lookups and typed-array lookups. The byte
// layout matches the engine's parser exactly (see gguf_* in gemma4_kernels.cu).

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;            // 0x46554747 "GGUF"
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
} g4d_gguf_header_t;
#pragma pack(pop)

enum {
    G4D_T_UINT8 = 0, G4D_T_INT8 = 1, G4D_T_UINT16 = 2, G4D_T_INT16 = 3,
    G4D_T_UINT32 = 4, G4D_T_INT32 = 5, G4D_T_FLOAT32 = 6, G4D_T_BOOL = 7,
    G4D_T_STRING = 8, G4D_T_ARRAY = 9, G4D_T_UINT64 = 10, G4D_T_INT64 = 11,
    G4D_T_FLOAT64 = 12,
};

typedef struct { uint32_t elem_type; uint64_t count; const uint8_t *data; } g4d_array_t;

static inline uint64_t g4d_scalar_size(uint32_t t) {
    switch (t) {
        case G4D_T_UINT8: case G4D_T_INT8: case G4D_T_BOOL:                return 1;
        case G4D_T_UINT16: case G4D_T_INT16:                              return 2;
        case G4D_T_UINT32: case G4D_T_INT32: case G4D_T_FLOAT32:          return 4;
        case G4D_T_UINT64: case G4D_T_INT64: case G4D_T_FLOAT64:          return 8;
        default: return 0; // STRING / ARRAY are variable
    }
}

static inline const char* g4d_read_str(const uint8_t **pp, const uint8_t *end, uint64_t *len_out) {
    const uint8_t *p = *pp;
    if (p + 8 > end) return NULL;
    uint64_t len; memcpy(&len, p, 8); p += 8;
    if (p + len > end) return NULL;
    const char *s = (const char *)p;
    p += len; *pp = p;
    if (len_out) *len_out = len;
    return s;
}

static inline int g4d_str_eq(const char *s, uint64_t len, const char *cstr) {
    return strlen(cstr) == len && memcmp(s, cstr, len) == 0;
}

static inline int g4d_skip_value(const uint8_t **pp, const uint8_t *end, uint32_t vtype) {
    const uint8_t *p = *pp;
    if (vtype == G4D_T_STRING) {
        if (!g4d_read_str(&p, end, NULL)) return -1;
    } else if (vtype == G4D_T_ARRAY) {
        if (p + 12 > end) return -1;
        uint32_t at; memcpy(&at, p, 4);
        uint64_t n;  memcpy(&n, p + 4, 8);
        p += 12;
        if (at == G4D_T_STRING) {
            for (uint64_t i = 0; i < n; i++)
                if (!g4d_read_str(&p, end, NULL)) return -1;
        } else {
            uint64_t sz = g4d_scalar_size(at);
            if (sz == 0) return -1;
            if (p + sz * n > end) return -1;
            p += sz * n;
        }
    } else {
        uint64_t sz = g4d_scalar_size(vtype);
        if (sz == 0 || p + sz > end) return -1;
        p += sz;
    }
    *pp = p;
    return 0;
}

// Find metadata value `key`. On match, copies a scalar into value_out (for scalar vtypes) OR
// fills a g4d_array_t (for G4D_T_ARRAY). Returns 0 on found+type-match, -1 otherwise.
static inline int g4d_meta(const uint8_t *data, uint64_t size, const char *key,
                           uint32_t expected_type, void *value_out) {
    const g4d_gguf_header_t *hdr = (const g4d_gguf_header_t *)data;
    if (size < sizeof(*hdr) || hdr->magic != 0x46554747u) return -1;
    const uint8_t *end = data + size;
    const uint8_t *p = data + sizeof(*hdr);
    for (uint64_t i = 0; i < hdr->metadata_kv_count; i++) {
        uint64_t klen = 0;
        const char *k = g4d_read_str(&p, end, &klen);
        if (!k) return -1;
        if (p + 4 > end) return -1;
        uint32_t vtype; memcpy(&vtype, p, 4); p += 4;
        if (g4d_str_eq(k, klen, key)) {
            if (vtype != expected_type) return -1;
            if (vtype == G4D_T_ARRAY) {
                if (p + 12 > end) return -1;
                g4d_array_t arr;
                memcpy(&arr.elem_type, p, 4);
                memcpy(&arr.count, p + 4, 8);
                arr.data = p + 12;
                memcpy(value_out, &arr, sizeof(arr));
            } else {
                uint64_t sz = g4d_scalar_size(vtype);
                if (sz == 0 || p + sz > end) return -1;
                memcpy(value_out, p, sz);
            }
            return 0;
        }
        if (g4d_skip_value(&p, end, vtype) != 0) return -1;
    }
    return -1;
}

static inline int g4d_u32(const uint8_t *d, uint64_t s, const char *k, uint32_t *out) {
    return g4d_meta(d, s, k, G4D_T_UINT32, out);
}
static inline int g4d_f32(const uint8_t *d, uint64_t s, const char *k, float *out) {
    return g4d_meta(d, s, k, G4D_T_FLOAT32, out);
}

// Copy the value of a STRING metadata key into out (NUL-terminated, truncated to outlen).
// Returns 0 on found, -1 otherwise. Mirrors g4d_meta's kv-walk (STRING isn't a fixed scalar).
static inline int g4d_str(const uint8_t *data, uint64_t size, const char *key,
                          char *out, size_t outlen) {
    const g4d_gguf_header_t *hdr = (const g4d_gguf_header_t *)data;
    if (size < sizeof(*hdr) || hdr->magic != 0x46554747u) return -1;
    const uint8_t *end = data + size;
    const uint8_t *p = data + sizeof(*hdr);
    for (uint64_t i = 0; i < hdr->metadata_kv_count; i++) {
        uint64_t klen = 0;
        const char *k = g4d_read_str(&p, end, &klen);
        if (!k) return -1;
        if (p + 4 > end) return -1;
        uint32_t vtype; memcpy(&vtype, p, 4); p += 4;
        if (g4d_str_eq(k, klen, key)) {
            if (vtype != G4D_T_STRING) return -1;
            uint64_t vlen = 0;
            const char *v = g4d_read_str(&p, end, &vlen);
            if (!v) return -1;
            size_t n = (vlen < outlen - 1) ? (size_t)vlen : (outlen - 1);
            memcpy(out, v, n); out[n] = '\0';
            return 0;
        }
        if (g4d_skip_value(&p, end, vtype) != 0) return -1;
    }
    return -1;
}

// Qwen3 dense → gemma4_model_config_t. Qwen3 is single-head-dim, FULL-causal on every layer
// (no sliding/global split), separate V projection, no softcap. Marked arch=GEMMA4_ARCH_QWEN3;
// all layers are treated as the engine's "global" (full-attention) class. See gemma4_detect.h
// docs and the qwen3.* metadata keys.
static inline int gemma4_detect_qwen3(const uint8_t *data, uint64_t size,
                                      gemma4_model_config_t *cfg,
                                      char *err, size_t errlen) {
    #define G4D_FAIL(...) do { if (err) snprintf(err, errlen, __VA_ARGS__); return -1; } while (0)
    uint32_t bc = 0, embd = 0, ffn = 0, hc = 0, hckv = 0, kl = 0;
    if (g4d_u32(data, size, "qwen3.block_count", &bc) != 0)            G4D_FAIL("missing qwen3.block_count");
    if (g4d_u32(data, size, "qwen3.embedding_length", &embd) != 0)     G4D_FAIL("missing qwen3.embedding_length");
    if (g4d_u32(data, size, "qwen3.feed_forward_length", &ffn) != 0)   G4D_FAIL("missing qwen3.feed_forward_length");
    if (g4d_u32(data, size, "qwen3.attention.head_count", &hc) != 0)   G4D_FAIL("missing qwen3.attention.head_count");
    if (g4d_u32(data, size, "qwen3.attention.head_count_kv", &hckv) != 0) G4D_FAIL("missing qwen3.attention.head_count_kv");
    if (g4d_u32(data, size, "qwen3.attention.key_length", &kl) != 0)   G4D_FAIL("missing qwen3.attention.key_length");

    if ((int)bc > GEMMA4_CAP_LAYERS) G4D_FAIL("block_count %u exceeds GEMMA4_CAP_LAYERS %d", bc, GEMMA4_CAP_LAYERS);
    if ((int)hc > GEMMA4_CAP_HEADS)  G4D_FAIL("head_count %u exceeds GEMMA4_CAP_HEADS %d", hc, GEMMA4_CAP_HEADS);
    if ((int)hckv > GEMMA4_CAP_KV_HEADS) G4D_FAIL("head_count_kv %u exceeds GEMMA4_CAP_KV_HEADS %d", hckv, GEMMA4_CAP_KV_HEADS);
    if ((int)kl > GEMMA4_GLOBAL_HEAD_DIM) G4D_FAIL("key_length %u exceeds GEMMA4_GLOBAL_HEAD_DIM %d", kl, GEMMA4_GLOBAL_HEAD_DIM);

    cfg->arch         = GEMMA4_ARCH_QWEN3;
    cfg->head_dim     = (int)kl;          // 128
    cfg->n_layers     = (int)bc;
    cfg->hidden_size  = (int)embd;
    cfg->intermediate = (int)ffn;
    cfg->n_heads      = (int)hc;
    // Qwen3 has the SAME KV head count on every layer (GQA). The engine routes all layers
    // through its "global" full-attention class, so n_kv_global carries the real per-layer KV
    // count; n_kv_sliding is set equal so the sliding-pool sizing/guards stay valid (unused).
    cfg->n_kv_global  = (int)hckv;
    cfg->n_kv_sliding = (int)hckv;

    uint32_t vocab = 0;
    if (g4d_u32(data, size, "qwen3.vocab_size", &vocab) == 0 && vocab > 0) {
        cfg->vocab_size = (int)vocab;
    } else {
        g4d_array_t toks;
        if (g4d_meta(data, size, "tokenizer.ggml.tokens", G4D_T_ARRAY, &toks) == 0)
            cfg->vocab_size = (int)toks.count;
    }
    if (cfg->vocab_size <= 0) G4D_FAIL("qwen3: could not determine vocab size");

    cfg->softcap = 0.0f;   // Qwen3 has no logit softcap

    float fb = 1000000.0f;
    g4d_f32(data, size, "qwen3.rope.freq_base", &fb);
    cfg->rope_theta_global  = fb;
    cfg->rope_theta_sliding = fb;   // single theta; sliding unused

    // Every layer is full-causal (the engine's "global" class).
    cfg->n_global = (int)bc;
    for (int i = 0; i < cfg->n_layers && i < GEMMA4_CAP_LAYERS; i++) cfg->is_global[i] = 1;
    return 0;
    #undef G4D_FAIL
}

// Qwen3 MoE (qwen3moe) → gemma4_model_config_t. The attention/norm/rope/KV/head_dim layout is
// IDENTICAL to Qwen3 dense (full-causal every layer, separate V, learned q/k norms, no softcap);
// the only architectural difference is the FFN, which is a sparse 128-expert top-8 SiLU-GLU
// mixture instead of one dense SiLU-GLU. We read the qwen3moe.* namespaced keys (mirroring
// detect_qwen3) plus the expert geometry (expert_count / expert_used_count /
// expert_feed_forward_length). cfg.intermediate carries the DENSE feed_forward_length (present in
// the GGUF but unused by the MoE FFN) so all existing dense scratch sizing stays valid and ample.
static inline int gemma4_detect_qwen3moe(const uint8_t *data, uint64_t size,
                                         gemma4_model_config_t *cfg,
                                         char *err, size_t errlen) {
    #define G4D_FAIL(...) do { if (err) snprintf(err, errlen, __VA_ARGS__); return -1; } while (0)
    uint32_t bc = 0, embd = 0, ffn = 0, hc = 0, hckv = 0, kl = 0;
    uint32_t nexp = 0, nused = 0, effn = 0;
    if (g4d_u32(data, size, "qwen3moe.block_count", &bc) != 0)              G4D_FAIL("missing qwen3moe.block_count");
    if (g4d_u32(data, size, "qwen3moe.embedding_length", &embd) != 0)       G4D_FAIL("missing qwen3moe.embedding_length");
    if (g4d_u32(data, size, "qwen3moe.feed_forward_length", &ffn) != 0)     ffn = 0;  // dense value, unused by MoE
    if (g4d_u32(data, size, "qwen3moe.attention.head_count", &hc) != 0)     G4D_FAIL("missing qwen3moe.attention.head_count");
    if (g4d_u32(data, size, "qwen3moe.attention.head_count_kv", &hckv) != 0) G4D_FAIL("missing qwen3moe.attention.head_count_kv");
    if (g4d_u32(data, size, "qwen3moe.attention.key_length", &kl) != 0)     G4D_FAIL("missing qwen3moe.attention.key_length");
    if (g4d_u32(data, size, "qwen3moe.expert_count", &nexp) != 0)           G4D_FAIL("missing qwen3moe.expert_count");
    if (g4d_u32(data, size, "qwen3moe.expert_used_count", &nused) != 0)     G4D_FAIL("missing qwen3moe.expert_used_count");
    if (g4d_u32(data, size, "qwen3moe.expert_feed_forward_length", &effn) != 0) G4D_FAIL("missing qwen3moe.expert_feed_forward_length");

    if ((int)bc > GEMMA4_CAP_LAYERS) G4D_FAIL("block_count %u exceeds GEMMA4_CAP_LAYERS %d", bc, GEMMA4_CAP_LAYERS);
    if ((int)hc > GEMMA4_CAP_HEADS)  G4D_FAIL("head_count %u exceeds GEMMA4_CAP_HEADS %d", hc, GEMMA4_CAP_HEADS);
    if ((int)hckv > GEMMA4_CAP_KV_HEADS) G4D_FAIL("head_count_kv %u exceeds GEMMA4_CAP_KV_HEADS %d", hckv, GEMMA4_CAP_KV_HEADS);
    if ((int)kl > GEMMA4_GLOBAL_HEAD_DIM) G4D_FAIL("key_length %u exceeds GEMMA4_GLOBAL_HEAD_DIM %d", kl, GEMMA4_GLOBAL_HEAD_DIM);
    // Cap the expert counts: the MoE top-k router kernel indexes a fixed-size stack array
    // (`bool used[128]`) by the runtime expert_count, so a >128-expert checkpoint would
    // corrupt the stack. Fail cleanly here instead. (Mirrors the dim caps above.)
    if ((int)nexp > GEMMA4_CAP_EXPERTS)       G4D_FAIL("expert_count %u exceeds GEMMA4_CAP_EXPERTS %d", nexp, GEMMA4_CAP_EXPERTS);
    if ((int)nused > GEMMA4_CAP_EXPERTS_USED) G4D_FAIL("expert_used_count %u exceeds GEMMA4_CAP_EXPERTS_USED %d", nused, GEMMA4_CAP_EXPERTS_USED);

    cfg->arch         = GEMMA4_ARCH_QWEN3MOE;
    cfg->head_dim     = (int)kl;            // 128
    cfg->n_layers     = (int)bc;
    cfg->hidden_size  = (int)embd;
    cfg->intermediate = (ffn > 0) ? (int)ffn : (int)effn; // dense FFN length (scratch sizing only)
    cfg->n_heads      = (int)hc;
    cfg->n_kv_global  = (int)hckv;
    cfg->n_kv_sliding = (int)hckv;
    cfg->n_experts      = (int)nexp;
    cfg->n_experts_used = (int)nused;
    cfg->expert_ffn     = (int)effn;

    uint32_t vocab = 0;
    if (g4d_u32(data, size, "qwen3moe.vocab_size", &vocab) == 0 && vocab > 0) {
        cfg->vocab_size = (int)vocab;
    } else {
        g4d_array_t toks;
        if (g4d_meta(data, size, "tokenizer.ggml.tokens", G4D_T_ARRAY, &toks) == 0)
            cfg->vocab_size = (int)toks.count;
    }
    if (cfg->vocab_size <= 0) G4D_FAIL("qwen3moe: could not determine vocab size");

    cfg->softcap = 0.0f;    // Qwen3 has no logit softcap

    float fb = 1000000.0f;
    g4d_f32(data, size, "qwen3moe.rope.freq_base", &fb);
    cfg->rope_theta_global  = fb;
    cfg->rope_theta_sliding = fb;

    // Every layer is full-causal (the engine's "global" class).
    cfg->n_global = (int)bc;
    for (int i = 0; i < cfg->n_layers && i < GEMMA4_CAP_LAYERS; i++) cfg->is_global[i] = 1;
    return 0;
    #undef G4D_FAIL
}

// Qwen3.5 hybrid (qwen35) → gemma4_model_config_t. Per-layer mix of FULL softmax-GQA layers
// (output-gated, partial-RoPE, learned per-head q/k norm — a superset of the Qwen3-dense full-attn
// block) and LINEAR gated-deltanet (SSM) layers. The cadence is period-`full_attention_interval`:
// layer i is FULL iff (i+1)%interval==0, else LINEAR. We read the qwen35.* namespaced keys: the
// usual block/embedding/ffn/head geometry (key_length 256, head_count 16, head_count_kv 4 GQA),
// the partial-rotary width (rope.dimension_count 64 of head_dim 256), the full-attention interval,
// and the gated-deltanet ssm.* dims (state 128, conv 4, inner 4096, group 16, time_step_rank 32).
// FULL layers are also flagged is_global[]=1 so they reuse the engine's global full-attention
// class; LINEAR layers carry attn_kind=GEMMA4_ATTN_LINEAR for the (later) GDN forward dispatch.
static inline int gemma4_detect_qwen3_5(const uint8_t *data, uint64_t size,
                                        gemma4_model_config_t *cfg,
                                        char *err, size_t errlen) {
    #define G4D_FAIL(...) do { if (err) snprintf(err, errlen, __VA_ARGS__); return -1; } while (0)
    uint32_t bc = 0, embd = 0, ffn = 0, hc = 0, hckv = 0, kl = 0;
    if (g4d_u32(data, size, "qwen35.block_count", &bc) != 0)              G4D_FAIL("missing qwen35.block_count");
    if (g4d_u32(data, size, "qwen35.embedding_length", &embd) != 0)       G4D_FAIL("missing qwen35.embedding_length");
    if (g4d_u32(data, size, "qwen35.feed_forward_length", &ffn) != 0)     G4D_FAIL("missing qwen35.feed_forward_length");
    if (g4d_u32(data, size, "qwen35.attention.head_count", &hc) != 0)     G4D_FAIL("missing qwen35.attention.head_count");
    if (g4d_u32(data, size, "qwen35.attention.head_count_kv", &hckv) != 0) G4D_FAIL("missing qwen35.attention.head_count_kv");
    if (g4d_u32(data, size, "qwen35.attention.key_length", &kl) != 0)     G4D_FAIL("missing qwen35.attention.key_length");

    if ((int)bc > GEMMA4_CAP_LAYERS) G4D_FAIL("block_count %u exceeds GEMMA4_CAP_LAYERS %d", bc, GEMMA4_CAP_LAYERS);
    if ((int)hc > GEMMA4_CAP_HEADS)  G4D_FAIL("head_count %u exceeds GEMMA4_CAP_HEADS %d", hc, GEMMA4_CAP_HEADS);
    if ((int)hckv > GEMMA4_CAP_KV_HEADS) G4D_FAIL("head_count_kv %u exceeds GEMMA4_CAP_KV_HEADS %d", hckv, GEMMA4_CAP_KV_HEADS);
    if ((int)kl > GEMMA4_GLOBAL_HEAD_DIM) G4D_FAIL("key_length %u exceeds GEMMA4_GLOBAL_HEAD_DIM %d", kl, GEMMA4_GLOBAL_HEAD_DIM);

    cfg->arch         = GEMMA4_ARCH_QWEN3_5;
    cfg->head_dim     = (int)kl;          // 256
    cfg->n_layers     = (int)bc;
    cfg->hidden_size  = (int)embd;
    cfg->intermediate = (int)ffn;
    cfg->n_heads      = (int)hc;
    // GQA: same KV head count on every FULL layer. The engine routes FULL layers through its
    // "global" full-attention class, so n_kv_global carries the real per-layer KV count;
    // n_kv_sliding is set equal so the sliding-pool sizing/guards stay valid (unused here).
    cfg->n_kv_global  = (int)hckv;
    cfg->n_kv_sliding = (int)hckv;

    uint32_t vocab = 0;
    if (g4d_u32(data, size, "qwen35.vocab_size", &vocab) == 0 && vocab > 0) {
        cfg->vocab_size = (int)vocab;
    } else {
        g4d_array_t toks;
        if (g4d_meta(data, size, "tokenizer.ggml.tokens", G4D_T_ARRAY, &toks) == 0)
            cfg->vocab_size = (int)toks.count;
    }
    if (cfg->vocab_size <= 0) G4D_FAIL("qwen35: could not determine vocab size");

    cfg->softcap = 0.0f;   // Qwen3.5 has no logit softcap

    float fb = 10000000.0f;   // rope theta 1e7
    g4d_f32(data, size, "qwen35.rope.freq_base", &fb);
    cfg->rope_theta_global  = fb;
    cfg->rope_theta_sliding = fb;

    // Partial rotary: rope is applied to only the first `rope.dimension_count` dims of head_dim.
    uint32_t rdim = 0;
    if (g4d_u32(data, size, "qwen35.rope.dimension_count", &rdim) == 0) cfg->rotary_dim = (int)rdim;

    // Full-attention cadence (period). Default to 4 if the kv is absent.
    uint32_t fai = 4;
    g4d_u32(data, size, "qwen35.full_attention_interval", &fai);
    cfg->full_attention_interval = (int)fai;

    // Gated-DeltaNet (SSM) geometry for the LINEAR layers.
    uint32_t ss = 0, ck = 0, isz = 0, gc = 0, tsr = 0;
    g4d_u32(data, size, "qwen35.ssm.state_size",     &ss);
    g4d_u32(data, size, "qwen35.ssm.conv_kernel",    &ck);
    g4d_u32(data, size, "qwen35.ssm.inner_size",     &isz);
    g4d_u32(data, size, "qwen35.ssm.group_count",    &gc);
    g4d_u32(data, size, "qwen35.ssm.time_step_rank", &tsr);
    cfg->ssm_state_size     = (int)ss;
    cfg->ssm_conv_kernel    = (int)ck;
    cfg->ssm_inner_size     = (int)isz;
    cfg->ssm_group_count    = (int)gc;
    cfg->ssm_time_step_rank = (int)tsr;

    // Per-layer attention kind: FULL iff (i+1)%interval==0, else LINEAR. FULL layers also
    // get is_global[]=1 so they reuse the engine's global full-attention path.
    int interval = (cfg->full_attention_interval > 0) ? cfg->full_attention_interval : 4;
    cfg->n_full   = 0;
    cfg->n_global = 0;
    for (int i = 0; i < cfg->n_layers && i < GEMMA4_CAP_LAYERS; i++) {
        int is_full = (((i + 1) % interval) == 0);
        cfg->attn_kind[i] = (uint8_t)(is_full ? GEMMA4_ATTN_FULL : GEMMA4_ATTN_LINEAR);
        cfg->is_global[i] = (uint8_t)is_full;
        if (is_full) { cfg->n_full++; cfg->n_global++; }
    }
    return 0;
    #undef G4D_FAIL
}

// ── Detection: GGUF → gemma4_model_config_t ─────────────────────────────────────────────────────
// `data`/`size` is the mmap'd (or fully read) GGUF file. Returns 0 on success, -1 on a fatal
// inconsistency (missing required kv, or counts exceeding the compiled capacity maxima). On
// failure it writes a human-readable reason to `err` (if non-NULL, size `errlen`).
static inline int gemma4_detect_from_gguf(const uint8_t *data, uint64_t size,
                                          gemma4_model_config_t *cfg,
                                          char *err, size_t errlen) {
    #define G4D_FAIL(...) do { if (err) snprintf(err, errlen, __VA_ARGS__); return -1; } while (0)
    memset(cfg, 0, sizeof(*cfg));

    // Branch on the checkpoint's declared architecture. Non-Gemma families parse their own
    // namespaced keys; Gemma-4 falls through to the original path below (arch stays GEMMA4).
    {
        char arch[32] = {0};
        if (g4d_str(data, size, "general.architecture", arch, sizeof(arch)) == 0) {
            if (strcmp(arch, "qwen3") == 0)
                return gemma4_detect_qwen3(data, size, cfg, err, errlen);
            if (strcmp(arch, "qwen3moe") == 0)
                return gemma4_detect_qwen3moe(data, size, cfg, err, errlen);
            if (strcmp(arch, "qwen35") == 0)
                return gemma4_detect_qwen3_5(data, size, cfg, err, errlen);
        }
    }
    cfg->arch = GEMMA4_ARCH_GEMMA4;

    uint32_t block_count = 0, embd = 0, ffn = 0, head_count = 0;
    if (g4d_u32(data, size, "gemma4.block_count", &block_count) != 0)
        G4D_FAIL("missing gemma4.block_count");
    if (g4d_u32(data, size, "gemma4.embedding_length", &embd) != 0)
        G4D_FAIL("missing gemma4.embedding_length");
    if (g4d_u32(data, size, "gemma4.feed_forward_length", &ffn) != 0)
        G4D_FAIL("missing gemma4.feed_forward_length");
    if (g4d_u32(data, size, "gemma4.attention.head_count", &head_count) != 0)
        G4D_FAIL("missing gemma4.attention.head_count");

    if ((int)block_count > GEMMA4_CAP_LAYERS)
        G4D_FAIL("block_count %u exceeds GEMMA4_CAP_LAYERS %d", block_count, GEMMA4_CAP_LAYERS);
    if ((int)head_count > GEMMA4_CAP_HEADS)
        G4D_FAIL("head_count %u exceeds GEMMA4_CAP_HEADS %d", head_count, GEMMA4_CAP_HEADS);

    cfg->n_layers     = (int)block_count;
    cfg->hidden_size  = (int)embd;
    cfg->intermediate = (int)ffn;
    cfg->n_heads      = (int)head_count;

    // vocab: prefer explicit kv if present, else fall back to the tokenizer token-array count.
    uint32_t vocab = 0;
    if (g4d_u32(data, size, "gemma4.vocab_size", &vocab) == 0 && vocab > 0) {
        cfg->vocab_size = (int)vocab;
    } else {
        g4d_array_t toks;
        if (g4d_meta(data, size, "tokenizer.ggml.tokens", G4D_T_ARRAY, &toks) == 0)
            cfg->vocab_size = (int)toks.count;
    }

    // softcap (final_logit_softcapping). Default to the Gemma-4 value if the kv is absent.
    float sc = 30.0f;
    g4d_f32(data, size, "gemma4.final_logit_softcapping", &sc);
    cfg->softcap = sc;

    // rope thetas: global (freq_base) and sliding/SWA (freq_base_swa).
    float rg = 1000000.0f, rs = 10000.0f;
    g4d_f32(data, size, "gemma4.rope.freq_base", &rg);
    g4d_f32(data, size, "gemma4.rope.freq_base_swa", &rs);
    cfg->rope_theta_global  = rg;
    cfg->rope_theta_sliding = rs;

    // Per-layer attention type + KV-head counts.
    //   gemma4.attention.sliding_window_pattern : bool[n_layers]  (1=sliding, 0=global) — primary.
    //   gemma4.attention.head_count_kv          : i32[n_layers]   — KV heads per layer; also the
    //                                                               fallback pattern source.
    g4d_array_t pat, kvarr;
    int have_pat = (g4d_meta(data, size, "gemma4.attention.sliding_window_pattern",
                             G4D_T_ARRAY, &pat) == 0 && pat.elem_type == G4D_T_BOOL);
    int have_kv  = (g4d_meta(data, size, "gemma4.attention.head_count_kv",
                             G4D_T_ARRAY, &kvarr) == 0 &&
                    (kvarr.elem_type == G4D_T_INT32 || kvarr.elem_type == G4D_T_UINT32));

    cfg->n_global = 0;
    cfg->n_kv_sliding = 0;
    cfg->n_kv_global  = 0;

    if (have_pat) {
        for (int i = 0; i < cfg->n_layers && (uint64_t)i < pat.count; i++) {
            int is_global = (pat.data[i] == 0);   // pattern: true=sliding ⇒ global when 0
            cfg->is_global[i] = (uint8_t)is_global;
            if (is_global) cfg->n_global++;
        }
    }

    if (have_kv) {
        // Use the per-layer KV head counts to (a) derive the sliding/global KV head counts and
        // (b) supply the attention pattern when the bool array is absent (sliding layers carry the
        // larger KV count, global layers the smaller).
        int kv_max = 0, kv_min = 0x7fffffff;
        for (int i = 0; i < cfg->n_layers && (uint64_t)i < kvarr.count; i++) {
            int32_t kv; memcpy(&kv, kvarr.data + (size_t)i * 4, 4);
            if (kv > kv_max) kv_max = kv;
            if (kv < kv_min) kv_min = kv;
        }
        cfg->n_kv_sliding = kv_max;                 // sliding layers: the larger KV count
        cfg->n_kv_global  = (kv_min == kv_max) ? kv_max : kv_min;

        if (!have_pat) {
            cfg->n_global = 0;
            for (int i = 0; i < cfg->n_layers && (uint64_t)i < kvarr.count; i++) {
                int32_t kv; memcpy(&kv, kvarr.data + (size_t)i * 4, 4);
                int is_global = (kv == cfg->n_kv_global) && (cfg->n_kv_global != cfg->n_kv_sliding);
                cfg->is_global[i] = (uint8_t)is_global;
                if (is_global) cfg->n_global++;
            }
        }
    }

    if (!have_pat && !have_kv)
        G4D_FAIL("no attention pattern (neither sliding_window_pattern nor head_count_kv)");

    // If we got the pattern from the bool array but no KV array, we can't know KV head counts.
    // That should not happen for real Gemma-4 GGUFs (both are present); guard loudly.
    if (cfg->n_kv_sliding == 0)
        G4D_FAIL("missing gemma4.attention.head_count_kv (KV head counts)");

    if (cfg->n_kv_sliding > GEMMA4_CAP_KV_HEADS)
        G4D_FAIL("n_kv_sliding %d exceeds GEMMA4_CAP_KV_HEADS %d",
                 cfg->n_kv_sliding, GEMMA4_CAP_KV_HEADS);

    return 0;
    #undef G4D_FAIL
}

// ── Detection: safetensors config.json → gemma4_model_config_t ──────────────────────────────────
// HF Gemma-4 config.json carries num_hidden_layers, hidden_size, intermediate_size,
// num_attention_heads, num_key_value_heads, head_dim, sliding_window_pattern, vocab_size,
// final_logit_softcapping, rope_theta / rope_local_base_freq. The NVFP4 checkpoints we load expose
// it via st::Model::config_json(); the per-layer global/sliding cadence follows the same
// 5-sliding/1-global rule (sliding_window_pattern = 6) used by Gemma-4. This is a thin scalar
// scanner mirroring nvfp4_loader.h's cfg_* helpers (kept here so this header stays standalone).
static inline int g4d_json_long(const char *j, const char *key, long *out) {
    const char *k = strstr(j, key);
    if (!k) return -1;
    const char *c = strchr(k + strlen(key), ':');
    if (!c) return -1;
    c++;
    while (*c == ' ' || *c == '\t' || *c == '\n' || *c == '\r') c++;
    char *e = NULL;
    long v = strtol(c, &e, 10);
    if (e == c) return -1;
    *out = v; return 0;
}
static inline int g4d_json_double(const char *j, const char *key, double *out) {
    const char *k = strstr(j, key);
    if (!k) return -1;
    const char *c = strchr(k + strlen(key), ':');
    if (!c) return -1;
    c++;
    while (*c == ' ' || *c == '\t' || *c == '\n' || *c == '\r') c++;
    char *e = NULL;
    double v = strtod(c, &e);
    if (e == c) return -1;
    *out = v; return 0;
}

static inline int gemma4_detect_from_config_json(const char *json,
                                                 gemma4_model_config_t *cfg,
                                                 char *err, size_t errlen) {
    #define G4D_FAIL(...) do { if (err) snprintf(err, errlen, __VA_ARGS__); return -1; } while (0)
    memset(cfg, 0, sizeof(*cfg));
    long v;
    if (g4d_json_long(json, "\"num_hidden_layers\"", &v) != 0) G4D_FAIL("config: num_hidden_layers");
    cfg->n_layers = (int)v;
    if (g4d_json_long(json, "\"hidden_size\"", &v) != 0) G4D_FAIL("config: hidden_size");
    cfg->hidden_size = (int)v;
    if (g4d_json_long(json, "\"intermediate_size\"", &v) != 0) G4D_FAIL("config: intermediate_size");
    cfg->intermediate = (int)v;
    if (g4d_json_long(json, "\"num_attention_heads\"", &v) != 0) G4D_FAIL("config: num_attention_heads");
    cfg->n_heads = (int)v;
    if (g4d_json_long(json, "\"num_key_value_heads\"", &v) == 0) cfg->n_kv_sliding = (int)v;
    if (g4d_json_long(json, "\"vocab_size\"", &v) == 0) cfg->vocab_size = (int)v;

    double d;
    cfg->softcap = (g4d_json_double(json, "\"final_logit_softcapping\"", &d) == 0) ? (float)d : 30.0f;
    cfg->rope_theta_global  = (g4d_json_double(json, "\"rope_theta\"", &d) == 0) ? (float)d : 1000000.0f;
    cfg->rope_theta_sliding = (g4d_json_double(json, "\"rope_local_base_freq\"", &d) == 0) ? (float)d : 10000.0f;

    // sliding_window_pattern is the period P (every Pth layer is global); default 6 for Gemma-4.
    long P = 6;
    g4d_json_long(json, "\"sliding_window_pattern\"", &P);
    if (P <= 0) P = 6;
    cfg->n_global = 0;
    for (int i = 0; i < cfg->n_layers && i < GEMMA4_CAP_LAYERS; i++) {
        int is_global = ((i + 1) % P == 0);
        cfg->is_global[i] = (uint8_t)is_global;
        if (is_global) cfg->n_global++;
    }
    // HF Gemma-4 uses the same KV head count for sliding and global layers; n_kv_global == n_kv_sliding.
    cfg->n_kv_global = cfg->n_kv_sliding;

    if (cfg->n_layers <= 0 || cfg->n_layers > GEMMA4_CAP_LAYERS)
        G4D_FAIL("config: n_layers %d out of range (cap %d)", cfg->n_layers, GEMMA4_CAP_LAYERS);
    return 0;
    #undef G4D_FAIL
}

#ifdef __cplusplus
}
#endif

#endif // GEMMA4_DETECT_H
