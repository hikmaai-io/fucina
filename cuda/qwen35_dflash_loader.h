// ABOUTME: Config-derived, bounds-checked loader schema for the z-lab Qwen3.5-9B-DFlash draft.
// ABOUTME: Validates every tensor rank/shape/dtype against config geometry before any CUDA alloc.
//
// The DFlash draft is a small (config.num_hidden_layers) transformer whose K/V for the context is
// cross-projected from the TARGET model's hidden states (see qwen3_dflash.py). This header parses
// the draft's config.json into a fully symbolic Geometry (never checkpoint-specific constants) and
// validates the safetensors tensor set against it. It is the hostile-input firewall: a mismatched
// or adversarial checkpoint is rejected with a precise reason BEFORE the engine allocates or
// uploads anything to the GPU. Pure host C++ on top of st::Model (safetensors.h); no CUDA here.
//
// Exact public schema (z-lab/Qwen3.5-9B-DFlash, config.json rev 5fc3b3d4), all symbols derived:
//   architectures=["DFlashDraftModel"], model_type="qwen3"
//   hidden_size H, intermediate_size I, num_hidden_layers L
//   num_attention_heads NQ, num_key_value_heads NKV, head_dim HD
//   vocab_size V, dflash_config.mask_token_id in [0,V)
//   dflash_config.target_layer_ids (num target features F); fc input = target_hidden * F
//   layer_types[L] in {sliding_attention, full_attention}; sliding_window > 0 if any sliding
// Per layer l the required tensors (all BF16 in the public checkpoint, but dtype is validated as a
// float weight class, not pinned to BF16, so FP8/NVFP4 variants remain loadable via the shared
// WeightRef/quant loaders):
//   layers.l.input_layernorm.weight [H], post_attention_layernorm.weight [H]
//   layers.l.self_attn.q_proj.weight [NQ*HD, H], k_proj/v_proj.weight [NKV*HD, H]
//   layers.l.self_attn.o_proj.weight [H, NQ*HD]
//   layers.l.self_attn.q_norm.weight [HD], k_norm.weight [HD]
//   layers.l.mlp.gate_proj.weight [I, H], up_proj.weight [I, H], down_proj.weight [H, I]
//   plus global: hidden_norm.weight [H], norm.weight [H], fc.weight [H, target_hidden*F]
// Optional reduced vocabulary: a d2t/draft_id_to_target_id [draft_vocab] integer map; each entry
// (as base+index) must stay within the target vocab. Absent => draft and target vocab match.
#ifndef FUCINA_QWEN35_DFLASH_LOADER_H
#define FUCINA_QWEN35_DFLASH_LOADER_H

#include "safetensors.h"
#include <string>
#include <cstring>
#include <cstdlib>
#include <vector>

namespace qwen35dflash {

enum LayerAttn { ATTN_FULL = 0, ATTN_SLIDING = 1 };

struct Geometry {
    int H = 0;               // hidden_size
    int I = 0;               // intermediate_size
    int L = 0;               // num_hidden_layers
    int NQ = 0;              // num_attention_heads
    int NKV = 0;             // num_key_value_heads
    int HD = 0;              // head_dim
    int V = 0;               // vocab_size
    int mask_token_id = -1;  // dflash_config.mask_token_id, in [0,V)
    int block_size = 0;      // dflash_config.block_size (drafted block length; informational)
    int target_hidden = 0;   // target hidden width feeding fc (defaults to H if absent)
    int num_target_features = 0;  // len(target_layer_ids); fc input = target_hidden * F
    int sliding_window = 0;  // > 0 when any layer is sliding
    bool use_aux_hidden = true;   // whether fc/aux-hidden path is present
    std::vector<uint8_t> layer_attn;  // per-layer LayerAttn, size L
    int draft_vocab = 0;     // reduced draft vocab if a d2t map is present, else == V
    bool has_d2t = false;

    int q_dim() const { return NQ * HD; }
    int kv_dim() const { return NKV * HD; }
    int fc_in() const { return target_hidden * num_target_features; }
};

// ── minimal config.json scanners (reused pattern from qwen35_fp8_loader.h) ──
inline bool cfg_find(const std::string& j, const char* key, size_t& pos) {
    size_t k = j.find(key);
    if (k == std::string::npos) return false;
    size_t c = j.find(':', k + std::strlen(key));
    if (c == std::string::npos) return false;
    pos = c + 1;
    while (pos < j.size() && (j[pos]==' '||j[pos]=='\t'||j[pos]=='\n'||j[pos]=='\r')) pos++;
    return true;
}
inline bool cfg_str(const std::string& j, const char* key, std::string& out) {
    size_t p; if (!cfg_find(j, key, p) || p >= j.size() || j[p] != '"') return false;
    size_t e = j.find('"', p + 1); if (e == std::string::npos) return false;
    out = j.substr(p + 1, e - p - 1); return true;
}
inline bool cfg_int(const std::string& j, const char* key, long& out) {
    size_t p; if (!cfg_find(j, key, p)) return false;
    char* end = nullptr; long v = std::strtol(j.c_str() + p, &end, 10);
    if (end == j.c_str() + p) return false;
    out = v; return true;
}
// Count elements of an int array value "key": [a, b, c] — used for target_layer_ids and to read
// the dflash_config sub-object's mask_token_id/block_size (searched inside the whole json, which is
// safe because those keys are unique in this schema).
inline int cfg_int_array_len(const std::string& j, const char* key) {
    size_t p; if (!cfg_find(j, key, p) || p >= j.size() || j[p] != '[') return -1;
    size_t e = j.find(']', p); if (e == std::string::npos) return -1;
    std::string inner = j.substr(p + 1, e - p - 1);
    if (inner.find_first_not_of(" \t\n\r") == std::string::npos) return 0;
    int commas = 0; for (char c : inner) if (c == ',') commas++;
    return commas + 1;
}

// A weight tensor must be a float weight class (BF16/F16/F32 or FP8/FP4 quantized). Norm/scale
// tensors are BF16/F16/F32. We validate the class, not a single pinned dtype, so quantized draft
// variants remain loadable through the shared WeightRef/FP8/NVFP4 paths.
inline bool is_float_weight(st::Dtype d) {
    return d==st::Dtype::BF16 || d==st::Dtype::F16 || d==st::Dtype::F32 ||
           d==st::Dtype::F8_E4M3 || d==st::Dtype::F8_E5M2 || d==st::Dtype::F4_E2M1 ||
           d==st::Dtype::U8;  // packed fp4 producers tag as U8
}
inline bool is_norm_dtype(st::Dtype d) {
    return d==st::Dtype::BF16 || d==st::Dtype::F16 || d==st::Dtype::F32;
}

inline std::string lkey(int l, const char* suffix) {
    return "layers." + std::to_string(l) + "." + suffix;
}

// Parse + validate the draft config into Geometry. Rejects malformed/hostile geometry with a
// precise reason. No tensor access here (pure config); tensor validation is validate_tensors().
inline bool parse_config(const std::string& cj, Geometry& g, std::string& err) {
    if (cj.empty()) { err = "no config.json next to draft checkpoint"; return false; }
    std::string arch, mtype;
    cfg_str(cj, "\"architectures\"", arch);   // grabs the first array string
    cfg_str(cj, "\"model_type\"", mtype);
    if (cj.find("DFlashDraftModel") == std::string::npos) {
        err = "config architectures is not DFlashDraftModel"; return false;
    }
    if (mtype.find("qwen3") == std::string::npos) {
        err = "config model_type is not a qwen3 draft"; return false;
    }
    long v = 0;
    auto need = [&](const char* k, int& dst, long lo, long hi) -> bool {
        if (!cfg_int(cj, k, v)) { err = std::string("config missing ") + k; return false; }
        if (v < lo || v > hi) { err = std::string("config ") + k + " out of range"; return false; }
        dst = (int)v; return true;
    };
    if (!need("\"hidden_size\"",         g.H,  1, 1<<20)) return false;
    if (!need("\"intermediate_size\"",   g.I,  1, 1<<22)) return false;
    if (!need("\"num_hidden_layers\"",   g.L,  1, 512))   return false;
    if (!need("\"num_attention_heads\"", g.NQ, 1, 4096))  return false;
    if (!need("\"num_key_value_heads\"", g.NKV,1, 4096))  return false;
    if (!need("\"head_dim\"",            g.HD, 1, 8192))  return false;
    if (!need("\"vocab_size\"",          g.V,  1, 1<<24)) return false;
    if (g.NQ % g.NKV != 0) { err = "num_attention_heads not a multiple of num_key_value_heads"; return false; }
    // dflash_config.mask_token_id (required for parallel drafting).
    long mt = -1;
    if (!cfg_int(cj, "\"mask_token_id\"", mt)) { err = "config missing dflash_config.mask_token_id"; return false; }
    if (mt < 0 || mt >= g.V) { err = "mask_token_id out of [0,vocab)"; return false; }
    g.mask_token_id = (int)mt;
    long bs = 0; cfg_int(cj, "\"block_size\"", bs); g.block_size = (bs > 0) ? (int)bs : 0;
    // target features from target_layer_ids (fallback 1 => plain hidden passthrough).
    int F = cfg_int_array_len(cj, "\"target_layer_ids\"");
    if (F < 0) F = cfg_int_array_len(cj, "\"layer_ids\"");
    g.num_target_features = (F > 0) ? F : 1;
    long th = 0;
    g.target_hidden = cfg_int(cj, "\"target_hidden_size\"", th) && th > 0 ? (int)th : g.H;
    // layer_types[L]: parse each entry; count must equal L.
    g.layer_attn.assign(g.L, ATTN_FULL);
    {
        size_t p;
        bool any_sliding = false;
        if (cfg_find(cj, "\"layer_types\"", p) && p < cj.size() && cj[p] == '[') {
            size_t e = cj.find(']', p);
            if (e == std::string::npos) { err = "malformed layer_types array"; return false; }
            std::string inner = cj.substr(p + 1, e - p - 1);
            int idx = 0; size_t q = 0;
            while (idx < g.L) {
                size_t qs = inner.find('"', q); if (qs == std::string::npos) break;
                size_t qe = inner.find('"', qs + 1); if (qe == std::string::npos) break;
                std::string lt = inner.substr(qs + 1, qe - qs - 1);
                if (lt == "sliding_attention") { g.layer_attn[idx] = ATTN_SLIDING; any_sliding = true; }
                else if (lt == "full_attention") { g.layer_attn[idx] = ATTN_FULL; }
                else { err = "unknown layer_type '" + lt + "'"; return false; }
                q = qe + 1; idx++;
            }
            if (idx != g.L) { err = "layer_types length does not match num_hidden_layers"; return false; }
        }
        if (any_sliding) {
            long sw = 0;
            if (!cfg_int(cj, "\"sliding_window\"", sw) || sw <= 0) {
                err = "sliding layers present but sliding_window missing/invalid"; return false;
            }
            g.sliding_window = (int)sw;
        }
    }
    g.draft_vocab = g.V;   // updated by validate_tensors if a d2t map is present
    return true;
}

// Validate the safetensors tensor set against Geometry. Returns false + precise err on the FIRST
// violation. MUST be called before any CUDA allocation. Also detects an optional reduced-vocab
// d2t map and range-checks it against the target vocab.
inline bool validate_tensors(const st::Model& m, Geometry& g, int target_vocab, std::string& err) {
    auto check = [&](const std::string& name, bool weight_class,
                     std::initializer_list<int> dims) -> bool {
        const st::Tensor* t = m.find(name);
        if (!t) { err = "missing tensor " + name; return false; }
        if (weight_class ? !is_float_weight(t->dtype) : !is_norm_dtype(t->dtype)) {
            err = "tensor " + name + " has unexpected dtype"; return false;
        }
        if ((int)t->shape.size() != (int)dims.size()) {
            err = "tensor " + name + " wrong rank"; return false;
        }
        int i = 0;
        for (int d : dims) {
            if (t->shape[i] != (int64_t)d) {
                err = "tensor " + name + " dim " + std::to_string(i) + " mismatch"; return false;
            }
            i++;
        }
        return true;
    };
    // Global tensors.
    if (!check("hidden_norm.weight", false, {g.H})) return false;
    if (!check("norm.weight",        false, {g.H})) return false;
    // fc is present only when aux-hidden is used; if absent, treat as passthrough draft.
    g.use_aux_hidden = m.has("fc.weight");
    if (g.use_aux_hidden && !check("fc.weight", true, {g.H, g.fc_in()})) return false;
    // Per-layer tensors.
    for (int l = 0; l < g.L; l++) {
        if (!check(lkey(l, "input_layernorm.weight"),          false, {g.H})) return false;
        if (!check(lkey(l, "post_attention_layernorm.weight"), false, {g.H})) return false;
        if (!check(lkey(l, "self_attn.q_proj.weight"), true, {g.q_dim(), g.H})) return false;
        if (!check(lkey(l, "self_attn.k_proj.weight"), true, {g.kv_dim(), g.H})) return false;
        if (!check(lkey(l, "self_attn.v_proj.weight"), true, {g.kv_dim(), g.H})) return false;
        if (!check(lkey(l, "self_attn.o_proj.weight"), true, {g.H, g.q_dim()})) return false;
        if (!check(lkey(l, "self_attn.q_norm.weight"), false, {g.HD})) return false;
        if (!check(lkey(l, "self_attn.k_norm.weight"), false, {g.HD})) return false;
        if (!check(lkey(l, "mlp.gate_proj.weight"), true, {g.I, g.H})) return false;
        if (!check(lkey(l, "mlp.up_proj.weight"),   true, {g.I, g.H})) return false;
        if (!check(lkey(l, "mlp.down_proj.weight"), true, {g.H, g.I})) return false;
    }
    // Optional reduced-vocab d2t map.
    const char* d2t_names[] = {"d2t", "draft_id_to_target_id"};
    const st::Tensor* d2t = nullptr; std::string d2t_name;
    for (const char* nm : d2t_names) { if ((d2t = m.find(nm))) { d2t_name = nm; break; } }
    if (d2t) {
        if (d2t->shape.size() != 1 || d2t->shape[0] <= 0) { err = "d2t map must be rank-1 non-empty"; return false; }
        if (d2t->dtype != st::Dtype::I64 && d2t->dtype != st::Dtype::I32) {
            err = "d2t map must be integer"; return false;
        }
        g.draft_vocab = (int)d2t->shape[0];
        g.has_d2t = true;
        if (g.draft_vocab > g.V) { err = "d2t draft_vocab exceeds config vocab_size"; return false; }
        // Range-check each mapped target id (draft->target is base+index, per qwen3_dflash.py).
        int tv = target_vocab > 0 ? target_vocab : g.V;
        size_t elt = (d2t->dtype == st::Dtype::I64) ? 8 : 4;
        if (d2t->nbytes < (size_t)g.draft_vocab * elt) { err = "d2t map truncated"; return false; }
        for (int i = 0; i < g.draft_vocab; i++) {
            int64_t delta = (d2t->dtype == st::Dtype::I64)
                ? ((const int64_t*)d2t->data)[i] : (int64_t)((const int32_t*)d2t->data)[i];
            int64_t target_id = (int64_t)i + delta;
            if (target_id < 0 || target_id >= tv) { err = "d2t maps token out of target vocab"; return false; }
        }
    }
    // Mask token embedding is provided by the shared TARGET embedding at mask_token_id; the draft
    // does not ship its own embed_tokens in this checkpoint. Reject a stray embed of wrong shape.
    if (const st::Tensor* e = m.find("embed_tokens.weight")) {
        if (e->shape.size() != 2 || e->shape[1] != g.H) { err = "embed_tokens.weight wrong shape"; return false; }
    }
    return true;
}

} // namespace qwen35dflash

#endif // FUCINA_QWEN35_DFLASH_LOADER_H
