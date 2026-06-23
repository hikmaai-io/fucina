// gemma4_e4b.h — runtime configuration + detection for Gemma-4-E4B.
//
// E4B is the on-device member of the Gemma-4 family: Gemma-3-style attention
// (5 sliding : 1 full, q/k RMSNorm, logit softcap) PLUS two memory-shaping
// mechanisms that the 12B/31B dense models do not have:
//
//   • Per-Layer Embeddings (PLE): a second, large embedding table
//     `embed_tokens_per_layer` [vocab, n_layers*ple_dim] feeds every decoder
//     layer a 256-d per-token vector through an input gate + projection. The
//     PLE table is the single biggest weight (5.6 GB BF16 — larger than the
//     main embedding), which is why it is the prime target for the FP8 index
//     (see e4b_ple_fp8.cuh): FP8 E4M3 halves it to ~2.7 GB.
//
//   • KV-cache sharing: the last `n_kv_shared_layers` layers reuse the KV
//     projected by an earlier layer instead of projecting their own, cutting
//     KV-cache footprint and the per-token K/V GEMV count.
//
// Per the project rule (runtime-model-detection), nothing here is a compile
// flag: the architecture and every dimension are read from the checkpoint
// metadata (config.json for safetensors, GGUF KV for GGUF). This header parses
// config.json; the GGUF path reads the equivalent `gemma4.*` keys in-engine.
//
// Header-only, host-side, no CUDA — include from the engine/loader TU.
#ifndef FUCINA_GEMMA4_E4B_H
#define FUCINA_GEMMA4_E4B_H

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

namespace e4b {

// One decoder layer's attention flavour (matches gemma4 layer_types).
enum class Attn { SLIDING = 0, FULL = 1 };

struct Config {
    // Core transformer dims (text_config).
    int  hidden_size          = 0;     // 2560
    int  intermediate_size    = 0;     // 10240 (GeGLU)
    int  n_layers             = 0;     // 42
    int  n_heads              = 0;     // 8
    int  n_kv_heads           = 0;     // 2
    int  head_dim             = 0;     // 256 (sliding)
    int  global_head_dim      = 0;     // 512 (full attn)
    int  sliding_window       = 0;     // 512
    int  vocab_size           = 0;     // 262144
    int  max_position         = 0;     // 131072

    // Per-Layer Embeddings.
    int  ple_dim              = 0;     // hidden_size_per_layer_input = 256
    int  ple_vocab            = 0;     // vocab_size_per_layer_input  = 262144
    // PLE table width per token = n_layers * ple_dim (10752 for E4B).
    int  ple_width()    const { return n_layers * ple_dim; }

    // KV-cache sharing: layers [n_layers - n_kv_shared_layers, n_layers) reuse
    // an earlier layer's KV instead of projecting their own.
    int  n_kv_shared_layers   = 0;     // 18

    // Numerics.
    float final_logit_softcap = 0.f;   // 30.0
    float rms_eps             = 0.f;   // 1e-6
    float rope_theta_sliding  = 0.f;   // 10000
    float rope_theta_full     = 0.f;   // 1e6
    float rope_partial_full   = 0.f;   // 0.25 (proportional rope on full-attn layers)
    bool  tie_word_embeddings = false; // true

    std::vector<Attn> layer_types;     // size n_layers

    bool valid() const {
        return hidden_size > 0 && n_layers > 0 && ple_dim > 0 &&
               (int)layer_types.size() == n_layers && ple_vocab > 0;
    }
    // First index of the KV-shared tail (== n_layers when sharing is off).
    int  kv_share_start() const { return n_layers - n_kv_shared_layers; }
    bool layer_shares_kv(int l) const { return l >= kv_share_start(); }
};

// ── tiny config.json scalar scanner ───────────────────────────────────────
// config.json is machine-generated and regular. We only need scalars by key
// inside a named object ("text_config"); good enough without a full parser.
namespace detail {

// Find the body span [b,e) of object "name" (the matching braces). Returns
// false if absent. Searches for `"name"` then the next '{' and brace-matches.
inline bool object_span(const std::string& j, const char* name,
                        size_t& b, size_t& e, size_t from = 0) {
    std::string key = std::string("\"") + name + "\"";
    size_t k = from;
    size_t open = std::string::npos;
    while ((k = j.find(key, k)) != std::string::npos) {
        // require `"key" : {` with only whitespace between — so a same-named
        // string *value* inside an array (e.g. layer_types) is not matched.
        size_t p = k + key.size();
        while (p < j.size() && (j[p]==' '||j[p]=='\t'||j[p]=='\n'||j[p]=='\r')) ++p;
        if (p < j.size() && j[p] == ':') {
            ++p;
            while (p < j.size() && (j[p]==' '||j[p]=='\t'||j[p]=='\n'||j[p]=='\r')) ++p;
            if (p < j.size() && j[p] == '{') { open = p; break; }
        }
        k += key.size();
    }
    if (open == std::string::npos) return false;
    int depth = 0; bool instr = false;
    for (size_t i = open; i < j.size(); ++i) {
        char c = j[i];
        if (instr) { if (c == '\\') ++i; else if (c == '"') instr = false; continue; }
        if (c == '"') instr = true;
        else if (c == '{') ++depth;
        else if (c == '}') { if (--depth == 0) { b = open + 1; e = i; return true; } }
    }
    return false;
}

// Scan [b,e) of j for `"key"` and return the raw token after ':' in [vb,ve).
inline bool raw_value(const std::string& j, size_t b, size_t e,
                     const char* keyname, size_t& vb, size_t& ve) {
    std::string key = std::string("\"") + keyname + "\"";
    size_t k = b;
    while ((k = j.find(key, k)) != std::string::npos && k < e) {
        size_t colon = j.find(':', k + key.size());
        if (colon == std::string::npos || colon >= e) return false;
        size_t p = colon + 1;
        while (p < e && (j[p]==' '||j[p]=='\t'||j[p]=='\n'||j[p]=='\r')) ++p;
        vb = p;
        // token ends at , } ] or whitespace (for bare scalars); strings handled by caller
        size_t q = p;
        if (q < e && j[q] == '"') { // string
            ++q; while (q < e && j[q] != '"') { if (j[q]=='\\') ++q; ++q; }
            ve = (q < e) ? q + 1 : e;
        } else {
            while (q < e && j[q]!=',' && j[q]!='}' && j[q]!=']' &&
                   j[q]!=' ' && j[q]!='\n' && j[q]!='\t' && j[q]!='\r') ++q;
            ve = q;
        }
        return true;
    }
    return false;
}

inline bool get_int(const std::string& j, size_t b, size_t e, const char* key, int& out) {
    size_t vb, ve; if (!raw_value(j, b, e, key, vb, ve)) return false;
    std::string t = j.substr(vb, ve - vb);
    if (t == "null") return false;
    out = (int)strtol(t.c_str(), nullptr, 10); return true;
}
inline bool get_float(const std::string& j, size_t b, size_t e, const char* key, float& out) {
    size_t vb, ve; if (!raw_value(j, b, e, key, vb, ve)) return false;
    std::string t = j.substr(vb, ve - vb);
    if (t == "null") return false;
    out = strtof(t.c_str(), nullptr); return true;
}
inline bool get_bool(const std::string& j, size_t b, size_t e, const char* key, bool& out) {
    size_t vb, ve; if (!raw_value(j, b, e, key, vb, ve)) return false;
    out = j.compare(vb, 4, "true") == 0; return true;
}

} // namespace detail

// Does this config.json describe a Gemma-4-E4B (PLE-bearing) text model?
// Distinguishing marks vs dense 12B/31B: a "text_config" with both
// hidden_size_per_layer_input and num_kv_shared_layers present and non-null.
inline bool is_e4b(const std::string& config_json) {
    size_t b, e;
    const std::string* j = &config_json;
    std::string scoped;
    if (detail::object_span(config_json, "text_config", b, e)) {
        // ok, scan within text_config
    } else {
        b = 0; e = config_json.size();   // flat config (GGUF-derived dumps)
    }
    int ple = 0, shared = 0;
    bool has_ple = detail::get_int(*j, b, e, "hidden_size_per_layer_input", ple) && ple > 0;
    bool has_shared = detail::get_int(*j, b, e, "num_kv_shared_layers", shared);
    (void)scoped;
    return has_ple && has_shared;
}

// Parse the full E4B config from config.json. Returns false if not E4B or on
// missing required fields (err set). layer_types is filled from the JSON array
// if present, else synthesized as a 5-sliding:1-full pattern.
inline bool parse(const std::string& config_json, Config& c, std::string& err) {
    size_t b, e;
    if (!detail::object_span(config_json, "text_config", b, e)) {
        b = 0; e = config_json.size();
    }
    using namespace detail;
    if (!get_int(config_json, b, e, "hidden_size", c.hidden_size)) { err = "no hidden_size"; return false; }
    get_int  (config_json, b, e, "intermediate_size", c.intermediate_size);
    if (!get_int(config_json, b, e, "num_hidden_layers", c.n_layers)) { err = "no num_hidden_layers"; return false; }
    get_int  (config_json, b, e, "num_attention_heads", c.n_heads);
    get_int  (config_json, b, e, "num_key_value_heads", c.n_kv_heads);
    get_int  (config_json, b, e, "head_dim", c.head_dim);
    get_int  (config_json, b, e, "global_head_dim", c.global_head_dim);
    get_int  (config_json, b, e, "sliding_window", c.sliding_window);
    get_int  (config_json, b, e, "vocab_size", c.vocab_size);
    get_int  (config_json, b, e, "max_position_embeddings", c.max_position);
    get_int  (config_json, b, e, "hidden_size_per_layer_input", c.ple_dim);
    get_int  (config_json, b, e, "vocab_size_per_layer_input", c.ple_vocab);
    get_int  (config_json, b, e, "num_kv_shared_layers", c.n_kv_shared_layers);
    get_float(config_json, b, e, "final_logit_softcapping", c.final_logit_softcap);
    get_float(config_json, b, e, "rms_norm_eps", c.rms_eps);
    get_bool (config_json, b, e, "tie_word_embeddings", c.tie_word_embeddings);

    // RoPE thetas live in a nested rope_parameters{sliding_attention{}, full_attention{}}.
    // Scope the sub-object search INSIDE rope_parameters so the "sliding_attention"
    // string entries in the layer_types array are not mistaken for the rope object.
    size_t rpb, rpe;
    if (object_span(config_json, "rope_parameters", rpb, rpe)) {
        size_t rb, re;
        if (object_span(config_json, "sliding_attention", rb, re, rpb) && re <= rpe)
            get_float(config_json, rb, re, "rope_theta", c.rope_theta_sliding);
        if (object_span(config_json, "full_attention", rb, re, rpb) && re <= rpe) {
            get_float(config_json, rb, re, "rope_theta", c.rope_theta_full);
            get_float(config_json, rb, re, "partial_rotary_factor", c.rope_partial_full);
        }
    }

    if (c.ple_dim <= 0 || c.ple_vocab <= 0) { err = "not an E4B config (no PLE)"; return false; }

    // layer_types: parse the JSON array of "sliding_attention"/"full_attention".
    c.layer_types.clear();
    size_t lt = config_json.find("\"layer_types\"", b);
    if (lt != std::string::npos && lt < e) {
        size_t lb = config_json.find('[', lt), le = config_json.find(']', lb);
        if (lb != std::string::npos && le != std::string::npos) {
            size_t p = lb;
            while ((p = config_json.find('"', p)) != std::string::npos && p < le) {
                size_t q = config_json.find('"', p + 1);
                std::string s = config_json.substr(p + 1, q - p - 1);
                if (s == "full_attention")     c.layer_types.push_back(Attn::FULL);
                else if (s == "sliding_attention") c.layer_types.push_back(Attn::SLIDING);
                p = q + 1;
            }
        }
    }
    if ((int)c.layer_types.size() != c.n_layers) {
        // Synthesize the canonical 5:1 pattern (full at every 6th, i.e. idx%6==5).
        c.layer_types.assign(c.n_layers, Attn::SLIDING);
        for (int i = 5; i < c.n_layers; i += 6) c.layer_types[i] = Attn::FULL;
    }
    return c.valid();
}

// safetensors tensor-name builders (model.language_model.* prefix).
struct Names {
    std::string prefix = "model.language_model.";
    std::string embed_tokens()        const { return prefix + "embed_tokens.weight"; }
    std::string embed_per_layer()     const { return prefix + "embed_tokens_per_layer.weight"; }
    std::string per_layer_model_proj()const { return prefix + "per_layer_model_projection.weight"; }
    std::string per_layer_proj_norm() const { return prefix + "per_layer_projection_norm.weight"; }
    std::string final_norm()          const { return prefix + "norm.weight"; }
    std::string L(int i, const char* s) const {
        return prefix + "layers." + std::to_string(i) + "." + s;
    }
};

} // namespace e4b

#endif // FUCINA_GEMMA4_E4B_H
