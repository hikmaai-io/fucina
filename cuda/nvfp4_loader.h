// NVFP4 safetensors → engine name mapping and per-projection key resolution.
//
// Two producer ecosystems emit NVFP4 with DIFFERENT tensor key names (verified against real
// checkpoints — see the nvfp4-safetensors-schema memory):
//
//   ModelOpt (nvidia/*-FP4):           <p>.weight        <p>.weight_scale  <p>.weight_scale_2     <p>.input_scale
//   compressed-tensors (RedHatAI/*):   <p>.weight_packed <p>.weight_scale  <p>.weight_global_scale <p>.input_global_scale
//
// Layout otherwise identical: packed U8 [out, in/2] (low nibble=even k), E4M3 linear block
// scales [out, in/16] (group 16), FP32 global = amax/(6*448). Multimodal Gemma 4 nests the LM
// under `model.language_model.layers.*` (text-only Llama/Qwen use `model.layers.*`). The vision/
// audio towers, lm_head, embeddings and all norms stay BF16/F32 and are NOT in the quant group.
//
// This header resolves, for a given (layer, projection), the three keys we actually consume
// (packed weight, block scale, global scale — input_scale is ignored: activations are quantized
// dynamically at runtime). Pure host string/lookup logic — testable without CUDA.
#ifndef FUCINA_NVFP4_LOADER_H
#define FUCINA_NVFP4_LOADER_H

#include "safetensors.h"
#include <string>
#include <cstring>
#include <cstdlib>
#include <vector>

namespace nvfp4ld {

enum class Naming { MODELOPT, COMPRESSED };

// The 7 dense projections fucina drives, by their HF module suffix under a layer.
enum Proj { P_Q, P_K, P_V, P_O, P_GATE, P_UP, P_DOWN, P_COUNT };
inline const char* proj_suffix(int p) {
    switch (p) {
        case P_Q:    return "self_attn.q_proj";
        case P_K:    return "self_attn.k_proj";
        case P_V:    return "self_attn.v_proj";
        case P_O:    return "self_attn.o_proj";
        case P_GATE: return "mlp.gate_proj";
        case P_UP:   return "mlp.up_proj";
        case P_DOWN: return "mlp.down_proj";
    }
    return "";
}

struct Layout {
    Naming      naming = Naming::MODELOPT;
    std::string layer_prefix;   // e.g. "model.language_model.layers." (trailing dot)
    std::string embed_key;      // BF16 embeddings
    std::string lmhead_key;     // BF16 LM head ("" if tied → reuse embed)
    std::string final_norm_key; // BF16 final norm
    int         n_layers = 0;
    bool        tie_word_embeddings = false;   // from config.json (authoritative)
    std::vector<std::string> ignore;           // modules left BF16 (config.json ignore list)
};

// The three NVFP4 tensors we consume for one projection.
struct ProjKeys { std::string packed, scale, gscale; };

// Normalize the per-tensor global scalar into a uniform DECODE MULTIPLIER such that
//   real = e2m1(nibble) * e4m3(block_scale) * global_mul   (always a multiply downstream).
// VERIFIED against the real RedHatAI/gemma-4-12B-it-NVFP4 file (q_proj global=7392, down=12928):
//   • compressed-tensors stores the LARGE reciprocal weight_global_scale = (6*448)/amax,
//     and reconstruction DIVIDES by it → the multiplier is 1/weight_global_scale.
//   • ModelOpt stores the SMALL weight_scale_2 = amax/(6*448) and MULTIPLIES → use as-is.
//
// The producer is detected upstream by the global-scale tensor KEY:
//   weight_global_scale → compressed-tensors (DIVIDE); weight_scale_2 → ModelOpt (MULTIPLY).
// INVARIANT: every downstream consumer assumes real = e2m1 * e4m3 * global_mul. If a third
// producer with a different convention appears, extend BOTH the key detection (Naming) and the
// branch below; do not let a new key fall through to the ModelOpt (multiply) default silently.
inline float global_mul(Naming naming, float raw) {
    return (naming == Naming::COMPRESSED) ? (raw != 0.f ? 1.0f / raw : 0.f) : raw;
}

// Suffix test.
inline bool ends_with(const std::string& s, const char* suf) {
    size_t n = std::strlen(suf);
    return s.size() >= n && memcmp(s.data() + s.size() - n, suf, n) == 0;
}
inline bool starts_with(const std::string& s, const std::string& pre) {
    return s.size() >= pre.size() && memcmp(s.data(), pre.data(), pre.size()) == 0;
}

// Resolve the per-projection keys for a layer under a detected layout.
inline ProjKeys proj_keys(const Layout& L, int layer, int p) {
    std::string base = L.layer_prefix + std::to_string(layer) + "." + proj_suffix(p) + ".";
    ProjKeys k;
    if (L.naming == Naming::COMPRESSED) {
        k.packed = base + "weight_packed";
        k.scale  = base + "weight_scale";
        k.gscale = base + "weight_global_scale";
    } else {
        k.packed = base + "weight";
        k.scale  = base + "weight_scale";
        k.gscale = base + "weight_scale_2";
    }
    return k;
}

// ── config.json quantization detection (rewritten from the agent's nvfp4_safetensors loader) ──
// Tensor-probing in detect() already pins the naming, but config.json gives two things the
// tensors don't: the authoritative `tie_word_embeddings` flag (so we know whether lm_head is a
// real tensor or aliases embed_tokens) and the `ignore`/`exclude_modules` list (which Linears
// were left BF16 — e.g. per-layer exclusions). Keys are unique enough to scan by substring,
// which is also nesting-agnostic (compressed-tensors nests under `quantization_config`).
struct QuantConfig {
    bool        is_nvfp4 = false;
    Naming      naming = Naming::MODELOPT;
    int         group_size = 16;
    bool        tie_word_embeddings = false;
    std::vector<std::string> ignore;   // concrete module names / trailing-'*' globs
};

// value just after `"key"` `:` — string, integer, or bool. Returns false if key absent.
inline bool cfg_find_colon(const std::string& j, const char* key, size_t& pos) {
    size_t k = j.find(key);
    if (k == std::string::npos) return false;
    size_t c = j.find(':', k + std::strlen(key));
    if (c == std::string::npos) return false;
    pos = c + 1;
    while (pos < j.size() && (j[pos]==' '||j[pos]=='\t'||j[pos]=='\n'||j[pos]=='\r')) pos++;
    return true;
}
inline bool cfg_str(const std::string& j, const char* key, std::string& out) {
    size_t p; if (!cfg_find_colon(j, key, p) || p >= j.size() || j[p] != '"') return false;
    size_t e = j.find('"', p + 1); if (e == std::string::npos) return false;
    out = j.substr(p + 1, e - p - 1); return true;
}
inline bool cfg_int(const std::string& j, const char* key, long& out) {
    size_t p; if (!cfg_find_colon(j, key, p)) return false;
    char* end = nullptr; long v = std::strtol(j.c_str() + p, &end, 10);
    if (end == j.c_str() + p) return false;
    out = v; return true;
}
inline bool cfg_bool(const std::string& j, const char* key, bool& out) {
    size_t p; if (!cfg_find_colon(j, key, p)) return false;
    out = (j.compare(p, 4, "true") == 0); return true;
}

inline bool parse_config(const std::string& j, QuantConfig& c) {
    std::string fmt, qm, qalgo;
    cfg_str(j, "\"format\"", fmt);
    cfg_str(j, "\"quant_method\"", qm);
    cfg_str(j, "\"quant_algo\"", qalgo);
    if (fmt == "nvfp4-pack-quantized" ||
        (qm == "compressed-tensors" && j.find("nvfp4-pack-quantized") != std::string::npos)) {
        c.is_nvfp4 = true; c.naming = Naming::COMPRESSED;
    } else if (qalgo.find("NVFP4") != std::string::npos ||
               (qm == "modelopt" && j.find("NVFP4") != std::string::npos)) {
        c.is_nvfp4 = true; c.naming = Naming::MODELOPT;
    } else {
        return false;
    }
    long g; if (cfg_int(j, "\"group_size\"", g) && g > 0) c.group_size = (int)g;
    cfg_bool(j, "\"tie_word_embeddings\"", c.tie_word_embeddings);
    // ignore / exclude_modules : ["a", "b", ...]
    size_t a = j.find("\"ignore\"");
    if (a == std::string::npos) a = j.find("\"exclude_modules\"");
    if (a != std::string::npos) {
        size_t lb = j.find('[', a), rb = j.find(']', lb);
        for (size_t p = lb + 1; lb != std::string::npos && rb != std::string::npos && p < rb; ) {
            size_t q0 = j.find('"', p); if (q0 == std::string::npos || q0 > rb) break;
            size_t q1 = j.find('"', q0 + 1); if (q1 == std::string::npos) break;
            c.ignore.push_back(j.substr(q0 + 1, q1 - q0 - 1));
            p = q1 + 1;
        }
    }
    return true;
}

// True if `module` matches any ignore entry: trailing-'*' prefix glob, else substring (so the
// bare token "lm_head" matches a fully-qualified module path).
inline bool is_ignored(const QuantConfig& c, const std::string& module) {
    for (const auto& pat : c.ignore) {
        if (pat.empty()) continue;
        if (pat.back() == '*') { if (module.compare(0, pat.size()-1, pat, 0, pat.size()-1) == 0) return true; }
        else if (module.find(pat) != std::string::npos) return true;
    }
    return false;
}

// Detect naming convention, layer prefix and the BF16 embed/lm_head/norm keys from a parsed
// checkpoint. Returns false + err if it doesn't look like an NVFP4 model we can map.
inline bool detect(const st::Model& m, Layout& out, std::string& err) {
    // naming: presence of any "*.weight_packed" ⇒ compressed-tensors, else ModelOpt (.weight).
    bool compressed = false;
    // layer prefix: prefer the multimodal nesting if present.
    const char* prefixes[] = { "model.language_model.layers.", "model.layers." };
    std::string prefix;
    // Probe layer 0 q_proj under each prefix / naming to pin both down at once.
    for (const char* pre : prefixes) {
        std::string b = std::string(pre) + "0.self_attn.q_proj.";
        if (m.has(b + "weight_packed")) { compressed = true; prefix = pre; break; }
        if (m.has(b + "weight"))        {                    prefix = pre; break; }
    }
    if (prefix.empty()) { err = "no recognizable NVFP4 layer-0 q_proj under known prefixes"; return false; }
    out.naming = compressed ? Naming::COMPRESSED : Naming::MODELOPT;
    out.layer_prefix = prefix;

    // count layers: walk until a layer's q_proj packed key is absent.
    int n = 0;
    for (;; n++) {
        ProjKeys k = proj_keys(out, n, P_Q);
        if (!m.has(k.packed)) break;
    }
    out.n_layers = n;
    if (n == 0) { err = "found prefix but zero layers"; return false; }

    // embed / final norm / lm_head. The embed lives under the same LM root as the layers.
    std::string root = out.layer_prefix.substr(0, out.layer_prefix.size() - std::strlen("layers."));
    out.embed_key     = root + "embed_tokens.weight";
    out.final_norm_key= root + "norm.weight";
    // Pull tie flag + ignore list from config.json when present (the ignore list is the real
    // value-add — which Linears stayed BF16).
    const std::string& cj = m.config_json();
    if (!cj.empty()) {
        QuantConfig qc;
        if (parse_config(cj, qc)) {
            out.tie_word_embeddings = qc.tie_word_embeddings;
            out.ignore = qc.ignore;
        }
    }
    // lm_head: PREFER an explicit lm_head.weight tensor when present — a materialized tensor is
    // authoritative over the (sometimes stale) tie flag. Only fall back to tied→embed when no
    // separate head tensor exists (then the tie flag confirms it).
    if (m.has("lm_head.weight"))             out.lmhead_key = "lm_head.weight";
    else if (m.has(root + "lm_head.weight")) out.lmhead_key = root + "lm_head.weight";
    else                                     out.lmhead_key = "";   // tied → reuse embed
    return true;
}

} // namespace nvfp4ld

#endif // FUCINA_NVFP4_LOADER_H
