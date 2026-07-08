// qwen35_fp8_loader.h — Qwen3.5 FP8 (DeepSeek-V3 block-quant) safetensors → engine key mapping.
//
// Mirrors nvfp4_loader.h but for the OFFICIAL Qwen/Qwen3.5-9B-FP8 checkpoint, swapping the NVFP4
// schema for DeepSeek block-fp8:
//   quantized Linear <p>.weight  F8_E4M3 [out,in]
//                    <p>.weight_scale_inv  BF16 [ceil(out/128), ceil(in/128)]
//                    dequant  W_bf16[o][i] = fp8(W[o][i]) * scale[o//128][i//128]
//   modules_to_not_convert (stay BF16/F32): all norms, embed_tokens, lm_head, conv1d, A_log,
//   dt_bias, in_proj_a, in_proj_b, linear_attn.norm  (no weight_scale_inv sibling).
//
// The text path lives under `model.language_model.*`; the vision tower (`model.visual.*`) and the
// MTP head (`mtp.*`, that is M6) are skipped here. Per-layer kind is the SAME period-4 hybrid the
// M0 detector pinned: FULL softmax-GQA iff (i+1)%full_attention_interval==0, else GDN linear —
// it equals config.text_config.layer_types[] exactly (verified against the FP8 config.json).
//
// Pure host string/JSON logic on top of st::Model (safetensors.h); no CUDA. The authoritative
// per-tensor quant decision is the presence of a `<weight>_scale_inv` sibling (robust to the
// per-layer modules_to_not_convert list), exposed via is_quantized().
#ifndef FUCINA_QWEN35_FP8_LOADER_H
#define FUCINA_QWEN35_FP8_LOADER_H

#include "safetensors.h"
#include <string>
#include <cstring>
#include <cstdlib>

namespace qwen35fp8 {

struct Layout {
    std::string lm_prefix;            // "model.language_model."        (trailing dot)
    std::string layer_prefix;         // "model.language_model.layers." (trailing dot)
    std::string embed_key;            // BF16 token embeddings
    std::string lmhead_key;           // BF16 untied LM head
    std::string final_norm_key;       // BF16 final norm
    int         n_layers = 0;
    int         full_attention_interval = 4;
    // nvidia ModelOpt MIXED_PRECISION repack (e.g. nvidia/Qwen3.6-35B-A3B-NVFP4): attn/GDN
    // projections are PER-TENSOR FP8 (`<w>_scale` F32 scalar, no `_scale_inv` block grid);
    // experts / shared expert / lm_head are native NVFP4 (`<w>_scale` E4M3 per-16-group +
    // `<w>_scale_2` F32 global). The fill path branches per tensor on the scale siblings.
    bool        modelopt = false;
};

// tiny config.json scanners (substring; the FP8 config keys we read are unambiguous in text_config)
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
    if (end == j.c_str() + p) return false; out = v; return true;
}

// "<layer_prefix><l>.<suffix>"  (suffix e.g. "input_layernorm.weight", "self_attn.q_proj.weight")
inline std::string lkey(const Layout& L, int l, const char* suffix) {
    return L.layer_prefix + std::to_string(l) + "." + suffix;
}

// FULL softmax-GQA layer iff (i+1)%interval==0, else GDN linear (period-4 hybrid).
inline bool is_full(const Layout& L, int l) {
    return L.full_attention_interval > 0 && ((l + 1) % L.full_attention_interval) == 0;
}

// A tensor is FP8 block-quantized iff it carries a "<name>_scale_inv" block-scale sibling.
inline bool is_quantized(const st::Model& m, const std::string& weight_key) {
    return m.has(weight_key + "_scale_inv");
}

// Recognize the Qwen3.5 FP8 text checkpoint from config.json + tensor probing, and pin the layer
// prefix / global keys / layer count. Returns false + err otherwise.
inline bool detect(const st::Model& m, Layout& out, std::string& err) {
    const std::string& cj = m.config_json();
    if (cj.empty()) { err = "no config.json next to checkpoint"; return false; }
    std::string mtype, qmethod;
    cfg_str(cj, "\"model_type\"", mtype);      // top-level "qwen3_5"
    cfg_str(cj, "\"quant_method\"", qmethod);  // "fp8" (Qwen block-FP8) or "modelopt" (nvidia NVFP4 repack)
    if (mtype.find("qwen3_5") == std::string::npos) { err = "config model_type is not qwen3_5"; return false; }
    if (qmethod != "fp8" && qmethod != "modelopt") { err = "config quant_method is not fp8/modelopt"; return false; }
    out.modelopt = (qmethod == "modelopt");

    out.lm_prefix    = "model.language_model.";
    out.layer_prefix = out.lm_prefix + "layers.";
    if (!m.has(out.layer_prefix + "0.input_layernorm.weight")) {
        err = "no model.language_model.layers.0 under expected prefix"; return false;
    }
    int n = 0;
    while (m.has(out.layer_prefix + std::to_string(n) + ".input_layernorm.weight")) n++;
    out.n_layers = n;
    if (n == 0) { err = "zero layers found"; return false; }

    long iv = 4; cfg_int(cj, "\"full_attention_interval\"", iv);
    out.full_attention_interval = (iv > 0) ? (int)iv : 4;

    out.embed_key      = out.lm_prefix + "embed_tokens.weight";
    out.final_norm_key = out.lm_prefix + "norm.weight";
    out.lmhead_key     = m.has("lm_head.weight") ? "lm_head.weight"
                       : (out.lm_prefix + "lm_head.weight");   // untied (tie_word_embeddings=false)
    return true;
}

} // namespace qwen35fp8

#endif // FUCINA_QWEN35_FP8_LOADER_H
