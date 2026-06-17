// Name-mapping test for nvfp4_loader.h: synthesize checkpoints in BOTH naming conventions
// (compressed-tensors + multimodal prefix, ModelOpt + text prefix) and assert detect() and
// proj_keys() resolve the right tensor keys, layer count, and embed/lm_head wiring.
// build: g++ -std=c++17 -O2 -Wall -Wextra cuda/nvfp4_loader_test.cc -o /tmp/nvfp4_ld && /tmp/nvfp4_ld
#include "nvfp4_loader.h"
#include "safetensors.h"
#include <cassert>
#include <cstdio>
#include <vector>
#include <string>

struct Entry { std::string name, dtype; std::vector<int64_t> shape; size_t bytes; };
static std::string build_st(const std::vector<Entry>& es) {
    std::string json = "{"; size_t off = 0; bool first = true; std::string data;
    for (const auto& e : es) {
        if (!first) json += ",";
        first = false;
        json += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
        for (size_t i = 0; i < e.shape.size(); i++) { if (i) json += ","; json += std::to_string(e.shape[i]); }
        json += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + e.bytes) + "]}";
        off += e.bytes; data.append(e.bytes, '\0');
    }
    json += "}";
    std::string out; uint64_t hlen = json.size();
    out.append((const char*)&hlen, 8); out += json; out += data;
    return out;
}
static void write_file(const std::string& p, const std::string& b) {
    FILE* f = fopen(p.c_str(), "wb"); assert(f); fwrite(b.data(), 1, b.size(), f); fclose(f);
}

// emit the 7 projections for one layer under a prefix in a given naming.
static void add_layer(std::vector<Entry>& es, const std::string& prefix, int l,
                      bool compressed, int out_dim, int in_dim) {
    const char* sufs[] = {"self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
                          "mlp.gate_proj","mlp.up_proj","mlp.down_proj"};
    for (const char* s : sufs) {
        std::string b = prefix + std::to_string(l) + "." + s + ".";
        es.push_back({b + (compressed ? "weight_packed" : "weight"), "U8", {out_dim, in_dim/2}, (size_t)out_dim*(in_dim/2)});
        es.push_back({b + "weight_scale", "F8_E4M3", {out_dim, in_dim/16}, (size_t)out_dim*(in_dim/16)});
        es.push_back({b + (compressed ? "weight_global_scale" : "weight_scale_2"), "F32", {1}, 4});
        es.push_back({b + (compressed ? "input_global_scale" : "input_scale"), "F32", {1}, 4});
    }
}

int main() {
    // ── A: compressed-tensors, multimodal prefix, untied lm_head ──
    {
        std::vector<Entry> es;
        std::string pre = "model.language_model.layers.";
        for (int l = 0; l < 3; l++) add_layer(es, pre, l, /*compressed=*/true, 64, 128);
        es.push_back({"model.language_model.embed_tokens.weight", "BF16", {256, 64}, 256*64*2});
        es.push_back({"model.language_model.norm.weight", "BF16", {64}, 128});
        es.push_back({"lm_head.weight", "BF16", {256, 64}, 256*64*2});
        write_file("/tmp/ct.safetensors", build_st(es));

        st::Model m; std::string err;
        assert(m.open("/tmp/ct.safetensors", err));
        nvfp4ld::Layout L;
        if (!nvfp4ld::detect(m, L, err)) { printf("FAIL detect CT: %s\n", err.c_str()); return 1; }
        assert(L.naming == nvfp4ld::Naming::COMPRESSED);
        assert(L.layer_prefix == "model.language_model.layers.");
        assert(L.n_layers == 3);
        assert(L.embed_key == "model.language_model.embed_tokens.weight");
        assert(L.final_norm_key == "model.language_model.norm.weight");
        assert(L.lmhead_key == "lm_head.weight");   // untied
        auto k = nvfp4ld::proj_keys(L, 1, nvfp4ld::P_GATE);
        assert(k.packed == "model.language_model.layers.1.mlp.gate_proj.weight_packed");
        assert(k.scale  == "model.language_model.layers.1.mlp.gate_proj.weight_scale");
        assert(k.gscale == "model.language_model.layers.1.mlp.gate_proj.weight_global_scale");
        assert(m.has(k.packed) && m.has(k.scale) && m.has(k.gscale));
        printf("compressed-tensors / multimodal: OK (%d layers, lm_head=%s)\n", L.n_layers, L.lmhead_key.c_str());
    }

    // ── B: ModelOpt, text prefix, tied lm_head (no lm_head.weight) ──
    {
        std::vector<Entry> es;
        std::string pre = "model.layers.";
        for (int l = 0; l < 2; l++) add_layer(es, pre, l, /*compressed=*/false, 32, 64);
        es.push_back({"model.embed_tokens.weight", "BF16", {100, 32}, 100*32*2});
        es.push_back({"model.norm.weight", "BF16", {32}, 64});
        write_file("/tmp/mo.safetensors", build_st(es));

        st::Model m; std::string err;
        assert(m.open("/tmp/mo.safetensors", err));
        nvfp4ld::Layout L;
        if (!nvfp4ld::detect(m, L, err)) { printf("FAIL detect MO: %s\n", err.c_str()); return 1; }
        assert(L.naming == nvfp4ld::Naming::MODELOPT);
        assert(L.layer_prefix == "model.layers.");
        assert(L.n_layers == 2);
        assert(L.embed_key == "model.embed_tokens.weight");
        assert(L.lmhead_key == "");   // tied
        auto k = nvfp4ld::proj_keys(L, 0, nvfp4ld::P_Q);
        assert(k.packed == "model.layers.0.self_attn.q_proj.weight");
        assert(k.gscale == "model.layers.0.self_attn.q_proj.weight_scale_2");
        assert(m.has(k.packed) && m.has(k.scale) && m.has(k.gscale));
        printf("modelopt / text / tied:          OK (%d layers, tied head)\n", L.n_layers);
    }

    // ── C: config.json parsing (both ecosystems) + ignore globbing ──
    {
        // compressed-tensors style
        std::string ct =
            "{\"architectures\":[\"Gemma4UnifiedForConditionalGeneration\"],"
            "\"tie_word_embeddings\":true,"
            "\"quantization_config\":{\"quant_method\":\"compressed-tensors\","
            "\"format\":\"nvfp4-pack-quantized\",\"group_size\":16,"
            "\"ignore\":[\"lm_head\",\"model.embed_vision.patch_dense\",\"re:.*audio.*\"]}}";
        nvfp4ld::QuantConfig c;
        assert(nvfp4ld::parse_config(ct, c));
        assert(c.is_nvfp4 && c.naming == nvfp4ld::Naming::COMPRESSED);
        assert(c.group_size == 16 && c.tie_word_embeddings == true);
        assert(c.ignore.size() == 3);
        assert(nvfp4ld::is_ignored(c, "model.language_model.layers.0.self_attn.q_proj") == false);
        assert(nvfp4ld::is_ignored(c, "lm_head"));                                  // substring
        assert(nvfp4ld::is_ignored(c, "model.embed_vision.patch_dense"));
        // modelopt style
        std::string mo =
            "{\"tie_word_embeddings\":false,\"quant_algo\":\"NVFP4\",\"group_size\":16,"
            "\"exclude_modules\":[\"lm_head\",\"model.layers.30.self_attn*\"]}";
        nvfp4ld::QuantConfig c2;
        assert(nvfp4ld::parse_config(mo, c2));
        assert(c2.naming == nvfp4ld::Naming::MODELOPT && c2.tie_word_embeddings == false);
        assert(nvfp4ld::is_ignored(c2, "model.layers.30.self_attn.q_proj"));        // trailing-* glob
        assert(!nvfp4ld::is_ignored(c2, "model.layers.31.self_attn.q_proj"));
        // not-NVFP4 config rejected
        nvfp4ld::QuantConfig c3;
        assert(!nvfp4ld::parse_config("{\"quant_method\":\"awq\"}", c3));
        printf("config.json parse + ignore glob:  OK\n");
    }

    printf("ALL OK\n");
    return 0;
}
