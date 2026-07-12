// ABOUTME: Host unit test for the DFlash draft loader schema (P2 of S1a): config + tensor
// ABOUTME: validation and hostile-input rejection, all BEFORE any CUDA allocation.
//
// build: g++ -std=c++17 -O2 -Wall -Wextra -Icuda cuda/qwen35_dflash_loader_test.cc -o /tmp/dflash_ld && /tmp/dflash_ld
#include "qwen35_dflash_loader.h"
#include <cstdio>
#include <cassert>

static void write_file(const std::string& path, const std::string& bytes) {
    FILE* f = fopen(path.c_str(), "wb"); assert(f);
    fwrite(bytes.data(), 1, bytes.size(), f); fclose(f);
}

struct Entry { std::string name, dtype; std::vector<int64_t> shape; std::string payload; };
static std::string build_st(const std::vector<Entry>& es) {
    std::string json = "{"; size_t off = 0; bool first = true; std::string data;
    for (const auto& e : es) {
        if (!first) json += ",";
        first = false;
        json += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
        for (size_t i = 0; i < e.shape.size(); i++) { if (i) json += ","; json += std::to_string(e.shape[i]); }
        json += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + e.payload.size()) + "]}";
        off += e.payload.size(); data += e.payload;
    }
    json += "}";
    std::string out; uint64_t hlen = json.size();
    out.append((const char*)&hlen, 8); out += json; out += data;
    return out;
}

// Tiny synthetic DFlash draft geometry (small H/I/L so payloads stay tiny) mirroring the real
// tensor NAMES and SHAPE RELATIONSHIPS exactly.
static const int H = 8, I = 16, L = 2, NQ = 4, NKV = 2, HD = 4, V = 64, F = 2;
static std::string bf16(int n) { return std::string((size_t)n * 2, '\x3c'); }  // n BF16 elems

static std::vector<Entry> good_tensors() {
    std::vector<Entry> es;
    es.push_back({"hidden_norm.weight", "BF16", {H}, bf16(H)});
    es.push_back({"norm.weight",        "BF16", {H}, bf16(H)});
    es.push_back({"fc.weight",          "BF16", {H, H*F}, bf16(H*H*F)});
    for (int l = 0; l < L; l++) {
        auto lk = [&](const char* s){ return "layers." + std::to_string(l) + "." + s; };
        es.push_back({lk("input_layernorm.weight"),          "BF16", {H}, bf16(H)});
        es.push_back({lk("post_attention_layernorm.weight"), "BF16", {H}, bf16(H)});
        es.push_back({lk("self_attn.q_proj.weight"), "BF16", {NQ*HD, H}, bf16(NQ*HD*H)});
        es.push_back({lk("self_attn.k_proj.weight"), "BF16", {NKV*HD, H}, bf16(NKV*HD*H)});
        es.push_back({lk("self_attn.v_proj.weight"), "BF16", {NKV*HD, H}, bf16(NKV*HD*H)});
        es.push_back({lk("self_attn.o_proj.weight"), "BF16", {H, NQ*HD}, bf16(H*NQ*HD)});
        es.push_back({lk("self_attn.q_norm.weight"), "BF16", {HD}, bf16(HD)});
        es.push_back({lk("self_attn.k_norm.weight"), "BF16", {HD}, bf16(HD)});
        es.push_back({lk("mlp.gate_proj.weight"), "BF16", {I, H}, bf16(I*H)});
        es.push_back({lk("mlp.up_proj.weight"),   "BF16", {I, H}, bf16(I*H)});
        es.push_back({lk("mlp.down_proj.weight"), "BF16", {H, I}, bf16(H*I)});
    }
    return es;
}

static std::string good_config() {
    // Mirrors the real config.json keys; 1 sliding + 1 full layer with a window.
    return std::string("{")
        + "\"architectures\":[\"DFlashDraftModel\"],\"model_type\":\"qwen3\","
        + "\"hidden_size\":" + std::to_string(H) + ","
        + "\"intermediate_size\":" + std::to_string(I) + ","
        + "\"num_hidden_layers\":" + std::to_string(L) + ","
        + "\"num_attention_heads\":" + std::to_string(NQ) + ","
        + "\"num_key_value_heads\":" + std::to_string(NKV) + ","
        + "\"head_dim\":" + std::to_string(HD) + ","
        + "\"vocab_size\":" + std::to_string(V) + ","
        + "\"sliding_window\":16,"
        + "\"layer_types\":[\"sliding_attention\",\"full_attention\"],"
        + "\"dflash_config\":{\"block_size\":16,\"mask_token_id\":48,"
        + "\"target_layer_ids\":[1,5]}"
        + "}";
}

static int failures = 0;
#define EXPECT_OK(call, what) do { std::string e; if (!(call)) { printf("FAIL(%s): unexpected reject: %s\n", what, e.c_str()); failures++; } } while(0)
#define EXPECT_REJECT(call, what) do { std::string e; if ((call)) { printf("FAIL(%s): accepted hostile input\n", what); failures++; } else { /* ok: rejected */ } } while(0)

int main() {
    using namespace qwen35dflash;

    // 1) Valid config parses to the expected geometry.
    {
        Geometry g; std::string e;
        if (!parse_config(good_config(), g, e)) { printf("FAIL: good config rejected: %s\n", e.c_str()); return 1; }
        if (g.H!=H||g.I!=I||g.L!=L||g.NQ!=NQ||g.NKV!=NKV||g.HD!=HD||g.V!=V) { printf("FAIL: geometry mismatch\n"); failures++; }
        if (g.mask_token_id != 48) { printf("FAIL: mask_token_id\n"); failures++; }
        if (g.num_target_features != F) { printf("FAIL: target features %d\n", g.num_target_features); failures++; }
        if (g.fc_in() != H*F) { printf("FAIL: fc_in %d\n", g.fc_in()); failures++; }
        if (g.sliding_window != 16) { printf("FAIL: sliding_window\n"); failures++; }
        if (g.layer_attn.size()!=2 || g.layer_attn[0]!=ATTN_SLIDING || g.layer_attn[1]!=ATTN_FULL) { printf("FAIL: layer_types\n"); failures++; }
    }

    // 2) Valid tensor set validates.
    {
        write_file("/tmp/dflash_good.safetensors", build_st(good_tensors()));
        st::Model m; std::string e;
        if (!m.open("/tmp/dflash_good.safetensors", e)) { printf("FAIL open good: %s\n", e.c_str()); return 1; }
        Geometry g; if (!parse_config(good_config(), g, e)) { printf("FAIL parse: %s\n", e.c_str()); return 1; }
        if (!validate_tensors(m, g, V, e)) { printf("FAIL: good tensors rejected: %s\n", e.c_str()); failures++; }
        if (g.has_d2t) { printf("FAIL: spurious d2t\n"); failures++; }
        if (g.draft_vocab != V) { printf("FAIL: draft_vocab != V\n"); failures++; }
    }

    // 3) Hostile config variants must be rejected with a reason.
    auto reject_cfg = [&](const std::string& patched, const char* what) {
        Geometry g; std::string e;
        if (parse_config(patched, g, e)) { printf("FAIL(%s): hostile config accepted\n", what); failures++; }
    };
    reject_cfg("{\"architectures\":[\"Llama\"],\"model_type\":\"qwen3\"}", "wrong arch");
    reject_cfg("{\"architectures\":[\"DFlashDraftModel\"],\"model_type\":\"gpt2\"}", "wrong model_type");
    {   // NQ not a multiple of NKV
        std::string c = good_config();
        size_t p = c.find("\"num_key_value_heads\":2"); c.replace(p, 24, "\"num_key_value_heads\":3");
        reject_cfg(c, "nq%nkv");
    }
    {   // mask_token_id out of vocab
        std::string c = good_config();
        size_t p = c.find("\"mask_token_id\":48"); c.replace(p, 18, "\"mask_token_id\":9999");
        reject_cfg(c, "mask oob");
    }
    {   // layer_types length mismatch
        std::string c = good_config();
        size_t p = c.find("\"layer_types\":[\"sliding_attention\",\"full_attention\"]");
        c.replace(p, 51, "\"layer_types\":[\"full_attention\"]");
        reject_cfg(c, "layer_types len");
    }
    {   // sliding layer but no window
        std::string c = good_config();
        size_t p = c.find("\"sliding_window\":16,"); c.replace(p, 20, "");
        reject_cfg(c, "no window");
    }

    // 4) Hostile tensor variants must be rejected before any allocation.
    auto validate_with = [&](std::vector<Entry> es, const char* what, bool expect_ok) {
        write_file("/tmp/dflash_hostile.safetensors", build_st(es));
        st::Model m; std::string e;
        if (!m.open("/tmp/dflash_hostile.safetensors", e)) { printf("FAIL open %s: %s\n", what, e.c_str()); failures++; return; }
        Geometry g; if (!parse_config(good_config(), g, e)) { printf("FAIL parse %s\n", what); failures++; return; }
        bool ok = validate_tensors(m, g, V, e);
        if (ok != expect_ok) { printf("FAIL(%s): got %s (%s)\n", what, ok?"accept":"reject", e.c_str()); failures++; }
    };
    {   // missing a required tensor
        auto es = good_tensors();
        es.erase(es.begin() + 3);   // remove first layer input_layernorm
        validate_with(es, "missing tensor", false);
    }
    {   // wrong q_proj out dim
        auto es = good_tensors();
        for (auto& e : es) if (e.name == "layers.0.self_attn.q_proj.weight") { e.shape = {NQ*HD + 1, H}; e.payload = bf16((NQ*HD+1)*H); }
        validate_with(es, "wrong q_proj rows", false);
    }
    {   // wrong rank on a norm
        auto es = good_tensors();
        for (auto& e : es) if (e.name == "norm.weight") { e.shape = {H, 1}; e.payload = bf16(H); }
        validate_with(es, "wrong norm rank", false);
    }
    {   // fc wrong input width
        auto es = good_tensors();
        for (auto& e : es) if (e.name == "fc.weight") { e.shape = {H, H*F + 3}; e.payload = bf16(H*(H*F+3)); }
        validate_with(es, "wrong fc width", false);
    }
    {   // non-float dtype on a weight
        auto es = good_tensors();
        for (auto& e : es) if (e.name == "layers.0.mlp.gate_proj.weight") { e.dtype = "I32"; e.payload = std::string((size_t)I*H*4, '\0'); }
        validate_with(es, "int weight", false);
    }

    // 5) Valid reduced-vocab d2t map accepted; out-of-range d2t rejected.
    {
        auto es = good_tensors();
        int dv = 32;
        std::string d2t; for (int i = 0; i < dv; i++) { int64_t delta = i; d2t.append((char*)&delta, 8); }
        es.push_back({"d2t", "I64", {dv}, d2t});
        write_file("/tmp/dflash_d2t.safetensors", build_st(es));
        st::Model m; std::string e; assert(m.open("/tmp/dflash_d2t.safetensors", e));
        Geometry g; assert(parse_config(good_config(), g, e));
        if (!validate_tensors(m, g, V, e)) { printf("FAIL: valid d2t rejected: %s\n", e.c_str()); failures++; }
        if (!g.has_d2t || g.draft_vocab != dv) { printf("FAIL: d2t not detected\n"); failures++; }
    }
    {   // d2t maps a token out of the target vocab
        auto es = good_tensors();
        int dv = 4;
        std::string d2t; for (int i = 0; i < dv; i++) { int64_t delta = V; d2t.append((char*)&delta, 8); }  // i+V >= V
        es.push_back({"d2t", "I64", {dv}, d2t});
        write_file("/tmp/dflash_d2t_bad.safetensors", build_st(es));
        st::Model m; std::string e; assert(m.open("/tmp/dflash_d2t_bad.safetensors", e));
        Geometry g; assert(parse_config(good_config(), g, e));
        if (validate_tensors(m, g, V, e)) { printf("FAIL: out-of-range d2t accepted\n"); failures++; }
    }

    if (failures) { printf("FAIL — DFlash loader schema (%d failures)\n", failures); return 1; }
    printf("PASS — DFlash loader schema: config geometry, tensor validation, hostile-input "
           "rejection (config + tensors), reduced-vocab d2t range check\n");
    return 0;
}
