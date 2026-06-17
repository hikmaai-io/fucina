// Standalone test for safetensors.h: build a synthetic single-file checkpoint + a 2-shard
// sharded checkpoint with an index.json, parse both, assert dtype/shape/data integrity.
// build: g++ -std=c++17 -O2 cuda/safetensors_test.cc -o /tmp/st_test && /tmp/st_test
#include "safetensors.h"
#include <cassert>
#include <cstdio>

static void write_file(const std::string& path, const std::string& bytes) {
    FILE* f = fopen(path.c_str(), "wb"); assert(f);
    fwrite(bytes.data(), 1, bytes.size(), f); fclose(f);
}

// Build a .safetensors blob from (name,dtype,shape,payload) entries. data_offsets are
// assigned contiguously in iteration order.
struct Entry { std::string name, dtype; std::vector<int64_t> shape; std::string payload; };
static std::string build_st(const std::vector<Entry>& es) {
    std::string json = "{";
    size_t off = 0; bool first = true;
    std::string data;
    for (const auto& e : es) {
        if (!first) json += ",";
        first = false;
        json += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
        for (size_t i = 0; i < e.shape.size(); i++) { if (i) json += ","; json += std::to_string(e.shape[i]); }
        json += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + e.payload.size()) + "]}";
        off += e.payload.size();
        data += e.payload;
    }
    json += "}";
    std::string out;
    uint64_t hlen = json.size();
    out.append((const char*)&hlen, 8);
    out += json;
    out += data;
    return out;
}

int main() {
    // ---- single file ----
    std::string p0; for (int i = 0; i < 8; i++) p0.push_back((char)(i*2));        // U8 weight, 8 bytes
    std::string p1; { float v[4] = {1.f, 2.f, 3.f, 4.f}; p1.assign((char*)v, 16); } // F32 scale_2
    std::string p2(4, '\x70');                                                     // F8_E4M3 block scales
    auto blob = build_st({
        {"model.layers.0.self_attn.q_proj.weight",        "U8",      {4, 4}, p0},  // [out=4, in/2=4]
        {"model.layers.0.self_attn.q_proj.weight_scale_2","F32",     {1},    p1},
        {"model.layers.0.self_attn.q_proj.weight_scale",  "F8_E4M3", {4, 2}, p2},
    });
    write_file("/tmp/st_single.safetensors", blob);

    std::string err;
    st::Model m;
    if (!m.open("/tmp/st_single.safetensors", err)) { printf("FAIL open single: %s\n", err.c_str()); return 1; }
    assert(m.count() == 3);
    const st::Tensor* w = m.find("model.layers.0.self_attn.q_proj.weight");
    assert(w && w->dtype == st::Dtype::U8);
    assert(w->shape.size() == 2 && w->shape[0] == 4 && w->shape[1] == 4);
    assert(w->nbytes == 8);
    for (int i = 0; i < 8; i++) assert(w->data[i] == (uint8_t)(i*2));
    const st::Tensor* s2 = m.find("model.layers.0.self_attn.q_proj.weight_scale_2");
    assert(s2 && s2->dtype == st::Dtype::F32 && s2->nbytes == 16);
    assert(((const float*)s2->data)[2] == 3.f);
    const st::Tensor* bs = m.find("model.layers.0.self_attn.q_proj.weight_scale");
    assert(bs && bs->dtype == st::Dtype::F8_E4M3 && bs->shape[0] == 4 && bs->shape[1] == 2);
    assert(!m.find("does.not.exist"));
    printf("single-file: OK (%zu tensors)\n", m.count());

    // ---- sharded: two shards + index.json ----
    mkdir("/tmp/st_shard", 0755);
    auto b0 = build_st({{"model.embed_tokens.weight", "BF16", {8, 2}, std::string(32, '\x01')}});
    auto b1 = build_st({{"model.layers.0.mlp.down_proj.weight", "U8", {2, 3}, std::string(6, '\x05')}});
    write_file("/tmp/st_shard/model-00001-of-00002.safetensors", b0);
    write_file("/tmp/st_shard/model-00002-of-00002.safetensors", b1);
    std::string index =
        "{\"metadata\":{\"total_size\":38},\"weight_map\":{"
        "\"model.embed_tokens.weight\":\"model-00001-of-00002.safetensors\","
        "\"model.layers.0.mlp.down_proj.weight\":\"model-00002-of-00002.safetensors\"}}";
    write_file("/tmp/st_shard/model.safetensors.index.json", index);

    st::Model ms;
    if (!ms.open("/tmp/st_shard", err)) { printf("FAIL open shard: %s\n", err.c_str()); return 1; }
    assert(ms.count() == 2);
    assert(ms.has("model.embed_tokens.weight"));
    const st::Tensor* dp = ms.find("model.layers.0.mlp.down_proj.weight");
    assert(dp && dp->dtype == st::Dtype::U8 && dp->data[0] == 5);
    printf("sharded:     OK (%zu tensors across 2 shards)\n", ms.count());

    printf("ALL OK\n");
    return 0;
}
