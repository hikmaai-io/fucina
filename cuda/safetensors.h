// Minimal safetensors reader for NVFP4 checkpoints (NVIDIA ModelOpt / vLLM format).
//
// fucina was born GGUF-only; NVFP4 weights, however, ship as *safetensors* — that is the
// format the NVFP4 producers (TensorRT-Model-Optimizer, llm-compressor) emit and that vLLM
// consumes. GGUF has no NVFP4 block-scaled type, so to feed the Blackwell FP4 tensor cores a
// *natively* quantised model (rather than re-deriving FP4 from QAT-Q4_0, which pushes E2M1
// rounding error onto the whole model) we read safetensors directly.
//
// This header is intentionally schema-agnostic: it parses the safetensors container only —
// the 8-byte header length, the JSON tensor table, multi-shard `*.index.json` resolution and
// the mmap of each shard. NVFP4 *semantics* (which tensor is the packed E2M1 weight, which the
// E4M3 block scale, the dequant math) live in the engine, on top of st_find().
//
// Container layout (one .safetensors file):
//   [u64 LE header_len][header_len bytes of JSON][raw tensor bytes]
// The JSON is a flat object: { "<name>": {"dtype":"F8_E4M3","shape":[..],"data_offsets":[b,e]},
//   ..., "__metadata__": {...} }. data_offsets are relative to the END of the JSON header.
//
// Sharded checkpoints carry `model.safetensors.index.json`:
//   { "metadata": {...}, "weight_map": { "<tensor>": "model-00001-of-0000N.safetensors", ... } }
// We resolve every tensor to its shard, mmap each shard once, and present a single flat table.
//
// Header-only, no CUDA — pure host parsing. Include from the engine TU.
#ifndef FUCINA_SAFETENSORS_H
#define FUCINA_SAFETENSORS_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace st {

// safetensors standard dtype strings → enum. Only the ones an NVFP4 Gemma checkpoint uses
// are acted on by the engine; the rest are parsed so lookups don't fail.
enum class Dtype {
    UNKNOWN, F64, F32, F16, BF16,
    F8_E4M3, F8_E5M2,   // block scales / fp8 weights
    F4_E2M1,            // some producers tag packed fp4 as F4_E2M1; most use U8
    I64, I32, I16, I8, U8, BOOL,
};

inline Dtype parse_dtype(const char* s, size_t n) {
    auto eq = [&](const char* k){ return n == strlen(k) && memcmp(s, k, n) == 0; };
    if (eq("F64"))     return Dtype::F64;
    if (eq("F32"))     return Dtype::F32;
    if (eq("F16"))     return Dtype::F16;
    if (eq("BF16"))    return Dtype::BF16;
    if (eq("F8_E4M3")) return Dtype::F8_E4M3;
    if (eq("F8_E5M2")) return Dtype::F8_E5M2;
    if (eq("F4_E2M1")) return Dtype::F4_E2M1;
    if (eq("I64"))     return Dtype::I64;
    if (eq("I32"))     return Dtype::I32;
    if (eq("I16"))     return Dtype::I16;
    if (eq("I8"))      return Dtype::I8;
    if (eq("U8"))      return Dtype::U8;
    if (eq("BOOL"))    return Dtype::BOOL;
    return Dtype::UNKNOWN;
}

// Bytes-per-element ×100 (F4_E2M1 packs 2/byte → 50). Used only for sanity checks; the
// authoritative byte span of a tensor is end-begin from data_offsets.
inline int dtype_bits(Dtype d) {
    switch (d) {
        case Dtype::F64: case Dtype::I64:                 return 64;
        case Dtype::F32: case Dtype::I32:                 return 32;
        case Dtype::F16: case Dtype::BF16: case Dtype::I16:return 16;
        case Dtype::F8_E4M3: case Dtype::F8_E5M2:
        case Dtype::I8: case Dtype::U8: case Dtype::BOOL:  return 8;
        case Dtype::F4_E2M1:                              return 4;
        default:                                          return 0;
    }
}

struct Tensor {
    Dtype                dtype = Dtype::UNKNOWN;
    std::vector<int64_t> shape;     // logical dims as recorded (packed fp4 => last dim = in/2)
    const uint8_t*       data = nullptr;  // device-uploadable host pointer into the mmap
    size_t               nbytes = 0;      // end - begin
};

// ── tiny JSON scanner ────────────────────────────────────────────────────────
// The safetensors header is machine-generated and regular: a flat object whose values are
// either strings, arrays of ints, or nested objects. We need exactly: object iteration,
// string values, and int arrays. Not a general JSON parser — just enough, robust to spaces.
namespace json {
struct P {
    const char* p; const char* e;
    P(const char* b, size_t n) : p(b), e(b + n) {}
    void ws() { while (p < e && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r')) ++p; }
    bool eat(char c) { ws(); if (p < e && *p == c) { ++p; return true; } return false; }
    char peek() { ws(); return p < e ? *p : '\0'; }
    // parse a JSON string (no escapes beyond \" / \\ appear in tensor names/dtypes, but
    // handle them) into out; p left just past the closing quote.
    bool str(std::string& out) {
        ws(); if (p >= e || *p != '"') return false; ++p; out.clear();
        while (p < e && *p != '"') {
            if (*p == '\\' && p + 1 < e) { ++p; out.push_back(*p=='n'?'\n':*p=='t'?'\t':*p); }
            else out.push_back(*p);
            ++p;
        }
        if (p >= e) return false;
        ++p;
        return true;
    }
    bool integer(int64_t& out) {
        ws(); const char* s = p; bool neg = false;
        if (p < e && (*p=='-'||*p=='+')) { neg = (*p=='-'); ++p; }
        if (p >= e || *p < '0' || *p > '9') { p = s; return false; }
        int64_t v = 0; while (p < e && *p>='0' && *p<='9') { v = v*10 + (*p-'0'); ++p; }
        out = neg ? -v : v; return true;
    }
    // skip an arbitrary JSON value (used for __metadata__ and unknown fields)
    void skip_value() {
        ws(); if (p >= e) return;
        if (*p == '"') { std::string t; str(t); return; }
        if (*p == '{' || *p == '[') {
            char open = *p, close = (open=='{') ? '}' : ']'; int depth = 0;
            while (p < e) {
                if (*p == '"') { std::string t; str(t); continue; }
                if (*p == open) ++depth; else if (*p == close) { --depth; if (depth==0){ ++p; return; } }
                ++p;
            }
            return;
        }
        // number / true / false / null
        while (p < e && *p!=',' && *p!='}' && *p!=']') ++p;
    }
    bool int_array(std::vector<int64_t>& out) {
        out.clear(); if (!eat('[')) return false;
        if (peek() == ']') { ++p; return true; }
        for (;;) {
            int64_t v;
            if (!integer(v)) return false;
            out.push_back(v);
            if (eat(']')) return true;
            if (!eat(',')) return false;
        }
    }
};
} // namespace json

// ── parsed checkpoint: owns the mmaps, exposes a name→Tensor table ───────────
class Model {
public:
    ~Model() { for (auto& m : maps_) if (m.base) munmap((void*)m.base, m.len); }

    // Open a single .safetensors file, or a directory / .index.json for a sharded model.
    // Returns true on success. On failure, err holds a human message.
    bool open(const char* path, std::string& err);

    const Tensor* find(const std::string& name) const {
        auto it = table_.find(name);
        return it == table_.end() ? nullptr : &it->second;
    }
    bool has(const std::string& name) const { return table_.count(name) != 0; }
    size_t count() const { return table_.size(); }
    // raw bytes of the optional config.json sitting next to the checkpoint (empty if none)
    const std::string& config_json() const { return config_; }

private:
    struct Map { const uint8_t* base = nullptr; size_t len = 0; };
    std::vector<Map> maps_;
    std::unordered_map<std::string, Tensor> table_;
    std::string config_;

    // mmap one shard and merge its tensor table; data pointers point into the mmap.
    bool load_shard(const std::string& file, std::string& err);
    static bool slurp(const std::string& path, std::string& out);
    static bool is_dir(const std::string& path);
    static std::string dirname_of(const std::string& path);
};

inline bool Model::is_dir(const std::string& path) {
    struct stat s; return stat(path.c_str(), &s) == 0 && S_ISDIR(s.st_mode);
}
inline std::string Model::dirname_of(const std::string& path) {
    size_t s = path.find_last_of('/');
    return s == std::string::npos ? std::string(".") : path.substr(0, s);
}
inline bool Model::slurp(const std::string& path, std::string& out) {
    int fd = ::open(path.c_str(), O_RDONLY); if (fd < 0) return false;
    struct stat s; if (fstat(fd, &s) != 0) { ::close(fd); return false; }
    out.resize((size_t)s.st_size);
    ssize_t got = 0; while (got < s.st_size) {
        ssize_t r = ::read(fd, &out[got], s.st_size - got);
        if (r <= 0) { ::close(fd); return false; } got += r;
    }
    ::close(fd); return true;
}

// Parse one shard's header and register its tensors against the mmap base.
inline bool Model::load_shard(const std::string& file, std::string& err) {
    int fd = ::open(file.c_str(), O_RDONLY);
    if (fd < 0) { err = "open " + file; return false; }
    struct stat sstat; if (fstat(fd, &sstat) != 0) { ::close(fd); err = "stat " + file; return false; }
    size_t len = (size_t)sstat.st_size;
    if (len < 8) { ::close(fd); err = "short file " + file; return false; }
    const uint8_t* base = (const uint8_t*)mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) { err = "mmap " + file; return false; }
    maps_.push_back({base, len});

    uint64_t hlen; memcpy(&hlen, base, 8);   // little-endian host (Spark is LE)
    if (8 + hlen > len) { err = "header overruns " + file; return false; }
    const char* jbase = (const char*)base + 8;
    const uint8_t* tdata = base + 8 + hlen;

    json::P j(jbase, hlen);
    if (!j.eat('{')) { err = "bad header in " + file; return false; }
    if (j.peek() == '}') { return true; }   // empty
    for (;;) {
        std::string key;
        if (!j.str(key)) { err = "expected key in " + file; return false; }
        if (!j.eat(':')) { err = "expected ':' in " + file; return false; }
        if (key == "__metadata__") {
            j.skip_value();
        } else {
            // value object: {"dtype":..,"shape":[..],"data_offsets":[b,e]}
            if (!j.eat('{')) { err = "expected tensor obj in " + file; return false; }
            Tensor t; int64_t off_b = -1, off_e = -1;
            for (;;) {
                std::string fld;
                if (!j.str(fld)) { err = "expected field in " + file; return false; }
                if (!j.eat(':')) { err = "expected ':' in " + file; return false; }
                if (fld == "dtype") { std::string d; if (!j.str(d)) return false;
                    t.dtype = parse_dtype(d.c_str(), d.size()); }
                else if (fld == "shape") { if (!j.int_array(t.shape)) return false; }
                else if (fld == "data_offsets") {
                    std::vector<int64_t> o;
                    if (!j.int_array(o) || o.size() != 2) return false;
                    off_b = o[0]; off_e = o[1];
                }
                else { j.skip_value(); }
                if (j.eat('}')) break;
                if (!j.eat(',')) { err = "bad tensor obj in " + file; return false; }
            }
            if (off_b < 0 || off_e < off_b) { err = "bad data_offsets for " + key; return false; }
            t.data   = tdata + off_b;
            t.nbytes = (size_t)(off_e - off_b);
            table_[key] = std::move(t);
        }
        if (j.eat('}')) break;
        if (!j.eat(',')) { err = "bad header object in " + file; return false; }
    }
    return true;
}

inline bool Model::open(const char* path, std::string& err) {
    std::string p(path);
    std::string dir, index;

    // Resolve to a directory + an entry point. Accept: a directory, an .index.json, or a
    // single .safetensors file.
    if (is_dir(p)) {
        dir = p;
        index = dir + "/model.safetensors.index.json";
    } else if (p.size() > 11 && p.compare(p.size()-11, 11, ".index.json") == 0) {
        index = p; dir = dirname_of(p);
    } else {
        // single file
        dir = dirname_of(p);
        if (!load_shard(p, err)) return false;
        // pick up a sibling config.json if present (ignored on failure)
        std::string c; if (slurp(dir + "/config.json", c)) config_ = std::move(c);
        return true;
    }

    // sharded: read the index, collect the distinct shard files, load each once.
    std::string idx;
    if (!slurp(index, idx)) {
        // no index — fall back to a lone model.safetensors in the dir
        std::string single = dir + "/model.safetensors";
        if (!load_shard(single, err)) { err = "no index.json and no model.safetensors in " + dir; return false; }
        std::string c; if (slurp(dir + "/config.json", c)) config_ = std::move(c);
        return true;
    }

    // parse weight_map: { "<tensor>": "<file>", ... } — collect distinct files in order.
    json::P j(idx.data(), idx.size());
    std::vector<std::string> shards; std::unordered_map<std::string,int> seen;
    if (!j.eat('{')) { err = "bad index.json"; return false; }
    for (;;) {
        std::string key;
        if (!j.str(key)) { err = "bad index key"; return false; }
        if (!j.eat(':')) { err = "bad index ':'"; return false; }
        if (key == "weight_map") {
            if (!j.eat('{')) { err = "bad weight_map"; return false; }
            if (j.peek() != '}') for (;;) {
                std::string tname, fname;
                if (!j.str(tname) || !j.eat(':') || !j.str(fname)) { err = "bad weight_map entry"; return false; }
                if (!seen.count(fname)) { seen[fname] = 1; shards.push_back(fname); }
                if (j.eat('}')) break;
                if (!j.eat(',')) { err = "bad weight_map sep"; return false; }
            } else { ++j.p; }
        } else { j.skip_value(); }
        if (j.eat('}')) break;
        if (!j.eat(',')) { err = "bad index object"; return false; }
    }
    if (shards.empty()) { err = "empty weight_map in index.json"; return false; }
    for (const auto& s : shards) if (!load_shard(dir + "/" + s, err)) return false;
    std::string c; if (slurp(dir + "/config.json", c)) config_ = std::move(c);
    return true;
}

} // namespace st

#endif // FUCINA_SAFETENSORS_H
