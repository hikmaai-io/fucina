// e4b_gguf.cuh — GGUF weight-source front-end for the Gemma-4-E4B engine.
//
// The E4B engine's validated forward/decode/KV-share/PLE/NVFP4 path consumes
// BF16 device tensors in HF nn.Linear layout [out_features, in_features],
// row-major. This header lets the SAME engine load those tensors from a GGUF
// (Q4_0-QAT / Q6_K / F16 / F32) instead of from BF16 safetensors: it parses the
// `gemma4.*` KV metadata into the SAME e4b::Config, and host-dequantizes each
// quantized tensor into the IDENTICAL row-major [out,in] BF16 buffer that the
// safetensors up_bf16() produces. Only the weight SOURCE changes; nothing
// downstream is touched.
//
// Linkage (mirrors e4b_nvfp4.cuh): the dense engine's gguf_* parse/dequant
// helpers in gemma4_kernels.cu are file-static (internal linkage), so they
// cannot be called cross-TU, and copying its external __global__
// dequant_to_bf16_kernel would clash in libfucina.a. We therefore keep PRIVATE
// `static` host-side copies here, in namespace e4bgguf. All dequant runs on the
// HOST over the mmap'd quant bytes (the loader is not perf-critical), then a
// single H2D upload — exactly the convert_q6k_to_q8_0 host pattern.
//
// GGML stores a weight with ne0 = in_features (contiguous) and ne1 =
// out_features; HF stores [out_features, in_features] row-major. These are the
// SAME bytes. The dequant marches the linear element index e = 0..n-1 in GGUF
// storage order and writes host_dst[e], which IS row-major [out,in]. No
// transpose — verified by the load+argmax parity gate in test_e4b_gguf_load.cu.
#ifndef FUCINA_E4B_GGUF_CUH
#define FUCINA_E4B_GGUF_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "gemma4_e4b.h"   // e4b::Config / e4b::Attn

namespace e4bgguf {

// ── GGUF header + value types (private static copies; see header note) ──────
#pragma pack(push, 1)
typedef struct {
    uint32_t magic;            // 0x46554747 "GGUF"
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
} gguf_header_t;
#pragma pack(pop)

typedef enum {
    GGUF_TYPE_UINT8   = 0,  GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,  GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,  GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,  GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,  GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10, GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_value_type_t;

// GGML element types we handle.
typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q6_K = 14,
} ggml_type_t;

#define E4BGGUF_ALIGNMENT 32

typedef struct { uint32_t elem_type; uint64_t count; const uint8_t* data; } gguf_array_t;
typedef struct { const char* ptr; uint64_t len; } gguf_str_t;

static inline uint64_t gguf_scalar_size(uint32_t t) {
    switch (t) {
        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL:   return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:                        return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32:return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64:return 8;
        default: return 0;
    }
}

static inline const char* gguf_read_str(const uint8_t** pp, const uint8_t* end, uint64_t* len_out) {
    const uint8_t* p = *pp;
    if (p + 8 > end) return nullptr;
    uint64_t len; memcpy(&len, p, 8); p += 8;
    if (p + len > end) return nullptr;
    const char* s = (const char*)p;
    p += len; *pp = p;
    if (len_out) *len_out = len;
    return s;
}

static inline int gguf_str_eq(const char* s, uint64_t len, const char* cstr) {
    return strlen(cstr) == len && memcmp(s, cstr, len) == 0;
}

static int gguf_skip_value(const uint8_t** pp, const uint8_t* end, uint32_t vtype) {
    const uint8_t* p = *pp;
    if (vtype == GGUF_TYPE_STRING) {
        if (!gguf_read_str(&p, end, nullptr)) return -1;
    } else if (vtype == GGUF_TYPE_ARRAY) {
        if (p + 12 > end) return -1;
        uint32_t at; memcpy(&at, p, 4);
        uint64_t n;  memcpy(&n, p + 4, 8);
        p += 12;
        if (at == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < n; i++) if (!gguf_read_str(&p, end, nullptr)) return -1;
        } else {
            uint64_t sz = gguf_scalar_size(at);
            if (sz == 0) return -1;
            if (p + sz * n > end) return -1;
            p += sz * n;
        }
    } else {
        uint64_t sz = gguf_scalar_size(vtype);
        if (sz == 0 || p + sz > end) return -1;
        p += sz;
    }
    *pp = p;
    return 0;
}

static const uint8_t* gguf_skip_metadata(const uint8_t* data, uint64_t size) {
    const gguf_header_t* hdr = (const gguf_header_t*)data;
    const uint8_t* end = data + size;
    const uint8_t* p = data + sizeof(gguf_header_t);
    for (uint64_t i = 0; i < hdr->metadata_kv_count; i++) {
        if (!gguf_read_str(&p, end, nullptr)) return nullptr;
        if (p + 4 > end) return nullptr;
        uint32_t vtype; memcpy(&vtype, p, 4); p += 4;
        if (gguf_skip_value(&p, end, vtype) != 0) return nullptr;
    }
    return p;
}

static uint64_t gguf_tensor_data_start(const uint8_t* data, uint64_t size) {
    const gguf_header_t* hdr = (const gguf_header_t*)data;
    const uint8_t* end = data + size;
    const uint8_t* p = gguf_skip_metadata(data, size);
    if (!p) return 0;
    for (uint64_t t = 0; t < hdr->tensor_count; t++) {
        if (!gguf_read_str(&p, end, nullptr)) return 0;
        if (p + 4 > end) return 0;
        uint32_t n_dims; memcpy(&n_dims, p, 4); p += 4;
        if (p + (uint64_t)n_dims * 8 + 12 > end) return 0;
        p += (uint64_t)n_dims * 8;   // dims
        p += 4;                      // ggml_type
        p += 8;                      // offset
    }
    uint64_t off = (uint64_t)(p - data);
    off = (off + (E4BGGUF_ALIGNMENT - 1)) & ~(uint64_t)(E4BGGUF_ALIGNMENT - 1);
    return off;
}

static int gguf_parse_metadata(const uint8_t* data, uint64_t size, const char* key,
                               void* value_out, gguf_value_type_t expected_type) {
    const gguf_header_t* hdr = (const gguf_header_t*)data;
    if (hdr->magic != 0x46554747) return -1;
    const uint8_t* end = data + size;
    const uint8_t* p = data + sizeof(gguf_header_t);
    for (uint64_t i = 0; i < hdr->metadata_kv_count; i++) {
        uint64_t klen = 0;
        const char* k = gguf_read_str(&p, end, &klen);
        if (!k) return -1;
        if (p + 4 > end) return -1;
        uint32_t vtype; memcpy(&vtype, p, 4); p += 4;
        if (gguf_str_eq(k, klen, key)) {
            if (vtype != (uint32_t)expected_type) return -1;
            switch (vtype) {
                case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL:    memcpy(value_out, p, 1); break;
                case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:                         memcpy(value_out, p, 2); break;
                case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: memcpy(value_out, p, 4); break;
                case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: memcpy(value_out, p, 8); break;
                case GGUF_TYPE_STRING: {
                    uint64_t slen; const char* s = gguf_read_str(&p, end, &slen);
                    if (!s) return -1;
                    gguf_str_t sv = { s, slen }; memcpy(value_out, &sv, sizeof(sv));
                    break;
                }
                case GGUF_TYPE_ARRAY: {
                    if (p + 12 > end) return -1;
                    gguf_array_t arr;
                    memcpy(&arr.elem_type, p, 4);
                    memcpy(&arr.count, p + 4, 8);
                    arr.data = p + 12;
                    memcpy(value_out, &arr, sizeof(arr));
                    break;
                }
                default: return -1;
            }
            return 0;
        }
        if (gguf_skip_value(&p, end, vtype) != 0) return -1;
    }
    return -1;
}

// Tensor lookup. On a name match returns absolute file offset of the tensor
// data, total element count, and GGML element type. Returns 0 on success.
static int gguf_find_tensor(const uint8_t* data, uint64_t size, const char* name,
                            uint64_t* offset_out, uint64_t* n_el_out, uint32_t* ggml_type_out) {
    const gguf_header_t* hdr = (const gguf_header_t*)data;
    if (size < sizeof(gguf_header_t)) return -1;
    if (hdr->magic != 0x46554747) return -1;
    if (hdr->version != 3) return -1;
    const uint8_t* end = data + size;
    const uint8_t* p = gguf_skip_metadata(data, size);
    if (!p) return -1;
    uint64_t tdata_start = gguf_tensor_data_start(data, size);
    if (tdata_start == 0 || tdata_start > size) return -1;
    for (uint64_t t = 0; t < hdr->tensor_count; t++) {
        uint64_t nlen = 0;
        const char* tname = gguf_read_str(&p, end, &nlen);
        if (!tname) return -1;
        if (p + 4 > end) return -1;
        uint32_t n_dims; memcpy(&n_dims, p, 4); p += 4;
        if (p + (uint64_t)n_dims * 8 + 12 > end) return -1;
        uint64_t n_el = 1;
        for (uint32_t d = 0; d < n_dims; d++) { uint64_t dv; memcpy(&dv, p, 8); p += 8; n_el *= dv; }
        uint32_t gtype; memcpy(&gtype, p, 4); p += 4;
        uint64_t toff;  memcpy(&toff, p, 8);  p += 8;
        if (gguf_str_eq(tname, nlen, name)) {
            if (toff > size - tdata_start) return -1;
            if (offset_out)    *offset_out = tdata_start + toff;
            if (n_el_out)      *n_el_out = n_el;
            if (ggml_type_out) *ggml_type_out = gtype;
            return 0;
        }
    }
    return -1;
}

// ── host fp16 → fp32 (private copy of gemma4_kernels.cu::h2f_host) ──────────
static inline float h2f_host(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF, f;
    if (e == 0) {
        if (m == 0) f = s;
        else { int ee = -1; do { ee++; m <<= 1; } while (!(m & 0x400));
               m &= 0x3FF; f = s | (uint32_t)((112 - ee) << 23) | (m << 13); }
    } else if (e == 0x1F) f = s | 0x7F800000u | (m << 13);
    else f = s | (uint32_t)((e + 112) << 23) | (m << 13);
    float o; memcpy(&o, &f, 4); return o;
}

// host f32 → bf16 (round-to-nearest-even). The safetensors path copies stored
// bf16 bits; the GGUF path computes bf16 from the dequantized f32, so use RNE to
// avoid a systematic 1-ULP bias that could shift a borderline argmax.
static inline __nv_bfloat16 f2bf16_host(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {           // NaN: keep it a NaN
        uint16_t bits = (uint16_t)((u >> 16) | 0x0040u);
        __nv_bfloat16 r; memcpy(&r, &bits, 2); return r;
    }
    uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);  // round-to-nearest-even
    u += rounding;
    uint16_t bits = (uint16_t)(u >> 16);
    __nv_bfloat16 r; memcpy(&r, &bits, 2); return r;
}

// ── host dequant: quant bytes → BF16, marching GGUF storage order ───────────
// Q4_0: block = 2-byte fp16 scale + 16 bytes of 32 nibbles (18 B/block).
//   byte j (0..15) holds elem[j] in the low nibble, elem[j+16] in the high
//   nibble; value = scale * (nibble - 8). (matches decode_weight fmt==2.)
static void dequant_q4_0_to_bf16(const uint8_t* src, int64_t n, __nv_bfloat16* dst) {
    const int64_t nblk = n / 32;
    for (int64_t b = 0; b < nblk; b++) {
        const uint8_t* blk = src + (size_t)b * 18;
        uint16_t hr = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        float scale = h2f_host(hr);
        for (int j = 0; j < 32; j++) {
            uint8_t byte = blk[2 + (j & 15)];
            int nib = (j < 16) ? (byte & 0x0F) : (byte >> 4);
            dst[b * 32 + j] = f2bf16_host(scale * (float)(nib - 8));
        }
    }
}

// Q6_K: super-block = 256 elems (ql[128] + qh[64] + scales[16](int8) + d(fp16)
// = 210 B). Exact unpack from convert_q6k_to_q8_0, emitting bf16 directly.
static void dequant_q6_k_to_bf16(const uint8_t* src, int64_t n, __nv_bfloat16* dst) {
    const int64_t n_super = n / 256;
    for (int64_t s = 0; s < n_super; s++) {
        const unsigned char* blk = src + (size_t)s * 210;
        const unsigned char* ql0 = blk;
        const unsigned char* qh0 = blk + 128;
        const int8_t*        sc0 = (const int8_t*)(blk + 192);
        uint16_t draw; memcpy(&draw, blk + 208, 2);
        float d = h2f_host(draw);
        __nv_bfloat16* o = dst + (size_t)s * 256;
        for (int n0 = 0; n0 < 256; n0 += 128) {
            const unsigned char* ql = ql0 + (n0/128)*64;
            const unsigned char* qh = qh0 + (n0/128)*32;
            const int8_t*        sc = sc0 + (n0/128)*8;
            for (int l = 0; l < 32; l++) {
                int is = l/16;
                int q1 = (int)((ql[l]    & 0xF) | (((qh[l]>>0)&3)<<4)) - 32;
                int q2 = (int)((ql[l+32] & 0xF) | (((qh[l]>>2)&3)<<4)) - 32;
                int q3 = (int)((ql[l]    >> 4)  | (((qh[l]>>4)&3)<<4)) - 32;
                int q4 = (int)((ql[l+32] >> 4)  | (((qh[l]>>6)&3)<<4)) - 32;
                o[n0+l+ 0] = f2bf16_host(d * sc[is+0] * q1);
                o[n0+l+32] = f2bf16_host(d * sc[is+2] * q2);
                o[n0+l+64] = f2bf16_host(d * sc[is+4] * q3);
                o[n0+l+96] = f2bf16_host(d * sc[is+6] * q4);
            }
        }
    }
}

static void dequant_f16_to_bf16(const uint8_t* src, int64_t n, __nv_bfloat16* dst) {
    const uint16_t* h = (const uint16_t*)src;
    for (int64_t i = 0; i < n; i++) dst[i] = f2bf16_host(h2f_host(h[i]));
}
static void copy_f32_to_bf16(const uint8_t* src, int64_t n, __nv_bfloat16* dst) {
    const float* f = (const float*)src;
    for (int64_t i = 0; i < n; i++) dst[i] = f2bf16_host(f[i]);
}

// Dispatch host dequant by GGML element type into a host BF16 buffer of n_el.
// Returns false on an unsupported type.
static bool gguf_dequant_to_bf16(const uint8_t* src, int64_t n_el, uint32_t ggml_type,
                                 __nv_bfloat16* host_dst) {
    switch (ggml_type) {
        case GGML_TYPE_F32:  copy_f32_to_bf16(src, n_el, host_dst);    return true;
        case GGML_TYPE_F16:  dequant_f16_to_bf16(src, n_el, host_dst); return true;
        case GGML_TYPE_Q4_0: dequant_q4_0_to_bf16(src, n_el, host_dst);return true;
        case GGML_TYPE_Q6_K: dequant_q6_k_to_bf16(src, n_el, host_dst);return true;
        default:
            fprintf(stderr, "e4b-gguf: unsupported ggml type %u\n", ggml_type);
            return false;
    }
}

// ── GgufFile: read-only mmap + tensor accessors ─────────────────────────────
struct GgufFile {
    const uint8_t* data = nullptr;
    uint64_t       size = 0;
    int            fd   = -1;

    bool open(const char* path, std::string& err) {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) { err = "open failed"; return false; }
        struct stat st;
        if (fstat(fd, &st) != 0) { err = "fstat failed"; ::close(fd); fd = -1; return false; }
        size = (uint64_t)st.st_size;
        void* m = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) { err = "mmap failed"; ::close(fd); fd = -1; return false; }
        data = (const uint8_t*)m;
        const gguf_header_t* hdr = (const gguf_header_t*)data;
        if (size < sizeof(gguf_header_t) || hdr->magic != 0x46554747 || hdr->version != 3) {
            err = "not a GGUF v3 file"; close(); return false;
        }
        return true;
    }
    void close() {
        if (data) { munmap((void*)data, size); data = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
        size = 0;
    }
    // file_off is the absolute offset of the tensor data within the mapping.
    bool find(const char* name, uint64_t* file_off, uint64_t* n_el, uint32_t* gtype) const {
        return gguf_find_tensor(data, size, name, file_off, n_el, gtype) == 0;
    }
    bool has(const char* name) const {
        uint64_t o, n; uint32_t g; return find(name, &o, &n, &g);
    }
    bool get_u32(const char* key, uint32_t* out) const {
        return gguf_parse_metadata(data, size, key, out, GGUF_TYPE_UINT32) == 0;
    }
    bool get_f32(const char* key, float* out) const {
        return gguf_parse_metadata(data, size, key, out, GGUF_TYPE_FLOAT32) == 0;
    }
    bool get_str(const char* key, gguf_str_t* out) const {
        return gguf_parse_metadata(data, size, key, out, GGUF_TYPE_STRING) == 0;
    }
    bool get_bool_arr(const char* key, gguf_array_t* out) const {
        return gguf_parse_metadata(data, size, key, out, GGUF_TYPE_ARRAY) == 0;
    }
};

// Cheap magic check (first 4 bytes == "GGUF") without mmapping the whole file.
static inline bool is_gguf_file(const char* path) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) return false;
    uint32_t magic = 0;
    ssize_t r = ::read(fd, &magic, 4);
    ::close(fd);
    return r == 4 && magic == 0x46554747u;
}

// ── config from GGUF KV metadata → the SAME e4b::Config ─────────────────────
// Mirrors gemma4_e4b.h::parse but reads `gemma4.*` keys + tensor probes. Fields
// absent from this GGUF (partial_rotary_factor) take the known Gemma-4 value
// (logged where assumed).
inline bool parse_gguf(const GgufFile& g, e4b::Config& c, std::string& err) {
    gguf_str_t arch;
    if (!g.get_str("general.architecture", &arch) || !gguf_str_eq(arch.ptr, arch.len, "gemma4")) {
        err = "general.architecture != gemma4"; return false;
    }
    uint32_t u;
    if (!g.get_u32("gemma4.embedding_length", &u))        { err = "no embedding_length"; return false; }
    c.hidden_size = (int)u;
    if (!g.get_u32("gemma4.feed_forward_length", &u))     { err = "no feed_forward_length"; return false; }
    c.intermediate_size = (int)u;
    if (!g.get_u32("gemma4.block_count", &u))             { err = "no block_count"; return false; }
    c.n_layers = (int)u;
    if (!g.get_u32("gemma4.attention.head_count", &u))    { err = "no head_count"; return false; }
    c.n_heads = (int)u;
    if (!g.get_u32("gemma4.attention.head_count_kv", &u)) { err = "no head_count_kv"; return false; }
    c.n_kv_heads = (int)u;
    if (!g.get_u32("gemma4.attention.key_length_swa", &u)){ err = "no key_length_swa"; return false; }
    c.head_dim = (int)u;                                  // sliding head_dim (256)
    if (!g.get_u32("gemma4.attention.key_length", &u))    { err = "no key_length"; return false; }
    c.global_head_dim = (int)u;                           // full head_dim (512)
    if (g.get_u32("gemma4.attention.sliding_window", &u)) c.sliding_window = (int)u;
    if (g.get_u32("gemma4.embedding_length_per_layer_input", &u)) c.ple_dim = (int)u;
    if (g.get_u32("gemma4.attention.shared_kv_layers", &u)) c.n_kv_shared_layers = (int)u;
    if (g.get_u32("gemma4.context_length", &u))           c.max_position = (int)u;

    // vocab + ple_vocab are NOT metadata — derive from the embedding tensors.
    {
        uint64_t off, n_el; uint32_t gt;
        if (!g.find("token_embd.weight", &off, &n_el, &gt) || c.hidden_size <= 0) {
            err = "no token_embd.weight"; return false;
        }
        c.vocab_size = (int)(n_el / (uint64_t)c.hidden_size);   // [vocab, hidden]
        c.ple_vocab = c.vocab_size;                             // PLE table shares vocab
    }

    float fv;
    c.rms_eps = 1e-6f;
    if (g.get_f32("gemma4.attention.layer_norm_rms_epsilon", &fv)) c.rms_eps = fv;
    c.rope_theta_sliding = 1e4f;
    if (g.get_f32("gemma4.rope.freq_base_swa", &fv)) c.rope_theta_sliding = fv;
    c.rope_theta_full = 1e6f;
    if (g.get_f32("gemma4.rope.freq_base", &fv)) c.rope_theta_full = fv;
    c.final_logit_softcap = 30.0f;
    if (g.get_f32("gemma4.final_logit_softcapping", &fv)) c.final_logit_softcap = fv;

    // partial_rotary_factor (proportional RoPE on full-attn layers) is NOT a
    // gemma4.* GGUF key — the dense engine has the same gap. Use the known
    // Gemma-4 value 0.25 (matches config.json full_attention.partial_rotary_factor).
    c.rope_partial_full = 0.25f;
    fprintf(stderr, "e4b-gguf: assuming full-attn partial_rotary_factor=%.2f "
            "(not carried in GGUF; Gemma-4 default), softcap=%.1f\n",
            c.rope_partial_full, c.final_logit_softcap);

    // tie_word_embeddings: true iff there is no separate output.weight tensor.
    c.tie_word_embeddings = !g.has("output.weight");

    // layer_types: sliding_window_pattern is bool[n_layers], 1=SLIDING, 0=FULL.
    c.layer_types.clear();
    gguf_array_t arr;
    if (g.get_bool_arr("gemma4.attention.sliding_window_pattern", &arr) &&
        (int)arr.count == c.n_layers && arr.elem_type == GGUF_TYPE_BOOL) {
        for (uint64_t i = 0; i < arr.count; i++) {
            uint8_t b = arr.data[i];
            c.layer_types.push_back(b ? e4b::Attn::SLIDING : e4b::Attn::FULL);
        }
    } else {
        // Synthesize the canonical 5:1 pattern (full at idx%6==5).
        c.layer_types.assign(c.n_layers, e4b::Attn::SLIDING);
        for (int i = 5; i < c.n_layers; i += 6) c.layer_types[i] = e4b::Attn::FULL;
    }
    return c.valid();
}

// GGUF tensor-name builder (no model.language_model. prefix).
struct GgufNames {
    std::string token_embd()      const { return "token_embd.weight"; }
    std::string ple_token_embd()  const { return "per_layer_token_embd.weight"; }
    std::string plm_proj()        const { return "per_layer_model_proj.weight"; }
    std::string ple_proj_norm()   const { return "per_layer_proj_norm.weight"; }
    std::string final_norm()      const { return "output_norm.weight"; }
    std::string L(int i, const char* s) const {
        return "blk." + std::to_string(i) + "." + s;
    }
};

} // namespace e4bgguf

#endif // FUCINA_E4B_GGUF_CUH
