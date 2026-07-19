// gemma4_kernels.cu - Gemma 4 12B CUDA inference for DGX Spark GB10
// Blackwell sm_120, FP8 Tensor Core native, Q8_0 fallback
//
// Architecture:
//   48 layers: 40 sliding window + 8 global attention
//   sliding: 8 KV heads, head_dim=256, RoPE theta=10000
//   global:  1 KV head,  head_dim=512, p-RoPE theta=1M, K=V unified
//   FFN: GeGLU (gelu_pytorch_tanh), intermediate=15360
//   Output: logit softcapping at 30.0

#include "gemma4_kernels.cuh"
#include "tensor_types.h"
#include "model_plan.h"
#include "device_allocation_set.h"
#include "expert_profile.h"
#include "gemma4_detect.h"   // M0: runtime arch auto-detection (gemma4_detect_from_gguf/_config_json)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>         // __nv_fp8_storage_t, conversion functions
#include "paged_kv_device.cuh" // paged KV: host block-table bookkeeping + device access kernels
#include "paged_prefix.h"      // cross-request prefix cache (RadixAttention): radix tree + refcount + LRU
                              // (Phase 2 continuous batching). Pulls paged_kv.h. Kernels are
                              // compiled into the engine TU but stay dormant until wired.
#include <cuda_fp4.h>         // __nv_fp4_storage_t, NVFP4 E2M1 conversion (FUCINA_FP4)
#include <cublasLt.h>         // NVFP4 block-scaled tensor-core GEMM (FUCINA_FP4)
#include <mma.h>              // nvcuda::wmma tf32 tensor cores (GDN chunk-scan matmuls)
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_pipeline.h>   // D32B: cp.async (__pipeline_memcpy_async) for smem weight prefetch

// POSIX headers for host-side file loading (mmap, open, etc.)
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <glob.h>

// ── Qwen3-MoE grouped-expert kernels (libdg) ────────────────────────────────────────────────────
// The sparse FFN reuses the DiffusionGemma MoE primitives (declared in diffusion_gemma_kernels.cuh,
// compiled into libdg.a which fucina already links). These are extern-C HOST launchers — no device
// symbol crosses the archive boundary, so no -dlink change is needed. Declared here (not via the
// header include, which would pull DG model #defines) to keep the gemma4 TU self-contained.
extern "C" {
    void dg_softmax_topk(const float *logits, int n_expert, int tokens, int topk,
                         int *out_idx, float *out_w, cudaStream_t s);
    void dg_moe_route(const int *tki, const float *tkw, const float *pes, int n_tokens, int n_used,
                      int n_expert, int *count, int *coloff, int *cursor, int *src, float *csc,
                      cudaStream_t s);
    void dg_moe_route_inv(const int *tki, const float *tkw, const float *pes, int n_tokens, int n_used,
                          int n_expert, int *count, int *coloff, int *cursor, int *src, float *csc,
                          int *invpos, int *active, int n_slot, cudaStream_t s);
    void dg_moe_reduce(float *out, const float *oe, const int *invpos, const float *csc,
                       int feat, int n_tokens, int n_used, cudaStream_t s);
    void dg_gather_cols(float *dst, const float *src, const int *idx, int feat, int ncols, cudaStream_t s);
    void dg_scatteradd_cols(float *dst, const float *src, const int *idx, const float *colscale,
                            int feat, int ncols, cudaStream_t s);
    void dg_quantize_q8_1(const float *x, int8_t *qx, float *dx, int *sx, int in_dim, int tokens,
                          cudaStream_t s);
    void dg_mmq_q4_K_grouped(float *out, const void *wbase, int64_t slab_stride, const int8_t *qx,
                             const float *dx, const int *sx, const int *coloff, const int *count,
                             const int *active, int n_slot,
                             int n_expert, int in_dim, int out_dim, cudaStream_t s);
    void dg_mmq_q8_0_grouped(float *out, const void *wbase, int64_t slab_stride, const int8_t *qx,
                             const float *dx, const int *sx, const int *coloff, const int *count,
                             int n_expert, int in_dim, int out_dim, cudaStream_t s);
    void dg_silu_mul(float *out, const float *gate, const float *up, int64_t n, cudaStream_t s);
    int dg_fp4_moe_grouped_mapped(void* D,const void* A,const void* A_sf,const void* B,const void* B_sf,
        const int* count,const int* coloff,const int* expert_slot,int groups,int N,int K,
        unsigned long long sfA_stride,unsigned long long sfB_stride,const float* alpha,cudaStream_t stream);
}

#define GEMMA4_MOE_TMAX 2048  // max tokens per MoE chunk (decode B≤16, prefill loops in chunks)
static int moe_alloc_scratch(gemma4_engine_t *eng);          // defined with the MoE FFN below

// NVFP4 safetensors loading (FORMAT_NVFP4): container parser, dequant math, name mapping, and
// the fused decode GEMV. All header-only; the decode kernel lives in nvfp4_gemv.cuh.
//
// Style note (rooted in fucina's "lean where it counts" rule): the engine parses GGUF in pure
// C (the gguf_* helpers above — const uint8_t*, no STL) because that path is exercised on the
// hot side. The NVFP4 LOADER, by contrast, leans on std::string/std::vector/unordered_map — but
// ONLY inside gemma4_engine_create and nvfp4_load_from_safetensors, both of which run exactly
// ONCE at startup. A safetensors header is JSON + a sharded index; hand-rolling that in C buys
// nothing but bug surface here. Every per-token/decode path (nvfp4_gemv, the routing in
// decode_layer) stays raw-pointer C-style — no STL ever crosses into the hot loop.
#include "safetensors.h"
#include "nvfp4.h"
#include "nvfp4_loader.h"
#include "nvfp4_gemv.cuh"
#include <thread>
#include <atomic>
#include <algorithm>
#include "fp8_block.cuh"          // M5: DeepSeek block-fp8 decode GEMV (Qwen3.5-9B FP8)
#include "qwen35_fp8_loader.h"    // M5: qwen35 FP8 safetensors → engine key mapping
// =========================================================================
// GGUF File Layout
// =========================================================================
//
// Standard GGUFv3 header:
//   magic       "GGUF" (0x46554747)
//   version     3
//   tensor_count
//   metadata_kv_count
//   metadata_kv[]  (key-value pairs, aligned to 32 bytes)
//   tensor_infos[] (name, n_dims, dims[], offset)
//   tensor_data[]  (padding to 32 bytes, then raw tensor data)
//
// Gemma 4 custom metadata keys:
//   gemma4.context_length        = 262144
//   gemma4.embedding_length      = 3840
//   gemma4.block_count           = 48
//   gemma4.head_count            = 16
//   gemma4.head_count_kv         = 8
//   gemma4.global_head_count_kv  = 1
//   gemma4.head_dim              = 256
//   gemma4.global_head_dim       = 512
//   gemma4.feed_forward_length   = 15360
//   gemma4.sliding_window        = 1024
//   gemma4.attention_k_eq_v      = true
//   gemma4.final_logit_softcapped = 30.0
//   gemma4.layer_types           = [0,0,0,0,0,1, 0,0,0,0,0,1, ...]
//   general.name                 = "Gemma 4 12B"

// =========================================================================
// ─── GGUF Parsing Structures ──────────────────────────────────────────
// =========================================================================

// The real GGUF v3 binary format (see ggml/gguf spec):
//
//   header: magic(u32) version(u32) tensor_count(u64) metadata_kv_count(u64)
//
//   string  := len(u64) + bytes[len]            (NO null terminator, NO padding)
//   kv      := key(string) + type(u32) + value
//   value   := scalar | string | array
//   array   := elem_type(u32) + count(u64) + elements
//
//   KV pairs are packed contiguously: there is NO 32-byte key padding and
//   NO per-entry alignment. After the last KV, the tensor_info array follows:
//
//   tensor_info := name(string) + n_dims(u32) + dims[n_dims](u64)
//                  + ggml_type(u32) + offset(u64)
//
//   After all tensor infos, the file is padded to `general.alignment`
//   (default 32) and the raw tensor data begins. tensor_info.offset is
//   relative to that tensor-data start.

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;        // 0x46554747 "GGUF"
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
} gguf_header_t;
#pragma pack(pop)

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_value_type_t;

// GGML tensor element types we care about (subset).
typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,   // 32-block: fp16 d, fp16 m, 16 nibble bytes (v = d*q + m)
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q4_K = 12,  // 256-superblock k-quant (Unsloth UD dynamic-quant token_embd)
    GGML_TYPE_Q5_K = 13,  // 256-superblock k-quant (some qwen3moe ffn_down_exps slabs)
    GGML_TYPE_Q6_K = 14,
} ggml_type_t;

#define GGUF_DEFAULT_ALIGNMENT 32

// Descriptor returned for array-typed metadata values.
typedef struct {
    uint32_t       elem_type;
    uint64_t       count;
    const uint8_t *data;
} gguf_array_t;

// Descriptor returned for string-typed metadata values.
typedef struct {
    const char *ptr;
    uint64_t    len;
} gguf_str_t;

// Size in bytes of a fixed-width GGUF scalar value type (0 for variable).
static inline uint64_t gguf_scalar_size(uint32_t t) {
    switch (t) {
        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL:   return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:                        return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32:return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64:return 8;
        default: return 0; // STRING / ARRAY are variable
    }
}

// Read a GGUF string at *pp (bounded by end). Returns pointer to the bytes
// (not null-terminated) and its length; advances *pp past the string.
// Returns NULL on overflow.
static inline const char* gguf_read_str(const uint8_t **pp, const uint8_t *end,
                                        uint64_t *len_out) {
    const uint8_t *p = *pp;
    if (p + 8 > end) return NULL;
    uint64_t len; memcpy(&len, p, 8); p += 8;
    if (p + len > end) return NULL;
    const char *s = (const char *)p;
    p += len;
    *pp = p;
    if (len_out) *len_out = len;
    return s;
}

// Compare a GGUF (ptr,len) string against a C string.
static inline int gguf_str_eq(const char *s, uint64_t len, const char *cstr) {
    return strlen(cstr) == len && memcmp(s, cstr, len) == 0;
}

// Skip a single GGUF value of the given type at *pp. Returns 0 on success.
static int gguf_skip_value(const uint8_t **pp, const uint8_t *end, uint32_t vtype) {
    const uint8_t *p = *pp;
    if (vtype == GGUF_TYPE_STRING) {
        if (!gguf_read_str(&p, end, NULL)) return -1;
    } else if (vtype == GGUF_TYPE_ARRAY) {
        if (p + 12 > end) return -1;
        uint32_t at; memcpy(&at, p, 4);
        uint64_t n;  memcpy(&n, p + 4, 8);
        p += 12;
        if (at == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < n; i++)
                if (!gguf_read_str(&p, end, NULL)) return -1;
        } else {
            uint64_t sz = gguf_scalar_size(at);
            if (sz == 0) return -1; // nested arrays not supported in GGUF
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

// Walk the metadata KV block and return a pointer just past it (i.e. the
// start of the tensor_info array). Returns NULL on parse error.
static const uint8_t* gguf_skip_metadata(const uint8_t *data, uint64_t size) {
    const gguf_header_t *hdr = (const gguf_header_t *)data;
    const uint8_t *end = data + size;
    const uint8_t *p = data + sizeof(gguf_header_t);
    for (uint64_t i = 0; i < hdr->metadata_kv_count; i++) {
        if (!gguf_read_str(&p, end, NULL)) return NULL; // key
        if (p + 4 > end) return NULL;
        uint32_t vtype; memcpy(&vtype, p, 4); p += 4;
        if (gguf_skip_value(&p, end, vtype) != 0) return NULL;
    }
    return p;
}

// Compute the absolute file offset where tensor data begins (after the
// tensor_info array, aligned to GGUF_DEFAULT_ALIGNMENT). Returns 0 on error.
static uint64_t gguf_tensor_data_start(const uint8_t *data, uint64_t size) {
    const gguf_header_t *hdr = (const gguf_header_t *)data;
    const uint8_t *end = data + size;
    const uint8_t *p = gguf_skip_metadata(data, size);
    if (!p) return 0;
    // Walk tensor infos
    for (uint64_t t = 0; t < hdr->tensor_count; t++) {
        if (!gguf_read_str(&p, end, NULL)) return 0; // name
        if (p + 4 > end) return 0;
        uint32_t n_dims; memcpy(&n_dims, p, 4); p += 4;
        if (p + (uint64_t)n_dims * 8 + 12 > end) return 0;
        p += (uint64_t)n_dims * 8; // dims
        p += 4;                    // ggml_type
        p += 8;                    // offset
    }
    uint64_t off = (uint64_t)(p - data);
    off = (off + (GGUF_DEFAULT_ALIGNMENT - 1)) & ~(uint64_t)(GGUF_DEFAULT_ALIGNMENT - 1);
    return off;
}

// =========================================================================
// ─── Tensor format conversion helpers ──────────────────────────────────
// =========================================================================

// FP8 E4M3 <-> FP32 conversions (native on Blackwell GB10 sm_121, CUDA 13.0)
// Uses __nv_fp8_storage_t (= unsigned char) and public CUDA 13 conversion APIs
static inline __device__ float fp8_to_float(__nv_fp8_storage_t v) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3)));
}

static inline __device__ __nv_fp8_storage_t float_to_fp8(float v) {
    v = fminf(fmaxf(v, -448.0f), 448.0f);
    return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// KV-cache element type: FP8 E4M3 (1 byte). K/V are post-(QK/V)-RMSNorm and (for K)
// post-RoPE, so O(1) magnitude — well inside E4M3's ±448 range. Storing the cache in
// FP8 cuts KV bytes/token 4× vs fp32 (the dominant decode + long-context cost); the
// flash attention kernels dequantize in-register. Indexing is unchanged (1 byte/elem).
typedef __nv_fp8_storage_t kv_t;

// ─── Unified weight element decode ─────────────────────────────────────
//
// Reads ONE logical weight element from a quantised weight tensor and returns
// it as float.  Supports both on-disk formats:
//
//                    element i is at byte i.
//
//   FORMAT_Q8_0 (1): weight is packed in 34-byte blocks of 32 elements.
//                    block layout: [fp16 scale (2 B)][int8 × 32 (32 B)]
//                    element i belongs to block i/32, lane i%32.
//
// `base` must point to the start of the FULL weight row (out_row × in_dim
// elements), NOT to an individual block.
__device__ __forceinline__ float decode_weight(
    const uint8_t *base, int i, int fmt)
{
    if (fmt == 2 /*FORMAT_Q4_0*/) {
        // Q4_0: block = 2-byte fp16 scale + 16 bytes of 32 nibbles (18 bytes).
        // byte j holds elem[j] in the low nibble and elem[j+16] in the high nibble;
        // value = scale * (nibble - 8).
        int b   = i >> 5;                      // block index (i / 32)
        int j   = i & 31;                      // lane in block (i % 32)
        const uint8_t *blk = base + (size_t)b * 18;
        __half_raw hr; hr.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        float scale = __half2float(__half(hr));
        uint8_t byte = blk[2 + (j & 15)];
        int nib = (j < 16) ? (byte & 0x0F) : (byte >> 4);
        return scale * (float)(nib - 8);
    } else {
        // Q8_0: block = 2-byte fp16 scale + 32 int8 qs
        int b   = i >> 5;                      // block index  (i / 32)
        int j   = i & 31;                      // lane in block (i % 32)
        const uint8_t *blk = base + b * 34;    // 34 bytes per block
        __half_raw hr; hr.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        float scale = __half2float(__half(hr));
        return scale * (float)((int8_t)blk[2 + j]);
    }
}

// Bytes per quantized weight ROW (out_dim rows of in_dim elements). FP8 = 1 B/elem,
// Q8_0 = 34 B / 32-block, Q4_0 = 18 B / 32-block. Used everywhere a kernel steps from
// one output row to the next — must match decode_weight's per-element block size.
__device__ __forceinline__ size_t wrow_bytes(int fmt, int in_dim) {
    if (fmt == 2 /*Q4_0*/) return (size_t)(in_dim/32) * 18;
    return (size_t)(in_dim/32) * 34;   // Q8_0
}

// =========================================================================
// ─── Utility Kernels ────────────────────────────────────────────────────
// =========================================================================

__global__ void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ void copy_f32_kernel(float *dst, const float *src, uint64_t n) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void copy_fp8_to_f32_kernel(
    float *dst,
    const unsigned char *src,
    uint64_t n)
{
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] = fp8_to_float(src[i]);
}

__global__ void copy_f32_to_fp8_kernel(
    unsigned char *dst,
    const float *src,
    uint64_t n)
{
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] = float_to_fp8(kv_codec_value(src, (int)i));
}

// Device-pos KV write for the CUDA-graph decode: the destination slot is computed
// from the device-resident position (*pos_ptr) instead of a host-baked pointer
// offset, so one captured graph replays for every token. Identical math to the
// scalar site's copy_f32_to_fp8_kernel(base + pos*stride, src, n).
__global__ void copy_f32_to_fp8_at_kernel(
    unsigned char *base, const int *pos_ptr, int stride,
    const float *src, int n, int cap)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) base[(size_t)((*pos_ptr) % cap) * stride + i] = float_to_fp8(kv_codec_value(src, i));
}

// Dequantize a weight matrix (Q8_0 or FP8) to BF16, preserving the source's
// row-major [out_dim][in_dim] element order (element e -> dst[e]). Used to build
// the persistent BF16 weight buffer for batched cuBLAS prefill (Step 2). Caller
// must keep per-call n < 2^31 (true for every projection: max = 15360×3840 ≈ 59M).
__global__ void dequant_to_bf16_kernel(
    __nv_bfloat16 *dst, const uint8_t *src, uint64_t n, int fmt)
{
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(decode_weight(src, (int)i, fmt));
}

// Convert f32 -> bf16 (for activations feeding the cuBLAS GEMM).
__global__ void f32_to_bf16_kernel(__nv_bfloat16 *dst, const float *src, uint64_t n) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// Convert f32 -> fp16 (for the cuBLAS GEMMs that read the __half K/V cache in place —
// mixing fp16 A with bf16 B in one GemmEx is not supported, so Q converts to fp16 too).
__global__ void f32_to_f16_kernel(__half *dst, const float *src, uint64_t n) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// Convert bf16 -> f32 (FORMAT_NVFP4: safetensors norm tensors ship BF16; the engine's norm
// stores (d_w_*_norm) are float, exactly like the GGUF path's UPLOAD_NORM destinations).
__global__ void bf16_to_f32_kernel(float *dst, const __nv_bfloat16 *src, uint64_t n) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

// BF16 embedding lookup (FORMAT_NVFP4). Mirror of embed_lookup_q8_0_kernel but the table is a
// dense BF16 [vocab × hidden] matrix (Gemma's UNSCALED embed_tokens — the √hidden scale is
// applied by the caller, identical to the Q8_0 path). One block per row; clamp the token id.
__global__ void embed_lookup_bf16_kernel(
    float               *out,    // [batch × hidden_size]
    const __nv_bfloat16 *table,  // [vocab_size × hidden_size] BF16
    const int32_t       *tokens, // [batch]
    int                  batch,
    int                  hidden_size)
{
    int row = blockIdx.x;
    if (row >= batch) return;
    int token = tokens[row];
    if (token < 0) token = 0;
    if (token >= GEMMA4_VOCAB_SIZE) token = 0;
    const __nv_bfloat16 *emb = table + (size_t)token * hidden_size;
    float *out_row = out + (size_t)row * hidden_size;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        out_row[i] = __bfloat162float(emb[i]);
}

// BF16 LM-head GEMV (FORMAT_NVFP4): logits[v] = Σ_h lmhead_bf16[v,h] · x[h], full-precision
// accumulate over a plain BF16 weight (no block/global scale). This reads the WHOLE 2 GB untied
// BF16 head per token (vocab 262144 × hidden 3840 × 2 B) — a big share of the decode — so like
// nvfp4_gemv it register-blocks the OUTPUT rows: each warp reduces BF16_HEAD_ROWS rows, keeping
// that many accumulators and issuing that many independent weight loads per k-stride while x is
// loaded once and reused (L1-hot). A naive warp-per-row version is latency-bound at ~66 GB/s.
// Capture-safe (no alloc / host sync). gridDim.x = ceil(out_dim / (WARPS*ROWS)).
#ifndef BF16_HEAD_WARPS
#define BF16_HEAD_WARPS 8
#endif
#ifndef BF16_HEAD_ROWS
#define BF16_HEAD_ROWS 4
#endif
__global__ void bf16_head_gemv_kernel(
    float               *__restrict__ y,   // [out_dim]
    const __nv_bfloat16 *__restrict__ w,   // [out_dim × in_dim]
    const float         *__restrict__ x,   // [in_dim]
    int in_dim, int out_dim)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * BF16_HEAD_WARPS + warp) * BF16_HEAD_ROWS;
    if (row0 >= out_dim) return;
    const int nrow = min(BF16_HEAD_ROWS, out_dim - row0);
    float acc[BF16_HEAD_ROWS];
    #pragma unroll
    for (int r = 0; r < BF16_HEAD_ROWS; r++) acc[r] = 0.f;
    for (int k = lane; k < in_dim; k += 32) {
        float xk = x[k];
        #pragma unroll
        for (int r = 0; r < BF16_HEAD_ROWS; r++) {
            if (r >= nrow) break;
            acc[r] += __bfloat162float(w[(size_t)(row0 + r) * in_dim + k]) * xk;
        }
    }
    #pragma unroll
    for (int r = 0; r < BF16_HEAD_ROWS; r++) {
        float a = acc[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0 && r < nrow) y[row0 + r] = a;
    }
}
static inline void bf16_head_gemv_launch(
    float *y, const __nv_bfloat16 *w, const float *x,
    int in_dim, int out_dim, cudaStream_t stream)
{
    const int per_blk = BF16_HEAD_WARPS * BF16_HEAD_ROWS;
    unsigned blocks = (unsigned)((out_dim + per_blk - 1) / per_blk);
    bf16_head_gemv_kernel<<<blocks, 32*BF16_HEAD_WARPS, 0, stream>>>(y, w, x, in_dim, out_dim);
}

// ─── EXACT two-pass greedy head (Q8_0 approx scan + BF16 rescore) ─────────────────────
// The untied BF16 head is 1 GB read per greedy token. Lossy heads flip the 248k-vocab
// argmax (the FP8 per-row head above was measured to degrade generation), so instead:
// (1) a Q8_0 copy of the head (0.53 GB) produces APPROXIMATE logits; (2) every index
// within Q8HEAD_MARGIN of the approx max is collected (cap Q8HEAD_MAXCAND); (3) the
// candidates alone are rescored with the EXACT BF16 rows (<=64 x H = 0.3 MB) and the
// argmax (lowest-index tie-break) is taken over the exact values. Output is BIT-IDENTICAL
// to the full BF16 head as long as the true argmax lands in the candidate set — the
// margin is far above the Q8_0 dot error, and the oracle/self-test gates verify it.
#define Q8HEAD_MARGIN  1.5f
#define Q8HEAD_MAXCAND 64

// Approx logits from the Q8_0 head (34 B / 32-elem block: fp16 scale + 32 int8), float acts.
__global__ void q8_head_gemv_kernel(
    float *__restrict__ y, const unsigned char *__restrict__ w, const float *__restrict__ x,
    int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * nwarps + warp;
    if (row >= out_dim) return;
    int nb = in_dim >> 5;
    const unsigned char *wr = w + (size_t)row * nb * 34;
    float acc = 0.f;
    for (int b = lane; b < nb; b += 32) {
        const unsigned char *blk = wr + (size_t)b * 34;
        __half_raw hs; hs.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        float d = __half2float(__half(hs));
        const int8_t *q = (const int8_t *)(blk + 2);
        const float *xb = x + b * 32;
        float p = 0.f;
        #pragma unroll
        for (int j = 0; j < 32; j++) p += (float)q[j] * xb[j];
        acc += d * p;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if (lane == 0) y[row] = acc;
}

// Pass 1b: per row, find the approx max then collect indices > max - margin (capped).
// cand[row][0..cnt) unordered; cnt clamped to cap. Two block-wide scans over the row.
__global__ void q8_head_candidates_kernel(
    const float *__restrict__ yapprox, int vocab, int *__restrict__ cand, int *__restrict__ cnt)
{
    int row = blockIdx.x;
    const float *yr = yapprox + (size_t)row * vocab;
    __shared__ float smax[32];
    float m = -1e30f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) m = fmaxf(m, yr[i]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = (blockDim.x + 31) >> 5;
    if (lane == 0) smax[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = (lane < nw) ? smax[lane] : -1e30f;
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
        if (lane == 0) smax[0] = m;
    }
    __syncthreads();
    float thr = smax[0] - Q8HEAD_MARGIN;
    if (threadIdx.x == 0) cnt[row] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        if (yr[i] > thr) {
            int p = atomicAdd(&cnt[row], 1);
            if (p < Q8HEAD_MAXCAND) cand[(size_t)row * Q8HEAD_MAXCAND + p] = i;
        }
    }
}

// Pass 2: exact BF16 rescore of the candidates + argmax with lowest-index tie-break.
// One block per row: warp w rescores candidates w, w+nw, ...; block-reduce (val, idx).
__global__ void q8_head_rescore_argmax_kernel(
    const __nv_bfloat16 *__restrict__ wbf, const float *__restrict__ x,
    const int *__restrict__ cand, const int *__restrict__ cnt,
    int in_dim, int *__restrict__ out_idx)
{
    int row = blockIdx.x;
    int n = cnt[row]; if (n > Q8HEAD_MAXCAND) n = Q8HEAD_MAXCAND;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    const float *xr = x + (size_t)row * in_dim;
    float bv = -1e30f; int bi = 0x7fffffff;
    for (int c = warp; c < n; c += nw) {
        int idx = cand[(size_t)row * Q8HEAD_MAXCAND + c];
        const __nv_bfloat16 *wr = wbf + (size_t)idx * in_dim;
        float acc = 0.f;
        for (int k = lane; k < in_dim; k += 32) acc += __bfloat162float(wr[k]) * xr[k];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
        if (acc > bv || (acc == bv && idx < bi)) { bv = acc; bi = idx; }
    }
    __shared__ float sv[32]; __shared__ int si[32];
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (warp == 0) {
        bv = (lane < nw) ? sv[lane] : -1e30f;
        bi = (lane < nw) ? si[lane] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, o);
            int   oi = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) out_idx[row] = bi;
    }
}

// ─── Argmax over vocab_size ───────────────────────────────────────────

__global__ void argmax_kernel(
    const float *logits,    // [vocab_size]
    int         *out_idx,   // scalar output
    int          vocab_size)
{
    // Use warp-level reduction for efficiency on vocab 262144
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float best_val = -1e30f;
    int   best_idx = 0;

    for (int i = idx; i < vocab_size; i += blockDim.x * gridDim.x) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xFFFFFFFF, best_val, offset);
        int   other_idx = __shfl_xor_sync(0xFFFFFFFF, best_idx, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    if (tid == 0) out_idx[0] = best_idx;
}

// ─── Softmax ──────────────────────────────────────────────────────────

__global__ void softmax_kernel(float *x, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Find max
    float max_val = -1e30f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        max_val = fmaxf(max_val, x[i]);
    }
    for (int offset = 32; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }

    // Sum exp
    float sum = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += expf(x[i] - max_val);
    }
    for (int offset = 32; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    // Normalize
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        x[i] = expf(x[i] - max_val) / sum;
    }
}

// =========================================================================
// ─── RMS Norm Kernels ───────────────────────────────────────────────────
// =========================================================================

// RMS Norm on a single vector of size n
// Shared-memory warp-reduction helper (intra-block, all warps).
// Requires a __shared__ float buf[32] visible to the caller.
// Warp-level butterfly sum — the reduced value ends up in ALL 32 lanes (no smem,
// no __syncthreads). Used by the tiled flash-prefill kernels (warp-per-query).
__device__ __forceinline__ float warp_reduce_sum_all(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, o);
    return v;
}

__device__ float block_reduce_sum(float val, float *smem) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    // Warp reduce
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, off);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    // Block reduce from warp accumulators
    int n_warps = (blockDim.x + 31) >> 5;
    val = (lane < n_warps) ? smem[lane] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, off);
    // Barrier before returning so a back-to-back call cannot overwrite smem
    // while another warp is still reading it (racecheck: 1620 hazards without).
    __syncthreads();
    return val;
}

// Block max-reduction, same structure/barrier discipline as block_reduce_sum.
__device__ float block_reduce_max(float val, float *smem) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, off));
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (lane < n_warps) ? smem[lane] : -1e30f;
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, off));
    __syncthreads();
    return val;
}

// RMSNorm: out = x / rms(x) * weight.  weight may be NULL (uses 1.0).
// Launched as: rms_norm_kernel<<<1, 256, 32*sizeof(float)>>>
__global__ void rms_norm_kernel(
    float       *out,
    const float *x,
    const float *weight,
    int          n,
    float        eps)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        ss += x[i] * x[i];
    ss = block_reduce_sum(ss, smem);

    float rms = rsqrtf(ss / n + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        float w = weight ? weight[i] : 1.0f;
        out[i] = x[i] * rms * w;
    }
}

// Per-head RMSNorm: used for Q and K norms in Gemma.
// blockIdx.x = head index; blockDim.x = head_dim (≤512).
// weight has shape [head_dim]; may be NULL.
// Launched as: per_head_rms_norm<<<n_heads, head_dim, 32*sizeof(float)>>>
__global__ void per_head_rms_norm_kernel(
    float       *qk,        // [n_heads × head_dim] in-place
    const float *weight,    // [head_dim]
    int          head_dim,
    float        eps)
{
    extern __shared__ float smem[];
    int head = blockIdx.x;
    int tid  = threadIdx.x;
    float *h = qk + head * head_dim;

    float ss = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x)
        ss += h[i] * h[i];
    ss = block_reduce_sum(ss, smem);

    float rms = rsqrtf(ss / head_dim + eps);
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float w = weight ? weight[i] : 1.0f;
        h[i] = h[i] * rms * w;
    }
}

// Batched RMS Norm: process 'rows' vectors of size 'n' in parallel
__global__ void rms_norm_rows_kernel(
    float       *out,
    const float *x,
    const float *weight,
    int          n,
    int          rows,
    float        eps)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const float *x_row = x + row * n;
    float *out_row = out + row * n;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        ss += x_row[i] * x_row[i];
    ss = block_reduce_sum(ss, smem);
    float rms = rsqrtf(ss / n + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        float w = weight ? weight[i] : 1.0f;
        out_row[i] = x_row[i] * rms * w;
    }
}

// =========================================================================
// ─── FP8 Matrix-Vector Multiply (GEMV, B=1 decode) ─────────────────────
// =========================================================================

// FP8 x FP32 → FP32 GEMV: out[j] = sum_i(weight[i,j] * x[i])
// weight is column-major FP8 [in_dim × out_dim]
// x is FP32 [in_dim]
// ────────────────────────────────────────────────────────────────
// All GEMV kernels below use decode_weight() for both FP8 and Q8_0.
// Launch configuration: blockIdx.x = output row, blockDim.x = 256,
// shared mem = 32 * sizeof(float).
// ────────────────────────────────────────────────────────────────

// ─── dp4a MMVQ for Q8_0 (llama.cpp-parity decode bandwidth) ─────────────
// The per-element gemv reads weights with scalar byte loads (~125-139 GB/s ceiling
// on GB10). MMVQ instead quantizes the activation to int8 and uses __dp4a (4 int8
// MACs / instruction) with wider int loads, reaching ~peak — the same path
// llama.cpp uses for q8_0. Q8_0 is symmetric (zero mean) so no Q8_1 sum-correction
// term is needed: out = Σ_block d_w·d_x · Σ __dp4a(qw, qx).

// Read 4 packed int8 as one int, tolerating the 2-byte alignment of a Q8_0 block's
// qs (it starts at byte offset 2 within the 34-byte block).
static __device__ __forceinline__ int q8_get_int_b2(const void *p, int i32) {
    const uint16_t *x16 = (const uint16_t *)p;
    return (int)x16[2*i32] | ((int)x16[2*i32 + 1] << 16);
}

// Quantize x[in_dim] to symmetric per-32-block int8 + per-block scale. qx[in_dim]
// is 4-byte aligned at every block boundary (32 | offset); dx[in_dim/32]. One warp
// per block. in_dim is a multiple of 32 for every gemma4 projection.
__global__ void quantize_q8_1_kernel(
    const float *x, int8_t *qx, float *dx, int *sx, int in_dim)
{
    int b = blockIdx.x, lane = threadIdx.x;     // 32 threads = one block
    int i = b*32 + lane;
    float v = (i < in_dim) ? x[i] : 0.0f;
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, o));
    float d  = a / 127.0f;
    float id = (d > 0.0f) ? 1.0f / d : 0.0f;
    int q = __float2int_rn(v * id);
    q = max(-127, min(127, q));
    if (i < in_dim) qx[i] = (int8_t)q;
    // Per-block Σ of the int8 activations. Folds the Q4_0 −8 nibble correction
    // (out = dw·dx·(Σ nibble·qx − 8·Σ qx)) so mmvq_q4_0 reads this instead of recomputing
    // Σqx via 8 dp4a/block. Identical integer to the old dp4a(0x01010101,·) sum → bit-exact.
    // in_dim is always a multiple of 32 for gemma4 projections (no partial block). (#6b.)
    int qsum = q;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) qsum += __shfl_xor_sync(0xFFFFFFFF, qsum, o);
    if (lane == 0) { dx[b] = d; sx[b] = qsum; }
}

// Transposed-layout variant of quantize_q8_1_kernel for the packed-Q4_K batched decode GEMV.
// Identical quantization math and dx/sx layout; ONLY the qx byte addresses change. The batched
// mixer kernel was measured LSU-bound: with the row-major [n][in_dim] layout each of its
// activation loads has a 32-byte lane stride (8 sector replays per LDG.128). This layout stores
// each token row as epochs of 32 windows so that a warp reading "window b = lane+32p, half h"
// is one DENSE LDG.128 (byte c of window b, token n → n*in_dim + (b>>5)*1024 + (c>>4)*512 +
// (b&31)*16 + (c&15)). Requires in_dim % 1024 == 0 (full 32-window epochs); callers gate.
__global__ void quantize_q8_1t_kernel(
    const float *x, int8_t *qxT, float *dx, int *sx, int in_dim)
{
    int b = blockIdx.x, lane = threadIdx.x;     // 32 threads = one 32-elem window
    int i = b*32 + lane;
    float v = x[i];
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, o));
    float d  = a / 127.0f;
    float id = (d > 0.0f) ? 1.0f / d : 0.0f;
    int q = __float2int_rn(v * id);
    q = max(-127, min(127, q));
    int nb32 = in_dim >> 5;
    int n = b / nb32, wb = b - n*nb32;          // token, window-within-token
    qxT[(size_t)n*in_dim + (size_t)(wb >> 5)*1024 + (lane >> 4)*512 + (size_t)(wb & 31)*16
        + (lane & 15)] = (int8_t)q;
    int qsum = q;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) qsum += __shfl_xor_sync(0xFFFFFFFF, qsum, o);
    if (lane == 0) { dx[b] = d; sx[b] = qsum; }
}

// BF16-input variant of quantize_q8_1_kernel: the prefill projection activation
// already lives as BF16 in d_inb (rms-norm / attn-out / geglu outputs), so the
// tiled-MMQ prefill path quantizes straight from BF16 instead of needing a FP32
// copy. Same per-32-block symmetric int8 + Σqx layout; one warp per block. The
// only difference is the load (BF16→float); the math is identical to the FP32
// kernel above, so MMQ prefill and the dp4a decode path are numerically aligned.
__global__ void quantize_q8_1_bf16_kernel(
    const __nv_bfloat16 *x, int8_t *qx, float *dx, int *sx, int in_dim)
{
    int b = blockIdx.x, lane = threadIdx.x;     // 32 threads = one block
    int i = b*32 + lane;
    float v = (i < in_dim) ? __bfloat162float(x[i]) : 0.0f;
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, o));
    float d  = a / 127.0f;
    float id = (d > 0.0f) ? 1.0f / d : 0.0f;
    int q = __float2int_rn(v * id);
    q = max(-127, min(127, q));
    if (i < in_dim) qx[i] = (int8_t)q;
    int qsum = q;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) qsum += __shfl_xor_sync(0xFFFFFFFF, qsum, o);
    if (lane == 0) { dx[b] = d; sx[b] = qsum; }
}

// MMVQ (warp-per-row): out[idx] = Σ_block d_w_b · d_x_b · Σ_k __dp4a(weight_qs[k], act_qs[k]).
// Each WARP owns one output row; its 32 lanes stride the nb blocks and reduce via __shfl
// (no block_reduce_sum / __syncthreads). nwarps rows per block. (DECODE-30-35 Step 5.)
__global__ void mmvq_q8_0_kernel(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx,
    int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int nb = in_dim >> 5;
    const uint8_t *wrow = weight + (size_t)idx * (size_t)nb * 34;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t *blk = wrow + (size_t)b * 34;
        __half_raw hr; hr.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        float dw = __half2float(__half(hr));
        const void *wqs = blk + 2;                       // weight int8 (2-byte aligned)
        const int  *xqs = (const int *)(qx + (size_t)b * 32);  // act int8 (4-byte aligned)
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++)
            sumi = __dp4a(q8_get_int_b2(wqs, k), xqs[k], sumi);
        acc += dw * dx[b] * (float)sumi;
    }
    acc = warp_reduce_sum_all(acc);
    if (lane == 0) out[idx] = acc;
}

// MMVQ for Q4_0 weights (the QAT 4-bit layers). Q4_0 block = fp16 scale + 16 bytes of
// 32 nibbles; value = dw*(nibble-8). The -8 offset makes it asymmetric, so unlike Q8_0
// we add a correction: out = dw·dx·(Σ nibble·qx − 8·Σ qx). The activation is the SAME
// per-block int8 quantization as the Q8_0 path (quantize_q8_1_kernel); the block sum is
// computed here from qx via dp4a(0x01010101,·). nibble layout matches llama.cpp
// (byte j → elem j low, elem j+16 high), read 4-bytes-at-a-time (8 nibbles).
__global__ void mmvq_q4_0_kernel(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int nb = in_dim >> 5;
    const uint8_t *wrow = weight + (size_t)idx * (size_t)nb * 18;
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t *blk = wrow + (size_t)b * 18;
        __half_raw hr; hr.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        float dw = __half2float(__half(hr));
        const void *wqs = blk + 2;                            // 16 nibble-bytes (2-byte aligned)
        const int  *xqs = (const int *)(qx + (size_t)b * 32); // 32 act int8 (4-byte aligned)
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int w   = q8_get_int_b2(wqs, k);     // 4 bytes = 8 nibbles
            int vlo = w & 0x0F0F0F0F;            // elems 4k..4k+3   (raw nibble 0..15)
            int vhi = (w >> 4) & 0x0F0F0F0F;     // elems 4k+16..4k+19
            sumi = __dp4a(vlo, xqs[k],     sumi);
            sumi = __dp4a(vhi, xqs[k + 4], sumi);
        }
        acc += dw * dx[b] * (float)(sumi - 8 * sx[b]);   // sx[b] = Σqx (precomputed in quantize)
    }
    acc = warp_reduce_sum_all(acc);
    if (lane == 0) out[idx] = acc;
}

// Batched MMVQ for Q4_0 weights: the dp4a analogue of gemv_batched_kernel_t. Reads each
// weight ROW once (12.65 GB / pass), decodes the 32 nibbles once, then reuses them across
// all NK quantized activation vectors via dp4a — so NK spec-draft tokens cost ~one token's
// weight bandwidth AT dp4a speed (~207 GB/s), not the scalar byte-load ceiling (~125 GB/s)
// that decode_weight hits. qx/dx are token-major: qx[n*in_dim + i], dx[n*(in_dim/32) + b],
// matching quantize_q8_1_kernel run over the whole [NK × in_dim] activation block at once.
// Per-block correction mirrors mmvq_q4_0_kernel: out = Σ_b dw·dx·(Σ nibble·qx − 8·Σ qx).
template<int NK>
__global__ void mmvq_q4_0_batched_kernel(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int nb = in_dim >> 5;
    const uint8_t *wrow = weight + (size_t)idx * (size_t)nb * 18;
    float acc[NK];
    #pragma unroll
    for (int n = 0; n < NK; n++) acc[n] = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t *blk = wrow + (size_t)b * 18;
        __half_raw hr; hr.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        float dw = __half2float(__half(hr));
        const void *wqs = blk + 2;                       // 16 nibble-bytes (2-byte aligned)
        int wv[8];                                       // decode the 32 nibbles ONCE
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int w = q8_get_int_b2(wqs, k);
            wv[2*k]   =  w        & 0x0F0F0F0F;          // elems 4k..4k+3   (low nibble)
            wv[2*k+1] = (w >> 4)  & 0x0F0F0F0F;          // elems 4k+16..4k+19 (high nibble)
        }
        #pragma unroll
        for (int n = 0; n < NK; n++) {
            const int *xqs = (const int *)(qx + (size_t)n*in_dim + (size_t)b*32);
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                sumi = __dp4a(wv[2*k],   xqs[k],     sumi);
                sumi = __dp4a(wv[2*k+1], xqs[k + 4], sumi);
            }
            acc[n] += dw * dx[(size_t)n*nb + b] * (float)(sumi - 8*sx[(size_t)n*nb + b]);
        }
    }
    #pragma unroll
    for (int n = 0; n < NK; n++) {
        float s = warp_reduce_sum_all(acc[n]);
        if (lane == 0) out[(size_t)n*out_dim + idx] = s;
    }
}

// Batched MMVQ for Q8_0 weights — same weight-row-reuse idea, no nibble unpack / -8 term.
template<int NK>
__global__ void mmvq_q8_0_batched_kernel(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx,
    int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int nb = in_dim >> 5;
    const uint8_t *wrow = weight + (size_t)idx * (size_t)nb * 34;
    float acc[NK];
    #pragma unroll
    for (int n = 0; n < NK; n++) acc[n] = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const uint8_t *blk = wrow + (size_t)b * 34;
        __half_raw hr; hr.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        float dw = __half2float(__half(hr));
        const void *wqs = blk + 2;                       // 32 weight int8 (2-byte aligned)
        int wv[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) wv[k] = q8_get_int_b2(wqs, k);
        #pragma unroll
        for (int n = 0; n < NK; n++) {
            const int *xqs = (const int *)(qx + (size_t)n*in_dim + (size_t)b*32);
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) sumi = __dp4a(wv[k], xqs[k], sumi);
            acc[n] += dw * dx[(size_t)n*nb + b] * (float)sumi;
        }
    }
    #pragma unroll
    for (int n = 0; n < NK; n++) {
        float s = warp_reduce_sum_all(acc[n]);
        if (lane == 0) out[(size_t)n*out_dim + idx] = s;
    }
}

// Dispatch the batched MMVQ (dp4a) to the compile-time-NK kernel. qx/dx must already hold
// the NK quantized activation vectors (token-major). K ≤ 8 → one weight pass; K > 8 splits
// into ≤8-wide chunks (each re-reads the weight, still far cheaper than K scalar decodes).
static void mmvq_batched_launch(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim, int K, int fmt, cudaStream_t stream)
{
    const int NWARPS = 8; int b = NWARPS*32;             // warp-per-row (Step 5): no smem
    dim3 g((out_dim + NWARPS - 1) / NWARPS);
    #define LAUNCH(NK,O)                                                                \
        do { if (fmt == 2)                                                              \
            mmvq_q4_0_batched_kernel<NK><<<g,b,0,stream>>>(out+(size_t)(O)*out_dim,weight,qx+(size_t)(O)*in_dim,dx+(size_t)(O)*(in_dim>>5),sx+(size_t)(O)*(in_dim>>5),in_dim,out_dim); \
        else                                                                           \
            mmvq_q8_0_batched_kernel<NK><<<g,b,0,stream>>>(out+(size_t)(O)*out_dim,weight,qx+(size_t)(O)*in_dim,dx+(size_t)(O)*(in_dim>>5),in_dim,out_dim); \
        } while (0)
    switch (K) {
        case 1: LAUNCH(1,0); return;  case 2: LAUNCH(2,0); return;
        case 3: LAUNCH(3,0); return;  case 4: LAUNCH(4,0); return;
        case 5: LAUNCH(5,0); return;  case 6: LAUNCH(6,0); return;
        case 7: LAUNCH(7,0); return;  case 8: LAUNCH(8,0); return;
        default: break;
    }
    // K>8: 32-row groups (32x weight-read amortization for wide prefill), then 8, then remainder.
    int o = 0;
    for (; o + 32 <= K; o += 32) LAUNCH(32, o);
    for (; o + 8  <= K; o += 8)  LAUNCH(8,  o);
    int rem = K - o;
    switch (rem) {
        case 1: LAUNCH(1,o); break;  case 2: LAUNCH(2,o); break;  case 3: LAUNCH(3,o); break;
        case 4: LAUNCH(4,o); break;  case 5: LAUNCH(5,o); break;  case 6: LAUNCH(6,o); break;
        case 7: LAUNCH(7,o); break;  default: break;
    }
    #undef LAUNCH
}

// ─── FUCINA_PACKED: repacked-Q4_0 decode GEMV (coalesced uint4 loads) ─────────────────────
// Repack one projection's native Q4_0 blocks (18 B = fp16 scale ‖ 16 nibble bytes) into a
// structure-of-arrays: [out_dim·nb × 16 quant bytes] then [out_dim·nb × fp16 scale]. Block
// i's quants land 16-B aligned at quants+i·16; its scale at scales[i] (raw fp16 bits). One
// thread per block; bandwidth-bound but runs ONCE at load. Total bytes per projection equal
// the native 18·nb·out_dim, so packed offsets mirror native offsets (no overlap).
__global__ void repack_q4_0_kernel(
    const uint8_t *src, uint8_t *quants, uint16_t *scales, size_t nblocks)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblocks) return;
    const uint8_t *blk = src + i * 18;
    scales[i] = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));   // raw fp16 scale bits
    uint8_t *d = quants + i * 16;
    #pragma unroll
    for (int k = 0; k < 16; k++) d[k] = blk[2 + k];            // 32 packed 4-bit weights
}

// Batched MMVQ over the REPACKED layout. Identical to mmvq_q4_0_batched_kernel — same warp-
// per-row, same nibble unpack, same dp4a + (Σnibble − 8·Σqx) fold, same output — except each
// block's 16 quant bytes arrive via one aligned uint4 load and the scale from a separate
// 2-B array. Bit-for-bit equal results; the only change is memory-access granularity.
template<int NK>
__global__ void mmvq_q4_0_packed_batched_kernel(
    float *out, const uint8_t *quants, const uint16_t *scales,
    const int8_t *qx, const float *dx, const int *sx, int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int nb = in_dim >> 5;
    const uint8_t  *qrow = quants + (size_t)idx * nb * 16;
    const uint16_t *srow = scales + (size_t)idx * nb;
    float acc[NK];
    #pragma unroll
    for (int n = 0; n < NK; n++) acc[n] = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        uint4 q = *(const uint4 *)(qrow + (size_t)b * 16);   // 128-bit coalesced load
        __half_raw hr; hr.x = srow[b];
        float dw = __half2float(__half(hr));
        int wv[8];                                            // unpack 32 nibbles ONCE
        int w0 = (int)q.x, w1 = (int)q.y, w2 = (int)q.z, w3 = (int)q.w;
        wv[0] = w0 & 0x0F0F0F0F; wv[1] = (w0 >> 4) & 0x0F0F0F0F;
        wv[2] = w1 & 0x0F0F0F0F; wv[3] = (w1 >> 4) & 0x0F0F0F0F;
        wv[4] = w2 & 0x0F0F0F0F; wv[5] = (w2 >> 4) & 0x0F0F0F0F;
        wv[6] = w3 & 0x0F0F0F0F; wv[7] = (w3 >> 4) & 0x0F0F0F0F;
        #pragma unroll
        for (int n = 0; n < NK; n++) {
            const int *xqs = (const int *)(qx + (size_t)n*in_dim + (size_t)b*32);
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                sumi = __dp4a(wv[2*k],   xqs[k],     sumi);
                sumi = __dp4a(wv[2*k+1], xqs[k + 4], sumi);
            }
            acc[n] += dw * dx[(size_t)n*nb + b] * (float)(sumi - 8*sx[(size_t)n*nb + b]);
        }
    }
    #pragma unroll
    for (int n = 0; n < NK; n++) {
        float s = warp_reduce_sum_all(acc[n]);
        if (lane == 0) out[(size_t)n*out_dim + idx] = s;
    }
}

// Dispatch the packed batched MMVQ to its compile-time-NK kernel (K ≤ 8 → one weight pass;
// K > 8 splits into ≤8-wide chunks, each re-reading the weight). Q4_0 only.
static void mmvq_q4_0_packed_batched_launch(
    float *out, const uint8_t *quants, const uint16_t *scales,
    const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim, int K, cudaStream_t stream)
{
    const int NWARPS = 8; int b = NWARPS*32;
    dim3 g((out_dim + NWARPS - 1) / NWARPS);
    #define LP(NK) mmvq_q4_0_packed_batched_kernel<NK><<<g,b,0,stream>>>( \
        out, quants, scales, qx, dx, sx, in_dim, out_dim)
    switch (K) {
        case 1: LP(1); break;  case 2: LP(2); break;
        case 3: LP(3); break;  case 4: LP(4); break;
        case 5: LP(5); break;  case 6: LP(6); break;
        case 7: LP(7); break;  case 8: LP(8); break;
        default:
            for (int o = 0; o < K; o += 8) {
                int kk = (K - o < 8) ? (K - o) : 8;
                mmvq_q4_0_packed_batched_launch(out + (size_t)o*out_dim, quants, scales,
                    qx + (size_t)o*in_dim, dx + (size_t)o*(in_dim>>5),
                    sx + (size_t)o*(in_dim>>5), in_dim, out_dim, kk, stream);
            }
    }
    #undef LP
}

// ─── Tiled MMQ GEMM for Q4_0 (prefill projections, dp4a) ────────────────────────────────
// Y[out_dim × N] = W_q4_0[out_dim × in_dim] · X_int8[in_dim × N], token-major output
// (out[col*out_dim + row]). A classic shared-memory tiled GEMM: each block computes a
// BM×BN output tile, looping over K in 32-elem Q4_0 blocks. Per K-step it cooperatively
// stages ONE block of BM weight rows and BN activation columns into smem (both coalesced),
// then every thread computes a 4×4 micro-tile from smem. This caches BOTH operands, so —
// unlike mmvq_q4_0_batched_kernel (weight re-read every ≤8 cols) and unlike a warp-per-row
// scheme (activation re-read per row-block, uncoalesced) — weight is re-read only
// ⌈N/BN⌉× and activation only ⌈out_dim/BM⌉×, both from coalesced bursts. dp4a math + the
// −8·Σqx correction match mmvq_q4_0_kernel; only the K-reduction order differs (per-block
// float accumulate), so results agree to float rounding, not bit-exactly.
#define MMQ_BM 64
#define MMQ_BN 64
__global__ void mmq_q4_0_tiled_kernel(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim, int N)
{
    __shared__ int   Ws[MMQ_BM][8];   // decoded weight nibbles (raw 0..15), 4 per int
    __shared__ int   Xs[MMQ_BN][8];   // activation int8, 4 per int
    __shared__ float Wd[MMQ_BM];      // per-row weight block scale
    __shared__ float Xd[MMQ_BN];      // per-col activation block scale
    __shared__ int   Xq[MMQ_BN];      // per-col Σ activation int8 (−8 fold)

    const int nb = in_dim >> 5;
    const int rowbase = blockIdx.x * MMQ_BM;
    const int colbase = blockIdx.y * MMQ_BN;
    const int t = threadIdx.x;        // 0..255 (16×16 → 4×4 micro-tile per thread)
    const int tx = t & 15, ty = t >> 4;

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    for (int b = 0; b < nb; b++) {
        // Stage W tile: 256 threads = 64 rows × 4 words; each decodes 8 nibbles → 2 ints.
        {
            int r = t >> 2, wk = t & 3, row = rowbase + r;
            if (row < out_dim) {
                const uint8_t *blk = weight + ((size_t)row*nb + b)*18;
                int w = q8_get_int_b2(blk + 2, wk);
                Ws[r][wk]     =  w        & 0x0F0F0F0F;
                Ws[r][wk + 4] = (w >> 4)  & 0x0F0F0F0F;
                if (wk == 0) {
                    __half_raw hr; hr.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
                    Wd[r] = __half2float(__half(hr));
                }
            } else { Ws[r][wk] = 0; Ws[r][wk + 4] = 0; if (wk == 0) Wd[r] = 0.0f; }
        }
        // Stage X tile: 256 threads = 64 cols × 4; each loads 2 of the 8 ints.
        {
            int c = t >> 2, xk = t & 3, col = colbase + c;
            if (col < N) {
                const int *xqs = (const int *)(qx + (size_t)col*in_dim + (size_t)b*32);
                Xs[c][xk*2]     = xqs[xk*2];
                Xs[c][xk*2 + 1] = xqs[xk*2 + 1];
                if (xk == 0) { Xd[c] = dx[(size_t)col*nb + b]; Xq[c] = sx[(size_t)col*nb + b]; }
            } else { Xs[c][xk*2] = 0; Xs[c][xk*2 + 1] = 0; if (xk == 0) { Xd[c] = 0.0f; Xq[c] = 0; } }
        }
        __syncthreads();

        // Micro-tile: integer dp4a over this block, then fold the per-block scales.
        int sumi[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int j = 0; j < 4; j++) sumi[i][j] = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int wv[4], xv[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) wv[i] = Ws[ty*4 + i][k];
            #pragma unroll
            for (int j = 0; j < 4; j++) xv[j] = Xs[tx*4 + j][k];
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) sumi[i][j] = __dp4a(wv[i], xv[j], sumi[i][j]);
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float dw = Wd[ty*4 + i];
            #pragma unroll
            for (int j = 0; j < 4; j++)
                acc[i][j] += dw * Xd[tx*4 + j] * (float)(sumi[i][j] - 8*Xq[tx*4 + j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int row = rowbase + ty*4 + i;
        if (row >= out_dim) continue;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int col = colbase + tx*4 + j;
            if (col < N) out[(size_t)col*out_dim + row] = acc[i][j];
        }
    }
}

// Largest N the tiled-MMQ prefill path serves. Above this the BF16 tensor-core GEMM
// (+ pipelined dequant) wins (dp4a is compute-bound there), so the prefill loops route to
// it. Tuned empirically against the BF16 path on GB10.
#define GEMMA4_MMQ_MAX_N 1024

// Launch the tiled Q4_0 MMQ. Grid tiles the [out_dim × N] output into BM×BN blocks; weight
// is re-read ⌈N/BN⌉× and activation ⌈out_dim/BM⌉×, both coalesced. Caller guarantees Q4_0.
static void mmq_q4_0_launch(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim, int N, cudaStream_t stream)
{
    dim3 g((unsigned)((out_dim + MMQ_BM - 1) / MMQ_BM),
           (unsigned)((N + MMQ_BN - 1) / MMQ_BN));
    mmq_q4_0_tiled_kernel<<<g, 256, 0, stream>>>(out, weight, qx, dx, sx, in_dim, out_dim, N);
}

// ─── Tiled MMQ GEMM for PACKED Q4_K (prefill projections, dp4a) ──────────────────────────
// Y[out_dim × N] = W_q4_k[out_dim × in_dim] · X_int8[in_dim × N], token-major output. The
// Q4_K analogue of mmq_q4_0_tiled_kernel: identical shared-memory tiled structure (BM×BN
// tile, K looped in 32-elem blocks, 4×4 micro-tile per thread, dp4a integer dot), so it
// reads the native (de-interleaved/PACKED) Q4_K weight ONCE — no BF16 materialize, no
// per-projection full-model dequant (the un-pipelined ~1244 ms/512-tok floor of the Qwen3
// prefill). REQUIRES the PACKED layout (build_packed_q4k): each 256-elem superblock is 144 B
// = header[d,dmin,scales(12)] at [0..15] then 8 sub-blocks of 16 de-interleaved bytes at
// [16+j*16 .. +15] (GGML-Q4_0 nibble convention, byte m = nib(elem m)|nib(elem m+16)<<4) —
// so one int word's low/high nibbles map to natural elem order exactly like Q4_0.
//
// (q4k_scale_min — the ggml get_scale_min_k4 6-bit (scale,min) unpack — is defined with the
// Q4_K mmvq kernels further below; forward-declared here.)
__device__ __forceinline__ void q4k_scale_min(const uint8_t *sc, int j, int *s, int *m);
// Single-token warp-per-row MMVQ launch (Step 5). NWARPS output rows per block, no smem.
static inline void mmvq_launch(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim, int fmt, cudaStream_t stream)
{
    const int NWARPS = 8; int b = NWARPS*32;
    int g = (out_dim + NWARPS - 1) / NWARPS;
    if (fmt == 2 /*Q4_0*/)
        mmvq_q4_0_kernel<<<g, b, 0, stream>>>(out, weight, qx, dx, sx, in_dim, out_dim);
    else
        mmvq_q8_0_kernel<<<g, b, 0, stream>>>(out, weight, qx, dx, in_dim, out_dim);
}

// ── Q6_K matvec for the native tied LM head (DECODE-30-35 Step 8, dp4a rework) ───────────
// The QAT model's tied token_embd/LM-head is Q6_K. Reading it NATIVELY (0.82 B/elem) instead
// of the load-time Q8_0 upconvert (1.06 B/elem) cuts ~0.24 GB off every token's V×H output
// projection — the LM head is ~15% of the per-token weight traffic. The first (fp32 scalar)
// version of this kernel was COMPUTE-bound (~76 GB/s: per-element 6-bit unpack + fp32 FMA,
// one whole superblock per lane so loads were 210-byte-strided/uncoalesced) and lost to the
// Q8_0 dp4a fallback. This version is the same dp4a/int8-activation form as the Q8_0/Q4_0
// MMVQ paths (llama.cpp vec_dot_q6_K_q8_1 equivalent): lanes stride 32-elem sub-blocks
// (adjacent lanes touch adjacent 32-byte ql regions → coalesced), values are built as int8
// (q-32 folded in via __vsubss4, exact: 0..63 minus 32 never saturates) and dotted with the
// quantize_q8_1_kernel int8 activation via __dp4a, with the per-16-elem int8 scales applied
// to the two 16-elem dp4a partial sums.
//
// Q6_K super-block = 256 elems: ql[128] | qh[64] | scales[16](int8) | d(fp16) = 210 B.
// Sub-block jj (0..7) of a superblock = half jj>>2, slot jj&3; elem l (0..31) of the slot:
//   q = ((slot<2 ? ql[base+l]&0xF : ql[base+l]>>4) | ((qh[l] >> 2*slot) & 3) << 4) - 32
//   base = (slot&1)*32, scale = sc[half*8 + slot*2 + (l>>4)]
// which matches the fp32 kernel this replaces / ggml dequantize_row_q6_K exactly (the
// scalar form indexed the same bytes as q1/q2/q3/q4 with is = l>>4).
__global__ void mmvq_q6_k_kernel(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx,
    int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int n_super = in_dim >> 8;                       // in_dim / 256
    int nb32 = in_dim >> 5;                          // 32-elem activation blocks
    const uint8_t *wrow = weight + (size_t)idx * (size_t)n_super * 210;
    float acc = 0.0f;
    for (int b = lane; b < nb32; b += 32) {
        const uint8_t *blk = wrow + (size_t)(b >> 3) * 210;
        int jj   = b & 7;
        int half = jj >> 2, slot = jj & 3;
        const uint8_t *qlp = blk + half*64 + (slot & 1)*32;
        const uint8_t *qhp = blk + 128 + half*32;
        const int8_t  *sc  = (const int8_t *)(blk + 192) + half*8 + slot*2;
        __half_raw hr; hr.x = (uint16_t)(blk[208] | ((uint16_t)blk[209] << 8));
        float d = __half2float(__half(hr));
        const int *xqs = (const int *)(qx + (size_t)b * 32);
        int shift = slot * 2;
        int sumi0 = 0, sumi1 = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int qlw = q8_get_int_b2(qlp, k);
            int qhw = q8_get_int_b2(qhp, k);
            int nib = (slot < 2) ? (qlw & 0x0F0F0F0F) : ((qlw >> 4) & 0x0F0F0F0F);
            int w   = __vsubss4(nib | (((qhw >> shift) & 0x03030303) << 4), 0x20202020);
            if (k < 4) sumi0 = __dp4a(w, xqs[k], sumi0);
            else       sumi1 = __dp4a(w, xqs[k], sumi1);
        }
        acc += d * dx[b] * (float)(sc[0]*sumi0 + sc[1]*sumi1);
    }
    acc = warp_reduce_sum_all(acc);
    if (lane == 0) out[idx] = acc;
}

// Batched Q6_K matvec (spec-verify LM head): unpack each 32-elem sub-block ONCE, dp4a it
// against all NK activation rows — the head is read once per verify pass, like Q8_0/Q4_0.
template<int NK>
__global__ void mmvq_q6_k_batched_kernel(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx,
    int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int n_super = in_dim >> 8;
    int nb32 = in_dim >> 5;
    const uint8_t *wrow = weight + (size_t)idx * (size_t)n_super * 210;
    float acc[NK];
    #pragma unroll
    for (int n = 0; n < NK; n++) acc[n] = 0.0f;
    for (int b = lane; b < nb32; b += 32) {
        const uint8_t *blk = wrow + (size_t)(b >> 3) * 210;
        int jj   = b & 7;
        int half = jj >> 2, slot = jj & 3;
        const uint8_t *qlp = blk + half*64 + (slot & 1)*32;
        const uint8_t *qhp = blk + 128 + half*32;
        const int8_t  *sc  = (const int8_t *)(blk + 192) + half*8 + slot*2;
        __half_raw hr; hr.x = (uint16_t)(blk[208] | ((uint16_t)blk[209] << 8));
        float d = __half2float(__half(hr));
        int shift = slot * 2;
        int sumi0[NK], sumi1[NK];
        #pragma unroll
        for (int n = 0; n < NK; n++) { sumi0[n] = 0; sumi1[n] = 0; }
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int qlw = q8_get_int_b2(qlp, k);
            int qhw = q8_get_int_b2(qhp, k);
            int nib = (slot < 2) ? (qlw & 0x0F0F0F0F) : ((qlw >> 4) & 0x0F0F0F0F);
            int w   = __vsubss4(nib | (((qhw >> shift) & 0x03030303) << 4), 0x20202020);
            #pragma unroll
            for (int n = 0; n < NK; n++) {
                int xv = *(const int *)(qx + (size_t)n*in_dim + (size_t)b*32 + k*4);
                if (k < 4) sumi0[n] = __dp4a(w, xv, sumi0[n]);
                else       sumi1[n] = __dp4a(w, xv, sumi1[n]);
            }
        }
        #pragma unroll
        for (int n = 0; n < NK; n++)
            acc[n] += d * dx[(size_t)n*nb32 + b] * (float)(sc[0]*sumi0[n] + sc[1]*sumi1[n]);
    }
    #pragma unroll
    for (int n = 0; n < NK; n++) { float v = warp_reduce_sum_all(acc[n]); if (lane==0) out[(size_t)n*out_dim+idx] = v; }
}

static inline void mmvq_q6_k_launch(
    float *out, const uint8_t *w, const int8_t *qx, const float *dx,
    int in_dim, int out_dim, cudaStream_t stream)
{
    const int NWARPS = 8; int b = NWARPS*32; int g = (out_dim + NWARPS - 1) / NWARPS;
    mmvq_q6_k_kernel<<<g, b, 0, stream>>>(out, w, qx, dx, in_dim, out_dim);
}

static void mmvq_q6_k_batched_launch(
    float *out, const uint8_t *w, const int8_t *qx, const float *dx,
    int in_dim, int out_dim, int K, cudaStream_t stream)
{
    const int NWARPS = 8; int b = NWARPS*32; dim3 g((out_dim + NWARPS - 1) / NWARPS);
    #define LQ6(NK) mmvq_q6_k_batched_kernel<NK><<<g,b,0,stream>>>(out,w,qx,dx,in_dim,out_dim)
    switch (K) {
        case 1: LQ6(1); break; case 2: LQ6(2); break; case 3: LQ6(3); break; case 4: LQ6(4); break;
        case 5: LQ6(5); break; case 6: LQ6(6); break; case 7: LQ6(7); break; case 8: LQ6(8); break;
        default:
            for (int o = 0; o < K; o += 8) {
                int kk = (K - o < 8) ? (K - o) : 8;
                mmvq_q6_k_batched_launch(out + (size_t)o*out_dim, w,
                                         qx + (size_t)o*in_dim, dx + (size_t)o*(in_dim>>5),
                                         in_dim, out_dim, kk, stream);
            }
    }
    #undef LQ6
}

// ─── Native Q4_K matvec (tied LM head, Unsloth UD models) ───────────────────────────────
// Q4_K super-block = 256 elems (144 B): fp16 d, fp16 dmin, 12 B packed 6-bit scales+mins
// (8 sub-blocks of 32), then 128 B of 4-bit quants. Sub-block j: value = d·s_j·q − dmin·m_j
// (ASYMMETRIC — separate per-sub-block min), so unlike Q4_0/Q6_K it also needs Σ activation
// per block: y = Σ_j [ d·s_j·Σ(q·xq)·dx − dmin·m_j·Σ(xq)·dx ]. dp4a gives Σ(q·xq); sx[b]=Σ(xq).
// Reading the head natively (4.5-bit) instead of the Q8_0 upconvert (8-bit) saves ~0.6 GB/token.
// ggml get_scale_min_k4 packing for the 6-bit (scale,min) of sub-block j.
__device__ __forceinline__ void q4k_scale_min(const uint8_t *sc, int j, int *s, int *m) {
    if (j < 4) { *s = sc[j] & 63; *m = sc[j + 4] & 63; }
    else { *s = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
           *m = (sc[j + 4] >> 4)   | ((sc[j]     >> 6) << 4); }
}

__global__ void mmvq_q4_k_kernel(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int n_super = in_dim >> 8;                        // in_dim / 256
    int nb32 = in_dim >> 5;                           // 32-elem activation blocks
    const uint8_t *wrow = weight + (size_t)idx * (size_t)n_super * 144;
    float acc = 0.0f;
    for (int b = lane; b < nb32; b += 32) {
        const uint8_t *blk = wrow + (size_t)(b >> 3) * 144;
        int j = b & 7;                               // sub-block within the superblock
        __half_raw hd; hd.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        __half_raw hm; hm.x = (uint16_t)(blk[2] | ((uint16_t)blk[3] << 8));
        float d = __half2float(__half(hd)), dmin = __half2float(__half(hm));
        int s, m; q4k_scale_min(blk + 4, j, &s, &m);
        const uint8_t *qbase = blk + 16 + (size_t)(j >> 1) * 32;   // 32 quant bytes (pair shares)
        int shift = (j & 1) ? 4 : 0;
        const int *xqs = (const int *)(qx + (size_t)b * 32);
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int qw  = q8_get_int_b2(qbase, k);
            int nib = (shift ? (qw >> 4) : qw) & 0x0F0F0F0F;
            sumi = __dp4a(nib, xqs[k], sumi);
        }
        acc += dx[b] * (d * (float)s * (float)sumi - dmin * (float)m * (float)sx[b]);
    }
    acc = warp_reduce_sum_all(acc);
    if (lane == 0) out[idx] = acc;
}

// Batched Q4_K matvec (spec-verify LM head): unpack each sub-block ONCE, dp4a vs all NK rows.
template<int NK>
__global__ void mmvq_q4_k_batched_kernel(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int n_super = in_dim >> 8;
    int nb32 = in_dim >> 5;
    const uint8_t *wrow = weight + (size_t)idx * (size_t)n_super * 144;
    float acc[NK];
    #pragma unroll
    for (int n = 0; n < NK; n++) acc[n] = 0.0f;
    for (int b = lane; b < nb32; b += 32) {
        const uint8_t *blk = wrow + (size_t)(b >> 3) * 144;
        int j = b & 7;
        __half_raw hd; hd.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        __half_raw hm; hm.x = (uint16_t)(blk[2] | ((uint16_t)blk[3] << 8));
        float d = __half2float(__half(hd)), dmin = __half2float(__half(hm));
        int s, m; q4k_scale_min(blk + 4, j, &s, &m);
        const uint8_t *qbase = blk + 16 + (size_t)(j >> 1) * 32;
        int shift = (j & 1) ? 4 : 0;
        int sumi[NK];
        #pragma unroll
        for (int n = 0; n < NK; n++) sumi[n] = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int qw  = q8_get_int_b2(qbase, k);
            int nib = (shift ? (qw >> 4) : qw) & 0x0F0F0F0F;
            #pragma unroll
            for (int n = 0; n < NK; n++) {
                int xv = *(const int *)(qx + (size_t)n*in_dim + (size_t)b*32 + k*4);
                sumi[n] = __dp4a(nib, xv, sumi[n]);
            }
        }
        #pragma unroll
        for (int n = 0; n < NK; n++)
            acc[n] += dx[(size_t)n*nb32 + b] *
                      (d * (float)s * (float)sumi[n] - dmin * (float)m * (float)sx[(size_t)n*nb32 + b]);
    }
    #pragma unroll
    for (int n = 0; n < NK; n++) { float v = warp_reduce_sum_all(acc[n]); if (lane==0) out[(size_t)n*out_dim+idx] = v; }
}

// GROUPED Q4_K decode GEMV (MoE experts, ~1-2 tokens/expert): warp-per-out-row like the FP8
// grouped kernel, but Q4_K superblocks via dp4a over the Q8_1-quantized gathered activations.
// The tiled dg_mmq_q4_K_grouped is prefill-shaped (16-column tiles) and measured 49 GB/s at
// B=1 (94% wasted dp4a on 1-token experts); this GEMV form matches the ~74%-BW mmvq family.
// grid (out_dim/nwarps, n_slot) with the active-expert indirection; static → graph-safe.
__global__ void mmvq_q4_k_grouped_gemv_kernel(
    float *out, const uint8_t *wbase, int64_t slab_stride,
    const int8_t *qx, const float *dx, const int *sx,
    const int *coloff, const int *count, const int *active, int in_dim, int out_dim)
{
    int e = active ? active[blockIdx.y] : (int)blockIdx.y;
    if (e < 0) return;
    int cnt = count[e];
    if (cnt <= 0) return;
    int off = coloff[e];
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int n_super = in_dim >> 8, nb32 = in_dim >> 5;
    const uint8_t *wrow = wbase + (size_t)e * slab_stride + (size_t)idx * (size_t)n_super * 144;
    for (int c0 = 0; c0 < cnt; c0 += 8) {          // 8-token chunks (weight re-read per chunk;
        int cc = (cnt - c0 < 8) ? (cnt - c0) : 8;  //  decode experts hold ~1-2 tokens)
        float acc[8];
        #pragma unroll
        for (int n = 0; n < 8; n++) acc[n] = 0.0f;
        for (int b = lane; b < nb32; b += 32) {
            const uint8_t *blk = wrow + (size_t)(b >> 3) * 144;
            int j = b & 7;
            __half_raw hd; hd.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
            __half_raw hm; hm.x = (uint16_t)(blk[2] | ((uint16_t)blk[3] << 8));
            float d = __half2float(__half(hd)), dmin = __half2float(__half(hm));
            int sV, mV; q4k_scale_min(blk + 4, j, &sV, &mV);
            const uint8_t *qbase = blk + 16 + (size_t)(j >> 1) * 32;
            int shift = (j & 1) ? 4 : 0;
            for (int n = 0; n < cc; n++) {
                const int8_t *qrow = qx + (size_t)(off + c0 + n) * in_dim;
                int sumi = 0;
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    int qw  = q8_get_int_b2(qbase, k);
                    int nib = (shift ? (qw >> 4) : qw) & 0x0F0F0F0F;
                    int xv  = *(const int *)(qrow + (size_t)b * 32 + k * 4);
                    sumi = __dp4a(nib, xv, sumi);
                }
                acc[n] += dx[(size_t)(off + c0 + n) * nb32 + b] *
                          (d * (float)sV * (float)sumi - dmin * (float)mV * (float)sx[(size_t)(off + c0 + n) * nb32 + b]);
            }
        }
        for (int n = 0; n < cc; n++) {
            float v = warp_reduce_sum_all(acc[n]);
            if (lane == 0) out[(size_t)(off + c0 + n) * out_dim + idx] = v;
        }
    }
}

static inline void mmvq_q4_k_grouped_gemv_launch(
    float *out, const void *wbase, int64_t slab_stride,
    const int8_t *qx, const float *dx, const int *sx,
    const int *coloff, const int *count, const int *active, int n_slot, int n_expert,
    int in_dim, int out_dim, cudaStream_t stream)
{
    const int NWARPS = 8; int b = NWARPS * 32;
    dim3 g((out_dim + NWARPS - 1) / NWARPS, active ? n_slot : n_expert);
    mmvq_q4_k_grouped_gemv_kernel<<<g, b, 0, stream>>>(
        out, (const uint8_t *)wbase, slab_stride, qx, dx, sx, coloff, count, active, in_dim, out_dim);
}

static inline void mmvq_q4_k_launch(
    float *out, const uint8_t *w, const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim, cudaStream_t stream)
{
    const int NWARPS = 8; int b = NWARPS*32; int g = (out_dim + NWARPS - 1) / NWARPS;
    mmvq_q4_k_kernel<<<g, b, 0, stream>>>(out, w, qx, dx, sx, in_dim, out_dim);
}

static void mmvq_q4_k_batched_launch(
    float *out, const uint8_t *w, const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim, int K, cudaStream_t stream)
{
    const int NWARPS = 8; int b = NWARPS*32; dim3 g((out_dim + NWARPS - 1) / NWARPS);
    #define LQ4K(NK) mmvq_q4_k_batched_kernel<NK><<<g,b,0,stream>>>(out,w,qx,dx,sx,in_dim,out_dim)
    switch (K) {
        case 1: LQ4K(1); break; case 2: LQ4K(2); break; case 3: LQ4K(3); break; case 4: LQ4K(4); break;
        case 5: LQ4K(5); break; case 6: LQ4K(6); break; case 7: LQ4K(7); break; case 8: LQ4K(8); break;
        default:
            for (int o = 0; o < K; o += 8) {
                int kk = (K - o < 8) ? (K - o) : 8;
                mmvq_q4_k_batched_launch(out + (size_t)o*out_dim, w,
                                         qx + (size_t)o*in_dim, dx + (size_t)o*(in_dim>>5),
                                         sx + (size_t)o*(in_dim>>5), in_dim, out_dim, kk, stream);
            }
    }
    #undef LQ4K
}

// ── PACKED Q4_K: de-interleaved superblock for coalesced uint4 loads ──────────────────────
// The native mmvq_q4_k_kernel reads each sub-block's quants 2-byte-granular and re-reads the
// sibling sub-block's 32 bytes (the pair shares 32 quant bytes, low/high nibble per j&1).
// Repack each 256-elem superblock to the SAME 144 B: header [d,dmin,scales(12)] verbatim at
// [0..15], then 8 sub-blocks of 16 DE-INTERLEAVED quant bytes at [16+j*16 .. +15], byte m =
// nib(elem m) | (nib(elem m+16)<<4) — the GGML-Q4_0 nibble convention. The packed kernel then
// reads one coalesced uint4 (16 B) per sub-block and reuses the Q4_0 unpack/dp4a; the integer
// dot is identical → BIT-IDENTICAL to mmvq_q4_k_kernel. Repack is per-superblock same-size, so
// it runs in place inside d_weights (no second full weight copy); aliasing within a superblock
// is avoided by the caller giving src≠dst (a temp), since sub-blocks j and j^1 share src bytes.
__global__ void repack_q4_k_kernel(const uint8_t *src, uint8_t *dst, size_t n_super) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_super) return;
    const uint8_t *blk = src + i * 144;
    uint8_t       *out = dst + i * 144;
    #pragma unroll
    for (int h = 0; h < 16; h++) out[h] = blk[h];           // header verbatim
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const uint8_t *qbase = blk + 16 + (size_t)(j >> 1) * 32;
        int shift = (j & 1) ? 4 : 0;
        uint8_t *p = out + 16 + (size_t)j * 16;
        #pragma unroll
        for (int m = 0; m < 16; m++) {
            int lo = (qbase[m]      >> shift) & 0xF;        // element m
            int hi = (qbase[m + 16] >> shift) & 0xF;        // element m+16
            p[m] = (uint8_t)(lo | (hi << 4));
        }
    }
}

template<int NK>
__global__ void mmvq_q4_k_packed_batched_kernel(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int n_super = in_dim >> 8, nb32 = in_dim >> 5;
    const uint8_t *wrow = weight + (size_t)idx * (size_t)n_super * 144;
    float acc[NK];
    #pragma unroll
    for (int n = 0; n < NK; n++) acc[n] = 0.0f;
    for (int b = lane; b < nb32; b += 32) {
        const uint8_t *blk = wrow + (size_t)(b >> 3) * 144;
        int j = b & 7;
        __half_raw hd; hd.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        __half_raw hm; hm.x = (uint16_t)(blk[2] | ((uint16_t)blk[3] << 8));
        float d = __half2float(__half(hd)), dmin = __half2float(__half(hm));
        int s, m; q4k_scale_min(blk + 4, j, &s, &m);
        uint4 q = *(const uint4 *)(blk + 16 + (size_t)j * 16);   // one coalesced load, 32 nibbles
        int wv[8];
        int w0=(int)q.x, w1=(int)q.y, w2=(int)q.z, w3=(int)q.w;
        wv[0]=w0&0x0F0F0F0F; wv[1]=(w0>>4)&0x0F0F0F0F;
        wv[2]=w1&0x0F0F0F0F; wv[3]=(w1>>4)&0x0F0F0F0F;
        wv[4]=w2&0x0F0F0F0F; wv[5]=(w2>>4)&0x0F0F0F0F;
        wv[6]=w3&0x0F0F0F0F; wv[7]=(w3>>4)&0x0F0F0F0F;
        #pragma unroll
        for (int n = 0; n < NK; n++) {
            const int *xqs = (const int *)(qx + (size_t)n*in_dim + (size_t)b*32);
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                sumi = __dp4a(wv[2*k],   xqs[k],     sumi);
                sumi = __dp4a(wv[2*k+1], xqs[k + 4], sumi);
            }
            acc[n] += dx[(size_t)n*nb32 + b] *
                      (d*(float)s*(float)sumi - dmin*(float)m*(float)sx[(size_t)n*nb32 + b]);
        }
    }
    #pragma unroll
    for (int n = 0; n < NK; n++) { float v = warp_reduce_sum_all(acc[n]); if (lane==0) out[(size_t)n*out_dim+idx] = v; }
}

static void mmvq_q4_k_packed_batched_launch(
    float *out, const uint8_t *w, const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim, int K, cudaStream_t stream)
{
    const int NWARPS = 8; int b = NWARPS*32; dim3 g((out_dim + NWARPS - 1) / NWARPS);
    // NK = activation columns served per single weight-row read. Wide prefill tiles want the
    // largest NK (weight bytes amortize over NK rows); decode (K<=16) keeps the small cases.
    #define LPP_Q4K(NK,O) mmvq_q4_k_packed_batched_kernel<NK><<<g,b,0,stream>>>( \
        out + (size_t)(O)*out_dim, w, qx + (size_t)(O)*in_dim, \
        dx + (size_t)(O)*(in_dim>>5), sx + (size_t)(O)*(in_dim>>5), in_dim, out_dim)
    switch (K) {
        case 1: LPP_Q4K(1,0); return; case 2: LPP_Q4K(2,0); return; case 3: LPP_Q4K(3,0); return;
        case 4: LPP_Q4K(4,0); return; case 5: LPP_Q4K(5,0); return; case 6: LPP_Q4K(6,0); return;
        case 7: LPP_Q4K(7,0); return; case 8: LPP_Q4K(8,0); return;
        default: break;
    }
    // K>8: 32-row groups (32x weight-read amortization for prefill), then 16 (B=16 decode reads
    // the mixer weights ONCE per step instead of twice — 22% of the step was these GEMVs), then 8.
    int o = 0;
    for (; o + 32 <= K; o += 32) LPP_Q4K(32, o);
    for (; o + 16 <= K; o += 16) LPP_Q4K(16, o);
    for (; o + 8  <= K; o += 8)  LPP_Q4K(8,  o);
    int rem = K - o;
    switch (rem) {
        case 1: LPP_Q4K(1,o); break; case 2: LPP_Q4K(2,o); break; case 3: LPP_Q4K(3,o); break;
        case 4: LPP_Q4K(4,o); break; case 5: LPP_Q4K(5,o); break; case 6: LPP_Q4K(6,o); break;
        case 7: LPP_Q4K(7,o); break; default: break;
    }
    #undef LPP_Q4K
}

static inline void mmvq_q4_k_packed_launch(
    float *out, const uint8_t *w, const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim, cudaStream_t stream)
{ mmvq_q4_k_packed_batched_launch(out, w, qx, dx, sx, in_dim, out_dim, 1, stream); }

// ── PACKED Q4_K, TRANSPOSED activations: the LSU-bound batched decode mixer fix ────────────
// Profiling the B=16 decode step showed mmvq_q4_k_packed_batched at ~45-58 GB/s while a raw
// stream over the same weights reaches >800 GB/s and time scales LINEARLY with NK — the kernel
// is bound by its ACTIVATION loads, not by weight latency or dp4a: with qx in row-major
// [n][in_dim], each inner LDG.128 has a 32-byte lane stride = 8 sector replays. Reading the
// quantize_q8_1t_kernel layout instead makes every activation load a dense LDG.128 (2 loads
// A=xqs[0..3], B=xqs[4..7] per (n, window)), halving LSU wavefronts per epoch; the superblock
// header + quants are staged up-front as uint4 pairs (PIPE=2 epochs) and the (scale,min) are
// decoded from the staged registers (q4k_scale_min_reg, integer-exact). The dp4a sequence
// (wv[2k]·A.k then wv[2k+1]·B.k, k ascending) and the per-acc[n] float accumulation order
// (b = lane, lane+32, … with the ORIGINAL (d·s)·sumi − (dmin·m)·Σx grouping) are exactly those
// of mmvq_q4_k_packed_batched_kernel → BITWISE-identical output. Measured on the decode shapes
// (weights uncached): NK=16 qkv 163.4→63.0 us, z 83.3→32.8, out_proj 88.2→44.5 (2.0-2.6×);
// NK=8/4 ~2-2.4×. NK=1 is NOT routed here (row-major is already fine at one token).
// Register variant of q4k_scale_min reading the 12 packed scale bytes from the staged header
// uint4 (sy = header bytes 4..7, sz = 8..11, sw = 12..15). Same integer decode.
__device__ __forceinline__ void q4k_scale_min_reg(
    uint32_t sy, uint32_t sz, uint32_t sw, int j, int *s, int *m)
{
    if (j < 4) { *s = (int)((sy >> (8*j)) & 63); *m = (int)((sz >> (8*j)) & 63); }
    else {
        int sh = 8*(j - 4);
        uint32_t bj4 = (sw >> sh) & 0xFF;        // sc[j+4]
        uint32_t bjm = (sy >> sh) & 0xFF;        // sc[j-4]
        uint32_t bj  = (sz >> sh) & 0xFF;        // sc[j]
        *s = (int)((bj4 & 0x0F) | ((bjm >> 6) << 4));
        *m = (int)((bj4 >> 4)   | ((bj  >> 6) << 4));
    }
}

template<int NK>
__global__ void mmvq_q4_k_packedT_batched_kernel(
    float *out, const uint8_t *weight, const int8_t *qxT, const float *dx, const int *sx,
    int in_dim, int out_dim)
{
    constexpr int PIPE = 2;                      // superblock epochs staged per chunk
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int n_super = in_dim >> 8, nb32 = in_dim >> 5;
    const uint8_t *wrow = weight + (size_t)idx * (size_t)n_super * 144;
    float acc[NK];
    #pragma unroll
    for (int n = 0; n < NK; n++) acc[n] = 0.0f;
    for (int b0 = lane; b0 < nb32; b0 += 32*PIPE) {
        // stage this chunk's independent weight loads (header + quants) up-front
        uint4 hh[PIPE], hq[PIPE];
        #pragma unroll
        for (int p = 0; p < PIPE; p++) {
            int b = b0 + 32*p;
            if (b < nb32) {
                const uint8_t *blk = wrow + (size_t)(b >> 3) * 144;
                hh[p] = __ldg((const uint4 *)blk);
                hq[p] = __ldg((const uint4 *)(blk + 16 + (size_t)(b & 7) * 16));
            }
        }
        #pragma unroll
        for (int p = 0; p < PIPE; p++) {
            int b = b0 + 32*p;
            if (b >= nb32) break;
            int j = b & 7;
            uint4 h = hh[p];
            __half_raw hd; hd.x = (uint16_t)(h.x & 0xFFFF);
            __half_raw hm; hm.x = (uint16_t)(h.x >> 16);
            float d = __half2float(__half(hd)), dmin = __half2float(__half(hm));
            int s, m; q4k_scale_min_reg(h.y, h.z, h.w, j, &s, &m);
            float ds = d*(float)s, dm = dmin*(float)m;   // original (d·s), (dmin·m) grouping
            uint4 q = hq[p];
            int wv[8];
            int w0=(int)q.x, w1=(int)q.y, w2=(int)q.z, w3=(int)q.w;
            wv[0]=w0&0x0F0F0F0F; wv[1]=(w0>>4)&0x0F0F0F0F;
            wv[2]=w1&0x0F0F0F0F; wv[3]=(w1>>4)&0x0F0F0F0F;
            wv[4]=w2&0x0F0F0F0F; wv[5]=(w2>>4)&0x0F0F0F0F;
            wv[6]=w3&0x0F0F0F0F; wv[7]=(w3>>4)&0x0F0F0F0F;
            const int8_t *xbase = qxT + (size_t)(b >> 5)*1024 + (size_t)(b & 31)*16;
            #pragma unroll
            for (int n = 0; n < NK; n++) {
                uint4 A = *(const uint4 *)(xbase + (size_t)n*in_dim);        // xqs[0..3]
                uint4 B = *(const uint4 *)(xbase + (size_t)n*in_dim + 512);  // xqs[4..7]
                int sumi = 0;
                sumi = __dp4a(wv[0], (int)A.x, sumi);
                sumi = __dp4a(wv[1], (int)B.x, sumi);
                sumi = __dp4a(wv[2], (int)A.y, sumi);
                sumi = __dp4a(wv[3], (int)B.y, sumi);
                sumi = __dp4a(wv[4], (int)A.z, sumi);
                sumi = __dp4a(wv[5], (int)B.z, sumi);
                sumi = __dp4a(wv[6], (int)A.w, sumi);
                sumi = __dp4a(wv[7], (int)B.w, sumi);
                acc[n] += dx[(size_t)n*nb32 + b] *
                          (ds*(float)sumi - dm*(float)sx[(size_t)n*nb32 + b]);
            }
        }
    }
    #pragma unroll
    for (int n = 0; n < NK; n++) { float v = warp_reduce_sum_all(acc[n]); if (lane==0) out[(size_t)n*out_dim+idx] = v; }
}

// K>8 MULTI-CHUNK variant (P2 dense decode): ONE launch covers all ceil(K/8) 8-token chunks
// with grid = (chunk, row-group) — blockIdx.x (fastest-dispatched) is the CHUNK, so all chunks
// of a row group are scheduled adjacently and the row's weight bytes stream from DRAM once and
// hit L2 (24 MB on GB10, ~90% hit measured) for the other chunks. This replaces the serial
// NK=16/8/rem ladder that re-read the whole weight slab per pass (3× at B=30 — 59 ms of a
// 91.6 ms dense decode step) while KEEPING the NK≤8 register footprint (68 regs vs 144 for
// the NK=16 tile at 16.6% occupancy). Per-token accumulation acc[n] walks the b-loop in the
// SAME lane order regardless of chunking ⇒ outputs BITWISE-identical to the single-chunk
// kernel and to the ladder it replaces. `ntok` guards the ≤8-token tail chunk.
// (Tried and rejected: PIPE=4 — 323 vs 366 agg @ B=16; 2-row register blocking — spills.)
// D32: MINBLK is a template arg so the wider NK=16 tile (16 acc regs vs 8) can trade a little
// occupancy for HALF the chunk count at K>16 (B=32 → 2 chunks not 4), cutting the redundant
// L2 weight bandwidth + per-chunk Q4_K dequant ALU that measured 65% of the B=16→32 step growth.
// D32B: DPSPLIT breaks the 8-deep SERIAL __dp4a chain (one `sumi` threaded through 8 dp4a =
// an 8-long ALU dependency chain, measured as the 23-28 cyc/inst warp latency at inst/cycle 0.2
// — pure latency starvation) into DPSPLIT INDEPENDENT partial sums that issue in parallel, then
// sums them at the end. Two's-complement int32 addition is associative AND commutative, so the
// final `sumi` is the sum of the SAME 8 dot4(wv,x) terms regardless of grouping → BITWISE-
// identical (hash-verified, not asserted). Near-zero register cost (DPSPLIT-1 extra int regs),
// shortens the per-token critical path 8→8/DPSPLIT for latency hiding without touching occupancy.
// D32B: PIPE (staged superblock epochs) is a template arg so deeper weight staging (PIPE=3/4 =
// more in-flight independent LDGs per warp) can be measured against the long-scoreboard stall
// (ncu: 17-20 cyc/issue on activation+weight loads = the dominant stall) that occupancy alone
// (53%, register-capped) cannot hide. Bit-identical: PIPE only changes how many b-windows are
// prefetched, not the per-(row,token) ascending-b accumulation order.
template<int NK, int MINBLK = 4, int DPSPLIT = 1, int PIPE = 2>
__global__ void __launch_bounds__(256, MINBLK) mmvq_q4_k_packedT_multi_kernel(
    float *out, const uint8_t *weight, const int8_t *qxT, const float *dx, const int *sx,
    int in_dim, int out_dim, int K)
{
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.y * nwarps + warp;        // output row (grid.y = row groups)
    if (idx >= out_dim) return;
    const int t0 = (int)blockIdx.x * NK;         // this chunk's first token
    const int ntok = (K - t0 < NK) ? (K - t0) : NK;
    const int8_t *qxTc = qxT + (size_t)t0 * in_dim;
    const float  *dxc  = dx + (size_t)t0 * (in_dim >> 5);
    const int    *sxc  = sx + (size_t)t0 * (in_dim >> 5);
    float        *outc = out + (size_t)t0 * out_dim;
    int n_super = in_dim >> 8, nb32 = in_dim >> 5;
    const uint8_t *wrow = weight + (size_t)idx * (size_t)n_super * 144;
    float acc[NK];
    #pragma unroll
    for (int n = 0; n < NK; n++) acc[n] = 0.0f;
    for (int b0 = lane; b0 < nb32; b0 += 32*PIPE) {
        uint4 hh[PIPE], hq[PIPE];
        #pragma unroll
        for (int p = 0; p < PIPE; p++) {
            int b = b0 + 32*p;
            if (b < nb32) {
                const uint8_t *blk = wrow + (size_t)(b >> 3) * 144;
                hh[p] = __ldg((const uint4 *)blk);
                hq[p] = __ldg((const uint4 *)(blk + 16 + (size_t)(b & 7) * 16));
            }
        }
        #pragma unroll
        for (int p = 0; p < PIPE; p++) {
            int b = b0 + 32*p;
            if (b >= nb32) break;
            int j = b & 7;
            uint4 h = hh[p];
            __half_raw hd; hd.x = (uint16_t)(h.x & 0xFFFF);
            __half_raw hm; hm.x = (uint16_t)(h.x >> 16);
            float d = __half2float(__half(hd)), dmin = __half2float(__half(hm));
            int s, m; q4k_scale_min_reg(h.y, h.z, h.w, j, &s, &m);
            float ds = d*(float)s, dm = dmin*(float)m;   // original (d·s), (dmin·m) grouping
            uint4 q = hq[p];
            int wv[8];
            int w0=(int)q.x, w1=(int)q.y, w2=(int)q.z, w3=(int)q.w;
            wv[0]=w0&0x0F0F0F0F; wv[1]=(w0>>4)&0x0F0F0F0F;
            wv[2]=w1&0x0F0F0F0F; wv[3]=(w1>>4)&0x0F0F0F0F;
            wv[4]=w2&0x0F0F0F0F; wv[5]=(w2>>4)&0x0F0F0F0F;
            wv[6]=w3&0x0F0F0F0F; wv[7]=(w3>>4)&0x0F0F0F0F;
            const int8_t *xbase = qxTc + (size_t)(b >> 5)*1024 + (size_t)(b & 31)*16;
            #pragma unroll
            for (int n = 0; n < NK; n++) {
                if (n >= ntok) break;
                // D32B: __ldg on activations measured NEUTRAL (455 vs 465, within noise) — they are
                // already L1-served at 90% hit, so the read-only path doesn't cut the latency.
                uint4 A = *(const uint4 *)(xbase + (size_t)n*in_dim);        // xqs[0..3]
                uint4 B = *(const uint4 *)(xbase + (size_t)n*in_dim + 512);  // xqs[4..7]
                int sumi;
                if (DPSPLIT >= 4) {
                    // 4 independent dp4a chains (2-deep each) → shortest critical path.
                    int s0=0,s1=0,s2=0,s3=0;
                    s0 = __dp4a(wv[0], (int)A.x, s0); s0 = __dp4a(wv[1], (int)B.x, s0);
                    s1 = __dp4a(wv[2], (int)A.y, s1); s1 = __dp4a(wv[3], (int)B.y, s1);
                    s2 = __dp4a(wv[4], (int)A.z, s2); s2 = __dp4a(wv[5], (int)B.z, s2);
                    s3 = __dp4a(wv[6], (int)A.w, s3); s3 = __dp4a(wv[7], (int)B.w, s3);
                    sumi = (s0 + s1) + (s2 + s3);   // int32 add is exact & associative
                } else if (DPSPLIT == 2) {
                    // 2 independent dp4a chains (4-deep each).
                    int s0=0,s1=0;
                    s0 = __dp4a(wv[0], (int)A.x, s0); s0 = __dp4a(wv[1], (int)B.x, s0);
                    s0 = __dp4a(wv[2], (int)A.y, s0); s0 = __dp4a(wv[3], (int)B.y, s0);
                    s1 = __dp4a(wv[4], (int)A.z, s1); s1 = __dp4a(wv[5], (int)B.z, s1);
                    s1 = __dp4a(wv[6], (int)A.w, s1); s1 = __dp4a(wv[7], (int)B.w, s1);
                    sumi = s0 + s1;
                } else {
                    sumi = 0;
                    sumi = __dp4a(wv[0], (int)A.x, sumi);
                    sumi = __dp4a(wv[1], (int)B.x, sumi);
                    sumi = __dp4a(wv[2], (int)A.y, sumi);
                    sumi = __dp4a(wv[3], (int)B.y, sumi);
                    sumi = __dp4a(wv[4], (int)A.z, sumi);
                    sumi = __dp4a(wv[5], (int)B.z, sumi);
                    sumi = __dp4a(wv[6], (int)A.w, sumi);
                    sumi = __dp4a(wv[7], (int)B.w, sumi);
                }
                acc[n] += dxc[(size_t)n*nb32 + b] *
                          (ds*(float)sumi - dm*(float)sxc[(size_t)n*nb32 + b]);
            }
        }
    }
    #pragma unroll
    for (int n = 0; n < NK; n++) {
        if (n >= ntok) break;
        float v = warp_reduce_sum_all(acc[n]);
        if (lane==0) outc[(size_t)n*out_dim+idx] = v;
    }
}


static void mmvq_q4_k_packedT_batched_launch(
    float *out, const uint8_t *w, const int8_t *qxT, const float *dx, const int *sx,
    int in_dim, int out_dim, int K, cudaStream_t stream)
{
    // D32B: NWARPS (block size = 32*NWARPS) is env-tunable for the K>16 multi-kernel only
    // (FUCINA_Q4K_NWARPS={4,8,16}, default 8) — more warps/block can raise resident-warp count to
    // hide the long-scoreboard load latency. K≤16 keeps NWARPS=8 (untouched winning path).
    const int NWARPS = 8; int b = NWARPS*32; dim3 g((out_dim + NWARPS - 1) / NWARPS);
    #define LPT_Q4K(NK,O) mmvq_q4_k_packedT_batched_kernel<NK><<<g,b,0,stream>>>( \
        out + (size_t)(O)*out_dim, w, qxT + (size_t)(O)*in_dim, \
        dx + (size_t)(O)*(in_dim>>5), sx + (size_t)(O)*(in_dim>>5), in_dim, out_dim)
    switch (K) {
        case 1: LPT_Q4K(1,0); return; case 2: LPT_Q4K(2,0); return; case 3: LPT_Q4K(3,0); return;
        case 4: LPT_Q4K(4,0); return; case 5: LPT_Q4K(5,0); return; case 6: LPT_Q4K(6,0); return;
        case 7: LPT_Q4K(7,0); return; case 8: LPT_Q4K(8,0); return;
        default: break;
    }
    #undef LPT_Q4K
    // K>8: one multi-chunk launch, weight read once (chunk-major dispatch → L2 reuse across the
    // per-chunk token groups of each row group). Bitwise-identical to the old NK=16/8/rem ladder
    // AND to the NK=8 multi-kernel (same per-(row,token) ascending-b accumulation; chunking only
    // changes WHICH block computes a (row,token) pair, not its arithmetic — NK-independent).
    // D32: for K>16 a wider chunk tile HALVES the chunk count (B=32 → 2 chunks at NK=16, 3 at
    // NK=12, vs 4 at NK=8), cutting the redundant L2 weight bandwidth + per-chunk Q4_K dequant
    // that measured 65% of the B=16→32 step growth — but a wider acc[] costs registers (occupancy).
    // The NK/MINBLK for K>16 is config-selectable (FUCINA_Q4K_BIGCHUNK={8,12,16}, default 12) so
    // the chunk-width vs occupancy tradeoff is MEASURED per box, not hardcoded. Every variant is
    // bitwise-identical (NK-independent per-(row,token) ascending-b accumulation). K≤16 keeps NK=8.
    // D32B verdict (docs/qwen35-d32b.md): the mixer is latency-bound and REGISTER-bound — the
    // shipped optimum is NK=12 tile + MINBLK=4 + DPSPLIT=2 + PIPE=2 (occ 53→70-86%, +9.1% @ B=32,
    // hash-verified bit-identical c6ab45eab1f2751c). Measured-dead variants (2/4-rows-per-warp,
    // cp.async staging, PIPE≥3) were removed after the sweep; the surviving knobs are:
    //   FUCINA_Q4K_BIGCHUNK={8,12,16} — K>16 chunk-tile width (chunk count vs acc[] registers)
    //   FUCINA_Q4K_DPSPLIT={1,2}      — split the 8-deep serial __dp4a chain (int32 add associative)
    //   FUCINA_Q4K_MINBLK={3,4}       — __launch_bounds__ min-blocks/SM (occupancy vs registers)
    //   FUCINA_Q4K_NWARPS={4,8}       — warps/block for the K>16 tile
    // None change the per-(row,token) ascending-b accumulation order → bit-identical by construction.
    static int dpsplit = -1, minblk_env = -1, nw_env = -1;
    if (dpsplit < 0)    { const char *e = getenv("FUCINA_Q4K_DPSPLIT"); dpsplit    = e ? atoi(e) : 2; }
    if (minblk_env < 0) { const char *e = getenv("FUCINA_Q4K_MINBLK");  minblk_env = e ? atoi(e) : 0; }
    if (nw_env < 0)     { const char *e = getenv("FUCINA_Q4K_NWARPS");  nw_env     = e ? atoi(e) : 8; }
    const int MW = nw_env > 0 ? nw_env : 8; const int mb_threads = MW*32;
    // Dispatch a (NK, MINBLK) tile at the shipped PIPE=2 across DPSPLIT∈{1,2}. MINBLK is fixed per
    // NK for register headroom (env override applies to the wide tiles where occupancy bites).
    #define D32B_DISP(NK, MB, gm) do { \
        int mb = minblk_env > 0 ? minblk_env : (MB); const int b = mb_threads; \
        if (mb <= 3) { \
            if (dpsplit==2) mmvq_q4_k_packedT_multi_kernel<NK,3,2,2><<<gm,b,0,stream>>>(out,w,qxT,dx,sx,in_dim,out_dim,K); \
            else            mmvq_q4_k_packedT_multi_kernel<NK,3,1,2><<<gm,b,0,stream>>>(out,w,qxT,dx,sx,in_dim,out_dim,K); \
        } else { \
            if (dpsplit==2) mmvq_q4_k_packedT_multi_kernel<NK,4,2,2><<<gm,b,0,stream>>>(out,w,qxT,dx,sx,in_dim,out_dim,K); \
            else            mmvq_q4_k_packedT_multi_kernel<NK,4,1,2><<<gm,b,0,stream>>>(out,w,qxT,dx,sx,in_dim,out_dim,K); \
        } } while(0)
    if (K > 16) {
        static int bigchunk = -1;
        if (bigchunk < 0) { const char *e = getenv("FUCINA_Q4K_BIGCHUNK"); bigchunk = e ? atoi(e) : 12; }
        if (bigchunk == 8) {
            dim3 gm((unsigned)((K + 7) / 8), (unsigned)((out_dim + MW - 1) / MW));
            D32B_DISP(8, 4, gm);
        } else if (bigchunk == 12) {
            dim3 gm((unsigned)((K + 11) / 12), (unsigned)((out_dim + MW - 1) / MW));
            D32B_DISP(12, 4, gm);   // D32B: MINBLK=4 measured-best (occ 53%→70%, +9.3% @ B=32)
        } else {
            dim3 gm((unsigned)((K + 15) / 16), (unsigned)((out_dim + MW - 1) / MW));
            D32B_DISP(16, 3, gm);
        }
        return;
    }
    // K in (8,16]: keep the untouched winning NK=8/2-chunk path at fixed NWARPS=8 (NWARPS env is
    // a K>16-only tuning knob) but still honor DPSPLIT/PIPE/MINBLK (all bit-identical).
    {
        const int b8 = 8*32;
        dim3 gm((unsigned)((K + 7) / 8), (unsigned)((out_dim + 8 - 1) / 8));
        int mb = minblk_env > 0 ? minblk_env : 4;
        if (mb <= 3) {
            if (dpsplit==2) mmvq_q4_k_packedT_multi_kernel<8,3,2,2><<<gm,b8,0,stream>>>(out,w,qxT,dx,sx,in_dim,out_dim,K);
            else            mmvq_q4_k_packedT_multi_kernel<8,3,1,2><<<gm,b8,0,stream>>>(out,w,qxT,dx,sx,in_dim,out_dim,K);
        } else {
            if (dpsplit==2) mmvq_q4_k_packedT_multi_kernel<8,4,2,2><<<gm,b8,0,stream>>>(out,w,qxT,dx,sx,in_dim,out_dim,K);
            else            mmvq_q4_k_packedT_multi_kernel<8,4,1,2><<<gm,b8,0,stream>>>(out,w,qxT,dx,sx,in_dim,out_dim,K);
        }
    }
    #undef D32B_DISP
}

// ─── Native Q4_K / Q6_K → BF16 dequant (fast Qwen3 prefill) ─────────────────────────
// Dequantize a full K-quant weight row-set [out_dim][in_dim] (whose in_dim is a multiple
// of 256) into the row-major BF16 buffer the cuBLAS prefill GEMM reads, element-for-element.
// Each thread owns ONE 256-element superblock and writes its 256 dequantized values in the
// SAME sequential element order the matching mmvq_q4_k / mmvq_q6_k decode kernels consume,
// so the dequantized weights are numerically the K-quant values (greedy-token faithful).
//
// Q4_K superblock = 144 B: half d, half dmin, 12 B packed 6-bit (scale,min)×8, 128 B 4-bit
// quants. Sub-block j: value = d·s_j·q − dmin·m_j (asymmetric). Mirrors mmvq_q4_k_kernel.
__global__ void dequant_q4_k_to_bf16_kernel(
    __nv_bfloat16 *dst, const uint8_t *src, uint64_t n_super)
{
    uint64_t sb = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= n_super) return;
    const uint8_t *blk = src + sb * 144;
    __half_raw hd; hd.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
    __half_raw hm; hm.x = (uint16_t)(blk[2] | ((uint16_t)blk[3] << 8));
    float d = __half2float(__half(hd)), dmin = __half2float(__half(hm));
    __nv_bfloat16 *out = dst + sb * 256;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int s, m; q4k_scale_min(blk + 4, j, &s, &m);
        const uint8_t *qbase = blk + 16 + (size_t)(j >> 1) * 32;
        int shift = (j & 1) ? 4 : 0;
        float ds = d * (float)s, dm = dmin * (float)m;
        for (int k = 0; k < 32; k++) {
            int q = (qbase[k] >> shift) & 0x0F;
            out[j * 32 + k] = __float2bfloat16(ds * (float)q - dm);
        }
    }
}

// PACKED Q4_K → BF16: build_packed_q4k de-interleaves each superblock IN PLACE (header
// verbatim, then 8 sub-blocks of 16 bytes where byte m = nib(elem m) | nib(elem m+16)<<4).
// When the engine has repacked (q4k_packed), the bulk Q4_K weights live in this layout, so
// the dequant must read it. Element order matches mmvq_q4_k_packed_batched_kernel: within a
// 32-elem sub-block, elems 0..15 are the low nibbles of bytes 0..15, elems 16..31 the highs.
__global__ void dequant_q4_k_packed_to_bf16_kernel(
    __nv_bfloat16 *dst, const uint8_t *src, uint64_t n_super)
{
    uint64_t sb = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= n_super) return;
    const uint8_t *blk = src + sb * 144;
    __half_raw hd; hd.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
    __half_raw hm; hm.x = (uint16_t)(blk[2] | ((uint16_t)blk[3] << 8));
    float d = __half2float(__half(hd)), dmin = __half2float(__half(hm));
    __nv_bfloat16 *out = dst + sb * 256;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int s, m; q4k_scale_min(blk + 4, j, &s, &m);
        const uint8_t *p = blk + 16 + (size_t)j * 16;
        float ds = d * (float)s, dm = dmin * (float)m;
        for (int e = 0; e < 32; e++) {
            int byte = p[e & 15];
            int q = (e < 16) ? (byte & 0x0F) : (byte >> 4);
            out[j * 32 + e] = __float2bfloat16(ds * (float)q - dm);
        }
    }
}

// Q6_K superblock = 210 B: 128 B low-4-bit (ql), 64 B high-2-bit (qh), 16 B int8 scales,
// half d. value = d·scale·(q6 − 32). Mirrors the canonical ggml dequantize_row_q6_K element
// order (the same order mmvq_q6_k_kernel reads its sequential 32-element activation blocks in).
__global__ void dequant_q6_k_to_bf16_kernel(
    __nv_bfloat16 *dst, const uint8_t *src, uint64_t n_super)
{
    uint64_t sb = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= n_super) return;
    const uint8_t *blk = src + sb * 210;
    const uint8_t *ql  = blk;
    const uint8_t *qh  = blk + 128;
    const int8_t  *sc  = (const int8_t *)(blk + 192);
    __half_raw hd; hd.x = (uint16_t)(blk[208] | ((uint16_t)blk[209] << 8));
    float d = __half2float(__half(hd));
    __nv_bfloat16 *out = dst + sb * 256;
    #pragma unroll
    for (int nb = 0; nb < 2; nb++) {
        const uint8_t *qln = ql + nb * 64;
        const uint8_t *qhn = qh + nb * 32;
        const int8_t  *scn = sc + nb * 8;
        __nv_bfloat16 *yb  = out + nb * 128;
        for (int l = 0; l < 32; l++) {
            int is = l >> 4;
            int q1 = (int)((qln[l]      & 0x0F) | (((qhn[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((qln[l + 32] & 0x0F) | (((qhn[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((qln[l]      >>   4) | (((qhn[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((qln[l + 32] >>   4) | (((qhn[l] >> 6) & 3) << 4)) - 32;
            yb[l +  0] = __float2bfloat16(d * (float)scn[is + 0] * (float)q1);
            yb[l + 32] = __float2bfloat16(d * (float)scn[is + 2] * (float)q2);
            yb[l + 64] = __float2bfloat16(d * (float)scn[is + 4] * (float)q3);
            yb[l + 96] = __float2bfloat16(d * (float)scn[is + 6] * (float)q4);
        }
    }
}

// Dispatch a single projection's dequant to BF16 by per-tensor format. Q4_K/Q6_K use the
// superblock kernels above (one thread per 256-elem superblock); Q4_0/Q8_0/FP8 fall back to
// the element-wise decode_weight path. n = in_dim*out_dim (multiple of 256 for K-quants).
static inline void dequant_proj_to_bf16(
    __nv_bfloat16 *dst, const uint8_t *src, uint64_t n, int fmt, bool q4k_packed,
    cudaStream_t stream)
{
    if (fmt == FORMAT_Q4_K) {
        uint64_t ns = n / 256;
        if (q4k_packed)
            dequant_q4_k_packed_to_bf16_kernel<<<(unsigned)((ns + 255) / 256), 256, 0, stream>>>(dst, src, ns);
        else
            dequant_q4_k_to_bf16_kernel<<<(unsigned)((ns + 255) / 256), 256, 0, stream>>>(dst, src, ns);
    } else if (fmt == FORMAT_Q6_K) {
        uint64_t ns = n / 256;
        dequant_q6_k_to_bf16_kernel<<<(unsigned)((ns + 255) / 256), 256, 0, stream>>>(dst, src, ns);
    } else {
        dequant_to_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(dst, src, n, fmt);
    }
}

// Block-FP8 (E4M3 weight + per-128×128 BF16 block scale) → BF16, element-wise. Feeds the
// tensor-core prefill GEMM: dequant ONCE per projection (cached when memory allows) instead
// of the FP8_MAXB-chunked GEMV that re-read the weight T/16 times per prefill tile — the
// measured 37× TTFT gap vs vLLM on the 35B MoE at a ~2k prompt.
__global__ void dequant_fp8_block_to_bf16_kernel(
    __nv_bfloat16 *dst, const uint8_t *w, const __nv_bfloat16 *sc, int in_dim, uint64_t n)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row = (int)(i / in_dim), col = (int)(i % in_dim);
    __nv_fp8_e4m3 v; v.__x = w[i];
    float bs = __bfloat162float(sc[(size_t)(row >> 7) * (in_dim >> 7) + (col >> 7)]);
    dst[i] = __float2bfloat16(float(v) * bs);
}


// =========================================================================
// =========================================================================
// ─── RoPE Kernels ────────────────────────────────────────────────────────
// =========================================================================

// Standard RoPE for sliding window attention
// theta = 10000, head_dim = 256 (all dims rotated)
// NEOX-style RoPE for sliding-window layers.
// Pairs: (d, d + head_dim/2) for d in [0, head_dim/2).
// theta_d = base ^ (-2*d / head_dim)  matching llama.cpp rotate_pairs stride=half.
// Launched: rope_sliding_kernel<<<n_heads, head_dim/2>>>
__global__ void rope_sliding_kernel(
    float *q,           // [n_heads × head_dim]
    float *k,           // [n_kv_heads × head_dim]
    int    pos,
    const int *pos_ptr, // non-NULL: device-resident position overrides pos (CUDA-graph path)
    int    n_heads,
    int    n_kv_heads,
    int    head_dim,
    float  theta_base)
{
    int d   = threadIdx.x;          // 0 .. head_dim/2 - 1
    int idx = blockIdx.x;           // head index
    int half = head_dim / 2;
    if (d >= half) return;
    if (pos_ptr) pos = *pos_ptr;

    float theta   = powf(theta_base, -2.0f * d / head_dim);
    float cos_val = cosf(pos * theta);
    float sin_val = sinf(pos * theta);

    // Q
    float *qh = q + idx * head_dim;
    float q0 = qh[d], q1 = qh[d + half];
    qh[d]        = q0 * cos_val - q1 * sin_val;
    qh[d + half] = q0 * sin_val + q1 * cos_val;

    // K (only for heads < n_kv_heads)
    if (idx < n_kv_heads) {
        float *kh = k + idx * head_dim;
        float k0 = kh[d], k1 = kh[d + half];
        kh[d]        = k0 * cos_val - k1 * sin_val;
        kh[d + half] = k0 * sin_val + k1 * cos_val;
    }
}

// NEOX RoPE for global-attention layers.
// Full head_dim=512 rotated; freq_factors are per-dim divisors from
// rope_freqs.weight (shape [head_dim/2]).
// p-RoPE: effective_pos = pos * (context_len / max_ctx).
// Launched: rope_global_kernel<<<n_heads, head_dim/2>>>
__global__ void rope_global_kernel(
    float       *q,             // [n_heads × head_dim]
    float       *k,             // [n_kv_heads × head_dim]
    int          pos,
    const int   *pos_ptr,       // non-NULL: device-resident position overrides pos
    int          context_len,
    int          n_heads,
    int          n_kv_heads,
    int          head_dim,      // 512
    float        theta_base,    // 1000000.0f
    const float *freq_factors)  // [head_dim/2] or NULL
{
    int d   = threadIdx.x;       // 0 .. head_dim/2 - 1
    int idx = blockIdx.x;
    int half = head_dim / 2;
    if (d >= half) return;
    if (pos_ptr) pos = *pos_ptr;
    (void)context_len;

    // Matches ggml_rope_ext NEOX (llama.cpp gemma4): theta = pos * base^(-2d/n) / ff.
    // freq_scale is 1.0 for gemma4 (no rope scaling metadata); positions are NOT
    // rescaled by context length — freq_factors alone implement p-RoPE.
    float ff    = freq_factors ? freq_factors[d] : 1.0f;
    float theta = powf(theta_base, -2.0f * d / head_dim) / ff;
    float cos_val = cosf(pos * theta);
    float sin_val = sinf(pos * theta);

    float *qh = q + idx * head_dim;
    float q0 = qh[d], q1 = qh[d + half];
    qh[d]        = q0 * cos_val - q1 * sin_val;
    qh[d + half] = q0 * sin_val + q1 * cos_val;

    if (idx < n_kv_heads) {
        float *kh = k + idx * head_dim;
        float k0 = kh[d], k1 = kh[d + half];
        kh[d]        = k0 * cos_val - k1 * sin_val;
        kh[d + half] = k0 * sin_val + k1 * cos_val;
    }
}

// =========================================================================
// ─── Sliding Window Attention Decode (single token) ─────────────────────
// =========================================================================

// Sliding-window GQA decode, single query token. FLASH (online-softmax) form:
// each thread owns output dim `tid` and accumulates V on the fly while tracking a
// running max `m` and denominator `l`, so NO O(window) score buffer is needed —
// shared memory is a constant 32 floats (block_reduce scratch only). This is what
// lifts the global-attn context cap (see global kernel) and keeps the math
// numerically stable. blockIdx.x = q_head, blockDim.x = head_dim.
// KV layout: [window_size][n_kv_heads][head_dim] ring buffer.
// Launch: sliding_attn_decode_kernel<<<n_heads, head_dim, 32*sizeof(float)>>>
// FLAT per-position sliding cache (DECODE-30-35 Step 3): k_cache/v_cache is the layer's
// [capacity][n_kv_heads][head_dim] buffer indexed by ABSOLUTE position (no ring wrap), so
// rewind is exact (no eviction). n_tokens = tokens in this layer's cache (incl. the current
// one, written before this call). Attend the window-bounded contiguous range
// [max(0,n_tokens-window), n_tokens-1] — matches llama.cpp [i-window+1, i] (window keys).
// REFERENCE ONLY: all live decode call sites use sliding_attn_decode_broadcast (split-K,
// warp-per-KV-head — see sliding_attn_splitk_kernel), which kills the per-key
// block_reduce_sum (two __syncthreads) this kernel pays ~window times per layer.
__global__ void sliding_attn_decode_kernel(
    float       *output,       // [n_heads × head_dim]
    const float *q,            // [n_heads × head_dim]
    const kv_t  *k_cache,      // [capacity][n_kv_heads][head_dim] FP8 (absolute pos)
    const kv_t  *v_cache,
    int          n_heads,
    int          n_kv_heads,
    int          head_dim,
    int          window,
    int          n_tokens)
{
    extern __shared__ float smem[];      // [32] block_reduce scratch only
    int q_head  = blockIdx.x;
    int kv_head = q_head / (n_heads / n_kv_heads); // GQA mapping
    int tid     = threadIdx.x;

    int window_len = min(n_tokens, window);
    int lo = n_tokens - window_len;      // first absolute position to attend

    float q_d = (tid < head_dim) ? q[q_head * head_dim + tid] : 0.0f;
    float acc = 0.0f;                    // this thread's output element
    float m   = -INFINITY;               // running row max
    float l   = 0.0f;                    // running denominator

    for (int i = 0; i < window_len; i++) {
        size_t pos = (size_t)(lo + i);   // absolute position (flat, no wrap)
        float k_d = (tid < head_dim)
            ? fp8_to_float(k_cache[(pos * n_kv_heads + kv_head) * head_dim + tid])
            : 0.0f;
        // Full dot product, broadcast to every thread (block_reduce_sum returns the
        // reduced value to all lanes; attention scale is 1.0 for gemma4).
        float s = block_reduce_sum(q_d * k_d, smem);
        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);     // m=-inf on i=0 ⇒ alpha=0 (acc/l are 0)
        float p     = __expf(s - m_new);
        l   = l * alpha + p;
        float v_d = (tid < head_dim)
            ? fp8_to_float(v_cache[(pos * n_kv_heads + kv_head) * head_dim + tid])
            : 0.0f;
        acc = acc * alpha + p * v_d;
        m   = m_new;
    }

    if (tid < head_dim)
        output[q_head * head_dim + tid] = (l > 0.0f) ? acc / l : 0.0f;
}

// =========================================================================
// ─── Global Attention Decode (single token) ─────────────────────────────
// =========================================================================

// Global attention decode, single query token. FLASH (online-softmax) form —
// identical structure to the sliding kernel above, over the full linear context.
// Because it keeps a constant 32-float scratch instead of an O(ctx_len) score
// buffer, there is NO context-length cap (the old version needed (32+ctx_len)
// floats of dynamic shared memory and silently failed past ~25K tokens). This is
// the kernel change that unlocks 256K context.
// blockIdx.x = q_head; blockDim.x = head_dim (512). K=V unified, both stored.
// Launch: global_attn_decode_kernel<<<n_heads, head_dim, 32*sizeof(float)>>>
__global__ void global_attn_decode_kernel(
    float       *output,       // [n_heads × head_dim]
    const float *q,            // [n_heads × head_dim]
    const kv_t  *k_cache,      // [ctx_len][head_dim] FP8
    const kv_t  *v_cache,      // [ctx_len][head_dim] FP8
    int          n_heads,
    int          head_dim,
    int          ctx_len)
{
    extern __shared__ float smem[];    // [32] block_reduce scratch only
    int q_head = blockIdx.x;
    int tid    = threadIdx.x;

    float q_d = (tid < head_dim) ? q[q_head * head_dim + tid] : 0.0f;
    float acc = 0.0f;
    float m   = -INFINITY;
    float l   = 0.0f;

    for (int t = 0; t < ctx_len; t++) {
        float k_d = (tid < head_dim) ? fp8_to_float(k_cache[t * head_dim + tid]) : 0.0f;
        float s = block_reduce_sum(q_d * k_d, smem);
        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);
        float p     = __expf(s - m_new);
        l   = l * alpha + p;
        float v_d = (tid < head_dim) ? fp8_to_float(v_cache[t * head_dim + tid]) : 0.0f;
        acc = acc * alpha + p * v_d;
        m   = m_new;
    }

    if (tid < head_dim)
        output[q_head * head_dim + tid] = (l > 0.0f) ? acc / l : 0.0f;
}

// =========================================================================
// ─── GQA-broadcast global flash-decode (split-K) ─ DECODE-30-35 Step 1 ──
// =========================================================================
// The naive global_attn_decode_kernel above launches <<<n_heads=16, head_dim>>> over a
// 1-KV-head cache, addressing k_cache[t*head_dim+tid] by t/tid ONLY — so all 16 query-head
// blocks re-stream the identical KV head from DRAM (16x redundant; at 131k that is ~17 GB
// of the ~24.5 GB/token budget). This pair of kernels kills that: each block loads every
// K[t]/V[t] tile EXACTLY ONCE and serves all NH query heads from it (GQA-broadcast). The
// sequence is split across blocks (flash-decoding split-K) so the single global KV head
// still saturates bandwidth across SMs; per-(split,head) online-softmax partials are merged
// in flash_decode_combine_kernel. Result is the flash-attention split, exact up to FP
// reassociation (greedy argmax matches the single-pass kernel). Scale 1.0.

// Block reduction of NH values at once (mirrors block_reduce_sum's barrier discipline and
// broadcast-to-all-threads property; 2 __syncthreads regardless of NH). smem needs
// (blockDim.x/32)*NH floats.
template<int NH>
__device__ __forceinline__ void block_reduce_sum_vec(float *val, float *smem) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    int n_warps = (blockDim.x + 31) >> 5;
    #pragma unroll
    for (int h = 0; h < NH; h++) {
        float v = val[h];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, off);
        val[h] = v;
    }
    if (lane == 0) {
        #pragma unroll
        for (int h = 0; h < NH; h++) smem[wid*NH + h] = val[h];
    }
    __syncthreads();
    // Every warp reads all warp-partials and reduces → full sum broadcast to all threads.
    #pragma unroll
    for (int h = 0; h < NH; h++) {
        float v = (lane < n_warps) ? smem[lane*NH + h] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, off);
        val[h] = v;
    }
    __syncthreads();   // back-to-back call guard (see block_reduce_sum)
}

// Phase 1: one block per sequence split, one WARP per query head (NH warps/block).
// Each lane owns HD/32 elements in registers; per-key dots reduce via __shfl only —
// NO __syncthreads in the key loop. (The old one-thread-per-dim variant paid TWO
// block-wide barriers PER KEY in block_reduce_sum_vec — nsys showed it at ~322 µs for
// a ~250-token context, 12% of generation GPU time. Same fix as the sliding split-K
// kernel.)
//
// SMEM TILE STAGING: the first warp-per-head version let all NH warps stream the K/V
// tile straight from global memory, betting they would share L1/L2 lines. nsys disproved
// that at depth: warps drift apart and each re-reads the single KV head from DRAM —
// 16× traffic, 1.83 ms/call at a 34k context (19.5 GB/s effective; the whole long-ctx
// decode decay). Now the block cooperatively stages each TILE-key K/V slab into shared
// memory ONCE (vectorized uint4 loads), and all NH head-warps consume it from smem —
// DRAM traffic is 1× by construction. Two __syncthreads per TILE keys (not per key).
// Lanes own 4-byte dim groups (uint loads from smem: bank-conflict-free), so the per-key
// dot is 5 shuffles as before.
#define GEMMA4_GLOBAL_ATTN_TILE 32   // keys staged per smem tile (2*TILE*HD = 32 KB at HD 512)

// NKV global KV heads (GQA fan-out NH/NKV). NKV==1 is the 12B broadcast layout
// (cache [ctx_len][HD]); NKV>1 is the 31B GQA layout (cache [ctx_len][NKV][HD]),
// each query warp h reading KV head h/(NH/NKV). The block stages ALL NKV*HD K/V
// bytes of each token into smem once, so every head-warp consumes its KV head from
// smem with zero extra DRAM traffic regardless of NKV.
template<int NH, int NKV, int HD>
__global__ void global_attn_splitk_kernel(
    float *part_acc,                          // [n_splits][NH][HD] (unnormalized)
    float *part_m, float *part_l,             // [n_splits][NH]
    const float *q,                           // [NH][HD]
    const kv_t *k_cache, const kv_t *v_cache, // [ctx_len][NKV][HD]
    int head_dim, int ctx_len, int n_splits)
{
    // Stage TILE tokens of NKV*HD bytes each per K and V slab. Hold the smem footprint
    // (2*TILE*NKV*HD) at the 12B level (32 KB) regardless of NKV by shrinking TILE by NKV,
    // so the static __shared__ tile stays under the 48 KB default cap for the 31B (NKV=4).
    constexpr int TILE = GEMMA4_GLOBAL_ATTN_TILE / NKV;
    constexpr int E = HD / 128;               // uint words per lane (4 at HD 512)
    constexpr int GQ = NH / NKV;              // query heads per KV head
    static_assert(HD % 128 == 0, "uint lane slices require HD multiple of 128");
    static_assert(TILE >= 1, "GEMMA4_GLOBAL_ATTN_TILE must be >= NKV");
    __shared__ unsigned char sk[TILE * NKV * HD], sv[TILE * NKV * HD];
    const uint32_t rmask = 0u; const int treebase = 0;   // single-row: tree mask never applies (no-op)
    int h    = threadIdx.x >> 5;              // warp = query head
    int kvh  = h / GQ;                        // this query head's KV head
    int lane = threadIdx.x & 31;
    int split = blockIdx.x;
    int per = (ctx_len + n_splits - 1) / n_splits;
    int t0  = split * per;
    int t1  = min(t0 + per, ctx_len);

    float qreg[E][4], acc[E][4], m = -INFINITY, l = 0.0f;
    const float *qp = q + (size_t)h * HD;
    #pragma unroll
    for (int e = 0; e < E; e++)
        #pragma unroll
        for (int j = 0; j < 4; j++) { qreg[e][j] = qp[4*(lane + 32*e) + j]; acc[e][j] = 0.0f; }

    for (int tb = t0; tb < t1; tb += TILE) {
        int tn = min(TILE, t1 - tb);
        {   // cooperative stage: tn tokens of NKV*HD fp8 bytes for K and V (16 B vectors;
            // KV rows are HD-byte aligned so the uint4 view is safe). Bounds block-uniform.
            int nvec = tn * (NKV * HD / 16);
            const uint4 *gk = (const uint4 *)(k_cache + (size_t)tb * NKV * HD);
            const uint4 *gv = (const uint4 *)(v_cache + (size_t)tb * NKV * HD);
            uint4 *sk4 = (uint4 *)sk, *sv4 = (uint4 *)sv;
            for (int i = threadIdx.x; i < nvec; i += NH*32) { sk4[i] = gk[i]; sv4[i] = gv[i]; }
        }
        __syncthreads();
        for (int tt = 0; tt < tn; tt++) {
            int kabs = tb + tt;
            if (rmask && kabs >= treebase && !((rmask >> (kabs - treebase)) & 1u)) continue;  // non-ancestor tree key
            const unsigned int *kw = (const unsigned int *)(sk + ((size_t)tt * NKV + kvh) * HD);
            const unsigned int *vw = (const unsigned int *)(sv + ((size_t)tt * NKV + kvh) * HD);
            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < E; e++) {
                unsigned int k4 = kw[lane + 32*e];
                dot += qreg[e][0] * fp8_to_float((kv_t)( k4        & 0xFF))
                     + qreg[e][1] * fp8_to_float((kv_t)((k4 >>  8) & 0xFF))
                     + qreg[e][2] * fp8_to_float((kv_t)((k4 >> 16) & 0xFF))
                     + qreg[e][3] * fp8_to_float((kv_t)( k4 >> 24        ));
            }
            float s = warp_reduce_sum_all(dot);   // __shfl only — no block sync
            float mn = fmaxf(m, s), al = __expf(m - mn), p = __expf(s - mn);
            l = l*al + p;
            #pragma unroll
            for (int e = 0; e < E; e++) {
                unsigned int v4 = vw[lane + 32*e];
                acc[e][0] = acc[e][0]*al + p*fp8_to_float((kv_t)( v4        & 0xFF));
                acc[e][1] = acc[e][1]*al + p*fp8_to_float((kv_t)((v4 >>  8) & 0xFF));
                acc[e][2] = acc[e][2]*al + p*fp8_to_float((kv_t)((v4 >> 16) & 0xFF));
                acc[e][3] = acc[e][3]*al + p*fp8_to_float((kv_t)( v4 >> 24        ));
            }
            m = mn;
        }
        __syncthreads();   // tile consumed by every warp before the next stage overwrites
    }

    float *pa = part_acc + ((size_t)split*NH + h)*HD;
    #pragma unroll
    for (int e = 0; e < E; e++)
        #pragma unroll
        for (int j = 0; j < 4; j++) pa[4*(lane + 32*e) + j] = acc[e][j];
    if (lane == 0) { part_m[split*NH + h] = m; part_l[split*NH + h] = l; }
}

// Phase 2: one block per head merges that head's n_splits online-softmax partials.
template<int NH>
__global__ void flash_decode_combine_kernel(
    float *out,                               // [NH][head_dim]
    const float *part_acc, const float *part_m, const float *part_l,
    int head_dim, int n_splits)
{
    int h   = blockIdx.x;
    int tid = threadIdx.x;
    float M = -INFINITY;
    for (int s = 0; s < n_splits; s++) M = fmaxf(M, part_m[s*NH + h]);
    float L = 0.0f, accv = 0.0f;
    for (int s = 0; s < n_splits; s++) {
        float ms = part_m[s*NH + h];
        if (ms == -INFINITY) continue;             // empty split (n_splits>ctx edge)
        float scale = __expf(ms - M);
        L    += part_l[s*NH + h] * scale;
        accv += part_acc[((size_t)s*NH + h)*head_dim + tid] * scale;
    }
    if (tid < head_dim)
        out[h*head_dim + tid] = (L > 0.0f) ? accv / L : 0.0f;
}

// =========================================================================
// ─── Sliding-window split-K flash decode (warp-per-KV-head) ─────────────
// =========================================================================
// The naive sliding_attn_decode_kernel above pays a block_reduce_sum — TWO full
// __syncthreads — PER KEY, launched as only <<<16 heads, 256 threads>>> (~1 block/SM,
// nothing to hide the barrier latency behind). With 40 sliding layers back-to-back on
// one stream that serial sync chain, not KV bandwidth, is what decays decode from
// ~20 tok/s to ~10 as the 1024-token window fills (~1.24 µs/key/layer measured).
// Same fix as the tiled prefill kernels (see flash_prefill_*_tiled): each lane owns
// head_dim/32 elements in registers and dots reduce via warp_reduce_sum_all — NO
// __syncthreads anywhere in the key loop — plus the global path's split-K so the
// ≤1024-key window spreads across blocks/SMs instead of one serial scan.
//
// Geometry: one warp per KV HEAD (NKV=8 warps = 256 threads/block), each warp serving
// its GQA group of GQ = NH/NKV = 2 query heads. Chosen over warp-per-query-head
// (16 warps) because each K/V tile is then loaded exactly ONCE per block (the
// GQA-broadcast principle of global_attn_splitk_kernel) for two extra register
// slices per lane (qreg+acc: 2×8+2×8 = 32 floats — half the global kernel's 64).
// blockIdx.x = split of the window range [n_tokens - window_len, n_tokens) over the
// FLAT [capacity][n_kv_heads][head_dim] FP8 cache (absolute positions, scale 1.0).
// Per-(split, q_head) online-softmax partials are written in the [n_splits][NH][head_dim]
// layout flash_decode_combine_kernel<NH> expects; an empty split (range exhausted, or
// window_len == 0) writes m = -INFINITY which the combine pass skips.
//
// HD is a TEMPLATE parameter, not a runtime arg, on purpose: the per-lane slice width
// (HD/32) bounds every e-loop below, and ptxas only honors #pragma unroll — and only
// keeps qreg/acc/kd/vd in registers — when that trip count is a compile-time constant.
// With a runtime head_dim the dynamic indices demoted all four arrays to local memory
// (192-byte stack frame, ~80 LDL/STL per key per warp in SASS), silently forfeiting the
// register-slice design this kernel exists for. constexpr slice mirrors how
// global_attn_splitk_kernel earns its 0-byte frame via the compile-time NH bound.

template<int NH, int NKV, int HD>
__global__ void sliding_attn_splitk_kernel(
    float *part_acc,                          // [n_splits][NH][HD] (unnormalized)
    float *part_m, float *part_l,             // [n_splits][NH]
    const float *q,                           // [NH][HD]
    const kv_t *k_cache, const kv_t *v_cache, // RING [cap][NKV][HD] FP8
    int window, int n_tokens, int n_splits, int cap)
{
    constexpr int GQ = NH / NKV;              // query heads per KV head (GQA group) = 2
    constexpr int slice = HD / 32;            // 8 floats/lane at HD 256
    static_assert(HD % 32 == 0, "lane-strided slices require HD multiple of warp size");
    int kv_head = threadIdx.x >> 5;           // one warp per KV head
    int lane    = threadIdx.x & 31;
    int split   = blockIdx.x;

    int window_len = min(n_tokens, window);
    int lo  = n_tokens - window_len;          // first absolute position to attend
    int per = (window_len + n_splits - 1) / n_splits;
    int i0  = split * per;
    int i1  = min(i0 + per, window_len);

    float qreg[GQ][slice], acc[GQ][slice], m[GQ], l[GQ];
    #pragma unroll
    for (int g = 0; g < GQ; g++) {
        const float *qp = q + (size_t)(kv_head*GQ + g)*HD;
        #pragma unroll
        for (int e = 0; e < slice; e++) { qreg[g][e] = qp[lane + 32*e]; acc[g][e] = 0.0f; }
        m[g] = -INFINITY; l[g] = 0.0f;
    }

    for (int i = i0; i < i1; i++) {
        size_t pos = (size_t)(lo + i) % (size_t)cap;   // ring slot for absolute pos lo+i
        const kv_t *kp = k_cache + (pos*NKV + kv_head)*HD;
        const kv_t *vp = v_cache + (pos*NKV + kv_head)*HD;
        float kd[slice], vd[slice];
        #pragma unroll
        for (int e = 0; e < slice; e++) {     // K/V tile read ONCE, reused for all GQ heads
            kd[e] = fp8_to_float(kp[lane + 32*e]);
            vd[e] = fp8_to_float(vp[lane + 32*e]);
        }
        #pragma unroll
        for (int g = 0; g < GQ; g++) {
            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < slice; e++) dot += qreg[g][e] * kd[e];
            float s = warp_reduce_sum_all(dot);   // __shfl only — no block sync
            float mn = fmaxf(m[g], s), al = __expf(m[g] - mn), p = __expf(s - mn);
            l[g] = l[g]*al + p;
            #pragma unroll
            for (int e = 0; e < slice; e++) acc[g][e] = acc[g][e]*al + p*vd[e];
            m[g] = mn;
        }
    }

    #pragma unroll
    for (int g = 0; g < GQ; g++) {
        int h = kv_head*GQ + g;               // same GQA mapping as q_head/(NH/NKV)
        float *pa = part_acc + ((size_t)split*NH + h)*HD;
        #pragma unroll
        for (int e = 0; e < slice; e++) pa[lane + 32*e] = acc[g][e];
        if (lane == 0) { part_m[split*NH + h] = m[g]; part_l[split*NH + h] = l[g]; }
    }
}

// =========================================================================
// ─── Row-batched spec-verify flash decode (one causal launch for K rows) ─
// =========================================================================
// The spec-verify in decode_batched forwards K=D+1 tokens; each row i attends the
// cache up to its own causal bound n_tokens0+i. The original loop launched K
// (splitk + combine) pairs per layer, serialized on the shared d_fa_* scratch —
// 3K launches/layer and zero row overlap. These kernels run ALL rows in one grid
// (blockIdx.y = row): row r owns scratch slots [r*GEMMA4_GLOBAL_MAX_SPLITS, ...)
// and recomputes splits_r with the EXACT formula of its single-row launcher, so
// every row's split partition, per-warp geometry, and combine merge order are
// BIT-IDENTICAL to the per-row launches they replace.

// Per-row split count, replicating global_attn_decode_broadcast / sliding_attn_
// decode_broadcast clamps (chunk is 64 for both; len ≥ 1 in every verify row).
static __device__ __forceinline__ int attn_row_splits(int len, int chunk) {
    int s = (len + chunk - 1) / chunk;
    if (s < 1) s = 1;
    if (s > GEMMA4_GLOBAL_MAX_SPLITS) s = GEMMA4_GLOBAL_MAX_SPLITS;
    if (s > len && len > 0) s = len;
    return s;
}

// Same smem tile staging as global_attn_splitk_kernel (see comment there): the block
// stages each TILE-key K/V slab once and all NH head-warps consume it from smem —
// without it every warp re-streamed the single KV head from DRAM (16× traffic), which
// made each verify row pay the full long-ctx decay. All tile bounds are block-uniform
// (split/ctx_len derive from blockIdx and the row), so the __syncthreads are safe; the
// tail-block early-return happens before the first barrier and is block-uniform too.
template<int NH, int NKV, int HD>
__global__ void global_attn_splitk_rows_kernel(
    float *part_acc, float *part_m, float *part_l,   // slot r*MAX_SPLITS+split
    const float *q,                                  // [K][NH*HD] (row-major)
    const kv_t *k_cache, const kv_t *v_cache,        // [capacity][NKV][HD]
    int n_tokens0,                                   // row r attends n_tokens0 + r keys
    const int *n_tokens0_ptr,                        // non-NULL: device override (graph path)
    const uint32_t *anc = nullptr)                   // TREE: per-row ancestor bitmask
{
    constexpr int TILE = GEMMA4_GLOBAL_ATTN_TILE / NKV;   // smem held at the 12B footprint
    constexpr int E = HD / 128;
    constexpr int GQ = NH / NKV;
    static_assert(HD % 128 == 0, "uint lane slices require HD multiple of 128");
    static_assert(TILE >= 1, "GEMMA4_GLOBAL_ATTN_TILE must be >= NKV");
    __shared__ unsigned char sk[TILE * NKV * HD], sv[TILE * NKV * HD];
    int r = blockIdx.y;
    if (n_tokens0_ptr) n_tokens0 = *n_tokens0_ptr;
    // TREE mask: when `anc` is set, a key in the tree region [treebase, ctx_len) is attended
    // only if it is this row's ancestor (bit set). The committed prefix is [0, pos); row 0 (the
    // root) sits at pos, so treebase = pos = n_tokens0-1 (n_tokens0 is pos+1) — DERIVED here so
    // the graph path needs no extra scalar. anc==NULL (linear) ⇒ no masking ⇒ bit-identical.
    uint32_t rmask = anc ? anc[r] : 0u;
    int treebase = n_tokens0 - 1;
    int ctx_len = n_tokens0 + r;
    int n_splits = attn_row_splits(ctx_len, GEMMA4_GLOBAL_SPLIT_CHUNK);
    int split = blockIdx.x;
    if (split >= n_splits) return;                   // tail blocks of shorter rows
    int h    = threadIdx.x >> 5;
    int kvh  = h / GQ;
    int lane = threadIdx.x & 31;
    int per = (ctx_len + n_splits - 1) / n_splits;
    int t0  = split * per;
    int t1  = min(t0 + per, ctx_len);

    float qreg[E][4], acc[E][4], m = -INFINITY, l = 0.0f;
    const float *qp = q + (size_t)r * NH * HD + (size_t)h * HD;
    #pragma unroll
    for (int e = 0; e < E; e++)
        #pragma unroll
        for (int j = 0; j < 4; j++) { qreg[e][j] = qp[4*(lane + 32*e) + j]; acc[e][j] = 0.0f; }

    for (int tb = t0; tb < t1; tb += TILE) {
        int tn = min(TILE, t1 - tb);
        {
            int nvec = tn * (NKV * HD / 16);
            const uint4 *gk = (const uint4 *)(k_cache + (size_t)tb * NKV * HD);
            const uint4 *gv = (const uint4 *)(v_cache + (size_t)tb * NKV * HD);
            uint4 *sk4 = (uint4 *)sk, *sv4 = (uint4 *)sv;
            for (int i = threadIdx.x; i < nvec; i += NH*32) { sk4[i] = gk[i]; sv4[i] = gv[i]; }
        }
        __syncthreads();
        for (int tt = 0; tt < tn; tt++) {
            int kabs = tb + tt;
            if (rmask && kabs >= treebase && !((rmask >> (kabs - treebase)) & 1u)) continue;  // non-ancestor tree key
            const unsigned int *kw = (const unsigned int *)(sk + ((size_t)tt * NKV + kvh) * HD);
            const unsigned int *vw = (const unsigned int *)(sv + ((size_t)tt * NKV + kvh) * HD);
            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < E; e++) {
                unsigned int k4 = kw[lane + 32*e];
                dot += qreg[e][0] * fp8_to_float((kv_t)( k4        & 0xFF))
                     + qreg[e][1] * fp8_to_float((kv_t)((k4 >>  8) & 0xFF))
                     + qreg[e][2] * fp8_to_float((kv_t)((k4 >> 16) & 0xFF))
                     + qreg[e][3] * fp8_to_float((kv_t)( k4 >> 24        ));
            }
            float s = warp_reduce_sum_all(dot);
            float mn = fmaxf(m, s), al = __expf(m - mn), p = __expf(s - mn);
            l = l*al + p;
            #pragma unroll
            for (int e = 0; e < E; e++) {
                unsigned int v4 = vw[lane + 32*e];
                acc[e][0] = acc[e][0]*al + p*fp8_to_float((kv_t)( v4        & 0xFF));
                acc[e][1] = acc[e][1]*al + p*fp8_to_float((kv_t)((v4 >>  8) & 0xFF));
                acc[e][2] = acc[e][2]*al + p*fp8_to_float((kv_t)((v4 >> 16) & 0xFF));
                acc[e][3] = acc[e][3]*al + p*fp8_to_float((kv_t)( v4 >> 24        ));
            }
            m = mn;
        }
        __syncthreads();
    }

    size_t slot = (size_t)r * GEMMA4_GLOBAL_MAX_SPLITS + split;
    float *pa = part_acc + (slot*NH + h)*HD;
    #pragma unroll
    for (int e = 0; e < E; e++)
        #pragma unroll
        for (int j = 0; j < 4; j++) pa[4*(lane + 32*e) + j] = acc[e][j];
    if (lane == 0) { part_m[slot*NH + h] = m; part_l[slot*NH + h] = l; }
}

template<int NH, int NKV, int HD>
__global__ void sliding_attn_splitk_rows_kernel(
    float *part_acc, float *part_m, float *part_l,   // slot r*MAX_SPLITS+split
    const float *q,                                  // [K][NH*HD]
    const kv_t *k_cache, const kv_t *v_cache,        // RING [cap][NKV][HD]
    int window, int n_tokens0, int cap,
    const int *n_tokens0_ptr,                        // non-NULL: device override (graph path)
    const uint32_t *anc = nullptr)                   // TREE: per-row ancestor bitmask
{
    constexpr int GQ = NH / NKV;
    constexpr int slice = HD / 32;
    int r = blockIdx.y;
    if (n_tokens0_ptr) n_tokens0 = *n_tokens0_ptr;
    uint32_t rmask = anc ? anc[r] : 0u;   // TREE mask; 0 (NULL) ⇒ no masking (linear path)
    int treebase = n_tokens0 - 1;         // committed pos (n_tokens0 = pos+1); DERIVED for graph-safety
    int n_tokens = n_tokens0 + r;
    int window_len = min(n_tokens, window);
    int n_splits = attn_row_splits(window_len, GEMMA4_SLIDING_SPLIT_CHUNK);
    int split = blockIdx.x;
    if (split >= n_splits) return;
    int kv_head = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;

    int lo  = n_tokens - window_len;
    int per = (window_len + n_splits - 1) / n_splits;
    int i0  = split * per;
    int i1  = min(i0 + per, window_len);

    float qreg[GQ][slice], acc[GQ][slice], m[GQ], l[GQ];
    #pragma unroll
    for (int g = 0; g < GQ; g++) {
        const float *qp = q + (size_t)r * NH * HD + (size_t)(kv_head*GQ + g)*HD;
        #pragma unroll
        for (int e = 0; e < slice; e++) { qreg[g][e] = qp[lane + 32*e]; acc[g][e] = 0.0f; }
        m[g] = -INFINITY; l[g] = 0.0f;
    }

    for (int i = i0; i < i1; i++) {
        int ap = lo + i;                               // absolute position of this key
        if (rmask && ap >= treebase && !((rmask >> (ap - treebase)) & 1u)) continue;  // non-ancestor tree key
        size_t pos = (size_t)ap % (size_t)cap;         // ring slot for absolute pos lo+i
        const kv_t *kp = k_cache + (pos*NKV + kv_head)*HD;
        const kv_t *vp = v_cache + (pos*NKV + kv_head)*HD;
        float kd[slice], vd[slice];
        #pragma unroll
        for (int e = 0; e < slice; e++) {
            kd[e] = fp8_to_float(kp[lane + 32*e]);
            vd[e] = fp8_to_float(vp[lane + 32*e]);
        }
        #pragma unroll
        for (int g = 0; g < GQ; g++) {
            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < slice; e++) dot += qreg[g][e] * kd[e];
            float s = warp_reduce_sum_all(dot);
            float mn = fmaxf(m[g], s), al = __expf(m[g] - mn), p = __expf(s - mn);
            l[g] = l[g]*al + p;
            #pragma unroll
            for (int e = 0; e < slice; e++) acc[g][e] = acc[g][e]*al + p*vd[e];
            m[g] = mn;
        }
    }

    size_t slot = (size_t)r * GEMMA4_GLOBAL_MAX_SPLITS + split;
    #pragma unroll
    for (int g = 0; g < GQ; g++) {
        int h = kv_head*GQ + g;
        float *pa = part_acc + (slot*NH + h)*HD;
        #pragma unroll
        for (int e = 0; e < slice; e++) pa[lane + 32*e] = acc[g][e];
        if (lane == 0) { part_m[slot*NH + h] = m[g]; part_l[slot*NH + h] = l[g]; }
    }
}

// Merge each row's partials. window > 0 → sliding row length min(n_tokens0+r, window)
// (chunk GEMMA4_SLIDING_SPLIT_CHUNK); window == 0 → global (len = n_tokens0+r, chunk
// GEMMA4_GLOBAL_SPLIT_CHUNK). Merge order s = 0..splits_r-1 matches the single-row
// flash_decode_combine_kernel exactly.
template<int NH>
__global__ void flash_decode_combine_rows_kernel(
    float *out,                                      // [K][out_stride]
    const float *part_acc, const float *part_m, const float *part_l,
    int head_dim, int window, int n_tokens0, const int *n_tokens0_ptr, int out_stride)
{
    int r   = blockIdx.y;
    int h   = blockIdx.x;
    int tid = threadIdx.x;
    if (n_tokens0_ptr) n_tokens0 = *n_tokens0_ptr;
    int len = n_tokens0 + r;
    int chunk = GEMMA4_GLOBAL_SPLIT_CHUNK;
    if (window > 0) { len = min(len, window); chunk = GEMMA4_SLIDING_SPLIT_CHUNK; }
    int n_splits = attn_row_splits(len, chunk);
    size_t base = (size_t)r * GEMMA4_GLOBAL_MAX_SPLITS;
    float M = -INFINITY;
    for (int s = 0; s < n_splits; s++) M = fmaxf(M, part_m[(base + s)*NH + h]);
    float L = 0.0f, accv = 0.0f;
    for (int s = 0; s < n_splits; s++) {
        float ms = part_m[(base + s)*NH + h];
        if (ms == -INFINITY) continue;
        float scale = __expf(ms - M);
        L    += part_l[(base + s)*NH + h] * scale;
        accv += part_acc[((base + s)*NH + h)*head_dim + tid] * scale;
    }
    if (tid < head_dim)
        out[(size_t)r*out_stride + h*head_dim + tid] = (L > 0.0f) ? accv / L : 0.0f;
}

// ─── Tiled-GEMM flash-prefill attention (online softmax over K/V tiles) ──
// FlashAttention-2's tiling expressed at host level over cuBLAS tensor-core
// GEMMs: attention = loop over K/V tiles of GEMMA4_FP_TILE_K positions; per
// tile S = Q·K_tileᵀ (one strided-batched BF16 GEMM over all heads), a fused
// kernel folds the tile into the running online softmax (per-row max m, sum
// l, rescale of the fp32 O accumulator), then O += P·V_tile (GEMM, beta=1).
// Memory is O(chunk × tile) — any context length — and the quadratic
// attention work runs on tensor cores (~89 TFLOPS) instead of the scalar
// warp-per-query kernels above (~9 TFLOPS peak, measured ~20% of that),
// which were the cause of the long-context prefill decay (298→83 tok/s
// over 1.4k→44k, measured clean 2026-06-11).

#define GEMMA4_FP_CHUNK  8192   // flash-prefill chunk (queries per pass)
#define GEMMA4_FP_TILE_K 2048   // K/V positions per attention tile

// Per-(head,query) online-softmax state init: m = -inf, l = 0.
__global__ void fp_ml_init_kernel(float *m, float *l, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { m[i] = -INFINITY; l[i] = 0.0f; }
}

// GQA broadcast straight from the chunk's fp32 K/V rows to BF16 GEMM operands:
// src [rows][n_kv_heads][head_dim] fp32 → dst [rows][n_heads][head_dim] bf16.
// grid=(ceil(head_dim/256), n_heads, rows).
__global__ void kv_broadcast_f32_bf16_kernel(
    __nv_bfloat16 *dst, const float *src,
    int rows, int n_heads, int n_kv_heads, int head_dim)
{
    int d   = blockIdx.x * blockDim.x + threadIdx.x;
    int h   = blockIdx.y;
    int row = blockIdx.z;
    if (d >= head_dim || row >= rows) return;
    int kvh = h / (n_heads / n_kv_heads);
    dst[((size_t)row * n_heads + h) * head_dim + d] =
        __float2bfloat16(src[((size_t)row * n_kv_heads + kvh) * head_dim + d]);
}

// History K/V tile: FP8 cache positions [t0, t0+tn) → BF16 GEMM operands with
// GQA broadcast, [tn][n_heads*head_dim]. FP8 E4M3 → BF16 is exact (3 ≤ 7
// mantissa bits, exponent range covered), so this loses nothing vs the scalar
// kernels' per-element fp8_to_float. Cache layout is flat-by-absolute-position
// [pos][n_kv_heads][head_dim] (global layers: n_kv_heads == 1).
// grid=(ceil(head_dim/256), n_heads, tn).
__global__ void fp_hist_tile_bf16_kernel(
    __nv_bfloat16 *kt, __nv_bfloat16 *vt,
    const kv_t *kc, const kv_t *vc,
    int t0, int tn, int n_heads, int n_kv_heads, int head_dim, int cap)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y, t = blockIdx.z;
    if (d >= head_dim || t >= tn) return;
    int kvh = h / (n_heads / n_kv_heads);
    size_t src = (((size_t)(t0 + t) % (size_t)cap) * n_kv_heads + kvh) * head_dim + d;
    size_t dst = ((size_t)t * n_heads + h) * head_dim + d;
    kt[dst] = __float2bfloat16(fp8_to_float(kc[src]));
    vt[dst] = __float2bfloat16(fp8_to_float(vc[src]));
}

// Fold one S tile into the running online softmax. S/P are col-major with
// ld = t_ld; the column of (query qi, head h) starts at qi·csq + h·csh —
// S(j, col) is the score of key (kbase+j) against query (q0+i), exactly as
// written by the preceding K_tileᵀ·Q GEMM, so a query's tile scores are
// CONTIGUOUS. Two layouts share this kernel:
//   sliding (GQA, strided-batched GEMM): csq = t_ld, csh = t_ld·chunk
//   global  (1 KV head, ONE wide GEMM over [hd × HEADS·cn] Q):
//           csq = HEADS·t_ld, csh = t_ld
// Per (query, head) block: masked tile max → m_new, α = e^(m−m_new),
// P = e^(s−m_new) (masked → 0), l = l·α + Σ P, O-row *= α (or := 0 on the
// query's first contributing tile, which also clears uninitialized scratch —
// NaN-proof, unlike multiplying by α=0). Masking: causal ka ≤ pos plus the
// sliding lower bound ka > pos−window when window > 0. Scale 1.0 (gemma4).
// grid=(qn, n_heads), block=256, smem=32 floats.
__global__ void fp_online_softmax_kernel(
    const __nv_bfloat16 *S, __nv_bfloat16 *P, float *O,
    float *mbuf, float *lbuf,
    int tn, long long csq, long long csh,
    int q0, int kbase, int abs0, int window,
    int head_dim, int oq, int ml_stride)
{
    extern __shared__ float red[];
    int qi = q0 + blockIdx.x, h = blockIdx.y;
    int tid = threadIdx.x, nt = blockDim.x;
    int pos = abs0 + qi;
    const __nv_bfloat16 *sc = S + (long long)qi * csq + (long long)h * csh;
    __nv_bfloat16 *pc = P + (long long)qi * csq + (long long)h * csh;

    float mx = -INFINITY;
    for (int j = tid; j < tn; j += nt) {
        int ka = kbase + j;
        if (ka <= pos && (window <= 0 || ka > pos - window))
            mx = fmaxf(mx, __bfloat162float(sc[j]));
    }
    mx = block_reduce_max(mx, red);

    int   mli   = h * ml_stride + qi;
    float m_old = mbuf[mli];
    float m_new = fmaxf(m_old, mx);
    if (m_new == -INFINITY) {
        // No valid key for this query yet (entire tile masked): P must still
        // be zeroed for the unconditional P·V GEMM; m/l/O stay untouched.
        for (int j = tid; j < tn; j += nt) pc[j] = __float2bfloat16(0.0f);
        return;
    }

    float lsum = 0.0f;
    for (int j = tid; j < tn; j += nt) {
        int  ka = kbase + j;
        bool ok = ka <= pos && (window <= 0 || ka > pos - window);
        float p = ok ? __expf(__bfloat162float(sc[j]) - m_new) : 0.0f;
        pc[j] = __float2bfloat16(p);
        lsum += p;
    }
    lsum = block_reduce_sum(lsum, red);

    float *orow = O + (size_t)qi * oq + (size_t)h * head_dim;
    if (m_old == -INFINITY) {
        // First contributing tile: O holds garbage (previous layer / fresh
        // cudaMalloc) — assign zero rather than scale by α=0 (0·NaN = NaN).
        for (int e = tid; e < head_dim; e += nt) orow[e] = 0.0f;
        if (tid == 0) { mbuf[mli] = m_new; lbuf[mli] = lsum; }
        return;
    }
    float alpha = __expf(m_old - m_new);
    if (alpha != 1.0f)
        for (int e = tid; e < head_dim; e += nt) orow[e] *= alpha;
    if (tid == 0) { mbuf[mli] = m_new; lbuf[mli] = lbuf[mli] * alpha + lsum; }
}

// Final pass: O /= l per (query, head) row. grid=(cn, n_heads).
__global__ void fp_attn_norm_kernel(
    float *O, const float *lbuf, int head_dim, int oq, int ml_stride)
{
    int qi = blockIdx.x, h = blockIdx.y;
    float l = lbuf[h * ml_stride + qi];
    float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    float *orow = O + (size_t)qi * oq + (size_t)h * head_dim;
    for (int e = threadIdx.x; e < head_dim; e += blockDim.x) orow[e] *= inv;
}

// =========================================================================
// ─── Batched-prefill kernels (Step 2 Phase 2 + Step 3 attention) ────────
// =========================================================================
// Token-major "rows" variants of decode_layer's elementwise steps, plus a
// flash-style (online-softmax) prefill attention. All operate on activations
// laid out [rows × ... × head_dim] with rows = ubatch tokens, so each block
// handles one (token[,head]) slice. Math matches the single-token kernels above.

// gelu_pytorch_tanh, needed by geglu_bf16_kernel below (the f32 geglu_kernel and
// its gelu_tanh live further down; declare the helper here to keep this together).
__device__ inline float gemma4_gelu_tanh(float x) {
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

// Per-head RMSNorm over [rows][n_heads][head_dim]. grid=(n_heads, rows),
// block=head_dim, smem=32 floats. weight [head_dim] or NULL.
__global__ void per_head_rms_norm_rows_kernel(
    float *qk, const float *weight, int n_heads, int head_dim, int rows, float eps)
{
    extern __shared__ float smem[];
    int head = blockIdx.x, row = blockIdx.y;
    if (row >= rows || head >= n_heads) return;
    int tid = threadIdx.x;
    float *h = qk + ((size_t)row * n_heads + head) * head_dim;
    float ss = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) ss += h[i] * h[i];
    ss = block_reduce_sum(ss, smem);
    float rms = rsqrtf(ss / head_dim + eps);
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float w = weight ? weight[i] : 1.0f;
        h[i] = h[i] * rms * w;
    }
}

// RMSNorm over [rows][n] writing BF16 directly (fuses the f32→bf16 convert that
// would otherwise precede each GEMM whose input is a normed hidden). weight [n].
__global__ void rms_norm_rows_bf16_kernel(
    __nv_bfloat16 *out, const float *x, const float *weight, int n, int rows, float eps)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const float *xr = x + (size_t)row * n;
    __nv_bfloat16 *orow = out + (size_t)row * n;
    float ss = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_reduce_sum(ss, smem);
    float rms = rsqrtf(ss / n + eps);
    for (int i = tid; i < n; i += blockDim.x) {
        float w = weight ? weight[i] : 1.0f;
        orow[i] = __float2bfloat16(xr[i] * rms * w);
    }
}

// GeGLU writing BF16 directly (feeds the down GEMM): out = gelu(gate)*up.
__global__ void geglu_bf16_kernel(
    __nv_bfloat16 *out, const float *gate, const float *up, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(gemma4_gelu_tanh(gate[i]) * up[i]);
}

// SiLU-GLU (Qwen3 FFN) writing BF16: out = silu(gate)*up, silu(x)=x*sigmoid(x).
__global__ void silu_glu_bf16_kernel(
    __nv_bfloat16 *out, const float *gate, const float *up, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    out[i] = __float2bfloat16((g / (1.0f + __expf(-g))) * up[i]);
}

// Broadcast K/V heads to all query heads for batched-attention GEMMs (GQA):
// src [rows][n_kv_heads][head_dim] → dst [rows][n_heads][head_dim], dst head h
// copies src kv head h/(n_heads/n_kv_heads). grid=(ceil(head_dim/256), n_heads, rows).
__global__ void kv_broadcast_bf16_kernel(
    __nv_bfloat16 *dst, const __nv_bfloat16 *src,
    int rows, int n_heads, int n_kv_heads, int head_dim)
{
    int d   = blockIdx.x * blockDim.x + threadIdx.x;
    int h   = blockIdx.y;
    int row = blockIdx.z;
    if (d >= head_dim || h >= n_heads || row >= rows) return;
    int kvh = h / (n_heads / n_kv_heads);
    dst[((size_t)row * n_heads + h) * head_dim + d] =
        src[((size_t)row * n_kv_heads + kvh) * head_dim + d];
}

// Masked row-softmax over a BATCH of col-major score matrices S [n_heads][N×N]
// (batch stride N*N). grid=(N, n_heads). Same masking as the single-matrix version.
__global__ void attn_softmax_batched_kernel(
    const float *S, __nv_bfloat16 *P, int N, int window)
{
    extern __shared__ float red[];
    int i = blockIdx.x, h = blockIdx.y;
    if (i >= N) return;
    const float   *Sh = S + (size_t)h * N * N;
    __nv_bfloat16 *Ph = P + (size_t)h * N * N;
    int tid = threadIdx.x, nt = blockDim.x;
    int start = (window > 0) ? max(0, i - (window - 1)) : 0;
    float m = -1e30f;
    for (int j = start + tid; j <= i; j += nt) m = fmaxf(m, Sh[(size_t)i + (size_t)j * N]);
    m = block_reduce_max(m, red);
    float l = 0.0f;
    for (int j = start + tid; j <= i; j += nt) l += __expf(Sh[(size_t)i + (size_t)j * N] - m);
    l = block_reduce_sum(l, red);
    float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (int j = tid; j < N; j += nt) {
        float p = 0.0f;
        if (j >= start && j <= i) p = __expf(Sh[(size_t)i + (size_t)j * N] - m) * inv;
        Ph[(size_t)i + (size_t)j * N] = __float2bfloat16(p);
    }
}

// RECTANGULAR causal softmax over KEY-MAJOR scores (base>0 chunked-prefill continuation):
// per head h, S_[h] is an (Scols x T) col-major matrix (ld=Scols, batch stride Scols*T) whose
// column i holds query (base+i)'s scores against keys j in [0,Scols). Query i attends keys
// j <= base+i; masked entries write fp16 zero. Being contiguous over keys j, every loop is
// fully coalesced — unlike attn_softmax_batched_kernel above whose row walk strides by N
// (measured 7 ms at N=2048). grid=(T, n_heads), block 256, smem 32 floats.
__global__ void attn_softmax_rect_kernel(const float *S_, __half *P, int T, int Scols, int base) {
    extern __shared__ float red[];
    int i = blockIdx.x, h = blockIdx.y;
    if (i >= T) return;
    const float *col = S_ + (size_t)h * Scols * T + (size_t)i * Scols;
    __half     *pc  = P  + (size_t)h * Scols * T + (size_t)i * Scols;
    int tid = threadIdx.x, nt = blockDim.x;
    int hi = base + i;                        // inclusive causal bound
    float m = -1e30f;
    for (int j = tid; j <= hi; j += nt) m = fmaxf(m, col[j]);
    m = block_reduce_max(m, red);
    float l = 0.0f;
    for (int j = tid; j <= hi; j += nt) l += __expf(col[j] - m);
    l = block_reduce_sum(l, red);
    float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (int j = tid; j < Scols; j += nt) {
        float p = (j <= hi) ? __expf(col[j] - m) * inv : 0.0f;
        pc[j] = __float2half(p);
    }
}

// NEOX RoPE over [rows][n_heads][head_dim] for Q and [rows][n_kv_heads][head_dim]
// for K. Position of row r is base_pos+r. ff=freq_factors (global) or NULL (=1,
// sliding). grid=(n_heads, rows), block=head_dim/2. Matches rope_*_kernel above.
__global__ void rope_rows_kernel(
    float *q, float *k, int base_pos, int n_heads, int n_kv_heads,
    int head_dim, int rows, float theta_base, const float *freq_factors,
    const int *base_pos_ptr = nullptr,               // non-NULL: device override (graph path)
    const int *depth = nullptr)                      // TREE: row's pos offset = depth[row] (else row)
{
    int d = threadIdx.x, half = head_dim / 2;
    if (d >= half) return;
    int head = blockIdx.x, row = blockIdx.y;
    if (row >= rows) return;
    if (base_pos_ptr) base_pos = *base_pos_ptr;
    int pos = base_pos + (depth ? depth[row] : row);
    float ff    = freq_factors ? freq_factors[d] : 1.0f;
    float theta = powf(theta_base, -2.0f * d / head_dim) / ff;
    float c = cosf(pos * theta), s = sinf(pos * theta);
    float *qh = q + ((size_t)row * n_heads + head) * head_dim;
    float q0 = qh[d], q1 = qh[d + half];
    qh[d] = q0 * c - q1 * s;  qh[d + half] = q0 * s + q1 * c;
    if (head < n_kv_heads) {
        float *kh = k + ((size_t)row * n_kv_heads + head) * head_dim;
        float k0 = kh[d], k1 = kh[d + half];
        kh[d] = k0 * c - k1 * s;  kh[d + half] = k0 * s + k1 * c;
    }
}


// Masked row-softmax over a col-major score matrix S [N×N] (S[i + j*N] = score of
// query i, key j, produced by a QK^T GEMM). For each query row i, softmax over
// keys j ∈ [start_i, i] (causal; sliding adds start_i = max(0, i-window+1)); all
// other j set to 0 so the following SV GEMM masks correctly. Writes BF16 P in the
// same col-major layout. Attention scale = 1.0 (gemma4). One block per query row.
__global__ void attn_softmax_colmajor_kernel(
    const float *S, __nv_bfloat16 *P, int N, int window)
{
    extern __shared__ float red[];
    int i = blockIdx.x;
    if (i >= N) return;
    int tid = threadIdx.x, nt = blockDim.x;
    int start = (window > 0) ? max(0, i - (window - 1)) : 0;

    float m = -1e30f;
    for (int j = start + tid; j <= i; j += nt)
        m = fmaxf(m, S[(size_t)i + (size_t)j * N]);
    m = block_reduce_max(m, red);

    float l = 0.0f;
    for (int j = start + tid; j <= i; j += nt)
        l += __expf(S[(size_t)i + (size_t)j * N] - m);
    l = block_reduce_sum(l, red);
    float inv = (l > 0.0f) ? 1.0f / l : 0.0f;

    for (int j = tid; j < N; j += nt) {
        float p = 0.0f;
        if (j >= start && j <= i)
            p = __expf(S[(size_t)i + (size_t)j * N] - m) * inv;
        P[(size_t)i + (size_t)j * N] = __float2bfloat16(p);
    }
}

// Scatter the batch's final K/V into the persistent sliding ring. Only the last
// `count` tokens (= min(rows, window)) survive the window, so we write exactly
// those — slots are then collision-free (residues unique over any window span)
// and hold their true most-recent occupant. kvhd = n_kv_heads*head_dim.
// k/vcache point at the layer's ring base [window][kvhd]. grid=(ceil(kvhd/256),
// count), block=256.
// RING per-position write: token at batch index t goes to absolute position base+t,
// stored at ring slot (base+t) % cap. cap == sliding_kv_capacity; for a context that
// never exceeds cap this is identity (flat). `window` is unused (kept for ABI parity
// with the global writer's call shape).
__global__ void kv_write_sliding_kernel(
    kv_t *kcache, kv_t *vcache, const float *kb, const float *vb,
    int base, int first, int count, int kvhd, int window, int cap,
    const int *base_ptr = nullptr)                     // non-NULL: device override (graph path)
{
    int i = blockIdx.y;                                // 0..count-1
    int j = blockIdx.x * blockDim.x + threadIdx.x;     // 0..kvhd-1
    if (i >= count || j >= kvhd) return;
    if (base_ptr) base = *base_ptr;
    int t    = first + i;                              // token index within batch
    size_t slot = ((size_t)base + t) % (size_t)cap;    // ring position
    kcache[slot * kvhd + j] = float_to_fp8(kv_codec_value(kb + (size_t)t * kvhd, j));
    vcache[slot * kvhd + j] = float_to_fp8(kv_codec_value(vb + (size_t)t * kvhd, j));
}

// Scatter the batch's K/V into the linear global cache at positions base..base+rows-1.
// k/vcache point at the layer slot's base [capacity][n_kv_global][head_dim]; per-token
// width is kvhd = n_kv_global*head_dim (12B: kvhd == GEMMA4_GLOBAL_HEAD_DIM). kb/vb are
// token-major [rows][kvhd] so all KV heads of a token are written contiguously — exactly
// the [pos][NKV][HD] layout the decode/flash global attention reads. grid=(ceil(kvhd/256),
// rows). For the 12B (n_kv_global=1) this is bit-identical to the old hd-only writer.
__global__ void kv_write_global_kernel(
    kv_t *kcache, kv_t *vcache, const float *kb, const float *vb,
    int base, int rows, int kvhd,
    const int *base_ptr = nullptr)                     // non-NULL: device override (graph path)
{
    int t = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows || j >= kvhd) return;
    if (base_ptr) base = *base_ptr;
    int pos = base + t;
    kcache[(size_t)pos * kvhd + j] = float_to_fp8(kv_codec_value(kb + (size_t)t * kvhd, j));
    vcache[(size_t)pos * kvhd + j] = float_to_fp8(kv_codec_value(vb + (size_t)t * kvhd, j));
}

// =========================================================================
// ─── GeGLU Activation ──────────────────────────────────────────────────
// =========================================================================

// gelu_pytorch_tanh: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__device__ inline float gelu_tanh(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x3)));
}

// GeGLU: out = gelu(gate) * up
__global__ void geglu_kernel(
    float       *out,
    const float *gate,
    const float *up,
    int          n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = gelu_tanh(gate[i]) * up[i];
}

// SiLU-GLU (Qwen3 FFN), f32: out = silu(gate)*up, silu(x)=x*sigmoid(x).
__global__ void silu_glu_kernel(
    float *out, const float *gate, const float *up, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    out[i] = (g / (1.0f + __expf(-g))) * up[i];
}

// ── Fused gate+up+GeGLU MMVQ (llama.cpp has_fusion analogue; audit #30 lever 2) ──────────
// Each WARP owns one INTERMEDIATE row idx: it dots the activation with BOTH the gate and the
// up weight row, then writes out[idx] = gelu(gate)·up directly. Two wins over the old
// gate-mmvq + up-mmvq + geglu (3 kernels): (1) the gate/up intermediates never round-trip to
// DRAM (~12 MB/token saved); (2) interleaving the gate and up block reads in one warp keeps
// 2× the weight loads in flight → higher effective bandwidth on the FFN (the largest weights),
// which is where llama.cpp's fused kernel pulls ahead. BIT-IDENTICAL to the unfused path:
// ga/ua are accumulated by the exact same dp4a/scale arithmetic as mmvq_q*_kernel and the
// final gelu_tanh(ga)·ua matches geglu_kernel.
__global__ void mmvq_q4_0_glu_kernel(
    float *out, const uint8_t *wgate, const uint8_t *wup,
    const int8_t *qx, const float *dx, const int *sx, int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int nb = in_dim >> 5;
    const uint8_t *grow = wgate + (size_t)idx * (size_t)nb * 18;
    const uint8_t *urow = wup   + (size_t)idx * (size_t)nb * 18;
    float ga = 0.0f, ua = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const int *xqs = (const int *)(qx + (size_t)b * 32);
        const uint8_t *gb = grow + (size_t)b * 18;
        const uint8_t *ub = urow + (size_t)b * 18;
        __half_raw gh; gh.x = (uint16_t)(gb[0] | ((uint16_t)gb[1] << 8));
        __half_raw uh; uh.x = (uint16_t)(ub[0] | ((uint16_t)ub[1] << 8));
        float dwg = __half2float(__half(gh)), dwu = __half2float(__half(uh));
        const void *gqs = gb + 2, *uqs = ub + 2;
        int gsum = 0, usum = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int gw = q8_get_int_b2(gqs, k), uw = q8_get_int_b2(uqs, k);
            int xa = xqs[k], xb = xqs[k + 4];
            gsum = __dp4a(gw & 0x0F0F0F0F, xa, gsum);
            gsum = __dp4a((gw >> 4) & 0x0F0F0F0F, xb, gsum);
            usum = __dp4a(uw & 0x0F0F0F0F, xa, usum);
            usum = __dp4a((uw >> 4) & 0x0F0F0F0F, xb, usum);
        }
        ga += dwg * dx[b] * (float)(gsum - 8 * sx[b]);
        ua += dwu * dx[b] * (float)(usum - 8 * sx[b]);
    }
    ga = warp_reduce_sum_all(ga);
    ua = warp_reduce_sum_all(ua);
    if (lane == 0) out[idx] = gelu_tanh(ga) * ua;
}

__global__ void mmvq_q8_0_glu_kernel(
    float *out, const uint8_t *wgate, const uint8_t *wup,
    const int8_t *qx, const float *dx, int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int nb = in_dim >> 5;
    const uint8_t *grow = wgate + (size_t)idx * (size_t)nb * 34;
    const uint8_t *urow = wup   + (size_t)idx * (size_t)nb * 34;
    float ga = 0.0f, ua = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const int *xqs = (const int *)(qx + (size_t)b * 32);
        const uint8_t *gb = grow + (size_t)b * 34;
        const uint8_t *ub = urow + (size_t)b * 34;
        __half_raw gh; gh.x = (uint16_t)(gb[0] | ((uint16_t)gb[1] << 8));
        __half_raw uh; uh.x = (uint16_t)(ub[0] | ((uint16_t)ub[1] << 8));
        float dwg = __half2float(__half(gh)), dwu = __half2float(__half(uh));
        const void *gqs = gb + 2, *uqs = ub + 2;
        int gsum = 0, usum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            gsum = __dp4a(q8_get_int_b2(gqs, k), xqs[k], gsum);
            usum = __dp4a(q8_get_int_b2(uqs, k), xqs[k], usum);
        }
        ga += dwg * dx[b] * (float)gsum;
        ua += dwu * dx[b] * (float)usum;
    }
    ga = warp_reduce_sum_all(ga);
    ua = warp_reduce_sum_all(ua);
    if (lane == 0) out[idx] = gelu_tanh(ga) * ua;
}

// Dispatch the fused gate+up+GLU MMVQ. qx/dx/sx already hold the FFN-norm activation's
// int8 quant (shared with the unfused path); fmt selects Q4_0 (2) or Q8_0.
static inline void mmvq_glu_launch(
    float *out, const uint8_t *wgate, const uint8_t *wup,
    const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim, int fmt, cudaStream_t stream)
{
    const int NWARPS = 8; int b = NWARPS * 32;
    int g = (out_dim + NWARPS - 1) / NWARPS;
    if (fmt == 2 /*Q4_0*/)
        mmvq_q4_0_glu_kernel<<<g, b, 0, stream>>>(out, wgate, wup, qx, dx, sx, in_dim, out_dim);
    else
        mmvq_q8_0_glu_kernel<<<g, b, 0, stream>>>(out, wgate, wup, qx, dx, in_dim, out_dim);
}

// =========================================================================
// ─── Logit Softcap ──────────────────────────────────────────────────────
// =========================================================================

// softcap(x, cap) = cap * tanh(x / cap)
__global__ void logit_softcap_kernel(
    float *logits,
    float  cap,
    int    n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    logits[i] = cap * tanhf(logits[i] / cap);
}

// Mask suppressed token ids with -inf (mirrors llama.cpp's gemma4 logits bias
// for tokenizer.ggml.suppress_tokens — known checkpoint issue where the model
// otherwise emits <image|>/<audio|> tokens).
__global__ void suppress_tokens_kernel(
    float         *logits,
    const int32_t *ids,
    int            n_ids,
    int            vocab)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ids) return;
    int id = ids[i];
    if (id >= 0 && id < vocab) logits[id] = -INFINITY;
}

// =========================================================================
// ─── Embedding Lookup ───────────────────────────────────────────────────
// =========================================================================


// Q8_0 embedding lookup: the embedding table is stored as Q8_0 blocks, one row
// per token of `hidden_size` elements = hidden_size/32 blocks of 34 bytes.
__global__ void embed_lookup_q8_0_kernel(
    float             *out,     // [batch × hidden_size]
    const unsigned char *table, // [vocab_size × hidden_size] in Q8_0
    const int32_t     *tokens,  // [batch]
    int                batch,
    int                hidden_size)
{
    int row = blockIdx.x;
    if (row >= batch) return;

    int token = tokens[row];
    if (token < 0) token = 0;
    if (token >= GEMMA4_VOCAB_SIZE) token = 0;

    const int block_size = 2 + 32;          // sizeof(block_q8_0) == 34
    int n_blocks = hidden_size / 32;
    const unsigned char *emb = table + (size_t)token * n_blocks * block_size;
    float *out_row = out + row * hidden_size;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        int b = i / 32;
        int j = i % 32;
        const unsigned char *blk = emb + (size_t)b * block_size;
        __half_raw d_raw;
        d_raw.x = (uint16_t)(blk[0] | (blk[1] << 8));
        float s = __half2float(__half(d_raw));
        int8_t q = (int8_t)blk[2 + j];
        out_row[i] = s * (float)q;
    }
}

// =========================================================================
// ─── Residual Add ───────────────────────────────────────────────────────
// =========================================================================


// Fused rms_norm_rows + residual_add: norm the sub-block output and fold
// it into the residual stream in ONE pass. Saves 2 kernel launches per layer.
__global__ void rms_norm_residual_add_kernel(
    float       *residual, // [rows*n] in/out
    const float *x,        // [rows*n] sub-block output to norm
    const float *weight,   // [n] RMSNorm weights
    int          n, int rows, float eps)
{
    extern __shared__ float smem[];
    int row = blockIdx.x, tid = threadIdx.x;
    float sum_sq = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        float v = x[(size_t)row * n + j];
        sum_sq += v * v;
    }
    smem[tid] = sum_sq; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float rms = __frsqrt_rn(smem[0] / (float)n + eps);
    __syncthreads();
    for (int j = tid; j < n; j += blockDim.x) {
        float v = x[(size_t)row * n + j];
        float w = weight ? weight[j] : 1.0f;
        residual[(size_t)row * n + j] += v * w * rms;
    }
}

__global__ void residual_add_kernel(
    float       *x,        // [n] in/out
    const float *residual, // [n]
    int          n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += residual[i];
}


// Simple scale: x[i] *= s  (replaces problematic cublasSscal)
__global__ void scale_kernel(float *x, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// =========================================================================
// ─── Engine Implementation ──────────────────────────────────────────────
// =========================================================================

// Projection indices (order matches proj_desc); declared before the engine struct
// because NVFP4 weight arrays are dimensioned [layer][PJ_COUNT].
enum { PJ_Q = 0, PJ_K, PJ_V, PJ_O, PJ_GATE, PJ_UP, PJ_DOWN, PJ_COUNT };

// ── Per-sequence state (continuous batching, Phase 1) ────────────────────────
// State that is logically OWNED BY ONE SEQUENCE, as opposed to the shared
// per-step compute scratch (d_x, d_attn_*, d_sb[...]) which is reused by
// whatever sequence is being stepped. Today the engine holds exactly one of
// these (`cur`), so behaviour is identical to the previous single `int
// global_n_tokens`; Phase 2+ grows this to an array (one slot per in-flight
// sequence) plus per-class KV block tables, and Phase 5 moves the MTP recurrent
// draft state (d_mtp_h / mtp_h_valid) in here so each sequence drafts from its
// own hidden. Keeping it in a named struct now lets the rest of the engine name
// the owner of a position without another wide rename later.
typedef struct gemma4_seq {
    int n_tokens;   // absolute position == count of tokens committed to this seq's KV

    // ── Paged KV (Phase 2, dormant until the write/read paths are wired) ──
    // Per-sequence block tables mapping logical token positions to physical pool
    // blocks. One table per cache CLASS (the logical→physical mapping differs:
    // the sliding table recycles leading blocks as the window advances; the
    // global table never recycles). The mapping is shared across all layers of a
    // class (a block reserves those 256 positions in EVERY layer of the class).
    // d_slid_blocks / d_glob_blocks mirror the host tables' block-id arrays on
    // device, refreshed when a table grows/recycles, for the kernels to index.
    PagedBlockTable slid_bt;     // sliding-class block table (host bookkeeping)
    PagedBlockTable glob_bt;     // global-class  block table (host bookkeeping)
    int            *d_slid_blocks;   // device copy of slid_bt.blocks (or NULL)
    int            *d_glob_blocks;   // device copy of glob_bt.blocks (or NULL)
    int             d_slid_cap;      // capacity of d_slid_blocks (elems)
    int             d_glob_cap;      // capacity of d_glob_blocks (elems)
    int             used;            // 1 if this slot is allocated to a live sequence

    // ── Per-sequence sampling params (continuous-batching batch path) ──────
    // Stored once at seq_add and applied on-device to THIS row's logits each
    // step. temp<=0 → exact greedy (argmax), keeping the batch self-test
    // byte-identical. seed + n_sampled give a reproducible per-row RNG stream:
    // the draw for the i-th sampled token is hash(seed, i), so two runs with the
    // same seed produce the same sequence regardless of batch composition.
    float           samp_temp;       // temperature (<=0 ⇒ greedy)
    int             samp_top_k;      // top-k (0 ⇒ disabled)
    float           samp_top_p;      // nucleus top-p (0 or >=1 ⇒ disabled)
    float           samp_min_p;      // min-p (0 ⇒ disabled)
    uint64_t        samp_seed;       // per-sequence RNG seed
    uint64_t        n_sampled;       // count of tokens sampled from this seq (RNG index)

    // ── Per-sequence MTP recurrent draft state (batch speculative decode) ──
    // The drafter is a recurrence (tok, h) → logits + next-h; each batch sequence
    // carries its OWN h so it drafts from its own committed hidden. d_mtp_h is the
    // seq's post-output-norm hidden paired with its pending token; mtp_h_valid clears
    // on prefill/rewind. Lazily allocated on first draft so non-spec slots cost nothing.
    float          *d_mtp_h;         // [cfg.hidden_size] this seq's recurrent h (or NULL)
    int             mtp_h_valid;     // 1 ⇒ d_mtp_h is paired with this seq's pending token
} gemma4_seq;

// Max concurrent sequences in one multi-seq batched decode. Bounded by the
// batched-decode scratch (d_sb / d_qx_b / d_fa_acc), which is sized for
// GEMMA4_SPEC_MAX rows — so B independent rows reuse that scratch directly.
#define GEMMA4_MAX_SEQS GEMMA4_SPEC_MAX
#include "qwen35_state.cuh"
#include "qwen35_dflash_plan.cuh"   // S1a DFlash shape/lookahead planner + enable/concurrency gate

// Runtime slot target for memory-heavy batched engines. The compile-time ceiling still sizes
// graph/scratch arrays, while --parallel (promoted to FUCINA_PAGED_MAXSEQS by the CLI) controls
// how many per-sequence state arenas are actually allocated. Direct C users retain the old cap.
static int requested_seq_capacity() {
    int cap = GEMMA4_MAX_SEQS;
    if (const char *e = getenv("FUCINA_PAGED_MAXSEQS")) {
        int v = atoi(e);
        if (v > 0 && v < cap) cap = v;
    }
    return cap;
}

// Prefill TILE width: how many prompt tokens one batched weight-pass covers (qwen35 hybrid).
// Decoupled from GEMMA4_MAX_SEQS (the decode/spec concurrency cap). The tensor-core prefill GEMM
// dequantizes each weight to BF16 ONCE PER TILE, so a wide tile amortizes that fixed dequant over
// the whole prompt (a 4096-token prompt = ONE tile = one dequant pass). The per-slot KV/state
// arenas stay MAX_SEQS-sized; only the per-row compute + activation scratch widens to this. 8192
// keeps a full ctx-8192 prompt as a SINGLE base==0 tile: one weight touch + all-tensor-core attn.
#define QWEN35_PF_TILE 8192

struct gemma4_engine {
    // GGUF data (mmap'd)
    const uint8_t *gguf_data;
    uint64_t       gguf_size;
    int            gguf_fd;

    // Tensor data copied to device memory at load (llama.cpp-style backend
    // buffer). When non-NULL, weight accessors return pointers into this
    // buffer instead of the mmap'd file, avoiding GPU page faults / host
    // memory reads on every GEMV.
    uint8_t  *d_weights;
    uint64_t  tdata_start;   // file offset where tensor data begins
    unsigned char *d_token_embd;  // QAT Q4_0 model: token_embd (Q6_K) converted to Q8_0 (for embed lookup)
    // Step 8: native-Q6_K tied LM head. d_lmhead_q6k points at the RAW Q6_K bytes in the device
    // weight blob (d_weights + output_weight offset); the output projection reads it directly
    // instead of the Q8_0 d_token_embd, cutting ~0.24 GB/token. lmhead_q6k=1 enables it.
    unsigned char *d_lmhead_q6k;
    int            lmhead_q6k;
    // Native Q4_K tied LM head (Unsloth UD models): d_lmhead_q4k points at the RAW Q4_K bytes in
    // d_weights (the tied token_embd), read 4.5-bit by the Q4_K dp4a matvec instead of the Q8_0
    // upconvert (d_token_embd) — saves ~0.6 GB/token. lmhead_q4k=1 enables it.
    unsigned char *d_lmhead_q4k;
    int            lmhead_q4k;

    // Unsloth UD dynamic-quant (31B): a handful of bulk projections ship in an off-format
    // GGML type (blk.0..6.ffn_down = Q4_1). At load each is REQUANTIZED to Q4_0 into its own
    // device buffer; weight_fp8() returns that buffer (instead of d_weights+offset) when the
    // requested tensor_offset matches an override entry, so every GEMV/MMQ/BF16-dequant site
    // reads valid Q4_0 with the engine format unchanged. 12B has zero overrides → no effect.
    int            n_wt_override;                       // 0 for 12B / non-dynamic GGUF
    uint64_t       wt_override_off[96];                 // ABS file offset of the overridden tensor
    unsigned char *wt_override_ptr[96];                 // device buffer with requantized bytes (Q4_0/Q8_0)

    // ROTATING per-layer BF16 dequant scratch for batched cuBLAS prefill (Step 2).
    // The 7 projection weights of the CURRENT layer only are dequantized Q8_0/FP8 →
    // BF16 here (row-major [out_dim][in_dim], same element order as the source) just
    // before that layer's GEMMs, then reused by the next layer. This replaces the
    // old persistent 21.8 GB d_bf16[48][7] buffer with a ~0.5 GB scratch — keeping
    // ~12 GB resident (not ~34 GB) so the GGUF stays page-cached across runs and
    // cold reloads do not re-pay the slow disk read. proj index: 0=q 1=k 2=v 3=o
    // 4=gate 5=up 6=down (global layers have no separate V → d_bf16_layer[.][2] NULL).
    //
    // DOUBLE-BUFFERED (ping/pong on layer&1): the dequant of layer L+1 runs on the
    // dedicated dq_stream into buffer (L+1)&1 while layer L's projection GEMMs read
    // buffer L&1 on the main stream — overlapping the bandwidth-bound Q4_0→BF16
    // dequant (~28 GB/full pass) behind the compute-bound tensor-core GEMMs instead
    // of serializing it before every layer. Events sequence the two streams: the
    // main stream waits ev_dq_done[L&1] before reading layer-L weights; dq_stream
    // waits ev_gemm_done[(L-2)&1] before reusing that buffer (it last fed layer L-2).
    __nv_bfloat16 *d_bf16_layer[2][7];  // [pingpong][proj] [out_dim×in_dim], max-sized
    int            bf16_ready;          // 1 once the scratch is allocated

    // FUCINA_PACKED decode path (opt-in): a parallel copy of the Q4_0 projection blob
    // repacked per projection into [16-B-aligned quants ‖ fp16 scales] so the decode
    // GEMV reads each block's 16 nibble bytes with ONE coalesced 128-bit (uint4) load
    // instead of the native 18-B block's 2-byte-granular loads. Same dp4a math / −8 fold
    // → bit-identical output. Same per-projection byte offset as d_weights (off-tdata),
    // so a projection's packed base = d_weights_packed + (weight_ptr − d_weights).
    uint8_t       *d_weights_packed;    // repacked Q4_0 blob (same size as d_weights)
    int            packed_ready;        // 1 once d_weights_packed is built
    int            q4k_packed;          // 1 once Q4_K bulk projections are repacked IN PLACE in d_weights
    cudaStream_t   dq_stream;           // weight-dequant stream (overlaps GEMMs)
    cudaEvent_t    ev_dq_done[2];       // dequant of buffer b complete
    cudaEvent_t    ev_gemm_done[2];     // last GEMM reading buffer b complete

    // ── Persistent prefill scratch (CUDA-graph safe) ─────────────────
    int    pf_scratch_ready;
    float  *d_pf_x, *d_pf_norm, *d_pf_q, *d_pf_k, *d_pf_v;
    float  *d_pf_attn, *d_pf_gate, *d_pf_up, *d_pf_scores;
    __nv_bfloat16 *d_pf_inb, *d_pf_qb, *d_pf_kb, *d_pf_vb;
    __nv_bfloat16 *d_pf_kbx, *d_pf_vbx, *d_pf_pb;
    // Paged single-pass prefill scratch: an N-row write_pos iota [0..NC-1] and the
    // per-row replicated PagedSeqView arrays (one sliding, one global) that let
    // paged_kv_write scatter ALL N prompt tokens of ONE sequence in one launch
    // (it resolves batch.seqs[row] per row, so the same view is replicated N times).
    int          *d_pf_wpos;          // [NC] = {0,1,...,NC-1}
    PagedSeqView *d_pf_views_slid;    // [NC] replicated sliding view
    PagedSeqView *d_pf_views_glob;    // [NC] replicated global  view

    // ── Tiled-GEMM flash-prefill attention scratch (~2.1 GB, lazy) ──
    // Sized for the widest layer (global oq = HEADS×512) at chunk
    // GEMMA4_FP_CHUNK and key tile GEMMA4_FP_TILE_K. S/P are per-head
    // batches [HEADS][CHUNK×TILE]; m/l the online-softmax running state.
    int fp_scratch_ready;
    __nv_bfloat16 *d_fp_qb, *d_fp_kbx, *d_fp_vbx;   // chunk Q / broadcast K,V
    __nv_bfloat16 *d_fp_kt, *d_fp_vt;               // history K/V tile (bf16)
    __nv_bfloat16 *d_fp_pb;                         // P tile batch (bf16)
    __nv_bfloat16 *d_fp_st;                         // S tile batch (bf16: S write/read
                                                    // traffic bounds the QKᵀ GEMM at k=512)
    float *d_fp_m, *d_fp_l;                         // online softmax state

    // ── Pinned double-buffer staging for KV snapshot save/restore ──
    // The snapshot pool lives in Go (pageable host memory); a direct pageable
    // cudaMemcpy2D runs at ~120 MB/s on GB10 (multi-second stalls per save).
    // Staging through pinned memory (DMA ↔ pinned overlapped with a host
    // memcpy pinned ↔ pageable) is the same pattern as the streamed weight
    // load. kv_stage_ready: 0 = unallocated, 1 = ready, -1 = alloc failed
    // (snapshot copies then use the old synchronous pageable fallback).
    void       *h_kv_stage[2];
    cudaEvent_t ev_kv_stage[2];
    int         kv_stage_ready;
    // CUDA Graph state
    int graph_mode;
    struct { cudaGraph_t g; cudaGraphExec_t e; int N, hits, misses; } graph;

    // Tensor offsets within mapped file (from start of tensor_data section)
    // All weights are stored in FP8 (or Q8_0)
    struct {
        uint64_t token_embd;        // [vocab_size × hidden_size]

        // Per-layer tensors [48]
        struct {
            uint64_t attn_q;         // [hidden_size × heads × head_dim]
            uint64_t attn_k;         // [hidden_size × kv_heads × head_dim]
            uint64_t attn_v;         // [hidden_size × kv_heads × head_dim]
            uint64_t attn_output;       // [hidden_size × hidden_size]
            uint64_t attn_norm;         // [hidden_size] pre-attn RMSNorm
            uint64_t attn_q_norm;       // [head_dim]   per-head Q RMSNorm
            uint64_t attn_k_norm;       // [head_dim]   per-head K RMSNorm
            uint64_t post_attn_norm;    // [hidden_size] post-attn sandwich norm

            uint64_t ffn_gate;          // [hidden_size × intermediate]
            uint64_t ffn_up;            // [hidden_size × intermediate]
            uint64_t ffn_down;          // [intermediate × hidden_size]
            uint64_t ffn_norm;          // [hidden_size] pre-FFN RMSNorm
            uint64_t post_ffn_norm;     // [hidden_size] post-FFN sandwich norm
            uint64_t layer_out_scale;   // [1]  scalar output multiplier
            // Per-tensor weight format (FORMAT_*). Qwen3 Q4_K_M mixes Q4_K/Q6_K per layer
            // (attn_v + ffn_down vary), so the bulk dispatch must read the tensor's own format
            // rather than the engine-wide eng->format. 0 = unset (e.g. global-layer attn_v).
            uint8_t fmt_q, fmt_k, fmt_v, fmt_o, fmt_gate, fmt_up, fmt_down;
            // Canonical descriptors are populated alongside the compatibility offsets/format
            // bytes. Qwen runtime paths migrate projection-family by projection-family.
            WeightRef ref_q, ref_k, ref_v, ref_o;
            WeightRef ref_gate, ref_up, ref_down;

            // ── Qwen3.5 hybrid gated-deltanet (LINEAR) layer tensors (GEMMA4_ARCH_QWEN3_5 only) ──
            // Zero on every other arch AND on the hybrid's FULL layers (those reuse attn_q/k/v/
            // output + attn_q_norm/attn_k_norm above; attn_q is the 2×-wide [query|gate] proj). The
            // shared pre-mixer / pre-FFN norms + dense SwiGLU FFN reuse attn_norm/ffn_norm/ffn_*.
            // GGUF tensor names (per the qwen35 export): attn_qkv, attn_gate, ssm_{alpha,beta,a,
            // dt.bias,conv1d,norm,out}. Dims confirmed: in_qkv [hidden→2·grp·st+inner],
            // in_z [hidden→inner], in_a/in_b [hidden→time_step_rank], a_log/dt_bias [time_step_rank],
            // conv1d [conv_kernel×conv_dim] F32, norm [state_size] F32, out [inner→hidden].
            struct {
                uint64_t in_qkv;   // blk.l.attn_qkv.weight   q+k+v fused in-proj (Q5_K here)
                uint64_t in_z;     // blk.l.attn_gate.weight  output-gate (z) in-proj
                uint64_t in_a;     // blk.l.ssm_alpha.weight  decay (a) in-proj  [→ time_step_rank]
                uint64_t in_b;     // blk.l.ssm_beta.weight   beta  (b) in-proj  [→ time_step_rank]
                uint64_t a_log;    // blk.l.ssm_a             A_log  [time_step_rank]  F32
                uint64_t dt_bias;  // blk.l.ssm_dt.bias       dt bias[time_step_rank]  F32
                uint64_t conv1d;   // blk.l.ssm_conv1d.weight depthwise causal conv  [kernel×conv_dim] F32
                uint64_t norm;     // blk.l.ssm_norm.weight   gated RMSNorm gain [state_size]  F32
                uint64_t out;      // blk.l.ssm_out.weight    out-proj  [inner→hidden]
                uint8_t  fmt_in_qkv, fmt_in_z, fmt_in_a, fmt_in_b, fmt_out;
                WeightRef ref_in_qkv, ref_in_z, ref_in_a, ref_in_b, ref_out;
            } ssm;
        } layers[GEMMA4_CAP_LAYERS];

        uint64_t output_norm;        // [hidden_size] (FP32)
        uint64_t output_weight;      // [hidden_size × vocab_size]
        uint8_t  output_fmt;         // FORMAT_* of the (untied) LM head; 0 = tied/unset
    } tensors;

    // 1 if output_weight aliases token_embd (tied embeddings), 0 if a separate
    // output.weight tensor was found in the GGUF.
    int             output_tied;

    // Model parameters
    tensor_format_t format;
    uint32_t        context_size;
    layer_type_t    layer_types[GEMMA4_CAP_LAYERS]; // 0=sliding, 1=global
    int             n_layers_sliding;
    int             n_layers_global;

    // Layer index helpers: which layers are global
    int             global_layer_indices[GEMMA4_CAP_LAYERS];
    int             n_global;
    // Inverse map: absolute layer id -> contiguous global cache slot
    // (0..n_layers_global-1), or -1 for sliding layers. The global KV cache is
    // allocated for n_layers_global slots only (not all GEMMA4_MAX_LAYERS), so
    // every read/write into d_global_k/v must index by global_slot[layer].
    int             global_slot[GEMMA4_CAP_LAYERS];

    // Device memory
    float  *d_scratch;         // scratch buffer [1M elements]
    float  *d_logits;          // [vocab_size]
    float  *d_x;               // current hidden state [hidden_size]
    float  *d_attn_q;          // [16 × head_dim] or [16 × global_head_dim]
    float  *d_attn_k;          // [n_kv_heads × head_dim]
    float  *d_attn_v;          // [n_kv_heads × head_dim] (sliding) or K copy (global)
    float  *d_attn_out;        // [hidden_size]
    float  *d_ffn_out;         // [intermediate]
    float  *d_ffn_gate;        // [intermediate]
    float  *d_ffn_up;          // [intermediate]
    float  *d_norm;            // [hidden_size] scratch for normed output
    float  *d_norm_w;          // [hidden_size] scratch for norm weights (host→dev)
    float  *d_residual;        // [hidden_size]
    float  *d_rope_freqs;      // [GEMMA4_GLOBAL_HEAD_DIM/2] freq_factors for global RoPE
    float  *d_head_norm_w;     // [GEMMA4_GLOBAL_HEAD_DIM] scratch for per-head norm weights

    // GQA-broadcast global flash-decode split-K scratch (DECODE-30-35 Step 1). Per-(split,
    // head) online-softmax partials, merged by flash_decode_combine_kernel. Sized for the
    // worst case (MAX_SPLITS × HEADS × GLOBAL_HEAD_DIM ≈ 4 MB), allocated once at create.
    float  *d_fa_acc;          // [MAX_SPLITS][HEADS][GLOBAL_HEAD_DIM] unnormalized acc
    float  *d_fa_m;            // [MAX_SPLITS][HEADS] running max
    float  *d_fa_l;            // [MAX_SPLITS][HEADS] running denominator

    // Engine-resident batched-decode (speculative) scratch, sized for GEMMA4_SPEC_MAX
    // rows and allocated once (lazily). Reused every decode_batched call so probe /
    // re-probe steps pay no per-call cudaMalloc/free — d_sb[12] holds, in order:
    // tok, x, norm, inf, q, k, v, attn, o, gate, up, logitsK.
    float  *d_sb[12];
    int     sb_ready;
    // NVFP4 batched-verify transposed-activation scratch: Xt[in_dim][K] (input-major) for the
    // weight-read-once batched GEMV (nvfp4_gemv.cuh). Sized SPEC_MAX × widest in_dim (= INTERMEDIATE,
    // the DOWN projection). Pre-allocated with d_sb so the captured verify does no alloc/host-sync.
    float  *d_specxt;
    int    *d_sample_id;       // 4-byte device scratch for GPU-side sampled token id
    float  *d_sample_p;        // 4-byte device scratch: drafter top-1 softmax prob
    // GPU-side spec verify (a): K sampled ids + K host draws stay on device, only the
    // K ids cross to host (vs the old K×262144 logit D2H + K CPU vocab scans).
    int    *d_spec_ids;        // [SPEC_MAX] per-row sampled token id
    float  *d_spec_rnd;        // [SPEC_MAX] per-row uniform draw (H2D once/step)
    // GPU repeat-penalty state (lazy, ~2 MB; only touched when a request sets
    // repeat_penalty != 1.0 — keeps such requests on the spec fast path instead
    // of the per-token 1 MB-logits-D2H CPU loop). d_pen_cnt[v] = occurrences of
    // v in the synced history; d_pen_hist mirrors the host hist incrementally;
    // d_pen_batch = the current verify step's [g, draft...] for per-row extras.
    int     *d_pen_cnt;        // [VOCAB] occurrence counts
    int32_t *d_pen_hist;       // [global_kv_capacity + 8] history mirror
    int32_t *d_pen_batch;      // [SPEC_MAX]
    // Device-chained MTP draft (b): maxd ids + confidences read back in ONE sync.
    int    *d_mtp_ids;         // [SPEC_MAX] chained draft ids
    float  *d_mtp_conf;        // [SPEC_MAX] per-draft top-1 confidence (PMIN gate)
    // CUDA-graph MTP forward (c): the ~57-launch assistant pass captured once and
    // replayed per drafted token. All per-call state is device-resident: the input
    // token (d_mtp_tok, fed by mtp_argmax_conf_kernel for chained tokens) and the
    // RoPE/attention position (d_mtp_pos, one 4-byte H2D per draft call).
    int32_t *d_mtp_tok;        // [1] current draft input token
    int     *d_mtp_pos;        // [1] n_past for this draft chain
    cudaGraphExec_t mtp_graph; // instantiated forward graph (NULL until captured)
    int     mtp_graph_failed;  // capture failed once → use the launch-per-kernel path
    // CUDA-graph PAGED MTP forward (batch spec path): like mtp_graph but for
    // mtp_forward_paged. ONE graph serves all slots — the per-slot recurrent h is
    // seeded into the FIXED scratch d_mtp_h_draft (D2D) before each slot's chain, and
    // the per-slot KV views go through the fixed d_ms_views_* the graph reads. Replays
    // per drafted token; mtp_argmax_conf_kernel (launched outside the graph) chains the
    // next token device-side, so a whole draft chain costs ZERO mid-chain syncs.
    float          *d_mtp_h_draft;   // [hidden_size] fixed recurrent-h scratch for the graph
    cudaGraphExec_t mtp_paged_graph; // instantiated paged forward graph (NULL until captured)
    int     mtp_paged_graph_failed;  // capture failed once → per-kernel paged path

    // ── Batched (B-row) MTP drafter scratch (continuous-batch spec path) ──
    // ONE B-row MTP forward drafts ALL eligible slots per token (replacing the per-slot
    // serial loop + a cudaStreamSynchronize PER slot), so a whole draft round costs ONE
    // sync. Sized to GEMMA4_MAX_SEQS rows; lazily allocated on first batched draft so the
    // non-batch paths cost nothing. Layout per buffer is [MAX_SEQS][dim] row-major, matching
    // the verify scratch so gemv_batched_w / *_rows_* kernels apply unchanged. Logits reuse
    // d_sb[11] (drafting precedes the verify forward that owns it, same stream).
    float   *d_mtpb_xh;   // [MAX_SEQS][2*hidden]  pre-projection input (embed‖h per row)
    float   *d_mtpb_cur;  // [MAX_SEQS][MTP_HIDDEN] recurrent layer state
    float   *d_mtpb_t1;   // [MAX_SEQS][MTP_HIDDEN] scratch
    float   *d_mtpb_t2;   // [MAX_SEQS][MTP_HIDDEN] scratch (residual)
    float   *d_mtpb_q;    // [MAX_SEQS][n_heads*GLOBAL_HEAD_DIM] Q (Q-only attention)
    float   *d_mtpb_attn; // [MAX_SEQS][n_heads*GLOBAL_HEAD_DIM] attention out
    float   *d_mtpb_ffa;  // [MAX_SEQS][MTP_FFN] gate
    float   *d_mtpb_ffb;  // [MAX_SEQS][MTP_FFN] up
    float   *d_mtpb_h;    // [MAX_SEQS][hidden] per-row recurrent h (seeded per round, updated in place)
    int     *d_mtpb_ids;  // [MAX_SEQS][SPEC_MAX] per-row drafted ids
    float   *d_mtpb_conf; // [MAX_SEQS][SPEC_MAX] per-row top-1 confidences
    int32_t *d_mtpb_tok;  // [MAX_SEQS] per-row current input token (chained device-side)
    int     *d_mtpb_pos;  // [MAX_SEQS] per-row chain position (constant: frozen committed prefix)
    int      mtpb_ready;  // 1 ⇒ batched scratch allocated

    // ── Gemma-4 MTP assistant drafter (llama.cpp PR #23398 equivalent) ──
    // The official ~423M assistant head: 4 Q-only layers (no K/V projections — they
    // attend the TARGET's KV cache: sliding layers → target layer 46, global → 47),
    // pre/post projections bridging the 3840 backbone hidden and its 1024 hidden,
    // its own Q8_0 unembed. Drafts maxd tokens per step recursively (h_next chain),
    // all at RoPE position n_past; the existing batched verify checks them.
    struct {
        int      loaded;
        uint8_t *d_w;              // whole assistant GGUF resident on device
        uint64_t pre_proj, post_proj, tok_embd, out_norm, rope_freqs;   // ABS offsets
        uint64_t attn_norm[4], wq[4], wo[4], q_norm[4], post_attn_norm[4];
        uint64_t ffn_norm[4], gate[4], up[4], down[4], post_ffw_norm[4];
        float    out_scale[4];
        int      is_global[4];
        int      wfmt;             // FORMAT_Q8_0 (12B head) or FORMAT_Q4_0 (31B head)
    } mtp;
    int     mtp_h_valid;       // d_mtp_h holds the h paired with the pending token g
    float  *d_mtp_h;           // [cfg.hidden_size] target post-output-norm hidden (recurrent h)
    float  *d_mtp_xh;          // [2×cfg.hidden_size] concat(embed(tok)·√hidden, h)
    float  *d_mtp_cur;         // [1024] residual stream
    float  *d_mtp_t1, *d_mtp_t2;  // [1024] norm/proj scratch
    float  *d_mtp_q;           // [cfg.n_heads×512] worst-case Q
    float  *d_mtp_attn;        // [cfg.n_heads×512] attention output
    float  *d_mtp_ffa, *d_mtp_ffb;  // [8192] FFN gate/up
    int8_t *d_qx;              // [GEMMA4_INTERMEDIATE] int8 activation for dp4a MMVQ
    float  *d_dx;              // [GEMMA4_INTERMEDIATE/32] per-block activation scales
    int    *d_sx;              // [GEMMA4_INTERMEDIATE/32] per-block Σ int8 act (Q4_0 −8 fold)
    int8_t *d_qx_b;            // [SPEC_MAX × INTERMEDIATE] int8 act for batched dp4a MMVQ
    float  *d_dx_b;            // [SPEC_MAX × INTERMEDIATE/32] per-block act scales (batched)
    int    *d_sx_b;            // [SPEC_MAX × INTERMEDIATE/32] per-block Σ int8 act (batched)
    // Tiled-MMQ prefill activation scratch: quantized [N × in_dim] for N ≤ MMQ_MAX_N over
    // the widest projection in_dim (INTERMEDIATE). Lazily allocated (Q4_0 models only).
    int8_t *d_pf_qx;           // [MMQ_MAX_N × INTERMEDIATE] int8 activation
    float  *d_pf_dx;           // [MMQ_MAX_N × INTERMEDIATE/32] per-block scales
    int    *d_pf_sx;           // [MMQ_MAX_N × INTERMEDIATE/32] per-block Σ int8 (−8 fold)
    int     mmq_ready;         // 1 once the MMQ prefill scratch is allocated

    // ── NVFP4 prefill (FUCINA_FP4): block-scaled FP4 tensor-core projections ──
    // Persistent per-(layer,proj) NVFP4 weights (~4.25 bpw, ~6.3 GB all layers) built once
    // on first prefill: packed E2M1 values + per-16-elem E4M3 block scales (cuBLASLt 32×4×4
    // swizzled layout) + a per-tensor fp32 global scale. cuBLASLt computes the GEMM ~2.4×
    // faster than the BF16 tensor-core path. Activation quantized per-GEMM into d_fp4_act*.
    cublasLtHandle_t cublaslt;
    uint8_t *d_fp4_w[GEMMA4_CAP_LAYERS][PJ_COUNT];   // packed [out_dim × in_dim/2]
    uint8_t *d_fp4_wsc[GEMMA4_CAP_LAYERS][PJ_COUNT]; // swizzled E4M3 [pad(out,128)×pad(in/16,4)]
    // NVFP4 SINGLE-STORE decode (FORMAT_NVFP4): the decode GEMV (nvfp4_gemv.cuh) reads LINEAR
    // E4M3 block scales [out,in/16], but d_fp4_wsc holds the SWIZZLED scales (for cuBLASLt) which
    // the GEMV cannot consume. So retain a per-projection LINEAR-scale copy uploaded straight off
    // disk (~0.75 GB extra; still a net win vs a duplicate Q4_0 store). NULL on non-NVFP4 models.
    uint8_t *d_fp4_wsc_lin[GEMMA4_CAP_LAYERS][PJ_COUNT]; // linear E4M3 [out_dim × in_dim/16]
    float   *d_fp4_gsw;        // device [MAX_LAYERS*PJ_COUNT] weight per-tensor global scales
    int      fp4_ready;        // 1 once persistent NVFP4 weights are built
    int      fp4_budget_ok;    // --gpu-mem-util verdict: 1 if the NVFP4-prefill copy fits the budget
    int      nvfp4_decode_ready; // 1 once NVFP4 store + linear scales + embed + norms are resident
    // BF16 non-quant tensors loaded from the safetensors checkpoint (FORMAT_NVFP4 only): the
    // embedding table doubles as the (tied) LM head. d_lmhead_bf16 ALIASES d_embed_bf16 when tied
    // — destroy frees it only if the pointers differ (double-free guard). NULL on GGUF models.
    __nv_bfloat16 *d_embed_bf16;   // [vocab × hidden] BF16 embeddings
    float *d_embed_f32;            // [vocab × hidden] F32 embeddings (FP8/qwen35: exact per-step input)
    __nv_bfloat16 *d_lmhead_bf16;  // [vocab × hidden] BF16 LM head (== d_embed_bf16 when tied)
    // FP8 E4M3 per-row-quantized UNTIED LM head (FORMAT_NVFP4 only). The untied BF16 head is 2 GB,
    // read every token; quantizing it to 1 B/elem per-row halves the head bandwidth. Set only when
    // the head is untied AND passes the load-time argmax accuracy gate; d_lmhead_bf16 is then freed
    // and set NULL (no tied-alias: untied is always a distinct allocation). NULL ⇒ use BF16 head.
    uint8_t *d_lmhead_fp8;         // [vocab × hidden] E4M3 weights, per-row scaled
    float   *d_lmhead_fp8_scale;   // [vocab] per-row dequant scale (amax/448)
    // EXACT two-pass greedy head (FP8_BLOCK/qwen35): Q8_0 approx scan + BF16 rescore of the
    // candidates. Unlike the lossy FP8 head above this is bit-identical (see q8_head_* kernels).
    unsigned char *d_lmhead_q8;    // [vocab/32 × 34 B] Q8_0 head copy (0.53 GB @248k vocab)
    int *d_head_cand;              // [MAX_SEQS × Q8HEAD_MAXCAND] candidate indices
    int *d_head_cnt;               // [MAX_SEQS] candidate counts
    uint8_t *d_fp4_act;        // activation packed E2M1 scratch (lazily sized to N×in/2)
    uint8_t *d_fp4_actsc;      // activation swizzled E4M3 block scales scratch
    uint8_t *d_fp4_actlin;     // activation LINEAR E4M3 block scales (pre-swizzle) scratch
    float   *d_fp4_gsact;      // device scalar: activation global scale (per GEMM)
    float   *d_fp4_alpha;      // device scalar: alpha = gs_w · gs_act (device pointer-mode)
    float   *d_fp4_amax;       // device scalar: activation amax reduction target
    size_t   fp4_act_cap;      // current activation-scratch capacity in tokens (N)
    void    *d_fp4_ws;         // cuBLASLt workspace
    cublasLtMatmulDesc_t fp4_desc; // cached NVFP4 matmul descriptor

    // Suppressed token ids (tokenizer.ggml.suppress_tokens) masked to -inf
    int32_t *d_suppress;
    int      n_suppress;

    // Norm weights preloaded to device at create time (was: per-layer
    // cudaMemcpyAsync from the mmap'd file on EVERY token — ~400 tiny H2D
    // copies per decoded token). Layout: [GEMMA4_MAX_LAYERS][stride].
    float *d_w_attn_norm;       // stride GEMMA4_HIDDEN_SIZE
    float *d_w_post_attn_norm;  // stride GEMMA4_HIDDEN_SIZE
    float *d_w_ffn_norm;        // stride GEMMA4_HIDDEN_SIZE
    float *d_w_post_ffn_norm;   // stride GEMMA4_HIDDEN_SIZE
    float *d_w_q_norm;          // stride GEMMA4_GLOBAL_HEAD_DIM
    float *d_w_k_norm;          // stride GEMMA4_GLOBAL_HEAD_DIM
    float *d_w_out_norm;        // [GEMMA4_HIDDEN_SIZE]
    float  h_out_scale[GEMMA4_CAP_LAYERS]; // layer_output_scale scalars (host)

    // KV cache (device) — FP8 E4M3 (1 byte/elem), see kv_t.
    // Sliding: [MAX_LAYERS][sliding_kv_capacity][8×256] per-position RING; absolute position p
    // lives at slot p % sliding_kv_capacity. Sliding-window attention reads only the last
    // GEMMA4_SLIDING_WINDOW positions, so the ring is capped well below context_size (see the
    // allocation): exact rewind/prefix-reuse within the last (cap-window) tokens, full
    // re-prefill fallback beyond it. Token count per sliding layer == global_n_tokens.
    kv_t   *d_sliding_k;       // [MAX_LAYERS × sliding_kv_capacity × 8 × 256]
    kv_t   *d_sliding_v;
    int     sliding_kv_capacity;

    // Global: 8 slots × ctx_size × 512. K and V stored separately because K gets
    // RMSNorm+RoPE while V gets only plain (weightless) RMSNorm.
    kv_t   *d_global_k;  // [n_layers_global × ctx_size × 512]
    kv_t   *d_global_v;
    gemma4_seq cur;      // the active sequence (Phase 1: the only one). cur.n_tokens
                         // is the absolute KV position, formerly `global_n_tokens`.
    int     global_kv_capacity;

    // ── Multi-sequence continuous-batching slots (Phase 3) ────────────────
    // Independent in-flight sequences, each owning its own paged block tables
    // into the shared pools. slots[0] is NOT eng->cur (the single-seq path keeps
    // using eng->cur untouched); these are a separate pool for the batched API.
    // Device scratch for one multi-seq batched forward over up to GEMMA4_MAX_SEQS
    // rows: per-row absolute positions (RoPE / write), per-class PagedSeqView
    // arrays, and per-row sampled token ids. Lazily allocated on first seq_add.
    gemma4_seq    slots[GEMMA4_MAX_SEQS];
    int           ms_ready;            // 1 once the ms_* device scratch is allocated
    int          *d_ms_pos;            // [MAX_SEQS] per-row absolute position
    int          *d_ms_outtok;         // [MAX_SEQS] per-row sampled token id
    PagedSeqView *d_ms_views_slid;     // [MAX_SEQS] sliding-class per-seq views
    PagedSeqView *d_ms_views_glob;     // [MAX_SEQS] global-class per-seq views
    // Per-row sampling params + RNG draw for the multiseq on-device sampler.
    float        *d_ms_temp;           // [MAX_SEQS] per-row temperature
    int          *d_ms_topk;           // [MAX_SEQS] per-row top-k
    float        *d_ms_topp;           // [MAX_SEQS] per-row top-p
    float        *d_ms_minp;           // [MAX_SEQS] per-row min-p
    float        *d_ms_rnd;            // [MAX_SEQS] per-row uniform draw (host-hashed)

    // CUDA-graph MULTI-SEQ continuous-batching decode: the B-row forward
    // (decode_multiseq_forward body) captured once PER batch size B (grids depend
    // on B) and replayed each step. Device-resident-pos trick as the spec-verify
    // graph: per-row positions (d_ms_pos), per-row block-table device pointers
    // (in d_ms_views_slid/glob) and per-row tokens (d_sb[0]) are refreshed OUTSIDE
    // the capture each step, so one captured graph replays across steps. The
    // attention launches at FIXED max split grids (each row tail-returns past its
    // own n_splits — bit-identical to the per-kernel path). Indexed by B
    // (1..GEMMA4_MAX_SEQS); [0] unused. NULL/failed at a B → that B uses per-kernel
    // launches. FUCINA_NO_BATCHED_GRAPH disables capture globally.
    cudaGraphExec_t multiseq_graph[GEMMA4_MAX_SEQS + 1];
    int      multiseq_graph_failed;    // global disable (env or capture failure)
    // per-B "captured" log guard (bit b set once logged). 64-bit so the (1ULL<<B)
    // shift is well-defined for every B in 1..GEMMA4_MAX_SEQS (asserted below).
    uint64_t multiseq_graph_logged;
    static_assert(GEMMA4_MAX_SEQS <= 63, "multiseq_graph_logged bitmask overflow");

    // Qwen3.5 hybrid runtime ownership: recurrent state, FULL KV, workspace, weight caches and
    // graph cache are isolated from the generic engine data model. See qwen35_state.cuh.
    qwen35_runtime_state q35;

    // ── Paged KV pools (Phase 2; allocated only when paged mode is enabled) ──
    // The continuous-batching KV store: one physical block pool per cache class,
    // shared by all in-flight sequences (each holds a per-seq block table into
    // these pools — see gemma4_seq). A block reserves PAGED_KV_BLOCK_TOKENS (256)
    // positions across EVERY layer of the class; layout per class is
    // [block_id][layer][offset][elems_per_token] (sliding elems=8×256, global
    // elems=1×512). d_*_pool_* are device storage; *_pool are the host free-lists
    // (paged_kv.h). NULL/zero when paged mode is off — the contiguous d_sliding_*
    // / d_global_* path above stays the default until the paged read/write paths
    // are wired and validated against it.
    int            paged_enabled;       // 1 once the pools are allocated
    int            paged_cap;           // max concurrent sequences the POOL can back at full
                                        // per-seq reservation (= min(MAX_SEQS, max_seqs+1)).
                                        // Admission is gated on this, NOT the slot count, so a
                                        // batch can never over-subscribe the block pool and
                                        // exhaust it mid-generation (would fail the whole batch).
    int            paged_read;          // 1 → decode reads attention from the paged pool
                                        // (the contiguous mirror still runs); used to flip
                                        // the read path and validate it drives generation.
    PagedBlockPool slid_pool;           // sliding-class block free-list (host)
    PagedBlockPool glob_pool;           // global-class  block free-list (host)
    // Cross-request prefix cache (RadixAttention) over the GLOBAL pool. Only ever
    // allocated/active on the full-attention single-pool geometry (n_layers_sliding
    // == 0, i.e. Qwen3); a hard no-op for Gemma's sliding+global geometry, which
    // keeps its KV bookkeeping byte-identical. See cuda/paged_prefix.h.
    PrefixTree     glob_prefix;         // radix tree + refcount + LRU (glob_prefix.refcount==NULL when off)
    int            prefix_cache_enabled;// 1 iff active for this engine
    kv_t          *d_slid_pool_k;       // [n_blocks × 40 sliding-layers × 256 × 2048] fp8
    kv_t          *d_slid_pool_v;
    kv_t          *d_glob_pool_k;       // [n_blocks × n_global-layers × 256 × 512] fp8
    kv_t          *d_glob_pool_v;
    // Max dynamic shared memory (bytes) the global-attn kernel may use, after
    // opting in past the 48 KB static cap. Used to bound n_ctx and to produce a
    // clear error instead of a silent kernel no-op when the cache outgrows it.
    int     global_attn_max_smem;

    // Convenience macro for CUDA free (used in destroy function)
    #define CUDA_FREE(ptr) do { if (ptr) { cudaFree(ptr); ptr = NULL; } } while(0)
    // Remove the duplicate CUDA_FREE in the destroy function
    // by undefining here and redefining there
    #define CUDA_FREE_HELPER 1

    // cuBLAS handle
    cublasHandle_t cublas;

    // CUDA stream
    cudaStream_t stream;

    // Device properties
    int    device_id;
    size_t free_mem;
    size_t total_mem;
    double gpu_mem_util;   // --gpu-mem-util: fraction of total_mem the engine may use (vLLM-style)

    // Timing accumulators
    float prefill_time_ms;
    float decode_time_ms;
    int   n_prefill_tokens;
    int   n_decode_tokens;
    // Engine-resident timing events reused across decodes (Step 4): no per-token churn/leak.
    cudaEvent_t ev_start, ev_stop;
    // Lazy decode timing (llama.cpp-audit #30): a recorded (start,stop) pair is harvested
    // on a LATER call once it has naturally completed — the hot path never blocks on it.
    int ev_pending;            // a (start,stop) pair is recorded and awaits readout
    int ev_pending_tokens;     // tokens the pending pair covers

    // CUDA-graph single-token decode (Step 10 redux): the whole embed→48 layers→head
    // sequence captured once and replayed per token. Per-call state is device-resident:
    // d_decpos[2] = {pos, pos+1} (rope/kv-write read [0], attention n_tokens reads [1]),
    // d_dectok = input token id. 12 bytes async H2D + 1 graph launch replace ~1,390
    // per-kernel launches. NULL/failed → the per-kernel scalar path stays in service.
    cudaGraphExec_t decode_graph;
    int      decode_graph_failed;
    int     *d_decpos;
    int32_t *d_dectok;

    // CUDA-graph BATCHED spec-verify decode: the K>1 forward [g,draft...] in one weight
    // pass, captured once PER K (grids depend on K) and replayed each verify step. Same
    // device-resident-pos trick as the single-token graph: d_specpos[2]={pos,pos+1},
    // d_spectok = the K input ids (engine-resident d_sb[0]). Indexed by K (1..SPEC_MAX);
    // batched_graph[0] unused. NULL/failed at a given K → that K uses per-kernel launches.
    cudaGraphExec_t batched_graph[GEMMA4_SPEC_MAX + 1];
    int      batched_graph_failed;     // global disable (env or capture failure)
    int     *d_specpos;                // {pos, pos+1}

    // Speculative-decode acceptance accounting (cumulative across all spec calls), so
    // /metrics can report τ. Updated in run_spec_loop; surfaced via getters.
    long     spec_steps;               // verify forwards (K>1 steps)
    long     spec_drafted;             // draft tokens proposed
    long     spec_accepted;            // draft tokens accepted
    long     spec_emitted;             // total tokens emitted by spec (accepted + bonus)

    // Loaded flag
    int loaded;

    // Cooperative prefill abort: set from another thread (no lock — the prefill
    // holds the engine mutex) via gemma4_engine_abort_prefill(); polled between
    // prefill chunks. Cleared at every prefill entry. Plain int is fine for a
    // sticky advisory flag (worst case the abort lands one chunk late).
    volatile int abort_req;

    // ─── M0: runtime model configuration ────────────────────────────────
    // Auto-detected from the checkpoint's own metadata (GGUF kv / safetensors
    // config.json) at create time. The host alloc / loop / KV-sizing paths read
    // these fields instead of the GEMMA4_* #defines (which remain the 12B values
    // and the constant head dims). Static arrays size to GEMMA4_CAP_*; cfg.n_*
    // give the live counts. See docs/dense-31b-89tok-plan.md (M0).
    gemma4_model_config_t cfg;

    // ─── Sparse-MoE FFN ──────────────────────────────
    // Per-layer expert/router tensor offsets (into d_weights, read via weight_fp8) + the per-layer
    // ffn_down_exps type. gate/up are uniformly Q4_K (grouped Q4_K GEMM); down is Q4_K on most
    // layers and Q5_K on a few — the Q5_K slabs are requantized to Q8_0 into moe_down_q8[l] at
    // load (the grouped down GEMM has no Q5_K path) and dispatched via dg_mmq_q8_0_grouped.
    uint64_t        moe_gate_exps[GEMMA4_CAP_LAYERS];   // blk.l.ffn_gate_exps.weight (Q4_K)
    uint64_t        moe_up_exps[GEMMA4_CAP_LAYERS];     // blk.l.ffn_up_exps.weight   (Q4_K)
    uint64_t        moe_down_exps[GEMMA4_CAP_LAYERS];   // blk.l.ffn_down_exps.weight (Q4_K layers)
    uint64_t        moe_router[GEMMA4_CAP_LAYERS];      // blk.l.ffn_gate_inp.weight  (F32)
    unsigned char  *moe_down_q8[GEMMA4_CAP_LAYERS];     // requantized Q8_0 down slab (Q5_K layers) or NULL
    int64_t         moe_gate_slab, moe_up_slab;         // bytes per expert (Q4_K)
    int64_t         moe_down_slab_q4k, moe_down_slab_q8;// bytes per expert (down)
    int             moe_loaded;                         // 1 once expert tensors are resolved
    // MoE forward scratch (lazy, QWEN3MOE only). Processes ≤ GEMMA4_MOE_TMAX tokens per chunk;
    // total assignments = tokens·n_experts_used. FP32 column-major [feat, tokens] throughout.
    int             moe_scratch_ready;
    size_t          moe_scratch_bytes; // exact bytes successfully allocated by moe_alloc_scratch
    float          *d_moe_rlogits;   // [TMAX·n_experts] router logits (per-token softmax input)
    int            *d_moe_tki;       // [TMAX·n_used] top-k expert ids
    float          *d_moe_tkw;       // [TMAX·n_used] renormalized router weights
    int            *d_moe_eidx;      // [TMAX·n_used] route src token (gather/scatter idx)
    int            *d_moe_invpos;    // [TMAX·n_used] assignment i → grouped column (deterministic reduce)
    float          *d_moe_ecs;       // [TMAX·n_used] route colscale (router weight)
    int            *d_moe_count;     // [n_experts] per-expert assignment count
    int            *d_moe_coloff;    // [n_experts] exclusive prefix (column base)
    int            *d_moe_cursor;    // [n_experts] route scratch
    float          *d_moe_ones;      // [n_experts] all-ones pes (Qwen3 has no per-expert scale)
    float          *d_moe_xe;        // [H·assign] gathered expert input
    float          *d_moe_gate;      // [expert_ffn·assign] grouped gate
    float          *d_moe_up;        // [expert_ffn·assign] grouped up
    float          *d_moe_act;       // [expert_ffn·assign] silu(gate)*up
    float          *d_moe_oe;        // [H·assign] grouped down output
    int8_t         *d_moe_q8;        // [H·assign] int8 quantized activations
    float          *d_moe_q8d;       // [(H/32)·assign] per-block scale
    int            *d_moe_q8s;       // [(H/32)·assign] per-block Σ

    // ─── Qwen3.5-MoE FP8 extras (FORMAT_FP8_BLOCK; qwen3_5_moe safetensors) ──
    // Expert weights reuse moe_{gate,up,down}_exps as FP8 slab offsets (moe_*_slab = bytes/expert);
    // the per-128×128 BF16 block-scales live in d_weights as parallel per-expert slabs.
    uint64_t        moe_gate_scales[GEMMA4_CAP_LAYERS]; // BF16 scale slab (per-expert stride below)
    uint64_t        moe_up_scales[GEMMA4_CAP_LAYERS];
    uint64_t        moe_down_scales[GEMMA4_CAP_LAYERS];
    uint64_t        moe_sh_gate[GEMMA4_CAP_LAYERS];     // shared_expert.{gate,up,down}_proj (FP8,
    uint64_t        moe_sh_up[GEMMA4_CAP_LAYERS];       //   scale via fp8_scale_tab like dense projs)
    uint64_t        moe_sh_down[GEMMA4_CAP_LAYERS];
    uint64_t        moe_sh_gatevec[GEMMA4_CAP_LAYERS];  // mlp.shared_expert_gate [H] f32 (sigmoid gate)
    int             moe_shared_inter;                   // shared_expert_intermediate (0 = no shared expert)
    float          *d_moe_shlog;     // [TMAX] shared-expert gate logits (per token)
    int            *d_moe_active;    // [TMAX·n_used] compacted active-expert ids (-1 padded)
    __nv_bfloat16  *d_moe_wbf[3];    // per-layer BF16 dequant of the gate/up/down expert slabs
    __nv_bfloat16  *d_moe_xbf;       // [TMAX·n_used × max(H,EFFN)] BF16 grouped activations
    __nv_bfloat16  *d_moe_shbf;      // PERSISTENT BF16 shared-expert projs [L][3][SI·H] (~240 MB)
    int             moe_shbf_ready[GEMMA4_CAP_LAYERS]; // per-layer first-touch dequant done
    int             moe_tc_off;      // 1 = tensor-core expert prefill unavailable (alloc failed)
    int             moe_experts_q4k; // 1 = experts requantized FP8→Q4_K at load (fewer decode bytes)

    // Calibration-only router telemetry. Allocated lazily by
    // gemma4_engine_moe_profile_start; NULL keeps normal serving unchanged.
    unsigned long long *d_moe_profile_count;   // [layer,expert] selected-route count
    double             *d_moe_profile_weight;  // [layer,expert] selected-weight sum
    double             *d_moe_profile_act_ss;  // [layer,5] activation sum-of-squares
    unsigned long long *d_moe_profile_act_n;   // [layer,5] activation element count
    unsigned int       *d_moe_profile_act_max; // [layer,5] max-abs float bits

    // NVFP4 grouped-CUTLASS experts (qwen3_5_moe DEFAULT): fused gate|up (N=2·EFFN) + down slabs
    // per layer, E2M1 packed + per-expert swizzled ue4m3 SF, consumed by dg_fp4_moe_grouped.
    int             moe_experts_fp4;                 // 1 = experts requantized FP8→NVFP4 at load
    uint8_t        *d_fp4m_gu[GEMMA4_CAP_LAYERS];    // [E][2·EFFN][H/2] packed E2M1
    uint8_t        *d_fp4m_gusf[GEMMA4_CAP_LAYERS];  // [E][fp4m_gu_sfB] swizzled ue4m3
    uint8_t        *d_fp4m_dn[GEMMA4_CAP_LAYERS];    // [E][H][EFFN/2]
    uint8_t        *d_fp4m_dnsf[GEMMA4_CAP_LAYERS];  // [E][fp4m_dn_sfB]
    // Opt-in bounded-memory SSD streaming: transformed slabs live in one immutable file;
    // compact host/device slot pools cache active logical (layer,expert) records.
    uint8_t        *h_fp4m_stage_gu, *h_fp4m_stage_gusf, *h_fp4m_stage_dn, *h_fp4m_stage_dnsf;
    uint8_t        *d_fp4m_stage_gu, *d_fp4m_stage_gusf, *d_fp4m_stage_dn, *d_fp4m_stage_dnsf;
    int             fp4m_ssd_stream, fp4m_slots;
    int            *d_fp4m_eslot;
    int             fp4m_ssd_fd;
    int64_t         fp4m_ssd_gu_off[GEMMA4_CAP_LAYERS], fp4m_ssd_gusf_off[GEMMA4_CAP_LAYERS];
    int64_t         fp4m_ssd_dn_off[GEMMA4_CAP_LAYERS], fp4m_ssd_dnsf_off[GEMMA4_CAP_LAYERS];
    uint64_t       *h_fp4m_ssd_hash; // [layer,expert,4] FNV-1a for gu/gusf/dn/dnsf records
    uint8_t        *h_fp4m_ssd_verified;
    int            *h_fp4m_slot_layer, *h_fp4m_slot_expert;
    uint64_t       *h_fp4m_slot_age, fp4m_slot_clock;
    uint64_t        fp4m_ssd_reads, fp4m_ssd_bytes, fp4m_ssd_checksum_fail;
    uint64_t        fp4m_cache_hits, fp4m_cache_misses, fp4m_prefetch_advice;
    fucina_expert_profile_t *ssd_expert_profile; // NULL unless SSD + explicit profile env gate
    ExpertWeightRef ref_fp4m_gu[GEMMA4_CAP_LAYERS];
    ExpertWeightRef ref_fp4m_dn[GEMMA4_CAP_LAYERS];
    ExpertWeightRef ref_moe_gate[GEMMA4_CAP_LAYERS];
    ExpertWeightRef ref_moe_up[GEMMA4_CAP_LAYERS];
    ExpertWeightRef ref_moe_down[GEMMA4_CAP_LAYERS];
    float          *d_fp4m_gsw;                      // device [L·2] per-(layer,proj) global scales
    unsigned long long fp4m_gu_sfB, fp4m_dn_sfB;     // per-expert SF strides (SF elements)
    uint8_t        *d_fp4m_a,  *d_fp4m_asf;          // per-step activation E2M1 + padded SF (K=H)
    uint8_t        *d_fp4m_a2, *d_fp4m_a2sf;         //   " for the down proj (K=EFFN)
    __nv_bfloat16  *d_fp4m_gu_out;                   // [A][2·EFFN] grouped-GEMM output
    __nv_bfloat16  *d_fp4m_dn_out;                   // [A][H]
    int            *d_fp4m_indptr, *d_fp4m_t2e;      // [E+1] prefix / [A] assignment→expert
    float          *d_fp4m_gsrow;                    // [A] per-row activation global scales

    // ─── LoRA adapter support ───────────────────────────────────────────

    // Registry-owned allocations null their compatibility slots during teardown.
    DeviceAllocationRegistry *device_allocations;
};

// ─── GGUF Parser ───────────────────────────────────────────────────────

// Look up a metadata key and copy its scalar value into *value_out.
// For GGUF_TYPE_ARRAY, *value_out receives a `gguf_array_t` describing the
// element type, count and a pointer to the first element.
//
// expected_type is the GGUF type the caller expects; if the stored type does
// not match, the function still returns 0 but does not write (callers may
// pass the actual stored type). For convenience, integer width promotion is
// NOT performed — ask for the exact type present in the file.
static int gguf_parse_metadata(
    const uint8_t *data,
    uint64_t       size,
    const char    *key,
    void          *value_out,
    gguf_value_type_t expected_type)
{
    const gguf_header_t *hdr = (const gguf_header_t *)data;
    if (hdr->magic != 0x46554747) return -1; // "GGUF"

    const uint8_t *end = data + size;
    const uint8_t *p = data + sizeof(gguf_header_t);

    for (uint64_t i = 0; i < hdr->metadata_kv_count; i++) {
        uint64_t klen = 0;
        const char *k = gguf_read_str(&p, end, &klen);
        if (!k) return -1;
        if (p + 4 > end) return -1;
        uint32_t vtype; memcpy(&vtype, p, 4); p += 4;

        if (gguf_str_eq(k, klen, key)) {
            if (vtype != (uint32_t)expected_type) return -1;
            switch (vtype) {
                case GGUF_TYPE_UINT8:
                case GGUF_TYPE_INT8:
                case GGUF_TYPE_BOOL:    memcpy(value_out, p, 1); break;
                case GGUF_TYPE_UINT16:
                case GGUF_TYPE_INT16:   memcpy(value_out, p, 2); break;
                case GGUF_TYPE_UINT32:
                case GGUF_TYPE_INT32:
                case GGUF_TYPE_FLOAT32: memcpy(value_out, p, 4); break;
                case GGUF_TYPE_UINT64:
                case GGUF_TYPE_INT64:
                case GGUF_TYPE_FLOAT64: memcpy(value_out, p, 8); break;
                case GGUF_TYPE_STRING: {
                    uint64_t slen; const char *s = gguf_read_str(&p, end, &slen);
                    if (!s) return -1;
                    gguf_str_t sv = { s, slen };
                    memcpy(value_out, &sv, sizeof(sv));
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

    return -1; // key not found
}

// Core tensor lookup. Walks the (variable-length) tensor_info array and, on a
// name match, returns:
//   *offset_out   = absolute byte offset in the file of the tensor data
//   *n_el_out     = total number of logical elements (product of dims)
//   *ggml_type_out= GGML element type (may be NULL)
static int gguf_find_tensor(
    const uint8_t *data,
    uint64_t       size,
    const char    *name,
    uint64_t      *offset_out,
    uint64_t      *n_el_out,
    uint32_t      *ggml_type_out)
{
    const gguf_header_t *hdr = (const gguf_header_t *)data;
    if (size < sizeof(gguf_header_t)) return -1;
    if (hdr->magic != 0x46554747) return -1;
    if (hdr->version != 3) return -1; // forward-compat guard: reject unknown GGUF versions

    const uint8_t *end = data + size;
    const uint8_t *p = gguf_skip_metadata(data, size);
    if (!p) return -1;

    uint64_t tdata_start = gguf_tensor_data_start(data, size);
    if (tdata_start == 0 || tdata_start > size) return -1;

    for (uint64_t t = 0; t < hdr->tensor_count; t++) {
        uint64_t nlen = 0;
        const char *tname = gguf_read_str(&p, end, &nlen);
        if (!tname) return -1;
        if (p + 4 > end) return -1;
        uint32_t n_dims; memcpy(&n_dims, p, 4); p += 4;
        if (p + (uint64_t)n_dims * 8 + 12 > end) return -1;
        uint64_t n_el = 1;
        for (uint32_t d = 0; d < n_dims; d++) {
            uint64_t dv; memcpy(&dv, p, 8); p += 8;
            n_el *= dv;
        }
        uint32_t gtype; memcpy(&gtype, p, 4); p += 4;
        uint64_t toff;  memcpy(&toff, p, 8);  p += 8;

        if (gguf_str_eq(tname, nlen, name)) {
            // Validate the tensor's data region lies within the file. toff is
            // relative to tdata_start; a corrupt offset (or one that, with the
            // tensor bytes, escapes the mapping) would otherwise drive an OOB
            // device read at inference. Exact byte size is quant-dependent, so we
            // bound the start here; the per-tensor byte span is re-checked at
            // upload where wrow_bytes() knows the format.
            if (toff > size - tdata_start) return -1;
            if (offset_out)    *offset_out = tdata_start + toff;
            if (n_el_out)      *n_el_out = n_el;
            if (ggml_type_out) *ggml_type_out = gtype;
            return 0;
        }
    }
    return -1;
}

// Backwards-compatible wrapper: returns absolute offset + element count.
static int gguf_find_tensor_offset(
    const uint8_t *data,
    uint64_t       size,
    const char    *name,
    uint64_t      *offset_out,
    uint64_t      *n_bytes_out)
{
    return gguf_find_tensor(data, size, name, offset_out, n_bytes_out, NULL);
}

// Resolve a tensor's per-dimension shape (ggml ne[]: ne[0]=fastest/in-dim, ne[1]=out-dim, …).
// Fills dims_out[0..3] (unused dims left at their incoming value is NOT guaranteed — caller
// should pre-zero) and *n_dims_out. Returns 0 on found, -1 otherwise. Mirrors gguf_find_tensor.
static int gguf_tensor_dims(
    const uint8_t *data,
    uint64_t       size,
    const char    *name,
    uint64_t       dims_out[4],
    int           *n_dims_out)
{
    const gguf_header_t *hdr = (const gguf_header_t *)data;
    if (size < sizeof(gguf_header_t)) return -1;
    if (hdr->magic != 0x46554747) return -1;
    if (hdr->version != 3) return -1;

    const uint8_t *end = data + size;
    const uint8_t *p = gguf_skip_metadata(data, size);
    if (!p) return -1;

    for (uint64_t t = 0; t < hdr->tensor_count; t++) {
        uint64_t nlen = 0;
        const char *tname = gguf_read_str(&p, end, &nlen);
        if (!tname) return -1;
        if (p + 4 > end) return -1;
        uint32_t n_dims; memcpy(&n_dims, p, 4); p += 4;
        if (p + (uint64_t)n_dims * 8 + 12 > end) return -1;
        uint64_t dv[4] = {1, 1, 1, 1};
        for (uint32_t d = 0; d < n_dims; d++) {
            uint64_t v; memcpy(&v, p, 8); p += 8;
            if (d < 4) dv[d] = v;
        }
        p += 12; // skip gtype(4) + offset(8)
        if (gguf_str_eq(tname, nlen, name)) {
            if (dims_out)   for (int d = 0; d < 4; d++) dims_out[d] = dv[d];
            if (n_dims_out) *n_dims_out = (int)n_dims;
            return 0;
        }
    }
    return -1;
}

// Map a GGML tensor type → fucina FORMAT_*; -1 for an unsupported bulk-weight type.
static inline int ggml_to_fmt(uint32_t gt) {
    switch (gt) {
        case GGML_TYPE_Q4_0: return FORMAT_Q4_0;
        // Q4_1 (Unsloth UD dynamic-quant ffn_down) has no native kernel; it is requantized to
        // Q4_0 at load into a per-tensor device buffer + pointer override (see the "Unsloth UD
        // dynamic-quant" block), so its resolved format IS Q4_0. Returning FORMAT_Q4_0 here lets
        // LOAD_WT_FMT record fmt_down=Q4_0 (matching the override bytes) instead of tripping the
        // unsupported-type guard and aborting the load before the requant ever runs.
        case GGML_TYPE_Q4_1: return FORMAT_Q4_0;
        case GGML_TYPE_Q8_0: return FORMAT_Q8_0;
        case GGML_TYPE_Q4_K: return FORMAT_Q4_K;
        case GGML_TYPE_Q5_K: return FORMAT_Q5_K;  // requantized to Q8_0 at load (no native kernel)
        case GGML_TYPE_Q6_K: return FORMAT_Q6_K;
        default: return -1;
    }
}

// ─── Engine Construction ──────────────────────────────────────────────

// Stream the GGUF tensor-data region into the device weight buffer using a pinned
// double-buffer: a sequential pread (full cache/NVMe bandwidth, ~11 GB/s) overlapped
// with a pinned H2D copy (~57 GB/s). This replaces a direct cudaMemcpy from the
// lazily-mmap'd file, which crawls at ~256 MB/s (~48 s for 11.8 GB) because it
// faults in 12 GB of file-backed pages ON DEMAND in random order through the CUDA
// pageable-copy path. `fd` must be the open GGUF; `foff` the tensor-data offset.
// Returns 0 on success, -1 on failure (caller falls back to the mmap copy).
static int upload_weights_streamed(unsigned char *d_dst, int fd,
                                   off_t foff, size_t tbytes)
{
    const size_t CHUNK = 256ull * 1024 * 1024;      // 256 MB per buffer
    unsigned char *pinned[2] = { NULL, NULL };
    cudaStream_t   s[2]      = { NULL, NULL };
    cudaEvent_t    done[2]   = { NULL, NULL };
    int rc = 0;

    if (cudaHostAlloc((void **)&pinned[0], CHUNK, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void **)&pinned[1], CHUNK, cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError(); rc = -1; goto cleanup;
    }
    cudaStreamCreate(&s[0]); cudaStreamCreate(&s[1]);
    cudaEventCreate(&done[0]); cudaEventCreate(&done[1]);

    {
        bool   inflight[2] = { false, false };
        size_t copied = 0;
        int    bi = 0;
        while (copied < tbytes) {
            size_t n = tbytes - copied; if (n > CHUNK) n = CHUNK;
            // Reuse buffer bi only after its previous H2D has drained.
            if (inflight[bi]) { cudaEventSynchronize(done[bi]); inflight[bi] = false; }
            // Sequential read into the pinned buffer (overlaps the other stream's H2D).
            size_t got = 0;
            while (got < n) {
                ssize_t r = pread(fd, pinned[bi] + got, n - got,
                                  foff + (off_t)(copied + got));
                if (r <= 0) { rc = -1; break; }
                got += (size_t)r;
            }
            if (rc) break;
            cudaMemcpyAsync(d_dst + copied, pinned[bi], n,
                            cudaMemcpyHostToDevice, s[bi]);
            cudaEventRecord(done[bi], s[bi]);
            inflight[bi] = true;
            copied += n;
            bi ^= 1;
        }
        cudaStreamSynchronize(s[0]);
        cudaStreamSynchronize(s[1]);
        if (cudaGetLastError() != cudaSuccess) rc = -1;
    }

cleanup:
    if (pinned[0]) cudaFreeHost(pinned[0]);
    if (pinned[1]) cudaFreeHost(pinned[1]);
    if (s[0])    cudaStreamDestroy(s[0]);
    if (s[1])    cudaStreamDestroy(s[1]);
    if (done[0]) cudaEventDestroy(done[0]);
    if (done[1]) cudaEventDestroy(done[1]);
    return rc;
}

// Explicit IEEE fp16<->fp32 (host) — avoid relying on __half host conversions.
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
static inline uint16_t f2h_host(float x) {
    uint32_t f; memcpy(&f, &x, 4);
    uint16_t s = (uint16_t)((f >> 16) & 0x8000);
    int32_t  e = (int32_t)((f >> 23) & 0xFF) - 112;   // 127 - 15
    uint32_t m = f & 0x7FFFFF;
    if (e >= 0x1F) return (uint16_t)(s | 0x7C00);
    if (e <= 0) {
        if (e < -10) return s;
        m |= 0x800000; uint32_t sh = (uint32_t)(14 - e);
        uint16_t hm = (uint16_t)(m >> sh);
        if ((m >> (sh - 1)) & 1) hm++;
        return (uint16_t)(s | hm);
    }
    uint16_t h = (uint16_t)(s | (uint16_t)(e << 10) | (uint16_t)(m >> 13));
    if (m & 0x1000) h++;
    return h;
}

// ─── Unsloth UD dynamic-quant mixed-type support (31B) ─────────────────────
// The Unsloth gemma-4-31B Q4_0 dynamic GGUF is mostly Q4_0, but a few tensors ship in
// other GGML types: blk.0..6.ffn_down = Q4_1 (20 B/32-block), token_embd = Q4_K
// (144 B/256-superblock). The engine's GEMV/MMQ kernels only read Q4_0/Q8_0/Q6_K, so at
// load we DEQUANTIZE those tensors to float and REQUANTIZE them into a format an existing
// kernel reads: ffn_down → Q4_0 (separate per-layer device buffer + pointer override),
// token_embd → Q8_0 (the same path Q6_K token_embd already uses). The 12B model contains
// none of these types, so all of this is dead for it (byte-identical).

// Q4_1 block = 32 elems: fp16 d, fp16 m, 16 nibble bytes. value = d*q + m
// (q is the raw 0..15 nibble; low nibbles → elems 0..15, high nibbles → elems 16..31).
static inline void dequant_q4_1_block(const unsigned char *blk, float out[32]) {
    uint16_t dr, mr; memcpy(&dr, blk, 2); memcpy(&mr, blk + 2, 2);
    float d = h2f_host(dr), m = h2f_host(mr);
    const unsigned char *qs = blk + 4;
    for (int j = 0; j < 16; j++) {
        out[j]      = d * (float)(qs[j] & 0x0F) + m;
        out[j + 16] = d * (float)(qs[j] >> 4)   + m;
    }
}

// Q4_K super-block = 256 elems (144 B): fp16 d, fp16 dmin, 12 B of 6-bit packed
// scales/mins (8 sub-blocks), then 128 B of 4-bit quants. Standard ggml k-quant layout.
// Sub-block i value = d*sc_i*q - dmin*m_i (sc_i,m_i are the 6-bit scale/min, q the nibble).
static inline void dequant_q4_k_superblock(const unsigned char *blk, float out[256]) {
    uint16_t dr, mr; memcpy(&dr, blk, 2); memcpy(&mr, blk + 2, 2);
    float d = h2f_host(dr), dmin = h2f_host(mr);
    const unsigned char *sc = blk + 4;       // 12 bytes of packed 6-bit scales+mins
    const unsigned char *qs = blk + 16;      // 128 bytes of 4-bit quants
    // Unpack the 8 sub-block (scale, min) pairs — ggml get_scale_min_k4 packing.
    auto get_sm = [&](int j, uint8_t *s, uint8_t *m) {
        if (j < 4) {
            *s = sc[j] & 63;
            *m = sc[j + 4] & 63;
        } else {
            *s = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
            *m = (sc[j + 4] >> 4)   | ((sc[j    ] >> 6) << 4);
        }
    };
    for (int j = 0; j < 8; j++) {
        uint8_t s, m; get_sm(j, &s, &m);
        float dl = d * (float)s, ml = dmin * (float)m;
        // Each pair of sub-blocks shares 32 quant bytes: lower nibble = sub-block 2k,
        // upper nibble = sub-block 2k+1. So sub-block j uses qs[(j/2)*32 .. +32], nibble j&1.
        const unsigned char *q = qs + (j / 2) * 32;
        int shift = (j & 1) ? 4 : 0;
        float *o = out + j * 32;
        for (int i = 0; i < 32; i++)
            o[i] = dl * (float)((q[i] >> shift) & 0x0F) - ml;
    }
}

// Standard ggml quantize_row_q4_0: per 32-element block, scale by the magnitude-max
// element so the symmetric nibble range maps to ±|amax|. d = amax / -8 (so the most
// negative element → nibble 0). Q4_0 block = fp16 d (2 B) + 16 nibble bytes (18 B).
static inline void quantize_row_q4_0_host(const float *x, unsigned char *dst, int64_t n_elem) {
    const int64_t n_blk = n_elem / 32;
    for (int64_t b = 0; b < n_blk; b++) {
        const float *xb = x + b * 32;
        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < 32; j++) {
            float v = xb[j];
            if (fabsf(v) > amax) { amax = fabsf(v); max = v; }
        }
        float d  = max / -8.0f;
        float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        unsigned char *ob = dst + (size_t)b * 18;
        uint16_t hb = f2h_host(d);
        ob[0] = (unsigned char)(hb & 0xFF); ob[1] = (unsigned char)(hb >> 8);
        for (int j = 0; j < 16; j++) {
            float x0 = xb[j]      * id + 8.5f;
            float x1 = xb[j + 16] * id + 8.5f;
            int q0 = (int)x0; q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
            int q1 = (int)x1; q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
            ob[2 + j] = (unsigned char)(q0 | (q1 << 4));
        }
    }
}

// Requantize a Q4_1 tensor → Q4_0 (host, once at load). Returns a malloc'd Q4_0 buffer of
// (n_elem/32)*18 bytes, or NULL on error. Used for the 31B ffn_down (layers 0..6).
static unsigned char* convert_q4_1_to_q4_0(const unsigned char *src, int64_t n_elem) {
    const int64_t n_blk = n_elem / 32;
    unsigned char *dst = (unsigned char *)malloc((size_t)n_blk * 18);
    if (!dst) return NULL;
    float f[32];
    for (int64_t b = 0; b < n_blk; b++) {
        dequant_q4_1_block(src + (size_t)b * 20, f);
        quantize_row_q4_0_host(f, dst + (size_t)b * 18, 32);
    }
    return dst;
}

// Requantize a Q4_K tensor → Q8_0 (host, once at load). Mirrors convert_q6k_to_q8_0 so the
// existing Q8_0 embed-lookup / LM-head kernels handle the converted token_embd unchanged.
// Returns a malloc'd Q8_0 buffer of (n_elem/32)*34 bytes, or NULL on error.
static unsigned char* convert_q4k_to_q8_0(const unsigned char *src, int64_t n_elem) {
    const int64_t n_super = n_elem / 256;
    const int64_t n_q8blk = n_elem / 32;
    unsigned char *dst = (unsigned char *)malloc((size_t)n_q8blk * 34);
    if (!dst) return NULL;
    float f[256];
    for (int64_t s = 0; s < n_super; s++) {
        dequant_q4_k_superblock(src + (size_t)s * 144, f);
        for (int sb = 0; sb < 8; sb++) {
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(f[sb * 32 + j]));
            float scale = amax / 127.0f, iscale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
            unsigned char *ob = dst + (size_t)(s * 8 + sb) * 34;
            uint16_t hb = f2h_host(scale);
            ob[0] = (unsigned char)(hb & 0xFF); ob[1] = (unsigned char)(hb >> 8);
            for (int j = 0; j < 32; j++) {
                int q = (int)lrintf(f[sb * 32 + j] * iscale);
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                ob[2 + j] = (unsigned char)(int8_t)q;
            }
        }
    }
    return dst;
}

// Q5_K super-block = 256 elems (176 B): fp16 d, fp16 dmin, 12 B of 6-bit packed scales/mins
// (same get_scale_min_k4 packing as Q4_K), 32 B of high bits (qh), 128 B of low nibbles (qs).
// Sub-block j (0..7) value = d*sc_j*(nib + 16*high_bit) - dmin*m_j, where nib is qs[(j/2)*32+i]'s
// low/high nibble (j even/odd) and high_bit = (qh[i] >> j) & 1. Verified against ggml-quants.git
// dequantize_row_q5_K (u1/u2 = 1<<sb / 2<<sb ⇒ qh bit == sub-block index). Output natural order.
static inline void dequant_q5_k_superblock(const unsigned char *blk, float out[256]) {
    uint16_t dr, mr; memcpy(&dr, blk, 2); memcpy(&mr, blk + 2, 2);
    float d = h2f_host(dr), dmin = h2f_host(mr);
    const unsigned char *sc = blk + 4;       // 12 bytes packed 6-bit scales+mins
    const unsigned char *qh = blk + 16;      // 32 bytes high bits (1 per element)
    const unsigned char *qs = blk + 48;      // 128 bytes low nibbles
    auto get_sm = [&](int j, uint8_t *s, uint8_t *m) {
        if (j < 4) { *s = sc[j] & 63; *m = sc[j + 4] & 63; }
        else { *s = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
               *m = (sc[j + 4] >> 4)   | ((sc[j    ] >> 6) << 4); }
    };
    for (int j = 0; j < 8; j++) {
        uint8_t s, m; get_sm(j, &s, &m);
        float dl = d * (float)s, ml = dmin * (float)m;
        const unsigned char *q = qs + (j / 2) * 32;
        int shift = (j & 1) ? 4 : 0;
        float *o = out + j * 32;
        for (int i = 0; i < 32; i++) {
            int lo = (q[i] >> shift) & 0x0F;
            int hi = (qh[i] >> j) & 1;
            o[i] = dl * (float)(lo + (hi << 4)) - ml;
        }
    }
}

// Requantize a Q5_K tensor → Q8_0 (host, once at load). Used for the qwen3moe ffn_down_exps slabs
// that ship Q5_K (the grouped down GEMM supports Q4_K/Q8_0/Q5_0 but not Q5_K). Mirrors
// convert_q4k_to_q8_0; preserves element order so the per-expert row-major [out_dim×in_dim] layout
// is unchanged (in_dim=expert_ffn=768 = 24·32 ⇒ Q8_0 blocks never straddle a row boundary).
static unsigned char* convert_q5k_to_q8_0(const unsigned char *src, int64_t n_elem) {
    const int64_t n_super = n_elem / 256;
    const int64_t n_q8blk = n_elem / 32;
    unsigned char *dst = (unsigned char *)malloc((size_t)n_q8blk * 34);
    if (!dst) return NULL;
    float f[256];
    for (int64_t s = 0; s < n_super; s++) {
        dequant_q5_k_superblock(src + (size_t)s * 176, f);
        for (int sb = 0; sb < 8; sb++) {
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(f[sb * 32 + j]));
            float scale = amax / 127.0f, iscale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
            unsigned char *ob = dst + (size_t)(s * 8 + sb) * 34;
            uint16_t hb = f2h_host(scale);
            ob[0] = (unsigned char)(hb & 0xFF); ob[1] = (unsigned char)(hb >> 8);
            for (int j = 0; j < 32; j++) {
                int q = (int)lrintf(f[sb * 32 + j] * iscale);
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                ob[2 + j] = (unsigned char)(int8_t)q;
            }
        }
    }
    return dst;
}

// Convert a Q6_K tensor (token_embd in the QAT model) to Q8_0, host-side, once at load.
// Q6_K super-block = 256 elems (ql[128] + qh[64] + scales[16](int8) + d(fp16) = 210 B);
// we dequant to float then requantize each 32-subblock to Q8_0 (fp16 scale + 32 int8).
// Returns a malloc'd Q8_0 buffer of (n_elem/32)*34 bytes, or NULL on error.
static unsigned char* convert_q6k_to_q8_0(const unsigned char *src, int64_t n_elem)
{
    const int64_t n_super = n_elem / 256;
    const int64_t n_q8blk = n_elem / 32;
    unsigned char *dst = (unsigned char*)malloc((size_t)n_q8blk * 34);
    if (!dst) return NULL;
    for (int64_t s = 0; s < n_super; s++) {
        const unsigned char *blk = src + (size_t)s * 210;
        const unsigned char *ql0 = blk;
        const unsigned char *qh0 = blk + 128;
        const int8_t        *sc0 = (const int8_t*)(blk + 192);
        uint16_t draw; memcpy(&draw, blk + 208, 2);
        float d = h2f_host(draw);
        float f[256];
        for (int n = 0; n < 256; n += 128) {
            const unsigned char *ql = ql0 + (n/128)*64;
            const unsigned char *qh = qh0 + (n/128)*32;
            const int8_t        *sc = sc0 + (n/128)*8;
            for (int l = 0; l < 32; l++) {
                int is = l/16;
                int q1 = (int)((ql[l]    & 0xF) | (((qh[l]>>0)&3)<<4)) - 32;
                int q2 = (int)((ql[l+32] & 0xF) | (((qh[l]>>2)&3)<<4)) - 32;
                int q3 = (int)((ql[l]    >> 4)  | (((qh[l]>>4)&3)<<4)) - 32;
                int q4 = (int)((ql[l+32] >> 4)  | (((qh[l]>>6)&3)<<4)) - 32;
                f[n+l+ 0] = d * sc[is+0] * q1;
                f[n+l+32] = d * sc[is+2] * q2;
                f[n+l+64] = d * sc[is+4] * q3;
                f[n+l+96] = d * sc[is+6] * q4;
            }
        }
        for (int sb = 0; sb < 8; sb++) {
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(f[sb*32+j]));
            float scale = amax / 127.0f, iscale = (scale > 0.0f) ? 1.0f/scale : 0.0f;
            unsigned char *ob = dst + (size_t)(s*8 + sb) * 34;
            uint16_t hb = f2h_host(scale);
            ob[0] = (unsigned char)(hb & 0xFF); ob[1] = (unsigned char)(hb >> 8);
            for (int j = 0; j < 32; j++) {
                int q = (int)lrintf(f[sb*32+j] * iscale);
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                ob[2+j] = (unsigned char)(int8_t)q;
            }
        }
    }
    return dst;
}

static int build_packed_q4(gemma4_engine_t *eng);  // FUCINA_PACKED: lazy/eager repack (body below)
static int build_packed_q4k(gemma4_engine_t *eng); // PACKED Q4_K: in-place superblock repack (body below)
static inline const unsigned char* weight_fp8(const gemma4_engine_t *eng, uint64_t tensor_offset);
static void gemma4_engine_paged_selftest(gemma4_engine_t *eng);       // Phase 2 inc 3 (body below)
static void gemma4_engine_paged_read_selftest(gemma4_engine_t *eng);  // Phase 2 inc 4 (body below)
static void gemma4_engine_paged_e2e_selftest(gemma4_engine_t *eng);   // Phase 2 inc 4b (body below)
static void gemma4_engine_batch_selftest(gemma4_engine_t *eng);       // Phase 3 (body below)
static void gemma4_engine_batch_decode_bench(gemma4_engine_t *eng);   // decode micro-bench (body below)
static void gemma4_engine_fast_prefill_selftest(gemma4_engine_t *eng); // dual-path prefill determinism (body below)
// NVFP4 single-store residency (body far below, near the prefill GEMM). Forward-declared so the
// create-fork can call it.
static int nvfp4_load_from_safetensors(gemma4_engine_t *eng, const char *path,
                                       const nvfp4ld::Layout *layout, st::Model *model);
// Qwen3.5 block-FP8 checkpoint → batched engine. setup_cfg sets eng->cfg early; fill_engine fills
// d_weights/tensors + FORMAT_FP8_BLOCK + scale table late (after create's d_weights=NULL reset).
// Bodies far below (near the FP8 oracle). Forward-declared so create can call them.
static int qwen35_fp8_setup_cfg(gemma4_engine_t *eng, st::Model &M, qwen35fp8::Layout &LO);
static int qwen35_fp8_fill_engine(gemma4_engine_t *eng, st::Model &M, qwen35fp8::Layout &LO);
static std::string q35moe_resolve_dir(const char *path);   // HF-cache root → snapshots/<hash>/ dir

// Build a proxy hidden state from a real embedding row: out[h] = emb[tok][h] · w_out_norm[h] / rms.
// The trained embedding rows live in the head's input space, and the final RMSNorm is exactly what
// the decode path applies before the head — so these are realistic head inputs for the accuracy gate.
__global__ void fp8gate_proxy_hidden_kernel(
    float *out, const __nv_bfloat16 *emb, const float *wnorm,
    const int32_t *toks, int hidden, float eps)
{
    const int row = blockIdx.x;
    const __nv_bfloat16 *e = emb + (size_t)toks[row] * hidden;
    __shared__ float s_ss;
    float ss = 0.f;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) { float v = __bfloat162float(e[h]); ss += v*v; }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    __shared__ float s_part[32];
    if ((threadIdx.x & 31) == 0) s_part[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f; for (int i = 0; i < (blockDim.x + 31) / 32; i++) t += s_part[i];
        s_ss = rsqrtf(t / hidden + eps);
    }
    __syncthreads();
    float inv = s_ss;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x)
        out[(size_t)row * hidden + h] = __bfloat162float(e[h]) * inv * wnorm[h];
}

// Load-time accuracy gate for the FP8 head: quantize the BF16 head, run BOTH heads on `nsamp` real
// proxy hidden states, and return the top-1 argmax match count. Caller keeps FP8 only on a full match.
static int fp8_head_accuracy_gate(gemma4_engine_t *eng, int nsamp, double *out_worst_l2)
{
    const int H = eng->cfg.hidden_size, V = GEMMA4_VOCAB_SIZE;
    float   *dX = NULL, *dYbf = NULL, *dYfp = NULL; int32_t *dTok = NULL;
    int     *dArgBf = NULL, *dArgFp = NULL;
    int match = -1; if (out_worst_l2) *out_worst_l2 = 0;
    if (cudaMalloc(&dX,(size_t)nsamp*H*sizeof(float)) != cudaSuccess) goto done;
    if (cudaMalloc(&dYbf,(size_t)nsamp*V*sizeof(float)) != cudaSuccess) goto done;
    if (cudaMalloc(&dYfp,(size_t)nsamp*V*sizeof(float)) != cudaSuccess) goto done;
    if (cudaMalloc(&dTok,(size_t)nsamp*sizeof(int32_t)) != cudaSuccess) goto done;
    if (cudaMalloc(&dArgBf,(size_t)nsamp*sizeof(int)) != cudaSuccess) goto done;
    if (cudaMalloc(&dArgFp,(size_t)nsamp*sizeof(int)) != cudaSuccess) goto done;
    {
        std::vector<int32_t> toks(nsamp);
        for (int i = 0; i < nsamp; i++) toks[i] = (int32_t)(((long long)(i+1) * 8675309) % V);
        cudaMemcpy(dTok, toks.data(), nsamp*sizeof(int32_t), cudaMemcpyHostToDevice);
        fp8gate_proxy_hidden_kernel<<<nsamp,256>>>(dX, eng->d_embed_bf16, eng->d_w_out_norm, dTok, H, GEMMA4_RMS_EPS);

        for (int r = 0; r < nsamp; r++) {
            bf16_head_gemv_launch(dYbf+(size_t)r*V, eng->d_lmhead_bf16, dX+(size_t)r*H, H, V, 0);
            fp8_head_gemv_launch (dYfp+(size_t)r*V, eng->d_lmhead_fp8, eng->d_lmhead_fp8_scale, dX+(size_t)r*H, H, V, 0);
            argmax_kernel<<<1,32>>>(dYbf+(size_t)r*V, dArgBf+r, V);
            argmax_kernel<<<1,32>>>(dYfp+(size_t)r*V, dArgFp+r, V);
        }
        cudaDeviceSynchronize();
        std::vector<int> abf(nsamp), afp(nsamp);
        cudaMemcpy(abf.data(), dArgBf, nsamp*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(afp.data(), dArgFp, nsamp*sizeof(int), cudaMemcpyDeviceToHost);
        match = 0; for (int i = 0; i < nsamp; i++) if (abf[i] == afp[i]) match++;
        if (out_worst_l2) {
            std::vector<float> ybf((size_t)nsamp*V), yfp((size_t)nsamp*V);
            cudaMemcpy(ybf.data(), dYbf, ybf.size()*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(yfp.data(), dYfp, yfp.size()*sizeof(float), cudaMemcpyDeviceToHost);
            double worst = 0;
            for (int r = 0; r < nsamp; r++) {
                double num=0,den=0;
                for (int v=0; v<V; v++){ double d=(double)yfp[(size_t)r*V+v]-ybf[(size_t)r*V+v]; num+=d*d; den+=(double)ybf[(size_t)r*V+v]*ybf[(size_t)r*V+v]; }
                double l2 = den>0? sqrt(num/den):0; if (l2>worst) worst=l2;
            }
            *out_worst_l2 = worst;
        }
    }
done:
    if (dX) cudaFree(dX); if (dYbf) cudaFree(dYbf); if (dYfp) cudaFree(dYfp);
    if (dTok) cudaFree(dTok); if (dArgBf) cudaFree(dArgBf); if (dArgFp) cudaFree(dArgFp);
    cudaGetLastError();
    return match;
}

// M0: derive the engine's per-layer attention bookkeeping from the auto-detected
// gemma4_model_config_t (eng->cfg). Called after cfg is populated (defaults at create top,
// then overwritten by gemma4_detect_from_gguf / _config_json). Fills layer_types[],
// global_layer_indices[], global_slot[] and the n_layers_* counts from cfg.is_global[].
// Keeping this in one place means every cfg source produces an identical engine state.
static void gemma4_apply_cfg(gemma4_engine_t *eng) {
    gemma4_model_config_t *c = &eng->cfg;
    eng->n_global         = 0;
    eng->n_layers_sliding = 0;

    // Qwen3.5 hybrid: lower the per-layer attention KIND (cfg.attn_kind[]) into layer_types.
    // FULL softmax-GQA layers route through the engine's global full-attention class; LINEAR
    // gated-deltanet layers are dispatched off cfg.attn_kind[] by the (M-stage) GDN forward, and
    // are parked as LAYER_SLIDING here purely so the layer_types array is fully populated (their
    // attention is NEVER taken via the sliding/global softmax path). Gated on QWEN3_5 so no other
    // arch's layer_types lowering changes.
    if (c->arch == GEMMA4_ARCH_QWEN3_5) {
        for (int i = 0; i < c->n_layers && i < GEMMA4_CAP_LAYERS; i++) {
            if (c->attn_kind[i] == GEMMA4_ATTN_FULL) {
                eng->layer_types[i] = LAYER_GLOBAL;
                eng->global_layer_indices[eng->n_global++] = i;
            } else {
                eng->layer_types[i] = LAYER_SLIDING;  // GDN (linear); dispatched on attn_kind[]
                eng->n_layers_sliding++;
            }
        }
        eng->n_layers_global = eng->n_global;
        return;
    }

    for (int i = 0; i < c->n_layers && i < GEMMA4_CAP_LAYERS; i++) {
        if (c->is_global[i]) {
            eng->layer_types[i] = LAYER_GLOBAL;
            eng->global_layer_indices[eng->n_global++] = i;
        } else {
            eng->layer_types[i] = LAYER_SLIDING;
            eng->n_layers_sliding++;
        }
    }
    eng->n_layers_global = eng->n_global;
}

// ── Qwen3.5 hybrid (M1): per-layer tensor-layout dump + shape validation ─────────────────────────
// Prints each layer index + KIND (FULL softmax-GQA / LINEAR gated-deltanet) with the resolved GGUF
// shapes of its tensor set, and checks every shape against the arch spec derived from eng->cfg.
// Returns 0 if every tensor is present and correctly shaped, -1 otherwise (→ the load aborts).
// Self-contained: re-queries the GGUF by tensor name (independent of the recorded offset table) so a
// renamed/misshaped tensor is caught at load rather than producing garbage in the (M2+) forward.
static int qwen35_dump_and_validate(const gemma4_engine_t *eng) {
    const uint8_t *gd = eng->gguf_data; const uint64_t gs = eng->gguf_size;
    const gemma4_model_config_t *c = &eng->cfg;
    const int H   = c->hidden_size;            // 4096 hidden
    const int HD  = c->head_dim;               // 256  attention head dim
    const int NH  = c->n_heads;                // 16   query heads
    const int NKV = c->n_kv_global;            // 4    kv heads (GQA)
    const int I   = c->intermediate;           // 12288 SwiGLU FFN
    const int ST  = c->ssm_state_size;         // 128  per-head key/value state dim
    const int CK  = c->ssm_conv_kernel;        // 4    depthwise conv kernel
    const int INNER = c->ssm_inner_size;       // 4096 value path (= time_step_rank*state)
    const int GRP = c->ssm_group_count;        // 16   key/query heads
    const int TSR = c->ssm_time_step_rank;     // 32   value heads (a/b/A_log/dt width)
    const int key_dim  = GRP * ST;             // 2048 (q and k each)
    const int conv_dim = 2 * key_dim + INNER;  // 8192 (q+k+v fused = qkv out = conv channels)
    int errs = 0, nfull = 0, nlin = 0;

    uint64_t dim[4]; int nd;
    auto fetch = [&](const char *name)->bool {
        for (int i=0;i<4;i++) dim[i]=0; nd=0;
        return gguf_tensor_dims(gd, gs, name, dim, &nd) == 0;
    };
    // 2D weight check: GGUF ne = [in, out] (ne[0] fastest = in-dim).
    auto chk2 = [&](const char *name, int ein, int eout)->void {
        if (!fetch(name)) { fprintf(stderr, "    MISSING  %s\n", name); errs++; return; }
        if (!(nd==2 && (int)dim[0]==ein && (int)dim[1]==eout)) {
            fprintf(stderr, "    MISSHAPE %s ne=[%llu,%llu] expected [%d,%d]\n", name,
                    (unsigned long long)dim[0],(unsigned long long)dim[1], ein, eout); errs++; }
    };
    auto chk1 = [&](const char *name, int e0)->void {
        if (!fetch(name)) { fprintf(stderr, "    MISSING  %s\n", name); errs++; return; }
        if (!(nd==1 && (int)dim[0]==e0)) {
            fprintf(stderr, "    MISSHAPE %s ne=[%llu] expected [%d]\n", name,
                    (unsigned long long)dim[0], e0); errs++; }
    };

    fprintf(stderr, "fucina: qwen35 layout — hidden=%d head_dim=%d (q-heads %d, kv-heads %d), "
            "intermediate=%d, conv_dim=%d, state=%dx%d, rotary_dim=%d, full-interval=%d\n",
            H, HD, NH, NKV, I, conv_dim, ST, ST, c->rotary_dim, c->full_attention_interval);

    char nm[128];
    for (int l = 0; l < c->n_layers; l++) {
        bool full = (c->attn_kind[l] == GEMMA4_ATTN_FULL);
        // shared norms + dense SwiGLU FFN (both layer kinds)
        snprintf(nm,sizeof(nm),"blk.%d.attn_norm.weight",l);           chk1(nm, H);
        snprintf(nm,sizeof(nm),"blk.%d.post_attention_norm.weight",l); chk1(nm, H);
        snprintf(nm,sizeof(nm),"blk.%d.ffn_gate.weight",l);            chk2(nm, H, I);
        snprintf(nm,sizeof(nm),"blk.%d.ffn_up.weight",l);              chk2(nm, H, I);
        snprintf(nm,sizeof(nm),"blk.%d.ffn_down.weight",l);            chk2(nm, I, H);
        if (full) {
            nfull++;
            fprintf(stderr, "  L%2d FULL   attn_q[%d->%d] k/v[%d->%d] o[%d->%d] q/k_norm[%d] "
                    "ffn[%d->%d->%d]\n", l, H, 2*NH*HD, H, NKV*HD, NH*HD, H, HD, H, I, H);
            snprintf(nm,sizeof(nm),"blk.%d.attn_q.weight",l);      chk2(nm, H, 2*NH*HD);
            snprintf(nm,sizeof(nm),"blk.%d.attn_k.weight",l);      chk2(nm, H, NKV*HD);
            snprintf(nm,sizeof(nm),"blk.%d.attn_v.weight",l);      chk2(nm, H, NKV*HD);
            snprintf(nm,sizeof(nm),"blk.%d.attn_output.weight",l); chk2(nm, NH*HD, H);
            snprintf(nm,sizeof(nm),"blk.%d.attn_q_norm.weight",l); chk1(nm, HD);
            snprintf(nm,sizeof(nm),"blk.%d.attn_k_norm.weight",l); chk1(nm, HD);
        } else {
            nlin++;
            fprintf(stderr, "  L%2d LINEAR qkv[%d->%d] z[%d->%d] a/b[%d->%d] A_log[%d] dt[%d] "
                    "conv[%dx%d] norm[%d] out[%d->%d]\n", l, H, conv_dim, H, INNER, H, TSR, TSR,
                    TSR, CK, conv_dim, ST, INNER, H);
            snprintf(nm,sizeof(nm),"blk.%d.attn_qkv.weight",l);   chk2(nm, H, conv_dim);
            snprintf(nm,sizeof(nm),"blk.%d.attn_gate.weight",l);  chk2(nm, H, INNER);
            snprintf(nm,sizeof(nm),"blk.%d.ssm_alpha.weight",l);  chk2(nm, H, TSR);
            snprintf(nm,sizeof(nm),"blk.%d.ssm_beta.weight",l);   chk2(nm, H, TSR);
            snprintf(nm,sizeof(nm),"blk.%d.ssm_a",l);             chk1(nm, TSR);
            snprintf(nm,sizeof(nm),"blk.%d.ssm_dt.bias",l);       chk1(nm, TSR);
            snprintf(nm,sizeof(nm),"blk.%d.ssm_conv1d.weight",l); chk2(nm, CK, conv_dim);
            snprintf(nm,sizeof(nm),"blk.%d.ssm_norm.weight",l);   chk1(nm, ST);
            snprintf(nm,sizeof(nm),"blk.%d.ssm_out.weight",l);    chk2(nm, INNER, H);
        }
    }
    // model-level tensors
    chk1("output_norm.weight", H);
    chk2("token_embd.weight", H, c->vocab_size);
    if (fetch("output.weight")) chk2("output.weight", H, c->vocab_size);  // untied lm_head

    int exp_full = 0;
    for (int i = 0; i < c->n_layers; i++) if (c->attn_kind[i] == GEMMA4_ATTN_FULL) exp_full++;
    fprintf(stderr, "fucina: qwen35 layers: %d FULL / %d LINEAR (expected %d FULL by (i+1)%%%d==0); "
            "shape errors=%d\n", nfull, nlin, exp_full, c->full_attention_interval, errs);
    if (nfull != exp_full || nfull + nlin != c->n_layers) {
        fprintf(stderr, "fucina: qwen35 layer-kind count mismatch\n"); errs++;
    }
    return errs == 0 ? 0 : -1;
}

// Compute the NVFP4-prefill weight footprint (bytes) WITHOUT building anything: the sum
// over (layer, projection) of the packed E2M1 weights [out_dim × in_dim/2] plus the
// swizzled E4M3 block scales [pad(out,128) × pad(in/16,4)]. Mirrors the per-projection
// sizing in build_fp4_weights (the d_fp4_w / d_fp4_wsc cudaMallocs) and the projection
// dimensions in proj_desc — kept self-contained because both live AFTER engine_create.
// Used by the --gpu-mem-util budget block to decide whether the lazy NVFP4-prefill copy
// (~= the model weight bytes, ~16.5 GiB on the 31B) can be afforded co-resident.
static uint64_t gemma4_fp4_prefill_footprint(const gemma4_engine_t *eng) {
    const int BLK = 16;  // NVFP4 block size (NVFP4_BLK, defined later)
    auto pad = [](int x, int m) { return ((x + m - 1) / m) * m; };
    uint64_t bytes = 0;
    int H = eng->cfg.hidden_size, I = eng->cfg.intermediate;
    for (int l = 0; l < eng->cfg.n_layers; l++) {
        layer_type_t lt = eng->layer_types[l];
        int hd  = (lt == LAYER_SLIDING) ? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM;
        int nkv = (lt == LAYER_SLIDING) ? eng->cfg.n_kv_sliding : eng->cfg.n_kv_global;
        int oq  = eng->cfg.n_heads * hd;
        int okv = nkv * hd;
        for (int p = 0; p < PJ_COUNT; p++) {
            int in_dim, out_dim;
            switch (p) {
                case PJ_Q:    in_dim = H;  out_dim = oq;  break;
                case PJ_K:    in_dim = H;  out_dim = okv; break;
                case PJ_V:    if (lt != LAYER_SLIDING) continue;  // global: V=K, no weight
                              in_dim = H;  out_dim = okv; break;
                case PJ_O:    in_dim = oq; out_dim = H;   break;
                case PJ_GATE: in_dim = H;  out_dim = I;   break;
                case PJ_UP:   in_dim = H;  out_dim = I;   break;
                case PJ_DOWN: in_dim = I;  out_dim = H;   break;
                default:      continue;
            }
            uint64_t packed = (uint64_t)out_dim * (in_dim / 2);
            uint64_t swsz   = (uint64_t)pad(out_dim, 128) * pad(in_dim / BLK, 4);
            bytes += packed + swsz;
        }
    }
    return bytes;
}

gemma4_engine_t* gemma4_engine_create(
    const char    *model_path,
    tensor_format_t format,
    uint32_t       context_size,
    int            device_id,
    double         gpu_mem_util)
{
    // Allocate engine
    gemma4_engine_t *eng = (gemma4_engine_t *)
        calloc(1, sizeof(gemma4_engine_t));
    if (!eng) return NULL;

    eng->format = format;
    eng->context_size = context_size;
    eng->device_id = device_id;
    // GPU-memory budget fraction. Clamp to a sane (0,1]; <=0 or absurd → the 0.90 default.
    if (gpu_mem_util <= 0.0 || gpu_mem_util > 1.0) gpu_mem_util = 0.90;
    eng->gpu_mem_util = gpu_mem_util;
    eng->loaded = 0;

    // Validate context
    if (context_size > GEMMA4_MAX_CTX) {
        fprintf(stderr, "fucina: context size %u exceeds max %u\n",
                context_size, GEMMA4_MAX_CTX);
        free(eng);
        return NULL;
    }

    // M0: the model architecture is auto-detected from the checkpoint's own metadata
    // (GGUF kv / safetensors config.json) into eng->cfg below, right after the file is
    // opened. Seed cfg with the Gemma-4-12B defaults so any read before detection is sane
    // and the 12B path is behavior-identical even if a checkpoint omits a kv. The derived
    // per-layer attention arrays (layer_types / global_layer_indices / n_*) are filled from
    // eng->cfg by gemma4_apply_cfg() once detection runs.
    {
        gemma4_model_config_t *c = &eng->cfg;
        memset(c, 0, sizeof(*c));
        c->n_layers          = GEMMA4_MAX_LAYERS;     // 48
        c->hidden_size       = GEMMA4_HIDDEN_SIZE;    // 3840
        c->intermediate      = GEMMA4_INTERMEDIATE;   // 15360
        c->n_heads           = GEMMA4_HEADS;          // 16
        c->n_kv_sliding      = GEMMA4_KV_HEADS;       // 8
        c->n_kv_global       = GEMMA4_GLOBAL_KV_HEADS;// 1
        c->vocab_size        = GEMMA4_VOCAB_SIZE;     // 262144
        c->softcap           = GEMMA4_SOFTCAP;        // 30
        c->rope_theta_global = 1000000.0f;
        c->rope_theta_sliding= 10000.0f;
        c->n_global = 0;
        for (int i = 0; i < GEMMA4_MAX_LAYERS; i++) {
            int is_global = ((i % 6) == 5);           // 5 sliding + 1 global
            c->is_global[i] = (uint8_t)is_global;
            if (is_global) c->n_global++;
        }
    }
    gemma4_apply_cfg(eng);   // derive layer_types / global_layer_indices / n_* from eng->cfg

    // CUDA setup
    cudaSetDevice(device_id);
    cudaStreamCreate(&eng->stream);
    // Engine-resident timing events (DECODE-30-35 Step 4): reused every decode instead of
    // per-token cudaEventCreate/Destroy. Also fixes a real per-token event LEAK in
    // gemma4_engine_decode (it created start/stop but never destroyed them).
    eng->ev_start = NULL; eng->ev_stop = NULL;
    cudaEventCreate(&eng->ev_start);
    cudaEventCreate(&eng->ev_stop);

    // Weight-dequant overlap stream + pipeline events (see d_bf16_layer comment). These
    // are pure ordering events (cudaEventDisableTiming keeps cross-stream waits cheap).
    // The BF16 prefill path (N > MMQ_MAX_N) always pipelines dequant; small/mid Q4_0
    // batches skip dequant entirely via the tiled-MMQ path.
    eng->dq_stream = NULL;
    cudaStreamCreate(&eng->dq_stream);
    for (int b = 0; b < 2; b++) {
        eng->ev_dq_done[b] = NULL; eng->ev_gemm_done[b] = NULL;
        cudaEventCreateWithFlags(&eng->ev_dq_done[b], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&eng->ev_gemm_done[b], cudaEventDisableTiming);
    }

    // Get memory info
    cudaMemGetInfo(&eng->free_mem, &eng->total_mem);

    // Create cuBLAS handle and bind to our stream
    cublasCreate(&eng->cublas);
    cublasSetStream(eng->cublas, eng->stream);
    {
        cublasSetMathMode(eng->cublas, CUBLAS_TENSOR_OP_MATH);
    }

    // CUDA Graph + persistent scratch init
    eng->graph_mode = 0; eng->graph.g = NULL; eng->graph.e = NULL;
    eng->graph.N = 0; eng->graph.hits = eng->graph.misses = 0;
    eng->ev_pending = 0; eng->ev_pending_tokens = 0;
    eng->d_specxt = NULL;
    eng->decode_graph = NULL; eng->decode_graph_failed = 0;
    eng->d_decpos = NULL; eng->d_dectok = NULL;
    for (int k = 0; k <= GEMMA4_SPEC_MAX; k++) eng->batched_graph[k] = NULL;
    eng->batched_graph_failed = 0; eng->d_specpos = NULL;
    for (int b = 0; b <= GEMMA4_MAX_SEQS; b++) eng->multiseq_graph[b] = NULL;
    eng->multiseq_graph_failed = 0; eng->multiseq_graph_logged = 0;
    // qwen35 M4 batched-decode arenas (lazy; allocated on first qwen35 seq_add).
    eng->q35.ready = 0; eng->q35.capacity = requested_seq_capacity();
    eng->q35.maxctx = 0; eng->q35.reserved_context = 0; eng->q35.graph_enabled = 1;
    // S2a GPU input-splicing default-on; FUCINA_QWEN35_NO_GPU_SPLICE=1 forces the host-copy path.
    eng->q35.gpu_splice_enabled = getenv("FUCINA_QWEN35_NO_GPU_SPLICE") ? 0 : 1;
    // S1a DFlash feature gate (default OFF). Parsed via the shared planner mapping so runtime and
    // tests agree; OFF keeps every path byte-identical to plain decode. Critical batch stays at the
    // conservative default (0) until a GB10 sweep measures the real crossover.
    eng->q35.dflash_mode = q35_dflash_mode_from_env(getenv("FUCINA_QWEN35_DFLASH"));
    eng->q35.dflash_critical_batch = 0;
    eng->q35.d_slot_tok = NULL; eng->q35.d_slot_pos = NULL;
    eng->q35.rowslot = NULL; eng->q35.chunk_scr = NULL; eng->q35.d_pf_seqmeta = NULL;
    eng->q35.pf_pos = NULL; eng->q35.pf_tok = NULL;
    eng->q35.attn_splits = 0; eng->q35.attn_tile = 0;
    eng->q35.part_m = NULL; eng->q35.part_l = NULL; eng->q35.part_o = NULL;
    eng->q35.wbf16[0] = NULL; eng->q35.wbf16[1] = NULL; eng->q35.xbf16 = NULL;
    eng->q35.qb = NULL; eng->q35.kb = NULL; eng->q35.vb = NULL;
    eng->q35.kbx = NULL; eng->q35.vbx = NULL; eng->q35.pb = NULL;
    eng->q35.scores = NULL; eng->q35.attn_cap = 0; eng->q35.wcache_on = 0;
    eng->q35.qh = NULL; eng->q35.qh_cap = 0;
    eng->q35.cont_scores = NULL; eng->q35.cont_p = NULL; eng->q35.cont_cap = 0;
    eng->q35.fp4_gsw = NULL; eng->q35.fp4_on = 0;
    eng->q35.fp8_scale_tab = NULL; eng->q35.fp8_scale_n = 0;
    eng->q35.model_bytes = eng->q35.workspace_bytes = eng->q35.per_slot_recurrent_bytes = 0;
    eng->q35.per_slot_kv_bytes = eng->q35.reserved_slot_kv_bytes = 0;
    eng->q35.committed_bytes = eng->q35.reserved_bytes = eng->q35.peak_bytes = 0;
    eng->q35.prefill_timing = getenv("FUCINA_QWEN35_PREFILL_TIMINGS") ? 1 : 0;
    eng->q35.prefill_dequant_ms = eng->q35.prefill_router_ms = 0;
    eng->q35.prefill_expert_ms = eng->q35.prefill_shared_ms = 0;
    eng->q35.jspace_enabled = eng->q35.jspace_topk = eng->q35.jspace_nlayers = 0;
    eng->q35.jspace_hidden = eng->q35.jspace_transport = eng->q35.jspace_norm = NULL;
    eng->q35.jspace_logits = eng->q35.jspace_dirs = NULL;
    eng->q35.jspace_steer_mask = NULL; eng->q35.jspace_steer_strength = NULL;
    eng->q35.jspace_bytes = 0;
    for (int l=0; l<GEMMA4_CAP_LAYERS; l++) {
        eng->q35.jspace_layer_enabled[l]=0; eng->q35.jspace_J[l]=NULL;
    }
    for (int l = 0; l < GEMMA4_CAP_LAYERS; l++)
        for (int p = 0; p < 12; p++) {
            eng->q35.wc[l][p] = NULL;
            eng->q35.fp4_w[l][p] = NULL; eng->q35.fp4_wsc[l][p] = NULL;
        }
    eng->q35.graph_failed = 0; eng->q35.graph_logged = 0;
    eng->q35.graph_count = 0;
    for (int i = 0; i < Q35_GRAPH_CACHE_CAP; i++) {
        eng->q35.graph_cache[i].exec = NULL;
        eng->q35.graph_cache[i].key = q35_graph_key{0, 0, 0};
    }
    eng->q35.allocated_slots = 0;
    for (int s = 0; s < GEMMA4_MAX_SEQS; s++) {
        eng->q35.recurrent_slab[s] = NULL;
        eng->q35.slot_allocated[s] = 0; eng->q35.kv_capacity[s] = 0;
        eng->q35.gdn_snap_slab[s] = NULL; eng->q35.gdn_snap_ntokens[s] = -1;
    }
    for (int l = 0; l < GEMMA4_CAP_LAYERS; l++) {
        eng->q35.S[l] = NULL; eng->q35.ring[l] = NULL;
        eng->q35.Kc[l] = NULL; eng->q35.Vc[l] = NULL;
        for (int s = 0; s < GEMMA4_MAX_SEQS; s++) {
            eng->q35.S_slot[l][s] = NULL; eng->q35.ring_slot[l][s] = NULL;
            eng->q35.Kc_slot[l][s] = NULL; eng->q35.Vc_slot[l][s] = NULL;
        }
    }
    for (int i = 0; i < 24; i++) eng->q35.sb[i] = NULL;
    eng->spec_steps = eng->spec_drafted = eng->spec_accepted = eng->spec_emitted = 0;
    eng->pf_scratch_ready = 0;
    eng->d_pf_x = eng->d_pf_norm = eng->d_pf_q = eng->d_pf_k = eng->d_pf_v = NULL;
    eng->d_pf_attn = eng->d_pf_gate = eng->d_pf_up = eng->d_pf_scores = NULL;
    eng->d_pf_inb = eng->d_pf_qb = eng->d_pf_kb = eng->d_pf_vb = NULL;
    eng->d_pf_kbx = eng->d_pf_vbx = eng->d_pf_pb = NULL;
    eng->d_pf_wpos = NULL; eng->d_pf_views_slid = NULL; eng->d_pf_views_glob = NULL;

    // Format autodetect. The NVFP4 checkpoint is commonly passed as a DIRECTORY (or an
    // .index.json / single .safetensors), which cannot be mmap'd here — st::Model::open handles
    // sharded layouts itself. So: stat first. A directory → NVFP4 (no engine mmap). A regular
    // file → mmap and read the 4-byte magic: 'GGUF' (0x46554747 LE) takes the GGUF flow; anything
    // else is a single-file NVFP4 safetensors checkpoint (its header is a u64 LE length, so byte 0
    // is never 'G'). For NVFP4 the engine's own mmap is redundant (st::Model::open re-maps it) and
    // is released below to avoid a transient ~10 GB double-map on RAM-constrained boxes.
    bool is_nvfp4 = false;
    eng->gguf_fd = -1;
    eng->gguf_data = NULL;
    eng->gguf_size = 0;
    {
        struct stat pst;
        if (stat(model_path, &pst) != 0) {
            perror("fucina: stat model");
            gemma4_engine_destroy(eng);
            return NULL;
        }
        if (S_ISDIR(pst.st_mode)) {
            is_nvfp4 = true;   // a sharded/dir checkpoint — st::Model::open resolves it
            fprintf(stderr, "fucina: loaded %s (directory)\n", model_path);
        } else {
            eng->gguf_fd = open(model_path, O_RDONLY);
            if (eng->gguf_fd < 0) {
                perror("fucina: open model");
                gemma4_engine_destroy(eng);
                return NULL;
            }
            struct stat st;
            fstat(eng->gguf_fd, &st);
            eng->gguf_size = st.st_size;
            eng->gguf_data = (const uint8_t *)mmap(
                NULL, eng->gguf_size, PROT_READ, MAP_PRIVATE, eng->gguf_fd, 0);
            if (eng->gguf_data == MAP_FAILED) {
                perror("fucina: mmap model");
                eng->gguf_data = NULL;
                gemma4_engine_destroy(eng);
                return NULL;
            }
            fprintf(stderr, "fucina: loaded %s (%.2f GB)\n",
                    model_path, eng->gguf_size / (1024.0 * 1024.0 * 1024.0));
            uint32_t magic = 0;
            if (eng->gguf_size >= 4) memcpy(&magic, eng->gguf_data, 4);
            is_nvfp4 = (magic != 0x46554747u);
        }
    }

    // NVFP4 layout (opened in the NVFP4 branch below; declared here so the redundant-mmap release
    // and the residency/BF16 load can all see it).
    st::Model nvfp4_model;
    nvfp4ld::Layout nvfp4_layout;
    // Qwen3.5 block-FP8: opened here (function scope) so the late weight-fill in the residency block
    // can still see the mmap'd tensors after the cfg is applied.
    st::Model fp8_model;
    qwen35fp8::Layout fp8_layout;

    int embd_is_q6k = 0;       // GGUF QAT-only (Q6_K token_embd → Q8_0)
    int embd_is_q4k = 0;       // Unsloth UD (Q4_K token_embd → Q8_0; native-Q4_K head enabled)
    uint32_t embd_src_type = 0; // token_embd GGML type (Q6_K or Q4_K → both become Q8_0)

    // ── Qwen3.5 block-FP8 checkpoint? (model_type qwen3_5 + quant_method fp8). Serve it through the
    // real batched engine — fill d_weights/tensors with FORMAT_FP8_BLOCK + the scale table — instead
    // of the B=1 qwen35_fp8_forward_greedy oracle. is_fp8 then skips every NVFP4/GGUF weight block.
    // The path may be an HF-cache root (models--Org--Name/ with snapshots/<hash>/) — resolve first. ──
    bool is_fp8 = false;
    if (is_nvfp4) {
        std::string ferr;
        std::string fp8_dir = q35moe_resolve_dir(model_path);
        if (fp8_model.open(fp8_dir.c_str(), ferr) && qwen35fp8::detect(fp8_model, fp8_layout, ferr)) {
            if (qwen35_fp8_setup_cfg(eng, fp8_model, fp8_layout) != 0) {
                fprintf(stderr, "fucina: Qwen3.5 FP8 cfg setup failed\n");
                gemma4_engine_destroy(eng); return NULL;
            }
            gemma4_apply_cfg(eng);   // derive layer_types / global_layer_indices / n_* from cfg
            is_fp8 = true;
            fprintf(stderr, "fucina: Qwen3.5 %s checkpoint detected → batched engine\n",
                    fp8_layout.compressed ? "compressed-tensors NVFP4/FP8" :
                    (fp8_layout.modelopt ? "ModelOpt NVFP4/FP8" : "block-FP8"));
        }
    }
    if (is_nvfp4 && !is_fp8) {
        std::string err;
        if (!nvfp4_model.open(model_path, err)) {
            fprintf(stderr, "fucina: NVFP4 open '%s' failed: %s\n", model_path, err.c_str());
            gemma4_engine_destroy(eng);
            return NULL;
        }
        if (!nvfp4ld::detect(nvfp4_model, nvfp4_layout, err)) {
            fprintf(stderr, "fucina: NVFP4 detect failed: %s\n", err.c_str());
            gemma4_engine_destroy(eng);
            return NULL;
        }
        // M0: auto-detect the architecture from the safetensors config.json (the checkpoint's
        // own metadata) into eng->cfg, then derive the per-layer pattern from it. Replaces the
        // old hard assertion that the model be exactly 48 layers (12B). config.json carries
        // num_hidden_layers / hidden_size / intermediate_size / num_attention_heads /
        // num_key_value_heads / sliding_window_pattern / final_logit_softcapping / rope_theta.
        {
            char derr[256] = {0};
            const std::string &cj = nvfp4_model.config_json();
            if (cj.empty() ||
                gemma4_detect_from_config_json(cj.c_str(), &eng->cfg, derr, sizeof(derr)) != 0) {
                fprintf(stderr, "fucina: NVFP4 config.json arch auto-detect failed: %s "
                        "(falling back to 12B defaults; %d-layer layout)\n",
                        cj.empty() ? "config.json missing" : derr, nvfp4_layout.n_layers);
                // eng->cfg holds the 12B defaults; ensure n_layers tracks the safetensors layout.
                eng->cfg.n_layers = nvfp4_layout.n_layers;
            }
            // The safetensors residency loop is sized by nvfp4_layout.n_layers; keep cfg in sync
            // so the two agree (config.json num_hidden_layers should equal the tensor layout).
            if (eng->cfg.n_layers != nvfp4_layout.n_layers) {
                fprintf(stderr, "fucina: NVFP4 config.json n_layers %d != tensor layout %d; "
                        "using tensor layout\n", eng->cfg.n_layers, nvfp4_layout.n_layers);
                eng->cfg.n_layers = nvfp4_layout.n_layers;
            }
            if (eng->cfg.n_layers > GEMMA4_CAP_LAYERS) {
                fprintf(stderr, "fucina: NVFP4 model has %d layers, exceeds capacity %d\n",
                        eng->cfg.n_layers, GEMMA4_CAP_LAYERS);
                gemma4_engine_destroy(eng);
                return NULL;
            }
            gemma4_apply_cfg(eng);   // layer_types / global_layer_indices / n_* from cfg
        }
        eng->format = FORMAT_NVFP4;   // override the placeholder hint passed from Go
        // Release the redundant create-time mmap of the checkpoint (NVFP4 sources weights from
        // the st::Model below, not the raw mmap).
        if (eng->gguf_data && eng->gguf_data != MAP_FAILED)
            munmap((void *)eng->gguf_data, eng->gguf_size);
        eng->gguf_data = NULL;
        fprintf(stderr, "fucina: NVFP4 safetensors checkpoint detected (%d layers, %s)\n",
                nvfp4_layout.n_layers,
                nvfp4_layout.naming == nvfp4ld::Naming::COMPRESSED ? "compressed-tensors" : "modelopt");
    }

    // Auto-detect the weight format from the GGUF tensor table. Trusting the CLI flag
    // is dangerous (decoding Q8_0 blocks as FP8 bytes yields NaNs). Detect from a LAYER
    // tensor (ffn_down) — token_embd may be a different type (Q6_K in the QAT Q4_0 model).
    if (!is_nvfp4) {
        // Detect the ENGINE (bulk) format from attn_q — it is the dominant Q4_0/Q8_0 type in
        // every supported export. The Unsloth UD dynamic-quant GGUF mixes a few off-format
        // tensors (Q4_1 ffn_down, Q4_K token_embd); those are handled by per-tensor requant
        // below, so they must NOT drive the engine format or trip the unsupported-type guard.
        uint64_t _off = 0, _n = 0; uint32_t gtype = 0;
        const char *names[] = {"blk.0.attn_q.weight", "blk.0.ffn_gate.weight"};
        int got_fmt = 0;
        for (int t = 0; t < 2 && !got_fmt; t++) {
            if (gguf_find_tensor(eng->gguf_data, eng->gguf_size, names[t], &_off, &_n, &gtype) == 0) {
                int bfmt = ggml_to_fmt(gtype);
                if (bfmt < 0) {
                    fprintf(stderr, "fucina: unsupported GGUF bulk tensor type %u — "
                            "only Q4_0 (QAT), Q8_0, Q4_K and Q6_K models are supported\n", gtype);
                    gemma4_engine_destroy(eng);
                    return NULL;
                }
                eng->format = (tensor_format_t)bfmt;
                got_fmt = 1;
            }
        }
        // token_embd / tied LM head type. QAT Q4_0 ships Q6_K; Unsloth UD ships Q4_K. Both get
        // dequantized + requantized to Q8_0 at load so the Q8_0 embed/head kernels handle them.
        if (gguf_find_tensor(eng->gguf_data, eng->gguf_size,
                "token_embd.weight", &_off, &_n, &embd_src_type) == 0) {
            if (embd_src_type == GGML_TYPE_Q6_K) {
                embd_is_q6k = 1;
                fprintf(stderr, "fucina: token_embd is Q6_K — will convert to Q8_0 at load\n");
            } else if (embd_src_type == GGML_TYPE_Q4_K) {
                embd_is_q4k = 1;
                fprintf(stderr, "fucina: token_embd is Q4_K — will convert to Q8_0 at load\n");
            } else if (embd_src_type != GGML_TYPE_Q4_0 && embd_src_type != GGML_TYPE_Q8_0) {
                fprintf(stderr, "fucina: WARNING token_embd type %u not directly supported "
                        "(expect Q6_K/Q4_K/Q4_0/Q8_0)\n", embd_src_type);
            }
        }
    }

    // ─── GGUF-only middle (tensor-offset table, attention pattern, rope_freqs, suppress
    // tokens, norm preload, bulk weight upload, token_embd convert). The NVFP4 path sources
    // all of this from the safetensors checkpoint in its own block below; everything past the
    // device-scratch alloc (KV cache, global-slot map, smem opt-in) is format-independent. ───
    if (!is_nvfp4) {
    // M0: auto-detect the full architecture (layer count, hidden/FFN sizes, head counts,
    // per-layer sliding/global pattern, softcap, rope thetas) from the GGUF's own kv metadata
    // into eng->cfg, then derive the engine's per-layer bookkeeping from it. This replaces the
    // old fixed 5-sliding/1-global default + standalone pattern parse: the same metadata
    // (gemma4.attention.sliding_window_pattern / head_count_kv) now flows through cfg.is_global[].
    // On the 12B GGUF this reproduces the previous 48/8-global/16-head geometry exactly.
    {
        char derr[256] = {0};
        if (gemma4_detect_from_gguf(eng->gguf_data, eng->gguf_size, &eng->cfg,
                                    derr, sizeof(derr)) != 0) {
            fprintf(stderr, "fucina: GGUF arch auto-detect failed: %s "
                            "(falling back to 12B defaults)\n", derr);
            // eng->cfg already holds the 12B defaults seeded at create top.
        } else if (eng->cfg.arch == GEMMA4_ARCH_QWEN3_5) {
            fprintf(stderr, "fucina: detected Qwen3.5 hybrid arch: %d layers (%d FULL softmax-GQA / "
                    "%d LINEAR gated-deltanet, full every %d), hidden %d, FFN %d, %d heads, %d KV "
                    "heads, head_dim %d, rotary_dim %d, vocab %d, rope_theta %.0f\n",
                    eng->cfg.n_layers, eng->cfg.n_full, eng->cfg.n_layers - eng->cfg.n_full,
                    eng->cfg.full_attention_interval, eng->cfg.hidden_size, eng->cfg.intermediate,
                    eng->cfg.n_heads, eng->cfg.n_kv_global, eng->cfg.head_dim, eng->cfg.rotary_dim,
                    eng->cfg.vocab_size, eng->cfg.rope_theta_global);
        } else {
            fprintf(stderr, "fucina: detected Gemma-4 arch: %d layers, hidden %d, FFN %d, "
                    "%d heads, KV %d sliding/%d global, %d global layers, softcap %.1f\n",
                    eng->cfg.n_layers, eng->cfg.hidden_size, eng->cfg.intermediate,
                    eng->cfg.n_heads, eng->cfg.n_kv_sliding, eng->cfg.n_kv_global,
                    eng->cfg.n_global, eng->cfg.softcap);
        }
        gemma4_apply_cfg(eng);   // layer_types / global_layer_indices / n_* from cfg.is_global[]
    }

    // Parse tensor offsets via gguf_find_tensor_offset.
    // A missing tensor leaves the field at its calloc'd 0 — but offset 0 is a
    // VALID offset (the first tensor's data), so a silent miss runs that layer
    // against the wrong bytes (garbage output, no error). LOAD_TENSOR_OFFSET
    // warns on a miss; LOAD_TENSOR_REQUIRED additionally marks the load as failed.
    // Only the model-defining tensors that EVERY valid gemma-4 GGUF must carry are
    // marked required — the per-layer loop runs to GEMMA4_MAX_LAYERS and a model
    // with fewer layers legitimately lacks the upper-layer tensors, so those stay
    // warnings (they read offset 0 but are never reached for absent layers).
    bool missing_required = false;
    #define LOAD_TENSOR_OFFSET(name, field) do { \
        uint64_t _off, _n; \
        if (gguf_find_tensor_offset(eng->gguf_data, eng->gguf_size, \
                name, &_off, &_n) == 0) { \
            eng->tensors.field = _off; \
        } else { \
            fprintf(stderr, "fucina: tensor '%s' not found in GGUF\n", name); \
        } \
    } while(0)
    #define LOAD_TENSOR_REQUIRED(name, field) do { \
        uint64_t _off, _n; \
        if (gguf_find_tensor_offset(eng->gguf_data, eng->gguf_size, \
                name, &_off, &_n) == 0) { \
            eng->tensors.field = _off; \
        } else { \
            fprintf(stderr, "fucina: required tensor '%s' missing from GGUF\n", name); \
            missing_required = true; \
        } \
    } while(0)
    // Record a weight tensor's own GGML format (Qwen3 Q4_K_M mixes Q4_K/Q6_K per layer).
    #define LOAD_WT_FMT(nm, fld) do { \
        uint64_t _o2 = 0, _n2 = 0; uint32_t _t2 = 0; \
        if (gguf_find_tensor(eng->gguf_data, eng->gguf_size, nm, &_o2, &_n2, &_t2) == 0) { \
            int _f2 = ggml_to_fmt(_t2); \
            if (_f2 < 0) { fprintf(stderr, "fucina: unsupported weight type %u for %s\n", _t2, nm); \
                           missing_required = true; } \
            eng->tensors.fld = (uint8_t)_f2; \
        } \
    } while(0)

    // Load embedding (required — also the tied LM head)
    LOAD_TENSOR_REQUIRED("token_embd.weight", token_embd);

    // Load per-layer tensors (M0: loop over the detected layer count, not the 12B max).
    char tname[128];
    for (int l = 0; l < eng->cfg.n_layers; l++) {
        // ── Qwen3.5 hybrid (M1): the per-layer tensor SET depends on the layer KIND. ──
        // FULL (softmax-GQA) layers ship the Qwen3-style attn_{q,k,v,output,q_norm,k_norm}
        // (attn_q is the 2×-wide [query|gate] proj); LINEAR (gated-deltanet) layers ship
        // attn_qkv/attn_gate + ssm_{alpha,beta,a,dt.bias,conv1d,norm,out}. BOTH kinds share
        // attn_norm (pre-mixer) + the pre-FFN RMSNorm (named "post_attention_norm" in this
        // GGUF → loaded into the ffn_norm slot) + the dense SwiGLU FFN. Gated on QWEN3_5 so
        // every other arch falls through to the original loop body byte-for-byte unchanged.
        if (eng->cfg.arch == GEMMA4_ARCH_QWEN3_5) {
            // shared: pre-mixer norm
            snprintf(tname, sizeof(tname), "blk.%d.attn_norm.weight", l);
            LOAD_TENSOR_OFFSET(tname, layers[l].attn_norm);
            // shared: pre-FFN RMSNorm (this GGUF names it post_attention_norm) → ffn_norm slot
            snprintf(tname, sizeof(tname), "blk.%d.post_attention_norm.weight", l);
            LOAD_TENSOR_OFFSET(tname, layers[l].ffn_norm);
            // shared: dense SwiGLU FFN
            snprintf(tname, sizeof(tname), "blk.%d.ffn_gate.weight", l);
            LOAD_TENSOR_OFFSET(tname, layers[l].ffn_gate); LOAD_WT_FMT(tname, layers[l].fmt_gate);
            snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", l);
            LOAD_TENSOR_OFFSET(tname, layers[l].ffn_up);   LOAD_WT_FMT(tname, layers[l].fmt_up);
            snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", l);
            LOAD_TENSOR_OFFSET(tname, layers[l].ffn_down); LOAD_WT_FMT(tname, layers[l].fmt_down);

            if (eng->cfg.attn_kind[l] == GEMMA4_ATTN_FULL) {
                snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", l);
                LOAD_TENSOR_OFFSET(tname, layers[l].attn_q);      LOAD_WT_FMT(tname, layers[l].fmt_q);
                snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", l);
                LOAD_TENSOR_OFFSET(tname, layers[l].attn_k);      LOAD_WT_FMT(tname, layers[l].fmt_k);
                snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", l);
                LOAD_TENSOR_OFFSET(tname, layers[l].attn_v);      LOAD_WT_FMT(tname, layers[l].fmt_v);
                snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", l);
                LOAD_TENSOR_OFFSET(tname, layers[l].attn_output); LOAD_WT_FMT(tname, layers[l].fmt_o);
                snprintf(tname, sizeof(tname), "blk.%d.attn_q_norm.weight", l);
                LOAD_TENSOR_OFFSET(tname, layers[l].attn_q_norm);
                snprintf(tname, sizeof(tname), "blk.%d.attn_k_norm.weight", l);
                LOAD_TENSOR_OFFSET(tname, layers[l].attn_k_norm);
            } else {
                // LINEAR (gated-deltanet) tensor set
                snprintf(tname, sizeof(tname), "blk.%d.attn_qkv.weight", l);
                LOAD_TENSOR_OFFSET(tname, layers[l].ssm.in_qkv);  LOAD_WT_FMT(tname, layers[l].ssm.fmt_in_qkv);
                snprintf(tname, sizeof(tname), "blk.%d.attn_gate.weight", l);
                LOAD_TENSOR_OFFSET(tname, layers[l].ssm.in_z);    LOAD_WT_FMT(tname, layers[l].ssm.fmt_in_z);
                snprintf(tname, sizeof(tname), "blk.%d.ssm_alpha.weight", l);
                LOAD_TENSOR_OFFSET(tname, layers[l].ssm.in_a);    LOAD_WT_FMT(tname, layers[l].ssm.fmt_in_a);
                snprintf(tname, sizeof(tname), "blk.%d.ssm_beta.weight", l);
                LOAD_TENSOR_OFFSET(tname, layers[l].ssm.in_b);    LOAD_WT_FMT(tname, layers[l].ssm.fmt_in_b);
                snprintf(tname, sizeof(tname), "blk.%d.ssm_a", l);
                LOAD_TENSOR_OFFSET(tname, layers[l].ssm.a_log);
                snprintf(tname, sizeof(tname), "blk.%d.ssm_dt.bias", l);
                LOAD_TENSOR_OFFSET(tname, layers[l].ssm.dt_bias);
                snprintf(tname, sizeof(tname), "blk.%d.ssm_conv1d.weight", l);
                LOAD_TENSOR_OFFSET(tname, layers[l].ssm.conv1d);
                snprintf(tname, sizeof(tname), "blk.%d.ssm_norm.weight", l);
                LOAD_TENSOR_OFFSET(tname, layers[l].ssm.norm);
                snprintf(tname, sizeof(tname), "blk.%d.ssm_out.weight", l);
                LOAD_TENSOR_OFFSET(tname, layers[l].ssm.out);     LOAD_WT_FMT(tname, layers[l].ssm.fmt_out);
            }
            continue;   // qwen35 layer fully loaded; skip the generic body below
        }

        snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].attn_q);
        LOAD_WT_FMT(tname, layers[l].fmt_q);

        snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].attn_k);
        LOAD_WT_FMT(tname, layers[l].fmt_k);

        // Gemma global layers use a unified K=V cache and ship no attn_v.weight. Qwen3 is
        // standard GQA: EVERY layer has a separate attn_v projection (even though all its layers
        // run through the engine's full-causal "global" class).
        if (eng->layer_types[l] == LAYER_SLIDING || 0) {
            snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", l);
            LOAD_TENSOR_OFFSET(tname, layers[l].attn_v);
            LOAD_WT_FMT(tname, layers[l].fmt_v);
        } else {
            eng->tensors.layers[l].attn_v = 0;
        }

        snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].attn_output);
        LOAD_WT_FMT(tname, layers[l].fmt_o);

        snprintf(tname, sizeof(tname), "blk.%d.attn_norm.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].attn_norm);

        snprintf(tname, sizeof(tname), "blk.%d.attn_q_norm.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].attn_q_norm);

        snprintf(tname, sizeof(tname), "blk.%d.attn_k_norm.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].attn_k_norm);

        snprintf(tname, sizeof(tname), "blk.%d.post_attention_norm.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].post_attn_norm);

        // Qwen3-MoE has NO dense FFN tensors (gate/up/down are 3D expert slabs loaded separately
        // below as moe_*); skip the dense names so they don't warn-spam "tensor not found".
        if (1) {
            snprintf(tname, sizeof(tname), "blk.%d.ffn_gate.weight", l);
            LOAD_TENSOR_OFFSET(tname, layers[l].ffn_gate);
            LOAD_WT_FMT(tname, layers[l].fmt_gate);

            snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", l);
            LOAD_TENSOR_OFFSET(tname, layers[l].ffn_up);
            LOAD_WT_FMT(tname, layers[l].fmt_up);

            snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", l);
            LOAD_TENSOR_OFFSET(tname, layers[l].ffn_down);
            LOAD_WT_FMT(tname, layers[l].fmt_down);
        }

        snprintf(tname, sizeof(tname), "blk.%d.ffn_norm.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].ffn_norm);

        snprintf(tname, sizeof(tname), "blk.%d.post_ffw_norm.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].post_ffn_norm);

        snprintf(tname, sizeof(tname), "blk.%d.layer_output_scale.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].layer_out_scale);
    }

    LOAD_TENSOR_REQUIRED("output_norm.weight", output_norm);

    // Output (LM head) projection. Gemma-4 ties the output projection to the
    // input embedding, so most GGUF exports do NOT contain a separate
    // "output.weight" tensor. Probe for an explicit head first; if absent,
    // alias it to token_embd.weight (tied embeddings).
    {
        uint64_t _off = 0, _n = 0; uint32_t _ot = 0;
        if (gguf_find_tensor(eng->gguf_data, eng->gguf_size,
                "output.weight", &_off, &_n, &_ot) == 0) {
            eng->tensors.output_weight = _off;
            eng->output_tied = 0;
            int ofmt = ggml_to_fmt(_ot);
            if (ofmt < 0) {
                fprintf(stderr, "fucina: unsupported output.weight type %u\n", _ot);
                missing_required = true;
            }
            eng->tensors.output_fmt = (uint8_t)ofmt;
        } else {
            eng->tensors.output_weight = eng->tensors.token_embd;
            eng->output_tied = 1;
            fprintf(stderr, "fucina: output.weight not present — using tied "
                            "token_embd.weight as LM head\n");
        }
    }

    #undef LOAD_TENSOR_OFFSET
    #undef LOAD_TENSOR_REQUIRED
    #undef LOAD_WT_FMT

    // Qwen3.5 hybrid (M1): dump the resolved per-layer tensor layout and validate every
    // shape against the arch spec. A missing or misshaped tensor flips missing_required so
    // the load aborts below rather than running the (M2+) forward against garbage bytes.
    if (eng->cfg.arch == GEMMA4_ARCH_QWEN3_5) {
        if (qwen35_dump_and_validate(eng) != 0) missing_required = true;
    }

    // A model-defining tensor was missing: the GGUF is corrupt or not a gemma-4
    // model. Fail the load rather than run inference against offset-0 garbage.
    // This is gemma4_engine_create, which returns the engine pointer — free it
    // and return NULL so the Go side sees a clean load failure.
    if (missing_required) {
        fprintf(stderr, "fucina: aborting load — required tensor(s) missing\n");
        free(eng);
        return NULL;
    }
    } // if (!is_nvfp4) — end GGUF-only middle

    // Allocate device memory
    // Scratch (1M floats = 4 MB)
    cudaMalloc(&eng->d_scratch,    4 * 1024 * 1024);
    cudaMalloc(&eng->d_logits,     GEMMA4_VOCAB_SIZE * sizeof(float));
    cudaMalloc(&eng->d_x,          eng->cfg.hidden_size * sizeof(float));
    // d_attn_q/k/v/out must be large enough for BOTH layer types (M0: runtime head counts).
    //   12B sliding: Q=16×256=4096  KV=8×256=2048 ; global Q=16×512=8192 KV=1×512=512
    //   31B sliding: Q=32×256=8192  KV=16×256=4096; global Q=32×512=16384 KV=4×512=2048
    int max_q    = eng->cfg.n_heads * GEMMA4_GLOBAL_HEAD_DIM;
    int max_kv   = eng->cfg.n_kv_sliding * GEMMA4_HEAD_DIM;
    // global KV (n_kv_global × global_head_dim) can exceed the sliding KV size on some configs;
    // take the larger so the K/V scratch holds both classes.
    { int g_kv = eng->cfg.n_kv_global * GEMMA4_GLOBAL_HEAD_DIM; if (g_kv > max_kv) max_kv = g_kv; }
    cudaMalloc(&eng->d_attn_q,     max_q  * sizeof(float));
    cudaMalloc(&eng->d_attn_k,     max_kv * sizeof(float));
    cudaMalloc(&eng->d_attn_v,     max_kv * sizeof(float));
    cudaMalloc(&eng->d_attn_out,   max_q  * sizeof(float));  // attention output, NOT hidden_size
    // GQA-broadcast global flash-decode split-K scratch (DECODE-30-35 Step 1).
    // Sized for GEMMA4_SPEC_MAX rows of GEMMA4_GLOBAL_MAX_SPLITS slots each (~64 MB):
    // the batched spec-verify attention runs all K rows' splits in ONE launch, row r
    // owning slot range [r*MAX_SPLITS, r*MAX_SPLITS + splits_r). Single-token paths
    // (decode_layer, MTP) keep using row-0's slot range unchanged.
    eng->d_fa_acc = NULL; eng->d_fa_m = NULL; eng->d_fa_l = NULL;
    cudaMalloc(&eng->d_fa_acc, (size_t)GEMMA4_SPEC_MAX * GEMMA4_GLOBAL_MAX_SPLITS
                                * eng->cfg.n_heads * GEMMA4_GLOBAL_HEAD_DIM * sizeof(float));
    cudaMalloc(&eng->d_fa_m,   (size_t)GEMMA4_SPEC_MAX * GEMMA4_GLOBAL_MAX_SPLITS
                                * eng->cfg.n_heads * sizeof(float));
    cudaMalloc(&eng->d_fa_l,   (size_t)GEMMA4_SPEC_MAX * GEMMA4_GLOBAL_MAX_SPLITS
                                * eng->cfg.n_heads * sizeof(float));
    cudaMalloc(&eng->d_ffn_out,    eng->cfg.intermediate * sizeof(float));
    cudaMalloc(&eng->d_ffn_gate,   eng->cfg.intermediate * sizeof(float));
    cudaMalloc(&eng->d_ffn_up,     eng->cfg.intermediate * sizeof(float));
    cudaMalloc(&eng->d_norm,       eng->cfg.hidden_size * sizeof(float));
    cudaMalloc(&eng->d_norm_w,     eng->cfg.hidden_size * sizeof(float));
    cudaMalloc(&eng->d_residual,   eng->cfg.hidden_size * sizeof(float));

    cudaMalloc(&eng->d_head_norm_w, GEMMA4_GLOBAL_HEAD_DIM * sizeof(float));

    // dp4a MMVQ activation-quant scratch (widest in_dim = intermediate).
    // M0: widest projection in_dim is the runtime FFN intermediate.
    int q_in = eng->cfg.intermediate;   // max dp4a GEMV in_dim: FFN I, GDN inner, attn o-proj, hidden
    if (eng->cfg.ssm_inner_size > q_in) q_in = eng->cfg.ssm_inner_size;
    if (eng->cfg.n_heads * eng->cfg.head_dim > q_in) q_in = eng->cfg.n_heads * eng->cfg.head_dim;
    if (eng->cfg.hidden_size > q_in) q_in = eng->cfg.hidden_size;
    eng->d_qx = NULL; eng->d_dx = NULL; eng->d_sx = NULL;
    cudaMalloc(&eng->d_qx, (size_t)q_in * sizeof(int8_t));
    cudaMalloc(&eng->d_dx, (size_t)(q_in/32) * sizeof(float));
    cudaMalloc(&eng->d_sx, (size_t)(q_in/32) * sizeof(int));   // Q4_0 −8 fold
    // Batched variant: NK activation vectors quantized at once. Sized for the WIDER of the
    // spec-decode cap (SPEC_MAX rows) and the qwen35 prefill tile (PF_TILE rows), since
    // gemv_batched_w quantizes K*in_dim into this scratch and prefill drives K up to PF_TILE.
    int q_rows = ((int)context_size < QWEN35_PF_TILE) ? (int)context_size : QWEN35_PF_TILE;
    if (q_rows < GEMMA4_SPEC_MAX) q_rows = GEMMA4_SPEC_MAX;
    eng->d_qx_b = NULL; eng->d_dx_b = NULL; eng->d_sx_b = NULL;
    cudaMalloc(&eng->d_qx_b, (size_t)q_rows * q_in * sizeof(int8_t));
    cudaMalloc(&eng->d_dx_b, (size_t)q_rows * (q_in/32) * sizeof(float));
    cudaMalloc(&eng->d_sx_b, (size_t)q_rows * (q_in/32) * sizeof(int));
    if (!eng->d_qx || !eng->d_dx || !eng->d_sx || !eng->d_qx_b || !eng->d_dx_b || !eng->d_sx_b) {
        fprintf(stderr, "fucina: dp4a activation scratch alloc failed\n");
        gemma4_engine_destroy(eng);
        return NULL;
    }
    eng->d_pf_qx = NULL; eng->d_pf_dx = NULL; eng->d_pf_sx = NULL; eng->mmq_ready = 0;
    // NVFP4 prefill (FUCINA_FP4) — all lazy, built on first prefill if enabled
    eng->cublaslt = NULL; eng->fp4_ready = 0; eng->fp4_budget_ok = 0; eng->fp4_desc = NULL;
    eng->d_fp4_gsw = NULL; eng->d_fp4_act = NULL; eng->d_fp4_actsc = NULL;
    eng->d_fp4_actlin = NULL; eng->d_fp4_gsact = NULL; eng->d_fp4_alpha = NULL;
    eng->d_fp4_amax = NULL; eng->fp4_act_cap = 0; eng->d_fp4_ws = NULL;
    eng->nvfp4_decode_ready = 0; eng->d_embed_bf16 = NULL; eng->d_lmhead_bf16 = NULL;
    eng->d_lmhead_fp8 = NULL; eng->d_lmhead_fp8_scale = NULL;
    for (int l=0;l<GEMMA4_CAP_LAYERS;l++) for (int p=0;p<PJ_COUNT;p++){
        eng->d_fp4_w[l][p]=NULL; eng->d_fp4_wsc[l][p]=NULL; eng->d_fp4_wsc_lin[l][p]=NULL; }

    // Load rope_freqs.weight into device buffer for global-layer RoPE.
    {
        uint64_t rf_off = 0, rf_n = 0;
        cudaMalloc(&eng->d_rope_freqs,
            GEMMA4_GLOBAL_HEAD_DIM / 2 * sizeof(float));
        if (is_nvfp4) {
            // safetensors has no rope_freqs.weight. Gemma-4's GLOBAL (full-attention) layers use
            // partial_rotary_factor=0.25 → only the first rotary_dim/2 = 64 of the 256 freq pairs
            // rotate; the rest pass through. The GGUF encodes this as freq_factors = 1.0 for the
            // first 64 entries and 1e30 (→ theta≈0 → identity) for the remaining 192. Replicate it
            // EXACTLY (verified against /home/mauromedda/hack/gem4d/model.gguf rope_freqs.weight).
            const int half = GEMMA4_GLOBAL_HEAD_DIM / 2;        // 256
            const int rot_half = half / 4;                      // 64 (partial_rotary_factor 0.25)
            float ff[GEMMA4_GLOBAL_HEAD_DIM / 2];
            for (int i = 0; i < half; i++) ff[i] = (i < rot_half) ? 1.0f : 1e30f;
            cudaMemcpy(eng->d_rope_freqs, ff, half * sizeof(float), cudaMemcpyHostToDevice);
        } else if (gguf_find_tensor_offset(eng->gguf_data, eng->gguf_size,
                "rope_freqs.weight", &rf_off, &rf_n) == 0) {
            // rope_freqs.weight shape=[256] F32; values are freq divisors.
            cudaMemcpy(eng->d_rope_freqs,
                eng->gguf_data + rf_off,
                (GEMMA4_GLOBAL_HEAD_DIM / 2) * sizeof(float),
                cudaMemcpyHostToDevice);
        } else {
            // No freq factors: fill with 1.0 (no-op divisor).
            float ones[GEMMA4_GLOBAL_HEAD_DIM / 2];
            for (int i = 0; i < GEMMA4_GLOBAL_HEAD_DIM / 2; i++) ones[i] = 1.0f;
            cudaMemcpy(eng->d_rope_freqs, ones,
                (GEMMA4_GLOBAL_HEAD_DIM / 2) * sizeof(float),
                cudaMemcpyHostToDevice);
        }
    }

    // Suppressed token ids → device list for the -inf logits mask (GGUF metadata only;
    // the NVFP4 safetensors checkpoint carries no suppress list → n_suppress stays 0).
    eng->d_suppress = NULL;
    eng->n_suppress = 0;
    if (!is_nvfp4) {
        gguf_array_t sarr;
        if (gguf_parse_metadata(eng->gguf_data, eng->gguf_size,
                "tokenizer.ggml.suppress_tokens", &sarr, GGUF_TYPE_ARRAY) == 0
            && (sarr.elem_type == GGUF_TYPE_INT32 || sarr.elem_type == GGUF_TYPE_UINT32)
            && sarr.count > 0)
        {
            eng->n_suppress = (int)sarr.count;
            cudaMalloc(&eng->d_suppress, sarr.count * sizeof(int32_t));
            cudaMemcpy(eng->d_suppress, sarr.data,
                       sarr.count * sizeof(int32_t), cudaMemcpyHostToDevice);
            fprintf(stderr, "fucina: %d suppressed tokens masked\n", eng->n_suppress);
        }
    }

    // ── Preload all F32 norm weights to device (once) ────────────────────
    {
        // M0: norm-store STRIDE is the runtime hidden size; the per-layer LOOP runs cfg.n_layers.
        // Allocate CAP_LAYERS slots (compile-time upper bound) so the device store is large enough
        // for any supported size; only the first cfg.n_layers are filled/read.
        const size_t hs = (size_t)eng->cfg.hidden_size, hd = GEMMA4_GLOBAL_HEAD_DIM;
        const int    nL = eng->cfg.n_layers;
        cudaMalloc(&eng->d_w_attn_norm,      GEMMA4_CAP_LAYERS * hs * sizeof(float));
        cudaMalloc(&eng->d_w_post_attn_norm, GEMMA4_CAP_LAYERS * hs * sizeof(float));
        cudaMalloc(&eng->d_w_ffn_norm,       GEMMA4_CAP_LAYERS * hs * sizeof(float));
        cudaMalloc(&eng->d_w_post_ffn_norm,  GEMMA4_CAP_LAYERS * hs * sizeof(float));
        cudaMalloc(&eng->d_w_q_norm,         GEMMA4_CAP_LAYERS * hd * sizeof(float));
        cudaMalloc(&eng->d_w_k_norm,         GEMMA4_CAP_LAYERS * hd * sizeof(float));
        cudaMalloc(&eng->d_w_out_norm,       hs * sizeof(float));

        // missing tensor (offset 0) → identity weight 1.0
        float *ones = (float *)malloc(hs * sizeof(float));
        for (size_t i = 0; i < hs; i++) ones[i] = 1.0f;

      if (!is_nvfp4) {
        #define UPLOAD_NORM(dst, off, n) do { \
            const void *src = (off) ? (const void *)(eng->gguf_data + (off)) : (const void *)ones; \
            cudaMemcpy((dst), src, (n) * sizeof(float), cudaMemcpyHostToDevice); \
        } while (0)

        for (int l = 0; l < nL; l++) {
            int head_dim = 0 ? eng->cfg.head_dim
                         : ((eng->layer_types[l] == LAYER_SLIDING)
                               ? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM);
            UPLOAD_NORM(eng->d_w_attn_norm      + l * hs, eng->tensors.layers[l].attn_norm,      hs);
            UPLOAD_NORM(eng->d_w_post_attn_norm + l * hs, eng->tensors.layers[l].post_attn_norm, hs);
            UPLOAD_NORM(eng->d_w_ffn_norm       + l * hs, eng->tensors.layers[l].ffn_norm,       hs);
            UPLOAD_NORM(eng->d_w_post_ffn_norm  + l * hs, eng->tensors.layers[l].post_ffn_norm,  hs);
            UPLOAD_NORM(eng->d_w_q_norm + l * hd, eng->tensors.layers[l].attn_q_norm, head_dim);
            UPLOAD_NORM(eng->d_w_k_norm + l * hd, eng->tensors.layers[l].attn_k_norm, head_dim);

            // layer_output_scale: single F32 scalar, read host-side once
            eng->h_out_scale[l] = 1.0f;
            if (eng->tensors.layers[l].layer_out_scale != 0)
                memcpy(&eng->h_out_scale[l],
                       eng->gguf_data + eng->tensors.layers[l].layer_out_scale,
                       sizeof(float));
        }
        UPLOAD_NORM(eng->d_w_out_norm, eng->tensors.output_norm, hs);
        #undef UPLOAD_NORM
      } else if (is_fp8) {
        // Qwen3.5 FP8: the qwen35 forward reads norms via Wf() from d_weights offsets (filled by
        // qwen35_fp8_fill_engine), NOT from this d_w_* store — leave it unused. h_out_scale=1.0
        // was set in setup_cfg.
      } else {
        // NVFP4: norms ship BF16 in the safetensors checkpoint. Upload each to a temp BF16
        // device buffer and convert to the FLOAT norm store via bf16_to_f32_kernel. A missing
        // tensor falls back to identity (1.0) — but a SILENT identity fallback on a real norm
        // produces fluent garbage, so every per-layer key is verified present below (and the
        // exact HF suffixes were dumped from the real checkpoint before wiring). q/k norm use
        // the head-dim stride (256) within the head-dim-wide (512) store slot. No per-layer
        // output scale exists in HF Gemma-4 NVFP4 → h_out_scale stays 1.0.
        const nvfp4ld::Layout &L = nvfp4_layout;
        st::Model &m = nvfp4_model;
        __nv_bfloat16 *d_tmp = nullptr;
        cudaMalloc(&d_tmp, hs * sizeof(__nv_bfloat16));   // widest norm = hidden
        auto upload_norm_bf16 = [&](float *dst, const std::string &key, int n) -> bool {
            const st::Tensor *t = m.find(key);
            if (!t || (int)(t->nbytes / sizeof(__nv_bfloat16)) < n) {
                // identity fallback (also covers a genuinely absent norm)
                cudaMemcpy(dst, ones, n * sizeof(float), cudaMemcpyHostToDevice);
                return false;
            }
            cudaMemcpy(d_tmp, t->data, (size_t)n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
            bf16_to_f32_kernel<<<(unsigned)((n + 255) / 256), 256>>>(dst, d_tmp, n);
            return true;
        };
        bool norm_ok = true;
        for (int l = 0; l < nL; l++) {
            int head_dim = (eng->layer_types[l] == LAYER_SLIDING)
                               ? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM;
            std::string pre = L.layer_prefix + std::to_string(l) + ".";
            norm_ok &= upload_norm_bf16(eng->d_w_attn_norm      + l * hs, pre + "input_layernorm.weight",          (int)hs);
            norm_ok &= upload_norm_bf16(eng->d_w_post_attn_norm + l * hs, pre + "post_attention_layernorm.weight", (int)hs);
            norm_ok &= upload_norm_bf16(eng->d_w_ffn_norm       + l * hs, pre + "pre_feedforward_layernorm.weight",(int)hs);
            norm_ok &= upload_norm_bf16(eng->d_w_post_ffn_norm  + l * hs, pre + "post_feedforward_layernorm.weight",(int)hs);
            norm_ok &= upload_norm_bf16(eng->d_w_q_norm + l * hd, pre + "self_attn.q_norm.weight", head_dim);
            norm_ok &= upload_norm_bf16(eng->d_w_k_norm + l * hd, pre + "self_attn.k_norm.weight", head_dim);
            // Per-layer output scalar (Gemma-4 "layer_scalar" — the GGUF's layer_output_scale).
            // It MULTIPLIES each layer's output (scale_kernel at the end of every layer); without
            // it the residual stream grows ~18× and the logits saturate the softcap → garbage.
            // BF16 [1] in the safetensors; default 1.0 only if genuinely absent.
            eng->h_out_scale[l] = 1.0f;
            {
                const st::Tensor *tls = m.find(pre + "layer_scalar");
                if (tls && tls->nbytes >= sizeof(__nv_bfloat16)) {
                    __nv_bfloat16 hb; memcpy(&hb, tls->data, sizeof(hb));
                    eng->h_out_scale[l] = __bfloat162float(hb);
                }
            }
        }
        norm_ok &= upload_norm_bf16(eng->d_w_out_norm, L.final_norm_key, (int)hs);
        cudaDeviceSynchronize();
        cudaFree(d_tmp);
        if (!norm_ok) {
            fprintf(stderr, "fucina: NVFP4 one or more norm tensors missing — "
                            "identity fallback would produce garbage; aborting\n");
            free(ones);
            gemma4_engine_destroy(eng);
            return NULL;
        }
      }
        free(ones);
    }

    eng->d_weights = NULL;
    eng->d_token_embd = NULL;
    eng->d_lmhead_q6k = NULL; eng->lmhead_q6k = 0;
    eng->d_lmhead_q4k = NULL; eng->lmhead_q4k = 0;
    eng->n_wt_override = 0;
  if (is_fp8) {
    // ── Qwen3.5 block-FP8 residency: fill d_weights/tensors/scale-table + Q8_0 embed/lm_head here,
    // AFTER create's unconditional d_weights=NULL reset above, so the fill survives. ─────────────
    if (qwen35_fp8_fill_engine(eng, fp8_model, fp8_layout) != 0) {
        fprintf(stderr, "fucina: Qwen3.5 FP8 weight fill failed\n");
        gemma4_engine_destroy(eng);
        return NULL;
    }
    eng->nvfp4_decode_ready = 0;   // FP8 uses the Q8_0/FP8-block decode path, not the BF16 NVFP4 one
  } else if (is_nvfp4) {
    // ── NVFP4 SINGLE-STORE residency + BF16 embed / LM head ──────────────
    // nvfp4_model's mmap MUST stay alive through nvfp4_load_from_safetensors (it copies the
    // tensor bytes H2D synchronously and ends with cudaDeviceSynchronize) and through the embed
    // /head uploads below. (A) Residency populates d_fp4_w / d_fp4_wsc / d_fp4_wsc_lin / d_fp4_gsw,
    // sets fp4_ready=1 and builds fp4_desc — prefill works immediately.
    if (nvfp4_load_from_safetensors(eng, model_path, &nvfp4_layout, &nvfp4_model) != 0) {
        fprintf(stderr, "fucina: NVFP4 residency failed\n");
        gemma4_engine_destroy(eng);
        return NULL;
    }
    // (B) BF16 embeddings [vocab × hidden], uploaded verbatim (Gemma stores the UNSCALED table;
    // the √hidden scale is applied post-lookup, identical to the GGUF path).
    {
        const st::Tensor *te = nvfp4_model.find(nvfp4_layout.embed_key);
        size_t want = (size_t)GEMMA4_VOCAB_SIZE * eng->cfg.hidden_size * sizeof(__nv_bfloat16);
        if (!te || te->nbytes != want) {
            fprintf(stderr, "fucina: NVFP4 embed '%s' missing or wrong size (%zu vs %zu)\n",
                    nvfp4_layout.embed_key.c_str(), te ? te->nbytes : 0, want);
            gemma4_engine_destroy(eng);
            return NULL;
        }
        if (cudaMalloc(&eng->d_embed_bf16, want) != cudaSuccess) {
            fprintf(stderr, "fucina: NVFP4 embed alloc failed\n");
            gemma4_engine_destroy(eng); return NULL;
        }
        cudaMemcpy(eng->d_embed_bf16, te->data, want, cudaMemcpyHostToDevice);
    }
    // (C) LM head: explicit lm_head.weight (untied) → its own buffer; else alias d_embed_bf16.
    if (!nvfp4_layout.lmhead_key.empty()) {
        const st::Tensor *th = nvfp4_model.find(nvfp4_layout.lmhead_key);
        size_t want = (size_t)GEMMA4_VOCAB_SIZE * eng->cfg.hidden_size * sizeof(__nv_bfloat16);
        if (!th || th->nbytes != want) {
            fprintf(stderr, "fucina: NVFP4 lm_head '%s' missing or wrong size\n", nvfp4_layout.lmhead_key.c_str());
            gemma4_engine_destroy(eng); return NULL;
        }
        if (cudaMalloc(&eng->d_lmhead_bf16, want) != cudaSuccess) {
            fprintf(stderr, "fucina: NVFP4 lm_head alloc failed\n");
            gemma4_engine_destroy(eng); return NULL;
        }
        cudaMemcpy(eng->d_lmhead_bf16, th->data, want, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        // FP8 E4M3 PER-ROW HEAD: the untied BF16 head is 2 GB read EVERY token. Quantizing it to 1
        // B/elem per-row would halve the head read, but FP8 (3 mantissa bits) flips the argmax over
        // a 262144-vocab head and DEGRADES real generation ("capital of France is France") even when
        // a load-time argmax gate on PROXY hidden states passes — the proxies don't reflect the real
        // decode distribution and errors compound. So the FP8 head is OPT-IN (FUCINA_FP8_HEAD=1) and
        // OFF by default; the BF16 head is correctness-critical and stays resident. The gate below is
        // kept for experimentation but never auto-enables.
        if (getenv("FUCINA_FP8_HEAD")) {
            const int H = eng->cfg.hidden_size, V = GEMMA4_VOCAB_SIZE;
            uint8_t *dq = NULL; float *dsc = NULL;
            if (cudaMalloc(&dq,(size_t)V*H) == cudaSuccess &&
                cudaMalloc(&dsc,(size_t)V*sizeof(float)) == cudaSuccess) {
                fp8_head_quantize_launch(dq, dsc, eng->d_lmhead_bf16, V, H, 0);
                cudaDeviceSynchronize();
                eng->d_lmhead_fp8 = dq; eng->d_lmhead_fp8_scale = dsc;
                double worst_l2 = 0;
                const int nsamp = 64;
                int match = fp8_head_accuracy_gate(eng, nsamp, &worst_l2);
                if (match == nsamp) {
                    // Gate passed: free the BF16 head (distinct alloc — untied, never aliases embed).
                    cudaFree(eng->d_lmhead_bf16); eng->d_lmhead_bf16 = NULL;
                    fprintf(stderr, "fucina: NVFP4 FP8 head ENABLED — argmax %d/%d match, worst logit L2rel=%.4f%%, head read 2.0→1.0 GB/token\n",
                            match, nsamp, 100*worst_l2);
                } else {
                    cudaFree(dq); cudaFree(dsc);
                    eng->d_lmhead_fp8 = NULL; eng->d_lmhead_fp8_scale = NULL;
                    fprintf(stderr, "fucina: NVFP4 FP8 head REJECTED by accuracy gate (argmax %d/%d, worst L2rel=%.4f%%) — keeping BF16 head\n",
                            match, nsamp, 100*worst_l2);
                }
            } else {
                if (dq) cudaFree(dq); if (dsc) cudaFree(dsc); cudaGetLastError();
                fprintf(stderr, "fucina: NVFP4 FP8 head alloc failed — keeping BF16 head\n");
            }
        }
    } else {
        eng->d_lmhead_bf16 = eng->d_embed_bf16;   // tied (destroy frees once via the alias guard)
    }
    cudaDeviceSynchronize();   // all H2D from nvfp4_model's mmap complete before it falls out of scope
    eng->output_tied = nvfp4_layout.lmhead_key.empty() ? 1 : 0;
    eng->nvfp4_decode_ready = 1;
    fprintf(stderr, "fucina: NVFP4 BF16 embed + %s LM head resident\n",
            nvfp4_layout.lmhead_key.empty() ? "tied" : "untied");
  } else {
    // ── Copy tensor data into device memory (llama.cpp-style residency) ──
    // GEMV kernels previously dereferenced the mmap'd file from the GPU.
    // That works on GB10 unified memory but pays page-fault + host-path
    // costs on every weight read. One bulk upload at load time instead.
    eng->tdata_start = gguf_tensor_data_start(eng->gguf_data, eng->gguf_size);
    if (eng->tdata_start != 0) {
        size_t tbytes = eng->gguf_size - eng->tdata_start;
        if (cudaMalloc(&eng->d_weights, tbytes) == cudaSuccess) {
            fprintf(stderr, "fucina: uploading %.2f GB of weights to device...\n",
                    tbytes / (1024.0*1024.0*1024.0));
            // Pinned, double-buffered sequential streaming (fast). Fall back to the
            // direct mmap copy only if the pinned path fails to initialize.
            if (upload_weights_streamed(eng->d_weights, eng->gguf_fd,
                                        (off_t)eng->tdata_start, tbytes) != 0 &&
                cudaMemcpy(eng->d_weights, eng->gguf_data + eng->tdata_start,
                           tbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                fprintf(stderr, "fucina: weight upload failed — falling back to mmap\n");
                cudaFree(eng->d_weights);
                eng->d_weights = NULL;
            }
        } else {
            fprintf(stderr, "fucina: cudaMalloc(%zu) failed — using mmap'd weights\n",
                    tbytes);
            cudaGetLastError(); // clear error state
        }
    }

    // Unsloth UD dynamic-quant (31B): requantize the off-format bulk projections to Q4_0 into
    // their own device buffers and register pointer overrides. Source bytes come from the mmap'd
    // GGUF (gguf_data + abs offset). Only ffn_down on a few layers is off-format here; the loop
    // scans every layer's ffn_down so it adapts to wherever the UD recipe places Q4_1. The 12B
    // model is uniformly Q4_0 → no tensor matches → n_wt_override stays 0 (byte-identical).
    if (eng->format == FORMAT_Q4_0 && eng->gguf_data) {
        for (int l = 0; l < eng->cfg.n_layers; l++) {
            char tn[128];
            snprintf(tn, sizeof(tn), "blk.%d.ffn_down.weight", l);
            uint64_t off = 0, nbytes = 0; uint32_t gt = 0;
            if (gguf_find_tensor(eng->gguf_data, eng->gguf_size, tn, &off, &nbytes, &gt) != 0)
                continue;
            if (gt == GGML_TYPE_Q4_0) continue;   // already native — read from d_weights
            // ffn_down is [intermediate × hidden] (runtime dims, not the 12B compile constants).
            int64_t n_elem = (int64_t)eng->cfg.intermediate * eng->cfg.hidden_size;
            unsigned char *q40 = NULL;
            const unsigned char *src = eng->gguf_data + off;
            if (gt == GGML_TYPE_Q4_1) {
                q40 = convert_q4_1_to_q4_0(src, n_elem);
            } else {
                fprintf(stderr, "fucina: blk.%d.ffn_down unsupported type %u for Q4_0 override\n", l, gt);
                continue;
            }
            if (!q40) { fprintf(stderr, "fucina: blk.%d.ffn_down requant alloc failed\n", l); continue; }
            size_t q40bytes = (size_t)(n_elem / 32) * 18;
            unsigned char *d_buf = NULL;
            if (cudaMalloc(&d_buf, q40bytes) == cudaSuccess) {
                cudaMemcpy(d_buf, q40, q40bytes, cudaMemcpyHostToDevice);
                if (eng->n_wt_override < (int)(sizeof(eng->wt_override_off)/sizeof(eng->wt_override_off[0]))) {
                    eng->wt_override_off[eng->n_wt_override] = off;
                    eng->wt_override_ptr[eng->n_wt_override] = d_buf;
                    eng->n_wt_override++;
                } else {
                    fprintf(stderr, "fucina: weight override table full — blk.%d.ffn_down NOT overridden\n", l);
                    cudaFree(d_buf);
                }
            } else {
                cudaGetLastError();
                fprintf(stderr, "fucina: blk.%d.ffn_down override cudaMalloc failed\n", l);
            }
            free(q40);
        }
        if (eng->n_wt_override)
            fprintf(stderr, "fucina: requantized %d off-format ffn_down tensor(s) Q4_1->Q4_0\n",
                    eng->n_wt_override);
    }

    // ── Q5_K bulk-weight requant → Q8_0 (Qwen3 UD mixed-precision attention) ──────────────
    // A few attention projections (attn_q/k/v/output) ship Q5_K, which has no native GEMV/dequant
    // kernel. Requantize each to Q8_0 into its own device buffer + register a pointer override, then
    // rewrite the per-tensor fmt to FORMAT_Q8_0 so every read site (gemv_w/dq_gemm/gemv_batched_w)
    // handles it unchanged. Fires only when a bulk weight is FORMAT_Q5_K → Gemma/Q4_K_M models are
    // byte-unchanged (no Q5_K tensors). Source bytes come from the mmap'd GGUF.
    if (eng->gguf_data) {
        int n_q5k_bulk = 0;
        char tn[128];
        const char *suf[] = {"attn_q","attn_k","attn_v","attn_output"};
        uint8_t *fmtfields[4];
        for (int l = 0; l < eng->cfg.n_layers; l++) {
            fmtfields[0] = &eng->tensors.layers[l].fmt_q;
            fmtfields[1] = &eng->tensors.layers[l].fmt_k;
            fmtfields[2] = &eng->tensors.layers[l].fmt_v;
            fmtfields[3] = &eng->tensors.layers[l].fmt_o;
            for (int s = 0; s < 4; s++) {
                if (*fmtfields[s] != (uint8_t)FORMAT_Q5_K) continue;
                snprintf(tn, sizeof(tn), "blk.%d.%s.weight", l, suf[s]);
                uint64_t off = 0, nel = 0; uint32_t gt = 0;
                if (gguf_find_tensor(eng->gguf_data, eng->gguf_size, tn, &off, &nel, &gt) != 0) continue;
                if (gt != GGML_TYPE_Q5_K) continue;
                unsigned char *q8 = convert_q5k_to_q8_0((const unsigned char*)(eng->gguf_data + off), (int64_t)nel);
                if (!q8) { fprintf(stderr, "fucina: %s Q5_K->Q8_0 alloc failed\n", tn); continue; }
                size_t q8bytes = (size_t)(nel / 32) * 34;
                unsigned char *d_buf = NULL;
                if (cudaMalloc(&d_buf, q8bytes) != cudaSuccess) {
                    cudaGetLastError(); free(q8);
                    fprintf(stderr, "fucina: %s Q8_0 cudaMalloc(%zu) failed\n", tn, q8bytes); continue;
                }
                cudaMemcpy(d_buf, q8, q8bytes, cudaMemcpyHostToDevice); free(q8);
                if (eng->n_wt_override < (int)(sizeof(eng->wt_override_off)/sizeof(eng->wt_override_off[0]))) {
                    eng->wt_override_off[eng->n_wt_override] = off;
                    eng->wt_override_ptr[eng->n_wt_override] = d_buf;
                    eng->n_wt_override++;
                    *fmtfields[s] = (uint8_t)FORMAT_Q8_0;   // read as Q8_0 thereafter
                    n_q5k_bulk++;
                } else {
                    fprintf(stderr, "fucina: weight override table full — %s NOT overridden\n", tn);
                    cudaFree(d_buf);
                }
            }
        }
        if (n_q5k_bulk)
            fprintf(stderr, "fucina: requantized %d Q5_K bulk attention tensor(s) -> Q8_0\n", n_q5k_bulk);
    }

    // ── Qwen3.5 GDN in-proj (attn_qkv) Q5_K → Q8_0 requant (P5 decode perf) ─────────────────
    // The 24 GDN/LINEAR layers' fused in-proj (blk.l.attn_qkv.weight, [CONVD=8192 × H=4096]) ships
    // Q5_K. The served decode/prefill formerly materialized the WHOLE weight to fp32 every layer,
    // every step (qwen35_dequant_q5k_f32_kernel → m2_gemm: ~134 MB write + 134 MB read per GDN
    // layer per token), bypassing the native dp4a GEMV every other projection uses. Requantize each
    // once to Q8_0 into its own device buffer + a pointer override, then flip fmt_in_qkv to Q8_0 so
    // every read site (oracle + decode_multiseq_body + prefill_chunk_body) reads it through the
    // validated gemv_batched_w / gemv_w dp4a path (8.5-bit weight, no per-step fp32 dequant) — the
    // SAME proven mechanism the attn_q/k/v/o Q5_K tensors above already use. Qwen3.5-only (attn_qkv
    // exists on no other arch; fmt_in_qkv is 0 elsewhere) → Gemma/Qwen3/MoE byte-unchanged.
    if (eng->gguf_data && eng->cfg.arch == GEMMA4_ARCH_QWEN3_5) {
        int n_q5k_qkv = 0; char tn[128];
        for (int l = 0; l < eng->cfg.n_layers; l++) {
            if (eng->tensors.layers[l].ssm.fmt_in_qkv != (uint8_t)FORMAT_Q5_K) continue;
            snprintf(tn, sizeof(tn), "blk.%d.attn_qkv.weight", l);
            uint64_t off = 0, nel = 0; uint32_t gt = 0;
            if (gguf_find_tensor(eng->gguf_data, eng->gguf_size, tn, &off, &nel, &gt) != 0) continue;
            if (gt != GGML_TYPE_Q5_K) continue;
            unsigned char *q8 = convert_q5k_to_q8_0((const unsigned char*)(eng->gguf_data + off), (int64_t)nel);
            if (!q8) { fprintf(stderr, "fucina: %s Q5_K->Q8_0 alloc failed\n", tn); continue; }
            size_t q8bytes = (size_t)(nel / 32) * 34;
            unsigned char *d_buf = NULL;
            if (cudaMalloc(&d_buf, q8bytes) != cudaSuccess) {
                cudaGetLastError(); free(q8);
                fprintf(stderr, "fucina: %s Q8_0 cudaMalloc(%zu) failed\n", tn, q8bytes); continue;
            }
            cudaMemcpy(d_buf, q8, q8bytes, cudaMemcpyHostToDevice); free(q8);
            if (eng->n_wt_override < (int)(sizeof(eng->wt_override_off)/sizeof(eng->wt_override_off[0]))) {
                eng->wt_override_off[eng->n_wt_override] = off;
                eng->wt_override_ptr[eng->n_wt_override] = d_buf;
                eng->n_wt_override++;
                eng->tensors.layers[l].ssm.fmt_in_qkv = (uint8_t)FORMAT_Q8_0;   // read as Q8_0 thereafter
                n_q5k_qkv++;
            } else {
                fprintf(stderr, "fucina: weight override table full — %s NOT overridden\n", tn);
                cudaFree(d_buf);
            }
        }
        if (n_q5k_qkv)
            fprintf(stderr, "fucina: requantized %d Qwen3.5 GDN in_qkv Q5_K tensor(s) -> Q8_0\n", n_q5k_qkv);
    }


    // QAT Q4_0 model: convert the Q6_K (or Unsloth UD Q4_K) token_embd (= tied LM head) to a
    // Q8_0 device buffer so the existing Q8_0 embed/LM-head kernels handle it (layers stay Q4_0).
    if (embd_src_type == GGML_TYPE_Q6_K || embd_src_type == GGML_TYPE_Q4_K) {
        int64_t n_elem = (int64_t)eng->cfg.vocab_size * eng->cfg.hidden_size;
        const unsigned char *src = (const unsigned char*)(eng->gguf_data + eng->tensors.token_embd);
        unsigned char *q8 = (embd_src_type == GGML_TYPE_Q6_K)
                            ? convert_q6k_to_q8_0(src, n_elem)
                            : convert_q4k_to_q8_0(src, n_elem);
        if (q8) {
            size_t q8bytes = (size_t)(n_elem/32) * 34;
            if (cudaMalloc(&eng->d_token_embd, q8bytes) == cudaSuccess)
                cudaMemcpy(eng->d_token_embd, q8, q8bytes, cudaMemcpyHostToDevice);
            free(q8);
            fprintf(stderr, "fucina: token_embd %s->Q8_0 converted (%.2f GB)\n",
                    embd_src_type == GGML_TYPE_Q6_K ? "Q6_K" : "Q4_K",
                    q8bytes / (1024.0*1024.0*1024.0));
        }
        if (!eng->d_token_embd)
            fprintf(stderr, "fucina: WARNING token_embd conversion failed\n");
    }

    // Step 8 (native Q6_K head), dp4a edition. The first fp32-scalar Q6_K kernel was
    // compute-bound (~76 GB/s, 10.9 ms/token) and lost to the Q8_0 dp4a fallback; the
    // kernel is now the same int8-activation dp4a form as the Q8_0/Q4_0 paths (and the
    // batched variant reads the head once per verify pass), so reading the head natively
    // saves the 0.24 GB/token the Q8_0 upconvert added (1.07 → 0.83 GB). d_lmhead_q6k
    // points INTO d_weights (the raw Q6_K bytes of the tied token_embd) — never freed.
    // d_token_embd (Q8_0 convert) is still used for the one-row embedding lookup.
    if (eng->format == FORMAT_Q4_0 && embd_is_q6k && eng->output_tied &&
        eng->d_weights && eng->d_token_embd) {
        eng->d_lmhead_q6k = (unsigned char *)(eng->d_weights
                            + (eng->tensors.token_embd - eng->tdata_start));
        eng->lmhead_q6k = 1;
        fprintf(stderr, "fucina: LM head native Q6_K dp4a (0.83 GB/token vs 1.07 Q8_0)\n");
    }
    // Native Q4_K head (Unsloth UD): same idea for the Q4_K tied token_embd. Reading the head as
    // native 4.5-bit Q4_K (dp4a, asymmetric scale+min) instead of the Q8_0 upconvert saves
    // ~0.6 GB/token off the V×H read — and is ~numerically identical (the Q8_0 store already
    // holds the Q4_K-dequantized values). d_lmhead_q4k points INTO d_weights (raw bytes, not freed);
    // d_token_embd (Q8_0) is still used for the one-row embedding lookup.
    if (eng->format == FORMAT_Q4_0 && embd_is_q4k && eng->output_tied &&
        eng->d_weights && eng->d_token_embd) {
        eng->d_lmhead_q4k = (unsigned char *)(eng->d_weights
                            + (eng->tensors.token_embd - eng->tdata_start));
        eng->lmhead_q4k = 1;
        fprintf(stderr, "fucina: LM head native Q4_K dp4a (4.5-bit head, ~0.6 GB/token saved vs Q8_0)\n");
    }
  } // else (!is_nvfp4) — GGUF weight residency

    // Build the absolute-id -> global-slot inverse map from global_layer_indices
    // (set by the layer-type detection above). Sliding layers map to -1.
    for (int l = 0; l < GEMMA4_CAP_LAYERS; l++) eng->global_slot[l] = -1;
    for (int g = 0; g < eng->n_layers_global; g++)
        eng->global_slot[eng->global_layer_indices[g]] = g;

    // Opt the global-attn decode kernel into the device's full dynamic shared
    // memory (it needs (32 + n_ctx) floats of scratch — beyond the 48 KB static
    // cap once n_ctx > ~12K). Without this the launch fails silently and stale
    // d_attn_out is consumed. We record the opted-in size to bound n_ctx.
    {
        int max_optin = 0;
        cudaDeviceGetAttribute(&max_optin,
            cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
        if (max_optin > 48 * 1024) {
            cudaError_t fa = cudaFuncSetAttribute(
                (const void *)global_attn_decode_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, max_optin);
            if (fa != cudaSuccess) { cudaGetLastError(); max_optin = 48 * 1024; }
        } else {
            max_optin = 48 * 1024;
        }
        eng->global_attn_max_smem = max_optin;
    }

    // ── GPU-memory budget (--gpu-mem-util) ───────────────────────────────────
    // Weights + fixed scratch/norms are already RESIDENT at this point. We must fit,
    // under budget = util * total_mem, the two remaining big allocations:
    //   - KV cache:  sliding ring (ctx-independent) + global (scales with ctx)
    //   - packed-Q4_0 decode copy (~= weights size; Q4_0 only, optional)
    // The budget auto-derives BOTH an effective global-KV ctx (≤ the user's --ctx,
    // never above) and whether the packed copy fits — unifying the scattered
    // FUCINA_NO_PACKED / FUCINA_SLIDING_RING env knobs (which still WIN if set).
    //
    // OUR footprint is measured as a DELTA: free_at_start - free_now. The GB10 is a
    // shared unified-memory box — other processes' VRAM (and host RAM) is already
    // subtracted from `free`, so the raw (total - free) would wrongly bill their bytes
    // to us. The delta isolates exactly what THIS engine allocated. Two ceilings then
    // apply: (a) the budget cap util*total on our own footprint, and (b) the bytes
    // physically still free right now (we cannot allocate what the OS doesn't have).
    int  budget_packed = 0;        // resolved packed-Q4_0 decision (1 = build it)
    bool packed_forced_off = false;
    {
        size_t free_now = 0, total_now = 0;
        cudaMemGetInfo(&free_now, &total_now);
        const double GiB = 1024.0 * 1024.0 * 1024.0;
        // total_mem was captured at create top; prefer the live total just in case.
        size_t total_mem = total_now ? total_now : eng->total_mem;
        uint64_t budget = (uint64_t)(eng->gpu_mem_util * (double)total_mem);
        // OUR resident bytes = what free dropped by since create-start (weights +
        // scratch + norms). Clamp to ≥0 (free can wobble up if another proc released).
        uint64_t free_start = eng->free_mem;
        uint64_t resident = (free_start > free_now) ? (free_start - free_now) : 0;
        // The free-delta is unreliable on this shared unified-memory box: another
        // process releasing VRAM mid-load makes free_now ≥ free_start, collapsing the
        // delta toward 0 and under-billing OUR weights — which would wrongly fit the
        // packed copy and OOM later. Floor resident at a KNOWN hard lower bound: the
        // bulk-weight bytes we actually uploaded (GGUF; NVFP4 has no contiguous blob,
        // fall back to the delta). This keeps the balance sheet honest and the budget
        // conservative regardless of neighbor churn.
        if (eng->d_weights && eng->gguf_size > eng->tdata_start) {
            uint64_t weight_floor = (uint64_t)(eng->gguf_size - eng->tdata_start);
            if (weight_floor > resident) resident = weight_floor;
        }
        // Headroom STILL free for our remaining allocs, with a small safety reserve so
        // CUDA-graph capture / activation scratch (lazy) don't immediately OOM.
        const uint64_t safety = 512ull << 20;   // 0.5 GiB
        uint64_t free_room = (free_now > safety) ? (free_now - safety) : 0;

        // Per-element KV byte cost (FP8 KV, K and V each), by class.
        const uint64_t kv_elem = sizeof(kv_t);
        // Sliding ring capacity (mirror the existing ring sizing exactly).
        int ring_w = 8192;
        if (const char *e = getenv("FUCINA_SLIDING_RING")) { if (*e) { int v = atoi(e); if (v >= GEMMA4_SLIDING_WINDOW) ring_w = v; } }
        const int ring_floor = GEMMA4_SLIDING_WINDOW + GEMMA4_SPEC_MAX;
        int ring_cap = (int)context_size;
        if (ring_cap > ring_w)     ring_cap = ring_w;
        if (ring_cap < ring_floor) ring_cap = ring_floor;
        // Sliding KV is K+V, ctx-independent once the ring caps it.
        // Qwen3.5 owns hybrid GDN/FULL-attention arenas allocated by ensure_q35_scratch; it never
        // touches the legacy contiguous Gemma KV. Billing/allocating both wasted GiBs at long ctx.
        const bool q35_hybrid = eng->cfg.arch == GEMMA4_ARCH_QWEN3_5;
        uint64_t sliding_bytes = q35_hybrid ? 0 : 2ull * (uint64_t)eng->cfg.n_layers *
            (uint64_t)eng->cfg.n_kv_sliding * (uint64_t)ring_cap *
            GEMMA4_HEAD_DIM * kv_elem;
        // Global KV is K+V and scales LINEARLY with ctx: per-token byte cost.
        uint64_t glob_per_tok = q35_hybrid ? 0 : 2ull * (uint64_t)eng->n_layers_global *
            (uint64_t)eng->cfg.n_kv_global * GEMMA4_GLOBAL_HEAD_DIM * kv_elem;
        // Packed-Q4_0 copy size (== resident bulk-weight bytes). NVFP4 has no Q4_0
        // store; non-Q4_0 / mmap'd weights also can't be packed.
        bool packed_possible = (eng->format == FORMAT_Q4_0) && eng->d_weights;
        const char *no_packed = getenv("FUCINA_NO_PACKED");
        bool env_no_packed = (no_packed && no_packed[0] == '1');
        uint64_t packed_bytes = packed_possible
            ? (uint64_t)(eng->gguf_size - eng->tdata_start) : 0;

        // REQUIRED floor = resident weights/scratch + sliding ring + a MIN global KV
        // (the ring_floor's worth of global ctx — enough to decode short prompts).
        uint64_t min_global = glob_per_tok * (uint64_t)ring_floor;
        uint64_t required = resident + sliding_bytes + min_global;
        // The required KV (sliding + min global) must fit BOTH the budget headroom and
        // the physically-free room; report whichever ceiling is tighter.
        uint64_t budget_room = (budget > resident) ? (budget - resident) : 0;
        uint64_t need_kv = sliding_bytes + min_global;
        if (required > budget || need_kv > free_room) {
            bool over_budget = (required > budget);
            fprintf(stderr,
                "fucina: GPU memory budget exceeded — model needs weights+scratch %.2f GiB "
                "+ min KV %.2f GiB; %s.\n"
                "        budget %.2f GiB at util %.2f of %.2f GiB total; %.2f GiB physically free.\n"
                "        Raise --gpu-mem-util, lower --ctx, free the GPU, or pick another --cuda-device.\n",
                resident / GiB, need_kv / GiB,
                over_budget ? "exceeds the util budget" : "exceeds the free GPU memory",
                budget / GiB, eng->gpu_mem_util, total_mem / GiB, free_now / GiB);
            gemma4_engine_destroy(eng);
            return NULL;
        }

        // Spend the remaining room: prefer the packed copy IF it fits while still
        // leaving room for at least the min global KV; otherwise drop it and give the
        // freed bytes to ctx. Then cap ctx so the global KV fits what's left. `avail`
        // is bounded by BOTH the util budget and the physically-free memory.
        uint64_t avail = budget_room - sliding_bytes;         // budget side
        uint64_t free_avail = free_room - sliding_bytes;      // physical side
        if (free_avail < avail) avail = free_avail;           // tighter ceiling wins
        if (packed_possible && !env_no_packed) {
            if (packed_bytes + min_global <= avail) {
                budget_packed = 1;
                avail -= packed_bytes;
            } else {
                packed_forced_off = true;   // would not leave room for a usable KV
            }
        } else if (env_no_packed) {
            // Honor the explicit env opt-out; not a budget decision.
            budget_packed = 0;
        }

        // ── NVFP4-prefill copy budget (the lazy ~16.5 GiB build_fp4_weights store) ──
        // The prefill speed path (FUCINA_FP4, ~2.4× BF16) builds a THIRD weight copy on
        // first long prefill: packed E2M1 [out×in/2] + swizzled E4M3 scales, ~= the model
        // weight bytes. It was never billed here, so a util-0.4 budget that excludes it let
        // the 31B climb to ~64 GiB. Bill it now: fp4_budget_ok is TRUE only if it fits the
        // remaining room (after resident + packed + the min-global KV floor) under BOTH the
        // util budget and the physically-free ceiling. When false, the use_fp4 prefill
        // decision (gemm path) declines to build it and falls back to BF16/MMQ (slower, but
        // no extra copy). An EXPLICIT FUCINA_FP4=1 force-overrides at the prefill site.
        // FORMAT_NVFP4 residency keeps no separate prefill copy (the store IS the residency),
        // so there is nothing extra to bill there — gate on a GGUF d_weights model. The Qwen3.5
        // HYBRID prefill uses q35_proj_gemm (BF16 cache), NEVER gemm_nvfp4 — and proj_desc can't
        // even describe its 2×-wide gated query / GDN tensors — so the NVFP4 copy would be unused,
        // wrong-shaped waste. Don't reserve its ~3.4 GB budget (frees it for the KV/weight-cache).
        bool fp4_possible = eng->d_weights && eng->format != FORMAT_NVFP4
                            && eng->cfg.arch != GEMMA4_ARCH_QWEN3_5;
        // EXPLICIT FUCINA_FP4=0 opts the prefill copy out entirely — it will never be built,
        // so don't reserve its bytes here (give that room to ctx) and report it as such.
        const char *fp4_env = getenv("FUCINA_FP4");
        bool fp4_env_off = (fp4_env && fp4_env[0] == '0');
        uint64_t fp4_prefill_bytes = (fp4_possible && !fp4_env_off)
                                     ? gemma4_fp4_prefill_footprint(eng) : 0;
        if (fp4_possible && !fp4_env_off) {
            // Must fit alongside the minimum global KV that the budget guarantees.
            eng->fp4_budget_ok = (fp4_prefill_bytes + min_global <= avail) ? 1 : 0;
            if (eng->fp4_budget_ok) avail -= fp4_prefill_bytes;
        } else {
            eng->fp4_budget_ok = 0;
        }

        // Effective global ctx = min(user ctx, avail / per-token global cost).
        uint32_t eff_ctx = context_size;
        if (glob_per_tok > 0) {
            uint64_t fit_ctx = avail / glob_per_tok;
            if (fit_ctx < eff_ctx) eff_ctx = (uint32_t)fit_ctx;
        }
        if (eff_ctx < (uint32_t)ring_floor) eff_ctx = (uint32_t)ring_floor;  // guaranteed by REQUIRED check
        bool ctx_capped = (eff_ctx < context_size);
        context_size = eff_ctx;
        eng->context_size = eff_ctx;
        eng->sliding_kv_capacity = ring_cap;

        // Recompute the final global KV at the (possibly capped) ctx for the log.
        uint64_t global_bytes = glob_per_tok * (uint64_t)eff_ctx;
        uint64_t total_use = resident + sliding_bytes + global_bytes +
                             (budget_packed ? packed_bytes : 0) +
                             (eng->fp4_budget_ok ? fp4_prefill_bytes : 0);

        // ── Balance sheet ────────────────────────────────────────────────────
        fprintf(stderr,
            "fucina: ── GPU memory budget ──────────────────────────────\n"
            "fucina:   total device mem : %8.2f GiB\n"
            "fucina:   free at start    : %8.2f GiB  (shared box: %.2f GiB held by others)\n"
            "fucina:   --gpu-mem-util   : %8.2f\n"
            "fucina:   budget (our cap) : %8.2f GiB\n"
            "fucina:   weights+scratch  : %8.2f GiB  (resident, measured)\n"
            "fucina:   KV sliding ring  : %8.2f GiB  (cap %d, ctx-independent)\n"
            "fucina:   KV global        : %8.2f GiB  (ctx %u%s)\n"
            "fucina:   packed Q4_0 copy : %8.2f GiB  (%s)\n"
            "fucina:   NVFP4 prefill wts: %8.2f GiB  (%s)\n"
            "fucina:   ─────────────────────────────────────────────────\n"
            "fucina:   our total        : %8.2f GiB  (%s budget by %.2f GiB)\n",
            total_mem / GiB,
            free_start / GiB, (total_mem > free_start ? (total_mem - free_start) : 0) / GiB,
            eng->gpu_mem_util, budget / GiB,
            resident / GiB,
            sliding_bytes / GiB, ring_cap,
            global_bytes / GiB, eff_ctx, ctx_capped ? ", CAPPED from --ctx" : "",
            (budget_packed ? packed_bytes : 0) / GiB,
            budget_packed ? "ON" :
                (packed_forced_off ? "OFF — no budget" :
                 (env_no_packed ? "OFF — FUCINA_NO_PACKED" :
                  (packed_possible ? "OFF" : "n/a"))),
            (eng->fp4_budget_ok ? fp4_prefill_bytes : 0) / GiB,
            !fp4_possible ? "n/a" :
                (fp4_env_off ? "OFF — FUCINA_FP4=0" :
                 (eng->fp4_budget_ok ? "ON — budget" : "OFF — budget")),
            total_use / GiB,
            total_use <= budget ? "within" : "OVER",
            (total_use <= budget ? (budget - total_use) : (total_use - budget)) / GiB);
    }

    // KV cache allocation
    // Sliding cache is a per-position RING: [MAX_LAYERS][cap][8×256], position p stored at
    // slot p % cap. Sliding-window attention only ever reads the last GEMMA4_SLIDING_WINDOW
    // positions, so cap need not equal context_size — capping it makes the sliding cache
    // (the dominant, ctx-scaling allocation) nearly context-independent: ~768 MB at cap=8192
    // vs ~21 GB flat @131k. cap = min(context_size, FUCINA_SLIDING_RING) (default 8192,
    // floored at the window). The margin cap-window bounds how far a prefix-reuse rewind can
    // look back exactly; deeper rewinds report failure (gemma4_engine_rewind) and the server
    // falls back to a full re-prefill. Speculation rewinds ≤ GEMMA4_SPEC_MAX, always inside it.
    // (eng->sliding_kv_capacity already set by the GPU-memory-budget block above.)
    // M0 A3: the sliding cache is indexed by ABSOLUTE layer id (cfg.n_layers slots) and now
    // sized by the runtime sliding KV-head count (8 on 12B, 16 on 31B). On 12B cfg.n_kv_sliding
    // == GEMMA4_KV_HEADS and cfg.n_layers == GEMMA4_MAX_LAYERS so this is byte-identical.
    const bool q35_hybrid_kv = eng->cfg.arch == GEMMA4_ARCH_QWEN3_5;
    size_t sliding_kv_size = q35_hybrid_kv ? 0 : (size_t)eng->cfg.n_layers *
        (size_t)eng->cfg.n_kv_sliding * (size_t)eng->sliding_kv_capacity * GEMMA4_HEAD_DIM * sizeof(kv_t);
    if (!q35_hybrid_kv && (cudaMalloc(&eng->d_sliding_k, sliding_kv_size) != cudaSuccess ||
        cudaMalloc(&eng->d_sliding_v, sliding_kv_size) != cudaSuccess)) {
        fprintf(stderr, "fucina: failed to allocate sliding KV cache (%.1f MB ×2)\n",
                sliding_kv_size / (1024.0*1024.0));
        cudaGetLastError();
        free(eng);
        return NULL;
    }

    // Global K and V caches: float, separate K and V.
    // Layout: [n_layers_global][ctx_size][head_dim] — only the global layers get
    // a slot (not all 48), so this is ~6× smaller than indexing by absolute id.
    eng->global_kv_capacity = context_size;
    size_t global_kv_size = q35_hybrid_kv ? 0 : (size_t)eng->n_layers_global *
        context_size * (size_t)eng->cfg.n_kv_global * GEMMA4_GLOBAL_HEAD_DIM * sizeof(kv_t);
    if (!q35_hybrid_kv && (cudaMalloc(&eng->d_global_k, global_kv_size) != cudaSuccess ||
        cudaMalloc(&eng->d_global_v, global_kv_size) != cudaSuccess)) {
        fprintf(stderr, "fucina: failed to allocate global KV cache (%.1f MB ×2)\n",
                global_kv_size / (1024.0*1024.0));
        cudaGetLastError();
        free(eng);
        return NULL;
    }

    eng->loaded = 1;
    fprintf(stderr, "fucina: engine initialized (%.2f GB model, %u ctx, %s)\n",
            eng->gguf_size / (1024.0*1024.0*1024.0),
            context_size,
            eng->format == FORMAT_NVFP4 ? "NVFP4" : (eng->format == FORMAT_Q4_0 ? "Q4_0" : "Q8_0"));
    fprintf(stderr, "fucina: %d sliding + %d global layers\n",
            eng->n_layers_sliding, eng->n_layers_global);
    fprintf(stderr, "fucina: KV cache: sliding=%.1f MB, global=%.1f MB\n",
            sliding_kv_size / (1024.0*1024.0),
            global_kv_size / (1024.0*1024.0));

    // ── Paged KV pools (Phase 2, opt-in via FUCINA_PAGED_KV) ──────────────
    // Allocate the shared block pools for continuous batching, sized to free
    // VRAM after weights+contiguous-KV are resident. Dormant for now: nothing
    // reads/writes them until the paged write/read paths are wired (inc 2/3), so
    // with the flag OFF (default) this is skipped and behaviour is unchanged.
    // Pool element layout per class: [layer][block_id][offset][elems]. The
    // sliding pool is indexed by ABSOLUTE layer id (GEMMA4_MAX_LAYERS slots, 8 of
    // them unused for the global layers) so the KV-write mirror can index it with
    // the same `layer` the contiguous cache uses — no sliding-slot map. The global
    // pool is indexed by the compact global_slot[layer] (n_layers_global slots).
    // KV codec select (Phase 6). FUCINA_KV_NVFP4 fake-quants KV through NVFP4
    // precision (E2M1 + per-16 E4M3 block scale) at every store site; default OFF
    // keeps flat FP8 byte-identical. Set the device constant once. See
    // docs/kv-quant-exploration.md.
    {
        int kv_nvfp4 = getenv("FUCINA_KV_NVFP4") ? 1 : 0;
        cudaMemcpyToSymbol(c_kv_nvfp4, &kv_nvfp4, sizeof(int));
        if (kv_nvfp4)
            printf("fucina: KV codec = NVFP4 fake-quant (E2M1 + per-16 E4M3 block scale) "
                   "[Phase 6 bench; ~4.5-bit precision, storage still FP8]\n");
    }

    // Paged pools are opt-in (FUCINA_PAGED_KV) for Gemma, but MANDATORY for the Qwen3
    // family: every Qwen3/Qwen3-MoE prefill+decode entry point routes exclusively
    // through the paged multiseq path (the non-paged gemma4_engine_prefill* decline
    // with -1/-2 for GEMMA4_IS_QWEN3_FAMILY), so without these pools a bare Qwen3
    // launch would 500 on every request. Auto-enable from the detected arch — runtime
    // model detection, no env flag required (Gemma stays opt-in / byte-identical).
    if (getenv("FUCINA_PAGED_KV")) {
        const int BT = PAGED_KV_BLOCK_TOKENS;
        // Per-token element count per KV-class. Qwen3 is single-head-dim (128) and routes every
        // layer through the global class, so glob_elems uses cfg.head_dim (NOT the 512 Gemma const)
        // — this MUST match the runtime okv = n_kv*head_dim used at write/read time in the pools.
        const int q3 = 0;
        const int slid_elems = eng->cfg.n_kv_sliding * (q3 ? eng->cfg.head_dim : GEMMA4_HEAD_DIM);
        const int glob_elems = eng->cfg.n_kv_global  * (q3 ? eng->cfg.head_dim : GEMMA4_GLOBAL_HEAD_DIM);
        // Per-block bytes for K (== V) across all layers of the class.
        size_t slid_block_k = (size_t)eng->cfg.n_layers    * BT * slid_elems * sizeof(kv_t);
        size_t glob_block_k = (size_t)eng->n_layers_global * BT * glob_elems * sizeof(kv_t);
        // K+V together = the cost of one block in each class.
        uint64_t slid_block_kv = 2ull * slid_block_k;
        uint64_t glob_block_kv = 2ull * glob_block_k;

        // Capacity-based sizing (NOT a raw VRAM fraction): the pool is dimensioned
        // by how many concurrent sequences fit. Per sequence:
        //   - sliding RECYCLES, so it only ever holds the window: a small, FIXED
        //     ceil(window/BT)+1 blocks regardless of sequence length.
        //   - global does NOT recycle, so it grows to ceil(maxctx/BT) blocks.
        // max_seqs = budget / per-seq cost, capped. maxctx defaults to a practical
        // serving context (env FUCINA_PAGED_MAXCTX), not the full 262k window.
        int slid_per_seq = (GEMMA4_SLIDING_WINDOW + BT - 1) / BT + 1;     // window blocks (+1 spill)
        int maxctx = 32768;
        if (const char *mc = getenv("FUCINA_PAGED_MAXCTX")) { int v = atoi(mc); if (v > 0) maxctx = v; }
        if (maxctx > (int)context_size) maxctx = (int)context_size;
        int glob_per_seq = (maxctx + BT - 1) / BT;                         // ctx blocks (no recycle)
        uint64_t per_seq_kv = (uint64_t)slid_per_seq * slid_block_kv +
                              (uint64_t)glob_per_seq * glob_block_kv;

        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const uint64_t reserve = 3ull << 30;   // 3 GiB headroom for activations/graphs
        uint64_t budget = (free_b > reserve) ? (free_b - reserve) : 0;
        int max_seqs = (per_seq_kv > 0) ? (int)(budget / per_seq_kv) : 0;
        if (max_seqs < 1)  max_seqs = 1;
        if (max_seqs > 64) max_seqs = 64;       // sane concurrency cap (tunable later)
        if (const char *ms = getenv("FUCINA_PAGED_MAXSEQS")) { int v = atoi(ms); if (v > 0) max_seqs = v; }
        // +1 sequence of slack so a brief over-subscription doesn't wedge admission.
        int slid_blocks = slid_per_seq * (max_seqs + 1);
        int glob_blocks = glob_per_seq * (max_seqs + 1);

        int ok = (slid_blocks > 0 && glob_blocks > 0) &&
                 paged_pool_init(&eng->slid_pool, slid_blocks, BT) == 0 &&
                 paged_pool_init(&eng->glob_pool, glob_blocks, BT) == 0;
        if (ok) {
            size_t slid_pool_bytes = (size_t)slid_blocks * slid_block_k;
            size_t glob_pool_bytes = (size_t)glob_blocks * glob_block_k;
            ok = cudaMalloc(&eng->d_slid_pool_k, slid_pool_bytes) == cudaSuccess &&
                 cudaMalloc(&eng->d_slid_pool_v, slid_pool_bytes) == cudaSuccess &&
                 cudaMalloc(&eng->d_glob_pool_k, glob_pool_bytes) == cudaSuccess &&
                 cudaMalloc(&eng->d_glob_pool_v, glob_pool_bytes) == cudaSuccess;
            if (ok) {
                eng->paged_enabled = 1;
                // The pools back (max_seqs+1) sequences at full per-seq reservation;
                // the slot array holds GEMMA4_MAX_SEQS. Concurrency is the min — this
                // is what seq_capacity() admits against, so the block pool is never
                // over-subscribed (each admitted seq is guaranteed its maxctx blocks).
                eng->paged_cap = (max_seqs + 1 < GEMMA4_MAX_SEQS) ? (max_seqs + 1) : GEMMA4_MAX_SEQS;
                // Cross-request prefix cache: only on the full-attention single-pool
                // geometry (no sliding layers => global blocks are never recycled, so
                // a shared prefix block stays valid for its whole cached lifetime).
                if (eng->n_layers_sliding == 0 && getenv("FUCINA_NO_PREFIX_CACHE") == NULL) {
                    prefix_tree_init(&eng->glob_prefix, &eng->glob_pool, 2 * glob_blocks + 16);
                    eng->prefix_cache_enabled = 1;   // default-on for the full-attention single-pool geometry
                    fprintf(stderr, "fucina: cross-request prefix cache ENABLED "
                            "(RadixAttention; %d global blocks, %d-token granularity)\n",
                            glob_blocks, BT);
                }
                fprintf(stderr,
                    "fucina: paged KV ENABLED — ~%d concurrent seqs @ maxctx %d "
                    "(block=%d tok): sliding %d blk %.1f GB, global %d blk %.1f GB\n",
                    max_seqs, maxctx, BT,
                    slid_blocks, (2.0*slid_pool_bytes) / (1024.0*1024.0*1024.0),
                    glob_blocks, (2.0*glob_pool_bytes) / (1024.0*1024.0*1024.0));
            }
        }
        if (!ok) {
            // Allocation failed → fall back to the contiguous path cleanly.
            cudaGetLastError();
            CUDA_FREE(eng->d_slid_pool_k); CUDA_FREE(eng->d_slid_pool_v);
            CUDA_FREE(eng->d_glob_pool_k); CUDA_FREE(eng->d_glob_pool_v);
            paged_pool_destroy(&eng->slid_pool);
            paged_pool_destroy(&eng->glob_pool);
            eng->paged_enabled = 0;
            fprintf(stderr, "fucina: paged KV requested but allocation failed — "
                            "using contiguous KV\n");
        }
    }

    // Repacked-Q4_0 decode GEMV: bit-exact, ~+2-3% decode via coalesced uint4 weight
    // loads; costs +weights-size VRAM. The GPU-memory-budget block above already decided
    // budget_packed (1 only if it fits under --gpu-mem-util AND FUCINA_NO_PACKED is not
    // set; an explicit FUCINA_NO_PACKED=1 forces it off there). Built eagerly now — before
    // any request or CUDA-graph capture — so the decode GEMV's packed branch is a pure read.
    // NVFP4 has NO Q4_0 store (single-store invariant) — skip the repack entirely.
    if (!is_nvfp4 && budget_packed) {
        build_packed_q4(eng);   // non-fatal: falls back if it fails
    }
    // Packed Q4_K (Qwen3 / K-quant): in-place superblock repack → coalesced uint4 GEMV loads.
    // Same 144 B per superblock → no extra weight VRAM, so it is NOT gated on budget_packed.
    // Bit-identical to the native Q4_K dp4a; non-fatal (falls back to native on failure).
    if (!is_nvfp4) {
        build_packed_q4k(eng);
    }

    // GGUF Qwen descriptors are bound after all override conversion and in-place repacking. The
    // safetensors loader binds the same fields in qwen35_fp8_fill_engine().
    if (eng->cfg.arch == GEMMA4_ARCH_QWEN3_5) {
        auto bind_ref=[&](WeightRef &ref,uint64_t off,uint8_t fmt,int out_dim,int in_dim){
            ref.data=weight_fp8(eng,off); ref.scale=nullptr; ref.global_scale=nullptr;
            ref.out_dim=out_dim; ref.in_dim=in_dim;
            ref.encoding=(fmt==FORMAT_Q4_K)?WeightEncoding::Q4_K:
                         (fmt==FORMAT_Q8_0)?WeightEncoding::Q8_0:
                         (fmt==FORMAT_Q4_0)?WeightEncoding::Q4_0:
                         (fmt==FORMAT_Q6_K)?WeightEncoding::Q6_K:WeightEncoding::FP8_BLOCK_128;
            ref.layout=(fmt==FORMAT_Q4_K && eng->q4k_packed)?TensorLayout::Q4K_PACKED:
                       (fmt==FORMAT_FP8_BLOCK)?TensorLayout::ROW_MAJOR:TensorLayout::GGML_NATIVE;
            ref.flags=WEIGHT_FLAG_PRIMARY |
                      ((ref.layout==TensorLayout::Q4K_PACKED)?WEIGHT_FLAG_PACKED:0);
        };
        const int H=eng->cfg.hidden_size, HD=eng->cfg.head_dim, NQ=eng->cfg.n_heads;
        const int NKV=eng->cfg.n_kv_global, INNER=eng->cfg.ssm_inner_size;
        const int CONVD=2*eng->cfg.ssm_group_count*eng->cfg.ssm_state_size+INNER;
        for(int l=0;l<eng->cfg.n_layers;l++){
            auto &T=eng->tensors.layers[l];
            if(eng->cfg.attn_kind[l]==GEMMA4_ATTN_FULL){
                bind_ref(T.ref_q,T.attn_q,T.fmt_q,2*NQ*HD,H);
                bind_ref(T.ref_k,T.attn_k,T.fmt_k,NKV*HD,H);
                bind_ref(T.ref_v,T.attn_v,T.fmt_v,NKV*HD,H);
                bind_ref(T.ref_o,T.attn_output,T.fmt_o,H,NQ*HD);
            } else {
                bind_ref(T.ssm.ref_in_qkv,T.ssm.in_qkv,T.ssm.fmt_in_qkv,CONVD,H);
                bind_ref(T.ssm.ref_in_z,T.ssm.in_z,T.ssm.fmt_in_z,INNER,H);
                bind_ref(T.ssm.ref_in_a,T.ssm.in_a,T.ssm.fmt_in_a,eng->cfg.ssm_time_step_rank,H);
                bind_ref(T.ssm.ref_in_b,T.ssm.in_b,T.ssm.fmt_in_b,eng->cfg.ssm_time_step_rank,H);
                bind_ref(T.ssm.ref_out,T.ssm.out,T.ssm.fmt_out,H,INNER);
            }
            if(eng->cfg.n_experts==0){
                bind_ref(T.ref_gate,T.ffn_gate,T.fmt_gate,eng->cfg.intermediate,H);
                bind_ref(T.ref_up,T.ffn_up,T.fmt_up,eng->cfg.intermediate,H);
                bind_ref(T.ref_down,T.ffn_down,T.fmt_down,H,eng->cfg.intermediate);
            }
        }
    }

    // Paged KV mirror validation (opt-in): decode a fixed run and assert the
    // paged pool matches the contiguous cache byte-for-byte. Non-fatal.
    if (getenv("FUCINA_PAGED_KV_SELFTEST"))   gemma4_engine_paged_selftest(eng);
    // Paged READ validation (opt-in): assert paged attention == contiguous attn.
    if (getenv("FUCINA_PAGED_READ_SELFTEST")) gemma4_engine_paged_read_selftest(eng);
    // Paged E2E validation (opt-in): assert paged-read generation == contiguous.
    if (getenv("FUCINA_PAGED_E2E_SELFTEST"))  gemma4_engine_paged_e2e_selftest(eng);
    // Multi-seq batched-decode validation (opt-in): batched(B) == B single decodes.
    if (getenv("FUCINA_BATCH_SELFTEST"))      gemma4_engine_batch_selftest(eng);
    if (getenv("FUCINA_FAST_PREFILL_SELFTEST")) gemma4_engine_fast_prefill_selftest(eng);
    if (getenv("FUCINA_BATCH_DECODE_BENCH"))  gemma4_engine_batch_decode_bench(eng);

    return eng;
}

void gemma4_engine_destroy(gemma4_engine_t *eng) {
    if (!eng) return;

    if (eng->ssd_expert_profile) {
        fucina_expert_stream_stats_t stats = {
            eng->fp4m_cache_hits, eng->fp4m_cache_misses, eng->fp4m_ssd_reads,
            eng->fp4m_ssd_bytes, eng->fp4m_ssd_checksum_fail, eng->fp4m_prefetch_advice};
        if (fucina_expert_profile_finish(eng->ssd_expert_profile, &stats) != 0)
            fprintf(stderr, "fucina: failed to atomically write SSD expert profile; inference result is unaffected\n");
        eng->ssd_expert_profile = NULL;
    }

    delete eng->device_allocations;
    eng->device_allocations = nullptr;

    if (eng->gguf_data && eng->gguf_data != MAP_FAILED) {
        munmap((void *)eng->gguf_data, eng->gguf_size);
    }
    if (eng->gguf_fd >= 0) close(eng->gguf_fd);

    // Avoid macro redefinition warning - undefine first
    #ifdef CUDA_FREE
    #undef CUDA_FREE
    #endif
    #define CUDA_FREE(ptr) do { if (ptr) cudaFree(ptr); } while(0)
    CUDA_FREE(eng->d_scratch);
    CUDA_FREE(eng->d_logits);
    CUDA_FREE(eng->d_x);
    CUDA_FREE(eng->d_attn_q);
    CUDA_FREE(eng->d_attn_k);
    CUDA_FREE(eng->d_attn_v);
    CUDA_FREE(eng->d_attn_out);
    CUDA_FREE(eng->d_fa_acc);
    CUDA_FREE(eng->d_fa_m);
    CUDA_FREE(eng->d_fa_l);
    CUDA_FREE(eng->d_ffn_out);
    CUDA_FREE(eng->d_ffn_gate);
    CUDA_FREE(eng->d_ffn_up);
    // Qwen3-MoE: per-layer requantized Q8_0 down slabs + forward scratch.
    for (int l = 0; l < GEMMA4_CAP_LAYERS; l++) CUDA_FREE(eng->moe_down_q8[l]);
    CUDA_FREE(eng->d_moe_rlogits); CUDA_FREE(eng->d_moe_tki);  CUDA_FREE(eng->d_moe_tkw);
    CUDA_FREE(eng->d_moe_eidx);    CUDA_FREE(eng->d_moe_ecs);  CUDA_FREE(eng->d_moe_count);
    CUDA_FREE(eng->d_moe_coloff);  CUDA_FREE(eng->d_moe_cursor); CUDA_FREE(eng->d_moe_ones);
    CUDA_FREE(eng->d_moe_xe);      CUDA_FREE(eng->d_moe_gate);  CUDA_FREE(eng->d_moe_up);
    CUDA_FREE(eng->d_moe_act);     CUDA_FREE(eng->d_moe_oe);    CUDA_FREE(eng->d_moe_shlog);
    CUDA_FREE(eng->d_moe_active);
    CUDA_FREE(eng->d_moe_profile_count); CUDA_FREE(eng->d_moe_profile_weight);
    CUDA_FREE(eng->d_moe_profile_act_ss); CUDA_FREE(eng->d_moe_profile_act_n);
    CUDA_FREE(eng->d_moe_profile_act_max);
    for (int i = 0; i < 3; i++) CUDA_FREE(eng->d_moe_wbf[i]);
    for (int i = 0; i < GEMMA4_CAP_LAYERS; i++) {
        CUDA_FREE(eng->d_fp4m_gu[i]); CUDA_FREE(eng->d_fp4m_gusf[i]);
        CUDA_FREE(eng->d_fp4m_dn[i]); CUDA_FREE(eng->d_fp4m_dnsf[i]);
    }
    if (eng->h_fp4m_stage_gu) free(eng->h_fp4m_stage_gu);
    if (eng->h_fp4m_stage_gusf) free(eng->h_fp4m_stage_gusf);
    if (eng->h_fp4m_stage_dn) free(eng->h_fp4m_stage_dn);
    if (eng->h_fp4m_stage_dnsf) free(eng->h_fp4m_stage_dnsf);
    CUDA_FREE(eng->d_fp4m_stage_gu); CUDA_FREE(eng->d_fp4m_stage_gusf);
    CUDA_FREE(eng->d_fp4m_stage_dn); CUDA_FREE(eng->d_fp4m_stage_dnsf); CUDA_FREE(eng->d_fp4m_eslot);
    if (eng->fp4m_ssd_fd > 0) close(eng->fp4m_ssd_fd);
    if (eng->h_fp4m_ssd_hash) free(eng->h_fp4m_ssd_hash);
    if (eng->h_fp4m_ssd_verified) free(eng->h_fp4m_ssd_verified);
    if (eng->h_fp4m_slot_layer) free(eng->h_fp4m_slot_layer);
    if (eng->h_fp4m_slot_expert) free(eng->h_fp4m_slot_expert);
    if (eng->h_fp4m_slot_age) free(eng->h_fp4m_slot_age);
    CUDA_FREE(eng->d_fp4m_gsw);
    CUDA_FREE(eng->d_fp4m_a);      CUDA_FREE(eng->d_fp4m_asf);
    CUDA_FREE(eng->d_fp4m_a2);     CUDA_FREE(eng->d_fp4m_a2sf);
    CUDA_FREE(eng->d_fp4m_gu_out); CUDA_FREE(eng->d_fp4m_dn_out);
    CUDA_FREE(eng->d_fp4m_indptr); CUDA_FREE(eng->d_fp4m_t2e);
    CUDA_FREE(eng->d_fp4m_gsrow);
    CUDA_FREE(eng->d_moe_xbf);     CUDA_FREE(eng->d_moe_shbf);
    CUDA_FREE(eng->d_moe_q8);      CUDA_FREE(eng->d_moe_q8d);   CUDA_FREE(eng->d_moe_q8s);
    CUDA_FREE(eng->d_norm);
    CUDA_FREE(eng->d_norm_w);
    CUDA_FREE(eng->d_residual);
    CUDA_FREE(eng->d_rope_freqs);
    CUDA_FREE(eng->d_head_norm_w);
    CUDA_FREE(eng->d_sliding_k);
    CUDA_FREE(eng->d_sliding_v);
    CUDA_FREE(eng->d_global_k);
    CUDA_FREE(eng->d_global_v);
    // Paged KV pools + the active sequence's block tables (no-ops when paged off).
    CUDA_FREE(eng->d_slid_pool_k);
    CUDA_FREE(eng->d_slid_pool_v);
    CUDA_FREE(eng->d_glob_pool_k);
    CUDA_FREE(eng->d_glob_pool_v);
    prefix_tree_destroy(&eng->glob_prefix);   // safe no-op when never initialized
    paged_pool_destroy(&eng->slid_pool);
    paged_pool_destroy(&eng->glob_pool);
    CUDA_FREE(eng->cur.d_slid_blocks);
    CUDA_FREE(eng->cur.d_glob_blocks);
    paged_table_free_struct(&eng->cur.slid_bt);
    paged_table_free_struct(&eng->cur.glob_bt);
    // Multi-seq slots + their device block-id arrays + the ms_* batched scratch.
    for (int i = 0; i < GEMMA4_MAX_SEQS; i++) {
        CUDA_FREE(eng->slots[i].d_slid_blocks);
        CUDA_FREE(eng->slots[i].d_glob_blocks);
        CUDA_FREE(eng->slots[i].d_mtp_h);
        paged_table_free_struct(&eng->slots[i].slid_bt);
        paged_table_free_struct(&eng->slots[i].glob_bt);
    }
    CUDA_FREE(eng->d_ms_pos);
    CUDA_FREE(eng->d_ms_outtok);
    CUDA_FREE(eng->d_ms_views_slid);
    CUDA_FREE(eng->d_ms_views_glob);
    CUDA_FREE(eng->d_ms_temp);
    CUDA_FREE(eng->d_ms_topk);
    CUDA_FREE(eng->d_ms_topp);
    CUDA_FREE(eng->d_ms_minp);
    CUDA_FREE(eng->d_ms_rnd);
    // qwen35 M4 batched-decode arenas + captured graphs (no-ops when never allocated).
    for (int i = 0; i < Q35_GRAPH_CACHE_CAP; i++)
        if (eng->q35.graph_cache[i].exec) {
            cudaGraphExecDestroy(eng->q35.graph_cache[i].exec);
            eng->q35.graph_cache[i].exec = NULL;
        }
    eng->q35.graph_count = 0;
    for (int l = 0; l < GEMMA4_CAP_LAYERS; l++) {
        // S_slot/ring_slot are non-owning views into recurrent_slab[slot].
        for (int s = 0; s < GEMMA4_MAX_SEQS; s++) {
            eng->q35.S_slot[l][s] = NULL; eng->q35.ring_slot[l][s] = NULL;
            CUDA_FREE(eng->q35.Kc_slot[l][s]); CUDA_FREE(eng->q35.Vc_slot[l][s]);
        }
        CUDA_FREE(eng->q35.S[l]); CUDA_FREE(eng->q35.ring[l]);
        CUDA_FREE(eng->q35.Kc[l]); CUDA_FREE(eng->q35.Vc[l]);
    }
    for (int s = 0; s < GEMMA4_MAX_SEQS; s++) CUDA_FREE(eng->q35.recurrent_slab[s]);
    for (int s = 0; s < GEMMA4_MAX_SEQS; s++) {   // P0 GDN rollback snapshots
        CUDA_FREE(eng->q35.gdn_snap_slab[s]); eng->q35.gdn_snap_ntokens[s] = -1;
    }
    for (int i = 0; i < 24; i++) CUDA_FREE(eng->q35.sb[i]);
    CUDA_FREE(eng->q35.rowslot);
    CUDA_FREE(eng->q35.chunk_scr);
    CUDA_FREE(eng->q35.d_pf_seqmeta);
    CUDA_FREE(eng->q35.part_m); CUDA_FREE(eng->q35.part_l); CUDA_FREE(eng->q35.part_o);
    CUDA_FREE(eng->q35.pf_pos);
    CUDA_FREE(eng->q35.pf_tok);
    CUDA_FREE(eng->q35.d_slot_tok);   // S2a persistent per-slot decode state
    CUDA_FREE(eng->q35.d_slot_pos);
    CUDA_FREE(eng->q35.wbf16[0]);
    CUDA_FREE(eng->q35.wbf16[1]);
    CUDA_FREE(eng->q35.xbf16);
    CUDA_FREE(eng->q35.qb); CUDA_FREE(eng->q35.kb); CUDA_FREE(eng->q35.vb);
    CUDA_FREE(eng->q35.kbx); CUDA_FREE(eng->q35.vbx); CUDA_FREE(eng->q35.pb);
    CUDA_FREE(eng->q35.scores);
    CUDA_FREE(eng->q35.qh); CUDA_FREE(eng->q35.cont_scores); CUDA_FREE(eng->q35.cont_p);
    for (int l = 0; l < GEMMA4_CAP_LAYERS; l++)
        for (int p = 0; p < 12; p++) {
            CUDA_FREE(eng->q35.wc[l][p]);
            CUDA_FREE(eng->q35.fp4_w[l][p]); CUDA_FREE(eng->q35.fp4_wsc[l][p]);
        }
    CUDA_FREE(eng->q35.fp4_gsw);
    for (int l=0; l<GEMMA4_CAP_LAYERS; l++) CUDA_FREE(eng->q35.jspace_J[l]);
    CUDA_FREE(eng->q35.jspace_hidden); CUDA_FREE(eng->q35.jspace_transport);
    CUDA_FREE(eng->q35.jspace_norm); CUDA_FREE(eng->q35.jspace_logits);
    CUDA_FREE(eng->q35.jspace_dirs); CUDA_FREE(eng->q35.jspace_steer_mask);
    CUDA_FREE(eng->q35.jspace_steer_strength);
    if (eng->q35.fp8_scale_tab) {
        for (int i = 0; i < eng->q35.fp8_scale_n; i++) if (eng->q35.fp8_scale_tab[i].s) cudaFree((void*)eng->q35.fp8_scale_tab[i].s);
        free(eng->q35.fp8_scale_tab); eng->q35.fp8_scale_tab = NULL; eng->q35.fp8_scale_n = 0;
    }
    CUDA_FREE(eng->d_suppress);
    CUDA_FREE(eng->d_w_attn_norm);
    CUDA_FREE(eng->d_w_post_attn_norm);
    CUDA_FREE(eng->d_w_ffn_norm);
    CUDA_FREE(eng->d_w_post_ffn_norm);
    CUDA_FREE(eng->d_w_q_norm);
    CUDA_FREE(eng->d_w_k_norm);
    CUDA_FREE(eng->d_w_out_norm);
    CUDA_FREE(eng->d_weights);
    CUDA_FREE(eng->d_weights_packed);
    CUDA_FREE(eng->d_token_embd);
    for (int i = 0; i < eng->n_wt_override; i++) {
        unsigned char *p = eng->wt_override_ptr[i];
        if (p) { cudaFree(p); eng->wt_override_ptr[i] = NULL; }
    }
    eng->n_wt_override = 0;
    for (int b = 0; b < 2; b++)
        for (int p = 0; p < 7; p++)
            CUDA_FREE(eng->d_bf16_layer[b][p]);
    for (int p = 0; p < 12; p++)
        CUDA_FREE(eng->d_sb[p]);
    CUDA_FREE(eng->d_specxt);
    CUDA_FREE(eng->d_sample_id);
    CUDA_FREE(eng->d_sample_p);
    CUDA_FREE(eng->d_spec_ids);
    CUDA_FREE(eng->d_spec_rnd);
    CUDA_FREE(eng->d_pen_cnt);
    CUDA_FREE(eng->d_pen_hist);
    CUDA_FREE(eng->d_pen_batch);
    CUDA_FREE(eng->d_mtp_ids);
    CUDA_FREE(eng->d_mtp_conf);
    CUDA_FREE(eng->d_mtp_tok);
    CUDA_FREE(eng->d_mtp_pos);
    CUDA_FREE(eng->d_mtp_h_draft);
    CUDA_FREE(eng->d_mtpb_xh);  CUDA_FREE(eng->d_mtpb_cur); CUDA_FREE(eng->d_mtpb_t1);
    CUDA_FREE(eng->d_mtpb_t2);  CUDA_FREE(eng->d_mtpb_q);   CUDA_FREE(eng->d_mtpb_attn);
    CUDA_FREE(eng->d_mtpb_ffa); CUDA_FREE(eng->d_mtpb_ffb); CUDA_FREE(eng->d_mtpb_h);
    CUDA_FREE(eng->d_mtpb_ids); CUDA_FREE(eng->d_mtpb_conf);
    CUDA_FREE(eng->d_mtpb_tok); CUDA_FREE(eng->d_mtpb_pos);
    if (eng->mtp_graph) { cudaGraphExecDestroy(eng->mtp_graph); eng->mtp_graph = NULL; }
    if (eng->mtp_paged_graph) { cudaGraphExecDestroy(eng->mtp_paged_graph); eng->mtp_paged_graph = NULL; }
    CUDA_FREE(eng->d_qx);
    CUDA_FREE(eng->d_dx);
    CUDA_FREE(eng->d_sx);
    CUDA_FREE(eng->d_qx_b);
    CUDA_FREE(eng->d_dx_b);
    CUDA_FREE(eng->d_sx_b);
    CUDA_FREE(eng->d_pf_qx);
    CUDA_FREE(eng->d_pf_dx);
    CUDA_FREE(eng->d_pf_sx);
    #undef CUDA_FREE


    if (eng->cublas) cublasDestroy(eng->cublas);
    // NVFP4 prefill resources
    if (eng->fp4_ready) {
        for (int l=0;l<GEMMA4_CAP_LAYERS;l++) for (int p=0;p<PJ_COUNT;p++){
            if (eng->d_fp4_w[l][p]) cudaFree(eng->d_fp4_w[l][p]);
            if (eng->d_fp4_wsc[l][p]) cudaFree(eng->d_fp4_wsc[l][p]);
            if (eng->d_fp4_wsc_lin[l][p]) cudaFree(eng->d_fp4_wsc_lin[l][p]); }
        if (eng->d_fp4_gsw)    cudaFree(eng->d_fp4_gsw);
        if (eng->d_fp4_act)    cudaFree(eng->d_fp4_act);
        if (eng->d_fp4_actsc)  cudaFree(eng->d_fp4_actsc);
        if (eng->d_fp4_actlin) cudaFree(eng->d_fp4_actlin);
        if (eng->d_fp4_gsact)  cudaFree(eng->d_fp4_gsact);
        if (eng->d_fp4_alpha)  cudaFree(eng->d_fp4_alpha);
        if (eng->d_fp4_amax)   cudaFree(eng->d_fp4_amax);
        if (eng->d_fp4_ws)     cudaFree(eng->d_fp4_ws);
        if (eng->fp4_desc)     cublasLtMatmulDescDestroy(eng->fp4_desc);
        if (eng->cublaslt)     cublasLtDestroy(eng->cublaslt);
    }
    // BF16 embed / LM head (FORMAT_NVFP4). Free the head ONLY if it is a distinct allocation —
    // when the head is tied it aliases d_embed_bf16, so freeing both would double-free.
    if (eng->d_lmhead_bf16 && eng->d_lmhead_bf16 != eng->d_embed_bf16)
        cudaFree(eng->d_lmhead_bf16);
    // FP8-quantized untied head (when the accuracy gate enabled it d_lmhead_bf16 was freed → NULL).
    // Never aliases d_embed_bf16 (untied), so no double-free guard needed.
    if (eng->d_lmhead_fp8)       cudaFree(eng->d_lmhead_fp8);
    if (eng->d_lmhead_q8)        cudaFree(eng->d_lmhead_q8);
    if (eng->d_head_cand)        cudaFree(eng->d_head_cand);
    if (eng->d_head_cnt)         cudaFree(eng->d_head_cnt);
    if (eng->d_lmhead_fp8_scale) cudaFree(eng->d_lmhead_fp8_scale);
    if (eng->d_embed_bf16) cudaFree(eng->d_embed_bf16);
    if (eng->ev_start) cudaEventDestroy(eng->ev_start);
    if (eng->ev_stop)  cudaEventDestroy(eng->ev_stop);
    if (eng->mtp.d_w)   cudaFree(eng->mtp.d_w);
    if (eng->d_mtp_h)   cudaFree(eng->d_mtp_h);
    if (eng->d_mtp_xh)  cudaFree(eng->d_mtp_xh);
    if (eng->d_mtp_cur) cudaFree(eng->d_mtp_cur);
    if (eng->d_mtp_t1)  cudaFree(eng->d_mtp_t1);
    if (eng->d_mtp_t2)  cudaFree(eng->d_mtp_t2);
    if (eng->d_mtp_q)   cudaFree(eng->d_mtp_q);
    if (eng->d_mtp_attn) cudaFree(eng->d_mtp_attn);
    if (eng->d_mtp_ffa) cudaFree(eng->d_mtp_ffa);
    if (eng->d_mtp_ffb) cudaFree(eng->d_mtp_ffb);
    if (eng->graph.e) cudaGraphExecDestroy(eng->graph.e);
    if (eng->graph.g) cudaGraphDestroy(eng->graph.g);
    if (eng->decode_graph) cudaGraphExecDestroy(eng->decode_graph);
    if (eng->d_decpos) cudaFree(eng->d_decpos);
    if (eng->d_dectok) cudaFree(eng->d_dectok);
    for (int k = 0; k <= GEMMA4_SPEC_MAX; k++)
        if (eng->batched_graph[k]) cudaGraphExecDestroy(eng->batched_graph[k]);
    if (eng->d_specpos) cudaFree(eng->d_specpos);
    for (int b = 0; b <= GEMMA4_MAX_SEQS; b++)
        if (eng->multiseq_graph[b]) cudaGraphExecDestroy(eng->multiseq_graph[b]);
    cudaFree(eng->d_pf_x); cudaFree(eng->d_pf_norm);
    cudaFree(eng->d_pf_q); cudaFree(eng->d_pf_k); cudaFree(eng->d_pf_v);
    cudaFree(eng->d_pf_attn); cudaFree(eng->d_pf_gate); cudaFree(eng->d_pf_up);
    cudaFree(eng->d_pf_scores);
    cudaFree(eng->d_pf_inb); cudaFree(eng->d_pf_qb);
    cudaFree(eng->d_pf_kb); cudaFree(eng->d_pf_vb);
    cudaFree(eng->d_pf_kbx); cudaFree(eng->d_pf_vbx); cudaFree(eng->d_pf_pb);
    cudaFree(eng->d_pf_wpos); cudaFree(eng->d_pf_views_slid); cudaFree(eng->d_pf_views_glob);
    cudaFree(eng->d_fp_qb); cudaFree(eng->d_fp_kbx); cudaFree(eng->d_fp_vbx);
    cudaFree(eng->d_fp_kt); cudaFree(eng->d_fp_vt); cudaFree(eng->d_fp_pb);
    cudaFree(eng->d_fp_st); cudaFree(eng->d_fp_m); cudaFree(eng->d_fp_l);
    for (int b = 0; b < 2; b++) {
        if (eng->h_kv_stage[b])  cudaFreeHost(eng->h_kv_stage[b]);
        if (eng->ev_kv_stage[b]) cudaEventDestroy(eng->ev_kv_stage[b]);
    }
    if (eng->stream) cudaStreamDestroy(eng->stream);
    if (eng->dq_stream) cudaStreamDestroy(eng->dq_stream);
    for (int b = 0; b < 2; b++) {
        if (eng->ev_dq_done[b]) cudaEventDestroy(eng->ev_dq_done[b]);
        if (eng->ev_gemm_done[b]) cudaEventDestroy(eng->ev_gemm_done[b]);
    }

    free(eng);
}

// ─── Weight access helpers ────────────────────────────────────────────

// Tensor offsets stored in eng->tensors are ABSOLUTE file byte offsets
// (gguf_find_tensor already adds the tensor-data start). When the weights
// were copied to device memory (d_weights), return a device pointer into that
// buffer; otherwise fall back to the mmap'd file (GB10 unified memory).
static inline const unsigned char* weight_fp8(
    const gemma4_engine_t *eng, uint64_t tensor_offset)
{
    // QAT Q4_0 model: token_embd (and the tied LM head, same offset) live in the
    // separately-converted Q8_0 buffer, not the in-blob Q6_K bytes.
    if (eng->d_token_embd && tensor_offset == eng->tensors.token_embd)
        return eng->d_token_embd;
    // Unsloth UD dynamic-quant: off-format bulk tensors (e.g. Q4_1 ffn_down) were
    // requantized to Q4_0 into their own buffers — read those instead of d_weights.
    // n_wt_override is 0 for the 12B model, so this is a single branch there.
    for (int i = 0; i < eng->n_wt_override; i++)
        if (eng->wt_override_off[i] == tensor_offset)
            return eng->wt_override_ptr[i];
    if (eng->d_weights)
        return (const unsigned char *)(eng->d_weights
                                       + (tensor_offset - eng->tdata_start));
    return (const unsigned char *)(eng->gguf_data + tensor_offset);
}

// ─── Format-dispatching GEMV / embed launchers ─────────────────────────
#define FMT(eng)  ((int)(eng)->format)

// FUCINA_PACKED gate (built eagerly at create when FUCINA_PACKED=1, so this is a pure read —
// no allocation in the hot/graph-captured path). Only Q4_0 layer projections live in
// d_weights, so the fmt==Q4_0 guard already excludes the (Q8_0) token_embd/LM-head; the
// pointer-range check is belt-and-suspenders. A projection's packed base is at the same
// offset into d_weights_packed as the native weight is into d_weights.
static inline bool use_packed_q4(const gemma4_engine_t *eng, int fmt, const uint8_t *weight) {
    // The weight must lie INSIDE the d_weights blob for `weight - d_weights` to index
    // d_weights_packed correctly. Unsloth UD ffn_down overrides are SEPARATE cudaMalloc'd
    // Q4_0 buffers (NOT in d_weights and NOT repacked), so the upper bound excludes them —
    // they take the native dp4a MMVQ path. (12B has no overrides; the bound is a no-op there.)
    return eng->packed_ready && fmt == FORMAT_Q4_0 &&
           eng->d_weights && weight >= eng->d_weights &&
           weight < eng->d_weights + (eng->gguf_size - eng->tdata_start);
}

// Q4_K analogue: the packed kernel reads the SAME 144-B superblock IN PLACE in d_weights (no
// second copy), so the only gate is "this weight was repacked". build_packed_q4k repacks EVERY
// Q4_K tensor that lives inside d_weights, so the d_weights pointer-range check is sound (every
// in-blob Q4_K weight is packed; off-blob overrides and the converted Q8_0 head are excluded).
static inline bool use_packed_q4k(const gemma4_engine_t *eng, int fmt, const uint8_t *weight) {
    return eng->q4k_packed && fmt == FORMAT_Q4_K &&
           eng->d_weights && weight >= eng->d_weights &&
           weight < eng->d_weights + (eng->gguf_size - eng->tdata_start);
}

// Q4_0-aware single-token MMVQ over an ALREADY-quantized activation (qx/dx/sx, K=1 layout):
// routes to the packed kernel when FUCINA_PACKED is active, else the native dp4a path. Used by
// the call sites that pre-quantize once and call mmvq_launch directly (decode_layer q/k/v).
static inline void mmvq_q4aware(
    const gemma4_engine_t *eng, float *out, const uint8_t *weight,
    const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim, int fmt, cudaStream_t stream)
{
    if (fmt == FORMAT_Q6_K) { mmvq_q6_k_launch(out, weight, qx, dx, in_dim, out_dim, stream); return; }
    if (fmt == FORMAT_Q4_K) {
        if (use_packed_q4k(eng, fmt, weight))
            mmvq_q4_k_packed_launch(out, weight, qx, dx, sx, in_dim, out_dim, stream);
        else
            mmvq_q4_k_launch(out, weight, qx, dx, sx, in_dim, out_dim, stream);
        return;
    }
    if (use_packed_q4(eng, fmt, weight)) {
        const uint8_t  *q = eng->d_weights_packed + (weight - eng->d_weights);
        const uint16_t *s = (const uint16_t *)(q + (size_t)out_dim * (in_dim >> 5) * 16);
        mmvq_q4_0_packed_batched_launch(out, q, s, qx, dx, sx, in_dim, out_dim, /*K=*/1, stream);
    } else {
        mmvq_launch(out, weight, qx, dx, sx, in_dim, out_dim, fmt, stream);
    }
}

// `wfmt` overrides the weight format (default −1 = engine format). The mixed QAT
// model passes FORMAT_Q8_0 for the converted token_embd/LM-head while layers are Q4_0.
static inline const __nv_bfloat16 *wscale_fp8(const gemma4_engine_t *eng, const uint8_t *w);  // fwd

static inline void gemv_w(
    const gemma4_engine_t *eng,
    float *out, const uint8_t *weight, const float *x,
    int in_dim, int out_dim, cudaStream_t stream, int wfmt = -1)
{
    int fmt = (wfmt < 0) ? FMT(eng) : wfmt;
    if (fmt == FORMAT_FP8_BLOCK) {   // block-FP8 (Qwen3.5-9B FP8): float activation, per-128 scale
        fp8_block_gemv_launch(out, weight, wscale_fp8(eng, weight), x, in_dim, out_dim, stream);
        return;
    }
    // Quantize the activation to int8 (+ per-block Σ for the Q4_0 −8 fold), then
    // warp-per-row dp4a MMVQ (llama.cpp-parity bandwidth, no block sync). Step 5.
    // The native Q6_K LM head rides the same int8 activation (its −32 is folded into
    // the weight values, so it needs only qx/dx).
    quantize_q8_1_kernel<<<in_dim/32, 32, 0, stream>>>(
        x, eng->d_qx, eng->d_dx, eng->d_sx, in_dim);
    if (fmt == FORMAT_Q6_K) {
        mmvq_q6_k_launch(out, weight, eng->d_qx, eng->d_dx, in_dim, out_dim, stream);
    } else if (fmt == FORMAT_Q4_K) {     // Q4_K (asymmetric → needs Σx = d_sx); packed when in-place repacked
        if (use_packed_q4k(eng, fmt, weight))
            mmvq_q4_k_packed_launch(out, weight, eng->d_qx, eng->d_dx, eng->d_sx, in_dim, out_dim, stream);
        else
            mmvq_q4_k_launch(out, weight, eng->d_qx, eng->d_dx, eng->d_sx, in_dim, out_dim, stream);
    } else if (use_packed_q4(eng, fmt, weight)) {
        const uint8_t  *q = eng->d_weights_packed + (weight - eng->d_weights);
        const uint16_t *s = (const uint16_t *)(q + (size_t)out_dim * (in_dim >> 5) * 16);
        mmvq_q4_0_packed_batched_launch(out, q, s, eng->d_qx, eng->d_dx, eng->d_sx,
                                        in_dim, out_dim, /*K=*/1, stream);
    } else {
        mmvq_launch(out, weight, eng->d_qx, eng->d_dx, eng->d_sx, in_dim, out_dim, fmt, stream);
    }
}

static inline int weight_ref_format(const WeightRef &weight) {
    switch (weight.encoding) {
        case WeightEncoding::Q8_0:          return FORMAT_Q8_0;
        case WeightEncoding::Q4_0:          return FORMAT_Q4_0;
        case WeightEncoding::Q4_K:          return FORMAT_Q4_K;
        case WeightEncoding::Q6_K:          return FORMAT_Q6_K;
        case WeightEncoding::FP8_BLOCK_128: return FORMAT_FP8_BLOCK;
        case WeightEncoding::NVFP4_LINEAR:
        case WeightEncoding::NVFP4_SWIZZLED:return FORMAT_NVFP4;
        default:                            return -1;
    }
}

// Descriptor dispatch for migrated projections. Unlike the compatibility overload above, this
// takes scale and packed-layout decisions directly from the tensor: no ptr→scale search and no
// pointer-range format inference occur in the hot path.
static inline void gemv_w(
    const gemma4_engine_t *eng, float *out, const WeightRef &weight,
    const float *x, cudaStream_t stream)
{
    const int fmt = weight_ref_format(weight);
    if (fmt == FORMAT_FP8_BLOCK) {
        fp8_block_gemv_launch(out, weight.data, (const __nv_bfloat16 *)weight.scale, x,
                              weight.in_dim, weight.out_dim, stream);
        return;
    }
    if (fmt == FORMAT_Q4_K) {
        quantize_q8_1_kernel<<<weight.in_dim/32, 32, 0, stream>>>(
            x, eng->d_qx, eng->d_dx, eng->d_sx, weight.in_dim);
        if (weight.layout == TensorLayout::Q4K_PACKED)
            mmvq_q4_k_packed_launch(out, weight.data, eng->d_qx, eng->d_dx, eng->d_sx,
                                    weight.in_dim, weight.out_dim, stream);
        else
            mmvq_q4_k_launch(out, weight.data, eng->d_qx, eng->d_dx, eng->d_sx,
                             weight.in_dim, weight.out_dim, stream);
        return;
    }
    // Compatibility for encodings not yet migrated; data and dimensions still come from WeightRef.
    gemv_w(eng, out, weight.data, x, weight.in_dim, weight.out_dim, stream, fmt);
}

// FORMAT_FP8_BLOCK: recover a weight's per-128 block-scale pointer (host binary search over the
// sorted ptr→scale table the FP8 loader built — compatibility paths only).
static inline const __nv_bfloat16 *wscale_fp8(const gemma4_engine_t *eng, const uint8_t *w) {
    int lo = 0, hi = eng->q35.fp8_scale_n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1; const uint8_t *mw = eng->q35.fp8_scale_tab[mid].w;
        if (mw == w) return eng->q35.fp8_scale_tab[mid].s;
        if (mw < w) lo = mid + 1; else hi = mid - 1;
    }
    return nullptr;
}

// Batched GEMV: Y[K][out_dim] = X[K][in_dim] · weightᵀ, weight read once for all K.
// One activation-quant pass over the whole [K × in_dim] block, then weight-row-reuse
// dp4a MMVQ (Q8_0/Q4_0) or the fp32 Q6_K head kernel. K ≤ GEMMA4_SPEC_MAX (callers).
static inline void gemv_batched_w(
    const gemma4_engine_t *eng,
    float *out, const uint8_t *weight, const float *x,
    int in_dim, int out_dim, int K, cudaStream_t stream, int wfmt = -1)
{
    int fmt = (wfmt < 0) ? FMT(eng) : wfmt;
    if (fmt == FORMAT_FP8_BLOCK) {        // block-FP8: float activation directly, no Q8_1 quant
        fp8_block_gemm_launch(out, weight, wscale_fp8(eng, weight), x, in_dim, out_dim, K, stream);
        return;
    }
    // Packed Q4_K at K>1: quantize straight into the transposed activation layout and use the
    // dense-load mixer kernel (measured 2-2.6× on the batched decode GEMVs, bit-identical).
    // K==1 keeps the row-major kernel (already dense enough at one token; measured faster).
    if (fmt == FORMAT_Q4_K && K > 1 && (in_dim & 1023) == 0 && use_packed_q4k(eng, fmt, weight)) {
        quantize_q8_1t_kernel<<<(K*in_dim)/32, 32, 0, stream>>>(
            x, eng->d_qx_b, eng->d_dx_b, eng->d_sx_b, in_dim);
        mmvq_q4_k_packedT_batched_launch(out, weight, eng->d_qx_b, eng->d_dx_b, eng->d_sx_b,
                                         in_dim, out_dim, K, stream);
        return;
    }
    quantize_q8_1_kernel<<<(K*in_dim)/32, 32, 0, stream>>>(
        x, eng->d_qx_b, eng->d_dx_b, eng->d_sx_b, K*in_dim);
    if (fmt == FORMAT_Q6_K) {            // native Q6_K LM head (Step 8), batched over K rows
        mmvq_q6_k_batched_launch(out, weight, eng->d_qx_b, eng->d_dx_b,
                                 in_dim, out_dim, K, stream);
        return;
    }
    if (fmt == FORMAT_Q4_K) {            // Q4_K, batched over K rows (needs Σx); packed when in-place repacked
        if (use_packed_q4k(eng, fmt, weight))
            mmvq_q4_k_packed_batched_launch(out, weight, eng->d_qx_b, eng->d_dx_b, eng->d_sx_b,
                                            in_dim, out_dim, K, stream);
        else
            mmvq_q4_k_batched_launch(out, weight, eng->d_qx_b, eng->d_dx_b, eng->d_sx_b,
                                     in_dim, out_dim, K, stream);
        return;
    }
    if (use_packed_q4(eng, fmt, weight)) {   // FUCINA_PACKED: coalesced uint4 weight loads
        const uint8_t  *q = eng->d_weights_packed + (weight - eng->d_weights);
        const uint16_t *s = (const uint16_t *)(q + (size_t)out_dim * (in_dim >> 5) * 16);
        mmvq_q4_0_packed_batched_launch(out, q, s, eng->d_qx_b, eng->d_dx_b, eng->d_sx_b,
                                        in_dim, out_dim, K, stream);
        return;
    }
    mmvq_batched_launch(out, weight, eng->d_qx_b, eng->d_dx_b, eng->d_sx_b,
                        in_dim, out_dim, K, fmt, stream);
}

static inline void gemv_batched_w(
    const gemma4_engine_t *eng, float *out, const WeightRef &weight,
    const float *x, int K, cudaStream_t stream)
{
    const int fmt = weight_ref_format(weight);
    if (fmt == FORMAT_FP8_BLOCK) {
        fp8_block_gemm_launch(out, weight.data, (const __nv_bfloat16 *)weight.scale, x,
                              weight.in_dim, weight.out_dim, K, stream);
        return;
    }
    if (fmt == FORMAT_Q4_K) {
        if (K > 1 && (weight.in_dim & 1023) == 0 &&
            weight.layout == TensorLayout::Q4K_PACKED) {
            quantize_q8_1t_kernel<<<(K*weight.in_dim)/32, 32, 0, stream>>>(
                x, eng->d_qx_b, eng->d_dx_b, eng->d_sx_b, weight.in_dim);
            mmvq_q4_k_packedT_batched_launch(out, weight.data, eng->d_qx_b, eng->d_dx_b,
                                             eng->d_sx_b, weight.in_dim, weight.out_dim, K, stream);
            return;
        }
        quantize_q8_1_kernel<<<(K*weight.in_dim)/32, 32, 0, stream>>>(
            x, eng->d_qx_b, eng->d_dx_b, eng->d_sx_b, K*weight.in_dim);
        if (weight.layout == TensorLayout::Q4K_PACKED)
            mmvq_q4_k_packed_batched_launch(out, weight.data, eng->d_qx_b, eng->d_dx_b,
                                             eng->d_sx_b, weight.in_dim, weight.out_dim, K, stream);
        else
            mmvq_q4_k_batched_launch(out, weight.data, eng->d_qx_b, eng->d_dx_b,
                                      eng->d_sx_b, weight.in_dim, weight.out_dim, K, stream);
        return;
    }
    gemv_batched_w(eng, out, weight.data, x, weight.in_dim, weight.out_dim, K, stream, fmt);
}

// Forward declarations: logits_head uses lmhead_w/embd_fmt (defined just below); the decode
// NVFP4 routing uses nvfp4_decode_proj (defined with the NVFP4 residency, far below).
static inline const unsigned char* lmhead_w(const gemma4_engine_t *eng);
static inline int embd_fmt(const gemma4_engine_t *eng);
static inline void nvfp4_decode_proj(gemma4_engine_t *eng, int l, int p,
        const float *x, float *y, int in_dim, int out_dim, cudaStream_t st);

static inline void embed_w(
    const gemma4_engine_t *eng,
    float *out, const uint8_t *table, const int32_t *tokens,
    int batch, int hidden_size, cudaStream_t stream, int efmt = -1)
{
    (void)efmt;   // the token_embd table is always Q8_0 (native, or converted from Q6_K)
    // NVFP4 always, and FP8/safetensors when the BF16 table is resident: use it (the per-step token
    // embedding feeds the recurrent GDN state, so Q8_0 rounding compounds over decode steps).
    if (eng->format == FORMAT_NVFP4 ||
        (eng->format == FORMAT_FP8_BLOCK && eng->d_embed_bf16)) {  // BF16 table (`table` unused)
        embed_lookup_bf16_kernel<<<batch, 256, 0, stream>>>(
            out, eng->d_embed_bf16, tokens, batch, hidden_size);
        return;
    }
    embed_lookup_q8_0_kernel<<<batch, 256, 0, stream>>>(
        out, table, tokens, batch, hidden_size);
}

// LM-head logits for FORMAT_NVFP4 (BF16 head GEMV), else the standard Q4_0/Q8_0/Q6_K MMVQ via
// gemv_w. One call shape for every single-token / last-token logits site (prefill + decode).
static inline void logits_head(
    const gemma4_engine_t *eng, float *logits, const float *x,
    int in_dim, int out_dim, cudaStream_t stream)
{
    if (eng->format == FORMAT_NVFP4) {
        if (eng->d_lmhead_fp8)   // accuracy-gated FP8 head: 1 B/elem (half the BF16 head read)
            fp8_head_gemv_launch(logits, eng->d_lmhead_fp8, eng->d_lmhead_fp8_scale, x, in_dim, out_dim, stream);
        else
            bf16_head_gemv_launch(logits, eng->d_lmhead_bf16, x, in_dim, out_dim, stream);
    } else
        gemv_w(eng, logits, lmhead_w(eng), x, in_dim, out_dim, stream, embd_fmt(eng));
}

// LM-head weight pointer + format. Step 8: when the tied head is kept NATIVE Q6_K
// (eng->lmhead_q6k, set at create), the output projection reads the raw Q6_K bytes from the
// device weight blob (d_lmhead_q6k) as FORMAT_Q6_K — cutting ~0.24 GB/token off the V×H read.
static inline int lmhead_native_q6k(const gemma4_engine_t *eng) {
    return eng->lmhead_q6k && eng->d_lmhead_q6k;
}
static inline int lmhead_native_q4k(const gemma4_engine_t *eng) {
    return eng->lmhead_q4k && eng->d_lmhead_q4k;
}
static inline const unsigned char* lmhead_w(const gemma4_engine_t *eng) {
    if (lmhead_native_q6k(eng)) return eng->d_lmhead_q6k;
    if (lmhead_native_q4k(eng)) return eng->d_lmhead_q4k;
    uint64_t off = eng->tensors.output_weight;
    return weight_fp8(eng, off);
}
static inline int embd_fmt(const gemma4_engine_t *eng) {
    if (lmhead_native_q6k(eng)) return (int)FORMAT_Q6_K;
    if (lmhead_native_q4k(eng)) return (int)FORMAT_Q4_K;
    // Untied head (e.g. Qwen3): use the head's own recorded format (Q6_K for Qwen3 Q4_K_M),
    // NOT the bulk engine format (Q4_K) which would mis-decode the Q6_K head bytes.
    if (!eng->output_tied) return (int)eng->tensors.output_fmt;
    return (eng->format == FORMAT_Q4_0 && eng->d_token_embd) ? (int)FORMAT_Q8_0 : FMT(eng);
}

// GQA-broadcast global flash-decode for a single query token (DECODE-30-35 Step 1). Drop-in
// replacement for global_attn_decode_kernel<<<n_heads,head_dim>>> at both the single-decode
// and spec-verify launch sites: same (out, q, kc, vc, n_heads, head_dim, ctx_len) contract,
// but each global K/V tile is read from DRAM ONCE (not n_heads×). Splits the context across
// up to GEMMA4_GLOBAL_MAX_SPLITS blocks so one KV head still saturates bandwidth, then merges
// the partials. Uses engine-resident scratch (d_fa_acc/m/l). n_heads must be GEMMA4_HEADS.
static inline void global_attn_decode_broadcast(
    gemma4_engine_t *eng, float *out, const float *q,
    const kv_t *kc, const kv_t *vc, int n_heads, int head_dim, int ctx_len,
    cudaStream_t stream)
{
    // Warp-per-head kernel: NH warps/block (512 threads at NH=16), head_dim is the
    // compile-time HD template param (global layers are always GEMMA4_GLOBAL_HEAD_DIM).
    // No shared memory, no __syncthreads — see kernel comment.
    int splits = (ctx_len + GEMMA4_GLOBAL_SPLIT_CHUNK - 1) / GEMMA4_GLOBAL_SPLIT_CHUNK;
    if (splits < 1) splits = 1;
    if (splits > GEMMA4_GLOBAL_MAX_SPLITS) splits = GEMMA4_GLOBAL_MAX_SPLITS;
    if (splits > ctx_len) splits = ctx_len;       // never launch an empty split
    // Dispatch on the runtime config: 12B = 16 heads / 1 global KV (broadcast),
    // 31B = 32 heads / 4 global KV (GQA fan-out 8). head_dim is constant (512).
    if (eng->cfg.n_heads == 32 && eng->cfg.n_kv_global == 4) {
        global_attn_splitk_kernel<32, 4, GEMMA4_GLOBAL_HEAD_DIM>
            <<<splits, 32*32, 0, stream>>>(
            eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, q, kc, vc, head_dim, ctx_len, splits);
        flash_decode_combine_kernel<32><<<n_heads, head_dim, 0, stream>>>(
            out, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, head_dim, splits);
    } else {
        global_attn_splitk_kernel<GEMMA4_HEADS, 1, GEMMA4_GLOBAL_HEAD_DIM>
            <<<splits, GEMMA4_HEADS*32, 0, stream>>>(
            eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, q, kc, vc, head_dim, ctx_len, splits);
        flash_decode_combine_kernel<GEMMA4_HEADS><<<n_heads, head_dim, 0, stream>>>(
            out, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, head_dim, splits);
    }
}

// Split-K sliding flash-decode for a single query token. Drop-in replacement for
// sliding_attn_decode_kernel<<<n_heads, head_dim>>> at all three decode launch sites
// (decode_layer, spec-verify rows, MTP drafter — the latter passes window-1 / n_tokens=pos,
// honored generically): same (out, q, kc, vc, n_heads, n_kv_heads, head_dim, window,
// n_tokens) contract, but NO __syncthreads in the key loop (warp-owned register slices,
// see sliding_attn_splitk_kernel) and the window split across blocks. n_heads must be
// GEMMA4_HEADS, n_kv_heads GEMMA4_KV_HEADS and head_dim GEMMA4_HEAD_DIM (all template-fixed:
// head_dim too, so the per-lane slice loops unroll and stay in registers — see kernel note).
//
// SCRATCH REUSE (d_fa_acc/m/l, shared with the global path): the buffers hold
// GEMMA4_GLOBAL_MAX_SPLITS(128) × GEMMA4_HEADS(16) × GEMMA4_GLOBAL_HEAD_DIM(512) floats.
// Sliding needs n_splits × 16 × head_dim(256); n_splits is clamped ≤ GEMMA4_GLOBAL_MAX_SPLITS
// below, so even the degenerate clamp case uses ≤ half the acc buffer (m/l: identical
// [splits][16] shape, capacity 128×16 ≥ n_splits×16). Sharing is race-free because every
// attention call in the engine — decode_layer's per-layer calls, the spec-verify per-row
// loop, and mtp_forward — issues on the SAME stream: each splitk+combine pair fully
// consumes its partials before the next pair's splitk overwrites them.
static inline void sliding_attn_decode_broadcast(
    gemma4_engine_t *eng, float *out, const float *q,
    const kv_t *kc, const kv_t *vc, int n_heads, int n_kv_heads, int head_dim,
    int window, int n_tokens, cudaStream_t stream)
{
    (void)n_kv_heads;  // GEMMA4_KV_HEADS by contract (warps/block below)
    int window_len = (n_tokens < window) ? n_tokens : window;
    int splits = (window_len + GEMMA4_SLIDING_SPLIT_CHUNK - 1) / GEMMA4_SLIDING_SPLIT_CHUNK;
    if (splits < 1) splits = 1;   // window_len==0 ⇒ one empty split ⇒ combine writes zeros
    if (splits > GEMMA4_GLOBAL_MAX_SPLITS) splits = GEMMA4_GLOBAL_MAX_SPLITS; // scratch cap
    // Dispatch on the runtime config: 12B = 16 heads / 8 sliding KV, 31B = 32 / 16.
    if (eng->cfg.n_heads == 32 && eng->cfg.n_kv_sliding == 16) {
        sliding_attn_splitk_kernel<32, 16, GEMMA4_HEAD_DIM>
            <<<splits, 16 * 32, 0, stream>>>(
                eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, q, kc, vc,
                window, n_tokens, splits, eng->sliding_kv_capacity);
        flash_decode_combine_kernel<32><<<n_heads, head_dim, 0, stream>>>(
            out, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, head_dim, splits);
    } else {
        sliding_attn_splitk_kernel<GEMMA4_HEADS, GEMMA4_KV_HEADS, GEMMA4_HEAD_DIM>
            <<<splits, GEMMA4_KV_HEADS * 32, 0, stream>>>(
                eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, q, kc, vc,
                window, n_tokens, splits, eng->sliding_kv_capacity);
        flash_decode_combine_kernel<GEMMA4_HEADS><<<n_heads, head_dim, 0, stream>>>(
            out, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, head_dim, splits);
    }
}

// ═════════════════════════════════════════════════════════════════════════
// =========================================================================
// ─── Single Layer Forward (Decode, B=1) ─────────────────────────────────
// =========================================================================

// Forward one layer for a single token decode
// Handles both sliding and global layers

// Per-head RMSNorm using a norm weight already resident on device (or NULL).
#define PER_HEAD_NORM(dst, dev_weight, n_h, h_dim) do { \
    per_head_rms_norm_kernel<<<(n_h), (h_dim), smem32, stream>>>( \
        (dst), (dev_weight), (h_dim), GEMMA4_RMS_EPS); \
} while(0)

// Device pointer to layer `layer`'s preloaded hidden-size norm weight.
#define NORM_W(arr) (eng->arr + (size_t)layer * eng->cfg.hidden_size)
#define HEAD_NORM_W(arr) (eng->arr + (size_t)layer * GEMMA4_GLOBAL_HEAD_DIM)

static void decode_timing_lap(gemma4_engine_t *eng);   // defined with the decode path

// d_pos (default NULL) switches the layer to DEVICE-resident position state for the
// CUDA-graph decode: d_pos[0] = pos (rope, KV write), d_pos[1] = pos+1 (attention
// n_tokens). NULL keeps the host-pos path byte-identical to before. The device-pos
// attention uses the *_rows kernels (r = 0) with a FIXED grid + n_tokens0_ptr — they
// recompute the split partition in-kernel with the exact host-launcher formula, so the
// two paths produce identical partials.
// ── Paged KV mirror-write (Phase 2 inc 3) ────────────────────────────────────
// Writes element e of logical position `pos` into ONE layer's paged pool
// sub-range, resolving (block,offset) through the active sequence's device block
// table. Runs ALONGSIDE the contiguous write while the paged read path is being
// validated; once the read path flips to paged, the contiguous write is dropped.
__global__ void paged_mirror_write_kernel(
        kv_t *pool_k, kv_t *pool_v,          // this layer's pool sub-range base
        const float *kb, const float *vb,    // projected K/V [elems]
        const int *block_table, int base, int n_blocks,
        int pos, int block_tokens, int elems)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= elems) return;
    PagedSeqView v; v.block_table = block_table; v.n_blocks = n_blocks;
    v.base = base; v.n_tokens = pos + 1;
    size_t idx = paged_elem_index(v, pos, e, block_tokens, elems);
    if (idx == (size_t)-1) return;
    pool_k[idx] = pkv_float_to_fp8(kv_codec_value(kb, e));
    pool_v[idx] = pkv_float_to_fp8(kv_codec_value(vb, e));
}

// Paged single-token decode attention (GQA), scale 1.0 (gemma4), online softmax.
// One block per query head; blockDim.x = head_dim. Reads this layer's pool
// sub-range through the active sequence's block table. window>0 bounds the scan
// to the last `window` positions (sliding); window==0 attends all (global). This
// is the correctness-first paged read (mirrors sliding_/global_attn_decode_kernel);
// the split-K perf path is a later optimisation.
__global__ void paged_attn_decode_kernel(
        float *out,                  // [n_heads*head_dim]
        const float *q,              // [n_heads*head_dim]
        const kv_t *pool_k, const kv_t *pool_v,
        const int *block_table, int base, int n_blocks,
        int n_heads, int n_kv_heads, int head_dim,
        int n_tokens, int window, int block_tokens, int elems_per_token)
{
    extern __shared__ float smem[];
    int h   = blockIdx.x;            // query head
    int tid = threadIdx.x;           // 0..head_dim-1
    int group    = n_heads / n_kv_heads;     // GQA group size (sliding 2, global 16)
    int elem_off = (h / group) * head_dim;   // this query head's kv-head slice
    int lo = (window > 0 && n_tokens > window) ? (n_tokens - window) : 0;
    PagedSeqView v; v.block_table = block_table; v.n_blocks = n_blocks;
    v.base = base; v.n_tokens = n_tokens;
    float q_d = (tid < head_dim) ? q[h * head_dim + tid] : 0.0f;
    float acc = 0.0f, m = -INFINITY, l = 0.0f;
    for (int p = lo; p < n_tokens; p++) {
        float k_d = 0.0f, val_d = 0.0f;
        if (tid < head_dim) {
            size_t idx = paged_elem_index(v, p, elem_off + tid, block_tokens, elems_per_token);
            if (idx != (size_t)-1) { k_d = pkv_fp8_to_float(pool_k[idx]); val_d = pkv_fp8_to_float(pool_v[idx]); }
        }
        float s     = pkv_block_reduce_sum(q_d * k_d, smem);
        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);
        float p_w   = __expf(s - m_new);
        l   = l * alpha + p_w;
        acc = acc * alpha + p_w * val_d;
        m   = m_new;
    }
    if (tid < head_dim) out[h * head_dim + tid] = (l > 0.0f) ? acc / l : 0.0f;
}

// RoPE for B INDEPENDENT rows, each at its OWN absolute position d_row_pos[row]
// (the multi-seq batched-decode counterpart of rope_rows_kernel, which uses the
// consecutive base_pos+row of a single sequence). grid=(n_heads,rows), block=hd/2.
__global__ void rope_rows_pos_kernel(
    float *q, float *k, const int *d_row_pos, int n_heads, int n_kv_heads,
    int head_dim, int rows, float theta_base, const float *freq_factors)
{
    int d = threadIdx.x, half = head_dim / 2;
    if (d >= half) return;
    int head = blockIdx.x, row = blockIdx.y;
    if (row >= rows) return;
    int pos = d_row_pos[row];
    float ff    = freq_factors ? freq_factors[d] : 1.0f;
    float theta = powf(theta_base, -2.0f * d / head_dim) / ff;
    float c = cosf(pos * theta), s = sinf(pos * theta);
    float *qh = q + ((size_t)row * n_heads + head) * head_dim;
    float q0 = qh[d], q1 = qh[d + half];
    qh[d] = q0 * c - q1 * s;  qh[d + half] = q0 * s + q1 * c;
    if (head < n_kv_heads) {
        float *kh = k + ((size_t)row * n_kv_heads + head) * head_dim;
        float k0 = kh[d], k1 = kh[d + half];
        kh[d] = k0 * c - k1 * s;  kh[d + half] = k0 * s + k1 * c;
    }
}

// Per-row argmax over a [rows][vocab] logits matrix: one block per row, writes the
// argmax token id to out_idx[row]. Same warp-shuffle reduction as argmax_kernel,
// generalised to B rows (block.y indexes the row); blockDim.x lanes stride vocab.
// Per-row greedy argmax with the lowest-index tie-break (== torch.argmax). One BLOCK per row,
// cross-warp shared reduce: a 262k vocab row scans in ~40 µs instead of the ~1.3 ms a single
// 32-lane warp took (the old <<<rows,32>>> launch was 2.5% of every decode step).
__global__ void argmax_rows_kernel(
    const float *logits, int *out_idx, int rows, int vocab_size)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    const float *lr = logits + (size_t)row * vocab_size;
    float best_val = -1e30f; int best_idx = 0;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        if (lr[i] > best_val) { best_val = lr[i]; best_idx = i; }   // strict > keeps lowest i per lane
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_xor_sync(0xFFFFFFFF, best_val, offset);
        int   oi = __shfl_xor_sync(0xFFFFFFFF, best_idx, offset);
        if (ov > best_val || (ov == best_val && oi < best_idx)) { best_val = ov; best_idx = oi; }
    }
    __shared__ float sv[32]; __shared__ int si[32];
    int warp = tid >> 5, lane = tid & 31, nwarp = (blockDim.x + 31) >> 5;
    if (lane == 0) { sv[warp] = best_val; si[warp] = best_idx; }
    __syncthreads();
    if (warp == 0) {
        best_val = (lane < nwarp) ? sv[lane] : -1e30f;
        best_idx = (lane < nwarp) ? si[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float ov = __shfl_xor_sync(0xFFFFFFFF, best_val, offset);
            int   oi = __shfl_xor_sync(0xFFFFFFFF, best_idx, offset);
            if (ov > best_val || (ov == best_val && oi < best_idx)) { best_val = ov; best_idx = oi; }
        }
        if (lane == 0) out_idx[row] = best_idx;
    }
}

// (Re)upload a host block table's id array to device, growing the device buffer
// if the host table outgrew it. The array is small (sliding ~5, global ~ctx/256)
// and changes each step (growth/recycle), so re-uploading it whole is cheap.
static int paged_upload_blocks(PagedBlockTable *t, int **d_blocks, int *d_cap,
                               cudaStream_t stream) {
    if (t->n <= 0) return 0;
    if (*d_cap < t->n) {
        if (*d_blocks) { cudaStreamSynchronize(stream); cudaFree(*d_blocks); }
        int newcap = (t->cap > t->n) ? t->cap : t->n;
        if (cudaMalloc(d_blocks, (size_t)newcap * sizeof(int)) != cudaSuccess) {
            *d_blocks = NULL; *d_cap = 0; cudaGetLastError(); return -1;
        }
        *d_cap = newcap;
    }
    cudaMemcpyAsync(*d_blocks, t->blocks, (size_t)t->n * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    return 0;
}

// Grow/recycle the active sequence's paged block tables to cover logical position
// `pos`, then refresh the device block-id arrays. Returns 0 / -1 on pool
// exhaustion or device error. No-op when paging is disabled.
static int paged_seq_sync(gemma4_engine_t *eng, int pos) {
    if (!eng->paged_enabled) return 0;
    gemma4_seq *s = &eng->cur;
    if (paged_table_ensure(&eng->slid_pool, &s->slid_bt, pos + 1) != 0) return -1;
    paged_table_advance_sliding(&eng->slid_pool, &s->slid_bt, pos + 1, GEMMA4_SLIDING_WINDOW);
    if (paged_table_ensure(&eng->glob_pool, &s->glob_bt, pos + 1) != 0) return -1;
    if (paged_upload_blocks(&s->slid_bt, &s->d_slid_blocks, &s->d_slid_cap, eng->stream) != 0) return -1;
    if (paged_upload_blocks(&s->glob_bt, &s->d_glob_blocks, &s->d_glob_cap, eng->stream) != 0) return -1;
    return 0;
}

static int ensure_ms_scratch(gemma4_engine_t *eng);   // fwd decl (body below decode_multiseq)

static int decode_layer(
    gemma4_engine_t *eng,
    int              layer,
    int              pos,
    int              context_len,
    cudaStream_t     stream,
    const int       *d_pos = NULL)
{
    // Runtime model geometry (12B: 3840/15360; 31B: 5376/21504). On 12B these equal the
    // old HS/IM defines so the path is byte-identical.
    const int HS = eng->cfg.hidden_size;
    const int IM = eng->cfg.intermediate;
    layer_type_t ltype = eng->layer_types[layer];
    if (eng->paged_read && ensure_ms_scratch(eng) != 0) return -1;  // 1-elem view + fa scratch
    int n_heads, n_kv_heads, head_dim, out_dim_q, out_dim_kv;
    if (ltype == LAYER_SLIDING) {
        n_heads    = eng->cfg.n_heads;   head_dim = GEMMA4_HEAD_DIM;
        n_kv_heads = eng->cfg.n_kv_sliding;
    } else {
        n_heads    = eng->cfg.n_heads;   head_dim = GEMMA4_GLOBAL_HEAD_DIM;
        n_kv_heads = eng->cfg.n_kv_global;
    }
    out_dim_q  = n_heads    * head_dim;
    out_dim_kv = n_kv_heads * head_dim;

    const int block    = 256;
    const int smem32   = 32 * sizeof(float);

    // ─────────────────────────────────────────────────────────────────
    // Save the pre-layer residual once; it will be added back after attn.
    // ─────────────────────────────────────────────────────────────────
    cudaMemcpyAsync(eng->d_residual, eng->d_x,
                    HS * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // ── 1. Pre-attention RMSNorm ──────────────────────────────────────
    rms_norm_kernel<<<1, block, smem32, stream>>>(
        eng->d_norm, eng->d_x, NORM_W(d_w_attn_norm),
        HS, GEMMA4_RMS_EPS);

    // ── 2-4. Q/K/V projections — quantize the attn-norm activation ONCE ───
    // (audit #30 lever 2): the three projections share one int8 quant of d_norm instead of
    // re-quantizing it per gemv_w call. q/k/v are always the layer's Q4_0/Q8_0 format (never
    // the Q6_K head), so mmvq_launch is called directly. Bit-identical (same quant input).
    const bool nvfp4 = (eng->format == FORMAT_NVFP4 && eng->nvfp4_decode_ready);
    if (nvfp4) {
        // NVFP4 store: the fused decode GEMV reads the float activation directly — no int8
        // quant (d_qx/d_dx/d_sx are neither produced nor consumed on this path). V on global
        // layers is the K-memcpy, NOT a PJ_V GEMV (proj_desc returns 0 there) — mirror that.
        nvfp4_decode_proj(eng, layer, PJ_Q, eng->d_norm, eng->d_attn_q, HS, out_dim_q, stream);
        nvfp4_decode_proj(eng, layer, PJ_K, eng->d_norm, eng->d_attn_k, HS, out_dim_kv, stream);
        PER_HEAD_NORM(eng->d_attn_q, HEAD_NORM_W(d_w_q_norm), n_heads, head_dim);
        if (ltype == LAYER_SLIDING) {
            nvfp4_decode_proj(eng, layer, PJ_V, eng->d_norm, eng->d_attn_v, HS, out_dim_kv, stream);
            PER_HEAD_NORM(eng->d_attn_v, NULL, n_kv_heads, head_dim);
        } else {
            cudaMemcpyAsync(eng->d_attn_v, eng->d_attn_k,
                out_dim_kv * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            PER_HEAD_NORM(eng->d_attn_v, NULL, n_kv_heads, head_dim);
        }
        PER_HEAD_NORM(eng->d_attn_k, HEAD_NORM_W(d_w_k_norm), n_kv_heads, head_dim);
    } else {
    quantize_q8_1_kernel<<<HS/32, 32, 0, stream>>>(
        eng->d_norm, eng->d_qx, eng->d_dx, eng->d_sx, HS);
    mmvq_q4aware(eng, eng->d_attn_q, weight_fp8(eng, eng->tensors.layers[layer].attn_q),
        eng->d_qx, eng->d_dx, eng->d_sx, HS, out_dim_q, (int)eng->tensors.layers[layer].fmt_q, stream);
    PER_HEAD_NORM(eng->d_attn_q, HEAD_NORM_W(d_w_q_norm), n_heads, head_dim);

    // ── 3. K projection → per-head RMSNorm ───────────────────────────────
    mmvq_q4aware(eng, eng->d_attn_k, weight_fp8(eng, eng->tensors.layers[layer].attn_k),
        eng->d_qx, eng->d_dx, eng->d_sx, HS, out_dim_kv, (int)eng->tensors.layers[layer].fmt_k, stream);
    // ── 4. V projection → plain RMSNorm (no weight) ──────────────────────
    // For global layers V = K BEFORE any norm/RoPE is applied.
    // For sliding layers V comes from a separate projection.
    if (ltype == LAYER_SLIDING) {
        mmvq_q4aware(eng, eng->d_attn_v, weight_fp8(eng, eng->tensors.layers[layer].attn_v),
            eng->d_qx, eng->d_dx, eng->d_sx, HS, out_dim_kv, (int)eng->tensors.layers[layer].fmt_v, stream);
        PER_HEAD_NORM(eng->d_attn_v, NULL, n_kv_heads, head_dim);
    } else {
        // Global: V = raw K projection (before K-norm)
        cudaMemcpyAsync(eng->d_attn_v, eng->d_attn_k,
            out_dim_kv * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        // V gets plain RMSNorm (no learnable weight)
        PER_HEAD_NORM(eng->d_attn_v, NULL, n_kv_heads, head_dim);
    }

    // Now apply K-norm (after V copy for global layers)
    PER_HEAD_NORM(eng->d_attn_k, HEAD_NORM_W(d_w_k_norm), n_kv_heads, head_dim);
    } // else (non-NVFP4 q/k/v)

    // ── 5. RoPE ──────────────────────────────────────────────────────────
    if (ltype == LAYER_SLIDING) {
        rope_sliding_kernel<<<n_heads, head_dim/2, 0, stream>>>(
                eng->d_attn_q, eng->d_attn_k,
                pos, d_pos, n_heads, n_kv_heads, head_dim, 10000.0f);
    } else {
        rope_global_kernel<<<n_heads, head_dim/2, 0, stream>>>(
                eng->d_attn_q, eng->d_attn_k,
                pos, d_pos, context_len, n_heads, n_kv_heads, head_dim,
                1000000.0f, eng->d_rope_freqs);
    }

    // ── 6. Write K (and V) into KV cache (fp32 activation → FP8 cache) ────
    int kv_size = n_kv_heads * head_dim;
    unsigned kvg = (kv_size + 255) / 256;
    if (ltype == LAYER_SLIDING) {
        // FLAT (Step 3): write this token at its ABSOLUTE position `pos`.
        size_t layer_stride = (size_t)eng->sliding_kv_capacity * eng->cfg.n_kv_sliding * GEMMA4_HEAD_DIM;
        kv_t *base_k = eng->d_sliding_k + (size_t)layer * layer_stride;
        kv_t *base_v = eng->d_sliding_v + (size_t)layer * layer_stride;
        if (d_pos) {
            copy_f32_to_fp8_at_kernel<<<kvg, 256, 0, stream>>>(base_k, d_pos, kv_size, eng->d_attn_k, kv_size, eng->sliding_kv_capacity);
            copy_f32_to_fp8_at_kernel<<<kvg, 256, 0, stream>>>(base_v, d_pos, kv_size, eng->d_attn_v, kv_size, eng->sliding_kv_capacity);
        } else {
            size_t rslot = (size_t)(pos % eng->sliding_kv_capacity) * kv_size;  // ring slot
            copy_f32_to_fp8_kernel<<<kvg, 256, 0, stream>>>(base_k + rslot, eng->d_attn_k, kv_size);
            copy_f32_to_fp8_kernel<<<kvg, 256, 0, stream>>>(base_v + rslot, eng->d_attn_v, kv_size);
            // Mirror into the paged pool (sliding indexed by absolute layer id).
            if (eng->paged_enabled) {
                size_t pls = (size_t)eng->slid_pool.n_blocks * PAGED_KV_BLOCK_TOKENS * kv_size;
                paged_mirror_write_kernel<<<kvg, 256, 0, stream>>>(
                    eng->d_slid_pool_k + (size_t)layer * pls,
                    eng->d_slid_pool_v + (size_t)layer * pls,
                    eng->d_attn_k, eng->d_attn_v,
                    eng->cur.d_slid_blocks, eng->cur.slid_bt.base, eng->cur.slid_bt.n,
                    pos, PAGED_KV_BLOCK_TOKENS, kv_size);
            }
        }
    } else {
        int n = pos;   // == eng->cur.n_tokens at every call site
        int slot = eng->global_slot[layer];
        // Per-token global cache stride = n_kv_global × head_dim (kv_size). For 12B
        // (n_kv_global=1) this equals GEMMA4_GLOBAL_HEAD_DIM as before; for 31B it is
        // 4×512 so the 4 KV heads of each token live contiguous in [ctx][NKV][HD].
        size_t tok_stride   = (size_t)eng->cfg.n_kv_global * GEMMA4_GLOBAL_HEAD_DIM;
        size_t layer_stride = (size_t)eng->global_kv_capacity * tok_stride;
        kv_t *base_k = eng->d_global_k + (size_t)slot * layer_stride;
        kv_t *base_v = eng->d_global_v + (size_t)slot * layer_stride;
        if (d_pos) {
            copy_f32_to_fp8_at_kernel<<<kvg, 256, 0, stream>>>(base_k, d_pos, (int)tok_stride, eng->d_attn_k, kv_size, eng->global_kv_capacity);
            copy_f32_to_fp8_at_kernel<<<kvg, 256, 0, stream>>>(base_v, d_pos, (int)tok_stride, eng->d_attn_v, kv_size, eng->global_kv_capacity);
        } else {
            copy_f32_to_fp8_kernel<<<kvg, 256, 0, stream>>>(base_k + (size_t)n*tok_stride, eng->d_attn_k, kv_size);
            copy_f32_to_fp8_kernel<<<kvg, 256, 0, stream>>>(base_v + (size_t)n*tok_stride, eng->d_attn_v, kv_size);
            // Mirror into the paged pool (global indexed by compact global_slot).
            if (eng->paged_enabled) {
                size_t pls = (size_t)eng->glob_pool.n_blocks * PAGED_KV_BLOCK_TOKENS * kv_size;
                paged_mirror_write_kernel<<<kvg, 256, 0, stream>>>(
                    eng->d_glob_pool_k + (size_t)slot * pls,
                    eng->d_glob_pool_v + (size_t)slot * pls,
                    eng->d_attn_k, eng->d_attn_v,
                    eng->cur.d_glob_blocks, eng->cur.glob_bt.base, eng->cur.glob_bt.n,
                    pos, PAGED_KV_BLOCK_TOKENS, kv_size);
            }
        }
    }

    // ── 7. Attention ─────────────────────────────────────────────────────
    if (ltype == LAYER_SLIDING) {
        size_t lstride = (size_t)eng->sliding_kv_capacity * eng->cfg.n_kv_sliding * GEMMA4_HEAD_DIM;
        if (d_pos) {
            // Graph path: fixed-grid rows kernels (r=0) with device n_tokens = d_pos[1].
            // Max sliding splits = ceil(WINDOW/CHUNK); shorter windows tail-return.
            const int max_splits =
                (GEMMA4_SLIDING_WINDOW + GEMMA4_SLIDING_SPLIT_CHUNK - 1) / GEMMA4_SLIDING_SPLIT_CHUNK;
            if (eng->cfg.n_heads == 32 && eng->cfg.n_kv_sliding == 16) {
                sliding_attn_splitk_rows_kernel<32, 16, GEMMA4_HEAD_DIM>
                    <<<dim3(max_splits, 1), 16*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_attn_q,
                        eng->d_sliding_k + (size_t)layer * lstride,
                        eng->d_sliding_v + (size_t)layer * lstride,
                        GEMMA4_SLIDING_WINDOW, 0, eng->sliding_kv_capacity, d_pos + 1);
                flash_decode_combine_rows_kernel<32>
                    <<<dim3(32, 1), head_dim, 0, stream>>>(
                        eng->d_attn_out, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                        head_dim, GEMMA4_SLIDING_WINDOW, 0, d_pos + 1, n_heads*head_dim);
            } else {
            sliding_attn_splitk_rows_kernel<GEMMA4_HEADS, GEMMA4_KV_HEADS, GEMMA4_HEAD_DIM>
                <<<dim3(max_splits, 1), GEMMA4_KV_HEADS*32, 0, stream>>>(
                    eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_attn_q,
                    eng->d_sliding_k + (size_t)layer * lstride,
                    eng->d_sliding_v + (size_t)layer * lstride,
                    GEMMA4_SLIDING_WINDOW, 0, eng->sliding_kv_capacity, d_pos + 1);
            flash_decode_combine_rows_kernel<GEMMA4_HEADS>
                <<<dim3(GEMMA4_HEADS, 1), head_dim, 0, stream>>>(
                    eng->d_attn_out, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                    head_dim, GEMMA4_SLIDING_WINDOW, 0, d_pos + 1, n_heads*head_dim);
            }
        } else if (eng->paged_read) {
            // Paged read: attend the pool through this sequence's block table,
            // via the SAME split-K kernels the batch path uses (B=1 view) — so the
            // paged read is now bit-identical to the contiguous split-K decode.
            size_t pls = (size_t)eng->slid_pool.n_blocks * PAGED_KV_BLOCK_TOKENS * kv_size;
            PagedSeqView hv; hv.block_table = eng->cur.d_slid_blocks; hv.n_blocks = eng->cur.slid_bt.n;
            hv.base = eng->cur.slid_bt.base; hv.n_tokens = pos + 1;
            cudaMemcpyAsync(eng->d_ms_views_slid, &hv, sizeof(PagedSeqView), cudaMemcpyHostToDevice, stream);
            if (n_heads == 32 && n_kv_heads == 16) {
                paged_sliding_attn_splitk_batched<32, 16, GEMMA4_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), 16*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_attn_q,
                        eng->d_slid_pool_k + (size_t)layer * pls, eng->d_slid_pool_v + (size_t)layer * pls,
                        eng->d_ms_views_slid, GEMMA4_SLIDING_WINDOW, GEMMA4_SLIDING_SPLIT_CHUNK,
                        PAGED_KV_BLOCK_TOKENS, kv_size);
                paged_flash_decode_combine_batched<32, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(32, 1), head_dim, 0, stream>>>(
                        eng->d_attn_out, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                        eng->d_ms_views_slid, head_dim, GEMMA4_SLIDING_WINDOW, GEMMA4_SLIDING_SPLIT_CHUNK);
            } else {
            paged_sliding_attn_splitk_batched<GEMMA4_HEADS, GEMMA4_KV_HEADS,
                                              GEMMA4_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), GEMMA4_KV_HEADS*32, 0, stream>>>(
                    eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_attn_q,
                    eng->d_slid_pool_k + (size_t)layer * pls, eng->d_slid_pool_v + (size_t)layer * pls,
                    eng->d_ms_views_slid, GEMMA4_SLIDING_WINDOW, GEMMA4_SLIDING_SPLIT_CHUNK,
                    PAGED_KV_BLOCK_TOKENS, kv_size);
            paged_flash_decode_combine_batched<GEMMA4_HEADS, GEMMA4_GLOBAL_MAX_SPLITS>
                <<<dim3(GEMMA4_HEADS, 1), head_dim, 0, stream>>>(
                    eng->d_attn_out, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                    eng->d_ms_views_slid, head_dim, GEMMA4_SLIDING_WINDOW, GEMMA4_SLIDING_SPLIT_CHUNK);
            }
        } else {
        // Split-K warp-per-KV-head (no per-key __syncthreads — see sliding_attn_splitk_kernel).
        sliding_attn_decode_broadcast(
                eng, eng->d_attn_out, eng->d_attn_q,
                eng->d_sliding_k + (size_t)layer * lstride,
                eng->d_sliding_v + (size_t)layer * lstride,
                n_heads, n_kv_heads, head_dim,
                GEMMA4_SLIDING_WINDOW,
                pos + 1,                         // FLAT: n_tokens in cache (incl. this one)
                stream);
        }
    } else {
        // n_ctx = tokens already in cache + this one (written above).
        // Flash kernel uses a constant 32-float scratch, so there is no longer any
        // context-length shared-memory cap (the old (32+n_ctx)-float buffer capped
        // ctx at ~25K and failed silently past it).
        int n_ctx = pos + 1;   // pos == eng->cur.n_tokens at every call site
        int slot = eng->global_slot[layer];
        size_t layer_stride = (size_t)eng->global_kv_capacity
                            * (size_t)eng->cfg.n_kv_global * GEMMA4_GLOBAL_HEAD_DIM;
        if (d_pos) {
            // Graph path: fixed-grid rows kernels (r=0) with device n_tokens = d_pos[1].
            if (eng->cfg.n_heads == 32 && eng->cfg.n_kv_global == 4) {
                global_attn_splitk_rows_kernel<32, 4, GEMMA4_GLOBAL_HEAD_DIM>
                    <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), 32*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_attn_q,
                        eng->d_global_k + (size_t)slot * layer_stride,
                        eng->d_global_v + (size_t)slot * layer_stride,
                        0, d_pos + 1);
                flash_decode_combine_rows_kernel<32>
                    <<<dim3(32, 1), head_dim, 0, stream>>>(
                        eng->d_attn_out, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                        head_dim, /*window=*/0, 0, d_pos + 1, n_heads*head_dim);
            } else {
            global_attn_splitk_rows_kernel<GEMMA4_HEADS, 1, GEMMA4_GLOBAL_HEAD_DIM>
                <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), GEMMA4_HEADS*32, 0, stream>>>(
                    eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_attn_q,
                    eng->d_global_k + (size_t)slot * layer_stride,
                    eng->d_global_v + (size_t)slot * layer_stride,
                    0, d_pos + 1);
            flash_decode_combine_rows_kernel<GEMMA4_HEADS>
                <<<dim3(GEMMA4_HEADS, 1), head_dim, 0, stream>>>(
                    eng->d_attn_out, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                    head_dim, /*window=*/0, 0, d_pos + 1, n_heads*head_dim);
            }
        } else if (eng->paged_read) {
            // Paged read: attend the pool through this sequence's block table,
            // via the SAME split-K kernels the batch path uses (B=1 view).
            size_t pls = (size_t)eng->glob_pool.n_blocks * PAGED_KV_BLOCK_TOKENS * kv_size;
            PagedSeqView hv; hv.block_table = eng->cur.d_glob_blocks; hv.n_blocks = eng->cur.glob_bt.n;
            hv.base = eng->cur.glob_bt.base; hv.n_tokens = n_ctx;
            cudaMemcpyAsync(eng->d_ms_views_glob, &hv, sizeof(PagedSeqView), cudaMemcpyHostToDevice, stream);
            if (n_heads == 32 && n_kv_heads == 4) {
                paged_global_attn_splitk_batched<32, 4, GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), 32*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_attn_q,
                        eng->d_glob_pool_k + (size_t)slot * pls, eng->d_glob_pool_v + (size_t)slot * pls,
                        eng->d_ms_views_glob, GEMMA4_GLOBAL_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, kv_size);
                paged_flash_decode_combine_batched<32, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(32, 1), head_dim, 0, stream>>>(
                        eng->d_attn_out, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                        eng->d_ms_views_glob, head_dim, /*window=*/0, GEMMA4_GLOBAL_SPLIT_CHUNK);
            } else {
            paged_global_attn_splitk_batched<GEMMA4_HEADS, GEMMA4_GLOBAL_KV_HEADS,
                                             GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), GEMMA4_HEADS*32, 0, stream>>>(
                    eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_attn_q,
                    eng->d_glob_pool_k + (size_t)slot * pls, eng->d_glob_pool_v + (size_t)slot * pls,
                    eng->d_ms_views_glob, GEMMA4_GLOBAL_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, kv_size);
            paged_flash_decode_combine_batched<GEMMA4_HEADS, GEMMA4_GLOBAL_MAX_SPLITS>
                <<<dim3(GEMMA4_HEADS, 1), head_dim, 0, stream>>>(
                    eng->d_attn_out, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                    eng->d_ms_views_glob, head_dim, /*window=*/0, GEMMA4_GLOBAL_SPLIT_CHUNK);
            }
        } else {
        // GQA-broadcast split-K (Step 1): reads each global K/V tile ONCE (was 16×).
        global_attn_decode_broadcast(
                eng, eng->d_attn_out, eng->d_attn_q,
                eng->d_global_k + (size_t)slot * layer_stride,
                eng->d_global_v + (size_t)slot * layer_stride,
                n_heads, head_dim, n_ctx, stream);
        {
            cudaError_t le = cudaGetLastError();
            if (le != cudaSuccess)
                fprintf(stderr, "fucina: global_attn_decode launch failed: %s\n",
                        cudaGetErrorString(le));
        }
        }
        // Do NOT increment global_n_tokens here; the engine-level functions
        // (prefill/decode) own the counter and advance it once per token.
    }

    // ── 8. Output projection → post-attn norm → residual add ─────────────
    if (nvfp4) {
        nvfp4_decode_proj(eng, layer, PJ_O, eng->d_attn_out, eng->d_x, out_dim_q, HS, stream);
    } else
    gemv_w(eng, eng->d_x,
        weight_fp8(eng, eng->tensors.layers[layer].attn_output),
        eng->d_attn_out, out_dim_q, HS, stream, (int)eng->tensors.layers[layer].fmt_o);
    // Post-attention sandwich norm
    rms_norm_kernel<<<1, block, smem32, stream>>>(
        eng->d_norm, eng->d_x, NORM_W(d_w_post_attn_norm),
        HS, GEMMA4_RMS_EPS);
    // Residual: normed_attn_proj + pre-layer input
    residual_add_kernel<<<(HS+255)/256, 256, 0, stream>>>(
        eng->d_norm, eng->d_residual, HS);
    // d_norm = attn_out = new residual for FFN
    cudaMemcpyAsync(eng->d_x,        eng->d_norm,
        HS * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(eng->d_residual,  eng->d_norm,
        HS * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // ── 9. Pre-FFN RMSNorm ────────────────────────────────────────────────
    rms_norm_kernel<<<1, block, smem32, stream>>>(
        eng->d_norm, eng->d_x, NORM_W(d_w_ffn_norm),
        HS, GEMMA4_RMS_EPS);

    // ── 10+11. Fused Gate + Up + GeGLU (audit #30 lever 2) ────────────────
    // Quantize the shared FFN-norm activation ONCE (+ the per-block Σ for the Q4_0 −8 fold),
    // then ONE fused kernel computes gate·up·GeGLU per row — the gate/up intermediates never
    // touch DRAM, and interleaving both weight reads in-warp lifts FFN bandwidth. Bit-identical
    // to the old gate-mmvq + up-mmvq + geglu (see mmvq_q*_glu_kernel).
    if (nvfp4) {
        // No fused FP4 GLU: two GEMVs into engine-resident gate/up scratch (graph-safe), then
        // the existing float geglu (gelu_tanh(gate)·up — matches the fused path) into d_ffn_out.
        nvfp4_decode_proj(eng, layer, PJ_GATE, eng->d_norm, eng->d_ffn_gate, HS, IM, stream);
        nvfp4_decode_proj(eng, layer, PJ_UP,   eng->d_norm, eng->d_ffn_up,   HS, IM, stream);
        geglu_kernel<<<(IM+255)/256, 256, 0, stream>>>(
            eng->d_ffn_out, eng->d_ffn_gate, eng->d_ffn_up, IM);
    } else {
    int fmt_gate = (int)eng->tensors.layers[layer].fmt_gate;
    int fmt_up   = (int)eng->tensors.layers[layer].fmt_up;
    if (fmt_gate == FORMAT_Q4_0 || fmt_gate == FORMAT_Q8_0) {
        // Fused gate·up·GeGLU (Q4_0/Q8_0 only) — gate/up intermediates never touch DRAM.
        quantize_q8_1_kernel<<<HS/32, 32, 0, stream>>>(
            eng->d_norm, eng->d_qx, eng->d_dx, eng->d_sx, HS);
        mmvq_glu_launch(eng->d_ffn_out,
            weight_fp8(eng, eng->tensors.layers[layer].ffn_gate),
            weight_fp8(eng, eng->tensors.layers[layer].ffn_up),
            eng->d_qx, eng->d_dx, eng->d_sx, HS, IM,
            fmt_gate, stream);
    } else {
        // Q4_K/Q6_K gate/up have no fused GLU: two GEMVs into gate/up scratch, then geglu.
        gemv_w(eng, eng->d_ffn_gate,
            weight_fp8(eng, eng->tensors.layers[layer].ffn_gate), eng->d_norm, HS, IM, stream, fmt_gate);
        gemv_w(eng, eng->d_ffn_up,
            weight_fp8(eng, eng->tensors.layers[layer].ffn_up),   eng->d_norm, HS, IM, stream, fmt_up);
        geglu_kernel<<<(IM+255)/256, 256, 0, stream>>>(
            eng->d_ffn_out, eng->d_ffn_gate, eng->d_ffn_up, IM);
    }
    }

    // ── 12. FFN down projection ───────────────────────────────────────────
    if (nvfp4) {
        nvfp4_decode_proj(eng, layer, PJ_DOWN, eng->d_ffn_out, eng->d_x, IM, HS, stream);
    } else
    gemv_w(eng, eng->d_x,
        weight_fp8(eng, eng->tensors.layers[layer].ffn_down),
        eng->d_ffn_out, IM, HS, stream, (int)eng->tensors.layers[layer].fmt_down);

    // ── 13. Post-FFN sandwich norm → residual add ─────────────────────────
    rms_norm_kernel<<<1, block, smem32, stream>>>(
        eng->d_norm, eng->d_x, NORM_W(d_w_post_ffn_norm),
        HS, GEMMA4_RMS_EPS);
    residual_add_kernel<<<(HS+255)/256, 256, 0, stream>>>(
        eng->d_norm, eng->d_residual, HS);
    cudaMemcpyAsync(eng->d_x, eng->d_norm,
        HS * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // ── 14. layer_output_scale (scalar preloaded at create) ──────────────
    if (eng->h_out_scale[layer] != 1.0f) {
        scale_kernel<<<(HS+255)/256, 256, 0, stream>>>(
            eng->d_x, HS, eng->h_out_scale[layer]);
    }

    return 0;
}

// =========================================================================
// ─── BF16 weight residency + batched GEMM (Step 2 foundation) ───────────
// =========================================================================

// Projection ids, indexing the rotating per-layer scratch eng->d_bf16_layer[p].

// Resolve a projection's source byte offset + GEMM dims for a layer. Returns 0
// for projections that do not exist (global layers have no separate V weight).
static int proj_desc(const gemma4_engine_t *eng, int layer, int p,
                     uint64_t *offset, int *in_dim, int *out_dim)
{
    layer_type_t lt = eng->layer_types[layer];
    int H   = eng->cfg.hidden_size;
    int I   = eng->cfg.intermediate;
    int hd  = (lt == LAYER_SLIDING) ? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM;
    int nkv = (lt == LAYER_SLIDING) ? eng->cfg.n_kv_sliding : eng->cfg.n_kv_global;
    int oq  = eng->cfg.n_heads * hd;
    int okv = nkv * hd;
    const __typeof__(eng->tensors.layers[0]) *L = &eng->tensors.layers[layer];
    switch (p) {
        case PJ_Q:    *offset = L->attn_q;      *in_dim = H; *out_dim = oq;  return 1;
        case PJ_K:    *offset = L->attn_k;      *in_dim = H; *out_dim = okv; return 1;
        case PJ_V:    if (lt != LAYER_SLIDING) return 0;  // global: V = K, no weight
                      *offset = L->attn_v;      *in_dim = H; *out_dim = okv; return 1;
        case PJ_O:    *offset = L->attn_output; *in_dim = oq;  *out_dim = H; return 1;
        case PJ_GATE: *offset = L->ffn_gate;    *in_dim = H; *out_dim = I; return 1;
        case PJ_UP:   *offset = L->ffn_up;      *in_dim = H; *out_dim = I; return 1;
        case PJ_DOWN: *offset = L->ffn_down;    *in_dim = I; *out_dim = H; return 1;
    }
    return 0;
}

// Allocate the ROTATING per-layer BF16 dequant scratch: one buffer per projection
// slot (0..6), each sized to the widest [out_dim×in_dim] that slot takes over all
// layers (sliding vs global differ). ~0.5 GB total vs the old persistent 21.8 GB.
// Idempotent. Returns 0 on success, -1 on allocation failure.
static int build_bf16_weights(gemma4_engine_t *eng)
{
    if (eng->bf16_ready) return 0;

    uint64_t maxn[PJ_COUNT] = {0};
    for (int l = 0; l < eng->cfg.n_layers; l++) {
        for (int p = 0; p < PJ_COUNT; p++) {
            uint64_t off; int in_dim, out_dim;
            if (!proj_desc(eng, l, p, &off, &in_dim, &out_dim)) continue;
            uint64_t n = (uint64_t)in_dim * out_dim;
            if (n > maxn[p]) maxn[p] = n;
        }
    }
    // Two ping/pong sets so the next layer's dequant can run concurrently with this
    // layer's GEMMs (see d_bf16_layer comment). ~2× the old 0.5 GB, still ≪ 21.8 GB.
    size_t total = 0;
    for (int b = 0; b < 2; b++)
        for (int p = 0; p < PJ_COUNT; p++) eng->d_bf16_layer[b][p] = NULL;
    for (int b = 0; b < 2; b++) {
        for (int p = 0; p < PJ_COUNT; p++) {
            if (maxn[p] == 0) continue;
            if (cudaMalloc(&eng->d_bf16_layer[b][p],
                           maxn[p] * sizeof(__nv_bfloat16)) != cudaSuccess) {
                fprintf(stderr, "fucina: BF16 dequant scratch alloc failed at buf %d proj %d "
                        "(%.2f GB in so far)\n", b, p, total / 1e9);
                cudaGetLastError();
                for (int bb = 0; bb < 2; bb++)
                    for (int q = 0; q < PJ_COUNT; q++)
                        if (eng->d_bf16_layer[bb][q]) { cudaFree(eng->d_bf16_layer[bb][q]); eng->d_bf16_layer[bb][q] = NULL; }
                return -1;
            }
            total += maxn[p] * sizeof(__nv_bfloat16);
        }
    }
    fprintf(stderr, "fucina: BF16 per-layer dequant scratch (%.2f GB rotating x2 "
            "pipelined, vs 21.8 GB persistent)\n", total / 1e9);
    eng->bf16_ready = 1;
    return 0;
}

// FUCINA_PACKED: build the repacked-Q4_0 decode blob (idempotent). Allocates a copy the same
// size as d_weights and repacks every Q4_0 projection in place (same per-projection offset)
// into [16-B-aligned quants ‖ fp16 scales]. Q4_0 models only. Returns 0 on success; on any
// failure leaves packed_ready=0 (callers fall back to the native dp4a path) — never fatal.
static int build_packed_q4(gemma4_engine_t *eng)
{
    if (eng->packed_ready) return 0;
    if (FMT(eng) != FORMAT_Q4_0 || !eng->d_weights) return -1;
    size_t tbytes = eng->gguf_size - eng->tdata_start;
    if (cudaMalloc(&eng->d_weights_packed, tbytes) != cudaSuccess) {
        fprintf(stderr, "fucina: packed-Q4_0 alloc failed (%.2f GB) — packed decode path off\n",
                tbytes / 1e9);
        cudaGetLastError();
        eng->d_weights_packed = NULL;
        return -1;
    }
    for (int l = 0; l < eng->cfg.n_layers; l++) {
        for (int p = 0; p < PJ_COUNT; p++) {
            uint64_t off; int in_dim, out_dim;
            if (!proj_desc(eng, l, p, &off, &in_dim, &out_dim)) continue;
            const uint8_t *src  = weight_fp8(eng, off);                 // native Q4_0 bytes
            uint8_t       *base = eng->d_weights_packed + (off - eng->tdata_start);
            size_t nb = (size_t)(in_dim >> 5);
            size_t nblocks = (size_t)out_dim * nb;
            uint16_t *scales = (uint16_t *)(base + nblocks * 16);
            repack_q4_0_kernel<<<(unsigned)((nblocks + 255) / 256), 256>>>(
                src, base, scales, nblocks);
        }
    }
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "fucina: packed-Q4_0 repack error: %s — packed decode path off\n",
                cudaGetErrorString(e));
        cudaFree(eng->d_weights_packed);
        eng->d_weights_packed = NULL;
        return -1;
    }
    fprintf(stderr, "fucina: packed-Q4_0 decode weights built (%.2f GB, 16-B-aligned quants "
            "‖ fp16 scales — coalesced uint4 GEMV loads)\n", tbytes / 1e9);
    eng->packed_ready = 1;
    return 0;
}

// PACKED Q4_K (Qwen3 / K-quant): repack every Q4_K bulk projection IN PLACE inside d_weights
// (same 144-B superblock → NO second weight copy). Repack is per-superblock but sub-blocks j and
// j^1 share their 32 source quant bytes, so an in-place superblock kernel would corrupt itself;
// we repack each tensor through ONE small reused temp (sized to the largest Q4_K tensor) and copy
// back. Idempotent; non-fatal — on any failure q4k_packed stays 0 and callers use the native
// Q4_K dp4a path. token_embd / output_weight are repacked too when Q4_K so use_packed_q4k's
// pointer-range gate is sound (the converted-to-Q8_0 head reads d_token_embd, not these bytes).
static int build_packed_q4k(gemma4_engine_t *eng)
{
    if (eng->q4k_packed) return 0;
    if (!eng->d_weights) return -1;
    // Qwen3.5 hybrid (M3): its single-seq forward reads Q4_K weights via the NATURAL-layout
    // native dp4a (gemv_w) and Q5_K via a natural-layout fp32 dequant; it must NOT see the
    // de-interleaved (packed) superblock layout. Skip the in-place repack so d_weights stays
    // natural Q4_K for qwen35. Gated on the new arch → every other arch keeps the packed
    // decode GEMV byte-identical.
    if (eng->cfg.arch == GEMMA4_ARCH_QWEN3_5) return 0;
    const char *no_packed = getenv("FUCINA_NO_PACKED");
    if (no_packed && no_packed[0] == '1') return -1;

    // Enumerate EVERY weight tensor by its GGUF name and repack the Q4_K ones using the
    // tensor's OWN element count from the GGUF header (NOT proj_desc dims — proj_desc uses
    // GEMMA4_GLOBAL_HEAD_DIM for the attn projections, which is wrong for Qwen3's head_dim,
    // so it would size the repack span incorrectly). off is absolute; n_el is the element
    // count (Q4_K bytes = n_el/256 * 144). This also naturally covers token_embd / output.
    auto tensor_q4k = [&](const char *nm, uint64_t *off, uint64_t *nsuper) -> bool {
        uint64_t o, ne; uint32_t gt;
        if (gguf_find_tensor(eng->gguf_data, eng->gguf_size, nm, &o, &ne, &gt) != 0) return false;
        if (gt != GGML_TYPE_Q4_K || (ne % 256) != 0) return false;
        if (o < eng->tdata_start) return false;
        if (o - eng->tdata_start >= (uint64_t)(eng->gguf_size - eng->tdata_start)) return false;
        *off = o; *nsuper = ne / 256; return true;
    };
    // Pass 1: count + max byte size for the reusable temp.
    size_t max_bytes = 0; int n_q4k = 0;
    auto names_loop = [&](auto fn){
        char nm[128];
        const char *suf[] = {"attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"};
        for (int l = 0; l < eng->cfg.n_layers; l++)
            for (int s = 0; s < 7; s++) {
                snprintf(nm, sizeof(nm), "blk.%d.%s.weight", l, suf[s]);
                uint64_t off, ns; if (tensor_q4k(nm, &off, &ns)) fn(off, ns);
            }
        uint64_t off, ns;
        if (tensor_q4k("token_embd.weight", &off, &ns)) fn(off, ns);
        if (tensor_q4k("output.weight",     &off, &ns)) fn(off, ns);
    };
    names_loop([&](uint64_t, uint64_t ns){ size_t b = (size_t)ns * 144;
                                           if (b > max_bytes) max_bytes = b; n_q4k++; });

    if (n_q4k == 0) return -1;   // no Q4_K bulk weights (e.g. Gemma Q4_0) — nothing to do

    uint8_t *tmp = NULL;
    if (cudaMalloc(&tmp, max_bytes) != cudaSuccess) {
        fprintf(stderr, "fucina: packed-Q4_K temp alloc failed (%.2f MB) — packed Q4_K off\n",
                max_bytes / 1e6);
        cudaGetLastError();
        return -1;
    }
    // Pass 2: repack each Q4_K tensor through the reusable temp (src≠dst avoids the
    // intra-superblock aliasing of an in-place kernel) then copy the packed bytes back.
    names_loop([&](uint64_t off, uint64_t n_super){
        size_t bytes = (size_t)n_super * 144;
        uint8_t *dst = eng->d_weights + (off - eng->tdata_start);   // in-place region
        repack_q4_k_kernel<<<(unsigned)((n_super + 255) / 256), 256>>>(dst, tmp, n_super);
        cudaMemcpy(dst, tmp, bytes, cudaMemcpyDeviceToDevice);      // packed bytes back
    });
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    cudaFree(tmp);
    if (e != cudaSuccess) {
        fprintf(stderr, "fucina: packed-Q4_K repack error: %s — packed Q4_K path off\n",
                cudaGetErrorString(e));
        return -1;
    }
    fprintf(stderr, "fucina: packed-Q4_K decode weights built in place (%d tensors, "
            "de-interleaved superblocks — coalesced uint4 GEMV loads)\n", n_q4k);
    eng->q4k_packed = 1;
    return 0;
}

// Dequantize layer `l`'s 7 projection weights (Q8_0/FP8 → BF16) into ping/pong
// buffer `buf` on `stream`. The double-buffer + dq_stream pipeline in the prefill
// loops (see d_bf16_layer comment) issues layer L+1's dequant here while layer L's
// GEMMs read the other buffer, so this bandwidth-bound pass overlaps compute.
static void dequant_layer_bf16_buf(gemma4_engine_t *eng, int l, int buf, cudaStream_t stream)
{
    for (int p = 0; p < PJ_COUNT; p++) {
        uint64_t off; int in_dim, out_dim;
        if (!proj_desc(eng, l, p, &off, &in_dim, &out_dim)) continue;
        uint64_t n = (uint64_t)in_dim * out_dim;
        const uint8_t *src = weight_fp8(eng, off);
        dequant_to_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
                eng->d_bf16_layer[buf][p], src, n, FMT(eng));
    }
}

// Y[n_tokens × out_dim] = W[out_dim × in_dim] @ X[n_tokens × in_dim], all
// token-major (row-major). W is BF16 col-major [in_dim × out_dim] (= source
// row-major [out_dim][in_dim]) → op_T; X is BF16 token-major → col-major
// [in_dim × n_tokens] op_N; C accumulates in FP32. See derivation in the plan.
static cublasStatus_t gemm_bf16(
    gemma4_engine_t *eng, const __nv_bfloat16 *W, const __nv_bfloat16 *X,
    float *Y, int in_dim, int out_dim, int n_tokens)
{
    const float alpha = 1.0f, beta = 0.0f;
    return cublasGemmEx(
        eng->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        out_dim, n_tokens, in_dim,
        &alpha,
        W, CUDA_R_16BF, in_dim,
        X, CUDA_R_16BF, in_dim,
        &beta,
        Y, CUDA_R_32F, out_dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Grouped (ragged) BF16 tensor-core GEMM over ACTIVE MoE experts in ONE cuBLAS launch.
// Each active expert e (hc[e]>0) contributes one problem: Y_e[out×n_e] = W_e[out×in] @ X_e[in×n_e]
// with W_e = Wbase + e·wstride, X_e = Xbase + ho[e]·in_dim, Y_e = Ybase + ho[e]·out_dim — the SAME
// per-expert cublasGemmEx(OP_T,OP_N) math as gemm_bf16, so results are bit-identical. Replaces the
// per-expert loop (E separate ~16-row GEMMs landing on 16×16 tiles + E× launch overhead) with a
// single grouped-batched kernel, the dominant 2k-prefill expert cost. Host pointer/dim arrays are
// stack-sized to GEMMA4 max experts (E ≤ 256). No-op when no expert is active.
static void gemm_bf16_grouped(gemma4_engine_t *eng,
    const __nv_bfloat16 *Wbase, size_t wstride, const __nv_bfloat16 *Xbase, float *Ybase,
    const int *hc, const int *ho, int E, int in_dim, int out_dim, cudaStream_t stream)
{
    // Per-active-expert cublasGemmEx loop — NOT cublasGemmGroupedBatchedEx. The grouped-batched
    // call is broken on this stack (CUDA 13 / GB10 sm_121): with bit-identical inputs it returned
    // all-zeros on its first invocation and NONDETERMINISTIC output on later ones (measured via
    // per-stage checksums: router/topk/gathered-X identical across runs, gate/up GEMM output
    // different every call) — the diverse-prompt serving corruption. The plain GemmEx loop is
    // bit-stable; its extra launch overhead only touches prefill-width calls.
    cublasSetStream(eng->cublas, stream);
    const float alpha = 1.0f, beta = 0.0f;
    for (int e = 0; e < E; e++) if (hc[e] > 0) {
        cublasGemmEx(eng->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            out_dim, hc[e], in_dim,
            &alpha,
            Wbase + (size_t)e * wstride, CUDA_R_16BF, in_dim,
            Xbase + (size_t)ho[e] * in_dim, CUDA_R_16BF, in_dim,
            &beta,
            Ybase + (size_t)ho[e] * out_dim, CUDA_R_32F, out_dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// NVFP4 (block-scaled FP4) prefill projections — FUCINA_FP4.
//
// cuBLASLt computes D[out×N] = W[out×in] @ X[in×N] with W,X in CUDA_R_4F_E2M1,
// per-16-element E4M3 block scales (VEC16_UE4M3) and a per-tensor fp32 global
// scale folded into alpha. ~2.4× the BF16 tensor-core GEMM on GB10. Block scale
// tensor uses cuBLASLt's 32×4×4 swizzled layout (validated bit-exact in
// cuda/test_fp4_gemm.cu). Accuracy is the gate (E2M1 = 1 mantissa bit) — flag-gated.
// ─────────────────────────────────────────────────────────────────────────
#define NVFP4_BLK 16

// amax over [n] elements (bf16) → atomic max into a device float (init 0).
__global__ void nvfp4_amax_bf16_kernel(const __nv_bfloat16 *x, uint64_t n, float *amax) {
    uint64_t i = blockIdx.x*(uint64_t)blockDim.x + threadIdx.x;
    float v = (i < n) ? fabsf(__bfloat162float(x[i])) : 0.f;
    // warp reduce then block atomic
    for (int o=16;o>0;o>>=1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    __shared__ float s[32];
    int lane = threadIdx.x&31, wid = threadIdx.x>>5;
    if (lane==0) s[wid]=v;
    __syncthreads();
    if (wid==0){ v = (lane < (blockDim.x+31)/32) ? s[lane] : 0.f;
        for(int o=16;o>0;o>>=1) v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,o));
        if(lane==0) atomicMax((int*)amax, __float_as_int(v)); } // values ≥0 → int order ok
}
// gs = amax / (6*448); guard 0.
__global__ void nvfp4_gs_kernel(const float *amax, float *gs) {
    float a = *amax; *gs = (a>0.f) ? a/(6.0f*448.0f) : 1e-30f;
}
// alpha[0] = gs_w · gs_act ; alpha[1] = 0 (beta), for device pointer-mode.
__global__ void nvfp4_alpha_kernel(const float *gsw, const float *gsact, float *alpha) {
    alpha[0] = (*gsw) * (*gsact); alpha[1] = 0.f;
}

// Quantize [rows][k] bf16 (k = contraction, contiguous) → packed E2M1 [rows][k/2] +
// LINEAR E4M3 block scales [rows][k/16]. gs read from device scalar.
__global__ void nvfp4_quant_bf16_kernel(const __nv_bfloat16 *__restrict__ X,
        uint8_t *__restrict__ fp4, uint8_t *__restrict__ bscale,
        int rows, int k, const float *gsp) {
    int row = blockIdx.y;
    int blk = blockIdx.x*blockDim.x + threadIdx.x;
    int nblk = k/NVFP4_BLK;
    if (row>=rows || blk>=nblk) return;
    float gs = *gsp;
    const __nv_bfloat16 *xr = X + (size_t)row*k + blk*NVFP4_BLK;
    float v[NVFP4_BLK], amax=0.f;
    #pragma unroll
    for(int i=0;i<NVFP4_BLK;i++){ v[i]=__bfloat162float(xr[i]); amax=fmaxf(amax,fabsf(v[i])); }
    float bs_stored = (amax>0.f) ? amax/6.0f/gs : 0.f;
    __nv_fp8_storage_t e = __nv_cvt_float_to_fp8(bs_stored, __NV_SATFINITE, __NV_E4M3);
    bscale[(size_t)row*nblk + blk] = (uint8_t)e;
    float bsf = __half2float(__half(__nv_cvt_fp8_to_halfraw(e, __NV_E4M3)));
    float divisor = gs*bsf; if (divisor<=0.f) divisor = 1e30f;
    uint8_t *o = fp4 + (size_t)row*(k/2) + blk*(NVFP4_BLK/2);
    #pragma unroll
    for(int i=0;i<NVFP4_BLK;i+=2){
        float2 p = make_float2(v[i]/divisor, v[i+1]/divisor);
        o[i/2] = (uint8_t)__nv_cvt_float2_to_fp4x2(p, __NV_E2M1, cudaRoundNearest);
    }
}
// Linear [outer][nblk] E4M3 scales → cuBLASLt 32×4×4 swizzled layout (nblk_pad mult of 4).
__global__ void nvfp4_swizzle_kernel(const uint8_t *__restrict__ lin, uint8_t *__restrict__ sw,
        int outer, int nblk, int nblk_pad) {
    int o = blockIdx.y*blockDim.y + threadIdx.y;
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    if (o>=outer || s>=nblk) return;
    int oo=o/128, oi=o%128, so=s/4, si=s%4;
    size_t off = ((size_t)oo*(nblk_pad/4)+so)*512 + (oi%32)*16 + (oi/32)*4 + si;
    sw[off] = lin[(size_t)o*nblk + s];
}
static inline int nvfp4_pad(int x,int m){ return ((x+m-1)/m)*m; }

// Quantize one operand (bf16 [rows][k]) into caller-provided packed+swizzled-scale
// buffers, computing its per-tensor global scale into gsp (device). amax_scratch is a
// device float (reset here). For weights: called once at build. For activations: per GEMM.
static void nvfp4_quantize(const __nv_bfloat16 *X, int rows, int k,
        uint8_t *fp4, uint8_t *lin_sc, uint8_t *sw_sc, float *gsp,
        float *amax_scratch, cudaStream_t st) {
    cudaMemsetAsync(amax_scratch, 0, sizeof(float), st);
    uint64_t n = (uint64_t)rows*k;
    nvfp4_amax_bf16_kernel<<<(unsigned)((n+255)/256),256,0,st>>>(X, n, amax_scratch);
    nvfp4_gs_kernel<<<1,1,0,st>>>(amax_scratch, gsp);
    int nblk = k/NVFP4_BLK;
    dim3 b(256), g((nblk+255)/256, rows);
    nvfp4_quant_bf16_kernel<<<g,b,0,st>>>(X, fp4, lin_sc, rows, k, gsp);
    int kp = nvfp4_pad(nblk,4);
    dim3 b2(32,8), g2((nblk+31)/32,(rows+7)/8);
    nvfp4_swizzle_kernel<<<g2,b2,0,st>>>(lin_sc, sw_sc, rows, nblk, kp);
}

// ── CUTLASS grouped block-scaled NVFP4 experts (qwen3_5_moe) ────────────────────────────────
// The sm120 ptr-array grouped GEMM in dg_fp4_moe.cu (libdg.a): D bf16 = (A·Bᵀ)·alpha per expert,
// A/B packed E2M1, SF swizzled ue4m3 with per-expert padded strides. Measured on GB10 vs the
// dp4a Q4_K grouped GEMV at the 35B expert shapes: 242 GB/s weight-read at 1 tok/expert
// (dp4a: 114) and 11.7 TFLOP/s at 16 tok/expert — the aggregate-decode AND prefill expert path.
extern "C" int dg_fp4_moe_grouped(
    void* D, const void* A, const void* A_sf, const void* B, const void* B_sf,
    const int* m_indptr, int num_groups, int N, int K,
    unsigned long long sfA_stride, unsigned long long sfB_stride,
    const float* alpha, cudaStream_t stream);
extern "C" void dg_fp4_sf_strides(int M_max, int N, int K,
    unsigned long long* sfA_stride, unsigned long long* sfB_stride);

// PER-ROW activation global scale: gsrow[t] = amax(X[t])/(6·448). Row-local by construction —
// a token's quantization must not depend on its batchmates (a batch-global amax made B=3 decode
// diverge from B=1 on the same rows; the batch self-test's row-independence gate caught it).
// The row scale is applied OUTSIDE the GEMM (alpha carries only the weight scale): scaling
// after the silu inputs / down output keeps the math per-row exact.
__global__ void q35fp4_row_gs_kernel(const float *__restrict__ X, int k, int total, float *gsrow) {
    int t = blockIdx.x;
    if (t >= total) return;
    const float *xr = X + (size_t)t * k;
    float m = 0.f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) m = fmaxf(m, fabsf(xr[i]));
    for (int o=16;o>0;o>>=1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    __shared__ float s[32];
    int lane = threadIdx.x&31, wid = threadIdx.x>>5;
    if (lane==0) s[wid]=m;
    __syncthreads();
    if (wid==0){ m = (lane < (blockDim.x+31)/32) ? s[lane] : 0.f;
        for(int o=16;o>0;o>>=1) m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,o));
        if(lane==0) gsrow[t] = (m>0.f) ? m/(6.0f*448.0f) : 1e-30f; }
}
// counting-sort outputs (coloff/count) → m_indptr[E+1] + assignment→expert map [total].
__global__ void q35fp4_grp_idx_kernel(const int *coloff, const int *count, int E, int total,
        int *indptr, int *t2e) {
    int ex = blockIdx.x*blockDim.x + threadIdx.x;
    if (ex < E) { int off = coloff[ex], c = count[ex]; indptr[ex] = off; for (int j=0;j<c;j++) t2e[off+j]=ex; }
    if (ex == 0) indptr[E] = total;
}
// Fused per-expert activation NVFP4 quant: X[total][k] f32 (expert-contiguous) → packed E2M1
// [total][k/2] + per-expert PADDED swizzled ue4m3 SF (offset math = nvfp4_swizzle_kernel's
// layout with the M-tile (row>>7) term, exactly what the CUTLASS grouped mainloop reads).
__global__ void q35fp4_quant_grp_kernel(const float *__restrict__ X, const int *__restrict__ t2e,
        const int *__restrict__ indptr, int k, int total, const float *__restrict__ gsrow,
        uint8_t *__restrict__ A_fp4, uint8_t *__restrict__ A_sf,
        unsigned long long sfAstride, int nKvec_pad) {
    int t = blockIdx.y, blk = blockIdx.x*blockDim.x + threadIdx.x, nblk = k/NVFP4_BLK;
    if (t >= total || blk >= nblk) return;
    float gs = gsrow[t];
    const float *xr = X + (size_t)t*k + blk*NVFP4_BLK;
    float v[NVFP4_BLK], amax = 0.f;
    #pragma unroll
    for (int i=0;i<NVFP4_BLK;i++){ v[i]=xr[i]; amax=fmaxf(amax,fabsf(v[i])); }
    float bs = (amax>0.f) ? amax/6.0f/gs : 0.f;
    __nv_fp8_storage_t e8 = __nv_cvt_float_to_fp8(bs, __NV_SATFINITE, __NV_E4M3);
    float div = gs*__half2float(__half(__nv_cvt_fp8_to_halfraw(e8, __NV_E4M3)));
    if (div <= 0.f) div = 1e30f;
    uint8_t *o = A_fp4 + (size_t)t*(k/2) + blk*(NVFP4_BLK/2);
    #pragma unroll
    for (int i=0;i<NVFP4_BLK;i+=2){
        float2 p = make_float2(v[i]/div, v[i+1]/div);
        o[i/2] = (uint8_t)__nv_cvt_float2_to_fp4x2(p, __NV_E2M1, cudaRoundNearest);
    }
    int ex = t2e[t], row = t - indptr[ex];
    int so = blk>>2, si = blk&3, om = row&31, im = (row&127)>>5, mt = row>>7, nKtiles = nKvec_pad>>2;
    size_t off = (size_t)ex*sfAstride + (size_t)mt*nKtiles*512 + (size_t)so*512 + (size_t)om*16 + im*4 + si;
    A_sf[off] = (uint8_t)e8;
}
// silu(gate)·up from the FUSED [total][2·effn] bf16 grouped-GEMM output (cols 0..effn = gate,
// effn..2effn = up) → f32, applying the per-row activation scale the GEMM's alpha did not carry.
// Same x/(1+__expf(-x)) as dg_silu_mul (oracle-parity form).
__global__ void q35fp4_gu_silu_mul_kernel(float *out, const __nv_bfloat16 *gu,
        const float *__restrict__ gsrow, int effn, int64_t n) {
    int64_t i = blockIdx.x*(int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t t = i/effn, j = i - t*effn;
    float gs = gsrow[t];
    float x = __bfloat162float(gu[t*2*effn + j]) * gs;
    float u = __bfloat162float(gu[t*2*effn + effn + j]) * gs;
    out[i] = (x / (1.0f + __expf(-x))) * u;
}
// down-proj bf16 output → f32 with the per-row activation scale (K=EFFN quant pass).
__global__ void q35fp4_dn_scale_kernel(float *out, const __nv_bfloat16 *D,
        const float *__restrict__ gsrow, int hdim, uint64_t n) {
    uint64_t i = blockIdx.x*(uint64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __bfloat162float(D[i]) * gsrow[i / hdim];
}

// Build persistent NVFP4 weights for every (layer,proj). Dequants each projection
// (Q4_0/Q8_0 → bf16 in a temp buffer) then NVFP4-quantizes it. Returns 0 / -1.
static int build_fp4_weights(gemma4_engine_t *eng) {
    if (eng->fp4_ready) return 0;
    if (cublasLtCreate(&eng->cublaslt) != CUBLAS_STATUS_SUCCESS) return -1;
    // find max projection element count for the temp bf16 + linear-scale scratch
    uint64_t maxn = 0; int max_in = 0, max_out = 0;
    for (int l=0;l<eng->cfg.n_layers;l++) for (int p=0;p<PJ_COUNT;p++){
        uint64_t off; int in_dim,out_dim; if(!proj_desc(eng,l,p,&off,&in_dim,&out_dim)) continue;
        uint64_t nn=(uint64_t)in_dim*out_dim; if(nn>maxn)maxn=nn;
        if(in_dim>max_in)max_in=in_dim; if(out_dim>max_out)max_out=out_dim;
    }
    __nv_bfloat16 *tmp_bf=nullptr; uint8_t *tmp_lin=nullptr;
    int ok=1;
    if (cudaMalloc(&tmp_bf, maxn*sizeof(__nv_bfloat16))!=cudaSuccess) ok=0;
    if (ok && cudaMalloc(&tmp_lin, (size_t)max_out*(max_in/NVFP4_BLK))!=cudaSuccess) ok=0;
    if (ok && cudaMalloc(&eng->d_fp4_gsw, (size_t)GEMMA4_CAP_LAYERS*PJ_COUNT*sizeof(float))!=cudaSuccess) ok=0;
    if (ok && cudaMalloc(&eng->d_fp4_amax, sizeof(float))!=cudaSuccess) ok=0;
    size_t wbytes=0;
    for (int l=0; ok && l<eng->cfg.n_layers; l++) for (int p=0; ok && p<PJ_COUNT; p++){
        uint64_t off; int in_dim,out_dim; if(!proj_desc(eng,l,p,&off,&in_dim,&out_dim)) continue;
        size_t packed=(size_t)out_dim*(in_dim/2);
        size_t swsz=(size_t)nvfp4_pad(out_dim,128)*nvfp4_pad(in_dim/NVFP4_BLK,4);
        if (cudaMalloc(&eng->d_fp4_w[l][p], packed)!=cudaSuccess){ok=0;break;}
        if (cudaMalloc(&eng->d_fp4_wsc[l][p], swsz)!=cudaSuccess){ok=0;break;}
        cudaMemset(eng->d_fp4_wsc[l][p],0,swsz);
        uint64_t nn=(uint64_t)in_dim*out_dim;
        dequant_to_bf16_kernel<<<(unsigned)((nn+255)/256),256>>>(tmp_bf, weight_fp8(eng,off), nn, FMT(eng));
        nvfp4_quantize(tmp_bf, out_dim, in_dim, eng->d_fp4_w[l][p], tmp_lin,
                       eng->d_fp4_wsc[l][p], eng->d_fp4_gsw + (l*PJ_COUNT+p), eng->d_fp4_amax, 0);
        wbytes += packed + swsz;
    }
    cudaDeviceSynchronize();
    if (tmp_bf) cudaFree(tmp_bf);
    if (tmp_lin) cudaFree(tmp_lin);
    if (!ok || cudaGetLastError()!=cudaSuccess) {
        fprintf(stderr,"fucina: NVFP4 weight build failed — BF16 prefill fallback\n");
        return -1;
    }
    // activation scratch global-scale + alpha + workspace + cached desc
    cudaMalloc(&eng->d_fp4_gsact, sizeof(float));
    cudaMalloc(&eng->d_fp4_alpha, 2*sizeof(float));   // [alpha, beta=0]
    eng->d_fp4_ws=nullptr; cudaMalloc(&eng->d_fp4_ws, 64ull<<20);
    cublasLtMatmulDescCreate(&eng->fp4_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opT=CUBLAS_OP_T, opN=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    int32_t smode=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &smode, sizeof(smode));
    cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &smode, sizeof(smode));
    int32_t pmode=CUBLASLT_POINTER_MODE_DEVICE;
    cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pmode, sizeof(pmode));
    eng->fp4_ready=1;
    fprintf(stderr,"fucina: NVFP4 prefill weights built (%.2f GB, ~2.4× BF16 tensor-core GEMM)\n", wbytes/1e9);
    return 0;
}

// Ensure activation NVFP4 scratch holds N tokens over the widest projection (INTERMEDIATE).
static int ensure_fp4_act(gemma4_engine_t *eng, int N) {
    if ((size_t)N <= eng->fp4_act_cap) return 0;
    if (eng->d_fp4_act)    cudaFree(eng->d_fp4_act);
    if (eng->d_fp4_actsc)  cudaFree(eng->d_fp4_actsc);
    if (eng->d_fp4_actlin) cudaFree(eng->d_fp4_actlin);
    eng->d_fp4_act=nullptr; eng->d_fp4_actsc=nullptr; eng->d_fp4_actlin=nullptr;
    size_t I = eng->cfg.intermediate;
    size_t packed=(size_t)N*(I/2);
    size_t swsz=(size_t)nvfp4_pad(N,128)*nvfp4_pad(I/NVFP4_BLK,4);
    size_t linsz=(size_t)N*(I/NVFP4_BLK);
    if (cudaMalloc(&eng->d_fp4_act, packed)!=cudaSuccess) { eng->fp4_act_cap=0; return -1; }
    if (cudaMalloc(&eng->d_fp4_actsc, swsz)!=cudaSuccess) { cudaFree(eng->d_fp4_act); eng->d_fp4_act=nullptr; eng->fp4_act_cap=0; return -1; }
    if (cudaMalloc(&eng->d_fp4_actlin, linsz)!=cudaSuccess) { cudaFree(eng->d_fp4_act); cudaFree(eng->d_fp4_actsc); eng->d_fp4_act=eng->d_fp4_actsc=nullptr; eng->fp4_act_cap=0; return -1; }
    eng->fp4_act_cap=N;
    return 0;
}

// NVFP4 projection: dst[out×N] = W[out×in] @ X[in×N]. X is the shared bf16 activation
// (d_inb) → quantized to NVFP4 here. Weight is persistent NVFP4. alpha = gs_w·gs_act
// (device pointer-mode). Falls through (returns false) if scratch alloc fails.
static bool gemm_nvfp4(gemma4_engine_t *eng, int l, int p, const __nv_bfloat16 *X,
        float *dst, int in_dim, int out_dim, int N, cudaStream_t st) {
    if (ensure_fp4_act(eng, N)!=0) return false;
    // quantize activation [N × in_dim] (rows=N, k=in_dim) → packed + swizzled scales
    int nblk=in_dim/NVFP4_BLK;
    cudaMemsetAsync(eng->d_fp4_actsc, 0,
        (size_t)nvfp4_pad(N,128)*nvfp4_pad(nblk,4), st);
    cudaMemsetAsync(eng->d_fp4_amax,0,sizeof(float),st);
    uint64_t nn=(uint64_t)N*in_dim;
    nvfp4_amax_bf16_kernel<<<(unsigned)((nn+255)/256),256,0,st>>>(X, nn, eng->d_fp4_amax);
    nvfp4_gs_kernel<<<1,1,0,st>>>(eng->d_fp4_amax, eng->d_fp4_gsact);
    dim3 b(256), g((nblk+255)/256, N);
    nvfp4_quant_bf16_kernel<<<g,b,0,st>>>(X, eng->d_fp4_act, eng->d_fp4_actlin, N, in_dim, eng->d_fp4_gsact);
    dim3 b2(32,8), g2((nblk+31)/32,(N+7)/8);
    nvfp4_swizzle_kernel<<<g2,b2,0,st>>>(eng->d_fp4_actlin, eng->d_fp4_actsc, N, nblk, nvfp4_pad(nblk,4));
    // alpha = gs_w · gs_act
    nvfp4_alpha_kernel<<<1,1,0,st>>>(eng->d_fp4_gsw + (l*PJ_COUNT+p), eng->d_fp4_gsact, eng->d_fp4_alpha);
    // descriptor scale pointers (weight=A, activation=B)
    cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &eng->d_fp4_wsc[l][p], sizeof(void*));
    cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &eng->d_fp4_actsc, sizeof(void*));
    cublasLtMatrixLayout_t Ad,Bd,Dd;
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_4F_E2M1, in_dim, out_dim, in_dim);
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_4F_E2M1, in_dim, N, in_dim);
    cublasLtMatrixLayoutCreate(&Dd, CUDA_R_32F, out_dim, N, out_dim);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    size_t ws=64ull<<20;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t res[4]; int found=0;
    cublasLtMatmulAlgoGetHeuristic(eng->cublaslt, eng->fp4_desc, Ad, Bd, Dd, Dd, pref, 4, res, &found);
    bool okrun=false;
    if (found>0) {
        cublasStatus_t s = cublasLtMatmul(eng->cublaslt, eng->fp4_desc,
            eng->d_fp4_alpha, eng->d_fp4_w[l][p], Ad, eng->d_fp4_act, Bd,
            eng->d_fp4_alpha+1, dst, Dd, dst, Dd, &res[0].algo, eng->d_fp4_ws, ws, st);
        okrun = (s==CUBLAS_STATUS_SUCCESS);
    }
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Dd);
    return okrun;
}

// gemm_nvfp4 variant for the Qwen3.5 hybrid: takes the weight's E2M1 buffer, its swizzled E4M3
// scales, and its fp32 global scale as EXPLICIT device pointers (proj_desc can't describe the
// hybrid's 2×-wide gated-query / GDN tensors). Reuses all shared machinery (cublaslt, fp4_desc,
// activation scratch, gsact/amax/alpha/ws). dst[out×N] = W[out×in] @ X[in×N], X bf16.
static bool gemm_nvfp4_q35(gemma4_engine_t *eng, const uint8_t *w_fp4, const uint8_t *wsc,
        const float *gsw_ptr, const __nv_bfloat16 *X, float *dst,
        int in_dim, int out_dim, int N, cudaStream_t st) {
    if (ensure_fp4_act(eng, N)!=0) return false;
    int nblk=in_dim/NVFP4_BLK;
    cudaMemsetAsync(eng->d_fp4_actsc, 0, (size_t)nvfp4_pad(N,128)*nvfp4_pad(nblk,4), st);
    cudaMemsetAsync(eng->d_fp4_amax,0,sizeof(float),st);
    uint64_t nn=(uint64_t)N*in_dim;
    nvfp4_amax_bf16_kernel<<<(unsigned)((nn+255)/256),256,0,st>>>(X, nn, eng->d_fp4_amax);
    nvfp4_gs_kernel<<<1,1,0,st>>>(eng->d_fp4_amax, eng->d_fp4_gsact);
    dim3 b(256), g((nblk+255)/256, N);
    nvfp4_quant_bf16_kernel<<<g,b,0,st>>>(X, eng->d_fp4_act, eng->d_fp4_actlin, N, in_dim, eng->d_fp4_gsact);
    dim3 b2(32,8), g2((nblk+31)/32,(N+7)/8);
    nvfp4_swizzle_kernel<<<g2,b2,0,st>>>(eng->d_fp4_actlin, eng->d_fp4_actsc, N, nblk, nvfp4_pad(nblk,4));
    nvfp4_alpha_kernel<<<1,1,0,st>>>(gsw_ptr, eng->d_fp4_gsact, eng->d_fp4_alpha);   // gsw_ptr (was l*PJ+p)
    cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &wsc, sizeof(void*));
    cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &eng->d_fp4_actsc, sizeof(void*));
    cublasLtMatrixLayout_t Ad,Bd,Dd;
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_4F_E2M1, in_dim, out_dim, in_dim);
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_4F_E2M1, in_dim, N, in_dim);
    cublasLtMatrixLayoutCreate(&Dd, CUDA_R_32F, out_dim, N, out_dim);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    size_t ws=64ull<<20;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t res[4]; int found=0;
    cublasLtMatmulAlgoGetHeuristic(eng->cublaslt, eng->fp4_desc, Ad, Bd, Dd, Dd, pref, 4, res, &found);
    bool okrun=false;
    if (found>0) {
        cublasStatus_t s = cublasLtMatmul(eng->cublaslt, eng->fp4_desc,
            eng->d_fp4_alpha, w_fp4, Ad, eng->d_fp4_act, Bd,
            eng->d_fp4_alpha+1, dst, Dd, dst, Dd, &res[0].algo, eng->d_fp4_ws, ws, st);
        okrun = (s==CUBLAS_STATUS_SUCCESS);
    }
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Dd);
    return okrun;
}

// ─────────────────────────────────────────────────────────────────────────
// NVFP4 SINGLE-STORE RESIDENCY (FORMAT_NVFP4) — load native NVFP4 weights from a safetensors
// checkpoint directly into the persistent prefill fields (d_fp4_w / d_fp4_wsc / d_fp4_gsw), so
// gemm_nvfp4 above drives prefill UNCHANGED and the decode GEMV reads the same store. Unlike
// build_fp4_weights (which dequants Q4_0→bf16→requantizes), here the E2M1 weights and E4M3 block
// scales come straight off disk — no Q4_0 copy is kept, saving ~6 GB on the 12B (the memory win).
//
// On disk each projection is: packed E2M1 [out, in/2] (low nibble=even k), LINEAR E4M3 block
// scales [out, in/16], and an FP32 global = amax/(6*448). cuBLASLt wants the block scales in its
// 32×4×4 swizzled layout (same as the activation path) and the global folded into alpha — so we
// upload the packed weights as-is, swizzle the linear scales with nvfp4_swizzle_kernel, and store
// the global into d_fp4_gsw. Returns 0 on success, <0 on failure (caller errors out — there is no
// Q4_0 fallback for an NVFP4 model). Norm/embed/lm_head (BF16) are loaded by the create path.
static int nvfp4_load_from_safetensors(gemma4_engine_t *eng, const char *path,
                                       const nvfp4ld::Layout *layout, st::Model *model)
{
    if (eng->fp4_ready) return 0;
    const nvfp4ld::Layout &L = *layout;
    st::Model &m = *model;

    if (cublasLtCreate(&eng->cublaslt) != CUBLAS_STATUS_SUCCESS) return -1;
    if (cudaMalloc(&eng->d_fp4_gsw, (size_t)GEMMA4_CAP_LAYERS*PJ_COUNT*sizeof(float)) != cudaSuccess) return -2;
    if (cudaMalloc(&eng->d_fp4_amax, sizeof(float)) != cudaSuccess) return -2;

    // map our PJ_* projection ids onto the loader's HF-suffix ids
    static const int PJ2HF[PJ_COUNT] = {
        nvfp4ld::P_Q, nvfp4ld::P_K, nvfp4ld::P_V, nvfp4ld::P_O,
        nvfp4ld::P_GATE, nvfp4ld::P_UP, nvfp4ld::P_DOWN };

    size_t wbytes = 0;
    int ok = 1;
    for (int l = 0; ok && l < eng->cfg.n_layers; l++) {
        for (int p = 0; ok && p < PJ_COUNT; p++) {
            uint64_t off; int in_dim, out_dim;
            if (!proj_desc(eng, l, p, &off, &in_dim, &out_dim)) continue;  // e.g. global V (=K)
            nvfp4ld::ProjKeys k = nvfp4ld::proj_keys(L, l, PJ2HF[p]);
            const st::Tensor *tw = m.find(k.packed);
            const st::Tensor *ts = m.find(k.scale);
            const st::Tensor *tg = m.find(k.gscale);
            if (!tw || !ts || !tg) {
                fprintf(stderr, "fucina: NVFP4 missing tensor for L%d P%d (%s)\n", l, p, k.packed.c_str());
                ok = 0; break;
            }
            const size_t packed_bytes = (size_t)out_dim * (in_dim / 2);
            const size_t lin_bytes    = (size_t)out_dim * (in_dim / NVFP4_BLK);
            if (tw->nbytes != packed_bytes || ts->nbytes != lin_bytes || tg->nbytes < sizeof(float)) {
                fprintf(stderr, "fucina: NVFP4 shape mismatch L%d P%d (packed %zu vs %zu, scale %zu vs %zu)\n",
                        l, p, tw->nbytes, packed_bytes, ts->nbytes, lin_bytes);
                ok = 0; break;
            }
            // packed E2M1 weights → persistent device store (verbatim, layout matches d_fp4_w)
            if (cudaMalloc(&eng->d_fp4_w[l][p], packed_bytes) != cudaSuccess) { ok = 0; break; }
            cudaMemcpy(eng->d_fp4_w[l][p], tw->data, packed_bytes, cudaMemcpyHostToDevice);
            // linear E4M3 scales → temp device buffer → swizzle into d_fp4_wsc
            const int nblk = in_dim / NVFP4_BLK;
            const size_t swsz = (size_t)nvfp4_pad(out_dim,128) * nvfp4_pad(nblk,4);
            uint8_t *d_lin = nullptr;
            if (cudaMalloc(&d_lin, lin_bytes) != cudaSuccess) { ok = 0; break; }
            cudaMemcpy(d_lin, ts->data, lin_bytes, cudaMemcpyHostToDevice);
            // Retain a LINEAR-scale copy for the decode GEMV (it cannot read the swizzled
            // cuBLASLt scales). Copied straight from host ts->data — identical bytes to d_lin,
            // independent of d_lin's lifetime (which the swizzle below frees). Freed in destroy.
            if (cudaMalloc(&eng->d_fp4_wsc_lin[l][p], lin_bytes) != cudaSuccess) { cudaFree(d_lin); ok = 0; break; }
            cudaMemcpy(eng->d_fp4_wsc_lin[l][p], ts->data, lin_bytes, cudaMemcpyHostToDevice);
            if (cudaMalloc(&eng->d_fp4_wsc[l][p], swsz) != cudaSuccess) { cudaFree(d_lin); ok = 0; break; }
            cudaMemset(eng->d_fp4_wsc[l][p], 0, swsz);
            dim3 b2(32,8), g2((nblk+31)/32,(out_dim+7)/8);
            nvfp4_swizzle_kernel<<<g2,b2>>>(d_lin, eng->d_fp4_wsc[l][p], out_dim, nblk, nvfp4_pad(nblk,4));
            cudaFree(d_lin);
            // per-tensor global → normalized decode MULTIPLIER (compressed-tensors stores the
            // reciprocal; see nvfp4ld::global_mul) → d_fp4_gsw[l*PJ+p]
            float gmul = nvfp4ld::global_mul(L.naming, *reinterpret_cast<const float*>(tg->data));
            cudaMemcpy(eng->d_fp4_gsw + (l*PJ_COUNT + p), &gmul, sizeof(float), cudaMemcpyHostToDevice);
            wbytes += packed_bytes + swsz + lin_bytes;
        }
    }
    cudaDeviceSynchronize();
    if (!ok || cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "fucina: NVFP4 safetensors residency failed\n");
        return -3;
    }

    // prefill activation scalars + workspace + cached cuBLASLt desc (identical to build_fp4_weights)
    cudaMalloc(&eng->d_fp4_gsact, sizeof(float));
    cudaMalloc(&eng->d_fp4_alpha, 2*sizeof(float));
    eng->d_fp4_ws = nullptr; cudaMalloc(&eng->d_fp4_ws, 64ull<<20);
    cublasLtMatmulDescCreate(&eng->fp4_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opT=CUBLAS_OP_T, opN=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    int32_t smode=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &smode, sizeof(smode));
    cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &smode, sizeof(smode));
    int32_t pmode=CUBLASLT_POINTER_MODE_DEVICE;
    cublasLtMatmulDescSetAttribute(eng->fp4_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pmode, sizeof(pmode));
    eng->fp4_ready = 1;
    (void)path;
    fprintf(stderr, "fucina: NVFP4 safetensors store resident (%.2f GB, single-store — no Q4_0 copy)\n",
            wbytes/1e9);
    return 0;
}

// Decode/GEMV projection over the NVFP4 single store: y[out] = (W_nvfp4 · x) for B=1, using the
// fused decode kernel (nvfp4_gemv.cuh). Sources the LINEAR E4M3 block scales from the engine's
// retained per-projection copy (d_fp4_wsc_lin — d_fp4_wsc holds the SWIZZLED cuBLASLt scales the
// GEMV cannot read) and the normalized per-tensor global from d_fp4_gsw[l*PJ_COUNT+p]. Capture-
// safe (no alloc / host sync; device-resident gs scalar). x and y are the float decode scratch.
static inline void nvfp4_decode_proj(gemma4_engine_t *eng, int l, int p,
        const float *x, float *y, int in_dim, int out_dim, cudaStream_t st) {
    nvfp4_gemv_launch(y, eng->d_fp4_w[l][p], eng->d_fp4_wsc_lin[l][p],
                      eng->d_fp4_gsw + (l*PJ_COUNT + p), x, in_dim, out_dim, st);
}

// Batched (K-row) NVFP4 projection for the spec-verify forward: y[K][out] = X[K][in]·W with the
// weight read ONCE for all K (vs nvfp4_decode_proj's per-row K× re-read). Transposes the token-major
// activation X[K][in] → Xt[in][K] into the pre-allocated d_specxt scratch (so the K values for an
// input index are contiguous/coalesced), then runs the weight-read-once batched GEMV. Output stays
// token-major [K][out]. Capture-safe: no alloc / host sync (d_specxt pre-allocated in ensure_spec_scratch).
static inline void nvfp4_decode_proj_batched(gemma4_engine_t *eng, int l, int p,
        const float *x, float *y, int in_dim, int out_dim, int K, cudaStream_t st) {
    nvfp4_xT_launch(eng->d_specxt, x, in_dim, K, st);
    nvfp4_gemv_batched_launch(y, eng->d_fp4_w[l][p], eng->d_fp4_wsc_lin[l][p],
                              eng->d_fp4_gsw + (l*PJ_COUNT + p), eng->d_specxt,
                              in_dim, out_dim, K, st);
}

// Lazily allocate the tiled-MMQ prefill activation scratch (Q4_0 models only): the
// quantized [N × in_dim] activation for N ≤ MMQ_MAX_N over the widest projection
// (in_dim = INTERMEDIATE). Idempotent. Returns 0 on success, -1 on failure (caller
// falls back to the BF16 path).
static int ensure_mmq_scratch(gemma4_engine_t *eng)
{
    if (eng->mmq_ready) return 0;
    const size_t Nmax = GEMMA4_MMQ_MAX_N;
    // Size the int8 activation scratch by the LARGEST projection in_dim, not just the FFN
    // intermediate: Qwen3's o-projection reads in_dim = n_heads·head_dim (= oq), and Qwen3-MoE
    // has a small cfg.intermediate (the dense FFN is unused — experts are separate), so oq can
    // exceed it. Under-sizing here would overflow d_pf_qx on the o-proj quantize. Cover both
    // the Qwen3 head_dim and the Gemma global head dim.
    size_t I = eng->cfg.intermediate;
    if ((size_t)eng->cfg.hidden_size > I) I = eng->cfg.hidden_size;
    size_t oq  = (size_t)eng->cfg.n_heads * (size_t)eng->cfg.head_dim;
    size_t oqg = (size_t)eng->cfg.n_heads * (size_t)GEMMA4_GLOBAL_HEAD_DIM;
    if (oq  > I) I = oq;
    if (oqg > I) I = oqg;
    int ok = 1;
    if (cudaMalloc(&eng->d_pf_qx, Nmax * I * sizeof(int8_t))     != cudaSuccess) ok = 0;
    if (cudaMalloc(&eng->d_pf_dx, Nmax * (I/32) * sizeof(float)) != cudaSuccess) ok = 0;
    if (cudaMalloc(&eng->d_pf_sx, Nmax * (I/32) * sizeof(int))   != cudaSuccess) ok = 0;
    if (!ok) {
        fprintf(stderr, "fucina: MMQ prefill scratch alloc failed — BF16 fallback\n");
        cudaGetLastError();
        if (eng->d_pf_qx) { cudaFree(eng->d_pf_qx); eng->d_pf_qx = NULL; }
        if (eng->d_pf_dx) { cudaFree(eng->d_pf_dx); eng->d_pf_dx = NULL; }
        if (eng->d_pf_sx) { cudaFree(eng->d_pf_sx); eng->d_pf_sx = NULL; }
        return -1;
    }
    fprintf(stderr, "fucina: tiled-MMQ prefill scratch (%.2f GB, native Q4_0 weights, "
            "no BF16 dequant for N ≤ %d)\n",
            (Nmax*I + Nmax*(I/32)*8) / 1e9, GEMMA4_MMQ_MAX_N);
    eng->mmq_ready = 1;
    return 0;
}

// Tiled-MMQ projection: Y[N×out_dim] = W_q4_0[out_dim×in_dim] @ X[N×in_dim], reading the
// native Q4_0 weight ONCE (no BF16 materialize). Quantizes the BF16 activation xb to int8
// in-place into the prefill scratch, then runs the one-weight-pass MMQ kernel. Replaces a
// dequant_layer_bf16 + gemm_bf16 pair for N ≤ MMQ_MAX_N on Q4_0 models.
static void mmq_proj(
    gemma4_engine_t *eng, const uint8_t *wq, const __nv_bfloat16 *xb,
    float *dst, int in_dim, int out_dim, int N, cudaStream_t stream)
{
    // One warp per 32-elem block over the whole flattened [N×in_dim] activation; the
    // kernel's length bound is the TOTAL element count (in_dim multiple of 32 ⇒ no block
    // straddles a token), matching the dp4a decode path's quantize launch convention.
    quantize_q8_1_bf16_kernel<<<(unsigned)((size_t)N*in_dim/32), 32, 0, stream>>>(
        xb, eng->d_pf_qx, eng->d_pf_dx, eng->d_pf_sx, N*in_dim);
    mmq_q4_0_launch(dst, wq, eng->d_pf_qx, eng->d_pf_dx, eng->d_pf_sx,
                    in_dim, out_dim, N, stream);
}

// Whether the prefill should take the native Q4_0 tiled-MMQ path for a batch of N tokens.
//
// MEASURED on GB10 (DGX Spark): the dp4a tiled-MMQ kernel is bit-faithful but ~5× SLOWER
// than the BF16 tensor-core GEMM for the N ≤ 1024 prefill range it targets — BF16 tensor
// cores far outrun dp4a on CUDA cores for these compute-bound GEMM shapes, and Phase 1's
// pipelined dequant already hides the per-layer dequant the MMQ path was meant to remove.
// (e.g. 529-tok suffix prefill: ~1245 tok/s BF16 vs ~225 tok/s MMQ.) Beating BF16 would
// require an int8 TENSOR-CORE (IMMA/mma.sync) MMQ — a separate, larger effort.
//
// So the MMQ path is compiled under BLACKWELL_NATIVE_FP4 (kernel kept for the future IMMA
// rework and for A/B) but is OFF by default; opt in at runtime with FUCINA_MMQ=1. The
// default prefill stays on the faster BF16 + always-pipelined-dequant path.
static inline bool mmq_enabled(gemma4_engine_t *eng, int N)
{
#if BLACKWELL_NATIVE_FP4
    static int opt = -1;
    if (opt < 0) { const char *e = getenv("FUCINA_MMQ"); opt = (e && e[0] == '1') ? 1 : 0; }
    if (!opt) return false;
    return (FMT(eng) == FORMAT_Q4_0) && (N <= GEMMA4_MMQ_MAX_N)
           && ensure_mmq_scratch(eng) == 0;
#else
    (void)eng; (void)N; return false;
#endif
}


// =========================================================================
// ─── Batched Prefill (Step 2 Phase 2 + Step 3) ──────────────────────────
static int alloc_prefill_scratch(gemma4_engine_t *eng);   // fwd decl (defined below)
static int ensure_fp_scratch(gemma4_engine_t *eng);       // fwd decl (defined below)
// =========================================================================
// Process the WHOLE prompt in one weight pass: per layer, one cuBLAS BF16 GEMM
// per projection over all N tokens (tensor-core bound) instead of N token-by-
// token GEMV passes (bandwidth bound). Reproduces decode_layer's math exactly,
// batched. Requires a FRESH sequence (global_n_tokens==0) so the batch's own
// linear K/V are the full attention context; returns -2 otherwise (caller falls
// back to gemma4_engine_prefill). Builds the BF16 weights on first use.
int gemma4_engine_prefill_batched(
    gemma4_engine_t *eng, const int32_t *tokens, int n_tokens, float *logits_out)
{
    eng->mtp_h_valid = 0;   // prefill invalidates the MTP drafter's recurrent h
    eng->abort_req = 0;     // chain head: a new prefill clears any stale abort
                            // (the -2 fallthroughs to flash/chunked keep it live)

    if (!eng->loaded || n_tokens <= 0) return -1;
    // Qwen3 has a different per-layer layout (full-causal all-layers, separate V,
    // q/k norms, silu-glu, no softcap) and is supported ONLY by the paged
    // multiseq path (paged_prefill_qwen3 / decode_multiseq_forward). This non-paged
    // eng->cur prefill is gemma-layout-only: running it on Qwen3 weights corrupts
    // the CUDA context (illegal launch → "invalid device context") and SIGSEGVs the
    // single-flight HTTP path. Decline early so warmup and any caller stay safe.
    // qwen35 is likewise paged-multiseq-only (its own hybrid forward); decline here too.
    if (eng->cfg.arch == GEMMA4_ARCH_QWEN3_5) return -2;
    if (eng->cur.n_tokens != 0) return -2;             // need fresh sequence
    if (n_tokens > eng->global_kv_capacity) return -2;    // would overflow cache
    // Batched attention materializes [HEADS][N×N] score buffers (fp32+bf16, ~6 B/elem).
    // At large N that is many GB (e.g. ~14 GB @ 11.7k tokens) — the giant per-request
    // cudaMalloc intermittently stalled or failed (the "stuck" prefills), so commit
    // 5fc80c9 routed everything >128 to the scalar FLASH path. But nsys showed flash
    // attention is ~70% of prefill wall time at 2k ctx (scalar fp8 loads, no tensor
    // cores: 615 tok/s vs llama.cpp's ~1200). Fix the ALLOC problem instead of avoiding
    // the fast path: N ≤ 4096 uses the PERSISTENT prefill scratch (~2 GB, allocated
    // once, no per-request cudaMalloc — no fragmentation), tensor-core GEMM attention
    // throughout. Larger prompts still use the chunked FLASH prefill (O(chunk+KV) mem).
    if (n_tokens > 4096) return -2;  // flash prefill beyond the persistent scratch size
    if (!eng->pf_scratch_ready && alloc_prefill_scratch(eng) != 0)
        return -2;                   // scratch alloc failed → flash fallback

    cudaStream_t stream = eng->stream;
    const int N   = n_tokens;
    const int H   = eng->cfg.hidden_size;
    const int I   = eng->cfg.intermediate;
    const int HD2 = 32 * sizeof(float);
    const int base = 0;

    // Tiled-MMQ fast path (BLACKWELL_NATIVE_FP4 default): Q4_0 weights, small/mid batch.
    // Reads the native Q4_0 weights ONCE per projection (no BF16 materialize, no per-layer
    // dequant pass) — eliminating the ~125 ms fixed full-model dequant that dominated
    // suffix prefills. Above MMQ_MAX_N the BF16 tensor-core GEMM (with pipelined dequant)
    // wins, so we fall through to it. mmq needs no BF16 scratch; only build it otherwise.
    bool use_mmq = mmq_enabled(eng, N);   // forced false for FORMAT_NVFP4 (no Q4_0 store)
    // NVFP4 tensor-core prefill (FUCINA_FP4=1): persistent NVFP4 weights + per-GEMM
    // activation quant + cuBLASLt block-scaled FP4 GEMM (~2.4× BF16). Needs no per-layer
    // dequant pipeline. Falls back to BF16 if weight/scratch build fails.
    // NVFP4 wins only for large-batch prefill: at small N the per-GEMM activation-quant
    // overhead cancels the tensor-core speedup (measured: ~tied ≤256 tok, 1.47× @2113).
    // DEFAULT ON with a 1024-token floor: that floor is tool-eval-validated CLEAN (core-15
    // 97/100 == BF16 baseline), whereas a 256 floor regressed an error-handling scenario
    // (90/100) — FP4 on the 256-1024 multi-turn prefills was the culprit. FUCINA_FP4=0 opts
    // out entirely; FUCINA_FP4_MIN overrides the floor. Lazy: weights build only when a
    // prefill actually clears the floor, so short-prompt sessions pay nothing.
    static int fp4_opt = -1, fp4_min = 1024, fp4_force = 0;
    if (fp4_opt < 0) {
        const char *e = getenv("FUCINA_FP4"); fp4_opt = (e && e[0]=='0') ? 0 : 1;  // on unless =0
        fp4_force = (e && e[0]=='1') ? 1 : 0;   // EXPLICIT =1 force-overrides the mem budget
        const char *mn = getenv("FUCINA_FP4_MIN"); if (mn) fp4_min = atoi(mn);
    }
    // The --gpu-mem-util budget block decided whether the lazy NVFP4-prefill copy (~16.5 GiB
    // on the 31B) fits co-resident. Honor that by DEFAULT: if it doesn't fit, never build it —
    // prefill falls back to BF16/MMQ (correct, slower). An EXPLICIT FUCINA_FP4=1 force-overrides
    // (build anyway, with a one-time over-budget warning). FORMAT_NVFP4 is handled below and is
    // unaffected (no separate copy; fp4_budget_ok is irrelevant there).
    bool fp4_budget = eng->fp4_budget_ok || fp4_force;
    if (fp4_force && !eng->fp4_budget_ok && !eng->fp4_ready && fp4_opt && !use_mmq && N >= fp4_min) {
        static int warned = 0;
        if (!warned) { warned = 1;
            fprintf(stderr, "fucina: WARNING FUCINA_FP4=1 forces the NVFP4-prefill copy "
                            "(~%.2f GiB) which EXCEEDS the --gpu-mem-util budget\n",
                            gemma4_fp4_prefill_footprint(eng) / (1024.0*1024.0*1024.0));
        }
    }
    bool use_fp4 = fp4_opt && fp4_budget && !use_mmq && N >= fp4_min
                         && build_fp4_weights(eng) == 0 && ensure_fp4_act(eng, N) == 0;
    // FORMAT_NVFP4 has NO Q4_0/BF16 fallback store (eng->d_weights is NULL): the prefill MUST
    // always use gemm_nvfp4, for EVERY N. Bypass the 1024-token floor and the MMQ/BF16 paths.
    // build_fp4_weights is a no-op here (fp4_ready=1 from residency); ensure_fp4_act sets up the
    // cuBLASLt activation-quant scratch and must succeed.
    if (eng->format == FORMAT_NVFP4) {
        if (ensure_fp4_act(eng, N) != 0) {
            fprintf(stderr, "fucina: NVFP4 activation scratch alloc failed (N=%d)\n", N);
            return -1;
        }
        use_mmq = false; use_fp4 = true;
    }
    if (!use_mmq && !use_fp4 && build_bf16_weights(eng) != 0) return -1;

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, stream);

    // BF16 path always pipelines the per-layer dequant (Phase 1, now unconditional): the
    // next layer's Q4_0→BF16 dequant runs on dq_stream while this layer's GEMMs run. Graph
    // capture cannot join that fork, so it stays off when we run the BF16 path (overlap is
    // the larger, every-N win). MMQ needs no dequant at all → ovl is false there.
    const bool ovl = !use_mmq && !use_fp4;
    const bool use_graph = false;
    bool graph_replay = false;
    if (use_graph && eng->graph.N == N && eng->graph.e) {
        graph_replay = true; eng->graph.hits++;
    } else if (use_graph) {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        eng->graph.misses++;
    }

    // ── Scratch (token-major). Sized to the widest dim each can take. ──
    const int OQ_MAX = eng->cfg.n_heads * GEMMA4_GLOBAL_HEAD_DIM; // 8192 (12B) / 16384 (31B)
    const int OKV_MAX = eng->cfg.n_kv_sliding * GEMMA4_HEAD_DIM;  // 2048 (12B) / 4096 (31B)
    const int HEADS = eng->cfg.n_heads;
    float *d_x=0,*d_norm=0,*d_q=0,*d_k=0,*d_v=0,*d_attn=0,*d_gate=0,*d_up=0,*d_scores=0;
    __nv_bfloat16 *d_inb=0,*d_qb=0,*d_kb=0,*d_vb=0,*d_kbx=0,*d_vbx=0,*d_pb=0;
    bool own_bufs = false;

    if (eng->pf_scratch_ready && N <= 4096) {
        d_x = eng->d_pf_x; d_norm = eng->d_pf_norm; d_q = eng->d_pf_q;
        d_k = eng->d_pf_k; d_v = eng->d_pf_v; d_attn = eng->d_pf_attn;
        d_gate = eng->d_pf_gate; d_up = eng->d_pf_up; d_scores = eng->d_pf_scores;
        d_inb = eng->d_pf_inb; d_qb = eng->d_pf_qb; d_kb = eng->d_pf_kb;
        d_vb = eng->d_pf_vb; d_kbx = eng->d_pf_kbx; d_vbx = eng->d_pf_vbx;
        d_pb = eng->d_pf_pb;
    } else {
        own_bufs = true;
        int ok = 1;
    #define PALLOC(p,elems) do{ if(cudaMalloc(&(p),(size_t)(elems))!=cudaSuccess){ok=0;} }while(0)
    PALLOC(d_x,    (size_t)N*H*sizeof(float));
    PALLOC(d_norm, (size_t)N*H*sizeof(float));
    PALLOC(d_q,    (size_t)N*OQ_MAX*sizeof(float));
    PALLOC(d_k,    (size_t)N*OKV_MAX*sizeof(float));
    PALLOC(d_v,    (size_t)N*OKV_MAX*sizeof(float));
    PALLOC(d_attn, (size_t)N*OQ_MAX*sizeof(float));
    PALLOC(d_gate, (size_t)N*I*sizeof(float));
    PALLOC(d_up,   (size_t)N*I*sizeof(float));
    PALLOC(d_inb,  (size_t)N*I*sizeof(__nv_bfloat16));
    // Attention scratch (batched-head GEMM path): bf16 Q + bf16 K/V expanded to
    // all query heads, fp32 score batch [HEADS][N×N] + bf16 prob batch.
    PALLOC(d_qb,   (size_t)N*OQ_MAX*sizeof(__nv_bfloat16));
    PALLOC(d_kb,   (size_t)N*OKV_MAX*sizeof(__nv_bfloat16));
    PALLOC(d_vb,   (size_t)N*OKV_MAX*sizeof(__nv_bfloat16));
    PALLOC(d_kbx,  (size_t)N*OQ_MAX*sizeof(__nv_bfloat16));
    PALLOC(d_vbx,  (size_t)N*OQ_MAX*sizeof(__nv_bfloat16));
    PALLOC(d_pb,   (size_t)HEADS*N*N*sizeof(__nv_bfloat16));
    PALLOC(d_scores,(size_t)HEADS*N*N*sizeof(float));
    #undef PALLOC
    float *fbufs[] = {d_x,d_norm,d_q,d_k,d_v,d_attn,d_gate,d_up,d_scores};
    __nv_bfloat16 *bbufs[] = {d_inb,d_qb,d_kb,d_vb,d_kbx,d_vbx,d_pb};
    if (!ok) {
        fprintf(stderr, "fucina: batched-prefill scratch alloc failed (N=%d) — fallback\n", N);
        cudaGetLastError();
        for (float *p : fbufs) if(p) cudaFree(p);
        for (__nv_bfloat16 *p : bbufs) if(p) cudaFree(p);
        return -2;
    }
    } // else (own_bufs)

    auto grid1d = [](size_t n){ return (unsigned)((n + 255) / 256); };
    if (!graph_replay) {
    // Pipelined per-layer weight dequant (see d_bf16_layer comment). issue_dequant(m)
    // dequantizes layer m into ping/pong buffer m&1; in overlap mode it runs on
    // dq_stream and the main stream gates each layer's GEMMs on ev_dq_done. curbuf
    // selects the buffer the current layer's GEMMs read. Serial mode keeps buf 0 and
    // dequants on the main stream (bit-identical to the old path).
    int curbuf = 0;
    auto issue_dequant = [&](int m){
        int b = m & 1;
        if (m >= 2) cudaStreamWaitEvent(eng->dq_stream, eng->ev_gemm_done[b], 0);
        dequant_layer_bf16_buf(eng, m, b, eng->dq_stream);
        cudaEventRecord(eng->ev_dq_done[b], eng->dq_stream);
    };
    // MMQ: native Q4_0 weight read once, no BF16 materialize. BF16: cuBLAS tensor-core
    // GEMM over the pipelined dequant scratch (curbuf).
    auto gemm_proj = [&](int l, int p, int in_dim, int out_dim, float *dst){
        if (use_fp4) {
            gemm_nvfp4(eng, l, p, d_inb, dst, in_dim, out_dim, N, stream);
        } else if (use_mmq) {
            uint64_t off; int idim, odim;
            proj_desc(eng, l, p, &off, &idim, &odim);
            mmq_proj(eng, weight_fp8(eng, off), d_inb, dst, in_dim, out_dim, N, stream);
        } else {
            gemm_bf16(eng, eng->d_bf16_layer[curbuf][p], d_inb, dst, in_dim, out_dim, N);
        }
    };

    // ── Embedding + √H scale, token-major [N][H] ──
    embed_w(eng, d_x, weight_fp8(eng, eng->tensors.token_embd), tokens, N, H, stream);
    scale_kernel<<<grid1d((size_t)N*H),256,0,stream>>>(d_x, N*H, sqrtf((float)H));

    if (ovl) issue_dequant(0);   // pipeline fill: layer 0's weights (BF16 path only)
    for (int l = 0; l < eng->cfg.n_layers; l++) {
        layer_type_t lt = eng->layer_types[l];
        int hd  = (lt==LAYER_SLIDING)? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM;
        int nkv = (lt==LAYER_SLIDING)? eng->cfg.n_kv_sliding : eng->cfg.n_kv_global;
        int oq  = HEADS*hd, okv = nkv*hd, kvhd = okv;
        const float *w_attn   = eng->d_w_attn_norm      + (size_t)l*H;
        const float *w_post_a = eng->d_w_post_attn_norm + (size_t)l*H;
        const float *w_ffn    = eng->d_w_ffn_norm       + (size_t)l*H;
        const float *w_post_f = eng->d_w_post_ffn_norm  + (size_t)l*H;
        const float *w_qn     = eng->d_w_q_norm + (size_t)l*GEMMA4_GLOBAL_HEAD_DIM;
        const float *w_kn     = eng->d_w_k_norm + (size_t)l*GEMMA4_GLOBAL_HEAD_DIM;

        // BF16 path: kick off the NEXT layer's dequant on dq_stream, then gate this layer's
        // GEMMs on its own (already-issued) dequant. MMQ path: no dequant — gemm_proj reads
        // the native Q4_0 weights directly.
        if (ovl) {
            curbuf = l & 1;
            if (l + 1 < eng->cfg.n_layers) issue_dequant(l + 1);
            cudaStreamWaitEvent(stream, eng->ev_dq_done[curbuf], 0);
        }

        // Sandwich-norm block with IN-PLACE residual: d_x stays the pre-block
        // hidden (the residual) until residual_add folds the normed sub-block
        // contribution back into it — so no separate d_res buffer or D2D copies.
        // pre-attn RMSNorm → input prep (one input feeds Q,K,V → smooth group 0)
        rms_norm_rows_bf16_kernel<<<N,256,HD2,stream>>>(d_inb, d_x, w_attn, H, N, GEMMA4_RMS_EPS);

        // Q,K (and V) projections (one normed input feeds all three)
        gemm_proj(l, PJ_Q, H, oq, d_q);
        per_head_rms_norm_rows_kernel<<<dim3(HEADS,N),hd,HD2,stream>>>(d_q, w_qn, HEADS, hd, N, GEMMA4_RMS_EPS);
        gemm_proj(l, PJ_K, H, okv, d_k);
        if (lt == LAYER_SLIDING) {
            gemm_proj(l, PJ_V, H, okv, d_v);
            per_head_rms_norm_rows_kernel<<<dim3(nkv,N),hd,HD2,stream>>>(d_v, NULL, nkv, hd, N, GEMMA4_RMS_EPS);
        } else {
            // global: V = raw K projection (pre K-norm), plain per-head RMS
            cudaMemcpyAsync(d_v, d_k, (size_t)N*okv*sizeof(float), cudaMemcpyDeviceToDevice, stream);
            per_head_rms_norm_rows_kernel<<<dim3(nkv,N),hd,HD2,stream>>>(d_v, NULL, nkv, hd, N, GEMMA4_RMS_EPS);
        }
        per_head_rms_norm_rows_kernel<<<dim3(nkv,N),hd,HD2,stream>>>(d_k, w_kn, nkv, hd, N, GEMMA4_RMS_EPS);

        // RoPE Q,K
        if (lt == LAYER_SLIDING)
            rope_rows_kernel<<<dim3(HEADS,N),hd/2,0,stream>>>(d_q, d_k, base, HEADS, nkv, hd, N, 10000.0f, NULL);
        else
            rope_rows_kernel<<<dim3(HEADS,N),hd/2,0,stream>>>(d_q, d_k, base, HEADS, nkv, hd, N, 1000000.0f, eng->d_rope_freqs);

        // Attention → d_attn [N][oq], batched over all heads via tensor-core GEMMs:
        // expand K/V to all query heads (GQA), S=Q·Kᵀ (col-major [HEADS][N×N]) →
        // masked softmax → P(bf16) → O=V·Pᵀ. Scale 1.0 (gemma4).
        int window = (lt==LAYER_SLIDING)? GEMMA4_SLIDING_WINDOW : 0;
        f32_to_bf16_kernel<<<grid1d((size_t)N*oq),256,0,stream>>>(d_qb, d_q, (size_t)N*oq);
        f32_to_bf16_kernel<<<grid1d((size_t)N*okv),256,0,stream>>>(d_kb, d_k, (size_t)N*okv);
        f32_to_bf16_kernel<<<grid1d((size_t)N*okv),256,0,stream>>>(d_vb, d_v, (size_t)N*okv);
        {
            kv_broadcast_bf16_kernel<<<dim3(grid1d(hd),HEADS,N),256,0,stream>>>(d_kbx, d_kb, N, HEADS, nkv, hd);
            kv_broadcast_bf16_kernel<<<dim3(grid1d(hd),HEADS,N),256,0,stream>>>(d_vbx, d_vb, N, HEADS, nkv, hd);
            const float a1=1.0f, b0=0.0f;
            long long sNN = (long long)N * N;
            // S[h] = Q[h]ᵀ·K[h]  (m=N,n=N,k=hd; A,B col-major [hd×N] ld=oq stride=hd)
            cublasGemmStridedBatchedEx(eng->cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, N, hd,
                &a1, d_qb,  CUDA_R_16BF, oq, (long long)hd,
                     d_kbx, CUDA_R_16BF, oq, (long long)hd,
                &b0, d_scores, CUDA_R_32F, N, sNN,
                HEADS, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            attn_softmax_batched_kernel<<<dim3(N,HEADS),256,HD2,stream>>>(d_scores, d_pb, N, window);
            // O[h] = V[h]·P[h]ᵀ  (m=hd,n=N,k=N → C col-major [hd×N] ld=oq stride=hd)
            cublasGemmStridedBatchedEx(eng->cublas, CUBLAS_OP_N, CUBLAS_OP_T, hd, N, N,
                &a1, d_vbx, CUDA_R_16BF, oq, (long long)hd,
                     d_pb,  CUDA_R_16BF, N,  sNN,
                &b0, d_attn, CUDA_R_32F, oq, (long long)hd,
                HEADS, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }

        // Write final K/V into the persistent cache for decode continuation.
        if (lt == LAYER_SLIDING) {
            // FLAT (Step 3): persist ALL N tokens at absolute positions [base, base+N-1] (was
            // only the last `window`) — required so a later rewind+suffix-decode can attend the
            // kept prefix. base==0 here (batched prefill is fresh-only; non-fresh defers).
            size_t lstride = (size_t)eng->sliding_kv_capacity * kvhd;
            kv_t *kc = eng->d_sliding_k + (size_t)l*lstride;
            kv_t *vc = eng->d_sliding_v + (size_t)l*lstride;
            kv_write_sliding_kernel<<<dim3(grid1d(kvhd),N),256,0,stream>>>(
                kc, vc, d_k, d_v, base, 0, N, kvhd, GEMMA4_SLIDING_WINDOW,
                eng->sliding_kv_capacity);
        } else {
            // GQA global cache layout [capacity][n_kv_global][head_dim]: per-token width
            // and stride are okv = nkv*hd (12B nkv=1 ⇒ okv==hd ⇒ bit-identical to before).
            int slot = eng->global_slot[l];
            size_t stride = (size_t)eng->global_kv_capacity * okv;
            kv_write_global_kernel<<<dim3(grid1d(okv),N),256,0,stream>>>(
                eng->d_global_k + slot*stride, eng->d_global_v + slot*stride,
                d_k, d_v, base, N, okv);
        }

        // O projection → temp d_q ; post-attn norm ; fold into residual d_x
        f32_to_bf16_kernel<<<grid1d((size_t)N*oq),256,0,stream>>>(d_inb, d_attn, (size_t)N*oq);
        gemm_proj(l, PJ_O, oq, H, d_q);
        rms_norm_residual_add_kernel<<<N,256,256*sizeof(float),stream>>>(d_x, d_q, w_post_a, H, N, GEMMA4_RMS_EPS);

        // FFN: pre-norm (d_x is now the post-attn hidden = FFN residual) → bf16
        // input (reused by gate+up), GeGLU, down → temp d_q, fold into d_x.
        rms_norm_rows_bf16_kernel<<<N,256,HD2,stream>>>(d_inb, d_x, w_ffn, H, N, GEMMA4_RMS_EPS);
        gemm_proj(l, PJ_GATE, H, I, d_gate);
        gemm_proj(l, PJ_UP,   H, I, d_up);
        geglu_bf16_kernel<<<grid1d((size_t)N*I),256,0,stream>>>(d_inb, d_gate, d_up, N*I);
        gemm_proj(l, PJ_DOWN, I, H, d_q);   // last GEMM reading curbuf's weights
        if (ovl) cudaEventRecord(eng->ev_gemm_done[curbuf], stream);
        rms_norm_residual_add_kernel<<<N,256,256*sizeof(float),stream>>>(d_x, d_q, w_post_f, H, N, GEMMA4_RMS_EPS);
        if (eng->h_out_scale[l] != 1.0f)
            scale_kernel<<<grid1d((size_t)N*H),256,0,stream>>>(d_x, N*H, eng->h_out_scale[l]);
    }
    if (ovl) cudaStreamSynchronize(eng->dq_stream);  // drain any trailing dequant

    } // if (!graph_replay)

    eng->cur.n_tokens += N;

    if (!graph_replay) {
    // ── Output norm + LM head + softcap on the LAST token only ──
    float *x_last = d_x + (size_t)(N-1)*H;
    rms_norm_kernel<<<1,256,HD2,stream>>>(eng->d_norm, x_last, eng->d_w_out_norm, H, GEMMA4_RMS_EPS);
    logits_head(eng, eng->d_logits, eng->d_norm, H, GEMMA4_VOCAB_SIZE, stream);
    logit_softcap_kernel<<<grid1d(GEMMA4_VOCAB_SIZE),256,0,stream>>>(
        eng->d_logits, GEMMA4_SOFTCAP, GEMMA4_VOCAB_SIZE);
    if (eng->n_suppress > 0)
        suppress_tokens_kernel<<<grid1d(eng->n_suppress),256,0,stream>>>(
            eng->d_logits, eng->d_suppress, eng->n_suppress, GEMMA4_VOCAB_SIZE);
    if (logits_out)
        cudaMemcpyAsync(logits_out, eng->d_logits, GEMMA4_VOCAB_SIZE*sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    } // if (!graph_replay)

    // D2H copy always needed (graph captures everything except host copy)
    if (graph_replay && logits_out)
        cudaMemcpyAsync(logits_out, eng->d_logits, GEMMA4_VOCAB_SIZE*sizeof(float),
                        cudaMemcpyDeviceToHost, stream);

    if (use_graph && !graph_replay) {
        cudaGraph_t g;
        cudaStreamEndCapture(stream, &g);
        cudaGraphInstantiate(&eng->graph.e, g, NULL, NULL, 0);
        if (eng->graph.g) cudaGraphDestroy(eng->graph.g);
        eng->graph.g = g; eng->graph.N = N;
        cudaGraphLaunch(eng->graph.e, stream);
    } else if (graph_replay) {
        cudaGraphLaunch(eng->graph.e, stream);
    }

    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    eng->prefill_time_ms += ms;
    eng->n_prefill_tokens += N;
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    if (own_bufs) {
        float *fb[]={d_x,d_norm,d_q,d_k,d_v,d_attn,d_gate,d_up,d_scores};
        __nv_bfloat16 *bb[]={d_inb,d_qb,d_kb,d_vb,d_kbx,d_vbx,d_pb};
        for (float *p : fb) if(p) cudaFree(p);
        for (__nv_bfloat16 *p : bb) if(p) cudaFree(p);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "fucina: batched-prefill CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Test override: when set, gemma4_engine_seq_add forces the token-by-token prefill
// (skips paged_prefill_batched). Used ONLY by the dual-path determinism self-test.
int g_fucina_force_slow_prefill = 0;

// Test override: when set, the qwen35 base>0 chunked-prefill continuation forces the scalar
// qwen35_b_attn_kernel instead of the tensor-core path. Used ONLY by test_qwen35_chunk_parity.
int g_fucina_q35_scalar_cont_attn = 0;

// ── Qwen3-MoE sparse FFN ────────────────────────────────────────────────────────────────────────
// The dense SiLU-GLU FFN is replaced by a 128-expert top-8 mixture. The router is a PLAIN GEMV
// (logits = ffn_gate_inp @ ffn_norm(x); NO rmsnorm, NO 1/sqrt(d), NO per-feature/per-expert scale,
// NO sigmoid/bias) — confirmed vs llama.cpp src/models/qwen3moe.cpp build_moe_ffn(GATING=SOFTMAX,
// norm_w=true, scale=1, exp_probs_b=nullptr). softmax over all 128 → top-8 → renormalize the 8
// weights to sum 1 (dg_softmax_topk does exactly this) → expert_out = Σ_k w_k·down_k(silu(gate_k(h))
// ·up_k(h)). Activations stay FP32 column-major [feat,tokens] (matching the gemma4 token-major row
// layout) so the dg_* grouped kernels apply directly. Do NOT mirror the DiffusionGemma router
// (which adds a pre-rmsnorm / 1-sqrt-d / per-expert pes scale) — Qwen3-MoE has none of those.

// Router GEMV: logits[t·E + e] = Σ_h W[e·H + h] · X[t·H + h]. W = ffn_gate_inp (F32, [hidden,n_expert]
// row-major with hidden contiguous → row e is W+e·H). One block per (expert, token-PAIR): the
// expert-row chunk is hoisted into registers once and reused for both tokens, and the 8-wide load
// hoisting keeps ~24 independent loads in flight per thread — this kernel is DRAM-LATENCY bound
// (44 us/layer at cn=16 in the naive one-block-per-(e,t) form; 33 us with this shape, cold-L2
// measured on GB10; a one-block-per-expert smem-staged form measured WORSE at 47-49 us: only E
// blocks can't hide the latency). Per (e,t) the FP evaluation order is EXACTLY the classic form —
// thread t accumulates k = t + i·blockDim ascending (hoisting reorders loads, not adds; the
// per-chunk k sequence i, i+256, …, i+7·256, i+2048, … is the same ascending sequence), then the
// identical 256-wide tree reduction → bit-identical logits.
enum { MOE_PROFILE_ACT_STAGES = 5 };

// Calibration-only activation magnitude reduction. A bounded grid walks the complete
// tensor, then contributes one sum/max per block; normal serving never launches it.
__global__ void moe_profile_activation_kernel(const float *x, size_t n,
                                               double *sum_squares,
                                               unsigned long long *elements,
                                               unsigned int *max_bits) {
    __shared__ double ss[256];
    __shared__ float mm[256];
    double sum = 0.0; float mx = 0.0f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (size_t)gridDim.x * blockDim.x) {
        float v = x[i], a = fabsf(v); sum += (double)v * v; mx = fmaxf(mx, a);
    }
    ss[threadIdx.x] = sum; mm[threadIdx.x] = mx; __syncthreads();
    for (int d = 128; d; d >>= 1) {
        if (threadIdx.x < d) { ss[threadIdx.x] += ss[threadIdx.x+d]; mm[threadIdx.x] = fmaxf(mm[threadIdx.x], mm[threadIdx.x+d]); }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(sum_squares, ss[0]);
        atomicMax(max_bits, __float_as_uint(mm[0]));
        if (blockIdx.x == 0) atomicAdd(elements, (unsigned long long)n);
    }
}

static void moe_profile_activation(gemma4_engine_t *eng, int layer, int stage,
                                   const float *x, size_t n, cudaStream_t stream) {
    if (!eng->d_moe_profile_act_ss || !x || n == 0 || stage < 0 || stage >= MOE_PROFILE_ACT_STAGES) return;
    int blocks = (int)((n + 255) / 256); if (blocks > 32) blocks = 32;
    size_t i = (size_t)layer * MOE_PROFILE_ACT_STAGES + stage;
    moe_profile_activation_kernel<<<blocks,256,0,stream>>>(x, n,
        eng->d_moe_profile_act_ss + i, eng->d_moe_profile_act_n + i,
        eng->d_moe_profile_act_max + i);
}

// Calibration-only top-k telemetry; never launched unless profiling was started.
__global__ void moe_profile_accum_kernel(const int *idx, const float *weight,
                                         unsigned long long *counts, double *weight_sums,
                                         int assignments) {
    int i = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= assignments) return;
    int e = idx[i];
    atomicAdd(counts + e, 1ULL);
    atomicAdd(weight_sums + e, (double)weight[i]);
}

__global__ void moe_router_gemv_kernel(const float *__restrict__ W, const float *__restrict__ X,
                                       float *__restrict__ logits, int H, int n_expert, int cn) {
    const int e = blockIdx.x, t0 = blockIdx.y * 2;
    const bool two = (t0 + 1 < cn);
    const float *w = W + (size_t)e * H;
    const float *x0 = X + (size_t)t0 * H;
    const float *x1 = x0 + H;
    float acc0 = 0.f, acc1 = 0.f;
    for (int i = threadIdx.x; i < H; i += (int)blockDim.x * 8) {
        float wv[8], xv0[8], xv1[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int k = i + j * (int)blockDim.x;
            if (k < H) { wv[j] = w[k]; xv0[j] = x0[k]; if (two) xv1[j] = x1[k]; }
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int k = i + j * (int)blockDim.x;
            if (k < H) acc0 += wv[j] * xv0[j];
        }
        if (two)
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int k = i + j * (int)blockDim.x;
                if (k < H) acc1 += wv[j] * xv1[j];
            }
    }
    __shared__ float sm[256];
    for (int b = 0; b < (two ? 2 : 1); b++) {
        sm[threadIdx.x] = b ? acc1 : acc0; __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) logits[(size_t)(t0 + b) * n_expert + e] = sm[0];
        __syncthreads();
    }
}

// Lazily allocate the MoE forward scratch (QWEN3MOE only). Sized for ≤ GEMMA4_MOE_TMAX tokens per
// chunk (total assignments = tokens·n_experts_used). Returns 0 ok / -1 on alloc failure.
static int moe_alloc_scratch(gemma4_engine_t *eng) {
    if (eng->moe_scratch_ready) return 0;
    const int H = eng->cfg.hidden_size, EFFN = eng->cfg.expert_ffn;
    const int E = eng->cfg.n_experts, U = eng->cfg.n_experts_used;
    const int T = GEMMA4_MOE_TMAX, A = T * U;
    int ok = 1;
    size_t allocated_bytes = 0;
    #define MOE_A(p, bytes) do { \
        size_t _n = (size_t)(bytes); \
        if (cudaMalloc(&(p), _n) != cudaSuccess) { cudaGetLastError(); ok = 0; } \
        else allocated_bytes += _n; \
    } while (0)
    MOE_A(eng->d_moe_rlogits, (size_t)T * E * sizeof(float));
    MOE_A(eng->d_moe_tki,     (size_t)A * sizeof(int));
    MOE_A(eng->d_moe_tkw,     (size_t)A * sizeof(float));
    MOE_A(eng->d_moe_eidx,    (size_t)A * sizeof(int));
    MOE_A(eng->d_moe_invpos,  (size_t)A * sizeof(int));
    MOE_A(eng->d_moe_ecs,     (size_t)A * sizeof(float));
    MOE_A(eng->d_moe_count,   (size_t)E * sizeof(int));
    MOE_A(eng->d_moe_coloff,  (size_t)E * sizeof(int));
    MOE_A(eng->d_moe_cursor,  (size_t)E * sizeof(int));
    MOE_A(eng->d_moe_ones,    (size_t)E * sizeof(float));
    MOE_A(eng->d_moe_xe,      (size_t)H * A * sizeof(float));
    MOE_A(eng->d_moe_gate,    (size_t)EFFN * A * sizeof(float));
    MOE_A(eng->d_moe_up,      (size_t)EFFN * A * sizeof(float));
    MOE_A(eng->d_moe_act,     (size_t)EFFN * A * sizeof(float));
    MOE_A(eng->d_moe_oe,      (size_t)H * A * sizeof(float));
    MOE_A(eng->d_moe_q8,      (size_t)H * A * sizeof(int8_t));
    MOE_A(eng->d_moe_q8d,     (size_t)(H / 32) * A * sizeof(float));
    MOE_A(eng->d_moe_q8s,     (size_t)(H / 32) * A * sizeof(int));
    MOE_A(eng->d_moe_shlog,   (size_t)T * sizeof(float));
    MOE_A(eng->d_moe_active,  (size_t)A * sizeof(int));
    if (eng->moe_experts_fp4) {
        // CUTLASS grouped NVFP4 per-step scratch. SF buffers sized for the widest per-call
        // stride (cn = T): pad(T,128) M-rows × pad(K/16,4) K-vecs per expert. Zeroed ONCE —
        // per-step writes cover every row < M_g, and rows ≥ M_g are only read for the padded
        // tail of each expert's M-tile, whose outputs the epilogue never stores.
        const size_t sf1 = (size_t)E * nvfp4_pad(T, 128) * nvfp4_pad(H / NVFP4_BLK, 4);
        const size_t sf2 = (size_t)E * nvfp4_pad(T, 128) * nvfp4_pad(EFFN / NVFP4_BLK, 4);
        MOE_A(eng->d_fp4m_a,      (size_t)A * (H / 2));
        MOE_A(eng->d_fp4m_asf,    sf1);
        MOE_A(eng->d_fp4m_a2,     (size_t)A * (EFFN / 2));
        MOE_A(eng->d_fp4m_a2sf,   sf2);
        MOE_A(eng->d_fp4m_gu_out, (size_t)A * 2 * EFFN * sizeof(__nv_bfloat16));
        MOE_A(eng->d_fp4m_dn_out, (size_t)A * H * sizeof(__nv_bfloat16));
        MOE_A(eng->d_fp4m_indptr, (size_t)(E + 1) * sizeof(int));
        MOE_A(eng->d_fp4m_t2e,    (size_t)A * sizeof(int));
        MOE_A(eng->d_fp4m_gsrow,  (size_t)A * sizeof(float));
        if (ok) { cudaMemset(eng->d_fp4m_asf, 0, sf1); cudaMemset(eng->d_fp4m_a2sf, 0, sf2); }
    }
    #undef MOE_A
    if (!ok) { fprintf(stderr, "fucina: MoE scratch alloc failed\n"); return -1; }
    float *ones = (float*)malloc((size_t)E * sizeof(float));
    if (!ones) return -1;
    for (int i = 0; i < E; i++) ones[i] = 1.0f;
    cudaMemcpy(eng->d_moe_ones, ones, (size_t)E * sizeof(float), cudaMemcpyHostToDevice);
    free(ones);
    eng->moe_scratch_bytes = allocated_bytes;
    eng->moe_scratch_ready = 1;
    return 0;
}

// Element-parallel natural-layout Q4_K → BF16 (the superblock-serial dequant_q4_k_to_bf16
// kernel runs ONE thread per 256-elem superblock — measured 24 ms per 268M-elem expert slab,
// 65% of a Q4K-mode MoE prefill). One thread per element; the 16-B headers are L2-hot.
__global__ void dequant_q4_k_slab_bf16_fast_kernel(
    __nv_bfloat16 *__restrict__ dst, const uint8_t *__restrict__ src, uint64_t n)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t sb = i >> 8;
    int r = (int)(i & 255), j = r >> 5, k = r & 31;
    const uint8_t *blk = src + sb * 144;
    __half_raw hd; hd.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
    __half_raw hm; hm.x = (uint16_t)(blk[2] | ((uint16_t)blk[3] << 8));
    float d = __half2float(__half(hd)), dmin = __half2float(__half(hm));
    int sV, mV; q4k_scale_min(blk + 4, j, &sV, &mV);
    const uint8_t *qbase = blk + 16 + (size_t)(j >> 1) * 32;
    int shift = (j & 1) ? 4 : 0;
    int q = (qbase[k] >> shift) & 0xF;
    dst[i] = __float2bfloat16(d * (float)sV * (float)q - dmin * (float)mV);
}

// Dequant a CONTIGUOUS FP8 expert slab (E consecutive [out×in] E4M3 experts, per-expert BF16
// block-scale slabs) → BF16. Feeds the tensor-core expert prefill GEMMs: the scalar-float
// grouped FP8 kernel measured 71.6% of a 2k-token MoE prefill (ALU-bound at 16 tok/expert).
__global__ void dequant_fp8_expert_slab_bf16_kernel(
    __nv_bfloat16 *dst, const uint8_t *w, const __nv_bfloat16 *sc,
    int in_dim, int out_dim, uint64_t n)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t per = (uint64_t)out_dim * in_dim;
    int e = (int)(i / per);
    uint64_t r = i - (uint64_t)e * per;
    int row = (int)(r / in_dim), col = (int)(r % in_dim);
    __nv_fp8_e4m3 v; v.__x = w[i];
    int sper = ((out_dim + 127) >> 7) * (in_dim >> 7);
    float bs = __bfloat162float(sc[(size_t)e * sper + (size_t)(row >> 7) * (in_dim >> 7) + (col >> 7)]);
    dst[i] = __float2bfloat16(float(v) * bs);
}

// Dequant a per-layer NVFP4 expert slab (packed E2M1 + per-expert 32×4×4-swizzled E4M3 SF +
// per-(layer,proj) global scale) → the BF16 slab the tensor-core expert prefill GEMMs read.
// row_off selects the gate (0) / up (MI) half of the fused gate|up slab; slab_rows is the
// per-expert row count of the SOURCE slab (2·MI for gate|up, H for down). Reconstruction is
// exactly the values the CUTLASS grouped GEMM consumes: e2m1 · e4m3(SF) · gs.
__global__ void dequant_nvfp4_expert_slab_bf16_kernel(
    __nv_bfloat16 *dst, const uint8_t *fp4, const uint8_t *sf, const float *gsp,
    int in_dim, int out_dim, int row_off, int slab_rows, unsigned long long sfB, uint64_t n)
{
    // one thread per 16-element SF block: swizzle math + SF decode once, 8 quant bytes in,
    // 16 bf16 out via two uint4 stores (n is the ELEMENT count; always a multiple of 16)
    uint64_t blk = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nblk_total = n / NVFP4_BLK;
    if (blk >= nblk_total) return;
    int nblk = in_dim / NVFP4_BLK, nbp = ((nblk + 3) / 4) * 4;
    uint64_t per = (uint64_t)out_dim * nblk;
    int e = (int)(blk / per);
    uint64_t r = blk - (uint64_t)e * per;
    int row = (int)(r / nblk), s = (int)(r % nblk);
    int srow = row_off + row;
    int oo = srow >> 7, oi = srow & 127, so = s >> 2, si = s & 3;
    size_t off = ((size_t)oo * (size_t)(nbp / 4) + so) * 512 + (size_t)(oi % 32) * 16 + (oi / 32) * 4 + si;
    float bsf = __half2float(__half(__nv_cvt_fp8_to_halfraw(
        (__nv_fp8_storage_t)sf[(size_t)e * sfB + off], __NV_E4M3)));
    float scale = bsf * (*gsp);
    const uint8_t *q = fp4 + (size_t)e * slab_rows * (in_dim / 2)
                     + (size_t)srow * (in_dim / 2) + (size_t)s * (NVFP4_BLK / 2);
    __nv_bfloat16 o[NVFP4_BLK];
    #pragma unroll
    for (int i = 0; i < NVFP4_BLK / 2; i++) {
        uint32_t byte = q[i];
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            uint32_t nib = h ? (byte >> 4) : (byte & 0x0F);
            // E2M1 magnitude table {0,.5,1,1.5,2,3,4,6} + sign bit — the dense NVFP4 GEMV's decode
            float tab = (nib & 4u) ? ((nib & 2u) ? ((nib & 1u) ? 6.f : 4.f) : ((nib & 1u) ? 3.f : 2.f))
                                   : ((nib & 2u) ? ((nib & 1u) ? 1.5f : 1.f) : ((nib & 1u) ? 0.5f : 0.f));
            o[2 * i + h] = __float2bfloat16(((nib & 8u) ? -tab : tab) * scale);
        }
    }
    uint4 *out16 = (uint4 *)(dst + ((uint64_t)e * per + r) * NVFP4_BLK);
    out16[0] = *(const uint4 *)&o[0];
    out16[1] = *(const uint4 *)&o[8];
}

// Dequant a CONTIGUOUS native-NVFP4 expert slab in CHECKPOINT layout (E consecutive experts,
// each `weight` U8 [out][in/2] packed E2M1 + `weight_scale` E4M3 [out][in/16] LINEAR + a
// per-expert `weight_scale_2` F32 global) → BF16. Feeds the same requant pipeline as the FP8
// source (q35_fp4_expert_slabs): both w and sc slabs are row-major-contiguous, so a 16-element
// group's quant bytes sit at blk*8 and its scale at sc[blk] — only gs needs the expert index.
__global__ void dequant_nvfp4ckpt_expert_slab_bf16_kernel(
    __nv_bfloat16 *dst, const uint8_t *w, const uint8_t *sc, const float *gs,
    int in_dim, int out_dim, uint64_t n)   // n = E·out·in elements (multiple of 16)
{
    uint64_t blk = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nblk_total = n / NVFP4_BLK;
    if (blk >= nblk_total) return;
    uint64_t per = (uint64_t)out_dim * (in_dim / NVFP4_BLK);
    int e = (int)(blk / per);
    float bsf = __half2float(__half(__nv_cvt_fp8_to_halfraw(
        (__nv_fp8_storage_t)sc[blk], __NV_E4M3)));
    float scale = bsf * gs[e];
    const uint8_t *q = w + blk * (NVFP4_BLK / 2);
    __nv_bfloat16 o[NVFP4_BLK];
    #pragma unroll
    for (int i = 0; i < NVFP4_BLK / 2; i++) {
        uint32_t byte = q[i];
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            uint32_t nib = h ? (byte >> 4) : (byte & 0x0F);
            float tab = (nib & 4u) ? ((nib & 2u) ? ((nib & 1u) ? 6.f : 4.f) : ((nib & 1u) ? 3.f : 2.f))
                                   : ((nib & 2u) ? ((nib & 1u) ? 1.5f : 1.f) : ((nib & 1u) ? 0.5f : 0.f));
            o[2 * i + h] = __float2bfloat16(((nib & 8u) ? -tab : tab) * scale);
        }
    }
    uint4 *out16 = (uint4 *)(dst + blk * NVFP4_BLK);
    out16[0] = *(const uint4 *)&o[0];
    out16[1] = *(const uint4 *)&o[8];
}

static inline uint64_t q35_expert_hash_update(uint64_t h,const void *p,size_t n){
    const uint8_t*b=(const uint8_t*)p;for(size_t i=0;i<n;i++){h^=b[i];h*=1099511628211ULL;}return h;
}

// out[t][i] += sigmoid(shlog[t]) · v[t][i] — the Qwen3.5-MoE shared-expert add (per-token
// sigmoid-gated) into the MoE block output. One thread per element, token = blockIdx.y.
__global__ void q35moe_b_shared_axpy_kernel(float *out, const float *v, const float *shlog, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (i >= H) return;
    float sg = 1.f / (1.f + expf(-shlog[t]));
    out[(size_t)t * H + i] += sg * v[(size_t)t * H + i];
}

static void m2_gemm(float *out, const float *x, const float *w, int T, int in_dim, int out_dim,
                    cudaStream_t stream);   // defined with the M2 kernels below (default stream=0 there)

// Sparse-MoE FFN (Qwen3-MoE GGUF/Q4_K and Qwen3.5-MoE safetensors/FP8-block).
// h_f32 [H·n] = ffn_norm(x) (token-major FP32). Writes out_f32 [H·n] (fully written inside)
// = Σ_k router_w_k · down_k(silu(gate_k(h))·up_k(h))  [+ sigmoid-gated shared expert on FP8].
// Processes ≤ GEMMA4_MOE_TMAX tokens per chunk (each token routes independently → chunking is
// bit-identical). Caller adds out_f32 to the residual. All-device (CUDA-graph-capturable).
static void moe_ffn(gemma4_engine_t *eng, int l, const float *h_f32, float *out_f32, int n,
                    cudaStream_t stream) {
    // P1 unification (see q35_proj_gemm): force the WIDE tensor-core expert path even for short
    // token counts so a short prompt's experts compute identically standalone (n<=32) and inside a
    // batched multi-seq prefill (n>32) — otherwise the scalar-grouped vs tensor-core-grouped split
    // breaks batched==standalone. FUCINA_UNIFY_BF16 lowers the n>2*FP8_MAXB thresholds to always.
    static int moe_unify = -1;
    if (moe_unify < 0) moe_unify = (getenv("FUCINA_NO_UNIFY") != NULL) ? 0 : 1;   // default ON (deterministic keeper)
    const int H = eng->cfg.hidden_size, EFFN = eng->cfg.expert_ffn;
    const int E = eng->cfg.n_experts, U = eng->cfg.n_experts_used;
    const int GU = 2 * EFFN;   // fused gate+up slab rows (SSD streaming pool geometry)
    const ExpertWeightRef &fp4_gu=eng->ref_fp4m_gu[l], &fp4_dn=eng->ref_fp4m_dn[l];
    const ExpertWeightRef &expert_gate=eng->ref_moe_gate[l], &expert_up=eng->ref_moe_up[l],
                          &expert_down=eng->ref_moe_down[l];
    // SSD streaming drops the full slabs at load; those GEMMs read the slot pool instead of the
    // per-layer descriptors (which then hold no resident data for routed experts).
    const uint8_t *stage_gu=eng->d_fp4m_stage_gu, *stage_gusf=eng->d_fp4m_stage_gusf;
    const uint8_t *stage_dn=eng->d_fp4m_stage_dn, *stage_dnsf=eng->d_fp4m_stage_dnsf;
    // Diagnostic phase timing is deliberately synchronizing and opt-in only. Reuse two events
    // as a moving boundary; each mark drains the stream, attributes the completed interval, then
    // swaps start/stop. Normal serving creates no events and takes no timing branches past `timing`.
    const bool timing = eng->q35.prefill_timing && n > 2 * FP8_MAXB;
    cudaEvent_t tev0=nullptr, tev1=nullptr;
    if (timing) { cudaEventCreate(&tev0); cudaEventCreate(&tev1); cudaEventRecord(tev0,stream); }
    auto timing_mark = [&](double *dst) {
        if (!timing) return;
        cudaEventRecord(tev1,stream); cudaEventSynchronize(tev1);
        float ms=0; cudaEventElapsedTime(&ms,tev0,tev1); *dst += ms;
        cudaEvent_t tmp=tev0; tev0=tev1; tev1=tmp;
    };
    // Q4_K-requantized experts route through the GGUF Q4_K grouped branch below (and skip the
    // FP8 tc-prefill dequant, which reads FP8 slabs). Router/shared expert stay on their paths.
    const int fp8 = (eng->format == FORMAT_FP8_BLOCK) && !eng->moe_experts_q4k && !eng->moe_experts_fp4;
    // tc prefill works for FP8, Q4_K AND NVFP4 slabs (each has a slab→BF16 dequant); wide calls
    // ride the grouped-batched cuBLAS BF16 GEMMs (measured faster than the CUTLASS NVFP4 grouped
    // GEMM at prefill widths: 2k TTFT 1.16 s vs 1.23 s), decode-sized calls keep CUTLASS NVFP4.
    const int tc_ok = (eng->format == FORMAT_FP8_BLOCK) && !eng->fp4m_ssd_stream;
    const float *router_w = (const float*)weight_fp8(eng, eng->moe_router[l]);
    const void  *gate_w   = expert_gate.weight.data;
    const void  *up_w     = expert_up.weight.data;
    // FP8: every expert proj is EFFN×H (or H×EFFN) E4M3 = the same bytes/expert; the BF16 block
    // scales sit in parallel per-expert slabs of (EFFN/128)·(H/128) elements (ceil for EFFN<128).
    const int64_t fp8_wslab = expert_gate.weight_stride;
    const int64_t fp8_sslab = expert_gate.scale_stride / (int64_t)sizeof(__nv_bfloat16);
    // ── Tensor-core expert PREFILL (fp8, wide call): dequant the 3 expert slabs → BF16 once per
    // LAYER, then per-group cublas GEMMs replace the scalar-float grouped kernel (measured 71.6%
    // of a 2k-token prefill, ALU-bound at ~16 tok/expert). Decode-sized calls keep the FP8 kernel.
    int tc = 0;
    // Under unify, SUPPRESS the tensor-core bf16 grouped path (cublas per-expert GEMMs pick an
    // atomic split-K algo at batched token counts → NONDETERMINISTIC run-to-run) and keep the
    // custom FP8-native fp8_block_grouped path (deterministic, FP8-precision) for ALL n, so short
    // and batched experts compute identically. See determinism gate.
    if (tc_ok && !moe_unify && n > 2 * FP8_MAXB && E <= 256 && !eng->moe_tc_off) {
        if (!eng->d_moe_xbf) {
            int xmax = (H > EFFN) ? H : EFFN;
            size_t wb = (size_t)E * EFFN * H * sizeof(__nv_bfloat16);
            int ok = cudaMalloc(&eng->d_moe_xbf, (size_t)GEMMA4_MOE_TMAX * U * xmax * sizeof(__nv_bfloat16)) == cudaSuccess;
            for (int i = 0; i < 3 && ok; i++) ok = cudaMalloc(&eng->d_moe_wbf[i], wb) == cudaSuccess;
            if (!ok) {
                cudaGetLastError(); eng->moe_tc_off = 1;
                if (eng->d_moe_xbf) { cudaFree(eng->d_moe_xbf); eng->d_moe_xbf = NULL; }
                for (int i = 0; i < 3; i++)
                    if (eng->d_moe_wbf[i]) { cudaFree(eng->d_moe_wbf[i]); eng->d_moe_wbf[i] = NULL; }
                fprintf(stderr, "fucina: MoE tensor-core prefill scratch alloc failed — scalar grouped kernels\n");
            }
        }
        if (!eng->moe_tc_off) {
            const uint64_t ne = (uint64_t)E * EFFN * H;
            const uint8_t *ws[3] = { (const uint8_t*)gate_w, (const uint8_t*)up_w,
                                     expert_down.weight.data };
            const void *scales[3] = { expert_gate.weight.scale, expert_up.weight.scale,
                                      expert_down.weight.scale };
            for (int p = 0; p < 3; p++) {
                if (eng->moe_experts_fp4)   // NVFP4 slabs: gate/up halves of the fused gu slab, then dn
                    dequant_nvfp4_expert_slab_bf16_kernel<<<(unsigned)((ne / NVFP4_BLK + 255) / 256), 256, 0, stream>>>(
                        eng->d_moe_wbf[p],
                        (const uint8_t*)((p == 2) ? fp4_dn.weight.data : fp4_gu.weight.data),
                        (const uint8_t*)((p == 2) ? fp4_dn.weight.scale : fp4_gu.weight.scale),
                        (const float*)((p == 2) ? fp4_dn.weight.global_scale : fp4_gu.weight.global_scale),
                        (p == 2) ? EFFN : H, (p == 2) ? H : EFFN,
                        (p == 1) ? EFFN : 0, (p == 2) ? H : 2 * EFFN,
                        (p == 2) ? fp4_dn.scale_stride : fp4_gu.scale_stride, ne);
                else if (eng->moe_experts_q4k)   // Q4_K slabs: element-parallel natural-layout dequant
                    dequant_q4_k_slab_bf16_fast_kernel<<<(unsigned)((ne + 255) / 256), 256, 0, stream>>>(
                        eng->d_moe_wbf[p], ws[p], ne);
                else
                    dequant_fp8_expert_slab_bf16_kernel<<<(unsigned)((ne + 255) / 256), 256, 0, stream>>>(
                        eng->d_moe_wbf[p], ws[p], (const __nv_bfloat16*)scales[p],
                        (p == 2) ? EFFN : H, (p == 2) ? H : EFFN, ne);
            }
            tc = 1;
        }
    }
    timing_mark(&eng->q35.prefill_dequant_ms);
    for (int t0 = 0; t0 < n; t0 += GEMMA4_MOE_TMAX) {
        int cn = n - t0; if (cn > GEMMA4_MOE_TMAX) cn = GEMMA4_MOE_TMAX;
        const float *h = h_f32 + (size_t)t0 * H;
        float *out = out_f32 + (size_t)t0 * H;
        int total = cn * U;
        // Router → softmax-E → top-U → renorm-to-sum-1 → counting-sort route (pes = ones[E]).
        // Wide (prefill) calls ride a cublas SGEMM — the block-per-(expert,token) GEMV re-walks
        // the router per token and measured 314 ms per 2.9k-token pass (11.6% of prefill).
        if (moe_unify || cn > 2 * FP8_MAXB) {
            const float alf = 1.0f, bet = 0.0f;
            cublasSgemm(eng->cublas, CUBLAS_OP_T, CUBLAS_OP_N, E, cn, H,
                        &alf, router_w, H, h, H, &bet, eng->d_moe_rlogits, E);
        } else
        moe_router_gemv_kernel<<<dim3(E, (unsigned)((cn + 1) / 2)), 256, 0, stream>>>(
            router_w, h, eng->d_moe_rlogits, H, E, cn);
        dg_softmax_topk(eng->d_moe_rlogits, E, cn, U, eng->d_moe_tki, eng->d_moe_tkw, stream);
        if (eng->d_moe_profile_count) {
            moe_profile_accum_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
                eng->d_moe_tki, eng->d_moe_tkw,
                eng->d_moe_profile_count + (size_t)l * E,
                eng->d_moe_profile_weight + (size_t)l * E, total);
        }
        int n_slot = (total < E) ? total : E;   // grid.y for active-expert grouped GEMMs
        dg_moe_route_inv(eng->d_moe_tki, eng->d_moe_tkw, eng->d_moe_ones, cn, U, E,
                         eng->d_moe_count, eng->d_moe_coloff, eng->d_moe_cursor,
                         eng->d_moe_eidx, eng->d_moe_ecs, eng->d_moe_invpos,
                         (fp8 || eng->moe_experts_q4k) ? eng->d_moe_active : NULL, n_slot, stream);
        // Gather expert inputs → grouped gate/up/down. FP8 keeps FLOAT activations (no Q8_1).
        dg_gather_cols(eng->d_moe_xe, h, eng->d_moe_eidx, H, total, stream);
        timing_mark(&eng->q35.prefill_router_ms);
        if (!tc && eng->moe_experts_fp4) {
            // CUTLASS grouped block-scaled NVFP4 experts: quantize the gathered activations per
            // expert group to E2M1 + swizzled ue4m3 SF, then ONE ptr-array tensor-core GEMM per
            // projection covers every expert (fused gate|up, then down). Serves decode AND
            // prefill — 2.1× the dp4a grouped GEMV weight bandwidth at 1 tok/expert, 11+ TFLOP/s
            // at prefill widths (test_dg_fp4_grouped on GB10).
            auto g1 = [](size_t n){ return (unsigned)((n + 255) / 256); };
            const int nbH = H / NVFP4_BLK,    nbHp = nvfp4_pad(nbH, 4);
            const int nbF = EFFN / NVFP4_BLK, nbFp = nvfp4_pad(nbF, 4);
            const unsigned long long sfA1 = (unsigned long long)nvfp4_pad(cn, 128) * nbHp;
            const unsigned long long sfA2 = (unsigned long long)nvfp4_pad(cn, 128) * nbFp;
            q35fp4_grp_idx_kernel<<<(E + 255) / 256, 256, 0, stream>>>(
                eng->d_moe_coloff, eng->d_moe_count, E, total, eng->d_fp4m_indptr, eng->d_fp4m_t2e);
            // PER-ROW activation scales (row-independence gate); GEMM alpha carries the weight
            // global scale only, the row scale is applied in the silu / output-scale kernels.
            q35fp4_row_gs_kernel<<<total, 256, 0, stream>>>(eng->d_moe_xe, H, total, eng->d_fp4m_gsrow);
            { dim3 b(256), g((nbH + 255) / 256, total);
              q35fp4_quant_grp_kernel<<<g, b, 0, stream>>>(eng->d_moe_xe, eng->d_fp4m_t2e,
                  eng->d_fp4m_indptr, H, total, eng->d_fp4m_gsrow, eng->d_fp4m_a, eng->d_fp4m_asf,
                  sfA1, nbHp); }
            int rc=0;
            std::vector<int> active_stream;
            if(eng->fp4m_ssd_stream) {   // never graph-captured (host I/O) — sync is fine
                std::vector<int> hc_stream(E);
                cudaMemcpyAsync(hc_stream.data(),eng->d_moe_count,E*sizeof(int),cudaMemcpyDeviceToHost,stream);
                cudaStreamSynchronize(stream);
                for(int e=0;e<E;e++)if(hc_stream[e]>0)active_stream.push_back(e);
                if(eng->ssd_expert_profile)
                    fucina_expert_profile_record(eng->ssd_expert_profile,l,hc_stream.data(),E);
                const char *pf=getenv("FUCINA_EXPERT_PREFETCH");
                if(!pf||pf[0]!='0') { // asynchronous page-cache lookahead for the next layer
                    int nl=(l+1)%eng->cfg.n_layers;const size_t gp=(size_t)GU*(H/2),gsp=(size_t)eng->fp4m_gu_sfB;
                    const size_t dp=(size_t)H*(EFFN/2),dsp=(size_t)eng->fp4m_dn_sfB;
                    for(int e:active_stream){posix_fadvise(eng->fp4m_ssd_fd,eng->fp4m_ssd_gu_off[nl]+(int64_t)e*gp,(off_t)gp,POSIX_FADV_WILLNEED);
                        posix_fadvise(eng->fp4m_ssd_fd,eng->fp4m_ssd_gusf_off[nl]+(int64_t)e*gsp,(off_t)gsp,POSIX_FADV_WILLNEED);
                        posix_fadvise(eng->fp4m_ssd_fd,eng->fp4m_ssd_dn_off[nl]+(int64_t)e*dp,(off_t)dp,POSIX_FADV_WILLNEED);
                        posix_fadvise(eng->fp4m_ssd_fd,eng->fp4m_ssd_dnsf_off[nl]+(int64_t)e*dsp,(off_t)dsp,POSIX_FADV_WILLNEED);eng->fp4m_prefetch_advice++;}
                }
            }
            std::vector<int> cache_map(E,-1);
            bool cache_ready=false;
            auto ssd_cache_prepare=[&]()->int {
                if(!eng->fp4m_ssd_stream || active_stream.size()>(size_t)eng->fp4m_slots)return -1;
                const size_t gp=(size_t)GU*(H/2),gsp=(size_t)eng->fp4m_gu_sfB;
                const size_t dp=(size_t)H*(EFFN/2),dsp=(size_t)eng->fp4m_dn_sfB;
                std::vector<unsigned char> used((size_t)eng->fp4m_slots,0);
                std::vector<int> missing;
                for(int e:active_stream){int found=-1;for(int s=0;s<eng->fp4m_slots;s++)
                    if(eng->h_fp4m_slot_layer[s]==l&&eng->h_fp4m_slot_expert[s]==e){found=s;break;}
                    if(found>=0){cache_map[e]=found;used[found]=1;eng->h_fp4m_slot_age[found]=++eng->fp4m_slot_clock;eng->fp4m_cache_hits++;}
                    else missing.push_back(e);}
                if(!missing.empty())cudaStreamSynchronize(stream);
                auto rd=[&](void*p,size_t n,int64_t at){uint8_t*b=(uint8_t*)p;size_t d=0;while(d<n){ssize_t r=pread(eng->fp4m_ssd_fd,b+d,n-d,at+(int64_t)d);if(r<=0)return false;d+=(size_t)r;}return true;};
                for(int e:missing){int slot=-1;uint64_t oldest=~0ULL;
                    for(int s=0;s<eng->fp4m_slots;s++)if(!used[s]&&eng->h_fp4m_slot_layer[s]<0){slot=s;break;}
                    if(slot<0)for(int s=0;s<eng->fp4m_slots;s++)if(!used[s]&&eng->h_fp4m_slot_age[s]<oldest){oldest=eng->h_fp4m_slot_age[s];slot=s;}
                    if(slot<0)return -11;used[slot]=1;cache_map[e]=slot;
                    uint8_t*hg=eng->h_fp4m_stage_gu+(size_t)slot*gp,*hs=eng->h_fp4m_stage_gusf+(size_t)slot*gsp;
                    uint8_t*hd=eng->h_fp4m_stage_dn+(size_t)slot*dp,*hds=eng->h_fp4m_stage_dnsf+(size_t)slot*dsp;
                    if(!rd(hg,gp,eng->fp4m_ssd_gu_off[l]+(int64_t)e*gp)||!rd(hs,gsp,eng->fp4m_ssd_gusf_off[l]+(int64_t)e*gsp)
                       ||!rd(hd,dp,eng->fp4m_ssd_dn_off[l]+(int64_t)e*dp)||!rd(hds,dsp,eng->fp4m_ssd_dnsf_off[l]+(int64_t)e*dsp))return -9;
                    eng->fp4m_ssd_reads+=4;eng->fp4m_ssd_bytes+=gp+gsp+dp+dsp;
                    const void*pp[4]={hg,hs,hd,hds};const size_t nn[4]={gp,gsp,dp,dsp};size_t z=((size_t)l*E+e)*4;
                    for(int p=0;p<4;p++)if(!eng->h_fp4m_ssd_verified[z+p]){uint64_t hh=q35_expert_hash_update(1469598103934665603ULL,pp[p],nn[p]);
                        if(hh!=eng->h_fp4m_ssd_hash[z+p]){eng->fp4m_ssd_checksum_fail++;return -10;}eng->h_fp4m_ssd_verified[z+p]=1;}
                    cudaMemcpyAsync(eng->d_fp4m_stage_gu+(size_t)slot*gp,hg,gp,cudaMemcpyHostToDevice,stream);
                    cudaMemcpyAsync(eng->d_fp4m_stage_gusf+(size_t)slot*gsp,hs,gsp,cudaMemcpyHostToDevice,stream);
                    cudaMemcpyAsync(eng->d_fp4m_stage_dn+(size_t)slot*dp,hd,dp,cudaMemcpyHostToDevice,stream);
                    cudaMemcpyAsync(eng->d_fp4m_stage_dnsf+(size_t)slot*dsp,hds,dsp,cudaMemcpyHostToDevice,stream);
                    eng->h_fp4m_slot_layer[slot]=l;eng->h_fp4m_slot_expert[slot]=e;eng->h_fp4m_slot_age[slot]=++eng->fp4m_slot_clock;eng->fp4m_cache_misses++;
                }
                cudaMemcpyAsync(eng->d_fp4m_eslot,cache_map.data(),E*sizeof(int),cudaMemcpyHostToDevice,stream);
                cache_ready=true;return 0;
            };
            auto ssd_gemm=[&](bool down)->int {
                const int slots=eng->fp4m_slots;
                const size_t wp=down?(size_t)H*(EFFN/2):(size_t)GU*(H/2);
                const size_t sp=down?(size_t)eng->fp4m_dn_sfB:(size_t)eng->fp4m_gu_sfB;
                uint8_t *hw=down?eng->h_fp4m_stage_dn:eng->h_fp4m_stage_gu;
                uint8_t *hs=down?eng->h_fp4m_stage_dnsf:eng->h_fp4m_stage_gusf;
                uint8_t *dw=down?eng->d_fp4m_stage_dn:eng->d_fp4m_stage_gu;
                uint8_t *ds=down?eng->d_fp4m_stage_dnsf:eng->d_fp4m_stage_gusf;
                int64_t wo=down?eng->fp4m_ssd_dn_off[l]:eng->fp4m_ssd_gu_off[l];
                int64_t so=down?eng->fp4m_ssd_dnsf_off[l]:eng->fp4m_ssd_gusf_off[l];
                auto rd=[&](void*p,size_t n,int64_t at){uint8_t*b=(uint8_t*)p;size_t d=0;while(d<n){ssize_t r=pread(eng->fp4m_ssd_fd,b+d,n-d,at+(int64_t)d);if(r<=0)return false;d+=(size_t)r;}return true;};
                int out_rc=0;
                for(size_t a0=0;a0<active_stream.size();a0+=slots) {
                    int ns=(int)std::min((size_t)slots,active_stream.size()-a0);
                    std::vector<int> map(E,-1);
                    for(int s=0;s<ns;s++){int e=active_stream[a0+s];map[e]=s;
                        if(!rd(hw+(size_t)s*wp,wp,wo+(int64_t)e*wp)||!rd(hs+(size_t)s*sp,sp,so+(int64_t)e*sp))return -9;
                        eng->fp4m_ssd_reads+=2;eng->fp4m_ssd_bytes+=wp+sp;
                        size_t z=((size_t)l*E+e)*4+(down?2:0);
                        if(!eng->h_fp4m_ssd_verified[z]){
                            uint64_t wh=q35_expert_hash_update(1469598103934665603ULL,hw+(size_t)s*wp,wp);
                            if(wh!=eng->h_fp4m_ssd_hash[z]){eng->fp4m_ssd_checksum_fail++;return -10;}
                            eng->h_fp4m_ssd_verified[z]=1;
                        }
                        if(!eng->h_fp4m_ssd_verified[z+1]){
                            uint64_t sh=q35_expert_hash_update(1469598103934665603ULL,hs+(size_t)s*sp,sp);
                            if(sh!=eng->h_fp4m_ssd_hash[z+1]){eng->fp4m_ssd_checksum_fail++;return -10;}
                            eng->h_fp4m_ssd_verified[z+1]=1;
                        }
                    }
                    cudaMemcpyAsync(dw,hw,(size_t)ns*wp,cudaMemcpyHostToDevice,stream);
                    cudaMemcpyAsync(ds,hs,(size_t)ns*sp,cudaMemcpyHostToDevice,stream);
                    cudaMemcpyAsync(eng->d_fp4m_eslot,map.data(),E*sizeof(int),cudaMemcpyHostToDevice,stream);
                    out_rc|=dg_fp4_moe_grouped_mapped(
                        down?(void*)eng->d_fp4m_dn_out:(void*)eng->d_fp4m_gu_out,
                        down?(void*)eng->d_fp4m_a2:(void*)eng->d_fp4m_a,
                        down?(void*)eng->d_fp4m_a2sf:(void*)eng->d_fp4m_asf,dw,ds,
                        eng->d_moe_count,eng->d_moe_coloff,eng->d_fp4m_eslot,E,
                        down?H:GU,down?EFFN:H,down?sfA2:sfA1,sp,
                        eng->d_fp4m_gsw+2*l+(down?1:0),stream);
                    cudaStreamSynchronize(stream);
                }
                for(int s=0;s<eng->fp4m_slots;s++){eng->h_fp4m_slot_layer[s]=-1;eng->h_fp4m_slot_expert[s]=-1;}
                return out_rc;
            };
            if(eng->fp4m_ssd_stream) {
                rc=ssd_cache_prepare();
                if(rc==-1)rc=ssd_gemm(false);
                else if(!rc)rc=dg_fp4_moe_grouped_mapped(eng->d_fp4m_gu_out,eng->d_fp4m_a,eng->d_fp4m_asf,
                    stage_gu,stage_gusf,eng->d_moe_count,eng->d_moe_coloff,eng->d_fp4m_eslot,E,GU,H,
                    sfA1,eng->fp4m_gu_sfB,eng->d_fp4m_gsw+2*l,stream);
            } else rc = dg_fp4_moe_grouped(eng->d_fp4m_gu_out, eng->d_fp4m_a, eng->d_fp4m_asf,
                fp4_gu.weight.data, (const uint8_t*)fp4_gu.weight.scale, eng->d_fp4m_indptr,
                fp4_gu.expert_count, fp4_gu.weight.out_dim, fp4_gu.weight.in_dim, sfA1, fp4_gu.scale_stride,
                fp4_gu.weight.global_scale, stream);
            q35fp4_gu_silu_mul_kernel<<<g1((size_t)total * EFFN), 256, 0, stream>>>(
                eng->d_moe_act, eng->d_fp4m_gu_out, eng->d_fp4m_gsrow, EFFN, (int64_t)total * EFFN);
            moe_profile_activation(eng, l, 3, eng->d_moe_act, (size_t)total*EFFN, stream);
            q35fp4_row_gs_kernel<<<total, 256, 0, stream>>>(eng->d_moe_act, EFFN, total, eng->d_fp4m_gsrow);
            { dim3 b(256), g((nbF + 255) / 256, total);
              q35fp4_quant_grp_kernel<<<g, b, 0, stream>>>(eng->d_moe_act, eng->d_fp4m_t2e,
                  eng->d_fp4m_indptr, EFFN, total, eng->d_fp4m_gsrow, eng->d_fp4m_a2, eng->d_fp4m_a2sf,
                  sfA2, nbFp); }
            if(eng->fp4m_ssd_stream) {
                if(cache_ready)rc|=dg_fp4_moe_grouped_mapped(eng->d_fp4m_dn_out,eng->d_fp4m_a2,eng->d_fp4m_a2sf,
                    stage_dn,stage_dnsf,eng->d_moe_count,eng->d_moe_coloff,eng->d_fp4m_eslot,E,H,EFFN,
                    sfA2,eng->fp4m_dn_sfB,eng->d_fp4m_gsw+2*l+1,stream);
                else rc|=ssd_gemm(true);
            } else rc |= dg_fp4_moe_grouped(eng->d_fp4m_dn_out, eng->d_fp4m_a2, eng->d_fp4m_a2sf,
                fp4_dn.weight.data, (const uint8_t*)fp4_dn.weight.scale, eng->d_fp4m_indptr,
                fp4_dn.expert_count, fp4_dn.weight.out_dim, fp4_dn.weight.in_dim, sfA2, fp4_dn.scale_stride,
                fp4_dn.weight.global_scale, stream);
            q35fp4_dn_scale_kernel<<<g1((size_t)total * H), 256, 0, stream>>>(
                eng->d_moe_oe, eng->d_fp4m_dn_out, eng->d_fp4m_gsrow, H, (uint64_t)total * H);
            if(rc==-10){fprintf(stderr,"fucina: expert-store checksum failure at layer %d\n",l);abort();}
            if (rc) {   // never expected after a successful load-time build + warmup
                static int warned = 0;
                if (!warned) { warned = 1; fprintf(stderr, "fucina: dg_fp4_moe_grouped rc=%d — MoE output invalid\n", rc); }
            }
        } else if (tc) {
            // Tensor-core grouped expert FFN: ragged per-expert cublas GEMMs over the BF16 slabs.
            // count/coloff come to the host (prefill path — never graph-captured, sync is fine).
            int hc[256], ho[256];
            cudaMemcpyAsync(hc, eng->d_moe_count,  E * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(ho, eng->d_moe_coloff, E * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            f32_to_bf16_kernel<<<(unsigned)(((size_t)total * H + 255) / 256), 256, 0, stream>>>(
                eng->d_moe_xbf, eng->d_moe_xe, (uint64_t)total * H);
            // gate/up: one grouped-batched GEMM each over the active experts (in=H, out=EFFN).
            gemm_bf16_grouped(eng, eng->d_moe_wbf[0], (size_t)EFFN * H, eng->d_moe_xbf,
                              eng->d_moe_gate, hc, ho, E, H, EFFN, stream);
            gemm_bf16_grouped(eng, eng->d_moe_wbf[1], (size_t)EFFN * H, eng->d_moe_xbf,
                              eng->d_moe_up, hc, ho, E, H, EFFN, stream);
            dg_silu_mul(eng->d_moe_act, eng->d_moe_gate, eng->d_moe_up, (int64_t)EFFN * total, stream);
            moe_profile_activation(eng, l, 3, eng->d_moe_act, (size_t)total*EFFN, stream);
            f32_to_bf16_kernel<<<(unsigned)(((size_t)total * EFFN + 255) / 256), 256, 0, stream>>>(
                eng->d_moe_xbf, eng->d_moe_act, (uint64_t)total * EFFN);
            // down: one grouped-batched GEMM over the active experts (in=EFFN, out=H).
            gemm_bf16_grouped(eng, eng->d_moe_wbf[2], (size_t)EFFN * H, eng->d_moe_xbf,
                              eng->d_moe_oe, hc, ho, E, EFFN, H, stream);
        } else if (fp8) {
            const __nv_bfloat16 *gs = (const __nv_bfloat16*)expert_gate.weight.scale;
            const __nv_bfloat16 *us = (const __nv_bfloat16*)expert_up.weight.scale;
            const __nv_bfloat16 *ds = (const __nv_bfloat16*)expert_down.weight.scale;
            const uint8_t *down_w = expert_down.weight.data;
            // FUSED gate+up+SiLU: one launch shares the x reads and routing lookups and writes
            // silu(gate)*up directly (no gate/up round-trip, no dg_silu_mul launch). Same math
            // order and __expf as the unfused trio → bit-identical.
            fp8_block_gemm_grouped_gateup_silu_launch(eng->d_moe_act,
                                          (const uint8_t*)gate_w, (const uint8_t*)up_w, fp8_wslab,
                                          gs, us, fp8_sslab,
                                          eng->d_moe_xe, eng->d_moe_coloff, eng->d_moe_count,
                                          eng->d_moe_active, n_slot, E, H, EFFN, stream);
            moe_profile_activation(eng, l, 3, eng->d_moe_act, (size_t)total*EFFN, stream);
            fp8_block_gemm_grouped_launch(eng->d_moe_oe, down_w, fp8_wslab, ds, fp8_sslab,
                                          eng->d_moe_act, eng->d_moe_coloff, eng->d_moe_count,
                                          eng->d_moe_active, n_slot, E, EFFN, H, stream);
        } else {
            dg_quantize_q8_1(eng->d_moe_xe, eng->d_moe_q8, eng->d_moe_q8d, eng->d_moe_q8s, H, total, stream);
            const int *q4act = eng->moe_experts_q4k ? eng->d_moe_active : NULL;
            // Decode-sized calls (<=32 tokens): the warp-per-row grouped GEMV — the tiled MMQ is
            // prefill-shaped (16-col tiles) and measured 49 GB/s on 1-token experts. Wide calls
            // (prefill chunks) keep the tile.
            const bool q4gemv = eng->moe_experts_q4k && cn <= 2 * FP8_MAXB;
            if (q4gemv) {
                mmvq_q4_k_grouped_gemv_launch(eng->d_moe_gate, gate_w, expert_gate.weight_stride,
                                    eng->d_moe_q8, eng->d_moe_q8d, eng->d_moe_q8s,
                                    eng->d_moe_coloff, eng->d_moe_count, q4act, n_slot, E, H, EFFN, stream);
                mmvq_q4_k_grouped_gemv_launch(eng->d_moe_up, up_w, expert_up.weight_stride,
                                    eng->d_moe_q8, eng->d_moe_q8d, eng->d_moe_q8s,
                                    eng->d_moe_coloff, eng->d_moe_count, q4act, n_slot, E, H, EFFN, stream);
            } else {
            dg_mmq_q4_K_grouped(eng->d_moe_gate, gate_w, expert_gate.weight_stride,
                                eng->d_moe_q8, eng->d_moe_q8d, eng->d_moe_q8s,
                                eng->d_moe_coloff, eng->d_moe_count, q4act, n_slot, E, H, EFFN, stream);
            dg_mmq_q4_K_grouped(eng->d_moe_up, up_w, expert_up.weight_stride,
                                eng->d_moe_q8, eng->d_moe_q8d, eng->d_moe_q8s,
                                eng->d_moe_coloff, eng->d_moe_count, q4act, n_slot, E, H, EFFN, stream);
            }
            // SiLU-GLU → quantize → grouped down (Q4_K native or Q8_0 requant).
            dg_silu_mul(eng->d_moe_act, eng->d_moe_gate, eng->d_moe_up, (int64_t)EFFN * total, stream);
            moe_profile_activation(eng, l, 3, eng->d_moe_act, (size_t)total*EFFN, stream);
            dg_quantize_q8_1(eng->d_moe_act, eng->d_moe_q8, eng->d_moe_q8d, eng->d_moe_q8s, EFFN, total, stream);
            if (eng->moe_down_q8[l]) {
                dg_mmq_q8_0_grouped(eng->d_moe_oe, eng->moe_down_q8[l], eng->moe_down_slab_q8,
                                    eng->d_moe_q8, eng->d_moe_q8d, eng->d_moe_q8s,
                                    eng->d_moe_coloff, eng->d_moe_count, E, EFFN, H, stream);
            } else {
                const void *down_w = expert_down.weight.data;
                if (q4gemv)
                    mmvq_q4_k_grouped_gemv_launch(eng->d_moe_oe, down_w, expert_down.weight_stride,
                                        eng->d_moe_q8, eng->d_moe_q8d, eng->d_moe_q8s,
                                        eng->d_moe_coloff, eng->d_moe_count, q4act, n_slot, E, EFFN, H, stream);
                else
                    dg_mmq_q4_K_grouped(eng->d_moe_oe, down_w, expert_down.weight_stride,
                                    eng->d_moe_q8, eng->d_moe_q8d, eng->d_moe_q8s,
                                    eng->d_moe_coloff, eng->d_moe_count, q4act, n_slot, E, EFFN, H, stream);
            }
        }
        // Combine expert outputs back to tokens, weighted by router prob (d_moe_ecs). Deterministic
        // per-token reduce (fixed k order via d_moe_invpos) — no atomicAdd, so bit-identical
        // run-to-run (was nondeterministic atomic scatter-add). out is fully written (no memset).
        dg_moe_reduce(out, eng->d_moe_oe, eng->d_moe_invpos, eng->d_moe_ecs, H, cn, U, stream);
        timing_mark(&eng->q35.prefill_expert_ms);
        // Qwen3.5-MoE shared expert: out += sigmoid(h·shared_expert_gate) · down_s(silu(gate_s(h))·up_s(h)).
        // Runs on the cn TOKENS (not assignments); reuses the grouped scratch, all consumed above.
        if (eng->moe_shared_inter > 0) {
            const int SI = eng->moe_shared_inter;
            const uint8_t *swg = weight_fp8(eng, eng->moe_sh_gate[l]);
            const uint8_t *swu = weight_fp8(eng, eng->moe_sh_up[l]);
            const uint8_t *swd = weight_fp8(eng, eng->moe_sh_down[l]);
            if (tc) {
                // Tensor-core shared expert: the 3 projections are tiny (SI×H FP8 ≈ 1 MB each),
                // so their BF16 forms PERSIST across the whole serve (L×3×SI×H BF16 ≈ 240 MB)
                // — dequant once per layer on first touch, then plain BF16 GEMMs. Replaces the
                // FP8_MAXB chunk loops that were 12.3% of a 2k-token prefill (15k launches).
                const size_t per = (size_t)SI * H;
                if (!eng->d_moe_shbf) {
                    if (cudaMalloc(&eng->d_moe_shbf, (size_t)eng->cfg.n_layers * 3 * per *
                                   sizeof(__nv_bfloat16)) != cudaSuccess) {
                        cudaGetLastError(); eng->moe_tc_off = 1; eng->d_moe_shbf = NULL;
                    } else {
                        memset(eng->moe_shbf_ready, 0, sizeof(eng->moe_shbf_ready));
                    }
                }
                __nv_bfloat16 *shbf = eng->d_moe_shbf ? eng->d_moe_shbf + (size_t)l * 3 * per : NULL;
                if (shbf && !eng->moe_shbf_ready[l]) {
                    dequant_fp8_block_to_bf16_kernel<<<(unsigned)((per + 255)/256),256,0,stream>>>(
                        shbf,           swg, wscale_fp8(eng, swg), H,  per);
                    dequant_fp8_block_to_bf16_kernel<<<(unsigned)((per + 255)/256),256,0,stream>>>(
                        shbf + per,     swu, wscale_fp8(eng, swu), H,  per);
                    dequant_fp8_block_to_bf16_kernel<<<(unsigned)((per + 255)/256),256,0,stream>>>(
                        shbf + 2*per,   swd, wscale_fp8(eng, swd), SI, per);
                    eng->moe_shbf_ready[l] = 1;
                }
                if (shbf) {
                    f32_to_bf16_kernel<<<(unsigned)(((size_t)cn * H + 255)/256),256,0,stream>>>(
                        eng->d_moe_xbf, h, (uint64_t)cn * H);
                    gemm_bf16(eng, shbf,       eng->d_moe_xbf, eng->d_moe_gate, H, SI, cn);
                    gemm_bf16(eng, shbf + per, eng->d_moe_xbf, eng->d_moe_up,   H, SI, cn);
                    dg_silu_mul(eng->d_moe_act, eng->d_moe_gate, eng->d_moe_up, (int64_t)SI * cn, stream);
                    moe_profile_activation(eng, l, 4, eng->d_moe_act, (size_t)cn*SI, stream);
                    f32_to_bf16_kernel<<<(unsigned)(((size_t)cn * SI + 255)/256),256,0,stream>>>(
                        eng->d_moe_xbf, eng->d_moe_act, (uint64_t)cn * SI);
                    gemm_bf16(eng, shbf + 2*per, eng->d_moe_xbf, eng->d_moe_xe, SI, H, cn);
                }
            }
            if (!tc || !eng->d_moe_shbf) {
                for (int b0 = 0; b0 < cn; b0 += FP8_MAXB) {
                    int bb = (cn - b0 < FP8_MAXB) ? (cn - b0) : FP8_MAXB;
                    // Gate+up in ONE dual launch: SI (512) rows/projection alone caps the
                    // warp pool; the doubled grid halves the parallelism starvation.
                    fp8_block_gemm_dual_launch(eng->d_moe_gate + (size_t)b0 * SI,
                                               eng->d_moe_up + (size_t)b0 * SI,
                                               swg, swu, wscale_fp8(eng, swg), wscale_fp8(eng, swu),
                                               h + (size_t)b0 * H, H, SI, bb, stream);
                }
                dg_silu_mul(eng->d_moe_act, eng->d_moe_gate, eng->d_moe_up, (int64_t)SI * cn, stream);
                moe_profile_activation(eng, l, 4, eng->d_moe_act, (size_t)cn*SI, stream);
                for (int b0 = 0; b0 < cn; b0 += FP8_MAXB) {
                    int bb = (cn - b0 < FP8_MAXB) ? (cn - b0) : FP8_MAXB;
                    fp8_block_gemm_launch(eng->d_moe_xe + (size_t)b0 * H, swd, wscale_fp8(eng, swd),
                                          eng->d_moe_act + (size_t)b0 * SI, SI, H, bb, stream);
                }
            }
            m2_gemm(eng->d_moe_shlog, h, (const float*)weight_fp8(eng, eng->moe_sh_gatevec[l]),
                    cn, H, 1, stream);
            q35moe_b_shared_axpy_kernel<<<dim3((unsigned)((H + 255) / 256), cn), 256, 0, stream>>>(
                out, eng->d_moe_xe, eng->d_moe_shlog, H);
        }
        timing_mark(&eng->q35.prefill_shared_ms);
    }
    if (timing) { cudaEventDestroy(tev0); cudaEventDestroy(tev1); }
}

// Per-row multiseq sampler (defined far below; paged_prefill_batched routes the
// last-token logits through it for first-token RNG parity with seq_add).
__global__ void sample_logits_ms_kernel(
    const float *logits, int V,
    const float *temps, const int *top_ks, const float *top_ps, const float *min_ps,
    const float *rnds, int *out_ids);

// Fast single-pass Qwen3 prefill (per-tensor Q4_K/Q6_K → BF16 dequant + cuBLAS GEMM,
// head_dim=128 full-causal attention). Defined after paged_prefill_batched; the Qwen3
// arch branch in paged_prefill_batched routes here. Returns 0 / -1 / -2 (same contract).
static int ensure_spec_scratch(gemma4_engine_t *eng);
static int ensure_ms_scratch(gemma4_engine_t *eng);
static int paged_slot_sync(gemma4_engine_t *eng, gemma4_seq *s, int pos);

// ─────────────────────────────────────────────────────────────────────────────
// Single-pass PAGED prefill (Phase 4 for the paged path).
//
// Clones the proven gemma4_engine_prefill_batched per-layer body VERBATIM (embed,
// RMSNorm, gemm_proj BF16/MMQ/NVFP4, per-head RMS-norm, RoPE, materialized
// [HEADS][N×N] tensor-core attention) but writes the projected K/V of ALL N prompt
// tokens into the SLOT's PAGED pools in ONE paged_kv_write launch per layer instead
// of one token-by-token decode pass. Replaces N full weight passes with ONE.
//
// Fresh-slot only (s->n_tokens==0, base=0). Gated by the caller to N <= sliding
// window (1024) so no sliding block is recycled mid-prefill. A P<=1024 prompt
// spans up to ceil(1024/256)=4 paged blocks; the scatter is multi-block-safe via
// the per-row block_table indirection (paged_elem_index), it does NOT assume one
// block. N>window / NVFP4-flash-unsupported / non-fresh fall back to the
// token-by-token seq_add loop (unchanged).
//
// Determinism: the K/V are computed with the SAME projection GEMMs, per-head
// RMS-norm order, RoPE thetas (sliding 10000 / global 1000000) and fp8 codec
// (paged_kv_write → pkv_float_to_fp8/kv_codec_value) as decode_multiseq_body, and
// scattered into the SAME per-layer pool slices (l*pls / global_slot[l]*pls) — so
// the paged KV after this pass is position-for-position identical to what the
// token-by-token loop produced. The last-token logits tail is reused verbatim and
// routed through the SAME greedy argmax / splitmix64 sampler as seq_add, so the
// first sampled token matches the token-by-token path. Returns 0 on success (and
// writes *first_tok_out + sets s->n_tokens / s->n_sampled), -2 to fall back, -1 on
// hard error.
// Cache-aware GLOBAL block-table growth for a BATCH slot. When the cross-request
// prefix cache is active, pull blocks via the refcounted allocator (free_stack
// first, then LRU reclaim) so every block a slot holds is reference-counted,
// registrable, and reclaimable; otherwise the plain pool allocator. The SLIDING
// table always uses the plain allocator (the cache is global-pool only and gated
// to the no-sliding geometry). NOT used by the single-flight eng->cur path.
static inline int glob_table_ensure(gemma4_engine_t *eng, gemma4_seq *s, int n_tokens) {
    if (eng->prefix_cache_enabled)
        return prefix_table_ensure(&eng->glob_prefix, &eng->glob_pool, &s->glob_bt, n_tokens);
    return paged_table_ensure(&eng->glob_pool, &s->glob_bt, n_tokens);
}

static int paged_prefill_batched(
    gemma4_engine_t *eng, gemma4_seq *s, const int32_t *tokens, int N,
    int32_t *first_tok_out)
{
    if (!eng->loaded || N <= 0) return -1;
    if (!eng->paged_enabled) return -2;
    // Qwen3 uses its own fast single-pass prefill (per-tensor Q4_K/Q6_K → BF16 dequant +
    // cuBLAS GEMM, head_dim=128 full-causal attention) — the Gemma sliding/global/sandwich-norm
    // body below does not apply. Same 0/-1/-2 contract; -2 falls back to token-by-token.
    // Qwen3-MoE shares this exact prefill path; only its FFN block differs (gated inside).
    if (s->n_tokens != 0) return -2;                       // fresh slot only
    if (N > eng->global_kv_capacity) return -2;            // would overflow pool
    if (N > 4096) return -2;                               // persistent scratch cap
    if (N > GEMMA4_SLIDING_WINDOW) return -2;              // avoid mid-prefill sliding recycle
    if (!eng->pf_scratch_ready && alloc_prefill_scratch(eng) != 0)
        return -2;

    cudaStream_t stream = eng->stream;
    const int H   = eng->cfg.hidden_size;
    const int I   = eng->cfg.intermediate;
    const int HD2 = 32 * sizeof(float);
    const int base = 0;                                    // fresh slot

    // Weight-format selection — IDENTICAL to gemma4_engine_prefill_batched so
    // NVFP4 / MMQ / BF16 models all keep working.
    bool use_mmq = mmq_enabled(eng, N);
    static int fp4_opt = -1, fp4_min = 1024, fp4_force = 0;
    if (fp4_opt < 0) {
        const char *e = getenv("FUCINA_FP4"); fp4_opt = (e && e[0]=='0') ? 0 : 1;
        fp4_force = (e && e[0]=='1') ? 1 : 0;
        const char *mn = getenv("FUCINA_FP4_MIN"); if (mn) fp4_min = atoi(mn);
    }
    bool fp4_budget = eng->fp4_budget_ok || fp4_force;
    bool use_fp4 = fp4_opt && fp4_budget && !use_mmq && N >= fp4_min
                         && build_fp4_weights(eng) == 0 && ensure_fp4_act(eng, N) == 0;
    if (eng->format == FORMAT_NVFP4) {
        if (ensure_fp4_act(eng, N) != 0) {
            fprintf(stderr, "fucina: NVFP4 activation scratch alloc failed (N=%d)\n", N);
            return -1;
        }
        use_mmq = false; use_fp4 = true;
    }
    if (!use_mmq && !use_fp4 && build_bf16_weights(eng) != 0) return -1;

    // ── Reserve ALL N positions in the slot's paged tables ONCE (base stays 0;
    //    N<=window ⇒ no sliding recycle). Mirror of paged_slot_sync(pos=N-1).
    if (paged_table_ensure(&eng->slid_pool, &s->slid_bt, N) != 0) return -2;
    if (glob_table_ensure(eng, s, N) != 0) return -2;
    if (paged_upload_blocks(&s->slid_bt, &s->d_slid_blocks, &s->d_slid_cap, stream) != 0) return -1;
    if (paged_upload_blocks(&s->glob_bt, &s->d_glob_blocks, &s->d_glob_cap, stream) != 0) return -1;

    // Build the N-row replicated write batch ONCE: every row r writes position r of
    // THIS one sequence. paged_kv_write resolves batch.seqs[row] per row, so the
    // single view is replicated N times. write_pos[r]=r comes from d_pf_wpos (iota).
    {
        // N is gated to <= GEMMA4_SLIDING_WINDOW (1024); spans up to 4 paged blocks
        // (256 tok each), resolved per-row via block_table. 1024*16B*2 = 32KB stack.
        PagedSeqView hvs[GEMMA4_SLIDING_WINDOW], hvg[GEMMA4_SLIDING_WINDOW];
        PagedSeqView vs; vs.block_table = s->d_slid_blocks; vs.n_blocks = s->slid_bt.n;
        vs.base = s->slid_bt.base; vs.n_tokens = N;
        PagedSeqView vg; vg.block_table = s->d_glob_blocks; vg.n_blocks = s->glob_bt.n;
        vg.base = s->glob_bt.base; vg.n_tokens = N;
        for (int r = 0; r < N; r++) { hvs[r] = vs; hvg[r] = vg; }
        cudaMemcpyAsync(eng->d_pf_views_slid, hvs, (size_t)N*sizeof(PagedSeqView), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(eng->d_pf_views_glob, hvg, (size_t)N*sizeof(PagedSeqView), cudaMemcpyHostToDevice, stream);
    }
    PagedWriteBatch wb_slid; wb_slid.seqs = eng->d_pf_views_slid; wb_slid.write_pos = eng->d_pf_wpos; wb_slid.n_rows = N;
    PagedWriteBatch wb_glob; wb_glob.seqs = eng->d_pf_views_glob; wb_glob.write_pos = eng->d_pf_wpos; wb_glob.n_rows = N;

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, stream);

    const bool ovl = !use_mmq && !use_fp4;
    const int OQ_MAX = eng->cfg.n_heads * GEMMA4_GLOBAL_HEAD_DIM;
    const int OKV_MAX = eng->cfg.n_kv_sliding * GEMMA4_HEAD_DIM;
    const int HEADS = eng->cfg.n_heads;
    (void)OQ_MAX; (void)OKV_MAX;

    // Persistent prefill scratch (N<=4096; required above).
    float *d_x = eng->d_pf_x, *d_q = eng->d_pf_q;
    float *d_k = eng->d_pf_k, *d_v = eng->d_pf_v, *d_attn = eng->d_pf_attn;
    float *d_gate = eng->d_pf_gate, *d_up = eng->d_pf_up, *d_scores = eng->d_pf_scores;
    __nv_bfloat16 *d_inb = eng->d_pf_inb, *d_qb = eng->d_pf_qb, *d_kb = eng->d_pf_kb;
    __nv_bfloat16 *d_vb = eng->d_pf_vb, *d_kbx = eng->d_pf_kbx, *d_vbx = eng->d_pf_vbx;
    __nv_bfloat16 *d_pb = eng->d_pf_pb;

    auto grid1d = [](size_t n){ return (unsigned)((n + 255) / 256); };

    int curbuf = 0;
    auto issue_dequant = [&](int m){
        int b = m & 1;
        if (m >= 2) cudaStreamWaitEvent(eng->dq_stream, eng->ev_gemm_done[b], 0);
        dequant_layer_bf16_buf(eng, m, b, eng->dq_stream);
        cudaEventRecord(eng->ev_dq_done[b], eng->dq_stream);
    };
    auto gemm_proj = [&](int l, int p, int in_dim, int out_dim, float *dst){
        if (use_fp4) {
            gemm_nvfp4(eng, l, p, d_inb, dst, in_dim, out_dim, N, stream);
        } else if (use_mmq) {
            uint64_t off; int idim, odim;
            proj_desc(eng, l, p, &off, &idim, &odim);
            mmq_proj(eng, weight_fp8(eng, off), d_inb, dst, in_dim, out_dim, N, stream);
        } else {
            gemm_bf16(eng, eng->d_bf16_layer[curbuf][p], d_inb, dst, in_dim, out_dim, N);
        }
    };

    // ── Embedding + √H scale, token-major [N][H] ──
    // tokens are host prompt ids; embed_w wants a device token buffer. Reuse the
    // multi-seq token scratch d_sb[0] (sized GEMMA4_MAX_SEQS; N may exceed it), so
    // upload through a dedicated path: d_pf_wpos is int-typed but holds the iota, not
    // tokens — copy the prompt into a temporary device buffer via d_q's bytes? No:
    // embed_w reads int32 ids. Use a small H2D into a reused int buffer.
    {
        // Reuse d_kbx (bf16, plenty large: N*OQ bf16 >= N ints) as a scratch byte area.
        int32_t *d_tok = (int32_t*)d_kbx;
        cudaMemcpyAsync(d_tok, tokens, (size_t)N*sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        embed_w(eng, d_x, weight_fp8(eng, eng->tensors.token_embd), d_tok, N, H, stream);
    }
    scale_kernel<<<grid1d((size_t)N*H),256,0,stream>>>(d_x, N*H, sqrtf((float)H));

    if (ovl) issue_dequant(0);
    for (int l = 0; l < eng->cfg.n_layers; l++) {
        layer_type_t lt = eng->layer_types[l];
        int hd  = (lt==LAYER_SLIDING)? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM;
        int nkv = (lt==LAYER_SLIDING)? eng->cfg.n_kv_sliding : eng->cfg.n_kv_global;
        int oq  = HEADS*hd, okv = nkv*hd;
        const float *w_attn   = eng->d_w_attn_norm      + (size_t)l*H;
        const float *w_post_a = eng->d_w_post_attn_norm + (size_t)l*H;
        const float *w_ffn    = eng->d_w_ffn_norm       + (size_t)l*H;
        const float *w_post_f = eng->d_w_post_ffn_norm  + (size_t)l*H;
        const float *w_qn     = eng->d_w_q_norm + (size_t)l*GEMMA4_GLOBAL_HEAD_DIM;
        const float *w_kn     = eng->d_w_k_norm + (size_t)l*GEMMA4_GLOBAL_HEAD_DIM;

        if (ovl) {
            curbuf = l & 1;
            if (l + 1 < eng->cfg.n_layers) issue_dequant(l + 1);
            cudaStreamWaitEvent(stream, eng->ev_dq_done[curbuf], 0);
        }

        rms_norm_rows_bf16_kernel<<<N,256,HD2,stream>>>(d_inb, d_x, w_attn, H, N, GEMMA4_RMS_EPS);

        gemm_proj(l, PJ_Q, H, oq, d_q);
        per_head_rms_norm_rows_kernel<<<dim3(HEADS,N),hd,HD2,stream>>>(d_q, w_qn, HEADS, hd, N, GEMMA4_RMS_EPS);
        gemm_proj(l, PJ_K, H, okv, d_k);
        if (lt == LAYER_SLIDING) {
            gemm_proj(l, PJ_V, H, okv, d_v);
            per_head_rms_norm_rows_kernel<<<dim3(nkv,N),hd,HD2,stream>>>(d_v, NULL, nkv, hd, N, GEMMA4_RMS_EPS);
        } else {
            cudaMemcpyAsync(d_v, d_k, (size_t)N*okv*sizeof(float), cudaMemcpyDeviceToDevice, stream);
            per_head_rms_norm_rows_kernel<<<dim3(nkv,N),hd,HD2,stream>>>(d_v, NULL, nkv, hd, N, GEMMA4_RMS_EPS);
        }
        per_head_rms_norm_rows_kernel<<<dim3(nkv,N),hd,HD2,stream>>>(d_k, w_kn, nkv, hd, N, GEMMA4_RMS_EPS);

        if (lt == LAYER_SLIDING)
            rope_rows_kernel<<<dim3(HEADS,N),hd/2,0,stream>>>(d_q, d_k, base, HEADS, nkv, hd, N, 10000.0f, NULL);
        else
            rope_rows_kernel<<<dim3(HEADS,N),hd/2,0,stream>>>(d_q, d_k, base, HEADS, nkv, hd, N, 1000000.0f, eng->d_rope_freqs);

        // Materialized [HEADS][N×N] tensor-core attention (chunk-internal, base=0,
        // no history term) — VERBATIM from prefill_batched.
        int window = (lt==LAYER_SLIDING)? GEMMA4_SLIDING_WINDOW : 0;
        f32_to_bf16_kernel<<<grid1d((size_t)N*oq),256,0,stream>>>(d_qb, d_q, (size_t)N*oq);
        f32_to_bf16_kernel<<<grid1d((size_t)N*okv),256,0,stream>>>(d_kb, d_k, (size_t)N*okv);
        f32_to_bf16_kernel<<<grid1d((size_t)N*okv),256,0,stream>>>(d_vb, d_v, (size_t)N*okv);
        {
            kv_broadcast_bf16_kernel<<<dim3(grid1d(hd),HEADS,N),256,0,stream>>>(d_kbx, d_kb, N, HEADS, nkv, hd);
            kv_broadcast_bf16_kernel<<<dim3(grid1d(hd),HEADS,N),256,0,stream>>>(d_vbx, d_vb, N, HEADS, nkv, hd);
            const float a1=1.0f, b0=0.0f;
            long long sNN = (long long)N * N;
            cublasGemmStridedBatchedEx(eng->cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, N, hd,
                &a1, d_qb,  CUDA_R_16BF, oq, (long long)hd,
                     d_kbx, CUDA_R_16BF, oq, (long long)hd,
                &b0, d_scores, CUDA_R_32F, N, sNN,
                HEADS, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            attn_softmax_batched_kernel<<<dim3(N,HEADS),256,HD2,stream>>>(d_scores, d_pb, N, window);
            cublasGemmStridedBatchedEx(eng->cublas, CUBLAS_OP_N, CUBLAS_OP_T, hd, N, N,
                &a1, d_vbx, CUDA_R_16BF, oq, (long long)hd,
                     d_pb,  CUDA_R_16BF, N,  sNN,
                &b0, d_attn, CUDA_R_32F, oq, (long long)hd,
                HEADS, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }

        // ── PAGED write: scatter ALL N tokens' K/V into the slot's pool slice for
        //    this layer (replaces kv_write_sliding/global_kernel). Per-layer pool
        //    base + okv stride MUST match decode_multiseq_body exactly.
        {
            size_t pls; pkv_t *pk, *pv; PagedWriteBatch *wb;
            if (lt == LAYER_SLIDING) {
                pls = (size_t)eng->slid_pool.n_blocks * PAGED_KV_BLOCK_TOKENS * okv;
                pk = eng->d_slid_pool_k + (size_t)l * pls;
                pv = eng->d_slid_pool_v + (size_t)l * pls;
                wb = &wb_slid;
            } else {
                int slot = eng->global_slot[l];
                pls = (size_t)eng->glob_pool.n_blocks * PAGED_KV_BLOCK_TOKENS * okv;
                pk = eng->d_glob_pool_k + (size_t)slot * pls;
                pv = eng->d_glob_pool_v + (size_t)slot * pls;
                wb = &wb_glob;
            }
            paged_kv_write<<<dim3(grid1d(okv),N),256,0,stream>>>(
                pk, pv, d_k, d_v, *wb, PAGED_KV_BLOCK_TOKENS, okv);
        }

        // O projection → temp d_q ; post-attn norm ; fold into residual d_x
        f32_to_bf16_kernel<<<grid1d((size_t)N*oq),256,0,stream>>>(d_inb, d_attn, (size_t)N*oq);
        gemm_proj(l, PJ_O, oq, H, d_q);
        rms_norm_residual_add_kernel<<<N,256,256*sizeof(float),stream>>>(d_x, d_q, w_post_a, H, N, GEMMA4_RMS_EPS);

        rms_norm_rows_bf16_kernel<<<N,256,HD2,stream>>>(d_inb, d_x, w_ffn, H, N, GEMMA4_RMS_EPS);
        gemm_proj(l, PJ_GATE, H, I, d_gate);
        gemm_proj(l, PJ_UP,   H, I, d_up);
        geglu_bf16_kernel<<<grid1d((size_t)N*I),256,0,stream>>>(d_inb, d_gate, d_up, N*I);
        gemm_proj(l, PJ_DOWN, I, H, d_q);
        if (ovl) cudaEventRecord(eng->ev_gemm_done[curbuf], stream);
        rms_norm_residual_add_kernel<<<N,256,256*sizeof(float),stream>>>(d_x, d_q, w_post_f, H, N, GEMMA4_RMS_EPS);
        if (eng->h_out_scale[l] != 1.0f)
            scale_kernel<<<grid1d((size_t)N*H),256,0,stream>>>(d_x, N*H, eng->h_out_scale[l]);
    }
    if (ovl) cudaStreamSynchronize(eng->dq_stream);

    // ── Output norm + LM head + softcap on the LAST token only ── (verbatim tail)
    float *x_last = d_x + (size_t)(N-1)*H;
    rms_norm_kernel<<<1,256,HD2,stream>>>(eng->d_norm, x_last, eng->d_w_out_norm, H, GEMMA4_RMS_EPS);
    logits_head(eng, eng->d_logits, eng->d_norm, H, GEMMA4_VOCAB_SIZE, stream);
    logit_softcap_kernel<<<grid1d(GEMMA4_VOCAB_SIZE),256,0,stream>>>(
        eng->d_logits, GEMMA4_SOFTCAP, GEMMA4_VOCAB_SIZE);
    if (eng->n_suppress > 0)
        suppress_tokens_kernel<<<grid1d(eng->n_suppress),256,0,stream>>>(
            eng->d_logits, eng->d_suppress, eng->n_suppress, GEMMA4_VOCAB_SIZE);

    // Sample the first token EXACTLY as seq_add: greedy argmax, or the per-row
    // splitmix64 sampler at RNG index n_sampled==0. Routes through eng->d_ms_outtok.
    if (s->samp_temp <= 0.0f) {
        argmax_rows_kernel<<<1,1024,0,stream>>>(eng->d_logits, eng->d_ms_outtok, 1, GEMMA4_VOCAB_SIZE);
    } else {
        float h_temp[1] = { s->samp_temp }, h_topp[1] = { s->samp_top_p };
        float h_minp[1] = { s->samp_min_p }, h_rnd[1];
        int   h_topk[1] = { s->samp_top_k };
        uint64_t z = (s->samp_seed ? s->samp_seed : 0x9e3779b97f4a7c15ULL) + 0x9e3779b97f4a7c15ULL * (s->n_sampled + 1);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z =  z ^ (z >> 31);
        h_rnd[0] = (float)((z >> 11) * (1.0 / 9007199254740992.0));
        cudaMemcpyAsync(eng->d_ms_temp, h_temp, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(eng->d_ms_topk, h_topk, sizeof(int),   cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(eng->d_ms_topp, h_topp, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(eng->d_ms_minp, h_minp, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(eng->d_ms_rnd,  h_rnd,  sizeof(float), cudaMemcpyHostToDevice, stream);
        sample_logits_ms_kernel<<<1,1024,0,stream>>>(
            eng->d_logits, GEMMA4_VOCAB_SIZE, eng->d_ms_temp, eng->d_ms_topk,
            eng->d_ms_topp, eng->d_ms_minp, eng->d_ms_rnd, eng->d_ms_outtok);
    }
    int32_t last_tok = 0;
    cudaMemcpyAsync(&last_tok, eng->d_ms_outtok, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);

    // Seed the per-SLOT MTP recurrent h (mirror of step_batch_spec 8809-8814):
    // next-h = the last-token output-norm hidden (eng->d_norm holds [1][H]).
    if (eng->mtp.loaded && s->samp_temp <= 0.0f) {
        if (!s->d_mtp_h && cudaMalloc(&s->d_mtp_h, (size_t)H*sizeof(float)) != cudaSuccess) s->d_mtp_h = NULL;
        if (s->d_mtp_h) {
            cudaMemcpyAsync(s->d_mtp_h, eng->d_norm, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, stream);
            s->mtp_h_valid = 1;
        }
    } else {
        s->mtp_h_valid = 0;
    }

    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    eng->prefill_time_ms += ms;
    eng->n_prefill_tokens += N;
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "fucina: paged-prefill CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    s->n_tokens = N;
    s->n_sampled = 1;                 // this seq produced one token (RNG index)
    if (first_tok_out) *first_tok_out = last_tok;
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fast single-pass Qwen3 prefill.
//
// Replaces the token-by-token seq_add loop for a FRESH Qwen3 sequence: every prompt
// token's projections run as ONE cuBLAS BF16 GEMM over all N rows (tensor-core bound)
// instead of N bandwidth-bound GEMV passes. The per-tensor Q4_K/Q6_K weights are
// dequantized to BF16 once per layer via dequant_proj_to_bf16 (numerically the K-quant
// values → greedy-token faithful). Attention is the materialized [HEADS][N×N] tensor-core
// path (head_dim=128, full causal, window=0), identical in math to the Gemma global path
// but Qwen3-shaped. K/V are scattered into the SAME per-layer global pool slices as
// decode_multiseq_body, position-for-position, so a later decode continues seamlessly.
//
// Determinism vs the token-by-token path: same pre-attn RMSNorm, q/k per-head RMS-norm,
// the explicit 1/sqrt(head_dim) Q scale (Qwen3's q_norm is learned, no baked scale), RoPE
// (theta 1e6, freq_factors), SiLU-GLU FFN, plain pre-norm residual adds (no sandwich norms),
// untied Q6_K LM head and the SAME greedy-argmax / splitmix64 sampler at RNG index 0 — so the
// first sampled token matches. Returns 0 on success, -2 to fall back, -1 on hard error.
//
// STAGE 9 — base-offset (suffix) mode. base==0 is the FRESH prefill above. base>0 is a
// COMPUTE-BOUND suffix prefill: the slot already holds `base` tokens in its paged pool
// (e.g. an adopted cross-request prefix), and tokens[0..N) are the divergent suffix at
// absolute positions base..base+N-1. Same single-weight-pass GEMM projections / FFN as the
// fresh path, but: (a) RoPE uses absolute positions base+r, (b) K/V are written into the
// pool at base+r, (c) attention for suffix query i covers history [0..base) UNION causal
// suffix prefix [base..base+i] — read through the slot's PAGED block table with the SAME
// split-K flash kernels decode_multiseq_body uses (paged_global_attn_splitk_batched), in
// ≤GEMMA4_MAX_SEQS-row chunks. That makes the suffix attention BIT-IDENTICAL to
// prefill_suffix_batched (same fp8 pool, same per-row n_tokens bound, same scan/combine
// order); only the projection method (one GEMM vs per-chunk GEMV) differs, exactly as the
// fresh GEMM path differs from token-by-token — so the first sampled token + greedy
// continuation match (the gate-1 dual-path test). Full-causal Qwen3/MoE ONLY (head_dim 128,
// window 0); Gemma's sliding/global geometry is excluded by the caller. do_sample gates the
// final LM head / sampler (suffix callers that want the first token pass 1).

// Chunked FLASH prefill: process the prompt in bounded chunks, each attending the
// frozen KV cache (history) + its own keys via the online-softmax flash kernels — no
// [HEADS][N×N] score buffer, so memory is O(chunk + KV) and arbitrary context (256k+)
// is possible (the GEMM batched path OOMs past ~25k). Projections still use BF16
// tensor-core GEMMs per chunk. Same logits_out contract as prefill_batched.
//
// SUFFIX-CAPABLE: unlike prefill_batched (whose GEMM attention is chunk-internal
// N×N with no history term), every per-chunk kernel here already takes an absolute
// base — chunk 2 of a fresh prefill IS a suffix prefill. A non-empty cache
// (global_n_tokens > 0) is therefore handled by offsetting the per-chunk base by
// the existing token count: RoPE positions, the flash kernels' history bounds and
// q_base, and the flat KV writes all use absolute positions. This is what keeps
// multi-turn agent suffixes (re-rendered assistant turn + tool result) off the
// ~127 tok/s chunked-decode fallback. Returns 0 / -2 (defer) / -1.
int gemma4_engine_prefill_flash(
    gemma4_engine_t *eng, const int32_t *tokens, int n_tokens, float *logits_out)
{
    eng->mtp_h_valid = 0;   // prefill invalidates the MTP drafter's recurrent h

    if (!eng->loaded || n_tokens <= 0) return -1;
    // Qwen3 is paged-path only (see gemma4_engine_prefill_batched) — this non-paged
    // flash prefill is gemma-layout-only. Decline so it never runs on Qwen3 weights.
    if (eng->cfg.arch == GEMMA4_ARCH_QWEN3_5) return -2;
    // FORMAT_NVFP4: this path is BF16/MMQ-only (no FP4 GEMM) and would deref the NULL Q4_0
    // store. prefill_batched forces use_fp4=true for all N (so it never returns -2 here), but
    // guard anyway so a future change can't silently route NVFP4 through the Q4_0/BF16 path.
    if (eng->format == FORMAT_NVFP4) return -2;
    const int base0 = eng->cur.n_tokens;               // 0 = fresh, >0 = suffix
    if (base0 + n_tokens > eng->global_kv_capacity) return -2;  // would overflow cache
    // Tiny suffixes stay on the chunked dp4a fallback: this path pays a fixed
    // per-chunk full-model BF16 dequant (~28 GB of traffic, ~125 ms) that only
    // amortizes past a few dozen tokens; below that two 16-token dp4a chunks win.
    if (base0 > 0 && n_tokens < 32) return -2;
    if (ensure_fp_scratch(eng) != 0) return -2;           // attention tile scratch

    cudaStream_t stream = eng->stream;
    const int H = eng->cfg.hidden_size, I = eng->cfg.intermediate, HD2 = 32*sizeof(float);
    const int HEADS = eng->cfg.n_heads, N = n_tokens;
    const int C = (N < GEMMA4_FP_CHUNK) ? N : GEMMA4_FP_CHUNK;  // chunk (amortizes weight re-reads)
    const int OQ_MAX = HEADS*GEMMA4_GLOBAL_HEAD_DIM, OKV_MAX = eng->cfg.n_kv_sliding*GEMMA4_HEAD_DIM;

    // Tiled-MMQ fast path: a Q4_0 suffix whose chunk fits MMQ_MAX_N (the agentic case —
    // re-rendered turn + tool result, ~hundreds of tokens) reads native Q4_0 weights once
    // per projection, with NO per-chunk full-model BF16 dequant. Above MMQ_MAX_N the chunk
    // is large enough that the BF16 tensor-core GEMM (+ pipelined dequant) wins. Since
    // C = min(N, FP_CHUNK), N ≤ MMQ_MAX_N ⇒ every chunk cn ≤ MMQ_MAX_N. BF16 scratch
    // failure is not fatal (chunked dp4a fallback needs none), so defer instead of failing.
    const bool use_mmq = mmq_enabled(eng, N);
    if (!use_mmq && build_bf16_weights(eng) != 0) return -2;

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, stream);

    float *d_x=0,*d_norm=0,*d_q=0,*d_k=0,*d_v=0,*d_attn=0,*d_gate=0,*d_up=0;
    __nv_bfloat16 *d_inb=0;
    int ok = 1;
    #define PALLOC(p,elems) do{ if(cudaMalloc(&(p),(size_t)(elems))!=cudaSuccess){ok=0;} }while(0)
    PALLOC(d_x,(size_t)C*H*sizeof(float));   PALLOC(d_norm,(size_t)C*H*sizeof(float));
    PALLOC(d_q,(size_t)C*OQ_MAX*sizeof(float)); PALLOC(d_k,(size_t)C*OKV_MAX*sizeof(float));
    PALLOC(d_v,(size_t)C*OKV_MAX*sizeof(float)); PALLOC(d_attn,(size_t)C*OQ_MAX*sizeof(float));
    PALLOC(d_gate,(size_t)C*I*sizeof(float)); PALLOC(d_up,(size_t)C*I*sizeof(float));
    PALLOC(d_inb,(size_t)C*I*sizeof(__nv_bfloat16));
    #undef PALLOC
    float *fbufs[] = {d_x,d_norm,d_q,d_k,d_v,d_attn,d_gate,d_up};
    if (!ok) {
        fprintf(stderr, "fucina: flash-prefill scratch alloc failed (C=%d) — fallback\n", C);
        cudaGetLastError();
        for (float *p : fbufs) if(p) cudaFree(p);
        if (d_inb) cudaFree(d_inb);
        return -2;
    }
    auto grid1d = [](size_t n){ return (unsigned)((n + 255) / 256); };
    // Pipelined per-layer weight dequant (see d_bf16_layer comment): overlap layer
    // L+1's Q4_0→BF16 dequant on dq_stream with layer L's GEMMs. curbuf is the buffer
    // the current layer's GEMMs read. Re-filled per chunk (chunks sync at their end).
    const bool ovl = !use_mmq;   // BF16 path always pipelines dequant; MMQ uses none
    int curbuf = 0, mmq_l = 0;
    auto issue_dequant = [&](int m){
        int b = m & 1;
        if (m >= 2) cudaStreamWaitEvent(eng->dq_stream, eng->ev_gemm_done[b], 0);
        dequant_layer_bf16_buf(eng, m, b, eng->dq_stream);
        cudaEventRecord(eng->ev_dq_done[b], eng->dq_stream);
    };
    // MMQ reads native Q4_0 weights once per projection; mmq_l carries the current layer
    // (set per layer below) so gemm_proj can resolve the weight offset via proj_desc.
    auto gemm_proj = [&](int p, int in_dim, int out_dim, float *dst, int rows){
        if (use_mmq) {
            uint64_t off; int idim, odim;
            proj_desc(eng, mmq_l, p, &off, &idim, &odim);
            mmq_proj(eng, weight_fp8(eng, off), d_inb, dst, in_dim, out_dim, rows, stream);
        } else {
            gemm_bf16(eng, eng->d_bf16_layer[curbuf][p], d_inb, dst, in_dim, out_dim, rows);
        }
    };

    int aborted = 0;
    for (int c0 = 0; c0 < N; c0 += C) {
        if (eng->abort_req) { aborted = 1; break; }       // client gone — stop between chunks
        int cn = (N - c0 < C) ? (N - c0) : C;
        const int abs0 = base0 + c0;                      // chunk's absolute base position
        embed_w(eng, d_x, weight_fp8(eng, eng->tensors.token_embd), tokens + c0, cn, H, stream);
        scale_kernel<<<grid1d((size_t)cn*H),256,0,stream>>>(d_x, cn*H, sqrtf((float)H));

        if (ovl) issue_dequant(0);   // pipeline fill for this chunk: layer 0's weights
        for (int l = 0; l < eng->cfg.n_layers; l++) {
            // Abort granularity: one 8192-token chunk runs ~30s at long context,
            // so poll per LAYER (~0.7s). Mid-chunk abort is safe for the same
            // reason as mid-prefill: nothing is accounted until global_n_tokens
            // advances, so partial KV writes are invisible and overwritten.
            if (eng->abort_req) { aborted = 1; break; }
            layer_type_t lt = eng->layer_types[l];
            int hd  = (lt==LAYER_SLIDING)? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM;
            int nkv = (lt==LAYER_SLIDING)? eng->cfg.n_kv_sliding : eng->cfg.n_kv_global;
            int oq  = HEADS*hd, okv = nkv*hd, kvhd = okv;
            const float *w_attn   = eng->d_w_attn_norm      + (size_t)l*H;
            const float *w_post_a = eng->d_w_post_attn_norm + (size_t)l*H;
            const float *w_ffn    = eng->d_w_ffn_norm       + (size_t)l*H;
            const float *w_post_f = eng->d_w_post_ffn_norm  + (size_t)l*H;
            const float *w_qn     = eng->d_w_q_norm + (size_t)l*GEMMA4_GLOBAL_HEAD_DIM;
            const float *w_kn     = eng->d_w_k_norm + (size_t)l*GEMMA4_GLOBAL_HEAD_DIM;

            mmq_l = l;                       // MMQ: weight offset resolved per layer
            if (ovl) {
                curbuf = l & 1;
                if (l + 1 < eng->cfg.n_layers) issue_dequant(l + 1);
                cudaStreamWaitEvent(stream, eng->ev_dq_done[curbuf], 0);
            }
            rms_norm_rows_bf16_kernel<<<cn,256,HD2,stream>>>(d_inb, d_x, w_attn, H, cn, GEMMA4_RMS_EPS);
            gemm_proj(PJ_Q, H, oq, d_q, cn);
            per_head_rms_norm_rows_kernel<<<dim3(HEADS,cn),hd,HD2,stream>>>(d_q, w_qn, HEADS, hd, cn, GEMMA4_RMS_EPS);
            gemm_proj(PJ_K, H, okv, d_k, cn);
            if (lt == LAYER_SLIDING) {
                gemm_proj(PJ_V, H, okv, d_v, cn);
            } else {
                cudaMemcpyAsync(d_v, d_k, (size_t)cn*okv*sizeof(float), cudaMemcpyDeviceToDevice, stream);
            }
            per_head_rms_norm_rows_kernel<<<dim3(nkv,cn),hd,HD2,stream>>>(d_v, NULL, nkv, hd, cn, GEMMA4_RMS_EPS);
            per_head_rms_norm_rows_kernel<<<dim3(nkv,cn),hd,HD2,stream>>>(d_k, w_kn, nkv, hd, cn, GEMMA4_RMS_EPS);

            float theta = (lt==LAYER_SLIDING)? 10000.0f : 1000000.0f;
            const float *ff = (lt==LAYER_SLIDING)? NULL : eng->d_rope_freqs;
            rope_rows_kernel<<<dim3(HEADS,cn),hd/2,0,stream>>>(d_q, d_k, abs0, HEADS, nkv, hd, cn, theta, ff);

            // ── Tiled-GEMM attention (tensor cores, online softmax) ──
            // Chunk queries attend the frozen history cache (FP8 → BF16
            // tiles) + this chunk's own keys (fp32 → BF16, GQA-broadcast),
            // one K/V tile at a time: S = Q·K_tileᵀ (strided-batched GEMM
            // over all heads), fp_online_softmax folds the tile into the
            // running m/l and rescales the fp32 O accumulator (d_attn),
            // then O += P·V_tile (GEMM, beta=1). See the kernel block at
            // fp_online_softmax_kernel for layouts and masking.
            {
                const int T = GEMMA4_FP_TILE_K;
                const long long sb = (long long)T * C;    // per-head S/P stride (sliding)
                const float a1 = 1.0f, b0 = 0.0f, b1 = 1.0f;
                const int window = (lt==LAYER_SLIDING) ? GEMMA4_SLIDING_WINDOW : 0;
                const bool global = (lt != LAYER_SLIDING);
                // Wide single-KV-head GEMM fast path only when there is exactly ONE global
                // KV head (12B): all HEADS query columns share it, so S / P·V are each one
                // wide GEMM with non-broadcast keys [cn][hd]. With n_kv_global>1 (31B GQA)
                // that broadcast is invalid — route through the per-head strided-batched
                // path (same machinery as sliding), keys GQA-broadcast to [cn][HEADS*hd].
                const bool global_wide = global && (nkv == 1);

                f32_to_bf16_kernel<<<grid1d((size_t)cn*oq),256,0,stream>>>(
                    eng->d_fp_qb, d_q, (size_t)cn*oq);
                if (global_wide) {
                    // 1 KV head: no GQA broadcast — keys/values stay [cn][hd].
                    f32_to_bf16_kernel<<<grid1d((size_t)cn*okv),256,0,stream>>>(
                        eng->d_fp_kbx, d_k, (size_t)cn*okv);
                    f32_to_bf16_kernel<<<grid1d((size_t)cn*okv),256,0,stream>>>(
                        eng->d_fp_vbx, d_v, (size_t)cn*okv);
                } else {
                    kv_broadcast_f32_bf16_kernel<<<dim3(grid1d(hd),HEADS,cn),256,0,stream>>>(
                        eng->d_fp_kbx, d_k, cn, HEADS, nkv, hd);
                    kv_broadcast_f32_bf16_kernel<<<dim3(grid1d(hd),HEADS,cn),256,0,stream>>>(
                        eng->d_fp_vbx, d_v, cn, HEADS, nkv, hd);
                }
                fp_ml_init_kernel<<<grid1d((size_t)HEADS*C),256,0,stream>>>(
                    eng->d_fp_m, eng->d_fp_l, HEADS*C);

                // One tile pass over queries [q0, q0+qn); kbase = absolute
                // position of tile key 0.
                //
                // GLOBAL_WIDE (12B, n_kv_global==1; 84% of attention FLOPs): Q
                // token-major [cn][HEADS][hd] IS col-major [hd × HEADS·cn] with
                // ld=hd, and all heads share the single KV head — so S and P·V
                // are each ONE wide GEMM (n = HEADS·qn), no broadcast, no
                // batching. Keys kt/vt are [tn][hd].
                // SLIDING (GQA 16:8, window-bounded) and 31B GLOBAL GQA
                // (n_kv_global=4): strided-batched GEMM over heads with broadcast
                // keys [tn][HEADS*hd]. The only global/sliding difference is the
                // mask (window=0 for global), handled in fp_online_softmax_kernel.
                auto tile_pass = [&](const __nv_bfloat16 *kt, const __nv_bfloat16 *vt,
                                     int tn, int kbase, int q0, int qn){
                    if (qn <= 0 || tn <= 0) return;
                    if (global_wide) {
                        // S(j, qi·H+h) = K_j·Q_(qi,h), col-major [tn × H·qn] ld=T.
                        cublasGemmEx(eng->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            tn, HEADS*qn, hd,
                            &a1, kt, CUDA_R_16BF, hd,
                                 eng->d_fp_qb + (size_t)q0*oq, CUDA_R_16BF, hd,
                            &b0, eng->d_fp_st + (size_t)q0*HEADS*T, CUDA_R_16BF, T,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                        fp_online_softmax_kernel<<<dim3(qn,HEADS),256,HD2,stream>>>(
                            eng->d_fp_st, eng->d_fp_pb, d_attn, eng->d_fp_m, eng->d_fp_l,
                            tn, (long long)HEADS*T, (long long)T,
                            q0, kbase, abs0, window, hd, oq, C);
                        // O[hd × H·qn] += V_tile · P, ld=hd, beta=1.
                        cublasGemmEx(eng->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                            hd, HEADS*qn, tn,
                            &a1, vt, CUDA_R_16BF, hd,
                                 eng->d_fp_pb + (size_t)q0*HEADS*T, CUDA_R_16BF, T,
                            &b1, d_attn + (size_t)q0*oq, CUDA_R_32F, hd,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                        return;
                    }
                    // S[h](j,i) = K_j·Q_i, col-major [tn × qn] ld=T per head.
                    cublasGemmStridedBatchedEx(eng->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        tn, qn, hd,
                        &a1, kt, CUDA_R_16BF, oq, (long long)hd,
                             eng->d_fp_qb + (size_t)q0*oq, CUDA_R_16BF, oq, (long long)hd,
                        &b0, eng->d_fp_st + (size_t)q0*T, CUDA_R_16BF, T, sb,
                        HEADS, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                    fp_online_softmax_kernel<<<dim3(qn,HEADS),256,HD2,stream>>>(
                        eng->d_fp_st, eng->d_fp_pb, d_attn, eng->d_fp_m, eng->d_fp_l,
                        tn, (long long)T, sb, q0, kbase, abs0, window, hd, oq, C);
                    // O[h] += V_tile[h]·P[h], col-major [hd × qn] ld=oq, beta=1.
                    cublasGemmStridedBatchedEx(eng->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        hd, qn, tn,
                        &a1, vt, CUDA_R_16BF, oq, (long long)hd,
                             eng->d_fp_pb + (size_t)q0*T, CUDA_R_16BF, T, sb,
                        &b1, d_attn + (size_t)q0*oq, CUDA_R_32F, oq, (long long)hd,
                        HEADS, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                };

                // History tiles (frozen FP8 cache). Sliding uses the ring (cap =
                // sliding_kv_capacity); global is flat full-ctx (cap = its capacity,
                // positions < ctx so the modulo is identity).
                const kv_t *hk, *hv;
                int hcap;
                if (lt == LAYER_SLIDING) {
                    size_t lstride = (size_t)eng->sliding_kv_capacity * kvhd;
                    hk = eng->d_sliding_k + (size_t)l*lstride;
                    hv = eng->d_sliding_v + (size_t)l*lstride;
                    hcap = eng->sliding_kv_capacity;
                } else {
                    int slot = eng->global_slot[l];
                    // GQA global cache [capacity][n_kv_global][head_dim]: per-token stride
                    // is okv = nkv*hd (12B nkv=1 ⇒ okv==GEMMA4_GLOBAL_HEAD_DIM, identical).
                    size_t stride = (size_t)eng->global_kv_capacity * okv;
                    hk = eng->d_global_k + (size_t)slot*stride;
                    hv = eng->d_global_v + (size_t)slot*stride;
                    hcap = eng->global_kv_capacity;
                }
                int hstart = 0;                           // sliding: only the window reach
                if (window > 0) { hstart = abs0 - (window - 1); if (hstart < 0) hstart = 0; }
                // Tile row stride: the wide single-KV-head global GEMM uses non-broadcast
                // keys [tn][hd] (okv==hd at nkv=1); every other path (sliding GQA, or 31B
                // global GQA) uses broadcast keys [tn][HEADS*hd] (= oq). History tiles match:
                // global_wide emits 1 KV head; else GQA-broadcast HEADS heads from nkv.
                const size_t krow = global_wide ? (size_t)okv : (size_t)oq;
                const int hist_nh = global_wide ? 1 : HEADS;
                for (int t0 = hstart; t0 < abs0; t0 += T) {
                    int tn = (abs0 - t0 < T) ? (abs0 - t0) : T;
                    fp_hist_tile_bf16_kernel<<<dim3(grid1d(hd),hist_nh,tn),256,0,stream>>>(
                        eng->d_fp_kt, eng->d_fp_vt, hk, hv, t0, tn, hist_nh, nkv, hd, hcap);
                    long long qe = cn;                    // queries that can see this tile
                    if (window > 0) {
                        qe = (long long)t0 + tn - 1 + window - abs0;
                        if (qe > cn) qe = cn;
                    }
                    tile_pass(eng->d_fp_kt, eng->d_fp_vt, tn, t0, 0, (int)qe);
                }
                // Chunk tiles (causal; queries before a tile see none of it).
                for (int t0 = 0; t0 < cn; t0 += T) {
                    int tn = (cn - t0 < T) ? (cn - t0) : T;
                    long long qe = cn;
                    if (window > 0) {
                        qe = (long long)t0 + tn - 1 + window;
                        if (qe > cn) qe = cn;
                    }
                    tile_pass(eng->d_fp_kbx + (size_t)t0*krow, eng->d_fp_vbx + (size_t)t0*krow,
                              tn, abs0 + t0, t0, (int)qe - t0);
                }
                fp_attn_norm_kernel<<<dim3(cn,HEADS),256,0,stream>>>(
                    d_attn, eng->d_fp_l, hd, oq, C);
            }

            // Persist this chunk's K/V at absolute positions [abs0, abs0+cn-1].
            if (lt == LAYER_SLIDING) {
                size_t lstride = (size_t)eng->sliding_kv_capacity * kvhd;
                kv_write_sliding_kernel<<<dim3(grid1d(kvhd),cn),256,0,stream>>>(
                    eng->d_sliding_k + (size_t)l*lstride, eng->d_sliding_v + (size_t)l*lstride,
                    d_k, d_v, abs0, 0, cn, kvhd, GEMMA4_SLIDING_WINDOW,
                    eng->sliding_kv_capacity);
            } else {
                // GQA global cache [capacity][n_kv_global][head_dim]: per-token stride and
                // width are okv = nkv*hd (12B nkv=1 ⇒ okv==hd ⇒ bit-identical to before).
                int slot = eng->global_slot[l];
                size_t stride = (size_t)eng->global_kv_capacity * okv;
                kv_write_global_kernel<<<dim3(grid1d(okv),cn),256,0,stream>>>(
                    eng->d_global_k + slot*stride, eng->d_global_v + slot*stride,
                    d_k, d_v, abs0, cn, okv);
            }

            f32_to_bf16_kernel<<<grid1d((size_t)cn*oq),256,0,stream>>>(d_inb, d_attn, (size_t)cn*oq);
            gemm_proj(PJ_O, oq, H, d_q, cn);
            rms_norm_residual_add_kernel<<<cn,256,256*sizeof(float),stream>>>(d_x, d_q, w_post_a, H, cn, GEMMA4_RMS_EPS);

            rms_norm_rows_bf16_kernel<<<cn,256,HD2,stream>>>(d_inb, d_x, w_ffn, H, cn, GEMMA4_RMS_EPS);
            gemm_proj(PJ_GATE, H, I, d_gate, cn);
            gemm_proj(PJ_UP,   H, I, d_up,   cn);
            geglu_bf16_kernel<<<grid1d((size_t)cn*I),256,0,stream>>>(d_inb, d_gate, d_up, cn*I);
            gemm_proj(PJ_DOWN, I, H, d_q, cn);   // last GEMM reading curbuf's weights
            if (ovl) cudaEventRecord(eng->ev_gemm_done[curbuf], stream);
        rms_norm_residual_add_kernel<<<cn,256,256*sizeof(float),stream>>>(d_x, d_q, w_post_f, H, cn, GEMMA4_RMS_EPS);
            if (eng->h_out_scale[l] != 1.0f)
                scale_kernel<<<grid1d((size_t)cn*H),256,0,stream>>>(d_x, cn*H, eng->h_out_scale[l]);
        }
        if (ovl) cudaStreamSynchronize(eng->dq_stream);  // drain trailing/aborted dequant

        if (aborted) break;
        if (c0 + cn >= N) {                               // last chunk: logits of last token
            float *x_last = d_x + (size_t)(cn-1)*H;
            rms_norm_kernel<<<1,256,HD2,stream>>>(eng->d_norm, x_last, eng->d_w_out_norm, H, GEMMA4_RMS_EPS);
            logits_head(eng, eng->d_logits, eng->d_norm, H, GEMMA4_VOCAB_SIZE, stream);
            logit_softcap_kernel<<<grid1d(GEMMA4_VOCAB_SIZE),256,0,stream>>>(
                eng->d_logits, GEMMA4_SOFTCAP, GEMMA4_VOCAB_SIZE);
            if (eng->n_suppress > 0)
                suppress_tokens_kernel<<<grid1d(eng->n_suppress),256,0,stream>>>(
                    eng->d_logits, eng->d_suppress, eng->n_suppress, GEMMA4_VOCAB_SIZE);
            if (logits_out)
                cudaMemcpyAsync(logits_out, eng->d_logits, GEMMA4_VOCAB_SIZE*sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);                    // chunk boundary (cache state advanced)
    }
    if (!aborted) {
        eng->cur.n_tokens = base0 + N;
        // The last chunk's output-normed last-token hidden (d_norm) is exactly the
        // recurrent h the MTP drafter pairs with the first sampled token — restore it
        // (parity with the chunked-decode suffix path and gemma4_engine_decode) so
        // drafting starts immediately after a suffix prefill.
        if (eng->mtp.loaded) {
            cudaMemcpyAsync(eng->d_mtp_h, eng->d_norm,
                            H * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
            eng->mtp_h_valid = 1;
        }
    }
    // On abort: global_n_tokens stays at base0 — KV entries written by completed
    // chunks sit beyond the accounted length, are never read, and are overwritten
    // by the next prefill. The cache state is exactly "nothing happened".

    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    if (!aborted) { eng->prefill_time_ms += ms; eng->n_prefill_tokens += N; }
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    for (float *p : fbufs) cudaFree(p);
    cudaFree(d_inb);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "fucina: flash-prefill CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return aborted ? -3 : 0;
}

// =========================================================================
// ─── Engine Prefill ────────────────────────────────────────────────────
// =========================================================================

int gemma4_engine_prefill(
    gemma4_engine_t *eng,
    const int32_t   *tokens,
    int              n_tokens,
    float           *logits_out)
{
    eng->mtp_h_valid = 0;   // prefill invalidates the MTP drafter's recurrent h

    if (!eng->loaded) return -1;
    // Qwen3 is paged-path only (see gemma4_engine_prefill_batched) — this non-paged
    // token-by-token loop uses gemma-layout decode_layer. Decline cleanly rather than
    // emit garbage / crash, so the single-flight HTTP fallthrough fails gracefully.
    if (eng->cfg.arch == GEMMA4_ARCH_QWEN3_5) return -1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, eng->stream);

    if (eng->cur.n_tokens + n_tokens > eng->global_kv_capacity) {
        fprintf(stderr, "fucina: prefill of %d tokens exceeds context (%d/%d used)\n",
                n_tokens, eng->cur.n_tokens, eng->global_kv_capacity);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        return -1;
    }

    // Chunked BATCHED prefill over the quantized weights: forward GEMMA4_SPEC_MAX
    // tokens per weight pass via the same dp4a path as the spec verify (proven
    // BIT-EXACT to the old per-token loop, ~K× fewer weight reads). This is the
    // path multi-turn SUFFIX prefills land on (the bf16 batched/flash prefills
    // need a fresh sequence), so it sets continued-conversation latency. Interior
    // chunks skip the LM head (logits_out=NULL); only the final chunk computes it.
    float *chunk_logits = NULL;
    if (logits_out || eng->mtp.loaded) {
        chunk_logits = (float *)malloc((size_t)GEMMA4_SPEC_MAX * GEMMA4_VOCAB_SIZE * sizeof(float));
        if (!chunk_logits) { cudaEventDestroy(start); cudaEventDestroy(stop); return -1; }
    }
    // decode_batched accumulates into the DECODE counters; this function's own events
    // bracket everything, so restore the decode counters to keep /metrics honest.
    float dec_ms = eng->decode_time_ms; int dec_n = eng->n_decode_tokens;
    int rc = 0, lastK = 0;
    for (int t = 0; t < n_tokens && rc == 0; t += GEMMA4_SPEC_MAX) {
        if (eng->abort_req) { rc = -3; break; }  // client gone — stop between chunks
        int K = (n_tokens - t < GEMMA4_SPEC_MAX) ? (n_tokens - t) : GEMMA4_SPEC_MAX;
        int last = (t + K == n_tokens);
        rc = gemma4_engine_decode_batched(eng, tokens + t, K, last ? chunk_logits : NULL);
        if (last) lastK = K;
    }
    if (rc != 0) {
        eng->ev_pending = 0;   // drop any in-flight chunk pair: it must not leak into
        eng->decode_time_ms = dec_ms; eng->n_decode_tokens = dec_n;   // decode /metrics
        free(chunk_logits);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        // -3 = cooperative abort: completed chunks stay committed in the KV
        // (global_n_tokens advanced); the next request's Rewind trims them.
        return rc == -3 ? -3 : -1;
    }
    if (logits_out && chunk_logits)
        memcpy(logits_out, chunk_logits + (size_t)(lastK - 1) * GEMMA4_VOCAB_SIZE,
               GEMMA4_VOCAB_SIZE * sizeof(float));
    free(chunk_logits);
    // The final chunk's last output-normed row (d_sb[2]) is the forward of the last
    // prompt token — exactly the recurrent h the MTP drafter pairs with the first
    // sampled token, so drafting works from the very first generated token.
    if (eng->mtp.loaded && lastK > 0) {
        cudaMemcpyAsync(eng->d_mtp_h, eng->d_sb[2] + (size_t)(lastK - 1) * eng->cfg.hidden_size,
                        eng->cfg.hidden_size * sizeof(float), cudaMemcpyDeviceToDevice, eng->stream);
        eng->mtp_h_valid = 1;
    }

    cudaEventRecord(stop, eng->stream);
    cudaEventSynchronize(stop);

    // The stream is drained: any lazy decode pair from the chunks above is complete.
    // Resolve it NOW, then restore the decode counters — this whole prefill is billed
    // to the prefill counters below (a pair resolving after the restore would leak).
    decode_timing_lap(eng);
    eng->decode_time_ms = dec_ms; eng->n_decode_tokens = dec_n;

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    eng->prefill_time_ms += ms;
    eng->n_prefill_tokens += n_tokens;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "fucina: prefill CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

// Lazy decode timing (llama.cpp-audit #30). The hot decode path used to
// cudaEventSynchronize EVERY token to read its own timing — a full pipeline drain on
// top of the sampler's structural sync, costing several ms/step of GPU idle while the
// host refilled the launch queue. Now the (start,stop) pair is recorded but harvested
// on a LATER call once the events have naturally completed (the caller's sampler D2H
// drains the stream between steps). A pair that hasn't completed when the events are
// next needed simply stays pending and that new step goes untimed — (ms, tokens) are
// only ever counted together, so the /metrics RATE stays unbiased.
static void decode_timing_lap(gemma4_engine_t *eng)
{
    if (!eng->ev_pending) return;
    if (cudaEventQuery(eng->ev_stop) == cudaSuccess) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, eng->ev_start, eng->ev_stop) == cudaSuccess) {
            eng->decode_time_ms  += ms;
            eng->n_decode_tokens += eng->ev_pending_tokens;
        }
        eng->ev_pending = 0;
    } else {
        cudaGetLastError();   // clear cudaErrorNotReady
    }
}

// The single-token forward with DEVICE-resident inputs (d_dectok / d_decpos): the body
// captured into the decode CUDA graph. Mirrors the scalar path of gemma4_engine_decode
// exactly (decode_layer dispatches the device-pos kernel variants when d_pos != NULL).
static void decode_forward_device(gemma4_engine_t *eng, cudaStream_t stream)
{
    const int H = eng->cfg.hidden_size;
    embed_w(eng, eng->d_x, weight_fp8(eng, eng->tensors.token_embd),
            eng->d_dectok, 1, H, stream);
    float sc = sqrtf((float)H);
    scale_kernel<<<(H+255)/256, 256, 0, stream>>>(
        eng->d_x, H, sc);
    for (int l = 0; l < eng->cfg.n_layers; l++)
        decode_layer(eng, l, /*pos=*/0, /*context_len=*/0, stream, eng->d_decpos);
    rms_norm_kernel<<<1, 256, 32*sizeof(float), stream>>>(
        eng->d_norm, eng->d_x, eng->d_w_out_norm, H, GEMMA4_RMS_EPS);
    if (eng->mtp.loaded)
        cudaMemcpyAsync(eng->d_mtp_h, eng->d_norm,
                        H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    logits_head(eng, eng->d_logits, eng->d_norm, H, GEMMA4_VOCAB_SIZE, stream);
    logit_softcap_kernel<<<(GEMMA4_VOCAB_SIZE + 255) / 256, 256, 0, stream>>>(
        eng->d_logits, GEMMA4_SOFTCAP, GEMMA4_VOCAB_SIZE);
    if (eng->n_suppress > 0)
        suppress_tokens_kernel<<<(eng->n_suppress + 255) / 256, 256, 0, stream>>>(
            eng->d_logits, eng->d_suppress, eng->n_suppress, GEMMA4_VOCAB_SIZE);
}

// Lazy one-time capture of the single-token decode (same pattern as mtp_graph_ensure).
// All buffers are engine-resident and every grid is fixed (the device-pos attention
// uses fixed-grid rows kernels), so one captured graph replays for every token. Any
// capture/instantiate failure permanently falls back to the per-kernel path.
static int decode_graph_ensure(gemma4_engine_t *eng)
{
    if (eng->decode_graph) return 0;
    if (eng->decode_graph_failed) return -1;
    // Debug escape hatch (parity A/B + emergency), mirroring llama.cpp's
    // GGML_CUDA_DISABLE_GRAPHS. The graph path is the default.
    static const int disabled = (getenv("FUCINA_NO_DECODE_GRAPH") != NULL);
    if (disabled) { eng->decode_graph_failed = 1; return -1; }
    if (!eng->d_decpos && cudaMalloc(&eng->d_decpos, 2*sizeof(int)) != cudaSuccess) {
        eng->d_decpos = NULL; eng->decode_graph_failed = 1; cudaGetLastError(); return -1;
    }
    if (!eng->d_dectok && cudaMalloc(&eng->d_dectok, sizeof(int32_t)) != cudaSuccess) {
        eng->d_dectok = NULL; eng->decode_graph_failed = 1; cudaGetLastError(); return -1;
    }
    cudaMemset(eng->d_decpos, 0, 2*sizeof(int));
    cudaMemset(eng->d_dectok, 0, sizeof(int32_t));
    cudaStream_t cs = NULL;
    cudaGraph_t  g  = NULL;
    int ok = cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) == cudaSuccess;
    if (ok && cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
        decode_forward_device(eng, cs);
        ok = cudaStreamEndCapture(cs, &g) == cudaSuccess && g != NULL;
    } else {
        ok = 0;
    }
    if (ok) ok = cudaGraphInstantiate(&eng->decode_graph, g, 0) == cudaSuccess;
    if (g)  cudaGraphDestroy(g);
    if (cs) cudaStreamDestroy(cs);
    if (!ok || !eng->decode_graph) {
        eng->decode_graph = NULL;
        eng->decode_graph_failed = 1;
        cudaGetLastError();
        fprintf(stderr, "fucina: decode graph capture failed — using per-kernel launches\n");
        return -1;
    }
    fprintf(stderr, "fucina: single-token decode CUDA graph captured\n");
    return 0;
}

int gemma4_engine_decode(
    gemma4_engine_t *eng,
    int32_t          token,
    float           *logits_out)
{
    if (!eng->loaded) return -1;

    decode_timing_lap(eng);
    cudaStream_t stream = eng->stream;
    int pos = eng->cur.n_tokens;
    int timing = !eng->ev_pending;     // events free → time this step (lazy readout)
    if (timing) cudaEventRecord(eng->ev_start, stream);

    // Graph fast path: 12 bytes of device state + ONE launch instead of ~1,390.
    int used_graph = 0;
    if (decode_graph_ensure(eng) == 0) {
        int pv[2] = { pos, pos + 1 };
        cudaMemcpyAsync(eng->d_decpos, pv, sizeof(pv), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(eng->d_dectok, &token, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        if (cudaGraphLaunch(eng->decode_graph, stream) == cudaSuccess) {
            used_graph = 1;
        } else {
            cudaGetLastError();
            cudaGraphExecDestroy(eng->decode_graph);
            eng->decode_graph = NULL;
            eng->decode_graph_failed = 1;
            fprintf(stderr, "fucina: decode graph replay failed — using per-kernel launches\n");
        }
    }

    if (!used_graph) {
        const int H = eng->cfg.hidden_size;
        // Embedding (format-aware: FP8 or Q8_0 table)
        embed_w(eng,
            eng->d_x,
            weight_fp8(eng, eng->tensors.token_embd),
            &token, 1, H, stream);

        // Gemma scales embeddings by √hidden_size
        { float sc = sqrtf((float)H);
          scale_kernel<<<(H+255)/256, 256, 0, stream>>>(
                eng->d_x, H, sc); }

        // Paged KV: ensure the active sequence's block tables cover `pos` and
        // the device block ids are fresh BEFORE any layer mirrors its KV write.
        if (eng->paged_enabled) paged_seq_sync(eng, pos);

        // Run all layers
        for (int l = 0; l < eng->cfg.n_layers; l++) {
            decode_layer(eng, l, pos, eng->cur.n_tokens + 1, stream);
        }

        // Output norm + projection + softcap
        rms_norm_kernel<<<1, 256, 32*sizeof(float), stream>>>(
            eng->d_norm, eng->d_x, eng->d_w_out_norm,
            H, GEMMA4_RMS_EPS);

        // MTP recurrent h: the LM-head input (post-output-norm hidden) is exactly the
        // h the assistant drafter pairs with the token sampled from these logits.
        if (eng->mtp.loaded) {
            cudaMemcpyAsync(eng->d_mtp_h, eng->d_norm,
                            H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }

        int vocab = GEMMA4_VOCAB_SIZE;
        logits_head(eng, eng->d_logits, eng->d_norm, H, vocab, stream);

        logit_softcap_kernel<<<(vocab + 255) / 256, 256, 0, stream>>>(
            eng->d_logits, GEMMA4_SOFTCAP, vocab);

        if (eng->n_suppress > 0)
            suppress_tokens_kernel<<<(eng->n_suppress + 255) / 256, 256, 0, stream>>>(
                eng->d_logits, eng->d_suppress, eng->n_suppress, vocab);
    }

    if (eng->mtp.loaded) eng->mtp_h_valid = 1;   // both paths refreshed d_mtp_h

    // Copy logits to host
    if (logits_out) {
        cudaMemcpyAsync(logits_out, eng->d_logits,
                        GEMMA4_VOCAB_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    }

    if (timing) {
        cudaEventRecord(eng->ev_stop, stream);
        eng->ev_pending = 1;
        eng->ev_pending_tokens = 1;
    }

    eng->cur.n_tokens++;

    // The pageable-host logits copy must be complete before the caller reads it.
    // The fast paths (logits_out == NULL: device sampling) skip this drain entirely.
    if (logits_out) cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "fucina: decode CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

// ── Paged KV self-test (Phase 2 inc 3) ───────────────────────────────────────
// Compare one layer's contiguous KV against its paged-pool mirror at every live
// position; atomically count mismatching fp8 elements.
__global__ void paged_compare_layer_kernel(
        const kv_t *ck, const kv_t *cv, int contig_pos_stride,
        const kv_t *pk, const kv_t *pv,
        const int *block_table, int base, int n_blocks,
        int n_pos, int elems, int block_tokens, int *mismatch)
{
    int pos = blockIdx.x;
    PagedSeqView v; v.block_table = block_table; v.n_blocks = n_blocks;
    v.base = base; v.n_tokens = n_pos;
    for (int e = threadIdx.x; e < elems; e += blockDim.x) {
        size_t ci = (size_t)pos * contig_pos_stride + e;
        size_t po = paged_elem_index(v, pos, e, block_tokens, elems);
        if (po == (size_t)-1) continue;
        if (ck[ci] != pk[po] || cv[ci] != pv[po]) atomicAdd(mismatch, 1);
    }
}

// Decode a fixed token run with the paged mirror active, then assert the pool is
// byte-identical to the contiguous cache at every live position, both classes.
// Triggered at create when FUCINA_PAGED_KV_SELFTEST is set. Non-fatal: it logs
// PASS/FAIL and restores engine state.
static void gemma4_engine_paged_selftest(gemma4_engine_t *eng) {
    if (!eng->paged_enabled) return;
    const int BT = PAGED_KV_BLOCK_TOKENS;
    const int K  = 300;   // < GEMMA4_SLIDING_WINDOW ⇒ sliding doesn't recycle (all live)
    const int slid_elems = GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM;
    const int glob_elems = GEMMA4_GLOBAL_KV_HEADS * GEMMA4_GLOBAL_HEAD_DIM;

    int saved_failed = eng->decode_graph_failed;
    eng->decode_graph_failed = 1;               // force the non-graph path (has the mirror)
    gemma4_engine_reset(eng);
    paged_table_release(&eng->slid_pool, &eng->cur.slid_bt);
    paged_table_release(&eng->glob_pool, &eng->cur.glob_bt);

    for (int i = 0; i < K; i++) {
        if (gemma4_engine_decode(eng, (int32_t)(100 + i), NULL) != 0) {
            fprintf(stderr, "fucina: paged self-test: decode failed at %d\n", i);
            eng->decode_graph_failed = saved_failed; return;
        }
    }
    cudaStreamSynchronize(eng->stream);

    int *d_mm = NULL;
    if (cudaMalloc(&d_mm, sizeof(int)) != cudaSuccess) { eng->decode_graph_failed = saved_failed; return; }
    cudaMemset(d_mm, 0, sizeof(int));
    size_t slid_cstride = (size_t)eng->sliding_kv_capacity * slid_elems;
    size_t slid_pstride = (size_t)eng->slid_pool.n_blocks  * BT * slid_elems;
    size_t glob_cstride = (size_t)eng->global_kv_capacity  * glob_elems;
    size_t glob_pstride = (size_t)eng->glob_pool.n_blocks  * BT * glob_elems;
    for (int L = 0; L < eng->cfg.n_layers; L++) {
        if (eng->layer_types[L] == LAYER_SLIDING) {
            paged_compare_layer_kernel<<<K, 256>>>(
                eng->d_sliding_k + (size_t)L*slid_cstride, eng->d_sliding_v + (size_t)L*slid_cstride, slid_elems,
                eng->d_slid_pool_k + (size_t)L*slid_pstride, eng->d_slid_pool_v + (size_t)L*slid_pstride,
                eng->cur.d_slid_blocks, eng->cur.slid_bt.base, eng->cur.slid_bt.n,
                K, slid_elems, BT, d_mm);
        } else {
            int slot = eng->global_slot[L];
            paged_compare_layer_kernel<<<K, 256>>>(
                eng->d_global_k + (size_t)slot*glob_cstride, eng->d_global_v + (size_t)slot*glob_cstride, glob_elems,
                eng->d_glob_pool_k + (size_t)slot*glob_pstride, eng->d_glob_pool_v + (size_t)slot*glob_pstride,
                eng->cur.d_glob_blocks, eng->cur.glob_bt.base, eng->cur.glob_bt.n,
                K, glob_elems, BT, d_mm);
        }
    }
    int mm = -1;
    cudaMemcpy(&mm, d_mm, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_mm);
    if (mm == 0)
        fprintf(stderr, "fucina: paged self-test PASSED — pool == contiguous over %d tokens × %d layers\n",
                K, eng->cfg.n_layers);
    else
        fprintf(stderr, "fucina: paged self-test FAILED — %d mismatching fp8 elements\n", mm);

    gemma4_engine_reset(eng);
    paged_table_release(&eng->slid_pool, &eng->cur.slid_bt);
    paged_table_release(&eng->glob_pool, &eng->cur.glob_bt);
    eng->decode_graph_failed = saved_failed;
}

// Validate the paged READ kernel against the production contiguous attention on
// real decoded KV: decode a fixed run (mirror populates the pool, proven == the
// contiguous cache), then for the first sliding and first global layer feed a
// synthetic Q through BOTH the contiguous broadcast and paged_attn_decode_kernel
// and assert the outputs agree (online-softmax over the same KV; tolerance for
// FMA-order divergence). Gated by FUCINA_PAGED_READ_SELFTEST.
static void gemma4_engine_paged_read_selftest(gemma4_engine_t *eng) {
    if (!eng->paged_enabled) return;
    const int BT = PAGED_KV_BLOCK_TOKENS;
    const int K  = 300;
    const int slid_elems = GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM;
    const int glob_elems = GEMMA4_GLOBAL_KV_HEADS * GEMMA4_GLOBAL_HEAD_DIM;

    int saved_failed = eng->decode_graph_failed;
    eng->decode_graph_failed = 1;
    gemma4_engine_reset(eng);
    paged_table_release(&eng->slid_pool, &eng->cur.slid_bt);
    paged_table_release(&eng->glob_pool, &eng->cur.glob_bt);
    for (int i = 0; i < K; i++) {
        if (gemma4_engine_decode(eng, (int32_t)(100 + i), NULL) != 0) {
            eng->decode_graph_failed = saved_failed; return;
        }
    }
    cudaStreamSynchronize(eng->stream);

    int maxq = GEMMA4_HEADS * GEMMA4_GLOBAL_HEAD_DIM;
    float *hQ = (float *)malloc((size_t)maxq * sizeof(float));
    float *hA = (float *)malloc((size_t)maxq * sizeof(float));
    float *hB = (float *)malloc((size_t)maxq * sizeof(float));
    float worst = 0.0f;
    size_t shmem = 32 * sizeof(float);

    for (int L = 0; L < eng->cfg.n_layers; L++) {   // every layer, both classes
        layer_type_t want = eng->layer_types[L];
        int n_heads    = GEMMA4_HEADS;
        int n_kv_heads = (want == LAYER_SLIDING) ? GEMMA4_KV_HEADS : GEMMA4_GLOBAL_KV_HEADS;
        int head_dim   = (want == LAYER_SLIDING) ? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM;
        int qn = n_heads * head_dim;
        for (int i = 0; i < qn; i++) hQ[i] = 0.2f * sinf(0.01f * (float)(i * 3 + L * 7 + 1));
        cudaMemcpy(eng->d_attn_q, hQ, (size_t)qn * sizeof(float), cudaMemcpyHostToDevice);

        if (want == LAYER_SLIDING) {
            size_t cls = (size_t)eng->sliding_kv_capacity * slid_elems;
            sliding_attn_decode_broadcast(eng, eng->d_attn_out, eng->d_attn_q,
                eng->d_sliding_k + (size_t)L*cls, eng->d_sliding_v + (size_t)L*cls,
                n_heads, n_kv_heads, head_dim, GEMMA4_SLIDING_WINDOW, K, eng->stream);
            cudaStreamSynchronize(eng->stream);
            cudaMemcpy(hA, eng->d_attn_out, (size_t)qn*sizeof(float), cudaMemcpyDeviceToHost);
            size_t pls = (size_t)eng->slid_pool.n_blocks * BT * slid_elems;
            paged_attn_decode_kernel<<<n_heads, head_dim, shmem, eng->stream>>>(
                eng->d_attn_out, eng->d_attn_q,
                eng->d_slid_pool_k + (size_t)L*pls, eng->d_slid_pool_v + (size_t)L*pls,
                eng->cur.d_slid_blocks, eng->cur.slid_bt.base, eng->cur.slid_bt.n,
                n_heads, n_kv_heads, head_dim, K, GEMMA4_SLIDING_WINDOW, BT, slid_elems);
        } else {
            int slot = eng->global_slot[L];
            size_t cls = (size_t)eng->global_kv_capacity * glob_elems;
            global_attn_decode_broadcast(eng, eng->d_attn_out, eng->d_attn_q,
                eng->d_global_k + (size_t)slot*cls, eng->d_global_v + (size_t)slot*cls,
                n_heads, head_dim, K, eng->stream);
            cudaStreamSynchronize(eng->stream);
            cudaMemcpy(hA, eng->d_attn_out, (size_t)qn*sizeof(float), cudaMemcpyDeviceToHost);
            size_t pls = (size_t)eng->glob_pool.n_blocks * BT * glob_elems;
            paged_attn_decode_kernel<<<n_heads, head_dim, shmem, eng->stream>>>(
                eng->d_attn_out, eng->d_attn_q,
                eng->d_glob_pool_k + (size_t)slot*pls, eng->d_glob_pool_v + (size_t)slot*pls,
                eng->cur.d_glob_blocks, eng->cur.glob_bt.base, eng->cur.glob_bt.n,
                n_heads, GEMMA4_GLOBAL_KV_HEADS, head_dim, K, 0, BT, glob_elems);
        }
        cudaStreamSynchronize(eng->stream);
        cudaMemcpy(hB, eng->d_attn_out, (size_t)qn*sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < qn; i++) { float d = fabsf(hA[i] - hB[i]); if (d > worst) worst = d; }
    }
    free(hQ); free(hA); free(hB);

    if (worst < 2e-2f)
        fprintf(stderr, "fucina: paged READ self-test PASSED — paged attn == contiguous attn (max abs diff %.3g)\n", worst);
    else
        fprintf(stderr, "fucina: paged READ self-test FAILED — max abs diff %.3g\n", worst);

    gemma4_engine_reset(eng);
    paged_table_release(&eng->slid_pool, &eng->cur.slid_bt);
    paged_table_release(&eng->glob_pool, &eng->cur.glob_bt);
    eng->decode_graph_failed = saved_failed;
}

// End-to-end proof that the paged read DRIVES generation correctly: feed the
// SAME fixed token run through decode twice — once reading the contiguous cache,
// once reading the paged pool — building identical KV both times, and compare the
// argmax (next-token prediction) at each step. Fixed input (not autoregressive
// argmax) avoids a flawed bit-exact-cascade comparison: the paged read uses a
// different summation order than the split-K contiguous path, so per-step logits
// differ at the fp-noise level (~1e-6, see the READ self-test) and feeding argmax
// back would amplify a single near-tie flip into total divergence. With fixed
// input the predictions must agree (a tiny number of genuine top-2 near-ties is
// tolerated). Gated by FUCINA_PAGED_E2E_SELFTEST.
static void gemma4_engine_paged_e2e_selftest(gemma4_engine_t *eng) {
    if (!eng->paged_enabled) return;
    const int K = 64;
    int saved_failed = eng->decode_graph_failed;
    eng->decode_graph_failed = 1;                 // mirror + paged read live on the scalar path
    float *logits = (float *)malloc((size_t)GEMMA4_VOCAB_SIZE * sizeof(float));
    int seqA[64], seqB[64];

    for (int run = 0; run < 2; run++) {
        eng->paged_read = run;                    // 0 = contiguous read, 1 = paged read
        gemma4_engine_reset(eng);
        paged_table_release(&eng->slid_pool, &eng->cur.slid_bt);
        paged_table_release(&eng->glob_pool, &eng->cur.glob_bt);
        for (int i = 0; i < K; i++) {
            if (gemma4_engine_decode(eng, (int32_t)(100 + i), logits) != 0) {  // SAME fixed input
                free(logits); eng->paged_read = 0; eng->decode_graph_failed = saved_failed; return;
            }
            (run == 0 ? seqA : seqB)[i] = gemma4_sample_argmax(logits, GEMMA4_VOCAB_SIZE);
        }
    }
    eng->paged_read = 0;
    free(logits);

    int mism = 0, first = -1;
    for (int i = 0; i < K; i++) if (seqA[i] != seqB[i]) { mism++; if (first < 0) first = i; }
    if (mism == 0)
        fprintf(stderr, "fucina: paged E2E self-test PASSED — %d-step next-token prediction identical "
                        "(paged read vs contiguous)\n", K);
    else
        // BIT-IDENTITY EXPECTED: both the paged and the contiguous read now run the
        // SAME split-K flash decode (paged just resolves K/V through the block table),
        // matching the contiguous summation order exactly. Any mismatch is a real
        // regression in that equivalence, not a benign near-tie flip — flag it loudly.
        fprintf(stderr, "fucina: paged E2E self-test FAILED — split-K paged read should be "
                        "BIT-IDENTICAL to contiguous, but only %d/%d agree (%d mismatch(es), "
                        "first step %d)\n", K - mism, K, mism, first);

    gemma4_engine_reset(eng);
    paged_table_release(&eng->slid_pool, &eng->cur.slid_bt);
    paged_table_release(&eng->glob_pool, &eng->cur.glob_bt);
    eng->decode_graph_failed = saved_failed;
}

// =========================================================================
// ─── Batched Decode (one weight pass over K tokens) ─────────────────────
// =========================================================================

// Lazily allocate the engine-resident batched-decode scratch (sized for the max
// draft width GEMMA4_SPEC_MAX), so decode_batched does no per-call cudaMalloc. d_sb
// order: tok, x, norm, inf, q, k, v, attn, o, gate, up, logitsK.
static int ensure_spec_scratch(gemma4_engine_t *eng)
{
    if (eng->sb_ready) return 0;
    const size_t M = GEMMA4_SPEC_MAX;
    const size_t H = eng->cfg.hidden_size, I = eng->cfg.intermediate, V = GEMMA4_VOCAB_SIZE;
    const size_t OQ = (size_t)eng->cfg.n_heads*GEMMA4_GLOBAL_HEAD_DIM;
    const size_t OKV = (size_t)eng->cfg.n_kv_sliding*GEMMA4_HEAD_DIM;
    size_t elems[12] = { M, M*H, M*H, M*I, M*OQ, M*OKV, M*OKV, M*OQ, M*H, M*I, M*I, M*V };
    for (int i = 0; i < 12; i++) {
        if (cudaMalloc(&eng->d_sb[i], elems[i]*sizeof(float)) != cudaSuccess) {
            for (int j = 0; j < i; j++) { cudaFree(eng->d_sb[j]); eng->d_sb[j] = NULL; }
            cudaGetLastError();
            return -1;
        }
    }
    // GPU-side verify scratch (a): K ids + K draws. d_sample_id is also used by the
    // cold/no-draft single-row device sample — allocate it here too (load_assistant
    // also allocates it, but lookup-only spec has no assistant).
    if (!eng->d_sample_id)
        if (cudaMalloc(&eng->d_sample_id, sizeof(int)) != cudaSuccess) { cudaGetLastError(); return -1; }
    if (!eng->d_spec_ids)
        if (cudaMalloc(&eng->d_spec_ids, M*sizeof(int)) != cudaSuccess) { cudaGetLastError(); return -1; }
    if (!eng->d_spec_rnd)
        if (cudaMalloc(&eng->d_spec_rnd, M*sizeof(float)) != cudaSuccess) { cudaGetLastError(); return -1; }
    // NVFP4 batched-verify transposed-activation scratch Xt[in_dim][K], in_dim ≤ INTERMEDIATE.
    // Pre-allocated once so the captured verify transposes into it with no alloc/host-sync.
    if (!eng->d_specxt)
        if (cudaMalloc(&eng->d_specxt, M*I*sizeof(float)) != cudaSuccess) { cudaGetLastError(); return -1; }
    eng->sb_ready = 1;
    return 0;
}

// Forward K tokens [pos .. pos+K-1] CONTINUING the current sequence, reading each
// weight matrix ONCE for all K (the projection/FFN GEMMs are the 12.65 GB/token
// bandwidth cost). This is the engine primitive that makes speculative decoding
// pay off: K sequential decodes read the weights K times; this reads them once,
// so an accepted draft of length K costs ~one token's bandwidth. K must be small
// (≤ GEMMA4_SPEC_MAX). Attention is done per token against the live cache (reusing
// the single-query flash decode kernels), so causality is exact. Writes
// logits_out[K × VOCAB] (row i = logits AFTER token i) and advances the cache by K.
// Returns 0 on success, -2 if it must defer to sequential decode (LoRA active),
// -1 on error. Math matches gemma4_engine_decode token-for-token.
// Internal: keep_dev=1 computes the K logit rows into d_sb[11] (device) WITHOUT the
// D2H copy, so the GPU-side spec verify (a) can sample them on-device. The public
// wrapper below passes keep_dev=0 (original behavior: D2H iff logits_out!=NULL).
// The K-row batched forward [g,draft...], extracted so it can be issued BOTH per-kernel
// (d_pos==NULL: host pos, exact-fit attention grids — original behaviour) and under CUDA-
// graph capture (d_pos!=NULL: pos read from device d_specpos[0]=pos / [1]=pos+1, attention
// at fixed MAX grids that tail-return, so one capture replays at any position). want_head
// runs the output-norm + LM head + softcap(+suppress) into d_sb[11]. Issues launches ONLY
// on `stream`; advances no engine state (caller owns d_tok H2D, global_n_tokens, timing,
// D2H, sampling). Bit-identical to the old inline body when d_pos==NULL.
static void decode_batched_forward(
    gemma4_engine_t *eng, int K, int pos, cudaStream_t stream,
    const int *d_pos, int want_head,
    const uint32_t *d_anc = nullptr,   // TREE: [K] per-row ancestor bitmask (device; treebase derived)
    const int *d_depth = nullptr)      // TREE: [K] per-row depth (RoPE pos offset)
{
    const int H   = eng->cfg.hidden_size;
    const int I   = eng->cfg.intermediate;
    const int HD2 = 32 * sizeof(float);
    const int HEADS = eng->cfg.n_heads;
    auto grid1d = [](size_t n){ return (unsigned)((n + 255) / 256); };

    int32_t *d_tok  = (int32_t*)eng->d_sb[0];
    float *d_x   = eng->d_sb[1],  *d_norm = eng->d_sb[2], *d_inf = eng->d_sb[3];
    float *d_q   = eng->d_sb[4],  *d_k    = eng->d_sb[5], *d_v   = eng->d_sb[6];
    float *d_attn= eng->d_sb[7],  *d_o    = eng->d_sb[8], *d_gate= eng->d_sb[9];
    float *d_up  = eng->d_sb[10], *d_logitsK = eng->d_sb[11];
    const int *d_ntok = d_pos ? d_pos + 1 : NULL;   // attention n_tokens0 = pos+1 (+row)

    // d_tok is pre-filled by the caller (graph: outside capture; per-kernel: just before).
    embed_w(eng, d_x, weight_fp8(eng, eng->tensors.token_embd), d_tok, K, H, stream);
    scale_kernel<<<grid1d((size_t)K*H),256,0,stream>>>(d_x, K*H, sqrtf((float)H));

    for (int l = 0; l < eng->cfg.n_layers; l++) {
        layer_type_t lt = eng->layer_types[l];
        int hd  = (lt==LAYER_SLIDING)? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM;
        int nkv = (lt==LAYER_SLIDING)? eng->cfg.n_kv_sliding : eng->cfg.n_kv_global;
        int oq  = HEADS*hd, okv = nkv*hd;
        const float *w_attn   = eng->d_w_attn_norm      + (size_t)l*H;
        const float *w_post_a = eng->d_w_post_attn_norm + (size_t)l*H;
        const float *w_ffn    = eng->d_w_ffn_norm       + (size_t)l*H;
        const float *w_post_f = eng->d_w_post_ffn_norm  + (size_t)l*H;
        const float *w_qn     = eng->d_w_q_norm + (size_t)l*GEMMA4_GLOBAL_HEAD_DIM;
        const float *w_kn     = eng->d_w_k_norm + (size_t)l*GEMMA4_GLOBAL_HEAD_DIM;
        const __typeof__(eng->tensors.layers[0]) *L = &eng->tensors.layers[l];

        // Pre-attn norm (fp32) → Q,K,V batched GEMV (Q8_0 read once, reused over K).
        // For NVFP4 there is no batched FP4 GEMV: the fused decode kernel is B=1, so loop the K
        // rows (the row-major d_inf/d_q/d_k/d_v scratch is [K][dim], one GEMV per row). Attention,
        // RoPE and norms below are weight-free and stay batched. (Reached only via the chunked
        // suffix-prefill / spec-verify paths; the public single-token decode uses decode_layer.)
        const bool nvfp4 = (eng->format == FORMAT_NVFP4 && eng->nvfp4_decode_ready);
        // Batched NVFP4 GEMV (weight read once for all K) when K fits the dispatch (≤ NVFP4_GEMV_KMAX);
        // larger K (rare; SPEC_MAX=16 > KMAX) falls back to the per-row loop. nvb()/nvb_proj wrap that.
        const bool nvfp4_b = nvfp4 && (K <= NVFP4_GEMV_KMAX);
        auto nvb_proj = [&](int pj, const float *xin, float *yout, int idim, int odim){
            if (nvfp4_b) nvfp4_decode_proj_batched(eng, l, pj, xin, yout, idim, odim, K, stream);
            else for (int r=0;r<K;r++) nvfp4_decode_proj(eng, l, pj, xin+(size_t)r*idim, yout+(size_t)r*odim, idim, odim, stream);
        };
        rms_norm_rows_kernel<<<K,256,HD2,stream>>>(d_inf, d_x, w_attn, H, K, GEMMA4_RMS_EPS);
        if (nvfp4) {
            nvb_proj(PJ_Q, d_inf, d_q, H, oq);
            per_head_rms_norm_rows_kernel<<<dim3(HEADS,K),hd,HD2,stream>>>(d_q, w_qn, HEADS, hd, K, GEMMA4_RMS_EPS);
            nvb_proj(PJ_K, d_inf, d_k, H, okv);
            if (lt == LAYER_SLIDING) {
                nvb_proj(PJ_V, d_inf, d_v, H, okv);
                per_head_rms_norm_rows_kernel<<<dim3(nkv,K),hd,HD2,stream>>>(d_v, NULL, nkv, hd, K, GEMMA4_RMS_EPS);
            } else {
                cudaMemcpyAsync(d_v, d_k, (size_t)K*okv*sizeof(float), cudaMemcpyDeviceToDevice, stream);
                per_head_rms_norm_rows_kernel<<<dim3(nkv,K),hd,HD2,stream>>>(d_v, NULL, nkv, hd, K, GEMMA4_RMS_EPS);
            }
            per_head_rms_norm_rows_kernel<<<dim3(nkv,K),hd,HD2,stream>>>(d_k, w_kn, nkv, hd, K, GEMMA4_RMS_EPS);
        } else {
        gemv_batched_w(eng, d_q, weight_fp8(eng, L->attn_q), d_inf, H, oq, K, stream, (int)L->fmt_q);
        per_head_rms_norm_rows_kernel<<<dim3(HEADS,K),hd,HD2,stream>>>(d_q, w_qn, HEADS, hd, K, GEMMA4_RMS_EPS);
        gemv_batched_w(eng, d_k, weight_fp8(eng, L->attn_k), d_inf, H, okv, K, stream, (int)L->fmt_k);
        if (lt == LAYER_SLIDING) {
            gemv_batched_w(eng, d_v, weight_fp8(eng, L->attn_v), d_inf, H, okv, K, stream, (int)L->fmt_v);
            per_head_rms_norm_rows_kernel<<<dim3(nkv,K),hd,HD2,stream>>>(d_v, NULL, nkv, hd, K, GEMMA4_RMS_EPS);
        } else {
            cudaMemcpyAsync(d_v, d_k, (size_t)K*okv*sizeof(float), cudaMemcpyDeviceToDevice, stream);
            per_head_rms_norm_rows_kernel<<<dim3(nkv,K),hd,HD2,stream>>>(d_v, NULL, nkv, hd, K, GEMMA4_RMS_EPS);
        }
        per_head_rms_norm_rows_kernel<<<dim3(nkv,K),hd,HD2,stream>>>(d_k, w_kn, nkv, hd, K, GEMMA4_RMS_EPS);
        } // else (non-NVFP4 q/k/v batched)

        if (lt == LAYER_SLIDING)
            rope_rows_kernel<<<dim3(HEADS,K),hd/2,0,stream>>>(d_q, d_k, pos, HEADS, nkv, hd, K, 10000.0f, NULL, d_pos, d_depth);
        else
            rope_rows_kernel<<<dim3(HEADS,K),hd/2,0,stream>>>(d_q, d_k, pos, HEADS, nkv, hd, K, 1000000.0f, eng->d_rope_freqs, d_pos, d_depth);

        // Attention, all K rows in ONE causal launch (row-batched): first scatter the
        // K rows' K/V to their absolute positions pos..pos+K-1 (one write launch —
        // row i's reads are bounded by n_tokens0+i, so later rows' entries are never
        // visible to earlier rows), then run every row's split-K attention in a single
        // grid (blockIdx.y = row) and merge per-row partials in a single combine.
        // Replaces 3K serialized launches/layer with 3; each row's split partition and
        // merge order replicate its old per-row launch exactly (bit-identical output).
        // Graph (d_pos) path uses MAX-extent split grids (tail-return) so the grid is
        // position-independent — same trick as decode_layer's single-token graph path.
        if (lt == LAYER_SLIDING) {
            // FLAT (Step 3): spec row i is the token at absolute position pos+i.
            size_t lstride = (size_t)eng->sliding_kv_capacity * okv;
            kv_t *kc = eng->d_sliding_k + (size_t)l*lstride;
            kv_t *vc = eng->d_sliding_v + (size_t)l*lstride;
            kv_write_sliding_kernel<<<dim3(grid1d(okv),K),256,0,stream>>>(
                kc, vc, d_k, d_v, pos, 0, K, okv, GEMMA4_SLIDING_WINDOW,
                eng->sliding_kv_capacity, d_pos);
            int max_splits;
            if (d_pos) {
                max_splits = (GEMMA4_SLIDING_WINDOW + GEMMA4_SLIDING_SPLIT_CHUNK - 1) / GEMMA4_SLIDING_SPLIT_CHUNK;
            } else {
                int max_len = pos + K; if (max_len > GEMMA4_SLIDING_WINDOW) max_len = GEMMA4_SLIDING_WINDOW;
                max_splits = (max_len + GEMMA4_SLIDING_SPLIT_CHUNK - 1) / GEMMA4_SLIDING_SPLIT_CHUNK;
                if (max_splits < 1) max_splits = 1;
                if (max_splits > GEMMA4_GLOBAL_MAX_SPLITS) max_splits = GEMMA4_GLOBAL_MAX_SPLITS;
            }
            if (eng->cfg.n_heads == 32 && eng->cfg.n_kv_sliding == 16) {
                sliding_attn_splitk_rows_kernel<32, 16, GEMMA4_HEAD_DIM>
                    <<<dim3(max_splits, K), 16*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, d_q, kc, vc,
                        GEMMA4_SLIDING_WINDOW, pos + 1, eng->sliding_kv_capacity, d_ntok, d_anc);
                flash_decode_combine_rows_kernel<32><<<dim3(HEADS, K), hd, 0, stream>>>(
                    d_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                    hd, GEMMA4_SLIDING_WINDOW, pos + 1, d_ntok, oq);
            } else {
            sliding_attn_splitk_rows_kernel<GEMMA4_HEADS, GEMMA4_KV_HEADS, GEMMA4_HEAD_DIM>
                <<<dim3(max_splits, K), GEMMA4_KV_HEADS*32, 0, stream>>>(
                    eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, d_q, kc, vc,
                    GEMMA4_SLIDING_WINDOW, pos + 1, eng->sliding_kv_capacity, d_ntok, d_anc);
            flash_decode_combine_rows_kernel<GEMMA4_HEADS><<<dim3(HEADS, K), hd, 0, stream>>>(
                d_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                hd, GEMMA4_SLIDING_WINDOW, pos + 1, d_ntok, oq);
            }
        } else {
            int slot = eng->global_slot[l];
            // Per-token global cache stride = okv (nkv*hd). 12B: nkv=1 ⇒ okv==hd ⇒ identical.
            size_t lstride = (size_t)eng->global_kv_capacity * okv;
            kv_t *kc = eng->d_global_k + (size_t)slot*lstride;
            kv_t *vc = eng->d_global_v + (size_t)slot*lstride;
            kv_write_global_kernel<<<dim3(grid1d(okv),K),256,0,stream>>>(
                kc, vc, d_k, d_v, pos, K, okv, d_pos);
            int max_splits;
            if (d_pos) {
                max_splits = GEMMA4_GLOBAL_MAX_SPLITS;
            } else {
                max_splits = (pos + K + GEMMA4_GLOBAL_SPLIT_CHUNK - 1) / GEMMA4_GLOBAL_SPLIT_CHUNK;
                if (max_splits < 1) max_splits = 1;
                if (max_splits > GEMMA4_GLOBAL_MAX_SPLITS) max_splits = GEMMA4_GLOBAL_MAX_SPLITS;
            }
            if (eng->cfg.n_heads == 32 && eng->cfg.n_kv_global == 4) {
                global_attn_splitk_rows_kernel<32, 4, GEMMA4_GLOBAL_HEAD_DIM>
                    <<<dim3(max_splits, K), 32*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, d_q, kc, vc, pos + 1, d_ntok, d_anc);
                flash_decode_combine_rows_kernel<32><<<dim3(HEADS, K), hd, 0, stream>>>(
                    d_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                    hd, /*window=*/0, pos + 1, d_ntok, oq);
            } else {
            global_attn_splitk_rows_kernel<GEMMA4_HEADS, 1, GEMMA4_GLOBAL_HEAD_DIM>
                <<<dim3(max_splits, K), GEMMA4_HEADS*32, 0, stream>>>(
                    eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, d_q, kc, vc, pos + 1, d_ntok, d_anc);
            flash_decode_combine_rows_kernel<GEMMA4_HEADS><<<dim3(HEADS, K), hd, 0, stream>>>(
                d_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                hd, /*window=*/0, pos + 1, d_ntok, oq);
            }
        }

        // O projection (input d_attn is already fp32) → d_o; post-attn norm; residual.
        if (nvfp4) nvb_proj(PJ_O, d_attn, d_o, oq, H);
        else gemv_batched_w(eng, d_o, weight_fp8(eng, L->attn_output), d_attn, oq, H, K, stream, (int)L->fmt_o);
        rms_norm_rows_kernel<<<K,256,HD2,stream>>>(d_norm, d_o, w_post_a, H, K, GEMMA4_RMS_EPS);
        residual_add_kernel<<<grid1d((size_t)K*H),256,0,stream>>>(d_x, d_norm, K*H);

        // FFN (fp32 batched GEMV throughout).
        rms_norm_rows_kernel<<<K,256,HD2,stream>>>(d_inf, d_x, w_ffn, H, K, GEMMA4_RMS_EPS);
        if (nvfp4) {
            nvb_proj(PJ_GATE, d_inf, d_gate, H, I);
            nvb_proj(PJ_UP,   d_inf, d_up,   H, I);
        } else {
            gemv_batched_w(eng, d_gate, weight_fp8(eng, L->ffn_gate), d_inf, H, I, K, stream, (int)L->fmt_gate);
            gemv_batched_w(eng, d_up,   weight_fp8(eng, L->ffn_up),   d_inf, H, I, K, stream, (int)L->fmt_up);
        }
        geglu_kernel<<<grid1d((size_t)K*I),256,0,stream>>>(d_inf, d_gate, d_up, K*I);
        if (nvfp4) nvb_proj(PJ_DOWN, d_inf, d_o, I, H);
        else gemv_batched_w(eng, d_o, weight_fp8(eng, L->ffn_down), d_inf, I, H, K, stream, (int)L->fmt_down);
        rms_norm_rows_kernel<<<K,256,HD2,stream>>>(d_norm, d_o, w_post_f, H, K, GEMMA4_RMS_EPS);
        residual_add_kernel<<<grid1d((size_t)K*H),256,0,stream>>>(d_x, d_norm, K*H);
        if (eng->h_out_scale[l] != 1.0f)
            scale_kernel<<<grid1d((size_t)K*H),256,0,stream>>>(d_x, K*H, eng->h_out_scale[l]);
    }

    // Output norm (batched) + LM head as ONE batched GEMV (tied LM head read once for all
    // K) + softcap + suppress. Result stays in d_sb[11] (d_logitsK); the caller D2Hs it
    // (per-kernel public path) or samples it on-device (keep_dev / graph path).
    if (want_head) {
        int vocab = GEMMA4_VOCAB_SIZE;
        rms_norm_rows_kernel<<<K,256,HD2,stream>>>(d_norm, d_x, eng->d_w_out_norm, H, K, GEMMA4_RMS_EPS);
        if (eng->format == FORMAT_NVFP4) {
            if (K <= NVFP4_GEMV_KMAX) {
                // Batched head: the head read ONCE for all K rows (transpose d_norm[K][H] →
                // d_specxt[H][K], then weight-read-once batched head GEMV). Output token-major.
                // FP8 head (gated) reads 1 B/elem — halves the per-K-batch head bandwidth vs BF16.
                nvfp4_xT_launch(eng->d_specxt, d_norm, H, K, stream);
                if (eng->d_lmhead_fp8)
                    fp8_head_gemv_batched_launch(d_logitsK, eng->d_lmhead_fp8, eng->d_lmhead_fp8_scale,
                                                 eng->d_specxt, H, vocab, K, stream);
                else
                    bf16_head_gemv_batched_launch(d_logitsK, eng->d_lmhead_bf16, eng->d_specxt,
                                                  H, vocab, K, stream);
            } else {
                for (int r=0;r<K;r++)
                    if (eng->d_lmhead_fp8)
                        fp8_head_gemv_launch(d_logitsK+(size_t)r*vocab, eng->d_lmhead_fp8, eng->d_lmhead_fp8_scale, d_norm+(size_t)r*H, H, vocab, stream);
                    else
                        bf16_head_gemv_launch(d_logitsK+(size_t)r*vocab, eng->d_lmhead_bf16, d_norm+(size_t)r*H, H, vocab, stream);
            }
        } else
        gemv_batched_w(eng, d_logitsK, lmhead_w(eng),
                       d_norm, H, vocab, K, stream, embd_fmt(eng));
        logit_softcap_kernel<<<grid1d((size_t)K*vocab),256,0,stream>>>(d_logitsK, GEMMA4_SOFTCAP, K*vocab);
        if (eng->n_suppress > 0)
            for (int i = 0; i < K; i++)
                suppress_tokens_kernel<<<grid1d(eng->n_suppress),256,0,stream>>>(
                    d_logitsK + (size_t)i*vocab, eng->d_suppress, eng->n_suppress, vocab);
    }
}

// Lazy one-time capture of the K-row spec-verify forward (one graph PER K, since grids
// depend on K). Replays via cudaGraphLaunch with device-resident pos (d_specpos) + the K
// input ids (d_sb[0]) refreshed per step — collapsing ~670 per-step launches to one. Same
// pattern + escape hatch as decode_graph_ensure. Any failure permanently disables the
// batched graph (per-kernel path stays in service).
static int batched_graph_ensure(gemma4_engine_t *eng, int K)
{
    if (K < 1 || K > GEMMA4_SPEC_MAX) return -1;
    if (eng->batched_graph[K]) return 0;
    if (eng->batched_graph_failed) return -1;
    static const int disabled = (getenv("FUCINA_NO_BATCHED_GRAPH") != NULL);
    if (disabled) { eng->batched_graph_failed = 1; return -1; }
    if (ensure_spec_scratch(eng) != 0) { eng->batched_graph_failed = 1; return -1; }
    if (!eng->d_specpos &&
        cudaMalloc(&eng->d_specpos, 2*sizeof(int)) != cudaSuccess) {
        eng->d_specpos = NULL; eng->batched_graph_failed = 1; cudaGetLastError(); return -1;
    }
    cudaMemset(eng->d_specpos, 0, 2*sizeof(int));
    cudaStream_t cs = NULL;
    cudaGraph_t  g  = NULL;
    int ok = cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) == cudaSuccess;
    if (ok && cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
        decode_batched_forward(eng, K, /*pos=*/0, cs, /*d_pos=*/eng->d_specpos, /*want_head=*/1);
        ok = cudaStreamEndCapture(cs, &g) == cudaSuccess && g != NULL;
    } else {
        ok = 0;
    }
    if (ok) ok = cudaGraphInstantiate(&eng->batched_graph[K], g, 0) == cudaSuccess;
    if (g)  cudaGraphDestroy(g);
    if (cs) cudaStreamDestroy(cs);
    if (!ok || !eng->batched_graph[K]) {
        eng->batched_graph[K] = NULL;
        eng->batched_graph_failed = 1;
        cudaGetLastError();
        fprintf(stderr, "fucina: batched spec-verify graph capture failed (K=%d) — per-kernel launches\n", K);
        return -1;
    }
    fprintf(stderr, "fucina: batched spec-verify CUDA graph captured (K=%d)\n", K);
    return 0;
}

static int decode_batched_dev(
    gemma4_engine_t *eng, const int32_t *tokens, int K, float *logits_out, int keep_dev)
{
    if (!eng->loaded || K <= 0) return -1;
    if (K > GEMMA4_SPEC_MAX) return -1;

    cudaStream_t stream = eng->stream;
    const int pos = eng->cur.n_tokens;            // captured; advanced only at end
    int32_t *d_tok = (int32_t*)eng->d_sb[0];

    // Engine-resident scratch (allocated once, sized for GEMMA4_SPEC_MAX rows), so
    // repeated/probe calls pay no per-call cudaMalloc/free. All fp32: the batched
    // GEMV reads Q8_0 directly (no BF16 dequant), K tokens cost ~one token's weight BW.
    if (ensure_spec_scratch(eng) != 0) {
        return -1;
    }

    // ── Graph fast path (spec-verify keep_dev only) ──────────────────────────────────
    // The K=1 single-token decode already graphs its forward; this graphs the K>1 verify
    // forward (the dominant decode cost during speculation). ~670 launches/step → one
    // replay. keep_dev leaves logits on-device (d_sb[11]); the caller samples + syncs.
    if (keep_dev && batched_graph_ensure(eng, K) == 0) {
        cudaMemcpyAsync(d_tok, tokens, (size_t)K*sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
        int pv[2] = { pos, pos + 1 };
        cudaMemcpyAsync(eng->d_specpos, pv, sizeof(pv), cudaMemcpyHostToDevice, stream);
        if (cudaGraphLaunch(eng->batched_graph[K], stream) == cudaSuccess) {
            eng->cur.n_tokens += K;
            return 0;
        }
        // Replay failed: retire the graph and fall through to per-kernel launches.
        cudaGetLastError();
        cudaGraphExecDestroy(eng->batched_graph[K]);
        eng->batched_graph[K] = NULL;
        eng->batched_graph_failed = 1;
        fprintf(stderr, "fucina: batched graph replay failed — using per-kernel launches\n");
    }

    // ── Per-kernel path (public path, interior prefill chunks, or graph fallback) ─────
    // Timing: events only on the public (keep_dev=0) path. The spec-verify caller
    // (keep_dev=1) syncs the stream itself right after sampling — its old per-step
    // cudaEventSynchronize here was a SECOND full host-blocking drain per verify
    // step that also serialized the sampler launch behind the forward; the caller
    // accumulates decode_ms with a host clock around its own sync instead.
    // The keep_dev=0 path uses the LAZY pair (decode_timing_lap): recorded here,
    // harvested on a later call — no per-call drain (llama.cpp-audit #30).
    decode_timing_lap(eng);
    int timing = !keep_dev && !eng->ev_pending;
    if (timing) cudaEventRecord(eng->ev_start, stream);

    cudaMemcpyAsync(d_tok, tokens, (size_t)K*sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    decode_batched_forward(eng, K, pos, stream, /*d_pos=*/NULL,
                           /*want_head=*/(logits_out || keep_dev));

    // logits_out==NULL (interior chunked-prefill chunks) skips the D2H; keep_dev leaves
    // the K rows in d_sb[11] for on-GPU verify.
    if (logits_out)
        cudaMemcpyAsync(logits_out, eng->d_sb[11],
                        (size_t)K*GEMMA4_VOCAB_SIZE*sizeof(float),
                        cudaMemcpyDeviceToHost, stream);

    eng->cur.n_tokens += K;

    if (timing) {
        cudaEventRecord(eng->ev_stop, stream);
        eng->ev_pending = 1;
        eng->ev_pending_tokens = K;
    }
    // The pageable-host logits copy must complete before the caller reads it; the
    // keep_dev path and the interior (logits_out == NULL) suffix-prefill chunks skip
    // the drain entirely.
    if (logits_out) cudaStreamSynchronize(stream);
    // events are engine-resident (Step 4) — not destroyed here.

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "fucina: decode_batched CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// Public wrapper (unchanged contract): D2H the K logit rows iff logits_out!=NULL.
int gemma4_engine_decode_batched(
    gemma4_engine_t *eng, const int32_t *tokens, int K, float *logits_out)
{
    return decode_batched_dev(eng, tokens, K, logits_out, /*keep_dev=*/0);
}

// d_sb[11] device logits accessor for the GPU-side verify (a).
static inline float *decode_batched_dev_logits(gemma4_engine_t *eng) {
    return eng->d_sb[11];
}

// ─── Tree spec data structures (used by the verify forward AND the drafter) ──
#define GEMMA4_TREE_MAX 16   // ≤ GEMMA4_SPEC_MAX: tree nodes reuse the K-row verify buffers
struct spec_tree {
    int    n;                         // node count (node 0 = root = committed g)
    int    parent[GEMMA4_TREE_MAX];   // parent[0] = -1
    int    depth [GEMMA4_TREE_MAX];   // pos offset from the committed position (root = 0)
    int    nchild[GEMMA4_TREE_MAX];   // children drafted under this node (head top-w)
    int32_t tok  [GEMMA4_TREE_MAX];   // token at the node (filled by the drafter)
    uint32_t anc [GEMMA4_TREE_MAX];   // bit c set iff node c is on the root→node path (incl self)
};

// Tree spec-verify forward: forwards the t->n tree-node tokens in ONE target weight pass under
// the ancestor mask. Node r writes KV to slot pos+r, is RoPE'd at pos+depth[r], and attends the
// committed prefix [0,pos) + its ancestor tree slots. Leaves t->n logit rows on-device (d_sb[11];
// row r = node r's next-token distribution) and advances eng->cur.n_tokens by t->n (the caller
// rewinds after picking the accepted path).
//
// GRAPHED (T2.2): the per-step measurement showed the ungraphed forward at ~232ms/step (vs ~80ms
// for the linear graphed verify) — ~660 raw launches/step were the bottleneck, NOT the drafter.
// We capture one graph per K (node count), reading pos from d_specpos and the mask/depth/tokens
// from device buffers updated before each replay (treebase is derived in-kernel, so the graph is
// pos-independent — same trick as batched_graph). Tree shape varies but only the buffer CONTENTS
// change, not the graph, so a single per-K graph serves every tree of that size.
static uint32_t *g_d_anc = NULL;   // [GEMMA4_TREE_MAX] device ancestor masks (graph reads by ptr)
static int      *g_d_depth = NULL; // [GEMMA4_TREE_MAX] device depths
static cudaGraphExec_t g_tree_graph[GEMMA4_SPEC_MAX + 1] = {0};
static int g_tree_graph_failed = 0;

static int tree_graph_ensure(gemma4_engine_t *eng, int K) {
    if (K < 1 || K > GEMMA4_SPEC_MAX) return -1;
    if (g_tree_graph[K]) return 0;
    if (g_tree_graph_failed) return -1;
    if (getenv("FUCINA_NO_TREE_GRAPH")) { g_tree_graph_failed = 1; return -1; }
    if (ensure_spec_scratch(eng) != 0) { g_tree_graph_failed = 1; return -1; }
    if (!eng->d_specpos && cudaMalloc(&eng->d_specpos, 2*sizeof(int)) != cudaSuccess) {
        eng->d_specpos = NULL; g_tree_graph_failed = 1; cudaGetLastError(); return -1; }
    if (!g_d_anc   && cudaMalloc(&g_d_anc,   GEMMA4_TREE_MAX*sizeof(uint32_t)) != cudaSuccess) { cudaGetLastError(); g_tree_graph_failed=1; return -1; }
    if (!g_d_depth && cudaMalloc(&g_d_depth, GEMMA4_TREE_MAX*sizeof(int))      != cudaSuccess) { cudaGetLastError(); g_tree_graph_failed=1; return -1; }
    cudaMemset(eng->d_specpos, 0, 2*sizeof(int));
    cudaStream_t cs = NULL; cudaGraph_t g = NULL;
    int ok = cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) == cudaSuccess;
    if (ok && cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
        decode_batched_forward(eng, K, /*pos=*/0, cs, /*d_pos=*/eng->d_specpos, /*want_head=*/1,
                               g_d_anc, g_d_depth);
        ok = cudaStreamEndCapture(cs, &g) == cudaSuccess && g != NULL;
    } else ok = 0;
    if (ok) ok = cudaGraphInstantiate(&g_tree_graph[K], g, 0) == cudaSuccess;
    if (g)  cudaGraphDestroy(g);
    if (cs) cudaStreamDestroy(cs);
    if (!ok || !g_tree_graph[K]) { g_tree_graph[K] = NULL; g_tree_graph_failed = 1; cudaGetLastError();
        fprintf(stderr, "fucina: tree spec-verify graph capture failed (K=%d) — per-kernel launches\n", K); return -1; }
    fprintf(stderr, "fucina: tree spec-verify CUDA graph captured (K=%d)\n", K);
    return 0;
}

static int decode_tree_dev(gemma4_engine_t *eng, const spec_tree *t) {
    if (!eng->loaded || t->n <= 0 || t->n > GEMMA4_SPEC_MAX) return -1;
    if (ensure_spec_scratch(eng) != 0) return -1;
    cudaStream_t stream = eng->stream;
    const int pos = eng->cur.n_tokens;
    int32_t *d_tok = (int32_t*)eng->d_sb[0];
    if (!g_d_anc   && cudaMalloc(&g_d_anc,   GEMMA4_TREE_MAX*sizeof(uint32_t)) != cudaSuccess) { cudaGetLastError(); return -1; }
    if (!g_d_depth && cudaMalloc(&g_d_depth, GEMMA4_TREE_MAX*sizeof(int))      != cudaSuccess) { cudaGetLastError(); return -1; }
    int32_t htok[GEMMA4_TREE_MAX]; uint32_t hanc[GEMMA4_TREE_MAX]; int hdep[GEMMA4_TREE_MAX];
    for (int i = 0; i < t->n; i++) { htok[i] = t->tok[i]; hanc[i] = t->anc[i]; hdep[i] = t->depth[i]; }
    cudaMemcpyAsync(d_tok,    htok, (size_t)t->n*sizeof(int32_t),  cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_d_anc,  hanc, (size_t)t->n*sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_d_depth,hdep, (size_t)t->n*sizeof(int),      cudaMemcpyHostToDevice, stream);
    // Graph fast path: pos via d_specpos, mask/depth/tokens via the device buffers just updated.
    if (tree_graph_ensure(eng, t->n) == 0) {
        int pv[2] = { pos, pos + 1 };
        cudaMemcpyAsync(eng->d_specpos, pv, sizeof(pv), cudaMemcpyHostToDevice, stream);
        if (cudaGraphLaunch(g_tree_graph[t->n], stream) == cudaSuccess) { eng->cur.n_tokens += t->n; return 0; }
        cudaGetLastError();   // replay failed → per-kernel fallback
    }
    decode_batched_forward(eng, t->n, pos, stream, /*d_pos=*/NULL, /*want_head=*/1, g_d_anc, g_d_depth);
    eng->cur.n_tokens += t->n;
    return 0;
}

// Commit an accepted tree path: move its scattered KV (slot pos+path[d]) into the contiguous
// slots pos+d for every layer (K and V). No-op when the path is already contiguous (path[d]==d,
// e.g. the width-1 trunk) — so the linear case pays nothing. path[0]=root(=0); d runs 1..L.
// Safe in-place: src slot (pos+path[d]) ≥ dst (pos+d) and strictly above every earlier dst,
// so no source is clobbered before it is read (nodes are created parent-before-child).
static void tree_commit_kv(gemma4_engine_t *eng, int pos, const int *path, int L) {
    cudaStream_t stream = eng->stream;
    for (int d = 1; d <= L; d++) {
        int src = pos + path[d], dst = pos + d;
        if (src == dst) continue;
        for (int l = 0; l < eng->cfg.n_layers; l++) {
            layer_type_t lt = eng->layer_types[l];
            int hd  = (lt==LAYER_SLIDING)? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM;
            int nkv = (lt==LAYER_SLIDING)? eng->cfg.n_kv_sliding : eng->cfg.n_kv_global;
            size_t okv = (size_t)nkv * hd;
            if (lt == LAYER_SLIDING) {
                size_t cap = eng->sliding_kv_capacity, base = (size_t)l * cap * okv;
                size_t s = base + (size_t)(src % (int)cap) * okv, t = base + (size_t)(dst % (int)cap) * okv;
                cudaMemcpyAsync(eng->d_sliding_k + t, eng->d_sliding_k + s, okv*sizeof(kv_t), cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(eng->d_sliding_v + t, eng->d_sliding_v + s, okv*sizeof(kv_t), cudaMemcpyDeviceToDevice, stream);
            } else {
                int slot = eng->global_slot[l]; size_t cap = eng->global_kv_capacity;
                size_t base = (size_t)slot * cap * okv;
                size_t s = base + (size_t)src * okv, t = base + (size_t)dst * okv;
                cudaMemcpyAsync(eng->d_global_k + t, eng->d_global_k + s, okv*sizeof(kv_t), cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(eng->d_global_v + t, eng->d_global_v + s, okv*sizeof(kv_t), cudaMemcpyDeviceToDevice, stream);
            }
        }
    }
}

// =========================================================================
// ─── Multi-sequence batched decode (Phase 3: B INDEPENDENT sequences) ────
// =========================================================================
// One batched forward over B independent sequences, each with its own absolute
// position and its own paged KV (per-seq block tables into the shared pools).
// Reuses decode_batched_forward's GEMV/norm/FFN machinery (the K rows are now
// independent sequences instead of consecutive positions of one sequence), but:
//   - RoPE uses each row's own absolute position (rope_rows_pos_kernel),
//   - KV mirror-writes scatter each row into ITS OWN block table (paged_kv_write),
//   - attention reads each row through ITS OWN block table (split-K paged kernels:
//     paged_{sliding,global}_attn_splitk_batched + paged_flash_decode_combine_batched).
// Requires paged mode. The contiguous d_sliding_*/d_global_* caches are NOT touched
// (they belong to the single-seq eng->cur path); the batch lives entirely in the pool.

extern "C" void gemma4_engine_seq_remove(gemma4_engine_t *eng, int slot);  // fwd decl

// Grow/recycle one slot's paged block tables to cover logical position `pos`, then
// refresh its device block-id arrays. Mirror of paged_seq_sync for an explicit slot.
static int paged_slot_sync(gemma4_engine_t *eng, gemma4_seq *s, int pos) {
    if (!eng->paged_enabled) return -1;
    if (paged_table_ensure(&eng->slid_pool, &s->slid_bt, pos + 1) != 0) return -1;
    paged_table_advance_sliding(&eng->slid_pool, &s->slid_bt, pos + 1, GEMMA4_SLIDING_WINDOW);
    if (glob_table_ensure(eng, s, pos + 1) != 0) return -1;
    if (paged_upload_blocks(&s->slid_bt, &s->d_slid_blocks, &s->d_slid_cap, eng->stream) != 0) return -1;
    if (paged_upload_blocks(&s->glob_bt, &s->d_glob_blocks, &s->d_glob_cap, eng->stream) != 0) return -1;
    return 0;
}

// Per-row multiseq sampler (defined with the rest of the sampling code below the
// spec loop; decode_multiseq_forward launches it). Greedy (temp<=0) is a
// lowest-index argmax matching argmax_rows_kernel; temp>0 uses sample_logit_row.
__global__ void sample_logits_ms_kernel(
    const float *logits, int V,
    const float *temps, const int *top_ks, const float *top_ps, const float *min_ps,
    const float *rnds, int *out_ids);

// Lazily allocate the multi-seq device scratch (positions, views, sampled ids).
static int ensure_ms_scratch(gemma4_engine_t *eng) {
    if (eng->ms_ready) return 0;
    const int N = GEMMA4_MAX_SEQS;
    int ok = cudaMalloc(&eng->d_ms_pos,    (size_t)N*sizeof(int)) == cudaSuccess
          && cudaMalloc(&eng->d_ms_outtok, (size_t)N*sizeof(int)) == cudaSuccess
          && cudaMalloc(&eng->d_ms_views_slid, (size_t)N*sizeof(PagedSeqView)) == cudaSuccess
          && cudaMalloc(&eng->d_ms_views_glob, (size_t)N*sizeof(PagedSeqView)) == cudaSuccess
          && cudaMalloc(&eng->d_ms_temp, (size_t)N*sizeof(float)) == cudaSuccess
          && cudaMalloc(&eng->d_ms_topk, (size_t)N*sizeof(int))   == cudaSuccess
          && cudaMalloc(&eng->d_ms_topp, (size_t)N*sizeof(float)) == cudaSuccess
          && cudaMalloc(&eng->d_ms_minp, (size_t)N*sizeof(float)) == cudaSuccess
          && cudaMalloc(&eng->d_ms_rnd,  (size_t)N*sizeof(float)) == cudaSuccess;
    if (!ok) {
        if (eng->d_ms_pos)    { cudaFree(eng->d_ms_pos);    eng->d_ms_pos = NULL; }
        if (eng->d_ms_outtok) { cudaFree(eng->d_ms_outtok); eng->d_ms_outtok = NULL; }
        if (eng->d_ms_views_slid) { cudaFree(eng->d_ms_views_slid); eng->d_ms_views_slid = NULL; }
        if (eng->d_ms_views_glob) { cudaFree(eng->d_ms_views_glob); eng->d_ms_views_glob = NULL; }
        if (eng->d_ms_temp) { cudaFree(eng->d_ms_temp); eng->d_ms_temp = NULL; }
        if (eng->d_ms_topk) { cudaFree(eng->d_ms_topk); eng->d_ms_topk = NULL; }
        if (eng->d_ms_topp) { cudaFree(eng->d_ms_topp); eng->d_ms_topp = NULL; }
        if (eng->d_ms_minp) { cudaFree(eng->d_ms_minp); eng->d_ms_minp = NULL; }
        if (eng->d_ms_rnd)  { cudaFree(eng->d_ms_rnd);  eng->d_ms_rnd  = NULL; }
        cudaGetLastError();
        return -1;
    }
    eng->ms_ready = 1;
    return 0;
}

// Kernel-launch-only body of the multi-seq forward over B rows. Reads ALL per-step
// varying inputs from DEVICE buffers (d_sb[0] tokens, d_ms_pos positions, d_ms_views_*
// per-row block tables/lengths) so it is CUDA-graph-capturable: the caller refreshes
// those device buffers each step OUTSIDE the capture and replays this body. `max_splits`
// fixes the attention split grids; pass the largest the batch can need (per-row tail-
// return makes any value >= a row's n_splits bit-identical). When greedy (want_argmax)
// it appends the argmax_rows_kernel so the whole step is one graph. Issues launches only
// on `stream`; advances no engine state. Always launches the full output head.
static void decode_multiseq_body(
    gemma4_engine_t *eng, int B, int max_splits, int want_argmax, cudaStream_t stream)
{
    const int H = eng->cfg.hidden_size, I = eng->cfg.intermediate, HEADS = eng->cfg.n_heads;
    const int HD2 = 32 * sizeof(float);
    auto grid1d = [](size_t n){ return (unsigned)((n + 255) / 256); };

    int32_t *d_tok = (int32_t*)eng->d_sb[0];
    float *d_x  = eng->d_sb[1],  *d_norm = eng->d_sb[2], *d_inf = eng->d_sb[3];
    float *d_q  = eng->d_sb[4],  *d_k    = eng->d_sb[5], *d_v   = eng->d_sb[6];
    float *d_attn = eng->d_sb[7], *d_o   = eng->d_sb[8], *d_gate= eng->d_sb[9];
    float *d_up = eng->d_sb[10], *d_logitsK = eng->d_sb[11];

    const int qwen3 = 0;
    // Embed (per-row token, device d_tok) + Gemma √H scale. Qwen3 does NOT scale embeddings.
    embed_w(eng, d_x, weight_fp8(eng, eng->tensors.token_embd), d_tok, B, H, stream);
    if (!qwen3)
        scale_kernel<<<grid1d((size_t)B*H),256,0,stream>>>(d_x, B*H, sqrtf((float)H));

    for (int l = 0; l < eng->cfg.n_layers; l++) {
        layer_type_t lt = eng->layer_types[l];
        int hd  = qwen3 ? eng->cfg.head_dim
                        : ((lt==LAYER_SLIDING)? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM);
        int nkv = (lt==LAYER_SLIDING)? eng->cfg.n_kv_sliding : eng->cfg.n_kv_global;
        int oq  = HEADS*hd, okv = nkv*hd;
        const float *w_attn   = eng->d_w_attn_norm      + (size_t)l*H;
        const float *w_post_a = eng->d_w_post_attn_norm + (size_t)l*H;
        const float *w_ffn    = eng->d_w_ffn_norm       + (size_t)l*H;
        const float *w_post_f = eng->d_w_post_ffn_norm  + (size_t)l*H;
        const float *w_qn     = eng->d_w_q_norm + (size_t)l*GEMMA4_GLOBAL_HEAD_DIM;
        const float *w_kn     = eng->d_w_k_norm + (size_t)l*GEMMA4_GLOBAL_HEAD_DIM;
        const __typeof__(eng->tensors.layers[0]) *L = &eng->tensors.layers[l];

        rms_norm_rows_kernel<<<B,256,HD2,stream>>>(d_inf, d_x, w_attn, H, B, GEMMA4_RMS_EPS);
        gemv_batched_w(eng, d_q, weight_fp8(eng, L->attn_q), d_inf, H, oq, B, stream, (int)L->fmt_q);
        per_head_rms_norm_rows_kernel<<<dim3(HEADS,B),hd,HD2,stream>>>(d_q, w_qn, HEADS, hd, B, GEMMA4_RMS_EPS);
        // Attention scale: Gemma bakes it into the constant q_norm/k_norm weights (kernel scale=1).
        // Qwen3's q_norm/k_norm are LEARNED (no baked scale) → apply 1/sqrt(head_dim) to Q here.
        if (qwen3)
            scale_kernel<<<grid1d((size_t)B*oq),256,0,stream>>>(d_q, B*oq, 1.0f/sqrtf((float)hd));
        gemv_batched_w(eng, d_k, weight_fp8(eng, L->attn_k), d_inf, H, okv, B, stream, (int)L->fmt_k);
        if (qwen3) {
            // Qwen3: separate V projection on EVERY layer, NO V norm.
            gemv_batched_w(eng, d_v, weight_fp8(eng, L->attn_v), d_inf, H, okv, B, stream, (int)L->fmt_v);
        } else if (lt == LAYER_SLIDING) {
            gemv_batched_w(eng, d_v, weight_fp8(eng, L->attn_v), d_inf, H, okv, B, stream, (int)L->fmt_v);
            per_head_rms_norm_rows_kernel<<<dim3(nkv,B),hd,HD2,stream>>>(d_v, NULL, nkv, hd, B, GEMMA4_RMS_EPS);
        } else {
            cudaMemcpyAsync(d_v, d_k, (size_t)B*okv*sizeof(float), cudaMemcpyDeviceToDevice, stream);
            per_head_rms_norm_rows_kernel<<<dim3(nkv,B),hd,HD2,stream>>>(d_v, NULL, nkv, hd, B, GEMMA4_RMS_EPS);
        }
        per_head_rms_norm_rows_kernel<<<dim3(nkv,B),hd,HD2,stream>>>(d_k, w_kn, nkv, hd, B, GEMMA4_RMS_EPS);

        // RoPE with each row's OWN absolute position.
        if (lt == LAYER_SLIDING)
            rope_rows_pos_kernel<<<dim3(HEADS,B),hd/2,0,stream>>>(d_q, d_k, eng->d_ms_pos, HEADS, nkv, hd, B, 10000.0f, NULL);
        else
            rope_rows_pos_kernel<<<dim3(HEADS,B),hd/2,0,stream>>>(d_q, d_k, eng->d_ms_pos, HEADS, nkv, hd, B, 1000000.0f, eng->d_rope_freqs);

        // Mirror K/V write + paged attention, both through per-row block tables.
        PagedSeqView *views = (lt==LAYER_SLIDING)? eng->d_ms_views_slid : eng->d_ms_views_glob;
        PagedWriteBatch wb; wb.seqs = views; wb.write_pos = eng->d_ms_pos; wb.n_rows = B;
        size_t pls; pkv_t *pk, *pv;
        if (lt == LAYER_SLIDING) {
            pls = (size_t)eng->slid_pool.n_blocks * PAGED_KV_BLOCK_TOKENS * okv;
            pk = eng->d_slid_pool_k + (size_t)l * pls;
            pv = eng->d_slid_pool_v + (size_t)l * pls;
        } else {
            int slot = eng->global_slot[l];
            pls = (size_t)eng->glob_pool.n_blocks * PAGED_KV_BLOCK_TOKENS * okv;
            pk = eng->d_glob_pool_k + (size_t)slot * pls;
            pv = eng->d_glob_pool_v + (size_t)slot * pls;
        }
        paged_kv_write<<<dim3(grid1d(okv),B),256,0,stream>>>(
            pk, pv, d_k, d_v, wb, PAGED_KV_BLOCK_TOKENS, okv);
        int window = (lt==LAYER_SLIDING) ? GEMMA4_SLIDING_WINDOW : 0;
        // Split-K paged attention (bit-identical to the contiguous split-K path:
        // same per-row n_splits/per/scan order/combine order, only the K/V address
        // is resolved through the block table). Per-(split,seq) partials into the
        // shared d_fa_acc/m/l scratch (slot seq*MAX_SPLITS+split), merged by the
        // paged combine. Replaces the sequential per-(head,seq) online-softmax.
        // Launch grid.x = max_splits split blocks (caller passes the largest the
        // batch can need). Each row still tail-returns past its own n_splits, so the
        // per-row partition/scan/combine order is unchanged and the result stays
        // bit-identical for ANY max_splits >= a row's n_splits — that fixed grid is
        // what lets one captured graph replay across steps at any position.
        int g_splits = max_splits;
        if (g_splits < 1) g_splits = 1;
        if (g_splits > GEMMA4_GLOBAL_MAX_SPLITS) g_splits = GEMMA4_GLOBAL_MAX_SPLITS;
        // 31B (n_heads=32, n_kv_sliding=16, n_kv_global=4) vs 12B template dispatch —
        // mirrors the n_heads==32 branches at every other attention call site. Without
        // this the paged batch path runs the 12B <16,8>/<16,1> kernels on the 31B and
        // silently drops half the heads (the global kernel is now GQA-aware via NKV).
        if (lt == LAYER_SLIDING) {
            if (HEADS == 32 && nkv == 16) {
                paged_sliding_attn_splitk_batched<32, 16, GEMMA4_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(g_splits, B), 16*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, d_q, pk, pv, views,
                        window, GEMMA4_SLIDING_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, okv);
                paged_flash_decode_combine_batched<32, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(HEADS, B), hd, 0, stream>>>(
                        d_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, views,
                        hd, window, GEMMA4_SLIDING_SPLIT_CHUNK);
            } else {
                paged_sliding_attn_splitk_batched<GEMMA4_HEADS, GEMMA4_KV_HEADS,
                                                  GEMMA4_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(g_splits, B), GEMMA4_KV_HEADS*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, d_q, pk, pv, views,
                        window, GEMMA4_SLIDING_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, okv);
                paged_flash_decode_combine_batched<GEMMA4_HEADS, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(HEADS, B), hd, 0, stream>>>(
                        d_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, views,
                        hd, window, GEMMA4_SLIDING_SPLIT_CHUNK);
            }
        } else {
            if (qwen3 && HEADS == 32 && nkv == 8) {
                // Qwen3 dense full-causal GQA: 32 query / 8 KV heads, head_dim 128.
                paged_global_attn_splitk_batched<32, 8, 128, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(g_splits, B), 32*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, d_q, pk, pv, views,
                        GEMMA4_GLOBAL_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, okv);
                paged_flash_decode_combine_batched<32, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(HEADS, B), hd, 0, stream>>>(
                        d_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, views,
                        hd, /*window=*/0, GEMMA4_GLOBAL_SPLIT_CHUNK);
            } else if (qwen3 && HEADS == 32 && nkv == 4) {
                // Qwen3-MoE full-causal GQA: 32 query / 4 KV heads, head_dim 128 (NOT the Gemma
                // 31B <32,4,512> below — same head/KV counts but head_dim differs).
                paged_global_attn_splitk_batched<32, 4, 128, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(g_splits, B), 32*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, d_q, pk, pv, views,
                        GEMMA4_GLOBAL_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, okv);
                paged_flash_decode_combine_batched<32, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(HEADS, B), hd, 0, stream>>>(
                        d_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, views,
                        hd, /*window=*/0, GEMMA4_GLOBAL_SPLIT_CHUNK);
            } else if (HEADS == 32 && nkv == 4) {
                paged_global_attn_splitk_batched<32, 4, GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(g_splits, B), 32*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, d_q, pk, pv, views,
                        GEMMA4_GLOBAL_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, okv);
                paged_flash_decode_combine_batched<32, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(HEADS, B), hd, 0, stream>>>(
                        d_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, views,
                        hd, /*window=*/0, GEMMA4_GLOBAL_SPLIT_CHUNK);
            } else {
                paged_global_attn_splitk_batched<GEMMA4_HEADS, GEMMA4_GLOBAL_KV_HEADS,
                                                 GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(g_splits, B), GEMMA4_HEADS*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, d_q, pk, pv, views,
                        GEMMA4_GLOBAL_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, okv);
                paged_flash_decode_combine_batched<GEMMA4_HEADS, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(HEADS, B), hd, 0, stream>>>(
                        d_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, views,
                        hd, /*window=*/0, GEMMA4_GLOBAL_SPLIT_CHUNK);
            }
        }

        // O proj → (post-attn sandwich norm) → residual; FFN; (post-FFN norm) → residual; out scale.
        // Qwen3 is standard pre-norm: NO sandwich norms — the attn/ffn outputs add to the residual
        // directly. Gemma applies a post_attn_norm / post_ffn_norm before each residual add.
        gemv_batched_w(eng, d_o, weight_fp8(eng, L->attn_output), d_attn, oq, H, B, stream, (int)L->fmt_o);
        if (qwen3) {
            residual_add_kernel<<<grid1d((size_t)B*H),256,0,stream>>>(d_x, d_o, B*H);
        } else {
            rms_norm_rows_kernel<<<B,256,HD2,stream>>>(d_norm, d_o, w_post_a, H, B, GEMMA4_RMS_EPS);
            residual_add_kernel<<<grid1d((size_t)B*H),256,0,stream>>>(d_x, d_norm, B*H);
        }

        rms_norm_rows_kernel<<<B,256,HD2,stream>>>(d_inf, d_x, w_ffn, H, B, GEMMA4_RMS_EPS);
        {
        gemv_batched_w(eng, d_gate, weight_fp8(eng, L->ffn_gate), d_inf, H, I, B, stream, (int)L->fmt_gate);
        gemv_batched_w(eng, d_up,   weight_fp8(eng, L->ffn_up),   d_inf, H, I, B, stream, (int)L->fmt_up);
        if (qwen3) silu_glu_kernel<<<grid1d((size_t)B*I),256,0,stream>>>(d_inf, d_gate, d_up, B*I);
        else       geglu_kernel  <<<grid1d((size_t)B*I),256,0,stream>>>(d_inf, d_gate, d_up, B*I);
        gemv_batched_w(eng, d_o, weight_fp8(eng, L->ffn_down), d_inf, I, H, B, stream, (int)L->fmt_down);
        if (qwen3) {
            residual_add_kernel<<<grid1d((size_t)B*H),256,0,stream>>>(d_x, d_o, B*H);
        } else {
            rms_norm_rows_kernel<<<B,256,HD2,stream>>>(d_norm, d_o, w_post_f, H, B, GEMMA4_RMS_EPS);
            residual_add_kernel<<<grid1d((size_t)B*H),256,0,stream>>>(d_x, d_norm, B*H);
        }
        }
        if (eng->h_out_scale[l] != 1.0f)
            scale_kernel<<<grid1d((size_t)B*H),256,0,stream>>>(d_x, B*H, eng->h_out_scale[l]);
    }

    // Output norm + LM head (batched) + softcap (+suppress) → d_logitsK [B][VOCAB].
    int vocab = eng->cfg.vocab_size;
    rms_norm_rows_kernel<<<B,256,HD2,stream>>>(d_norm, d_x, eng->d_w_out_norm, H, B, GEMMA4_RMS_EPS);
    gemv_batched_w(eng, d_logitsK, lmhead_w(eng), d_norm, H, vocab, B, stream, embd_fmt(eng));
    if (eng->cfg.softcap > 0.0f)
        logit_softcap_kernel<<<grid1d((size_t)B*vocab),256,0,stream>>>(d_logitsK, eng->cfg.softcap, B*vocab);
    if (eng->n_suppress > 0)
        for (int i = 0; i < B; i++)
            suppress_tokens_kernel<<<grid1d(eng->n_suppress),256,0,stream>>>(
                d_logitsK + (size_t)i*vocab, eng->d_suppress, eng->n_suppress, vocab);

    // Greedy argmax (graph-capturable: reads only d_logitsK). The per-row
    // temperature sampler stays in the host wrapper (off the graph path).
    if (want_argmax)
        argmax_rows_kernel<<<B,1024,0,stream>>>(d_logitsK, eng->d_ms_outtok, B, vocab);
}

// Lazy one-time capture of the multi-seq forward at batch size B (one graph PER B,
// grids depend on B). Replays via cudaGraphLaunch with device-resident per-row
// positions (d_ms_pos), per-row views (d_ms_views_*) and tokens (d_sb[0]) refreshed
// each step OUTSIDE the capture — collapsing the ~B-row, 48-layer launch storm to one
// replay. Attention launches at the FULL split grid (every position fits). Greedy only
// (argmax appended); temperature rows fall back to the per-kernel host path. Same escape
// hatch as batched_graph_ensure. FUCINA_NO_BATCHED_GRAPH disables.
static int multiseq_graph_ensure(gemma4_engine_t *eng, int B)
{
    if (B < 1 || B > GEMMA4_MAX_SEQS) return -1;
    if (eng->multiseq_graph[B]) return 0;
    if (eng->multiseq_graph_failed) return -1;
    static const int disabled = (getenv("FUCINA_NO_BATCHED_GRAPH") != NULL);
    if (disabled) { eng->multiseq_graph_failed = 1; return -1; }
    if (ensure_spec_scratch(eng) != 0 || ensure_ms_scratch(eng) != 0) {
        eng->multiseq_graph_failed = 1; return -1;
    }
    cudaStream_t cs = NULL;
    cudaGraph_t  g  = NULL;
    int ok = cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) == cudaSuccess;
    if (ok && cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
        decode_multiseq_body(eng, B, GEMMA4_GLOBAL_MAX_SPLITS, /*want_argmax=*/1, cs);
        ok = cudaStreamEndCapture(cs, &g) == cudaSuccess && g != NULL;
    } else {
        ok = 0;
    }
    if (ok) ok = cudaGraphInstantiate(&eng->multiseq_graph[B], g, 0) == cudaSuccess;
    if (g)  cudaGraphDestroy(g);
    if (cs) cudaStreamDestroy(cs);
    if (!ok || !eng->multiseq_graph[B]) {
        eng->multiseq_graph[B] = NULL;
        eng->multiseq_graph_failed = 1;
        cudaGetLastError();
        fprintf(stderr, "fucina: multiseq batch graph capture failed (B=%d) — per-kernel launches\n", B);
        return -1;
    }
    if (!(eng->multiseq_graph_logged & (1ULL << B))) {
        fprintf(stderr, "fucina: multiseq batch graph captured (B=%d)\n", B);
        eng->multiseq_graph_logged |= (1ULL << B);
    }
    return 0;
}

// Refresh the per-step varying DEVICE inputs (tokens, positions, per-row paged views)
// for a multi-seq forward over B rows on `stream`. Shared by the graph-replay and
// per-kernel paths so both feed bit-identical device state. Returns the longest row's
// position+1 (for the per-kernel split-grid sizing). slv/in_tok/positions are host.
static int multiseq_upload_inputs(
    gemma4_engine_t *eng, gemma4_seq **slv, const int32_t *in_tok,
    const int *positions, int B, cudaStream_t stream)
{
    int32_t *d_tok = (int32_t*)eng->d_sb[0];
    cudaMemcpyAsync(d_tok, in_tok, (size_t)B*sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(eng->d_ms_pos, positions, (size_t)B*sizeof(int), cudaMemcpyHostToDevice, stream);
    PagedSeqView hvs[GEMMA4_MAX_SEQS], hvg[GEMMA4_MAX_SEQS];
    int max_len = 0;
    for (int r = 0; r < B; r++) {
        gemma4_seq *s = slv[r];
        int np = positions[r] + 1;                 // tokens visible after this write (incl. new)
        if (np > max_len) max_len = np;
        hvs[r].block_table = s->d_slid_blocks; hvs[r].n_blocks = s->slid_bt.n;
        hvs[r].base = s->slid_bt.base; hvs[r].n_tokens = np;
        hvg[r].block_table = s->d_glob_blocks; hvg[r].n_blocks = s->glob_bt.n;
        hvg[r].base = s->glob_bt.base; hvg[r].n_tokens = np;
    }
    cudaMemcpyAsync(eng->d_ms_views_slid, hvs, (size_t)B*sizeof(PagedSeqView), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(eng->d_ms_views_glob, hvg, (size_t)B*sizeof(PagedSeqView), cudaMemcpyHostToDevice, stream);
    return max_len;
}

// Run ONE batched forward over the B slots given by `slv[]` (each slot's KV tables
// already synced to its CURRENT position; in_tok[] are the B input tokens to feed,
// positions[] the B absolute positions of those tokens). Leaves the B logit rows in
// d_sb[11]; if want_sample, also writes per-row sampled ids into eng->d_ms_outtok.
// All B rows reuse the SPEC_MAX-sized batched scratch (B <= GEMMA4_MAX_SEQS).
//
// Fast path: if every row is greedy (temp<=0) the captured per-B CUDA graph replays
// the whole forward + argmax in one launch (device-resident positions/views/tokens
// refreshed just above). Any temperature row, capture failure, or FUCINA_NO_BATCHED_GRAPH
// falls back to the per-kernel body (which also runs the per-row temperature sampler).
static int decode_multiseq_forward(
    gemma4_engine_t *eng, gemma4_seq **slv, const int32_t *in_tok,
    const int *positions, int B, int want_sample, const int *rng_off)
{
    if (!eng->paged_enabled || B <= 0 || B > GEMMA4_MAX_SEQS) return -1;
    if (ensure_spec_scratch(eng) != 0 || ensure_ms_scratch(eng) != 0) return -1;
    cudaStream_t stream = eng->stream;
    const int vocab = eng->cfg.vocab_size;

    int any_sample = 0;
    for (int r = 0; r < B; r++) if (slv[r]->samp_temp > 0.0f) { any_sample = 1; break; }

    // Always refresh the device-resident per-step inputs first (shared by both paths).
    int max_len = multiseq_upload_inputs(eng, slv, in_tok, positions, B, stream);

    // ── Graph fast path (greedy batches only) ────────────────────────────────────
    if (want_sample && !any_sample && multiseq_graph_ensure(eng, B) == 0) {
        if (cudaGraphLaunch(eng->multiseq_graph[B], stream) == cudaSuccess)
            return 0;
        // Replay failed: retire the graph and fall through to per-kernel launches.
        cudaGetLastError();
        cudaGraphExecDestroy(eng->multiseq_graph[B]);
        eng->multiseq_graph[B] = NULL;
        eng->multiseq_graph_failed = 1;
        fprintf(stderr, "fucina: multiseq batch graph replay failed — using per-kernel launches\n");
    }

    // ── Per-kernel path ──────────────────────────────────────────────────────────
    // Size the split grid to the LONGEST row (saves split blocks at short contexts);
    // greedy argmax folded into the body, temperature sampler dispatched after.
    int max_splits;
    {   // worst case over both classes: global chunk over the full max_len.
        int gs = (max_len + GEMMA4_GLOBAL_SPLIT_CHUNK - 1) / GEMMA4_GLOBAL_SPLIT_CHUNK;
        if (gs < 1) gs = 1;
        if (gs > GEMMA4_GLOBAL_MAX_SPLITS) gs = GEMMA4_GLOBAL_MAX_SPLITS;
        max_splits = gs;
    }
    int want_argmax = (want_sample && !any_sample);
    decode_multiseq_body(eng, B, max_splits, want_argmax, stream);

    if (want_sample && any_sample) {
        float *d_logitsK = eng->d_sb[11];
        float h_temp[GEMMA4_MAX_SEQS], h_topp[GEMMA4_MAX_SEQS], h_minp[GEMMA4_MAX_SEQS], h_rnd[GEMMA4_MAX_SEQS];
        int   h_topk[GEMMA4_MAX_SEQS];
        for (int r = 0; r < B; r++) {
            gemma4_seq *s = slv[r];
            h_temp[r] = s->samp_temp;
            h_topk[r] = s->samp_top_k;
            h_topp[r] = s->samp_top_p;
            h_minp[r] = s->samp_min_p;
            // splitmix64(seed, index): reproducible per (seed, token ordinal),
            // independent of batch position so two runs match regardless of
            // how rows are grouped into steps. rng_off[r] is the row's sampled-
            // ordinal offset within this seq's burst (0 for a plain decode row or
            // the spec anchor; 1+j for the j-th draft/verify row), so every verify
            // position of a slot draws an INDEPENDENT uniform — making sampled spec
            // distributionally identical to (and bit-reproducible with) plain
            // per-token sampling. NULL ⇒ all-zero ⇒ unchanged for non-spec callers.
            int roff = rng_off ? rng_off[r] : 0;
            uint64_t z = (s->samp_seed ? s->samp_seed : 0x9e3779b97f4a7c15ULL) + 0x9e3779b97f4a7c15ULL * (s->n_sampled + 1 + roff);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            z =  z ^ (z >> 31);
            h_rnd[r] = (float)((z >> 11) * (1.0 / 9007199254740992.0));
        }
        cudaMemcpyAsync(eng->d_ms_temp, h_temp, (size_t)B*sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(eng->d_ms_topk, h_topk, (size_t)B*sizeof(int),   cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(eng->d_ms_topp, h_topp, (size_t)B*sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(eng->d_ms_minp, h_minp, (size_t)B*sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(eng->d_ms_rnd,  h_rnd,  (size_t)B*sizeof(float), cudaMemcpyHostToDevice, stream);
        sample_logits_ms_kernel<<<B,1024,0,stream>>>(
            d_logitsK, vocab, eng->d_ms_temp, eng->d_ms_topk, eng->d_ms_topp,
            eng->d_ms_minp, eng->d_ms_rnd, eng->d_ms_outtok);
    }
    return 0;
}

// ── Multi-sequence continuous-batching C ABI ────────────────────────────────
// All require paged mode (FUCINA_PAGED_KV). slots[] are opaque ids 0..MAX_SEQS-1.

// Allocate a free slot, prefill `prompt` into THAT slot's paged KV (correctness
// first: loop single-token multi-seq forwards over the slot alone), sample the
// first token greedily, write it to *first_token_out. Returns slot id (>=0) or -1.
// Fast suffix prefill for the cross-request prefix cache. After adopting a shared
// prefix (s->n_tokens == base), prefill the divergent suffix [base, base+M) in
// BATCHED chunks of up to GEMMA4_MAX_SEQS rows — one weight pass per chunk —
// instead of token-by-token. Each chunk is a decode_multiseq_forward over the SAME
// seq at consecutive positions: row j writes its K/V then attends [0, base+j+1) via
// its per-row PagedSeqView.n_tokens bound, so it sees the adopted prefix (pool) plus
// the earlier suffix rows written THIS pass — exactly the ragged spec-verify
// causality, hence bit-identical to the token-by-token path. Samples the first
// generated token on the final row. Returns 0 / -1.
// ── Qwen3.5 (qwen35) M4 batched continuous-batching entry points (defined far below,
//    after the M2/M3 mixer kernels they build on). The extern "C" continuous-batching
//    ABI routes to these for GEMMA4_ARCH_QWEN3_5 (its hybrid GDN/full state does not fit
//    the paged fp8 KV pool — it carries its own fp32 GDN/conv/KV arenas). ──
static int qwen35_seq_add(gemma4_engine_t *eng, const int32_t *prompt, int n_prompt,
                          int32_t *first_token_out, float temp, int top_k, float top_p,
                          float min_p, uint64_t seed);
static int qwen35_seq_add_multiseq(gemma4_engine_t *eng, const int32_t *tokens_flat,
                          const int *lens, int M, const float *temps, const int *topks,
                          const float *topps, const float *minps, const uint64_t *seeds,
                          int *out_slots, int32_t *out_first);
static int qwen35_step_batch(gemma4_engine_t *eng, const int *slots,
                             const int32_t *in_tokens, int B, int32_t *out_tokens);
static int qwen35_seq_open(gemma4_engine_t *eng, float temp, int top_k, float top_p,
                           float min_p, uint64_t seed);
static int qwen35_seq_prefill_chunk(gemma4_engine_t *eng, int slot, const int32_t *tokens,
                                    int n, int do_sample, int32_t *first_token_out);

static int prefill_suffix_batched(gemma4_engine_t *eng, gemma4_seq *s,
                                  const int32_t *suffix, int M, int do_sample, int32_t *first_tok_out)
{
    const int CH = GEMMA4_MAX_SEQS;
    gemma4_seq *rows[GEMMA4_MAX_SEQS];
    int positions[GEMMA4_MAX_SEQS];
    int32_t last_tok = 0;
    int done = 0;
    while (done < M) {
        int C = M - done; if (C > CH) C = CH;
        int base = s->n_tokens;
        if (paged_slot_sync(eng, s, base + C - 1) != 0) return -1;   // grow block tables for this chunk
        for (int j = 0; j < C; j++) { rows[j] = s; positions[j] = base + j; }
        int want_sample = (do_sample && done + C >= M);             // sample only the final row, only if asked
        if (decode_multiseq_forward(eng, rows, suffix + done, positions, C, want_sample, NULL) != 0)
            return -1;
        if (want_sample)
            cudaMemcpyAsync(&last_tok, eng->d_ms_outtok + (C - 1), sizeof(int32_t),
                            cudaMemcpyDeviceToHost, eng->stream);
        s->n_tokens = base + C;
        done += C;
    }
    cudaStreamSynchronize(eng->stream);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (do_sample) {
        s->n_sampled++;   // one token produced (RNG index advances once), matching token-by-token
        if (first_tok_out) *first_tok_out = last_tok;
    }
    return 0;
}

extern "C" int gemma4_engine_seq_add(
    gemma4_engine_t *eng, const int32_t *prompt, int n_prompt, int32_t *first_token_out,
    float temp, int top_k, float top_p, float min_p, uint64_t seed)
{
    if (eng && eng->loaded && eng->cfg.arch == GEMMA4_ARCH_QWEN3_5)
        return qwen35_seq_add(eng, prompt, n_prompt, first_token_out, temp, top_k, top_p, min_p, seed);
    if (!eng || !eng->loaded || !eng->paged_enabled || !prompt || n_prompt <= 0) return -1;
    int slot = -1;
    for (int i = 0; i < GEMMA4_MAX_SEQS; i++) if (!eng->slots[i].used) { slot = i; break; }
    if (slot < 0) return -1;
    if (ensure_spec_scratch(eng) != 0 || ensure_ms_scratch(eng) != 0) return -1;

    gemma4_seq *s = &eng->slots[slot];
    // Fresh tables (struct was zeroed at create / released on remove).
    s->n_tokens = 0; s->slid_bt.base = 0; s->slid_bt.n = 0; s->glob_bt.base = 0; s->glob_bt.n = 0;
    s->used = 1;
    // Per-sequence sampling params (applied on-device every sampled token).
    s->samp_temp = temp; s->samp_top_k = top_k; s->samp_top_p = top_p;
    s->samp_min_p = min_p; s->samp_seed = seed; s->n_sampled = 0;
    s->mtp_h_valid = 0;   // no recurrence yet; seeded after the first decoded token

    gemma4_seq *one[1] = { s };
    int32_t last_tok = 0;

    // ── Cross-request prefix cache (RadixAttention): adopt the longest cached
    //    FULL-block prefix of this prompt (read-only, refcounted) and prefill only
    //    the divergent suffix. Gated to the full-attention single-pool geometry. A
    //    miss (nshared==0) falls through to the normal fast/slow prefill below,
    //    byte-for-byte unchanged. Shared blocks are immutable (full + already
    //    written) so reuse is lossless and needs no copy-on-write.
    int pc_nshared = 0;
    if (eng->prefix_cache_enabled) {
        int max_share = (n_prompt - 1) / PAGED_KV_BLOCK_TOKENS;   // always leave >=1 suffix token to forward
        if (max_share > 0 && paged_table_reserve(&s->glob_bt, max_share) == 0) {
            pc_nshared = prefix_lookup(&eng->glob_prefix, prompt, n_prompt,
                                       s->glob_bt.blocks, max_share);
            s->glob_bt.n = pc_nshared; s->glob_bt.base = 0;
        }
    }
    if (pc_nshared > 0) {
        int shared_tok = pc_nshared * PAGED_KV_BLOCK_TOKENS;
        s->n_tokens = shared_tok;
        // The sliding table is never read on this geometry; start it past the
        // shared region so it never allocates blocks for the reused prefix.
        s->slid_bt.base = shared_tok; s->slid_bt.n = 0;
        // Prefill the suffix. STAGE 9: for Qwen3/MoE try the COMPUTE-BOUND base-offset GEMM
        // suffix prefill first (one weight pass for ALL suffix tokens) — same -2-falls-back
        // contract the fresh path uses. -2 / unsupported → the lossless batched-chunk
        // decode_multiseq path (prefill_suffix_batched), one weight pass per ≤16-token chunk.
        // FUCINA_NO_FAST_PREFILL / g_fucina_force_slow_prefill force the chunk path (self-test).
        static int no_fast_sfx = -1;
        if (no_fast_sfx < 0) no_fast_sfx = (getenv("FUCINA_NO_FAST_PREFILL") != NULL);
        extern int g_fucina_force_slow_prefill;
        int sfx_rc = -2;
        if (sfx_rc != 0 &&
            prefill_suffix_batched(eng, s, prompt + shared_tok,
                                   n_prompt - shared_tok, /*do_sample=*/1, first_token_out) != 0) {
            gemma4_engine_seq_remove(eng, slot); return -1;
        }
        // Register this seq's full prompt blocks (shared ones skipped inside) so
        // concurrent/later requests reuse them.
        prefix_register(&eng->glob_prefix, prompt, n_prompt / PAGED_KV_BLOCK_TOKENS, s->glob_bt.blocks);
        return slot;
    }

    // ── Phase 4: SINGLE-PASS paged prefill (default ON). Processes the whole prompt
    //    in ONE weight pass via paged_prefill_batched instead of token-by-token. It
    //    is gated (fresh slot, N<=window, supported format) and returns -2 to fall
    //    through to the token-by-token loop below on any unsupported case, so worst
    //    case is no speedup — never a correctness regression. The first sampled token
    //    and the scattered paged KV are bit-identical to the token-by-token path (see
    //    paged_prefill_batched). FUCINA_NO_FAST_PREFILL forces the slow loop (for the
    //    dual-path determinism self-test).
    static int no_fast = -1;
    if (no_fast < 0) no_fast = (getenv("FUCINA_NO_FAST_PREFILL") != NULL);
    extern int g_fucina_force_slow_prefill;   // test override (self-test only)
    if (!no_fast && !g_fucina_force_slow_prefill) {
        int32_t ft = 0;
        int rc = paged_prefill_batched(eng, s, prompt, n_prompt, &ft);
        if (rc == 0) {
            cudaStreamSynchronize(eng->stream);
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                fprintf(stderr, "fucina: seq_add: fast-prefill CUDA error: %s\n", cudaGetErrorString(e));
                gemma4_engine_seq_remove(eng, slot); return -1;
            }
            // Register the cold-prefilled full prompt blocks so later requests reuse them.
            if (eng->prefix_cache_enabled)
                prefix_register(&eng->glob_prefix, prompt, n_prompt / PAGED_KV_BLOCK_TOKENS, s->glob_bt.blocks);
            if (first_token_out) *first_token_out = ft;
            return slot;
        }
        if (rc == -1) { gemma4_engine_seq_remove(eng, slot); return -1; }
        // rc == -2: unsupported (N>window / non-fresh / format) → token-by-token below.
        // s state is untouched on -2 (table_ensure may have grown tables — harmless;
        // the loop re-syncs per token from pos=0 since s->n_tokens is still 0).
    }

    // NOTE: token-by-token fallback (one decode_multiseq_forward per prompt position).
    // Correct but slow; used only when the single-pass fast path declines (rc==-2).
    for (int i = 0; i < n_prompt; i++) {
        int pos = s->n_tokens;
        if (paged_slot_sync(eng, s, pos) != 0) { gemma4_engine_seq_remove(eng, slot); return -1; }
        int32_t tok = prompt[i];
        int positions[1] = { pos };
        int want_sample = (i == n_prompt - 1);
        if (decode_multiseq_forward(eng, one, &tok, positions, 1, want_sample, NULL) != 0) {
            fprintf(stderr, "fucina: seq_add: decode_multiseq_forward failed at pos %d (%s)\n",
                    pos, cudaGetErrorString(cudaGetLastError()));
            gemma4_engine_seq_remove(eng, slot); return -1;
        }
        s->n_tokens = pos + 1;
        if (want_sample) {
            cudaMemcpyAsync(&last_tok, eng->d_ms_outtok, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, eng->stream);
            s->n_sampled++;   // this seq has now produced one token (RNG index)
        }
    }
    cudaStreamSynchronize(eng->stream);
    { cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) { fprintf(stderr, "fucina: seq_add: post-prefill CUDA error: %s\n", cudaGetErrorString(e));
                              gemma4_engine_seq_remove(eng, slot); return -1; } }
    if (eng->prefix_cache_enabled)
        prefix_register(&eng->glob_prefix, prompt, n_prompt / PAGED_KV_BLOCK_TOKENS, s->glob_bt.blocks);
    if (first_token_out) *first_token_out = last_tok;
    return slot;
}

// P1: BATCHED multi-sequence admission-prefill (Qwen3.5 only). Admits M prompts in ONE forward
// (weights amortized across all rows) instead of M serial single-seq prefills — the fix for the
// measured N=32 TTFT (32 x ~52 ms). Returns M (all admitted, out_slots/out_first filled), 0 when
// the engine does not support it (caller falls back to serial AddSeq), or -1 on error. Rollback
// on error frees any slots taken. Toggle off with FUCINA_NO_BATCHED_PREFILL.
extern "C" int gemma4_engine_seq_add_multiseq(
    gemma4_engine_t *eng, const int32_t *tokens_flat, const int *lens, int M,
    const float *temps, const int *topks, const float *topps, const float *minps,
    const uint64_t *seeds, int *out_slots, int32_t *out_first)
{
    if (!eng || !eng->loaded) return -1;
    if (eng->cfg.arch != GEMMA4_ARCH_QWEN3_5) return 0;   // unsupported → caller serial-admits
    static int disabled = -1;
    if (disabled < 0) disabled = (getenv("FUCINA_NO_BATCHED_PREFILL") != NULL);
    if (disabled) return 0;
    return qwen35_seq_add_multiseq(eng, tokens_flat, lens, M, temps, topks, topps, minps, seeds,
                                   out_slots, out_first);
}

// DEBUG (test-only): copy the just-computed first-token logits. nrows==1 → single-seq d_logits
// (VOC floats, as left by gemma4_engine_seq_add); nrows>1 → the batched head output d_sb[11]
// (nrows*VOC, as left by gemma4_engine_seq_add_multiseq). Used by the multiseq gate to measure
// the batched-vs-standalone logit-difference distribution that justifies the tolerance bound.
extern "C" int gemma4_engine_debug_logits(gemma4_engine_t *eng, float *out, int nrows) {
    if (!eng || !eng->loaded || nrows < 1 || !out) return -1;
    int VOC = eng->cfg.vocab_size;
    const float *src = (nrows == 1) ? eng->d_logits : eng->d_sb[11];
    if (!src) return -1;
    cudaMemcpy(out, src, (size_t)nrows * VOC * sizeof(float), cudaMemcpyDeviceToHost);
    return (cudaGetLastError() == cudaSuccess) ? VOC : -1;   // returns VOC on success
}

// ─── Chunked prefill (interleaved with decode) ───────────────────────────────
//
// gemma4_engine_seq_open reserves a free slot with EMPTY KV and stores the
// per-sequence sampling params, WITHOUT prefilling — the scheduler then feeds the
// prompt in bounded chunks via gemma4_engine_seq_prefill_chunk, interleaving those
// chunks with decode steps of the other slots so a long prompt never blocks the
// batch. Mirrors the slot-allocation + state-init prologue of gemma4_engine_seq_add.
// Chunked-prefill open WITH cross-request prefix-cache adoption. Reserves a slot, adopts
// the longest cached FULL-block prefix of `prompt` (read-only, refcounted) into the slot's
// KV, and reports how many prompt tokens are already satisfied (*shared_out) so the
// scheduler chunk-prefills ONLY the divergent suffix prompt[*shared_out:]. Without this,
// routing short prompts to the chunked interleave path would re-prefill cached prefixes
// and forfeit the prefix-cache win. Returns slot or -1.
extern "C" int gemma4_engine_seq_open_prefix(
    gemma4_engine_t *eng, const int32_t *prompt, int n_prompt, int *shared_out,
    float temp, int top_k, float top_p, float min_p, uint64_t seed)
{
    if (shared_out) *shared_out = 0;
    if (eng && eng->loaded && eng->cfg.arch == GEMMA4_ARCH_QWEN3_5)
        return qwen35_seq_open(eng, temp, top_k, top_p, min_p, seed);   // no radix cache on hybrid state
    int slot = gemma4_engine_seq_open(eng, temp, top_k, top_p, min_p, seed);
    if (slot < 0) return -1;
    if (!eng->prefix_cache_enabled || !prompt || n_prompt <= 0) return slot;
    gemma4_seq *s = &eng->slots[slot];
    int max_share = (n_prompt - 1) / PAGED_KV_BLOCK_TOKENS;   // leave >=1 suffix token to prefill
    if (max_share <= 0 || paged_table_reserve(&s->glob_bt, max_share) != 0) return slot;
    int nshared = prefix_lookup(&eng->glob_prefix, prompt, n_prompt, s->glob_bt.blocks, max_share);
    if (nshared <= 0) return slot;
    int shared_tok = nshared * PAGED_KV_BLOCK_TOKENS;
    s->glob_bt.n = nshared; s->glob_bt.base = 0;
    s->n_tokens = shared_tok;
    s->slid_bt.base = shared_tok; s->slid_bt.n = 0;   // slid never read on this geometry
    if (shared_out) *shared_out = shared_tok;
    return slot;
}

extern "C" int gemma4_engine_seq_open(
    gemma4_engine_t *eng, float temp, int top_k, float top_p, float min_p, uint64_t seed)
{
    if (eng && eng->loaded && eng->cfg.arch == GEMMA4_ARCH_QWEN3_5)
        return qwen35_seq_open(eng, temp, top_k, top_p, min_p, seed);
    if (!eng || !eng->loaded || !eng->paged_enabled) return -1;
    int slot = -1;
    for (int i = 0; i < GEMMA4_MAX_SEQS; i++) if (!eng->slots[i].used) { slot = i; break; }
    if (slot < 0) return -1;
    if (ensure_spec_scratch(eng) != 0 || ensure_ms_scratch(eng) != 0) return -1;

    gemma4_seq *s = &eng->slots[slot];
    // Fresh tables (struct was zeroed at create / released on remove).
    s->n_tokens = 0; s->slid_bt.base = 0; s->slid_bt.n = 0; s->glob_bt.base = 0; s->glob_bt.n = 0;
    s->used = 1;
    s->samp_temp = temp; s->samp_top_k = top_k; s->samp_top_p = top_p;
    s->samp_min_p = min_p; s->samp_seed = seed; s->n_sampled = 0;
    s->mtp_h_valid = 0;
    return slot;
}

// gemma4_engine_seq_prefill_chunk appends `n` prompt tokens to an OPEN slot's paged
// KV at the slot's CURRENT position (resumable suffix prefill). It is the token-by-token
// fallback loop of gemma4_engine_seq_add, generalized to start at s->n_tokens and to
// sample only when do_sample is set (the final chunk) — so splitting a prompt across
// chunks is invisible to the model: each token is forwarded at the same absolute
// position with the same per-token computation, and the paged KV after the final chunk
// is position-for-position identical to a one-shot seq_add of the whole prompt (and so
// is the first sampled token). Returns 0 on success, -1 on error (caller frees the slot).
extern "C" int gemma4_engine_seq_prefill_chunk(
    gemma4_engine_t *eng, int slot, const int32_t *tokens, int n,
    int do_sample, int32_t *first_token_out)
{
    if (eng && eng->loaded && eng->cfg.arch == GEMMA4_ARCH_QWEN3_5)
        return qwen35_seq_prefill_chunk(eng, slot, tokens, n, do_sample, first_token_out);
    if (!eng || !eng->loaded || !eng->paged_enabled) return -1;
    if (slot < 0 || slot >= GEMMA4_MAX_SEQS || !tokens || n <= 0) return -1;
    gemma4_seq *s = &eng->slots[slot];
    if (!s->used) return -1;

    // FAST path (default): prefill the chunk in batched ≤GEMMA4_MAX_SEQS-row passes (one
    // weight pass per ≤16 tokens) instead of token-by-token — bit-identical (the same
    // ragged-causality the prefix cache uses). This is what makes short-prompt chunked
    // prefill cheap enough to interleave with decode. FUCINA_NO_FAST_PREFILL /
    // g_fucina_force_slow_prefill force the token-by-token path for the determinism self-test.
    extern int g_fucina_force_slow_prefill;
    static int no_fast_chunk = -1;
    if (no_fast_chunk < 0) no_fast_chunk = (getenv("FUCINA_NO_FAST_PREFILL") != NULL);
    if (!no_fast_chunk && !g_fucina_force_slow_prefill)
        return prefill_suffix_batched(eng, s, tokens, n, do_sample, first_token_out);

    gemma4_seq *one[1] = { s };
    int32_t last_tok = 0;
    for (int i = 0; i < n; i++) {
        int pos = s->n_tokens;
        if (paged_slot_sync(eng, s, pos) != 0) return -1;     // out of KV blocks
        int32_t tok = tokens[i];
        int positions[1] = { pos };
        int want_sample = (do_sample && i == n - 1);
        if (decode_multiseq_forward(eng, one, &tok, positions, 1, want_sample, NULL) != 0) {
            fprintf(stderr, "fucina: seq_prefill_chunk: forward failed at pos %d (%s)\n",
                    pos, cudaGetErrorString(cudaGetLastError()));
            return -1;
        }
        s->n_tokens = pos + 1;
        if (want_sample) {
            cudaMemcpyAsync(&last_tok, eng->d_ms_outtok, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, eng->stream);
            s->n_sampled++;   // this seq has now produced one token (RNG index)
        }
    }
    cudaStreamSynchronize(eng->stream);
    { cudaError_t e = cudaGetLastError();
      if (e != cudaSuccess) { fprintf(stderr, "fucina: seq_prefill_chunk: CUDA error: %s\n", cudaGetErrorString(e));
                              return -1; } }
    if (do_sample && first_token_out) *first_token_out = last_tok;
    return 0;
}

// One batched forward over the B given slots: feed in_tokens[i] at each slot's
// current position, advance positions, sample one greedy token per slot into
// out_tokens. Returns 0 on success, -1 on error.
extern "C" int gemma4_engine_step_batch(
    gemma4_engine_t *eng, const int *slots, const int32_t *in_tokens, int B, int32_t *out_tokens)
{
    if (eng && eng->loaded && eng->cfg.arch == GEMMA4_ARCH_QWEN3_5)
        return qwen35_step_batch(eng, slots, in_tokens, B, out_tokens);
    if (!eng || !eng->loaded || !eng->paged_enabled || B <= 0 || B > GEMMA4_MAX_SEQS) return -1;
    if (ensure_spec_scratch(eng) != 0 || ensure_ms_scratch(eng) != 0) return -1;

    // Per-row admission: a row whose paged block table cannot grow (pool out of
    // blocks / sequence past its reserved context) is marked with the -1 sentinel
    // and EXCLUDED from this forward, so one sequence hitting its KV limit cannot
    // fail (and evict) the whole batch. The scheduler treats out == -1 as a
    // graceful per-sequence stop. Hard errors (bad slot id) still fail the call.
    gemma4_seq *slv[GEMMA4_MAX_SEQS];
    int positions[GEMMA4_MAX_SEQS];
    int32_t in2[GEMMA4_MAX_SEQS];
    int rowmap[GEMMA4_MAX_SEQS];
    int Bv = 0;
    for (int r = 0; r < B; r++) {
        int id = slots[r];
        if (id < 0 || id >= GEMMA4_MAX_SEQS || !eng->slots[id].used) return -1;
        gemma4_seq *s = &eng->slots[id];
        if (paged_slot_sync(eng, s, s->n_tokens) != 0) {   // out of KV blocks → stop this row
            if (out_tokens) out_tokens[r] = -1;
            continue;
        }
        slv[Bv] = s; positions[Bv] = s->n_tokens; in2[Bv] = in_tokens[r]; rowmap[Bv] = r; Bv++;
    }
    if (Bv == 0) return 0;   // every row hit its limit; caller evicts the -1 rows
    if (decode_multiseq_forward(eng, slv, in2, positions, Bv, /*want_sample=*/1, NULL) != 0)
        return -1;
    int32_t outs[GEMMA4_MAX_SEQS];
    cudaMemcpyAsync(outs, eng->d_ms_outtok, (size_t)Bv*sizeof(int32_t),
                    cudaMemcpyDeviceToHost, eng->stream);
    cudaStreamSynchronize(eng->stream);
    if (cudaGetLastError() != cudaSuccess) return -1;
    for (int v = 0; v < Bv; v++) {
        slv[v]->n_tokens = positions[v] + 1;
        slv[v]->n_sampled++;   // advance this seq's per-row RNG index
        if (out_tokens) out_tokens[rowmap[v]] = outs[v];
    }
    return 0;
}

// ── Stage 18 — FUSED prefill+decode ──────────────────────────────────────────
// Mix B_dec DECODE rows and pf_len PREFILL-CHUNK rows of ONE prefilling sequence into a
// SINGLE decode_multiseq_forward, so an arriving prompt's prefill never blocks the active
// sequences' decode (vLLM-style chunked prefill + continuous batching). Qwen3 family ONLY
// (full-causal geometry); Gemma returns -2 and keeps its own non-fused chunked path.
//
// Row layout (M = B_dec + pf_len rows, M <= GEMMA4_MAX_SEQS):
//   rows [0, B_dec)   decode : row r → slot dec_slots[r], token dec_toks[r], pos = slot->n_tokens
//   rows [B_dec, M)   prefill: ALL = pf_slot, token pf_chunk[j], pos = pf_base + j
// Each row writes its K/V at positions[r] into its OWN seq's paged pool and attends
// [0, positions[r]+1) via its per-row PagedSeqView.n_tokens bound (multiseq_upload_inputs), so
// cross-sequence isolation is automatic (distinct block tables) and the prefill rows see the
// adopted prefix + the earlier prefill rows of THIS pass (intra-pass causal) — exactly the
// prefill_suffix_batched causality. rng_off is NULL for every row: a decode row draws the SAME
// splitmix64 index it would in a plain step_batch (roff 0, its own n_sampled), so a co-batched
// decode is BYTE-IDENTICAL to a plain step_batch of the same rows; the prefill rows' samples are
// discarded except the LAST when pf_is_final (= the prompt's first generated token → *pf_first_out),
// matching prefill_suffix_batched's sample (also roff 0 over n_sampled==0).
//
// COMMIT: each surviving dec_slot n_tokens += 1 and n_sampled += 1 (out_dec[r] = sampled,
// out_dec_lens[r] = 1); a row whose KV could not grow is excluded from the forward and reports
// out_dec[r] = -1 / out_dec_lens[r] = -1 (the step_batch sentinel). pf_slot n_tokens += pf_len;
// pf_slot n_sampled += pf_is_final.
//
// GRAPH note: M varies per fused pass, so the greedy CUDA-graph fast path may not have a graph
// captured for this M — that's fine, decode_multiseq_forward falls to the per-kernel body
// (correctness first). When a graph for size M does exist (e.g. a prior plain step_batch of the
// same width) it is reused and is bit-identical (both graph paths use the fixed max split count).
extern "C" int gemma4_engine_step_batch_fused(
    gemma4_engine_t *eng,
    const int *dec_slots, const int32_t *dec_toks, int B_dec,
    int pf_slot, const int32_t *pf_chunk, int pf_len, int pf_is_final,
    int32_t *out_dec, int *out_dec_lens, int32_t *pf_first_out)
{
    // Legacy Stage-18 fused prefill+decode removed with legacy Qwen3. Qwen3.5 fuses via the
    // qwen35 batched engine; this returns -2 so the Go FusedPrefillEngine falls back cleanly.
    (void)dec_slots; (void)dec_toks; (void)B_dec; (void)pf_slot; (void)pf_chunk;
    (void)pf_len; (void)pf_is_final; (void)out_dec; (void)out_dec_lens; (void)pf_first_out;
    if (!eng || !eng->loaded) return -1;
    return -2;
}

static void mtp_draft_paged_batched(
    gemma4_engine_t *eng, gemma4_seq **dr_seq, const int32_t *dr_tok, int ndr,
    int32_t (*draft_out)[GEMMA4_SPEC_MAX], int *Dout, int max_draft);   // fwd decl (defined below)

// MTP speculative batched step. For each active slot: draft up to a per-slot budget with
// the paged drafter, then verify ALL slots' runs in ONE batched target forward, accept the
// longest target-matching prefix per slot, commit the accepted KV, and return each slot's
// emitted run. Byte-identical to gemma4_engine_step_batch: the verify forward re-derives
// every emitted token from the target distribution (greedy rows argmaxed, temp>0 rows
// sampled by decode_multiseq_forward), so a wrong draft only lowers the accept rate.
//
// Row budget: the batched scratch holds GEMMA4_MAX_SEQS rows, shared between the B pending
// tokens and their drafts — so total verify rows (B + Σ drafts) is capped at GEMMA4_MAX_SEQS
// and the per-slot draft length is (MAX_SEQS - B)/B. B==MAX_SEQS ⇒ no drafts ⇒ plain decode.
//
// out_tokens is [B * GEMMA4_SPEC_MAX] (each input row's run, contiguous); out_lens[B] is the
// per-row run length (>=1), or -1 if that slot hit its KV limit (scheduler evicts it).
// Shared verify+commit core. ext_drafts (when non-NULL) supplies EXTERNAL per-slot
// drafts (flat [B*GEMMA4_SPEC_MAX], ext_dlens[B] the count per row) — the model-agnostic
// prompt-lookup path driven from Go; otherwise the engine self-drafts with the MTP head.
// Either way the SAME batched target verify re-derives every emitted token, so the output
// is byte-identical to plain step_batch and lossless w.r.t. greedy decode.
static int step_batch_spec_impl(
    gemma4_engine_t *eng, const int *slots, const int32_t *in_tokens, int B,
    int32_t *out_tokens, int *out_lens,
    const int32_t *ext_drafts, const int *ext_dlens)
{
    // qwen35 hybrid: the GGUF carries no MTP head, and the stateful GDN/conv recurrence makes
    // a wide parallel draft-row verify (the gemma/qwen3 path below) both inapplicable AND unsafe
    // (decode_multiseq_forward is gemma/qwen3-layout — it would corrupt qwen35 weights). Serve
    // the spec ABI as plain LOSSLESS single-token steps: each row's run = {real argmax}, byte-
    // identical to greedy decode. (A future GGUF with the MTP head can widen this.)
    if (eng && eng->loaded && eng->cfg.arch == GEMMA4_ARCH_QWEN3_5) {
        if (B <= 0 || B > GEMMA4_MAX_SEQS) return -1;
        int32_t toks[GEMMA4_MAX_SEQS];
        int rc = qwen35_step_batch(eng, slots, in_tokens, B, toks);
        if (rc != 0) return rc;
        for (int r = 0; r < B; r++) {
            if (out_lens)   out_lens[r] = (toks[r] == -1) ? -1 : 1;  // -1 = KV-exhausted sentinel
            if (out_tokens) out_tokens[r * GEMMA4_SPEC_MAX] = toks[r];
        }
        return 0;
    }
    if (!eng || !eng->loaded || !eng->paged_enabled || B <= 0 || B > GEMMA4_MAX_SEQS) return -1;
    if (ensure_spec_scratch(eng) != 0 || ensure_ms_scratch(eng) != 0) return -1;
    cudaStream_t stream = eng->stream;
    const int H = eng->cfg.hidden_size;

    gemma4_seq *act[GEMMA4_MAX_SEQS]; int32_t g[GEMMA4_MAX_SEQS]; int rowmap[GEMMA4_MAX_SEQS];
    int A = 0;
    for (int r = 0; r < B; r++) {
        if (out_lens) out_lens[r] = 0;
        int id = slots[r];
        if (id < 0 || id >= GEMMA4_MAX_SEQS || !eng->slots[id].used) return -1;
        act[A] = &eng->slots[id]; g[A] = in_tokens[r]; rowmap[A] = r; A++;
    }

    // Per-slot draft budget so B + Σ drafts <= GEMMA4_MAX_SEQS verify rows, capped at
    // draft-k (FUCINA_BATCH_DRAFT_K). The whole round is drafted in ONE B-row forward per
    // token (mtp_draft_paged_batched) with a single sync, so draft depth is cheap; the cost
    // that matters is the verify ROW count R = A + Σdrafts, which is per-row compute-bound on
    // GB10 (the tree-spec finding). Spark defaults: draft_cap 6, and AUTO-DISABLE spec below a
    // minimum useful depth (FUCINA_BATCH_MIN_DRAFT, default 2) — at high concurrency the budget
    // (MAX_SEQS-A)/A collapses to depth 1, which ~doubles verify rows for a marginal τ, so those
    // batches fall back to plain decode. (A==1 single-stream keeps deep drafts.)
    int32_t draft[GEMMA4_MAX_SEQS][GEMMA4_SPEC_MAX];
    int D[GEMMA4_MAX_SEQS];
    for (int i = 0; i < A; i++) D[i] = 0;

    if (ext_drafts) {
        // EXTERNAL drafts (Go prompt-lookup): copy per-slot, then greedily clamp the per-slot
        // lengths so the flattened verify rows R = A + ΣD stay within the GEMMA4_MAX_SEQS
        // scratch budget (the Go scheduler already budgets this, but clamp defensively so a
        // wrong caller can never overrun the scratch). Greedy = earlier rows keep their full
        // draft; later rows lose tail drafts first.
        int rows_left = GEMMA4_MAX_SEQS - A;   // verify rows beyond the A mandatory anchors
        for (int i = 0; i < A; i++) {
            int d = ext_dlens ? ext_dlens[rowmap[i]] : 0;
            if (d < 0) d = 0;
            if (d > GEMMA4_SPEC_MAX - 1) d = GEMMA4_SPEC_MAX - 1;
            if (d > rows_left) d = rows_left;
            const int32_t *src = ext_drafts + (size_t)rowmap[i] * GEMMA4_SPEC_MAX;
            for (int j = 0; j < d; j++) draft[i][j] = src[j];
            D[i] = d;
            rows_left -= d;
        }
    } else {
    // Per-slot draft budget so B + Σ drafts <= GEMMA4_MAX_SEQS verify rows, capped at
    // draft-k (FUCINA_BATCH_DRAFT_K). The whole round is drafted in ONE B-row forward per
    // token (mtp_draft_paged_batched) with a single sync, so draft depth is cheap; the cost
    // that matters is the verify ROW count R = A + Σdrafts, which is per-row compute-bound on
    // GB10 (the tree-spec finding). Spark defaults: draft_cap 6, and AUTO-DISABLE spec below a
    // minimum useful depth (FUCINA_BATCH_MIN_DRAFT, default 2) — at high concurrency the budget
    // (MAX_SEQS-A)/A collapses to depth 1, which ~doubles verify rows for a marginal τ, so those
    // batches fall back to plain decode. (A==1 single-stream keeps deep drafts.)
    static int draft_cap = -1, min_draft = -1;
    if (draft_cap < 0) { const char *e = getenv("FUCINA_BATCH_DRAFT_K");   draft_cap = e ? atoi(e) : 6; }
    if (min_draft < 0) { const char *e = getenv("FUCINA_BATCH_MIN_DRAFT"); min_draft = e ? atoi(e) : 2; }
    int per_slot = (eng->mtp.loaded && GEMMA4_MAX_SEQS > A) ? (GEMMA4_MAX_SEQS - A) / A : 0;
    if (per_slot > draft_cap) per_slot = draft_cap;
    if (per_slot > GEMMA4_SPEC_MAX - 1) per_slot = GEMMA4_SPEC_MAX - 1;
    if (per_slot < min_draft) per_slot = 0;   // plain batched decode for this round

    // Gather the rows whose recurrence is seeded (mtp_h_valid set by a prior step's
    // post-verify update) and greedy, then draft them ALL in one batched round.
    if (per_slot > 0) {
        gemma4_seq *dr_seq[GEMMA4_MAX_SEQS]; int32_t dr_tok[GEMMA4_MAX_SEQS]; int dr_idx[GEMMA4_MAX_SEQS];
        int ndr = 0;
        for (int i = 0; i < A; i++) {
            gemma4_seq *s = act[i];
            if (s->samp_temp <= 0.0f && s->mtp_h_valid && s->d_mtp_h) {
                dr_seq[ndr] = s; dr_tok[ndr] = g[i]; dr_idx[ndr] = i; ndr++;
            }
        }
        if (ndr > 0) {
            int32_t dr_draft[GEMMA4_MAX_SEQS][GEMMA4_SPEC_MAX]; int dr_D[GEMMA4_MAX_SEQS];
            mtp_draft_paged_batched(eng, dr_seq, dr_tok, ndr, dr_draft, dr_D, per_slot);
            for (int k = 0; k < ndr; k++) {
                int i = dr_idx[k];
                D[i] = dr_D[k];
                for (int j = 0; j < dr_D[k]; j++) draft[i][j] = dr_draft[k][j];
            }
        }
    }
    }

    // Flatten (slot × [g, draft...]) into R verify rows; grow each slot's KV to cover them.
    gemma4_seq *vslv[GEMMA4_MAX_SEQS]; int32_t vtok[GEMMA4_MAX_SEQS]; int vpos[GEMMA4_MAX_SEQS];
    int vrng[GEMMA4_MAX_SEQS];   // per-row sampled-ordinal offset (anchor 0, draft j → 1+j)
    int off[GEMMA4_MAX_SEQS]; int R = 0;
    for (int i = 0; i < A; i++) {
        gemma4_seq *s = act[i];
        int pos = s->n_tokens, K = D[i] + 1;
        int okp = (paged_table_ensure(&eng->slid_pool, &s->slid_bt, pos + K) == 0 &&
                   glob_table_ensure(eng, s, pos + K) == 0) ? 0 : -1;
        if (okp == 0) {
            // Recycle sliding by the COMMITTED pos only, so every draft row's window stays
            // mapped (draft positions are within their own window of pos..pos+K-1).
            paged_table_advance_sliding(&eng->slid_pool, &s->slid_bt, pos + 1, GEMMA4_SLIDING_WINDOW);
            if (paged_upload_blocks(&s->slid_bt, &s->d_slid_blocks, &s->d_slid_cap, stream) != 0 ||
                paged_upload_blocks(&s->glob_bt, &s->d_glob_blocks, &s->d_glob_cap, stream) != 0) okp = -1;
        }
        if (okp != 0) {                       // KV exhausted → stop this slot
            if (out_lens) out_lens[rowmap[i]] = -1;
            D[i] = -1; off[i] = -1; s->mtp_h_valid = 0;
            continue;
        }
        off[i] = R;
        vslv[R] = s; vtok[R] = g[i];        vpos[R] = pos;     vrng[R] = 0;     R++;   // row 0: pending token
        for (int j = 0; j < D[i]; j++) { vslv[R] = s; vtok[R] = draft[i][j]; vpos[R] = pos + 1 + j; vrng[R] = 1 + j; R++; }
    }
    if (R == 0) return 0;

    // ONE batched verify forward (greedy rows argmaxed, temp>0 rows sampled). No state mutation.
    // vrng gives each verify row an INDEPENDENT per-position uniform (anchor draws the
    // n_sampled-th token's stream, draft j the (n_sampled+1+j)-th), so a committed run of a
    // accepted draft tokens consumes the exact RNG indices a plain per-token sampled decode
    // would — making sampled spec lossless (distributionally identical AND reproducible),
    // not just argmax-greedy-safe.
    if (decode_multiseq_forward(eng, vslv, vtok, vpos, R, /*want_sample=*/1, vrng) != 0) return -1;
    int32_t outs[GEMMA4_MAX_SEQS];
    cudaMemcpyAsync(outs, eng->d_ms_outtok, (size_t)R*sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return -1;
    if (cudaGetLastError() != cudaSuccess) return -1;

    // Per-slot acceptance + commit + recurrence update + emit.
    static int dbg = -1;
    if (dbg < 0) dbg = (getenv("FUCINA_SPEC_DEBUG") != NULL);
    int dbg_drafted = 0, dbg_accepted = 0;
    for (int i = 0; i < A; i++) {
        if (D[i] < 0) continue;                              // dropped (KV-exhausted)
        int o = off[i], Di = D[i];
        int a = 0;
        while (a < Di && outs[o + a] == draft[i][a]) a++;    // longest target-matching prefix
        if (dbg) { dbg_drafted += Di; dbg_accepted += a; }
        gemma4_seq *s = act[i];
        int pos = s->n_tokens;
        int32_t *run_out = out_tokens + (size_t)rowmap[i] * GEMMA4_SPEC_MAX;
        for (int j = 0; j <= a; j++) run_out[j] = outs[o + j];   // a accepted + 1 bonus/resample
        if (out_lens) out_lens[rowmap[i]] = a + 1;
        s->n_tokens   = pos + a + 1;                         // g + a accepted drafts committed
        s->n_sampled += (a + 1);
        // Seed/refresh the recurrence for the NEXT step: next-h = the target output-norm
        // hidden of row (o+a), which pairs with the new pending token outs[o+a]. Allocate
        // lazily here (greedy + MTP only) so the FIRST step bootstraps drafting on step 2.
        if (eng->mtp.loaded && s->samp_temp <= 0.0f) {
            if (!s->d_mtp_h && cudaMalloc(&s->d_mtp_h, (size_t)H*sizeof(float)) != cudaSuccess) s->d_mtp_h = NULL;
            if (s->d_mtp_h) {
                cudaMemcpyAsync(s->d_mtp_h, eng->d_sb[2] + (size_t)(o + a) * H,
                                (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, stream);
                s->mtp_h_valid = 1;
            }
        }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return -1;
    if (dbg) {
        static long td = 0, ta = 0, steps = 0;
        td += dbg_drafted; ta += dbg_accepted; steps++;
        if ((steps % 32) == 0)
            fprintf(stderr, "fucina: spec-batch step %ld: drafted=%d accepted=%d (cum %ld/%ld = %.0f%%)\n",
                    steps, dbg_drafted, dbg_accepted, ta, td, td ? 100.0*ta/td : 0.0);
    }
    return 0;
}

// MTP speculative batched step (self-drafting). Byte-identical to the committed ABI.
extern "C" int gemma4_engine_step_batch_spec(
    gemma4_engine_t *eng, const int *slots, const int32_t *in_tokens, int B,
    int32_t *out_tokens, int *out_lens)
{
    return step_batch_spec_impl(eng, slots, in_tokens, B, out_tokens, out_lens, NULL, NULL);
}

// Speculative batched step verifying EXTERNAL per-slot drafts (Go prompt-lookup): drafts is
// flat [B*GEMMA4_SPEC_MAX], dlens[B] the count per row. Same lossless verify as the MTP path;
// works for any arch (no MTP head required), which is what the model-agnostic drafter needs.
extern "C" int gemma4_engine_step_batch_spec_ext(
    gemma4_engine_t *eng, const int *slots, const int32_t *in_tokens, int B,
    int32_t *out_tokens, int *out_lens,
    const int32_t *drafts, const int *dlens)
{
    return step_batch_spec_impl(eng, slots, in_tokens, B, out_tokens, out_lens, drafts, dlens);
}

// Free a slot's block tables back to the pools and mark it free.
extern "C" void gemma4_engine_seq_remove(gemma4_engine_t *eng, int slot) {
    if (!eng || slot < 0 || slot >= GEMMA4_MAX_SEQS) return;
    if (eng->cfg.arch == GEMMA4_ARCH_QWEN3_5) {
        // qwen35 carries no paged KV / prefix-cache state; the per-slot GDN/conv/KV arenas
        // are re-zeroed on the next seq_add, so releasing the slot is just clearing the flag.
        eng->slots[slot].used = 0; eng->slots[slot].n_tokens = 0;
        return;
    }
    gemma4_seq *s = &eng->slots[slot];
    paged_table_release(&eng->slid_pool, &s->slid_bt);
    if (eng->prefix_cache_enabled) {
        // Decrement refcounts: a registered block at refcount 0 stays cached
        // (EVICTABLE, re-hittable); a private/tail block returns to the pool.
        prefix_release(&eng->glob_prefix, &eng->glob_pool, s->glob_bt.blocks, s->glob_bt.n);
        s->glob_bt.n = 0; s->glob_bt.base = 0;
    } else {
        paged_table_release(&eng->glob_pool, &s->glob_bt);
    }
    s->n_tokens = 0; s->used = 0;
    s->mtp_h_valid = 0;   // recurrence does not survive slot reuse
    // d_slid_blocks/d_glob_blocks and d_mtp_h device buffers are retained for slot reuse.
}

// Number of free slots available for new sequences.
extern "C" int gemma4_engine_seq_capacity(gemma4_engine_t *eng) {
    // qwen35 hybrid: no paged-KV pool. Capacity is the number of state/KV arenas actually
    // allocated (runtime-capped by --parallel/FUCINA_PAGED_MAXSEQS), not the compile-time array.
    if (eng && eng->loaded && eng->cfg.arch == GEMMA4_ARCH_QWEN3_5) {
        int used = 0;
        for (int i = 0; i < eng->q35.capacity; i++) if (eng->slots[i].used) used++;
        int free_eff = eng->q35.capacity - used;
        return free_eff > 0 ? free_eff : 0;
    }
    if (!eng || !eng->paged_enabled) return 0;
    // Free capacity is bounded by what the BLOCK POOL can back (paged_cap), not
    // the raw slot count: admitting more would over-subscribe the pool and let one
    // sequence's mid-generation block growth fail (and evict) the entire batch.
    int used = 0;
    for (int i = 0; i < GEMMA4_MAX_SEQS; i++) if (eng->slots[i].used) used++;
    int free_eff = eng->paged_cap - used;
    return free_eff > 0 ? free_eff : 0;
}

// Enable/disable the cross-request prefix cache. Effective ONLY on the full-
// attention single-pool geometry (n_layers_sliding==0, the tree allocated at
// create); a no-op for Gemma. Also the A/B override the lossless test uses to
// force a cold reference run.
extern "C" void gemma4_engine_set_prefix_cache(gemma4_engine_t *eng, int enable) {
    if (!eng) return;
    eng->prefix_cache_enabled = (enable && eng->n_layers_sliding == 0 &&
                                 eng->glob_prefix.refcount != NULL) ? 1 : 0;
}

// Register any full GLOBAL blocks of a slot's sequence that have completed (prompt
// AND generated text) using the caller's authoritative committed token history
// [0,n). Lets a later request reuse this sequence's GENERATED continuation, not just
// its prompt (e.g. a multi-turn chat whose next prompt = prior prompt+response).
// Idempotent: prefix_register skips blocks already in the tree, so the scheduler can
// call it whenever a sequence crosses a 256-token boundary. No-op when disabled.
extern "C" void gemma4_engine_prefix_commit(gemma4_engine_t *eng, int slot,
                                            const int32_t *history, int n) {
    if (!eng || !eng->prefix_cache_enabled || slot < 0 || slot >= GEMMA4_MAX_SEQS) return;
    gemma4_seq *s = &eng->slots[slot];
    if (!s->used || n <= 0) return;
    int full = n / PAGED_KV_BLOCK_TOKENS;
    if (full > s->glob_bt.n) full = s->glob_bt.n;   // only blocks actually backed by KV
    prefix_register(&eng->glob_prefix, history, full, s->glob_bt.blocks);
}

// Observability counters (all zero when the cache is disabled/uninitialized).
extern "C" void gemma4_engine_prefix_cache_stats(const gemma4_engine_t *eng,
        uint64_t *lookups, uint64_t *hit_blocks, uint64_t *cached_blocks, uint64_t *evictions) {
    const PrefixTree *c = (eng && eng->prefix_cache_enabled) ? &eng->glob_prefix : NULL;
    prefix_tree_stats(c, lookups, hit_blocks, cached_blocks, evictions);
}

// Self-test (FUCINA_BATCH_SELFTEST): GEMMA4_MAX_SEQS (32) distinct short token
// sequences, each run through single-seq decode (greedy argmax, K steps) and then
// together as a full-width batch via seq_add + step_batch; assert the batched
// per-step argmax is BYTE-IDENTICAL to the single-seq argmax for every sequence
// (32-wide batched decode == 32 independent single-row decodes, greedy temp=0).
// This is the 32-concurrency correctness gate. Mirrors the FUCINA_PAGED_*_SELFTEST
// create hooks. Needs FUCINA_PAGED_MAXSEQS>=32 so 32 slots are addable at once.
static void gemma4_engine_batch_selftest(gemma4_engine_t *eng) {
    if (!eng->paged_enabled) return;
    // qwen35 has its own hybrid batched path (own fp32 GDN/conv/KV arenas, not the paged
    // fp8 pool) and a dedicated M4 gate (qwen35_batch_selftest); this Gemma/Qwen3 greedy+
    // sampling self-test does not apply to it.
    if (eng->cfg.arch == GEMMA4_ARCH_QWEN3_5) return;
    const int NSEQ = GEMMA4_MAX_SEQS;   // 32-wide: exercises the full multiseq decode batch
    const int KSTEP = 32;
    const int NP = 4;
    // NSEQ distinct short prompts (token ids in-vocab; clear of specials).
    int32_t prompt[NSEQ][4];
    for (int q = 0; q < NSEQ; q++)
        for (int j = 0; j < NP; j++)
            prompt[q][j] = 100 + q * 13 + j * 37;

    // ── Reference: each sequence alone through the single-seq slot path ──
    // Use a private slot (seq_add prefills then we step_batch with B=1), which is
    // the SAME math the batch uses but one row at a time — the regression bar is
    // batched(B=3) == B separate single-row runs.
    int32_t ref[NSEQ][KSTEP];
    for (int q = 0; q < NSEQ; q++) {
        int32_t first = 0;
        int slot = gemma4_engine_seq_add(eng, prompt[q], NP, &first, 0.0f, 0, 0.0f, 0.0f, 0);
        if (slot < 0) { fprintf(stderr, "fucina: batch self-test: seq_add(ref) failed\n"); return; }
        int32_t tok = first;
        for (int k = 0; k < KSTEP; k++) {
            ref[q][k] = tok;
            int32_t nxt = 0; int sl = slot;
            if (gemma4_engine_step_batch(eng, &sl, &tok, 1, &nxt) != 0) {
                fprintf(stderr, "fucina: batch self-test: step(ref) failed\n");
                gemma4_engine_seq_remove(eng, slot); return;
            }
            tok = nxt;
        }
        gemma4_engine_seq_remove(eng, slot);
    }

    // ── Batched: all NSEQ together, B=NSEQ each step ──
    int slots[NSEQ]; int32_t cur[NSEQ]; int32_t bat[NSEQ][KSTEP];
    for (int q = 0; q < NSEQ; q++) {
        int32_t first = 0;
        slots[q] = gemma4_engine_seq_add(eng, prompt[q], NP, &first, 0.0f, 0, 0.0f, 0.0f, 0);
        if (slots[q] < 0) { fprintf(stderr, "fucina: batch self-test: seq_add(batch) failed\n"); return; }
        cur[q] = first;
    }
    for (int k = 0; k < KSTEP; k++) {
        int32_t nxt[NSEQ];
        for (int q = 0; q < NSEQ; q++) bat[q][k] = cur[q];
        if (gemma4_engine_step_batch(eng, slots, cur, NSEQ, nxt) != 0) {
            fprintf(stderr, "fucina: batch self-test: step(batch) failed\n");
            for (int q = 0; q < NSEQ; q++) gemma4_engine_seq_remove(eng, slots[q]);
            return;
        }
        for (int q = 0; q < NSEQ; q++) cur[q] = nxt[q];
    }
    for (int q = 0; q < NSEQ; q++) gemma4_engine_seq_remove(eng, slots[q]);

    // ── Compare ──
    int all_pass = 1;
    for (int q = 0; q < NSEQ; q++) {
        int agree = 0, first_mism = -1;
        for (int k = 0; k < KSTEP; k++) {
            if (ref[q][k] == bat[q][k]) agree++;
            else if (first_mism < 0) first_mism = k;
        }
        double pct = 100.0 * agree / KSTEP;
        int pass = (agree == KSTEP);   // byte-identity: 32-wide batched == per-row single
        all_pass &= pass;
        fprintf(stderr,
            "fucina: batch self-test seq %d: %d/%d argmax agree (%.0f%%)%s%s\n",
            q, agree, KSTEP, pct,
            (first_mism >= 0) ? " first-mismatch step " : "",
            "");
        if (first_mism >= 0)
            fprintf(stderr, "fucina:   seq %d first mismatch at step %d (ref %d vs batch %d)\n",
                    q, first_mism, ref[q][first_mism], bat[q][first_mism]);
    }
    fprintf(stderr, "fucina: batch self-test %s — batched(B=%d) decode vs %d single-seq decodes\n",
            all_pass ? "PASSED" : "FAILED", NSEQ, NSEQ);

    // ── temp>0 per-row sampling: deterministic, reproducible, NON-greedy ──
    // Run prompt[0] twice with a FIXED seed at temp>0 and assert: (a) the two
    // runs are byte-identical (reproducible RNG stream keyed on seed+token idx),
    // and (b) the sampled sequence DIFFERS from the greedy sequence ref[0]
    // (proves the params are actually honored, not silently argmax). A high temp
    // makes a divergence within KSTEP steps overwhelmingly likely.
    {
        const float TEMP = 1.3f; const int TOPK = 64; const uint64_t SEED = 1234567ULL;
        int32_t s1[KSTEP], s2[KSTEP];
        for (int pass = 0; pass < 2; pass++) {
            int32_t *dst = pass ? s2 : s1;
            int32_t first = 0;
            int slot = gemma4_engine_seq_add(eng, prompt[0], NP, &first, TEMP, TOPK, 0.0f, 0.0f, SEED);
            if (slot < 0) { fprintf(stderr, "fucina: batch self-test: seq_add(sample) failed\n"); return; }
            int32_t tok = first;
            for (int k = 0; k < KSTEP; k++) {
                dst[k] = tok;
                int32_t nxt = 0; int sl = slot;
                if (gemma4_engine_step_batch(eng, &sl, &tok, 1, &nxt) != 0) {
                    fprintf(stderr, "fucina: batch self-test: step(sample) failed\n");
                    gemma4_engine_seq_remove(eng, slot); return;
                }
                tok = nxt;
            }
            gemma4_engine_seq_remove(eng, slot);
        }
        int reproducible = 1, differs_from_greedy = 0;
        for (int k = 0; k < KSTEP; k++) {
            if (s1[k] != s2[k]) reproducible = 0;
            if (s1[k] != ref[0][k]) differs_from_greedy = 1;
        }
        fprintf(stderr,
            "fucina: batch self-test sampling (temp=%.2f seed=%llu): reproducible=%s non-greedy=%s — %s\n",
            TEMP, (unsigned long long)SEED,
            reproducible ? "yes" : "NO", differs_from_greedy ? "yes" : "NO",
            (reproducible && differs_from_greedy) ? "PASSED" : "FAILED");
    }
}

// =========================================================================
// ─── Token-tree speculative decode ──────────────────────────────────────
// =========================================================================


// =========================================================================
// ─── Gemma-4 MTP assistant drafter (llama.cpp PR #23398 equivalent) ─────
// =========================================================================
// Google ships an official ~423M "assistant" head with Gemma 4: 4 transformer layers
// (pattern [sliding,sliding,sliding,global]) with Q-ONLY attention — no K/V projections;
// every sliding layer attends the TARGET's layer-46 sliding KV and the global layer the
// target's layer-47 global KV (llama.cpp share(il) = n_layer-2 / n_layer-1) — plus
// nextn.pre_projection ([embed(tok)·√3840 ; h] 7680 → 1024), the per-layer sandwich
// identical to the target's, output_norm, its own 1024-wide Q8_0 unembed, and
// nextn.post_projection (1024 → 3840) producing the next recurrent h. Drafting is
// recursive — (h, tok) → (logits → argmax tok', h') — with ALL draft tokens at RoPE
// position n_past (per the gemma4_assistant reference), reading the FROZEN committed
// target cache and writing NO KV, so the draft leaves no state and the existing
// batched verify + exact rewind machinery applies unchanged. Measured by llama.cpp on
// this hardware class: >2× dense-model decode at ~0.59 average acceptance.

#define GEMMA4_MTP_LAYERS  4
#define GEMMA4_MTP_HIDDEN  1024
#define GEMMA4_MTP_FFN     8192

static inline const uint8_t *mtp_w(const gemma4_engine_t *eng, uint64_t off) {
    return eng->mtp.d_w + off;
}
static inline const float *mtp_f32(const gemma4_engine_t *eng, uint64_t off) {
    return (const float *)(eng->mtp.d_w + off);
}

// Spec-batch correctness self-test (FUCINA_SPEC_BATCH_SELFTEST), run after the MTP head
// loads. Invariant: step_batch_spec(B) emits the SAME token sequence as plain step_batch(B) —
// the verify re-derives every emitted token from the target, so drafting only changes which
// tokens arrive as "accepted" vs "bonus", never the sequence. So the batched drafter is
// correct iff the flattened spec stream is BYTE-IDENTICAL to the non-spec batch stream.
static void gemma4_engine_spec_batch_selftest(gemma4_engine_t *eng) {
    if (!eng->paged_enabled || !eng->mtp.loaded) return;
    const int NSEQ = GEMMA4_MAX_SEQS, NEED = 48, NP = 4;   // toward MAX_SEQS verify rows
    int32_t prompt[NSEQ][4];
    for (int q = 0; q < NSEQ; q++)
        for (int j = 0; j < NP; j++)
            prompt[q][j] = 100 + q * 13 + j * 37;

    // ── Reference: all NSEQ as a batch through the NON-spec step_batch (greedy) ──
    int slots[NSEQ]; int32_t cur[NSEQ]; int32_t ref[NSEQ][NEED];
    for (int q = 0; q < NSEQ; q++) {
        int32_t first = 0;
        slots[q] = gemma4_engine_seq_add(eng, prompt[q], NP, &first, 0.0f, 0, 0.0f, 0.0f, 0);
        if (slots[q] < 0) { fprintf(stderr, "fucina: spec-batch self-test: seq_add(ref) failed\n"); return; }
        ref[q][0] = first; cur[q] = first;
    }
    for (int k = 1; k < NEED; k++) {
        int32_t nxt[NSEQ];
        if (gemma4_engine_step_batch(eng, slots, cur, NSEQ, nxt) != 0) {
            for (int q = 0; q < NSEQ; q++) gemma4_engine_seq_remove(eng, slots[q]);
            fprintf(stderr, "fucina: spec-batch self-test: step_batch(ref) failed\n"); return;
        }
        for (int q = 0; q < NSEQ; q++) { ref[q][k] = nxt[q]; cur[q] = nxt[q]; }
    }
    for (int q = 0; q < NSEQ; q++) gemma4_engine_seq_remove(eng, slots[q]);

    // ── Spec-batch: all NSEQ through step_batch_spec; flatten each slot's emitted runs ──
    int32_t got[NSEQ][NEED + GEMMA4_SPEC_MAX]; int ng[NSEQ] = {0};
    for (int q = 0; q < NSEQ; q++) {
        int32_t first = 0;
        slots[q] = gemma4_engine_seq_add(eng, prompt[q], NP, &first, 0.0f, 0, 0.0f, 0.0f, 0);
        if (slots[q] < 0) { fprintf(stderr, "fucina: spec-batch self-test: seq_add(spec) failed\n"); return; }
        got[q][ng[q]++] = first; cur[q] = first;
    }
    int32_t out[NSEQ * GEMMA4_SPEC_MAX]; int lens[NSEQ]; int guard = 0;
    for (;;) {
        int need_more = 0;
        for (int q = 0; q < NSEQ; q++) if (ng[q] < NEED) need_more = 1;
        if (!need_more || guard++ > NEED * 2) break;
        if (gemma4_engine_step_batch_spec(eng, slots, cur, NSEQ, out, lens) != 0) {
            fprintf(stderr, "fucina: spec-batch self-test: step_batch_spec failed\n"); break;
        }
        for (int q = 0; q < NSEQ; q++) {
            int n = lens[q];
            if (n <= 0) continue;                       // -1 KV-exhausted / 0 none
            for (int j = 0; j < n && ng[q] < NEED + GEMMA4_SPEC_MAX; j++)
                got[q][ng[q]++] = out[q * GEMMA4_SPEC_MAX + j];
            cur[q] = got[q][ng[q] - 1];                 // next input = last emitted
        }
    }
    for (int q = 0; q < NSEQ; q++) gemma4_engine_seq_remove(eng, slots[q]);

    // ── Compare: must be byte-identical over NEED tokens ──
    int all_pass = 1;
    for (int q = 0; q < NSEQ; q++) {
        int agree = 0, first_mism = -1;
        for (int k = 0; k < NEED; k++) {
            if (k < ng[q] && ref[q][k] == got[q][k]) agree++;
            else if (first_mism < 0) first_mism = k;
        }
        int pass = (agree == NEED); all_pass &= pass;
        fprintf(stderr, "fucina: spec-batch self-test seq %d: %d/%d match%s\n",
                q, agree, NEED, pass ? "" : " (MISMATCH)");
        if (first_mism >= 0 && first_mism < ng[q])
            fprintf(stderr, "fucina:   seq %d first mismatch at %d (ref %d vs spec %d)\n",
                    q, first_mism, ref[q][first_mism], got[q][first_mism]);
    }
    fprintf(stderr, "fucina: spec-batch self-test %s — step_batch_spec(B=%d) vs non-spec batch\n",
            all_pass ? "PASSED" : "FAILED", NSEQ);
}

// Dual-path prefill determinism self-test (FUCINA_FAST_PREFILL_SELFTEST): prefill the
// SAME prompt twice — once via the single-pass paged_prefill_batched (fast, default)
// and once via the forced token-by-token loop — then GREEDY-decode K steps from each
// and assert IDENTICAL token streams. This proves the scattered paged KV is bit-
// identical to the token-by-token path (not just the first sampled token). A few late
// fp near-tie flips are tolerated (PASS if >= 90% of the K continuation tokens agree
// AND the first sampled token matches exactly).
static void gemma4_engine_fast_prefill_selftest(gemma4_engine_t *eng) {
    if (!eng->paged_enabled) return;
    const int KSTEP = 24;
    // A few prompt lengths exercising the materialized N×N attention at different N.
    const int lens[] = { 1, 16, 64, 200, 256 };
    const int NL = (int)(sizeof(lens)/sizeof(lens[0]));
    int all_pass = 1;
    for (int li = 0; li < NL; li++) {
        int P = lens[li];
        int32_t prompt[256];
        for (int i = 0; i < P; i++) prompt[i] = 100 + (i * 37 + 11) % 700;

        // ── Reference: token-by-token prefill (forced) + greedy continuation ──
        int32_t ref[KSTEP]; int32_t ref_first = 0;
        g_fucina_force_slow_prefill = 1;
        int slot = gemma4_engine_seq_add(eng, prompt, P, &ref_first, 0.0f, 0, 0.0f, 0.0f, 0);
        g_fucina_force_slow_prefill = 0;
        if (slot < 0) { fprintf(stderr, "fucina: fast-prefill self-test: slow seq_add failed (P=%d)\n", P); return; }
        {
            int32_t tok = ref_first;
            for (int k = 0; k < KSTEP; k++) {
                ref[k] = tok; int32_t nxt = 0; int sl = slot;
                if (gemma4_engine_step_batch(eng, &sl, &tok, 1, &nxt) != 0) {
                    gemma4_engine_seq_remove(eng, slot); return;
                }
                tok = nxt;
            }
        }
        gemma4_engine_seq_remove(eng, slot);

        // ── Fast: single-pass paged prefill + greedy continuation ──
        int32_t got[KSTEP]; int32_t fast_first = 0;
        slot = gemma4_engine_seq_add(eng, prompt, P, &fast_first, 0.0f, 0, 0.0f, 0.0f, 0);
        if (slot < 0) { fprintf(stderr, "fucina: fast-prefill self-test: fast seq_add failed (P=%d)\n", P); return; }
        {
            int32_t tok = fast_first;
            for (int k = 0; k < KSTEP; k++) {
                got[k] = tok; int32_t nxt = 0; int sl = slot;
                if (gemma4_engine_step_batch(eng, &sl, &tok, 1, &nxt) != 0) {
                    gemma4_engine_seq_remove(eng, slot); return;
                }
                tok = nxt;
            }
        }
        gemma4_engine_seq_remove(eng, slot);

        int agree = 0, first_mism = -1;
        for (int k = 0; k < KSTEP; k++) {
            if (ref[k] == got[k]) agree++;
            else if (first_mism < 0) first_mism = k;
        }
        int first_ok = (ref_first == fast_first);
        int pass = first_ok && (agree >= (KSTEP * 9 + 9) / 10);   // >=90%
        all_pass &= pass;
        fprintf(stderr, "fucina: fast-prefill self-test P=%d: first %s (slow=%d fast=%d), cont %d/%d match%s\n",
                P, first_ok ? "OK" : "MISMATCH", ref_first, fast_first, agree, KSTEP,
                pass ? "" : (first_mism >= 0 ? " (DIVERGE)" : ""));
    }
    fprintf(stderr, "fucina: fast-prefill self-test %s — single-pass paged prefill vs token-by-token\n",
            all_pass ? "PASSED" : "FAILED");
}

// Decode-step micro-bench (FUCINA_BATCH_DECODE_BENCH): prefill B synthetic sequences to
// ~L tokens, then time K batched decode steps via the NON-spec step_batch (isolates the
// batched forward from prefill / HTTP / spec). Prints ms/step so the global-attention
// register-spill fix can be measured directly without a server run.
static void gemma4_engine_batch_decode_bench(gemma4_engine_t *eng) {
    if (!eng->paged_enabled) return;
    const int B = 4, L = 256, K = 40;
    int slots[GEMMA4_MAX_SEQS]; int32_t cur[GEMMA4_MAX_SEQS];
    // Prefill-cost probe: one seq_add of a P-token prompt (token-by-token prefill path).
    {
        const int P = 256; int32_t pr[256];
        for (int i = 0; i < P; i++) pr[i] = 100 + (i % 700);
        struct timespec p0, p1; cudaStreamSynchronize(eng->stream);
        clock_gettime(CLOCK_MONOTONIC, &p0);
        int32_t f = 0; int sl = gemma4_engine_seq_add(eng, pr, P, &f, 0.0f, 0, 0.0f, 0.0f, 0);
        cudaStreamSynchronize(eng->stream);
        clock_gettime(CLOCK_MONOTONIC, &p1);
        double pms = (p1.tv_sec - p0.tv_sec)*1e3 + (p1.tv_nsec - p0.tv_nsec)/1e6;
        fprintf(stderr, "fucina: prefill-probe: %d-token seq_add = %.1f ms (%.2f ms/token)\n", P, pms, pms/P);
        if (sl >= 0) gemma4_engine_seq_remove(eng, sl);
    }
    int32_t prefill[8] = {100,200,300,400,500,600,700,800};
    for (int q = 0; q < B; q++) {
        int32_t first = 0;
        slots[q] = gemma4_engine_seq_add(eng, prefill, 8, &first, 0.0f, 0, 0.0f, 0.0f, 0);
        if (slots[q] < 0) { fprintf(stderr, "fucina: decode-bench: seq_add failed\n"); return; }
        cur[q] = first;
    }
    // Grow each sequence to ~L tokens with throwaway decode steps (warm + realistic ctx).
    for (int k = 8; k < L; k++) {
        int32_t nxt[GEMMA4_MAX_SEQS];
        if (gemma4_engine_step_batch(eng, slots, cur, B, nxt) != 0) break;
        for (int q = 0; q < B; q++) cur[q] = nxt[q];
    }
    cudaStreamSynchronize(eng->stream);
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int k = 0; k < K; k++) {
        int32_t nxt[GEMMA4_MAX_SEQS];
        gemma4_engine_step_batch(eng, slots, cur, B, nxt);
        for (int q = 0; q < B; q++) cur[q] = nxt[q];
    }
    cudaStreamSynchronize(eng->stream);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)/1e6;
    fprintf(stderr, "fucina: decode-bench B=%d ctx~%d: %.1f ms/step (%.1f tok/s aggregate over %d steps)\n",
            B, L, ms/K, (double)B*K/(ms/1e3), K);
    for (int q = 0; q < B; q++) gemma4_engine_seq_remove(eng, slots[q]);
}

// Load the assistant GGUF (separate small file) and upload it whole to the device.
// Returns 0 on success. Safe to call once after the target model is loaded.
int gemma4_engine_load_assistant(gemma4_engine_t *eng, const char *path)
{
    if (!eng || !eng->loaded || !path) return -1;
    if (eng->mtp.loaded) return 0;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "fucina: assistant open failed: %s\n", path); return -1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    uint64_t fsize = (uint64_t)st.st_size;
    uint8_t *host = (uint8_t *)mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (host == MAP_FAILED) { fprintf(stderr, "fucina: assistant mmap failed\n"); return -1; }

    int ok = 1;
    ok &= cudaMalloc(&eng->mtp.d_w, fsize) == cudaSuccess;
    if (ok) ok &= cudaMemcpy(eng->mtp.d_w, host, fsize, cudaMemcpyHostToDevice) == cudaSuccess;

    uint64_t n_el = 0;
    // Matrix weights are EITHER Q8_0 (the 12B head) OR Q4_0 (the 31B QAT head) — the mmvq
    // path handles both; we detect the format from the first matrix tensor and require every
    // matrix tensor to match it. Norms/scales/rope_freqs are F32. Reject anything else instead
    // of silently producing garbage drafts. mtp.wfmt is later passed to every drafter gemv.
    uint32_t mtp_mat_ty = 0;   // 0 = not yet seen; first matrix tensor sets it
    // MTP_FIND  → a matrix (Q8_0 or Q4_0; consistent across all matrix tensors)
    // MTP_FINDF → an F32 norm/scale
    #define MTP_FIND_T(dst, want_type, namefmt, ...) do { \
        char _nm[128]; snprintf(_nm, sizeof(_nm), namefmt, ##__VA_ARGS__); \
        uint64_t _off = 0; uint32_t _ty = 0; \
        if (gguf_find_tensor(host, fsize, _nm, &_off, &n_el, &_ty) != 0) { \
            fprintf(stderr, "fucina: assistant tensor missing: %s\n", _nm); ok = 0; \
        } else if (_ty != (uint32_t)(want_type)) { \
            fprintf(stderr, "fucina: assistant tensor %s has type %u (want %u)\n", \
                    _nm, _ty, (uint32_t)(want_type)); ok = 0; \
        } else { (dst) = _off; } \
    } while (0)
    #define MTP_FIND(dst, namefmt, ...) do { \
        char _nm[128]; snprintf(_nm, sizeof(_nm), namefmt, ##__VA_ARGS__); \
        uint64_t _off = 0; uint32_t _ty = 0; \
        if (gguf_find_tensor(host, fsize, _nm, &_off, &n_el, &_ty) != 0) { \
            fprintf(stderr, "fucina: assistant tensor missing: %s\n", _nm); ok = 0; \
        } else if (_ty != GGML_TYPE_Q8_0 && _ty != GGML_TYPE_Q4_0) { \
            fprintf(stderr, "fucina: assistant matrix %s has type %u (want Q8_0/Q4_0)\n", \
                    _nm, _ty); ok = 0; \
        } else if (mtp_mat_ty && _ty != mtp_mat_ty) { \
            fprintf(stderr, "fucina: assistant matrix %s type %u != %u (mixed quant)\n", \
                    _nm, _ty, mtp_mat_ty); ok = 0; \
        } else { mtp_mat_ty = _ty; (dst) = _off; } \
    } while (0)
    #define MTP_FINDF(dst, namefmt, ...) MTP_FIND_T(dst, GGML_TYPE_F32,  namefmt, ##__VA_ARGS__)

    if (ok) {
        MTP_FIND(eng->mtp.pre_proj,   "nextn.pre_projection.weight");
        MTP_FIND(eng->mtp.post_proj,  "nextn.post_projection.weight");
        MTP_FIND(eng->mtp.tok_embd,   "token_embd.weight");
        MTP_FINDF(eng->mtp.out_norm,   "output_norm.weight");
        MTP_FINDF(eng->mtp.rope_freqs, "rope_freqs.weight");
        for (int l = 0; l < GEMMA4_MTP_LAYERS; l++) {
            MTP_FINDF(eng->mtp.attn_norm[l],      "blk.%d.attn_norm.weight", l);
            MTP_FIND(eng->mtp.wq[l],             "blk.%d.attn_q.weight", l);
            MTP_FIND(eng->mtp.wo[l],             "blk.%d.attn_output.weight", l);
            MTP_FINDF(eng->mtp.q_norm[l],         "blk.%d.attn_q_norm.weight", l);
            MTP_FINDF(eng->mtp.post_attn_norm[l], "blk.%d.post_attention_norm.weight", l);
            MTP_FINDF(eng->mtp.ffn_norm[l],       "blk.%d.ffn_norm.weight", l);
            MTP_FIND(eng->mtp.gate[l],           "blk.%d.ffn_gate.weight", l);
            MTP_FIND(eng->mtp.up[l],             "blk.%d.ffn_up.weight", l);
            MTP_FIND(eng->mtp.down[l],           "blk.%d.ffn_down.weight", l);
            MTP_FINDF(eng->mtp.post_ffw_norm[l],  "blk.%d.post_ffw_norm.weight", l);
            uint64_t so = 0;
            MTP_FINDF(so, "blk.%d.layer_output_scale.weight", l);
            eng->mtp.out_scale[l] = 1.0f;
            if (ok) memcpy(&eng->mtp.out_scale[l], host + so, sizeof(float));
            // assistant pattern: sliding,sliding,sliding,global — verified against the
            // q_norm width below (256 sliding / 512 global) instead of trusting metadata.
            eng->mtp.is_global[l] = 0;
            if (ok) {
                uint64_t qoff = 0; uint32_t qtype = 0; uint64_t qn = 0;
                char nm[64]; snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", l);
                if (gguf_find_tensor(host, fsize, nm, &qoff, &qn, &qtype) == 0)
                    eng->mtp.is_global[l] = (qn == GEMMA4_GLOBAL_HEAD_DIM);
            }
        }
    }
    // Drafter gemv weight format: Q8_0 (12B head) or Q4_0 (31B QAT head). The MTP weights
    // live in eng->mtp.d_w (NOT eng->d_weights), so use_packed_q4 is false and gemv_w takes
    // the native dp4a MMVQ path for whichever format — bit-correct for any weight pointer.
    eng->mtp.wfmt = (mtp_mat_ty == GGML_TYPE_Q4_0) ? FORMAT_Q4_0 : FORMAT_Q8_0;
    #undef MTP_FIND
    #undef MTP_FINDF
    #undef MTP_FIND_T

    // scratch
    // Target-interfacing scratch is sized from the RUNTIME target config (31B hidden=5376,
    // n_heads=32) — NOT the 12B #defines, which would under-allocate and corrupt memory when
    // gemma4_decode copies cfg.hidden_size floats into d_mtp_h / the drafter writes 32 heads.
    const int MH  = eng->cfg.hidden_size;
    const int MNH = eng->cfg.n_heads;
    ok &= cudaMalloc(&eng->d_mtp_h,    MH   * sizeof(float)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_mtp_xh,   2*MH * sizeof(float)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_mtp_cur,  GEMMA4_MTP_HIDDEN    * sizeof(float)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_mtp_t1,   GEMMA4_MTP_HIDDEN    * sizeof(float)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_mtp_t2,   GEMMA4_MTP_HIDDEN    * sizeof(float)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_mtp_q,    MNH*GEMMA4_GLOBAL_HEAD_DIM * sizeof(float)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_mtp_attn, MNH*GEMMA4_GLOBAL_HEAD_DIM * sizeof(float)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_mtp_ffa,  GEMMA4_MTP_FFN       * sizeof(float)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_mtp_ffb,  GEMMA4_MTP_FFN       * sizeof(float)) == cudaSuccess;
    if (!eng->d_sample_id)
        ok &= cudaMalloc(&eng->d_sample_id, sizeof(int)) == cudaSuccess;
    if (!eng->d_sample_p)
        ok &= cudaMalloc(&eng->d_sample_p, sizeof(float)) == cudaSuccess;
    // Device-chained draft scratch (b): maxd ids + confidences, one sync/step.
    if (!eng->d_mtp_ids)
        ok &= cudaMalloc(&eng->d_mtp_ids, GEMMA4_SPEC_MAX*sizeof(int)) == cudaSuccess;
    if (!eng->d_mtp_conf)
        ok &= cudaMalloc(&eng->d_mtp_conf, GEMMA4_SPEC_MAX*sizeof(float)) == cudaSuccess;
    // CUDA-graph draft scratch (c): device-resident input token + chain position.
    if (!eng->d_mtp_tok)
        ok &= cudaMalloc(&eng->d_mtp_tok, sizeof(int32_t)) == cudaSuccess;
    if (!eng->d_mtp_pos)
        ok &= cudaMalloc(&eng->d_mtp_pos, sizeof(int)) == cudaSuccess;

    munmap(host, fsize);
    if (!ok) {
        if (eng->mtp.d_w) { cudaFree(eng->mtp.d_w); eng->mtp.d_w = NULL; }
        fprintf(stderr, "fucina: assistant load FAILED — MTP drafting disabled\n");
        return -1;
    }
    eng->mtp.loaded = 1;
    eng->mtp_h_valid = 0;
    fprintf(stderr, "fucina: MTP assistant loaded (%s, %.0f MB) — draft-mtp speculation ON\n",
            path, fsize / (1024.0 * 1024.0));
    if (getenv("FUCINA_SPEC_BATCH_SELFTEST")) gemma4_engine_spec_batch_selftest(eng);
    return 0;
}

// One assistant forward: (tok, d_mtp_h) → logits in eng->d_logits + next h in d_mtp_h.
// All draft tokens use RoPE position n_past and attend the frozen committed target KV.
//
// (c) CUDA-graph form: ALL per-call state is read from DEVICE memory — the input
// token from tok_ptr (eng->d_mtp_tok, refreshed by mtp_argmax_conf_kernel between
// chained tokens) and the position from pos_ptr (eng->d_mtp_pos, one 4-byte H2D per
// draft call) — so the ~57-kernel launch sequence is capturable ONCE and replayed
// for every drafted token of every spec step. The attention launches use the
// rows-kernels' device-n_tokens override with a FIXED MAX_SPLITS grid (blocks past
// the in-kernel split count exit immediately), since a graph cannot change grids.
static void mtp_forward(gemma4_engine_t *eng, const int32_t *tok_ptr,
                        const int *pos_ptr, cudaStream_t stream)
{
    // H is the TARGET hidden (3840 @12B / 5376 @31B) — the MTP head's pre_projection
    // in-dim is 2*H (embed‖h) and its post_projection out-dim is H. AH/FF are the
    // MTP-INTERNAL widths (1024/8192) and stay constant across target sizes. NH is the
    // target's query-head count (16/32): the drafter's Q-only attention runs NH heads
    // against the target's KV.
    const int H = eng->cfg.hidden_size, AH = GEMMA4_MTP_HIDDEN, FF = GEMMA4_MTP_FFN;
    const int NH = eng->cfg.n_heads;
    const int smem32 = 32 * (int)sizeof(float);

    // x = target_embd(tok)·√H  ‖  h   → pre_projection (2H→1024) → cur [1024]
    // The MTP head shares the TARGET model's token embedding — so embed via embed_w, which reads
    // the BF16 table (d_embed_bf16) for NVFP4 and the Q8_0 table otherwise. The previous hard
    // Q8_0 lookup deref'd a NULL d_weights under NVFP4, garbaging h → the drafter always declined.
    embed_w(eng, eng->d_mtp_xh, weight_fp8(eng, eng->tensors.token_embd), tok_ptr, 1, H, stream);
    scale_kernel<<<(H+255)/256, 256, 0, stream>>>(eng->d_mtp_xh, H, sqrtf((float)H));
    cudaMemcpyAsync(eng->d_mtp_xh + H, eng->d_mtp_h, H * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    gemv_w(eng, eng->d_mtp_cur, mtp_w(eng, eng->mtp.pre_proj), eng->d_mtp_xh,
           2*H, AH, stream, eng->mtp.wfmt);

    for (int l = 0; l < GEMMA4_MTP_LAYERS; l++) {
        const int is_g     = eng->mtp.is_global[l];
        const int head_dim = is_g ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_HEAD_DIM;
        const int qdim     = NH * head_dim;   // NH = target query heads (16 @12B / 32 @31B)

        // attention (Q-only; K/V come from the target's cache)
        rms_norm_kernel<<<1, 256, smem32, stream>>>(
            eng->d_mtp_t1, eng->d_mtp_cur, mtp_f32(eng, eng->mtp.attn_norm[l]), AH, GEMMA4_RMS_EPS);
        gemv_w(eng, eng->d_mtp_q, mtp_w(eng, eng->mtp.wq[l]), eng->d_mtp_t1,
               AH, qdim, stream, eng->mtp.wfmt);
        per_head_rms_norm_kernel<<<NH, head_dim, smem32, stream>>>(
            eng->d_mtp_q, mtp_f32(eng, eng->mtp.q_norm[l]), head_dim, GEMMA4_RMS_EPS);
        if (is_g) {
            rope_global_kernel<<<NH, head_dim/2, 0, stream>>>(
                eng->d_mtp_q, eng->d_mtp_q, 0, pos_ptr, 0, NH, /*n_kv_heads=*/0,
                head_dim, eng->cfg.rope_theta_global, mtp_f32(eng, eng->mtp.rope_freqs));
            // target layer 47 = the LAST global layer (llama.cpp share(il)=n_layer-1).
            // Fixed-grid split-K with device n_tokens (= *pos_ptr): same split formula
            // as global_attn_decode_broadcast, so the math is bit-identical to it.
            int slot = eng->global_slot[eng->cfg.n_layers - 1];
            size_t lstride = (size_t)eng->global_kv_capacity
                           * (size_t)eng->cfg.n_kv_global * GEMMA4_GLOBAL_HEAD_DIM;
            if (eng->cfg.n_heads == 32 && eng->cfg.n_kv_global == 4) {
                global_attn_splitk_rows_kernel<32, 4, GEMMA4_GLOBAL_HEAD_DIM>
                    <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), 32*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_mtp_q,
                        eng->d_global_k + (size_t)slot * lstride,
                        eng->d_global_v + (size_t)slot * lstride,
                        0, pos_ptr);
                flash_decode_combine_rows_kernel<32>
                    <<<dim3(32, 1), head_dim, 0, stream>>>(
                        eng->d_mtp_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                        head_dim, /*window=*/0, 0, pos_ptr, 32*head_dim);
            } else {
            global_attn_splitk_rows_kernel<GEMMA4_HEADS, 1, GEMMA4_GLOBAL_HEAD_DIM>
                <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), GEMMA4_HEADS*32, 0, stream>>>(
                    eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_mtp_q,
                    eng->d_global_k + (size_t)slot * lstride,
                    eng->d_global_v + (size_t)slot * lstride,
                    0, pos_ptr);
            flash_decode_combine_rows_kernel<GEMMA4_HEADS>
                <<<dim3(GEMMA4_HEADS, 1), head_dim, 0, stream>>>(
                    eng->d_mtp_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                    head_dim, /*window=*/0, 0, pos_ptr, GEMMA4_HEADS*head_dim);
            }
        } else {
            rope_sliding_kernel<<<NH, head_dim/2, 0, stream>>>(
                eng->d_mtp_q, eng->d_mtp_q, 0, pos_ptr, NH, /*n_kv_heads=*/0,
                head_dim, eng->cfg.rope_theta_sliding);
            // target layer 46 = the LAST sliding layer (share(il)=n_layer-2); the
            // drafter attends window-1 keys of the frozen cache at n_tokens = *pos_ptr.
            size_t lstride = (size_t)eng->sliding_kv_capacity * eng->cfg.n_kv_sliding * GEMMA4_HEAD_DIM;
            if (eng->cfg.n_heads == 32 && eng->cfg.n_kv_sliding == 16) {
                sliding_attn_splitk_rows_kernel<32, 16, GEMMA4_HEAD_DIM>
                    <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), 16*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_mtp_q,
                        eng->d_sliding_k + (size_t)(eng->cfg.n_layers - 2) * lstride,
                        eng->d_sliding_v + (size_t)(eng->cfg.n_layers - 2) * lstride,
                        GEMMA4_SLIDING_WINDOW - 1, 0, eng->sliding_kv_capacity, pos_ptr);
                flash_decode_combine_rows_kernel<32>
                    <<<dim3(32, 1), head_dim, 0, stream>>>(
                        eng->d_mtp_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                        head_dim, GEMMA4_SLIDING_WINDOW - 1, 0, pos_ptr, 32*head_dim);
            } else {
            sliding_attn_splitk_rows_kernel<GEMMA4_HEADS, GEMMA4_KV_HEADS, GEMMA4_HEAD_DIM>
                <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), GEMMA4_KV_HEADS*32, 0, stream>>>(
                    eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_mtp_q,
                    eng->d_sliding_k + (size_t)(eng->cfg.n_layers - 2) * lstride,
                    eng->d_sliding_v + (size_t)(eng->cfg.n_layers - 2) * lstride,
                    GEMMA4_SLIDING_WINDOW - 1, 0, eng->sliding_kv_capacity, pos_ptr);
            flash_decode_combine_rows_kernel<GEMMA4_HEADS>
                <<<dim3(GEMMA4_HEADS, 1), head_dim, 0, stream>>>(
                    eng->d_mtp_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                    head_dim, GEMMA4_SLIDING_WINDOW - 1, 0, pos_ptr, GEMMA4_HEADS*head_dim);
            }
        }
        gemv_w(eng, eng->d_mtp_t1, mtp_w(eng, eng->mtp.wo[l]), eng->d_mtp_attn,
               qdim, AH, stream, eng->mtp.wfmt);

        rms_norm_kernel<<<1, 256, smem32, stream>>>(
            eng->d_mtp_t2, eng->d_mtp_t1, mtp_f32(eng, eng->mtp.post_attn_norm[l]), AH, GEMMA4_RMS_EPS);
        residual_add_kernel<<<(AH+255)/256, 256, 0, stream>>>(eng->d_mtp_t2, eng->d_mtp_cur, AH);

        // FFN (GeGLU 1024 → 8192 → 1024)
        rms_norm_kernel<<<1, 256, smem32, stream>>>(
            eng->d_mtp_t1, eng->d_mtp_t2, mtp_f32(eng, eng->mtp.ffn_norm[l]), AH, GEMMA4_RMS_EPS);
        gemv_w(eng, eng->d_mtp_ffa, mtp_w(eng, eng->mtp.gate[l]), eng->d_mtp_t1,
               AH, FF, stream, eng->mtp.wfmt);
        gemv_w(eng, eng->d_mtp_ffb, mtp_w(eng, eng->mtp.up[l]), eng->d_mtp_t1,
               AH, FF, stream, eng->mtp.wfmt);
        geglu_kernel<<<(FF+255)/256, 256, 0, stream>>>(eng->d_mtp_ffa, eng->d_mtp_ffa, eng->d_mtp_ffb, FF);
        gemv_w(eng, eng->d_mtp_t1, mtp_w(eng, eng->mtp.down[l]), eng->d_mtp_ffa,
               FF, AH, stream, eng->mtp.wfmt);
        rms_norm_kernel<<<1, 256, smem32, stream>>>(
            eng->d_mtp_cur, eng->d_mtp_t1, mtp_f32(eng, eng->mtp.post_ffw_norm[l]), AH, GEMMA4_RMS_EPS);
        residual_add_kernel<<<(AH+255)/256, 256, 0, stream>>>(eng->d_mtp_cur, eng->d_mtp_t2, AH);
        if (eng->mtp.out_scale[l] != 1.0f)
            scale_kernel<<<(AH+255)/256, 256, 0, stream>>>(eng->d_mtp_cur, AH, eng->mtp.out_scale[l]);
    }

    // output norm → assistant unembed (NO softcap — the assistant has none) + next h
    rms_norm_kernel<<<1, 256, smem32, stream>>>(
        eng->d_mtp_t1, eng->d_mtp_cur, mtp_f32(eng, eng->mtp.out_norm), AH, GEMMA4_RMS_EPS);
    gemv_w(eng, eng->d_logits, mtp_w(eng, eng->mtp.tok_embd), eng->d_mtp_t1,
           AH, GEMMA4_VOCAB_SIZE, stream, eng->mtp.wfmt);
    gemv_w(eng, eng->d_mtp_h, mtp_w(eng, eng->mtp.post_proj), eng->d_mtp_t1,
           AH, H, stream, eng->mtp.wfmt);
}

// Paged, per-slot variant of mtp_forward for the continuous-batch spec path. Identical
// MTP head math, except: (1) the recurrent hidden is the caller's per-slot buffer `d_h`
// (read as the ‖h half of the pre-projection, overwritten with next-h) instead of the
// shared eng->d_mtp_h, and (2) the Q-only attention reads the slot's LAST sliding / LAST
// global layer KV from the PAGED pools through a B=1 PagedSeqView (`vslid`/`vglob`, frozen
// at the committed length) using the same split-K batched kernels the verify path uses —
// the contiguous single-seq cache is empty for batch sequences. Logits land in eng->d_logits
// (drafting is sequential per slot, so the shared logit/scratch buffers are reused safely).
// Exactness is NOT required here: the verify pass re-derives every emitted token from the
// target, so a divergent draft only lowers the accept rate, never changes the output.
static void mtp_forward_paged(gemma4_engine_t *eng, const int32_t *tok_ptr, const int *pos_ptr,
                              float *d_h, const PagedSeqView *vslid, const PagedSeqView *vglob,
                              cudaStream_t stream)
{
    const int H = eng->cfg.hidden_size, AH = GEMMA4_MTP_HIDDEN, FF = GEMMA4_MTP_FFN;
    const int NH = eng->cfg.n_heads;
    const int smem32 = 32 * (int)sizeof(float);

    embed_w(eng, eng->d_mtp_xh, weight_fp8(eng, eng->tensors.token_embd), tok_ptr, 1, H, stream);
    scale_kernel<<<(H+255)/256, 256, 0, stream>>>(eng->d_mtp_xh, H, sqrtf((float)H));
    cudaMemcpyAsync(eng->d_mtp_xh + H, d_h, H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    gemv_w(eng, eng->d_mtp_cur, mtp_w(eng, eng->mtp.pre_proj), eng->d_mtp_xh, 2*H, AH, stream, eng->mtp.wfmt);

    for (int l = 0; l < GEMMA4_MTP_LAYERS; l++) {
        const int is_g     = eng->mtp.is_global[l];
        const int head_dim = is_g ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_HEAD_DIM;
        const int qdim     = NH * head_dim;

        rms_norm_kernel<<<1, 256, smem32, stream>>>(
            eng->d_mtp_t1, eng->d_mtp_cur, mtp_f32(eng, eng->mtp.attn_norm[l]), AH, GEMMA4_RMS_EPS);
        gemv_w(eng, eng->d_mtp_q, mtp_w(eng, eng->mtp.wq[l]), eng->d_mtp_t1, AH, qdim, stream, eng->mtp.wfmt);
        per_head_rms_norm_kernel<<<NH, head_dim, smem32, stream>>>(
            eng->d_mtp_q, mtp_f32(eng, eng->mtp.q_norm[l]), head_dim, GEMMA4_RMS_EPS);
        if (is_g) {
            rope_global_kernel<<<NH, head_dim/2, 0, stream>>>(
                eng->d_mtp_q, eng->d_mtp_q, 0, pos_ptr, 0, NH, /*n_kv_heads=*/0,
                head_dim, eng->cfg.rope_theta_global, mtp_f32(eng, eng->mtp.rope_freqs));
            int gslot = eng->global_slot[eng->cfg.n_layers - 1];
            size_t okv = (size_t)eng->cfg.n_kv_global * GEMMA4_GLOBAL_HEAD_DIM;
            size_t pls = (size_t)eng->glob_pool.n_blocks * PAGED_KV_BLOCK_TOKENS * okv;
            const pkv_t *pk = eng->d_glob_pool_k + (size_t)gslot * pls;
            const pkv_t *pv = eng->d_glob_pool_v + (size_t)gslot * pls;
            if (eng->cfg.n_heads == 32 && eng->cfg.n_kv_global == 4) {
                paged_global_attn_splitk_batched<32, 4, GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), 32*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_mtp_q, pk, pv, vglob,
                        GEMMA4_GLOBAL_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, (int)okv);
                paged_flash_decode_combine_batched<32, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(32, 1), head_dim, 0, stream>>>(
                        eng->d_mtp_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, vglob,
                        head_dim, /*window=*/0, GEMMA4_GLOBAL_SPLIT_CHUNK);
            } else {
                paged_global_attn_splitk_batched<GEMMA4_HEADS, GEMMA4_GLOBAL_KV_HEADS,
                                                 GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), GEMMA4_HEADS*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_mtp_q, pk, pv, vglob,
                        GEMMA4_GLOBAL_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, (int)okv);
                paged_flash_decode_combine_batched<GEMMA4_HEADS, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(GEMMA4_HEADS, 1), head_dim, 0, stream>>>(
                        eng->d_mtp_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, vglob,
                        head_dim, /*window=*/0, GEMMA4_GLOBAL_SPLIT_CHUNK);
            }
        } else {
            rope_sliding_kernel<<<NH, head_dim/2, 0, stream>>>(
                eng->d_mtp_q, eng->d_mtp_q, 0, pos_ptr, NH, /*n_kv_heads=*/0,
                head_dim, eng->cfg.rope_theta_sliding);
            size_t okv = (size_t)eng->cfg.n_kv_sliding * GEMMA4_HEAD_DIM;
            size_t pls = (size_t)eng->slid_pool.n_blocks * PAGED_KV_BLOCK_TOKENS * okv;
            const pkv_t *pk = eng->d_slid_pool_k + (size_t)(eng->cfg.n_layers - 2) * pls;
            const pkv_t *pv = eng->d_slid_pool_v + (size_t)(eng->cfg.n_layers - 2) * pls;
            const int win = GEMMA4_SLIDING_WINDOW - 1;   // frozen cache: window-1 prior keys
            if (eng->cfg.n_heads == 32 && eng->cfg.n_kv_sliding == 16) {
                paged_sliding_attn_splitk_batched<32, 16, GEMMA4_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), 16*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_mtp_q, pk, pv, vslid,
                        win, GEMMA4_SLIDING_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, (int)okv);
                paged_flash_decode_combine_batched<32, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(32, 1), head_dim, 0, stream>>>(
                        eng->d_mtp_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, vslid,
                        head_dim, win, GEMMA4_SLIDING_SPLIT_CHUNK);
            } else {
                paged_sliding_attn_splitk_batched<GEMMA4_HEADS, GEMMA4_KV_HEADS,
                                                  GEMMA4_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), GEMMA4_KV_HEADS*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_mtp_q, pk, pv, vslid,
                        win, GEMMA4_SLIDING_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, (int)okv);
                paged_flash_decode_combine_batched<GEMMA4_HEADS, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(GEMMA4_HEADS, 1), head_dim, 0, stream>>>(
                        eng->d_mtp_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, vslid,
                        head_dim, win, GEMMA4_SLIDING_SPLIT_CHUNK);
            }
        }
        gemv_w(eng, eng->d_mtp_t1, mtp_w(eng, eng->mtp.wo[l]), eng->d_mtp_attn, qdim, AH, stream, eng->mtp.wfmt);

        rms_norm_kernel<<<1, 256, smem32, stream>>>(
            eng->d_mtp_t2, eng->d_mtp_t1, mtp_f32(eng, eng->mtp.post_attn_norm[l]), AH, GEMMA4_RMS_EPS);
        residual_add_kernel<<<(AH+255)/256, 256, 0, stream>>>(eng->d_mtp_t2, eng->d_mtp_cur, AH);

        rms_norm_kernel<<<1, 256, smem32, stream>>>(
            eng->d_mtp_t1, eng->d_mtp_t2, mtp_f32(eng, eng->mtp.ffn_norm[l]), AH, GEMMA4_RMS_EPS);
        gemv_w(eng, eng->d_mtp_ffa, mtp_w(eng, eng->mtp.gate[l]), eng->d_mtp_t1, AH, FF, stream, eng->mtp.wfmt);
        gemv_w(eng, eng->d_mtp_ffb, mtp_w(eng, eng->mtp.up[l]), eng->d_mtp_t1, AH, FF, stream, eng->mtp.wfmt);
        geglu_kernel<<<(FF+255)/256, 256, 0, stream>>>(eng->d_mtp_ffa, eng->d_mtp_ffa, eng->d_mtp_ffb, FF);
        gemv_w(eng, eng->d_mtp_t1, mtp_w(eng, eng->mtp.down[l]), eng->d_mtp_ffa, FF, AH, stream, eng->mtp.wfmt);
        rms_norm_kernel<<<1, 256, smem32, stream>>>(
            eng->d_mtp_cur, eng->d_mtp_t1, mtp_f32(eng, eng->mtp.post_ffw_norm[l]), AH, GEMMA4_RMS_EPS);
        residual_add_kernel<<<(AH+255)/256, 256, 0, stream>>>(eng->d_mtp_cur, eng->d_mtp_t2, AH);
        if (eng->mtp.out_scale[l] != 1.0f)
            scale_kernel<<<(AH+255)/256, 256, 0, stream>>>(eng->d_mtp_cur, AH, eng->mtp.out_scale[l]);
    }

    rms_norm_kernel<<<1, 256, smem32, stream>>>(
        eng->d_mtp_t1, eng->d_mtp_cur, mtp_f32(eng, eng->mtp.out_norm), AH, GEMMA4_RMS_EPS);
    gemv_w(eng, eng->d_logits, mtp_w(eng, eng->mtp.tok_embd), eng->d_mtp_t1,
           AH, GEMMA4_VOCAB_SIZE, stream, eng->mtp.wfmt);
    gemv_w(eng, d_h, mtp_w(eng, eng->mtp.post_proj), eng->d_mtp_t1, AH, H, stream, eng->mtp.wfmt);
}

// Fused argmax + top-1 softmax probability of the drafter's logits in ONE pass:
// p(top1) = 1 / Σ_i exp(l_i − l_max) (online max/sum, no second sweep). The prob
// feeds ONLY the drafter's confidence gate — verification still samples every
// position from the TARGET distribution, so output exactness never depends on it.
// Tie-break may differ from gemma4_sample_argmax on bit-equal logits; harmless
// for a draft (any proposal is checked).
__global__ void mtp_argmax_conf_kernel(
    const float *logits, int V, int *out_id, float *out_p, int32_t *out_tok)
{
    const int T = blockDim.x, tid = threadIdx.x;
    float m = -INFINITY, s = 0.0f; int id = -1;
    for (int i = tid; i < V; i += T) {
        float v = logits[i];
        if (v > m) { s = s * __expf(m - v) + 1.0f; m = v; id = i; }
        else        s += __expf(v - m);
    }
    __shared__ float sm[1024], ss[1024];
    __shared__ int   sid[1024];
    sm[tid] = m; ss[tid] = s; sid[tid] = id;
    __syncthreads();
    for (int off = T / 2; off > 0; off >>= 1) {
        if (tid < off) {
            float m2 = sm[tid + off], s2 = ss[tid + off];
            if (m2 > sm[tid]) {
                ss[tid] = ss[tid] * __expf(sm[tid] - m2) + s2;
                sm[tid] = m2; sid[tid] = sid[tid + off];
            } else {
                ss[tid] += s2 * __expf(m2 - sm[tid]);
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out_id = sid[0];
        *out_p  = 1.0f / ss[0];
        if (out_tok) *out_tok = sid[0];   // feeds the next graph replay's embed
    }
}

// Drafter confidence gate (llama.cpp common/speculative.cpp "only collect very
// high-confidence draft tokens", --draft-p-min). Measured here on chat-template
// text (the REPL/server workload) the assistant's blind argmax chain accepted
// only ~55% of proposed tokens — each rejected slot still costs ~0.23x a decode
// in the batched verify plus a full ~444MB assistant pass, which capped spec at
// ~1.1-1.2x over plain decode. (The historical "84-94% acceptance" numbers came
// from RAW one-shot prompts, where the instruction-tuned QAT model degenerates
// into trivially-predictable repetition.) Cutting the draft at the first low-
// confidence position keeps only the high-acceptance subset. Measured (greedy
// chat-template essay, 400 tok): acceptance 55%→77%, 20.3→22.4 tok/s = 1.35x the
// plain-decode REPL baseline (16.6); the easy prose/code regimes IMPROVE slightly
// (90→95%, 31.2→31.9 tok/s) since confidence stays high there and the gate only
// trims the bad tails. Threshold swept 0.60/0.75/0.85 → 21.7/22.4/22.0 tok/s.
#define GEMMA4_MTP_PMIN 0.00f

// ── Tree-spec de-risk diagnostic (FUCINA_SPEC_TOPK_DIAG=1) ──────────────────
// Measures, at every draw-and-match REJECT (target sampled a token != the MTP head's
// argmax), whether that target token was nonetheless in the head's top-4. A high hit
// rate ⇒ a width-w token TREE would recover the reject ⇒ tree-spec is worth building.
// Zero cost when off. g_diag_top4[step] = the head's top-4 ids at that draft step.
static int   g_topk_diag = -1;
static int   g_diag_top4[GEMMA4_SPEC_MAX][4];
static long  g_diag_rej = 0, g_diag_hit2 = 0, g_diag_hit3 = 0, g_diag_hit4 = 0;
__global__ void mask_one_neg_inf_kernel(float *logits, const int *idx) {
    if (threadIdx.x == 0) logits[*idx] = -INFINITY;
}

// =========================================================================
// ─── Tree speculative decode (T1) — KV-free drafter ─────────────────────
// =========================================================================
// The MTP head is a PURE recurrence in h: mtp_forward(tok, h) → (children logits, h').
// Its attention reads the FROZEN target prefix at a FIXED position, never drafted tokens
// (docs/tree-spec-plan.md). So a token TREE is drafted with NO KV bookkeeping: each node N
// runs the head once on (tok[N], h[N]); all of N's children inherit the SAME output h'[N]
// and differ only by token (the top-w of N's logits). Verify (T1b) forwards the whole tree
// in ONE target weight pass under an ancestor mask. Acceptance stays per-edge draw-and-match,
// so the target distribution is preserved exactly — at temp=0 a width-1 tree is byte-identical
// to the linear chain (the correctness gate). spec_tree / GEMMA4_TREE_MAX are declared above
// (before decode_tree_dev, which consumes the same struct).

// Build a static template: a depth-`depth` rank-0 trunk, plus at each trunk node shallower
// than `branch_until` add (width-1) sibling branches (the parent's ranks 1..width-1), each
// continuing rank-0 for `branch_len` more steps. width=1 ⇒ a pure linear chain (the T1a gate).
// Fills topology only (parent/depth/nchild/anc + node count); tokens are filled by the drafter.
// `rank[i]` = which of the parent's top-w tokens node i takes (0 = argmax trunk).
static int tree_build_template(spec_tree *t, int rank_out[GEMMA4_TREE_MAX],
                               int depth, int width, int branch_until, int branch_len)
{
    for (int i = 0; i < GEMMA4_TREE_MAX; i++) { t->parent[i] = -1; t->depth[i] = 0;
        t->nchild[i] = 0; t->anc[i] = 0; rank_out[i] = 0; }
    int n = 1; t->parent[0] = -1; t->depth[0] = 0; t->anc[0] = 1u;   // root
    int trunk_prev = 0;
    for (int d = 1; d <= depth && n < GEMMA4_TREE_MAX; d++) {
        int trunk = n++;                                   // rank-0 continuation of the trunk
        t->parent[trunk] = trunk_prev; t->depth[trunk] = d; rank_out[trunk] = 0;
        t->anc[trunk] = t->anc[trunk_prev] | (1u << trunk);
        t->nchild[trunk_prev]++;
        // sibling branches off the SAME parent (alternative tokens at this depth)
        if (d <= branch_until) {
            for (int w = 1; w < width && n < GEMMA4_TREE_MAX; w++) {
                int br = n++; int bprev = br;
                t->parent[br] = trunk_prev; t->depth[br] = d; rank_out[br] = w;
                t->anc[br] = t->anc[trunk_prev] | (1u << br);
                t->nchild[trunk_prev]++;
                for (int k = 1; k <= branch_len && n < GEMMA4_TREE_MAX; k++) {
                    int c = n++;
                    t->parent[c] = bprev; t->depth[c] = d + k; rank_out[c] = 0;
                    t->anc[c] = t->anc[bprev] | (1u << c);
                    t->nchild[bprev]++; bprev = c;
                }
            }
        }
        trunk_prev = trunk;
    }
    t->n = n;
    return n;
}

static int mtp_graph_ensure(gemma4_engine_t *eng);   // fwd decl (defined with the linear drafter)

// Device scratch for the per-node output h (h'[N], the children's input h) and the per-node
// token (chained ON-DEVICE so the drafter never syncs mid-tree). Lazily sized.
static float   *g_tree_hout   = NULL;   // [GEMMA4_TREE_MAX][H]
static int32_t *g_d_treetok   = NULL;   // [GEMMA4_TREE_MAX] node tokens (device)

// Draft the tree: fill t->tok by running the head once per node in parent-before-child order.
// T2: FULLY ON-DEVICE — each node's children are argmax'd straight into g_d_treetok[child]
// (which the child's own forward then reads as input), and h flows through g_tree_hout. The
// ONLY host sync is a single D2H of the n tokens at the very end. (The old version synced once
// PER CHILD — ~30 host round-trips/step — which dominated the step cost.) Returns node count.
static int mtp_draft_tree(gemma4_engine_t *eng, int32_t g, spec_tree *t,
                          const int rank[GEMMA4_TREE_MAX])
{
    if (!eng->mtp.loaded || !eng->mtp_h_valid || eng->cur.n_tokens <= 0) return 0;
    cudaStream_t stream = eng->stream;
    const int H = eng->cfg.hidden_size;
    if (!g_tree_hout && cudaMalloc(&g_tree_hout, (size_t)GEMMA4_TREE_MAX * H * sizeof(float)) != cudaSuccess)
        { cudaGetLastError(); return 0; }
    if (!g_d_treetok && cudaMalloc(&g_d_treetok, GEMMA4_TREE_MAX * sizeof(int32_t)) != cudaSuccess)
        { cudaGetLastError(); return 0; }
    int posv = eng->cur.n_tokens;
    cudaMemcpyAsync(eng->d_mtp_pos, &posv, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(g_d_treetok, &g, sizeof(int32_t), cudaMemcpyHostToDevice, stream);  // root token
    t->tok[0] = g;
    // Each node's head forward is replayed from the captured mtp_graph (ONE op/node) instead of
    // ~48 raw kernel launches — the head's kernels are tiny/latency-bound, so per-node launches
    // dominated the step. The graph reads eng->d_mtp_tok + eng->d_mtp_h (set per node below).
    int use_graph = (mtp_graph_ensure(eng) == 0);
    // Process nodes in index order (template: parent index < child index). All on-stream, no sync.
    for (int N = 0; N < t->n; N++) {
        if (t->nchild[N] == 0) continue;                       // leaf: no forward needed
        // Set the graph's inputs: token N (chained on-device) and input h.
        cudaMemcpyAsync(eng->d_mtp_tok, g_d_treetok + N, sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        if (N != 0)                                            // root keeps the committed h in d_mtp_h
            cudaMemcpyAsync(eng->d_mtp_h, g_tree_hout + (size_t)t->parent[N] * H, H * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        if (use_graph && cudaGraphLaunch(eng->mtp_graph, stream) != cudaSuccess) {
            use_graph = 0; cudaGetLastError();
        }
        if (!use_graph)
            mtp_forward(eng, eng->d_mtp_tok, eng->d_mtp_pos, stream);   // → d_logits + h'
        cudaMemcpyAsync(g_tree_hout + (size_t)N * H, eng->d_mtp_h, H * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        // Children = head top-(nchild): argmax straight into g_d_treetok[child], then pop. Siblings
        // are stored in rank order so consecutive argmax+pop yields ranks 0,1,2,… No host sync.
        for (int c = 0; c < t->n; c++) {
            if (t->parent[c] != N) continue;
            argmax_kernel<<<1, 32, 0, stream>>>(eng->d_logits, g_d_treetok + c, GEMMA4_VOCAB_SIZE);
            mask_one_neg_inf_kernel<<<1, 1, 0, stream>>>(eng->d_logits, g_d_treetok + c);
        }
    }
    (void)rank;
    // Single sync: pull all node tokens to host (the verify uploads them; topology needs them).
    int32_t htok[GEMMA4_TREE_MAX];
    cudaMemcpyAsync(htok, g_d_treetok, (size_t)t->n * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) { cudaGetLastError(); return 0; }
    for (int i = 1; i < t->n; i++) t->tok[i] = htok[i];
    return t->n;
}

// Draft up to max_draft tokens with the assistant; returns the count proposed.
// Requires a valid recurrent h (eng->mtp_h_valid, paired with g). Greedy argmax
// draft, cut at the first token whose draft prob < GEMMA4_MTP_PMIN — the verify's
// accept rule is what preserves the target distribution.
//
// (b) Device-chained with early-stop: step j's argmax id (written to d_mtp_ids[j])
// feeds step j+1's embed via the DEVICE pointer and the recurrent h flows through
// d_mtp_h on-device, so no token readback is needed mid-chain. We still read back
// the confidence after each forward to decide whether to keep drafting — the
// assistant forward is ~7% of a target pass, so an extra unwanted forward is NOT
// free, and a measured A/B (chain-then-truncate vs early-stop) showed the wasted
// forwards past the confidence cut cost MORE than the saved syncs on this GPU
// (37.5 vs 38.4 tok/s). Keeping the early-stop, but the id no longer crosses the
// bus (only the 4-byte confidence does) and the next embed reads the device id.
// Lazy one-time capture of the assistant forward into a CUDA graph. The forward's
// per-call state is device-resident (d_mtp_tok / d_mtp_pos), so a single captured
// graph replays for every drafted token of every spec step: ~57 kernel launches
// collapse to one cudaGraphLaunch between the per-token confidence syncs. On any
// capture/instantiate failure the per-kernel launch path stays in service.
static int mtp_graph_ensure(gemma4_engine_t *eng)
{
    if (eng->mtp_graph) return 0;
    if (eng->mtp_graph_failed) return -1;
    cudaStream_t cs = NULL;
    cudaGraph_t  g  = NULL;
    int ok = cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) == cudaSuccess;
    if (ok && cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
        mtp_forward(eng, eng->d_mtp_tok, eng->d_mtp_pos, cs);
        ok = cudaStreamEndCapture(cs, &g) == cudaSuccess && g != NULL;
    } else {
        ok = 0;
    }
    if (ok) ok = cudaGraphInstantiate(&eng->mtp_graph, g, 0) == cudaSuccess;
    if (g)  cudaGraphDestroy(g);
    if (cs) cudaStreamDestroy(cs);
    if (!ok || !eng->mtp_graph) {
        eng->mtp_graph = NULL;
        eng->mtp_graph_failed = 1;
        cudaGetLastError();   // clear any capture-path error state
        fprintf(stderr, "fucina: MTP forward graph capture failed — using per-kernel launches\n");
        return -1;
    }
    return 0;
}

static int mtp_draft(gemma4_engine_t *eng, int32_t g, int32_t *draft_out, int max_draft)
{
    if (!eng->mtp.loaded || !eng->mtp_h_valid || max_draft <= 0) return 0;
    if (eng->cur.n_tokens <= 0) return 0;
    if (max_draft > GEMMA4_SPEC_MAX - 1) max_draft = GEMMA4_SPEC_MAX - 1;
    if (!eng->d_mtp_ids || !eng->d_mtp_conf) return 0;
    if (!eng->d_mtp_tok || !eng->d_mtp_pos) return 0;
    cudaStream_t stream = eng->stream;
    // Draft confidence gate (llama.cpp --draft-p-min): cut the chain at the first token
    // whose draft prob < pmin. Disabled by default (0.0); FUCINA_MTP_PMIN tunes it. The
    // verify accept rule preserves the target distribution regardless of where we cut, so
    // this only trades raw draft length for acceptance precision (a measured win on prose).
    static float pmin = -1.0f;
    if (pmin < 0.0f) { const char *e = getenv("FUCINA_MTP_PMIN"); pmin = e ? (float)atof(e) : GEMMA4_MTP_PMIN; }
    // Per-call device state: the chain position (n_past, same for every draft token)
    // and the first input token. Chained tokens are written into d_mtp_tok on-device
    // by mtp_argmax_conf_kernel — no id ever crosses the bus mid-chain.
    int posv = eng->cur.n_tokens;
    cudaMemcpyAsync(eng->d_mtp_pos, &posv, sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(eng->d_mtp_tok, &g, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    int use_graph = (mtp_graph_ensure(eng) == 0);
    int produced = 0;
    for (int j = 0; j < max_draft; j++) {
        if (use_graph && cudaGraphLaunch(eng->mtp_graph, stream) != cudaSuccess) {
            use_graph = 0;                    // replay failed → per-kernel for the rest
            cudaGetLastError();
        }
        if (!use_graph)
            mtp_forward(eng, eng->d_mtp_tok, eng->d_mtp_pos, stream);
        mtp_argmax_conf_kernel<<<1, 1024, 0, stream>>>(
            eng->d_logits, GEMMA4_VOCAB_SIZE, eng->d_mtp_ids + j, eng->d_mtp_conf + j,
            eng->d_mtp_tok);
        float conf = 0.0f;
        cudaMemcpyAsync(&conf, eng->d_mtp_conf + j, sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        if (cudaStreamSynchronize(stream) != cudaSuccess) break;
        if (conf < pmin) break;   // low confidence → stop drafting here (FUCINA_MTP_PMIN)
        // Diagnostic: capture this step's top-4 head ids (4× masked-argmax). Runs AFTER the
        // chain token is fixed (d_mtp_tok already written), then trashes d_logits — safe, the
        // next replay recomputes it. d_sample_id is free during drafting.
        if (g_topk_diag > 0 && j < GEMMA4_SPEC_MAX) {
            for (int t = 0; t < 4; t++) {
                argmax_kernel<<<1, 32, 0, stream>>>(eng->d_logits, eng->d_sample_id, GEMMA4_VOCAB_SIZE);
                cudaMemcpyAsync(&g_diag_top4[j][t], eng->d_sample_id, sizeof(int),
                                cudaMemcpyDeviceToHost, stream);
                mask_one_neg_inf_kernel<<<1, 1, 0, stream>>>(eng->d_logits, eng->d_sample_id);
            }
            cudaStreamSynchronize(stream);
        }
        produced++;
    }
    if (produced == 0) return 0;
    if (cudaGetLastError() != cudaSuccess) return 0;
    int32_t ids[GEMMA4_SPEC_MAX];
    cudaMemcpyAsync(ids, eng->d_mtp_ids, (size_t)produced*sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return 0;
    for (int j = 0; j < produced; j++) {
        if (ids[j] < 0) return j;
        draft_out[j] = ids[j];
    }
    return produced;
}

// Lazy one-time capture of mtp_forward_paged into a CUDA graph. ONE graph serves every
// slot: the forward reads only FIXED device buffers — the input token (d_mtp_tok), the
// chain position (d_mtp_pos), the recurrent h (d_mtp_h_draft, seeded per-slot before each
// chain), and the per-slot KV views (d_ms_views_slid/glob, uploaded per-slot to the same
// addresses). So ~57 kernel launches per drafted token collapse to one cudaGraphLaunch.
// Requires d_mtp_h_draft allocated. On any capture failure the per-kernel path stays.
static int mtp_paged_graph_ensure(gemma4_engine_t *eng)
{
    if (eng->mtp_paged_graph) return 0;
    if (eng->mtp_paged_graph_failed) return -1;
    if (getenv("FUCINA_NO_BATCHED_GRAPH")) { eng->mtp_paged_graph_failed = 1; return -1; }
    if (!eng->d_mtp_h_draft) { eng->mtp_paged_graph_failed = 1; return -1; }
    cudaStream_t cs = NULL;
    cudaGraph_t  g  = NULL;
    int ok = cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking) == cudaSuccess;
    if (ok && cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
        mtp_forward_paged(eng, eng->d_mtp_tok, eng->d_mtp_pos, eng->d_mtp_h_draft,
                          eng->d_ms_views_slid, eng->d_ms_views_glob, cs);
        ok = cudaStreamEndCapture(cs, &g) == cudaSuccess && g != NULL;
    } else {
        ok = 0;
    }
    if (ok) ok = cudaGraphInstantiate(&eng->mtp_paged_graph, g, 0) == cudaSuccess;
    if (g)  cudaGraphDestroy(g);
    if (cs) cudaStreamDestroy(cs);
    if (!ok || !eng->mtp_paged_graph) {
        eng->mtp_paged_graph = NULL;
        eng->mtp_paged_graph_failed = 1;
        cudaGetLastError();
        fprintf(stderr, "fucina: paged MTP forward graph capture failed — per-kernel launches\n");
        return -1;
    }
    fprintf(stderr, "fucina: paged MTP forward graph captured (batch spec drafter on-device)\n");
    return 0;
}


// ── Batched (B-row) MTP drafter ──────────────────────────────────────────────
// Replaces the per-slot serial mtp_draft_paged loop (a cudaStreamSynchronize PER
// active slot) with ONE B-row forward per drafted token and ONE sync per round.

// Lazily allocate the [MAX_SEQS][dim] batched-drafter scratch. Logits reuse d_sb[11].
static int ensure_mtpb_scratch(gemma4_engine_t *eng)
{
    if (eng->mtpb_ready) return 0;
    const size_t M = GEMMA4_MAX_SEQS;
    const size_t H = eng->cfg.hidden_size;
    const size_t AH = GEMMA4_MTP_HIDDEN, FF = GEMMA4_MTP_FFN;
    const size_t QD = (size_t)eng->cfg.n_heads * GEMMA4_GLOBAL_HEAD_DIM;  // widest (global) Q
    int ok = cudaMalloc(&eng->d_mtpb_xh,   M*2*H*sizeof(float)) == cudaSuccess
          && cudaMalloc(&eng->d_mtpb_cur,  M*AH *sizeof(float)) == cudaSuccess
          && cudaMalloc(&eng->d_mtpb_t1,   M*AH *sizeof(float)) == cudaSuccess
          && cudaMalloc(&eng->d_mtpb_t2,   M*AH *sizeof(float)) == cudaSuccess
          && cudaMalloc(&eng->d_mtpb_q,    M*QD *sizeof(float)) == cudaSuccess
          && cudaMalloc(&eng->d_mtpb_attn, M*QD *sizeof(float)) == cudaSuccess
          && cudaMalloc(&eng->d_mtpb_ffa,  M*FF *sizeof(float)) == cudaSuccess
          && cudaMalloc(&eng->d_mtpb_ffb,  M*FF *sizeof(float)) == cudaSuccess
          && cudaMalloc(&eng->d_mtpb_h,    M*H  *sizeof(float)) == cudaSuccess
          && cudaMalloc(&eng->d_mtpb_ids,  M*GEMMA4_SPEC_MAX*sizeof(int))   == cudaSuccess
          && cudaMalloc(&eng->d_mtpb_conf, M*GEMMA4_SPEC_MAX*sizeof(float)) == cudaSuccess
          && cudaMalloc(&eng->d_mtpb_tok,  M*sizeof(int32_t)) == cudaSuccess
          && cudaMalloc(&eng->d_mtpb_pos,  M*sizeof(int))     == cudaSuccess;
    if (!ok) {
        void *ps[] = { eng->d_mtpb_xh, eng->d_mtpb_cur, eng->d_mtpb_t1, eng->d_mtpb_t2,
                       eng->d_mtpb_q, eng->d_mtpb_attn, eng->d_mtpb_ffa, eng->d_mtpb_ffb,
                       eng->d_mtpb_h, eng->d_mtpb_ids, eng->d_mtpb_conf, eng->d_mtpb_tok,
                       eng->d_mtpb_pos };
        for (size_t i = 0; i < sizeof(ps)/sizeof(ps[0]); i++) if (ps[i]) cudaFree(ps[i]);
        eng->d_mtpb_xh = eng->d_mtpb_cur = eng->d_mtpb_t1 = eng->d_mtpb_t2 = NULL;
        eng->d_mtpb_q = eng->d_mtpb_attn = eng->d_mtpb_ffa = eng->d_mtpb_ffb = NULL;
        eng->d_mtpb_h = NULL; eng->d_mtpb_conf = NULL;
        eng->d_mtpb_ids = NULL; eng->d_mtpb_pos = NULL; eng->d_mtpb_tok = NULL;
        cudaGetLastError();
        return -1;
    }
    eng->mtpb_ready = 1;
    return 0;
}

// Per-row argmax + top-1 softmax confidence over a [rows][V] logits matrix (one block
// per row, V-strided online max/sum — same math as mtp_argmax_conf_kernel). Writes the
// argmax id into ids[row*SPEC_MAX + j] and the bonus next input token into tok[row]
// (chains the next forward device-side); conf[row*SPEC_MAX + j] is the top-1 prob.
__global__ void mtp_argmax_conf_rows_kernel(
    const float *logits, int V, int *ids, float *conf, int32_t *tok, int j, int spec_max)
{
    const int row = blockIdx.x, T = blockDim.x, tid = threadIdx.x;
    const float *lr = logits + (size_t)row * V;
    float m = -INFINITY, s = 0.0f; int id = -1;
    for (int i = tid; i < V; i += T) {
        float v = lr[i];
        if (v > m) { s = s * __expf(m - v) + 1.0f; m = v; id = i; }
        else        s += __expf(v - m);
    }
    __shared__ float sm[1024], ss[1024];
    __shared__ int   sid[1024];
    sm[tid] = m; ss[tid] = s; sid[tid] = id;
    __syncthreads();
    for (int off = T / 2; off > 0; off >>= 1) {
        if (tid < off) {
            float m2 = sm[tid + off], s2 = ss[tid + off];
            if (m2 > sm[tid]) { ss[tid] = ss[tid] * __expf(sm[tid] - m2) + s2; sm[tid] = m2; sid[tid] = sid[tid + off]; }
            else              { ss[tid] += s2 * __expf(m2 - sm[tid]); }
        }
        __syncthreads();
    }
    if (tid == 0) {
        ids[(size_t)row * spec_max + j]  = sid[0];
        conf[(size_t)row * spec_max + j] = 1.0f / ss[0];
        tok[row] = sid[0];
    }
}

// B-row variant of mtp_forward_paged: drafts ALL B rows of the head in one pass using the
// batched GEMV + *_rows_* kernels (the same machinery the verify forward uses). Inputs are
// fixed device buffers: per-row token (d_mtpb_tok), per-row position (d_mtpb_pos), per-row
// recurrent h (d_mtpb_h, read as the ‖h half AND overwritten with next-h), and the per-row
// frozen KV views (d_ms_views_slid/glob). Logits land in d_sb[11] ([B][V]). Q-only attention
// reads the target's LAST sliding/global layer KV from the paged pools through B views.
// Exactness is NOT required (the verify re-derives every emitted token); a divergent draft
// only lowers the accept rate.
static void mtp_forward_paged_batched(gemma4_engine_t *eng, int B, cudaStream_t stream)
{
    const int H = eng->cfg.hidden_size, AH = GEMMA4_MTP_HIDDEN, FF = GEMMA4_MTP_FFN;
    const int NH = eng->cfg.n_heads, V = GEMMA4_VOCAB_SIZE;
    const int smem32 = 32 * (int)sizeof(float);
    auto grid1d = [](size_t n){ return (unsigned)((n + 255) / 256); };

    // xh[row] = [ √H · embed(tok_row) (H) | h_row (H) ]. Embed into d_sb[1] ([B][H]),
    // scale, then strided-copy both halves into the [B][2H] pre-projection input.
    embed_w(eng, eng->d_sb[1], weight_fp8(eng, eng->tensors.token_embd), eng->d_mtpb_tok, B, H, stream);
    scale_kernel<<<grid1d((size_t)B*H), 256, 0, stream>>>(eng->d_sb[1], B*H, sqrtf((float)H));
    cudaMemcpy2DAsync(eng->d_mtpb_xh,     2*H*sizeof(float), eng->d_sb[1],   H*sizeof(float),
                      H*sizeof(float), B, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(eng->d_mtpb_xh + H, 2*H*sizeof(float), eng->d_mtpb_h,  H*sizeof(float),
                      H*sizeof(float), B, cudaMemcpyDeviceToDevice, stream);
    gemv_batched_w(eng, eng->d_mtpb_cur, mtp_w(eng, eng->mtp.pre_proj), eng->d_mtpb_xh, 2*H, AH, B, stream, eng->mtp.wfmt);

    for (int l = 0; l < GEMMA4_MTP_LAYERS; l++) {
        const int is_g     = eng->mtp.is_global[l];
        const int head_dim = is_g ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_HEAD_DIM;
        const int qdim     = NH * head_dim;

        rms_norm_rows_kernel<<<B, 256, smem32, stream>>>(
            eng->d_mtpb_t1, eng->d_mtpb_cur, mtp_f32(eng, eng->mtp.attn_norm[l]), AH, B, GEMMA4_RMS_EPS);
        gemv_batched_w(eng, eng->d_mtpb_q, mtp_w(eng, eng->mtp.wq[l]), eng->d_mtpb_t1, AH, qdim, B, stream, eng->mtp.wfmt);
        per_head_rms_norm_rows_kernel<<<dim3(NH, B), head_dim, smem32, stream>>>(
            eng->d_mtpb_q, mtp_f32(eng, eng->mtp.q_norm[l]), NH, head_dim, B, GEMMA4_RMS_EPS);

        // RoPE with each row's OWN frozen position (Q-only ⇒ n_kv_heads=0 skips K).
        if (is_g)
            rope_rows_pos_kernel<<<dim3(NH, B), head_dim/2, 0, stream>>>(
                eng->d_mtpb_q, eng->d_mtpb_q, eng->d_mtpb_pos, NH, /*n_kv=*/0,
                head_dim, B, eng->cfg.rope_theta_global, mtp_f32(eng, eng->mtp.rope_freqs));
        else
            rope_rows_pos_kernel<<<dim3(NH, B), head_dim/2, 0, stream>>>(
                eng->d_mtpb_q, eng->d_mtpb_q, eng->d_mtpb_pos, NH, /*n_kv=*/0,
                head_dim, B, eng->cfg.rope_theta_sliding, NULL);

        int g_splits = GEMMA4_GLOBAL_MAX_SPLITS;
        if (is_g) {
            int gslot = eng->global_slot[eng->cfg.n_layers - 1];
            size_t okv = (size_t)eng->cfg.n_kv_global * GEMMA4_GLOBAL_HEAD_DIM;
            size_t pls = (size_t)eng->glob_pool.n_blocks * PAGED_KV_BLOCK_TOKENS * okv;
            const pkv_t *pk = eng->d_glob_pool_k + (size_t)gslot * pls;
            const pkv_t *pv = eng->d_glob_pool_v + (size_t)gslot * pls;
            if (NH == 32 && eng->cfg.n_kv_global == 4) {
                paged_global_attn_splitk_batched<32, 4, GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(g_splits, B), 32*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_mtpb_q, pk, pv, eng->d_ms_views_glob,
                        GEMMA4_GLOBAL_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, (int)okv);
                paged_flash_decode_combine_batched<32, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(32, B), head_dim, 0, stream>>>(
                        eng->d_mtpb_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_ms_views_glob,
                        head_dim, /*window=*/0, GEMMA4_GLOBAL_SPLIT_CHUNK);
            } else {
                paged_global_attn_splitk_batched<GEMMA4_HEADS, GEMMA4_GLOBAL_KV_HEADS, GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(g_splits, B), GEMMA4_HEADS*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_mtpb_q, pk, pv, eng->d_ms_views_glob,
                        GEMMA4_GLOBAL_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, (int)okv);
                paged_flash_decode_combine_batched<GEMMA4_HEADS, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(GEMMA4_HEADS, B), head_dim, 0, stream>>>(
                        eng->d_mtpb_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_ms_views_glob,
                        head_dim, /*window=*/0, GEMMA4_GLOBAL_SPLIT_CHUNK);
            }
        } else {
            size_t okv = (size_t)eng->cfg.n_kv_sliding * GEMMA4_HEAD_DIM;
            size_t pls = (size_t)eng->slid_pool.n_blocks * PAGED_KV_BLOCK_TOKENS * okv;
            const pkv_t *pk = eng->d_slid_pool_k + (size_t)(eng->cfg.n_layers - 2) * pls;
            const pkv_t *pv = eng->d_slid_pool_v + (size_t)(eng->cfg.n_layers - 2) * pls;
            const int win = GEMMA4_SLIDING_WINDOW - 1;   // frozen cache: window-1 prior keys
            if (NH == 32 && eng->cfg.n_kv_sliding == 16) {
                paged_sliding_attn_splitk_batched<32, 16, GEMMA4_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(g_splits, B), 16*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_mtpb_q, pk, pv, eng->d_ms_views_slid,
                        win, GEMMA4_SLIDING_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, (int)okv);
                paged_flash_decode_combine_batched<32, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(32, B), head_dim, 0, stream>>>(
                        eng->d_mtpb_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_ms_views_slid,
                        head_dim, win, GEMMA4_SLIDING_SPLIT_CHUNK);
            } else {
                paged_sliding_attn_splitk_batched<GEMMA4_HEADS, GEMMA4_KV_HEADS, GEMMA4_HEAD_DIM, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(g_splits, B), GEMMA4_KV_HEADS*32, 0, stream>>>(
                        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_mtpb_q, pk, pv, eng->d_ms_views_slid,
                        win, GEMMA4_SLIDING_SPLIT_CHUNK, PAGED_KV_BLOCK_TOKENS, (int)okv);
                paged_flash_decode_combine_batched<GEMMA4_HEADS, GEMMA4_GLOBAL_MAX_SPLITS>
                    <<<dim3(GEMMA4_HEADS, B), head_dim, 0, stream>>>(
                        eng->d_mtpb_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_ms_views_slid,
                        head_dim, win, GEMMA4_SLIDING_SPLIT_CHUNK);
            }
        }
        gemv_batched_w(eng, eng->d_mtpb_t1, mtp_w(eng, eng->mtp.wo[l]), eng->d_mtpb_attn, qdim, AH, B, stream, eng->mtp.wfmt);

        rms_norm_rows_kernel<<<B, 256, smem32, stream>>>(
            eng->d_mtpb_t2, eng->d_mtpb_t1, mtp_f32(eng, eng->mtp.post_attn_norm[l]), AH, B, GEMMA4_RMS_EPS);
        residual_add_kernel<<<grid1d((size_t)B*AH), 256, 0, stream>>>(eng->d_mtpb_t2, eng->d_mtpb_cur, B*AH);

        rms_norm_rows_kernel<<<B, 256, smem32, stream>>>(
            eng->d_mtpb_t1, eng->d_mtpb_t2, mtp_f32(eng, eng->mtp.ffn_norm[l]), AH, B, GEMMA4_RMS_EPS);
        gemv_batched_w(eng, eng->d_mtpb_ffa, mtp_w(eng, eng->mtp.gate[l]), eng->d_mtpb_t1, AH, FF, B, stream, eng->mtp.wfmt);
        gemv_batched_w(eng, eng->d_mtpb_ffb, mtp_w(eng, eng->mtp.up[l]),   eng->d_mtpb_t1, AH, FF, B, stream, eng->mtp.wfmt);
        geglu_kernel<<<grid1d((size_t)B*FF), 256, 0, stream>>>(eng->d_mtpb_ffa, eng->d_mtpb_ffa, eng->d_mtpb_ffb, B*FF);
        gemv_batched_w(eng, eng->d_mtpb_t1, mtp_w(eng, eng->mtp.down[l]), eng->d_mtpb_ffa, FF, AH, B, stream, eng->mtp.wfmt);
        rms_norm_rows_kernel<<<B, 256, smem32, stream>>>(
            eng->d_mtpb_cur, eng->d_mtpb_t1, mtp_f32(eng, eng->mtp.post_ffw_norm[l]), AH, B, GEMMA4_RMS_EPS);
        residual_add_kernel<<<grid1d((size_t)B*AH), 256, 0, stream>>>(eng->d_mtpb_cur, eng->d_mtpb_t2, B*AH);
        if (eng->mtp.out_scale[l] != 1.0f)
            scale_kernel<<<grid1d((size_t)B*AH), 256, 0, stream>>>(eng->d_mtpb_cur, B*AH, eng->mtp.out_scale[l]);
    }

    rms_norm_rows_kernel<<<B, 256, smem32, stream>>>(
        eng->d_mtpb_t1, eng->d_mtpb_cur, mtp_f32(eng, eng->mtp.out_norm), AH, B, GEMMA4_RMS_EPS);
    gemv_batched_w(eng, eng->d_sb[11], mtp_w(eng, eng->mtp.tok_embd), eng->d_mtpb_t1, AH, V, B, stream, eng->mtp.wfmt);
    gemv_batched_w(eng, eng->d_mtpb_h, mtp_w(eng, eng->mtp.post_proj), eng->d_mtpb_t1, AH, H, B, stream, eng->mtp.wfmt);
}

// Batched per-round draft for ndr eligible rows (each row's recurrence already seeded:
// dr_seq[k]->mtp_h_valid && d_mtp_h). Drafts up to max_draft tokens for ALL rows in one
// B-row forward per token, chains device-side (argmax_conf writes the next token into
// d_mtpb_tok), then does ONE readback + host-side pmin truncation per row. Fills
// draft_out[k][0..Dout[k]-1] and Dout[k]. No state is committed (the verify re-derives).
static void mtp_draft_paged_batched(
    gemma4_engine_t *eng, gemma4_seq **dr_seq, const int32_t *dr_tok, int ndr,
    int32_t (*draft_out)[GEMMA4_SPEC_MAX], int *Dout, int max_draft)
{
    for (int k = 0; k < ndr; k++) Dout[k] = 0;
    if (!eng->mtp.loaded || ndr <= 0 || max_draft <= 0) return;
    if (max_draft > GEMMA4_SPEC_MAX - 1) max_draft = GEMMA4_SPEC_MAX - 1;
    if (ensure_mtpb_scratch(eng) != 0) return;
    cudaStream_t stream = eng->stream;
    const int H = eng->cfg.hidden_size;

    static float pmin = -1.0f;
    if (pmin < 0.0f) { const char *e = getenv("FUCINA_MTP_PMIN"); pmin = e ? (float)atof(e) : GEMMA4_MTP_PMIN; }

    // Seed per-row inputs: token, frozen position, frozen B=1-style views, recurrent h.
    int32_t h_tok[GEMMA4_MAX_SEQS]; int h_pos[GEMMA4_MAX_SEQS];
    PagedSeqView hvs[GEMMA4_MAX_SEQS], hvg[GEMMA4_MAX_SEQS];
    for (int k = 0; k < ndr; k++) {
        gemma4_seq *s = dr_seq[k];
        int pos = s->n_tokens;
        h_tok[k] = dr_tok[k]; h_pos[k] = pos;
        hvs[k].block_table = s->d_slid_blocks; hvs[k].n_blocks = s->slid_bt.n; hvs[k].base = s->slid_bt.base; hvs[k].n_tokens = pos;
        hvg[k].block_table = s->d_glob_blocks; hvg[k].n_blocks = s->glob_bt.n; hvg[k].base = s->glob_bt.base; hvg[k].n_tokens = pos;
        cudaMemcpyAsync(eng->d_mtpb_h + (size_t)k*H, s->d_mtp_h, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    cudaMemcpyAsync(eng->d_mtpb_tok, h_tok, (size_t)ndr*sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(eng->d_mtpb_pos, h_pos, (size_t)ndr*sizeof(int),     cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(eng->d_ms_views_slid, hvs, (size_t)ndr*sizeof(PagedSeqView), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(eng->d_ms_views_glob, hvg, (size_t)ndr*sizeof(PagedSeqView), cudaMemcpyHostToDevice, stream);

    // Chain max_draft tokens with NO mid-chain sync; argmax_conf chains the next token.
    for (int j = 0; j < max_draft; j++) {
        mtp_forward_paged_batched(eng, ndr, stream);
        mtp_argmax_conf_rows_kernel<<<ndr, 1024, 0, stream>>>(
            eng->d_sb[11], GEMMA4_VOCAB_SIZE, eng->d_mtpb_ids, eng->d_mtpb_conf,
            eng->d_mtpb_tok, j, GEMMA4_SPEC_MAX);
    }

    // ONE readback for the whole round, then host-side pmin truncation per row.
    int32_t ids[GEMMA4_MAX_SEQS*GEMMA4_SPEC_MAX]; float conf[GEMMA4_MAX_SEQS*GEMMA4_SPEC_MAX];
    cudaMemcpyAsync(ids,  eng->d_mtpb_ids,  (size_t)ndr*GEMMA4_SPEC_MAX*sizeof(int),   cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(conf, eng->d_mtpb_conf, (size_t)ndr*GEMMA4_SPEC_MAX*sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return;
    if (cudaGetLastError() != cudaSuccess) return;
    for (int k = 0; k < ndr; k++) {
        int produced = 0;
        for (int j = 0; j < max_draft; j++) {
            int id = ids[(size_t)k*GEMMA4_SPEC_MAX + j];
            if (id < 0 || conf[(size_t)k*GEMMA4_SPEC_MAX + j] < pmin) break;
            draft_out[k][j] = id;
            produced++;
        }
        Dout[k] = produced;
    }
}

// =========================================================================
// ─── Greedy Speculative Decode (prompt-lookup) ──────────────────────────
// =========================================================================

// Longest-suffix n-gram draft: find the most recent earlier occurrence of the
// current suffix of hist[0..n-1] (lengths max_ng..min_ng) and propose the up-to
// max_d tokens that followed it. Zero model cost. Returns the draft length.
// Consensus prompt-lookup draft. Finds the longest suffix of `hist` (length
// max_ng..min_ng) that recurs earlier, then drafts its continuation — but only as
// far as the recent occurrences AGREE. A draft token is emitted only when it is the
// continuation in a strict majority of the most-recent matches; the draft is cut at
// the first low-consensus position. This raises acceptance precision (fewer wasted
// draft tokens, hence higher effective tok/s) on structured/repetitive text, and
// degrades exactly to plain prompt-lookup when only one match exists (thresh==1).
// Verify still samples each position from the exact target distribution, so accuracy
// is unchanged regardless of draft quality. Reports the matched n-gram length and
// occurrence count via out_ng/out_nocc (both 0 when no draft) so the caller's drafter
// policy can judge how trustworthy the draft is. Returns draft length.
static int prompt_lookup_draft(const int32_t *hist, int n, int32_t *draft,
                               int max_d, int min_ng, int max_ng,
                               int *out_ng, int *out_nocc)
{
    *out_ng = 0; *out_nocc = 0;
    if (max_d <= 0) return 0;
    enum { MAX_OCC = 16 };
    int occ[MAX_OCC];
    for (int ng = max_ng; ng >= min_ng; ng--) {
        if (n < ng + 1) continue;
        const int32_t *suf = hist + n - ng;
        // collect up to MAX_OCC most-recent earlier occurrences of the suffix
        int nocc = 0;
        for (int i = n - ng - 1; i >= 0 && nocc < MAX_OCC; i--) {
            int match = 1;
            for (int j = 0; j < ng; j++) if (hist[i+j] != suf[j]) { match = 0; break; }
            if (match) occ[nocc++] = i + ng;   // position right after the matched context
        }
        if (nocc == 0) continue;
        // Strict majority. occ[0] is the draft source and trivially agrees with itself,
        // so the old weak-majority thresh (nocc+1)/2 == 1 at nocc==2 let an n-gram seen
        // exactly TWICE draft full-length with zero real corroboration (and the ng cap
        // below only guards nocc==1). nocc/2+1 demands at least one OTHER occurrence
        // agree: nocc 2→2, 3→2, 4→3, 5→3; still 1 when nocc==1 (plain lookup + ng cap).
        int thresh = nocc / 2 + 1;
        // Confidence cap: one or two occurrences are only trusted as far ahead as
        // the matched context is long (a 5-gram match → up to 5 tokens; a 2-gram
        // → 2). Three or more agreeing occurrences are trusted to the full budget.
        int cap = max_d;
        if (nocc <= 2 && cap > ng) cap = ng;
        int d = 0;
        for (int dd = 0; dd < cap; dd++) {
            int p0 = occ[0] + dd;               // most-recent occurrence's continuation
            if (p0 >= n) break;
            int cand = hist[p0], agree = 0;
            for (int k = 0; k < nocc; k++) {
                int p = occ[k] + dd;
                if (p < n && hist[p] == cand) agree++;
            }
            if (agree < thresh) break;          // low consensus → stop drafting here
            draft[d++] = cand;
        }
        if (d > 0) { *out_ng = ng; *out_nocc = nocc; return d; }
    }
    return 0;
}

// ─── GPU repeat-penalty (keeps repeat_penalty != 1.0 on the spec path) ──
// Semantics mirror the Go sampler EXACTLY (internal/sampler.Sample): applied
// on the RAW (softcapped) logits BEFORE temperature, once PER OCCURRENCE in
// the past-token window — which is the ENTIRE sequence (kv.CurrentTokens()) —
// positive logit /= rp, negative *= rp, and SKIPPED at temp <= 0 (Sample
// short-circuits to argmax before the penalty). Per-occurrence application is
// reproduced bit-exactly by counting occurrences and applying c sequential
// divisions/multiplications: every application acts on the same value with
// the same operation, so the float result equals the Go loop's regardless of
// occurrence order (sign is stable under /,× by rp > 0, so Go's per-
// application sign test equals one up-front test).

__global__ void pen_count_kernel(int *cnt, const int32_t *toks, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int t = toks[i];
    if (t >= 0 && t < GEMMA4_VOCAB_SIZE) atomicAdd(&cnt[t], 1);
}

// Penalize K rows of logits. Row k's window additionally contains the draft
// tokens PRECEDING its position: batch[1..k] (batch[0] = g is already in the
// synced history — run_spec_loop appends g to hist before drafting).
__global__ void pen_apply_rows_kernel(
    float *logitsK, int V, const int *cnt, const int32_t *batch, float rp)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    if (v >= V) return;
    int c = cnt[v];
    for (int j = 1; j <= k; j++) if (batch[j] == v) c++;
    if (c == 0) return;
    // IEEE round-to-nearest intrinsics: the build uses --use_fast_math, which
    // lowers plain `/` to a reciprocal approximation and breaks bit-parity
    // with the Go sampler's float32 ops (proven by the pen_test harness).
    float x = logitsK[(size_t)k * V + v];
    if (x > 0.0f) { for (int i = 0; i < c; i++) x = __fdiv_rn(x, rp); }
    else          { for (int i = 0; i < c; i++) x = __fmul_rn(x, rp); }
    logitsK[(size_t)k * V + v] = x;
}

static int ensure_pen_scratch(gemma4_engine_t *eng) {
    if (eng->d_pen_cnt) return 0;
    size_t hist_cap = (size_t)eng->global_kv_capacity + 8;
    if (cudaMalloc(&eng->d_pen_cnt, (size_t)GEMMA4_VOCAB_SIZE * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&eng->d_pen_hist, hist_cap * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&eng->d_pen_batch, GEMMA4_SPEC_MAX * sizeof(int32_t)) != cudaSuccess) {
        cudaGetLastError();
        if (eng->d_pen_cnt)   { cudaFree(eng->d_pen_cnt);   eng->d_pen_cnt = NULL; }
        if (eng->d_pen_hist)  { cudaFree(eng->d_pen_hist);  eng->d_pen_hist = NULL; }
        if (eng->d_pen_batch) { cudaFree(eng->d_pen_batch); eng->d_pen_batch = NULL; }
        return -1;
    }
    return 0;
}

// Host-side twin for the host_sample call sites (first token after prefill +
// the rare decode-failure fallbacks). Identical loop to internal/sampler.
static void host_apply_penalty(float *logits, const int32_t *past, int n, float rp) {
    for (int i = 0; i < n; i++) {
        int32_t id = past[i];
        if (id < 0 || id >= GEMMA4_VOCAB_SIZE) continue;
        if (logits[id] > 0.0f) logits[id] /= rp; else logits[id] *= rp;
    }
}

// Host-side sampler mirroring sample_logits_kernel: temp<=0 → argmax; else
// temperature → top-k (bounded partial selection) → softmax → top-p → min-p →
// multinomial(rnd). Used by the speculative loop, where sampling each draft
// position from the TARGET distribution and accepting iff it equals the draft
// preserves the exact target sampling distribution (the draft proposal is a point
// mass, so standard speculative sampling reduces to this). rnd ∈ [0,1).
#define HSAMP_MAXK 256
static int host_sample(const float *logits, int V, float temp,
                       int top_k, float top_p, float min_p, double rnd)
{
    if (temp <= 0.0f) return gemma4_sample_argmax(logits, V);
    int K = (top_k > 0 && top_k < V) ? top_k : HSAMP_MAXK;
    if (K > HSAMP_MAXK) K = HSAMP_MAXK;
    // bounded size-K min-heap of indices by logit
    int   hid[HSAMP_MAXK]; int hn = 0;
    for (int i = 0; i < V; i++) {
        float v = logits[i];
        if (hn < K) {
            int j = hn++; hid[j] = i;
            while (j > 0) { int p=(j-1)/2; if (logits[hid[p]]<=logits[hid[j]]) break;
                int t=hid[p]; hid[p]=hid[j]; hid[j]=t; j=p; }
        } else if (v > logits[hid[0]]) {
            hid[0] = i; int j=0;
            for (;;) { int l=2*j+1,r=2*j+2,s=j;
                if (l<K && logits[hid[l]]<logits[hid[s]]) s=l;
                if (r<K && logits[hid[r]]<logits[hid[s]]) s=r;
                if (s==j) break; int t=hid[j]; hid[j]=hid[s]; hid[s]=t; j=s; }
        }
    }
    // sort descending by logit (insertion; K small)
    for (int a=1;a<hn;a++){ int id=hid[a]; float lv=logits[id]; int b=a-1;
        while (b>=0 && logits[hid[b]]<lv){ hid[b+1]=hid[b]; b--; } hid[b+1]=id; }
    float invT = 1.0f/temp, mx = logits[hid[0]], probs[HSAMP_MAXK], sum=0.0f;
    for (int a=0;a<hn;a++){ probs[a]=expf((logits[hid[a]]-mx)*invT); sum+=probs[a]; }
    for (int a=0;a<hn;a++) probs[a]/=sum;
    if (top_p>0.0f && top_p<1.0f){ float c=0; int cut=hn;
        for (int a=0;a<hn;a++){ c+=probs[a]; if (c>=top_p){cut=a+1;break;} } hn=cut; }
    if (min_p>0.0f){ float th=min_p*probs[0]; int keep=0;
        for (int a=0;a<hn;a++){ if (probs[a]>=th) keep=a+1; else break; } if (keep>0) hn=keep; }
    float z=0; for (int a=0;a<hn;a++) z+=probs[a];
    float r=(float)rnd*z, acc=0;
    for (int a=0;a<hn;a++){ acc+=probs[a]; if (r<=acc) return hid[a]; }
    return hid[hn>0?hn-1:0];
}

// xorshift128+ style PRNG for reproducible host sampling draws.
static inline double spec_rng(uint64_t *s) {
    uint64_t x = *s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; *s = x;
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}

// Monotonic host clock in ms. Used to time regions that already end with a
// stream sync (the spec verify step), replacing a redundant per-step
// cudaEventSynchronize in decode_batched_dev.
static inline double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

// fwd decls — the GPU samplers (a) are defined below the spec loop with the rest of
// the sampling code, but run_spec_loop launches them.
__global__ void sample_logits_kernel(
    const float *logits, int V, float temp, int top_k, float top_p, float min_p,
    float rnd, int *out_id);
__global__ void sample_logits_batched_kernel(
    const float *logits, int V, float temp, int top_k, float top_p, float min_p,
    const float *rnds, int *out_ids);
// (sample_logits_ms_kernel is forward-declared above decode_multiseq_forward.)

static int run_spec_loop(
    gemma4_engine_t *eng, int32_t *hist, int n, float *logits,
    int32_t *out_tokens, int max_new, const int32_t *stop_ids, int n_stop,
    int draft_k, float temp, int top_k, float top_p, float min_p, float rep_pen,
    uint64_t seed, int *n_accepted_out, gemma4_token_cb cb, void *cb_ud)
{
    int V = GEMMA4_VOCAB_SIZE;
    // (a) Verify logits stay on device (d_sb[11]); only K ids cross to host — no Lbuf.
    int32_t batch[GEMMA4_SPEC_MAX];
    if (ensure_spec_scratch(eng) != 0) return -1;  // d_spec_ids/d_spec_rnd, d_sb

    // GPU repeat-penalty (see pen_apply_rows_kernel). Matches the Go sampler:
    // active only for temp > 0 (Sample() short-circuits to raw argmax first),
    // window = the whole sequence. Occurrence counts build incrementally:
    // hist is append-only (rejected drafts never enter it), so each step
    // uploads + counts only the tokens accepted since the last sync.
    const int use_pen = (rep_pen != 1.0f && rep_pen > 0.0f && temp > 0.0f);
    int pen_synced = 0;
    if (use_pen) {
        if (ensure_pen_scratch(eng) != 0) return -1;   // fail hard, never silently unpenalized
        cudaMemsetAsync(eng->d_pen_cnt, 0, (size_t)V * sizeof(int), eng->stream);
    }
    auto pen_sync = [&](int upto) {
        if (upto <= pen_synced) return;
        cudaMemcpyAsync(eng->d_pen_hist + pen_synced, hist + pen_synced,
                        (size_t)(upto - pen_synced) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, eng->stream);
        pen_count_kernel<<<(upto - pen_synced + 255) / 256, 256, 0, eng->stream>>>(
            eng->d_pen_cnt, eng->d_pen_hist + pen_synced, upto - pen_synced);
        pen_synced = upto;
    };
    auto is_stop = [&](int t){ for (int s=0;s<n_stop;s++) if (stop_ids[s]==t) return 1; return 0; };

    uint64_t rng = seed ? seed : 0x9e3779b97f4a7c15ULL;
    if (g_topk_diag < 0) { const char *e = getenv("FUCINA_SPEC_TOPK_DIAG"); g_topk_diag = (e && e[0]=='1') ? 1 : 0; }
    g_diag_rej = g_diag_hit2 = g_diag_hit3 = g_diag_hit4 = 0;
    long mtp_calls = 0, mtp_drafted = 0, mtp_accepted = 0;   // drafter diagnostics
    long mtp_declines = 0;   // assistant invoked but drafted 0 (paid ≥1 forward, invisible otherwise)
    long lk_calls  = 0, lk_drafted  = 0, lk_accepted  = 0, lk_displaced = 0;
    // Cold counter stays SHARED across drafters: a fully-rejected step says the local
    // text is hostile to speculation whichever drafter produced it, and sharing keeps a
    // hard cap on consecutive wasted verifies — per-drafter counters would let the two
    // drafters alternate total rejections and never trip the one-step backoff.
    int cold = 0;             // consecutive fully-rejected draft attempts (light backoff)
    // Per-drafter acceptance EMAs (accepted tokens/step; both start optimistic at
    // draft_k). A single shared EMA conflated the drafters' quality: low LOOKUP
    // acceptance shrank maxd to its floor of 2, which then throttled the 85-94%-
    // acceptance MTP drafter too, locking long chats into cheap ~1.0x lookup drafts.
    float ema_lookup = (float)draft_k, ema_mtp = (float)draft_k;


    if (use_pen) host_apply_penalty(logits, hist, n, rep_pen);
    int g = host_sample(logits, V, temp, top_k, top_p, min_p, spec_rng(&rng));
    int generated = 0, total_accepted = 0, stop = 0;
    while (generated < max_new && !stop) {
        out_tokens[generated++] = g; hist[n++] = g;
        if (cb && cb(g, cb_ud)) break;   // streaming consumer asked to stop here
        if (is_stop(g)) break;

        int pos = eng->cur.n_tokens;
        // FLAT KV (Step 3): rewind is now exact at ANY ctx, so the draft is no longer clamped
        // to the 1024 sliding window — only to the cache capacity (don't write past the buffer).
        // This is what re-enables speculation past 1024 ctx.
        int room = eng->global_kv_capacity - 1 - pos;
        // Adaptive draft length. A K-token batched verify costs ~2x a single decode on
        // this hardware (the per-token attention/sampling work is NOT fully amortized by
        // the one shared weight pass), so break-even needs accepted/D > ~0.23 — over-
        // drafting on low-acceptance text loses. Size the draft to the recent accepted
        // run (+1 probe token): verbatim/repetitive text grows it to draft_k for a big
        // win; novel text shrinks it toward the floor so a rejected draft stays cheap.
        // Light backoff: after a short run of total rejections, skip one step, then retry.
        // The budget comes from the drafter we'd PREFER this step (MTP when loaded with a
        // valid recurrent h, lookup otherwise). Budgeting from ema_mtp while MTP is
        // preferred also keeps the displacement bar honest: a shrunken ema_lookup must
        // not lower the full-length requirement lookup has to clear to displace MTP.
        int mtp_ready = eng->mtp.loaded && eng->mtp_h_valid;
        // Per-drafter budgets: each drafter is sized by ITS OWN acceptance record.
        // A single budget keyed off the preferred drafter let MTP's mediocre EMA
        // clamp a ~99%-accepting lookup chain to ~2-token drafts on repetitive
        // agent text (file/diff re-emission) — the dominant tool-call decode
        // regime (measured: 62.9 tok/s where ~85-100 was available).
        int maxd_mtp = (int)(ema_mtp + 1.5f);
        if (maxd_mtp > draft_k) maxd_mtp = draft_k;
        if (maxd_mtp < 2)       maxd_mtp = 2;
        // Lookup drafts are host-side free (only the verify cost scales, ~0.23x a
        // decode per row): when its EMA saturates the configured cap, let the cap
        // grow toward the batch limit (K = D+1 ≤ GEMMA4_SPEC_MAX) — acceptance was
        // measured to scale with draft length on verbatim text (k=12→9.38, k=15→
        // 10.71 avg accepted), and the EMA shrinks it back within ~2 steps when
        // the text turns novel.
        int cap_lk = draft_k;
        // lk_accepted > 0: the EMA starts optimistic at draft_k, so without the
        // evidence gate the cap would unlock on step 1 with zero track record.
        if (lk_accepted > 0 && ema_lookup >= (float)draft_k - 1.0f) cap_lk = GEMMA4_SPEC_MAX - 1;
        int maxd_lk = (int)(ema_lookup + 1.5f);
        if (maxd_lk > cap_lk) maxd_lk = cap_lk;
        if (maxd_lk < 2)      maxd_lk = 2;
        if (cold >= 4) { maxd_mtp = 0; maxd_lk = 0; cold = 0; }
        if (maxd_mtp > room) maxd_mtp = room; if (maxd_mtp < 0) maxd_mtp = 0;
        if (maxd_lk  > room) maxd_lk  = room; if (maxd_lk  < 0) maxd_lk  = 0;
        if (generated + 1 >= max_new) { maxd_mtp = 0; maxd_lk = 0; }

        // Drafter policy (one config): consensus prompt-lookup is FREE, but when the MTP
        // assistant is available it may displace MTP only when its draft is genuinely
        // STRONG — full budget length AND a >=3-gram match corroborated by >=2 occurrences
        // (under the strict majority in prompt_lookup_draft). Previously ANY full-length
        // lookup draft starved MTP (the llama.cpp draft-mtp config — measured 84-94% draft
        // acceptance on novel prose/code where lookup gets almost nothing), and long chats
        // fed a feedback loop: more history => more spurious bigram matches => more ~1.0x
        // lookup drafts. If MTP declines to draft, fall back to whatever lookup found; and
        // without an assistant, today's behavior stands — lookup keeps any draft it found,
        // partial included. The first step after a fresh prefill single-decodes once to
        // establish the recurrent h. The MTP budget stays EMA-clamped on purpose: an A/B
        // with the full draft_k budget (confidence-gate-only truncation) measured WORSE
        // (20.8 vs 22.4 tok/s on chat text) — each drafted token costs a full ~444MB
        // assistant pass + a stream sync, so medium-confidence tokens the gate admits
        // past the recent accepted run don't pay for themselves.
        int lk_ng = 0, lk_nocc = 0;
        // ── Tree spec path (FUCINA_SPEC_TREE=width; 1 ⇒ linear-as-tree, ≥2 ⇒ branching) ──
        // Runs BEFORE the linear drafter so the committed recurrent h (d_mtp_h) is intact for
        // the root forward. On success it commits the longest draw-match path and `continue`s;
        // any failure falls through to the linear path (which re-establishes h from its verify).
        static int g_tree_w = -2;
        if (g_tree_w == -2) { const char *e = getenv("FUCINA_SPEC_TREE"); g_tree_w = e ? atoi(e) : 0; }
        if (g_tree_w >= 1 && mtp_ready && eng->mtp_h_valid && maxd_mtp > 0 && eng->cur.n_tokens > 0) {
            const int Hh = eng->cfg.hidden_size;
            spec_tree tt; int rk[GEMMA4_TREE_MAX];
            int tdepth = maxd_mtp < draft_k ? maxd_mtp : draft_k;
            tree_build_template(&tt, rk, tdepth, g_tree_w, /*branch_until=*/2, /*branch_len=*/2);
            static int g_tt = -1; static double g_dt = 0, g_vt = 0; static long g_ns = 0, g_nnodes = 0;
            if (g_tt < 0) { const char *e = getenv("FUCINA_SPEC_TREE_TIMING"); g_tt = (e && e[0]=='1') ? 1 : 0; }
            double _t0 = g_tt ? (cudaStreamSynchronize(eng->stream), now_ms()) : 0;
            int nn = mtp_draft_tree(eng, g, &tt, rk);
            if (g_tt) { cudaStreamSynchronize(eng->stream); g_dt += now_ms() - _t0; g_ns++; g_nnodes += nn; }
            double _t1 = g_tt ? now_ms() : 0;
            int posT = eng->cur.n_tokens;
            if (nn >= 2 && decode_tree_dev(eng, &tt) == 0) {
                if (g_tt) { cudaStreamSynchronize(eng->stream); g_vt += now_ms() - _t1;
                    if ((g_ns % 20) == 0) fprintf(stderr, "fucina: [tree-timing] %ld steps  draft=%.1fms/step verify=%.1fms/step  avg %.1f nodes\n",
                        g_ns, g_dt/g_ns, g_vt/g_ns, (double)g_nnodes/g_ns); }
                double vt0 = now_ms();
                float rnds[GEMMA4_TREE_MAX]; for (int i = 0; i < nn; i++) rnds[i] = (float)spec_rng(&rng);
                cudaMemcpyAsync(eng->d_spec_rnd, rnds, (size_t)nn*sizeof(float), cudaMemcpyHostToDevice, eng->stream);
                sample_logits_batched_kernel<<<nn, 1024, 0, eng->stream>>>(
                    decode_batched_dev_logits(eng), V, temp, top_k, top_p, min_p,
                    eng->d_spec_rnd, eng->d_spec_ids);
                int ids[GEMMA4_TREE_MAX];
                cudaMemcpyAsync(ids, eng->d_spec_ids, (size_t)nn*sizeof(int), cudaMemcpyDeviceToHost, eng->stream);
                if (cudaStreamSynchronize(eng->stream) != cudaSuccess) { stop = 1; break; }
                eng->decode_time_ms += (float)(now_ms() - vt0); eng->n_decode_tokens += nn;
                // Longest root→leaf path where each edge's draft token == the parent's target sample.
                int path[GEMMA4_TREE_MAX]; int Lp = 0, cur = 0;
                for (;;) {
                    int tgt = ids[cur], nx = -1;
                    for (int c = 0; c < nn; c++) if (tt.parent[c] == cur && tt.tok[c] == tgt) { nx = c; break; }
                    if (nx < 0) break;
                    path[++Lp] = nx; cur = nx;
                }
                tree_commit_kv(eng, posT, path, Lp);               // no-op when the path is contiguous
                if (eng->mtp.loaded) {
                    cudaMemcpyAsync(eng->d_mtp_h, eng->d_sb[2] + (size_t)cur * Hh,
                                    (size_t)Hh * sizeof(float), cudaMemcpyDeviceToDevice, eng->stream);
                    eng->mtp_h_valid = 1;
                }
                int keep = posT + 1 + Lp;
                if (keep < eng->cur.n_tokens && gemma4_engine_rewind(eng, keep) != 0) { stop = 1; break; }
                eng->spec_steps++; eng->spec_drafted += (nn - 1); eng->spec_accepted += Lp; eng->spec_emitted += (Lp + 1);
                mtp_calls++; mtp_drafted += (nn - 1); mtp_accepted += Lp;
                ema_mtp = 0.5f*ema_mtp + 0.5f*(float)Lp;
                total_accepted += Lp;
                for (int d = 1; d <= Lp && generated < max_new; d++) {
                    int tk = tt.tok[path[d]]; out_tokens[generated++] = tk; hist[n++] = tk;
                    if (cb && cb(tk, cb_ud)) { stop = 1; break; }
                    if (is_stop(tk)) { stop = 1; break; }
                }
                if (stop) break;
                g = ids[cur];
                continue;
            }
            // decode_tree_dev advanced n_tokens but we did not commit — rewind to the prefix so
            // the linear fallthrough starts clean. (h was clobbered by drafting; the linear
            // verify re-establishes it from d_sb[2], so output stays correct.)
            if (eng->cur.n_tokens > posT) gemma4_engine_rewind(eng, posT);
        }
        int D = prompt_lookup_draft(hist, n, batch+1, maxd_lk, 2, draft_k, &lk_ng, &lk_nocc);
        int from_mtp = 0;
        // T1a debug: one-shot — draft a tree at this step and print it next to the linear
        // chain, to verify the KV-free drafter (width-1 trunk MUST equal the linear drafts).
        static int g_tree_dbg = -1;
        if (g_tree_dbg < 0) { const char *e = getenv("FUCINA_SPEC_TREE_DEBUG"); g_tree_dbg = (e && e[0]=='1') ? 1 : 0; }
        if (g_tree_dbg == 1 && eng->mtp.loaded && eng->mtp_h_valid && maxd_mtp > 0) {
            g_tree_dbg = 2;   // one-shot
            int wdt = 2; { const char *w = getenv("FUCINA_SPEC_TREE_WIDTH"); if (w) wdt = atoi(w); }
            spec_tree tt; int rk[GEMMA4_TREE_MAX];
            tree_build_template(&tt, rk, /*depth=*/6, /*width=*/wdt, /*branch_until=*/2, /*branch_len=*/2);
            // Both drafters consume the committed recurrent h from d_mtp_h and overwrite it;
            // snapshot it so the linear and tree drafts start from the SAME (g, h) state.
            const int Hd = eng->cfg.hidden_size;
            float *h0 = NULL; cudaMalloc(&h0, (size_t)Hd*sizeof(float));
            cudaMemcpyAsync(h0, eng->d_mtp_h, (size_t)Hd*sizeof(float), cudaMemcpyDeviceToDevice, eng->stream);
            int32_t lin[GEMMA4_SPEC_MAX]; int Dl = mtp_draft(eng, g, lin, 6);
            cudaMemcpyAsync(eng->d_mtp_h, h0, (size_t)Hd*sizeof(float), cudaMemcpyDeviceToDevice, eng->stream);
            eng->mtp_h_valid = 1;
            int nn = mtp_draft_tree(eng, g, &tt, rk);
            cudaMemcpyAsync(eng->d_mtp_h, h0, (size_t)Hd*sizeof(float), cudaMemcpyDeviceToDevice, eng->stream);
            cudaStreamSynchronize(eng->stream); cudaFree(h0);
            fprintf(stderr, "fucina: [tree-dbg] width=%d nodes=%d  linear(%d): ", wdt, nn, Dl);
            for (int i = 0; i < Dl; i++) fprintf(stderr, "%d ", lin[i]);
            fprintf(stderr, "\nfucina: [tree-dbg] tree nodes [idx:parent@depth=tok]:\n");
            for (int i = 0; i < nn; i++)
                fprintf(stderr, "   %2d: p%-2d @d%d  tok=%d%s\n", i, tt.parent[i], tt.depth[i],
                        tt.tok[i], (tt.parent[i]<0?"  (root=g)":""));
        }
        // The displacement bar stays pinned to the MTP budget: to displace MTP a
        // lookup draft must reach the full length MTP would have been allowed,
        // with a >=3-gram match corroborated by >=2 occurrences. (Pinning to the
        // lookup budget would let a shrunken ema_lookup lower the bar.)
        int lookup_strong = (D > 0 && maxd_mtp > 0 && D >= maxd_mtp && lk_ng >= 3 && lk_nocc >= 2);
        if (mtp_ready && maxd_mtp > 0 && !lookup_strong) {
            int Dm = mtp_draft(eng, g, batch+1, maxd_mtp);  // drafts into a local buf; on 0 the
            if (Dm > 0) {                                   // lookup draft in batch+1 is intact
                if (D > 0) lk_displaced++;                  // weak lookup draft lost to MTP
                D = Dm; from_mtp = 1;
                mtp_calls++; mtp_drafted += Dm;
            } else {
                mtp_declines++;   // paid ≥1 assistant forward, drafted nothing
            }
        }
        // Count lookup drafts only when they reach the verify (displaced drafts
        // were inflating the denominator: Req1's "0% of 71 proposed" was really
        // 2 verified drafts — a stats artifact).
        if (D > 0 && !from_mtp) { lk_calls++; lk_drafted += D; }
        if (D == 0) {
            // Cold/no-draft step: decode g WITHOUT the 262k D2H, then sample on-device
            // (eng->d_logits already carries softcap+suppress). decode() also refreshes
            // the MTP recurrent h. Mirrors the verify path's on-GPU sampling.
            if (gemma4_engine_decode(eng, g, NULL) != 0) { stop = 1; break; }
            if (use_pen) {
                pen_sync(n);
                pen_apply_rows_kernel<<<dim3((V + 255) / 256, 1), 256, 0, eng->stream>>>(
                    eng->d_logits, V, eng->d_pen_cnt, eng->d_pen_batch, rep_pen);
            }
            float rnd1 = (float)spec_rng(&rng);
            sample_logits_kernel<<<1, 1024, 0, eng->stream>>>(
                eng->d_logits, V, temp, top_k, top_p, min_p, rnd1, eng->d_sample_id);
            int gid = -1;
            cudaMemcpyAsync(&gid, eng->d_sample_id, sizeof(int),
                            cudaMemcpyDeviceToHost, eng->stream);
            if (cudaStreamSynchronize(eng->stream) != cudaSuccess || gid < 0) { stop = 1; break; }
            g = gid;
            continue;
        }
        batch[0] = g;
        int K = D + 1;
        // (a) GPU-side verify: forward [g,draft...] keeping the K logit rows on device
        // (d_sb[11]), pre-draw K host uniforms, sample every row on the GPU, and read
        // back only the K ids. Replaces the K×262144 logit D2H + K full-vocab CPU
        // host_sample scans. Each row is sampled from the exact target distribution
        // (greedy temp<=0 → argmax → bit-identical to host_sample), so acceptance still
        // preserves the target distribution.
        float rnds[GEMMA4_SPEC_MAX]; int ids[GEMMA4_SPEC_MAX];
        for (int i = 0; i < K; i++) rnds[i] = (float)spec_rng(&rng);
        double vt0 = now_ms();   // decode_batched_dev(keep_dev=1) skips event timing
        if (decode_batched_dev(eng, batch, K, NULL, /*keep_dev=*/1) != 0) {
            if (gemma4_engine_decode(eng, g, logits) != 0) { stop = 1; break; }
            if (use_pen) host_apply_penalty(logits, hist, n, rep_pen);
            g = host_sample(logits, V, temp, top_k, top_p, min_p, spec_rng(&rng));
            continue;
        }
        if (use_pen) {
            pen_sync(n);
            cudaMemcpyAsync(eng->d_pen_batch, batch, (size_t)K*sizeof(int32_t),
                            cudaMemcpyHostToDevice, eng->stream);
            pen_apply_rows_kernel<<<dim3((V + 255) / 256, K), 256, 0, eng->stream>>>(
                decode_batched_dev_logits(eng), V, eng->d_pen_cnt, eng->d_pen_batch, rep_pen);
        }
        cudaMemcpyAsync(eng->d_spec_rnd, rnds, (size_t)K*sizeof(float),
                        cudaMemcpyHostToDevice, eng->stream);
        sample_logits_batched_kernel<<<K, 1024, 0, eng->stream>>>(
            decode_batched_dev_logits(eng), V, temp, top_k, top_p, min_p,
            eng->d_spec_rnd, eng->d_spec_ids);
        cudaMemcpyAsync(ids, eng->d_spec_ids, (size_t)K*sizeof(int),
                        cudaMemcpyDeviceToHost, eng->stream);
        if (cudaStreamSynchronize(eng->stream) != cudaSuccess) {
            if (gemma4_engine_decode(eng, g, logits) != 0) { stop = 1; break; }
            if (use_pen) host_apply_penalty(logits, hist, n, rep_pen);
            g = host_sample(logits, V, temp, top_k, top_p, min_p, spec_rng(&rng));
            continue;
        }
        // Stream is drained: charge the whole verify step (forward + sample + D2H)
        // to the decode counters, replacing the event timing skipped under keep_dev.
        eng->decode_time_ms += (float)(now_ms() - vt0);
        eng->n_decode_tokens += K;
        int a = 0;
        while (a < D && ids[a] == batch[1+a]) a++;   // first mismatch = accept length
        // Tree-spec de-risk: at the reject row a, was the target token in the head's top-4?
        // top4[a][0] == batch[1+a] (the rejected argmax) by construction, so a hit means
        // a width-2/3/4 tree branch would have carried the verify past this position.
        if (g_topk_diag > 0 && from_mtp && a < D) {
            g_diag_rej++;
            int tgt = ids[a];
            if      (tgt == g_diag_top4[a][1]) g_diag_hit2++;
            else if (tgt == g_diag_top4[a][2]) g_diag_hit3++;
            else if (tgt == g_diag_top4[a][3]) g_diag_hit4++;
        }
        total_accepted += a;
        // Cumulative acceptance accounting for /metrics (τ vs other engines). Each verify
        // step is one target forward yielding (a accepted drafts + 1 committed bonus/resample).
        eng->spec_steps++;
        eng->spec_drafted  += D;
        eng->spec_accepted += a;
        eng->spec_emitted  += (a + 1);
        // Update ONLY the EMA of the drafter that actually drafted this step — the other
        // drafter's track record is untouched evidence about ITS quality, not this one's.
        if (from_mtp) { mtp_accepted += a; ema_mtp    = 0.5f*ema_mtp    + 0.5f*(float)a; }
        else          { lk_accepted  += a; ema_lookup = 0.5f*ema_lookup + 0.5f*(float)a; }
        cold = (a == 0) ? cold + 1 : 0;
        // MTP recurrent h: the next g comes from logits row a (reject) or row D (all
        // accepted) — stash that row's output-normed hidden (d_sb[2]) as the next h.
        if (eng->mtp.loaded) {
            int hrow = (a < D) ? a : D;
            cudaMemcpyAsync(eng->d_mtp_h, eng->d_sb[2] + (size_t)hrow * GEMMA4_HIDDEN_SIZE,
                            GEMMA4_HIDDEN_SIZE * sizeof(float),
                            cudaMemcpyDeviceToDevice, eng->stream);
            eng->mtp_h_valid = 1;
        }
        int keep = pos + 1 + a;
        // FLAT KV (Step 3): rewind is exact now; check the return defensively (design :4756).
        if (keep < eng->cur.n_tokens && gemma4_engine_rewind(eng, keep) != 0) {
            fprintf(stderr, "fucina: spec rewind to %d failed — stopping spec\n", keep);
            stop = 1; break;
        }
        for (int i = 0; i < a && generated < max_new; i++) {
            int t = batch[1+i]; out_tokens[generated++] = t; hist[n++] = t;
            if (cb && cb(t, cb_ud)) { stop = 1; break; }
            if (is_stop(t)) { stop = 1; break; }
        }
        if (stop) break;
        // Next g is the GPU-sampled token of the first unaccepted row (reject resample
        // at row a) or the bonus row D (all accepted) — already drawn on-device above.
        g = ids[a];
    }
    if (eng->mtp.loaded && (mtp_calls > 0 || mtp_declines > 0))
        fprintf(stderr, "fucina: [mtp] %ld draft calls, %ld proposed, %ld accepted (%.0f%%), %ld declined\n",
                mtp_calls, mtp_drafted, mtp_accepted,
                100.0 * mtp_accepted / (double)(mtp_drafted > 0 ? mtp_drafted : 1),
                mtp_declines);
    // Lookup symmetric to [mtp], plus the count of weak drafts the policy handed to MTP
    // instead — that displacement rate IS the new policy, so it must be visible here
    // (also when displacement was TOTAL, i.e. zero verified lookup drafts).
    if (lk_calls > 0 || lk_displaced > 0)
        fprintf(stderr, "fucina: [lookup] %ld draft calls, %ld proposed, %ld accepted (%.0f%%), %ld displaced by mtp\n",
                lk_calls, lk_drafted, lk_accepted,
                100.0 * lk_accepted / (double)(lk_drafted > 0 ? lk_drafted : 1),
                lk_displaced);
    if (g_topk_diag > 0 && g_diag_rej > 0) {
        long c2 = g_diag_hit2, c4 = g_diag_hit2 + g_diag_hit3 + g_diag_hit4;
        fprintf(stderr, "fucina: [tree-diag] %ld rejects — target in head top-2: %ld (%.0f%%), "
                "top-4: %ld (%.0f%%) | a width-w tree recovers this fraction of rejects\n",
                g_diag_rej, c2, 100.0*c2/g_diag_rej, c4, 100.0*c4/g_diag_rej);
    }
    if (n_accepted_out) *n_accepted_out = total_accepted;
    return generated;
}

// [g, draft...] in ONE weight pass (gemma4_engine_decode_batched), commits the
// confirmed prefix, and gets the next step's logits for free from the same pass —
// so a matched draft of length a yields (1+a) tokens per ~one token's bandwidth.
// Works for BOTH greedy (temp<=0) and sampling (temp>0): each position is drawn
// from the target distribution and the draft is accepted iff the draw matches it,
// which preserves the exact target distribution. Draft length is capped to keep the
// partial-accept KV rollback inside the sliding window (full speedup for the first
// ~window tokens; plain decode beyond). Fills out_tokens (≤ max_new), returns the
// count generated. n_accepted_out (or NULL) receives the total drafts accepted.
int gemma4_engine_generate_spec(
    gemma4_engine_t *eng,
    const int32_t   *prompt, int n_prompt,
    int32_t         *out_tokens, int max_new,
    const int32_t   *stop_ids, int n_stop,
    int              draft_k,
    float            temp, int top_k, float top_p, float min_p, float repeat_penalty,
    uint64_t seed,
    int             *n_accepted_out)
{
    if (!eng || !eng->loaded || n_prompt <= 0 || max_new <= 0) return -1;
    if (draft_k > GEMMA4_SPEC_MAX - 1) draft_k = GEMMA4_SPEC_MAX - 1;
    int V = GEMMA4_VOCAB_SIZE;

    int cap = n_prompt + max_new + 8;
    int32_t *hist = (int32_t*)malloc((size_t)cap*sizeof(int32_t));
    float   *logits = (float*)malloc((size_t)V*sizeof(float));
    if (!hist || !logits) { free(hist); free(logits); return -1; }

    memcpy(hist, prompt, (size_t)n_prompt*sizeof(int32_t));
    int n = n_prompt;

    // Prefill the prompt → logits predicting the first generated token. Use the
    // BATCHED prefill (one weight pass over the whole prompt, ~900 tok/s); fall back
    // to token-by-token only if it declines (non-fresh sequence). The token-by-token
    // path would otherwise prefill the entire prompt at decode speed (~15 tok/s).
    int prc = gemma4_engine_prefill_batched(eng, prompt, n_prompt, logits);
    if (prc == -2) prc = gemma4_engine_prefill_flash(eng, prompt, n_prompt, logits);
    if (prc == -2) prc = gemma4_engine_prefill(eng, prompt, n_prompt, logits);
    if (prc != 0) {
        free(hist); free(logits); return -1;
    }

    int generated = run_spec_loop(eng, hist, n, logits, out_tokens, max_new,
        stop_ids, n_stop, draft_k, temp, top_k, top_p, min_p, repeat_penalty,
        seed, n_accepted_out, NULL, NULL);

    free(hist); free(logits);
    return generated;
}

// Streaming server/REPL path: gemma4_engine_generate_spec_continue plus a per-token
// callback (see gemma4_token_cb). The callback fires between verify steps on the
// calling thread, so SSE/console streaming rides the speculative fast path instead
// of the per-token decode loop. cb == NULL behaves exactly like _continue.
int gemma4_engine_generate_spec_stream(
    gemma4_engine_t *eng,
    const int32_t   *history, int n_history,
    const float     *first_logits,
    int32_t         *out_tokens, int max_new,
    const int32_t   *stop_ids, int n_stop,
    int              draft_k,
    float            temp, int top_k, float top_p, float min_p, float repeat_penalty,
    uint64_t seed,
    int             *n_accepted_out,
    gemma4_token_cb  cb, void *cb_user_data)
{
    if (!eng || !eng->loaded || n_history < 0 || max_new <= 0) return -1;
    if (draft_k > GEMMA4_SPEC_MAX - 1) draft_k = GEMMA4_SPEC_MAX - 1;
    int V = GEMMA4_VOCAB_SIZE;

    int cap = n_history + max_new + 8;
    int32_t *hist = (int32_t*)malloc((size_t)cap*sizeof(int32_t));
    float   *logits = (float*)malloc((size_t)V*sizeof(float));
    if (!hist || !logits) { free(hist); free(logits); return -1; }
    if (n_history > 0) memcpy(hist, history, (size_t)n_history*sizeof(int32_t));
    memcpy(logits, first_logits, (size_t)V*sizeof(float));

    int generated = run_spec_loop(eng, hist, n_history, logits, out_tokens, max_new,
        stop_ids, n_stop, draft_k, temp, top_k, top_p, min_p, repeat_penalty,
        seed, n_accepted_out, cb, cb_user_data);

    free(hist); free(logits);
    return generated;
}

// Server path: continue speculative generation from the ALREADY-prefilled engine
// state. `history` is the prompt tokens currently in the cache (for n-gram drafting),
// `first_logits` the post-prefill logits predicting the first generated token. Does
// NOT prefill. Same exact-distribution guarantees as gemma4_engine_generate_spec.
int gemma4_engine_generate_spec_continue(
    gemma4_engine_t *eng,
    const int32_t   *history, int n_history,
    const float     *first_logits,
    int32_t         *out_tokens, int max_new,
    const int32_t   *stop_ids, int n_stop,
    int              draft_k,
    float            temp, int top_k, float top_p, float min_p, float repeat_penalty,
    uint64_t seed,
    int             *n_accepted_out)
{
    return gemma4_engine_generate_spec_stream(eng, history, n_history, first_logits,
        out_tokens, max_new, stop_ids, n_stop, draft_k,
        temp, top_k, top_p, min_p, repeat_penalty, seed, n_accepted_out, NULL, NULL);
}

// =========================================================================
// ─── Sampling ───────────────────────────────────────────────────────────
// =========================================================================

int gemma4_sample_argmax(
    const float *logits,
    int          vocab_size)
{
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

// =========================================================================
// ─── GPU-side sampling (no logits D2H) ──────────────────────────────────
// =========================================================================
// One block samples the whole vocab on the GPU and writes a single token id, so a
// decode step returns 4 bytes instead of copying 262k logits to host and selecting
// on the CPU. temp<=0 → argmax. Otherwise the full pipeline: temperature → top-k
// (value-threshold via binary search, no global sort) → softmax → top-p → min-p →
// multinomial(rnd). Candidates above the top-k threshold (≤ SAMP_MAXK) are gathered
// to shared memory and the small tail (sort/top-p/min-p/draw) runs on one thread.
// Logits are expected to already carry softcap + suppression.
#define SAMP_MAXK 256
// Block-cooperative sampler BODY (one block, blockDim.x threads). Factored out of
// sample_logits_kernel so both the single-row decode sampler and the batched
// spec-verify sampler (one block per draft row) share identical math — the verify
// MUST sample each row from the exact same distribution as a plain decode, else
// speculative decoding would not be distribution-exact. Writes one id to *out_id.
__device__ __forceinline__ void sample_logit_row(
    const float *logits, int V, float temp, int top_k, float top_p, float min_p,
    float rnd, int *out_id)
{
    const int T = blockDim.x, tid = threadIdx.x;
    __shared__ float red[32];
    __shared__ int   sCount;

    // ── Greedy: block argmax ──
    if (temp <= 0.0f) {
        float best = -INFINITY; int bi = 0;
        for (int i = tid; i < V; i += T) { float v = logits[i]; if (v > best) { best = v; bi = i; } }
        // reduce (value,index) across the block via shared memory
        __shared__ float vred[1024]; __shared__ int ired[1024];
        vred[tid] = best; ired[tid] = bi; __syncthreads();
        for (int s = T >> 1; s > 0; s >>= 1) {
            if (tid < s && vred[tid+s] > vred[tid]) { vred[tid] = vred[tid+s]; ired[tid] = ired[tid+s]; }
            __syncthreads();
        }
        if (tid == 0) *out_id = ired[0];
        return;
    }

    // ── 1. global max (for numerically-stable softmax + threshold hi bound) ──
    float m = -INFINITY;
    for (int i = tid; i < V; i += T) m = fmaxf(m, logits[i]);
    m = block_reduce_max(m, red);
    __shared__ float sM; if (tid == 0) sM = m; __syncthreads(); m = sM;

    // ── 2. top-k value threshold via binary search on the logit value ──
    __shared__ float sThr;
    if (top_k > 0 && top_k < V) {
        float mn = INFINITY;
        for (int i = tid; i < V; i += T) mn = fminf(mn, logits[i]);
        mn = -block_reduce_max(-mn, red);
        // Suppressed tokens are -inf, which would make sLo = -inf, mid = -inf and the
        // search never move sHi off the global max — thr == max degenerates EVERY temp>0
        // top-k draw to argmax-candidates only (different seeds, identical output). All
        // real logits are softcapped to ±GEMMA4_SOFTCAP, so the true top-k threshold is
        // ≥ -GEMMA4_SOFTCAP and a finite floor is always correct.
        mn = fmaxf(mn, -GEMMA4_SOFTCAP);
        __shared__ float sLo, sHi; if (tid == 0) { sLo = mn; sHi = m; } __syncthreads();
        for (int it = 0; it < 32; it++) {
            float mid = 0.5f * (sLo + sHi);
            int c = 0;
            for (int i = tid; i < V; i += T) if (logits[i] >= mid) c++;
            float cf = block_reduce_sum((float)c, red);
            if (tid == 0) { if ((int)cf > top_k) sLo = mid; else sHi = mid; }
            __syncthreads();
        }
        // sHi guarantees count ≤ top_k (≤ SAMP_MAXK), so every candidate fits
        // in the gather buffer — no random truncation.  sLo would admit MORE than
        // top_k, risking loss of true top-K tokens when count >> SAMP_MAXK.
        if (tid == 0) sThr = sHi;
    } else if (tid == 0) sThr = -INFINITY;
    __syncthreads();
    float thr = sThr;

    // ── 3. gather candidates >= threshold into shared memory (capped) ──
    __shared__ int   cid[SAMP_MAXK];
    __shared__ float clg[SAMP_MAXK];
    if (tid == 0) sCount = 0; __syncthreads();
    for (int i = tid; i < V; i += T) {
        if (logits[i] >= thr) {
            int p = atomicAdd(&sCount, 1);
            if (p < SAMP_MAXK) { cid[p] = i; clg[p] = logits[i]; }
        }
    }
    __syncthreads();
    int N = sCount < SAMP_MAXK ? sCount : SAMP_MAXK;

    // ── 4. small tail on one thread: sort desc, softmax, top-p, min-p, draw ──
    if (tid == 0) {
        for (int a = 1; a < N; a++) {          // insertion sort, N ≤ SAMP_MAXK
            int ci = cid[a]; float cv = clg[a]; int b = a - 1;
            while (b >= 0 && clg[b] < cv) { clg[b+1] = clg[b]; cid[b+1] = cid[b]; b--; }
            clg[b+1] = cv; cid[b+1] = ci;
        }
        float invT = 1.0f / temp, mx = clg[0], sum = 0.0f;
        float pr[SAMP_MAXK];
        for (int a = 0; a < N; a++) { pr[a] = __expf((clg[a]-mx)*invT); sum += pr[a]; }
        for (int a = 0; a < N; a++) pr[a] /= sum;
        if (top_p > 0.0f && top_p < 1.0f) {
            float cum = 0.0f; int cut = N;
            for (int a = 0; a < N; a++) { cum += pr[a]; if (cum >= top_p) { cut = a+1; break; } }
            N = cut;
        }
        if (min_p > 0.0f) {
            float th = min_p * pr[0]; int keep = 0;
            for (int a = 0; a < N; a++) { if (pr[a] >= th) keep = a+1; else break; }
            if (keep > 0) N = keep;
        }
        float z = 0.0f; for (int a = 0; a < N; a++) z += pr[a];
        float r = rnd * z, acc = 0.0f; int sel = cid[N > 0 ? N-1 : 0];
        for (int a = 0; a < N; a++) { acc += pr[a]; if (r <= acc) { sel = cid[a]; break; } }
        *out_id = sel;
    }
}

// Single-row wrapper (unchanged public behavior): one block samples eng->d_logits.
__global__ void sample_logits_kernel(
    const float *logits, int V, float temp, int top_k, float top_p, float min_p,
    float rnd, int *out_id)
{
    sample_logit_row(logits, V, temp, top_k, top_p, min_p, rnd, out_id);
}

// Batched spec-verify sampler: one block per draft row. Block b samples row
// logits + b*V with its own draw rnds[b] and writes out_ids[b]. Same per-row math
// as the single decode sampler, so each verify position is a valid target sample
// (greedy: argmax → bit-identical to a plain decode at that position).
__global__ void sample_logits_batched_kernel(
    const float *logits, int V, float temp, int top_k, float top_p, float min_p,
    const float *rnds, int *out_ids)
{
    int row = blockIdx.x;
    sample_logit_row(logits + (size_t)row * V, V, temp, top_k, top_p, min_p,
                     rnds[row], out_ids + row);
}

// Multi-sequence per-row sampler for the continuous-batching batch path: one
// block samples row r of logits[B][V] with row r's OWN params (temp/top_k/top_p/
// min_p) and its own uniform draw rnds[r]. Each row is an independent in-flight
// sequence, so params differ across rows. temp<=0 takes a greedy argmax with the
// SAME lowest-index tie-break as argmax_rows_kernel, so a temp==0 row is
// byte-identical to the greedy batch path (the batch self-test stays 32/32);
// temp>0 rows go through the shared sample_logit_row pipeline.
__global__ void sample_logits_ms_kernel(
    const float *logits, int V,
    const float *temps, const int *top_ks, const float *top_ps, const float *min_ps,
    const float *rnds, int *out_ids)
{
    int row = blockIdx.x;
    const float *lr = logits + (size_t)row * V;
    float temp = temps[row];
    if (temp <= 0.0f) {
        // Greedy: block-wide argmax, lowest index wins ties (== argmax_rows_kernel).
        int tid = threadIdx.x, T = blockDim.x;
        float best_val = -1e30f; int best_idx = 0;
        for (int i = tid; i < V; i += T) { if (lr[i] > best_val) { best_val = lr[i]; best_idx = i; } }
        // Reduction arrays sized for the max blockDim.x (1024). This MUST match the
        // launch config in decode_multiseq_forward (<<<B,1024,...>>>); changing the
        // block size there requires resizing these arrays.
        __shared__ float vred[1024]; __shared__ int ired[1024];
        vred[tid] = best_val; ired[tid] = best_idx; __syncthreads();
        for (int s = T >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                float ov = vred[tid+s]; int oi = ired[tid+s];
                if (ov > vred[tid] || (ov == vred[tid] && oi < ired[tid])) { vred[tid] = ov; ired[tid] = oi; }
            }
            __syncthreads();
        }
        if (tid == 0) out_ids[row] = ired[0];
        return;
    }
    sample_logit_row(lr, V, temp, top_ks[row], top_ps[row], min_ps[row],
                     rnds[row], out_ids + row);
}

// Sample a token from the engine's resident logits (eng->d_logits) entirely on the
// GPU; only the 4-byte token id is copied to host. rnd ∈ [0,1) is the host RNG draw
// (unused for greedy). Returns the token id, or -1 on error.
int gemma4_engine_sample_device(
    gemma4_engine_t *eng, float temp, int top_k, float top_p, float min_p, float rnd)
{
    if (!eng || !eng->loaded) return -1;
    if (!eng->d_sample_id) {
        if (cudaMalloc(&eng->d_sample_id, sizeof(int)) != cudaSuccess) {
            cudaGetLastError(); return -1;
        }
    }
    sample_logits_kernel<<<1, 1024, 0, eng->stream>>>(
        eng->d_logits, GEMMA4_VOCAB_SIZE, temp, top_k, top_p, min_p, rnd,
        eng->d_sample_id);
    int id = -1;
    cudaMemcpyAsync(&id, eng->d_sample_id, sizeof(int), cudaMemcpyDeviceToHost, eng->stream);
    cudaStreamSynchronize(eng->stream);
    if (cudaGetLastError() != cudaSuccess) return -1;
    return id;
}

// =========================================================================
// ─── LoRA Public API ────────────────────────────────────────────────────
// =========================================================================




// =========================================================================
// ─── Diagnostics ────────────────────────────────────────────────────────
// =========================================================================

void gemma4_engine_print_info(const gemma4_engine_t *eng) {
    if (!eng) return;
    printf("=== fucina Engine Info ===\n");
    printf("Model size:  %.2f GB\n", eng->gguf_size / (1024.0*1024.0*1024.0));
    printf("Context:     %u tokens\n", eng->context_size);
    printf("Format:      %s\n", eng->format == FORMAT_Q4_0 ? "Q4_0" : "Q8_0");
    printf("Layers:      %d total (%d sliding, %d global)\n",
            eng->cfg.n_layers, eng->n_layers_sliding, eng->n_layers_global);
    printf("Hidden:      %d -> %d -> %d\n",
            eng->cfg.hidden_size, eng->cfg.intermediate, eng->cfg.hidden_size);
    printf("Heads:       %d Q, %d KV sliding, %d KV global\n",
            eng->cfg.n_heads, eng->cfg.n_kv_sliding, eng->cfg.n_kv_global);
    printf("Head dim:    %d sliding, %d global\n",
            GEMMA4_HEAD_DIM, GEMMA4_GLOBAL_HEAD_DIM);
    printf("Sliding win: %d tokens\n", GEMMA4_SLIDING_WINDOW);
    printf("Vocab:       %d\n", GEMMA4_VOCAB_SIZE);
    printf("Global layers at: ");
    for (int i = 0; i < eng->n_global; i++)
        printf("%d ", eng->global_layer_indices[i]);
    printf("\n");
    printf("Device:      %d (%s)\n", eng->device_id, "NVIDIA GB10 Blackwell");
    printf("Memory:      %.1f GB free / %.1f GB total\n",
            eng->free_mem / (1024.0*1024.0*1024.0),
            eng->total_mem / (1024.0*1024.0*1024.0));
}

void gemma4_engine_print_timing(const gemma4_engine_t *eng) {
    if (!eng) return;
    decode_timing_lap((gemma4_engine_t *)eng);   // resolve any completed lazy pair
    printf("=== fucina Timing ===\n");
    if (eng->n_prefill_tokens > 0) {
        printf("Prefill:  %d tokens in %.1f ms = %.0f t/s\n",
                eng->n_prefill_tokens, eng->prefill_time_ms,
                eng->n_prefill_tokens / (eng->prefill_time_ms / 1000.0f));
    }
    if (eng->n_decode_tokens > 0) {
        printf("Decode:   %d tokens in %.1f ms = %.1f t/s\n",
                eng->n_decode_tokens, eng->decode_time_ms,
                eng->n_decode_tokens / (eng->decode_time_ms / 1000.0f));
    }
    if (eng->fp4m_ssd_stream) {
        double hr=(eng->fp4m_cache_hits+eng->fp4m_cache_misses)?
            (double)eng->fp4m_cache_hits/(eng->fp4m_cache_hits+eng->fp4m_cache_misses):0.0;
        printf("Experts:  SSD reads=%llu bytes=%.3f GiB checksum_failures=%llu slots=%d cache_hit=%.1f%% (%llu/%llu) prefetch=%llu\n",
               (unsigned long long)eng->fp4m_ssd_reads,eng->fp4m_ssd_bytes/(1024.0*1024*1024),
               (unsigned long long)eng->fp4m_ssd_checksum_fail,eng->fp4m_slots,100.0*hr,
               (unsigned long long)eng->fp4m_cache_hits,
               (unsigned long long)(eng->fp4m_cache_hits+eng->fp4m_cache_misses),
               (unsigned long long)eng->fp4m_prefetch_advice);
    }
}

int gemma4_engine_get_n_layers(const gemma4_engine_t *eng) {
    return eng ? eng->cfg.n_layers : 0;
}

// Detected expert count (0 = dense). The Go scheduler uses this to gate speculative
// decoding OFF for sparse models: a K-token verify re-reads each drafted token's OWN
// top-k experts (the dominant weight bytes do NOT amortize across draft rows, unlike a
// dense model's single weight pass), so spec costs ~K× expert bandwidth for <K accepted
// tokens — it doesn't bring value on MoE, exactly the case the goal says to avoid.
int gemma4_engine_n_experts(const gemma4_engine_t *eng) {
    return (eng && eng->loaded) ? eng->cfg.n_experts : 0;
}

int gemma4_engine_moe_profile_shape(const gemma4_engine_t *eng,
                                    int *n_layers, int *n_experts, int *top_k) {
    if (!eng || !eng->loaded || eng->cfg.n_experts <= 0) return -1;
    if (n_layers) *n_layers = eng->cfg.n_layers;
    if (n_experts) *n_experts = eng->cfg.n_experts;
    if (top_k) *top_k = eng->cfg.n_experts_used;
    return 0;
}

int gemma4_engine_moe_profile_start(gemma4_engine_t *eng) {
    if (!eng || !eng->loaded || eng->cfg.n_experts <= 0) return -1;
    const size_t n = (size_t)eng->cfg.n_layers * eng->cfg.n_experts;
    if (!eng->d_moe_profile_count &&
        cudaMalloc(&eng->d_moe_profile_count, n * sizeof(unsigned long long)) != cudaSuccess)
        return -1;
    if (!eng->d_moe_profile_weight &&
        cudaMalloc(&eng->d_moe_profile_weight, n * sizeof(double)) != cudaSuccess) {
        cudaFree(eng->d_moe_profile_count); eng->d_moe_profile_count = NULL;
        return -1;
    }
    const size_t na = (size_t)eng->cfg.n_layers * MOE_PROFILE_ACT_STAGES;
    bool act_ok = (eng->d_moe_profile_act_ss || cudaMalloc(&eng->d_moe_profile_act_ss, na*sizeof(double)) == cudaSuccess)
               && (eng->d_moe_profile_act_n || cudaMalloc(&eng->d_moe_profile_act_n, na*sizeof(unsigned long long)) == cudaSuccess)
               && (eng->d_moe_profile_act_max || cudaMalloc(&eng->d_moe_profile_act_max, na*sizeof(unsigned int)) == cudaSuccess);
    if (!act_ok) return -1;
    cudaMemsetAsync(eng->d_moe_profile_count, 0, n * sizeof(unsigned long long), eng->stream);
    cudaMemsetAsync(eng->d_moe_profile_weight, 0, n * sizeof(double), eng->stream);
    cudaMemsetAsync(eng->d_moe_profile_act_ss, 0, na * sizeof(double), eng->stream);
    cudaMemsetAsync(eng->d_moe_profile_act_n, 0, na * sizeof(unsigned long long), eng->stream);
    cudaMemsetAsync(eng->d_moe_profile_act_max, 0, na * sizeof(unsigned int), eng->stream);
    return cudaStreamSynchronize(eng->stream) == cudaSuccess ? 0 : -1;
}

int gemma4_engine_moe_profile_snapshot(gemma4_engine_t *eng,
                                       uint64_t *counts, double *weight_sums,
                                       size_t capacity) {
    if (!eng || !counts || !weight_sums || !eng->d_moe_profile_count) return -1;
    const size_t n = (size_t)eng->cfg.n_layers * eng->cfg.n_experts;
    if (capacity < n || cudaStreamSynchronize(eng->stream) != cudaSuccess) return -1;
    if (cudaMemcpy(counts, eng->d_moe_profile_count, n * sizeof(uint64_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    if (cudaMemcpy(weight_sums, eng->d_moe_profile_weight, n * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    return 0;
}

int gemma4_engine_moe_profile_activation_snapshot(gemma4_engine_t *eng,
                                       double *sum_squares, uint64_t *elements,
                                       float *max_abs, size_t capacity) {
    if (!eng || !sum_squares || !elements || !max_abs || !eng->d_moe_profile_act_ss) return -1;
    const size_t n = (size_t)eng->cfg.n_layers * MOE_PROFILE_ACT_STAGES;
    if (capacity < n || cudaStreamSynchronize(eng->stream) != cudaSuccess) return -1;
    std::vector<unsigned int> bits(n);
    if (cudaMemcpy(sum_squares, eng->d_moe_profile_act_ss, n*sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(elements, eng->d_moe_profile_act_n, n*sizeof(uint64_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(bits.data(), eng->d_moe_profile_act_max, n*sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    for (size_t i=0;i<n;i++) memcpy(max_abs+i, bits.data()+i, sizeof(float));
    return 0;
}

// Detected architecture family. The Qwen3/Qwen3-MoE forward is served ONLY through
// the paged multiseq + continuous-batching path (single-flight prefill declines),
// so the Go server uses this to auto-enable the batch scheduler for those models.
int gemma4_engine_is_qwen3_family(const gemma4_engine_t *eng) {
    // qwen35 is part of the Qwen3 family for SERVING-CONTROL purposes (the Go server uses
    // this to auto-enable continuous batching and to reject the single-flight path): it is
    // served exclusively through the paged-multiseq ABI (seq_add/step_batch route to the
    // qwen35 hybrid impls). It is intentionally NOT in the GEMMA4_IS_QWEN3_FAMILY macro,
    // which gates the fp8 paged-KV POOL allocation — qwen35 carries its own fp32 GDN/conv/
    // FULL-KV arenas instead, so it must skip that pool path. Hence the explicit OR here.
    return (eng && (0 ||
                    eng->cfg.arch == GEMMA4_ARCH_QWEN3_5)) ? 1 : 0;
}

// ─── Timing accessors (for Go-side speed logging) ─────────────────────
float gemma4_engine_prefill_ms(const gemma4_engine_t *eng) {
    return eng ? eng->prefill_time_ms : 0.0f;
}
int gemma4_engine_prefill_tokens(const gemma4_engine_t *eng) {
    return eng ? eng->n_prefill_tokens : 0;
}
float gemma4_engine_decode_ms(const gemma4_engine_t *eng) {
    if (!eng) return 0.0f;
    decode_timing_lap((gemma4_engine_t *)eng);   // resolve any completed lazy pair
    return eng->decode_time_ms;
}
int gemma4_engine_decode_tokens(const gemma4_engine_t *eng) {
    if (!eng) return 0;
    decode_timing_lap((gemma4_engine_t *)eng);
    return eng->n_decode_tokens;
}
long gemma4_engine_spec_steps(const gemma4_engine_t *eng)    { return eng ? eng->spec_steps    : 0; }
long gemma4_engine_spec_drafted(const gemma4_engine_t *eng)  { return eng ? eng->spec_drafted  : 0; }
long gemma4_engine_spec_accepted(const gemma4_engine_t *eng) { return eng ? eng->spec_accepted : 0; }
long gemma4_engine_spec_emitted(const gemma4_engine_t *eng)  { return eng ? eng->spec_emitted  : 0; }

int gemma4_engine_get_context_size(const gemma4_engine_t *eng) {
    return eng ? eng->context_size : 0;
}

// ─── KV cache state management ────────────────────────────────────────
//
// The engine holds a single append-only KV cache (sliding-window ring buffers
// for local layers + a linear FP8 cache for global layers). These helpers let
// the server implement prefix reuse:
//
//   - gemma4_engine_n_tokens(): how many tokens are currently materialized in
//     the KV cache (i.e. the global cursor position).
//   - gemma4_engine_reset():    drop all cached KV state so the next Prefill
//     starts a fresh sequence at position 0.
//
// Resetting does NOT free or zero device memory; it only rewinds the cursors,
// which is sufficient because attention reads are bounded by these counters.
int gemma4_engine_n_tokens(const gemma4_engine_t *eng) {
    return eng ? eng->cur.n_tokens : 0;
}

void gemma4_engine_reset(gemma4_engine_t *eng) {
    if (!eng) return;
    eng->cur.n_tokens = 0;
    eng->mtp_h_valid = 0;
}

// Rewind the KV cache to keep only the first `n_keep` tokens, discarding the
// rest. This enables prefix reuse: when a new request shares a prefix with the
// cached sequence, we rewind to the shared length and prefill only the suffix.
//
// Correctness with the FLAT per-position KV buffers (DECODE-30-35 Step 3):
//   Both the global cache and (now) the sliding cache store K/V for absolute token t at
//   index t — nothing is ever overwritten within the context, so rewinding to any n_keep
//   simply drops positions [n_keep, L) and the kept prefix [0, n_keep) is bit-intact. The
//   old ring REFUSED once it wrapped past the 1024 window (it lost evicted keys); the flat
//   buffer makes EVERY rewind exact, which is what re-enables speculation past 1024 ctx and
//   correct multi-turn prefix reuse (the path commit 4bdb431 had to revert). Always succeeds.
//
// Returns 0 on success, -1 only on a bad argument.
int gemma4_engine_rewind(gemma4_engine_t *eng, int n_keep) {
    if (!eng) return -1;
    if (n_keep < 0 || n_keep > eng->cur.n_tokens) return -1;
    if (n_keep == eng->cur.n_tokens) return 0; // nothing to discard

    // Ring sliding cache: once the sequence has exceeded the ring capacity, only the
    // last `cap` positions survive, so the sliding window for n_keep is intact only if
    // n_keep >= H - (cap - window). A deeper rewind cannot be served exactly — report
    // failure so KVCache.Prefill falls back to a full re-prefill (Rewind()==false). When
    // H <= cap nothing has been overwritten and every rewind is exact, as before.
    if (eng->cur.n_tokens > eng->sliding_kv_capacity &&
        n_keep < eng->cur.n_tokens -
                 (eng->sliding_kv_capacity - GEMMA4_SLIDING_WINDOW)) {
        return -1;
    }

    eng->cur.n_tokens = n_keep;
    return 0;
}

// ─── KV sequence snapshot / restore ──────────────────────────────────
//
// With the flat per-position layout, a sequence's complete KV state is the
// first n_tokens positions of each layer's K/V region — four strided 2D
// copies (sliding K/V over all 48 layer slots, global K/V over the compact
// global-layer slots). The host buffer layout is simply those four regions
// concatenated. Used by the server's multi-conversation prefix cache: saving
// the live conversation before an unrelated request evicts it turns the
// later "switch back" from a full re-prefill (~100 s at 20k ctx) into a
// memcpy (~ms on unified memory).

// kv_seq_pitches fills the per-buffer row sizes (bytes per token per layer)
// and layer pitches (bytes between consecutive layers' regions).
static void kv_seq_pitches(const gemma4_engine_t *eng,
                           size_t *srow, size_t *spitch,
                           size_t *grow, size_t *gpitch) {
    *srow   = (size_t)eng->cfg.n_kv_sliding * GEMMA4_HEAD_DIM * sizeof(kv_t);
    *spitch = (size_t)eng->sliding_kv_capacity * (*srow);
    *grow   = (size_t)eng->cfg.n_kv_global * GEMMA4_GLOBAL_HEAD_DIM * sizeof(kv_t);
    *gpitch = (size_t)eng->global_kv_capacity * (*grow);
}

size_t gemma4_engine_kv_state_size(const gemma4_engine_t *eng, int n_tokens) {
    if (!eng || n_tokens <= 0 || n_tokens > eng->sliding_kv_capacity ||
        n_tokens > eng->global_kv_capacity) return 0;
    size_t srow, spitch, grow, gpitch;
    kv_seq_pitches(eng, &srow, &spitch, &grow, &gpitch);
    return 2 * (size_t)eng->cfg.n_layers    * (size_t)n_tokens * srow +
           2 * (size_t)eng->n_layers_global * (size_t)n_tokens * grow;
}

// Pinned staging size per buffer (two buffers; chunks alternate between them
// so the DMA of chunk i overlaps the host memcpy of chunk i−1).
#define GEMMA4_KV_STAGE_BYTES (64u << 20)

static int ensure_kv_stage(gemma4_engine_t *eng) {
    if (eng->kv_stage_ready) return eng->kv_stage_ready;
    for (int b = 0; b < 2; b++) {
        if (cudaHostAlloc(&eng->h_kv_stage[b], GEMMA4_KV_STAGE_BYTES,
                          cudaHostAllocDefault) != cudaSuccess ||
            cudaEventCreateWithFlags(&eng->ev_kv_stage[b],
                                     cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            for (int j = 0; j < 2; j++) {
                if (eng->h_kv_stage[j])  { cudaFreeHost(eng->h_kv_stage[j]); eng->h_kv_stage[j] = NULL; }
                if (eng->ev_kv_stage[j]) { cudaEventDestroy(eng->ev_kv_stage[j]); eng->ev_kv_stage[j] = NULL; }
            }
            fprintf(stderr, "fucina: kv-snapshot pinned staging alloc failed — pageable fallback\n");
            eng->kv_stage_ready = -1;
            return -1;
        }
    }
    eng->kv_stage_ready = 1;
    return 1;
}

// kv_seq_copy moves the first n_tokens of all four KV regions between the
// engine and a host buffer. dir: cudaMemcpyDeviceToHost (save) or
// cudaMemcpyHostToDevice (restore).
//
// Fast path: the host side (the Go snapshot pool) is PAGEABLE, and a direct
// pageable cudaMemcpy2D measured ~120 MB/s (a 4.3 GB save stalled requests
// ~20-35 s). Each region row is a CONTIGUOUS per-layer block, so the copy is
// really 112 flat blocks — stream them through the two pinned staging
// buffers: async DMA on eng->stream (which also orders the save after all
// prior device work) ping-ponging with a host memcpy between staging and the
// pageable buffer. Falls back to the old synchronous Memcpy2D when pinned
// allocation is unavailable.
static int kv_seq_copy(gemma4_engine_t *eng, char *buf, int n_tokens,
                       enum cudaMemcpyKind dir) {
    size_t srow, spitch, grow, gpitch;
    kv_seq_pitches(eng, &srow, &spitch, &grow, &gpitch);
    const size_t swidth = (size_t)n_tokens * srow; // contiguous bytes per layer
    const size_t gwidth = (size_t)n_tokens * grow;

    struct { char *dev; size_t pitch, width; int layers; } regions[4] = {
        { (char *)eng->d_sliding_k, spitch, swidth, eng->cfg.n_layers },
        { (char *)eng->d_sliding_v, spitch, swidth, eng->cfg.n_layers },
        { (char *)eng->d_global_k,  gpitch, gwidth, eng->n_layers_global },
        { (char *)eng->d_global_v,  gpitch, gwidth, eng->n_layers_global },
    };

    if (ensure_kv_stage(eng) == 1) {
        cudaStream_t stream = eng->stream;
        // In-flight chunk per staging buffer (host side of the pending copy).
        struct { char *host; size_t bytes; } pend[2] = {{NULL,0},{NULL,0}};
        int b = 0, fail = 0;
        for (int r = 0; r < 4 && !fail; r++) {
            for (int l = 0; l < regions[r].layers && !fail; l++) {
                char *dev = regions[r].dev + (size_t)l * regions[r].pitch;
                for (size_t off = 0; off < regions[r].width && !fail; off += GEMMA4_KV_STAGE_BYTES) {
                    size_t sz = regions[r].width - off;
                    if (sz > GEMMA4_KV_STAGE_BYTES) sz = GEMMA4_KV_STAGE_BYTES;
                    if (dir == cudaMemcpyDeviceToHost) {
                        // Drain the buffer's previous chunk (DMA done → host memcpy out).
                        if (pend[b].host) {
                            if (cudaEventSynchronize(eng->ev_kv_stage[b]) != cudaSuccess) { fail = 1; break; }
                            memcpy(pend[b].host, eng->h_kv_stage[b], pend[b].bytes);
                        }
                        if (cudaMemcpyAsync(eng->h_kv_stage[b], dev + off, sz, dir, stream) != cudaSuccess ||
                            cudaEventRecord(eng->ev_kv_stage[b], stream) != cudaSuccess) { fail = 1; break; }
                        pend[b].host = buf; pend[b].bytes = sz;
                    } else {
                        // Reuse gate: the buffer's previous H2D must have completed.
                        if (pend[b].host &&
                            cudaEventSynchronize(eng->ev_kv_stage[b]) != cudaSuccess) { fail = 1; break; }
                        memcpy(eng->h_kv_stage[b], buf, sz);
                        if (cudaMemcpyAsync(dev + off, eng->h_kv_stage[b], sz, dir, stream) != cudaSuccess ||
                            cudaEventRecord(eng->ev_kv_stage[b], stream) != cudaSuccess) { fail = 1; break; }
                        pend[b].host = buf; pend[b].bytes = sz;
                    }
                    buf += sz;
                    b ^= 1;
                }
            }
        }
        if (!fail) {
            for (int j = 0; j < 2; j++) {   // drain both buffers
                if (!pend[j].host) continue;
                if (cudaEventSynchronize(eng->ev_kv_stage[j]) != cudaSuccess) { fail = 1; break; }
                if (dir == cudaMemcpyDeviceToHost)
                    memcpy(pend[j].host, eng->h_kv_stage[j], pend[j].bytes);
            }
        }
        if (!fail) return 0;
        fprintf(stderr, "fucina: staged kv_seq_copy failed (%s) — engine state %s\n",
                cudaGetErrorString(cudaGetLastError()),
                dir == cudaMemcpyDeviceToHost ? "intact (save aborted)" : "UNDEFINED");
        return -1;
    }

    // Fallback: synchronous pageable Memcpy2D (orders after prior device work).
    for (int r = 0; r < 4; r++) {
        cudaError_t err;
        if (dir == cudaMemcpyDeviceToHost) {
            err = cudaMemcpy2D(buf, regions[r].width,
                               regions[r].dev, regions[r].pitch,
                               regions[r].width, regions[r].layers, dir);
        } else {
            err = cudaMemcpy2D(regions[r].dev, regions[r].pitch,
                               buf, regions[r].width,
                               regions[r].width, regions[r].layers, dir);
        }
        if (err != cudaSuccess) {
            fprintf(stderr, "fucina: kv_seq_copy region %d failed: %s\n",
                    r, cudaGetErrorString(err));
            return -1;
        }
        buf += regions[r].width * regions[r].layers;
    }
    return 0;
}

int gemma4_engine_kv_save(gemma4_engine_t *eng, void *buf, int n_tokens) {
    if (!eng || !buf || n_tokens <= 0 || n_tokens > eng->cur.n_tokens) return -1;
    return kv_seq_copy(eng, (char *)buf, n_tokens, cudaMemcpyDeviceToHost);
}

int gemma4_engine_kv_restore(gemma4_engine_t *eng, const void *buf, int n_tokens) {
    if (!eng || !buf || n_tokens <= 0 ||
        n_tokens > eng->sliding_kv_capacity || n_tokens > eng->global_kv_capacity)
        return -1;
    if (kv_seq_copy(eng, (char *)buf, n_tokens, cudaMemcpyHostToDevice) != 0)
        return -1;
    eng->cur.n_tokens = n_tokens;
    eng->mtp_h_valid = 0; // the MTP draft state belonged to the replaced sequence
    return 0;
}

// ─── Persistent prefill scratch allocator ───────────────────────────
static int alloc_prefill_scratch(gemma4_engine_t *eng) {
    if (eng->pf_scratch_ready) return 0;
    const int NC = 4096, H = eng->cfg.hidden_size, I = eng->cfg.intermediate;
    const int OQ = eng->cfg.n_heads * GEMMA4_GLOBAL_HEAD_DIM;
    const int OK = eng->cfg.n_kv_sliding * GEMMA4_HEAD_DIM;
    const int HE = eng->cfg.n_heads;
    cudaError_t ok = cudaSuccess;
    #define A(p,sz) if(ok==cudaSuccess) ok=cudaMalloc(&(p),(size_t)(sz))
    A(eng->d_pf_x, (size_t)NC*H*sizeof(float));
    A(eng->d_pf_norm, (size_t)NC*H*sizeof(float));
    A(eng->d_pf_q, (size_t)NC*OQ*sizeof(float));
    A(eng->d_pf_k, (size_t)NC*OK*sizeof(float));
    A(eng->d_pf_v, (size_t)NC*OK*sizeof(float));
    A(eng->d_pf_attn, (size_t)NC*OQ*sizeof(float));
    A(eng->d_pf_gate, (size_t)NC*I*sizeof(float));
    A(eng->d_pf_up, (size_t)NC*I*sizeof(float));
    A(eng->d_pf_scores, (size_t)HE*NC*NC*sizeof(float));
    A(eng->d_pf_inb, (size_t)NC*I*sizeof(__nv_bfloat16));
    A(eng->d_pf_qb, (size_t)NC*OQ*sizeof(__nv_bfloat16));
    A(eng->d_pf_kb, (size_t)NC*OK*sizeof(__nv_bfloat16));
    A(eng->d_pf_vb, (size_t)NC*OK*sizeof(__nv_bfloat16));
    A(eng->d_pf_kbx, (size_t)NC*OQ*sizeof(__nv_bfloat16));
    A(eng->d_pf_vbx, (size_t)NC*OQ*sizeof(__nv_bfloat16));
    A(eng->d_pf_pb, (size_t)HE*NC*NC*sizeof(__nv_bfloat16));
    // Paged single-pass prefill: write_pos iota + replicated per-row views.
    A(eng->d_pf_wpos, (size_t)NC*sizeof(int));
    A(eng->d_pf_views_slid, (size_t)NC*sizeof(PagedSeqView));
    A(eng->d_pf_views_glob, (size_t)NC*sizeof(PagedSeqView));
    #undef A
    if (ok != cudaSuccess) { cudaGetLastError(); return -1; }
    {   // fill the write_pos iota once (engine-lifetime, never changes)
        int *h_wpos = (int*)malloc((size_t)NC*sizeof(int));
        if (h_wpos) {
            for (int i = 0; i < NC; i++) h_wpos[i] = i;
            cudaMemcpy(eng->d_pf_wpos, h_wpos, (size_t)NC*sizeof(int), cudaMemcpyHostToDevice);
            free(h_wpos);
        } else { return -1; }
    }
    eng->pf_scratch_ready = 1;
    fprintf(stderr, "fucina: persistent prefill scratch (N<=%d)\n", NC);
    return 0;
}

// Tiled-GEMM flash-prefill attention scratch (~2.1 GB), lazy on first flash
// prefill, engine-lifetime. Sized for the widest layer (global oq = 16×512)
// at the full chunk; smaller chunks/layers just pack tighter (runtime ld /
// strides). Failure is non-fatal: the caller defers to the chunked-decode
// fallback, same as a BF16-weights failure.
static int ensure_fp_scratch(gemma4_engine_t *eng) {
    if (eng->fp_scratch_ready) return 0;
    const size_t C  = GEMMA4_FP_CHUNK, T = GEMMA4_FP_TILE_K;
    const size_t OQ = (size_t)eng->cfg.n_heads * GEMMA4_GLOBAL_HEAD_DIM;
    const size_t HC = (size_t)eng->cfg.n_heads * C;
    cudaError_t ok = cudaSuccess;
    #define A(p,sz) if(ok==cudaSuccess) ok=cudaMalloc(&(p),(size_t)(sz))
    A(eng->d_fp_qb,  C*OQ*sizeof(__nv_bfloat16));
    A(eng->d_fp_kbx, C*OQ*sizeof(__nv_bfloat16));
    A(eng->d_fp_vbx, C*OQ*sizeof(__nv_bfloat16));
    A(eng->d_fp_kt,  T*OQ*sizeof(__nv_bfloat16));
    A(eng->d_fp_vt,  T*OQ*sizeof(__nv_bfloat16));
    A(eng->d_fp_st,  HC*T*sizeof(__nv_bfloat16));
    A(eng->d_fp_pb,  HC*T*sizeof(__nv_bfloat16));
    A(eng->d_fp_m,   HC*sizeof(float));
    A(eng->d_fp_l,   HC*sizeof(float));
    #undef A
    if (ok != cudaSuccess) {
        cudaGetLastError();
        __nv_bfloat16 *bb[] = {eng->d_fp_qb, eng->d_fp_kbx, eng->d_fp_vbx,
                               eng->d_fp_kt, eng->d_fp_vt, eng->d_fp_pb};
        for (__nv_bfloat16 *p : bb) cudaFree(p);
        cudaFree(eng->d_fp_st); cudaFree(eng->d_fp_m); cudaFree(eng->d_fp_l);
        eng->d_fp_qb = eng->d_fp_kbx = eng->d_fp_vbx = NULL;
        eng->d_fp_kt = eng->d_fp_vt = eng->d_fp_pb = eng->d_fp_st = NULL;
        eng->d_fp_m = eng->d_fp_l = NULL;
        return -1;
    }
    eng->fp_scratch_ready = 1;
    return 0;
}

void gemma4_engine_set_graph_mode(gemma4_engine_t *eng, int mode) {
    if (!eng) return;
    if (mode == 1 && !eng->pf_scratch_ready) {
        if (alloc_prefill_scratch(eng) != 0) { eng->graph_mode = 0; return; }
    }
    eng->graph_mode = (mode >= 0 && mode <= 1) ? mode : 0;
}

void gemma4_engine_graph_stats(const gemma4_engine_t *eng,
    int *hits, int *misses, int *captures, int *launches) {
    if (!eng) return;
    if (hits) *hits = eng->graph.hits;
    if (misses) *misses = eng->graph.misses;
    if (captures) *captures = eng->graph.N > 0 ? 1 : 0;
    if (launches) *launches = eng->graph.hits;
}

// Eagerly run the lazy first-prefill setup (persistent prefill scratch ≈2.9 GB +
// rotating BF16 dequant scratch ≈0.5 GB) so request #1's prefill timer measures
// prefill, not one-time cudaMallocs (~0.5-2.1 s otherwise charged to it). Both
// halves are idempotent. A scratch-alloc failure is non-fatal (the batched path
// defers to flash, which needs only the BF16 scratch).
int gemma4_engine_warmup(gemma4_engine_t *eng) {
    if (!eng || !eng->loaded) return -1;
    // qwen35 hybrid warmup: pre-pay the per-slot fp32 GDN/conv/FULL-KV arena allocations
    // (ensure_q35_scratch) AND capture the B=1 decode graph with a tiny throwaway sequence,
    // via its OWN paged ABI. The gemma/qwen3 warmup below (alloc_prefill_scratch +
    // build_bf16_weights + decode_graph_ensure/batched_graph_ensure + cuBLAS prefill passes)
    // is gemma-layout-only and would corrupt the qwen35 weights/context, so it must be skipped.
    if (eng->cfg.arch == GEMMA4_ARCH_QWEN3_5) {
        const char *nw0 = getenv("FUCINA_NO_WARMUP_PASS");
        if (nw0 && nw0[0] == '1') return 0;
        int32_t wp[5] = { 760, 6511, 314, 9338, 369 };  // "The capital of France is" (valid ids)
        int32_t first = 0;
        int slot = gemma4_engine_seq_add(eng, wp, 5, &first, 0.f, 0, 0.f, 0.f, 0);
        if (slot >= 0) {
            int sl = slot; int32_t in = first, out = 0;
            for (int i = 0; i < 2; i++) { gemma4_engine_step_batch(eng, &sl, &in, 1, &out); in = out; }
            gemma4_engine_seq_remove(eng, slot);
        }
        return 0;
    }
    int rc = 0;
    if (alloc_prefill_scratch(eng) != 0) rc = -1;
    if (build_bf16_weights(eng) != 0)    rc = -1;
    if (rc != 0) return rc;

    // Allocating the scratch is not enough to make the FIRST real prefill fast:
    // cuBLAS lazily inits its library + GEMM workspaces on the first cublasGemmEx,
    // CUDA lazily loads each kernel module on its first launch, and the persistent
    // scratch pages fault in on first touch. Paid on request #1, this is the whole
    // gap between fucina's cold prefill (~1385 tok/s) and its warm steady state
    // (~1714 tok/s). Run one real batched prefill over a representative-width dummy
    // prompt here so all of it is paid BEFORE the server announces "listening", then
    // rewind the sequence so the warmup leaves no KV state behind. (Escape hatch:
    // FUCINA_NO_WARMUP_PASS=1, mirroring FUCINA_NO_DECODE_GRAPH.)
    const char *nw = getenv("FUCINA_NO_WARMUP_PASS");
    if (nw && nw[0] == '1') return 0;

    // (1) Pre-capture EVERY CUDA graph now (capture-only, no KV state touched) so
    // request #1's generation pays zero graph-capture cost. Lazy capture otherwise
    // lands on the first generated tokens — the cold turns the agentic bench showed
    // (turn-0/1 decode well below steady state). Covers the single-token decode graph,
    // the MTP draft graph (only if the assistant is loaded — mtp_forward reads the MTP
    // buffers), and the batched spec-verify graph for every K.
    decode_graph_ensure(eng);
    if (eng->mtp.loaded) mtp_graph_ensure(eng);
    for (int k = 1; k <= GEMMA4_SPEC_MAX; k++) batched_graph_ensure(eng, k);

    // (2) Real passes that fault in scratch + load every lazily-bound CUDA module and
    // let cuBLAS init its workspaces, across ALL the shapes request #1 will hit — not
    // just the one fresh batched prefill. Each is rewound so warmup leaks no KV state.
    const int WN = 2048;   // ≤ 4096 → exercises the batched tensor-core prefill path
    int32_t *toks = (int32_t*)malloc((size_t)WN * sizeof(int32_t));
    float   *logits = (float*)malloc((size_t)GEMMA4_VOCAB_SIZE * sizeof(float));
    if (toks && logits) {
        for (int i = 0; i < WN; i++) toks[i] = i % 64;  // arbitrary valid token ids

        // (a) batched fresh prefill, then a few decodes — warms the decode mmvq kernels
        // (distinct modules from the cuBLAS prefill GEMMs) and replays the decode graph.
        if (gemma4_engine_prefill_batched(eng, toks, WN, logits) == 0) {
            for (int i = 0; i < 4; i++) gemma4_engine_decode(eng, toks[i % 64], logits);
            cudaStreamSynchronize(eng->stream);
        }
        gemma4_engine_reset(eng);   // rewind cursors — warmup must not leak KV state

        // (b) flash continuation path (a suffix prefill on a non-empty cache) — the
        // turn-1+ path the fresh batched warmup above never exercises. Warm it at a
        // REPRESENTATIVE width (~2k): cuBLAS re-selects GEMM kernels per N, so a tiny
        // 256-tok flash pass leaves the first real ~2.5k suffix prefill (turn 1) cold.
        if (gemma4_engine_prefill_batched(eng, toks, 256, logits) == 0) {
            gemma4_engine_prefill_flash(eng, toks, WN, logits);   // WN=2048-wide flash
            cudaStreamSynchronize(eng->stream);
        }
        gemma4_engine_reset(eng);

        // (c) small-N fresh prefill — warms the short-prompt GEMM shapes of turn 0
        // (cuBLAS re-selects kernels per N; the WN=2048 pass doesn't cover N≈128).
        if (gemma4_engine_prefill_batched(eng, toks, 128, logits) == 0)
            cudaStreamSynchronize(eng->stream);
        gemma4_engine_reset(eng);
    }
    free(toks);
    free(logits);
    return 0;
}

// Cooperative prefill abort: called from a DIFFERENT thread than the one blocked
// inside a prefill (which holds the Go engine mutex). The chunked prefill loops
// poll the flag between chunks and return -3; the flag is cleared at the next
// prefill's entry (chain head). Safe to call at any time — at worst a no-op.
void gemma4_engine_abort_prefill(gemma4_engine_t *eng) {
    if (eng) eng->abort_req = 1;
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// Qwen3.5 is kept in the same CUDA translation unit for access to the engine-internal
// projection/MoE primitives and graph-safe static helpers, but its implementation is split by
// responsibility so the generic Gemma/Qwen3 runtime is no longer a 20K-line monolith.
#include "qwen35_kernels.cuh"
#include "qwen35_jspace.cuh"
#include "qwen35_runtime.cuh"
#include "qwen35_backend.cuh"

// ─── P0 (S1a) GDN rollback ABI ───────────────────────────────────────────────────────
// Thin extern "C" wrappers around the q35_gdn_* statics, so the DFlash verify path (and the P0
// lossless-rollback gate) can snapshot -> speculatively advance -> commit(accepted_len) a slot's
// GDN recurrent state. Byte-identical continuation is proven by test_qwen35_gdn_rollback.
extern "C" int gemma4_engine_q35_gdn_snapshot(gemma4_engine_t *eng, int slot) {
    return q35_gdn_snapshot(eng, slot);
}
extern "C" int gemma4_engine_q35_gdn_commit(gemma4_engine_t *eng, int slot,
                                            const int32_t *accepted, int j, int32_t *out_next) {
    return q35_gdn_commit(eng, slot, accepted, j, out_next);
}
extern "C" int gemma4_engine_q35_gdn_rewind(gemma4_engine_t *eng, int slot) {
    return q35_gdn_commit(eng, slot, nullptr, 0, nullptr);
}
