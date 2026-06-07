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
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>         // __nv_fp8_storage_t, conversion functions
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

// POSIX headers for host-side file loading (mmap, open, etc.)
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace cg = cooperative_groups;

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
    GGML_TYPE_Q8_0 = 8,
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

// ─── Unified weight element decode ─────────────────────────────────────
//
// Reads ONE logical weight element from a quantised weight tensor and returns
// it as float.  Supports both on-disk formats:
//
//   FORMAT_FP8  (0): weight is a flat array of 1-byte FP8 E4M3 values.
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
    if (fmt == 0 /*FORMAT_FP8*/) {
        return fp8_to_float((__nv_fp8_storage_t)base[i]);
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
    if (i < n) dst[i] = float_to_fp8(src[i]);
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

// Single-output GEMV: out[j] = sum_i W[j,i] * x[i]
__global__ void gemv_kernel(
    float          *out,
    const uint8_t  *weight,   // row-major [out_dim × in_dim]
    const float    *x,
    int             in_dim,
    int             out_dim,
    int             fmt)      // 0=FP8, 1=Q8_0
{
    extern __shared__ float smem[];
    int idx = blockIdx.x;
    if (idx >= out_dim) return;
    const uint8_t *row = weight + (size_t)idx * (fmt == 0 ? in_dim : (in_dim/32)*34);
    float sum = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x)
        sum += decode_weight(row, i, fmt) * x[i];
    // Use cooperative_groups reduction instead of custom block_reduce
    namespace cg = cooperative_groups;
    cg::thread_block tb = cg::this_thread_block();
    sum = cg::reduce(tb, sum, cg::plus<float>());
    if (threadIdx.x == 0) out[idx] = sum;
}

// Fused gate+up GEMV (for FFN): computes both projections in one pass.
__global__ void gemv_pair_kernel(
    float          *out_gate,
    float          *out_up,
    const uint8_t  *weight_gate,
    const uint8_t  *weight_up,
    const float    *x,
    int             in_dim,
    int             out_dim,
    int             fmt)
{
    extern __shared__ float smem[];
    namespace cg = cooperative_groups;
    cg::thread_block tb = cg::this_thread_block();
    int idx = blockIdx.x;
    if (idx >= out_dim) return;
    size_t row_bytes = fmt == 0 ? in_dim : (size_t)(in_dim/32)*34;
    const uint8_t *rg = weight_gate + (size_t)idx * row_bytes;
    const uint8_t *ru = weight_up   + (size_t)idx * row_bytes;
    float sg = 0.0f, su = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float xi = x[i];
        sg += decode_weight(rg, i, fmt) * xi;
        su += decode_weight(ru, i, fmt) * xi;
    }
    sg = cg::reduce(tb, sg, cg::plus<float>());
    su = cg::reduce(tb, su, cg::plus<float>());
    if (threadIdx.x == 0) { out_gate[idx] = sg; out_up[idx] = su; }
}

// =========================================================================
// ─── Unified LoRA GEMV: out[j] = Wx[j] + scale*(B*(A*x))[j] ────────────────
// Works for both FP8 and Q8_0 via decode_weight(). All reductions use
// block_reduce_sum (correct for blockDim.x = 256).
// A: [rank × in_dim]  B: [out_dim × rank]  (both FP32, row-major)
__global__ void gemv_lora_kernel(
    float          *out,
    const uint8_t  *weight,
    const float    *lora_a,
    const float    *lora_b,
    const float    *x,
    int             in_dim,
    int             out_dim,
    int             rank,
    float           lora_scale,
    int             fmt)
{
    extern __shared__ float smem[];
    int idx = blockIdx.x;
    if (idx >= out_dim) return;

    size_t row_bytes = fmt == 0 ? (size_t)in_dim : (size_t)(in_dim/32)*34;
    const uint8_t *row = weight + (size_t)idx * row_bytes;

    float base = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x)
        base += decode_weight(row, i, fmt) * x[i];
    base = block_reduce_sum(base, smem);

    float lora = 0.0f;
    if (lora_scale != 0.0f && lora_a && lora_b && rank > 0) {
        for (int k = 0; k < rank; k++) {
            float ak = 0.0f;
            const float *ar = lora_a + (size_t)k * in_dim;
            for (int i = threadIdx.x; i < in_dim; i += blockDim.x) ak += ar[i] * x[i];
            ak = block_reduce_sum(ak, smem);
            lora += lora_b[(size_t)idx * rank + k] * ak;
        }
        lora *= lora_scale;
    }

    if (threadIdx.x == 0) out[idx] = base + lora;
}

// Fused gate+up LoRA GEMV for FFN.
__global__ void gemv_lora_pair_kernel(
    float          *out_gate,
    float          *out_up,
    const uint8_t  *weight_gate,
    const uint8_t  *weight_up,
    const float    *lora_a_gate,
    const float    *lora_b_gate,
    const float    *lora_a_up,
    const float    *lora_b_up,
    const float    *x,
    int             in_dim,
    int             out_dim,
    int             rank_gate,
    int             rank_up,
    float           lora_scale,
    int             fmt)
{
    extern __shared__ float smem[];
    int idx = blockIdx.x;
    if (idx >= out_dim) return;

    size_t row_bytes = fmt == 0 ? (size_t)in_dim : (size_t)(in_dim/32)*34;
    const uint8_t *rg = weight_gate + (size_t)idx * row_bytes;
    const uint8_t *ru = weight_up   + (size_t)idx * row_bytes;

    float sg = 0.0f, su = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float xi = x[i];
        sg += decode_weight(rg, i, fmt) * xi;
        su += decode_weight(ru, i, fmt) * xi;
    }
    sg = cg::reduce(tb, sg, cg::plus<float>());
    su = cg::reduce(tb, su, cg::plus<float>());

    float lg = 0.0f, lu = 0.0f;
    if (lora_scale != 0.0f) {
        for (int k = 0; k < rank_gate && lora_a_gate && lora_b_gate; k++) {
            float ak = 0.0f;
            const float *ar = lora_a_gate + (size_t)k * in_dim;
            for (int i = threadIdx.x; i < in_dim; i += blockDim.x) ak += ar[i] * x[i];
            ak = block_reduce_sum(ak, smem);
            lg += lora_b_gate[(size_t)idx * rank_gate + k] * ak;
        }
        for (int k = 0; k < rank_up && lora_a_up && lora_b_up; k++) {
            float ak = 0.0f;
            const float *ar = lora_a_up + (size_t)k * in_dim;
            for (int i = threadIdx.x; i < in_dim; i += blockDim.x) ak += ar[i] * x[i];
            ak = block_reduce_sum(ak, smem);
            lu += lora_b_up[(size_t)idx * rank_up + k] * ak;
        }
        lg *= lora_scale; lu *= lora_scale;
    }

    if (threadIdx.x == 0) { out_gate[idx] = sg + lg; out_up[idx] = su + lu; }
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
    int    n_heads,
    int    n_kv_heads,
    int    head_dim,
    float  theta_base)
{
    int d   = threadIdx.x;          // 0 .. head_dim/2 - 1
    int idx = blockIdx.x;           // head index
    int half = head_dim / 2;
    if (d >= half) return;

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

    // p-RoPE position scaling
    float eff_pos = (float)pos * ((float)context_len / (float)GEMMA4_MAX_CTX);

    float ff    = freq_factors ? freq_factors[d] : 1.0f;
    float theta = powf(theta_base, -2.0f * d / head_dim) / ff;
    float cos_val = cosf(eff_pos * theta);
    float sin_val = sinf(eff_pos * theta);

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

// Sliding-window GQA decode, single query token.
// blockIdx.x = q_head (0..n_heads-1), blockDim.x = head_dim.
// KV layout: [n_kv_heads][window_size][head_dim] ring buffer.
// Shared memory: 32 floats for block_reduce + window_size floats for scores.
// Launch: sliding_attn_decode_kernel<<<n_heads, head_dim,
//   (32 + window_size)*sizeof(float)>>>
__global__ void sliding_attn_decode_kernel(
    float       *output,       // [n_heads × head_dim]
    const float *q,            // [n_heads × head_dim]
    const float *k_cache,      // [n_kv_heads][window_size][head_dim]
    const float *v_cache,
    int          n_heads,
    int          n_kv_heads,
    int          head_dim,
    int          window_size,
    int          cursor,
    int          filled)
{
    extern __shared__ float smem[];      // [32] reduce + [window_size] scores
    float *scores = smem + 32;

    int q_head  = blockIdx.x;
    int kv_head = q_head / (n_heads / n_kv_heads); // GQA mapping
    int tid     = threadIdx.x;

    int window_len = min(filled, window_size);

    // ── Compute QK dot products ──────────────────────────────────────
    float q_d = (tid < head_dim) ? q[q_head * head_dim + tid] : 0.0f;

    for (int t = 0; t < window_len; t++) {
        int ring_idx = (cursor - window_len + t + window_size) % window_size;
        float k_d = (tid < head_dim)
            ? k_cache[kv_head * window_size * head_dim + ring_idx * head_dim + tid]
            : 0.0f;
        float s = q_d * k_d;
        s = block_reduce_sum(s, smem);   // smem[0..31] used here, scores untouched
        if (tid == 0) scores[t] = s;
        __syncthreads();
    }

    // ── Online softmax (thread 0 computes on scores[]) ───────────────
    if (tid == 0) {
        float mx = scores[0];
        for (int t = 1; t < window_len; t++) mx = fmaxf(mx, scores[t]);
        float denom = 0.0f;
        for (int t = 0; t < window_len; t++) {
            scores[t] = expf(scores[t] - mx);
            denom += scores[t];
        }
        float inv = 1.0f / denom;
        for (int t = 0; t < window_len; t++) scores[t] *= inv;
    }
    __syncthreads();

    // ── Weighted sum of V ────────────────────────────────────────────
    float out = 0.0f;
    if (tid < head_dim) {
        for (int t = 0; t < window_len; t++) {
            int ring_idx = (cursor - window_len + t + window_size) % window_size;
            float v_d = v_cache[kv_head * window_size * head_dim + ring_idx * head_dim + tid];
            out += scores[t] * v_d;
        }
        output[q_head * head_dim + tid] = out;
    }
}

// =========================================================================
// ─── Global Attention Decode (single token) ─────────────────────────────
// =========================================================================

// Global attention decode, single query token.
// blockIdx.x = q_head; blockDim.x = head_dim (512).
// KV cache layout: [ctx_len][head_dim] floats (K=V unified, only K stored).
// Shared memory: 32 floats for reduce + ctx_len floats for scores.
// Launch: global_attn_decode_kernel<<<n_heads, head_dim,
//   (32 + ctx_len)*sizeof(float)>>>
__global__ void global_attn_decode_kernel(
    float       *output,       // [n_heads × head_dim]
    const float *q,            // [n_heads × head_dim]
    const float *k_cache,      // [ctx_len][head_dim]
    const float *v_cache,      // [ctx_len][head_dim]
    int          n_heads,
    int          head_dim,
    int          ctx_len)
{
    extern __shared__ float smem[];    // [32] reduce  +  [ctx_len] scores
    float *scores = smem + 32;

    int q_head = blockIdx.x;
    int tid    = threadIdx.x;

    float q_d = (tid < head_dim) ? q[q_head * head_dim + tid] : 0.0f;

    // ── QK dot products ───────────────────────────────────────────────
    for (int t = 0; t < ctx_len; t++) {
        float k_d = (tid < head_dim) ? k_cache[t * head_dim + tid] : 0.0f;
        float s   = q_d * k_d;
        s = block_reduce_sum(s, smem);
        if (tid == 0) scores[t] = s;
        __syncthreads();
    }

    // ── Softmax (thread 0) ─────────────────────────────────────────
    if (tid == 0) {
        float mx = scores[0];
        for (int t = 1; t < ctx_len; t++) mx = fmaxf(mx, scores[t]);
        float denom = 0.0f;
        for (int t = 0; t < ctx_len; t++) {
            scores[t] = expf(scores[t] - mx);
            denom += scores[t];
        }
        float inv = 1.0f / denom;
        for (int t = 0; t < ctx_len; t++) scores[t] *= inv;
    }
    __syncthreads();

    // ── Weighted sum of V ─────────────────────────────────────────
    float out = 0.0f;
    if (tid < head_dim) {
        for (int t = 0; t < ctx_len; t++)
            out += scores[t] * v_cache[t * head_dim + tid];
        output[q_head * head_dim + tid] = out;
    }
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

// =========================================================================
// ─── Embedding Lookup ───────────────────────────────────────────────────
// =========================================================================

__global__ void embed_lookup_kernel(
    float             *out,     // [batch × hidden_size]
    const unsigned char *table, // [vocab_size × hidden_size] in FP8
    const int32_t     *tokens,  // [batch]
    int                batch,
    int                hidden_size)
{
    int row = blockIdx.x;
    if (row >= batch) return;

    int token = tokens[row];
    if (token < 0) token = 0;
    if (token >= GEMMA4_VOCAB_SIZE) token = 0;

    const unsigned char *emb = table + token * hidden_size;
    float *out_row = out + row * hidden_size;

    int i = threadIdx.x;
    for (; i < hidden_size; i += blockDim.x) {
        out_row[i] = fp8_to_float(emb[i]);
    }
}

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

struct gemma4_engine {
    // GGUF data (mmap'd)
    const uint8_t *gguf_data;
    uint64_t       gguf_size;
    int            gguf_fd;

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
        } layers[GEMMA4_MAX_LAYERS];

        uint64_t output_norm;        // [hidden_size] (FP32)
        uint64_t output_weight;      // [hidden_size × vocab_size]
    } tensors;

    // 1 if output_weight aliases token_embd (tied embeddings), 0 if a separate
    // output.weight tensor was found in the GGUF.
    int             output_tied;

    // Model parameters
    tensor_format_t format;
    uint32_t        context_size;
    layer_type_t    layer_types[GEMMA4_MAX_LAYERS]; // 0=sliding, 1=global
    int             n_layers_sliding;
    int             n_layers_global;

    // Layer index helpers: which layers are global
    int             global_layer_indices[GEMMA4_MAX_LAYERS];
    int             n_global;

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

    // KV cache (device)
    // Sliding: 40 layers × 8 heads × 1024 × 256 (FP16 for precision)
    float  *d_sliding_k;       // [40 × 8 × 1024 × 256]
    float  *d_sliding_v;       // [40 × 8 × 1024 × 256]
    int     sliding_cursor[GEMMA4_MAX_LAYERS];
    int     sliding_filled[GEMMA4_MAX_LAYERS];

    // Global: 8 layers × 1 head × ctx_size × 512 (float)
    // K and V are stored separately because K gets RMSNorm+RoPE while
    // V gets only plain RMSNorm (no weight, no RoPE).
    float  *d_global_k;  // [GEMMA4_MAX_LAYERS × ctx_size × 512]
    float  *d_global_v;  // [GEMMA4_MAX_LAYERS × ctx_size × 512]
    int     global_n_tokens;
    int     global_kv_capacity;

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

    // Timing accumulators
    float prefill_time_ms;
    float decode_time_ms;
    int   n_prefill_tokens;
    int   n_decode_tokens;

    // Loaded flag
    int loaded;

    // ─── LoRA adapter support ───────────────────────────────────────────
    int               lora_loaded;       // 1 if LoRA adapters are loaded
    layer_lora_set_t  lora[GEMMA4_MAX_LAYERS]; // per-layer LoRA adapters
    lora_adapter_t    lora_output;       // output projection LoRA
    char              lora_path[1024];   // path to loaded LoRA GGUF
    float             lora_scale;        // global scale multiplier

    // LoRA scratch: intermediate buffer for rank-dim vectors
    float *d_lora_scratch;  // [max_rank × hidden_size] for A(x) intermediates
    float *d_lora_out;      // [hidden_size] LoRA contribution added to base output
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
    if (hdr->magic != 0x46554747) return -1;

    const uint8_t *end = data + size;
    const uint8_t *p = gguf_skip_metadata(data, size);
    if (!p) return -1;

    uint64_t tdata_start = gguf_tensor_data_start(data, size);
    if (tdata_start == 0) return -1;

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

// ─── Engine Construction ──────────────────────────────────────────────

gemma4_engine_t* gemma4_engine_create(
    const char    *model_path,
    tensor_format_t format,
    uint32_t       context_size,
    int            device_id)
{
    // Allocate engine
    gemma4_engine_t *eng = (gemma4_engine_t *)
        calloc(1, sizeof(gemma4_engine_t));
    if (!eng) return NULL;

    eng->format = format;
    eng->context_size = context_size;
    eng->device_id = device_id;
    eng->loaded = 0;

    // Validate context
    if (context_size > GEMMA4_MAX_CTX) {
        fprintf(stderr, "gem4d: context size %u exceeds max %u\n",
                context_size, GEMMA4_MAX_CTX);
        free(eng);
        return NULL;
    }

    // Default layer types: 5 sliding + 1 global, repeated 8 times
    for (int i = 0; i < 48; i++) {
        int pos_in_block = i % 6;
        if (pos_in_block == 5) {
            eng->layer_types[i] = LAYER_GLOBAL;
            eng->global_layer_indices[eng->n_global++] = i;
        } else {
            eng->layer_types[i] = LAYER_SLIDING;
            eng->n_layers_sliding++;
        }
    }
    eng->n_layers_global = eng->n_global;

    // CUDA setup
    cudaSetDevice(device_id);
    cudaStreamCreate(&eng->stream);

    // Get memory info
    cudaMemGetInfo(&eng->free_mem, &eng->total_mem);

    // Create cuBLAS handle and bind to our stream
    cublasCreate(&eng->cublas);
    cublasSetStream(eng->cublas, eng->stream);
    {
        int math_mode = CUBLAS_DEFAULT_MATH;
#if defined(CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)
        // FP8 path uses default math
#endif
        cublasSetMathMode(eng->cublas, (cublasMath_t)math_mode);
    }

    // Open and mmap the GGUF file
    eng->gguf_fd = open(model_path, O_RDONLY);
    if (eng->gguf_fd < 0) {
        perror("gem4d: open model");
        gemma4_engine_destroy(eng);
        return NULL;
    }

    struct stat st;
    fstat(eng->gguf_fd, &st);
    eng->gguf_size = st.st_size;

    eng->gguf_data = (const uint8_t *)mmap(
        NULL, eng->gguf_size, PROT_READ, MAP_PRIVATE, eng->gguf_fd, 0);
    if (eng->gguf_data == MAP_FAILED) {
        perror("gem4d: mmap model");
        gemma4_engine_destroy(eng);
        return NULL;
    }

    fprintf(stderr, "gem4d: loaded %s (%.2f GB)\n",
            model_path, eng->gguf_size / (1024.0 * 1024.0 * 1024.0));

    // Parse the per-layer attention pattern from GGUF metadata, overriding the
    // defaults computed above. The real Gemma-4 GGUF exposes this as:
    //
    //   gemma4.attention.sliding_window_pattern : bool[block_count]
    //       true  => sliding-window attention layer
    //       false => global attention layer
    //
    // Older/alternate exports instead carry an i32 array of KV-head counts:
    //
    //   gemma4.attention.head_count_kv : i32[block_count]
    //       value == GEMMA4_GLOBAL_KV_HEADS (1) => global layer
    //       otherwise                            => sliding layer
    //
    // We try the bool pattern first, then fall back to the head-count array.
    bool pattern_parsed = false;
    gguf_array_t arr;
    if (gguf_parse_metadata(eng->gguf_data, eng->gguf_size,
            "gemma4.attention.sliding_window_pattern", &arr, GGUF_TYPE_ARRAY) == 0
        && arr.elem_type == GGUF_TYPE_BOOL)
    {
        eng->n_layers_sliding = 0;
        eng->n_global = 0;
        for (uint64_t i = 0; i < arr.count && i < GEMMA4_MAX_LAYERS; i++) {
            bool is_sliding = arr.data[i] != 0;
            if (is_sliding) {
                eng->layer_types[i] = LAYER_SLIDING;
                eng->n_layers_sliding++;
            } else {
                eng->layer_types[i] = LAYER_GLOBAL;
                eng->global_layer_indices[eng->n_global++] = (int)i;
            }
        }
        eng->n_layers_global = eng->n_global;
        pattern_parsed = true;
    }
    if (!pattern_parsed &&
        gguf_parse_metadata(eng->gguf_data, eng->gguf_size,
            "gemma4.attention.head_count_kv", &arr, GGUF_TYPE_ARRAY) == 0
        && (arr.elem_type == GGUF_TYPE_INT32 || arr.elem_type == GGUF_TYPE_UINT32))
    {
        eng->n_layers_sliding = 0;
        eng->n_global = 0;
        for (uint64_t i = 0; i < arr.count && i < GEMMA4_MAX_LAYERS; i++) {
            int32_t kv; memcpy(&kv, arr.data + i * 4, 4);
            if (kv == GEMMA4_GLOBAL_KV_HEADS) {
                eng->layer_types[i] = LAYER_GLOBAL;
                eng->global_layer_indices[eng->n_global++] = (int)i;
            } else {
                eng->layer_types[i] = LAYER_SLIDING;
                eng->n_layers_sliding++;
            }
        }
        eng->n_layers_global = eng->n_global;
        pattern_parsed = true;
    }
    if (!pattern_parsed) {
        fprintf(stderr, "gem4d: warning: no attention pattern in GGUF, "
                        "using default 5-sliding/1-global cadence\n");
    }

    // Parse tensor offsets via gguf_find_tensor_offset
    // Helper macro: find tensor, store offset
    #define LOAD_TENSOR_OFFSET(name, field) do { \
        uint64_t _off, _n; \
        if (gguf_find_tensor_offset(eng->gguf_data, eng->gguf_size, \
                name, &_off, &_n) == 0) { \
            eng->tensors.field = _off; \
        } else { \
            fprintf(stderr, "gem4d: tensor '%s' not found in GGUF\n", name); \
        } \
    } while(0)

    // Load embedding
    LOAD_TENSOR_OFFSET("token_embd.weight", token_embd);

    // Load per-layer tensors
    char tname[128];
    for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
        snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].attn_q);

        snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].attn_k);

        // Global layers use a unified K=V cache and ship no attn_v.weight.
        if (eng->layer_types[l] == LAYER_SLIDING) {
            snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", l);
            LOAD_TENSOR_OFFSET(tname, layers[l].attn_v);
        } else {
            eng->tensors.layers[l].attn_v = 0;
        }

        snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].attn_output);

        snprintf(tname, sizeof(tname), "blk.%d.attn_norm.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].attn_norm);

        snprintf(tname, sizeof(tname), "blk.%d.attn_q_norm.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].attn_q_norm);

        snprintf(tname, sizeof(tname), "blk.%d.attn_k_norm.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].attn_k_norm);

        snprintf(tname, sizeof(tname), "blk.%d.post_attention_norm.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].post_attn_norm);

        snprintf(tname, sizeof(tname), "blk.%d.ffn_gate.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].ffn_gate);

        snprintf(tname, sizeof(tname), "blk.%d.ffn_up.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].ffn_up);

        snprintf(tname, sizeof(tname), "blk.%d.ffn_down.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].ffn_down);

        snprintf(tname, sizeof(tname), "blk.%d.ffn_norm.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].ffn_norm);

        snprintf(tname, sizeof(tname), "blk.%d.post_ffw_norm.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].post_ffn_norm);

        snprintf(tname, sizeof(tname), "blk.%d.layer_output_scale.weight", l);
        LOAD_TENSOR_OFFSET(tname, layers[l].layer_out_scale);
    }

    LOAD_TENSOR_OFFSET("output_norm.weight", output_norm);

    // Output (LM head) projection. Gemma-4 ties the output projection to the
    // input embedding, so most GGUF exports do NOT contain a separate
    // "output.weight" tensor. Probe for an explicit head first; if absent,
    // alias it to token_embd.weight (tied embeddings).
    {
        uint64_t _off = 0, _n = 0;
        if (gguf_find_tensor_offset(eng->gguf_data, eng->gguf_size,
                "output.weight", &_off, &_n) == 0) {
            eng->tensors.output_weight = _off;
            eng->output_tied = 0;
        } else {
            eng->tensors.output_weight = eng->tensors.token_embd;
            eng->output_tied = 1;
            fprintf(stderr, "gem4d: output.weight not present — using tied "
                            "token_embd.weight as LM head\n");
        }
    }

    #undef LOAD_TENSOR_OFFSET

    // Allocate device memory
    // Scratch (1M floats = 4 MB)
    cudaMalloc(&eng->d_scratch,    4 * 1024 * 1024);
    cudaMalloc(&eng->d_logits,     GEMMA4_VOCAB_SIZE * sizeof(float));
    cudaMalloc(&eng->d_x,          GEMMA4_HIDDEN_SIZE * sizeof(float));
    // d_attn_q/k/v/out must be large enough for BOTH layer types.
    //   sliding: Q=16×256=4096  KV=8×256=2048
    //   global:  Q=16×512=8192  KV=1×512=512
    int max_q    = GEMMA4_HEADS * GEMMA4_GLOBAL_HEAD_DIM;          // 8192
    int max_kv   = GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM;              // 2048
    cudaMalloc(&eng->d_attn_q,     max_q  * sizeof(float));
    cudaMalloc(&eng->d_attn_k,     max_kv * sizeof(float));
    cudaMalloc(&eng->d_attn_v,     max_kv * sizeof(float));
    cudaMalloc(&eng->d_attn_out,   max_q  * sizeof(float));  // attention output, NOT hidden_size
    cudaMalloc(&eng->d_ffn_out,    GEMMA4_INTERMEDIATE * sizeof(float));
    cudaMalloc(&eng->d_ffn_gate,   GEMMA4_INTERMEDIATE * sizeof(float));
    cudaMalloc(&eng->d_ffn_up,     GEMMA4_INTERMEDIATE * sizeof(float));
    cudaMalloc(&eng->d_norm,       GEMMA4_HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&eng->d_norm_w,     GEMMA4_HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&eng->d_residual,   GEMMA4_HIDDEN_SIZE * sizeof(float));

    cudaMalloc(&eng->d_head_norm_w, GEMMA4_GLOBAL_HEAD_DIM * sizeof(float));

    // Load rope_freqs.weight into device buffer for global-layer RoPE.
    {
        uint64_t rf_off = 0, rf_n = 0;
        cudaMalloc(&eng->d_rope_freqs,
            GEMMA4_GLOBAL_HEAD_DIM / 2 * sizeof(float));
        if (gguf_find_tensor_offset(eng->gguf_data, eng->gguf_size,
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

    // KV cache allocation
    // Sliding: 40 × 8 × 1024 × 256 × sizeof(float) = 40 × 8 × 1024 × 256 × 4 = 320 MB
    size_t sliding_kv_size = (size_t)GEMMA4_MAX_LAYERS *
        GEMMA4_KV_HEADS * GEMMA4_SLIDING_WINDOW * GEMMA4_HEAD_DIM * sizeof(float);
    cudaMalloc(&eng->d_sliding_k, sliding_kv_size);
    cudaMalloc(&eng->d_sliding_v, sliding_kv_size);

    // Global K and V caches: float, separate K and V.
    // Layout: [GEMMA4_MAX_LAYERS][ctx_size][head_dim]
    eng->global_kv_capacity = context_size;
    size_t global_kv_size = (size_t)GEMMA4_MAX_LAYERS *
        context_size * GEMMA4_GLOBAL_HEAD_DIM * sizeof(float);
    cudaMalloc(&eng->d_global_k, global_kv_size);
    cudaMalloc(&eng->d_global_v, global_kv_size);

    // LoRA scratch buffers
    cudaMalloc(&eng->d_lora_scratch,
        (size_t)GEMMA4_MAX_LORA_RANK * GEMMA4_HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&eng->d_lora_out,
        GEMMA4_HIDDEN_SIZE * sizeof(float));
    eng->lora_loaded = 0;
    eng->lora_path[0] = '\0';
    eng->lora_scale = 1.0f;
    memset(eng->lora, 0, sizeof(eng->lora));
    memset(&eng->lora_output, 0, sizeof(eng->lora_output));

    eng->loaded = 1;
    fprintf(stderr, "gem4d: engine initialized (%.2f GB model, %u ctx, %s)\n",
            eng->gguf_size / (1024.0*1024.0*1024.0),
            context_size,
            format == FORMAT_FP8 ? "FP8" : "Q8_0");
    fprintf(stderr, "gem4d: %d sliding + %d global layers\n",
            eng->n_layers_sliding, eng->n_layers_global);
    fprintf(stderr, "gem4d: KV cache: sliding=%.1f MB, global=%.1f MB\n",
            sliding_kv_size / (1024.0*1024.0),
            global_kv_size / (1024.0*1024.0));

    return eng;
}

void gemma4_engine_destroy(gemma4_engine_t *eng) {
    if (!eng) return;

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
    CUDA_FREE(eng->d_ffn_out);
    CUDA_FREE(eng->d_ffn_gate);
    CUDA_FREE(eng->d_ffn_up);
    CUDA_FREE(eng->d_norm);
    CUDA_FREE(eng->d_norm_w);
    CUDA_FREE(eng->d_residual);
    CUDA_FREE(eng->d_rope_freqs);
    CUDA_FREE(eng->d_head_norm_w);
    CUDA_FREE(eng->d_sliding_k);
    CUDA_FREE(eng->d_sliding_v);
    CUDA_FREE(eng->d_global_k);
    CUDA_FREE(eng->d_global_v);
    CUDA_FREE(eng->d_lora_scratch);
    CUDA_FREE(eng->d_lora_out);
    #undef CUDA_FREE

    // LoRA cleanup
    if (eng->lora_loaded) {
        for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
            if (eng->lora[l].q.d_a) cudaFree(eng->lora[l].q.d_a);
            if (eng->lora[l].q.d_b) cudaFree(eng->lora[l].q.d_b);
            if (eng->lora[l].k.d_a) cudaFree(eng->lora[l].k.d_a);
            if (eng->lora[l].k.d_b) cudaFree(eng->lora[l].k.d_b);
            if (eng->lora[l].v.d_a) cudaFree(eng->lora[l].v.d_a);
            if (eng->lora[l].v.d_b) cudaFree(eng->lora[l].v.d_b);
            if (eng->lora[l].o.d_a) cudaFree(eng->lora[l].o.d_a);
            if (eng->lora[l].o.d_b) cudaFree(eng->lora[l].o.d_b);
            if (eng->lora[l].gate.d_a) cudaFree(eng->lora[l].gate.d_a);
            if (eng->lora[l].gate.d_b) cudaFree(eng->lora[l].gate.d_b);
            if (eng->lora[l].up.d_a) cudaFree(eng->lora[l].up.d_a);
            if (eng->lora[l].up.d_b) cudaFree(eng->lora[l].up.d_b);
            if (eng->lora[l].down.d_a) cudaFree(eng->lora[l].down.d_a);
            if (eng->lora[l].down.d_b) cudaFree(eng->lora[l].down.d_b);
        }
        if (eng->lora_output.d_a) cudaFree(eng->lora_output.d_a);
        if (eng->lora_output.d_b) cudaFree(eng->lora_output.d_b);
    }

    if (eng->cublas) cublasDestroy(eng->cublas);
    if (eng->stream) cudaStreamDestroy(eng->stream);

    free(eng);
}

// ─── Weight access helpers ────────────────────────────────────────────

// Tensor offsets stored in eng->tensors are ABSOLUTE file byte offsets
// (gguf_find_tensor already adds the tensor-data start). These helpers simply
// index into the mmap'd file.
static inline const unsigned char* weight_fp8(
    const gemma4_engine_t *eng, uint64_t tensor_offset)
{
    return (const unsigned char *)(eng->gguf_data + tensor_offset);
}

static inline const float* weight_f32(
    const gemma4_engine_t *eng, uint64_t tensor_offset)
{
    return (const float *)(eng->gguf_data + tensor_offset);
}

// ─── Format-dispatching GEMV / embed launchers ─────────────────────────
//
// Quantized weights in the GGUF may be stored either as FP8 E4M3 (one byte per
// Thin wrappers so call-sites stay readable.
#define FMT(eng)  ((int)(eng)->format)

static inline void gemv_w(
    const gemma4_engine_t *eng,
    float *out, const uint8_t *weight, const float *x,
    int in_dim, int out_dim, cudaStream_t stream)
{
    gemv_kernel<<<out_dim, 256, 32*sizeof(float), stream>>>(
        out, weight, x, in_dim, out_dim, FMT(eng));
}

static inline void embed_w(
    const gemma4_engine_t *eng,
    float *out, const uint8_t *table, const int32_t *tokens,
    int batch, int hidden_size, cudaStream_t stream)
{
    if (eng->format == FORMAT_Q8_0)
        embed_lookup_q8_0_kernel<<<batch, 256, 0, stream>>>(
            out, table, tokens, batch, hidden_size);
    else
        embed_lookup_kernel<<<batch, 256, 0, stream>>>(
            out, table, tokens, batch, hidden_size);
}

// =========================================================================
// ─── Single Layer Forward (Decode, B=1) ─────────────────────────────────
// =========================================================================

// Forward one layer for a single token decode
// Handles both sliding and global layers
// Forward declarations for kernels defined later in this file.
__global__ void gemv_lora_kernel(float*,const uint8_t*,const float*,const float*,const float*,int,int,int,float,int);
__global__ void gemv_lora_pair_kernel(float*,float*,const uint8_t*,const uint8_t*,const float*,const float*,const float*,const float*,const float*,int,int,int,int,float,int);

// Upload a per-head norm weight (host→device) and call per_head_rms_norm_kernel.
#define PER_HEAD_NORM(dst, host_weight, n_h, h_dim) do { \
    if (host_weight) { \
        cudaMemcpyAsync(eng->d_head_norm_w, (host_weight), \
            (h_dim) * sizeof(float), cudaMemcpyHostToDevice, stream); \
    } \
    per_head_rms_norm_kernel<<<(n_h), (h_dim), smem32, stream>>>( \
        (dst), (host_weight) ? eng->d_head_norm_w : NULL, (h_dim), GEMMA4_RMS_EPS); \
} while(0)

// ─── Convenience: upload an F32 norm weight to d_norm and return its pointer ─
// Upload norm weight into d_norm_w (separate from d_norm output buffer).
#define LOAD_NORM_W(field) do { \
    cudaMemcpyAsync(eng->d_norm_w, \
        weight_f32(eng, eng->tensors.layers[layer].field), \
        GEMMA4_HIDDEN_SIZE * sizeof(float), \
        cudaMemcpyHostToDevice, stream); \
} while(0)

static int decode_layer(
    gemma4_engine_t *eng,
    int              layer,
    int              pos,
    int              context_len,
    cudaStream_t     stream)
{
    layer_type_t ltype = eng->layer_types[layer];
    int n_heads, n_kv_heads, head_dim, out_dim_q, out_dim_kv;
    if (ltype == LAYER_SLIDING) {
        n_heads    = GEMMA4_HEADS;       head_dim = GEMMA4_HEAD_DIM;
        n_kv_heads = GEMMA4_KV_HEADS;
    } else {
        n_heads    = GEMMA4_HEADS;       head_dim = GEMMA4_GLOBAL_HEAD_DIM;
        n_kv_heads = GEMMA4_GLOBAL_KV_HEADS;
    }
    out_dim_q  = n_heads    * head_dim;
    out_dim_kv = n_kv_heads * head_dim;

    const int block    = 256;
    const int smem32   = 32 * sizeof(float);
    const int smemH    = (32 + GEMMA4_HIDDEN_SIZE)  * sizeof(float); // unused here

    // ─────────────────────────────────────────────────────────────────
    // Save the pre-layer residual once; it will be added back after attn.
    // ─────────────────────────────────────────────────────────────────
    cudaMemcpyAsync(eng->d_residual, eng->d_x,
                    GEMMA4_HIDDEN_SIZE * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // ── 1. Pre-attention RMSNorm ──────────────────────────────────────
    LOAD_NORM_W(attn_norm);
    rms_norm_kernel<<<1, block, smem32, stream>>>(
        eng->d_norm, eng->d_x, eng->d_norm_w,
        GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_EPS);

    // ── 2. Q projection (NO per-head norm, NO RoPE for test) ──────────
    if (eng->lora_loaded && eng->lora[layer].q.active) {
        lora_adapter_t *lq = &eng->lora[layer].q;
        gemv_lora_kernel<<<out_dim_q, block, smem32, stream>>>(
            eng->d_attn_q, weight_fp8(eng, eng->tensors.layers[layer].attn_q),
            lq->d_a, lq->d_b, eng->d_norm,
            GEMMA4_HIDDEN_SIZE, out_dim_q, lq->rank, lq->scale * eng->lora_scale, FMT(eng));
    } else {
        gemv_w(eng, eng->d_attn_q,
            weight_fp8(eng, eng->tensors.layers[layer].attn_q),
            eng->d_norm, GEMMA4_HIDDEN_SIZE, out_dim_q, stream);
    }
    PER_HEAD_NORM(eng->d_attn_q,
        weight_f32(eng, eng->tensors.layers[layer].attn_q_norm),
        n_heads, head_dim);

    // ── 3. K projection → per-head RMSNorm ───────────────────────────────
    if (eng->lora_loaded && eng->lora[layer].k.active) {
        lora_adapter_t *lk = &eng->lora[layer].k;
        gemv_lora_kernel<<<out_dim_kv, block, smem32, stream>>>(
            eng->d_attn_k, weight_fp8(eng, eng->tensors.layers[layer].attn_k),
            lk->d_a, lk->d_b, eng->d_norm,
            GEMMA4_HIDDEN_SIZE, out_dim_kv, lk->rank, lk->scale * eng->lora_scale, FMT(eng));
    } else {
        gemv_w(eng, eng->d_attn_k,
            weight_fp8(eng, eng->tensors.layers[layer].attn_k),
            eng->d_norm, GEMMA4_HIDDEN_SIZE, out_dim_kv, stream);
    }
    // ── 4. V projection → plain RMSNorm (no weight) ──────────────────────
    // For global layers V = K BEFORE any norm/RoPE is applied.
    // For sliding layers V comes from a separate projection.
    if (ltype == LAYER_SLIDING) {
        if (eng->lora_loaded && eng->lora[layer].v.active) {
            lora_adapter_t *lv = &eng->lora[layer].v;
            gemv_lora_kernel<<<out_dim_kv, block, smem32, stream>>>(
                eng->d_attn_v, weight_fp8(eng, eng->tensors.layers[layer].attn_v),
                lv->d_a, lv->d_b, eng->d_norm,
                GEMMA4_HIDDEN_SIZE, out_dim_kv, lv->rank, lv->scale * eng->lora_scale, FMT(eng));
        } else {
            gemv_w(eng, eng->d_attn_v,
                weight_fp8(eng, eng->tensors.layers[layer].attn_v),
                eng->d_norm, GEMMA4_HIDDEN_SIZE, out_dim_kv, stream);
        }
        PER_HEAD_NORM(eng->d_attn_v, NULL, n_kv_heads, head_dim);
    } else {
        // Global: V = raw K projection (before K-norm)
        cudaMemcpyAsync(eng->d_attn_v, eng->d_attn_k,
            out_dim_kv * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        // V gets plain RMSNorm (no learnable weight)
        PER_HEAD_NORM(eng->d_attn_v, NULL, n_kv_heads, head_dim);
    }

    // Now apply K-norm (after V copy for global layers)
    PER_HEAD_NORM(eng->d_attn_k,
        weight_f32(eng, eng->tensors.layers[layer].attn_k_norm),
        n_kv_heads, head_dim);

    // ── 5. RoPE ──────────────────────────────────────────────────────────
    if (ltype == LAYER_SLIDING) {
        rope_sliding_kernel<<<n_heads, head_dim/2, 0, stream>>>(
            eng->d_attn_q, eng->d_attn_k,
            pos, n_heads, n_kv_heads, head_dim, 10000.0f);
    } else {
        rope_global_kernel<<<n_heads, head_dim/2, 0, stream>>>(
            eng->d_attn_q, eng->d_attn_k,
            pos, context_len, n_heads, n_kv_heads, head_dim,
            1000000.0f, eng->d_rope_freqs);
    }

    // ── 6. Write K (and V) into KV cache ─────────────────────────────────
    int kv_size = n_kv_heads * head_dim;
    if (ltype == LAYER_SLIDING) {
        int cursor = eng->sliding_cursor[layer];
        size_t layer_stride = (size_t)GEMMA4_SLIDING_WINDOW * GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM;
        float *k_slot = eng->d_sliding_k + layer * layer_stride
                        + (size_t)cursor * GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM;
        float *v_slot = eng->d_sliding_v + layer * layer_stride
                        + (size_t)cursor * GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM;
        cudaMemcpyAsync(k_slot, eng->d_attn_k,
            kv_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(v_slot, eng->d_attn_v,
            kv_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        eng->sliding_cursor[layer] = (cursor + 1) % GEMMA4_SLIDING_WINDOW;
        if (eng->sliding_filled[layer] < GEMMA4_SLIDING_WINDOW)
            eng->sliding_filled[layer]++;
    } else {
        int n = eng->global_n_tokens;
        size_t layer_stride = (size_t)eng->global_kv_capacity * GEMMA4_GLOBAL_HEAD_DIM;
        float *k_slot = eng->d_global_k + layer * layer_stride
                        + (size_t)n * GEMMA4_GLOBAL_HEAD_DIM;
        float *v_slot = eng->d_global_v + layer * layer_stride
                        + (size_t)n * GEMMA4_GLOBAL_HEAD_DIM;
        cudaMemcpyAsync(k_slot, eng->d_attn_k,
            kv_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(v_slot, eng->d_attn_v,
            kv_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    // ── 7. Attention ─────────────────────────────────────────────────────
    if (ltype == LAYER_SLIDING) {
        int smem_sl = (32 + GEMMA4_SLIDING_WINDOW) * (int)sizeof(float);
        sliding_attn_decode_kernel<<<n_heads, head_dim, smem_sl, stream>>>(
            eng->d_attn_out, eng->d_attn_q,
            eng->d_sliding_k + (size_t)layer * GEMMA4_SLIDING_WINDOW
                               * GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM,
            eng->d_sliding_v + (size_t)layer * GEMMA4_SLIDING_WINDOW
                               * GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM,
            n_heads, n_kv_heads, head_dim,
            GEMMA4_SLIDING_WINDOW,
            eng->sliding_cursor[layer],
            eng->sliding_filled[layer]);
    } else {
        // n_ctx = tokens already in cache + this one (written above)
        int n_ctx = eng->global_n_tokens + 1;
        int smem_gl = (32 + n_ctx) * (int)sizeof(float);
        size_t layer_stride = (size_t)eng->global_kv_capacity * GEMMA4_GLOBAL_HEAD_DIM;
        global_attn_decode_kernel<<<n_heads, head_dim, smem_gl, stream>>>(
            eng->d_attn_out, eng->d_attn_q,
            eng->d_global_k + layer * layer_stride,
            eng->d_global_v + layer * layer_stride,
            n_heads, head_dim, n_ctx);
        // Do NOT increment global_n_tokens here; the engine-level functions
        // (prefill/decode) own the counter and advance it once per token.
    }

    // ── 8. Output projection → post-attn norm → residual add ─────────────
    if (eng->lora_loaded && eng->lora[layer].o.active) {
        lora_adapter_t *lo = &eng->lora[layer].o;
        gemv_lora_kernel<<<GEMMA4_HIDDEN_SIZE, block, smem32, stream>>>(
            eng->d_x, weight_fp8(eng, eng->tensors.layers[layer].attn_output),
            lo->d_a, lo->d_b, eng->d_attn_out,
            out_dim_q, GEMMA4_HIDDEN_SIZE, lo->rank, lo->scale * eng->lora_scale, FMT(eng));
    } else {
        gemv_w(eng, eng->d_x,
            weight_fp8(eng, eng->tensors.layers[layer].attn_output),
            eng->d_attn_out, out_dim_q, GEMMA4_HIDDEN_SIZE, stream);
    }
    // Post-attention sandwich norm
    LOAD_NORM_W(post_attn_norm);
    rms_norm_kernel<<<1, block, smem32, stream>>>(
        eng->d_norm, eng->d_x, eng->d_norm_w, GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_EPS);
    // Residual: normed_attn_proj + pre-layer input
    residual_add_kernel<<<(GEMMA4_HIDDEN_SIZE+255)/256, 256, 0, stream>>>(
        eng->d_norm, eng->d_residual, GEMMA4_HIDDEN_SIZE);
    // d_norm = attn_out = new residual for FFN
    cudaMemcpyAsync(eng->d_x,        eng->d_norm,
        GEMMA4_HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(eng->d_residual,  eng->d_norm,
        GEMMA4_HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // ── 9. Pre-FFN RMSNorm ────────────────────────────────────────────────
    LOAD_NORM_W(ffn_norm);
    rms_norm_kernel<<<1, block, smem32, stream>>>(
        eng->d_norm, eng->d_x, eng->d_norm_w, GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_EPS);

    // ── 10. Gate + Up projections ─────────────────────────────────────────
    if (eng->lora_loaded && (eng->lora[layer].gate.active || eng->lora[layer].up.active)) {
        lora_adapter_t *lg = &eng->lora[layer].gate;
        lora_adapter_t *lu = &eng->lora[layer].up;
        gemv_lora_pair_kernel<<<GEMMA4_INTERMEDIATE, block, smem32, stream>>>(
            eng->d_ffn_gate, eng->d_ffn_up,
            weight_fp8(eng, eng->tensors.layers[layer].ffn_gate),
            weight_fp8(eng, eng->tensors.layers[layer].ffn_up),
            lg->active ? lg->d_a : NULL, lg->active ? lg->d_b : NULL,
            lu->active ? lu->d_a : NULL, lu->active ? lu->d_b : NULL,
            eng->d_norm, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE,
            lg->active ? lg->rank : 0, lu->active ? lu->rank : 0,
            eng->lora_scale, FMT(eng));
    } else if (eng->format == FORMAT_Q8_0) {
        gemv_kernel<<<GEMMA4_INTERMEDIATE, block, smem32, stream>>>(
            eng->d_ffn_gate, weight_fp8(eng, eng->tensors.layers[layer].ffn_gate), eng->d_norm, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE, FMT(eng));
        gemv_kernel<<<GEMMA4_INTERMEDIATE, block, smem32, stream>>>(
            eng->d_ffn_up, weight_fp8(eng, eng->tensors.layers[layer].ffn_up), eng->d_norm, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE, FMT(eng));
    } else {
        gemv_pair_kernel<<<GEMMA4_INTERMEDIATE, block, smem32, stream>>>(
            eng->d_ffn_gate, eng->d_ffn_up,
            weight_fp8(eng, eng->tensors.layers[layer].ffn_gate),
            weight_fp8(eng, eng->tensors.layers[layer].ffn_up),
            eng->d_norm, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE, FMT(eng));
    }

    // ── 11. GeGLU activation ─────────────────────────────────────────────
    geglu_kernel<<<(GEMMA4_INTERMEDIATE+255)/256, 256, 0, stream>>>(
        eng->d_ffn_out, eng->d_ffn_gate, eng->d_ffn_up, GEMMA4_INTERMEDIATE);

    // ── 12. FFN down projection ───────────────────────────────────────────
    if (eng->lora_loaded && eng->lora[layer].down.active) {
        lora_adapter_t *ld = &eng->lora[layer].down;
        gemv_lora_kernel<<<GEMMA4_HIDDEN_SIZE, block, smem32, stream>>>(
            eng->d_x, weight_fp8(eng, eng->tensors.layers[layer].ffn_down),
            ld->d_a, ld->d_b, eng->d_ffn_out,
            GEMMA4_INTERMEDIATE, GEMMA4_HIDDEN_SIZE,
            ld->rank, ld->scale * eng->lora_scale, FMT(eng));
    } else {
        gemv_w(eng, eng->d_x,
            weight_fp8(eng, eng->tensors.layers[layer].ffn_down),
            eng->d_ffn_out, GEMMA4_INTERMEDIATE, GEMMA4_HIDDEN_SIZE, stream);
    }

    // ── 13. Post-FFN sandwich norm → residual add ─────────────────────────
    LOAD_NORM_W(post_ffn_norm);
    rms_norm_kernel<<<1, block, smem32, stream>>>(
        eng->d_norm, eng->d_x, eng->d_norm_w, GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_EPS);
    residual_add_kernel<<<(GEMMA4_HIDDEN_SIZE+255)/256, 256, 0, stream>>>(
        eng->d_norm, eng->d_residual, GEMMA4_HIDDEN_SIZE);
    cudaMemcpyAsync(eng->d_x, eng->d_norm,
        GEMMA4_HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // ── 14. layer_output_scale ────────────────────────────────────────────
    if (eng->tensors.layers[layer].layer_out_scale != 0) {
        float s;
        cudaMemcpy(&s,
            weight_f32(eng, eng->tensors.layers[layer].layer_out_scale),
            sizeof(float), cudaMemcpyHostToHost);
        scale_kernel<<<(GEMMA4_HIDDEN_SIZE+255)/256, 256, 0, stream>>>(
            eng->d_x, GEMMA4_HIDDEN_SIZE, s); }

    return 0;
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
    if (!eng->loaded) return -1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, eng->stream);

    cudaStream_t stream = eng->stream;

    // Process each token in sequence (autoregressive prefill)
    // For batch=1, this is the same as decode but we update KV cache
    // In production, use batched prefill with flash attention

    for (int t = 0; t < n_tokens; t++) {
        int pos = eng->global_n_tokens;

        // Embedding lookup (format-aware: FP8 or Q8_0 table)
        embed_w(eng,
            eng->d_x,
            weight_fp8(eng, eng->tensors.token_embd),
            tokens + t, 1, GEMMA4_HIDDEN_SIZE, stream);

        // Gemma scales embeddings by √hidden_size
        { float sc = sqrtf((float)GEMMA4_HIDDEN_SIZE);
          scale_kernel<<<(GEMMA4_HIDDEN_SIZE+255)/256, 256, 0, stream>>>(
            eng->d_x, GEMMA4_HIDDEN_SIZE, sc); }

        // Run all layers
        for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
            decode_layer(eng, l, pos, eng->global_n_tokens + 1, stream);
        }
        eng->global_n_tokens++;

        // Output norm
        const float *out_norm_w = weight_f32(eng, eng->tensors.output_norm);
        cudaMemcpyAsync(eng->d_norm_w, out_norm_w, GEMMA4_HIDDEN_SIZE * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        rms_norm_kernel<<<1, 256, 32*sizeof(float), stream>>>(
            eng->d_norm, eng->d_x, eng->d_norm_w,
            GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_EPS);

        // Output projection: [3840 → 262144]. The LM head is the (tied) token
        // embedding, stored in the same format as the rest of the weights, so
        // dispatch on eng->format. gemv launches one block per output logit.
        int vocab = GEMMA4_VOCAB_SIZE;
        gemv_w(eng, eng->d_logits,
            weight_fp8(eng, eng->tensors.output_weight),
            eng->d_norm,
            GEMMA4_HIDDEN_SIZE, vocab, stream);

        // Logit softcap
        logit_softcap_kernel<<<(vocab + 255) / 256, 256, 0, stream>>>(
            eng->d_logits, GEMMA4_SOFTCAP, vocab);

        // Copy logits back if needed (only for last token usually)
        if (logits_out && t == n_tokens - 1) {
            cudaMemcpyAsync(logits_out, eng->d_logits,
                            vocab * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
        }
    }

    cudaEventRecord(stop, eng->stream);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    eng->prefill_time_ms += ms;
    eng->n_prefill_tokens += n_tokens;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// =========================================================================
// ─── Engine Decode ──────────────────────────────────────────────────────
// =========================================================================

int gemma4_engine_decode(
    gemma4_engine_t *eng,
    int32_t          token,
    float           *logits_out)
{
    if (!eng->loaded) return -1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, eng->stream);

    cudaStream_t stream = eng->stream;
    int pos = eng->global_n_tokens;

    // Embedding (format-aware: FP8 or Q8_0 table)
    embed_w(eng,
        eng->d_x,
        weight_fp8(eng, eng->tensors.token_embd),
        &token, 1, GEMMA4_HIDDEN_SIZE, stream);

    // Gemma scales embeddings by √hidden_size
    { float sc = sqrtf((float)GEMMA4_HIDDEN_SIZE);
      scale_kernel<<<(GEMMA4_HIDDEN_SIZE+255)/256, 256, 0, stream>>>(
            eng->d_x, GEMMA4_HIDDEN_SIZE, sc); }

    // Run all 48 layers
    for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
        decode_layer(eng, l, pos, eng->global_n_tokens + 1, stream);
    }

    // Output norm + projection + softcap
    const float *out_norm_w = weight_f32(eng, eng->tensors.output_norm);
    cudaMemcpyAsync(eng->d_norm_w, out_norm_w, GEMMA4_HIDDEN_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    rms_norm_kernel<<<1, 256, 32*sizeof(float), stream>>>(
        eng->d_norm, eng->d_x, eng->d_norm_w,
        GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_EPS);

    int vocab = GEMMA4_VOCAB_SIZE;
    if (eng->lora_loaded && eng->lora_output.active) {
        lora_adapter_t *lo = &eng->lora_output;
        gemv_lora_kernel<<<vocab, 256, 32*sizeof(float), stream>>>(
            eng->d_logits,
            weight_fp8(eng, eng->tensors.output_weight),
            lo->d_a, lo->d_b,
            eng->d_norm,
            GEMMA4_HIDDEN_SIZE, vocab,
            lo->rank, lo->scale * eng->lora_scale, FMT(eng));
    } else {
        gemv_w(eng, eng->d_logits,
            weight_fp8(eng, eng->tensors.output_weight),
            eng->d_norm,
            GEMMA4_HIDDEN_SIZE, vocab, stream);
    }

    logit_softcap_kernel<<<(vocab + 255) / 256, 256, 0, stream>>>(
        eng->d_logits, GEMMA4_SOFTCAP, vocab);

    // Copy logits to host
    if (logits_out) {
        cudaMemcpyAsync(logits_out, eng->d_logits,
                        vocab * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    }

    cudaEventRecord(stop, eng->stream);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    eng->decode_time_ms += ms;
    eng->n_decode_tokens++;

    eng->global_n_tokens++;

    return 0;
}

// =========================================================================
// ─── Verification Batch ─────────────────────────────────────────────────
// =========================================================================

int gemma4_engine_verify_batch(
    gemma4_engine_t *eng,
    const int32_t   *draft_tokens,
    int              K,
    const int32_t   *positions,
    float           *logits_out)
{
    if (!eng->loaded || K <= 0) return -1;

    // For now, verify draft tokens one by one
    // In production, this should use batched attention (CUDA Graph)
    // with weights shared across the batch

    int n_accepted = 0;

    for (int k = 0; k < K; k++) {
        float *host_logits = (float *)malloc(GEMMA4_VOCAB_SIZE * sizeof(float));

        // Save current context position before this verification
        int saved_n_tokens = eng->global_n_tokens;

        // Decode candidate token k
        gemma4_engine_decode(eng, draft_tokens[k], host_logits);

        // Check if greedy sample matches draft
        int greedy = gemma4_sample_argmax(host_logits, GEMMA4_VOCAB_SIZE);

        if (greedy == draft_tokens[k]) {
            n_accepted = k + 1;

            // Copy logits for accepted token
            if (logits_out) {
                memcpy(logits_out + k * GEMMA4_VOCAB_SIZE,
                       host_logits, GEMMA4_VOCAB_SIZE * sizeof(float));
            }
        } else {
            // Mismatch: rollback to before this token
            // Use the greedy token as the "correct" next token
            eng->global_n_tokens = saved_n_tokens;

            // Output the greedy logits
            if (logits_out) {
                memcpy(logits_out, host_logits,
                       GEMMA4_VOCAB_SIZE * sizeof(float));
            }
            free(host_logits);
            return n_accepted;
        }

        free(host_logits);
    }

    return n_accepted;  // All K accepted
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
// ─── LoRA GGUF loader helpers ───────────────────────────────────────────
// =========================================================================

// Parse a LoRA GGUF file and validate it matches the base model
static int lora_find_weight_offset(
    const uint8_t *lora_data,
    uint64_t       lora_size,
    const char    *tensor_name,
    uint64_t      *offset_out,
    uint64_t      *n_el_out)
{
    return gguf_find_tensor_offset(lora_data, lora_size, tensor_name,
                                    offset_out, n_el_out);
}

// Load a single LoRA adapter from GGUF tensors
// Looks for lora.{layer}.{weight}.weight_a and weight_b
// And metadata lora.{layer}.{weight}.alpha
static int load_lora_adapter(
    lora_adapter_t    *adapter,
    const uint8_t     *lora_data,
    uint64_t           lora_size,
    const char        *base_name,   // e.g. "lora.0.attn_q"
    int                in_dim,
    int                out_dim)
{
    char name_a[288], name_b[288], alpha_key[288];
    snprintf(name_a, sizeof(name_a), "%s.weight_a", base_name);
    snprintf(name_b, sizeof(name_b), "%s.weight_b", base_name);
    snprintf(alpha_key, sizeof(alpha_key), "%s.alpha", base_name);

    uint64_t off_a = 0, off_b = 0, n_a = 0, n_b = 0;
    int found_a = lora_find_weight_offset(lora_data, lora_size, name_a, &off_a, &n_a);
    int found_b = lora_find_weight_offset(lora_data, lora_size, name_b, &off_b, &n_b);
    if (found_a != 0 || found_b != 0) return -1;

    // Determine rank: weight_a is [in_dim × rank], weight_b is [rank × out_dim]
    // n_a = in_dim * rank, n_b = rank * out_dim
    // Derive rank from dimensions: n_a / in_dim should equal n_b / out_dim
    int rank_a = (int)(n_a / in_dim);
    int rank_b = (int)(n_b / out_dim);
    if (rank_a != rank_b || rank_a <= 0 || rank_a > GEMMA4_MAX_LORA_RANK) {
        fprintf(stderr, "gem4d: LoRA rank mismatch for %s: a=%d b=%d\n",
                base_name, rank_a, rank_b);
        return -1;
    }
    int rank = rank_a;

    // Parse alpha from metadata (default to 1.0)
    float alpha = (float)rank;  // default: alpha = rank (so scale = 1.0)
    gguf_parse_metadata(lora_data, lora_size, alpha_key,
                        &alpha, GGUF_TYPE_FLOAT32);
    float scale = alpha / (float)rank;

    // Allocate device memory
    float *d_a = NULL, *d_b = NULL;
    size_t bytes_a = (size_t)in_dim * rank * sizeof(float);
    size_t bytes_b = (size_t)rank * out_dim * sizeof(float);

    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);

    // off_a / off_b are ABSOLUTE file offsets (gguf_find_tensor adds the
    // tensor-data start), so index directly into the mmap'd LoRA file.
    const float *host_a = (const float *)(lora_data + off_a);
    const float *host_b = (const float *)(lora_data + off_b);
    cudaMemcpy(d_a, host_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, host_b, bytes_b, cudaMemcpyHostToDevice);

    // Fill adapter struct
    if (adapter->active) {
        cudaFree(adapter->d_a);
        cudaFree(adapter->d_b);
    }
    adapter->d_a  = d_a;
    adapter->d_b  = d_b;
    adapter->scale = scale;
    adapter->rank = rank;
    adapter->input_dim = in_dim;
    adapter->output_dim = out_dim;
    adapter->active = 1;

    fprintf(stderr, "gem4d: LoRA loaded %s (rank=%d, alpha=%.1f)\n",
            base_name, rank, alpha);
    return 0;
}

// =========================================================================
// ─── LoRA Public API ────────────────────────────────────────────────────
// =========================================================================

int gemma4_engine_load_lora(
    gemma4_engine_t *eng,
    const char      *lora_path,
    float            scale)
{
    if (!eng || !lora_path) return -1;

    // Open and mmap LoRA GGUF
    int fd = open(lora_path, O_RDONLY);
    if (fd < 0) { perror("gem4d: open lora"); return -1; }

    struct stat st;
    fstat(fd, &st);
    uint64_t lora_size = st.st_size;

    const uint8_t *lora_data = (const uint8_t *)mmap(
        NULL, lora_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (lora_data == MAP_FAILED) {
        perror("gem4d: mmap lora");
        close(fd);
        return -1;
    }

    // Validate GGUF magic
    const gguf_header_t *hdr = (const gguf_header_t *)lora_data;
    if (hdr->magic != 0x46554747) {
        fprintf(stderr, "gem4d: invalid LoRA GGUF magic\n");
        munmap((void *)lora_data, lora_size);
        close(fd);
        return -1;
    }

    fprintf(stderr, "gem4d: loading LoRA from %s...\n", lora_path);
    eng->lora_scale = scale;

    // Unload previous LoRA if any
    if (eng->lora_loaded) {
        gemma4_engine_unload_lora(eng);
    }

    // Try loading per-layer LoRA adapters
    char tname[256];
    int  n_loaded = 0;

    // Q, K, V, O for each layer
    for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
        int q_dim, kv_dim;
        if (eng->layer_types[l] == LAYER_SLIDING) {
            q_dim  = GEMMA4_HEADS * GEMMA4_HEAD_DIM;              // 4096
            kv_dim = GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM;           // 2048
        } else {
            q_dim  = GEMMA4_HEADS * GEMMA4_GLOBAL_HEAD_DIM;       // 8192
            kv_dim = GEMMA4_GLOBAL_KV_HEADS * GEMMA4_GLOBAL_HEAD_DIM; // 512
        }

        snprintf(tname, sizeof(tname), "lora.%d.attn_q", l);
        if (load_lora_adapter(&eng->lora[l].q, lora_data, lora_size,
                tname, GEMMA4_HIDDEN_SIZE, q_dim) == 0) n_loaded++;

        snprintf(tname, sizeof(tname), "lora.%d.attn_k", l);
        if (load_lora_adapter(&eng->lora[l].k, lora_data, lora_size,
                tname, GEMMA4_HIDDEN_SIZE, kv_dim) == 0) n_loaded++;

        snprintf(tname, sizeof(tname), "lora.%d.attn_v", l);
        if (load_lora_adapter(&eng->lora[l].v, lora_data, lora_size,
                tname, GEMMA4_HIDDEN_SIZE, kv_dim) == 0) n_loaded++;

        snprintf(tname, sizeof(tname), "lora.%d.attn_output", l);
        if (load_lora_adapter(&eng->lora[l].o, lora_data, lora_size,
                tname, GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE) == 0) n_loaded++;

        snprintf(tname, sizeof(tname), "lora.%d.ffn_gate", l);
        if (load_lora_adapter(&eng->lora[l].gate, lora_data, lora_size,
                tname, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE) == 0) n_loaded++;

        snprintf(tname, sizeof(tname), "lora.%d.ffn_up", l);
        if (load_lora_adapter(&eng->lora[l].up, lora_data, lora_size,
                tname, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE) == 0) n_loaded++;

        snprintf(tname, sizeof(tname), "lora.%d.ffn_down", l);
        if (load_lora_adapter(&eng->lora[l].down, lora_data, lora_size,
                tname, GEMMA4_INTERMEDIATE, GEMMA4_HIDDEN_SIZE) == 0) n_loaded++;
    }

    // Output projection LoRA
    if (load_lora_adapter(&eng->lora_output, lora_data, lora_size,
            "lora.output", GEMMA4_HIDDEN_SIZE, GEMMA4_VOCAB_SIZE) == 0)
        n_loaded++;

    strncpy(eng->lora_path, lora_path, sizeof(eng->lora_path) - 1);
    eng->lora_loaded = (n_loaded > 0);

    if (n_loaded > 0) {
        fprintf(stderr, "gem4d: LoRA loaded %d adapters from %s (scale=%.2f)\n",
                n_loaded, lora_path, scale);
    } else {
        fprintf(stderr, "gem4d: warning: no LoRA adapters found in %s\n", lora_path);
    }

    // Cleanup temp mapping (we keep the weights in GPU memory)
    munmap((void *)lora_data, lora_size);
    close(fd);

    return n_loaded > 0 ? 0 : -1;
}

void gemma4_engine_unload_lora(gemma4_engine_t *eng) {
    if (!eng || !eng->lora_loaded) return;

    for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
        #define LORA_FREE(field) do { \
            if (eng->lora[l].field.d_a) cudaFree(eng->lora[l].field.d_a); \
            if (eng->lora[l].field.d_b) cudaFree(eng->lora[l].field.d_b); \
            eng->lora[l].field.d_a = NULL; \
            eng->lora[l].field.d_b = NULL; \
            eng->lora[l].field.active = 0; \
        } while(0)
        LORA_FREE(q);    LORA_FREE(k);    LORA_FREE(v);
        LORA_FREE(o);    LORA_FREE(gate); LORA_FREE(up);
        LORA_FREE(down);
        #undef LORA_FREE
    }
    if (eng->lora_output.d_a) cudaFree(eng->lora_output.d_a);
    if (eng->lora_output.d_b) cudaFree(eng->lora_output.d_b);
    eng->lora_output.d_a = NULL;
    eng->lora_output.d_b = NULL;
    eng->lora_output.active = 0;

    eng->lora_loaded = 0;
    eng->lora_path[0] = '\0';
    fprintf(stderr, "gem4d: LoRA adapters unloaded\n");
}

int gemma4_engine_has_lora(const gemma4_engine_t *eng) {
    return eng ? eng->lora_loaded : 0;
}

// =========================================================================
// ─── Diagnostics ────────────────────────────────────────────────────────
// =========================================================================

void gemma4_engine_print_info(const gemma4_engine_t *eng) {
    if (!eng) return;
    printf("=== gem4d Engine Info ===\n");
    printf("Model size:  %.2f GB\n", eng->gguf_size / (1024.0*1024.0*1024.0));
    printf("Context:     %u tokens\n", eng->context_size);
    printf("Format:      %s\n", eng->format == FORMAT_FP8 ? "FP8" : "Q8_0");
    printf("Layers:      %d total (%d sliding, %d global)\n",
            GEMMA4_MAX_LAYERS, eng->n_layers_sliding, eng->n_layers_global);
    printf("Hidden:      %d -> %d -> %d\n",
            GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE, GEMMA4_HIDDEN_SIZE);
    printf("Heads:       %d Q, %d KV sliding, %d KV global\n",
            GEMMA4_HEADS, GEMMA4_KV_HEADS, GEMMA4_GLOBAL_KV_HEADS);
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
    if (eng->lora_loaded) {
        printf("LoRA:        %s (scale=%.2f)\n", eng->lora_path, eng->lora_scale);
        int n_active = 0;
        for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
            if (eng->lora[l].q.active) n_active++;
        }
        printf("LoRA:        %d layers active\n", n_active);
    }
}

void gemma4_engine_print_timing(const gemma4_engine_t *eng) {
    if (!eng) return;
    printf("=== gem4d Timing ===\n");
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
}

int gemma4_engine_get_n_layers(const gemma4_engine_t *eng) {
    return eng ? GEMMA4_MAX_LAYERS : 0;
}

// ─── Timing accessors (for Go-side speed logging) ─────────────────────
float gemma4_engine_prefill_ms(const gemma4_engine_t *eng) {
    return eng ? eng->prefill_time_ms : 0.0f;
}
int gemma4_engine_prefill_tokens(const gemma4_engine_t *eng) {
    return eng ? eng->n_prefill_tokens : 0;
}
float gemma4_engine_decode_ms(const gemma4_engine_t *eng) {
    return eng ? eng->decode_time_ms : 0.0f;
}
int gemma4_engine_decode_tokens(const gemma4_engine_t *eng) {
    return eng ? eng->n_decode_tokens : 0;
}

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
    return eng ? eng->global_n_tokens : 0;
}

void gemma4_engine_reset(gemma4_engine_t *eng) {
    if (!eng) return;
    eng->global_n_tokens = 0;
    for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
        eng->sliding_cursor[l] = 0;
        eng->sliding_filled[l] = 0;
    }
}

// Rewind the KV cache to keep only the first `n_keep` tokens, discarding the
// rest. This enables prefix reuse: when a new request shares a prefix with the
// cached sequence, we rewind to the shared length and prefill only the suffix.
//
// Correctness with the sliding-window ring buffers:
//   Each local layer stores K/V for absolute token t at ring slot (t % window).
//   If the original sequence length exceeded the window, the buffer wrapped and
//   the slots for the kept prefix [n_keep-window, n_keep) may have been
//   overwritten by later tokens — rewinding would then read stale KV. We detect
//   this case and refuse (return -1) so the caller can do a full reset+reprefill.
//   When the buffer never wrapped (len <= window), every token still occupies
//   its own slot and rewinding is exact.
//
// Returns 0 on success, -1 if the rewind would be unsafe (caller should reset).
int gemma4_engine_rewind(gemma4_engine_t *eng, int n_keep) {
    if (!eng) return -1;
    if (n_keep < 0 || n_keep > eng->global_n_tokens) return -1;
    if (n_keep == eng->global_n_tokens) return 0; // nothing to discard

    // Unsafe if the sliding ring buffer has wrapped past the kept prefix.
    if (eng->global_n_tokens > GEMMA4_SLIDING_WINDOW) return -1;

    eng->global_n_tokens = n_keep;
    for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
        eng->sliding_cursor[l] = n_keep % GEMMA4_SLIDING_WINDOW;
        eng->sliding_filled[l] = n_keep < GEMMA4_SLIDING_WINDOW
                                     ? n_keep : GEMMA4_SLIDING_WINDOW;
    }
    return 0;
}

// =========================================================================
// ─── Session Save/Load ─────────────────────────────────────────────────
// =========================================================================

int gemma4_engine_save_session(
    gemma4_engine_t *eng,
    uint8_t         *buffer,
    uint64_t        *size)
{
    if (!eng) return -1;

    // Calculate KV cache size
    uint64_t sliding_bytes = (uint64_t)GEMMA4_MAX_LAYERS *
        GEMMA4_KV_HEADS * GEMMA4_SLIDING_WINDOW * GEMMA4_HEAD_DIM * sizeof(float);

    uint64_t global_bytes = (uint64_t)eng->n_layers_global *
        GEMMA4_GLOBAL_KV_HEADS * eng->global_n_tokens * GEMMA4_GLOBAL_HEAD_DIM * sizeof(unsigned char);

    uint64_t needed = sizeof(uint32_t) * 3 +  // n_tokens, n_global, metadata
        sliding_bytes + global_bytes;

    if (!buffer) {
        *size = needed;
        return 0;
    }

    if (*size < needed) return -1;

    uint8_t *p = buffer;
    *(uint32_t *)p = eng->global_n_tokens; p += 4;
    *(uint32_t *)p = eng->n_layers_global; p += 4;
    *(uint32_t *)p = (uint32_t)eng->global_kv_capacity; p += 4;

    // Save sliding KV
    cudaMemcpy(p, eng->d_sliding_k, sliding_bytes, cudaMemcpyDeviceToHost);
    p += sliding_bytes;
    cudaMemcpy(p, eng->d_sliding_v, sliding_bytes, cudaMemcpyDeviceToHost);
    p += sliding_bytes;

    // Save global KV K and V (only filled portion)
    cudaMemcpy(p, eng->d_global_k, global_bytes, cudaMemcpyDeviceToHost);
    p += global_bytes;
    cudaMemcpy(p, eng->d_global_v, global_bytes, cudaMemcpyDeviceToHost);
    p += global_bytes;

    *size = needed;
    return 0;
}

int gemma4_engine_load_session(
    gemma4_engine_t *eng,
    const uint8_t   *buffer,
    uint64_t         size)
{
    if (!eng || !buffer) return -1;

    const uint8_t *p = buffer;
    eng->global_n_tokens = *(const uint32_t *)p; p += 4;
    int n_global = *(const uint32_t *)p; p += 4;
    (void)n_global; // validate if needed
    eng->global_kv_capacity = *(const uint32_t *)p; p += 4;

    uint64_t sliding_bytes = (uint64_t)GEMMA4_MAX_LAYERS *
        GEMMA4_KV_HEADS * GEMMA4_SLIDING_WINDOW * GEMMA4_HEAD_DIM * sizeof(float);

    // Restore sliding KV
    cudaMemcpy(eng->d_sliding_k, p, sliding_bytes, cudaMemcpyHostToDevice);
    p += sliding_bytes;
    cudaMemcpy(eng->d_sliding_v, p, sliding_bytes, cudaMemcpyHostToDevice);
    p += sliding_bytes;

    // Restore global KV
    uint64_t global_bytes = (uint64_t)GEMMA4_MAX_LAYERS *
        eng->global_n_tokens * GEMMA4_GLOBAL_HEAD_DIM * sizeof(float);
    cudaMemcpy(eng->d_global_k, p, global_bytes, cudaMemcpyHostToDevice);
    p += global_bytes;
    cudaMemcpy(eng->d_global_v, p, global_bytes, cudaMemcpyHostToDevice);

    // Reset sliding cursors (invalidate for safety)
    for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
        eng->sliding_cursor[l] = eng->global_n_tokens % GEMMA4_SLIDING_WINDOW;
        eng->sliding_filled[l] = min(eng->global_n_tokens, GEMMA4_SLIDING_WINDOW);
    }

    return 0;
}
