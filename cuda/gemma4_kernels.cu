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
    sum = block_reduce_sum(sum, smem);
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
    extern __shared__ float smem[];    int idx = blockIdx.x;
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
    sg = block_reduce_sum(sg, smem);
    su = block_reduce_sum(su, smem);
    if (threadIdx.x == 0) { out_gate[idx] = sg; out_up[idx] = su; }
}

// Batched GEMV on a quantized weight: out[NK][out_dim] = x[NK][in_dim] · Wᵀ, reading
// each weight row ONCE and reusing the decoded element across all NK input vectors.
// This is the decode-speed lever: NK draft tokens cost ~one token's WEIGHT bandwidth
// (12.65 GB), not NK× — what makes speculative decoding pay. NK is a COMPILE-TIME
// constant so acc[NK] stays in REGISTERS and the k-loops fully unroll (a dynamic K
// puts acc[] in local memory, whose per-element RMW traffic scales with K and erases
// the win). Token-major: x[k*in_dim+i], out[k*out_dim+idx]. One block per output row.
template<int NK>
__global__ void gemv_batched_kernel_t(
    float          *out,      // [NK × out_dim]
    const uint8_t  *weight,   // row-major [out_dim × in_dim]
    const float    *x,        // [NK × in_dim]
    int             in_dim,
    int             out_dim,
    int             fmt)
{
    extern __shared__ float smem[];
    int idx = blockIdx.x;
    if (idx >= out_dim) return;
    const uint8_t *row = weight + (size_t)idx * (fmt == 0 ? in_dim : (size_t)(in_dim/32)*34);
    float acc[NK];
    #pragma unroll
    for (int k = 0; k < NK; k++) acc[k] = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        float w = decode_weight(row, i, fmt);          // decode the weight element ONCE
        #pragma unroll
        for (int k = 0; k < NK; k++) acc[k] += w * x[(size_t)k*in_dim + i];
    }
    #pragma unroll
    for (int k = 0; k < NK; k++) {
        float s = block_reduce_sum(acc[k], smem);
        if (threadIdx.x == 0) out[(size_t)k*out_dim + idx] = s;
    }
}

// Dispatch the batched GEMV to the compile-time-NK kernel. K ≤ 8 → one weight pass;
// K > 8 is split into ≤8-wide chunks (each re-reads the weight, still far cheaper
// than K separate decodes).
static void gemv_batched_launch(
    float *out, const uint8_t *weight, const float *x,
    int in_dim, int out_dim, int K, int fmt, cudaStream_t stream)
{
    dim3 g(out_dim); int b = 256; size_t sm = 32*sizeof(float);
    #define LAUNCH(NK) gemv_batched_kernel_t<NK><<<g,b,sm,stream>>>(out,weight,x,in_dim,out_dim,fmt)
    switch (K) {
        case 1: LAUNCH(1); break;  case 2: LAUNCH(2); break;
        case 3: LAUNCH(3); break;  case 4: LAUNCH(4); break;
        case 5: LAUNCH(5); break;  case 6: LAUNCH(6); break;
        case 7: LAUNCH(7); break;  case 8: LAUNCH(8); break;
        default:
            for (int o = 0; o < K; o += 8) {
                int kk = (K - o < 8) ? (K - o) : 8;
                gemv_batched_launch(out + (size_t)o*out_dim, weight,
                                    x + (size_t)o*in_dim, in_dim, out_dim, kk, fmt, stream);
            }
    }
    #undef LAUNCH
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
    sg = block_reduce_sum(sg, smem);
    su = block_reduce_sum(su, smem);

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
__global__ void sliding_attn_decode_kernel(
    float       *output,       // [n_heads × head_dim]
    const float *q,            // [n_heads × head_dim]
    const float *k_cache,      // [window_size][n_kv_heads][head_dim]
    const float *v_cache,
    int          n_heads,
    int          n_kv_heads,
    int          head_dim,
    int          window_size,
    int          cursor,
    int          filled)
{
    extern __shared__ float smem[];      // [32] block_reduce scratch only
    int q_head  = blockIdx.x;
    int kv_head = q_head / (n_heads / n_kv_heads); // GQA mapping
    int tid     = threadIdx.x;

    int window_len = min(filled, window_size);

    float q_d = (tid < head_dim) ? q[q_head * head_dim + tid] : 0.0f;
    float acc = 0.0f;                    // this thread's output element
    float m   = -INFINITY;               // running row max
    float l   = 0.0f;                    // running denominator

    for (int t = 0; t < window_len; t++) {
        int ring_idx = (cursor - window_len + t + window_size) % window_size;
        float k_d = (tid < head_dim)
            ? k_cache[(ring_idx * n_kv_heads + kv_head) * head_dim + tid]
            : 0.0f;
        // Full dot product, broadcast to every thread (block_reduce_sum returns the
        // reduced value to all lanes; attention scale is 1.0 for gemma4).
        float s = block_reduce_sum(q_d * k_d, smem);
        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);     // m=-inf on t=0 ⇒ alpha=0 (acc/l are 0)
        float p     = __expf(s - m_new);
        l   = l * alpha + p;
        float v_d = (tid < head_dim)
            ? v_cache[(ring_idx * n_kv_heads + kv_head) * head_dim + tid]
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
    const float *k_cache,      // [ctx_len][head_dim]
    const float *v_cache,      // [ctx_len][head_dim]
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
        float k_d = (tid < head_dim) ? k_cache[t * head_dim + tid] : 0.0f;
        float s = block_reduce_sum(q_d * k_d, smem);
        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);
        float p     = __expf(s - m_new);
        l   = l * alpha + p;
        float v_d = (tid < head_dim) ? v_cache[t * head_dim + tid] : 0.0f;
        acc = acc * alpha + p * v_d;
        m   = m_new;
    }

    if (tid < head_dim)
        output[q_head * head_dim + tid] = (l > 0.0f) ? acc / l : 0.0f;
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

// NEOX RoPE over [rows][n_heads][head_dim] for Q and [rows][n_kv_heads][head_dim]
// for K. Position of row r is base_pos+r. ff=freq_factors (global) or NULL (=1,
// sliding). grid=(n_heads, rows), block=head_dim/2. Matches rope_*_kernel above.
__global__ void rope_rows_kernel(
    float *q, float *k, int base_pos, int n_heads, int n_kv_heads,
    int head_dim, int rows, float theta_base, const float *freq_factors)
{
    int d = threadIdx.x, half = head_dim / 2;
    if (d >= half) return;
    int head = blockIdx.x, row = blockIdx.y;
    if (row >= rows) return;
    int pos = base_pos + row;
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
__global__ void kv_write_sliding_kernel(
    float *kcache, float *vcache, const float *kb, const float *vb,
    int base, int first, int count, int kvhd, int window)
{
    int i = blockIdx.y;                                // 0..count-1
    int j = blockIdx.x * blockDim.x + threadIdx.x;     // 0..kvhd-1
    if (i >= count || j >= kvhd) return;
    int t    = first + i;                              // token index within batch
    int slot = (base + t) % window;
    kcache[(size_t)slot * kvhd + j] = kb[(size_t)t * kvhd + j];
    vcache[(size_t)slot * kvhd + j] = vb[(size_t)t * kvhd + j];
}

// Scatter the batch's K/V into the linear global cache at positions base..base+rows-1.
// k/vcache point at the layer slot's base [capacity][hd]. grid=(ceil(hd/256), rows).
__global__ void kv_write_global_kernel(
    float *kcache, float *vcache, const float *kb, const float *vb,
    int base, int rows, int hd)
{
    int t = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows || j >= hd) return;
    int pos = base + t;
    kcache[(size_t)pos * hd + j] = kb[(size_t)t * hd + j];
    vcache[(size_t)pos * hd + j] = vb[(size_t)t * hd + j];
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

    // Tensor data copied to device memory at load (llama.cpp-style backend
    // buffer). When non-NULL, weight accessors return pointers into this
    // buffer instead of the mmap'd file, avoiding GPU page faults / host
    // memory reads on every GEMV.
    uint8_t  *d_weights;
    uint64_t  tdata_start;   // file offset where tensor data begins

    // ROTATING per-layer BF16 dequant scratch for batched cuBLAS prefill (Step 2).
    // The 7 projection weights of the CURRENT layer only are dequantized Q8_0/FP8 →
    // BF16 here (row-major [out_dim][in_dim], same element order as the source) just
    // before that layer's GEMMs, then reused by the next layer. This replaces the
    // old persistent 21.8 GB d_bf16[48][7] buffer with a ~0.5 GB scratch — keeping
    // ~12 GB resident (not ~34 GB) so the GGUF stays page-cached across runs and
    // cold reloads do not re-pay the slow disk read. proj index: 0=q 1=k 2=v 3=o
    // 4=gate 5=up 6=down (global layers have no separate V → d_bf16_layer[2] NULL).
    __nv_bfloat16 *d_bf16_layer[7];     // [out_dim×in_dim] per projection, max-sized
    int            bf16_ready;          // 1 once the scratch is allocated

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
    // Inverse map: absolute layer id -> contiguous global cache slot
    // (0..n_layers_global-1), or -1 for sliding layers. The global KV cache is
    // allocated for n_layers_global slots only (not all GEMMA4_MAX_LAYERS), so
    // every read/write into d_global_k/v must index by global_slot[layer].
    int             global_slot[GEMMA4_MAX_LAYERS];

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

    // Engine-resident batched-decode (speculative) scratch, sized for GEMMA4_SPEC_MAX
    // rows and allocated once (lazily). Reused every decode_batched call so probe /
    // re-probe steps pay no per-call cudaMalloc/free — d_sb[12] holds, in order:
    // tok, x, norm, inf, q, k, v, attn, o, gate, up, logitsK.
    float  *d_sb[12];
    int     sb_ready;

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
    float  h_out_scale[GEMMA4_MAX_LAYERS]; // layer_output_scale scalars (host)

    // KV cache (device)
    // Sliding: 40 layers × 1024 window × 8 heads × 256 head_dim (fp32).
    float  *d_sliding_k;       // [40 × 1024 × 8 × 256]
    float  *d_sliding_v;
    int     sliding_cursor[GEMMA4_MAX_LAYERS];
    int     sliding_filled[GEMMA4_MAX_LAYERS];

    // Global: 8 slots × ctx_size × 512 (fp32). K and V stored separately because
    // K gets RMSNorm+RoPE while V gets only plain (weightless) RMSNorm.
    float  *d_global_k;  // [n_layers_global × ctx_size × 512]
    float  *d_global_v;
    int     global_n_tokens;
    int     global_kv_capacity;
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

    // Auto-detect the weight format from the GGUF tensor table. Trusting the
    // CLI flag is dangerous: decoding Q8_0 blocks as FP8 bytes yields NaNs.
    {
        uint64_t _off = 0, _n = 0; uint32_t gtype = 0;
        if (gguf_find_tensor(eng->gguf_data, eng->gguf_size,
                "token_embd.weight", &_off, &_n, &gtype) == 0) {
            tensor_format_t detected =
                (gtype == GGML_TYPE_Q8_0) ? FORMAT_Q8_0 : FORMAT_FP8;
            if (detected != eng->format) {
                fprintf(stderr, "gem4d: GGUF tensor type %u — overriding format "
                                "%s -> %s\n", gtype,
                        eng->format == FORMAT_FP8 ? "fp8" : "q8_0",
                        detected     == FORMAT_FP8 ? "fp8" : "q8_0");
                eng->format = detected;
            }
        }
    }

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

    // Suppressed token ids → device list for the -inf logits mask
    eng->d_suppress = NULL;
    eng->n_suppress = 0;
    {
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
            fprintf(stderr, "gem4d: %d suppressed tokens masked\n", eng->n_suppress);
        }
    }

    // ── Preload all F32 norm weights to device (once) ────────────────────
    {
        const size_t hs = GEMMA4_HIDDEN_SIZE, hd = GEMMA4_GLOBAL_HEAD_DIM;
        cudaMalloc(&eng->d_w_attn_norm,      GEMMA4_MAX_LAYERS * hs * sizeof(float));
        cudaMalloc(&eng->d_w_post_attn_norm, GEMMA4_MAX_LAYERS * hs * sizeof(float));
        cudaMalloc(&eng->d_w_ffn_norm,       GEMMA4_MAX_LAYERS * hs * sizeof(float));
        cudaMalloc(&eng->d_w_post_ffn_norm,  GEMMA4_MAX_LAYERS * hs * sizeof(float));
        cudaMalloc(&eng->d_w_q_norm,         GEMMA4_MAX_LAYERS * hd * sizeof(float));
        cudaMalloc(&eng->d_w_k_norm,         GEMMA4_MAX_LAYERS * hd * sizeof(float));
        cudaMalloc(&eng->d_w_out_norm,       hs * sizeof(float));

        // missing tensor (offset 0) → identity weight 1.0
        float *ones = (float *)malloc(hs * sizeof(float));
        for (size_t i = 0; i < hs; i++) ones[i] = 1.0f;

        #define UPLOAD_NORM(dst, off, n) do { \
            const void *src = (off) ? (const void *)(eng->gguf_data + (off)) : (const void *)ones; \
            cudaMemcpy((dst), src, (n) * sizeof(float), cudaMemcpyHostToDevice); \
        } while (0)

        for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
            int head_dim = (eng->layer_types[l] == LAYER_SLIDING)
                               ? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM;
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
        free(ones);
    }

    // ── Copy tensor data into device memory (llama.cpp-style residency) ──
    // GEMV kernels previously dereferenced the mmap'd file from the GPU.
    // That works on GB10 unified memory but pays page-fault + host-path
    // costs on every weight read. One bulk upload at load time instead.
    eng->d_weights = NULL;
    eng->tdata_start = gguf_tensor_data_start(eng->gguf_data, eng->gguf_size);
    if (eng->tdata_start != 0) {
        size_t tbytes = eng->gguf_size - eng->tdata_start;
        if (cudaMalloc(&eng->d_weights, tbytes) == cudaSuccess) {
            fprintf(stderr, "gem4d: uploading %.2f GB of weights to device...\n",
                    tbytes / (1024.0*1024.0*1024.0));
            // Pinned, double-buffered sequential streaming (fast). Fall back to the
            // direct mmap copy only if the pinned path fails to initialize.
            if (upload_weights_streamed(eng->d_weights, eng->gguf_fd,
                                        (off_t)eng->tdata_start, tbytes) != 0 &&
                cudaMemcpy(eng->d_weights, eng->gguf_data + eng->tdata_start,
                           tbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                fprintf(stderr, "gem4d: weight upload failed — falling back to mmap\n");
                cudaFree(eng->d_weights);
                eng->d_weights = NULL;
            }
        } else {
            fprintf(stderr, "gem4d: cudaMalloc(%zu) failed — using mmap'd weights\n",
                    tbytes);
            cudaGetLastError(); // clear error state
        }
    }

    // Build the absolute-id -> global-slot inverse map from global_layer_indices
    // (set by the layer-type detection above). Sliding layers map to -1.
    for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) eng->global_slot[l] = -1;
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

    // KV cache allocation
    // Sliding: 40 × 8 × 1024 × 256 × sizeof(float) = 40 × 8 × 1024 × 256 × 4 = 320 MB
    size_t sliding_kv_size = (size_t)GEMMA4_MAX_LAYERS *
        GEMMA4_KV_HEADS * GEMMA4_SLIDING_WINDOW * GEMMA4_HEAD_DIM * sizeof(float);
    if (cudaMalloc(&eng->d_sliding_k, sliding_kv_size) != cudaSuccess ||
        cudaMalloc(&eng->d_sliding_v, sliding_kv_size) != cudaSuccess) {
        fprintf(stderr, "gem4d: failed to allocate sliding KV cache (%.1f MB ×2)\n",
                sliding_kv_size / (1024.0*1024.0));
        cudaGetLastError();
        free(eng);
        return NULL;
    }

    // Global K and V caches: float, separate K and V.
    // Layout: [n_layers_global][ctx_size][head_dim] — only the global layers get
    // a slot (not all 48), so this is ~6× smaller than indexing by absolute id.
    eng->global_kv_capacity = context_size;
    size_t global_kv_size = (size_t)eng->n_layers_global *
        context_size * GEMMA4_GLOBAL_HEAD_DIM * sizeof(float);
    if (cudaMalloc(&eng->d_global_k, global_kv_size) != cudaSuccess ||
        cudaMalloc(&eng->d_global_v, global_kv_size) != cudaSuccess) {
        fprintf(stderr, "gem4d: failed to allocate global KV cache (%.1f MB ×2)\n",
                global_kv_size / (1024.0*1024.0));
        cudaGetLastError();
        free(eng);
        return NULL;
    }

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
    CUDA_FREE(eng->d_suppress);
    CUDA_FREE(eng->d_w_attn_norm);
    CUDA_FREE(eng->d_w_post_attn_norm);
    CUDA_FREE(eng->d_w_ffn_norm);
    CUDA_FREE(eng->d_w_post_ffn_norm);
    CUDA_FREE(eng->d_w_q_norm);
    CUDA_FREE(eng->d_w_k_norm);
    CUDA_FREE(eng->d_w_out_norm);
    CUDA_FREE(eng->d_weights);
    for (int p = 0; p < 7; p++)
        CUDA_FREE(eng->d_bf16_layer[p]);
    for (int p = 0; p < 12; p++)
        CUDA_FREE(eng->d_sb[p]);
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
// (gguf_find_tensor already adds the tensor-data start). When the weights
// were copied to device memory (d_weights), return a device pointer into that
// buffer; otherwise fall back to the mmap'd file (GB10 unified memory).
static inline const unsigned char* weight_fp8(
    const gemma4_engine_t *eng, uint64_t tensor_offset)
{
    if (eng->d_weights)
        return (const unsigned char *)(eng->d_weights
                                       + (tensor_offset - eng->tdata_start));
    return (const unsigned char *)(eng->gguf_data + tensor_offset);
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

// Batched GEMV: Y[K][out_dim] = X[K][in_dim] · weightᵀ, weight read once for all K.
static inline void gemv_batched_w(
    const gemma4_engine_t *eng,
    float *out, const uint8_t *weight, const float *x,
    int in_dim, int out_dim, int K, cudaStream_t stream)
{
    gemv_batched_launch(out, weight, x, in_dim, out_dim, K, FMT(eng), stream);
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

// Per-head RMSNorm using a norm weight already resident on device (or NULL).
#define PER_HEAD_NORM(dst, dev_weight, n_h, h_dim) do { \
    per_head_rms_norm_kernel<<<(n_h), (h_dim), smem32, stream>>>( \
        (dst), (dev_weight), (h_dim), GEMMA4_RMS_EPS); \
} while(0)

// Device pointer to layer `layer`'s preloaded hidden-size norm weight.
#define NORM_W(arr) (eng->arr + (size_t)layer * GEMMA4_HIDDEN_SIZE)
#define HEAD_NORM_W(arr) (eng->arr + (size_t)layer * GEMMA4_GLOBAL_HEAD_DIM)

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
    rms_norm_kernel<<<1, block, smem32, stream>>>(
        eng->d_norm, eng->d_x, NORM_W(d_w_attn_norm),
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
    PER_HEAD_NORM(eng->d_attn_q, HEAD_NORM_W(d_w_q_norm), n_heads, head_dim);

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
    PER_HEAD_NORM(eng->d_attn_k, HEAD_NORM_W(d_w_k_norm), n_kv_heads, head_dim);

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
        int slot = eng->global_slot[layer];
        size_t layer_stride = (size_t)eng->global_kv_capacity * GEMMA4_GLOBAL_HEAD_DIM;
        float *k_slot = eng->d_global_k + (size_t)slot * layer_stride
                        + (size_t)n * GEMMA4_GLOBAL_HEAD_DIM;
        float *v_slot = eng->d_global_v + (size_t)slot * layer_stride
                        + (size_t)n * GEMMA4_GLOBAL_HEAD_DIM;
        cudaMemcpyAsync(k_slot, eng->d_attn_k,
            kv_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(v_slot, eng->d_attn_v,
            kv_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    // ── 7. Attention ─────────────────────────────────────────────────────
    if (ltype == LAYER_SLIDING) {
        int smem_sl = 32 * (int)sizeof(float);   // flash: constant scratch
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
        // n_ctx = tokens already in cache + this one (written above).
        // Flash kernel uses a constant 32-float scratch, so there is no longer any
        // context-length shared-memory cap (the old (32+n_ctx)-float buffer capped
        // ctx at ~25K and failed silently past it).
        int n_ctx = eng->global_n_tokens + 1;
        int smem_gl = 32 * (int)sizeof(float);
        int slot = eng->global_slot[layer];
        size_t layer_stride = (size_t)eng->global_kv_capacity * GEMMA4_GLOBAL_HEAD_DIM;
        global_attn_decode_kernel<<<n_heads, head_dim, smem_gl, stream>>>(
            eng->d_attn_out, eng->d_attn_q,
            eng->d_global_k + (size_t)slot * layer_stride,
            eng->d_global_v + (size_t)slot * layer_stride,
            n_heads, head_dim, n_ctx);
        {
            cudaError_t le = cudaGetLastError();
            if (le != cudaSuccess)
                fprintf(stderr, "gem4d: global_attn_decode launch failed: %s\n",
                        cudaGetErrorString(le));
        }
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
    rms_norm_kernel<<<1, block, smem32, stream>>>(
        eng->d_norm, eng->d_x, NORM_W(d_w_post_attn_norm),
        GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_EPS);
    // Residual: normed_attn_proj + pre-layer input
    residual_add_kernel<<<(GEMMA4_HIDDEN_SIZE+255)/256, 256, 0, stream>>>(
        eng->d_norm, eng->d_residual, GEMMA4_HIDDEN_SIZE);
    // d_norm = attn_out = new residual for FFN
    cudaMemcpyAsync(eng->d_x,        eng->d_norm,
        GEMMA4_HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(eng->d_residual,  eng->d_norm,
        GEMMA4_HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // ── 9. Pre-FFN RMSNorm ────────────────────────────────────────────────
    rms_norm_kernel<<<1, block, smem32, stream>>>(
        eng->d_norm, eng->d_x, NORM_W(d_w_ffn_norm),
        GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_EPS);

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
    rms_norm_kernel<<<1, block, smem32, stream>>>(
        eng->d_norm, eng->d_x, NORM_W(d_w_post_ffn_norm),
        GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_EPS);
    residual_add_kernel<<<(GEMMA4_HIDDEN_SIZE+255)/256, 256, 0, stream>>>(
        eng->d_norm, eng->d_residual, GEMMA4_HIDDEN_SIZE);
    cudaMemcpyAsync(eng->d_x, eng->d_norm,
        GEMMA4_HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // ── 14. layer_output_scale (scalar preloaded at create) ──────────────
    if (eng->h_out_scale[layer] != 1.0f) {
        scale_kernel<<<(GEMMA4_HIDDEN_SIZE+255)/256, 256, 0, stream>>>(
            eng->d_x, GEMMA4_HIDDEN_SIZE, eng->h_out_scale[layer]);
    }

    return 0;
}

// =========================================================================
// ─── BF16 weight residency + batched GEMM (Step 2 foundation) ───────────
// =========================================================================

// Projection ids, indexing the rotating per-layer scratch eng->d_bf16_layer[p].
enum { PJ_Q = 0, PJ_K, PJ_V, PJ_O, PJ_GATE, PJ_UP, PJ_DOWN, PJ_COUNT };

// Resolve a projection's source byte offset + GEMM dims for a layer. Returns 0
// for projections that do not exist (global layers have no separate V weight).
static int proj_desc(const gemma4_engine_t *eng, int layer, int p,
                     uint64_t *offset, int *in_dim, int *out_dim)
{
    layer_type_t lt = eng->layer_types[layer];
    int hd  = (lt == LAYER_SLIDING) ? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM;
    int nkv = (lt == LAYER_SLIDING) ? GEMMA4_KV_HEADS : GEMMA4_GLOBAL_KV_HEADS;
    int oq  = GEMMA4_HEADS * hd;
    int okv = nkv * hd;
    const __typeof__(eng->tensors.layers[0]) *L = &eng->tensors.layers[layer];
    switch (p) {
        case PJ_Q:    *offset = L->attn_q;      *in_dim = GEMMA4_HIDDEN_SIZE; *out_dim = oq;  return 1;
        case PJ_K:    *offset = L->attn_k;      *in_dim = GEMMA4_HIDDEN_SIZE; *out_dim = okv; return 1;
        case PJ_V:    if (lt != LAYER_SLIDING) return 0;  // global: V = K, no weight
                      *offset = L->attn_v;      *in_dim = GEMMA4_HIDDEN_SIZE; *out_dim = okv; return 1;
        case PJ_O:    *offset = L->attn_output; *in_dim = oq;  *out_dim = GEMMA4_HIDDEN_SIZE; return 1;
        case PJ_GATE: *offset = L->ffn_gate;    *in_dim = GEMMA4_HIDDEN_SIZE; *out_dim = GEMMA4_INTERMEDIATE; return 1;
        case PJ_UP:   *offset = L->ffn_up;      *in_dim = GEMMA4_HIDDEN_SIZE; *out_dim = GEMMA4_INTERMEDIATE; return 1;
        case PJ_DOWN: *offset = L->ffn_down;    *in_dim = GEMMA4_INTERMEDIATE; *out_dim = GEMMA4_HIDDEN_SIZE; return 1;
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
    for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
        for (int p = 0; p < PJ_COUNT; p++) {
            uint64_t off; int in_dim, out_dim;
            if (!proj_desc(eng, l, p, &off, &in_dim, &out_dim)) continue;
            uint64_t n = (uint64_t)in_dim * out_dim;
            if (n > maxn[p]) maxn[p] = n;
        }
    }

    size_t total = 0;
    for (int p = 0; p < PJ_COUNT; p++) {
        eng->d_bf16_layer[p] = NULL;
        if (maxn[p] == 0) continue;
        if (cudaMalloc(&eng->d_bf16_layer[p],
                       maxn[p] * sizeof(__nv_bfloat16)) != cudaSuccess) {
            fprintf(stderr, "gem4d: BF16 dequant scratch alloc failed at proj %d "
                    "(%.2f GB in so far)\n", p, total / 1e9);
            cudaGetLastError();
            for (int q = 0; q < PJ_COUNT; q++)
                if (eng->d_bf16_layer[q]) { cudaFree(eng->d_bf16_layer[q]); eng->d_bf16_layer[q] = NULL; }
            return -1;
        }
        total += maxn[p] * sizeof(__nv_bfloat16);
    }
    fprintf(stderr, "gem4d: BF16 per-layer dequant scratch (%.2f GB rotating, "
            "vs 21.8 GB persistent)\n", total / 1e9);
    eng->bf16_ready = 1;
    return 0;
}

// Dequantize the CURRENT layer's 7 projection weights (Q8_0/FP8 → BF16) into the
// rotating scratch, just before that layer's GEMMs. Single-stream ordering makes
// this safe: the next layer's dequant is enqueued after this layer's GEMMs finish
// reading the scratch, so there is no overwrite hazard.
static void dequant_layer_bf16(gemma4_engine_t *eng, int l, cudaStream_t stream)
{
    for (int p = 0; p < PJ_COUNT; p++) {
        uint64_t off; int in_dim, out_dim;
        if (!proj_desc(eng, l, p, &off, &in_dim, &out_dim)) continue;
        uint64_t n = (uint64_t)in_dim * out_dim;
        const uint8_t *src = weight_fp8(eng, off);
        dequant_to_bf16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
            eng->d_bf16_layer[p], src, n, FMT(eng));
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
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}


// =========================================================================
// ─── Batched Prefill (Step 2 Phase 2 + Step 3) ──────────────────────────
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
    if (!eng->loaded || n_tokens <= 0) return -1;
    if (eng->global_n_tokens != 0) return -2;             // need fresh sequence
    if (n_tokens > eng->global_kv_capacity) return -2;    // would overflow cache
    if (n_tokens > 65535) return -2;                      // grid.y bound
    if (build_bf16_weights(eng) != 0) return -1;

    cudaStream_t stream = eng->stream;
    const int N   = n_tokens;
    const int H   = GEMMA4_HIDDEN_SIZE;
    const int I   = GEMMA4_INTERMEDIATE;
    const int HD2 = 32 * sizeof(float);
    const int base = 0;

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, stream);

    // ── Scratch (token-major). Sized to the widest dim each can take. ──
    const int OQ_MAX = GEMMA4_HEADS * GEMMA4_GLOBAL_HEAD_DIM; // 8192
    const int OKV_MAX = GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM;    // 2048
    float *d_x=0,*d_norm=0,*d_q=0,*d_k=0,*d_v=0,*d_attn=0,*d_gate=0,*d_up=0,*d_scores=0;
    __nv_bfloat16 *d_inb=0,*d_qb=0,*d_kb=0,*d_vb=0,*d_kbx=0,*d_vbx=0,*d_pb=0;
    const int HEADS = GEMMA4_HEADS;
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
        fprintf(stderr, "gem4d: batched-prefill scratch alloc failed (N=%d) — fallback\n", N);
        cudaGetLastError();
        for (float *p : fbufs) if(p) cudaFree(p);
        for (__nv_bfloat16 *p : bbufs) if(p) cudaFree(p);
        return -2;
    }

    auto grid1d = [](size_t n){ return (unsigned)((n + 255) / 256); };
    // One projection GEMM: bf16 normed input d_inb → fp32 token-major [N][out].
    // Weight comes from the rotating per-layer scratch (dequant'd at layer top).
    auto gemm_proj = [&](int l, int p, int in_dim, int out_dim, float *dst){
        (void)l;
        gemm_bf16(eng, eng->d_bf16_layer[p], d_inb, dst, in_dim, out_dim, N);
    };

    // ── Embedding + √H scale, token-major [N][H] ──
    embed_w(eng, d_x, weight_fp8(eng, eng->tensors.token_embd), tokens, N, H, stream);
    scale_kernel<<<grid1d((size_t)N*H),256,0,stream>>>(d_x, N*H, sqrtf((float)H));

    for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
        layer_type_t lt = eng->layer_types[l];
        int hd  = (lt==LAYER_SLIDING)? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM;
        int nkv = (lt==LAYER_SLIDING)? GEMMA4_KV_HEADS : GEMMA4_GLOBAL_KV_HEADS;
        int oq  = GEMMA4_HEADS*hd, okv = nkv*hd, kvhd = okv;
        const float *w_attn   = eng->d_w_attn_norm      + (size_t)l*H;
        const float *w_post_a = eng->d_w_post_attn_norm + (size_t)l*H;
        const float *w_ffn    = eng->d_w_ffn_norm       + (size_t)l*H;
        const float *w_post_f = eng->d_w_post_ffn_norm  + (size_t)l*H;
        const float *w_qn     = eng->d_w_q_norm + (size_t)l*GEMMA4_GLOBAL_HEAD_DIM;
        const float *w_kn     = eng->d_w_k_norm + (size_t)l*GEMMA4_GLOBAL_HEAD_DIM;

        // Dequant THIS layer's 7 projection weights Q8_0/FP8 → BF16 rotating scratch.
        dequant_layer_bf16(eng, l, stream);

        // Sandwich-norm block with IN-PLACE residual: d_x stays the pre-block
        // hidden (the residual) until residual_add folds the normed sub-block
        // contribution back into it — so no separate d_res buffer or D2D copies.
        // pre-attn RMSNorm → input prep (one input feeds Q,K,V → smooth group 0)
        rms_norm_rows_bf16_kernel<<<N,256,HD2,stream>>>(d_inb, d_x, w_attn, H, N, GEMMA4_RMS_EPS);

        // Q,K (and V) projections (one normed input feeds all three)
        gemm_proj(l, PJ_Q, H, oq, d_q);
        per_head_rms_norm_rows_kernel<<<dim3(GEMMA4_HEADS,N),hd,HD2,stream>>>(d_q, w_qn, GEMMA4_HEADS, hd, N, GEMMA4_RMS_EPS);
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
            rope_rows_kernel<<<dim3(GEMMA4_HEADS,N),hd/2,0,stream>>>(d_q, d_k, base, GEMMA4_HEADS, nkv, hd, N, 10000.0f, NULL);
        else
            rope_rows_kernel<<<dim3(GEMMA4_HEADS,N),hd/2,0,stream>>>(d_q, d_k, base, GEMMA4_HEADS, nkv, hd, N, 1000000.0f, eng->d_rope_freqs);

        // Attention → d_attn [N][oq], batched over all heads via tensor-core GEMMs:
        // expand K/V to all query heads (GQA), S=Q·Kᵀ (col-major [HEADS][N×N]) →
        // masked softmax → P(bf16) → O=V·Pᵀ. Scale 1.0 (gemma4).
        int window = (lt==LAYER_SLIDING)? GEMMA4_SLIDING_WINDOW : 0;
        f32_to_bf16_kernel<<<grid1d((size_t)N*oq),256,0,stream>>>(d_qb, d_q, (size_t)N*oq);
        f32_to_bf16_kernel<<<grid1d((size_t)N*okv),256,0,stream>>>(d_kb, d_k, (size_t)N*okv);
        f32_to_bf16_kernel<<<grid1d((size_t)N*okv),256,0,stream>>>(d_vb, d_v, (size_t)N*okv);
        {
            kv_broadcast_bf16_kernel<<<dim3(grid1d(hd),GEMMA4_HEADS,N),256,0,stream>>>(d_kbx, d_kb, N, GEMMA4_HEADS, nkv, hd);
            kv_broadcast_bf16_kernel<<<dim3(grid1d(hd),GEMMA4_HEADS,N),256,0,stream>>>(d_vbx, d_vb, N, GEMMA4_HEADS, nkv, hd);
            const float a1=1.0f, b0=0.0f;
            long long sNN = (long long)N * N;
            // S[h] = Q[h]ᵀ·K[h]  (m=N,n=N,k=hd; A,B col-major [hd×N] ld=oq stride=hd)
            cublasGemmStridedBatchedEx(eng->cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, N, hd,
                &a1, d_qb,  CUDA_R_16BF, oq, (long long)hd,
                     d_kbx, CUDA_R_16BF, oq, (long long)hd,
                &b0, d_scores, CUDA_R_32F, N, sNN,
                GEMMA4_HEADS, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            attn_softmax_batched_kernel<<<dim3(N,GEMMA4_HEADS),256,HD2,stream>>>(d_scores, d_pb, N, window);
            // O[h] = V[h]·P[h]ᵀ  (m=hd,n=N,k=N → C col-major [hd×N] ld=oq stride=hd)
            cublasGemmStridedBatchedEx(eng->cublas, CUBLAS_OP_N, CUBLAS_OP_T, hd, N, N,
                &a1, d_vbx, CUDA_R_16BF, oq, (long long)hd,
                     d_pb,  CUDA_R_16BF, N,  sNN,
                &b0, d_attn, CUDA_R_32F, oq, (long long)hd,
                GEMMA4_HEADS, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }

        // Write final K/V into the persistent cache for decode continuation.
        if (lt == LAYER_SLIDING) {
            int count = (N < GEMMA4_SLIDING_WINDOW)? N : GEMMA4_SLIDING_WINDOW;
            int first = N - count;
            float *kc = eng->d_sliding_k + (size_t)l*GEMMA4_SLIDING_WINDOW*kvhd;
            float *vc = eng->d_sliding_v + (size_t)l*GEMMA4_SLIDING_WINDOW*kvhd;
            kv_write_sliding_kernel<<<dim3(grid1d(kvhd),count),256,0,stream>>>(
                kc, vc, d_k, d_v, base, first, count, kvhd, GEMMA4_SLIDING_WINDOW);
            eng->sliding_cursor[l] = (base + N) % GEMMA4_SLIDING_WINDOW;
            eng->sliding_filled[l] = (base + N < GEMMA4_SLIDING_WINDOW)? base+N : GEMMA4_SLIDING_WINDOW;
        } else {
            int slot = eng->global_slot[l];
            size_t stride = (size_t)eng->global_kv_capacity * GEMMA4_GLOBAL_HEAD_DIM;
            kv_write_global_kernel<<<dim3(grid1d(hd),N),256,0,stream>>>(
                eng->d_global_k + slot*stride, eng->d_global_v + slot*stride,
                d_k, d_v, base, N, hd);
        }

        // O projection → temp d_q ; post-attn norm ; fold into residual d_x
        f32_to_bf16_kernel<<<grid1d((size_t)N*oq),256,0,stream>>>(d_inb, d_attn, (size_t)N*oq);
        gemm_proj(l, PJ_O, oq, H, d_q);
        rms_norm_rows_kernel<<<N,256,HD2,stream>>>(d_norm, d_q, w_post_a, H, N, GEMMA4_RMS_EPS);
        residual_add_kernel<<<grid1d((size_t)N*H),256,0,stream>>>(d_x, d_norm, N*H);

        // FFN: pre-norm (d_x is now the post-attn hidden = FFN residual) → bf16
        // input (reused by gate+up), GeGLU, down → temp d_q, fold into d_x.
        rms_norm_rows_bf16_kernel<<<N,256,HD2,stream>>>(d_inb, d_x, w_ffn, H, N, GEMMA4_RMS_EPS);
        gemm_proj(l, PJ_GATE, H, I, d_gate);
        gemm_proj(l, PJ_UP,   H, I, d_up);
        geglu_bf16_kernel<<<grid1d((size_t)N*I),256,0,stream>>>(d_inb, d_gate, d_up, N*I);
        gemm_proj(l, PJ_DOWN, I, H, d_q);
        rms_norm_rows_kernel<<<N,256,HD2,stream>>>(d_norm, d_q, w_post_f, H, N, GEMMA4_RMS_EPS);
        residual_add_kernel<<<grid1d((size_t)N*H),256,0,stream>>>(d_x, d_norm, N*H);
        if (eng->h_out_scale[l] != 1.0f)
            scale_kernel<<<grid1d((size_t)N*H),256,0,stream>>>(d_x, N*H, eng->h_out_scale[l]);
    }

    eng->global_n_tokens += N;

    // ── Output norm + LM head + softcap on the LAST token only ──
    float *x_last = d_x + (size_t)(N-1)*H;
    rms_norm_kernel<<<1,256,HD2,stream>>>(eng->d_norm, x_last, eng->d_w_out_norm, H, GEMMA4_RMS_EPS);
    gemv_w(eng, eng->d_logits, weight_fp8(eng, eng->tensors.output_weight),
           eng->d_norm, H, GEMMA4_VOCAB_SIZE, stream);
    logit_softcap_kernel<<<grid1d(GEMMA4_VOCAB_SIZE),256,0,stream>>>(
        eng->d_logits, GEMMA4_SOFTCAP, GEMMA4_VOCAB_SIZE);
    if (eng->n_suppress > 0)
        suppress_tokens_kernel<<<grid1d(eng->n_suppress),256,0,stream>>>(
            eng->d_logits, eng->d_suppress, eng->n_suppress, GEMMA4_VOCAB_SIZE);
    if (logits_out)
        cudaMemcpyAsync(logits_out, eng->d_logits, GEMMA4_VOCAB_SIZE*sizeof(float),
                        cudaMemcpyDeviceToHost, stream);

    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    eng->prefill_time_ms += ms;
    eng->n_prefill_tokens += N;
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    for (float *p : fbufs) cudaFree(p);
    for (__nv_bfloat16 *p : bbufs) cudaFree(p);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gem4d: batched-prefill CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
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

        // Output norm + LM head + softcap only for the LAST prefill token.
        // The LM head is a 262144-row GEMV (~1 GB of weight reads) — running it
        // for every prompt token wasted >20% of prefill bandwidth for logits
        // that were thrown away. llama.cpp does the same via inp_out_ids.
        if (t != n_tokens - 1) continue;

        rms_norm_kernel<<<1, 256, 32*sizeof(float), stream>>>(
            eng->d_norm, eng->d_x, eng->d_w_out_norm,
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

        if (eng->n_suppress > 0)
            suppress_tokens_kernel<<<(eng->n_suppress + 255) / 256, 256, 0, stream>>>(
                eng->d_logits, eng->d_suppress, eng->n_suppress, vocab);

        if (logits_out) {
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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gem4d: prefill CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

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
    rms_norm_kernel<<<1, 256, 32*sizeof(float), stream>>>(
        eng->d_norm, eng->d_x, eng->d_w_out_norm,
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

    if (eng->n_suppress > 0)
        suppress_tokens_kernel<<<(eng->n_suppress + 255) / 256, 256, 0, stream>>>(
            eng->d_logits, eng->d_suppress, eng->n_suppress, vocab);

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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gem4d: decode CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
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
    const size_t H = GEMMA4_HIDDEN_SIZE, I = GEMMA4_INTERMEDIATE, V = GEMMA4_VOCAB_SIZE;
    const size_t OQ = (size_t)GEMMA4_HEADS*GEMMA4_GLOBAL_HEAD_DIM;
    const size_t OKV = (size_t)GEMMA4_KV_HEADS*GEMMA4_HEAD_DIM;
    size_t elems[12] = { M, M*H, M*H, M*I, M*OQ, M*OKV, M*OKV, M*OQ, M*H, M*I, M*I, M*V };
    for (int i = 0; i < 12; i++) {
        if (cudaMalloc(&eng->d_sb[i], elems[i]*sizeof(float)) != cudaSuccess) {
            for (int j = 0; j < i; j++) { cudaFree(eng->d_sb[j]); eng->d_sb[j] = NULL; }
            cudaGetLastError();
            return -1;
        }
    }
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
int gemma4_engine_decode_batched(
    gemma4_engine_t *eng, const int32_t *tokens, int K, float *logits_out)
{
    if (!eng->loaded || K <= 0) return -1;
    if (eng->lora_loaded) return -2;                 // batched path has no LoRA support
    if (K > GEMMA4_SPEC_MAX) return -1;

    cudaStream_t stream = eng->stream;
    const int H   = GEMMA4_HIDDEN_SIZE;
    const int I   = GEMMA4_INTERMEDIATE;
    const int HD2 = 32 * sizeof(float);
    const int HEADS = GEMMA4_HEADS;
    const int pos = eng->global_n_tokens;            // captured; advanced only at end

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, stream);

    // Engine-resident scratch (allocated once, sized for GEMMA4_SPEC_MAX rows), so
    // repeated/probe calls pay no per-call cudaMalloc/free. All fp32: the batched
    // GEMV reads Q8_0 directly (no BF16 dequant), K tokens cost ~one token's weight BW.
    if (ensure_spec_scratch(eng) != 0) {
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        return -1;
    }
    int32_t *d_tok  = (int32_t*)eng->d_sb[0];
    float *d_x   = eng->d_sb[1],  *d_norm = eng->d_sb[2], *d_inf = eng->d_sb[3];
    float *d_q   = eng->d_sb[4],  *d_k    = eng->d_sb[5], *d_v   = eng->d_sb[6];
    float *d_attn= eng->d_sb[7],  *d_o    = eng->d_sb[8], *d_gate= eng->d_sb[9];
    float *d_up  = eng->d_sb[10], *d_logitsK = eng->d_sb[11];

    auto grid1d = [](size_t n){ return (unsigned)((n + 255) / 256); };

    cudaMemcpyAsync(d_tok, tokens, (size_t)K*sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    embed_w(eng, d_x, weight_fp8(eng, eng->tensors.token_embd), d_tok, K, H, stream);
    scale_kernel<<<grid1d((size_t)K*H),256,0,stream>>>(d_x, K*H, sqrtf((float)H));

    for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
        layer_type_t lt = eng->layer_types[l];
        int hd  = (lt==LAYER_SLIDING)? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM;
        int nkv = (lt==LAYER_SLIDING)? GEMMA4_KV_HEADS : GEMMA4_GLOBAL_KV_HEADS;
        int oq  = HEADS*hd, okv = nkv*hd;
        const float *w_attn   = eng->d_w_attn_norm      + (size_t)l*H;
        const float *w_post_a = eng->d_w_post_attn_norm + (size_t)l*H;
        const float *w_ffn    = eng->d_w_ffn_norm       + (size_t)l*H;
        const float *w_post_f = eng->d_w_post_ffn_norm  + (size_t)l*H;
        const float *w_qn     = eng->d_w_q_norm + (size_t)l*GEMMA4_GLOBAL_HEAD_DIM;
        const float *w_kn     = eng->d_w_k_norm + (size_t)l*GEMMA4_GLOBAL_HEAD_DIM;
        const __typeof__(eng->tensors.layers[0]) *L = &eng->tensors.layers[l];

        // Pre-attn norm (fp32) → Q,K,V batched GEMV (Q8_0 read once, reused over K).
        rms_norm_rows_kernel<<<K,256,HD2,stream>>>(d_inf, d_x, w_attn, H, K, GEMMA4_RMS_EPS);
        gemv_batched_w(eng, d_q, weight_fp8(eng, L->attn_q), d_inf, H, oq, K, stream);
        per_head_rms_norm_rows_kernel<<<dim3(HEADS,K),hd,HD2,stream>>>(d_q, w_qn, HEADS, hd, K, GEMMA4_RMS_EPS);
        gemv_batched_w(eng, d_k, weight_fp8(eng, L->attn_k), d_inf, H, okv, K, stream);
        if (lt == LAYER_SLIDING) {
            gemv_batched_w(eng, d_v, weight_fp8(eng, L->attn_v), d_inf, H, okv, K, stream);
            per_head_rms_norm_rows_kernel<<<dim3(nkv,K),hd,HD2,stream>>>(d_v, NULL, nkv, hd, K, GEMMA4_RMS_EPS);
        } else {
            cudaMemcpyAsync(d_v, d_k, (size_t)K*okv*sizeof(float), cudaMemcpyDeviceToDevice, stream);
            per_head_rms_norm_rows_kernel<<<dim3(nkv,K),hd,HD2,stream>>>(d_v, NULL, nkv, hd, K, GEMMA4_RMS_EPS);
        }
        per_head_rms_norm_rows_kernel<<<dim3(nkv,K),hd,HD2,stream>>>(d_k, w_kn, nkv, hd, K, GEMMA4_RMS_EPS);

        if (lt == LAYER_SLIDING)
            rope_rows_kernel<<<dim3(HEADS,K),hd/2,0,stream>>>(d_q, d_k, pos, HEADS, nkv, hd, K, 10000.0f, NULL);
        else
            rope_rows_kernel<<<dim3(HEADS,K),hd/2,0,stream>>>(d_q, d_k, pos, HEADS, nkv, hd, K, 1000000.0f, eng->d_rope_freqs);

        // Attention: per token, write its K/V into the live cache then attend against
        // everything up to it (exact causality; reuses the flash decode kernels).
        const int smemA = 32 * (int)sizeof(float);
        if (lt == LAYER_SLIDING) {
            size_t lstride = (size_t)GEMMA4_SLIDING_WINDOW * okv;
            float *kc = eng->d_sliding_k + (size_t)l*lstride;
            float *vc = eng->d_sliding_v + (size_t)l*lstride;
            for (int i = 0; i < K; i++) {
                int cur = eng->sliding_cursor[l];
                kv_write_sliding_kernel<<<dim3(grid1d(okv),1),256,0,stream>>>(
                    kc, vc, d_k + (size_t)i*okv, d_v + (size_t)i*okv,
                    cur, 0, 1, okv, GEMMA4_SLIDING_WINDOW);
                eng->sliding_cursor[l] = (cur + 1) % GEMMA4_SLIDING_WINDOW;
                if (eng->sliding_filled[l] < GEMMA4_SLIDING_WINDOW) eng->sliding_filled[l]++;
                sliding_attn_decode_kernel<<<HEADS, hd, smemA, stream>>>(
                    d_attn + (size_t)i*oq, d_q + (size_t)i*oq, kc, vc,
                    HEADS, nkv, hd, GEMMA4_SLIDING_WINDOW,
                    eng->sliding_cursor[l], eng->sliding_filled[l]);
            }
        } else {
            int slot = eng->global_slot[l];
            size_t lstride = (size_t)eng->global_kv_capacity * GEMMA4_GLOBAL_HEAD_DIM;
            float *kc = eng->d_global_k + (size_t)slot*lstride;
            float *vc = eng->d_global_v + (size_t)slot*lstride;
            for (int i = 0; i < K; i++) {
                int gpos = pos + i;
                kv_write_global_kernel<<<dim3(grid1d(hd),1),256,0,stream>>>(
                    kc, vc, d_k + (size_t)i*okv, d_v + (size_t)i*okv, gpos, 1, hd);
                global_attn_decode_kernel<<<HEADS, hd, smemA, stream>>>(
                    d_attn + (size_t)i*oq, d_q + (size_t)i*oq, kc, vc,
                    HEADS, hd, gpos + 1);
            }
        }

        // O projection (input d_attn is already fp32) → d_o; post-attn norm; residual.
        gemv_batched_w(eng, d_o, weight_fp8(eng, L->attn_output), d_attn, oq, H, K, stream);
        rms_norm_rows_kernel<<<K,256,HD2,stream>>>(d_norm, d_o, w_post_a, H, K, GEMMA4_RMS_EPS);
        residual_add_kernel<<<grid1d((size_t)K*H),256,0,stream>>>(d_x, d_norm, K*H);

        // FFN (fp32 batched GEMV throughout).
        rms_norm_rows_kernel<<<K,256,HD2,stream>>>(d_inf, d_x, w_ffn, H, K, GEMMA4_RMS_EPS);
        gemv_batched_w(eng, d_gate, weight_fp8(eng, L->ffn_gate), d_inf, H, I, K, stream);
        gemv_batched_w(eng, d_up,   weight_fp8(eng, L->ffn_up),   d_inf, H, I, K, stream);
        geglu_kernel<<<grid1d((size_t)K*I),256,0,stream>>>(d_inf, d_gate, d_up, K*I);
        gemv_batched_w(eng, d_o, weight_fp8(eng, L->ffn_down), d_inf, I, H, K, stream);
        rms_norm_rows_kernel<<<K,256,HD2,stream>>>(d_norm, d_o, w_post_f, H, K, GEMMA4_RMS_EPS);
        residual_add_kernel<<<grid1d((size_t)K*H),256,0,stream>>>(d_x, d_norm, K*H);
        if (eng->h_out_scale[l] != 1.0f)
            scale_kernel<<<grid1d((size_t)K*H),256,0,stream>>>(d_x, K*H, eng->h_out_scale[l]);
    }

    // Output norm (batched) + LM head as ONE batched GEMV (tied LM head read once
    // for all K) + softcap + suppress, then D2H all K logit rows.
    int vocab = GEMMA4_VOCAB_SIZE;
    rms_norm_rows_kernel<<<K,256,HD2,stream>>>(d_norm, d_x, eng->d_w_out_norm, H, K, GEMMA4_RMS_EPS);
    gemv_batched_w(eng, d_logitsK, weight_fp8(eng, eng->tensors.output_weight),
                   d_norm, H, vocab, K, stream);
    logit_softcap_kernel<<<grid1d((size_t)K*vocab),256,0,stream>>>(d_logitsK, GEMMA4_SOFTCAP, K*vocab);
    if (eng->n_suppress > 0)
        for (int i = 0; i < K; i++)
            suppress_tokens_kernel<<<grid1d(eng->n_suppress),256,0,stream>>>(
                d_logitsK + (size_t)i*vocab, eng->d_suppress, eng->n_suppress, vocab);
    if (logits_out)
        cudaMemcpyAsync(logits_out, d_logitsK, (size_t)K*vocab*sizeof(float),
                        cudaMemcpyDeviceToHost, stream);

    eng->global_n_tokens += K;

    cudaEventRecord(t1, stream);
    cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    eng->decode_time_ms += ms;
    eng->n_decode_tokens += K;
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gem4d: decode_batched CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
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
    (void)positions; // positions are implied by the cache cursor; reserved for batched verify

    // For now, verify draft tokens one by one. Each gemma4_engine_decode advances
    // BOTH the global token counter and every sliding layer's ring cursor/fill,
    // so a correct rollback on mismatch must restore all of them — otherwise the
    // rejected token's K/V stays in-window and corrupts the next real decode.
    float *host_logits = (float *)malloc(GEMMA4_VOCAB_SIZE * sizeof(float));
    if (!host_logits) return -1;

    int saved_cursor[GEMMA4_MAX_LAYERS];
    int saved_filled[GEMMA4_MAX_LAYERS];

    int n_accepted = 0;

    for (int k = 0; k < K; k++) {
        // Snapshot the full KV-cursor state before applying this candidate.
        int saved_n_tokens = eng->global_n_tokens;
        memcpy(saved_cursor, eng->sliding_cursor, sizeof(saved_cursor));
        memcpy(saved_filled, eng->sliding_filled, sizeof(saved_filled));

        // Decode candidate token k (mutates the KV cache + counters)
        gemma4_engine_decode(eng, draft_tokens[k], host_logits);

        // Check if greedy sample matches draft
        int greedy = gemma4_sample_argmax(host_logits, GEMMA4_VOCAB_SIZE);

        if (greedy == draft_tokens[k]) {
            n_accepted = k + 1;

            // Copy logits for accepted token (slot k per the header contract)
            if (logits_out) {
                memcpy(logits_out + (size_t)k * GEMMA4_VOCAB_SIZE,
                       host_logits, GEMMA4_VOCAB_SIZE * sizeof(float));
            }
        } else {
            // Mismatch: roll the cache back to before this token so the rejected
            // K/V is discarded from BOTH the global cursor and every sliding ring.
            eng->global_n_tokens = saved_n_tokens;
            memcpy(eng->sliding_cursor, saved_cursor, sizeof(saved_cursor));
            memcpy(eng->sliding_filled, saved_filled, sizeof(saved_filled));

            // Emit the greedy logits for this position (slot k, not 0).
            if (logits_out) {
                memcpy(logits_out + (size_t)k * GEMMA4_VOCAB_SIZE,
                       host_logits, GEMMA4_VOCAB_SIZE * sizeof(float));
            }
            free(host_logits);
            return n_accepted;
        }
    }

    free(host_logits);
    return n_accepted;  // All K accepted
}

// =========================================================================
// ─── Greedy Speculative Decode (prompt-lookup) ──────────────────────────
// =========================================================================

// Longest-suffix n-gram draft: find the most recent earlier occurrence of the
// current suffix of hist[0..n-1] (lengths max_ng..min_ng) and propose the up-to
// max_d tokens that followed it. Zero model cost. Returns the draft length.
static int prompt_lookup_draft(const int32_t *hist, int n, int32_t *draft,
                               int max_d, int min_ng, int max_ng)
{
    if (max_d <= 0) return 0;
    for (int ng = max_ng; ng >= min_ng; ng--) {
        if (n < ng + 1) continue;
        const int32_t *suf = hist + n - ng;
        for (int i = n - ng - 1; i >= 0; i--) {
            int match = 1;
            for (int j = 0; j < ng; j++) if (hist[i+j] != suf[j]) { match = 0; break; }
            if (!match) continue;
            int d = 0;
            for (int j = i + ng; j < n && d < max_d; j++) draft[d++] = hist[j];
            if (d > 0) return d;
        }
    }
    return 0;
}

// Greedy generation with prompt-lookup speculative decoding. Each step forwards
// [g, draft...] in ONE weight pass (gemma4_engine_decode_batched), commits the
// confirmed prefix, and gets the next step's logits for free from the same pass —
// so a matched draft of length a yields (1+a) tokens per ~one token's bandwidth.
// Greedy/argmax only (exact: accepted iff the model's argmax equals the draft).
// Draft length is capped to keep the partial-accept KV rollback inside the sliding
// window (full speedup for the first ~window tokens; plain decode beyond).
// Fills out_tokens (≤ max_new), returns the count generated. n_accepted_out (or
// NULL) receives the total drafts accepted (for measuring the acceptance rate).
int gemma4_engine_generate_spec(
    gemma4_engine_t *eng,
    const int32_t   *prompt, int n_prompt,
    int32_t         *out_tokens, int max_new,
    const int32_t   *stop_ids, int n_stop,
    int              draft_k,
    int             *n_accepted_out)
{
    if (!eng || !eng->loaded || n_prompt <= 0 || max_new <= 0) return -1;
    if (draft_k > GEMMA4_SPEC_MAX - 1) draft_k = GEMMA4_SPEC_MAX - 1;
    int V = GEMMA4_VOCAB_SIZE;

    int cap = n_prompt + max_new + 8;
    int32_t *hist = (int32_t*)malloc((size_t)cap*sizeof(int32_t));
    float   *logits = (float*)malloc((size_t)V*sizeof(float));
    float   *Lbuf = (float*)malloc((size_t)(draft_k+1)*V*sizeof(float));
    int32_t  batch[GEMMA4_SPEC_MAX];
    if (!hist || !logits || !Lbuf) { free(hist); free(logits); free(Lbuf); return -1; }

    auto is_stop = [&](int t){ for (int s=0;s<n_stop;s++) if (stop_ids[s]==t) return 1; return 0; };

    memcpy(hist, prompt, (size_t)n_prompt*sizeof(int32_t));
    int n = n_prompt;

    // Prefill the prompt → logits predicting the first generated token.
    if (gemma4_engine_prefill(eng, prompt, n_prompt, logits) != 0) {
        free(hist); free(logits); free(Lbuf); return -1;
    }

    // Adaptive speculation: track the accepted-fraction EMA and stop drafting when
    // it falls low (novel text where prompt-lookup mostly misses), re-probing every
    // PROBE steps so a workload that turns copy-heavy resumes speculating. This caps
    // the wrong-draft overhead so spec never meaningfully regresses vs plain greedy.
    const float SPEC_ON = 0.45f;   // need a healthy hit rate to keep drafting
    const int   PROBE   = 64;      // re-probe rarely so misses stay cheap
    float ema = 1.0f;          // optimistic start
    int   since_probe = 0;

    int generated = 0, total_accepted = 0, stop = 0;
    while (generated < max_new && !stop) {
        int g = gemma4_sample_argmax(logits, V);
        out_tokens[generated++] = g; hist[n++] = g;
        if (is_stop(g)) break;

        int pos = eng->global_n_tokens;                 // g not yet in the cache
        int room = GEMMA4_SLIDING_WINDOW - 1 - pos;      // keep rollback inside the window
        int want_spec = (ema > SPEC_ON) || (since_probe >= PROBE);
        int maxd = want_spec ? draft_k : 0;
        if (maxd > room) maxd = room; if (maxd < 0) maxd = 0;
        if (generated + 1 >= max_new) maxd = 0;          // no point drafting past the budget

        int D = prompt_lookup_draft(hist, n, batch+1, maxd, 2, draft_k);

        // No n-gram match → nothing to speculate. Use the fast single-token decode
        // (avoids decode_batched's per-call allocation/sync overhead), so spec is
        // never slower than plain greedy on novel text — only the matched-but-wrong
        // drafts cost anything, and a matched-and-right draft is the win.
        if (D == 0) {
            if (gemma4_engine_decode(eng, g, logits) != 0) { stop = 1; break; }
            since_probe++;
            continue;
        }

        batch[0] = g;
        int K = D + 1;
        if (gemma4_engine_decode_batched(eng, batch, K, Lbuf) != 0) {
            if (gemma4_engine_decode(eng, g, logits) != 0) { stop = 1; break; }
            continue;
        }

        // Accept the longest draft prefix whose predecessor's argmax matches it.
        int a = 0;
        while (a < D && gemma4_sample_argmax(Lbuf + (size_t)a*V, V) == batch[1+a]) a++;
        total_accepted += a;
        ema = 0.6f*ema + 0.4f*((float)a/(float)D);    // update acceptance estimate
        since_probe = 0;

        // Discard the rejected drafts from the KV cache (keep g + a accepted).
        int keep = pos + 1 + a;
        if (keep < eng->global_n_tokens) gemma4_engine_rewind(eng, keep);

        for (int i = 0; i < a && generated < max_new; i++) {
            int t = batch[1+i]; out_tokens[generated++] = t; hist[n++] = t;
            if (is_stop(t)) { stop = 1; break; }
        }
        memcpy(logits, Lbuf + (size_t)a*V, (size_t)V*sizeof(float));  // next-step logits, free
    }

    if (n_accepted_out) *n_accepted_out = total_accepted;
    free(hist); free(logits); free(Lbuf);
    return generated;
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

// Serialized session layout (one shared layout used by save AND load):
//   [u32 global_n_tokens][u32 n_layers_global][u32 global_kv_capacity]
//   [int sliding_cursor[GEMMA4_MAX_LAYERS]]
//   [int sliding_filled[GEMMA4_MAX_LAYERS]]
//   [float sliding_k full buffer][float sliding_v full buffer]
//   [float global_k: n_layers_global × n_tokens × head_dim]   (used rows only)
//   [float global_v: n_layers_global × n_tokens × head_dim]
// The global section stores only the used rows of each global slot, packed
// without the per-slot capacity gap, so it is independent of how either engine
// sized its cache — load validates the saved n_tokens fits this engine.
static uint64_t session_sliding_bytes(void) {
    return (uint64_t)GEMMA4_MAX_LAYERS *
        GEMMA4_KV_HEADS * GEMMA4_SLIDING_WINDOW * GEMMA4_HEAD_DIM * sizeof(float);
}

int gemma4_engine_save_session(
    gemma4_engine_t *eng,
    uint8_t         *buffer,
    uint64_t        *size)
{
    if (!eng || !size) return -1;

    uint64_t sliding_bytes = session_sliding_bytes();
    uint64_t cursor_bytes  = 2 * (uint64_t)GEMMA4_MAX_LAYERS * sizeof(int);
    uint64_t global_rows   = (uint64_t)eng->n_layers_global * eng->global_n_tokens;
    uint64_t global_bytes  = global_rows * GEMMA4_GLOBAL_HEAD_DIM * sizeof(float);

    uint64_t needed = sizeof(uint32_t) * 3 + cursor_bytes
        + 2 * sliding_bytes + 2 * global_bytes;

    if (!buffer) { *size = needed; return 0; }
    if (*size < needed) return -1;

    uint8_t *p = buffer;
    *(uint32_t *)p = (uint32_t)eng->global_n_tokens;    p += 4;
    *(uint32_t *)p = (uint32_t)eng->n_layers_global;    p += 4;
    *(uint32_t *)p = (uint32_t)eng->global_kv_capacity; p += 4;

    memcpy(p, eng->sliding_cursor, GEMMA4_MAX_LAYERS * sizeof(int));
    p += GEMMA4_MAX_LAYERS * sizeof(int);
    memcpy(p, eng->sliding_filled, GEMMA4_MAX_LAYERS * sizeof(int));
    p += GEMMA4_MAX_LAYERS * sizeof(int);

    cudaMemcpy(p, eng->d_sliding_k, sliding_bytes, cudaMemcpyDeviceToHost);
    p += sliding_bytes;
    cudaMemcpy(p, eng->d_sliding_v, sliding_bytes, cudaMemcpyDeviceToHost);
    p += sliding_bytes;

    // Global K/V: copy only the first n_tokens rows of each global slot, packed.
    uint64_t used_per_slot = (uint64_t)eng->global_n_tokens
        * GEMMA4_GLOBAL_HEAD_DIM * sizeof(float);
    size_t slot_stride = (size_t)eng->global_kv_capacity * GEMMA4_GLOBAL_HEAD_DIM;
    for (int g = 0; g < eng->n_layers_global; g++) {
        cudaMemcpy(p, eng->d_global_k + (size_t)g * slot_stride,
                   used_per_slot, cudaMemcpyDeviceToHost);
        p += used_per_slot;
    }
    for (int g = 0; g < eng->n_layers_global; g++) {
        cudaMemcpy(p, eng->d_global_v + (size_t)g * slot_stride,
                   used_per_slot, cudaMemcpyDeviceToHost);
        p += used_per_slot;
    }

    *size = needed;
    return 0;
}

int gemma4_engine_load_session(
    gemma4_engine_t *eng,
    const uint8_t   *buffer,
    uint64_t         size)
{
    if (!eng || !buffer) return -1;

    uint64_t sliding_bytes = session_sliding_bytes();
    uint64_t min_header = sizeof(uint32_t) * 3
        + 2 * (uint64_t)GEMMA4_MAX_LAYERS * sizeof(int) + 2 * sliding_bytes;
    if (size < min_header) return -1;

    const uint8_t *p = buffer;
    int n_tokens   = (int)*(const uint32_t *)p; p += 4;
    int n_global   = (int)*(const uint32_t *)p; p += 4;
    p += 4; // saved global_kv_capacity — informational; this engine keeps its own

    // Validate against THIS engine — do not blindly overwrite capacity (no
    // realloc happens, so a larger saved cache would overflow the device buffer).
    if (n_global != eng->n_layers_global) return -1;
    if (n_tokens < 0 || n_tokens > eng->global_kv_capacity) return -1;

    uint64_t global_rows  = (uint64_t)n_global * n_tokens;
    uint64_t global_bytes = global_rows * GEMMA4_GLOBAL_HEAD_DIM * sizeof(float);
    uint64_t needed = min_header + 2 * global_bytes;
    if (size < needed) return -1;

    eng->global_n_tokens = n_tokens;

    memcpy(eng->sliding_cursor, p, GEMMA4_MAX_LAYERS * sizeof(int));
    p += GEMMA4_MAX_LAYERS * sizeof(int);
    memcpy(eng->sliding_filled, p, GEMMA4_MAX_LAYERS * sizeof(int));
    p += GEMMA4_MAX_LAYERS * sizeof(int);

    cudaMemcpy(eng->d_sliding_k, p, sliding_bytes, cudaMemcpyHostToDevice);
    p += sliding_bytes;
    cudaMemcpy(eng->d_sliding_v, p, sliding_bytes, cudaMemcpyHostToDevice);
    p += sliding_bytes;

    uint64_t used_per_slot = (uint64_t)n_tokens
        * GEMMA4_GLOBAL_HEAD_DIM * sizeof(float);
    size_t slot_stride = (size_t)eng->global_kv_capacity * GEMMA4_GLOBAL_HEAD_DIM;
    for (int g = 0; g < n_global; g++) {
        cudaMemcpy(eng->d_global_k + (size_t)g * slot_stride, p,
                   used_per_slot, cudaMemcpyHostToDevice);
        p += used_per_slot;
    }
    for (int g = 0; g < n_global; g++) {
        cudaMemcpy(eng->d_global_v + (size_t)g * slot_stride, p,
                   used_per_slot, cudaMemcpyHostToDevice);
        p += used_per_slot;
    }

    return 0;
}
