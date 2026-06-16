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
#include "paged_kv_device.cuh" // paged KV: host block-table bookkeeping + device access kernels
                              // (Phase 2 continuous batching). Pulls paged_kv.h. Kernels are
                              // compiled into the engine TU but stay dormant until wired.
#include <cuda_fp4.h>         // __nv_fp4_storage_t, NVFP4 E2M1 conversion (FUCINA_FP4)
#include <cublasLt.h>         // NVFP4 block-scaled tensor-core GEMM (FUCINA_FP4)
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
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q8_0 = 8,
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
    if (i < n) dst[i] = float_to_fp8(src[i]);
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
    if (i < n) base[(size_t)((*pos_ptr) % cap) * stride + i] = float_to_fp8(src[i]);
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
    #define LAUNCH(NK)                                                                  \
        do { if (fmt == 2)                                                              \
            mmvq_q4_0_batched_kernel<NK><<<g,b,0,stream>>>(out,weight,qx,dx,sx,in_dim,out_dim); \
        else                                                                           \
            mmvq_q8_0_batched_kernel<NK><<<g,b,0,stream>>>(out,weight,qx,dx,in_dim,out_dim); \
        } while (0)
    switch (K) {
        case 1: LAUNCH(1); break;  case 2: LAUNCH(2); break;
        case 3: LAUNCH(3); break;  case 4: LAUNCH(4); break;
        case 5: LAUNCH(5); break;  case 6: LAUNCH(6); break;
        case 7: LAUNCH(7); break;  case 8: LAUNCH(8); break;
        default:
            for (int o = 0; o < K; o += 8) {
                int kk = (K - o < 8) ? (K - o) : 8;
                mmvq_batched_launch(out + (size_t)o*out_dim, weight,
                                    qx + (size_t)o*in_dim, dx + (size_t)o*(in_dim>>5),
                                    sx + (size_t)o*(in_dim>>5),
                                    in_dim, out_dim, kk, fmt, stream);
            }
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

template<int NH, int HD>
__global__ void global_attn_splitk_kernel(
    float *part_acc,                          // [n_splits][NH][HD] (unnormalized)
    float *part_m, float *part_l,             // [n_splits][NH]
    const float *q,                           // [NH][HD]
    const kv_t *k_cache, const kv_t *v_cache, // [ctx_len][HD]
    int head_dim, int ctx_len, int n_splits)
{
    constexpr int TILE = GEMMA4_GLOBAL_ATTN_TILE;
    constexpr int E = HD / 128;               // uint words per lane (4 at HD 512)
    static_assert(HD % 128 == 0, "uint lane slices require HD multiple of 128");
    __shared__ unsigned char sk[TILE * HD], sv[TILE * HD];
    int h    = threadIdx.x >> 5;              // warp = query head
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
        {   // cooperative stage: tn rows of HD fp8 bytes for K and V (16 B vectors; KV rows
            // are HD-byte aligned so the uint4 view is safe). Bounds are block-uniform.
            int nvec = tn * (HD / 16);
            const uint4 *gk = (const uint4 *)(k_cache + (size_t)tb * HD);
            const uint4 *gv = (const uint4 *)(v_cache + (size_t)tb * HD);
            uint4 *sk4 = (uint4 *)sk, *sv4 = (uint4 *)sv;
            for (int i = threadIdx.x; i < nvec; i += NH*32) { sk4[i] = gk[i]; sv4[i] = gv[i]; }
        }
        __syncthreads();
        for (int tt = 0; tt < tn; tt++) {
            const unsigned int *kw = (const unsigned int *)(sk + (size_t)tt * HD);
            const unsigned int *vw = (const unsigned int *)(sv + (size_t)tt * HD);
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
template<int NH, int HD>
__global__ void global_attn_splitk_rows_kernel(
    float *part_acc, float *part_m, float *part_l,   // slot r*MAX_SPLITS+split
    const float *q,                                  // [K][NH*HD] (row-major)
    const kv_t *k_cache, const kv_t *v_cache,        // [capacity][HD]
    int n_tokens0,                                   // row r attends n_tokens0 + r keys
    const int *n_tokens0_ptr)                        // non-NULL: device override (graph path)
{
    constexpr int TILE = GEMMA4_GLOBAL_ATTN_TILE;
    constexpr int E = HD / 128;
    static_assert(HD % 128 == 0, "uint lane slices require HD multiple of 128");
    __shared__ unsigned char sk[TILE * HD], sv[TILE * HD];
    int r = blockIdx.y;
    if (n_tokens0_ptr) n_tokens0 = *n_tokens0_ptr;
    int ctx_len = n_tokens0 + r;
    int n_splits = attn_row_splits(ctx_len, GEMMA4_GLOBAL_SPLIT_CHUNK);
    int split = blockIdx.x;
    if (split >= n_splits) return;                   // tail blocks of shorter rows
    int h    = threadIdx.x >> 5;
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
            int nvec = tn * (HD / 16);
            const uint4 *gk = (const uint4 *)(k_cache + (size_t)tb * HD);
            const uint4 *gv = (const uint4 *)(v_cache + (size_t)tb * HD);
            uint4 *sk4 = (uint4 *)sk, *sv4 = (uint4 *)sv;
            for (int i = threadIdx.x; i < nvec; i += NH*32) { sk4[i] = gk[i]; sv4[i] = gv[i]; }
        }
        __syncthreads();
        for (int tt = 0; tt < tn; tt++) {
            const unsigned int *kw = (const unsigned int *)(sk + (size_t)tt * HD);
            const unsigned int *vw = (const unsigned int *)(sv + (size_t)tt * HD);
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
    const int *n_tokens0_ptr)                        // non-NULL: device override (graph path)
{
    constexpr int GQ = NH / NKV;
    constexpr int slice = HD / 32;
    int r = blockIdx.y;
    if (n_tokens0_ptr) n_tokens0 = *n_tokens0_ptr;
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
        size_t pos = (size_t)(lo + i) % (size_t)cap;   // ring slot for absolute pos lo+i
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
    int head_dim, int rows, float theta_base, const float *freq_factors,
    const int *base_pos_ptr = nullptr)               // non-NULL: device override (graph path)
{
    int d = threadIdx.x, half = head_dim / 2;
    if (d >= half) return;
    int head = blockIdx.x, row = blockIdx.y;
    if (row >= rows) return;
    if (base_pos_ptr) base_pos = *base_pos_ptr;
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
    kcache[slot * kvhd + j] = float_to_fp8(kb[(size_t)t * kvhd + j]);
    vcache[slot * kvhd + j] = float_to_fp8(vb[(size_t)t * kvhd + j]);
}

// Scatter the batch's K/V into the linear global cache at positions base..base+rows-1.
// k/vcache point at the layer slot's base [capacity][hd]. grid=(ceil(hd/256), rows).
__global__ void kv_write_global_kernel(
    kv_t *kcache, kv_t *vcache, const float *kb, const float *vb,
    int base, int rows, int hd,
    const int *base_ptr = nullptr)                     // non-NULL: device override (graph path)
{
    int t = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows || j >= hd) return;
    if (base_ptr) base = *base_ptr;
    int pos = base + t;
    kcache[(size_t)pos * hd + j] = float_to_fp8(kb[(size_t)t * hd + j]);
    vcache[(size_t)pos * hd + j] = float_to_fp8(vb[(size_t)t * hd + j]);
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
} gemma4_seq;

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
    cudaStream_t   dq_stream;           // weight-dequant stream (overlaps GEMMs)
    cudaEvent_t    ev_dq_done[2];       // dequant of buffer b complete
    cudaEvent_t    ev_gemm_done[2];     // last GEMM reading buffer b complete

    // ── Persistent prefill scratch (CUDA-graph safe) ─────────────────
    int    pf_scratch_ready;
    float  *d_pf_x, *d_pf_norm, *d_pf_q, *d_pf_k, *d_pf_v;
    float  *d_pf_attn, *d_pf_gate, *d_pf_up, *d_pf_scores;
    __nv_bfloat16 *d_pf_inb, *d_pf_qb, *d_pf_kb, *d_pf_vb;
    __nv_bfloat16 *d_pf_kbx, *d_pf_vbx, *d_pf_pb;

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
    } mtp;
    int     mtp_h_valid;       // d_mtp_h holds the h paired with the pending token g
    float  *d_mtp_h;           // [3840] target post-output-norm hidden (recurrent h)
    float  *d_mtp_xh;          // [7680] concat(embed(tok)·√3840, h)
    float  *d_mtp_cur;         // [1024] residual stream
    float  *d_mtp_t1, *d_mtp_t2;  // [1024] norm/proj scratch
    float  *d_mtp_q;           // [16×512] worst-case Q
    float  *d_mtp_attn;        // [16×512] attention output
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
    uint8_t *d_fp4_w[GEMMA4_MAX_LAYERS][PJ_COUNT];   // packed [out_dim × in_dim/2]
    uint8_t *d_fp4_wsc[GEMMA4_MAX_LAYERS][PJ_COUNT]; // swizzled E4M3 [pad(out,128)×pad(in/16,4)]
    float   *d_fp4_gsw;        // device [MAX_LAYERS*PJ_COUNT] weight per-tensor global scales
    int      fp4_ready;        // 1 once persistent NVFP4 weights are built
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
    float  h_out_scale[GEMMA4_MAX_LAYERS]; // layer_output_scale scalars (host)

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
    PagedBlockPool slid_pool;           // sliding-class block free-list (host)
    PagedBlockPool glob_pool;           // global-class  block free-list (host)
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

    // ─── LoRA adapter support ───────────────────────────────────────────
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
static void gemma4_engine_paged_selftest(gemma4_engine_t *eng);  // Phase 2 inc 3 (body below)

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
        fprintf(stderr, "fucina: context size %u exceeds max %u\n",
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
    eng->decode_graph = NULL; eng->decode_graph_failed = 0;
    eng->d_decpos = NULL; eng->d_dectok = NULL;
    for (int k = 0; k <= GEMMA4_SPEC_MAX; k++) eng->batched_graph[k] = NULL;
    eng->batched_graph_failed = 0; eng->d_specpos = NULL;
    eng->spec_steps = eng->spec_drafted = eng->spec_accepted = eng->spec_emitted = 0;
    eng->pf_scratch_ready = 0;
    eng->d_pf_x = eng->d_pf_norm = eng->d_pf_q = eng->d_pf_k = eng->d_pf_v = NULL;
    eng->d_pf_attn = eng->d_pf_gate = eng->d_pf_up = eng->d_pf_scores = NULL;
    eng->d_pf_inb = eng->d_pf_qb = eng->d_pf_kb = eng->d_pf_vb = NULL;
    eng->d_pf_kbx = eng->d_pf_vbx = eng->d_pf_pb = NULL;

    // Open and mmap the GGUF file
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
        gemma4_engine_destroy(eng);
        return NULL;
    }

    fprintf(stderr, "fucina: loaded %s (%.2f GB)\n",
            model_path, eng->gguf_size / (1024.0 * 1024.0 * 1024.0));

    // Auto-detect the weight format from the GGUF tensor table. Trusting the CLI flag
    // is dangerous (decoding Q8_0 blocks as FP8 bytes yields NaNs). Detect from a LAYER
    // tensor (ffn_down) — token_embd may be a different type (Q6_K in the QAT Q4_0 model).
    int embd_is_q6k = 0;
    {
        uint64_t _off = 0, _n = 0; uint32_t gtype = 0;
        const char *names[] = {"blk.0.ffn_down.weight", "blk.0.attn_q.weight"};
        for (int t = 0; t < 2; t++) {
            if (gguf_find_tensor(eng->gguf_data, eng->gguf_size, names[t], &_off, &_n, &gtype) == 0) {
                if (gtype != GGML_TYPE_Q4_0 && gtype != GGML_TYPE_Q8_0) {
                    fprintf(stderr, "fucina: unsupported GGUF layer tensor type %u — "
                            "only Q4_0 (QAT) and Q8_0 models are supported\n", gtype);
                    gemma4_engine_destroy(eng);
                    return NULL;
                }
                eng->format = (gtype == GGML_TYPE_Q4_0) ? FORMAT_Q4_0 : FORMAT_Q8_0;
                break;
            }
        }
        // token_embd / tied LM head type (Q6_K in the QAT model → convert to Q8_0 at load)
        uint32_t etype = 0;
        if (gguf_find_tensor(eng->gguf_data, eng->gguf_size,
                "token_embd.weight", &_off, &_n, &etype) == 0 && etype == GGML_TYPE_Q6_K) {
            embd_is_q6k = 1;
            fprintf(stderr, "fucina: token_embd is Q6_K — will convert to Q8_0 at load\n");
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
        fprintf(stderr, "fucina: warning: no attention pattern in GGUF, "
                        "using default 5-sliding/1-global cadence\n");
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

    // Load embedding (required — also the tied LM head)
    LOAD_TENSOR_REQUIRED("token_embd.weight", token_embd);

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

    LOAD_TENSOR_REQUIRED("output_norm.weight", output_norm);

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
            fprintf(stderr, "fucina: output.weight not present — using tied "
                            "token_embd.weight as LM head\n");
        }
    }

    #undef LOAD_TENSOR_OFFSET
    #undef LOAD_TENSOR_REQUIRED

    // A model-defining tensor was missing: the GGUF is corrupt or not a gemma-4
    // model. Fail the load rather than run inference against offset-0 garbage.
    // This is gemma4_engine_create, which returns the engine pointer — free it
    // and return NULL so the Go side sees a clean load failure.
    if (missing_required) {
        fprintf(stderr, "fucina: aborting load — required tensor(s) missing\n");
        free(eng);
        return NULL;
    }

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
    // GQA-broadcast global flash-decode split-K scratch (DECODE-30-35 Step 1).
    // Sized for GEMMA4_SPEC_MAX rows of GEMMA4_GLOBAL_MAX_SPLITS slots each (~64 MB):
    // the batched spec-verify attention runs all K rows' splits in ONE launch, row r
    // owning slot range [r*MAX_SPLITS, r*MAX_SPLITS + splits_r). Single-token paths
    // (decode_layer, MTP) keep using row-0's slot range unchanged.
    eng->d_fa_acc = NULL; eng->d_fa_m = NULL; eng->d_fa_l = NULL;
    cudaMalloc(&eng->d_fa_acc, (size_t)GEMMA4_SPEC_MAX * GEMMA4_GLOBAL_MAX_SPLITS
                                * GEMMA4_HEADS * GEMMA4_GLOBAL_HEAD_DIM * sizeof(float));
    cudaMalloc(&eng->d_fa_m,   (size_t)GEMMA4_SPEC_MAX * GEMMA4_GLOBAL_MAX_SPLITS
                                * GEMMA4_HEADS * sizeof(float));
    cudaMalloc(&eng->d_fa_l,   (size_t)GEMMA4_SPEC_MAX * GEMMA4_GLOBAL_MAX_SPLITS
                                * GEMMA4_HEADS * sizeof(float));
    cudaMalloc(&eng->d_ffn_out,    GEMMA4_INTERMEDIATE * sizeof(float));
    cudaMalloc(&eng->d_ffn_gate,   GEMMA4_INTERMEDIATE * sizeof(float));
    cudaMalloc(&eng->d_ffn_up,     GEMMA4_INTERMEDIATE * sizeof(float));
    cudaMalloc(&eng->d_norm,       GEMMA4_HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&eng->d_norm_w,     GEMMA4_HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&eng->d_residual,   GEMMA4_HIDDEN_SIZE * sizeof(float));

    cudaMalloc(&eng->d_head_norm_w, GEMMA4_GLOBAL_HEAD_DIM * sizeof(float));

    // dp4a MMVQ activation-quant scratch (widest in_dim = intermediate).
    eng->d_qx = NULL; eng->d_dx = NULL; eng->d_sx = NULL;
    cudaMalloc(&eng->d_qx, (size_t)GEMMA4_INTERMEDIATE * sizeof(int8_t));
    cudaMalloc(&eng->d_dx, (size_t)(GEMMA4_INTERMEDIATE/32) * sizeof(float));
    cudaMalloc(&eng->d_sx, (size_t)(GEMMA4_INTERMEDIATE/32) * sizeof(int));   // Q4_0 −8 fold
    // Batched (speculative-decode) variant: NK ≤ SPEC_MAX activation vectors at once.
    eng->d_qx_b = NULL; eng->d_dx_b = NULL; eng->d_sx_b = NULL;
    cudaMalloc(&eng->d_qx_b, (size_t)GEMMA4_SPEC_MAX * GEMMA4_INTERMEDIATE * sizeof(int8_t));
    cudaMalloc(&eng->d_dx_b, (size_t)GEMMA4_SPEC_MAX * (GEMMA4_INTERMEDIATE/32) * sizeof(float));
    cudaMalloc(&eng->d_sx_b, (size_t)GEMMA4_SPEC_MAX * (GEMMA4_INTERMEDIATE/32) * sizeof(int));
    if (!eng->d_qx || !eng->d_dx || !eng->d_sx || !eng->d_qx_b || !eng->d_dx_b || !eng->d_sx_b) {
        fprintf(stderr, "fucina: dp4a activation scratch alloc failed\n");
        gemma4_engine_destroy(eng);
        return NULL;
    }
    eng->d_pf_qx = NULL; eng->d_pf_dx = NULL; eng->d_pf_sx = NULL; eng->mmq_ready = 0;
    // NVFP4 prefill (FUCINA_FP4) — all lazy, built on first prefill if enabled
    eng->cublaslt = NULL; eng->fp4_ready = 0; eng->fp4_desc = NULL;
    eng->d_fp4_gsw = NULL; eng->d_fp4_act = NULL; eng->d_fp4_actsc = NULL;
    eng->d_fp4_actlin = NULL; eng->d_fp4_gsact = NULL; eng->d_fp4_alpha = NULL;
    eng->d_fp4_amax = NULL; eng->fp4_act_cap = 0; eng->d_fp4_ws = NULL;
    for (int l=0;l<GEMMA4_MAX_LAYERS;l++) for (int p=0;p<PJ_COUNT;p++){
        eng->d_fp4_w[l][p]=NULL; eng->d_fp4_wsc[l][p]=NULL; }

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
            fprintf(stderr, "fucina: %d suppressed tokens masked\n", eng->n_suppress);
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

    // QAT Q4_0 model: convert the Q6_K token_embd (= tied LM head) to a Q8_0 device
    // buffer so the existing Q8_0 embed/LM-head kernels handle it (layers stay Q4_0).
    eng->d_token_embd = NULL;
    if (eng->format == FORMAT_Q4_0 && embd_is_q6k) {
        int64_t n_elem = (int64_t)GEMMA4_VOCAB_SIZE * GEMMA4_HIDDEN_SIZE;
        const unsigned char *q6 = (const unsigned char*)(eng->gguf_data + eng->tensors.token_embd);
        unsigned char *q8 = convert_q6k_to_q8_0(q6, n_elem);
        if (q8) {
            size_t q8bytes = (size_t)(n_elem/32) * 34;
            if (cudaMalloc(&eng->d_token_embd, q8bytes) == cudaSuccess)
                cudaMemcpy(eng->d_token_embd, q8, q8bytes, cudaMemcpyHostToDevice);
            free(q8);
            fprintf(stderr, "fucina: token_embd Q6_K->Q8_0 converted (%.2f GB)\n",
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
    eng->d_lmhead_q6k = NULL; eng->lmhead_q6k = 0;
    if (eng->format == FORMAT_Q4_0 && embd_is_q6k && eng->output_tied &&
        eng->d_weights && eng->d_token_embd) {
        eng->d_lmhead_q6k = (unsigned char *)(eng->d_weights
                            + (eng->tensors.token_embd - eng->tdata_start));
        eng->lmhead_q6k = 1;
        fprintf(stderr, "fucina: LM head native Q6_K dp4a (0.83 GB/token vs 1.07 Q8_0)\n");
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
    // Sliding cache is a per-position RING: [MAX_LAYERS][cap][8×256], position p stored at
    // slot p % cap. Sliding-window attention only ever reads the last GEMMA4_SLIDING_WINDOW
    // positions, so cap need not equal context_size — capping it makes the sliding cache
    // (the dominant, ctx-scaling allocation) nearly context-independent: ~768 MB at cap=8192
    // vs ~21 GB flat @131k. cap = min(context_size, FUCINA_SLIDING_RING) (default 8192,
    // floored at the window). The margin cap-window bounds how far a prefix-reuse rewind can
    // look back exactly; deeper rewinds report failure (gemma4_engine_rewind) and the server
    // falls back to a full re-prefill. Speculation rewinds ≤ GEMMA4_SPEC_MAX, always inside it.
    {
        int ring_w = 8192;                          // default ring capacity (window + margin)
        const char *e = getenv("FUCINA_SLIDING_RING");
        if (e && *e) { int v = atoi(e); if (v >= GEMMA4_SLIDING_WINDOW) ring_w = v; }
        // Floor at window + spec_max: a spec-verify batch writes K≤SPEC_MAX draft
        // positions while each row reads its window, so window+SPEC_MAX consecutive
        // ring slots must be collision-free. (Only binds for a tiny FUCINA_SLIDING_RING;
        // the default 8192 clears it by ~7×.)
        const int ring_floor = GEMMA4_SLIDING_WINDOW + GEMMA4_SPEC_MAX;
        int cap = (int)context_size;
        if (cap > ring_w)      cap = ring_w;
        if (cap < ring_floor)  cap = ring_floor;
        eng->sliding_kv_capacity = cap;
    }
    size_t sliding_kv_size = (size_t)GEMMA4_MAX_LAYERS *
        GEMMA4_KV_HEADS * (size_t)eng->sliding_kv_capacity * GEMMA4_HEAD_DIM * sizeof(kv_t);
    if (cudaMalloc(&eng->d_sliding_k, sliding_kv_size) != cudaSuccess ||
        cudaMalloc(&eng->d_sliding_v, sliding_kv_size) != cudaSuccess) {
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
    size_t global_kv_size = (size_t)eng->n_layers_global *
        context_size * GEMMA4_GLOBAL_HEAD_DIM * sizeof(kv_t);
    if (cudaMalloc(&eng->d_global_k, global_kv_size) != cudaSuccess ||
        cudaMalloc(&eng->d_global_v, global_kv_size) != cudaSuccess) {
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
            eng->format == FORMAT_Q4_0 ? "Q4_0" : "Q8_0");
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
    if (getenv("FUCINA_PAGED_KV")) {
        const int BT = PAGED_KV_BLOCK_TOKENS;
        const int slid_elems = GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM;          // 8×256 = 2048
        const int glob_elems = GEMMA4_GLOBAL_KV_HEADS * GEMMA4_GLOBAL_HEAD_DIM; // 1×512 = 512
        // Per-block bytes for K (== V) across all layers of the class.
        size_t slid_block_k = (size_t)GEMMA4_MAX_LAYERS    * BT * slid_elems * sizeof(kv_t);
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

    // Repacked-Q4_0 decode GEMV: DEFAULT-ON for dense 12B+MTP (bit-exact, ~+2-3% decode
    // via coalesced uint4 weight loads; costs +6.96 GB weight VRAM — fine on GB10's 128 GB
    // unified memory). Built eagerly now — before any request or CUDA-graph capture — so
    // the decode GEMV's packed branch is a pure read. Opt out with FUCINA_NO_PACKED=1.
    {
        const char *off = getenv("FUCINA_NO_PACKED");
        if (!(off && off[0] == '1')) build_packed_q4(eng);   // non-fatal: falls back if it fails
    }

    // Paged KV mirror validation (opt-in): decode a fixed run and assert the
    // paged pool matches the contiguous cache byte-for-byte. Non-fatal.
    if (getenv("FUCINA_PAGED_KV_SELFTEST")) gemma4_engine_paged_selftest(eng);

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
    CUDA_FREE(eng->d_fa_acc);
    CUDA_FREE(eng->d_fa_m);
    CUDA_FREE(eng->d_fa_l);
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
    // Paged KV pools + the active sequence's block tables (no-ops when paged off).
    CUDA_FREE(eng->d_slid_pool_k);
    CUDA_FREE(eng->d_slid_pool_v);
    CUDA_FREE(eng->d_glob_pool_k);
    CUDA_FREE(eng->d_glob_pool_v);
    paged_pool_destroy(&eng->slid_pool);
    paged_pool_destroy(&eng->glob_pool);
    CUDA_FREE(eng->cur.d_slid_blocks);
    CUDA_FREE(eng->cur.d_glob_blocks);
    paged_table_free_struct(&eng->cur.slid_bt);
    paged_table_free_struct(&eng->cur.glob_bt);
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
    for (int b = 0; b < 2; b++)
        for (int p = 0; p < 7; p++)
            CUDA_FREE(eng->d_bf16_layer[b][p]);
    for (int p = 0; p < 12; p++)
        CUDA_FREE(eng->d_sb[p]);
    for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
    }
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
    if (eng->mtp_graph) { cudaGraphExecDestroy(eng->mtp_graph); eng->mtp_graph = NULL; }
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
        for (int l=0;l<GEMMA4_MAX_LAYERS;l++) for (int p=0;p<PJ_COUNT;p++){
            if (eng->d_fp4_w[l][p]) cudaFree(eng->d_fp4_w[l][p]);
            if (eng->d_fp4_wsc[l][p]) cudaFree(eng->d_fp4_wsc[l][p]); }
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
    cudaFree(eng->d_pf_x); cudaFree(eng->d_pf_norm);
    cudaFree(eng->d_pf_q); cudaFree(eng->d_pf_k); cudaFree(eng->d_pf_v);
    cudaFree(eng->d_pf_attn); cudaFree(eng->d_pf_gate); cudaFree(eng->d_pf_up);
    cudaFree(eng->d_pf_scores);
    cudaFree(eng->d_pf_inb); cudaFree(eng->d_pf_qb);
    cudaFree(eng->d_pf_kb); cudaFree(eng->d_pf_vb);
    cudaFree(eng->d_pf_kbx); cudaFree(eng->d_pf_vbx); cudaFree(eng->d_pf_pb);
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
    return eng->packed_ready && fmt == FORMAT_Q4_0 &&
           eng->d_weights && weight >= eng->d_weights;
}

// Q4_0-aware single-token MMVQ over an ALREADY-quantized activation (qx/dx/sx, K=1 layout):
// routes to the packed kernel when FUCINA_PACKED is active, else the native dp4a path. Used by
// the call sites that pre-quantize once and call mmvq_launch directly (decode_layer q/k/v).
static inline void mmvq_q4aware(
    const gemma4_engine_t *eng, float *out, const uint8_t *weight,
    const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim, int fmt, cudaStream_t stream)
{
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
static inline void gemv_w(
    const gemma4_engine_t *eng,
    float *out, const uint8_t *weight, const float *x,
    int in_dim, int out_dim, cudaStream_t stream, int wfmt = -1)
{
    int fmt = (wfmt < 0) ? FMT(eng) : wfmt;
    // Quantize the activation to int8 (+ per-block Σ for the Q4_0 −8 fold), then
    // warp-per-row dp4a MMVQ (llama.cpp-parity bandwidth, no block sync). Step 5.
    // The native Q6_K LM head rides the same int8 activation (its −32 is folded into
    // the weight values, so it needs only qx/dx).
    quantize_q8_1_kernel<<<in_dim/32, 32, 0, stream>>>(
        x, eng->d_qx, eng->d_dx, eng->d_sx, in_dim);
    if (fmt == FORMAT_Q6_K) {
        mmvq_q6_k_launch(out, weight, eng->d_qx, eng->d_dx, in_dim, out_dim, stream);
    } else if (use_packed_q4(eng, fmt, weight)) {
        const uint8_t  *q = eng->d_weights_packed + (weight - eng->d_weights);
        const uint16_t *s = (const uint16_t *)(q + (size_t)out_dim * (in_dim >> 5) * 16);
        mmvq_q4_0_packed_batched_launch(out, q, s, eng->d_qx, eng->d_dx, eng->d_sx,
                                        in_dim, out_dim, /*K=*/1, stream);
    } else {
        mmvq_launch(out, weight, eng->d_qx, eng->d_dx, eng->d_sx, in_dim, out_dim, fmt, stream);
    }
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
    quantize_q8_1_kernel<<<(K*in_dim)/32, 32, 0, stream>>>(
        x, eng->d_qx_b, eng->d_dx_b, eng->d_sx_b, K*in_dim);
    if (fmt == FORMAT_Q6_K) {            // native Q6_K LM head (Step 8), batched over K rows
        mmvq_q6_k_batched_launch(out, weight, eng->d_qx_b, eng->d_dx_b,
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

static inline void embed_w(
    const gemma4_engine_t *eng,
    float *out, const uint8_t *table, const int32_t *tokens,
    int batch, int hidden_size, cudaStream_t stream, int efmt = -1)
{
    (void)efmt;   // the token_embd table is always Q8_0 (native, or converted from Q6_K)
    embed_lookup_q8_0_kernel<<<batch, 256, 0, stream>>>(
        out, table, tokens, batch, hidden_size);
}

// LM-head weight pointer + format. Step 8: when the tied head is kept NATIVE Q6_K
// (eng->lmhead_q6k, set at create), the output projection reads the raw Q6_K bytes from the
// device weight blob (d_lmhead_q6k) as FORMAT_Q6_K — cutting ~0.24 GB/token off the V×H read.
static inline int lmhead_native_q6k(const gemma4_engine_t *eng) {
    return eng->lmhead_q6k && eng->d_lmhead_q6k;
}
static inline const unsigned char* lmhead_w(const gemma4_engine_t *eng) {
    if (lmhead_native_q6k(eng)) return eng->d_lmhead_q6k;
    uint64_t off = eng->tensors.output_weight;
    return weight_fp8(eng, off);
}
static inline int embd_fmt(const gemma4_engine_t *eng) {
    if (lmhead_native_q6k(eng)) return (int)FORMAT_Q6_K;
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
    global_attn_splitk_kernel<GEMMA4_HEADS, GEMMA4_GLOBAL_HEAD_DIM>
        <<<splits, GEMMA4_HEADS*32, 0, stream>>>(
        eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, q, kc, vc, head_dim, ctx_len, splits);
    flash_decode_combine_kernel<GEMMA4_HEADS><<<n_heads, head_dim, 0, stream>>>(
        out, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, head_dim, splits);
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
    sliding_attn_splitk_kernel<GEMMA4_HEADS, GEMMA4_KV_HEADS, GEMMA4_HEAD_DIM>
        <<<splits, GEMMA4_KV_HEADS * 32, 0, stream>>>(
            eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, q, kc, vc,
            window, n_tokens, splits, eng->sliding_kv_capacity);
    flash_decode_combine_kernel<GEMMA4_HEADS><<<n_heads, head_dim, 0, stream>>>(
        out, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, head_dim, splits);
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
#define NORM_W(arr) (eng->arr + (size_t)layer * GEMMA4_HIDDEN_SIZE)
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
    pool_k[idx] = pkv_float_to_fp8(kb[e]);
    pool_v[idx] = pkv_float_to_fp8(vb[e]);
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

static int decode_layer(
    gemma4_engine_t *eng,
    int              layer,
    int              pos,
    int              context_len,
    cudaStream_t     stream,
    const int       *d_pos = NULL)
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

    // ── 2-4. Q/K/V projections — quantize the attn-norm activation ONCE ───
    // (audit #30 lever 2): the three projections share one int8 quant of d_norm instead of
    // re-quantizing it per gemv_w call. q/k/v are always the layer's Q4_0/Q8_0 format (never
    // the Q6_K head), so mmvq_launch is called directly. Bit-identical (same quant input).
    quantize_q8_1_kernel<<<GEMMA4_HIDDEN_SIZE/32, 32, 0, stream>>>(
        eng->d_norm, eng->d_qx, eng->d_dx, eng->d_sx, GEMMA4_HIDDEN_SIZE);
    mmvq_q4aware(eng, eng->d_attn_q, weight_fp8(eng, eng->tensors.layers[layer].attn_q),
        eng->d_qx, eng->d_dx, eng->d_sx, GEMMA4_HIDDEN_SIZE, out_dim_q, (int)eng->format, stream);
    PER_HEAD_NORM(eng->d_attn_q, HEAD_NORM_W(d_w_q_norm), n_heads, head_dim);

    // ── 3. K projection → per-head RMSNorm ───────────────────────────────
    mmvq_q4aware(eng, eng->d_attn_k, weight_fp8(eng, eng->tensors.layers[layer].attn_k),
        eng->d_qx, eng->d_dx, eng->d_sx, GEMMA4_HIDDEN_SIZE, out_dim_kv, (int)eng->format, stream);
    // ── 4. V projection → plain RMSNorm (no weight) ──────────────────────
    // For global layers V = K BEFORE any norm/RoPE is applied.
    // For sliding layers V comes from a separate projection.
    if (ltype == LAYER_SLIDING) {
        mmvq_q4aware(eng, eng->d_attn_v, weight_fp8(eng, eng->tensors.layers[layer].attn_v),
            eng->d_qx, eng->d_dx, eng->d_sx, GEMMA4_HIDDEN_SIZE, out_dim_kv, (int)eng->format, stream);
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
        size_t layer_stride = (size_t)eng->sliding_kv_capacity * GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM;
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
        size_t layer_stride = (size_t)eng->global_kv_capacity * GEMMA4_GLOBAL_HEAD_DIM;
        kv_t *base_k = eng->d_global_k + (size_t)slot * layer_stride;
        kv_t *base_v = eng->d_global_v + (size_t)slot * layer_stride;
        if (d_pos) {
            copy_f32_to_fp8_at_kernel<<<kvg, 256, 0, stream>>>(base_k, d_pos, GEMMA4_GLOBAL_HEAD_DIM, eng->d_attn_k, kv_size, eng->global_kv_capacity);
            copy_f32_to_fp8_at_kernel<<<kvg, 256, 0, stream>>>(base_v, d_pos, GEMMA4_GLOBAL_HEAD_DIM, eng->d_attn_v, kv_size, eng->global_kv_capacity);
        } else {
            copy_f32_to_fp8_kernel<<<kvg, 256, 0, stream>>>(base_k + (size_t)n*GEMMA4_GLOBAL_HEAD_DIM, eng->d_attn_k, kv_size);
            copy_f32_to_fp8_kernel<<<kvg, 256, 0, stream>>>(base_v + (size_t)n*GEMMA4_GLOBAL_HEAD_DIM, eng->d_attn_v, kv_size);
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
        size_t lstride = (size_t)eng->sliding_kv_capacity * GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM;
        if (d_pos) {
            // Graph path: fixed-grid rows kernels (r=0) with device n_tokens = d_pos[1].
            // Max sliding splits = ceil(WINDOW/CHUNK); shorter windows tail-return.
            const int max_splits =
                (GEMMA4_SLIDING_WINDOW + GEMMA4_SLIDING_SPLIT_CHUNK - 1) / GEMMA4_SLIDING_SPLIT_CHUNK;
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
        size_t layer_stride = (size_t)eng->global_kv_capacity * GEMMA4_GLOBAL_HEAD_DIM;
        if (d_pos) {
            // Graph path: fixed-grid rows kernels (r=0) with device n_tokens = d_pos[1].
            global_attn_splitk_rows_kernel<GEMMA4_HEADS, GEMMA4_GLOBAL_HEAD_DIM>
                <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), GEMMA4_HEADS*32, 0, stream>>>(
                    eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_attn_q,
                    eng->d_global_k + (size_t)slot * layer_stride,
                    eng->d_global_v + (size_t)slot * layer_stride,
                    0, d_pos + 1);
            flash_decode_combine_rows_kernel<GEMMA4_HEADS>
                <<<dim3(GEMMA4_HEADS, 1), head_dim, 0, stream>>>(
                    eng->d_attn_out, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                    head_dim, /*window=*/0, 0, d_pos + 1, n_heads*head_dim);
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
    gemv_w(eng, eng->d_x,
        weight_fp8(eng, eng->tensors.layers[layer].attn_output),
        eng->d_attn_out, out_dim_q, GEMMA4_HIDDEN_SIZE, stream);
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

    // ── 10+11. Fused Gate + Up + GeGLU (audit #30 lever 2) ────────────────
    // Quantize the shared FFN-norm activation ONCE (+ the per-block Σ for the Q4_0 −8 fold),
    // then ONE fused kernel computes gate·up·GeGLU per row — the gate/up intermediates never
    // touch DRAM, and interleaving both weight reads in-warp lifts FFN bandwidth. Bit-identical
    // to the old gate-mmvq + up-mmvq + geglu (see mmvq_q*_glu_kernel).
    quantize_q8_1_kernel<<<GEMMA4_HIDDEN_SIZE/32, 32, 0, stream>>>(
        eng->d_norm, eng->d_qx, eng->d_dx, eng->d_sx, GEMMA4_HIDDEN_SIZE);
    mmvq_glu_launch(eng->d_ffn_out,
        weight_fp8(eng, eng->tensors.layers[layer].ffn_gate),
        weight_fp8(eng, eng->tensors.layers[layer].ffn_up),
        eng->d_qx, eng->d_dx, eng->d_sx, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE,
        (int)eng->format, stream);

    // ── 12. FFN down projection ───────────────────────────────────────────
    gemv_w(eng, eng->d_x,
        weight_fp8(eng, eng->tensors.layers[layer].ffn_down),
        eng->d_ffn_out, GEMMA4_INTERMEDIATE, GEMMA4_HIDDEN_SIZE, stream);

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
    for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
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

// Build persistent NVFP4 weights for every (layer,proj). Dequants each projection
// (Q4_0/Q8_0 → bf16 in a temp buffer) then NVFP4-quantizes it. Returns 0 / -1.
static int build_fp4_weights(gemma4_engine_t *eng) {
    if (eng->fp4_ready) return 0;
    if (cublasLtCreate(&eng->cublaslt) != CUBLAS_STATUS_SUCCESS) return -1;
    // find max projection element count for the temp bf16 + linear-scale scratch
    uint64_t maxn = 0; int max_in = 0, max_out = 0;
    for (int l=0;l<GEMMA4_MAX_LAYERS;l++) for (int p=0;p<PJ_COUNT;p++){
        uint64_t off; int in_dim,out_dim; if(!proj_desc(eng,l,p,&off,&in_dim,&out_dim)) continue;
        uint64_t nn=(uint64_t)in_dim*out_dim; if(nn>maxn)maxn=nn;
        if(in_dim>max_in)max_in=in_dim; if(out_dim>max_out)max_out=out_dim;
    }
    __nv_bfloat16 *tmp_bf=nullptr; uint8_t *tmp_lin=nullptr;
    int ok=1;
    if (cudaMalloc(&tmp_bf, maxn*sizeof(__nv_bfloat16))!=cudaSuccess) ok=0;
    if (ok && cudaMalloc(&tmp_lin, (size_t)max_out*(max_in/NVFP4_BLK))!=cudaSuccess) ok=0;
    if (ok && cudaMalloc(&eng->d_fp4_gsw, (size_t)GEMMA4_MAX_LAYERS*PJ_COUNT*sizeof(float))!=cudaSuccess) ok=0;
    if (ok && cudaMalloc(&eng->d_fp4_amax, sizeof(float))!=cudaSuccess) ok=0;
    size_t wbytes=0;
    for (int l=0; ok && l<GEMMA4_MAX_LAYERS; l++) for (int p=0; ok && p<PJ_COUNT; p++){
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
    size_t I = GEMMA4_INTERMEDIATE;
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

// Lazily allocate the tiled-MMQ prefill activation scratch (Q4_0 models only): the
// quantized [N × in_dim] activation for N ≤ MMQ_MAX_N over the widest projection
// (in_dim = INTERMEDIATE). Idempotent. Returns 0 on success, -1 on failure (caller
// falls back to the BF16 path).
static int ensure_mmq_scratch(gemma4_engine_t *eng)
{
    if (eng->mmq_ready) return 0;
    const size_t Nmax = GEMMA4_MMQ_MAX_N;
    const size_t I    = GEMMA4_INTERMEDIATE;
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
    const int H   = GEMMA4_HIDDEN_SIZE;
    const int I   = GEMMA4_INTERMEDIATE;
    const int HD2 = 32 * sizeof(float);
    const int base = 0;

    // Tiled-MMQ fast path (BLACKWELL_NATIVE_FP4 default): Q4_0 weights, small/mid batch.
    // Reads the native Q4_0 weights ONCE per projection (no BF16 materialize, no per-layer
    // dequant pass) — eliminating the ~125 ms fixed full-model dequant that dominated
    // suffix prefills. Above MMQ_MAX_N the BF16 tensor-core GEMM (with pipelined dequant)
    // wins, so we fall through to it. mmq needs no BF16 scratch; only build it otherwise.
    const bool use_mmq = mmq_enabled(eng, N);
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
    static int fp4_opt = -1, fp4_min = 1024;
    if (fp4_opt < 0) {
        const char *e = getenv("FUCINA_FP4"); fp4_opt = (e && e[0]=='0') ? 0 : 1;  // on unless =0
        const char *mn = getenv("FUCINA_FP4_MIN"); if (mn) fp4_min = atoi(mn);
    }
    const bool use_fp4 = fp4_opt && !use_mmq && N >= fp4_min
                         && build_fp4_weights(eng) == 0 && ensure_fp4_act(eng, N) == 0;
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
    const int OQ_MAX = GEMMA4_HEADS * GEMMA4_GLOBAL_HEAD_DIM; // 8192
    const int OKV_MAX = GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM;    // 2048
    const int HEADS = GEMMA4_HEADS;
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

        // BF16 path: kick off the NEXT layer's dequant on dq_stream, then gate this layer's
        // GEMMs on its own (already-issued) dequant. MMQ path: no dequant — gemm_proj reads
        // the native Q4_0 weights directly.
        if (ovl) {
            curbuf = l & 1;
            if (l + 1 < GEMMA4_MAX_LAYERS) issue_dequant(l + 1);
            cudaStreamWaitEvent(stream, eng->ev_dq_done[curbuf], 0);
        }

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
                GEMMA4_HEADS, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            attn_softmax_batched_kernel<<<dim3(N,GEMMA4_HEADS),256,HD2,stream>>>(d_scores, d_pb, N, window);
            // O[h] = V[h]·P[h]ᵀ  (m=hd,n=N,k=N → C col-major [hd×N] ld=oq stride=hd)
            cublasGemmStridedBatchedEx(eng->cublas, CUBLAS_OP_N, CUBLAS_OP_T, hd, N, N,
                &a1, d_vbx, CUDA_R_16BF, oq, (long long)hd,
                     d_pb,  CUDA_R_16BF, N,  sNN,
                &b0, d_attn, CUDA_R_32F, oq, (long long)hd,
                GEMMA4_HEADS, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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
            int slot = eng->global_slot[l];
            size_t stride = (size_t)eng->global_kv_capacity * GEMMA4_GLOBAL_HEAD_DIM;
            kv_write_global_kernel<<<dim3(grid1d(hd),N),256,0,stream>>>(
                eng->d_global_k + slot*stride, eng->d_global_v + slot*stride,
                d_k, d_v, base, N, hd);
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
    gemv_w(eng, eng->d_logits, lmhead_w(eng),
           eng->d_norm, H, GEMMA4_VOCAB_SIZE, stream, embd_fmt(eng));
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
    const int base0 = eng->cur.n_tokens;               // 0 = fresh, >0 = suffix
    if (base0 + n_tokens > eng->global_kv_capacity) return -2;  // would overflow cache
    // Tiny suffixes stay on the chunked dp4a fallback: this path pays a fixed
    // per-chunk full-model BF16 dequant (~28 GB of traffic, ~125 ms) that only
    // amortizes past a few dozen tokens; below that two 16-token dp4a chunks win.
    if (base0 > 0 && n_tokens < 32) return -2;
    if (ensure_fp_scratch(eng) != 0) return -2;           // attention tile scratch

    cudaStream_t stream = eng->stream;
    const int H = GEMMA4_HIDDEN_SIZE, I = GEMMA4_INTERMEDIATE, HD2 = 32*sizeof(float);
    const int HEADS = GEMMA4_HEADS, N = n_tokens;
    const int C = (N < GEMMA4_FP_CHUNK) ? N : GEMMA4_FP_CHUNK;  // chunk (amortizes weight re-reads)
    const int OQ_MAX = HEADS*GEMMA4_GLOBAL_HEAD_DIM, OKV_MAX = GEMMA4_KV_HEADS*GEMMA4_HEAD_DIM;

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
        for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
            // Abort granularity: one 8192-token chunk runs ~30s at long context,
            // so poll per LAYER (~0.7s). Mid-chunk abort is safe for the same
            // reason as mid-prefill: nothing is accounted until global_n_tokens
            // advances, so partial KV writes are invisible and overwritten.
            if (eng->abort_req) { aborted = 1; break; }
            layer_type_t lt = eng->layer_types[l];
            int hd  = (lt==LAYER_SLIDING)? GEMMA4_HEAD_DIM : GEMMA4_GLOBAL_HEAD_DIM;
            int nkv = (lt==LAYER_SLIDING)? GEMMA4_KV_HEADS : GEMMA4_GLOBAL_KV_HEADS;
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
                if (l + 1 < GEMMA4_MAX_LAYERS) issue_dequant(l + 1);
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

                f32_to_bf16_kernel<<<grid1d((size_t)cn*oq),256,0,stream>>>(
                    eng->d_fp_qb, d_q, (size_t)cn*oq);
                if (global) {
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
                // GLOBAL layers (84% of attention FLOPs): Q token-major
                // [cn][HEADS][hd] IS col-major [hd × HEADS·cn] with ld=hd, and
                // all heads share the single KV head — so S and P·V are each
                // ONE wide GEMM (n = HEADS·qn), no broadcast, no batching.
                // Keys kt/vt are [tn][hd].
                // SLIDING layers (GQA 16:8, window-bounded): strided-batched
                // GEMM over heads with broadcast keys [tn][HEADS*hd].
                auto tile_pass = [&](const __nv_bfloat16 *kt, const __nv_bfloat16 *vt,
                                     int tn, int kbase, int q0, int qn){
                    if (qn <= 0 || tn <= 0) return;
                    if (global) {
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
                    size_t stride = (size_t)eng->global_kv_capacity * GEMMA4_GLOBAL_HEAD_DIM;
                    hk = eng->d_global_k + (size_t)slot*stride;
                    hv = eng->d_global_v + (size_t)slot*stride;
                    hcap = eng->global_kv_capacity;
                }
                int hstart = 0;                           // sliding: only the window reach
                if (window > 0) { hstart = abs0 - (window - 1); if (hstart < 0) hstart = 0; }
                // Tile row stride: global keys are non-broadcast [tn][hd],
                // sliding keys are broadcast [tn][HEADS*hd].
                const size_t krow = global ? (size_t)okv : (size_t)oq;
                for (int t0 = hstart; t0 < abs0; t0 += T) {
                    int tn = (abs0 - t0 < T) ? (abs0 - t0) : T;
                    fp_hist_tile_bf16_kernel<<<dim3(grid1d(hd),global?1:HEADS,tn),256,0,stream>>>(
                        eng->d_fp_kt, eng->d_fp_vt, hk, hv, t0, tn, global?1:HEADS, nkv, hd, hcap);
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
                int slot = eng->global_slot[l];
                size_t stride = (size_t)eng->global_kv_capacity * GEMMA4_GLOBAL_HEAD_DIM;
                kv_write_global_kernel<<<dim3(grid1d(hd),cn),256,0,stream>>>(
                    eng->d_global_k + slot*stride, eng->d_global_v + slot*stride,
                    d_k, d_v, abs0, cn, hd);
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
            gemv_w(eng, eng->d_logits, lmhead_w(eng),
                   eng->d_norm, H, GEMMA4_VOCAB_SIZE, stream, embd_fmt(eng));
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
                            GEMMA4_HIDDEN_SIZE * sizeof(float),
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
        cudaMemcpyAsync(eng->d_mtp_h, eng->d_sb[2] + (size_t)(lastK - 1) * GEMMA4_HIDDEN_SIZE,
                        GEMMA4_HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, eng->stream);
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
    embed_w(eng, eng->d_x, weight_fp8(eng, eng->tensors.token_embd),
            eng->d_dectok, 1, GEMMA4_HIDDEN_SIZE, stream);
    float sc = sqrtf((float)GEMMA4_HIDDEN_SIZE);
    scale_kernel<<<(GEMMA4_HIDDEN_SIZE+255)/256, 256, 0, stream>>>(
        eng->d_x, GEMMA4_HIDDEN_SIZE, sc);
    for (int l = 0; l < GEMMA4_MAX_LAYERS; l++)
        decode_layer(eng, l, /*pos=*/0, /*context_len=*/0, stream, eng->d_decpos);
    rms_norm_kernel<<<1, 256, 32*sizeof(float), stream>>>(
        eng->d_norm, eng->d_x, eng->d_w_out_norm, GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_EPS);
    if (eng->mtp.loaded)
        cudaMemcpyAsync(eng->d_mtp_h, eng->d_norm,
                        GEMMA4_HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    gemv_w(eng, eng->d_logits, lmhead_w(eng), eng->d_norm,
           GEMMA4_HIDDEN_SIZE, GEMMA4_VOCAB_SIZE, stream, embd_fmt(eng));
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
        // Embedding (format-aware: FP8 or Q8_0 table)
        embed_w(eng,
            eng->d_x,
            weight_fp8(eng, eng->tensors.token_embd),
            &token, 1, GEMMA4_HIDDEN_SIZE, stream);

        // Gemma scales embeddings by √hidden_size
        { float sc = sqrtf((float)GEMMA4_HIDDEN_SIZE);
          scale_kernel<<<(GEMMA4_HIDDEN_SIZE+255)/256, 256, 0, stream>>>(
                eng->d_x, GEMMA4_HIDDEN_SIZE, sc); }

        // Paged KV: ensure the active sequence's block tables cover `pos` and
        // the device block ids are fresh BEFORE any layer mirrors its KV write.
        if (eng->paged_enabled) paged_seq_sync(eng, pos);

        // Run all 48 layers
        for (int l = 0; l < GEMMA4_MAX_LAYERS; l++) {
            decode_layer(eng, l, pos, eng->cur.n_tokens + 1, stream);
        }

        // Output norm + projection + softcap
        rms_norm_kernel<<<1, 256, 32*sizeof(float), stream>>>(
            eng->d_norm, eng->d_x, eng->d_w_out_norm,
            GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_EPS);

        // MTP recurrent h: the LM-head input (post-output-norm hidden) is exactly the
        // h the assistant drafter pairs with the token sampled from these logits.
        if (eng->mtp.loaded) {
            cudaMemcpyAsync(eng->d_mtp_h, eng->d_norm,
                            GEMMA4_HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }

        int vocab = GEMMA4_VOCAB_SIZE;
        gemv_w(eng, eng->d_logits,
            lmhead_w(eng),
            eng->d_norm,
            GEMMA4_HIDDEN_SIZE, vocab, stream, embd_fmt(eng));

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
    for (int L = 0; L < GEMMA4_MAX_LAYERS; L++) {
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
                K, GEMMA4_MAX_LAYERS);
    else
        fprintf(stderr, "fucina: paged self-test FAILED — %d mismatching fp8 elements\n", mm);

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
    // GPU-side verify scratch (a): K ids + K draws. d_sample_id is also used by the
    // cold/no-draft single-row device sample — allocate it here too (load_assistant
    // also allocates it, but lookup-only spec has no assistant).
    if (!eng->d_sample_id)
        if (cudaMalloc(&eng->d_sample_id, sizeof(int)) != cudaSuccess) { cudaGetLastError(); return -1; }
    if (!eng->d_spec_ids)
        if (cudaMalloc(&eng->d_spec_ids, M*sizeof(int)) != cudaSuccess) { cudaGetLastError(); return -1; }
    if (!eng->d_spec_rnd)
        if (cudaMalloc(&eng->d_spec_rnd, M*sizeof(float)) != cudaSuccess) { cudaGetLastError(); return -1; }
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
    const int *d_pos, int want_head)
{
    const int H   = GEMMA4_HIDDEN_SIZE;
    const int I   = GEMMA4_INTERMEDIATE;
    const int HD2 = 32 * sizeof(float);
    const int HEADS = GEMMA4_HEADS;
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
            rope_rows_kernel<<<dim3(HEADS,K),hd/2,0,stream>>>(d_q, d_k, pos, HEADS, nkv, hd, K, 10000.0f, NULL, d_pos);
        else
            rope_rows_kernel<<<dim3(HEADS,K),hd/2,0,stream>>>(d_q, d_k, pos, HEADS, nkv, hd, K, 1000000.0f, eng->d_rope_freqs, d_pos);

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
            sliding_attn_splitk_rows_kernel<GEMMA4_HEADS, GEMMA4_KV_HEADS, GEMMA4_HEAD_DIM>
                <<<dim3(max_splits, K), GEMMA4_KV_HEADS*32, 0, stream>>>(
                    eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, d_q, kc, vc,
                    GEMMA4_SLIDING_WINDOW, pos + 1, eng->sliding_kv_capacity, d_ntok);
            flash_decode_combine_rows_kernel<GEMMA4_HEADS><<<dim3(HEADS, K), hd, 0, stream>>>(
                d_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                hd, GEMMA4_SLIDING_WINDOW, pos + 1, d_ntok, oq);
        } else {
            int slot = eng->global_slot[l];
            size_t lstride = (size_t)eng->global_kv_capacity * GEMMA4_GLOBAL_HEAD_DIM;
            kv_t *kc = eng->d_global_k + (size_t)slot*lstride;
            kv_t *vc = eng->d_global_v + (size_t)slot*lstride;
            kv_write_global_kernel<<<dim3(grid1d(hd),K),256,0,stream>>>(
                kc, vc, d_k, d_v, pos, K, hd, d_pos);
            int max_splits;
            if (d_pos) {
                max_splits = GEMMA4_GLOBAL_MAX_SPLITS;
            } else {
                max_splits = (pos + K + GEMMA4_GLOBAL_SPLIT_CHUNK - 1) / GEMMA4_GLOBAL_SPLIT_CHUNK;
                if (max_splits < 1) max_splits = 1;
                if (max_splits > GEMMA4_GLOBAL_MAX_SPLITS) max_splits = GEMMA4_GLOBAL_MAX_SPLITS;
            }
            global_attn_splitk_rows_kernel<GEMMA4_HEADS, GEMMA4_GLOBAL_HEAD_DIM>
                <<<dim3(max_splits, K), GEMMA4_HEADS*32, 0, stream>>>(
                    eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, d_q, kc, vc, pos + 1, d_ntok);
            flash_decode_combine_rows_kernel<GEMMA4_HEADS><<<dim3(HEADS, K), hd, 0, stream>>>(
                d_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                hd, /*window=*/0, pos + 1, d_ntok, oq);
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

    // Output norm (batched) + LM head as ONE batched GEMV (tied LM head read once for all
    // K) + softcap + suppress. Result stays in d_sb[11] (d_logitsK); the caller D2Hs it
    // (per-kernel public path) or samples it on-device (keep_dev / graph path).
    if (want_head) {
        int vocab = GEMMA4_VOCAB_SIZE;
        rms_norm_rows_kernel<<<K,256,HD2,stream>>>(d_norm, d_x, eng->d_w_out_norm, H, K, GEMMA4_RMS_EPS);
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
    // Weights must be Q8_0 (the mmvq path assumes it) and norms/scales F32 — reject any
    // other assistant quantization instead of silently producing garbage drafts.
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
    #define MTP_FIND(dst, namefmt, ...)  MTP_FIND_T(dst, GGML_TYPE_Q8_0, namefmt, ##__VA_ARGS__)
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
    #undef MTP_FIND
    #undef MTP_FINDF
    #undef MTP_FIND_T

    // scratch
    ok &= cudaMalloc(&eng->d_mtp_h,    GEMMA4_HIDDEN_SIZE   * sizeof(float)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_mtp_xh,   2*GEMMA4_HIDDEN_SIZE * sizeof(float)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_mtp_cur,  GEMMA4_MTP_HIDDEN    * sizeof(float)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_mtp_t1,   GEMMA4_MTP_HIDDEN    * sizeof(float)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_mtp_t2,   GEMMA4_MTP_HIDDEN    * sizeof(float)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_mtp_q,    GEMMA4_HEADS*GEMMA4_GLOBAL_HEAD_DIM * sizeof(float)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_mtp_attn, GEMMA4_HEADS*GEMMA4_GLOBAL_HEAD_DIM * sizeof(float)) == cudaSuccess;
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
    const int H = GEMMA4_HIDDEN_SIZE, AH = GEMMA4_MTP_HIDDEN, FF = GEMMA4_MTP_FFN;
    const int smem32 = 32 * (int)sizeof(float);

    // x = target_embd(tok)·√3840  ‖  h   → pre_projection → cur [1024]
    embed_lookup_q8_0_kernel<<<1, 256, 0, stream>>>(
        eng->d_mtp_xh, weight_fp8(eng, eng->tensors.token_embd), tok_ptr, 1, H);
    scale_kernel<<<(H+255)/256, 256, 0, stream>>>(eng->d_mtp_xh, H, sqrtf((float)H));
    cudaMemcpyAsync(eng->d_mtp_xh + H, eng->d_mtp_h, H * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    gemv_w(eng, eng->d_mtp_cur, mtp_w(eng, eng->mtp.pre_proj), eng->d_mtp_xh,
           2*H, AH, stream, FORMAT_Q8_0);

    for (int l = 0; l < GEMMA4_MTP_LAYERS; l++) {
        const int is_g     = eng->mtp.is_global[l];
        const int head_dim = is_g ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_HEAD_DIM;
        const int qdim     = GEMMA4_HEADS * head_dim;

        // attention (Q-only; K/V come from the target's cache)
        rms_norm_kernel<<<1, 256, smem32, stream>>>(
            eng->d_mtp_t1, eng->d_mtp_cur, mtp_f32(eng, eng->mtp.attn_norm[l]), AH, GEMMA4_RMS_EPS);
        gemv_w(eng, eng->d_mtp_q, mtp_w(eng, eng->mtp.wq[l]), eng->d_mtp_t1,
               AH, qdim, stream, FORMAT_Q8_0);
        per_head_rms_norm_kernel<<<GEMMA4_HEADS, head_dim, smem32, stream>>>(
            eng->d_mtp_q, mtp_f32(eng, eng->mtp.q_norm[l]), head_dim, GEMMA4_RMS_EPS);
        if (is_g) {
            rope_global_kernel<<<GEMMA4_HEADS, head_dim/2, 0, stream>>>(
                eng->d_mtp_q, eng->d_mtp_q, 0, pos_ptr, 0, GEMMA4_HEADS, /*n_kv_heads=*/0,
                head_dim, 1000000.0f, mtp_f32(eng, eng->mtp.rope_freqs));
            // target layer 47 = the LAST global layer (llama.cpp share(il)=n_layer-1).
            // Fixed-grid split-K with device n_tokens (= *pos_ptr): same split formula
            // as global_attn_decode_broadcast, so the math is bit-identical to it.
            int slot = eng->global_slot[GEMMA4_MAX_LAYERS - 1];
            size_t lstride = (size_t)eng->global_kv_capacity * GEMMA4_GLOBAL_HEAD_DIM;
            global_attn_splitk_rows_kernel<GEMMA4_HEADS, GEMMA4_GLOBAL_HEAD_DIM>
                <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), GEMMA4_HEADS*32, 0, stream>>>(
                    eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_mtp_q,
                    eng->d_global_k + (size_t)slot * lstride,
                    eng->d_global_v + (size_t)slot * lstride,
                    0, pos_ptr);
            flash_decode_combine_rows_kernel<GEMMA4_HEADS>
                <<<dim3(GEMMA4_HEADS, 1), head_dim, 0, stream>>>(
                    eng->d_mtp_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                    head_dim, /*window=*/0, 0, pos_ptr, GEMMA4_HEADS*head_dim);
        } else {
            rope_sliding_kernel<<<GEMMA4_HEADS, head_dim/2, 0, stream>>>(
                eng->d_mtp_q, eng->d_mtp_q, 0, pos_ptr, GEMMA4_HEADS, /*n_kv_heads=*/0,
                head_dim, 10000.0f);
            // target layer 46 = the LAST sliding layer (share(il)=n_layer-2); the
            // drafter attends window-1 keys of the frozen cache at n_tokens = *pos_ptr.
            size_t lstride = (size_t)eng->sliding_kv_capacity * GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM;
            sliding_attn_splitk_rows_kernel<GEMMA4_HEADS, GEMMA4_KV_HEADS, GEMMA4_HEAD_DIM>
                <<<dim3(GEMMA4_GLOBAL_MAX_SPLITS, 1), GEMMA4_KV_HEADS*32, 0, stream>>>(
                    eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l, eng->d_mtp_q,
                    eng->d_sliding_k + (size_t)(GEMMA4_MAX_LAYERS - 2) * lstride,
                    eng->d_sliding_v + (size_t)(GEMMA4_MAX_LAYERS - 2) * lstride,
                    GEMMA4_SLIDING_WINDOW - 1, 0, eng->sliding_kv_capacity, pos_ptr);
            flash_decode_combine_rows_kernel<GEMMA4_HEADS>
                <<<dim3(GEMMA4_HEADS, 1), head_dim, 0, stream>>>(
                    eng->d_mtp_attn, eng->d_fa_acc, eng->d_fa_m, eng->d_fa_l,
                    head_dim, GEMMA4_SLIDING_WINDOW - 1, 0, pos_ptr, GEMMA4_HEADS*head_dim);
        }
        gemv_w(eng, eng->d_mtp_t1, mtp_w(eng, eng->mtp.wo[l]), eng->d_mtp_attn,
               qdim, AH, stream, FORMAT_Q8_0);

        rms_norm_kernel<<<1, 256, smem32, stream>>>(
            eng->d_mtp_t2, eng->d_mtp_t1, mtp_f32(eng, eng->mtp.post_attn_norm[l]), AH, GEMMA4_RMS_EPS);
        residual_add_kernel<<<(AH+255)/256, 256, 0, stream>>>(eng->d_mtp_t2, eng->d_mtp_cur, AH);

        // FFN (GeGLU 1024 → 8192 → 1024)
        rms_norm_kernel<<<1, 256, smem32, stream>>>(
            eng->d_mtp_t1, eng->d_mtp_t2, mtp_f32(eng, eng->mtp.ffn_norm[l]), AH, GEMMA4_RMS_EPS);
        gemv_w(eng, eng->d_mtp_ffa, mtp_w(eng, eng->mtp.gate[l]), eng->d_mtp_t1,
               AH, FF, stream, FORMAT_Q8_0);
        gemv_w(eng, eng->d_mtp_ffb, mtp_w(eng, eng->mtp.up[l]), eng->d_mtp_t1,
               AH, FF, stream, FORMAT_Q8_0);
        geglu_kernel<<<(FF+255)/256, 256, 0, stream>>>(eng->d_mtp_ffa, eng->d_mtp_ffa, eng->d_mtp_ffb, FF);
        gemv_w(eng, eng->d_mtp_t1, mtp_w(eng, eng->mtp.down[l]), eng->d_mtp_ffa,
               FF, AH, stream, FORMAT_Q8_0);
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
           AH, GEMMA4_VOCAB_SIZE, stream, FORMAT_Q8_0);
    gemv_w(eng, eng->d_mtp_h, mtp_w(eng, eng->mtp.post_proj), eng->d_mtp_t1,
           AH, GEMMA4_HIDDEN_SIZE, stream, FORMAT_Q8_0);
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
        if (conf < GEMMA4_MTP_PMIN) break;   // low confidence → stop drafting here
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
        int D = prompt_lookup_draft(hist, n, batch+1, maxd_lk, 2, draft_k, &lk_ng, &lk_nocc);
        int from_mtp = 0;
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
    *srow   = (size_t)GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM * sizeof(kv_t);
    *spitch = (size_t)eng->sliding_kv_capacity * (*srow);
    *grow   = (size_t)GEMMA4_GLOBAL_HEAD_DIM * sizeof(kv_t);
    *gpitch = (size_t)eng->global_kv_capacity * (*grow);
}

size_t gemma4_engine_kv_state_size(const gemma4_engine_t *eng, int n_tokens) {
    if (!eng || n_tokens <= 0 || n_tokens > eng->sliding_kv_capacity ||
        n_tokens > eng->global_kv_capacity) return 0;
    size_t srow, spitch, grow, gpitch;
    kv_seq_pitches(eng, &srow, &spitch, &grow, &gpitch);
    return 2 * (size_t)GEMMA4_MAX_LAYERS    * (size_t)n_tokens * srow +
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
        { (char *)eng->d_sliding_k, spitch, swidth, GEMMA4_MAX_LAYERS },
        { (char *)eng->d_sliding_v, spitch, swidth, GEMMA4_MAX_LAYERS },
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
    const int NC = 4096, H = GEMMA4_HIDDEN_SIZE, I = GEMMA4_INTERMEDIATE;
    const int OQ = GEMMA4_HEADS * GEMMA4_GLOBAL_HEAD_DIM;
    const int OK = GEMMA4_KV_HEADS * GEMMA4_HEAD_DIM;
    const int HE = GEMMA4_HEADS;
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
    #undef A
    if (ok != cudaSuccess) { cudaGetLastError(); return -1; }
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
    const size_t OQ = (size_t)GEMMA4_HEADS * GEMMA4_GLOBAL_HEAD_DIM;
    const size_t HC = (size_t)GEMMA4_HEADS * C;
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

