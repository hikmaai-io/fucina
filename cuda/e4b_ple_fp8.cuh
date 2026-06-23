// e4b_ple_fp8.cuh — FP8 E4M3 "index" codec for the Gemma-4-E4B Per-Layer
// Embedding table (`embed_tokens_per_layer`, [vocab, n_layers*ple_dim]).
//
// Why: the PLE table is the single biggest weight in E4B — 262144 × 10752 ×
// 2 B = 5.6 GB in BF16, larger than the main token embedding. It is pure
// lookup (one row per token, gathered once at the start of a forward pass),
// so it tolerates aggressive quantization with no GEMV-accumulation error.
// Storing it FP8 E4M3 with a per-row scale halves it to ~2.7 GB + 1 MB of
// scales — the headline memory win for memory-constrained clients, and the
// "fp8 index" the feature asks for.
//
// Layout (mirrors the FP8 untied-head codec in gemma4_kernels.cu):
//   q     : uint8 (E4M3 bits)  [vocab * width]   row-major
//   scale : float              [vocab]           per-row  amax/448
// Dequant: value = fp8_to_float(q[row*width + col]) * scale[row].
//
// E4M3 max finite magnitude is 448; per-row amax scaling keeps the largest
// element at the top of the representable range so small magnitudes still get
// mantissa bits. Rows that are all-zero get scale 0 (dequant → 0).
#ifndef FUCINA_E4B_PLE_FP8_CUH
#define FUCINA_E4B_PLE_FP8_CUH

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define E4B_FP8_E4M3_MAX 448.0f

__device__ __forceinline__ float e4b_fp8_to_float(__nv_fp8_storage_t v) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3)));
}
__device__ __forceinline__ __nv_fp8_storage_t e4b_float_to_fp8(float v) {
    return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// One block per row; threads stride the row. Two passes: blockwide amax, then
// scaled quantize. width is typically 10752 (n_layers*ple_dim).
__global__ void e4b_ple_quantize_kernel(
    const __nv_bfloat16* __restrict__ src,   // [rows * width] BF16
    __nv_fp8_storage_t*  __restrict__ q,      // [rows * width] out
    float*               __restrict__ scale,  // [rows] out
    int rows, int width)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* s = src + (size_t)row * width;
    __nv_fp8_storage_t*  o = q   + (size_t)row * width;

    __shared__ float sh_amax[256];
    float amax = 0.f;
    for (int c = threadIdx.x; c < width; c += blockDim.x)
        amax = fmaxf(amax, fabsf(__bfloat162float(s[c])));
    sh_amax[threadIdx.x] = amax;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sh_amax[threadIdx.x] = fmaxf(sh_amax[threadIdx.x], sh_amax[threadIdx.x + stride]);
        __syncthreads();
    }
    float sc = sh_amax[0] / E4B_FP8_E4M3_MAX;
    if (threadIdx.x == 0) scale[row] = sc;
    float inv = (sc > 0.f) ? (1.f / sc) : 0.f;
    for (int c = threadIdx.x; c < width; c += blockDim.x)
        o[c] = e4b_float_to_fp8(__bfloat162float(s[c]) * inv);
}

// Gather + dequant: for each of n requested tokens, write its width-wide
// per-layer-input vector (FP8 row × per-row scale) into out (row-major
// [n * width]). This is the per-token "index" used once per forward pass.
__global__ void e4b_ple_lookup_kernel(
    const __nv_fp8_storage_t* __restrict__ q,     // [vocab * width]
    const float*              __restrict__ scale, // [vocab]
    const int32_t*            __restrict__ tokens,// [n]
    float*                    __restrict__ out,    // [n * width]
    int n, int width)
{
    int i = blockIdx.x;
    if (i >= n) return;
    int t = tokens[i];
    const __nv_fp8_storage_t* row = q + (size_t)t * width;
    float sc = scale[t];
    float* o = out + (size_t)i * width;
    for (int c = threadIdx.x; c < width; c += blockDim.x)
        o[c] = e4b_fp8_to_float(row[c]) * sc;
}

// ── host launch wrappers ──────────────────────────────────────────────────

// Quantize a BF16 table [rows*width] (already on device) into q + scale.
inline void e4b_ple_quantize_launch(
    const __nv_bfloat16* d_src, __nv_fp8_storage_t* d_q, float* d_scale,
    int rows, int width, cudaStream_t stream = 0)
{
    e4b_ple_quantize_kernel<<<rows, 256, 0, stream>>>(d_src, d_q, d_scale, rows, width);
}

inline void e4b_ple_lookup_launch(
    const __nv_fp8_storage_t* d_q, const float* d_scale,
    const int32_t* d_tokens, float* d_out, int n, int width,
    cudaStream_t stream = 0)
{
    e4b_ple_lookup_kernel<<<n, 256, 0, stream>>>(d_q, d_scale, d_tokens, d_out, n, width);
}

#endif // FUCINA_E4B_PLE_FP8_CUH
