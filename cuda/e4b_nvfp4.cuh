// e4b_nvfp4.cuh — NVFP4 weight path foundation for the Gemma-4-E4B engine.
//
// E4B currently holds every projection weight as BF16 and runs the decode GEMVs
// through cuBLAS, so single-token decode is bound on reading the full BF16
// weight footprint each step (measured ~16.8 tok/s). NVFP4 stores the same
// weight at 4.5 bit (E2M1 nibble + one E4M3 block-scale per 16 + a per-tensor
// FP32 scalar) — ~3.6× fewer weight bytes than BF16 — so the bandwidth-bound
// decode GEMV reads far less. This header is the reusable building block the
// engine integration calls; it is validated standalone by test_e4b_nvfp4.cu.
//
// Two pieces:
//   1. e4b_nvfp4_quantize : BF16 weight [out,in] (device) → NVFP4 store, on GPU.
//   2. e4b_nvfp4_gemv_bf16 : y[out] = W_nvfp4 · x[in], bf16 in/out, via the tuned
//      register-blocked nvfp4_gemv_kernel (reused from nvfp4_gemv.cuh).
//
// The on-disk-compatible reconstruction is real = e2m1(nibble)·block_e4m3·gs,
// exactly the convention nvfp4.h / the dense decode GEMV use, so the same kernel
// and host oracle apply unchanged.
#ifndef FUCINA_E4B_NVFP4_CUH
#define FUCINA_E4B_NVFP4_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include "nvfp4.h"   // decode math / host oracle (header-only, all inline/HD — no link symbols)

// Self-contained on purpose: the decode GEMV kernel below is a private copy of the tuned
// register-blocked kernel in nvfp4_gemv.cuh (validated bit-identical in test_e4b_nvfp4.cu).
// We do NOT include nvfp4_gemv.cuh because its __global__ kernels have external linkage, and
// gemma4_kernels.cu already includes it — bundling both .o into libfucina.a would be a
// duplicate-symbol clash. Keeping our kernel `static` (internal linkage) avoids that.
namespace e4bfp4 {

#ifndef E4B_GEMV_WARPS
#define E4B_GEMV_WARPS 8
#endif
#ifndef E4B_GEMV_ROWS
#define E4B_GEMV_ROWS 4
#endif

// device E4M3 byte → float (CUDA intrinsic; matches nvfp4.h's host oracle).
__device__ __forceinline__ float e4m3_dev(uint8_t b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}

// One warp reduces E4B_GEMV_ROWS consecutive output rows; activation loaded once/group and
// reused across rows (L1-hot), each lane issues ROWS independent weight loads to hide latency.
// (Private copy of nvfp4_gemv.cuh's kernel — see header note above.)
static __global__ void e4b_gemv_kernel(
    float* __restrict__ y, const uint8_t* __restrict__ wpacked, const uint8_t* __restrict__ wscale,
    const float* __restrict__ gs, const float* __restrict__ x, int in_dim, int out_dim) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * E4B_GEMV_WARPS + warp) * E4B_GEMV_ROWS;
    if (row0 >= out_dim) return;
    const int nrow = min(E4B_GEMV_ROWS, out_dim - row0);
    const size_t wrowb = (size_t)(in_dim / 2);
    const size_t srowb = (size_t)(in_dim / 16);
    const int ngrp = in_dim / 32;

    const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    auto dec = [&lut](uint32_t nib) -> float { float v = lut[nib & 7u]; return (nib & 8u) ? -v : v; };
    auto mac8 = [&](uint32_t u, const float* xb, float bs) {
        float a = 0.f;
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            uint32_t byte = (u >> (8 * b)) & 0xFFu;
            a += dec(byte & 0xF) * xb[2 * b];
            a += dec(byte >> 4)  * xb[2 * b + 1];
        }
        return a * bs;
    };

    float acc[E4B_GEMV_ROWS];
    #pragma unroll
    for (int r = 0; r < E4B_GEMV_ROWS; r++) acc[r] = 0.f;

    for (int g = lane; g < ngrp; g += 32) {
        const float* xb = x + (size_t)g * 32;
        #pragma unroll
        for (int r = 0; r < E4B_GEMV_ROWS; r++) {
            if (r >= nrow) break;
            const uint8_t* wrow = wpacked + (size_t)(row0 + r) * wrowb;
            const uint8_t* srow = wscale  + (size_t)(row0 + r) * srowb;
            uint4 packed = *reinterpret_cast<const uint4*>(wrow + (size_t)g * 16);
            float bs0 = e4m3_dev(srow[(size_t)g * 2 + 0]);
            float bs1 = e4m3_dev(srow[(size_t)g * 2 + 1]);
            acc[r] += mac8(packed.x, xb + 0,  bs0);
            acc[r] += mac8(packed.y, xb + 8,  bs0);
            acc[r] += mac8(packed.z, xb + 16, bs1);
            acc[r] += mac8(packed.w, xb + 24, bs1);
        }
    }
    const float g_scale = *gs;
    #pragma unroll
    for (int r = 0; r < E4B_GEMV_ROWS; r++) {
        float a = acc[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0 && r < nrow) y[row0 + r] = a * g_scale;
    }
}

static inline void e4b_gemv_launch(float* y, const uint8_t* wpacked, const uint8_t* wscale,
                                   const float* gs, const float* x, int in_dim, int out_dim,
                                   cudaStream_t stream) {
    const int per_blk = E4B_GEMV_WARPS * E4B_GEMV_ROWS;
    unsigned blocks = (unsigned)((out_dim + per_blk - 1) / per_blk);
    e4b_gemv_kernel<<<blocks, 32 * E4B_GEMV_WARPS, 0, stream>>>(y, wpacked, wscale, gs, x, in_dim, out_dim);
}

// A quantized NVFP4 weight, device-resident. Layout matches nvfp4.h exactly:
//   packed : U8 [out_dim][in_dim/2]   two E2M1 nibbles/byte (low=even k, high=odd k)
//   scale  : E4M3 [out_dim][in_dim/16] LINEAR block scales
//   gs     : device FP32 scalar = weight_scale_2 (the decode multiplier)
struct Weight {
    uint8_t* packed = nullptr;
    uint8_t* scale  = nullptr;
    float*   gs     = nullptr;
    int      in_dim = 0;
    int      out_dim = 0;
    uint64_t bytes  = 0;
};

inline void weight_free(Weight* w) {
    if (!w) return;
    if (w->packed) cudaFree(w->packed);
    if (w->scale)  cudaFree(w->scale);
    if (w->gs)     cudaFree(w->gs);
    w->packed = w->scale = nullptr; w->gs = nullptr; w->bytes = 0;
}

// ── E2M1 nearest-encode: float → 4-bit nibble (bit3 = sign, bits2..0 = magnitude code). ──
// Magnitude LUT {0,.5,1,1.5,2,3,4,6}; thresholds are the midpoints between them.
__device__ __forceinline__ uint8_t e2m1_encode(float v) {
    float a = fabsf(v);
    uint8_t code;
    if      (a < 0.25f) code = 0;
    else if (a < 0.75f) code = 1;
    else if (a < 1.25f) code = 2;
    else if (a < 1.75f) code = 3;
    else if (a < 2.5f)  code = 4;
    else if (a < 3.5f)  code = 5;
    else if (a < 5.0f)  code = 6;
    else                code = 7;
    return (v < 0.0f) ? (uint8_t)(code | 0x8u) : code;
}

// Per output-row amax over a BF16 weight row [in_dim] → rowamax[row]. One block/row.
__global__ void rowamax_kernel(const __nv_bfloat16* __restrict__ W,
                               float* __restrict__ rowamax, int in_dim) {
    int row = blockIdx.x;
    const __nv_bfloat16* wr = W + (size_t)row * in_dim;
    float a = 0.f;
    for (int c = threadIdx.x; c < in_dim; c += blockDim.x)
        a = fmaxf(a, fabsf(__bfloat162float(wr[c])));
    __shared__ float s[256];
    s[threadIdx.x] = a; __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (threadIdx.x < o) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + o]);
        __syncthreads();
    }
    if (threadIdx.x == 0) rowamax[row] = s[0];
}

// Reduce rowamax[out_dim] → gs = global_amax / (6*448) (the E4M3 block scales then
// fit, since block_scale = amax_block/6 / gs ≤ amax_tensor/6 / gs = 448). One block.
__global__ void global_scale_kernel(const float* __restrict__ rowamax, int out_dim,
                                    float* __restrict__ gs) {
    float a = 0.f;
    for (int i = threadIdx.x; i < out_dim; i += blockDim.x) a = fmaxf(a, rowamax[i]);
    __shared__ float s[256];
    s[threadIdx.x] = a; __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (threadIdx.x < o) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + o]);
        __syncthreads();
    }
    if (threadIdx.x == 0) { float amax = s[0]; *gs = (amax > 0.f) ? amax / (6.f * 448.f) : 1.f; }
}

// Quantize one 16-element block per thread: compute the E4M3 block scale, then encode
// the 16 elements (faithfully — dividing by the DECODED block scale, as decode will use it).
__global__ void quant_kernel(const __nv_bfloat16* __restrict__ W, const float* __restrict__ gs,
                             uint8_t* __restrict__ packed, uint8_t* __restrict__ scale,
                             int in_dim) {
    int row  = blockIdx.x;
    int nblk = in_dim >> 4;
    int b    = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= nblk) return;

    const __nv_bfloat16* wr = W + (size_t)row * in_dim + (size_t)b * 16;
    float g = *gs;
    float vals[16], amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 16; i++) { float v = __bfloat162float(wr[i]); vals[i] = v; amax = fmaxf(amax, fabsf(v)); }

    // block scale (float) = amax/6 ; store quantized to E4M3 of (block_scale / gs)
    float ratio = (g > 0.f) ? (amax / 6.0f) / g : 0.f;
    __nv_fp8_storage_t e = __nv_cvt_float_to_fp8(ratio, __NV_SATFINITE, __NV_E4M3);
    scale[(size_t)row * nblk + b] = (uint8_t)e;

    // faithful element encode: divide by the DECODED block scale × gs (what decode reconstructs with)
    float bsd   = __half2float(__half(__nv_cvt_fp8_to_halfraw(e, __NV_E4M3)));
    float denom = bsd * g;
    float inv   = (denom > 0.f) ? 1.0f / denom : 0.f;
    uint8_t* pr = packed + (size_t)row * (in_dim >> 1) + (size_t)b * 8;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint8_t lo = e2m1_encode(vals[2 * j]     * inv);
        uint8_t hi = e2m1_encode(vals[2 * j + 1] * inv);
        pr[j] = (uint8_t)(lo | (hi << 4));
    }
}

// Quantize a device BF16 weight [out_dim, in_dim] (row-major) into w (allocates packed/scale/gs).
// in_dim must be a multiple of 16 (true for all E4B projection dims). Returns false on alloc/launch
// error. Synchronous w.r.t. allocation; the kernels run on the default stream.
inline bool e4b_nvfp4_quantize(const __nv_bfloat16* d_W, int out_dim, int in_dim, Weight* w) {
    if (!w || in_dim <= 0 || out_dim <= 0 || (in_dim & 15)) return false;
    w->in_dim = in_dim; w->out_dim = out_dim;
    size_t pbytes = (size_t)out_dim * (in_dim / 2);
    size_t sbytes = (size_t)out_dim * (in_dim / 16);
    float* d_rowamax = nullptr;
    if (cudaMalloc(&w->packed, pbytes) != cudaSuccess) return false;
    if (cudaMalloc(&w->scale,  sbytes) != cudaSuccess) { weight_free(w); return false; }
    if (cudaMalloc(&w->gs, sizeof(float)) != cudaSuccess) { weight_free(w); return false; }
    if (cudaMalloc(&d_rowamax, (size_t)out_dim * sizeof(float)) != cudaSuccess) { weight_free(w); return false; }

    rowamax_kernel<<<out_dim, 256>>>(d_W, d_rowamax, in_dim);
    global_scale_kernel<<<1, 256>>>(d_rowamax, out_dim, w->gs);
    dim3 grid(out_dim, ((in_dim / 16) + 255) / 256);
    quant_kernel<<<grid, 256>>>(d_W, w->gs, w->packed, w->scale, in_dim);
    cudaFree(d_rowamax);

    w->bytes = pbytes + sbytes + sizeof(float);
    return cudaGetLastError() == cudaSuccess;
}

// ── bf16 GEMV wrapper around the tuned register-blocked NVFP4 kernel ──
// The kernel reads/writes float, so x[in] is converted bf16→f32 and y[out] f32→bf16. xf[in] and
// yf[out] are caller-provided scratch (kept off the per-call malloc path the engine forward uses).
__global__ void to_f32(const __nv_bfloat16* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = __bfloat162float(x[i]);
}
__global__ void to_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = __float2bfloat16(x[i]);
}

inline void e4b_nvfp4_gemv_bf16(__nv_bfloat16* y, const Weight& w, const __nv_bfloat16* x,
                                float* xf, float* yf, cudaStream_t stream) {
    to_f32<<<(w.in_dim + 255) / 256, 256, 0, stream>>>(x, xf, w.in_dim);
    e4b_gemv_launch(yf, w.packed, w.scale, w.gs, xf, w.in_dim, w.out_dim, stream);
    to_bf16<<<(w.out_dim + 255) / 256, 256, 0, stream>>>(yf, y, w.out_dim);
}

// ════════════════════════════════════════════════════════════════════════════
// FP8 (E4M3) per-row-scaled weight — the HIGHER-PRECISION quant for the routing/
// decision projections (attention Q/K "index"; the output LM head). 8-bit weight =
// 2× NVFP4's precision, still half BF16's bandwidth. Reconstruct: real =
// e4m3(q[o,k]) · rowscale[o], with rowscale[o] = amax(row)/448 (symmetric per-row).
// Used where NVFP4's 4-bit would perturb the softmax/argmax too much.
// ════════════════════════════════════════════════════════════════════════════
struct Fp8Weight {
    __nv_fp8_storage_t* q  = nullptr;   // [out_dim][in_dim] E4M3
    float*              rs = nullptr;   // [out_dim] per-row scale
    int in_dim = 0, out_dim = 0;
    uint64_t bytes = 0;
};

inline void fp8_weight_free(Fp8Weight* w) {
    if (!w) return;
    if (w->q)  cudaFree(w->q);
    if (w->rs) cudaFree(w->rs);
    w->q = nullptr; w->rs = nullptr; w->bytes = 0;
}

// Per-row absmax → E4M3 cast. One block per output row.
__global__ void fp8_quant_kernel(const __nv_bfloat16* __restrict__ W,
                                 __nv_fp8_storage_t* __restrict__ q,
                                 float* __restrict__ rs, int in_dim) {
    int row = blockIdx.x;
    const __nv_bfloat16* wr = W + (size_t)row * in_dim;
    float a = 0.f;
    for (int c = threadIdx.x; c < in_dim; c += blockDim.x) a = fmaxf(a, fabsf(__bfloat162float(wr[c])));
    __shared__ float s[256];
    s[threadIdx.x] = a; __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (threadIdx.x < o) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + o]);
        __syncthreads();
    }
    float scale = s[0] * (1.0f / 448.0f);
    if (!(scale > 0.f)) scale = 1.f;
    if (threadIdx.x == 0) rs[row] = scale;
    float inv = 1.0f / scale;
    __nv_fp8_storage_t* qr = q + (size_t)row * in_dim;
    for (int c = threadIdx.x; c < in_dim; c += blockDim.x)
        qr[c] = __nv_cvt_float_to_fp8(__bfloat162float(wr[c]) * inv, __NV_SATFINITE, __NV_E4M3);
}

// Register-blocked FP8 GEMV: one warp reduces E4B_GEMV_ROWS rows; 16 fp8 (one uint4) per group.
static __global__ void e4b_fp8_gemv_kernel(
    float* __restrict__ y, const __nv_fp8_storage_t* __restrict__ q, const float* __restrict__ rs,
    const float* __restrict__ x, int in_dim, int out_dim) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * E4B_GEMV_WARPS + warp) * E4B_GEMV_ROWS;
    if (row0 >= out_dim) return;
    const int nrow = min(E4B_GEMV_ROWS, out_dim - row0);
    const int ngrp = in_dim / 16;

    float acc[E4B_GEMV_ROWS];
    #pragma unroll
    for (int r = 0; r < E4B_GEMV_ROWS; r++) acc[r] = 0.f;

    for (int g = lane; g < ngrp; g += 32) {
        const float* xb = x + (size_t)g * 16;
        #pragma unroll
        for (int r = 0; r < E4B_GEMV_ROWS; r++) {
            if (r >= nrow) break;
            const __nv_fp8_storage_t* qr = q + (size_t)(row0 + r) * in_dim;
            uint4 w4 = *reinterpret_cast<const uint4*>(qr + (size_t)g * 16);
            const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w4);
            float a = 0.f;
            #pragma unroll
            for (int j = 0; j < 16; j++) a += e4m3_dev(wb[j]) * xb[j];
            acc[r] += a;
        }
    }
    #pragma unroll
    for (int r = 0; r < E4B_GEMV_ROWS; r++) {
        float a = acc[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0 && r < nrow) y[row0 + r] = a * rs[row0 + r];
    }
}

inline bool e4b_fp8_quantize(const __nv_bfloat16* d_W, int out_dim, int in_dim, Fp8Weight* w) {
    if (!w || in_dim <= 0 || out_dim <= 0 || (in_dim & 15)) return false;
    w->in_dim = in_dim; w->out_dim = out_dim;
    size_t qb = (size_t)out_dim * in_dim;
    if (cudaMalloc(&w->q, qb) != cudaSuccess) return false;
    if (cudaMalloc(&w->rs, (size_t)out_dim * sizeof(float)) != cudaSuccess) { fp8_weight_free(w); return false; }
    fp8_quant_kernel<<<out_dim, 256>>>(d_W, w->q, w->rs, in_dim);
    w->bytes = qb + (size_t)out_dim * sizeof(float);
    return cudaGetLastError() == cudaSuccess;
}

// FP8 GEMV into a FLOAT output (yf is the result; no bf16 cast). Used directly by the logits head
// (writes fp32 logits) and wrapped below for the bf16 attention projections.
inline void e4b_fp8_gemv_f32(float* yf, const Fp8Weight& w, const __nv_bfloat16* x, float* xf,
                             cudaStream_t stream) {
    to_f32<<<(w.in_dim + 255) / 256, 256, 0, stream>>>(x, xf, w.in_dim);
    const int per = E4B_GEMV_WARPS * E4B_GEMV_ROWS;
    unsigned blocks = (unsigned)((w.out_dim + per - 1) / per);
    e4b_fp8_gemv_kernel<<<blocks, 32 * E4B_GEMV_WARPS, 0, stream>>>(yf, w.q, w.rs, xf, w.in_dim, w.out_dim);
}

inline void e4b_fp8_gemv_bf16(__nv_bfloat16* y, const Fp8Weight& w, const __nv_bfloat16* x,
                              float* xf, float* yf, cudaStream_t stream) {
    e4b_fp8_gemv_f32(yf, w, x, xf, stream);
    to_bf16<<<(w.out_dim + 255) / 256, 256, 0, stream>>>(yf, y, w.out_dim);
}

} // namespace e4bfp4

#endif // FUCINA_E4B_NVFP4_CUH
