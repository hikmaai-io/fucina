// NVFP4 decode GEMV — the bandwidth-bound single-token (B=1) projection for the NVFP4 weight
// store. This is the decode-path counterpart to the cuBLASLt block-scaled GEMM used for prefill:
// at N=1 the projection is a GEMV (arithmetic intensity ~1), so the FP4 tensor cores give nothing
// — the only lever is reading the 4.5-bit weight footprint ONCE, coalesced, at full LPDDR5X
// bandwidth, and dequantizing in-register. Since NVFP4 weighs exactly what Q4_0 does (4 bit E2M1
// + 8-bit E4M3 per 16 = 4.5 bit, same as Q4_0's 4 bit + fp16 per 32), the bar is the tuned dp4a
// Q4_0 decode (~125–139 GB/s on GB10): match it, and the single-store NVFP4 path costs no decode
// speed while saving the ~6 GB Q4_0 duplicate.
//
//   y[out] = ( Σ_k e2m1(W[out,k]) · block_scale_e4m3[out, k/16] · x[k] ) · weight_scale_2
//
// Geometry: register-blocking on the OUTPUT rows is what gets us there. A naive warp-per-row
// kernel stalls at ~66 GB/s — one uint4 weight load then a long dequant+FMA chain before the next
// load starves the memory pipe (latency-bound, not throughput-bound; confirmed: a shared E4M3 LUT
// that cut the per-byte ALU did nothing). Instead each warp reduces NVFP4_GEMV_ROWS (=4)
// consecutive rows: every lane keeps 4 accumulators and issues 4 INDEPENDENT weight loads per
// 32-weight group (4× the in-flight memory traffic to hide latency) while the activation x is
// loaded once per group and reused across all 4 rows (L1-hot). Measured on GB10 sm_121a:
// 152 GB/s (3840×3840 q), 176 (3840×15360 down), 193 (15360×3840 gate/up) — at/above dp4a parity.
// ROWS=8 regresses down_proj (register pressure); ROWS=2 leaves ~30% on the table.
//
// Weights are the NVFP4 layout loaded verbatim (see nvfp4.h / safetensors.h):
//   wpacked : U8 [out_dim, in_dim/2]   two nibbles/byte, low=even k, high=odd k
//   wscale  : E4M3 [out_dim, in_dim/16] LINEAR (not swizzled — the prefill swizzle is cuBLASLt-only)
//   gs      : device scalar = the normalized global decode multiplier (nvfp4ld::global_mul)
// Requires in_dim % 32 == 0 (true for all Gemma-4 projection dims) and 16-B-aligned rows
// (cudaMalloc + in_dim/2 multiple of 16 ⇒ satisfied). CUDA-graph-capturable: no allocation,
// no host sync, scalar global read from device memory.
#ifndef FUCINA_NVFP4_GEMV_CUH
#define FUCINA_NVFP4_GEMV_CUH

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include "nvfp4.h"   // nvfp4::e2m1_decode (host+device)

// device E4M3 byte → float, via the CUDA conversion intrinsic (matches the engine's existing
// NVFP4 code exactly; the software nvfp4::e4m3_decode is the host oracle).
__device__ __forceinline__ float nvfp4_e4m3_dev(uint8_t b) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3)));
}

// One warp (32 lanes) cooperatively reduces NVFP4_GEMV_ROWS consecutive output rows, with
// NVFP4_GEMV_WARPS warps/block. Register-blocking on the output rows is the key lever at N=1:
// each lane keeps ROWS accumulators and issues ROWS independent weight loads per group (more
// memory-level parallelism to hide latency), while the activation x is loaded ONCE per group and
// reused across all ROWS rows (L1-hot). gridDim.x = ceil(out_dim / (WARPS*ROWS)).
#ifndef NVFP4_GEMV_WARPS
#define NVFP4_GEMV_WARPS 8
#endif
#ifndef NVFP4_GEMV_ROWS
#define NVFP4_GEMV_ROWS 4
#endif
__global__ void nvfp4_gemv_kernel(
    float* __restrict__ y,
    const uint8_t* __restrict__ wpacked,   // [out_dim][in_dim/2]
    const uint8_t* __restrict__ wscale,    // [out_dim][in_dim/16] E4M3 linear
    const float*   __restrict__ gs,        // device scalar (weight_scale_2)
    const float*   __restrict__ x,         // [in_dim] activation
    int in_dim, int out_dim)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * NVFP4_GEMV_WARPS + warp) * NVFP4_GEMV_ROWS;
    if (row0 >= out_dim) return;
    const int nrow = min(NVFP4_GEMV_ROWS, out_dim - row0);   // tail block may be partial

    const size_t wrowb = (size_t)(in_dim / 2);
    const size_t srowb = (size_t)(in_dim / 16);
    const int ngrp = in_dim / 32;

    const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    auto dec = [&lut](uint32_t nib) -> float { float v = lut[nib & 7u]; return (nib & 8u) ? -v : v; };
    // 8 weights (one uint32) · x[xb..xb+8), scaled by bs, into the running accumulator.
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

    float acc[NVFP4_GEMV_ROWS];
    #pragma unroll
    for (int r = 0; r < NVFP4_GEMV_ROWS; r++) acc[r] = 0.f;

    for (int g = lane; g < ngrp; g += 32) {
        const float* xb = x + (size_t)g * 32;
        #pragma unroll
        for (int r = 0; r < NVFP4_GEMV_ROWS; r++) {
            if (r >= nrow) break;
            const uint8_t* wrow = wpacked + (size_t)(row0 + r) * wrowb;
            const uint8_t* srow = wscale  + (size_t)(row0 + r) * srowb;
            uint4 packed = *reinterpret_cast<const uint4*>(wrow + (size_t)g * 16);
            float bs0 = nvfp4_e4m3_dev(srow[(size_t)g * 2 + 0]);
            float bs1 = nvfp4_e4m3_dev(srow[(size_t)g * 2 + 1]);
            acc[r] += mac8(packed.x, xb + 0,  bs0);
            acc[r] += mac8(packed.y, xb + 8,  bs0);
            acc[r] += mac8(packed.z, xb + 16, bs1);
            acc[r] += mac8(packed.w, xb + 24, bs1);
        }
    }
    const float g_scale = *gs;
    #pragma unroll
    for (int r = 0; r < NVFP4_GEMV_ROWS; r++) {
        float a = acc[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0 && r < nrow) y[row0 + r] = a * g_scale;
    }
}

// Launch: one warp per NVFP4_GEMV_ROWS output rows.
static inline void nvfp4_gemv_launch(
    float* y, const uint8_t* wpacked, const uint8_t* wscale, const float* gs,
    const float* x, int in_dim, int out_dim, cudaStream_t stream)
{
    const int per_blk = NVFP4_GEMV_WARPS * NVFP4_GEMV_ROWS;
    unsigned blocks = (unsigned)((out_dim + per_blk - 1) / per_blk);
    nvfp4_gemv_kernel<<<blocks, 32 * NVFP4_GEMV_WARPS, 0, stream>>>(
        y, wpacked, wscale, gs, x, in_dim, out_dim);
}

#endif // FUCINA_NVFP4_GEMV_CUH
