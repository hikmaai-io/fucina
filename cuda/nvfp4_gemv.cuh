// NVFP4 decode GEMV — the bandwidth-bound single-token (B=1) projection for an NVFP4 weight
// store. This is the decode-path counterpart to the cuBLASLt block-scaled GEMM used for
// prefill: at N=1 the projection is a GEMV (arithmetic intensity ~1), so the FP4 tensor cores
// give nothing — the lever is reading the 4.5-bit weight footprint ONCE, coalesced, and
// dequantizing in-register. One warp per output row; each lane streams 32-weight (uint4)
// chunks of the packed E2M1 row, applies the two E4M3 block scales covering them, FMAs against
// the activation, warp-reduces, and folds in the per-tensor global scale.
//
//   y[out] = ( Σ_k e2m1(W[out,k]) · block_scale_e4m3[out, k/16] · x[k] ) · weight_scale_2
//
// Weights are the ModelOpt NVFP4 layout loaded verbatim (see nvfp4.h / safetensors.h):
//   wpacked : U8 [out_dim, in_dim/2]   two nibbles/byte, low=even k, high=odd k
//   wscale  : E4M3 [out_dim, in_dim/16] LINEAR (not swizzled) block scales
//   gs      : device scalar weight_scale_2
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

// One warp (32 lanes) per output row, WARPS_PER_BLK warps/block for occupancy. blockDim.x must
// be 32*WARPS_PER_BLK; gridDim.x = ceil(out_dim / WARPS_PER_BLK). The activation x (≤60 KB) is
// re-read by every row but stays L2-resident, so DRAM traffic is the weight footprint.
#ifndef NVFP4_GEMV_WARPS
#define NVFP4_GEMV_WARPS 8
#endif
__global__ void nvfp4_gemv_kernel(
    float* __restrict__ y,
    const uint8_t* __restrict__ wpacked,   // [out_dim][in_dim/2]
    const uint8_t* __restrict__ wscale,    // [out_dim][in_dim/16] E4M3 linear
    const float*   __restrict__ gs,        // device scalar (weight_scale_2)
    const float*   __restrict__ x,         // [in_dim] activation
    int in_dim, int out_dim)
{
    const int warp = threadIdx.x >> 5;                  // 0..WARPS-1
    const int lane = threadIdx.x & 31;                  // 0..31
    const int row  = blockIdx.x * NVFP4_GEMV_WARPS + warp;
    if (row >= out_dim) return;

    const uint8_t* wrow = wpacked + (size_t)row * (in_dim / 2);
    const uint8_t* srow = wscale  + (size_t)row * (in_dim / 16);
    const int ngrp = in_dim / 32;                       // 32-weight (16-byte) groups

    // E2M1 magnitude LUT in registers (signed via bit3). Kept local so the compiler keeps it
    // in registers/constant cache rather than re-deriving per element.
    const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    auto dec = [&lut](uint32_t nib) -> float { float v = lut[nib & 7u]; return (nib & 8u) ? -v : v; };
    // Accumulate 8 weights packed in one uint32 `u` against x[xb..xb+8) with block scale bs.
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

    float acc = 0.f;
    for (int g = lane; g < ngrp; g += 32) {
        // 16 packed bytes = 32 weights, one coalesced 128-bit load (stays in registers)
        uint4 packed = *reinterpret_cast<const uint4*>(wrow + (size_t)g * 16);
        // two E4M3 block scales covering these 32 weights (blocks 2g, 2g+1); .x/.y → bs0, .z/.w → bs1
        float bs0 = nvfp4_e4m3_dev(srow[(size_t)g * 2 + 0]);
        float bs1 = nvfp4_e4m3_dev(srow[(size_t)g * 2 + 1]);
        const float* xb = x + (size_t)g * 32;
        acc += mac8(packed.x, xb + 0,  bs0);
        acc += mac8(packed.y, xb + 8,  bs0);
        acc += mac8(packed.z, xb + 16, bs1);
        acc += mac8(packed.w, xb + 24, bs1);
    }
    // warp reduce
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc * (*gs);
}

// Launch: one warp per output row. Stream is engine/graph-supplied.
static inline void nvfp4_gemv_launch(
    float* y, const uint8_t* wpacked, const uint8_t* wscale, const float* gs,
    const float* x, int in_dim, int out_dim, cudaStream_t stream)
{
    unsigned blocks = (unsigned)((out_dim + NVFP4_GEMV_WARPS - 1) / NVFP4_GEMV_WARPS);
    nvfp4_gemv_kernel<<<blocks, 32 * NVFP4_GEMV_WARPS, 0, stream>>>(
        y, wpacked, wscale, gs, x, in_dim, out_dim);
}

#endif // FUCINA_NVFP4_GEMV_CUH
