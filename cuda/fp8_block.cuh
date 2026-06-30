// fp8_block.cuh — DeepSeek-style FP8 block-scaled decode GEMV, for Qwen3.5/3.6 FP8
// checkpoints (qwen3_5 / qwen3_5_moe). Weights are F8_E4M3 [out_dim][in_dim] row-major;
// the dequant scale is per 128x128 weight block: weight_scale_inv BF16 [out_dim/128][in_dim/128].
//   W_real[o][i] = fp8_e4m3(W[o][i]) * scale[o/128][i/128]
//   out[o] = Σ_i W_real[o][i]·x[i] = Σ_ib scale[o/128][ib] · Σ_{i∈ib} fp8(W[o][i])·x[i]
// Warp-per-row; in_dim/out_dim must be multiples of 128 (true for Qwen3.5-MoE: hidden 2048,
// q 8192, kv 512, moe_intermediate 512, all /128). Activation x is float (residual stream).
#ifndef FUCINA_FP8_BLOCK_CUH
#define FUCINA_FP8_BLOCK_CUH
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define FP8BLK 128

static __global__ void fp8_block_gemv_kernel(
    float *out, const uint8_t *w, const __nv_bfloat16 *wscale,
    const float *x, int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;          // output row
    if (idx >= out_dim) return;
    int nblk    = in_dim / FP8BLK;                  // i-blocks of 128
    int sstride = in_dim / FP8BLK;                  // scale row stride
    int srow    = idx / FP8BLK;                     // this row's o-block
    const uint8_t *wrow = w + (size_t)idx * in_dim;
    float acc = 0.0f;
    for (int ib = 0; ib < nblk; ib++) {
        float bs = __bfloat162float(wscale[(size_t)srow * sstride + ib]);
        float p = 0.0f;                             // Σ fp8(W)·x within this 128-block
        #pragma unroll
        for (int t = 0; t < FP8BLK; t += 32) {
            int i = ib * FP8BLK + lane + t;         // 4 strided elems per lane per block
            __nv_fp8_e4m3 wb; wb.__x = wrow[i];
            p += float(wb) * x[i];
        }
        acc += bs * p;                              // apply the block scale, accumulate
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
    if (lane == 0) out[idx] = acc;
}

// out[out_dim] = W[out_dim,in_dim] · x[in_dim], W in FP8 block-scaled.
static inline void fp8_block_gemv_launch(
    float *out, const uint8_t *w, const __nv_bfloat16 *wscale,
    const float *x, int in_dim, int out_dim, cudaStream_t stream)
{
    const int WPB = 4;
    int blocks = (out_dim + WPB - 1) / WPB;
    fp8_block_gemv_kernel<<<blocks, WPB * 32, 0, stream>>>(out, w, wscale, x, in_dim, out_dim);
}

#endif // FUCINA_FP8_BLOCK_CUH
