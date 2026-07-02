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

#define FP8_MAXB 16   // GEMMA4_MAX_SEQS — max rows a batched decode/prefill-chunk step carries

// BATCHED: out[B][out_dim] = W[out_dim,in_dim] · X[B][in_dim], W FP8 block-scaled. Warp-per-row;
// the weight bytes are read ONCE per 128-block and reused across the B activation rows (the
// bandwidth amortization the dp4a batched kernels rely on). Static grid → CUDA-graph-capturable.
static __global__ void fp8_block_gemm_kernel(
    float *out, const uint8_t *w, const __nv_bfloat16 *wscale,
    const float *x, int in_dim, int out_dim, int B)
{
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;          // output row
    if (idx >= out_dim) return;
    int nblk = in_dim / FP8BLK, sstride = in_dim / FP8BLK, srow = idx / FP8BLK;
    const uint8_t *wrow = w + (size_t)idx * in_dim;
    float acc[FP8_MAXB];
    #pragma unroll
    for (int b = 0; b < FP8_MAXB; b++) acc[b] = 0.0f;
    for (int ib = 0; ib < nblk; ib++) {
        float bs = __bfloat162float(wscale[(size_t)srow * sstride + ib]);
        float wv[4];                               // this lane's 4 fp8 weights for the block
        #pragma unroll
        for (int t = 0; t < 4; t++) { __nv_fp8_e4m3 wb; wb.__x = wrow[ib*FP8BLK + lane + t*32]; wv[t] = float(wb); }
        for (int b = 0; b < B; b++) {
            const float *xb = x + (size_t)b * in_dim;
            float p = 0.0f;
            #pragma unroll
            for (int t = 0; t < 4; t++) p += wv[t] * xb[ib*FP8BLK + lane + t*32];
            acc[b] += bs * p;
        }
    }
    for (int b = 0; b < B; b++) {
        float a = acc[b];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xFFFFFFFFu, a, o);
        if (lane == 0) out[(size_t)b * out_dim + idx] = a;
    }
}

static inline void fp8_block_gemm_launch(
    float *out, const uint8_t *w, const __nv_bfloat16 *wscale,
    const float *x, int in_dim, int out_dim, int B, cudaStream_t stream)
{
    const int WPB = 4;
    int blocks = (out_dim + WPB - 1) / WPB;
    fp8_block_gemm_kernel<<<blocks, WPB * 32, 0, stream>>>(out, w, wscale, x, in_dim, out_dim, B);
}

// GROUPED (MoE): out[total][out_dim] = per-expert W_e · X, where the `total` rows of X are grouped
// expert-contiguously — expert e owns rows [coloff[e], coloff[e]+count[e]) and uses the FP8 weight
// slab wbase + e*w_stride with block-scale sbase + e*s_stride. Mirrors dg_mmq_*_grouped but the
// activation is FLOAT (no Q8_1 quant) and the weight is block-FP8. One block per (out-row-tile,
// active-expert slot); the expert's tokens are processed in FP8_MAXB-row chunks (weight read once
// per chunk). `active` (optional) maps grid slot → expert id (-1 pads) so a decode-sized grid of
// n_slot=B·topk blocks replaces the full-E grid (E=256 → 32× fewer blocks at B=1). Static grid
// (n_slot known per B) → CUDA-graph-capturable. in_dim/out_dim multiples of 128.
static __global__ void fp8_block_gemm_grouped_kernel(
    float *out, const uint8_t *wbase, int64_t w_stride, const __nv_bfloat16 *sbase, int64_t s_stride,
    const float *x, const int *coloff, const int *count, const int *active, int in_dim, int out_dim)
{
    int e = active ? active[blockIdx.y] : (int)blockIdx.y;
    if (e < 0) return;
    int cnt = count[e];
    if (cnt <= 0) return;
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;          // output row
    if (idx >= out_dim) return;
    int off = coloff[e];
    const uint8_t     *w  = wbase + (size_t)e * w_stride;
    const __nv_bfloat16 *sc = sbase + (size_t)e * s_stride;
    const float *xe = x   + (size_t)off * in_dim;
    float       *oe = out + (size_t)off * out_dim;
    int nblk = in_dim / FP8BLK, sstride = in_dim / FP8BLK, srow = idx / FP8BLK;
    const uint8_t *wrow = w + (size_t)idx * in_dim;
    for (int b0 = 0; b0 < cnt; b0 += FP8_MAXB) {
        int B = (cnt - b0 < FP8_MAXB) ? (cnt - b0) : FP8_MAXB;
        float acc[FP8_MAXB];
        #pragma unroll
        for (int b = 0; b < FP8_MAXB; b++) acc[b] = 0.0f;
        for (int ib = 0; ib < nblk; ib++) {
            float bs = __bfloat162float(sc[(size_t)srow * sstride + ib]);
            float wv[4];
            #pragma unroll
            for (int t = 0; t < 4; t++) { __nv_fp8_e4m3 wb; wb.__x = wrow[ib*FP8BLK + lane + t*32]; wv[t] = float(wb); }
            for (int b = 0; b < B; b++) {
                const float *xb = xe + (size_t)(b0 + b) * in_dim;
                float p = 0.0f;
                #pragma unroll
                for (int t = 0; t < 4; t++) p += wv[t] * xb[ib*FP8BLK + lane + t*32];
                acc[b] += bs * p;
            }
        }
        for (int b = 0; b < B; b++) {
            float a = acc[b];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xFFFFFFFFu, a, o);
            if (lane == 0) oe[(size_t)(b0 + b) * out_dim + idx] = a;
        }
    }
}

// active/n_slot: optional compacted active-expert list (grid.y = n_slot); active=NULL → grid.y =
// n_expert with slot==expert (the full-E reference behavior, bitwise-identical per expert).
static inline void fp8_block_gemm_grouped_launch(
    float *out, const uint8_t *wbase, int64_t w_stride, const __nv_bfloat16 *sbase, int64_t s_stride,
    const float *x, const int *coloff, const int *count, const int *active, int n_slot,
    int n_expert, int in_dim, int out_dim, cudaStream_t stream)
{
    const int WPB = 4;
    dim3 grid((out_dim + WPB - 1) / WPB, active ? n_slot : n_expert);
    fp8_block_gemm_grouped_kernel<<<grid, WPB * 32, 0, stream>>>(
        out, wbase, w_stride, sbase, s_stride, x, coloff, count, active, in_dim, out_dim);
}

#endif // FUCINA_FP8_BLOCK_CUH
