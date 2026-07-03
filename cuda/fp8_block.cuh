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
        // lane owns 4 CONTIGUOUS elems: one uchar4 weight load + one float4 activation load
        // (4x fewer load instrs than the byte-strided form; wrow is 32B-aligned and in_dim a
        // multiple of 128, so both vector loads are aligned).
        uchar4 wq = *((const uchar4 *)(wrow + ib * FP8BLK) + lane);
        float4 xv = *((const float4 *)(x + ib * FP8BLK) + lane);
        __nv_fp8_e4m3 w0, w1, w2, w3; w0.__x = wq.x; w1.__x = wq.y; w2.__x = wq.z; w3.__x = wq.w;
        float p = float(w0)*xv.x + float(w1)*xv.y + float(w2)*xv.z + float(w3)*xv.w;
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
#define FP8_GMAXB 8   // grouped-kernel chunk: decode has ~1-2 tokens/expert, so 8 covers them
                      // in one pass at half the acc-register pressure of FP8_MAXB (occupancy)

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
        uchar4 wq = *((const uchar4 *)(wrow + ib * FP8BLK) + lane);   // 4 contiguous elems/lane
        float wv[4];
        { __nv_fp8_e4m3 w0, w1, w2, w3; w0.__x = wq.x; w1.__x = wq.y; w2.__x = wq.z; w3.__x = wq.w;
          wv[0] = float(w0); wv[1] = float(w1); wv[2] = float(w2); wv[3] = float(w3); }
        for (int b = 0; b < B; b++) {
            float4 xv = *((const float4 *)(x + (size_t)b * in_dim + ib * FP8BLK) + lane);
            acc[b] += bs * (wv[0]*xv.x + wv[1]*xv.y + wv[2]*xv.z + wv[3]*xv.w);
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
    for (int b0 = 0; b0 < cnt; b0 += FP8_GMAXB) {
        int B = (cnt - b0 < FP8_GMAXB) ? (cnt - b0) : FP8_GMAXB;
        float acc[FP8_GMAXB];
        #pragma unroll
        for (int b = 0; b < FP8_GMAXB; b++) acc[b] = 0.0f;
        for (int ib = 0; ib < nblk; ib++) {
            float bs = __bfloat162float(sc[(size_t)srow * sstride + ib]);
            uchar4 wq = *((const uchar4 *)(wrow + ib * FP8BLK) + lane);   // 4 contiguous elems/lane
            float wv[4];
            { __nv_fp8_e4m3 w0, w1, w2, w3; w0.__x = wq.x; w1.__x = wq.y; w2.__x = wq.z; w3.__x = wq.w;
              wv[0] = float(w0); wv[1] = float(w1); wv[2] = float(w2); wv[3] = float(w3); }
            for (int b = 0; b < B; b++) {
                float4 xv = *((const float4 *)(xe + (size_t)(b0 + b) * in_dim + ib * FP8BLK) + lane);
                acc[b] += bs * (wv[0]*xv.x + wv[1]*xv.y + wv[2]*xv.z + wv[3]*xv.w);
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

// FUSED grouped gate+up+SiLU (MoE decode slice 1): one launch computes act = silu(gate(x))*up(x)
// for every (expert, token) assignment — the gate and up slabs are indexed identically, so one
// block computes BOTH projections for its (out-tile, active-expert slot), reading the activation
// row once and writing silu(g)*u directly (no d_moe_gate/d_moe_up round-trip, no dg_silu_mul).
// Same math order per projection as fp8_block_gemm_grouped_kernel. Static grid → graph-safe.
static __global__ void fp8_block_gemm_grouped_gateup_silu_kernel(
    float *act, const uint8_t *gbase, const uint8_t *ubase, int64_t w_stride,
    const __nv_bfloat16 *gsbase, const __nv_bfloat16 *usbase, int64_t s_stride,
    const float *x, const int *coloff, const int *count, const int *active,
    int in_dim, int out_dim)
{
    int e = active ? active[blockIdx.y] : (int)blockIdx.y;
    if (e < 0) return;
    int cnt = count[e];
    if (cnt <= 0) return;
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int off = coloff[e];
    const uint8_t *gw = gbase + (size_t)e * w_stride;
    const uint8_t *uw = ubase + (size_t)e * w_stride;
    const __nv_bfloat16 *gs = gsbase + (size_t)e * s_stride;
    const __nv_bfloat16 *us = usbase + (size_t)e * s_stride;
    const float *xe = x + (size_t)off * in_dim;
    float *ae = act + (size_t)off * out_dim;
    int nblk = in_dim / FP8BLK, sstride = in_dim / FP8BLK, srow = idx / FP8BLK;
    const uint8_t *grow = gw + (size_t)idx * in_dim, *urow = uw + (size_t)idx * in_dim;
    for (int b0 = 0; b0 < cnt; b0 += FP8_GMAXB) {
        int B = (cnt - b0 < FP8_GMAXB) ? (cnt - b0) : FP8_GMAXB;
        float gacc[FP8_GMAXB], uacc[FP8_GMAXB];
        #pragma unroll
        for (int b = 0; b < FP8_GMAXB; b++) { gacc[b] = 0.0f; uacc[b] = 0.0f; }
        for (int ib = 0; ib < nblk; ib++) {
            float gbs = __bfloat162float(gs[(size_t)srow * sstride + ib]);
            float ubs = __bfloat162float(us[(size_t)srow * sstride + ib]);
            uchar4 gq = *((const uchar4 *)(grow + ib * FP8BLK) + lane);
            uchar4 uq = *((const uchar4 *)(urow + ib * FP8BLK) + lane);
            float gv[4], uv[4];
            { __nv_fp8_e4m3 a0,a1,a2,a3; a0.__x=gq.x; a1.__x=gq.y; a2.__x=gq.z; a3.__x=gq.w;
              gv[0]=float(a0); gv[1]=float(a1); gv[2]=float(a2); gv[3]=float(a3); }
            { __nv_fp8_e4m3 a0,a1,a2,a3; a0.__x=uq.x; a1.__x=uq.y; a2.__x=uq.z; a3.__x=uq.w;
              uv[0]=float(a0); uv[1]=float(a1); uv[2]=float(a2); uv[3]=float(a3); }
            for (int b = 0; b < B; b++) {
                float4 xv = *((const float4 *)(xe + (size_t)(b0 + b) * in_dim + ib * FP8BLK) + lane);
                gacc[b] += gbs * (gv[0]*xv.x + gv[1]*xv.y + gv[2]*xv.z + gv[3]*xv.w);
                uacc[b] += ubs * (uv[0]*xv.x + uv[1]*xv.y + uv[2]*xv.z + uv[3]*xv.w);
            }
        }
        for (int b = 0; b < B; b++) {
            float g = gacc[b], u = uacc[b];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                g += __shfl_xor_sync(0xFFFFFFFFu, g, o);
                u += __shfl_xor_sync(0xFFFFFFFFu, u, o);
            }
            if (lane == 0) {
                float sg = g / (1.0f + __expf(-g));   // SiLU, same form as dg_silu_mul
                ae[(size_t)(b0 + b) * out_dim + idx] = sg * u;
            }
        }
    }
}

static inline void fp8_block_gemm_grouped_gateup_silu_launch(
    float *act, const uint8_t *gbase, const uint8_t *ubase, int64_t w_stride,
    const __nv_bfloat16 *gsbase, const __nv_bfloat16 *usbase, int64_t s_stride,
    const float *x, const int *coloff, const int *count, const int *active, int n_slot,
    int n_expert, int in_dim, int out_dim, cudaStream_t stream)
{
    const int WPB = 4;
    dim3 grid((out_dim + WPB - 1) / WPB, active ? n_slot : n_expert);
    fp8_block_gemm_grouped_gateup_silu_kernel<<<grid, WPB * 32, 0, stream>>>(
        act, gbase, ubase, w_stride, gsbase, usbase, s_stride, x, coloff, count, active,
        in_dim, out_dim);
}
