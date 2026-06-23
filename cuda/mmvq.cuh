// mmvq.cuh — shared dp4a MMVQ (int8 matvec) decode kernels for the fucina engines.
//
// Single source of truth for the quantized-weight decode/batched GEMV path that was
// born in the dense gemma4 engine (gemma4_kernels.cu) and is reused by the standalone
// Gemma-4-E4B engine. llama.cpp's mul_mat_vec_q (MMVQ): the activation is quantized to
// per-32-block symmetric int8 (Q8_1-style, with the block sum for Q4_0's −8 fold) and
// dotted against the native quant weight blocks via __dp4a — weights stay resident in
// their on-disk Q4_0 (18 B) / Q8_0 (34 B) / Q6_K (210 B) layout, never materialized to BF16.
//
// EVERYTHING here is `static` (internal linkage), including the templates, so the header
// can be included by BOTH gemma4_kernels.cu and e4b_engine.cu without duplicate-symbol
// clashes in libfucina.a (the e4b_nvfp4.cuh precedent). The GLU helper is named
// mmvq_gelu_tanh so it does not collide with either TU's own global gelu_tanh.
//
// fmt codes: 2 = Q4_0, else Q8_0 (matches the dense engine's wfmt).
#ifndef FUCINA_MMVQ_CUH
#define FUCINA_MMVQ_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// ── device helpers ───────────────────────────────────────────────────────────
// Warp-level butterfly sum — the reduced value ends up in ALL 32 lanes (no smem, no sync).
static __device__ __forceinline__ float warp_reduce_sum_all(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, o);
    return v;
}

// Read 4 packed int8 as one int, tolerating the 2-byte alignment of a Q8_0/Q4_0 block's
// qs (it starts at byte offset 2 within the block).
static __device__ __forceinline__ int q8_get_int_b2(const void *p, int i32) {
    const uint16_t *x16 = (const uint16_t *)p;
    return (int)x16[2*i32] | ((int)x16[2*i32 + 1] << 16);
}

// gelu_pytorch_tanh, for the fused GLU MMVQ kernels (renamed to avoid colliding with the
// including TU's own gelu_tanh).
static __device__ inline float mmvq_gelu_tanh(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x3)));
}

// ── activation quantizers (Q8_1-style: int8 + per-block scale + Σqx block sum) ──
// Quantize x[in_dim] to symmetric per-32-block int8 + per-block scale. qx[in_dim] is
// 4-byte aligned at every block boundary; dx[in_dim/32]; sx[in_dim/32] = Σqx (folds the
// Q4_0 −8 nibble correction). One warp per block; in_dim a multiple of 32.
static __global__ void quantize_q8_1_kernel(
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
    int qsum = q;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) qsum += __shfl_xor_sync(0xFFFFFFFF, qsum, o);
    if (lane == 0) { dx[b] = d; sx[b] = qsum; }
}

// BF16-input variant: the activation already lives as BF16 (rms-norm / attn-out / geglu
// outputs), so quantize straight from BF16. Same layout/math as the FP32 kernel.
static __global__ void quantize_q8_1_bf16_kernel(
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

// ── single-token MMVQ (warp-per-row) ─────────────────────────────────────────
// out[idx] = Σ_block d_w_b · d_x_b · Σ_k __dp4a(weight_qs[k], act_qs[k]).
static __global__ void mmvq_q8_0_kernel(
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
        const void *wqs = blk + 2;
        const int  *xqs = (const int *)(qx + (size_t)b * 32);
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++)
            sumi = __dp4a(q8_get_int_b2(wqs, k), xqs[k], sumi);
        acc += dw * dx[b] * (float)sumi;
    }
    acc = warp_reduce_sum_all(acc);
    if (lane == 0) out[idx] = acc;
}

// Q4_0: block = fp16 scale + 16 bytes of 32 nibbles; value = dw*(nibble-8). The -8 offset
// is corrected via the precomputed Σqx (sx): out = dw·dx·(Σ nibble·qx − 8·Σ qx).
static __global__ void mmvq_q4_0_kernel(
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
        const void *wqs = blk + 2;
        const int  *xqs = (const int *)(qx + (size_t)b * 32);
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int w   = q8_get_int_b2(wqs, k);
            int vlo = w & 0x0F0F0F0F;
            int vhi = (w >> 4) & 0x0F0F0F0F;
            sumi = __dp4a(vlo, xqs[k],     sumi);
            sumi = __dp4a(vhi, xqs[k + 4], sumi);
        }
        acc += dw * dx[b] * (float)(sumi - 8 * sx[b]);
    }
    acc = warp_reduce_sum_all(acc);
    if (lane == 0) out[idx] = acc;
}

// ── batched MMVQ (NK tokens, one weight pass) ────────────────────────────────
// Reads each weight ROW once, decodes the nibbles once, reuses across NK quantized
// activation vectors (token-major qx[n*in_dim+i], dx/sx[n*nb+b]).
template<int NK>
static __global__ void mmvq_q4_0_batched_kernel(
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
        const void *wqs = blk + 2;
        int wv[8];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int w = q8_get_int_b2(wqs, k);
            wv[2*k]   =  w        & 0x0F0F0F0F;
            wv[2*k+1] = (w >> 4)  & 0x0F0F0F0F;
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

template<int NK>
static __global__ void mmvq_q8_0_batched_kernel(
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
        const void *wqs = blk + 2;
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

// K ≤ 8 → one weight pass; K > 8 splits into ≤8-wide chunks (each re-reads the weight).
static void mmvq_batched_launch(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim, int K, int fmt, cudaStream_t stream)
{
    const int NWARPS = 8; int b = NWARPS*32;
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

// ── FUCINA_PACKED: repacked-Q4_0 (coalesced uint4 loads) ─────────────────────
// Repack native Q4_0 blocks into SoA: [out_dim·nb × 16 quant bytes] then [× fp16 scale].
static __global__ void repack_q4_0_kernel(
    const uint8_t *src, uint8_t *quants, uint16_t *scales, size_t nblocks)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nblocks) return;
    const uint8_t *blk = src + i * 18;
    scales[i] = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
    uint8_t *d = quants + i * 16;
    #pragma unroll
    for (int k = 0; k < 16; k++) d[k] = blk[2 + k];
}

template<int NK>
static __global__ void mmvq_q4_0_packed_batched_kernel(
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
        uint4 q = *(const uint4 *)(qrow + (size_t)b * 16);
        __half_raw hr; hr.x = srow[b];
        float dw = __half2float(__half(hr));
        int wv[8];
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

// Dequant native Q4_0 blocks → dense BF16 [out_dim, in_dim] (row-major). For the prefill
// (T>1) cuBLAS GEMM, which is more efficient than chunked MMVQ at large token counts. One
// thread per 18-byte block (value = d*(nibble-8), nibble layout = j low, j+16 high).
static __global__ void dequant_q4_0_to_bf16_kernel(
    const uint8_t *weight, __nv_bfloat16 *out, int in_dim, int64_t nblk)
{
    int64_t blk = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= nblk) return;
    const uint8_t *b = weight + blk * 18;
    __half_raw hr; hr.x = (uint16_t)(b[0] | ((uint16_t)b[1] << 8));
    float d = __half2float(__half(hr));
    __nv_bfloat16 *o = out + blk * 32;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        o[j]      = __float2bfloat16(d * (float)((b[2+j] & 0x0F) - 8));
        o[j + 16] = __float2bfloat16(d * (float)((b[2+j] >> 4)   - 8));
    }
}

// ── Q6_K MMVQ (native tied LM head; 210-byte superblocks) ────────────────────
static __global__ void mmvq_q6_k_kernel(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx,
    int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int n_super = in_dim >> 8;
    int nb32 = in_dim >> 5;
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

template<int NK>
static __global__ void mmvq_q6_k_batched_kernel(
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

// ── fused gate+up+GeGLU MMVQ ─────────────────────────────────────────────────
static __global__ void mmvq_q4_0_glu_kernel(
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
    if (lane == 0) out[idx] = mmvq_gelu_tanh(ga) * ua;
}

static __global__ void mmvq_q8_0_glu_kernel(
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
    if (lane == 0) out[idx] = mmvq_gelu_tanh(ga) * ua;
}

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

#endif // FUCINA_MMVQ_CUH
