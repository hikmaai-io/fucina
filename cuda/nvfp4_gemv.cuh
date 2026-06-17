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
#include <cuda_bf16.h>
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

// ─────────────────────────────────────────────────────────────────────────
// BATCHED NVFP4 decode GEMV — y[K][out] = X[K][in]·W, the weight read ONCE per K rows.
//
// This is the spec-verify lever: a K-token verify forward does the 7 NVFP4 projections for K
// rows at once. The naive way (K× single-token) re-reads the whole weight K times → a K=4 verify
// costs ~4× a decode → MTP net-negative. Here the weight (the bandwidth-dominant term: a uint4 of
// 8 nibbles per lane-group) is dequantized ONCE and FMA'd into all K activation columns, so the
// weight is read once for the entire K-batch — the read count is amortized exactly like Q4_0's
// batched dp4a.
//
// The activation must be TRANSPOSED first: X[K][in] (token-major, what the engine produces) →
// Xt[in][K] (input-major). With Xt, the K activation values for a given input index k live
// CONTIGUOUSLY at Xt[k*K .. k*K+K-1], so the per-group activation reads are coalesced across the
// warp (a naive x[r*in+k] batched kernel strides by in_dim across the K rows → 11.7 GB/s). The
// transpose is a cheap separate kernel over K·in floats (negligible vs the weight read).
//
// Register pressure: each warp keeps acc[ROWS][K]. We block ROWS=2 output rows × up to K=5 → 10
// accumulators, plus the K activation values per weight reused across ROWS. Keep ROWS×K sane.
// ROWS is the latency-hiding lever here: each warp issues ROWS INDEPENDENT weight loads per group,
// and the weight (the bandwidth-dominant read) is touched once for the whole K-batch — so unlike
// the single-token kernel (memory-bound at ROWS=4) the batched verify has K× the compute per byte
// and needs deep memory-level parallelism to stay weight-bound. Swept on GB10 sm_121a: ROWS=12,
// WARPS=2 gives the best batched(K=3..8) speedup over K× single-token across q/down/gate-up
// (2.8–3.4×, 105–190 GB/s weight-read-once). ROWS<8 is latency-bound (<60 GB/s); WARPS>2 regresses
// (fewer independent in-flight rows per scheduler). K=6..8 hit the 255-reg cap but DO NOT spill.
#ifndef NVFP4_GEMV_B_WARPS
#define NVFP4_GEMV_B_WARPS 2
#endif
#ifndef NVFP4_GEMV_B_ROWS
#define NVFP4_GEMV_B_ROWS 12
#endif
#ifndef NVFP4_GEMV_KMAX
#define NVFP4_GEMV_KMAX 8
#endif

// Transpose X[K][in] (token-major, row stride = in_dim) → Xt[in][K] (input-major, row stride = K).
// One thread per (k_in, row) element. Trivially coalesced on the OUTPUT write across blockIdx.x.
__global__ void nvfp4_xT_kernel(
    float* __restrict__ xt,            // [in_dim][K]
    const float* __restrict__ x,       // [K][in_dim]
    int in_dim, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // input index 0..in_dim-1
    if (i >= in_dim) return;
    #pragma unroll 1
    for (int r = 0; r < K; r++)
        xt[(size_t)i * K + r] = x[(size_t)r * in_dim + i];
}
static inline void nvfp4_xT_launch(
    float* xt, const float* x, int in_dim, int K, cudaStream_t stream)
{
    nvfp4_xT_kernel<<<(in_dim + 255) / 256, 256, 0, stream>>>(xt, x, in_dim, K);
}

// One warp reduces NVFP4_GEMV_B_ROWS consecutive output rows for ALL K activation columns.
// Each weight nibble is dequantized ONCE and FMA'd into K accumulators (one per activation
// column). acc[r][c] = Σ_k W[row0+r,k]·Xt[k,c]. xt is the transposed activation [in_dim][K].
template<int K>
__global__ void nvfp4_gemv_batched_kernel(
    float* __restrict__ y,                  // [K][out_dim]  (token-major output)
    const uint8_t* __restrict__ wpacked,    // [out_dim][in_dim/2]
    const uint8_t* __restrict__ wscale,     // [out_dim][in_dim/16] E4M3 linear
    const float*   __restrict__ gs,         // device scalar (weight_scale_2)
    const float*   __restrict__ xt,         // [in_dim][K] transposed activation
    int in_dim, int out_dim)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * NVFP4_GEMV_B_WARPS + warp) * NVFP4_GEMV_B_ROWS;
    if (row0 >= out_dim) return;
    const int nrow = min(NVFP4_GEMV_B_ROWS, out_dim - row0);

    const size_t wrowb = (size_t)(in_dim / 2);
    const size_t srowb = (size_t)(in_dim / 16);
    const int ngrp = in_dim / 32;

    // Branchless E2M1 nibble → float (NO indexed LUT — a runtime-indexed local array forces
    // local-memory loads, which dominate the K-wide inner loop; this is pure register ALU).
    // mag(code) over 0..7 = {0,.5,1,1.5,2,3,4,6}: build the fp32 exponent/mantissa from the bits.
    //   exp2 = code>>1 (0..3), m = code&1.  code0 → 0.  else value = (1 + .5·m)·2^(exp2-1).
    auto dec = [](uint32_t nib) -> float {
        uint32_t mag3 = nib & 7u;
        uint32_t e2   = mag3 >> 1;                 // 0..3
        uint32_t m    = mag3 & 1u;
        // normals (e2>=1): value = (1 + .5·m)·2^(e2-1) → exp field 126+e2, mantissa bit = m.
        uint32_t bits = ((126u + e2) << 23) | (m << 22);
        float v = __int_as_float((int)bits);
        if (e2 == 0u) v = 0.5f * (float)m;         // codes 0,1 → 0, 0.5 (subnormal range)
        return (nib & 8u) ? -v : v;
    };

    float acc[NVFP4_GEMV_B_ROWS][K];
    #pragma unroll
    for (int r = 0; r < NVFP4_GEMV_B_ROWS; r++)
        #pragma unroll
        for (int c = 0; c < K; c++) acc[r][c] = 0.f;

    for (int g = lane; g < ngrp; g += 32) {
        const int xi = g * 32;
        // Load all ROWS weights for this group FIRST — independent loads issued together give
        // the memory-level parallelism that hides LPDDR5X latency (the single-token kernel's
        // trick). The weight is the bandwidth-dominant read and is touched ONCE per K-batch.
        uint32_t uu[NVFP4_GEMV_B_ROWS][4];
        float  bs0[NVFP4_GEMV_B_ROWS], bs1[NVFP4_GEMV_B_ROWS];
        #pragma unroll
        for (int r = 0; r < NVFP4_GEMV_B_ROWS; r++) {
            int rr = (r < nrow) ? r : 0;   // clamp tail rows to a valid address (result discarded)
            const uint8_t* wrow = wpacked + (size_t)(row0 + rr) * wrowb;
            const uint8_t* srow = wscale  + (size_t)(row0 + rr) * srowb;
            uint4 p = *reinterpret_cast<const uint4*>(wrow + (size_t)g * 16);
            uu[r][0] = p.x; uu[r][1] = p.y; uu[r][2] = p.z; uu[r][3] = p.w;
            bs0[r] = nvfp4_e4m3_dev(srow[(size_t)g * 2 + 0]);
            bs1[r] = nvfp4_e4m3_dev(srow[(size_t)g * 2 + 1]);
        }
        // Dequant each weight ONCE, FMA into ROWS×K accumulators. xt is read directly (input-major,
        // so the K values for input k are contiguous at xt[k*K..] → coalesced; tiny K·in footprint
        // stays L1-hot and is reused across all ROWS via the L1 cache, not re-fetched from DRAM).
        #pragma unroll
        for (int b = 0; b < 32; b += 2) {
            const float* xa = xt + (size_t)(xi + b) * K;     // input b   (K contiguous)
            const float* xc = xa + K;                        // input b+1 (K contiguous)
            int q = b >> 3, sub = (b >> 1) & 3;              // which uint32, which byte
            #pragma unroll
            for (int r = 0; r < NVFP4_GEMV_B_ROWS; r++) {
                uint32_t byte = (uu[r][q] >> (8 * sub)) & 0xFFu;
                float bs = (q < 2) ? bs0[r] : bs1[r];
                float w0 = dec(byte & 0xF) * bs;
                float w1 = dec(byte >> 4)  * bs;
                #pragma unroll
                for (int c = 0; c < K; c++) acc[r][c] += w0 * xa[c] + w1 * xc[c];
            }
        }
    }
    const float g_scale = *gs;
    #pragma unroll
    for (int r = 0; r < NVFP4_GEMV_B_ROWS; r++) {
        if (r >= nrow) break;
        #pragma unroll
        for (int c = 0; c < K; c++) {
            float a = acc[r][c];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0) y[(size_t)c * out_dim + row0 + r] = a * g_scale;
        }
    }
}

// Dispatch the batched NVFP4 GEMV to its compile-time-K kernel. `xt` must already hold the
// transposed activation [in_dim][K] (call nvfp4_xT_launch first). Output y is token-major
// [K][out_dim]. CUDA-graph-capturable: no allocation, no host sync.
static inline void nvfp4_gemv_batched_launch(
    float* y, const uint8_t* wpacked, const uint8_t* wscale, const float* gs,
    const float* xt, int in_dim, int out_dim, int K, cudaStream_t stream)
{
    const int per_blk = NVFP4_GEMV_B_WARPS * NVFP4_GEMV_B_ROWS;
    unsigned blocks = (unsigned)((out_dim + per_blk - 1) / per_blk);
    dim3 b(32 * NVFP4_GEMV_B_WARPS), g(blocks);
    #define NVFP4_B_DISPATCH(NK) \
        case NK: nvfp4_gemv_batched_kernel<NK><<<g,b,0,stream>>>( \
                     y, wpacked, wscale, gs, xt, in_dim, out_dim); break;
    switch (K) {
        NVFP4_B_DISPATCH(1) NVFP4_B_DISPATCH(2) NVFP4_B_DISPATCH(3) NVFP4_B_DISPATCH(4)
        NVFP4_B_DISPATCH(5) NVFP4_B_DISPATCH(6) NVFP4_B_DISPATCH(7) NVFP4_B_DISPATCH(8)
        default: break;
    }
    #undef NVFP4_B_DISPATCH
}

// ─────────────────────────────────────────────────────────────────────────
// BATCHED BF16 LM-HEAD GEMV — y[K][vocab] = X[K][hidden]·Head, the 2 GB head read ONCE per K rows.
//
// The untied NVFP4 lm_head is BF16 [vocab][hidden] (~2 GB), read every token. The per-row loop in
// the spec-verify reads it K times. Same weight-read-once recipe as the NVFP4 batched GEMV: the
// activation is transposed to Xt[hidden][K] (nvfp4_xT_launch — the K values for a hidden index are
// then contiguous → coalesced) and the warp register-blocks BF16_HEAD_B_ROWS vocab rows, FMA'ing
// each BF16 weight into all K activation columns. Output is token-major [K][vocab]. The head row is
// read once for the whole K-batch → halves+ the head bandwidth of the K× single-token loop.
#ifndef BF16_HEAD_B_WARPS
#define BF16_HEAD_B_WARPS 2
#endif
#ifndef BF16_HEAD_B_ROWS
#define BF16_HEAD_B_ROWS 12
#endif
// Single-token BF16 head GEMV (register-blocked, weight read once), the K×-loop reference for the
// batched head bench. Mirrors the engine's bf16_head_gemv_kernel; named distinctly so this header
// can be included alongside the engine TU without a duplicate symbol.
__global__ void bf16_head_gemv1_kernel(
    float* __restrict__ y, const __nv_bfloat16* __restrict__ w,
    const float* __restrict__ x, int in_dim, int out_dim)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * BF16_HEAD_B_WARPS + warp) * BF16_HEAD_B_ROWS;
    if (row0 >= out_dim) return;
    const int nrow = min(BF16_HEAD_B_ROWS, out_dim - row0);
    float acc[BF16_HEAD_B_ROWS];
    #pragma unroll
    for (int r = 0; r < BF16_HEAD_B_ROWS; r++) acc[r] = 0.f;
    for (int k = lane; k < in_dim; k += 32) {
        float xk = x[k];
        #pragma unroll
        for (int r = 0; r < BF16_HEAD_B_ROWS; r++) {
            if (r >= nrow) break;
            acc[r] += __bfloat162float(w[(size_t)(row0 + r) * in_dim + k]) * xk;
        }
    }
    #pragma unroll
    for (int r = 0; r < BF16_HEAD_B_ROWS; r++) {
        float a = acc[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0 && r < nrow) y[row0 + r] = a;
    }
}
static inline void bf16_head_gemv1_launch(
    float* y, const __nv_bfloat16* w, const float* x, int in_dim, int out_dim, cudaStream_t stream)
{
    const int per_blk = BF16_HEAD_B_WARPS * BF16_HEAD_B_ROWS;
    unsigned blocks = (unsigned)((out_dim + per_blk - 1) / per_blk);
    bf16_head_gemv1_kernel<<<blocks, 32 * BF16_HEAD_B_WARPS, 0, stream>>>(y, w, x, in_dim, out_dim);
}

template<int K>
__global__ void bf16_head_gemv_batched_kernel(
    float*               __restrict__ y,    // [K][out_dim] token-major
    const __nv_bfloat16* __restrict__ w,    // [out_dim][in_dim]
    const float*         __restrict__ xt,   // [in_dim][K] transposed activation
    int in_dim, int out_dim)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * BF16_HEAD_B_WARPS + warp) * BF16_HEAD_B_ROWS;
    if (row0 >= out_dim) return;
    const int nrow = min(BF16_HEAD_B_ROWS, out_dim - row0);

    float acc[BF16_HEAD_B_ROWS][K];
    #pragma unroll
    for (int r = 0; r < BF16_HEAD_B_ROWS; r++)
        #pragma unroll
        for (int c = 0; c < K; c++) acc[r][c] = 0.f;

    for (int k = lane; k < in_dim; k += 32) {
        const float* xp = xt + (size_t)k * K;       // K contiguous activation values for input k
        // Load all ROWS weights for this k first (independent loads → memory-level parallelism).
        float wv[BF16_HEAD_B_ROWS];
        #pragma unroll
        for (int r = 0; r < BF16_HEAD_B_ROWS; r++) {
            int rr = (r < nrow) ? r : 0;
            wv[r] = __bfloat162float(w[(size_t)(row0 + rr) * in_dim + k]);
        }
        #pragma unroll
        for (int r = 0; r < BF16_HEAD_B_ROWS; r++) {
            #pragma unroll
            for (int c = 0; c < K; c++) acc[r][c] += wv[r] * xp[c];
        }
    }
    #pragma unroll
    for (int r = 0; r < BF16_HEAD_B_ROWS; r++) {
        if (r >= nrow) break;
        #pragma unroll
        for (int c = 0; c < K; c++) {
            float a = acc[r][c];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0) y[(size_t)c * out_dim + row0 + r] = a;
        }
    }
}
// Dispatch the batched BF16 head to its compile-time-K kernel. `xt` = transposed activation
// [in_dim][K] (nvfp4_xT_launch). Output token-major [K][out_dim]. CUDA-graph-capturable.
static inline void bf16_head_gemv_batched_launch(
    float* y, const __nv_bfloat16* w, const float* xt,
    int in_dim, int out_dim, int K, cudaStream_t stream)
{
    const int per_blk = BF16_HEAD_B_WARPS * BF16_HEAD_B_ROWS;
    unsigned blocks = (unsigned)((out_dim + per_blk - 1) / per_blk);
    dim3 b(32 * BF16_HEAD_B_WARPS), g(blocks);
    #define BF16_HB_DISPATCH(NK) \
        case NK: bf16_head_gemv_batched_kernel<NK><<<g,b,0,stream>>>( \
                     y, w, xt, in_dim, out_dim); break;
    switch (K) {
        BF16_HB_DISPATCH(1) BF16_HB_DISPATCH(2) BF16_HB_DISPATCH(3) BF16_HB_DISPATCH(4)
        BF16_HB_DISPATCH(5) BF16_HB_DISPATCH(6) BF16_HB_DISPATCH(7) BF16_HB_DISPATCH(8)
        default: break;
    }
    #undef BF16_HB_DISPATCH
}

// ─────────────────────────────────────────────────────────────────────────
// FP8 E4M3 PER-ROW LM-HEAD — quantize + GEMV (single + batched).
//
// The untied NVFP4 lm_head is BF16 [vocab][hidden] (~2 GB), read every token. Quantizing it to
// E4M3 (1 B/elem) with ONE dequant scale per vocab row exactly halves the head read. E4M3's ~2
// decimal digits of mantissa is plenty for a logits GEMV that immediately feeds an argmax/softmax —
// the load-time accuracy gate (engine) confirms the FP8-head argmax matches the BF16-head argmax.
//
//   q[v,h] = e4m3( w_bf16[v,h] / scale[v] ),  scale[v] = amax_h(|w_bf16[v,h]|) / 448
//   logits[v] = scale[v] · Σ_h e4m3_decode(q[v,h]) · x[h]
//
// The GEMVs mirror bf16_head_gemv (register-block ROWS output rows, x once / weight once) but read 1
// byte/elem and apply the per-row scale at the warp reduction. Capture-safe: no alloc / host sync.

// Per-row quantize w_bf16[vocab][hidden] → (q_fp8[vocab][hidden], scale[vocab]). One block per row;
// the block reduces amax over the row, then writes E4M3 quantized weights. Load-time only (not hot).
__global__ void fp8_head_quantize_kernel(
    uint8_t*             __restrict__ q,      // [vocab][hidden] E4M3 out
    float*               __restrict__ scale,  // [vocab] out
    const __nv_bfloat16* __restrict__ w,      // [vocab][hidden] BF16 in
    int hidden)
{
    const int row = blockIdx.x;
    const __nv_bfloat16* wr = w + (size_t)row * hidden;
    uint8_t* qr = q + (size_t)row * hidden;
    __shared__ float s_amax[32];

    float amax = 0.f;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x)
        amax = fmaxf(amax, fabsf(__bfloat162float(wr[h])));
    // block reduce max
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((threadIdx.x & 31) == 0) s_amax[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_amax[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) s_amax[0] = v;
    }
    __syncthreads();
    amax = s_amax[0];

    const float sc  = amax > 0.f ? amax / 448.f : 1.f;   // E4M3 max finite = 448
    const float inv = 1.f / sc;
    if (threadIdx.x == 0) scale[row] = sc;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float v = __bfloat162float(wr[h]) * inv;
        qr[h] = (uint8_t)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
}
static inline void fp8_head_quantize_launch(
    uint8_t* q, float* scale, const __nv_bfloat16* w, int vocab, int hidden, cudaStream_t stream)
{
    fp8_head_quantize_kernel<<<vocab, 256, 0, stream>>>(q, scale, w, hidden);
}

#ifndef FP8_HEAD_WARPS
#define FP8_HEAD_WARPS 8
#endif
#ifndef FP8_HEAD_ROWS
#define FP8_HEAD_ROWS 4
#endif
// Single-token FP8 head GEMV: logits[v] = scale[v]·Σ_h e4m3(q[v,h])·x[h]. Register-blocks ROWS rows
// (independent weight loads to hide latency, x reused L1-hot). Each lane loads a uchar4 (4 contiguous
// E4M3 weights) per step so a warp reads a coalesced 128 B line — 1 B/elem at full bandwidth.
// Requires in_dim % 4 == 0 (true for Gemma-4 hidden=3840). Capture-safe.
__global__ void fp8_head_gemv_kernel(
    float*         __restrict__ y,       // [out_dim]
    const uint8_t* __restrict__ q,       // [out_dim][in_dim] E4M3
    const float*   __restrict__ scale,   // [out_dim] per-row dequant
    const float*   __restrict__ x,       // [in_dim]
    int in_dim, int out_dim)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * FP8_HEAD_WARPS + warp) * FP8_HEAD_ROWS;
    if (row0 >= out_dim) return;
    const int nrow = min(FP8_HEAD_ROWS, out_dim - row0);
    const int in4 = in_dim >> 2;                       // groups of 4 columns
    float acc[FP8_HEAD_ROWS];
    #pragma unroll
    for (int r = 0; r < FP8_HEAD_ROWS; r++) acc[r] = 0.f;
    for (int g = lane; g < in4; g += 32) {
        const float4 xk = reinterpret_cast<const float4*>(x)[g];
        #pragma unroll
        for (int r = 0; r < FP8_HEAD_ROWS; r++) {
            if (r >= nrow) break;
            const uchar4 wq = reinterpret_cast<const uchar4*>(q + (size_t)(row0 + r) * in_dim)[g];
            acc[r] += nvfp4_e4m3_dev(wq.x) * xk.x + nvfp4_e4m3_dev(wq.y) * xk.y
                    + nvfp4_e4m3_dev(wq.z) * xk.z + nvfp4_e4m3_dev(wq.w) * xk.w;
        }
    }
    #pragma unroll
    for (int r = 0; r < FP8_HEAD_ROWS; r++) {
        float a = acc[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0 && r < nrow) y[row0 + r] = a * scale[row0 + r];
    }
}
static inline void fp8_head_gemv_launch(
    float* y, const uint8_t* q, const float* scale, const float* x,
    int in_dim, int out_dim, cudaStream_t stream)
{
    const int per_blk = FP8_HEAD_WARPS * FP8_HEAD_ROWS;
    unsigned blocks = (unsigned)((out_dim + per_blk - 1) / per_blk);
    fp8_head_gemv_kernel<<<blocks, 32 * FP8_HEAD_WARPS, 0, stream>>>(y, q, scale, x, in_dim, out_dim);
}

#ifndef FP8_HEAD_B_WARPS
#define FP8_HEAD_B_WARPS 2
#endif
#ifndef FP8_HEAD_B_ROWS
#define FP8_HEAD_B_ROWS 12
#endif
// Batched FP8 head GEMV: y[K][out] = scale[out]·(Xt[in][K]·Q). Each E4M3 weight is dequantized ONCE
// and FMA'd into all K activation columns (weight read once for the K-batch). xt = transposed
// activation [in_dim][K] (nvfp4_xT_launch). Output token-major [K][out_dim]. Capture-safe.
template<int K>
__global__ void fp8_head_gemv_batched_kernel(
    float*         __restrict__ y,       // [K][out_dim] token-major
    const uint8_t* __restrict__ q,       // [out_dim][in_dim] E4M3
    const float*   __restrict__ scale,   // [out_dim] per-row dequant
    const float*   __restrict__ xt,      // [in_dim][K] transposed activation
    int in_dim, int out_dim)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * FP8_HEAD_B_WARPS + warp) * FP8_HEAD_B_ROWS;
    if (row0 >= out_dim) return;
    const int nrow = min(FP8_HEAD_B_ROWS, out_dim - row0);

    float acc[FP8_HEAD_B_ROWS][K];
    #pragma unroll
    for (int r = 0; r < FP8_HEAD_B_ROWS; r++)
        #pragma unroll
        for (int c = 0; c < K; c++) acc[r][c] = 0.f;

    const int in4 = in_dim >> 2;                        // groups of 4 columns (uchar4 weight load)
    for (int g = lane; g < in4; g += 32) {
        const float* xp = xt + (size_t)(g << 2) * K;    // K-contiguous activations for these 4 k's
        #pragma unroll
        for (int r = 0; r < FP8_HEAD_B_ROWS; r++) {
            int rr = (r < nrow) ? r : 0;
            const uchar4 wq = reinterpret_cast<const uchar4*>(q + (size_t)(row0 + rr) * in_dim)[g];
            const float w0 = nvfp4_e4m3_dev(wq.x), w1 = nvfp4_e4m3_dev(wq.y),
                        w2 = nvfp4_e4m3_dev(wq.z), w3 = nvfp4_e4m3_dev(wq.w);
            #pragma unroll
            for (int c = 0; c < K; c++)
                acc[r][c] += w0 * xp[c] + w1 * xp[K + c] + w2 * xp[2*K + c] + w3 * xp[3*K + c];
        }
    }
    #pragma unroll
    for (int r = 0; r < FP8_HEAD_B_ROWS; r++) {
        if (r >= nrow) break;
        const float sc = scale[row0 + r];
        #pragma unroll
        for (int c = 0; c < K; c++) {
            float a = acc[r][c];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0) y[(size_t)c * out_dim + row0 + r] = a * sc;
        }
    }
}
static inline void fp8_head_gemv_batched_launch(
    float* y, const uint8_t* q, const float* scale, const float* xt,
    int in_dim, int out_dim, int K, cudaStream_t stream)
{
    const int per_blk = FP8_HEAD_B_WARPS * FP8_HEAD_B_ROWS;
    unsigned blocks = (unsigned)((out_dim + per_blk - 1) / per_blk);
    dim3 b(32 * FP8_HEAD_B_WARPS), g(blocks);
    #define FP8_HB_DISPATCH(NK) \
        case NK: fp8_head_gemv_batched_kernel<NK><<<g,b,0,stream>>>( \
                     y, q, scale, xt, in_dim, out_dim); break;
    switch (K) {
        FP8_HB_DISPATCH(1) FP8_HB_DISPATCH(2) FP8_HB_DISPATCH(3) FP8_HB_DISPATCH(4)
        FP8_HB_DISPATCH(5) FP8_HB_DISPATCH(6) FP8_HB_DISPATCH(7) FP8_HB_DISPATCH(8)
        default: break;
    }
    #undef FP8_HB_DISPATCH
}

#endif // FUCINA_NVFP4_GEMV_CUH
