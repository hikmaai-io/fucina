// e4b_q4_0.cuh — native Q4_0 weight path for the Gemma-4-E4B engine.
//
// The hybrid NVFP4/FP8 decode path (e4b_nvfp4.cuh) cuts decode bandwidth, but it
// dequantizes the on-disk Q4_0 GGUF → BF16 at load and then RE-quantizes BF16 →
// NVFP4, keeping BOTH resident (+3 GB) and adding a second lossy round-trip. This
// header is the production-correct alternative: keep the original Q4_0 QAT weights
// bit-for-bit and decode straight off them with an int8 dp4a GEMV — exactly how
// llama.cpp's MMVQ (`mul_mat_vec_q`) works.
//
//   weight  : Q4_0 block = 32 elems, one fp16 scale d, value[k] = d*(nibble[k]-8),
//             nibble[k] in 0..15.  Stored structure-of-arrays so reads are aligned:
//               qs : U8  [out][in/2]   16 nibble-bytes/block (low nibble = elem j,
//                                       high nibble = elem j+16)
//               d  : F32 [out][in/32]  per-block scale (GGUF fp16 widened once)
//   activation: quantized at GEMV time to Q8_1-style int8 (symmetric, per-32 block
//               scale d_a = amax/127) plus the block sum s_a, which corrects Q4_0's
//               -8 zero point:  sum_k d*(n_k-8)*d_a*q_k = d*d_a*(dp4a(n,q) - 8*s_a).
//
// Validated standalone by test_e4b_q4_0.cu (GEMV vs host f32 oracle).
#ifndef FUCINA_E4B_Q4_0_CUH
#define FUCINA_E4B_Q4_0_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstring>
#include <vector>

// Self-contained (like e4b_nvfp4.cuh): all __global__ kernels are `static` (internal
// linkage) so bundling this TU's .o next to gemma4_kernels.o in libfucina.a cannot
// clash. The namespace is distinct from e4bfp4 so both headers coexist in one TU.
namespace e4bq40 {

#ifndef E4B_Q40_WARPS
#define E4B_Q40_WARPS 8
#endif
#ifndef E4B_Q40_ROWS
#define E4B_Q40_ROWS 4
#endif

// ── native Q4_0 weight, device-resident SoA ──────────────────────────────────
struct Q40Weight {
    uint8_t* qs = nullptr;   // [out_dim][in_dim/2]  nibble bytes
    float*   d  = nullptr;   // [out_dim][in_dim/32] block scales
    int in_dim = 0, out_dim = 0;
    uint64_t bytes = 0;
};

inline void q40_free(Q40Weight* w) {
    if (!w) return;
    if (w->qs) cudaFree(w->qs);
    if (w->d)  cudaFree(w->d);
    w->qs = nullptr; w->d = nullptr; w->bytes = 0;
}

// host fp16 → f32 (Q4_0 block scales are stored fp16 in the GGUF stream).
static inline float q40_h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF, f;
    if (e == 0) {
        if (m == 0) f = s;
        else { e = 112; while (!(m & 0x400)) { m <<= 1; e--; }
               m &= 0x3FF; f = s | (e << 23) | (m << 13); }
    } else if (e == 0x1F) f = s | 0x7F800000u | (m << 13);
    else f = s | ((e + 112) << 23) | (m << 13);
    float o; memcpy(&o, &f, 4); return o;
}

// ── activation quantizer: bf16 x[in] → int8 qa[in] + f32 da[in/32] + i32 sa[in/32] ──
// One block (32 threads) per 32-elem group: warp-reduce amax → scale, round, warp-sum.
static __global__ void q8_quant_kernel(const __nv_bfloat16* __restrict__ x,
                                       int8_t* __restrict__ qa, float* __restrict__ da,
                                       int32_t* __restrict__ sa) {
    const int g = blockIdx.x, lane = threadIdx.x;       // blockDim.x == 32
    float v = __bfloat162float(x[(size_t)g * 32 + lane]);
    float a = fabsf(v);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float d = a * (1.0f / 127.0f);
    if (!(d > 0.f)) d = 1.0f;
    int q = __float2int_rn(v / d);
    q = max(-127, min(127, q));
    qa[(size_t)g * 32 + lane] = (int8_t)q;
    int s = q;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) { da[g] = d; sa[g] = s; }
}

// ── dp4a GEMV: y[out] = W_q4_0 · x, one warp per E4B_Q40_ROWS output rows ──
static __global__ void q40_gemv_kernel(
    float* __restrict__ y, const uint8_t* __restrict__ qs, const float* __restrict__ d,
    const int8_t* __restrict__ qa, const float* __restrict__ da, const int32_t* __restrict__ sa,
    int in_dim, int out_dim) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * E4B_Q40_WARPS + warp) * E4B_Q40_ROWS;
    if (row0 >= out_dim) return;
    const int nrow = min(E4B_Q40_ROWS, out_dim - row0);
    const int ngrp = in_dim / 32;
    const size_t qrowb = (size_t)(in_dim / 2);    // nibble bytes per weight row
    const size_t drow  = (size_t)(in_dim / 32);   // scales per weight row
    const uint32_t* qa32 = reinterpret_cast<const uint32_t*>(qa);

    float acc[E4B_Q40_ROWS];
    #pragma unroll
    for (int r = 0; r < E4B_Q40_ROWS; r++) acc[r] = 0.f;

    for (int g = lane; g < ngrp; g += 32) {
        const float da_g = da[g];
        const int   sa_g = sa[g];
        #pragma unroll
        for (int r = 0; r < E4B_Q40_ROWS; r++) {
            if (r >= nrow) break;
            const uint8_t* wrow = qs + (size_t)(row0 + r) * qrowb + (size_t)g * 16;
            uint4 wv = *reinterpret_cast<const uint4*>(wrow);
            const uint32_t wp[4] = { wv.x, wv.y, wv.z, wv.w };
            int isum = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                uint32_t lo = wp[i] & 0x0F0F0F0Fu;          // elems g*32 + i*4 .. +3
                uint32_t hi = (wp[i] >> 4) & 0x0F0F0F0Fu;   // elems g*32 + 16 + i*4 .. +3
                isum = __dp4a((int)lo, (int)qa32[(size_t)g * 8 + i],     isum);
                isum = __dp4a((int)hi, (int)qa32[(size_t)g * 8 + 4 + i], isum);
            }
            const float dw = d[(size_t)(row0 + r) * drow + g];
            acc[r] += dw * da_g * (float)(isum - 8 * sa_g);
        }
    }
    #pragma unroll
    for (int r = 0; r < E4B_Q40_ROWS; r++) {
        float a = acc[r];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0 && r < nrow) y[row0 + r] = a;
    }
}

static __global__ void q40_to_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = __float2bfloat16(x[i]);
}

// Dequant a Q4_0 SoA weight back to a dense BF16 [out_dim, in_dim] (row-major) — the
// prefill path's cuBLAS GEMM reads this, so the BF16 projection weights need not stay
// resident. Reproduces the loader's host dequant bit-for-bit (value = d*(nibble-8),
// RNE to bf16), one thread per 32-elem block.
static __global__ void q40_dequant_kernel(const uint8_t* __restrict__ qs, const float* __restrict__ d,
                                          __nv_bfloat16* __restrict__ out, int in_dim, int64_t nblk) {
    int64_t blk = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= nblk) return;
    const int nb = in_dim / 32;
    const int64_t row = blk / nb, g = blk % nb;
    const uint8_t* b = qs + blk * 16;
    const float scale = d[blk];
    __nv_bfloat16* o = out + row * in_dim + g * 32;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        o[j]      = __float2bfloat16(scale * (float)((b[j] & 0x0F) - 8));
        o[j + 16] = __float2bfloat16(scale * (float)((b[j] >> 4)   - 8));
    }
}

// Dequant W (Q4_0 SoA) → dst BF16 [out_dim, in_dim]. dst must hold out_dim*in_dim bf16.
inline void e4b_q4_0_dequant_bf16(const Q40Weight& w, __nv_bfloat16* dst, cudaStream_t s) {
    const int64_t nblk = (int64_t)w.out_dim * (w.in_dim / 32);
    q40_dequant_kernel<<<(unsigned)((nblk + 255) / 256), 256, 0, s>>>(w.qs, w.d, dst, w.in_dim, nblk);
}

// ── builders ─────────────────────────────────────────────────────────────────
// From a raw GGUF Q4_0 tensor (row-major [out_dim, in_dim], in_dim multiple of 32):
// split the 18-byte blocks (fp16 scale + 16 nibble bytes) into the SoA arrays. No
// dequant — the original QAT nibbles are kept verbatim.
inline bool e4b_q4_0_from_gguf(const uint8_t* src, int out_dim, int in_dim, Q40Weight* w) {
    if (!w || !src || in_dim <= 0 || out_dim <= 0 || (in_dim & 31)) return false;
    w->in_dim = in_dim; w->out_dim = out_dim;
    const int    nb = in_dim / 32;                       // blocks per row
    const size_t nblk = (size_t)out_dim * nb;
    const size_t qbytes = (size_t)out_dim * (in_dim / 2);
    std::vector<uint8_t> hqs(qbytes);
    std::vector<float>   hd(nblk);
    for (size_t blk = 0; blk < nblk; blk++) {
        const uint8_t* b = src + blk * 18;
        hd[blk] = q40_h2f((uint16_t)(b[0] | ((uint16_t)b[1] << 8)));
        memcpy(&hqs[blk * 16], b + 2, 16);
    }
    if (cudaMalloc(&w->qs, qbytes) != cudaSuccess) return false;
    if (cudaMalloc(&w->d, nblk * sizeof(float)) != cudaSuccess) { q40_free(w); return false; }
    if (cudaMemcpy(w->qs, hqs.data(), qbytes, cudaMemcpyHostToDevice) != cudaSuccess) { q40_free(w); return false; }
    if (cudaMemcpy(w->d, hd.data(), nblk * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) { q40_free(w); return false; }
    w->bytes = qbytes + nblk * sizeof(float);
    return true;
}

// host Q4_0 quantize of one 32-elem block (llama.cpp convention): pick the element
// with max |x|, d = that/-8, q = round(x/d)+8 clamped 0..15. Used by from_bf16 and
// as the test oracle's encoder.
static inline void q40_quant_block_host(const float* x, float* d_out, uint8_t* nib16) {
    float amax = 0.f, max = 0.f;
    for (int i = 0; i < 32; i++) { float a = fabsf(x[i]); if (a > amax) { amax = a; max = x[i]; } }
    float d = max / -8.0f;
    float id = (d != 0.f) ? 1.0f / d : 0.f;
    *d_out = d;
    uint8_t q[32];
    for (int i = 0; i < 32; i++) {
        int qi = (int)lroundf(x[i] * id) + 8;
        q[i] = (uint8_t)(qi < 0 ? 0 : (qi > 15 ? 15 : qi));
    }
    for (int j = 0; j < 16; j++) nib16[j] = (uint8_t)(q[j] | (q[j + 16] << 4));
}

// From a device BF16 weight [out_dim, in_dim] (host round-trip; for the safetensors
// path and tests). Produces the IDENTICAL SoA layout as from_gguf.
inline bool e4b_q4_0_from_bf16(const __nv_bfloat16* d_W, int out_dim, int in_dim, Q40Weight* w) {
    if (!w || in_dim <= 0 || out_dim <= 0 || (in_dim & 31)) return false;
    const size_t n = (size_t)out_dim * in_dim;
    std::vector<__nv_bfloat16> hw(n);
    if (cudaMemcpy(hw.data(), d_W, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    const int nb = in_dim / 32;
    std::vector<uint8_t> blocks((size_t)out_dim * nb * 18);
    for (size_t blk = 0; blk < (size_t)out_dim * nb; blk++) {
        float xb[32];
        for (int i = 0; i < 32; i++) xb[i] = __bfloat162float(hw[blk * 32 + i]);
        float d; uint8_t nib[16];
        q40_quant_block_host(xb, &d, nib);
        uint8_t* b = &blocks[blk * 18];
        uint16_t hr; { __half hh = __float2half(d); memcpy(&hr, &hh, 2); }
        b[0] = (uint8_t)(hr & 0xFF); b[1] = (uint8_t)(hr >> 8);
        memcpy(b + 2, nib, 16);
    }
    return e4b_q4_0_from_gguf(blocks.data(), out_dim, in_dim, w);
}

// ── GEMV launch + bf16 wrapper ────────────────────────────────────────────────
// Scratch (caller-owned, sized to the largest in_dim/out_dim used): qa[in], da[in/32],
// sa[in/32], yf[out]. Kept off the per-call malloc path like the nvfp4 wrapper.
static inline void q40_gemv_launch(float* yf, const Q40Weight& w, const int8_t* qa,
                                   const float* da, const int32_t* sa, cudaStream_t s) {
    const int per = E4B_Q40_WARPS * E4B_Q40_ROWS;
    unsigned blocks = (unsigned)((w.out_dim + per - 1) / per);
    q40_gemv_kernel<<<blocks, 32 * E4B_Q40_WARPS, 0, s>>>(yf, w.qs, w.d, qa, da, sa, w.in_dim, w.out_dim);
}

// y[out] (bf16) = W · x[in] (bf16). yf holds the f32 result; pass yf to a _f32 caller
// (e.g. the logits head) to skip the bf16 cast.
inline void e4b_q4_0_gemv_f32(float* yf, const Q40Weight& w, const __nv_bfloat16* x,
                              int8_t* qa, float* da, int32_t* sa, cudaStream_t s) {
    q8_quant_kernel<<<w.in_dim / 32, 32, 0, s>>>(x, qa, da, sa);
    q40_gemv_launch(yf, w, qa, da, sa, s);
}

inline void e4b_q4_0_gemv_bf16(__nv_bfloat16* y, const Q40Weight& w, const __nv_bfloat16* x,
                               int8_t* qa, float* da, int32_t* sa, float* yf, cudaStream_t s) {
    e4b_q4_0_gemv_f32(yf, w, x, qa, da, sa, s);
    q40_to_bf16<<<(w.out_dim + 255) / 256, 256, 0, s>>>(yf, y, w.out_dim);
}

} // namespace e4bq40

#endif // FUCINA_E4B_Q4_0_CUH
