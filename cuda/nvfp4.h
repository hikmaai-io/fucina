// NVFP4 reconstruction math — the two-level microscaling dequant shared by the safetensors
// loader (validation / bf16 materialization) and the decode GEMV kernel.
//
// A value stored in an NVFP4 (ModelOpt) checkpoint reconstructs as:
//
//     real = e2m1_decode(nibble) * block_scale_e4m3 * weight_scale_2
//
//   • nibble            4-bit E2M1 (1 sign, 2 exp, 1 mantissa), max magnitude 6.0
//   • block_scale_e4m3  one F8_E4M3 value per 16 input elements (max 448), stored LINEAR
//   • weight_scale_2    per-tensor FP32 scalar = global_amax / (6*448)
//
// `weight_scale2` here is always the DECODE MULTIPLIER (real = e2m1·block·multiplier). The two
// producer conventions store DIFFERENT raw scalars — VERIFIED on RedHatAI/gemma-4-12B-it-NVFP4
// (q_proj raw global = 7392): compressed-tensors stores the LARGE reciprocal (6*448)/amax and the
// caller must pass 1/raw; ModelOpt stores the SMALL amax/(6*448) and passes it as-is. Normalize at
// load with nvfp4ld::global_mul() so everything downstream multiplies uniformly.
//
// Host helpers here are the correctness ORACLE (used in tests + optional CPU validation).
// The device decode path uses CUDA's __nv_cvt_fp8_to_halfraw for E4M3 (matches the engine's
// existing NVFP4 code); this header keeps a software E4M3 decode so host code has no CUDA dep.
#ifndef FUCINA_NVFP4_H
#define FUCINA_NVFP4_H

#include <cstdint>

#ifdef __CUDACC__
#define NVFP4_HD __host__ __device__ __forceinline__
#else
#define NVFP4_HD inline
#endif

namespace nvfp4 {

// ── E2M1 decode: 4-bit nibble → float. bit3 = sign, bits2..0 index the magnitude LUT. ──
NVFP4_HD float e2m1_decode(uint32_t nib) {
    // magnitudes for codes 0..7: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float v = lut[nib & 7u];
    return (nib & 8u) ? -v : v;
}

// ── Software E4M3 (float8_e4m3fn) → float, for host-side scale decode. ──
// 1 sign | 4 exp (bias 7) | 3 mantissa. No inf; 0xFF/0x7F are NaN. Subnormals: exp==0.
// Max finite magnitude = 448 (S.1111.110). 0x7F/0xFF (S.1111.111) = NaN.
inline float e4m3_decode(uint8_t b) {
    uint32_t sign = (b >> 7) & 1u;
    uint32_t exp  = (b >> 3) & 0xFu;
    uint32_t man  = b & 0x7u;
    float s = sign ? -1.f : 1.f;
    if (exp == 0) {                       // subnormal: value = man/8 * 2^(1-7)
        return s * (float)man / 8.0f * 0.015625f;   // 2^-6 = 0.015625
    }
    if (exp == 0xF && man == 0x7) {       // S.1111.111 = NaN (the only NaN in e4m3fn)
        return s * (0.0f / 0.0f);
    }
    // normal: value = (1 + man/8) * 2^(exp-7)
    float mant = 1.0f + (float)man / 8.0f;
    int e = (int)exp - 7;
    // ldexp without <cmath>: multiply by 2^e
    float scale = 1.0f;
    if (e >= 0) for (int i = 0; i < e; i++) scale *= 2.0f;
    else        for (int i = 0; i < -e; i++) scale *= 0.5f;
    return s * mant * scale;
}

// ── Full reconstruction of one element. ──
//   nibble        the 4-bit E2M1 code for this element
//   block_e4m3    the raw F8_E4M3 byte covering this element's 16-block
//   weight_scale2 per-tensor FP32 global (the value stored on disk)
NVFP4_HD float reconstruct(uint32_t nibble, float block_scale, float weight_scale2) {
    return e2m1_decode(nibble) * block_scale * weight_scale2;
}

// Host reconstruct that decodes the E4M3 byte itself (no CUDA dep).
inline float reconstruct_host(uint32_t nibble, uint8_t block_e4m3, float weight_scale2) {
    return e2m1_decode(nibble) * e4m3_decode(block_e4m3) * weight_scale2;
}

// ── Packed-weight element access ──
// ModelOpt `.weight` is U8 [out, in/2]: byte j of a row holds element 2j in the LOW nibble
// and element 2j+1 in the HIGH nibble. `row` points at the start of the out-row (in/2 bytes).
NVFP4_HD uint32_t nibble_at(const uint8_t* row, int col /*0..in-1*/) {
    uint8_t byte = row[col >> 1];
    return (col & 1) ? (uint32_t)(byte >> 4) : (uint32_t)(byte & 0x0F);
}

// Dequant a full row [in] of a ModelOpt NVFP4 weight to float (host reference / oracle).
//   wrow   in/2 packed bytes
//   srow   in/16 E4M3 block-scale bytes (linear)
//   gs     weight_scale_2 (scalar)
//   out    in floats
inline void dequant_row_host(const uint8_t* wrow, const uint8_t* srow, float gs,
                             int in_dim, float* out) {
    for (int c = 0; c < in_dim; c++) {
        uint8_t sb = srow[c >> 4];                 // 16 elems per block scale
        out[c] = reconstruct_host(nibble_at(wrow, c), sb, gs);
    }
}

} // namespace nvfp4

#endif // FUCINA_NVFP4_H
