// ABOUTME: Deterministic stateless counter RNG for DFlash speculative decoding, keyed by
// ABOUTME: (request_seed, absolute_position, domain) so draft and verifier derive identical draws.
//
// DFlash keeps byte-identical determinism THROUGH rejection sampling by having the draft and the
// verifier independently derive the SAME pseudo-random draw for a given token position, with no
// shared mutable RNG state. The only inputs are the per-request seed, the token's ABSOLUTE
// position in the sequence, and a small domain separator (so the acceptance uniform, the residual
// resample uniform, and the plain sample uniform at one position never collide).
//
// Implementation is pure integer mixing (splitmix64-based, three independent finalizer rounds
// forming a counter-based PRF) so it is BIT-IDENTICAL on host and device: no floating-point in the
// state path, and a single fixed integer->float conversion at the end. This header is
// `__host__ __device__` when compiled by nvcc and plain host C++ otherwise, so the CPU oracle and
// the CUDA kernel share one source of truth.
#ifndef FUCINA_QWEN35_DFLASH_RNG_CUH
#define FUCINA_QWEN35_DFLASH_RNG_CUH

#include <cstdint>

#if defined(__CUDACC__)
#define Q35_DFLASH_HD __host__ __device__
#else
#define Q35_DFLASH_HD
#endif

// Domain separators keep the uniform streams at one (seed, position) independent.
enum {
    Q35_DFLASH_DOMAIN_ACCEPT   = 0x0u,  // acceptance test uniform u for probabilistic rejection
    Q35_DFLASH_DOMAIN_RESIDUAL = 0x1u,  // residual-distribution resample after a rejection
    Q35_DFLASH_DOMAIN_SAMPLE   = 0x2u,  // plain per-position sample (draft/bonus)
};

// splitmix64 finalizer — a strong integer avalanche. Used as the mixing primitive.
Q35_DFLASH_HD static inline uint64_t q35_dflash_splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Counter-based PRF: PRF(seed, position, domain) -> 64 uniform bits. Stateless and pure. The three
// inputs are folded into one 64-bit block, then run through three chained splitmix64 finalizers
// keyed by the seed so distinct seeds diverge fully. absolute_position is taken as int64 but
// interpreted as its unsigned bit pattern; negative positions never occur in decode.
Q35_DFLASH_HD static inline uint64_t q35_dflash_prf(uint64_t seed, int64_t absolute_position,
                                                    uint32_t domain) {
    uint64_t pos = (uint64_t)absolute_position;
    // Distinct odd multipliers spread position and domain across all bits before mixing.
    uint64_t block = pos * 0xD1B54A32D192ED03ULL
                   + (uint64_t)domain * 0xCA5A826395121157ULL
                   + 0x2545F4914F6CDD1DULL;
    uint64_t k = seed * 0x9E3779B97F4A7C15ULL + 0x165667B19E3779F9ULL;
    uint64_t x = q35_dflash_splitmix64(block ^ k);
    x = q35_dflash_splitmix64(x + seed);
    x = q35_dflash_splitmix64(x ^ (k + 0x27D4EB2F165667C5ULL));
    return x;
}

// Uniform double in the OPEN interval (0,1): the top 53 bits mapped to [0,1) then nudged off zero
// so log(u) is always finite (the acceptance test uses log(u)). Deterministic, host/device equal.
Q35_DFLASH_HD static inline double q35_dflash_uniform_open(uint64_t seed, int64_t absolute_position,
                                                           uint32_t domain) {
    uint64_t bits = q35_dflash_prf(seed, absolute_position, domain);
    // 53-bit mantissa: [0, 2^53) -> [0,1).
    double u = (double)(bits >> 11) * (1.0 / 9007199254740992.0);  // 2^-53
    // Nudge strictly inside (0,1): smallest step is 2^-53.
    const double tiny = 1.0 / 9007199254740992.0;
    if (u <= 0.0) u = tiny;
    if (u >= 1.0) u = 1.0 - tiny;
    return u;
}

#endif // FUCINA_QWEN35_DFLASH_RNG_CUH
