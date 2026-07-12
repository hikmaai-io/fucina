// ABOUTME: Deterministic DFlash rejection sampler (greedy + probabilistic), host/device shared.
// ABOUTME: Distribution-preserving verification of K draft tokens against target logits.
//
// Given, for one request at a verify step: the K draft token ids, the target logits for each of
// the (1+K) verification positions, the draft logits for each of the K drafted positions (or a
// one-hot marker when only greedy draft ids are available), the per-request seed, and the absolute
// position of each drafted token, this computes:
//   - accepted_len j in [0,K]: the number of leading draft tokens accepted;
//   - the emitted token at the first rejected position (or the bonus token if all K accept).
//
// Greedy (temperature==0): accept draft_i iff it equals argmax(target_i); on first mismatch emit
// target argmax; if all K match emit target argmax of the bonus position. Deterministic outright.
//
// Probabilistic (temperature>0): standard speculative-decoding rejection (Leviathan et al. 2023).
// Accept while p_i(x_i) > u_i * q_i(x_i), with u_i = CounterRNG(seed, pos_i, ACCEPT). On first
// rejection, resample from the residual r(x) = max(p_i(x)-q_i(x),0)/Z using a deterministic
// residual draw at (seed, pos_i, RESIDUAL). If all accept, sample the bonus from the target
// distribution at (seed, pos_bonus, SAMPLE). All draws are stateless (seed, position, domain) so
// the verifier reproduces the draft's stream independently. This is a compact host/device
// reference: it materializes softmax over the provided vocab slice. The production CUDA path may
// fuse these steps, but must match this oracle bit-for-bit on the same inputs.
#ifndef FUCINA_QWEN35_DFLASH_REJECT_CUH
#define FUCINA_QWEN35_DFLASH_REJECT_CUH

#include <cstdint>
#include <cmath>
#include "qwen35_dflash_rng.cuh"

// Result of verifying one request's K-token draft block.
struct q35_dflash_verify_result {
    int     accepted_len;   // j in [0, K]
    int32_t emitted_token;  // token emitted at position j (residual/argmax) or the bonus token
};

// Argmax over a contiguous logits row [0, vocab). Deterministic tie-break: lowest index wins.
Q35_DFLASH_HD static inline int32_t q35_dflash_argmax(const float *logits, int vocab) {
    int32_t best = 0; float bestv = logits[0];
    for (int v = 1; v < vocab; v++) {
        float lv = logits[v];
        if (lv > bestv) { bestv = lv; best = v; }
    }
    return best;
}

// Softmax probability of token `tok` in a logits row, computed in a numerically stable way
// (subtract max). Host/device identical (double accumulation).
Q35_DFLASH_HD static inline double q35_dflash_softmax_prob(const float *logits, int vocab,
                                                           int32_t tok) {
    double mx = logits[0];
    for (int v = 1; v < vocab; v++) if ((double)logits[v] > mx) mx = logits[v];
    double sum = 0.0;
    for (int v = 0; v < vocab; v++) sum += exp((double)logits[v] - mx);
    double num = exp((double)logits[tok] - mx);
    return num / sum;
}

// Greedy verification. target_logits is (1+K) rows of `vocab`; draft_tokens is K ids.
Q35_DFLASH_HD static inline q35_dflash_verify_result q35_dflash_verify_greedy(
        const float *target_logits, int vocab, const int32_t *draft_tokens, int K) {
    q35_dflash_verify_result r; r.accepted_len = 0; r.emitted_token = -1;
    for (int i = 0; i < K; i++) {
        int32_t targ = q35_dflash_argmax(target_logits + (size_t)i * vocab, vocab);
        if (targ == draft_tokens[i]) {
            r.accepted_len = i + 1;
        } else {
            r.emitted_token = targ;   // emit target argmax at the first mismatch
            return r;
        }
    }
    // All K accepted: emit the bonus token = argmax of the (K+1)-th target row.
    r.emitted_token = q35_dflash_argmax(target_logits + (size_t)K * vocab, vocab);
    return r;
}

// Probabilistic verification. target_logits and draft_logits are per-position rows of `vocab`
// (draft_logits has K rows, one per drafted position; target_logits has 1+K rows). pos[i] is the
// absolute position of drafted token i; pos_bonus is the absolute position of the bonus token.
Q35_DFLASH_HD static inline q35_dflash_verify_result q35_dflash_verify_prob(
        const float *target_logits, const float *draft_logits, int vocab,
        const int32_t *draft_tokens, int K, uint64_t seed,
        const int64_t *pos, int64_t pos_bonus) {
    q35_dflash_verify_result r; r.accepted_len = 0; r.emitted_token = -1;
    for (int i = 0; i < K; i++) {
        const float *trow = target_logits + (size_t)i * vocab;
        const float *drow = draft_logits + (size_t)i * vocab;
        int32_t x = draft_tokens[i];
        double p = q35_dflash_softmax_prob(trow, vocab, x);
        double q = q35_dflash_softmax_prob(drow, vocab, x);
        double u = q35_dflash_uniform_open(seed, pos[i], Q35_DFLASH_DOMAIN_ACCEPT);
        // Accept iff p > u*q (q>0 since it produced x). Equivalent stable log form.
        if (p > u * q) {
            r.accepted_len = i + 1;
            continue;
        }
        // Rejected at i: resample from residual r(x) = max(p(x)-q(x),0)/Z via inverse-CDF with a
        // deterministic residual draw. Compute Z and the CDF in one stable pass.
        double mt = trow[0], md = drow[0];
        for (int v = 1; v < vocab; v++) { if ((double)trow[v]>mt) mt=trow[v]; if ((double)drow[v]>md) md=drow[v]; }
        double st = 0.0, sd = 0.0;
        for (int v = 0; v < vocab; v++) { st += exp((double)trow[v]-mt); sd += exp((double)drow[v]-md); }
        double Z = 0.0;
        for (int v = 0; v < vocab; v++) {
            double pv = exp((double)trow[v]-mt)/st;
            double qv = exp((double)drow[v]-md)/sd;
            double res = pv - qv; if (res < 0.0) res = 0.0;
            Z += res;
        }
        double ur = q35_dflash_uniform_open(seed, pos[i], Q35_DFLASH_DOMAIN_RESIDUAL);
        int32_t emit = q35_dflash_argmax(trow, vocab);  // fallback if Z underflows
        if (Z > 0.0) {
            double target_c = ur * Z, acc = 0.0;
            for (int v = 0; v < vocab; v++) {
                double pv = exp((double)trow[v]-mt)/st;
                double qv = exp((double)drow[v]-md)/sd;
                double res = pv - qv; if (res < 0.0) res = 0.0;
                acc += res;
                if (acc >= target_c) { emit = v; break; }
            }
        }
        r.emitted_token = emit;
        return r;
    }
    // All K accepted: sample the bonus token from the target distribution at pos_bonus.
    const float *brow = target_logits + (size_t)K * vocab;
    double mt = brow[0];
    for (int v = 1; v < vocab; v++) if ((double)brow[v]>mt) mt=brow[v];
    double st = 0.0;
    for (int v = 0; v < vocab; v++) st += exp((double)brow[v]-mt);
    double ub = q35_dflash_uniform_open(seed, pos_bonus, Q35_DFLASH_DOMAIN_SAMPLE);
    double target_c = ub * st, acc = 0.0;
    int32_t emit = q35_dflash_argmax(brow, vocab);
    for (int v = 0; v < vocab; v++) {
        acc += exp((double)brow[v]-mt);
        if (acc >= target_c) { emit = v; break; }
    }
    r.emitted_token = emit;
    r.accepted_len = K;
    return r;
}

#endif // FUCINA_QWEN35_DFLASH_REJECT_CUH
