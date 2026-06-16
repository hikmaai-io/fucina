// kv_quant_explore.cc — offline comparison of KV-cache quantization schemes for
// the fucina Gemma-4 engine. Phase 6 ("KV quant refinement"): decide whether to
// move the KV cache off flat 8-bit FP8 E4M3 (today's storage, no scaling) onto a
// more accurate / smaller codec before paying for a hot-path kernel rewrite.
//
// Schemes compared, head-to-head on the SAME synthetic post-norm K/V vectors:
//   fp8            current engine: raw E4M3, 8.0 bit/elem, no scale            (baseline)
//   fp8_pertok     E4M3 with one fp16 per-vector amax scale, 8.06 bit/elem
//   nvfp4          E2M1 + per-16 E4M3 block scale (the weight codec), 4.5 bit
//   tq_mse_b4      TurboQuant-MSE (arXiv 2504.19874): randomized Hadamard rotation
//                  + per-coord Lloyd-Max normal centroids, 4 bit + fp16 norm
//   tq_mse_b3      same, 3 bit + fp16 norm
//
// We report, per scheme: relative MSE  E||x-x~||^2 / E||x||^2, mean cosine error
// 1-cos(x,x~), inner-product bias (the thing MSE-optimal quant gets wrong, which
// TurboQuant-prod's QJL residual would fix), inner-product RMS error vs a random
// query, and the effective bits/elem. The inner-product metrics are what actually
// drives attention logits, so they are the decision metric, not raw MSE.
//
// Host-only (no CUDA): E4M3 and E2M1 are emulated with round-to-nearest so this
// builds and runs anywhere. The FP8 emulation matches __nv_cvt_float_to_fp8's
// SATFINITE round-to-nearest closely enough for a relative ranking. Build:
//   make kv-quant-explore   (or: c++ -O2 -o kv_quant_explore cuda/kv_quant_explore.cc -lm)
//
// fucina is research/lab code (CONTRIBUTING.md): this stays an offline tool — it
// does not link the engine and does not touch the stabilized inference path.

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

// ─── FP8 E4M3 emulation (1-4-3, bias 7, max 448, SATFINITE round-to-nearest) ───
// Returns the round-tripped float (quantize then dequantize), matching how the
// engine stores then reads each KV element.
static float quant_e4m3(float x) {
    if (x == 0.0f) return 0.0f;
    float s = x < 0 ? -1.0f : 1.0f;
    x = fabsf(x);
    if (x >= 448.0f) return s * 448.0f;             // saturate to E4M3 max
    int e = (int)floorf(log2f(x));
    if (e < -6) {                                   // subnormal: step = 2^-9
        float step = ldexpf(1.0f, -9);
        float q = roundf(x / step) * step;
        return s * q;
    }
    if (e > 8) e = 8;
    float step = ldexpf(1.0f, e - 3);               // 3 mantissa bits at this exponent
    float q = roundf(x / step) * step;
    if (q >= 448.0f) q = 448.0f;
    return s * q;
}

// ─── FP4 E2M1 emulation (1-2-1, values {0,.5,1,1.5,2,3,4,6}) ───
static float quant_e2m1(float v) {
    static const float lv[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float s = v < 0 ? -1.0f : 1.0f;
    v = fabsf(v);
    float best = lv[0], bd = fabsf(v - lv[0]);
    for (int i = 1; i < 8; i++) { float d = fabsf(v - lv[i]); if (d < bd) { bd = d; best = lv[i]; } }
    return s * best;
}

// ─── Fast Walsh-Hadamard transform (in place, n a power of 2, unnormalized) ───
static void fwht(float* a, int n) {
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += (len << 1))
            for (int j = i; j < i + len; j++) {
                float u = a[j], w = a[j + len];
                a[j] = u + w; a[j + len] = u - w;
            }
}

// Randomized Hadamard transform Π = (1/√n) · H · D  (orthonormal). D is a fixed
// ±1 diagonal (data-oblivious, shared across all tokens). Π is its own structure
// of inverse: Π^T y = D · ((1/√n) H y) since H is symmetric and HH = nI.
struct RHT {
    int n; float inv_sqrt_n; std::vector<float> sign;
    void init(int n_, uint64_t seed) {
        n = n_; inv_sqrt_n = 1.0f / sqrtf((float)n);
        sign.resize(n);
        std::mt19937_64 g(seed);
        for (int i = 0; i < n; i++) sign[i] = (g() & 1) ? 1.0f : -1.0f;
    }
    void fwd(const float* x, float* y) const {          // y = Π x
        for (int i = 0; i < n; i++) y[i] = x[i] * sign[i];
        fwht(y, n);
        for (int i = 0; i < n; i++) y[i] *= inv_sqrt_n;
    }
    void inv(const float* y, float* x) const {          // x = Π^T y
        for (int i = 0; i < n; i++) x[i] = y[i];
        fwht(x, n);
        for (int i = 0; i < n; i++) x[i] *= inv_sqrt_n * sign[i];
    }
};

// ─── Lloyd-Max scalar centroids for the standard normal (2^b levels) ───
// The coordinates of a randomly-rotated unit vector are ~ N(0, 1/d); we quantize
// z = y·√d (~N(0,1)) against these centroids and dequant y = c/√d.
static std::vector<float> lloyd_max_normal(int bits) {
    int K = 1 << bits;
    // Fine grid of the standard normal as the empirical density.
    const int G = 200000;
    std::vector<float> g(G);
    std::mt19937_64 r(12345);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int i = 0; i < G; i++) g[i] = nd(r);
    std::sort(g.begin(), g.end());
    // init centroids at quantiles
    std::vector<float> c(K);
    for (int k = 0; k < K; k++) c[k] = g[(size_t)((k + 0.5) / K * G)];
    for (int it = 0; it < 50; it++) {
        std::vector<double> sum(K, 0); std::vector<long> cnt(K, 0);
        for (int i = 0; i < G; i++) {
            // nearest centroid (sorted, but linear is fine for K<=16)
            int best = 0; float bd = fabsf(g[i] - c[0]);
            for (int k = 1; k < K; k++) { float d = fabsf(g[i] - c[k]); if (d < bd) { bd = d; best = k; } }
            sum[best] += g[i]; cnt[best]++;
        }
        for (int k = 0; k < K; k++) if (cnt[k]) c[k] = (float)(sum[k] / cnt[k]);
    }
    return c;
}

static float nearest_centroid(const std::vector<float>& c, float z) {
    float best = c[0], bd = fabsf(z - c[0]);
    for (size_t k = 1; k < c.size(); k++) { float d = fabsf(z - c[k]); if (d < bd) { bd = d; best = c[k]; } }
    return best;
}

// ─── Per-scheme quantize→dequantize of one vector (writes x~ into out) ───
static void q_fp8(const float* x, float* out, int d) {
    for (int i = 0; i < d; i++) out[i] = quant_e4m3(x[i]);
}
static void q_fp8_pertok(const float* x, float* out, int d) {
    float amax = 0; for (int i = 0; i < d; i++) amax = fmaxf(amax, fabsf(x[i]));
    if (amax == 0) { for (int i = 0; i < d; i++) out[i] = 0; return; }
    float s = amax / 448.0f;                          // map amax → E4M3 max
    for (int i = 0; i < d; i++) out[i] = quant_e4m3(x[i] / s) * s;
}
static void q_nvfp4(const float* x, float* out, int d) {
    const int BLK = 16;
    for (int b0 = 0; b0 < d; b0 += BLK) {
        int n = std::min(BLK, d - b0);
        float amax = 0; for (int i = 0; i < n; i++) amax = fmaxf(amax, fabsf(x[b0 + i]));
        if (amax == 0) { for (int i = 0; i < n; i++) out[b0 + i] = 0; continue; }
        float bs = quant_e4m3(amax / 6.0f);           // per-block scale, itself E4M3
        if (bs == 0) bs = amax / 6.0f;
        for (int i = 0; i < n; i++) out[b0 + i] = quant_e2m1(x[b0 + i] / bs) * bs;
    }
}
static void q_turbo(const float* x, float* out, int d, const RHT& rht,
                    const std::vector<float>& cent, std::vector<float>& yscratch,
                    std::vector<float>& yqscratch) {
    float norm = 0; for (int i = 0; i < d; i++) norm += x[i] * x[i];
    norm = sqrtf(norm);
    if (norm == 0) { for (int i = 0; i < d; i++) out[i] = 0; return; }
    float inv = 1.0f / norm, sq = sqrtf((float)d);
    std::vector<float> xu(d);
    for (int i = 0; i < d; i++) xu[i] = x[i] * inv;   // unit vector
    rht.fwd(xu.data(), yscratch.data());              // y = Π x_unit, coords ~ N(0,1/d)
    for (int i = 0; i < d; i++) {
        float z = yscratch[i] * sq;                   // ~ N(0,1)
        yqscratch[i] = nearest_centroid(cent, z) / sq;
    }
    std::vector<float> xr(d);
    rht.inv(yqscratch.data(), xr.data());             // back to original basis
    for (int i = 0; i < d; i++) out[i] = norm * xr[i];
}

struct Metrics { double rel_mse=0, cos_err=0, ip_bias=0, ip_rmse=0, ip_abs=0; int bits_x100=0; };

int main(int argc, char** argv) {
    const int Ns = 4096;                              // vectors per dim
    int dims[2] = {256, 512};                         // sliding / global head_dim
    int n_outlier = argc > 1 ? atoi(argv[1]) : 4;     // channels that wreck per-tensor FP8
    float outlier_std = argc > 2 ? atof(argv[2]) : 6.0f;
    uint64_t seed = 0xC0FFEE;

    printf("# fucina KV-quant exploration (Phase 6). %d vectors/dim, %d outlier channels @ std %.0f\n",
           Ns, n_outlier, outlier_std);
    printf("# rel_mse = E||x-x~||^2/E||x||^2 ; cos_err = 1-cos ; ip_bias/ip_rmse vs random unit query\n\n");

    std::vector<float> cent_b4 = lloyd_max_normal(4);
    std::vector<float> cent_b3 = lloyd_max_normal(3);

    for (int di = 0; di < 2; di++) {
        int d = dims[di];
        RHT rht; rht.init(d, seed + d);
        std::mt19937_64 r(seed ^ (0x9E3779B97F4A7C15ull * d));
        std::normal_distribution<float> nd(0.0f, 1.0f);

        // outlier channel mask (a few high-variance coords, as in real K caches)
        std::vector<float> chan_std(d, 1.0f);
        for (int k = 0; k < n_outlier; k++) chan_std[(size_t)(r() % d)] = outlier_std;

        const char* names[5] = {"fp8", "fp8_pertok", "nvfp4", "tq_mse_b4", "tq_mse_b3"};
        Metrics m[5];
        double bits[5] = {8.0, 8.0 + 16.0/d, 4.0 + 8.0/16.0, 4.0 + 16.0/d, 3.0 + 16.0/d};
        for (int s = 0; s < 5; s++) m[s].bits_x100 = (int)lround(bits[s] * 100);

        std::vector<float> x(d), out(d), q(d), ys(d), yq(d);
        double sum_xnorm2 = 0;
        for (int v = 0; v < Ns; v++) {
            for (int i = 0; i < d; i++) x[i] = nd(r) * chan_std[i];
            double xn2 = 0; for (int i = 0; i < d; i++) xn2 += (double)x[i]*x[i];
            sum_xnorm2 += xn2;
            // a fresh random unit query for the inner-product metrics
            double qn = 0; for (int i = 0; i < d; i++) { q[i] = nd(r); qn += (double)q[i]*q[i]; }
            qn = sqrt(qn); for (int i = 0; i < d; i++) q[i] /= (float)qn;
            double qx = 0; for (int i = 0; i < d; i++) qx += (double)q[i]*x[i];

            for (int s = 0; s < 5; s++) {
                switch (s) {
                    case 0: q_fp8(x.data(), out.data(), d); break;
                    case 1: q_fp8_pertok(x.data(), out.data(), d); break;
                    case 2: q_nvfp4(x.data(), out.data(), d); break;
                    case 3: q_turbo(x.data(), out.data(), d, rht, cent_b4, ys, yq); break;
                    case 4: q_turbo(x.data(), out.data(), d, rht, cent_b3, ys, yq); break;
                }
                double e2 = 0, dot = 0, on2 = 0, qxt = 0;
                for (int i = 0; i < d; i++) {
                    double e = (double)x[i] - out[i];
                    e2 += e*e; dot += (double)x[i]*out[i]; on2 += (double)out[i]*out[i];
                    qxt += (double)q[i]*out[i];
                }
                m[s].rel_mse += e2;
                m[s].cos_err += (xn2>0 && on2>0) ? (1.0 - dot/sqrt(xn2*on2)) : 0.0;
                double ipe = qxt - qx;
                m[s].ip_bias += ipe;
                m[s].ip_rmse += ipe*ipe;
                m[s].ip_abs  += fabs(qx);
            }
        }

        printf("=== head_dim %d ===\n", d);
        printf("%-12s %10s %10s %12s %12s %8s\n",
               "scheme", "rel_mse", "cos_err", "ip_bias", "ip_rmse", "bit/el");
        for (int s = 0; s < 5; s++) {
            double rel_mse = m[s].rel_mse / sum_xnorm2;
            double cos_err = m[s].cos_err / Ns;
            double ip_bias = m[s].ip_bias / Ns;                 // mean signed IP error
            double ip_rmse = sqrt(m[s].ip_rmse / Ns);           // RMS IP error
            printf("%-12s %10.2e %10.2e %12.2e %12.2e %8.2f\n",
                   names[s], rel_mse, cos_err, ip_bias, ip_rmse, m[s].bits_x100/100.0);
        }
        printf("\n");
    }
    printf("# Read: lower rel_mse/cos_err/ip_rmse = more accurate; ip_bias near 0 = unbiased\n");
    printf("# (MSE-optimal TurboQuant is biased by design; the QJL-residual variant would zero ip_bias)\n");
    return 0;
}
