// ABOUTME: CUDA<->CPU parity gate for the DFlash counter RNG + rejection sampler (P1 of S1a).
// ABOUTME: Runs the shared header on-device and asserts bit-identical results vs the host oracle.
//
// The DFlash determinism guarantee relies on the draft (device) and verifier (device) deriving the
// SAME draw as the host oracle for any (seed, absolute_position, domain). This gate runs the exact
// same __host__ __device__ header on the GPU and compares every RNG bit pattern and every rejection
// result (accepted_len + emitted token) against the CPU computation, bit-for-bit.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_parity.cu -o t && ./t
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "qwen35_dflash_rng.cuh"
#include "qwen35_dflash_reject.cuh"

#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    printf("CUDA error %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 2; } } while(0)

// ── Device kernels exercising the shared header ──
__global__ void k_prf(const uint64_t *seeds, const int64_t *pos, const uint32_t *dom,
                      uint64_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = q35_dflash_prf(seeds[i], pos[i], dom[i]);
}
__global__ void k_uniform(const uint64_t *seeds, const int64_t *pos, const uint32_t *dom,
                          double *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = q35_dflash_uniform_open(seeds[i], pos[i], dom[i]);
}
__global__ void k_verify_greedy(const float *tl, int vocab, const int32_t *dt, int K,
                                int *acc, int32_t *emit) {
    auto r = q35_dflash_verify_greedy(tl, vocab, dt, K);
    *acc = r.accepted_len; *emit = r.emitted_token;
}
__global__ void k_verify_prob(const float *tl, const float *dl, int vocab, const int32_t *dt,
                             int K, uint64_t seed, const int64_t *pos, int64_t posb,
                             int *acc, int32_t *emit) {
    auto r = q35_dflash_verify_prob(tl, dl, vocab, dt, K, seed, pos, posb);
    *acc = r.accepted_len; *emit = r.emitted_token;
}

int main() {
    int fails = 0;

    // ── RNG bit parity over a spread of (seed, position, domain) ──
    std::vector<uint64_t> seeds; std::vector<int64_t> pos; std::vector<uint32_t> dom;
    for (uint64_t s : {0ull, 1ull, 0x12345678ull, 0xDEADBEEFCAFEBABEull, 0x9e3779b97f4a7c15ull})
        for (int64_t p : {0, 1, 2, 7, 63, 100, 1000, 262143})
            for (uint32_t d : {Q35_DFLASH_DOMAIN_ACCEPT, Q35_DFLASH_DOMAIN_RESIDUAL, Q35_DFLASH_DOMAIN_SAMPLE}) {
                seeds.push_back(s); pos.push_back(p); dom.push_back(d);
            }
    int n = (int)seeds.size();
    uint64_t *d_seeds; int64_t *d_pos; uint32_t *d_dom; uint64_t *d_bits; double *d_u;
    CUDA_OK(cudaMalloc(&d_seeds, n*sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&d_pos,   n*sizeof(int64_t)));
    CUDA_OK(cudaMalloc(&d_dom,   n*sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_bits,  n*sizeof(uint64_t)));
    CUDA_OK(cudaMalloc(&d_u,     n*sizeof(double)));
    CUDA_OK(cudaMemcpy(d_seeds, seeds.data(), n*sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_pos,   pos.data(),   n*sizeof(int64_t),  cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_dom,   dom.data(),   n*sizeof(uint32_t), cudaMemcpyHostToDevice));
    int tpb = 128, blocks = (n + tpb - 1) / tpb;
    k_prf<<<blocks, tpb>>>(d_seeds, d_pos, d_dom, d_bits, n);
    k_uniform<<<blocks, tpb>>>(d_seeds, d_pos, d_dom, d_u, n);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint64_t> gbits(n); std::vector<double> gu(n);
    CUDA_OK(cudaMemcpy(gbits.data(), d_bits, n*sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(gu.data(),    d_u,    n*sizeof(double),   cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) {
        uint64_t hb = q35_dflash_prf(seeds[i], pos[i], dom[i]);
        double   hu = q35_dflash_uniform_open(seeds[i], pos[i], dom[i]);
        if (hb != gbits[i]) { printf("FAIL prf parity @%d host=%016llx dev=%016llx\n", i,
                                     (unsigned long long)hb, (unsigned long long)gbits[i]); fails++; }
        // Bit-identical double compare (same integer->double path both sides).
        if (hu != gu[i]) { printf("FAIL uniform parity @%d host=%.17g dev=%.17g\n", i, hu, gu[i]); fails++; }
    }
    printf("RNG parity: %d (seed,pos,domain) triples, prf+uniform bit-identical host==device\n", n);

    // ── Greedy verify parity ──
    {
        const int vocab = 16, K = 4;
        std::vector<float> tl((size_t)(K+1)*vocab, 0.0f);
        int argmaxes[5] = {5, 2, 7, 2, 11};
        for (int row = 0; row <= K; row++)
            for (int v = 0; v < vocab; v++) tl[(size_t)row*vocab+v] = (v==argmaxes[row])?10.0f:(float)(v%3)*0.1f;
        // A draft that matches the first 3 then diverges.
        int32_t dt[4] = {5, 2, 7, 9};
        float *d_tl; int32_t *d_dt; int *d_acc; int32_t *d_emit;
        CUDA_OK(cudaMalloc(&d_tl, tl.size()*sizeof(float)));
        CUDA_OK(cudaMalloc(&d_dt, K*sizeof(int32_t)));
        CUDA_OK(cudaMalloc(&d_acc, sizeof(int)));
        CUDA_OK(cudaMalloc(&d_emit, sizeof(int32_t)));
        CUDA_OK(cudaMemcpy(d_tl, tl.data(), tl.size()*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_dt, dt, K*sizeof(int32_t), cudaMemcpyHostToDevice));
        k_verify_greedy<<<1,1>>>(d_tl, vocab, d_dt, K, d_acc, d_emit);
        CUDA_OK(cudaDeviceSynchronize());
        int gacc; int32_t gemit;
        CUDA_OK(cudaMemcpy(&gacc, d_acc, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(&gemit, d_emit, sizeof(int32_t), cudaMemcpyDeviceToHost));
        auto h = q35_dflash_verify_greedy(tl.data(), vocab, dt, K);
        if (gacc != h.accepted_len || gemit != h.emitted_token) {
            printf("FAIL greedy parity host(len=%d emit=%d) dev(len=%d emit=%d)\n",
                   h.accepted_len, h.emitted_token, gacc, gemit); fails++;
        } else printf("greedy parity: len=%d emit=%d host==device\n", gacc, gemit);
        cudaFree(d_tl); cudaFree(d_dt); cudaFree(d_acc); cudaFree(d_emit);
    }

    // ── Probabilistic verify parity over several seeds ──
    {
        const int vocab = 12, K = 3;
        std::vector<float> tl((size_t)(K+1)*vocab), dl((size_t)K*vocab);
        for (int row = 0; row <= K; row++)
            for (int v = 0; v < vocab; v++) tl[(size_t)row*vocab+v] = (float)((v*7 + row*3) % 11) * 0.4f;
        for (int row = 0; row < K; row++)
            for (int v = 0; v < vocab; v++) dl[(size_t)row*vocab+v] = (float)((v*5 + row*2) % 9) * 0.5f;
        int32_t dt[3];
        for (int i = 0; i < K; i++) dt[i] = q35_dflash_argmax(&dl[(size_t)i*vocab], vocab);
        int64_t pos[3] = {500, 501, 502}, posb = 503;
        float *d_tl,*d_dl; int32_t *d_dt; int64_t *d_pos; int *d_acc; int32_t *d_emit;
        CUDA_OK(cudaMalloc(&d_tl, tl.size()*sizeof(float)));
        CUDA_OK(cudaMalloc(&d_dl, dl.size()*sizeof(float)));
        CUDA_OK(cudaMalloc(&d_dt, K*sizeof(int32_t)));
        CUDA_OK(cudaMalloc(&d_pos, K*sizeof(int64_t)));
        CUDA_OK(cudaMalloc(&d_acc, sizeof(int)));
        CUDA_OK(cudaMalloc(&d_emit, sizeof(int32_t)));
        CUDA_OK(cudaMemcpy(d_tl, tl.data(), tl.size()*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_dl, dl.data(), dl.size()*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_dt, dt, K*sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_pos, pos, K*sizeof(int64_t), cudaMemcpyHostToDevice));
        int nseed = 0, agree = 0;
        for (uint64_t seed : {1ull, 2ull, 3ull, 42ull, 1000ull, 0xABCDEFull, 0xFFFFFFFFull, 7777ull}) {
            k_verify_prob<<<1,1>>>(d_tl, d_dl, vocab, d_dt, K, seed, d_pos, posb, d_acc, d_emit);
            CUDA_OK(cudaDeviceSynchronize());
            int gacc; int32_t gemit;
            CUDA_OK(cudaMemcpy(&gacc, d_acc, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(&gemit, d_emit, sizeof(int32_t), cudaMemcpyDeviceToHost));
            auto h = q35_dflash_verify_prob(tl.data(), dl.data(), vocab, dt, K, seed, pos, posb);
            nseed++;
            if (gacc != h.accepted_len || gemit != h.emitted_token) {
                printf("FAIL prob parity seed=%llu host(len=%d emit=%d) dev(len=%d emit=%d)\n",
                       (unsigned long long)seed, h.accepted_len, h.emitted_token, gacc, gemit); fails++;
            } else agree++;
        }
        printf("prob parity: %d/%d seeds bit-identical host==device\n", agree, nseed);
        cudaFree(d_tl); cudaFree(d_dl); cudaFree(d_dt); cudaFree(d_pos); cudaFree(d_acc); cudaFree(d_emit);
    }

    cudaFree(d_seeds); cudaFree(d_pos); cudaFree(d_dom); cudaFree(d_bits); cudaFree(d_u);
    if (fails) { printf("FAIL — DFlash RNG/rejection CUDA parity (%d mismatches)\n", fails); return 1; }
    printf("PASS — DFlash RNG/rejection CUDA parity: RNG bits, greedy, and probabilistic rejection "
           "bit-identical CPU vs CUDA\n");
    return 0;
}
