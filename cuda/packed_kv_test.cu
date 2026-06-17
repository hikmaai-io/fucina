// packed_kv_test.cu — proves the packed NVFP4 KV storage (paged_kv_device.cuh,
// pkv_pack_row / pkv_unpack) is BIT-IDENTICAL to the FP8-fake-quant NVFP4 round-trip
// (pkv_nvfp4_roundtrip) that the engine already benches. If they match, swapping the
// 1-byte/elem FP8 cache for the 0.5625-byte/elem packed layout changes ONLY memory,
// never generation numerics — the precondition for making it the default.
//
// Build/run:  make packed-kv-test
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include "paged_kv_device.cuh"

// Pack each row, unpack every element, and compare to the fake-quant round-trip.
__global__ void pack_roundtrip_kernel(const float* rows, int n_rows, int E,
                                      uint8_t* nib, uint8_t* scl, float* unpacked) {
    int r = blockIdx.x;
    if (r >= n_rows) return;
    const float* row = rows + (size_t)r * E;
    uint8_t* rn = nib + (size_t)r * (E / 2);
    uint8_t* rs = scl + (size_t)r * (E / 16);
    if (threadIdx.x == 0) pkv_pack_row(row, E, rn, rs);   // one thread owns the row
    __syncthreads();
    for (int e = threadIdx.x; e < E; e += blockDim.x)
        unpacked[(size_t)r * E + e] = pkv_unpack(rn, rs, e);
}

// The storage-independent reference: the fake-quant the engine already uses/benches.
__global__ void fakequant_kernel(const float* rows, int n_rows, int E, float* out) {
    int r = blockIdx.x;
    if (r >= n_rows) return;
    const float* row = rows + (size_t)r * E;
    for (int e = threadIdx.x; e < E; e += blockDim.x)
        out[(size_t)r * E + e] = pkv_nvfp4_roundtrip(row, e);
}

static bool run(int E, const char* label) {
    const int N = 4096;
    std::mt19937 g(1234 + E);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> h(N * E);
    // Gaussian + a few outlier channels (as real post-norm K/V).
    std::vector<float> cstd(E, 1.f);
    for (int k = 0; k < 4; k++) cstd[g() % E] = 6.f;
    for (int r = 0; r < N; r++)
        for (int e = 0; e < E; e++) h[r * E + e] = nd(g) * cstd[e];

    float *d_rows, *d_unp, *d_fq; uint8_t *d_nib, *d_scl;
    cudaMalloc(&d_rows, sizeof(float) * N * E);
    cudaMalloc(&d_unp,  sizeof(float) * N * E);
    cudaMalloc(&d_fq,   sizeof(float) * N * E);
    cudaMalloc(&d_nib,  (size_t)N * (E / 2));
    cudaMalloc(&d_scl,  (size_t)N * (E / 16));
    cudaMemcpy(d_rows, h.data(), sizeof(float) * N * E, cudaMemcpyHostToDevice);

    pack_roundtrip_kernel<<<N, 256>>>(d_rows, N, E, d_nib, d_scl, d_unp);
    fakequant_kernel<<<N, 256>>>(d_rows, N, E, d_fq);
    cudaDeviceSynchronize();

    std::vector<float> unp(N * E), fq(N * E);
    cudaMemcpy(unp.data(), d_unp, sizeof(float) * N * E, cudaMemcpyDeviceToHost);
    cudaMemcpy(fq.data(),  d_fq,  sizeof(float) * N * E, cudaMemcpyDeviceToHost);

    size_t mism = 0; double rel = 0, xn = 0;
    for (size_t i = 0; i < (size_t)N * E; i++) {
        if (unp[i] != fq[i]) mism++;                 // exact bit compare
        double d = (double)unp[i] - h[i]; rel += d * d; xn += (double)h[i] * h[i];
    }
    double bytes_fp8 = (double)E, bytes_pak = E / 2 + E / 16;
    printf("  [%s] E=%d  packed-vs-fakequant mismatches=%zu/%d  rel_mse=%.2e  "
           "bytes/elem %.3f→%.4f (%.2f× smaller)\n",
           label, E, mism, N * E, rel / xn, bytes_fp8 / E, bytes_pak / E,
           bytes_fp8 / bytes_pak);
    cudaFree(d_rows); cudaFree(d_unp); cudaFree(d_fq); cudaFree(d_nib); cudaFree(d_scl);
    return mism == 0;
}

int main() {
    printf("packed NVFP4 KV storage — bit-faithful to the FP8 fake-quant?\n");
    bool ok = true;
    ok &= run(256, "sliding hd=256");   // GEMMA4_HEAD_DIM
    ok &= run(512, "global  hd=512");   // GEMMA4_GLOBAL_HEAD_DIM
    ok &= run(2048, "sliding row nkv*hd"); // 8*256, full token row
    if (ok) { printf("packed_kv: ALL TESTS PASSED (unpack(pack(x)) == fake-quant, exact)\n"); return 0; }
    printf("packed_kv: FAILED\n");
    return 1;
}
