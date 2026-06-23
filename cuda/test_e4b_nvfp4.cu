// test_e4b_nvfp4.cu — validate + benchmark the E4B NVFP4 weight-path foundation.
//
// Three checks per weight shape:
//   1. KERNEL CORRECTNESS: device NVFP4 GEMV vs a host oracle that dequantizes the
//      SAME packed bytes (nvfp4::dequant_row_host) and dots with x. Must match to
//      float-accumulation tolerance — proves the kernel reads the layout correctly.
//   2. QUANT QUALITY: NVFP4 GEMV vs a full-precision (FP32) reference over the
//      original weights. Reports SNR (dB) and relative L2 — the FP4 error budget.
//   3. BANDWIDTH: GEMV GB/s (weight bytes / time), the decode-relevant metric.
//
// Shapes mirror real E4B FFN projections (hidden 2560, intermediate 10240).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "e4b_nvfp4.cuh"
#include "nvfp4.h"

static uint32_t s_rng = 0x12345678u;
static float frand() { s_rng = s_rng * 1664525u + 1013904223u; return ((s_rng >> 8) / 16777216.0f) * 2.0f - 1.0f; }

static bool run_shape(int out_dim, int in_dim, const char* name) {
    printf("\n== %s : out=%d in=%d ==\n", name, out_dim, in_dim);
    const size_t WN = (size_t)out_dim * in_dim;

    // Host weights (~N(0,0.05), a realistic projection scale) + activation.
    std::vector<float>          hWf(WN);
    std::vector<__nv_bfloat16>  hW(WN);
    std::vector<__nv_bfloat16>  hx(in_dim);
    std::vector<float>          hxf(in_dim);
    for (size_t i = 0; i < WN; i++)     { float v = frand() * 0.05f; hWf[i] = v; hW[i] = __float2bfloat16(v); }
    for (int i = 0; i < in_dim; i++)    { float v = frand();          hxf[i] = v; hx[i] = __float2bfloat16(v); }

    // Device weight + activation.
    __nv_bfloat16 *dW, *dx, *dy; float *dxf, *dyf;
    cudaMalloc(&dW, WN * 2); cudaMalloc(&dx, in_dim * 2); cudaMalloc(&dy, out_dim * 2);
    cudaMalloc(&dxf, in_dim * 4); cudaMalloc(&dyf, out_dim * 4);
    cudaMemcpy(dW, hW.data(), WN * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx.data(), in_dim * 2, cudaMemcpyHostToDevice);

    // Quantize.
    e4bfp4::Weight w;
    if (!e4bfp4::e4b_nvfp4_quantize(dW, out_dim, in_dim, &w)) { printf("  quantize FAILED\n"); return false; }
    cudaDeviceSynchronize();
    printf("  quantized: %.2f MB (%.2f bit/elem vs 16.0 BF16)\n",
           w.bytes / 1e6, 8.0 * w.bytes / (double)WN);

    // Run GEMV.
    e4bfp4::e4b_nvfp4_gemv_bf16(dy, w, dx, dxf, dyf, 0);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("  GEMV launch error: %s\n", cudaGetErrorString(err)); return false; }
    std::vector<float> hy(out_dim);
    cudaMemcpy(hy.data(), dyf, out_dim * 4, cudaMemcpyDeviceToHost); // read the f32 result directly

    // Pull packed bytes + gs back for the host oracle.
    std::vector<uint8_t> hp((size_t)out_dim * (in_dim / 2)), hs((size_t)out_dim * (in_dim / 16));
    float hgs = 0.f;
    cudaMemcpy(hp.data(), w.packed, hp.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(hs.data(), w.scale,  hs.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hgs, w.gs, 4, cudaMemcpyDeviceToHost);

    // (1) kernel vs host oracle over the SAME packed bytes — whole-vector relative L2 (robust to
    //     individual near-zero rows where ± cancellation makes per-element rel error meaningless;
    //     a correct kernel differs from the double-precision oracle only by float accumulation,
    //     ~1e-3, while a layout/math bug is O(1)). (2) full-precision reference for FP4 SNR.
    std::vector<float> row(in_dim);
    double num_k = 0, den_k = 0, num_q = 0, den = 0, max_abs = 0;
    for (int o = 0; o < out_dim; o++) {
        nvfp4::dequant_row_host(&hp[(size_t)o * (in_dim / 2)], &hs[(size_t)o * (in_dim / 16)], hgs, in_dim, row.data());
        double oracle = 0, full = 0;
        for (int k = 0; k < in_dim; k++) { oracle += (double)row[k] * hxf[k]; full += (double)hWf[(size_t)o * in_dim + k] * hxf[k]; }
        num_k  += (hy[o] - oracle) * (hy[o] - oracle); den_k += oracle * oracle;
        max_abs = fmax(max_abs, fabs(hy[o] - oracle));
        num_q  += (full - hy[o]) * (full - hy[o]);     den   += full * full;
    }
    double kern_l2 = sqrt(num_k / (den_k + 1e-12));
    double rel_l2  = sqrt(num_q / (den + 1e-12));
    double snr_db  = 10.0 * log10(den / (num_q + 1e-12));
    bool kern_ok = kern_l2 < 1e-2;
    printf("  kernel vs oracle : rel_L2=%.2e  max_abs=%.2e  %s\n",
           kern_l2, max_abs, kern_ok ? "OK" : "*** FAIL ***");
    printf("  FP4 quant quality: rel_L2=%.4f  SNR=%.1f dB\n", rel_l2, snr_db);

    // (3) bandwidth: time the GEMV alone over many iters.
    const int ITERS = 200;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    e4bfp4::e4b_nvfp4_gemv_bf16(dy, w, dx, dxf, dyf, 0); cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++) e4bfp4::e4b_gemv_launch(dyf, w.packed, w.scale, w.gs, dxf, in_dim, out_dim, 0);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    double wbytes = (double)w.bytes - 4; // packed+scale (gs scalar negligible)
    double gbps = wbytes * ITERS / (ms / 1e3) / 1e9;
    printf("  GEMV bandwidth   : %.1f GB/s  (%.3f ms/call)\n", gbps, ms / ITERS);

    e4bfp4::weight_free(&w);
    cudaFree(dW); cudaFree(dx); cudaFree(dy); cudaFree(dxf); cudaFree(dyf);
    return kern_ok;
}

// FP8 (E4M3) path: device GEMV vs host oracle (e4m3·rowscale) + FP4-vs-FP8 SNR contrast + BW.
static bool run_shape_fp8(int out_dim, int in_dim, const char* name) {
    printf("\n== [FP8] %s : out=%d in=%d ==\n", name, out_dim, in_dim);
    const size_t WN = (size_t)out_dim * in_dim;
    std::vector<float> hWf(WN); std::vector<__nv_bfloat16> hW(WN), hx(in_dim); std::vector<float> hxf(in_dim);
    for (size_t i = 0; i < WN; i++)  { float v = frand() * 0.05f; hWf[i] = v; hW[i] = __float2bfloat16(v); }
    for (int i = 0; i < in_dim; i++) { float v = frand();         hxf[i] = v; hx[i] = __float2bfloat16(v); }

    __nv_bfloat16 *dW, *dx; float *dxf, *dyf;
    cudaMalloc(&dW, WN * 2); cudaMalloc(&dx, in_dim * 2); cudaMalloc(&dxf, in_dim * 4); cudaMalloc(&dyf, out_dim * 4);
    cudaMemcpy(dW, hW.data(), WN * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx.data(), in_dim * 2, cudaMemcpyHostToDevice);

    e4bfp4::Fp8Weight w;
    if (!e4bfp4::e4b_fp8_quantize(dW, out_dim, in_dim, &w)) { printf("  fp8 quantize FAILED\n"); return false; }
    cudaDeviceSynchronize();
    printf("  quantized: %.2f MB (%.2f bit/elem)\n", w.bytes / 1e6, 8.0 * w.bytes / (double)WN);

    e4bfp4::e4b_fp8_gemv_f32(dyf, w, dx, dxf, 0); cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("  GEMV error: %s\n", cudaGetErrorString(err)); return false; }
    std::vector<float> hy(out_dim); cudaMemcpy(hy.data(), dyf, out_dim * 4, cudaMemcpyDeviceToHost);

    std::vector<uint8_t> hq(WN); std::vector<float> hrs(out_dim);
    cudaMemcpy(hq.data(), w.q, WN, cudaMemcpyDeviceToHost);
    cudaMemcpy(hrs.data(), w.rs, out_dim * 4, cudaMemcpyDeviceToHost);

    double num_k = 0, den_k = 0, num_q = 0, den = 0;
    for (int o = 0; o < out_dim; o++) {
        double oracle = 0, full = 0;
        for (int k = 0; k < in_dim; k++) {
            oracle += (double)nvfp4::e4m3_decode(hq[(size_t)o * in_dim + k]) * hrs[o] * hxf[k];
            full   += (double)hWf[(size_t)o * in_dim + k] * hxf[k];
        }
        num_k += (hy[o] - oracle) * (hy[o] - oracle); den_k += oracle * oracle;
        num_q += (full - hy[o]) * (full - hy[o]);       den   += full * full;
    }
    double kern_l2 = sqrt(num_k / (den_k + 1e-12)), snr_db = 10.0 * log10(den / (num_q + 1e-12));
    bool kern_ok = kern_l2 < 1e-2;
    printf("  kernel vs oracle : rel_L2=%.2e  %s\n", kern_l2, kern_ok ? "OK" : "*** FAIL ***");
    printf("  FP8 quant quality: SNR=%.1f dB  (vs ~20 dB NVFP4 — the index/decision precision win)\n", snr_db);

    const int ITERS = 200; cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++) e4bfp4::e4b_fp8_gemv_f32(dyf, w, dx, dxf, 0);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    printf("  GEMV bandwidth   : %.1f GB/s\n", ((double)w.bytes - out_dim * 4) * ITERS / (ms / 1e3) / 1e9);

    e4bfp4::fp8_weight_free(&w);
    cudaFree(dW); cudaFree(dx); cudaFree(dxf); cudaFree(dyf);
    return kern_ok;
}

int main() {
    int dev = 0; cudaSetDevice(dev);
    bool ok = true;
    printf("──── NVFP4 (4.5-bit, content path) ────");
    ok &= run_shape(10240, 2560, "gate/up_proj (FFN in)");
    ok &= run_shape(2560, 10240, "down_proj (FFN out)");
    ok &= run_shape(2560, 2048, "o_proj (attn content)");
    printf("\n──── FP8 (8-bit, index/decision path) ────");
    ok &= run_shape_fp8(2048, 2560, "q_proj (attn index)");
    ok &= run_shape_fp8(512, 2560, "k_proj (attn index)");
    ok &= run_shape_fp8(262144, 2560, "lm_head (decision)");
    printf("\n%s\n", ok ? "PASS: NVFP4+FP8 hybrid E4B foundation validated" : "FAIL");
    return ok ? 0 : 1;
}
