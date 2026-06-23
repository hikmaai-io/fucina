// test_e4b_q4_0.cu — validate + benchmark the native Q4_0 dp4a decode GEMV.
//
// Per E4B weight shape:
//   1. KERNEL CORRECTNESS: device Q4_0 dp4a GEMV vs a host oracle that decodes the
//      SAME SoA nibbles/scales and dots with x. The only difference is the int8
//      activation quant (Q8_1), so rel_L2 ~ 1/127 ≈ a few e-3; a layout/dp4a bug is O(1).
//   2. QUANT QUALITY: GEMV vs full-precision FP32 reference over the original weights
//      → SNR (dB), the Q4_0 error budget (same QAT weights llama.cpp would read).
//   3. BANDWIDTH: GEMV GB/s (weight bytes / time), the decode-relevant metric.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "e4b_q4_0.cuh"

static uint32_t s_rng = 0x12345678u;
static float frand() { s_rng = s_rng * 1664525u + 1013904223u; return ((s_rng >> 8) / 16777216.0f) * 2.0f - 1.0f; }

// Decode one Q4_0 SoA row to f32 (mirrors the kernel's value = d*(nibble-8)).
static void q40_dequant_row_host(const uint8_t* qs_row, const float* d_row, int in_dim, float* out) {
    const int nb = in_dim / 32;
    for (int g = 0; g < nb; g++) {
        const uint8_t* b = qs_row + (size_t)g * 16;
        float d = d_row[g];
        for (int j = 0; j < 16; j++) {
            out[g * 32 + j]      = d * (float)((b[j] & 0x0F) - 8);
            out[g * 32 + j + 16] = d * (float)((b[j] >> 4)   - 8);
        }
    }
}

static bool run_shape(int out_dim, int in_dim, const char* name) {
    printf("\n== %s : out=%d in=%d ==\n", name, out_dim, in_dim);
    const size_t WN = (size_t)out_dim * in_dim;

    std::vector<float>         hWf(WN);
    std::vector<__nv_bfloat16> hW(WN), hx(in_dim);
    std::vector<float>         hxf(in_dim);
    for (size_t i = 0; i < WN; i++)  { float v = frand() * 0.05f; hWf[i] = v; hW[i] = __float2bfloat16(v); }
    for (int i = 0; i < in_dim; i++) { float v = frand();         hxf[i] = v; hx[i] = __float2bfloat16(v); }

    __nv_bfloat16 *dW, *dx, *dy; float *dyf, *dda; int8_t* dqa; int32_t* dsa;
    cudaMalloc(&dW, WN * 2); cudaMalloc(&dx, in_dim * 2); cudaMalloc(&dy, out_dim * 2);
    cudaMalloc(&dyf, out_dim * 4); cudaMalloc(&dqa, in_dim); cudaMalloc(&dda, (in_dim / 32) * 4); cudaMalloc(&dsa, (in_dim / 32) * 4);
    cudaMemcpy(dW, hW.data(), WN * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx.data(), in_dim * 2, cudaMemcpyHostToDevice);

    e4bq40::Q40Weight w;
    if (!e4bq40::e4b_q4_0_from_bf16(dW, out_dim, in_dim, &w)) { printf("  quantize FAILED\n"); return false; }
    cudaDeviceSynchronize();
    printf("  quantized: %.2f MB (%.2f bit/elem vs 16.0 BF16)\n", w.bytes / 1e6, 8.0 * w.bytes / (double)WN);

    e4bq40::e4b_q4_0_gemv_bf16(dy, w, dx, dqa, dda, dsa, dyf, 0);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("  GEMV launch error: %s\n", cudaGetErrorString(err)); return false; }
    std::vector<float> hy(out_dim);
    cudaMemcpy(hy.data(), dyf, out_dim * 4, cudaMemcpyDeviceToHost);

    // Pull SoA back for the host oracle.
    std::vector<uint8_t> hqs((size_t)out_dim * (in_dim / 2));
    std::vector<float>   hd((size_t)out_dim * (in_dim / 32));
    cudaMemcpy(hqs.data(), w.qs, hqs.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(hd.data(),  w.d,  hd.size() * 4, cudaMemcpyDeviceToHost);

    std::vector<float> row(in_dim);
    double num_k = 0, den_k = 0, num_q = 0, den = 0, max_abs = 0;
    const int nb = in_dim / 32;
    for (int o = 0; o < out_dim; o++) {
        q40_dequant_row_host(&hqs[(size_t)o * (in_dim / 2)], &hd[(size_t)o * nb], in_dim, row.data());
        double oracle = 0, full = 0;
        for (int k = 0; k < in_dim; k++) { oracle += (double)row[k] * hxf[k]; full += (double)hWf[(size_t)o * in_dim + k] * hxf[k]; }
        num_k += (hy[o] - oracle) * (hy[o] - oracle); den_k += oracle * oracle;
        max_abs = fmax(max_abs, fabs(hy[o] - oracle));
        num_q += (full - hy[o]) * (full - hy[o]);     den   += full * full;
    }
    double kern_l2 = sqrt(num_k / (den_k + 1e-12));
    double rel_l2  = sqrt(num_q / (den + 1e-12));
    double snr_db  = 10.0 * log10(den / (num_q + 1e-12));
    bool kern_ok = kern_l2 < 2e-2;   // int8 activation quant floor (~1/127)
    printf("  kernel vs oracle : rel_L2=%.2e  max_abs=%.2e  %s\n", kern_l2, max_abs, kern_ok ? "OK" : "*** FAIL ***");
    printf("  Q4_0 quant quality: rel_L2=%.4f  SNR=%.1f dB\n", rel_l2, snr_db);

    const int ITERS = 200;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    e4bq40::e4b_q4_0_gemv_f32(dyf, w, dx, dqa, dda, dsa, 0); cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int i = 0; i < ITERS; i++) e4bq40::e4b_q4_0_gemv_f32(dyf, w, dx, dqa, dda, dsa, 0);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    double gbps = (double)w.bytes * ITERS / (ms / 1e3) / 1e9;
    printf("  GEMV bandwidth   : %.1f GB/s  (%.3f ms/call)\n", gbps, ms / ITERS);

    e4bq40::q40_free(&w);
    cudaFree(dW); cudaFree(dx); cudaFree(dy); cudaFree(dyf); cudaFree(dqa); cudaFree(dda); cudaFree(dsa);
    return kern_ok;
}

int main() {
    cudaSetDevice(0);
    bool ok = true;
    printf("──── native Q4_0 (4.5-bit) dp4a decode GEMV ────");
    ok &= run_shape(10240, 2560, "gate/up_proj (FFN in)");
    ok &= run_shape(2560, 10240, "down_proj (FFN out)");
    ok &= run_shape(2048, 2560, "q_proj (attn)");
    ok &= run_shape(512,  2560, "k/v_proj (attn)");
    ok &= run_shape(2560, 2048, "o_proj (attn out)");
    printf("\n%s\n", ok ? "PASS: native Q4_0 dp4a GEMV validated" : "FAIL");
    return ok ? 0 : 1;
}
