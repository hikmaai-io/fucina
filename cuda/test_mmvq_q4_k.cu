// test_mmvq_q4_k.cu — standalone numeric validation of the native Q4_K decode GEMV
// (mmvq_q4_k_kernel in mmvq.cuh). Generates random Q4_K weight superblocks, dequantizes
// them on the host for a full-precision reference dot against random activations, then
// runs the device kernel on q8_1-quantized activations and asserts cosine >= 0.999
// (the only error source is the q8_1 activation quantization, exactly as in decode).
//
// Build: nvcc -arch=sm_121a -O3 cuda/test_mmvq_q4_k.cu -o /tmp/test_mmvq_q4_k && run.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "mmvq.cuh"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

// Host mirror of the device sub-block scale/min unpack.
static void h_scale_min(int j, const uint8_t *q, int *d, int *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)   | ((q[j    ] >> 6) << 4);
    }
}

static float frand(float lo, float hi) { return lo + (hi - lo) * (rand() / (float)RAND_MAX); }

int main() {
    srand(1234);
    const int IN = 1024;        // 4 superblocks
    const int OUT = 96;
    const int NSUP = IN / 256;
    const int NB32 = IN / 32;
    const int BLK = 144;        // Q4_K bytes

    // ---- random Q4_K weights (row-major: OUT rows, NSUP blocks of 144 bytes) ----
    std::vector<uint8_t> w((size_t)OUT * NSUP * BLK);
    for (size_t i = 0; i < w.size(); i++) w[i] = (uint8_t)(rand() & 0xFF);
    // Patch each block's d/dmin to small positive halves so dequant is well-scaled.
    for (int o = 0; o < OUT; o++)
        for (int s = 0; s < NSUP; s++) {
            uint8_t *blk = &w[((size_t)o * NSUP + s) * BLK];
            __half d  = __float2half(frand(0.005f, 0.05f));
            __half dm = __float2half(frand(0.0f, 0.03f));
            uint16_t dr = *(uint16_t*)&d, mr = *(uint16_t*)&dm;
            blk[0] = dr & 0xFF; blk[1] = dr >> 8;
            blk[2] = mr & 0xFF; blk[3] = mr >> 8;
        }

    // ---- host dequant -> full-precision W[OUT][IN] ----
    std::vector<float> Wf((size_t)OUT * IN);
    for (int o = 0; o < OUT; o++)
        for (int s = 0; s < NSUP; s++) {
            const uint8_t *blk = &w[((size_t)o * NSUP + s) * BLK];
            __half_raw dr; dr.x = (uint16_t)(blk[0] | (blk[1] << 8));
            __half_raw mr; mr.x = (uint16_t)(blk[2] | (blk[3] << 8));
            float d = __half2float(__half(dr)), dmin = __half2float(__half(mr));
            const uint8_t *sc12 = blk + 4, *qs = blk + 16;
            for (int j = 0; j < 8; j++) {
                int sc, mn; h_scale_min(j, sc12, &sc, &mn);
                const uint8_t *qb = qs + 32 * (j >> 1);
                for (int l = 0; l < 32; l++) {
                    int nib = (j & 1) ? (qb[l] >> 4) : (qb[l] & 0x0F);
                    Wf[(size_t)o * IN + s * 256 + j * 32 + l] = d * sc * nib - dmin * mn;
                }
            }
        }

    // ---- random activations + q8_1 quantization (mirrors quantize_q8_1_kernel) ----
    std::vector<float> x(IN);
    for (int i = 0; i < IN; i++) x[i] = frand(-2.0f, 2.0f);
    std::vector<int8_t> qx(IN);
    std::vector<float> dxh(NB32);
    std::vector<int> sxh(NB32);
    for (int b = 0; b < NB32; b++) {
        float a = 0.0f;
        for (int l = 0; l < 32; l++) a = fmaxf(a, fabsf(x[b * 32 + l]));
        float d = a / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
        int sum = 0;
        for (int l = 0; l < 32; l++) {
            int q = (int)lrintf(x[b * 32 + l] * id);
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            qx[b * 32 + l] = (int8_t)q; sum += q;
        }
        dxh[b] = d; sxh[b] = sum;
    }

    // ---- reference out[o] = Σ Wf[o][i]·x[i] ----
    std::vector<float> ref(OUT, 0.0f);
    for (int o = 0; o < OUT; o++) {
        double s = 0;
        for (int i = 0; i < IN; i++) s += (double)Wf[(size_t)o * IN + i] * x[i];
        ref[o] = (float)s;
    }

    // ---- device kernel ----
    uint8_t *d_w; int8_t *d_qx; float *d_dx, *d_out; int *d_sx;
    CK(cudaMalloc(&d_w, w.size()));
    CK(cudaMalloc(&d_qx, IN));
    CK(cudaMalloc(&d_dx, NB32 * sizeof(float)));
    CK(cudaMalloc(&d_sx, NB32 * sizeof(int)));
    CK(cudaMalloc(&d_out, OUT * sizeof(float)));
    CK(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_qx, qx.data(), IN, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dx, dxh.data(), NB32 * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_sx, sxh.data(), NB32 * sizeof(int), cudaMemcpyHostToDevice));
    mmvq_q4_k_launch(d_out, d_w, d_qx, d_dx, d_sx, IN, OUT, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> out(OUT);
    CK(cudaMemcpy(out.data(), d_out, OUT * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- compare (cosine + max rel error) ----
    double dot = 0, na = 0, nb = 0, maxrel = 0;
    for (int o = 0; o < OUT; o++) {
        dot += (double)out[o] * ref[o];
        na  += (double)out[o] * out[o];
        nb  += (double)ref[o] * ref[o];
        float denom = fmaxf(fabsf(ref[o]), 1e-3f);
        maxrel = fmax(maxrel, fabs(out[o] - ref[o]) / denom);
    }
    double cosine = dot / (sqrt(na) * sqrt(nb) + 1e-12);
    printf("mmvq_q4_K native: cosine=%.6f  max_rel_err=%.4f  (sample ref=%.4f out=%.4f)\n",
           cosine, maxrel, ref[0], out[0]);
    int ok = cosine >= 0.999;

    // ---- packed path: repack same weights, run packed kernel, assert BIT-IDENTICAL ----
    uint8_t *d_wp; CK(cudaMalloc(&d_wp, w.size()));
    int n_super_total = OUT * (IN / 256);
    repack_q4_k_kernel<<<(n_super_total + 255) / 256, 256>>>(d_w, d_wp, n_super_total);
    float *d_outp; CK(cudaMalloc(&d_outp, OUT * sizeof(float)));
    mmvq_q4_k_packed_launch(d_outp, d_wp, d_qx, d_dx, d_sx, IN, OUT, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> outp(OUT);
    CK(cudaMemcpy(outp.data(), d_outp, OUT * sizeof(float), cudaMemcpyDeviceToHost));
    int bitident = (memcmp(outp.data(), out.data(), OUT * sizeof(float)) == 0);
    printf("mmvq_q4_K packed vs native: %s\n", bitident ? "BIT-IDENTICAL" : "DIFFER");
    if (!bitident) {
        for (int o = 0; o < OUT && o < 5; o++)
            printf("  o=%d native=%.6f packed=%.6f\n", o, out[o], outp[o]);
    }
    ok = ok && bitident;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
