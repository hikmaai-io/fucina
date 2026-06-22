// Unit test for the native Q4_K LM-head matvec (mmvq_q4_k_kernel in gemma4_kernels.cu).
// Validates the Q4_K super-block decode (fp16 d/dmin, 6-bit packed scale+min, 4-bit quants,
// asymmetric value = d·s·q − dmin·m) + the int8-dp4a GEMV math against a double-precision host
// reference that decodes the SAME bytes and uses the SAME int8-quantized activation. Build:
//   nvcc -O3 -arch=sm_121a -std=c++17 cuda/test_q4k_gemv.cu -o /tmp/q4k && /tmp/q4k
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

// ── device kernel under test (copied verbatim from gemma4_kernels.cu) ──────────────────────
static __device__ __forceinline__ int q8_get_int_b2(const void *p, int i32) {
    const uint16_t *x16 = (const uint16_t *)p;
    return (int)x16[2*i32] | ((int)x16[2*i32 + 1] << 16);
}
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    return v;
}
__device__ __forceinline__ void q4k_scale_min(const uint8_t *sc, int j, int *s, int *m) {
    if (j < 4) { *s = sc[j] & 63; *m = sc[j + 4] & 63; }
    else { *s = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
           *m = (sc[j + 4] >> 4)   | ((sc[j]     >> 6) << 4); }
}
__global__ void mmvq_q4_k_kernel(
    float *out, const uint8_t *weight, const int8_t *qx, const float *dx, const int *sx,
    int in_dim, int out_dim)
{
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int idx = blockIdx.x * nwarps + warp;
    if (idx >= out_dim) return;
    int n_super = in_dim >> 8, nb32 = in_dim >> 5;
    const uint8_t *wrow = weight + (size_t)idx * (size_t)n_super * 144;
    float acc = 0.0f;
    for (int b = lane; b < nb32; b += 32) {
        const uint8_t *blk = wrow + (size_t)(b >> 3) * 144;
        int j = b & 7;
        __half_raw hd; hd.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
        __half_raw hm; hm.x = (uint16_t)(blk[2] | ((uint16_t)blk[3] << 8));
        float d = __half2float(__half(hd)), dmin = __half2float(__half(hm));
        int s, m; q4k_scale_min(blk + 4, j, &s, &m);
        const uint8_t *qbase = blk + 16 + (size_t)(j >> 1) * 32;
        int shift = (j & 1) ? 4 : 0;
        const int *xqs = (const int *)(qx + (size_t)b * 32);
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int qw  = q8_get_int_b2(qbase, k);
            int nib = (shift ? (qw >> 4) : qw) & 0x0F0F0F0F;
            sumi = __dp4a(nib, xqs[k], sumi);
        }
        acc += dx[b] * (d * (float)s * (float)sumi - dmin * (float)m * (float)sx[b]);
    }
    acc = warp_sum(acc);
    if (lane == 0) out[idx] = acc;
}

// ── host reference ───────────────────────────────────────────────────────────────────────
static float h2f(uint16_t h) { __half_raw r; r.x = h; return __half2float(*(__half*)&r); }
static uint16_t f2h(float f) { __half hh = __float2half(f); __half_raw r = *(__half_raw*)&hh; return r.x; }

static void host_q4k_scale_min(const uint8_t *sc, int j, int *s, int *m) {
    if (j < 4) { *s = sc[j] & 63; *m = sc[j + 4] & 63; }
    else { *s = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
           *m = (sc[j + 4] >> 4)   | ((sc[j]     >> 6) << 4); }
}
// Decode one 144-B Q4_K super-block → 256 fp32 (matches dequant_q4_k_superblock in the engine).
static void host_dequant_q4k(const uint8_t *blk, float out[256]) {
    float d = h2f((uint16_t)(blk[0] | (blk[1] << 8)));
    float dmin = h2f((uint16_t)(blk[2] | (blk[3] << 8)));
    const uint8_t *sc = blk + 4, *qs = blk + 16;
    for (int j = 0; j < 8; j++) {
        int s, m; host_q4k_scale_min(sc, j, &s, &m);
        float dl = d * (float)s, ml = dmin * (float)m;
        const uint8_t *q = qs + (j / 2) * 32;
        int shift = (j & 1) ? 4 : 0;
        for (int i = 0; i < 32; i++) out[j*32 + i] = dl * (float)((q[i] >> shift) & 0x0F) - ml;
    }
}

static bool run_shape(int in_dim, int out_dim, unsigned seed) {
    std::mt19937 rng(seed);
    int n_super = in_dim / 256, nb32 = in_dim / 32;
    std::vector<uint8_t> wq((size_t)out_dim * n_super * 144);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> dn(0.0f, 0.05f), xn(0.0f, 1.0f);
    // random valid Q4_K bytes: small fp16 d/dmin, random 6-bit scale/min pack, random nibbles
    for (size_t blk = 0; blk < wq.size() / 144; blk++) {
        uint8_t *bp = &wq[blk * 144];
        uint16_t dh = f2h(fabsf(dn(rng)) + 1e-3f), mh = f2h(fabsf(dn(rng)));
        bp[0]=dh&0xFF; bp[1]=dh>>8; bp[2]=mh&0xFF; bp[3]=mh>>8;
        for (int i = 4; i < 144; i++) bp[i] = (uint8_t)byte(rng);   // scales/mins + quants
    }
    std::vector<float> x(in_dim);
    for (int k = 0; k < in_dim; k++) x[k] = xn(rng);

    // int8-quantize the activation per 32-block (engine-style: scale = absmax/127, sx = Σqx).
    std::vector<int8_t> qx(in_dim); std::vector<float> dx(nb32); std::vector<int> sx(nb32);
    for (int b = 0; b < nb32; b++) {
        float amax = 0.f; for (int i = 0; i < 32; i++) amax = std::max(amax, fabsf(x[b*32+i]));
        float scale = (amax > 0) ? amax / 127.0f : 1.0f, inv = 1.0f / scale;
        int ssum = 0;
        for (int i = 0; i < 32; i++) {
            int q = (int)lrintf(x[b*32+i] * inv); q = q < -127 ? -127 : (q > 127 ? 127 : q);
            qx[b*32+i] = (int8_t)q; ssum += q;
        }
        dx[b] = scale; sx[b] = ssum;
    }
    // double-precision reference: y[o] = Σ_k w_dequant[o][k] * (qx[k]*dx[blk(k)])
    std::vector<double> ref(out_dim, 0.0);
    std::vector<float> wdec(256);
    for (int o = 0; o < out_dim; o++) {
        for (int sblk = 0; sblk < n_super; sblk++) {
            host_dequant_q4k(&wq[((size_t)o*n_super + sblk)*144], wdec.data());
            for (int i = 0; i < 256; i++) {
                int k = sblk*256 + i, b = k / 32;
                ref[o] += (double)wdec[i] * ((double)qx[k] * (double)dx[b]);
            }
        }
    }

    uint8_t *d_w; int8_t *d_qx; float *d_dx, *d_out; int *d_sx;
    cudaMalloc(&d_w, wq.size()); cudaMalloc(&d_qx, in_dim); cudaMalloc(&d_dx, nb32*sizeof(float));
    cudaMalloc(&d_sx, nb32*sizeof(int)); cudaMalloc(&d_out, out_dim*sizeof(float));
    cudaMemcpy(d_w, wq.data(), wq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qx, qx.data(), in_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dx, dx.data(), nb32*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sx, sx.data(), nb32*sizeof(int), cudaMemcpyHostToDevice);
    const int NW = 8; mmvq_q4_k_kernel<<<(out_dim+NW-1)/NW, NW*32>>>(d_out, d_w, d_qx, d_dx, d_sx, in_dim, out_dim);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA err: %s\n", cudaGetErrorString(cudaGetLastError())); return false; }
    std::vector<float> out(out_dim);
    cudaMemcpy(out.data(), d_out, out_dim*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_w); cudaFree(d_qx); cudaFree(d_dx); cudaFree(d_sx); cudaFree(d_out);

    double num = 0, den = 0;
    for (int o = 0; o < out_dim; o++) { double e = out[o]-ref[o]; num += e*e; den += ref[o]*ref[o]; }
    double rel = sqrt(num / (den + 1e-30));
    bool ok = rel < 2e-3;
    printf("  [%-5s] in=%-6d out=%-6d  rel_L2=%.2e\n", ok?"PASS":"FAIL", in_dim, out_dim, rel);
    return ok;
}

int main() {
    printf("Q4_K head matvec unit test (kernel vs host reference decode)\n");
    bool ok = true;
    ok &= run_shape(5376, 4096, 1);
    ok &= run_shape(5376, 262144, 2);   // 31B LM head: hidden 5376 → vocab 262144
    printf("%s\n", ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
