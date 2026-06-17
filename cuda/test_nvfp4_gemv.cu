// Numerical + bandwidth test for the NVFP4 decode GEMV kernel (nvfp4_gemv.cuh).
//
// Quantizes a random weight matrix to the ModelOpt NVFP4 layout on the host (E2M1 packed,
// linear E4M3 block scales, FP32 global = amax/(6*448)), runs the kernel, and compares against
// the EXACT reference: a float dot product of the SAME dequantized NVFP4 values (nvfp4.h
// oracle) with x. Agreement must be ~fp-accumulation epsilon (the kernel and the reference
// multiply identical reconstructed values), which validates packing, scale indexing, the
// uint4 unpack, and the global-scale fold. Then reports effective decode bandwidth.
//
// build: nvcc -arch=sm_121a -O3 -std=c++17 cuda/test_nvfp4_gemv.cu -o /tmp/nvfp4_gemv && /tmp/nvfp4_gemv
#include "nvfp4_gemv.cuh"
#include "nvfp4.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// ── host NVFP4 quantizer (ModelOpt convention) ──
// nearest-representable E4M3 byte for a non-negative target (brute force over the 256 codes;
// skips the NaN codes 0x7F/0xFF). Exact nearest — fine for a test.
static uint8_t e4m3_encode_nearest(float target) {
    uint8_t best = 0; float bestd = 1e30f;
    for (int b = 0; b < 256; b++) {
        if (b == 0x7F || b == 0xFF) continue;           // NaN
        if (b & 0x80) continue;                          // scales are non-negative → sign 0
        float v = nvfp4::e4m3_decode((uint8_t)b);
        if (std::isnan(v)) continue;
        float d = std::fabs(v - target);
        if (d < bestd) { bestd = d; best = (uint8_t)b; }
    }
    return best;
}
// nearest E2M1 nibble (signed) for a value
static uint32_t e2m1_encode_nearest(float v) {
    const float mag[8] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    float a = std::fabs(v); int bi = 0; float bd = 1e30f;
    for (int i = 0; i < 8; i++) { float d = std::fabs(mag[i]-a); if (d < bd){bd=d;bi=i;} }
    uint32_t nib = (uint32_t)bi;
    if (v < 0 && bi != 0) nib |= 8u;
    return nib;
}

int main(int argc, char** argv) {
    int out_dim = argc>1?atoi(argv[1]):3840;
    int in_dim  = argc>2?atoi(argv[2]):3840;
    srand(7);
    printf("NVFP4 decode GEMV  out=%d in=%d  (%.1f MB weights)\n",
           out_dim, in_dim, (out_dim*(in_dim/2.0) + out_dim*(in_dim/16.0))/1e6);

    auto randn=[&](){ float u1=(rand()+1.0f)/(RAND_MAX+2.0f),u2=(rand()+1.0f)/(RAND_MAX+2.0f);
                      return sqrtf(-2*logf(u1))*cosf(6.2831853f*u2); };

    std::vector<float> W((size_t)out_dim*in_dim), x(in_dim);
    for (auto& v : W) v = 0.05f * randn();      // weight-like
    for (auto& v : x) v = randn();

    // global scale = amax/(6*448)
    float amax=0.f; for (float v: W) amax=fmaxf(amax,fabsf(v));
    float gs = amax/(6.0f*448.0f); if (gs<=0) gs=1e-30f;

    // quantize row-major: packed [out][in/2], linear E4M3 [out][in/16]
    std::vector<uint8_t> wpacked((size_t)out_dim*(in_dim/2)), wscale((size_t)out_dim*(in_dim/16));
    std::vector<float> deq((size_t)out_dim*in_dim);   // reconstructed NVFP4 values (oracle)
    for (int r=0; r<out_dim; r++){
        const float* wr = W.data() + (size_t)r*in_dim;
        uint8_t* pr = wpacked.data() + (size_t)r*(in_dim/2);
        uint8_t* sr = wscale.data()  + (size_t)r*(in_dim/16);
        float* dr = deq.data() + (size_t)r*in_dim;
        for (int blk=0; blk<in_dim/16; blk++){
            float bamax=0.f; for(int i=0;i<16;i++) bamax=fmaxf(bamax,fabsf(wr[blk*16+i]));
            float target = bamax/6.0f/gs;            // block_scale before E4M3 rounding
            uint8_t sb = e4m3_encode_nearest(target);
            sr[blk] = sb;
            float divisor = nvfp4::e4m3_decode(sb)*gs; if (divisor<=0) divisor=1e30f;
            for (int i=0;i<16;i++){
                int k = blk*16+i;
                uint32_t nib = e2m1_encode_nearest(wr[k]/divisor);
                // pack: low nibble = even k, high = odd k
                if ((k&1)==0) pr[k>>1] = (uint8_t)((pr[k>>1] & 0xF0) | (nib & 0x0F));
                else          pr[k>>1] = (uint8_t)((pr[k>>1] & 0x0F) | ((nib & 0x0F)<<4));
                dr[k] = nvfp4::reconstruct(nib, nvfp4::e4m3_decode(sb), gs);  // oracle value
            }
        }
    }

    // reference: y_ref[r] = Σ_k deq[r,k]·x[k]   (exact float dot of the SAME values)
    std::vector<float> yref(out_dim);
    for (int r=0;r<out_dim;r++){ double a=0; const float* dr=deq.data()+(size_t)r*in_dim;
        for(int k=0;k<in_dim;k++) a += (double)dr[k]*x[k]; yref[r]=(float)a; }

    // ── device run ──
    uint8_t *dW,*dS; float *dx,*dy,*dgs;
    CK(cudaMalloc(&dW, wpacked.size())); CK(cudaMalloc(&dS, wscale.size()));
    CK(cudaMalloc(&dx, in_dim*4)); CK(cudaMalloc(&dy, out_dim*4)); CK(cudaMalloc(&dgs, 4));
    CK(cudaMemcpy(dW,wpacked.data(),wpacked.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dS,wscale.data(),wscale.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx,x.data(),in_dim*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dgs,&gs,4,cudaMemcpyHostToDevice));

    nvfp4_gemv_launch(dy,dW,dS,dgs,dx,in_dim,out_dim,0);
    CK(cudaDeviceSynchronize());
    std::vector<float> y(out_dim); CK(cudaMemcpy(y.data(),dy,out_dim*4,cudaMemcpyDeviceToHost));

    double se=0,sr=0,mx=0; for(int r=0;r<out_dim;r++){ double e=(double)y[r]-yref[r];
        se+=e*e; sr+=(double)yref[r]*yref[r]; mx=fmax(mx,fabs(e)); }
    double l2rel = sqrt(se/(sr+1e-30));
    printf("kernel vs oracle:  L2rel=%.6f%%  max|d|=%.6g  %s\n",
           100*l2rel, mx, l2rel<1e-4 ? "OK" : "*** MISMATCH ***");

    // ── bandwidth ──
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    for(int i=0;i<5;i++) nvfp4_gemv_launch(dy,dW,dS,dgs,dx,in_dim,out_dim,0);
    CK(cudaDeviceSynchronize());
    int R=200; cudaEventRecord(s);
    for(int i=0;i<R;i++) nvfp4_gemv_launch(dy,dW,dS,dgs,dx,in_dim,out_dim,0);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms; cudaEventElapsedTime(&ms,s,e); ms/=R;
    double bytes = (double)out_dim*(in_dim/2) + (double)out_dim*(in_dim/16) + in_dim*4.0;
    printf("  %.4f ms/gemv   %.1f GB/s (weight+scale+act read)\n", ms, bytes/(ms/1e3)/1e9);

    // ── BATCHED spec-verify GEMV: y[K][out] = X[K][in]·W, weight read ONCE for K rows ──
    // Correctness vs K× single-token (per row), then profitability vs K× single-token.
    int K = argc>3?atoi(argv[3]):5;
    bool batched_ok = true; double worst_l2 = 0; float speedup = 0, bw_batched = 0;
    {
        // K random activation rows (token-major X[K][in])
        std::vector<float> Xk((size_t)K*in_dim);
        for (auto& v : Xk) v = randn();

        // device buffers
        float *dXk, *dXt, *dYk;
        CK(cudaMalloc(&dXk, (size_t)K*in_dim*4));
        CK(cudaMalloc(&dXt, (size_t)K*in_dim*4));
        CK(cudaMalloc(&dYk, (size_t)K*out_dim*4));
        CK(cudaMemcpy(dXk, Xk.data(), (size_t)K*in_dim*4, cudaMemcpyHostToDevice));

        // run: transpose then batched
        nvfp4_xT_launch(dXt, dXk, in_dim, K, 0);
        nvfp4_gemv_batched_launch(dYk, dW, dS, dgs, dXt, in_dim, out_dim, K, 0);
        CK(cudaDeviceSynchronize());
        std::vector<float> Yk((size_t)K*out_dim);
        CK(cudaMemcpy(Yk.data(), dYk, (size_t)K*out_dim*4, cudaMemcpyDeviceToHost));

        // reference: the SAME float oracle the single-token test passes against (deq·x in double),
        // computed per row. This is the meaningful correctness bar — both kernels differ from it
        // only by FMA-accumulation rounding (the single-token kernel itself is ~1e-7 off it).
        // Also track the kernel-vs-single-token divergence for transparency (pure fp reordering).
        std::vector<float> Yst((size_t)K*out_dim);
        for (int r=0; r<K; r++) {
            CK(cudaMemcpy(dx, Xk.data()+(size_t)r*in_dim, in_dim*4, cudaMemcpyHostToDevice));
            nvfp4_gemv_launch(dy, dW, dS, dgs, dx, in_dim, out_dim, 0);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(Yst.data()+(size_t)r*out_dim, dy, out_dim*4, cudaMemcpyDeviceToHost));
        }
        double worst_vs_st = 0;
        for (int r=0; r<K; r++) {
            const float* xr = Xk.data()+(size_t)r*in_dim;
            const float* yr = Yk.data()+(size_t)r*out_dim;
            const float* sr = Yst.data()+(size_t)r*out_dim;
            double se=0,srr=0, se2=0,srr2=0;
            for (int o=0;o<out_dim;o++){
                double oref=0; const float* dr=deq.data()+(size_t)o*in_dim;
                for (int kk=0;kk<in_dim;kk++) oref += (double)dr[kk]*xr[kk];
                double d=(double)yr[o]-oref; se+=d*d; srr+=oref*oref;          // vs float oracle
                double d2=(double)yr[o]-sr[o]; se2+=d2*d2; srr2+=(double)sr[o]*sr[o]; // vs single-tok
            }
            worst_l2    = fmax(worst_l2,    sqrt(se/(srr+1e-30)));
            worst_vs_st = fmax(worst_vs_st, sqrt(se2/(srr2+1e-30)));
        }
        // Bar: same 1e-4% fp32 dot-product floor the single-token oracle test uses (these dims
        // accumulate 3840–15360 terms; 1e-5%/1e-7 is below the fp32 accumulation floor — indeed the
        // batched kernel is CLOSER to the oracle than the single-token kernel is). The kernel
        // reconstructs bit-identical NVFP4 values; only FMA-accumulation rounding differs.
        batched_ok = (worst_l2 < 1e-6);
        printf("batched(K=%d) vs float oracle:  worst L2rel=%.8f%%  (vs single-token kernel: %.8f%%)  %s\n",
               K, 100*worst_l2, 100*worst_vs_st, batched_ok ? "OK" : "*** MISMATCH ***");

        // profitability: time batched (xT + gemv) vs K× single-token
        for(int i=0;i<5;i++){ nvfp4_xT_launch(dXt,dXk,in_dim,K,0);
            nvfp4_gemv_batched_launch(dYk,dW,dS,dgs,dXt,in_dim,out_dim,K,0); }
        CK(cudaDeviceSynchronize());
        cudaEventRecord(s);
        for(int i=0;i<R;i++){ nvfp4_xT_launch(dXt,dXk,in_dim,K,0);
            nvfp4_gemv_batched_launch(dYk,dW,dS,dgs,dXt,in_dim,out_dim,K,0); }
        cudaEventRecord(e); cudaEventSynchronize(e);
        float msb; cudaEventElapsedTime(&msb,s,e); msb/=R;

        cudaEventRecord(s);
        for(int i=0;i<R;i++) for(int r=0;r<K;r++)
            nvfp4_gemv_launch(dy,dW,dS,dgs,dx,in_dim,out_dim,0);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float mss; cudaEventElapsedTime(&mss,s,e); mss/=R;

        speedup = mss/msb;
        bw_batched = (float)(bytes/(msb/1e3)/1e9);   // same weight bytes, read once for K
        printf("  batched %.4f ms | %d× single %.4f ms | speedup %.2fx | %.1f GB/s (weight read once)\n",
               msb, K, mss, speedup, bw_batched);
        printf("  %s (bar: correct vs oracle AND >=1.5x)\n",
               (batched_ok && speedup>=1.5f) ? "*** PROFITABLE ***" : "--- below bar ---");

        cudaFree(dXk); cudaFree(dXt); cudaFree(dYk);
    }

    // ── BATCHED BF16 LM-HEAD GEMV: y[K][vocab] = X[K][hidden]·Head, 2 GB head read ONCE for K ──
    // Only run on the "head-like" shape (out >= 4*in, tall) to keep the bench focused; reuse the
    // current in_dim as hidden and a vocab-sized out_dim.
    bool head_ok = true; float head_speedup = 0;
    {
        int hidden = in_dim;
        int vocab  = 16384;             // representative tall head slice (full vocab is ~262k)
        std::vector<__nv_bfloat16> Wh((size_t)vocab*hidden);
        std::vector<float> Whf((size_t)vocab*hidden);
        // oracle uses the SAME bf16-rounded values the kernel reads (else we'd measure bf16
        // quantization error, ~0.17%, not kernel correctness).
        for (size_t i=0;i<Wh.size();i++){ __nv_bfloat16 b=__float2bfloat16(0.05f*randn());
            Wh[i]=b; Whf[i]=__bfloat162float(b); }
        std::vector<float> Xk((size_t)K*hidden);
        for (auto& v : Xk) v = randn();

        __nv_bfloat16 *dWh; float *dXk,*dXt,*dYk,*dy1;
        CK(cudaMalloc(&dWh, Wh.size()*sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&dXk, (size_t)K*hidden*4)); CK(cudaMalloc(&dXt, (size_t)K*hidden*4));
        CK(cudaMalloc(&dYk, (size_t)K*vocab*4));  CK(cudaMalloc(&dy1, (size_t)vocab*4));
        CK(cudaMemcpy(dWh, Wh.data(), Wh.size()*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dXk, Xk.data(), (size_t)K*hidden*4, cudaMemcpyHostToDevice));

        nvfp4_xT_launch(dXt, dXk, hidden, K, 0);
        bf16_head_gemv_batched_launch(dYk, dWh, dXt, hidden, vocab, K, 0);
        CK(cudaDeviceSynchronize());
        std::vector<float> Yk((size_t)K*vocab); CK(cudaMemcpy(Yk.data(),dYk,(size_t)K*vocab*4,cudaMemcpyDeviceToHost));

        // correctness vs double-precision oracle (Whf·Xk) per row
        double worst=0;
        for (int r=0;r<K;r++){ const float* xr=Xk.data()+(size_t)r*hidden; const float* yr=Yk.data()+(size_t)r*vocab;
            double se=0,srr=0;
            for (int o=0;o<vocab;o++){ double oref=0; const float* wr=Whf.data()+(size_t)o*hidden;
                for(int kk=0;kk<hidden;kk++) oref+=(double)wr[kk]*xr[kk];
                double d=(double)yr[o]-oref; se+=d*d; srr+=oref*oref; }
            worst=fmax(worst,sqrt(se/(srr+1e-30))); }
        head_ok = (worst < 1e-6);
        printf("bf16-head batched(K=%d) vocab=%d hidden=%d vs oracle:  worst L2rel=%.8f%%  %s\n",
               K, vocab, hidden, 100*worst, head_ok?"OK":"*** MISMATCH ***");

        for(int i=0;i<5;i++){ nvfp4_xT_launch(dXt,dXk,hidden,K,0);
            bf16_head_gemv_batched_launch(dYk,dWh,dXt,hidden,vocab,K,0); }
        CK(cudaDeviceSynchronize());
        cudaEventRecord(s);
        for(int i=0;i<R;i++){ nvfp4_xT_launch(dXt,dXk,hidden,K,0);
            bf16_head_gemv_batched_launch(dYk,dWh,dXt,hidden,vocab,K,0); }
        cudaEventRecord(e); cudaEventSynchronize(e);
        float msb; cudaEventElapsedTime(&msb,s,e); msb/=R;

        cudaEventRecord(s);
        for(int i=0;i<R;i++) for(int r=0;r<K;r++)
            bf16_head_gemv1_launch(dy1,dWh,dXk+(size_t)r*hidden,hidden,vocab,0);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float mss; cudaEventElapsedTime(&mss,s,e); mss/=R;
        head_speedup = mss/msb;
        double hbytes=(double)vocab*hidden*2.0;
        printf("  batched %.4f ms | %d× single %.4f ms | speedup %.2fx | %.1f GB/s (head read once)  %s\n",
               msb, K, mss, head_speedup, hbytes/(msb/1e3)/1e9,
               (head_ok && head_speedup>=1.5f)?"*** PROFITABLE ***":"--- below bar ---");

        cudaFree(dWh); cudaFree(dXk); cudaFree(dXt); cudaFree(dYk); cudaFree(dy1);
    }

    return (l2rel<1e-4 && batched_ok && speedup>=1.5f && head_ok && head_speedup>=1.5f) ? 0 : 1;
}
