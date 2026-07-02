// test_fp8_block.cu — validate the FP8 block-scaled decode GEMV (fp8_block.cuh) vs a host
// full-precision dequant+dot reference. Mirrors Qwen3.5-MoE FP8 shapes (in 2048, out 8192).
// PASS at cosine >= 0.999 (FP8 E4M3 weight rounding is the only error; activation is float).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include "fp8_block.cuh"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

static float frand(float lo, float hi){ return lo + (hi-lo)*(rand()/(float)RAND_MAX); }

int main() {
    srand(7);
    const int IN = 2048, OUT = 8192;          // Qwen3.5-MoE q_proj-ish shape
    const int SB = IN/128, OB = OUT/128;       // scale blocks

    // random FP8 weights + per-128x128-block BF16 scales
    std::vector<uint8_t> w((size_t)OUT*IN);
    for (auto &b : w) {
        __nv_fp8_e4m3 v; v = __nv_fp8_e4m3(frand(-0.4f, 0.4f)); b = v.__x;  // realistic small weights
    }
    std::vector<__nv_bfloat16> wscale((size_t)OB*SB);
    std::vector<float> wscale_f((size_t)OB*SB);
    for (size_t i=0;i<wscale.size();i++){ float s=frand(0.02f,0.08f); wscale[i]=__float2bfloat16(s); wscale_f[i]=__bfloat162float(wscale[i]); }

    std::vector<float> x(IN);
    for (int i=0;i<IN;i++) x[i]=frand(-2.0f,2.0f);

    // host reference: out[o] = Σ_i fp8(W[o][i])·scale[o/128][i/128]·x[i]
    std::vector<float> ref(OUT,0.0f);
    for (int o=0;o<OUT;o++){
        double s=0;
        for (int i=0;i<IN;i++){
            __nv_fp8_e4m3 wb; wb.__x = w[(size_t)o*IN+i];
            float bs = wscale_f[(size_t)(o/128)*SB + (i/128)];
            s += (double)(float(wb)*bs)*x[i];
        }
        ref[o]=(float)s;
    }

    uint8_t *d_w; __nv_bfloat16 *d_s; float *d_x,*d_out;
    CK(cudaMalloc(&d_w,w.size()));
    CK(cudaMalloc(&d_s,wscale.size()*sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&d_x,IN*sizeof(float)));
    CK(cudaMalloc(&d_out,OUT*sizeof(float)));
    CK(cudaMemcpy(d_w,w.data(),w.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_s,wscale.data(),wscale.size()*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x,x.data(),IN*sizeof(float),cudaMemcpyHostToDevice));
    fp8_block_gemv_launch(d_out,d_w,d_s,d_x,IN,OUT,0);
    CK(cudaDeviceSynchronize());
    std::vector<float> out(OUT);
    CK(cudaMemcpy(out.data(),d_out,OUT*sizeof(float),cudaMemcpyDeviceToHost));

    double dot=0,na=0,nb=0;
    for (int o=0;o<OUT;o++){ dot+=(double)out[o]*ref[o]; na+=(double)out[o]*out[o]; nb+=(double)ref[o]*ref[o]; }
    double cosine = dot/(sqrt(na)*sqrt(nb)+1e-12);
    printf("fp8_block_gemv: cosine=%.6f  (sample ref=%.4f out=%.4f)\n", cosine, ref[0], out[0]);
    int ok = cosine >= 0.999;

    // ── BATCHED parity: fp8_block_gemm (B rows) must equal B independent B=1 gemv calls, BITWISE ──
    const int B = 5;
    std::vector<float> xb((size_t)B*IN);
    for (int b=0;b<B;b++) for (int i=0;i<IN;i++) xb[(size_t)b*IN+i] = frand(-2.0f,2.0f);
    float *d_xb,*d_outb,*d_out1;
    CK(cudaMalloc(&d_xb,(size_t)B*IN*sizeof(float)));
    CK(cudaMalloc(&d_outb,(size_t)B*OUT*sizeof(float)));
    CK(cudaMalloc(&d_out1,OUT*sizeof(float)));
    CK(cudaMemcpy(d_xb,xb.data(),(size_t)B*IN*sizeof(float),cudaMemcpyHostToDevice));
    fp8_block_gemm_launch(d_outb,d_w,d_s,d_xb,IN,OUT,B,0);
    CK(cudaDeviceSynchronize());
    std::vector<float> outb((size_t)B*OUT), out1(OUT);
    CK(cudaMemcpy(outb.data(),d_outb,(size_t)B*OUT*sizeof(float),cudaMemcpyDeviceToHost));
    int rowmism=0;
    for (int b=0;b<B;b++){
        fp8_block_gemv_launch(d_out1,d_w,d_s,d_xb+(size_t)b*IN,IN,OUT,0);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(out1.data(),d_out1,OUT*sizeof(float),cudaMemcpyDeviceToHost));
        for (int o=0;o<OUT;o++) if (outb[(size_t)b*OUT+o] != out1[o]) rowmism++;
    }
    printf("fp8_block_gemm (B=%d) == %d× B=1 : %s (%d mismatches)\n", B, B, rowmism==0?"PASS":"FAIL", rowmism);
    ok = ok && (rowmism==0);

    // ── GROUPED (MoE) parity: fp8_block_gemm_grouped must equal per-expert B=1 gemv, BITWISE ──
    // E experts, each its own FP8 weight+scale; `total` rows grouped expert-contiguously.
    {
        const int E = 3, cnt[E] = {2, 3, 1};
        int coloff[E], total = 0;
        for (int e=0;e<E;e++){ coloff[e]=total; total+=cnt[e]; }
        const int gSB = IN/128, gOB = OUT/128;
        std::vector<uint8_t> gw((size_t)E*OUT*IN);
        for (auto &b : gw){ __nv_fp8_e4m3 v; v=__nv_fp8_e4m3(frand(-0.4f,0.4f)); b=v.__x; }
        std::vector<__nv_bfloat16> gs((size_t)E*gOB*gSB);
        for (auto &s : gs) s=__float2bfloat16(frand(0.02f,0.08f));
        std::vector<float> gx((size_t)total*IN);
        for (auto &v : gx) v=frand(-2.0f,2.0f);
        uint8_t *d_gw; __nv_bfloat16 *d_gs; float *d_gx,*d_gout,*d_g1; int *d_coloff,*d_count;
        CK(cudaMalloc(&d_gw,gw.size())); CK(cudaMalloc(&d_gs,gs.size()*sizeof(__nv_bfloat16)));
        CK(cudaMalloc(&d_gx,(size_t)total*IN*sizeof(float))); CK(cudaMalloc(&d_gout,(size_t)total*OUT*sizeof(float)));
        CK(cudaMalloc(&d_g1,(size_t)OUT*sizeof(float))); CK(cudaMalloc(&d_coloff,E*sizeof(int))); CK(cudaMalloc(&d_count,E*sizeof(int)));
        CK(cudaMemcpy(d_gw,gw.data(),gw.size(),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_gs,gs.data(),gs.size()*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_gx,gx.data(),(size_t)total*IN*sizeof(float),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_coloff,coloff,E*sizeof(int),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_count,cnt,E*sizeof(int),cudaMemcpyHostToDevice));
        fp8_block_gemm_grouped_launch(d_gout, d_gw, (int64_t)OUT*IN, d_gs, (int64_t)gOB*gSB,
                                      d_gx, d_coloff, d_count, NULL, 0, E, IN, OUT, 0);
        CK(cudaDeviceSynchronize());
        std::vector<float> gout((size_t)total*OUT), g1(OUT);
        CK(cudaMemcpy(gout.data(),d_gout,(size_t)total*OUT*sizeof(float),cudaMemcpyDeviceToHost));
        int gmism=0;
        for (int e=0;e<E;e++) for (int r=0;r<cnt[e];r++){
            int row=coloff[e]+r;
            fp8_block_gemv_launch(d_g1, d_gw+(size_t)e*OUT*IN, d_gs+(size_t)e*gOB*gSB, d_gx+(size_t)row*IN, IN, OUT, 0);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(g1.data(),d_g1,OUT*sizeof(float),cudaMemcpyDeviceToHost));
            for (int o=0;o<OUT;o++) if (gout[(size_t)row*OUT+o]!=g1[o]) gmism++;
        }
        printf("fp8_block_gemm_grouped (E=%d, tokens=%d) == per-expert B=1 : %s (%d mismatches)\n",
               E, total, gmism==0?"PASS":"FAIL", gmism);
        ok = ok && (gmism==0);

        // active-expert-list variant (decode-sized grid.y with -1 padding) must be BITWISE == full-E
        {
            const int NSLOT = E + 2;                 // active ids + -1 pads
            int h_active[NSLOT];
            for (int e=0;e<E;e++) h_active[e]=e;
            h_active[E]=-1; h_active[E+1]=-1;
            int *d_active; float *d_gout2;
            CK(cudaMalloc(&d_active,NSLOT*sizeof(int)));
            CK(cudaMalloc(&d_gout2,(size_t)total*OUT*sizeof(float)));
            CK(cudaMemcpy(d_active,h_active,NSLOT*sizeof(int),cudaMemcpyHostToDevice));
            fp8_block_gemm_grouped_launch(d_gout2, d_gw, (int64_t)OUT*IN, d_gs, (int64_t)gOB*gSB,
                                          d_gx, d_coloff, d_count, d_active, NSLOT, E, IN, OUT, 0);
            CK(cudaDeviceSynchronize());
            std::vector<float> gout2((size_t)total*OUT);
            CK(cudaMemcpy(gout2.data(),d_gout2,(size_t)total*OUT*sizeof(float),cudaMemcpyDeviceToHost));
            int amism=0;
            for (size_t i=0;i<gout2.size();i++) if (gout2[i]!=gout[i]) amism++;
            printf("fp8_block_gemm_grouped active-list == full-E : %s (%d mismatches)\n",
                   amism==0?"PASS":"FAIL", amism);
            ok = ok && (amism==0);
            cudaFree(d_active); cudaFree(d_gout2);
        }
    }

    printf("%s\n", ok?"PASS":"FAIL");
    return ok?0:1;
}
