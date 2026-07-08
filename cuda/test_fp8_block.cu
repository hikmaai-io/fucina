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
    printf("%s\n", ok?"PASS":"FAIL");
    return ok?0:1;
}
