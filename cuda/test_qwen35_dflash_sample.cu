// ABOUTME: Gate for DFlash greedy draft sampling (LM head argmax) vs a host reference.
// ABOUTME: Projects sampled query hidden rows through a BF16 LM head and argmaxes; parity + ties.
//
// Validates q35_dflash_sample_greedy: the device LM-head projection + deterministic argmax over K
// sampled rows must match a host double reference (lowest index wins ties). Self-contained with a
// synthetic BF16 LM head and hidden rows at the real draft geometry (H=4096); the head projection
// is weight-independent math, so this runs anywhere. Includes an explicit tie case to lock the
// lowest-index rule.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_sample.cu -o t && ./t
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "qwen35_dflash_forward.cuh"

#define CUDA_OK(c) do{ cudaError_t e=(c); if(e!=cudaSuccess){ printf("CUDA %s @ %d\n", cudaGetErrorString(e), __LINE__); return 2; } }while(0)
static inline uint16_t f_bf16(float f){ uint32_t x; memcpy(&x,&f,4); return (uint16_t)(x>>16); }
static inline float bf16_f(uint16_t u){ uint32_t x=(uint32_t)u<<16; float f; memcpy(&f,&x,4); return f; }

int main(){
    const int H=4096, vocab=2048, K=5;
    std::vector<uint16_t> lm((size_t)vocab*H);
    for(int o=0;o<vocab;o++) for(int i=0;i<H;i++) lm[(size_t)o*H+i]=f_bf16(0.02f*std::sin(0.001f*i+0.05f*o));
    std::vector<float> hid((size_t)K*H);
    for(int r=0;r<K;r++) for(int i=0;i<H;i++) hid[(size_t)r*H+i]=0.03f*std::cos(0.002f*i+0.3f*r);

    // device
    __nv_bfloat16* dLM; CUDA_OK(cudaMalloc(&dLM,lm.size()*2)); CUDA_OK(cudaMemcpy(dLM,lm.data(),lm.size()*2,cudaMemcpyHostToDevice));
    float* dHid; CUDA_OK(cudaMalloc(&dHid,hid.size()*4)); CUDA_OK(cudaMemcpy(dHid,hid.data(),hid.size()*4,cudaMemcpyHostToDevice));
    int32_t* dTok; CUDA_OK(cudaMalloc(&dTok,K*sizeof(int32_t)));
    q35_dflash_sample_greedy(dHid,dLM,K,H,vocab,dTok,0);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<int32_t> tok(K); CUDA_OK(cudaMemcpy(tok.data(),dTok,K*sizeof(int32_t),cudaMemcpyDeviceToHost));

    // host reference (double accumulation, lowest-index tie)
    int fails=0;
    for(int r=0;r<K;r++){
        double bv=-1e300; int bi=0;
        for(int o=0;o<vocab;o++){ double acc=0; for(int i=0;i<H;i++) acc+=(double)hid[(size_t)r*H+i]*(double)bf16_f(lm[(size_t)o*H+i]); if(acc>bv){ bv=acc; bi=o; } }
        if(tok[r]!=bi){ printf("FAIL: row %d device=%d host=%d\n",r,tok[r],bi); fails++; }
    }

    // explicit tie: two identical head rows -> lowest index must win.
    {
        std::vector<uint16_t> lm2((size_t)4*H); for(int i=0;i<H;i++){ uint16_t a=f_bf16(0.01f*std::sin(0.003f*i)); lm2[i]=a; lm2[(size_t)1*H+i]=f_bf16(0.02f); lm2[(size_t)2*H+i]=a; lm2[(size_t)3*H+i]=f_bf16(-0.02f); }
        std::vector<float> h(H); for(int i=0;i<H;i++) h[i]=0.05f*std::cos(0.002f*i);
        __nv_bfloat16* d2; cudaMalloc(&d2,lm2.size()*2); cudaMemcpy(d2,lm2.data(),lm2.size()*2,cudaMemcpyHostToDevice);
        float* dh; cudaMalloc(&dh,H*4); cudaMemcpy(dh,h.data(),H*4,cudaMemcpyHostToDevice);
        int32_t* dt; cudaMalloc(&dt,sizeof(int32_t));
        q35_dflash_sample_greedy(dh,d2,1,H,4,dt,0); cudaDeviceSynchronize();
        int32_t t; cudaMemcpy(&t,dt,sizeof(int32_t),cudaMemcpyDeviceToHost);
        // rows 0 and 2 are identical; whichever is the argmax, lowest index (0) must win iff row 0 is a max.
        double s0=0,s2=0; for(int i=0;i<H;i++){ s0+=(double)h[i]*(double)bf16_f(lm2[i]); s2+=(double)h[i]*(double)bf16_f(lm2[(size_t)2*H+i]); }
        double s1=0,s3=0; for(int i=0;i<H;i++){ s1+=(double)h[i]*(double)bf16_f(lm2[(size_t)1*H+i]); s3+=(double)h[i]*(double)bf16_f(lm2[(size_t)3*H+i]); }
        int host_best=0; double bv=s0; if(s1>bv){bv=s1;host_best=1;} if(s2>bv){bv=s2;host_best=2;} if(s3>bv){bv=s3;host_best=3;}
        if(t!=host_best){ printf("FAIL tie: device=%d host=%d (s0=%g s1=%g s2=%g s3=%g)\n",t,host_best,s0,s1,s2,s3); fails++; }
        cudaFree(d2);cudaFree(dh);cudaFree(dt);
    }

    cudaFree(dLM);cudaFree(dHid);cudaFree(dTok);
    if(fails){ printf("FAIL — DFlash draft sampling (%d)\n",fails); return 1; }
    printf("PASS — DFlash greedy draft sampling: LM-head argmax matches host reference for %d rows + tie rule\n",K);
    return 0;
}
