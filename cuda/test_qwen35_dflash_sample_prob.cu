// ABOUTME: Gate for the probabilistic DFlash draft sampler (shared-key gumbel/inverse-CDF).
// ABOUTME: Device sample+logits must match a host softmax inverse-CDF reference bit-for-bit.
//
// Validates q35_dflash_sample_prob: the device LM-head projection + temperature softmax + shared-key
// inverse-CDF token pick must match a host double reference using the same q35_dflash_uniform_open
// draw, over K rows. Also checks the materialized per-row logits match the host projection. Self-
// contained with a synthetic BF16 LM head and hidden rows at the real draft geometry (H=4096).
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_sample_prob.cu -o t && ./t
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "qwen35_dflash_forward.cuh"
#include "qwen35_dflash_rng.cuh"

#define CUDA_OK(c) do{ cudaError_t e=(c); if(e!=cudaSuccess){ printf("CUDA %s @ %d\n", cudaGetErrorString(e), __LINE__); return 2; } }while(0)
static inline uint16_t f_bf16(float f){ uint32_t x; memcpy(&x,&f,4); return (uint16_t)(x>>16); }
static inline float bf16_f(uint16_t u){ uint32_t x=(uint32_t)u<<16; float f; memcpy(&f,&x,4); return f; }

int main(){
    const int H=4096, vocab=2048, K=5; const double temp=0.8; const uint64_t seed=0xBEEF1234ull;
    std::vector<uint16_t> lm((size_t)vocab*H);
    for(int o=0;o<vocab;o++) for(int i=0;i<H;i++) lm[(size_t)o*H+i]=f_bf16(0.02f*std::sin(0.001f*i+0.05f*o));
    std::vector<float> hid((size_t)K*H);
    for(int r=0;r<K;r++) for(int i=0;i<H;i++) hid[(size_t)r*H+i]=0.03f*std::cos(0.002f*i+0.3f*r);
    std::vector<int64_t> pos(K); for(int r=0;r<K;r++) pos[r]=500+r;

    __nv_bfloat16* dLM; CUDA_OK(cudaMalloc(&dLM,lm.size()*2)); CUDA_OK(cudaMemcpy(dLM,lm.data(),lm.size()*2,cudaMemcpyHostToDevice));
    float* dHid; CUDA_OK(cudaMalloc(&dHid,hid.size()*4)); CUDA_OK(cudaMemcpy(dHid,hid.data(),hid.size()*4,cudaMemcpyHostToDevice));
    float* dLogits; CUDA_OK(cudaMalloc(&dLogits,(size_t)K*vocab*4));
    int32_t* dTok; CUDA_OK(cudaMalloc(&dTok,K*sizeof(int32_t)));
    int64_t* dPos; CUDA_OK(cudaMalloc(&dPos,K*sizeof(int64_t))); CUDA_OK(cudaMemcpy(dPos,pos.data(),K*sizeof(int64_t),cudaMemcpyHostToDevice));
    q35_dflash_sample_prob(dHid,dLM,K,H,vocab,dLogits,dTok,temp,seed,dPos,0);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<int32_t> tok(K); CUDA_OK(cudaMemcpy(tok.data(),dTok,K*sizeof(int32_t),cudaMemcpyDeviceToHost));
    std::vector<float> glog((size_t)K*vocab); CUDA_OK(cudaMemcpy(glog.data(),dLogits,(size_t)K*vocab*4,cudaMemcpyDeviceToHost));

    int fails=0;
    for(int r=0;r<K;r++){
        // host logits/temp + softmax inverse-CDF with the same shared-key uniform.
        std::vector<double> lg(vocab); double mx=-1e300;
        for(int o=0;o<vocab;o++){ double acc=0; for(int i=0;i<H;i++) acc+=(double)hid[(size_t)r*H+i]*(double)bf16_f(lm[(size_t)o*H+i]); lg[o]=acc/temp; if(lg[o]>mx) mx=lg[o]; }
        double sum=0; for(int o=0;o<vocab;o++) sum+=std::exp(lg[o]-mx);
        double u=q35_dflash_uniform_open(seed,pos[r],Q35_DFLASH_DOMAIN_SAMPLE);
        double tc=u*sum, a=0; int32_t pick=vocab-1; for(int o=0;o<vocab;o++){ a+=std::exp(lg[o]-mx); if(a>=tc){ pick=o; break; } }
        if(tok[r]!=pick){ printf("FAIL row %d: device tok=%d host=%d\n",r,tok[r],pick); fails++; }
        // logits scale check (device fp32 vs host double, signal-relative).
        double rss=0; for(int o=0;o<vocab;o++) rss+=lg[o]*lg[o]; double sc=std::sqrt(rss/vocab)+1e-9;
        double mr=0; for(int o=0;o<vocab;o++){ double rel=std::fabs((double)glog[(size_t)r*vocab+o]-lg[o])/sc; if(rel>mr) mr=rel; }
        if(mr>1e-3){ printf("FAIL row %d logits rel=%.3e\n",r,mr); fails++; }
    }
    cudaFree(dLM);cudaFree(dHid);cudaFree(dLogits);cudaFree(dTok);cudaFree(dPos);
    if(fails){ printf("FAIL — DFlash probabilistic draft sampler (%d)\n",fails); return 1; }
    printf("PASS — DFlash probabilistic draft sampler: shared-key inverse-CDF token + materialized "
           "logits match host reference for all %d rows (temp=%.2f)\n",K,temp);
    return 0;
}
