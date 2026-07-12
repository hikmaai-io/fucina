// ABOUTME: Gate for the device probabilistic verify-accept kernel vs the P1 host oracle.
// ABOUTME: accepted_len + emitted token over target+draft logit blocks, multiple seeds.
//
// Validates q35_dflash_verify_prob_device (the serving-path probabilistic accept step) against the
// P1 host oracle q35_dflash_verify_prob over synthetic target/draft logit blocks and many seeds.
// Self-contained. The device kernel calls the shared __host__ __device__ reference, so this also
// confirms the reject header compiles + runs on-device.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_verify_prob.cu -o t && ./t
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "qwen35_dflash_forward.cuh"

#define CUDA_OK(c) do{ cudaError_t e=(c); if(e!=cudaSuccess){ printf("CUDA %s @ %d\n", cudaGetErrorString(e), __LINE__); return 2; } }while(0)

int main(){
    const int vocab=24, K=4;
    std::vector<float> tl((size_t)(K+1)*vocab), dl((size_t)K*vocab);
    for(int r=0;r<=K;r++) for(int v=0;v<vocab;v++) tl[(size_t)r*vocab+v]=(float)((v*7+r*3)%11)*0.4f;
    for(int r=0;r<K;r++) for(int v=0;v<vocab;v++) dl[(size_t)r*vocab+v]=(float)((v*5+r*2)%9)*0.5f;
    int32_t dt[4]; for(int i=0;i<K;i++) dt[i]=q35_dflash_argmax(&dl[(size_t)i*vocab],vocab);
    int64_t pos[4]={500,501,502,503}, posb=504;

    float *dT,*dD; int32_t *dDT; int64_t *dPos; int *dAcc; int32_t *dEmit;
    CUDA_OK(cudaMalloc(&dT,tl.size()*4)); CUDA_OK(cudaMemcpy(dT,tl.data(),tl.size()*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&dD,dl.size()*4)); CUDA_OK(cudaMemcpy(dD,dl.data(),dl.size()*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&dDT,K*sizeof(int32_t))); CUDA_OK(cudaMemcpy(dDT,dt,K*sizeof(int32_t),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&dPos,K*sizeof(int64_t))); CUDA_OK(cudaMemcpy(dPos,pos,K*sizeof(int64_t),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&dAcc,sizeof(int))); CUDA_OK(cudaMalloc(&dEmit,sizeof(int32_t)));

    int nseed=0, agree=0;
    for(uint64_t seed : {1ull,2ull,3ull,42ull,1000ull,0xABCDEFull,0xFFFFFFFFull,7777ull,0x9e3779b9ull,123456789ull}){
        q35_dflash_verify_prob_device(dT,dD,vocab,dDT,K,seed,dPos,posb,dAcc,dEmit,0);
        CUDA_OK(cudaDeviceSynchronize());
        int gacc; int32_t gemit; CUDA_OK(cudaMemcpy(&gacc,dAcc,sizeof(int),cudaMemcpyDeviceToHost)); CUDA_OK(cudaMemcpy(&gemit,dEmit,sizeof(int32_t),cudaMemcpyDeviceToHost));
        auto h=q35_dflash_verify_prob(tl.data(),dl.data(),vocab,dt,K,seed,pos,posb);
        nseed++;
        if(gacc!=h.accepted_len || gemit!=h.emitted_token){ printf("FAIL seed=%llu dev(len=%d emit=%d) host(len=%d emit=%d)\n",(unsigned long long)seed,gacc,gemit,h.accepted_len,h.emitted_token); }
        else agree++;
    }
    cudaFree(dT);cudaFree(dD);cudaFree(dDT);cudaFree(dPos);cudaFree(dAcc);cudaFree(dEmit);
    if(agree!=nseed){ printf("FAIL — device probabilistic verify-accept (%d/%d)\n",agree,nseed); return 1; }
    printf("PASS — DFlash device probabilistic verify-accept matches P1 host oracle for %d seeds\n",nseed);
    return 0;
}
