// ABOUTME: Gate for the device greedy verify-accept kernel vs the P1 host oracle.
// ABOUTME: accepted_len + emitted token over a (1+K) target logit block, for every j in 0..K.
//
// Validates q35_dflash_verify_greedy_device (the serving-path accept step) against the P1 host
// oracle q35_dflash_verify_greedy over synthetic target logit blocks constructed so the greedy
// accepted length is exactly j, for every j in 0..K, plus the all-accept bonus case. Self-contained.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_verify.cu -o t && ./t
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "qwen35_dflash_forward.cuh"
#include "qwen35_dflash_reject.cuh"

#define CUDA_OK(c) do{ cudaError_t e=(c); if(e!=cudaSuccess){ printf("CUDA %s @ %d\n", cudaGetErrorString(e), __LINE__); return 2; } }while(0)

int main(){
    const int vocab=32, K=6, rows=K+1;
    int fails=0;
    float *dL; int32_t *dDraft,*dRow,*dEmit; int* dAcc;
    CUDA_OK(cudaMalloc(&dL,(size_t)rows*vocab*4)); CUDA_OK(cudaMalloc(&dDraft,K*sizeof(int32_t)));
    CUDA_OK(cudaMalloc(&dRow,rows*sizeof(int32_t))); CUDA_OK(cudaMalloc(&dEmit,sizeof(int32_t))); CUDA_OK(cudaMalloc(&dAcc,sizeof(int)));

    for(int target_j=0; target_j<=K; target_j++){
        // Build a (K+1)-row logit block with a known argmax per row, and drafts that match the
        // first target_j rows then diverge (unless target_j==K -> all match, bonus emitted).
        std::vector<float> L((size_t)rows*vocab, 0.0f);
        std::vector<int32_t> argmax(rows), draft(K);
        for(int r=0;r<rows;r++){ int am=(r*5+3)%vocab; argmax[r]=am; for(int v=0;v<vocab;v++) L[(size_t)r*vocab+v]=(v==am)?10.0f:0.1f*(v%3); }
        for(int i=0;i<K;i++) draft[i]=(i<target_j)?argmax[i]:(argmax[i]+1)%vocab;

        CUDA_OK(cudaMemcpy(dL,L.data(),(size_t)rows*vocab*4,cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(dDraft,draft.data(),K*sizeof(int32_t),cudaMemcpyHostToDevice));
        q35_dflash_verify_greedy_device(dL,rows,vocab,dDraft,K,dRow,dAcc,dEmit,0);
        CUDA_OK(cudaDeviceSynchronize());
        int gacc; int32_t gemit; CUDA_OK(cudaMemcpy(&gacc,dAcc,sizeof(int),cudaMemcpyDeviceToHost)); CUDA_OK(cudaMemcpy(&gemit,dEmit,sizeof(int32_t),cudaMemcpyDeviceToHost));

        // P1 host oracle over the same block.
        auto h=q35_dflash_verify_greedy(L.data(),vocab,draft.data(),K);
        if(gacc!=h.accepted_len || gemit!=h.emitted_token){ printf("FAIL j=%d: device(len=%d emit=%d) host(len=%d emit=%d)\n",target_j,gacc,gemit,h.accepted_len,h.emitted_token); fails++; }
        if(gacc!=target_j){ printf("FAIL j=%d: constructed accept != device %d\n",target_j,gacc); fails++; }
    }
    cudaFree(dL);cudaFree(dDraft);cudaFree(dRow);cudaFree(dEmit);cudaFree(dAcc);
    if(fails){ printf("FAIL — DFlash device verify-accept (%d)\n",fails); return 1; }
    printf("PASS — DFlash device greedy verify-accept matches P1 host oracle for all j in 0..%d\n",K);
    return 0;
}
