// ABOUTME: Gate for the DFlash aux gather (capture [slot][row][H] -> drafter [row][F*H]).
// ABOUTME: Verifies the transpose/gather layout the real-drafter wiring depends on.
//
// The target aux capture stores [feature_slot][row][H]; the drafter fc wants concat[row][f*H+i].
// This gate fills a synthetic capture buffer with a known function of (slot,row,i), gathers a row
// window, and asserts every gathered element lands at the right concat offset. Self-contained.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_aux_gather.cu -o t && ./t
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "qwen35_dflash_forward.cuh"

#define CUDA_OK(c) do{ cudaError_t e=(c); if(e!=cudaSuccess){ printf("CUDA %s @ %d\n", cudaGetErrorString(e), __LINE__); return 2; } }while(0)

int main(){
    const int F=8, maxrows=32, H=64, row_base=3, rows=7;
    std::vector<float> aux((size_t)F*maxrows*H);
    auto val=[&](int f,int r,int i){ return (float)(f*1000000 + r*1000 + i); };
    for(int f=0;f<F;f++) for(int r=0;r<maxrows;r++) for(int i=0;i<H;i++) aux[(size_t)f*maxrows*H+(size_t)r*H+i]=val(f,r,i);

    float *dAux,*dOut; CUDA_OK(cudaMalloc(&dAux,aux.size()*4)); CUDA_OK(cudaMemcpy(dAux,aux.data(),aux.size()*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&dOut,(size_t)rows*F*H*4));
    q35_dflash_aux_gather(dAux,dOut,F,maxrows,H,row_base,rows,0);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> out((size_t)rows*F*H); CUDA_OK(cudaMemcpy(out.data(),dOut,out.size()*4,cudaMemcpyDeviceToHost));

    int fails=0;
    for(int r=0;r<rows && fails<10;r++) for(int f=0;f<F && fails<10;f++) for(int i=0;i<H;i++){
        float got=out[(size_t)r*F*H+(size_t)f*H+i], want=val(f,row_base+r,i);
        if(got!=want){ printf("FAIL r=%d f=%d i=%d got=%g want=%g\n",r,f,i,got,want); fails++; break; }
    }
    cudaFree(dAux); cudaFree(dOut);
    if(fails){ printf("FAIL — DFlash aux gather (%d)\n",fails); return 1; }
    printf("PASS — DFlash aux gather: [slot][row][H] -> [row][F*H] concat for a %d-row window (F=%d H=%d)\n",rows,F,H);
    return 0;
}
