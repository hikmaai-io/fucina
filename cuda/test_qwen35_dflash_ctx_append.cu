// ABOUTME: Gate: incremental context-KV append == full recompute (accumulating draft KV cache).
// ABOUTME: Split the context into chunks, append each, and compare to a one-shot full precompute.
//
// The serving loop grows the draft context KV cache incrementally (each decode step inserts only the
// newly-committed target tokens). This gate proves that appending context in chunks
// (precompute_context_kv_at with dst_row>0) yields a cache byte-identical to a single full
// precompute over all rows at once, on the real weights. SKIPs cleanly when the checkpoint is absent.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_ctx_append.cu -o t && ./t
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include "qwen35_dflash_forward.cuh"

#define CUDA_OK(c) do{ cudaError_t e=(c); if(e!=cudaSuccess){ printf("CUDA %s @ %d\n", cudaGetErrorString(e), __LINE__); return 2; } }while(0)
static bool exists(const std::string& p){ struct stat s; return stat(p.c_str(),&s)==0; }

int main(int argc,char**argv){
    const char* def="/opt/spark/models/models--z-lab--Qwen3.5-9B-DFlash/"
                    "snapshots/5fc3b3d474760f18c516db87d84c37edbfd3ede6";
    std::string dir=(argc>1)?argv[1]:def;
    if(!exists(dir+"/model.safetensors")){ printf("SKIP — real DFlash checkpoint absent\n"); return 0; }
    std::ifstream f(dir+"/config.json"); std::stringstream ss; ss<<f.rdbuf(); std::string cj=ss.str();
    q35_dflash_residency R{}; std::string err;
    if(!qwen35dflash::parse_config(cj,R.geom,err)){ printf("FAIL parse: %s\n",err.c_str()); return 1; }
    st::Model M; if(!M.open((dir+"/model.safetensors").c_str(),err)){ printf("FAIL open: %s\n",err.c_str()); return 1; }
    if(!qwen35dflash::validate_tensors(M,R.geom,R.geom.V,err)){ printf("FAIL validate: %s\n",err.c_str()); return 1; }
    if(q35_dflash_residency_upload(&R,M,err)!=0){ printf("FAIL upload: %s\n",err.c_str()); return 1; }

    const auto& g=R.geom; const int H=g.H,kvd=g.kv_dim(),L=g.L,cap=32,total=13; const double theta=1e7; const float eps=1e-6f;
    std::vector<float> X((size_t)total*H); for(int r=0;r<total;r++) for(int i=0;i<H;i++) X[(size_t)r*H+i]=0.04f*std::sin(0.0015f*i+0.5f*r)+0.008f*(r+1);
    std::vector<int> pos(total); for(int r=0;r<total;r++) pos[r]=r;

    float* dX; CUDA_OK(cudaMalloc(&dX,(size_t)total*H*4)); CUDA_OK(cudaMemcpy(dX,X.data(),(size_t)total*H*4,cudaMemcpyHostToDevice));
    int* dPos; CUDA_OK(cudaMalloc(&dPos,total*sizeof(int))); CUDA_OK(cudaMemcpy(dPos,pos.data(),total*sizeof(int),cudaMemcpyHostToDevice));
    std::vector<float*> fullK(L),fullV(L),incK(L),incV(L);
    for(int l=0;l<L;l++){ CUDA_OK(cudaMalloc(&fullK[l],(size_t)cap*kvd*4)); CUDA_OK(cudaMalloc(&fullV[l],(size_t)cap*kvd*4)); CUDA_OK(cudaMalloc(&incK[l],(size_t)cap*kvd*4)); CUDA_OK(cudaMalloc(&incV[l],(size_t)cap*kvd*4)); }
    q35_dflash_ctx_scratch c{}; if(!q35_dflash_ctx_scratch_alloc(&c,g,cap)){ printf("FAIL scratch\n"); return 1; }

    // Full precompute over all `total` rows at once.
    q35_dflash_precompute_context_kv_at(R,c,dX,dPos,total,fullK.data(),fullV.data(),0,cap,theta,eps,0);
    // Incremental: chunks of 5, 4, 4 appended at growing offsets.
    int chunks[3]={5,4,4}, off=0;
    for(int ci=0;ci<3;ci++){ int n=chunks[ci]; q35_dflash_precompute_context_kv_at(R,c,dX+(size_t)off*H,dPos+off,n,incK.data(),incV.data(),off,cap,theta,eps,0); off+=n; }
    CUDA_OK(cudaDeviceSynchronize());

    int fails=0;
    for(int l=0;l<L && !fails;l++){
        std::vector<float> a((size_t)total*kvd),b((size_t)total*kvd),av((size_t)total*kvd),bv((size_t)total*kvd);
        CUDA_OK(cudaMemcpy(a.data(),fullK[l],(size_t)total*kvd*4,cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(b.data(),incK[l],(size_t)total*kvd*4,cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(av.data(),fullV[l],(size_t)total*kvd*4,cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(bv.data(),incV[l],(size_t)total*kvd*4,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<a.size();i++){ if(a[i]!=b[i] || av[i]!=bv[i]){ printf("FAIL L%d @ %zu K(%g vs %g) V(%g vs %g)\n",l,i,a[i],b[i],av[i],bv[i]); fails++; break; } }
    }
    for(int l=0;l<L;l++){ cudaFree(fullK[l]);cudaFree(fullV[l]);cudaFree(incK[l]);cudaFree(incV[l]); }
    q35_dflash_ctx_scratch_free(&c); cudaFree(dX); cudaFree(dPos); q35_dflash_residency_free(&R);
    if(fails){ printf("FAIL — DFlash context append != full recompute (%d)\n",fails); return 1; }
    printf("PASS — DFlash incremental context-KV append == full recompute (byte-identical) across "
           "%d layers, %d rows in chunks 5+4+4 (accumulating draft KV cache correct)\n",L,total);
    return 0;
}
