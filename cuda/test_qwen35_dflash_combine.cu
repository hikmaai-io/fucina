// ABOUTME: Gate for the DFlash aux-hidden combine (fc) on real weights — target->draft interface.
// ABOUTME: fc(concat of F target-layer hidden states) -> draft input hidden, device vs host double.
//
// The draft consumes fc(concat of F=8 target-layer hidden states) as its per-row input hidden
// (target_layer_ids [1,5,9,13,17,21,25,29], fc_in = 8*4096 = 32768). This gate validates
// q35_dflash_combine_aux reading the resident BF16 fc weight against a host double reference on the
// real z-lab checkpoint. This is the exact interface the target engine must produce to drive the
// draft. SKIPs cleanly when the checkpoint is absent.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_combine.cu -o t && ./t
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
static inline float bf16f(uint16_t u){ uint32_t x=(uint32_t)u<<16; float f; memcpy(&f,&x,4); return f; }
static std::vector<float> tf(const st::Tensor* t){ size_t n=t->nbytes/2; const uint16_t* p=(const uint16_t*)t->data; std::vector<float> o(n); for(size_t i=0;i<n;i++) o[i]=bf16f(p[i]); return o; }

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
    if(!R.geom.use_aux_hidden){ printf("SKIP — checkpoint has no fc/aux-hidden path\n"); return 0; }
    if(q35_dflash_residency_upload(&R,M,err)!=0){ printf("FAIL upload: %s\n",err.c_str()); return 1; }

    const auto& g=R.geom; const int H=g.H, fin=g.fc_in(), rows=6;
    std::vector<float> cat((size_t)rows*fin);
    for(int r=0;r<rows;r++) for(int i=0;i<fin;i++) cat[(size_t)r*fin+i]=0.02f*std::sin(0.0007f*i+0.3f*r)+0.004f*(r+1);

    float* dCat; CUDA_OK(cudaMalloc(&dCat,(size_t)rows*fin*4)); CUDA_OK(cudaMemcpy(dCat,cat.data(),(size_t)rows*fin*4,cudaMemcpyHostToDevice));
    float* dOut; CUDA_OK(cudaMalloc(&dOut,(size_t)rows*H*4));
    q35_dflash_combine_aux(R,dCat,dOut,rows,0);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> out((size_t)rows*H); CUDA_OK(cudaMemcpy(out.data(),dOut,(size_t)rows*H*4,cudaMemcpyDeviceToHost));

    auto fc=tf(M.find("fc.weight"));   // [H, fin]
    double max_rel=0;
    for(int r=0;r<rows;r++){
        std::vector<double> ref(H); double rss=0;
        for(int o=0;o<H;o++){ double a=0; for(int i=0;i<fin;i++) a+=(double)cat[(size_t)r*fin+i]*(double)fc[(size_t)o*fin+i]; ref[o]=a; rss+=a*a; }
        double scale=std::sqrt(rss/H)+1e-12;
        for(int o=0;o<H;o++){ double rel=std::fabs((double)out[(size_t)r*H+o]-ref[o])/scale; if(rel>max_rel) max_rel=rel; }
    }
    cudaFree(dCat); cudaFree(dOut); q35_dflash_residency_free(&R);
    printf("aux-hidden combine (fc) parity on real weights: max signal-rel err=%.3e (rows=%d fc_in=%d)\n",max_rel,rows,fin);
    const double TOL=2e-3;
    if(max_rel>TOL){ printf("FAIL — fc combine parity exceeds %.1e\n",TOL); return 1; }
    printf("PASS — DFlash aux-hidden combine (fc) matches host double reference on real weights within %.1e\n",TOL);
    return 0;
}
