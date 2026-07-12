// ABOUTME: Gate for the single DFlash drafting entry point (fc->precompute->query->sample).
// ABOUTME: On real weights: K in-vocab tokens + run-to-run determinism (repeated identical inputs).
//
// Validates q35_dflash_draft_greedy end to end on the real z-lab weights: one call runs the full
// drafting pipeline (fc combine of aux features -> context-KV precompute -> query forward -> greedy
// LM-head argmax) and must produce K draft tokens that are all in [0,vocab). Two calls with
// identical inputs must produce byte-identical tokens (the determinism the verify path relies on).
// The shared LM head is synthetic here (the drafting compute is validated numerically by the other
// gates; this gate exercises the composed entry point + determinism). SKIPs when checkpoint absent.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_draft.cu -o t && ./t
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "qwen35_dflash_forward.cuh"

#define CUDA_OK(c) do{ cudaError_t e=(c); if(e!=cudaSuccess){ printf("CUDA %s @ %d\n", cudaGetErrorString(e), __LINE__); return 2; } }while(0)
static bool exists(const std::string& p){ struct stat s; return stat(p.c_str(),&s)==0; }
static inline uint16_t f_bf16(float f){ uint32_t x; memcpy(&x,&f,4); return (uint16_t)(x>>16); }

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

    const auto& g=R.geom; const int H=g.H, fin=g.fc_in(), K=6, vocab=4096, num_ctx=12, ctx_cap=64, rows=1+K;
    const double theta=1e7; const float eps=1e-6f;

    // Synthetic aux features (target hidden concat) + a synthetic shared LM head.
    std::vector<float> caux((size_t)num_ctx*fin), qaux((size_t)rows*fin);
    for(int r=0;r<num_ctx;r++) for(int i=0;i<fin;i++) caux[(size_t)r*fin+i]=0.02f*std::sin(0.0007f*i+0.3f*r)+0.004f*(r+1);
    for(int r=0;r<rows;r++) for(int i=0;i<fin;i++) qaux[(size_t)r*fin+i]=0.02f*std::cos(0.0009f*i+0.35f*r)+0.003f*(r+1);
    std::vector<uint16_t> lm((size_t)vocab*H); for(int o=0;o<vocab;o++) for(int i=0;i<H;i++) lm[(size_t)o*H+i]=f_bf16(0.02f*std::sin(0.001f*i+0.05f*o));
    std::vector<int> cpos(num_ctx); for(int r=0;r<num_ctx;r++) cpos[r]=r;
    std::vector<int> qpos(rows); for(int r=0;r<rows;r++) qpos[r]=num_ctx-1+r;   // query follows context

    float *dCaux,*dQaux; __nv_bfloat16* dLM; int32_t *dT1,*dT2;
    CUDA_OK(cudaMalloc(&dCaux,(size_t)num_ctx*fin*4)); CUDA_OK(cudaMemcpy(dCaux,caux.data(),(size_t)num_ctx*fin*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&dQaux,(size_t)rows*fin*4)); CUDA_OK(cudaMemcpy(dQaux,qaux.data(),(size_t)rows*fin*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&dLM,lm.size()*2)); CUDA_OK(cudaMemcpy(dLM,lm.data(),lm.size()*2,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&dT1,K*sizeof(int32_t))); CUDA_OK(cudaMalloc(&dT2,K*sizeof(int32_t)));

    q35_dflash_drafter D{}; if(!q35_dflash_drafter_init(&D,R,K,ctx_cap)){ printf("FAIL drafter init\n"); return 1; }

    if(q35_dflash_draft_greedy(R,D,dCaux,cpos.data(),num_ctx,dQaux,qpos.data(),dLM,vocab,dT1,theta,eps,0)!=0){ printf("FAIL draft 1\n"); return 1; }
    CUDA_OK(cudaDeviceSynchronize());
    if(q35_dflash_draft_greedy(R,D,dCaux,cpos.data(),num_ctx,dQaux,qpos.data(),dLM,vocab,dT2,theta,eps,0)!=0){ printf("FAIL draft 2\n"); return 1; }
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<int32_t> t1(K),t2(K);
    CUDA_OK(cudaMemcpy(t1.data(),dT1,K*sizeof(int32_t),cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(t2.data(),dT2,K*sizeof(int32_t),cudaMemcpyDeviceToHost));

    int fails=0;
    for(int i=0;i<K;i++){ if(t1[i]<0||t1[i]>=vocab){ printf("FAIL: draft token %d = %d out of vocab\n",i,t1[i]); fails++; } if(t1[i]!=t2[i]){ printf("FAIL: nondeterministic at %d: %d vs %d\n",i,t1[i],t2[i]); fails++; } }
    printf("draft tokens:"); for(int i=0;i<K;i++) printf(" %d",t1[i]); printf("\n");

    q35_dflash_drafter_free(&D,R); q35_dflash_residency_free(&R);
    cudaFree(dCaux);cudaFree(dQaux);cudaFree(dLM);cudaFree(dT1);cudaFree(dT2);
    if(fails){ printf("FAIL — DFlash drafting entry point (%d)\n",fails); return 1; }
    printf("PASS — DFlash drafting entry point: %d in-vocab draft tokens, run-to-run byte-identical "
           "(fc -> precompute -> query forward -> greedy sample composed over real weights)\n",K);
    return 0;
}
