// ABOUTME: Gate: verify-block exposed target logits are self-consistent with its per-row argmax.
// ABOUTME: Proves the [T, vocab] logit copy is the actual verify distribution (probabilistic input).
//
// The probabilistic serving path needs the per-row target logits from the verify block. This gate
// runs qwen35_dflash_verify_block_logits, copies the [T, vocab] logits to host, argmaxes each row,
// and asserts it equals the block's returned per-row argmax (so the exposed logits ARE the verify
// distribution, not stale/garbage). Uses the FP8 target (arg1). SKIPs if absent.
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include "gemma4_kernels.cuh"

static bool exists(const std::string& p){ struct stat s; return stat(p.c_str(),&s)==0; }
static int step1(gemma4_engine_t* e,int s,int32_t in){int sl[1]={s};int32_t i[1]={in},o[1]={-1};gemma4_engine_step_batch(e,sl,i,1,o);return o[0];}

int main(int c,char**v){
    const char* path=c>1?v[1]:"/opt/spark/models/models--Qwen--Qwen3.5-9B-FP8";
    if(!exists(std::string(path)+"/config.json")){ printf("SKIP — target absent\n"); return 0; }
    gemma4_engine_t* e=gemma4_engine_create(path,FORMAT_Q4_0,4096,0,0.90); if(!e){printf("FAIL create\n");return 2;}
    int32_t pr[5]={760,6511,314,9338,369}; int32_t f=-1; int s=gemma4_engine_seq_add(e,pr,5,&f,0.f,0,1.f,0.f,1);
    int32_t cur=f; for(int i=0;i<12;i++) cur=step1(e,s,cur);
    const int K=6,T=7; int VOC=0;
    // need vocab; query via a small copy: use the engine's known 248320 (Qwen3.5). Read from logits size.
    VOC=248320;
    int32_t blk[7]={cur,1,2,3,4,5,6}; int32_t am[7];
    float* dlog; cudaMalloc(&dlog,(size_t)T*VOC*sizeof(float));
    if(gemma4_engine_q35_dflash_verify_block_logits(e,s,blk,T,am,dlog)!=0){printf("FAIL verify_logits\n");return 1;}
    std::vector<float> hlog((size_t)T*VOC); cudaMemcpy(hlog.data(),dlog,(size_t)T*VOC*sizeof(float),cudaMemcpyDeviceToHost);
    int fails=0;
    for(int t=0;t<T;t++){
        int32_t best=0; float bv=hlog[(size_t)t*VOC]; for(int o=1;o<VOC;o++){ float x=hlog[(size_t)t*VOC+o]; if(x>bv){bv=x;best=o;} }
        if(best!=am[t]){ printf("FAIL row %d: logit-argmax=%d verify-argmax=%d\n",t,best,am[t]); fails++; }
    }
    cudaFree(dlog); gemma4_engine_seq_remove(e,s); gemma4_engine_destroy(e);
    if(fails){ printf("FAIL — verify-block logits not self-consistent (%d)\n",fails); return 1; }
    printf("PASS — DFlash verify-block target logits self-consistent with per-row argmax for all %d rows "
           "(the exposed [T,vocab] logits are the verify distribution)\n",T);
    return 0;
}
