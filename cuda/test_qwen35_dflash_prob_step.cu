// ABOUTME: Functional gate for the probabilistic DFlash real step (assembly correctness).
// ABOUTME: Emits in-vocab tokens and is deterministic for a fixed seed on the real FP8 target.
//
// The probabilistic real step is distribution-preserving BY the P1 math (proven statistically in
// qwen35-dflash-prob-dist-test, TV=0.0015). This gate validates the ENGINE ASSEMBLY: over a run it
// (a) produces only in-vocab tokens and (b) is run-to-run byte-identical for a fixed seed (the
// shared-key RNG makes the whole path deterministic). Requires the FP8 target (arg1) +
// FUCINA_QWEN35_DFLASH_PATH. SKIPs if absent.
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <sys/stat.h>
#include "gemma4_kernels.cuh"

static bool exists(const std::string& p){ struct stat s; return stat(p.c_str(),&s)==0; }
static int step1(gemma4_engine_t* e,int s,int32_t in){int sl[1]={s};int32_t i[1]={in},o[1]={-1};gemma4_engine_step_batch(e,sl,i,1,o);return o[0];}

static bool run(gemma4_engine_t* e, uint64_t seed, int N, std::vector<int32_t>& emitted, int VOC){
    int32_t pr[5]={760,6511,314,9338,369};
    gemma4_engine_q35_dflash_reset_context(e);
    int32_t f=-1; int s=gemma4_engine_seq_add(e,pr,5,&f,0.f,0,1.f,0.f,1);
    if(s<0) return false;
    int32_t cur=f; for(int i=0;i<8;i++) cur=step1(e,s,cur);
    int32_t bonus=cur; int steps=0;
    while((int)emitted.size()<N && steps<N+8){
        int32_t out[64]; int n=0; int32_t nb=0;
        if(gemma4_engine_q35_dflash_real_step_prob(e,s,bonus,0.8,seed,out,&n,&nb)!=0){ gemma4_engine_seq_remove(e,s); return false; }
        for(int i=0;i<n;i++){ if(out[i]<0||out[i]>=VOC){ printf("OOB token %d\n",out[i]); gemma4_engine_seq_remove(e,s); return false; } emitted.push_back(out[i]); }
        bonus=nb; steps++;
    }
    gemma4_engine_seq_remove(e,s);
    return true;
}

int main(int c,char**v){
    const char* path=c>1?v[1]:"/opt/spark/models/models--Qwen--Qwen3.5-9B-FP8";
    const char* dpath=getenv("FUCINA_QWEN35_DFLASH_PATH");
    if(!exists(std::string(path)+"/config.json")){ printf("SKIP — target absent\n"); return 0; }
    if(!dpath||!exists(std::string(dpath)+"/model.safetensors")){ printf("SKIP — draft path unset/absent\n"); return 0; }
    gemma4_engine_t* e=gemma4_engine_create(path,FORMAT_Q4_0,4096,0,0.90); if(!e){printf("FAIL create\n");return 2;}
    if(gemma4_engine_q35_dflash_load(e)!=0){ printf("FAIL draft load\n"); return 1; }
    const int VOC=248320, N=24;
    std::vector<int32_t> a,b;
    if(!run(e,0xABCDEF12ull,N,a,VOC)){ printf("FAIL run a\n"); return 1; }
    if(!run(e,0xABCDEF12ull,N,b,VOC)){ printf("FAIL run b\n"); return 1; }
    gemma4_engine_destroy(e);
    if(a.size()!=b.size()){ printf("FAIL: size mismatch %zu vs %zu\n",a.size(),b.size()); return 1; }
    int fails=0; for(size_t i=0;i<a.size();i++) if(a[i]!=b[i]){ if(fails<3)printf("NONDET @%zu: %d vs %d\n",i,a[i],b[i]); fails++; }
    if(fails){ printf("FAIL — probabilistic step not deterministic for fixed seed (%d)\n",fails); return 1; }
    printf("PASS — DFlash probabilistic real step: %zu in-vocab tokens, run-to-run byte-identical for "
           "a fixed seed (assembly correct; distribution preservation proven separately)\n", a.size());
    return 0;
}
