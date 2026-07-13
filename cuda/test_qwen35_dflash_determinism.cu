// ABOUTME: Gate: the greedy DFlash real serving step is run-to-run byte-identical (determinism).
// ABOUTME: Two independent runs over the same prompt must emit the exact same token stream.
//
// P5 requires repeated-run determinism. The measure gate proves losslessness (emitted == plain
// greedy decode); this gate proves the DFlash greedy serving path itself is reproducible: two fresh
// runs (fresh sequences, fresh drafter context) over the same prompt emit byte-identical streams AND
// the same per-step accepted counts. Any hidden nondeterminism (uninitialized scratch, races in the
// draft/verify/commit kernels, stale GDN snapshot) would surface here. Requires the FP8 target
// (arg1) + FUCINA_QWEN35_DFLASH_PATH. SKIPs if absent.
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <sys/stat.h>
#include "gemma4_kernels.cuh"

static bool exists(const std::string& p){ struct stat s; return stat(p.c_str(),&s)==0; }
static int step1(gemma4_engine_t* e,int s,int32_t in){int sl[1]={s};int32_t i[1]={in},o[1]={-1};gemma4_engine_step_batch(e,sl,i,1,o);return o[0];}

// One full greedy DFlash run over the prompt; fills `emitted` and `accepts` (per-step accepted len).
static bool run(gemma4_engine_t* e, const int32_t* pr, int NP, int N,
                std::vector<int32_t>& emitted, std::vector<int>& accepts){
    gemma4_engine_q35_dflash_reset_context(e);
    int32_t f=-1; int s=gemma4_engine_seq_add(e,pr,NP,&f,0.f,0,1.f,0.f,1);
    if(s<0) return false;
    int32_t cur=f; for(int i=0;i<8;i++) cur=step1(e,s,cur);
    int32_t bonus=cur; int steps=0;
    while((int)emitted.size()<N && steps<N+8){
        int32_t out[64]; int n=0; int32_t nb=0;
        if(gemma4_engine_q35_dflash_real_step(e,s,bonus,out,&n,&nb)!=0){ gemma4_engine_seq_remove(e,s); return false; }
        for(int i=0;i<n;i++) emitted.push_back(out[i]);
        accepts.push_back(n-1);   // accepted drafts this step (emitted = accepted + 1 correction)
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

    int32_t pr[6]={760,6511,314,9338,369,785}; const int NP=6, N=40;
    std::vector<int32_t> a,b; std::vector<int> aa,ba;
    if(!run(e,pr,NP,N,a,aa)){ printf("FAIL run a\n"); return 1; }
    if(!run(e,pr,NP,N,b,ba)){ printf("FAIL run b\n"); return 1; }
    gemma4_engine_destroy(e);

    if(a.size()!=b.size()){ printf("FAIL: emitted size %zu vs %zu\n",a.size(),b.size()); return 1; }
    int fails=0;
    for(size_t i=0;i<a.size();i++) if(a[i]!=b[i]){ if(fails<3) printf("NONDET token @%zu: %d vs %d\n",i,a[i],b[i]); fails++; }
    if(aa.size()!=ba.size()){ printf("FAIL: step count %zu vs %zu\n",aa.size(),ba.size()); return 1; }
    for(size_t i=0;i<aa.size();i++) if(aa[i]!=ba[i]){ if(fails<6) printf("NONDET accept @%zu: %d vs %d\n",i,aa[i],ba[i]); fails++; }

    if(fails){ printf("FAIL — greedy DFlash real step NOT run-to-run deterministic (%d)\n",fails); return 1; }
    printf("PASS — greedy DFlash real step run-to-run DETERMINISTIC: %zu emitted tokens + %zu per-step "
           "accept counts byte-identical across two independent runs\n", a.size(), aa.size());
    return 0;
}
