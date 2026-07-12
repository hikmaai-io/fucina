// ABOUTME: End-to-end gate with the REAL draft model: emitted stream == plain greedy decode.
// ABOUTME: Drives gemma4_engine_q35_dflash_real_step and asserts losslessness + reports accept rate.
//
// This is the full-loop S1a validation with the ACTUAL resident draft model (not synthetic drafts):
//   1. plain greedy reference from a warm prefix;
//   2. from an identical rebuilt prefix, run real DFlash steps (draft with the loaded draft model
//      over the growing context, verify+accept+commit) until >= N tokens emitted;
//   3. assert the first N emitted tokens are byte-identical to greedy (losslessness), and report the
//      MEASURED mean accepted length with the real drafter.
// Requires the target GGUF (arg1) and FUCINA_QWEN35_DFLASH_PATH. SKIPs if either absent.
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <sys/stat.h>
#include "gemma4_kernels.cuh"

static bool exists(const std::string& p){ struct stat s; return stat(p.c_str(),&s)==0; }
static int step1(gemma4_engine_t* eng,int slot,int32_t in){ int s[1]={slot}; int32_t i[1]={in},o[1]={-1}; if(gemma4_engine_step_batch(eng,s,i,1,o)!=0){ printf("step fail\n"); exit(2);} return o[0]; }

int main(int argc,char**argv){
    const char* path=(argc>1)?argv[1]:"/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf";
    const char* dpath=getenv("FUCINA_QWEN35_DFLASH_PATH");
    if(!exists(path)){ printf("SKIP — target absent\n"); return 0; }
    if(!dpath||!exists(std::string(dpath)+"/model.safetensors")){ printf("SKIP — draft path unset/absent\n"); return 0; }
    gemma4_engine_t* eng=gemma4_engine_create(path,FORMAT_Q4_0,4096,0,0.90);
    if(!eng){ printf("FAIL create\n"); return 2; }
    if(gemma4_engine_q35_dflash_load(eng)!=0){ printf("FAIL draft load\n"); return 1; }

    int32_t prompt[5]={760,6511,314,9338,369}; const int NP=5, WARM=16, N=32;

    // (1) plain greedy reference.
    int32_t first=-1; int slot=gemma4_engine_seq_add(eng,prompt,NP,&first,0.f,0,1.f,0.f,1);
    if(slot<0){ printf("FAIL seq_add\n"); return 2; }
    int32_t cur=first; for(int i=0;i<WARM;i++) cur=step1(eng,slot,cur);
    int32_t bonus0=cur;
    std::vector<int32_t> ref; ref.reserve(N+1);
    { int32_t c=cur; for(int i=0;i<N+1;i++){ ref.push_back(c); c=step1(eng,slot,c); } }
    gemma4_engine_seq_remove(eng,slot);

    // (2) real DFlash run from identical prefix.
    gemma4_engine_q35_dflash_reset_context(eng);
    int32_t re=-1; int slot2=gemma4_engine_seq_add(eng,prompt,NP,&re,0.f,0,1.f,0.f,1);
    if(slot2<0){ printf("FAIL seq_add2\n"); return 2; }
    { int32_t c=re; for(int i=0;i<WARM;i++) c=step1(eng,slot2,c); }

    std::vector<int32_t> emitted; int32_t bonus=bonus0; int steps=0; long accepted_sum=0;
    while((int)emitted.size() < N && steps < N+8){
        int32_t out[64]; int n=0; int32_t nb=0;
        if(gemma4_engine_q35_dflash_real_step(eng,slot2,bonus,out,&n,&nb)!=0){ printf("FAIL real_step\n"); return 1; }
        for(int i=0;i<n;i++) emitted.push_back(out[i]);
        accepted_sum += (n-1); bonus=nb; steps++;
    }

    int fails=0;
    for(int i=0;i<N;i++){ if(emitted[i]!=ref[i+1]){ printf("FAIL @ %d: dflash=%d greedy=%d\n",i,emitted[i],ref[i+1]); fails++; if(fails>5) break; } }
    double mean_accept = steps ? (double)accepted_sum/steps : 0.0;
    double mean_emit = steps ? (double)emitted.size()/steps : 0.0;
    gemma4_engine_seq_remove(eng,slot2); gemma4_engine_destroy(eng);

    if(fails){ printf("FAIL — real DFlash end-to-end losslessness (%d mismatches)\n",fails); return 1; }
    printf("PASS — real DFlash end-to-end: first %d emitted tokens BYTE-IDENTICAL to plain greedy "
           "decode over %d steps with the RESIDENT draft model; MEASURED mean accepted drafts/step "
           "= %.3f, mean emitted/step = %.3f (losslessness proven; acceptance is a real measurement)\n",
           N, steps, mean_accept, mean_emit);
    return 0;
}
