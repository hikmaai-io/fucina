// ABOUTME: DFlash acceptance + throughput measurement over multiple prompts (real numbers).
// ABOUTME: Asserts losslessness per prompt; reports MEASURED accept length and greedy-vs-DFlash time.
//
// For each prompt: (a) generate N tokens with plain greedy decode, timing it; (b) generate the same
// N tokens with the real DFlash loop, timing it; (c) assert the DFlash emitted stream is BYTE-
// IDENTICAL to greedy (losslessness); (d) accumulate verify steps + emitted tokens for a MEASURED
// mean accepted length. Prints per-prompt and aggregate numbers. Requires the FP8 target (arg1) and
// FUCINA_QWEN35_DFLASH_PATH. SKIPs if absent. Times are wall-clock on a flocked GPU; they are a
// coarse single-stream (B=1) indication, reported honestly (not a tuned benchmark).
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <sys/stat.h>
#include "gemma4_kernels.cuh"

static bool exists(const std::string& p){ struct stat s; return stat(p.c_str(),&s)==0; }
static int step1(gemma4_engine_t* eng,int slot,int32_t in){ int s[1]={slot}; int32_t i[1]={in},o[1]={-1}; if(gemma4_engine_step_batch(eng,s,i,1,o)!=0){ printf("step fail\n"); exit(2);} return o[0]; }
static double now_s(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc,char**argv){
    const char* path=(argc>1)?argv[1]:"/opt/spark/models/models--Qwen--Qwen3.5-9B-FP8";
    const char* dpath=getenv("FUCINA_QWEN35_DFLASH_PATH");
    if(!exists(std::string(path)+"/config.json")){ printf("SKIP — target absent\n"); return 0; }
    if(!dpath||!exists(std::string(dpath)+"/model.safetensors")){ printf("SKIP — draft path unset/absent\n"); return 0; }
    gemma4_engine_t* eng=gemma4_engine_create(path,FORMAT_Q4_0,4096,0,0.90);
    if(!eng){ printf("FAIL create\n"); return 2; }
    if(gemma4_engine_q35_dflash_load(eng)!=0){ printf("FAIL draft load\n"); return 1; }

    // A few varied prompts (in-vocab token id sequences). N generated tokens each.
    std::vector<std::vector<int32_t>> prompts = {
        {760,6511,314,9338,369},                 // "The capital of France is"
        {785,6722,315,9625,374,1024,2048,4096},
        {100,200,300,400,500,600,700,800,900,1000},
    };
    const int N=48;
    long agg_steps=0, agg_emitted=0; int fails=0; double greedy_t=0, dflash_t=0;

    for(size_t pi=0; pi<prompts.size(); pi++){
        auto& pr=prompts[pi]; int NP=pr.size();
        // greedy reference (timed)
        int32_t first=-1; int slot=gemma4_engine_seq_add(eng,pr.data(),NP,&first,0.f,0,1.f,0.f,1);
        if(slot<0){ printf("FAIL seq_add\n"); return 2; }
        // small warmup to a non-trivial state
        int32_t cur=first; for(int i=0;i<8;i++) cur=step1(eng,slot,cur);
        int32_t bonus0=cur;
        std::vector<int32_t> ref; ref.reserve(N+1);
        double t0=now_s(); { int32_t c=cur; for(int i=0;i<N+1;i++){ ref.push_back(c); c=step1(eng,slot,c); } } greedy_t+=now_s()-t0;
        gemma4_engine_seq_remove(eng,slot);

        // DFlash run (timed)
        gemma4_engine_q35_dflash_reset_context(eng);
        int32_t re=-1; int slot2=gemma4_engine_seq_add(eng,pr.data(),NP,&re,0.f,0,1.f,0.f,1);
        if(slot2<0){ printf("FAIL seq_add2\n"); return 2; }
        { int32_t c=re; for(int i=0;i<8;i++) c=step1(eng,slot2,c); }
        std::vector<int32_t> emitted; int32_t bonus=bonus0; int steps=0;
        double t1=now_s();
        while((int)emitted.size()<N && steps<N+8){
            int32_t out[64]; int n=0; int32_t nb=0;
            if(gemma4_engine_q35_dflash_real_step(eng,slot2,bonus,out,&n,&nb)!=0){ printf("FAIL real_step\n"); return 1; }
            for(int i=0;i<n;i++) emitted.push_back(out[i]);
            bonus=nb; steps++;
        }
        dflash_t+=now_s()-t1;
        gemma4_engine_seq_remove(eng,slot2);

        int pf=0; for(int i=0;i<N;i++) if(emitted[i]!=ref[i+1]){ if(pf<3) printf("  P%zu MISMATCH @%d: %d vs %d\n",pi,i,emitted[i],ref[i+1]); pf++; }
        fails+=pf;
        agg_steps+=steps; agg_emitted+=(int)emitted.size();
        printf("  prompt %zu: %s, %d steps, %d emitted, mean accept=%.2f/step\n",
               pi, pf? "LOSSLESS-FAIL":"lossless-OK", steps, (int)emitted.size(),
               steps? (double)emitted.size()/steps : 0.0);
    }
    gemma4_engine_destroy(eng);

    double mean_accept = agg_steps? (double)agg_emitted/agg_steps : 0.0;
    printf("AGGREGATE: %ld verify steps, %ld emitted tokens, MEASURED mean emitted/step = %.3f "
           "(accepted drafts/step = %.3f)\n", agg_steps, agg_emitted, mean_accept, mean_accept-1.0);
    printf("WALL-CLOCK (B=1, single-stream, coarse): greedy %.3fs, DFlash %.3fs over %zu prompts x %d tokens\n",
           greedy_t, dflash_t, prompts.size(), N);
    if(fails){ printf("FAIL — DFlash measurement: %d lossless mismatches (losslessness broken)\n",fails); return 1; }
    printf("PASS — DFlash measurement: emitted BYTE-IDENTICAL to greedy on all prompts (lossless); "
           "MEASURED mean emitted/step = %.3f. Numbers are real single-stream measurements.\n", mean_accept);
    return 0;
}
