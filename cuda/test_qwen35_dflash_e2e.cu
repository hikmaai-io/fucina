// ABOUTME: End-to-end greedy DFlash losslessness gate: emitted stream == plain greedy decode.
// ABOUTME: Runs DFlash steps with arbitrary/adversarial drafts on the real target; asserts identity.
//
// The losslessness contract: greedy DFlash must produce EXACTLY the token stream that plain greedy
// (non-speculative) decode produces, regardless of draft content (wrong drafts only lower the accept
// rate). This gate:
//   1. records N tokens of plain greedy decode from a warm prefix (the reference);
//   2. from an identical rebuilt prefix, runs DFlash greedy steps until >= N tokens are emitted,
//      feeding intentionally-varied drafts (some correct-ish, some adversarial/wrong);
//   3. asserts the first N emitted tokens are byte-identical to the reference, and reports the
//      realized mean acceptance length (emitted / verify-steps) as a MEASURED number.
// Uses the real target GGUF (arg1). This is the S1a end-to-end losslessness proof.
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
    if(!exists(path)){ printf("SKIP — target checkpoint absent\n"); return 0; }
    gemma4_engine_t* eng=gemma4_engine_create(path,FORMAT_Q4_0,4096,0,0.90);
    if(!eng){ printf("FAIL create\n"); return 2; }

    int32_t prompt[5]={760,6511,314,9338,369}; const int NP=5, WARM=16, K=6, N=40;

    // (1) plain greedy reference.
    int32_t first=-1; int slot=gemma4_engine_seq_add(eng,prompt,NP,&first,0.f,0,1.f,0.f,1);
    if(slot<0){ printf("FAIL seq_add\n"); return 2; }
    int32_t cur=first; for(int i=0;i<WARM;i++) cur=step1(eng,slot,cur);
    // cur is the bonus (next token to decode) at the boundary.
    int32_t bonus0=cur;
    // ref[0] = bonus0 (already decided), ref[1..] = the greedy continuation after bonus. DFlash
    // emits the tokens AFTER bonus, so it is compared against ref[1..].
    std::vector<int32_t> ref; ref.reserve(N+1);
    { int32_t c=cur; for(int i=0;i<N+1;i++){ ref.push_back(c); c=step1(eng,slot,c); } }
    gemma4_engine_seq_remove(eng,slot);

    // (2) DFlash greedy run from an identical rebuilt prefix, with varied/adversarial drafts.
    int32_t re=-1; int slot2=gemma4_engine_seq_add(eng,prompt,NP,&re,0.f,0,1.f,0.f,1);
    if(slot2<0){ printf("FAIL seq_add2\n"); return 2; }
    { int32_t c=re; for(int i=0;i<WARM;i++) c=step1(eng,slot2,c); }

    std::vector<int32_t> emitted; emitted.reserve(N+K);
    int32_t bonus=bonus0; int steps=0; long accepted_sum=0;
    // Draft strategy: alternate between the reference continuation (high accept) and adversarial
    // fixed tokens (zero accept) so the gate exercises j from 0..K. We use the reference tokens we
    // ALREADY know for the "good" drafts (a real drafter would approximate these).
    while((int)emitted.size() < N){
        int32_t draft[16];
        int already=emitted.size();
        bool good = (steps % 2 == 0);
        for(int i=0;i<K;i++){
            // emitted[already..] aligns to ref[already+1..]; a good draft guesses those greedy tokens.
            if(good && already+1+i < (int)ref.size()) draft[i]=ref[already+1+i];
            else draft[i]=(int32_t)((steps*7+i*13+1)%1000);                       // adversarial
        }
        int32_t out[16]; int n=0; int32_t nb=0;
        if(gemma4_engine_q35_dflash_greedy_step(eng,slot2,bonus,draft,K,out,&n,&nb)!=0){ printf("FAIL greedy_step\n"); return 1; }
        for(int i=0;i<n;i++) emitted.push_back(out[i]);
        accepted_sum += (n-1);   // n = accepted drafts + 1 correction; accepted = n-1
        bonus=nb; steps++;
        if(steps>N+5) break;
    }

    int fails=0;
    for(int i=0;i<N;i++){ if(emitted[i]!=ref[i+1]){ printf("FAIL @ %d: dflash=%d greedy=%d\n",i,emitted[i],ref[i+1]); fails++; if(fails>5) break; } }
    double mean_accept = steps ? (double)(accepted_sum + steps)/steps : 0.0;   // (accepted+1 correction)/step = emitted/step
    gemma4_engine_seq_remove(eng,slot2); gemma4_engine_destroy(eng);

    if(fails){ printf("FAIL — DFlash end-to-end greedy losslessness (%d mismatches in first %d)\n",fails,N); return 1; }
    printf("PASS — DFlash end-to-end greedy: first %d emitted tokens byte-identical to plain greedy "
           "decode across %d verify steps with mixed/adversarial drafts; MEASURED mean emitted/step "
           "= %.3f (losslessness proven; acceptance is workload/draft-quality dependent)\n",
           N, steps, mean_accept);
    return 0;
}
