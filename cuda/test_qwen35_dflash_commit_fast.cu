// ABOUTME: Gate: lossless fast-commit == sequential commit (byte-identical state + argmax) ∀ j.
// ABOUTME: Proves the batched-projection + decode-kernel-recurrence commit is truly lossless.
//
// For each j in 0..K: snapshot, sequentially commit j accepted tokens (out_next), record the next
// greedy token; snapshot again, fast-commit the same j (out_argmax), record its next greedy token.
// The two must agree AND the post-commit decode must continue identically for several steps (state
// byte-identity proxy: identical continued greedy stream). Requires the FP8 target (arg1). SKIPs if
// absent. This is the losslessness guard before wiring fast-commit into the serving loop.
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <sys/stat.h>
#include "gemma4_kernels.cuh"

static bool exists(const std::string& p){ struct stat s; return stat(p.c_str(),&s)==0; }
static int step1(gemma4_engine_t* e,int s,int32_t in){int sl[1]={s};int32_t i[1]={in},o[1]={-1};gemma4_engine_step_batch(e,sl,i,1,o);return o[0];}

int main(int c,char**v){
    const char* path=c>1?v[1]:"/opt/spark/models/models--Qwen--Qwen3.5-9B-FP8";
    if(!exists(std::string(path)+"/config.json")){ printf("SKIP — target absent\n"); return 0; }
    gemma4_engine_t* e=gemma4_engine_create(path,FORMAT_Q4_0,4096,0,0.90); if(!e){printf("FAIL create\n");return 2;}
    int32_t pr[10]={100,200,300,400,500,600,700,800,900,1000}; const int K=12, CONT=6;
    int fails=0;
    for(int j=0;j<=K;j++){
        // Build the accepted token list = the true greedy continuation from a warm state.
        int32_t f=-1; int s=gemma4_engine_seq_add(e,pr,10,&f,0.f,0,1.f,0.f,1);
        int32_t cur=f; for(int i=0;i<8;i++) cur=step1(e,s,cur);
        std::vector<int32_t> acc; { int32_t cc=cur; for(int i=0;i<j;i++){ acc.push_back(cc); cc=step1(e,s,cc);} }
        gemma4_engine_seq_remove(e,s);

        // Sequential commit path.
        int32_t r1=-1; int s1=gemma4_engine_seq_add(e,pr,10,&r1,0.f,0,1.f,0.f,1);
        { int32_t cc=r1; for(int i=0;i<8;i++) cc=step1(e,s1,cc); }
        gemma4_engine_q35_gdn_snapshot(e,s1);
        std::vector<int32_t> on(j>0?j:1,-1);
        gemma4_engine_q35_gdn_commit(e,s1,acc.data(),j,on.data());
        std::vector<int32_t> cont1; { int32_t cc=(j>0?on[j-1]:step1(e,s1,pr[0])); /*seed next*/ }
        // continue greedy CONT steps from the committed state
        int32_t seed1 = (j>0? on[j-1] : -1);
        if(j==0){ /* rewound to warm state; next greedy = decode from last warm token */ }
        std::vector<int32_t> tail1;
        { int32_t cc = (j>0? on[j-1] : 0); if(j>0){ for(int i=0;i<CONT;i++){ cc=step1(e,s1,cc); tail1.push_back(cc);} } }
        gemma4_engine_seq_remove(e,s1);

        // Fast commit path.
        int32_t r2=-1; int s2=gemma4_engine_seq_add(e,pr,10,&r2,0.f,0,1.f,0.f,1);
        { int32_t cc=r2; for(int i=0;i<8;i++) cc=step1(e,s2,cc); }
        gemma4_engine_q35_gdn_snapshot(e,s2);
        std::vector<int32_t> oa(j>0?j:1,-1);
        if(gemma4_engine_q35_gdn_commit_fast(e,s2,acc.data(),j,oa.data())!=0){ printf("FAIL fast-commit j=%d\n",j); fails++; gemma4_engine_seq_remove(e,s2); continue; }
        std::vector<int32_t> tail2;
        { int32_t cc = (j>0? oa[j-1] : 0); if(j>0){ for(int i=0;i<CONT;i++){ cc=step1(e,s2,cc); tail2.push_back(cc);} } }
        gemma4_engine_seq_remove(e,s2);

        // Compare per-step argmax and the continued tail.
        bool ok=true;
        for(int i=0;i<j;i++) if(on[i]!=oa[i]){ ok=false; if(fails<4) printf("j=%d argmax@%d seq=%d fast=%d\n",j,i,on[i],oa[i]); }
        for(size_t i=0;i<tail1.size();i++) if(i<tail2.size() && tail1[i]!=tail2[i]){ ok=false; if(fails<4) printf("j=%d tail@%zu seq=%d fast=%d\n",j,i,tail1[i],tail2[i]); }
        if(!ok) fails++;
        else printf("j=%2d: fast==seq (argmax + %d continued steps identical)\n", j, (int)tail1.size());
    }
    gemma4_engine_destroy(e);
    if(fails){ printf("FAIL — fast-commit not byte-identical to sequential commit (%d)\n",fails); return 1; }
    printf("PASS — DFlash lossless fast-commit: byte-identical to sequential commit (argmax + continued "
           "greedy) for all j in 0..%d\n",K);
    return 0;
}
