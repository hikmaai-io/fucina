// ABOUTME: GPU gate proving the GB10 four-row Q8 head emits the incumbent greedy stream exactly.
// ABOUTME: Runs both kernels through the production B=1 graph path on one Qwen3.6 MoE engine.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include "gemma4_kernels.cuh"

static int run(gemma4_engine_t *eng, int rows, int32_t *out, int n) {
    if (gemma4_engine_debug_set_q35_head_rows(eng, rows) != 0) return -1;
    const int32_t prompt[] = {760,6511,314,9338,369};
    int32_t tok=-1;
    int slot=gemma4_engine_seq_add(eng,prompt,5,&tok,0.f,0,0.f,0.f,1);
    if(slot<0) return -1;
    for(int i=0;i<n;i++) {
        out[i]=tok;
        int sl=slot, next=-1;
        if(gemma4_engine_step_batch(eng,&sl,&tok,1,&next)!=0) {
            gemma4_engine_seq_remove(eng,slot); return -1;
        }
        tok=next;
    }
    gemma4_engine_seq_remove(eng,slot);
    return 0;
}

int main(int argc,char **argv) {
    const char *model=argc>1?argv[1]:
        "/opt/spark/models/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/0b2752837483aa34b3db6e83e151b150c0e00e49";
    gemma4_engine_t *eng=gemma4_engine_create(model,FORMAT_Q4_0,4096,0,0.90);
    if(!eng) { fprintf(stderr,"create failed\n"); return 2; }
    if(gemma4_engine_n_experts(eng)<=0) {
        fprintf(stderr,"gate requires sparse Qwen3.6 MoE\n");
        gemma4_engine_destroy(eng); return 2;
    }
    if(gemma4_engine_debug_set_q35_head_rows(eng,3)==0) {
        fprintf(stderr,"invalid rows accepted\n");
        gemma4_engine_destroy(eng); return 1;
    }
    constexpr int N=128;
    int32_t legacy[N],blocked[N];
    if(run(eng,1,legacy,N)!=0 || run(eng,4,blocked,N)!=0) {
        fprintf(stderr,"decode failed\n");
        gemma4_engine_destroy(eng); return 2;
    }
    int first=-1;
    for(int i=0;i<N;i++) if(legacy[i]!=blocked[i]) { first=i; break; }
    gemma4_engine_destroy(eng);
    if(first>=0) {
        fprintf(stderr,"FAIL: rows/warp 1 vs 4 first greedy mismatch at %d: %d != %d\n",
                first,legacy[first],blocked[first]);
        return 1;
    }
    printf("PASS — Qwen3.6 MoE exact Q8 head rows/warp 1 == 4 for %d production graph tokens\n",N);
    return 0;
}
