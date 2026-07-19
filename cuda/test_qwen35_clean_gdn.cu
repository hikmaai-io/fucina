// ABOUTME: Byte/hash gate for exact clean-prefix Qwen3.5 GDN admission and continuation state.
// ABOUTME: Compares incumbent and clean dispatch for lengths 1..65 and M=1..32 in one engine.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "gemma4_kernels.cuh"

struct Result {
    std::vector<int32_t> first;
    std::vector<int32_t> continuation;
    std::vector<float> logits;
    std::vector<std::vector<uint8_t>> pre_state;
    std::vector<std::vector<uint8_t>> post_state;
};

static bool run_case(gemma4_engine_t *eng,const std::vector<int> &lens,bool clean,Result *out) {
    const int M=(int)lens.size();
    int total=0; for(int n:lens) total+=n;
    std::vector<int32_t> toks(total);
    int off=0;
    for(int s=0;s<M;s++) {
        for(int i=0;i<lens[s];i++)
            toks[off+i]=(int32_t)(100+((uint32_t)i*1103515245u+(uint32_t)s*2654435761u+17u)%30000u);
        off+=lens[s];
    }
    if(gemma4_engine_debug_set_q35_clean_gdn(eng,clean?1:0)!=0) return false;
    std::vector<int> slots(M); out->first.resize(M);
    int rc=gemma4_engine_seq_add_multiseq(eng,toks.data(),lens.data(),M,nullptr,nullptr,nullptr,
                                          nullptr,nullptr,slots.data(),out->first.data());
    if(rc!=M){ std::fprintf(stderr,"seq_add_multiseq rc=%d M=%d\n",rc,M); return false; }

    // nrows==1 debug_logits names the single-sequence buffer, so compare the multiseq head
    // directly only for M>1. M=1 is still covered by first token and complete state bytes.
    if(M>1) {
        out->logits.resize((size_t)M*248320);
        if(gemma4_engine_debug_logits(eng,out->logits.data(),M)!=248320) return false;
    }
    out->pre_state.resize(M);
    for(int s=0;s<M;s++) {
        size_t bytes=gemma4_engine_q35_state_size(eng,lens[s]);
        void *state=gemma4_host_alloc(bytes);
        if(!state || gemma4_engine_q35_state_save(eng,slots[s],state,lens[s])!=0) return false;
        out->pre_state[s].resize(bytes);
        std::memcpy(out->pre_state[s].data(),state,bytes); gemma4_host_free(state);
    }

    std::vector<int32_t> in=out->first,next(M);
    out->continuation.clear(); out->continuation.reserve((size_t)M*32);
    for(int step=0;step<32;step++) {
        if(gemma4_engine_step_batch(eng,slots.data(),in.data(),M,next.data())!=0) return false;
        out->continuation.insert(out->continuation.end(),next.begin(),next.end());
        in=next;
    }
    out->post_state.resize(M);
    for(int s=0;s<M;s++) {
        int nt=lens[s]+32;
        size_t bytes=gemma4_engine_q35_state_size(eng,nt);
        void *state=gemma4_host_alloc(bytes);
        if(!state || gemma4_engine_q35_state_save(eng,slots[s],state,nt)!=0) return false;
        out->post_state[s].resize(bytes);
        std::memcpy(out->post_state[s].data(),state,bytes); gemma4_host_free(state);
        gemma4_engine_seq_remove(eng,slots[s]);
    }
    return true;
}

static bool equal(const Result &a,const Result &b,const char *label) {
    if(a.first!=b.first || a.continuation!=b.continuation || a.pre_state!=b.pre_state ||
       a.post_state!=b.post_state || a.logits.size()!=b.logits.size() ||
       (!a.logits.empty() && std::memcmp(a.logits.data(),b.logits.data(),a.logits.size()*sizeof(float)))) {
        std::fprintf(stderr,"MISMATCH %s first=%d continuation=%d pre=%d post=%d logits=%d\n",label,
            a.first==b.first,a.continuation==b.continuation,a.pre_state==b.pre_state,
            a.post_state==b.post_state,a.logits.size()==b.logits.size() &&
            (a.logits.empty() || !std::memcmp(a.logits.data(),b.logits.data(),a.logits.size()*sizeof(float))));
        return false;
    }
    return true;
}

int main(int argc,char **argv) {
    if(argc<2){ std::fprintf(stderr,"usage: %s MODEL\n",argv[0]); return 2; }
    unsetenv("FUCINA_QWEN35_CLEAN_GDN");
    gemma4_engine_t *eng=gemma4_engine_create(argv[1],FORMAT_Q4_0,256,0,0.90);
    if(!eng){ std::fprintf(stderr,"engine_create failed\n"); return 1; }
    bool ok=true;
    { Result warm; ok=run_case(eng,std::vector<int>(1,3),false,&warm); } // capture/warm M=1 decode graph
    for(int n=1;n<=65 && ok;n++) {
        Result base,clean; std::vector<int> lens(1,n);
        ok=run_case(eng,lens,false,&base)&&run_case(eng,lens,true,&clean);
        char label[32]; std::snprintf(label,sizeof(label),"M1-L%d",n);
        if(ok) ok=equal(base,clean,label);
    }
    const int Ms[]={2,4,8,16,32};
    for(int M:Ms) if(ok) {
        std::vector<int> lens(M);
        int max_len=(M==32)?31:65; // stay under the 1024-row admission tile
        for(int i=0;i<M;i++) lens[i]=1+((i*17+M*7)%max_len);
        Result warm,base,clean;
        ok=run_case(eng,lens,false,&warm)&&run_case(eng,lens,false,&base)&&
           run_case(eng,lens,true,&clean);
        char label[32]; std::snprintf(label,sizeof(label),"M%d-mixed",M);
        if(ok) ok=equal(base,clean,label);
    }
    gemma4_engine_destroy(eng);
    std::printf("%s — clean GDN lengths 1..65, mixed M=1/2/4/8/16/32, logits/state/32-token continuation\n",
                ok?"PASS":"FAIL");
    return ok?0:1;
}
