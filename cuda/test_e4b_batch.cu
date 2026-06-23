// test_e4b_batch.cu — validate continuous batching: B sequences of DIFFERENT
// lengths decoded together via seq_add/step_batch must produce the exact same
// tokens as each sequence decoded independently (single-sequence greedy).

#include <cstdio>
#include <vector>
#include <cstdint>
#include "e4b_engine.h"

static const char* kDir =
    "/opt/spark/models/hub/models--google--gemma-4-E4B-it/snapshots/"
    "fee6332c1abaafb77f6f9624236c63aa2f1d0187";

int main(int argc, char** argv){
    const char* dir=(argc>1)?argv[1]:kDir;
    e4b_engine_t* eng=e4b_engine_create(dir,4096,0);
    if(!eng){ fprintf(stderr,"FAIL create\n"); return 1; }

    // three prompts of different lengths (valid token-id subsequences)
    std::vector<std::vector<int32_t>> prompts = {
        {818,5279,529,7001,563},   // "The capital of France is"
        {818,5279,529},            // "The capital of"
        {818,5279},                // "The capital"
    };
    const int B=(int)prompts.size(), NGEN=8;

    printf("seq capacity: %d\n", e4b_engine_seq_capacity(eng));

    // reference: each sequence decoded independently (single-seq greedy, slot 0)
    std::vector<std::vector<int32_t>> ref(B, std::vector<int32_t>(NGEN));
    for (int i=0;i<B;i++)
        if (e4b_engine_generate_greedy(eng, prompts[i].data(), (int)prompts[i].size(),
                                       ref[i].data(), NGEN, nullptr, 0)!=NGEN){
            fprintf(stderr,"FAIL ref gen %d\n",i); e4b_engine_destroy(eng); return 1; }

    // batched: add all sequences, then step them together
    std::vector<int> slot(B); std::vector<int32_t> cur(B);
    std::vector<std::vector<int32_t>> got(B, std::vector<int32_t>(NGEN));
    for (int i=0;i<B;i++){
        slot[i]=e4b_engine_seq_add(eng, prompts[i].data(), (int)prompts[i].size(), &cur[i]);
        if (slot[i]<0){ fprintf(stderr,"FAIL seq_add %d\n",i); e4b_engine_destroy(eng); return 1; }
        got[i][0]=cur[i];
    }
    printf("added slots: "); for(int i=0;i<B;i++) printf("%d ",slot[i]); printf("\n");
    for (int step=1; step<NGEN; ++step){
        std::vector<int32_t> nxt(B);
        if (e4b_engine_step_batch(eng, slot.data(), cur.data(), B, nxt.data())!=0){
            fprintf(stderr,"FAIL step_batch\n"); e4b_engine_destroy(eng); return 1; }
        for (int i=0;i<B;i++){ got[i][step]=nxt[i]; cur[i]=nxt[i]; }
    }
    for (int i=0;i<B;i++) e4b_engine_seq_remove(eng, slot[i]);
    e4b_engine_destroy(eng);

    int mism=0;
    for (int i=0;i<B;i++){
        printf("seq %d (len %zu): batched=", i, prompts[i].size());
        for(int j=0;j<NGEN;j++) printf("%d ",got[i][j]);
        printf("| indep="); for(int j=0;j<NGEN;j++) printf("%d ",ref[i][j]);
        bool ok=true; for(int j=0;j<NGEN;j++) if(got[i][j]!=ref[i][j]){ok=false;mism++;}
        printf("%s\n", ok?"  ✓":"  <-- MISMATCH");
    }
    if (mism){ fprintf(stderr,"FAIL: %d mismatches batched vs independent\n",mism); return 1; }
    printf("PASS: continuous batching == independent decode (%d seqs × %d tokens)\n", B, NGEN);
    return 0;
}
