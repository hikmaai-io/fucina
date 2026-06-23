// test_e4b_generate.cu — validate incremental decode (KV cache) against HF greedy.
// Reads /tmp/e4b_gen_ref.bin (prompt ids + HF greedy continuation), runs
// e4b_engine_generate_greedy, and requires an exact token-for-token match.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include "e4b_engine.h"

static const char* kDir =
    "/opt/spark/models/hub/models--google--gemma-4-E4B-it/snapshots/"
    "fee6332c1abaafb77f6f9624236c63aa2f1d0187";

static bool rd(FILE* f, int32_t* v){ return fread(v,4,1,f)==1; }

int main(int argc, char** argv){
    const char* dir=(argc>1)?argv[1]:kDir;
    const char* ref=(argc>2)?argv[2]:"/tmp/e4b_gen_ref.bin";
    FILE* f=fopen(ref,"rb"); if(!f){ fprintf(stderr,"open %s\n",ref); return 1; }
    int32_t np; if(!rd(f,&np)){ fprintf(stderr,"bad ref\n"); return 1; }
    std::vector<int32_t> prompt(np); for(int i=0;i<np;i++) if(!rd(f,&prompt[i])) return 1;
    int32_t ng; if(!rd(f,&ng)) return 1;
    std::vector<int32_t> expect(ng); for(int i=0;i<ng;i++) if(!rd(f,&expect[i])) return 1;
    fclose(f);

    e4b_engine_t* eng=e4b_engine_create(dir,4096,0);
    if(!eng){ fprintf(stderr,"FAIL create\n"); return 1; }

    std::vector<int32_t> got(ng);
    int n=e4b_engine_generate_greedy(eng, prompt.data(), np, got.data(), ng, nullptr, 0);
    printf("n_past after generate: %d (prompt %d + %d new)\n", e4b_engine_n_past(eng), np, ng);
    e4b_engine_destroy(eng);
    if(n!=ng){ fprintf(stderr,"FAIL: generated %d, expected %d\n",n,ng); return 1; }

    int mism=0;
    printf("  idx | mine | hf\n");
    for(int i=0;i<ng;i++){ bool m=got[i]!=expect[i]; mism+=m;
        if(i<12) printf("  %3d | %5d | %5d %s\n", i, got[i], expect[i], m?"  <-- MISMATCH":""); }
    if(mism){ fprintf(stderr,"FAIL: %d/%d tokens differ from HF greedy\n",mism,ng); return 1; }
    printf("PASS: incremental decode matches HF greedy (%d tokens exact)\n", ng);
    return 0;
}
