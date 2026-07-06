// test_e4b_spec_stream.cu — increment 5 continue-path losslessness check.
//
// The server drives e4b_engine_spec_stream (CONTINUE from a live KV the server already
// prefilled, h0 re-derived from the last history token). This must be byte-identical to
// plain greedy on the SAME prompt — exactly the contract the e4bServer relies on. We
// prefill with e4b_engine_prefill (capturing first_logits), then call spec_stream and
// compare its token ids to e4b_engine_generate_greedy.
#include <cstdio>
#include <vector>
#include "e4b_engine.h"

static const char* kGGUF =
    "/opt/spark/models/hub/models--google--gemma-4-E4B-it-qat-q4_0-gguf/snapshots/"
    "bb3b92e6f031fa438b409f898dd9f14f499a0cb0/gemma-4-E4B_q4_0-it.gguf";
static const char* kMTP =
    "/opt/spark/models/hub/models--unsloth--gemma-4-E4B-it-qat-GGUF/snapshots/"
    "bbcd9d849c2541ecc2af7ef64b3c3c2c7aa14e96/MTP/gemma-4-E4B-it-Q4_0-MTP.gguf";

static std::vector<int32_t> g_emitted;
static int emit_cb(int32_t tok, void*){ g_emitted.push_back(tok); return 0; }

int main(int argc, char** argv){
    const char* base = (argc>1)?argv[1]:kGGUF;
    const char* mtp  = (argc>2)?argv[2]:kMTP;
    const int MAX_NEW = 160;
    std::vector<int32_t> prompt = {2, 651, 2134, 603, 476, 4906, 576, 573, 1879, 235292};
    const int T = (int)prompt.size();

    e4b_engine_t* eng = e4b_engine_create(base, 4096, 1, 0);
    if(!eng){ fprintf(stderr,"FAIL: engine create\n"); return 1; }

    // baseline plain greedy
    std::vector<int32_t> base_out(MAX_NEW);
    int nb = e4b_engine_generate_greedy(eng, prompt.data(), T, base_out.data(), MAX_NEW, nullptr, 0);
    if(nb<128){ fprintf(stderr,"FAIL: baseline produced %d tokens\n", nb); return 1; }
    base_out.resize(nb);
    printf("baseline greedy: %d tokens\n", nb);

    if(e4b_engine_load_assistant(eng, mtp)!=0){ fprintf(stderr,"FAIL: load assistant\n"); return 1; }

    // server-style: prefill the prompt (fresh KV), capture first_logits, then spec_stream CONTINUE.
    int V = e4b_engine_vocab_size(eng);
    std::vector<float> first_logits(V);
    e4b_engine_reset(eng);
    if(e4b_engine_prefill(eng, prompt.data(), T, first_logits.data())!=0){ fprintf(stderr,"FAIL: prefill\n"); return 1; }

    std::vector<int32_t> spec_out(MAX_NEW);
    g_emitted.clear();
    int ns = e4b_engine_spec_stream(eng, prompt.data(), T, first_logits.data(),
                                    spec_out.data(), MAX_NEW, nullptr, 0, emit_cb, nullptr);
    if(ns<0){ fprintf(stderr,"FAIL: spec_stream rc=%d\n", ns); return 1; }
    spec_out.resize(ns);
    printf("spec_stream:     %d tokens (emitted via cb: %d)\n", ns, (int)g_emitted.size());

    // cb must have seen exactly the returned tokens, in order
    if((int)g_emitted.size()!=ns){ fprintf(stderr,"FAIL: cb saw %d, returned %d\n",(int)g_emitted.size(),ns); return 1; }
    for(int i=0;i<ns;i++) if(g_emitted[i]!=spec_out[i]){ fprintf(stderr,"FAIL: cb[%d]=%d != out=%d\n",i,g_emitted[i],spec_out[i]); return 1; }

    int ncmp = nb<ns?nb:ns;
    int div=-1; for(int i=0;i<ncmp;i++) if(base_out[i]!=spec_out[i]){div=i;break;}
    if(div<0 && nb!=ns) div=ncmp;
    if(div>=0){
        fprintf(stderr,"FAIL: spec_stream diverges from greedy at %d\n", div);
        int lo=div>4?div-4:0, hi=div+4<ncmp?div+4:ncmp;
        for(int i=lo;i<hi;i++) fprintf(stderr,"  %3d : %6d / %6d%s\n",i,base_out[i],spec_out[i],base_out[i]!=spec_out[i]?"  <-- DIFF":"");
        return 1;
    }
    printf("PASS: spec_stream CONTINUE path is BYTE-IDENTICAL to plain greedy over %d tokens\n", ncmp);
    e4b_engine_destroy(eng);
    return 0;
}
