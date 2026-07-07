// test_e4b_spec.cu — E4B MTP increments 3+4 DECISIVE GATE.
//
// Greedy speculative decode (draft head) must be LOSSLESS: for the SAME prompt, the token
// ids produced by e4b_engine_generate_spec_greedy (assistant loaded) must be BYTE-IDENTICAL
// to e4b_engine_generate_greedy (plain greedy baseline). This compares >=128 tokens and
// asserts exact equality, printing the first divergent index on any mismatch + tok/s of each.
// See docs/e4b-mtp-plan.md §"Decisive correctness gate".
#include <cstdio>
#include <vector>
#include <chrono>
#include "e4b_engine.h"

static const char* kGGUF =
    "/opt/spark/models/hub/models--google--gemma-4-E4B-it-qat-q4_0-gguf/snapshots/"
    "bb3b92e6f031fa438b409f898dd9f14f499a0cb0/gemma-4-E4B_q4_0-it.gguf";
static const char* kMTP =
    "/opt/spark/models/hub/models--unsloth--gemma-4-E4B-it-qat-GGUF/snapshots/"
    "bbcd9d849c2541ecc2af7ef64b3c3c2c7aa14e96/MTP/gemma-4-E4B-it-Q4_0-MTP.gguf";

int main(int argc, char** argv){
    const char* base = (argc>1)?argv[1]:kGGUF;
    const char* mtp  = (argc>2)?argv[2]:kMTP;

    const int MAX_NEW = 160;   // >= 128 as required
    // A fixed, reproducible prompt (BOS=2 + a few common ids). Greedy is deterministic.
    std::vector<int32_t> prompt = {2, 651, 2134, 603, 476, 4906, 576, 573, 1879, 235292};
    const int T = (int)prompt.size();

    e4b_engine_t* eng = e4b_engine_create(base, 4096, 1, 0);
    if(!eng){ fprintf(stderr,"FAIL: engine create\n"); return 1; }

    // ── baseline: plain greedy (no assistant needed) ──
    std::vector<int32_t> base_out(MAX_NEW);
    auto t0 = std::chrono::high_resolution_clock::now();
    int nb = e4b_engine_generate_greedy(eng, prompt.data(), T, base_out.data(), MAX_NEW, nullptr, 0);
    auto t1 = std::chrono::high_resolution_clock::now();
    if (nb < 0){ fprintf(stderr,"FAIL: baseline greedy rc=%d\n", nb); e4b_engine_destroy(eng); return 1; }
    double base_s = std::chrono::duration<double>(t1-t0).count();
    base_out.resize(nb);
    printf("baseline greedy: %d tokens in %.3fs = %.1f tok/s\n", nb, base_s, nb/base_s);

    // ── speculative: load the assistant, run greedy spec on the SAME prompt ──
    int rc = e4b_engine_load_assistant(eng, mtp);
    if (rc!=0){ fprintf(stderr,"FAIL: load_assistant rc=%d\n", rc); e4b_engine_destroy(eng); return 1; }
    if (!e4b_engine_has_assistant(eng)){ fprintf(stderr,"FAIL: has_assistant false after load\n"); e4b_engine_destroy(eng); return 1; }

    std::vector<int32_t> spec_out(MAX_NEW);
    auto t2 = std::chrono::high_resolution_clock::now();
    int ns = e4b_engine_generate_spec_greedy(eng, prompt.data(), T, spec_out.data(), MAX_NEW, nullptr, 0);
    auto t3 = std::chrono::high_resolution_clock::now();
    if (ns < 0){ fprintf(stderr,"FAIL: spec greedy rc=%d\n", ns); e4b_engine_destroy(eng); return 1; }
    double spec_s = std::chrono::duration<double>(t3-t2).count();
    spec_out.resize(ns);
    printf("spec greedy:     %d tokens in %.3fs = %.1f tok/s  (%.2fx)\n",
           ns, spec_s, ns/spec_s, base_s>0 ? (base_s/spec_s):0.0);

    // ── DECISIVE GATE: byte-identical token ids ──
    if (nb < 128 || ns < 128){
        fprintf(stderr,"FAIL: produced fewer than 128 tokens (baseline=%d spec=%d)\n", nb, ns);
        e4b_engine_destroy(eng); return 1;
    }
    int ncmp = nb < ns ? nb : ns;
    int first_div = -1;
    for (int i=0;i<ncmp;i++){ if(base_out[i]!=spec_out[i]){ first_div=i; break; } }
    if (first_div<0 && nb!=ns) first_div = ncmp;   // one is a prefix of the other

    if (first_div >= 0){
        fprintf(stderr,"FAIL: token streams diverge at index %d", first_div);
        if (first_div<ncmp) fprintf(stderr," (baseline=%d spec=%d)\n", base_out[first_div], spec_out[first_div]);
        else fprintf(stderr," (length mismatch baseline=%d spec=%d)\n", nb, ns);
        // dump a small window for debugging
        int lo = first_div>4?first_div-4:0, hi = first_div+4<ncmp?first_div+4:ncmp;
        fprintf(stderr,"  idx : baseline / spec\n");
        for (int i=lo;i<hi;i++) fprintf(stderr,"  %3d : %6d / %6d%s\n", i, base_out[i], spec_out[i],
                                        base_out[i]!=spec_out[i]?"  <-- DIFF":"");
        e4b_engine_destroy(eng); return 1;
    }

    printf("PASS: greedy spec is BYTE-IDENTICAL to plain greedy over %d tokens (lossless)\n", ncmp);
    e4b_engine_destroy(eng);
    return 0;
}
