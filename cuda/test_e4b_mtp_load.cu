// test_e4b_mtp_load.cu — E4B MTP draft head.
// Increment 1: load the gemma4-assistant draft head into an E4B engine, verify residency.
// Increment 2: prefill a short prompt, grab the real recurrent h (post-final-norm last
// row) + the last token, run ONE drafter forward, assert the drafted id is in-range and
// finite (no crash / NaN), and print it. See docs/e4b-mtp-plan.md.
#include <cstdio>
#include <cmath>
#include <vector>
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
    e4b_engine_t* eng = e4b_engine_create(base, 4096, 1, 0);
    if(!eng){ fprintf(stderr,"FAIL: engine create\n"); return 1; }
    if (e4b_engine_has_assistant(eng)){ fprintf(stderr,"FAIL: assistant reported before load\n"); return 1; }
    int rc = e4b_engine_load_assistant(eng, mtp);
    if (rc!=0){ fprintf(stderr,"FAIL: load_assistant rc=%d\n", rc); e4b_engine_destroy(eng); return 1; }
    if (!e4b_engine_has_assistant(eng)){ fprintf(stderr,"FAIL: has_assistant false after load\n"); e4b_engine_destroy(eng); return 1; }
    printf("PASS: E4B assistant draft head loaded + resident\n");

    // ── Increment 2: one drafter forward from a real recurrent h ──
    const int H = e4b_engine_hidden_size(eng);
    const int V = e4b_engine_vocab_size(eng);
    // A short, fixed token prompt (BOS=2 + a few common ids). Exact ids don't matter for the
    // crash/range/NaN gate; we just need a real prefill so fin_out is the true target hidden.
    std::vector<int32_t> prompt = {2, 651, 2134, 603, 476};
    const int T = (int)prompt.size();

    std::vector<float> fin(  (size_t)T*H );
    std::vector<float> lastlogits(V);
    if (e4b_engine_forward_debug(eng, prompt.data(), T, nullptr, nullptr, fin.data(), lastlogits.data())!=0){
        fprintf(stderr,"FAIL: forward_debug\n"); e4b_engine_destroy(eng); return 1; }

    // recurrent h = post-final-norm hidden of the LAST token; tok = last prompt token;
    // pos = absolute RoPE position of the draft point = T-1 (n_past after prefill is T, the
    // drafter predicts the token AT position pos using the hidden of the token at pos).
    std::vector<float> h(fin.begin() + (size_t)(T-1)*H, fin.begin() + (size_t)T*H);
    int32_t tok = prompt[T-1];
    int pos = T - 1;

    // sanity: h must be finite
    for (int i=0;i<H;i++) if(!std::isfinite(h[i])){ fprintf(stderr,"FAIL: recurrent h has non-finite at %d\n", i); e4b_engine_destroy(eng); return 1; }

    int32_t draft = -1;
    if (e4b_engine_mtp_forward_debug(eng, tok, pos, h.data(), &draft)!=0){
        fprintf(stderr,"FAIL: mtp_forward_debug\n"); e4b_engine_destroy(eng); return 1; }
    if (draft < 0 || draft >= V){
        fprintf(stderr,"FAIL: drafted id %d out of range [0,%d)\n", draft, V); e4b_engine_destroy(eng); return 1; }
    // next recurrent h written back must be finite
    int bad=0; for(int i=0;i<H;i++) if(!std::isfinite(h[i])) bad++;
    if (bad){ fprintf(stderr,"FAIL: next recurrent h has %d non-finite entries\n", bad); e4b_engine_destroy(eng); return 1; }

    printf("PASS: drafter forward — tok=%d pos=%d -> drafted_id=%d (in range [0,%d))\n",
           tok, pos, draft, V);
    printf("DRAFTED_ID=%d\n", draft);

    e4b_engine_destroy(eng);
    return 0;
}
