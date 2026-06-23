// test_e4b_bench.cu — E4B prefill + decode throughput baseline (BF16 weights).
// Not a correctness test (see test_e4b_generate.cu for that): it measures pure
// engine tok/s so we have real E4B numbers to compare against the dense engine.
//
// Usage: e4b_bench [model_dir] [prefill_tokens] [decode_tokens]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <chrono>
#include "e4b_engine.h"

static const char* kDir =
    "/opt/spark/models/hub/models--google--gemma-4-E4B-it/snapshots/"
    "fee6332c1abaafb77f6f9624236c63aa2f1d0187";

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0){
    return std::chrono::duration<double,std::milli>(clk::now()-t0).count();
}

int main(int argc, char** argv){
    const char* dir = (argc>1)?argv[1]:kDir;
    int n_pf  = (argc>2)?atoi(argv[2]):293;   // match the 31B comparison prompt
    int n_dec = (argc>3)?atoi(argv[3]):256;

    e4b_engine_t* eng = e4b_engine_create(dir, 4096, 0);
    if(!eng){ fprintf(stderr,"FAIL create\n"); return 1; }
    int V = e4b_engine_vocab_size(eng);
    printf("e4b_bench: %d layers, hidden %d, vocab %d, %.2f GB resident\n",
           e4b_engine_n_layers(eng), e4b_engine_hidden_size(eng), V,
           e4b_engine_device_bytes(eng)/1e9);

    // Synthetic prompt: token ids cycling in [1, 1000) (content is irrelevant to
    // throughput; avoid 0/specials). Decode feeds a constant token each step —
    // decode latency is independent of the token value.
    std::vector<int32_t> prompt(n_pf);
    for(int i=0;i<n_pf;i++) prompt[i] = 1 + (i % 997);
    std::vector<float> logits(V);

    // ── Warmup (kernels JIT/autotune, cuBLAS handles, first-touch allocs) ──
    e4b_engine_reset(eng);
    if(e4b_engine_prefill(eng, prompt.data(), n_pf, logits.data())!=0){ fprintf(stderr,"FAIL prefill warmup\n"); return 1; }
    for(int i=0;i<8;i++) e4b_engine_decode(eng, 42, logits.data());

    // ── Prefill timing (fresh cache) ──
    e4b_engine_reset(eng);
    auto t0 = clk::now();
    if(e4b_engine_prefill(eng, prompt.data(), n_pf, logits.data())!=0){ fprintf(stderr,"FAIL prefill\n"); return 1; }
    double pf_ms = ms_since(t0);

    // ── Decode timing (steady state, single token/step) ──
    auto t1 = clk::now();
    for(int i=0;i<n_dec;i++)
        if(e4b_engine_decode(eng, 42, logits.data())!=0){ fprintf(stderr,"FAIL decode %d\n",i); return 1; }
    double dec_ms = ms_since(t1);

    printf("\n== E4B throughput (BF16 weights, single-stream) ==\n");
    printf("prefill : %d tokens  %.2f ms  %.1f tok/s\n", n_pf, pf_ms, n_pf*1000.0/pf_ms);
    printf("decode  : %d tokens  %.2f ms  %.1f tok/s  (%.2f ms/token)\n",
           n_dec, dec_ms, n_dec*1000.0/dec_ms, dec_ms/n_dec);

    e4b_engine_destroy(eng);
    return 0;
}
