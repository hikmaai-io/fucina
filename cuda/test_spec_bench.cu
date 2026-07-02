// test_spec_bench.cu — effective single-stream decode tok/s of the SERVED spec path
// (gemma4_engine_step_batch_spec_ext: external prompt-lookup drafts + one batched lossless
// verify) vs the plain step_batch baseline, on the same engine/prompt. The drafter is the
// same n-gram prompt-lookup the Go server uses (match the last NGRAM generated ids earlier
// in the context, propose the K tokens that followed). Throwaway measurement harness.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include "gemma4_kernels.cuh"

// prompt-lookup draft: find the most recent earlier occurrence of the last NGRAM ids in ctx
// and copy up to K following ids. Returns draft length (0 if no match).
static int pld_draft(const std::vector<int32_t> &ctx, int ngram, int K, int32_t *out) {
    int n = (int)ctx.size();
    if (n < ngram + 1) return 0;
    for (int s = n - ngram - 1; s >= 0; s--) {
        bool m = true;
        for (int j = 0; j < ngram; j++) if (ctx[s + j] != ctx[n - ngram + j]) { m = false; break; }
        if (!m) continue;
        int c = 0;
        for (int j = s + ngram; j < n && c < K; j++) out[c++] = ctx[j];
        return c;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "/opt/spark/models/gemma-4-12b-it-qat-q4_0.gguf";
    int NGEN  = (argc > 2) ? atoi(argv[2]) : 256;
    int K     = (argc > 3) ? atoi(argv[3]) : 8;      // draft tokens per verify
    int NGRAM = (argc > 4) ? atoi(argv[4]) : 2;

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.90);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    int32_t prompt[5] = { 760, 6511, 314, 9338, 369 };

    // ── baseline: plain greedy step_batch ──
    double base_sec = 0; std::vector<int32_t> base_ids;
    {
        int32_t first = 0;
        int slot = gemma4_engine_seq_add(eng, prompt, 5, &first, 0.f, 0, 0.f, 0.f, 0);
        if (slot < 0) { fprintf(stderr, "seq_add failed\n"); return 2; }
        int32_t tok = first; base_ids.push_back(first);
        for (int k = 0; k < 8; k++) { int32_t n=0; gemma4_engine_step_batch(eng, &slot, &tok, 1, &n); tok=n; base_ids.push_back(n); }  // warm
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < NGEN; k++) {
            int32_t n = 0;
            if (gemma4_engine_step_batch(eng, &slot, &tok, 1, &n) != 0) { fprintf(stderr,"step failed\n"); return 2; }
            tok = n; base_ids.push_back(n);
        }
        cudaDeviceSynchronize();
        base_sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        gemma4_engine_seq_remove(eng, slot);
    }

    // ── spec: prompt-lookup drafts + step_batch_spec_ext ──
    double spec_sec = 0; std::vector<int32_t> spec_ids;
    long drafted = 0, accepted = 0, calls = 0;
    {
        int32_t first = 0;
        int slot = gemma4_engine_seq_add(eng, prompt, 5, &first, 0.f, 0, 0.f, 0.f, 0);
        if (slot < 0) { fprintf(stderr, "seq_add(spec) failed\n"); return 2; }
        std::vector<int32_t> ctx(prompt, prompt + 5);
        ctx.push_back(first); spec_ids.push_back(first);
        int32_t pend = first;
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        while ((int)spec_ids.size() < NGEN) {
            int32_t drafts[GEMMA4_SPEC_MAX]; int dlen = pld_draft(ctx, NGRAM, K, drafts);
            int32_t out[GEMMA4_SPEC_MAX]; int olen = 0;
            if (gemma4_engine_step_batch_spec_ext(eng, &slot, &pend, 1, out, &olen, drafts, &dlen) != 0 || olen < 1) {
                fprintf(stderr, "spec step failed (olen=%d)\n", olen); return 2;
            }
            calls++; drafted += dlen; accepted += olen - 1;
            for (int j = 0; j < olen; j++) { ctx.push_back(out[j]); spec_ids.push_back(out[j]); }
            pend = out[olen - 1];
        }
        cudaDeviceSynchronize();
        spec_sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        gemma4_engine_seq_remove(eng, slot);
    }

    // lossless check: spec continuation must equal the baseline greedy ids
    int nchk = (int)((base_ids.size() - 9 < spec_ids.size() - 1) ? base_ids.size() - 9 : spec_ids.size() - 1);
    int mism = 0;
    for (int i = 0; i < nchk; i++) if (base_ids[9 + i] != spec_ids[1 + i]) mism++;

    int ntok = (int)spec_ids.size() - 1;
    printf("SPEC-BENCH baseline: %d tok in %.3f s = %.2f tok/s\n", NGEN, base_sec, NGEN / base_sec);
    printf("SPEC-BENCH spec    : %d tok in %.3f s = %.2f tok/s  (K=%d ngram=%d, %ld verify calls, "
           "drafted %ld accepted %ld = %.2f acc/call, lossless %s)\n",
           ntok, spec_sec, ntok / spec_sec, K, NGRAM, calls, drafted, accepted,
           calls ? (double)accepted / calls : 0.0, mism == 0 ? "PASS" : "FAIL");
    gemma4_engine_destroy(eng);
    return mism == 0 ? 0 : 1;
}
