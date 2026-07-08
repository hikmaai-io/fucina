// test_q4k_mmq_bench.cu — prefill TTFT microbench for the Q4_K MMQ vs BF16 prefill path.
// Times gemma4_engine_seq_add (the fresh paged_prefill_qwen3) for N=128/512/1024 tokens,
// median of several warm runs. Toggle the path with FUCINA_Q4K_MMQ=1 (MMQ) vs unset (BF16 default).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "gemma4_kernels.cuh"

static double time_prefill(gemma4_engine_t *eng, int N) {
    std::vector<int32_t> toks(N);
    for (int i = 0; i < N; i++) toks[i] = 1000 + (i % 4096);   // arbitrary valid ids
    const int reps = 7;
    std::vector<double> ms;
    for (int r = 0; r < reps; r++) {
        int32_t first = 0;
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        int slot = gemma4_engine_seq_add(eng, toks.data(), N, &first, 0.0f, 0, 0.0f, 0.0f, 0);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        if (slot < 0) { fprintf(stderr, "seq_add failed N=%d\n", N); return -1; }
        gemma4_engine_seq_remove(eng, slot);
        double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (r >= 2) ms.push_back(dt);   // drop 2 warmups
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size()/2];
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3-8B-abliterated.Q4_K_M.gguf";
    setenv("FUCINA_PAGED_KV", "1", 1);
    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.60);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }
    const char *mmq = getenv("FUCINA_Q4K_MMQ");
    printf("=== prefill TTFT (%s) — %s ===\n",
           (mmq && mmq[0]=='1') ? "Q4_K MMQ" : "BF16 dequant", path);
    int Ns[] = {128, 512, 1024};
    for (int i = 0; i < 3; i++) {
        double m = time_prefill(eng, Ns[i]);
        printf("  N=%4d : %8.2f ms  (%7.1f tok/s)\n", Ns[i], m, 1000.0*Ns[i]/m);
    }
    gemma4_engine_destroy(eng);
    return 0;
}
