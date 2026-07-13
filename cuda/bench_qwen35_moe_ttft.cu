// ABOUTME: Microbench isolating the Qwen3.5 MoE batched-admission prefill cost at M=1..32.
// ABOUTME: Times gemma4_engine_seq_add_multiseq (the N=32 TTFT critical path) via CUDA events,
// so the per-phase attribution excludes HTTP/scheduler wall-time. Diverse ~15-tok prompts.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "gemma4_kernels.cuh"

static void mkprompt(int32_t *p, int n, uint32_t seed) {
    for (int i = 0; i < n; i++)
        p[i] = (int32_t)((((uint32_t)i * 1103515245u + 12345u + seed * 2654435761u) >> 8) % 30000u + 100u);
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char **argv) {
    const char *moe = (argc > 1) ? argv[1]
        : "/opt/spark/models/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/0b2752837483aa34b3db6e83e151b150c0e00e49";
    const int PROMPT_LEN = (argc > 2) ? atoi(argv[2]) : 15;   // ~15-tok admission prompts
    const int REPS = (argc > 3) ? atoi(argv[3]) : 12;

    gemma4_engine_t *eng = gemma4_engine_create(moe, FORMAT_Q4_0, 4096, 0, 0.90);
    if (!eng) { printf("engine_create failed\n"); return 1; }

    int Ms_all[] = {1, 2, 4, 8, 16, 32};
    int Ms[6]; int nM = 0;
    const char *only = getenv("BENCH_ONLY_M");
    for (int i = 0; i < 6; i++) { if (!only || Ms_all[i] == atoi(only)) Ms[nM++] = Ms_all[i]; }
    printf("# Qwen3.5 MoE batched-admission prefill microbench (prompt_len=%d, reps=%d)\n", PROMPT_LEN, REPS);
    printf("# %4s %10s %10s %12s\n", "M", "median_ms", "p95_ms", "ms/seq");
    for (int mi = 0; mi < nM; mi++) {
        int M = Ms[mi];
        int Ttot = M * PROMPT_LEN;
        std::vector<int> lens(M, PROMPT_LEN);
        std::vector<int32_t> toks(Ttot);

        std::vector<double> times;
        // warmup + timed reps; fresh slots each rep (remove between).
        for (int r = 0; r < REPS + 2; r++) {
            for (int i = 0; i < M; i++) mkprompt(toks.data() + i * PROMPT_LEN, PROMPT_LEN, 7u + 1000u * i + 31u * r);
            std::vector<int> slots(M); std::vector<int32_t> firsts(M);
            cudaDeviceSynchronize();
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventRecord(e0);
            int rc = gemma4_engine_seq_add_multiseq(eng, toks.data(), lens.data(), M,
                                                    NULL, NULL, NULL, NULL, NULL,
                                                    slots.data(), firsts.data());
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            if (rc != M) { printf("  M=%d rc=%d FAIL\n", M, rc); return 1; }
            for (int i = 0; i < M; i++) gemma4_engine_seq_remove(eng, slots[i]);
            if (r >= 2) times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        double med = median(times);
        double p95 = times[(size_t)(0.95 * (times.size() - 1) + 0.5)];
        printf("  %4d %10.2f %10.2f %12.3f\n", M, med, p95, med / M);
    }
    gemma4_engine_destroy(eng);
    return 0;
}
