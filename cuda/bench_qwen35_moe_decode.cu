// ABOUTME: Low-concurrency MoE decode-step microbench — times the SERVED batched decode step
// ABOUTME: (gemma4_engine_step_batch / qwen35 CUDA-graph) at B=1,2,4,8 on Qwen3.5-35B-A3B-FP8.
//
// Isolates decode kernel/step time from HTTP/scheduler/TTFT: admits B sequences with DISTINCT
// short prompts (diverse MoE routing), warms the per-B batch graph, then times NSTEP graph-replayed
// decode steps via CUDA events. Prints ms/step and aggregate tok/s per B — the engine-only ceiling
// the served N=B aggregate sits just below. Pair with nsys --cuda-graph-trace=node for per-kernel
// attribution. Run under flock /tmp/fucina_gpu.lock. Throwaway L-moe-lowc attribution harness.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include "gemma4_kernels.cuh"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/0b2752837483aa34b3db6e83e151b150c0e00e49";
    int NSTEP = (argc > 2) ? atoi(argv[2]) : 128;
    int WARM  = (argc > 3) ? atoi(argv[3]) : 16;
    int Bs[]  = {1, 2, 4, 8};

    // FORMAT_Q4_0 is a placeholder — the qwen35 loader auto-detects FP8-block + fp4 experts.
    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.90);
    if (!eng) { fprintf(stderr, "engine_create failed (GPU memory? need ~24 GiB)\n"); return 2; }

    // 8 distinct short prompts so concurrent rows route to different experts (diverse decode).
    int32_t P[8][6] = {
        {  760, 6511,  314, 9338,  369,  -1},   //
        { 2082, 1235, 8091,  502,   -1,  -1},
        { 9906,  649,  387,  264,  502,  -1},
        {  791, 3363,  315, 7902,  374,  -1},
        {  40,  1053,  1093,  311,  198,  -1},
        { 5159,  836,  374,  16344, 323,  -1},
        { 3923,  374,  279,  6864,  315,  -1},
        { 7778,  757,  264,  2875,  502,  -1},
    };
    auto plen = [&](int i){ int n=0; while(n<6 && P[i][n]>=0) n++; return n; };

    printf("MoE decode-step microbench — Qwen3.5-35B-A3B-FP8 (served step_batch / CUDA-graph)\n");
    printf("%3s %10s %12s %14s\n", "B", "ms/step", "tok/s(agg)", "tok/s/stream");
    for (int bi = 0; bi < 4; bi++) {
        int B = Bs[bi];
        std::vector<int> slot(B); std::vector<int32_t> tok(B);
        for (int i = 0; i < B; i++) {
            int32_t first = 0;
            int s = gemma4_engine_seq_add(eng, P[i % 8], plen(i % 8), &first, 0.f, 0, 0.f, 0.f, 0);
            if (s < 0) { fprintf(stderr, "seq_add %d failed at B=%d\n", i, B); return 2; }
            slot[i] = s; tok[i] = first;
        }
        std::vector<int32_t> nxt(B);
        for (int k = 0; k < WARM; k++) {
            if (gemma4_engine_step_batch(eng, slot.data(), tok.data(), B, nxt.data()) != 0) {
                fprintf(stderr, "warm step failed B=%d\n", B); return 2; }
            for (int i = 0; i < B; i++) tok[i] = nxt[i];
        }
        cudaDeviceSynchronize();
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int k = 0; k < NSTEP; k++) {
            if (gemma4_engine_step_batch(eng, slot.data(), tok.data(), B, nxt.data()) != 0) {
                fprintf(stderr, "step failed B=%d\n", B); return 2; }
            for (int i = 0; i < B; i++) tok[i] = nxt[i];
        }
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        double per = ms / NSTEP;
        double agg = (double)B / (per / 1e3);
        printf("%3d %10.4f %12.1f %14.1f\n", B, per, agg, agg / B);
        for (int i = 0; i < B; i++) gemma4_engine_seq_remove(eng, slot[i]);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    gemma4_engine_destroy(eng);
    return 0;
}
