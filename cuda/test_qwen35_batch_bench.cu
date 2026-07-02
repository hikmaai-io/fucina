// test_qwen35_batch_bench.cu — aggregate decode tok/s of the SERVED qwen35 path at B=1..MAX_SEQS
// (gemma4_engine_step_batch → qwen35_decode_multiseq_body / CUDA-graph). Loads the checkpoint,
// prefills B short prompts, warms the batch graph per B, then times NSTEP B-row decode steps.
// Prints per-B aggregate + per-stream tok/s. Throwaway measurement harness (not a gate).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include "gemma4_kernels.cuh"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/models--Qwen--Qwen3.5-35B-A3B-FP8";
    int NSTEP = (argc > 2) ? atoi(argv[2]) : 96;
    int BMAX  = (argc > 3) ? atoi(argv[3]) : 16;
    const int WARM = 6;

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.90);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    for (int B = 1; B <= BMAX; B <<= 1) {
        int slot[64]; int32_t cur[64];
        for (int q = 0; q < B; q++) {
            // distinct short prompts (in-vocab ids) so rows sit at slightly different states
            int32_t prompt[6] = { 760, 6511, 314, 9338, 369, (int32_t)(1000 + 37*q) };
            int np = 5 + (q & 1);
            int32_t first = 0;
            slot[q] = gemma4_engine_seq_add(eng, prompt, np, &first, 0.f, 0, 0.f, 0.f, 0);
            if (slot[q] < 0) { fprintf(stderr, "seq_add failed (B=%d q=%d)\n", B, q); return 2; }
            cur[q] = first;
        }
        for (int k = 0; k < WARM; k++) {
            int32_t nxt[64];
            if (gemma4_engine_step_batch(eng, slot, cur, B, nxt) != 0) { fprintf(stderr,"warm failed\n"); return 2; }
            for (int q = 0; q < B; q++) cur[q] = nxt[q];
        }
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < NSTEP; k++) {
            int32_t nxt[64];
            if (gemma4_engine_step_batch(eng, slot, cur, B, nxt) != 0) { fprintf(stderr,"step failed\n"); return 2; }
            for (int q = 0; q < B; q++) cur[q] = nxt[q];
        }
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        printf("BATCH-BENCH B=%2d: %d steps in %7.3f s  = %7.2f tok/s aggregate  (%6.2f tok/s/stream)\n",
               B, NSTEP, sec, (double)NSTEP * B / sec, (double)NSTEP / sec);
        for (int q = 0; q < B; q++) gemma4_engine_seq_remove(eng, slot[q]);
    }
    gemma4_engine_destroy(eng);
    return 0;
}
