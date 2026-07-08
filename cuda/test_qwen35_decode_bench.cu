// test_qwen35_decode_bench.cu — decode tok/s of the SERVED qwen35 path (gemma4_engine_step_batch
// → qwen35_decode_multiseq_body / CUDA-graph). Loads the GGUF, prefills a short prompt via
// gemma4_engine_seq_add, warms the batch graph, then times NSTEP single-token decode steps.
// Prints tok/s. Throwaway P5 measurement harness (not a gate).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include "gemma4_kernels.cuh"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf";
    int NSTEP = (argc > 2) ? atoi(argv[2]) : 128;
    int WARM  = 8;

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.90);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    int32_t prompt[5] = { 760, 6511, 314, 9338, 369 };
    int32_t first = 0;
    int slot = gemma4_engine_seq_add(eng, prompt, 5, &first, 0.f, 0, 0.f, 0.f, 0);
    if (slot < 0) { fprintf(stderr, "seq_add failed\n"); gemma4_engine_destroy(eng); return 2; }

    int32_t tok = first;
    // warmup (graph capture happens on first step at this B)
    for (int k = 0; k < WARM; k++) {
        int32_t nxt = 0; int sl = slot;
        if (gemma4_engine_step_batch(eng, &sl, &tok, 1, &nxt) != 0) {
            fprintf(stderr, "warm step failed\n"); gemma4_engine_destroy(eng); return 2; }
        tok = nxt;
    }
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < NSTEP; k++) {
        int32_t nxt = 0; int sl = slot;
        if (gemma4_engine_step_batch(eng, &sl, &tok, 1, &nxt) != 0) {
            fprintf(stderr, "step failed\n"); gemma4_engine_destroy(eng); return 2; }
        tok = nxt;
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    gemma4_engine_seq_remove(eng, slot);
    gemma4_engine_destroy(eng);

    printf("DECODE-BENCH: %d steps in %.4f s = %.2f tok/s (B=1, served step_batch path)\n",
           NSTEP, sec, NSTEP / sec);
    return 0;
}
