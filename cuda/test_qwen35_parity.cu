// test_qwen35_parity.cu — M3 numeric parity of fucina's Qwen3.5 hybrid single-seq forward
// vs llama.cpp (llama-simple) on the SAME Q4_K_M GGUF, greedy argmax.
//
// Drives the qwen35 hybrid stack (24 GDN linear layers + 8 FULL output-gated softmax-GQA
// layers, period-4) token-by-token through qwen35_forward_greedy, carrying the GDN recurrent
// state + conv ring + the per-FULL-layer KV cache, and checks the greedy continuation matches
// the M0 llama-simple reference for "The capital of France is".
//
// llama.cpp (llama-simple, qwen35 fused Gated-DeltaNet, greedy) reference — confirmed at M3:
//   prompt ids:   [760, 6511, 314, 9338, 369]              ("The capital of France is")
//   continuation: [11751, 13, 198, 57590, 369, 279, 6511, 314, ...]  (" Paris.\nParis is the capital of")
//                 = " Paris.\nParis is the capital of France.\nThese"
// The M3 8/8 gate = fucina's first 8 greedy ids == [11751,13,198,57590,369,279,6511,314].
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "gemma4_kernels.cuh"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf";

    int32_t in_ids[] = { 760, 6511, 314, 9338, 369 };
    const int NP = 5, NGEN = 12, GATE = 8;
    int32_t ref[NGEN] = { 11751, 13, 198, 57590, 369, 279, 6511, 314, 9338, 13, 198, 9205 };

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.90);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    int32_t got[NGEN] = {0};
    if (qwen35_forward_greedy(eng, in_ids, NP, got, NGEN) != 0) {
        fprintf(stderr, "qwen35_forward_greedy failed\n");
        gemma4_engine_destroy(eng); return 2;
    }
    gemma4_engine_destroy(eng);

    int agree8 = 0, agreeAll = 0;
    printf("pos | fucina | llama.cpp\n");
    for (int k = 0; k < NGEN; k++) {
        int ok = (got[k] == ref[k]);
        agreeAll += ok;
        if (k < GATE) agree8 += ok;
        printf("%3d | %6d | %6d  %s%s\n", k, got[k], ref[k],
               ok ? "" : "  <-- MISMATCH", (k == GATE - 1) ? "   [gate boundary]" : "");
    }
    printf("greedy continuation: %d/%d gate match, %d/%d total\n", agree8, GATE, agreeAll, NGEN);
    printf("%s\n", (agree8 == GATE) ? "PASS — 8/8 greedy parity with llama.cpp (qwen35 hybrid)"
                                    : "FAIL — qwen35 hybrid greedy parity");
    return (agree8 == GATE) ? 0 : 1;
}
