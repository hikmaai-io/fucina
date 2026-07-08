// test_qwen35_fp8.cu — M5 parity of fucina's Qwen3.5 FP8 (DeepSeek block-fp8) forward vs the
// torch oracle (cuda/qwen35_fp8_ref.py) on the SAME official Qwen3.5-9B FP8 checkpoint, greedy.
//
// Loads the FP8 safetensors (model.language_model.* text path), drives the hybrid stack token-by-
// token through qwen35_fp8_forward_greedy (fp8_block decode GEMV for the projections), and checks
// the first 8 greedy continuation ids of "The capital of France is" match the oracle.
//
// torch oracle (HF Qwen3_5 modeling math, FP8 block-dequant; cuda/qwen35_fp8_ref.py) — greedy:
//   prompt ids:   [760, 6511, 314, 9338, 369]                ("The capital of France is")
//   continuation: [11751, 13, 198, 760, 6511, 314, 9338, 369, ...] (" Paris.\nThe capital of France is")
// The M5 8/8 gate = fucina's first 8 greedy ids == [11751,13,198,760,6511,314,9338,369].
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "gemma4_kernels.cuh"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/models--Qwen--Qwen3.5-9B-FP8";

    int32_t in_ids[] = { 760, 6511, 314, 9338, 369 };
    const int NP = 5, NGEN = 12, GATE = 8;
    int32_t ref[NGEN] = { 11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198, 760 };

    void *m = qwen35_fp8_load(path);
    if (!m) { fprintf(stderr, "qwen35_fp8_load failed\n"); return 2; }

    int32_t got[NGEN] = {0};
    if (qwen35_fp8_forward_greedy(m, in_ids, NP, got, NGEN) != 0) {
        fprintf(stderr, "qwen35_fp8_forward_greedy failed\n");
        qwen35_fp8_free(m); return 2;
    }
    qwen35_fp8_free(m);

    int agree8 = 0, agreeAll = 0;
    printf("pos | fucina | oracle\n");
    for (int k = 0; k < NGEN; k++) {
        int ok = (got[k] == ref[k]);
        agreeAll += ok;
        if (k < GATE) agree8 += ok;
        printf("%3d | %6d | %6d  %s%s\n", k, got[k], ref[k],
               ok ? "" : "  <-- MISMATCH", (k == GATE - 1) ? "   [gate boundary]" : "");
    }
    printf("greedy continuation: %d/%d gate match, %d/%d total\n", agree8, GATE, agreeAll, NGEN);
    printf("%s\n", (agree8 == GATE) ? "PASS — 8/8 greedy parity vs torch FP8 oracle (qwen35 FP8)"
                                    : "FAIL — qwen35 FP8 greedy parity");
    return (agree8 == GATE) ? 0 : 1;
}
