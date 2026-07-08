// test_qwen35_fp8_engine.cu — validate that the Qwen3.5-9B block-FP8 checkpoint served through the
// REAL batched gemma4_engine_t (gemma4_engine_create → FORMAT_FP8_BLOCK loader) reproduces the torch
// FP8 oracle greedy continuation, i.e. the loader lowered every FP8/Q8_0/f32 tensor into d_weights
// at the offsets the shared qwen35 forward reads. This is the EXTERNAL correctness gate the batched
// self-test can't give (that one only checks batched==token-by-token on the SAME engine).
//
// torch oracle (cuda/qwen35_fp8_ref.py), greedy:
//   prompt ids:   [760, 6511, 314, 9338, 369]                 ("The capital of France is")
//   continuation: [11751, 13, 198, 760, 6511, 314, 9338, 369] (" Paris.\nThe capital of France is")
// GATE = fucina engine's first 8 greedy ids == the oracle's; PLUS the batched self-test (row
// independence + graph determinism + batched==token-by-token) must PASS on the same engine.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "gemma4_kernels.cuh"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/models--Qwen--Qwen3.5-9B-FP8";

    int32_t in_ids[] = { 760, 6511, 314, 9338, 369 };
    const int NP = 5, NGEN = 12, GATE = 8;
    int32_t ref[GATE] = { 11751, 13, 198, 760, 6511, 314, 9338, 369 };

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.90);
    if (!eng) { fprintf(stderr, "create failed (is %s the FP8 checkpoint dir?)\n", path); return 2; }

    // (A) EXTERNAL oracle parity via the token-by-token forward (gemv_w FP8-block path).
    int32_t got[NGEN] = {0};
    if (qwen35_forward_greedy(eng, in_ids, NP, got, NGEN) != 0) {
        fprintf(stderr, "qwen35_forward_greedy failed\n"); gemma4_engine_destroy(eng); return 2;
    }
    int agree = 0;
    printf("pos | engine | oracle\n");
    for (int k = 0; k < GATE; k++) {
        int ok = (got[k] == ref[k]); agree += ok;
        printf("%3d | %6d | %6d  %s\n", k, got[k], ref[k], ok ? "" : "  <-- MISMATCH");
    }
    printf("FP8 engine oracle-parity: %d/%d\n", agree, GATE);

    // (B) batched self-consistency (row independence + graph-on==off + batched==token-by-token).
    int self = qwen35_batch_selftest(eng);

    gemma4_engine_destroy(eng);
    int pass = (agree == GATE) && (self == 0);
    printf("%s — qwen35 FP8-9B served through the batched engine (oracle %d/%d, self-test %s)\n",
           pass ? "PASS" : "FAIL", agree, GATE, self == 0 ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
