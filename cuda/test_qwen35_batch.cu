// test_qwen35_batch.cu — M4 gate for fucina's Qwen3.5 hybrid PAGED-BATCHED + CUDA-graph decode.
//
// Loads the qwen35 Q4_K_M GGUF and runs qwen35_batch_selftest, which drives the continuous-
// batching ABI (gemma4_engine_seq_add prefill + gemma4_engine_step_batch decode) and asserts:
//   (1) B-row batched decode (graph ON) is BIT-IDENTICAL per row to that row run alone B=1
//       (graph OFF) — the batch self-test invariant (row independence + graph correctness);
//   (2) graph-ON == graph-OFF for the B-row batch (CUDA-graph determinism);
//   (3) the batched path reproduces the proven M3 single-seq forward (qwen35_forward_greedy)
//       — the France->Paris 8/8 greedy continuation.
// Exits 0 on PASS.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "gemma4_kernels.cuh"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf";

    // Small context keeps the fp32 FULL-layer K/V arena modest; the gate only decodes ~30 tokens.
    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.90);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    int rc = qwen35_batch_selftest(eng);
    gemma4_engine_destroy(eng);

    printf("%s\n", (rc == 0) ? "PASS — qwen35 M4 batched-decode gate"
                             : "FAIL — qwen35 M4 batched-decode gate");
    return rc;
}
