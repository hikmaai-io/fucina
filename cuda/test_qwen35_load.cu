// test_qwen35_load.cu — M1 loader gate for the Qwen3.5 hybrid (qwen35) GGUF.
//
// Loads the qwen35 Q4_K_M checkpoint through gemma4_engine_create and asserts the load
// succeeds. The loader (gemma4_kernels.cu qwen35_dump_and_validate) prints, at load time,
// each layer index + KIND (FULL softmax-GQA / LINEAR gated-deltanet) with the resolved GGUF
// tensor shapes and validates every shape against the arch spec (hidden 4096, head_dim 256,
// conv_dim 8192, state 128x128, 24 LINEAR / 8 FULL). A missing or misshaped tensor makes
// gemma4_engine_create return NULL → this gate fails. No forward is exercised here (M1 = load
// only); the dump+validate output is the deliverable.
#include <cstdio>
#include <cstdlib>
#include "gemma4_kernels.cuh"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf";

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.90);
    if (!eng) {
        fprintf(stderr, "FAIL — gemma4_engine_create returned NULL (missing/misshaped tensor)\n");
        return 1;
    }
    printf("PASS — qwen35 GGUF loaded; all per-layer tensor shapes validated (see dump above)\n");
    gemma4_engine_destroy(eng);
    return 0;
}
