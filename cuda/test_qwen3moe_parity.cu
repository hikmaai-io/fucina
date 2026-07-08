// test_qwen3moe_parity.cu — numeric parity of fucina's Qwen3-MoE forward vs llama.cpp.
// Feeds the EXACT input token ids llama.cpp produced for the RAW completion of
// "The capital of France is" through fucina's arch-driven multiseq path (seq_add + greedy
// step_batch) and checks the greedy continuation matches llama.cpp's (same UD-Q4_K_XL GGUF).
//
// Reference (llama-completion, -no-cnv --temp 0 --top-k 1, greedy):
//   input:        [785, 6722, 315, 9625, 374]              ("The capital of France is")
//   continuation: [12095, 13, 576, 6722, 315, 279, 3639, 4180]
//                 (" Paris. The capital of the United States")
//
// Besides the 8/8 token match, this also self-checks coherence: a wrong router or a corrupted
// Q5_K-requantized layer would diverge from llama.cpp token-for-token.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "gemma4_kernels.cuh"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf";
    setenv("FUCINA_PAGED_KV", "1", 1);   // arch-driven multiseq path requires paged KV

    int32_t in_ids[] = { 785, 6722, 315, 9625, 374 };
    const int NP = 5, NGEN = 8;
    int32_t ref[NGEN] = { 12095, 13, 576, 6722, 315, 279, 3639, 4180 };

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.60);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    int32_t first = 0;
    int slot = gemma4_engine_seq_add(eng, in_ids, NP, &first, 0.0f, 0, 0.0f, 0.0f, 0);
    if (slot < 0) { fprintf(stderr, "seq_add failed\n"); return 2; }

    int32_t got[NGEN];
    int32_t tok = first;
    for (int k = 0; k < NGEN; k++) {
        got[k] = tok;
        int32_t nxt = 0; int sl = slot;
        if (gemma4_engine_step_batch(eng, &sl, &tok, 1, &nxt) != 0) {
            fprintf(stderr, "step_batch failed at %d\n", k); return 2;
        }
        tok = nxt;
    }
    gemma4_engine_seq_remove(eng, slot);

    int agree = 0;
    printf("pos | fucina | llama.cpp\n");
    for (int k = 0; k < NGEN; k++) {
        int ok = (got[k] == ref[k]);
        agree += ok;
        printf("%3d | %6d | %6d  %s\n", k, got[k], ref[k], ok ? "" : "  <-- MISMATCH");
    }
    double pct = 100.0 * agree / NGEN;
    printf("greedy continuation: %d/%d match (%.0f%%)\n", agree, NGEN, pct);
    printf("%s\n", (agree == NGEN) ? "PASS — exact greedy parity with llama.cpp"
                                   : (agree >= NGEN*3/4 ? "CLOSE (quant/reduction-order near-ties)" : "FAIL"));
    gemma4_engine_destroy(eng);
    return (agree == NGEN) ? 0 : 1;
}
