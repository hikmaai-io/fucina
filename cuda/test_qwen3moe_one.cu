// test_qwen3moe_one.cu — single resident sequence, 8 greedy steps. Reference (parity
// agent + llama.cpp): " Paris. The capital of the United States" = 12095,13,576,6722,315,279,3639,4180.
// Then add a SECOND seq and re-step the FIRST to see if residency corrupts it.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "gemma4_kernels.cuh"
int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "/opt/spark/models/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf";
    setenv("FUCINA_PAGED_KV", "1", 1);
    int32_t prompt[] = { 785, 6722, 315, 9625, 374 };
    const int NP = 5;
    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.6);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    int32_t f = 0;
    int s = gemma4_engine_seq_add(eng, prompt, NP, &f, 0.0f, 0, 0.0f, 0.0f, 0);
    printf("SINGLE seq: %d", f);
    int32_t c = f;
    for (int k = 0; k < 7; k++) { int32_t n = 0; int sl = s; gemma4_engine_step_batch(eng, &sl, &c, 1, &n); printf(",%d", n); c = n; }
    printf("\n  expect: 12095,13,576,6722,315,279,3639,4180\n");

    // B=2 INDEPENDENCE: two seqs at DIFFERENT positions of the same deterministic trajectory
    // 12095,13,576,6722,315,279,3639,4180,...  seqA is advanced to idx7 (current token 4180),
    // seqB only to idx2 (current token 576). A correct multi-row MoE FFN must continue EACH row
    // independently in ONE step_batch(B=2): row0 -> trajectory[8] (== single-seq next), row1 ->
    // trajectory[3] = 6722. The rows MUST differ — if the batched MoE collapses rows they come out
    // equal (the pre-fix symptom). (seqA==s above already produced 4180 as its last token => c.)
    const int32_t TRAJ3 = 6722;   // trajectory[3]
    // Reference: single-seq next token after 4180 (idx8), on a fresh clone driven the same 8 tokens.
    int32_t fr = 0; int sr = gemma4_engine_seq_add(eng, prompt, NP, &fr, 0.0f, 0, 0.0f, 0.0f, 0);
    int32_t cr = fr; int32_t ref0 = 0;
    for (int k = 0; k < 8; k++) { int32_t n = 0; int sl = sr; gemma4_engine_step_batch(eng, &sl, &cr, 1, &n); cr = n; ref0 = n; }
    // seqB: fresh, advance only to idx2 (current token 576).
    int32_t fb = 0; int sb = gemma4_engine_seq_add(eng, prompt, NP, &fb, 0.0f, 0, 0.0f, 0.0f, 0);
    int32_t cb = fb; for (int k = 0; k < 2; k++) { int32_t n = 0; int sl = sb; gemma4_engine_step_batch(eng, &sl, &cb, 1, &n); cb = n; }
    // TRUE batch step over the two DIVERGENT rows.
    int slots2[2] = { s, sb }; int32_t in2[2] = { c, cb }; int32_t out2[2] = { 0, 0 };
    gemma4_engine_step_batch(eng, slots2, in2, 2, out2);
    printf("B=2 divergent ctx: in=[%d,%d] out=[%d,%d]  (expect row0=%d, row1=%d, rows differ)\n",
           c, cb, out2[0], out2[1], ref0, TRAJ3);
    int ok = (out2[0] == ref0) && (out2[1] == TRAJ3) && (out2[0] != out2[1]);
    printf("%s\n", ok ? "B=2 INDEPENDENT OK" : "B=2 ROW-COLLAPSE FAIL");
    gemma4_engine_destroy(eng);
    return ok ? 0 : 1;
}
