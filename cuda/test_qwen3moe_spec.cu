// test_qwen3moe_spec.cu — DSpark speculative-decode lossless gate on Qwen3-30B-A3B MoE.
//
// Exercises gemma4_engine_step_batch_spec_ext (the EXTERNAL prompt-lookup verify ABI:
// no MTP head required — the path the continuous-batching scheduler uses for Qwen3/MoE).
// The MoE FFN lives inside decode_multiseq_forward, which the verify reuses unchanged, so
// this proves DSpark spec is lossless on the sparse-MoE forward.
//
//   (A) LOSSLESS d=0: a spec step with ZERO drafts must produce the SAME greedy token
//       stream as plain step_batch — the spec path with no drafts == decode.
//   (B) ACCEPT: feeding the model's own true next token T as a 1-draft accepts it
//       (run = [T, U], len 2) and advances the committed length by 2.
//   (C) REJECT: a wrong draft is rejected — the run still emits the correct token T (len 1).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "gemma4_kernels.cuh"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf";
    setenv("FUCINA_PAGED_KV", "1", 1);
    int32_t prompt[] = { 785, 6722, 315, 9625, 374 };   // "The capital of France is"
    const int NP = 5, STEPS = 24;

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.6);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    // ── (A) d=0 equivalence: rs via step_batch, ss via step_batch_spec_ext(no drafts) ──
    int32_t fr = 0, fs = 0;
    int rs = gemma4_engine_seq_add(eng, prompt, NP, &fr, 0.0f, 0, 0.0f, 0.0f, 0);
    int ss = gemma4_engine_seq_add(eng, prompt, NP, &fs, 0.0f, 0, 0.0f, 0.0f, 0);
    if (rs < 0 || ss < 0 || fr != fs) { fprintf(stderr, "seq_add mismatch %d/%d\n", fr, fs); return 2; }

    int32_t cr = fr, cs = fs; int agree = 0;
    for (int k = 0; k < STEPS; k++) {
        int32_t rn = 0; int rsl = rs;
        if (gemma4_engine_step_batch(eng, &rsl, &cr, 1, &rn) != 0) { fprintf(stderr, "step_batch fail\n"); return 2; }
        int32_t out[GEMMA4_SPEC_MAX]; int lens = 0; int dl = 0; int sl = ss; int32_t dr[GEMMA4_SPEC_MAX];
        if (gemma4_engine_step_batch_spec_ext(eng, &sl, &cs, 1, out, &lens, dr, &dl) != 0) { fprintf(stderr, "step_spec_ext fail\n"); return 2; }
        if (lens != 1) { fprintf(stderr, "d=0 run_len=%d (expected 1) at step %d\n", lens, k); return 1; }
        if (rn == out[0]) agree++;
        else printf("  step %d: step_batch=%d spec=%d MISMATCH\n", k, rn, out[0]);
        cr = rn; cs = out[0];
    }
    printf("(A) d=0 LOSSLESS: %d/%d greedy tokens byte-identical to step_batch\n", agree, STEPS);
    gemma4_engine_seq_remove(eng, rs);
    gemma4_engine_seq_remove(eng, ss);

    // ── (B) ACCEPT a correct draft. rb (step_batch) reveals the true T then U;
    //        sb (spec_ext) is fed anchor + draft=[T] and must emit run [T, U]. ──
    int32_t fb = 0, fc = 0;
    int rb = gemma4_engine_seq_add(eng, prompt, NP, &fb, 0.0f, 0, 0.0f, 0.0f, 0);
    int sb = gemma4_engine_seq_add(eng, prompt, NP, &fc, 0.0f, 0, 0.0f, 0.0f, 0);
    if (rb < 0 || sb < 0 || fb != fc) { fprintf(stderr, "seq_add(B) mismatch\n"); return 2; }
    int32_t T = 0, U = 0; int rbl = rb;
    if (gemma4_engine_step_batch(eng, &rbl, &fb, 1, &T) != 0) { fprintf(stderr, "step_batch T fail\n"); return 2; }
    if (gemma4_engine_step_batch(eng, &rbl, &T,  1, &U) != 0) { fprintf(stderr, "step_batch U fail\n"); return 2; }

    int32_t outB[GEMMA4_SPEC_MAX]; int lensB = 0; int dlB = 1; int slB = sb; int32_t drB[GEMMA4_SPEC_MAX] = { T };
    if (gemma4_engine_step_batch_spec_ext(eng, &slB, &fc, 1, outB, &lensB, drB, &dlB) != 0) { fprintf(stderr, "spec accept fail\n"); return 2; }
    int okB = (lensB == 2 && outB[0] == T && outB[1] == U);
    printf("(B) ACCEPT correct draft: run_len=%d run=[%d,%d] expect [%d,%d] -> %s\n",
           lensB, outB[0], lensB > 1 ? outB[1] : -1, T, U, okB ? "PASS" : "FAIL");
    gemma4_engine_seq_remove(eng, rb);
    gemma4_engine_seq_remove(eng, sb);

    // ── (C) REJECT a wrong draft: anchor produces T regardless; a bogus draft is
    //        rejected so the run is just [T] (len 1) — lossless under misprediction. ──
    int32_t fd = 0;
    int sc = gemma4_engine_seq_add(eng, prompt, NP, &fd, 0.0f, 0, 0.0f, 0.0f, 0);
    if (sc < 0) { fprintf(stderr, "seq_add(C) fail\n"); return 2; }
    int32_t bogus = (T == 0) ? 1 : 0;   // any token != T
    int32_t outC[GEMMA4_SPEC_MAX]; int lensC = 0; int dlC = 1; int slC = sc; int32_t drC[GEMMA4_SPEC_MAX] = { bogus };
    if (gemma4_engine_step_batch_spec_ext(eng, &slC, &fd, 1, outC, &lensC, drC, &dlC) != 0) { fprintf(stderr, "spec reject fail\n"); return 2; }
    int okC = (lensC == 1 && outC[0] == T);
    printf("(C) REJECT wrong draft: run_len=%d run=[%d] expect [%d] -> %s\n",
           lensC, outC[0], T, okC ? "PASS" : "FAIL");
    gemma4_engine_seq_remove(eng, sc);

    gemma4_engine_destroy(eng);
    int allok = (agree == STEPS) && okB && okC;
    printf("%s — DSpark spec lossless on Qwen3-MoE (d=0 == decode, accept grows, reject corrects)\n",
           allok ? "PASS" : "FAIL");
    return allok ? 0 : 1;
}
