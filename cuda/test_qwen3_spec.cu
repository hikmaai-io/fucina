// test_qwen3_spec.cu — verify the ragged spec-verify ABI (gemma4_engine_step_batch_spec).
// (1) EQUIVALENCE: an anchor-only spec step (d=0) must produce the same greedy token
//     stream as the non-spec step_batch — i.e. the ragged path with no drafts == decode.
// (2) ACCEPT: feeding the correct next token as a draft must accept it (run_len==2) and
//     advance the committed length by 2; a wrong draft accepts 0 (run_len==1).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "gemma4_kernels.cuh"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3-8B-abliterated.Q4_K_M.gguf";
    setenv("FUCINA_PAGED_KV", "1", 1);
    int32_t prompt[] = { 785, 6722, 315, 9625, 374 };
    const int NP = 5, STEPS = 16;
    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    // Two identical sequences: one driven by step_batch, one by step_batch_spec(d=0).
    int32_t f0 = 0, f1 = 0;
    int s0 = gemma4_engine_seq_add(eng, prompt, NP, &f0, 0.0f, 0, 0.0f, 0.0f, 0);
    int s1 = gemma4_engine_seq_add(eng, prompt, NP, &f1, 0.0f, 0, 0.0f, 0.0f, 0);
    if (s0 < 0 || s1 < 0 || f0 != f1) { fprintf(stderr, "seq_add mismatch %d/%d\n", f0, f1); return 2; }

    int32_t a = f0, b = f1; int agree = 0;
    for (int k = 0; k < STEPS; k++) {
        int32_t nb = 0; int sl = s0;
        if (gemma4_engine_step_batch(eng, &sl, &a, 1, &nb) != 0) { fprintf(stderr,"step_batch fail\n"); return 2; }
        gemma4_spec_req req = { s1, b, NULL, 0 };
        int32_t run[GEMMA4_SPEC_MAX + 1]; int rl = 0, er = 0;
        if (gemma4_engine_step_batch_spec(eng, &req, 1, run, &rl, &er) != 0) { fprintf(stderr,"step_spec fail\n"); return 2; }
        if (rl != 1) { fprintf(stderr, "d=0 run_len=%d (expected 1)\n", rl); return 1; }
        if (nb == run[0]) agree++;
        else printf("  step %d: step_batch=%d spec=%d MISMATCH\n", k, nb, run[0]);
        a = nb; b = run[0];
    }
    printf("(1) d=0 equivalence: %d/%d greedy tokens match step_batch\n", agree, STEPS);
    int ok = (agree == STEPS);

    // (2) ACCEPT: at the current state of s1, what does the model predict next? Use it as a
    // correct draft and confirm the spec step ACCEPTS it (run_len==2).
    {
        gemma4_spec_req probe = { s1, b, NULL, 0 };
        int32_t pr[GEMMA4_SPEC_MAX + 1]; int prl = 0, er = 0;
        gemma4_engine_step_batch_spec(eng, &probe, 1, pr, &prl, &er);  // prl==1, pr[0]=next greedy
        // rewind isn't exposed; just check the accept logic on a fresh seq instead:
        int32_t f2 = 0; int s2 = gemma4_engine_seq_add(eng, prompt, NP, &f2, 0.0f, 0, 0.0f, 0.0f, 0);
        // first, learn the 2 greedy tokens after f2 via two d=0 steps on a sibling
        int32_t f3 = 0; int s3 = gemma4_engine_seq_add(eng, prompt, NP, &f3, 0.0f, 0, 0.0f, 0.0f, 0);
        gemma4_spec_req q = { s3, f3, NULL, 0 }; int32_t r3[GEMMA4_SPEC_MAX+1]; int l3=0,e3=0;
        gemma4_engine_step_batch_spec(eng, &q, 1, r3, &l3, &e3); int32_t tok1 = r3[0];
        gemma4_spec_req q2 = { s3, tok1, NULL, 0 }; gemma4_engine_step_batch_spec(eng, &q2, 1, r3, &l3, &e3); int32_t tok2 = r3[0];
        // now on s2, draft = [tok1]: anchor f2 + draft tok1 → should accept tok1 (run_len 2) and emit [tok1, tok2]
        int32_t drafts[1] = { tok1 };
        gemma4_spec_req sp = { s2, f2, drafts, 1 };
        int32_t rr[GEMMA4_SPEC_MAX+1]; int rrl = 0, ee = 0;
        gemma4_engine_step_batch_spec(eng, &sp, 1, rr, &rrl, &ee);
        int acc_ok = (rrl == 2 && rr[0] == tok1 && rr[1] == tok2);
        printf("(2) accept correct draft: run_len=%d run=[%d,%d] expected [%d,%d] %s\n",
               rrl, rr[0], rrl>1?rr[1]:-1, tok1, tok2, acc_ok ? "OK" : "FAIL");
        // wrong draft → accept 0 (run_len 1)
        int32_t bad[1] = { tok1 ^ 12345 };
        int32_t f4 = 0; int s4 = gemma4_engine_seq_add(eng, prompt, NP, &f4, 0.0f, 0, 0.0f, 0.0f, 0);
        gemma4_spec_req sp2 = { s4, f4, bad, 1 };
        int32_t rr2[GEMMA4_SPEC_MAX+1]; int rrl2 = 0, ee2 = 0;
        gemma4_engine_step_batch_spec(eng, &sp2, 1, rr2, &rrl2, &ee2);
        int rej_ok = (rrl2 == 1 && rr2[0] == tok1);  // anchor f4==f2 → same next greedy tok1
        printf("(2) reject wrong draft: run_len=%d run0=%d expected 1/%d %s\n",
               rrl2, rr2[0], tok1, rej_ok ? "OK" : "FAIL");
        ok = ok && acc_ok && rej_ok;
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    gemma4_engine_destroy(eng);
    return ok ? 0 : 1;
}
