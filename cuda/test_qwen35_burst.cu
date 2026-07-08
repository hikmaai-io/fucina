// test_qwen35_burst.cu — diverse-prompt burst-admission gate for the qwen35 batched engine.
//
// Regression guard for the conc-N diverse-prompt serving corruption (fixed in the commit that
// replaced cublasGemmGroupedBatchedEx): identical-prompt benches mask row-mixing and GEMM
// nondeterminism because every row computes the same values, so this gate drives DIVERSE
// long prompts (>2*FP8_MAXB tokens — the wide tensor-core prefill path) through the
// continuous-batching ABI and asserts, all greedy:
//   (1) determinism  — re-prefilling the same prompt reproduces the same continuation, and
//                      5 back-to-back prefill/remove cycles yield the same first token;
//   (2) burst        — 4 diverse rows admitted back-to-back (no decode between) then decoded
//                      in lockstep are BIT-IDENTICAL per row to that row run alone;
//   (3) warmup+burst — (2) still holds after a 16-seq lockstep warmup staircase (the server's
//                      boot graph warmup; caught the 27B runtime-NQ partials overflow at B>=11).
// Exits 0 on PASS.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include "gemma4_kernels.cuh"

static const int K = 12;
// Natural-text token ids (Qwen3.5/3.6 vocab): stable greedy continuations, unlike random ids
// whose near-uniform logits flip on benign numeric drift.
// A: "The ocean covers most of our planet. ... The largest ocean on Earth is the"
// B: "Mathematics is the study of numbers ... The smallest prime number is"
static const int NPA = 34, NPB = 40;
static const int32_t PROMPT_A[NPA] = {760,17415,14103,1379,314,1004,11247,13,1049,76589,279,9682,11,17882,11391,314,9140,11,321,18082,1691,314,279,22817,567,35097,13,561,7526,17415,383,8964,369,279};
static const int32_t PROMPT_B[NPB] = {8549,32588,369,279,3788,314,4947,11,20182,11,321,12261,13,357,9944,1324,369,264,5629,1324,6826,1056,799,421,682,874,6572,3319,39852,975,1056,799,321,4924,13,561,23856,9944,1324,369};

static int run_solo(gemma4_engine_t *eng, const int32_t *prompt, int NP, int32_t *out) {
    int32_t first = 0;
    int slot = gemma4_engine_seq_add(eng, prompt, NP, &first, 0.f, 0, 0.f, 0.f, 0);
    if (slot < 0) { fprintf(stderr, "burst gate: solo seq_add failed\n"); return -1; }
    int32_t tok = first;
    for (int k = 0; k < K; k++) {
        out[k] = tok;
        int32_t nxt = 0; int sl = slot;
        if (gemma4_engine_step_batch(eng, &sl, &tok, 1, &nxt) != 0) {
            fprintf(stderr, "burst gate: solo step failed\n");
            gemma4_engine_seq_remove(eng, slot); return -1;
        }
        tok = nxt;
    }
    gemma4_engine_seq_remove(eng, slot);
    return 0;
}

static int cmp(const char *tag, int q, const int32_t *ref, const int32_t *got) {
    int agree = 0; while (agree < K && ref[agree] == got[agree]) agree++;
    printf("qwen35 burst %-12s row %d: %2d/%d%s", tag, q, agree, K, agree == K ? "\n" : "  got[");
    if (agree < K) {
        for (int k = 0; k < K; k++) printf("%d%s", got[k], k + 1 < K ? "," : "] solo [");
        for (int k = 0; k < K; k++) printf("%d%s", ref[k], k + 1 < K ? "," : "]\n");
    }
    return agree == K;
}

// 4 diverse rows admitted back-to-back, decoded in lockstep, compared to their solo runs.
static int burst4(gemma4_engine_t *eng, const char *tag, int32_t solo4[4][K]) {
    const int32_t *P4[4] = { PROMPT_A, PROMPT_B, PROMPT_A, PROMPT_B };
    const int NP4[4] = { NPA, NPB, 20, 24 };
    int slot[4]; int32_t cur[4], first;
    for (int q = 0; q < 4; q++) {
        slot[q] = gemma4_engine_seq_add(eng, P4[q], NP4[q], &first, 0.f, 0, 0.f, 0.f, 0);
        if (slot[q] < 0) { fprintf(stderr, "burst gate: burst seq_add failed\n"); return 0; }
        cur[q] = first;
    }
    int32_t got[4][K];
    for (int k = 0; k < K; k++) {
        int32_t nxt[4];
        for (int q = 0; q < 4; q++) got[q][k] = cur[q];
        if (gemma4_engine_step_batch(eng, slot, cur, 4, nxt) != 0) {
            fprintf(stderr, "burst gate: burst step failed\n"); return 0;
        }
        for (int q = 0; q < 4; q++) cur[q] = nxt[q];
    }
    for (int q = 0; q < 4; q++) gemma4_engine_seq_remove(eng, slot[q]);
    int pass = 1;
    for (int q = 0; q < 4; q++) pass &= cmp(tag, q, solo4[q], got[q]);
    return pass;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf";
    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.90);
    if (!eng) { fprintf(stderr, "burst gate: create failed\n"); return 2; }
    int pass = 1;

    // (1) determinism: solo references + re-solo + 5 prefill/remove first-token repeats
    const int32_t *P4[4] = { PROMPT_A, PROMPT_B, PROMPT_A, PROMPT_B };
    const int NP4[4] = { NPA, NPB, 20, 24 };
    int32_t solo4[4][K];
    for (int q = 0; q < 4; q++) if (run_solo(eng, P4[q], NP4[q], solo4[q])) return 2;
    { int32_t re[K];
      if (run_solo(eng, PROMPT_A, NPA, re)) return 2;
      pass &= cmp("re-solo", 0, solo4[0], re); }
    { int32_t f0 = -1; int detok = 1;
      for (int it = 0; it < 5; it++) {
          int32_t first = 0;
          int slot = gemma4_engine_seq_add(eng, PROMPT_A, NPA, &first, 0.f, 0, 0.f, 0.f, 0);
          if (slot < 0) { fprintf(stderr, "burst gate: repeat seq_add failed\n"); return 2; }
          gemma4_engine_seq_remove(eng, slot);
          if (it == 0) f0 = first;
          detok &= (first == f0);
      }
      printf("qwen35 burst prefill-repeat: %s\n", detok ? "5/5 identical" : "MISMATCH");
      pass &= detok; }

    // (2) diverse burst from idle
    pass &= burst4(eng, "idle", solo4);

    // (3) 16-seq lockstep warmup staircase (server boot shape), then the burst again
    {
        const int MS = gemma4_engine_seq_capacity(eng);   // free slots == MAX_SEQS here
        int ws[64]; int32_t wc[64], first;
        int nw = 0;
        for (int i = 0; i < MS && i < 64; i++) {
            int s = gemma4_engine_seq_add(eng, PROMPT_A, 8, &first, 0.f, 0, 0.f, 0.f, 0);
            if (s < 0) break;
            ws[nw] = s; wc[nw] = first; nw++;
        }
        int wok = (nw == MS && nw > 0);
        for (int k = 0; k < 4 && wok; k++) {
            int32_t nxt[64];
            wok &= gemma4_engine_step_batch(eng, ws, wc, nw, nxt) == 0;
            for (int i = 0; i < nw; i++) wc[i] = nxt[i];
        }
        for (int i = 0; i < nw; i++) gemma4_engine_seq_remove(eng, ws[i]);
        printf("qwen35 burst warmup (B=%d x4 steps): %s\n", nw, wok ? "ok" : "FAILED");
        pass &= wok;
    }
    pass &= burst4(eng, "post-warmup", solo4);

    gemma4_engine_destroy(eng);
    printf("%s\n", pass ? "PASS — qwen35 diverse-burst gate" : "FAIL — qwen35 diverse-burst gate");
    return pass ? 0 : 1;
}
