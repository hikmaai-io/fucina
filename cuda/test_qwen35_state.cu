// test_qwen35_state.cu — gate for the Qwen3.5 hybrid per-SLOT state snapshot
// (the batched engine's per-conversation cache: GDN state S + conv rings +
// FULL-layer fp32 K/V prefix).
//
// Protocol (all greedy, so every path must be BIT-IDENTICAL):
//   1. seq_add(prompt) + T decode steps → committed sequence C (n_tokens long),
//      with the last sampled token `pending` NOT yet fed.
//   2. q35_state_save at n_tokens, then continue M more steps in place → REF.
//   3. remove the slot. Occupy slot 0 with a dummy sequence so the restore
//      lands in a DIFFERENT slot than the save (cross-slot correctness).
//   4. seq_open → restore(buf, n_tokens) → feed `pending` and step M → WARM.
//   5. cold re-run: seq_add(C) → boundary argmax must equal `pending`.
//   6. PASS iff WARM == REF for all M tokens (bit-identical) and the cold
//      boundary matches. The cold CONTINUATION is reported, not asserted:
//      prefill-derived GDN state is not bit-equal to decode-committed state
//      (fp32 path non-associativity; both sides pass their own oracle gates),
//      so greedy can legally fork after a few steps. The snapshot's promise is
//      exactly WARM==REF: the restored conversation continues as if never
//      evicted.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "gemma4_kernels.cuh"

static int step1(gemma4_engine_t *eng, int slot, int32_t in) {
    int slots[1] = { slot };
    int32_t ins[1] = { in }, outs[1] = { -1 };
    if (gemma4_engine_step_batch(eng, slots, ins, 1, outs) != 0) {
        fprintf(stderr, "step_batch failed\n"); exit(2);
    }
    return outs[0];
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf";

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.90);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    // "The capital of France is" — the pinned parity prompt.
    int32_t prompt[5] = { 760, 6511, 314, 9338, 369 };
    const int NP = 5, T = 24, M = 16;

    int32_t first = -1;
    int slot = gemma4_engine_seq_add(eng, prompt, NP, &first, 0.f, 0, 1.f, 0.f, 1);
    if (slot < 0) { fprintf(stderr, "seq_add failed\n"); return 2; }

    // Commit T tokens; committed sequence C = prompt + [first, o0..o_{T-2}],
    // with o_{T-1} = pending (sampled, never fed).
    int32_t C[NP + 64]; memcpy(C, prompt, sizeof(prompt));
    int nc = NP;
    int32_t cur = first;
    for (int i = 0; i < T; i++) {
        C[nc++] = cur;
        cur = step1(eng, slot, cur);
    }
    int32_t pending = cur;

    int n_tokens = gemma4_engine_seq_ntokens(eng, slot);
    if (n_tokens != nc) {
        fprintf(stderr, "FAIL: seq_ntokens=%d, committed=%d\n", n_tokens, nc); return 1;
    }

    size_t sz = gemma4_engine_q35_state_size(eng, n_tokens);
    if (sz == 0) { fprintf(stderr, "FAIL: state_size=0\n"); return 1; }
    void *buf = malloc(sz);
    if (gemma4_engine_q35_state_save(eng, slot, buf, n_tokens) != 0) {
        fprintf(stderr, "FAIL: state_save\n"); return 1;
    }
    printf("saved slot %d at %d tokens (%.1f MiB)\n", slot, n_tokens, sz / (1024.0 * 1024));

    // Reference continuation in place.
    int32_t REF[M]; cur = pending;
    for (int i = 0; i < M; i++) { REF[i] = cur; cur = step1(eng, slot, cur); }
    gemma4_engine_seq_remove(eng, slot);

    // Occupy the lowest slot with a dummy so the restore lands elsewhere.
    int32_t dummyfirst = -1;
    int dummy = gemma4_engine_seq_add(eng, prompt, 3, &dummyfirst, 0.f, 0, 1.f, 0.f, 1);
    if (dummy < 0) { fprintf(stderr, "dummy seq_add failed\n"); return 2; }

    int slot2 = gemma4_engine_seq_open(eng, 0.f, 0, 1.f, 0.f, 1);
    if (slot2 < 0) { fprintf(stderr, "seq_open failed\n"); return 2; }
    if (gemma4_engine_q35_state_restore(eng, slot2, buf, n_tokens) != 0) {
        fprintf(stderr, "FAIL: state_restore\n"); return 1;
    }
    if (gemma4_engine_seq_ntokens(eng, slot2) != n_tokens) {
        fprintf(stderr, "FAIL: restored n_tokens mismatch\n"); return 1;
    }
    int32_t WARM[M]; cur = pending;
    for (int i = 0; i < M; i++) { WARM[i] = cur; cur = step1(eng, slot2, cur); }
    gemma4_engine_seq_remove(eng, slot2);
    gemma4_engine_seq_remove(eng, dummy);

    // Cold re-run of the full committed sequence (production prefill path).
    int32_t coldfirst = -1;
    int slot3 = gemma4_engine_seq_add(eng, C, nc, &coldfirst, 0.f, 0, 1.f, 0.f, 1);
    if (slot3 < 0) { fprintf(stderr, "cold seq_add failed\n"); return 2; }
    if (coldfirst != pending) {
        fprintf(stderr, "FAIL: cold first token %d != pending %d\n", coldfirst, pending); return 1;
    }
    // Diagnostic: byte-compare the prefilled state against the decode-
    // committed snapshot (informational — quantifies the known fp32 gap
    // between the two paths; measured ~all GDN S entries, max|d|~0.36).
    {
        void *buf2 = malloc(sz);
        if (gemma4_engine_q35_state_save(eng, slot3, buf2, n_tokens) != 0) {
            fprintf(stderr, "FAIL: cold state_save\n"); return 1;
        }
        if (memcmp(buf, buf2, sz) != 0) {
            const float *a = (const float *)buf, *b = (const float *)buf2;
            size_t nf = sz / 4, ndiff = 0, firstd = (size_t)-1;
            float maxad = 0.f;
            for (size_t i = 0; i < nf; i++) {
                if (a[i] != b[i]) {
                    if (firstd == (size_t)-1) firstd = i;
                    float d = fabsf(a[i] - b[i]);
                    if (d > maxad) maxad = d;
                    ndiff++;
                }
            }
            printf("state diff: %zu/%zu floats differ, first@%zu, max|d|=%g\n",
                   ndiff, nf, firstd, maxad);
        } else {
            printf("state diff: prefilled state == decode-committed state (bit-exact)\n");
        }
        free(buf2);
    }
    int32_t COLD[M]; cur = coldfirst;
    for (int i = 0; i < M; i++) { COLD[i] = cur; cur = step1(eng, slot3, cur); }
    gemma4_engine_seq_remove(eng, slot3);

    int bad = 0, coldAgree = 0;
    for (int i = 0; i < M; i++) {
        if (WARM[i] != REF[i]) {
            fprintf(stderr, "WARM mismatch @%d: ref=%d warm=%d\n", i, REF[i], WARM[i]);
            bad++;
        }
        if (COLD[i] == REF[i] && coldAgree == i) coldAgree++;
    }
    free(buf);
    gemma4_engine_destroy(eng);
    if (bad) { printf("FAIL — qwen35 state snapshot gate (%d/%d warm mismatches)\n", bad, M); return 1; }
    printf("PASS — qwen35 state snapshot gate: warm==ref %d/%d bit-identical (cross-slot restore); "
           "cold boundary ok, cold agrees %d/%d (informational)\n", M, M, coldAgree, M);
    return 0;
}
