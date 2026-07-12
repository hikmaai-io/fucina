// ABOUTME: P0 (S1a) gate for lossless GDN snapshot/rewind/commit — the DFlash (1+K) verification
// ABOUTME: prerequisite. Proves commit(j) leaves state byte-identical to j sequential decodes.
//
// The hybrid Qwen3.5-9B has 24 GDN + 8 full-attention layers; a DFlash verify pass advances a
// slot through (1+K) speculative tokens but keeps only the first j accepted. This gate asserts the
// core losslessness contract for every j in 0..K:
//
//   REFERENCE(j):  from a fixed committed prefix, decode exactly j tokens one-by-one; capture the
//                  resulting GDN recurrent-state snapshot Rj and the (j+1)-th produced token Tj.
//   ROLLBACK(j):   snapshot the same prefix; speculatively advance the SAME K candidate tokens
//                  (as one-by-one steps, mimicking a verify pass that over-runs); commit(j) to
//                  restore+replay only the j accepted tokens; capture its state Cj and next token.
//
//   PASS iff for every j: Cj == Rj BYTE-IDENTICAL (recurrent_slab compare via q35_state_save's
//   GDN portion) AND the next produced token matches. j=0 (pure rewind) and j=K (full accept) are
//   included. This is byte-identical, not within-a-bound: commit re-runs the exact sequential
//   decode path, so any drift is a real bug.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
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
    const int NP = 5, WARMUP = 20, K = 6;

    // Warm a slot to a non-trivial committed prefix, greedy (deterministic).
    int32_t first = -1;
    int slot = gemma4_engine_seq_add(eng, prompt, NP, &first, 0.f, 0, 1.f, 0.f, 1);
    if (slot < 0) { fprintf(stderr, "seq_add failed\n"); return 2; }
    int32_t cur = first;
    for (int i = 0; i < WARMUP; i++) cur = step1(eng, slot, cur);
    // `cur` is the first candidate token to be decoded at the verify boundary.

    int n0 = gemma4_engine_seq_ntokens(eng, slot);
    // State buffers are n_tokens-dependent (the FULL-layer K/V slice grows), so size PER j. sz0 is
    // the boundary; szj[j] sizes the state after n0+j committed tokens. Comparing the full state
    // (GDN recurrent + FULL K/V) at n0+j is strictly stronger than comparing GDN alone.
    size_t sz0 = gemma4_engine_q35_state_size(eng, n0);
    if (sz0 == 0) { fprintf(stderr, "FAIL: state_size=0\n"); return 1; }

    // Snapshot the boundary state to a host buffer so we can restore the SAME starting point for
    // each reference/rollback pair (the engine snapshot is device-side and consumed by commit).
    void *boundary = malloc(sz0);
    if (gemma4_engine_q35_state_save(eng, slot, boundary, n0) != 0) {
        fprintf(stderr, "FAIL: boundary save\n"); return 1;
    }

    // The K candidate tokens are the greedy continuation from the boundary (a realistic verify
    // batch: draft proposals that the target would itself have produced, so acceptance is high).
    int32_t cand[16]; int32_t ref_next[16];
    {
        // Produce candidates + the reference "sequential" next tokens by decoding K in place, then
        // restore the boundary for the rollback phase.
        int32_t c = cur;
        for (int j = 0; j < K; j++) { cand[j] = c; c = step1(eng, slot, c); }
    }

    // Collect reference state Rj + next token for each j by sequential decode from the boundary.
    void *refbuf[16] = {0}; size_t szj[16] = {0};
    for (int j = 0; j <= K; j++) {
        if (gemma4_engine_q35_state_restore(eng, slot, boundary, n0) != 0) {
            fprintf(stderr, "FAIL: restore boundary (ref j=%d)\n", j); return 1;
        }
        int32_t c = cand[0];
        for (int i = 0; i < j; i++) c = step1(eng, slot, c);
        // state after exactly j sequential decodes:
        szj[j] = gemma4_engine_q35_state_size(eng, n0 + j);
        refbuf[j] = malloc(szj[j]);
        if (gemma4_engine_q35_state_save(eng, slot, refbuf[j], n0 + j) != 0) {
            fprintf(stderr, "FAIL: ref save j=%d\n", j); return 1;
        }
        ref_next[j] = c;  // the token that WOULD be decoded next (input to step j)
    }

    // Rollback phase: for each j, restore boundary, snapshot via the P0 primitive, speculatively
    // advance all K candidates (over-run, as a verify pass does), then commit(j).
    int fails = 0;
    for (int j = 0; j <= K; j++) {
        if (gemma4_engine_q35_state_restore(eng, slot, boundary, n0) != 0) {
            fprintf(stderr, "FAIL: restore boundary (rollback j=%d)\n", j); return 1;
        }
        if (gemma4_engine_q35_gdn_snapshot(eng, slot) != 0) {
            fprintf(stderr, "FAIL: gdn_snapshot j=%d\n", j); return 1;
        }
        // Speculative over-run: advance all K candidates in place (mutates GDN state past accept).
        int32_t c = cand[0];
        for (int i = 0; i < K; i++) c = step1(eng, slot, c);
        // Commit only the first j accepted candidates.
        int32_t out_next[16];
        if (gemma4_engine_q35_gdn_commit(eng, slot, cand, j, out_next) != 0) {
            fprintf(stderr, "FAIL: gdn_commit j=%d\n", j); return 1;
        }
        if (gemma4_engine_seq_ntokens(eng, slot) != n0 + j) {
            fprintf(stderr, "FAIL: j=%d ntokens=%d expected=%d\n",
                    j, gemma4_engine_seq_ntokens(eng, slot), n0 + j); fails++;
        }
        // Byte-identical recurrent state vs the j-sequential reference.
        void *cmt = malloc(szj[j]);
        if (gemma4_engine_q35_state_save(eng, slot, cmt, n0 + j) != 0) {
            fprintf(stderr, "FAIL: commit save j=%d\n", j); return 1;
        }
        if (memcmp(cmt, refbuf[j], szj[j]) != 0) {
            size_t nf = szj[j] / 4, ndiff = 0; const float *a=(const float*)cmt,*b=(const float*)refbuf[j];
            for (size_t i = 0; i < nf; i++) if (a[i] != b[i]) ndiff++;
            fprintf(stderr, "FAIL: j=%d state NOT byte-identical (%zu/%zu floats differ)\n",
                    j, ndiff, nf);
            fails++;
        }
        // Next-token parity: the token produced by the last committed replay step (out_next[j-1])
        // must equal the reference sequential next token at that position (ref_next[j]).
        if (j > 0 && out_next[j-1] != ref_next[j]) {
            fprintf(stderr, "FAIL: j=%d commit next-token %d != ref %d\n",
                    j, out_next[j-1], ref_next[j]);
            fails++;
        }
        free(cmt);
        fprintf(stderr, "  j=%d: ntokens=%d state=%s\n", j, n0 + j,
                (fails == 0) ? "ok" : "see-above");
    }

    for (int j = 0; j <= K; j++) free(refbuf[j]);
    free(boundary);
    gemma4_engine_seq_remove(eng, slot);
    gemma4_engine_destroy(eng);

    if (fails) { printf("FAIL — qwen35 GDN rollback gate (%d checks failed)\n", fails); return 1; }
    printf("PASS — qwen35 GDN rollback gate: commit(j) byte-identical to j sequential decodes "
           "for all j in 0..%d (rewind j=0 and full-accept j=%d included)\n", K, K);
    return 0;
}
