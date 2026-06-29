// test_qwen3_prefix.cu — losslessness of the cross-request prefix cache (RadixAttention).
//
// The prefix cache lets a request reuse another request's already-computed KV for a
// shared FULL-block prompt prefix, prefilling only the divergent suffix. This MUST be
// lossless: the greedy token stream of a cache-served request is bit-identical to a
// cold request that recomputes the whole prompt.
//
// (1) COLD reference: cache OFF, full prompt prefilled, record first token + N decode tokens.
// (2) WARM sequential: cache ON, same prompt run twice; both streams == cold; the 2nd run
//     adopts cached blocks (hit_blocks increases), proving the suffix-only path ran.
// (3) CONCURRENT: two same-prefix sequences live at once share blocks; both streams == cold.
//
//   make qwen3-prefix-test  (model arg optional; defaults to the Qwen3-8B Q4_K_M GGUF)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "gemma4_kernels.cuh"
#include "paged_kv.h"   // PAGED_KV_BLOCK_TOKENS

static const int STEPS = 16;

// Run one sequence to completion: seq_add(prompt) then STEPS greedy decode steps.
// Fills out[0..STEPS] (first token + STEPS continuations). Returns slot used, <0 on error.
static int run_seq(gemma4_engine_t *eng, const int32_t *prompt, int NP, int32_t *out) {
    int32_t f = 0;
    int s = gemma4_engine_seq_add(eng, prompt, NP, &f, 0.0f, 0, 0.0f, 0.0f, 0);
    if (s < 0) return -1;
    out[0] = f;
    int32_t cur = f;
    for (int k = 0; k < STEPS; k++) {
        int32_t nb = 0; int sl = s;
        if (gemma4_engine_step_batch(eng, &sl, &cur, 1, &nb) != 0) return -1;
        out[1 + k] = nb; cur = nb;
    }
    gemma4_engine_seq_remove(eng, s);
    return s;
}

static int cmp_stream(const char *tag, const int32_t *a, const int32_t *b) {
    for (int i = 0; i <= STEPS; i++) {
        if (a[i] != b[i]) {
            fprintf(stderr, "FAIL %s: token %d differs (cold %d vs %d)\n", tag, i, a[i], b[i]);
            return 1;
        }
    }
    printf("  OK  %s: %d-token greedy stream bit-identical to cold\n", tag, STEPS + 1);
    return 0;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3-8B-abliterated.Q4_K_M.gguf";
    setenv("FUCINA_PAGED_KV", "1", 1);
    // Shrink the global pool so the eviction section below actually forces LRU
    // reclaim with a handful of distinct prompts (48 glob blocks at maxseqs=2).
    setenv("FUCINA_PAGED_MAXSEQS", "2", 1);

    // A deterministic >=512-token prompt (>= 2 full 256-token blocks, so >=1 block is
    // shareable while a non-trivial suffix is recomputed). Values are arbitrary valid ids.
    const int NP = 600;
    int32_t prompt[NP];
    for (int i = 0; i < NP; i++) prompt[i] = (int32_t)(((i * 1103515245u + 12345u) >> 8) % 30000u + 100u);

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.90);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    int rc = 0;
    int32_t cold[1 + STEPS], warm1[1 + STEPS], warm2[1 + STEPS];

    // (1) COLD reference — cache forced OFF.
    gemma4_engine_set_prefix_cache(eng, 0);
    if (run_seq(eng, prompt, NP, cold) < 0) { fprintf(stderr, "cold run failed\n"); return 2; }
    printf("cold reference captured (first token %d)\n", cold[0]);

    // (2) WARM sequential — cache ON. Run 1 is the cold producer (registers blocks);
    //     run 2 must ADOPT them (hit_blocks > 0) and stay bit-identical.
    gemma4_engine_set_prefix_cache(eng, 1);
    uint64_t lk0, hb0, cb0, ev0;
    gemma4_engine_prefix_cache_stats(eng, &lk0, &hb0, &cb0, &ev0);

    if (run_seq(eng, prompt, NP, warm1) < 0) { fprintf(stderr, "warm run 1 failed\n"); return 2; }
    rc |= cmp_stream("warm#1 (producer)", cold, warm1);

    uint64_t lk1, hb1, cb1, ev1;
    gemma4_engine_prefix_cache_stats(eng, &lk1, &hb1, &cb1, &ev1);
    if (run_seq(eng, prompt, NP, warm2) < 0) { fprintf(stderr, "warm run 2 failed\n"); return 2; }
    rc |= cmp_stream("warm#2 (cache hit)", cold, warm2);

    uint64_t lk2, hb2, cb2, ev2;
    gemma4_engine_prefix_cache_stats(eng, &lk2, &hb2, &cb2, &ev2);
    long hit_run2 = (long)(hb2 - hb1);
    if (hit_run2 <= 0) {
        fprintf(stderr, "FAIL: warm#2 adopted no cached blocks (hit_blocks delta=%ld) — "
                        "suffix-only path did not engage\n", hit_run2);
        rc |= 1;
    } else {
        printf("  OK  warm#2 adopted %ld cached blocks (%d tokens of prefill skipped)\n",
               hit_run2, (int)hit_run2 * PAGED_KV_BLOCK_TOKENS);
    }
    printf("stats: lookups=%llu hit_blocks=%llu cached_blocks=%llu evictions=%llu\n",
           (unsigned long long)lk2, (unsigned long long)hb2,
           (unsigned long long)cb2, (unsigned long long)ev2);

    // (3) CONCURRENT — two same-prefix sequences live at the same time share blocks.
    {
        int32_t fa = 0, fb = 0;
        int sa = gemma4_engine_seq_add(eng, prompt, NP, &fa, 0.0f, 0, 0.0f, 0.0f, 0);
        int sb = gemma4_engine_seq_add(eng, prompt, NP, &fb, 0.0f, 0, 0.0f, 0.0f, 0);
        if (sa < 0 || sb < 0) { fprintf(stderr, "concurrent seq_add failed\n"); return 2; }
        int32_t ca[1 + STEPS], cb_[1 + STEPS];
        ca[0] = fa; cb_[0] = fb;
        int32_t ua = fa, ub = fb;
        for (int k = 0; k < STEPS; k++) {
            int slots[2] = { sa, sb }; int32_t in[2] = { ua, ub }; int32_t out[2] = { 0, 0 };
            if (gemma4_engine_step_batch(eng, slots, in, 2, out) != 0) { fprintf(stderr, "concurrent step failed\n"); return 2; }
            ca[1 + k] = out[0]; cb_[1 + k] = out[1]; ua = out[0]; ub = out[1];
        }
        gemma4_engine_seq_remove(eng, sa);
        gemma4_engine_seq_remove(eng, sb);
        rc |= cmp_stream("concurrent A", cold, ca);
        rc |= cmp_stream("concurrent B", cold, cb_);
    }

    // (4) EVICTION — fill the (shrunk) pool with distinct cached prompts until LRU
    //     reclaim recycles physical blocks, then re-run the first (now-evicted) prompt
    //     and assert it recomputes bit-identically. Catches device-side corruption from
    //     reusing a recycled block id.
    {
        const int ND = 30;             // distinct 2-block prompts; > pool/2 forces eviction
        auto mkd = [&](int d, int32_t *p) {
            for (int i = 0; i < NP; i++)
                p[i] = (int32_t)(((((uint32_t)d * 2654435761u) ^ (i * 1103515245u + 12345u)) >> 8) % 30000u + 100u);
        };
        int32_t d0[NP];   mkd(0, d0);
        int32_t cold_d0[1 + STEPS];
        gemma4_engine_set_prefix_cache(eng, 0);
        if (run_seq(eng, d0, NP, cold_d0) < 0) { fprintf(stderr, "evict cold d0 failed\n"); return 2; }
        gemma4_engine_set_prefix_cache(eng, 1);

        uint64_t lkA, hbA, cbA, evA; gemma4_engine_prefix_cache_stats(eng, &lkA, &hbA, &cbA, &evA);
        int32_t tmp[1 + STEPS], buf[NP];
        for (int d = 0; d < ND; d++) { mkd(d, buf); if (run_seq(eng, buf, NP, tmp) < 0) { fprintf(stderr, "evict run %d failed\n", d); return 2; } }
        uint64_t lkB, hbB, cbB, evB; gemma4_engine_prefix_cache_stats(eng, &lkB, &hbB, &cbB, &evB);
        long evicted = (long)(evB - evA);
        if (evicted <= 0) {
            fprintf(stderr, "FAIL: eviction never triggered (evictions delta=%ld) — pool too large for the stress\n", evicted);
            rc |= 1;
        } else {
            printf("  OK  forced %ld LRU evictions across %d distinct prompts\n", evicted, ND);
        }
        // d0 is the oldest released entry → evicted; re-running it must recompute losslessly.
        int32_t warm_d0[1 + STEPS];
        if (run_seq(eng, d0, NP, warm_d0) < 0) { fprintf(stderr, "evict re-run d0 failed\n"); return 2; }
        rc |= cmp_stream("post-evict d0", cold_d0, warm_d0);
    }

    gemma4_engine_destroy(eng);
    if (rc == 0) printf("PASS — prefix cache is lossless (cold == cache-served, concurrent shared, post-eviction)\n");
    else         printf("FAIL — prefix cache diverged from cold reference\n");
    return rc ? 1 : 0;
}
