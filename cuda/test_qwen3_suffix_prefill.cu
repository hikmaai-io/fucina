// test_qwen3_suffix_prefill.cu — Stage 9 losslessness of the COMPUTE-BOUND base-offset
// (suffix) prefill (paged_prefill_qwen3 base>0 mode).
//
// When a request hits the cross-request prefix cache, the divergent suffix is now prefilled
// by the single-pass GEMM path at a nonzero base (history attention through the paged block
// table) instead of the bandwidth-bound chunked decode_multiseq path. This MUST be lossless:
// a sequence built as prefix-then-suffix must produce a greedy token stream bit-identical to
// one built one-shot.
//
//   (a) ONE-SHOT reference (cache OFF): seq_add(full prompt) → first token + STEPS continuations.
//   (b) PREFIX+SUFFIX (cache ON): prime the cache with the full prompt, then re-run it so the
//       2nd request ADOPTS the shared prefix blocks and prefills ONLY the suffix via the new
//       GEMM path (hit_blocks must increase). Its STEPS+1 greedy stream must == (a).
//
// Runs on BOTH Qwen3-8B dense and Qwen3-30B-A3B MoE.
//   make qwen3-suffix-test            (defaults: dense + MoE GGUFs)
//   /tmp/fucina_qwen3_suffix <dense.gguf> [<moe.gguf>]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "gemma4_kernels.cuh"
#include "paged_kv.h"   // PAGED_KV_BLOCK_TOKENS

static const int STEPS = 24;   // >= 16 required by the gate

// seq_add(prompt) then STEPS greedy decode steps → out[0..STEPS].
static int run_seq(gemma4_engine_t *eng, const int32_t *prompt, int NP, int32_t *out) {
    int32_t f = 0;
    int s = gemma4_engine_seq_add(eng, prompt, NP, &f, 0.0f, 0, 0.0f, 0.0f, 0);
    if (s < 0) return -1;
    out[0] = f;
    int32_t cur = f;
    for (int k = 0; k < STEPS; k++) {
        int32_t nb = 0; int sl = s;
        if (gemma4_engine_step_batch(eng, &sl, &cur, 1, &nb) != 0) { gemma4_engine_seq_remove(eng, s); return -1; }
        out[1 + k] = nb; cur = nb;
    }
    gemma4_engine_seq_remove(eng, s);
    return s;
}

static int cmp_stream(const char *tag, const int32_t *a, const int32_t *b) {
    for (int i = 0; i <= STEPS; i++) {
        if (a[i] != b[i]) {
            fprintf(stderr, "  FAIL %s: token %d differs (one-shot %d vs suffix %d)\n", tag, i, a[i], b[i]);
            return 1;
        }
    }
    printf("  OK  %s: %d-token greedy stream bit-identical (one-shot == prefix+suffix)\n", tag, STEPS + 1);
    return 0;
}

// One model: prove the GEMM suffix prefill is lossless vs one-shot. NP chosen so >=1 full
// 256-token block is shareable (adopted) while a non-trivial suffix (>16 tok → multi-chunk
// attention) is recomputed at base>0.
static int run_model(const char *path, double mem_util, int NP) {
    printf("=== %s (NP=%d) ===\n", path, NP);
    int32_t *prompt = (int32_t*)malloc((size_t)NP*sizeof(int32_t));
    for (int i = 0; i < NP; i++)
        prompt[i] = (int32_t)(((i * 1103515245u + 12345u) >> 8) % 30000u + 100u);

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, mem_util);
    if (!eng) { fprintf(stderr, "  create failed: %s\n", path); free(prompt); return 2; }

    int rc = 0;
    int32_t one_shot[1 + STEPS], suffix[1 + STEPS];

    // (a) ONE-SHOT reference — cache OFF (fresh GEMM prefill, base=0).
    gemma4_engine_set_prefix_cache(eng, 0);
    if (run_seq(eng, prompt, NP, one_shot) < 0) { fprintf(stderr, "  one-shot run failed\n"); rc = 2; goto done; }
    printf("  one-shot reference captured (first token %d)\n", one_shot[0]);

    // (b) PREFIX+SUFFIX — cache ON. Run #1 primes (registers blocks); run #2 adopts the
    //     shared prefix and prefills the suffix via the base>0 GEMM path.
    gemma4_engine_set_prefix_cache(eng, 1);
    {
        int32_t prime[1 + STEPS];
        if (run_seq(eng, prompt, NP, prime) < 0) { fprintf(stderr, "  prime run failed\n"); rc = 2; goto done; }
        uint64_t lk0, hb0, cb0, ev0; gemma4_engine_prefix_cache_stats(eng, &lk0, &hb0, &cb0, &ev0);
        if (run_seq(eng, prompt, NP, suffix) < 0) { fprintf(stderr, "  suffix run failed\n"); rc = 2; goto done; }
        uint64_t lk1, hb1, cb1, ev1; gemma4_engine_prefix_cache_stats(eng, &lk1, &hb1, &cb1, &ev1);
        long hit = (long)(hb1 - hb0);
        if (hit <= 0) {
            fprintf(stderr, "  FAIL: suffix run adopted no cached blocks (hit delta=%ld) — base>0 path did not engage\n", hit);
            rc |= 1;
        } else {
            printf("  OK  suffix run adopted %ld cached blocks → %d-token GEMM suffix prefill at base %d\n",
                   hit, NP - (int)hit * PAGED_KV_BLOCK_TOKENS, (int)hit * PAGED_KV_BLOCK_TOKENS);
        }
        rc |= cmp_stream("suffix-prefill", one_shot, suffix);
    }

done:
    gemma4_engine_destroy(eng);
    free(prompt);
    return rc;
}

int main(int argc, char **argv) {
    const char *dense = (argc > 1) ? argv[1] : "/opt/spark/models/Qwen3-8B-abliterated.Q4_K_M.gguf";
    const char *moe   = (argc > 2) ? argv[2] : "/opt/spark/models/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf";
    setenv("FUCINA_PAGED_KV", "1", 1);

    int rc = 0;
    // Dense: 600-tok prompt → adopt 2 full blocks (512), suffix 88 (>16 → 6 attention chunks).
    rc |= run_model(dense, 0.90, 600);
    // MoE: same shape, mem-util 0.6 per the model's memory budget.
    rc |= run_model(moe, 0.60, 600);

    if (rc == 0) printf("PASS — base-offset GEMM suffix prefill is lossless (one-shot == prefix+suffix) on dense + MoE\n");
    else         printf("FAIL — suffix prefill diverged from one-shot reference\n");
    return rc ? 1 : 0;
}
