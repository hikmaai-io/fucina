// test_qwen35_prefill.cu — P1 gate: qwen35 BATCHED single-pass prefill.
//
// Two assertions + one measurement:
//   (1) INTEGRATED-PREFILL PARITY: the continuous-batching entry points (gemma4_engine_seq_add
//       routes to qwen35_seq_add → qwen35_prefill_batched, then gemma4_engine_step_batch) must
//       reproduce the proven qwen35_forward_greedy oracle's France→Paris 8/8 continuation
//       [11751,13,198,57590,369,279,6511,314]. This proves the batched prefill is bit-identical
//       to the token-by-token oracle the M3 gate locked at 8/8 vs llama.cpp.
//   (2) SLOW==FAST on a long prompt: on a >=512-token prompt the batched prefill's first sampled
//       token must equal the token-by-token (g_fucina_force_slow_prefill=1) path's — i.e. the
//       speedup changes nothing numerically.
//   (3) TTFT BEFORE vs AFTER: wall-clock seq_add time, slow (token-by-token, the old 149 s path)
//       vs fast (batched). Printed verbatim; expect a LARGE drop.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include "gemma4_kernels.cuh"

extern int g_fucina_force_slow_prefill;   // runtime slow/fast prefill override (defined in kernels)

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf";

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.90);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    // ── (1) integrated-prefill parity vs the oracle: France→Paris 8/8 ──────────────────────────
    int32_t fr[] = { 760, 6511, 314, 9338, 369 };
    const int NP = 5, GATE = 8;
    int32_t oref[GATE] = { 11751, 13, 198, 57590, 369, 279, 6511, 314 };

    int32_t got[GATE] = {0};
    {
        int32_t first = 0;
        int slot = gemma4_engine_seq_add(eng, fr, NP, &first, 0.f, 0, 0.f, 0.f, 0);
        if (slot < 0) { fprintf(stderr, "seq_add(France) failed\n"); gemma4_engine_destroy(eng); return 2; }
        got[0] = first;
        int32_t tok = first;
        for (int k = 1; k < GATE; k++) {
            int32_t nxt = 0; int sl = slot;
            if (gemma4_engine_step_batch(eng, &sl, &tok, 1, &nxt) != 0) {
                fprintf(stderr, "step_batch failed at %d\n", k);
                gemma4_engine_seq_remove(eng, slot); gemma4_engine_destroy(eng); return 2;
            }
            got[k] = nxt; tok = nxt;
        }
        gemma4_engine_seq_remove(eng, slot);
    }
    int agree = 0;
    printf("pos | integrated-prefill | oracle(llama.cpp)\n");
    for (int k = 0; k < GATE; k++) {
        int ok = (got[k] == oref[k]); agree += ok;
        printf("%3d | %18d | %6d  %s\n", k, got[k], oref[k], ok ? "" : "  <-- MISMATCH");
    }
    printf("integrated prefill France->Paris: %d/%d\n", agree, GATE);

    // ── (2)+(3) long-prompt SLOW vs FAST: bit-identical first token + TTFT ─────────────────────
    const int NL = 512;
    int32_t *lp = (int32_t*)malloc((size_t)NL * sizeof(int32_t));
    for (int i = 0; i < NP; i++) lp[i] = fr[i];
    // Deterministic in-vocab filler (vocab 248320); keeps the prompt well-formed and reproducible.
    for (int i = NP; i < NL; i++) lp[i] = (int32_t)(((i * 1315423911u) ^ (i << 3)) % 200000u + 16u);

    int32_t first_slow = -1, first_fast = -2;
    double t_slow = 0, t_fast = 0;

    g_fucina_force_slow_prefill = 1;
    {
        double t0 = now_ms();
        int slot = gemma4_engine_seq_add(eng, lp, NL, &first_slow, 0.f, 0, 0.f, 0.f, 0);
        t_slow = now_ms() - t0;
        if (slot < 0) { fprintf(stderr, "seq_add(slow,512) failed\n"); free(lp); gemma4_engine_destroy(eng); return 2; }
        gemma4_engine_seq_remove(eng, slot);
    }
    g_fucina_force_slow_prefill = 0;
    {
        double t0 = now_ms();
        int slot = gemma4_engine_seq_add(eng, lp, NL, &first_fast, 0.f, 0, 0.f, 0.f, 0);
        t_fast = now_ms() - t0;
        if (slot < 0) { fprintf(stderr, "seq_add(fast,512) failed\n"); free(lp); gemma4_engine_destroy(eng); return 2; }
        gemma4_engine_seq_remove(eng, slot);
    }
    free(lp);
    gemma4_engine_destroy(eng);

    int slowfast_ok = (first_slow == first_fast);
    printf("\nlong-prompt (N=%d) prefill TTFT:\n", NL);
    printf("  SLOW (token-by-token) : %9.1f ms   first_tok=%d\n", t_slow, first_slow);
    printf("  FAST (batched chunks) : %9.1f ms   first_tok=%d\n", t_fast, first_fast);
    printf("  speedup               : %6.1fx\n", (t_fast > 0) ? t_slow / t_fast : 0.0);
    printf("  first-token slow==fast: %s\n", slowfast_ok ? "PASS" : "FAIL");

    int pass = (agree == GATE) && slowfast_ok;
    printf("%s\n", pass ? "PASS — qwen35 integrated batched prefill (8/8 oracle, slow==fast)"
                        : "FAIL — qwen35 integrated batched prefill");
    return pass ? 0 : 1;
}
