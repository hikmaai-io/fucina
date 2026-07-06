// test_qwen35_chunk_parity.cu — chunked-prefill CONTINUATION (base>0) parity + timing gate.
//
// The base==0 prefill tile is tensor-core (q35_full_attn_tc) but every base>0 continuation
// chunk runs FULL-layer attention over the whole fp16 K/V cache. This harness pins the
// token-level contract of that continuation path against the one-shot prefill:
//   (A) one-shot   : seq_add(prompt)                          -> first + 24 greedy tokens
//   (B) chunked/sc : seq_open + prefill_chunk(1024, rest) with g_fucina_q35_scalar_cont_attn=1
//                    (the scalar qwen35_b_attn_kernel continuation)   -> first + 24 tokens
//   (C) chunked/tc : same split with the default tensor-core continuation -> first + 24 tokens
// It prints the first token of each run, the 25-token agreement of B and C against A (the
// chunked-vs-one-shot bar) and of C against B (TC-vs-scalar), plus the wall time of the
// continuation chunk under both paths. Gate (see docs in the TC-continuation commit):
// C's agreement with A must be >= B's, and C's first token must match A's whenever B's does.
// NOTE: chunked-vs-one-shot was NEVER bitwise for qwen35 (chunk 1 is TC, continuation was
// scalar), so the bar is token-level agreement, exactly like the TC base==0 precedent.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include "gemma4_kernels.cuh"

extern int g_fucina_q35_scalar_cont_attn;   // test override (defined in gemma4_kernels.cu)

static const int NGEN = 24;      // greedy continuation tokens after the first sampled one
static const int NTOK = NGEN + 1;
static const int PROMPT_LEN = 2400;
static const int CHUNK1 = 1024;  // serving chunk size (bridge.go PrefillChunkHint for MoE)

// Parse whitespace/comma-separated token ids from a file, skipping '#'-comment lines.
static std::vector<int32_t> read_ids(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    std::vector<int32_t> ids; char line[8192];
    while (fgets(line, sizeof(line), f)) {
        char *s = line; while (*s == ' ' || *s == '\t') s++;
        if (*s == '#' || *s == '\0' || *s == '\n') continue;
        const char *p = s;
        while (*p) {
            if ((*p >= '0' && *p <= '9') || (*p == '-' && p[1] >= '0' && p[1] <= '9')) {
                char *end = nullptr; long v = strtol(p, &end, 10); ids.push_back((int32_t)v); p = end;
            } else p++;
        }
    }
    fclose(f);
    return ids;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int decode_n(gemma4_engine_t *eng, int slot, int32_t first, int32_t *out) {
    out[0] = first;
    int32_t cur = first;
    for (int k = 0; k < NGEN; k++) {
        int32_t nxt;
        if (gemma4_engine_step_batch(eng, &slot, &cur, 1, &nxt) != 0) return -1;
        out[k + 1] = nxt; cur = nxt;
    }
    return 0;
}

// One-shot: seq_add the whole prompt, then NGEN greedy steps.
static int run_oneshot(gemma4_engine_t *eng, const std::vector<int32_t> &p, int32_t *out) {
    int32_t first = 0;
    double t0 = now_ms();
    int slot = gemma4_engine_seq_add(eng, p.data(), (int)p.size(), &first, 0.f, 0, 0.f, 0.f, 0);
    double t1 = now_ms();
    if (slot < 0) { fprintf(stderr, "seq_add failed\n"); return -1; }
    printf("ONESHOT prefill %.1f ms first=%d\n", t1 - t0, first);
    int rc = decode_n(eng, slot, first, out);
    gemma4_engine_seq_remove(eng, slot);
    return rc;
}

// Chunked: seq_open, prefill CHUNK1 tokens, then the rest (base>0 continuation), NGEN steps.
static int run_chunked(gemma4_engine_t *eng, const std::vector<int32_t> &p, const char *tag,
                       int32_t *out) {
    int n = (int)p.size();
    int slot = gemma4_engine_seq_open(eng, 0.f, 0, 0.f, 0.f, 0);
    if (slot < 0) { fprintf(stderr, "seq_open failed\n"); return -1; }
    double t0 = now_ms();
    if (gemma4_engine_seq_prefill_chunk(eng, slot, p.data(), CHUNK1, 0, NULL) != 0) {
        fprintf(stderr, "prefill_chunk 1 failed\n"); gemma4_engine_seq_remove(eng, slot); return -1;
    }
    double t1 = now_ms();
    int32_t first = 0;
    if (gemma4_engine_seq_prefill_chunk(eng, slot, p.data() + CHUNK1, n - CHUNK1, 1, &first) != 0) {
        fprintf(stderr, "prefill_chunk 2 failed\n"); gemma4_engine_seq_remove(eng, slot); return -1;
    }
    double t2 = now_ms();
    printf("CHUNKED[%s] chunk1(T=%d,base=0) %.1f ms  chunk2(T=%d,base=%d) %.1f ms  first=%d\n",
           tag, CHUNK1, t1 - t0, n - CHUNK1, CHUNK1, t2 - t1, first);
    int rc = decode_n(eng, slot, first, out);
    gemma4_engine_seq_remove(eng, slot);
    return rc;
}

static int agree(const int32_t *a, const int32_t *b) {
    int n = 0;
    for (int i = 0; i < NTOK; i++) if (a[i] == b[i]) n++;
    return n;
}

static void dump(const char *tag, const int32_t *t) {
    printf("%s:", tag);
    for (int i = 0; i < NTOK; i++) printf(" %d", t[i]);
    printf("\n");
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "/opt/spark/models/models--Qwen--Qwen3.5-35B-A3B-FP8";
    const char *ids  = (argc > 2) ? argv[2] : "cuda/qwen35_longctx_4k.ids";
    std::vector<int32_t> p = read_ids(ids);
    if ((int)p.size() < PROMPT_LEN) { fprintf(stderr, "prompt too short (%zu)\n", p.size()); return 2; }
    p.resize(PROMPT_LEN);

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 8192, 0, 0.90);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    int32_t A[NTOK], B[NTOK], C[NTOK];
    if (run_oneshot(eng, p, A) != 0) return 2;
    g_fucina_q35_scalar_cont_attn = 1;
    if (run_chunked(eng, p, "scalar", B) != 0) return 2;
    g_fucina_q35_scalar_cont_attn = 0;
    if (run_chunked(eng, p, "tc", C) != 0) return 2;
    gemma4_engine_destroy(eng);

    dump("TOKENS ONESHOT   ", A);
    dump("TOKENS CHUNK-SCAL", B);
    dump("TOKENS CHUNK-TC  ", C);
    int ba = agree(B, A), ca = agree(C, A), cb = agree(C, B);
    printf("PARITY first: oneshot=%d scalar=%d tc=%d\n", A[0], B[0], C[0]);
    printf("PARITY agree/%d: scalar-vs-oneshot=%d tc-vs-oneshot=%d tc-vs-scalar=%d\n",
           NTOK, ba, ca, cb);
    // Gate: the TC continuation must track one-shot at least as well as the scalar one.
    int pass = (ca >= ba) && (C[0] == A[0] || B[0] != A[0]);
    printf("CHUNK-PARITY %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
