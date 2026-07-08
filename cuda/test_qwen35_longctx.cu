// test_qwen35_longctx.cu — P3 gate: qwen35 long-context greedy argmax parity vs llama.cpp.
//
// The Gated-DeltaNet fp32 recurrent state accumulates over every prefill+decode position, so
// long-context argmax flips are exactly where state drift would surface. This gate drives the
// INTEGRATED continuous-batching path (gemma4_engine_seq_add -> qwen35 batched single-pass
// prefill, then gemma4_engine_step_batch decode) on a ~1k-token AND a ~4k-token natural-text
// prompt, and asserts the greedy continuation matches the llama.cpp reference over >=32 tokens.
//
// Reference: generated with llama.cpp (libllama, commit c5fe75b9) greedy-argmax on the SAME
// /opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf, fed the SAME token ids (no BOS) as
// fucina — i.e. bit-identical inputs. The reference harness was validated to reproduce the M3
// France->Paris pin [11751,13,198,57590,...] before pinning these. Prompt ids live in the
// committed data files cuda/qwen35_longctx_{1k,4k}.ids; the 40-token references are pinned below.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include "gemma4_kernels.cuh"

// llama.cpp greedy-argmax reference, first 40 continuation tokens.
static const int32_t REF_1K[] = {
    9333, 220, 17, 13, 190657, 5101, 19053, 20239, 539, 264, 10286, 314, 54042, 421, 13523,
    513, 1990, 12689, 87006, 13, 4188, 4833, 310, 1301, 279, 8937, 314, 23883, 1452, 3000,
    506, 89687, 26, 3672, 2714, 310, 5357, 279, 23222, 2002 };
static const int32_t REF_4K[] = {
    9333, 220, 22, 13, 34460, 513, 8455, 23438, 13, 15767, 3213, 22892, 11, 264, 4129, 421,
    4816, 2166, 21030, 310, 3166, 264, 1016, 57858, 668, 76892, 264, 80329, 1472, 6235, 6730,
    11, 524, 539, 5176, 694, 539, 39604, 11, 13376 };
static const int NGEN = 40;

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

// Drive the integrated path; fill got[0..NGEN). Returns 0 on success.
static int run_integrated(const char *path, const std::vector<int32_t> &prompt, int32_t *got) {
    int n_prompt = (int)prompt.size();
    uint32_t ctx = (uint32_t)(n_prompt + NGEN + 64);
    if (ctx < 4096) ctx = 4096;
    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, ctx, 0, 0.90);
    if (!eng) { fprintf(stderr, "create failed (ctx=%u)\n", ctx); return 2; }

    int32_t first = 0;
    int slot = gemma4_engine_seq_add(eng, prompt.data(), n_prompt, &first, 0.f, 0, 0.f, 0.f, 0);
    if (slot < 0) { fprintf(stderr, "seq_add failed\n"); gemma4_engine_destroy(eng); return 2; }
    got[0] = first;
    int32_t tok = first;
    for (int k = 1; k < NGEN; k++) {
        int32_t nxt = 0; int sl = slot;
        if (gemma4_engine_step_batch(eng, &sl, &tok, 1, &nxt) != 0) {
            fprintf(stderr, "step_batch failed at %d\n", k);
            gemma4_engine_seq_remove(eng, slot); gemma4_engine_destroy(eng); return 2;
        }
        got[k] = nxt; tok = nxt;
    }
    gemma4_engine_seq_remove(eng, slot);
    gemma4_engine_destroy(eng);
    return 0;
}

static int check(const char *tag, int n_prompt, const int32_t *got, const int32_t *ref) {
    int agree = 0;
    printf("\n== %s (prompt=%d tok, NGEN=%d) ==\n", tag, n_prompt, NGEN);
    printf("  pos | fucina | llama.cpp\n");
    for (int k = 0; k < NGEN; k++) {
        int ok = (got[k] == ref[k]); agree += ok;
        printf("  %3d | %6d | %6d  %s\n", k, got[k], ref[k], ok ? "" : "  <-- MISMATCH");
    }
    printf("  %s: %d/%d argmax parity\n", tag, agree, NGEN);
    return agree == NGEN;
}

int main(int argc, char **argv) {
    const char *path  = (argc > 1) ? argv[1] : "/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf";
    const char *f1k   = (argc > 2) ? argv[2] : "cuda/qwen35_longctx_1k.ids";
    const char *f4k   = (argc > 3) ? argv[3] : "cuda/qwen35_longctx_4k.ids";

    std::vector<int32_t> p1 = read_ids(f1k);
    std::vector<int32_t> p4 = read_ids(f4k);
    printf("loaded prompts: 1k=%zu tokens, 4k=%zu tokens\n", p1.size(), p4.size());

    int32_t g1[NGEN] = {0}, g4[NGEN] = {0};
    if (run_integrated(path, p1, g1) != 0) return 2;
    if (run_integrated(path, p4, g4) != 0) return 2;

    int ok1 = check("~1k context", (int)p1.size(), g1, REF_1K);
    int ok4 = check("~4k context", (int)p4.size(), g4, REF_4K);

    int pass = ok1 && ok4;
    printf("\n%s\n", pass ? "PASS — qwen35 long-context argmax parity vs llama.cpp (1k AND 4k, 40/40)"
                          : "FAIL — qwen35 long-context argmax parity vs llama.cpp");
    return pass ? 0 : 1;
}
