// ABOUTME: D32 byte-identity harness — dumps B=32 greedy decode token stream so BASE (mixer
// ABOUTME: multi<8>) vs OPT (multi<16>) can be diffed, proving the K>16 mixer change is lossless.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "gemma4_kernels.cuh"

// Emit `NSTEP` greedy tokens for each of B seqs and print them as a flat stream. Deterministic
// (temp=0). Two builds that produce identical output over the K>16 decode mixer path are
// bitwise-identical on that path. Usage: <model> [NSTEP] [B].
int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "/opt/spark/models/models--Qwen--Qwen3.5-9B-FP8";
    int NSTEP = (argc > 2) ? atoi(argv[2]) : 24;
    int B     = (argc > 3) ? atoi(argv[3]) : 32;

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.90);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    int slot[64]; int32_t cur[64];
    for (int q = 0; q < B; q++) {
        int32_t prompt[6] = { 760, 6511, 314, 9338, 369, (int32_t)(1000 + 37*q) };
        int np = 5 + (q & 1);
        int32_t first = 0;
        slot[q] = gemma4_engine_seq_add(eng, prompt, np, &first, 0.f, 0, 0.f, 0.f, 0);
        if (slot[q] < 0) { fprintf(stderr, "seq_add failed (q=%d)\n", q); return 2; }
        cur[q] = first;
    }
    // Print first tokens then the decode stream.
    unsigned long long h = 1469598103934665603ULL;   // FNV-1a over the token stream
    for (int q = 0; q < B; q++) { h ^= (unsigned)cur[q]; h *= 1099511628211ULL; }
    for (int k = 0; k < NSTEP; k++) {
        int32_t nxt[64];
        if (gemma4_engine_step_batch(eng, slot, cur, B, nxt) != 0) { fprintf(stderr,"step failed\n"); return 2; }
        for (int q = 0; q < B; q++) { cur[q] = nxt[q]; h ^= (unsigned)nxt[q]; h *= 1099511628211ULL; }
    }
    printf("D32-BYTEIDENT B=%d NSTEP=%d streamhash=%016llx\n", B, NSTEP, h);
    for (int q = 0; q < B; q++) gemma4_engine_seq_remove(eng, slot[q]);
    gemma4_engine_destroy(eng);
    return 0;
}
