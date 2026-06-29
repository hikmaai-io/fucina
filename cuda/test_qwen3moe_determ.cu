// test_qwen3moe_determ.cu — is the MoE forward bit-deterministic across independent rows?
// Two identical sequences, BOTH driven by plain step_batch (NO spec). If they diverge,
// the MoE expert scatter-add (atomicAdd, nondeterministic float order) is the cause — which
// means the spec d=0 "mismatches" are that same noise, not a spec-path bug.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "gemma4_kernels.cuh"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf";
    setenv("FUCINA_PAGED_KV", "1", 1);
    int32_t prompt[] = { 785, 6722, 315, 9625, 374 };
    const int NP = 5, STEPS = 24;
    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.6);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }
    int32_t fa = 0, fb = 0;
    int a = gemma4_engine_seq_add(eng, prompt, NP, &fa, 0.0f, 0, 0.0f, 0.0f, 0);
    int b = gemma4_engine_seq_add(eng, prompt, NP, &fb, 0.0f, 0, 0.0f, 0.0f, 0);
    if (a < 0 || b < 0) { fprintf(stderr, "seq_add fail\n"); return 2; }
    printf("first: a=%d b=%d %s\n", fa, fb, fa == fb ? "(equal)" : "(DIFFER)");
    int32_t ca = fa, cb = fb; int agree = (fa == fb);
    for (int k = 0; k < STEPS; k++) {
        int32_t na = 0, nb = 0; int sa = a, sb = b;
        gemma4_engine_step_batch(eng, &sa, &ca, 1, &na);
        gemma4_engine_step_batch(eng, &sb, &cb, 1, &nb);
        if (na == nb) agree++;
        else printf("  step %d: seqA=%d seqB=%d DIVERGE\n", k, na, nb);
        ca = na; cb = nb;
    }
    printf("two plain step_batch seqs agree %d/%d\n", agree, STEPS + 1);
    printf("%s\n", agree == STEPS + 1 ? "DETERMINISTIC" : "NON-DETERMINISTIC (MoE atomicAdd scatter)");
    gemma4_engine_destroy(eng);
    return 0;
}
