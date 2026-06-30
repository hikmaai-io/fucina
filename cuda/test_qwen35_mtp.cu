// test_qwen35_mtp.cu — M6 LOSSLESS-spec gate for the Qwen3.5 single-MTP draft head.
//
// On the SAME official Qwen3.5-9B FP8 checkpoint (the only one that ships the 22 mtp.* tensors —
// the GGUF conversion drops the head), assert that the MTP-drafted speculative decode
// (qwen35_fp8_spec_greedy) emits the IDENTICAL token sequence to plain greedy decode
// (qwen35_fp8_forward_greedy = the M5-proven 8/8 backbone) — i.e. spec is LOSSLESS — AND that the
// draft accept-rate is > 0. This is the M6 gate: "step_batch_spec emits the IDENTICAL token
// sequence to plain step_batch, with draft accept-rate > 0 and stable" (single-sequence B=1, the
// unit that proves losslessness on this self-contained FP8 path, mirroring the M5 self-contained
// gate; the MTP head only changes how many backbone steps run per call, never which token).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "gemma4_kernels.cuh"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/models--Qwen--Qwen3.5-9B-FP8";

    int32_t in_ids[] = { 760, 6511, 314, 9338, 369 };   // "The capital of France is"
    const int NP = 5, NGEN = 24;

    void *m = qwen35_fp8_load(path);
    if (!m) { fprintf(stderr, "qwen35_fp8_load failed\n"); return 2; }

    int32_t plain[NGEN] = {0}, spec[NGEN] = {0};
    if (qwen35_fp8_forward_greedy(m, in_ids, NP, plain, NGEN) != 0) {
        fprintf(stderr, "qwen35_fp8_forward_greedy failed\n"); qwen35_fp8_free(m); return 2;
    }
    long drafted = 0, accepted = 0;
    int rc = qwen35_fp8_spec_greedy(m, in_ids, NP, spec, NGEN, &drafted, &accepted);
    qwen35_fp8_free(m);
    if (rc != 0) { fprintf(stderr, "qwen35_fp8_spec_greedy failed (rc=%d)\n", rc); return 2; }

    int identical = 1, first_mism = -1;
    printf("pos | plain  | spec\n");
    for (int k = 0; k < NGEN; k++) {
        int ok = (plain[k] == spec[k]);
        if (!ok && first_mism < 0) first_mism = k;
        identical &= ok;
        printf("%3d | %6d | %6d  %s\n", k, plain[k], spec[k], ok ? "" : "  <-- MISMATCH");
    }
    double rate = drafted > 0 ? (double)accepted / (double)drafted : 0.0;
    printf("lossless: %s (first mismatch %d)\n", identical ? "YES" : "NO", first_mism);
    printf("MTP drafts: drafted=%ld accepted=%ld accept_rate=%.3f\n", drafted, accepted, rate);

    int pass = identical && accepted > 0;
    printf("%s\n", pass
        ? "PASS — MTP spec LOSSLESS (== plain greedy) with accept-rate > 0 (qwen35 M6)"
        : "FAIL — qwen35 M6 MTP lossless-spec gate");
    return pass ? 0 : 1;
}
