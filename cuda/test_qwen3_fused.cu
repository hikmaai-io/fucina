// test_qwen3_fused.cu — Stage 18 losslessness of FUSED prefill+decode.
//
// gemma4_engine_step_batch_fused mixes B_dec DECODE rows and pf_len PREFILL-chunk rows of ONE
// prefilling sequence into a SINGLE batched forward, so an arriving prompt's prefill never blocks
// the active sequences' decode. This MUST be byte-identical to running the two halves separately:
//
//   (1) LOSSLESS-PREFILL (cardinal): a sequence prefilled via the FUSED path while co-batched with
//       N>=2 unrelated DECODE rows must produce the SAME first generated token and a >=16-token
//       greedy continuation as the SAME sequence prefilled standalone (seq_open + seq_prefill_chunk,
//       no co-batch). Identical first-token + continuation transitively proves the paged KV is
//       position-for-position identical (any KV divergence would surface in the greedy stream).
//   (2) LOSSLESS-DECODE: the co-batched decode rows must be byte-identical to a plain step_batch of
//       those same rows WITHOUT the prefill (the prefill rows ride a different block table and cannot
//       perturb the decode rows; rng_off is NULL so each decode row draws the same sampler index).
//
// Runs on BOTH Qwen3-8B dense and Qwen3-30B-A3B MoE.
//   make qwen3-fused-test            (defaults: dense + MoE GGUFs)
//   /tmp/fucina_qwen3_fused <dense.gguf> [<moe.gguf>]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "gemma4_kernels.cuh"

static const int STEPS  = 20;   // >=16 continuation tokens after prefill
static const int NDEC   = 2;    // co-batched decode rows (N>=2)
static const int PFLEN  = 40;   // prefilling prompt length → multiple fused passes
static const int DECLEN = 8;    // decoy prompt length
static const int L      = 14;   // prefill tokens per fused pass (<= GEMMA4_MAX_SEQS - NDEC)
static const int NPASS  = (PFLEN + L - 1) / L;

static void mkprompt(int32_t *p, int n, uint32_t seed) {
    for (int i = 0; i < n; i++)
        p[i] = (int32_t)((((uint32_t)i * 1103515245u + 12345u + seed * 2654435761u) >> 8) % 30000u + 100u);
}

// Reference (no fusion): seq_add the NDEC decoys, then NPASS plain step_batch(B=NDEC) decode steps.
// ref_dec[d][k] is decoy d's token at step k. Slots are freed before returning.
static int decoy_decode_ref(gemma4_engine_t *eng, int32_t dec_prompt[NDEC][DECLEN],
                            int32_t ref_dec[NDEC][NPASS]) {
    int dslot[NDEC]; int32_t cur[NDEC];
    for (int d = 0; d < NDEC; d++) {
        int32_t f = 0;
        dslot[d] = gemma4_engine_seq_add(eng, dec_prompt[d], DECLEN, &f, 0.0f, 0, 0.0f, 0.0f, 0);
        if (dslot[d] < 0) { fprintf(stderr, "  decoy ref: seq_add failed\n"); return 2; }
        cur[d] = f;
    }
    for (int k = 0; k < NPASS; k++) {
        int sl[NDEC]; int32_t in[NDEC], out[NDEC];
        for (int d = 0; d < NDEC; d++) { sl[d] = dslot[d]; in[d] = cur[d]; }
        if (gemma4_engine_step_batch(eng, sl, in, NDEC, out) != 0) { fprintf(stderr, "  decoy ref: step_batch failed\n"); return 2; }
        for (int d = 0; d < NDEC; d++) { ref_dec[d][k] = out[d]; cur[d] = out[d]; }
    }
    for (int d = 0; d < NDEC; d++) gemma4_engine_seq_remove(eng, dslot[d]);
    return 0;
}

// Reference (no fusion): seq_open + seq_prefill_chunk(whole prompt, L-token chunks), then STEPS
// greedy decode. ref_pf[0] is the first generated token, ref_pf[1..STEPS] the continuation.
static int prefill_ref(gemma4_engine_t *eng, const int32_t *pf_prompt, int32_t ref_pf[1 + STEPS]) {
    int ps = gemma4_engine_seq_open(eng, 0.0f, 0, 0.0f, 0.0f, 0);
    if (ps < 0) { fprintf(stderr, "  prefill ref: seq_open failed\n"); return 2; }
    int done = 0; int32_t first = 0;
    while (done < PFLEN) {
        int c = PFLEN - done; if (c > L) c = L;
        int last = (done + c >= PFLEN);
        if (gemma4_engine_seq_prefill_chunk(eng, ps, pf_prompt + done, c, last, &first) != 0) {
            fprintf(stderr, "  prefill ref: seq_prefill_chunk failed\n"); gemma4_engine_seq_remove(eng, ps); return 2;
        }
        done += c;
    }
    ref_pf[0] = first; int32_t cur = first;
    for (int k = 0; k < STEPS; k++) {
        int sl = ps; int32_t o = 0;
        if (gemma4_engine_step_batch(eng, &sl, &cur, 1, &o) != 0) { fprintf(stderr, "  prefill ref: decode failed\n"); gemma4_engine_seq_remove(eng, ps); return 2; }
        ref_pf[1 + k] = o; cur = o;
    }
    gemma4_engine_seq_remove(eng, ps);
    return 0;
}

// Fused run: seq_add the NDEC decoys (mid-decode), seq_open the prefilling seq, then drive the
// prompt in via step_batch_fused co-batched with the decoy decode; capture the decoys' tokens
// (fz_dec) and the prefilled seq's first token + continuation (fz_pf).
static int fused_run(gemma4_engine_t *eng, int32_t dec_prompt[NDEC][DECLEN], const int32_t *pf_prompt,
                     int32_t fz_dec[NDEC][NPASS], int32_t fz_pf[1 + STEPS]) {
    int ds[NDEC]; int32_t dc[NDEC];
    for (int d = 0; d < NDEC; d++) {
        int32_t f = 0;
        ds[d] = gemma4_engine_seq_add(eng, dec_prompt[d], DECLEN, &f, 0.0f, 0, 0.0f, 0.0f, 0);
        if (ds[d] < 0) { fprintf(stderr, "  fused: seq_add failed\n"); return 2; }
        dc[d] = f;
    }
    int ps = gemma4_engine_seq_open(eng, 0.0f, 0, 0.0f, 0.0f, 0);
    if (ps < 0) { fprintf(stderr, "  fused: seq_open failed\n"); return 2; }

    int done = 0, k = 0; int32_t pf_first = 0;
    while (done < PFLEN) {
        int c = PFLEN - done; if (c > L) c = L;
        int last = (done + c >= PFLEN);
        int dsl[NDEC]; int32_t din[NDEC], dout[NDEC]; int dlen[NDEC]; int32_t pff = 0;
        for (int d = 0; d < NDEC; d++) { dsl[d] = ds[d]; din[d] = dc[d]; }
        int rc = gemma4_engine_step_batch_fused(eng, dsl, din, NDEC, ps, pf_prompt + done, c, last,
                                                dout, dlen, &pff);
        if (rc != 0) { fprintf(stderr, "  fused: step_batch_fused failed (rc %d)\n", rc); return 2; }
        for (int d = 0; d < NDEC; d++) { fz_dec[d][k] = dout[d]; dc[d] = dout[d]; }
        if (last) pf_first = pff;
        done += c; k++;
    }

    // Decode the prefilled seq ALONE for STEPS (matches the prefill reference which was P alone).
    fz_pf[0] = pf_first; int32_t cur = pf_first;
    for (int kk = 0; kk < STEPS; kk++) {
        int sl = ps; int32_t o = 0;
        if (gemma4_engine_step_batch(eng, &sl, &cur, 1, &o) != 0) { fprintf(stderr, "  fused: post-decode failed\n"); return 2; }
        fz_pf[1 + kk] = o; cur = o;
    }
    for (int d = 0; d < NDEC; d++) gemma4_engine_seq_remove(eng, ds[d]);
    gemma4_engine_seq_remove(eng, ps);
    return 0;
}

static int run_model(const char *path, double mem_util) {
    printf("=== %s ===\n", path);
    int32_t pf_prompt[PFLEN]; mkprompt(pf_prompt, PFLEN, 7);
    int32_t dec_prompt[NDEC][DECLEN];
    for (int d = 0; d < NDEC; d++) mkprompt(dec_prompt[d], DECLEN, 100 + (uint32_t)d);

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, mem_util);
    if (!eng) { fprintf(stderr, "  create failed: %s\n", path); return 2; }
    gemma4_engine_set_prefix_cache(eng, 0);   // isolate fusion from cross-request prefix cache

    int rc = 0;
    int32_t ref_dec[NDEC][NPASS], fz_dec[NDEC][NPASS];
    int32_t ref_pf[1 + STEPS], fz_pf[1 + STEPS];

    if ((rc = decoy_decode_ref(eng, dec_prompt, ref_dec)) != 0) { gemma4_engine_destroy(eng); return rc; }
    if ((rc = prefill_ref(eng, pf_prompt, ref_pf)) != 0)        { gemma4_engine_destroy(eng); return rc; }
    if ((rc = fused_run(eng, dec_prompt, pf_prompt, fz_dec, fz_pf)) != 0) { gemma4_engine_destroy(eng); return rc; }

    // (2) LOSSLESS-DECODE
    int dec_bad = 0;
    for (int d = 0; d < NDEC && !dec_bad; d++)
        for (int k = 0; k < NPASS; k++)
            if (fz_dec[d][k] != ref_dec[d][k]) {
                fprintf(stderr, "  FAIL LOSSLESS-DECODE: row %d step %d differs (fused %d vs plain %d)\n",
                        d, k, fz_dec[d][k], ref_dec[d][k]); dec_bad = 1; break;
            }
    if (dec_bad) rc |= 1;
    else printf("  OK  LOSSLESS-DECODE: %d co-batched decode rows byte-identical over %d fused passes\n", NDEC, NPASS);

    // (1) LOSSLESS-PREFILL
    int pf_bad = 0;
    for (int i = 0; i <= STEPS; i++)
        if (fz_pf[i] != ref_pf[i]) {
            fprintf(stderr, "  FAIL LOSSLESS-PREFILL: token %d differs (fused %d vs standalone %d)\n",
                    i, fz_pf[i], ref_pf[i]); pf_bad = 1; break;
        }
    if (pf_bad) rc |= 1;
    else printf("  OK  LOSSLESS-PREFILL: first token + %d-token continuation byte-identical (fused == standalone)\n", STEPS);

    gemma4_engine_destroy(eng);
    return rc;
}

int main(int argc, char **argv) {
    const char *dense = (argc > 1) ? argv[1] : "/opt/spark/models/Qwen3-8B-abliterated.Q4_K_M.gguf";
    const char *moe   = (argc > 2) ? argv[2] : "/opt/spark/models/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf";
    setenv("FUCINA_PAGED_KV", "1", 1);

    int rc = 0;
    rc |= run_model(dense, 0.90);
    rc |= run_model(moe, 0.60);

    if (rc == 0) printf("PASS — fused prefill+decode is lossless (decode==plain step_batch, prefill==standalone) on dense + MoE\n");
    else         printf("FAIL — fused path diverged from the standalone references\n");
    return rc ? 1 : 0;
}
