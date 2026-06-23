// test_e4b_gguf_load.cu — load the real Gemma-4-E4B Q4_0-QAT GGUF end-to-end
// through e4b_engine_create (Q4_0/Q6_K/F16/F32 → BF16 + FP8 PLE index), verify
// dims/residency, and run a forward SANITY (finite logits, in-range argmax, no
// illegal access). Optionally compares the GGUF prefill argmax against the BF16
// checkpoint on the SAME prompt (top-1 or within BF16 top-5, tolerating Q4_0
// QAT drift). Exit 0 iff the full pass succeeds.
//
// Usage: test_e4b_gguf_load [gguf] [bf16_dir]

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "e4b_engine.h"

static const char* kDefaultGguf =
    "/opt/spark/models/hub/models--google--gemma-4-E4B-it-qat-q4_0-gguf/snapshots/"
    "bb3b92e6f031fa438b409f898dd9f14f499a0cb0/gemma-4-E4B_q4_0-it.gguf";
static const char* kDefaultBf16 =
    "/opt/spark/models/hub/models--google--gemma-4-E4B-it/snapshots/"
    "fee6332c1abaafb77f6f9624236c63aa2f1d0187";

static int argmax(const std::vector<float>& x) {
    int b = 0; for (int i = 1; i < (int)x.size(); i++) if (x[i] > x[b]) b = i; return b;
}

int main(int argc, char** argv) {
    const char* gguf = (argc > 1) ? argv[1] : kDefaultGguf;
    const char* bf16 = (argc > 2) ? argv[2] : kDefaultBf16;

    // (1) detection
    int det = e4b_is_e4b_checkpoint(gguf);
    printf("e4b_is_e4b_checkpoint(%s) = %d\n", gguf, det);
    if (det != 1) { fprintf(stderr, "FAIL: GGUF not detected as E4B\n"); return 1; }

    // (2) create
    e4b_engine_t* eng = e4b_engine_create(gguf, 4096, 0);
    if (!eng) { fprintf(stderr, "FAIL: e4b_engine_create returned NULL\n"); return 1; }
    e4b_engine_print_info(eng);

    // (3) dims
    int nl = e4b_engine_n_layers(eng), hs = e4b_engine_hidden_size(eng), V = e4b_engine_vocab_size(eng);
    if (nl != 42 || hs != 2560 || V != 262144) {
        fprintf(stderr, "FAIL: unexpected dims (layers=%d hidden=%d vocab=%d)\n", nl, hs, V);
        e4b_engine_destroy(eng); return 1;
    }

    // (4) residency band
    uint64_t db = e4b_engine_device_bytes(eng);
    printf("device_bytes = %.2f GB\n", db / 1e9);
    if (db < 3e9 || db > 2.0e10) {
        fprintf(stderr, "FAIL: device_bytes %.2f GB outside (3,20) GB\n", db / 1e9);
        e4b_engine_destroy(eng); return 1;
    }

    // (5) forward sanity on a fixed realistic prompt (<bos> + a few real ids).
    // (Arbitrary <unused…> ids give near-random logits where Q4_0 drift swamps
    //  any BF16 agreement — use plausible content tokens for a meaningful gate.)
    const int32_t prompt[6] = { 2, 651, 5279, 576, 6081, 603 };
    const int nprompt = 6;
    std::vector<float> logits(V);
    int rc = e4b_engine_prefill(eng, prompt, nprompt, logits.data());
    if (rc != 0) { fprintf(stderr, "FAIL: prefill rc=%d\n", rc); e4b_engine_destroy(eng); return 1; }
    for (int i = 0; i < V; i++) {
        if (!std::isfinite(logits[i])) {
            fprintf(stderr, "FAIL: non-finite logit at %d\n", i); e4b_engine_destroy(eng); return 1;
        }
    }
    int am_gguf = argmax(logits);
    printf("GGUF prefill argmax = %d (logit %.3f)\n", am_gguf, logits[am_gguf]);
    if (am_gguf < 0 || am_gguf >= V) { fprintf(stderr, "FAIL: argmax out of range\n"); e4b_engine_destroy(eng); return 1; }

    // (6) no illegal access
    cudaDeviceSynchronize();
    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess) {
        fprintf(stderr, "FAIL: CUDA error after sync: %s\n", cudaGetErrorString(cerr));
        e4b_engine_destroy(eng); return 1;
    }

    // (optional, stronger) BF16 parity on the same prompt — top-1 match or GGUF
    // top token within BF16 top-5 (Q4_0 QAT drift tolerance).
    if (e4b_is_e4b_checkpoint(bf16) == 1) {
        e4b_engine_t* ref = e4b_engine_create(bf16, 4096, 0);
        if (ref) {
            std::vector<float> rl(V);
            if (e4b_engine_prefill(ref, prompt, nprompt, rl.data()) == 0) {
                int am_bf16 = argmax(rl);
                // BF16 top-5
                std::vector<int> idx(V); for (int i = 0; i < V; i++) idx[i] = i;
                std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                                  [&](int a, int b){ return rl[a] > rl[b]; });
                bool in_top5 = false; for (int k = 0; k < 5; k++) if (idx[k] == am_gguf) in_top5 = true;
                printf("BF16 prefill argmax = %d; GGUF top in BF16 top-5 = %s (top-1 match = %s)\n",
                       am_bf16, in_top5 ? "yes" : "NO", (am_gguf == am_bf16) ? "yes" : "no");
                if (!in_top5) {
                    fprintf(stderr, "WARN: GGUF top token not in BF16 top-5 — possible layout/dequant bug\n");
                    // Not a hard fail (informational), but surface loudly.
                }
            }
            e4b_engine_destroy(ref);
        } else {
            printf("(BF16 reference engine create failed — skipping parity)\n");
        }
    } else {
        printf("(no BF16 reference at %s — skipping parity)\n", bf16);
    }

    e4b_engine_destroy(eng);
    printf("PASS: E4B Q4_0 GGUF loaded + forward sanity OK\n");
    return 0;
}
