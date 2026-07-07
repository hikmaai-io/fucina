// test_e4b_ple_fp8.cu — end-to-end check of the Gemma-4-E4B foundation:
//   1. detect + parse the E4B architecture from the checkpoint's config.json
//   2. read real BF16 Per-Layer-Embedding rows from the safetensors
//   3. FP8-quantize them per-row, dequant-on-lookup, and compare to BF16
//      (cosine + relative L2 per row) — the accuracy gate for the "fp8 index"
//
// Usage: test_e4b_ple_fp8 [model_dir]
//   model_dir defaults to the on-box google/gemma-4-E4B-it snapshot.
//
// This validates the codec on the actual weights without needing the full
// 42-layer forward pass, so the memory win (5.6 GB → 2.7 GB) is verifiable in
// isolation. Exit 0 iff config parses AND every sampled row clears the gate.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "safetensors.h"
#include "gemma4_e4b.h"
#include "e4b_ple_fp8.cuh"

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); return 2; } } while(0)

static const char* kDefaultDir =
    "/opt/spark/models/hub/models--google--gemma-4-E4B-it/snapshots/"
    "fee6332c1abaafb77f6f9624236c63aa2f1d0187";

int main(int argc, char** argv) {
    const char* dir = (argc > 1) ? argv[1] : kDefaultDir;
    printf("E4B foundation test — model: %s\n", dir);

    // ── 1. open checkpoint + parse config ─────────────────────────────────
    st::Model model;
    std::string err;
    if (!model.open(dir, err)) { fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    if (model.config_json().empty()) { fprintf(stderr, "no config.json next to checkpoint\n"); return 1; }

    if (!e4b::is_e4b(model.config_json())) {
        fprintf(stderr, "FAIL: config.json not detected as Gemma-4-E4B\n"); return 1;
    }
    e4b::Config cfg;
    if (!e4b::parse(model.config_json(), cfg, err)) {
        fprintf(stderr, "FAIL: parse: %s\n", err.c_str()); return 1;
    }
    int n_full = 0; for (auto a : cfg.layer_types) n_full += (a == e4b::Attn::FULL);
    printf("  detected E4B: hidden=%d ff=%d layers=%d heads=%d kv_heads=%d\n",
           cfg.hidden_size, cfg.intermediate_size, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads);
    printf("  ple_dim=%d ple_vocab=%d ple_width=%d  kv_shared=%d (share@%d)  full_layers=%d\n",
           cfg.ple_dim, cfg.ple_vocab, cfg.ple_width(), cfg.n_kv_shared_layers,
           cfg.kv_share_start(), n_full);
    printf("  sliding_window=%d softcap=%.1f rms_eps=%.0e rope(swa/full)=%.0f/%.0f tie=%d\n",
           cfg.sliding_window, cfg.final_logit_softcap, cfg.rms_eps,
           cfg.rope_theta_sliding, cfg.rope_theta_full, (int)cfg.tie_word_embeddings);

    // sanity: dims must match the known E4B shape
    if (cfg.hidden_size != 2560 || cfg.n_layers != 42 || cfg.ple_dim != 256 ||
        cfg.ple_width() != 10752) {
        fprintf(stderr, "FAIL: parsed dims do not match expected E4B\n"); return 1;
    }

    // ── 2. locate the PLE table tensor ────────────────────────────────────
    e4b::Names nm;
    const st::Tensor* ple = model.find(nm.embed_per_layer());
    if (!ple) { fprintf(stderr, "FAIL: missing %s\n", nm.embed_per_layer().c_str()); return 1; }
    if (ple->dtype != st::Dtype::BF16 || ple->shape.size() != 2) {
        fprintf(stderr, "FAIL: PLE tensor not BF16 2-D\n"); return 1;
    }
    int64_t rows = ple->shape[0], width = ple->shape[1];
    printf("  PLE tensor %s [%lld, %lld] BF16 = %.2f GB  (FP8 → %.2f GB)\n",
           nm.embed_per_layer().c_str(), (long long)rows, (long long)width,
           rows * width * 2.0 / 1e9, rows * width * 1.0 / 1e9);
    if (width != cfg.ple_width()) { fprintf(stderr, "FAIL: PLE width != n_layers*ple_dim\n"); return 1; }

    // ── 3. sample N rows, quantize per-row FP8, lookup, compare ───────────
    const int N = 512;
    std::vector<int32_t> toks(N);
    // deterministic spread across the vocab (incl. row 0 and last row)
    for (int i = 0; i < N; ++i) toks[i] = (int32_t)((int64_t)i * (rows - 1) / (N - 1));

    const __nv_bfloat16* host_table = (const __nv_bfloat16*)ple->data;

    // upload the sampled rows contiguously as a [N, width] BF16 block
    size_t row_bytes = (size_t)width * sizeof(__nv_bfloat16);
    __nv_bfloat16* d_src;  CK(cudaMalloc(&d_src, (size_t)N * row_bytes));
    std::vector<__nv_bfloat16> ref((size_t)N * width);
    for (int i = 0; i < N; ++i) {
        const __nv_bfloat16* r = host_table + (size_t)toks[i] * width;
        memcpy(&ref[(size_t)i * width], r, row_bytes);
    }
    CK(cudaMemcpy(d_src, ref.data(), (size_t)N * row_bytes, cudaMemcpyHostToDevice));

    __nv_fp8_storage_t* d_q;     CK(cudaMalloc(&d_q, (size_t)N * width));
    float*              d_scale; CK(cudaMalloc(&d_scale, (size_t)N * sizeof(float)));
    e4b_ple_quantize_launch(d_src, d_q, d_scale, N, (int)width);
    CK(cudaDeviceSynchronize());

    // lookup uses contiguous indices 0..N-1 into the quantized block
    std::vector<int32_t> idx(N); for (int i = 0; i < N; ++i) idx[i] = i;
    int32_t* d_tok; CK(cudaMalloc(&d_tok, N * sizeof(int32_t)));
    CK(cudaMemcpy(d_tok, idx.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice));
    float* d_out; CK(cudaMalloc(&d_out, (size_t)N * width * sizeof(float)));
    e4b_ple_lookup_launch(d_q, d_scale, d_tok, d_out, N, (int)width);
    CK(cudaDeviceSynchronize());

    std::vector<float> got((size_t)N * width);
    CK(cudaMemcpy(got.data(), d_out, (size_t)N * width * sizeof(float), cudaMemcpyDeviceToHost));

    // per-row cosine similarity + relative L2 vs BF16 reference
    double worst_cos = 1.0, sum_cos = 0.0, worst_rel = 0.0;
    int worst_row = -1;
    for (int i = 0; i < N; ++i) {
        double dot = 0, na = 0, nb = 0, diff = 0;
        for (int c = 0; c < width; ++c) {
            double a = (double)__bfloat162float(ref[(size_t)i * width + c]);
            double b = (double)got[(size_t)i * width + c];
            dot += a * b; na += a * a; nb += b * b; diff += (a - b) * (a - b);
        }
        double cos = (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : 1.0;
        double rel = (na > 0) ? sqrt(diff / na) : 0.0;
        sum_cos += cos;
        if (cos < worst_cos) { worst_cos = cos; worst_row = i; }
        if (rel > worst_rel) worst_rel = rel;
    }
    printf("  FP8 PLE codec over %d rows: mean cos=%.6f  worst cos=%.6f (tok %d)  worst relL2=%.4f%%\n",
           N, sum_cos / N, worst_cos, worst_row >= 0 ? toks[worst_row] : -1, 100 * worst_rel);

    cudaFree(d_src); cudaFree(d_q); cudaFree(d_scale); cudaFree(d_tok); cudaFree(d_out);

    // Gate: per-row cosine ≥ 0.999 across the board. FP8 E4M3 with per-row amax
    // scaling on a pure lookup table comfortably clears this.
    const double kCosGate = 0.999;
    if (worst_cos < kCosGate) {
        fprintf(stderr, "FAIL: worst cosine %.6f < gate %.4f\n", worst_cos, kCosGate);
        return 1;
    }
    printf("PASS: E4B config detected + FP8 PLE index within accuracy gate (cos≥%.3f)\n", kCosGate);
    return 0;
}
