// Real-checkpoint verification for the NVFP4 loader stack (safetensors.h + nvfp4_loader.h +
// nvfp4.h) — opens an actual NVFP4 safetensors model, detects the layout, validates a sample
// projection's shapes, and dequantizes one real weight row. Proves the whole load path end-to-end
// short of the CUDA engine. No GPU needed.
//
// build: g++ -std=c++17 -O2 -Wall -Wextra cuda/nvfp4_inspect.cc -o /tmp/nvfp4_inspect
// run:   /tmp/nvfp4_inspect /path/to/gemma-4-12B-it-NVFP4
#include "safetensors.h"
#include "nvfp4_loader.h"
#include "nvfp4.h"
#include <cstdio>
#include <cmath>

static const char* dt(st::Dtype d) {
    switch (d) {
        case st::Dtype::U8: return "U8"; case st::Dtype::F8_E4M3: return "F8_E4M3";
        case st::Dtype::F32: return "F32"; case st::Dtype::BF16: return "BF16";
        case st::Dtype::F16: return "F16"; default: return "?";
    }
}
static void shp(const st::Tensor* t) {
    printf("[");
    for (size_t i = 0; i < t->shape.size(); i++) printf("%s%lld", i?",":"", (long long)t->shape[i]);
    printf("] %s %zuB", dt(t->dtype), t->nbytes);
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <model-dir-or-.safetensors>\n", argv[0]); return 2; }
    st::Model m; std::string err;
    if (!m.open(argv[1], err)) { printf("open failed: %s\n", err.c_str()); return 1; }
    printf("opened: %zu tensors\n", m.count());

    nvfp4ld::Layout L;
    if (!nvfp4ld::detect(m, L, err)) { printf("detect failed: %s\n", err.c_str()); return 1; }
    printf("naming      : %s\n", L.naming == nvfp4ld::Naming::COMPRESSED ? "compressed-tensors" : "modelopt");
    printf("layer_prefix: %s\n", L.layer_prefix.c_str());
    printf("n_layers    : %d\n", L.n_layers);
    printf("embed       : %s (%s)\n", L.embed_key.c_str(), m.has(L.embed_key) ? "found" : "MISSING");
    printf("final_norm  : %s (%s)\n", L.final_norm_key.c_str(), m.has(L.final_norm_key) ? "found" : "MISSING");
    printf("lm_head     : %s\n", L.lmhead_key.empty() ? "(tied → embed)" : L.lmhead_key.c_str());
    printf("tie_word_emb: %s (from config.json)\n", L.tie_word_embeddings ? "true" : "false");
    printf("ignore list : %zu entries", L.ignore.size());
    for (size_t i = 0; i < L.ignore.size() && i < 6; i++) printf(" [%s]", L.ignore[i].c_str());
    printf("\n  lm_head ignored? %s\n", nvfp4ld::is_ignored(
        nvfp4ld::QuantConfig{false, L.naming, 16, L.tie_word_embeddings, L.ignore}, "lm_head") ? "yes" : "no");

    // Validate every projection of layer 0 resolves with consistent dims.
    const char* pn[] = {"q","k","v","o","gate","up","down"};
    printf("\nlayer 0 projections:\n");
    const st::Tensor* qpacked = nullptr; const st::Tensor* qscale = nullptr; const st::Tensor* qg = nullptr;
    for (int p = 0; p < nvfp4ld::P_COUNT; p++) {
        auto k = nvfp4ld::proj_keys(L, 0, p);
        const st::Tensor* tw = m.find(k.packed);
        const st::Tensor* ts = m.find(k.scale);
        const st::Tensor* tg = m.find(k.gscale);
        printf("  %-4s ", pn[p]);
        if (!tw || !ts || !tg) { printf("MISSING (%s)\n", k.packed.c_str()); continue; }
        // packed [out, in/2], scale [out, in/16]
        long long out = tw->shape[0], in2 = tw->shape[1];
        long long sout = ts->shape[0], sin16 = ts->shape[1];
        bool shape_ok = (sout == out) && (in2 == sin16 * 8);   // in/2 == 8*(in/16)
        printf("packed "); shp(tw); printf("  scale "); shp(ts);
        printf("  gscale %s  in=%lld out=%lld  %s\n",
               dt(tg->dtype), in2*2, out, shape_ok ? "OK" : "*** SHAPE MISMATCH ***");
        if (p == 0) { qpacked = tw; qscale = ts; qg = tg; }
    }

    // Dequant the first 16 values of q_proj row 0 against the oracle, print them + the global.
    if (qpacked && qscale && qg) {
        float raw = *reinterpret_cast<const float*>(qg->data);
        float gs  = nvfp4ld::global_mul(L.naming, raw);   // normalized decode multiplier
        printf("\nq_proj global raw=%.6g  → decode multiplier=%.6g  (tensor amax≈6*448*mul=%.4f)\n",
               raw, gs, 6.f*448.f*gs);
        int in_dim = (int)(qpacked->shape[1] * 2);
        float row[32];
        nvfp4::dequant_row_host(qpacked->data, qscale->data, gs, in_dim < 32 ? in_dim : 32, row);
        printf("q_proj[0, 0:16] dequant:");
        float amax = 0;
        for (int i = 0; i < 16; i++) { printf(" % .4f", row[i]); amax = fmaxf(amax, fabsf(row[i])); }
        printf("\n  |max| of first 16 = %.4f (weight-like ⇒ O(0.01–0.1))\n", amax);
    }
    printf("\nLOADER STACK OK ON REAL CHECKPOINT\n");
    return 0;
}
