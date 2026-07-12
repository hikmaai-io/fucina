// ABOUTME: DFlash draft-model device residency: uploads the real BF16 draft weights + binds views.
// ABOUTME: The persistent weight substrate the P3 (1+K) draft forward reads; config-derived sizing.
//
// The DFlash draft is a small second model whose weights live resident on-device as BF16 (the
// checkpoint's native dtype). This module owns that residency: it validates the checkpoint via the
// P2 loader Geometry, allocates one contiguous device slab, uploads every draft tensor into it, and
// binds per-layer non-owning views for the forward to read. It deliberately mirrors the engine's
// existing BF16-upload pattern (host bf16 bytes -> cudaMemcpy -> device pointer) rather than
// introducing a parallel tensor system; the views are plain device pointers into the slab.
//
// No CUDA math here (that is the forward). This is allocation + upload + view binding only, so it is
// unit-testable by loading the real checkpoint and spot-checking device bytes against the file.
#ifndef FUCINA_QWEN35_DFLASH_RESIDENCY_CUH
#define FUCINA_QWEN35_DFLASH_RESIDENCY_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <string>
#include <vector>
#include "safetensors.h"
#include "qwen35_dflash_loader.h"

// Per-layer device views (non-owning pointers into the resident slab). All BF16.
struct q35_dflash_layer_w {
    const __nv_bfloat16 *input_norm;   // [H]
    const __nv_bfloat16 *q_proj;       // [NQ*HD, H]
    const __nv_bfloat16 *k_proj;       // [NKV*HD, H]
    const __nv_bfloat16 *v_proj;       // [NKV*HD, H]
    const __nv_bfloat16 *o_proj;       // [H, NQ*HD]
    const __nv_bfloat16 *q_norm;       // [HD]
    const __nv_bfloat16 *k_norm;       // [HD]
    const __nv_bfloat16 *post_norm;    // [H]
    const __nv_bfloat16 *gate_proj;    // [I, H]
    const __nv_bfloat16 *up_proj;      // [I, H]
    const __nv_bfloat16 *down_proj;    // [H, I]
};

struct q35_dflash_residency {
    qwen35dflash::Geometry geom;
    __nv_bfloat16 *slab;               // one contiguous device allocation owning all weights
    size_t         slab_bytes;
    q35_dflash_layer_w *layers;        // [L] host array of view structs
    const __nv_bfloat16 *hidden_norm;  // [H]
    const __nv_bfloat16 *final_norm;   // [H]
    const __nv_bfloat16 *fc;           // [H, fc_in] (only if use_aux_hidden)
    int            ready;
};

// Free a residency (idempotent).
static inline void q35_dflash_residency_free(q35_dflash_residency *R) {
    if (!R) return;
    if (R->slab) cudaFree(R->slab);
    R->slab = nullptr; R->slab_bytes = 0; R->ready = 0;
    delete[] R->layers; R->layers = nullptr;
    R->hidden_norm = R->final_norm = R->fc = nullptr;
}

// Load + validate + upload the real draft checkpoint to device. Returns 0 on success; on any
// failure returns non-zero and leaves R zeroed/freed (no partial residency). err carries a reason.
// The caller must have opened M against the draft safetensors and parsed/validated Geometry into
// R->geom via the P2 loader BEFORE calling (so hostile shapes are rejected before this allocates).
static inline int q35_dflash_residency_upload(q35_dflash_residency *R, const st::Model &M,
                                              std::string &err) {
    if (!R) { err = "null residency"; return -1; }
    const qwen35dflash::Geometry &g = R->geom;
    // Collect (device_view_slot, tensor) in a fixed order; compute total BF16 element count.
    struct Item { const __nv_bfloat16 **slot; std::string name; size_t elems; };
    std::vector<Item> items;
    R->layers = new q35_dflash_layer_w[g.L];
    auto add = [&](const __nv_bfloat16 **slot, const std::string &name) -> bool {
        const st::Tensor *t = M.find(name);
        if (!t) { err = "missing " + name; return false; }
        if (t->dtype != st::Dtype::BF16) { err = "expected BF16 for " + name; return false; }
        items.push_back({slot, name, t->nbytes / 2});
        return true;
    };
    bool ok = add(&R->hidden_norm, "hidden_norm.weight") && add(&R->final_norm, "norm.weight");
    if (ok && g.use_aux_hidden) ok = add(&R->fc, "fc.weight");
    for (int l = 0; l < g.L && ok; l++) {
        auto lk = [&](const char *s){ return "layers." + std::to_string(l) + "." + s; };
        q35_dflash_layer_w &w = R->layers[l];
        ok = add(&w.input_norm, lk("input_layernorm.weight")) &&
             add(&w.q_proj, lk("self_attn.q_proj.weight")) &&
             add(&w.k_proj, lk("self_attn.k_proj.weight")) &&
             add(&w.v_proj, lk("self_attn.v_proj.weight")) &&
             add(&w.o_proj, lk("self_attn.o_proj.weight")) &&
             add(&w.q_norm, lk("self_attn.q_norm.weight")) &&
             add(&w.k_norm, lk("self_attn.k_norm.weight")) &&
             add(&w.post_norm, lk("post_attention_layernorm.weight")) &&
             add(&w.gate_proj, lk("mlp.gate_proj.weight")) &&
             add(&w.up_proj, lk("mlp.up_proj.weight")) &&
             add(&w.down_proj, lk("mlp.down_proj.weight"));
    }
    if (!ok) { delete[] R->layers; R->layers = nullptr; return -2; }

    size_t total_elems = 0;
    for (auto &it : items) total_elems += it.elems;
    R->slab_bytes = total_elems * sizeof(__nv_bfloat16);
    if (cudaMalloc(&R->slab, R->slab_bytes) != cudaSuccess) {
        cudaGetLastError(); err = "cudaMalloc draft slab failed"; delete[] R->layers; R->layers = nullptr; return -3;
    }
    // Upload each tensor contiguously; bind its view to the slab offset.
    size_t off = 0;
    for (auto &it : items) {
        const st::Tensor *t = M.find(it.name);
        __nv_bfloat16 *dst = R->slab + off;
        if (cudaMemcpy(dst, t->data, it.elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaGetLastError(); err = "upload failed for " + it.name;
            q35_dflash_residency_free(R); return -4;
        }
        *it.slot = dst;
        off += it.elems;
    }
    R->ready = 1;
    return 0;
}

#endif // FUCINA_QWEN35_DFLASH_RESIDENCY_CUH
