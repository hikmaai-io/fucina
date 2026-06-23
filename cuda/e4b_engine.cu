// e4b_engine.cu — Gemma-4-E4B engine: BF16 weight loader + FP8 PLE index.
//
// Phase 2 (this file as it stands): detect/parse the checkpoint, upload every
// language_model.* weight to device as BF16, and quantize the Per-Layer-
// Embedding table to FP8 E4M3 at load time. Forward pass lands in later phases;
// the loader is independently verifiable (load test reports residency).
//
// Weight tensors are HF nn.Linear layout [out_features, in_features], row-major.

#include "e4b_engine.h"
#include "safetensors.h"
#include "gemma4_e4b.h"
#include "e4b_ple_fp8.cuh"
#include "e4b_nvfp4.cuh"   // NVFP4 weight path (quantizer + decode GEMV) — FUCINA_E4B_FP4
#include "mmvq.cuh"        // shared dp4a MMVQ kernels (native Q4_0/Q6_K decode; same as dense 12B)
#include "e4b_gguf.cuh"    // GGUF weight source (Q4_0/Q6_K/F16/F32 → BF16 front-end)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

#define CKN(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"e4b: CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); return nullptr; } } while(0)

struct Layer {
    // attention
    __nv_bfloat16 *wq=nullptr, *wk=nullptr, *wv=nullptr, *wo=nullptr;
    __nv_bfloat16 *q_norm=nullptr, *k_norm=nullptr;        // [head_dim]
    // mlp (GeGLU)
    __nv_bfloat16 *w_gate=nullptr, *w_up=nullptr, *w_down=nullptr;
    // Hybrid decode quant (FUCINA_E4B_FP4). BF16 kept for the T>1 prefill GEMM.
    //   content (tolerates 4-bit) → NVFP4: FFN gate/up/down, attention V/O.
    //   index/decision (drives softmax routing) → FP8 E4M3: attention Q/K.
    e4bfp4::Weight    fp4_gate, fp4_up, fp4_down, fp4_wv, fp4_wo;
    e4bfp4::Fp8Weight fp8_wq, fp8_wk;
    // Native Q4_0 decode (GGUF Q4_0): the original QAT 18-byte blocks copied verbatim
    // from the GGUF and decoded via the SHARED dp4a MMVQ kernels (mmvq.cuh) — the same
    // path the dense 12B uses. Prefill dequants these → BF16 scratch for the cuBLAS GEMM.
    // Dims are derivable from cfg per layer, so only the device block streams are stored.
    uint8_t *q40_wq=nullptr, *q40_wk=nullptr, *q40_wv=nullptr, *q40_wo=nullptr;
    uint8_t *q40_gate=nullptr, *q40_up=nullptr, *q40_down=nullptr;
    // norms [hidden]
    __nv_bfloat16 *input_ln=nullptr, *post_attn_ln=nullptr;
    __nv_bfloat16 *pre_ff_ln=nullptr, *post_ff_ln=nullptr, *post_ple_ln=nullptr;
    // per-layer embedding combine
    __nv_bfloat16 *ple_in_gate=nullptr;   // [ple_dim, hidden]
    __nv_bfloat16 *ple_proj=nullptr;      // [hidden, ple_dim]
    float          layer_scalar=1.0f;
};

// One independent sequence's KV state. Each slot owns its per-layer K/V caches
// ([max_ctx, n_kv, hd], sharing-aware: shared layers alias the provider slot-cache),
// plus its own position. Multiple slots are driven concurrently by step_batch so
// the weight GEMMs are read once for B tokens — the continuous-batching throughput
// win, identical in spirit to the gemma4 (12B/31B) paged path.
struct Slot {
    bool active=false;
    int  n_past=0;
    std::vector<__nv_bfloat16*> kc, vc;   // [n_layers]
    std::vector<bool>           owned;
};

} // namespace

struct e4b_engine {
    e4b::Config cfg;
    cublasHandle_t cublas=nullptr;
    int device_id=0;
    uint32_t ctx=0;

    // global weights
    __nv_bfloat16 *d_embed=nullptr;          // [vocab, hidden] (tied head)
    __nv_bfloat16 *d_plm_proj=nullptr;       // per_layer_model_projection [ple_width, hidden]
    __nv_bfloat16 *d_ple_proj_norm=nullptr;  // [ple_dim]
    __nv_bfloat16 *d_final_norm=nullptr;     // [hidden]

    // FP8 Per-Layer-Embedding index
    __nv_fp8_storage_t *d_ple_fp8=nullptr;   // [vocab * ple_width]
    float              *d_ple_scale=nullptr; // [vocab]

    std::vector<Layer> layers;
    uint64_t dev_bytes=0;

    // NVFP4 decode FFN path (default ON; FUCINA_E4B_FP4=0 to disable): per-layer fp4
    // gate/up/down quantized at load; single-token decode reads them instead of the BF16
    // weights. d_fp4_xf/yf are
    // the persistent f32 GEMV scratch (max projection dim = intermediate_size).
    int    use_fp4=0;
    float *d_fp4_xf=nullptr, *d_fp4_yf=nullptr;
    e4bfp4::Fp8Weight fp8_head;   // tied LM head (decision projection) — FP8 E4M3

    // Native Q4_0 decode path (GGUF Q4_0; mutually exclusive with use_fp4/NVFP4). The 7
    // matmul projections decode straight off the on-disk Q4_0 nibbles via the SHARED MMVQ
    // kernels (mmvq.cuh); the tied head is the native Q6_K token_embd (mmvq_q6_k). Activation
    // Q8_1 scratch sized for the batched path (max_seqs rows): qa[S*FF]+da/sa[S*FF/32].
    int      use_q40=0;
    int8_t  *d_q40_qa=nullptr;   // [max_seqs*FF] int8
    float   *d_q40_da=nullptr;   // [max_seqs*FF/32]
    int32_t *d_q40_sa=nullptr;   // [max_seqs*FF/32]
    uint8_t *d_q6k_head=nullptr; // native Q6_K tied LM head [vocab][hidden/256*210]

    // Persistent single-token (T==1) decode scratch — allocated once at create, reused every
    // token instead of ~17 cudaMalloc/cudaFree + two RoPE-table recomputes per step. Removes
    // per-token alloc/launch-serialization overhead and is the prerequisite for CUDA-graph
    // decode capture (a graph cannot contain cudaMalloc). Prefill (T>1) still mallocs per call.
    struct DecodeScratch {
        bool ready=false;
        int32_t* ids=nullptr;
        __nv_bfloat16 *hidden=nullptr,*norm=nullptr,*tmpH=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,
                      *attn=nullptr,*gate=nullptr,*up=nullptr,*act=nullptr,*pleg=nullptr,*ctx_bf=nullptr;
        float *ple_lookup=nullptr,*ple_ctx=nullptr,*pli=nullptr,*logits_f=nullptr,*invf_s=nullptr,*invf_f=nullptr;
        int  *npast=nullptr;   // device-resident position for the graph-captured decode kernels
    } dec;
    // Prefill (T>1) dequant scratch: when use_q40 the BF16 projection weights are freed,
    // so prefill dequants each Q4_0 projection → this reused BF16 buffer for the cuBLAS GEMM.
    // Sized to the largest projection (FF×H). Serial reuse is safe — all on the null stream.
    __nv_bfloat16 *d_q40_wdq=nullptr;

    // ── KV cache: one Slot per concurrent sequence ──
    // Per layer per slot: K/V cache [max_ctx, n_kv, hd_of_layer] bf16. Shared
    // layers (≥24) do NOT own a cache — they alias the provider layer's WITHIN the
    // same slot (sliding→last sliding <24, full→last full <24). slots[0] backs the
    // single-sequence prefill/decode/generate API; seq_add/step_batch use any slot.
    int max_ctx = 0;
    int max_seqs = 0;
    std::vector<Slot> slots;
    // provider layer index per attention type (last non-shared of that type)
    int prov_sliding=-1, prov_full=-1;
};

// Allocate (or alias) a slot's per-layer K/V caches. Returns false on OOM.
static bool e4b_slot_alloc(e4b_engine* eng, Slot& s){
    const e4b::Config& c = eng->cfg;
    s.kc.assign(c.n_layers,nullptr); s.vc.assign(c.n_layers,nullptr);
    s.owned.assign(c.n_layers,false); s.n_past=0;
    for (int i=0;i<c.n_layers;++i){
        const bool full=(c.layer_types[i]==e4b::Attn::FULL);
        const int hd=full?c.global_head_dim:c.head_dim;
        if (c.layer_shares_kv(i)){
            int p = full?eng->prov_full:eng->prov_sliding;
            s.kc[i]=s.kc[p]; s.vc[i]=s.vc[p]; s.owned[i]=false;
        } else {
            size_t bytes=(size_t)eng->max_ctx*c.n_kv_heads*hd*sizeof(__nv_bfloat16);
            if (cudaMalloc(&s.kc[i],bytes)!=cudaSuccess || cudaMalloc(&s.vc[i],bytes)!=cudaSuccess) return false;
            s.owned[i]=true; eng->dev_bytes += 2*bytes;
        }
    }
    return true;
}
static void e4b_slot_free(e4b_engine* eng, Slot& s){
    for (size_t i=0;i<s.kc.size();++i) if (s.owned[i]){ if(s.kc[i])cudaFree(s.kc[i]); if(s.vc[i])cudaFree(s.vc[i]); }
    s.kc.clear(); s.vc.clear(); s.owned.clear(); s.active=false; s.n_past=0;
}

namespace {

// Upload a BF16 tensor by name. Optionally check element count. Adds to *total.
__nv_bfloat16* up_bf16(const st::Model& m, const std::string& name,
                       int64_t expect_elems, uint64_t* total) {
    const st::Tensor* t = m.find(name);
    if (!t) { fprintf(stderr, "e4b: missing tensor %s\n", name.c_str()); return nullptr; }
    if (t->dtype != st::Dtype::BF16) {
        fprintf(stderr, "e4b: %s not BF16\n", name.c_str()); return nullptr;
    }
    if (expect_elems > 0) {
        int64_t got = 1; for (auto d : t->shape) got *= d;
        if (got != expect_elems) {
            fprintf(stderr, "e4b: %s has %lld elems, expected %lld\n",
                    name.c_str(), (long long)got, (long long)expect_elems);
            return nullptr;
        }
    }
    __nv_bfloat16* d=nullptr;
    if (cudaMalloc(&d, t->nbytes) != cudaSuccess) {
        fprintf(stderr, "e4b: cudaMalloc %zu for %s failed\n", t->nbytes, name.c_str()); return nullptr;
    }
    if (cudaMemcpy(d, t->data, t->nbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "e4b: H2D %s failed\n", name.c_str()); cudaFree(d); return nullptr;
    }
    *total += t->nbytes;
    return d;
}

// Read a 1-element BF16 scalar to host float.
bool read_bf16_scalar(const st::Model& m, const std::string& name, float* out) {
    const st::Tensor* t = m.find(name);
    if (!t || t->dtype != st::Dtype::BF16 || t->nbytes < 2) return false;
    uint16_t bits; memcpy(&bits, t->data, 2);
    uint32_t f = (uint32_t)bits << 16; float v; memcpy(&v, &f, 4); *out = v;
    return true;
}

// ── GGUF weight source: dequantize a tensor by name → BF16 device buffer ────
// Produces the IDENTICAL row-major [out,in] BF16 device tensor up_bf16 yields
// from safetensors (Q4_0/Q6_K/F16/F32 host-dequant + single H2D). Optionally
// checks the element count. Adds to *total. Returns nullptr (logged) on any miss.
__nv_bfloat16* up_bf16_gguf(const e4bgguf::GgufFile& g, const std::string& name,
                            int64_t expect_elems, uint64_t* total) {
    uint64_t off, n_el; uint32_t gtype;
    if (!g.find(name.c_str(), &off, &n_el, &gtype)) {
        fprintf(stderr, "e4b: missing GGUF tensor %s\n", name.c_str()); return nullptr;
    }
    if (expect_elems > 0 && (int64_t)n_el != expect_elems) {
        fprintf(stderr, "e4b: GGUF %s has %llu elems, expected %lld\n",
                name.c_str(), (unsigned long long)n_el, (long long)expect_elems);
        return nullptr;
    }
    std::vector<__nv_bfloat16> host((size_t)n_el);
    if (!e4bgguf::gguf_dequant_to_bf16(g.data + off, (int64_t)n_el, gtype, host.data())) {
        fprintf(stderr, "e4b: GGUF dequant %s failed\n", name.c_str()); return nullptr;
    }
    const size_t nbytes = (size_t)n_el * sizeof(__nv_bfloat16);
    __nv_bfloat16* d = nullptr;
    if (cudaMalloc(&d, nbytes) != cudaSuccess) {
        fprintf(stderr, "e4b: cudaMalloc %zu for %s failed\n", nbytes, name.c_str()); return nullptr;
    }
    if (cudaMemcpy(d, host.data(), nbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "e4b: H2D %s failed\n", name.c_str()); cudaFree(d); return nullptr;
    }
    *total += nbytes;
    return d;
}

// Copy a GGUF tensor's raw quantized block stream verbatim to device (no dequant). Used
// for the native Q4_0 projections (18 B/32-elem block) and the Q6_K tied head (210 B/256).
// Validates the ggml type and element count; logs + returns false on any mismatch.
bool copy_blocks_gguf(const e4bgguf::GgufFile& g, const std::string& name, uint32_t want_type,
                      size_t block_bytes, int block_elems, int64_t expect_elems,
                      uint8_t** d_out, uint64_t* total) {
    uint64_t off, n_el; uint32_t gtype;
    if (!g.find(name.c_str(), &off, &n_el, &gtype)) {
        fprintf(stderr, "e4b: missing GGUF tensor %s\n", name.c_str()); return false;
    }
    if (gtype != want_type) {
        fprintf(stderr, "e4b: %s is ggml type %u, expected %u (native decode)\n",
                name.c_str(), gtype, want_type); return false;
    }
    if ((int64_t)n_el != expect_elems) {
        fprintf(stderr, "e4b: %s elem mismatch (%llu vs %lld)\n", name.c_str(),
                (unsigned long long)n_el, (long long)expect_elems); return false;
    }
    const size_t bytes = ((size_t)n_el / block_elems) * block_bytes;
    if (cudaMalloc(d_out, bytes) != cudaSuccess) {
        fprintf(stderr, "e4b: cudaMalloc %zu for %s failed\n", bytes, name.c_str()); return false;
    }
    if (cudaMemcpy(*d_out, g.data + off, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "e4b: H2D %s failed\n", name.c_str()); cudaFree(*d_out); *d_out=nullptr; return false;
    }
    *total += bytes;
    return true;
}

// Read a 1-element F32 (or F16) GGUF scalar to host float. Absent ⇒ false (caller keeps default).
bool read_scalar_gguf(const e4bgguf::GgufFile& g, const std::string& name, float* out) {
    uint64_t off, n_el; uint32_t gtype;
    if (!g.find(name.c_str(), &off, &n_el, &gtype) || n_el < 1) return false;
    if (gtype == e4bgguf::GGML_TYPE_F32) { float v; memcpy(&v, g.data + off, 4); *out = v; return true; }
    if (gtype == e4bgguf::GGML_TYPE_F16) { uint16_t h; memcpy(&h, g.data + off, 2); *out = e4bgguf::h2f_host(h); return true; }
    return false;
}

} // namespace

extern "C" int e4b_is_e4b_checkpoint(const char *path) {
    // GGUF E4B: a gemma4 GGUF that ALSO carries the PLE table tensor and the
    // shared-KV metadata. A dense gemma4 GGUF (12B/31B) has neither, so this is
    // an unambiguous dense-vs-E4B discriminator (lightweight mmap header scan,
    // no full weight parse).
    if (e4bgguf::is_gguf_file(path)) {
        e4bgguf::GgufFile g; std::string gerr;
        if (!g.open(path, gerr)) return -1;
        e4bgguf::gguf_str_t arch;
        bool is_gemma4 = g.get_str("general.architecture", &arch) &&
                         e4bgguf::gguf_str_eq(arch.ptr, arch.len, "gemma4");
        uint32_t shared = 0;
        bool e4b = is_gemma4 && g.has("per_layer_token_embd.weight") &&
                   g.get_u32("gemma4.attention.shared_kv_layers", &shared);
        g.close();
        return e4b ? 1 : 0;
    }
    st::Model m; std::string err;
    if (!m.open(path, err)) return -1;
    if (m.config_json().empty()) return -1;
    return e4b::is_e4b(m.config_json()) ? 1 : 0;
}

// ── safetensors weight load (BF16) — fills eng->cfg already parsed ──────────
// Returns false (and leaves eng for the caller to destroy) on any error.
static bool e4b_load_weights_safetensors(e4b_engine* eng, const char* model_path) {
    st::Model m; std::string err;
    if (!m.open(model_path, err)) { fprintf(stderr, "e4b: open %s: %s\n", model_path, err.c_str()); return false; }
    if (m.config_json().empty()) { fprintf(stderr, "e4b: no config.json next to checkpoint\n"); return false; }
    if (!e4b::is_e4b(m.config_json())) { fprintf(stderr, "e4b: not a Gemma-4-E4B checkpoint\n"); return false; }
    if (!e4b::parse(m.config_json(), eng->cfg, err)) {
        fprintf(stderr, "e4b: config parse: %s\n", err.c_str()); return false;
    }
    const e4b::Config& c = eng->cfg;
    e4b::Names nm;
    const int H = c.hidden_size, V = c.vocab_size, PD = c.ple_dim;

    // ── global weights ────────────────────────────────────────────────────
    eng->d_embed         = up_bf16(m, nm.embed_tokens(),     (int64_t)V * H,            &eng->dev_bytes);
    eng->d_plm_proj      = up_bf16(m, nm.per_layer_model_proj(), (int64_t)c.ple_width() * H, &eng->dev_bytes);
    eng->d_ple_proj_norm = up_bf16(m, nm.per_layer_proj_norm(), PD,                      &eng->dev_bytes);
    eng->d_final_norm    = up_bf16(m, nm.final_norm(),       H,                          &eng->dev_bytes);
    if (!eng->d_embed || !eng->d_plm_proj || !eng->d_ple_proj_norm || !eng->d_final_norm) return false;

    // ── FP8 Per-Layer-Embedding index ──────────────────────────────────────
    {
        const st::Tensor* ple = m.find(nm.embed_per_layer());
        if (!ple || ple->dtype != st::Dtype::BF16 || ple->shape.size() != 2 ||
            ple->shape[0] != V || ple->shape[1] != c.ple_width()) {
            fprintf(stderr, "e4b: bad PLE table %s\n", nm.embed_per_layer().c_str()); return false;
        }
        const int width = c.ple_width();
        __nv_bfloat16* d_src=nullptr;
        if (cudaMalloc(&d_src, ple->nbytes) != cudaSuccess) {
            fprintf(stderr, "e4b: PLE temp alloc %zu failed\n", ple->nbytes); return false;
        }
        if (cudaMemcpy(d_src, ple->data, ple->nbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "e4b: PLE H2D failed\n"); cudaFree(d_src); return false;
        }
        if (cudaMalloc(&eng->d_ple_fp8, (size_t)V * width) != cudaSuccess ||
            cudaMalloc(&eng->d_ple_scale, (size_t)V * sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "e4b: PLE FP8 alloc failed\n"); cudaFree(d_src); return false;
        }
        e4b_ple_quantize_launch(d_src, eng->d_ple_fp8, eng->d_ple_scale, V, width);
        if (cudaDeviceSynchronize() != cudaSuccess) { cudaFree(d_src); return false; }
        cudaFree(d_src);
        eng->dev_bytes += (uint64_t)V * width + (uint64_t)V * sizeof(float);
    }

    // ── per-layer weights ───────────────────────────────────────────────────
    eng->layers.resize(c.n_layers);
    for (int i = 0; i < c.n_layers; ++i) {
        Layer& L = eng->layers[i];
        // Full-attention layers use global_head_dim (512); sliding use head_dim (256).
        const int hd  = (c.layer_types[i] == e4b::Attn::FULL) ? c.global_head_dim : c.head_dim;
        const int qd  = c.n_heads    * hd;
        const int kvd = c.n_kv_heads * hd;
        L.wq    = up_bf16(m, nm.L(i, "self_attn.q_proj.weight"), (int64_t)qd  * H, &eng->dev_bytes);
        L.wk    = up_bf16(m, nm.L(i, "self_attn.k_proj.weight"), (int64_t)kvd * H, &eng->dev_bytes);
        L.wv    = up_bf16(m, nm.L(i, "self_attn.v_proj.weight"), (int64_t)kvd * H, &eng->dev_bytes);
        L.wo    = up_bf16(m, nm.L(i, "self_attn.o_proj.weight"), (int64_t)H * qd,  &eng->dev_bytes);
        L.q_norm= up_bf16(m, nm.L(i, "self_attn.q_norm.weight"), hd,               &eng->dev_bytes);
        L.k_norm= up_bf16(m, nm.L(i, "self_attn.k_norm.weight"), hd,               &eng->dev_bytes);
        L.w_gate= up_bf16(m, nm.L(i, "mlp.gate_proj.weight"), (int64_t)c.intermediate_size * H, &eng->dev_bytes);
        L.w_up  = up_bf16(m, nm.L(i, "mlp.up_proj.weight"),   (int64_t)c.intermediate_size * H, &eng->dev_bytes);
        L.w_down= up_bf16(m, nm.L(i, "mlp.down_proj.weight"), (int64_t)H * c.intermediate_size, &eng->dev_bytes);
        L.input_ln    = up_bf16(m, nm.L(i, "input_layernorm.weight"),            H, &eng->dev_bytes);
        L.post_attn_ln= up_bf16(m, nm.L(i, "post_attention_layernorm.weight"),   H, &eng->dev_bytes);
        L.pre_ff_ln   = up_bf16(m, nm.L(i, "pre_feedforward_layernorm.weight"),  H, &eng->dev_bytes);
        L.post_ff_ln  = up_bf16(m, nm.L(i, "post_feedforward_layernorm.weight"), H, &eng->dev_bytes);
        L.post_ple_ln = up_bf16(m, nm.L(i, "post_per_layer_input_norm.weight"),  H, &eng->dev_bytes);
        L.ple_in_gate = up_bf16(m, nm.L(i, "per_layer_input_gate.weight"), (int64_t)PD * H, &eng->dev_bytes);
        L.ple_proj    = up_bf16(m, nm.L(i, "per_layer_projection.weight"), (int64_t)H * PD, &eng->dev_bytes);
        read_bf16_scalar(m, nm.L(i, "layer_scalar"), &L.layer_scalar);  // optional
        if (!L.wq||!L.wk||!L.wv||!L.wo||!L.q_norm||!L.k_norm||!L.w_gate||!L.w_up||!L.w_down||
            !L.input_ln||!L.post_attn_ln||!L.pre_ff_ln||!L.post_ff_ln||!L.post_ple_ln||
            !L.ple_in_gate||!L.ple_proj) {
            fprintf(stderr, "e4b: layer %d incomplete\n", i);
            return false;
        }
    }
    return true;
}

// ── GGUF weight load (Q4_0/Q6_K/F16/F32 → BF16) — fills eng->cfg from KV ─────
// Mirrors the safetensors body tensor-for-tensor (same shapes, same row-major
// [out,in] BF16 buffers) so the downstream forward/decode path is unchanged.
// The only structural difference: the GGUF omits attn_k/attn_v/attn_k_norm for
// KV-shared layers (24..41), so those finds are gated on !layer_shares_kv(i)
// and left nullptr — exactly what the forward's `if(!shared)` path expects.
static bool e4b_load_weights_gguf(e4b_engine* eng, const char* model_path) {
    e4bgguf::GgufFile g; std::string gerr;
    if (!g.open(model_path, gerr)) { fprintf(stderr, "e4b: GGUF open %s: %s\n", model_path, gerr.c_str()); return false; }
    if (!e4bgguf::parse_gguf(g, eng->cfg, gerr)) {
        fprintf(stderr, "e4b: GGUF config: %s\n", gerr.c_str()); g.close(); return false;
    }
    const e4b::Config& c = eng->cfg;
    e4bgguf::GgufNames nm;
    const int H = c.hidden_size, V = c.vocab_size, PD = c.ple_dim;
    bool ok = true;

    // ── global weights ────────────────────────────────────────────────────
    eng->d_embed         = up_bf16_gguf(g, nm.token_embd(),    (int64_t)V * H,             &eng->dev_bytes);
    eng->d_plm_proj      = up_bf16_gguf(g, nm.plm_proj(),      (int64_t)c.ple_width() * H, &eng->dev_bytes);
    eng->d_ple_proj_norm = up_bf16_gguf(g, nm.ple_proj_norm(), PD,                         &eng->dev_bytes);
    eng->d_final_norm    = up_bf16_gguf(g, nm.final_norm(),    H,                          &eng->dev_bytes);
    if (!eng->d_embed || !eng->d_plm_proj || !eng->d_ple_proj_norm || !eng->d_final_norm) { g.close(); return false; }

    // ── FP8 Per-Layer-Embedding index ──────────────────────────────────────
    // Dequant the Q6_K PLE table [V, ple_width] to a device BF16 temp, then run
    // the SAME e4b_ple_quantize_launch the safetensors path uses → FP8 + scale.
    {
        const int width = c.ple_width();
        __nv_bfloat16* d_src = up_bf16_gguf(g, nm.ple_token_embd(), (int64_t)V * width, &eng->dev_bytes);
        if (!d_src) { fprintf(stderr, "e4b: GGUF PLE table dequant failed\n"); g.close(); return false; }
        // up_bf16_gguf counted the BF16 temp into dev_bytes; back it out (it is freed below).
        eng->dev_bytes -= (uint64_t)V * width * sizeof(__nv_bfloat16);
        if (cudaMalloc(&eng->d_ple_fp8, (size_t)V * width) != cudaSuccess ||
            cudaMalloc(&eng->d_ple_scale, (size_t)V * sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "e4b: PLE FP8 alloc failed\n"); cudaFree(d_src); g.close(); return false;
        }
        e4b_ple_quantize_launch(d_src, eng->d_ple_fp8, eng->d_ple_scale, V, width);
        if (cudaDeviceSynchronize() != cudaSuccess) { cudaFree(d_src); g.close(); return false; }
        cudaFree(d_src);
        eng->dev_bytes += (uint64_t)V * width + (uint64_t)V * sizeof(float);
    }

    // ── per-layer weights ───────────────────────────────────────────────────
    eng->layers.resize(c.n_layers);
    for (int i = 0; i < c.n_layers && ok; ++i) {
        Layer& L = eng->layers[i];
        const int hd  = (c.layer_types[i] == e4b::Attn::FULL) ? c.global_head_dim : c.head_dim;
        const int qd  = c.n_heads    * hd;
        const int kvd = c.n_kv_heads * hd;
        const bool shared = c.layer_shares_kv(i);
        L.wq    = up_bf16_gguf(g, nm.L(i, "attn_q.weight"),      (int64_t)qd  * H, &eng->dev_bytes);
        L.wo    = up_bf16_gguf(g, nm.L(i, "attn_output.weight"), (int64_t)H * qd,  &eng->dev_bytes);
        L.q_norm= up_bf16_gguf(g, nm.L(i, "attn_q_norm.weight"), hd,               &eng->dev_bytes);  // all layers
        if (!shared) {  // K/V/K_norm present only for layers 0..23 in the GGUF
            L.wk    = up_bf16_gguf(g, nm.L(i, "attn_k.weight"),      (int64_t)kvd * H, &eng->dev_bytes);
            L.wv    = up_bf16_gguf(g, nm.L(i, "attn_v.weight"),      (int64_t)kvd * H, &eng->dev_bytes);
            L.k_norm= up_bf16_gguf(g, nm.L(i, "attn_k_norm.weight"), hd,               &eng->dev_bytes);
        }
        L.w_gate= up_bf16_gguf(g, nm.L(i, "ffn_gate.weight"), (int64_t)c.intermediate_size * H, &eng->dev_bytes);
        L.w_up  = up_bf16_gguf(g, nm.L(i, "ffn_up.weight"),   (int64_t)c.intermediate_size * H, &eng->dev_bytes);
        L.w_down= up_bf16_gguf(g, nm.L(i, "ffn_down.weight"), (int64_t)H * c.intermediate_size, &eng->dev_bytes);
        L.input_ln    = up_bf16_gguf(g, nm.L(i, "attn_norm.weight"),           H, &eng->dev_bytes);
        L.post_attn_ln= up_bf16_gguf(g, nm.L(i, "post_attention_norm.weight"), H, &eng->dev_bytes);
        L.pre_ff_ln   = up_bf16_gguf(g, nm.L(i, "ffn_norm.weight"),            H, &eng->dev_bytes);
        L.post_ff_ln  = up_bf16_gguf(g, nm.L(i, "post_ffw_norm.weight"),       H, &eng->dev_bytes);
        L.post_ple_ln = up_bf16_gguf(g, nm.L(i, "post_norm.weight"),           H, &eng->dev_bytes);  // post_per_layer_input_norm
        L.ple_in_gate = up_bf16_gguf(g, nm.L(i, "inp_gate.weight"), (int64_t)PD * H, &eng->dev_bytes);
        L.ple_proj    = up_bf16_gguf(g, nm.L(i, "proj.weight"),     (int64_t)H * PD, &eng->dev_bytes);
        read_scalar_gguf(g, nm.L(i, "layer_output_scale.weight"), &L.layer_scalar);  // optional F32 scalar
        // Completeness: shared layers legitimately have no wk/wv/k_norm in the GGUF.
        bool layer_ok = L.wq && L.wo && L.q_norm && L.w_gate && L.w_up && L.w_down &&
                        L.input_ln && L.post_attn_ln && L.pre_ff_ln && L.post_ff_ln && L.post_ple_ln &&
                        L.ple_in_gate && L.ple_proj;
        if (!shared) layer_ok = layer_ok && L.wk && L.wv && L.k_norm;
        if (!layer_ok) { fprintf(stderr, "e4b: GGUF layer %d incomplete\n", i); ok = false; }
    }

    // ── native Q4_0 decode setup (default ON for Q4_0 GGUF; opt-out FUCINA_E4B_FP4=0) ──
    // Copy the 7 matmul projections (Q4_0) + the tied head (Q6_K) verbatim from the GGUF and
    // decode them through the SHARED dp4a MMVQ kernels (mmvq.cuh) — the exact path the dense
    // 12B uses (native Q4_0 batched GEMV + native Q6_K head). The BF16 projections are freed;
    // prefill dequants Q4_0 → BF16 scratch for the cuBLAS GEMM.
    if (ok) if (const char* e = getenv("FUCINA_E4B_FP4"); !(e && e[0]=='0')) {
        const int H_ = H, FF = c.intermediate_size, V_ = V, S = 8;  // S = max concurrent seqs
        bool q40 = true;
        for (int i = 0; i < c.n_layers && q40; ++i) {
            Layer& L = eng->layers[i];
            const bool full = (c.layer_types[i] == e4b::Attn::FULL);
            const int hd = full ? c.global_head_dim : c.head_dim;
            const int qd = c.n_heads * hd, kvd = c.n_kv_heads * hd;
            auto cp = [&](const char* t, int64_t el, uint8_t** d){
                return copy_blocks_gguf(g, nm.L(i, t), e4bgguf::GGML_TYPE_Q4_0, 18, 32, el, d, &eng->dev_bytes); };
            q40 &= cp("attn_q.weight",      (int64_t)qd*H_,  &L.q40_wq);
            q40 &= cp("attn_output.weight", (int64_t)H_*qd,  &L.q40_wo);
            q40 &= cp("ffn_gate.weight",    (int64_t)FF*H_,  &L.q40_gate);
            q40 &= cp("ffn_up.weight",      (int64_t)FF*H_,  &L.q40_up);
            q40 &= cp("ffn_down.weight",    (int64_t)H_*FF,  &L.q40_down);
            if (!c.layer_shares_kv(i)) {
                q40 &= cp("attn_k.weight", (int64_t)kvd*H_, &L.q40_wk);
                q40 &= cp("attn_v.weight", (int64_t)kvd*H_, &L.q40_wv);
            }
        }
        // tied LM head: native Q6_K token_embd (mmvq_q6_k) — 6.5-bit, no re-quant, argmax-precise.
        q40 &= copy_blocks_gguf(g, nm.token_embd(), e4bgguf::GGML_TYPE_Q6_K, 210, 256,
                                (int64_t)V_*H_, &eng->d_q6k_head, &eng->dev_bytes);
        // activation Q8_1 scratch (batched: up to S rows) + f32 MMVQ output (S*FF) + prefill dequant.
        q40 &= (cudaMalloc(&eng->d_q40_qa, (size_t)S*FF)                      == cudaSuccess);
        q40 &= (cudaMalloc(&eng->d_q40_da, (size_t)S*(FF/32)*sizeof(float))   == cudaSuccess);
        q40 &= (cudaMalloc(&eng->d_q40_sa, (size_t)S*(FF/32)*sizeof(int32_t)) == cudaSuccess);
        q40 &= (cudaMalloc(&eng->d_fp4_yf, (size_t)S*FF*sizeof(float))        == cudaSuccess);
        const size_t maxproj = (size_t)FF * H_;
        q40 &= (cudaMalloc(&eng->d_q40_wdq, maxproj * sizeof(__nv_bfloat16))  == cudaSuccess);
        if (cudaDeviceSynchronize() != cudaSuccess || !q40) {
            fprintf(stderr, "e4b: native Q4_0 decode setup failed — falling back to BF16 decode\n");
        } else {
            eng->use_q40 = 1;
            // BF16 projections no longer needed — prefill dequants Q4_0→scratch, decode reads
            // the native nibbles. Free them (the ~half-memory win).
            uint64_t freed = 0;
            auto FR = [&](__nv_bfloat16*& p, int64_t elems){ if (p){ cudaFree(p); p=nullptr; freed += (uint64_t)elems*sizeof(__nv_bfloat16); } };
            for (int i = 0; i < c.n_layers; ++i) {
                Layer& L = eng->layers[i];
                const bool full = (c.layer_types[i] == e4b::Attn::FULL);
                const int hd = full ? c.global_head_dim : c.head_dim;
                const int qd = c.n_heads * hd, kvd = c.n_kv_heads * hd;
                FR(L.wq, (int64_t)qd*H_); FR(L.wo, (int64_t)H_*qd);
                FR(L.w_gate, (int64_t)FF*H_); FR(L.w_up, (int64_t)FF*H_); FR(L.w_down, (int64_t)H_*FF);
                if (!c.layer_shares_kv(i)) { FR(L.wk, (int64_t)kvd*H_); FR(L.wv, (int64_t)kvd*H_); }
            }
            eng->dev_bytes -= freed;
            eng->dev_bytes += (size_t)S*FF + (size_t)S*(FF/32)*8 + (size_t)S*FF*sizeof(float)
                            + maxproj*sizeof(__nv_bfloat16);
            fprintf(stderr, "e4b: native Q4_0 decode ON (matmuls@Q4_0 + head@Q6_K via shared MMVQ; "
                    "freed %.2f GB BF16 projections, prefill dequants on the fly)\n", freed / 1e9);
        }
    }

    g.close();
    return ok;
}

extern "C" e4b_engine_t *e4b_engine_create(const char *model_path, uint32_t context_size, int device_id) {
    CKN(cudaSetDevice(device_id));

    e4b_engine* eng = new e4b_engine();
    eng->device_id = device_id;
    eng->ctx = context_size;
    if (cublasCreate(&eng->cublas) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "e4b: cublasCreate failed\n"); delete eng; return nullptr;
    }

    // Fork on file magic: GGUF (Q4_0-QAT/Q6_K) vs BF16 safetensors. Both fill the
    // SAME eng->cfg + the SAME BF16 device tensors; the shared tail (KV slots,
    // optional NVFP4 decode) below is weight-source-agnostic.
    const bool is_gguf = e4bgguf::is_gguf_file(model_path);
    if (is_gguf ? !e4b_load_weights_gguf(eng, model_path)
                : !e4b_load_weights_safetensors(eng, model_path)) {
        e4b_engine_destroy(eng); return nullptr;
    }
    const e4b::Config& c = eng->cfg;
    const int H = c.hidden_size, V = c.vocab_size;

    // ── KV cache slots (sharing-aware) ──────────────────────────────────────
    eng->max_ctx = (int)context_size;
    eng->max_seqs = 8;                          // concurrent sequences (continuous batching)
    // provider per attention type = last non-shared layer of that type
    for (int j=0;j<c.kv_share_start();++j)
        (c.layer_types[j]==e4b::Attn::FULL ? eng->prov_full : eng->prov_sliding) = j;
    eng->slots.resize(eng->max_seqs);
    if (!e4b_slot_alloc(eng, eng->slots[0])) {  // slot 0 backs the single-seq API
        fprintf(stderr,"e4b: slot 0 KV alloc failed\n"); e4b_engine_destroy(eng); return nullptr;
    }
    eng->slots[0].active=true;

    // ── NVFP4 decode FFN path — safetensors (BF16) checkpoints only ──────────
    // GGUF Q4_0 checkpoints decode off the native Q4_0 nibbles instead (set up in the
    // loader, eng->use_q40), which is the same footprint at higher fidelity. This NVFP4
    // re-quant path covers the BF16 safetensors checkpoint, where there is no on-disk
    // Q4_0 to read. Default ON; opt-out FUCINA_E4B_FP4=0 (forces BF16 decode).
    if (!is_gguf && !eng->use_q40)
    if (const char* e = getenv("FUCINA_E4B_FP4"); !(e && e[0]=='0')) {
        const int H_=c.hidden_size, FF=c.intermediate_size, V_=c.vocab_size;
        bool ok=true;
        for (int i=0;i<c.n_layers && ok;++i){
            Layer& L=eng->layers[i];
            const bool full=(c.layer_types[i]==e4b::Attn::FULL);
            const int hd=full?c.global_head_dim:c.head_dim;
            const int qd=c.n_heads*hd, kvd=c.n_kv_heads*hd;
            // content → NVFP4 (4.5-bit): FFN gate/up/down, attention V/O.
            ok &= e4bfp4::e4b_nvfp4_quantize(L.w_gate, FF, H_, &L.fp4_gate);   // [FF,H]
            ok &= e4bfp4::e4b_nvfp4_quantize(L.w_up,   FF, H_, &L.fp4_up);     // [FF,H]
            ok &= e4bfp4::e4b_nvfp4_quantize(L.w_down, H_, FF, &L.fp4_down);   // [H,FF]
            ok &= e4bfp4::e4b_nvfp4_quantize(L.wo,     H_, qd, &L.fp4_wo);     // [H,qd]
            // index → FP8 E4M3 (8-bit): attention Q (all layers), K (only layers that project it;
            // KV-shared layers reuse a provider's K and never call wk).
            ok &= e4bfp4::e4b_fp8_quantize(L.wq, qd, H_, &L.fp8_wq);          // [qd,H]
            if (!c.layer_shares_kv(i)) {
                ok &= e4bfp4::e4b_fp8_quantize(L.wk, kvd, H_, &L.fp8_wk);     // [kvd,H]
                ok &= e4bfp4::e4b_nvfp4_quantize(L.wv, kvd, H_, &L.fp4_wv);   // [kvd,H]
            }
        }
        // decision → FP8 E4M3: the tied LM head [V,H], read once per token for logits.
        ok &= e4bfp4::e4b_fp8_quantize(eng->d_embed, V_, H_, &eng->fp8_head);
        // f32 GEMV scratch: max projection dim is FF (the FFN); head writes its own per-call buffer.
        ok &= (cudaMalloc(&eng->d_fp4_xf, (size_t)FF*sizeof(float))==cudaSuccess);
        ok &= (cudaMalloc(&eng->d_fp4_yf, (size_t)FF*sizeof(float))==cudaSuccess);
        if (cudaDeviceSynchronize()!=cudaSuccess || !ok) {
            fprintf(stderr,"e4b: hybrid quant failed — falling back to BF16 decode\n");
        } else {
            eng->use_fp4=1;
            uint64_t fp4b=0, fp8b=eng->fp8_head.bytes;
            for (Layer& L:eng->layers){
                fp4b += L.fp4_gate.bytes+L.fp4_up.bytes+L.fp4_down.bytes+L.fp4_wv.bytes+L.fp4_wo.bytes;
                fp8b += L.fp8_wq.bytes+L.fp8_wk.bytes;
            }
            eng->dev_bytes += fp4b + fp8b + 2*(uint64_t)FF*sizeof(float);
            fprintf(stderr,"e4b: hybrid decode quant ON (+%.2f GB: content@NVFP4 FFN+V/O %.2f GB, "
                    "index@FP8 Q/K+head %.2f GB; BF16 kept for prefill)\n",
                    (fp4b+fp8b)/1e9, fp4b/1e9, fp8b/1e9);
        }
    }

    fprintf(stderr, "e4b: loaded Gemma-4-E4B — %d layers, hidden %d, %.2f GB resident "
            "(FP8 PLE index saves ~%.2f GB vs BF16; KV %d ctx × up to %d seqs)\n",
            c.n_layers, H, eng->dev_bytes / 1e9,
            ((uint64_t)V * c.ple_width()) / 1e9, eng->max_ctx, eng->max_seqs);
    return eng;
}

extern "C" void e4b_engine_destroy(e4b_engine_t *eng) {
    if (!eng) return;
    auto F = [](void* p){ if (p) cudaFree(p); };
    F(eng->d_embed); F(eng->d_plm_proj); F(eng->d_ple_proj_norm); F(eng->d_final_norm);
    F(eng->d_ple_fp8); F(eng->d_ple_scale);
    for (Layer& L : eng->layers) {
        F(L.wq); F(L.wk); F(L.wv); F(L.wo); F(L.q_norm); F(L.k_norm);
        F(L.w_gate); F(L.w_up); F(L.w_down);
        F(L.input_ln); F(L.post_attn_ln); F(L.pre_ff_ln); F(L.post_ff_ln); F(L.post_ple_ln);
        F(L.ple_in_gate); F(L.ple_proj);
        e4bfp4::weight_free(&L.fp4_gate); e4bfp4::weight_free(&L.fp4_up); e4bfp4::weight_free(&L.fp4_down);
        e4bfp4::weight_free(&L.fp4_wv);   e4bfp4::weight_free(&L.fp4_wo);
        e4bfp4::fp8_weight_free(&L.fp8_wq); e4bfp4::fp8_weight_free(&L.fp8_wk);
        auto FU = [](uint8_t* p){ if (p) cudaFree(p); };
        FU(L.q40_wq); FU(L.q40_wk); FU(L.q40_wv); FU(L.q40_wo);
        FU(L.q40_gate); FU(L.q40_up); FU(L.q40_down);
    }
    e4bfp4::fp8_weight_free(&eng->fp8_head);
    F(eng->d_fp4_xf); F(eng->d_fp4_yf);
    if (eng->d_q40_qa) cudaFree(eng->d_q40_qa);
    if (eng->d_q40_da) cudaFree(eng->d_q40_da);
    if (eng->d_q40_sa) cudaFree(eng->d_q40_sa);
    if (eng->d_q40_wdq) cudaFree(eng->d_q40_wdq);
    if (eng->d_q6k_head) cudaFree(eng->d_q6k_head);
    { auto& d=eng->dec; F(d.ids); F(d.hidden); F(d.norm); F(d.tmpH); F(d.q); F(d.k); F(d.v); F(d.attn);
      F(d.gate); F(d.up); F(d.act); F(d.pleg); F(d.ctx_bf); F(d.ple_lookup); F(d.ple_ctx); F(d.pli);
      F(d.logits_f); F(d.invf_s); F(d.invf_f); F(d.npast); }
    for (Slot& s : eng->slots) e4b_slot_free(eng, s);
    if (eng->cublas) cublasDestroy(eng->cublas);
    delete eng;
}

extern "C" void e4b_engine_print_info(const e4b_engine_t *eng) {
    if (!eng) return;
    const e4b::Config& c = eng->cfg;
    int n_full = 0; for (auto a : c.layer_types) n_full += (a == e4b::Attn::FULL);
    printf("Gemma-4-E4B engine:\n");
    printf("  hidden=%d ff=%d layers=%d (%d full / %d sliding)\n",
           c.hidden_size, c.intermediate_size, c.n_layers, n_full, c.n_layers - n_full);
    printf("  heads=%d kv_heads=%d head_dim=%d global_head_dim=%d sliding_window=%d\n",
           c.n_heads, c.n_kv_heads, c.head_dim, c.global_head_dim, c.sliding_window);
    printf("  PLE: dim=%d width=%d (FP8 index)   KV-shared tail=%d (share@%d)\n",
           c.ple_dim, c.ple_width(), c.n_kv_shared_layers, c.kv_share_start());
    printf("  softcap=%.1f rms_eps=%.0e rope(swa/full)=%.0f/%.0f tie=%d\n",
           c.final_logit_softcap, c.rms_eps, c.rope_theta_sliding, c.rope_theta_full,
           (int)c.tie_word_embeddings);
    printf("  device bytes resident: %.2f GB\n", eng->dev_bytes / 1e9);
}

extern "C" int e4b_engine_n_layers(const e4b_engine_t *eng)     { return eng ? eng->cfg.n_layers : 0; }
extern "C" int e4b_engine_hidden_size(const e4b_engine_t *eng)  { return eng ? eng->cfg.hidden_size : 0; }
extern "C" int e4b_engine_vocab_size(const e4b_engine_t *eng)   { return eng ? eng->cfg.vocab_size : 0; }
extern "C" uint64_t e4b_engine_device_bytes(const e4b_engine_t *eng) { return eng ? eng->dev_bytes : 0; }

// ════════════════════════════════════════════════════════════════════════════
// Forward pass (Phase 3) — correctness-first BF16 prefill.
//
// All matmuls go through cuBLAS (bf16 in, fp32 accumulate). Norms, RoPE, the
// GeGLU activation, attention softmax and the PLE combine run in fp32 in custom
// kernels, matching modeling_gemma4.py (RMSNorm/softmax computed in float).
// The running residual stream is kept in bf16 (as in HF, dtype=bfloat16).
// ════════════════════════════════════════════════════════════════════════════
namespace {

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float b2f(bf16 x){ return __bfloat162float(x); }
__device__ __forceinline__ bf16  f2b(float x){ return __float2bfloat16(x); }

__device__ __forceinline__ float gelu_tanh(float x){
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}

// RMSNorm over the last dim: y = x*rsqrt(mean(x^2)+eps) * w  (w optional).
// Gemma4: NO (1+w). One block per row.
__global__ void rmsnorm_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                               bf16* __restrict__ y, int rows, int dim, float eps){
    int r = blockIdx.x; if (r >= rows) return;
    const bf16* xr = x + (size_t)r*dim; bf16* yr = y + (size_t)r*dim;
    __shared__ float sh[256];
    float ss = 0.f;
    for (int i=threadIdx.x;i<dim;i+=blockDim.x){ float v=b2f(xr[i]); ss+=v*v; }
    sh[threadIdx.x]=ss; __syncthreads();
    for (int s=blockDim.x>>1;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
    float inv = rsqrtf(sh[0]/dim + eps);
    for (int i=threadIdx.x;i<dim;i+=blockDim.x){
        float v = b2f(xr[i])*inv;
        if (w) v *= b2f(w[i]);
        yr[i]=f2b(v);
    }
}

// Scaled token-embedding gather: out[t,:] = embed[ids[t],:] * scale.
__global__ void embed_kernel(const bf16* __restrict__ embed, const int32_t* __restrict__ ids,
                             bf16* __restrict__ out, int T, int H, float scale){
    int t = blockIdx.x; if (t>=T) return;
    const bf16* row = embed + (size_t)ids[t]*H;
    bf16* o = out + (size_t)t*H;
    for (int i=threadIdx.x;i<H;i+=blockDim.x) o[i]=f2b(b2f(row[i])*scale);
}

// y = a + b  (elementwise, bf16)
__global__ void add_kernel(bf16* __restrict__ a, const bf16* __restrict__ b, int n){
    int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]=f2b(b2f(a[i])+b2f(b[i]));
}
// x *= s (scalar)
__global__ void scale_kernel(bf16* __restrict__ x, float s, int n){
    int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]=f2b(b2f(x[i])*s);
}
// GeGLU: out[i] = gelu_tanh(gate[i]) * up[i]
__global__ void geglu_kernel(const bf16* __restrict__ gate, const bf16* __restrict__ up,
                             bf16* __restrict__ out, int n){
    int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=f2b(gelu_tanh(b2f(gate[i]))*b2f(up[i]));
}

// Per-head RMSNorm over head_dim, applied in place to [T, n_heads, hd].
// w optional (V uses with_scale=false → w=NULL). One block per (token,head).
__global__ void head_rmsnorm_kernel(bf16* __restrict__ x, const bf16* __restrict__ w,
                                    int T, int n_heads, int hd, float eps){
    int idx = blockIdx.x; if (idx >= T*n_heads) return;
    bf16* v = x + (size_t)idx*hd;
    __shared__ float sh[256];
    float ss=0.f;
    for(int i=threadIdx.x;i<hd;i+=blockDim.x){ float a=b2f(v[i]); ss+=a*a; }
    sh[threadIdx.x]=ss; __syncthreads();
    for(int s=blockDim.x>>1;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
    float inv=rsqrtf(sh[0]/hd+eps);
    for(int i=threadIdx.x;i<hd;i+=blockDim.x){ float a=b2f(v[i])*inv; if(w) a*=b2f(w[i]); v[i]=f2b(a); }
}

// RoPE (rotate_half) over [T, n_heads, hd] with per-pair angle = pos * inv_freq[i].
// inv_freq has hd/2 entries (zeros ⇒ NoPE dims). pos = token index.
__global__ void rope_kernel(bf16* __restrict__ x, const float* __restrict__ inv_freq,
                            int T, int n_heads, int hd, int pos_offset){
    int idx = blockIdx.x; if (idx >= T*n_heads) return;
    int t = idx / n_heads;
    bf16* v = x + (size_t)idx*hd;
    int half = hd/2;
    for (int i=threadIdx.x;i<half;i+=blockDim.x){
        float ang = (float)(pos_offset + t) * inv_freq[i];
        float c=cosf(ang), s=sinf(ang);
        float a=b2f(v[i]), b=b2f(v[i+half]);
        v[i]      = f2b(a*c - b*s);
        v[i+half] = f2b(b*c + a*s);
    }
}

// Causal (optionally sliding-window) GQA attention reading from the KV cache.
// q:[T,n_heads,hd] are the NEW tokens at absolute positions [n_past, n_past+T).
// kcache/vcache:[max_ctx,n_kv,hd] hold all keys/values up to n_past+T. window<=0 ⇒
// full causal. One block per (head, new-query); threads loop dim/keys. scores held
// in dynamic shared memory sized to the longest attended span (n_past+T).
__global__ void attn_cache_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kcache,
                                  const bf16* __restrict__ vcache, bf16* __restrict__ out,
                                  int T, int n_past, int n_heads, int n_kv, int hd,
                                  float scaling, int window){
    int h = blockIdx.x, t = blockIdx.y;
    if (h>=n_heads || t>=T) return;
    int P = n_past + t;                       // absolute query position
    int group = n_heads / n_kv, kvh = h/group;
    const bf16* qv = q + ((size_t)t*n_heads + h)*hd;
    extern __shared__ float scores[];         // [len], indexed relative to lo
    int lo = (window>0) ? (P-window+1) : 0; if (lo<0) lo=0;
    int len = P - lo + 1;
    for (int j=threadIdx.x; j<len; j+=blockDim.x){
        const bf16* kv = kcache + ((size_t)(lo+j)*n_kv + kvh)*hd;
        float dot=0.f; for(int d=0;d<hd;d++) dot += b2f(qv[d])*b2f(kv[d]);
        scores[j]=dot*scaling;
    }
    __syncthreads();
    __shared__ float ssum;
    if (threadIdx.x==0){
        float m=-1e30f; for(int j=0;j<len;j++) m=fmaxf(m,scores[j]);
        float sm=0.f; for(int j=0;j<len;j++){ float e=expf(scores[j]-m); scores[j]=e; sm+=e; }
        ssum=sm;
    }
    __syncthreads();
    float inv = 1.f/ssum;
    bf16* ov = out + ((size_t)t*n_heads + h)*hd;
    for (int d=threadIdx.x; d<hd; d+=blockDim.x){
        float acc=0.f;
        for (int j=0;j<len;j++){ const bf16* vv=vcache+((size_t)(lo+j)*n_kv+kvh)*hd; acc += scores[j]*b2f(vv[d]); }
        ov[d]=f2b(acc*inv);
    }
}

// Copy new-token K (or V) [T, n_kv*hd] into the cache at absolute offset n_past.
__global__ void kv_store_kernel(const bf16* __restrict__ src, bf16* __restrict__ cache,
                                int T, int row, int n_past){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < T*row) cache[(size_t)n_past*row + i] = src[i];
}

// ── CUDA-graph-friendly decode (T==1) kernels: DEVICE-resident position, FIXED smem ──
// kv_store reading the position from device memory (one new token at *d_npast).
__global__ void kv_store_dev_kernel(const bf16* __restrict__ src, bf16* __restrict__ cache,
                                    int row, const int* __restrict__ d_npast){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < row) cache[(size_t)(*d_npast)*row + i] = src[i];
}

// Flash (online-softmax, tiled) decode attention for T==1. Unlike attn_cache_kernel — whose
// scores[] smem grows with n_past and which takes n_past as a host arg — this uses FIXED smem
// (TILE scores + hd accumulator) and reads the position from device memory, so the whole decode
// forward can be captured once into a CUDA graph and replayed each token. One block per head.
template<int TILE>
__global__ void attn_flash_decode_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ kcache, const bf16* __restrict__ vcache,
    bf16* __restrict__ out, const int* __restrict__ d_npast,
    int n_heads, int n_kv, int hd, float scaling, int window)
{
    int h = blockIdx.x; if (h >= n_heads) return;
    const int P = *d_npast;                       // query absolute position (t=0)
    const int group = n_heads / n_kv, kvh = h / group;
    int lo = (window > 0) ? (P - window + 1) : 0; if (lo < 0) lo = 0;
    const bf16* qv = q + (size_t)h * hd;

    extern __shared__ float smem[];
    float* sc  = smem;          // [TILE] tile scores
    float* acc = smem + TILE;   // [hd]   running weighted-V accumulator
    __shared__ float m_run, l_run, corr;

    for (int d = threadIdx.x; d < hd; d += blockDim.x) acc[d] = 0.f;
    if (threadIdx.x == 0) { m_run = -1e30f; l_run = 0.f; }
    __syncthreads();

    for (int base = lo; base <= P; base += TILE) {
        const int tlen = min(TILE, P - base + 1);
        for (int j = threadIdx.x; j < tlen; j += blockDim.x) {
            const bf16* kv = kcache + ((size_t)(base + j) * n_kv + kvh) * hd;
            float dot = 0.f;
            for (int d = 0; d < hd; d++) dot += b2f(qv[d]) * b2f(kv[d]);
            sc[j] = dot * scaling;
        }
        __syncthreads();
        if (threadIdx.x == 0) {                   // online-softmax state update for this tile
            float tm = -1e30f;
            for (int j = 0; j < tlen; j++) tm = fmaxf(tm, sc[j]);
            float new_m = fmaxf(m_run, tm);
            corr = expf(m_run - new_m);
            float tl = 0.f;
            for (int j = 0; j < tlen; j++) { float e = expf(sc[j] - new_m); sc[j] = e; tl += e; }
            l_run = l_run * corr + tl; m_run = new_m;
        }
        __syncthreads();
        for (int d = threadIdx.x; d < hd; d += blockDim.x) {
            float a = acc[d] * corr;
            for (int j = 0; j < tlen; j++) {
                const bf16* vv = vcache + ((size_t)(base + j) * n_kv + kvh) * hd;
                a += sc[j] * b2f(vv[d]);
            }
            acc[d] = a;
        }
        __syncthreads();
    }
    const float inv = 1.f / l_run;
    bf16* ov = out + (size_t)h * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) ov[d] = f2b(acc[d] * inv);
}

// ── batched (multi-sequence) variants: B sequences, one new token each ──
// RoPE with a per-sequence absolute position. x:[B,n_heads,hd], pos:[B].
__global__ void rope_batch_kernel(bf16* __restrict__ x, const float* __restrict__ inv_freq,
                                  int B, int n_heads, int hd, const int* __restrict__ pos){
    int idx = blockIdx.x; if (idx >= B*n_heads) return;
    int b = idx / n_heads; bf16* v = x + (size_t)idx*hd; int half = hd/2;
    for (int i=threadIdx.x;i<half;i+=blockDim.x){
        float ang=(float)pos[b]*inv_freq[i]; float cc=cosf(ang), ss=sinf(ang);
        float a=b2f(v[i]), bb=b2f(v[i+half]);
        v[i]=f2b(a*cc-bb*ss); v[i+half]=f2b(bb*cc+a*ss);
    }
}
// Store each sequence's new K/V row into ITS cache at ITS position. src:[B,row];
// caches[b] is that slot's [ctx,row] cache; pos[b] the write offset.
__global__ void kv_store_batch_kernel(const bf16* __restrict__ src, bf16* const* __restrict__ caches,
                                      const int* __restrict__ pos, int B, int row){
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i>=B*row) return;
    int b=i/row, e=i%row;
    caches[b][(size_t)pos[b]*row + e] = src[i];
}
// Batched GQA attention: sequence b's single query (at pos[b]) attends ITS cache
// [0..pos[b]] (sliding window if window>0). One block per (head, sequence).
__global__ void attn_batch_kernel(const bf16* __restrict__ q, bf16* const* __restrict__ kcaches,
                                  bf16* const* __restrict__ vcaches, const int* __restrict__ pos,
                                  bf16* __restrict__ out, int B, int n_heads, int n_kv, int hd,
                                  float scaling, int window){
    int h = blockIdx.x, b = blockIdx.y; if (h>=n_heads || b>=B) return;
    int P = pos[b]; int group=n_heads/n_kv, kvh=h/group;
    const bf16* qv = q + ((size_t)b*n_heads + h)*hd;
    const bf16* kc = kcaches[b]; const bf16* vc = vcaches[b];
    extern __shared__ float scores[];
    int lo = (window>0)?(P-window+1):0; if(lo<0) lo=0; int len=P-lo+1;
    for (int j=threadIdx.x;j<len;j+=blockDim.x){
        const bf16* kv=kc+((size_t)(lo+j)*n_kv+kvh)*hd;
        float dot=0.f; for(int d=0;d<hd;d++) dot+=b2f(qv[d])*b2f(kv[d]);
        scores[j]=dot*scaling;
    }
    __syncthreads(); __shared__ float ssum;
    if (threadIdx.x==0){ float m=-1e30f; for(int j=0;j<len;j++) m=fmaxf(m,scores[j]);
        float sm=0.f; for(int j=0;j<len;j++){ float e=expf(scores[j]-m); scores[j]=e; sm+=e; } ssum=sm; }
    __syncthreads(); float inv=1.f/ssum;
    bf16* ov = out + ((size_t)b*n_heads + h)*hd;
    for (int d=threadIdx.x;d<hd;d+=blockDim.x){
        float acc=0.f; for(int j=0;j<len;j++){ const bf16* vv=vc+((size_t)(lo+j)*n_kv+kvh)*hd; acc+=scores[j]*b2f(vv[d]); }
        ov[d]=f2b(acc*inv);
    }
}

// Build per-layer-input from FP8 PLE lookup (×16) + context projection, combined.
// ple_lookup_raw:[T,ple_width] (dequant, unscaled)  context:[T,ple_width] (already
// projected, scaled by 1/sqrt(H), and per-256-block RMSNormed on host side? no —
// we do norm separately). Here we just do: out = (context + 16*lookup) * (1/sqrt2)
// for layer `li`'s 256-slice. Simpler: done per-layer in the combine path below.

// PLE gate: g[t,p] = gelu_tanh(g[t,p]) * per_layer_input[t, li, p].
// g is contiguous [T,PD]; pli is [T, n_layers, PD] (row stride = width = n_layers*PD).
__global__ void ple_gate_strided(bf16* __restrict__ g, const float* __restrict__ pli,
                                 int T, int PD, int width, int li){
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i>=T*PD) return;
    int t = i/PD, p = i%PD;
    g[i] = f2b(gelu_tanh(b2f(g[i])) * pli[(size_t)t*width + (size_t)li*PD + p]);
}

// Compute per_layer_input[T, n_layers, ple_dim] from FP8 lookup + projected context.
// lookup_raw:[T, width] dequant (unscaled). ctx_normed:[T, width] = RMSNorm-per-256 of
// (proj*1/sqrtH). out = (ctx_normed + 16*lookup_raw) * (1/sqrt2).  width=n_layers*ple_dim.
__global__ void ple_combine_kernel(const float* __restrict__ lookup_raw,
                                   const float* __restrict__ ctx_normed,
                                   float* __restrict__ out, int n){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) out[i] = (ctx_normed[i] + 16.0f*lookup_raw[i]) * 0.70710678f;
}

// RMSNorm over contiguous groups of `dim` within a [rows, dim] fp32 buffer (PLE
// projection norm over each 256-slice). w optional. in-place fp32.
__global__ void rmsnorm_f32_grouped(float* __restrict__ x, const bf16* __restrict__ w,
                                    int rows, int dim, float eps){
    int r = blockIdx.x; if (r>=rows) return;
    float* xr = x + (size_t)r*dim;
    __shared__ float sh[256];
    float ss=0.f; for(int i=threadIdx.x;i<dim;i+=blockDim.x){ float v=xr[i]; ss+=v*v; }
    sh[threadIdx.x]=ss; __syncthreads();
    for(int s=blockDim.x>>1;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
    float inv=rsqrtf(sh[0]/dim+eps);
    for(int i=threadIdx.x;i<dim;i+=blockDim.x){ float v=xr[i]*inv; if(w) v*=b2f(w[i]); xr[i]=v; }
}

// bf16→fp32 copy
__global__ void to_f32_kernel(const bf16* __restrict__ x, float* __restrict__ y, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=b2f(x[i]);
}
// logit softcap: y = tanh(x/cap)*cap
__global__ void softcap_kernel(float* __restrict__ x, int n, float cap){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]=tanhf(x[i]/cap)*cap;
}

// out[T,N] = x[T,K] @ W^T, with W[N,K] row-major bf16. cuBLAS column-major trick.
inline bool linear(cublasHandle_t h, const bf16* W, const bf16* x, bf16* out,
                   int T, int N, int K){
    const float alpha=1.f, beta=0.f;
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K,
        &alpha, W, CUDA_R_16BF, K, x, CUDA_R_16BF, K,
        &beta, out, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
}

// device inv_freq builders (host computes, returns device ptr; caller frees)
float* make_inv_freq_sliding(int hd, float base){
    int half=hd/2; std::vector<float> f(half);
    for(int i=0;i<half;i++) f[i]=1.f/powf(base,(float)(2*i)/hd);
    float* d; if(cudaMalloc(&d,half*sizeof(float))!=cudaSuccess) return nullptr;
    cudaMemcpy(d,f.data(),half*sizeof(float),cudaMemcpyHostToDevice); return d;
}
float* make_inv_freq_proportional(int hd, float base, float partial){
    int half=hd/2; std::vector<float> f(half,0.f);
    int rope_angles = (int)(partial*hd)/2;           // (partial*hd)//2
    for(int i=0;i<rope_angles;i++) f[i]=1.f/powf(base,(float)(2*i)/hd);
    float* d; if(cudaMalloc(&d,half*sizeof(float))!=cudaSuccess) return nullptr;
    cudaMemcpy(d,f.data(),half*sizeof(float),cudaMemcpyHostToDevice); return d;
}

} // namespace

// Process T new tokens at eng->n_past, writing K/V into the persistent cache and
// advancing n_past by T. Works for both prefill (n_past==0, T==prompt) and decode
// (T==1). Optionally captures debug activations / the last token's logits.
// Single-sequence today; the cache layout (per-layer [ctx,n_kv,hd], sharing-aware)
// is the basis for the paged multi-sequence version (parallelism-top-invariant).
static int e4b_step(e4b_engine* eng, Slot& slot, const int32_t* tokens, int T,
                    float* emb_out, float* l0_out, float* fin_out, float* logits_last_out){
    if (!eng || T<=0) return -1;
    const e4b::Config& c = eng->cfg;
    const int H=c.hidden_size, FF=c.intermediate_size, PD=c.ple_dim, V=c.vocab_size;
    const int W = c.ple_width();
    const float eps=c.rms_eps;
    const int n_past = slot.n_past;
    if (n_past + T > eng->max_ctx){
        fprintf(stderr,"e4b: context overflow (%d + %d > %d)\n", n_past, T, eng->max_ctx); return -1;
    }
    cudaSetDevice(eng->device_id);
    auto CKF=[&](cudaError_t e)->bool{ if(e!=cudaSuccess){ fprintf(stderr,"e4b step: %s\n",cudaGetErrorString(e)); return false;} return true; };

    const int QMAX=c.n_heads*c.global_head_dim, KVMAX=c.n_kv_heads*c.global_head_dim;
    // Single-token decode reuses persistent scratch (no per-token cudaMalloc/free, no RoPE-table
    // recompute) — the hot path and the CUDA-graph prerequisite. Prefill (T>1) mallocs per call.
    const bool persist = (T==1);
    if (persist && !eng->dec.ready) {
        auto& d = eng->dec; bool ok2=true;
        ok2&=CKF(cudaMalloc(&d.ids,sizeof(int32_t)));
        ok2&=CKF(cudaMalloc(&d.hidden,(size_t)H*2)); ok2&=CKF(cudaMalloc(&d.norm,(size_t)H*2)); ok2&=CKF(cudaMalloc(&d.tmpH,(size_t)H*2));
        ok2&=CKF(cudaMalloc(&d.q,(size_t)QMAX*2)); ok2&=CKF(cudaMalloc(&d.k,(size_t)KVMAX*2)); ok2&=CKF(cudaMalloc(&d.v,(size_t)KVMAX*2));
        ok2&=CKF(cudaMalloc(&d.attn,(size_t)QMAX*2)); ok2&=CKF(cudaMalloc(&d.gate,(size_t)FF*2)); ok2&=CKF(cudaMalloc(&d.up,(size_t)FF*2));
        ok2&=CKF(cudaMalloc(&d.act,(size_t)FF*2)); ok2&=CKF(cudaMalloc(&d.pleg,(size_t)PD*2)); ok2&=CKF(cudaMalloc(&d.ctx_bf,(size_t)W*2));
        ok2&=CKF(cudaMalloc(&d.ple_lookup,(size_t)W*4)); ok2&=CKF(cudaMalloc(&d.ple_ctx,(size_t)W*4)); ok2&=CKF(cudaMalloc(&d.pli,(size_t)W*4));
        ok2&=CKF(cudaMalloc(&d.logits_f,(size_t)V*4));
        ok2&=CKF(cudaMalloc(&d.npast,sizeof(int)));
        d.invf_s = make_inv_freq_sliding(c.head_dim, c.rope_theta_sliding);
        d.invf_f = make_inv_freq_proportional(c.global_head_dim, c.rope_theta_full, c.rope_partial_full);
        if(!ok2 || !d.invf_s || !d.invf_f) return -1;
        d.ready = true;
    }

    float *d_invf_s, *d_invf_f;
    int32_t* d_ids; bf16 *d_hidden,*d_norm,*d_tmpH,*d_q,*d_k,*d_v,*d_attn,*d_gate,*d_up,*d_act,*d_pleg;
    float *d_ple_lookup,*d_ple_ctx,*d_pli;
    bool ok=true;
    if (persist) {
        auto& d = eng->dec;
        d_invf_s=d.invf_s; d_invf_f=d.invf_f; d_ids=d.ids;
        d_hidden=d.hidden; d_norm=d.norm; d_tmpH=d.tmpH; d_q=d.q; d_k=d.k; d_v=d.v; d_attn=d.attn;
        d_gate=d.gate; d_up=d.up; d_act=d.act; d_pleg=d.pleg;
        d_ple_lookup=d.ple_lookup; d_ple_ctx=d.ple_ctx; d_pli=d.pli;
    } else {
        d_invf_s = make_inv_freq_sliding(c.head_dim, c.rope_theta_sliding);
        d_invf_f = make_inv_freq_proportional(c.global_head_dim, c.rope_theta_full, c.rope_partial_full);
        if(!d_invf_s||!d_invf_f) return -1;
        ok&=CKF(cudaMalloc(&d_ids,T*sizeof(int32_t)));
        ok&=CKF(cudaMalloc(&d_hidden,(size_t)T*H*2)); ok&=CKF(cudaMalloc(&d_norm,(size_t)T*H*2));
        ok&=CKF(cudaMalloc(&d_tmpH,(size_t)T*H*2));
        ok&=CKF(cudaMalloc(&d_q,(size_t)T*QMAX*2)); ok&=CKF(cudaMalloc(&d_k,(size_t)T*KVMAX*2));
        ok&=CKF(cudaMalloc(&d_v,(size_t)T*KVMAX*2)); ok&=CKF(cudaMalloc(&d_attn,(size_t)T*QMAX*2));
        ok&=CKF(cudaMalloc(&d_gate,(size_t)T*FF*2)); ok&=CKF(cudaMalloc(&d_up,(size_t)T*FF*2));
        ok&=CKF(cudaMalloc(&d_act,(size_t)T*FF*2)); ok&=CKF(cudaMalloc(&d_pleg,(size_t)T*PD*2));
        ok&=CKF(cudaMalloc(&d_ple_lookup,(size_t)T*W*4)); ok&=CKF(cudaMalloc(&d_ple_ctx,(size_t)T*W*4));
        ok&=CKF(cudaMalloc(&d_pli,(size_t)T*W*4));
        if(!ok) return -1;
    }
    cudaMemcpy(d_ids,tokens,T*sizeof(int32_t),cudaMemcpyHostToDevice);
    if (persist) cudaMemcpy(eng->dec.npast,&n_past,sizeof(int),cudaMemcpyHostToDevice);

    auto GRID=[&](int n){ return (n+255)/256; };

    // ── embedding (×sqrt(H)) ──
    embed_kernel<<<T,256>>>(eng->d_embed,d_ids,d_hidden,T,H,sqrtf((float)H));
    if(emb_out){ float* tmp; cudaMalloc(&tmp,(size_t)T*H*4); to_f32_kernel<<<GRID(T*H),256>>>(d_hidden,tmp,T*H);
        cudaMemcpy(emb_out,tmp,(size_t)T*H*4,cudaMemcpyDeviceToHost); cudaFree(tmp); }

    // ── PLE precompute (token-identity lookup + context projection, combined) ──
    e4b_ple_lookup_launch(eng->d_ple_fp8, eng->d_ple_scale, d_ids, d_ple_lookup, T, W);
    {
        bf16* d_ctx_bf = persist ? eng->dec.ctx_bf : nullptr;
        if (!persist) cudaMalloc(&d_ctx_bf,(size_t)T*W*2);
        linear(eng->cublas, eng->d_plm_proj, d_hidden, d_ctx_bf, T, W, H);
        to_f32_kernel<<<GRID(T*W),256>>>(d_ctx_bf,d_ple_ctx,T*W);
        if (!persist) cudaFree(d_ctx_bf);
        // The 1/sqrt(H) scale on the projection is a no-op before per-256 RMSNorm
        // (RMSNorm is scale-invariant), so we skip it and normalize directly.
        rmsnorm_f32_grouped<<<T*c.n_layers,256>>>(d_ple_ctx, eng->d_ple_proj_norm, T*c.n_layers, PD, eps);
    }
    ple_combine_kernel<<<GRID(T*W),256>>>(d_ple_lookup, d_ple_ctx, d_pli, T*W);

    // ── decoder layers ──
    for (int li=0; li<c.n_layers; ++li){
        const Layer& L = eng->layers[li];
        const bool full = (c.layer_types[li]==e4b::Attn::FULL);
        const int hd = full ? c.global_head_dim : c.head_dim;
        const int qd = c.n_heads*hd, kvd=c.n_kv_heads*hd;
        const bool shared = c.layer_shares_kv(li);
        const int window = full ? 0 : c.sliding_window;
        float* d_invf = full ? d_invf_f : d_invf_s;
        bf16* kcache = slot.kc[li];   // aliases the provider's for shared layers
        bf16* vcache = slot.vc[li];

        // residual = hidden; input_layernorm
        cudaMemcpy(d_tmpH,d_hidden,(size_t)T*H*2,cudaMemcpyDeviceToDevice);
        rmsnorm_kernel<<<T,256>>>(d_hidden,L.input_ln,d_norm,T,H,eps);

        // Decode quant (T==1). Two mutually exclusive strategies:
        //   use_q40 (GGUF Q4_0): every matmul via the SHARED dp4a MMVQ (mmvq.cuh), off the native
        //   on-disk nibbles — quantize the bf16 activation to Q8_1, mmvq → f32, cast to bf16.
        //   use_fp4 (safetensors): Q/K via FP8 (index precision), V/O+FFN via NVFP4 (content).
        // Prefill (T>1) under use_q40 dequants Q4_0→scratch then cuBLAS; the BF16 path keeps cuBLAS.
        const bool fp4_dec = eng->use_fp4 && T==1;
        const bool q40_dec = eng->use_q40 && T==1;
        auto q40dec = [&](bf16* y, const uint8_t* wb, const bf16* x, int out, int in){
            quantize_q8_1_bf16_kernel<<<in/32,32,0,0>>>(x, eng->d_q40_qa, eng->d_q40_da, eng->d_q40_sa, in);
            mmvq_launch(eng->d_fp4_yf, wb, eng->d_q40_qa, eng->d_q40_da, eng->d_q40_sa, in, out, 2, 0);
            e4bfp4::to_bf16<<<(out+255)/256,256,0,0>>>(eng->d_fp4_yf, y, out);
        };
        auto prefill_q40 = [&](bf16* y, const uint8_t* wb, const bf16* x, int out, int in){
            int64_t nblk=(int64_t)out*(in/32);
            dequant_q4_0_to_bf16_kernel<<<(unsigned)((nblk+255)/256),256,0,0>>>(wb, eng->d_q40_wdq, in, nblk);
            linear(eng->cublas, eng->d_q40_wdq, x, y, T, out, in);
        };

        // Q (always projected) — q_norm, rope at absolute positions
        if      (q40_dec)      q40dec(d_q, L.q40_wq, d_norm, qd, H);
        else if (fp4_dec)      e4bfp4::e4b_fp8_gemv_bf16(d_q, L.fp8_wq, d_norm, eng->d_fp4_xf, eng->d_fp4_yf, 0);
        else if (eng->use_q40) prefill_q40(d_q, L.q40_wq, d_norm, qd, H);
        else                   linear(eng->cublas,L.wq,d_norm,d_q,T,qd,H);
        head_rmsnorm_kernel<<<T*c.n_heads,256>>>(d_q,L.q_norm,T,c.n_heads,hd,eps);
        if (persist) rope_batch_kernel<<<c.n_heads,256>>>(d_q,d_invf,1,c.n_heads,hd,eng->dec.npast);
        else         rope_kernel<<<T*c.n_heads,256>>>(d_q,d_invf,T,c.n_heads,hd,n_past);

        // K/V: non-shared layers project + store into their cache; shared layers
        // skip (their provider — an earlier layer this same step — already wrote it).
        if (!shared){
            if (q40_dec){
                q40dec(d_k, L.q40_wk, d_norm, kvd, H);
                q40dec(d_v, L.q40_wv, d_norm, kvd, H);
            } else if (fp4_dec){
                e4bfp4::e4b_fp8_gemv_bf16(d_k, L.fp8_wk, d_norm, eng->d_fp4_xf, eng->d_fp4_yf, 0);
                e4bfp4::e4b_nvfp4_gemv_bf16(d_v, L.fp4_wv, d_norm, eng->d_fp4_xf, eng->d_fp4_yf, 0);
            } else if (eng->use_q40){
                prefill_q40(d_k, L.q40_wk, d_norm, kvd, H);
                prefill_q40(d_v, L.q40_wv, d_norm, kvd, H);
            } else {
                linear(eng->cublas,L.wk,d_norm,d_k,T,kvd,H);
                linear(eng->cublas,L.wv,d_norm,d_v,T,kvd,H);
            }
            head_rmsnorm_kernel<<<T*c.n_kv_heads,256>>>(d_k,L.k_norm,T,c.n_kv_heads,hd,eps);
            head_rmsnorm_kernel<<<T*c.n_kv_heads,256>>>(d_v,nullptr,T,c.n_kv_heads,hd,eps); // v_norm (no weight)
            if (persist){
                rope_batch_kernel<<<c.n_kv_heads,256>>>(d_k,d_invf,1,c.n_kv_heads,hd,eng->dec.npast);
                kv_store_dev_kernel<<<GRID(kvd),256>>>(d_k,kcache,kvd,eng->dec.npast);
                kv_store_dev_kernel<<<GRID(kvd),256>>>(d_v,vcache,kvd,eng->dec.npast);
            } else {
                rope_kernel<<<T*c.n_kv_heads,256>>>(d_k,d_invf,T,c.n_kv_heads,hd,n_past);
                kv_store_kernel<<<GRID(T*kvd),256>>>(d_k,kcache,T,kvd,n_past);
                kv_store_kernel<<<GRID(T*kvd),256>>>(d_v,vcache,T,kvd,n_past);
            }
        }

        // attention over the cache [0, n_past+T). Decode (T==1) uses the fixed-smem, device-
        // position flash kernel (CUDA-graph-able); prefill uses the dynamic-smem multi-query path.
        if (persist){
            const int smemF = (256 + hd) * (int)sizeof(float);
            attn_flash_decode_kernel<256><<<c.n_heads,256,smemF>>>(
                d_q,kcache,vcache,d_attn,eng->dec.npast,c.n_heads,c.n_kv_heads,hd,1.0f,window);
        } else {
            dim3 ag(c.n_heads,T); size_t sh=(size_t)(n_past+T)*sizeof(float);
            attn_cache_kernel<<<ag,256,sh>>>(d_q,kcache,vcache,d_attn,T,n_past,c.n_heads,c.n_kv_heads,hd,1.0f,window);
        }

        // o_proj → d_norm; post_attention_layernorm; hidden = residual + that
        if      (q40_dec)      q40dec(d_norm, L.q40_wo, d_attn, H, qd);
        else if (fp4_dec)      e4bfp4::e4b_nvfp4_gemv_bf16(d_norm, L.fp4_wo, d_attn, eng->d_fp4_xf, eng->d_fp4_yf, 0);
        else if (eng->use_q40) prefill_q40(d_norm, L.q40_wo, d_attn, H, qd);
        else                   linear(eng->cublas,L.wo,d_attn,d_norm,T,H,qd);
        rmsnorm_kernel<<<T,256>>>(d_norm,L.post_attn_ln,d_norm,T,H,eps);
        add_kernel<<<GRID(T*H),256>>>(d_tmpH,d_norm,T*H);
        cudaMemcpy(d_hidden,d_tmpH,(size_t)T*H*2,cudaMemcpyDeviceToDevice);

        // FFN: residual; pre_ff norm; GeGLU; post_ff norm; residual add
        cudaMemcpy(d_tmpH,d_hidden,(size_t)T*H*2,cudaMemcpyDeviceToDevice);
        rmsnorm_kernel<<<T,256>>>(d_hidden,L.pre_ff_ln,d_norm,T,H,eps);
        // FFN projections. Single-token decode (T==1) reads the NVFP4 weights via the
        // bandwidth-tuned GEMV (default; unless FUCINA_E4B_FP4=0); prefill (T>1) stays on the BF16
        // tensor-core GEMM. GeGLU and the surrounding norms/residual are identical.
        if (q40_dec){
            // Unfused gate/up + GeGLU + down (matches the batched path bit-for-bit; the fused
            // mmvq_glu keeps gate·up in f32 and would diverge from the batched unfused rounding).
            q40dec(d_gate, L.q40_gate, d_norm, FF, H);
            q40dec(d_up,   L.q40_up,   d_norm, FF, H);
            geglu_kernel<<<GRID(T*FF),256>>>(d_gate,d_up,d_act,T*FF);
            q40dec(d_norm, L.q40_down, d_act, H, FF);
        } else if (fp4_dec){
            e4bfp4::e4b_nvfp4_gemv_bf16(d_gate, L.fp4_gate, d_norm, eng->d_fp4_xf, eng->d_fp4_yf, 0);
            e4bfp4::e4b_nvfp4_gemv_bf16(d_up,   L.fp4_up,   d_norm, eng->d_fp4_xf, eng->d_fp4_yf, 0);
            geglu_kernel<<<GRID(T*FF),256>>>(d_gate,d_up,d_act,T*FF);
            e4bfp4::e4b_nvfp4_gemv_bf16(d_norm, L.fp4_down, d_act,  eng->d_fp4_xf, eng->d_fp4_yf, 0);
        } else if (eng->use_q40){
            prefill_q40(d_gate, L.q40_gate, d_norm, FF, H);
            prefill_q40(d_up,   L.q40_up,   d_norm, FF, H);
            geglu_kernel<<<GRID(T*FF),256>>>(d_gate,d_up,d_act,T*FF);
            prefill_q40(d_norm, L.q40_down, d_act,  H, FF);
        } else {
            linear(eng->cublas,L.w_gate,d_norm,d_gate,T,FF,H);
            linear(eng->cublas,L.w_up,d_norm,d_up,T,FF,H);
            geglu_kernel<<<GRID(T*FF),256>>>(d_gate,d_up,d_act,T*FF);
            linear(eng->cublas,L.w_down,d_act,d_norm,T,H,FF);
        }
        rmsnorm_kernel<<<T,256>>>(d_norm,L.post_ff_ln,d_norm,T,H,eps);
        add_kernel<<<GRID(T*H),256>>>(d_tmpH,d_norm,T*H);
        cudaMemcpy(d_hidden,d_tmpH,(size_t)T*H*2,cudaMemcpyDeviceToDevice);

        // PLE combine: residual; gate(hidden)→256; gelu; ×per_layer_input; proj; norm; add
        cudaMemcpy(d_tmpH,d_hidden,(size_t)T*H*2,cudaMemcpyDeviceToDevice);
        linear(eng->cublas,L.ple_in_gate,d_hidden,d_pleg,T,PD,H);
        ple_gate_strided<<<GRID(T*PD),256>>>(d_pleg, d_pli, T, PD, W, li);
        linear(eng->cublas,L.ple_proj,d_pleg,d_norm,T,H,PD);
        rmsnorm_kernel<<<T,256>>>(d_norm,L.post_ple_ln,d_norm,T,H,eps);
        add_kernel<<<GRID(T*H),256>>>(d_tmpH,d_norm,T*H);
        scale_kernel<<<GRID(T*H),256>>>(d_tmpH,L.layer_scalar,T*H);  // ×layer_scalar
        cudaMemcpy(d_hidden,d_tmpH,(size_t)T*H*2,cudaMemcpyDeviceToDevice);

        if (li==0 && l0_out){ float* tmp; cudaMalloc(&tmp,(size_t)T*H*4);
            to_f32_kernel<<<GRID(T*H),256>>>(d_hidden,tmp,T*H);
            cudaMemcpy(l0_out,tmp,(size_t)T*H*4,cudaMemcpyDeviceToHost); cudaFree(tmp); }
    }

    // final norm
    rmsnorm_kernel<<<T,256>>>(d_hidden,eng->d_final_norm,d_norm,T,H,eps);
    if(fin_out){ float* tmp; cudaMalloc(&tmp,(size_t)T*H*4);
        to_f32_kernel<<<GRID(T*H),256>>>(d_norm,tmp,T*H);
        cudaMemcpy(fin_out,tmp,(size_t)T*H*4,cudaMemcpyDeviceToHost); cudaFree(tmp); }

    // logits of the last token (tied head + softcap). ALWAYS a 1-row projection (last token
    // only), so the quantized head applies whenever quant is on — including prefill's last
    // token. Writes fp32 logits directly. use_q40 → native Q6_K head (mmvq_q6_k); the
    // safetensors NVFP4 path → FP8 head; BF16 fallback → cuBLAS.
    if (logits_last_out){
        const bf16* xrow = d_norm + (size_t)(T-1)*H;
        float* d_logits_f = persist ? eng->dec.logits_f : nullptr;
        if (!persist) cudaMalloc(&d_logits_f,(size_t)V*4);
        if (eng->use_q40){   // native Q6_K tied head via shared MMVQ
            quantize_q8_1_bf16_kernel<<<H/32,32,0,0>>>(xrow, eng->d_q40_qa, eng->d_q40_da, eng->d_q40_sa, H);
            mmvq_q6_k_launch(d_logits_f, eng->d_q6k_head, eng->d_q40_qa, eng->d_q40_da, H, V, 0);
        } else if (eng->use_fp4){
            e4bfp4::e4b_fp8_gemv_f32(d_logits_f, eng->fp8_head, xrow, eng->d_fp4_xf, 0);
        } else {
            bf16* d_logits_bf; cudaMalloc(&d_logits_bf,(size_t)V*2);
            linear(eng->cublas, eng->d_embed, xrow, d_logits_bf, 1, V, H);
            to_f32_kernel<<<GRID(V),256>>>(d_logits_bf,d_logits_f,V);
            cudaFree(d_logits_bf);
        }
        softcap_kernel<<<GRID(V),256>>>(d_logits_f,V,c.final_logit_softcap);
        cudaMemcpy(logits_last_out,d_logits_f,(size_t)V*4,cudaMemcpyDeviceToHost);
        if (!persist) cudaFree(d_logits_f);
    }

    cudaError_t err=cudaDeviceSynchronize();
    if (!persist){   // prefill scratch is per-call; decode scratch is persistent (freed at destroy)
        cudaFree(d_ids);cudaFree(d_hidden);cudaFree(d_norm);cudaFree(d_tmpH);cudaFree(d_q);cudaFree(d_k);
        cudaFree(d_v);cudaFree(d_attn);cudaFree(d_gate);cudaFree(d_up);cudaFree(d_act);cudaFree(d_pleg);
        cudaFree(d_ple_lookup);cudaFree(d_ple_ctx);cudaFree(d_pli);
        cudaFree(d_invf_s);cudaFree(d_invf_f);
    }
    if(err!=cudaSuccess){ fprintf(stderr,"e4b step sync: %s\n",cudaGetErrorString(err)); return -1; }
    slot.n_past += T;     // commit the new tokens to the cache
    return 0;
}

extern "C" void e4b_engine_reset(e4b_engine_t *eng){ if(eng) eng->slots[0].n_past=0; }
extern "C" int  e4b_engine_n_past(const e4b_engine_t *eng){ return eng ? eng->slots[0].n_past : 0; }

extern "C" int e4b_engine_prefill(e4b_engine_t *eng, const int32_t *tokens, int n_tokens, float *logits_out){
    if(!eng) return -1;
    eng->slots[0].n_past = 0;                  // fresh sequence on slot 0
    return e4b_step(eng, eng->slots[0], tokens, n_tokens, nullptr, nullptr, nullptr, logits_out);
}

extern "C" int e4b_engine_decode(e4b_engine_t *eng, int32_t token, float *logits_out){
    if(!eng) return -1;
    return e4b_step(eng, eng->slots[0], &token, 1, nullptr, nullptr, nullptr, logits_out);
}

extern "C" int e4b_engine_forward_debug(e4b_engine_t *eng, const int32_t *tokens, int n_tokens,
                                        float *emb_out, float *l0_out, float *fin_out, float *logits_last_out){
    if(!eng) return -1;
    eng->slots[0].n_past = 0;
    return e4b_step(eng, eng->slots[0], tokens, n_tokens, emb_out, l0_out, fin_out, logits_last_out);
}

// Greedy generation: prefill `prompt`, then argmax-decode up to max_new tokens,
// stopping at any id in stop_ids. Returns the count written to out_tokens (≥0), -1 on error.
extern "C" int e4b_engine_generate_greedy(e4b_engine_t *eng, const int32_t *prompt, int n_prompt,
                                          int32_t *out_tokens, int max_new,
                                          const int32_t *stop_ids, int n_stop){
    if(!eng || n_prompt<=0) return -1;
    const int V = eng->cfg.vocab_size;
    std::vector<float> logits(V);
    if (e4b_engine_prefill(eng, prompt, n_prompt, logits.data())!=0) return -1;
    auto argmax=[&](const std::vector<float>& x){ int b=0; for(int i=1;i<V;i++) if(x[i]>x[b]) b=i; return b; };
    auto is_stop=[&](int32_t t){ for(int i=0;i<n_stop;i++) if(stop_ids[i]==t) return true; return false; };
    int n=0;
    int32_t next = argmax(logits);
    while (n < max_new){
        if (is_stop(next)) break;
        out_tokens[n++] = next;
        if (n >= max_new) break;
        if (e4b_engine_decode(eng, next, logits.data())!=0) return -1;
        next = argmax(logits);
    }
    return n;
}

// ════════════════════════════════════════════════════════════════════════════
// Continuous batching — multiple sequences decoded in ONE weight pass.
//
// Prefill (seq_add) is per-sequence (e4b_step on the slot). Decode is batched:
// step_batch feeds B slots one token each, runs all the projection/FFN/PLE GEMMs
// over [B] rows (weights read once for B tokens — the throughput win), and does
// attention per-sequence against each slot's own KV cache. Greedy sampling for now.
// ════════════════════════════════════════════════════════════════════════════
static int e4b_step_batch_decode(e4b_engine* eng, const int* slot_ids, const int32_t* in_tokens,
                                 int B, int32_t* out_tokens){
    if (!eng || B<=0) return -1;
    const e4b::Config& c = eng->cfg;
    const int H=c.hidden_size, FF=c.intermediate_size, PD=c.ple_dim, V=c.vocab_size, W=c.ple_width();
    const float eps=c.rms_eps;
    cudaSetDevice(eng->device_id);
    auto GRID=[&](int n){ return (n+255)/256; };

    // per-sequence positions + a max for shared-mem sizing
    std::vector<int> hpos(B); int maxP=0;
    for (int i=0;i<B;i++){ Slot& s=eng->slots[slot_ids[i]];
        if (s.n_past+1>eng->max_ctx){ fprintf(stderr,"e4b: slot %d ctx overflow\n",slot_ids[i]); return -1; }
        hpos[i]=s.n_past; if(s.n_past>maxP) maxP=s.n_past; }

    float* d_invf_s = make_inv_freq_sliding(c.head_dim, c.rope_theta_sliding);
    float* d_invf_f = make_inv_freq_proportional(c.global_head_dim, c.rope_theta_full, c.rope_partial_full);
    if(!d_invf_s||!d_invf_f) return -1;

    const int QMAX=c.n_heads*c.global_head_dim, KVMAX=c.n_kv_heads*c.global_head_dim;
    int32_t* d_ids; bf16 *d_hidden,*d_norm,*d_tmpH,*d_q,*d_k,*d_v,*d_attn,*d_gate,*d_up,*d_act,*d_pleg;
    float *d_ple_lookup,*d_ple_ctx,*d_pli;
    int* d_pos; bf16 **d_kptr, **d_vptr;
    cudaMalloc(&d_ids,B*sizeof(int32_t)); cudaMalloc(&d_pos,B*sizeof(int));
    cudaMalloc(&d_kptr,B*sizeof(bf16*)); cudaMalloc(&d_vptr,B*sizeof(bf16*));
    cudaMalloc(&d_hidden,(size_t)B*H*2); cudaMalloc(&d_norm,(size_t)B*H*2); cudaMalloc(&d_tmpH,(size_t)B*H*2);
    cudaMalloc(&d_q,(size_t)B*QMAX*2); cudaMalloc(&d_k,(size_t)B*KVMAX*2); cudaMalloc(&d_v,(size_t)B*KVMAX*2);
    cudaMalloc(&d_attn,(size_t)B*QMAX*2); cudaMalloc(&d_gate,(size_t)B*FF*2); cudaMalloc(&d_up,(size_t)B*FF*2);
    cudaMalloc(&d_act,(size_t)B*FF*2); cudaMalloc(&d_pleg,(size_t)B*PD*2);
    cudaMalloc(&d_ple_lookup,(size_t)B*W*4); cudaMalloc(&d_ple_ctx,(size_t)B*W*4); cudaMalloc(&d_pli,(size_t)B*W*4);
    cudaMemcpy(d_ids,in_tokens,B*sizeof(int32_t),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos,hpos.data(),B*sizeof(int),cudaMemcpyHostToDevice);

    embed_kernel<<<B,256>>>(eng->d_embed,d_ids,d_hidden,B,H,sqrtf((float)H));
    e4b_ple_lookup_launch(eng->d_ple_fp8, eng->d_ple_scale, d_ids, d_ple_lookup, B, W);
    { bf16* d_ctx_bf; cudaMalloc(&d_ctx_bf,(size_t)B*W*2);
      linear(eng->cublas, eng->d_plm_proj, d_hidden, d_ctx_bf, B, W, H);
      to_f32_kernel<<<GRID(B*W),256>>>(d_ctx_bf,d_ple_ctx,B*W); cudaFree(d_ctx_bf);
      rmsnorm_f32_grouped<<<B*c.n_layers,256>>>(d_ple_ctx, eng->d_ple_proj_norm, B*c.n_layers, PD, eps); }
    ple_combine_kernel<<<GRID(B*W),256>>>(d_ple_lookup, d_ple_ctx, d_pli, B*W);

    std::vector<bf16*> hk(B), hv(B);
    for (int li=0; li<c.n_layers; ++li){
        const Layer& L=eng->layers[li];
        const bool full=(c.layer_types[li]==e4b::Attn::FULL);
        const int hd=full?c.global_head_dim:c.head_dim, qd=c.n_heads*hd, kvd=c.n_kv_heads*hd;
        const bool shared=c.layer_shares_kv(li); const int window=full?0:c.sliding_window;
        float* d_invf=full?d_invf_f:d_invf_s;
        // Projection over B rows. Under use_q40 the BF16 weights were freed, so dequant the
        // Q4_0 nibbles → scratch then GEMM (weights still read once for the B-token batch).
        auto bproj = [&](bf16* y, __nv_bfloat16* Wbf, const uint8_t* wb40, const bf16* x, int out, int in){
            if (eng->use_q40){
                // Batched MMVQ: each weight ROW is read once and dp4a'd against all B tokens
                // (token-major qx/dx/sx). Per-token math is identical to mmvq_launch, so batched
                // == single-seq decode bit-for-bit — the continuous-batching weight-reuse win.
                quantize_q8_1_bf16_kernel<<<(B*in)/32,32,0,0>>>(x, eng->d_q40_qa, eng->d_q40_da, eng->d_q40_sa, B*in);
                mmvq_batched_launch(eng->d_fp4_yf, wb40, eng->d_q40_qa, eng->d_q40_da, eng->d_q40_sa, in, out, B, 2, 0);
                e4bfp4::to_bf16<<<(unsigned)(((size_t)B*out+255)/256),256,0,0>>>(eng->d_fp4_yf, y, B*out);
            } else linear(eng->cublas, Wbf, x, y, B, out, in);
        };
        // gather this layer's per-slot cache pointers
        for (int b=0;b<B;b++){ Slot& s=eng->slots[slot_ids[b]]; hk[b]=s.kc[li]; hv[b]=s.vc[li]; }
        cudaMemcpy(d_kptr,hk.data(),B*sizeof(bf16*),cudaMemcpyHostToDevice);
        cudaMemcpy(d_vptr,hv.data(),B*sizeof(bf16*),cudaMemcpyHostToDevice);

        cudaMemcpy(d_tmpH,d_hidden,(size_t)B*H*2,cudaMemcpyDeviceToDevice);
        rmsnorm_kernel<<<B,256>>>(d_hidden,L.input_ln,d_norm,B,H,eps);
        bproj(d_q,L.wq,L.q40_wq,d_norm,qd,H);
        head_rmsnorm_kernel<<<B*c.n_heads,256>>>(d_q,L.q_norm,B,c.n_heads,hd,eps);
        rope_batch_kernel<<<B*c.n_heads,256>>>(d_q,d_invf,B,c.n_heads,hd,d_pos);
        if (!shared){
            bproj(d_k,L.wk,L.q40_wk,d_norm,kvd,H);
            bproj(d_v,L.wv,L.q40_wv,d_norm,kvd,H);
            head_rmsnorm_kernel<<<B*c.n_kv_heads,256>>>(d_k,L.k_norm,B,c.n_kv_heads,hd,eps);
            rope_batch_kernel<<<B*c.n_kv_heads,256>>>(d_k,d_invf,B,c.n_kv_heads,hd,d_pos);
            head_rmsnorm_kernel<<<B*c.n_kv_heads,256>>>(d_v,nullptr,B,c.n_kv_heads,hd,eps);
            kv_store_batch_kernel<<<GRID(B*kvd),256>>>(d_k,d_kptr,d_pos,B,kvd);
            kv_store_batch_kernel<<<GRID(B*kvd),256>>>(d_v,d_vptr,d_pos,B,kvd);
        }
        dim3 ag(c.n_heads,B); size_t sh=(size_t)(maxP+1)*sizeof(float);
        attn_batch_kernel<<<ag,256,sh>>>(d_q,d_kptr,d_vptr,d_pos,d_attn,B,c.n_heads,c.n_kv_heads,hd,1.0f,window);
        bproj(d_norm,L.wo,L.q40_wo,d_attn,H,qd);
        rmsnorm_kernel<<<B,256>>>(d_norm,L.post_attn_ln,d_norm,B,H,eps);
        add_kernel<<<GRID(B*H),256>>>(d_tmpH,d_norm,B*H);
        cudaMemcpy(d_hidden,d_tmpH,(size_t)B*H*2,cudaMemcpyDeviceToDevice);

        cudaMemcpy(d_tmpH,d_hidden,(size_t)B*H*2,cudaMemcpyDeviceToDevice);
        rmsnorm_kernel<<<B,256>>>(d_hidden,L.pre_ff_ln,d_norm,B,H,eps);
        bproj(d_gate,L.w_gate,L.q40_gate,d_norm,FF,H);
        bproj(d_up,L.w_up,L.q40_up,d_norm,FF,H);
        geglu_kernel<<<GRID(B*FF),256>>>(d_gate,d_up,d_act,B*FF);
        bproj(d_norm,L.w_down,L.q40_down,d_act,H,FF);
        rmsnorm_kernel<<<B,256>>>(d_norm,L.post_ff_ln,d_norm,B,H,eps);
        add_kernel<<<GRID(B*H),256>>>(d_tmpH,d_norm,B*H);
        cudaMemcpy(d_hidden,d_tmpH,(size_t)B*H*2,cudaMemcpyDeviceToDevice);

        cudaMemcpy(d_tmpH,d_hidden,(size_t)B*H*2,cudaMemcpyDeviceToDevice);
        linear(eng->cublas,L.ple_in_gate,d_hidden,d_pleg,B,PD,H);
        ple_gate_strided<<<GRID(B*PD),256>>>(d_pleg,d_pli,B,PD,W,li);
        linear(eng->cublas,L.ple_proj,d_pleg,d_norm,B,H,PD);
        rmsnorm_kernel<<<B,256>>>(d_norm,L.post_ple_ln,d_norm,B,H,eps);
        add_kernel<<<GRID(B*H),256>>>(d_tmpH,d_norm,B*H);
        scale_kernel<<<GRID(B*H),256>>>(d_tmpH,L.layer_scalar,B*H);
        cudaMemcpy(d_hidden,d_tmpH,(size_t)B*H*2,cudaMemcpyDeviceToDevice);
    }
    rmsnorm_kernel<<<B,256>>>(d_hidden,eng->d_final_norm,d_norm,B,H,eps);
    // logits for all B rows → argmax each. use_q40 → native Q6_K head per row (same kernel as
    // single-seq, so batched == independent); else cuBLAS over the BF16 tied embedding.
    float* d_logits_f; cudaMalloc(&d_logits_f,(size_t)B*V*4);
    if (eng->use_q40){   // native Q6_K head, batched over B rows (read once); bit-matches single-seq
        quantize_q8_1_bf16_kernel<<<(B*H)/32,32,0,0>>>(d_norm, eng->d_q40_qa, eng->d_q40_da, eng->d_q40_sa, B*H);
        mmvq_q6_k_batched_launch(d_logits_f, eng->d_q6k_head, eng->d_q40_qa, eng->d_q40_da, H, V, B, 0);
    } else {
        bf16* d_logits_bf; cudaMalloc(&d_logits_bf,(size_t)B*V*2);
        linear(eng->cublas, eng->d_embed, d_norm, d_logits_bf, B, V, H);
        to_f32_kernel<<<GRID(B*V),256>>>(d_logits_bf,d_logits_f,B*V);
        cudaFree(d_logits_bf);
    }
    softcap_kernel<<<GRID(B*V),256>>>(d_logits_f,B*V,c.final_logit_softcap);
    std::vector<float> hl((size_t)B*V);
    cudaMemcpy(hl.data(),d_logits_f,(size_t)B*V*4,cudaMemcpyDeviceToHost);
    cudaError_t err=cudaDeviceSynchronize();
    for (int b=0;b<B;b++){ const float* r=&hl[(size_t)b*V]; int am=0; for(int i=1;i<V;i++) if(r[i]>r[am]) am=i;
        out_tokens[b]=am; eng->slots[slot_ids[b]].n_past += 1; }

    cudaFree(d_ids);cudaFree(d_pos);cudaFree(d_kptr);cudaFree(d_vptr);cudaFree(d_hidden);cudaFree(d_norm);
    cudaFree(d_tmpH);cudaFree(d_q);cudaFree(d_k);cudaFree(d_v);cudaFree(d_attn);cudaFree(d_gate);cudaFree(d_up);
    cudaFree(d_act);cudaFree(d_pleg);cudaFree(d_ple_lookup);cudaFree(d_ple_ctx);cudaFree(d_pli);
    cudaFree(d_logits_f);cudaFree(d_invf_s);cudaFree(d_invf_f);
    return err==cudaSuccess?0:-1;
}

extern "C" int e4b_engine_seq_capacity(e4b_engine_t *eng){
    if(!eng) return 0; int n=0; for(size_t i=1;i<eng->slots.size();++i) if(!eng->slots[i].active) n++; return n;
}

extern "C" int e4b_engine_seq_add(e4b_engine_t *eng, const int32_t *prompt, int n_prompt,
                                  int32_t *first_token_out){
    if(!eng || n_prompt<=0) return -1;
    int sid=-1; for(size_t i=1;i<eng->slots.size();++i) if(!eng->slots[i].active){ sid=(int)i; break; }
    if(sid<0){ fprintf(stderr,"e4b: no free sequence slot\n"); return -1; }
    Slot& s=eng->slots[sid];
    if (s.kc.empty() && !e4b_slot_alloc(eng,s)){ fprintf(stderr,"e4b: slot %d alloc failed\n",sid); return -1; }
    s.active=true; s.n_past=0;
    std::vector<float> logits(eng->cfg.vocab_size);
    if (e4b_step(eng, s, prompt, n_prompt, nullptr,nullptr,nullptr, logits.data())!=0){ s.active=false; return -1; }
    if (first_token_out){ int am=0; int V=eng->cfg.vocab_size; for(int i=1;i<V;i++) if(logits[i]>logits[am]) am=i; *first_token_out=am; }
    return sid;
}

extern "C" int e4b_engine_step_batch(e4b_engine_t *eng, const int *slots, const int32_t *in_tokens,
                                     int B, int32_t *out_tokens){
    return e4b_step_batch_decode(eng, slots, in_tokens, B, out_tokens);
}

extern "C" void e4b_engine_seq_remove(e4b_engine_t *eng, int slot){
    if(!eng || slot<1 || slot>=(int)eng->slots.size()) return;
    eng->slots[slot].active=false; eng->slots[slot].n_past=0;   // keep caches for reuse
}
