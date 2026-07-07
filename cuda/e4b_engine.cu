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

// ── MTP draft head ("gemma4-assistant") for speculative decode ───────────────
// A tiny (78M) recurrent multi-token-predictor: 4 Q-only decoder layers that attend
// the TARGET model's KV cache (last sliding + last full layer) and, from the target's
// post-final-norm hidden h + the last token, predict the next few tokens. Greedy
// speculation with it is LOSSLESS (verify accepts only target-matching drafts). All
// weights dequantized Q4_0→BF16 at load; norms/scales/rope kept F32. See docs/e4b-mtp-plan.md.
struct E4bMtpLayer {
    __nv_bfloat16 *wq=nullptr, *wo=nullptr;            // attn_q [AH→qdim], attn_output [qdim→AH]
    __nv_bfloat16 *gate=nullptr, *up=nullptr, *down=nullptr;  // FFN
    // Native Q4_0 block streams (Stage 2): the drafter forward decodes these straight off the
    // on-disk nibbles via the SHARED dp4a MMVQ (mmvq.cuh), replacing the cuBLAS BF16 GEMVs.
    // Loaded only when the assistant GGUF projections are Q4_0 (use_q40); BF16 above is freed.
    uint8_t *q40_wq=nullptr, *q40_wo=nullptr, *q40_gate=nullptr, *q40_up=nullptr, *q40_down=nullptr;
    float *attn_norm=nullptr, *q_norm=nullptr, *post_attn_norm=nullptr;
    float *ffn_norm=nullptr, *post_ffw_norm=nullptr;
    // BF16 copies of the F32 norms above — rmsnorm_kernel/head_rmsnorm_kernel take BF16
    // weights, so the drafter forward reads these (converted once at load).
    __nv_bfloat16 *attn_norm_b=nullptr, *q_norm_b=nullptr, *post_attn_norm_b=nullptr;
    __nv_bfloat16 *ffn_norm_b=nullptr, *post_ffw_norm_b=nullptr;
    float  out_scale=1.0f;
    bool   is_global=false;                            // blk.3: full attn (hd 512), else sliding (256)
    int    head_dim=0, qdim=0;                         // hd = is_global?512:256; qdim = q_heads*hd
};
// Max drafts per spec round (host K cap). The verify forward processes at most K+1 rows,
// and the drafter's d_argmax holds K ids. Sizes the persistent verify scratch.
#define E4B_MTP_KMAX 16
#define E4B_MTP_VROWS (E4B_MTP_KMAX + 1)

struct E4bMtp {
    bool loaded=false;
    int  AH=0, FF=0, n_layers=0, q_heads=0, kv_heads=0, vocab=0, H_out=0;
    int  sliding_window=512;
    float rms_eps=1e-6f, rope_theta_sliding=1e4f, rope_theta_global=1e6f;
    __nv_bfloat16 *pre_proj=nullptr;    // [5120 → AH]  (concat(embed·√H, h))
    __nv_bfloat16 *post_proj=nullptr;   // [AH → H_out] (next recurrent h)
    __nv_bfloat16 *unembed=nullptr;     // token_embd [AH → vocab]
    // Native Q4_0 block streams for the global projections (Stage 2 dp4a drafter forward).
    uint8_t *q40_pre=nullptr, *q40_post=nullptr, *q40_unembed=nullptr;
    int  use_q40=0;                     // drafter forward uses dp4a Q4_0 GEMVs (assistant is Q4_0)
    float *out_norm=nullptr;            // [AH]
    __nv_bfloat16 *out_norm_b=nullptr;  // BF16 copy of out_norm for rmsnorm_kernel
    float *rope_freqs=nullptr;          // [head_dim] (unused by rope_kernel; kept for parity)
    std::vector<E4bMtpLayer> layers;
    uint64_t dev_bytes=0;

    // RoPE inv_freq tables (built once at load): sliding (head_dim, θ_sliding) and global
    // (global_head_dim, θ_global). The drafter is Q-only; q is rope'd with these.
    float *invf_sliding=nullptr, *invf_global=nullptr;
    int    global_head_dim=0;           // = target cfg.global_head_dim (for the blk.3 path)

    // Persistent drafter-forward scratch (allocated lazily on first e4b_mtp_forward).
    bool   scratch_ready=false;
    __nv_bfloat16 *d_xh=nullptr;        // [2*H_out] concat(embed·√H, h)
    __nv_bfloat16 *d_cur=nullptr;       // [AH] recurrent layer activation
    __nv_bfloat16 *d_t1=nullptr, *d_t2=nullptr;          // [AH] temporaries
    __nv_bfloat16 *d_q=nullptr, *d_attn=nullptr;         // [q_heads*global_head_dim]
    __nv_bfloat16 *d_ffa=nullptr, *d_ffb=nullptr;        // [FF]
    __nv_bfloat16 *d_logits_bf=nullptr; // [vocab] bf16 GEMV output (unembed)
    float         *d_logits=nullptr;    // [vocab]
    int           *d_pos=nullptr;       // device RoPE/attn position
    int           *d_argmax=nullptr;    // device argmax results [E4B_MTP_KMAX]
    int32_t       *d_tok=nullptr;       // device token id (for embed_kernel; chained on-GPU)
    float         *d_draft_h=nullptr;   // [H_out] recurrent h fed to the drafter chain — a FIXED
                                        // device pointer so the captured graph references it
    // dp4a drafter scratch (Stage 2): Q8_1-quantized activation + f32 MMVQ output, sized for
    // the largest projection. qa[max_in] int8, da/sa[max_in/32], yf[max_out] f32.
    // max_in = 2*H_out (pre_proj); max_out = vocab (unembed). Reused across every GEMV.
    int8_t  *dq_qa=nullptr;             // [max_in]
    float   *dq_da=nullptr;             // [max_in/32]
    int32_t *dq_sa=nullptr;            // [max_in/32]
    float   *dq_yf=nullptr;            // [max_out]

    // CUDA-graph capture of the drafter forward (mirror of the dense mtp_graph and the E4B
    // decode graph). The ~50-kernel/GEMV sequence is captured ONCE on mstream and replayed per
    // drafted token; only d_pos changes per replay (one tiny H2D OUTSIDE the graph) and the
    // chained input token flows through d_tok on-device (argmax writes the next id into it).
    // FUCINA_E4B_MTP_NOGRAPH=1 forces the per-launch path; capture failure auto-falls-back.
    cudaStream_t    mstream=nullptr;
    cudaGraphExec_t mgraph=nullptr;
    bool            graph_ready=false, graph_disabled=false;
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
        // Split-K flash-decode partials: each (head,split) block writes a local (m,l) and an
        // un-normalized acc[hd]; a combine kernel merges the splits. Sized [n_heads*SPLITS].
        float *fa_m=nullptr,*fa_l=nullptr,*fa_acc=nullptr;   // fa_acc:[n_heads*SPLITS*hd_max]
        int  *npast=nullptr;   // device-resident position for the graph-captured decode kernels
        // CUDA-graph decode: the T==1 forward runs on cstream; captured once and replayed per
        // token (only dec.ids/dec.npast change, updated OUTSIDE the graph; logits D2H'd after the
        // replay). FUCINA_E4B_NOGRAPH=1 keeps the per-kernel-launch path.
        cudaStream_t    cstream=nullptr;
        cudaGraphExec_t gexec=nullptr;
        bool graph_ready=false, graph_disabled=false;
    } dec;
    // Prefill (T>1) dequant scratch: when use_q40 the BF16 projection weights are freed,
    // so prefill dequants each Q4_0 projection → this reused BF16 buffer for the cuBLAS GEMM.
    // Sized to the largest projection (FF×H). Serial reuse is safe — all on the null stream.
    __nv_bfloat16 *d_q40_wdq=nullptr;

    // ── Persistent SPEC-VERIFY scratch (T ≤ E4B_MTP_KMAX+1 rows) ──
    // The verify path (e4b_step with out_argmax_rows set) is the ONLY T>1 forward in the hot
    // spec loop; it ran the prefill malloc path (~17 cudaMalloc/free + 2 RoPE-table rebuilds
    // + d_lg[T*V]/d_lg_bf/d_am mallocs) EVERY round. These persistent buffers, allocated once
    // for the max verify width, remove all per-round allocation. Bit-identical to the malloc
    // path (same kernels, same data). NULL until the first verify; freed at destroy.
    struct VScratch {
        bool ready=false;
        int32_t* ids=nullptr;
        __nv_bfloat16 *hidden=nullptr,*norm=nullptr,*tmpH=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,
                      *attn=nullptr,*gate=nullptr,*up=nullptr,*act=nullptr,*pleg=nullptr,*ctx_bf=nullptr;
        float *ple_lookup=nullptr,*ple_ctx=nullptr,*pli=nullptr,*invf_s=nullptr,*invf_f=nullptr;
        float *lg=nullptr; __nv_bfloat16 *lg_bf=nullptr; int *am=nullptr;   // verify head [VROWS*V]
        float *logits_last=nullptr; __nv_bfloat16 *logits_last_bf=nullptr;  // last-row logits [V]
        // CUDA-graph capture of the verify forward (Stage 2). The forward runs at a FIXED T=VROWS
        // on cstream and is captured once / replayed per round; only npast (device base position)
        // and ids change per replay (H2D'd OUTSIDE the captured region), and fin/argmax are D2H'd
        // after the replay. FUCINA_E4B_VERIFY_NOGRAPH=1 forces per-launch; capture failure falls
        // back. fin is the persistent device post-final-norm buffer [VROWS*H] (the prior path
        // cudaMalloc'd it per call inside the forward — illegal under capture).
        int   *npast=nullptr;   // device base position (n_past) for the verify-graph kernels
        float *fin=nullptr;     // [VROWS*H] device post-final-norm rows (D2H'd after replay)
        int    cap_T=0;         // T the graph was captured at (must match to replay)
        cudaStream_t    cstream=nullptr;
        cudaGraphExec_t gexec=nullptr;
        bool graph_ready=false, graph_disabled=false;
    } vs;

    // ── KV cache: one Slot per concurrent sequence ──
    // Per layer per slot: FULL layers cache [max_ctx, n_kv, hd] bf16; SLIDING layers
    // cache only [sliding_cap, n_kv, hd] as a per-position RING (position p → slot
    // p % sliding_cap). Sliding-window attention reads only the last `sliding_window`
    // positions, so the ring need not equal ctx — capping it makes the sliding cache
    // (the dominant, ctx-scaling term) nearly ctx-independent. Shared layers (≥24) do
    // NOT own a cache — they alias the provider layer's WITHIN the same slot. slots[0]
    // backs the single-sequence prefill/decode/generate API; seq_add/step_batch use any.
    int max_ctx = 0;
    int sliding_cap = 0;   // ring capacity for sliding layers (== max_ctx ⇒ no wrap)
    int max_seqs = 0;
    std::vector<Slot> slots;
    // provider layer index per attention type (last non-shared of that type)
    int prov_sliding=-1, prov_full=-1;

    // Optional MTP draft head for speculative decode (e4b_engine_load_assistant).
    E4bMtp mtp;

    // Cumulative speculative-decode acceptance counters (exposed via the *_spec_* getters
    // for the server /metrics). steps = verify rounds; drafted = drafts proposed; accepted =
    // drafts that matched greedy; emitted = tokens committed (g + accepted per round).
    long spec_steps=0, spec_drafted=0, spec_accepted=0, spec_emitted=0;
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
            // FULL layers hold the whole ctx; SLIDING layers hold only the ring.
            const int rows = full ? eng->max_ctx : eng->sliding_cap;
            size_t bytes=(size_t)rows*c.n_kv_heads*hd*sizeof(__nv_bfloat16);
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

// Upload an F32 GGUF tensor to a device float buffer (norms / scales / rope freqs). Optionally
// checks the element count. Adds to *total. Returns nullptr (logged) on any miss/dtype mismatch.
float* up_f32_gguf(const e4bgguf::GgufFile& g, const std::string& name,
                   int64_t expect_elems, uint64_t* total) {
    uint64_t off, n_el; uint32_t gtype;
    if (!g.find(name.c_str(), &off, &n_el, &gtype)) {
        fprintf(stderr, "e4b: missing GGUF tensor %s\n", name.c_str()); return nullptr;
    }
    if (gtype != e4bgguf::GGML_TYPE_F32) {
        fprintf(stderr, "e4b: %s not F32 (ggml type %u)\n", name.c_str(), gtype); return nullptr;
    }
    if (expect_elems > 0 && (int64_t)n_el != expect_elems) {
        fprintf(stderr, "e4b: F32 %s has %llu elems, expected %lld\n",
                name.c_str(), (unsigned long long)n_el, (long long)expect_elems); return nullptr;
    }
    const size_t nbytes = (size_t)n_el * sizeof(float);
    float* d = nullptr;
    if (cudaMalloc(&d, nbytes) != cudaSuccess) {
        fprintf(stderr, "e4b: cudaMalloc %zu for %s failed\n", nbytes, name.c_str()); return nullptr; }
    if (cudaMemcpy(d, g.data + off, nbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "e4b: H2D %s failed\n", name.c_str()); cudaFree(d); return nullptr; }
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
        const int H_ = H, FF = c.intermediate_size, V_ = V, S = 32;  // S = max concurrent seqs (batch-scratch ceiling)
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

extern "C" e4b_engine_t *e4b_engine_create(const char *model_path, uint32_t context_size,
                                           int max_seqs, int device_id) {
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

    // ── KV cache slots: memory-budgeted auto-fit ────────────────────────────
    // E4B previously allocated per-slot KV for the FULL requested ctx × a hardcoded
    // 8 slots with NO memory check — the dominant term in the unified-memory OOMs
    // (8 × ~14 GiB/slot @262144 ctx ≫ 128 GB). Now, AFTER every weight/quant copy
    // is resident, query free device memory (and the optional FUCINA_MEM_BUDGET_GB
    // *total-device* cap) and shrink max_seqs and/or max_ctx so the KV cache provably
    // fits. Honor the caller's desired concurrency first, then context, with a hard
    // floor — refuse cleanly rather than letting the kernel OOM-kill the box.
    for (int j=0;j<c.kv_share_start();++j)   // provider = last non-shared layer of each attn type
        (c.layer_types[j]==e4b::Attn::FULL ? eng->prov_full : eng->prov_sliding) = j;

    // Per-token, per-slot KV bytes, SPLIT by attention type (shared layers alias, so
    // they cost 0). FULL layers scale with ctx; SLIDING layers scale with the ring cap
    // (≤ ctx) because sliding attention only reads the last `sliding_window` positions.
    // Mirrors e4b_slot_alloc so the budget matches what we actually cudaMalloc.
    uint64_t full_per_tok=0, slid_per_tok=0;
    for (int i=0;i<c.n_layers;++i){
        if (c.layer_shares_kv(i)) continue;
        const bool full=(c.layer_types[i]==e4b::Attn::FULL);
        const int  hd  = full?c.global_head_dim:c.head_dim;
        const uint64_t b = 2ull*(uint64_t)c.n_kv_heads*hd*sizeof(__nv_bfloat16);   // K + V
        if (full) full_per_tok += b; else slid_per_tok += b;
    }

    // Sliding ring capacity (FUCINA_SLIDING_RING, shared with the dense engine; default
    // 8192). Floored at sliding_window + a prefill-chunk margin so chunked prefill
    // (chunk = cap - window) always makes forward progress and the live window never
    // wraps onto itself. cap = min(fit_ctx, RING) — when fit_ctx ≤ RING the ring is
    // inactive (flat cache, byte-identical to pre-ring behavior, no chunking).
    int RING = 8192;
    if (const char* e=getenv("FUCINA_SLIDING_RING")){ int v=atoi(e); if (v>0) RING=v; }
    const int ring_floor = c.sliding_window + 256;
    if (RING < ring_floor) RING = ring_floor;

    int want_ctx  = (context_size > 262144u) ? 262144 : (int)context_size;  // model max
    if (want_ctx < 1) want_ctx = 1;
    int want_seqs = max_seqs>0 ? max_seqs : 8;
    if (want_seqs > 32) want_seqs = 32;   // loader Q8_1 batch-scratch is sized S=32 (see above)
    if (want_seqs < 1) want_seqs = 1;

    size_t free_b=0, total_b=0;
    cudaMemGetInfo(&free_b, &total_b);
    uint64_t kv_avail = free_b;
    // FUCINA_MEM_BUDGET_GB caps the WHOLE engine (weights + KV), not just headroom —
    // lets a client reserve a unified-memory box for other work (e.g. "41 GB" of 128).
    if (const char* mb = getenv("FUCINA_MEM_BUDGET_GB")) {
        double gb = atof(mb);
        if (gb > 0) {
            uint64_t budget    = (uint64_t)(gb * 1e9);   // decimal GB (matches all logs)
            uint64_t kv_budget = budget > eng->dev_bytes ? budget - eng->dev_bytes : 0;
            if (kv_budget < kv_avail) kv_avail = kv_budget;
        }
    }
    const uint64_t RESERVE = 1ull<<30;   // decode/prefill scratch + allocator fragmentation
    kv_avail = kv_avail > RESERVE ? kv_avail - RESERVE : 0;

    // Per-slot KV bytes at a given ctx, accounting for the sliding ring.
    auto per_slot=[&](uint64_t ctx)->uint64_t{
        uint64_t sc = ctx < (uint64_t)RING ? ctx : (uint64_t)RING;
        return full_per_tok*ctx + slid_per_tok*sc;
    };
    const int MIN_CTX = 1024;            // don't shrink an honored slot below this
    int fit_seqs = want_seqs, fit_ctx = want_ctx;
    for (;;) {
        const uint64_t per_seq = kv_avail / (uint64_t)fit_seqs;
        uint64_t fc;
        if (per_slot((uint64_t)want_ctx) <= per_seq) {
            fc = (uint64_t)want_ctx;                         // request fits
        } else {
            const uint64_t at_ring = per_slot((uint64_t)RING);   // (full+slid)*RING
            if (per_seq >= at_ring && full_per_tok>0)
                fc = (uint64_t)RING + (per_seq - at_ring)/full_per_tok;   // ctx > RING regime
            else {
                const uint64_t slope = full_per_tok + slid_per_tok;       // ctx ≤ RING regime
                fc = slope ? per_seq/slope : (uint64_t)want_ctx;
            }
        }
        if (fc > (uint64_t)want_ctx) fc = (uint64_t)want_ctx;
        fit_ctx = (int)fc;
        // Honor concurrency first; only trade a slot once ctx would drop below the floor.
        if (fit_ctx >= want_ctx || fit_ctx >= MIN_CTX || fit_seqs == 1) break;
        fit_seqs--;
    }
    if (fit_ctx < 1) {
        fprintf(stderr,"e4b: FATAL — not enough device memory for even one KV sequence "
                "(need %.2f GB/slot @%d ctx; %.2f GB available after %.2f GB weights). "
                "Lower --ctx, raise FUCINA_MEM_BUDGET_GB, or free device memory.\n",
                (double)per_slot((uint64_t)MIN_CTX)/1e9, MIN_CTX,
                (double)kv_avail/1e9, (double)eng->dev_bytes/1e9);
        e4b_engine_destroy(eng); return nullptr;
    }
    eng->max_ctx     = fit_ctx;
    eng->sliding_cap = (fit_ctx < RING) ? fit_ctx : RING;   // ring inactive when ctx ≤ RING
    eng->max_seqs    = fit_seqs;
    eng->ctx         = (uint32_t)fit_ctx;   // report the value actually provisioned
    if (fit_ctx < want_ctx || fit_seqs < want_seqs)
        fprintf(stderr,"e4b: KV auto-fit to memory — ctx %d→%d, seqs %d→%d "
                "(KV %.2f GB = %.2f GB/slot × %d; %.2f GB avail, %.2f GB weights)\n",
                want_ctx, fit_ctx, want_seqs, fit_seqs,
                (double)per_slot((uint64_t)fit_ctx)*fit_seqs/1e9, (double)per_slot((uint64_t)fit_ctx)/1e9,
                fit_seqs, (double)kv_avail/1e9, (double)eng->dev_bytes/1e9);
    if (eng->sliding_cap < fit_ctx)
        fprintf(stderr,"e4b: sliding-window ring ON — cap %d (window %d); sliding KV is ctx-"
                "independent, prefill chunked at %d\n",
                eng->sliding_cap, c.sliding_window, eng->sliding_cap - c.sliding_window);

    eng->slots.resize(eng->max_seqs);
    if (!e4b_slot_alloc(eng, eng->slots[0])) {  // slot 0 backs the single-seq API
        fprintf(stderr,"e4b: slot 0 KV alloc failed\n"); e4b_engine_destroy(eng); return nullptr;
    }
    eng->slots[0].active=true;

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
      F(d.logits_f); F(d.invf_s); F(d.invf_f); F(d.npast); F(d.fa_m); F(d.fa_l); F(d.fa_acc);
      if(d.gexec) cudaGraphExecDestroy(d.gexec); if(d.cstream) cudaStreamDestroy(d.cstream); }
    { auto& s=eng->vs; F(s.ids); F(s.hidden); F(s.norm); F(s.tmpH); F(s.q); F(s.k); F(s.v); F(s.attn);
      F(s.gate); F(s.up); F(s.act); F(s.pleg); F(s.ctx_bf); F(s.ple_lookup); F(s.ple_ctx); F(s.pli);
      F(s.lg); F(s.lg_bf); F(s.am); F(s.logits_last); F(s.logits_last_bf); F(s.invf_s); F(s.invf_f);
      F(s.npast); F(s.fin);
      if(s.gexec) cudaGraphExecDestroy(s.gexec); if(s.cstream) cudaStreamDestroy(s.cstream); }
    for (Slot& s : eng->slots) e4b_slot_free(eng, s);
    { E4bMtp& m=eng->mtp; auto FB=[](__nv_bfloat16* p){ if(p) cudaFree(p); }; auto FF2=[](float* p){ if(p) cudaFree(p); }; auto FI=[](int* p){ if(p) cudaFree(p); }; auto F3=[](int32_t* p){ if(p) cudaFree(p); };
      FB(m.pre_proj); FB(m.post_proj); FB(m.unembed); FF2(m.out_norm); FB(m.out_norm_b); FF2(m.rope_freqs);
      FF2(m.invf_sliding); FF2(m.invf_global);
      auto FU=[](uint8_t* p){ if(p) cudaFree(p); };
      FU(m.q40_pre); FU(m.q40_post); FU(m.q40_unembed);
      for (E4bMtpLayer& L:m.layers){ FB(L.wq);FB(L.wo);FB(L.gate);FB(L.up);FB(L.down);
        FU(L.q40_wq);FU(L.q40_wo);FU(L.q40_gate);FU(L.q40_up);FU(L.q40_down);
        FF2(L.attn_norm);FF2(L.q_norm);FF2(L.post_attn_norm);FF2(L.ffn_norm);FF2(L.post_ffw_norm);
        FB(L.attn_norm_b);FB(L.q_norm_b);FB(L.post_attn_norm_b);FB(L.ffn_norm_b);FB(L.post_ffw_norm_b); }
      // drafter-forward scratch
      FB(m.d_xh);FB(m.d_cur);FB(m.d_t1);FB(m.d_t2);FB(m.d_q);FB(m.d_attn);FB(m.d_ffa);FB(m.d_ffb);
      FB(m.d_logits_bf); FF2(m.d_logits); FI(m.d_pos); FI(m.d_argmax); F3(m.d_tok); FF2(m.d_draft_h);
      auto FI8=[](int8_t* p){ if(p) cudaFree(p); };
      FI8(m.dq_qa); FF2(m.dq_da); F3(m.dq_sa); FF2(m.dq_yf);
      if(m.mgraph) cudaGraphExecDestroy(m.mgraph); if(m.mstream) cudaStreamDestroy(m.mstream); }
    if (eng->cublas) cublasDestroy(eng->cublas);
    delete eng;
}

// Forward decls of kernels/helpers defined further down (used by the MTP loader/forward).
// Must share the unnamed namespace of their definitions to avoid creating new symbols.
namespace {
__global__ void f32_to_bf16_kernel(const float*, __nv_bfloat16*, int);
float* make_inv_freq_sliding(int hd, float base);
float* make_inv_freq_proportional(int hd, float base, float partial);
}

// ── MTP draft-head loader (increment 1: load + residency; forward/spec follow) ──
// Loads the gemma4-assistant GGUF: dequant Q4_0 weights → BF16, keep F32 norms/scales/rope.
// All dims are DERIVED from tensor element-counts + the target config (no GGUF-KV parse): AH
// from pre_projection / (2·H_out), FF from ffn_gate / AH, per-layer head_dim from attn_q_norm
// width (→ is_global when == global_head_dim), qdim from attn_q / AH. See docs/e4b-mtp-plan.md.
extern "C" int e4b_engine_load_assistant(e4b_engine_t *eng, const char *path){
    if(!eng || !path) return -1;
    cudaSetDevice(eng->device_id);
    e4bgguf::GgufFile g; std::string err;
    if(!g.open(path, err)){ fprintf(stderr,"e4b-mtp: open %s: %s\n", path, err.c_str()); return -1; }
    const e4b::Config& c = eng->cfg;
    E4bMtp& m = eng->mtp;
    m.H_out=c.hidden_size; m.vocab=c.vocab_size; m.rms_eps=c.rms_eps;
    m.rope_theta_sliding=c.rope_theta_sliding; m.rope_theta_global=c.rope_theta_full;
    m.sliding_window=c.sliding_window; m.kv_heads=c.n_kv_heads;
    auto nel=[&](const std::string& nm)->int64_t{ uint64_t o,n; uint32_t t; return g.find(nm.c_str(),&o,&n,&t)?(int64_t)n:-1; };

    const int64_t pre_n = nel("nextn.pre_projection.weight");
    if (pre_n <= 0){ fprintf(stderr,"e4b-mtp: %s is not a gemma4-assistant head (no nextn.pre_projection)\n", path); g.close(); return -1; }
    m.AH = (int)(pre_n / (2*(int64_t)m.H_out));               // pre_proj [2·H_out → AH]
    m.n_layers = 0; while (nel("blk."+std::to_string(m.n_layers)+".attn_q.weight") > 0) m.n_layers++;
    const int64_t ff_n = nel("blk.0.ffn_gate.weight");
    m.FF = (ff_n>0 && m.AH>0) ? (int)(ff_n / (int64_t)m.AH) : 0;
    if (m.AH<=0 || m.n_layers<=0 || m.FF<=0){ fprintf(stderr,"e4b-mtp: bad dims (AH %d FF %d layers %d)\n", m.AH,m.FF,m.n_layers); g.close(); return -1; }

    uint64_t tot=0; bool ok=true;
    auto B=[&](const std::string& nm){ return up_bf16_gguf(g, nm, 0, &tot); };
    auto FL=[&](const std::string& nm){ return up_f32_gguf(g, nm, 0, &tot); };
    m.pre_proj  = B("nextn.pre_projection.weight");  ok&=!!m.pre_proj;
    m.post_proj = B("nextn.post_projection.weight"); ok&=!!m.post_proj;
    m.unembed   = B("token_embd.weight");            ok&=!!m.unembed;
    m.out_norm  = FL("output_norm.weight");          ok&=!!m.out_norm;
    m.rope_freqs= FL("rope_freqs.weight");           // optional (rope_kernel builds its own)
    m.layers.assign(m.n_layers, E4bMtpLayer{});
    for (int l=0;l<m.n_layers && ok;++l){
        E4bMtpLayer& L=m.layers[l];
        const std::string p="blk."+std::to_string(l)+".";
        L.head_dim = (int)nel(p+"attn_q_norm.weight");        // 256 sliding / 512 global
        L.is_global = (L.head_dim == c.global_head_dim);
        const int64_t qel = nel(p+"attn_q.weight");
        L.qdim = (L.head_dim>0 && qel>0) ? (int)(qel / (int64_t)m.AH) : 0;
        L.wq=B(p+"attn_q.weight"); L.wo=B(p+"attn_output.weight");
        L.gate=B(p+"ffn_gate.weight"); L.up=B(p+"ffn_up.weight"); L.down=B(p+"ffn_down.weight");
        L.attn_norm=FL(p+"attn_norm.weight"); L.q_norm=FL(p+"attn_q_norm.weight");
        L.post_attn_norm=FL(p+"post_attention_norm.weight");
        L.ffn_norm=FL(p+"ffn_norm.weight"); L.post_ffw_norm=FL(p+"post_ffw_norm.weight");
        L.out_scale=1.0f; read_scalar_gguf(g, p+"layer_output_scale.weight", &L.out_scale);
        ok &= L.wq&&L.wo&&L.gate&&L.up&&L.down&&L.attn_norm&&L.q_norm&&L.post_attn_norm
              &&L.ffn_norm&&L.post_ffw_norm&&L.head_dim>0&&L.qdim>0;
    }
    m.q_heads = (!m.layers.empty()&&m.layers[0].head_dim>0) ? m.layers[0].qdim/m.layers[0].head_dim : 0;

    // ── Stage 2: native Q4_0 block streams for the drafter dp4a forward ──
    // The assistant projections are Q4_0; copy them verbatim and decode through the SHARED
    // dp4a MMVQ (mmvq.cuh) instead of cuBLAS BF16 GEMVs (kills per-call cuBLAS overhead on the
    // tiny drafter dims, and is graph-capturable). Default ON for Q4_0 assistants; FUCINA_E4B
    // _MTP_BF16=1 keeps the BF16 cuBLAS path. The BF16 projections above are freed when q40 ON.
    bool q40_assist = ok;
    if (q40_assist) if (const char* e=getenv("FUCINA_E4B_MTP_BF16"); e && e[0]=='1') q40_assist=false;
    if (q40_assist) {
        const int AH=m.AH, FF=m.FF, V=m.vocab, H=m.H_out;
        auto cp=[&](const std::string& nm, int64_t el, uint8_t** d){
            return copy_blocks_gguf(g, nm, e4bgguf::GGML_TYPE_Q4_0, 18, 32, el, d, &tot); };
        q40_assist &= cp("nextn.pre_projection.weight",  (int64_t)AH*(2*H), &m.q40_pre);
        q40_assist &= cp("nextn.post_projection.weight", (int64_t)H*AH,     &m.q40_post);
        q40_assist &= cp("token_embd.weight",            (int64_t)V*AH,     &m.q40_unembed);
        for (int l=0;l<m.n_layers && q40_assist;++l){
            E4bMtpLayer& L=m.layers[l];
            const std::string p="blk."+std::to_string(l)+".";
            q40_assist &= cp(p+"attn_q.weight",      (int64_t)L.qdim*AH, &L.q40_wq);
            q40_assist &= cp(p+"attn_output.weight", (int64_t)AH*L.qdim, &L.q40_wo);
            q40_assist &= cp(p+"ffn_gate.weight",    (int64_t)FF*AH,     &L.q40_gate);
            q40_assist &= cp(p+"ffn_up.weight",      (int64_t)FF*AH,     &L.q40_up);
            q40_assist &= cp(p+"ffn_down.weight",    (int64_t)AH*FF,     &L.q40_down);
        }
    }
    g.close();
    if (!ok){ fprintf(stderr,"e4b-mtp: assistant load FAILED\n"); return -1; }
    if (q40_assist) {
        m.use_q40 = 1;
        // BF16 projections no longer needed — the dp4a forward reads the native Q4_0 nibbles.
        // Free them (the ~half-memory win), mirroring the target use_q40 setup.
        auto FB=[&](__nv_bfloat16*& p){ if(p){ cudaFree(p); p=nullptr; } };
        FB(m.pre_proj); FB(m.post_proj); FB(m.unembed);
        for (auto& L : m.layers){ FB(L.wq); FB(L.wo); FB(L.gate); FB(L.up); FB(L.down); }
    } else {
        // Free any partial Q4_0 copies; keep the BF16 cuBLAS path.
        auto FU=[&](uint8_t*& p){ if(p){ cudaFree(p); p=nullptr; } };
        FU(m.q40_pre); FU(m.q40_post); FU(m.q40_unembed);
        for (auto& L : m.layers){ FU(L.q40_wq); FU(L.q40_wo); FU(L.q40_gate); FU(L.q40_up); FU(L.q40_down); }
    }

    // Convert the F32 norm weights → BF16 (rmsnorm_kernel/head_rmsnorm_kernel take BF16
    // weights). One copy per norm, counted into dev_bytes; the F32 originals are kept.
    auto NB=[&](float* src, int n)->__nv_bfloat16*{
        if(!src||n<=0) return nullptr; __nv_bfloat16* d=nullptr;
        if(cudaMalloc(&d,(size_t)n*sizeof(__nv_bfloat16))!=cudaSuccess) return nullptr;
        f32_to_bf16_kernel<<<(n+255)/256,256>>>(src,d,n); tot+=(size_t)n*sizeof(__nv_bfloat16); return d;
    };
    m.out_norm_b = NB(m.out_norm, m.AH); ok&=!!m.out_norm_b;
    for (int l=0;l<m.n_layers && ok;++l){
        E4bMtpLayer& L=m.layers[l];
        L.attn_norm_b      = NB(L.attn_norm,      m.AH);
        L.q_norm_b         = NB(L.q_norm,         L.head_dim);
        L.post_attn_norm_b = NB(L.post_attn_norm, m.AH);
        L.ffn_norm_b       = NB(L.ffn_norm,       m.AH);
        L.post_ffw_norm_b  = NB(L.post_ffw_norm,  m.AH);
        ok &= L.attn_norm_b&&L.q_norm_b&&L.post_attn_norm_b&&L.ffn_norm_b&&L.post_ffw_norm_b;
    }
    cudaDeviceSynchronize();
    if (!ok){ fprintf(stderr,"e4b-mtp: assistant norm BF16 conversion FAILED\n"); return -1; }

    // RoPE inv_freq tables for the drafter (Q-only): sliding layers use head_dim/θ_sliding,
    // the global layer uses global_head_dim/θ_global (matching the target's rope construction).
    m.global_head_dim   = c.global_head_dim;
    m.invf_sliding = make_inv_freq_sliding(c.head_dim, c.rope_theta_sliding);
    m.invf_global  = make_inv_freq_proportional(c.global_head_dim, c.rope_theta_full, c.rope_partial_full);

    m.dev_bytes=tot; m.loaded=true; eng->dev_bytes += tot;
    int n_glob=0; for(auto&L:m.layers) n_glob+=L.is_global;
    fprintf(stderr,"e4b-mtp: assistant loaded — %d layers (%d sliding + %d global), AH=%d FF=%d "
            "q_heads=%d kv_heads=%d, %.2f MB resident; drafts vs target slot-0 KV "
            "(prov_sliding=%d, prov_full=%d) [forward=%s]\n",
            m.n_layers, m.n_layers-n_glob, n_glob, m.AH, m.FF, m.q_heads, m.kv_heads,
            tot/1e6, eng->prov_sliding, eng->prov_full, m.use_q40?"dp4a-Q4_0":"cuBLAS-BF16");
    return 0;
}
extern "C" int e4b_engine_has_assistant(const e4b_engine_t *eng){ return (eng && eng->mtp.loaded)?1:0; }

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
extern "C" uint32_t e4b_engine_max_ctx(const e4b_engine_t *eng)  { return eng ? (uint32_t)eng->max_ctx : 0; }
extern "C" int      e4b_engine_max_seqs(const e4b_engine_t *eng) { return eng ? eng->max_seqs : 0; }

// Cumulative speculative-decode counters (for the server /metrics). τ = emitted/steps.
extern "C" long e4b_engine_spec_steps(const e4b_engine_t *eng)    { return eng ? eng->spec_steps : 0; }
extern "C" long e4b_engine_spec_drafted(const e4b_engine_t *eng)  { return eng ? eng->spec_drafted : 0; }
extern "C" long e4b_engine_spec_accepted(const e4b_engine_t *eng) { return eng ? eng->spec_accepted : 0; }
extern "C" long e4b_engine_spec_emitted(const e4b_engine_t *eng)  { return eng ? eng->spec_emitted : 0; }

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

// Block sum-of-squares reduction via warp shuffles + one cross-warp shared pass.
// Returns the full block sum on every thread. Works for blockDim up to 1024 (32 warps).
// llama.cpp uses the same warp-shuffle norm; this replaces the 256-thread shared-mem tree
// so the wide 2560-elem rows can be normed by a full 1024-thread block (fewer elems/thread,
// fewer sync barriers). The summation order differs from the old tree, but the verified gate
// (greedy token-id equivalence) confirms argmax is unchanged.
__device__ __forceinline__ float block_sumsq(float ss){
    #pragma unroll
    for (int o=16;o>0;o>>=1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane==0) warp_sums[wid] = ss;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    float total = 0.f;
    if (threadIdx.x < nwarps) total = warp_sums[threadIdx.x];
    #pragma unroll
    for (int o=16;o>0;o>>=1) total += __shfl_down_sync(0xffffffffu, total, o);
    __shared__ float bsum;
    if (threadIdx.x==0) bsum = total;
    __syncthreads();
    return bsum;
}

// RMSNorm over the last dim: y = x*rsqrt(mean(x^2)+eps) * w  (w optional).
// Gemma4: NO (1+w). One block per row. 1024-thread warp-shuffle reduction.
__global__ void rmsnorm_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                               bf16* __restrict__ y, int rows, int dim, float eps){
    int r = blockIdx.x; if (r >= rows) return;
    const bf16* xr = x + (size_t)r*dim; bf16* yr = y + (size_t)r*dim;
    float ss = 0.f;
    for (int i=threadIdx.x;i<dim;i+=blockDim.x){ float v=b2f(xr[i]); ss+=v*v; }
    float inv = rsqrtf(block_sumsq(ss)/dim + eps);
    for (int i=threadIdx.x;i<dim;i+=blockDim.x){
        float v = b2f(xr[i])*inv;
        if (w) v *= b2f(w[i]);
        yr[i]=f2b(v);
    }
}

// Fused RMSNorm + residual-add (+ optional scalar) — one launch, one global round-trip.
// Computes, per element:  out = (resid + f2b(norm(x)*w)) [* layer_scalar].
// This reproduces the EXACT bit pattern of the unfused decode epilogue
//   rmsnorm_kernel(x,w,nrm); add_kernel(resid,nrm); [scale_kernel(resid,s);] hidden=resid
// because the intermediate norm value is rounded to bf16 (f2b) then re-read (b2f) for the
// add, and the post-add value is rounded to bf16 then re-read for the scale — the same
// double round-trips the separate kernels incur. resid is read in place and overwritten as
// the residual buffer (d_tmpH) so the caller can drop the trailing D2D memcpy into d_hidden
// (out == d_hidden). The sum-of-squares reduction is byte-identical to rmsnorm_kernel (same
// 256-thread strided accumulate + shared-mem tree), so token ids are unchanged.
__global__ void rmsnorm_resid_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                     const bf16* __restrict__ resid, bf16* __restrict__ out,
                                     int rows, int dim, float eps, float scale, int has_scale){
    int r = blockIdx.x; if (r >= rows) return;
    const bf16* xr = x + (size_t)r*dim;
    const bf16* rr = resid + (size_t)r*dim;
    bf16* outr = out + (size_t)r*dim;
    float ss = 0.f;
    for (int i=threadIdx.x;i<dim;i+=blockDim.x){ float v=b2f(xr[i]); ss+=v*v; }
    float inv = rsqrtf(block_sumsq(ss)/dim + eps);
    for (int i=threadIdx.x;i<dim;i+=blockDim.x){
        float v = b2f(xr[i])*inv;
        if (w) v *= b2f(w[i]);
        // f2b/b2f round-trip on the norm value matches the standalone rmsnorm + add path.
        float acc = b2f(rr[i]) + b2f(f2b(v));
        if (has_scale){ acc = b2f(f2b(acc)) * scale; }
        outr[i] = f2b(acc);
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
                                  float scaling, int window, int cap){
    int h = blockIdx.x, t = blockIdx.y;
    if (h>=n_heads || t>=T) return;
    int P = n_past + t;                       // absolute query position
    int group = n_heads / n_kv, kvh = h/group;
    const bf16* qv = q + ((size_t)t*n_heads + h)*hd;
    extern __shared__ float scores[];         // [len], indexed relative to lo
    int lo = (window>0) ? (P-window+1) : 0; if (lo<0) lo=0;
    int len = P - lo + 1;                      // <= window for sliding layers
    for (int j=threadIdx.x; j<len; j+=blockDim.x){
        const bf16* kv = kcache + ((size_t)((lo+j)%cap)*n_kv + kvh)*hd;
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
        for (int j=0;j<len;j++){ const bf16* vv=vcache+((size_t)((lo+j)%cap)*n_kv+kvh)*hd; acc += scores[j]*b2f(vv[d]); }
        ov[d]=f2b(acc*inv);
    }
}

// Copy new-token K (or V) [T, n_kv*hd] into the cache. Absolute position (n_past+t)
// is stored at ring slot (n_past+t) % cap; cap == cache rows (== max_ctx for full
// layers ⇒ no wrap, == sliding_cap for the sliding ring).
__global__ void kv_store_kernel(const bf16* __restrict__ src, bf16* __restrict__ cache,
                                int T, int row, int n_past, int cap){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < T*row){
        int t = i/row, e = i%row;
        cache[(size_t)((n_past + t) % cap)*row + e] = src[i];
    }
}

// ── CUDA-graph-friendly decode (T==1) kernels: DEVICE-resident position, FIXED smem ──
// kv_store reading the position from device memory (one new token at *d_npast).
__global__ void kv_store_dev_kernel(const bf16* __restrict__ src, bf16* __restrict__ cache,
                                    int row, const int* __restrict__ d_npast, int cap){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < row) cache[(size_t)((*d_npast) % cap)*row + i] = src[i];
}

// ── Split-K decode flash attention (T==1) ──────────────────────────────────────
// The old one-block-per-head kernel launched only n_heads (8) blocks and did the
// online-softmax + score reduction serially on thread 0 — badly under-occupied and
// 2.26x slower than llama's flash_attn_ext_vec. This pair splits the KV range across
// E4B_FA_SPLITS blocks per head (grid = n_heads*SPLITS, raising block count to 64–128),
// parallelizes the q·k dot across the warp, and the score reduction across the block;
// a tiny combine kernel merges the per-split (m,l,acc) partials. Grid + smem are FIXED
// (position read from d_npast), so the whole thing stays CUDA-graph-capturable.
#ifndef E4B_FA_SPLITS
#define E4B_FA_SPLITS 32   // tuned on GB10: 32 > 16 > 8 at all ctx; 64 regresses (combine cost)
#endif
#define E4B_FA_HDMAX 512   // global_head_dim upper bound (fa_acc stride)

// warp-shuffle reduction
__device__ __forceinline__ float warp_sum(float v){
    for(int o=16;o>0;o>>=1) v+=__shfl_xor_sync(0xffffffff,v,o);
    return v;
}

// One block per (head, split). Block = NW warps × 32. Each warp runs an independent
// online-softmax over its OWN strided subset of this split's keys into a private acc[NW][hd]
// region (no inter-warp races); lanes within a warp split the hd dot (warp-reduced) and own
// strided V dims. After the warp loop the NW per-warp (m,l,acc) are merged into one block
// partial and written to part_m/part_l/part_acc[(head*SPLITS+split)*hd]. The combine kernel
// then merges the SPLITS block-partials per head into the final output.
template<int NW>
__global__ void attn_flash_decode_split_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ kcache, const bf16* __restrict__ vcache,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    const int* __restrict__ d_npast,
    int n_heads, int n_kv, int hd, float scaling, int window, int cap, int n_splits)
{
    const int h  = blockIdx.x;
    const int sp = blockIdx.y;
    const int P  = *d_npast;
    const int group = n_heads / n_kv, kvh = h / group;
    int lo = (window > 0) ? (P - window + 1) : 0; if (lo < 0) lo = 0;
    const int total = P - lo + 1;                 // number of keys this head attends to
    const int per   = (total + n_splits - 1) / n_splits;   // keys per split (contiguous)
    const int s_beg = lo + sp * per;
    const int s_end = min(lo + (sp + 1) * per, P + 1);     // exclusive

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;            // 0..NW-1
    const bf16* qv = q + (size_t)h * hd;

    extern __shared__ float smem[];
    float* wacc = smem;                           // [NW*hd] per-warp un-normalized acc
    __shared__ float w_m[NW], w_l[NW];            // per-warp running max / denom
    float* myacc = wacc + (size_t)warp * hd;

    for (int d = lane; d < hd; d += 32) myacc[d] = 0.f;
    float m_run = -1e30f, l_run = 0.f;

    // each warp strides over its split's keys (stride NW); lanes split the hd dot.
    for (int kpos = s_beg + warp; kpos < s_end; kpos += NW) {
        const bf16* kv = kcache + ((size_t)(kpos % cap) * n_kv + kvh) * hd;
        float dot = 0.f;
        for (int d = lane; d < hd; d += 32) dot += b2f(qv[d]) * b2f(kv[d]);
        dot = warp_sum(dot) * scaling;            // full q·k, broadcast to all lanes
        float new_m = fmaxf(m_run, dot);
        float corr  = __expf(m_run - new_m);
        float e     = __expf(dot - new_m);
        l_run = l_run * corr + e;
        const bf16* vv = vcache + ((size_t)(kpos % cap) * n_kv + kvh) * hd;
        for (int d = lane; d < hd; d += 32) myacc[d] = myacc[d] * corr + e * b2f(vv[d]);
        m_run = new_m;
    }
    if (lane == 0) { w_m[warp] = m_run; w_l[warp] = l_run; }
    __syncthreads();

    // merge the NW per-warp partials into one block partial (warp 0 drives the math; all
    // warps cooperatively rescale their acc into the shared block max).
    __shared__ float blk_m, blk_l;
    if (threadIdx.x == 0) {
        float bm = -1e30f;
        for (int w = 0; w < NW; w++) bm = fmaxf(bm, w_m[w]);
        float bl = 0.f;
        for (int w = 0; w < NW; w++) bl += w_l[w] * __expf(w_m[w] - bm);
        blk_m = bm; blk_l = bl;
    }
    __syncthreads();
    // rescale each warp's acc to the block max, then sum across warps into part_acc.
    const size_t base = ((size_t)(h * n_splits + sp)) * E4B_FA_HDMAX;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float s = 0.f;
        for (int w = 0; w < NW; w++) s += wacc[(size_t)w * hd + d] * __expf(w_m[w] - blk_m);
        part_acc[base + d] = s;
    }
    if (threadIdx.x == 0) {
        const int pi = h * n_splits + sp;
        // empty split (no keys) → neutral partial that the combine ignores.
        if (s_beg >= s_end) { part_m[pi] = -1e30f; part_l[pi] = 0.f; }
        else                { part_m[pi] = blk_m;  part_l[pi] = blk_l; }
    }
}

// Merge the SPLITS per-head block-partials (m,l,acc) into the final attention output.
// One block per head; threads own strided hd dims for the acc combine.
__global__ void attn_flash_combine_kernel(
    const float* __restrict__ part_m, const float* __restrict__ part_l,
    const float* __restrict__ part_acc, bf16* __restrict__ out,
    int n_heads, int hd, int n_splits)
{
    const int h = blockIdx.x; if (h >= n_heads) return;
    __shared__ float g_m, g_l;
    if (threadIdx.x == 0) {
        float gm = -1e30f;
        for (int s = 0; s < n_splits; s++) gm = fmaxf(gm, part_m[h * n_splits + s]);
        float gl = 0.f;
        for (int s = 0; s < n_splits; s++) {
            float m = part_m[h * n_splits + s];
            if (m > -1e29f) gl += part_l[h * n_splits + s] * __expf(m - gm);
        }
        g_m = gm; g_l = gl;
    }
    __syncthreads();
    const float inv = 1.f / g_l;
    bf16* ov = out + (size_t)h * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float a = 0.f;
        for (int s = 0; s < n_splits; s++) {
            float m = part_m[h * n_splits + s];
            if (m > -1e29f) {
                float w = __expf(m - g_m);
                a += part_acc[((size_t)(h * n_splits + s)) * E4B_FA_HDMAX + d] * w;
            }
        }
        ov[d] = f2b(a * inv);
    }
}

// Flash (online-softmax, tiled) decode attention for T==1. Unlike attn_cache_kernel — whose
// scores[] smem grows with n_past and which takes n_past as a host arg — this uses FIXED smem
// (TILE scores + hd accumulator) and reads the position from device memory, so the whole decode
// forward can be captured once into a CUDA graph and replayed each token. One block per head.
template<int TILE>
__global__ void attn_flash_decode_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ kcache, const bf16* __restrict__ vcache,
    bf16* __restrict__ out, const int* __restrict__ d_npast,
    int n_heads, int n_kv, int hd, float scaling, int window, int cap)
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
            const bf16* kv = kcache + ((size_t)((base + j) % cap) * n_kv + kvh) * hd;
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
                const bf16* vv = vcache + ((size_t)((base + j) % cap) * n_kv + kvh) * hd;
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

// ── CUDA-graph-friendly SPEC-VERIFY (T>1, FIXED T) kernels ──
// The spec-verify forward runs at a fixed T=K+1 rows every round; like the T==1 decode it
// can be captured once and replayed if (a) the absolute base position lives in DEVICE memory
// (so only it changes per replay) and (b) smem is FIXED (independent of n_past). These three
// kernels mirror the decode-graph kernels (rope_batch/kv_store_dev/attn_flash_decode) but over
// T rows whose absolute position is (*d_npast + t). They are bit-identical in math to the
// host-arg prefill kernels (rope_kernel/kv_store_kernel/attn_cache_kernel) the verify used
// before — same dot/softmax, same ring index ((pos)%cap) — just with the position read from
// device and FIXED tiled smem (online softmax) so the graph replay is valid at any n_past.

// RoPE over T rows at absolute positions (*d_npast + t). x:[T,n_heads,hd].
__global__ void rope_verify_kernel(bf16* __restrict__ x, const float* __restrict__ inv_freq,
                                   int T, int n_heads, int hd, const int* __restrict__ d_npast){
    int idx = blockIdx.x; if (idx >= T*n_heads) return;
    int t = idx / n_heads; bf16* v = x + (size_t)idx*hd; int half = hd/2;
    const int pos = *d_npast + t;
    for (int i=threadIdx.x;i<half;i+=blockDim.x){
        float ang=(float)pos*inv_freq[i]; float cc=cosf(ang), ss=sinf(ang);
        float a=b2f(v[i]), bb=b2f(v[i+half]);
        v[i]=f2b(a*cc-bb*ss); v[i+half]=f2b(bb*cc+a*ss);
    }
}
// Store T new-token K/V rows [T,row] into the cache at absolute positions (*d_npast + t),
// ring slot (pos%cap). Mirrors kv_store_kernel but reads the base from device memory.
__global__ void kv_store_verify_kernel(const bf16* __restrict__ src, bf16* __restrict__ cache,
                                       int T, int row, const int* __restrict__ d_npast, int cap){
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i >= T*row) return;
    int t = i/row, e = i%row;
    cache[(size_t)((*d_npast + t) % cap)*row + e] = src[i];
}
// Flash (online-softmax, tiled, FIXED smem) causal/sliding GQA over T new queries at absolute
// positions (*d_npast + t). One block per (head, t). Bit-identical softmax result to
// attn_cache_kernel (same scores, same normalization) but with device position + fixed smem so
// the verify forward can be CUDA-graph-captured. smem = (TILE + hd) floats, independent of n_past.
template<int TILE>
__global__ void attn_flash_verify_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ kcache, const bf16* __restrict__ vcache,
    bf16* __restrict__ out, const int* __restrict__ d_npast, int T,
    int n_heads, int n_kv, int hd, float scaling, int window, int cap){
    int h = blockIdx.x, t = blockIdx.y;
    if (h>=n_heads || t>=T) return;
    const int P = *d_npast + t;                   // this query's absolute position
    const int group = n_heads / n_kv, kvh = h/group;
    int lo = (window>0) ? (P-window+1) : 0; if (lo<0) lo=0;
    const bf16* qv = q + ((size_t)t*n_heads + h)*hd;

    extern __shared__ float smem[];
    float* sc  = smem;          // [TILE]
    float* acc = smem + TILE;   // [hd]
    __shared__ float m_run, l_run, corr;
    for (int d=threadIdx.x; d<hd; d+=blockDim.x) acc[d]=0.f;
    if (threadIdx.x==0){ m_run=-1e30f; l_run=0.f; }
    __syncthreads();

    for (int base=lo; base<=P; base+=TILE){
        const int tlen = min(TILE, P-base+1);
        for (int j=threadIdx.x; j<tlen; j+=blockDim.x){
            const bf16* kv = kcache + ((size_t)((base+j)%cap)*n_kv + kvh)*hd;
            float dot=0.f; for(int d=0;d<hd;d++) dot += b2f(qv[d])*b2f(kv[d]);
            sc[j]=dot*scaling;
        }
        __syncthreads();
        if (threadIdx.x==0){
            float tm=-1e30f; for(int j=0;j<tlen;j++) tm=fmaxf(tm,sc[j]);
            float new_m=fmaxf(m_run,tm); corr=expf(m_run-new_m);
            float tl=0.f; for(int j=0;j<tlen;j++){ float e=expf(sc[j]-new_m); sc[j]=e; tl+=e; }
            l_run=l_run*corr+tl; m_run=new_m;
        }
        __syncthreads();
        for (int d=threadIdx.x; d<hd; d+=blockDim.x){
            float a=acc[d]*corr;
            for (int j=0;j<tlen;j++){ const bf16* vv=vcache+((size_t)((base+j)%cap)*n_kv+kvh)*hd; a += sc[j]*b2f(vv[d]); }
            acc[d]=a;
        }
        __syncthreads();
    }
    const float inv=1.f/l_run;
    bf16* ov = out + ((size_t)t*n_heads + h)*hd;
    for (int d=threadIdx.x; d<hd; d+=blockDim.x) ov[d]=f2b(acc[d]*inv);
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
                                      const int* __restrict__ pos, int B, int row, int cap){
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i>=B*row) return;
    int b=i/row, e=i%row;
    caches[b][(size_t)(pos[b] % cap)*row + e] = src[i];
}
// Batched GQA attention: sequence b's single query (at pos[b]) attends ITS cache
// [0..pos[b]] (sliding window if window>0). One block per (head, sequence).
__global__ void attn_batch_kernel(const bf16* __restrict__ q, bf16* const* __restrict__ kcaches,
                                  bf16* const* __restrict__ vcaches, const int* __restrict__ pos,
                                  bf16* __restrict__ out, int B, int n_heads, int n_kv, int hd,
                                  float scaling, int window, int cap){
    int h = blockIdx.x, b = blockIdx.y; if (h>=n_heads || b>=B) return;
    int P = pos[b]; int group=n_heads/n_kv, kvh=h/group;
    const bf16* qv = q + ((size_t)b*n_heads + h)*hd;
    const bf16* kc = kcaches[b]; const bf16* vc = vcaches[b];
    extern __shared__ float scores[];
    int lo = (window>0)?(P-window+1):0; if(lo<0) lo=0; int len=P-lo+1;
    for (int j=threadIdx.x;j<len;j+=blockDim.x){
        const bf16* kv=kc+((size_t)((lo+j)%cap)*n_kv+kvh)*hd;
        float dot=0.f; for(int d=0;d<hd;d++) dot+=b2f(qv[d])*b2f(kv[d]);
        scores[j]=dot*scaling;
    }
    __syncthreads(); __shared__ float ssum;
    if (threadIdx.x==0){ float m=-1e30f; for(int j=0;j<len;j++) m=fmaxf(m,scores[j]);
        float sm=0.f; for(int j=0;j<len;j++){ float e=expf(scores[j]-m); scores[j]=e; sm+=e; } ssum=sm; }
    __syncthreads(); float inv=1.f/ssum;
    bf16* ov = out + ((size_t)b*n_heads + h)*hd;
    for (int d=threadIdx.x;d<hd;d+=blockDim.x){
        float acc=0.f; for(int j=0;j<len;j++){ const bf16* vv=vc+((size_t)((lo+j)%cap)*n_kv+kvh)*hd; acc+=scores[j]*b2f(vv[d]); }
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
// fp32→bf16 copy (used to convert the MTP F32 norm weights to BF16 once at load)
__global__ void f32_to_bf16_kernel(const float* __restrict__ x, bf16* __restrict__ y, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=f2b(x[i]);
}
// single-block argmax over a [n] fp32 vector → out[0] (drafter greedy pick).
// argmax over logits[n]. out[0] = id. If out_tok != nullptr also writes the id as int32
// (used by the graph-captured drafter so the next replay's embed reads the chained token
// straight off the GPU — no host round-trip between draft steps).
__global__ void argmax_kernel(const float* __restrict__ x, int n, int* __restrict__ out,
                              int32_t* __restrict__ out_tok=nullptr){
    __shared__ float sv[256]; __shared__ int si[256];
    float bv=-1e30f; int bi=0;
    for(int i=threadIdx.x;i<n;i+=blockDim.x){ if(x[i]>bv){ bv=x[i]; bi=i; } }
    sv[threadIdx.x]=bv; si[threadIdx.x]=bi; __syncthreads();
    for(int s=blockDim.x>>1;s>0;s>>=1){
        if(threadIdx.x<s && sv[threadIdx.x+s]>sv[threadIdx.x]){ sv[threadIdx.x]=sv[threadIdx.x+s]; si[threadIdx.x]=si[threadIdx.x+s]; }
        __syncthreads();
    }
    if(threadIdx.x==0){ out[0]=si[0]; if(out_tok) out_tok[0]=si[0]; }
}
// per-row argmax over logits[T,V] → out[T] (one block per row; used by spec-verify).
__global__ void argmax_rows_kernel(const float* __restrict__ x, int T, int V, int* __restrict__ out){
    int row=blockIdx.x; if(row>=T) return;
    const float* r = x + (size_t)row*V;
    __shared__ float sv[256]; __shared__ int si[256];
    float bv=-1e30f; int bi=0;
    for(int i=threadIdx.x;i<V;i+=blockDim.x){ if(r[i]>bv){ bv=r[i]; bi=i; } }
    sv[threadIdx.x]=bv; si[threadIdx.x]=bi; __syncthreads();
    for(int s=blockDim.x>>1;s>0;s>>=1){
        if(threadIdx.x<s && sv[threadIdx.x+s]>sv[threadIdx.x]){ sv[threadIdx.x]=sv[threadIdx.x+s]; si[threadIdx.x]=si[threadIdx.x+s]; }
        __syncthreads();
    }
    if(threadIdx.x==0) out[row]=si[0];
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
                    float* emb_out, float* l0_out, float* fin_out, float* logits_last_out,
                    int32_t* out_argmax_rows = nullptr){
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
    // The spec-verify forward (out_argmax_rows set, T = #candidates ≤ K+1) reuses a SEPARATE
    // persistent scratch sized for VROWS rows — no per-round cudaMalloc / RoPE rebuild. Same
    // math as the prefill malloc path (bit-identical), just stable buffers.
    const bool verify = (!persist && out_argmax_rows && T <= E4B_MTP_VROWS);
    if (verify && !eng->vs.ready) {
        auto& s = eng->vs; bool ok2=true; const int R=E4B_MTP_VROWS;
        ok2&=CKF(cudaMalloc(&s.ids,(size_t)R*sizeof(int32_t)));
        ok2&=CKF(cudaMalloc(&s.hidden,(size_t)R*H*2)); ok2&=CKF(cudaMalloc(&s.norm,(size_t)R*H*2)); ok2&=CKF(cudaMalloc(&s.tmpH,(size_t)R*H*2));
        ok2&=CKF(cudaMalloc(&s.q,(size_t)R*QMAX*2)); ok2&=CKF(cudaMalloc(&s.k,(size_t)R*KVMAX*2)); ok2&=CKF(cudaMalloc(&s.v,(size_t)R*KVMAX*2));
        ok2&=CKF(cudaMalloc(&s.attn,(size_t)R*QMAX*2)); ok2&=CKF(cudaMalloc(&s.gate,(size_t)R*FF*2)); ok2&=CKF(cudaMalloc(&s.up,(size_t)R*FF*2));
        ok2&=CKF(cudaMalloc(&s.act,(size_t)R*FF*2)); ok2&=CKF(cudaMalloc(&s.pleg,(size_t)R*PD*2)); ok2&=CKF(cudaMalloc(&s.ctx_bf,(size_t)R*W*2));
        ok2&=CKF(cudaMalloc(&s.ple_lookup,(size_t)R*W*4)); ok2&=CKF(cudaMalloc(&s.ple_ctx,(size_t)R*W*4)); ok2&=CKF(cudaMalloc(&s.pli,(size_t)R*W*4));
        ok2&=CKF(cudaMalloc(&s.lg,(size_t)R*V*4)); ok2&=CKF(cudaMalloc(&s.lg_bf,(size_t)R*V*2)); ok2&=CKF(cudaMalloc(&s.am,(size_t)R*sizeof(int)));
        ok2&=CKF(cudaMalloc(&s.logits_last,(size_t)V*4)); ok2&=CKF(cudaMalloc(&s.logits_last_bf,(size_t)V*2));
        ok2&=CKF(cudaMalloc(&s.npast,sizeof(int)));        // device base position (verify-graph)
        ok2&=CKF(cudaMalloc(&s.fin,(size_t)R*H*4));        // persistent post-final-norm rows
        s.invf_s = make_inv_freq_sliding(c.head_dim, c.rope_theta_sliding);
        s.invf_f = make_inv_freq_proportional(c.global_head_dim, c.rope_theta_full, c.rope_partial_full);
        if(!ok2 || !s.invf_s || !s.invf_f) return -1;
        // Capture stream for the verify CUDA graph (non-default stream required for capture).
        // FUCINA_E4B_VERIFY_NOGRAPH=1 forces the per-launch path.
        if (const char* e=getenv("FUCINA_E4B_VERIFY_NOGRAPH"); e && e[0]=='1') s.graph_disabled=true;
        if (!s.graph_disabled && cudaStreamCreateWithFlags(&s.cstream,cudaStreamNonBlocking)!=cudaSuccess) s.graph_disabled=true;
        s.ready = true;
    }
    if (persist && !eng->dec.ready) {
        auto& d = eng->dec; bool ok2=true;
        ok2&=CKF(cudaMalloc(&d.ids,sizeof(int32_t)));
        ok2&=CKF(cudaMalloc(&d.hidden,(size_t)H*2)); ok2&=CKF(cudaMalloc(&d.norm,(size_t)H*2)); ok2&=CKF(cudaMalloc(&d.tmpH,(size_t)H*2));
        ok2&=CKF(cudaMalloc(&d.q,(size_t)QMAX*2)); ok2&=CKF(cudaMalloc(&d.k,(size_t)KVMAX*2)); ok2&=CKF(cudaMalloc(&d.v,(size_t)KVMAX*2));
        ok2&=CKF(cudaMalloc(&d.attn,(size_t)QMAX*2)); ok2&=CKF(cudaMalloc(&d.gate,(size_t)FF*2)); ok2&=CKF(cudaMalloc(&d.up,(size_t)FF*2));
        ok2&=CKF(cudaMalloc(&d.act,(size_t)FF*2)); ok2&=CKF(cudaMalloc(&d.pleg,(size_t)PD*2)); ok2&=CKF(cudaMalloc(&d.ctx_bf,(size_t)W*2));
        ok2&=CKF(cudaMalloc(&d.ple_lookup,(size_t)W*4)); ok2&=CKF(cudaMalloc(&d.ple_ctx,(size_t)W*4)); ok2&=CKF(cudaMalloc(&d.pli,(size_t)W*4));
        ok2&=CKF(cudaMalloc(&d.logits_f,(size_t)V*4));
        // split-K flash-decode partials (graph-fixed grid n_heads*SPLITS)
        ok2&=CKF(cudaMalloc(&d.fa_m,(size_t)c.n_heads*E4B_FA_SPLITS*4));
        ok2&=CKF(cudaMalloc(&d.fa_l,(size_t)c.n_heads*E4B_FA_SPLITS*4));
        ok2&=CKF(cudaMalloc(&d.fa_acc,(size_t)c.n_heads*E4B_FA_SPLITS*E4B_FA_HDMAX*4));
        ok2&=CKF(cudaMalloc(&d.npast,sizeof(int)));
        d.invf_s = make_inv_freq_sliding(c.head_dim, c.rope_theta_sliding);
        d.invf_f = make_inv_freq_proportional(c.global_head_dim, c.rope_theta_full, c.rope_partial_full);
        if(!ok2 || !d.invf_s || !d.invf_f) return -1;
        // Capture stream for the CUDA-graph decode (a non-default stream is required for stream
        // capture). FUCINA_E4B_NOGRAPH=1 forces the per-kernel path.
        if (const char* e=getenv("FUCINA_E4B_NOGRAPH"); e && e[0]=='1') d.graph_disabled=true;
        if (!d.graph_disabled && cudaStreamCreateWithFlags(&d.cstream,cudaStreamNonBlocking)!=cudaSuccess) d.graph_disabled=true;
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
    } else if (verify) {
        auto& s = eng->vs;
        d_invf_s=s.invf_s; d_invf_f=s.invf_f; d_ids=s.ids;
        d_hidden=s.hidden; d_norm=s.norm; d_tmpH=s.tmpH; d_q=s.q; d_k=s.k; d_v=s.v; d_attn=s.attn;
        d_gate=s.gate; d_up=s.up; d_act=s.act; d_pleg=s.pleg;
        d_ple_lookup=s.ple_lookup; d_ple_ctx=s.ple_ctx; d_pli=s.pli;
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
    // The whole T==1 decode forward runs on a capture stream so it can be captured into a CUDA
    // graph and replayed; prefill (T>1) and the no-graph fallback use the default stream (st=0).
    // Decode (T==1) runs on dec.cstream; the spec-verify (fixed-T) forward runs on vs.cstream so
    // it too can be stream-captured into a CUDA graph. Prefill / no-graph use the default stream.
    cudaStream_t st = (persist && !eng->dec.graph_disabled) ? eng->dec.cstream
                    : (verify && !eng->vs.graph_disabled)  ? eng->vs.cstream
                    : (cudaStream_t)0;
    cublasSetStream(eng->cublas, st);
    // vgraph: route the verify forward through the device-position, fixed-smem attention kernels
    // (rope_verify/kv_store_verify/attn_flash_verify) so the whole T-row forward is capturable.
    // When the verify graph is disabled (env / capture failure) we keep the host-arg prefill
    // kernels — those bake n_past into the launch and so cannot be replayed.
    const bool vgraph = verify && !eng->vs.graph_disabled;
    // Inputs onto the stream (for the graph these are the only per-replay updates, done before
    // the launch — see the capture/replay block below).
    cudaMemcpyAsync(d_ids,tokens,T*sizeof(int32_t),cudaMemcpyHostToDevice,st);
    if (persist) cudaMemcpyAsync(eng->dec.npast,&n_past,sizeof(int),cudaMemcpyHostToDevice,st);
    if (verify)  cudaMemcpyAsync(eng->vs.npast,&n_past,sizeof(int),cudaMemcpyHostToDevice,st);

    auto GRID=[&](int n){ return (n+255)/256; };

    // The entire forward (embed → layers → final norm → head logits into dec.logits_f) is
    // wrapped so it can be either captured into a CUDA graph (single-token decode) or run
    // eagerly (prefill / capture fallback). Everything inside runs on `st`; the only per-step
    // inputs (token id, n_past) were H2D'd before this lambda, and the logits D2H is after it.
    auto forward = [&](){
    // ── embedding (×sqrt(H)) ──
    embed_kernel<<<T,256, 0, st>>>(eng->d_embed,d_ids,d_hidden,T,H,sqrtf((float)H));
    if(emb_out){ float* tmp; cudaMalloc(&tmp,(size_t)T*H*4); to_f32_kernel<<<GRID(T*H),256, 0, st>>>(d_hidden,tmp,T*H);
        cudaMemcpy(emb_out,tmp,(size_t)T*H*4,cudaMemcpyDeviceToHost); cudaFree(tmp); }

    // ── PLE precompute (token-identity lookup + context projection, combined) ──
    e4b_ple_lookup_launch(eng->d_ple_fp8, eng->d_ple_scale, d_ids, d_ple_lookup, T, W, st);
    {
        bf16* d_ctx_bf = persist ? eng->dec.ctx_bf : (verify ? eng->vs.ctx_bf : nullptr);
        if (!persist && !verify) cudaMalloc(&d_ctx_bf,(size_t)T*W*2);
        linear(eng->cublas, eng->d_plm_proj, d_hidden, d_ctx_bf, T, W, H);
        to_f32_kernel<<<GRID(T*W),256, 0, st>>>(d_ctx_bf,d_ple_ctx,T*W);
        if (!persist && !verify) cudaFree(d_ctx_bf);
        // The 1/sqrt(H) scale on the projection is a no-op before per-256 RMSNorm
        // (RMSNorm is scale-invariant), so we skip it and normalize directly.
        rmsnorm_f32_grouped<<<T*c.n_layers,256, 0, st>>>(d_ple_ctx, eng->d_ple_proj_norm, T*c.n_layers, PD, eps);
    }
    ple_combine_kernel<<<GRID(T*W),256, 0, st>>>(d_ple_lookup, d_ple_ctx, d_pli, T*W);

    // ── decoder layers ──
    for (int li=0; li<c.n_layers; ++li){
        const Layer& L = eng->layers[li];
        const bool full = (c.layer_types[li]==e4b::Attn::FULL);
        const int hd = full ? c.global_head_dim : c.head_dim;
        const int qd = c.n_heads*hd, kvd=c.n_kv_heads*hd;
        const bool shared = c.layer_shares_kv(li);
        const int window = full ? 0 : c.sliding_window;
        const int cap = full ? eng->max_ctx : eng->sliding_cap;  // ring capacity (full ⇒ no wrap)
        float* d_invf = full ? d_invf_f : d_invf_s;
        bf16* kcache = slot.kc[li];   // aliases the provider's for shared layers
        bf16* vcache = slot.vc[li];

        // residual = hidden; input_layernorm
        cudaMemcpyAsync(d_tmpH,d_hidden,(size_t)T*H*2,cudaMemcpyDeviceToDevice,st);
        rmsnorm_kernel<<<T,1024, 0, st>>>(d_hidden,L.input_ln,d_norm,T,H,eps);

        // Decode quant (T==1). Two mutually exclusive strategies:
        //   use_q40 (GGUF Q4_0): every matmul via the SHARED dp4a MMVQ (mmvq.cuh), off the native
        //   on-disk nibbles — quantize the bf16 activation to Q8_1, mmvq → f32, cast to bf16.
        //   use_fp4 (safetensors): Q/K via FP8 (index precision), V/O+FFN via NVFP4 (content).
        // Prefill (T>1) under use_q40 dequants Q4_0→scratch then cuBLAS; the BF16 path keeps cuBLAS.
        const bool fp4_dec = eng->use_fp4 && T==1;
        const bool q40_dec = eng->use_q40 && T==1;
        auto q40dec = [&](bf16* y, const uint8_t* wb, const bf16* x, int out, int in){
            quantize_q8_1_bf16_kernel<<<in/32,32,0,st>>>(x, eng->d_q40_qa, eng->d_q40_da, eng->d_q40_sa, in);
            mmvq_launch(eng->d_fp4_yf, wb, eng->d_q40_qa, eng->d_q40_da, eng->d_q40_sa, in, out, 2, st);
            e4bfp4::to_bf16<<<(out+255)/256,256,0,st>>>(eng->d_fp4_yf, y, out);
        };
        // Quantize-once split of q40dec: the SAME activation feeds several Q4_0 GEMVs
        // (d_norm → q,k,v and → gate,up). Quantizing it once into the shared Q8_1 scratch
        // and replaying the mmvq is bit-identical (the kernel reads the same qa/da/sa) but
        // drops ~126 redundant quantize_q8_1 launches/token. q40quant writes the scratch;
        // q40gemv consumes it (no intervening re-quant on the same stream).
        auto q40quant = [&](const bf16* x, int in){
            quantize_q8_1_bf16_kernel<<<in/32,32,0,st>>>(x, eng->d_q40_qa, eng->d_q40_da, eng->d_q40_sa, in);
        };
        auto q40gemv = [&](bf16* y, const uint8_t* wb, int out, int in){
            mmvq_launch(eng->d_fp4_yf, wb, eng->d_q40_qa, eng->d_q40_da, eng->d_q40_sa, in, out, 2, st);
            e4bfp4::to_bf16<<<(out+255)/256,256,0,st>>>(eng->d_fp4_yf, y, out);
        };
        auto prefill_q40 = [&](bf16* y, const uint8_t* wb, const bf16* x, int out, int in){
            int64_t nblk=(int64_t)out*(in/32);
            dequant_q4_0_to_bf16_kernel<<<(unsigned)((nblk+255)/256),256,0,0>>>(wb, eng->d_q40_wdq, in, nblk);
            linear(eng->cublas, eng->d_q40_wdq, x, y, T, out, in);
        };
        // SMALL-T (verify / small prefill) Q4_0 projection: batched dp4a MMVQ over the T rows —
        // each Q4_0 weight ROW is read ONCE and dp4a'd against all T tokens (token-major Q8_1
        // activation in the shared d_q40_qa/da/sa scratch, sized S*FF=32*FF). This is the SAME
        // kernel family as the T==1 decode (q40dec/mmvq_launch) and step_batch's bproj (:2490),
        // so per-row math is bit-identical → the verify stays lossless. NO full weight dequant
        // (unlike prefill_q40). Only valid while T ≤ S (the batch-scratch ceiling); the caller
        // gates on T ≤ Q40_SMALL_T. cuBLAS prefill_q40 still wins for LARGE T.
        auto batched_q40 = [&](bf16* y, const uint8_t* wb, const bf16* x, int out, int in){
            quantize_q8_1_bf16_kernel<<<(T*in)/32,32,0,st>>>(x, eng->d_q40_qa, eng->d_q40_da, eng->d_q40_sa, T*in);
            mmvq_batched_launch(eng->d_fp4_yf, wb, eng->d_q40_qa, eng->d_q40_da, eng->d_q40_sa, in, out, T, 2, st);
            e4bfp4::to_bf16<<<(unsigned)(((size_t)T*out+255)/256),256,0,st>>>(eng->d_fp4_yf, y, T*out);
        };
        // Small-T (verify forward T=K+1 always qualifies; small prefill chunks too) routes Q4_0
        // projections through the batched dp4a path (read weights once, no BF16 dequant). S=32 is
        // the d_q40_qa scratch ceiling; above it cuBLAS GEMM wins, so fall back to prefill_q40.
        const int Q40_SMALL_T = 32;
        const bool q40_small = eng->use_q40 && T <= Q40_SMALL_T;
        auto wproj_q40 = [&](bf16* y, const uint8_t* wb, const bf16* x, int out, int in){
            if (q40_small) batched_q40(y, wb, x, out, in);
            else           prefill_q40(y, wb, x, out, in);
        };

        // d_norm (input_layernorm output) feeds q, k AND v — quantize it to Q8_1 ONCE
        // and reuse the scratch across all three projections (decode path only).
        if (q40_dec) q40quant(d_norm, H);
        // Q (always projected) — q_norm, rope at absolute positions
        if      (q40_dec)      q40gemv(d_q, L.q40_wq, qd, H);
        else if (fp4_dec)      e4bfp4::e4b_fp8_gemv_bf16(d_q, L.fp8_wq, d_norm, eng->d_fp4_xf, eng->d_fp4_yf, st);
        else if (eng->use_q40) wproj_q40(d_q, L.q40_wq, d_norm, qd, H);
        else                   linear(eng->cublas,L.wq,d_norm,d_q,T,qd,H);
        head_rmsnorm_kernel<<<T*c.n_heads,256, 0, st>>>(d_q,L.q_norm,T,c.n_heads,hd,eps);
        if      (persist) rope_batch_kernel<<<c.n_heads,256, 0, st>>>(d_q,d_invf,1,c.n_heads,hd,eng->dec.npast);
        else if (vgraph)  rope_verify_kernel<<<T*c.n_heads,256, 0, st>>>(d_q,d_invf,T,c.n_heads,hd,eng->vs.npast);
        else              rope_kernel<<<T*c.n_heads,256, 0, st>>>(d_q,d_invf,T,c.n_heads,hd,n_past);

        // K/V: non-shared layers project + store into their cache; shared layers
        // skip (their provider — an earlier layer this same step — already wrote it).
        if (!shared){
            if (q40_dec){
                // Reuse the d_norm Q8_1 scratch quantized above (same activation).
                q40gemv(d_k, L.q40_wk, kvd, H);
                q40gemv(d_v, L.q40_wv, kvd, H);
            } else if (fp4_dec){
                e4bfp4::e4b_fp8_gemv_bf16(d_k, L.fp8_wk, d_norm, eng->d_fp4_xf, eng->d_fp4_yf, st);
                e4bfp4::e4b_nvfp4_gemv_bf16(d_v, L.fp4_wv, d_norm, eng->d_fp4_xf, eng->d_fp4_yf, st);
            } else if (eng->use_q40){
                wproj_q40(d_k, L.q40_wk, d_norm, kvd, H);
                wproj_q40(d_v, L.q40_wv, d_norm, kvd, H);
            } else {
                linear(eng->cublas,L.wk,d_norm,d_k,T,kvd,H);
                linear(eng->cublas,L.wv,d_norm,d_v,T,kvd,H);
            }
            head_rmsnorm_kernel<<<T*c.n_kv_heads,256, 0, st>>>(d_k,L.k_norm,T,c.n_kv_heads,hd,eps);
            head_rmsnorm_kernel<<<T*c.n_kv_heads,256, 0, st>>>(d_v,nullptr,T,c.n_kv_heads,hd,eps); // v_norm (no weight)
            if (persist){
                rope_batch_kernel<<<c.n_kv_heads,256, 0, st>>>(d_k,d_invf,1,c.n_kv_heads,hd,eng->dec.npast);
                kv_store_dev_kernel<<<GRID(kvd),256, 0, st>>>(d_k,kcache,kvd,eng->dec.npast,cap);
                kv_store_dev_kernel<<<GRID(kvd),256, 0, st>>>(d_v,vcache,kvd,eng->dec.npast,cap);
            } else if (vgraph){
                rope_verify_kernel<<<T*c.n_kv_heads,256, 0, st>>>(d_k,d_invf,T,c.n_kv_heads,hd,eng->vs.npast);
                kv_store_verify_kernel<<<GRID(T*kvd),256, 0, st>>>(d_k,kcache,T,kvd,eng->vs.npast,cap);
                kv_store_verify_kernel<<<GRID(T*kvd),256, 0, st>>>(d_v,vcache,T,kvd,eng->vs.npast,cap);
            } else {
                rope_kernel<<<T*c.n_kv_heads,256, 0, st>>>(d_k,d_invf,T,c.n_kv_heads,hd,n_past);
                kv_store_kernel<<<GRID(T*kvd),256, 0, st>>>(d_k,kcache,T,kvd,n_past,cap);
                kv_store_kernel<<<GRID(T*kvd),256, 0, st>>>(d_v,vcache,T,kvd,n_past,cap);
            }
        }

        // attention over the cache [0, n_past+T). Decode (T==1) uses the fixed-smem, device-
        // position flash kernel (CUDA-graph-able); prefill uses the dynamic-smem multi-query path.
        if (persist){
            // Split-K flash decode: n_heads*SPLITS blocks (4 warps each) compute per-split
            // (m,l,acc) partials, then a combine kernel merges them per head. Grid + smem fixed
            // (position from d_npast) → CUDA-graph-capturable. Far higher occupancy than the old
            // one-block-per-head kernel for the bandwidth-bound KV reads.
            constexpr int NW = 4;
            const int smemS = NW * hd * (int)sizeof(float);
            dim3 sg(c.n_heads, E4B_FA_SPLITS);
            attn_flash_decode_split_kernel<NW><<<sg, NW*32, smemS, st>>>(
                d_q,kcache,vcache,eng->dec.fa_m,eng->dec.fa_l,eng->dec.fa_acc,
                eng->dec.npast,c.n_heads,c.n_kv_heads,hd,1.0f,window,cap,E4B_FA_SPLITS);
            attn_flash_combine_kernel<<<c.n_heads, 256, 0, st>>>(
                eng->dec.fa_m,eng->dec.fa_l,eng->dec.fa_acc,d_attn,c.n_heads,hd,E4B_FA_SPLITS);
        } else if (vgraph){
            // Fixed-smem tiled flash over T rows (device base position) — capturable. One block
            // per (head, t). smem = (TILE + hd) floats, independent of n_past.
            const int smemF = (256 + hd) * (int)sizeof(float);
            dim3 ag(c.n_heads,T);
            attn_flash_verify_kernel<256><<<ag,256,smemF, st>>>(
                d_q,kcache,vcache,d_attn,eng->vs.npast,T,c.n_heads,c.n_kv_heads,hd,1.0f,window,cap);
        } else {
            // smem holds the attended span: sliding layers read at most `window` keys,
            // so cap the allocation there instead of growing with n_past+T.
            dim3 ag(c.n_heads,T);
            const int span = (window>0 && window<(n_past+T)) ? window : (n_past+T);
            size_t sh=(size_t)span*sizeof(float);
            attn_cache_kernel<<<ag,256,sh, st>>>(d_q,kcache,vcache,d_attn,T,n_past,c.n_heads,c.n_kv_heads,hd,1.0f,window,cap);
        }

        // o_proj → d_norm; post_attention_layernorm; hidden = residual + that
        if      (q40_dec)      q40dec(d_norm, L.q40_wo, d_attn, H, qd);
        else if (fp4_dec)      e4bfp4::e4b_nvfp4_gemv_bf16(d_norm, L.fp4_wo, d_attn, eng->d_fp4_xf, eng->d_fp4_yf, st);
        else if (eng->use_q40) wproj_q40(d_norm, L.q40_wo, d_attn, H, qd);
        else                   linear(eng->cublas,L.wo,d_attn,d_norm,T,H,qd);
        // Fused post_attn norm + residual add → d_hidden (also leaves the new residual in
        // d_tmpH, which the FFN block re-copies from d_hidden below — so write both).
        rmsnorm_resid_kernel<<<T,1024, 0, st>>>(d_norm,L.post_attn_ln,d_tmpH,d_hidden,T,H,eps,0.f,0);

        // FFN: residual; pre_ff norm; GeGLU; post_ff norm; residual add
        cudaMemcpyAsync(d_tmpH,d_hidden,(size_t)T*H*2,cudaMemcpyDeviceToDevice,st);
        rmsnorm_kernel<<<T,1024, 0, st>>>(d_hidden,L.pre_ff_ln,d_norm,T,H,eps);
        // FFN projections. Single-token decode (T==1) reads the NVFP4 weights via the
        // bandwidth-tuned GEMV (default; unless FUCINA_E4B_FP4=0); prefill (T>1) stays on the BF16
        // tensor-core GEMM. GeGLU and the surrounding norms/residual are identical.
        if (q40_dec){
            // Unfused gate/up + GeGLU + down (matches the batched path bit-for-bit; the fused
            // mmvq_glu keeps gate·up in f32 and would diverge from the batched unfused rounding).
            // d_norm (pre_ff_ln output) feeds BOTH gate and up — quantize once, reuse.
            q40quant(d_norm, H);
            q40gemv(d_gate, L.q40_gate, FF, H);
            q40gemv(d_up,   L.q40_up,   FF, H);
            geglu_kernel<<<GRID(T*FF),256, 0, st>>>(d_gate,d_up,d_act,T*FF);
            q40dec(d_norm, L.q40_down, d_act, H, FF);
        } else if (fp4_dec){
            e4bfp4::e4b_nvfp4_gemv_bf16(d_gate, L.fp4_gate, d_norm, eng->d_fp4_xf, eng->d_fp4_yf, st);
            e4bfp4::e4b_nvfp4_gemv_bf16(d_up,   L.fp4_up,   d_norm, eng->d_fp4_xf, eng->d_fp4_yf, st);
            geglu_kernel<<<GRID(T*FF),256, 0, st>>>(d_gate,d_up,d_act,T*FF);
            e4bfp4::e4b_nvfp4_gemv_bf16(d_norm, L.fp4_down, d_act,  eng->d_fp4_xf, eng->d_fp4_yf, st);
        } else if (eng->use_q40){
            wproj_q40(d_gate, L.q40_gate, d_norm, FF, H);
            wproj_q40(d_up,   L.q40_up,   d_norm, FF, H);
            geglu_kernel<<<GRID(T*FF),256, 0, st>>>(d_gate,d_up,d_act,T*FF);
            wproj_q40(d_norm, L.q40_down, d_act,  H, FF);
        } else {
            linear(eng->cublas,L.w_gate,d_norm,d_gate,T,FF,H);
            linear(eng->cublas,L.w_up,d_norm,d_up,T,FF,H);
            geglu_kernel<<<GRID(T*FF),256, 0, st>>>(d_gate,d_up,d_act,T*FF);
            linear(eng->cublas,L.w_down,d_act,d_norm,T,H,FF);
        }
        // Fused post_ff norm + residual add → d_hidden.
        rmsnorm_resid_kernel<<<T,1024, 0, st>>>(d_norm,L.post_ff_ln,d_tmpH,d_hidden,T,H,eps,0.f,0);

        // PLE combine: residual; gate(hidden)→256; gelu; ×per_layer_input; proj; norm; add
        cudaMemcpyAsync(d_tmpH,d_hidden,(size_t)T*H*2,cudaMemcpyDeviceToDevice,st);
        linear(eng->cublas,L.ple_in_gate,d_hidden,d_pleg,T,PD,H);
        ple_gate_strided<<<GRID(T*PD),256, 0, st>>>(d_pleg, d_pli, T, PD, W, li);
        linear(eng->cublas,L.ple_proj,d_pleg,d_norm,T,H,PD);
        // Fused post_ple norm + residual add + ×layer_scalar → d_hidden.
        rmsnorm_resid_kernel<<<T,1024, 0, st>>>(d_norm,L.post_ple_ln,d_tmpH,d_hidden,T,H,eps,L.layer_scalar,1);

        if (li==0 && l0_out){ float* tmp; cudaMalloc(&tmp,(size_t)T*H*4);
            to_f32_kernel<<<GRID(T*H),256, 0, st>>>(d_hidden,tmp,T*H);
            cudaMemcpy(l0_out,tmp,(size_t)T*H*4,cudaMemcpyDeviceToHost); cudaFree(tmp); }
    }

    // final norm
    rmsnorm_kernel<<<T,1024, 0, st>>>(d_hidden,eng->d_final_norm,d_norm,T,H,eps);
    if(fin_out){
        if (verify){
            // Verify (graph-capturable): write into the persistent device buffer; the host D2H is
            // issued AFTER the replay so the captured region has no cudaMalloc/Memcpy.
            to_f32_kernel<<<GRID(T*H),256, 0, st>>>(d_norm,eng->vs.fin,T*H);
        } else {
            float* tmp; cudaMalloc(&tmp,(size_t)T*H*4);
            to_f32_kernel<<<GRID(T*H),256, 0, st>>>(d_norm,tmp,T*H);
            cudaMemcpy(fin_out,tmp,(size_t)T*H*4,cudaMemcpyDeviceToHost); cudaFree(tmp);
        }
    }

    // ── all-rows verify head (spec-decode): project d_norm[0..T-1] → [T,V], softcap,
    // per-row argmax, D2H. Same head as the last-row path but over every row; equivalent
    // to running e4b_step over the candidates and reading each row's greedy pick. Only
    // active when out_argmax_rows is set (T>1 verify); never on the graph/decode path.
    if (out_argmax_rows){
        float* d_lg = verify ? eng->vs.lg : nullptr;
        if (!verify) cudaMalloc(&d_lg,(size_t)T*V*4);
        if (eng->use_q40){   // native Q6_K head, batched over T rows (read once)
            quantize_q8_1_bf16_kernel<<<(T*H)/32,32,0,st>>>(d_norm, eng->d_q40_qa, eng->d_q40_da, eng->d_q40_sa, T*H);
            mmvq_q6_k_batched_launch(d_lg, eng->d_q6k_head, eng->d_q40_qa, eng->d_q40_da, H, V, T, st);
        } else {
            bf16* d_lg_bf = verify ? eng->vs.lg_bf : nullptr;
            if (!verify) cudaMalloc(&d_lg_bf,(size_t)T*V*2);
            linear(eng->cublas, eng->d_embed, d_norm, d_lg_bf, T, V, H);
            to_f32_kernel<<<GRID(T*V),256, 0, st>>>(d_lg_bf,d_lg,T*V);
            if (!verify) cudaFree(d_lg_bf);
        }
        softcap_kernel<<<GRID(T*V),256, 0, st>>>(d_lg,T*V,c.final_logit_softcap);
        int* d_am = verify ? eng->vs.am : nullptr;
        if (!verify) cudaMalloc(&d_am,(size_t)T*sizeof(int));
        argmax_rows_kernel<<<T,256, 0, st>>>(d_lg,T,V,d_am);
        // Verify (graph-capturable): leave the argmax in the persistent device buffer; the D2H
        // and sync happen AFTER the replay (outside the captured region). The malloc fallback
        // (non-verify) D2Hs + syncs + frees here as before.
        if (!verify){
            cudaMemcpyAsync(out_argmax_rows,d_am,(size_t)T*sizeof(int),cudaMemcpyDeviceToHost,st);
            cudaStreamSynchronize(st);
            cudaFree(d_am); cudaFree(d_lg);
        }
    }

    // logits of the last token (tied head + softcap). ALWAYS a 1-row projection (last token
    // only), so the quantized head applies whenever quant is on — including prefill's last
    // token. Writes fp32 logits directly. use_q40 → native Q6_K head (mmvq_q6_k); the
    // safetensors NVFP4 path → FP8 head; BF16 fallback → cuBLAS.
    if (logits_last_out){
        const bf16* xrow = d_norm + (size_t)(T-1)*H;
        float* d_logits_f = persist ? eng->dec.logits_f : nullptr;
        if (!persist) cudaMalloc(&d_logits_f,(size_t)V*4);
        if (eng->use_q40){   // native Q6_K tied head via shared MMVQ
            quantize_q8_1_bf16_kernel<<<H/32,32,0,st>>>(xrow, eng->d_q40_qa, eng->d_q40_da, eng->d_q40_sa, H);
            mmvq_q6_k_launch(d_logits_f, eng->d_q6k_head, eng->d_q40_qa, eng->d_q40_da, H, V, st);
        } else if (eng->use_fp4){
            e4bfp4::e4b_fp8_gemv_f32(d_logits_f, eng->fp8_head, xrow, eng->d_fp4_xf, st);
        } else {
            bf16* d_logits_bf; cudaMalloc(&d_logits_bf,(size_t)V*2);
            linear(eng->cublas, eng->d_embed, xrow, d_logits_bf, 1, V, H);
            to_f32_kernel<<<GRID(V),256, 0, st>>>(d_logits_bf,d_logits_f,V);
            cudaFree(d_logits_bf);
        }
        softcap_kernel<<<GRID(V),256, 0, st>>>(d_logits_f,V,c.final_logit_softcap);
        // Decode (persist): logits stay in dec.logits_f; the D2H is issued AFTER the graph
        // replay below (it must not be part of the captured region). Prefill (!persist):
        // per-call scratch, copy + free here.
        if (!persist){ cudaMemcpy(logits_last_out,d_logits_f,(size_t)V*4,cudaMemcpyDeviceToHost); cudaFree(d_logits_f); }
    }
    };  // end forward

    // ── run the forward: CUDA-graph capture/replay for single-token decode, else direct ──
    // The decode body is identical every step except for the two device inputs (token id,
    // n_past) already H2D'd above; capture it once and replay, eliding ~250 launch overheads.
    const bool graph_path = persist && !eng->dec.graph_disabled &&
                            logits_last_out && !emb_out && !l0_out && !fin_out;
    // Verify-graph path: the FIXED-T spec-verify forward (out_argmax_rows + fin_out, no
    // logits_last/emb/l0) is capturable just like decode — device base position (vs.npast) and
    // ids are the only per-replay inputs, and the head writes argmax/fin into persistent device
    // buffers that are D2H'd after the replay. A captured graph is valid only at the T it was
    // captured at (block dims / row counts are baked), so guard the replay on vs.cap_T==T.
    const bool vgraph_path = vgraph && out_argmax_rows && fin_out &&
                             !logits_last_out && !emb_out && !l0_out;
    if (vgraph_path && eng->vs.graph_ready && eng->vs.cap_T==T){
        cudaGraphLaunch(eng->vs.gexec, st);
    } else if (vgraph_path && eng->vs.graph_ready){
        // T changed (rare near-context-full Tv=1 fallback): the captured graph no longer matches.
        // Run eagerly this round; keep the existing graph for the common T.
        forward();
    } else if (vgraph_path){
        // First verify at this T: eager warmup (cuBLAS workspace / lazy init must not happen
        // during capture), then capture and replay. Warmup writes KV at the same device n_past
        // the replay will, so it is idempotent.
        forward();
        cudaStreamSynchronize(st);
        cudaGraph_t g=nullptr;
        cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
        forward();
        cudaError_t cap = cudaStreamEndCapture(st,&g);
        if (cap==cudaSuccess && cudaGraphInstantiate(&eng->vs.gexec,g,0)==cudaSuccess){
            eng->vs.graph_ready=true; eng->vs.cap_T=T;
            cudaGraphDestroy(g);
            cudaGraphLaunch(eng->vs.gexec, st);    // capture records but does not execute
        } else {
            eng->vs.graph_disabled=true; cudaGetLastError();
            if(g) cudaGraphDestroy(g);
            fprintf(stderr,"e4b: verify CUDA-graph capture failed (%s); verify falls back to eager\n",
                    cudaGetErrorString(cap));
            forward();                              // eager fallback (executes)
        }
    } else if (graph_path && eng->dec.graph_ready){
        cudaGraphLaunch(eng->dec.gexec, st);
    } else if (graph_path){
        // First decode: run once eagerly to warm up (cuBLAS workspace allocation and any
        // lazy one-time init must NOT happen during capture, or BeginCapture aborts). The
        // warmup writes KV at the same device n_past the replay will, so it is idempotent.
        forward();
        cudaStreamSynchronize(st);
        cudaGraph_t g=nullptr;
        cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal);
        forward();
        cudaError_t cap = cudaStreamEndCapture(st,&g);
        if (cap==cudaSuccess && cudaGraphInstantiate(&eng->dec.gexec,g,0)==cudaSuccess){
            eng->dec.graph_ready=true;
            cudaGraphDestroy(g);
            cudaGraphLaunch(eng->dec.gexec, st);   // capture records but does not execute
        } else {
            eng->dec.graph_disabled=true; cudaGetLastError();
            if(g) cudaGraphDestroy(g);
            fprintf(stderr,"e4b: CUDA-graph capture failed (%s); decode falls back to eager\n",
                    cudaGetErrorString(cap));
            forward();                              // eager fallback (executes)
        }
    } else {
        forward();                                  // prefill / debug: direct
    }

    // Decode logits D2H — outside the captured region so the host buffer can change per call.
    if (logits_last_out && persist)
        cudaMemcpyAsync(logits_last_out, eng->dec.logits_f, (size_t)V*4, cudaMemcpyDeviceToHost, st);
    // Verify argmax + post-final-norm D2H — outside the captured region (host buffers vary per
    // round). The non-graph verify path D2H'd argmax inside forward and fin via cudaMemcpy; here
    // the graph (and eager-fallback) path moves both out. Eager verify also lands here because
    // forward() left them in the persistent device buffers.
    if (verify){
        cudaMemcpyAsync(out_argmax_rows, eng->vs.am, (size_t)T*sizeof(int), cudaMemcpyDeviceToHost, st);
        if (fin_out) cudaMemcpyAsync(fin_out, eng->vs.fin, (size_t)T*H*4, cudaMemcpyDeviceToHost, st);
    }

    cudaError_t err = (persist || vgraph) ? cudaStreamSynchronize(st) : cudaDeviceSynchronize();
    if (!persist && !verify){   // prefill scratch is per-call; decode + verify scratch are
        // persistent (freed at destroy). Freeing the verify buffers here would dangle eng->vs.
        cudaFree(d_ids);cudaFree(d_hidden);cudaFree(d_norm);cudaFree(d_tmpH);cudaFree(d_q);cudaFree(d_k);
        cudaFree(d_v);cudaFree(d_attn);cudaFree(d_gate);cudaFree(d_up);cudaFree(d_act);cudaFree(d_pleg);
        cudaFree(d_ple_lookup);cudaFree(d_ple_ctx);cudaFree(d_pli);
        cudaFree(d_invf_s);cudaFree(d_invf_f);
    }
    if(err!=cudaSuccess){ fprintf(stderr,"e4b step sync: %s\n",cudaGetErrorString(err)); return -1; }
    slot.n_past += T;     // commit the new tokens to the cache
    return 0;
}

// Reset rewinds slot 0 to empty AND drops the prefix-cache history: an explicit reset
// means the next prompt is unrelated, so no reuse. Keeping the invariant "hist non-empty
// ⇒ n_past is the true KV high-water" (only Prefill/Decode advance n_past thereafter)
// makes the sliding-ring rewind-safety check sound.
extern "C" void e4b_engine_reset(e4b_engine_t *eng){ if(eng) eng->slots[0].n_past=0; }
extern "C" int  e4b_engine_n_past(const e4b_engine_t *eng){ return eng ? eng->slots[0].n_past : 0; }

// Append `suffix[0,n)` into slot 0 at its CURRENT n_past, honoring the sliding ring:
// when active (sliding_cap < max_ctx), a single attn pass over more than
// (sliding_cap - window) tokens would wrap the ring onto positions still inside a live
// query's window. So we drive e4b_step in chunks of (sliding_cap - window): within a
// chunk the span [oldest-needed, newest-written] is < sliding_cap, so no slot collides.
// Output-equivalent to a single pass (attention is causal, positions are absolute).
static int e4b_append_slot(e4b_engine* eng, Slot& slot, const int32_t* suffix, int n,
                           float* logits_out){
    if (n <= 0) return -1;
    if (slot.n_past + n > eng->max_ctx){
        fprintf(stderr,"e4b: context overflow (%d + %d > %d)\n", slot.n_past, n, eng->max_ctx); return -1; }
    const int chunk = eng->sliding_cap - eng->cfg.sliding_window;
    if (eng->sliding_cap >= eng->max_ctx || chunk < 1 || n <= chunk)
        return e4b_step(eng, slot, suffix, n, nullptr,nullptr,nullptr, logits_out);
    for (int off=0; off<n; off+=chunk){
        const int t = (n-off<chunk)?(n-off):chunk;
        const bool last = (off+t>=n);
        const int rc = e4b_step(eng, slot, suffix+off, t, nullptr,nullptr,nullptr,
                                last?logits_out:nullptr);
        if (rc!=0) return rc;
    }
    return 0;
}

extern "C" int e4b_engine_prefill(e4b_engine_t *eng, const int32_t *tokens, int n_tokens, float *logits_out){
    if(!eng) return -1;
    eng->slots[0].n_past = 0;                       // fresh sequence on slot 0
    return e4b_append_slot(eng, eng->slots[0], tokens, n_tokens, logits_out);
}

// Cache-reuse prefill: append `suffix` at the CURRENT n_past (set by a prior Rewind).
// The server's KVCache manager computes the shared prefix, Rewinds slot 0 to it, then
// calls this with only the divergent suffix — turning a full re-prefill into suffix-only.
extern "C" int e4b_engine_prefill_append(e4b_engine_t *eng, const int32_t *suffix, int n, float *logits_out){
    if(!eng) return -1;
    return e4b_append_slot(eng, eng->slots[0], suffix, n, logits_out);
}

// Rewind slot 0's KV to n_keep tokens for prefix reuse. Returns 1 if safe, 0 if the
// sliding ring has already overwritten the window for position n_keep (rewind depth
// > sliding_cap - window) — the caller then does a full reset + re-prefill. Full-only
// layers (and the ring-inactive case) can always rewind.
extern "C" int e4b_engine_rewind(e4b_engine_t *eng, int n_keep){
    if(!eng) return 0;
    Slot& s = eng->slots[0];
    if (n_keep < 0 || n_keep > s.n_past) return 0;
    if (eng->sliding_cap < eng->max_ctx){
        const int max_rewind = eng->sliding_cap - eng->cfg.sliding_window;
        if (s.n_past - n_keep > max_rewind) return 0;   // sliding window for n_keep is gone
    }
    s.n_past = n_keep;
    return 1;
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

// ── MTP drafter forward (increment 2) ───────────────────────────────────────
// ONE assistant-head step: from the recurrent hidden h (d_h_io, [H_out]) of the
// preceding token + token id `tok` at absolute RoPE position `pos`, produce the next
// draft token id (*out_draft_id) and overwrite d_h_io with the next recurrent hidden.
// Q-only: each layer attends the TARGET slot's provider-layer KV (prov_sliding for the
// 3 sliding layers, prov_full for the global layer). All GEMVs via cuBLAS `linear` over
// the BF16-dequantized assistant weights. See docs/e4b-mtp-plan.md §"Drafter forward".
// Lazily allocate the persistent drafter scratch (once, freed in destroy). All buffers
// live in the E4bMtp struct so the hot loop never cudaMallocs.
static bool e4b_mtp_scratch_ensure(e4b_engine* eng){
    E4bMtp& m = eng->mtp;
    if (m.scratch_ready) return true;
    const int H=m.H_out, AH=m.AH, FF=m.FF, V=m.vocab;
    const int QM = m.q_heads * m.global_head_dim;   // largest q/attn width (global layer)
    auto MB=[&](__nv_bfloat16** p,int n)->bool{ return cudaMalloc(p,(size_t)n*sizeof(__nv_bfloat16))==cudaSuccess; };
    bool ok = MB(&m.d_xh,2*H) && MB(&m.d_cur,AH) && MB(&m.d_t1,AH) && MB(&m.d_t2,AH)
           && MB(&m.d_q,QM) && MB(&m.d_attn,QM) && MB(&m.d_ffa,FF) && MB(&m.d_ffb,FF)
           && MB(&m.d_logits_bf,V)
           && cudaMalloc(&m.d_logits,(size_t)V*sizeof(float))==cudaSuccess
           && cudaMalloc(&m.d_pos,sizeof(int))==cudaSuccess
           && cudaMalloc(&m.d_argmax,(size_t)E4B_MTP_KMAX*sizeof(int))==cudaSuccess
           && cudaMalloc(&m.d_tok,sizeof(int32_t))==cudaSuccess
           && cudaMalloc(&m.d_draft_h,(size_t)H*sizeof(float))==cudaSuccess;
    if (ok && m.use_q40){
        const int max_in = 2*H;     // pre_proj activation (largest GEMV input)
        const int max_out = V;      // unembed (largest GEMV output)
        ok = ok && cudaMalloc(&m.dq_qa,(size_t)max_in)==cudaSuccess
                && cudaMalloc(&m.dq_da,(size_t)(max_in/32)*sizeof(float))==cudaSuccess
                && cudaMalloc(&m.dq_sa,(size_t)(max_in/32)*sizeof(int32_t))==cudaSuccess
                && cudaMalloc(&m.dq_yf,(size_t)max_out*sizeof(float))==cudaSuccess;
    }
    if(!ok){ fprintf(stderr,"e4b-mtp: forward scratch alloc failed\n"); return false; }
    m.scratch_ready=true;
    return true;
}

// Drafter forward BODY (no argmax): xh=[embed(d_tok)·√H | d_h_io] → pre_proj → decoder
// layers (Q-only vs the TARGET provider-layer KV at device pos m.d_pos) → out_norm →
// logits in m.d_logits_bf, and the next recurrent hidden post_proj·t1 → d_h_io (f32).
// EVERYTHING runs on `st` and reads its per-step inputs (m.d_tok, m.d_pos, d_h_io) from
// DEVICE memory, so this exact sequence is capturable into a CUDA graph and replayed per
// draft token. cuBLAS must already be bound to `st` (cublasSetStream) by the caller.
static bool e4b_mtp_forward_body(e4b_engine* eng, Slot& slot, float* d_h_io, cudaStream_t st){
    E4bMtp& m = eng->mtp;
    const e4b::Config& c = eng->cfg;
    const int H=m.H_out, AH=m.AH, FF=m.FF, V=m.vocab;
    const float eps = m.rms_eps;
    auto GRID=[&](int n){ return (unsigned)((n+255)/256); };

    // Stage 2: dp4a Q4_0 GEMV (mirror of the target q40dec lambda) — quantize the BF16
    // activation to Q8_1, dp4a-MMVQ off the native Q4_0 weight nibbles (read once), convert
    // the f32 result back to BF16. No cuBLAS, graph-capturable, tiny-dim friendly. Reuses the
    // single persistent drafter dp4a scratch (sized for the largest in/out). in must be %32.
    const bool q40 = m.use_q40;
    auto dp4a=[&](const uint8_t* wb, const bf16* x, bf16* y, int out, int in){
        quantize_q8_1_bf16_kernel<<<in/32,32,0,st>>>(x, m.dq_qa, m.dq_da, m.dq_sa, in);
        mmvq_launch(m.dq_yf, wb, m.dq_qa, m.dq_da, m.dq_sa, in, out, 2, st);
        e4bfp4::to_bf16<<<(out+255)/256,256,0,st>>>(m.dq_yf, y, out);
    };

    // 1. xh = [ embed(tok)·√H | h ]  (H + H = 2H).
    embed_kernel<<<1,256,0,st>>>(eng->d_embed, m.d_tok, m.d_xh, 1, H, sqrtf((float)H));
    f32_to_bf16_kernel<<<(H+255)/256,256,0,st>>>(d_h_io, m.d_xh + H, H);
    // 2. cur = pre_proj · xh   (pre_proj [AH ← 2H]).
    if(q40) dp4a(m.q40_pre, m.d_xh, m.d_cur, AH, 2*H);
    else if(!linear(eng->cublas, m.pre_proj, m.d_xh, m.d_cur, 1, AH, 2*H)) return false;

    // 3. decoder layers (Q-only attention vs the target provider-layer KV).
    for (int l=0; l<m.n_layers; ++l){
        E4bMtpLayer& L = m.layers[l];
        const bool   glob   = L.is_global;
        const int    hd     = glob ? m.global_head_dim : c.head_dim;
        const int    qd     = m.q_heads * hd;
        const int    window = glob ? 0 : c.sliding_window;
        const int    cap    = glob ? eng->max_ctx : eng->sliding_cap;
        const int    prov   = glob ? eng->prov_full : eng->prov_sliding;
        float* invf         = glob ? m.invf_global : m.invf_sliding;
        const __nv_bfloat16* kc = slot.kc[prov];
        const __nv_bfloat16* vc = slot.vc[prov];

        rmsnorm_kernel<<<1,1024,0,st>>>(m.d_cur, L.attn_norm_b, m.d_t1, 1, AH, eps);
        if(q40) dp4a(L.q40_wq, m.d_t1, m.d_q, qd, AH);
        else if(!linear(eng->cublas, L.wq, m.d_t1, m.d_q, 1, qd, AH)) return false;
        head_rmsnorm_kernel<<<m.q_heads,256,0,st>>>(m.d_q, L.q_norm_b, 1, m.q_heads, hd, eps);
        rope_batch_kernel<<<m.q_heads,256,0,st>>>(m.d_q, invf, 1, m.q_heads, hd, m.d_pos);
        const int smemF = (256 + hd) * (int)sizeof(float);
        attn_flash_decode_kernel<256><<<m.q_heads,256,smemF,st>>>(
            m.d_q, kc, vc, m.d_attn, m.d_pos, m.q_heads, m.kv_heads, hd, 1.0f, window, cap);
        if(q40) dp4a(L.q40_wo, m.d_attn, m.d_t1, AH, qd);
        else if(!linear(eng->cublas, L.wo, m.d_attn, m.d_t1, 1, AH, qd)) return false;
        rmsnorm_kernel<<<1,1024,0,st>>>(m.d_t1, L.post_attn_norm_b, m.d_t2, 1, AH, eps);
        add_kernel<<<GRID(AH),256,0,st>>>(m.d_t2, m.d_cur, AH);

        rmsnorm_kernel<<<1,1024,0,st>>>(m.d_t2, L.ffn_norm_b, m.d_t1, 1, AH, eps);
        if(q40){ dp4a(L.q40_gate, m.d_t1, m.d_ffa, FF, AH); dp4a(L.q40_up, m.d_t1, m.d_ffb, FF, AH); }
        else {
            if(!linear(eng->cublas, L.gate, m.d_t1, m.d_ffa, 1, FF, AH)) return false;
            if(!linear(eng->cublas, L.up,   m.d_t1, m.d_ffb, 1, FF, AH)) return false;
        }
        geglu_kernel<<<GRID(FF),256,0,st>>>(m.d_ffa, m.d_ffb, m.d_ffa, FF);
        if(q40) dp4a(L.q40_down, m.d_ffa, m.d_t1, AH, FF);
        else if(!linear(eng->cublas, L.down, m.d_ffa, m.d_t1, 1, AH, FF)) return false;
        rmsnorm_kernel<<<1,1024,0,st>>>(m.d_t1, L.post_ffw_norm_b, m.d_cur, 1, AH, eps);
        add_kernel<<<GRID(AH),256,0,st>>>(m.d_cur, m.d_t2, AH);
        scale_kernel<<<GRID(AH),256,0,st>>>(m.d_cur, L.out_scale, AH);
    }

    // 4. output: t1 = rmsnorm(cur, out_norm); logits = unembed·t1 (NO softcap) → m.d_logits_bf;
    //    h_next = post_proj·t1 → d_h_io (f32). (argmax is OUTSIDE the body / graph.)
    rmsnorm_kernel<<<1,1024,0,st>>>(m.d_cur, m.out_norm_b, m.d_t1, 1, AH, eps);
    if(q40){
        // unembed: dp4a directly to the f32 logits (mmvq outputs f32) — argmax reads m.d_logits.
        quantize_q8_1_bf16_kernel<<<AH/32,32,0,st>>>(m.d_t1, m.dq_qa, m.dq_da, m.dq_sa, AH);
        mmvq_launch(m.d_logits, m.q40_unembed, m.dq_qa, m.dq_da, m.dq_sa, AH, V, 2, st);
        // post_proj: dp4a to the f32 recurrent h (mmvq outputs f32) — fed back to the chain.
        quantize_q8_1_bf16_kernel<<<AH/32,32,0,st>>>(m.d_t1, m.dq_qa, m.dq_da, m.dq_sa, AH);
        mmvq_launch(d_h_io, m.q40_post, m.dq_qa, m.dq_da, m.dq_sa, AH, H, 2, st);
    } else {
        if(!linear(eng->cublas, m.unembed, m.d_t1, m.d_logits_bf, 1, V, AH)) return false;
        to_f32_kernel<<<GRID(V),256,0,st>>>(m.d_logits_bf, m.d_logits, V);
        if(!linear(eng->cublas, m.post_proj, m.d_t1, m.d_xh, 1, H, AH)) return false;  // reuse d_xh[0:H]
        to_f32_kernel<<<GRID(H),256,0,st>>>(m.d_xh, d_h_io, H);
    }
    return true;
}

// Lazy one-time capture of e4b_mtp_forward_body into a CUDA graph (mirror of the dense
// mtp_graph_ensure + the E4B decode graph). Warms up once eagerly first (cuBLAS workspace
// alloc must not run during capture) on m.mstream, then captures. FUCINA_E4B_MTP_NOGRAPH=1
// or any capture/instantiate failure leaves graph_disabled and the per-launch path runs.
static void e4b_mtp_graph_ensure(e4b_engine* eng, Slot& slot, float* d_h_io){
    E4bMtp& m = eng->mtp;
    if (m.graph_ready || m.graph_disabled) return;
    if (const char* e=getenv("FUCINA_E4B_MTP_NOGRAPH"); e && e[0]=='1'){ m.graph_disabled=true; return; }
    if (cudaStreamCreateWithFlags(&m.mstream,cudaStreamNonBlocking)!=cudaSuccess){ m.graph_disabled=true; return; }
    cublasSetStream(eng->cublas, m.mstream);
    // Warm up once eagerly (idempotent: writes only scratch + d_h_io, which the caller
    // re-seeds before the real chain). d_tok/d_pos already hold valid first-draft values.
    if(!e4b_mtp_forward_body(eng, slot, d_h_io, m.mstream)){ m.graph_disabled=true; return; }
    cudaStreamSynchronize(m.mstream);
    cudaGraph_t g=nullptr;
    cudaStreamBeginCapture(m.mstream, cudaStreamCaptureModeThreadLocal);
    bool body_ok = e4b_mtp_forward_body(eng, slot, d_h_io, m.mstream);
    cudaError_t cap = cudaStreamEndCapture(m.mstream,&g);
    if (body_ok && cap==cudaSuccess && cudaGraphInstantiate(&m.mgraph,g,0)==cudaSuccess){
        m.graph_ready=true;
        cudaGraphDestroy(g);
    } else {
        m.graph_disabled=true; cudaGetLastError();
        if(g) cudaGraphDestroy(g);
        fprintf(stderr,"e4b-mtp: CUDA-graph capture failed (%s); drafter falls back to eager\n",
                cudaGetErrorString(cap));
    }
}

// ── MTP drafter forward (increment 2; eager single-shot wrapper for the debug API) ──
// ONE assistant-head step from HOST tok/pos with a DEVICE recurrent-h (d_h_io); returns
// the draft id. The hot spec loop does NOT call this — it drives the body + graph directly
// (see e4b_mtp_draft below) to keep the chain on-GPU. Kept for e4b_engine_mtp_forward_debug.
static int e4b_mtp_forward(e4b_engine* eng, Slot& slot, int32_t tok, int pos,
                           float* d_h_io, int32_t* out_draft_id){
    if(!eng || !eng->mtp.loaded) return -1;
    E4bMtp& m = eng->mtp;
    cudaSetDevice(eng->device_id);
    if(!e4b_mtp_scratch_ensure(eng)) return -1;
    cudaMemcpy(m.d_tok,&tok,sizeof(int32_t),cudaMemcpyHostToDevice);
    cudaMemcpy(m.d_pos,&pos,sizeof(int),cudaMemcpyHostToDevice);
    cublasSetStream(eng->cublas, 0);
    if(!e4b_mtp_forward_body(eng, slot, d_h_io, 0)) return -1;
    argmax_kernel<<<1,256>>>(m.d_logits, m.vocab, m.d_argmax);
    int draft=0; cudaMemcpy(&draft, m.d_argmax, sizeof(int), cudaMemcpyDeviceToHost);
    cudaError_t e = cudaDeviceSynchronize();
    if (e!=cudaSuccess){ fprintf(stderr,"e4b-mtp forward: %s\n", cudaGetErrorString(e)); return -1; }
    if (out_draft_id) *out_draft_id = draft;
    return 0;
}

// ── Drafter draft loop: chain up to K draft tokens on the GPU ─────────────────
// Seeds m.d_tok = g and m.d_draft_h = recurrent h (host `h`), then for i in [0,K): set
// m.d_pos = base_pos+i (one tiny H2D, OUTSIDE the graph), replay the captured body (or launch
// eagerly), and run argmax → d_argmax[i] (+ chains the id into m.d_tok for the next replay's
// embed). NO sync between drafts; one D2H of the K ids + a single stream sync at the end.
// Returns D, the number of drafts produced, with drafts[0..D) filled.
static int e4b_mtp_draft(e4b_engine* eng, Slot& slot, int32_t g, const float* h, int base_pos,
                         int K, int K_room, int32_t* drafts){
    E4bMtp& m = eng->mtp;
    const int H = m.H_out;
    if(!e4b_mtp_scratch_ensure(eng)) return -1;
    if (K > E4B_MTP_KMAX) K = E4B_MTP_KMAX;
    // Try to capture the graph once. The warmup CONSUMES and OVERWRITES m.d_draft_h, so we
    // seed it with the real recurrent h here so the capture is well-formed, then RE-seed it
    // below (after capture) since the warmup clobbered it.
    if(!m.graph_ready && !m.graph_disabled){
        cudaMemcpy(m.d_draft_h, h, (size_t)H*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(m.d_tok,&g,sizeof(int32_t),cudaMemcpyHostToDevice);
        cudaMemcpy(m.d_pos,&base_pos,sizeof(int),cudaMemcpyHostToDevice);
        e4b_mtp_graph_ensure(eng, slot, /*d_h_io=*/m.d_draft_h);
    }
    const bool use_graph = m.graph_ready;
    cudaStream_t st = use_graph ? m.mstream : (cudaStream_t)0;
    cublasSetStream(eng->cublas, st);

    // Seed the chain: recurrent h and the first input token g (both on the working stream so
    // the graph/eager body reads the fresh values, not the capture-warmup leftovers).
    cudaMemcpyAsync(m.d_draft_h, h, (size_t)H*sizeof(float), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(m.d_tok,&g,sizeof(int32_t),cudaMemcpyHostToDevice,st);

    int D=0;
    for (int i=0;i<K;i++){
        const int pos = base_pos + i;
        if (pos < 0) break;
        if (i >= K_room) break;
        cudaMemcpyAsync(m.d_pos,&pos,sizeof(int),cudaMemcpyHostToDevice,st);
        if (use_graph){
            if (cudaGraphLaunch(m.mgraph, st)!=cudaSuccess){
                // replay failed → fall back to eager for the rest of this chain.
                cudaGetLastError();
                if(!e4b_mtp_forward_body(eng, slot, m.d_draft_h, st)) return -1;
            }
        } else {
            if(!e4b_mtp_forward_body(eng, slot, m.d_draft_h, st)) return -1;
        }
        // argmax → d_argmax[i] and chain the id into d_tok for the next replay's embed.
        argmax_kernel<<<1,256,0,st>>>(m.d_logits, m.vocab, m.d_argmax + i, m.d_tok);
        D++;
    }
    if (D>0){
        cudaMemcpyAsync(drafts, m.d_argmax, (size_t)D*sizeof(int), cudaMemcpyDeviceToHost, st);
    }
    cudaError_t e = cudaStreamSynchronize(st);
    if (e!=cudaSuccess){ fprintf(stderr,"e4b-mtp draft: %s\n", cudaGetErrorString(e)); return -1; }
    return D;
}

// Debug C wrapper: one drafter forward on slot 0 with a HOST recurrent-h buffer.
extern "C" int e4b_engine_mtp_forward_debug(e4b_engine_t *eng, int32_t tok, int pos,
                                            float *h_io, int32_t *draft_id){
    if(!eng || !eng->mtp.loaded || !h_io) return -1;
    cudaSetDevice(eng->device_id);
    const int H = eng->mtp.H_out;
    float* d_h=nullptr;
    if(cudaMalloc(&d_h,(size_t)H*sizeof(float))!=cudaSuccess) return -1;
    cudaMemcpy(d_h, h_io, (size_t)H*sizeof(float), cudaMemcpyHostToDevice);
    int32_t draft=-1;
    int rc = e4b_mtp_forward(eng, eng->slots[0], tok, pos, d_h, &draft);
    if (rc==0){ cudaMemcpy(h_io, d_h, (size_t)H*sizeof(float), cudaMemcpyDeviceToHost);
                if(draft_id) *draft_id=draft; }
    cudaFree(d_h);
    return rc;
}

// ── Increment 3: all-rows verify ────────────────────────────────────────────
// Run the SAME forward as e4b_step over the T candidate tokens (causal per-token attn
// + ring KV store at [n_past, n_past+T), advances n_past by T), but project EVERY row's
// post-final-norm hidden → logits, softcap, per-row argmax. out_argmax[i] = greedy pick
// for the model state AFTER token i (i.e. the model's prediction for candidate[i+1]).
// out_fin (optional, [T*H] host f32) receives every row's post-final-norm hidden — the
// next-step recurrent h is one of these rows. Bit-identical to decoding the candidates
// one at a time and reading each step's argmax.
static int e4b_step_verify(e4b_engine* eng, Slot& slot, const int32_t* tokens, int T,
                           int32_t* out_argmax, float* out_fin){
    if(!eng || T<=0) return -1;
    return e4b_step(eng, slot, tokens, T, nullptr, nullptr, out_fin, nullptr, out_argmax);
}

// ── Increment 4: greedy speculative decode (MTP draft head) ──────────────────
// Lossless: byte-identical to e4b_engine_generate_greedy. Prefill → first greedy token g
// + recurrent h0 (last prompt row's post-final-norm hidden); then loop: draft up to K tokens
// by chaining the assistant head, build candidate=[g,d0..d_{K-1}], verify all K+1 rows in one
// target forward, accept the longest prefix whose row-argmax matches the next candidate,
// emit g + accepted drafts, rewind KV to drop the unverified tail, refresh h, set the next g.
extern "C" int e4b_engine_generate_spec_greedy(e4b_engine_t *eng, const int32_t *prompt, int n_prompt,
                                               int32_t *out_tokens, int max_new,
                                               const int32_t *stop_ids, int n_stop){
    if(!eng || n_prompt<=0) return -1;
    // No assistant → fall back to plain greedy (still lossless, just no speedup).
    if(!eng->mtp.loaded)
        return e4b_engine_generate_greedy(eng, prompt, n_prompt, out_tokens, max_new, stop_ids, n_stop);

    cudaSetDevice(eng->device_id);
    const int V = eng->cfg.vocab_size, H = eng->cfg.hidden_size;
    int K = 4;
    if (const char* e=getenv("FUCINA_E4B_DRAFT_K")){ int k=atoi(e); if(k>=1 && k<=16) K=k; }

    auto is_stop=[&](int32_t t){ for(int i=0;i<n_stop;i++) if(stop_ids[i]==t) return true; return false; };
    auto argmaxv=[&](const std::vector<float>& x){ int b=0; for(int i=1;i<V;i++) if(x[i]>x[b]) b=i; return b; };

    // Prefill: drive e4b_step in ring-safe chunks (like e4b_append_slot), capturing the
    // LAST chunk's post-final-norm hidden for the recurrent h0 and the last token's logits.
    eng->slots[0].n_past = 0;
    Slot& slot = eng->slots[0];
    std::vector<float> logits(V);
    std::vector<float> h0(H);
    {
        const int chunk0 = eng->sliding_cap - eng->cfg.sliding_window;
        const bool nochunk = (eng->sliding_cap >= eng->max_ctx) || chunk0 < 1 || n_prompt <= chunk0;
        const int chunk = nochunk ? n_prompt : chunk0;
        for (int off=0; off<n_prompt; off+=chunk){
            const int t = (n_prompt-off<chunk)?(n_prompt-off):chunk;
            const bool last = (off+t>=n_prompt);
            std::vector<float> fin; if(last) fin.resize((size_t)t*H);
            int rc = e4b_step(eng, slot, prompt+off, t, nullptr,nullptr,
                              last?fin.data():nullptr, last?logits.data():nullptr);
            if(rc!=0) return -1;
            if(last){ std::copy(fin.begin()+(size_t)(t-1)*H, fin.begin()+(size_t)t*H, h0.begin()); }
        }
    }

    int n=0;                          // tokens emitted
    int32_t g = argmaxv(logits);      // committed greedy token to emit next
    std::vector<int32_t> drafts(K), cand(K+1), vargmax(K+1);
    std::vector<float> vfin;          // [ (K+1)*H ] verify post-final-norm rows
    std::vector<float> d_h = h0;      // recurrent h fed to the drafter (host mirror)

    if(!e4b_mtp_scratch_ensure(eng)) return -1;
    const bool dbg = []{ const char* e=getenv("FUCINA_E4B_SPEC_DEBUG"); return e && e[0]=='1'; }();
    long total_accepted=0, total_drafted=0, rounds=0;
    double t_draft=0, t_verify=0;
    auto NOW=[]{ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; };

    while (n < max_new){
        if (is_stop(g)) break;
        const int n_before = slot.n_past;     // KV high-water before verifying the candidates
        double _td0 = dbg?NOW():0;

        // 1. draft up to K tokens by chaining the assistant head. Each draft is conditioned on
        //    the previous candidate token at the SAME absolute position the target will verify it.
        //    The drafter attends the TARGET KV [0,pos] inclusive. Token g (cur_tok at i=0) will
        //    occupy target slot n_before once committed, but its KV is NOT yet in the cache here,
        //    so the drafter attends the real context up to n_before-1: pos = n_before-1 + i (the
        //    last already-cached target position when predicting candidate[i+1]). h already carries
        //    g's information (it is g's post-final-norm hidden).
        // Chain up to K drafts on-GPU (graph replay; only d_pos changes per step). base_pos =
        // n_before-1; room = max_ctx - n_before (need n_before+1+i <= max_ctx to verify draft i).
        int D = e4b_mtp_draft(eng, slot, g, d_h.data(), n_before - 1, K, eng->max_ctx - n_before, drafts.data());
        if (D < 0) return -1;
        if (dbg) t_draft += NOW()-_td0;

        // 2. candidate = [g, d0 .. d_{D-1}]  (length D+1). Verify ALL rows in one target forward.
        cand[0]=g; for(int i=0;i<D;i++) cand[i+1]=drafts[i];
        const int Tc = D+1;
        if (n_before + Tc > eng->max_ctx){
            // not enough context to verify the full candidate — verify just g (no draft).
            cand.resize(1); cand[0]=g;
        }
        const int Tv = (int)((n_before + Tc > eng->max_ctx) ? 1 : Tc);
        vfin.resize((size_t)Tv*H);
        double _tv0 = dbg?NOW():0;
        if (e4b_step_verify(eng, slot, cand.data(), Tv, vargmax.data(), vfin.data())!=0){ return -1; }
        if (dbg) t_verify += NOW()-_tv0;
        // verify advanced n_past by Tv; remember to rewind to the accepted length.

        // 3. accept: a = longest prefix of drafts that the target agrees with.
        //    vargmax[i] is the target's greedy next-token after candidate[i]; a draft d_i
        //    (== cand[i+1]) is accepted iff vargmax[i] == cand[i+1].
        int a=0;
        const int Dv = Tv-1;          // drafts actually verified
        while (a < Dv && vargmax[a]==cand[a+1]) a++;
        eng->spec_steps++; eng->spec_drafted+=Dv; eng->spec_accepted+=a;
        if (dbg){ total_accepted+=a; total_drafted+=Dv; rounds++; }

        // 4. emit g + the `a` accepted drafts (= cand[0..a]); honor stop_ids + max_new.
        bool stopped=false;
        for (int i=0;i<=a;i++){
            int32_t tk = cand[i];
            if (is_stop(tk)){ stopped=true; break; }
            out_tokens[n++]=tk; eng->spec_emitted++;
            if (n>=max_new){ stopped=true; break; }
        }

        // 5. the next committed token g' = target's prediction after the last accepted token
        //    = vargmax[a]. Rewind KV to keep exactly the committed tokens (n_before + (a+1));
        //    those rows are real (verified) and remain in the cache. The drafter cache is the
        //    target cache (Q-only), so nothing extra to rewind.
        int32_t g_next = vargmax[a];
        int keep = n_before + (a+1);
        if (keep < slot.n_past){
            if (e4b_engine_rewind(eng, keep)!=1){ return -1; }
        }

        // 6. refresh recurrent h for the next draft round = post-final-norm hidden of the LAST
        //    committed token (row `a` of the verify forward).
        std::copy(vfin.begin()+(size_t)a*H, vfin.begin()+(size_t)(a+1)*H, d_h.begin());

        if (stopped) break;
        g = g_next;
    }

    if (dbg) fprintf(stderr,"e4b-spec timing: draft=%.3fs verify=%.3fs (%.0f%% verify)\n",
                     t_draft, t_verify, 100.0*t_verify/(t_draft+t_verify+1e-9));
    if (dbg) fprintf(stderr,"e4b-spec: %ld rounds, %ld/%ld drafts accepted (%.1f%%), mean accept=%.2f tok/round, %d emitted\n",
                     rounds, total_accepted, total_drafted,
                     total_drafted?100.0*total_accepted/total_drafted:0.0,
                     rounds?(double)(total_accepted+rounds)/rounds:0.0, n);
    return n;
}

// ── Increment 5: streaming greedy speculative decode (server path) ───────────
// CONTINUE variant of e4b_engine_generate_spec_greedy: instead of re-prefilling, it
// resumes from slot 0's CURRENT KV (the server has already prefilled `history`, whose
// length must equal n_past) and the last-token logits the server captured in
// `first_logits`. Emits every committed token through `cb(tok, ud)` IN ORDER between
// verify rounds; cb returning non-zero stops generation after that token. out_tokens
// receives ALL committed tokens (including any the callback declined to render and any
// accepted-but-unemitted tail of the final round) so the server can reconcile the prefix
// cache with the engine KV (engine n_past advances by exactly the committed count).
// Bit-identical to plain greedy. No assistant ⇒ falls back to a plain greedy decode loop
// driving the same callback. Returns tokens written (≥0), -1 on error.
typedef int (*e4b_spec_token_cb)(int32_t tok, void* ud);

extern "C" int e4b_engine_spec_stream(e4b_engine_t *eng,
                                      const int32_t *history, int n_hist,
                                      const float *first_logits,
                                      int32_t *out_tokens, int max_new,
                                      const int32_t *stop_ids, int n_stop,
                                      e4b_spec_token_cb cb, void *ud){
    if(!eng || max_new<=0 || !first_logits || !out_tokens) return -1;
    cudaSetDevice(eng->device_id);
    const int V = eng->cfg.vocab_size, H = eng->cfg.hidden_size;
    Slot& slot = eng->slots[0];

    auto is_stop=[&](int32_t t){ for(int i=0;i<n_stop;i++) if(stop_ids[i]==t) return true; return false; };
    auto argmaxv=[&](const float* x){ int b=0; for(int i=1;i<V;i++) if(x[i]>x[b]) b=i; return b; };

    int32_t g = argmaxv(first_logits);   // committed greedy token to emit next

    // No assistant → plain greedy decode loop, still driving the callback (lossless,
    // no speedup). Mirrors e4b_engine_generate_greedy but continues from the live KV.
    if(!eng->mtp.loaded){
        std::vector<float> logits(V);
        int n=0;
        while(n<max_new){
            if(is_stop(g)) break;
            out_tokens[n++]=g;
            if(cb && cb(g, ud)) break;
            if(n>=max_new) break;
            if(e4b_engine_decode(eng, g, logits.data())!=0) return -1;
            g = argmaxv(logits.data());
        }
        return n;
    }

    int K = 4;
    if (const char* e=getenv("FUCINA_E4B_DRAFT_K")){ int k=atoi(e); if(k>=1 && k<=16) K=k; }

    // Recover the recurrent h0 = post-final-norm hidden of the LAST history token (the
    // token that produced first_logits). The server didn't capture it, so re-derive it:
    // rewind one token then re-run that single token through the verify forward, which
    // re-writes its KV (restoring n_past) and returns its post-final-norm hidden in row 0.
    // (n_hist>=1 is guaranteed by the server: a prompt was prefilled before we run.)
    std::vector<float> h0(H);
    if(n_hist>=1 && slot.n_past>=1){
        const int32_t last = history[n_hist-1];
        if(e4b_engine_rewind(eng, slot.n_past-1)!=1) return -1;
        std::vector<float> fin1((size_t)H); int32_t am1=0;
        if(e4b_step_verify(eng, slot, &last, 1, &am1, fin1.data())!=0) return -1;
        std::copy(fin1.begin(), fin1.end(), h0.begin());
    }

    int n=0;                          // tokens committed (written to out_tokens)
    std::vector<int32_t> drafts(K), cand(K+1), vargmax(K+1);
    std::vector<float> vfin;          // [(K+1)*H] verify post-final-norm rows
    std::vector<float> d_h = h0;      // recurrent h fed to the drafter (host mirror)

    if(!e4b_mtp_scratch_ensure(eng)) return -1;
    bool emit_stop=false;
    while (n < max_new && !emit_stop){
        if (is_stop(g)) break;
        const int n_before = slot.n_past;

        // 1. draft up to K tokens on-GPU (see e4b_engine_generate_spec_greedy / e4b_mtp_draft).
        int D = e4b_mtp_draft(eng, slot, g, d_h.data(), n_before - 1, K, eng->max_ctx - n_before, drafts.data());
        if (D < 0) return -1;

        // 2. verify candidate=[g,d0..] in one target forward.
        cand[0]=g; for(int i=0;i<D;i++) cand[i+1]=drafts[i];
        const int Tc = D+1;
        const int Tv = (int)((n_before + Tc > eng->max_ctx) ? 1 : Tc);
        vfin.resize((size_t)Tv*H);
        if (e4b_step_verify(eng, slot, cand.data(), Tv, vargmax.data(), vfin.data())!=0){ return -1; }

        // 3. accept longest greedy-matching prefix.
        int a=0; const int Dv=Tv-1;
        while (a < Dv && vargmax[a]==cand[a+1]) a++;
        eng->spec_steps++; eng->spec_drafted+=Dv; eng->spec_accepted+=a;

        // 4. emit g + the `a` accepted drafts, honoring stop_ids / max_new / callback.
        bool stopped=false;
        for (int i=0;i<=a;i++){
            int32_t tk=cand[i];
            if (is_stop(tk)){ stopped=true; break; }
            out_tokens[n++]=tk; eng->spec_emitted++;
            if (cb && cb(tk, ud)){ emit_stop=true; }
            if (n>=max_new){ stopped=true; break; }
            if (emit_stop){ break; }   // callback asked to stop, but the run is committed in KV
        }

        // 5. next committed token + rewind KV to the committed length.
        int32_t g_next = vargmax[a];
        int keep = n_before + (a+1);
        if (keep < slot.n_past){
            if (e4b_engine_rewind(eng, keep)!=1){ return -1; }
        }
        // 6. refresh recurrent h = post-final-norm hidden of the last committed token.
        std::copy(vfin.begin()+(size_t)a*H, vfin.begin()+(size_t)(a+1)*H, d_h.begin());

        if (stopped) break;
        g = g_next;
    }

    return n;
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
        const int cap = full ? eng->max_ctx : eng->sliding_cap;  // ring capacity (full ⇒ no wrap)
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
        rmsnorm_kernel<<<B,1024>>>(d_hidden,L.input_ln,d_norm,B,H,eps);
        bproj(d_q,L.wq,L.q40_wq,d_norm,qd,H);
        head_rmsnorm_kernel<<<B*c.n_heads,256>>>(d_q,L.q_norm,B,c.n_heads,hd,eps);
        rope_batch_kernel<<<B*c.n_heads,256>>>(d_q,d_invf,B,c.n_heads,hd,d_pos);
        if (!shared){
            bproj(d_k,L.wk,L.q40_wk,d_norm,kvd,H);
            bproj(d_v,L.wv,L.q40_wv,d_norm,kvd,H);
            head_rmsnorm_kernel<<<B*c.n_kv_heads,256>>>(d_k,L.k_norm,B,c.n_kv_heads,hd,eps);
            rope_batch_kernel<<<B*c.n_kv_heads,256>>>(d_k,d_invf,B,c.n_kv_heads,hd,d_pos);
            head_rmsnorm_kernel<<<B*c.n_kv_heads,256>>>(d_v,nullptr,B,c.n_kv_heads,hd,eps);
            kv_store_batch_kernel<<<GRID(B*kvd),256>>>(d_k,d_kptr,d_pos,B,kvd,cap);
            kv_store_batch_kernel<<<GRID(B*kvd),256>>>(d_v,d_vptr,d_pos,B,kvd,cap);
        }
        dim3 ag(c.n_heads,B);
        const int span = (window>0 && window<(maxP+1)) ? window : (maxP+1);
        size_t sh=(size_t)span*sizeof(float);
        attn_batch_kernel<<<ag,256,sh>>>(d_q,d_kptr,d_vptr,d_pos,d_attn,B,c.n_heads,c.n_kv_heads,hd,1.0f,window,cap);
        bproj(d_norm,L.wo,L.q40_wo,d_attn,H,qd);
        rmsnorm_kernel<<<B,1024>>>(d_norm,L.post_attn_ln,d_norm,B,H,eps);
        add_kernel<<<GRID(B*H),256>>>(d_tmpH,d_norm,B*H);
        cudaMemcpy(d_hidden,d_tmpH,(size_t)B*H*2,cudaMemcpyDeviceToDevice);

        cudaMemcpy(d_tmpH,d_hidden,(size_t)B*H*2,cudaMemcpyDeviceToDevice);
        rmsnorm_kernel<<<B,1024>>>(d_hidden,L.pre_ff_ln,d_norm,B,H,eps);
        bproj(d_gate,L.w_gate,L.q40_gate,d_norm,FF,H);
        bproj(d_up,L.w_up,L.q40_up,d_norm,FF,H);
        geglu_kernel<<<GRID(B*FF),256>>>(d_gate,d_up,d_act,B*FF);
        bproj(d_norm,L.w_down,L.q40_down,d_act,H,FF);
        rmsnorm_kernel<<<B,1024>>>(d_norm,L.post_ff_ln,d_norm,B,H,eps);
        add_kernel<<<GRID(B*H),256>>>(d_tmpH,d_norm,B*H);
        cudaMemcpy(d_hidden,d_tmpH,(size_t)B*H*2,cudaMemcpyDeviceToDevice);

        cudaMemcpy(d_tmpH,d_hidden,(size_t)B*H*2,cudaMemcpyDeviceToDevice);
        linear(eng->cublas,L.ple_in_gate,d_hidden,d_pleg,B,PD,H);
        ple_gate_strided<<<GRID(B*PD),256>>>(d_pleg,d_pli,B,PD,W,li);
        linear(eng->cublas,L.ple_proj,d_pleg,d_norm,B,H,PD);
        rmsnorm_kernel<<<B,1024>>>(d_norm,L.post_ple_ln,d_norm,B,H,eps);
        add_kernel<<<GRID(B*H),256>>>(d_tmpH,d_norm,B*H);
        scale_kernel<<<GRID(B*H),256>>>(d_tmpH,L.layer_scalar,B*H);
        cudaMemcpy(d_hidden,d_tmpH,(size_t)B*H*2,cudaMemcpyDeviceToDevice);
    }
    rmsnorm_kernel<<<B,1024>>>(d_hidden,eng->d_final_norm,d_norm,B,H,eps);
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
    s.active=true; s.n_past=0;                       // fresh sequence in this slot
    std::vector<float> logits(eng->cfg.vocab_size);
    if (e4b_append_slot(eng, s, prompt, n_prompt, logits.data())!=0){ s.active=false; return -1; }
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
