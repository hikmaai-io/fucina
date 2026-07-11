// Qwen3.5 FP8/NVFP4 loaders, MTP and MoE backend implementations.
// Internal implementation fragment: included once by gemma4_kernels.cu.
#pragma once

// =========================================================================
// ─── M5: Qwen3.5 FP8 block-quant safetensors loader + forward ────────────
// =========================================================================
// Loads the OFFICIAL Qwen/Qwen3.5-9B-FP8 checkpoint (DeepSeek-V3 block-fp8 on the bulk Linears)
// and runs the same hybrid forward as the M3 GGUF path, swapping the quantized dp4a GEMV for the
// validated fp8_block decode GEMV (cuda/fp8_block.cuh). This path is fully self-contained (its own
// device buffers + q35fp8_model handle); it never touches gemma4_engine, so Gemma/Qwen3/MoE and
// the qwen35 GGUF path stay byte-identical.
//
// Two HF-vs-GGUF conventions the GGUF conversion hid (the GGUF baked them in / pre-permuted), which
// MUST be applied to the RAW safetensors here — verified against transformers modeling_qwen3_5.py
// and the torch oracle cuda/qwen35_fp8_ref.py:
//   (1) Qwen3_5RMSNorm = _norm(x) * (1 + weight)  → bake +1 into every norm gain EXCEPT the gated
//       linear_attn.norm (Qwen3_5RMSNormGated, plain weight).
//   (2) GDN q/k (16 key heads) → 32 value heads expand by repeat_INTERLEAVE (kh = vh/(NVH/NKH)),
//       NOT the TILE (vh%NKH) the GGUF path uses on its pre-permuted v-heads.

// BF16 → F32 with an optional additive bias (add=1.0f bakes the RMSNorm +1; 0.0f otherwise).
__global__ void q35fp8_bf16_to_f32_kernel(float *dst, const __nv_bfloat16 *src, size_t n, float add) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]) + add;
}
__global__ void q35fp8_f32_bias_kernel(float *d, float add, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] += add;
}
// A_log (F32) → decay coefficient ssm_a = -exp(A_log), consumed directly by m2_decay_beta_kernel.
__global__ void q35fp8_neg_exp_kernel(float *a, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = -__expf(a[i]);
}
// GDN single-step delta-rule for the FP8 (HF-native) path: identical to qwen35_gdn_step_kernel but
// q/k expand to the v-heads by repeat_INTERLEAVE (kh = vh/(NVH/NKH)) — the HF convention for the
// un-permuted safetensors v-heads (the GGUF path tiles its pre-permuted heads instead).
__global__ void qwen35_fp8_gdn_step_kernel(float *core, const float *q, const float *k,
                                           const float *v, const float *g, const float *beta,
                                           float *S_io) {
    int vh  = blockIdx.x;
    int kh  = vh / (M2_NVH / M2_NKH);   // repeat_interleave: v-head vh ↔ k/q-head vh/(NVH/NKH)
    int tid = threadIdx.x;
    extern __shared__ float sm[];
    float *S   = sm;                    // [SD*SD] k-major: S[kd*SD+vd]
    float *kt  = S + M2_SD * M2_SD;     // [SD]
    float *qt  = kt + M2_SD;            // [SD]
    float *dlt = qt + M2_SD;            // [SD]
    float *Sg  = S_io + (size_t)vh * M2_SD * M2_SD;
    for (int idx = tid; idx < M2_SD * M2_SD; idx += blockDim.x) S[idx] = Sg[idx];
    __syncthreads();
    float scale = rsqrtf((float)M2_SD);
    if (tid < M2_SD) {
        kt[tid] = k[(size_t)kh * M2_SD + tid];
        qt[tid] = q[(size_t)kh * M2_SD + tid] * scale;
    }
    __syncthreads();
    float gt = __expf(g[vh]);
    float bt = beta[vh];
    for (int idx = tid; idx < M2_SD * M2_SD; idx += blockDim.x) S[idx] *= gt;
    __syncthreads();
    if (tid < M2_SD) {
        float acc = 0.f;
        for (int kd = 0; kd < M2_SD; kd++) acc += S[kd * M2_SD + tid] * kt[kd];
        float vv = v[(size_t)vh * M2_SD + tid];
        dlt[tid] = (vv - acc) * bt;
    }
    __syncthreads();
    for (int idx = tid; idx < M2_SD * M2_SD; idx += blockDim.x) {
        int kd = idx / M2_SD, vd = idx % M2_SD;
        S[idx] += kt[kd] * dlt[vd];
    }
    __syncthreads();
    if (tid < M2_SD) {
        float acc = 0.f;
        for (int kd = 0; kd < M2_SD; kd++) acc += S[kd * M2_SD + tid] * qt[kd];
        core[(size_t)vh * M2_SD + tid] = acc;
    }
    __syncthreads();
    for (int idx = tid; idx < M2_SD * M2_SD; idx += blockDim.x) Sg[idx] = S[idx];
}

// One FP8 block-quant projection: F8_E4M3 weight [out,in] + BF16 block scale [out/128,in/128].
struct q35fp8_proj { const uint8_t *w; const __nv_bfloat16 *s; int in_dim, out_dim; };
struct q35fp8_layer {
    int is_full;
    q35fp8_proj q, k, v, o;                 // FULL: gated-q / k / v / o
    float *q_norm, *k_norm;                 // FULL: per-head RMSNorm gains (+1 baked) [HEAD]
    q35fp8_proj inqkv, inz, out;            // GDN: in_proj_qkv / in_proj_z / out_proj
    float *ina, *inb;                       // GDN: in_proj_a / in_proj_b  f32 [TSR][H] (m2_gemm)
    float *conv1d;                          // GDN: depthwise conv f32 [CONVD][CK]
    float *a_coef;                          // GDN: -exp(A_log) f32 [TSR]
    float *dt_bias;                         // GDN: f32 [TSR]
    float *ssm_norm;                        // GDN: gated RMSNorm gain (NO +1) f32 [SD]
    float *in_norm, *post_norm;             // shared: pre-mixer / pre-FFN RMSNorm (+1 baked) [H]
    q35fp8_proj gate, up, down;             // shared: SwiGLU MLP
};
// M6: single-MTP draft head (mtp.* tensors; FP8 checkpoint only — the GGUF dropped them).
// One FULL-attention decoder layer reused verbatim from the backbone full-attn shape, plus the
// fc-fusion prologue (concat[norm(embed), norm(h_prev)] → fc) and a final mtp.norm before the
// (shared, untied) lm_head. h_prev = the main model's POST-final-norm hidden (vLLM convention).
struct q35fp8_mtp {
    int   loaded;
    float *pre_e, *pre_h;             // pre_fc_norm_embedding / _hidden  [H] (+1)
    float *fc;                        // fc.weight BF16→F32 [H, 2H] (m2_gemm: OUT=H, INN=2H)
    float *in_norm, *post_norm, *fnorm;  // input_layernorm / post_attention_layernorm / mtp.norm [H] (+1)
    q35fp8_proj q, k, v, o;           // self_attn (gated 2× q), FP8 block
    float *q_norm, *k_norm;           // per-head RMSNorm gains [HEAD] (+1)
    q35fp8_proj gate, up, down;       // SwiGLU MLP, FP8 block
};
struct q35fp8_model {
    gemma4_model_config_t cfg;
    cudaStream_t stream;
    int   n_layers, vocab, inter;
    float *embed;       // [vocab*H] f32
    float *out_norm;    // [H] f32 (+1 baked)
    float *lm_head;     // [vocab*H] f32
    float *d_logits;    // [vocab]
    q35fp8_layer L[GEMMA4_CAP_LAYERS];
    q35fp8_mtp   mtp;   // M6 draft head
};

static void *q35fp8_up_bytes(const void *host, size_t nbytes) {
    void *d = nullptr;
    if (cudaMalloc(&d, nbytes) != cudaSuccess) return nullptr;
    cudaMemcpy(d, host, nbytes, cudaMemcpyHostToDevice);
    return d;
}
// Tensor → freshly-allocated F32 device buffer (with optional bias, e.g. +1 for the Gemma-style
// norms). DTYPE-AWARE: these small params (norms, A_log, dt_bias, in_proj_a/b) ship as EITHER F32
// or BF16 depending on the checkpoint export — Qwen3.5-35B-A3B-FP8 stored them F32, the Qwen3.6
// repack stores them BF16. Read by the RECORDED dtype: a BF16 tensor copied as raw F32 bytes (the
// old assumption) halves the element count and silently corrupts (the qwen3.6 A_log/norm garbage).
static float *q35fp8_up_bf16_f32(const st::Tensor *t, float add) {
    if (!t) return nullptr;
    if (t->dtype == st::Dtype::F32) {
        size_t n = t->nbytes / 4;
        float *dst = (float *)q35fp8_up_bytes(t->data, t->nbytes);
        if (dst && add != 0.0f)
            q35fp8_f32_bias_kernel<<<(unsigned)((n + 255) / 256), 256>>>(dst, add, n);
        return dst;
    }
    size_t n = t->nbytes / 2;
    __nv_bfloat16 *tmp = (__nv_bfloat16 *)q35fp8_up_bytes(t->data, t->nbytes);
    if (!tmp) return nullptr;
    float *dst = nullptr;
    if (cudaMalloc(&dst, n * sizeof(float)) != cudaSuccess) { cudaFree(tmp); return nullptr; }
    q35fp8_bf16_to_f32_kernel<<<(unsigned)((n + 255) / 256), 256>>>(dst, tmp, n, add);
    cudaFree(tmp);
    return dst;
}
// Dtype-aware F32 upcast (no bias): F32 → copy, BF16 → convert. Was "copy raw bytes as F32",
// which corrupted BF16 A_log / norm.weight in the Qwen3.6 checkpoints.
static float *q35fp8_up_f32(const st::Tensor *t) {
    return q35fp8_up_bf16_f32(t, 0.0f);
}
static bool q35fp8_load_proj(const st::Model &M, const std::string &key,
                             q35fp8_proj &P, int out_dim, int in_dim) {
    const st::Tensor *w = M.find(key);
    const st::Tensor *s = M.find(key + "_scale_inv");
    if (!w || !s) { fprintf(stderr, "qwen35_fp8_load: missing %s (+_scale_inv)\n", key.c_str()); return false; }
    P.w = (const uint8_t *)q35fp8_up_bytes(w->data, w->nbytes);
    P.s = (const __nv_bfloat16 *)q35fp8_up_bytes(s->data, s->nbytes);
    P.in_dim = in_dim; P.out_dim = out_dim;
    return P.w && P.s;
}

// ── Serve the block-FP8 checkpoint through the REAL gemma4_engine_t (batched decode+prefill+graph)
//    instead of the B=1 qwen35_fp8_forward_greedy oracle. Fills eng->d_weights + eng->tensors with
//    byte offsets and FORMAT_FP8_BLOCK so the existing qwen35_decode_multiseq_body serves it via
//    Wq(off)/gemv_batched_w; the per-128 block scale comes from eng->q35.fp8_scale_tab (wscale_fp8). ──
static inline float q35_bf16_to_f32_host(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }

// BF16 tensor → Q8_0 (host): per-32-block [fp16 scale][32 int8]. n_elem % 32 == 0.
static unsigned char *q35_bf16_to_q8_0_host(const uint16_t *src, int64_t n_elem){
    int64_t nblk = n_elem/32;
    unsigned char *dst = (unsigned char*)malloc((size_t)nblk*34);
    if(!dst) return nullptr;
    for(int64_t b=0;b<nblk;b++){
        float f[32], amax=0.0f;
        for(int j=0;j<32;j++){ f[j]=q35_bf16_to_f32_host(src[b*32+j]); amax=fmaxf(amax,fabsf(f[j])); }
        float scale=amax/127.0f, iscale=(scale>0.0f)?1.0f/scale:0.0f;
        unsigned char *ob=dst+(size_t)b*34;
        uint16_t hb=f2h_host(scale); ob[0]=(unsigned char)(hb&0xFF); ob[1]=(unsigned char)(hb>>8);
        for(int j=0;j<32;j++){ int q=(int)lrintf(f[j]*iscale); q=q<-127?-127:(q>127?127:q); ob[2+j]=(unsigned char)(int8_t)q; }
    }
    return dst;
}

// float[256·n_super] → Q4_K superblocks (ggml block_q4_K, 144 B: fp16 d,dmin + 12 B packed
// 6-bit (scale,min)×8 + 128 B nibbles). Exact inverse of q4k_scale_min + the dequant kernels'
// element order. Simple one-pass per-sub-block scale/min fit (llama.cpp's make_qkx2 iterative
// refinement would squeeze ~1% more SQNR; acceptable for the FP8→Q4_K requant path whose gate
// is a greedy-match quality test, not bitwise parity).
static void q35_f32_to_q4_K_host(const float *src, unsigned char *dst, int64_t n_super) {
    for (int64_t sb = 0; sb < n_super; sb++) {
        const float *w = src + sb * 256;
        unsigned char *blk = dst + sb * 144;
        float scales[8], mins[8];
        for (int j = 0; j < 8; j++) {
            const float *v = w + j * 32;
            float lo = v[0], hi = v[0];
            for (int k = 1; k < 32; k++) { lo = fminf(lo, v[k]); hi = fmaxf(hi, v[k]); }
            float mn = fmaxf(0.0f, -lo);              // dequant is d·s·q − dmin·m with m ≥ 0
            float sc = (hi + mn) / 15.0f;
            scales[j] = fmaxf(sc, 0.0f); mins[j] = mn;
        }
        float dmax = 0.0f, mmax = 0.0f;
        for (int j = 0; j < 8; j++) { dmax = fmaxf(dmax, scales[j]); mmax = fmaxf(mmax, mins[j]); }
        float d = dmax / 63.0f, dm = mmax / 63.0f;
        float id = d > 0 ? 1.0f / d : 0.0f, im = dm > 0 ? 1.0f / dm : 0.0f;
        uint8_t ls6[8], lm6[8];
        for (int j = 0; j < 8; j++) {
            int ls = (int)lrintf(scales[j] * id); ls6[j] = (uint8_t)(ls < 0 ? 0 : ls > 63 ? 63 : ls);
            int lm = (int)lrintf(mins[j]  * im); lm6[j] = (uint8_t)(lm < 0 ? 0 : lm > 63 ? 63 : lm);
        }
        uint16_t hd = f2h_host(d), hm = f2h_host(dm);
        blk[0] = (uint8_t)(hd & 0xFF); blk[1] = (uint8_t)(hd >> 8);
        blk[2] = (uint8_t)(hm & 0xFF); blk[3] = (uint8_t)(hm >> 8);
        uint8_t *sc8 = blk + 4;
        for (int j = 0; j < 12; j++) sc8[j] = 0;
        for (int j = 0; j < 8; j++) {              // llama.cpp packing (inverse of get_scale_min_k4)
            if (j < 4) { sc8[j] |= ls6[j]; sc8[j + 4] |= lm6[j]; }
            else {
                sc8[j + 4]  = (uint8_t)((ls6[j] & 0xF) | ((lm6[j] & 0xF) << 4));
                sc8[j - 4] |= (uint8_t)((ls6[j] >> 4) << 6);
                sc8[j]     |= (uint8_t)((lm6[j] >> 4) << 6);
            }
        }
        uint8_t *qs = blk + 16;
        for (int g = 0; g < 4; g++) {              // byte m of group g: sub 2g low nibble, 2g+1 high
            for (int m = 0; m < 32; m++) {
                int qlo, qhi;
                {
                    int j = 2 * g; float ds = d * (float)ls6[j], dmn = dm * (float)lm6[j];
                    float x = w[j * 32 + m];
                    qlo = ds > 0 ? (int)lrintf((x + dmn) / ds) : 0;
                    qlo = qlo < 0 ? 0 : qlo > 15 ? 15 : qlo;
                }
                {
                    int j = 2 * g + 1; float ds = d * (float)ls6[j], dmn = dm * (float)lm6[j];
                    float x = w[j * 32 + m];
                    qhi = ds > 0 ? (int)lrintf((x + dmn) / ds) : 0;
                    qhi = qhi < 0 ? 0 : qhi > 15 ? 15 : qhi;
                }
                qs[g * 32 + m] = (uint8_t)(qlo | (qhi << 4));
            }
        }
    }
}

// Host FP8-E4M3 → float (1s4e3m, bias 7, no inf; S.1111.111 = NaN), via a 256-entry LUT.
static float q35_fp8_lut_val(int v) {
    int sg = v >> 7, e = (v >> 3) & 0xF, m = v & 7;
    float sign = sg ? -1.0f : 1.0f;
    if (e == 0) return sign * (m / 8.0f) * (1.0f / 64.0f);
    if (e == 15 && m == 7) return 0.0f;   // NaN encoding never appears in real weights
    return sign * (1.0f + m / 8.0f) * exp2f((float)(e - 7));
}

// ── ModelOpt (nvidia NVFP4 repack) host-side conversion helpers ──────────────────────────────
// The nvidia Qwen3.6 checkpoints store attn/GDN projections as PER-TENSOR FP8 (F32 scalar scale)
// and experts/shared-expert/lm_head as native NVFP4 (E4M3 per-16-group scales + F32 global).
// These map every such tensor onto a representation the FP8-block engine already serves.

static inline uint16_t q35_f32_to_bf16_host(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);           // round-to-nearest-even
    return (uint16_t)(u >> 16);
}

// float → FP8 E4M3 (e4m3fn: bias 7, no inf, max 448), round-to-nearest.
static uint8_t q35_f32_to_e4m3_host(float f) {
    if (f != f) return 0x7F;
    uint8_t sign = f < 0 ? 0x80 : 0;
    float a = fabsf(f);
    if (a > 448.f) a = 448.f;
    if (a < 0.87890625e-3f) {                 // < half the smallest subnormal (2^-9) → ±0
        // fallthrough: subnormal rounding below handles [2^-10, 2^-6); exact 0 here
        if (a < 0.9765625e-3f) return sign;   // 2^-10
    }
    int e; float m = frexpf(a, &e);           // a = m·2^e, m ∈ [0.5,1)
    e -= 1; m *= 2.f;                         // m ∈ [1,2)
    if (e < -6) {                             // subnormal: q = round(a·2^9), value q/8·2^-6
        int q = (int)lrintf(a * 512.f);
        if (q > 7) q = 7;
        return sign | (uint8_t)q;
    }
    int mant = (int)lrintf((m - 1.f) * 8.f);
    if (mant == 8) { mant = 0; e++; }
    if (e > 8 || (e == 8 && mant == 7)) { e = 8; mant = 6; }   // clamp to 448, avoid NaN code
    return sign | (uint8_t)((e + 7) << 3) | (uint8_t)mant;
}

// Per-tensor FP8 → the engine's 128×128 BF16 block-scale grid: broadcast the F32 scalar.
// (BF16 rounding of the scale is ≤0.2% — negligible vs the e4m3 weight quantization itself.)
static uint16_t *q35_synth_scale_grid(float s, int out_dim, int in_dim) {
    size_t n = (size_t)((out_dim + 127) / 128) * (size_t)((in_dim + 127) / 128);
    uint16_t *g = (uint16_t *)malloc(n * 2);
    if (!g) return nullptr;
    uint16_t b = q35_f32_to_bf16_host(s);
    for (size_t i = 0; i < n; i++) g[i] = b;
    return g;
}

// Native NVFP4 tensor (packed U8 [out][in/2] + E4M3 [out][in/16] + F32 global) → host BF16
// [out][in]. Threaded: the lm_head is 248320×2048 (508M elements).
static uint16_t *q35_nvfp4_to_bf16_host(const uint8_t *w, const uint8_t *sc, float gs,
                                        int out_dim, int in_dim) {
    uint16_t *dst = (uint16_t *)malloc((size_t)out_dim * in_dim * 2);
    if (!dst) return nullptr;
    const int ng = in_dim / 16;
    std::atomic<int> next(0);
    auto worker = [&]() {
        for (;;) {
            int r = next.fetch_add(1);
            if (r >= out_dim) break;
            const uint8_t *wr = w + (size_t)r * (in_dim / 2);
            const uint8_t *sr = sc + (size_t)r * ng;
            uint16_t *o = dst + (size_t)r * in_dim;
            for (int c = 0; c < in_dim; c++)
                o[c] = q35_f32_to_bf16_host(
                    nvfp4::e2m1_decode(nvfp4::nibble_at(wr, c)) *
                    nvfp4::e4m3_decode(sr[c / 16]) * gs);
        }
    };
    std::vector<std::thread> th;
    for (int t = 0; t < 12; t++) th.emplace_back(worker);
    for (auto &x : th) x.join();
    return dst;
}

// E4M3 weight + scalar or per-output-channel multiplier → host BF16. Unsloth compressed-
// tensors uses BF16 [out,1] scales, while ModelOpt uses one F32 scalar. Treating the former as
// a scalar silently produces fluent-looking repeated-token corruption, so shape/dtype handling
// is deliberately centralized here.
static uint16_t *q35_fp8_scaled_to_bf16_host(const uint8_t *w, const st::Tensor *scale,
                                             int out_dim, int in_dim) {
    if(!scale) return nullptr;
    const bool scalar=scale->nbytes==4;
    const bool row_bf16=scale->dtype==st::Dtype::BF16 && scale->nbytes==(size_t)out_dim*2;
    const bool row_f32=scale->dtype==st::Dtype::F32 && scale->nbytes==(size_t)out_dim*4;
    if(!scalar&&!row_bf16&&!row_f32) return nullptr;
    uint16_t *dst=(uint16_t*)malloc((size_t)out_dim*in_dim*2);
    if(!dst) return nullptr;
    static float lut[256]; static bool ready=false;
    if(!ready){ for(int i=0;i<256;i++) lut[i]=q35_fp8_lut_val(i); ready=true; }
    std::atomic<int> next(0);
    auto worker=[&](){ for(;;){
        int r=next.fetch_add(1); if(r>=out_dim) break;
        float mul=scalar?*(const float*)scale->data:
                  (row_bf16?q35_bf16_to_f32_host(((const uint16_t*)scale->data)[r]):
                            ((const float*)scale->data)[r]);
        const uint8_t *src=w+(size_t)r*in_dim; uint16_t *out=dst+(size_t)r*in_dim;
        for(int c=0;c<in_dim;c++) out[c]=q35_f32_to_bf16_host(lut[src[c]]*mul);
    }};
    std::vector<std::thread> th; for(int t=0;t<12;t++) th.emplace_back(worker);
    for(auto &x:th) x.join();
    return dst;
}

// Host BF16 [out][in] → FP8-block (E4M3 bytes + per-128×128 BF16 scale grid, scale = amax/448),
// the exact layout the FORMAT_FP8_BLOCK kernels consume. Used for the ModelOpt shared expert
// (native NVFP4 with no direct engine consumer — the extra e4m3 rounding on top of the 4-bit
// values is ~3%, small vs the FP4 quantization). Returns malloc'd (w8, grid) or false.
static bool q35_bf16_to_fp8_block_host(const uint16_t *src, int out_dim, int in_dim,
                                       uint8_t **w_out, uint16_t **s_out) {
    const int nbr = (out_dim + 127) / 128, nbc = (in_dim + 127) / 128;
    uint8_t  *w8 = (uint8_t *)malloc((size_t)out_dim * in_dim);
    uint16_t *sg = (uint16_t *)malloc((size_t)nbr * nbc * 2);
    if (!w8 || !sg) { free(w8); free(sg); return false; }
    std::atomic<int> next(0);
    auto worker = [&]() {
        for (;;) {
            int br = next.fetch_add(1);
            if (br >= nbr) break;
            const int r0 = br * 128, r1 = r0 + 128 > out_dim ? out_dim : r0 + 128;
            for (int bc = 0; bc < nbc; bc++) {
                const int c0 = bc * 128, c1 = c0 + 128 > in_dim ? in_dim : c0 + 128;
                float amax = 0.f;
                for (int r = r0; r < r1; r++)
                    for (int c = c0; c < c1; c++) {
                        float v = q35_bf16_to_f32_host(src[(size_t)r * in_dim + c]);
                        amax = fmaxf(amax, fabsf(v));
                    }
                float scale = amax > 0.f ? amax / 448.f : 1.f;
                // round the scale to BF16 FIRST so encode uses exactly what dequant will read
                scale = q35_bf16_to_f32_host(q35_f32_to_bf16_host(scale));
                sg[(size_t)br * nbc + bc] = q35_f32_to_bf16_host(scale);
                float inv = 1.f / scale;
                for (int r = r0; r < r1; r++)
                    for (int c = c0; c < c1; c++) {
                        float v = q35_bf16_to_f32_host(src[(size_t)r * in_dim + c]);
                        w8[(size_t)r * in_dim + c] = q35_f32_to_e4m3_host(v * inv);
                    }
            }
        }
    };
    std::vector<std::thread> th;
    for (int t = 0; t < 12; t++) th.emplace_back(worker);
    for (auto &x : th) x.join();
    *w_out = w8; *s_out = sg;
    return true;
}

// One expert projection in the native-NVFP4 checkpoint: packed codes, linear e4m3 group scales,
// per-expert F32 global. Passed alongside (or instead of) the FP8 source arrays to
// q35_fp4_expert_slabs.
struct q35_nv4src { const uint8_t *w; const uint8_t *sc; float gs; };

// Build the NVFP4 expert slabs for ONE layer from the host FP8 checkpoint tensors: upload each
// proj's FP8+scales, dequant to BF16 on device, per-(layer,proj) global scale over ALL experts,
// then per-expert E2M1 quant + 128x4-swizzled ue4m3 SF in exactly the layout dg_fp4_moe_grouped
// consumes. gate|up are FUSED into one [2*MI][H] group per expert (gate rows 0..MI). 4.5 bpw —
// same bytes as the Q4_K requant it replaces, but tensor-core-consumable. Returns 0 / -1.
static int q35_fp4_expert_slabs(gemma4_engine_t *eng, int l, int L,
        const uint8_t **wg, const uint16_t **sg, const uint8_t **wu, const uint16_t **su,
        const uint8_t **wd, const uint16_t **sd,
        const q35_nv4src *ng, const q35_nv4src *nu, const q35_nv4src *nd,  // native NVFP4 source (or NULL)
        int E, int MI, int H) {
    const int GU = 2 * MI;
    const size_t wb = (size_t)MI * H;                    // FP8 bytes / expert (all 3 projs)
    const int sper = ((MI + 127) / 128) * (H / 128);     // BF16 scale elems / expert (all projs)
    const bool nv4 = (ng != NULL);
    if (!eng->d_fp4m_gsw &&
        cudaMalloc(&eng->d_fp4m_gsw, (size_t)L * 2 * sizeof(float)) != cudaSuccess) return -1;
    { unsigned long long a, b;
      dg_fp4_sf_strides(0, GU, H, &a, &b); eng->fp4m_gu_sfB = b;
      dg_fp4_sf_strides(0, H, MI, &a, &b); eng->fp4m_dn_sfB = b; }
    uint8_t *d8 = NULL, *tlin = NULL, *d_s4 = NULL;
    __nv_bfloat16 *dsc = NULL, *gbf = NULL, *ubf = NULL;
    float *amax = NULL, *d_gs = NULL;
    int ok = 1;
    // FP8 source: full-byte weights + tiny 128×128 scale grids. NVFP4 source: half-byte codes
    // (reuse d8, first wb/2) + per-16 e4m3 scales (wb/16) + per-expert F32 globals.
    ok &= cudaMalloc(&d8,   (size_t)E * (nv4 ? wb / 2 : wb)) == cudaSuccess;
    if (nv4) {
        ok &= cudaMalloc(&d_s4, (size_t)E * (wb / NVFP4_BLK)) == cudaSuccess;
        ok &= cudaMalloc(&d_gs, (size_t)E * sizeof(float)) == cudaSuccess;
    } else {
        ok &= cudaMalloc(&dsc, (size_t)E * sper * sizeof(__nv_bfloat16)) == cudaSuccess;
    }
    ok &= cudaMalloc(&gbf,  (size_t)E * wb * sizeof(__nv_bfloat16)) == cudaSuccess;
    ok &= cudaMalloc(&ubf,  (size_t)E * wb * sizeof(__nv_bfloat16)) == cudaSuccess;
    ok &= cudaMalloc(&tlin, (size_t)GU * (H / NVFP4_BLK)) == cudaSuccess;  // ≥ H*(MI/16) too
    ok &= cudaMalloc(&amax, sizeof(float)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_fp4m_gu[l],   (size_t)E * GU * (H / 2)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_fp4m_gusf[l], (size_t)E * eng->fp4m_gu_sfB) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_fp4m_dn[l],   (size_t)E * H * (MI / 2)) == cudaSuccess;
    ok &= cudaMalloc(&eng->d_fp4m_dnsf[l], (size_t)E * eng->fp4m_dn_sfB) == cudaSuccess;
    if (ok) {
        cudaMemset(eng->d_fp4m_gusf[l], 0, (size_t)E * eng->fp4m_gu_sfB);
        cudaMemset(eng->d_fp4m_dnsf[l], 0, (size_t)E * eng->fp4m_dn_sfB);
    }
    // upload one proj's E experts + scales, dequant the whole slab to BF16
    auto dq = [&](const uint8_t **w, const uint16_t **s, int ind, int outd, __nv_bfloat16 *dst) {
        for (int e = 0; e < E && ok; e++) {
            ok &= cudaMemcpy(d8 + (size_t)e * wb, w[e], wb, cudaMemcpyHostToDevice) == cudaSuccess;
            ok &= cudaMemcpy((uint8_t*)dsc + (size_t)e * sper * 2, s[e], (size_t)sper * 2,
                             cudaMemcpyHostToDevice) == cudaSuccess;
        }
        uint64_t n = (uint64_t)E * wb;
        if (ok) dequant_fp8_expert_slab_bf16_kernel<<<(unsigned)((n + 255) / 256), 256>>>(
                    dst, d8, dsc, ind, outd, n);
    };
    // same, from the native NVFP4 checkpoint tensors (codes + linear e4m3 scales + per-expert gs)
    auto dq4 = [&](const q35_nv4src *s4, int ind, int outd, __nv_bfloat16 *dst) {
        std::vector<float> gs(E);
        for (int e = 0; e < E && ok; e++) {
            ok &= cudaMemcpy(d8 + (size_t)e * (wb / 2), s4[e].w, wb / 2,
                             cudaMemcpyHostToDevice) == cudaSuccess;
            ok &= cudaMemcpy(d_s4 + (size_t)e * (wb / NVFP4_BLK), s4[e].sc, wb / NVFP4_BLK,
                             cudaMemcpyHostToDevice) == cudaSuccess;
            gs[e] = s4[e].gs;
        }
        ok &= cudaMemcpy(d_gs, gs.data(), (size_t)E * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess;
        uint64_t n = (uint64_t)E * wb;
        if (ok) dequant_nvfp4ckpt_expert_slab_bf16_kernel<<<(unsigned)((n / NVFP4_BLK + 255) / 256), 256>>>(
                    dst, d8, d_s4, d_gs, ind, outd, n);
    };
    if (ok) {   // ── fused gate|up: global scale over BOTH tensors, one SF block per expert ──
        if (nv4) { dq4(ng, H, MI, gbf); dq4(nu, H, MI, ubf); }
        else     { dq(wg, sg, H, MI, gbf); dq(wu, su, H, MI, ubf); }
        cudaMemset(amax, 0, sizeof(float));
        const uint64_t n = (uint64_t)E * wb;
        nvfp4_amax_bf16_kernel<<<(unsigned)((n + 255) / 256), 256>>>(gbf, n, amax);
        nvfp4_amax_bf16_kernel<<<(unsigned)((n + 255) / 256), 256>>>(ubf, n, amax);
        nvfp4_gs_kernel<<<1, 1>>>(amax, eng->d_fp4m_gsw + 2 * l);
        const int nb = H / NVFP4_BLK, nbp = nvfp4_pad(nb, 4);
        for (int e = 0; e < E; e++) {
            dim3 b(256), g((nb + 255) / 256, MI);
            uint8_t *fp4 = eng->d_fp4m_gu[l] + (size_t)e * GU * (H / 2);
            nvfp4_quant_bf16_kernel<<<g, b>>>(gbf + (size_t)e * wb, fp4, tlin,
                                              MI, H, eng->d_fp4m_gsw + 2 * l);
            nvfp4_quant_bf16_kernel<<<g, b>>>(ubf + (size_t)e * wb, fp4 + (size_t)MI * (H / 2),
                                              tlin + (size_t)MI * nb, MI, H, eng->d_fp4m_gsw + 2 * l);
            dim3 b2(32, 8), g2((nb + 31) / 32, (GU + 7) / 8);
            nvfp4_swizzle_kernel<<<g2, b2>>>(tlin, eng->d_fp4m_gusf[l] + (size_t)e * eng->fp4m_gu_sfB,
                                             GU, nb, nbp);
        }
    }
    if (ok) {   // ── down [H][MI] ──
        if (nv4) dq4(nd, MI, H, gbf);
        else     dq(wd, sd, MI, H, gbf);
        cudaMemset(amax, 0, sizeof(float));
        const uint64_t n = (uint64_t)E * wb;
        nvfp4_amax_bf16_kernel<<<(unsigned)((n + 255) / 256), 256>>>(gbf, n, amax);
        nvfp4_gs_kernel<<<1, 1>>>(amax, eng->d_fp4m_gsw + 2 * l + 1);
        const int nb = MI / NVFP4_BLK, nbp = nvfp4_pad(nb, 4);
        for (int e = 0; e < E; e++) {
            dim3 b(256), g((nb + 255) / 256, H);
            nvfp4_quant_bf16_kernel<<<g, b>>>(gbf + (size_t)e * wb,
                                              eng->d_fp4m_dn[l] + (size_t)e * H * (MI / 2),
                                              tlin, H, MI, eng->d_fp4m_gsw + 2 * l + 1);
            dim3 b2(32, 8), g2((nb + 31) / 32, (H + 7) / 8);
            nvfp4_swizzle_kernel<<<g2, b2>>>(tlin, eng->d_fp4m_dnsf[l] + (size_t)e * eng->fp4m_dn_sfB,
                                             H, nb, nbp);
        }
    }
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) ok = 0;
    if (d8) cudaFree(d8);   if (dsc) cudaFree(dsc); if (gbf) cudaFree(gbf);
    if (ubf) cudaFree(ubf); if (tlin) cudaFree(tlin); if (amax) cudaFree(amax);
    if (d_s4) cudaFree(d_s4); if (d_gs) cudaFree(d_gs);
    if (!ok) {
        auto FR = [](uint8_t *&p){ if (p) { cudaFree(p); p = NULL; } };
        FR(eng->d_fp4m_gu[l]); FR(eng->d_fp4m_gusf[l]); FR(eng->d_fp4m_dn[l]); FR(eng->d_fp4m_dnsf[l]);
    }
    return ok ? 0 : -1;
}

struct q35_resident_seed { int l,e; double score; };
static int q35_seed_ssd_residency(gemma4_engine_t *eng,const char *plan,int L,int E,int MI,int H) {
    if(!plan||!*plan)return 0;FILE*f=fopen(plan,"rb");if(!f){fprintf(stderr,"fucina: cannot open residency plan %s\n",plan);return -1;}
    fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);std::string raw;
    if(n<=0||n>(64L<<20)){fclose(f);return -1;}raw.resize((size_t)n);bool ok=fread(raw.data(),1,(size_t)n,f)==(size_t)n;fclose(f);
    if(!ok||raw.find("fucina-expert-residency-v1")==std::string::npos){fprintf(stderr,"fucina: invalid residency plan\n");return -1;}
    std::vector<q35_resident_seed> seeds;size_t p=0;
    while((p=raw.find("\"layers.",p))!=std::string::npos){int l=-1,e=-1;if(sscanf(raw.c_str()+p+1,"layers.%d.experts.%d",&l,&e)!=2){p+=8;continue;}
        size_t end=raw.find('}',p);if(end==std::string::npos)break;size_t tier=raw.find("\"tier\"",p);
        if(tier<end&&raw.find("\"vram\"",tier)<end){double score=0;size_t ip=raw.find("\"importance\"",p);
            if(ip<end){ip=raw.find(':',ip);if(ip<end)score=strtod(raw.c_str()+ip+1,nullptr);}if(l>=0&&l<L&&e>=0&&e<E)seeds.push_back({l,e,score});}p=end+1;}
    std::sort(seeds.begin(),seeds.end(),[](const q35_resident_seed&a,const q35_resident_seed&b){if(a.score!=b.score)return a.score>b.score;if(a.l!=b.l)return a.l<b.l;return a.e<b.e;});
    const size_t gp=(size_t)2*MI*(H/2),gsp=(size_t)eng->fp4m_gu_sfB,dp=(size_t)H*(MI/2),dsp=(size_t)eng->fp4m_dn_sfB;
    auto rd=[&](void*x,size_t z,int64_t at){uint8_t*b=(uint8_t*)x;size_t d=0;while(d<z){ssize_t r=pread(eng->fp4m_ssd_fd,b+d,z-d,at+(int64_t)d);if(r<=0)return false;d+=(size_t)r;}return true;};
    int loaded=0;for(const auto&s:seeds){if(loaded>=eng->fp4m_slots)break;int slot=loaded;
        uint8_t*hg=eng->h_fp4m_stage_gu+(size_t)slot*gp,*hs=eng->h_fp4m_stage_gusf+(size_t)slot*gsp;
        uint8_t*hd=eng->h_fp4m_stage_dn+(size_t)slot*dp,*hds=eng->h_fp4m_stage_dnsf+(size_t)slot*dsp;
        if(!rd(hg,gp,eng->fp4m_ssd_gu_off[s.l]+(int64_t)s.e*gp)||!rd(hs,gsp,eng->fp4m_ssd_gusf_off[s.l]+(int64_t)s.e*gsp)
          ||!rd(hd,dp,eng->fp4m_ssd_dn_off[s.l]+(int64_t)s.e*dp)||!rd(hds,dsp,eng->fp4m_ssd_dnsf_off[s.l]+(int64_t)s.e*dsp))return -1;
        const void*pp[4]={hg,hs,hd,hds};const size_t nn[4]={gp,gsp,dp,dsp};size_t z=((size_t)s.l*E+s.e)*4;
        for(int q=0;q<4;q++){uint64_t h=q35_expert_hash_update(1469598103934665603ULL,pp[q],nn[q]);if(h!=eng->h_fp4m_ssd_hash[z+q])return -1;eng->h_fp4m_ssd_verified[z+q]=1;}
        cudaMemcpy(eng->d_fp4m_stage_gu+(size_t)slot*gp,hg,gp,cudaMemcpyHostToDevice);cudaMemcpy(eng->d_fp4m_stage_gusf+(size_t)slot*gsp,hs,gsp,cudaMemcpyHostToDevice);
        cudaMemcpy(eng->d_fp4m_stage_dn+(size_t)slot*dp,hd,dp,cudaMemcpyHostToDevice);cudaMemcpy(eng->d_fp4m_stage_dnsf+(size_t)slot*dsp,hds,dsp,cudaMemcpyHostToDevice);
        eng->h_fp4m_slot_layer[slot]=s.l;eng->h_fp4m_slot_expert[slot]=s.e;eng->h_fp4m_slot_age[slot]=++eng->fp4m_slot_clock;loaded++;}
    fprintf(stderr,"fucina: residency plan seeded %d/%d expert slots from %s\n",loaded,eng->fp4m_slots,plan);return 0;
}

// Persist transformed grouped-NVFP4 experts to SSD and retain only a compact LRU slot pool.
static int q35_enable_ssd_expert_stream(gemma4_engine_t *eng, int L, int E, int MI, int H) {
    const char *ssd=getenv("FUCINA_EXPERT_STREAM_SSD");
    if(!ssd || !ssd[0] || !eng->moe_experts_fp4) return 0;
    const size_t guB=(size_t)E*2*MI*(H/2), gusfB=(size_t)E*eng->fp4m_gu_sfB;
    const size_t dnB=(size_t)E*H*(MI/2), dnsfB=(size_t)E*eng->fp4m_dn_sfB;
    int slots=512;const char*s=getenv("FUCINA_EXPERT_STREAM_SLOTS");if(s)slots=atoi(s);if(slots<1)slots=1;if(slots>4096)slots=4096;
    eng->fp4m_slots=slots;
    eng->h_fp4m_slot_layer=(int*)malloc((size_t)slots*sizeof(int));
    eng->h_fp4m_slot_expert=(int*)malloc((size_t)slots*sizeof(int));
    eng->h_fp4m_slot_age=(uint64_t*)calloc((size_t)slots,sizeof(uint64_t));
    for(int s=0;s<slots;s++){eng->h_fp4m_slot_layer[s]=-1;eng->h_fp4m_slot_expert[s]=-1;}
    bool ok=eng->h_fp4m_slot_layer&&eng->h_fp4m_slot_expert&&eng->h_fp4m_slot_age
         && cudaMalloc(&eng->d_fp4m_stage_gu,guB/E*slots)==cudaSuccess
         && cudaMalloc(&eng->d_fp4m_stage_gusf,gusfB/E*slots)==cudaSuccess
         && cudaMalloc(&eng->d_fp4m_stage_dn,dnB/E*slots)==cudaSuccess
         && cudaMalloc(&eng->d_fp4m_stage_dnsf,dnsfB/E*slots)==cudaSuccess
         && cudaMalloc(&eng->d_fp4m_eslot,(size_t)E*sizeof(int))==cudaSuccess;
    if(ok) {
        eng->fp4m_ssd_fd=open(ssd,O_CREAT|O_TRUNC|O_RDWR,0600); ok=eng->fp4m_ssd_fd>=0;
        size_t hc=(size_t)(slots>E?slots:E);
        eng->h_fp4m_stage_gu=(uint8_t*)malloc(guB/E*hc);eng->h_fp4m_stage_gusf=(uint8_t*)malloc(gusfB/E*hc);
        eng->h_fp4m_stage_dn=(uint8_t*)malloc(dnB/E*hc);eng->h_fp4m_stage_dnsf=(uint8_t*)malloc(dnsfB/E*hc);
        eng->h_fp4m_ssd_hash=(uint64_t*)malloc((size_t)L*E*4*sizeof(uint64_t));
        eng->h_fp4m_ssd_verified=(uint8_t*)calloc((size_t)L*E*4,1);
        ok=ok&&eng->h_fp4m_stage_gu&&eng->h_fp4m_stage_gusf&&eng->h_fp4m_stage_dn&&eng->h_fp4m_stage_dnsf&&eng->h_fp4m_ssd_hash&&eng->h_fp4m_ssd_verified;
        const char hdr[16]="FUCINAEXPERT1\n"; if(ok) ok=pwrite(eng->fp4m_ssd_fd,hdr,sizeof(hdr),0)==(ssize_t)sizeof(hdr);
    }
    int64_t off=4096;
    auto wr=[&](const void*p,size_t n,int64_t at){ const uint8_t*b=(const uint8_t*)p;size_t d=0;
        while(d<n){ssize_t w=pwrite(eng->fp4m_ssd_fd,b+d,n-d,at+(int64_t)d);if(w<=0)return false;d+=(size_t)w;}return true;};
    for(int l=0;l<L&&ok;l++) {
        uint8_t *hg=eng->h_fp4m_stage_gu,*hs=eng->h_fp4m_stage_gusf;
        uint8_t *hd=eng->h_fp4m_stage_dn,*hds=eng->h_fp4m_stage_dnsf;
        ok=hg&&hs&&hd&&hds;
        if(ok) ok=cudaMemcpy(hg,eng->d_fp4m_gu[l],guB,cudaMemcpyDeviceToHost)==cudaSuccess
              && cudaMemcpy(hs,eng->d_fp4m_gusf[l],gusfB,cudaMemcpyDeviceToHost)==cudaSuccess
              && cudaMemcpy(hd,eng->d_fp4m_dn[l],dnB,cudaMemcpyDeviceToHost)==cudaSuccess
              && cudaMemcpy(hds,eng->d_fp4m_dnsf[l],dnsfB,cudaMemcpyDeviceToHost)==cudaSuccess;
        if(ok){
          const size_t gp=guB/E,gsp=gusfB/E,dp=dnB/E,dsp=dnsfB/E;
          for(int e=0;e<E;e++){size_t z=((size_t)l*E+e)*4;
            eng->h_fp4m_ssd_hash[z+0]=q35_expert_hash_update(1469598103934665603ULL,hg+(size_t)e*gp,gp);
            eng->h_fp4m_ssd_hash[z+1]=q35_expert_hash_update(1469598103934665603ULL,hs+(size_t)e*gsp,gsp);
            eng->h_fp4m_ssd_hash[z+2]=q35_expert_hash_update(1469598103934665603ULL,hd+(size_t)e*dp,dp);
            eng->h_fp4m_ssd_hash[z+3]=q35_expert_hash_update(1469598103934665603ULL,hds+(size_t)e*dsp,dsp);}
          eng->fp4m_ssd_gu_off[l]=off;ok=wr(hg,guB,off);off+=(int64_t)guB;
          eng->fp4m_ssd_gusf_off[l]=off;ok=ok&&wr(hs,gusfB,off);off+=(int64_t)gusfB;
          eng->fp4m_ssd_dn_off[l]=off;ok=ok&&wr(hd,dnB,off);off+=(int64_t)dnB;
          eng->fp4m_ssd_dnsf_off[l]=off;ok=ok&&wr(hds,dnsfB,off);off+=(int64_t)dnsfB; }
        if(ok) { cudaFree(eng->d_fp4m_gu[l]);cudaFree(eng->d_fp4m_gusf[l]);cudaFree(eng->d_fp4m_dn[l]);cudaFree(eng->d_fp4m_dnsf[l]);
                 eng->d_fp4m_gu[l]=eng->d_fp4m_gusf[l]=eng->d_fp4m_dn[l]=eng->d_fp4m_dnsf[l]=NULL; }
    }
    if(ok){
        fsync(eng->fp4m_ssd_fd);
        if(q35_seed_ssd_residency(eng,getenv("FUCINA_EXPERT_RESIDENCY_PLAN"),L,E,MI,H)!=0)ok=false;
        posix_fadvise(eng->fp4m_ssd_fd,0,off,POSIX_FADV_DONTNEED);
        if(ok)eng->fp4m_ssd_stream=1;
    }
    if(!ok) { fprintf(stderr,"fucina: SSD expert streaming setup failed\n"); return -1; }
    fprintf(stderr,"fucina: expert SSD streaming ON (slots=%d, device staging %.2f GiB, backing %.2f GiB)\n",
            slots,(guB+gusfB+dnB+dnsfB)*(double)slots/E/(1024.0*1024*1024),L*(guB+gusfB+dnB+dnsfB)/(1024.0*1024*1024));
    return 0;
}

// Requantize E consecutive FP8 experts (per-expert BF16 128x128 block scales) into ONE contiguous
// Q4_K slab (E x (out*in/256) x 144 B), threaded across experts. The dominant decode bytes drop
// 8 -> 4.5 bpw; quality is gated by greedy-match tests, not bitwise parity (double quantization).
static unsigned char *q35_fp8_experts_to_q4k(
    const uint8_t *const *wp, const uint16_t *const *sp, int E, int out_dim, int in_dim)
{
    static float lut[256]; static bool lut_init = false;
    if (!lut_init) { for (int i = 0; i < 256; i++) lut[i] = q35_fp8_lut_val(i); lut_init = true; }
    const size_t per_w = (size_t)out_dim * in_dim;
    const size_t per_q = per_w / 256 * 144;
    unsigned char *slab = (unsigned char *)malloc((size_t)E * per_q);
    if (!slab) return nullptr;
    std::atomic<int> next(0);
    std::atomic<bool> fail(false);
    auto worker = [&]() {
        std::vector<float> f(per_w);
        for (;;) {
            int e = next.fetch_add(1);
            if (e >= E) break;
            const uint8_t  *we = wp[e];
            const uint16_t *se = sp[e];
            if (!we || !se) { fail = true; break; }
            for (size_t i = 0; i < per_w; i++) {
                int row = (int)(i / in_dim), col = (int)(i % in_dim);
                float bs = q35_bf16_to_f32_host(se[(size_t)(row >> 7) * (in_dim >> 7) + (col >> 7)]);
                f[i] = lut[we[i]] * bs;
            }
            q35_f32_to_q4_K_host(f.data(), slab + (size_t)e * per_q, (int64_t)(per_w / 256));
        }
    };
    std::vector<std::thread> th;
    for (int t = 0; t < 12; t++) th.emplace_back(worker);
    for (auto &x : th) x.join();
    if (fail) { free(slab); return nullptr; }
    return slab;
}

// One descriptor per tensor placed into eng->d_weights.
struct q35fp8_desc {
    uint64_t *off; uint8_t *fmtp; const void *host; size_t bytes; int fmt;
    const void *scale_host; size_t scale_bytes; std::string logical_name;
};

// Resolve a logical `<module>.weight` across block-FP8/ModelOpt and compressed-tensors. Packed
// NVFP4 preserves the output dimension in shape[0] and halves only shape[1].
static inline const st::Tensor *q35_weight_tensor(st::Model &M, const qwen35fp8::Layout &LO,
                                                  const std::string &weight_key) {
    if (const st::Tensor *t=M.find(weight_key)) return t;
    if (LO.compressed && weight_key.size() >= 6 &&
        weight_key.compare(weight_key.size()-6,6,"weight") == 0)
        return M.find(weight_key.substr(0,weight_key.size()-6)+"weight_packed");
    return nullptr;
}

struct q35_native_nv4 {
    const st::Tensor *packed=nullptr, *scale=nullptr, *global=nullptr;
    float global_mul=0.f;
};
static inline q35_native_nv4 q35_native_nv4_resolve(st::Model &M,
                                                     const qwen35fp8::Layout &LO,
                                                     const std::string &weight_key) {
    q35_native_nv4 r;
    std::string base=weight_key.substr(0,weight_key.size()-6);
    if (LO.compressed) {
        r.packed=M.find(base+"weight_packed");
        r.scale=M.find(base+"weight_scale");
        r.global=M.find(base+"weight_global_scale");
        if (r.global) { float raw=*(const float*)r.global->data; r.global_mul=raw!=0.f?1.f/raw:0.f; }
    } else if (LO.modelopt) {
        r.packed=M.find(weight_key);
        r.scale=M.find(weight_key+"_scale");
        r.global=M.find(weight_key+"_scale_2");
        if (r.global) r.global_mul=*(const float*)r.global->data;
    }
    return r;
}

// A qwen3_5_moe checkpoint iff the per-layer sparse experts are present (dense has mlp.gate_proj).
static inline bool qwen35_fp8_is_moe(st::Model &M, qwen35fp8::Layout &LO) {
    return q35_weight_tensor(M,LO,qwen35fp8::lkey(LO,0,"mlp.experts.0.gate_proj.weight")) != nullptr;
}

enum class q35_source_kind : uint8_t { FP8_BLOCK, FP8_SCALED, NVFP4 };

static bool q35_shape_is(const st::Tensor *t,int64_t rows,int64_t cols) {
    return t && t->shape.size()==2 && t->shape[0]==rows && t->shape[1]==cols;
}

// Producer adapter/preflight for one logical weight. This normalizes official block-FP8,
// ModelOpt NVFP4, and compressed-tensors naming without allocating or selecting a CUDA kernel.
static bool q35_validate_weight_source(st::Model &M,const qwen35fp8::Layout &LO,
                                       const std::string &key,int out_dim,int in_dim,
                                       q35_source_kind *kind,std::string &error) {
    const st::Tensor *w=M.find(key), *inv=M.find(key+"_scale_inv");
    if(w&&inv){
        const size_t scale_bytes=(size_t)((out_dim+127)/128)*((in_dim+127)/128)*2;
        if(w->dtype!=st::Dtype::F8_E4M3 || !q35_shape_is(w,out_dim,in_dim)) {
            error=key+": expected F8_E4M3 [out,in]"; return false;
        }
        if(inv->dtype!=st::Dtype::BF16 || inv->nbytes!=scale_bytes) {
            error=key+"_scale_inv: expected BF16 128x128 scale grid"; return false;
        }
        *kind=q35_source_kind::FP8_BLOCK; return true;
    }
    if(LO.mixed_nvfp4()){
        q35_native_nv4 n4=q35_native_nv4_resolve(M,LO,key);
        if((n4.packed&&n4.packed->dtype==st::Dtype::U8)||n4.global){
            if(!n4.packed||!n4.scale||!n4.global || n4.packed->dtype!=st::Dtype::U8 ||
               !q35_shape_is(n4.packed,out_dim,in_dim/2) || n4.packed->nbytes!=(size_t)out_dim*in_dim/2 ||
               (n4.scale->dtype!=st::Dtype::U8 && n4.scale->dtype!=st::Dtype::F8_E4M3) || n4.scale->nbytes==0 ||
               n4.global->dtype!=st::Dtype::F32 || n4.global->nbytes!=4 || n4.global_mul==0.f) {
                error=key+": incomplete or malformed native NVFP4 weight/scale/global triplet"; return false;
            }
            *kind=q35_source_kind::NVFP4; return true;
        }
        const st::Tensor *scaled=M.find(key), *scale=M.find(key+"_scale");
        if(!scaled||!scale || scaled->dtype!=st::Dtype::F8_E4M3 ||
           !q35_shape_is(scaled,out_dim,in_dim) ||
           (scale->dtype!=st::Dtype::F32 && scale->dtype!=st::Dtype::BF16) || scale->nbytes==0) {
            error=key+": expected scaled FP8 weight and scale"; return false;
        }
        *kind=q35_source_kind::FP8_SCALED; return true;
    }
    error=key+": missing weight or scale"; return false;
}

static bool q35_validate_vector(st::Model &M,const std::string &key,int64_t elements,
                                bool allow_f32,std::string &error) {
    const st::Tensor *t=M.find(key);
    int64_t count=1; if(t) for(int64_t d:t->shape) count*=d;
    if(!t || count!=elements ||
       (t->dtype!=st::Dtype::BF16 && (!allow_f32 || t->dtype!=st::Dtype::F32))) {
        error=key+": wrong element count or dtype"; return false;
    }
    return true;
}

// Complete host-only schema pass. It runs from setup_cfg, before engine CUDA allocations begin.
static bool q35_preflight(st::Model &M,qwen35fp8::Layout &LO,int H,int HD,int NQ,int NKV,
                          int INNER,int TSR,int I,int VOC,int E,int MI,bool moe,std::string &error) {
    q35_source_kind k;
    const st::Tensor *embed=M.find(LO.embed_key), *head=M.find(LO.lmhead_key);
    if(!embed || embed->dtype!=st::Dtype::BF16 || !q35_shape_is(embed,VOC,H)) {
        error=LO.embed_key+": expected BF16 [vocab,hidden]"; return false;
    }
    if(!head){ error=LO.lmhead_key+": missing lm head"; return false; }
    if(!LO.mixed_nvfp4() && (head->dtype!=st::Dtype::BF16 || !q35_shape_is(head,VOC,H))) {
        error=LO.lmhead_key+": expected BF16 [vocab,hidden]"; return false;
    }
    if(!q35_validate_vector(M,LO.final_norm_key,H,false,error)) return false;
    const int CONVD=2*M2_KEYD+INNER;
    for(int l=0;l<LO.n_layers;l++){
        auto lk=[&](const char *s){ return qwen35fp8::lkey(LO,l,s); };
        if(!q35_validate_vector(M,lk("input_layernorm.weight"),H,false,error) ||
           !q35_validate_vector(M,lk("post_attention_layernorm.weight"),H,false,error)) return false;
        if(qwen35fp8::is_full(LO,l)){
            if(!q35_validate_weight_source(M,LO,lk("self_attn.q_proj.weight"),2*NQ*HD,H,&k,error) ||
               !q35_validate_weight_source(M,LO,lk("self_attn.k_proj.weight"),NKV*HD,H,&k,error) ||
               !q35_validate_weight_source(M,LO,lk("self_attn.v_proj.weight"),NKV*HD,H,&k,error) ||
               !q35_validate_weight_source(M,LO,lk("self_attn.o_proj.weight"),H,NQ*HD,&k,error) ||
               !q35_validate_vector(M,lk("self_attn.q_norm.weight"),HD,false,error) ||
               !q35_validate_vector(M,lk("self_attn.k_norm.weight"),HD,false,error)) return false;
        } else {
            if(!q35_validate_weight_source(M,LO,lk("linear_attn.in_proj_qkv.weight"),CONVD,H,&k,error) ||
               !q35_validate_weight_source(M,LO,lk("linear_attn.in_proj_z.weight"),INNER,H,&k,error) ||
               !q35_validate_weight_source(M,LO,lk("linear_attn.out_proj.weight"),H,INNER,&k,error) ||
               !q35_validate_vector(M,lk("linear_attn.A_log"),TSR,true,error) ||
               !q35_validate_vector(M,lk("linear_attn.dt_bias"),TSR,false,error) ||
               !q35_validate_vector(M,lk("linear_attn.norm.weight"),M2_SD,true,error)) return false;
            const st::Tensor *a=M.find(lk("linear_attn.in_proj_a.weight"));
            const st::Tensor *b=M.find(lk("linear_attn.in_proj_b.weight"));
            const st::Tensor *conv=M.find(lk("linear_attn.conv1d.weight"));
            if(!a||!b||!conv || a->dtype!=st::Dtype::BF16 || b->dtype!=st::Dtype::BF16 ||
               !q35_shape_is(a,TSR,H) || !q35_shape_is(b,TSR,H) ||
               conv->dtype!=st::Dtype::BF16 || conv->nbytes!=(size_t)CONVD*M2_CK*2) {
                error=lk("linear_attn")+": malformed a/b/conv tensors"; return false;
            }
        }
        if(moe){
            if(!q35_validate_vector(M,lk("mlp.shared_expert_gate.weight"),H,false,error) ||
               !q35_validate_weight_source(M,LO,lk("mlp.shared_expert.gate_proj.weight"),I,H,&k,error) ||
               !q35_validate_weight_source(M,LO,lk("mlp.shared_expert.up_proj.weight"),I,H,&k,error) ||
               !q35_validate_weight_source(M,LO,lk("mlp.shared_expert.down_proj.weight"),H,I,&k,error)) return false;
            const st::Tensor *router=M.find(lk("mlp.gate.weight"));
            if(!router || router->dtype!=st::Dtype::BF16 || !q35_shape_is(router,E,H)) {
                error=lk("mlp.gate.weight")+": malformed router"; return false;
            }
            q35_source_kind layer_kind=q35_source_kind::FP8_BLOCK; bool first=true;
            const char *proj[3]={"gate_proj","up_proj","down_proj"};
            for(int p=0;p<3;p++) for(int e=0;e<E;e++){
                char name[96]; snprintf(name,sizeof(name),"mlp.experts.%d.%s.weight",e,proj[p]);
                q35_source_kind ek; int od=p==2?H:MI, id=p==2?MI:H;
                if(!q35_validate_weight_source(M,LO,lk(name),od,id,&ek,error)) return false;
                if(first){ layer_kind=ek; first=false; }
                else if(ek!=layer_kind){ error=lk("mlp.experts")+": mixed expert formats within layer"; return false; }
            }
        } else if(!q35_validate_weight_source(M,LO,lk("mlp.gate_proj.weight"),I,H,&k,error) ||
                  !q35_validate_weight_source(M,LO,lk("mlp.up_proj.weight"),I,H,&k,error) ||
                  !q35_validate_weight_source(M,LO,lk("mlp.down_proj.weight"),H,I,&k,error)) return false;
    }
    return true;
}

// Set eng->cfg from the FP8 checkpoint. H / NKV / vocab / FFN dims come from the tensor shapes
// (9B dense: H=4096 NKV=4; 35B-A3B MoE: H=2048 NKV=2 E=256 top-8 MI=SI=512); the head/GDN geometry
// is the shared M2_* baked set. Called EARLY (before the cfg-dependent scratch allocations); the
// weight fill runs later, after create's d_weights=NULL reset.
static int qwen35_fp8_setup_cfg(gemma4_engine_t *eng, st::Model &M, qwen35fp8::Layout &LO) {
    // Head/GDN geometry SHARED across all Qwen3.5/3.6 sizes stays baked (M2_*): head_dim, state_size,
    // conv_kernel, key-head count, rotary_dim, theta. The dims that VARY by size are DERIVED from the
    // loaded tensor shapes (NOT the substring config scanner — VL checkpoints carry a vision_config
    // that shadows num_attention_heads / linear_num_value_heads):
    //   n_heads (NQ)          = q_proj rows / (2*head_dim)   [fused q+gate pack]
    //   ssm_inner_size (VALD) = in_proj_z rows               [value-path width = n_val_heads*SD]
    //   ssm_time_step_rank    = n_val_heads = INNER/SD       [A_log/dt_bias/a/b width]
    // 9B(H4096/NKV4)+35B-A3B(H2048/NKV2) derive back to the M2_* values (regression-safe); 27B-dense
    // derives NQ24/INNER6144/NVH48 → runtime support with no new #defines.
    const int HD=M2_HEAD, SD=M2_SD, CK=M2_CK;
    const bool moe = qwen35_fp8_is_moe(M, LO);
    const st::Tensor *embT=M.find(LO.embed_key);
    // NKV/NQ from a FULL layer's k_proj/q_proj rows (layer full_attention_interval-1 is always FULL).
    const st::Tensor *kT=M.find(qwen35fp8::lkey(LO,LO.full_attention_interval-1,"self_attn.k_proj.weight"));
    const st::Tensor *qT=M.find(qwen35fp8::lkey(LO,LO.full_attention_interval-1,"self_attn.q_proj.weight"));
    // in_proj_z from a LINEAR (GDN) layer — layer 0 is LINEAR for the period-4 hybrid.
    const st::Tensor *zT=M.find(qwen35fp8::lkey(LO,0,"linear_attn.in_proj_z.weight"));
    const st::Tensor *gateT=q35_weight_tensor(M,LO,qwen35fp8::lkey(
        LO,0,moe ? "mlp.shared_expert.gate_proj.weight" : "mlp.gate_proj.weight"));
    if(!embT||!kT||!qT||!zT||!gateT){ fprintf(stderr,"qwen35_fp8_engine: missing embed/k_proj/q_proj/in_proj_z/gate\n"); return -1; }
    const int VOC=(int)embT->shape[0], H=(int)embT->shape[1], NKV=(int)kT->shape[0]/HD;
    const int NQ=(int)qT->shape[0]/(2*HD);     // fused q+gate: q_proj rows = 2*NQ*HD
    const int INNER=(int)zT->shape[0];          // value-path inner width = NVH*SD
    const int NVH=INNER/SD, TSR=NVH;            // num value heads (== A_log/dt_bias width)
    const int I=(int)gateT->shape[0], L=LO.n_layers;
    gemma4_model_config_t *c=&eng->cfg;
    c->arch=GEMMA4_ARCH_QWEN3_5; c->n_layers=L; c->hidden_size=H; c->head_dim=HD; c->n_heads=NQ;
    c->n_kv_global=NKV; c->n_kv_sliding=NKV;
    c->intermediate=I; c->vocab_size=VOC; c->rotary_dim=M2_ROT; c->full_attention_interval=LO.full_attention_interval;
    c->ssm_state_size=SD; c->ssm_conv_kernel=CK; c->ssm_inner_size=INNER; c->ssm_group_count=M2_NKH; c->ssm_time_step_rank=TSR;
    fprintf(stderr,"fucina: qwen35 geometry (runtime-derived) — NQ=%d NKV=%d NVH=%d INNER=%d CONVD=%d (M2 baked NQ%d/NVH%d/INNER%d)\n",
            NQ, NKV, NVH, INNER, 2*M2_KEYD+INNER, M2_NQ, M2_NVH, M2_VALD);
    int E=0, MI=0;
    if (moe) {
        const st::Tensor *rgT=M.find(qwen35fp8::lkey(LO,0,"mlp.gate.weight"));
        const st::Tensor *egT=q35_weight_tensor(M,LO,qwen35fp8::lkey(LO,0,"mlp.experts.0.gate_proj.weight"));
        if(!rgT||!egT){ fprintf(stderr,"qwen35_fp8_engine: missing router/expert tensors\n"); return -1; }
        E=(int)rgT->shape[0]; MI=(int)egT->shape[0];
        c->n_experts=E; c->expert_ffn=MI;
        long tk=8; qwen35fp8::cfg_int(M.config_json(),"\"num_experts_per_tok\"",tk);
        c->n_experts_used=(tk>0)?(int)tk:8;
        eng->moe_shared_inter=I;      // I above = shared_expert_intermediate for MoE
    }
    std::string preflight_error;
    if(!q35_preflight(M,LO,H,HD,NQ,NKV,INNER,TSR,I,VOC,E,MI,moe,preflight_error)){
        fprintf(stderr,"qwen35_fp8_engine: preflight failed: %s\n",preflight_error.c_str());
        return -1;
    }
    fprintf(stderr,"fucina: qwen35 host preflight passed (all required tensors validated before CUDA allocation)\n");
    int nfull=0;
    for(int l=0;l<L;l++){ bool full=qwen35fp8::is_full(LO,l);
        c->attn_kind[l]=full?GEMMA4_ATTN_FULL:GEMMA4_ATTN_LINEAR; c->is_global[l]=full?1:0; nfull+=full?1:0; }
    c->n_full=nfull; c->n_global=nfull;
    eng->format=FORMAT_FP8_BLOCK; eng->tdata_start=0;
    eng->output_tied=0;                                     // untied LM head (tie_word_embeddings=false)
    for(int l=0;l<GEMMA4_CAP_LAYERS;l++) eng->h_out_scale[l]=1.0f;  // qwen35 has no per-layer output scale
    return 0;
}

// Optional Phase-B precision policy. Parsing is load-once: locate an exact canonical
// tensor key and read the codec from that JSON object's bounded span. No STL reaches decode.
struct q35_precision_policy {
    bool enabled=false;
    std::string raw, path;
    int kept_fp8=0, requant_q4k=0;
    bool load() {
        const char *p=getenv("FUCINA_PRECISION_POLICY"); if(!p||!*p) return true;
        path=p; FILE *f=fopen(p,"rb"); if(!f){ fprintf(stderr,"fucina: cannot open precision policy %s\n",p); return false; }
        fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
        if(n<=0||n>(64L<<20)){ fclose(f); fprintf(stderr,"fucina: invalid precision policy size\n"); return false; }
        raw.resize((size_t)n); bool ok=fread(raw.data(),1,(size_t)n,f)==(size_t)n; fclose(f);
        if(!ok||raw.find("fucina-precision-policy-v1")==std::string::npos){ fprintf(stderr,"fucina: invalid precision policy format\n"); return false; }
        if(raw.find("\"codec\": \"int2\"")!=std::string::npos || raw.find("\"codec\":\"int2\"")!=std::string::npos){
            fprintf(stderr,"fucina: precision policy requests INT2 but no accepted sub-4-bit kernel is available\n"); return false;
        }
        enabled=true; return true;
    }
    std::string codec(const std::string &checkpoint_key) const {
        if(!enabled) return "";
        size_t lp=checkpoint_key.find("layers.");
        std::string key=lp==std::string::npos?checkpoint_key:checkpoint_key.substr(lp);
        size_t p=raw.find("\""+key+"\""); if(p==std::string::npos) return "";
        size_t end=raw.find('}',p),c=raw.find("\"codec\"",p);
        if(end==std::string::npos||c==std::string::npos||c>end) return "";
        c=raw.find(':',c); if(c==std::string::npos||c>end) return "";
        size_t q1=raw.find('"',c),q2=q1==std::string::npos?q1:raw.find('"',q1+1);
        return q1==std::string::npos||q2==std::string::npos||q2>end?"":raw.substr(q1+1,q2-q1-1);
    }
};

static int q35_cuda_allocate(void *,void **ptr,size_t bytes) {
    return cudaMalloc(ptr,bytes)==cudaSuccess?0:1;
}
static void q35_cuda_release(void *,void *ptr) { if(ptr) cudaFree(ptr); }
static int q35_cuda_upload(void *,void *dst,const void *src,size_t bytes) {
    return cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice)==cudaSuccess?0:1;
}

static int qwen35_fp8_fill_engine(gemma4_engine_t *eng, st::Model &M, qwen35fp8::Layout &LO) {
    const int H=eng->cfg.hidden_size, HD=M2_HEAD, NQ=eng->cfg.n_heads, NKV=eng->cfg.n_kv_global;
    const int CONVD=(2*M2_KEYD+eng->cfg.ssm_inner_size), INNER=eng->cfg.ssm_inner_size;
    const int VOC=eng->cfg.vocab_size, I=eng->cfg.intermediate, L=LO.n_layers;
    size_t fill_free_before=0, fill_total=0, fill_free_after_experts=0;
    size_t fill_free_after_core=0, fill_free_before_scratch=0, fill_free_after_scratch=0;
    cudaMemGetInfo(&fill_free_before, &fill_total);
    const q35_host_meminfo host_mem_before=q35_read_host_meminfo();
    DeviceAllocationOps allocation_ops{nullptr,q35_cuda_allocate,q35_cuda_release,q35_cuda_upload};
    if(!eng->device_allocations) eng->device_allocations=new DeviceAllocationRegistry(allocation_ops);
    DeviceAllocationSet load_allocation(allocation_ops);

    q35_precision_policy precision_policy;
    if(!precision_policy.load()) return -1;
    std::vector<q35fp8_desc> D;
    std::vector<void*> tofree;          // host temporaries (bf16→f32 / bf16→Q8_0) freed after upload
    auto find=[&](const std::string&k)->const st::Tensor*{ return M.find(k); };
    auto lk=[&](int l,const char*s){ return qwen35fp8::lkey(LO,l,s); };
    // FP8 block weight: E4M3 bytes + BF16 block-scale sibling → FORMAT_FP8_BLOCK + scale-table entry.
    // Mixed-NVFP4 fallbacks when no `_scale_inv` grid exists:
    //   per-tensor FP8 (`<w>_scale` F32 scalar) → broadcast the scalar into a block grid;
    //   native NVFP4 (ModelOpt or compressed-tensors naming) → dequant→BF16→FP8-block for
    //   consumers that do not yet have a direct FP4 projection (principally shared expert).
    auto put_fp8=[&](uint64_t*off,uint8_t*fmtp,const std::string&key,int out_dim,int in_dim)->bool{
        const st::Tensor*w=find(key), *s=find(key+"_scale_inv");
        if(w&&s){ D.push_back({off,fmtp,w->data,(size_t)out_dim*in_dim,FORMAT_FP8_BLOCK,s->data,s->nbytes,key}); return true; }
        if(LO.mixed_nvfp4()){
            const size_t gridb=(size_t)((out_dim+127)/128)*((in_dim+127)/128)*2;
            q35_native_nv4 n4=q35_native_nv4_resolve(M,LO,key);
            if(n4.packed&&n4.scale&&n4.global&&n4.packed->dtype==st::Dtype::U8){
                uint16_t*bf=q35_nvfp4_to_bf16_host((const uint8_t*)n4.packed->data,
                                                   (const uint8_t*)n4.scale->data,n4.global_mul,
                                                   out_dim,in_dim);
                uint8_t*w8=nullptr; uint16_t*sg=nullptr;
                if(bf&&q35_bf16_to_fp8_block_host(bf,out_dim,in_dim,&w8,&sg)){
                    free(bf); tofree.push_back(w8); tofree.push_back(sg);
                    D.push_back({off,fmtp,w8,(size_t)out_dim*in_dim,FORMAT_FP8_BLOCK,sg,gridb,key});
                    return true;
                }
                free(bf);
            } else if(w&&w->dtype==st::Dtype::F8_E4M3){
                const st::Tensor*ps=find(key+"_scale");
                if(ps&&ps->nbytes==4){
                    uint16_t*sg=q35_synth_scale_grid(*(const float*)ps->data,out_dim,in_dim);
                    if(sg){ tofree.push_back(sg);
                        D.push_back({off,fmtp,w->data,(size_t)out_dim*in_dim,FORMAT_FP8_BLOCK,sg,gridb,key});
                        return true; }
                } else if(ps){
                    uint16_t *bf=q35_fp8_scaled_to_bf16_host((const uint8_t*)w->data,ps,out_dim,in_dim);
                    uint8_t *w8=nullptr; uint16_t *sg=nullptr;
                    if(bf&&q35_bf16_to_fp8_block_host(bf,out_dim,in_dim,&w8,&sg)){
                        free(bf); tofree.push_back(w8); tofree.push_back(sg);
                        D.push_back({off,fmtp,w8,(size_t)out_dim*in_dim,FORMAT_FP8_BLOCK,sg,gridb,key});
                        return true;
                    }
                    free(bf);
                }
            }
        }
        fprintf(stderr,"qwen35_fp8_engine: missing %s\n",key.c_str()); return false;
    };
    // BF16 tensor → f32 in d_weights (norms/conv/dt_bias). neg_exp for A_log; scalar +1 baked for norms.
    auto put_f32_from_bf16=[&](uint64_t*off,const std::string&key,float add,bool neg_exp)->bool{
        const st::Tensor*t=find(key); if(!t){ fprintf(stderr,"qwen35_fp8_engine: missing %s\n",key.c_str()); return false; }
        int64_t n=(int64_t)(t->nbytes/2); float*f=(float*)malloc((size_t)n*4);
        const uint16_t*src=(const uint16_t*)t->data;
        for(int64_t i=0;i<n;i++){ float v=q35_bf16_to_f32_host(src[i]); f[i]=neg_exp?(-expf(v)):(v+add); }
        tofree.push_back(f); D.push_back({off,nullptr,f,(size_t)n*4,-1,nullptr,0,key}); return true;
    };
    // A_log / linear_attn.norm → d_weights, optional neg_exp (ssm_a = -exp(A_log)). DTYPE-AWARE:
    // these ship as F32 (Qwen3.5-35B-A3B-FP8) OR BF16 (the Qwen3.6 repack). Read by the recorded
    // dtype — taking a BF16 tensor as raw F32 halves the element count and corrupts the values
    // (the qwen3.6 A_log/norm garbage regression); BF16 delegates to the converting loader.
    auto put_f32=[&](uint64_t*off,const std::string&key,bool neg_exp)->bool{
        const st::Tensor*t=find(key); if(!t){ fprintf(stderr,"qwen35_fp8_engine: missing %s\n",key.c_str()); return false; }
        if(t->dtype==st::Dtype::BF16) return put_f32_from_bf16(off,key,0.0f,neg_exp);
        int64_t n=(int64_t)(t->nbytes/4);
        if(neg_exp){ float*f=(float*)malloc((size_t)n*4); const float*src=(const float*)t->data;
            for(int64_t i=0;i<n;i++) f[i]=-expf(src[i]); tofree.push_back(f); D.push_back({off,nullptr,f,(size_t)n*4,-1,nullptr,0,key}); }
        else D.push_back({off,nullptr,t->data,(size_t)n*4,-1,nullptr,0,key});
        return true;
    };
    // FP8 proj → Q4_K in d_weights (FUCINA_MOE_Q4K mixer requant: 8 → 4.5 bpw; served by the
    // existing dp4a Q4_K batched GEMV + BF16-dequant prefill). Uses the expert requantizer with E=1.
    auto put_q4k_from_fp8=[&](uint64_t*off,uint8_t*fmtp,const std::string&key,int out_dim,int in_dim)->bool{
        const st::Tensor*w=find(key), *sc=find(key+"_scale_inv");
        const uint8_t *wp=w?(const uint8_t*)w->data:nullptr;
        const uint16_t *sp=nullptr; uint16_t *synth=nullptr; uint8_t *requant=nullptr;
        if(w&&sc) sp=(const uint16_t*)sc->data;
        else if(w&&LO.mixed_nvfp4()&&w->dtype==st::Dtype::F8_E4M3){
            const st::Tensor*ps=find(key+"_scale");
            if(ps&&ps->nbytes==4){ synth=q35_synth_scale_grid(*(const float*)ps->data,out_dim,in_dim); sp=synth; }
            else if(ps){
                uint16_t *bf=q35_fp8_scaled_to_bf16_host(wp,ps,out_dim,in_dim);
                if(bf){ q35_bf16_to_fp8_block_host(bf,out_dim,in_dim,&requant,&synth); free(bf); }
                if(requant&&synth){ wp=requant; sp=synth; }
            }
        }
        if(!w||!sp){ fprintf(stderr,"qwen35_fp8_engine: missing %s\n",key.c_str()); free(requant); free(synth); return false; }
        unsigned char *q4=q35_fp8_experts_to_q4k(&wp,&sp,1,out_dim,in_dim);
        free(requant); free(synth);
        if(!q4) return false;
        tofree.push_back(q4);
        D.push_back({off,fmtp,q4,(size_t)((size_t)out_dim*in_dim/256)*144,FORMAT_Q4_K,nullptr,0,key});
        return true;
    };
    // Raw-bytes descriptor (no format tag, no scale-table entry): builds the CONTIGUOUS per-layer
    // MoE expert slabs — E consecutive FP8 weight blocks (EFFN·H each, 32-aligned by size) and E
    // consecutive BF16 scale blocks. `off` records only the slab base (expert 0).
    static uint64_t q35moe_off_sink;
    auto put_raw=[&](uint64_t*off,const std::string&key,size_t bytes)->bool{
        const st::Tensor*t=find(key);
        if(!t||(size_t)t->nbytes!=bytes){ fprintf(stderr,"qwen35_fp8_engine: missing/size %s\n",key.c_str()); return false; }
        D.push_back({off,nullptr,t->data,bytes,-1,nullptr,0,key}); return true;
    };
    const bool moe = qwen35_fp8_is_moe(M, LO);
    const int E = eng->cfg.n_experts, MI = eng->cfg.expert_ffn, SI = eng->moe_shared_inter;
    // FUCINA_MOE_Q4K also requants the MIXER + SHARED-EXPERT projections (the B=1 bytes lever:
    // 1.27+0.12 GB FP8 → ~0.72+0.07 GB Q4_K); in_a/in_b/norms/embed/head paths are unchanged.
    // DEFAULT-ON for qwen3_5_moe (ship defaults, not flags): the Q4_K-requant serving beat the
    // FP8 path at every batch size once the grouped GEMV landed, with the greedy oracle still
    // 8/8 and every self-test bitwise across a full day of gates. FUCINA_MOE_FP8=1 restores the
    // pure-FP8 serving as the escape hatch.
    // DENSE checkpoints (9B/27B, no experts) get the SAME treatment via the SAME put_w/FORMAT_Q4_K
    // machinery: previously q4k_mode was `moe &&`-gated, so dense attn/GDN (already routed through
    // put_w below) silently stayed FP8-only and dense FFN was hardcoded put_fp8 with no Q4_K path
    // at all — an oversight, not a measured tradeoff (the requant is byte-for-byte the identical
    // FP8-E4M3-weight + BF16-block-scale source format MoE already proves correct). Dropping the
    // `moe &&` extends 8→4.5 bpw to attn+GDN+FFN on dense too (~40% of resident weight bytes on a
    // bandwidth-bound decode = both a memory win and a direct decode-speed win).
    const bool q4k_mode = !getenv("FUCINA_MOE_FP8");
    // Experts DEFAULT to NVFP4 (CUTLASS sm120 grouped block-scaled GEMM, 4.5 bpw): the tensor-core
    // tiles amortize the expert weight read across every token routed to the expert, where the
    // dp4a grouped GEMV is memory-LATENCY-bound (measured B=16 aggregate: identical-prompt
    // 436→600 tok/s, diverse 307→372; B=1 −3%, still ahead of vLLM). The CUDA-13 abort that
    // demoted this path (placeholder-TMA cuTensorMapEncodeTiled err 700, a103b74) no longer
    // reproduces on the current driver stack — gated by oracle 8/8 + self-test + prefill parity.
    // Wide (prefill) calls dequant the NVFP4 slabs → BF16 and ride the grouped-cuBLAS tc path
    // (faster than the CUTLASS GEMM at prefill widths). FUCINA_MOE_Q4K=1 restores the Q4_K
    // grouped-GEMV experts (also the automatic fallback if the NVFP4 build fails);
    // FUCINA_MOE_FP8=1 = raw FP8.
    bool fp4_mode = q4k_mode && !getenv("FUCINA_MOE_Q4K");
    auto put_w=[&](uint64_t*off,uint8_t*fmtp,const std::string&key,int od,int idm)->bool{
        std::string codec=precision_policy.codec(key);
        if(codec=="fp8_block" || codec=="fp8_or_bf16") {
            precision_policy.kept_fp8++;
            return put_fp8(off,fmtp,key,od,idm);
        }
        if(q4k_mode) precision_policy.requant_q4k++;
        return q4k_mode ? put_q4k_from_fp8(off,fmtp,key,od,idm) : put_fp8(off,fmtp,key,od,idm);
    };

    bool ok=true;
    auto &TS=eng->tensors;
    for(int l=0; ok && l<L; l++){
        auto &T=TS.layers[l];
        ok=ok && put_f32_from_bf16(&T.attn_norm, lk(l,"input_layernorm.weight"), 1.0f,false);
        ok=ok && put_f32_from_bf16(&T.ffn_norm,  lk(l,"post_attention_layernorm.weight"), 1.0f,false);
        if(moe){
            // ── Qwen3_5MoeSparseMoeBlock: router (f32) + E-expert FP8 slabs + shared expert ──
            ok=ok && put_f32_from_bf16(&eng->moe_router[l],     lk(l,"mlp.gate.weight"), 0.0f,false);          // [E×H]
            ok=ok && put_f32_from_bf16(&eng->moe_sh_gatevec[l], lk(l,"mlp.shared_expert_gate.weight"), 0.0f,false); // [H]
            ok=ok && put_fp8(&eng->moe_sh_gate[l],nullptr, lk(l,"mlp.shared_expert.gate_proj.weight"), SI, H);
            ok=ok && put_fp8(&eng->moe_sh_up[l],  nullptr, lk(l,"mlp.shared_expert.up_proj.weight"),   SI, H);
            ok=ok && put_fp8(&eng->moe_sh_down[l],nullptr, lk(l,"mlp.shared_expert.down_proj.weight"), H, SI);
            // (shared expert stays FP8: its GEMMs read wscale_fp8 by pointer — small bytes anyway)
            const size_t wb=(size_t)MI*H, sb=(size_t)((MI+127)/128)*(H/128)*2;   // bytes/expert: FP8 w, BF16 scale
            struct { const char *proj; uint64_t *woff, *soff; } EX[3] = {
                {"gate_proj", &eng->moe_gate_exps[l], &eng->moe_gate_scales[l]},
                {"up_proj",   &eng->moe_up_exps[l],   &eng->moe_up_scales[l]},
                {"down_proj", &eng->moe_down_exps[l], &eng->moe_down_scales[l]},
            };
            // FUCINA_MOE_Q4K=1: requantize the experts FP8→Q4_K at load (8 → 4.5 bpw on the
            // dominant decode bytes) and serve them through the EXISTING GGUF Q4_K grouped
            // machinery (dg_quantize_q8_1 + dg_mmq_q4_K_grouped). Opt-in until the greedy-match
            // quality gate flips it default. All decode-kernel knobs measured flat — fewer bytes
            // is the only remaining decode lever (see moe35b-vllm-headtohead).
            if(LO.mixed_nvfp4()){
                // Mixed checkpoints may use native NVFP4 (ModelOpt or compressed-tensors naming)
                // or per-tensor FP8 for a whole expert layer. The accurate Unsloth variant keeps
                // its final layers FP8; the Fast variant is native NVFP4 throughout.
                std::vector<q35_nv4src> n4[3];
                bool native=true;
                for(int p=0; p<3 && native; p++){
                    n4[p].assign(E,{nullptr,nullptr,0.f});
                    for(int e=0; e<E && native; e++){
                        char buf[96]; snprintf(buf,sizeof(buf),"mlp.experts.%d.%s.weight",e,EX[p].proj);
                        std::string key=lk(l,buf);
                        q35_native_nv4 src=q35_native_nv4_resolve(M,LO,key);
                        if(!src.packed||!src.scale||!src.global){ native=false; break; }
                        n4[p][e]={(const uint8_t*)src.packed->data,
                                  (const uint8_t*)src.scale->data,src.global_mul};
                    }
                }
                bool built=false;
                if(native) built=q35_fp4_expert_slabs(eng,l,L,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
                    n4[0].data(),n4[1].data(),n4[2].data(),E,MI,H)==0;
                if(!native){
                    std::vector<const uint8_t*> w3[3];
                    std::vector<const uint16_t*> s3[3];
                    std::vector<void*> converted;
                    bool scalar=true;
                    for(int p=0;p<3&&scalar;p++){
                        const int od=p==2?H:MI, idm=p==2?MI:H;
                        w3[p].assign(E,nullptr); s3[p].assign(E,nullptr);
                        for(int e=0;e<E&&scalar;e++){
                            char buf[96]; snprintf(buf,sizeof(buf),"mlp.experts.%d.%s.weight",e,EX[p].proj);
                            std::string key=lk(l,buf);
                            const st::Tensor *w=find(key), *sc=find(key+"_scale");
                            if(!w||!sc||w->dtype!=st::Dtype::F8_E4M3){ scalar=false; break; }
                            uint16_t *grid=nullptr; uint8_t *w8=nullptr;
                            if(sc->nbytes==4) grid=q35_synth_scale_grid(*(const float*)sc->data,od,idm);
                            else {
                                uint16_t *bf=q35_fp8_scaled_to_bf16_host((const uint8_t*)w->data,sc,od,idm);
                                if(bf){ q35_bf16_to_fp8_block_host(bf,od,idm,&w8,&grid); free(bf); }
                            }
                            if(!grid){ free(w8); scalar=false; break; }
                            converted.push_back(grid); if(w8) converted.push_back(w8);
                            w3[p][e]=w8?w8:(const uint8_t*)w->data; s3[p][e]=grid;
                        }
                    }
                    if(scalar) built=q35_fp4_expert_slabs(eng,l,L,w3[0].data(),s3[0].data(),
                        w3[1].data(),s3[1].data(),w3[2].data(),s3[2].data(),
                        nullptr,nullptr,nullptr,E,MI,H)==0;
                    for(void *p:converted) free(p);
                }
                if(built) eng->moe_experts_fp4=1;
                else {
                    fprintf(stderr,"fucina: mixed NVFP4/FP8 expert build failed at layer %d — aborting load\n",l);
                    ok=false;
                }
            } else if(fp4_mode){
                // NVFP4 experts: gather the 3 projs' host FP8+scale pointers, build the fused
                // gate|up + down E2M1 slabs on device. Failure on layer 0 falls back to Q4_K;
                // failure after layer 0 aborts the load (mixed expert formats are not servable).
                std::vector<const uint8_t*>  w3[3]; std::vector<const uint16_t*> s3[3];
                bool got=true;
                for(int p=0; p<3 && got; p++){
                    w3[p].assign(E,nullptr); s3[p].assign(E,nullptr);
                    for(int e=0; e<E && got; e++){
                        char buf[96]; snprintf(buf,sizeof(buf),"mlp.experts.%d.%s.weight",e,EX[p].proj);
                        const st::Tensor *w=find(lk(l,buf));
                        snprintf(buf,sizeof(buf),"mlp.experts.%d.%s.weight_scale_inv",e,EX[p].proj);
                        const st::Tensor *sct=find(lk(l,buf));
                        if(!w||!sct){ got=false; break; }
                        w3[p][e]=(const uint8_t*)w->data; s3[p][e]=(const uint16_t*)sct->data;
                    }
                }
                if(got && q35_fp4_expert_slabs(eng,l,L,w3[0].data(),s3[0].data(),
                        w3[1].data(),s3[1].data(),w3[2].data(),s3[2].data(),
                        nullptr,nullptr,nullptr,E,MI,H)==0){
                    eng->moe_experts_fp4 = 1;
                } else if(eng->moe_experts_fp4){
                    fprintf(stderr,"fucina: NVFP4 expert build failed at layer %d — aborting load\n",l);
                    ok=false;
                } else {
                    fprintf(stderr,"fucina: NVFP4 expert build failed — Q4_K expert fallback\n");
                    fp4_mode=false;
                }
            }
            if(eng->moe_experts_fp4 && eng->d_fp4m_gu[l]){
                if(l==0) ok=ok&&load_allocation.adopt((void**)&eng->d_fp4m_gsw,eng->d_fp4m_gsw,
                                                       (size_t)L*2*sizeof(float),"expert_global_scales");
                ok=ok&&load_allocation.adopt((void**)&eng->d_fp4m_gu[l],eng->d_fp4m_gu[l],
                                              (size_t)E*2*MI*(H/2),"expert_gate_up_nvfp4");
                ok=ok&&load_allocation.adopt((void**)&eng->d_fp4m_gusf[l],eng->d_fp4m_gusf[l],
                                              (size_t)E*eng->fp4m_gu_sfB,"expert_gate_up_scales");
                ok=ok&&load_allocation.adopt((void**)&eng->d_fp4m_dn[l],eng->d_fp4m_dn[l],
                                              (size_t)E*H*(MI/2),"expert_down_nvfp4");
                ok=ok&&load_allocation.adopt((void**)&eng->d_fp4m_dnsf[l],eng->d_fp4m_dnsf[l],
                                              (size_t)E*eng->fp4m_dn_sfB,"expert_down_scales");
            }
            if(!LO.mixed_nvfp4() && !fp4_mode && q4k_mode){
                for(int p=0; p<3 && ok; p++){
                    int od = (p==2)? H : MI, idm = (p==2)? MI : H;
                    std::vector<const uint8_t*>  wpv(E, nullptr);
                    std::vector<const uint16_t*> spv(E, nullptr);
                    for(int e=0; e<E && ok; e++){
                        char buf[96]; snprintf(buf,sizeof(buf),"mlp.experts.%d.%s.weight",e,EX[p].proj);
                        const st::Tensor *w=find(lk(l,buf));
                        snprintf(buf,sizeof(buf),"mlp.experts.%d.%s.weight_scale_inv",e,EX[p].proj);
                        const st::Tensor *sct=find(lk(l,buf));
                        if(!w||!sct){ ok=false; break; }
                        wpv[e]=(const uint8_t*)w->data; spv[e]=(const uint16_t*)sct->data;
                    }
                    if(!ok) break;
                    unsigned char *slab=q35_fp8_experts_to_q4k(wpv.data(), spv.data(), E, od, idm);
                    if(!slab){ ok=false; break; }
                    tofree.push_back(slab);
                    std::string slab_name=std::string("mlp.experts.")+EX[p].proj+".q4k_slab";
                    D.push_back({EX[p].woff,nullptr,slab,(size_t)E*((size_t)od*idm/256)*144,-1,nullptr,0,
                                 lk(l,slab_name.c_str())});
                }
                if(ok){
                    eng->moe_experts_q4k = 1;
                    eng->moe_gate_slab     = (int64_t)((size_t)MI*H/256)*144;
                    eng->moe_up_slab       = eng->moe_gate_slab;
                    eng->moe_down_slab_q4k = (int64_t)((size_t)H*MI/256)*144;
                }
            } else if(!LO.mixed_nvfp4() && !fp4_mode)
            for(int p=0; p<3 && ok; p++){
                for(int e=0; e<E && ok; e++){
                    char buf[96]; snprintf(buf,sizeof(buf),"mlp.experts.%d.%s.weight",e,EX[p].proj);
                    ok=ok && put_raw(e==0?EX[p].woff:&q35moe_off_sink, lk(l,buf), wb);
                }
                for(int e=0; e<E && ok; e++){
                    char buf[96]; snprintf(buf,sizeof(buf),"mlp.experts.%d.%s.weight_scale_inv",e,EX[p].proj);
                    ok=ok && put_raw(e==0?EX[p].soff:&q35moe_off_sink, lk(l,buf), sb);
                }
            }
        } else {
            ok=ok && put_w(&T.ffn_gate,&T.fmt_gate, lk(l,"mlp.gate_proj.weight"), I, H);
            ok=ok && put_w(&T.ffn_up,  &T.fmt_up,   lk(l,"mlp.up_proj.weight"),   I, H);
            ok=ok && put_w(&T.ffn_down,&T.fmt_down, lk(l,"mlp.down_proj.weight"), H, I);
        }
        if(qwen35fp8::is_full(LO,l)){
            ok=ok && put_w(&T.attn_q,&T.fmt_q, lk(l,"self_attn.q_proj.weight"), 2*NQ*HD, H);
            ok=ok && put_w(&T.attn_k,&T.fmt_k, lk(l,"self_attn.k_proj.weight"), NKV*HD, H);
            ok=ok && put_w(&T.attn_v,&T.fmt_v, lk(l,"self_attn.v_proj.weight"), NKV*HD, H);
            ok=ok && put_w(&T.attn_output,&T.fmt_o, lk(l,"self_attn.o_proj.weight"), H, NQ*HD);
            ok=ok && put_f32_from_bf16(&T.attn_q_norm, lk(l,"self_attn.q_norm.weight"), 1.0f,false);
            ok=ok && put_f32_from_bf16(&T.attn_k_norm, lk(l,"self_attn.k_norm.weight"), 1.0f,false);
        } else {
            ok=ok && put_w(&T.ssm.in_qkv,&T.ssm.fmt_in_qkv, lk(l,"linear_attn.in_proj_qkv.weight"), CONVD, H);
            ok=ok && put_w(&T.ssm.in_z,  &T.ssm.fmt_in_z,   lk(l,"linear_attn.in_proj_z.weight"),   INNER, H);
            ok=ok && put_w(&T.ssm.out,   &T.ssm.fmt_out,    lk(l,"linear_attn.out_proj.weight"),    H, INNER);
            // in_a/in_b (alpha/beta, out=32) stay f32 like the torch oracle (m2_gemm): they feed the
            // GDN decay/beta and the recurrent state compounds Q8_0 error over decode steps.
            ok=ok && put_f32_from_bf16(&T.ssm.in_a, lk(l,"linear_attn.in_proj_a.weight"), 0.0f,false);
            ok=ok && put_f32_from_bf16(&T.ssm.in_b, lk(l,"linear_attn.in_proj_b.weight"), 0.0f,false);
            ok=ok && put_f32_from_bf16(&T.ssm.conv1d,  lk(l,"linear_attn.conv1d.weight"), 0.0f,false);
            ok=ok && put_f32_from_bf16(&T.ssm.dt_bias, lk(l,"linear_attn.dt_bias"),       0.0f,false);
            ok=ok && put_f32(&T.ssm.norm,  lk(l,"linear_attn.norm.weight"), false);
            // ssm_a = -exp(A_log): m2_decay_beta computes g = ssm_a·softplus(ac+dt_bias), and the
            // decay is masked at p0 (S=0), so a wrong sign/scale here only shows from the 2nd token.
            ok=ok && put_f32(&T.ssm.a_log, lk(l,"linear_attn.A_log"),       true);
        }
    }
    if(ok && q35_enable_ssd_expert_stream(eng,L,E,MI,H)!=0) ok=false;
    // globals: output_norm (f32, +1), lm_head → Q8_0
    ok=ok && put_f32_from_bf16(&TS.output_norm, LO.final_norm_key, 1.0f,false);
    if(precision_policy.enabled)
        fprintf(stderr,"fucina: precision policy applied from %s (core FP8=%d, Q4_K=%d, routed experts=%s)\n",
                precision_policy.path.c_str(),precision_policy.kept_fp8,precision_policy.requant_q4k,
                fp4_mode?"NVFP4":"fallback");
    if(!ok){ for(void*p:tofree) free(p); return -2; }
    // At this point the persistent expert slabs are resident, while the descriptor-backed core
    // weights are still host-side. This checkpoint separates expert setup from the bulk upload.
    cudaMemGetInfo(&fill_free_after_experts, &fill_total);

    // The immutable plan is now authoritative for core/scales layout. Descriptors are uploaded at
    // their planned offsets; an accounting mismatch fails before the core cudaMalloc.
    ModelPlan model_plan; std::vector<uint32_t> plan_ids; plan_ids.reserve(D.size());
    std::string plan_error;
    for(const auto &d:D){
        PlannedTensor p; p.source.logical_name=d.logical_name; p.source.source_name=d.logical_name;
        p.source.dtype=d.fmt==FORMAT_FP8_BLOCK?"F8_E4M3":(d.fmt==FORMAT_Q4_K?"F8_E4M3":"host");
        p.source.bytes=d.bytes;
        p.transform=d.fmt==FORMAT_Q4_K?TensorTransform::FP8_TO_Q4K:TensorTransform::COPY;
        p.destination=d.fmt==FORMAT_FP8_BLOCK?WeightEncoding::FP8_BLOCK_128:
                      (d.fmt==FORMAT_Q4_K?WeightEncoding::Q4_K:WeightEncoding::F32);
        p.arena=AllocationClass::CORE_WEIGHTS; p.consumer="qwen_projection_or_metadata";
        p.bytes=d.bytes; p.alignment=32;
        plan_ids.push_back((uint32_t)model_plan.tensors().size());
        if(!model_plan.add(std::move(p),plan_error)){ fprintf(stderr,"qwen35 model plan: %s\n",plan_error.c_str()); for(void*x:tofree)free(x); return -2; }
        if(d.scale_host){
            PlannedTensor s; s.source.logical_name=d.logical_name+".runtime_scale";
            s.source.source_name=s.source.logical_name; s.source.dtype="BF16"; s.source.bytes=d.scale_bytes;
            s.transform=TensorTransform::COPY; s.destination=WeightEncoding::BF16;
            s.arena=AllocationClass::SCALES; s.consumer="fp8_block_dispatch";
            s.bytes=d.scale_bytes; s.alignment=1;
            if(!model_plan.add(std::move(s),plan_error)){ fprintf(stderr,"qwen35 model plan: %s\n",plan_error.c_str()); for(void*x:tofree)free(x); return -2; }
        }
    }
    if(moe&&eng->moe_experts_fp4){
        for(int l=0;l<L;l++){
            const std::string sample=lk(l,"mlp.experts.0.gate_proj.weight");
            q35_source_kind sk; if(!q35_validate_weight_source(M,LO,sample,MI,H,&sk,plan_error)) return -2;
            const TensorTransform tr=sk==q35_source_kind::NVFP4?TensorTransform::NVFP4_REBASE:TensorTransform::FP8_TO_NVFP4;
            const char *dtype=sk==q35_source_kind::NVFP4?"NVFP4":"F8_E4M3";
            auto add_expert=[&](const char *suffix,size_t bytes,WeightEncoding dst,AllocationClass arena){
                PlannedTensor p; p.source.logical_name=lk(l,suffix); p.source.source_name=sample;
                p.source.dtype=dtype; p.source.bytes=bytes; p.transform=tr; p.destination=dst;
                p.arena=arena; p.consumer="grouped_expert_decode"; p.bytes=bytes; p.alignment=1;
                return model_plan.add(std::move(p),plan_error);
            };
            if((eng->d_fp4m_gu[l]&&!add_expert("mlp.experts.gate_up.nvfp4",(size_t)E*MI*H,WeightEncoding::NVFP4_LINEAR,AllocationClass::EXPERT_SLABS)) ||
               (eng->d_fp4m_gusf[l]&&!add_expert("mlp.experts.gate_up.scale",(size_t)E*eng->fp4m_gu_sfB,WeightEncoding::FP8_ROW,AllocationClass::EXPERT_SLABS)) ||
               (eng->d_fp4m_dn[l]&&!add_expert("mlp.experts.down.nvfp4",(size_t)E*H*(MI/2),WeightEncoding::NVFP4_LINEAR,AllocationClass::EXPERT_SLABS)) ||
               (eng->d_fp4m_dnsf[l]&&!add_expert("mlp.experts.down.scale",(size_t)E*eng->fp4m_dn_sfB,WeightEncoding::FP8_ROW,AllocationClass::EXPERT_SLABS))){
                fprintf(stderr,"qwen35 model plan: %s\n",plan_error.c_str()); return -2;
            }
        }
        PlannedTensor gs; gs.source.logical_name="mlp.experts.global_scales";
        gs.source.source_name=gs.source.logical_name; gs.source.dtype="F32"; gs.source.bytes=(size_t)L*2*sizeof(float);
        gs.transform=TensorTransform::COPY; gs.destination=WeightEncoding::F32;
        gs.arena=AllocationClass::EXPERT_SLABS; gs.consumer="grouped_expert_decode";
        gs.bytes=gs.source.bytes; gs.alignment=1;
        if(!model_plan.add(std::move(gs),plan_error)){ fprintf(stderr,"qwen35 model plan: %s\n",plan_error.c_str()); return -2; }
    }
    const size_t embed_elems=(size_t)VOC*H;
    auto add_repr=[&](const char *logical,const std::string &source,const char *dtype,
                      TensorTransform tr,WeightEncoding dst,AllocationClass arena,
                      const char *consumer,size_t bytes){
        PlannedTensor p; p.source.logical_name=logical; p.source.source_name=source;
        p.source.dtype=dtype; p.source.bytes=bytes; p.transform=tr; p.destination=dst;
        p.arena=arena; p.consumer=consumer; p.bytes=bytes; p.alignment=1;
        return model_plan.add(std::move(p),plan_error);
    };
    if(!add_repr("token_embedding.q8",LO.embed_key,"BF16",TensorTransform::BF16_TO_Q8_0,
                 WeightEncoding::Q8_0,AllocationClass::EMBEDDING_HEAD,"compat_embedding",(embed_elems/32)*34) ||
       !add_repr("token_embedding.bf16",LO.embed_key,"BF16",TensorTransform::COPY,
                 WeightEncoding::BF16,AllocationClass::EMBEDDING_HEAD,"decode_embedding",embed_elems*2) ||
       !add_repr("token_embedding.f32",LO.embed_key,"BF16",TensorTransform::BF16_TO_F32,
                 WeightEncoding::F32,AllocationClass::EMBEDDING_HEAD,"oracle_embedding",embed_elems*4) ||
       !add_repr("lm_head.bf16",LO.lmhead_key,LO.mixed_nvfp4()?"quantized":"BF16",
                 LO.mixed_nvfp4()?TensorTransform::QUANT_TO_BF16:TensorTransform::COPY,
                 WeightEncoding::BF16,AllocationClass::EMBEDDING_HEAD,"exact_head_rescore",embed_elems*2) ||
       !add_repr("lm_head.q8",LO.lmhead_key,"BF16",TensorTransform::BF16_TO_Q8_0,
                 WeightEncoding::Q8_0,AllocationClass::EMBEDDING_HEAD,"approximate_head_search",(embed_elems/32)*34) ||
       !add_repr("lm_head.candidates","runtime","I32",TensorTransform::COPY,
                 WeightEncoding::F32,AllocationClass::EMBEDDING_HEAD,"exact_head_rescore",
                 (size_t)GEMMA4_MAX_SEQS*Q8HEAD_MAXCAND*sizeof(int)) ||
       !add_repr("lm_head.candidate_counts","runtime","I32",TensorTransform::COPY,
                 WeightEncoding::F32,AllocationClass::EMBEDDING_HEAD,"exact_head_rescore",
                 (size_t)GEMMA4_MAX_SEQS*sizeof(int)) ||
       !add_repr("logits","runtime","F32",TensorTransform::COPY,
                 WeightEncoding::F32,AllocationClass::WORKSPACE,"sampling",(size_t)VOC*sizeof(float))){
        fprintf(stderr,"qwen35 model plan: %s\n",plan_error.c_str()); for(void*x:tofree)free(x); return -2;
    }
    if(!model_plan.finalize(plan_error)){ fprintf(stderr,"qwen35 model plan: %s\n",plan_error.c_str()); for(void*x:tofree)free(x); return -2; }
    const size_t total=model_plan.bytes(AllocationClass::CORE_WEIGHTS);
    if(moe&&eng->moe_experts_fp4){
        for(int l=0;l<L;l++){
            auto bind_expert=[&](ExpertWeightRef &ref,void *data,void *scale,float *global,
                                 int out_dim,int in_dim,uint64_t expert_stride,uint64_t scale_stride){
                ref.weight.data=(const uint8_t*)data; ref.weight.scale=scale; ref.weight.global_scale=global;
                ref.expert_count=E; ref.weight.out_dim=out_dim; ref.weight.in_dim=in_dim;
                ref.weight_stride=(int64_t)expert_stride; ref.scale_stride=(int64_t)scale_stride;
                ref.weight.encoding=WeightEncoding::NVFP4_SWIZZLED;
                ref.weight.layout=TensorLayout::NVFP4_SCALE_SWIZZLED;
                ref.weight.flags=WEIGHT_FLAG_PRIMARY|WEIGHT_FLAG_PACKED|WEIGHT_FLAG_GROUPED;
            };
            bind_expert(eng->ref_fp4m_gu[l],eng->d_fp4m_gu[l],eng->d_fp4m_gusf[l],
                        eng->d_fp4m_gsw+2*l,2*MI,H,(uint64_t)2*MI*(H/2),eng->fp4m_gu_sfB);
            bind_expert(eng->ref_fp4m_dn[l],eng->d_fp4m_dn[l],eng->d_fp4m_dnsf[l],
                        eng->d_fp4m_gsw+2*l+1,H,MI,(uint64_t)H*(MI/2),eng->fp4m_dn_sfB);
        }
    }
    if(const char *path=getenv("FUCINA_TENSOR_PLAN_JSON")){
        const std::string json=model_plan.json(); FILE *f=fopen(path,"wb");
        if(f){ fwrite(json.data(),1,json.size(),f); fputc('\n',f); fclose(f); }
    }
    fprintf(stderr,"fucina: qwen35 tensor plan finalized: entries=%zu core=%.2f GiB scales=%.2f MiB\n",
            model_plan.tensors().size(),total/(1024.0*1024*1024),
            model_plan.bytes(AllocationClass::SCALES)/(1024.0*1024));
    if(!load_allocation.allocate((void**)&eng->d_weights,total,"qwen_core_weights")){
        for(void*p:tofree) free(p); return -3;
    }
    eng->gguf_size=total;
    // scale table (built alongside), then sorted by weight ptr for wscale_fp8's binary search.
    eng->q35.fp8_scale_tab=(qwen35_runtime_state::fp8_scent*)malloc(D.size()*sizeof(qwen35_runtime_state::fp8_scent));
    eng->q35.fp8_scale_n=0;
    for(size_t i=0;i<D.size();i++){
        auto &d=D[i]; const size_t run=model_plan.tensors()[plan_ids[i]].arena_offset;
        *d.off = run;                                     // tdata_start=0 → planned byte position
        if(d.fmtp) *d.fmtp=(uint8_t)d.fmt;
        if(!load_allocation.upload(eng->d_weights+run,d.host,d.bytes)){
            for(void*p:tofree) free(p); return -4;
        }
        if(d.fmt==FORMAT_FP8_BLOCK && d.scale_host){
            auto &entry=eng->q35.fp8_scale_tab[eng->q35.fp8_scale_n];
            entry.w=eng->d_weights+run; entry.s=nullptr;
            if(!load_allocation.allocate((void**)&entry.s,d.scale_bytes,"qwen_fp8_scale")){
                for(void*p:tofree) free(p); return -4;
            }
            if(!load_allocation.upload((void*)entry.s,d.scale_host,d.scale_bytes)){
                for(void*p:tofree) free(p); return -4;
            }
            eng->q35.fp8_scale_n++;
        }
    }
    // insertion-sort the scale table by weight ptr (small: one entry per FP8 proj) for wscale_fp8's
    // binary search. std::sort would pull in <algorithm>; this avoids the include.
    for(int i=1;i<eng->q35.fp8_scale_n;i++){ qwen35_runtime_state::fp8_scent key=eng->q35.fp8_scale_tab[i]; int j=i-1;
        while(j>=0 && eng->q35.fp8_scale_tab[j].w>key.w){ eng->q35.fp8_scale_tab[j+1]=eng->q35.fp8_scale_tab[j]; j--; }
        eng->q35.fp8_scale_tab[j+1]=key; }
    for(void*p:tofree) free(p);
    cudaMemGetInfo(&fill_free_after_core, &fill_total);

    // FUCINA_MOE_Q4K: repack the requanted Q4_K mixer projections in place (de-interleaved
    // superblocks → the coalesced-uint4 mmvq_q4_k_packed kernels, ~74% of BW on the GGUF path
    // vs the natural-layout GEMV this mode ran at). gemv_w / gemv_batched_w / the prefill
    // dequant all branch on use_packed_q4k, which checks the PER-TENSOR fmt — the Q4_K expert
    // slabs (fmt=-1 raw, consumed natural by dg_mmq + the tc-prefill dequant) are excluded.
    if(q4k_mode){
        size_t maxb=0; int nq4=0;
        for(auto&d:D) if(d.fmt==FORMAT_Q4_K){ if(d.bytes>maxb) maxb=d.bytes; nq4++; }
        uint8_t *tmp=NULL;
        if(nq4>0 && cudaMalloc(&tmp,maxb)==cudaSuccess){
            for(auto&d:D) if(d.fmt==FORMAT_Q4_K){
                uint64_t ns=d.bytes/144;
                uint8_t *dst=eng->d_weights + *d.off;   // tdata_start == 0
                repack_q4_k_kernel<<<(unsigned)((ns+255)/256),256>>>(dst,tmp,ns);
                cudaMemcpy(dst,tmp,d.bytes,cudaMemcpyDeviceToDevice);
            }
            cudaDeviceSynchronize();
            if(cudaGetLastError()==cudaSuccess){
                eng->q4k_packed=1;
                fprintf(stderr,"fucina: qwen35-Q4K mixer repacked in place (%d tensors → packed dp4a GEMV)\n",nq4);
            } else fprintf(stderr,"fucina: qwen35-Q4K repack error — natural-layout Q4_K kept\n");
            cudaFree(tmp);
        }
    }

    // Populate canonical mixer descriptors only after optional in-place Q4_K repacking, so layout
    // is authoritative before any runtime path can observe it. Compatibility offsets/fmt bytes stay
    // populated while the remaining projection families migrate.
    auto bind_ref=[&](WeightRef &ref,uint64_t off,uint8_t fmt,int out_dim,int in_dim){
        ref.data=eng->d_weights+off;
        ref.scale=(fmt==FORMAT_FP8_BLOCK)?(const void*)wscale_fp8(eng,ref.data):nullptr;
        ref.global_scale=nullptr;
        ref.out_dim=out_dim; ref.in_dim=in_dim;
        ref.encoding=(fmt==FORMAT_FP8_BLOCK)?WeightEncoding::FP8_BLOCK_128:
                     (fmt==FORMAT_Q4_K)?WeightEncoding::Q4_K:
                     (fmt==FORMAT_Q8_0)?WeightEncoding::Q8_0:
                     (fmt==FORMAT_Q4_0)?WeightEncoding::Q4_0:WeightEncoding::Q6_K;
        ref.layout=(fmt==FORMAT_Q4_K && eng->q4k_packed)?TensorLayout::Q4K_PACKED:
                   (fmt==FORMAT_FP8_BLOCK)?TensorLayout::ROW_MAJOR:TensorLayout::GGML_NATIVE;
        ref.flags=WEIGHT_FLAG_PRIMARY |
                  ((ref.layout==TensorLayout::Q4K_PACKED)?WEIGHT_FLAG_PACKED:0);
    };
    for(int l=0;l<L;l++){
        auto &T=TS.layers[l];
        if(qwen35fp8::is_full(LO,l)){
            bind_ref(T.ref_q,T.attn_q,T.fmt_q,2*NQ*HD,H);
            bind_ref(T.ref_k,T.attn_k,T.fmt_k,NKV*HD,H);
            bind_ref(T.ref_v,T.attn_v,T.fmt_v,NKV*HD,H);
            bind_ref(T.ref_o,T.attn_output,T.fmt_o,H,NQ*HD);
        } else {
            bind_ref(T.ssm.ref_in_qkv,T.ssm.in_qkv,T.ssm.fmt_in_qkv,CONVD,H);
            bind_ref(T.ssm.ref_in_z,T.ssm.in_z,T.ssm.fmt_in_z,INNER,H);
            bind_ref(T.ssm.ref_out,T.ssm.out,T.ssm.fmt_out,H,INNER);
        }
        if(!moe){
            bind_ref(T.ref_gate,T.ffn_gate,T.fmt_gate,I,H);
            bind_ref(T.ref_up,T.ffn_up,T.fmt_up,I,H);
            bind_ref(T.ref_down,T.ffn_down,T.fmt_down,H,I);
        } else if(!eng->moe_experts_fp4){
            auto bind_expert=[&](ExpertWeightRef &ref,uint64_t off,uint64_t scale_off,
                                 int out_dim,int in_dim,int64_t weight_stride,int64_t scale_stride){
                ref.weight.data=eng->d_weights+off;
                ref.weight.scale=scale_off?(const void*)(eng->d_weights+scale_off):nullptr;
                ref.weight.global_scale=nullptr;
                ref.weight.out_dim=out_dim; ref.weight.in_dim=in_dim;
                ref.weight.encoding=eng->moe_experts_q4k?WeightEncoding::Q4_K:WeightEncoding::FP8_BLOCK_128;
                ref.weight.layout=eng->moe_experts_q4k?TensorLayout::GGML_NATIVE:TensorLayout::ROW_MAJOR;
                ref.weight.flags=WEIGHT_FLAG_PRIMARY|WEIGHT_FLAG_GROUPED;
                ref.expert_count=E; ref.weight_stride=weight_stride; ref.scale_stride=scale_stride;
            };
            const int64_t wstride=(int64_t)MI*H;
            const int64_t sstride=(int64_t)((MI+127)/128)*(H/128)*sizeof(__nv_bfloat16);
            bind_expert(eng->ref_moe_gate[l],eng->moe_gate_exps[l],eng->moe_gate_scales[l],MI,H,
                        eng->moe_experts_q4k?eng->moe_gate_slab:wstride,sstride);
            bind_expert(eng->ref_moe_up[l],eng->moe_up_exps[l],eng->moe_up_scales[l],MI,H,
                        eng->moe_experts_q4k?eng->moe_up_slab:wstride,sstride);
            bind_expert(eng->ref_moe_down[l],eng->moe_down_exps[l],eng->moe_down_scales[l],H,MI,
                        eng->moe_experts_q4k?eng->moe_down_slab_q4k:wstride,sstride);
        }
    }

    auto tx_upload=[&](void **slot,const void *host,size_t bytes,const char *label){
        return load_allocation.allocate(slot,bytes,label) &&
               load_allocation.upload(*slot,host,bytes);
    };

    // token_embd → Q8_0 (separate d_token_embd; embed_w reads it).
    const st::Tensor *embT=M.find(LO.embed_key);
    if(!embT){ return -5; }
    { int64_t n=(int64_t)VOC*H; unsigned char*q8=q35_bf16_to_q8_0_host((const uint16_t*)embT->data,n);
      if(!q8) return -5;
      bool uploaded=tx_upload((void**)&eng->d_token_embd,q8,(size_t)(n/32)*34,"token_embedding_q8");
      free(q8); if(!uploaded) return -5; }
    // token_embd ALSO uploaded BF16 (d_embed_bf16); embed_w prefers it for FP8 (per-step input
    // precision matters for the recurrent GDN state). d_token_embd (Q8_0) stays for the != NULL gate.
    if(!tx_upload((void**)&eng->d_embed_bf16,embT->data,embT->nbytes,"token_embedding_bf16")) return -5;
    if(!load_allocation.allocate((void**)&eng->d_embed_f32,embed_elems*sizeof(float),"token_embedding_f32")) return -5;
    q35fp8_bf16_to_f32_kernel<<<(unsigned)((embed_elems+255)/256),256>>>(
        eng->d_embed_f32,eng->d_embed_bf16,embed_elems,0.0f);
    // lm_head → BF16 (untied): the logits GEMV rides bf16_head_gemv_launch for the FP8 path (the
    // 248320-vocab argmax is Q8_0-sensitive). Uploaded verbatim from the checkpoint's BF16 head.
    { const st::Tensor *lmT=M.find(LO.lmhead_key);
      if(!lmT){ fprintf(stderr,"qwen35_fp8_engine: missing lm_head %s\n",LO.lmhead_key.c_str()); return -5; }
      // Mixed checkpoints quantize lm_head either per-tensor FP8 (Unsloth) or native NVFP4
      // (ModelOpt). Normalize both into the existing BF16 + exact-Q8-rescore head stores.
      const uint16_t *lm_src=(const uint16_t*)lmT->data;
      uint16_t *lm_deq=nullptr;
      if(LO.mixed_nvfp4()){
          q35_native_nv4 n4=q35_native_nv4_resolve(M,LO,LO.lmhead_key);
          if(n4.packed&&n4.scale&&n4.global){
              lm_deq=q35_nvfp4_to_bf16_host((const uint8_t*)n4.packed->data,
                                            (const uint8_t*)n4.scale->data,n4.global_mul,VOC,H);
          } else if(lmT->dtype==st::Dtype::F8_E4M3){
              const st::Tensor *sc=M.find(LO.lmhead_key+"_scale");
              if(sc) lm_deq=q35_fp8_scaled_to_bf16_host((const uint8_t*)lmT->data,sc,VOC,H);
          }
          if(!lm_deq){ fprintf(stderr,"qwen35_fp8_engine: quantized lm_head missing scales\n"); return -5; }
          lm_src=lm_deq;
      }
      if(!tx_upload((void**)&eng->d_lmhead_bf16,lm_src,(size_t)VOC*H*2,"lm_head_bf16")){
          free(lm_deq); return -5;
      }
      // Q8_0 copy for the exact two-pass greedy head (approx scan; BF16 stays for the rescore).
      { int64_t n=(int64_t)VOC*H; unsigned char*q8=q35_bf16_to_q8_0_host(lm_src,n);
        if(!q8){ free(lm_deq); return -5; }
        bool uploaded=tx_upload((void**)&eng->d_lmhead_q8,q8,(size_t)(n/32)*34,"lm_head_q8");
        free(q8); if(!uploaded){ free(lm_deq); return -5; }
        if(!load_allocation.allocate((void**)&eng->d_head_cand,
                                     (size_t)GEMMA4_MAX_SEQS*Q8HEAD_MAXCAND*sizeof(int),"lm_head_candidates") ||
           !load_allocation.allocate((void**)&eng->d_head_cnt,
                                     (size_t)GEMMA4_MAX_SEQS*sizeof(int),"lm_head_candidate_counts")){
            free(lm_deq); return -5;
        } }
      free(lm_deq); }
    if(!load_allocation.allocate((void**)&eng->d_logits,(size_t)VOC*sizeof(float),"logits")) return -6;
    cudaMemGetInfo(&fill_free_before_scratch, &fill_total);
    if(moe && moe_alloc_scratch(eng)!=0) return -6;
    cudaMemGetInfo(&fill_free_after_scratch, &fill_total);
    cudaDeviceSynchronize();
    if(cudaGetLastError()!=cudaSuccess){ fprintf(stderr,"qwen35_fp8_engine: upload error\n"); return -7; }

    // Exact allocation ledger for the FP8/NVFP4 model-load path. Unlike the historical
    // free-at-start minus free-now estimate, these bytes cannot be perturbed by another process
    // allocating or releasing unified memory while the CPU-heavy expert requantization runs.
    const size_t scale_bytes=model_plan.bytes(AllocationClass::SCALES);
    const size_t expert_bytes=model_plan.bytes(AllocationClass::EXPERT_SLABS);
    const size_t embed_bytes=(embed_elems/32)*34 + embed_elems*2 + embed_elems*4;
    const size_t representation_bytes=model_plan.bytes(AllocationClass::EMBEDDING_HEAD);
    const size_t head_bytes=representation_bytes-embed_bytes;
    const size_t misc_bytes=model_plan.bytes(AllocationClass::WORKSPACE);
    const size_t ledger_bytes=expert_bytes + total + scale_bytes + embed_bytes + head_bytes +
                              eng->moe_scratch_bytes + misc_bytes;
    eng->q35.model_bytes=ledger_bytes;
    size_t fill_free_final=0; cudaMemGetInfo(&fill_free_final, &fill_total);
    const q35_host_meminfo host_mem_final=q35_read_host_meminfo();
    const double GiB=1024.0*1024.0*1024.0;
    auto delta_gib=[&](size_t a,size_t b){ return ((double)a-(double)b)/GiB; };
    const char *mixer_desc=precision_policy.enabled&&precision_policy.kept_fp8>0
        ? (precision_policy.requant_q4k>0?"mixed-FP8/Q4_K":"policy-FP8")
        : (q4k_mode?"Q4_K":"FP8");
    fprintf(stderr,
        "fucina: qwen35 allocation decision: source=%s mixer=%s experts=%s d_weights=%.2f GiB\n"
        "fucina: qwen35 allocation ledger: experts=%.2f core=%.2f scales=%.2f embed=%.2f "
        "head=%.2f moe-scratch=%.2f misc=%.2f total=%.2f GiB\n"
        "fucina: qwen35 free-memory trace: enter=%.2f after-experts=%.2f after-core=%.2f "
        "before-scratch=%.2f after-scratch=%.2f final=%.2f GiB\n"
        "fucina: qwen35 host-memory trace: free %.2f->%.2f available %.2f->%.2f "
        "cached %.2f->%.2f reclaimable %.2f->%.2f GiB\n"
        "fucina: qwen35 observed deltas: experts=%+.2f core=%+.2f embed/head=%+.2f "
        "moe-scratch=%+.2f fill-total=%+.2f ledger-residual=%+.2f GiB\n",
        LO.compressed?"compressed-NVFP4":(LO.modelopt?"ModelOpt-NVFP4":"block-FP8"),
        mixer_desc,
        !moe?"n/a":(eng->moe_experts_fp4?"NVFP4":(eng->moe_experts_q4k?"Q4_K":"FP8")),
        total/GiB,
        expert_bytes/GiB,total/GiB,scale_bytes/GiB,embed_bytes/GiB,head_bytes/GiB,
        eng->moe_scratch_bytes/GiB,misc_bytes/GiB,ledger_bytes/GiB,
        fill_free_before/GiB,fill_free_after_experts/GiB,fill_free_after_core/GiB,
        fill_free_before_scratch/GiB,fill_free_after_scratch/GiB,fill_free_final/GiB,
        host_mem_before.mem_free/GiB,host_mem_final.mem_free/GiB,
        host_mem_before.mem_available/GiB,host_mem_final.mem_available/GiB,
        host_mem_before.cached/GiB,host_mem_final.cached/GiB,
        host_mem_before.sreclaimable/GiB,host_mem_final.sreclaimable/GiB,
        delta_gib(fill_free_before,fill_free_after_experts),
        delta_gib(fill_free_after_experts,fill_free_after_core),
        delta_gib(fill_free_after_core,fill_free_before_scratch),
        delta_gib(fill_free_before_scratch,fill_free_after_scratch),
        delta_gib(fill_free_before,fill_free_final),
        delta_gib(fill_free_before,fill_free_final)-ledger_bytes/GiB);
    fprintf(stderr,"fucina: qwen35 %s%s served via batched engine (%d layers, vocab %d, H %d, NKV %d%s, %.2f GiB)\n",
            mixer_desc, moe?"-MoE":"-dense", L, VOC, H, NKV,
            moe?", E-experts slabbed":"", total/GiB);
    if(!load_allocation.commit(*eng->device_allocations)) return -8;
    return 0;
}

extern "C" void *qwen35_fp8_load(const char *path) {
    st::Model M; std::string err;
    if (!M.open(path, err)) { fprintf(stderr, "qwen35_fp8_load: open: %s\n", err.c_str()); return nullptr; }
    qwen35fp8::Layout LO;
    if (!qwen35fp8::detect(M, LO, err)) { fprintf(stderr, "qwen35_fp8_load: detect: %s\n", err.c_str()); return nullptr; }

    const int H = M2_H, HD = M2_HEAD, NQ = M2_NQ, NKV = M2_NKV;
    const int CONVD = M2_CONVDIM, INNER = M2_VALD, TSR = M2_TSR, SD = M2_SD, CK = M2_CK;
    const st::Tensor *embT = M.find(LO.embed_key);
    const st::Tensor *gateT = M.find(qwen35fp8::lkey(LO, 0, "mlp.gate_proj.weight"));
    if (!embT || !gateT) { fprintf(stderr, "qwen35_fp8_load: missing embed/gate_proj\n"); return nullptr; }
    int VOC = (int)embT->shape[0];
    int I   = (int)gateT->shape[0];

    q35fp8_model *m = new q35fp8_model();
    memset(m, 0, sizeof(*m));
    m->stream = 0;
    m->n_layers = LO.n_layers; m->vocab = VOC; m->inter = I;
    gemma4_model_config_t *c = &m->cfg;
    c->arch = GEMMA4_ARCH_QWEN3_5;
    c->n_layers = LO.n_layers; c->hidden_size = H; c->head_dim = HD;
    c->n_heads = NQ; c->n_kv_global = NKV; c->intermediate = I; c->vocab_size = VOC;
    c->rotary_dim = M2_ROT; c->full_attention_interval = LO.full_attention_interval;
    c->ssm_state_size = SD; c->ssm_conv_kernel = CK; c->ssm_inner_size = INNER;
    c->ssm_group_count = M2_NKH; c->ssm_time_step_rank = TSR;
    for (int l = 0; l < LO.n_layers; l++)
        c->attn_kind[l] = qwen35fp8::is_full(LO, l) ? GEMMA4_ATTN_FULL : GEMMA4_ATTN_LINEAR;

    bool ok = true;
    m->embed    = q35fp8_up_bf16_f32(embT, 0.0f);
    m->out_norm = q35fp8_up_bf16_f32(M.find(LO.final_norm_key), 1.0f);   // +1
    m->lm_head  = q35fp8_up_bf16_f32(M.find(LO.lmhead_key), 0.0f);
    if (cudaMalloc(&m->d_logits, (size_t)VOC * sizeof(float)) != cudaSuccess) ok = false;
    ok = ok && m->embed && m->out_norm && m->lm_head;

    for (int l = 0; l < LO.n_layers && ok; l++) {
        q35fp8_layer &T = m->L[l];
        T.is_full = qwen35fp8::is_full(LO, l);
        T.in_norm   = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "input_layernorm.weight")), 1.0f);
        T.post_norm = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "post_attention_layernorm.weight")), 1.0f);
        ok = ok && T.in_norm && T.post_norm;
        // SwiGLU MLP (every layer)
        ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "mlp.gate_proj.weight"), T.gate, I, H);
        ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "mlp.up_proj.weight"),   T.up,   I, H);
        ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "mlp.down_proj.weight"), T.down, H, I);
        if (T.is_full) {
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "self_attn.q_proj.weight"), T.q, 2*NQ*HD, H);
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "self_attn.k_proj.weight"), T.k, NKV*HD, H);
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "self_attn.v_proj.weight"), T.v, NKV*HD, H);
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "self_attn.o_proj.weight"), T.o, H, NQ*HD);
            T.q_norm = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "self_attn.q_norm.weight")), 1.0f);
            T.k_norm = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "self_attn.k_norm.weight")), 1.0f);
            ok = ok && T.q_norm && T.k_norm;
        } else {
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "linear_attn.in_proj_qkv.weight"), T.inqkv, CONVD, H);
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "linear_attn.in_proj_z.weight"),   T.inz,   INNER, H);
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "linear_attn.out_proj.weight"),    T.out,   H, INNER);
            T.ina      = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "linear_attn.in_proj_a.weight")), 0.0f);
            T.inb      = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "linear_attn.in_proj_b.weight")), 0.0f);
            T.conv1d   = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "linear_attn.conv1d.weight")), 0.0f);
            T.dt_bias  = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "linear_attn.dt_bias")), 0.0f);
            T.ssm_norm = q35fp8_up_f32(M.find(qwen35fp8::lkey(LO, l, "linear_attn.norm.weight")));   // gated: NO +1
            T.a_coef   = q35fp8_up_f32(M.find(qwen35fp8::lkey(LO, l, "linear_attn.A_log")));
            if (T.a_coef) q35fp8_neg_exp_kernel<<<(unsigned)((TSR + 255) / 256), 256>>>(T.a_coef, TSR);
            ok = ok && T.ina && T.inb && T.conv1d && T.dt_bias && T.ssm_norm && T.a_coef;
        }
    }
    // ── M6: optional single-MTP draft head (mtp.*). Best-effort: a checkpoint without the head
    //    (or a partial one) simply leaves m->mtp.loaded=0 — the M5 forward is unaffected. ──
    {
        q35fp8_mtp &P = m->mtp;
        auto MT = [&](const char *k){ return std::string("mtp.") + k; };
        bool mok = true;
        P.pre_e     = q35fp8_up_bf16_f32(M.find(MT("pre_fc_norm_embedding.weight")), 1.0f);
        P.pre_h     = q35fp8_up_bf16_f32(M.find(MT("pre_fc_norm_hidden.weight")),    1.0f);
        P.fc        = q35fp8_up_bf16_f32(M.find(MT("fc.weight")),                    0.0f);  // [H,2H]
        P.in_norm   = q35fp8_up_bf16_f32(M.find(MT("layers.0.input_layernorm.weight")),          1.0f);
        P.post_norm = q35fp8_up_bf16_f32(M.find(MT("layers.0.post_attention_layernorm.weight")), 1.0f);
        P.fnorm     = q35fp8_up_bf16_f32(M.find(MT("norm.weight")),                  1.0f);
        P.q_norm    = q35fp8_up_bf16_f32(M.find(MT("layers.0.self_attn.q_norm.weight")), 1.0f);
        P.k_norm    = q35fp8_up_bf16_f32(M.find(MT("layers.0.self_attn.k_norm.weight")), 1.0f);
        mok = mok && P.pre_e && P.pre_h && P.fc && P.in_norm && P.post_norm && P.fnorm && P.q_norm && P.k_norm;
        mok = mok && q35fp8_load_proj(M, MT("layers.0.self_attn.q_proj.weight"), P.q, 2*NQ*HD, H);
        mok = mok && q35fp8_load_proj(M, MT("layers.0.self_attn.k_proj.weight"), P.k, NKV*HD,  H);
        mok = mok && q35fp8_load_proj(M, MT("layers.0.self_attn.v_proj.weight"), P.v, NKV*HD,  H);
        mok = mok && q35fp8_load_proj(M, MT("layers.0.self_attn.o_proj.weight"), P.o, H, NQ*HD);
        mok = mok && q35fp8_load_proj(M, MT("layers.0.mlp.gate_proj.weight"), P.gate, I, H);
        mok = mok && q35fp8_load_proj(M, MT("layers.0.mlp.up_proj.weight"),   P.up,   I, H);
        mok = mok && q35fp8_load_proj(M, MT("layers.0.mlp.down_proj.weight"), P.down, H, I);
        cudaDeviceSynchronize();
        if (cudaGetLastError() != cudaSuccess) mok = false;
        P.loaded = mok ? 1 : 0;
        fprintf(stderr, "qwen35_fp8_load: MTP draft head %s\n", mok ? "loaded (22 tensors)" : "absent/partial — spec disabled");
    }
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) ok = false;
    if (!ok) { fprintf(stderr, "qwen35_fp8_load: upload/alloc failed\n"); qwen35_fp8_free(m); return nullptr; }
    fprintf(stderr, "qwen35_fp8_load: loaded %d layers (vocab %d, inter %d) from %s\n",
            m->n_layers, VOC, I, path);
    return m;
}

extern "C" int qwen35_fp8_forward_greedy(void *model, const int32_t *in_ids, int n_prompt,
                                         int32_t *out_ids, int n_gen) {
    q35fp8_model *m = (q35fp8_model *)model;
    if (!m) return -1;
    const gemma4_model_config_t *c = &m->cfg;
    cudaStream_t st = m->stream;
    const float eps = M2_EPS;
    const int H = M2_H, HD = M2_HEAD, NQ = M2_NQ, NKV = M2_NKV;
    const int INNER = M2_VALD, CONVD = M2_CONVDIM, KEYD = M2_KEYD;
    const int NKH = M2_NKH, NVH = M2_NVH, SD = M2_SD, TSR = M2_TSR, ROT = M2_ROT;
    const int I = m->inter, VOC = m->vocab, L = m->n_layers;
    const int nsteps = n_prompt + n_gen - 1;
    (void)ROT; (void)KEYD;

    size_t smGDN = ((size_t)SD * SD + 3 * SD) * sizeof(float);
    if (cudaFuncSetAttribute(qwen35_fp8_gdn_step_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smGDN) != cudaSuccess) {
        fprintf(stderr, "qwen35_fp8_forward_greedy: GDN shared-mem opt-in failed (%zu B)\n", smGDN);
        return -4;
    }

    int rc = 0;
    int *d_arg = nullptr;
    float *x=nullptr,*xn=nullptr,*qg=nullptr,*qb=nullptr,*gate=nullptr,*kb=nullptr,*vb=nullptr,
          *attn=nullptr,*mix=nullptr,*qkv=nullptr,*conv_out=nullptr,*zc=nullptr,*ac=nullptr,
          *bc=nullptr,*gg=nullptr,*bb=nullptr,*qh=nullptr,*kh=nullptr,*vh=nullptr,*core=nullptr,
          *gnorm=nullptr,*ffn_g=nullptr,*ffn_u=nullptr,*ffn_a=nullptr;
    float **Kc = (float**)calloc(L, sizeof(float*));
    float **Vc = (float**)calloc(L, sizeof(float*));
    float **Sst = (float**)calloc(L, sizeof(float*));
    float **ring = (float**)calloc(L, sizeof(float*));
    if (!Kc || !Vc || !Sst || !ring) { rc = -5; goto cleanup; }

    #define CKM(p, n) do { if (cudaMalloc(&(p), (size_t)(n) * sizeof(float)) != cudaSuccess) { \
        fprintf(stderr, "qwen35_fp8_forward_greedy: cudaMalloc failed (%s)\n", #p); rc = -6; goto cleanup; } } while(0)
    if (cudaMalloc(&d_arg, sizeof(int)) != cudaSuccess) { rc = -6; goto cleanup; }
    CKM(x, H); CKM(xn, H); CKM(qg, 2*NQ*HD); CKM(qb, NQ*HD);
    CKM(gate, NQ*HD); CKM(kb, NKV*HD); CKM(vb, NKV*HD); CKM(attn, NQ*HD);
    CKM(mix, H); CKM(qkv, CONVD); CKM(conv_out, CONVD); CKM(zc, INNER);
    CKM(ac, TSR); CKM(bc, TSR); CKM(gg, TSR); CKM(bb, TSR);
    CKM(qh, NKH*SD); CKM(kh, NKH*SD); CKM(vh, NVH*SD); CKM(core, NVH*SD);
    CKM(gnorm, INNER); CKM(ffn_g, I); CKM(ffn_u, I); CKM(ffn_a, I);
    for (int l = 0; l < L; l++) {
        if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
            CKM(Kc[l], (size_t)nsteps*NKV*HD);
            CKM(Vc[l], (size_t)nsteps*NKV*HD);
        } else {
            CKM(Sst[l],  (size_t)NVH*SD*SD);
            CKM(ring[l], (size_t)CONVD*(M2_CK-1));
            cudaMemsetAsync(Sst[l],  0, (size_t)NVH*SD*SD*sizeof(float), st);
            cudaMemsetAsync(ring[l], 0, (size_t)CONVD*(M2_CK-1)*sizeof(float), st);
        }
    }

    for (int p = 0; p < nsteps; p++) {
        int32_t token = (p < n_prompt) ? in_ids[p] : out_ids[p - n_prompt];
        cudaMemcpyAsync(x, m->embed + (size_t)token * H, (size_t)H * sizeof(float),
                        cudaMemcpyDeviceToDevice, st);
        for (int l = 0; l < L; l++) {
            q35fp8_layer &T = m->L[l];
            rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(xn, x, T.in_norm, H, 1, eps);
            if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
                fp8_block_gemv_launch(qg, T.q.w, T.q.s, xn, H, 2*NQ*HD, st);
                fp8_block_gemv_launch(kb, T.k.w, T.k.s, xn, H, NKV*HD,  st);
                fp8_block_gemv_launch(vb, T.v.w, T.v.s, xn, H, NKV*HD,  st);
                m2_split_query_gate_kernel<<<(NQ*HD+255)/256,256,0,st>>>(qb, gate, qg, 1, NQ);
                per_head_rms_norm_rows_kernel<<<dim3(NQ,1),256,32*sizeof(float),st>>>(qb, T.q_norm, NQ, HD, 1, eps);
                per_head_rms_norm_rows_kernel<<<dim3(NKV,1),256,32*sizeof(float),st>>>(kb, T.k_norm, NKV, HD, 1, eps);
                qwen35_rope_pos_kernel<<<dim3((ROT/2+31)/32,NQ),32,0,st>>>(qb, NQ, p);
                qwen35_rope_pos_kernel<<<dim3((ROT/2+31)/32,NKV),32,0,st>>>(kb, NKV, p);
                cudaMemcpyAsync(Kc[l]+(size_t)p*NKV*HD, kb, (size_t)NKV*HD*sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(Vc[l]+(size_t)p*NKV*HD, vb, (size_t)NKV*HD*sizeof(float), cudaMemcpyDeviceToDevice, st);
                qwen35_attn_step_kernel<<<NQ,256,(size_t)(p+1)*sizeof(float),st>>>(attn, qb, Kc[l], Vc[l], p, M2_NKV);
                m2_sigmoid_gate_mul_kernel<<<(NQ*HD+255)/256,256,0,st>>>(attn, gate, NQ*HD);
                fp8_block_gemv_launch(mix, T.o.w, T.o.s, attn, NQ*HD, H, st);
            } else {
                fp8_block_gemv_launch(qkv, T.inqkv.w, T.inqkv.s, xn, H, CONVD, st);
                fp8_block_gemv_launch(zc,  T.inz.w,   T.inz.s,   xn, H, INNER, st);
                m2_gemm(ac, xn, T.ina, 1, H, TSR);   // in_proj_a → alpha (decay)
                m2_gemm(bc, xn, T.inb, 1, H, TSR);   // in_proj_b → beta
                qwen35_conv_step_kernel<<<(CONVD+127)/128,128,0,st>>>(conv_out, qkv, ring[l], T.conv1d, CONVD);
                cudaMemcpyAsync(qh, conv_out,        (size_t)KEYD*sizeof(float),  cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(kh, conv_out+KEYD,   (size_t)KEYD*sizeof(float),  cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(vh, conv_out+2*KEYD, (size_t)INNER*sizeof(float), cudaMemcpyDeviceToDevice, st);
                m2_l2norm_heads_kernel<<<dim3(NKH,1),128,0,st>>>(qh, NKH, SD, 1);
                m2_l2norm_heads_kernel<<<dim3(NKH,1),128,0,st>>>(kh, NKH, SD, 1);
                m2_decay_beta_kernel<<<(TSR+255)/256,256,0,st>>>(gg, bb, ac, bc, T.a_coef, T.dt_bias, 1, TSR);
                qwen35_fp8_gdn_step_kernel<<<NVH,128,smGDN,st>>>(core, qh, kh, vh, gg, bb, Sst[l]);
                m2_gated_norm_kernel<<<dim3(NVH,1),128,0,st>>>(gnorm, core, zc, T.ssm_norm, 1, NVH, INNER);
                fp8_block_gemv_launch(mix, T.out.w, T.out.s, gnorm, INNER, H, st);
            }
            qwen35_add_kernel<<<(H+255)/256,256,0,st>>>(x, mix, H);
            rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(xn, x, T.post_norm, H, 1, eps);
            fp8_block_gemv_launch(ffn_g, T.gate.w, T.gate.s, xn, H, I, st);
            fp8_block_gemv_launch(ffn_u, T.up.w,   T.up.s,   xn, H, I, st);
            silu_glu_kernel<<<(I+255)/256,256,0,st>>>(ffn_a, ffn_g, ffn_u, I);
            fp8_block_gemv_launch(mix, T.down.w, T.down.s, ffn_a, I, H, st);
            qwen35_add_kernel<<<(H+255)/256,256,0,st>>>(x, mix, H);
        }
        if (p >= n_prompt - 1) {
            rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(xn, x, m->out_norm, H, 1, eps);
            m2_gemm(m->d_logits, xn, m->lm_head, 1, H, VOC);
            qwen35_argmax_kernel<<<1,256,0,st>>>(m->d_logits, VOC, d_arg);
            int argmax = 0;
            cudaMemcpyAsync(&argmax, d_arg, sizeof(int), cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
            out_ids[p - (n_prompt - 1)] = argmax;
        }
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { fprintf(stderr, "qwen35_fp8_forward_greedy: CUDA error at pos %d: %s\n",
                                        p, cudaGetErrorString(e)); rc = -7; goto cleanup; }
    }

cleanup:
    #undef CKM
    cudaStreamSynchronize(st);
    if (d_arg) cudaFree(d_arg);
    { float *bufs[] = {x,xn,qg,qb,gate,kb,vb,attn,mix,qkv,conv_out,zc,ac,bc,gg,bb,qh,kh,vh,core,gnorm,ffn_g,ffn_u,ffn_a};
      for (float *b : bufs) if (b) cudaFree(b); }
    if (Kc)  { for (int l=0;l<L;l++) if (Kc[l])  cudaFree(Kc[l]);  free(Kc); }
    if (Vc)  { for (int l=0;l<L;l++) if (Vc[l])  cudaFree(Vc[l]);  free(Vc); }
    if (Sst) { for (int l=0;l<L;l++) if (Sst[l]) cudaFree(Sst[l]); free(Sst); }
    if (ring){ for (int l=0;l<L;l++) if (ring[l])cudaFree(ring[l]);free(ring); }
    return rc;
}

extern "C" void qwen35_fp8_free(void *model) {
    q35fp8_model *m = (q35fp8_model *)model;
    if (!m) return;
    auto FW = [](q35fp8_proj &P){ if (P.w) cudaFree((void*)P.w); if (P.s) cudaFree((void*)P.s); };
    if (m->embed) cudaFree(m->embed);
    if (m->out_norm) cudaFree(m->out_norm);
    if (m->lm_head) cudaFree(m->lm_head);
    if (m->d_logits) cudaFree(m->d_logits);
    for (int l = 0; l < m->n_layers; l++) {
        q35fp8_layer &T = m->L[l];
        if (T.in_norm) cudaFree(T.in_norm); if (T.post_norm) cudaFree(T.post_norm);
        FW(T.gate); FW(T.up); FW(T.down);
        if (T.is_full) {
            FW(T.q); FW(T.k); FW(T.v); FW(T.o);
            if (T.q_norm) cudaFree(T.q_norm); if (T.k_norm) cudaFree(T.k_norm);
        } else {
            FW(T.inqkv); FW(T.inz); FW(T.out);
            if (T.ina) cudaFree(T.ina); if (T.inb) cudaFree(T.inb);
            if (T.conv1d) cudaFree(T.conv1d); if (T.dt_bias) cudaFree(T.dt_bias);
            if (T.ssm_norm) cudaFree(T.ssm_norm); if (T.a_coef) cudaFree(T.a_coef);
        }
    }
    if (m->mtp.loaded) {
        q35fp8_mtp &P = m->mtp;
        float *bf[] = {P.pre_e,P.pre_h,P.fc,P.in_norm,P.post_norm,P.fnorm,P.q_norm,P.k_norm};
        for (float *b : bf) if (b) cudaFree(b);
        FW(P.q); FW(P.k); FW(P.v); FW(P.o); FW(P.gate); FW(P.up); FW(P.down);
    }
    delete m;
}

// =========================================================================
// ─── M6: Qwen3.5 single-MTP draft head + LOSSLESS speculative decode ──────
// =========================================================================
// Self-contained on the M5 FP8 path (the only checkpoint that ships the mtp.* head — the GGUF
// conversion drops it). The backbone hybrid mixer is STATEFUL (per-V-head GDN recurrent state +
// conv ring + per-FULL-layer KV), so unlike a paged-softmax model the draft tokens cannot be
// verified in one wide parallel forward and rolled back. Instead the verify is a SEQUENTIAL
// stop-at-first-reject single-row decode: process the pending token (always 1 emit), and while the
// next draft matched the just-computed target argmax, keep stepping. Because we NEVER step a
// rejected draft, the recurrent state lands exactly at "after the accepted prefix" with NO rollback
// — and every emitted token is the real backbone greedy argmax, so the continuation is
// BIT-IDENTICAL to qwen35_fp8_forward_greedy (lossless). The MTP only changes how many backbone
// steps run per call; it never changes which token is emitted.
//
// MTP draft = chain-local: a fresh tiny KV cache per round, seeded with h_prev = the backbone's
// POST-final-norm hidden of the last committed token (vLLM Qwen3_5MultiTokenPredictor convention).
// h_prev already carries full context, so even draft-chain-only attention drafts well (torch ref:
// mean ~1.4 accepted/round). Faithful persistent-KV would lift that to ~2.8 but needs a prompt-wide
// MTP pass + rollback; chain-local is the lossless, accept>0, stable minimum the gate asks for.

#define Q35_MTP_DEPTH_DEFAULT 4

struct q35fp8_ctx {
    q35fp8_model *m; cudaStream_t st; int maxctx, L;
    // backbone single-token scratch (mirrors qwen35_fp8_forward_greedy locals)
    float *x,*xn,*qg,*qb,*gate,*kb,*vb,*attn,*mix,*qkv,*conv_out,*zc,*ac,*bc,*gg,*bb,
          *qh,*kh,*vh,*core,*gnorm,*ffn_g,*ffn_u,*ffn_a,*hbuf;
    int   *d_arg;
    float **Kc,**Vc,**Sst,**ring;             // persistent per-layer state
    // MTP scratch + chain-local KV
    float *me,*mcat,*mxfc,*mxn,*mqg,*mqb,*mgate,*mkb,*mvb,*mattn,*mmix,*mr2,*mxn2,*mxn3,
          *mffg,*mffu,*mffa,*mhid,*mKc,*mVc;
};

static void q35fp8_ctx_free(q35fp8_ctx *c) {
    if (!c) return;
    float *bb[] = {c->x,c->xn,c->qg,c->qb,c->gate,c->kb,c->vb,c->attn,c->mix,c->qkv,c->conv_out,
        c->zc,c->ac,c->bc,c->gg,c->bb,c->qh,c->kh,c->vh,c->core,c->gnorm,c->ffn_g,c->ffn_u,c->ffn_a,
        c->hbuf,c->me,c->mcat,c->mxfc,c->mxn,c->mqg,c->mqb,c->mgate,c->mkb,c->mvb,c->mattn,c->mmix,
        c->mr2,c->mxn2,c->mxn3,c->mffg,c->mffu,c->mffa,c->mhid,c->mKc,c->mVc};
    for (float *b : bb) if (b) cudaFree(b);
    if (c->d_arg) cudaFree(c->d_arg);
    if (c->Kc)  { for (int l=0;l<c->L;l++) if (c->Kc[l])  cudaFree(c->Kc[l]);  free(c->Kc); }
    if (c->Vc)  { for (int l=0;l<c->L;l++) if (c->Vc[l])  cudaFree(c->Vc[l]);  free(c->Vc); }
    if (c->Sst) { for (int l=0;l<c->L;l++) if (c->Sst[l]) cudaFree(c->Sst[l]); free(c->Sst); }
    if (c->ring){ for (int l=0;l<c->L;l++) if (c->ring[l])cudaFree(c->ring[l]);free(c->ring); }
}

static int q35fp8_ctx_init(q35fp8_ctx *c, q35fp8_model *m, int maxctx) {
    memset(c, 0, sizeof(*c));
    c->m = m; c->st = m->stream; c->maxctx = maxctx; c->L = m->n_layers;
    const int H=M2_H, HD=M2_HEAD, NQ=M2_NQ, NKV=M2_NKV, INNER=M2_VALD, CONVD=M2_CONVDIM,
              NKH=M2_NKH, NVH=M2_NVH, SD=M2_SD, TSR=M2_TSR, I=m->inter, Dm=GEMMA4_SPEC_MAX;
    bool ok = true;
    #define CKC(p,n) do { if (cudaMalloc(&(c->p),(size_t)(n)*sizeof(float))!=cudaSuccess) ok=false; } while(0)
    CKC(x,H); CKC(xn,H); CKC(qg,2*NQ*HD); CKC(qb,NQ*HD); CKC(gate,NQ*HD); CKC(kb,NKV*HD);
    CKC(vb,NKV*HD); CKC(attn,NQ*HD); CKC(mix,H); CKC(qkv,CONVD); CKC(conv_out,CONVD); CKC(zc,INNER);
    CKC(ac,TSR); CKC(bc,TSR); CKC(gg,TSR); CKC(bb,TSR); CKC(qh,NKH*SD); CKC(kh,NKH*SD);
    CKC(vh,NVH*SD); CKC(core,NVH*SD); CKC(gnorm,INNER); CKC(ffn_g,I); CKC(ffn_u,I); CKC(ffn_a,I);
    CKC(hbuf,H);
    CKC(me,H); CKC(mcat,2*H); CKC(mxfc,H); CKC(mxn,H); CKC(mqg,2*NQ*HD); CKC(mqb,NQ*HD);
    CKC(mgate,NQ*HD); CKC(mkb,NKV*HD); CKC(mvb,NKV*HD); CKC(mattn,NQ*HD); CKC(mmix,H); CKC(mr2,H);
    CKC(mxn2,H); CKC(mxn3,H); CKC(mffg,I); CKC(mffu,I); CKC(mffa,I); CKC(mhid,H);
    CKC(mKc,(size_t)Dm*NKV*HD); CKC(mVc,(size_t)Dm*NKV*HD);
    #undef CKC
    if (cudaMalloc(&c->d_arg, sizeof(int)) != cudaSuccess) ok = false;
    c->Kc=(float**)calloc(c->L,sizeof(float*)); c->Vc=(float**)calloc(c->L,sizeof(float*));
    c->Sst=(float**)calloc(c->L,sizeof(float*)); c->ring=(float**)calloc(c->L,sizeof(float*));
    if (!c->Kc||!c->Vc||!c->Sst||!c->ring) ok=false;
    for (int l=0; l<c->L && ok; l++) {
        if (m->cfg.attn_kind[l]==GEMMA4_ATTN_FULL) {
            if (cudaMalloc(&c->Kc[l],(size_t)maxctx*NKV*HD*sizeof(float))!=cudaSuccess) ok=false;
            if (cudaMalloc(&c->Vc[l],(size_t)maxctx*NKV*HD*sizeof(float))!=cudaSuccess) ok=false;
        } else {
            if (cudaMalloc(&c->Sst[l], (size_t)NVH*SD*SD*sizeof(float))!=cudaSuccess) ok=false;
            if (cudaMalloc(&c->ring[l],(size_t)CONVD*(M2_CK-1)*sizeof(float))!=cudaSuccess) ok=false;
            if (ok) { cudaMemsetAsync(c->Sst[l],0,(size_t)NVH*SD*SD*sizeof(float),c->st);
                      cudaMemsetAsync(c->ring[l],0,(size_t)CONVD*(M2_CK-1)*sizeof(float),c->st); }
        }
    }
    if (!ok) { q35fp8_ctx_free(c); return -1; }
    return 0;
}

// One backbone token at absolute position `pos` (advances Kc/Vc/Sst/ring in place). Bit-identical
// to one iteration of qwen35_fp8_forward_greedy's loop. want_hidden → ctx->hbuf = POST-final-norm
// hidden (h_prev for the MTP); returns the greedy argmax when want_argmax, else -1.
static int q35fp8_main_step(q35fp8_ctx *c, int32_t token, int pos, int want_hidden, int want_argmax) {
    q35fp8_model *m = c->m; cudaStream_t st = c->st;
    const gemma4_model_config_t *cfg = &m->cfg; const float eps = M2_EPS;
    const int H=M2_H, HD=M2_HEAD, NQ=M2_NQ, NKV=M2_NKV, INNER=M2_VALD, CONVD=M2_CONVDIM,
              KEYD=M2_KEYD, NKH=M2_NKH, NVH=M2_NVH, SD=M2_SD, TSR=M2_TSR, ROT=M2_ROT,
              I=m->inter, VOC=m->vocab, L=c->L;
    size_t smGDN = ((size_t)SD*SD + 3*SD)*sizeof(float);
    cudaMemcpyAsync(c->x, m->embed + (size_t)token*H, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, st);
    for (int l=0; l<L; l++) {
        q35fp8_layer &T = m->L[l];
        rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(c->xn, c->x, T.in_norm, H, 1, eps);
        if (cfg->attn_kind[l]==GEMMA4_ATTN_FULL) {
            fp8_block_gemv_launch(c->qg, T.q.w, T.q.s, c->xn, H, 2*NQ*HD, st);
            fp8_block_gemv_launch(c->kb, T.k.w, T.k.s, c->xn, H, NKV*HD, st);
            fp8_block_gemv_launch(c->vb, T.v.w, T.v.s, c->xn, H, NKV*HD, st);
            m2_split_query_gate_kernel<<<(NQ*HD+255)/256,256,0,st>>>(c->qb, c->gate, c->qg, 1, NQ);
            per_head_rms_norm_rows_kernel<<<dim3(NQ,1),256,32*sizeof(float),st>>>(c->qb, T.q_norm, NQ, HD, 1, eps);
            per_head_rms_norm_rows_kernel<<<dim3(NKV,1),256,32*sizeof(float),st>>>(c->kb, T.k_norm, NKV, HD, 1, eps);
            qwen35_rope_pos_kernel<<<dim3((ROT/2+31)/32,NQ),32,0,st>>>(c->qb, NQ, pos);
            qwen35_rope_pos_kernel<<<dim3((ROT/2+31)/32,NKV),32,0,st>>>(c->kb, NKV, pos);
            cudaMemcpyAsync(c->Kc[l]+(size_t)pos*NKV*HD, c->kb, (size_t)NKV*HD*sizeof(float), cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(c->Vc[l]+(size_t)pos*NKV*HD, c->vb, (size_t)NKV*HD*sizeof(float), cudaMemcpyDeviceToDevice, st);
            qwen35_attn_step_kernel<<<NQ,256,(size_t)(pos+1)*sizeof(float),st>>>(c->attn, c->qb, c->Kc[l], c->Vc[l], pos, M2_NKV);
            m2_sigmoid_gate_mul_kernel<<<(NQ*HD+255)/256,256,0,st>>>(c->attn, c->gate, NQ*HD);
            fp8_block_gemv_launch(c->mix, T.o.w, T.o.s, c->attn, NQ*HD, H, st);
        } else {
            fp8_block_gemv_launch(c->qkv, T.inqkv.w, T.inqkv.s, c->xn, H, CONVD, st);
            fp8_block_gemv_launch(c->zc,  T.inz.w,   T.inz.s,   c->xn, H, INNER, st);
            m2_gemm(c->ac, c->xn, T.ina, 1, H, TSR);
            m2_gemm(c->bc, c->xn, T.inb, 1, H, TSR);
            qwen35_conv_step_kernel<<<(CONVD+127)/128,128,0,st>>>(c->conv_out, c->qkv, c->ring[l], T.conv1d, CONVD);
            cudaMemcpyAsync(c->qh, c->conv_out,        (size_t)KEYD*sizeof(float),  cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(c->kh, c->conv_out+KEYD,   (size_t)KEYD*sizeof(float),  cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(c->vh, c->conv_out+2*KEYD, (size_t)INNER*sizeof(float), cudaMemcpyDeviceToDevice, st);
            m2_l2norm_heads_kernel<<<dim3(NKH,1),128,0,st>>>(c->qh, NKH, SD, 1);
            m2_l2norm_heads_kernel<<<dim3(NKH,1),128,0,st>>>(c->kh, NKH, SD, 1);
            m2_decay_beta_kernel<<<(TSR+255)/256,256,0,st>>>(c->gg, c->bb, c->ac, c->bc, T.a_coef, T.dt_bias, 1, TSR);
            qwen35_fp8_gdn_step_kernel<<<NVH,128,smGDN,st>>>(c->core, c->qh, c->kh, c->vh, c->gg, c->bb, c->Sst[l]);
            m2_gated_norm_kernel<<<dim3(NVH,1),128,0,st>>>(c->gnorm, c->core, c->zc, T.ssm_norm, 1, NVH, INNER);
            fp8_block_gemv_launch(c->mix, T.out.w, T.out.s, c->gnorm, INNER, H, st);
        }
        qwen35_add_kernel<<<(H+255)/256,256,0,st>>>(c->x, c->mix, H);
        rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(c->xn, c->x, T.post_norm, H, 1, eps);
        fp8_block_gemv_launch(c->ffn_g, T.gate.w, T.gate.s, c->xn, H, I, st);
        fp8_block_gemv_launch(c->ffn_u, T.up.w,   T.up.s,   c->xn, H, I, st);
        silu_glu_kernel<<<(I+255)/256,256,0,st>>>(c->ffn_a, c->ffn_g, c->ffn_u, I);
        fp8_block_gemv_launch(c->mix, T.down.w, T.down.s, c->ffn_a, I, H, st);
        qwen35_add_kernel<<<(H+255)/256,256,0,st>>>(c->x, c->mix, H);
    }
    int argmax = -1;
    if (want_hidden || want_argmax) {
        rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(c->xn, c->x, m->out_norm, H, 1, eps);
        if (want_hidden)
            cudaMemcpyAsync(c->hbuf, c->xn, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, st);
        if (want_argmax) {
            m2_gemm(m->d_logits, c->xn, m->lm_head, 1, H, VOC);
            qwen35_argmax_kernel<<<1,256,0,st>>>(m->d_logits, VOC, c->d_arg);
            cudaMemcpyAsync(&argmax, c->d_arg, sizeof(int), cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
        }
    }
    return argmax;
}

// One MTP draft step (chain-local KV at index `idx`, absolute RoPE position `pos`). in_tok is the
// token to embed; h_prev[H] the previous hidden (backbone post-norm for idx 0, else the MTP
// residual stream of the prior depth). Writes the new MTP residual stream to ctx->mhid (re-fed as
// h_prev for the next depth) and returns the drafted argmax.
static int q35fp8_mtp_step(q35fp8_ctx *c, int32_t in_tok, const float *h_prev, int idx, int pos) {
    q35fp8_model *m = c->m; cudaStream_t st = c->st; q35fp8_mtp &P = m->mtp;
    const float eps = M2_EPS;
    const int H=M2_H, HD=M2_HEAD, NQ=M2_NQ, NKV=M2_NKV, ROT=M2_ROT, I=m->inter, VOC=m->vocab;
    // fc-fusion: cat[ norm_e(embed(in_tok)) | norm_h(h_prev) ] → fc → residual stream xfc
    cudaMemcpyAsync(c->me, m->embed + (size_t)in_tok*H, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, st);
    rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(c->mcat,     c->me,  P.pre_e, H, 1, eps);
    rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(c->mcat + H, h_prev, P.pre_h, H, 1, eps);
    m2_gemm(c->mxfc, c->mcat, P.fc, 1, 2*H, H);
    // one FULL-attn decoder layer (gated GQA + SwiGLU), residual = mxfc
    rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(c->mxn, c->mxfc, P.in_norm, H, 1, eps);
    fp8_block_gemv_launch(c->mqg, P.q.w, P.q.s, c->mxn, H, 2*NQ*HD, st);
    fp8_block_gemv_launch(c->mkb, P.k.w, P.k.s, c->mxn, H, NKV*HD, st);
    fp8_block_gemv_launch(c->mvb, P.v.w, P.v.s, c->mxn, H, NKV*HD, st);
    m2_split_query_gate_kernel<<<(NQ*HD+255)/256,256,0,st>>>(c->mqb, c->mgate, c->mqg, 1, NQ);
    per_head_rms_norm_rows_kernel<<<dim3(NQ,1),256,32*sizeof(float),st>>>(c->mqb, P.q_norm, NQ, HD, 1, eps);
    per_head_rms_norm_rows_kernel<<<dim3(NKV,1),256,32*sizeof(float),st>>>(c->mkb, P.k_norm, NKV, HD, 1, eps);
    qwen35_rope_pos_kernel<<<dim3((ROT/2+31)/32,NQ),32,0,st>>>(c->mqb, NQ, pos);
    qwen35_rope_pos_kernel<<<dim3((ROT/2+31)/32,NKV),32,0,st>>>(c->mkb, NKV, pos);
    cudaMemcpyAsync(c->mKc+(size_t)idx*NKV*HD, c->mkb, (size_t)NKV*HD*sizeof(float), cudaMemcpyDeviceToDevice, st);
    cudaMemcpyAsync(c->mVc+(size_t)idx*NKV*HD, c->mvb, (size_t)NKV*HD*sizeof(float), cudaMemcpyDeviceToDevice, st);
    qwen35_attn_step_kernel<<<NQ,256,(size_t)(idx+1)*sizeof(float),st>>>(c->mattn, c->mqb, c->mKc, c->mVc, idx, M2_NKV);
    m2_sigmoid_gate_mul_kernel<<<(NQ*HD+255)/256,256,0,st>>>(c->mattn, c->mgate, NQ*HD);
    fp8_block_gemv_launch(c->mmix, P.o.w, P.o.s, c->mattn, NQ*HD, H, st);
    cudaMemcpyAsync(c->mr2, c->mxfc, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, st);
    qwen35_add_kernel<<<(H+255)/256,256,0,st>>>(c->mr2, c->mmix, H);            // r2 = xfc + attn_out
    rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(c->mxn2, c->mr2, P.post_norm, H, 1, eps);
    fp8_block_gemv_launch(c->mffg, P.gate.w, P.gate.s, c->mxn2, H, I, st);
    fp8_block_gemv_launch(c->mffu, P.up.w,   P.up.s,   c->mxn2, H, I, st);
    silu_glu_kernel<<<(I+255)/256,256,0,st>>>(c->mffa, c->mffg, c->mffu, I);
    fp8_block_gemv_launch(c->mmix, P.down.w, P.down.s, c->mffa, I, H, st);
    qwen35_add_kernel<<<(H+255)/256,256,0,st>>>(c->mr2, c->mmix, H);            // r2 += mlp → MTP residual stream
    cudaMemcpyAsync(c->mhid, c->mr2, (size_t)H*sizeof(float), cudaMemcpyDeviceToDevice, st);
    rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(c->mxn3, c->mr2, P.fnorm, H, 1, eps);
    m2_gemm(m->d_logits, c->mxn3, m->lm_head, 1, H, VOC);
    qwen35_argmax_kernel<<<1,256,0,st>>>(m->d_logits, VOC, c->d_arg);
    int draft = -1;
    cudaMemcpyAsync(&draft, c->d_arg, sizeof(int), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    return draft;
}

// LOSSLESS speculative greedy decode. Emits EXACTLY the same out_ids[0..n_gen-1] as
// qwen35_fp8_forward_greedy (verified by the gate), while drafting with the MTP head. Writes the
// cumulative drafted/accepted counts (accept-rate proxy) when the pointers are non-NULL.
extern "C" int qwen35_fp8_spec_greedy(void *model, const int32_t *in_ids, int n_prompt,
                                      int32_t *out_ids, int n_gen, long *drafted_out, long *accepted_out) {
    q35fp8_model *m = (q35fp8_model *)model;
    if (!m || n_prompt <= 0 || n_gen <= 0) return -1;
    if (!m->mtp.loaded) { fprintf(stderr, "qwen35_fp8_spec_greedy: MTP head not loaded\n"); return -2; }
    static int depth = -1;
    if (depth < 0) { const char *e = getenv("FUCINA_QWEN35_MTP_K"); depth = e ? atoi(e) : Q35_MTP_DEPTH_DEFAULT; }
    if (depth < 1) depth = 1;
    if (depth > GEMMA4_SPEC_MAX - 1) depth = GEMMA4_SPEC_MAX - 1;

    size_t smGDN = ((size_t)M2_SD*M2_SD + 3*M2_SD)*sizeof(float);
    if (cudaFuncSetAttribute(qwen35_fp8_gdn_step_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smGDN) != cudaSuccess) return -4;

    q35fp8_ctx ctx;
    int maxctx = n_prompt + n_gen + 8;
    if (q35fp8_ctx_init(&ctx, m, maxctx) != 0) return -5;
    cudaStream_t st = ctx.st;
    int rc = 0; long drafted = 0, accepted = 0;
    int32_t draft[GEMMA4_SPEC_MAX];

    // ── Prefill: process the prompt, capturing h_prev (post-norm hidden of the last prompt token)
    //    and the first generated token N0 (= out_ids[0]). ──
    for (int p = 0; p < n_prompt - 1; p++)
        if (q35fp8_main_step(&ctx, in_ids[p], p, 0, 0) , cudaGetLastError() != cudaSuccess) { rc = -7; goto done; }
    {
        int N0 = q35fp8_main_step(&ctx, in_ids[n_prompt-1], n_prompt-1, /*want_hidden=*/1, /*want_argmax=*/1);
        if (N0 < 0 || cudaGetLastError() != cudaSuccess) { rc = -7; goto done; }
        out_ids[0] = N0;
    }

    // ── Spec decode: draft up to `depth`, verify SEQUENTIALLY (stop at first reject). ──
    {
        int emitted = 1;                 // out_ids[0] already produced
        int32_t pend = out_ids[0];       // pending token to process next
        int pos = n_prompt;              // absolute position of `pend`
        while (emitted < n_gen) {
            // 1) draft d_0..d_{D-1} from the chain (h_prev = backbone hidden of last committed token).
            int D = depth; if (D > n_gen - emitted) D = n_gen - emitted;   // never draft past the budget
            int32_t dtok = pend; const float *hprev = ctx.hbuf;
            for (int j = 0; j < D; j++) {
                int d = q35fp8_mtp_step(&ctx, dtok, hprev, j, pos + j);
                if (d < 0 || cudaGetLastError() != cudaSuccess) { rc = -8; goto done; }
                draft[j] = d; dtok = d; hprev = ctx.mhid;
            }
            drafted += D;
            // 2) verify: always emit the pending token's target; extend while drafts match.
            int t = q35fp8_main_step(&ctx, pend, pos, /*want_hidden=*/1, /*want_argmax=*/1);
            if (t < 0 || cudaGetLastError() != cudaSuccess) { rc = -7; goto done; }
            out_ids[emitted++] = t; pos++;
            int a = 0;
            while (a < D && t == draft[a] && emitted < n_gen) {
                // draft[a] was correct ⇒ step it (it equals t) and read the next target.
                pend = draft[a];
                t = q35fp8_main_step(&ctx, pend, pos, /*want_hidden=*/1, /*want_argmax=*/1);
                if (t < 0 || cudaGetLastError() != cudaSuccess) { rc = -7; goto done; }
                out_ids[emitted++] = t; pos++; a++;
            }
            accepted += a;
            pend = t;                    // next pending token = last target; h_prev = ctx.hbuf (fresh)
        }
    }

done:
    cudaStreamSynchronize(st);
    if (cudaGetLastError() != cudaSuccess && rc == 0) rc = -9;
    if (drafted_out)  *drafted_out  = drafted;
    if (accepted_out) *accepted_out = accepted;
    q35fp8_ctx_free(&ctx);
    return rc;
}


// =========================================================================
// ─── P6: Qwen3.5-35B-A3B MoE hybrid (qwen3_5_moe) FP8 reference path ──────
// =========================================================================
// Self-contained FP8 forward for the OFFICIAL Qwen3.5-35B-A3B-FP8 checkpoint. Same hybrid mixer as
// the 9B dense path (30 GDN gated-deltanet linear + 10 FULL output-gated softmax-GQA, period-4) but
// hidden 2048, 2 KV heads, and the dense SwiGLU MLP replaced by the Qwen3_5MoeSparseMoeBlock:
//   router  : logits = x·gate.weight.T ([256]); probs=softmax(logits); (w,idx)=topk(probs,8);
//             w /= w.sum()           (Qwen3_5MoeTopKRouter — softmax over all experts, then renorm)
//   experts : Σ_j w_j · down_e( silu(gate_e(x))·up_e(x) ),  e=idx_j,  moe_intermediate 512
//   shared  : sigmoid(x·shared_expert_gate.weight.T) · down_s(silu(gate_s(x))·up_s(x)), inter 512
//   ffn_out = experts + shared
// Distinct struct + device buffers from the 9B q35fp8 path → the dense 9B forward stays byte-
// identical. Every GDN/attn/norm kernel is reused verbatim (GDN geometry is hidden-independent);
// only the FULL-attn KV stride (NKV=2) needs an nkv-parameterized step kernel (q35moe_attn_step),
// and the FFN site is the new MoE block. Weights via the validated fp8_block_gemv (no dequant); the
// router gate + shared_expert_gate stay BF16 (modules_to_not_convert) → m2_gemm in F32.

// out[i] += scale * v[i]  (expert/shared accumulation into the MoE residual).
__global__ void q35moe_axpy_kernel(float *out, const float *v, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += scale * v[i];
}

// Host top-k router: select the 8 largest of router_logits[E] (softmax is monotone, so top-k by
// logit == top-k by prob; lowest-index tie-break, matching torch.topk), then the renormalized
// mixture weights = softmax restricted to the selected logits (Z cancels in p_i / Σ p_top8).
static void q35moe_route_host(const float *logits, int E, int topk, int *idx, float *w) {
    for (int j = 0; j < topk; j++) {
        int bi = -1; float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            int taken = 0;
            for (int t = 0; t < j; t++) if (idx[t] == e) { taken = 1; break; }
            if (taken) continue;
            if (logits[e] > bv) { bv = logits[e]; bi = e; }
        }
        idx[j] = bi;
    }
    float mx = -1e30f;
    for (int j = 0; j < topk; j++) mx = fmaxf(mx, logits[idx[j]]);
    float s = 0.f;
    for (int j = 0; j < topk; j++) { w[j] = expf(logits[idx[j]] - mx); s += w[j]; }
    for (int j = 0; j < topk; j++) w[j] /= s;
}

struct q35moe_layer {
    int is_full;
    q35fp8_proj q, k, v, o;                 // FULL: gated-q / k / v / o
    float *q_norm, *k_norm;                 // FULL: per-head RMSNorm gains (+1 baked) [HEAD]
    q35fp8_proj inqkv, inz, out;            // GDN: in_proj_qkv / in_proj_z / out_proj
    float *ina, *inb;                       // GDN: in_proj_a / in_proj_b  f32 [TSR][H]
    float *conv1d, *a_coef, *dt_bias, *ssm_norm;
    float *in_norm, *post_norm;             // shared: pre-mixer / pre-FFN RMSNorm (+1 baked) [H]
    // ── MoE FFN (Qwen3_5MoeSparseMoeBlock) ──
    float *router_w;                        // mlp.gate [E×H] f32 (BF16→f32, NO +1)
    q35fp8_proj *ex_gate, *ex_up, *ex_down; // arrays[E] FP8 block expert projections
    q35fp8_proj sh_gate, sh_up, sh_down;    // shared expert SwiGLU (FP8 block)
    float *sh_gate_w;                       // mlp.shared_expert_gate [H] f32 (BF16→f32)
};
struct q35moe_model {
    gemma4_model_config_t cfg;
    cudaStream_t stream;
    int n_layers, vocab, H, NKV, E, topk, moe_inter, shared_inter;
    float *embed, *out_norm, *lm_head, *d_logits;
    q35moe_layer L[GEMMA4_CAP_LAYERS];
};

extern "C" void qwen35_moe_fp8_free(void *model);

// Resolve an HF-cache root (models--Org--Name/, which holds the shards under snapshots/<hash>/) to
// the directory that actually carries model.safetensors.index.json. A path that already points at
// the shards (an index.json sibling, or a lone model.safetensors) is returned unchanged.
static std::string q35moe_resolve_dir(const char *path) {
    std::string p(path);
    struct stat sb;
    if (stat((p + "/model.safetensors.index.json").c_str(), &sb) == 0) return p;
    if (stat((p + "/model.safetensors").c_str(), &sb) == 0) return p;
    glob_t g; std::string out = p;
    if (glob((p + "/snapshots/*/model.safetensors.index.json").c_str(), 0, nullptr, &g) == 0 && g.gl_pathc > 0) {
        std::string f = g.gl_pathv[0];
        out = f.substr(0, f.rfind('/'));
    }
    globfree(&g);
    return out;
}

extern "C" void *qwen35_moe_fp8_load(const char *path) {
    st::Model M; std::string err;
    std::string dir = q35moe_resolve_dir(path);
    if (!M.open(dir.c_str(), err)) { fprintf(stderr, "qwen35_moe_fp8_load: open: %s\n", err.c_str()); return nullptr; }
    qwen35fp8::Layout LO;
    if (!qwen35fp8::detect(M, LO, err)) { fprintf(stderr, "qwen35_moe_fp8_load: detect: %s\n", err.c_str()); return nullptr; }

    // MoE checkpoint iff the per-layer sparse experts are present (the 9B dense path has mlp.gate_proj).
    if (!M.has(qwen35fp8::lkey(LO, 0, "mlp.experts.0.gate_proj.weight"))) {
        fprintf(stderr, "qwen35_moe_fp8_load: not a qwen3_5_moe checkpoint (no mlp.experts.*) — use qwen35_fp8_load\n");
        return nullptr;
    }

    const int HD = M2_HEAD, NQ = M2_NQ;
    const int CONVD = M2_CONVDIM, INNER = M2_VALD, TSR = M2_TSR, SD = M2_SD, CK = M2_CK;
    const st::Tensor *embT  = M.find(LO.embed_key);
    const st::Tensor *kT0   = M.find(qwen35fp8::lkey(LO, 3, "self_attn.k_proj.weight")); // layer 3 = FULL
    const st::Tensor *egT   = M.find(qwen35fp8::lkey(LO, 0, "mlp.experts.0.gate_proj.weight"));
    const st::Tensor *sgT   = M.find(qwen35fp8::lkey(LO, 0, "mlp.shared_expert.gate_proj.weight"));
    const st::Tensor *rgT   = M.find(qwen35fp8::lkey(LO, 0, "mlp.gate.weight"));
    if (!embT || !kT0 || !egT || !sgT || !rgT) {
        fprintf(stderr, "qwen35_moe_fp8_load: missing embed/k_proj/expert/shared/router tensors\n"); return nullptr;
    }
    const int VOC = (int)embT->shape[0];
    const int H   = (int)embT->shape[1];               // hidden 2048
    const int NKV = (int)kT0->shape[0] / HD;           // 512/256 = 2
    const int MI  = (int)egT->shape[0];                // moe_intermediate 512
    const int SI  = (int)sgT->shape[0];                // shared_expert_intermediate 512
    const int E   = (int)rgT->shape[0];                // num_experts 256
    long tk = 8; qwen35fp8::cfg_int(M.config_json(), "\"num_experts_per_tok\"", tk);
    const int TOPK = (tk > 0) ? (int)tk : 8;

    q35moe_model *m = new q35moe_model();
    memset(m, 0, sizeof(*m));
    m->stream = 0;
    m->n_layers = LO.n_layers; m->vocab = VOC; m->H = H; m->NKV = NKV;
    m->E = E; m->topk = TOPK; m->moe_inter = MI; m->shared_inter = SI;
    gemma4_model_config_t *c = &m->cfg;
    c->arch = GEMMA4_ARCH_QWEN3_5;
    c->n_layers = LO.n_layers; c->hidden_size = H; c->head_dim = HD;
    c->n_heads = NQ; c->n_kv_global = NKV; c->vocab_size = VOC;
    c->rotary_dim = M2_ROT; c->full_attention_interval = LO.full_attention_interval;
    c->ssm_state_size = SD; c->ssm_conv_kernel = CK; c->ssm_inner_size = INNER;
    c->ssm_group_count = M2_NKH; c->ssm_time_step_rank = TSR;
    c->n_experts = E; c->n_experts_used = TOPK; c->expert_ffn = MI;
    for (int l = 0; l < LO.n_layers; l++)
        c->attn_kind[l] = qwen35fp8::is_full(LO, l) ? GEMMA4_ATTN_FULL : GEMMA4_ATTN_LINEAR;

    bool ok = true;
    m->embed    = q35fp8_up_bf16_f32(embT, 0.0f);
    m->out_norm = q35fp8_up_bf16_f32(M.find(LO.final_norm_key), 1.0f);   // +1
    m->lm_head  = q35fp8_up_bf16_f32(M.find(LO.lmhead_key), 0.0f);
    if (cudaMalloc(&m->d_logits, (size_t)VOC * sizeof(float)) != cudaSuccess) ok = false;
    ok = ok && m->embed && m->out_norm && m->lm_head;

    for (int l = 0; l < LO.n_layers && ok; l++) {
        q35moe_layer &T = m->L[l];
        T.is_full = qwen35fp8::is_full(LO, l);
        T.in_norm   = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "input_layernorm.weight")), 1.0f);
        T.post_norm = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "post_attention_layernorm.weight")), 1.0f);
        ok = ok && T.in_norm && T.post_norm;
        // ── mixer ──
        if (T.is_full) {
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "self_attn.q_proj.weight"), T.q, 2*NQ*HD, H);
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "self_attn.k_proj.weight"), T.k, NKV*HD, H);
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "self_attn.v_proj.weight"), T.v, NKV*HD, H);
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "self_attn.o_proj.weight"), T.o, H, NQ*HD);
            T.q_norm = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "self_attn.q_norm.weight")), 1.0f);
            T.k_norm = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "self_attn.k_norm.weight")), 1.0f);
            ok = ok && T.q_norm && T.k_norm;
        } else {
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "linear_attn.in_proj_qkv.weight"), T.inqkv, CONVD, H);
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "linear_attn.in_proj_z.weight"),   T.inz,   INNER, H);
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "linear_attn.out_proj.weight"),    T.out,   H, INNER);
            T.ina      = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "linear_attn.in_proj_a.weight")), 0.0f);
            T.inb      = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "linear_attn.in_proj_b.weight")), 0.0f);
            T.conv1d   = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "linear_attn.conv1d.weight")), 0.0f);
            T.dt_bias  = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "linear_attn.dt_bias")), 0.0f);
            T.ssm_norm = q35fp8_up_f32(M.find(qwen35fp8::lkey(LO, l, "linear_attn.norm.weight")));   // gated: NO +1
            T.a_coef   = q35fp8_up_f32(M.find(qwen35fp8::lkey(LO, l, "linear_attn.A_log")));
            if (T.a_coef) q35fp8_neg_exp_kernel<<<(unsigned)((TSR + 255) / 256), 256>>>(T.a_coef, TSR);
            ok = ok && T.ina && T.inb && T.conv1d && T.dt_bias && T.ssm_norm && T.a_coef;
        }
        // ── MoE FFN ──
        T.router_w  = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "mlp.gate.weight")), 0.0f);        // [E×H]
        T.sh_gate_w = q35fp8_up_bf16_f32(M.find(qwen35fp8::lkey(LO, l, "mlp.shared_expert_gate.weight")), 0.0f); // [H]
        ok = ok && T.router_w && T.sh_gate_w;
        ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "mlp.shared_expert.gate_proj.weight"), T.sh_gate, SI, H);
        ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "mlp.shared_expert.up_proj.weight"),   T.sh_up,   SI, H);
        ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, "mlp.shared_expert.down_proj.weight"), T.sh_down, H, SI);
        T.ex_gate = (q35fp8_proj*)calloc(E, sizeof(q35fp8_proj));
        T.ex_up   = (q35fp8_proj*)calloc(E, sizeof(q35fp8_proj));
        T.ex_down = (q35fp8_proj*)calloc(E, sizeof(q35fp8_proj));
        if (!T.ex_gate || !T.ex_up || !T.ex_down) { ok = false; break; }
        for (int e = 0; e < E && ok; e++) {
            char buf[96];
            snprintf(buf, sizeof(buf), "mlp.experts.%d.gate_proj.weight", e);
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, buf), T.ex_gate[e], MI, H);
            snprintf(buf, sizeof(buf), "mlp.experts.%d.up_proj.weight", e);
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, buf), T.ex_up[e],   MI, H);
            snprintf(buf, sizeof(buf), "mlp.experts.%d.down_proj.weight", e);
            ok = ok && q35fp8_load_proj(M, qwen35fp8::lkey(LO, l, buf), T.ex_down[e], H, MI);
        }
        if ((l % 8) == 0) cudaDeviceSynchronize();   // bound in-flight launches over the big expert load
    }
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) ok = false;
    if (!ok) { fprintf(stderr, "qwen35_moe_fp8_load: upload/alloc failed\n"); qwen35_moe_fp8_free(m); return nullptr; }
    fprintf(stderr, "qwen35_moe_fp8_load: loaded %d layers (vocab %d, hidden %d, %d KV heads, "
            "%d experts top-%d, moe_inter %d, shared_inter %d) from %s\n",
            m->n_layers, VOC, H, NKV, E, TOPK, MI, SI, path);
    return m;
}

extern "C" int qwen35_moe_fp8_forward_greedy(void *model, const int32_t *in_ids, int n_prompt,
                                             int32_t *out_ids, int n_gen) {
    q35moe_model *m = (q35moe_model *)model;
    if (!m) return -1;
    const gemma4_model_config_t *c = &m->cfg;
    cudaStream_t st = m->stream;
    const float eps = M2_EPS;
    const int H = m->H, HD = M2_HEAD, NQ = M2_NQ, NKV = m->NKV;
    const int INNER = M2_VALD, CONVD = M2_CONVDIM, KEYD = M2_KEYD;
    const int NKH = M2_NKH, NVH = M2_NVH, SD = M2_SD, TSR = M2_TSR, ROT = M2_ROT;
    const int VOC = m->vocab, L = m->n_layers, E = m->E, TOPK = m->topk, MI = m->moe_inter, SI = m->shared_inter;
    const int nsteps = n_prompt + n_gen - 1;
    (void)ROT; (void)KEYD;

    size_t smGDN = ((size_t)SD * SD + 3 * SD) * sizeof(float);
    if (cudaFuncSetAttribute(qwen35_fp8_gdn_step_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smGDN) != cudaSuccess) {
        fprintf(stderr, "qwen35_moe_fp8_forward_greedy: GDN shared-mem opt-in failed (%zu B)\n", smGDN);
        return -4;
    }

    int rc = 0;
    int *d_arg = nullptr;
    int   *idx = (int*)malloc((size_t)TOPK * sizeof(int));
    float *wts = (float*)malloc((size_t)TOPK * sizeof(float));
    float *h_rlog = (float*)malloc((size_t)E * sizeof(float));
    float h_shlog = 0.f;
    float *x=nullptr,*xn=nullptr,*qg=nullptr,*qb=nullptr,*gate=nullptr,*kb=nullptr,*vb=nullptr,
          *attn=nullptr,*mix=nullptr,*qkv=nullptr,*conv_out=nullptr,*zc=nullptr,*ac=nullptr,
          *bc=nullptr,*gg=nullptr,*bb=nullptr,*qh=nullptr,*kh=nullptr,*vh=nullptr,*core=nullptr,
          *gnorm=nullptr,*rlog=nullptr,*moe_acc=nullptr,*eg=nullptr,*eu=nullptr,*ea=nullptr,
          *ed=nullptr,*sg=nullptr,*su=nullptr,*sa=nullptr,*sd=nullptr,*shlog=nullptr;
    float **Kc = (float**)calloc(L, sizeof(float*));
    float **Vc = (float**)calloc(L, sizeof(float*));
    float **Sst = (float**)calloc(L, sizeof(float*));
    float **ring = (float**)calloc(L, sizeof(float*));
    if (!Kc || !Vc || !Sst || !ring || !idx || !wts || !h_rlog) { rc = -5; goto cleanup; }

    #define CKM(p, n) do { if (cudaMalloc(&(p), (size_t)(n) * sizeof(float)) != cudaSuccess) { \
        fprintf(stderr, "qwen35_moe_fp8_forward_greedy: cudaMalloc failed (%s)\n", #p); rc = -6; goto cleanup; } } while(0)
    if (cudaMalloc(&d_arg, sizeof(int)) != cudaSuccess) { rc = -6; goto cleanup; }
    CKM(x, H); CKM(xn, H); CKM(qg, 2*NQ*HD); CKM(qb, NQ*HD);
    CKM(gate, NQ*HD); CKM(kb, NKV*HD); CKM(vb, NKV*HD); CKM(attn, NQ*HD);
    CKM(mix, H); CKM(qkv, CONVD); CKM(conv_out, CONVD); CKM(zc, INNER);
    CKM(ac, TSR); CKM(bc, TSR); CKM(gg, TSR); CKM(bb, TSR);
    CKM(qh, NKH*SD); CKM(kh, NKH*SD); CKM(vh, NVH*SD); CKM(core, NVH*SD);
    CKM(gnorm, INNER);
    CKM(rlog, E); CKM(moe_acc, H); CKM(eg, MI); CKM(eu, MI); CKM(ea, MI); CKM(ed, H);
    CKM(sg, SI); CKM(su, SI); CKM(sa, SI); CKM(sd, H); CKM(shlog, 1);
    for (int l = 0; l < L; l++) {
        if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
            CKM(Kc[l], (size_t)nsteps*NKV*HD);
            CKM(Vc[l], (size_t)nsteps*NKV*HD);
        } else {
            CKM(Sst[l],  (size_t)NVH*SD*SD);
            CKM(ring[l], (size_t)CONVD*(M2_CK-1));
            cudaMemsetAsync(Sst[l],  0, (size_t)NVH*SD*SD*sizeof(float), st);
            cudaMemsetAsync(ring[l], 0, (size_t)CONVD*(M2_CK-1)*sizeof(float), st);
        }
    }

    for (int p = 0; p < nsteps; p++) {
        int32_t token = (p < n_prompt) ? in_ids[p] : out_ids[p - n_prompt];
        cudaMemcpyAsync(x, m->embed + (size_t)token * H, (size_t)H * sizeof(float),
                        cudaMemcpyDeviceToDevice, st);
        for (int l = 0; l < L; l++) {
            q35moe_layer &T = m->L[l];
            rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(xn, x, T.in_norm, H, 1, eps);
            if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
                fp8_block_gemv_launch(qg, T.q.w, T.q.s, xn, H, 2*NQ*HD, st);
                fp8_block_gemv_launch(kb, T.k.w, T.k.s, xn, H, NKV*HD,  st);
                fp8_block_gemv_launch(vb, T.v.w, T.v.s, xn, H, NKV*HD,  st);
                m2_split_query_gate_kernel<<<(NQ*HD+255)/256,256,0,st>>>(qb, gate, qg, 1, NQ);
                per_head_rms_norm_rows_kernel<<<dim3(NQ,1),256,32*sizeof(float),st>>>(qb, T.q_norm, NQ, HD, 1, eps);
                per_head_rms_norm_rows_kernel<<<dim3(NKV,1),256,32*sizeof(float),st>>>(kb, T.k_norm, NKV, HD, 1, eps);
                qwen35_rope_pos_kernel<<<dim3((ROT/2+31)/32,NQ),32,0,st>>>(qb, NQ, p);
                qwen35_rope_pos_kernel<<<dim3((ROT/2+31)/32,NKV),32,0,st>>>(kb, NKV, p);
                cudaMemcpyAsync(Kc[l]+(size_t)p*NKV*HD, kb, (size_t)NKV*HD*sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(Vc[l]+(size_t)p*NKV*HD, vb, (size_t)NKV*HD*sizeof(float), cudaMemcpyDeviceToDevice, st);
                qwen35_attn_step_kernel<<<NQ,256,(size_t)(p+1)*sizeof(float),st>>>(attn, qb, Kc[l], Vc[l], p, NKV);
                m2_sigmoid_gate_mul_kernel<<<(NQ*HD+255)/256,256,0,st>>>(attn, gate, NQ*HD);
                fp8_block_gemv_launch(mix, T.o.w, T.o.s, attn, NQ*HD, H, st);
            } else {
                fp8_block_gemv_launch(qkv, T.inqkv.w, T.inqkv.s, xn, H, CONVD, st);
                fp8_block_gemv_launch(zc,  T.inz.w,   T.inz.s,   xn, H, INNER, st);
                m2_gemm(ac, xn, T.ina, 1, H, TSR);
                m2_gemm(bc, xn, T.inb, 1, H, TSR);
                qwen35_conv_step_kernel<<<(CONVD+127)/128,128,0,st>>>(conv_out, qkv, ring[l], T.conv1d, CONVD);
                cudaMemcpyAsync(qh, conv_out,        (size_t)KEYD*sizeof(float),  cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(kh, conv_out+KEYD,   (size_t)KEYD*sizeof(float),  cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(vh, conv_out+2*KEYD, (size_t)INNER*sizeof(float), cudaMemcpyDeviceToDevice, st);
                m2_l2norm_heads_kernel<<<dim3(NKH,1),128,0,st>>>(qh, NKH, SD, 1);
                m2_l2norm_heads_kernel<<<dim3(NKH,1),128,0,st>>>(kh, NKH, SD, 1);
                m2_decay_beta_kernel<<<(TSR+255)/256,256,0,st>>>(gg, bb, ac, bc, T.a_coef, T.dt_bias, 1, TSR);
                qwen35_fp8_gdn_step_kernel<<<NVH,128,smGDN,st>>>(core, qh, kh, vh, gg, bb, Sst[l]);
                m2_gated_norm_kernel<<<dim3(NVH,1),128,0,st>>>(gnorm, core, zc, T.ssm_norm, 1, NVH, INNER);
                fp8_block_gemv_launch(mix, T.out.w, T.out.s, gnorm, INNER, H, st);
            }
            qwen35_add_kernel<<<(H+255)/256,256,0,st>>>(x, mix, H);
            // ── MoE FFN (Qwen3_5MoeSparseMoeBlock) ──
            rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(xn, x, T.post_norm, H, 1, eps);
            m2_gemm(rlog, xn, T.router_w, 1, H, E);
            cudaMemcpyAsync(h_rlog, rlog, (size_t)E*sizeof(float), cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
            q35moe_route_host(h_rlog, E, TOPK, idx, wts);
            cudaMemsetAsync(moe_acc, 0, (size_t)H*sizeof(float), st);
            for (int j = 0; j < TOPK; j++) {
                q35fp8_proj &G = T.ex_gate[idx[j]], &U = T.ex_up[idx[j]], &D = T.ex_down[idx[j]];
                fp8_block_gemv_launch(eg, G.w, G.s, xn, H, MI, st);
                fp8_block_gemv_launch(eu, U.w, U.s, xn, H, MI, st);
                silu_glu_kernel<<<(MI+255)/256,256,0,st>>>(ea, eg, eu, MI);
                fp8_block_gemv_launch(ed, D.w, D.s, ea, MI, H, st);
                q35moe_axpy_kernel<<<(H+255)/256,256,0,st>>>(moe_acc, ed, wts[j], H);
            }
            // shared expert (sigmoid-gated)
            fp8_block_gemv_launch(sg, T.sh_gate.w, T.sh_gate.s, xn, H, SI, st);
            fp8_block_gemv_launch(su, T.sh_up.w,   T.sh_up.s,   xn, H, SI, st);
            silu_glu_kernel<<<(SI+255)/256,256,0,st>>>(sa, sg, su, SI);
            fp8_block_gemv_launch(sd, T.sh_down.w, T.sh_down.s, sa, SI, H, st);
            m2_gemm(shlog, xn, T.sh_gate_w, 1, H, 1);
            cudaMemcpyAsync(&h_shlog, shlog, sizeof(float), cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
            float sgate = 1.f / (1.f + expf(-h_shlog));
            q35moe_axpy_kernel<<<(H+255)/256,256,0,st>>>(moe_acc, sd, sgate, H);
            qwen35_add_kernel<<<(H+255)/256,256,0,st>>>(x, moe_acc, H);
        }
        if (p >= n_prompt - 1) {
            rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(xn, x, m->out_norm, H, 1, eps);
            m2_gemm(m->d_logits, xn, m->lm_head, 1, H, VOC);
            qwen35_argmax_kernel<<<1,256,0,st>>>(m->d_logits, VOC, d_arg);
            int argmax = 0;
            cudaMemcpyAsync(&argmax, d_arg, sizeof(int), cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
            out_ids[p - (n_prompt - 1)] = argmax;
        }
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { fprintf(stderr, "qwen35_moe_fp8_forward_greedy: CUDA error at pos %d: %s\n",
                                        p, cudaGetErrorString(e)); rc = -7; goto cleanup; }
    }

cleanup:
    #undef CKM
    cudaStreamSynchronize(st);
    if (d_arg) cudaFree(d_arg);
    free(idx); free(wts); free(h_rlog);
    { float *bufs[] = {x,xn,qg,qb,gate,kb,vb,attn,mix,qkv,conv_out,zc,ac,bc,gg,bb,qh,kh,vh,core,gnorm,
                       rlog,moe_acc,eg,eu,ea,ed,sg,su,sa,sd,shlog};
      for (float *b : bufs) if (b) cudaFree(b); }
    if (Kc)  { for (int l=0;l<L;l++) if (Kc[l])  cudaFree(Kc[l]);  free(Kc); }
    if (Vc)  { for (int l=0;l<L;l++) if (Vc[l])  cudaFree(Vc[l]);  free(Vc); }
    if (Sst) { for (int l=0;l<L;l++) if (Sst[l]) cudaFree(Sst[l]); free(Sst); }
    if (ring){ for (int l=0;l<L;l++) if (ring[l])cudaFree(ring[l]);free(ring); }
    return rc;
}

extern "C" void qwen35_moe_fp8_free(void *model) {
    q35moe_model *m = (q35moe_model *)model;
    if (!m) return;
    auto FW = [](q35fp8_proj &P){ if (P.w) cudaFree((void*)P.w); if (P.s) cudaFree((void*)P.s); };
    if (m->embed) cudaFree(m->embed);
    if (m->out_norm) cudaFree(m->out_norm);
    if (m->lm_head) cudaFree(m->lm_head);
    if (m->d_logits) cudaFree(m->d_logits);
    for (int l = 0; l < m->n_layers; l++) {
        q35moe_layer &T = m->L[l];
        if (T.in_norm) cudaFree(T.in_norm); if (T.post_norm) cudaFree(T.post_norm);
        if (T.is_full) {
            FW(T.q); FW(T.k); FW(T.v); FW(T.o);
            if (T.q_norm) cudaFree(T.q_norm); if (T.k_norm) cudaFree(T.k_norm);
        } else {
            FW(T.inqkv); FW(T.inz); FW(T.out);
            if (T.ina) cudaFree(T.ina); if (T.inb) cudaFree(T.inb);
            if (T.conv1d) cudaFree(T.conv1d); if (T.dt_bias) cudaFree(T.dt_bias);
            if (T.ssm_norm) cudaFree(T.ssm_norm); if (T.a_coef) cudaFree(T.a_coef);
        }
        if (T.router_w) cudaFree(T.router_w); if (T.sh_gate_w) cudaFree(T.sh_gate_w);
        FW(T.sh_gate); FW(T.sh_up); FW(T.sh_down);
        if (T.ex_gate) { for (int e=0;e<m->E;e++) FW(T.ex_gate[e]); free(T.ex_gate); }
        if (T.ex_up)   { for (int e=0;e<m->E;e++) FW(T.ex_up[e]);   free(T.ex_up); }
        if (T.ex_down) { for (int e=0;e<m->E;e++) FW(T.ex_down[e]); free(T.ex_down); }
    }
    delete m;
}
