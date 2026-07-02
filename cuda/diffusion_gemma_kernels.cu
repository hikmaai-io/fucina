// diffusion_gemma_kernels.cu — DiffusionGemma CUDA kernels.
//
// Phase 1: quantized-weight dequantization for the formats present in
// diffusiongemma-26B-A4B-it (Q4_K, Q5_0, Q6_K, Q8_0, Q4_0). Each dequant routine matches
// ggml's canonical dequantize_row_* byte-for-byte (validated by test_diffusion_dequant.cu
// against gguf-py). One thread per output element: simple and correct; the perf-critical
// path is the fused MMVQ/GEMM in later phases, not bulk dequant.

#include "diffusion_gemma_kernels.cuh"

__device__ __forceinline__ float dg_half2float(uint16_t h) {
    __half_raw hr; hr.x = h; return __half2float(*(const __half *)&hr);
}

// Typed store so the dequant kernels can target fp32 (validated path) or bf16 (GEMM path).
template<typename T> __device__ __forceinline__ void dg_st(T* o, int64_t i, float v);
template<> __device__ __forceinline__ void dg_st<float>(float* o, int64_t i, float v){ o[i]=v; }
template<> __device__ __forceinline__ void dg_st<__nv_bfloat16>(__nv_bfloat16* o, int64_t i, float v){ o[i]=__float2bfloat16(v); }

__global__ void dg_f32_to_bf16_kernel(const float* in, __nv_bfloat16* out, int64_t n){
    int64_t i = blockIdx.x*(int64_t)blockDim.x + threadIdx.x; if(i<n) out[i]=__float2bfloat16(in[i]);
}
extern "C" void dg_f32_to_bf16(const float* in, __nv_bfloat16* out, int64_t n, cudaStream_t s){
    dg_f32_to_bf16_kernel<<<(n+255)/256,256,0,s>>>(in,out,n);
}

// ─── Q8_0: d(fp16) + 32×int8 ────────────────────────────────────────────
template<typename T> __global__ void dg_deq_q8_0_kernel(const uint8_t *raw, int64_t n, T *out) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const dg_block_q8_0 *b = (const dg_block_q8_0 *)raw + (i >> 5);
    dg_st(out, i, dg_half2float(b->d) * (float)b->qs[i & 31]);
}

// ─── Q4_0: d(fp16) + 16 nibble bytes; value = d*(nibble-8) ──────────────
template<typename T> __global__ void dg_deq_q4_0_kernel(const uint8_t *raw, int64_t n, T *out) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const dg_block_q4_0 *b = (const dg_block_q4_0 *)raw + (i >> 5);
    int j = i & 31;                 // 0..31
    uint8_t byte = b->qs[j & 15];   // low half (j<16) and high half (j>=16) share bytes 0..15
    int nib = (j < 16) ? (byte & 0xF) : (byte >> 4);
    dg_st(out, i, dg_half2float(b->d) * (float)(nib - 8));
}

// ─── Q5_0: d(fp16) + qh(4=32 bits) + 16 nibble bytes; 5th bit from qh ───
template<typename T> __global__ void dg_deq_q5_0_kernel(const uint8_t *raw, int64_t n, T *out) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const dg_block_q5_0 *b = (const dg_block_q5_0 *)raw + (i >> 5);
    uint32_t qh; memcpy(&qh, b->qh, 4);
    int j = i & 31;
    if (j < 16) {                    // first half: low nibble + bit j
        uint8_t xh = ((qh >> (j + 0)) << 4) & 0x10;
        int x = ((b->qs[j] & 0x0F) | xh) - 16;
        dg_st(out, i, (float)x * dg_half2float(b->d));
    } else {                         // second half: high nibble + bit (j-16+12)
        int jj = j - 16;
        uint8_t xh = ((qh >> (jj + 12))) & 0x10;
        int x = ((b->qs[jj] >> 4) | xh) - 16;
        dg_st(out, i, (float)x * dg_half2float(b->d));
    }
}

// ─── Q4_K: super-block (256), d/dmin(fp16) + 12 packed 6-bit scale/min + 128 nibbles ──
// Mirrors get_scale_min_k4 + dequantize_row_q4_K.
__device__ __forceinline__ void dg_get_scale_min_k4(int j, const uint8_t *q, int *d, int *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}
template<typename T> __global__ void dg_deq_q4_K_kernel(const uint8_t *raw, int64_t n, T *out) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const dg_block_q4_K *b = (const dg_block_q4_K *)raw + (i >> 8);  // /256
    int idx = (int)(i & 255);
    int group = idx >> 6;            // 0..3   (each spans 64 elems / advances qs by 32)
    int rem   = idx & 63;            // 0..63
    int hi    = rem >> 5;            // 0 low-nibble half, 1 high-nibble half
    int sc_i  = 2 * group + hi;
    int sc, m;
    dg_get_scale_min_k4(sc_i, b->scales, &sc, &m);
    float d   = dg_half2float(b->d) * (float)sc;
    float mn  = dg_half2float(b->dmin) * (float)m;
    uint8_t byte = b->qs[group * 32 + (rem & 31)];
    int nib = hi ? (byte >> 4) : (byte & 0xF);
    dg_st(out, i, d * (float)nib - mn);
}

// ─── Q6_K: super-block (256), ql(128) qh(64) scales(16 int8) d(fp16) ─────
// Mirrors dequantize_row_q6_K: value = d * scales[is] * (q - 32).
template<typename T> __global__ void dg_deq_q6_K_kernel(const uint8_t *raw, int64_t n, T *out) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const dg_block_q6_K *b = (const dg_block_q6_K *)raw + (i >> 8);
    int idx  = (int)(i & 255);
    int half = idx >> 7;             // 0/1: which 128-elem half
    int wh   = idx & 127;            // within-half 0..127
    int quad = wh >> 5;              // 0..3
    int l    = wh & 31;             // 0..31
    int is   = l >> 4;              // 0/1
    const uint8_t *ql = b->ql + half * 64;
    const uint8_t *qh = b->qh + half * 32;
    const int8_t  *sc = b->scales + half * 8;
    int q, scale;
    switch (quad) {
        case 0: q = (ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4); scale = sc[is + 0]; break;
        case 1: q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4); scale = sc[is + 2]; break;
        case 2: q = (ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4); scale = sc[is + 4]; break;
        default:q = (ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4); scale = sc[is + 6]; break;
    }
    dg_st(out, i, dg_half2float(b->d) * (float)scale * (float)(q - 32));
}

template<typename T>
static int dg_dequant_T(int ggml_type, const void *raw_dev, int64_t n_elem, T *out_dev, cudaStream_t stream) {
    const uint8_t *raw = (const uint8_t *)raw_dev;
    int bn = dg_block_nelem(ggml_type);
    if (bn == 0 || (n_elem % bn) != 0) return -1;
    int threads = 256;
    int64_t blocks = (n_elem + threads - 1) / threads;
    dim3 g((unsigned)blocks);
    switch (ggml_type) {
        case DG_GGML_Q8_0: dg_deq_q8_0_kernel<<<g, threads, 0, stream>>>(raw, n_elem, out_dev); break;
        case DG_GGML_Q4_0: dg_deq_q4_0_kernel<<<g, threads, 0, stream>>>(raw, n_elem, out_dev); break;
        case DG_GGML_Q5_0: dg_deq_q5_0_kernel<<<g, threads, 0, stream>>>(raw, n_elem, out_dev); break;
        case DG_GGML_Q4_K: dg_deq_q4_K_kernel<<<g, threads, 0, stream>>>(raw, n_elem, out_dev); break;
        case DG_GGML_Q6_K: dg_deq_q6_K_kernel<<<g, threads, 0, stream>>>(raw, n_elem, out_dev); break;
        default: return -2;
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -3;
}
extern "C" int dg_dequant(int ggml_type, const void *raw_dev, int64_t n_elem,
                          float *out_dev, cudaStream_t stream) {
    return dg_dequant_T<float>(ggml_type, raw_dev, n_elem, out_dev, stream);
}
extern "C" int dg_dequant_bf16(int ggml_type, const void *raw_dev, int64_t n_elem,
                               __nv_bfloat16 *out_dev, cudaStream_t stream) {
    return dg_dequant_T<__nv_bfloat16>(ggml_type, raw_dev, n_elem, out_dev, stream);
}

// ════════════════════════════════════════════════════════════════════════
// Phase 2: forward-pass elementwise kernels. Activations column-major
// [features, tokens]: element (f,t) at f + features*t.
// ════════════════════════════════════════════════════════════════════════
#include <math.h>

__device__ __forceinline__ float dg_gelu_tanh(float x) {
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

// One block per token (column); blockDim.x threads cooperate over `feat` features.
__global__ void dg_rmsnorm_kernel(float *out, const float *in, const float *w, int feat, float eps) {
    int t = blockIdx.x;
    const float *col = in + (size_t)t * feat;
    float *ocol = out + (size_t)t * feat;
    __shared__ float red[256];
    float ss = 0.f;
    for (int i = threadIdx.x; i < feat; i += blockDim.x) ss += col[i] * col[i];
    red[threadIdx.x] = ss; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
    float inv = rsqrtf(red[0] / feat + eps);
    for (int i = threadIdx.x; i < feat; i += blockDim.x) ocol[i] = col[i] * inv * (w ? w[i] : 1.0f);
}
extern "C" void dg_rmsnorm(float *out, const float *in, const float *w, int feat, int tokens, float eps, cudaStream_t s) {
    dg_rmsnorm_kernel<<<tokens, 256, 0, s>>>(out, in, w, feat, eps);
}

// per-head norm over head_dim for [head_dim, n_head, tokens]; one block per (head,token).
__global__ void dg_head_rmsnorm_kernel(float *x, const float *w, int hd, float eps) {
    int blk = blockIdx.x;                     // head + n_head*token
    float *col = x + (size_t)blk * hd;
    __shared__ float red[512];
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += col[i] * col[i];
    red[threadIdx.x] = ss; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
    float inv = rsqrtf(red[0] / hd + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) col[i] = col[i] * inv * (w ? w[i] : 1.0f);
}
extern "C" void dg_head_rmsnorm(float *x, const float *w, int hd, int n_head, int tokens, float eps, cudaStream_t s) {
    int nb = n_head * tokens;
    int thr = hd < 512 ? hd : 512;
    dg_head_rmsnorm_kernel<<<nb, thr, 0, s>>>(x, w, hd, eps);
}

// NEOX rope on [head_dim, n_head, tokens]; one block per (head,token), threads over hd/2.
__global__ void dg_rope_kernel(float *x, const int *pos, int hd, int n_head, float theta_base, const float *ff) {
    int blk = blockIdx.x;                     // head + n_head*token
    int t = blk / n_head;
    float *col = x + (size_t)blk * hd;
    int half = hd >> 1;
    for (int d = threadIdx.x; d < half; d += blockDim.x) {
        float fct = ff ? ff[d] : 1.0f;
        float theta = (float)pos[t] * powf(theta_base, -2.0f * d / hd) / fct;
        float c = cosf(theta), sn = sinf(theta);
        float x0 = col[d], x1 = col[d + half];
        col[d]        = x0 * c - x1 * sn;
        col[d + half] = x0 * sn + x1 * c;
    }
}
extern "C" void dg_rope(float *x, const int *pos, int hd, int n_head, int tokens, float theta_base, const float *ff, cudaStream_t s) {
    int nb = n_head * tokens;
    int thr = (hd / 2) < 256 ? (hd / 2) : 256;
    dg_rope_kernel<<<nb, thr, 0, s>>>(x, pos, hd, n_head, theta_base, ff);
}

__global__ void dg_gelu_mul_kernel(float *out, const float *g, const float *u, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = dg_gelu_tanh(g[i]) * u[i];
}
extern "C" void dg_gelu_mul(float *out, const float *g, const float *u, int64_t n, cudaStream_t s) {
    dg_gelu_mul_kernel<<<(n + 255) / 256, 256, 0, s>>>(out, g, u, n);
}

// SiLU-GLU over SEPARATE gate/up buffers: out = silu(g)*u, silu(x)=x*sigmoid(x). Qwen3-MoE keeps
// gate and up as distinct expert slabs (ffn_gate_exps / ffn_up_exps), so the fused split variant
// above is not usable — this consumes the two grouped-GEMM outputs directly.
__global__ void dg_silu_mul_kernel(float *out, const float *g, const float *u, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) { float x = g[i]; out[i] = (x / (1.0f + __expf(-x))) * u[i]; }
}
extern "C" void dg_silu_mul(float *out, const float *g, const float *u, int64_t n, cudaStream_t s) {
    dg_silu_mul_kernel<<<(n + 255) / 256, 256, 0, s>>>(out, g, u, n);
}

// gateup is [2*half, ncols] column-major; out[half,ncols] = gelu(gateup[0:half]) * gateup[half:2half]
__global__ void dg_split_gelu_mul_kernel(float *out, const float *gu, int half, int ncols) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t n = (int64_t)half * ncols;
    if (i >= n) return;
    int r = i % half, c = i / half;
    const float *col = gu + (size_t)c * (2 * half);
    out[i] = dg_gelu_tanh(col[r]) * col[half + r];
}
extern "C" void dg_split_gelu_mul(float *out, const float *gu, int half, int ncols, cudaStream_t s) {
    int64_t n = (int64_t)half * ncols;
    dg_split_gelu_mul_kernel<<<(n + 255) / 256, 256, 0, s>>>(out, gu, half, ncols);
}

__global__ void dg_add_kernel(float *o, const float *a, const float *b, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; if (i < n) o[i] = a[i] + b[i];
}
extern "C" void dg_add(float *o, const float *a, const float *b, int64_t n, cudaStream_t s) {
    dg_add_kernel<<<(n + 255) / 256, 256, 0, s>>>(o, a, b, n);
}

__global__ void dg_scale_kernel(float *x, int64_t n, float sc) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; if (i < n) x[i] *= sc;
}
extern "C" void dg_scale(float *x, int64_t n, float sc, cudaStream_t s) {
    dg_scale_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, n, sc);
}

__global__ void dg_mul_vec_cols_kernel(float *x, const float *vec, int feat, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; if (i < n) x[i] *= vec[i % feat];
}
extern "C" void dg_mul_vec_cols(float *x, const float *vec, int feat, int tokens, cudaStream_t s) {
    int64_t n = (int64_t)feat * tokens;
    dg_mul_vec_cols_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, vec, feat, n);
}

__global__ void dg_mul_region_scalar_kernel(float *x, int feat, int t0, int t1, const float *sc) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t n = (int64_t)feat * (t1 - t0);
    if (i >= n) return;
    x[(size_t)t0 * feat + i] *= sc[0];
}
extern "C" void dg_mul_region_scalar(float *x, int feat, int t0, int t1, const float *sc, cudaStream_t s) {
    int64_t n = (int64_t)feat * (t1 - t0);
    if (n <= 0) return;
    dg_mul_region_scalar_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, feat, t0, t1, sc);
}

__global__ void dg_softcap_kernel(float *x, int64_t n, float cap) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}
extern "C" void dg_softcap(float *x, int64_t n, float cap, cudaStream_t s) {
    dg_softcap_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, n, cap);
}

// embed gather: dst[:,t] = tok_f32[ids[t]*n_embd ...] * embed_scale
__global__ void dg_embed_gather_kernel(float *dst, const float *tok, const int *ids, int n_embd, float esc) {
    int t = blockIdx.x;
    const float *row = tok + (size_t)ids[t] * n_embd;
    float *ocol = dst + (size_t)t * n_embd;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) ocol[i] = row[i] * esc;
}
extern "C" void dg_embed_gather(float *dst, const float *tok, const int *ids, int n_embd, int tokens, float esc, cudaStream_t s) {
    dg_embed_gather_kernel<<<tokens, 256, 0, s>>>(dst, tok, ids, n_embd, esc);
}

// per-token softmax over n_expert → top-k (selection sort) → normalize. One block per token.
// Probs are computed ONCE in parallel into shared memory; the thread-0 selection then scans
// smem (the old form recomputed expf inside the k×ne selection loop — 2048 SERIAL expf per
// token, measured 178 µs/call = 21% of a B=1 MoE-35B decode step). Same expf(L-gmax)/denom
// values and the same strict-> lowest-index tie-break → bit-identical selection and weights.
__global__ void dg_softmax_topk_kernel(const float *logits, int ne, int topk, int *oidx, float *ow) {
    int t = blockIdx.x;
    const float *L = logits + (size_t)t * ne;
    __shared__ float sm[256];
    __shared__ float mx[256];
    __shared__ float pr[256];   // per-expert probs (ne <= 256: DG 128, Qwen3.5-MoE 256)
    float m = -1e30f;
    for (int i = threadIdx.x; i < ne; i += blockDim.x) m = fmaxf(m, L[i]);
    mx[threadIdx.x] = m; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (threadIdx.x < s) mx[threadIdx.x] = fmaxf(mx[threadIdx.x], mx[threadIdx.x + s]); __syncthreads(); }
    float gmax = mx[0];
    float se = 0.f;
    for (int i = threadIdx.x; i < ne; i += blockDim.x) se += expf(L[i] - gmax);
    sm[threadIdx.x] = se; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s]; __syncthreads(); }
    float denom = sm[0];
    for (int i = threadIdx.x; i < ne; i += blockDim.x) pr[i] = expf(L[i] - gmax) / denom;
    __syncthreads();
    if (threadIdx.x == 0) {
        float probs_sum = 0.f;
        bool used[256]; for (int i = 0; i < ne; i++) used[i] = false;
        for (int k = 0; k < topk; k++) {
            float best = -1.f; int bi = -1;
            for (int i = 0; i < ne; i++) { if (!used[i] && pr[i] > best) { best = pr[i]; bi = i; } }
            used[bi] = true; oidx[t * topk + k] = bi; ow[t * topk + k] = best; probs_sum += best;
        }
        for (int k = 0; k < topk; k++) ow[t * topk + k] /= probs_sum;  // normalize top-k
    }
}
extern "C" void dg_softmax_topk(const float *logits, int ne, int tokens, int topk, int *oidx, float *ow, cudaStream_t s) {
    dg_softmax_topk_kernel<<<tokens, 128, 0, s>>>(logits, ne, topk, oidx, ow);
}

__global__ void dg_gather_cols_kernel(float *dst, const float *src, const int *idx, int feat, int ncols) {
    int j = blockIdx.x; if (j >= ncols) return;
    const float *sc = src + (size_t)idx[j] * feat;
    float *dc = dst + (size_t)j * feat;
    for (int i = threadIdx.x; i < feat; i += blockDim.x) dc[i] = sc[i];
}
extern "C" void dg_gather_cols(float *dst, const float *src, const int *idx, int feat, int ncols, cudaStream_t s) {
    if (ncols <= 0) return;
    dg_gather_cols_kernel<<<ncols, 256, 0, s>>>(dst, src, idx, feat, ncols);
}

__global__ void dg_scatteradd_cols_kernel(float *dst, const float *src, const int *idx, const float *cs, int feat, int ncols) {
    int j = blockIdx.x; if (j >= ncols) return;
    float sc = cs[j];
    float *dc = dst + (size_t)idx[j] * feat;
    const float *sco = src + (size_t)j * feat;
    for (int i = threadIdx.x; i < feat; i += blockDim.x) atomicAdd(&dc[i], sco[i] * sc);
}
extern "C" void dg_scatteradd_cols(float *dst, const float *src, const int *idx, const float *cs, int feat, int ncols, cudaStream_t s) {
    if (ncols <= 0) return;
    dg_scatteradd_cols_kernel<<<ncols, 256, 0, s>>>(dst, src, idx, cs, feat, ncols);
}

// Region-aware bidirectional attention, scale=1.0, materialized per (head,query).
// q:[hd,n_head,T] k,v:[hd,n_kv,T]. mask per llama.cpp diffusion: prompt query causal (swa-clipped);
// canvas query bidirectional (sliding: last n_swa-1 prompt + all canvas; global: all).
__global__ void dg_attention_kernel(float *out, const float *q, const float *k, const float *v,
                                    int hd, int n_head, int n_kv, int T, int P, int n_swa, int is_sliding) {
    int h = blockIdx.x, qpos = blockIdx.y;
    int kvh = h / (n_head / n_kv);
    const float *qv = q + ((size_t)(qpos * n_head + h)) * hd;
    extern __shared__ float sc[];           // scores[T]
    float *red = sc + T;                    // reduction buffer [blockDim.x]
    bool q_is_canvas = qpos >= P;
    long canvas_prompt_lo = (long)P - n_swa + 1;
    // 1. scores
    for (int kk = 0; kk < T; kk++) {
        bool k_is_canvas = kk >= P;
        bool allow;
        if (q_is_canvas) {
            allow = is_sliding ? (k_is_canvas || (kk >= canvas_prompt_lo)) : true;
        } else {
            allow = (!k_is_canvas) && (kk <= qpos);
            if (allow && is_sliding && (qpos - kk) >= n_swa) allow = false;
        }
        float part = 0.f;
        if (allow) {
            const float *kv_ = k + ((size_t)(kk * n_kv + kvh)) * hd;
            for (int d = threadIdx.x; d < hd; d += blockDim.x) part += qv[d] * kv_[d];
        }
        red[threadIdx.x] = part; __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
        if (threadIdx.x == 0) sc[kk] = allow ? red[0] : -1e30f;
        __syncthreads();
    }
    // 2. softmax over keys (thread 0), store probs in sc
    __shared__ float ssum;
    if (threadIdx.x == 0) {
        float m = -1e30f; for (int kk = 0; kk < T; kk++) m = fmaxf(m, sc[kk]);
        float s = 0.f; for (int kk = 0; kk < T; kk++) { float e = expf(sc[kk] - m); sc[kk] = e; s += e; }
        ssum = s;
    }
    __syncthreads();
    float inv = 1.0f / ssum;
    // 3. weighted sum of V
    float *oc = out + ((size_t)(qpos * n_head + h)) * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        for (int kk = 0; kk < T; kk++) { if (sc[kk] > 0.f) acc += sc[kk] * v[((size_t)(kk * n_kv + kvh)) * hd + d]; }
        oc[d] = acc * inv;
    }
}
extern "C" void dg_attention(float *out, const float *q, const float *k, const float *v,
                             int hd, int n_head, int n_kv, int T, int P, int n_swa, int is_sliding, cudaStream_t s) {
    dim3 g(n_head, T);
    int thr = hd < 256 ? hd : 256;
    size_t smem = (size_t)(T + thr) * sizeof(float);
    dg_attention_kernel<<<g, thr, smem, s>>>(out, q, k, v, hd, n_head, n_kv, T, P, n_swa, is_sliding);
}

// Canvas-only bidirectional attention against a CACHED prompt K/V (P positions, computed once
// during prefill) plus the fresh canvas K/V (C positions). One block per (head, canvas query).
// qc/ck/cv carry only the C canvas columns; pk/pv carry only the P prompt columns. Semantics
// are identical to dg_attention_kernel's canvas-query branch (sliding: last n_swa-1 prompt + all
// canvas; global: all), just reading the two K/V regions from separate buffers.
__global__ void dg_attention_canvas_kernel(float *out, const float *qc,
        const __nv_bfloat16 *pk, const __nv_bfloat16 *pv, const float *ck, const float *cv,
        int hd, int n_head, int n_kv, int P, int C, int n_swa, int is_sliding) {
    int h = blockIdx.x, c = blockIdx.y;        // c = canvas query index in [0,C)
    int kvh = h / (n_head / n_kv);
    const float *qv = qc + ((size_t)(c * n_head + h)) * hd;
    extern __shared__ float sm[];
    float *qs = sm;                            // [hd] cached query
    float *sc = sm + hd;                        // [P+C] scores/probs
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = blockDim.x >> 5;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) qs[d] = qv[d];
    __syncthreads();
    long canvas_prompt_lo = (long)P - n_swa + 1;
    const int TT = P + C;
    // 1. scores — ONE WARP PER KEY (coalesced dot + __shfl reduction, no per-key __syncthreads).
    for (int k = warp; k < TT; k += nwarps) {
        float part = 0.f;
        bool allow = true;
        if (k < P) { // prompt key from the bf16 cache
            allow = is_sliding ? ((long)k >= canvas_prompt_lo) : true;
            const __nv_bfloat16 *kb = pk + ((size_t)(k * n_kv + kvh)) * hd;
            if (allow) for (int d = lane; d < hd; d += 32) part += qs[d] * __bfloat162float(kb[d]);
        } else {     // fresh canvas key (fp32)
            const float *kf = ck + ((size_t)((k - P) * n_kv + kvh)) * hd;
            for (int d = lane; d < hd; d += 32) part += qs[d] * kf[d];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xFFFFFFFF, part, o);
        if (lane == 0) sc[k] = allow ? part : -1e30f;
    }
    __syncthreads();
    // 2. softmax over the P+C keys (thread 0)
    __shared__ float ssum;
    if (threadIdx.x == 0) {
        float m = -1e30f; for (int kk = 0; kk < TT; kk++) m = fmaxf(m, sc[kk]);
        float s = 0.f; for (int kk = 0; kk < TT; kk++) { float e = expf(sc[kk] - m); sc[kk] = e; s += e; }
        ssum = s;
    }
    __syncthreads();
    float inv = 1.0f / ssum;
    // 3. weighted sum of V (prompt V from cache, canvas V fresh) — one output dim per thread
    float *oc = out + ((size_t)(c * n_head + h)) * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        for (int kk = 0; kk < P; kk++) { float p = sc[kk];     if (p > 0.f) acc += p * __bfloat162float(pv[((size_t)(kk * n_kv + kvh)) * hd + d]); }
        for (int j = 0; j < C; j++) { float p = sc[P + j]; if (p > 0.f) acc += p * cv[((size_t)(j * n_kv + kvh)) * hd + d]; }
        oc[d] = acc * inv;
    }
}
extern "C" void dg_attention_canvas(float *out, const float *qc, const __nv_bfloat16 *pk, const __nv_bfloat16 *pv,
        const float *ck, const float *cv, int hd, int n_head, int n_kv, int P, int C,
        int n_swa, int is_sliding, cudaStream_t s) {
    dim3 g(n_head, C);
    size_t smem = (size_t)(hd + P + C) * sizeof(float);
    dg_attention_canvas_kernel<<<g, 256, smem, s>>>(out, qc, pk, pv, ck, cv, hd, n_head, n_kv, P, C, n_swa, is_sliding);
}

// Chunked CAUSAL prefill attention: each chunk query c (global position Pc+c) attends the cached
// prefix K/V (Pc earlier tokens, bf16) + the chunk's own fresh K/V (CH tokens, fp32), causally
// (key ≤ query) within the sliding window. Lets prefill run chunk-by-chunk so the activation
// buffers are CH-sized, not max_prompt-sized. Same warp-per-key + online structure as the canvas.
__global__ void dg_attention_chunk_kernel(float *out, const float *qc,
        const __nv_bfloat16 *pk, const __nv_bfloat16 *pv, const float *ck, const float *cv,
        int hd, int n_head, int n_kv, int Pc, int CH, int n_swa, int is_sliding) {
    int h = blockIdx.x, c = blockIdx.y;        // c = chunk query index in [0,CH)
    int kvh = h / (n_head / n_kv);
    const float *qv = qc + ((size_t)(c * n_head + h)) * hd;
    extern __shared__ float sm[];
    float *qs = sm; float *sc = sm + hd;       // [hd] query, [Pc+CH] scores
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarps = blockDim.x >> 5;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) qs[d] = qv[d];
    __syncthreads();
    const long qg = (long)Pc + c;              // global query position
    const int TT = Pc + CH;
    for (int k = warp; k < TT; k += nwarps) {
        float part = 0.f; bool allow;
        if (k < Pc) {                          // prefix key (global pos k), bf16
            allow = is_sliding ? (qg - (long)k < n_swa) : true;
            const __nv_bfloat16 *kb = pk + ((size_t)(k * n_kv + kvh)) * hd;
            if (allow) for (int d = lane; d < hd; d += 32) part += qs[d] * __bfloat162float(kb[d]);
        } else {                               // chunk key (local j, global Pc+j), fp32 — causal
            int j = k - Pc;
            allow = (j <= c) && (is_sliding ? (c - j < n_swa) : true);
            const float *kf = ck + ((size_t)(j * n_kv + kvh)) * hd;
            if (allow) for (int d = lane; d < hd; d += 32) part += qs[d] * kf[d];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xFFFFFFFF, part, o);
        if (lane == 0) sc[k] = allow ? part : -1e30f;
    }
    __syncthreads();
    __shared__ float ssum;
    if (threadIdx.x == 0) {
        float m = -1e30f; for (int kk = 0; kk < TT; kk++) m = fmaxf(m, sc[kk]);
        float s = 0.f; for (int kk = 0; kk < TT; kk++) { float e = expf(sc[kk] - m); sc[kk] = e; s += e; }
        ssum = s;
    }
    __syncthreads();
    float inv = 1.0f / ssum;
    float *oc = out + ((size_t)(c * n_head + h)) * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.f;
        for (int kk = 0; kk < Pc; kk++) { float p = sc[kk];     if (p > 0.f) acc += p * __bfloat162float(pv[((size_t)(kk * n_kv + kvh)) * hd + d]); }
        for (int j = 0; j < CH; j++)    { float p = sc[Pc + j]; if (p > 0.f) acc += p * cv[((size_t)(j * n_kv + kvh)) * hd + d]; }
        oc[d] = acc * inv;
    }
}
extern "C" void dg_attention_chunk(float *out, const float *qc, const __nv_bfloat16 *pk, const __nv_bfloat16 *pv,
        const float *ck, const float *cv, int hd, int n_head, int n_kv, int Pc, int CH,
        int n_swa, int is_sliding, cudaStream_t s) {
    dim3 g(n_head, CH);
    size_t smem = (size_t)(hd + Pc + CH) * sizeof(float);
    dg_attention_chunk_kernel<<<g, 256, smem, s>>>(out, qc, pk, pv, ck, cv, hd, n_head, n_kv, Pc, CH, n_swa, is_sliding);
}

// ════════════════════════════════════════════════════════════════════════
// Phase 3: diffusion sampler kernels (per canvas column over the vocab).
// logits laid out [vocab, C] column-major (column c = canvas position c).
// ════════════════════════════════════════════════════════════════════════

// Per-column softmax over vocab → probs[vocab,C] (for self-conditioning soft-embedding).
__global__ void dg_softmax_cols_kernel(const float* logits, float* probs, int vocab){
    int c=blockIdx.x; const float* L=logits+(size_t)c*vocab; float* P=probs+(size_t)c*vocab;
    __shared__ float r[256];
    float m=-1e30f; for(int i=threadIdx.x;i<vocab;i+=blockDim.x) m=fmaxf(m,L[i]);
    r[threadIdx.x]=m; __syncthreads();
    for(int s=blockDim.x>>1;s>0;s>>=1){ if(threadIdx.x<s) r[threadIdx.x]=fmaxf(r[threadIdx.x],r[threadIdx.x+s]); __syncthreads(); }
    float gmax=r[0];
    float se=0.f; for(int i=threadIdx.x;i<vocab;i+=blockDim.x) se+=expf(L[i]-gmax);
    r[threadIdx.x]=se; __syncthreads();
    for(int s=blockDim.x>>1;s>0;s>>=1){ if(threadIdx.x<s) r[threadIdx.x]+=r[threadIdx.x+s]; __syncthreads(); }
    float inv=1.0f/r[0];
    for(int i=threadIdx.x;i<vocab;i+=blockDim.x) P[i]=expf(L[i]-gmax)*inv;
}
extern "C" void dg_softmax_cols(const float* logits, float* probs, int vocab, int C, cudaStream_t s){
    dg_softmax_cols_kernel<<<C,256,0,s>>>(logits,probs,vocab);
}

// One block per column: argmax token, Shannon entropy of softmax, and a multinomial sample
// using rnd[c] ∈ [0,1).  logits are assumed already temperature-scaled.
__global__ void dg_sample_step_kernel(const float* logits, const float* rnd,
                                      int* out_sample, int* out_argmax, float* out_entropy, int vocab){
    int c=blockIdx.x; const float* L=logits+(size_t)c*vocab;
    __shared__ float rv[256]; __shared__ int ri[256];
    // max + argmax
    float m=-1e30f; int mi=0;
    for(int i=threadIdx.x;i<vocab;i+=blockDim.x){ if(L[i]>m){m=L[i];mi=i;} }
    rv[threadIdx.x]=m; ri[threadIdx.x]=mi; __syncthreads();
    for(int s=blockDim.x>>1;s>0;s>>=1){ if(threadIdx.x<s){ if(rv[threadIdx.x+s]>rv[threadIdx.x]){rv[threadIdx.x]=rv[threadIdx.x+s];ri[threadIdx.x]=ri[threadIdx.x+s];} } __syncthreads(); }
    float gmax=rv[0]; int amax=ri[0];
    // S1=sum exp(li-gmax), S2=sum exp*(li-gmax)
    __shared__ float s1[256]; __shared__ float s2[256];
    float a=0.f,b=0.f; for(int i=threadIdx.x;i<vocab;i+=blockDim.x){ float e=expf(L[i]-gmax); a+=e; b+=e*(L[i]-gmax); }
    s1[threadIdx.x]=a; s2[threadIdx.x]=b; __syncthreads();
    for(int s=blockDim.x>>1;s>0;s>>=1){ if(threadIdx.x<s){ s1[threadIdx.x]+=s1[threadIdx.x+s]; s2[threadIdx.x]+=s2[threadIdx.x+s]; } __syncthreads(); }
    float Z=s1[0];
    if(threadIdx.x==0){ out_argmax[c]=amax; out_entropy[c]=logf(Z) - s2[0]/Z; }
    // Multinomial inverse-CDF, parallelized: contiguous chunk per thread so the cumulative order
    // matches a sequential scan (same pick), but the per-thread serial work is vocab/blockDim, not
    // vocab. Each thread sums its chunk's exp; an exclusive prefix-sum locates the owning chunk; the
    // owner re-scans only its chunk. (Was: thread 0 scanning all 262144 — ~27ms/step.)
    __shared__ float part[256]; __shared__ float pre[256];
    int chunk=(vocab+blockDim.x-1)/blockDim.x;
    int lo=threadIdx.x*chunk, hi=lo+chunk; if(hi>vocab) hi=vocab; if(lo>vocab) lo=vocab;
    float ps=0.f; for(int i=lo;i<hi;i++) ps+=expf(L[i]-gmax);
    part[threadIdx.x]=ps; __syncthreads();
    if(threadIdx.x==0){ float acc=0.f; for(int t=0;t<blockDim.x;t++){ pre[t]=acc; acc+=part[t]; } }
    __syncthreads();
    float target=rnd[c]*Z;
    bool owner = (target>=pre[threadIdx.x]) &&
                 (threadIdx.x==blockDim.x-1 || target<pre[threadIdx.x]+part[threadIdx.x]);
    if(owner){ float cum=pre[threadIdx.x]; int pick=hi>lo?hi-1:vocab-1;
        for(int i=lo;i<hi;i++){ cum+=expf(L[i]-gmax); if(cum>=target){ pick=i; break; } }
        out_sample[c]=pick; }
}
extern "C" void dg_sample_step(const float* logits, const float* rnd, int* out_sample, int* out_argmax,
                               float* out_entropy, int vocab, int C, cudaStream_t s){
    dg_sample_step_kernel<<<C,256,0,s>>>(logits,rnd,out_sample,out_argmax,out_entropy,vocab);
}

// ════════════════════════════════════════════════════════════════════════
// Phase 4: fused quantized matmul (dp4a) — reads the quant weight ONCE, no
// fp32 round-trip. Replaces dequant-to-fp32 + cuBLAS sgemm. Mirrors the proven
// gemma4_kernels.cu MMQ/MMVQ math (quantize_q8_1 + tiled dp4a), extended to the
// k-quant sub-block scales the diffusion model uses (Q4_K here).
// ════════════════════════════════════════════════════════════════════════

// Read 4 packed bytes as one int via two 16-bit loads (all quant qs/qh in this model are ≥2-byte
// aligned), i.e. 2 memory transactions instead of 4 scalar byte loads — matters in the MMQ weight
// staging hot path.
__device__ __forceinline__ int dg_u32(const uint8_t* p){
    const uint16_t* h=(const uint16_t*)p;
    return (int)((uint32_t)h[0] | ((uint32_t)h[1]<<16));
}

// Symmetric per-32-block int8 activation quantization (Q8_1-style). One warp per 32-block;
// the whole [in_dim,tokens] matrix is treated as one flat array of (in_dim*tokens/32) blocks,
// so dx[gb]/sx[gb] with gb = col*(in_dim/32)+blk match the MMQ kernel's column indexing.
__global__ void dg_quantize_q8_1_kernel(const float* x, int8_t* qx, float* dx, int* sx, int64_t total){
    int64_t gb = blockIdx.x; int lane = threadIdx.x;
    int64_t i = gb*32 + lane;
    float v = (i < total) ? x[i] : 0.f;
    float a = fabsf(v);
    #pragma unroll
    for(int o=16;o>0;o>>=1) a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFF, a, o));
    float d = a/127.f; float id = (d>0.f) ? 1.f/d : 0.f;
    int q = max(-127, min(127, __float2int_rn(v*id)));
    if(i < total) qx[i] = (int8_t)q;
    int qsum = q;
    #pragma unroll
    for(int o=16;o>0;o>>=1) qsum += __shfl_xor_sync(0xFFFFFFFF, qsum, o);
    if(lane==0){ dx[gb]=d; sx[gb]=qsum; }
}
extern "C" void dg_quantize_q8_1(const float* x, int8_t* qx, float* dx, int* sx,
                                 int in_dim, int tokens, cudaStream_t s){
    int64_t total = (int64_t)in_dim*tokens;          // always a multiple of 32
    dg_quantize_q8_1_kernel<<<(unsigned)(total/32), 32, 0, s>>>(x, qx, dx, sx, total);
}

// Tiled-MMQ GEMM for Q4_K weights. Y[out_dim×N] = W_q4_K[out_dim×in_dim] · X_int8[in_dim×N],
// token-major output out[col*out_dim+row]. Structure mirrors mmq_q4_0_tiled_kernel: each block
// computes a BM×BN tile, looping K in 32-elem blocks; per K-step it stages BM weight rows and BN
// activation cols into smem, then a 4×4 micro-tile per thread does the dp4a.
//
// Q4_K specifics: 256-elem superblock = {d, dmin (fp16), scales[12] (8×6-bit sub-scale + 8×6-bit
// sub-min), qs[128]}. The 8 sub-blocks of 32 elems tile the superblock contiguously, and the
// contiguous-order sub-block index p (0..7) is exactly the scale index (verified vs the dequant
// kernel: group=idx>>6, hi=(idx>>5)&1, sc_i=2*group+hi == p). Sub-block p's nibbles come from
// qs[(p>>1)*32 + k] (low nibble if p even, high if p odd). Per-element weight value is
// d*sc_p*nib − dmin*m_p, so the fold per 32-block is  dx·(d·sc_p·Σnib·qx − dmin·m_p·Σqx) =
// dx·(Wde·sumi − Wmo·Σqx), with Wde=d·sc_p, Wmo=dmin·m_p, Σqx=sx (precomputed). int8 activation
// ⇒ agrees with the fp32 dequant path to quantization error, not bit-exactly.
// One BM×BN output tile of a Q4_K matmul. `weight` is the (already expert-offset) row-major
// Q4_K weight; rows [rowbase,rowbase+BM) × the BN columns starting at absolute column `colbase`,
// of which only the first `ncol` are valid (the rest are written by no-one). qx/dx/sx are the
// quantized activations indexed by ABSOLUTE column. Shared by the dense and grouped-expert
// launchers so the dp4a math lives in exactly one place.
__device__ __forceinline__ void dg_mmq_q4_K_tile(
        float* out, const uint8_t* weight, const int8_t* qx, const float* dx, const int* sx,
        int in_dim, int out_dim, int rowbase, int colbase, int ncol){
    __shared__ int   Ws[DG_MMQ_BM][8];   // weight nibbles (raw 0..15), 4 per int
    __shared__ float Wde[DG_MMQ_BM];     // d·sc_p   (per row, per 32-block)
    __shared__ float Wmo[DG_MMQ_BM];     // dmin·m_p
    __shared__ int   Xs[DG_MMQ_BN][8];   // activation int8, 4 per int
    __shared__ float Xd[DG_MMQ_BN];      // per-col activation block scale
    __shared__ int   Xq[DG_MMQ_BN];      // per-col Σ activation int8

    const int nb  = in_dim >> 5;         // 32-blocks
    const int nsb = in_dim >> 8;         // 256-superblocks
    const int t = threadIdx.x;           // 0..255 (16×16 → 4×4 micro-tile)
    const int tx = t & 15, ty = t >> 4;

    float acc[4][4];
    #pragma unroll
    for(int i=0;i<4;i++)
        #pragma unroll
        for(int j=0;j<4;j++) acc[i][j]=0.f;

    for(int b=0;b<nb;b++){
        const int SB = b>>3, p = b&7;
        // Stage W tile: 256 threads = 64 rows × 4; each thread loads 2 of the 8 nibble-ints.
        {
            int r=t>>2, wk=t&3, row=rowbase+r;
            if(row < out_dim){
                const uint8_t* blk = weight + ((size_t)row*nsb + SB)*144;
                const uint8_t* qs  = blk + 16 + (size_t)(p>>1)*32;
                int w0 = dg_u32(qs + (2*wk)*4);
                int w1 = dg_u32(qs + (2*wk+1)*4);
                if(p&1){ Ws[r][2*wk]=(w0>>4)&0x0F0F0F0F; Ws[r][2*wk+1]=(w1>>4)&0x0F0F0F0F; }
                else   { Ws[r][2*wk]= w0    &0x0F0F0F0F; Ws[r][2*wk+1]= w1    &0x0F0F0F0F; }
                if(wk==0){
                    __half_raw hd; hd.x=(uint16_t)(blk[0]|((uint16_t)blk[1]<<8));
                    __half_raw hm; hm.x=(uint16_t)(blk[2]|((uint16_t)blk[3]<<8));
                    float d=__half2float(*(const __half*)&hd), dmin=__half2float(*(const __half*)&hm);
                    int sc,m; dg_get_scale_min_k4(p, blk+4, &sc, &m);
                    Wde[r]=d*(float)sc; Wmo[r]=dmin*(float)m;
                }
            } else { Ws[r][2*wk]=0; Ws[r][2*wk+1]=0; if(wk==0){Wde[r]=0.f;Wmo[r]=0.f;} }
        }
        // Stage X tile: 256 threads = 64 cols × 4; each loads 2 of the 8 ints.
        {
            int c=t>>2, xk=t&3;
            if(c < ncol){
                size_t col=(size_t)colbase+c;
                const int* xqs=(const int*)(qx + col*in_dim + (size_t)b*32);
                Xs[c][xk*2]=xqs[xk*2]; Xs[c][xk*2+1]=xqs[xk*2+1];
                if(xk==0){ Xd[c]=dx[col*nb+b]; Xq[c]=sx[col*nb+b]; }
            } else { Xs[c][xk*2]=0; Xs[c][xk*2+1]=0; if(xk==0){Xd[c]=0.f;Xq[c]=0;} }
        }
        __syncthreads();

        int sumi[4][4];
        #pragma unroll
        for(int i=0;i<4;i++)
            #pragma unroll
            for(int j=0;j<4;j++) sumi[i][j]=0;
        #pragma unroll
        for(int k=0;k<8;k++){
            int wv[4], xv[4];
            #pragma unroll
            for(int i=0;i<4;i++) wv[i]=Ws[ty*4+i][k];
            #pragma unroll
            for(int j=0;j<4;j++) xv[j]=Xs[tx*4+j][k];
            #pragma unroll
            for(int i=0;i<4;i++)
                #pragma unroll
                for(int j=0;j<4;j++) sumi[i][j]=__dp4a(wv[i],xv[j],sumi[i][j]);
        }
        #pragma unroll
        for(int i=0;i<4;i++){
            float wde=Wde[ty*4+i], wmo=Wmo[ty*4+i];
            #pragma unroll
            for(int j=0;j<4;j++)
                acc[i][j] += Xd[tx*4+j]*( wde*(float)sumi[i][j] - wmo*(float)Xq[tx*4+j] );
        }
        __syncthreads();
    }

    #pragma unroll
    for(int i=0;i<4;i++){
        int row=rowbase+ty*4+i; if(row>=out_dim) continue;
        #pragma unroll
        for(int j=0;j<4;j++){ int lc=tx*4+j; if(lc<ncol) out[((size_t)colbase+lc)*out_dim+row]=acc[i][j]; }
    }
}

// Dense Q4_K matmul: a regular BM×BN grid over [out_dim × N].
__global__ void dg_mmq_q4_K_tiled_kernel(float* out, const uint8_t* weight,
        const int8_t* qx, const float* dx, const int* sx, int in_dim, int out_dim, int N){
    int colbase = blockIdx.y*DG_MMQ_BN;
    int ncol = N-colbase; if(ncol>DG_MMQ_BN) ncol=DG_MMQ_BN;
    dg_mmq_q4_K_tile(out, weight, qx, dx, sx, in_dim, out_dim, blockIdx.x*DG_MMQ_BM, colbase, ncol);
}
extern "C" void dg_mmq_q4_K(float* out, const void* weight, const int8_t* qx, const float* dx,
                            const int* sx, int in_dim, int out_dim, int N, cudaStream_t s){
    dim3 g((unsigned)((out_dim+DG_MMQ_BM-1)/DG_MMQ_BM), (unsigned)((N+DG_MMQ_BN-1)/DG_MMQ_BN));
    dg_mmq_q4_K_tiled_kernel<<<g,256,0,s>>>(out,(const uint8_t*)weight,qx,dx,sx,in_dim,out_dim,N);
}

// ── Grouped expert Q4_K matmul: ALL active experts in ONE launch ─────────────────────────
// The per-expert dequant+sgemm loop (128 tiny launches, each ~22 blocks → under-occupancy +
// launch overhead) is the real cost; this maps blockIdx.x → one (expert, row-tile, col-tile)
// of work via host-built descriptor arrays, so the whole MoE layer runs at full occupancy and
// each expert's weight slab streams from DRAM exactly once. qx/dx/sx and out are the FLATTENED
// gathered assignments [in_dim × total] (columns grouped contiguously per expert); te/trb/tcb/tnc
// give each tile's expert, row-tile index, absolute column base, and valid column count.
// ── On-GPU MoE routing (counting-sort by expert) — replaces the per-layer D2H→host→H2D ───────
// Inputs (device): tki[n_assign] top-k expert ids, tkw[n_assign] router weights, pes[n_expert]
// per-expert scale. n_assign = n_tokens·n_used. Outputs (device): count[e]=#assignments to e,
// coloff[e]=exclusive prefix (column base), src[pos]=source token, csc[pos]=tkw·pes grouped by e.
__global__ void dg_moe_count_kernel(const int* tki, int n_assign, int* count){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n_assign) atomicAdd(&count[tki[i]],1);
}
__global__ void dg_moe_scan_kernel(const int* count, int* coloff, int* cursor, int n_expert){
    if(threadIdx.x==0){ int a=0; for(int e=0;e<n_expert;e++){ coloff[e]=a; cursor[e]=a; a+=count[e]; } }
}
__global__ void dg_moe_scatter_kernel(const int* tki, const float* tkw, const float* pes,
        int n_assign, int n_used, int* cursor, int* src, float* csc){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n_assign) return;
    int ex=tki[i]; int pos=atomicAdd(&cursor[ex],1);
    src[pos]=i/n_used; csc[pos]=tkw[i]*pes[ex];
}
extern "C" void dg_moe_route(const int* tki, const float* tkw, const float* pes,
        int n_tokens, int n_used, int n_expert, int* count, int* coloff, int* cursor,
        int* src, float* csc, cudaStream_t s){
    int n_assign=n_tokens*n_used;
    cudaMemsetAsync(count,0,(size_t)n_expert*4,s);
    dg_moe_count_kernel<<<(n_assign+255)/256,256,0,s>>>(tki,n_assign,count);
    dg_moe_scan_kernel<<<1,32,0,s>>>(count,coloff,cursor,n_expert);
    dg_moe_scatter_kernel<<<(n_assign+255)/256,256,0,s>>>(tki,tkw,pes,n_assign,n_used,cursor,src,csc);
}

// Same counting-sort route, but also records invpos[i]=pos — the grouped column assigned to
// assignment i (= token·n_used + k). The cursor atomicAdd makes the column ORDER nondeterministic,
// but invpos captures the exact (token,k)→column map so a later per-token reduce can sum each
// token's contributions in FIXED k order (no atomics) and stay bit-identical run-to-run.
__global__ void dg_moe_scatter_inv_kernel(const int* tki, const float* tkw, const float* pes,
        int n_assign, int n_used, int* cursor, int* src, float* csc, int* invpos){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n_assign) return;
    int ex=tki[i]; int pos=atomicAdd(&cursor[ex],1);
    src[pos]=i/n_used; csc[pos]=tkw[i]*pes[ex]; invpos[i]=pos;
}
// Scan variant that ALSO compacts the ids of experts with count>0 into active[0..], padding -1 up
// to n_slot. Lets a grouped expert GEMM launch a grid of n_slot=min(n_assign,n_expert) blocks
// (STATIC per batch size → CUDA-graph-safe) instead of all n_expert — at decode B·topk ≪ n_expert
// that removes thousands of early-return blocks per launch.
__global__ void dg_moe_scan_active_kernel(const int* count, int* coloff, int* cursor,
        int n_expert, int* active, int n_slot){
    if(threadIdx.x==0){
        int a=0, na=0;
        for(int e=0;e<n_expert;e++){
            coloff[e]=a; cursor[e]=a; a+=count[e];
            if(count[e]>0 && na<n_slot) active[na++]=e;
        }
        for(int j=na;j<n_slot;j++) active[j]=-1;
    }
}
// active/n_slot: optional (active=NULL skips) compacted active-expert list, see scan_active above.
extern "C" void dg_moe_route_inv(const int* tki, const float* tkw, const float* pes,
        int n_tokens, int n_used, int n_expert, int* count, int* coloff, int* cursor,
        int* src, float* csc, int* invpos, int* active, int n_slot, cudaStream_t s){
    int n_assign=n_tokens*n_used;
    cudaMemsetAsync(count,0,(size_t)n_expert*4,s);
    dg_moe_count_kernel<<<(n_assign+255)/256,256,0,s>>>(tki,n_assign,count);
    if(active) dg_moe_scan_active_kernel<<<1,32,0,s>>>(count,coloff,cursor,n_expert,active,n_slot);
    else       dg_moe_scan_kernel<<<1,32,0,s>>>(count,coloff,cursor,n_expert);
    dg_moe_scatter_inv_kernel<<<(n_assign+255)/256,256,0,s>>>(tki,tkw,pes,n_assign,n_used,cursor,src,csc,invpos);
}

// Deterministic per-token expert reduce. out[t] = Σ_k oe[invpos[t·n_used+k]]·csc[invpos[...]],
// summed in fixed k order. For a fixed (t,k), both oe[pos] (that token's k-th expert output) and
// csc[pos] (its router weight) are value-deterministic regardless of the nondeterministic column
// order, so this fully replaces the atomicAdd scatter-add and removes all MoE float nondeterminism.
// One block per token; out is fully written (no pre-memset needed).
__global__ void dg_moe_reduce_kernel(float* out, const float* oe, const int* invpos,
        const float* csc, int feat, int n_used){
    int t=blockIdx.x;
    for(int h=threadIdx.x; h<feat; h+=blockDim.x){
        float acc=0.f;
        for(int k=0;k<n_used;k++){ int pos=invpos[t*n_used+k]; acc += oe[(size_t)pos*feat+h]*csc[pos]; }
        out[(size_t)t*feat+h]=acc;
    }
}
extern "C" void dg_moe_reduce(float* out, const float* oe, const int* invpos, const float* csc,
        int feat, int n_tokens, int n_used, cudaStream_t s){
    if(n_tokens<=0) return;
    dg_moe_reduce_kernel<<<n_tokens,256,0,s>>>(out,oe,invpos,csc,feat,n_used);
}

// Grouped expert Q4_K — narrow-column tiled MMQ (BM=64 × BN=16). Keeps the tiled kernel's smem
// activation reuse (one staged column tile feeds all 64 rows → ~out_dim/64 L2 re-reads, vs
// out_dim× for a warp-per-row scheme) AND removes the BN=64 padding waste: BN=16 ≈ the average
// tokens-per-expert (256·8/128=16), so almost no wasted dp4a. Each thread owns 4 rows × 1 col.
// grid=(row-tiles, num_active); blockIdx.y→active expert; the block sweeps the expert's ⌈ne/16⌉
// column tiles. Q4_K per-32-block fold: dx·(d·sc_p·Σnib·qx − dmin·m_p·Σqx).
#define DG_GBM 64
#define DG_GBN 16
// Stage one 32-block b (Q4_K) into the given smem buffer slice (for double-buffering).
__device__ __forceinline__ void dg_q4k_stage(int (*Ws)[8], float* Wde, float* Wmo,
        int (*Xs)[8], float* Xd, int* Xq, const uint8_t* weight, const int8_t* qx,
        const float* dx, const int* sx, int in_dim, int out_dim, int rowbase, int colbase, int ncol, int b){
    const int nb=in_dim>>5, nsb=in_dim>>8; const int t=threadIdx.x;
    const int SB=b>>3, p=b&7;
    { int r=t>>2, wk=t&3, row=rowbase+r;
      if(row<out_dim){
          const uint8_t* blk=weight+((size_t)row*nsb+SB)*144; const uint8_t* qs=blk+16+(size_t)(p>>1)*32;
          uint2 ww=*(const uint2*)(qs+(size_t)(2*wk)*4); int w0=(int)ww.x, w1=(int)ww.y;  // 8B aligned
          if(p&1){ Ws[r][2*wk]=(w0>>4)&0x0F0F0F0F; Ws[r][2*wk+1]=(w1>>4)&0x0F0F0F0F; }
          else   { Ws[r][2*wk]= w0    &0x0F0F0F0F; Ws[r][2*wk+1]= w1    &0x0F0F0F0F; }
          if(wk==0){ __half_raw hd; hd.x=(uint16_t)(blk[0]|((uint16_t)blk[1]<<8));
              __half_raw hm; hm.x=(uint16_t)(blk[2]|((uint16_t)blk[3]<<8));
              int sc,mn; dg_get_scale_min_k4(p,blk+4,&sc,&mn);
              Wde[r]=__half2float(*(const __half*)&hd)*(float)sc; Wmo[r]=__half2float(*(const __half*)&hm)*(float)mn; }
      } else { Ws[r][2*wk]=0; Ws[r][2*wk+1]=0; if(wk==0){Wde[r]=0.f;Wmo[r]=0.f;} } }
    if(t<DG_GBN*8){ int c=t>>3, xk=t&7;
        if(c<ncol){ size_t col=(size_t)colbase+c; const int* xqs=(const int*)(qx+col*in_dim+(size_t)b*32);
            Xs[c][xk]=xqs[xk]; if(xk==0){ Xd[c]=dx[col*nb+b]; Xq[c]=sx[col*nb+b]; } }
        else { Xs[c][xk]=0; if(xk==0){Xd[c]=0.f;Xq[c]=0;} } }
}

// Double-buffered BN16 tile — best-measured MoE structure (superblock/int4/larger-tile variants all
// tested flat or worse via ncu; the kernel is latency-bound near its HW limit at ~5.2× over ref).
__device__ __forceinline__ void dg_mmq16_q4_K_tile(
        float* out, const uint8_t* weight, const int8_t* qx, const float* dx, const int* sx,
        int in_dim, int out_dim, int rowbase, int colbase, int ncol){
    __shared__ int   Ws[2][DG_GBM][8]; __shared__ float Wde[2][DG_GBM], Wmo[2][DG_GBM];
    __shared__ int   Xs[2][DG_GBN][8]; __shared__ float Xd[2][DG_GBN]; __shared__ int Xq[2][DG_GBN];
    const int nb=in_dim>>5;
    const int t=threadIdx.x, tx=t&15, ty=t>>4;
    float acc[4]; acc[0]=acc[1]=acc[2]=acc[3]=0.f;
    dg_q4k_stage(Ws[0],Wde[0],Wmo[0],Xs[0],Xd[0],Xq[0],weight,qx,dx,sx,in_dim,out_dim,rowbase,colbase,ncol,0);
    for(int b=0;b<nb;b++){
        __syncthreads();
        int cur=b&1;
        if(b+1<nb) dg_q4k_stage(Ws[(b+1)&1],Wde[(b+1)&1],Wmo[(b+1)&1],Xs[(b+1)&1],Xd[(b+1)&1],Xq[(b+1)&1],
                                weight,qx,dx,sx,in_dim,out_dim,rowbase,colbase,ncol,b+1);
        int s0=0,s1=0,s2=0,s3=0;
        #pragma unroll
        for(int k=0;k<8;k++){ int xv=Xs[cur][tx][k];
            s0=__dp4a(Ws[cur][ty*4+0][k],xv,s0); s1=__dp4a(Ws[cur][ty*4+1][k],xv,s1);
            s2=__dp4a(Ws[cur][ty*4+2][k],xv,s2); s3=__dp4a(Ws[cur][ty*4+3][k],xv,s3); }
        float xd=Xd[cur][tx]; int xq=Xq[cur][tx];
        acc[0]+=xd*(Wde[cur][ty*4+0]*(float)s0-Wmo[cur][ty*4+0]*(float)xq);
        acc[1]+=xd*(Wde[cur][ty*4+1]*(float)s1-Wmo[cur][ty*4+1]*(float)xq);
        acc[2]+=xd*(Wde[cur][ty*4+2]*(float)s2-Wmo[cur][ty*4+2]*(float)xq);
        acc[3]+=xd*(Wde[cur][ty*4+3]*(float)s3-Wmo[cur][ty*4+3]*(float)xq);
    }
    if(tx<ncol){ size_t col=(size_t)colbase+tx;
        #pragma unroll
        for(int i=0;i<4;i++){ int row=rowbase+ty*4+i; if(row<out_dim) out[col*out_dim+row]=acc[i]; } }
}
__global__ void __launch_bounds__(256,6) dg_mmq_q4_K_grouped_kernel(float* out, const uint8_t* wbase, int64_t slab_stride,
        const int8_t* qx, const float* dx, const int* sx,
        const int* coloff, const int* count, int in_dim, int out_dim){
    int e=blockIdx.y; int ne=count[e]; if(ne==0) return; int co=coloff[e];
    const uint8_t* weight=wbase+(size_t)e*slab_stride; int rb=blockIdx.x*DG_GBM;
    for(int cb=0; cb<ne; cb+=DG_GBN){ int nc=ne-cb; if(nc>DG_GBN) nc=DG_GBN;
        dg_mmq16_q4_K_tile(out,weight,qx,dx,sx,in_dim,out_dim,rb,co+cb,nc); __syncthreads(); }
}
extern "C" void dg_mmq_q4_K_grouped(float* out, const void* wbase, int64_t slab_stride,
        const int8_t* qx, const float* dx, const int* sx,
        const int* coloff, const int* count, int n_expert,
        int in_dim, int out_dim, cudaStream_t s){
    dim3 g((unsigned)((out_dim+DG_GBM-1)/DG_GBM), (unsigned)n_expert);
    dg_mmq_q4_K_grouped_kernel<<<g,256,0,s>>>(
        out,(const uint8_t*)wbase,slab_stride,qx,dx,sx,coloff,count,in_dim,out_dim);
}

// ── 32-block formats (Q8_0, Q5_0) — the expert/dense `down` projection ───────────────────
// Spread 4 bits of `h` to bit-4 of 4 consecutive int8 lanes (the 5th bit of a Q5_0 quant).
__device__ __forceinline__ int dg_spread4_bit4(int h){
    return ((h&1)<<4) | (((h>>1)&1)<<12) | (((h>>2)&1)<<20) | (((h>>3)&1)<<28);
}

// One BM×BN output tile for a legacy 32-block weight. FMT: 0=Q8_0 (34B, symmetric, value=d·qs),
// 1=Q5_0 (22B, value=d·(q5−16), q5 = nibble | 5th-bit-from-qh; nibble layout = Q4_0: byte j →
// elem j low, elem j+16 high). Activation int8 (quantize_q8_1) gives Σqx=sx; the fold per
// 32-block is dx·d·(Σq·qx − OFF·Σqx) with OFF=0 (Q8_0) / 16 (Q5_0). Shared microtile/fold with
// the Q4_K tile; only weight decode + the OFF offset differ.
template<int FMT>
__device__ __forceinline__ void dg_mmq_q32_tile(
        float* out, const uint8_t* weight, const int8_t* qx, const float* dx, const int* sx,
        int in_dim, int out_dim, int rowbase, int colbase, int ncol){
    __shared__ int   Ws[DG_MMQ_BM][8];
    __shared__ float Wd[DG_MMQ_BM];
    __shared__ int   Xs[DG_MMQ_BN][8];
    __shared__ float Xd[DG_MMQ_BN];
    __shared__ int   Xq[DG_MMQ_BN];
    const int nb = in_dim >> 5;
    const int blkb = (FMT==0) ? 34 : 22;
    const int OFF  = (FMT==0) ?  0 : 16;
    const int t = threadIdx.x, tx = t & 15, ty = t >> 4;

    float acc[4][4];
    #pragma unroll
    for(int i=0;i<4;i++)
        #pragma unroll
        for(int j=0;j<4;j++) acc[i][j]=0.f;

    for(int b=0;b<nb;b++){
        // Stage W: 256 threads = 64 rows × 4; each thread fills 2 of the 8 ints.
        {
            int r=t>>2, wk=t&3, row=rowbase+r;
            if(row < out_dim){
                const uint8_t* blk = weight + ((size_t)row*nb + b)*blkb;
                if(FMT==0){
                    const uint8_t* qs = blk + 2;
                    Ws[r][2*wk]   = dg_u32(qs + (2*wk)*4);
                    Ws[r][2*wk+1] = dg_u32(qs + (2*wk+1)*4);
                } else {
                    int qh = dg_u32(blk + 2); const uint8_t* qs = blk + 6;
                    #pragma unroll
                    for(int mm=0;mm<2;mm++){ int m=2*wk+mm, k=m&3; int w=dg_u32(qs + 4*k); int val,hb;
                        // 5th bit of element e is qh bit e: low nibbles → elems 4k..4k+3,
                        // high nibbles → elems 16+4k..16+4k+3.
                        if(m<4){ val=w&0x0F0F0F0F;      hb=(qh>>(4*k))   &0xF; }
                        else   { val=(w>>4)&0x0F0F0F0F; hb=(qh>>(4*k+16))&0xF; }
                        Ws[r][m] = val | dg_spread4_bit4(hb); }
                }
                if(wk==0){ __half_raw hr; hr.x=(uint16_t)(blk[0]|((uint16_t)blk[1]<<8)); Wd[r]=__half2float(*(const __half*)&hr); }
            } else { Ws[r][2*wk]=0; Ws[r][2*wk+1]=0; if(wk==0) Wd[r]=0.f; }
        }
        // Stage X (identical to the Q4_K tile).
        {
            int c=t>>2, xk=t&3;
            if(c < ncol){
                size_t col=(size_t)colbase+c;
                const int* xqs=(const int*)(qx + col*in_dim + (size_t)b*32);
                Xs[c][xk*2]=xqs[xk*2]; Xs[c][xk*2+1]=xqs[xk*2+1];
                if(xk==0){ Xd[c]=dx[col*nb+b]; Xq[c]=sx[col*nb+b]; }
            } else { Xs[c][xk*2]=0; Xs[c][xk*2+1]=0; if(xk==0){Xd[c]=0.f;Xq[c]=0;} }
        }
        __syncthreads();

        int sumi[4][4];
        #pragma unroll
        for(int i=0;i<4;i++)
            #pragma unroll
            for(int j=0;j<4;j++) sumi[i][j]=0;
        #pragma unroll
        for(int k=0;k<8;k++){
            int wv[4], xv[4];
            #pragma unroll
            for(int i=0;i<4;i++) wv[i]=Ws[ty*4+i][k];
            #pragma unroll
            for(int j=0;j<4;j++) xv[j]=Xs[tx*4+j][k];
            #pragma unroll
            for(int i=0;i<4;i++)
                #pragma unroll
                for(int j=0;j<4;j++) sumi[i][j]=__dp4a(wv[i],xv[j],sumi[i][j]);
        }
        #pragma unroll
        for(int i=0;i<4;i++){
            float wd=Wd[ty*4+i];
            #pragma unroll
            for(int j=0;j<4;j++)
                acc[i][j] += Xd[tx*4+j]*wd*( (float)sumi[i][j] - (float)(OFF*Xq[tx*4+j]) );
        }
        __syncthreads();
    }

    #pragma unroll
    for(int i=0;i<4;i++){
        int row=rowbase+ty*4+i; if(row>=out_dim) continue;
        #pragma unroll
        for(int j=0;j<4;j++){ int lc=tx*4+j; if(lc<ncol) out[((size_t)colbase+lc)*out_dim+row]=acc[i][j]; }
    }
}

// Narrow-column (BM=64 × BN=16) grouped tile for the 32-block down formats — same smem-reuse +
// no-padding design as dg_mmq16_q4_K_tile. FMT 0=Q8_0 (34B, OFF=0), 1=Q5_0 (22B, OFF=16).
template<int FMT>
__device__ __forceinline__ void dg_mmq16_q32_tile(
        float* out, const uint8_t* weight, const int8_t* qx, const float* dx, const int* sx,
        int in_dim, int out_dim, int rowbase, int colbase, int ncol){
    __shared__ int Ws[DG_GBM][8]; __shared__ float Wd[DG_GBM];
    __shared__ int Xs[DG_GBN][8]; __shared__ float Xd[DG_GBN]; __shared__ int Xq[DG_GBN];
    const int nb=in_dim>>5;
    const int blkb=(FMT==0)?34:22; const int OFF=(FMT==0)?0:16;
    const int t=threadIdx.x, tx=t&15, ty=t>>4;
    float acc[4]; acc[0]=acc[1]=acc[2]=acc[3]=0.f;
    for(int b=0;b<nb;b++){
        { int r=t>>2, wk=t&3, row=rowbase+r;
          if(row<out_dim){ const uint8_t* blk=weight+((size_t)row*nb+b)*blkb;
              if(FMT==0){ const uint8_t* qs=blk+2; Ws[r][2*wk]=dg_u32(qs+(2*wk)*4); Ws[r][2*wk+1]=dg_u32(qs+(2*wk+1)*4); }
              else { int qh=dg_u32(blk+2); const uint8_t* qs=blk+6;
                  #pragma unroll
                  for(int mm=0;mm<2;mm++){ int m=2*wk+mm,k=m&3; int w=dg_u32(qs+4*k),val,hb;
                      if(m<4){ val=w&0x0F0F0F0F; hb=(qh>>(4*k))&0xF; } else { val=(w>>4)&0x0F0F0F0F; hb=(qh>>(4*k+16))&0xF; }
                      Ws[r][m]=val|dg_spread4_bit4(hb); } }
              if(wk==0){ __half_raw hr; hr.x=(uint16_t)(blk[0]|((uint16_t)blk[1]<<8)); Wd[r]=__half2float(*(const __half*)&hr); }
          } else { Ws[r][2*wk]=0; Ws[r][2*wk+1]=0; if(wk==0) Wd[r]=0.f; } }
        if(t<DG_GBN*8){ int c=t>>3, xk=t&7;
            if(c<ncol){ size_t col=(size_t)colbase+c; const int* xqs=(const int*)(qx+col*in_dim+(size_t)b*32);
                Xs[c][xk]=xqs[xk]; if(xk==0){ Xd[c]=dx[col*nb+b]; Xq[c]=sx[col*nb+b]; } }
            else { Xs[c][xk]=0; if(xk==0){Xd[c]=0.f;Xq[c]=0;} } }
        __syncthreads();
        int s0=0,s1=0,s2=0,s3=0;
        #pragma unroll
        for(int k=0;k<8;k++){ int xv=Xs[tx][k];
            s0=__dp4a(Ws[ty*4+0][k],xv,s0); s1=__dp4a(Ws[ty*4+1][k],xv,s1);
            s2=__dp4a(Ws[ty*4+2][k],xv,s2); s3=__dp4a(Ws[ty*4+3][k],xv,s3); }
        float xd=Xd[tx]; int xq=Xq[tx];
        acc[0]+=xd*Wd[ty*4+0]*((float)s0-(float)(OFF*xq)); acc[1]+=xd*Wd[ty*4+1]*((float)s1-(float)(OFF*xq));
        acc[2]+=xd*Wd[ty*4+2]*((float)s2-(float)(OFF*xq)); acc[3]+=xd*Wd[ty*4+3]*((float)s3-(float)(OFF*xq));
        __syncthreads();
    }
    if(tx<ncol){ size_t col=(size_t)colbase+tx;
        #pragma unroll
        for(int i=0;i<4;i++){ int row=rowbase+ty*4+i; if(row<out_dim) out[col*out_dim+row]=acc[i]; } }
}
template<int FMT>
__global__ void dg_mmq_q32_grouped_kernel(float* out, const uint8_t* wbase, int64_t slab_stride,
        const int8_t* qx, const float* dx, const int* sx,
        const int* coloff, const int* count, int in_dim, int out_dim){
    int e=blockIdx.y; int ne=count[e]; if(ne==0) return; int co=coloff[e];
    const uint8_t* weight = wbase + (size_t)e*slab_stride;
    int rb = blockIdx.x*DG_GBM;
    for(int cb=0; cb<ne; cb+=DG_GBN){ int nc=ne-cb; if(nc>DG_GBN) nc=DG_GBN;
        dg_mmq16_q32_tile<FMT>(out, weight, qx, dx, sx, in_dim, out_dim, rb, co+cb, nc);
        __syncthreads(); }
}
extern "C" void dg_mmq_q8_0_grouped(float* out, const void* wbase, int64_t slab_stride,
        const int8_t* qx, const float* dx, const int* sx,
        const int* coloff, const int* count, int n_expert,
        int in_dim, int out_dim, cudaStream_t s){
    dim3 g((unsigned)((out_dim+DG_MMQ_BM-1)/DG_MMQ_BM), (unsigned)n_expert);
    dg_mmq_q32_grouped_kernel<0><<<g,256,0,s>>>(out,(const uint8_t*)wbase,slab_stride,qx,dx,sx,coloff,count,in_dim,out_dim);
}
extern "C" void dg_mmq_q5_0_grouped(float* out, const void* wbase, int64_t slab_stride,
        const int8_t* qx, const float* dx, const int* sx,
        const int* coloff, const int* count, int n_expert,
        int in_dim, int out_dim, cudaStream_t s){
    dim3 g((unsigned)((out_dim+DG_MMQ_BM-1)/DG_MMQ_BM), (unsigned)n_expert);
    dg_mmq_q32_grouped_kernel<1><<<g,256,0,s>>>(out,(const uint8_t*)wbase,slab_stride,qx,dx,sx,coloff,count,in_dim,out_dim);
}
