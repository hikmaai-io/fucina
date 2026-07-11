// Qwen3.5 device kernels and single-sequence numerical oracle.
// Internal implementation fragment: included once by gemma4_kernels.cu.
#pragma once

// ─── Qwen3.5 hybrid (qwen35) M2: GDN + gated-full-attn mixer kernels + parity self-test ─────
// ═══════════════════════════════════════════════════════════════════════════════════════════
// Net-new math for the qwen35 hybrid, gated behind the qwen35 arch when wired into the forward
// (M3). Implemented here as standalone fp32 device kernels + a self-test that loads the torch
// reference (cuda/qwen35_layer_ref.py) and compares. All conventions are taken VERBATIM from
// llama.cpp src/models/qwen35.cpp + delta-net-base.cpp and HF modeling_qwen3_next.py:
//   * RMSNorm gain applied directly (x_norm*w); the GGUF bakes the +1.
//   * ssm_a already stores -exp(A_log); decay g = ssm_a * softplus(alpha + dt_bias).
//   * delta-rule q scaled by 1/sqrt(head_dim) AFTER l2-norm; q,k l2-normed over head_dim.
//   * conv1d depthwise causal k=4 then SiLU on concat[q;k;v]; split q,k(16 heads) v(32 heads);
//     q,k repeat_interleave 16→32 (v-head h uses k-head h/2).
//   * RMSNormGated = RMSNorm(o)*SiLU(z) (gate NOT in variance).
//   * partial NEOX RoPE on the first rotary_dim dims (mrope collapses to this for text).

// qwen35 fixed geometry (9B Q4_K_M). Used only by the M2 self-test kernels below.
#define M2_H        4096
#define M2_HEAD     256
#define M2_NQ       16
#define M2_NKV      4
#define M2_ROT      64
#define M2_THETA    10000000.0f
#define M2_CONVDIM  8192
#define M2_KEYD     2048
#define M2_VALD     4096
#define M2_NKH      16
#define M2_NVH      32
#define M2_SD       128
#define M2_TSR      32
#define M2_CK       4
#define M2_CHUNK    64
#define M2_EPS      1e-6f

// out[n,o] = sum_i in[n,i]*W[o,i]   (W row-major [OUT,IN]; one block per (o,n), reduces IN).
__global__ void m2_gemm_kernel(float *out, const float *in, const float *W,
                               int N, int INN, int OUT) {
    int o = blockIdx.x, n = blockIdx.y;
    if (o >= OUT || n >= N) return;
    __shared__ float red[32];
    const float *xr = in + (size_t)n * INN;
    const float *wr = W  + (size_t)o * INN;
    float acc = 0.f;
    for (int i = threadIdx.x; i < INN; i += blockDim.x) acc += xr[i] * wr[i];
    acc = block_reduce_sum(acc, red);
    if (threadIdx.x == 0) out[(size_t)n * OUT + o] = acc;
}

// Split the 2×-wide q projection qg[N,16,512] (per head [query(256)|gate(256)]) into a
// contiguous query q[N,16,256] and a head-major gate[N,4096].
__global__ void m2_split_query_gate_kernel(float *q, float *gate, const float *qg, int N, int nq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;     // over N*NQ*HEAD
    if (idx >= N * nq * M2_HEAD) return;
    int d = idx % M2_HEAD;
    int h = (idx / M2_HEAD) % nq;
    int n = idx / (M2_HEAD * nq);
    q[idx]    = qg[((size_t)n * nq + h) * (M2_HEAD * 2) + d];
    gate[(size_t)n * (nq * M2_HEAD) + h * M2_HEAD + d] =
        qg[((size_t)n * nq + h) * (M2_HEAD * 2) + M2_HEAD + d];
}

// Partial NEOX RoPE on the first ROT dims of each head, positions = row index. In-place.
__global__ void m2_partial_rope_kernel(float *x, int n_heads, int rows) {
    int i   = blockIdx.x * blockDim.x + threadIdx.x;     // 0..ROT/2-1
    int h   = blockIdx.y;
    int row = blockIdx.z;
    if (i >= M2_ROT / 2 || h >= n_heads || row >= rows) return;
    float *hd = x + ((size_t)row * n_heads + h) * M2_HEAD;
    float inv = powf(M2_THETA, -(float)(2 * i) / (float)M2_ROT);
    float ang = (float)row * inv;
    float c = cosf(ang), s = sinf(ang);
    float a = hd[i], b = hd[i + M2_ROT / 2];
    hd[i]             = a * c - b * s;
    hd[i + M2_ROT / 2] = b * c + a * s;
}

// Causal GQA softmax: out[N,NQ,HEAD] from q[N,NQ,HEAD], k/v[N,NKV,HEAD]. One block per
// (query-head, query-row); scores cached in dynamic shared (size = rows floats).
__global__ void m2_gqa_attn_kernel(float *out, const float *q, const float *k, const float *v,
                                   int N) {
    int hd  = blockIdx.x;        // query head 0..NQ-1
    int qi  = blockIdx.y;        // query row
    if (hd >= M2_NQ || qi >= N) return;
    int kv  = hd / (M2_NQ / M2_NKV);
    int tid = threadIdx.x;
    extern __shared__ float sc[];     // [N] scores
    const float *qr = q + ((size_t)qi * M2_NQ + hd) * M2_HEAD;
    float scale = rsqrtf((float)M2_HEAD);
    // scores for keys 0..qi
    for (int j = tid; j <= qi; j += blockDim.x) {
        const float *kr = k + ((size_t)j * M2_NKV + kv) * M2_HEAD;
        float acc = 0.f;
        for (int d = 0; d < M2_HEAD; d++) acc += qr[d] * kr[d];
        sc[j] = acc * scale;
    }
    __syncthreads();
    __shared__ float red[32];
    // max
    float m = -1e30f;
    for (int j = tid; j <= qi; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = block_reduce_max(m, red);
    __shared__ float msh;
    if (tid == 0) msh = m;
    __syncthreads();
    m = msh;
    // exp + sum
    float ssum = 0.f;
    for (int j = tid; j <= qi; j += blockDim.x) { float e = __expf(sc[j] - m); sc[j] = e; ssum += e; }
    ssum = block_reduce_sum(ssum, red);
    __shared__ float ssh;
    if (tid == 0) ssh = ssum;
    __syncthreads();
    float inv = 1.f / ssh;
    // weighted sum of v over head_dim (thread per dim)
    for (int d = tid; d < M2_HEAD; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j <= qi; j++)
            acc += sc[j] * v[((size_t)j * M2_NKV + kv) * M2_HEAD + d];
        out[((size_t)qi * M2_NQ + hd) * M2_HEAD + d] = acc * inv;
    }
}

// attn[i] *= sigmoid(gate[i])
__global__ void m2_sigmoid_gate_mul_kernel(float *attn, const float *gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    attn[i] *= 1.f / (1.f + __expf(-g));
}

// Causal depthwise conv1d (k=4) over CONVDIM channels + SiLU. out[t,c] =
// silu( sum_j cw[c*4+j] * in[t-3+j, c] ), zero left-pad. in/out [N, CONVDIM].
__global__ void m2_conv_silu_kernel(float *out, const float *in, const float *cw, int N) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (c >= M2_CONVDIM || t >= N) return;
    float acc = 0.f;
    #pragma unroll
    for (int j = 0; j < M2_CK; j++) {
        int tt = t - (M2_CK - 1) + j;
        if (tt >= 0) acc += cw[c * M2_CK + j] * in[(size_t)tt * M2_CONVDIM + c];
    }
    out[(size_t)t * M2_CONVDIM + c] = acc / (1.f + __expf(-acc));   // SiLU
}

// L2-norm per head over head_dim: x[rows,n_heads,sd] /= sqrt(sum(x^2)+eps). In-place.
__global__ void m2_l2norm_heads_kernel(float *x, int n_heads, int sd, int rows) {
    int head = blockIdx.x, row = blockIdx.y;
    if (head >= n_heads || row >= rows) return;
    __shared__ float red[32];
    float *h = x + ((size_t)row * n_heads + head) * sd;
    float ss = 0.f;
    for (int i = threadIdx.x; i < sd; i += blockDim.x) ss += h[i] * h[i];
    ss = block_reduce_sum(ss, red);
    float invn = rsqrtf(ss + M2_EPS);
    for (int i = threadIdx.x; i < sd; i += blockDim.x) h[i] *= invn;
}

// Per-(token,v-head) decay g = ssm_a*softplus(a+dt_bias) and beta = sigmoid(b).
__global__ void m2_decay_beta_kernel(float *g_out, float *beta_out, const float *a,
                                     const float *b, const float *ssm_a, const float *dt_bias,
                                     int N, int tsr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;     // N*TSR
    if (idx >= N * tsr) return;
    int h = idx % tsr;
    float x = a[idx] + dt_bias[h];
    float sp = (x > 0.f) ? (x + log1pf(__expf(-x))) : log1pf(__expf(x));   // softplus
    g_out[idx]    = ssm_a[h] * sp;
    beta_out[idx] = 1.f / (1.f + __expf(-b[idx]));
}

// Gated RMSNorm + SiLU(z): out[N,VALD] = RMSNorm(core, ssm_norm) * silu(z), per v-head.
__global__ void m2_gated_norm_kernel(float *out, const float *core, const float *z,
                                     const float *ssm_norm, int N, int nvh, int vald) {
    int vh = blockIdx.x, row = blockIdx.y;
    if (vh >= nvh || row >= N) return;
    __shared__ float red[32];
    const float *c = core + ((size_t)row * nvh + vh) * M2_SD;
    const float *zz = z   + ((size_t)row * nvh + vh) * M2_SD;
    float ss = 0.f;
    for (int i = threadIdx.x; i < M2_SD; i += blockDim.x) ss += c[i] * c[i];
    ss = block_reduce_sum(ss, red);
    float rms = rsqrtf(ss / M2_SD + M2_EPS);
    for (int i = threadIdx.x; i < M2_SD; i += blockDim.x) {
        float zv = zz[i];
        float silu = zv / (1.f + __expf(-zv));
        // RMSNorm(o)*ssm_norm gain, then * SiLU(z) gate (gate NOT in the variance).
        out[(size_t)row * vald + vh * M2_SD + i] = c[i] * rms * ssm_norm[i] * silu;
    }
}

// GDN single-step delta-rule recurrence. core[N,NVH,SD] from q,k[NPAD,NKH,SD], v[NPAD,NVH,SD],
// g,beta[NPAD,NVH]. One block per v-head; fp32 state S[SD(k)×SD(v)] in shared. q scaled 1/sqrt(SD).
__global__ void m2_gdn_recurrent_kernel(float *core, const float *q, const float *k,
                                        const float *v, const float *g, const float *beta, int N) {
    int vh  = blockIdx.x;
    int kh  = vh % M2_NKH;   // TILE expand (HF repeat): v-head vh ↔ k/q-head vh % NKH
    int tid = threadIdx.x;
    extern __shared__ float sm[];
    float *S   = sm;                 // [SD*SD] k-major: S[kd*SD+vd]
    float *kt  = S + M2_SD * M2_SD;  // [SD]
    float *qt  = kt + M2_SD;         // [SD]
    float *dlt = qt + M2_SD;         // [SD]
    for (int idx = tid; idx < M2_SD * M2_SD; idx += blockDim.x) S[idx] = 0.f;
    __syncthreads();
    float scale = rsqrtf((float)M2_SD);
    for (int t = 0; t < N; t++) {
        if (tid < M2_SD) {
            kt[tid] = k[((size_t)t * M2_NKH + kh) * M2_SD + tid];
            qt[tid] = q[((size_t)t * M2_NKH + kh) * M2_SD + tid] * scale;
        }
        __syncthreads();
        float gt = __expf(g[(size_t)t * M2_NVH + vh]);
        float bt = beta[(size_t)t * M2_NVH + vh];
        for (int idx = tid; idx < M2_SD * M2_SD; idx += blockDim.x) S[idx] *= gt;
        __syncthreads();
        if (tid < M2_SD) {                            // kv_mem + delta, thread per v-dim
            float acc = 0.f;
            for (int kd = 0; kd < M2_SD; kd++) acc += S[kd * M2_SD + tid] * kt[kd];
            float vv = v[((size_t)t * M2_NVH + vh) * M2_SD + tid];
            dlt[tid] = (vv - acc) * bt;
        }
        __syncthreads();
        for (int idx = tid; idx < M2_SD * M2_SD; idx += blockDim.x) {   // S += k ⊗ delta
            int kd = idx / M2_SD, vd = idx % M2_SD;
            S[idx] += kt[kd] * dlt[vd];
        }
        __syncthreads();
        if (tid < M2_SD) {                            // o = S^T q
            float acc = 0.f;
            for (int kd = 0; kd < M2_SD; kd++) acc += S[kd * M2_SD + tid] * qt[kd];
            core[((size_t)t * M2_NVH + vh) * M2_SD + tid] = acc;
        }
        __syncthreads();
    }
}

// GDN chunked-scan (prefill form): one block per v-head; chunk size CHUNK, head_dim SD. fp32
// inter-chunk state S in shared, k-chunk cached in shared. Per-chunk WY/UT matrices live in
// per-v-head global scratch (Tm,u,kcd,vnew,aintra). Mathematically identical to the recurrence.
// scratch layout per v-head: [Tm(CS*CS) | u(CS*SD) | kcd(CS*SD) | vnew(CS*SD) | aintra(CS*CS)].
#define M2_SCR_PER ( M2_CHUNK*M2_CHUNK + M2_CHUNK*M2_SD*3 + M2_CHUNK*M2_CHUNK )
__global__ void m2_gdn_chunk_kernel(float *core, const float *q, const float *k, const float *v,
                                    const float *g, const float *beta, float *scratch,
                                    int N, int NPAD) {
    int vh  = blockIdx.x;
    int kh  = vh % M2_NKH;   // TILE expand (HF repeat): v-head vh ↔ k/q-head vh % NKH
    int tid = threadIdx.x;
    const int CS = M2_CHUNK, SD = M2_SD;
    extern __shared__ float sm[];
    float *S    = sm;                 // [SD*SD]
    float *kch  = S + SD * SD;        // [CS*SD] cached k-chunk
    float *gc   = kch + CS * SD;      // [CS] cumulative decay
    float *bet  = gc + CS;            // [CS] beta per chunk row
    float *rowb = bet + CS;           // [CS] scratch row for fwd-subst
    float *Tm    = scratch + (size_t)vh * M2_SCR_PER;
    float *u     = Tm + CS * CS;
    float *kcd   = u + CS * SD;
    float *vnew  = kcd + CS * SD;
    float *aintra= vnew + CS * SD;
    float scale = rsqrtf((float)SD);
    for (int idx = tid; idx < SD * SD; idx += blockDim.x) S[idx] = 0.f;
    __syncthreads();

    int n_chunks = NPAD / CS;
    for (int c = 0; c < n_chunks; c++) {
        // load chunk k into shared, beta, and (serial) cumulative decay gc
        for (int idx = tid; idx < CS * SD; idx += blockDim.x) {
            int r = idx / SD, d = idx % SD;
            int gt = c * CS + r;
            kch[idx] = (gt < N) ? k[((size_t)gt * M2_NKH + kh) * SD + d] : 0.f;
        }
        for (int r = tid; r < CS; r += blockDim.x) {
            int gt = c * CS + r;
            bet[r] = (gt < N) ? beta[(size_t)gt * M2_NVH + vh] : 0.f;
        }
        __syncthreads();
        if (tid == 0) {
            float acc = 0.f;
            for (int r = 0; r < CS; r++) {
                int gt = c * CS + r;
                acc += (gt < N) ? g[(size_t)gt * M2_NVH + vh] : 0.f;
                gc[r] = acc;
            }
        }
        __syncthreads();
        // A[i,j] = -<k_beta[i],k[j]>*exp(gc[i]-gc[j]) for i>j else 0   (k_beta = k*beta)
        for (int idx = tid; idx < CS * CS; idx += blockDim.x) {
            int i = idx / CS, j = idx % CS;
            float val = 0.f;
            if (i > j) {
                float dot = 0.f;
                for (int d = 0; d < SD; d++) dot += kch[i * SD + d] * bet[i] * kch[j * SD + d];
                val = -dot * __expf(gc[i] - gc[j]);
            }
            Tm[idx] = val;
        }
        __syncthreads();
        // forward substitution: Tm = (I - A)^{-1} - I  (then +I below). Sequential over rows.
        for (int i = 1; i < CS; i++) {
            for (int m = tid; m < i; m += blockDim.x) rowb[m] = Tm[i * CS + m];
            __syncthreads();
            for (int m = tid; m < i; m += blockDim.x) {
                float acc = rowb[m];
                for (int nn = 0; nn < i; nn++) acc += rowb[nn] * Tm[nn * CS + m];
                Tm[i * CS + m] = acc;
            }
            __syncthreads();
        }
        for (int i = tid; i < CS; i += blockDim.x) Tm[i * CS + i] += 1.f;   // +I
        __syncthreads();
        // u = T @ v_beta ; kcd = T @ (k_beta*exp(gc))
        for (int idx = tid; idx < CS * SD; idx += blockDim.x) {
            int i = idx / SD, d = idx % SD;
            float su = 0.f, sk = 0.f;
            for (int s = 0; s < CS; s++) {
                float t = Tm[i * CS + s];
                int gt = c * CS + s;
                float vv = (gt < N) ? v[((size_t)gt * M2_NVH + vh) * SD + d] : 0.f;
                su += t * (vv * bet[s]);
                sk += t * (kch[s * SD + d] * bet[s] * __expf(gc[s]));
            }
            u[idx] = su; kcd[idx] = sk;
        }
        __syncthreads();
        // v_new = u - kcd @ S
        for (int idx = tid; idx < CS * SD; idx += blockDim.x) {
            int i = idx / SD, d = idx % SD;
            float vp = 0.f;
            for (int kd = 0; kd < SD; kd++) vp += kcd[i * SD + kd] * S[kd * SD + d];
            vnew[idx] = u[idx] - vp;
        }
        __syncthreads();
        // a_intra[i,j] = <q[i],k[j]>*exp(gc[i]-gc[j]) for i>=j else 0  (q scaled)
        for (int idx = tid; idx < CS * CS; idx += blockDim.x) {
            int i = idx / CS, j = idx % CS;
            float val = 0.f;
            if (i >= j) {
                int gti = c * CS + i;
                float dot = 0.f;
                for (int d = 0; d < SD; d++) {
                    float qv = (gti < N) ? q[((size_t)gti * M2_NKH + kh) * SD + d] * scale : 0.f;
                    dot += qv * kch[j * SD + d];
                }
                val = dot * __expf(gc[i] - gc[j]);
            }
            aintra[idx] = val;
        }
        __syncthreads();
        // core[i,d] = (q[i]*exp(gc[i])) @ S  +  a_intra @ v_new
        for (int idx = tid; idx < CS * SD; idx += blockDim.x) {
            int i = idx / SD, d = idx % SD;
            int gti = c * CS + i;
            if (gti >= N) continue;
            float inter = 0.f;
            float egi = __expf(gc[i]);
            for (int kd = 0; kd < SD; kd++)
                inter += q[((size_t)gti * M2_NKH + kh) * SD + kd] * scale * egi * S[kd * SD + d];
            float intra = 0.f;
            for (int j = 0; j <= i; j++) intra += aintra[i * CS + j] * vnew[j * SD + d];
            core[((size_t)gti * M2_NVH + vh) * SD + d] = inter + intra;
        }
        __syncthreads();
        // S = S*exp(gc[last]) + sum_r k[r]*exp(gc[last]-gc[r]) ⊗ v_new[r]
        float glast = gc[CS - 1];
        for (int idx = tid; idx < SD * SD; idx += blockDim.x) {
            int kd = idx / SD, d = idx % SD;
            float acc = S[idx] * __expf(glast);
            for (int r = 0; r < CS; r++)
                acc += kch[r * SD + kd] * __expf(glast - gc[r]) * vnew[r * SD + d];
            S[idx] = acc;
        }
        __syncthreads();
    }
}

// ─── host helpers for the self-test ───────────────────────────────────────────
static float *m2_dev(const float *host, size_t n) {
    float *d = nullptr;
    if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, host, n * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}
static void m2_gemm(float *out, const float *in, const float *W, int N, int INN, int OUT,
                    cudaStream_t stream = 0) {
    dim3 grid((unsigned)OUT, (unsigned)N);
    m2_gemm_kernel<<<grid, 128, 0, stream>>>(out, in, W, N, INN, OUT);
}
// max-abs-rel error of dev[n] vs reference dev_ref[n]: max|a-b| / max|ref|.
static double m2_relerr(const float *d_a, const float *d_ref, size_t n, double *out_maxabs) {
    float *a = (float*)malloc(n * sizeof(float)), *b = (float*)malloc(n * sizeof(float));
    cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_ref, n * sizeof(float), cudaMemcpyDeviceToHost);
    double mad = 0, mref = 0;
    for (size_t i = 0; i < n; i++) { double dd = fabs((double)a[i]-b[i]); if (dd>mad) mad=dd;
        double rr = fabs((double)b[i]); if (rr>mref) mref=rr; }
    free(a); free(b);
    if (out_maxabs) *out_maxabs = mad;
    return mad / (mref > 1e-9 ? mref : 1e-9);
}

int qwen35_m2_layer_selftest(const char *ref_bin_path) {
    FILE *f = fopen(ref_bin_path, "rb");
    if (!f) { fprintf(stderr, "M2: cannot open %s (run cuda/qwen35_layer_ref.py first)\n", ref_bin_path); return 1; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    char *blob = (char*)malloc(fsz);
    if (fread(blob, 1, fsz, f) != (size_t)fsz) { fprintf(stderr, "M2: short read\n"); fclose(f); return 1; }
    fclose(f);
    int *hdr = (int*)blob;
    if (hdr[0] != 0x51573532) { fprintf(stderr, "M2: bad magic\n"); return 1; }
    int N = hdr[1], H = hdr[2];
    if (H != M2_H) { fprintf(stderr, "M2: H mismatch %d\n", H); return 1; }
    float *cur = (float*)(blob + 12);
    auto take = [&](size_t n) -> float* { float *p = cur; cur += n; return p; };

    // FULL block
    float *h_attn_norm = take(M2_H);
    float *h_Wq = take((size_t)M2_NQ*M2_HEAD*2*M2_H);
    float *h_Wk = take((size_t)M2_NKV*M2_HEAD*M2_H);
    float *h_Wv = take((size_t)M2_NKV*M2_HEAD*M2_H);
    float *h_Wo = take((size_t)M2_H*(M2_NQ*M2_HEAD));
    float *h_qn = take(M2_HEAD);
    float *h_kn = take(M2_HEAD);
    float *h_in_full = take((size_t)N*M2_H);
    float *h_ref_full = take((size_t)N*M2_H);
    // GDN block
    float *h_gn = take(M2_H);
    float *h_Wqkv = take((size_t)M2_CONVDIM*M2_H);
    float *h_Wgate = take((size_t)M2_VALD*M2_H);
    float *h_Wbeta = take((size_t)M2_TSR*M2_H);
    float *h_Walpha = take((size_t)M2_TSR*M2_H);
    float *h_conv = take((size_t)M2_CONVDIM*M2_CK);
    float *h_ssma = take(M2_TSR);
    float *h_dtb = take(M2_TSR);
    float *h_ssmn = take(M2_SD);
    float *h_Wout = take((size_t)M2_H*M2_VALD);
    float *h_in_gdn = take((size_t)N*M2_H);
    float *h_ref_recur = take((size_t)N*M2_H);
    float *h_ref_chunk = take((size_t)N*M2_H);
    if ((char*)cur != blob + fsz) { fprintf(stderr, "M2: layout mismatch (cur off %ld, fsz %ld)\n",
        (long)((char*)cur - blob), fsz); return 1; }

    int rc = 0;
    // Opt-in to >48KB dynamic shared for the GDN state kernels. The attribute MUST be <= the
    // device's MaxSharedMemoryPerBlockOptin or cudaFuncSetAttribute fails and the launch (which
    // requests the larger size) silently aborts, leaving the output zeroed.
    size_t smR_bytes = ((size_t)M2_SD*M2_SD + 3*M2_SD)*sizeof(float);
    size_t smC_bytes = ((size_t)M2_SD*M2_SD + M2_CHUNK*M2_SD + 3*M2_CHUNK)*sizeof(float);
    cudaError_t aR = cudaFuncSetAttribute(m2_gdn_recurrent_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smR_bytes);
    cudaError_t aC = cudaFuncSetAttribute(m2_gdn_chunk_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smC_bytes);
    if (aR != cudaSuccess || aC != cudaSuccess) {
        fprintf(stderr, "M2: cudaFuncSetAttribute failed (recur %zuB:%s, chunk %zuB:%s) — "
            "exceeds device MaxSharedMemoryPerBlockOptin\n",
            smR_bytes, cudaGetErrorString(aR), smC_bytes, cudaGetErrorString(aC));
        free(blob); return 1;
    }

    // ───────── FULL layer ─────────
    {
        float *Wq=m2_dev(h_Wq,(size_t)M2_NQ*M2_HEAD*2*M2_H), *Wk=m2_dev(h_Wk,(size_t)M2_NKV*M2_HEAD*M2_H),
              *Wv=m2_dev(h_Wv,(size_t)M2_NKV*M2_HEAD*M2_H), *Wo=m2_dev(h_Wo,(size_t)M2_H*M2_NQ*M2_HEAD),
              *an=m2_dev(h_attn_norm,M2_H), *qn=m2_dev(h_qn,M2_HEAD), *kn=m2_dev(h_kn,M2_HEAD),
              *hin=m2_dev(h_in_full,(size_t)N*M2_H), *ref=m2_dev(h_ref_full,(size_t)N*M2_H);
        float *x,*qg,*q,*gate,*k,*v,*attn,*out;
        cudaMalloc(&x,(size_t)N*M2_H*sizeof(float));
        cudaMalloc(&qg,(size_t)N*M2_NQ*M2_HEAD*2*sizeof(float));
        cudaMalloc(&q,(size_t)N*M2_NQ*M2_HEAD*sizeof(float));
        cudaMalloc(&gate,(size_t)N*M2_VALD*sizeof(float));
        cudaMalloc(&k,(size_t)N*M2_NKV*M2_HEAD*sizeof(float));
        cudaMalloc(&v,(size_t)N*M2_NKV*M2_HEAD*sizeof(float));
        cudaMalloc(&attn,(size_t)N*M2_NQ*M2_HEAD*sizeof(float));
        cudaMalloc(&out,(size_t)N*M2_H*sizeof(float));
        rms_norm_rows_kernel<<<N,256,32*sizeof(float)>>>(x,hin,an,M2_H,N,M2_EPS);
        m2_gemm(qg,x,Wq,N,M2_H,M2_NQ*M2_HEAD*2);
        m2_gemm(k,x,Wk,N,M2_H,M2_NKV*M2_HEAD);
        m2_gemm(v,x,Wv,N,M2_H,M2_NKV*M2_HEAD);
        m2_split_query_gate_kernel<<<(N*M2_NQ*M2_HEAD+255)/256,256>>>(q,gate,qg,N,M2_NQ);
        per_head_rms_norm_rows_kernel<<<dim3(M2_NQ,N),256,32*sizeof(float)>>>(q,qn,M2_NQ,M2_HEAD,N,M2_EPS);
        per_head_rms_norm_rows_kernel<<<dim3(M2_NKV,N),256,32*sizeof(float)>>>(k,kn,M2_NKV,M2_HEAD,N,M2_EPS);
        m2_partial_rope_kernel<<<dim3((M2_ROT/2+31)/32,M2_NQ,N),32>>>(q,M2_NQ,N);
        m2_partial_rope_kernel<<<dim3((M2_ROT/2+31)/32,M2_NKV,N),32>>>(k,M2_NKV,N);
        m2_gqa_attn_kernel<<<dim3(M2_NQ,N),256,(size_t)N*sizeof(float)>>>(attn,q,k,v,N);
        m2_sigmoid_gate_mul_kernel<<<(N*M2_VALD+255)/256,256>>>(attn,gate,N*M2_VALD);
        m2_gemm(out,attn,Wo,N,M2_NQ*M2_HEAD,M2_H);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { fprintf(stderr,"M2 FULL: %s\n",cudaGetErrorString(e)); rc=1; }
        double mad=0, rel=m2_relerr(out,ref,(size_t)N*M2_H,&mad);
        printf("M2 FULL-attn  : max-abs %.3e  rel %.3e  -> %s\n", mad, rel, rel<1e-2?"PASS":"FAIL");
        if (rel>=1e-2) rc=1;
        cudaFree(Wq);cudaFree(Wk);cudaFree(Wv);cudaFree(Wo);cudaFree(an);cudaFree(qn);cudaFree(kn);
        cudaFree(hin);cudaFree(ref);cudaFree(x);cudaFree(qg);cudaFree(q);cudaFree(gate);cudaFree(k);
        cudaFree(v);cudaFree(attn);cudaFree(out);
    }

    // ───────── GDN layer (recurrent + chunk) ─────────
    {
        int NPAD = ((N + M2_CHUNK - 1) / M2_CHUNK) * M2_CHUNK;
        float *Wqkv=m2_dev(h_Wqkv,(size_t)M2_CONVDIM*M2_H), *Wgate=m2_dev(h_Wgate,(size_t)M2_VALD*M2_H),
              *Wbeta=m2_dev(h_Wbeta,(size_t)M2_TSR*M2_H), *Walpha=m2_dev(h_Walpha,(size_t)M2_TSR*M2_H),
              *conv=m2_dev(h_conv,(size_t)M2_CONVDIM*M2_CK), *ssma=m2_dev(h_ssma,M2_TSR),
              *dtb=m2_dev(h_dtb,M2_TSR), *ssmn=m2_dev(h_ssmn,M2_SD), *Wout=m2_dev(h_Wout,(size_t)M2_H*M2_VALD),
              *gn=m2_dev(h_gn,M2_H), *hin=m2_dev(h_in_gdn,(size_t)N*M2_H),
              *refR=m2_dev(h_ref_recur,(size_t)N*M2_H), *refC=m2_dev(h_ref_chunk,(size_t)N*M2_H);
        float *x,*qkv,*z,*b,*a,*conv_out,*qk_q,*qk_k,*vv,*gg,*bb,*core,*gnorm,*outR,*outC,*scratch;
        cudaMalloc(&x,(size_t)N*M2_H*sizeof(float));
        cudaMalloc(&qkv,(size_t)N*M2_CONVDIM*sizeof(float));
        cudaMalloc(&z,(size_t)N*M2_VALD*sizeof(float));
        cudaMalloc(&b,(size_t)N*M2_TSR*sizeof(float));
        cudaMalloc(&a,(size_t)N*M2_TSR*sizeof(float));
        cudaMalloc(&conv_out,(size_t)N*M2_CONVDIM*sizeof(float));
        cudaMalloc(&qk_q,(size_t)NPAD*M2_NKH*M2_SD*sizeof(float));
        cudaMalloc(&qk_k,(size_t)NPAD*M2_NKH*M2_SD*sizeof(float));
        cudaMalloc(&vv,(size_t)NPAD*M2_NVH*M2_SD*sizeof(float));
        cudaMalloc(&gg,(size_t)NPAD*M2_TSR*sizeof(float));
        cudaMalloc(&bb,(size_t)NPAD*M2_TSR*sizeof(float));
        cudaMalloc(&core,(size_t)N*M2_NVH*M2_SD*sizeof(float));
        cudaMalloc(&gnorm,(size_t)N*M2_VALD*sizeof(float));
        cudaMalloc(&outR,(size_t)N*M2_H*sizeof(float));
        cudaMalloc(&outC,(size_t)N*M2_H*sizeof(float));
        cudaMalloc(&scratch,(size_t)M2_NVH*M2_SCR_PER*sizeof(float));
        cudaMemset(qk_q,0,(size_t)NPAD*M2_NKH*M2_SD*sizeof(float));
        cudaMemset(qk_k,0,(size_t)NPAD*M2_NKH*M2_SD*sizeof(float));
        cudaMemset(vv,0,(size_t)NPAD*M2_NVH*M2_SD*sizeof(float));
        cudaMemset(gg,0,(size_t)NPAD*M2_TSR*sizeof(float));
        cudaMemset(bb,0,(size_t)NPAD*M2_TSR*sizeof(float));
        rms_norm_rows_kernel<<<N,256,32*sizeof(float)>>>(x,hin,gn,M2_H,N,M2_EPS);
        m2_gemm(qkv,x,Wqkv,N,M2_H,M2_CONVDIM);
        m2_gemm(z,x,Wgate,N,M2_H,M2_VALD);
        m2_gemm(b,x,Wbeta,N,M2_H,M2_TSR);
        m2_gemm(a,x,Walpha,N,M2_H,M2_TSR);
        m2_conv_silu_kernel<<<dim3((M2_CONVDIM+127)/128,N),128>>>(conv_out,qkv,conv,N);
        // split conv_out[N,8192] -> q[NPAD,16,128], k[NPAD,16,128], v[NPAD,32,128]
        // q = conv[:, 0:2048], k = conv[:, 2048:4096], v = conv[:, 4096:8192]
        for (int n=0;n<N;n++) {
            cudaMemcpy(qk_q+(size_t)n*M2_NKH*M2_SD, conv_out+(size_t)n*M2_CONVDIM,            (size_t)M2_KEYD*sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(qk_k+(size_t)n*M2_NKH*M2_SD, conv_out+(size_t)n*M2_CONVDIM+M2_KEYD,    (size_t)M2_KEYD*sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(vv  +(size_t)n*M2_NVH*M2_SD, conv_out+(size_t)n*M2_CONVDIM+2*M2_KEYD,  (size_t)M2_VALD*sizeof(float), cudaMemcpyDeviceToDevice);
        }
        m2_l2norm_heads_kernel<<<dim3(M2_NKH,N),128>>>(qk_q,M2_NKH,M2_SD,N);
        m2_l2norm_heads_kernel<<<dim3(M2_NKH,N),128>>>(qk_k,M2_NKH,M2_SD,N);
        m2_decay_beta_kernel<<<(N*M2_TSR+255)/256,256>>>(gg,bb,a,b,ssma,dtb,N,M2_TSR);
        size_t smR = ((size_t)M2_SD*M2_SD + 3*M2_SD)*sizeof(float);
        m2_gdn_recurrent_kernel<<<M2_NVH,128,smR>>>(core,qk_q,qk_k,vv,gg,bb,N);
        m2_gated_norm_kernel<<<dim3(M2_NVH,N),128,32*sizeof(float)>>>(gnorm,core,z,ssmn,N,M2_NVH,M2_VALD);
        m2_gemm(outR,gnorm,Wout,N,M2_VALD,M2_H);
        cudaError_t e1 = cudaDeviceSynchronize();
        if (e1 != cudaSuccess) { fprintf(stderr,"M2 GDN-recur: %s\n",cudaGetErrorString(e1)); rc=1; }
        double madR=0, relR=m2_relerr(outR,refR,(size_t)N*M2_H,&madR);
        printf("M2 GDN recur  : max-abs %.3e  rel %.3e  -> %s\n", madR, relR, relR<1e-2?"PASS":"FAIL");
        if (relR>=1e-2) rc=1;
        // chunk path (reuse same projections / conv / l2norm / decay)
        cudaMemset(core,0,(size_t)N*M2_NVH*M2_SD*sizeof(float));
        size_t smC = ((size_t)M2_SD*M2_SD + M2_CHUNK*M2_SD + 3*M2_CHUNK)*sizeof(float);
        m2_gdn_chunk_kernel<<<M2_NVH,128,smC>>>(core,qk_q,qk_k,vv,gg,bb,scratch,N,NPAD);
        m2_gated_norm_kernel<<<dim3(M2_NVH,N),128,32*sizeof(float)>>>(gnorm,core,z,ssmn,N,M2_NVH,M2_VALD);
        m2_gemm(outC,gnorm,Wout,N,M2_VALD,M2_H);
        cudaError_t e2 = cudaDeviceSynchronize();
        if (e2 != cudaSuccess) { fprintf(stderr,"M2 GDN-chunk: %s\n",cudaGetErrorString(e2)); rc=1; }
        double madC=0, relC=m2_relerr(outC,refC,(size_t)N*M2_H,&madC);
        printf("M2 GDN chunk  : max-abs %.3e  rel %.3e  -> %s\n", madC, relC, relC<1e-2?"PASS":"FAIL");
        if (relC>=1e-2) rc=1;
        double madE=0, relE=m2_relerr(outC,outR,(size_t)N*M2_H,&madE);
        printf("M2 GDN chunk==recur: max-abs %.3e  rel %.3e  -> %s\n", madE, relE, relE<1e-3?"PASS":"FAIL");
        if (relE>=1e-3) rc=1;
        cudaFree(Wqkv);cudaFree(Wgate);cudaFree(Wbeta);cudaFree(Walpha);cudaFree(conv);cudaFree(ssma);
        cudaFree(dtb);cudaFree(ssmn);cudaFree(Wout);cudaFree(gn);cudaFree(hin);cudaFree(refR);cudaFree(refC);
        cudaFree(x);cudaFree(qkv);cudaFree(z);cudaFree(b);cudaFree(a);cudaFree(conv_out);cudaFree(qk_q);
        cudaFree(qk_k);cudaFree(vv);cudaFree(gg);cudaFree(bb);cudaFree(core);cudaFree(gnorm);cudaFree(outR);
        cudaFree(outC);cudaFree(scratch);
    }
    free(blob);
    return rc;
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// ─── Qwen3.5 hybrid (qwen35) M3: single-seq hybrid forward (8/8 greedy argmax parity) ───────
// ═══════════════════════════════════════════════════════════════════════════════════════════
// Token-by-token forward over the qwen35 GGUF that carries the GDN recurrent state + conv ring
// (LINEAR layers) and a per-FULL-layer KV cache across tokens. Per-layer dispatch is driven by
// cfg.attn_kind[]: LINEAR → gated-deltanet step (fp32 state), FULL → output-gated softmax GQA.
//
// Reuse map: projections / FFN / lm_head go through the VALIDATED quantized dp4a gemv_w
// (native Q4_K / Q6_K — the same kernels Qwen3 reached 8/8 with); the only non-dp4a weight is
// the LINEAR layers' Q5_K attn_qkv (no native Q5_K kernel) which is dequantized to fp32 and run
// through the M2 fp32 m2_gemm. The mixer math reuses the M2 kernels (split q|gate, per-head
// q/k norm, partial NEOX RoPE, GDN conv/l2norm/decay/recurrence/gated-norm) which M2 already
// validated < 1e-2 vs the HF torch reference. The embedding uses the engine's Q8_0
// token_embd convert (d_token_embd). Everything here is gated on GEMMA4_ARCH_QWEN3_5.

// Partial NEOX RoPE on the first M2_ROT dims of each head at an ARBITRARY position `pos`
// (the M2 kernel hard-codes pos = row index; decode needs the real sequence position). 1 row.
__global__ void qwen35_rope_pos_kernel(float *x, int n_heads, int pos) {
    int i   = blockIdx.x * blockDim.x + threadIdx.x;     // 0..ROT/2-1
    int h   = blockIdx.y;
    if (i >= M2_ROT / 2 || h >= n_heads) return;
    float *hd = x + (size_t)h * M2_HEAD;
    float inv = powf(M2_THETA, -(float)(2 * i) / (float)M2_ROT);
    float ang = (float)pos * inv;
    float c = cosf(ang), s = sinf(ang);
    float a = hd[i], b = hd[i + M2_ROT / 2];
    hd[i]              = a * c - b * s;
    hd[i + M2_ROT / 2] = b * c + a * s;
}

// Single-query causal GQA softmax against the KV cache: query head hd at position `pos`
// attends cached keys 0..pos. Kc/Vc are [pos+1][nkv][HEAD] (current pos already written).
// One block per query head; scores in dynamic shared (pos+1 floats). Mirrors m2_gqa_attn.
// nkv is RUNTIME: 4 on the 9B, 2 on the 35B-A3B MoE (all other dims shared).
__global__ void qwen35_attn_step_kernel(float *out, const float *q,
                                        const float *Kc, const float *Vc, int pos, int nkv) {
    int hd  = blockIdx.x;            // query head 0..NQ-1
    int tid = threadIdx.x;
    int kv  = hd / (M2_NQ / nkv);
    extern __shared__ float sc[];    // [pos+1] scores
    const float *qr = q + (size_t)hd * M2_HEAD;
    float scale = rsqrtf((float)M2_HEAD);
    for (int j = tid; j <= pos; j += blockDim.x) {
        const float *kr = Kc + ((size_t)j * nkv + kv) * M2_HEAD;
        float acc = 0.f;
        for (int d = 0; d < M2_HEAD; d++) acc += qr[d] * kr[d];
        sc[j] = acc * scale;
    }
    __syncthreads();
    __shared__ float red[32];
    float m = -1e30f;
    for (int j = tid; j <= pos; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = block_reduce_max(m, red);
    __shared__ float msh; if (tid == 0) msh = m; __syncthreads(); m = msh;
    float ssum = 0.f;
    for (int j = tid; j <= pos; j += blockDim.x) { float e = __expf(sc[j] - m); sc[j] = e; ssum += e; }
    ssum = block_reduce_sum(ssum, red);
    __shared__ float ssh; if (tid == 0) ssh = ssum; __syncthreads();
    float inv = 1.f / ssh;
    for (int d = tid; d < M2_HEAD; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j <= pos; j++)
            acc += sc[j] * Vc[((size_t)j * nkv + kv) * M2_HEAD + d];
        out[(size_t)hd * M2_HEAD + d] = acc * inv;
    }
}

// Stateful causal depthwise conv1d (k=CK) + SiLU for ONE token over conv_dim channels. The
// previous CK-1 inputs live in a per-channel ring buffer `ring` [conv_dim][CK-1] (oldest..
// newest); after computing the output the ring is shifted left and the current input appended.
// Equivalent to m2_conv_silu's zero-left-pad form when the ring starts zeroed. cw is [conv_dim,CK].
__global__ void qwen35_conv_step_kernel(float *conv_out, const float *qkv, float *ring,
                                        const float *cw, int conv_dim) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const int K = M2_CK, HK = M2_CK - 1;
    float acc = 0.f;
    #pragma unroll
    for (int j = 0; j < HK; j++) acc += cw[c * K + j] * ring[(size_t)c * HK + j];
    acc += cw[c * K + HK] * qkv[c];
    conv_out[c] = acc / (1.f + __expf(-acc));                 // SiLU
    #pragma unroll
    for (int j = 0; j < HK - 1; j++) ring[(size_t)c * HK + j] = ring[(size_t)c * HK + j + 1];
    ring[(size_t)c * HK + (HK - 1)] = qkv[c];
}

// GDN single-step delta-rule recurrence for ONE token that LOADS the carried per-v-head fp32
// state S[SD×SD] from S_io, applies the step, and STORES it back. Identical arithmetic to
// m2_gdn_recurrent_kernel run one token at a time (the M2 chunk==recur gate proved the
// recurrence is correct). One block per v-head; q/k indexed by k-head kh = vh/(NVH/NKH).
__global__ void qwen35_gdn_step_kernel(float *core, const float *q, const float *k,
                                       const float *v, const float *g, const float *beta,
                                       float *S_io) {
    int vh  = blockIdx.x;
    // q/k (16 heads) expand to the 32 v-heads by TILING (v-head vh ↔ k/q-head vh % NKH), i.e.
    // HF's repeat(NVH/NKH) — NOT repeat_interleave (vh/(NVH/NKH)). Verified token-for-token vs
    // llama.cpp's GATED_DELTA_NET: interleave gives generic output, tile recovers " Paris".
    int kh  = vh % M2_NKH;
    int tid = threadIdx.x;
    extern __shared__ float sm[];
    float *S   = sm;                 // [SD*SD] k-major: S[kd*SD+vd]
    float *kt  = S + M2_SD * M2_SD;  // [SD]
    float *qt  = kt + M2_SD;         // [SD]
    float *dlt = qt + M2_SD;         // [SD]
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
    if (tid < M2_SD) {                            // kv_mem + delta, thread per v-dim
        float acc = 0.f;
        for (int kd = 0; kd < M2_SD; kd++) acc += S[kd * M2_SD + tid] * kt[kd];
        float vv = v[(size_t)vh * M2_SD + tid];
        dlt[tid] = (vv - acc) * bt;
    }
    __syncthreads();
    for (int idx = tid; idx < M2_SD * M2_SD; idx += blockDim.x) {   // S += k ⊗ delta
        int kd = idx / M2_SD, vd = idx % M2_SD;
        S[idx] += kt[kd] * dlt[vd];
    }
    __syncthreads();
    if (tid < M2_SD) {                            // o = S^T q
        float acc = 0.f;
        for (int kd = 0; kd < M2_SD; kd++) acc += S[kd * M2_SD + tid] * qt[kd];
        core[(size_t)vh * M2_SD + tid] = acc;
    }
    __syncthreads();
    for (int idx = tid; idx < M2_SD * M2_SD; idx += blockDim.x) Sg[idx] = S[idx];
}

// out[i] += y[i]   (residual add over n elements)
__global__ void qwen35_add_kernel(float *x, const float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// Greedy argmax over v[n] (single block, lowest-index tie-break) → *out_idx.
__global__ void qwen35_argmax_kernel(const float *v, int n, int *out_idx) {
    int tid = threadIdx.x;
    float bv = -1e30f; int bi = 0;
    for (int i = tid; i < n; i += blockDim.x) { float x = v[i]; if (x > bv) { bv = x; bi = i; } }
    __shared__ float sv[256]; __shared__ int si[256];
    sv[tid] = bv; si[tid] = bi; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (sv[tid + s] > sv[tid] || (sv[tid + s] == sv[tid] && si[tid + s] < si[tid])) {
                sv[tid] = sv[tid + s]; si[tid] = si[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *out_idx = si[0];
}

// Single-seq greedy hybrid forward. Fills out_ids[0..n_gen-1] with the greedy continuation of
// the prompt in_ids[0..n_prompt-1]. Returns 0 on success, non-zero on error. Token-by-token:
// each position runs one full hybrid layer stack, carrying GDN state + conv ring + KV cache.
// FP8/safetensors GDN uses the repeat-interleave head expansion (body defined near the FP8 oracle).
__global__ void qwen35_fp8_gdn_step_kernel(float *core, const float *q, const float *k,
                                           const float *v, const float *g, const float *beta, float *S_io);
extern "C" int qwen35_forward_greedy(gemma4_engine_t *eng, const int32_t *in_ids, int n_prompt,
                                     int32_t *out_ids, int n_gen) {
    if (!eng || eng->cfg.arch != GEMMA4_ARCH_QWEN3_5) {
        fprintf(stderr, "qwen35_forward_greedy: engine is not a qwen35 arch\n"); return -1;
    }
    const gemma4_model_config_t *c = &eng->cfg;
    // The M2/M3 mixer kernels bake the shared qwen35 geometry (head/GDN dims) as #defines;
    // H and NKV are RUNTIME (9B: 4096/4, 35B-A3B MoE: 2048/2). Refuse a baked-dim mismatch.
    if (c->head_dim != M2_HEAD || c->n_heads != M2_NQ || c->ssm_state_size != M2_SD ||
        c->ssm_group_count != M2_NKH || c->ssm_time_step_rank != M2_NVH ||
        c->ssm_conv_kernel != M2_CK || c->rotary_dim != M2_ROT) {
        fprintf(stderr, "qwen35_forward_greedy: geometry mismatch vs M2 constants\n"); return -2;
    }
    if (!eng->d_token_embd) {
        fprintf(stderr, "qwen35_forward_greedy: token_embd Q8_0 convert missing\n"); return -3;
    }
    cudaStream_t st = eng->stream;
    const float eps = 1e-6f;
    const int H = c->hidden_size, HD = M2_HEAD, NQ = M2_NQ, NKV = c->n_kv_global;
    const int INNER = M2_VALD, CONVD = M2_CONVDIM, KEYD = M2_KEYD;
    const int NKH = M2_NKH, NVH = M2_NVH, SD = M2_SD, TSR = M2_TSR, ROT = M2_ROT;
    const int I = c->intermediate, VOC = c->vocab_size, L = c->n_layers;
    // Positions processed: the last prompt token (which yields out_ids[0]) plus n_gen-1
    // re-fed drafted tokens. p ∈ [0, nsteps); out_ids[i] is produced at p = (n_prompt-1)+i.
    const int nsteps = n_prompt + n_gen - 1;
    (void)ROT; (void)KEYD;

    // ── GDN state shared-mem opt-in (S[SD×SD] + kt+qt+dlt = ~65.5 KB > 48 KB default). ──
    size_t smGDN = ((size_t)SD * SD + 3 * SD) * sizeof(float);
    if (cudaFuncSetAttribute(qwen35_gdn_step_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smGDN) != cudaSuccess) {
        fprintf(stderr, "qwen35_forward_greedy: GDN shared-mem opt-in failed (%zu B)\n", smGDN);
        return -4;
    }
    // FP8/safetensors GDN uses the interleave kernel — opt IT into the same dynamic-smem cap.
    if (eng->format == FORMAT_FP8_BLOCK &&
        cudaFuncSetAttribute(qwen35_fp8_gdn_step_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smGDN) != cudaSuccess) {
        fprintf(stderr, "qwen35_forward_greedy: FP8 GDN shared-mem opt-in failed (%zu B)\n", smGDN);
        return -4;
    }
    // Opt the fp32 attention ORACLE (qwen35_attn_step_kernel, (pos+1)*4 dynamic scores) into the
    // full per-block shared limit. Without this the oracle caps at the 48 KB default (~pos 12287),
    // so a long-context argmax-parity gate that reaches the real deploy context (up to q35_maxctx
    // ~25 k) is not runnable — exactly the missing gate the shipped flash-decoding + fp16-KV decode
    // (and any future recurrent-state precision change) needs to validate drift against. Test-only
    // path; the opt-in is free when the kernel requests less.
    {
        int optinMax = 49152, dev = 0; cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&optinMax, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        cudaError_t ae = cudaFuncSetAttribute(qwen35_attn_step_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, optinMax);  // best-effort; oracle only
        if (ae != cudaSuccess) cudaGetLastError();                    // do not poison pos-0 error check
    }

    auto Wq  = [&](uint64_t off) -> const uint8_t* { return weight_fp8(eng, off); };
    auto Wf  = [&](uint64_t off) -> const float*   {
        return (const float*)(eng->d_weights + (off - eng->tdata_start)); };

    int rc = 0;
    // ── per-token scratch ──
    int32_t *d_tok = nullptr; int *d_arg = nullptr;
    float *x=nullptr,*xn=nullptr,*qg=nullptr,*qb=nullptr,*gate=nullptr,*kb=nullptr,*vb=nullptr,
          *attn=nullptr,*mix=nullptr,*qkv=nullptr,*conv_out=nullptr,*zc=nullptr,*ac=nullptr,
          *bc=nullptr,*gg=nullptr,*bb=nullptr,*qh=nullptr,*kh=nullptr,*vh=nullptr,*core=nullptr,
          *gnorm=nullptr,*ffn_g=nullptr,*ffn_u=nullptr,*ffn_a=nullptr;
    // ── persistent per-layer state (only the relevant layer kind is touched) ──
    float **Kc = (float**)calloc(L, sizeof(float*));
    float **Vc = (float**)calloc(L, sizeof(float*));
    float **Sst = (float**)calloc(L, sizeof(float*));
    float **ring = (float**)calloc(L, sizeof(float*));
    if (!Kc || !Vc || !Sst || !ring) { rc = -5; goto cleanup; }

    #define CK_MALLOC(p, n) do { if (cudaMalloc(&(p), (size_t)(n) * sizeof(float)) != cudaSuccess) { \
        fprintf(stderr, "qwen35_forward_greedy: cudaMalloc failed (%s)\n", #p); rc = -6; goto cleanup; } } while(0)
    if (cudaMalloc(&d_tok, sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&d_arg, sizeof(int))     != cudaSuccess) { rc = -6; goto cleanup; }
    CK_MALLOC(x, H); CK_MALLOC(xn, H); CK_MALLOC(qg, 2*NQ*HD); CK_MALLOC(qb, NQ*HD);
    CK_MALLOC(gate, NQ*HD); CK_MALLOC(kb, NKV*HD); CK_MALLOC(vb, NKV*HD); CK_MALLOC(attn, NQ*HD);
    CK_MALLOC(mix, H); CK_MALLOC(qkv, CONVD); CK_MALLOC(conv_out, CONVD); CK_MALLOC(zc, INNER);
    CK_MALLOC(ac, TSR); CK_MALLOC(bc, TSR); CK_MALLOC(gg, TSR); CK_MALLOC(bb, TSR);
    CK_MALLOC(qh, NKH*SD); CK_MALLOC(kh, NKH*SD); CK_MALLOC(vh, NVH*SD); CK_MALLOC(core, NVH*SD);
    CK_MALLOC(gnorm, INNER); CK_MALLOC(ffn_g, I); CK_MALLOC(ffn_u, I); CK_MALLOC(ffn_a, I);
    for (int l = 0; l < L; l++) {
        if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
            CK_MALLOC(Kc[l], (size_t)nsteps*NKV*HD);
            CK_MALLOC(Vc[l], (size_t)nsteps*NKV*HD);
        } else {
            CK_MALLOC(Sst[l],  (size_t)NVH*SD*SD);
            CK_MALLOC(ring[l], (size_t)CONVD*(M2_CK-1));
            cudaMemsetAsync(Sst[l],  0, (size_t)NVH*SD*SD*sizeof(float), st);
            cudaMemsetAsync(ring[l], 0, (size_t)CONVD*(M2_CK-1)*sizeof(float), st);
        }
    }

    for (int p = 0; p < nsteps; p++) {
        int32_t token = (p < n_prompt) ? in_ids[p] : out_ids[p - n_prompt];
        cudaMemcpyAsync(d_tok, &token, sizeof(int32_t), cudaMemcpyHostToDevice, st);
        if (eng->format == FORMAT_FP8_BLOCK && eng->d_embed_f32)   // exact f32 embed (oracle parity)
            cudaMemcpyAsync(x, eng->d_embed_f32 + (size_t)token*H, (size_t)H*sizeof(float),
                            cudaMemcpyDeviceToDevice, st);
        else
            embed_w(eng, x, eng->d_token_embd, d_tok, 1, H, st);

        for (int l = 0; l < L; l++) {
            // pre-mixer RMSNorm
            rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(xn, x, Wf(eng->tensors.layers[l].attn_norm), H, 1, eps);
            const auto &T = eng->tensors.layers[l];
            if (c->attn_kind[l] == GEMMA4_ATTN_FULL) {
                gemv_w(eng, qg, T.ref_q, xn, st);
                gemv_w(eng, kb, T.ref_k, xn, st);
                gemv_w(eng, vb, T.ref_v, xn, st);
                m2_split_query_gate_kernel<<<(NQ*HD+255)/256,256,0,st>>>(qb, gate, qg, 1, NQ);
                per_head_rms_norm_rows_kernel<<<dim3(NQ,1),256,32*sizeof(float),st>>>(qb, Wf(T.attn_q_norm), NQ, HD, 1, eps);
                per_head_rms_norm_rows_kernel<<<dim3(NKV,1),256,32*sizeof(float),st>>>(kb, Wf(T.attn_k_norm), NKV, HD, 1, eps);
                qwen35_rope_pos_kernel<<<dim3((ROT/2+31)/32,NQ),32,0,st>>>(qb, NQ, p);
                qwen35_rope_pos_kernel<<<dim3((ROT/2+31)/32,NKV),32,0,st>>>(kb, NKV, p);
                cudaMemcpyAsync(Kc[l]+(size_t)p*NKV*HD, kb, (size_t)NKV*HD*sizeof(float), cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(Vc[l]+(size_t)p*NKV*HD, vb, (size_t)NKV*HD*sizeof(float), cudaMemcpyDeviceToDevice, st);
                qwen35_attn_step_kernel<<<NQ,256,(size_t)(p+1)*sizeof(float),st>>>(attn, qb, Kc[l], Vc[l], p, NKV);
                m2_sigmoid_gate_mul_kernel<<<(NQ*HD+255)/256,256,0,st>>>(attn, gate, NQ*HD);
                gemv_w(eng, mix, T.ref_o, attn, st);
            } else {
                // GDN (LINEAR): in_qkv requantized Q5_K→Q8_0 at load → native dp4a gemv_w (P5), like
                // every other projection (no per-step fp32 dequant). fmt_in_qkv is Q8_0 post-requant.
                gemv_w(eng, qkv, T.ssm.ref_in_qkv, xn, st);
                gemv_w(eng, zc, T.ssm.ref_in_z, xn, st);
                if (eng->format == FORMAT_FP8_BLOCK) {   // in_a/in_b are f32 (oracle parity)
                    m2_gemm(ac, xn, Wf(T.ssm.in_a), 1, H, TSR);   // alpha
                    m2_gemm(bc, xn, Wf(T.ssm.in_b), 1, H, TSR);   // beta
                } else {
                    gemv_w(eng, ac, T.ssm.ref_in_a, xn, st);   // alpha
                    gemv_w(eng, bc, T.ssm.ref_in_b, xn, st);   // beta
                }
                qwen35_conv_step_kernel<<<(CONVD+127)/128,128,0,st>>>(conv_out, qkv, ring[l], Wf(T.ssm.conv1d), CONVD);
                cudaMemcpyAsync(qh, conv_out,             (size_t)KEYD*sizeof(float),  cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(kh, conv_out+KEYD,        (size_t)KEYD*sizeof(float),  cudaMemcpyDeviceToDevice, st);
                cudaMemcpyAsync(vh, conv_out+2*KEYD,      (size_t)INNER*sizeof(float), cudaMemcpyDeviceToDevice, st);
                m2_l2norm_heads_kernel<<<dim3(NKH,1),128,0,st>>>(qh, NKH, SD, 1);
                m2_l2norm_heads_kernel<<<dim3(NKH,1),128,0,st>>>(kh, NKH, SD, 1);
                m2_decay_beta_kernel<<<(TSR+255)/256,256,0,st>>>(gg, bb, ac, bc, Wf(T.ssm.a_log), Wf(T.ssm.dt_bias), 1, TSR);
                // Head expansion differs by checkpoint layout: the GGUF (llama.cpp) permutes the GDN
                // q/k heads → TILE (vh%NKH); the FP8/safetensors keep HF order → repeat_INTERLEAVE
                // (vh/(NVH/NKH)), matching the torch-verified oracle. Route by format.
                if (eng->format == FORMAT_FP8_BLOCK)
                    qwen35_fp8_gdn_step_kernel<<<NVH,128,smGDN,st>>>(core, qh, kh, vh, gg, bb, Sst[l]);
                else
                    qwen35_gdn_step_kernel<<<NVH,128,smGDN,st>>>(core, qh, kh, vh, gg, bb, Sst[l]);
                m2_gated_norm_kernel<<<dim3(NVH,1),128,0,st>>>(gnorm, core, zc, Wf(T.ssm.norm), 1, NVH, INNER);
                gemv_w(eng, mix, T.ssm.ref_out, gnorm, st);
            }
            qwen35_add_kernel<<<(H+255)/256,256,0,st>>>(x, mix, H);
            // pre-FFN RMSNorm (post_attention_norm loaded into the ffn_norm slot) + FFN
            rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(xn, x, Wf(T.ffn_norm), H, 1, eps);
            if (c->n_experts > 0) {   // Qwen3.5-MoE sparse block (experts + shared) → mix
                moe_ffn(eng, l, xn, mix, 1, st);
            } else {
                gemv_w(eng, ffn_g, T.ref_gate, xn, st);
                gemv_w(eng, ffn_u, T.ref_up, xn, st);
                silu_glu_kernel<<<(I+255)/256,256,0,st>>>(ffn_a, ffn_g, ffn_u, I);
                gemv_w(eng, mix, T.ref_down, ffn_a, st);
            }
            qwen35_add_kernel<<<(H+255)/256,256,0,st>>>(x, mix, H);
        }

        if (p >= n_prompt - 1) {   // final norm + lm_head + argmax (skip on interior prompt tokens)
            rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(xn, x, Wf(eng->tensors.output_norm), H, 1, eps);
            if (eng->format == FORMAT_FP8_BLOCK)   // BF16 untied head (Q8_0 flips the 248320-vocab argmax)
                bf16_head_gemv_launch(eng->d_logits, eng->d_lmhead_bf16, xn, H, VOC, st);
            else
                gemv_w(eng, eng->d_logits, Wq(eng->tensors.output_weight), xn, H, VOC, st, eng->tensors.output_fmt);
            qwen35_argmax_kernel<<<1,256,0,st>>>(eng->d_logits, VOC, d_arg);
            int argmax = 0;
            cudaMemcpyAsync(&argmax, d_arg, sizeof(int), cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
            out_ids[p - (n_prompt - 1)] = argmax;
        }
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { fprintf(stderr, "qwen35_forward_greedy: CUDA error at pos %d: %s\n",
                                        p, cudaGetErrorString(e)); rc = -7; goto cleanup; }
    }

cleanup:
    #undef CK_MALLOC
    cudaStreamSynchronize(st);
    if (d_tok) cudaFree(d_tok); if (d_arg) cudaFree(d_arg);
    float *bufs[] = {x,xn,qg,qb,gate,kb,vb,attn,mix,qkv,conv_out,zc,ac,bc,gg,bb,qh,kh,vh,core,gnorm,ffn_g,ffn_u,ffn_a};
    for (float *b : bufs) if (b) cudaFree(b);
    if (Kc)  { for (int l=0;l<L;l++) if (Kc[l])  cudaFree(Kc[l]);  free(Kc); }
    if (Vc)  { for (int l=0;l<L;l++) if (Vc[l])  cudaFree(Vc[l]);  free(Vc); }
    if (Sst) { for (int l=0;l<L;l++) if (Sst[l]) cudaFree(Sst[l]); free(Sst); }
    if (ring){ for (int l=0;l<L;l++) if (ring[l])cudaFree(ring[l]);free(ring); }
    return rc;
}

// =========================================================================
// ─── M4: Qwen3.5 paged-batched + CUDA-graph hybrid decode ───────────────
// =========================================================================
// The B-row continuous-batching decode for qwen35. Each row is an INDEPENDENT sequence
// at its own absolute position carrying its OWN hybrid state: the 8 FULL layers an fp32
// K/V cache, the 24 LINEAR layers an fp32 GDN recurrent state S + conv ring. All state
// lives in per-SLOT arenas (engine d_q35_*), addressed by the per-row→slot map
// d_q35_rowslot — so a row's math touches only its own slot and the batch is exactly
// B independent single-row decodes (the batch self-test invariant). The same arithmetic
// the M3 single-seq forward (qwen35_forward_greedy) reached 8/8 parity with, lifted to
// B rows: the FULL-attn / GDN / conv kernels are the M3 stateful kernels with an added
// (row,slot) index; the projections (in_qkv now Q8_0 like the rest) use the batched dp4a
// gemv (gemv_batched_w) — all row-independent. Per-step varying inputs
// (tokens d_sb[0], positions d_ms_pos, row→slot d_q35_rowslot) are DEVICE-resident so the
// whole body is CUDA-graph-capturable: captured once per B (q35_graph), replayed per step.

// per-row batched compute-scratch slots (d_q35_sb[]); sizes set in ensure_q35_scratch.
enum {
    Q35_X=0, Q35_XN, Q35_QG, Q35_QB, Q35_GATE, Q35_KB, Q35_VB, Q35_ATTN, Q35_MIX, Q35_QKV,
    Q35_CONV, Q35_ZC, Q35_AC, Q35_BC, Q35_GG, Q35_BB, Q35_QH, Q35_KH, Q35_VH, Q35_CORE,
    Q35_GNORM, Q35_FG, Q35_FU, Q35_FA
};

// Partial NEOX RoPE on the first ROT dims of each head for B rows, each at its OWN absolute
// position pos[row]. x is [B][n_heads][HEAD] row-major. Bit-identical to M3 qwen35_rope_pos.
__global__ void qwen35_b_rope_kernel(float *x, int n_heads, const int *pos, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 0..ROT/2-1
    int h = blockIdx.y, r = blockIdx.z;
    if (i >= M2_ROT / 2 || h >= n_heads || r >= B) return;
    float *hd = x + ((size_t)r * n_heads + h) * M2_HEAD;
    float inv = powf(M2_THETA, -(float)(2 * i) / (float)M2_ROT);
    float ang = (float)pos[r] * inv;
    float c = cosf(ang), s = sinf(ang);
    float a = hd[i], b = hd[i + M2_ROT / 2];
    hd[i]              = a * c - b * s;
    hd[i + M2_ROT / 2] = b * c + a * s;
}

// Write each row's current K/V (kb/vb [B][NKV*HD]) into the per-slot FULL-layer K/V cache
// at (slot=rowslot[r], pos=pos[r]). Cache base = slot*maxctx*NKV*HD; token offset = pos*NKV*HD.
__global__ void qwen35_b_kv_write_kernel(__half *const *Kslots, __half *const *Vslots,
                                         const float *kb, const float *vb,
                                         const int *pos, const int *rowslot, int maxctx, int B,
                                         int nkv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // over nkv*HD
    int r   = blockIdx.y;
    if (r >= B || idx >= nkv * M2_HEAD) return;
    int slot = rowslot[r], p = pos[r];
    size_t base = (size_t)p * nkv * M2_HEAD;
    size_t src  = (size_t)r * nkv * M2_HEAD + idx;
    Kslots[slot][base + idx] = __float2half(kb[src]);
    Vslots[slot][base + idx] = __float2half(vb[src]);
}

// Single-query causal GQA softmax for B rows against the per-slot FULL-layer K/V cache.
// One block per (query-head, row); reads pos[r]/slot=rowslot[r]; scores in dynamic shared
// (maxctx floats; only [0..pos] touched). Arithmetic bit-identical to M3 qwen35_attn_step.
__global__ void qwen35_b_attn_kernel(float *out, const float *q,
                                     __half *const *Kslots, __half *const *Vslots,
                                     const int *pos, const int *rowslot, int maxctx, int B,
                                     int nkv, int nq) {
    int hd = blockIdx.x;   // query head 0..NQ-1
    int r  = blockIdx.y;   // row
    if (hd >= nq || r >= B) return;
    int slot = rowslot[r], p = pos[r];
    int kv  = hd / (nq / nkv);
    int tid = threadIdx.x;
    extern __shared__ float sc[];   // [maxctx]
    const float *qr = q + ((size_t)r * nq + hd) * M2_HEAD;
    const __half *Kb = Kslots[slot];   // fp16 cache: convert on read
    const __half *Vb = Vslots[slot];
    float scale = rsqrtf((float)M2_HEAD);
    for (int j = tid; j <= p; j += blockDim.x) {
        const __half *kr = Kb + ((size_t)j * nkv + kv) * M2_HEAD;
        float acc = 0.f;
        for (int d = 0; d < M2_HEAD; d++) acc += qr[d] * __half2float(kr[d]);
        sc[j] = acc * scale;
    }
    __syncthreads();
    __shared__ float red[32];
    float m = -1e30f;
    for (int j = tid; j <= p; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = block_reduce_max(m, red);
    __shared__ float msh; if (tid == 0) msh = m; __syncthreads(); m = msh;
    float ssum = 0.f;
    for (int j = tid; j <= p; j += blockDim.x) { float e = __expf(sc[j] - m); sc[j] = e; ssum += e; }
    ssum = block_reduce_sum(ssum, red);
    __shared__ float ssh; if (tid == 0) ssh = ssum; __syncthreads();
    float inv = 1.f / ssh;
    for (int d = tid; d < M2_HEAD; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j <= p; j++)
            acc += sc[j] * __half2float(Vb[((size_t)j * nkv + kv) * M2_HEAD + d]);
        out[((size_t)r * nq + hd) * M2_HEAD + d] = acc * inv;
    }
}

// ── Flash-decoding (split-KV) FULL-layer attention: raise decode occupancy ────────────────────────
// The kernel above is ONE block per (head,row) → only NQ=16 blocks at B=1, on a 48-SM GB10, each
// looping ALL positions serially: the long-context single-stream decode bottleneck. Flash-decoding
// adds a SPLIT dimension so B=1 launches NQ*S blocks (fills the SMs) and each block's serial loop is
// ~S× shorter. Each PARTIAL block runs the SAME score-parallel body over its position slice and
// writes an UNNORMALIZED online-softmax partial (m_i, l_i, o_i[HD]); the COMBINE kernel flash-merges
// the S partials per (head,row). Graph-safe: S is a setup constant (q35_attn_splits from q35_maxctx),
// never runtime p (p only bounds the intra-block loop / idle-exits empty splits). NOT bit-identical
// to the fp32 oracle (reduction reordered) — argmax-identical is the bar (gated by part-D).
//
// This is DECODE-ONLY: the prefill-continuation path (base>0 / T>tile) keeps qwen35_b_attn_kernel,
// so the maxctx-score-in-shared context cap stays until that path is converted too (a follow-up).
__global__ void qwen35_flash_partial_kernel(
    const float *q, __half *const *Kslots, __half *const *Vslots,
    const int *pos, const int *rowslot, int maxctx, int B, int nkv, int S,
    float *part_m, float *part_l, float *part_o, int nq) {   // [B*NQ*S], [B*NQ*S], [B*NQ*S*HD]
    int hd = blockIdx.x, r = blockIdx.y, s = blockIdx.z;
    if (hd >= nq || r >= B) return;
    int slot = rowslot[r], p = pos[r];
    int tile = (maxctx + S - 1) / S;
    int lo = s * tile, hi = lo + tile - 1; if (hi > p) hi = p;
    size_t pidx = (((size_t)r * nq + hd) * S + s);
    int tid = threadIdx.x;
    if (lo > p) {                          // empty split → sentinel (combine drops it, NaN-safe)
        if (tid == 0) { part_m[pidx] = -1e30f; part_l[pidx] = 0.f; }
        for (int d = tid; d < M2_HEAD; d += blockDim.x) part_o[pidx * M2_HEAD + d] = 0.f;
        return;
    }
    int kv = hd / (nq / nkv);           // runtime nkv (MoE 2 → hd/8, 9B 4 → hd/4)
    extern __shared__ float sc[];          // [tile] — bounded, ctx-independent (~2 KB)
    const float *qr = q + ((size_t)r * nq + hd) * M2_HEAD;
    const __half *Kb = Kslots[slot];
    const __half *Vb = Vslots[slot];
    float scale = rsqrtf((float)M2_HEAD);
    int n = hi - lo + 1;
    for (int jj = tid; jj < n; jj += blockDim.x) {
        const __half *kr = Kb + ((size_t)(lo + jj) * nkv + kv) * M2_HEAD;
        float acc = 0.f;
        for (int d = 0; d < M2_HEAD; d++) acc += qr[d] * __half2float(kr[d]);
        sc[jj] = acc * scale;
    }
    __syncthreads();
    __shared__ float red[32];
    float m = -1e30f;
    for (int jj = tid; jj < n; jj += blockDim.x) m = fmaxf(m, sc[jj]);
    m = block_reduce_max(m, red);
    __shared__ float msh; if (tid == 0) msh = m; __syncthreads(); m = msh;
    float ssum = 0.f;
    for (int jj = tid; jj < n; jj += blockDim.x) { float e = __expf(sc[jj] - m); sc[jj] = e; ssum += e; }
    ssum = block_reduce_sum(ssum, red);
    __shared__ float ssh; if (tid == 0) ssh = ssum; __syncthreads();
    if (tid == 0) { part_m[pidx] = m; part_l[pidx] = ssh; }
    for (int d = tid; d < M2_HEAD; d += blockDim.x) {
        float acc = 0.f;
        for (int jj = 0; jj < n; jj++)
            acc += sc[jj] * __half2float(Vb[((size_t)(lo + jj) * nkv + kv) * M2_HEAD + d]);
        part_o[pidx * M2_HEAD + d] = acc;   // UNNORMALIZED (combine divides by the merged sum)
    }
}

// Flash-decoding COMBINE: merge the S per-(head,row) partials with the online-softmax rescale
// (m = max_s m_s; out = Σ_s e^{m_s-m}·o_s / Σ_s e^{m_s-m}·l_s). Grid dim3(NQ,B), thread d owns out
// dim d. Empty splits (m_s == -1e30) are skipped — never multiply 0·Inf.
__global__ void qwen35_flash_combine_kernel(
    float *out, const float *part_m, const float *part_l, const float *part_o, int B, int S, int nq) {
    int hd = blockIdx.x, r = blockIdx.y;
    if (hd >= nq || r >= B) return;
    int tid = threadIdx.x;
    size_t base = ((size_t)r * nq + hd) * S;
    float m = -1e30f;
    for (int s = 0; s < S; s++) { float ms = part_m[base + s]; if (ms > m) m = ms; }
    float l = 0.f;
    for (int s = 0; s < S; s++) {
        float ms = part_m[base + s];
        if (ms <= -1e30f) continue;
        l += __expf(ms - m) * part_l[base + s];
    }
    float inv = 1.f / fmaxf(l, 1e-20f);
    for (int d = tid; d < M2_HEAD; d += blockDim.x) {
        float od = 0.f;
        for (int s = 0; s < S; s++) {
            float ms = part_m[base + s];
            if (ms <= -1e30f) continue;
            od += __expf(ms - m) * part_o[(base + s) * M2_HEAD + d];
        }
        out[((size_t)r * nq + hd) * M2_HEAD + d] = od * inv;
    }
}

// Stateful causal depthwise conv1d (k=CK) + SiLU for B rows over conv_dim channels; each row's
// CK-1 history lives in its per-slot ring [conv_dim][CK-1]. Bit-identical to M3 qwen35_conv_step.
__global__ void qwen35_b_conv_kernel(float *conv_out, const float *qkv, float *const *ring_slots,
                                     const float *cw, const int *rowslot, int conv_dim, int B) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (c >= conv_dim || r >= B) return;
    const int K = M2_CK, HK = M2_CK - 1;
    int slot = rowslot[r];
    float *rg = ring_slots[slot] + (size_t)c * HK;               // [HK] oldest..newest
    const float *xr = qkv + (size_t)r * conv_dim;
    float acc = 0.f;
    #pragma unroll
    for (int j = 0; j < HK; j++) acc += cw[c * K + j] * rg[j];
    acc += cw[c * K + HK] * xr[c];
    conv_out[(size_t)r * conv_dim + c] = acc / (1.f + __expf(-acc));   // SiLU
    #pragma unroll
    for (int j = 0; j < HK - 1; j++) rg[j] = rg[j + 1];
    rg[HK - 1] = xr[c];
}

// Split each row's conv output [B][CONVD] into contiguous qh[B][KEYD], kh[B][KEYD], vh[B][VALD].
__global__ void qwen35_b_split_qkv_kernel(float *qh, float *kh, float *vh, const float *conv, int B, int convd, int vald) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // over CONVD
    int r   = blockIdx.y;
    if (r >= B || idx >= convd) return;
    const float *cr = conv + (size_t)r * convd;
    if      (idx < M2_KEYD)     qh[(size_t)r * M2_KEYD + idx]               = cr[idx];
    else if (idx < 2 * M2_KEYD) kh[(size_t)r * M2_KEYD + (idx - M2_KEYD)]  = cr[idx];
    else                        vh[(size_t)r * vald + (idx - 2*M2_KEYD)] = cr[idx];
}

// GDN single-step delta-rule recurrence for B rows; each (v-head,row) block loads/stores the
// carried fp32 state S[SD×SD] from the per-slot arena. Bit-identical to M3 qwen35_gdn_step.
__global__ void qwen35_b_gdn_kernel(float *core, const float *q, const float *k, const float *v,
                                    const float *g, const float *beta, __nv_bfloat16 *const *S_slots,
                                    const int *rowslot, int B, int interleave, int nvh) {
    int vh = blockIdx.x;
    int r  = blockIdx.y;
    if (vh >= nvh || r >= B) return;
    int slot = rowslot[r];
    // GGUF (llama.cpp) permutes GDN q/k heads → TILE (vh%NKH); FP8/safetensors keep HF order →
    // repeat-INTERLEAVE (vh/(NVH/NKH)). See [[qwen35-fp8-engine-serving]].
    int kh  = interleave ? (vh / (nvh / M2_NKH)) : (vh % M2_NKH);
    int tid = threadIdx.x;
    extern __shared__ float sm[];
    float *S   = sm;                 // [SD*SD] k-major
    float *kt  = S + M2_SD * M2_SD;  // [SD]
    float *qt  = kt + M2_SD;         // [SD]
    float *dlt = qt + M2_SD;         // [SD]
    __nv_bfloat16 *Sg = S_slots[slot] + (size_t)vh * M2_SD * M2_SD;
    // Vectorized state load: 8 bf16 per 16-byte uint4 access (Sg is (SD*SD*2B)-aligned per head;
    // elementwise bf16→f32 convert, values identical to the scalar loop).
    static_assert((M2_SD * M2_SD) % 8 == 0, "GDN state must be uint4-tileable");
    const uint4 *Sg_ld = reinterpret_cast<const uint4 *>(Sg);
    for (int v4 = tid; v4 < (M2_SD * M2_SD) / 8; v4 += blockDim.x) {
        uint4 pkt = Sg_ld[v4];
        const __nv_bfloat16 *pe = reinterpret_cast<const __nv_bfloat16 *>(&pkt);
        float *dst = S + (size_t)v4 * 8;
        #pragma unroll
        for (int j = 0; j < 8; j++) dst[j] = __bfloat162float(pe[j]);
    }
    __syncthreads();
    float scale = rsqrtf((float)M2_SD);
    const float *qr = q + (size_t)r * M2_NKH * M2_SD;
    const float *kr = k + (size_t)r * M2_NKH * M2_SD;
    const float *vr = v + (size_t)r * nvh * M2_SD;
    if (tid < M2_SD) {
        kt[tid] = kr[(size_t)kh * M2_SD + tid];
        qt[tid] = qr[(size_t)kh * M2_SD + tid] * scale;
    }
    __syncthreads();
    float gt = __expf(g[(size_t)r * nvh + vh]);
    float bt = beta[(size_t)r * nvh + vh];
    for (int idx = tid; idx < M2_SD * M2_SD; idx += blockDim.x) S[idx] *= gt;
    __syncthreads();
    if (tid < M2_SD) {
        float acc = 0.f;
        for (int kd = 0; kd < M2_SD; kd++) acc += S[kd * M2_SD + tid] * kt[kd];
        float vv = vr[(size_t)vh * M2_SD + tid];
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
        core[((size_t)r * nvh + vh) * M2_SD + tid] = acc;
    }
    __syncthreads();
    // Vectorized state store: pack 8 f32→bf16 into one 16-byte uint4 write (same values as scalar).
    uint4 *Sg_st = reinterpret_cast<uint4 *>(Sg);
    for (int v4 = tid; v4 < (M2_SD * M2_SD) / 8; v4 += blockDim.x) {
        const float *src = S + (size_t)v4 * 8;
        uint4 pkt;
        __nv_bfloat16 *pe = reinterpret_cast<__nv_bfloat16 *>(&pkt);
        #pragma unroll
        for (int j = 0; j < 8; j++) pe[j] = __float2bfloat16(src[j]);
        Sg_st[v4] = pkt;
    }
}

// ── PREFILL chunked GDN/conv (P-perf): replace the per-token recurrence launches ────────────
// These process a whole T-row prefill TILE (all rows the same slot) in ONE launch each, carrying
// the genuine recurrence through the per-slot arenas (S / conv ring) exactly as the per-token
// kernels do — so the produced state and `core` are bit-identical to the token-by-token path.

// Batched stateful causal depthwise conv1d (k=CK) + SiLU over a T-row tile. Reads the per-slot
// ring for the CK-1 positions BEFORE the tile (carry from earlier chunks) and the tile itself for
// in-tile history. Does NOT mutate the ring (see qwen35_b_ring_update_kernel, run after).
__global__ void qwen35_b_conv_chunk_kernel(float *conv_out, const float *qkv,
                                           float *const *ring_slots,
                                           const float *cw, int slot, int conv_dim, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (c >= conv_dim || r >= T) return;
    const int K = M2_CK, HK = M2_CK - 1;
    const float *rg = ring_slots[slot] + (size_t)c * HK;           // [HK] positions base-HK..base-1
    float acc = 0.f;
    #pragma unroll
    for (int j = 0; j < K; j++) {
        int sr = r - (K - 1) + j;                                  // tile-relative source row
        float val = (sr >= 0) ? qkv[(size_t)sr * conv_dim + c] : rg[HK + sr];
        acc += cw[c * K + j] * val;
    }
    conv_out[(size_t)r * conv_dim + c] = acc / (1.f + __expf(-acc));   // SiLU
}

// Advance the per-slot conv ring to hold the last CK-1 raw-qkv inputs of the tile (= positions
// base+T-HK..base+T-1), handling T<HK by folding in the old ring. Run AFTER the conv reads.
__global__ void qwen35_b_ring_update_kernel(float *const *ring_slots,
                                            const float *qkv, int slot, int conv_dim, int T) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const int HK = M2_CK - 1;
    float *rg = ring_slots[slot] + (size_t)c * HK;
    float old[M2_CK - 1];
    #pragma unroll
    for (int j = 0; j < HK; j++) old[j] = rg[j];
    float nv[M2_CK - 1];
    #pragma unroll
    for (int idx = 0; idx < HK; idx++) {
        int rel = T - HK + idx;                                    // tile-relative position
        nv[idx] = (rel >= 0) ? qkv[(size_t)rel * conv_dim + c] : old[HK + rel];
    }
    #pragma unroll
    for (int idx = 0; idx < HK; idx++) rg[idx] = nv[idx];
}

// Block-cooperative tf32 tensor-core GEMM: C[M×N] = A[M×K]·B[K×N] (+= if `add`), all row-major
// fp32 (shared or global). tf32 (10-bit mantissa) preserves the delta-rule precision far better
// than bf16. M,N multiples of 16; K multiple of 8. Warps split the 16×16 output tiles round-robin.
__device__ inline void wmma_gemm_tf32(float *C, const float *A, const float *B,
                                      int M, int N, int K, int ldc, int lda, int ldb,
                                      int warp, int n_warps, bool add) {
    using namespace nvcuda;
    int n_nt = N / 16, n_tiles = (M / 16) * n_nt;
    for (int tile = warp; tile < n_tiles; tile += n_warps) {
        int mt = (tile / n_nt) * 16, nt = (tile % n_nt) * 16;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> cf;
        wmma::fill_fragment(cf, 0.0f);
        for (int k = 0; k < K; k += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> bf;
            wmma::load_matrix_sync(af, A + (size_t)mt * lda + k, lda);
            wmma::load_matrix_sync(bf, B + (size_t)k * ldb + nt, ldb);
            #pragma unroll
            for (int t = 0; t < af.num_elements; t++) af.x[t] = wmma::__float_to_tf32(af.x[t]);
            #pragma unroll
            for (int t = 0; t < bf.num_elements; t++) bf.x[t] = wmma::__float_to_tf32(bf.x[t]);
            wmma::mma_sync(cf, af, bf, cf);
        }
        float *cp = C + (size_t)mt * ldc + nt;
        if (add) {
            wmma::fragment<wmma::accumulator, 16, 16, 8, float> of;
            wmma::load_matrix_sync(of, cp, ldc, wmma::mem_row_major);
            #pragma unroll
            for (int t = 0; t < cf.num_elements; t++) cf.x[t] += of.x[t];
        }
        wmma::store_matrix_sync(cp, cf, ldc, wmma::mem_row_major);
    }
}

// Per-v-head GDN scratch: WY/UT staging (M2_SCR_PER) + tf32-GEMM temps (qgs, kchsT, ctmp, Stmp).
#define Q35_GDN_SCR (M2_SCR_PER + 3*M2_CHUNK*M2_SD + M2_SD*M2_SD)

// GDN chunked parallel-scan (WY/UT form) over a T-row prefill TILE, carrying the per-slot fp32
// state S in/out of the arena S_arena[slot]. Mathematically identical to qwen35_b_gdn_kernel run
// token-by-token (and to the M2 chunk==recur self-test). One block per v-head, CHUNK=64.
// q/k indexed [row,NKH,SD]; v/g/beta indexed [row,NVH]; reads inputs only at rows < T (guarded),
// so the T-row scratch needs no NPAD zero-padding. NPAD = roundup(T,CHUNK) drives the chunk count.
__global__ void qwen35_b_gdn_chunk_kernel(float *core, const float *q, const float *k,
                                          const float *v, const float *g, const float *beta,
                                          __nv_bfloat16 *const *S_slots, float *scratch,
                                          int slot, int N, int NPAD, int interleave, int nvh) {
    int vh  = blockIdx.x;
    // TILE (GGUF) vs repeat-INTERLEAVE (FP8/safetensors). See [[qwen35-fp8-engine-serving]].
    int kh  = interleave ? (vh / (nvh / M2_NKH)) : (vh % M2_NKH);
    int tid = threadIdx.x;
    const int CS = M2_CHUNK, SD = M2_SD;
    extern __shared__ float sm[];
    float *S    = sm;                 // [SD*SD] carried state
    float *kch  = S + SD * SD;        // [CS*SD] cached k-chunk
    float *gc   = kch + CS * SD;      // [CS] cumulative decay
    float *bet  = gc + CS;            // [CS] beta per chunk row
    float *rowb = bet + CS;           // [CS] fwd-subst scratch row
    float *Tm    = scratch + (size_t)vh * Q35_GDN_SCR;
    float *u     = Tm + CS * CS;
    float *kcd   = u + CS * SD;
    float *vnew  = kcd + CS * SD;
    float *aintra= vnew + CS * SD;
    float *qgs   = aintra + CS * CS;     // [CS*SD] q·scale·exp(gc) for the inter GEMM
    float *kchsT = qgs + CS * SD;        // [SD*CS] (k·decay)^T for the state-update GEMM
    float *ctmp  = kchsT + SD * CS;      // [CS*SD] core tile (before N-guarded copy-out)
    float *Stmp  = ctmp + CS * SD;       // [SD*SD] state-update GEMM result
    __nv_bfloat16 *Sg = S_slots[slot] + (size_t)vh * SD * SD;
    float scale = rsqrtf((float)SD);
    int warp = tid >> 5, n_warps = blockDim.x >> 5;
    for (int idx = tid; idx < SD * SD; idx += blockDim.x) S[idx] = __bfloat162float(Sg[idx]);   // carry-in
    __syncthreads();

    int n_chunks = NPAD / CS;
    for (int c = 0; c < n_chunks; c++) {
        for (int idx = tid; idx < CS * SD; idx += blockDim.x) {
            int r = idx / SD, d = idx % SD;
            int gt = c * CS + r;
            kch[idx] = (gt < N) ? k[((size_t)gt * M2_NKH + kh) * SD + d] : 0.f;
        }
        for (int r = tid; r < CS; r += blockDim.x) {
            int gt = c * CS + r;
            bet[r] = (gt < N) ? beta[(size_t)gt * nvh + vh] : 0.f;
        }
        __syncthreads();
        if (tid == 0) {
            float acc = 0.f;
            for (int r = 0; r < CS; r++) {
                int gt = c * CS + r;
                acc += (gt < N) ? g[(size_t)gt * nvh + vh] : 0.f;
                gc[r] = acc;
            }
        }
        __syncthreads();
        for (int idx = tid; idx < CS * CS; idx += blockDim.x) {
            int i = idx / CS, j = idx % CS;
            float val = 0.f;
            if (i > j) {
                float dot = 0.f;
                for (int d = 0; d < SD; d++) dot += kch[i * SD + d] * bet[i] * kch[j * SD + d];
                val = -dot * __expf(gc[i] - gc[j]);
            }
            Tm[idx] = val;
        }
        __syncthreads();
        for (int i = 1; i < CS; i++) {
            for (int m = tid; m < i; m += blockDim.x) rowb[m] = Tm[i * CS + m];
            __syncthreads();
            for (int m = tid; m < i; m += blockDim.x) {
                float acc = rowb[m];
                for (int nn = 0; nn < i; nn++) acc += rowb[nn] * Tm[nn * CS + m];
                Tm[i * CS + m] = acc;
            }
            __syncthreads();
        }
        for (int i = tid; i < CS; i += blockDim.x) Tm[i * CS + i] += 1.f;
        __syncthreads();
        for (int idx = tid; idx < CS * SD; idx += blockDim.x) {
            int i = idx / SD, d = idx % SD;
            float su = 0.f, sk = 0.f;
            for (int s = 0; s < CS; s++) {
                float t = Tm[i * CS + s];
                int gt = c * CS + s;
                float vv = (gt < N) ? v[((size_t)gt * nvh + vh) * SD + d] : 0.f;
                su += t * (vv * bet[s]);
                sk += t * (kch[s * SD + d] * bet[s] * __expf(gc[s]));
            }
            u[idx] = su; kcd[idx] = sk;
        }
        __syncthreads();
        // vnew = u - kcd@S  (tensor-core: vnew←kcd@S, then subtract u)
        wmma_gemm_tf32(vnew, kcd, S, CS, SD, SD, SD, SD, SD, warp, n_warps, false);
        __syncthreads();
        for (int idx = tid; idx < CS * SD; idx += blockDim.x) vnew[idx] = u[idx] - vnew[idx];
        // aintra[i,j] = <q[i],k[j]>·exp(gc[i]-gc[j]) (i>=j); qgs[i,d] = q·scale·exp(gc[i])
        for (int idx = tid; idx < CS * CS; idx += blockDim.x) {
            int i = idx / CS, j = idx % CS;
            float val = 0.f;
            if (i >= j) {
                int gti = c * CS + i;
                float dot = 0.f;
                for (int d = 0; d < SD; d++) {
                    float qv = (gti < N) ? q[((size_t)gti * M2_NKH + kh) * SD + d] * scale : 0.f;
                    dot += qv * kch[j * SD + d];
                }
                val = dot * __expf(gc[i] - gc[j]);
            }
            aintra[idx] = val;
        }
        for (int idx = tid; idx < CS * SD; idx += blockDim.x) {
            int i = idx / SD, d = idx % SD, gti = c * CS + i;
            qgs[idx] = (gti < N) ? q[((size_t)gti * M2_NKH + kh) * SD + d] * scale * __expf(gc[i]) : 0.f;
        }
        __syncthreads();
        // core = qgs@S (inter) + aintra@vnew (intra), into ctmp, then N-guarded copy to core
        wmma_gemm_tf32(ctmp, qgs, S, CS, SD, SD, SD, SD, SD, warp, n_warps, false);
        __syncthreads();
        wmma_gemm_tf32(ctmp, aintra, vnew, CS, SD, CS, SD, CS, SD, warp, n_warps, true);
        __syncthreads();
        for (int idx = tid; idx < CS * SD; idx += blockDim.x) {
            int i = idx / SD, d = idx % SD, gti = c * CS + i;
            if (gti < N) core[((size_t)gti * nvh + vh) * SD + d] = ctmp[idx];
        }
        // S = S·exp(glast) + (k·decay)^T @ vnew  (tensor-core)
        float glast = gc[CS - 1];
        for (int idx = tid; idx < SD * CS; idx += blockDim.x) {
            int kd = idx / CS, r = idx % CS;                 // kchsT[kd][r] = kch[r][kd]·exp(glast-gc[r])
            kchsT[idx] = kch[r * SD + kd] * __expf(glast - gc[r]);
        }
        __syncthreads();
        wmma_gemm_tf32(Stmp, kchsT, vnew, SD, SD, CS, SD, CS, SD, warp, n_warps, false);
        __syncthreads();
        for (int idx = tid; idx < SD * SD; idx += blockDim.x) S[idx] = S[idx] * __expf(glast) + Stmp[idx];
        __syncthreads();
    }
    for (int idx = tid; idx < SD * SD; idx += blockDim.x) Sg[idx] = __float2bfloat16(S[idx]);   // carry-out
}

