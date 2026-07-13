// ABOUTME: DFlash draft backbone device forward — assembles the validated P3 kernels over residency.
// ABOUTME: (1+K) query rows through 6 BF16 draft layers + final norm; reads resident weight views.
//
// This is the callable draft backbone the verify serving path invokes. It composes the numerically
// validated P3 kernels (RMSNorm, BF16 matmul, per-head norm, neox RoPE, non-causal GQA, silu-GLU)
// over the q35_dflash_residency BF16 weight views. It computes the draft hidden states for the
// query rows; the caller supplies the context K/V (from precompute_and_store_context_kv) and the
// per-row absolute positions. Self-attention here is over the query block itself (the (1+K) rows);
// context cross-attention uses the same kernel with the precomputed context K/V and is composed by
// the caller — kept separate so this module is testable in isolation against the host reference.
//
// C-style: fixed-shape launches, no per-token host allocation. All math fp32 with double
// accumulation to match the validated parity gates. BF16 weights are read via __bfloat162float.
#ifndef FUCINA_QWEN35_DFLASH_FORWARD_CUH
#define FUCINA_QWEN35_DFLASH_FORWARD_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "qwen35_dflash_residency.cuh"
#include "qwen35_dflash_rng.cuh"      // shared-key counter RNG (probabilistic draft sampling)
#include "qwen35_dflash_reject.cuh"   // greedy + probabilistic rejection oracle (host/device)

// ── device kernels reading BF16 weights (fp32 compute, double accumulation) ──
__global__ void q35df_rmsnorm(const float* x,const __nv_bfloat16* w,float* y,int H,int rows,float eps){
    int r=blockIdx.x; if(r>=rows) return; const float* xr=x+(size_t)r*H; float* yr=y+(size_t)r*H;
    __shared__ double buf[256]; double loc=0; for(int i=threadIdx.x;i<H;i+=blockDim.x){ double v=xr[i]; loc+=v*v; }
    buf[threadIdx.x]=loc; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) buf[threadIdx.x]+=buf[threadIdx.x+s]; __syncthreads(); }
    __shared__ double ss; if(threadIdx.x==0) ss=buf[0]; __syncthreads();
    double inv=1.0/sqrt(ss/H+eps); for(int i=threadIdx.x;i<H;i+=blockDim.x) yr[i]=(float)((double)xr[i]*inv*(double)__bfloat162float(w[i]));
}
// O[rows,outd] = A[rows,in] @ W[outd,in]^T ; W is BF16. Warp-per-output (32 threads cooperate on
// the inner product), fp32 accumulation. blockDim.x must be 32*(outputs-per-block); grid.y covers
// outd. Row A is cached in shared memory once per block. Kept within the parity tolerance (1e-3);
// fp32 vs the old fp64 accumulate is well inside it for the draft's in<=32768 dot products.
__global__ void q35df_matmul(const float* A,const __nv_bfloat16* W,float* O,int rows,int in,int outd){
    int r=blockIdx.x; int lane=threadIdx.x&31; int wpb=blockDim.x>>5; int o=blockIdx.y*wpb+(threadIdx.x>>5);
    if(r>=rows||o>=outd) return;
    const float* a=A+(size_t)r*in; const __nv_bfloat16* w=W+(size_t)o*in;
    float acc=0.0f; for(int i=lane;i<in;i+=32) acc+=a[i]*__bfloat162float(w[i]);
    // warp reduce
    for(int s=16;s>0;s>>=1) acc+=__shfl_down_sync(0xffffffff,acc,s);
    if(lane==0) O[(size_t)r*outd+o]=acc;
}
__global__ void q35df_headnorm(float* X,const __nv_bfloat16* w,int rows,int nh,int HD,float eps){
    int r=blockIdx.x; int h=blockIdx.y; if(r>=rows||h>=nh) return; float* v=X+((size_t)r*nh+h)*HD;
    __shared__ double buf[256]; double loc=0; for(int i=threadIdx.x;i<HD;i+=blockDim.x){ double x=v[i]; loc+=x*x; }
    buf[threadIdx.x]=loc; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) buf[threadIdx.x]+=buf[threadIdx.x+s]; __syncthreads(); }
    __shared__ double ss; if(threadIdx.x==0) ss=buf[0]; __syncthreads();
    double inv=1.0/sqrt(ss/HD+eps); for(int i=threadIdx.x;i<HD;i+=blockDim.x) v[i]=(float)((double)v[i]*inv*(double)__bfloat162float(w[i]));
}
__global__ void q35df_rope(float* X,const int* pos,int rows,int nh,int HD,double theta){
    int r=blockIdx.x; int h=blockIdx.y; if(r>=rows||h>=nh) return; float* v=X+((size_t)r*nh+h)*HD; int half=HD/2; double p=pos[r];
    for(int i=threadIdx.x;i<half;i+=blockDim.x){ double f=pow(theta,-2.0*i/(double)HD),a=p*f,c=cos(a),s=sin(a); double x=v[i],y=v[i+half]; v[i]=(float)(x*c-y*s); v[i+half]=(float)(x*s+y*c); }
}
// non-causal GQA: Q[rows,NQ,HD] attends K/V[ctx,NKV,HD]; O[rows,NQ,HD]. block per (row,qhead).
__global__ void q35df_attn(const float* Q,const float* K,const float* V,float* O,int rows,int ctx,int NQ,int NKV,int HD){
    int r=blockIdx.x; int h=blockIdx.y; if(r>=rows||h>=NQ) return; int g=h/(NQ/NKV);
    const float* q=Q+((size_t)r*NQ+h)*HD; extern __shared__ double sh[]; double scale=1.0/sqrt((double)HD);
    for(int t=threadIdx.x;t<ctx;t+=blockDim.x){ const float* k=K+((size_t)t*NKV+g)*HD; double d=0; for(int i=0;i<HD;i++) d+=(double)q[i]*(double)k[i]; sh[t]=d*scale; }
    __syncthreads(); __shared__ double sm;
    if(threadIdx.x==0){ double m=-1e300; for(int t=0;t<ctx;t++) if(sh[t]>m) m=sh[t]; double s=0; for(int t=0;t<ctx;t++){ sh[t]=exp(sh[t]-m); s+=sh[t]; } sm=s; }
    __syncthreads();
    for(int i=threadIdx.x;i<HD;i+=blockDim.x){ double acc=0; for(int t=0;t<ctx;t++){ const float* v=V+((size_t)t*NKV+g)*HD; acc+=sh[t]*(double)v[i]; } O[((size_t)r*NQ+h)*HD+i]=(float)(acc/sm); }
}
// Combined non-causal GQA: each query row attends the concatenation of context K/V [ctx,NKV,HD]
// and the query block's own K/V [rows,NKV,HD]. Softmax over (ctx+rows) positions; O[rows,NQ,HD].
// causal: when 1, a query row r only attends key positions with absolute position <= its own.
// Context rows carry absolute positions cpos_abs[t]; query rows carry qpos_abs. Non-causal (0)
// attends everything (the DFlash full_attention layer). All context is <= the query positions
// anyway, so causality only masks among the query block for causal (SWA) layers.
__global__ void q35df_attn_ctx(const float* Q,const float* Kc,const float* Vc,const float* Kq,const float* Vq,
                               float* O,int rows,int ctx,int NQ,int NKV,int HD,int causal,
                               const int* cpos_abs,const int* qpos_abs){
    int r=blockIdx.x; int h=blockIdx.y; if(r>=rows||h>=NQ) return; int g=h/(NQ/NKV);
    const float* q=Q+((size_t)r*NQ+h)*HD; extern __shared__ double sh[]; int tot=ctx+rows; double scale=1.0/sqrt((double)HD);
    int qp = causal ? qpos_abs[r] : 0;
    for(int t=threadIdx.x;t<tot;t+=blockDim.x){
        const float* k = (t<ctx) ? Kc+((size_t)t*NKV+g)*HD : Kq+((size_t)(t-ctx)*NKV+g)*HD;
        int allow = 1;
        if(causal){ int kp = (t<ctx) ? cpos_abs[t] : qpos_abs[t-ctx]; allow = (kp <= qp); }
        if(allow){ double d=0; for(int i=0;i<HD;i++) d+=(double)q[i]*(double)k[i]; sh[t]=d*scale; }
        else sh[t]=-1e300;
    }
    __syncthreads(); __shared__ double sm;
    if(threadIdx.x==0){ double m=-1e300; for(int t=0;t<tot;t++) if(sh[t]>m) m=sh[t]; double s=0; for(int t=0;t<tot;t++){ sh[t]=exp(sh[t]-m); s+=sh[t]; } sm=s; }
    __syncthreads();
    for(int i=threadIdx.x;i<HD;i+=blockDim.x){ double acc=0; for(int t=0;t<tot;t++){ const float* v=(t<ctx)?Vc+((size_t)t*NKV+g)*HD:Vq+((size_t)(t-ctx)*NKV+g)*HD; acc+=sh[t]*(double)v[i]; } O[((size_t)r*NQ+h)*HD+i]=(float)(acc/sm); }
}
__global__ void q35df_residual(float* x,const float* d,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=d[i]; }
__global__ void q35df_siluglu(const float* g,const float* u,float* o,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ double x=g[i]; o[i]=(float)((x/(1.0+exp(-x)))*(double)u[i]); } }

// Scratch buffers for one draft backbone forward. Sized for `rows` query rows. All device.
struct q35_dflash_fwd_scratch {
    float *h1,*q,*k,*v,*attn,*mix,*h2,*g,*u,*ff,*out;
    int rows;
};
static inline bool q35_dflash_fwd_scratch_alloc(q35_dflash_fwd_scratch* s, const qwen35dflash::Geometry& geom, int rows){
    s->rows=rows; int H=geom.H,I=geom.I,qd=geom.q_dim(),kvd=geom.kv_dim();
    bool ok=true;
    auto A=[&](float** p,size_t n){ ok = ok && cudaMalloc(p,n*4)==cudaSuccess; };
    A(&s->h1,(size_t)rows*H); A(&s->q,(size_t)rows*qd); A(&s->k,(size_t)rows*kvd); A(&s->v,(size_t)rows*kvd);
    A(&s->attn,(size_t)rows*qd); A(&s->mix,(size_t)rows*H); A(&s->h2,(size_t)rows*H);
    A(&s->g,(size_t)rows*I); A(&s->u,(size_t)rows*I); A(&s->ff,(size_t)rows*I); A(&s->out,(size_t)rows*H);
    if(!ok) cudaGetLastError();
    return ok;
}
static inline void q35_dflash_fwd_scratch_free(q35_dflash_fwd_scratch* s){
    float* ps[]={s->h1,s->q,s->k,s->v,s->attn,s->mix,s->h2,s->g,s->u,s->ff,s->out};
    for(float* p:ps) if(p) cudaFree(p);
    *s=q35_dflash_fwd_scratch{};
}

// Draft backbone forward over `rows` query rows with self-attention within the query block.
// x [rows,H] is the input embedding (mutated in place through residuals); pos [rows] absolute
// positions; out receives the final-normed hidden [rows,H]. Reads resident BF16 weight views.
// theta is the config RoPE base. Matches the validated backbone parity kernels bit-for-bit.
static inline void q35_dflash_backbone_forward(const q35_dflash_residency& R, q35_dflash_fwd_scratch& s,
                                               float* x, const int* dpos, double theta, float eps,
                                               cudaStream_t st){
    const qwen35dflash::Geometry& g=R.geom;
    const int H=g.H,I=g.I,HD=g.HD,NQ=g.NQ,NKV=g.NKV,qd=g.q_dim(),kvd=g.kv_dim(),rows=s.rows,ctx=s.rows;
    auto gr=[&](int n){ return (unsigned)((n+255)/256); };
    for(int l=0;l<g.L;l++){
        const q35_dflash_layer_w& w=R.layers[l];
        q35df_rmsnorm<<<rows,256,0,st>>>(x,w.input_norm,s.h1,H,rows,eps);
        q35df_matmul<<<dim3(rows,(qd+7)/8),256,0,st>>>(s.h1,w.q_proj,s.q,rows,H,qd);
        q35df_matmul<<<dim3(rows,(kvd+7)/8),256,0,st>>>(s.h1,w.k_proj,s.k,rows,H,kvd);
        q35df_matmul<<<dim3(rows,(kvd+7)/8),256,0,st>>>(s.h1,w.v_proj,s.v,rows,H,kvd);
        q35df_headnorm<<<dim3(rows,NQ),128,0,st>>>(s.q,w.q_norm,rows,NQ,HD,eps);
        q35df_headnorm<<<dim3(rows,NKV),128,0,st>>>(s.k,w.k_norm,rows,NKV,HD,eps);
        q35df_rope<<<dim3(rows,NQ),64,0,st>>>(s.q,dpos,rows,NQ,HD,theta);
        q35df_rope<<<dim3(rows,NKV),64,0,st>>>(s.k,dpos,rows,NKV,HD,theta);
        q35df_attn<<<dim3(rows,NQ),128,(size_t)ctx*sizeof(double),st>>>(s.q,s.k,s.v,s.attn,rows,ctx,NQ,NKV,HD);
        q35df_matmul<<<dim3(rows,(H+7)/8),256,0,st>>>(s.attn,w.o_proj,s.mix,rows,qd,H);
        q35df_residual<<<gr(rows*H),256,0,st>>>(x,s.mix,rows*H);
        q35df_rmsnorm<<<rows,256,0,st>>>(x,w.post_norm,s.h2,H,rows,eps);
        q35df_matmul<<<dim3(rows,(I+7)/8),256,0,st>>>(s.h2,w.gate_proj,s.g,rows,H,I);
        q35df_matmul<<<dim3(rows,(I+7)/8),256,0,st>>>(s.h2,w.up_proj,s.u,rows,H,I);
        q35df_siluglu<<<gr(rows*I),256,0,st>>>(s.g,s.u,s.ff,rows*I);
        q35df_matmul<<<dim3(rows,(H+7)/8),256,0,st>>>(s.ff,w.down_proj,s.mix,rows,I,H);
        q35df_residual<<<gr(rows*H),256,0,st>>>(x,s.mix,rows*H);
    }
    q35df_rmsnorm<<<rows,256,0,st>>>(x,R.final_norm,s.out,H,rows,eps);
}

// ── Aux gather: capture layout -> drafter concat layout ──
// The target aux capture stores features as [feature_slot][row][H] (slot-major, stride maxrows*H).
// The drafter's fc expects, per row, the concatenation concat[row][f*H + i] over the F features.
// This kernel gathers a contiguous [rows] window (row_base..row_base+rows) into concat_out
// [rows, F*H]. One thread per element. F = num_target_features.
__global__ void q35df_aux_gather(const float* aux, float* concat_out, int F, int maxrows,
                                 int H, int row_base, int rows){
    long idx = (long)blockIdx.x*blockDim.x + threadIdx.x;
    long total = (long)rows * F * H;
    if (idx >= total) return;
    int i = idx % H; long t = idx / H; int fslot = t % F; int r = t / F;
    concat_out[(size_t)r*F*H + (size_t)fslot*H + i] = aux[(size_t)fslot*maxrows*H + (size_t)(row_base+r)*H + i];
}
static inline void q35_dflash_aux_gather(const float* aux, float* concat_out, int F, int maxrows,
                                         int H, int row_base, int rows, cudaStream_t st){
    long total=(long)rows*F*H; unsigned b=(unsigned)((total+255)/256);
    q35df_aux_gather<<<b,256,0,st>>>(aux, concat_out, F, maxrows, H, row_base, rows);
}

// ── Aux-hidden combine (the target->draft input interface) ──
// When use_aux_hidden, the draft's input hidden for each row is fc(concat of F target-layer hidden
// states), fc: [H, F*H]. This is the exact interface the target engine feeds: it gathers the F
// configured target layers' hidden states, concatenates them per row [rows, F*H], and projects
// through the BF16 fc weight to [rows, H]. Matmul reuses q35df_matmul. This produces the input to
// precompute_context_kv (for context rows) and to the query forward (for the query rows).
static inline void q35_dflash_combine_aux(const q35_dflash_residency& R, const float* concat_aux,
        float* out, int rows, cudaStream_t st){
    const qwen35dflash::Geometry& g=R.geom;
    // fc: out[rows,H] = concat_aux[rows, F*H] @ fc^T ; fc is [H, F*H].
    q35df_matmul<<<dim3(rows,(g.H+7)/8),256,0,st>>>(concat_aux, R.fc, out, rows, g.fc_in(), g.H);
}

// ── Context-KV precompute (the DFlash cross-attention trick) ──
// The draft never re-runs its layers over the context. The TARGET model's hidden states for the
// context rows are projected into EACH draft layer's K/V once: hidden RMSNorm -> per-layer K/V
// projection -> grouped per-head K-norm -> neox RoPE on K. This callable produces, for every draft
// layer l, the context K [num_ctx, NKV, HD] and V [num_ctx, NKV, HD] in ctxK[l]/ctxV[l] device
// buffers the caller owns. Matches the validated precompute parity kernels bit-for-bit. Runs eagerly
// (variable num_ctx), outside any captured graph.
struct q35_dflash_ctx_scratch {
    float *normed;   // [num_ctx, H]
    int    num_ctx;
};
static inline bool q35_dflash_ctx_scratch_alloc(q35_dflash_ctx_scratch* c, const qwen35dflash::Geometry& g, int num_ctx){
    c->num_ctx=num_ctx;
    return cudaMalloc(&c->normed,(size_t)num_ctx*g.H*4)==cudaSuccess;
}
static inline void q35_dflash_ctx_scratch_free(q35_dflash_ctx_scratch* c){ if(c->normed) cudaFree(c->normed); *c=q35_dflash_ctx_scratch{}; }

// Precompute context K/V for all layers into caller buffers at a WRITE OFFSET (in rows). ctxK[l]/
// ctxV[l] point at persistent per-layer caches [cap*NKV*HD]; this writes num_ctx rows starting at
// dst_row. ctx_hidden [num_ctx,H] is the target hidden; ctx_pos [num_ctx] absolute positions. When
// dst_row==0 this is a full (re)compute; dst_row>0 APPENDS new context to an accumulating cache
// (each decode step inserts only the newly-committed target tokens, matching vLLM's per-step
// precompute_and_store_context_kv into a persistent cache). Deterministic: appended rows are a pure
// function of their own hidden+position, so append == full recompute over the same rows.
static inline void q35_dflash_precompute_context_kv_at(const q35_dflash_residency& R,
        q35_dflash_ctx_scratch& c, const float* ctx_hidden, const int* ctx_pos, int num_ctx,
        float* const* ctxK, float* const* ctxV, int dst_row, int cap, double theta, float eps,
        cudaStream_t st){
    const qwen35dflash::Geometry& g=R.geom;
    const int H=g.H,HD=g.HD,NKV=g.NKV,kvd=g.kv_dim();
    if(num_ctx<=0 || dst_row+num_ctx>cap) return;
    q35df_rmsnorm<<<num_ctx,256,0,st>>>(ctx_hidden,R.hidden_norm,c.normed,H,num_ctx,eps);
    for(int l=0;l<g.L;l++){
        const q35_dflash_layer_w& w=R.layers[l];
        float* kdst=ctxK[l]+(size_t)dst_row*kvd; float* vdst=ctxV[l]+(size_t)dst_row*kvd;
        q35df_matmul<<<dim3(num_ctx,(kvd+7)/8),256,0,st>>>(c.normed,w.k_proj,kdst,num_ctx,H,kvd);
        q35df_matmul<<<dim3(num_ctx,(kvd+7)/8),256,0,st>>>(c.normed,w.v_proj,vdst,num_ctx,H,kvd);
        q35df_headnorm<<<dim3(num_ctx,NKV),128,0,st>>>(kdst,w.k_norm,num_ctx,NKV,HD,eps);
        q35df_rope<<<dim3(num_ctx,NKV),64,0,st>>>(kdst,ctx_pos,num_ctx,NKV,HD,theta);
    }
}
// Full (re)compute at offset 0 (back-compat wrapper).
static inline void q35_dflash_precompute_context_kv(const q35_dflash_residency& R,
        q35_dflash_ctx_scratch& c, const float* ctx_hidden, const int* ctx_pos,
        float* const* ctxK, float* const* ctxV, double theta, float eps, cudaStream_t st){
    q35_dflash_precompute_context_kv_at(R,c,ctx_hidden,ctx_pos,c.num_ctx,ctxK,ctxV,0,c.num_ctx,theta,eps,st);
}

// Full DFlash query forward: the (1+K) query rows attend the precomputed context K/V (ctxK/ctxV
// per layer) PLUS their own query-block K/V, non-causally, through all draft layers + final norm.
// This is the true DFlash drafting forward. x [rows,H] is the query embedding (mutated in place);
// dpos [rows] absolute positions; ctxK[l]/ctxV[l] the context K/V from precompute; num_ctx context
// rows. out receives the final-normed hidden. Reads resident BF16 weights.
static inline void q35_dflash_query_forward(const q35_dflash_residency& R, q35_dflash_fwd_scratch& s,
        float* x, const int* dpos, int num_ctx, float* const* ctxK, float* const* ctxV,
        double theta, float eps, cudaStream_t st, const int* dctx_pos=nullptr){
    const qwen35dflash::Geometry& g=R.geom;
    const int H=g.H,I=g.I,HD=g.HD,NQ=g.NQ,NKV=g.NKV,qd=g.q_dim(),kvd=g.kv_dim(),rows=s.rows;
    auto gr=[&](int n){ return (unsigned)((n+255)/256); };
    size_t smem=(size_t)(num_ctx+rows)*sizeof(double);
    for(int l=0;l<g.L;l++){
        const q35_dflash_layer_w& w=R.layers[l];
        // Sliding-window (SWA) draft layers are CAUSAL; the full_attention layer is non-causal
        // (per vLLM _resolve_layer_attention). Causal masking needs device absolute positions.
        int causal = (l < (int)g.layer_attn.size() && g.layer_attn[l]==qwen35dflash::ATTN_SLIDING) ? 1 : 0;
        q35df_rmsnorm<<<rows,256,0,st>>>(x,w.input_norm,s.h1,H,rows,eps);
        q35df_matmul<<<dim3(rows,(qd+7)/8),256,0,st>>>(s.h1,w.q_proj,s.q,rows,H,qd);
        q35df_matmul<<<dim3(rows,(kvd+7)/8),256,0,st>>>(s.h1,w.k_proj,s.k,rows,H,kvd);
        q35df_matmul<<<dim3(rows,(kvd+7)/8),256,0,st>>>(s.h1,w.v_proj,s.v,rows,H,kvd);
        q35df_headnorm<<<dim3(rows,NQ),128,0,st>>>(s.q,w.q_norm,rows,NQ,HD,eps);
        q35df_headnorm<<<dim3(rows,NKV),128,0,st>>>(s.k,w.k_norm,rows,NKV,HD,eps);
        q35df_rope<<<dim3(rows,NQ),64,0,st>>>(s.q,dpos,rows,NQ,HD,theta);
        q35df_rope<<<dim3(rows,NKV),64,0,st>>>(s.k,dpos,rows,NKV,HD,theta);
        q35df_attn_ctx<<<dim3(rows,NQ),128,smem,st>>>(s.q,ctxK[l],ctxV[l],s.k,s.v,s.attn,rows,num_ctx,NQ,NKV,HD,causal,dctx_pos,dpos);
        q35df_matmul<<<dim3(rows,(H+7)/8),256,0,st>>>(s.attn,w.o_proj,s.mix,rows,qd,H);
        q35df_residual<<<gr(rows*H),256,0,st>>>(x,s.mix,rows*H);
        q35df_rmsnorm<<<rows,256,0,st>>>(x,w.post_norm,s.h2,H,rows,eps);
        q35df_matmul<<<dim3(rows,(I+7)/8),256,0,st>>>(s.h2,w.gate_proj,s.g,rows,H,I);
        q35df_matmul<<<dim3(rows,(I+7)/8),256,0,st>>>(s.h2,w.up_proj,s.u,rows,H,I);
        q35df_siluglu<<<gr(rows*I),256,0,st>>>(s.g,s.u,s.ff,rows*I);
        q35df_matmul<<<dim3(rows,(H+7)/8),256,0,st>>>(s.ff,w.down_proj,s.mix,rows,I,H);
        q35df_residual<<<gr(rows*H),256,0,st>>>(x,s.mix,rows*H);
    }
    q35df_rmsnorm<<<rows,256,0,st>>>(x,R.final_norm,s.out,H,rows,eps);
}

// ── Draft sampling ──
// After the query forward, the K mask-token query rows (offsets 1..K in the (1+K) block) each
// predict a draft token via the LM head (shared with the target). Greedy drafting takes the argmax
// of each sampled row's logits. This kernel projects one sampled hidden row through the BF16 LM
// head [vocab, H] and reduces to an argmax, one block per sampled row. Deterministic (lowest index
// wins ties), matching q35_dflash_argmax. The probabilistic path uses the P1 shared-key gumbel over
// the same logits; kept out of this fixed-shape kernel so greedy stays a pure argmax.
__global__ void q35df_head_argmax(const float* hidden, const __nv_bfloat16* lm_head, int H, int vocab,
                                  int32_t* out_tok){
    int row=blockIdx.x; const float* h=hidden+(size_t)row*H;
    // Cache the hidden row in shared memory (H<=8192 floats = 32KB) so the vocab loop reads it fast.
    extern __shared__ float hs[];
    for(int i=threadIdx.x;i<H;i+=blockDim.x) hs[i]=h[i];
    __syncthreads();
    __shared__ float bestv[256]; __shared__ int besti[256];
    float bv=-1e30f; int bi=0;
    for(int o=threadIdx.x;o<vocab;o+=blockDim.x){
        const __nv_bfloat16* w=lm_head+(size_t)o*H; float acc=0.0f; for(int i=0;i<H;i++) acc+=hs[i]*__bfloat162float(w[i]);
        if(acc>bv || (acc==bv && o<bi)){ bv=acc; bi=o; }
    }
    bestv[threadIdx.x]=bv; besti[threadIdx.x]=bi; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s){ if(bestv[threadIdx.x+s]>bestv[threadIdx.x] || (bestv[threadIdx.x+s]==bestv[threadIdx.x] && besti[threadIdx.x+s]<besti[threadIdx.x])){ bestv[threadIdx.x]=bestv[threadIdx.x+s]; besti[threadIdx.x]=besti[threadIdx.x+s]; } } __syncthreads(); }
    if(threadIdx.x==0) out_tok[row]=besti[0];
}

// Greedy draft sampling: argmax of the LM head over each of the K sampled query rows. sampled_hidden
// is [K, H] (the mask-token rows gathered from the query forward's final-normed output); lm_head is
// the shared BF16 [vocab, H]; out_tok receives K draft token ids.
// Batched greedy head argmax: reads the 2 GB BF16 LM head ONCE for all K rows (each weight row is
// reused across the K hidden rows), vs q35df_head_argmax's K separate passes. Each thread owns one
// vocab token, loads its H-vector weight once, and computes the dot with all K hidden rows (cached
// in shared memory), tracking a per-vocab-tile local argmax per row; a final reduce over tiles picks
// the global argmax per row. Deterministic lowest-index tie-break preserved. This is the K-fold
// weight-traffic reduction (the draft head is bandwidth-bound). Grid: (ceil(vocab/BLK)); the per-
// row partial argmax is written to scratch [K, nblocks] then reduced. To keep it simple + still
// read weights once, we use one launch that atomically maintains a per-row best via a block-local
// reduction into global per-row {val,idx} using a lock-free max on a packed 64-bit (val,idx).
// Warp-per-vocab-token: 32 threads cooperate on each token's H-dot (strided + warp-shuffle reduce),
// reading the weight row once and accumulating against all K hidden rows (cached in shared memory).
// Lane 0 updates the per-row global best via atomicMax on a packed (sortable-float<<32 | ~idx) key
// => (max value, lowest index) tie-break. Reads the 2 GB head once for all K rows AND parallelizes
// the H-loop 32x. Block: 256 threads = 8 warps; grid.x = ceil(vocab/8). Shared: K*H floats hidden.
__global__ void q35df_head_argmax_batched(const float* hidden, const __nv_bfloat16* lm_head, int H,
                                          int vocab, int K, unsigned long long* row_best){
    // No shared cache (K*H too large); hidden [K,H] is small and stays L2-resident. The 2 GB WEIGHT
    // read is the cost, done once per token here (reused across the K rows), with the H-dot split
    // across the warp.
    int lane=threadIdx.x&31; int warp=threadIdx.x>>5; int wpb=blockDim.x>>5;
    int o = blockIdx.x*wpb + warp; if(o>=vocab) return;
    const __nv_bfloat16* w = lm_head + (size_t)o*H;
    for(int r=0;r<K;r++){
        const float* h = hidden + (size_t)r*H;
        float acc=0.0f; for(int i=lane;i<H;i+=32) acc+=h[i]*__bfloat162float(w[i]);
        for(int s=16;s>0;s>>=1) acc+=__shfl_down_sync(0xffffffffu,acc,s);
        if(lane==0){
            unsigned int fb = __float_as_uint(acc); fb ^= (fb>>31)?0xffffffffu:0x80000000u;
            unsigned long long packed = ((unsigned long long)fb<<32) | (unsigned int)(0xffffffffu - (unsigned)o);
            atomicMax(&row_best[r], packed);
        }
    }
}
__global__ void q35df_head_best_decode(const unsigned long long* row_best, int K, int32_t* out_tok){
    int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=K) return;
    unsigned long long p=row_best[r]; unsigned int idx = 0xffffffffu - (unsigned)(p & 0xffffffffu);
    out_tok[r]=(int32_t)idx;
}
static inline void q35_dflash_sample_greedy(const float* sampled_hidden, const __nv_bfloat16* lm_head,
        int K, int H, int vocab, int32_t* out_tok, cudaStream_t st){
    // Batched head: one weight pass for all K rows. Needs a [K] u64 best buffer (init to 0).
    static unsigned long long* row_best=nullptr; static int rb_cap=0;
    if(rb_cap<K){ if(row_best) cudaFree(row_best); cudaMalloc(&row_best,(size_t)K*sizeof(unsigned long long)); rb_cap=K; }
    cudaMemsetAsync(row_best,0,(size_t)K*sizeof(unsigned long long),st);
    int blk=256; int wpb=blk>>5;   // 8 warps/block, one warp per vocab token
    q35df_head_argmax_batched<<<(vocab+wpb-1)/wpb,blk,0,st>>>(sampled_hidden,lm_head,H,vocab,K,row_best);
    q35df_head_best_decode<<<(K+31)/32,32,0,st>>>(row_best,K,out_tok);
}

// Probabilistic draft sampling with the shared-key uniform: row r samples from softmax(logits_r/T)
// via inverse-CDF using u = CounterRNG(seed, pos_r, SAMPLE). Also materializes the per-row logits
// into out_logits [K, vocab] (fp32) so the verifier can compute the rejection ratio against the same
// draft distribution. One block per row: compute logits (LM head), softmax denom, inverse-CDF pick.
// Matches q35_dflash_uniform_open + softmax inverse-CDF (the P1 draft-sample contract) bit-for-bit.
__global__ void q35df_head_sample(const float* hidden, const __nv_bfloat16* lm_head, int H, int vocab,
                                  float* out_logits, int32_t* out_tok, double temp,
                                  uint64_t seed, const int64_t* pos){
    int row=blockIdx.x; const float* h=hidden+(size_t)row*H; float* lo=out_logits+(size_t)row*vocab;
    // Compute logits row (each thread strides the vocab), and track max for stable softmax.
    __shared__ double smax; __shared__ double ssum;
    double lmax=-1e300;
    for(int o=threadIdx.x;o<vocab;o+=blockDim.x){ const __nv_bfloat16* w=lm_head+(size_t)o*H; double acc=0; for(int i=0;i<H;i++) acc+=(double)h[i]*(double)__bfloat162float(w[i]); double l=acc/temp; lo[o]=(float)l; if(l>lmax) lmax=l; }
    // block max reduce
    __shared__ double rbuf[256]; rbuf[threadIdx.x]=lmax; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s && rbuf[threadIdx.x+s]>rbuf[threadIdx.x]) rbuf[threadIdx.x]=rbuf[threadIdx.x+s]; __syncthreads(); }
    if(threadIdx.x==0) smax=rbuf[0]; __syncthreads();
    // block sum of exp
    double ps=0; for(int o=threadIdx.x;o<vocab;o+=blockDim.x) ps+=exp((double)lo[o]-smax);
    rbuf[threadIdx.x]=ps; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) rbuf[threadIdx.x]+=rbuf[threadIdx.x+s]; __syncthreads(); }
    if(threadIdx.x==0) ssum=rbuf[0]; __syncthreads();
    // inverse-CDF on thread 0 (vocab-serial; K rows in parallel across blocks).
    if(threadIdx.x==0){
        double u=q35_dflash_uniform_open(seed, pos[row], Q35_DFLASH_DOMAIN_SAMPLE);
        double target=u*ssum, acc=0; int32_t pick=vocab-1;
        for(int o=0;o<vocab;o++){ acc+=exp((double)lo[o]-smax); if(acc>=target){ pick=o; break; } }
        out_tok[row]=pick;
    }
}
static inline void q35_dflash_sample_prob(const float* sampled_hidden, const __nv_bfloat16* lm_head,
        int K, int H, int vocab, float* out_logits, int32_t* out_tok, double temp,
        uint64_t seed, const int64_t* dpos, cudaStream_t st){
    q35df_head_sample<<<K,256,0,st>>>(sampled_hidden, lm_head, H, vocab, out_logits, out_tok, temp, seed, dpos);
}

// ── Greedy verify-accept over the target logit block ──
// Given target logits for the (1+K) verify rows [rows, vocab] and the K draft tokens, compute the
// greedy accepted length j and the emitted token at position j (target argmax at the first mismatch,
// or the bonus argmax if all K match). Matches q35_dflash_verify_greedy bit-for-bit. Row t of the
// logits block is the target's prediction for the token AFTER query offset t; draft[i] is compared
// against argmax(row i) for i in [0,K). Single block: one thread-team argmaxes each needed row.
// out_accept[0] = j; out_emit[0] = emitted token. Deterministic (lowest index wins ties).
__global__ void q35df_verify_argmax(const float* logits, int rows, int vocab, int32_t* row_argmax){
    // Each block handles one row's argmax; grid.x = rows.
    __shared__ double bv[256]; __shared__ int bi[256];
    int row=blockIdx.x; const float* lr=logits+(size_t)row*vocab;
    double lv=-1e300; int li=0;
    for(int o=threadIdx.x;o<vocab;o+=blockDim.x){ double v=lr[o]; if(v>lv || (v==lv && o<li)){ lv=v; li=o; } }
    bv[threadIdx.x]=lv; bi[threadIdx.x]=li; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s){ if(bv[threadIdx.x+s]>bv[threadIdx.x] || (bv[threadIdx.x+s]==bv[threadIdx.x] && bi[threadIdx.x+s]<bi[threadIdx.x])){ bv[threadIdx.x]=bv[threadIdx.x+s]; bi[threadIdx.x]=bi[threadIdx.x+s]; } } __syncthreads(); }
    if(threadIdx.x==0) row_argmax[row]=bi[0];
}
// Reduce the per-row argmaxes (in row_argmax[rows]) + draft to accepted_len + emitted (single thread).
__global__ void q35df_verify_reduce(const int32_t* row_argmax, const int32_t* draft, int K,
                                    int* out_accept, int32_t* out_emit){
    if(threadIdx.x||blockIdx.x) return;
    int j=0;
    for(int i=0;i<K;i++){ if(row_argmax[i]==draft[i]) j++; else break; }
    // emitted: target argmax at position j (row j is the bonus row when j==K).
    *out_accept=j; *out_emit=row_argmax[j];
}

// Host-side driver: rows must be K+1 (K draft rows + 1 bonus row). row_argmax scratch [rows] int32.
static inline void q35_dflash_verify_greedy_device(const float* logits, int rows, int vocab,
        const int32_t* draft, int K, int32_t* row_argmax, int* out_accept, int32_t* out_emit,
        cudaStream_t st){
    q35df_verify_argmax<<<rows,256,0,st>>>(logits, rows, vocab, row_argmax);
    q35df_verify_reduce<<<1,1,0,st>>>(row_argmax, draft, K, out_accept, out_emit);
}

// ── Query token embedding (bonus + mask rows) ──
// The DFlash QUERY rows embed token ids (the bonus/last-accepted token at offset 0 and the mask
// token at offsets 1..K), NOT fc(aux). Only the CONTEXT K/V comes from fc(aux). This kernel gathers
// the BF16 shared embedding rows for the (1+K) query token ids into a [rows, H] fp32 buffer.
__global__ void q35df_embed(const int32_t* ids, const __nv_bfloat16* embed, float* out, int rows, int H){
    int r=blockIdx.x; if(r>=rows) return; int id=ids[r]; const __nv_bfloat16* e=embed+(size_t)id*H; float* o=out+(size_t)r*H;
    for(int i=threadIdx.x;i<H;i+=blockDim.x) o[i]=__bfloat162float(e[i]);
}
static inline void q35_dflash_embed_query(const int32_t* dids, const __nv_bfloat16* embed, float* out,
                                          int rows, int H, cudaStream_t st){
    q35df_embed<<<rows,256,0,st>>>(dids, embed, out, rows, H);
}

// ── Probabilistic verify-accept over target+draft logit blocks (single request, one block) ──
// Device analog of q35_dflash_verify_prob operating on resident logit blocks. target_logits is
// (1+K) rows [vocab]; draft_logits is K rows [vocab]; draft_tokens the K drafted ids; pos[i] the
// absolute position of drafted token i; pos_bonus the bonus position. Uses the shared-key uniforms
// (seed, pos, ACCEPT/RESIDUAL/SAMPLE) so it reproduces the draft's own draws. Single block (the
// vocab-serial rejection is cheap for one request); out_accept[0]=j, out_emit[0]=emitted token.
// Matches the P1 host oracle q35_dflash_verify_prob bit-for-bit.
__global__ void q35df_verify_prob_kernel(const float* target_logits, const float* draft_logits,
        int vocab, const int32_t* draft_tokens, int K, uint64_t seed,
        const int64_t* pos, int64_t pos_bonus, int* out_accept, int32_t* out_emit){
    if(threadIdx.x||blockIdx.x) return;
    // Reuse the shared host/device reference directly (it is __host__ __device__).
    q35_dflash_verify_result r = q35_dflash_verify_prob(target_logits, draft_logits, vocab,
                                                        draft_tokens, K, seed, pos, pos_bonus);
    *out_accept = r.accepted_len; *out_emit = r.emitted_token;
}
static inline void q35_dflash_verify_prob_device(const float* target_logits, const float* draft_logits,
        int vocab, const int32_t* draft_tokens, int K, uint64_t seed,
        const int64_t* pos, int64_t pos_bonus, int* out_accept, int32_t* out_emit, cudaStream_t st){
    q35df_verify_prob_kernel<<<1,1,0,st>>>(target_logits, draft_logits, vocab, draft_tokens, K,
                                           seed, pos, pos_bonus, out_accept, out_emit);
}

// ── Single drafting entry point (the verify loop calls this) ──
// Owns all drafting scratch + per-layer context K/V buffers for one request's draft step. Produced
// once per residency; reused across steps (fixed (1+K) query shape, variable context length up to a
// cap). C-style: no per-token host allocation; buffers sized at init to the context cap.
struct q35_dflash_drafter {
    q35_dflash_ctx_scratch ctx;
    q35_dflash_fwd_scratch fwd;
    float **ctxK;                 // [L] each [ctx_cap*NKV*HD]
    float **ctxV;
    float  *ctx_hidden;           // [ctx_cap, H] fc-combined context input
    float  *query_hidden;         // [1+K, H]  query input = embed(query_ids) (mutated by forward)
    int    *ctx_pos;              // [ctx_cap]
    int    *query_pos;            // [1+K]
    int32_t *query_ids;           // [1+K]  bonus token + K mask tokens
    int     ctx_cap;
    int     K;
    int     ready;
    int     ctxlen;               // rows of valid accumulated context in ctxK/ctxV (B=1 serving)
    int     ctx_aux_cap;          // rows of context aux held (== ctx_cap)
    float  *ctx_aux_concat;       // [ctx_aux_cap, F*H]: accumulated per-token target aux (context)
    int    *ctx_abs_pos;          // [ctx_aux_cap] host: absolute position of each context row
    float  *draft_logits;         // [K, vocab] probabilistic draft logits (NULL until prob path used)
    int64_t *sample_pos64;        // [K] device: absolute positions of the K sampled draft rows
    int     vocab;
};

static inline bool q35_dflash_drafter_init(q35_dflash_drafter* D, const q35_dflash_residency& R,
                                           int K, int ctx_cap){
    const qwen35dflash::Geometry& g=R.geom; const int kvd=g.kv_dim(), H=g.H, rows=1+K;
    *D=q35_dflash_drafter{}; D->K=K; D->ctx_cap=ctx_cap;
    bool ok=q35_dflash_ctx_scratch_alloc(&D->ctx,g,ctx_cap) && q35_dflash_fwd_scratch_alloc(&D->fwd,g,rows);
    D->ctxK=new float*[g.L]; D->ctxV=new float*[g.L];
    for(int l=0;l<g.L;l++){ ok = ok && cudaMalloc(&D->ctxK[l],(size_t)ctx_cap*kvd*4)==cudaSuccess;
                            ok = ok && cudaMalloc(&D->ctxV[l],(size_t)ctx_cap*kvd*4)==cudaSuccess; }
    ok = ok && cudaMalloc(&D->ctx_hidden,(size_t)ctx_cap*H*4)==cudaSuccess;
    ok = ok && cudaMalloc(&D->query_hidden,(size_t)rows*H*4)==cudaSuccess;
    ok = ok && cudaMalloc(&D->ctx_pos,(size_t)ctx_cap*sizeof(int))==cudaSuccess;
    ok = ok && cudaMalloc(&D->query_pos,(size_t)rows*sizeof(int))==cudaSuccess;
    ok = ok && cudaMalloc(&D->query_ids,(size_t)rows*sizeof(int32_t))==cudaSuccess;
    // Hold the full accumulated context aux up to the context cap (config sliding window bounds it).
    D->ctx_aux_cap = ctx_cap;
    ok = ok && cudaMalloc(&D->ctx_aux_concat,(size_t)ctx_cap*g.fc_in()*4)==cudaSuccess;
    D->ctx_abs_pos = (int*)malloc(sizeof(int)*ctx_cap);
    ok = ok && (D->ctx_abs_pos != nullptr);
    // Probabilistic buffers (draft logits [K,vocab] + int64 sample positions). Allocated here so the
    // prob path pays no per-step alloc; ~K*vocab*4 = 16 * 248320 * 4 = 15.9 MiB.
    D->vocab = g.V;
    ok = ok && cudaMalloc(&D->draft_logits,(size_t)K*(size_t)g.V*4)==cudaSuccess;
    ok = ok && cudaMalloc(&D->sample_pos64,(size_t)K*sizeof(int64_t))==cudaSuccess;
    D->ctxlen=0;
    if(!ok) cudaGetLastError();
    D->ready=ok?1:0; return ok;
}
static inline void q35_dflash_drafter_free(q35_dflash_drafter* D, const q35_dflash_residency& R){
    if(D->ctxK){ for(int l=0;l<R.geom.L;l++){ if(D->ctxK[l]) cudaFree(D->ctxK[l]); if(D->ctxV[l]) cudaFree(D->ctxV[l]); } delete[] D->ctxK; delete[] D->ctxV; }
    q35_dflash_ctx_scratch_free(&D->ctx); q35_dflash_fwd_scratch_free(&D->fwd);
    if(D->ctx_hidden) cudaFree(D->ctx_hidden); if(D->query_hidden) cudaFree(D->query_hidden);
    if(D->ctx_pos) cudaFree(D->ctx_pos); if(D->query_pos) cudaFree(D->query_pos);
    if(D->query_ids) cudaFree(D->query_ids);
    if(D->ctx_aux_concat) cudaFree(D->ctx_aux_concat);
    if(D->ctx_abs_pos) free(D->ctx_abs_pos);
    if(D->draft_logits) cudaFree(D->draft_logits);
    if(D->sample_pos64) cudaFree(D->sample_pos64);
    *D=q35_dflash_drafter{};
}

// One draft step. Per qwen3_dflash.py: the CONTEXT K/V is projected from fc(aux) but the QUERY rows
// EMBED token ids (bonus + mask tokens), not fc(aux). ctx_concat_aux [num_ctx, F*H] is the target's
// gathered aux features for the context rows; query_ids [1+K] are the query token ids (bonus token
// at offset 0, mask_token_id at offsets 1..K); embed is the shared BF16 embedding [vocab,H].
// ctx_positions/query_positions are absolute positions. Produces K greedy draft tokens into
// out_draft. Steps: fc combine (context) -> precompute context K/V -> embed query ids -> query
// forward -> LM-head argmax on the K mask rows. All validated stages; deterministic. num_ctx<=cap.
static inline int q35_dflash_draft_greedy(const q35_dflash_residency& R, q35_dflash_drafter& D,
        const float* ctx_concat_aux, const int* ctx_positions, int num_ctx,
        const int32_t* query_ids, const int* query_positions,
        const __nv_bfloat16* embed, const __nv_bfloat16* lm_head, int vocab, int32_t* out_draft,
        double theta, float eps, cudaStream_t st){
    if(!D.ready || num_ctx>D.ctx_cap) return -1;
    const qwen35dflash::Geometry& g=R.geom; const int H=g.H, rows=1+D.K;
    D.ctx.num_ctx=num_ctx;
    cudaMemcpyAsync(D.ctx_pos, ctx_positions, (size_t)num_ctx*sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(D.query_pos, query_positions, (size_t)rows*sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(D.query_ids, query_ids, (size_t)rows*sizeof(int32_t), cudaMemcpyHostToDevice, st);
    // Context input = fc(aux); query input = embed(query token ids).
    q35_dflash_combine_aux(R, ctx_concat_aux, D.ctx_hidden, num_ctx, st);
    q35_dflash_embed_query(D.query_ids, embed, D.query_hidden, rows, H, st);
    // precompute context K/V, then query forward.
    q35_dflash_precompute_context_kv(R, D.ctx, D.ctx_hidden, D.ctx_pos, D.ctxK, D.ctxV, theta, eps, st);
    q35_dflash_query_forward(R, D.fwd, D.query_hidden, D.query_pos, num_ctx, D.ctxK, D.ctxV, theta, eps, st, D.ctx_pos);
    // sample the K mask rows (query offsets 1..K) from the final-normed query output.
    q35_dflash_sample_greedy(D.fwd.out + (size_t)1*H, lm_head, D.K, H, vocab, out_draft, st);
    return 0;
}

// Probabilistic draft step: identical to q35_dflash_draft_greedy through the query forward, but the
// K mask rows are sampled from softmax(logits/temp) via the shared-key uniform (seed, sample_pos64,
// SAMPLE), and the per-row draft logits are retained in D.draft_logits for the rejection sampler.
// sample_pos64 [K] are the absolute positions of the drafted tokens (query offsets 1..K).
static inline int q35_dflash_draft_prob(const q35_dflash_residency& R, q35_dflash_drafter& D,
        const float* ctx_concat_aux, const int* ctx_positions, int num_ctx,
        const int32_t* query_ids, const int* query_positions, const int64_t* sample_pos64,
        const __nv_bfloat16* embed, const __nv_bfloat16* lm_head, int vocab, int32_t* out_draft,
        double temp, uint64_t seed, double theta, float eps, cudaStream_t st){
    if(!D.ready || num_ctx>D.ctx_cap || !D.draft_logits) return -1;
    const qwen35dflash::Geometry& g=R.geom; const int H=g.H, rows=1+D.K;
    D.ctx.num_ctx=num_ctx;
    cudaMemcpyAsync(D.ctx_pos, ctx_positions, (size_t)num_ctx*sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(D.query_pos, query_positions, (size_t)rows*sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(D.query_ids, query_ids, (size_t)rows*sizeof(int32_t), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(D.sample_pos64, sample_pos64, (size_t)D.K*sizeof(int64_t), cudaMemcpyHostToDevice, st);
    q35_dflash_combine_aux(R, ctx_concat_aux, D.ctx_hidden, num_ctx, st);
    q35_dflash_embed_query(D.query_ids, embed, D.query_hidden, rows, H, st);
    q35_dflash_precompute_context_kv(R, D.ctx, D.ctx_hidden, D.ctx_pos, D.ctxK, D.ctxV, theta, eps, st);
    q35_dflash_query_forward(R, D.fwd, D.query_hidden, D.query_pos, num_ctx, D.ctxK, D.ctxV, theta, eps, st, D.ctx_pos);
    // Probabilistic sample of the K mask rows (query offsets 1..K), retaining per-row draft logits.
    q35_dflash_sample_prob(D.fwd.out + (size_t)1*H, lm_head, D.K, H, vocab, D.draft_logits,
                           out_draft, temp, seed, D.sample_pos64, st);
    return 0;
}

#endif // FUCINA_QWEN35_DFLASH_FORWARD_CUH
