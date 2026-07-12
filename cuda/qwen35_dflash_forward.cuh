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

// ── device kernels reading BF16 weights (fp32 compute, double accumulation) ──
__global__ void q35df_rmsnorm(const float* x,const __nv_bfloat16* w,float* y,int H,int rows,float eps){
    int r=blockIdx.x; if(r>=rows) return; const float* xr=x+(size_t)r*H; float* yr=y+(size_t)r*H;
    __shared__ double buf[256]; double loc=0; for(int i=threadIdx.x;i<H;i+=blockDim.x){ double v=xr[i]; loc+=v*v; }
    buf[threadIdx.x]=loc; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) buf[threadIdx.x]+=buf[threadIdx.x+s]; __syncthreads(); }
    __shared__ double ss; if(threadIdx.x==0) ss=buf[0]; __syncthreads();
    double inv=1.0/sqrt(ss/H+eps); for(int i=threadIdx.x;i<H;i+=blockDim.x) yr[i]=(float)((double)xr[i]*inv*(double)__bfloat162float(w[i]));
}
// O[rows,outd] = A[rows,in] @ W[outd,in]^T ; W is BF16.
__global__ void q35df_matmul(const float* A,const __nv_bfloat16* W,float* O,int rows,int in,int outd){
    int r=blockIdx.x; int o=blockIdx.y*blockDim.x+threadIdx.x; if(r>=rows||o>=outd) return;
    const float* a=A+(size_t)r*in; const __nv_bfloat16* w=W+(size_t)o*in;
    double acc=0; for(int i=0;i<in;i++) acc+=(double)a[i]*(double)__bfloat162float(w[i]);
    O[(size_t)r*outd+o]=(float)acc;
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
__global__ void q35df_attn_ctx(const float* Q,const float* Kc,const float* Vc,const float* Kq,const float* Vq,
                               float* O,int rows,int ctx,int NQ,int NKV,int HD){
    int r=blockIdx.x; int h=blockIdx.y; if(r>=rows||h>=NQ) return; int g=h/(NQ/NKV);
    const float* q=Q+((size_t)r*NQ+h)*HD; extern __shared__ double sh[]; int tot=ctx+rows; double scale=1.0/sqrt((double)HD);
    for(int t=threadIdx.x;t<tot;t+=blockDim.x){
        const float* k = (t<ctx) ? Kc+((size_t)t*NKV+g)*HD : Kq+((size_t)(t-ctx)*NKV+g)*HD;
        double d=0; for(int i=0;i<HD;i++) d+=(double)q[i]*(double)k[i]; sh[t]=d*scale;
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
        q35df_matmul<<<dim3(rows,(qd+255)/256),256,0,st>>>(s.h1,w.q_proj,s.q,rows,H,qd);
        q35df_matmul<<<dim3(rows,(kvd+255)/256),256,0,st>>>(s.h1,w.k_proj,s.k,rows,H,kvd);
        q35df_matmul<<<dim3(rows,(kvd+255)/256),256,0,st>>>(s.h1,w.v_proj,s.v,rows,H,kvd);
        q35df_headnorm<<<dim3(rows,NQ),128,0,st>>>(s.q,w.q_norm,rows,NQ,HD,eps);
        q35df_headnorm<<<dim3(rows,NKV),128,0,st>>>(s.k,w.k_norm,rows,NKV,HD,eps);
        q35df_rope<<<dim3(rows,NQ),64,0,st>>>(s.q,dpos,rows,NQ,HD,theta);
        q35df_rope<<<dim3(rows,NKV),64,0,st>>>(s.k,dpos,rows,NKV,HD,theta);
        q35df_attn<<<dim3(rows,NQ),128,(size_t)ctx*sizeof(double),st>>>(s.q,s.k,s.v,s.attn,rows,ctx,NQ,NKV,HD);
        q35df_matmul<<<dim3(rows,(H+255)/256),256,0,st>>>(s.attn,w.o_proj,s.mix,rows,qd,H);
        q35df_residual<<<gr(rows*H),256,0,st>>>(x,s.mix,rows*H);
        q35df_rmsnorm<<<rows,256,0,st>>>(x,w.post_norm,s.h2,H,rows,eps);
        q35df_matmul<<<dim3(rows,(I+255)/256),256,0,st>>>(s.h2,w.gate_proj,s.g,rows,H,I);
        q35df_matmul<<<dim3(rows,(I+255)/256),256,0,st>>>(s.h2,w.up_proj,s.u,rows,H,I);
        q35df_siluglu<<<gr(rows*I),256,0,st>>>(s.g,s.u,s.ff,rows*I);
        q35df_matmul<<<dim3(rows,(H+255)/256),256,0,st>>>(s.ff,w.down_proj,s.mix,rows,I,H);
        q35df_residual<<<gr(rows*H),256,0,st>>>(x,s.mix,rows*H);
    }
    q35df_rmsnorm<<<rows,256,0,st>>>(x,R.final_norm,s.out,H,rows,eps);
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
    q35df_matmul<<<dim3(rows,(g.H+255)/256),256,0,st>>>(concat_aux, R.fc, out, rows, g.fc_in(), g.H);
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

// Precompute context K/V for all layers. ctxK[l]/ctxV[l] must each be [num_ctx*NKV*HD] device
// buffers. ctx_hidden [num_ctx,H] is the target hidden state; ctx_pos [num_ctx] absolute positions.
static inline void q35_dflash_precompute_context_kv(const q35_dflash_residency& R,
        q35_dflash_ctx_scratch& c, const float* ctx_hidden, const int* ctx_pos,
        float* const* ctxK, float* const* ctxV, double theta, float eps, cudaStream_t st){
    const qwen35dflash::Geometry& g=R.geom;
    const int H=g.H,HD=g.HD,NKV=g.NKV,kvd=g.kv_dim(),num_ctx=c.num_ctx;
    // Shared hidden RMSNorm (hidden_norm) across layers.
    q35df_rmsnorm<<<num_ctx,256,0,st>>>(ctx_hidden,R.hidden_norm,c.normed,H,num_ctx,eps);
    for(int l=0;l<g.L;l++){
        const q35_dflash_layer_w& w=R.layers[l];
        q35df_matmul<<<dim3(num_ctx,(kvd+255)/256),256,0,st>>>(c.normed,w.k_proj,ctxK[l],num_ctx,H,kvd);
        q35df_matmul<<<dim3(num_ctx,(kvd+255)/256),256,0,st>>>(c.normed,w.v_proj,ctxV[l],num_ctx,H,kvd);
        q35df_headnorm<<<dim3(num_ctx,NKV),128,0,st>>>(ctxK[l],w.k_norm,num_ctx,NKV,HD,eps);
        q35df_rope<<<dim3(num_ctx,NKV),64,0,st>>>(ctxK[l],ctx_pos,num_ctx,NKV,HD,theta);
    }
}

// Full DFlash query forward: the (1+K) query rows attend the precomputed context K/V (ctxK/ctxV
// per layer) PLUS their own query-block K/V, non-causally, through all draft layers + final norm.
// This is the true DFlash drafting forward. x [rows,H] is the query embedding (mutated in place);
// dpos [rows] absolute positions; ctxK[l]/ctxV[l] the context K/V from precompute; num_ctx context
// rows. out receives the final-normed hidden. Reads resident BF16 weights.
static inline void q35_dflash_query_forward(const q35_dflash_residency& R, q35_dflash_fwd_scratch& s,
        float* x, const int* dpos, int num_ctx, float* const* ctxK, float* const* ctxV,
        double theta, float eps, cudaStream_t st){
    const qwen35dflash::Geometry& g=R.geom;
    const int H=g.H,I=g.I,HD=g.HD,NQ=g.NQ,NKV=g.NKV,qd=g.q_dim(),kvd=g.kv_dim(),rows=s.rows;
    auto gr=[&](int n){ return (unsigned)((n+255)/256); };
    size_t smem=(size_t)(num_ctx+rows)*sizeof(double);
    for(int l=0;l<g.L;l++){
        const q35_dflash_layer_w& w=R.layers[l];
        q35df_rmsnorm<<<rows,256,0,st>>>(x,w.input_norm,s.h1,H,rows,eps);
        q35df_matmul<<<dim3(rows,(qd+255)/256),256,0,st>>>(s.h1,w.q_proj,s.q,rows,H,qd);
        q35df_matmul<<<dim3(rows,(kvd+255)/256),256,0,st>>>(s.h1,w.k_proj,s.k,rows,H,kvd);
        q35df_matmul<<<dim3(rows,(kvd+255)/256),256,0,st>>>(s.h1,w.v_proj,s.v,rows,H,kvd);
        q35df_headnorm<<<dim3(rows,NQ),128,0,st>>>(s.q,w.q_norm,rows,NQ,HD,eps);
        q35df_headnorm<<<dim3(rows,NKV),128,0,st>>>(s.k,w.k_norm,rows,NKV,HD,eps);
        q35df_rope<<<dim3(rows,NQ),64,0,st>>>(s.q,dpos,rows,NQ,HD,theta);
        q35df_rope<<<dim3(rows,NKV),64,0,st>>>(s.k,dpos,rows,NKV,HD,theta);
        q35df_attn_ctx<<<dim3(rows,NQ),128,smem,st>>>(s.q,ctxK[l],ctxV[l],s.k,s.v,s.attn,rows,num_ctx,NQ,NKV,HD);
        q35df_matmul<<<dim3(rows,(H+255)/256),256,0,st>>>(s.attn,w.o_proj,s.mix,rows,qd,H);
        q35df_residual<<<gr(rows*H),256,0,st>>>(x,s.mix,rows*H);
        q35df_rmsnorm<<<rows,256,0,st>>>(x,w.post_norm,s.h2,H,rows,eps);
        q35df_matmul<<<dim3(rows,(I+255)/256),256,0,st>>>(s.h2,w.gate_proj,s.g,rows,H,I);
        q35df_matmul<<<dim3(rows,(I+255)/256),256,0,st>>>(s.h2,w.up_proj,s.u,rows,H,I);
        q35df_siluglu<<<gr(rows*I),256,0,st>>>(s.g,s.u,s.ff,rows*I);
        q35df_matmul<<<dim3(rows,(H+255)/256),256,0,st>>>(s.ff,w.down_proj,s.mix,rows,I,H);
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
    __shared__ double bestv[256]; __shared__ int besti[256];
    double bv=-1e300; int bi=0;
    for(int o=threadIdx.x;o<vocab;o+=blockDim.x){
        const __nv_bfloat16* w=lm_head+(size_t)o*H; double acc=0; for(int i=0;i<H;i++) acc+=(double)h[i]*(double)__bfloat162float(w[i]);
        if(acc>bv || (acc==bv && o<bi)){ bv=acc; bi=o; }
    }
    bestv[threadIdx.x]=bv; besti[threadIdx.x]=bi; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s){ if(bestv[threadIdx.x+s]>bestv[threadIdx.x] || (bestv[threadIdx.x+s]==bestv[threadIdx.x] && besti[threadIdx.x+s]<besti[threadIdx.x])){ bestv[threadIdx.x]=bestv[threadIdx.x+s]; besti[threadIdx.x]=besti[threadIdx.x+s]; } } __syncthreads(); }
    if(threadIdx.x==0) out_tok[row]=besti[0];
}

// Greedy draft sampling: argmax of the LM head over each of the K sampled query rows. sampled_hidden
// is [K, H] (the mask-token rows gathered from the query forward's final-normed output); lm_head is
// the shared BF16 [vocab, H]; out_tok receives K draft token ids.
static inline void q35_dflash_sample_greedy(const float* sampled_hidden, const __nv_bfloat16* lm_head,
        int K, int H, int vocab, int32_t* out_tok, cudaStream_t st){
    q35df_head_argmax<<<K,256,0,st>>>(sampled_hidden, lm_head, H, vocab, out_tok);
}

#endif // FUCINA_QWEN35_DFLASH_FORWARD_CUH
