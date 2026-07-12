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

#endif // FUCINA_QWEN35_DFLASH_FORWARD_CUH
