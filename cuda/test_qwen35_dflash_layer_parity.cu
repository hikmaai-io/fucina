// ABOUTME: P3 full DFlash draft-layer forward parity on real weights (device fp32 vs host double).
// ABOUTME: Composes input_norm->QKV->q/k-norm->RoPE->non-causal GQA->o_proj->res->MLP->res.
//
// This is the end-to-end composition test for one DFlash decoder layer (DFlashQwen3DecoderLayer):
//   h1 = rmsnorm(x, input_layernorm)
//   q,k,v = h1 @ {Wq,Wk,Wv}^T ; per-head q/k RMSNorm ; neox RoPE(q,k)
//   attn = non-causal GQA(q over precomputed context K/V + this layer's own query K/V)
//   x = x + attn @ Wo^T
//   h2 = rmsnorm(x, post_attention_layernorm)
//   x = x + (silu(h2@Wgate^T) * (h2@Wup^T)) @ Wdown^T
// It validates the composed device fp32 path against a host double reference on the REAL layer-0
// weights, for the (1+K) query rows attending a synthetic context. This de-risks the full-stack
// composition before it is wired into the engine. SKIPs cleanly when the checkpoint is absent.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_layer_parity.cu -o t && ./t
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include "safetensors.h"

#define CUDA_OK(c) do{ cudaError_t e=(c); if(e!=cudaSuccess){ printf("CUDA %s @ %d\n", cudaGetErrorString(e), __LINE__); return 2; } }while(0)
static bool exists(const std::string& p){ struct stat s; return stat(p.c_str(),&s)==0; }
static inline float bf16f(uint16_t u){ uint32_t x=(uint32_t)u<<16; float f; memcpy(&f,&x,4); return f; }
static std::vector<float> to_f32(const st::Tensor* t){ size_t n=t->nbytes/2; const uint16_t* p=(const uint16_t*)t->data; std::vector<float> o(n); for(size_t i=0;i<n;i++) o[i]=bf16f(p[i]); return o; }

// ---- device kernels (fp32, double accumulation) ----
__global__ void k_rmsnorm(const float* x,const float* w,float* y,int H,int rows,float eps){
    int r=blockIdx.x; if(r>=rows) return; const float* xr=x+(size_t)r*H; float* yr=y+(size_t)r*H;
    __shared__ double buf[256]; double loc=0; for(int i=threadIdx.x;i<H;i+=blockDim.x){ double v=xr[i]; loc+=v*v; }
    buf[threadIdx.x]=loc; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) buf[threadIdx.x]+=buf[threadIdx.x+s]; __syncthreads(); }
    __shared__ double ss; if(threadIdx.x==0) ss=buf[0]; __syncthreads();
    double inv=1.0/sqrt(ss/H+eps); for(int i=threadIdx.x;i<H;i+=blockDim.x) yr[i]=(float)((double)xr[i]*inv*(double)w[i]);
}
__global__ void k_matmul(const float* A,const float* W,float* O,int rows,int in,int outd){
    int r=blockIdx.x; int o=blockIdx.y*blockDim.x+threadIdx.x; if(r>=rows||o>=outd) return;
    const float* a=A+(size_t)r*in; const float* w=W+(size_t)o*in; double acc=0; for(int i=0;i<in;i++) acc+=(double)a[i]*(double)w[i];
    O[(size_t)r*outd+o]=(float)acc;
}
__global__ void k_headnorm(float* X,const float* w,int rows,int nh,int HD,float eps){
    int r=blockIdx.x; int h=blockIdx.y; if(r>=rows||h>=nh) return; float* v=X+((size_t)r*nh+h)*HD;
    __shared__ double buf[256]; double loc=0; for(int i=threadIdx.x;i<HD;i+=blockDim.x){ double x=v[i]; loc+=x*x; }
    buf[threadIdx.x]=loc; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) buf[threadIdx.x]+=buf[threadIdx.x+s]; __syncthreads(); }
    __shared__ double ss; if(threadIdx.x==0) ss=buf[0]; __syncthreads();
    double inv=1.0/sqrt(ss/HD+eps); for(int i=threadIdx.x;i<HD;i+=blockDim.x) v[i]=(float)((double)v[i]*inv*(double)w[i]);
}
__global__ void k_rope(float* X,const int* pos,int rows,int nh,int HD,double theta){
    int r=blockIdx.x; int h=blockIdx.y; if(r>=rows||h>=nh) return; float* v=X+((size_t)r*nh+h)*HD; int half=HD/2; double p=pos[r];
    for(int i=threadIdx.x;i<half;i+=blockDim.x){ double f=pow(theta,-2.0*i/(double)HD),a=p*f,c=cos(a),s=sin(a); double x=v[i],y=v[i+half]; v[i]=(float)(x*c-y*s); v[i+half]=(float)(x*s+y*c); }
}
// non-causal GQA: Q[rows,NQ,HD] attends K/V[ctx,NKV,HD]; O[rows,NQ,HD]. one block per (row,qhead).
__global__ void k_attn(const float* Q,const float* K,const float* V,float* O,int rows,int ctx,int NQ,int NKV,int HD){
    int r=blockIdx.x; int h=blockIdx.y; if(r>=rows||h>=NQ) return; int g=h/(NQ/NKV);
    const float* q=Q+((size_t)r*NQ+h)*HD; extern __shared__ double sh[]; double scale=1.0/sqrt((double)HD);
    for(int t=threadIdx.x;t<ctx;t+=blockDim.x){ const float* k=K+((size_t)t*NKV+g)*HD; double d=0; for(int i=0;i<HD;i++) d+=(double)q[i]*(double)k[i]; sh[t]=d*scale; }
    __syncthreads(); __shared__ double sm;
    if(threadIdx.x==0){ double m=-1e300; for(int t=0;t<ctx;t++) if(sh[t]>m) m=sh[t]; double s=0; for(int t=0;t<ctx;t++){ sh[t]=exp(sh[t]-m); s+=sh[t]; } sm=s; }
    __syncthreads();
    for(int i=threadIdx.x;i<HD;i+=blockDim.x){ double acc=0; for(int t=0;t<ctx;t++){ const float* v=V+((size_t)t*NKV+g)*HD; acc+=sh[t]*(double)v[i]; } O[((size_t)r*NQ+h)*HD+i]=(float)(acc/sm); }
}
__global__ void k_residual(float* x,const float* d,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=d[i]; }
__global__ void k_siluglu(const float* g,const float* u,float* o,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ double x=g[i]; o[i]=(float)((x/(1.0+exp(-x)))*(double)u[i]); } }

int main(int argc,char**argv){
    const char* def="/opt/spark/models/models--z-lab--Qwen3.5-9B-DFlash/"
                    "snapshots/5fc3b3d474760f18c516db87d84c37edbfd3ede6";
    std::string dir=(argc>1)?argv[1]:def;
    if(!exists(dir+"/model.safetensors")){ printf("SKIP — real DFlash checkpoint absent\n"); return 0; }
    std::string err; st::Model m; if(!m.open((dir+"/model.safetensors").c_str(),err)){ printf("FAIL open: %s\n",err.c_str()); return 1; }

    const int H=4096, HD=128, NQ=32, NKV=8, I=12288, qd=NQ*HD, kvd=NKV*HD, rows=7, ctx=rows; // (1+K)=7 self-attending
    const float eps=1e-6f; const double theta=1e7;
    auto W=[&](const char* s){ auto t=m.find("layers.0."+std::string(s)); if(!t){ printf("missing %s\n",s); exit(1);} return to_f32(t); };
    std::vector<float> inln=W("input_layernorm.weight"), Wq=W("self_attn.q_proj.weight"), Wk=W("self_attn.k_proj.weight"),
        Wv=W("self_attn.v_proj.weight"), Wo=W("self_attn.o_proj.weight"), qn=W("self_attn.q_norm.weight"),
        kn=W("self_attn.k_norm.weight"), poln=W("post_attention_layernorm.weight"),
        Wg=W("mlp.gate_proj.weight"), Wu=W("mlp.up_proj.weight"), Wd=W("mlp.down_proj.weight");

    std::vector<float> X((size_t)rows*H); for(int r=0;r<rows;r++) for(int i=0;i<H;i++) X[(size_t)r*H+i]=0.03f*std::sin(0.002f*i+0.4f*r)+0.005f*(r+1);
    std::vector<int> pos(rows); for(int r=0;r<rows;r++) pos[r]=100+r;

    // ---------- host double reference ----------
    auto rmsn=[&](const std::vector<double>& v,const std::vector<float>& w,int n){ double ss=0; for(int i=0;i<n;i++) ss+=v[i]*v[i]; double inv=1.0/std::sqrt(ss/n+eps); std::vector<double> o(n); for(int i=0;i<n;i++) o[i]=v[i]*inv*(double)w[i]; return o; };
    std::vector<double> xref((size_t)rows*H); for(size_t i=0;i<xref.size();i++) xref[i]=X[i];
    std::vector<double> Qh((size_t)rows*qd), Kh((size_t)rows*kvd), Vh((size_t)rows*kvd);
    for(int r=0;r<rows;r++){
        std::vector<double> xr(H); for(int i=0;i<H;i++) xr[i]=xref[(size_t)r*H+i];
        auto h1=rmsn(xr,inln,H);
        for(int o=0;o<qd;o++){ double a=0; for(int i=0;i<H;i++) a+=h1[i]*(double)Wq[(size_t)o*H+i]; Qh[(size_t)r*qd+o]=a; }
        for(int o=0;o<kvd;o++){ double a=0,b=0; for(int i=0;i<H;i++){ a+=h1[i]*(double)Wk[(size_t)o*H+i]; b+=h1[i]*(double)Wv[(size_t)o*H+i]; } Kh[(size_t)r*kvd+o]=a; Vh[(size_t)r*kvd+o]=b; }
        // q/k head norm + rope
        for(int hh=0;hh<NQ;hh++){ double ss=0; for(int i=0;i<HD;i++){ double x=Qh[(size_t)r*qd+hh*HD+i]; ss+=x*x; } double inv=1.0/std::sqrt(ss/HD+eps);
            std::vector<double> t(HD); for(int i=0;i<HD;i++) t[i]=Qh[(size_t)r*qd+hh*HD+i]*inv*(double)qn[i];
            int half=HD/2; for(int i=0;i<half;i++){ double f=std::pow(theta,-2.0*i/(double)HD),a=pos[r]*f,c=std::cos(a),s=std::sin(a),x=t[i],y=t[i+half]; Qh[(size_t)r*qd+hh*HD+i]=x*c-y*s; Qh[(size_t)r*qd+hh*HD+i+half]=x*s+y*c; } }
        for(int hh=0;hh<NKV;hh++){ double ss=0; for(int i=0;i<HD;i++){ double x=Kh[(size_t)r*kvd+hh*HD+i]; ss+=x*x; } double inv=1.0/std::sqrt(ss/HD+eps);
            std::vector<double> t(HD); for(int i=0;i<HD;i++) t[i]=Kh[(size_t)r*kvd+hh*HD+i]*inv*(double)kn[i];
            int half=HD/2; for(int i=0;i<half;i++){ double f=std::pow(theta,-2.0*i/(double)HD),a=pos[r]*f,c=std::cos(a),s=std::sin(a),x=t[i],y=t[i+half]; Kh[(size_t)r*kvd+hh*HD+i]=x*c-y*s; Kh[(size_t)r*kvd+hh*HD+i+half]=x*s+y*c; } }
    }
    // non-causal GQA + o_proj + residual + MLP
    double scale=1.0/std::sqrt((double)HD);
    for(int r=0;r<rows;r++){
        std::vector<double> attn(qd,0);
        for(int hh=0;hh<NQ;hh++){ int g=hh/(NQ/NKV); std::vector<double> sc(ctx);
            for(int t=0;t<ctx;t++){ double d=0; for(int i=0;i<HD;i++) d+=Qh[(size_t)r*qd+hh*HD+i]*Kh[(size_t)t*kvd+g*HD+i]; sc[t]=d*scale; }
            double mmax=-1e300; for(double x:sc) if(x>mmax) mmax=x; double s=0; for(double&x:sc){ x=std::exp(x-mmax); s+=x; }
            for(int i=0;i<HD;i++){ double a=0; for(int t=0;t<ctx;t++) a+=sc[t]*Vh[(size_t)t*kvd+g*HD+i]; attn[hh*HD+i]=a/s; } }
        for(int o=0;o<H;o++){ double a=0; for(int i=0;i<qd;i++) a+=attn[i]*(double)Wo[(size_t)o*qd+i]; xref[(size_t)r*H+o]+=a; }
        std::vector<double> xr(H); for(int i=0;i<H;i++) xr[i]=xref[(size_t)r*H+i];
        auto h2=rmsn(xr,poln,H);
        std::vector<double> ff(I); for(int o=0;o<I;o++){ double gg=0,uu=0; for(int i=0;i<H;i++){ gg+=h2[i]*(double)Wg[(size_t)o*H+i]; uu+=h2[i]*(double)Wu[(size_t)o*H+i]; } ff[o]=(gg/(1.0+std::exp(-gg)))*uu; }
        for(int o=0;o<H;o++){ double a=0; for(int i=0;i<I;i++) a+=ff[i]*(double)Wd[(size_t)o*I+i]; xref[(size_t)r*H+o]+=a; }
    }

    // ---------- device path ----------
    auto up=[&](const std::vector<float>& v){ float* d; cudaMalloc(&d,v.size()*4); cudaMemcpy(d,v.data(),v.size()*4,cudaMemcpyHostToDevice); return d; };
    float *dX=up(X),*dInln=up(inln),*dWq=up(Wq),*dWk=up(Wk),*dWv=up(Wv),*dWo=up(Wo),*dQn=up(qn),*dKn=up(kn),
          *dPoln=up(poln),*dWg=up(Wg),*dWu=up(Wu),*dWd=up(Wd);
    int* dPos=(int*)up(std::vector<float>()); cudaMalloc(&dPos,rows*sizeof(int)); cudaMemcpy(dPos,pos.data(),rows*sizeof(int),cudaMemcpyHostToDevice);
    float *dH1,*dQ,*dK,*dV,*dAttn,*dMix,*dH2,*dG,*dU,*dFF;
    CUDA_OK(cudaMalloc(&dH1,(size_t)rows*H*4)); CUDA_OK(cudaMalloc(&dQ,(size_t)rows*qd*4));
    CUDA_OK(cudaMalloc(&dK,(size_t)rows*kvd*4)); CUDA_OK(cudaMalloc(&dV,(size_t)rows*kvd*4));
    CUDA_OK(cudaMalloc(&dAttn,(size_t)rows*qd*4)); CUDA_OK(cudaMalloc(&dMix,(size_t)rows*H*4));
    CUDA_OK(cudaMalloc(&dH2,(size_t)rows*H*4)); CUDA_OK(cudaMalloc(&dG,(size_t)rows*I*4));
    CUDA_OK(cudaMalloc(&dU,(size_t)rows*I*4)); CUDA_OK(cudaMalloc(&dFF,(size_t)rows*I*4));
    auto gr=[&](int n){ return (unsigned)((n+255)/256); };
    k_rmsnorm<<<rows,256>>>(dX,dInln,dH1,H,rows,eps);
    k_matmul<<<dim3(rows,(qd+255)/256),256>>>(dH1,dWq,dQ,rows,H,qd);
    k_matmul<<<dim3(rows,(kvd+255)/256),256>>>(dH1,dWk,dK,rows,H,kvd);
    k_matmul<<<dim3(rows,(kvd+255)/256),256>>>(dH1,dWv,dV,rows,H,kvd);
    k_headnorm<<<dim3(rows,NQ),128>>>(dQ,dQn,rows,NQ,HD,eps);
    k_headnorm<<<dim3(rows,NKV),128>>>(dK,dKn,rows,NKV,HD,eps);
    k_rope<<<dim3(rows,NQ),64>>>(dQ,dPos,rows,NQ,HD,theta);
    k_rope<<<dim3(rows,NKV),64>>>(dK,dPos,rows,NKV,HD,theta);
    k_attn<<<dim3(rows,NQ),128,(size_t)ctx*sizeof(double)>>>(dQ,dK,dV,dAttn,rows,ctx,NQ,NKV,HD);
    k_matmul<<<dim3(rows,(H+255)/256),256>>>(dAttn,dWo,dMix,rows,qd,H);
    k_residual<<<gr(rows*H),256>>>(dX,dMix,rows*H);
    k_rmsnorm<<<rows,256>>>(dX,dPoln,dH2,H,rows,eps);
    k_matmul<<<dim3(rows,(I+255)/256),256>>>(dH2,dWg,dG,rows,H,I);
    k_matmul<<<dim3(rows,(I+255)/256),256>>>(dH2,dWu,dU,rows,H,I);
    k_siluglu<<<gr(rows*I),256>>>(dG,dU,dFF,rows*I);
    k_matmul<<<dim3(rows,(H+255)/256),256>>>(dFF,dWd,dMix,rows,I,H);
    k_residual<<<gr(rows*H),256>>>(dX,dMix,rows*H);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> out((size_t)rows*H); CUDA_OK(cudaMemcpy(out.data(),dX,(size_t)rows*H*4,cudaMemcpyDeviceToHost));

    double max_rel=0, oss=0; for(double v:xref) oss+=v*v; double oscale=std::sqrt(oss/xref.size())+1e-12;
    for(size_t i=0;i<xref.size();i++){ double rel=std::fabs((double)out[i]-xref[i])/oscale; if(rel>max_rel) max_rel=rel; }
    printf("full draft-layer forward parity on real weights: max signal-rel err=%.3e (rows=%d)\n",max_rel,rows);
    const double TOL=5e-4;
    if(max_rel>TOL){ printf("FAIL — layer parity exceeds %.1e\n",TOL); return 1; }
    printf("PASS — DFlash full draft-layer forward matches host double reference on real weights within %.1e\n",TOL);
    return 0;
}
