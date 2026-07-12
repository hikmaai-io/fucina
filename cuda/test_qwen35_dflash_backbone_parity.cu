// ABOUTME: P3 full 6-layer DFlash draft backbone forward parity on real weights (device vs host).
// ABOUTME: All 6 real layers + final norm; validates cross-layer residual propagation + binding.
//
// Extends the single-layer parity to the WHOLE draft backbone: the (1+K) query rows flow through
// all 6 real DFlash decoder layers (each: input_norm -> QKV -> q/k-norm -> RoPE -> non-causal GQA
// -> o_proj -> res -> post-norm -> silu-GLU MLP -> res), then the final norm.weight. This validates
// cross-layer residual propagation and per-layer weight binding across the entire stack on the real
// z-lab/Qwen3.5-9B-DFlash weights, device fp32 vs a host double reference. The shared embedding /
// LM head GEMMs are plain matmuls already covered by k_matmul parity; this gate is the backbone.
// SKIPs cleanly when the checkpoint is absent.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_backbone_parity.cu -o t && ./t
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

static const int H=4096, HD=128, NQ=32, NKV=8, I=12288, qd=NQ*HD, kvd=NKV*HD, L=6, rows=7;
static const float EPS=1e-6f; static const double THETA=1e7;

int main(int argc,char**argv){
    const char* def="/opt/spark/models/models--z-lab--Qwen3.5-9B-DFlash/"
                    "snapshots/5fc3b3d474760f18c516db87d84c37edbfd3ede6";
    std::string dir=(argc>1)?argv[1]:def;
    if(!exists(dir+"/model.safetensors")){ printf("SKIP — real DFlash checkpoint absent\n"); return 0; }
    std::string err; st::Model m; if(!m.open((dir+"/model.safetensors").c_str(),err)){ printf("FAIL open: %s\n",err.c_str()); return 1; }
    auto WL=[&](int l,const char* s){ auto t=m.find("layers."+std::to_string(l)+"."+s); if(!t){ printf("missing L%d %s\n",l,s); exit(1);} return to_f32(t); };
    auto WG=[&](const char* s){ auto t=m.find(s); if(!t){ printf("missing %s\n",s); exit(1);} return to_f32(t); };

    std::vector<float> X((size_t)rows*H); for(int r=0;r<rows;r++) for(int i=0;i<H;i++) X[(size_t)r*H+i]=0.03f*std::sin(0.002f*i+0.4f*r)+0.005f*(r+1);
    std::vector<int> pos(rows); for(int r=0;r<rows;r++) pos[r]=100+r;
    std::vector<float> finalnorm=WG("norm.weight");

    // ---------- host double reference (all 6 layers + final norm) ----------
    auto rmsn=[&](const std::vector<double>& v,const std::vector<float>& w,int n){ double ss=0; for(int i=0;i<n;i++) ss+=v[i]*v[i]; double inv=1.0/std::sqrt(ss/n+EPS); std::vector<double> o(n); for(int i=0;i<n;i++) o[i]=v[i]*inv*(double)w[i]; return o; };
    std::vector<double> xref((size_t)rows*H); for(size_t i=0;i<xref.size();i++) xref[i]=X[i];
    for(int l=0;l<L;l++){
        auto inln=WL(l,"input_layernorm.weight"),Wq=WL(l,"self_attn.q_proj.weight"),Wk=WL(l,"self_attn.k_proj.weight"),
             Wv=WL(l,"self_attn.v_proj.weight"),Wo=WL(l,"self_attn.o_proj.weight"),qn=WL(l,"self_attn.q_norm.weight"),
             kn=WL(l,"self_attn.k_norm.weight"),poln=WL(l,"post_attention_layernorm.weight"),
             Wg=WL(l,"mlp.gate_proj.weight"),Wu=WL(l,"mlp.up_proj.weight"),Wd=WL(l,"mlp.down_proj.weight");
        std::vector<double> Qh((size_t)rows*qd),Kh((size_t)rows*kvd),Vh((size_t)rows*kvd);
        for(int r=0;r<rows;r++){
            std::vector<double> xr(H); for(int i=0;i<H;i++) xr[i]=xref[(size_t)r*H+i]; auto h1=rmsn(xr,inln,H);
            for(int o=0;o<qd;o++){ double a=0; for(int i=0;i<H;i++) a+=h1[i]*(double)Wq[(size_t)o*H+i]; Qh[(size_t)r*qd+o]=a; }
            for(int o=0;o<kvd;o++){ double a=0,b=0; for(int i=0;i<H;i++){ a+=h1[i]*(double)Wk[(size_t)o*H+i]; b+=h1[i]*(double)Wv[(size_t)o*H+i]; } Kh[(size_t)r*kvd+o]=a; Vh[(size_t)r*kvd+o]=b; }
            for(int hh=0;hh<NQ;hh++){ double ss=0; for(int i=0;i<HD;i++){ double x=Qh[(size_t)r*qd+hh*HD+i]; ss+=x*x; } double inv=1.0/std::sqrt(ss/HD+EPS); std::vector<double> t(HD); for(int i=0;i<HD;i++) t[i]=Qh[(size_t)r*qd+hh*HD+i]*inv*(double)qn[i]; int half=HD/2; for(int i=0;i<half;i++){ double f=std::pow(THETA,-2.0*i/(double)HD),a=pos[r]*f,c=std::cos(a),s=std::sin(a),x=t[i],y=t[i+half]; Qh[(size_t)r*qd+hh*HD+i]=x*c-y*s; Qh[(size_t)r*qd+hh*HD+i+half]=x*s+y*c; } }
            for(int hh=0;hh<NKV;hh++){ double ss=0; for(int i=0;i<HD;i++){ double x=Kh[(size_t)r*kvd+hh*HD+i]; ss+=x*x; } double inv=1.0/std::sqrt(ss/HD+EPS); std::vector<double> t(HD); for(int i=0;i<HD;i++) t[i]=Kh[(size_t)r*kvd+hh*HD+i]*inv*(double)kn[i]; int half=HD/2; for(int i=0;i<half;i++){ double f=std::pow(THETA,-2.0*i/(double)HD),a=pos[r]*f,c=std::cos(a),s=std::sin(a),x=t[i],y=t[i+half]; Kh[(size_t)r*kvd+hh*HD+i]=x*c-y*s; Kh[(size_t)r*kvd+hh*HD+i+half]=x*s+y*c; } }
        }
        double scale=1.0/std::sqrt((double)HD);
        for(int r=0;r<rows;r++){
            std::vector<double> attn(qd,0);
            for(int hh=0;hh<NQ;hh++){ int g=hh/(NQ/NKV); std::vector<double> sc(rows); for(int t=0;t<rows;t++){ double d=0; for(int i=0;i<HD;i++) d+=Qh[(size_t)r*qd+hh*HD+i]*Kh[(size_t)t*kvd+g*HD+i]; sc[t]=d*scale; } double mmax=-1e300; for(double x:sc) if(x>mmax) mmax=x; double s=0; for(double&x:sc){ x=std::exp(x-mmax); s+=x; } for(int i=0;i<HD;i++){ double a=0; for(int t=0;t<rows;t++) a+=sc[t]*Vh[(size_t)t*kvd+g*HD+i]; attn[hh*HD+i]=a/s; } }
            for(int o=0;o<H;o++){ double a=0; for(int i=0;i<qd;i++) a+=attn[i]*(double)Wo[(size_t)o*qd+i]; xref[(size_t)r*H+o]+=a; }
            std::vector<double> xr(H); for(int i=0;i<H;i++) xr[i]=xref[(size_t)r*H+i]; auto h2=rmsn(xr,poln,H);
            std::vector<double> ff(I); for(int o=0;o<I;o++){ double gg=0,uu=0; for(int i=0;i<H;i++){ gg+=h2[i]*(double)Wg[(size_t)o*H+i]; uu+=h2[i]*(double)Wu[(size_t)o*H+i]; } ff[o]=(gg/(1.0+std::exp(-gg)))*uu; }
            for(int o=0;o<H;o++){ double a=0; for(int i=0;i<I;i++) a+=ff[i]*(double)Wd[(size_t)o*I+i]; xref[(size_t)r*H+o]+=a; }
        }
    }
    for(int r=0;r<rows;r++){ std::vector<double> xr(H); for(int i=0;i<H;i++) xr[i]=xref[(size_t)r*H+i]; auto o=rmsn(xr,finalnorm,H); for(int i=0;i<H;i++) xref[(size_t)r*H+i]=o[i]; }

    // ---------- device path ----------
    auto up=[&](const std::vector<float>& v){ float* d; cudaMalloc(&d,v.size()*4); cudaMemcpy(d,v.data(),v.size()*4,cudaMemcpyHostToDevice); return d; };
    float* dX=up(X); int* dPos; cudaMalloc(&dPos,rows*sizeof(int)); cudaMemcpy(dPos,pos.data(),rows*sizeof(int),cudaMemcpyHostToDevice);
    float *dH1,*dQ,*dK,*dV,*dAttn,*dMix,*dH2,*dG,*dU,*dFF;
    CUDA_OK(cudaMalloc(&dH1,(size_t)rows*H*4)); CUDA_OK(cudaMalloc(&dQ,(size_t)rows*qd*4)); CUDA_OK(cudaMalloc(&dK,(size_t)rows*kvd*4));
    CUDA_OK(cudaMalloc(&dV,(size_t)rows*kvd*4)); CUDA_OK(cudaMalloc(&dAttn,(size_t)rows*qd*4)); CUDA_OK(cudaMalloc(&dMix,(size_t)rows*H*4));
    CUDA_OK(cudaMalloc(&dH2,(size_t)rows*H*4)); CUDA_OK(cudaMalloc(&dG,(size_t)rows*I*4)); CUDA_OK(cudaMalloc(&dU,(size_t)rows*I*4)); CUDA_OK(cudaMalloc(&dFF,(size_t)rows*I*4));
    auto gr=[&](int n){ return (unsigned)((n+255)/256); };
    for(int l=0;l<L;l++){
        float* dInln=up(WL(l,"input_layernorm.weight")); float* dWq=up(WL(l,"self_attn.q_proj.weight")); float* dWk=up(WL(l,"self_attn.k_proj.weight"));
        float* dWv=up(WL(l,"self_attn.v_proj.weight")); float* dWo=up(WL(l,"self_attn.o_proj.weight")); float* dQn=up(WL(l,"self_attn.q_norm.weight"));
        float* dKn=up(WL(l,"self_attn.k_norm.weight")); float* dPoln=up(WL(l,"post_attention_layernorm.weight"));
        float* dWg=up(WL(l,"mlp.gate_proj.weight")); float* dWu=up(WL(l,"mlp.up_proj.weight")); float* dWd=up(WL(l,"mlp.down_proj.weight"));
        k_rmsnorm<<<rows,256>>>(dX,dInln,dH1,H,rows,EPS);
        k_matmul<<<dim3(rows,(qd+255)/256),256>>>(dH1,dWq,dQ,rows,H,qd);
        k_matmul<<<dim3(rows,(kvd+255)/256),256>>>(dH1,dWk,dK,rows,H,kvd);
        k_matmul<<<dim3(rows,(kvd+255)/256),256>>>(dH1,dWv,dV,rows,H,kvd);
        k_headnorm<<<dim3(rows,NQ),128>>>(dQ,dQn,rows,NQ,HD,EPS);
        k_headnorm<<<dim3(rows,NKV),128>>>(dK,dKn,rows,NKV,HD,EPS);
        k_rope<<<dim3(rows,NQ),64>>>(dQ,dPos,rows,NQ,HD,THETA);
        k_rope<<<dim3(rows,NKV),64>>>(dK,dPos,rows,NKV,HD,THETA);
        k_attn<<<dim3(rows,NQ),128,(size_t)rows*sizeof(double)>>>(dQ,dK,dV,dAttn,rows,rows,NQ,NKV,HD);
        k_matmul<<<dim3(rows,(H+255)/256),256>>>(dAttn,dWo,dMix,rows,qd,H);
        k_residual<<<gr(rows*H),256>>>(dX,dMix,rows*H);
        k_rmsnorm<<<rows,256>>>(dX,dPoln,dH2,H,rows,EPS);
        k_matmul<<<dim3(rows,(I+255)/256),256>>>(dH2,dWg,dG,rows,H,I);
        k_matmul<<<dim3(rows,(I+255)/256),256>>>(dH2,dWu,dU,rows,H,I);
        k_siluglu<<<gr(rows*I),256>>>(dG,dU,dFF,rows*I);
        k_matmul<<<dim3(rows,(H+255)/256),256>>>(dFF,dWd,dMix,rows,I,H);
        k_residual<<<gr(rows*H),256>>>(dX,dMix,rows*H);
        CUDA_OK(cudaDeviceSynchronize());
        cudaFree(dInln);cudaFree(dWq);cudaFree(dWk);cudaFree(dWv);cudaFree(dWo);cudaFree(dQn);cudaFree(dKn);cudaFree(dPoln);cudaFree(dWg);cudaFree(dWu);cudaFree(dWd);
    }
    float* dFinal=up(finalnorm); float* dOut; CUDA_OK(cudaMalloc(&dOut,(size_t)rows*H*4));
    k_rmsnorm<<<rows,256>>>(dX,dFinal,dOut,H,rows,EPS); CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> out((size_t)rows*H); CUDA_OK(cudaMemcpy(out.data(),dOut,(size_t)rows*H*4,cudaMemcpyDeviceToHost));

    double max_rel=0, oss=0; for(double v:xref) oss+=v*v; double oscale=std::sqrt(oss/xref.size())+1e-12;
    for(size_t i=0;i<xref.size();i++){ double rel=std::fabs((double)out[i]-xref[i])/oscale; if(rel>max_rel) max_rel=rel; }
    printf("full 6-layer draft backbone parity on real weights: max signal-rel err=%.3e (L=%d rows=%d)\n",max_rel,L,rows);
    const double TOL=1e-3;
    if(max_rel>TOL){ printf("FAIL — backbone parity exceeds %.1e\n",TOL); return 1; }
    printf("PASS — DFlash full 6-layer backbone forward matches host double reference on real weights within %.1e\n",TOL);
    return 0;
}
