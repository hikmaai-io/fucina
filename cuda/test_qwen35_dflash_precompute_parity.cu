// ABOUTME: P3 context-KV precompute parity on the REAL DFlash weights (device fp32 vs host double).
// ABOUTME: Validates fused KV projection + grouped K-norm — qwen3_dflash _project/_normalize_context.
//
// The DFlash draft cross-projects the TARGET hidden states into per-layer K/V. This gate validates
// the projection+norm stages of that precompute on the real z-lab/Qwen3.5-9B-DFlash weights:
//   normed = rmsnorm(X, hidden_norm, eps)          [num_ctx, H]
//   for each layer l:  K_l = normed @ Wk_l^T,  V_l = normed @ Wv_l^T   [num_ctx, kv_dim]
//   K_l per-head RMSNorm with k_norm_l over each HD-vector               (grouped K-norm)
// The device path (fp32 kernels reading the bf16 weights) must match a host double-precision
// reference reading the SAME weights, within a bf16-weight tolerance. RoPE + cache insert are the
// next stage (separate gate). SKIPs cleanly when the checkpoint is absent.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_precompute_parity.cu -o t && ./t
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

// bf16 (stored as uint16) -> float
static inline float bf16f(uint16_t u){ uint32_t x=(uint32_t)u<<16; float f; memcpy(&f,&x,4); return f; }

// Convert a bf16 safetensors tensor to a host float vector.
static std::vector<float> to_f32(const st::Tensor* t){
    size_t n = t->nbytes/2;
    const uint16_t* p=(const uint16_t*)t->data;
    std::vector<float> out(n);
    for(size_t i=0;i<n;i++) out[i]=bf16f(p[i]);
    return out;
}

// ── device kernels ──
__global__ void k_rmsnorm(const float* x, const float* w, float* y, int H, int rows, float eps){
    int r=blockIdx.x; if(r>=rows) return;
    const float* xr=x+(size_t)r*H; float* yr=y+(size_t)r*H;
    __shared__ double ss;
    double local=0; for(int i=threadIdx.x;i<H;i+=blockDim.x){ double v=xr[i]; local+=v*v; }
    // block reduce
    __shared__ double buf[256]; buf[threadIdx.x]=local; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) buf[threadIdx.x]+=buf[threadIdx.x+s]; __syncthreads(); }
    if(threadIdx.x==0) ss=buf[0]; __syncthreads();
    double inv=1.0/sqrt(ss/H+eps);
    for(int i=threadIdx.x;i<H;i+=blockDim.x) yr[i]=(float)((double)xr[i]*inv*(double)w[i]);
}
// K = normed @ W^T ; W is [out, H] row-major. grid (rows, out/256)
__global__ void k_proj(const float* normed, const float* W, float* out, int rows, int H, int outd){
    int r=blockIdx.x; int o=blockIdx.y*blockDim.x+threadIdx.x; if(r>=rows||o>=outd) return;
    const float* xr=normed+(size_t)r*H; const float* wr=W+(size_t)o*H;
    double acc=0; for(int i=0;i<H;i++) acc+=(double)xr[i]*(double)wr[i];
    out[(size_t)r*outd+o]=(float)acc;
}
// per-head RMSNorm over each HD-slice of K [rows, NKV*HD], weight k_norm[HD].
__global__ void k_headnorm(float* K, const float* w, int rows, int NKV, int HD, float eps){
    int r=blockIdx.x; int h=blockIdx.y; if(r>=rows||h>=NKV) return;
    float* v=K+((size_t)r*NKV+h)*HD;
    __shared__ double buf[256]; double local=0;
    for(int i=threadIdx.x;i<HD;i+=blockDim.x){ double x=v[i]; local+=x*x; }
    buf[threadIdx.x]=local; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) buf[threadIdx.x]+=buf[threadIdx.x+s]; __syncthreads(); }
    __shared__ double ss; if(threadIdx.x==0) ss=buf[0]; __syncthreads();
    double inv=1.0/sqrt(ss/HD+eps);
    for(int i=threadIdx.x;i<HD;i+=blockDim.x) v[i]=(float)((double)v[i]*inv*(double)w[i]);
}
// Neox RoPE on each K head-vector [rows, NKV*HD], per-row absolute position pos[r], base theta.
// Neox pairs dim i with i+HD/2: (x_i, x_j) -> (x_i c - x_j s, x_i s + x_j c), angle = pos*theta^(-2i/HD).
__global__ void k_rope(float* K, const int* pos, int rows, int NKV, int HD, double theta){
    int r=blockIdx.x; int h=blockIdx.y; if(r>=rows||h>=NKV) return;
    float* v=K+((size_t)r*NKV+h)*HD; int half=HD/2; double p=(double)pos[r];
    for(int i=threadIdx.x;i<half;i+=blockDim.x){
        double freq=pow(theta, -2.0*(double)i/(double)HD);
        double ang=p*freq, c=cos(ang), s=sin(ang);
        double x=v[i], y=v[i+half];
        v[i]     =(float)(x*c - y*s);
        v[i+half]=(float)(x*s + y*c);
    }
}

int main(int argc,char**argv){
    const char* def="/opt/spark/models/models--z-lab--Qwen3.5-9B-DFlash/"
                    "snapshots/5fc3b3d474760f18c516db87d84c37edbfd3ede6";
    std::string dir=(argc>1)?argv[1]:def;
    if(!exists(dir+"/model.safetensors")){ printf("SKIP — real DFlash checkpoint absent at %s\n",dir.c_str()); return 0; }
    std::string err; st::Model m;
    if(!m.open((dir+"/model.safetensors").c_str(),err)){ printf("FAIL open: %s\n",err.c_str()); return 1; }

    const int H=4096, HD=128, NKV=8, kv=NKV*HD, L=6, num_ctx=4;
    const float eps=1e-6f;

    // Synthetic but deterministic context hidden states X [num_ctx, H].
    std::vector<float> X((size_t)num_ctx*H);
    for(int r=0;r<num_ctx;r++) for(int i=0;i<H;i++) X[(size_t)r*H+i]=0.05f*std::sin(0.001f*i + 0.3f*r) + 0.01f*(r+1);

    auto hn_t=m.find("hidden_norm.weight");
    if(!hn_t){ printf("FAIL: no hidden_norm\n"); return 1; }
    std::vector<float> hn=to_f32(hn_t);

    // Device buffers reused across layers.
    const double theta=10000000.0;
    float *dX,*dHN,*dNormed,*dW,*dK,*dV,*dKN; int *dPos;
    std::vector<int> ctxpos(num_ctx); for(int r=0;r<num_ctx;r++) ctxpos[r]=17+r;  // arbitrary abs positions
    CUDA_OK(cudaMalloc(&dX,(size_t)num_ctx*H*4));
    CUDA_OK(cudaMalloc(&dHN,(size_t)H*4));
    CUDA_OK(cudaMalloc(&dNormed,(size_t)num_ctx*H*4));
    CUDA_OK(cudaMalloc(&dW,(size_t)kv*H*4));
    CUDA_OK(cudaMalloc(&dK,(size_t)num_ctx*kv*4));
    CUDA_OK(cudaMalloc(&dV,(size_t)num_ctx*kv*4));
    CUDA_OK(cudaMalloc(&dKN,(size_t)HD*4));
    CUDA_OK(cudaMalloc(&dPos,(size_t)num_ctx*sizeof(int)));
    CUDA_OK(cudaMemcpy(dPos,ctxpos.data(),(size_t)num_ctx*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dX,X.data(),(size_t)num_ctx*H*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dHN,hn.data(),(size_t)H*4,cudaMemcpyHostToDevice));

    // Host double reference: normed rows.
    std::vector<double> normed((size_t)num_ctx*H);
    for(int r=0;r<num_ctx;r++){
        double ss=0; for(int i=0;i<H;i++){ double v=X[(size_t)r*H+i]; ss+=v*v; }
        double inv=1.0/std::sqrt(ss/H+eps);
        for(int i=0;i<H;i++) normed[(size_t)r*H+i]=(double)X[(size_t)r*H+i]*inv*(double)hn[i];
    }
    // Device normed.
    k_rmsnorm<<<num_ctx,256>>>(dX,dHN,dNormed,H,num_ctx,eps);
    CUDA_OK(cudaDeviceSynchronize());

    // Signal-relative error: max |device-ref| over the RMS scale of the reference vector. Pointwise
    // relative error is meaningless for post-norm elements near zero; the RMS scale is the honest
    // denominator for a normalized vector. (V uses the same scale for consistency.)
    double max_rel_k=0, max_rel_v=0;
    for(int l=0;l<L;l++){
        auto kt=m.find("layers."+std::to_string(l)+".self_attn.k_proj.weight");
        auto vt=m.find("layers."+std::to_string(l)+".self_attn.v_proj.weight");
        auto knt=m.find("layers."+std::to_string(l)+".self_attn.k_norm.weight");
        if(!kt||!vt||!knt){ printf("FAIL: missing layer %d tensors\n",l); return 1; }
        std::vector<float> Wk=to_f32(kt), Wv=to_f32(vt), kn=to_f32(knt);

        // ---- Device K/V projection + K head-norm ----
        CUDA_OK(cudaMemcpy(dW,Wk.data(),(size_t)kv*H*4,cudaMemcpyHostToDevice));
        dim3 g(num_ctx,(kv+255)/256); k_proj<<<g,256>>>(dNormed,dW,dK,num_ctx,H,kv);
        CUDA_OK(cudaMemcpy(dW,Wv.data(),(size_t)kv*H*4,cudaMemcpyHostToDevice));
        k_proj<<<g,256>>>(dNormed,dW,dV,num_ctx,H,kv);
        CUDA_OK(cudaMemcpy(dKN,kn.data(),(size_t)HD*4,cudaMemcpyHostToDevice));
        dim3 gh(num_ctx,NKV); k_headnorm<<<gh,128>>>(dK,dKN,num_ctx,NKV,HD,eps);
        // RoPE stage on K (the last precompute step before cache insert).
        k_rope<<<gh,64>>>(dK,dPos,num_ctx,NKV,HD,theta);
        CUDA_OK(cudaDeviceSynchronize());
        std::vector<float> Kd((size_t)num_ctx*kv), Vd((size_t)num_ctx*kv);
        CUDA_OK(cudaMemcpy(Kd.data(),dK,(size_t)num_ctx*kv*4,cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(Vd.data(),dV,(size_t)num_ctx*kv*4,cudaMemcpyDeviceToHost));

        // ---- Host double reference ----
        for(int r=0;r<num_ctx;r++){
            // V (no norm): signal scale = RMS of the reference kv-vector for this row.
            {
                std::vector<double> Vh(kv); double ss=0;
                for(int o=0;o<kv;o++){ double acc=0; for(int i=0;i<H;i++) acc+=normed[(size_t)r*H+i]*(double)Wv[(size_t)o*H+i]; Vh[o]=acc; ss+=acc*acc; }
                double scale=std::sqrt(ss/kv)+1e-12;
                for(int o=0;o<kv;o++){ double rel=std::fabs(Vd[(size_t)r*kv+o]-Vh[o])/scale; if(rel>max_rel_v) max_rel_v=rel; }
            }
            // K then per-head norm; signal scale = RMS of each normed head vector (~1 by construction).
            std::vector<double> Kh(kv);
            for(int o=0;o<kv;o++){ double acc=0; for(int i=0;i<H;i++) acc+=normed[(size_t)r*H+i]*(double)Wk[(size_t)o*H+i]; Kh[o]=acc; }
            for(int h=0;h<NKV;h++){
                double ss=0; for(int i=0;i<HD;i++){ double x=Kh[h*HD+i]; ss+=x*x; }
                double inv=1.0/std::sqrt(ss/HD+eps);
                std::vector<double> nrm(HD);
                for(int i=0;i<HD;i++) nrm[i]=Kh[h*HD+i]*inv*(double)kn[i];
                // Neox RoPE reference on the normed head vector.
                int half=HD/2; double p=(double)ctxpos[r]; std::vector<double> ref(HD); double rss=0;
                for(int i=0;i<half;i++){
                    double freq=std::pow(theta,-2.0*(double)i/(double)HD);
                    double ang=p*freq, c=std::cos(ang), s=std::sin(ang);
                    double x=nrm[i], y=nrm[i+half];
                    ref[i]=x*c-y*s; ref[i+half]=x*s+y*c;
                }
                for(int i=0;i<HD;i++) rss+=ref[i]*ref[i];
                double scale=std::sqrt(rss/HD)+1e-12;
                for(int i=0;i<HD;i++){
                    double dk=Kd[(size_t)r*kv+h*HD+i];
                    double rel=std::fabs(dk-ref[i])/scale; if(rel>max_rel_k) max_rel_k=rel;
                }
            }
        }
    }
    cudaFree(dX);cudaFree(dHN);cudaFree(dNormed);cudaFree(dW);cudaFree(dK);cudaFree(dV);cudaFree(dKN);cudaFree(dPos);

    printf("precompute parity on real weights: max_rel K=%.3e V=%.3e (L=%d num_ctx=%d)\n",
           max_rel_k, max_rel_v, L, num_ctx);
    // fp32 device vs double host over 4096-length dot products of bf16 weights: fp32 rounding
    // dominates; a few 1e-4 relative is expected. Fail only on a real divergence.
    const double TOL=2e-3;
    if(max_rel_k>TOL || max_rel_v>TOL){ printf("FAIL — precompute parity exceeds tol %.1e\n",TOL); return 1; }
    printf("PASS — DFlash context-KV precompute (RMSNorm + fused KV proj + grouped K-norm + neox "
           "RoPE) matches host double reference on real weights within %.1e\n",TOL);
    return 0;
}
