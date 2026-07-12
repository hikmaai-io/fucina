// ABOUTME: Gate for the callable DFlash context-KV precompute over residency (real weights).
// ABOUTME: Cross-projects target hidden -> per-layer context K/V vs a host double reference.
//
// Validates q35_dflash_precompute_context_kv reading resident BF16 views on the real z-lab weights:
// for every draft layer, context K (normed + neox-RoPE'd) and V match a host double reference of
// hidden RMSNorm -> K/V projection -> grouped K-norm -> RoPE. This is the DFlash cross-attention
// trick's producer. SKIPs cleanly when the checkpoint is absent.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_ctxkv.cu -o t && ./t
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include "qwen35_dflash_forward.cuh"

#define CUDA_OK(c) do{ cudaError_t e=(c); if(e!=cudaSuccess){ printf("CUDA %s @ %d\n", cudaGetErrorString(e), __LINE__); return 2; } }while(0)
static bool exists(const std::string& p){ struct stat s; return stat(p.c_str(),&s)==0; }
static inline float bf16f(uint16_t u){ uint32_t x=(uint32_t)u<<16; float f; memcpy(&f,&x,4); return f; }
static std::vector<float> to_f32(const st::Tensor* t){ size_t n=t->nbytes/2; const uint16_t* p=(const uint16_t*)t->data; std::vector<float> o(n); for(size_t i=0;i<n;i++) o[i]=bf16f(p[i]); return o; }

int main(int argc,char**argv){
    const char* def="/opt/spark/models/models--z-lab--Qwen3.5-9B-DFlash/"
                    "snapshots/5fc3b3d474760f18c516db87d84c37edbfd3ede6";
    std::string dir=(argc>1)?argv[1]:def;
    if(!exists(dir+"/model.safetensors")){ printf("SKIP — real DFlash checkpoint absent\n"); return 0; }
    std::ifstream f(dir+"/config.json"); std::stringstream ss; ss<<f.rdbuf(); std::string cj=ss.str();
    q35_dflash_residency R{}; std::string err;
    if(!qwen35dflash::parse_config(cj,R.geom,err)){ printf("FAIL parse: %s\n",err.c_str()); return 1; }
    st::Model M; if(!M.open((dir+"/model.safetensors").c_str(),err)){ printf("FAIL open: %s\n",err.c_str()); return 1; }
    if(!qwen35dflash::validate_tensors(M,R.geom,R.geom.V,err)){ printf("FAIL validate: %s\n",err.c_str()); return 1; }
    if(q35_dflash_residency_upload(&R,M,err)!=0){ printf("FAIL upload: %s\n",err.c_str()); return 1; }

    const auto& g=R.geom; const int H=g.H,HD=g.HD,NKV=g.NKV,kvd=g.kv_dim(),num_ctx=5,L=g.L;
    const double theta=1e7; const float eps=1e-6f;

    std::vector<float> Xc((size_t)num_ctx*H); for(int r=0;r<num_ctx;r++) for(int i=0;i<H;i++) Xc[(size_t)r*H+i]=0.04f*std::sin(0.0015f*i+0.5f*r)+0.008f*(r+1);
    std::vector<int> pos(num_ctx); for(int r=0;r<num_ctx;r++) pos[r]=30+r;

    // device
    float* dX; CUDA_OK(cudaMalloc(&dX,(size_t)num_ctx*H*4)); CUDA_OK(cudaMemcpy(dX,Xc.data(),(size_t)num_ctx*H*4,cudaMemcpyHostToDevice));
    int* dPos; CUDA_OK(cudaMalloc(&dPos,num_ctx*sizeof(int))); CUDA_OK(cudaMemcpy(dPos,pos.data(),num_ctx*sizeof(int),cudaMemcpyHostToDevice));
    std::vector<float*> dK(L),dV(L);
    for(int l=0;l<L;l++){ CUDA_OK(cudaMalloc(&dK[l],(size_t)num_ctx*kvd*4)); CUDA_OK(cudaMalloc(&dV[l],(size_t)num_ctx*kvd*4)); }
    q35_dflash_ctx_scratch c{}; if(!q35_dflash_ctx_scratch_alloc(&c,g,num_ctx)){ printf("FAIL ctx scratch\n"); return 1; }
    q35_dflash_precompute_context_kv(R,c,dX,dPos,dK.data(),dV.data(),theta,eps,0);
    CUDA_OK(cudaDeviceSynchronize());

    // host double reference
    auto hn=to_f32(M.find("hidden_norm.weight"));
    std::vector<double> normed((size_t)num_ctx*H);
    for(int r=0;r<num_ctx;r++){ double ssq=0; for(int i=0;i<H;i++){ double v=Xc[(size_t)r*H+i]; ssq+=v*v; } double inv=1.0/std::sqrt(ssq/H+eps); for(int i=0;i<H;i++) normed[(size_t)r*H+i]=(double)Xc[(size_t)r*H+i]*inv*(double)hn[i]; }

    double max_rel_k=0,max_rel_v=0;
    for(int l=0;l<L;l++){
        auto Wk=to_f32(M.find("layers."+std::to_string(l)+".self_attn.k_proj.weight"));
        auto Wv=to_f32(M.find("layers."+std::to_string(l)+".self_attn.v_proj.weight"));
        auto kn=to_f32(M.find("layers."+std::to_string(l)+".self_attn.k_norm.weight"));
        std::vector<float> Kd((size_t)num_ctx*kvd),Vd((size_t)num_ctx*kvd);
        CUDA_OK(cudaMemcpy(Kd.data(),dK[l],(size_t)num_ctx*kvd*4,cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(Vd.data(),dV[l],(size_t)num_ctx*kvd*4,cudaMemcpyDeviceToHost));
        for(int r=0;r<num_ctx;r++){
            // V
            std::vector<double> Vh(kvd); double vss=0;
            for(int o=0;o<kvd;o++){ double a=0; for(int i=0;i<H;i++) a+=normed[(size_t)r*H+i]*(double)Wv[(size_t)o*H+i]; Vh[o]=a; vss+=a*a; }
            double vscale=std::sqrt(vss/kvd)+1e-12;
            for(int o=0;o<kvd;o++){ double rel=std::fabs(Vd[(size_t)r*kvd+o]-Vh[o])/vscale; if(rel>max_rel_v) max_rel_v=rel; }
            // K + head-norm + rope
            std::vector<double> Kh(kvd); for(int o=0;o<kvd;o++){ double a=0; for(int i=0;i<H;i++) a+=normed[(size_t)r*H+i]*(double)Wk[(size_t)o*H+i]; Kh[o]=a; }
            for(int h=0;h<NKV;h++){ double ssk=0; for(int i=0;i<HD;i++){ double x=Kh[h*HD+i]; ssk+=x*x; } double inv=1.0/std::sqrt(ssk/HD+eps);
                std::vector<double> t(HD); for(int i=0;i<HD;i++) t[i]=Kh[h*HD+i]*inv*(double)kn[i];
                int half=HD/2; std::vector<double> ref(HD); double rss=0;
                for(int i=0;i<half;i++){ double ff=std::pow(theta,-2.0*i/(double)HD),a=pos[r]*ff,cc=std::cos(a),sinv=std::sin(a),x=t[i],y=t[i+half]; ref[i]=x*cc-y*sinv; ref[i+half]=x*sinv+y*cc; }
                for(int i=0;i<HD;i++) rss+=ref[i]*ref[i]; double kscale=std::sqrt(rss/HD)+1e-12;
                for(int i=0;i<HD;i++){ double rel=std::fabs(Kd[(size_t)r*kvd+h*HD+i]-ref[i])/kscale; if(rel>max_rel_k) max_rel_k=rel; }
            }
        }
    }
    for(int l=0;l<L;l++){ cudaFree(dK[l]); cudaFree(dV[l]); }
    q35_dflash_ctx_scratch_free(&c); cudaFree(dX); cudaFree(dPos); q35_dflash_residency_free(&R);

    printf("callable context-KV precompute over residency parity: max_rel K=%.3e V=%.3e (L=%d num_ctx=%d)\n",max_rel_k,max_rel_v,L,num_ctx);
    const double TOL=2e-3;
    if(max_rel_k>TOL||max_rel_v>TOL){ printf("FAIL — ctx-KV parity exceeds %.1e\n",TOL); return 1; }
    printf("PASS — DFlash callable context-KV precompute over residency matches host double reference within %.1e\n",TOL);
    return 0;
}
