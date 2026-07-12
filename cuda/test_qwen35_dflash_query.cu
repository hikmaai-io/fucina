// ABOUTME: Gate for the full DFlash query forward (context + self attention) on real weights.
// ABOUTME: precompute context K/V then (1+K) query rows attend context++own block vs host double.
//
// Validates the TRUE DFlash drafting forward end to end on the real z-lab weights: it runs the
// callable context-KV precompute, then the query forward where each of the (1+K) query rows attends
// the concatenation of the precomputed context K/V and the query block's own K/V (non-causal, GQA),
// through all 6 layers + final norm. Device fp32 (double accum) vs a host double reference computing
// the same composed attention. SKIPs cleanly when the checkpoint is absent.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_query.cu -o t && ./t
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
static std::vector<float> tf(const st::Tensor* t){ size_t n=t->nbytes/2; const uint16_t* p=(const uint16_t*)t->data; std::vector<float> o(n); for(size_t i=0;i<n;i++) o[i]=bf16f(p[i]); return o; }

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

    const auto& g=R.geom; const int H=g.H,HD=g.HD,NQ=g.NQ,NKV=g.NKV,I=g.I,qd=g.q_dim(),kvd=g.kv_dim();
    const int num_ctx=9, rows=7, L=g.L; const double theta=1e7; const float eps=1e-6f;

    // Context hidden (target states) and query embeddings + positions.
    std::vector<float> Xc((size_t)num_ctx*H); for(int r=0;r<num_ctx;r++) for(int i=0;i<H;i++) Xc[(size_t)r*H+i]=0.04f*std::sin(0.0015f*i+0.5f*r)+0.008f*(r+1);
    std::vector<int> cpos(num_ctx); for(int r=0;r<num_ctx;r++) cpos[r]=r;         // context positions 0..num_ctx-1
    std::vector<float> Xq((size_t)rows*H); for(int r=0;r<rows;r++) for(int i=0;i<H;i++) Xq[(size_t)r*H+i]=0.03f*std::sin(0.002f*i+0.4f*r)+0.005f*(r+1);
    std::vector<int> qpos(rows); for(int r=0;r<rows;r++) qpos[r]=num_ctx+r;        // query positions follow context

    // ---- device: precompute context K/V, then query forward ----
    float *dXc,*dXq; int *dCpos,*dQpos;
    CUDA_OK(cudaMalloc(&dXc,(size_t)num_ctx*H*4)); CUDA_OK(cudaMemcpy(dXc,Xc.data(),(size_t)num_ctx*H*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&dXq,(size_t)rows*H*4)); CUDA_OK(cudaMemcpy(dXq,Xq.data(),(size_t)rows*H*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&dCpos,num_ctx*sizeof(int))); CUDA_OK(cudaMemcpy(dCpos,cpos.data(),num_ctx*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&dQpos,rows*sizeof(int))); CUDA_OK(cudaMemcpy(dQpos,qpos.data(),rows*sizeof(int),cudaMemcpyHostToDevice));
    std::vector<float*> dK(L),dV(L); for(int l=0;l<L;l++){ CUDA_OK(cudaMalloc(&dK[l],(size_t)num_ctx*kvd*4)); CUDA_OK(cudaMalloc(&dV[l],(size_t)num_ctx*kvd*4)); }
    q35_dflash_ctx_scratch c{}; if(!q35_dflash_ctx_scratch_alloc(&c,g,num_ctx)){ printf("FAIL ctx scratch\n"); return 1; }
    q35_dflash_fwd_scratch s{}; if(!q35_dflash_fwd_scratch_alloc(&s,g,rows)){ printf("FAIL fwd scratch\n"); return 1; }
    q35_dflash_precompute_context_kv(R,c,dXc,dCpos,dK.data(),dV.data(),theta,eps,0);
    q35_dflash_query_forward(R,s,dXq,dQpos,num_ctx,dK.data(),dV.data(),theta,eps,0);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> out((size_t)rows*H); CUDA_OK(cudaMemcpy(out.data(),s.out,(size_t)rows*H*4,cudaMemcpyDeviceToHost));

    // ---- host double reference ----
    auto WG=[&](const char* n){ return tf(M.find(n)); };
    auto WL=[&](int l,const char* sfx){ return tf(M.find("layers."+std::to_string(l)+"."+sfx)); };
    auto rmsn=[&](const std::vector<double>& v,const std::vector<float>& w,int n){ double sq=0; for(int i=0;i<n;i++) sq+=v[i]*v[i]; double inv=1.0/std::sqrt(sq/n+eps); std::vector<double> o(n); for(int i=0;i<n;i++) o[i]=v[i]*inv*(double)w[i]; return o; };
    auto ropev=[&](std::vector<double>& t,int p){ int half=HD/2; for(int i=0;i<half;i++){ double ff=std::pow(theta,-2.0*i/(double)HD),a=p*ff,cc=std::cos(a),sv=std::sin(a),x=t[i],y=t[i+half]; t[i]=x*cc-y*sv; t[i+half]=x*sv+y*cc; } };

    // context K/V per layer (host)
    auto hn=WG("hidden_norm.weight");
    std::vector<double> cnorm((size_t)num_ctx*H);
    for(int r=0;r<num_ctx;r++){ std::vector<double> xr(H); for(int i=0;i<H;i++) xr[i]=Xc[(size_t)r*H+i]; auto o=rmsn(xr,hn,H); for(int i=0;i<H;i++) cnorm[(size_t)r*H+i]=o[i]; }
    std::vector<std::vector<double>> Kc(L),Vc(L);
    for(int l=0;l<L;l++){
        auto Wk=WL(l,"self_attn.k_proj.weight"),Wv=WL(l,"self_attn.v_proj.weight"),kn=WL(l,"self_attn.k_norm.weight");
        Kc[l].assign((size_t)num_ctx*kvd,0); Vc[l].assign((size_t)num_ctx*kvd,0);
        for(int r=0;r<num_ctx;r++){
            for(int o=0;o<kvd;o++){ double a=0,b=0; for(int i=0;i<H;i++){ a+=cnorm[(size_t)r*H+i]*(double)Wk[(size_t)o*H+i]; b+=cnorm[(size_t)r*H+i]*(double)Wv[(size_t)o*H+i]; } Kc[l][(size_t)r*kvd+o]=a; Vc[l][(size_t)r*kvd+o]=b; }
            for(int h=0;h<NKV;h++){ double sq=0; for(int i=0;i<HD;i++){ double x=Kc[l][(size_t)r*kvd+h*HD+i]; sq+=x*x; } double inv=1.0/std::sqrt(sq/HD+eps); std::vector<double> t(HD); for(int i=0;i<HD;i++) t[i]=Kc[l][(size_t)r*kvd+h*HD+i]*inv*(double)kn[i]; ropev(t,cpos[r]); for(int i=0;i<HD;i++) Kc[l][(size_t)r*kvd+h*HD+i]=t[i]; }
        }
    }
    // query forward (host), attending context ++ own block
    std::vector<double> xq((size_t)rows*H); for(size_t i=0;i<xq.size();i++) xq[i]=Xq[i];
    double scale=1.0/std::sqrt((double)HD);
    for(int l=0;l<L;l++){
        auto inln=WL(l,"input_layernorm.weight"),Wq=WL(l,"self_attn.q_proj.weight"),Wk=WL(l,"self_attn.k_proj.weight"),Wv=WL(l,"self_attn.v_proj.weight"),
             Wo=WL(l,"self_attn.o_proj.weight"),qn=WL(l,"self_attn.q_norm.weight"),kn=WL(l,"self_attn.k_norm.weight"),poln=WL(l,"post_attention_layernorm.weight"),
             Wg=WL(l,"mlp.gate_proj.weight"),Wu=WL(l,"mlp.up_proj.weight"),Wd=WL(l,"mlp.down_proj.weight");
        std::vector<double> Qh((size_t)rows*qd),Kq((size_t)rows*kvd),Vq((size_t)rows*kvd);
        for(int r=0;r<rows;r++){
            std::vector<double> xr(H); for(int i=0;i<H;i++) xr[i]=xq[(size_t)r*H+i]; auto h1=rmsn(xr,inln,H);
            for(int o=0;o<qd;o++){ double a=0; for(int i=0;i<H;i++) a+=h1[i]*(double)Wq[(size_t)o*H+i]; Qh[(size_t)r*qd+o]=a; }
            for(int o=0;o<kvd;o++){ double a=0,b=0; for(int i=0;i<H;i++){ a+=h1[i]*(double)Wk[(size_t)o*H+i]; b+=h1[i]*(double)Wv[(size_t)o*H+i]; } Kq[(size_t)r*kvd+o]=a; Vq[(size_t)r*kvd+o]=b; }
            for(int h=0;h<NQ;h++){ double sq=0; for(int i=0;i<HD;i++){ double x=Qh[(size_t)r*qd+h*HD+i]; sq+=x*x; } double inv=1.0/std::sqrt(sq/HD+eps); std::vector<double> t(HD); for(int i=0;i<HD;i++) t[i]=Qh[(size_t)r*qd+h*HD+i]*inv*(double)qn[i]; ropev(t,qpos[r]); for(int i=0;i<HD;i++) Qh[(size_t)r*qd+h*HD+i]=t[i]; }
            for(int h=0;h<NKV;h++){ double sq=0; for(int i=0;i<HD;i++){ double x=Kq[(size_t)r*kvd+h*HD+i]; sq+=x*x; } double inv=1.0/std::sqrt(sq/HD+eps); std::vector<double> t(HD); for(int i=0;i<HD;i++) t[i]=Kq[(size_t)r*kvd+h*HD+i]*inv*(double)kn[i]; ropev(t,qpos[r]); for(int i=0;i<HD;i++) Kq[(size_t)r*kvd+h*HD+i]=t[i]; }
        }
        for(int r=0;r<rows;r++){
            std::vector<double> attn(qd,0);
            for(int h=0;h<NQ;h++){ int gg=h/(NQ/NKV); int tot=num_ctx+rows; std::vector<double> sc(tot);
                for(int t=0;t<num_ctx;t++){ double d=0; for(int i=0;i<HD;i++) d+=Qh[(size_t)r*qd+h*HD+i]*Kc[l][(size_t)t*kvd+gg*HD+i]; sc[t]=d*scale; }
                for(int t=0;t<rows;t++){ double d=0; for(int i=0;i<HD;i++) d+=Qh[(size_t)r*qd+h*HD+i]*Kq[(size_t)t*kvd+gg*HD+i]; sc[num_ctx+t]=d*scale; }
                double mmax=-1e300; for(double x:sc) if(x>mmax) mmax=x; double sm=0; for(double&x:sc){ x=std::exp(x-mmax); sm+=x; }
                for(int i=0;i<HD;i++){ double a=0; for(int t=0;t<num_ctx;t++) a+=sc[t]*Vc[l][(size_t)t*kvd+gg*HD+i]; for(int t=0;t<rows;t++) a+=sc[num_ctx+t]*Vq[(size_t)t*kvd+gg*HD+i]; attn[h*HD+i]=a/sm; } }
            for(int o=0;o<H;o++){ double a=0; for(int i=0;i<qd;i++) a+=attn[i]*(double)Wo[(size_t)o*qd+i]; xq[(size_t)r*H+o]+=a; }
            std::vector<double> xr(H); for(int i=0;i<H;i++) xr[i]=xq[(size_t)r*H+i]; auto h2=rmsn(xr,poln,H);
            std::vector<double> ff(I); for(int o=0;o<I;o++){ double gg2=0,uu=0; for(int i=0;i<H;i++){ gg2+=h2[i]*(double)Wg[(size_t)o*H+i]; uu+=h2[i]*(double)Wu[(size_t)o*H+i]; } ff[o]=(gg2/(1.0+std::exp(-gg2)))*uu; }
            for(int o=0;o<H;o++){ double a=0; for(int i=0;i<I;i++) a+=ff[i]*(double)Wd[(size_t)o*I+i]; xq[(size_t)r*H+o]+=a; }
        }
    }
    { auto fn=WG("norm.weight"); for(int r=0;r<rows;r++){ std::vector<double> xr(H); for(int i=0;i<H;i++) xr[i]=xq[(size_t)r*H+i]; auto o=rmsn(xr,fn,H); for(int i=0;i<H;i++) xq[(size_t)r*H+i]=o[i]; } }

    double max_rel=0, oss=0; for(double v:xq) oss+=v*v; double oscale=std::sqrt(oss/xq.size())+1e-12;
    for(size_t i=0;i<xq.size();i++){ double rel=std::fabs((double)out[i]-xq[i])/oscale; if(rel>max_rel) max_rel=rel; }
    for(int l=0;l<L;l++){ cudaFree(dK[l]); cudaFree(dV[l]); }
    q35_dflash_ctx_scratch_free(&c); q35_dflash_fwd_scratch_free(&s); cudaFree(dXc); cudaFree(dXq); cudaFree(dCpos); cudaFree(dQpos); q35_dflash_residency_free(&R);
    printf("full DFlash query forward (context+self attention) parity: max signal-rel err=%.3e (num_ctx=%d rows=%d)\n",max_rel,num_ctx,rows);
    const double TOL=1e-3;
    if(max_rel>TOL){ printf("FAIL — query forward parity exceeds %.1e\n",TOL); return 1; }
    printf("PASS — DFlash full query forward matches host double reference on real weights within %.1e\n",TOL);
    return 0;
}
