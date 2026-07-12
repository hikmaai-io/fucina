// ABOUTME: Gate for the callable DFlash draft backbone forward over residency (real weights).
// ABOUTME: Runs q35_dflash_backbone_forward reading resident BF16 views vs a host double reference.
//
// Proves the assembled, callable draft backbone (the form the verify path invokes) matches a host
// double-precision reference on the real z-lab weights — i.e. the residency + forward module
// reproduce the standalone backbone parity, now reading BF16 device views instead of pre-uploaded
// f32 buffers. SKIPs cleanly when the checkpoint is absent.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_forward.cu -o t && ./t
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

    const auto& g=R.geom; const int H=g.H,HD=g.HD,NQ=g.NQ,NKV=g.NKV,I=g.I,qd=g.q_dim(),kvd=g.kv_dim(),rows=7;
    const double theta=1e7; const float eps=1e-6f;

    std::vector<float> X((size_t)rows*H); for(int r=0;r<rows;r++) for(int i=0;i<H;i++) X[(size_t)r*H+i]=0.03f*std::sin(0.002f*i+0.4f*r)+0.005f*(r+1);
    std::vector<int> pos(rows); for(int r=0;r<rows;r++) pos[r]=100+r;

    // ---- host double reference (same as backbone parity) reading f32 weights ----
    auto WL=[&](int l,const char* s){ return to_f32(M.find("layers."+std::to_string(l)+"."+s)); };
    auto rmsn=[&](const std::vector<double>& v,const std::vector<float>& w,int n){ double ss=0; for(int i=0;i<n;i++) ss+=v[i]*v[i]; double inv=1.0/std::sqrt(ss/n+eps); std::vector<double> o(n); for(int i=0;i<n;i++) o[i]=v[i]*inv*(double)w[i]; return o; };
    std::vector<double> xref((size_t)rows*H); for(size_t i=0;i<xref.size();i++) xref[i]=X[i];
    for(int l=0;l<g.L;l++){
        auto inln=WL(l,"input_layernorm.weight"),Wq=WL(l,"self_attn.q_proj.weight"),Wk=WL(l,"self_attn.k_proj.weight"),Wv=WL(l,"self_attn.v_proj.weight"),
             Wo=WL(l,"self_attn.o_proj.weight"),qn=WL(l,"self_attn.q_norm.weight"),kn=WL(l,"self_attn.k_norm.weight"),poln=WL(l,"post_attention_layernorm.weight"),
             Wg=WL(l,"mlp.gate_proj.weight"),Wu=WL(l,"mlp.up_proj.weight"),Wd=WL(l,"mlp.down_proj.weight");
        std::vector<double> Qh((size_t)rows*qd),Kh((size_t)rows*kvd),Vh((size_t)rows*kvd);
        for(int r=0;r<rows;r++){
            std::vector<double> xr(H); for(int i=0;i<H;i++) xr[i]=xref[(size_t)r*H+i]; auto h1=rmsn(xr,inln,H);
            for(int o=0;o<qd;o++){ double a=0; for(int i=0;i<H;i++) a+=h1[i]*(double)Wq[(size_t)o*H+i]; Qh[(size_t)r*qd+o]=a; }
            for(int o=0;o<kvd;o++){ double a=0,b=0; for(int i=0;i<H;i++){ a+=h1[i]*(double)Wk[(size_t)o*H+i]; b+=h1[i]*(double)Wv[(size_t)o*H+i]; } Kh[(size_t)r*kvd+o]=a; Vh[(size_t)r*kvd+o]=b; }
            for(int hh=0;hh<NQ;hh++){ double ssq=0; for(int i=0;i<HD;i++){ double x=Qh[(size_t)r*qd+hh*HD+i]; ssq+=x*x; } double inv=1.0/std::sqrt(ssq/HD+eps); std::vector<double> t(HD); for(int i=0;i<HD;i++) t[i]=Qh[(size_t)r*qd+hh*HD+i]*inv*(double)qn[i]; int half=HD/2; for(int i=0;i<half;i++){ double ff=std::pow(theta,-2.0*i/(double)HD),a=pos[r]*ff,c=std::cos(a),s=std::sin(a),x=t[i],y=t[i+half]; Qh[(size_t)r*qd+hh*HD+i]=x*c-y*s; Qh[(size_t)r*qd+hh*HD+i+half]=x*s+y*c; } }
            for(int hh=0;hh<NKV;hh++){ double ssk=0; for(int i=0;i<HD;i++){ double x=Kh[(size_t)r*kvd+hh*HD+i]; ssk+=x*x; } double inv=1.0/std::sqrt(ssk/HD+eps); std::vector<double> t(HD); for(int i=0;i<HD;i++) t[i]=Kh[(size_t)r*kvd+hh*HD+i]*inv*(double)kn[i]; int half=HD/2; for(int i=0;i<half;i++){ double ff=std::pow(theta,-2.0*i/(double)HD),a=pos[r]*ff,c=std::cos(a),s=std::sin(a),x=t[i],y=t[i+half]; Kh[(size_t)r*kvd+hh*HD+i]=x*c-y*s; Kh[(size_t)r*kvd+hh*HD+i+half]=x*s+y*c; } }
        }
        double scale=1.0/std::sqrt((double)HD);
        for(int r=0;r<rows;r++){
            std::vector<double> attn(qd,0);
            for(int hh=0;hh<NQ;hh++){ int gg=hh/(NQ/NKV); std::vector<double> sc(rows); for(int t=0;t<rows;t++){ double d=0; for(int i=0;i<HD;i++) d+=Qh[(size_t)r*qd+hh*HD+i]*Kh[(size_t)t*kvd+gg*HD+i]; sc[t]=d*scale; } double mmax=-1e300; for(double x:sc) if(x>mmax) mmax=x; double sm=0; for(double&x:sc){ x=std::exp(x-mmax); sm+=x; } for(int i=0;i<HD;i++){ double a=0; for(int t=0;t<rows;t++) a+=sc[t]*Vh[(size_t)t*kvd+gg*HD+i]; attn[hh*HD+i]=a/sm; } }
            for(int o=0;o<H;o++){ double a=0; for(int i=0;i<qd;i++) a+=attn[i]*(double)Wo[(size_t)o*qd+i]; xref[(size_t)r*H+o]+=a; }
            std::vector<double> xr(H); for(int i=0;i<H;i++) xr[i]=xref[(size_t)r*H+i]; auto h2=rmsn(xr,poln,H);
            std::vector<double> ff(I); for(int o=0;o<I;o++){ double gg2=0,uu=0; for(int i=0;i<H;i++){ gg2+=h2[i]*(double)Wg[(size_t)o*H+i]; uu+=h2[i]*(double)Wu[(size_t)o*H+i]; } ff[o]=(gg2/(1.0+std::exp(-gg2)))*uu; }
            for(int o=0;o<H;o++){ double a=0; for(int i=0;i<I;i++) a+=ff[i]*(double)Wd[(size_t)o*I+i]; xref[(size_t)r*H+o]+=a; }
        }
    }
    { auto fn=to_f32(M.find("norm.weight")); for(int r=0;r<rows;r++){ std::vector<double> xr(H); for(int i=0;i<H;i++) xr[i]=xref[(size_t)r*H+i]; auto o=rmsn(xr,fn,H); for(int i=0;i<H;i++) xref[(size_t)r*H+i]=o[i]; } }

    // ---- device: callable module forward reading BF16 views ----
    float* dX; CUDA_OK(cudaMalloc(&dX,(size_t)rows*H*4)); CUDA_OK(cudaMemcpy(dX,X.data(),(size_t)rows*H*4,cudaMemcpyHostToDevice));
    int* dPos; CUDA_OK(cudaMalloc(&dPos,rows*sizeof(int))); CUDA_OK(cudaMemcpy(dPos,pos.data(),rows*sizeof(int),cudaMemcpyHostToDevice));
    q35_dflash_fwd_scratch s{}; if(!q35_dflash_fwd_scratch_alloc(&s,g,rows)){ printf("FAIL scratch alloc\n"); return 1; }
    q35_dflash_backbone_forward(R,s,dX,dPos,theta,eps,0);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> out((size_t)rows*H); CUDA_OK(cudaMemcpy(out.data(),s.out,(size_t)rows*H*4,cudaMemcpyDeviceToHost));

    double max_rel=0, oss=0; for(double v:xref) oss+=v*v; double oscale=std::sqrt(oss/xref.size())+1e-12;
    for(size_t i=0;i<xref.size();i++){ double rel=std::fabs((double)out[i]-xref[i])/oscale; if(rel>max_rel) max_rel=rel; }
    q35_dflash_fwd_scratch_free(&s); cudaFree(dX); cudaFree(dPos); q35_dflash_residency_free(&R);
    printf("callable draft backbone forward (resident BF16 views) parity: max signal-rel err=%.3e (rows=%d)\n",max_rel,rows);
    const double TOL=1e-3;
    if(max_rel>TOL){ printf("FAIL — forward module parity exceeds %.1e\n",TOL); return 1; }
    printf("PASS — DFlash callable backbone forward over residency matches host double reference within %.1e\n",TOL);
    return 0;
}
