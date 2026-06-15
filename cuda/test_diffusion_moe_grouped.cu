// test_diffusion_moe_grouped.cu — parity + perf of the GROUPED expert Q4_K matmul vs the
// current per-expert dequant+sgemm loop. Mirrors one real MoE layer: C=256 canvas tokens,
// each routed to 8 of 128 experts (random), gate_up projection (Q4_K, 2816→1408).
//
// Build: nvcc -O2 -arch=sm_121a cuda/diffusion_gemma_kernels.cu cuda/test_diffusion_moe_grouped.cu -lcublas -o /tmp/dg_moe_grouped
// Run:   /tmp/dg_moe_grouped <model.gguf>

#include "diffusion_gemma_kernels.cuh"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ fprintf(stderr,"CUDA %s @ %d\n",cudaGetErrorString(e_),__LINE__); exit(1);} }while(0)
#define CB(x) do{ cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"cuBLAS %d @ %d\n",(int)s_,__LINE__); exit(1);} }while(0)

enum { GT_STR=8, GT_ARR=9 };
static uint64_t scalar_sz(uint32_t t){switch(t){case 0:case 1:case 7:return 1;case 2:case 3:return 2;case 4:case 5:case 6:return 4;case 10:case 11:case 12:return 8;default:return 0;}}
struct DGT { uint8_t *dev=nullptr; int type=0; int ndim=0; int64_t ne[4]={1,1,1,1}; int64_t nbytes=0,nelem=0; uint64_t offset=0; };
static int64_t row_bytes(int t,int64_t ne0){ int bb=dg_block_bytes(t),bn=dg_block_nelem(t); if(t==DG_GGML_F32)return ne0*4; if(t==DG_GGML_F16)return ne0*2; return (ne0/bn)*bb; }
#define BM 64
#define BN 64

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [tensor]\n",argv[0]); return 1; }
    const std::string want = argc>2 ? argv[2] : "blk.0.ffn_gate_up_exps.weight";
    const int C=256, K=8, E=128;

    int fd=open(argv[1],O_RDONLY); struct stat stt; fstat(fd,&stt);
    uint8_t* base=(uint8_t*)mmap(nullptr,stt.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    const uint8_t* p=base+8; uint64_t nt=*(uint64_t*)p; p+=8; uint64_t nkv=*(uint64_t*)p; p+=8;
    auto rd_str=[&](const uint8_t*&q){ uint64_t l=*(uint64_t*)q; q+=8; std::string s((const char*)q,l); q+=l; return s; };
    auto skip_val=[&](const uint8_t*&q,uint32_t vt){ if(vt==GT_STR){uint64_t l=*(uint64_t*)q;q+=8+l;} else if(vt==GT_ARR){uint32_t at=*(uint32_t*)q;q+=4;uint64_t c=*(uint64_t*)q;q+=8; if(at==GT_STR){for(uint64_t i=0;i<c;i++){uint64_t l=*(uint64_t*)q;q+=8+l;}} else q+=c*scalar_sz(at);} else q+=scalar_sz(vt); };
    for(uint64_t i=0;i<nkv;i++){ rd_str(p); uint32_t vt=*(uint32_t*)p; p+=4; skip_val(p,vt); }
    std::unordered_map<std::string,DGT> T;
    for(uint64_t i=0;i<nt;i++){ std::string nm=rd_str(p); DGT t; t.ndim=*(uint32_t*)p; p+=4; for(int d=0;d<t.ndim;d++){t.ne[d]=*(int64_t*)p;p+=8;} t.type=*(uint32_t*)p;p+=4; t.offset=*(uint64_t*)p;p+=8; t.nelem=1; for(int d=0;d<t.ndim;d++)t.nelem*=t.ne[d]; int64_t rb=row_bytes(t.type,t.ne[0]),rows=1; for(int d=1;d<t.ndim;d++)rows*=t.ne[d]; t.nbytes=rb*rows; T[nm]=t; }
    uint64_t off=(uint64_t)(p-base); off=(off+31)&~31ull; const uint8_t* data=base+off;

    DGT& W=T.at(want);
    int in_dim=(int)W.ne[0], out_dim=(int)W.ne[1];
    int64_t slab_bytes=row_bytes(W.type,W.ne[0])*out_dim, slab_elem=(int64_t)in_dim*out_dim;
    CK(cudaMalloc(&W.dev,W.nbytes)); CK(cudaMemcpy(W.dev,data+W.offset,W.nbytes,cudaMemcpyHostToDevice));

    // ── routing: each of C tokens picks K distinct experts; group assignments by expert ──
    uint64_t s=0x9e37ull; auto irand=[&](int n){ s^=s<<13; s^=s>>7; s^=s<<17; return (int)((s>>11)%n); };
    std::vector<std::vector<int>> et(E);                  // et[e] = token columns routed to e
    for(int c=0;c<C;c++){ std::vector<char> used(E,0); for(int k=0;k<K;k++){ int e; do{e=irand(E);}while(used[e]); used[e]=1; et[e].push_back(c); } }
    std::vector<int> coloff(E+1,0); for(int e=0;e<E;e++) coloff[e+1]=coloff[e]+(int)et[e].size();
    int total=coloff[E];
    printf("%s  in=%d out=%d  C=%d K=%d  total_assign=%d  active_experts=%d\n",
           want.c_str(),in_dim,out_dim,C,K,total,(int)std::count_if(et.begin(),et.end(),[](auto&v){return !v.empty();}));

    // base token activations X[in_dim × C]; assignment j uses column of its source token
    std::vector<float> hX((size_t)in_dim*C);
    auto frand=[&](){ s^=s<<13; s^=s>>7; s^=s<<17; return ((s>>11)*(1.0/9007199254740992.0))*2.0-1.0; };
    for(auto&v:hX) v=(float)frand();
    float* dX; CK(cudaMalloc(&dX,hX.size()*4)); CK(cudaMemcpy(dX,hX.data(),hX.size()*4,cudaMemcpyHostToDevice));

    // gathered flattened activation [in_dim × total], columns grouped per expert
    std::vector<float> hXe((size_t)in_dim*total);
    for(int e=0;e<E;e++) for(size_t j=0;j<et[e].size();j++){ int tok=et[e][j]; int col=coloff[e]+(int)j;
        memcpy(&hXe[(size_t)col*in_dim], &hX[(size_t)tok*in_dim], in_dim*4); }
    float* dXe; CK(cudaMalloc(&dXe,hXe.size()*4)); CK(cudaMemcpy(dXe,hXe.data(),hXe.size()*4,cudaMemcpyHostToDevice));

    cublasHandle_t cub; CB(cublasCreate(&cub)); cublasSetMathMode(cub,CUBLAS_PEDANTIC_MATH);
    float one=1.f,zero=0.f;
    float* d_w32; CK(cudaMalloc(&d_w32,slab_elem*4));
    float* d_ref; CK(cudaMalloc(&d_ref,(size_t)out_dim*total*4));   // [out_dim × total]
    // REFERENCE: per-expert dequant + sgemm over its gathered columns
    for(int e=0;e<E;e++){ int ne=(int)et[e].size(); if(!ne) continue;
        dg_dequant(W.type,W.dev+(size_t)e*slab_bytes,slab_elem,d_w32,0);
        CB(cublasSgemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,out_dim,ne,in_dim,&one,d_w32,in_dim,
                       dXe+(size_t)coloff[e]*in_dim,in_dim,&zero,d_ref+(size_t)coloff[e]*out_dim,out_dim)); }
    CK(cudaDeviceSynchronize());

    // GROUPED: quantize flattened activation, build tile descriptors, single launch
    int nb=in_dim/32; int8_t* dqx; float* ddx; int* dsx;
    CK(cudaMalloc(&dqx,(size_t)in_dim*total)); CK(cudaMalloc(&ddx,(size_t)nb*total*4)); CK(cudaMalloc(&dsx,(size_t)nb*total*4));
    float* d_grp; CK(cudaMalloc(&d_grp,(size_t)out_dim*total*4));
    std::vector<int> hcount(E), hcoloff(E);            // new grouped API: per-expert count + coloff (all E)
    for(int e=0;e<E;e++){ hcount[e]=(int)et[e].size(); hcoloff[e]=coloff[e]; }
    int *dcount,*dcoloff;
    CK(cudaMalloc(&dcount,E*4)); CK(cudaMalloc(&dcoloff,E*4));
    CK(cudaMemcpy(dcount,hcount.data(),E*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dcoloff,hcoloff.data(),E*4,cudaMemcpyHostToDevice));
    int num_tiles=E;   // (label only)
    auto grouped=[&](){ switch(W.type){
        case DG_GGML_Q4_K: dg_mmq_q4_K_grouped(d_grp,W.dev,slab_bytes,dqx,ddx,dsx,dcoloff,dcount,E,in_dim,out_dim,0); break;
        case DG_GGML_Q8_0: dg_mmq_q8_0_grouped(d_grp,W.dev,slab_bytes,dqx,ddx,dsx,dcoloff,dcount,E,in_dim,out_dim,0); break;
        case DG_GGML_Q5_0: dg_mmq_q5_0_grouped(d_grp,W.dev,slab_bytes,dqx,ddx,dsx,dcoloff,dcount,E,in_dim,out_dim,0); break;
        default: fprintf(stderr,"no grouped kernel for type %d\n",W.type); exit(1); } };
    dg_quantize_q8_1(dXe,dqx,ddx,dsx,in_dim,total,0);
    grouped();
    CK(cudaPeekAtLastError()); CK(cudaDeviceSynchronize());

    // compare
    std::vector<float> ref((size_t)out_dim*total), grp((size_t)out_dim*total);
    CK(cudaMemcpy(ref.data(),d_ref,ref.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(grp.data(),d_grp,grp.size()*4,cudaMemcpyDeviceToHost));
    double sdiff=0,sref=0,mx=0;
    for(size_t i=0;i<ref.size();i++){ double d=fabs((double)grp[i]-ref[i]); mx=fmax(mx,d); sdiff+=d*d; sref+=(double)ref[i]*ref[i]; }
    printf("parity: max|Δ|=%.4f  L2rel=%.5f  tiles=%d\n", mx, sqrt(sdiff)/(sqrt(sref)+1e-12), num_tiles);

    // timing (streaming, REP iters)
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); const int REP=30;
    cudaEventRecord(a);
    for(int r=0;r<REP;r++) for(int e=0;e<E;e++){ int ne=(int)et[e].size(); if(!ne) continue;
        dg_dequant(W.type,W.dev+(size_t)e*slab_bytes,slab_elem,d_w32,0);
        CB(cublasSgemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,out_dim,ne,in_dim,&one,d_w32,in_dim,
                       dXe+(size_t)coloff[e]*in_dim,in_dim,&zero,d_ref+(size_t)coloff[e]*out_dim,out_dim)); }
    cudaEventRecord(b); cudaEventSynchronize(b); float tref=0; cudaEventElapsedTime(&tref,a,b);
    cudaEventRecord(a);
    for(int r=0;r<REP;r++){ dg_quantize_q8_1(dXe,dqx,ddx,dsx,in_dim,total,0); grouped(); }
    cudaEventRecord(b); cudaEventSynchronize(b); float tgrp=0; cudaEventElapsedTime(&tgrp,a,b);
    printf("MoE layer (%s):  ref(per-expert dequant+sgemm)=%.3f ms  grouped-fused=%.3f ms  speedup=%.2fx\n",
           want.c_str(), tref/REP, tgrp/REP, tref/tgrp);
    return 0;
}
