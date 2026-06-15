// test_diffusion_moe_stream.cu — REALISTIC perf test of the fused dp4a path vs dequant+sgemm.
// Streams all 128 expert slabs of one MoE layer (gate_up, Q4_K, 285 MB ≫ L2) so the fp32
// dequant round-trip cannot be hidden by cache — this is what the engine actually does per step
// (≈77 distinct experts × 30 layers, no weight reuse). Each expert gets N columns.
//
// Build: nvcc -O2 -arch=sm_121a cuda/diffusion_gemma_kernels.cu cuda/test_diffusion_moe_stream.cu -lcublas -o /tmp/dg_moe_stream
// Run:   /tmp/dg_moe_stream <model.gguf> [N]

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

int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [N]\n",argv[0]); return 1; }
    int N = argc>2 ? atoi(argv[2]) : 16;
    const std::string want="blk.0.ffn_gate_up_exps.weight";

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
    int in_dim=(int)W.ne[0], out_dim=(int)W.ne[1], E=(int)W.ne[2];
    int64_t slab_bytes=row_bytes(W.type,W.ne[0])*out_dim;
    printf("%s type=%d in=%d out=%d experts=%d N=%d  (%.0f MB total)\n",
           want.c_str(),W.type,in_dim,out_dim,E,N,W.nbytes/1e6);
    CK(cudaMalloc(&W.dev,W.nbytes)); CK(cudaMemcpy(W.dev,data+W.offset,W.nbytes,cudaMemcpyHostToDevice));

    std::vector<float> hx((size_t)in_dim*N);
    uint64_t s=0x1234567ull; auto rnd=[&](){ s^=s<<13; s^=s>>7; s^=s<<17; return ((s>>11)*(1.0/9007199254740992.0))*2.0-1.0; };
    for(auto&v:hx) v=(float)rnd();
    float* d_x; CK(cudaMalloc(&d_x,hx.size()*4)); CK(cudaMemcpy(d_x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));

    cublasHandle_t cub; CB(cublasCreate(&cub)); cublasSetMathMode(cub,CUBLAS_PEDANTIC_MATH);
    float one=1.f,zero=0.f;
    int64_t slab_elem=(int64_t)in_dim*out_dim;
    float* d_w32; CK(cudaMalloc(&d_w32,slab_elem*4));
    float* d_out; CK(cudaMalloc(&d_out,(size_t)out_dim*N*4));
    int nb=in_dim/32; int8_t* d_qx; float* d_dx; int* d_sx;
    CK(cudaMalloc(&d_qx,(size_t)in_dim*N)); CK(cudaMalloc(&d_dx,(size_t)nb*N*4)); CK(cudaMalloc(&d_sx,(size_t)nb*N*4));

    auto warm=[&](){ CK(cudaDeviceSynchronize()); };
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int REP=20;

    // REFERENCE: per-expert dequant + sgemm, streaming all E slabs
    warm(); cudaEventRecord(a);
    for(int r=0;r<REP;r++) for(int e=0;e<E;e++){
        dg_dequant(W.type,W.dev+(size_t)e*slab_bytes,slab_elem,d_w32,0);
        cublasSgemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,out_dim,N,in_dim,&one,d_w32,in_dim,d_x,in_dim,&zero,d_out,out_dim);
    }
    cudaEventRecord(b); cudaEventSynchronize(b); float tref=0; cudaEventElapsedTime(&tref,a,b);

    // FUSED: per-expert quantize-once + dp4a MMQ, streaming all E slabs
    warm(); cudaEventRecord(a);
    for(int r=0;r<REP;r++){
        dg_quantize_q8_1(d_x,d_qx,d_dx,d_sx,in_dim,N,0);   // activation shared across experts
        for(int e=0;e<E;e++)
            dg_mmq_q4_K(d_out,W.dev+(size_t)e*slab_bytes,d_qx,d_dx,d_sx,in_dim,out_dim,N,0);
    }
    cudaEventRecord(b); cudaEventSynchronize(b); float tfus=0; cudaEventElapsedTime(&tfus,a,b);

    printf("per-layer expert pass (%d experts, N=%d):  ref=%.3f ms  fused=%.3f ms  speedup=%.2fx\n",
           E, N, tref/REP, tfus/REP, tref/tfus);
    return 0;
}
