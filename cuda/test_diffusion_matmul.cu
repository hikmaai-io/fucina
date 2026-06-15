// test_diffusion_matmul.cu — parity harness for the fused dp4a quant matmul kernels.
//
// For a chosen quantized weight tensor from the real GGUF, compares the FUSED path
// (dg_quantize_q8_1 + dg_mmq_*) against the REFERENCE path the engine currently uses
// (dg_dequant → cuBLAS sgemm, fp32). The fused path quantizes activations to int8, so it is
// NOT bit-exact; we report max/mean absolute and relative error. PASS = mean relative error
// below a small threshold (int8-activation dp4a, the standard llama.cpp inference path).
//
// Build: nvcc -O2 -arch=sm_121a cuda/diffusion_gemma_kernels.cu cuda/test_diffusion_matmul.cu -lcublas -o /tmp/dg_matmul_test
// Run:   /tmp/dg_matmul_test <model.gguf> [tensor_name] [N]

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
    if(argc<2){ fprintf(stderr,"usage: %s model.gguf [tensor] [N]\n",argv[0]); return 1; }
    std::string want = argc>2 ? argv[2] : "blk.0.attn_q.weight";
    int N = argc>3 ? atoi(argv[3]) : 256;

    int fd=open(argv[1],O_RDONLY); struct stat stt; fstat(fd,&stt);
    uint8_t* base=(uint8_t*)mmap(nullptr,stt.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    const uint8_t* p=base+8; uint64_t nt=*(uint64_t*)p; p+=8; uint64_t nkv=*(uint64_t*)p; p+=8;
    auto rd_str=[&](const uint8_t*&q){ uint64_t l=*(uint64_t*)q; q+=8; std::string s((const char*)q,l); q+=l; return s; };
    auto skip_val=[&](const uint8_t*&q,uint32_t vt){ if(vt==GT_STR){uint64_t l=*(uint64_t*)q;q+=8+l;} else if(vt==GT_ARR){uint32_t at=*(uint32_t*)q;q+=4;uint64_t c=*(uint64_t*)q;q+=8; if(at==GT_STR){for(uint64_t i=0;i<c;i++){uint64_t l=*(uint64_t*)q;q+=8+l;}} else q+=c*scalar_sz(at);} else q+=scalar_sz(vt); };
    for(uint64_t i=0;i<nkv;i++){ rd_str(p); uint32_t vt=*(uint32_t*)p; p+=4; skip_val(p,vt); }
    std::unordered_map<std::string,DGT> T;
    for(uint64_t i=0;i<nt;i++){ std::string nm=rd_str(p); DGT t; t.ndim=*(uint32_t*)p; p+=4; for(int d=0;d<t.ndim;d++){t.ne[d]=*(int64_t*)p;p+=8;} t.type=*(uint32_t*)p;p+=4; t.offset=*(uint64_t*)p;p+=8; t.nelem=1; for(int d=0;d<t.ndim;d++)t.nelem*=t.ne[d]; int64_t rb=row_bytes(t.type,t.ne[0]),rows=1; for(int d=1;d<t.ndim;d++)rows*=t.ne[d]; t.nbytes=rb*rows; T[nm]=t; }
    uint64_t off=(uint64_t)(p-base); off=(off+31)&~31ull; const uint8_t* data=base+off;

    auto it=T.find(want); if(it==T.end()){ fprintf(stderr,"tensor '%s' not found\n",want.c_str()); return 1; }
    DGT& W=it->second;
    int in_dim=(int)W.ne[0], out_dim=(int)W.ne[1];
    int64_t slab_elem=(int64_t)in_dim*out_dim;   // 3D expert tensors: test slab (expert) 0 only
    printf("tensor=%s type=%d in_dim=%d out_dim=%d N=%d%s\n", want.c_str(), W.type, in_dim, out_dim, N,
           W.ndim>2?" (expert slab 0)":"");
    CK(cudaMalloc(&W.dev,W.nbytes)); CK(cudaMemcpy(W.dev,data+W.offset,W.nbytes,cudaMemcpyHostToDevice));

    // random activation x [in_dim × N], column-major
    std::vector<float> hx((size_t)in_dim*N);
    uint64_t s=0x1234567ull; auto rnd=[&](){ s^=s<<13; s^=s>>7; s^=s<<17; return ((s>>11)*(1.0/9007199254740992.0))*2.0-1.0; };
    for(auto&v:hx) v=(float)rnd();
    float* d_x; CK(cudaMalloc(&d_x,hx.size()*4)); CK(cudaMemcpy(d_x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));

    // ── REFERENCE: dequant → cuBLAS sgemm (exactly the engine's `mm`) ──
    cublasHandle_t cub; CB(cublasCreate(&cub)); cublasSetMathMode(cub,CUBLAS_PEDANTIC_MATH);
    float* d_w32; CK(cudaMalloc(&d_w32,(size_t)in_dim*out_dim*4));
    if(dg_dequant(W.type,W.dev,slab_elem,d_w32,0)!=0){ fprintf(stderr,"dequant failed\n"); return 1; }
    float* d_ref; CK(cudaMalloc(&d_ref,(size_t)out_dim*N*4));
    float one=1.f,zero=0.f;
    CB(cublasSgemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,out_dim,N,in_dim,&one,d_w32,in_dim,d_x,in_dim,&zero,d_ref,out_dim));
    CK(cudaDeviceSynchronize());

    // ── FUSED: quantize activation → dp4a tiled MMQ ──
    int nb=in_dim/32;
    int8_t* d_qx; float* d_dx; int* d_sx;
    CK(cudaMalloc(&d_qx,(size_t)in_dim*N)); CK(cudaMalloc(&d_dx,(size_t)nb*N*4)); CK(cudaMalloc(&d_sx,(size_t)nb*N*4));
    float* d_fus; CK(cudaMalloc(&d_fus,(size_t)out_dim*N*4));
    dg_quantize_q8_1(d_x,d_qx,d_dx,d_sx,in_dim,N,0);
    if(W.type==DG_GGML_Q4_K) dg_mmq_q4_K(d_fus,W.dev,d_qx,d_dx,d_sx,in_dim,out_dim,N,0);
    else { fprintf(stderr,"no fused kernel for type %d yet\n",W.type); return 1; }
    CK(cudaPeekAtLastError()); CK(cudaDeviceSynchronize());

    // ── compare ──
    std::vector<float> ref((size_t)out_dim*N), fus((size_t)out_dim*N);
    CK(cudaMemcpy(ref.data(),d_ref,ref.size()*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(fus.data(),d_fus,fus.size()*4,cudaMemcpyDeviceToHost));
    double maxabs=0, sumabs=0, sumref2=0, sumdiff2=0, maxrel=0; double refmax=0;
    for(size_t i=0;i<ref.size();i++){
        double dabs=fabs((double)fus[i]-ref[i]);
        maxabs=fmax(maxabs,dabs); sumabs+=dabs;
        sumref2+=(double)ref[i]*ref[i]; sumdiff2+=dabs*dabs;
        refmax=fmax(refmax,fabs((double)ref[i]));
    }
    double meanabs=sumabs/ref.size();
    double l2rel=sqrt(sumdiff2)/(sqrt(sumref2)+1e-12);
    // argmax-per-column agreement (the metric the forward test cares about)
    int agree=0;
    for(int c=0;c<N;c++){ int ar=0,af=0; float vr=-1e30f,vf=-1e30f;
        for(int r=0;r<out_dim;r++){ float a=ref[(size_t)c*out_dim+r]; if(a>vr){vr=a;ar=r;} float b=fus[(size_t)c*out_dim+r]; if(b>vf){vf=b;af=r;} }
        if(ar==af) agree++; }
    printf("ref|max|=%.4f  max|Δ|=%.4f  mean|Δ|=%.5f  L2rel=%.5f  argmax-agree=%d/%d\n",
           refmax,maxabs,meanabs,l2rel,agree,N);

    // ── timing: reference (dequant+sgemm) vs fused (quantize+MMQ), per matmul ──
    const int IT=100; cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for(int i=0;i<IT;i++){ dg_dequant(W.type,W.dev,slab_elem,d_w32,0);
        cublasSgemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,out_dim,N,in_dim,&one,d_w32,in_dim,d_x,in_dim,&zero,d_ref,out_dim); }
    cudaEventRecord(b); cudaEventSynchronize(b); float tref=0; cudaEventElapsedTime(&tref,a,b);
    cudaEventRecord(a);
    for(int i=0;i<IT;i++){ dg_quantize_q8_1(d_x,d_qx,d_dx,d_sx,in_dim,N,0);
        dg_mmq_q4_K(d_fus,W.dev,d_qx,d_dx,d_sx,in_dim,out_dim,N,0); }
    cudaEventRecord(b); cudaEventSynchronize(b); float tfus=0; cudaEventElapsedTime(&tfus,a,b);
    printf("time/matmul: ref(dequant+sgemm)=%.3f ms  fused(dp4a MMQ)=%.3f ms  speedup=%.2fx\n",
           tref/IT, tfus/IT, tref/tfus);

    bool pass = (l2rel < 0.03) && (agree >= (int)(0.98*N));
    printf("%s\n", pass?"PASS":"FAIL");
    return pass?0:2;
}
