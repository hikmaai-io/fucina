// NVFP4 cuBLASLt GEMM: correctness (scale-swizzle validation) + accuracy + speed.
//
// Computes D[m,n] = A[m,k] @ B[k,n] where A=weight (op_T, [k,m] stored), B=act (op_N,
// [k,n]).  Both quantized to NVFP4 (E2M1 values + per-16 E4M3 block scales + per-tensor
// fp32 global scale folded into alpha).  Validates the block-scale swizzle layout by
// comparing cuBLASLt vs an fp32 reference GEMM of the SAME dequantized NVFP4 values
// (=> ~0 error iff swizzle correct), then reports accuracy vs the true-fp32 GEMM and
// speed vs BF16 cublasGemmEx.
//
// build: nvcc -O3 -arch=sm_121a cuda/test_fp4_gemm.cu -lcublasLt -lcublas -o /tmp/fp4_gemm
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <library_types.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define LK(x) do{ cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  printf("cublasLt err %s @%d: %d\n",#x,__LINE__,(int)s); exit(1);} }while(0)

static const int BLK = 16; // NVFP4 block size along K

// ---- E2M1 decode (4-bit nibble -> float) for the reference dequant ----
__device__ __forceinline__ float e2m1_decode(uint32_t nib){
    const float lut[8] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    float v = lut[nib & 7u];
    return (nib & 8u) ? -v : v;
}

// ---- quantize [rows][k] row-major (k = contraction, contiguous) to NVFP4 ----
// out: packed fp4 [rows][k/2]; linear E4M3 block scales [rows][k/16]; uses global gs.
__global__ void quant_nvfp4(const float* __restrict__ X, uint8_t* __restrict__ fp4,
                            uint8_t* __restrict__ bscale, int rows, int k, float gs){
    int row = blockIdx.y;
    int blk = blockIdx.x*blockDim.x + threadIdx.x;   // which 16-block along k
    int nblk = k/BLK;
    if (row>=rows || blk>=nblk) return;
    const float* xr = X + (size_t)row*k + blk*BLK;
    float amax=0.f;
    #pragma unroll
    for(int i=0;i<BLK;i++) amax=fmaxf(amax, fabsf(xr[i]));
    // per-block stored scale in (0,448], quantized to E4M3
    float bs_stored = (amax>0.f) ? amax/6.0f/gs : 0.f;
    __nv_fp8_storage_t e = __nv_cvt_float_to_fp8(bs_stored, __NV_SATFINITE, __NV_E4M3);
    bscale[(size_t)row*nblk + blk] = (uint8_t)e;
    // effective per-block divisor = gs * decode(e)
    float bsf = __half2float(__half(__nv_cvt_fp8_to_halfraw(e, __NV_E4M3)));
    float divisor = gs * bsf;
    if (divisor<=0.f) divisor = 1e30f;
    uint8_t* o = fp4 + (size_t)row*(k/2) + blk*(BLK/2);
    #pragma unroll
    for(int i=0;i<BLK;i+=2){
        float2 v = make_float2(xr[i]/divisor, xr[i+1]/divisor);
        __nv_fp4x2_storage_t s = __nv_cvt_float2_to_fp4x2(v, __NV_E2M1, cudaRoundNearest);
        o[i/2] = (uint8_t)s;
    }
}

// ---- reference dequant: NVFP4 -> fp32 [rows][k] (what the HW effectively multiplies) ----
__global__ void dequant_nvfp4(const uint8_t* __restrict__ fp4, const uint8_t* __restrict__ bscale,
                              float* __restrict__ Y, int rows, int k, float gs){
    int row=blockIdx.y; int j=blockIdx.x*blockDim.x+threadIdx.x; // element pair index
    if(row>=rows || j>=k/2) return;
    int nblk=k/BLK;
    uint8_t byte = fp4[(size_t)row*(k/2)+j];
    int eidx0=2*j, blk=eidx0/BLK;
    __nv_fp8_storage_t e = (__nv_fp8_storage_t)bscale[(size_t)row*nblk+blk];
    float bsf = __half2float(__half(__nv_cvt_fp8_to_halfraw(e, __NV_E4M3)));
    float mul = gs*bsf;
    Y[(size_t)row*k+eidx0]   = e2m1_decode(byte & 0xF) * mul;       // low nibble = even idx
    Y[(size_t)row*k+eidx0+1] = e2m1_decode((byte>>4)&0xF) * mul;    // high nibble = odd idx
}

// ---- swizzle linear [outer][nblk] E4M3 scales -> cuBLASLt 32x4x4 blocked layout ----
// padded: outer->128, nblk->4.  candidate layout under test.
static int sf_pad(int x,int m){ return ((x+m-1)/m)*m; }
__global__ void swizzle_scales(const uint8_t* __restrict__ lin, uint8_t* __restrict__ sw,
                               int outer, int nblk, int nblk_pad){
    int o=blockIdx.y*blockDim.y+threadIdx.y;
    int s=blockIdx.x*blockDim.x+threadIdx.x;
    if(o>=outer||s>=nblk) return;
    int oo=o/128, oi=o%128, so=s/4, si=s%4;
    size_t base=((size_t)oo*(nblk_pad/4)+so)*512;
    size_t off=base + (oi%32)*16 + (oi/32)*4 + si;
    sw[off]=lin[(size_t)o*nblk+s];
}

__global__ void f32_to_bf16(__nv_bfloat16* d,const float* s,size_t n){
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) d[i]=__float2bfloat16(s[i]);
}

// host helpers
static float hamax(const std::vector<float>& v){ float a=0; for(float x:v) a=fmaxf(a,fabsf(x)); return a; }

struct Quant { uint8_t *fp4, *scale_sw; float gs; float* dequant; };
// quantize a host [rows][k] matrix into device NVFP4 + swizzled scales; also returns
// the fp32 dequantization of the NVFP4 values (for swizzle validation reference).
static Quant quantize(const std::vector<float>& host,int rows,int k){
    float amax=hamax(host); float gs=amax/(6.0f*448.0f); if(gs<=0)gs=1e-30f;
    float* dX; CK(cudaMalloc(&dX,host.size()*4)); CK(cudaMemcpy(dX,host.data(),host.size()*4,cudaMemcpyHostToDevice));
    int nblk=k/BLK;
    uint8_t *dfp4,*dlin; CK(cudaMalloc(&dfp4,(size_t)rows*(k/2))); CK(cudaMalloc(&dlin,(size_t)rows*nblk));
    dim3 b(256), g((nblk+255)/256, rows); quant_nvfp4<<<g,b>>>(dX,dfp4,dlin,rows,k,gs);
    // dequant fp32 from LINEAR scales (HW-equivalent values) for the swizzle ref
    float* ddq; CK(cudaMalloc(&ddq,(size_t)rows*k*4));
    { dim3 bb(256),gg((k/2+255)/256,rows); dequant_nvfp4<<<gg,bb>>>(dfp4,dlin,ddq,rows,k,gs); }
    int op=sf_pad(rows,128), kp=sf_pad(nblk,4); uint8_t* dsw; size_t swsz=(size_t)op*kp;
    CK(cudaMalloc(&dsw,swsz)); CK(cudaMemset(dsw,0,swsz));
    dim3 b2(32,8), g2((nblk+31)/32,(rows+7)/8); swizzle_scales<<<g2,b2>>>(dlin,dsw,rows,nblk,kp);
    CK(cudaDeviceSynchronize());
    cudaFree(dX); cudaFree(dlin);
    return {dfp4,dsw,gs,ddq};
}

int main(int argc,char**argv){
    int m = argc>1?atoi(argv[1]):3840;   // out_dim (weight rows)
    int n = argc>2?atoi(argv[2]):2048;   // tokens
    int k = argc>3?atoi(argv[3]):3840;   // in_dim (contraction)
    int outlier = argc>4?atoi(argv[4]):1;// inject gemma4-style activation outliers into B
    srand(1234);
    printf("NVFP4 GEMM probe  m=%d n=%d k=%d  outlier=%d\n",m,n,k,outlier);

    // A = weight [m][k] (row-major; contraction k contiguous) ~ N(0, 0.05) like real weights
    // B = activation [n][k] ~ N(0,1) with optional structured per-channel outliers
    std::vector<float> A((size_t)m*k), B((size_t)n*k);
    auto randn=[&](){ float u1=(rand()+1.0f)/(RAND_MAX+2.0f),u2=(rand()+1.0f)/(RAND_MAX+2.0f);
                      return sqrtf(-2*logf(u1))*cosf(6.2831853f*u2); };
    for(auto&x:A) x=0.05f*randn();
    for(int i=0;i<n;i++) for(int j=0;j<k;j++){
        float v=randn();
        if(outlier && (j%512)==0) v*=200.0f;          // few channels with 200x outliers (gemma4)
        B[(size_t)i*k+j]=v;
    }

    Quant qA=quantize(A,m,k), qB=quantize(B,n,k);
    float alpha=qA.gs*qB.gs, beta=0.f;

    // ---- cuBLASLt NVFP4 matmul: D[m,n] fp32 ----
    cublasLtHandle_t lt; LK(cublasLtCreate(&lt));
    cublasLtMatmulDesc_t op; LK(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    cublasOperation_t opT=CUBLAS_OP_T,opN=CUBLAS_OP_N;
    LK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&opT,sizeof(opT)));
    LK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&opN,sizeof(opN)));
    int32_t sm=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    LK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&sm,sizeof(sm)));
    LK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&sm,sizeof(sm)));
    LK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&qA.scale_sw,sizeof(void*)));
    LK(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&qB.scale_sw,sizeof(void*)));
    cublasLtMatrixLayout_t Ad,Bd,Dd;
    LK(cublasLtMatrixLayoutCreate(&Ad,CUDA_R_4F_E2M1,k,m,k));
    LK(cublasLtMatrixLayoutCreate(&Bd,CUDA_R_4F_E2M1,k,n,k));
    LK(cublasLtMatrixLayoutCreate(&Dd,CUDA_R_32F,m,n,m));
    cublasLtMatmulPreference_t pref; LK(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws=64ull<<20; void* dws; CK(cudaMalloc(&dws,ws));
    LK(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)));
    cublasLtMatmulHeuristicResult_t res[8]; int found=0;
    LK(cublasLtMatmulAlgoGetHeuristic(lt,op,Ad,Bd,Dd,Dd,pref,8,res,&found));
    if(!found){ printf("no algo\n"); return 1; }
    float* dD; CK(cudaMalloc(&dD,(size_t)m*n*4));
    LK(cublasLtMatmul(lt,op,&alpha,qA.fp4,Ad,qB.fp4,Bd,&beta,dD,Dd,dD,Dd,&res[0].algo,dws,ws,0));
    CK(cudaDeviceSynchronize());

    cublasHandle_t cb; cublasCreate(&cb);
    // ---- reference 1: fp32 GEMM of the DEQUANTIZED nvfp4 values (validates swizzle) ----
    // Same NVFP4 values cuBLASLt uses => correct swizzle gives ~0 error, wrong gives garbage.
    float* dDswz; CK(cudaMalloc(&dDswz,(size_t)m*n*4));
    { float a1=1,b0=0;
      cublasGemmEx(cb,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,&a1,qA.dequant,CUDA_R_32F,k,qB.dequant,CUDA_R_32F,k,
                   &b0,dDswz,CUDA_R_32F,m,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); }
    CK(cudaDeviceSynchronize());
    {
        std::vector<float> hN((size_t)m*n), hS((size_t)m*n);
        CK(cudaMemcpy(hN.data(),dD,(size_t)m*n*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hS.data(),dDswz,(size_t)m*n*4,cudaMemcpyDeviceToHost));
        double se=0,sr=0,mx=0; for(size_t i=0;i<hN.size();i++){double e=(double)hN[i]-hS[i]; se+=e*e; sr+=(double)hS[i]*hS[i]; mx=fmax(mx,fabs(e));}
        printf("SWIZZLE CHECK (NVFP4 vs fp32-GEMM-of-same-dequant): L2rel=%.4f%% max|d|=%.5f  %s\n",
               100*sqrt(se/(sr+1e-30)), mx, sqrt(se/(sr+1e-30))<0.02 ? "SWIZZLE OK" : "*** SWIZZLE WRONG ***");
    }

    // ---- reference 2: true fp32 GEMM (accuracy ground truth), via cuBLAS BF16 ----
    __nv_bfloat16 *dAbf,*dBbf; CK(cudaMalloc(&dAbf,(size_t)m*k*2)); CK(cudaMalloc(&dBbf,(size_t)n*k*2));
    float *dAf,*dBf; CK(cudaMalloc(&dAf,(size_t)m*k*4)); CK(cudaMalloc(&dBf,(size_t)n*k*4));
    CK(cudaMemcpy(dAf,A.data(),(size_t)m*k*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBf,B.data(),(size_t)n*k*4,cudaMemcpyHostToDevice));
    f32_to_bf16<<<((size_t)m*k+255)/256,256>>>(dAbf,dAf,(size_t)m*k);
    f32_to_bf16<<<((size_t)n*k+255)/256,256>>>(dBbf,dBf,(size_t)n*k);
    float *dDref; CK(cudaMalloc(&dDref,(size_t)m*n*4));
    float a1=1,b0=0;
    cublasGemmEx(cb,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,&a1,dAbf,CUDA_R_16BF,k,dBbf,CUDA_R_16BF,k,
                 &b0,dDref,CUDA_R_32F,m,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CK(cudaDeviceSynchronize());

    // ---- compare NVFP4 vs bf16-ref ----
    std::vector<float> hD((size_t)m*n), hR((size_t)m*n);
    CK(cudaMemcpy(hD.data(),dD,(size_t)m*n*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hR.data(),dDref,(size_t)m*n*4,cudaMemcpyDeviceToHost));
    double se=0,sr=0,maxabs=0; int argok=0;
    for(int col=0; col<n; col++){
        // per-column (per-token) argmax over m, like a projection's output usage
        int aD=0,aR=0; float vD=-1e30f,vR=-1e30f;
        for(int row=0; row<m; row++){
            float d=hD[(size_t)col*m+row], r=hR[(size_t)col*m+row]; // col-major D (m x n), ld=m
            double e=(double)d-r; se+=e*e; sr+=(double)r*r; maxabs=fmax(maxabs,fabs(e));
            if(d>vD){vD=d;aD=row;} if(r>vR){vR=r;aR=row;}
        }
        if(aD==aR) argok++;
    }
    double l2rel=sqrt(se/(sr+1e-30));
    printf("NVFP4 vs BF16-ref:  L2rel=%.4f%%   max|delta|=%.4f   argmax match=%d/%d\n",
           100*l2rel, maxabs, argok, n);

    // ---- speed: NVFP4 vs BF16 ----
    auto bench=[&](const char* tag, auto fn){
        for(int i=0;i<3;i++) fn();
        CK(cudaDeviceSynchronize());
        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        int R=30; cudaEventRecord(s);
        for(int i=0;i<R;i++) fn();
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms,s,e);
        double gf=2.0*m*n*k*R/(ms/1e3)/1e12;
        printf("  %-8s %.3f ms/gemm   %.1f TFLOP/s\n",tag,ms/R,gf);
        cudaEventDestroy(s); cudaEventDestroy(e);
    };
    bench("NVFP4", [&]{ cublasLtMatmul(lt,op,&alpha,qA.fp4,Ad,qB.fp4,Bd,&beta,dD,Dd,dD,Dd,&res[0].algo,dws,ws,0); });
    bench("BF16", [&]{ cublasGemmEx(cb,CUBLAS_OP_T,CUBLAS_OP_N,m,n,k,&a1,dAbf,CUDA_R_16BF,k,dBbf,CUDA_R_16BF,k,
                       &b0,dDref,CUDA_R_32F,m,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP); });
    return 0;
}
