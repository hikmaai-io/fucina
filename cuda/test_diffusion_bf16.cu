// test_diffusion_bf16.cu — does real BF16 tensor-core cublasGemmEx beat fp32 SIMT cublasSgemm for
// the diffusion engine's dense/vocab GEMM shapes on this GPU? (TF32 was slower; this tests genuine
// bf16 inputs.) out[M×N] = A[K×M]^T · B[K×N], column-major, the engine's OP_T/OP_N convention.
//
// Build: nvcc -O2 -arch=sm_121a cuda/test_diffusion_bf16.cu -lcublas -o /tmp/dg_bf16

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("cuda %s %d\n",cudaGetErrorString(e),__LINE__);return 1;} }while(0)
#define CB(x) do{ cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cublas %d @ %d\n",(int)s,__LINE__);return 1;} }while(0)

__global__ void f2bf(const float* in, __nv_bfloat16* out, int64_t n){ int64_t i=blockIdx.x*256+threadIdx.x; if(i<n) out[i]=__float2bfloat16(in[i]); }

static int bench(cublasHandle_t cub, int M,int N,int K,const char* name){
    float *dA,*dB,*dC; CK(cudaMalloc(&dA,(size_t)K*M*4)); CK(cudaMalloc(&dB,(size_t)K*N*4)); CK(cudaMalloc(&dC,(size_t)M*N*4));
    std::vector<float> h((size_t)K*M); for(auto&v:h) v=(float)((rand()%200-100)/100.0); CK(cudaMemcpy(dA,h.data(),h.size()*4,cudaMemcpyHostToDevice));
    h.resize((size_t)K*N); for(auto&v:h) v=(float)((rand()%200-100)/100.0); CK(cudaMemcpy(dB,h.data(),h.size()*4,cudaMemcpyHostToDevice));
    __nv_bfloat16 *bA,*bB; CK(cudaMalloc(&bA,(size_t)K*M*2)); CK(cudaMalloc(&bB,(size_t)K*N*2));
    f2bf<<<(K*M+255)/256,256>>>(dA,bA,(int64_t)K*M); f2bf<<<(K*N+255)/256,256>>>(dB,bB,(int64_t)K*N);
    float one=1.f,zero=0.f; const int IT=50;
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    // fp32 SIMT (PEDANTIC)
    cublasSetMathMode(cub,CUBLAS_PEDANTIC_MATH);
    cudaEventRecord(a);
    for(int i=0;i<IT;i++) CB(cublasSgemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,M,N,K,&one,dA,K,dB,K,&zero,dC,M));
    cudaEventRecord(b); cudaEventSynchronize(b); float t32=0; cudaEventElapsedTime(&t32,a,b);
    // bf16 tensor-core (compute fp32)
    cublasSetMathMode(cub,CUBLAS_DEFAULT_MATH);
    cudaEventRecord(a);
    for(int i=0;i<IT;i++) CB(cublasGemmEx(cub,CUBLAS_OP_T,CUBLAS_OP_N,M,N,K,&one,bA,CUDA_R_16BF,K,bB,CUDA_R_16BF,K,&zero,dC,CUDA_R_32F,M,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    cudaEventRecord(b); cudaEventSynchronize(b); float tbf=0; cudaEventElapsedTime(&tbf,a,b);
    printf("%-22s M=%-7d N=%-4d K=%-7d  fp32-simt=%.3fms  bf16-tc=%.3fms  speedup=%.2fx\n",name,M,N,K,t32/IT,tbf/IT,t32/tbf);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(bA);cudaFree(bB); return 0;
}
int main(){
    cublasHandle_t cub; cublasCreate(&cub);
    bench(cub,4096,256,2816,"dense attn_q");
    bench(cub,2816,256,4096,"dense attn_out");
    bench(cub,2112,256,2816,"dense ffn_gate");
    bench(cub,262144,256,2816,"LM head");
    bench(cub,2816,256,262144,"self-cond soft-embed");
    return 0;
}
