// Diagnostic: is the zero-algo result a malformed descriptor or genuine lack of
// FP4 support?  Control = FP8 E4M3 with identical scaffolding; if FP8 yields algos
// the scaffolding is sound and FP4 is genuinely unsupported by cuBLASLt here.
// Also tries: FP4 w/ real scale pointers set, FP4 w/ FP32 output, FP8 block-scaled.
// build: nvcc -O3 -arch=sm_121a cuda/test_fp4_probe2.cu -lcublasLt -o /tmp/fp4_probe2
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <library_types.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static int heur(cublasLtHandle_t lt, cudaDataType abType, cudaDataType dType,
                int scaleMode, void* aScale, void* bScale,
                int m, int n, int k, const char* tag) {
    cublasLtMatmulDesc_t op=nullptr;
    if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F)!=CUBLAS_STATUS_SUCCESS){printf("descCreate fail\n");return 0;}
    cublasOperation_t opT=CUBLAS_OP_T, opN=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    if (scaleMode >= 0) {
        int32_t sm=scaleMode;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
    }
    if (aScale) cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &aScale, sizeof(aScale));
    if (bScale) cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bScale, sizeof(bScale));

    cublasLtMatrixLayout_t Ad,Bd,Dd;
    cublasLtMatrixLayoutCreate(&Ad, abType, k, m, k);
    cublasLtMatrixLayoutCreate(&Bd, abType, k, n, k);
    cublasLtMatrixLayoutCreate(&Dd, dType, m, n, m);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    size_t ws=64ull*1024*1024;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t res[8]; int found=0;
    cublasStatus_t st=cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Dd, Dd, pref, 8, res, &found);
    printf("  %-40s status=%d algos=%d %s\n", tag, (int)st, found, found>0?"GO":"--");
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Ad);cublasLtMatrixLayoutDestroy(Bd);cublasLtMatrixLayoutDestroy(Dd);
    cublasLtMatmulDescDestroy(op);
    return found;
}

int main(){
    int dev=0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
    printf("Device: %s sm_%d%d  CUDA build 13.0\n", p.name, p.major, p.minor);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    int m=4096,n=2048,k=4096;
    void *sA,*sB; CK(cudaMalloc(&sA, m*k)); CK(cudaMalloc(&sB, n*k)); // generous scale bufs

    printf("\n[control] FP8 E4M3, per-tensor scalar scale (should GO if scaffolding ok):\n");
    heur(lt, CUDA_R_8F_E4M3, CUDA_R_16BF, -1, nullptr, nullptr, m,n,k, "FP8 e4m3, no scale mode");
    heur(lt, CUDA_R_8F_E4M3, CUDA_R_32F,  -1, nullptr, nullptr, m,n,k, "FP8 e4m3, 32F out");
    heur(lt, CUDA_R_8F_E4M3, CUDA_R_16BF, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, sA, sB, m,n,k, "FP8 e4m3 + scalar32F scale ptrs");

    printf("\n[FP4 NVFP4 vec16] variants:\n");
    heur(lt, CUDA_R_4F_E2M1, CUDA_R_16BF, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, sA, sB, m,n,k, "NVFP4 +scale ptrs, bf16 out");
    heur(lt, CUDA_R_4F_E2M1, CUDA_R_32F,  CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, sA, sB, m,n,k, "NVFP4 +scale ptrs, 32F out");
    heur(lt, CUDA_R_4F_E2M1, CUDA_R_16F,  CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, sA, sB, m,n,k, "NVFP4 +scale ptrs, 16F out");

    printf("\n[FP4 MXFP4 vec32] variants:\n");
    heur(lt, CUDA_R_4F_E2M1, CUDA_R_16BF, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, sA, sB, m,n,k, "MXFP4 +scale ptrs, bf16 out");
    heur(lt, CUDA_R_4F_E2M1, CUDA_R_16F,  CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, sA, sB, m,n,k, "MXFP4 +scale ptrs, 16F out");

    printf("\n[control] BF16 (must GO):\n");
    heur(lt, CUDA_R_16BF, CUDA_R_32F, -1, nullptr, nullptr, m,n,k, "BF16 baseline");

    cublasLtDestroy(lt);
    return 0;
}
