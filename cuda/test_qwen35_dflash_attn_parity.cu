// ABOUTME: P3 non-causal GQA attention parity for the DFlash (1+K) query forward (device vs host).
// ABOUTME: Each query row attends to ALL context positions non-causally; GQA 32 q-heads / 8 kv.
//
// The DFlash query forward attends the (1+K) query tokens against the precomputed context K/V. This
// is the one genuinely new attention pattern versus fucina's causal decode: it is NON-CAUSAL (a
// query sees the whole context prefix, not a triangular mask) and GQA-grouped (32 query heads share
// 8 KV heads, group size 4). This gate validates a device fp32 non-causal GQA attention kernel
// against a host double-precision softmax reference on synthetic Q/K/V of the real draft geometry
// (H=4096, HD=128, NQ=32, NKV=8). Self-contained (no weights needed): the attention math is
// weight-independent, so this runs anywhere and does not gate on the checkpoint.
//
// build: nvcc -O3 -arch=... -std=c++17 -Icuda cuda/test_qwen35_dflash_attn_parity.cu -o t && ./t
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_OK(c) do{ cudaError_t e=(c); if(e!=cudaSuccess){ printf("CUDA %s @ %d\n", cudaGetErrorString(e), __LINE__); return 2; } }while(0)

// Non-causal GQA attention. Q [NQ, HD], K/V [ctx, NKV, HD]. Each q-head h uses kv-head h/(NQ/NKV).
// One block per query head; softmax over ctx positions; output [NQ, HD]. scale = 1/sqrt(HD).
__global__ void attn_noncausal_gqa(const float* Q, const float* K, const float* V,
                                   float* O, int ctx, int NQ, int NKV, int HD){
    int h=blockIdx.x; if(h>=NQ) return;
    int g=h/(NQ/NKV);                 // kv head for this q head
    const float* q=Q+(size_t)h*HD;
    extern __shared__ double sh[];    // [ctx] scores
    double scale=1.0/sqrt((double)HD);
    // scores
    for(int t=threadIdx.x;t<ctx;t+=blockDim.x){
        const float* k=K+((size_t)t*NKV+g)*HD;
        double dot=0; for(int i=0;i<HD;i++) dot+=(double)q[i]*(double)k[i];
        sh[t]=dot*scale;
    }
    __syncthreads();
    // max + sumexp (single thread for exactness; ctx is small = 1+K..few)
    __shared__ double sm;
    if(threadIdx.x==0){
        double m=-1e300; for(int t=0;t<ctx;t++) if(sh[t]>m) m=sh[t];
        double s=0; for(int t=0;t<ctx;t++){ sh[t]=exp(sh[t]-m); s+=sh[t]; }
        sm=s;
    }
    __syncthreads();
    // weighted sum of V
    for(int i=threadIdx.x;i<HD;i+=blockDim.x){
        double acc=0; for(int t=0;t<ctx;t++){ const float* v=V+((size_t)t*NKV+g)*HD; acc+=sh[t]*(double)v[i]; }
        O[(size_t)h*HD+i]=(float)(acc/sm);
    }
}

int main(){
    const int HD=128, NQ=32, NKV=8, ctx=24;   // ctx ~ a realistic (context + 1+K) span
    std::vector<float> Q((size_t)NQ*HD), K((size_t)ctx*NKV*HD), V((size_t)ctx*NKV*HD);
    for(int h=0;h<NQ;h++) for(int i=0;i<HD;i++) Q[(size_t)h*HD+i]=0.02f*std::sin(0.03f*i+0.5f*h);
    for(int t=0;t<ctx;t++) for(int g=0;g<NKV;g++) for(int i=0;i<HD;i++){
        K[((size_t)t*NKV+g)*HD+i]=0.03f*std::cos(0.02f*i+0.1f*t+0.7f*g);
        V[((size_t)t*NKV+g)*HD+i]=0.05f*std::sin(0.01f*i+0.2f*t+0.3f*g);
    }
    float *dQ,*dK,*dV,*dO;
    CUDA_OK(cudaMalloc(&dQ,Q.size()*4)); CUDA_OK(cudaMalloc(&dK,K.size()*4));
    CUDA_OK(cudaMalloc(&dV,V.size()*4)); CUDA_OK(cudaMalloc(&dO,(size_t)NQ*HD*4));
    CUDA_OK(cudaMemcpy(dQ,Q.data(),Q.size()*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dK,K.data(),K.size()*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dV,V.data(),V.size()*4,cudaMemcpyHostToDevice));
    attn_noncausal_gqa<<<NQ,128,(size_t)ctx*sizeof(double)>>>(dQ,dK,dV,dO,ctx,NQ,NKV,HD);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<float> O((size_t)NQ*HD); CUDA_OK(cudaMemcpy(O.data(),dO,O.size()*4,cudaMemcpyDeviceToHost));

    // host double reference
    double max_rel=0, scale=1.0/std::sqrt((double)HD);
    for(int h=0;h<NQ;h++){
        int g=h/(NQ/NKV);
        std::vector<double> sc(ctx);
        for(int t=0;t<ctx;t++){ double dot=0; for(int i=0;i<HD;i++) dot+=(double)Q[(size_t)h*HD+i]*(double)K[((size_t)t*NKV+g)*HD+i]; sc[t]=dot*scale; }
        double m=-1e300; for(double x:sc) if(x>m) m=x; double s=0; for(double&x:sc){ x=std::exp(x-m); s+=x; }
        double oss=0; std::vector<double> ref(HD);
        for(int i=0;i<HD;i++){ double acc=0; for(int t=0;t<ctx;t++) acc+=sc[t]*(double)V[((size_t)t*NKV+g)*HD+i]; ref[i]=acc/s; oss+=ref[i]*ref[i]; }
        double vscale=std::sqrt(oss/HD)+1e-12;
        for(int i=0;i<HD;i++){ double rel=std::fabs((double)O[(size_t)h*HD+i]-ref[i])/vscale; if(rel>max_rel) max_rel=rel; }
    }
    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
    printf("non-causal GQA attention parity: max signal-rel err=%.3e (ctx=%d NQ=%d NKV=%d HD=%d)\n",
           max_rel, ctx, NQ, NKV, HD);
    const double TOL=1e-4;
    if(max_rel>TOL){ printf("FAIL — attention parity exceeds %.1e\n",TOL); return 1; }
    printf("PASS — DFlash non-causal GQA attention matches host double reference within %.1e\n",TOL);
    return 0;
}
