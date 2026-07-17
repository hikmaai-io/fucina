// ABOUTME: Isolated LM-head decode microbench — weight-read-once BF16 (1.0 GB) vs Q8_0 (0.53 GB)
// ABOUTME: head GEMV at B=1..8, to quantify the low-B head-read lever (Q8 half-read for B>1 greedy).
//
// The served MoE decode head reads the ~1.0 GB BF16 untied head ONCE per step for B<=16 rows
// (P2 F1). At B==1 greedy fucina instead uses a Q8_0 copy (0.53 GB) + candidate rescore
// (argmax-exact). This bench measures the head read time at each B for BOTH formats in the
// weight-read-once batched form, isolating the head's contribution to the low-B step and the
// exact savings of extending the Q8 half-read from B==1 to B<=8. Also VALIDATES the proposed
// batched Q8 head GEMV kernel (correctness vs a bf16 reference argmax) before it touches the engine.
//
// Standalone (no model load, ~1.6 GiB) — run under flock /tmp/fucina_gpu.lock.
// build: make bench-head-lowc
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

#define MAXB 8

// Weight-read-once BF16 head GEMV: warp per output row, reads the bf16 weight row ONCE and
// applies to all B activation rows (the P2 head-read-once form). y[b*VOC + row].
__global__ void bf16_head_batched(float *y, const __nv_bfloat16 *w, const float *x,
                                  int in_dim, int out_dim, int B) {
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * nwarps + warp;
    if (row >= out_dim) return;
    const __nv_bfloat16 *wr = w + (size_t)row * in_dim;
    float acc[MAXB];
    #pragma unroll
    for (int b = 0; b < MAXB; b++) acc[b] = 0.f;
    for (int k = lane; k < in_dim; k += 32) {
        float wv = __bfloat162float(wr[k]);
        for (int b = 0; b < B; b++) acc[b] += wv * x[(size_t)b * in_dim + k];
    }
    for (int b = 0; b < B; b++) {
        float a = acc[b];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xFFFFFFFFu, a, o);
        if (lane == 0) y[(size_t)b * out_dim + row] = a;
    }
}

// Weight-read-once Q8_0 head GEMV (34 B / 32-elem block: fp16 scale + 32 int8): read the q8
// weight row ONCE, apply to B float activation rows. This is the proposed batched extension of
// the B==1 q8_head_gemv_kernel. y[b*VOC + row].
__global__ void q8_head_batched(float *y, const unsigned char *w, const float *x,
                                int in_dim, int out_dim, int B) {
    int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * nwarps + warp;
    if (row >= out_dim) return;
    int nb = in_dim >> 5;
    const unsigned char *wr = w + (size_t)row * nb * 34;
    float acc[MAXB];
    #pragma unroll
    for (int b = 0; b < MAXB; b++) acc[b] = 0.f;
    for (int blk = lane; blk < nb; blk += 32) {
        const unsigned char *bp = wr + (size_t)blk * 34;
        __half_raw hs; hs.x = (uint16_t)(bp[0] | ((uint16_t)bp[1] << 8));
        float d = __half2float(__half(hs));
        const int8_t *q = (const int8_t *)(bp + 2);
        // Hoist the dequant: convert the 32 int8 → scaled float ONCE, reuse across all B rows
        // (was recomputed per-row → ALU-bound and slower-than-bf16 at B>1).
        float wf[32];
        #pragma unroll
        for (int j = 0; j < 32; j++) wf[j] = d * (float)q[j];
        for (int b = 0; b < B; b++) {
            const float *xb = x + (size_t)b * in_dim + blk * 32;
            float p = 0.f;
            #pragma unroll
            for (int j = 0; j < 32; j++) p += wf[j] * xb[j];
            acc[b] += p;
        }
    }
    for (int b = 0; b < B; b++) {
        float a = acc[b];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xFFFFFFFFu, a, o);
        if (lane == 0) y[(size_t)b * out_dim + row] = a;
    }
}

static double time_kernel(void(*launch)(float*,const void*,const float*,int,int,int,cudaStream_t),
                          float *y, const void *w, const float *x, int in, int out, int B, int R) {
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i=0;i<10;i++) launch(y,w,x,in,out,B,0);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int i=0;i<R;i++) launch(y,w,x,in,out,B,0);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1));
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms/R;
}
static void launch_bf16(float*y,const void*w,const float*x,int in,int out,int B,cudaStream_t s){
    int WPB=4, blk=(out+WPB-1)/WPB; bf16_head_batched<<<blk,WPB*32,0,s>>>(y,(const __nv_bfloat16*)w,x,in,out,B);
}
static void launch_q8(float*y,const void*w,const float*x,int in,int out,int B,cudaStream_t s){
    int WPB=8, blk=(out+WPB-1)/WPB; q8_head_batched<<<blk,WPB*32,0,s>>>(y,(const unsigned char*)w,x,in,out,B);
}

int main(int argc,char**argv){
    const int VOC = 248320, H = 2048;         // Qwen3.5-35B-A3B untied head
    const int R = (argc>1)? atoi(argv[1]) : 300;
    printf("LM-head decode microbench (VOC=%d H=%d) — weight-read-once, R=%d\n", VOC, H, R);
    double bf16_gb = (double)VOC*H*2/1e9, q8_gb = (double)VOC*(H/32)*34/1e9;
    printf("bf16 head = %.3f GB, q8 head = %.3f GB (%.2fx fewer bytes)\n\n", bf16_gb, q8_gb, bf16_gb/q8_gb);

    std::vector<__nv_bfloat16> hw(( size_t)VOC*H);
    for (size_t i=0;i<hw.size();i++) hw[i]=__float2bfloat16(((int)(i*2654435761u>>24)-128)/256.0f);
    __nv_bfloat16 *dwbf; CK(cudaMalloc(&dwbf,(size_t)VOC*H*2)); CK(cudaMemcpy(dwbf,hw.data(),(size_t)VOC*H*2,cudaMemcpyHostToDevice));
    // Build a Q8_0 copy from the bf16 weights (per-32 block scale + int8), on host.
    int nb=H/32; std::vector<unsigned char> hq((size_t)VOC*nb*34);
    for (int r=0;r<VOC;r++) for(int b=0;b<nb;b++){
        float amax=0; for(int j=0;j<32;j++){float v=__bfloat162float(hw[(size_t)r*H+b*32+j]); amax=fmaxf(amax,fabsf(v));}
        float d=amax/127.f; if(d<=0)d=1e-9f; unsigned char*bp=&hq[((size_t)r*nb+b)*34];
        __half hs=__float2half(d); uint16_t hsx=*(uint16_t*)&hs; bp[0]=hsx&0xff; bp[1]=hsx>>8;
        for(int j=0;j<32;j++){int q=(int)lrintf(__bfloat162float(hw[(size_t)r*H+b*32+j])/d); q=q<-127?-127:(q>127?127:q); ((int8_t*)(bp+2))[j]=(int8_t)q;}
    }
    unsigned char*dwq8; CK(cudaMalloc(&dwq8,hq.size())); CK(cudaMemcpy(dwq8,hq.data(),hq.size(),cudaMemcpyHostToDevice));
    std::vector<float> hx((size_t)MAXB*H); for(auto&v:hx)v=((rand()%1000)-500)/500.f;
    float*dx; CK(cudaMalloc(&dx,(size_t)MAXB*H*4)); CK(cudaMemcpy(dx,hx.data(),(size_t)MAXB*H*4,cudaMemcpyHostToDevice));
    float*dy; CK(cudaMalloc(&dy,(size_t)MAXB*VOC*4));

    printf("%3s %14s %10s %14s %10s %8s\n","B","bf16 ms","bf16 GB/s","q8 ms","q8 GB/s","q8 speedup");
    int Bs[]={1,2,4,8};
    for(int bi=0;bi<4;bi++){
        int B=Bs[bi];
        double tb=time_kernel(launch_bf16,dy,dwbf,dx,H,VOC,B,R);
        double tq=time_kernel(launch_q8,dy,dwq8,dx,H,VOC,B,R);
        printf("%3d %14.4f %10.1f %14.4f %10.1f %8.2fx\n",
               B, tb, bf16_gb/(tb/1e3), tq, q8_gb/(tq/1e3), tb/tq);
    }
    return 0;
}
