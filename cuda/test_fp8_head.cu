// Accuracy + bandwidth gate for the FP8 E4M3 per-row LM head (nvfp4_gemv.cuh).
//
// The untied NVFP4 lm_head is BF16 [vocab][hidden] (~2 GB), read every token. This test:
//   (a) quantizes a representative head slice BF16 → E4M3 per-row (fp8_head_quantize_kernel),
//   (b) runs the FP8 head GEMV (single + batched) and the BF16 head GEMV on the SAME hidden
//       states, and reports the per-token top-1 argmax MATCH RATE (FP8 vs BF16) + logit L2rel.
//       The engine gate keeps FP8 only if the top-1 match is perfect; otherwise it keeps BF16.
//   (c) confirms the FP8 head reads exactly HALF the bytes of the BF16 head (1 B vs 2 B / elem).
//
// "Real" hidden states: Gemma post-norm hidden states are unit-RMS-ish per the final RMSNorm, so
// we draw x ~ N(0,1) and RMS-normalize it (the actual head input distribution). The head rows are
// drawn from a heavy-tailed-ish distribution to exercise per-row amax scaling.
//
// build: nvcc -arch=sm_121a -O3 -std=c++17 cuda/test_fp8_head.cu -o /tmp/fp8h && /tmp/fp8h
#include "nvfp4_gemv.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_bf16.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static int argmax(const float* v, int n){ int b=0; float bv=v[0]; for(int i=1;i<n;i++) if(v[i]>bv){bv=v[i];b=i;} return b; }

int main(int argc, char** argv) {
    int hidden = argc>1 ? atoi(argv[1]) : 3840;
    int vocab  = argc>2 ? atoi(argv[2]) : 262144;   // full Gemma-4 vocab
    int K      = argc>3 ? atoi(argv[3]) : 5;
    srand(1234);

    // BF16 head [vocab][hidden]
    std::vector<__nv_bfloat16> hW((size_t)vocab*hidden);
    for (size_t i=0;i<hW.size();i++){
        float g = ((float)rand()/RAND_MAX*2-1);
        float v = g*g*g*0.06f;                         // heavy-tailed-ish, ~head magnitudes
        hW[i] = __float2bfloat16(v);
    }
    // K hidden states, RMS-normalized (final-norm output distribution)
    std::vector<float> hX((size_t)K*hidden);
    for (int r=0;r<K;r++){
        double ss=0; for(int h=0;h<hidden;h++){ float g=((float)rand()/RAND_MAX*2-1); hX[(size_t)r*hidden+h]=g; ss+=(double)g*g; }
        float inv = (float)(1.0/sqrt(ss/hidden+1e-6));
        for(int h=0;h<hidden;h++) hX[(size_t)r*hidden+h]*=inv;
    }

    __nv_bfloat16 *dW; uint8_t *dQ; float *dScale, *dX, *dXt, *dYbf, *dYfp8, *dYfp8b;
    CK(cudaMalloc(&dW,(size_t)vocab*hidden*sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&dQ,(size_t)vocab*hidden));
    CK(cudaMalloc(&dScale,(size_t)vocab*sizeof(float)));
    CK(cudaMalloc(&dX,(size_t)K*hidden*sizeof(float)));
    CK(cudaMalloc(&dXt,(size_t)hidden*K*sizeof(float)));
    CK(cudaMalloc(&dYbf,(size_t)K*vocab*sizeof(float)));
    CK(cudaMalloc(&dYfp8,(size_t)K*vocab*sizeof(float)));
    CK(cudaMalloc(&dYfp8b,(size_t)K*vocab*sizeof(float)));
    CK(cudaMemcpy(dW,hW.data(),hW.size()*sizeof(__nv_bfloat16),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX,hX.data(),hX.size()*sizeof(float),cudaMemcpyHostToDevice));

    // Quantize per-row
    fp8_head_quantize_launch(dQ, dScale, dW, vocab, hidden, 0);
    CK(cudaDeviceSynchronize());

    // BF16 head (oracle) + FP8 single + FP8 batched, per token
    nvfp4_xT_launch(dXt, dX, hidden, K, 0);
    fp8_head_gemv_batched_launch(dYfp8b, dQ, dScale, dXt, hidden, vocab, K, 0);
    for (int r=0;r<K;r++){
        bf16_head_gemv1_launch(dYbf +(size_t)r*vocab, dW, dX+(size_t)r*hidden, hidden, vocab, 0);
        fp8_head_gemv_launch (dYfp8+(size_t)r*vocab, dQ, dScale, dX+(size_t)r*hidden, hidden, vocab, 0);
    }
    CK(cudaDeviceSynchronize());

    std::vector<float> Ybf((size_t)K*vocab), Yfp8((size_t)K*vocab), Yfp8b((size_t)K*vocab);
    CK(cudaMemcpy(Ybf.data(),  dYbf,  Ybf.size()*sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Yfp8.data(), dYfp8, Yfp8.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Yfp8b.data(),dYfp8b,Yfp8b.size()*sizeof(float),cudaMemcpyDeviceToHost));

    int top1_ok=0; double worstL2=0; int batched_argmax_ok=0;
    for (int r=0;r<K;r++){
        const float* bf  = Ybf.data() +(size_t)r*vocab;
        const float* fp  = Yfp8.data()+(size_t)r*vocab;
        const float* fpb = Yfp8b.data()+(size_t)r*vocab;
        int a_bf=argmax(bf,vocab), a_fp=argmax(fp,vocab), a_fpb=argmax(fpb,vocab);
        if (a_bf==a_fp) top1_ok++;
        if (a_bf==a_fpb) batched_argmax_ok++;
        double num=0,den=0; for(int v=0;v<vocab;v++){ double d=fp[v]-bf[v]; num+=d*d; den+=(double)bf[v]*bf[v]; }
        double l2 = den>0? sqrt(num/den):0; if(l2>worstL2) worstL2=l2;
    }
    printf("FP8 head  vocab=%d hidden=%d K=%d\n", vocab, hidden, K);
    printf("  top-1 argmax match (single FP8 vs BF16):  %d/%d  %s\n", top1_ok, K, top1_ok==K?"PASS":"*** FAIL ***");
    printf("  top-1 argmax match (batched FP8 vs BF16): %d/%d  %s\n", batched_argmax_ok, K, batched_argmax_ok==K?"PASS":"*** FAIL ***");
    printf("  worst logit L2rel = %.5f%%\n", 100*worstL2);

    // Bandwidth: head bytes BF16 = vocab*hidden*2 ; FP8 = vocab*hidden*1 + vocab*4 (scales)
    cudaEvent_t ev0,ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
    int iters=20;
    CK(cudaEventRecord(ev0,0));
    for(int it=0;it<iters;it++) bf16_head_gemv1_launch(dYbf,dW,dX,hidden,vocab,0);
    CK(cudaEventRecord(ev1,0)); CK(cudaEventSynchronize(ev1));
    float msbf; CK(cudaEventElapsedTime(&msbf,ev0,ev1)); msbf/=iters;
    CK(cudaEventRecord(ev0,0));
    for(int it=0;it<iters;it++) fp8_head_gemv_launch(dYfp8,dQ,dScale,dX,hidden,vocab,0);
    CK(cudaEventRecord(ev1,0)); CK(cudaEventSynchronize(ev1));
    float msfp; CK(cudaEventElapsedTime(&msfp,ev0,ev1)); msfp/=iters;
    double bf_bytes=(double)vocab*hidden*2;
    double fp_bytes=(double)vocab*hidden*1 + (double)vocab*4;
    printf("  single-token head:  BF16 %.4f ms (%.1f GB/s) | FP8 %.4f ms (%.1f GB/s) | %.2fx faster\n",
           msbf, bf_bytes/(msbf/1e3)/1e9, msfp, fp_bytes/(msfp/1e3)/1e9, msbf/msfp);
    printf("  head bytes/token: BF16 %.0f MB | FP8 %.0f MB | saving %.0f MB (%.2fx less)\n",
           bf_bytes/1e6, fp_bytes/1e6, (bf_bytes-fp_bytes)/1e6, bf_bytes/fp_bytes);

    bool pass = (top1_ok==K) && (batched_argmax_ok==K);
    return pass?0:1;
}
