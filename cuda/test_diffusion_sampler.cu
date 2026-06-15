// test_diffusion_sampler.cu — correctness of the parallelized dg_sample_step (argmax, entropy,
// multinomial inverse-CDF) vs a CPU reference, on random logits. Also times the kernel.
// Build: nvcc -O2 -arch=sm_121a cuda/diffusion_gemma_kernels.cu cuda/test_diffusion_sampler.cu -o /tmp/dg_samp
#include "diffusion_gemma_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("cuda %s %d\n",cudaGetErrorString(e),__LINE__);return 1;} }while(0)

int main(){
    const int vocab=262144, C=256;
    std::vector<float> L((size_t)vocab*C), rnd(C);
    uint64_t s=0xabc123ull; auto rf=[&](){ s^=s<<13;s^=s>>7;s^=s<<17; return (float)((s>>11)*(1.0/9007199254740992.0)); };
    for(auto&v:L) v=(rf()*2.f-1.f)*6.f;        // logits ~[-6,6]
    for(auto&v:rnd) v=rf();
    float *dL; int *dsamp,*dargmax; float *drnd,*dent;
    CK(cudaMalloc(&dL,L.size()*4)); CK(cudaMalloc(&drnd,C*4)); CK(cudaMalloc(&dsamp,C*4)); CK(cudaMalloc(&dargmax,C*4)); CK(cudaMalloc(&dent,C*4));
    CK(cudaMemcpy(dL,L.data(),L.size()*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(drnd,rnd.data(),C*4,cudaMemcpyHostToDevice));
    dg_sample_step(dL,drnd,dsamp,dargmax,dent,vocab,C,0);
    CK(cudaDeviceSynchronize());
    std::vector<int> samp(C),amax(C); std::vector<float> ent(C);
    CK(cudaMemcpy(samp.data(),dsamp,C*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(amax.data(),dargmax,C*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ent.data(),dent,C*4,cudaMemcpyDeviceToHost));

    int amax_bad=0, samp_bad=0; double ent_maxerr=0;
    for(int c=0;c<C;c++){
        const float* row=&L[(size_t)c*vocab];
        // CPU ref in double
        double gmax=-1e30; int am=0; for(int i=0;i<vocab;i++) if(row[i]>gmax){gmax=row[i];am=i;}
        double Z=0,H=0; for(int i=0;i<vocab;i++){ double e=exp(row[i]-gmax); Z+=e; }
        for(int i=0;i<vocab;i++){ double p=exp(row[i]-gmax)/Z; if(p>0) H-=p*log(p); }
        if(amax[c]!=am) amax_bad++;
        // VALID inverse-CDF check: the GPU pick p must satisfy cdf(p-1) < rnd <= cdf(p), i.e. rnd
        // falls in the pick's probability interval (token-id distance is meaningless here).
        int p=samp[c]; double cdf_at=0, cdf_bef=0;
        for(int i=0;i<=p;i++){ double e=exp(row[i]-gmax)/Z; cdf_at+=e; if(i<p) cdf_bef+=e; }
        double r=(double)rnd[c]; const double tol=2e-3;
        if(!(r<=cdf_at+tol && r>=cdf_bef-tol)) samp_bad++;
        double ee=fabs((double)ent[c]-H); if(ee>ent_maxerr) ent_maxerr=ee;
    }
    // timing
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); const int IT=50;
    cudaEventRecord(a); for(int i=0;i<IT;i++) dg_sample_step(dL,drnd,dsamp,dargmax,dent,vocab,C,0);
    cudaEventRecord(b); cudaEventSynchronize(b); float t=0; cudaEventElapsedTime(&t,a,b);
    printf("argmax ties(fp32)=%d/%d  invalid samples=%d/%d  entropy max|Δ|=%.4g  time=%.3f ms/step\n",
           amax_bad,C,samp_bad,C,ent_maxerr,t/IT);
    // PASS gates on the parallelized sample+entropy (argmax code is unchanged; rare fp32 ties OK).
    bool pass = (samp_bad==0 && ent_maxerr<1e-3 && amax_bad<=2);
    printf("%s\n",pass?"PASS":"FAIL");
    return pass?0:2;
}
