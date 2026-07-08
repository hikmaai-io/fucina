// test_q4k_wmma.cu — correctness + micro-bench of the grouped Q4_K WMMA GEMM vs a host
// reference (dequant + f64 dot). Self-contained: carries its own tiny float→Q4_K encoder
// (same ggml block_q4_K layout as the engine's q35_f32_to_q4_K_host).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "q4k_wmma.cuh"

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

static float frand(float lo, float hi){ return lo + (hi-lo)*(rand()/(float)RAND_MAX); }
static uint16_t f2h(float f){ __half h=__float2half(f); uint16_t u; memcpy(&u,&h,2); return u; }

static void enc_q4k(const float *src, unsigned char *dst, int64_t n_super) {
    for (int64_t sb = 0; sb < n_super; sb++) {
        const float *w = src + sb*256; unsigned char *blk = dst + sb*144;
        float scales[8], mins[8];
        for (int j=0;j<8;j++){ const float*v=w+j*32; float lo=v[0],hi=v[0];
            for(int k=1;k<32;k++){lo=fminf(lo,v[k]);hi=fmaxf(hi,v[k]);}
            float mn=fmaxf(0.f,-lo); scales[j]=fmaxf((hi+mn)/15.f,0.f); mins[j]=mn; }
        float dmax=0,mmax=0; for(int j=0;j<8;j++){dmax=fmaxf(dmax,scales[j]);mmax=fmaxf(mmax,mins[j]);}
        float d=dmax/63.f, dm=mmax/63.f, id=d>0?1.f/d:0.f, im=dm>0?1.f/dm:0.f;
        uint8_t ls6[8],lm6[8];
        for(int j=0;j<8;j++){ int ls=(int)lrintf(scales[j]*id); ls6[j]=(uint8_t)(ls<0?0:ls>63?63:ls);
            int lm=(int)lrintf(mins[j]*im); lm6[j]=(uint8_t)(lm<0?0:lm>63?63:lm); }
        uint16_t hd=f2h(d), hm=f2h(dm);
        blk[0]=hd&0xFF; blk[1]=hd>>8; blk[2]=hm&0xFF; blk[3]=hm>>8;
        uint8_t*sc8=blk+4; for(int j=0;j<12;j++)sc8[j]=0;
        for(int j=0;j<8;j++){ if(j<4){sc8[j]|=ls6[j];sc8[j+4]|=lm6[j];}
            else{sc8[j+4]=(uint8_t)((ls6[j]&0xF)|((lm6[j]&0xF)<<4));
                 sc8[j-4]|=(uint8_t)((ls6[j]>>4)<<6); sc8[j]|=(uint8_t)((lm6[j]>>4)<<6);} }
        uint8_t*qs=blk+16;
        for(int g=0;g<4;g++)for(int m=0;m<32;m++){
            int q2[2];
            for(int h=0;h<2;h++){ int j=2*g+h; float ds=d*(float)ls6[j],dmn=dm*(float)lm6[j];
                float xx=w[j*32+m]; int q= ds>0?(int)lrintf((xx+dmn)/ds):0;
                q2[h]=q<0?0:q>15?15:q; }
            qs[g*32+m]=(uint8_t)(q2[0]|(q2[1]<<4)); }
    }
}
static void dec_q4k(const unsigned char *blk0, float *out, int64_t n_super){
    for(int64_t sb=0;sb<n_super;sb++){ const unsigned char*blk=blk0+sb*144; float*o=out+sb*256;
        __half hd,hm; memcpy(&hd,blk,2); memcpy(&hm,blk+2,2);
        float d=__half2float(hd), dm=__half2float(hm);
        for(int j=0;j<8;j++){ int s,m;
            if(j<4){s=blk[4+j]&63;m=blk[4+j+4]&63;}
            else{s=(blk[4+j+4]&0xF)|((blk[4+j-4]>>6)<<4); m=(blk[4+j+4]>>4)|((blk[4+j]>>6)<<4);}
            const unsigned char*qb=blk+16+(j>>1)*32; int sh=(j&1)?4:0;
            for(int k=0;k<32;k++) o[j*32+k]=d*(float)s*(float)((qb[k]>>sh)&0xF)-dm*(float)m; } }
}

int main(){
    srand(11);
    const int E=64, IN=2048, OUT=512;   // decode-like: 64 active experts, 1-2 tokens each
    int cnt[E]; for(int e=0;e<E;e++) cnt[e]=16;
    int coloff[E], total=0; for(int e=0;e<E;e++){coloff[e]=total; total+=cnt[e];}
    const size_t per_w=(size_t)OUT*IN, per_q=per_w/256*144;

    std::vector<float> wf((size_t)E*per_w);
    for(auto&v:wf)v=frand(-0.5f,0.5f);
    std::vector<unsigned char> wq((size_t)E*per_q);
    for(int e=0;e<E;e++) enc_q4k(wf.data()+(size_t)e*per_w, wq.data()+(size_t)e*per_q, per_w/256);
    // reference uses the DEQUANTIZED values (so only the GEMM path is under test)
    std::vector<float> wd((size_t)E*per_w);
    for(int e=0;e<E;e++) dec_q4k(wq.data()+(size_t)e*per_q, wd.data()+(size_t)e*per_w, per_w/256);

    std::vector<float> x((size_t)total*IN);
    for(auto&v:x)v=frand(-1.f,1.f);
    std::vector<float> ref((size_t)total*OUT,0.f);
    for(int e=0;e<E;e++)for(int t=0;t<cnt[e];t++)for(int o=0;o<OUT;o++){
        double s=0; const float*wr=wd.data()+(size_t)e*per_w+(size_t)o*IN;
        const float*xr=x.data()+(size_t)(coloff[e]+t)*IN;
        for(int i=0;i<IN;i++)s+=(double)wr[i]*xr[i];
        ref[(size_t)(coloff[e]+t)*OUT+o]=(float)s; }

    unsigned char*d_w; float*d_x,*d_o; int*d_co,*d_cn;
    CK(cudaMalloc(&d_w,wq.size())); CK(cudaMalloc(&d_x,x.size()*4));
    CK(cudaMalloc(&d_o,(size_t)total*OUT*4));
    CK(cudaMalloc(&d_co,E*4)); CK(cudaMalloc(&d_cn,E*4));
    CK(cudaMemcpy(d_w,wq.data(),wq.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_co,coloff,E*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_cn,cnt,E*4,cudaMemcpyHostToDevice));

    q4k_wmma_grouped_launch(d_o,d_w,(int64_t)per_q,d_x,d_co,d_cn,NULL,0,E,IN,OUT,0);
    CK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)total*OUT);
    CK(cudaMemcpy(got.data(),d_o,got.size()*4,cudaMemcpyDeviceToHost));

    double dot=0,na=0,nb=0,maxrel=0;
    for(size_t i=0;i<got.size();i++){ dot+=(double)got[i]*ref[i]; na+=(double)got[i]*got[i]; nb+=(double)ref[i]*ref[i];
        if(fabs((double)ref[i])>0.5) maxrel=fmax(maxrel,fabs((double)got[i]-ref[i])/fabs((double)ref[i])); }
    double cosine=dot/(sqrt(na)*sqrt(nb)+1e-12);
    printf("q4k_wmma_grouped: cosine=%.6f maxrel=%.4f (bf16 mma vs f64 ref)\n", cosine, maxrel);
    int ok = cosine > 0.999 && maxrel < 0.05;

    // micro-bench: 40-layer-like loop
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for(int it=0;it<200;it++)
        q4k_wmma_grouped_launch(d_o,d_w,(int64_t)per_q,d_x,d_co,d_cn,NULL,0,E,IN,OUT,0);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms,t0,t1);
    double bytes=(double)E*per_q*200;  // weight bytes (dominant)
    printf("q4k_wmma_grouped: %.3f ms/launch, weight-BW %.1f GB/s\n", ms/200, bytes/(ms/1e3)/1e9);
    printf("%s\n", ok?"PASS":"FAIL");
    return ok?0:1;
}
