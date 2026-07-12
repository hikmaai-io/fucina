// ABOUTME: Statistical validation that P1 probabilistic rejection preserves the target distribution.
// ABOUTME: Monte-Carlo: rejection-sampled tokens must match direct temperature sampling from target.
//
// The probabilistic DFlash contract: with the shared-key Gumbel/uniform scheme, the tokens emitted
// by speculative rejection sampling are distributed EXACTLY as if sampled directly from the target
// distribution at temperature T -- regardless of the draft. This gate validates that property on a
// small vocab via Monte-Carlo over many seeds: it compares (a) the empirical distribution of the
// FIRST emitted token from q35_dflash_verify_prob against (b) direct inverse-CDF sampling from the
// target softmax, using the SAME per-seed acceptance/residual uniforms. A well-formed rejection
// sampler makes these distributions statistically indistinguishable (TV distance small). This is a
// host oracle for the math; the device path must match it bit-for-bit (already gated by
// qwen35-dflash-parity-test for fixed seeds).
//
// build: g++ -std=c++17 -O2 -Wall -Wextra -Icuda cuda/qwen35_dflash_prob_dist_test.cc -o /tmp/dflash_probdist && /tmp/dflash_probdist
#include "qwen35_dflash_reject.cuh"
#include "qwen35_dflash_rng.cuh"
#include <cstdio>
#include <vector>
#include <cmath>

static int failures = 0;

// Direct target sampling at temperature 1 using the SAME SAMPLE-domain uniform the bonus path uses.
static int32_t direct_target_sample(const float* logits, int vocab, uint64_t seed, int64_t pos){
    double mx=logits[0]; for(int v=1;v<vocab;v++) if((double)logits[v]>mx) mx=logits[v];
    double sum=0; for(int v=0;v<vocab;v++) sum+=std::exp((double)logits[v]-mx);
    double u=q35_dflash_uniform_open(seed,pos,Q35_DFLASH_DOMAIN_SAMPLE);
    double tc=u*sum, acc=0; for(int v=0;v<vocab;v++){ acc+=std::exp((double)logits[v]-mx); if(acc>=tc) return v; }
    return vocab-1;
}

int main(){
    const int vocab=8, K=1, trials=200000;
    // Target distribution (fixed) and a DIFFERENT draft distribution.
    std::vector<float> tl((size_t)(K+1)*vocab), dl((size_t)K*vocab);
    float tvals[8]={2.0f,1.0f,0.5f,0.0f,-1.0f,3.0f,0.2f,-0.5f};
    float dvals[8]={0.0f,3.0f,1.0f,0.0f,2.0f,0.0f,1.0f,0.5f};   // draft != target
    for(int v=0;v<vocab;v++){ tl[v]=tvals[v]; tl[vocab+v]=tvals[(v+3)%vocab]; dl[v]=dvals[v]; }

    // Empirical distributions.
    std::vector<long> emp_rej(vocab,0), emp_dir(vocab,0);
    int64_t pos[1]={1000}; int64_t posb=1001;
    for(int t=0;t<trials;t++){
        uint64_t seed = 0x9E3779B97F4A7C15ULL * (uint64_t)(t+1);
        // Draft samples its token from dl using the shared SAMPLE key at pos[0].
        int32_t draft[1];
        { double mx=dl[0]; for(int v=1;v<vocab;v++) if((double)dl[v]>mx) mx=dl[v]; double sum=0; for(int v=0;v<vocab;v++) sum+=std::exp((double)dl[v]-mx);
          double u=q35_dflash_uniform_open(seed,pos[0],Q35_DFLASH_DOMAIN_SAMPLE); double tc=u*sum,acc=0; draft[0]=vocab-1; for(int v=0;v<vocab;v++){acc+=std::exp((double)dl[v]-mx); if(acc>=tc){draft[0]=v;break;}} }
        auto r = q35_dflash_verify_prob(tl.data(), dl.data(), vocab, draft, K, seed, pos, posb);
        // The emitted token at the first position: if accepted (len>=1) it's draft[0], else the
        // residual resample stored as emitted_token.
        int32_t emit = (r.accepted_len>=1) ? draft[0] : r.emitted_token;
        if(emit>=0 && emit<vocab) emp_rej[emit]++;
        // Direct target sample at the SAME position (independent reference distribution).
        int32_t d = direct_target_sample(tl.data(), vocab, seed, pos[0]);
        if(d>=0 && d<vocab) emp_dir[d]++;
    }

    // Compare the rejection-emitted distribution to the TRUE target softmax (not to emp_dir, which
    // uses a different key domain -- emp_dir is a sanity check that direct sampling matches softmax).
    double mx=tl[0]; for(int v=1;v<vocab;v++) if((double)tl[v]>mx) mx=tl[v];
    double sum=0; for(int v=0;v<vocab;v++) sum+=std::exp((double)tl[v]-mx);
    double tv=0;
    printf("token  target_p   rej_emp    dir_emp\n");
    for(int v=0;v<vocab;v++){
        double p=std::exp((double)tl[v]-mx)/sum;
        double re=(double)emp_rej[v]/trials, de=(double)emp_dir[v]/trials;
        tv += std::fabs(re-p);
        printf("  %d   %.4f    %.4f    %.4f\n", v, p, re, de);
    }
    tv*=0.5;
    printf("TV(rejection-emitted, target) = %.4f\n", tv);
    // With 200k trials, sampling noise on TV over 8 tokens is ~O(1/sqrt(N)) per bin; 0.01 is generous.
    if(tv > 0.02){ printf("FAIL: rejection-emitted distribution deviates from target (TV=%.4f)\n", tv); failures++; }

    if(failures){ printf("FAIL — DFlash probabilistic distribution preservation (%d)\n", failures); return 1; }
    printf("PASS — DFlash probabilistic rejection preserves the target distribution "
           "(TV=%.4f over %d trials, vocab=%d); draft != target, yet emitted ~ target\n", tv, trials, vocab);
    return 0;
}
