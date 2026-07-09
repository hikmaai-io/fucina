// Opt-in Qwen3.5 Jacobian-lens readout and intervention support.
// Debug-only: none of these allocations or launches exist unless gemma4_engine_jspace_load runs.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

__global__ void q35_jspace_matvec_kernel(float *out, const __half *J, const float *x, int H) {
    int o=blockIdx.x, tid=threadIdx.x;
    if (o>=H) return;
    float sum=0.f;
    const __half *row=J+(size_t)o*H;
    for (int i=tid; i<H; i+=blockDim.x) sum += __half2float(row[i])*x[i];
    __shared__ float red[256]; red[tid]=sum; __syncthreads();
    for (int s=128; s; s>>=1) { if (tid<s) red[tid]+=red[tid+s]; __syncthreads(); }
    if (tid==0) out[o]=red[0];
}

__global__ void q35_jspace_tmatvec_kernel(float *out, const __half *J, const float *w, int H) {
    int i=blockIdx.x, tid=threadIdx.x;
    if (i>=H) return;
    float sum=0.f;
    for (int o=tid; o<H; o+=blockDim.x) sum += __half2float(J[(size_t)o*H+i])*w[o];
    __shared__ float red[256]; red[tid]=sum; __syncthreads();
    for (int s=128; s; s>>=1) { if (tid<s) red[tid]+=red[tid+s]; __syncthreads(); }
    if (tid==0) out[i]=red[0];
}

__global__ void q35_jspace_normalize_kernel(float *x, int H) {
    int tid=threadIdx.x; float sum=0.f;
    for (int i=tid; i<H; i+=blockDim.x) sum += x[i]*x[i];
    __shared__ float red[256]; red[tid]=sum; __syncthreads();
    for (int s=128; s; s>>=1) { if (tid<s) red[tid]+=red[tid+s]; __syncthreads(); }
    float inv=rsqrtf(fmaxf(red[0],1e-20f));
    for (int i=tid; i<H; i+=blockDim.x) x[i]*=inv;
}

__global__ void q35_jspace_bf16_row_kernel(float *out, const __nv_bfloat16 *table,
                                            int token, int H) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<H) out[i]=__bfloat162float(table[(size_t)token*H+i]);
}

// Runtime-controlled so a captured graph remains valid when the REPL changes steering commands.
__global__ void q35_jspace_apply_kernel(float *x, int rows, int H, int layer,
                                         const float *dirs, const int *mask,
                                         const float *strength) {
    int r=blockIdx.x, tid=threadIdx.x;
    if (r>=rows || !mask[layer]) return;
    float a=*strength;
    if (a==0.f) return;
    float *xr=x+(size_t)r*H; const float *d=dirs+(size_t)layer*H;
    float sum=0.f;
    for (int i=tid; i<H; i+=blockDim.x) sum += xr[i]*xr[i];
    __shared__ float red[256]; red[tid]=sum; __syncthreads();
    for (int s=128; s; s>>=1) { if (tid<s) red[tid]+=red[tid+s]; __syncthreads(); }
    float scale=a*sqrtf(fmaxf(red[0],0.f));
    for (int i=tid; i<H; i+=blockDim.x) xr[i] += scale*d[i];
}

static inline void q35_jspace_after_layer(gemma4_engine_t *eng, float *x, int rows,
                                           int layer, cudaStream_t st) {
    if (!eng->q35.jspace_enabled) return;
    q35_jspace_apply_kernel<<<rows,256,0,st>>>(x, rows, eng->cfg.hidden_size, layer,
        eng->q35.jspace_dirs, eng->q35.jspace_steer_mask, eng->q35.jspace_steer_strength);
    if (eng->q35.jspace_layer_enabled[layer])
        cudaMemcpyAsync(eng->q35.jspace_hidden+(size_t)layer*eng->cfg.hidden_size,
                        x+(size_t)(rows-1)*eng->cfg.hidden_size,
                        (size_t)eng->cfg.hidden_size*sizeof(float), cudaMemcpyDeviceToDevice, st);
}

static void q35_jspace_release(gemma4_engine_t *eng) {
    for (int l=0; l<GEMMA4_CAP_LAYERS; l++) {
        if (eng->q35.jspace_J[l]) cudaFree(eng->q35.jspace_J[l]);
        eng->q35.jspace_J[l]=NULL; eng->q35.jspace_layer_enabled[l]=0;
    }
    void *ptrs[] = {eng->q35.jspace_hidden,eng->q35.jspace_transport,eng->q35.jspace_norm,
                    eng->q35.jspace_logits,eng->q35.jspace_dirs,
                    eng->q35.jspace_steer_mask,eng->q35.jspace_steer_strength};
    for (void *p:ptrs) if (p) cudaFree(p);
    eng->q35.jspace_hidden=eng->q35.jspace_transport=eng->q35.jspace_norm=NULL;
    eng->q35.jspace_logits=eng->q35.jspace_dirs=NULL;
    eng->q35.jspace_steer_mask=NULL; eng->q35.jspace_steer_strength=NULL;
    eng->q35.jspace_enabled=eng->q35.jspace_nlayers=0; eng->q35.jspace_bytes=0;
}

// FJSPACE1 format: magic[8], version/u32, d_model/u32, model_layers/u32,
// entries/u32, n_prompts/u32, then repeated {layer/i32,reserved/u32,J[H,H]/f16 row-major}.
extern "C" int gemma4_engine_jspace_load(gemma4_engine_t *eng, const char *path, int topk) {
    if (!eng || !path || eng->cfg.arch!=GEMMA4_ARCH_QWEN3_5 || eng->q35.jspace_enabled) return -1;
    FILE *f=fopen(path,"rb"); if (!f) return -1;
    char magic[8]; uint32_t version=0,H=0,L=0,E=0,np=0;
    bool ok=fread(magic,1,8,f)==8 && fread(&version,4,1,f)==1 && fread(&H,4,1,f)==1 &&
            fread(&L,4,1,f)==1 && fread(&E,4,1,f)==1 && fread(&np,4,1,f)==1;
    ok = ok && memcmp(magic,"FJSPACE1",8)==0 && version==1 && H==(uint32_t)eng->cfg.hidden_size &&
         L==(uint32_t)eng->cfg.n_layers && E>0 && E<=(uint32_t)eng->cfg.n_layers;
    std::vector<uint16_t> host;
    if (ok) host.resize((size_t)H*H);
    size_t bytes=0;
    for (uint32_t e=0; ok && e<E; e++) {
        int32_t layer=-1; uint32_t reserved=0;
        ok=fread(&layer,4,1,f)==1 && fread(&reserved,4,1,f)==1 && layer>=0 &&
           layer<(int)L && !eng->q35.jspace_J[layer] &&
           fread(host.data(),sizeof(uint16_t),host.size(),f)==host.size();
        if (!ok) break;
        ok=cudaMalloc(&eng->q35.jspace_J[layer],host.size()*sizeof(uint16_t))==cudaSuccess;
        if (ok) ok=cudaMemcpy(eng->q35.jspace_J[layer],host.data(),host.size()*sizeof(uint16_t),
                              cudaMemcpyHostToDevice)==cudaSuccess;
        if (ok) { eng->q35.jspace_layer_enabled[layer]=1; eng->q35.jspace_nlayers++; bytes+=host.size()*2; }
    }
    fclose(f);
    const int V=eng->cfg.vocab_size;
    if (ok) ok=cudaMalloc(&eng->q35.jspace_hidden,(size_t)L*H*sizeof(float))==cudaSuccess;
    if (ok) ok=cudaMalloc(&eng->q35.jspace_transport,(size_t)H*sizeof(float))==cudaSuccess;
    if (ok) ok=cudaMalloc(&eng->q35.jspace_norm,(size_t)H*sizeof(float))==cudaSuccess;
    if (ok) ok=cudaMalloc(&eng->q35.jspace_logits,(size_t)V*sizeof(float))==cudaSuccess;
    if (ok) ok=cudaMalloc(&eng->q35.jspace_dirs,(size_t)L*H*sizeof(float))==cudaSuccess;
    if (ok) ok=cudaMalloc(&eng->q35.jspace_steer_mask,(size_t)L*sizeof(int))==cudaSuccess;
    if (ok) ok=cudaMalloc(&eng->q35.jspace_steer_strength,sizeof(float))==cudaSuccess;
    if (ok) {
        cudaMemset(eng->q35.jspace_hidden,0,(size_t)L*H*sizeof(float));
        cudaMemset(eng->q35.jspace_dirs,0,(size_t)L*H*sizeof(float));
        cudaMemset(eng->q35.jspace_steer_mask,0,(size_t)L*sizeof(int));
        cudaMemset(eng->q35.jspace_steer_strength,0,sizeof(float));
        bytes += ((size_t)2*L*H+2*H+V)*sizeof(float)+(size_t)L*sizeof(int)+sizeof(float);
        eng->q35.jspace_topk=topk<1?8:(topk>32?32:topk);
        eng->q35.jspace_bytes=bytes; eng->q35.jspace_enabled=1;
        eng->q35.committed_bytes+=bytes; eng->q35.reserved_bytes+=bytes;
        if (eng->q35.committed_bytes>eng->q35.peak_bytes) eng->q35.peak_bytes=eng->q35.committed_bytes;
        fprintf(stderr,"fucina: J-space debug enabled: %d layers, H=%u, lens=%.2f GiB, n_prompts=%u\n",
                eng->q35.jspace_nlayers,H,bytes/(1024.0*1024*1024),np);
        return 0;
    }
    q35_jspace_release(eng); cudaGetLastError(); return -1;
}

extern "C" int gemma4_engine_jspace_snapshot(gemma4_engine_t *eng, int max_layers, int max_topk,
                                                int *layers, int *token_ids, float *probs) {
    if (!eng || !eng->q35.jspace_enabled || !layers || !token_ids || !probs) return -1;
    const int H=eng->cfg.hidden_size,V=eng->cfg.vocab_size,K=std::min(max_topk,eng->q35.jspace_topk);
    if (K<1) return -1;
    std::vector<float> logits(V); std::vector<int> order(V);
    int nout=0; cudaStream_t st=eng->stream;
    const float *normw=(const float*)(eng->d_weights+(eng->tensors.output_norm-eng->tdata_start));
    for (int l=0; l<eng->cfg.n_layers && nout<max_layers; l++) if (eng->q35.jspace_layer_enabled[l]) {
        q35_jspace_matvec_kernel<<<H,256,0,st>>>(eng->q35.jspace_transport,eng->q35.jspace_J[l],
                                                 eng->q35.jspace_hidden+(size_t)l*H,H);
        rms_norm_rows_kernel<<<1,256,32*sizeof(float),st>>>(eng->q35.jspace_norm,
            eng->q35.jspace_transport,normw,H,1,1e-6f);
        if (eng->format==FORMAT_FP8_BLOCK)
            bf16_head_gemv_launch(eng->q35.jspace_logits,eng->d_lmhead_bf16,
                                   eng->q35.jspace_norm,H,V,st);
        else
            logits_head(eng,eng->q35.jspace_logits,eng->q35.jspace_norm,H,V,st);
        cudaMemcpyAsync(logits.data(),eng->q35.jspace_logits,(size_t)V*sizeof(float),
                        cudaMemcpyDeviceToHost,st);
        if (cudaStreamSynchronize(st)!=cudaSuccess) return -1;
        for (int i=0;i<V;i++) order[i]=i;
        std::partial_sort(order.begin(),order.begin()+K,order.end(),
                          [&](int a,int b){ return logits[a]>logits[b]; });
        float m=*std::max_element(logits.begin(),logits.end()); double sum=0.0;
        for (float z:logits) sum+=std::exp((double)z-m);
        layers[nout]=l;
        for (int k=0;k<K;k++) {
            int id=order[k]; token_ids[nout*K+k]=id;
            probs[nout*K+k]=(float)(std::exp((double)logits[id]-m)/sum);
        }
        nout++;
    }
    return nout;
}

extern "C" int gemma4_engine_jspace_steer(gemma4_engine_t *eng, int token_id, float strength,
                                             const int *layers, int n_layers) {
    if (!eng || !eng->q35.jspace_enabled || token_id<0 || token_id>=eng->cfg.vocab_size) return -1;
    const int H=eng->cfg.hidden_size,L=eng->cfg.n_layers;
    std::vector<int> mask(L,0); cudaStream_t st=eng->stream;
    if (n_layers<=0) for (int l=0;l<L;l++) mask[l]=eng->q35.jspace_layer_enabled[l]?1:0;
    else for (int i=0;i<n_layers;i++) if (layers[i]>=0 && layers[i]<L &&
             eng->q35.jspace_layer_enabled[layers[i]]) mask[layers[i]]=1;
    // Recover the unembedding direction. Qwen GGUF uses tied embeddings; safetensors exposes a
    // resident BF16 LM head. Both paths produce the same fp32 vector expected by J_l^T @ w_t.
    if (eng->format==FORMAT_FP8_BLOCK || eng->format==FORMAT_NVFP4) {
        if (!eng->d_lmhead_bf16) return -1;
        q35_jspace_bf16_row_kernel<<<(H+255)/256,256,0,st>>>(eng->q35.jspace_transport,
                                                             eng->d_lmhead_bf16,token_id,H);
    } else {
        int32_t *dtok=nullptr; if (cudaMalloc(&dtok,sizeof(int32_t))!=cudaSuccess) return -1;
        cudaMemcpyAsync(dtok,&token_id,sizeof(int32_t),cudaMemcpyHostToDevice,st);
        embed_w(eng,eng->q35.jspace_transport,eng->d_token_embd,dtok,1,H,st);
        cudaFree(dtok);
    }
    for (int l=0;l<L;l++) if (mask[l]) {
        q35_jspace_tmatvec_kernel<<<H,256,0,st>>>(eng->q35.jspace_dirs+(size_t)l*H,
            eng->q35.jspace_J[l],eng->q35.jspace_transport,H);
        q35_jspace_normalize_kernel<<<1,256,0,st>>>(eng->q35.jspace_dirs+(size_t)l*H,H);
    }
    strength=fmaxf(-1.f,fminf(1.f,strength));
    cudaMemcpyAsync(eng->q35.jspace_steer_mask,mask.data(),(size_t)L*sizeof(int),cudaMemcpyHostToDevice,st);
    cudaMemcpyAsync(eng->q35.jspace_steer_strength,&strength,sizeof(float),cudaMemcpyHostToDevice,st);
    return cudaStreamSynchronize(st)==cudaSuccess?0:-1;
}

extern "C" void gemma4_engine_jspace_clear_steer(gemma4_engine_t *eng) {
    if (!eng || !eng->q35.jspace_enabled) return;
    cudaMemsetAsync(eng->q35.jspace_steer_mask,0,(size_t)eng->cfg.n_layers*sizeof(int),eng->stream);
    cudaMemsetAsync(eng->q35.jspace_steer_strength,0,sizeof(float),eng->stream);
}
