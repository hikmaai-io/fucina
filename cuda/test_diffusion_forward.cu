// test_diffusion_forward.cu — DiffusionGemma single-pass [prompt|canvas] forward, validated
// against llama.cpp (llama-diffusion-gemma-eval) canvas logits.
//
// Mirrors src/models/diffusion-gemma.cpp + gemma4-common.h exactly (zero self-conditioning):
//   prompt rows = embed*sqrt(n_embd); canvas rows = rmsnorm_noscale(embed*sqrt(n_embd))
//   per layer: attn (region mask) -> post_attn_norm -> +res ; dense-FFN + 128-expert MoE (parallel)
//   region scalar (prompt=enc_out_scale, canvas=out_scale); output_norm; tied LM head; softcap 30.
//
// Correctness-first: quantized weights resident, dequant-per-matmul to fp32 + cuBLAS sgemm,
// materialized attention (N is tiny), expert-grouped MoE. Not optimized — this is the parity gate.
//
// Build: nvcc -O2 -arch=sm_121a cuda/diffusion_gemma_kernels.cu cuda/test_diffusion_forward.cu \
//            -lcublas -o /tmp/test_dg_fwd
// Run:   /tmp/test_dg_fwd <model.gguf> /tmp/dg_dequant/prompt.i32 /tmp/dg_dequant/canvas.i32 \
//            /tmp/dg_dequant/ref_logits.bin

#include "diffusion_gemma_kernels.cuh"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s @ %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); exit(1);} } while(0)
#define CB(x) do { cublasStatus_t st=(x); if(st!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"cuBLAS %d @ %s:%d\n",(int)st,__FILE__,__LINE__); exit(1);} } while(0)

// ─── GGUF parsing (minimal: tensor name → type/dims/offset) ─────────────
enum { GT_U8=0,GT_I8=1,GT_U16=2,GT_I16=3,GT_U32=4,GT_I32=5,GT_F32=6,GT_BOOL=7,GT_STR=8,GT_ARR=9,GT_U64=10,GT_I64=11,GT_F64=12 };
static uint64_t scalar_sz(uint32_t t){switch(t){case GT_U8:case GT_I8:case GT_BOOL:return 1;case GT_U16:case GT_I16:return 2;case GT_U32:case GT_I32:case GT_F32:return 4;case GT_U64:case GT_I64:case GT_F64:return 8;default:return 0;}}

struct DGT { uint8_t *dev=nullptr; int type=0; int ndim=0; int64_t ne[4]={1,1,1,1}; int64_t nbytes=0; int64_t nelem=0; uint64_t offset=0; };

static int64_t row_bytes(int type, int64_t ne0){
    int bb=dg_block_bytes(type), bn=dg_block_nelem(type);
    if(type==DG_GGML_F32) return ne0*4;
    if(type==DG_GGML_F16) return ne0*2;
    return (ne0/bn)*bb;
}
static int64_t tensor_bytes(const DGT&t){
    int64_t rb=row_bytes(t.type,t.ne[0]); int64_t rows=1; for(int i=1;i<t.ndim;i++) rows*=t.ne[i]; return rb*rows;
}

int main(int argc, char** argv){
    if(argc<5){ fprintf(stderr,"usage: %s model.gguf prompt.i32 canvas.i32 ref_logits.bin\n",argv[0]); return 1; }
    const char* model_path=argv[1];

    // ── read input ids + reference logits ──
    auto read_i32=[](const char*p){ FILE*f=fopen(p,"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); std::vector<int> v(n/4); if(fread(v.data(),4,v.size(),f)!=v.size()){fprintf(stderr,"read fail %s\n",p);exit(1);} fclose(f); return v; };
    std::vector<int> prompt=read_i32(argv[2]), canvas=read_i32(argv[3]);
    int P=prompt.size(), C=canvas.size(), N=P+C;
    std::vector<int> ids; ids.insert(ids.end(),prompt.begin(),prompt.end()); ids.insert(ids.end(),canvas.begin(),canvas.end());
    FILE* rf=fopen(argv[4],"rb"); fseek(rf,0,SEEK_END); long rsz=ftell(rf); fseek(rf,0,SEEK_SET);
    std::vector<float> ref(rsz/4); if(fread(ref.data(),4,ref.size(),rf)!=ref.size()){fprintf(stderr,"ref read fail\n");return 1;} fclose(rf);
    fprintf(stderr,"P=%d C=%d N=%d  ref=%ld floats (expect C*vocab=%lld)\n",P,C,N,(long)ref.size(),(long long)C*DG_VOCAB);

    // ── mmap GGUF, parse tensor table ──
    int fd=open(model_path,O_RDONLY); struct stat stt; fstat(fd,&stt);
    uint8_t* base=(uint8_t*)mmap(nullptr,stt.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(base==MAP_FAILED){perror("mmap");return 1;}
    const uint8_t* p=base; const uint8_t* end=base+stt.st_size;
    uint32_t magic=*(uint32_t*)p; p+=4; uint32_t ver=*(uint32_t*)p; p+=4;
    if(magic!=0x46554747){fprintf(stderr,"bad magic\n");return 1;}
    uint64_t n_tensors=*(uint64_t*)p; p+=8; uint64_t n_kv=*(uint64_t*)p; p+=8;
    auto rd_str=[&](const uint8_t*&q){ uint64_t l=*(uint64_t*)q; q+=8; std::string s((const char*)q,l); q+=l; return s; };
    auto skip_val=[&](const uint8_t*&q, uint32_t vt){
        if(vt==GT_STR){ uint64_t l=*(uint64_t*)q; q+=8+l; }
        else if(vt==GT_ARR){ uint32_t at=*(uint32_t*)q; q+=4; uint64_t cnt=*(uint64_t*)q; q+=8;
            if(at==GT_STR){ for(uint64_t i=0;i<cnt;i++){ uint64_t l=*(uint64_t*)q; q+=8+l; } }
            else q+=cnt*scalar_sz(at);
        } else q+=scalar_sz(vt);
    };
    for(uint64_t i=0;i<n_kv;i++){ rd_str(p); uint32_t vt=*(uint32_t*)p; p+=4; skip_val(p,vt); }
    std::unordered_map<std::string,DGT> T;
    for(uint64_t i=0;i<n_tensors;i++){
        std::string name=rd_str(p);
        DGT t; t.ndim=*(uint32_t*)p; p+=4;
        for(int d=0;d<t.ndim;d++){ t.ne[d]=*(int64_t*)p; p+=8; }
        t.type=*(uint32_t*)p; p+=4; t.offset=*(uint64_t*)p; p+=8;
        t.nelem=1; for(int d=0;d<t.ndim;d++) t.nelem*=t.ne[d];
        t.nbytes=tensor_bytes(t);
        T[name]=t;
    }
    uint64_t off=(uint64_t)(p-base); off=(off+31)&~31ull; const uint8_t* data=base+off;
    fprintf(stderr,"parsed %lu tensors, data@%lu\n",(unsigned long)n_tensors,(unsigned long)off);

    // ── upload all tensors to device (quantized, resident) ──
    size_t total=0;
    for(auto& kv:T){ DGT&t=kv.second; CK(cudaMalloc(&t.dev,t.nbytes)); CK(cudaMemcpy(t.dev,data+t.offset,t.nbytes,cudaMemcpyHostToDevice)); total+=t.nbytes; }
    fprintf(stderr,"uploaded %.2f GB of weights\n",total/1e9);

    cublasHandle_t cub; CB(cublasCreate(&cub));
    CB(cublasSetMathMode(cub, CUBLAS_PEDANTIC_MATH));   // strict fp32 (no TF32 tensor cores) to match the ref
    auto G=[&](const std::string&n)->DGT&{ auto it=T.find(n); if(it==T.end()){fprintf(stderr,"MISSING tensor %s\n",n.c_str());exit(1);} return it->second; };
    auto has=[&](const std::string&n){ return T.count(n)>0; };

    const int n_embd=DG_HIDDEN, n_ff=DG_FFN_INTERMEDIATE, n_head=DG_HEADS, vocab=DG_VOCAB;
    const float eps=DG_RMS_EPS, esc=sqrtf((float)n_embd);

    // ── scratch ──
    float *d_wscr; CK(cudaMalloc(&d_wscr,(size_t)26'000'000*4));           // weight dequant (≥ global wq 23M)
    float *d_tok;  CK(cudaMalloc(&d_tok,(size_t)vocab*n_embd*4));          // token_embd dequant (2.96 GB)
    CK(cudaDeviceSynchronize());
    { DGT& te=G("token_embd.weight"); if(dg_dequant(te.type,te.dev,te.nelem,d_tok,0)){fprintf(stderr,"tok dequant fail\n");return 1;} }
    CK(cudaDeviceSynchronize());

    int *d_ids; CK(cudaMalloc(&d_ids,N*4)); CK(cudaMemcpy(d_ids,ids.data(),N*4,cudaMemcpyHostToDevice));
    std::vector<int> posh(N); for(int i=0;i<N;i++) posh[i]=i;
    int *d_pos; CK(cudaMalloc(&d_pos,N*4)); CK(cudaMemcpy(d_pos,posh.data(),N*4,cudaMemcpyHostToDevice));

    auto AB=[&](size_t nf){ float*q; CK(cudaMalloc(&q,nf*4)); return q; };
    float *d_inpL=AB((size_t)n_embd*N), *d_cur=AB((size_t)n_embd*N), *d_attnout=AB((size_t)n_embd*N);
    float *d_q=AB((size_t)DG_GLOBAL_HEAD_DIM*n_head*N), *d_kraw=AB((size_t)DG_GLOBAL_HEAD_DIM*n_head*N);
    float *d_k=AB((size_t)DG_GLOBAL_HEAD_DIM*n_head*N), *d_v=AB((size_t)DG_GLOBAL_HEAD_DIM*n_head*N);
    float *d_attn=AB((size_t)DG_GLOBAL_HEAD_DIM*n_head*N);
    float *d_dense=AB((size_t)n_embd*N), *d_moe=AB((size_t)n_embd*N), *d_ffn=AB((size_t)n_embd*N);
    float *d_tmp=AB((size_t)n_ff*N), *d_tmp2=AB((size_t)n_ff*N), *d_rtmp=AB((size_t)n_embd*N);
    float *d_rlogits=AB((size_t)DG_N_EXPERTS*N);
    float *d_moeout=AB((size_t)n_embd*N);
    float *d_xe=AB((size_t)n_embd*N), *d_gu=AB((size_t)2*DG_EXPERT_FFN*N), *d_act=AB((size_t)DG_EXPERT_FFN*N), *d_oe=AB((size_t)n_embd*N);
    int   *d_eidx; CK(cudaMalloc(&d_eidx,N*4)); float *d_ecs; CK(cudaMalloc(&d_ecs,N*4));
    int   *d_topk_idx; CK(cudaMalloc(&d_topk_idx,N*DG_N_EXPERTS_USED*4));
    float *d_topk_w;  CK(cudaMalloc(&d_topk_w,N*DG_N_EXPERTS_USED*4));
    float *d_logits=AB((size_t)vocab*N);

    const float one=1.f, zero=0.f;
    // mm: out[outd,T] = W[outd,in] @ x[in,T].  W given as a DGT (ne0=in, ne1=out) or raw slab.
    auto mm_raw=[&](float* out, const uint8_t* raw, int type, int64_t in, int64_t outd, const float* x, int T_){
        const float* A;
        if(type==DG_GGML_F32) A=(const float*)raw;
        else { if(dg_dequant(type,raw,in*outd,d_wscr,0)){fprintf(stderr,"dequant fail t=%d\n",type);exit(1);} A=d_wscr; }
        CB(cublasSgemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,(int)outd,T_,(int)in,&one,A,(int)in,x,(int)in,&zero,out,(int)outd));
    };
    auto mm=[&](float* out, DGT& W, const float* x, int T_){ mm_raw(out,W.dev,W.type,W.ne[0],W.ne[1],x,T_); };

    // ── embeddings (region) ──
    dg_embed_gather(d_inpL,d_tok,d_ids,n_embd,N,esc,0);
    dg_rmsnorm(d_inpL+(size_t)P*n_embd, d_inpL+(size_t)P*n_embd, nullptr, n_embd, C, eps, 0); // canvas rows
    CK(cudaDeviceSynchronize());

    DGT& rope_freqs = G("rope_freqs.weight");

    std::vector<int>   th_idx(N*DG_N_EXPERTS_USED);
    std::vector<float> th_w(N*DG_N_EXPERTS_USED);
    std::vector<float> pes(DG_N_EXPERTS);

    for(int il=0; il<DG_MAX_LAYERS; il++){
        bool global = (il%6==5);           // global at 5,11,17,23,29
        bool sliding = !global;
        int hd = sliding?DG_HEAD_DIM:DG_GLOBAL_HEAD_DIM;
        int nkv = sliding?DG_KV_HEADS_SLIDING:DG_KV_HEADS_GLOBAL;
        float theta = sliding?DG_ROPE_THETA_SLIDING:DG_ROPE_THETA_GLOBAL;
        const float* ff = sliding?nullptr:(const float*)rope_freqs.dev;
        char b[64];
        auto L=[&](const char* s){ snprintf(b,64,"blk.%d.%s",il,s); return std::string(b); };

        // attn
        dg_rmsnorm(d_cur,d_inpL,(float*)G(L("attn_norm.weight")).dev,n_embd,N,eps,0);
        mm(d_q, G(L("attn_q.weight")), d_cur, N);
        dg_head_rmsnorm(d_q,(float*)G(L("attn_q_norm.weight")).dev,hd,n_head,N,eps,0);
        dg_rope(d_q,d_pos,hd,n_head,N,theta,ff,0);
        mm(d_kraw, G(L("attn_k.weight")), d_cur, N);
        // K = k_norm(kraw) -> rope ; V = v_norm( sliding? vproj : kraw )  (no rope, no scale)
        CK(cudaMemcpy(d_k,d_kraw,(size_t)hd*nkv*N*4,cudaMemcpyDeviceToDevice));
        dg_head_rmsnorm(d_k,(float*)G(L("attn_k_norm.weight")).dev,hd,nkv,N,eps,0);
        dg_rope(d_k,d_pos,hd,nkv,N,theta,ff,0);
        if(sliding){ mm(d_v, G(L("attn_v.weight")), d_cur, N); }
        else { CK(cudaMemcpy(d_v,d_kraw,(size_t)hd*nkv*N*4,cudaMemcpyDeviceToDevice)); }
        dg_head_rmsnorm(d_v,nullptr,hd,nkv,N,eps,0);   // v-norm: no scale
        dg_attention(d_attn,d_q,d_k,d_v,hd,n_head,nkv,N,P,DG_SLIDING_WINDOW,sliding?1:0,0);
        mm(d_cur, G(L("attn_output.weight")), d_attn, N);
        dg_rmsnorm(d_cur,d_cur,(float*)G(L("post_attention_norm.weight")).dev,n_embd,N,eps,0);
        dg_add(d_attnout,d_cur,d_inpL,(int64_t)n_embd*N,0);

        // dense FFN (shared expert)
        dg_rmsnorm(d_cur,d_attnout,(float*)G(L("ffn_norm.weight")).dev,n_embd,N,eps,0);
        mm(d_tmp,  G(L("ffn_gate.weight")), d_cur, N);   // [n_ff,N]
        mm(d_tmp2, G(L("ffn_up.weight")),   d_cur, N);
        dg_gelu_mul(d_tmp,d_tmp,d_tmp2,(int64_t)n_ff*N,0);
        mm(d_dense, G(L("ffn_down.weight")), d_tmp, N);  // [n_embd,N]
        dg_rmsnorm(d_dense,d_dense,(float*)G(L("post_ffw_norm_1.weight")).dev,n_embd,N,eps,0);

        // MoE
        dg_rmsnorm(d_xe,d_attnout,(float*)G(L("pre_ffw_norm_2.weight")).dev,n_embd,N,eps,0); // expert input (reuse d_xe? no, need per-expert gather). use d_moe as moe_in
        float* d_moein=d_rtmp; // borrow; we'll recompute router tmp into d_cur
        dg_rmsnorm(d_moein,d_attnout,(float*)G(L("pre_ffw_norm_2.weight")).dev,n_embd,N,eps,0);
        // router: rmsnorm_noscale(attn_out) * 1/sqrt(n_embd) * gate_inp_s ; logits = gate_inp @ tmp
        dg_rmsnorm(d_cur,d_attnout,nullptr,n_embd,N,eps,0);
        dg_scale(d_cur,(int64_t)n_embd*N,1.0f/sqrtf((float)n_embd),0);
        dg_mul_vec_cols(d_cur,(float*)G(L("ffn_gate_inp.scale")).dev,n_embd,N,0);
        mm(d_rlogits, G(L("ffn_gate_inp.weight")), d_cur, N);   // [128,N]
        dg_softmax_topk(d_rlogits,DG_N_EXPERTS,N,DG_N_EXPERTS_USED,d_topk_idx,d_topk_w,0);
        CK(cudaMemcpy(th_idx.data(),d_topk_idx,N*DG_N_EXPERTS_USED*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(th_w.data(),d_topk_w,N*DG_N_EXPERTS_USED*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pes.data(),G(L("ffn_down_exps.scale")).dev,DG_N_EXPERTS*4,cudaMemcpyDeviceToHost));
        CK(cudaMemset(d_moeout,0,(size_t)n_embd*N*4));
        DGT& gue=G(L("ffn_gate_up_exps.weight"));   // [n_embd, 2*ff_exp, 128] Q4_K
        DGT& dwe=G(L("ffn_down_exps.weight"));       // [ff_exp, n_embd, 128] Q8_0
        int64_t gue_slab=row_bytes(gue.type,gue.ne[0])* (2*DG_EXPERT_FFN);  // bytes per expert
        int64_t dwe_slab=row_bytes(dwe.type,dwe.ne[0])* n_embd;
        // build per-expert token lists on host
        std::vector<std::vector<int>> etok(DG_N_EXPERTS);
        std::vector<std::vector<float>> ecs(DG_N_EXPERTS);
        for(int t=0;t<N;t++) for(int k=0;k<DG_N_EXPERTS_USED;k++){
            int ex=th_idx[t*DG_N_EXPERTS_USED+k]; float w=th_w[t*DG_N_EXPERTS_USED+k];
            etok[ex].push_back(t); ecs[ex].push_back(w*pes[ex]);
        }
        for(int ex=0;ex<DG_N_EXPERTS;ex++){
            int ne=etok[ex].size(); if(!ne) continue;
            CK(cudaMemcpy(d_eidx,etok[ex].data(),ne*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_ecs,ecs[ex].data(),ne*4,cudaMemcpyHostToDevice));
            dg_gather_cols(d_xe,d_moein,d_eidx,n_embd,ne,0);
            mm_raw(d_gu, gue.dev + (size_t)ex*gue_slab, gue.type, n_embd, 2*DG_EXPERT_FFN, d_xe, ne); // [1408,ne]
            dg_split_gelu_mul(d_act,d_gu,DG_EXPERT_FFN,ne,0);                                        // [704,ne]
            mm_raw(d_oe, dwe.dev + (size_t)ex*dwe_slab, dwe.type, DG_EXPERT_FFN, n_embd, d_act, ne);  // [2816,ne]
            dg_scatteradd_cols(d_moeout,d_oe,d_eidx,d_ecs,n_embd,ne,0);
        }
        dg_rmsnorm(d_moe,d_moeout,(float*)G(L("post_ffw_norm_2.weight")).dev,n_embd,N,eps,0);

        // combine
        dg_add(d_ffn,d_dense,d_moe,(int64_t)n_embd*N,0);
        dg_rmsnorm(d_ffn,d_ffn,(float*)G(L("post_ffw_norm.weight")).dev,n_embd,N,eps,0);
        dg_add(d_inpL,d_ffn,d_attnout,(int64_t)n_embd*N,0);
        // region scalar
        dg_mul_region_scalar(d_inpL,n_embd,0,P,(float*)G(L("enc_layer_output_scale.weight")).dev,0);
        dg_mul_region_scalar(d_inpL,n_embd,P,N,(float*)G(L("layer_output_scale.weight")).dev,0);
        CK(cudaDeviceSynchronize());
    }

    // final
    dg_rmsnorm(d_cur,d_inpL,(float*)G("output_norm.weight").dev,n_embd,N,eps,0);
    // lm head (tied token_embd, dequanted): logits[vocab,N] = tok[vocab,n_embd] @ cur
    CB(cublasSgemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,vocab,N,n_embd,&one,d_tok,n_embd,d_cur,n_embd,&zero,d_logits,vocab));
    dg_softcap(d_logits,(int64_t)vocab*N,DG_SOFTCAP,0);
    CK(cudaDeviceSynchronize());

    // compare canvas logits (cols P..N-1) to ref [C, vocab]
    std::vector<float> got((size_t)C*vocab);
    CK(cudaMemcpy(got.data(),d_logits+(size_t)P*vocab,(size_t)C*vocab*4,cudaMemcpyDeviceToHost));
    { FILE* mf=fopen("/tmp/dg_dequant/mine.bin","wb"); fwrite(got.data(),4,got.size(),mf); fclose(mf); }
    // parity restricted to CONFIDENT positions (ref top1-top2 gap > 1.0): there the model is
    // decided and numerics can't flip it — this isolates correctness from the random-canvas noise floor.
    { int conf=0, conf_match=0; double conf_maxe=0;
      for(int c=0;c<C;c++){ const float*g=got.data()+(size_t)c*vocab; const float*r=ref.data()+(size_t)c*vocab;
        float r1=-1e30f,r2=-1e30f; int ar=0; for(int v=0;v<vocab;v++){ if(r[v]>r1){r2=r1;r1=r[v];ar=v;} else if(r[v]>r2)r2=r[v]; }
        if(r1-r2>1.0f){ conf++; int ag=0; for(int v=0;v<vocab;v++) if(g[v]>g[ag])ag=v; if(ag==ar)conf_match++;
          double e=fabs((double)g[ar]-r[ar]); if(e>conf_maxe)conf_maxe=e; } }
      printf("  CONFIDENT positions (ref gap>1.0): %d, argmax match %d/%d, max|Δ| at winner=%.3f\n",conf,conf_match,conf,conf_maxe); }
    double max_abs=0,sum_abs=0; long long argmax_match=0;
    int npos_bad=0; double worst_pos_err=0; int worst_pos=-1;
    for(int c=0;c<C;c++){
        const float* g=got.data()+(size_t)c*vocab; const float* r=ref.data()+(size_t)c*vocab;
        int ag=0,ar=0; double pmax=0;
        for(int v=0;v<vocab;v++){ double e=fabs((double)g[v]-r[v]); if(e>max_abs)max_abs=e; if(e>pmax)pmax=e; sum_abs+=e; if(g[v]>g[ag])ag=v; if(r[v]>r[ar])ar=v; }
        if(ag==ar) argmax_match++; else {
            // top1-top2 gap in ref at this position (near-tie ⇒ precision, not bug)
            float r1=-1e30f,r2=-1e30f; for(int v=0;v<vocab;v++){ if(r[v]>r1){r2=r1;r1=r[v];} else if(r[v]>r2)r2=r[v]; }
            if(npos_bad<6) printf("  mismatch pos %d: mine=%d ref=%d  refgap=%.4f  posMax|Δ|=%.3f\n",c,ag,ar,r1-r2,pmax);
            npos_bad++;
        }
        if(pmax>worst_pos_err){worst_pos_err=pmax;worst_pos=c;}
    }
    // error histogram over all logits
    long long h01=0,h05=0,h1=0,h2=0,hbig=0;
    for(size_t i=0;i<got.size();i++){ double e=fabs((double)got[i]-ref[i]); if(e<0.1)h01++; else if(e<0.5)h05++; else if(e<1)h1++; else if(e<2)h2++; else hbig++; }
    double tot=(double)got.size();
    printf("\nFORWARD PARITY vs llama.cpp:\n");
    printf("  max|Δ| = %.4e   mean|Δ| = %.4e   argmax match = %lld/%d  (worst pos %d err %.3f)\n",
           max_abs, sum_abs/((double)C*vocab), argmax_match, C, worst_pos, worst_pos_err);
    printf("  |Δ|<0.1: %.2f%%  <0.5: %.2f%%  <1: %.2f%%  <2: %.2f%%  >=2: %.2f%%\n",
           100*h01/tot,100*h05/tot,100*h1/tot,100*h2/tot,100*hbig/tot);
    { float p[8]; CK(cudaMemcpy(p,G("blk.0.ffn_down_exps.scale").dev,8*4,cudaMemcpyDeviceToHost));
      printf("  per_expert_scale[0..7] L0 = %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]); }
    bool pass = (argmax_match==C) && (max_abs < 0.5);
    printf("  %s\n", pass?"PASS":"FAIL (investigate)");
    return pass?0:1;
}
