// dg_generate.cu — DiffusionGemma end-to-end block-diffusion text generation.
//
// Reuses the validated single-pass [prompt|canvas] forward (test_diffusion_forward.cu) and adds:
//   • self-conditioning (prev-step softmax → soft embedding → gated MLP → canvas embed)
//   • the entropy-bound sampler + linear temperature schedule + renoise
//   • stable-and-confident stopping
// per the HF reference (generation_diffusion_gemma.py) and llama.cpp diffusion-gemma.cpp.
//
// Build: nvcc -O2 -arch=sm_121a cuda/diffusion_gemma_kernels.cu cuda/dg_generate.cu -lcublas -o /tmp/dg_gen
// Run:   /tmp/dg_gen <model.gguf> /tmp/dg_dequant/gen_prompt.i32 [steps] [seed] > out_ids.txt

#include "diffusion_gemma_kernels.cuh"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ fprintf(stderr,"CUDA %s @ %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__); exit(1);} } while(0)
#define CB(x) do { cublasStatus_t st=(x); if(st!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"cuBLAS %d @ %s:%d\n",(int)st,__FILE__,__LINE__); exit(1);} } while(0)

enum { GT_STR=8, GT_ARR=9 };
static uint64_t scalar_sz(uint32_t t){switch(t){case 0:case 1:case 7:return 1;case 2:case 3:return 2;case 4:case 5:case 6:return 4;case 10:case 11:case 12:return 8;default:return 0;}}
struct DGT { uint8_t *dev=nullptr; int type=0; int ndim=0; int64_t ne[4]={1,1,1,1}; int64_t nbytes=0,nelem=0; uint64_t offset=0; };
static int64_t row_bytes(int t,int64_t ne0){ int bb=dg_block_bytes(t),bn=dg_block_nelem(t); if(t==DG_GGML_F32)return ne0*4; if(t==DG_GGML_F16)return ne0*2; return (ne0/bn)*bb; }

static uint64_t xs=0x2545F4914F6CDD1Dull;
static double urand(){ xs^=xs<<13; xs^=xs>>7; xs^=xs<<17; return (xs>>11)*(1.0/9007199254740992.0); }

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s model.gguf prompt.i32 [steps] [seed]\n",argv[0]); return 1; }
    int max_steps = argc>3?atoi(argv[3]):DG_MAX_DENOISE_STEPS;
    if(argc>4) xs = (uint64_t)strtoull(argv[4],0,10)|1;

    auto read_i32=[](const char*p){ FILE*f=fopen(p,"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); std::vector<int> v(n/4); if(fread(v.data(),4,v.size(),f)!=v.size()){exit(1);} fclose(f); return v; };
    std::vector<int> prompt=read_i32(argv[2]);
    const int P=prompt.size(), C=DG_CANVAS_LENGTH, N=P+C;
    fprintf(stderr,"prompt=%d canvas=%d N=%d steps=%d\n",P,C,N,max_steps);

    // ── parse + upload GGUF ──
    int fd=open(argv[1],O_RDONLY); struct stat stt; fstat(fd,&stt);
    uint8_t* base=(uint8_t*)mmap(nullptr,stt.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    const uint8_t* p=base+8; uint64_t nt=*(uint64_t*)p; p+=8; uint64_t nkv=*(uint64_t*)p; p+=8;
    auto rd_str=[&](const uint8_t*&q){ uint64_t l=*(uint64_t*)q; q+=8; std::string s((const char*)q,l); q+=l; return s; };
    auto skip_val=[&](const uint8_t*&q,uint32_t vt){ if(vt==GT_STR){uint64_t l=*(uint64_t*)q;q+=8+l;} else if(vt==GT_ARR){uint32_t at=*(uint32_t*)q;q+=4;uint64_t c=*(uint64_t*)q;q+=8; if(at==GT_STR){for(uint64_t i=0;i<c;i++){uint64_t l=*(uint64_t*)q;q+=8+l;}} else q+=c*scalar_sz(at);} else q+=scalar_sz(vt); };
    for(uint64_t i=0;i<nkv;i++){ rd_str(p); uint32_t vt=*(uint32_t*)p; p+=4; skip_val(p,vt); }
    std::unordered_map<std::string,DGT> T;
    for(uint64_t i=0;i<nt;i++){ std::string nm=rd_str(p); DGT t; t.ndim=*(uint32_t*)p; p+=4; for(int d=0;d<t.ndim;d++){t.ne[d]=*(int64_t*)p;p+=8;} t.type=*(uint32_t*)p;p+=4; t.offset=*(uint64_t*)p;p+=8; t.nelem=1; for(int d=0;d<t.ndim;d++)t.nelem*=t.ne[d]; int64_t rb=row_bytes(t.type,t.ne[0]),rows=1; for(int d=1;d<t.ndim;d++)rows*=t.ne[d]; t.nbytes=rb*rows; T[nm]=t; }
    uint64_t off=(uint64_t)(p-base); off=(off+31)&~31ull; const uint8_t* data=base+off;
    for(auto&kv:T){ DGT&t=kv.second; CK(cudaMalloc(&t.dev,t.nbytes)); CK(cudaMemcpy(t.dev,data+t.offset,t.nbytes,cudaMemcpyHostToDevice)); }
    fprintf(stderr,"weights uploaded\n");

    cublasHandle_t cub; CB(cublasCreate(&cub)); CB(cublasSetMathMode(cub,CUBLAS_PEDANTIC_MATH));
    auto G=[&](const std::string&n)->DGT&{ auto it=T.find(n); if(it==T.end()){fprintf(stderr,"MISSING %s\n",n.c_str());exit(1);} return it->second; };

    const int n_embd=DG_HIDDEN,n_ff=DG_FFN_INTERMEDIATE,n_head=DG_HEADS,vocab=DG_VOCAB;
    const float eps=DG_RMS_EPS,esc=sqrtf((float)n_embd),one=1.f,zero=0.f;

    float *d_wscr; CK(cudaMalloc(&d_wscr,(size_t)26000000*4));
    float *d_tok;  CK(cudaMalloc(&d_tok,(size_t)vocab*n_embd*4));
    { DGT&te=G("token_embd.weight"); dg_dequant(te.type,te.dev,te.nelem,d_tok,0); } CK(cudaDeviceSynchronize());

    int *d_ids; CK(cudaMalloc(&d_ids,N*4));
    int *d_pos; CK(cudaMalloc(&d_pos,N*4)); { std::vector<int> ph(N); for(int i=0;i<N;i++)ph[i]=i; CK(cudaMemcpy(d_pos,ph.data(),N*4,cudaMemcpyHostToDevice)); }
    auto AB=[&](size_t nf){ float*q; CK(cudaMalloc(&q,nf*4)); return q; };
    float *d_inpL=AB((size_t)n_embd*N),*d_cur=AB((size_t)n_embd*N),*d_attnout=AB((size_t)n_embd*N);
    float *d_q=AB((size_t)DG_GLOBAL_HEAD_DIM*n_head*N),*d_kraw=AB((size_t)DG_GLOBAL_HEAD_DIM*n_head*N);
    float *d_k=AB((size_t)DG_GLOBAL_HEAD_DIM*n_head*N),*d_v=AB((size_t)DG_GLOBAL_HEAD_DIM*n_head*N),*d_attn=AB((size_t)DG_GLOBAL_HEAD_DIM*n_head*N);
    float *d_dense=AB((size_t)n_embd*N),*d_moe=AB((size_t)n_embd*N),*d_ffn=AB((size_t)n_embd*N);
    float *d_tmp=AB((size_t)n_ff*N),*d_tmp2=AB((size_t)n_ff*N),*d_rtmp=AB((size_t)n_embd*N);
    float *d_rlogits=AB((size_t)DG_N_EXPERTS*N),*d_moeout=AB((size_t)n_embd*N);
    float *d_xe=AB((size_t)n_embd*N),*d_gu=AB((size_t)2*DG_EXPERT_FFN*N),*d_act=AB((size_t)DG_EXPERT_FFN*N),*d_oe=AB((size_t)n_embd*N);
    int *d_eidx; CK(cudaMalloc(&d_eidx,N*4)); float *d_ecs; CK(cudaMalloc(&d_ecs,N*4));
    int *d_tki; CK(cudaMalloc(&d_tki,N*DG_N_EXPERTS_USED*4)); float *d_tkw; CK(cudaMalloc(&d_tkw,N*DG_N_EXPERTS_USED*4));
    float *d_logits=AB((size_t)vocab*N);
    // SC + sampler buffers
    float *d_sc=AB((size_t)vocab*C);                 // prev-step scaled canvas logits (self-cond input)
    float *d_scprob=AB((size_t)vocab*C), *d_soft=AB((size_t)n_embd*C), *d_scn=AB((size_t)n_embd*C);
    float *d_scg=AB((size_t)n_ff*C), *d_scu=AB((size_t)n_ff*C), *d_scsig=AB((size_t)n_embd*C);
    float *d_rnd; CK(cudaMalloc(&d_rnd,C*4));
    int *d_sample; CK(cudaMalloc(&d_sample,C*4)); int *d_argmax; CK(cudaMalloc(&d_argmax,C*4)); float *d_ent; CK(cudaMalloc(&d_ent,C*4));

    DGT& rope_freqs=G("rope_freqs.weight");
    std::vector<int> th_idx(N*DG_N_EXPERTS_USED); std::vector<float> th_w(N*DG_N_EXPERTS_USED), pes(DG_N_EXPERTS);

    auto mm_raw=[&](float*out,const uint8_t*raw,int type,int64_t in,int64_t outd,const float*x,int T_){
        const float*A; if(type==DG_GGML_F32)A=(const float*)raw; else{ dg_dequant(type,raw,in*outd,d_wscr,0); A=d_wscr; }
        CB(cublasSgemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,(int)outd,T_,(int)in,&one,A,(int)in,x,(int)in,&zero,out,(int)outd)); };
    auto mm=[&](float*out,DGT&W,const float*x,int T_){ mm_raw(out,W.dev,W.type,W.ne[0],W.ne[1],x,T_); };

    // ── forward over [prompt|canvas]; canvas = current canvas ids; optional self-conditioning ──
    std::vector<int> canvas(C);
    auto forward=[&](float sc_use){
        // build ids = prompt ++ canvas
        std::vector<int> ids(N); for(int i=0;i<P;i++)ids[i]=prompt[i]; for(int i=0;i<C;i++)ids[P+i]=canvas[i];
        CK(cudaMemcpy(d_ids,ids.data(),N*4,cudaMemcpyHostToDevice));
        dg_embed_gather(d_inpL,d_tok,d_ids,n_embd,N,esc,0);          // scaled embed (all rows)
        float* d_canvas_emb = d_inpL+(size_t)P*n_embd;               // canvas region
        if(sc_use>0.5f){
            // soft = (softmax(sc_logits) @ token_embd) * embed_scale
            dg_softmax_cols(d_sc,d_scprob,vocab,C,0);
            CB(cublasSgemm(cub,CUBLAS_OP_N,CUBLAS_OP_N,n_embd,C,vocab,&one,d_tok,n_embd,d_scprob,vocab,&zero,d_soft,n_embd));
            dg_scale(d_soft,(int64_t)n_embd*C,esc,0);
            dg_rmsnorm(d_scn,d_soft,(float*)G("self_cond_pre_norm.weight").dev,n_embd,C,eps,0);
            mm(d_scg,G("self_cond_gate.weight"),d_scn,C);
            mm(d_scu,G("self_cond_up.weight"),d_scn,C);
            dg_gelu_mul(d_scg,d_scg,d_scu,(int64_t)n_ff*C,0);
            mm(d_scsig,G("self_cond_down.weight"),d_scg,C);          // [n_embd,C]
            dg_add(d_canvas_emb,d_canvas_emb,d_scsig,(int64_t)n_embd*C,0);
        }
        dg_rmsnorm(d_canvas_emb,d_canvas_emb,nullptr,n_embd,C,eps,0); // canvas: rmsnorm_noscale(scaled+sc)

        char b[64]; auto L=[&](const char*s){ snprintf(b,64,"blk.%d.%s",0,s); return std::string(b); };
        for(int il=0;il<DG_MAX_LAYERS;il++){
            bool global=(il%6==5); bool sliding=!global;
            int hd=sliding?DG_HEAD_DIM:DG_GLOBAL_HEAD_DIM, nkv=sliding?DG_KV_HEADS_SLIDING:DG_KV_HEADS_GLOBAL;
            float theta=sliding?DG_ROPE_THETA_SLIDING:DG_ROPE_THETA_GLOBAL;
            const float* ff=sliding?nullptr:(const float*)rope_freqs.dev;
            auto LL=[&](const char*s){ snprintf(b,64,"blk.%d.%s",il,s); return std::string(b); };
            dg_rmsnorm(d_cur,d_inpL,(float*)G(LL("attn_norm.weight")).dev,n_embd,N,eps,0);
            mm(d_q,G(LL("attn_q.weight")),d_cur,N); dg_head_rmsnorm(d_q,(float*)G(LL("attn_q_norm.weight")).dev,hd,n_head,N,eps,0); dg_rope(d_q,d_pos,hd,n_head,N,theta,ff,0);
            mm(d_kraw,G(LL("attn_k.weight")),d_cur,N);
            CK(cudaMemcpy(d_k,d_kraw,(size_t)hd*nkv*N*4,cudaMemcpyDeviceToDevice)); dg_head_rmsnorm(d_k,(float*)G(LL("attn_k_norm.weight")).dev,hd,nkv,N,eps,0); dg_rope(d_k,d_pos,hd,nkv,N,theta,ff,0);
            if(sliding) mm(d_v,G(LL("attn_v.weight")),d_cur,N); else CK(cudaMemcpy(d_v,d_kraw,(size_t)hd*nkv*N*4,cudaMemcpyDeviceToDevice));
            dg_head_rmsnorm(d_v,nullptr,hd,nkv,N,eps,0);
            dg_attention(d_attn,d_q,d_k,d_v,hd,n_head,nkv,N,P,DG_SLIDING_WINDOW,sliding?1:0,0);
            mm(d_cur,G(LL("attn_output.weight")),d_attn,N);
            dg_rmsnorm(d_cur,d_cur,(float*)G(LL("post_attention_norm.weight")).dev,n_embd,N,eps,0);
            dg_add(d_attnout,d_cur,d_inpL,(int64_t)n_embd*N,0);
            // dense
            dg_rmsnorm(d_cur,d_attnout,(float*)G(LL("ffn_norm.weight")).dev,n_embd,N,eps,0);
            mm(d_tmp,G(LL("ffn_gate.weight")),d_cur,N); mm(d_tmp2,G(LL("ffn_up.weight")),d_cur,N);
            dg_gelu_mul(d_tmp,d_tmp,d_tmp2,(int64_t)n_ff*N,0); mm(d_dense,G(LL("ffn_down.weight")),d_tmp,N);
            dg_rmsnorm(d_dense,d_dense,(float*)G(LL("post_ffw_norm_1.weight")).dev,n_embd,N,eps,0);
            // moe
            float* d_moein=d_rtmp; dg_rmsnorm(d_moein,d_attnout,(float*)G(LL("pre_ffw_norm_2.weight")).dev,n_embd,N,eps,0);
            dg_rmsnorm(d_cur,d_attnout,nullptr,n_embd,N,eps,0); dg_scale(d_cur,(int64_t)n_embd*N,1.0f/sqrtf((float)n_embd),0);
            dg_mul_vec_cols(d_cur,(float*)G(LL("ffn_gate_inp.scale")).dev,n_embd,N,0);
            mm(d_rlogits,G(LL("ffn_gate_inp.weight")),d_cur,N);
            dg_softmax_topk(d_rlogits,DG_N_EXPERTS,N,DG_N_EXPERTS_USED,d_tki,d_tkw,0);
            CK(cudaMemcpy(th_idx.data(),d_tki,N*DG_N_EXPERTS_USED*4,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(th_w.data(),d_tkw,N*DG_N_EXPERTS_USED*4,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(pes.data(),G(LL("ffn_down_exps.scale")).dev,DG_N_EXPERTS*4,cudaMemcpyDeviceToHost));
            CK(cudaMemset(d_moeout,0,(size_t)n_embd*N*4));
            DGT&gue=G(LL("ffn_gate_up_exps.weight")), &dwe=G(LL("ffn_down_exps.weight"));
            int64_t gslab=row_bytes(gue.type,gue.ne[0])*(2*DG_EXPERT_FFN), dslab=row_bytes(dwe.type,dwe.ne[0])*n_embd;
            std::vector<std::vector<int>> et(DG_N_EXPERTS); std::vector<std::vector<float>> ec(DG_N_EXPERTS);
            for(int t=0;t<N;t++) for(int k=0;k<DG_N_EXPERTS_USED;k++){ int ex=th_idx[t*DG_N_EXPERTS_USED+k]; et[ex].push_back(t); ec[ex].push_back(th_w[t*DG_N_EXPERTS_USED+k]*pes[ex]); }
            for(int ex=0;ex<DG_N_EXPERTS;ex++){ int ne=et[ex].size(); if(!ne)continue;
                CK(cudaMemcpy(d_eidx,et[ex].data(),ne*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_ecs,ec[ex].data(),ne*4,cudaMemcpyHostToDevice));
                dg_gather_cols(d_xe,d_moein,d_eidx,n_embd,ne,0);
                mm_raw(d_gu,gue.dev+(size_t)ex*gslab,gue.type,n_embd,2*DG_EXPERT_FFN,d_xe,ne);
                dg_split_gelu_mul(d_act,d_gu,DG_EXPERT_FFN,ne,0);
                mm_raw(d_oe,dwe.dev+(size_t)ex*dslab,dwe.type,DG_EXPERT_FFN,n_embd,d_act,ne);
                dg_scatteradd_cols(d_moeout,d_oe,d_eidx,d_ecs,n_embd,ne,0);
            }
            dg_rmsnorm(d_moe,d_moeout,(float*)G(LL("post_ffw_norm_2.weight")).dev,n_embd,N,eps,0);
            dg_add(d_ffn,d_dense,d_moe,(int64_t)n_embd*N,0);
            dg_rmsnorm(d_ffn,d_ffn,(float*)G(LL("post_ffw_norm.weight")).dev,n_embd,N,eps,0);
            dg_add(d_inpL,d_ffn,d_attnout,(int64_t)n_embd*N,0);
            dg_mul_region_scalar(d_inpL,n_embd,0,P,(float*)G(LL("enc_layer_output_scale.weight")).dev,0);
            dg_mul_region_scalar(d_inpL,n_embd,P,N,(float*)G(LL("layer_output_scale.weight")).dev,0);
        }
        dg_rmsnorm(d_cur,d_inpL,(float*)G("output_norm.weight").dev,n_embd,N,eps,0);
        CB(cublasSgemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,vocab,N,n_embd,&one,d_tok,n_embd,d_cur,n_embd,&zero,d_logits,vocab));
        dg_softcap(d_logits,(int64_t)vocab*N,DG_SOFTCAP,0);
        CK(cudaDeviceSynchronize());
    };

    // ── diffusion loop ──
    for(int i=0;i<C;i++) canvas[i]=(int)(urand()*vocab);   // random init
    std::vector<int> argmax_canvas(C,-1), prev_argmax(C,-2);
    std::vector<int> hsample(C),hargmax(C); std::vector<float> hent(C),hrnd(C);
    float sc_use=0.f;
    int used_steps=0;
    for(int step=max_steps; step>=1; step--){
        used_steps++;
        float Ttemp = DG_T_MIN + (DG_T_MAX-DG_T_MIN)*((float)step/max_steps);
        forward(sc_use);
        float* d_clog = d_logits+(size_t)P*vocab;            // canvas logits [vocab,C]
        dg_scale(d_clog,(int64_t)vocab*C,1.0f/Ttemp,0);      // temperature
        for(int c=0;c<C;c++) hrnd[c]=(float)urand();
        CK(cudaMemcpy(d_rnd,hrnd.data(),C*4,cudaMemcpyHostToDevice));
        dg_sample_step(d_clog,d_rnd,d_sample,d_argmax,d_ent,vocab,C,0);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hsample.data(),d_sample,C*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hargmax.data(),d_argmax,C*4,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hent.data(),d_ent,C*4,cudaMemcpyDeviceToHost));
        // entropy-bound accept: sort positions by entropy asc; accept while cumsum-cur <= bound
        std::vector<int> ord(C); std::iota(ord.begin(),ord.end(),0);
        std::sort(ord.begin(),ord.end(),[&](int a,int b){return hent[a]<hent[b];});
        std::vector<char> accept(C,0); double cum=0;
        for(int r=0;r<C;r++){ int c=ord[r]; cum+=hent[c]; if(cum-hent[c] <= DG_ENTROPY_BOUND) accept[c]=1; }
        argmax_canvas=hargmax;
        // renoise: accepted -> sample, else fresh random
        for(int c=0;c<C;c++) canvas[c] = accept[c]? hsample[c] : (int)(urand()*vocab);
        // self-cond for next step = this step's (temperature-scaled) canvas logits
        CK(cudaMemcpy(d_sc,d_clog,(size_t)vocab*C*4,cudaMemcpyDeviceToDevice)); sc_use=1.f;
        // stop: stable (argmax==prev) AND confident (mean entropy < thresh)
        double me=0; for(int c=0;c<C;c++) me+=hent[c]; me/=C;
        bool stable=(argmax_canvas==prev_argmax); prev_argmax=argmax_canvas;
        int eos_at=-1; for(int c=0;c<C;c++) if(argmax_canvas[c]==DG_EOS_ID){eos_at=c;break;}
        fprintf(stderr,"step %2d T=%.3f mean_ent=%.4f accepted=%d/%d %s%s\n",step,Ttemp,me,
                (int)std::count(accept.begin(),accept.end(),(char)1),C, stable?"STABLE ":"", eos_at>=0?"[eos]":"");
        if(stable && me<DG_CONFIDENCE_THRESH){ fprintf(stderr,"converged (stable+confident)\n"); break; }
    }

    // emit committed argmax canvas ids (trim at first eos)
    int outn=C; for(int c=0;c<C;c++) if(argmax_canvas[c]==DG_EOS_ID){ outn=c; break; }
    fprintf(stderr,"done in %d steps; %d output tokens\n",used_steps,outn);
    for(int c=0;c<outn;c++) printf("%d ",argmax_canvas[c]);
    printf("\n");
    return 0;
}
