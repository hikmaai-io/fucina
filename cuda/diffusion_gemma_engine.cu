// diffusion_gemma_engine.cu — C ABI implementation of the DiffusionGemma engine.
// Refactor of dg_generate.cu (validated end-to-end) into a reusable create/generate/free API.

#include "diffusion_gemma_engine.h"
#include "diffusion_gemma_kernels.cuh"
#include <cublas_v2.h>
#include <cublasLt.h>        // NVFP4 block-scaled FP4 tensor-core dense GEMM (DG_FP4)
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ fprintf(stderr,"[dg] CUDA %s @ %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__); return; } } while(0)
#define CKR(x,r) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ fprintf(stderr,"[dg] CUDA %s @ %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__); return (r); } } while(0)
#define CB(x) do { cublasStatus_t st=(x); if(st!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"[dg] cuBLAS %d @ %s:%d\n",(int)st,__FILE__,__LINE__); return -1; } } while(0)

enum { GTYPE_STR=8, GTYPE_ARR=9 };
static uint64_t dg_scalar_sz(uint32_t t){switch(t){case 0:case 1:case 7:return 1;case 2:case 3:return 2;case 4:case 5:case 6:return 4;case 10:case 11:case 12:return 8;default:return 0;}}
static int64_t dg_row_bytes(int t,int64_t ne0){ int bb=dg_block_bytes(t),bn=dg_block_nelem(t); if(t==DG_GGML_F32)return ne0*4; if(t==DG_GGML_F16)return ne0*2; return (ne0/bn)*bb; }

struct DGTensor { uint8_t *dev=nullptr; int type=0; int ndim=0; int64_t ne[4]={1,1,1,1}; int64_t nbytes=0,nelem=0; uint64_t offset=0;
                  __nv_bfloat16 *dev_bf=nullptr;       // lazily-cached bf16 dequant (dense GEMM path)
                  uint8_t *dev_fp4=nullptr, *dev_fp4sc=nullptr; float gs_w=0.f; }; // NVFP4 cache (DG_FP4)

// ─── lightweight GGUF metadata scan (architecture / canvas_length) ──────
// Returns: out_arch (if non-null) gets general.architecture; *out_canvas gets diffusion.canvas_length.
static int dg_scan_meta(const char *path, std::string *out_arch, int *out_canvas) {
    int fd=open(path,O_RDONLY); if(fd<0) return -1;
    struct stat st; if(fstat(fd,&st)){close(fd);return -1;}
    uint8_t *base=(uint8_t*)mmap(nullptr,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    if(base==MAP_FAILED) return -1;
    const uint8_t *p=base; if(*(uint32_t*)p!=0x46554747){munmap(base,st.st_size);return -1;}
    p+=8; /*magic+ver*/ p+=8; /*n_tensors*/ uint64_t nkv=*(uint64_t*)p; p+=8;
    auto rd_str=[&](const uint8_t*&q){ uint64_t l=*(uint64_t*)q; q+=8; std::string s((const char*)q,l); q+=l; return s; };
    int rc=-1;
    for(uint64_t i=0;i<nkv;i++){
        std::string key=rd_str(p); uint32_t vt=*(uint32_t*)p; p+=4;
        if(vt==GTYPE_STR){ std::string v=rd_str(p); if(key=="general.architecture"&&out_arch)*out_arch=v; rc=0; }
        else if(vt==GTYPE_ARR){ uint32_t at=*(uint32_t*)p; p+=4; uint64_t c=*(uint64_t*)p; p+=8;
            if(at==GTYPE_STR){ for(uint64_t j=0;j<c;j++){ uint64_t l=*(uint64_t*)p; p+=8+l; } } else p+=c*dg_scalar_sz(at); }
        else { // scalar; capture canvas_length (u32/i32)
            if(key=="diffusion.canvas_length"&&out_canvas){ *out_canvas=*(int32_t*)p; }
            p+=dg_scalar_sz(vt);
        }
    }
    munmap(base,st.st_size);
    return rc;
}

extern "C" int dg_gguf_is_diffusion(const char *path){
    std::string arch; if(dg_scan_meta(path,&arch,nullptr)!=0) return -1;
    return arch=="diffusion-gemma" ? 1 : 0;
}
extern "C" int dg_gguf_canvas_length(const char *path){
    int cl=-1; if(dg_scan_meta(path,nullptr,&cl)!=0) return -1; return cl;
}

// ─── engine ─────────────────────────────────────────────────────────────
struct dg_engine {
    std::unordered_map<std::string,DGTensor> T;
    uint8_t *mmap_base=nullptr; size_t mmap_size=0;
    cublasHandle_t cub=nullptr;
    int canvas_length=DG_CANVAS_LENGTH, max_prompt=0, N_max=0;
    int n_embd=DG_HIDDEN, n_ff=DG_FFN_INTERMEDIATE, n_head=DG_HEADS, vocab=DG_VOCAB;
    // device buffers
    float *d_wscr=nullptr,*d_tok=nullptr;
    // bf16 tensor-core GEMM path: token_embd in bf16 (LM head + self-cond), a bf16 weight scratch
    // (dense projections dequant straight to bf16), and a bf16 activation scratch.
    __nv_bfloat16 *d_tok_bf=nullptr,*d_actbf=nullptr;
    int *d_ids=nullptr,*d_pos=nullptr;
    float *d_inpL=nullptr,*d_cur=nullptr,*d_attnout=nullptr,*d_q=nullptr,*d_kraw=nullptr,*d_k=nullptr,*d_v=nullptr,*d_attn=nullptr;
    float *d_dense=nullptr,*d_moe=nullptr,*d_ffn=nullptr,*d_tmp=nullptr,*d_tmp2=nullptr,*d_rtmp=nullptr,*d_rlogits=nullptr,*d_moeout=nullptr;
    // grouped-expert MoE scratch (all active experts processed in one launch per projection):
    // flattened gathered assignments [feat × Tmax] grouped by expert, the int8 activation scratch,
    // and the per-(expert,tile) descriptors uploaded each layer.
    float *d_xe_all=nullptr,*d_gu_all=nullptr,*d_act_all=nullptr,*d_oe_all=nullptr;
    int *d_eidx_all=nullptr; float *d_ecs_all=nullptr;
    int8_t *d_q8=nullptr; float *d_q8d=nullptr; int *d_q8s=nullptr;
    int *d_count=nullptr,*d_coloff=nullptr,*d_cursor=nullptr;  // on-GPU MoE routing (counting-sort)
    int Tmax=0;
    int *d_tki=nullptr; float *d_tkw=nullptr; float *d_clogits=nullptr;        // canvas logits [vocab,C]
    float *d_sc=nullptr,*d_scprob=nullptr,*d_soft=nullptr,*d_scn=nullptr,*d_scg=nullptr,*d_scu=nullptr,*d_scsig=nullptr,*d_rnd=nullptr;
    int *d_sample=nullptr,*d_argmax=nullptr; float *d_ent=nullptr;
    // per-layer prompt K/V cache (computed once per Generate during prefill, then read by every
    // denoising step's canvas-only forward) — sized [head_dim*n_kv, max_prompt] per layer.
    __nv_bfloat16 *d_pk[DG_MAX_LAYERS]={nullptr}, *d_pv[DG_MAX_LAYERS]={nullptr}; // prompt K/V cache (bf16, ½ the fp32 size)
    int cached_P=0;
    // stats from the most recent dg_engine_generate (surfaced via dg_engine_last_stats)
    float last_prefill_ms=0.f, last_denoise_ms=0.f; int last_steps=0;
    DGTensor *rope_freqs=nullptr;
    // ── NVFP4 dense GEMM (DG_FP4=1): persistent per-weight NVFP4 cache (DGTensor.dev_fp4*)
    // + per-GEMM activation quant + cuBLASLt block-scaled FP4 matmul (~2-3× the bf16 path
    // on the N=256 dense shapes). cuBLASLt validated bit-exact (32×4×4 scale swizzle). ──
    int fp4_enabled=-1;                 // -1 unprobed, 0 off, 1 on (from DG_FP4 env)
    cublasLtHandle_t lt=nullptr; cublasLtMatmulDesc_t fp4_desc=nullptr; void *fp4_ws=nullptr;
    uint8_t *d_fp4_act=nullptr,*d_fp4_actsc=nullptr,*d_fp4_actlin=nullptr; size_t fp4_act_cap=0;
    float *d_fp4_gsact=nullptr,*d_fp4_alpha=nullptr,*d_fp4_amax=nullptr;

    // ── NVFP4 MoE experts (CLI --fp4-moe / -dm): grouped block-scaled FP4 tensor-core expert GEMM
    // (CUTLASS sm120, 2.56× the dp4a path). Persistent per-layer NVFP4 expert weights +
    // per-step fused per-expert activation quant. dg_fp4_moe_grouped is in libdg/dg_fp4_moe.cu.
    int fp4moe_enabled=-1, fp4moe_ready=0, fp4moe_want=0;   // want set at create from the CLI flag
    uint8_t *d_gu_fp4[DG_MAX_LAYERS]={nullptr}, *d_gu_sf[DG_MAX_LAYERS]={nullptr};   // gate_up
    uint8_t *d_dn_fp4[DG_MAX_LAYERS]={nullptr}, *d_dn_sf[DG_MAX_LAYERS]={nullptr};   // down
    float gu_gsw[DG_MAX_LAYERS]={0}, dn_gsw[DG_MAX_LAYERS]={0};                      // shared weight global/layer
    unsigned long long gu_sfBstride=0, dn_sfBstride=0;
    // per-step activation scratch (sized to Tmax tokens × widest k=n_embd)
    uint8_t *d_moe_afp4=nullptr,*d_moe_asf=nullptr; float *d_moe_gsa=nullptr,*d_moe_amax=nullptr;
    int *d_moe_indptr=nullptr,*d_moe_t2e=nullptr; __nv_bfloat16 *d_moe_obf=nullptr;
    float *d_moe_alpha=nullptr;            // device alpha (gsw·gsa) — no per-call D2H
    cudaStream_t moe_stream=nullptr; cudaEvent_t moe_done=nullptr, moe_route=nullptr; // overlap MoE w/ dense FFN
};

static uint64_t dg_xs=0x2545F4914F6CDD1Dull;
static inline double dg_urand(){ dg_xs^=dg_xs<<13; dg_xs^=dg_xs>>7; dg_xs^=dg_xs<<17; return (dg_xs>>11)*(1.0/9007199254740992.0); }

extern "C" dg_engine_t *dg_engine_create(const char *path, int max_prompt, int fp4_moe){
    if(max_prompt<=0) max_prompt=1024;
    dg_engine *e=new dg_engine();
    e->max_prompt=max_prompt;
    e->fp4moe_want = fp4_moe ? 1 : 0;   // NVFP4 MoE experts (CLI --fp4-moe / -dm)
    int cl=dg_gguf_canvas_length(path); if(cl>0) e->canvas_length=cl;
    e->N_max=max_prompt+e->canvas_length;

    int fd=open(path,O_RDONLY); if(fd<0){delete e;return nullptr;}
    struct stat st; fstat(fd,&st); e->mmap_size=st.st_size;
    e->mmap_base=(uint8_t*)mmap(nullptr,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    if(e->mmap_base==MAP_FAILED){delete e;return nullptr;}
    const uint8_t *p=e->mmap_base+8; uint64_t nt=*(uint64_t*)p; p+=8; uint64_t nkv=*(uint64_t*)p; p+=8;
    auto rd_str=[&](const uint8_t*&q){ uint64_t l=*(uint64_t*)q; q+=8; std::string s((const char*)q,l); q+=l; return s; };
    auto skip_val=[&](const uint8_t*&q,uint32_t vt){ if(vt==GTYPE_STR){uint64_t l=*(uint64_t*)q;q+=8+l;} else if(vt==GTYPE_ARR){uint32_t at=*(uint32_t*)q;q+=4;uint64_t c=*(uint64_t*)q;q+=8; if(at==GTYPE_STR){for(uint64_t i=0;i<c;i++){uint64_t l=*(uint64_t*)q;q+=8+l;}} else q+=c*dg_scalar_sz(at);} else q+=dg_scalar_sz(vt); };
    for(uint64_t i=0;i<nkv;i++){ rd_str(p); uint32_t vt=*(uint32_t*)p; p+=4; skip_val(p,vt); }
    for(uint64_t i=0;i<nt;i++){ std::string nm=rd_str(p); DGTensor t; t.ndim=*(uint32_t*)p; p+=4; for(int d=0;d<t.ndim;d++){t.ne[d]=*(int64_t*)p;p+=8;} t.type=*(uint32_t*)p;p+=4; t.offset=*(uint64_t*)p;p+=8; t.nelem=1; for(int d=0;d<t.ndim;d++)t.nelem*=t.ne[d]; int64_t rb=dg_row_bytes(t.type,t.ne[0]),rows=1; for(int d=1;d<t.ndim;d++)rows*=t.ne[d]; t.nbytes=rb*rows; e->T[nm]=t; }
    uint64_t off=(uint64_t)(p-e->mmap_base); off=(off+31)&~31ull; const uint8_t *data=e->mmap_base+off;
    for(auto&kv:e->T){ DGTensor&t=kv.second; CKR(cudaMalloc(&t.dev,t.nbytes),nullptr); CKR(cudaMemcpy(t.dev,data+t.offset,t.nbytes,cudaMemcpyHostToDevice),nullptr); }

    if(cublasCreate(&e->cub)!=CUBLAS_STATUS_SUCCESS){delete e;return nullptr;}
    cublasSetMathMode(e->cub,CUBLAS_DEFAULT_MATH);    // enables bf16 tensor-core gemmEx; sgemm stays fp32

    // Cap max_prompt to fit GPU memory. Every prompt token costs ~1 MB of scratch + KV cache
    // (the N-sized activation buffers + the per-layer prompt K/V), so the model's full 262144
    // context would need ~256 GB. Reserve headroom for the vocab buffers (~5 GB), the cuBLAS
    // workspace + fragmentation, and the lazily-built NVFP4 MoE weights (~13 GB) so a large
    // --ctx caps gracefully here instead of OOMing at startup.
    {
        size_t freeB=0,totalB=0; cudaMemGetInfo(&freeB,&totalB);
        // Chunked prefill makes activation/MoE scratch constant (CH-sized), so a prompt token now
        // costs only its bf16 K/V cache: Σ_layers 2·hd·nkv·2 B ≈ 225 KB/token (was ~850 KB).
        const size_t per_tok = 240000;                                    // bytes / prompt token (bf16 KV only)
        size_t reserve = ((size_t)(e->fp4moe_want ? 26ULL : 12ULL)) << 30; // GB (vocab bufs + cuBLAS + FP4 MoE)
        size_t budget = (freeB > reserve) ? (freeB - reserve) : freeB/2;
        int feasible = (int)(budget / per_tok); if(feasible < 1024) feasible = 1024;
        if(e->max_prompt > feasible){
            fprintf(stderr,"dg: max_prompt %d exceeds GPU-memory budget — capping to %d tokens "
                    "(%.1f GB free)\n", e->max_prompt, feasible, freeB/1e9);
            e->max_prompt = feasible; max_prompt = feasible;   // keep the local param in sync (Tmax etc.)
            e->N_max = e->max_prompt + e->canvas_length;
        }
    }

    // Activation/MoE scratch is sized for the busiest single forward pass, not the whole prompt:
    // prefill runs DG_PREFILL_CHUNK tokens at a time and the canvas forward runs C — so ABN tokens
    // is the most any buffer ever holds. Only the per-layer K/V cache (d_pk/d_pv) scales with the
    // full prompt. This is the chunked-prefill memory win: scratch is constant, ~4× larger context.
    const int n_embd=e->n_embd,n_ff=e->n_ff,n_head=e->n_head,vocab=e->vocab,C=e->canvas_length;
    const int ABN=(DG_PREFILL_CHUNK>C?DG_PREFILL_CHUNK:C), N=ABN;
    CKR(cudaMalloc(&e->d_wscr,(size_t)26000000*4),nullptr);
    CKR(cudaMalloc(&e->d_tok,(size_t)vocab*n_embd*4),nullptr);
    { DGTensor&te=e->T.at("token_embd.weight"); dg_dequant(te.type,te.dev,te.nelem,e->d_tok,0); }
    // bf16 copy of token_embd for the LM-head + self-cond tensor-core GEMMs, plus bf16 scratch.
    CKR(cudaMalloc(&e->d_tok_bf,(size_t)vocab*n_embd*2),nullptr);
    dg_f32_to_bf16(e->d_tok,e->d_tok_bf,(int64_t)vocab*n_embd,0);
    CKR(cudaMalloc(&e->d_actbf,(size_t)vocab*C*2),nullptr);
    CKR(cudaDeviceSynchronize(),nullptr);
    auto AB=[&](size_t nf){ float*q; cudaMalloc(&q,nf*4); return q; };
    CKR(cudaMalloc(&e->d_ids,N*4),nullptr); CKR(cudaMalloc(&e->d_pos,N*4),nullptr);
    e->d_inpL=AB((size_t)n_embd*N); e->d_cur=AB((size_t)n_embd*N); e->d_attnout=AB((size_t)n_embd*N);
    e->d_q=AB((size_t)DG_GLOBAL_HEAD_DIM*n_head*N); e->d_kraw=AB((size_t)DG_GLOBAL_HEAD_DIM*n_head*N);
    e->d_k=AB((size_t)DG_GLOBAL_HEAD_DIM*n_head*N); e->d_v=AB((size_t)DG_GLOBAL_HEAD_DIM*n_head*N); e->d_attn=AB((size_t)DG_GLOBAL_HEAD_DIM*n_head*N);
    e->d_dense=AB((size_t)n_embd*N); e->d_moe=AB((size_t)n_embd*N); e->d_ffn=AB((size_t)n_embd*N);
    e->d_tmp=AB((size_t)n_ff*N); e->d_tmp2=AB((size_t)n_ff*N); e->d_rtmp=AB((size_t)n_embd*N);
    e->d_rlogits=AB((size_t)DG_N_EXPERTS*N); e->d_moeout=AB((size_t)n_embd*N);
    // grouped MoE scratch, sized for the most assignments in one forward: Tmax = ABN tokens × 8.
    e->Tmax=ABN*DG_N_EXPERTS_USED;
    const int Tmax=e->Tmax;
    e->d_xe_all=AB((size_t)n_embd*Tmax); e->d_gu_all=AB((size_t)2*DG_EXPERT_FFN*Tmax);
    e->d_act_all=AB((size_t)DG_EXPERT_FFN*Tmax); e->d_oe_all=AB((size_t)n_embd*Tmax);
    CKR(cudaMalloc(&e->d_eidx_all,(size_t)Tmax*4),nullptr); e->d_ecs_all=AB((size_t)Tmax);
    CKR(cudaMalloc(&e->d_q8,(size_t)n_embd*Tmax),nullptr);            // int8 activation, max in_dim=n_embd
    e->d_q8d=AB((size_t)(n_embd/32)*Tmax); CKR(cudaMalloc(&e->d_q8s,(size_t)(n_embd/32)*Tmax*4),nullptr);
    // per-active-expert grouped-MMQ args (tiny — built + uploaded once per MoE layer).
    CKR(cudaMalloc(&e->d_count,(size_t)DG_N_EXPERTS*4),nullptr);
    CKR(cudaMalloc(&e->d_coloff,(size_t)DG_N_EXPERTS*4),nullptr);
    CKR(cudaMalloc(&e->d_cursor,(size_t)DG_N_EXPERTS*4),nullptr);
    CKR(cudaMalloc(&e->d_tki,N*DG_N_EXPERTS_USED*4),nullptr); e->d_tkw=AB((size_t)N*DG_N_EXPERTS_USED);
    e->d_clogits=AB((size_t)vocab*C);
    e->d_sc=AB((size_t)vocab*C); e->d_scprob=AB((size_t)vocab*C); e->d_soft=AB((size_t)n_embd*C); e->d_scn=AB((size_t)n_embd*C);
    e->d_scg=AB((size_t)n_ff*C); e->d_scu=AB((size_t)n_ff*C); e->d_scsig=AB((size_t)n_embd*C); e->d_rnd=AB((size_t)C);
    CKR(cudaMalloc(&e->d_sample,C*4),nullptr); CKR(cudaMalloc(&e->d_argmax,C*4),nullptr); e->d_ent=AB((size_t)C);
    // per-layer prompt K/V cache (sliding: 256×8, global: 512×2) sized for the largest prompt.
    for(int il=0;il<DG_MAX_LAYERS;il++){ bool global=(il%6==5);
        int hd=global?DG_GLOBAL_HEAD_DIM:DG_HEAD_DIM, nkv=global?DG_KV_HEADS_GLOBAL:DG_KV_HEADS_SLIDING;
        CKR(cudaMalloc(&e->d_pk[il],(size_t)hd*nkv*e->max_prompt*sizeof(__nv_bfloat16)),nullptr);
        CKR(cudaMalloc(&e->d_pv[il],(size_t)hd*nkv*e->max_prompt*sizeof(__nv_bfloat16)),nullptr); }
    e->rope_freqs=&e->T.at("rope_freqs.weight");
    CKR(cudaDeviceSynchronize(),nullptr);
    return (dg_engine_t*)e;
}

extern "C" int dg_engine_canvas_length(const dg_engine_t *eng){ return eng? ((const dg_engine*)eng)->canvas_length : -1; }
// The ACTUAL prompt window after the GPU-memory cap in dg_engine_create (≤ the requested value).
extern "C" int dg_engine_max_prompt(const dg_engine_t *eng){ return eng? ((const dg_engine*)eng)->max_prompt : -1; }

extern "C" void dg_engine_free(dg_engine_t *eng){
    if(!eng) return; dg_engine *e=(dg_engine*)eng;
    for(auto&kv:e->T){ if(kv.second.dev) cudaFree(kv.second.dev); if(kv.second.dev_bf) cudaFree(kv.second.dev_bf);
        if(kv.second.dev_fp4) cudaFree(kv.second.dev_fp4); if(kv.second.dev_fp4sc) cudaFree(kv.second.dev_fp4sc); }
    float* ptrs[]={e->d_wscr,e->d_tok,e->d_inpL,e->d_cur,e->d_attnout,e->d_q,e->d_kraw,e->d_k,e->d_v,e->d_attn,e->d_dense,e->d_moe,e->d_ffn,e->d_tmp,e->d_tmp2,e->d_rtmp,e->d_rlogits,e->d_moeout,e->d_xe_all,e->d_gu_all,e->d_act_all,e->d_oe_all,e->d_ecs_all,e->d_q8d,e->d_tkw,e->d_clogits,e->d_sc,e->d_scprob,e->d_soft,e->d_scn,e->d_scg,e->d_scu,e->d_scsig,e->d_rnd,e->d_ent};
    for(float*q:ptrs) if(q) cudaFree(q);
    if(e->d_q8) cudaFree(e->d_q8);
    if(e->d_tok_bf) cudaFree(e->d_tok_bf); if(e->d_actbf) cudaFree(e->d_actbf);
    int* iptrs[]={e->d_ids,e->d_pos,e->d_eidx_all,e->d_tki,e->d_sample,e->d_argmax,e->d_count,e->d_coloff,e->d_cursor,e->d_q8s};
    for(int*q:iptrs) if(q) cudaFree(q);
    for(int il=0;il<DG_MAX_LAYERS;il++){ if(e->d_pk[il]) cudaFree(e->d_pk[il]); if(e->d_pv[il]) cudaFree(e->d_pv[il]); }
    if(e->cub) cublasDestroy(e->cub);
    if(e->d_fp4_act) cudaFree(e->d_fp4_act); if(e->d_fp4_actsc) cudaFree(e->d_fp4_actsc); if(e->d_fp4_actlin) cudaFree(e->d_fp4_actlin);
    if(e->d_fp4_gsact) cudaFree(e->d_fp4_gsact); if(e->d_fp4_alpha) cudaFree(e->d_fp4_alpha); if(e->d_fp4_amax) cudaFree(e->d_fp4_amax);
    if(e->fp4_ws) cudaFree(e->fp4_ws);
    if(e->fp4_desc) cublasLtMatmulDescDestroy(e->fp4_desc);
    if(e->lt) cublasLtDestroy(e->lt);
    // NVFP4 MoE expert buffers
    if(e->fp4moe_ready){
        for(int il=0;il<DG_MAX_LAYERS;il++){
            if(e->d_gu_fp4[il]) cudaFree(e->d_gu_fp4[il]); if(e->d_gu_sf[il]) cudaFree(e->d_gu_sf[il]);
            if(e->d_dn_fp4[il]) cudaFree(e->d_dn_fp4[il]); if(e->d_dn_sf[il]) cudaFree(e->d_dn_sf[il]); }
        if(e->d_moe_afp4) cudaFree(e->d_moe_afp4); if(e->d_moe_asf) cudaFree(e->d_moe_asf);
        if(e->d_moe_indptr) cudaFree(e->d_moe_indptr); if(e->d_moe_t2e) cudaFree(e->d_moe_t2e);
        if(e->d_moe_gsa) cudaFree(e->d_moe_gsa); if(e->d_moe_amax) cudaFree(e->d_moe_amax);
        if(e->d_moe_alpha) cudaFree(e->d_moe_alpha); if(e->d_moe_obf) cudaFree(e->d_moe_obf);
        if(e->moe_stream) cudaStreamDestroy(e->moe_stream);
        if(e->moe_done) cudaEventDestroy(e->moe_done); if(e->moe_route) cudaEventDestroy(e->moe_route);
    }
    if(e->mmap_base&&e->mmap_base!=MAP_FAILED) munmap(e->mmap_base,e->mmap_size);
    delete e;
}

// ─── NVFP4 (block-scaled FP4) dense GEMM — DG_FP4 ───────────────────────
// Mirrors the gemma4 NVFP4 prefill path: per-16-elem E4M3 block scales (cuBLASLt
// VEC16_UE4M3, 32×4×4 swizzle) + per-tensor fp32 global scale folded into a
// device-pointer alpha. dg_ prefix avoids colliding with libfucina's symbols
// (the final binary links both archives). ~2-3× the bf16 tensor-core GEMM @ N=256.
#define DG_NVFP4_BLK 16
static inline int dg_nvfp4_pad(int x,int m){ return ((x+m-1)/m)*m; }

__global__ void dg_nvfp4_amax_bf16(const __nv_bfloat16 *x, uint64_t n, float *amax){
    uint64_t i=blockIdx.x*(uint64_t)blockDim.x+threadIdx.x;
    float v=(i<n)?fabsf(__bfloat162float(x[i])):0.f;
    for(int o=16;o>0;o>>=1) v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,o));
    __shared__ float s[32]; int lane=threadIdx.x&31,wid=threadIdx.x>>5;
    if(lane==0)s[wid]=v; __syncthreads();
    if(wid==0){ v=(lane<(blockDim.x+31)/32)?s[lane]:0.f;
        for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,o));
        if(lane==0)atomicMax((int*)amax,__float_as_int(v)); }
}
__global__ void dg_nvfp4_gs(const float *amax,float *gs){ float a=*amax; *gs=(a>0.f)?a/(6.0f*448.0f):1e-30f; }
__global__ void dg_nvfp4_alpha(const float *gsw,const float *gsa,float *al){ al[0]=(*gsw)*(*gsa); al[1]=0.f; }
__global__ void dg_nvfp4_quant_bf16(const __nv_bfloat16 *__restrict__ X, uint8_t *__restrict__ fp4,
        uint8_t *__restrict__ bscale, int rows, int k, const float *gsp){
    int row=blockIdx.y, blk=blockIdx.x*blockDim.x+threadIdx.x, nblk=k/DG_NVFP4_BLK;
    if(row>=rows||blk>=nblk) return;
    float gs=*gsp; const __nv_bfloat16 *xr=X+(size_t)row*k+blk*DG_NVFP4_BLK;
    float v[DG_NVFP4_BLK],amax=0.f;
    #pragma unroll
    for(int i=0;i<DG_NVFP4_BLK;i++){ v[i]=__bfloat162float(xr[i]); amax=fmaxf(amax,fabsf(v[i])); }
    float bs=(amax>0.f)?amax/6.0f/gs:0.f;
    __nv_fp8_storage_t e=__nv_cvt_float_to_fp8(bs,__NV_SATFINITE,__NV_E4M3);
    bscale[(size_t)row*nblk+blk]=(uint8_t)e;
    float div=gs*__half2float(__half(__nv_cvt_fp8_to_halfraw(e,__NV_E4M3))); if(div<=0.f)div=1e30f;
    uint8_t *o=fp4+(size_t)row*(k/2)+blk*(DG_NVFP4_BLK/2);
    #pragma unroll
    for(int i=0;i<DG_NVFP4_BLK;i+=2){
        float2 p=make_float2(v[i]/div,v[i+1]/div);
        o[i/2]=(uint8_t)__nv_cvt_float2_to_fp4x2(p,__NV_E2M1,cudaRoundNearest);
    }
}
__global__ void dg_nvfp4_swizzle(const uint8_t *__restrict__ lin,uint8_t *__restrict__ sw,
        int outer,int nblk,int nblk_pad){
    int o=blockIdx.y*blockDim.y+threadIdx.y, s=blockIdx.x*blockDim.x+threadIdx.x;
    if(o>=outer||s>=nblk) return;
    int oo=o/128,oi=o%128,so=s/4,si=s%4;
    sw[((size_t)oo*(nblk_pad/4)+so)*512+(oi%32)*16+(oi/32)*4+si]=lin[(size_t)o*nblk+s];
}
// quantize bf16 [rows×k] → packed fp4 + swizzled E4M3 scales + global scale (device).
static void dg_nvfp4_quantize(const __nv_bfloat16 *X,int rows,int k,uint8_t *fp4,
        uint8_t *lin,uint8_t *sw,float *gsp,float *amax,cudaStream_t st){
    cudaMemsetAsync(amax,0,sizeof(float),st);
    uint64_t n=(uint64_t)rows*k;
    dg_nvfp4_amax_bf16<<<(unsigned)((n+255)/256),256,0,st>>>(X,n,amax);
    dg_nvfp4_gs<<<1,1,0,st>>>(amax,gsp);
    int nblk=k/DG_NVFP4_BLK; dim3 b(256),g((nblk+255)/256,rows);
    dg_nvfp4_quant_bf16<<<g,b,0,st>>>(X,fp4,lin,rows,k,gsp);
    dim3 b2(32,8),g2((nblk+31)/32,(rows+7)/8);
    dg_nvfp4_swizzle<<<g2,b2,0,st>>>(lin,sw,rows,nblk,dg_nvfp4_pad(nblk,4));
}
// one-time cuBLASLt setup (handle, desc w/ device-ptr alpha + VEC16 scale modes, workspace,
// activation global-scale/alpha/amax scalars). idempotent. returns 1 on success.
static int dg_fp4_ensure(dg_engine *e){
    if(e->fp4_enabled>=0) return e->fp4_enabled;
    const char *env=getenv("DG_FP4"); int want=(env&&env[0]=='1')?1:0;
    if(!want){ e->fp4_enabled=0; return 0; }
    if(cublasLtCreate(&e->lt)!=CUBLAS_STATUS_SUCCESS){ e->fp4_enabled=0; return 0; }
    cublasLtMatmulDescCreate(&e->fp4_desc,CUBLAS_COMPUTE_32F,CUDA_R_32F);
    cublasOperation_t opT=CUBLAS_OP_T,opN=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(e->fp4_desc,CUBLASLT_MATMUL_DESC_TRANSA,&opT,sizeof(opT));
    cublasLtMatmulDescSetAttribute(e->fp4_desc,CUBLASLT_MATMUL_DESC_TRANSB,&opN,sizeof(opN));
    int32_t sm=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(e->fp4_desc,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&sm,sizeof(sm));
    cublasLtMatmulDescSetAttribute(e->fp4_desc,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&sm,sizeof(sm));
    int32_t pm=CUBLASLT_POINTER_MODE_DEVICE;
    cublasLtMatmulDescSetAttribute(e->fp4_desc,CUBLASLT_MATMUL_DESC_POINTER_MODE,&pm,sizeof(pm));
    cudaMalloc(&e->fp4_ws,64ull<<20);
    cudaMalloc(&e->d_fp4_gsact,sizeof(float));
    cudaMalloc(&e->d_fp4_alpha,2*sizeof(float));
    cudaMalloc(&e->d_fp4_amax,sizeof(float));
    fprintf(stderr,"dg: NVFP4 dense GEMM ON (DG_FP4=1) — cuBLASLt block-scaled FP4 tensor cores\n");
    e->fp4_enabled=1; return 1;
}
// grow the activation NVFP4 scratch to hold [k×N] (k≤ max in_dim). sized to n_ff (widest).
static int dg_fp4_act(dg_engine *e,int k,int N){
    size_t need=(size_t)N; if(need<=e->fp4_act_cap && e->d_fp4_act) return 0;
    if(e->d_fp4_act)cudaFree(e->d_fp4_act); if(e->d_fp4_actsc)cudaFree(e->d_fp4_actsc); if(e->d_fp4_actlin)cudaFree(e->d_fp4_actlin);
    int K=e->n_ff>e->n_embd?e->n_ff:e->n_embd; if(k>K)K=k;
    size_t packed=(size_t)N*(K/2), sw=(size_t)dg_nvfp4_pad(N,128)*dg_nvfp4_pad(K/DG_NVFP4_BLK,4), lin=(size_t)N*(K/DG_NVFP4_BLK);
    if(cudaMalloc(&e->d_fp4_act,packed)!=cudaSuccess){e->fp4_act_cap=0;return -1;}
    if(cudaMalloc(&e->d_fp4_actsc,sw)!=cudaSuccess){e->fp4_act_cap=0;return -1;}
    if(cudaMalloc(&e->d_fp4_actlin,lin)!=cudaSuccess){e->fp4_act_cap=0;return -1;}
    e->fp4_act_cap=N; return 0;
}
// NVFP4 dense GEMM: out[outN] = W[out×in] @ X[in×N]. Builds W's NVFP4 cache once (from
// W.dev_bf), quantizes the bf16 activation per call. Returns 0 / -1 (caller falls back).
static int dg_dense_mm_fp4(dg_engine *e, float *out, DGTensor &W, int N){
    int in=(int)W.ne[0], outd=(int)W.ne[1];
    if((in%DG_NVFP4_BLK)||(outd%DG_NVFP4_BLK)) return -1;   // need 16-mult dims
    if(dg_fp4_act(e,in,N)!=0) return -1;
    if(!W.dev_fp4){
        if(!W.dev_bf){ if(cudaMalloc(&W.dev_bf,(size_t)in*outd*2)!=cudaSuccess) return -1;
            dg_dequant_bf16(W.type,W.dev,(int64_t)in*outd,W.dev_bf,0); }
        size_t packed=(size_t)outd*(in/2), sw=(size_t)dg_nvfp4_pad(outd,128)*dg_nvfp4_pad(in/DG_NVFP4_BLK,4);
        if(cudaMalloc(&W.dev_fp4,packed)!=cudaSuccess) return -1;
        if(cudaMalloc(&W.dev_fp4sc,sw)!=cudaSuccess) return -1;
        cudaMemset(W.dev_fp4sc,0,sw);
        float *gsw; cudaMalloc(&gsw,sizeof(float));   // temp: bake W global scale into W.gs_w
        uint8_t *lin; cudaMalloc(&lin,(size_t)outd*(in/DG_NVFP4_BLK));
        dg_nvfp4_quantize(W.dev_bf,outd,in,W.dev_fp4,lin,W.dev_fp4sc,gsw,e->d_fp4_amax,0);
        cudaMemcpy(&W.gs_w,gsw,sizeof(float),cudaMemcpyDeviceToHost);
        cudaFree(gsw); cudaFree(lin);
    }
    // activation: convert x→bf16 already done by caller into d_actbf; quantize it
    int nblk=in/DG_NVFP4_BLK;
    cudaMemsetAsync(e->d_fp4_actsc,0,(size_t)dg_nvfp4_pad(N,128)*dg_nvfp4_pad(nblk,4),0);
    cudaMemsetAsync(e->d_fp4_amax,0,sizeof(float),0);
    uint64_t nn=(uint64_t)N*in;
    dg_nvfp4_amax_bf16<<<(unsigned)((nn+255)/256),256>>>(e->d_actbf,nn,e->d_fp4_amax);
    dg_nvfp4_gs<<<1,1>>>(e->d_fp4_amax,e->d_fp4_gsact);
    dim3 b(256),g((nblk+255)/256,N);
    dg_nvfp4_quant_bf16<<<g,b>>>(e->d_actbf,e->d_fp4_act,e->d_fp4_actlin,N,in,e->d_fp4_gsact);
    dim3 b2(32,8),g2((nblk+31)/32,(N+7)/8);
    dg_nvfp4_swizzle<<<g2,b2>>>(e->d_fp4_actlin,e->d_fp4_actsc,N,nblk,dg_nvfp4_pad(nblk,4));
    // alpha = gs_w · gs_act (weight gs uploaded once)
    static float *gsw_dev=nullptr; if(!gsw_dev) cudaMalloc(&gsw_dev,sizeof(float));
    cudaMemcpy(gsw_dev,&W.gs_w,sizeof(float),cudaMemcpyHostToDevice);
    dg_nvfp4_alpha<<<1,1>>>(gsw_dev,e->d_fp4_gsact,e->d_fp4_alpha);
    cublasLtMatmulDescSetAttribute(e->fp4_desc,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&W.dev_fp4sc,sizeof(void*));
    cublasLtMatmulDescSetAttribute(e->fp4_desc,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&e->d_fp4_actsc,sizeof(void*));
    cublasLtMatrixLayout_t Ad,Bd,Dd;
    cublasLtMatrixLayoutCreate(&Ad,CUDA_R_4F_E2M1,in,outd,in);
    cublasLtMatrixLayoutCreate(&Bd,CUDA_R_4F_E2M1,in,N,in);
    cublasLtMatrixLayoutCreate(&Dd,CUDA_R_32F,outd,N,outd);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    size_t ws=64ull<<20;
    cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws));
    cublasLtMatmulHeuristicResult_t res[4]; int found=0;
    cublasLtMatmulAlgoGetHeuristic(e->lt,e->fp4_desc,Ad,Bd,Dd,Dd,pref,4,res,&found);
    int rc=-1;
    if(found>0){
        cublasStatus_t s=cublasLtMatmul(e->lt,e->fp4_desc,e->d_fp4_alpha,W.dev_fp4,Ad,
            e->d_fp4_act,Bd,e->d_fp4_alpha+1,out,Dd,out,Dd,&res[0].algo,e->fp4_ws,ws,0);
        rc=(s==CUBLAS_STATUS_SUCCESS)?0:-1;
    }
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Ad);cublasLtMatrixLayoutDestroy(Bd);cublasLtMatrixLayoutDestroy(Dd);
    return rc;
}

// ─── NVFP4 MoE experts (--fp4-moe) — grouped block-scaled FP4 tensor-core GEMM ───
extern "C" int dg_fp4_moe_grouped(void* D, const void* A, const void* A_sf, const void* B,
    const void* B_sf, const int* m_indptr, int num_groups, int N, int K,
    unsigned long long sfA_stride, unsigned long long sfB_stride, const float* alpha, cudaStream_t s);
extern "C" void dg_fp4_sf_strides(int M_max, int N, int K,
    unsigned long long* sfA_stride, unsigned long long* sfB_stride);

// amax over [n] fp32 → atomicMax into a device float (init 0; values ≥0 → int order ok)
__global__ void dg_nvfp4_amax_f32(const float* x, uint64_t n, float* amax){
    uint64_t i=blockIdx.x*(uint64_t)blockDim.x+threadIdx.x;
    float v=(i<n)?fabsf(x[i]):0.f;
    for(int o=16;o>0;o>>=1) v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,o));
    __shared__ float s[32]; int lane=threadIdx.x&31,wid=threadIdx.x>>5;
    if(lane==0)s[wid]=v; __syncthreads();
    if(wid==0){ v=(lane<(blockDim.x+31)/32)?s[lane]:0.f;
        for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,o));
        if(lane==0)atomicMax((int*)amax,__float_as_int(v)); }
}
// counting-sort outputs (coloff/count) → m_indptr[E+1] + tok→expert map [total]
__global__ void dg_build_grp_idx(const int* coloff, const int* count, int E, int total,
    int* indptr, int* t2e){
    int ex=blockIdx.x*blockDim.x+threadIdx.x;
    if(ex<E){ int off=coloff[ex], c=count[ex]; indptr[ex]=off; for(int j=0;j<c;j++) t2e[off+j]=ex; }
    if(ex==0) indptr[E]=total;
}
// fused per-expert activation NVFP4 quant: X[total][k] fp32 (grouped by expert) → packed E2M1
// [total][k/2] dense + per-expert padded swizzled E4M3 SF (the layout CUTLASS reads).
__global__ void dg_nvfp4_quant_grp(const float* __restrict__ X, const int* __restrict__ t2e,
    const int* __restrict__ indptr, int k, int total, const float* gsp,
    uint8_t* __restrict__ A_fp4, uint8_t* __restrict__ A_sf,
    unsigned long long sfAstride, int nKvec_pad){
    int t=blockIdx.y, blk=blockIdx.x*blockDim.x+threadIdx.x, nblk=k/DG_NVFP4_BLK;
    if(t>=total||blk>=nblk) return;
    float gs=*gsp; const float* xr=X+(size_t)t*k+blk*DG_NVFP4_BLK;
    float v[DG_NVFP4_BLK],amax=0.f;
    #pragma unroll
    for(int i=0;i<DG_NVFP4_BLK;i++){ v[i]=xr[i]; amax=fmaxf(amax,fabsf(v[i])); }
    float bs=(amax>0.f)?amax/6.0f/gs:0.f;
    __nv_fp8_storage_t e8=__nv_cvt_float_to_fp8(bs,__NV_SATFINITE,__NV_E4M3);
    float div=gs*__half2float(__half(__nv_cvt_fp8_to_halfraw(e8,__NV_E4M3))); if(div<=0.f)div=1e30f;
    uint8_t* o=A_fp4+(size_t)t*(k/2)+blk*(DG_NVFP4_BLK/2);
    #pragma unroll
    for(int i=0;i<DG_NVFP4_BLK;i+=2){ float2 p=make_float2(v[i]/div,v[i+1]/div);
        o[i/2]=(uint8_t)__nv_cvt_float2_to_fp4x2(p,__NV_E2M1,cudaRoundNearest); }
    int ex=t2e[t], row=t-indptr[ex];
    int so=blk>>2, si=blk&3, om=row&31, im=(row&127)>>5, mt=row>>7, nKtiles=nKvec_pad>>2;
    size_t off=(size_t)ex*sfAstride + (size_t)mt*nKtiles*512 + (size_t)so*512 + (size_t)om*16 + im*4 + si;
    A_sf[off]=(uint8_t)e8;
}
__global__ void dg_bf16tof32(const __nv_bfloat16* x, float* y, uint64_t n){
    uint64_t i=blockIdx.x*(uint64_t)blockDim.x+threadIdx.x; if(i<n) y[i]=__bfloat162float(x[i]); }
// alpha = gsw (host const, baked into the kernel arg) · gsa (device) — device scalar, no D2H.
__global__ void dg_alpha_kernel(float gsw, const float* gsa, float* al){ *al = gsw * (*gsa); }

// Build persistent NVFP4 expert weights for all layers (gate_up + down), shared global per
// layer-proj. One-time, on first --fp4-moe use. From the GGUF Q4_K/Q8_0/Q5_0 experts.
static int dg_fp4_moe_build(dg_engine* e){
    if(e->fp4moe_ready) return 0;
    int n_embd=e->n_embd, E=DG_N_EXPERTS, gu_out=2*DG_EXPERT_FFN, dn_in=DG_EXPERT_FFN;
    { unsigned long long a,b; dg_fp4_sf_strides(0,gu_out,n_embd,&a,&b); e->gu_sfBstride=b; }
    { unsigned long long a,b; dg_fp4_sf_strides(0,n_embd,dn_in,&a,&b); e->dn_sfBstride=b; }
    size_t gu_elem=(size_t)gu_out*n_embd, dn_elem=(size_t)n_embd*dn_in, maxelem=gu_elem>dn_elem?gu_elem:dn_elem;
    __nv_bfloat16* tbf; uint8_t* tlin; float *gsdev,*amaxdev;
    if(cudaMalloc(&tbf,maxelem*2)!=cudaSuccess) return -1;
    cudaMalloc(&tlin,(size_t)gu_out*(n_embd/DG_NVFP4_BLK));
    cudaMalloc(&gsdev,4); cudaMalloc(&amaxdev,4);
    char nm[80]; size_t wbytes=0;
    auto build_proj=[&](DGTensor& W,int out_dim,int in_dim,uint8_t** wfp4,uint8_t** wsf,
                        unsigned long long sfB,float* gsw_out)->int{
        int64_t slab=dg_row_bytes(W.type,W.ne[0])*out_dim; size_t elem=(size_t)out_dim*in_dim;
        if(cudaMalloc(wfp4,(size_t)E*out_dim*(in_dim/2))!=cudaSuccess) return -1;
        if(cudaMalloc(wsf,(size_t)E*sfB)!=cudaSuccess) return -1; cudaMemset(*wsf,0,(size_t)E*sfB);
        float wamax=0.f,h;
        for(int ex=0;ex<E;ex++){ dg_dequant_bf16(W.type,(uint8_t*)W.dev+(size_t)ex*slab,elem,tbf,0);
            cudaMemset(amaxdev,0,4); dg_nvfp4_amax_bf16<<<(unsigned)((elem+255)/256),256>>>(tbf,elem,amaxdev);
            cudaMemcpy(&h,amaxdev,4,cudaMemcpyDeviceToHost); wamax=fmaxf(wamax,h); }
        float gsw=wamax/(6.f*448.f); if(gsw<=0)gsw=1e-30f; *gsw_out=gsw; cudaMemcpy(gsdev,&gsw,4,cudaMemcpyHostToDevice);
        int nblk=in_dim/DG_NVFP4_BLK, nblk_pad=dg_nvfp4_pad(nblk,4);
        for(int ex=0;ex<E;ex++){ dg_dequant_bf16(W.type,(uint8_t*)W.dev+(size_t)ex*slab,elem,tbf,0);
            dim3 b(256),g((nblk+255)/256,out_dim);
            dg_nvfp4_quant_bf16<<<g,b>>>(tbf,*wfp4+(size_t)ex*out_dim*(in_dim/2),tlin,out_dim,in_dim,gsdev);
            dim3 b2(32,8),g2((nblk+31)/32,(out_dim+7)/8);
            dg_nvfp4_swizzle<<<g2,b2>>>(tlin,*wsf+(size_t)ex*sfB,out_dim,nblk,nblk_pad); }
        wbytes += (size_t)E*out_dim*(in_dim/2) + (size_t)E*sfB; return 0;
    };
    for(int il=0; il<DG_MAX_LAYERS; il++){
        snprintf(nm,80,"blk.%d.ffn_gate_up_exps.weight",il); DGTensor& gW=e->T.at(nm);
        snprintf(nm,80,"blk.%d.ffn_down_exps.weight",il);    DGTensor& dW=e->T.at(nm);
        if(build_proj(gW,gu_out,n_embd,&e->d_gu_fp4[il],&e->d_gu_sf[il],e->gu_sfBstride,&e->gu_gsw[il])) return -1;
        if(build_proj(dW,n_embd,dn_in,&e->d_dn_fp4[il],&e->d_dn_sf[il],e->dn_sfBstride,&e->dn_gsw[il])) return -1;
    }
    cudaFree(tbf); cudaFree(tlin); cudaFree(gsdev); cudaFree(amaxdev); cudaDeviceSynchronize();
    if(cudaGetLastError()!=cudaSuccess){ fprintf(stderr,"dg: NVFP4 MoE weight build error\n"); return -1; }
    // per-step activation scratch (sized to Tmax tokens × widest k=n_embd)
    int maxn=e->Tmax/DG_N_EXPERTS_USED;
    unsigned long long sfA_alloc=(unsigned long long)dg_nvfp4_pad(maxn,128)*dg_nvfp4_pad(n_embd/DG_NVFP4_BLK,4);
    cudaMalloc(&e->d_moe_afp4,(size_t)e->Tmax*(n_embd/2));
    cudaMalloc(&e->d_moe_asf,(size_t)E*sfA_alloc);
    cudaMalloc(&e->d_moe_indptr,(size_t)(E+1)*4); cudaMalloc(&e->d_moe_t2e,(size_t)e->Tmax*4);
    cudaMalloc(&e->d_moe_gsa,4); cudaMalloc(&e->d_moe_amax,4); cudaMalloc(&e->d_moe_alpha,4);
    cudaMalloc(&e->d_moe_obf,(size_t)e->Tmax*n_embd*sizeof(__nv_bfloat16));   // n_embd ≥ 2*EXPERT_FFN
    cudaStreamCreate(&e->moe_stream);
    cudaEventCreateWithFlags(&e->moe_done,cudaEventDisableTiming);
    cudaEventCreateWithFlags(&e->moe_route,cudaEventDisableTiming);
    e->fp4moe_ready=1;
    fprintf(stderr,"dg: NVFP4 MoE experts built (%.2f GB, CUTLASS grouped FP4 ~2.56x dp4a)\n",wbytes/1e9);
    return 0;
}
// one MoE projection in NVFP4: Dout[total][out_dim] fp32 = grouped( X[total][k] @ W ).
static int dg_fp4_moe_gemm(dg_engine* e, const float* X, int k, int out_dim, int total, int n,
    uint8_t* Wfp4, uint8_t* Wsf, float gsw, unsigned long long sfBstride, float* Dout, cudaStream_t st){
    int E=DG_N_EXPERTS, nblk=k/DG_NVFP4_BLK, nblk_pad=dg_nvfp4_pad(nblk,4);
    unsigned long long sfAstride=(unsigned long long)dg_nvfp4_pad(n,128)*nblk_pad;   // M_max ≤ n
    dg_build_grp_idx<<<(E+255)/256,256,0,st>>>(e->d_coloff,e->d_count,E,total,e->d_moe_indptr,e->d_moe_t2e);
    cudaMemsetAsync(e->d_moe_amax,0,4,st);
    dg_nvfp4_amax_f32<<<(unsigned)(((size_t)total*k+255)/256),256,0,st>>>(X,(uint64_t)total*k,e->d_moe_amax);
    dg_nvfp4_gs<<<1,1,0,st>>>(e->d_moe_amax,e->d_moe_gsa);
    cudaMemsetAsync(e->d_moe_asf,0,(size_t)E*sfAstride,st);
    dim3 b(256),g((nblk+255)/256,total);
    dg_nvfp4_quant_grp<<<g,b,0,st>>>(X,e->d_moe_t2e,e->d_moe_indptr,k,total,e->d_moe_gsa,
                                e->d_moe_afp4,e->d_moe_asf,sfAstride,nblk_pad);
    dg_alpha_kernel<<<1,1,0,st>>>(gsw,e->d_moe_gsa,e->d_moe_alpha);   // device alpha, no D2H
    int rc=dg_fp4_moe_grouped(e->d_moe_obf,e->d_moe_afp4,e->d_moe_asf,Wfp4,Wsf,e->d_moe_indptr,
                              E,out_dim,k,sfAstride,sfBstride,e->d_moe_alpha,st);
    if(rc!=0){ fprintf(stderr,"[dg] fp4 moe gemm rc=%d\n",rc); return -1; }
    dg_bf16tof32<<<(unsigned)(((size_t)total*out_dim+255)/256),256,0,st>>>(e->d_moe_obf,Dout,(uint64_t)total*out_dim);
    return 0;
}

// Dense projection matmul: dequant→cuBLAS sgemm. (Fused dp4a MMQ was A/B-tested here and is a wash
// for the full N=256 dense shapes — cuBLAS sgemm already matches it — so dense stays on cuBLAS while
// the SKINNY expert path uses the fused grouped kernels where it wins big.) Column-major [feat,tok].
static int dg_dense_mm(dg_engine *e, float *out, DGTensor &W, const float *x, int N){
    const float one=1.f,zero=0.f;
    if(W.type==DG_GGML_F32){   // router only (small) — keep fp32 sgemm
        if(cublasSgemm(e->cub,CUBLAS_OP_T,CUBLAS_OP_N,(int)W.ne[1],N,(int)W.ne[0],&one,(const float*)W.dev,(int)W.ne[0],x,(int)W.ne[0],&zero,out,(int)W.ne[1])!=CUBLAS_STATUS_SUCCESS) return -1;
        return 0;
    }
    // Dense weights are constant across denoising steps → dequant to bf16 ONCE and cache (was
    // re-dequantized every step). Then per step: convert activation → bf16, bf16 tensor-core GEMM.
    if(!W.dev_bf){ if(cudaMalloc(&W.dev_bf,(size_t)W.ne[0]*W.ne[1]*2)!=cudaSuccess) return -1;
        dg_dequant_bf16(W.type,W.dev,W.ne[0]*W.ne[1],W.dev_bf,0); }
    dg_f32_to_bf16(x,e->d_actbf,(int64_t)W.ne[0]*N,0);
    // DG_FP4_DEBUG: compute FP4 into a scratch, bf16 into out, log L2rel/NaN, use bf16.
    static int dbg=-1; if(dbg<0){const char*d=getenv("DG_FP4_DEBUG");dbg=(d&&d[0]=='1')?1:0;}
    if(dbg && dg_fp4_ensure(e)){
        static float* sc=nullptr; static size_t scn=0; size_t need=(size_t)W.ne[1]*N;
        if(need>scn){ if(sc)cudaFree(sc); cudaMalloc(&sc,need*4); scn=need; }
        if(dg_dense_mm_fp4(e,sc,W,N)==0){
            cublasGemmEx(e->cub,CUBLAS_OP_T,CUBLAS_OP_N,(int)W.ne[1],N,(int)W.ne[0],&one,
                W.dev_bf,CUDA_R_16BF,(int)W.ne[0],e->d_actbf,CUDA_R_16BF,(int)W.ne[0],
                &zero,out,CUDA_R_32F,(int)W.ne[1],CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            static int cnt=0;
            if(cnt<8){ std::vector<float> a(need),b(need);
                cudaMemcpy(a.data(),sc,need*4,cudaMemcpyDeviceToHost);
                cudaMemcpy(b.data(),out,need*4,cudaMemcpyDeviceToHost);
                double se=0,sr=0,mx=0; int nan=0;
                for(size_t i=0;i<need;i++){ if(!isfinite(a[i]))nan++; double e2=(double)a[i]-b[i]; se+=e2*e2; sr+=(double)b[i]*b[i]; if(fabs(e2)>mx)mx=fabs(e2); }
                fprintf(stderr,"[DG_FP4_DEBUG] %dx%d N=%d  L2rel=%.3f%% max|d|=%.4f nan=%d\n",
                    (int)W.ne[1],(int)W.ne[0],N,100*sqrt(se/(sr+1e-30)),mx,nan); cnt++; }
            return 0;
        }
    }
    // NVFP4 tensor-core fast path (DG_FP4=1) — falls through to bf16 on any failure.
    if(dg_fp4_ensure(e) && dg_dense_mm_fp4(e,out,W,N)==0) return 0;
    if(cublasGemmEx(e->cub,CUBLAS_OP_T,CUBLAS_OP_N,(int)W.ne[1],N,(int)W.ne[0],&one,
                    W.dev_bf,CUDA_R_16BF,(int)W.ne[0], e->d_actbf,CUDA_R_16BF,(int)W.ne[0],
                    &zero,out,CUDA_R_32F,(int)W.ne[1],CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)!=CUBLAS_STATUS_SUCCESS) return -1;
    return 0;
}

// ── MoE block shared by prefill and the canvas forward ──────────────────
// Routes `n` tokens of d_attnout through the parallel dense FFN + top-8/128 expert MoE and
// writes the combined, post-normed result into e->d_ffn. (Identical math to the original
// single-pass forward; only the token count `n` changes between prompt prefill and canvas.)
static int dg_layer_ffn(dg_engine *e, int il, int n, int canvas){
    const int n_embd=e->n_embd,n_ff=e->n_ff; const float eps=DG_RMS_EPS;
    auto& T=e->T;
    auto G=[&](const std::string&nm)->DGTensor&{ return T.at(nm); };
    char b[64]; auto LL=[&](const char*s){ snprintf(b,64,"blk.%d.%s",il,s); return std::string(b); };
    auto mm=[&](float*out,DGTensor&W,const float*x,int T_)->int{ return dg_dense_mm(e,out,W,x,T_); };
    // NVFP4 MoE (CLI --fp4-moe / -dm → fp4moe_want): CUTLASS grouped FP4 tensor-core experts
    // (~2.56× dp4a). Used for the canvas (256 tok) AND full prefill chunks — chunked prefill packs
    // ≥256 tok/forward (≥32 tok/expert), so the 128-row M-tile padding is amortized as well as the
    // canvas; only tiny tail chunks (<256 tok, few tok/expert) fall back to dp4a. Built once on use.
    if(e->fp4moe_enabled<0){ e->fp4moe_enabled=e->fp4moe_want;
        if(e->fp4moe_enabled && dg_fp4_moe_build(e)!=0){ fprintf(stderr,"dg: NVFP4 MoE build failed — dp4a fallback\n"); e->fp4moe_enabled=0; } }
    bool fp4moe = e->fp4moe_enabled && e->fp4moe_ready && (canvas || n>=DG_CANVAS_LENGTH);

    DGTensor&gue=G(LL("ffn_gate_up_exps.weight")), &dwe=G(LL("ffn_down_exps.weight"));
    int64_t gslab=dg_row_bytes(gue.type,gue.ne[0])*(2*DG_EXPERT_FFN), dslab=dg_row_bytes(dwe.type,dwe.ne[0])*n_embd;
    int total=n*DG_N_EXPERTS_USED;
    if(total>e->Tmax){ fprintf(stderr,"[dg] MoE total %d > Tmax %d\n",total,e->Tmax); return -1; }

    // ── MoE routing (stream 0): router GEMM + counting-sort + gather ──
    float* d_moein=e->d_rtmp; dg_rmsnorm(d_moein,e->d_attnout,(float*)G(LL("pre_ffw_norm_2.weight")).dev,n_embd,n,eps,0);
    dg_rmsnorm(e->d_cur,e->d_attnout,nullptr,n_embd,n,eps,0); dg_scale(e->d_cur,(int64_t)n_embd*n,1.0f/sqrtf((float)n_embd),0);
    dg_mul_vec_cols(e->d_cur,(float*)G(LL("ffn_gate_inp.scale")).dev,n_embd,n,0);
    if(mm(e->d_rlogits,G(LL("ffn_gate_inp.weight")),e->d_cur,n))return -1;
    dg_softmax_topk(e->d_rlogits,DG_N_EXPERTS,n,DG_N_EXPERTS_USED,e->d_tki,e->d_tkw,0);
    dg_moe_route(e->d_tki,e->d_tkw,(float*)G(LL("ffn_down_exps.scale")).dev,n,DG_N_EXPERTS_USED,
                 DG_N_EXPERTS,e->d_count,e->d_coloff,e->d_cursor,e->d_eidx_all,e->d_ecs_all,0);
    dg_gather_cols(e->d_xe_all,d_moein,e->d_eidx_all,n_embd,total,0);

    // ── experts ── FP4: on moe_stream, overlapped with the dense FFN on stream 0 (disjoint buffers).
    if(fp4moe){
        cudaEventRecord(e->moe_route,0);                       // stream-0 routing+gather done
        cudaStreamWaitEvent(e->moe_stream,e->moe_route,0);
        cudaMemsetAsync(e->d_moeout,0,(size_t)n_embd*n*4,e->moe_stream);
        cudaStream_t ms=e->moe_stream;
        if(dg_fp4_moe_gemm(e,e->d_xe_all,n_embd,2*DG_EXPERT_FFN,total,n,e->d_gu_fp4[il],e->d_gu_sf[il],e->gu_gsw[il],e->gu_sfBstride,e->d_gu_all,ms)) return -1;
        dg_split_gelu_mul(e->d_act_all,e->d_gu_all,DG_EXPERT_FFN,total,ms);
        if(dg_fp4_moe_gemm(e,e->d_act_all,DG_EXPERT_FFN,n_embd,total,n,e->d_dn_fp4[il],e->d_dn_sf[il],e->dn_gsw[il],e->dn_sfBstride,e->d_oe_all,ms)) return -1;
        dg_scatteradd_cols(e->d_moeout,e->d_oe_all,e->d_eidx_all,e->d_ecs_all,n_embd,total,ms);
        cudaEventRecord(e->moe_done,e->moe_stream);
    } else {
        CKR(cudaMemset(e->d_moeout,0,(size_t)n_embd*n*4),-1);  // dp4a grouped, all on stream 0
        dg_quantize_q8_1(e->d_xe_all,e->d_q8,e->d_q8d,e->d_q8s,n_embd,total,0);
        dg_mmq_q4_K_grouped(e->d_gu_all,gue.dev,gslab,e->d_q8,e->d_q8d,e->d_q8s,e->d_coloff,e->d_count,DG_N_EXPERTS,n_embd,2*DG_EXPERT_FFN,0);
        dg_split_gelu_mul(e->d_act_all,e->d_gu_all,DG_EXPERT_FFN,total,0);
        dg_quantize_q8_1(e->d_act_all,e->d_q8,e->d_q8d,e->d_q8s,DG_EXPERT_FFN,total,0);
        if(dwe.type==DG_GGML_Q8_0)      dg_mmq_q8_0_grouped(e->d_oe_all,dwe.dev,dslab,e->d_q8,e->d_q8d,e->d_q8s,e->d_coloff,e->d_count,DG_N_EXPERTS,DG_EXPERT_FFN,n_embd,0);
        else if(dwe.type==DG_GGML_Q5_0) dg_mmq_q5_0_grouped(e->d_oe_all,dwe.dev,dslab,e->d_q8,e->d_q8d,e->d_q8s,e->d_coloff,e->d_count,DG_N_EXPERTS,DG_EXPERT_FFN,n_embd,0);
        else { fprintf(stderr,"[dg] unexpected ffn_down_exps type %d (layer %d)\n",dwe.type,il); return -1; }
        dg_scatteradd_cols(e->d_moeout,e->d_oe_all,e->d_eidx_all,e->d_ecs_all,n_embd,total,0);
    }

    // ── dense FFN (stream 0) — runs concurrently with the FP4 experts on moe_stream ──
    dg_rmsnorm(e->d_cur,e->d_attnout,(float*)G(LL("ffn_norm.weight")).dev,n_embd,n,eps,0);
    if(mm(e->d_tmp,G(LL("ffn_gate.weight")),e->d_cur,n))return -1; if(mm(e->d_tmp2,G(LL("ffn_up.weight")),e->d_cur,n))return -1;
    dg_gelu_mul(e->d_tmp,e->d_tmp,e->d_tmp2,(int64_t)n_ff*n,0); if(mm(e->d_dense,G(LL("ffn_down.weight")),e->d_tmp,n))return -1;
    dg_rmsnorm(e->d_dense,e->d_dense,(float*)G(LL("post_ffw_norm_1.weight")).dev,n_embd,n,eps,0);

    // ── combine (stream 0 waits for the experts) ──
    if(fp4moe) cudaStreamWaitEvent(0,e->moe_done,0);
    dg_rmsnorm(e->d_moe,e->d_moeout,(float*)G(LL("post_ffw_norm_2.weight")).dev,n_embd,n,eps,0);
    dg_add(e->d_ffn,e->d_dense,e->d_moe,(int64_t)n_embd*n,0);
    dg_rmsnorm(e->d_ffn,e->d_ffn,(float*)G(LL("post_ffw_norm.weight")).dev,n_embd,n,eps,0);
    return 0;
}

// PREFILL (range) — causal forward over `n` tokens, appending each layer's K/V into the cache at
// token offset `base` (so it serves later blocks). Runs CHUNK-MAJOR: each DG_PREFILL_CHUNK-token
// slice flows through all 30 layers, attending the already-cached prefix [0,c0) (bf16) + its own
// fresh K/V (fp32), CAUSALLY at GLOBAL positions c0.. — so activation/MoE scratch only ever holds
// CH tokens, not the whole context. No LM head. Used for the initial prompt (base=0) AND to extend
// the context with each committed answer block in multi-block generation (base=current ctx length).
static int dg_prefill_range(dg_engine *e, const int32_t *toks, int n, int base){
    const int n_embd=e->n_embd,n_head=e->n_head; const float eps=DG_RMS_EPS,esc=sqrtf((float)n_embd);
    auto& T=e->T;
    auto G=[&](const std::string&nm)->DGTensor&{ return T.at(nm); };
    auto mm=[&](float*out,DGTensor&W,const float*x,int T_)->int{ return dg_dense_mm(e,out,W,x,T_); };
    char b[64];
    for(int off=0;off<n;off+=DG_PREFILL_CHUNK){
        const int ch=(n-off<DG_PREFILL_CHUNK)?(n-off):DG_PREFILL_CHUNK;  // tokens in this chunk
        const int c0=base+off;                                          // global position / cache offset
        std::vector<int> ids(ch); for(int i=0;i<ch;i++)ids[i]=toks[off+i];
        CKR(cudaMemcpy(e->d_ids,ids.data(),ch*4,cudaMemcpyHostToDevice),-1);
        std::vector<int> ph(ch); for(int i=0;i<ch;i++)ph[i]=c0+i;       // GLOBAL positions for rope
        CKR(cudaMemcpy(e->d_pos,ph.data(),ch*4,cudaMemcpyHostToDevice),-1);
        dg_embed_gather(e->d_inpL,e->d_tok,e->d_ids,n_embd,ch,esc,0);
        for(int il=0;il<DG_MAX_LAYERS;il++){
            bool global=(il%6==5); bool sliding=!global;
            int hd=sliding?DG_HEAD_DIM:DG_GLOBAL_HEAD_DIM, nkv=sliding?DG_KV_HEADS_SLIDING:DG_KV_HEADS_GLOBAL;
            float theta=sliding?DG_ROPE_THETA_SLIDING:DG_ROPE_THETA_GLOBAL;
            const float* ff=sliding?nullptr:(const float*)e->rope_freqs->dev;
            auto LL=[&](const char*s){ snprintf(b,64,"blk.%d.%s",il,s); return std::string(b); };
            dg_rmsnorm(e->d_cur,e->d_inpL,(float*)G(LL("attn_norm.weight")).dev,n_embd,ch,eps,0);
            if(mm(e->d_q,G(LL("attn_q.weight")),e->d_cur,ch))return -1;
            dg_head_rmsnorm(e->d_q,(float*)G(LL("attn_q_norm.weight")).dev,hd,n_head,ch,eps,0); dg_rope(e->d_q,e->d_pos,hd,n_head,ch,theta,ff,0);
            if(mm(e->d_kraw,G(LL("attn_k.weight")),e->d_cur,ch))return -1;
            CKR(cudaMemcpy(e->d_k,e->d_kraw,(size_t)hd*nkv*ch*4,cudaMemcpyDeviceToDevice),-1);
            dg_head_rmsnorm(e->d_k,(float*)G(LL("attn_k_norm.weight")).dev,hd,nkv,ch,eps,0); dg_rope(e->d_k,e->d_pos,hd,nkv,ch,theta,ff,0);
            if(sliding){ if(mm(e->d_v,G(LL("attn_v.weight")),e->d_cur,ch))return -1; }
            else CKR(cudaMemcpy(e->d_v,e->d_kraw,(size_t)hd*nkv*ch*4,cudaMemcpyDeviceToDevice),-1);
            dg_head_rmsnorm(e->d_v,nullptr,hd,nkv,ch,eps,0);
            // chunk queries attend cached prefix K/V [0,c0) (bf16) + this chunk's fresh K/V (fp32), causally
            dg_attention_chunk(e->d_attn,e->d_q,e->d_pk[il],e->d_pv[il],e->d_k,e->d_v,hd,n_head,nkv,c0,ch,DG_SLIDING_WINDOW,sliding?1:0,0);
            // append this chunk's K/V to the cache at token offset c0 (bf16 — ½ the fp32 footprint)
            dg_f32_to_bf16(e->d_k,e->d_pk[il]+(size_t)hd*nkv*c0,(int64_t)hd*nkv*ch,0);
            dg_f32_to_bf16(e->d_v,e->d_pv[il]+(size_t)hd*nkv*c0,(int64_t)hd*nkv*ch,0);
            if(mm(e->d_cur,G(LL("attn_output.weight")),e->d_attn,ch))return -1;
            dg_rmsnorm(e->d_cur,e->d_cur,(float*)G(LL("post_attention_norm.weight")).dev,n_embd,ch,eps,0);
            dg_add(e->d_attnout,e->d_cur,e->d_inpL,(int64_t)n_embd*ch,0);
            if(dg_layer_ffn(e,il,ch,0))return -1;
            dg_add(e->d_inpL,e->d_ffn,e->d_attnout,(int64_t)n_embd*ch,0);
            dg_mul_region_scalar(e->d_inpL,n_embd,0,ch,(float*)G(LL("enc_layer_output_scale.weight")).dev,0);
        }
    }
    CKR(cudaDeviceSynchronize(),-1);
    e->cached_P=base+n;
    return 0;
}
static int dg_prefill(dg_engine *e, const int32_t *prompt, int P){ return dg_prefill_range(e,prompt,P,0); }

// CANVAS forward — one bidirectional denoising pass over the C-token canvas (run every step).
// Reads the cached prompt K/V (e->d_pk/d_pv) plus fresh canvas K/V; writes canvas logits to
// e->d_clogits. Requires dg_prefill() to have run for this prompt first.
// active_lo: frozen-position skipping (DG_FREEZE) forwards only the unconverged canvas SUFFIX
// [active_lo,Cw). The converged prefix [0,active_lo) has been committed to the K/V cache, so P is
// the EXTENDED prefix length (ctx_len+active_lo) and the canvas positions resume at P. Because d_sc
// and d_clogits are column-major [vocab,Cw], the self-cond input and LM-head output stay aligned to
// absolute canvas columns by simply offsetting their pointers by active_lo — no buffer shifts. With
// active_lo==0 this is byte-for-byte the original full-canvas forward.
static int dg_forward_canvas(dg_engine *e, int P, const std::vector<int> &canvas, int active_lo, float sc_use){
    const int n_embd=e->n_embd,n_ff=e->n_ff,n_head=e->n_head,vocab=e->vocab;
    const int C=(int)canvas.size()-active_lo; const float eps=DG_RMS_EPS,esc=sqrtf((float)n_embd),one=1.f,zero=0.f;
    auto& T=e->T; cublasHandle_t cub=e->cub;
    auto G=[&](const std::string&n)->DGTensor&{ return T.at(n); };
    auto mm=[&](float*out,DGTensor&W,const float*x,int T_)->int{ return dg_dense_mm(e,out,W,x,T_); };
    float* sc_in=e->d_sc+(size_t)active_lo*vocab;        // self-cond reads this step's active columns
    float* lg_out=e->d_clogits+(size_t)active_lo*vocab;  // LM head writes the active columns
    std::vector<int> ids(canvas.begin()+active_lo,canvas.end());
    CKR(cudaMemcpy(e->d_ids,ids.data(),C*4,cudaMemcpyHostToDevice),-1);
    std::vector<int> ph(C); for(int i=0;i<C;i++)ph[i]=P+i; CKR(cudaMemcpy(e->d_pos,ph.data(),C*4,cudaMemcpyHostToDevice),-1);
    dg_embed_gather(e->d_inpL,e->d_tok,e->d_ids,n_embd,C,esc,0);
    float* cvemb=e->d_inpL;     // the whole buffer is the canvas now
    if(sc_use>0.5f){
        dg_softmax_cols(sc_in,e->d_scprob,vocab,C,0);
        dg_f32_to_bf16(e->d_scprob,e->d_actbf,(int64_t)vocab*C,0);   // self-cond soft-embed: bf16 TC GEMM
        CB(cublasGemmEx(cub,CUBLAS_OP_N,CUBLAS_OP_N,n_embd,C,vocab,&one,e->d_tok_bf,CUDA_R_16BF,n_embd,
                        e->d_actbf,CUDA_R_16BF,vocab,&zero,e->d_soft,CUDA_R_32F,n_embd,
                        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        dg_scale(e->d_soft,(int64_t)n_embd*C,esc,0);
        dg_rmsnorm(e->d_scn,e->d_soft,(float*)G("self_cond_pre_norm.weight").dev,n_embd,C,eps,0);
        mm(e->d_scg,G("self_cond_gate.weight"),e->d_scn,C); mm(e->d_scu,G("self_cond_up.weight"),e->d_scn,C);
        dg_gelu_mul(e->d_scg,e->d_scg,e->d_scu,(int64_t)n_ff*C,0);
        mm(e->d_scsig,G("self_cond_down.weight"),e->d_scg,C);
        dg_add(cvemb,cvemb,e->d_scsig,(int64_t)n_embd*C,0);
    }
    dg_rmsnorm(cvemb,cvemb,nullptr,n_embd,C,eps,0);
    char b[64];
    for(int il=0;il<DG_MAX_LAYERS;il++){
        bool global=(il%6==5); bool sliding=!global;
        int hd=sliding?DG_HEAD_DIM:DG_GLOBAL_HEAD_DIM, nkv=sliding?DG_KV_HEADS_SLIDING:DG_KV_HEADS_GLOBAL;
        float theta=sliding?DG_ROPE_THETA_SLIDING:DG_ROPE_THETA_GLOBAL;
        const float* ff=sliding?nullptr:(const float*)e->rope_freqs->dev;
        auto LL=[&](const char*s){ snprintf(b,64,"blk.%d.%s",il,s); return std::string(b); };
        dg_rmsnorm(e->d_cur,e->d_inpL,(float*)G(LL("attn_norm.weight")).dev,n_embd,C,eps,0);
        if(mm(e->d_q,G(LL("attn_q.weight")),e->d_cur,C))return -1;
        dg_head_rmsnorm(e->d_q,(float*)G(LL("attn_q_norm.weight")).dev,hd,n_head,C,eps,0); dg_rope(e->d_q,e->d_pos,hd,n_head,C,theta,ff,0);
        if(mm(e->d_kraw,G(LL("attn_k.weight")),e->d_cur,C))return -1;
        CKR(cudaMemcpy(e->d_k,e->d_kraw,(size_t)hd*nkv*C*4,cudaMemcpyDeviceToDevice),-1);
        dg_head_rmsnorm(e->d_k,(float*)G(LL("attn_k_norm.weight")).dev,hd,nkv,C,eps,0); dg_rope(e->d_k,e->d_pos,hd,nkv,C,theta,ff,0);
        if(sliding){ if(mm(e->d_v,G(LL("attn_v.weight")),e->d_cur,C))return -1; }
        else CKR(cudaMemcpy(e->d_v,e->d_kraw,(size_t)hd*nkv*C*4,cudaMemcpyDeviceToDevice),-1);
        dg_head_rmsnorm(e->d_v,nullptr,hd,nkv,C,eps,0);
        // canvas queries attend cached prefix K/V (P) + fresh canvas K/V (C). On sliding layers,
        // prefix keys older than the window are masked to 0 for EVERY canvas query — so skip them
        // entirely: start the prefix at lo=P-SWA+1 (BIT-EXACT, caps these 25/30 layers at ~1024
        // prefix keys no matter how long the multi-block context grows, and shrinks the smem too).
        int lo = sliding ? (P - DG_SLIDING_WINDOW + 1) : 0; if(lo < 0) lo = 0;
        const __nv_bfloat16 *pk = e->d_pk[il] + (size_t)lo*nkv*hd, *pv = e->d_pv[il] + (size_t)lo*nkv*hd;
        dg_attention_canvas(e->d_attn,e->d_q,pk,pv,e->d_k,e->d_v,hd,n_head,nkv,P-lo,C,DG_SLIDING_WINDOW,sliding?1:0,0);
        if(mm(e->d_cur,G(LL("attn_output.weight")),e->d_attn,C))return -1;
        dg_rmsnorm(e->d_cur,e->d_cur,(float*)G(LL("post_attention_norm.weight")).dev,n_embd,C,eps,0);
        dg_add(e->d_attnout,e->d_cur,e->d_inpL,(int64_t)n_embd*C,0);
        if(dg_layer_ffn(e,il,C,1))return -1;
        dg_add(e->d_inpL,e->d_ffn,e->d_attnout,(int64_t)n_embd*C,0);
        dg_mul_region_scalar(e->d_inpL,n_embd,0,C,(float*)G(LL("layer_output_scale.weight")).dev,0);
    }
    dg_rmsnorm(e->d_cur,e->d_inpL,(float*)G("output_norm.weight").dev,n_embd,C,eps,0);
    dg_f32_to_bf16(e->d_cur,e->d_actbf,(int64_t)n_embd*C,0);          // LM head: bf16 TC GEMM
    CB(cublasGemmEx(cub,CUBLAS_OP_T,CUBLAS_OP_N,vocab,C,n_embd,&one,e->d_tok_bf,CUDA_R_16BF,n_embd,
                    e->d_actbf,CUDA_R_16BF,n_embd,&zero,lg_out,CUDA_R_32F,vocab,
                    CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    dg_softcap(lg_out,(int64_t)vocab*C,DG_SOFTCAP,0);
    // no sync here — the caller pipelines scale+sample on the same stream and syncs once before D2H
    return 0;
}

// WARMUP — run a dummy prefill + 2 canvas forwards at load time so the FIRST real request doesn't
// pay one-time costs mid-answer: the 12.85 GB NVFP4 MoE build, cuBLAS algo selection, and kernel
// module loads. Uses a 256-token dummy prompt (exercises the FP4 prefill-chunk path) + both the
// sc_use=0 and self-cond (sc_use=1) canvas paths. Results are discarded; cache is reset after.
extern "C" void dg_engine_warmup(dg_engine_t *eng){
    dg_engine *e=(dg_engine*)eng;
    if(!e) return;
    const int wn=DG_CANVAS_LENGTH;
    std::vector<int32_t> dummy(wn,DG_BOS_ID);
    if(dg_prefill(e,dummy.data(),wn)!=0) return;
    std::vector<int> canvas(e->canvas_length,DG_MASK_ID);
    cudaMemset(e->d_sc,0,(size_t)e->vocab*e->canvas_length*4);     // valid self-cond input for step 2
    dg_forward_canvas(e,wn,canvas,0,0.f);
    dg_forward_canvas(e,wn,canvas,0,1.f);
    cudaDeviceSynchronize();
    e->cached_P=0;                                                  // let the first real request re-prefill cleanly
}

extern "C" int dg_engine_generate(dg_engine_t *eng, const int32_t *prompt, int n_prompt,
                                  int max_steps, float t_min, float t_max, float entropy_bound,
                                  uint64_t seed, int eot_id, int32_t *out_ids, int max_out){
    dg_engine *e=(dg_engine*)eng;
    if(!e||n_prompt<=0||n_prompt>e->max_prompt) return -1;
    if(max_steps<=0) max_steps=DG_MAX_DENOISE_STEPS;
    const int vocab=e->vocab, C=e->canvas_length;
    dg_xs = seed ? (seed|1ull) : 0x2545F4914F6CDD1Dull;
    cudaEvent_t ev0,ev1,ev2; cudaEventCreate(&ev0); cudaEventCreate(&ev1); cudaEventCreate(&ev2);
    cudaEventRecord(ev0);

    // Prefill the prompt once: cache per-layer prompt K/V so each denoising step only forwards the
    // 256-token canvas. (ev0→ev1 = prompt prefill; ev1→ev2 = the whole multi-block answer, which
    // includes both the denoise loops AND the inter-block context extensions.)
    if(dg_prefill(e,prompt,n_prompt)!=0) return -1;
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    int nsteps=0, total=0, ctx_len=n_prompt;

    // Frozen-position skipping (DG_FREEZE, experimental): once a contiguous canvas PREFIX converges
    // (argmax stable for FREEZE_STREAK steps AND low entropy), commit it to the causal K/V cache and
    // shrink the active window to the suffix — later steps then forward fewer positions, cutting the
    // MoE wall as the block converges. APPROXIMATE: frozen positions stop attending the future, so
    // their cached K/V are causal not bidirectional → OFF by default, A/B'd for quality.
    const bool freeze = getenv("DG_FREEZE")!=nullptr;
    const bool frz_dbg = getenv("DG_FREEZE_DEBUG")!=nullptr;
    // A position joins the converged prefix once its argmax holds for FREEZE_STREAK steps with
    // per-token entropy < FREEZE_ENT. FREEZE_ENT is LOOSE (a position is confident long before the
    // whole block hits the global 0.005 convergence bar) — overridable via env to tune the
    // quality/speed trade. Commit when the prefix grows ≥FREEZE_MIN (amortizes the commit forward).
    int FREEZE_STREAK=2, FREEZE_MIN=8; float FREEZE_ENT=0.10f;
    if(const char*s=getenv("DG_FREEZE_ENT")) FREEZE_ENT=atof(s);
    if(const char*s=getenv("DG_FREEZE_STREAK")) FREEZE_STREAK=atoi(s);
    if(const char*s=getenv("DG_FREEZE_MIN")) FREEZE_MIN=atoi(s);

    std::vector<int> canvas(C), argmax_canvas(C,-1), prev_argmax(C,-2), hsample(C), hargmax(C), streak(C,0);
    std::vector<float> hent(C), hrnd(C);
    // MULTI-BLOCK: DiffusionGemma denoises one 256-token block at a time. After committing a block
    // we append its tokens to the causal K/V cache (dg_prefill_range at ctx_len) and decode the
    // next block — chaining blocks until a block emits EOS/end-of-turn, the output buffer fills, or
    // the cache is full. This is what lets answers exceed one canvas (code, long-form).
    bool done=false;
    while(!done){
        for(int i=0;i<C;i++) canvas[i]=(int)(dg_urand()*vocab);   // fresh random canvas per block
        std::fill(prev_argmax.begin(),prev_argmax.end(),-2);
        std::fill(streak.begin(),streak.end(),0);
        int active_lo=0;            // frozen prefix [0,active_lo) already committed to the cache this block
        float sc_use=0.f;
        for(int step=max_steps; step>=1; step--){
            if(active_lo>=C) break;                              // whole block frozen+committed
            const int Cact=C-active_lo;
            float* lg=e->d_clogits+(size_t)active_lo*vocab;      // active logit columns
            float Ttemp=t_min+(t_max-t_min)*((float)step/max_steps);
            if(dg_forward_canvas(e,ctx_len+active_lo,canvas,active_lo,sc_use)!=0) return -1;
            dg_scale(lg,(int64_t)vocab*Cact,1.0f/Ttemp,0);
            for(int i=0;i<Cact;i++) hrnd[i]=(float)dg_urand();
            CKR(cudaMemcpy(e->d_rnd,hrnd.data(),Cact*4,cudaMemcpyHostToDevice),-1);
            dg_sample_step(lg,e->d_rnd,e->d_sample,e->d_argmax,e->d_ent,vocab,Cact,0);
            CKR(cudaDeviceSynchronize(),-1);
            CKR(cudaMemcpy(hsample.data(),e->d_sample,Cact*4,cudaMemcpyDeviceToHost),-1);
            CKR(cudaMemcpy(hargmax.data(),e->d_argmax,Cact*4,cudaMemcpyDeviceToHost),-1);
            CKR(cudaMemcpy(hent.data(),e->d_ent,Cact*4,cudaMemcpyDeviceToHost),-1);
            // entropy-bound accept over the ACTIVE positions (compacted [0,Cact) → absolute active_lo+i)
            std::vector<int> ord(Cact); std::iota(ord.begin(),ord.end(),0);
            std::sort(ord.begin(),ord.end(),[&](int a,int b){return hent[a]<hent[b];});
            std::vector<char> accept(Cact,0); double cum=0;
            for(int r=0;r<Cact;r++){ int c=ord[r]; cum+=hent[c]; if(cum-hent[c]<=entropy_bound) accept[c]=1; }
            for(int i=0;i<Cact;i++){ int c=active_lo+i;
                argmax_canvas[c]=hargmax[i]; canvas[c]=accept[i]?hsample[i]:(int)(dg_urand()*vocab); }
            // self-cond: next step reads this step's logits. Swap the two vocab×C buffers instead of a
            // 256 MB device-to-device copy (still aligned by absolute column — see dg_forward_canvas).
            std::swap(e->d_clogits,e->d_sc); sc_use=1.f;
            double me=0; bool all_stable=true;
            for(int i=0;i<Cact;i++){ int c=active_lo+i; me+=hent[i];
                if(argmax_canvas[c]!=prev_argmax[c]) all_stable=false;
                if(freeze){ bool st=(argmax_canvas[c]==prev_argmax[c] && hent[i]<FREEZE_ENT); streak[c]= st?streak[c]+1:0; }
                prev_argmax[c]=argmax_canvas[c]; }
            me/=(Cact>0?Cact:1);
            nsteps++;
            // FREEZE: grow the converged contiguous prefix (never freezing an EOS/eot position) and,
            // when it advances by ≥FREEZE_MIN and fits the cache, commit that chunk + shrink the window.
            if(freeze){
                int f=active_lo;
                while(f<C && streak[f]>=FREEZE_STREAK && argmax_canvas[f]!=DG_EOS_ID
                      && !(eot_id>0&&argmax_canvas[f]==eot_id)) f++;
                if(f>=active_lo+FREEZE_MIN && ctx_len+f<=e->max_prompt){
                    std::vector<int32_t> blk(argmax_canvas.begin()+active_lo,argmax_canvas.begin()+f);
                    if(dg_prefill_range(e,blk.data(),f-active_lo,ctx_len+active_lo)!=0) return -1;
                    active_lo=f;
                }
            }
            if(all_stable && me<DG_CONFIDENCE_THRESH) break;
        }
        // commit point: frozen prefix [0,active_lo) is already cached. Find the first EOS/end-of-turn
        // across the whole canvas; output [0,blk_n); for chaining append the not-yet-frozen part.
        int blk_n=C; bool stop=false;
        for(int c=0;c<C;c++){ int t=argmax_canvas[c]; if(t==DG_EOS_ID||(eot_id>0&&t==eot_id)){ blk_n=c; stop=true; break; } }
        if(frz_dbg) fprintf(stderr,"[frz] block done: frozen_prefix=%d/%d blk_n=%d steps_so_far=%d\n",active_lo,C,blk_n,nsteps);
        int take=blk_n; if(total+take>max_out) take=max_out-total;
        for(int c=0;c<take;c++) out_ids[total+c]=argmax_canvas[c];
        total+=take;
        if(stop || total>=max_out || ctx_len+blk_n>e->max_prompt || blk_n==0) break;
        // extend the causal context with the part not already frozen, [active_lo,blk_n), then continue
        if(blk_n>active_lo){
            std::vector<int32_t> blk(argmax_canvas.begin()+active_lo,argmax_canvas.begin()+blk_n);
            if(dg_prefill_range(e,blk.data(),blk_n-active_lo,ctx_len+active_lo)!=0) return -1;
        }
        ctx_len+=blk_n;
    }
    cudaEventRecord(ev2); cudaEventSynchronize(ev2);
    float pf=0,dn=0; cudaEventElapsedTime(&pf,ev0,ev1); cudaEventElapsedTime(&dn,ev1,ev2);
    e->last_prefill_ms=pf; e->last_denoise_ms=dn; e->last_steps=nsteps;
    if(getenv("DG_TIMING"))
        fprintf(stderr,"[dg] P=%d out=%d steps=%d  prefill %.1fms  generate %.1fms (%.1fms/step)\n",
                n_prompt,total,nsteps,pf,dn,nsteps?dn/nsteps:0.f);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1); cudaEventDestroy(ev2);
    return total;
}

extern "C" void dg_engine_last_stats(const dg_engine_t *eng, float *prefill_ms,
                                     float *denoise_ms, int *steps){
    if(!eng) return; const dg_engine *e=(const dg_engine*)eng;
    if(prefill_ms)*prefill_ms=e->last_prefill_ms;
    if(denoise_ms)*denoise_ms=e->last_denoise_ms;
    if(steps)*steps=e->last_steps;
}
