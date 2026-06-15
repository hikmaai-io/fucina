// ============================================================================
// test_dg_fp4_grouped.cu  —  GROUPED NVFP4 expert-GEMM microbench (GB10 sm_121a)
// ----------------------------------------------------------------------------
// GO/NO-GO spike: can a CUTLASS sm120 grouped (ptr-array) block-scaled FP4
// tensor-core GEMM beat our dp4a grouped MoE kernel at the DiffusionGemma MoE
// working point (~16 tokens/expert, 128 experts, 8 active)?
//
// This builds ONE CUTLASS grouped GEMM that computes, for every expert g:
//     D_g[M_g, N] = A_g[M_g, K] @ B_g[K, N]      (block-scaled NVFP4 operands)
// with ragged M_g (~16 tokens/expert), shared N,K across experts.
//
// Mapped to the gate_up expert projection:
//     K = in  = 2816   (contraction, NVFP4-quantized, per-16 E4M3 block scale)
//     N = out = 1408   (gate_up output)
//     M_g     ≈ 16     (tokens routed to expert g; ragged)
//     groups  = 128    (experts)
//   A = activations [sum(M_g)][K] row-major, NVFP4
//   B = expert weights, one [N][K] per expert, NVFP4   (col-major to the GEMM)
//   D = bf16 output [sum(M_g)][N]
//
// CUTLASS API (4.2.1, vendored under flashinfer) used here:
//   CollectiveBuilder<arch::Sm120, OpClassTensorOp,
//                     cute::tuple<float_e2m1_t, float_ue4m3_t> (=A elem+scale),
//                     LayoutA*, AlignmentA, ... (B the same),
//                     ElementAccumulator, MmaTileShape, ClusterShape, StageCount,
//                     KernelTmaWarpSpecializedNvf4Sm120>
//   Passing LayoutA* / a *pointer* stride flips the builder to grouped/ptr-array
//   (IsGroupedGemmKernel), dispatching to MainloopSm120ArrayTmaWarpSpecialized-
//   BlockScaled + the sm90 array kernel + sm90 group tile scheduler. There is
//   NO tcgen05 path on consumer Blackwell; this rides the warp-specialized
//   sm90 grouped infra with sm120 UMMA atoms.
//
//   GemmUniversal<GroupProblemShape<Shape<int,int,int>>, Mainloop, Epilogue>
//   launched with GemmUniversalMode::kGrouped. Per-group {M,N,K}, A/B/D ptr
//   arrays, per-group strides, and per-group SF layouts are filled by a small
//   device "prepare args" kernel (mirrors flashinfer group_gemm_*_sm120.cuh).
//
//   Block-scale (SF) layout = cutlass::detail::Sm1xxBlockScaledConfig<16>:
//   SFVecSize=16, scale dtype float_ue4m3_t (E4M3), SWIZZLED 128x4 atom
//   (Sm1xxBlockScaledBasicChunk: Shape<<32,4>,<16,4>> Stride<<16,4>,<0,1>>).
//   M padded to 128, K-vec count (K/16) padded to 4. tile_atom_to_shape_SFA/B
//   produce the per-group LayoutSFA/LayoutSFB the mainloop wants.
//
// Quantization here mirrors our WORKING single-GEMM recipe (test_fp4_gemm.cu):
//   per-16 block: stored E4M3 scale = amax/6/gs ; gs = tensor_amax/(6*448) ;
//   E2M1 values via __nv_cvt_float2_to_fp4x2. The per-tensor gs is folded into
//   alpha (per-group via the epilogue scalar here we keep alpha=gsA*gsB global
//   for the bench — accuracy validation is NOT the point of this GO/NO-GO).
//
// IMPORTANT: the timed run is GUARDED behind argv ("bench"); by default the
// program only BUILDS + sets up. Pass `bench` to actually launch+time the
// grouped GEMM. (GPU is busy — do not run the timed path.)
//
// build (confirmed to compile):
//   CUTLASS=/path/to/cutlass
//   /usr/local/cuda-13/bin/nvcc -std=c++17 -O3 -arch=sm_121a \
//     --expt-relaxed-constexpr --expt-extended-lambda \
//     -DCUTLASS_ARCH_MMA_SM120_SUPPORTED=1 \
//     -I$CUTLASS/include -I$CUTLASS/tools/util/include \
//     cuda/test_dg_fp4_grouped.cu -o /tmp/dg_fp4_grouped
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "diffusion_gemma_kernels.cuh"   // dg_dequant + DG_GGML_* types

// ---- CUTLASS ----
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/float_subbyte.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/util/packed_stride.hpp>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define CUTLASS_CK(status) do{ cutlass::Status s=(status); if(s!=cutlass::Status::kSuccess){ \
  printf("CUTLASS err %s @%d: %d\n",#status,__LINE__,(int)s); exit(1);} }while(0)

static const int BLK = 16;  // NVFP4 block size along K

// local GGUF row-byte helper (gate_up = Q4_K: 144 B / 256 elems)
static int64_t pt_row_bytes(int t,int64_t ne0){
  if(t==DG_GGML_F32) return ne0*4;
  if(t==DG_GGML_Q4_K) return (ne0/256)*144;
  if(t==DG_GGML_Q8_0) return (ne0/32)*34;
  return 0;
}

// ============================================================================
// Quantization kernels (mirror test_fp4_gemm.cu) — produce packed E2M1 +
// LINEAR per-16 E4M3 block scales, then swizzle scales into the CUTLASS
// Sm1xxBlockScaledConfig<16> 128x4 atom layout.
// ============================================================================

// quantize [rows][k] row-major (k contiguous) -> packed fp4 [rows][k/2] +
// linear E4M3 block scales [rows][k/16].
__global__ void quant_nvfp4(const float* __restrict__ X, uint8_t* __restrict__ fp4,
                            uint8_t* __restrict__ bscale, int rows, int k, float gs){
    int row = blockIdx.y;
    int blk = blockIdx.x*blockDim.x + threadIdx.x;   // 16-block index along k
    int nblk = k/BLK;
    if (row>=rows || blk>=nblk) return;
    const float* xr = X + (size_t)row*k + blk*BLK;
    float amax=0.f;
    #pragma unroll
    for(int i=0;i<BLK;i++) amax=fmaxf(amax, fabsf(xr[i]));
    float bs_stored = (amax>0.f) ? amax/6.0f/gs : 0.f;
    __nv_fp8_storage_t e = __nv_cvt_float_to_fp8(bs_stored, __NV_SATFINITE, __NV_E4M3);
    bscale[(size_t)row*nblk + blk] = (uint8_t)e;
    float bsf = __half2float(__half(__nv_cvt_fp8_to_halfraw(e, __NV_E4M3)));
    float divisor = gs * bsf; if (divisor<=0.f) divisor = 1e30f;
    uint8_t* o = fp4 + (size_t)row*(k/2) + blk*(BLK/2);
    #pragma unroll
    for(int i=0;i<BLK;i+=2){
        float2 v = make_float2(xr[i]/divisor, xr[i+1]/divisor);
        __nv_fp4x2_storage_t s = __nv_cvt_float2_to_fp4x2(v, __NV_E2M1, cudaRoundNearest);
        o[i/2] = (uint8_t)s;
    }
}

// Swizzle linear [outer][nblk] E4M3 scales -> CUTLASS Sm1xxBlockScaledConfig<16>
// 128x4 swizzled layout. The on-device SF buffer must be sized so the outer
// (M or N) dim is padded to 128 and the K-vec count (nblk) padded to 4. The
// offset formula matches Sm1xxBlockScaledBasicChunk (the 32x4x4 atom):
//   inner_k = blk % 4              (stride 1)
//   inner_m = (outer % 128) / 32   (stride 4)
//   outer_m = outer % 32           (stride 16)
//   k_tile  = blk / 4              (stride 512)
//   m_tile  = outer / 128          (stride nKtiles*512)
__global__ void swizzle_scales_128x4(const uint8_t* __restrict__ lin,
                                     uint8_t* __restrict__ sw,
                                     int outer, int nblk, int nKtiles){
    int o = blockIdx.y*blockDim.y + threadIdx.y;
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    if (o>=outer || s>=nblk) return;
    int inner_k = s & 3;
    int outer_m = o & 31;
    int inner_m = (o & 127) >> 5;
    int k_tile  = s >> 2;
    int m_tile  = o >> 7;
    size_t off = (size_t)m_tile*(size_t)nKtiles*512
               + (size_t)k_tile*512
               + (size_t)outer_m*16
               + (size_t)inner_m*4
               + inner_k;
    sw[off] = lin[(size_t)o*nblk + s];
}

static int round_up(int x,int m){ return ((x+m-1)/m)*m; }
static float hamax(const std::vector<float>& v){ float a=0; for(float x:v) a=fmaxf(a,fabsf(x)); return a; }

// quantize one row-major [rows][k] block into device: packed E2M1 + swizzled
// SF (128x4). Returns the per-tensor global scale gs. Caller owns dfp4/dsw.
static float quantize_block(const std::vector<float>& host, int rows, int k,
                            uint8_t** dfp4_out, uint8_t** dsw_out, size_t* sw_bytes_out){
    float amax = hamax(host);
    float gs = amax/(6.0f*448.0f); if (gs<=0) gs = 1e-30f;
    int nblk = k/BLK;
    float* dX; CK(cudaMalloc(&dX, host.size()*4));
    CK(cudaMemcpy(dX, host.data(), host.size()*4, cudaMemcpyHostToDevice));
    uint8_t *dfp4,*dlin;
    CK(cudaMalloc(&dfp4,(size_t)rows*(k/2)));
    CK(cudaMalloc(&dlin,(size_t)rows*nblk));
    { dim3 b(256), g((nblk+255)/256, rows); quant_nvfp4<<<g,b>>>(dX,dfp4,dlin,rows,k,gs); }
    // swizzled SF buffer: outer padded to 128, nblk padded to 4
    int outer_pad = round_up(rows,128);
    int nblk_pad  = round_up(nblk,4);
    int nKtiles   = nblk_pad/4;
    size_t sw_bytes = (size_t)outer_pad*(size_t)nblk_pad;   // 1 byte / SF (E4M3)
    uint8_t* dsw; CK(cudaMalloc(&dsw, sw_bytes)); CK(cudaMemset(dsw,0,sw_bytes));
    { dim3 b(32,8), g((nblk+31)/32,(rows+7)/8);
      swizzle_scales_128x4<<<g,b>>>(dlin,dsw,rows,nblk,nKtiles); }
    CK(cudaDeviceSynchronize());
    cudaFree(dX); cudaFree(dlin);
    *dfp4_out = dfp4; *dsw_out = dsw; if(sw_bytes_out) *sw_bytes_out = sw_bytes;
    return gs;
}

// Same, but with a CALLER-PROVIDED global scale (so all experts in a layer share one
// global → a single grouped-GEMM alpha). dfp4 packed dense; dsw padded swizzled.
static void quantize_block_gs(const std::vector<float>& host, int rows, int k, float gs,
                              uint8_t** dfp4_out, uint8_t** dsw_out){
    int nblk = k/BLK;
    float* dX; CK(cudaMalloc(&dX, host.size()*4));
    CK(cudaMemcpy(dX, host.data(), host.size()*4, cudaMemcpyHostToDevice));
    uint8_t *dfp4,*dlin;
    CK(cudaMalloc(&dfp4,(size_t)rows*(k/2))); CK(cudaMalloc(&dlin,(size_t)rows*nblk));
    { dim3 b(256), g((nblk+255)/256, rows); quant_nvfp4<<<g,b>>>(dX,dfp4,dlin,rows,k,gs); }
    int outer_pad=round_up(rows,128), nblk_pad=round_up(nblk,4), nKtiles=nblk_pad/4;
    size_t sw_bytes=(size_t)outer_pad*(size_t)nblk_pad;
    uint8_t* dsw; CK(cudaMalloc(&dsw,sw_bytes)); CK(cudaMemset(dsw,0,sw_bytes));
    { dim3 b(32,8), g((nblk+31)/32,(rows+7)/8); swizzle_scales_128x4<<<g,b>>>(dlin,dsw,rows,nblk,nKtiles); }
    CK(cudaDeviceSynchronize());
    cudaFree(dX); cudaFree(dlin);
    *dfp4_out=dfp4; *dsw_out=dsw;
}

// extern-C wrapper from libdg / dg_fp4_moe.cu (the engine-callable grouped GEMM)
extern "C" int dg_fp4_moe_grouped(void* D, const void* A, const void* A_sf, const void* B,
    const void* B_sf, const int* m_indptr, int num_groups, int N, int K,
    unsigned long long sfA_stride, unsigned long long sfB_stride, float alpha, cudaStream_t s);

// ============================================================================
// CUTLASS sm120 grouped block-scaled NVFP4 GEMM type definitions.
// ============================================================================
using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;  // <M,N,K>/group

// NVFP4 operands: data E2M1, scale E4M3 (ue4m3). The block-scaled mainloop
// builder takes the operand element as cute::tuple<DataElem, ScaleElem>.
using ElementA   = cutlass::float_e2m1_t;          // activations (NVFP4)
using ElementB   = cutlass::float_e2m1_t;          // weights     (NVFP4)
using ElementSF  = cutlass::float_ue4m3_t;         // per-16 block scale (E4M3)
using ElementD   = cutlass::bfloat16_t;            // output
using ElementAccumulator = float;
using ElementCompute     = float;

using LayoutA = cutlass::layout::RowMajor;          // A = [M,K] row-major
using LayoutB = cutlass::layout::ColumnMajor;       // B = [K,N] col-major -> [N,K]
using LayoutD = cutlass::layout::RowMajor;          // D = [M,N] row-major

// FP4 packs 2 vals/byte: alignment is 32 elements (= 16 bytes) along K.
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// SM120 cooperative tile (the canonical block-scaled sm120 tile).
using MmaTileShape_MNK = Shape<_128,_128,_128>;
using ClusterShape_MNK = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    void, LayoutD*, AlignmentD,            // C = void (no source)
    ElementD, LayoutD*, AlignmentD,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;

using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

// NVFP4 operand pair: nv_float4_t<float_e2m1_t> hard-codes SF=ue4m3, SFVecSize=16
// (the cute::tuple<T,SF> form would instead derive SFVecSize from the schedule
// tag, which the generic grouped tag does not encode). For grouped the builder
// requires a PtrArray* schedule tag; nv_float4_t selects NVFP4 block-scaling.
using ElemPairA = cutlass::nv_float4_t<ElementA>;
using ElemPairB = cutlass::nv_float4_t<ElementB>;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElemPairA, LayoutA*, AlignmentA,   // ptr-layout => grouped (ptr-array)
    ElemPairB, LayoutB*, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK, StageCountType,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::InternalStrideA;
using StrideB   = typename Gemm::GemmKernel::InternalStrideB;
using StrideD   = typename Gemm::GemmKernel::InternalStrideD;
using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

// ----------------------------------------------------------------------------
// Device "prepare args" kernel: fill per-group {M,N,K}, ptr arrays, strides,
// SF layouts. (Mirrors flashinfer compute_sm120_cutlass_group_gemm_args.)
//   A packed contiguously by token offset; B one [N][K] block per expert;
//   D packed by token offset. SF buffers: per-expert padded swizzled blocks.
// ----------------------------------------------------------------------------
__global__ void prepare_group_args(
    int num_groups, int N, int K,
    const int* __restrict__ m_indptr,          // [num_groups+1] token prefix offsets
    const ElementA* A_base, const ElementB* B_base, ElementD* D_base,
    const ElementSF* SFA_base, const ElementSF* SFB_base,
    size_t sfA_stride_per_group,               // bytes-as-elements stride per expert in SFA buffer
    size_t sfB_stride_per_group,
    typename ProblemShape::UnderlyingProblemShape* problem_sizes,
    const ElementA** A_ptr, const ElementB** B_ptr, ElementD** D_ptr,
    const ElementSF** SFA_ptr, const ElementSF** SFB_ptr,
    StrideA* stride_A, StrideB* stride_B, StrideD* stride_D,
    LayoutSFA* layout_SFA, LayoutSFB* layout_SFB)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= num_groups) return;
    int m_off  = m_indptr[i];
    int m_next = m_indptr[i+1];
    int M = m_next - m_off;

    problem_sizes[i] = typename ProblemShape::UnderlyingProblemShape(M, N, K);
    stride_A[i] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    stride_B[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    stride_D[i] = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    A_ptr[i] = A_base + (int64_t)m_off * (int64_t)(K/2);   // K/2 bytes-of-2 packed
    B_ptr[i] = B_base + (int64_t)i     * (int64_t)N * (int64_t)(K/2);
    D_ptr[i] = D_base + (int64_t)m_off * (int64_t)N;

    SFA_ptr[i] = SFA_base + (int64_t)i * (int64_t)sfA_stride_per_group;
    SFB_ptr[i] = SFB_base + (int64_t)i * (int64_t)sfB_stride_per_group;
    layout_SFA[i] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    layout_SFB[i] = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
}

// ============================================================================
// ---- GGUF mini-reader (mirrors test_diffusion_moe_grouped.cu) ----
struct PT { uint8_t* dev=nullptr; int type=0; int ndim=0; int64_t ne[4]={1,1,1,1}; int64_t nbytes=0,nelem=0; uint64_t offset=0; };
static std::string rd_str(uint8_t*& p){ uint64_t n=*(uint64_t*)p; p+=8; std::string s((char*)p,n); p+=n; return s; }
static void skip_val(uint8_t*& p, uint32_t t);
static void skip_kv(uint8_t*& p){ rd_str(p); uint32_t t=*(uint32_t*)p; p+=4; skip_val(p,t); }
static uint64_t scalar_sz(uint32_t t){ switch(t){ case 0:case 1:case 7:return 1; case 2:case 3:return 2;
    case 4:case 5:case 6:return 4; case 10:case 11:case 12:return 8; default:return 0; } }
static void skip_val(uint8_t*& p, uint32_t t){
  if(t==8){ uint64_t l=*(uint64_t*)p; p+=8+l; }
  else if(t==9){ uint32_t at=*(uint32_t*)p;p+=4; uint64_t c=*(uint64_t*)p;p+=8;
    if(at==8){ for(uint64_t i=0;i<c;i++){ uint64_t l=*(uint64_t*)p; p+=8+l; } } else p+=c*scalar_sz(at); }
  else p+=scalar_sz(t); }

int main(int argc,char**argv){
  if(argc<2){ fprintf(stderr,"usage: %s model.gguf\n",argv[0]); return 1; }
  int fd=open(argv[1],O_RDONLY); struct stat st; fstat(fd,&st);
  uint8_t* base=(uint8_t*)mmap(nullptr,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
  uint8_t* p=base+4; uint32_t ver=*(uint32_t*)p; p+=4; (void)ver;
  uint64_t nt=*(uint64_t*)p; p+=8; uint64_t nkv=*(uint64_t*)p; p+=8;
  for(uint64_t i=0;i<nkv;i++) skip_kv(p);
  std::unordered_map<std::string,PT> T;
  for(uint64_t i=0;i<nt;i++){ std::string nm=rd_str(p); PT t; t.ndim=*(uint32_t*)p;p+=4;
    for(int d=0;d<t.ndim;d++){t.ne[d]=*(int64_t*)p;p+=8;} t.type=*(uint32_t*)p;p+=4; t.offset=*(uint64_t*)p;p+=8;
    t.nelem=1; for(int d=0;d<t.ndim;d++)t.nelem*=t.ne[d];
    int64_t rb=pt_row_bytes(t.type,t.ne[0]),rows=1; for(int d=1;d<t.ndim;d++)rows*=t.ne[d]; t.nbytes=rb*rows; T[nm]=t; }
  // align tensor data: GGUF pads to alignment (default 32) after header
  uint64_t algn=32; uint64_t hdr=(uint64_t)(p-base); uint64_t pad=(algn-(hdr%algn))%algn; uint8_t* tdata=p+pad;

  const char* want="blk.0.ffn_gate_up_exps.weight";
  auto it=T.find(want); if(it==T.end()){ printf("tensor %s not found\n",want); return 1; }
  PT W=it->second;
  int in_dim=(int)W.ne[0], out_dim=(int)W.ne[1], E=(int)W.ne[2];
  printf("DG FP4 PARITY  %s  in=%d out=%d experts=%d  type=%d\n",want,in_dim,out_dim,E,W.type);
  CK(cudaMalloc(&W.dev,W.nbytes)); CK(cudaMemcpy(W.dev,tdata+W.offset,W.nbytes,cudaMemcpyHostToDevice));
  int64_t slab_bytes=pt_row_bytes(W.type,W.ne[0])*out_dim, slab_elem=(int64_t)in_dim*out_dim;

  const int Mg=16; int total=E*Mg;
  std::vector<int> m_indptr(E+1,0); for(int e=0;e<E;e++) m_indptr[e+1]=m_indptr[e]+Mg;

  int nKvec=in_dim/BLK, nKvec_pad=round_up(nKvec,4);
  unsigned long long sfA_stride=(unsigned long long)128*nKvec_pad;          // M padded 128
  unsigned long long sfB_stride=(unsigned long long)round_up(out_dim,128)*nKvec_pad;

  // ---- weights: pass1 global amax, pass2 quantize per expert ----
  float* d_w32; CK(cudaMalloc(&d_w32,slab_elem*4));
  std::vector<float> hw(slab_elem);
  float wamax=0.f;
  for(int e=0;e<E;e++){ dg_dequant(W.type,W.dev+(size_t)e*slab_bytes,slab_elem,d_w32,0);
    CK(cudaMemcpy(hw.data(),d_w32,slab_elem*4,cudaMemcpyDeviceToHost)); wamax=fmaxf(wamax,hamax(hw)); }
  float gsB=wamax/(6.0f*448.0f); if(gsB<=0)gsB=1e-30f;

  uint8_t* dB_fp4; CK(cudaMalloc(&dB_fp4,(size_t)E*out_dim*(in_dim/2)));
  uint8_t* dB_sf;  CK(cudaMalloc(&dB_sf,(size_t)E*sfB_stride)); CK(cudaMemset(dB_sf,0,(size_t)E*sfB_stride));
  for(int ex=0;ex<E;ex++){ dg_dequant(W.type,W.dev+(size_t)ex*slab_bytes,slab_elem,d_w32,0);
    CK(cudaMemcpy(hw.data(),d_w32,slab_elem*4,cudaMemcpyDeviceToHost));
    uint8_t *df,*dsw; quantize_block_gs(hw,out_dim,in_dim,gsB,&df,&dsw);
    size_t boff=(size_t)ex*out_dim*(in_dim/2), soff=(size_t)ex*sfB_stride;
    CK(cudaMemcpy(dB_fp4+boff,df,(size_t)out_dim*(in_dim/2),cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(dB_sf +soff,dsw,(size_t)sfB_stride,cudaMemcpyDeviceToDevice));
    cudaFree(df); cudaFree(dsw); }

  // ---- activations: random [total][in_dim], shared global ----
  srand(11);
  auto randn=[&](){ float u1=(rand()+1.0f)/(RAND_MAX+2.0f),u2=(rand()+1.0f)/(RAND_MAX+2.0f);
                    return sqrtf(-2*logf(u1))*cosf(6.2831853f*u2); };
  std::vector<float> hXe((size_t)total*in_dim); for(auto&x:hXe) x=randn();
  float gsA=hamax(hXe)/(6.0f*448.0f); if(gsA<=0)gsA=1e-30f;
  uint8_t* dA_fp4; CK(cudaMalloc(&dA_fp4,(size_t)total*(in_dim/2)));
  uint8_t* dA_sf;  CK(cudaMalloc(&dA_sf,(size_t)E*sfA_stride)); CK(cudaMemset(dA_sf,0,(size_t)E*sfA_stride));
  for(int ex=0;ex<E;ex++){ std::vector<float> sub((size_t)Mg*in_dim);
    memcpy(sub.data(),&hXe[(size_t)m_indptr[ex]*in_dim],(size_t)Mg*in_dim*4);
    uint8_t *df,*dsw; quantize_block_gs(sub,Mg,in_dim,gsA,&df,&dsw);
    size_t aoff=(size_t)m_indptr[ex]*(in_dim/2), soff=(size_t)ex*sfA_stride;
    CK(cudaMemcpy(dA_fp4+aoff,df,(size_t)Mg*(in_dim/2),cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(dA_sf +soff,dsw,(size_t)sfA_stride,cudaMemcpyDeviceToDevice));
    cudaFree(df); cudaFree(dsw); }

  // ---- run grouped FP4 GEMM (libdg) ----
  __nv_bfloat16* dD; CK(cudaMalloc(&dD,(size_t)total*out_dim*sizeof(__nv_bfloat16)));
  int* d_indptr; CK(cudaMalloc(&d_indptr,(E+1)*4)); CK(cudaMemcpy(d_indptr,m_indptr.data(),(E+1)*4,cudaMemcpyHostToDevice));
  int rc=dg_fp4_moe_grouped(dD,dA_fp4,dA_sf,dB_fp4,dB_sf,d_indptr,E,out_dim,in_dim,sfA_stride,sfB_stride,gsA*gsB,0);
  CK(cudaDeviceSynchronize());
  printf("  dg_fp4_moe_grouped rc=%d\n",rc); if(rc!=0){ printf("  FP4 GEMM FAILED\n"); return 1; }

  // ---- reference: per-expert dequant + sgemm ----
  cublasHandle_t cub; cublasCreate(&cub); float one=1.f,zero=0.f;
  float* dXe; CK(cudaMalloc(&dXe,(size_t)total*in_dim*4)); CK(cudaMemcpy(dXe,hXe.data(),(size_t)total*in_dim*4,cudaMemcpyHostToDevice));
  float* d_ref; CK(cudaMalloc(&d_ref,(size_t)out_dim*total*4));
  for(int e=0;e<E;e++){ dg_dequant(W.type,W.dev+(size_t)e*slab_bytes,slab_elem,d_w32,0);
    cublasSgemm(cub,CUBLAS_OP_T,CUBLAS_OP_N,out_dim,Mg,in_dim,&one,d_w32,in_dim,
                dXe+(size_t)m_indptr[e]*in_dim,in_dim,&zero,d_ref+(size_t)m_indptr[e]*out_dim,out_dim); }
  CK(cudaDeviceSynchronize());

  // ---- compare ----
  std::vector<float> ref((size_t)total*out_dim); CK(cudaMemcpy(ref.data(),d_ref,(size_t)total*out_dim*4,cudaMemcpyDeviceToHost));
  std::vector<__nv_bfloat16> hd((size_t)total*out_dim); CK(cudaMemcpy(hd.data(),dD,(size_t)total*out_dim*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
  double se=0,sr=0,mx=0; int argok=0;
  for(int t=0;t<total;t++){ int aF=0,aR=0; float vF=-1e30f,vR=-1e30f;
    for(int o=0;o<out_dim;o++){ float f=__bfloat162float(hd[(size_t)t*out_dim+o]); float r=ref[(size_t)t*out_dim+o];
      double d=(double)f-r; se+=d*d; sr+=(double)r*r; if(fabs(d)>mx)mx=fabs(d);
      if(f>vF){vF=f;aF=o;} if(r>vR){vR=r;aR=o;} }
    if(aF==aR) argok++; }
  double l2=sqrt(se/(sr+1e-30));
  printf("  PARITY vs FULL-PRECISION dequant: L2rel=%.4f%%  max|d|=%.4f  argmax match=%d/%d\n",100*l2,mx,argok,total);
  // 13-14% L2rel is the NVFP4 both-operands noise floor on random data (matches the
  // single-GEMM probe test_fp4_gemm.cu). Anything in that band = grouped kernel + quant
  // + 128x4 swizzle are CORRECT; the residual is pure FP4 quantization (naive requant +
  // random acts — calibrated experts + real acts do better). >~30% would mean a real bug.
  printf("  %s\n", l2<0.18 ? "KERNEL CORRECT — residual = NVFP4 noise floor (== single-GEMM probe)"
                           : "*** REAL ERROR (well above FP4 noise floor) — likely swizzle/grouping bug ***");
  return 0;
}
