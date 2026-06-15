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
#include <cmath>
#include <vector>
#include <string>

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
int main(int argc, char** argv){
    bool do_bench = false;
    for (int i=1;i<argc;i++){ if (std::string(argv[i])=="bench") do_bench=true; }

    // DiffusionGemma gate_up expert shape.
    const int K = 2816;          // in  (contraction)
    const int N = 1408;          // out
    const int num_groups = 128;  // experts
    const int tok_per_expert = 16;

    printf("DG grouped NVFP4 microbench  K(in)=%d N(out)=%d  experts=%d  ~tok/expert=%d\n",
           K, N, num_groups, tok_per_expert);
    printf("  CUTLASS sm120 grouped block-scaled NVFP4 (E2M1 + per-16 E4M3 SF, 128x4 swizzle)\n");
    printf("  mode: %s\n", do_bench ? "BENCH (timed launch)" : "SETUP-ONLY (no timed launch)");

    srand(7);
    auto randn=[&](){ float u1=(rand()+1.0f)/(RAND_MAX+2.0f),u2=(rand()+1.0f)/(RAND_MAX+2.0f);
                      return sqrtf(-2*logf(u1))*cosf(6.2831853f*u2); };

    // ragged token counts ~ tok_per_expert (here: fixed for a clean bench)
    std::vector<int> m_per(num_groups, tok_per_expert);
    std::vector<int> m_indptr(num_groups+1, 0);
    for (int i=0;i<num_groups;i++) m_indptr[i+1] = m_indptr[i] + m_per[i];
    int total_M = m_indptr[num_groups];
    printf("  total tokens (sum M_g) = %d\n", total_M);

    // --- Host data: activations A[total_M][K] ~ N(0,1), weights per expert [N][K] ~ N(0,0.05)
    std::vector<float> hA((size_t)total_M*K), hB((size_t)num_groups*N*K);
    for (auto& x:hA) x = randn();
    for (auto& x:hB) x = 0.05f*randn();

    // --- Quantize A (one big [total_M][K]) and B (treated as [num_groups*N][K]) ---
    uint8_t *dA_fp4=nullptr,*dA_sw=nullptr; size_t a_sw=0;
    float gsA = quantize_block(hA, total_M, K, &dA_fp4, &dA_sw, &a_sw);

    uint8_t *dB_fp4=nullptr,*dB_sw=nullptr; size_t b_sw=0;
    float gsB = quantize_block(hB, num_groups*N, K, &dB_fp4, &dB_sw, &b_sw);
    printf("  gsA=%.3e gsB=%.3e  SFA buf=%.1f KiB  SFB buf=%.1f KiB\n",
           gsA, gsB, a_sw/1024.0, b_sw/1024.0);

    // Per-group SF strides (in SF elements). A is one packed [total_M][K] block
    // so per-expert A SF is NOT a simple stride — for a correct grouped layout we
    // quantize A per-expert instead. To keep this microbench's SF layout exact &
    // per-group-contiguous, re-quantize A per expert into a padded layout:
    //   each expert's A-SF block = pad(M_g,128) x pad(K/16,4) bytes.
    // (We compute strides for the prepare kernel below.)
    int nKvec      = K/BLK;
    int nKvec_pad  = round_up(nKvec,4);
    int Mpad_g     = round_up(tok_per_expert,128);     // 128 (since M_g<128)
    int Npad       = round_up(N,128);
    size_t sfA_stride = (size_t)Mpad_g*(size_t)nKvec_pad;   // per-expert A SF elems
    size_t sfB_stride = (size_t)Npad  *(size_t)nKvec_pad;   // per-expert B SF elems

    // Re-quantize A PER EXPERT into a per-group-contiguous swizzled SF buffer so
    // SFA_ptr[i] = SFA_base + i*sfA_stride is valid. (B is already per-expert.)
    uint8_t* dA_sf_grouped; CK(cudaMalloc(&dA_sf_grouped, (size_t)num_groups*sfA_stride));
    CK(cudaMemset(dA_sf_grouped,0,(size_t)num_groups*sfA_stride));
    // B SF likewise needs per-expert padded blocks of sfB_stride; our quantize_block
    // produced one [num_groups*N] swizzled buffer with outer padded to 128 across the
    // WHOLE num_groups*N — not per-expert. Re-quantize B per expert for exact strides.
    uint8_t* dB_sf_grouped; CK(cudaMalloc(&dB_sf_grouped, (size_t)num_groups*sfB_stride));
    CK(cudaMemset(dB_sf_grouped,0,(size_t)num_groups*sfB_stride));

    // Per-expert (re)quantization to fill the grouped, per-expert-padded SF buffers
    // and the packed E2M1 data (data packing is dense/contiguous, SF is padded).
    {
        // A: per expert M_g rows
        for (int g=0; g<num_groups; g++){
            int Mg = m_per[g], moff = m_indptr[g];
            std::vector<float> sub((size_t)Mg*K);
            for (int r=0;r<Mg;r++) for(int c=0;c<K;c++) sub[(size_t)r*K+c]=hA[(size_t)(moff+r)*K+c];
            uint8_t *df=nullptr,*dsw=nullptr; size_t bytes=0;
            quantize_block(sub, Mg, K, &df, &dsw, &bytes);
            // copy this expert's swizzled SF (which was sized pad(Mg,128)xpad(K/16,4)
            // = sfA_stride) into the grouped buffer slot.
            CK(cudaMemcpy(dA_sf_grouped + (size_t)g*sfA_stride, dsw, sfA_stride,
                          cudaMemcpyDeviceToDevice));
            // copy packed data into the dense A_fp4 region for these rows
            CK(cudaMemcpy(dA_fp4 + (size_t)moff*(K/2), df, (size_t)Mg*(K/2),
                          cudaMemcpyDeviceToDevice));
            cudaFree(df); cudaFree(dsw);
        }
        // B: per expert N rows
        for (int g=0; g<num_groups; g++){
            std::vector<float> sub((size_t)N*K);
            for (size_t e=0;e<(size_t)N*K;e++) sub[e]=hB[(size_t)g*N*K+e];
            uint8_t *df=nullptr,*dsw=nullptr; size_t bytes=0;
            quantize_block(sub, N, K, &df, &dsw, &bytes);
            CK(cudaMemcpy(dB_sf_grouped + (size_t)g*sfB_stride, dsw, sfB_stride,
                          cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(dB_fp4 + (size_t)g*N*(K/2), df, (size_t)N*(K/2),
                          cudaMemcpyDeviceToDevice));
            cudaFree(df); cudaFree(dsw);
        }
    }
    cudaFree(dA_sw); cudaFree(dB_sw);   // discard the non-grouped swizzle buffers

    // --- Output ---
    ElementD* dD; CK(cudaMalloc(&dD,(size_t)total_M*N*sizeof(ElementD)));

    // --- Device m_indptr ---
    int* d_indptr; CK(cudaMalloc(&d_indptr,(num_groups+1)*sizeof(int)));
    CK(cudaMemcpy(d_indptr,m_indptr.data(),(num_groups+1)*sizeof(int),cudaMemcpyHostToDevice));

    // --- Per-group argument arrays (device) ---
    typename ProblemShape::UnderlyingProblemShape* d_problem;
    const ElementA** d_Aptr; const ElementB** d_Bptr; ElementD** d_Dptr;
    const ElementSF** d_SFAptr; const ElementSF** d_SFBptr;
    StrideA* d_strA; StrideB* d_strB; StrideD* d_strD;
    LayoutSFA* d_layA; LayoutSFB* d_layB;
    CK(cudaMalloc(&d_problem, num_groups*sizeof(*d_problem)));
    CK(cudaMalloc(&d_Aptr,    num_groups*sizeof(*d_Aptr)));
    CK(cudaMalloc(&d_Bptr,    num_groups*sizeof(*d_Bptr)));
    CK(cudaMalloc(&d_Dptr,    num_groups*sizeof(*d_Dptr)));
    CK(cudaMalloc(&d_SFAptr,  num_groups*sizeof(*d_SFAptr)));
    CK(cudaMalloc(&d_SFBptr,  num_groups*sizeof(*d_SFBptr)));
    CK(cudaMalloc(&d_strA,    num_groups*sizeof(*d_strA)));
    CK(cudaMalloc(&d_strB,    num_groups*sizeof(*d_strB)));
    CK(cudaMalloc(&d_strD,    num_groups*sizeof(*d_strD)));
    CK(cudaMalloc(&d_layA,    num_groups*sizeof(*d_layA)));
    CK(cudaMalloc(&d_layB,    num_groups*sizeof(*d_layB)));

    {
        int t = num_groups < 256 ? num_groups : 256;
        int b = (num_groups + t - 1)/t;
        prepare_group_args<<<b,t>>>(
            num_groups, N, K, d_indptr,
            reinterpret_cast<const ElementA*>(dA_fp4),
            reinterpret_cast<const ElementB*>(dB_fp4),
            dD,
            reinterpret_cast<const ElementSF*>(dA_sf_grouped),
            reinterpret_cast<const ElementSF*>(dB_sf_grouped),
            sfA_stride, sfB_stride,
            d_problem, d_Aptr, d_Bptr, d_Dptr, d_SFAptr, d_SFBptr,
            d_strA, d_strB, d_strD, d_layA, d_layB);
        CK(cudaDeviceSynchronize());
    }
    printf("  prepared %d per-group problems / ptr / stride / SF-layout arrays\n", num_groups);

    // --- Build CUTLASS grouped arguments ---
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_groups, d_problem, /*host problem sizes*/ nullptr},
        {   // mainloop
            d_Aptr, d_strA,
            d_Bptr, d_strB,
            d_SFAptr, d_layA,
            d_SFBptr, d_layB,
        },
        {   // epilogue
            {},          // thread (alpha/beta below)
            nullptr, nullptr,   // C ptr / stride
            d_Dptr, d_strD,
        },
        hw_info
    };
    arguments.epilogue.thread.alpha = gsA * gsB;   // fold per-tensor global scales
    arguments.epilogue.thread.beta  = 0.0f;

    Gemm gemm;
    size_t ws_size = Gemm::get_workspace_size(arguments);
    void* d_ws=nullptr; if (ws_size) CK(cudaMalloc(&d_ws, ws_size));
    printf("  CUTLASS workspace = %zu bytes\n", ws_size);

    cutlass::Status can = gemm.can_implement(arguments);
    printf("  can_implement: %s\n", can==cutlass::Status::kSuccess ? "YES" : "NO");
    if (can != cutlass::Status::kSuccess){
        printf("  (kernel not implementable for this shape on this build) status=%d\n",(int)can);
    }

    if (!do_bench){
        printf("SETUP OK — skipping timed launch (pass 'bench' to run on an idle GPU).\n");
        return 0;
    }

    // ----- timed path (GUARDED) -----
    CUTLASS_CK(gemm.can_implement(arguments));
    CUTLASS_CK(gemm.initialize(arguments, d_ws));

    // warmup
    for (int i=0;i<5;i++) CUTLASS_CK(gemm.run());
    CK(cudaDeviceSynchronize());

    cudaEvent_t ev0,ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
    const int R = 200;
    CK(cudaEventRecord(ev0));
    for (int i=0;i<R;i++) CUTLASS_CK(gemm.run());
    CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
    float ms=0; CK(cudaEventElapsedTime(&ms,ev0,ev1));
    double per = ms/R;
    double flops = 2.0*(double)total_M*N*K;   // sum over groups = total_M*N*K
    printf("GROUPED NVFP4 GEMM: %.4f ms/iter   %.2f TFLOP/s (eff)\n",
           per, flops/(per/1e3)/1e12);
    return 0;
}
