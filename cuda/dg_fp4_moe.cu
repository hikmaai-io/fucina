// Grouped block-scaled NVFP4 expert GEMM for the DiffusionGemma MoE — engine-callable
// wrapper around the CUTLASS 4.x sm120 ptr-array block-scaled path (validated 2.56× over
// our dp4a grouped kernel at 16 tok/expert in cuda/test_dg_fp4_grouped.cu).
//
// Boundary: the engine produces, per step, the activations grouped-by-expert + per-expert
// padded swizzled E4M3 scales (A, A_sf), and once-on-load the per-expert weight slabs
// (B, B_sf) + the per-tensor global scales folded into `alpha`. This module does the
// per-group argument setup + the grouped GEMM. D is bf16 [total_M, N] grouped-by-expert.
//
// Compiled as its own object (CUTLASS needs --expt-relaxed-constexpr etc.) and archived
// into libdg.a; exposes a plain extern "C" API so diffusion_gemma_engine.cu stays CUTLASS-free.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
using ElementA   = cutlass::float_e2m1_t;
using ElementB   = cutlass::float_e2m1_t;
using ElementSF  = cutlass::float_ue4m3_t;
using ElementD   = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute     = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using MmaTileShape_MNK = Shape<_128,_128,_128>;
using ClusterShape_MNK = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    void, LayoutD*, AlignmentD,
    ElementD, LayoutD*, AlignmentD,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
using ElemPairA = cutlass::nv_float4_t<ElementA>;
using ElemPairB = cutlass::nv_float4_t<ElementB>;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElemPairA, LayoutA*, AlignmentA,
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

__global__ void dg_fp4_prepare(
    int num_groups, int N, int K,
    const int* __restrict__ m_indptr, const int* __restrict__ m_count,
    const int* __restrict__ m_coloff, const int* __restrict__ expert_slot,
    const ElementA* A_base, const ElementB* B_base, ElementD* D_base,
    const ElementSF* SFA_base, const ElementSF* SFB_base,
    size_t sfA_stride, size_t sfB_stride,
    typename ProblemShape::UnderlyingProblemShape* problem_sizes,
    const ElementA** A_ptr, const ElementB** B_ptr, ElementD** D_ptr,
    const ElementSF** SFA_ptr, const ElementSF** SFB_ptr,
    StrideA* stride_A, StrideB* stride_B, StrideD* stride_D,
    LayoutSFA* layout_SFA, LayoutSFB* layout_SFB)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= num_groups) return;
    int mapped = expert_slot ? expert_slot[i] : i;
    int m_off = m_count ? m_coloff[i] : m_indptr[i];
    int M = m_count ? (mapped >= 0 ? m_count[i] : 0) : (m_indptr[i+1] - m_off);
    if (mapped < 0) mapped = 0;
    problem_sizes[i] = typename ProblemShape::UnderlyingProblemShape(M, N, K);
    stride_A[i] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    stride_B[i] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    stride_D[i] = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    A_ptr[i] = A_base + (int64_t)m_off * (int64_t)(K/2);
    B_ptr[i] = B_base + (int64_t)mapped * (int64_t)N * (int64_t)(K/2);
    D_ptr[i] = D_base + (int64_t)m_off * (int64_t)N;
    SFA_ptr[i] = SFA_base + (int64_t)i * (int64_t)sfA_stride;
    SFB_ptr[i] = SFB_base + (int64_t)mapped * (int64_t)sfB_stride;
    layout_SFA[i] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    layout_SFB[i] = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
}

// Cached per-group argument scratch (allocated once for the fixed expert count).
namespace {
struct Scratch {
    int cap = 0;
    typename ProblemShape::UnderlyingProblemShape* problem = nullptr;
    const ElementA** Aptr = nullptr; const ElementB** Bptr = nullptr; ElementD** Dptr = nullptr;
    const ElementSF** SFAptr = nullptr; const ElementSF** SFBptr = nullptr;
    StrideA* strA = nullptr; StrideB* strB = nullptr; StrideD* strD = nullptr;
    LayoutSFA* layA = nullptr; LayoutSFB* layB = nullptr;
    void* ws = nullptr; size_t ws_bytes = 0;
} g;
static bool ensure(int ng){
    if (ng <= g.cap) return true;
    auto F=[&](void** p){ if(*p) cudaFree(*p); };
    F((void**)&g.problem);F((void**)&g.Aptr);F((void**)&g.Bptr);F((void**)&g.Dptr);
    F((void**)&g.SFAptr);F((void**)&g.SFBptr);F((void**)&g.strA);F((void**)&g.strB);
    F((void**)&g.strD);F((void**)&g.layA);F((void**)&g.layB);
    int ok=1;
    ok&=cudaMalloc(&g.problem,ng*sizeof(*g.problem))==cudaSuccess;
    ok&=cudaMalloc(&g.Aptr,ng*sizeof(*g.Aptr))==cudaSuccess;
    ok&=cudaMalloc(&g.Bptr,ng*sizeof(*g.Bptr))==cudaSuccess;
    ok&=cudaMalloc(&g.Dptr,ng*sizeof(*g.Dptr))==cudaSuccess;
    ok&=cudaMalloc(&g.SFAptr,ng*sizeof(*g.SFAptr))==cudaSuccess;
    ok&=cudaMalloc(&g.SFBptr,ng*sizeof(*g.SFBptr))==cudaSuccess;
    ok&=cudaMalloc(&g.strA,ng*sizeof(*g.strA))==cudaSuccess;
    ok&=cudaMalloc(&g.strB,ng*sizeof(*g.strB))==cudaSuccess;
    ok&=cudaMalloc(&g.strD,ng*sizeof(*g.strD))==cudaSuccess;
    ok&=cudaMalloc(&g.layA,ng*sizeof(*g.layA))==cudaSuccess;
    ok&=cudaMalloc(&g.layB,ng*sizeof(*g.layB))==cudaSuccess;
    if(!ok) return false;
    g.cap = ng; return true;
}
} // namespace

// Grouped NVFP4 expert GEMM. D[total_M,N] bf16 = (A·Bᵀ)·alpha, grouped by expert via
// m_indptr[num_groups+1] (device, token prefix offsets). A/B packed E2M1; A_sf/B_sf per-expert
// padded swizzled E4M3 (strides sfA_stride/sfB_stride in SF elements). alpha = gsA·gsB.
// Returns 0 on success, <0 on failure (caller falls back to dp4a).
static int dg_fp4_moe_grouped_impl(
    void* D, const void* A, const void* A_sf, const void* B, const void* B_sf,
    const int* m_indptr, const int* m_count, const int* m_coloff, const int* expert_slot,
    int num_groups, int N, int K, unsigned long long sfA_stride, unsigned long long sfB_stride,
    const float* alpha, cudaStream_t stream) {
    if (!ensure(num_groups)) return -1;
    int t = num_groups < 256 ? num_groups : 256, b = (num_groups + t - 1)/t;
    dg_fp4_prepare<<<b,t,0,stream>>>(
        num_groups, N, K, m_indptr, m_count, m_coloff, expert_slot,
        (const ElementA*)A, (const ElementB*)B, (ElementD*)D,
        (const ElementSF*)A_sf, (const ElementSF*)B_sf,
        (size_t)sfA_stride, (size_t)sfB_stride,
        g.problem, g.Aptr, g.Bptr, g.Dptr, g.SFAptr, g.SFBptr,
        g.strA, g.strB, g.strD, g.layA, g.layB);

    cutlass::KernelHardwareInfo hw; hw.device_id = 0;
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_groups, g.problem, nullptr},
        { g.Aptr, g.strA, g.Bptr, g.strB, g.SFAptr, g.layA, g.SFBptr, g.layB },
        { {}, nullptr, nullptr, g.Dptr, g.strD },
        hw
    };
    args.epilogue.thread.alpha_ptr = alpha;   // device-pointer alpha (no D2H sync per call)
    args.epilogue.thread.beta      = 0.0f;

    static Gemm gemm;
    size_t need = Gemm::get_workspace_size(args);
    if (need > g.ws_bytes) { if (g.ws) cudaFree(g.ws);
        if (cudaMalloc(&g.ws, need) != cudaSuccess) { g.ws_bytes=0; return -2; } g.ws_bytes = need; }
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -3;
    if (gemm.initialize(args, g.ws, stream) != cutlass::Status::kSuccess) return -4;
    if (gemm.run(stream) != cutlass::Status::kSuccess) return -5;
    return 0;
}

extern "C" int dg_fp4_moe_grouped(
    void* D, const void* A, const void* A_sf, const void* B, const void* B_sf,
    const int* m_indptr, int num_groups, int N, int K,
    unsigned long long sfA_stride, unsigned long long sfB_stride,
    const float* alpha, cudaStream_t stream) {
    return dg_fp4_moe_grouped_impl(D,A,A_sf,B,B_sf,m_indptr,nullptr,nullptr,nullptr,
        num_groups,N,K,sfA_stride,sfB_stride,alpha,stream);
}

extern "C" int dg_fp4_moe_grouped_mapped(
    void* D, const void* A, const void* A_sf, const void* B, const void* B_sf,
    const int* m_count, const int* m_coloff, const int* expert_slot,
    int num_groups, int N, int K, unsigned long long sfA_stride, unsigned long long sfB_stride,
    const float* alpha, cudaStream_t stream) {
    return dg_fp4_moe_grouped_impl(D,A,A_sf,B,B_sf,nullptr,m_count,m_coloff,expert_slot,
        num_groups,N,K,sfA_stride,sfB_stride,alpha,stream);
}

// Per-expert padded SF strides (SF elements) for a given shape — the engine sizes its
// per-expert swizzled-scale buffers with these. pad(outer,128)·pad(K/16,4).
extern "C" void dg_fp4_sf_strides(int M_max, int N, int K,
        unsigned long long* sfA_stride, unsigned long long* sfB_stride){
    auto ru=[](int x,int m){ return ((x+m-1)/m)*m; };
    int nKvec_pad = ru(K/16, 4);
    *sfA_stride = (unsigned long long)ru(M_max,128) * (unsigned long long)nKvec_pad;
    *sfB_stride = (unsigned long long)ru(N,128)     * (unsigned long long)nKvec_pad;
}
