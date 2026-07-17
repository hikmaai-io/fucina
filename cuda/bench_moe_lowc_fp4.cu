// ABOUTME: Low-concurrency MoE decode microbench for the CUTLASS sm120 grouped NVFP4 expert GEMM
// ABOUTME: (dg_fp4_moe_grouped's kernel), at Qwen3.5-35B-A3B shapes; sweeps active experts = B*topk.
//
// The default FP8-MoE serving path (fp4_mode) routes every decode expert projection through a
// CUTLASS grouped block-scaled NVFP4 GEMM. At low concurrency (B=1..8) each active expert holds
// ~1 token, so this bench answers the L-moe-lowc attribution question directly:
//   does the grouped GEMM cost scale ~linearly with the number of ACTIVE experts (=> each expert's
//   weights read once, bandwidth floor, nothing to fix) or is there large fixed per-group overhead
//   / MMA-tile waste at M_g=1 (=> a fixable low-B inefficiency)?
//
// It times the two real projections (gate|up fused: K=H=2048 N=2*EFFN=1024; down: K=EFFN=512
// N=H=2048) over num_groups = {8,16,32,64,128,256} active experts at tok_per_expert tokens each
// (B = num_groups/topk for topk=8). Prints ms/iter, weight bytes read, effective GB/s, TFLOP/s.
//
// Structure/CUTLASS recipe mirror the validated cuda/test_dg_fp4_grouped.cu (same sm120 grouped
// block-scaled NVFP4 type stack, same per-16 E4M3 128x4-swizzled SF). Accuracy is not the point;
// timing scaling vs active-expert count is.
//
// build: make bench-moe-lowc-fp4 ; run under flock /tmp/fucina_gpu.lock.
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

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/float_subbyte.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/util/packed_stride.hpp>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
#define CUTLASS_CK(status) do{ cutlass::Status s=(status); if(s!=cutlass::Status::kSuccess){ \
  printf("CUTLASS err %s @%d: %d\n",#status,__LINE__,(int)s); exit(1);} }while(0)

static const int BLK = 16;

__global__ void quant_nvfp4(const float* __restrict__ X, uint8_t* __restrict__ fp4,
                            uint8_t* __restrict__ bscale, int rows, int k, float gs){
    int row = blockIdx.y;
    int blk = blockIdx.x*blockDim.x + threadIdx.x;
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
    int outer_pad = round_up(rows,128);
    int nblk_pad  = round_up(nblk,4);
    int nKtiles   = nblk_pad/4;
    size_t sw_bytes = (size_t)outer_pad*(size_t)nblk_pad;
    uint8_t* dsw; CK(cudaMalloc(&dsw, sw_bytes)); CK(cudaMemset(dsw,0,sw_bytes));
    { dim3 b(32,8), g((nblk+31)/32,(rows+7)/8);
      swizzle_scales_128x4<<<g,b>>>(dlin,dsw,rows,nblk,nKtiles); }
    CK(cudaDeviceSynchronize());
    cudaFree(dX); cudaFree(dlin);
    *dfp4_out = dfp4; *dsw_out = dsw; if(sw_bytes_out) *sw_bytes_out = sw_bytes;
    return gs;
}

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

__global__ void prepare_group_args(
    int num_groups, int N, int K,
    const int* __restrict__ m_indptr,
    const ElementA* A_base, const ElementB* B_base, ElementD* D_base,
    const ElementSF* SFA_base, const ElementSF* SFB_base,
    size_t sfA_stride_per_group, size_t sfB_stride_per_group,
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
    A_ptr[i] = A_base + (int64_t)m_off * (int64_t)(K/2);
    B_ptr[i] = B_base + (int64_t)i     * (int64_t)N * (int64_t)(K/2);
    D_ptr[i] = D_base + (int64_t)m_off * (int64_t)N;
    SFA_ptr[i] = SFA_base + (int64_t)i * (int64_t)sfA_stride_per_group;
    SFB_ptr[i] = SFB_base + (int64_t)i * (int64_t)sfB_stride_per_group;
    layout_SFA[i] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    layout_SFB[i] = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
}

// Time the grouped NVFP4 GEMM for `num_groups` active experts, `tok` tokens each, shape N x K.
static void run_cfg(const char* proj, int num_groups, int tok, int K, int N, int R){
    srand(7);
    auto randn=[&](){ float u1=(rand()+1.0f)/(RAND_MAX+2.0f),u2=(rand()+1.0f)/(RAND_MAX+2.0f);
                      return sqrtf(-2*logf(u1))*cosf(6.2831853f*u2); };
    std::vector<int> m_per(num_groups, tok);
    std::vector<int> m_indptr(num_groups+1, 0);
    for (int i=0;i<num_groups;i++) m_indptr[i+1] = m_indptr[i] + m_per[i];
    int total_M = m_indptr[num_groups];

    std::vector<float> hA((size_t)total_M*K), hB((size_t)num_groups*N*K);
    for (auto& x:hA) x = randn();
    for (auto& x:hB) x = 0.05f*randn();

    uint8_t *dA_fp4=nullptr,*dA_sw=nullptr; size_t a_sw=0;
    float gsA = quantize_block(hA, total_M, K, &dA_fp4, &dA_sw, &a_sw);
    uint8_t *dB_fp4=nullptr,*dB_sw=nullptr; size_t b_sw=0;
    float gsB = quantize_block(hB, num_groups*N, K, &dB_fp4, &dB_sw, &b_sw);

    int nKvec      = K/BLK;
    int nKvec_pad  = round_up(nKvec,4);
    int Mpad_g     = round_up(tok,128);
    int Npad       = round_up(N,128);
    size_t sfA_stride = (size_t)Mpad_g*(size_t)nKvec_pad;
    size_t sfB_stride = (size_t)Npad  *(size_t)nKvec_pad;

    uint8_t* dA_sf_grouped; CK(cudaMalloc(&dA_sf_grouped, (size_t)num_groups*sfA_stride));
    CK(cudaMemset(dA_sf_grouped,0,(size_t)num_groups*sfA_stride));
    uint8_t* dB_sf_grouped; CK(cudaMalloc(&dB_sf_grouped, (size_t)num_groups*sfB_stride));
    CK(cudaMemset(dB_sf_grouped,0,(size_t)num_groups*sfB_stride));
    {
        for (int g=0; g<num_groups; g++){
            int Mg = m_per[g], moff = m_indptr[g];
            std::vector<float> sub((size_t)Mg*K);
            for (int r=0;r<Mg;r++) for(int c=0;c<K;c++) sub[(size_t)r*K+c]=hA[(size_t)(moff+r)*K+c];
            uint8_t *df=nullptr,*dsw=nullptr; size_t bytes=0;
            quantize_block(sub, Mg, K, &df, &dsw, &bytes);
            CK(cudaMemcpy(dA_sf_grouped + (size_t)g*sfA_stride, dsw, sfA_stride, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(dA_fp4 + (size_t)moff*(K/2), df, (size_t)Mg*(K/2), cudaMemcpyDeviceToDevice));
            cudaFree(df); cudaFree(dsw);
        }
        for (int g=0; g<num_groups; g++){
            std::vector<float> sub((size_t)N*K);
            for (size_t e=0;e<(size_t)N*K;e++) sub[e]=hB[(size_t)g*N*K+e];
            uint8_t *df=nullptr,*dsw=nullptr; size_t bytes=0;
            quantize_block(sub, N, K, &df, &dsw, &bytes);
            CK(cudaMemcpy(dB_sf_grouped + (size_t)g*sfB_stride, dsw, sfB_stride, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(dB_fp4 + (size_t)g*N*(K/2), df, (size_t)N*(K/2), cudaMemcpyDeviceToDevice));
            cudaFree(df); cudaFree(dsw);
        }
    }
    cudaFree(dA_sw); cudaFree(dB_sw);

    ElementD* dD; CK(cudaMalloc(&dD,(size_t)total_M*N*sizeof(ElementD)));
    int* d_indptr; CK(cudaMalloc(&d_indptr,(num_groups+1)*sizeof(int)));
    CK(cudaMemcpy(d_indptr,m_indptr.data(),(num_groups+1)*sizeof(int),cudaMemcpyHostToDevice));

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
            reinterpret_cast<const ElementB*>(dB_fp4), dD,
            reinterpret_cast<const ElementSF*>(dA_sf_grouped),
            reinterpret_cast<const ElementSF*>(dB_sf_grouped),
            sfA_stride, sfB_stride,
            d_problem, d_Aptr, d_Bptr, d_Dptr, d_SFAptr, d_SFBptr,
            d_strA, d_strB, d_strD, d_layA, d_layB);
        CK(cudaDeviceSynchronize());
    }

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_groups, d_problem, nullptr},
        { d_Aptr, d_strA, d_Bptr, d_strB, d_SFAptr, d_layA, d_SFBptr, d_layB },
        { {}, nullptr, nullptr, d_Dptr, d_strD },
        hw_info
    };
    arguments.epilogue.thread.alpha = gsA * gsB;
    arguments.epilogue.thread.beta  = 0.0f;
    Gemm gemm;
    size_t ws_size = Gemm::get_workspace_size(arguments);
    void* d_ws=nullptr; if (ws_size) CK(cudaMalloc(&d_ws, ws_size));
    if (gemm.can_implement(arguments) != cutlass::Status::kSuccess){
        printf("  %-4s groups=%3d tok=%d  can_implement=NO\n", proj, num_groups, tok); return;
    }
    CUTLASS_CK(gemm.initialize(arguments, d_ws));
    for (int i=0;i<10;i++) CUTLASS_CK(gemm.run());
    CK(cudaDeviceSynchronize());
    cudaEvent_t ev0,ev1; CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1));
    CK(cudaEventRecord(ev0));
    for (int i=0;i<R;i++) CUTLASS_CK(gemm.run());
    CK(cudaEventRecord(ev1)); CK(cudaEventSynchronize(ev1));
    float ms=0; CK(cudaEventElapsedTime(&ms,ev0,ev1));
    double per = ms/R;
    // weight bytes = num_groups * N * K/2 (E2M1) + SF bytes (num_groups*sfB_stride)
    double wbytes = (double)num_groups*N*(K/2) + (double)num_groups*sfB_stride;
    double flops = 2.0*(double)total_M*N*K;
    printf("  %-4s groups=%3d tok=%d  M=%4d  %.4f ms  wt=%.2f MB  %.1f GB/s  %.2f TFLOP/s\n",
           proj, num_groups, tok, total_M, per, wbytes/1e6,
           wbytes/(per/1e3)/1e9, flops/(per/1e3)/1e12);
    cudaFree(dA_fp4); cudaFree(dB_fp4); cudaFree(dA_sf_grouped); cudaFree(dB_sf_grouped);
    cudaFree(dD); cudaFree(d_indptr); cudaFree(d_problem);
    cudaFree(d_Aptr); cudaFree(d_Bptr); cudaFree(d_Dptr); cudaFree(d_SFAptr); cudaFree(d_SFBptr);
    cudaFree(d_strA); cudaFree(d_strB); cudaFree(d_strD); cudaFree(d_layA); cudaFree(d_layB);
    if (d_ws) cudaFree(d_ws);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
}

int main(int argc, char** argv){
    // Qwen3.5-35B-A3B-FP8 MoE: H=2048, EFFN=512, topk=8, E=256.
    const int H = 2048, EFFN = 512;
    const int R = (argc>1)? atoi(argv[1]) : 300;
    int tok = (argc>2)? atoi(argv[2]) : 1;   // tokens per active expert (decode ~1)
    printf("MoE low-conc NVFP4 grouped-GEMM microbench (Qwen3.5-35B-A3B: H=%d EFFN=%d topk=8 E=256)\n", H, EFFN);
    printf("gate|up proj: K(in)=H=%d N(out)=2*EFFN=%d ;  down proj: K(in)=EFFN=%d N(out)=H=%d\n",
           H, 2*EFFN, EFFN, H);
    printf("num_groups = active experts = B*topk (B = groups/8). tok/expert=%d, R=%d iters.\n\n", tok, R);
    int sweep[] = {8,16,32,64,128,256};
    printf("== gate|up (K=%d N=%d) ==\n", H, 2*EFFN);
    for (int g : sweep) run_cfg("gu", g, tok, H, 2*EFFN, R);
    printf("== down (K=%d N=%d) ==\n", EFFN, H);
    for (int g : sweep) run_cfg("dn", g, tok, EFFN, H, R);
    return 0;
}
