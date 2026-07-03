// q4k_wmma.cuh — grouped tensor-core GEMM over NATURAL-layout Q4_K expert slabs.
//
// out[tok][out_dim] = X[tok][in] · W_e[out][in]^T for every (expert, token) assignment, with the
// Q4_K weights dequantized to BF16 in SHARED MEMORY tile-by-tile and consumed by wmma 16x16x16
// BF16 MMAs. Replaces (a) the prefill dequant-to-global round-trip (467 ms/pass measured) and
// (b) the dp4a GEMV's scalar math at decode — weight bytes still read exactly once per
// (expert, ≤16-token group), but the MACs ride tensor cores.
//
// Tile: one block = (expert slot, 64-row out-tile); 4 warps each own a 16-row n-subtile; the
// 16-token A tile is staged once per k-step by the whole block (f32→bf16, zero-padded rows).
// Natural ggml block_q4_K layout — same element order as the dequant kernels.
#ifndef FUCINA_Q4K_WMMA_CUH
#define FUCINA_Q4K_WMMA_CUH
#include <mma.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ggml get_scale_min_k4 (duplicated so this header stays standalone).
static __device__ __forceinline__ void q4kw_scale_min(const uint8_t *sc, int j, int *s, int *m) {
    if (j < 4) { *s = sc[j] & 63; *m = sc[j + 4] & 63; }
    else {
        *s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        *m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
    }
}

#define Q4KW_NTILE 64   // out-rows per block (4 warps × one 16x16 wmma n-subtile)
#define Q4KW_MTILE 16   // tokens per group (one wmma m-tile, zero-padded)

__global__ void q4k_wmma_grouped_kernel(
    float *__restrict__ out,                 // [total][out_dim]
    const uint8_t *__restrict__ wbase, int64_t slab_stride,
    const float *__restrict__ x,             // [total][in_dim] f32 (gathered, expert-contiguous)
    const int *__restrict__ coloff, const int *__restrict__ count,
    const int *__restrict__ active, int in_dim, int out_dim)
{
    int e = active ? active[blockIdx.y] : (int)blockIdx.y;
    if (e < 0) return;
    int cnt = count[e];
    if (cnt <= 0) return;
    int off = coloff[e];
    int n0 = blockIdx.x * Q4KW_NTILE;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    const uint8_t *wslab = wbase + (size_t)e * slab_stride;
    int n_super_row = in_dim >> 8;

    __shared__ __nv_bfloat16 sA[Q4KW_MTILE][24];             // A tile (stride 24: bank-friendly)
    __shared__ __nv_bfloat16 sB[4][16][24];                  // per-warp W subtile
    __shared__ float sC[4][16][16];

    for (int g0 = 0; g0 < cnt; g0 += Q4KW_MTILE) {
        int nreal = cnt - g0; if (nreal > Q4KW_MTILE) nreal = Q4KW_MTILE;
        const float *xg = x + (size_t)(off + g0) * in_dim;

        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c;
        nvcuda::wmma::fill_fragment(c, 0.0f);

        for (int k0 = 0; k0 < in_dim; k0 += 16) {
            // stage A: whole block fills the 16x16 token tile (rows >= nreal are zero)
            for (int t = threadIdx.x; t < Q4KW_MTILE * 16; t += blockDim.x) {
                int m = t >> 4, k = t & 15;
                sA[m][k] = (m < nreal) ? __float2bfloat16(xg[(size_t)m * in_dim + k0 + k])
                                       : __float2bfloat16(0.0f);
            }
            // stage B: warp w decodes its 16 weight rows for k [k0, k0+16)
            int rbase = n0 + warp * 16;
            for (int t = lane; t < 16 * 16; t += 32) {
                int r = t >> 4, k = t & 15;
                int row = rbase + r, kk = k0 + k;
                const uint8_t *blk = wslab + ((size_t)row * n_super_row + (kk >> 8)) * 144;
                int r256 = kk & 255, j = r256 >> 5, kin = r256 & 31;
                __half_raw hd; hd.x = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8));
                __half_raw hm; hm.x = (uint16_t)(blk[2] | ((uint16_t)blk[3] << 8));
                float d = __half2float(__half(hd)), dmin = __half2float(__half(hm));
                int sV, mV; q4kw_scale_min(blk + 4, j, &sV, &mV);
                const uint8_t *qbase = blk + 16 + (size_t)(j >> 1) * 32;
                int shift = (j & 1) ? 4 : 0;
                int q = (qbase[kin] >> shift) & 0xF;
                sB[warp][r][k] = __float2bfloat16(d * (float)sV * (float)q - dmin * (float)mV);
            }
            __syncthreads();

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                                   nvcuda::wmma::row_major> a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                                   nvcuda::wmma::col_major> b;
            nvcuda::wmma::load_matrix_sync(a, &sA[0][0], 24);
            nvcuda::wmma::load_matrix_sync(b, &sB[warp][0][0], 24);   // W[n][k] rows = kxn col-major
            nvcuda::wmma::mma_sync(c, a, b, c);
            __syncthreads();
        }

        nvcuda::wmma::store_matrix_sync(&sC[warp][0][0], c, 16, nvcuda::wmma::mem_row_major);
        __syncwarp();
        for (int t = lane; t < nreal * 16; t += 32) {
            int m = t / 16, n = t % 16;
            out[(size_t)(off + g0 + m) * out_dim + n0 + warp * 16 + n] = sC[warp][m][n];
        }
        __syncthreads();
    }
}

static inline void q4k_wmma_grouped_launch(
    float *out, const void *wbase, int64_t slab_stride, const float *x,
    const int *coloff, const int *count, const int *active,
    int n_slot, int n_expert, int in_dim, int out_dim, cudaStream_t stream)
{
    dim3 grid(out_dim / Q4KW_NTILE, active ? n_slot : n_expert);
    q4k_wmma_grouped_kernel<<<grid, 128, 0, stream>>>(
        out, (const uint8_t *)wbase, slab_stride, x, coloff, count, active, in_dim, out_dim);
}

#endif // FUCINA_Q4K_WMMA_CUH
