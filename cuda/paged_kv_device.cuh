// paged_kv_device.cuh — device-side paged-KV access kernels (Phase 2/3 infra).
//
// This header is the DEVICE counterpart to paged_kv.h. paged_kv.h owns the host
// bookkeeping (a free-list pool of physical blocks + per-sequence block tables
// mapping logical token positions to physical block ids); THIS header owns the
// CUDA kernels that read/write a sequence's KV through that block-table
// indirection, so a logically-contiguous sequence can live in physically
// scattered (and recyclable) blocks.
//
// The engine today (cuda/gemma4_kernels.cu) stores K/V for a layer contiguously
// as [pos][nkv*head_dim] (sliding) or [pos][head_dim] (global). The paged
// equivalent stores, per cache class, a single physical pool:
//
//     pool[ block_id ][ offset_in_block ][ elems_per_token ]   (fp8, kv_t)
//
// where elems_per_token = nkv*head_dim. A logical token position `p` of some
// sequence maps to (block_id, offset) = (table[p/BT], p%BT) via paged.h's
// paged_table_lookup (BT = PAGED_KV_BLOCK_TOKENS = 256). One physical block here
// holds BT positions of ONE layer (the engine uses separate device pools / block
// sub-ranges per layer; from a single kernel's point of view we always operate on
// one layer's pool, so we never stride by layer inside here).
//
// Goal of this file: prove that block-table indirection produces bit-identical
// attention output to the classic contiguous layout (same fp8 values, same scan
// order). It is correctness-first: the attention kernel is a flash/online-softmax
// single-(query-)head reference, mirroring sliding_attn_decode_kernel /
// global_attn_decode_kernel in gemma4_kernels.cu but with logical→physical
// translation per position. Perf tuning (split-K, vectorised loads, GQA
// broadcast) is deliberately out of scope and left to the engine stream.
//
// Style follows cuda/gemma4_kernels.cu: blockIdx.y selects the row/query,
// blockIdx.x*blockDim.x+threadIdx.x walks the head_dim, fp8 dequant in-register,
// online softmax with a 32-float block-reduce scratch.

#ifndef FUCINA_PAGED_KV_DEVICE_CUH
#define FUCINA_PAGED_KV_DEVICE_CUH

#include <cuda_fp8.h>      // __nv_fp8_storage_t + public CUDA 13 conversion APIs
#include <math.h>
#include "paged_kv.h"      // PAGED_KV_BLOCK_TOKENS, host pool/table types (no CUDA dep)

// ─────────────────────────────────────────────────────────────────────────────
// FP8 E4M3 conversion — small static __device__ copies of the engine's helpers
// (gemma4_kernels.cu:229-236). Replicated here on purpose: this header must not
// pull in the engine .cu (another stream owns it). Keep these byte-identical to
// the originals so paged and contiguous paths quantise to the SAME fp8 bits.
// ─────────────────────────────────────────────────────────────────────────────
typedef __nv_fp8_storage_t pkv_t;   // 1-byte FP8 storage element (== engine kv_t)

static inline __device__ float pkv_fp8_to_float(pkv_t v) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3)));
}

static inline __device__ pkv_t pkv_float_to_fp8(float v) {
    v = fminf(fmaxf(v, -448.0f), 448.0f);              // saturate to E4M3 range
    return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// ─────────────────────────────────────────────────────────────────────────────
// POD descriptors the engine fills per batched step. All pointers are DEVICE
// pointers. There is no CUDA-runtime type in here so the engine can build these
// on the host and memcpy them as plain bytes.
// ─────────────────────────────────────────────────────────────────────────────

// One row's view of a sequence's paged KV for the layer being processed.
// Mirrors what paged_kv.h's PagedBlockTable knows, flattened for the device:
//   - block_table: device int* of physical block ids (table->blocks copied to GPU)
//   - n_blocks   : table->n   (how many block ids are valid in block_table)
//   - base       : table->base (logical token index of block_table[0]; advances
//                  as a sliding sequence recycles leading blocks)
//   - n_tokens   : current logical length of the sequence (exclusive upper bound)
// Logical position p (base <= p < n_tokens) resolves to
//   block_id = block_table[p/BT - base/BT],  offset = p % BT.
struct PagedSeqView {
    const int *block_table;   // device: physical block ids, length n_blocks
    int        n_blocks;      // valid entries in block_table
    int        base;          // logical pos of block_table[0] (multiple of BT)
    int        n_tokens;      // logical sequence length (positions [0, n_tokens))
};

// A batch of rows for a write step. Row r writes ONE token (its current
// position write_pos[r]) of sequence seqs[r] into the pool. Different rows may
// belong to different sequences / block tables.
struct PagedWriteBatch {
    const PagedSeqView *seqs;       // device: B per-row views
    const int          *write_pos;  // device: B logical positions to write (one per row)
    int                 n_rows;     // B
};

// ─────────────────────────────────────────────────────────────────────────────
// Logical→physical address translation (device inline). Returns the flat fp8
// element index into the pool for (logical position p, element e), or (size_t)-1
// if p is not currently mapped (recycled / out of range). elems_per_token =
// nkv*head_dim. Pool element layout: [block_id][offset][elems_per_token].
// This is the device mirror of paged_kv.h's paged_table_lookup.
// ─────────────────────────────────────────────────────────────────────────────
static inline __device__ size_t paged_elem_index(
        const PagedSeqView &v, int p, int e, int block_tokens, int elems_per_token) {
    if (p < v.base) return (size_t)-1;
    int rel_block = p / block_tokens - v.base / block_tokens;
    if (rel_block < 0 || rel_block >= v.n_blocks) return (size_t)-1;
    int block_id = v.block_table[rel_block];
    int offset   = p % block_tokens;
    return (((size_t)block_id * block_tokens + offset) * (size_t)elems_per_token) + (size_t)e;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: paged_kv_write — scatter a batch of newly-projected K/V into the pool.
//
// Generalises kv_write_sliding_kernel / kv_write_global_kernel (gemma4_kernels.cu
// :2155-2184) to block-table indirection: instead of a single contiguous base +
// ring modulo, each row resolves its (block,offset) through its own sequence's
// block table. Each row writes its current token (batch.write_pos[row]).
//
//   kb / vb : float inputs, [n_rows][elems_per_token], row-major.
//   k_pool / v_pool : the layer's fp8 pools (one per cache class). For the
//             GLOBAL class K==V unified — the engine passes the same pointer for
//             both and identical data, exactly as today.
//   grid  = (ceil(elems_per_token/256), n_rows), block = 256.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void paged_kv_write(
        pkv_t *k_pool, pkv_t *v_pool,
        const float *kb, const float *vb,
        PagedWriteBatch batch,
        int block_tokens, int elems_per_token)
{
    int row = blockIdx.y;
    int e   = blockIdx.x * blockDim.x + threadIdx.x;   // element within token
    if (row >= batch.n_rows || e >= elems_per_token) return;

    const PagedSeqView v = batch.seqs[row];
    int p = batch.write_pos[row];                      // logical position to write
    size_t idx = paged_elem_index(v, p, e, block_tokens, elems_per_token);
    if (idx == (size_t)-1) return;                     // unmapped → skip (engine bug if hit)

    size_t src = (size_t)row * elems_per_token + e;
    k_pool[idx] = pkv_float_to_fp8(kb[src]);
    v_pool[idx] = pkv_float_to_fp8(vb[src]);
}

// ─────────────────────────────────────────────────────────────────────────────
// 32-lane block reduce (sum) — same shape as gemma4_kernels.cu's block_reduce_sum
// but self-contained. blockDim.x assumed <= 1024 (<=32 warps). Returns the sum to
// ALL lanes (broadcast), like the engine's reducer.
// ─────────────────────────────────────────────────────────────────────────────
static inline __device__ float pkv_block_reduce_sum(float val, float *smem) {
    for (int o = 16; o > 0; o >>= 1) val += __shfl_down_sync(0xffffffffu, val, o);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = val;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? smem[lane] : 0.0f;
    if (warp == 0) {
        for (int o = 16; o > 0; o >>= 1) val += __shfl_down_sync(0xffffffffu, val, o);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();
    return smem[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: paged_attn_gather — single-(query-)head flash attention reading KV
// through a sequence's block table. Correctness reference, not perf-tuned.
//
// Computes, for one query head (blockDim.x = head_dim threads, one output element
// per thread), online-softmax over the sequence's valid positions [lo, n_tokens):
//     out[d] = ( sum_p softmax(q·k_p) * v_p )[d]
// using logical→physical translation per position. `lo` lets the caller bound the
// scan to a sliding window: lo = max(0, n_tokens - window) reproduces the engine's
// sliding kernel; lo = 0 reproduces the global kernel. Scan order (ascending
// logical position) is identical to the contiguous kernels, so with identical fp8
// values the result is bit-identical.
//
// This reference handles a SINGLE kv head per launch to stay simple. That is
// exactly the GLOBAL class (nkv=1) and one (kv-head) slice of the SLIDING class.
// For a full multi-head sliding layer the engine launches one grid per kv head
// (or generalises with a GQA map — out of scope here). The kv-head slice is
// selected via (elems_per_token = pool stride, elem_off = kv_head*head_dim).
//
//   out   : float [head_dim].
//   q     : float [head_dim] for this query head (caller-offset).
//   k_pool/v_pool : layer fp8 pools. elems_per_token is the pool's per-token
//           stride; pass elems_per_token = nkv*head_dim and elem_off =
//           kv_head*head_dim to read just this head's slice.
//   blockDim.x = head_dim (<= 1024). grid = (1,1).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void paged_attn_gather(
        float       *out,          // [head_dim]
        const float *q,            // [head_dim]
        const pkv_t *k_pool,
        const pkv_t *v_pool,
        PagedSeqView view,         // this sequence's block table + length
        int          head_dim,
        int          block_tokens,
        int          elems_per_token,  // pool stride (nkv*head_dim)
        int          elem_off,         // offset into a token for this kv head's slice
        int          lo)               // first logical position to attend (window low bound)
{
    extern __shared__ float smem[];    // [<=32] block-reduce scratch
    int tid = threadIdx.x;

    float q_d = (tid < head_dim) ? q[tid] : 0.0f;
    float acc = 0.0f;                  // this thread's output element
    float m   = -INFINITY;             // running row max
    float l   = 0.0f;                  // running denominator

    int hi = view.n_tokens;
    for (int p = lo; p < hi; p++) {
        // logical → physical for element (elem_off + tid) of this head's slice
        float k_d = 0.0f, v_d = 0.0f;
        if (tid < head_dim) {
            size_t idx = paged_elem_index(view, p, elem_off + tid, block_tokens, elems_per_token);
            if (idx != (size_t)-1) {
                k_d = pkv_fp8_to_float(k_pool[idx]);
                v_d = pkv_fp8_to_float(v_pool[idx]);
            }
        }
        float s     = pkv_block_reduce_sum(q_d * k_d, smem);   // dot, broadcast (scale 1.0)
        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);   // m=-inf on first iter ⇒ alpha=0
        float p_w   = __expf(s - m_new);
        l   = l * alpha + p_w;
        acc = acc * alpha + p_w * v_d;
        m   = m_new;
    }

    if (tid < head_dim)
        out[tid] = (l > 0.0f) ? acc / l : 0.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: paged_attn_decode_batched — the CONTINUOUS-BATCHING primitive. One
// batched decode forward attends B INDEPENDENT sequences at once: block (h, s)
// computes query head h of sequence s, attending s's own KV through views[s]'s
// block table (its own length/base). This is what lets one forward pass serve
// many in-flight requests — the core of the scheduler's StepBatch. Correctness
// reference (per-head online softmax); split-K perf is a later optimisation.
//
//   out : float [n_seq][n_heads*head_dim] row-major.
//   q   : float [n_seq][n_heads*head_dim] row-major (each seq's query row).
//   k_pool/v_pool : ONE layer's pool (shared by all sequences; each indexes its
//         own blocks via views[s]). elems_per_token = nkv*head_dim.
//   views : [n_seq] PagedSeqView (per-seq block table + length + base).
//   GQA: query head h reads kv head h/(n_heads/n_kv_heads). window>0 ⇒ sliding.
//   grid = (n_heads, n_seq), block = head_dim.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void paged_attn_decode_batched(
        float       *out,
        const float *q,
        const pkv_t *k_pool,
        const pkv_t *v_pool,
        const PagedSeqView *views,
        int n_heads, int n_kv_heads, int head_dim,
        int window, int block_tokens, int elems_per_token)
{
    extern __shared__ float smem[];
    int h   = blockIdx.x;            // query head
    int s   = blockIdx.y;            // sequence (row)
    int tid = threadIdx.x;
    PagedSeqView v = views[s];
    int group    = n_heads / n_kv_heads;
    int elem_off = (h / group) * head_dim;
    int lo = (window > 0 && v.n_tokens > window) ? (v.n_tokens - window) : 0;
    if (v.base > lo) lo = v.base;   // never scan recycled (unmapped) leading positions

    size_t qbase = ((size_t)s * n_heads + h) * head_dim;
    float q_d = (tid < head_dim) ? q[qbase + tid] : 0.0f;
    float acc = 0.0f, m = -INFINITY, l = 0.0f;
    for (int p = lo; p < v.n_tokens; p++) {
        float k_d = 0.0f, val_d = 0.0f;
        if (tid < head_dim) {
            size_t idx = paged_elem_index(v, p, elem_off + tid, block_tokens, elems_per_token);
            if (idx != (size_t)-1) { k_d = pkv_fp8_to_float(k_pool[idx]); val_d = pkv_fp8_to_float(v_pool[idx]); }
        }
        float s_dot = pkv_block_reduce_sum(q_d * k_d, smem);
        float m_new = fmaxf(m, s_dot);
        float alpha = __expf(m - m_new);
        float p_w   = __expf(s_dot - m_new);
        l   = l * alpha + p_w;
        acc = acc * alpha + p_w * val_d;
        m   = m_new;
    }
    if (tid < head_dim) out[qbase + tid] = (l > 0.0f) ? acc / l : 0.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// SPLIT-K paged batched attention — the FAST continuous-batching primitive.
//
// paged_attn_decode_batched above is correctness-first: one block per (head,seq)
// scans the whole sequence with a per-key __syncthreads block-reduce. These three
// kernels replace it with the same flash-decoding split-K design the CONTIGUOUS
// decode uses (sliding_attn_splitk_rows_kernel / global_attn_splitk_rows_kernel +
// flash_decode_combine_rows_kernel in gemma4_kernels.cu), but reading K/V through
// each row's block table instead of a flat ring/contiguous cache.
//
// BIT-IDENTICAL to the contiguous split-K path by construction: for a given row,
//   - the split count n_splits is computed by the SAME formula (paged_row_splits,
//     a copy of attn_row_splits),
//   - `per`, `lo`, the i0..i1 partition, and the ascending-i scan order match the
//     contiguous rows kernels exactly,
//   - the per-lane register-slice dot/online-softmax recurrence is identical,
//   - the combine pass merges splits in the SAME order s = 0..n_splits-1.
// The only difference is the address of K[p]/V[p]: contiguous uses (p % cap), paged
// resolves p through the block table (paged_elem_index). With identical fp8 values
// at identical logical positions, every FP op reassociates identically → same bits.
//
// Geometry mirrors the contiguous kernels:
//   sliding: grid (n_splits, n_kv_heads, n_seq), block = NKV warps? No — one warp
//            per KV head like the contiguous kernel: block = NKV*32 threads, warp =
//            kv_head, each warp serves its GQ = NH/NKV query heads. grid.x = split,
//            grid.y = seq.  (NKV warps in ONE block; matches sliding_attn_splitk.)
//   global : grid (n_splits, n_seq), block = NH*32 threads, warp = query head
//            (GQA-broadcast over the single KV head).  (matches global_attn_splitk.)
// Per-row scratch slot = (seq*MAX_SPLITS + split) — same layout the rows kernels +
// combine use, so the engine reuses d_fa_acc/m/l unchanged.
// ─────────────────────────────────────────────────────────────────────────────

// Copy of attn_row_splits (gemma4_kernels.cu) so this header stays engine-free.
static __device__ __forceinline__ int paged_row_splits(int len, int chunk, int max_splits) {
    int s = (len + chunk - 1) / chunk;
    if (s < 1) s = 1;
    if (s > max_splits) s = max_splits;
    if (s > len && len > 0) s = len;
    return s;
}

// Sliding split-K, paged, batched over seqs. One warp per KV head; GQ query heads
// per warp. grid = (max_splits, n_seq); tail blocks (split >= n_splits_row) return.
template<int NH, int NKV, int HD, int MAX_SPLITS>
__global__ void paged_sliding_attn_splitk_batched(
    float *part_acc, float *part_m, float *part_l,   // slot (seq*MAX_SPLITS + split)
    const float *q,                                  // [n_seq][NH*HD] row-major
    const pkv_t *k_pool, const pkv_t *v_pool,        // ONE layer's pool
    const PagedSeqView *views,
    int window, int chunk, int block_tokens, int elems_per_token)
{
    constexpr int GQ = NH / NKV;
    constexpr int slice = HD / 32;
    static_assert(HD % 32 == 0, "lane-strided slices require HD multiple of warp size");
    int seq   = blockIdx.y;
    int split = blockIdx.x;
    PagedSeqView v = views[seq];
    int n_tokens   = v.n_tokens;
    int window_len = (n_tokens < window) ? n_tokens : window;
    int n_splits   = paged_row_splits(window_len, chunk, MAX_SPLITS);
    if (split >= n_splits) return;                   // tail blocks
    int kv_head = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;

    int lo  = n_tokens - window_len;
    if (v.base > lo) {
        // Recycled leading positions are unmapped; contiguous never hits this
        // (ring keeps the whole window live). Clamp so we only scan mapped pos —
        // the engine sizes the sliding pool to hold the full window, so in the
        // batch this clamp is a no-op (lo >= base always); kept for safety.
        lo = v.base;
        window_len = n_tokens - lo;
        n_splits   = paged_row_splits(window_len, chunk, MAX_SPLITS);
        if (split >= n_splits) return;
    }
    int per = (window_len + n_splits - 1) / n_splits;
    int i0  = split * per;
    int i1  = min(i0 + per, window_len);

    float qreg[GQ][slice], acc[GQ][slice], m[GQ], l[GQ];
    #pragma unroll
    for (int g = 0; g < GQ; g++) {
        const float *qp = q + (size_t)seq * NH * HD + (size_t)(kv_head*GQ + g)*HD;
        #pragma unroll
        for (int e = 0; e < slice; e++) { qreg[g][e] = qp[lane + 32*e]; acc[g][e] = 0.0f; }
        m[g] = -INFINITY; l[g] = 0.0f;
    }

    int elem_off = kv_head * HD;   // this kv head's slice into the per-token stride
    for (int i = i0; i < i1; i++) {
        int p = lo + i;            // absolute logical position
        float kd[slice], vd[slice];
        #pragma unroll
        for (int e = 0; e < slice; e++) {
            size_t idx = paged_elem_index(v, p, elem_off + lane + 32*e, block_tokens, elems_per_token);
            kd[e] = (idx != (size_t)-1) ? pkv_fp8_to_float(k_pool[idx]) : 0.0f;
            vd[e] = (idx != (size_t)-1) ? pkv_fp8_to_float(v_pool[idx]) : 0.0f;
        }
        #pragma unroll
        for (int g = 0; g < GQ; g++) {
            float dot = 0.0f;
            #pragma unroll
            for (int e = 0; e < slice; e++) dot += qreg[g][e] * kd[e];
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
            float s = dot;
            float mn = fmaxf(m[g], s), al = __expf(m[g] - mn), pw = __expf(s - mn);
            l[g] = l[g]*al + pw;
            #pragma unroll
            for (int e = 0; e < slice; e++) acc[g][e] = acc[g][e]*al + pw*vd[e];
            m[g] = mn;
        }
    }

    size_t slot = (size_t)seq * MAX_SPLITS + split;
    #pragma unroll
    for (int g = 0; g < GQ; g++) {
        int h = kv_head*GQ + g;
        float *pa = part_acc + (slot*NH + h)*HD;
        #pragma unroll
        for (int e = 0; e < slice; e++) pa[lane + 32*e] = acc[g][e];
        if (lane == 0) { part_m[slot*NH + h] = m[g]; part_l[slot*NH + h] = l[g]; }
    }
}

// Global split-K, paged, batched over seqs. One warp per query head (GQA-broadcast
// over the single KV head). grid = (max_splits, n_seq); tail blocks return.
//
// LANE LAYOUT matches global_attn_splitk_rows_kernel EXACTLY (not the sliding
// lane-per-element layout): each lane owns E = HD/128 groups of 4 CONTIGUOUS dims
// (qreg[E][4], element 4*(lane+32*e)+j), and the per-lane dot accumulates those 4
// terms per group BEFORE the warp reduce. Matching this grouping is what makes the
// dot product bit-identical to the contiguous path (FP add reassociation order).
template<int NH, int HD, int MAX_SPLITS>
__global__ void paged_global_attn_splitk_batched(
    float *part_acc, float *part_m, float *part_l,   // slot (seq*MAX_SPLITS + split)
    const float *q,                                  // [n_seq][NH*HD] row-major
    const pkv_t *k_pool, const pkv_t *v_pool,        // ONE layer's pool (nkv=1)
    const PagedSeqView *views,
    int chunk, int block_tokens, int elems_per_token)
{
    constexpr int E = HD / 128;      // groups of 4 dims per lane (4 at HD 512)
    static_assert(HD % 128 == 0, "global lane slices require HD multiple of 128");
    int seq   = blockIdx.y;
    int split = blockIdx.x;
    PagedSeqView v = views[seq];
    int ctx_len  = v.n_tokens;
    int lo = (v.base > 0) ? v.base : 0;   // global pool holds full ctx; base 0 normally
    int len = ctx_len - lo;
    int n_splits = paged_row_splits(len, chunk, MAX_SPLITS);
    if (split >= n_splits) return;
    int h    = threadIdx.x >> 5;     // warp = query head
    int lane = threadIdx.x & 31;

    int per = (len + n_splits - 1) / n_splits;
    int t0  = lo + split * per;
    int t1  = min(t0 + per, ctx_len);

    float qreg[E][4], acc[E][4], m = -INFINITY, l = 0.0f;
    const float *qp = q + (size_t)seq * NH * HD + (size_t)h * HD;
    #pragma unroll
    for (int e = 0; e < E; e++)
        #pragma unroll
        for (int j = 0; j < 4; j++) { qreg[e][j] = qp[4*(lane + 32*e) + j]; acc[e][j] = 0.0f; }

    for (int p = t0; p < t1; p++) {
        float kd[E][4], vd[E][4];
        #pragma unroll
        for (int e = 0; e < E; e++)
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                size_t idx = paged_elem_index(v, p, 4*(lane + 32*e) + j, block_tokens, elems_per_token);
                kd[e][j] = (idx != (size_t)-1) ? pkv_fp8_to_float(k_pool[idx]) : 0.0f;
                vd[e][j] = (idx != (size_t)-1) ? pkv_fp8_to_float(v_pool[idx]) : 0.0f;
            }
        float dot = 0.0f;
        #pragma unroll
        for (int e = 0; e < E; e++)
            dot += qreg[e][0]*kd[e][0] + qreg[e][1]*kd[e][1]
                 + qreg[e][2]*kd[e][2] + qreg[e][3]*kd[e][3];
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);
        float s = dot;
        float mn = fmaxf(m, s), al = __expf(m - mn), pw = __expf(s - mn);
        l = l*al + pw;
        #pragma unroll
        for (int e = 0; e < E; e++)
            #pragma unroll
            for (int j = 0; j < 4; j++) acc[e][j] = acc[e][j]*al + pw*vd[e][j];
        m = mn;
    }

    size_t slot = (size_t)seq * MAX_SPLITS + split;
    float *pa = part_acc + (slot*NH + h)*HD;
    #pragma unroll
    for (int e = 0; e < E; e++)
        #pragma unroll
        for (int j = 0; j < 4; j++) pa[4*(lane + 32*e) + j] = acc[e][j];
    if (lane == 0) { part_m[slot*NH + h] = m; part_l[slot*NH + h] = l; }
}

// Merge each seq-row's split partials into out[seq][NH*HD]. Mirrors
// flash_decode_combine_rows_kernel: merge order s = 0..n_splits-1, scratch slot
// (seq*MAX_SPLITS + split). window>0 ⇒ sliding (len=min(n_tokens,window)), else
// global (len=n_tokens). grid = (NH, n_seq), block = HD threads.
template<int NH, int MAX_SPLITS>
__global__ void paged_flash_decode_combine_batched(
    float *out,                                      // [n_seq][NH*HD]
    const float *part_acc, const float *part_m, const float *part_l,
    const PagedSeqView *views,
    int head_dim, int window, int chunk)
{
    int seq = blockIdx.y;
    int h   = blockIdx.x;
    int tid = threadIdx.x;
    PagedSeqView v = views[seq];
    int len = v.n_tokens;
    int lo  = 0;
    if (window > 0) { int wl = (len < window) ? len : window; lo = len - wl; len = wl; }
    if (v.base > lo) { len = v.n_tokens - v.base; }   // mirror the kernels' base clamp
    int n_splits = paged_row_splits(len, chunk, MAX_SPLITS);
    size_t base = (size_t)seq * MAX_SPLITS;
    float M = -INFINITY;
    for (int s = 0; s < n_splits; s++) M = fmaxf(M, part_m[(base + s)*NH + h]);
    float L = 0.0f, accv = 0.0f;
    for (int s = 0; s < n_splits; s++) {
        float ms = part_m[(base + s)*NH + h];
        if (ms == -INFINITY) continue;
        float scale = __expf(ms - M);
        L    += part_l[(base + s)*NH + h] * scale;
        accv += part_acc[((base + s)*NH + h)*head_dim + tid] * scale;
    }
    if (tid < head_dim)
        out[(size_t)seq*NH*head_dim + h*head_dim + tid] = (L > 0.0f) ? accv / L : 0.0f;
}

#endif // FUCINA_PAGED_KV_DEVICE_CUH
