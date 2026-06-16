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

#endif // FUCINA_PAGED_KV_DEVICE_CUH
