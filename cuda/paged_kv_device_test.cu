// paged_kv_device_test.cu — standalone correctness test for the paged-KV device
// kernels (paged_kv_device.cuh). Builds a tiny pool + a few sequences with
// INTERLEAVED block tables (so logically-contiguous tokens land in physically
// scattered blocks), writes known K/V through paged_kv_write, fills a CONTIGUOUS
// reference cache the classic way, runs attention BOTH ways, and asserts the
// outputs are bit-identical (same fp8 values, same scan order ⇒ exact).
//
// Also tests a SLIDING sequence whose leading blocks have been recycled
// (table.base > 0) to prove logical→physical resolution honours `base`.
//
// Build (CUDA 13, GB10 / Blackwell sm_121a):
//   /usr/local/cuda/bin/nvcc -arch=sm_121a -o /tmp/paged_kv_device_test \
//       cuda/paged_kv_device_test.cu && /tmp/paged_kv_device_test
//
// Prints "paged_kv_device: ALL TESTS PASSED" on success.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "paged_kv.h"            // host pool/table bookkeeping
#include "paged_kv_device.cuh"  // the kernels under test

// ── Device reference kernel: the SAME flash-attention math as paged_attn_gather,
// but reading a CONTIGUOUS fp8 cache [pos][elems_per_token] the classic way (no
// block table). Because it shares the exact arithmetic (same __expf, same
// __shfl reduction, same scan order) the only difference vs paged_attn_gather is
// the address translation — so a match here is BIT-EXACT, proving the
// indirection alone. (gen-style mirror of global_attn_decode_kernel.)
__global__ void contig_attn_gather(
        float *out, const float *q,
        const pkv_t *kc, const pkv_t *vc,   // contiguous [n_tokens][elems_per_token]
        int head_dim, int elems_per_token, int elem_off,
        int lo, int n_tokens)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float q_d = (tid < head_dim) ? q[tid] : 0.0f;
    float acc = 0.0f, m = -INFINITY, l = 0.0f;
    for (int p = lo; p < n_tokens; p++) {
        float k_d = 0.0f, v_d = 0.0f;
        if (tid < head_dim) {
            size_t idx = (size_t)p * elems_per_token + elem_off + tid;
            k_d = pkv_fp8_to_float(kc[idx]);
            v_d = pkv_fp8_to_float(vc[idx]);
        }
        float s     = pkv_block_reduce_sum(q_d * k_d, smem);
        float m_new = fmaxf(m, s);
        float alpha = __expf(m - m_new);
        float p_w   = __expf(s - m_new);
        l   = l * alpha + p_w;
        acc = acc * alpha + p_w * v_d;
        m   = m_new;
    }
    if (tid < head_dim) out[tid] = (l > 0.0f) ? acc / l : 0.0f;
}

// ── Device readback kernel: dequant every element of a sequence's KV through
// the PAGED block table into a dense [n_tokens][elems_per_token] float buffer.
// No accumulation, so this is the cleanest BIT-EXACT proof of the indirection:
// compared against the same dequant of the contiguous reference it must match to
// the bit. grid=(ceil(ept/256), n_tokens), block=256.
__global__ void paged_readback(
        float *dst,                        // [n_tokens][elems_per_token]
        const pkv_t *k_pool,
        PagedSeqView view, int block_tokens, int elems_per_token)
{
    int p = blockIdx.y;                     // logical position
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= view.n_tokens || e >= elems_per_token) return;
    if (p < view.base) { dst[(size_t)p * elems_per_token + e] = 0.0f; return; }
    size_t idx = paged_elem_index(view, p, e, block_tokens, elems_per_token);
    float val = (idx == (size_t)-1) ? 0.0f : pkv_fp8_to_float(k_pool[idx]);
    dst[(size_t)p * elems_per_token + e] = val;
}

#define CK(call)                                                          \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                    cudaGetErrorString(_e), __FILE__, __LINE__);          \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

// ── Host-side fp8 E4M3 round-trip, matching the device helpers exactly. We use
// it to build the CONTIGUOUS reference cache with the SAME quantisation the
// device write performs, so the only thing under test is the indirection, not
// fp8 rounding. (We rely on the host CUDA fp8 conversion intrinsics, available
// from cuda_fp8.h in host code under nvcc.)
#include <cuda_fp8.h>
static inline __nv_fp8_storage_t h_float_to_fp8(float v) {
    v = fminf(fmaxf(v, -448.0f), 448.0f);
    return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// ── Geometry (Gemma 4). We test BOTH cache classes' per-token element counts.
static const int BT = PAGED_KV_BLOCK_TOKENS;     // 256

// A deterministic "projected" K/V value for (sequence, logical position, element).
// Small magnitudes (well inside E4M3 ±448) so quantisation is the only loss.
static float gen_k(int seq, int pos, int e) {
    return 0.5f * sinf(0.01f * (float)(seq * 131 + pos * 7 + e * 3 + 1));
}
static float gen_v(int seq, int pos, int e) {
    return 0.5f * cosf(0.013f * (float)(seq * 97 + pos * 5 + e * 11 + 2));
}
static float gen_q(int seq, int e) {
    return 0.3f * sinf(0.02f * (float)(seq * 17 + e * 13 + 5));
}

// Host reference: contiguous single-head flash attention over [lo, n_tokens),
// reading an fp8 contiguous cache [pos][elems_per_token] for one head slice.
static void ref_attn(const std::vector<__nv_fp8_storage_t> &kc,
                     const std::vector<__nv_fp8_storage_t> &vc,
                     const std::vector<float> &q,
                     int head_dim, int elems_per_token, int elem_off,
                     int lo, int n_tokens,
                     std::vector<float> &out) {
    out.assign(head_dim, 0.0f);
    std::vector<float> acc(head_dim, 0.0f);
    float m = -INFINITY, l = 0.0f;
    for (int p = lo; p < n_tokens; p++) {
        // dot
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float kd = __half2float(__half(__nv_cvt_fp8_to_halfraw(
                kc[(size_t)p * elems_per_token + elem_off + d], __NV_E4M3)));
            s += q[d] * kd;
        }
        float m_new = fmaxf(m, s);
        float alpha = expf(m - m_new);
        float pw    = expf(s - m_new);
        l = l * alpha + pw;
        for (int d = 0; d < head_dim; d++) {
            float vd = __half2float(__half(__nv_cvt_fp8_to_halfraw(
                vc[(size_t)p * elems_per_token + elem_off + d], __NV_E4M3)));
            acc[d] = acc[d] * alpha + pw * vd;
        }
        m = m_new;
    }
    for (int d = 0; d < head_dim; d++) out[d] = (l > 0.0f) ? acc[d] / l : 0.0f;
}

static int g_fail = 0;
static float max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
    float mx = 0.0f;
    for (size_t i = 0; i < a.size(); i++) mx = fmaxf(mx, fabsf(a[i] - b[i]));
    return mx;
}

// ═════════════════════════════════════════════════════════════════════════════
// Test driver for one cache class (parameterised by nkv/head_dim).
//   - builds a pool of n_blocks
//   - creates `n_seq` sequences with interleaved block tables
//   - writes K/V via paged_kv_write, plus a contiguous reference
//   - runs paged_attn_gather vs ref_attn for one query head per (seq, kv_head)
//   - optionally recycles leading blocks on the last sliding seq (base>0)
// ═════════════════════════════════════════════════════════════════════════════
static void run_class(const char *name, int nkv, int head_dim,
                      const std::vector<int> &seq_lens, bool recycle_last) {
    int ept = nkv * head_dim;                 // elems per token
    int n_seq = (int)seq_lens.size();

    // Pool big enough for all sequences' blocks (plus slack).
    int total_blocks = 0;
    for (int n : seq_lens) total_blocks += (n + BT - 1) / BT + 1;
    int n_blocks = total_blocks + 4;

    PagedBlockPool pool;
    if (paged_pool_init(&pool, n_blocks, BT) != 0) { printf("  pool init failed\n"); g_fail++; return; }

    std::vector<PagedBlockTable> tabs(n_seq);
    for (int s = 0; s < n_seq; s++) paged_table_init(&tabs[s]);

    // INTERLEAVE: allocate one block at a time round-robin across sequences so
    // each sequence's logically-contiguous tokens land in physically scattered
    // blocks. We grow tables to cover their lengths but force the allocation
    // order to be interleaved by ensuring 1 block at a time across seqs.
    {
        // how many blocks each seq ultimately needs
        std::vector<int> need(n_seq);
        int maxneed = 0;
        for (int s = 0; s < n_seq; s++) {
            need[s] = (seq_lens[s] + BT - 1) / BT;
            if (need[s] > maxneed) maxneed = need[s];
        }
        for (int b = 0; b < maxneed; b++)
            for (int s = 0; s < n_seq; s++)
                if (b < need[s]) {
                    int got = paged_pool_alloc(&pool);
                    if (got < 0) { printf("  pool exhausted\n"); g_fail++; }
                    paged_table_reserve(&tabs[s], tabs[s].n + 1);
                    tabs[s].blocks[tabs[s].n++] = got;
                }
    }

    // Optionally recycle the leading block(s) of the LAST sequence to set base>0.
    // We emulate the sliding recycle: drop block[0], advance base by BT. Only do
    // this if the seq spans >1 block so positions in the dropped block are no
    // longer attended (we then attend from lo = base).
    int recycle_base = 0;
    if (recycle_last && tabs[n_seq - 1].n > 1) {
        PagedBlockTable &t = tabs[n_seq - 1];
        paged_pool_free(&pool, t.blocks[0]);
        memmove(&t.blocks[0], &t.blocks[1], (size_t)(t.n - 1) * sizeof(int));
        t.n--;
        t.base = BT;
        recycle_base = BT;
    }

    // ── Build host K/V (float) for ALL tokens of ALL sequences, and the
    // CONTIGUOUS fp8 reference cache per sequence (only its mapped positions).
    // We'll write to the device pool one token at a time via paged_kv_write
    // batches; here we batch ALL (seq, pos) writes whose position is mapped.
    struct WriteItem { int seq; int pos; };
    std::vector<WriteItem> items;
    for (int s = 0; s < n_seq; s++) {
        int lo = (s == n_seq - 1 && recycle_last) ? recycle_base : 0;
        for (int p = lo; p < seq_lens[s]; p++) items.push_back({s, p});
    }
    int B = (int)items.size();

    // Host float K/V batch [B][ept], row-major.
    std::vector<float> hKb((size_t)B * ept), hVb((size_t)B * ept);
    for (int i = 0; i < B; i++)
        for (int e = 0; e < ept; e++) {
            hKb[(size_t)i * ept + e] = gen_k(items[i].seq, items[i].pos, e);
            hVb[(size_t)i * ept + e] = gen_v(items[i].seq, items[i].pos, e);
        }

    // Per-sequence contiguous reference fp8 cache, indexed by ABSOLUTE position.
    std::vector<std::vector<__nv_fp8_storage_t>> refK(n_seq), refV(n_seq);
    for (int s = 0; s < n_seq; s++) {
        refK[s].assign((size_t)seq_lens[s] * ept, 0);
        refV[s].assign((size_t)seq_lens[s] * ept, 0);
    }
    for (int i = 0; i < B; i++)
        for (int e = 0; e < ept; e++) {
            refK[items[i].seq][(size_t)items[i].pos * ept + e] =
                h_float_to_fp8(hKb[(size_t)i * ept + e]);
            refV[items[i].seq][(size_t)items[i].pos * ept + e] =
                h_float_to_fp8(hVb[(size_t)i * ept + e]);
        }

    // ── Device pool + block tables + PagedSeqViews + write batch ──
    pkv_t *dKpool, *dVpool;
    size_t pool_elems = (size_t)n_blocks * BT * ept;
    CK(cudaMalloc(&dKpool, pool_elems * sizeof(pkv_t)));
    CK(cudaMalloc(&dVpool, pool_elems * sizeof(pkv_t)));
    CK(cudaMemset(dKpool, 0, pool_elems * sizeof(pkv_t)));
    CK(cudaMemset(dVpool, 0, pool_elems * sizeof(pkv_t)));

    // Copy each table's blocks[] to device; build a host PagedSeqView per seq.
    std::vector<int *> dTables(n_seq, nullptr);
    std::vector<PagedSeqView> hViews(n_seq);
    for (int s = 0; s < n_seq; s++) {
        CK(cudaMalloc(&dTables[s], (size_t)tabs[s].n * sizeof(int)));
        CK(cudaMemcpy(dTables[s], tabs[s].blocks, (size_t)tabs[s].n * sizeof(int),
                      cudaMemcpyHostToDevice));
        hViews[s] = PagedSeqView{dTables[s], tabs[s].n, tabs[s].base, seq_lens[s]};
    }

    // Per-row view + write_pos (which view, which logical position to write).
    std::vector<PagedSeqView> hRowViews(B);
    std::vector<int> hWritePos(B);
    for (int i = 0; i < B; i++) {
        hRowViews[i]  = hViews[items[i].seq];
        hWritePos[i]  = items[i].pos;
    }
    PagedSeqView *dRowViews; int *dWritePos;
    CK(cudaMalloc(&dRowViews, (size_t)B * sizeof(PagedSeqView)));
    CK(cudaMalloc(&dWritePos, (size_t)B * sizeof(int)));
    CK(cudaMemcpy(dRowViews, hRowViews.data(), (size_t)B * sizeof(PagedSeqView), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWritePos, hWritePos.data(), (size_t)B * sizeof(int), cudaMemcpyHostToDevice));

    float *dKb, *dVb;
    CK(cudaMalloc(&dKb, (size_t)B * ept * sizeof(float)));
    CK(cudaMalloc(&dVb, (size_t)B * ept * sizeof(float)));
    CK(cudaMemcpy(dKb, hKb.data(), (size_t)B * ept * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dVb, hVb.data(), (size_t)B * ept * sizeof(float), cudaMemcpyHostToDevice));

    PagedWriteBatch batch{dRowViews, dWritePos, B};

    // Launch write: grid=(ceil(ept/256), B), block=256.
    dim3 wblock(256);
    dim3 wgrid((ept + 255) / 256, B);
    paged_kv_write<<<wgrid, wblock>>>(dKpool, dVpool, dKb, dVb, batch, BT, ept);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    // ── Upload each sequence's CONTIGUOUS fp8 reference cache to device, so we
    // can run the SAME-arithmetic contig_attn_gather and get a BIT-EXACT compare.
    std::vector<pkv_t *> dRefK(n_seq, nullptr), dRefV(n_seq, nullptr);
    for (int s = 0; s < n_seq; s++) {
        size_t n = refK[s].size();
        CK(cudaMalloc(&dRefK[s], n * sizeof(pkv_t)));
        CK(cudaMalloc(&dRefV[s], n * sizeof(pkv_t)));
        CK(cudaMemcpy(dRefK[s], refK[s].data(), n * sizeof(pkv_t), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dRefV[s], refV[s].data(), n * sizeof(pkv_t), cudaMemcpyHostToDevice));
    }

    // ── BIT-EXACT proof of the indirection: dequant every mapped element through
    // the paged block table and compare to the contiguous reference. Same fp8
    // bytes, no accumulation ⇒ must match to the bit (diff exactly 0). This is the
    // real correctness gate for the address translation.
    float readback_diff = 0.0f;
    for (int s = 0; s < n_seq; s++) {
        size_t n = (size_t)seq_lens[s] * ept;
        float *dDense; CK(cudaMalloc(&dDense, n * sizeof(float)));
        dim3 rblk(256), rgrid((ept + 255) / 256, seq_lens[s]);
        paged_readback<<<rgrid, rblk>>>(dDense, dKpool, hViews[s], BT, ept);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<float> dense(n);
        CK(cudaMemcpy(dense.data(), dDense, n * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(dDense);
        // Compare to host dequant of the contiguous reference, for mapped positions.
        int lo = tabs[s].base;
        for (int p = lo; p < seq_lens[s]; p++)
            for (int e = 0; e < ept; e++) {
                float want = __half2float(__half(__nv_cvt_fp8_to_halfraw(
                    refK[s][(size_t)p * ept + e], __NV_E4M3)));
                float got = dense[(size_t)p * ept + e];
                readback_diff = fmaxf(readback_diff, fabsf(got - want));
            }
    }

    // ── Attention, for every (seq, kv_head). One query head per kv head (no GQA
    // broadcast in this reference). lo = base for the recycled sliding seq.
    //   worst_pc   : paged GPU vs contiguous GPU — same online-softmax math, only
    //                indirection differs. Tiny float-contraction (FMA) divergence
    //                may remain, so checked against a tight epsilon, not == 0.
    //   worst_host : paged GPU vs host float reference (numerical sanity).
    float worst_pc = 0.0f, worst_host = 0.0f;
    float *dOut, *dOutC, *dQ;
    CK(cudaMalloc(&dOut,  head_dim * sizeof(float)));
    CK(cudaMalloc(&dOutC, head_dim * sizeof(float)));
    CK(cudaMalloc(&dQ,    head_dim * sizeof(float)));

    for (int s = 0; s < n_seq; s++) {
        int lo = tabs[s].base;   // recycled leading positions are no longer attended
        for (int kvh = 0; kvh < nkv; kvh++) {
            int elem_off = kvh * head_dim;

            std::vector<float> hQ(head_dim);
            for (int d = 0; d < head_dim; d++) hQ[d] = gen_q(s * 31 + kvh, d);
            CK(cudaMemcpy(dQ, hQ.data(), head_dim * sizeof(float), cudaMemcpyHostToDevice));

            size_t shmem = 32 * sizeof(float);

            // paged gather (block-table indirected)
            paged_attn_gather<<<1, head_dim, shmem>>>(
                dOut, dQ, dKpool, dVpool, hViews[s],
                head_dim, BT, ept, elem_off, lo);
            CK(cudaGetLastError());
            // contiguous gather (same math, no indirection)
            contig_attn_gather<<<1, head_dim, shmem>>>(
                dOutC, dQ, dRefK[s], dRefV[s],
                head_dim, ept, elem_off, lo, seq_lens[s]);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());

            std::vector<float> gpu(head_dim), gpuC(head_dim);
            CK(cudaMemcpy(gpu.data(),  dOut,  head_dim * sizeof(float), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(gpuC.data(), dOutC, head_dim * sizeof(float), cudaMemcpyDeviceToHost));

            // host float reference (numerical sanity)
            std::vector<float> ref;
            ref_attn(refK[s], refV[s], hQ, head_dim, ept, elem_off, lo, seq_lens[s], ref);

            worst_pc   = fmaxf(worst_pc,   max_abs_diff(gpu, gpuC));
            worst_host = fmaxf(worst_host, max_abs_diff(gpu, ref));
        }
    }

    // ── Batched multi-seq primitive: paged_attn_decode_batched attends ALL
    // sequences in ONE launch (block (h,s) → seq s's KV via its block table).
    // Validate it row-by-row against the single-seq gather: must be bit-identical
    // (same math, same per-seq indirection). This is the continuous-batching
    // attention the scheduler's StepBatch will call.
    float worst_batched = 0.0f;
    {
        int nh = nkv;   // one query head per kv head (group 1) for this check
        PagedSeqView *dViews;
        CK(cudaMalloc(&dViews, (size_t)n_seq * sizeof(PagedSeqView)));
        CK(cudaMemcpy(dViews, hViews.data(), (size_t)n_seq * sizeof(PagedSeqView), cudaMemcpyHostToDevice));

        size_t qsz = (size_t)n_seq * nh * head_dim;
        std::vector<float> hQb(qsz);
        for (int s = 0; s < n_seq; s++)
            for (int h = 0; h < nh; h++)
                for (int d = 0; d < head_dim; d++)
                    hQb[((size_t)s * nh + h) * head_dim + d] = gen_q(s * 31 + h, d);
        float *dQb, *dOb;
        CK(cudaMalloc(&dQb, qsz * sizeof(float)));
        CK(cudaMalloc(&dOb, qsz * sizeof(float)));
        CK(cudaMemcpy(dQb, hQb.data(), qsz * sizeof(float), cudaMemcpyHostToDevice));

        dim3 bgrid(nh, n_seq);
        paged_attn_decode_batched<<<bgrid, head_dim, 32 * sizeof(float)>>>(
            dOb, dQb, dKpool, dVpool, dViews, nh, nkv, head_dim, /*window=*/0, BT, ept);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<float> hOb(qsz);
        CK(cudaMemcpy(hOb.data(), dOb, qsz * sizeof(float), cudaMemcpyDeviceToHost));

        // reference: single-seq gather per (s, h)
        for (int s = 0; s < n_seq; s++) {
            int lo = tabs[s].base;
            for (int h = 0; h < nh; h++) {
                std::vector<float> hQ(head_dim);
                for (int d = 0; d < head_dim; d++) hQ[d] = gen_q(s * 31 + h, d);
                CK(cudaMemcpy(dQ, hQ.data(), head_dim * sizeof(float), cudaMemcpyHostToDevice));
                paged_attn_gather<<<1, head_dim, 32 * sizeof(float)>>>(
                    dOut, dQ, dKpool, dVpool, hViews[s], head_dim, BT, ept, h * head_dim, lo);
                CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
                std::vector<float> g(head_dim);
                CK(cudaMemcpy(g.data(), dOut, head_dim * sizeof(float), cudaMemcpyDeviceToHost));
                for (int d = 0; d < head_dim; d++)
                    worst_batched = fmaxf(worst_batched,
                        fabsf(g[d] - hOb[((size_t)s * nh + h) * head_dim + d]));
            }
        }
        cudaFree(dViews); cudaFree(dQb); cudaFree(dOb);
    }

    // readback_diff: MUST be exactly 0 (bit-exact indirection).
    // worst_pc / worst_host: float-arithmetic tolerance only.
    bool ok = (readback_diff == 0.0f) && (worst_pc < 1e-2f) && (worst_host < 1e-2f)
              && (worst_batched == 0.0f);
    printf("  [%-7s] nkv=%d hd=%d seqs=%d%s  readback=%.3g (EXACT)  "
           "attn paged-vs-contig=%.3g  paged-vs-host=%.3g  batched-vs-single=%.3g  %s\n",
           name, nkv, head_dim, n_seq, recycle_last ? " (recycled base>0)" : "",
           readback_diff, worst_pc, worst_host, worst_batched, ok ? "OK" : "FAIL");
    if (!ok) g_fail++;

    cudaFree(dOutC);
    for (int s = 0; s < n_seq; s++) { cudaFree(dRefK[s]); cudaFree(dRefV[s]); }

    // cleanup
    cudaFree(dOut); cudaFree(dQ);
    cudaFree(dKb); cudaFree(dVb);
    cudaFree(dRowViews); cudaFree(dWritePos);
    for (int s = 0; s < n_seq; s++) cudaFree(dTables[s]);
    cudaFree(dKpool); cudaFree(dVpool);
    for (int s = 0; s < n_seq; s++) paged_table_free_struct(&tabs[s]);
    paged_pool_destroy(&pool);
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    printf("paged_kv_device test on: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    // GLOBAL class: nkv=1, head_dim=512. Sequences shorter and longer than a block.
    run_class("global", 1, 512, std::vector<int>{300, 100, 520}, /*recycle_last=*/false);

    // SLIDING class: nkv=8, head_dim=256. Multiple kv heads, interleaved tables.
    run_class("sliding", 8, 256, std::vector<int>{257, 90, 600}, /*recycle_last=*/false);

    // SLIDING with recycled leading block on the last seq (base>0, len spans 3 blocks).
    run_class("sliding", 8, 256, std::vector<int>{257, 90, 700}, /*recycle_last=*/true);

    if (g_fail == 0) {
        printf("paged_kv_device: ALL TESTS PASSED\n");
        return 0;
    }
    printf("paged_kv_device: %d TEST(S) FAILED\n", g_fail);
    return 1;
}
