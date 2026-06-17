// paged_kv.h — host-side paged KV-cache bookkeeping for continuous batching.
//
// This header owns ONLY the allocation bookkeeping for a paged KV cache: a
// fixed pool of physical blocks (a free-list) and per-sequence block tables
// that map logical token positions to physical blocks. It has NO CUDA
// dependency on purpose — the policy here (which block a token lives in, when a
// block can be recycled, how many blocks fit a VRAM budget) is pure integer
// logic and is unit-tested on the host without a GPU. The engine allocates the
// matching device storage (n_blocks * bytes_per_block) and indexes it using the
// block ids these tables hand out.
//
// Geometry (Gemma 4, see gemma4_kernels.cuh):
//   - one BlockPool per cache class: SLIDING (40 layers, nkv=8, hd=256) and
//     GLOBAL (8 layers, nkv=1, hd=512). The two classes have different
//     bytes-per-token, hence separate pools sized independently.
//   - a block holds PAGED_KV_BLOCK_TOKENS positions for EVERY layer in its
//     class; the device side strides by layer within the block.
//
// Sliding-window recycling: a sliding sequence attends only the last
// GEMMA4_SLIDING_WINDOW tokens, so once the window has fully advanced past a
// block, that block is freed back to the pool. A sliding sequence therefore
// holds a BOUNDED number of blocks (ceil(window/block_tokens)+1) regardless of
// how long it runs — this is what lets many long sequences share one pool.
//
// Thread-safety: none. The scheduler owns the pools on a single goroutine/thread
// and calls these between batched steps.

#ifndef FUCINA_PAGED_KV_H
#define FUCINA_PAGED_KV_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef PAGED_KV_BLOCK_TOKENS
#define PAGED_KV_BLOCK_TOKENS 256   // positions per block (tunable; power-of-two friendly)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ── Block pool: a free-list over n_blocks physical blocks. ───────────────────
typedef struct {
    int  n_blocks;       // total physical blocks backing this pool
    int  block_tokens;   // positions per block
    int *free_stack;     // LIFO of currently-free block ids (size n_blocks)
    int  free_top;       // count of free ids on the stack (== free blocks)
} PagedBlockPool;

// ── Per-sequence block table: logical_token/block_tokens -> physical block id.
typedef struct {
    int *blocks;         // physical block ids; blocks[i] covers tokens [i*bt,(i+1)*bt)
    int  n;              // number of mapped blocks
    int  cap;            // capacity of blocks[]
    int  base;           // logical token index of blocks[0] (advances as sliding recycles)
} PagedBlockTable;

// Compute how many blocks of bytes_per_block fit in a VRAM budget, after holding
// back `reserve_bytes`. Returns >= 0 (0 if nothing fits). Used to size the pool
// dynamically from free VRAM after weights are resident.
static inline int paged_blocks_for_budget(uint64_t free_bytes,
                                          uint64_t reserve_bytes,
                                          uint64_t bytes_per_block) {
    if (bytes_per_block == 0) return 0;
    if (free_bytes <= reserve_bytes) return 0;
    uint64_t usable = free_bytes - reserve_bytes;
    uint64_t n = usable / bytes_per_block;
    if (n > (uint64_t)0x7fffffff) n = 0x7fffffff;
    return (int)n;
}

// Initialise a pool of n_blocks. Returns 0 on success, -1 on OOM/bad args.
// All blocks start free; ids are 0..n_blocks-1.
static inline int paged_pool_init(PagedBlockPool *p, int n_blocks, int block_tokens) {
    if (!p || n_blocks < 0 || block_tokens <= 0) return -1;
    p->n_blocks = n_blocks;
    p->block_tokens = block_tokens;
    p->free_top = n_blocks;
    p->free_stack = NULL;
    if (n_blocks > 0) {
        p->free_stack = (int *)malloc((size_t)n_blocks * sizeof(int));
        if (!p->free_stack) { p->n_blocks = 0; p->free_top = 0; return -1; }
        // Push in reverse so the first alloc hands out block 0 (nicer locality).
        for (int i = 0; i < n_blocks; i++) p->free_stack[i] = n_blocks - 1 - i;
    }
    return 0;
}

static inline void paged_pool_destroy(PagedBlockPool *p) {
    if (!p) return;
    free(p->free_stack);
    p->free_stack = NULL;
    p->n_blocks = p->free_top = 0;
}

static inline int paged_pool_free_blocks(const PagedBlockPool *p) {
    return p ? p->free_top : 0;
}

// Pop one free block id, or -1 if the pool is exhausted.
static inline int paged_pool_alloc(PagedBlockPool *p) {
    if (!p || p->free_top <= 0) return -1;
    return p->free_stack[--p->free_top];
}

// Return a block id to the pool. Caller must not double-free (debug-checked).
static inline void paged_pool_free(PagedBlockPool *p, int block_id) {
    if (!p || block_id < 0 || block_id >= p->n_blocks) return;
    if (p->free_top >= p->n_blocks) return;   // pool already full — ignore stray free
    p->free_stack[p->free_top++] = block_id;
}

// ── Block tables ─────────────────────────────────────────────────────────────
static inline void paged_table_init(PagedBlockTable *t) {
    if (!t) return;
    t->blocks = NULL; t->n = 0; t->cap = 0; t->base = 0;
}

static inline void paged_table_free_struct(PagedBlockTable *t) {
    if (!t) return;
    free(t->blocks);
    t->blocks = NULL; t->n = 0; t->cap = 0; t->base = 0;
}

// Grow blocks[] capacity to at least want. Returns 0 / -1 on OOM.
static inline int paged_table_reserve(PagedBlockTable *t, int want) {
    if (t->cap >= want) return 0;
    int ncap = t->cap ? t->cap * 2 : 4;
    while (ncap < want) ncap *= 2;
    int *nb = (int *)realloc(t->blocks, (size_t)ncap * sizeof(int));
    if (!nb) return -1;
    t->blocks = nb; t->cap = ncap;
    return 0;
}

// Ensure the table maps at least `n_tokens` total logical positions, pulling
// blocks from the pool as needed. Returns 0 on success, -1 if the pool is
// exhausted (caller must evict/backpressure). On -1 the table is left in a
// valid (partially-grown) state and nothing is leaked.
static inline int paged_table_ensure(PagedBlockPool *p, PagedBlockTable *t, int n_tokens) {
    if (!p || !t || n_tokens < 0) return -1;
    // total blocks needed to cover logical tokens [base, n_tokens)
    int need_end_block = (n_tokens + p->block_tokens - 1) / p->block_tokens; // ceil
    int base_block = t->base / p->block_tokens;
    int need = need_end_block - base_block;     // blocks from base_block onward
    if (need <= t->n) return 0;
    if (paged_table_reserve(t, need) != 0) return -1;
    while (t->n < need) {
        int b = paged_pool_alloc(p);
        if (b < 0) return -1;                   // pool exhausted
        t->blocks[t->n++] = b;
    }
    return 0;
}

// Release every block in the table back to the pool (sequence finished/evicted).
static inline void paged_table_release(PagedBlockPool *p, PagedBlockTable *t) {
    if (!p || !t) return;
    for (int i = 0; i < t->n; i++) paged_pool_free(p, t->blocks[i]);
    t->n = 0; t->base = 0;
}

// Resolve a logical token position to (physical_block_id, offset_in_block).
// Returns -1 if the position is not currently mapped (recycled or not allocated).
static inline int paged_table_lookup(const PagedBlockPool *p, const PagedBlockTable *t,
                                     int logical_pos, int *block_id_out, int *offset_out) {
    if (!p || !t || logical_pos < t->base) return -1;
    int rel_block = logical_pos / p->block_tokens - t->base / p->block_tokens;
    if (rel_block < 0 || rel_block >= t->n) return -1;
    if (block_id_out) *block_id_out = t->blocks[rel_block];
    if (offset_out)   *offset_out   = logical_pos % p->block_tokens;
    return 0;
}

// Sliding-window recycle: after a sequence reaches live_len tokens and only the
// last `window` are needed, free any leading blocks that fall entirely below the
// live window, advancing base. Returns the number of blocks recycled. This keeps
// a sliding sequence's footprint bounded by ceil(window/block_tokens)+1 blocks.
static inline int paged_table_advance_sliding(PagedBlockPool *p, PagedBlockTable *t,
                                              int live_len, int window) {
    if (!p || !t || live_len <= 0) return 0;
    int keep_from = live_len - window;          // first logical pos still needed
    if (keep_from <= 0) return 0;
    int keep_base_block = keep_from / p->block_tokens;   // block holding keep_from
    int cur_base_block  = t->base / p->block_tokens;
    int recycled = 0;
    while (cur_base_block < keep_base_block && t->n > 0) {
        paged_pool_free(p, t->blocks[0]);
        memmove(&t->blocks[0], &t->blocks[1], (size_t)(t->n - 1) * sizeof(int));
        t->n--;
        cur_base_block++;
        recycled++;
    }
    t->base = cur_base_block * p->block_tokens;
    return recycled;
}

#ifdef __cplusplus
}
#endif

#endif // FUCINA_PAGED_KV_H
