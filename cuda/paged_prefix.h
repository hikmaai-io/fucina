// paged_prefix.h — host-side cross-request prefix cache (RadixAttention-style).
//
// Implements automatic KV-cache reuse across requests that share a token prefix
// (SGLang RadixAttention, Zheng et al. 2024), layered over the paged_kv.h
// PagedBlockPool free-list. Like paged_kv.h this header is PURE INTEGER LOGIC
// with NO CUDA dependency — the radix tree, the 64-bit chained hash, reference
// counting and LRU eviction are all host-unit-testable on the CPU. The engine
// owns one PrefixTree over the GLOBAL block pool and indexes the same device
// storage by the physical block ids this tree hands out.
//
// Why this is lossless: paged attention reads K/V for a logical position through
// the per-sequence block table (block_table indirection), so a sequence can read
// another sequence's already-filled physical block at the SAME logical offset and
// get byte-identical attention. We share ONLY complete, immutable 256-token
// blocks of a matched prefix (the partial tail block is always private), so a
// shared block is never written after it is registered — no copy-on-write needed.
//
// Three-state partition: every physical block of the pool is in exactly one of:
//   FREE       — on PagedBlockPool.free_stack, refcount 0, blk_node NULL.
//   IN-USE     — refcount > 0 (held by >=1 live sequence). May be registered
//                (blk_node != NULL, shareable) or private (blk_node NULL, e.g. a
//                tail block that has not filled yet).
//   EVICTABLE  — refcount 0, registered (blk_node != NULL), on the LRU list:
//                still re-hittable, reclaimable on demand.
// Conservation invariant (asserted in tests):
//   free_top + #{refcount>0} + #{refcount==0 && registered} == n_blocks.
//
// Thread-safety: none. The engine mutates the tree on its single owning thread
// (the same one that owns the pools), under the bridge engine mutex.

#ifndef FUCINA_PAGED_PREFIX_H
#define FUCINA_PAGED_PREFIX_H

#include "paged_kv.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PREFIX_HASH_SEED 0xcbf29ce484222325ULL   // FNV-1a 64-bit offset basis
#define PREFIX_HASH_PRIME 0x100000001b3ULL

// Chained FNV-1a: fold one block's `n` token ids into the parent block's hash so
// a block's key encodes its ENTIRE prefix [0,(depth+1)*block_tokens). Two
// sequences that share a later block but differ earlier therefore hash apart.
static inline uint64_t prefix_block_hash(uint64_t parent, const int32_t *toks, int n) {
    uint64_t h = parent;
    for (int i = 0; i < n; i++) {
        uint32_t t = (uint32_t)toks[i];
        for (int b = 0; b < 4; b++) {
            h ^= (uint64_t)((t >> (b * 8)) & 0xffu);
            h *= PREFIX_HASH_PRIME;
        }
    }
    return h;
}

// One radix-tree node == one cached, fully-written physical block.
typedef struct PrefixNode {
    uint64_t            hash;       // chained hash of tokens[0,(depth+1)*BT)
    int32_t            *tokens;     // BT token ids in THIS block (exact memcmp guard)
    int                 block_id;   // physical GLOBAL pool block id this node owns
    int                 depth;      // 0 = first block of the sequence
    int                 n_children; // cached children (must be 0 to evict — leaf-only)
    struct PrefixNode  *parent;
    struct PrefixNode  *hnext;      // hash-bucket singly-linked chain
    struct PrefixNode  *lru_prev;   // EVICTABLE doubly-linked list (only when refcount==0)
    struct PrefixNode  *lru_next;
    int                 on_lru;     // 1 iff currently linked on the LRU list
} PrefixNode;

typedef struct {
    int           block_tokens;     // == pool->block_tokens (256)
    int           n_blocks;         // == pool->n_blocks
    int           nbuckets;
    PrefixNode  **buckets;          // hash -> node chain
    int          *refcount;         // [n_blocks] per physical block id
    PrefixNode  **blk_node;         // [n_blocks] back-pointer (NULL = private/free)
    PrefixNode   *lru_head;         // MRU
    PrefixNode   *lru_tail;         // LRU (evicted first)
    int           n_evictable;      // #{refcount==0 && registered} (== LRU length)
    // stats (observability only)
    uint64_t      lookups;          // prefix_lookup calls
    uint64_t      hit_blocks;       // blocks adopted from cache (prefill saved)
    uint64_t      cached_blocks;    // nodes currently in the tree
    uint64_t      evictions;        // LRU evictions
} PrefixTree;

// ── lifecycle ────────────────────────────────────────────────────────────────
static inline int prefix_tree_init(PrefixTree *c, const PagedBlockPool *pool, int nbuckets) {
    if (!c || !pool || pool->n_blocks < 0) return -1;
    memset(c, 0, sizeof(*c));
    c->block_tokens = pool->block_tokens;
    c->n_blocks     = pool->n_blocks;
    if (nbuckets < 16) nbuckets = 16;
    c->nbuckets = nbuckets;
    c->buckets  = (PrefixNode **)calloc((size_t)nbuckets, sizeof(PrefixNode *));
    c->refcount = (int *)calloc((size_t)(pool->n_blocks > 0 ? pool->n_blocks : 1), sizeof(int));
    c->blk_node = (PrefixNode **)calloc((size_t)(pool->n_blocks > 0 ? pool->n_blocks : 1), sizeof(PrefixNode *));
    if (!c->buckets || !c->refcount || !c->blk_node) {
        free(c->buckets); free(c->refcount); free(c->blk_node);
        memset(c, 0, sizeof(*c));
        return -1;
    }
    return 0;
}

static inline void prefix_tree_destroy(PrefixTree *c) {
    if (!c) return;
    for (int i = 0; i < c->nbuckets; i++) {
        PrefixNode *n = c->buckets ? c->buckets[i] : NULL;
        while (n) { PrefixNode *nx = n->hnext; free(n->tokens); free(n); n = nx; }
    }
    free(c->buckets); free(c->refcount); free(c->blk_node);
    memset(c, 0, sizeof(*c));
}

// ── internal helpers ─────────────────────────────────────────────────────────
static inline void prefix_lru_unlink(PrefixTree *c, PrefixNode *n) {
    if (!n->on_lru) return;
    if (n->lru_prev) n->lru_prev->lru_next = n->lru_next; else c->lru_head = n->lru_next;
    if (n->lru_next) n->lru_next->lru_prev = n->lru_prev; else c->lru_tail = n->lru_prev;
    n->lru_prev = n->lru_next = NULL;
    n->on_lru = 0;
    c->n_evictable--;
}

// Push to MRU head (a block that just dropped to refcount 0 but stays cached).
static inline void prefix_lru_push_head(PrefixTree *c, PrefixNode *n) {
    n->lru_prev = NULL;
    n->lru_next = c->lru_head;
    if (c->lru_head) c->lru_head->lru_prev = n; else c->lru_tail = n;
    c->lru_head = n;
    n->on_lru = 1;
    c->n_evictable++;
}

static inline PrefixNode *prefix_map_find(const PrefixTree *c, uint64_t hash,
                                          const int32_t *toks, int n) {
    if (c->nbuckets <= 0) return NULL;
    PrefixNode *n0 = c->buckets[hash % (uint64_t)c->nbuckets];
    for (PrefixNode *p = n0; p; p = p->hnext) {
        if (p->hash == hash && memcmp(p->tokens, toks, (size_t)n * sizeof(int32_t)) == 0)
            return p;     // exact: hash AND full token content match
    }
    return NULL;
}

static inline void prefix_map_insert(PrefixTree *c, PrefixNode *n) {
    uint64_t b = n->hash % (uint64_t)c->nbuckets;
    n->hnext = c->buckets[b];
    c->buckets[b] = n;
}

static inline void prefix_map_remove(PrefixTree *c, PrefixNode *n) {
    uint64_t b = n->hash % (uint64_t)c->nbuckets;
    PrefixNode **pp = &c->buckets[b];
    while (*pp && *pp != n) pp = &(*pp)->hnext;
    if (*pp == n) *pp = n->hnext;
}

// Detach a leaf node from the tree, return its physical block to the caller.
// Caller must ensure n->n_children == 0 and n->on_lru (it is being evicted).
static inline void prefix_evict_node(PrefixTree *c, PrefixNode *n) {
    prefix_lru_unlink(c, n);
    prefix_map_remove(c, n);
    if (n->parent) n->parent->n_children--;
    c->blk_node[n->block_id] = NULL;
    c->cached_blocks--;
    c->evictions++;
    free(n->tokens);
    free(n);
}

// ── allocation ───────────────────────────────────────────────────────────────
// Allocate a unique physical block for private (unshared) use: take from the
// free_stack first (preserving LIFO locality); if the pool is empty, reclaim the
// LRU-tail LEAF (a cached but unreferenced block with no cached children). Sets
// refcount = 1, blk_node = NULL (private). Returns -1 only if the pool is truly
// exhausted (no free block AND nothing evictable).
static inline int prefix_alloc(PrefixTree *c, PagedBlockPool *pool) {
    int b = paged_pool_alloc(pool);
    if (b < 0) {
        // Evict the least-recently-used LEAF (walk tail->head past non-leaves).
        PrefixNode *n = c->lru_tail;
        while (n && n->n_children != 0) n = n->lru_prev;
        if (!n) return -1;                 // nothing reclaimable
        b = n->block_id;
        prefix_evict_node(c, n);           // returns block to "free" conceptually
    }
    c->refcount[b] = 1;
    c->blk_node[b] = NULL;
    return b;
}

// ── lookup / adopt ───────────────────────────────────────────────────────────
// Find the longest cached prefix of `tokens` at FULL-block granularity. Adopts
// each matched block (refcount++, unlinks from LRU if it was evictable), writes
// the adopted physical block ids into out_block_ids[0..ret), and returns the
// number of shared blocks. `max_blocks` bounds out_block_ids. The matched chain
// is verified by exact token memcmp at every block, so a hash collision can never
// adopt a wrong block.
static inline int prefix_lookup(PrefixTree *c, const int32_t *tokens, int n_tokens,
                                int *out_block_ids, int max_blocks) {
    c->lookups++;
    int BT = c->block_tokens;
    int full_blocks = n_tokens / BT;            // complete blocks available in prompt
    uint64_t h = PREFIX_HASH_SEED;
    int shared = 0;
    for (int i = 0; i < full_blocks && shared < max_blocks; i++) {
        h = prefix_block_hash(h, tokens + (size_t)i * BT, BT);
        PrefixNode *node = prefix_map_find(c, h, tokens + (size_t)i * BT, BT);
        if (!node) break;                       // first miss ends the shared chain
        // adopt
        if (node->on_lru) prefix_lru_unlink(c, node);
        c->refcount[node->block_id]++;
        out_block_ids[shared++] = node->block_id;
        c->hit_blocks++;
    }
    return shared;
}

// ── register ─────────────────────────────────────────────────────────────────
// Make the first `n_full_blocks` FULL, fully-written blocks of a sequence
// shareable. block_ids[i] is the physical block holding logical block i; tokens
// is the full token sequence (>= n_full_blocks*BT ids). Blocks already registered
// (blk_node set, i.e. the shared prefix) are skipped; newly-completed blocks get
// a node chained onto the prior block. Refcounts are NOT changed — the producing
// sequence keeps the references it already holds.
static inline void prefix_register(PrefixTree *c, const int32_t *tokens,
                                   int n_full_blocks, const int *block_ids) {
    int BT = c->block_tokens;
    uint64_t h = PREFIX_HASH_SEED;
    PrefixNode *parent = NULL;
    for (int i = 0; i < n_full_blocks; i++) {
        h = prefix_block_hash(h, tokens + (size_t)i * BT, BT);
        int bid = block_ids[i];
        PrefixNode *existing = c->blk_node[bid];
        if (existing) { parent = existing; continue; }   // already a node (shared prefix)
        // A node for this exact prefix may already exist on a DIFFERENT physical
        // block (rare; only if two cold producers raced before either registered).
        // Single-threaded engine makes this essentially impossible, but guard it:
        // leave this block private rather than create a duplicate-content node.
        PrefixNode *dup = prefix_map_find(c, h, tokens + (size_t)i * BT, BT);
        if (dup) { parent = dup; continue; }
        PrefixNode *node = (PrefixNode *)calloc(1, sizeof(PrefixNode));
        if (!node) return;                                // OOM: stop registering (block stays private)
        node->tokens = (int32_t *)malloc((size_t)BT * sizeof(int32_t));
        if (!node->tokens) { free(node); return; }
        memcpy(node->tokens, tokens + (size_t)i * BT, (size_t)BT * sizeof(int32_t));
        node->hash = h;
        node->block_id = bid;
        node->depth = i;
        node->parent = parent;
        if (parent) parent->n_children++;
        prefix_map_insert(c, node);
        c->blk_node[bid] = node;
        c->cached_blocks++;
        parent = node;
    }
}

// ── release ──────────────────────────────────────────────────────────────────
// Release `n` blocks held by a finished sequence. Each: refcount--; on reaching
// 0, a registered block becomes EVICTABLE (pushed to the LRU, stays cached for
// future hits), an unregistered (private/tail) block is returned to the pool.
static inline void prefix_release(PrefixTree *c, PagedBlockPool *pool,
                                  const int *block_ids, int n) {
    for (int i = 0; i < n; i++) {
        int bid = block_ids[i];
        if (bid < 0 || bid >= c->n_blocks) continue;
        if (c->refcount[bid] <= 0) continue;             // defensive: never go negative
        c->refcount[bid]--;
        if (c->refcount[bid] == 0) {
            if (c->blk_node[bid]) prefix_lru_push_head(c, c->blk_node[bid]);  // EVICTABLE
            else paged_pool_free(pool, bid);                                  // private -> free
        }
    }
}

// ── stats ────────────────────────────────────────────────────────────────────
static inline void prefix_tree_stats(const PrefixTree *c, uint64_t *lookups,
                                     uint64_t *hit_blocks, uint64_t *cached_blocks,
                                     uint64_t *evictions) {
    if (lookups)       *lookups       = c ? c->lookups : 0;
    if (hit_blocks)    *hit_blocks    = c ? c->hit_blocks : 0;
    if (cached_blocks) *cached_blocks = c ? c->cached_blocks : 0;
    if (evictions)     *evictions     = c ? c->evictions : 0;
}

#ifdef __cplusplus
}
#endif

#endif // FUCINA_PAGED_PREFIX_H
