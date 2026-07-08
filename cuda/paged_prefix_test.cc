// paged_prefix_test.cc — host unit tests for the cross-request prefix cache.
//
// Pure CPU, no GPU: validates the radix tree, the chained-hash + exact-token
// memcmp guard, reference counting, the FREE/IN-USE/EVICTABLE three-state
// conservation invariant, leaf-only LRU eviction, and no double-free.
//
//   g++ -std=c++17 -O2 -Wall -Wextra cuda/paged_prefix_test.cc -o /tmp/x && /tmp/x

#include "paged_prefix.h"
#include <vector>
#include <cstdio>
#include <cassert>
#include <cstdint>

static int g_checks = 0;
#define CHECK(c) do { g_checks++; if (!(c)) { \
    std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)

// Conservation: free_top + #{refcount>0} + #{refcount==0 && registered} == n_blocks,
// and the registered-but-unreferenced count equals the LRU length counter.
static void conservation(const PrefixTree *c, const PagedBlockPool *p,
                         int *inuse_out, int *evict_out) {
    int inuse = 0, evict = 0;
    for (int b = 0; b < c->n_blocks; b++) {
        if (c->refcount[b] > 0) inuse++;
        else if (c->blk_node[b]) evict++;
    }
    assert(evict == c->n_evictable);
    assert(p->free_top + inuse + evict == c->n_blocks);
    if (inuse_out) *inuse_out = inuse;
    if (evict_out) *evict_out = evict;
}

// A host-side stand-in for the engine's seq_add: adopt the cached prefix, alloc
// private blocks for the rest, register the full blocks. Returns the seq's block
// list (full blocks then the partial tail, if any).
struct Seq { std::vector<int> blocks; std::vector<int32_t> toks; };

static Seq add_seq(PrefixTree *c, PagedBlockPool *pool, const std::vector<int32_t> &toks) {
    int BT = c->block_tokens;
    int full  = (int)toks.size() / BT;
    int total = ((int)toks.size() + BT - 1) / BT;   // includes a partial tail block
    std::vector<int> shared(full > 0 ? full : 1);
    int ns = prefix_lookup(c, toks.data(), (int)toks.size(), shared.data(), full);
    std::vector<int> blocks(shared.begin(), shared.begin() + ns);
    for (int i = ns; i < total; i++) {
        int b = prefix_alloc(c, pool);
        assert(b >= 0);
        blocks.push_back(b);
    }
    prefix_register(c, toks.data(), full, blocks.data());
    return {blocks, toks};
}

static void remove_seq(PrefixTree *c, PagedBlockPool *pool, const Seq &s) {
    prefix_release(c, pool, s.blocks.data(), (int)s.blocks.size());
}

static std::vector<int32_t> mk(std::initializer_list<int> v) {
    return std::vector<int32_t>(v.begin(), v.end());
}

int main() {
    const int BT = 4;

    // ── Test 1: basic cross-request reuse + refcount ──────────────────────────
    {
        PagedBlockPool pool; PrefixTree c;
        assert(paged_pool_init(&pool, 32, BT) == 0);
        assert(prefix_tree_init(&c, &pool, 64) == 0);

        // prompt = 2 full blocks (8 tok) + 1 tail token
        auto P = mk({1,2,3,4, 5,6,7,8, 9});
        Seq A = add_seq(&c, &pool, P);
        CHECK(c.cached_blocks == 2);          // two full blocks registered
        CHECK(c.refcount[A.blocks[0]] == 1);
        CHECK(c.refcount[A.blocks[1]] == 1);
        CHECK(c.blk_node[A.blocks[2]] == NULL);   // tail private, unregistered
        conservation(&c, &pool, nullptr, nullptr);

        // second identical request reuses both full blocks
        uint64_t hb0 = c.hit_blocks;
        Seq B = add_seq(&c, &pool, P);
        CHECK(c.hit_blocks - hb0 == 2);       // adopted 2 cached blocks
        CHECK(B.blocks[0] == A.blocks[0]);    // same physical blocks
        CHECK(B.blocks[1] == A.blocks[1]);
        CHECK(B.blocks[2] != A.blocks[2]);    // distinct private tail
        CHECK(c.refcount[A.blocks[0]] == 2);  // shared by A and B
        CHECK(c.refcount[A.blocks[1]] == 2);
        conservation(&c, &pool, nullptr, nullptr);

        // releasing A keeps the blocks cached (B still holds them)
        remove_seq(&c, &pool, A);
        CHECK(c.refcount[B.blocks[0]] == 1);
        CHECK(c.cached_blocks == 2);
        CHECK(c.n_evictable == 0);            // none dropped to 0 yet
        conservation(&c, &pool, nullptr, nullptr);

        // releasing B drops the shared blocks to refcount 0 -> EVICTABLE (still cached)
        remove_seq(&c, &pool, B);
        CHECK(c.cached_blocks == 2);
        CHECK(c.n_evictable == 2);
        conservation(&c, &pool, nullptr, nullptr);

        // a third identical request re-hits the now-evictable cached blocks
        uint64_t hb1 = c.hit_blocks;
        Seq D = add_seq(&c, &pool, P);
        CHECK(c.hit_blocks - hb1 == 2);
        CHECK(c.n_evictable == 0);            // re-adopted off the LRU
        remove_seq(&c, &pool, D);
        prefix_tree_destroy(&c); paged_pool_destroy(&pool);
    }

    // ── Test 2: divergent prefix shares only the common chain ─────────────────
    {
        PagedBlockPool pool; PrefixTree c;
        assert(paged_pool_init(&pool, 32, BT) == 0);
        assert(prefix_tree_init(&c, &pool, 64) == 0);

        Seq A = add_seq(&c, &pool, mk({1,2,3,4, 5,6,7,8, 0}));   // blocks {1234},{5678}
        // shares block0 (1,2,3,4) but block1 differs -> only 1 shared
        std::vector<int> out(8);
        int ns = prefix_lookup(&c, std::vector<int32_t>(mk({1,2,3,4, 9,9,9,9})).data(), 8, out.data(), 8);
        CHECK(ns == 1);
        CHECK(out[0] == A.blocks[0]);
        // undo the adopt from the bare lookup above
        prefix_release(&c, &pool, out.data(), ns);

        // a completely different prompt shares nothing
        int ns2 = prefix_lookup(&c, std::vector<int32_t>(mk({7,7,7,7})).data(), 4, out.data(), 8);
        CHECK(ns2 == 0);
        conservation(&c, &pool, nullptr, nullptr);
        remove_seq(&c, &pool, A);
        prefix_tree_destroy(&c); paged_pool_destroy(&pool);
    }

    // ── Test 3: memcmp guard — same hash, different tokens must NOT match ──────
    {
        PagedBlockPool pool; PrefixTree c;
        assert(paged_pool_init(&pool, 8, BT) == 0);
        assert(prefix_tree_init(&c, &pool, 16) == 0);

        // Register one block, then probe the map with a DIFFERENT token block that
        // we force to the producer's exact hash value: the memcmp must reject it.
        auto T = mk({4,3,2,1});
        Seq A = add_seq(&c, &pool, T);                 // one full block node
        PrefixNode *node = c.blk_node[A.blocks[0]];
        CHECK(node != NULL);
        auto T2 = mk({9,9,9,9});
        CHECK(prefix_map_find(&c, node->hash, T2.data(), BT) == NULL);  // collision rejected
        CHECK(prefix_map_find(&c, node->hash, T.data(),  BT) == node);  // exact match accepted
        remove_seq(&c, &pool, A);
        prefix_tree_destroy(&c); paged_pool_destroy(&pool);
    }

    // ── Test 4: alloc prefers free_stack, then evicts the LRU-tail leaf ────────
    {
        PagedBlockPool pool; PrefixTree c;
        assert(paged_pool_init(&pool, 6, BT) == 0);     // tiny pool: 6 blocks
        assert(prefix_tree_init(&c, &pool, 32) == 0);

        // Fill the pool with 6 distinct SINGLE-block cached prompts, all released
        // (so all 6 blocks become EVICTABLE leaves on the LRU).
        std::vector<Seq> seqs;
        for (int i = 0; i < 6; i++) {
            auto P = mk({100+i, 200+i, 300+i, 400+i});  // distinct full block
            seqs.push_back(add_seq(&c, &pool, P));
        }
        CHECK(pool.free_top == 0);
        CHECK(c.cached_blocks == 6);
        for (auto &s : seqs) remove_seq(&c, &pool, s);
        CHECK(c.n_evictable == 6);
        conservation(&c, &pool, nullptr, nullptr);

        // A new distinct prompt must evict an LRU leaf (free_stack is empty).
        uint64_t ev0 = c.evictions;
        Seq N = add_seq(&c, &pool, mk({7,7,7,7}));
        CHECK(c.evictions - ev0 == 1);                  // exactly one block reclaimed
        CHECK(c.refcount[N.blocks[0]] == 1);
        conservation(&c, &pool, nullptr, nullptr);
        remove_seq(&c, &pool, N);
        prefix_tree_destroy(&c); paged_pool_destroy(&pool);
    }

    // ── Test 5: leaf-only eviction never orphans a live child ─────────────────
    {
        PagedBlockPool pool; PrefixTree c;
        assert(paged_pool_init(&pool, 4, BT) == 0);
        assert(prefix_tree_init(&c, &pool, 16) == 0);

        // One 2-block chain held LIVE (parent block0 has a cached child block1).
        Seq live = add_seq(&c, &pool, mk({1,2,3,4, 5,6,7,8}));   // uses 2 blocks, kept
        // Two more blocks: one evictable leaf, one evictable.
        Seq tmp = add_seq(&c, &pool, mk({9,9,9,9}));             // 1 block
        remove_seq(&c, &pool, tmp);                              // block -> evictable leaf
        CHECK(pool.free_top == 1);                               // 4 - 3 used = 1 free
        conservation(&c, &pool, nullptr, nullptr);

        // Allocate until we force eviction; the parent of the LIVE chain (block0,
        // n_children==1) must NOT be evicted — only the evictable leaf may go.
        int b1 = prefix_alloc(&c, &pool);   // takes the 1 free block
        CHECK(b1 >= 0);
        int b2 = prefix_alloc(&c, &pool);   // must evict the {9,9,9,9} leaf, not block0
        CHECK(b2 >= 0);
        CHECK(c.refcount[live.blocks[0]] == 1);     // live parent untouched
        CHECK(c.blk_node[live.blocks[0]] != NULL);
        conservation(&c, &pool, nullptr, nullptr);
        // pool now exhausted with no evictable leaf -> alloc fails gracefully
        int b3 = prefix_alloc(&c, &pool);
        CHECK(b3 == -1);
        prefix_tree_destroy(&c); paged_pool_destroy(&pool);
    }

    // ── Test 6: randomized conservation + no double-free ──────────────────────
    {
        PagedBlockPool pool; PrefixTree c;
        assert(paged_pool_init(&pool, 24, BT) == 0);
        assert(prefix_tree_init(&c, &pool, 128) == 0);

        // A small corpus of overlapping prompts (shared prefixes are common).
        std::vector<std::vector<int32_t>> corpus = {
            mk({1,2,3,4, 5,6,7,8, 9,9,9,9}),
            mk({1,2,3,4, 5,6,7,8, 7,7,7,7}),   // shares 2 blocks with [0]
            mk({1,2,3,4, 0,0,0,0}),            // shares 1 block with [0]
            mk({2,2,2,2, 3,3,3,3}),
            mk({1,2,3,4, 5,6,7,8}),            // shares 2 blocks with [0]
        };
        std::vector<Seq> live;
        uint32_t rng = 0x1234567u;
        auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
        for (int step = 0; step < 4000; step++) {
            bool do_add = (next() & 1) || live.empty();
            if (do_add && live.size() < 8) {
                const auto &P = corpus[next() % corpus.size()];
                // only add if the pool can plausibly hold it (else skip — engine backpressures)
                int need = ((int)P.size() + BT - 1) / BT;
                if (pool.free_top + c.n_evictable >= need) live.push_back(add_seq(&c, &pool, P));
            } else if (!live.empty()) {
                int i = next() % live.size();
                remove_seq(&c, &pool, live[i]);
                live.erase(live.begin() + i);
            }
            conservation(&c, &pool, nullptr, nullptr);
        }
        // drain
        for (auto &s : live) remove_seq(&c, &pool, s);
        live.clear();
        int inuse = 0, evict = 0;
        conservation(&c, &pool, &inuse, &evict);
        CHECK(inuse == 0);                       // every reference released
        CHECK(pool.free_top + evict == 24);      // all blocks free or cached-evictable
        prefix_tree_destroy(&c); paged_pool_destroy(&pool);
    }

    std::printf("PASS — paged_prefix: %d checks across 6 tests\n", g_checks);
    return 0;
}
