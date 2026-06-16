// Host unit test for paged_kv.h — no GPU required. Build: see Makefile target
// `paged-kv-test`, or: g++ -std=c++17 -O2 cuda/paged_kv_test.cc -o /tmp/pkt && /tmp/pkt
#include "paged_kv.h"
#include <cstdio>
#include <cassert>

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); failures++; } } while (0)

static void test_pool_alloc_free_exhaust() {
    PagedBlockPool p;
    CHECK(paged_pool_init(&p, 4, 256) == 0);
    CHECK(paged_pool_free_blocks(&p) == 4);
    int a = paged_pool_alloc(&p);   // expect 0 (reverse-pushed)
    int b = paged_pool_alloc(&p);
    int c = paged_pool_alloc(&p);
    int d = paged_pool_alloc(&p);
    CHECK(a == 0 && b == 1 && c == 2 && d == 3);
    CHECK(paged_pool_free_blocks(&p) == 0);
    CHECK(paged_pool_alloc(&p) == -1);            // exhausted
    paged_pool_free(&p, b);
    CHECK(paged_pool_free_blocks(&p) == 1);
    CHECK(paged_pool_alloc(&p) == 1);             // got b back
    paged_pool_free(&p, 0); paged_pool_free(&p, 1);
    paged_pool_free(&p, 2); paged_pool_free(&p, 3);
    paged_pool_free(&p, 3);                       // stray free past full — ignored
    CHECK(paged_pool_free_blocks(&p) == 4);
    paged_pool_destroy(&p);
}

static void test_budget() {
    // 10 GiB free, hold back 1 GiB, 2 MiB blocks -> floor(9GiB/2MiB)
    uint64_t GiB = 1ull << 30, MiB = 1ull << 20;
    int n = paged_blocks_for_budget(10 * GiB, 1 * GiB, 2 * MiB);
    CHECK(n == (int)((9 * GiB) / (2 * MiB)));     // 4608
    CHECK(paged_blocks_for_budget(1 * GiB, 2 * GiB, MiB) == 0);  // reserve > free
    CHECK(paged_blocks_for_budget(100, 0, 0) == 0);             // bad block size
}

static void test_table_growth_and_lookup() {
    PagedBlockPool p; paged_pool_init(&p, 8, 256);
    PagedBlockTable t; paged_table_init(&t);
    // need 600 tokens -> ceil(600/256)=3 blocks
    CHECK(paged_table_ensure(&p, &t, 600) == 0);
    CHECK(t.n == 3);
    CHECK(paged_pool_free_blocks(&p) == 5);
    // idempotent: ensuring fewer tokens allocates nothing more
    CHECK(paged_table_ensure(&p, &t, 100) == 0);
    CHECK(t.n == 3);
    // lookups
    int blk, off;
    CHECK(paged_table_lookup(&p, &t, 0, &blk, &off) == 0 && off == 0);
    CHECK(paged_table_lookup(&p, &t, 257, &blk, &off) == 0 && off == 1);  // block 1
    CHECK(paged_table_lookup(&p, &t, 599, &blk, &off) == 0 && off == 599 % 256);
    CHECK(paged_table_lookup(&p, &t, 9999, &blk, &off) == -1);            // unmapped
    paged_table_release(&p, &t);
    CHECK(paged_pool_free_blocks(&p) == 8);
    paged_table_free_struct(&t);
    paged_pool_destroy(&p);
}

static void test_pool_exhaustion_on_ensure() {
    PagedBlockPool p; paged_pool_init(&p, 2, 256);
    PagedBlockTable t; paged_table_init(&t);
    // want 1000 tokens -> 4 blocks, pool only has 2 -> -1
    CHECK(paged_table_ensure(&p, &t, 1000) == -1);
    CHECK(paged_pool_free_blocks(&p) == 0);   // both consumed, no leak
    paged_table_release(&p, &t);
    CHECK(paged_pool_free_blocks(&p) == 2);
    paged_table_free_struct(&t);
    paged_pool_destroy(&p);
}

static void test_sliding_recycle_bounds() {
    const int BT = 256, WINDOW = 1024;
    PagedBlockPool p; paged_pool_init(&p, 64, BT);
    PagedBlockTable t; paged_table_init(&t);
    int max_held = 0;
    // simulate a 10k-token sliding sequence, advancing the window each step-ish
    for (int len = 1; len <= 10000; len += 137) {
        CHECK(paged_table_ensure(&p, &t, len) == 0);
        paged_table_advance_sliding(&p, &t, len, WINDOW);
        if (t.n > max_held) max_held = t.n;
        // every still-needed position must resolve
        int lo = len - WINDOW; if (lo < 0) lo = 0;
        int blk, off;
        CHECK(paged_table_lookup(&p, &t, len - 1, &blk, &off) == 0);   // newest
        CHECK(paged_table_lookup(&p, &t, lo, &blk, &off) == 0);        // window tail
    }
    // bounded footprint: ceil(window/bt)+1 = 4+1 = 5 blocks, never the full 40+
    CHECK(max_held <= (WINDOW + BT - 1) / BT + 1);
    printf("  sliding max blocks held = %d (bound %d)\n",
           max_held, (WINDOW + BT - 1) / BT + 1);
    paged_table_release(&p, &t);
    paged_table_free_struct(&t);
    paged_pool_destroy(&p);
}

int main() {
    test_pool_alloc_free_exhaust();
    test_budget();
    test_table_growth_and_lookup();
    test_pool_exhaustion_on_ensure();
    test_sliding_recycle_bounds();
    if (failures == 0) { printf("paged_kv: ALL TESTS PASSED\n"); return 0; }
    printf("paged_kv: %d FAILURE(S)\n", failures);
    return 1;
}
