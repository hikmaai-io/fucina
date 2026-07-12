// ABOUTME: Host unit test for the S2b graph-key helpers (no CUDA/GPU required).
// ABOUTME: Verifies key construction, dominance dispatch, and decode-first batch ordering.
#include "qwen35_graph_key.cuh"
#include <cstdio>
#include <cstring>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); failures++; } \
} while (0)

static bool key_eq(q35_graph_key a, q35_graph_key b) {
    return a.num_tokens == b.num_tokens && a.num_reqs == b.num_reqs &&
           a.uniform_token_count == b.uniform_token_count;
}

int main() {
    // ── decode key: B rows == B reqs == 1 token each ──
    for (int B = 1; B <= 32; B++) {
        q35_graph_key k = q35_make_decode_key(B);
        CHECK(k.num_tokens == B && k.num_reqs == B && k.uniform_token_count == 1,
              "decode key triple");
    }

    // ── spec key: R reqs of (1+K) tokens ──
    {
        q35_graph_key k = q35_make_spec_key(4, 5); // 4 reqs, 1+4 tokens each
        CHECK(k.num_tokens == 20 && k.num_reqs == 4 && k.uniform_token_count == 5,
              "spec key triple");
    }

    // ── dominance: exact self-match always dominates ──
    for (int B = 1; B <= 32; B++) {
        q35_graph_key k = q35_make_decode_key(B);
        CHECK(q35_graph_dominates(k, k), "self-dominance");
    }

    // ── dominance: uniform_token_count must match exactly ──
    {
        q35_graph_key dec = q35_make_decode_key(8);       // utc=1
        q35_graph_key spec = q35_make_spec_key(8, 1);     // same triple as decode(8)? no: 8*1
        CHECK(key_eq(dec, spec), "decode(8) == spec(8,1) triple");
        q35_graph_key spec2 = q35_make_spec_key(4, 2);    // utc=2, 8 tokens, 4 reqs
        CHECK(!q35_graph_dominates(dec, spec2), "utc mismatch blocks dominance (1 vs 2)");
        CHECK(!q35_graph_dominates(spec2, dec), "utc mismatch blocks dominance (2 vs 1)");
    }

    // ── dominance: larger capture covers smaller runtime shape at same utc ──
    {
        q35_graph_key big = q35_make_decode_key(16);
        q35_graph_key small = q35_make_decode_key(8);
        CHECK(q35_graph_dominates(big, small), "16 dominates 8 (same utc)");
        CHECK(!q35_graph_dominates(small, big), "8 does not dominate 16");
    }

    // ── EXACT-match dispatch (the decode path): a larger graph must NOT serve a smaller batch ──
    // Regression lock for the S2b perf bug: dominance let a 31-row graph replay a 4-row decode
    // step (fucina runs num_tokens REAL rows, no input padding), ~8× waste. Decode dispatch is
    // exact-match; only a padded future S1 path may use dominance.
    {
        q35_graph_key big = q35_make_decode_key(31);
        q35_graph_key small = q35_make_decode_key(4);
        CHECK(!q35_graph_exact_match(big, small), "31-row graph must NOT match a 4-row step");
        CHECK(!q35_graph_exact_match(small, big), "4-row graph must NOT match a 31-row step");
        CHECK(q35_graph_exact_match(big, big), "exact-match is reflexive");
        for (int B = 1; B <= 32; B++) {
            q35_graph_key k = q35_make_decode_key(B);
            CHECK(q35_graph_exact_match(k, k), "decode key exact self-match");
            if (B < 32) CHECK(!q35_graph_exact_match(q35_make_decode_key(B + 1), k),
                              "B+1 graph must not serve B step");
        }
        // exact-match must also separate a decode key from a spec key of equal num_tokens.
        q35_graph_key dec8 = q35_make_decode_key(8);     // (8,8,1)
        q35_graph_key spec = q35_make_spec_key(4, 2);    // (8,4,2)
        CHECK(!q35_graph_exact_match(dec8, spec), "decode(8) must not match spec(4x2) at equal tokens");
    }

    // ── decode-first ordering: pure 1-token decode is the identity permutation ──
    {
        int qlen[8]; for (int i = 0; i < 8; i++) qlen[i] = 1;
        int out[8];
        q35_sort_batch_decode_first(qlen, 8, out);
        for (int i = 0; i < 8; i++) CHECK(out[i] == i, "uniform decode order is identity");
    }

    // ── decode-first ordering: mixed step puts decode (qlen 1) before extend/prefill ──
    {
        // rows: [prefill=64, decode=1, extend=4, decode=1, prefill=32]
        int qlen[5] = { 64, 1, 4, 1, 32 };
        int out[5];
        q35_sort_batch_decode_first(qlen, 5, out);
        // expected stable order by ascending qlen: decodes first in original order (rows 1,3),
        // then extend (row 2, qlen 4), then prefills by qlen (row 4 = 32 before row 0 = 64).
        int expect[5] = { 1, 3, 2, 4, 0 };
        for (int i = 0; i < 5; i++) CHECK(out[i] == expect[i], "mixed-step decode-first order");
        // and the leading run is a contiguous block of the smallest qlen (uniform decodes lead).
        CHECK(qlen[out[0]] == 1 && qlen[out[1]] == 1, "leading uniform decode run");
    }

    // ── decode-first ordering: stability among equal qlen preserves input order ──
    {
        int qlen[4] = { 1, 1, 1, 1 };
        int out[4];
        q35_sort_batch_decode_first(qlen, 4, out);
        int expect[4] = { 0, 1, 2, 3 };
        CHECK(memcmp(out, expect, sizeof(out)) == 0, "stable equal-qlen order");
    }

    if (failures == 0) { printf("PASS — S2b graph-key helpers\n"); return 0; }
    fprintf(stderr, "FAIL — %d S2b graph-key check(s)\n", failures);
    return 1;
}
