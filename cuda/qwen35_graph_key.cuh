// ABOUTME: S2b CUDA-graph cache key for the Qwen3.5 decode/verify step.
// ABOUTME: Pure host-testable shape triple + dominance dispatch + decode-first batch ordering.
#pragma once

// A decode/verify step is identified by the shape triple
//   num_tokens          — total query rows across the batch
//   num_reqs            — number of distinct requests (slots) in the batch
//   uniform_token_count — query tokens PER request when uniform (1 for plain decode,
//                         1+K for a DFlash (1+K)-token spec-decode/verify batch), else 0
// For plain decode num_tokens == num_reqs and uniform_token_count == 1, so a row count B maps to
// exactly one key — bit-identical to the pre-S2b graph[B] scheme. The triple is what lets a FULL
// graph also serve a uniform multi-token batch the day S1/DFlash lands.
struct q35_graph_key {
    int num_tokens;
    int num_reqs;
    int uniform_token_count;
};

// Plain 1-token-per-request decode: every request contributes exactly one query row.
static inline q35_graph_key q35_make_decode_key(int B) {
    return q35_graph_key{ B, B, 1 };
}

// (1+K)-token spec-decode/verify batch: R requests, each contributing exactly 1+K query rows.
static inline q35_graph_key q35_make_spec_key(int num_reqs, int tokens_per_req) {
    return q35_graph_key{ num_reqs * tokens_per_req, num_reqs, tokens_per_req };
}

// Dominance dispatch: a captured graph with key `cap` can serve a runtime shape `want` when it
// covers it in every dimension. Captures are exact-shape until S2a pads device buffers to max
// shapes; then `>=` lets one larger capture serve smaller uniform batches. uniform_token_count
// must match exactly (it changes the per-request query layout, not just a row count).
static inline bool q35_graph_dominates(const q35_graph_key &cap, const q35_graph_key &want) {
    return cap.uniform_token_count == want.uniform_token_count &&
           cap.num_tokens >= want.num_tokens &&
           cap.num_reqs   >= want.num_reqs;
}

// Decode-first batch ordering. Fill out[] with a STABLE permutation of [0,B) ordering rows by
// ascending per-row query-token count: decode (1 token) leads, then short-extend, then prefill
// (largest). Uniform decodes thus form a leading contiguous run that maps to one graph key. Stable
// insertion by (qlen, original index) keeps equal-qlen rows in input order — for pure 1-token
// decode every qlen is 1, so out[] is the identity and the batch is bit-identical. C-style, no STL.
static inline void q35_sort_batch_decode_first(const int *qlen, int B, int *out) {
    for (int i = 0; i < B; i++) {
        int j = i;
        while (j > 0 && qlen[out[j - 1]] > qlen[i]) { out[j] = out[j - 1]; j--; }
        out[j] = i;
    }
}
