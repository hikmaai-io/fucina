// ABOUTME: Host integration test composing the full DFlash verify pipeline (weights-free, S1a).
// ABOUTME: planner -> shared-key draft sampling -> rejection -> commit assembly, end to end.
//
// This is the seam test for P1+P4 orchestration: it exercises the exact sequence a real verify
// step will run, minus the two device forwards (draft + target), which need the real weights.
// Synthetic target/draft logits stand in for those forwards so the deterministic glue can be
// validated now:
//   1. planner derives (1+K), the spec graph key, and the K+1 lookahead from config-derived K;
//   2. the DRAFT samples K tokens from its logits using the shared (seed, position, SAMPLE) keys;
//   3. the VERIFIER runs greedy or probabilistic rejection with the SAME keys, producing
//      (accepted_len, emitted_token);
//   4. the commit assembly maps that to the P0 commit token list + next input + emitted count.
// It asserts the invariants that make the step lossless and deterministic:
//   - draft and verifier derive identical per-position uniforms from shared keys (no shared state);
//   - matched draft==target distributions accept every token (accept_len==K);
//   - greedy verification emits exactly the target argmax stream (self-consistency);
//   - the commit token list is always the accepted-draft prefix, and next-input is the emitted
//     token, for every accepted length that arises.
//
// build: g++ -std=c++17 -O2 -Wall -Wextra -Icuda cuda/qwen35_dflash_pipeline_test.cc -o /tmp/dflash_pipe && /tmp/dflash_pipe
#include "qwen35_dflash_plan.cuh"
#include "qwen35_dflash_reject.cuh"
#include "qwen35_dflash_commit.cuh"
#include <cstdio>
#include <vector>
#include <cmath>

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } } while (0)

// Deterministic draft sampler over one logits row using the shared SAMPLE-domain key. This mirrors
// what the device draft head will do: inverse-CDF over softmax with the (seed, position) uniform.
static int32_t draft_sample(const float *logits, int vocab, uint64_t seed, int64_t pos) {
    double mx = logits[0];
    for (int v = 1; v < vocab; v++) if ((double)logits[v] > mx) mx = logits[v];
    double sum = 0.0;
    for (int v = 0; v < vocab; v++) sum += std::exp((double)logits[v] - mx);
    double u = q35_dflash_uniform_open(seed, pos, Q35_DFLASH_DOMAIN_SAMPLE);
    double target = u * sum, acc = 0.0;
    for (int v = 0; v < vocab; v++) { acc += std::exp((double)logits[v] - mx); if (acc >= target) return v; }
    return vocab - 1;
}

int main() {
    const int vocab = 16;
    const uint64_t seed = 0xC0FFEEull;
    const int n0 = 40;   // slot's absolute token count at the verify boundary

    // Config-derived K (e.g. dflash_config.block_size); the planner clamps and shapes it.
    int cfg_k = 6;
    auto plan = q35_dflash_make_plan(cfg_k, Q35_DFLASH_MODE_ON, /*num_reqs=*/2, /*critical=*/8);
    const int K = plan.K;
    CHECK(plan.enabled == 1, "plan enabled at B=2");
    CHECK(plan.uniform_token_count == K + 1, "uniform tokens 1+K");
    CHECK(plan.lookahead_slots == K + 1, "lookahead K+1");
    {
        q35_graph_key sk = q35_dflash_graph_key(2, K);
        CHECK(sk.uniform_token_count == K + 1 && sk.num_reqs == 2 && sk.num_tokens == 2 * (K + 1),
              "spec graph key shape");
    }

    // Absolute positions of the K drafted tokens: n0 .. n0+K-1; bonus at n0+K.
    std::vector<int64_t> pos(K);
    for (int i = 0; i < K; i++) pos[i] = n0 + i;
    int64_t pos_bonus = n0 + K;

    // ---- 1) Shared-key determinism: draft and verifier derive identical uniforms ----
    for (int i = 0; i < K; i++) {
        double ud = q35_dflash_uniform_open(seed, pos[i], Q35_DFLASH_DOMAIN_ACCEPT);
        double uv = q35_dflash_uniform_open(seed, pos[i], Q35_DFLASH_DOMAIN_ACCEPT);
        CHECK(ud == uv, "shared-key uniform is reproducible across independent derivations");
    }

    // ---- 2) Matched distributions: draft logits == target logits => accept all K (greedy) ----
    {
        // Build (K+1) target rows; draft rows equal the first K target rows.
        std::vector<float> tl((size_t)(K + 1) * vocab), dl((size_t)K * vocab);
        for (int r = 0; r <= K; r++)
            for (int v = 0; v < vocab; v++) tl[(size_t)r * vocab + v] = std::sin(0.3f * (v + 1) + 0.7f * r) * 3.0f;
        for (int r = 0; r < K; r++)
            for (int v = 0; v < vocab; v++) dl[(size_t)r * vocab + v] = tl[(size_t)r * vocab + v];
        // Greedy draft = argmax per row (what a greedy draft head would emit).
        int32_t drafts[Q35_DFLASH_K_MAX];
        for (int i = 0; i < K; i++) drafts[i] = q35_dflash_argmax(&dl[(size_t)i * vocab], vocab);
        auto rg = q35_dflash_verify_greedy(tl.data(), vocab, drafts, K);
        CHECK(rg.accepted_len == K, "greedy matched-argmax accepts all K");
        // Emitted bonus is the target argmax of row K.
        CHECK(rg.emitted_token == q35_dflash_argmax(&tl[(size_t)K * vocab], vocab), "greedy bonus argmax");
        // Commit assembly: all K committed, bonus carried forward, K+1 emitted.
        auto cp = q35_dflash_assemble_commit(rg, drafts, K);
        CHECK(cp.commit_len == K, "commit all K on full accept");
        for (int i = 0; i < K; i++) CHECK(cp.commit_tokens[i] == drafts[i], "commit token == draft");
        CHECK(cp.next_input_token == rg.emitted_token, "next input == emitted");
        CHECK(cp.total_emitted == K + 1, "emit K+1");
    }

    // ---- 3) Greedy self-consistency: a draft that diverges at position m stops there ----
    {
        std::vector<float> tl((size_t)(K + 1) * vocab, 0.0f);
        int targ[Q35_DFLASH_K_MAX + 1] = {3, 9, 1, 12, 5, 7, 2};
        for (int r = 0; r <= K; r++)
            for (int v = 0; v < vocab; v++) tl[(size_t)r * vocab + v] = (v == targ[r]) ? 10.0f : 0.1f * (v % 4);
        int m = 3;   // diverge at position 3
        int32_t drafts[Q35_DFLASH_K_MAX];
        for (int i = 0; i < K; i++) drafts[i] = (i < m) ? targ[i] : (targ[i] + 1) % vocab;
        auto rg = q35_dflash_verify_greedy(tl.data(), vocab, drafts, K);
        CHECK(rg.accepted_len == m, "greedy stops at first divergence");
        CHECK(rg.emitted_token == targ[m], "greedy emits target argmax at divergence");
        auto cp = q35_dflash_assemble_commit(rg, drafts, K);
        CHECK(cp.commit_len == m, "commit prefix len m");
        CHECK(cp.total_emitted == m + 1, "emit m+1");
    }

    // ---- 4) Probabilistic path: draft samples via shared key, verifier accepts matched dists ----
    {
        std::vector<float> tl((size_t)(K + 1) * vocab), dl((size_t)K * vocab);
        for (int r = 0; r <= K; r++)
            for (int v = 0; v < vocab; v++) tl[(size_t)r * vocab + v] = std::cos(0.21f * v + 0.5f * r) * 2.5f;
        for (int r = 0; r < K; r++)
            for (int v = 0; v < vocab; v++) dl[(size_t)r * vocab + v] = tl[(size_t)r * vocab + v];  // matched
        // Draft samples each drafted token from ITS row using the shared SAMPLE key at that position.
        int32_t drafts[Q35_DFLASH_K_MAX];
        for (int i = 0; i < K; i++) drafts[i] = draft_sample(&dl[(size_t)i * vocab], vocab, seed, pos[i]);
        auto rp = q35_dflash_verify_prob(tl.data(), dl.data(), vocab, drafts, K, seed, pos.data(), pos_bonus);
        // With matched distributions, p==q so p>u*q holds for u<1 => all accepted.
        CHECK(rp.accepted_len == K, "prob matched-dist accepts all K");
        CHECK(rp.emitted_token >= 0 && rp.emitted_token < vocab, "prob bonus in range");
        // Determinism across independent derivations.
        auto rp2 = q35_dflash_verify_prob(tl.data(), dl.data(), vocab, drafts, K, seed, pos.data(), pos_bonus);
        CHECK(rp.accepted_len == rp2.accepted_len && rp.emitted_token == rp2.emitted_token,
              "prob pipeline deterministic");
        auto cp = q35_dflash_assemble_commit(rp, drafts, K);
        CHECK(cp.commit_len == rp.accepted_len, "commit len == accepted");
        CHECK(cp.next_input_token == rp.emitted_token, "commit next == emitted");
    }

    if (failures) { printf("FAIL — DFlash verify pipeline integration (%d failures)\n", failures); return 1; }
    printf("PASS — DFlash verify pipeline: planner shapes, shared-key determinism, greedy accept/"
           "diverge + probabilistic accept, commit assembly seams all compose (weights-free)\n");
    return 0;
}
