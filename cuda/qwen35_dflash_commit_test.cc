// ABOUTME: Host unit test for the DFlash verify->commit assembly (P4 orchestration, weights-free).
// ABOUTME: Verifies commit token sequence, next-input token, and emitted count for every j in 0..K.
//
// build: g++ -std=c++17 -O2 -Wall -Wextra -Icuda cuda/qwen35_dflash_commit_test.cc -o /tmp/dflash_commit && /tmp/dflash_commit
#include "qwen35_dflash_commit.cuh"
#include <cstdio>

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } } while (0)

int main() {
    const int K = 6;
    int32_t drafts[K] = {101, 102, 103, 104, 105, 106};

    // For every accepted length j in 0..K, the commit plan must:
    //  - replay exactly the first j draft tokens (state advances only through accepted tokens),
    //  - carry the emitted token forward as the next input,
    //  - append j+1 tokens to history.
    for (int j = 0; j <= K; j++) {
        q35_dflash_verify_result r; r.accepted_len = j; r.emitted_token = 900 + j;
        auto p = q35_dflash_assemble_commit(r, drafts, K);
        CHECK(p.commit_len == j, "commit_len == j");
        for (int i = 0; i < j; i++) CHECK(p.commit_tokens[i] == drafts[i], "commit token matches draft");
        CHECK(p.next_input_token == 900 + j, "next input is emitted token");
        CHECK(p.total_emitted == j + 1, "total emitted j+1");
    }

    // j=0 (full rejection at first token): nothing committed, only the emitted token carries forward.
    {
        q35_dflash_verify_result r; r.accepted_len = 0; r.emitted_token = 777;
        auto p = q35_dflash_assemble_commit(r, drafts, K);
        CHECK(p.commit_len == 0, "j=0 commit empty");
        CHECK(p.next_input_token == 777, "j=0 next input");
        CHECK(p.total_emitted == 1, "j=0 emits 1 (just the corrected token)");
    }

    // j=K (full accept): all K drafts committed, bonus token carries forward, K+1 emitted.
    {
        q35_dflash_verify_result r; r.accepted_len = K; r.emitted_token = 555;
        auto p = q35_dflash_assemble_commit(r, drafts, K);
        CHECK(p.commit_len == K, "j=K commit all");
        for (int i = 0; i < K; i++) CHECK(p.commit_tokens[i] == drafts[i], "j=K token");
        CHECK(p.next_input_token == 555, "j=K bonus next input");
        CHECK(p.total_emitted == K + 1, "j=K emits K+1");
    }

    // Defensive clamping: an out-of-range accepted_len is clamped to [0,K] (P1 guarantees range,
    // but the assembly must never read past the draft array).
    {
        q35_dflash_verify_result r; r.accepted_len = K + 5; r.emitted_token = 1;
        auto p = q35_dflash_assemble_commit(r, drafts, K);
        CHECK(p.commit_len == K, "over-range accepted_len clamped to K");
        q35_dflash_verify_result r2; r2.accepted_len = -3; r2.emitted_token = 2;
        auto p2 = q35_dflash_assemble_commit(r2, drafts, K);
        CHECK(p2.commit_len == 0, "negative accepted_len clamped to 0");
    }

    if (failures) { printf("FAIL — DFlash commit assembly (%d failures)\n", failures); return 1; }
    printf("PASS — DFlash commit assembly: commit=accepted drafts, next-input=emitted token, "
           "history+=(j+1) for all j in 0..K; range clamping\n");
    return 0;
}
