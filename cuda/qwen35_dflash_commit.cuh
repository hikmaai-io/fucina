// ABOUTME: DFlash verify->commit assembly (P4 orchestration): maps a rejection result to the exact
// ABOUTME: P0 commit input sequence and the next-step input token. Pure host/device token logic.
//
// This is the glue between P1 (rejection sampler: accepted_len j + emitted token e) and P0 (lossless
// GDN commit that replays accepted tokens). It is weights-free and fully deterministic, so it is
// unit-testable before the draft forward exists.
//
// Fucina decode step semantics (qwen35_step_batch): feeding token T advances the slot through ONE
// position (T occupies the current position; GDN state advances through it) and predicts the next.
//
// One DFlash verify step, for a slot at absolute token count n0 with draft proposals d_1..d_K:
//   - The verifier accepts the leading j drafts (positions n0..n0+j-1) and emits token e at position
//     n0+j. e is the target argmax (greedy) or residual/bonus sample (probabilistic) from P1.
//   - To make GDN state byte-identical to sequential decoding, P0 commit replays EXACTLY the j
//     accepted draft tokens d_1..d_j (this header's commit_tokens[], commit_len=j). After commit the
//     slot is at n0+j, proven byte-identical by qwen35-gdn-rollback-test.
//   - e is NOT fed into state this step; it becomes the NEXT step's input token (it occupies
//     position n0+j and its own forward happens next step). total_emitted = j+1 tokens are appended
//     to the request's output history this step: [d_1..d_j, e].
//
// This separation is what keeps the whole step replayable and lossless: state advances only through
// committed tokens; the emitted token rides forward as the next input (the same contract S2's
// zero-host-feedback splice already uses for plain decode).
#ifndef FUCINA_QWEN35_DFLASH_COMMIT_CUH
#define FUCINA_QWEN35_DFLASH_COMMIT_CUH

#include <cstdint>
#include "qwen35_dflash_reject.cuh"
#include "qwen35_dflash_plan.cuh"   // Q35_DFLASH_K_MAX

#if defined(__CUDACC__)
#define Q35_DFLASH_COMMIT_HD __host__ __device__
#else
#define Q35_DFLASH_COMMIT_HD
#endif

// The assembled plan for one request's verify step. commit_tokens/commit_len feed P0 commit;
// next_input_token is fed at the following step; total_emitted tokens are appended to history.
struct q35_dflash_commit_plan {
    int32_t commit_tokens[Q35_DFLASH_K_MAX];  // the j accepted draft tokens to replay into state
    int     commit_len;                       // j in [0,K]
    int32_t next_input_token;                 // emitted token e (next step's input), never -1
    int     total_emitted;                    // j+1 tokens appended to output history this step
};

// Assemble the commit plan from a verify result and the draft proposals. Bounds: 0<=K<=K_MAX and
// 0<=r.accepted_len<=K (guaranteed by the P1 sampler). Returns the plan by value; no allocation.
Q35_DFLASH_COMMIT_HD static inline q35_dflash_commit_plan q35_dflash_assemble_commit(
        const q35_dflash_verify_result &r, const int32_t *draft_tokens, int K) {
    q35_dflash_commit_plan p;
    int j = r.accepted_len;
    if (j < 0) j = 0;
    if (j > K) j = K;
    for (int i = 0; i < j; i++) p.commit_tokens[i] = draft_tokens[i];
    p.commit_len = j;
    p.next_input_token = r.emitted_token;   // target argmax / residual / bonus, always a real id
    p.total_emitted = j + 1;                // the j accepted drafts plus the emitted token
    return p;
}

#endif // FUCINA_QWEN35_DFLASH_COMMIT_CUH
