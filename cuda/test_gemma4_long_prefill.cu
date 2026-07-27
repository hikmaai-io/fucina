// Regression for dense Gemma-4's classic chunked prefill across the 4096-token boundary.
//
// 4376 tokens are deliberate: the classic path forwards 128 full 32-row chunks through
// position 4096, then eight more full chunks and a 24-row tail.  Global attention grows
// from 64 to 69 splits (64 keys/split), while every row keeps its own 128-slot scratch
// slab.  This must be the engine's first batched operation: historically decode_batched_dev
// took d_sb[0] before lazy scratch allocation, so the first token upload targeted NULL and
// the otherwise-valid long prefill failed with cudaErrorInvalidValue.
#include <cstdio>
#include <cstdint>
#include <vector>

#include "gemma4_kernels.cuh"

static constexpr int kPromptTokens = 4376;
static_assert(kPromptTokens > 4096, "regression must cross the persistent-prefill boundary");
static_assert((4096 + GEMMA4_GLOBAL_SPLIT_CHUNK - 1) / GEMMA4_GLOBAL_SPLIT_CHUNK == 64,
              "4096-token global-attention split pin changed");
static_assert((kPromptTokens + GEMMA4_GLOBAL_SPLIT_CHUNK - 1) /
                  GEMMA4_GLOBAL_SPLIT_CHUNK == 69,
              "4376-token global-attention split pin changed");
static_assert(69 <= GEMMA4_GLOBAL_MAX_SPLITS, "regression exceeds per-row scratch stride");

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1]
        : "/opt/spark/models/gemma-4-12b-it-qat-q4_0.gguf";

    std::vector<int32_t> prompt(kPromptTokens);
    prompt[0] = GEMMA4_BOS_ID;
    for (int i = 1; i < kPromptTokens; ++i)
        prompt[i] = (i & 1) ? 106 : 107; // deterministic, in-vocabulary Gemma tokens

    gemma4_engine_t *eng = gemma4_engine_create(
        model, FORMAT_Q4_0, /*context_size=*/8192, /*device_id=*/0, /*gpu_mem_util=*/0.90);
    if (!eng) {
        std::fprintf(stderr, "gemma4-long-prefill: engine create failed\n");
        return 2;
    }

    // Call the classic path directly and before warmup so its lazy scratch ordering is covered.
    int rc = gemma4_engine_prefill(eng, prompt.data(), kPromptTokens, nullptr);
    gemma4_engine_destroy(eng);
    if (rc != 0) {
        std::fprintf(stderr, "gemma4-long-prefill: 4376-token classic prefill failed (rc=%d)\n", rc);
        return 1;
    }

    std::printf("PASS — Gemma-4 12B Q4 classic prefill crossed 4096 (%d tokens, 69 splits)\n",
                kPromptTokens);
    return 0;
}
