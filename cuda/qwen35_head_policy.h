// ABOUTME: Runtime gate for the Qwen3.6 MoE exact Q8/BF16 greedy LM-head scan.
// ABOUTME: Keeps the register-blocked kernel target-specific and provides an immediate rollback.
#pragma once

#include <stdlib.h>

// Return the number of output rows a warp evaluates concurrently. The transformation only
// reuses one activation load across independent vocabulary rows; each row retains the legacy
// b/j accumulation and warp-reduction order. The optimized kernels remain explicit A/B choices
// until the GB10 exactness and speed acceptance gates have run; an unset override stays on the
// incumbent one-row kernel.
static inline int q35_q8_head_rows_policy(const char *env) {
    if (env && *env) {
        char *end = nullptr;
        long v = strtol(env, &end, 10);
        if (end && *end == '\0' && (v == 1 || v == 2 || v == 4)) return (int)v;
    }
    return 1; // unset, malformed, and unsupported values fail closed to the incumbent kernel
}
