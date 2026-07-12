// ABOUTME: Host unit test for the DFlash shape/lookahead planner + enable/concurrency gate (S1a).
// ABOUTME: Verifies (1+K) shapes, S2 graph keys, N+1 lookahead, and default-off gating.
//
// build: g++ -std=c++17 -O2 -Wall -Wextra -Icuda cuda/qwen35_dflash_plan_test.cc -o /tmp/dflash_plan && /tmp/dflash_plan
#include "qwen35_dflash_plan.cuh"
#include <cstdio>

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } } while (0)

int main() {
    // K clamping: config-derived, bounded to [1, K_MAX].
    CHECK(q35_dflash_clamp_k(0) == 1, "clamp k=0 -> 1");
    CHECK(q35_dflash_clamp_k(-5) == 1, "clamp k<0 -> 1");
    CHECK(q35_dflash_clamp_k(8) == 8, "clamp k=8");
    CHECK(q35_dflash_clamp_k(16) == 16, "clamp k=16");
    CHECK(q35_dflash_clamp_k(999) == Q35_DFLASH_K_MAX, "clamp k>max");

    // Shapes: uniform tokens = 1+K, lookahead = K+1.
    for (int K = 1; K <= Q35_DFLASH_K_MAX; K++) {
        CHECK(q35_dflash_uniform_tokens(K) == 1 + K, "uniform tokens");
        CHECK(q35_dflash_lookahead(K) == K + 1, "lookahead K+1");
    }

    // S2 graph key: R requests * (1+K) rows; never aliases a decode key.
    {
        int K = 6, R = 3;
        q35_graph_key sk = q35_dflash_graph_key(R, K);
        CHECK(sk.num_reqs == R, "spec key reqs");
        CHECK(sk.uniform_token_count == 1 + K, "spec key utc");
        CHECK(sk.num_tokens == R * (1 + K), "spec key tokens");
        q35_graph_key dk = q35_make_decode_key(R);
        CHECK(!q35_graph_exact_match(dk, sk), "decode key must not match spec key");
        CHECK(dk.uniform_token_count != sk.uniform_token_count, "utc separates decode/spec");
    }

    // Gate: OFF disables always; ON/AUTO enable only for 0 < B < critical.
    CHECK(q35_dflash_gate(Q35_DFLASH_MODE_OFF, 1, 8) == 0, "OFF disabled at B=1");
    CHECK(q35_dflash_gate(Q35_DFLASH_MODE_OFF, 4, 8) == 0, "OFF disabled at B=4");
    CHECK(q35_dflash_gate(Q35_DFLASH_MODE_ON, 0, 8) == 0, "ON disabled at B=0");
    CHECK(q35_dflash_gate(Q35_DFLASH_MODE_ON, 1, 8) == 1, "ON enabled at B=1");
    CHECK(q35_dflash_gate(Q35_DFLASH_MODE_ON, 7, 8) == 1, "ON enabled at B=7");
    CHECK(q35_dflash_gate(Q35_DFLASH_MODE_ON, 8, 8) == 0, "ON disabled at B=critical");
    CHECK(q35_dflash_gate(Q35_DFLASH_MODE_ON, 16, 8) == 0, "ON disabled above critical");
    CHECK(q35_dflash_gate(Q35_DFLASH_MODE_AUTO, 4, 8) == 1, "AUTO enabled at B=4");
    CHECK(q35_dflash_gate(Q35_DFLASH_MODE_AUTO, 4, 4) == 0, "AUTO disabled at B=critical=4");
    // Default critical batch when <=0 passed.
    CHECK(q35_dflash_gate(Q35_DFLASH_MODE_ON, Q35_DFLASH_CRITICAL_BATCH_DEFAULT - 1, 0) == 1, "default cb enable");
    CHECK(q35_dflash_gate(Q35_DFLASH_MODE_ON, Q35_DFLASH_CRITICAL_BATCH_DEFAULT, 0) == 0, "default cb disable");

    // Env mapping: default off; "1" -> ON; "auto" -> AUTO; anything else -> OFF.
    CHECK(q35_dflash_mode_from_env(nullptr) == Q35_DFLASH_MODE_OFF, "env null -> OFF");
    CHECK(q35_dflash_mode_from_env("") == Q35_DFLASH_MODE_OFF, "env empty -> OFF");
    CHECK(q35_dflash_mode_from_env("0") == Q35_DFLASH_MODE_OFF, "env 0 -> OFF");
    CHECK(q35_dflash_mode_from_env("1") == Q35_DFLASH_MODE_ON, "env 1 -> ON");
    CHECK(q35_dflash_mode_from_env("auto") == Q35_DFLASH_MODE_AUTO, "env auto -> AUTO");
    CHECK(q35_dflash_mode_from_env("yes") == Q35_DFLASH_MODE_OFF, "env garbage -> OFF");

    // Full plan: disabled by default, correct shapes when enabled.
    {
        q35_dflash_plan off = q35_dflash_make_plan(8, Q35_DFLASH_MODE_OFF, 1, 8);
        CHECK(off.enabled == 0, "default plan disabled");
        CHECK(off.K == 8 && off.uniform_token_count == 9 && off.lookahead_slots == 9, "off plan shapes still derived");
        q35_dflash_plan on = q35_dflash_make_plan(16, Q35_DFLASH_MODE_ON, 2, 8);
        CHECK(on.enabled == 1, "on plan enabled at B=2");
        CHECK(on.K == 16 && on.uniform_token_count == 17 && on.lookahead_slots == 17, "on plan shapes");
        q35_dflash_plan gated = q35_dflash_make_plan(16, Q35_DFLASH_MODE_ON, 32, 8);
        CHECK(gated.enabled == 0, "on plan gated off at high batch");
    }

    if (failures) { printf("FAIL — DFlash planner/gate (%d failures)\n", failures); return 1; }
    printf("PASS — DFlash planner/gate: (1+K) shapes, S2 spec graph key, N+1 lookahead, "
           "default-off + concurrency gating, env mode mapping\n");
    return 0;
}
