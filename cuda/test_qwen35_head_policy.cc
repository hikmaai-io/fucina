// ABOUTME: CPU gate for staged Qwen3.6 MoE LM-head rows-per-warp dispatch and rollback parsing.
// ABOUTME: Proves the unset and malformed policies retain the incumbent one-row kernel.
#include <cassert>
#include "qwen35_head_policy.h"

int main() {
    assert(q35_q8_head_rows_policy(nullptr) == 1);
    assert(q35_q8_head_rows_policy("") == 1);
    assert(q35_q8_head_rows_policy("1") == 1);
    assert(q35_q8_head_rows_policy("2") == 2);
    assert(q35_q8_head_rows_policy("4") == 4);
    assert(q35_q8_head_rows_policy("3") == 1);
    assert(q35_q8_head_rows_policy("4x") == 1);
    assert(q35_q8_head_rows_policy("-1") == 1);
    return 0;
}
