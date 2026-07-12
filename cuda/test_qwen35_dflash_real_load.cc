// ABOUTME: Real-checkpoint validation gate for the DFlash draft loader (P2 against real weights).
// ABOUTME: Opens z-lab/Qwen3.5-9B-DFlash and validates its 69-tensor schema before any CUDA alloc.
//
// Unlike the synthetic qwen35_dflash_loader_test, this opens the REAL downloaded checkpoint and
// asserts the config-derived Geometry + tensor validation accept it, pinning the exact geometry
// (H=4096 L=6 NQ=32 NKV=8 HD=128 V=248320 mask=248077 F=8 fc_in=32768 window=4096, layers
// S S S S S F, no d2t). Skips cleanly (exit 0) when the checkpoint is absent so CI without weights
// stays green.
//
// build: g++ -std=c++17 -O2 -Wall -Wextra -Icuda cuda/test_qwen35_dflash_real_load.cc -o /tmp/dflash_real && /tmp/dflash_real [snapshot-dir]
#include "qwen35_dflash_loader.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

static bool exists(const std::string& p) { struct stat s; return stat(p.c_str(), &s) == 0; }

int main(int argc, char** argv) {
    const char* def = "/opt/spark/models/models--z-lab--Qwen3.5-9B-DFlash/"
                      "snapshots/5fc3b3d474760f18c516db87d84c37edbfd3ede6";
    std::string dir = (argc > 1) ? argv[1] : def;
    if (!exists(dir + "/model.safetensors") || !exists(dir + "/config.json")) {
        printf("SKIP — real DFlash checkpoint not present at %s\n", dir.c_str());
        return 0;
    }
    std::ifstream f(dir + "/config.json");
    std::stringstream ss; ss << f.rdbuf();
    std::string cj = ss.str();

    qwen35dflash::Geometry g; std::string e;
    if (!qwen35dflash::parse_config(cj, g, e)) { printf("FAIL: real config rejected: %s\n", e.c_str()); return 1; }

    st::Model m;
    if (!m.open((dir + "/model.safetensors").c_str(), e)) { printf("FAIL: open real ckpt: %s\n", e.c_str()); return 1; }

    if (!qwen35dflash::validate_tensors(m, g, g.V, e)) { printf("FAIL: real tensors rejected: %s\n", e.c_str()); return 1; }

    // Pin the exact expected geometry of the public checkpoint (regression lock).
    int fails = 0;
    auto eq = [&](const char* what, long got, long want){ if (got != want) { printf("FAIL: %s = %ld, expected %ld\n", what, got, want); fails++; } };
    eq("H", g.H, 4096); eq("I", g.I, 12288); eq("L", g.L, 6);
    eq("NQ", g.NQ, 32); eq("NKV", g.NKV, 8); eq("HD", g.HD, 128);
    eq("V", g.V, 248320); eq("mask_token_id", g.mask_token_id, 248077);
    eq("num_target_features", g.num_target_features, 8); eq("fc_in", g.fc_in(), 32768);
    eq("sliding_window", g.sliding_window, 4096);
    eq("tensor_count", (long)m.count(), 69);
    eq("has_d2t", g.has_d2t ? 1 : 0, 0);
    // Layer pattern S S S S S F.
    if (g.layer_attn.size() != 6) { printf("FAIL: layer count\n"); fails++; }
    else for (int i = 0; i < 6; i++) {
        int want = (i < 5) ? qwen35dflash::ATTN_SLIDING : qwen35dflash::ATTN_FULL;
        if (g.layer_attn[i] != want) { printf("FAIL: layer %d attn kind\n", i); fails++; }
    }

    if (fails) { printf("FAIL — real DFlash loader gate (%d checks)\n", fails); return 1; }
    printf("PASS — real z-lab/Qwen3.5-9B-DFlash loads + validates: 69 tensors, H=4096 L=6 NQ=32 "
           "NKV=8 HD=128 V=248320 mask=248077 F=8 fc_in=32768 window=4096, layers S S S S S F\n");
    return 0;
}
