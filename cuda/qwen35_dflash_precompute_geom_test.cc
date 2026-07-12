// ABOUTME: Host unit test for the DFlash context-KV precompute geometry (P3 shape planning).
// ABOUTME: Checks fused KV GEMM / K-norm / RoPE / cache shapes vs the real draft config dims.
//
// build: g++ -std=c++17 -O2 -Wall -Wextra -Icuda cuda/qwen35_dflash_precompute_geom_test.cc -o /tmp/dflash_pcgeom && /tmp/dflash_pcgeom
#include "qwen35_dflash_precompute_geom.cuh"
#include <cstdio>

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", (m)); failures++; } } while (0)

int main() {
    // Real z-lab/Qwen3.5-9B-DFlash draft dims (config-derived): H=4096, HD=128, NKV=8, L=6.
    // kv_dim = 8*128 = 1024.
    {
        q35_dflash_precompute_geom g;
        const int L = 6, H = 4096, HD = 128, NKV = 8, num_ctx = 40;
        bool ok = q35_dflash_precompute_geometry(L, H, HD, NKV, num_ctx, &g);
        CHECK(ok, "geometry computes for real dims");
        CHECK(g.kv_dim == 1024, "kv_dim = NKV*HD");
        CHECK(g.fused_kv_weight_rows == (int64_t)L * 2 * 1024, "fused KV weight rows L*2*kv_dim");
        CHECK(g.fused_kv_weight_elems == g.fused_kv_weight_rows * H, "fused KV weight elems * H");
        CHECK(g.kv_proj_out_elems == (int64_t)num_ctx * L * 2 * 1024, "GEMM out = num_ctx*L*2*kv_dim");
        CHECK(g.k_all_elems == (int64_t)L * num_ctx * 1024, "K-all = L*num_ctx*kv_dim");
        CHECK(g.v_all_elems == g.k_all_elems, "V-all == K-all");
        CHECK(g.knorm_weight_elems == (int64_t)L * HD, "K-norm stack = L*HD");
        CHECK(g.rope_rows == (int64_t)L * num_ctx, "RoPE rows = L*num_ctx");
        CHECK(g.ctx_norm_elems == (int64_t)num_ctx * H, "ctx norm = num_ctx*H");
    }

    // Scaling in num_ctx is linear on every context-dependent buffer; weight buffer is context-free.
    {
        q35_dflash_precompute_geom a, b;
        q35_dflash_precompute_geometry(6, 4096, 128, 8, 10, &a);
        q35_dflash_precompute_geometry(6, 4096, 128, 8, 20, &b);
        CHECK(b.kv_proj_out_elems == 2 * a.kv_proj_out_elems, "GEMM out scales linearly in num_ctx");
        CHECK(b.k_all_elems == 2 * a.k_all_elems, "K-all scales linearly");
        CHECK(b.rope_rows == 2 * a.rope_rows, "RoPE rows scale linearly");
        CHECK(b.fused_kv_weight_elems == a.fused_kv_weight_elems, "weight buffer context-free");
        CHECK(b.knorm_weight_elems == a.knorm_weight_elems, "K-norm stack context-free");
    }

    // Defensive: non-positive dims rejected; null out rejected.
    {
        q35_dflash_precompute_geom g;
        CHECK(!q35_dflash_precompute_geometry(0, 4096, 128, 8, 40, &g), "reject L=0");
        CHECK(!q35_dflash_precompute_geometry(6, 0, 128, 8, 40, &g), "reject H=0");
        CHECK(!q35_dflash_precompute_geometry(6, 4096, 0, 8, 40, &g), "reject HD=0");
        CHECK(!q35_dflash_precompute_geometry(6, 4096, 128, 0, 40, &g), "reject NKV=0");
        CHECK(!q35_dflash_precompute_geometry(6, 4096, 128, 8, 0, &g), "reject num_ctx=0");
        CHECK(!q35_dflash_precompute_geometry(6, 4096, 128, 8, 40, nullptr), "reject null out");
    }

    // Overflow guard: absurd dims must be rejected, not silently wrap.
    {
        q35_dflash_precompute_geom g;
        CHECK(!q35_dflash_precompute_geometry(1<<20, 1<<20, 1<<20, 1<<20, 1<<20, &g), "reject overflow dims");
    }

    if (failures) { printf("FAIL — DFlash precompute geometry (%d failures)\n", failures); return 1; }
    printf("PASS — DFlash precompute geometry: fused KV GEMM / grouped K-norm / batched RoPE / "
           "cache-insert shapes config-derived, linear-in-ctx scaling, overflow-guarded\n");
    return 0;
}
