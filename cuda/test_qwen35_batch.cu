// test_qwen35_batch.cu — M4 gate for fucina's Qwen3.5 hybrid PAGED-BATCHED + CUDA-graph decode.
//
// Loads the qwen35 Q4_K_M GGUF and runs qwen35_batch_selftest, which drives the continuous-
// batching ABI (gemma4_engine_seq_add prefill + gemma4_engine_step_batch decode) and asserts:
//   (1) B-row batched decode (graph ON) is BIT-IDENTICAL per row to that row run alone B=1
//       (graph OFF) — the batch self-test invariant (row independence + graph correctness);
//   (2) graph-ON == graph-OFF for the B-row batch (CUDA-graph determinism);
//   (3) the batched path reproduces the proven M3 single-seq forward (qwen35_forward_greedy)
//       — the France->Paris 8/8 greedy continuation.
// Exits 0 on PASS.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "gemma4_kernels.cuh"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf";

    // Small context keeps the fp32 FULL-layer K/V arena modest; the gate only decodes ~30 tokens.
    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 4096, 0, 0.90);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    int rc = qwen35_batch_selftest(eng);

    // Cross the 256-token initial KV block through resumable prefill, then replay the already
    // captured B=1 decode graph. This gates prefix migration + pointer-table graph indirection.
    gemma4_memory_stats_t before_growth{}, after_growth{};
    int gs = gemma4_engine_seq_open(eng, 0.f, 0, 0.f, 0.f, 7);
    int32_t gp[300], first=0, next=0;
    for (int i=0; i<300; i++) gp[i] = 100 + (i % 1000);
    if (gs < 0 || gemma4_engine_seq_prefill_chunk(eng, gs, gp, 200, 0, nullptr) != 0) {
        fprintf(stderr, "qwen35 block-KV gate: first chunk failed\n"); rc=1;
    } else {
        gemma4_engine_memory_stats(eng, &before_growth);
        if (gemma4_engine_seq_prefill_chunk(eng, gs, gp+200, 100, 1, &first) != 0) {
            fprintf(stderr, "qwen35 block-KV gate: growth chunk failed\n"); rc=1;
        } else {
            gemma4_engine_memory_stats(eng, &after_growth);
            int one=gs;
            if (after_growth.qwen_committed_bytes <= before_growth.qwen_committed_bytes ||
                gemma4_engine_step_batch(eng, &one, &first, 1, &next) != 0) {
                fprintf(stderr, "qwen35 block-KV gate: accounting/graph replay failed\n"); rc=1;
            } else {
                fprintf(stderr, "qwen35 block-KV gate: 256->512 growth + graph replay — PASS\n");
            }
        }
    }
    if (gs >= 0) gemma4_engine_seq_remove(eng, gs);

    gemma4_memory_stats_t ms{};
    gemma4_engine_memory_stats(eng, &ms);
    if (ms.qwen_workspace_bytes == 0 || ms.qwen_committed_bytes == 0 ||
        ms.qwen_reserved_bytes < ms.qwen_committed_bytes || ms.qwen_capacity <= 0 ||
        ms.qwen_allocated_slots != 3 || ms.qwen_reserved_context <= 0 ||
        ms.qwen_reserved_context > ms.qwen_max_context) {
        fprintf(stderr, "qwen35 memory-plan gate failed: workspace=%llu committed=%llu "
                        "reserved=%llu capacity=%d allocated=%d\n",
                (unsigned long long)ms.qwen_workspace_bytes,
                (unsigned long long)ms.qwen_committed_bytes,
                (unsigned long long)ms.qwen_reserved_bytes, (int)ms.qwen_capacity,
                (int)ms.qwen_allocated_slots);
        rc = 1;
    } else {
        fprintf(stderr, "qwen35 memory-plan gate: workspace=%.2f GiB committed=%.2f GiB "
                        "reserved=%.2f GiB capacity=%d allocated=%d slotctx=%d maxctx=%d — PASS\n",
                ms.qwen_workspace_bytes/(1024.0*1024*1024),
                ms.qwen_committed_bytes/(1024.0*1024*1024),
                ms.qwen_reserved_bytes/(1024.0*1024*1024),
                (int)ms.qwen_capacity, (int)ms.qwen_allocated_slots,
                (int)ms.qwen_reserved_context, (int)ms.qwen_max_context);
    }
    gemma4_engine_destroy(eng);

    printf("%s\n", (rc == 0) ? "PASS — qwen35 M4 batched-decode gate"
                             : "FAIL — qwen35 M4 batched-decode gate");
    return rc;
}
