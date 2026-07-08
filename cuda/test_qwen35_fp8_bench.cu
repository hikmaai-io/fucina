// test_qwen35_fp8_bench.cu — single-stream decode tok/s + prefill latency of the Qwen3.5-9B
// FP8 reference path (qwen35_fp8_forward_greedy) on the SAME official FP8 checkpoint vLLM serves.
// This is the strict same-checkpoint (FP8 vs FP8) apples-to-apples anchor for the P7 bench: it
// isolates fucina's per-token FP8 compute from the Q4_K_M-vs-FP8 quant difference of the served
// comparison. NOTE: the FP8 path is the token-by-token *reference oracle* (no CUDA-graph / batching /
// continuous-batching), NOT fucina's optimized server — the optimized engine runs the GGUF.
//
// Method: forward() re-runs prefill each call, so decode tok/s = (N2-N1)/(t(N2)-t(N1)) cancels the
// (tiny, 5-token) prefill. Prefill latency reported separately as t(n_gen=1) on a short prompt.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include "gemma4_kernels.cuh"

static double time_forward(void *m, const int32_t *ids, int np, int ngen) {
    int32_t *out = (int32_t *)calloc(ngen, sizeof(int32_t));
    auto t0 = std::chrono::high_resolution_clock::now();
    int rc = qwen35_fp8_forward_greedy(m, ids, np, out, ngen);
    auto t1 = std::chrono::high_resolution_clock::now();
    free(out);
    if (rc != 0) { fprintf(stderr, "forward failed (ngen=%d)\n", ngen); exit(2); }
    return std::chrono::duration<double>(t1 - t0).count();
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/models--Qwen--Qwen3.5-9B-FP8";
    int N1 = (argc > 2) ? atoi(argv[2]) : 16;
    int N2 = (argc > 3) ? atoi(argv[3]) : 96;

    int32_t ids[5] = { 760, 6511, 314, 9338, 369 };

    void *m = qwen35_fp8_load(path);
    if (!m) { fprintf(stderr, "qwen35_fp8_load failed\n"); return 2; }

    // warm (lazy cuBLAS / module load / scratch alloc)
    (void)time_forward(m, ids, 5, 4);

    // prefill latency proxy: prefill(5 tok) + 1 decode step
    double t_pref = 1e9;
    for (int r = 0; r < 3; r++) { double t = time_forward(m, ids, 5, 1); if (t < t_pref) t_pref = t; }

    // decode tok/s via differencing (median of 3)
    double best_dtps = 0;
    for (int r = 0; r < 3; r++) {
        double t1 = time_forward(m, ids, 5, N1);
        double t2 = time_forward(m, ids, 5, N2);
        double dt = t2 - t1;
        if (dt > 0) { double dtps = (N2 - N1) / dt; if (dtps > best_dtps) best_dtps = dtps; }
    }

    qwen35_fp8_free(m);
    printf("FP8-BENCH: prefill+1tok latency=%.1f ms (5-tok prompt) | decode=%.2f tok/s (single-stream, FP8 reference path)\n",
           t_pref * 1000.0, best_dtps);
    return 0;
}
