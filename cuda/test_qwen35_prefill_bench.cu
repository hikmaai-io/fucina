// test_qwen35_prefill_bench.cu — warm 2k prefill timing for the MoE FP8 engine.
// Usage: bench <model> [N=2000] [iters=5]. Prefills an N-token prompt `iters` times
// (warm), reporting median wall-clock TTFT. Throwaway harness (not a gate).
#include "gemma4_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <chrono>
using namespace std::chrono;
static double now_ms() {
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "/opt/spark/models/models--Qwen--Qwen3.5-35B-A3B-FP8";
    int N     = (argc > 2) ? atoi(argv[2]) : 2000;
    int iters = (argc > 3) ? atoi(argv[3]) : 5;

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 8192, 0, 0.90);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    int32_t *lp = (int32_t*)malloc((size_t)N * sizeof(int32_t));
    int32_t fr[5] = { 760, 6511, 314, 9338, 369 };
    for (int i = 0; i < 5 && i < N; i++) lp[i] = fr[i];
    for (int i = 5; i < N; i++) lp[i] = (int32_t)(((i * 1315423911u) ^ (i << 3)) % 200000u + 16u);

    double *ts = (double*)malloc(iters * sizeof(double));
    for (int it = 0; it < iters; it++) {
        int32_t first = -1;
        double t0 = now_ms();
        int slot = gemma4_engine_seq_add(eng, lp, N, &first, 0.f, 0, 0.f, 0.f, 0);
        double t = now_ms() - t0;
        if (slot < 0) { fprintf(stderr, "seq_add failed at iter %d\n", it); return 2; }
        gemma4_engine_seq_remove(eng, slot);
        ts[it] = t;
        printf("  iter %d: %.1f ms  first=%d\n", it, t, first);
    }
    std::sort(ts, ts + iters);
    printf("PREFILL-BENCH N=%d: median %.1f ms  (%.0f tok/s)\n",
           N, ts[iters/2], N * 1000.0 / ts[iters/2]);
    gemma4_engine_destroy(eng);
    return 0;
}
