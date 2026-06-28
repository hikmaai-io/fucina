// test_qwen3_bench.cu — decode throughput of fucina's Qwen3 dense path.
// Single-stream tok/s and concurrent (continuous-batching) aggregate tok/s, on the
// arch-driven multiseq path. Greedy. Same model as the parity test.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <chrono>
#include "gemma4_kernels.cuh"

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3-8B-abliterated.Q4_K_M.gguf";
    setenv("FUCINA_PAGED_KV", "1", 1);
    int32_t prompt[] = { 785, 6722, 315, 9625, 374 };
    const int NP = 5;
    const int WARM = 16, STEPS = 256;

    gemma4_engine_t *eng = gemma4_engine_create(path, FORMAT_Q4_0, 8192, 0, 0.90);
    if (!eng) { fprintf(stderr, "create failed\n"); return 2; }

    // ---- single-stream decode ----
    {
        int32_t first = 0;
        int slot = gemma4_engine_seq_add(eng, prompt, NP, &first, 0.0f, 0, 0.0f, 0.0f, 0);
        int32_t tok = first;
        for (int k = 0; k < WARM; k++) { int32_t n=0; int s=slot; gemma4_engine_step_batch(eng,&s,&tok,1,&n); tok=n; }
        double t0 = now_s();
        for (int k = 0; k < STEPS; k++) { int32_t n=0; int s=slot; gemma4_engine_step_batch(eng,&s,&tok,1,&n); tok=n; }
        double dt = now_s() - t0;
        printf("fucina Qwen3-8B Q4_K  single-stream decode: %.1f tok/s  (%d steps, %.3fs)\n",
               STEPS / dt, STEPS, dt);
        gemma4_engine_seq_remove(eng, slot);
    }

    // ---- concurrent decode (continuous batching) at several batch sizes ----
    for (int B : {4, 8, 16, 24, 32}) {
        std::vector<int> slots(B); std::vector<int32_t> cur(B);
        bool ok = true;
        for (int q = 0; q < B; q++) {
            int32_t first = 0;
            slots[q] = gemma4_engine_seq_add(eng, prompt, NP, &first, 0.0f, 0, 0.0f, 0.0f, (uint64_t)(q+1));
            if (slots[q] < 0) { ok = false; break; }
            cur[q] = first;
        }
        if (!ok) { printf("fucina Qwen3-8B Q4_K  B=%d: admission failed (capacity)\n", B);
                   for (int q=0;q<B;q++) if(slots[q]>=0) gemma4_engine_seq_remove(eng,slots[q]); continue; }
        std::vector<int32_t> nxt(B);
        for (int k = 0; k < WARM; k++) { gemma4_engine_step_batch(eng, slots.data(), cur.data(), B, nxt.data()); cur = nxt; }
        double t0 = now_s();
        for (int k = 0; k < STEPS; k++) { gemma4_engine_step_batch(eng, slots.data(), cur.data(), B, nxt.data()); cur = nxt; }
        double dt = now_s() - t0;
        printf("fucina Qwen3-8B Q4_K  B=%-2d  aggregate decode: %.1f tok/s  (per-seq %.1f, %d steps, %.3fs)\n",
               B, (double)B * STEPS / dt, (double)STEPS / dt, STEPS, dt);
        for (int q = 0; q < B; q++) gemma4_engine_seq_remove(eng, slots[q]);
    }

    gemma4_engine_destroy(eng);
    return 0;
}
