// test_qwen35_layer_parity.cu — M2 per-layer kernel parity gate for the Qwen3.5 hybrid.
//
// Loads the torch reference binary produced by cuda/qwen35_layer_ref.py (dequantized fp32
// weights + a fixed input hidden + the reference mixer outputs for a FULL softmax-attention
// layer and a GDN gated-deltanet layer), runs fucina's qwen35 M2 mixer kernels in fp32, and
// asserts max-abs relative error < 1e-2 vs torch for BOTH layer kinds AND that the GDN
// chunked-scan output matches the single-step recurrence. The actual compare + print lives in
// qwen35_m2_layer_selftest (cuda/gemma4_kernels.cu); this is the GPU entry point + gate.
#include <cstdio>
#include "gemma4_kernels.cuh"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "/tmp/qwen35_m2_ref.bin";
    int rc = qwen35_m2_layer_selftest(path);
    if (rc != 0) { fprintf(stderr, "FAIL — qwen35 M2 per-layer parity gate (rc=%d)\n", rc); return 1; }
    printf("PASS — qwen35 M2 per-layer parity (FULL + GDN recur/chunk, chunk==recur)\n");
    return 0;
}
