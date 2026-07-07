// test_e4b_load.cu — load the real Gemma-4-E4B checkpoint end-to-end through
// e4b_engine_create (BF16 weights + FP8 PLE index) and report residency.
// Exit 0 iff detection + full weight load succeed. Usage: test_e4b_load [dir]

#include <cstdio>
#include "e4b_engine.h"

static const char* kDefaultDir =
    "/opt/spark/models/hub/models--google--gemma-4-E4B-it/snapshots/"
    "fee6332c1abaafb77f6f9624236c63aa2f1d0187";

int main(int argc, char** argv) {
    const char* dir = (argc > 1) ? argv[1] : kDefaultDir;

    int det = e4b_is_e4b_checkpoint(dir);
    printf("e4b_is_e4b_checkpoint(%s) = %d\n", dir, det);
    if (det != 1) { fprintf(stderr, "FAIL: not detected as E4B\n"); return 1; }

    e4b_engine_t* eng = e4b_engine_create(dir, 4096, 1, 0);
    if (!eng) { fprintf(stderr, "FAIL: e4b_engine_create returned NULL\n"); return 1; }

    e4b_engine_print_info(eng);

    if (e4b_engine_n_layers(eng) != 42 || e4b_engine_hidden_size(eng) != 2560) {
        fprintf(stderr, "FAIL: unexpected dims\n"); e4b_engine_destroy(eng); return 1;
    }
    e4b_engine_destroy(eng);
    printf("PASS: E4B checkpoint loaded (BF16 weights + FP8 PLE index)\n");
    return 0;
}
