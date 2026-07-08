// gemma4_detect_test.c — standalone host test for runtime Gemma-4 architecture detection.
//
// Opens a GGUF, fills a gemma4_model_config_t via gemma4_detect_from_gguf, prints it, and (when a
// second "expect" arg is given) asserts the dense-31B geometry. Compiles WITHOUT CUDA — it only
// pulls in gemma4_detect.h (which is CUDA-free). Build:
//   cc -std=c11 -Icuda cuda/gemma4_detect_test.c -o /tmp/gemma4_detect_test
// Run:
//   /tmp/gemma4_detect_test <model.gguf> [31b]

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "gemma4_detect.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [31b]\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    int expect_31b = (argc >= 3 && strcmp(argv[2], "31b") == 0);

    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); close(fd); return 1; }
    uint64_t size = (uint64_t)st.st_size;
    const uint8_t *data = (const uint8_t *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    gemma4_model_config_t cfg;
    char err[256] = {0};
    int rc = gemma4_detect_from_gguf(data, size, &cfg, err, sizeof(err));
    if (rc != 0) {
        fprintf(stderr, "detect FAILED: %s\n", err);
        munmap((void *)data, size); close(fd);
        return 1;
    }

    printf("=== detected Gemma-4 config from %s ===\n", path);
    printf("  n_layers          = %d\n", cfg.n_layers);
    printf("  hidden_size       = %d\n", cfg.hidden_size);
    printf("  intermediate(FFN) = %d\n", cfg.intermediate);
    printf("  n_heads           = %d\n", cfg.n_heads);
    printf("  n_kv_sliding      = %d\n", cfg.n_kv_sliding);
    printf("  n_kv_global       = %d\n", cfg.n_kv_global);
    printf("  vocab_size        = %d\n", cfg.vocab_size);
    printf("  softcap           = %.3f\n", cfg.softcap);
    printf("  rope_theta_global = %.1f\n", cfg.rope_theta_global);
    printf("  rope_theta_sliding= %.1f\n", cfg.rope_theta_sliding);
    printf("  n_global          = %d\n", cfg.n_global);
    printf("  is_global[]       = ");
    for (int i = 0; i < cfg.n_layers; i++) printf("%d", cfg.is_global[i]);
    printf("\n");

    munmap((void *)data, size);
    close(fd);

    if (expect_31b) {
        int ok = 1;
        #define CHECK(field, want) do { \
            if ((cfg.field) != (want)) { \
                fprintf(stderr, "MISMATCH %s = %d, expected %d\n", #field, (int)(cfg.field), (int)(want)); \
                ok = 0; \
            } \
        } while (0)
        CHECK(n_layers, 60);
        CHECK(hidden_size, 5376);
        CHECK(intermediate, 21504);
        CHECK(n_heads, 32);
        CHECK(n_kv_sliding, 16);
        CHECK(n_kv_global, 4);
        CHECK(vocab_size, 262144);
        CHECK(n_global, 10);
        if (cfg.softcap != 30.0f) { fprintf(stderr, "MISMATCH softcap = %f\n", cfg.softcap); ok = 0; }
        if (cfg.rope_theta_global != 1000000.0f) { fprintf(stderr, "MISMATCH rope_global\n"); ok = 0; }
        if (cfg.rope_theta_sliding != 10000.0f) { fprintf(stderr, "MISMATCH rope_sliding\n"); ok = 0; }
        #undef CHECK
        if (!ok) { fprintf(stderr, "=== 31B detection FAILED ===\n"); return 1; }
        printf("=== 31B detection PASSED ===\n");
    }
    return 0;
}
