// test_qwen35_detect.cc — host-only auto-detection gate for the Qwen3.5 hybrid (qwen35).
//
// Proves gemma4_detect_from_gguf reads the qwen35.* GGUF metadata into a gemma4_model_config_t
// WITHOUT building the CUDA engine: it mmaps the GGUF, runs detection, prints every detected dim
// and the full 32-layer FULL/LINEAR attention-kind pattern, and asserts:
//   - arch == GEMMA4_ARCH_QWEN3_5
//   - the period-`full_attention_interval` cadence (FULL iff (i+1)%interval==0)
//   - the authoritative Qwen3.5-9B dims (32 layers, hidden 4096, head_dim 256, 16/4 heads GQA,
//     vocab 248320, rope 1e7, partial-rotary 64, SwiGLU 12288, ssm 128/4/4096/16/32, 8 FULL layers).
//
// CUDA-free: gemma4_detect.h carries its own minimal GGUF reader, so this links with plain g++.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "gemma4_detect.h"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1]
        : "/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf";

    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open %s failed\n", path); return 2; }
    struct stat st;
    if (fstat(fd, &st) != 0) { fprintf(stderr, "fstat failed\n"); close(fd); return 2; }
    uint64_t size = (uint64_t)st.st_size;
    void *map = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); close(fd); return 2; }
    const uint8_t *data = (const uint8_t *)map;

    gemma4_model_config_t cfg;
    char err[256] = {0};
    int rc = gemma4_detect_from_gguf(data, size, &cfg, err, sizeof(err));
    if (rc != 0) { fprintf(stderr, "detect failed: %s\n", err); munmap(map, size); close(fd); return 2; }

    printf("== Qwen3.5 (qwen35) detection from %s ==\n", path);
    printf("arch                   = %d (expect %d GEMMA4_ARCH_QWEN3_5)\n", cfg.arch, GEMMA4_ARCH_QWEN3_5);
    printf("n_layers               = %d\n", cfg.n_layers);
    printf("hidden_size            = %d\n", cfg.hidden_size);
    printf("intermediate (SwiGLU)  = %d\n", cfg.intermediate);
    printf("n_heads (q)            = %d\n", cfg.n_heads);
    printf("n_kv (GQA)             = %d\n", cfg.n_kv_global);
    printf("head_dim (key_length)  = %d\n", cfg.head_dim);
    printf("vocab_size             = %d\n", cfg.vocab_size);
    printf("softcap                = %.1f\n", cfg.softcap);
    printf("rope_theta             = %.1f\n", cfg.rope_theta_global);
    printf("rotary_dim (partial)   = %d  (of head_dim %d)\n", cfg.rotary_dim, cfg.head_dim);
    printf("full_attention_interval= %d\n", cfg.full_attention_interval);
    printf("n_full / n_linear      = %d / %d\n", cfg.n_full, cfg.n_layers - cfg.n_full);
    printf("ssm.state_size         = %d\n", cfg.ssm_state_size);
    printf("ssm.conv_kernel        = %d\n", cfg.ssm_conv_kernel);
    printf("ssm.inner_size         = %d\n", cfg.ssm_inner_size);
    printf("ssm.group_count        = %d\n", cfg.ssm_group_count);
    printf("ssm.time_step_rank     = %d\n", cfg.ssm_time_step_rank);

    printf("per-layer attn kind (F=FULL softmax-GQA, L=LINEAR gated-deltanet):\n  ");
    for (int i = 0; i < cfg.n_layers; i++) {
        printf("%c", cfg.attn_kind[i] == GEMMA4_ATTN_FULL ? 'F' : 'L');
        if ((i % 4) == 3) printf(" ");
    }
    printf("\n  FULL layer indices:");
    for (int i = 0; i < cfg.n_layers; i++)
        if (cfg.attn_kind[i] == GEMMA4_ATTN_FULL) printf(" %d", i);
    printf("\n");

    // ── Assertions ──────────────────────────────────────────────────────────────────
    int fail = 0;
    #define CHK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fail++; } } while (0)
    CHK(cfg.arch == GEMMA4_ARCH_QWEN3_5, "arch %d != QWEN3_5", cfg.arch);
    CHK(cfg.n_layers == 32, "n_layers %d != 32", cfg.n_layers);
    CHK(cfg.hidden_size == 4096, "hidden_size %d != 4096", cfg.hidden_size);
    CHK(cfg.intermediate == 12288, "intermediate %d != 12288", cfg.intermediate);
    CHK(cfg.n_heads == 16, "n_heads %d != 16", cfg.n_heads);
    CHK(cfg.n_kv_global == 4, "n_kv %d != 4", cfg.n_kv_global);
    CHK(cfg.head_dim == 256, "head_dim %d != 256", cfg.head_dim);
    CHK(cfg.vocab_size == 248320, "vocab_size %d != 248320", cfg.vocab_size);
    CHK(cfg.softcap == 0.0f, "softcap %.1f != 0", cfg.softcap);
    CHK(cfg.rope_theta_global == 10000000.0f, "rope_theta %.1f != 1e7", cfg.rope_theta_global);
    CHK(cfg.rotary_dim == 64, "rotary_dim %d != 64", cfg.rotary_dim);
    CHK(cfg.full_attention_interval == 4, "interval %d != 4", cfg.full_attention_interval);
    CHK(cfg.n_full == 8, "n_full %d != 8", cfg.n_full);
    CHK(cfg.ssm_state_size == 128, "ssm.state_size %d != 128", cfg.ssm_state_size);
    CHK(cfg.ssm_conv_kernel == 4, "ssm.conv_kernel %d != 4", cfg.ssm_conv_kernel);
    CHK(cfg.ssm_inner_size == 4096, "ssm.inner_size %d != 4096", cfg.ssm_inner_size);
    CHK(cfg.ssm_group_count == 16, "ssm.group_count %d != 16", cfg.ssm_group_count);
    CHK(cfg.ssm_time_step_rank == 32, "ssm.time_step_rank %d != 32", cfg.ssm_time_step_rank);
    // period-4 cadence: FULL exactly at indices 3,7,11,...,31
    for (int i = 0; i < cfg.n_layers; i++) {
        int want_full = (((i + 1) % 4) == 0);
        int got_full  = (cfg.attn_kind[i] == GEMMA4_ATTN_FULL);
        CHK(want_full == got_full, "layer %d kind mismatch (want %s)", i, want_full ? "FULL" : "LINEAR");
    }
    #undef CHK

    munmap(map, size);
    close(fd);
    if (fail) { printf("DETECT GATE: FAIL (%d checks)\n", fail); return 1; }
    printf("DETECT GATE: PASS — qwen35 hybrid descriptor + period-4 FULL/LINEAR pattern + all dims\n");
    return 0;
}
