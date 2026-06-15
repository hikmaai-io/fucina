// test_diffusion_dequant.cu — validate DiffusionGemma dequant kernels against gguf-py.
//
// Reads dumps produced by scripts/dg_dump_tensor.py (<tag>.{meta,raw,ref}), dequantizes the
// raw bytes on-GPU via dg_dequant(), and compares element-for-element with the reference fp32.
//
// Build:  nvcc -O2 -arch=sm_121 cuda/diffusion_gemma_kernels.cu cuda/test_diffusion_dequant.cu \
//             -o /tmp/test_dg_dequant
// Run:    /tmp/test_dg_dequant /tmp/dg_dequant

#include "diffusion_gemma_kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s @ %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__); exit(1);} } while(0)

static std::vector<uint8_t> read_file(const std::string &p) {
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", p.c_str()); exit(1); }
    fclose(f);
    return v;
}

struct Case { const char *tag; const char *name; };

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : "/tmp/dg_dequant";
    Case cases[] = {
        {"q4_k", "Q4_K (blk.0.attn_q)"},
        {"q5_0", "Q5_0 (self_cond_down)"},
        {"q6_k", "Q6_K (blk.0.attn_v)"},
        {"q8_0", "Q8_0 (blk.0.ffn_down)"},
    };
    int n_cases = sizeof(cases) / sizeof(cases[0]);
    int failures = 0;

    for (int c = 0; c < n_cases; c++) {
        std::string base = std::string(dir) + "/" + cases[c].tag;
        // meta: "<ggml_type> <n_elem> <raw_bytes>"
        FILE *mf = fopen((base + ".meta").c_str(), "r");
        if (!mf) { fprintf(stderr, "missing %s.meta — run scripts/dg_dump_tensor.py\n", base.c_str()); return 1; }
        int gtype; long long n_elem, raw_bytes;
        if (fscanf(mf, "%d %lld %lld", &gtype, &n_elem, &raw_bytes) != 3) { fprintf(stderr, "bad meta\n"); return 1; }
        fclose(mf);

        std::vector<uint8_t> raw = read_file(base + ".raw");
        std::vector<uint8_t> refb = read_file(base + ".ref");
        if ((long long)raw.size() != raw_bytes) { fprintf(stderr, "raw size mismatch\n"); return 1; }
        const float *ref = (const float *)refb.data();
        if ((long long)(refb.size() / 4) != n_elem) { fprintf(stderr, "ref size mismatch\n"); return 1; }

        uint8_t *d_raw; float *d_out;
        CK(cudaMalloc(&d_raw, raw.size()));
        CK(cudaMalloc(&d_out, n_elem * sizeof(float)));
        CK(cudaMemcpy(d_raw, raw.data(), raw.size(), cudaMemcpyHostToDevice));

        int rc = dg_dequant(gtype, d_raw, n_elem, d_out, 0);
        CK(cudaDeviceSynchronize());
        if (rc != 0) { fprintf(stderr, "dg_dequant rc=%d for %s\n", rc, cases[c].name); failures++; continue; }

        std::vector<float> got(n_elem);
        CK(cudaMemcpy(got.data(), d_out, n_elem * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(d_raw); cudaFree(d_out);

        double max_abs = 0.0, sum_abs = 0.0; long long n_bad = 0; long long worst_i = -1;
        for (long long i = 0; i < n_elem; i++) {
            double e = fabs((double)got[i] - (double)ref[i]);
            if (e > max_abs) { max_abs = e; worst_i = i; }
            sum_abs += e;
            if (e > 1e-3) n_bad++;
        }
        bool pass = (max_abs <= 1e-3) && (n_bad == 0);
        printf("[%s] %-26s n=%lld  max|Δ|=%.3e  mean|Δ|=%.3e  bad(>1e-3)=%lld  %s\n",
               pass ? "PASS" : "FAIL", cases[c].name, n_elem, max_abs, sum_abs / n_elem, n_bad,
               pass ? "" : "");
        if (!pass) {
            failures++;
            printf("      worst @ %lld: got=%.8f ref=%.8f\n", worst_i,
                   (double)got[worst_i], (double)ref[worst_i]);
        }
    }

    printf("\n%s (%d/%d cases passed)\n", failures ? "DEQUANT TESTS FAILED" : "ALL DEQUANT TESTS PASSED",
           n_cases - failures, n_cases);
    return failures ? 1 : 0;
}
