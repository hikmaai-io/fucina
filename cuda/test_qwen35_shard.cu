// Phase-E hardware gate: the host-visible fp32 layer frontier must preserve the
// exact Qwen3.5 greedy stream. Compares one [0,L) boundary with [0,cut)+[cut,L)
// on independent slots, requiring every final logit byte and 8 greedy ids to
// match. This exercises real KV/GDN state on both sides of the cut.
#include "gemma4_kernels.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int argmax(const std::vector<float>& x) {
    return (int)(std::max_element(x.begin(), x.end()) - x.begin());
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen3.5-model>\n", argv[0]);
        return 2;
    }
    setenv("FUCINA_PAGED_KV", "1", 1);
    setenv("FUCINA_PAGED_MAXSEQS", "2", 1);
    gemma4_engine_t *eng = gemma4_engine_create(argv[1], FORMAT_Q8_0, 4096, 0, 0.90);
    if (!eng) return 1;
    const int L = gemma4_engine_get_n_layers(eng);
    const int H = gemma4_engine_get_hidden_size(eng);
    const int V = gemma4_engine_get_vocab_size(eng);
    const int cut = L / 2;
    if (L < 2 || H <= 0 || V <= 0) {
        std::fprintf(stderr, "invalid geometry L=%d H=%d V=%d\n", L, H, V);
        gemma4_engine_destroy(eng); return 1;
    }
    int full = gemma4_engine_seq_open(eng, 0, 0, 0, 0, 0);
    int split = gemma4_engine_seq_open(eng, 0, 0, 0, 0, 0);
    if (full < 0 || split < 0) {
        std::fprintf(stderr, "cannot open shard slots\n");
        gemma4_engine_destroy(eng); return 1;
    }

    // Existing Qwen oracle prompt: "The capital of France is".
    const int32_t prompt[] = {760, 6511, 314, 9338, 369};
    const int np = (int)(sizeof(prompt) / sizeof(prompt[0]));
    std::vector<float> emb(H), frontier(H), logits_full(V), logits_split(V);
    const int ref[8] = {11751, 13, 198, 760, 6511, 314, 9338, 369};
    int32_t token = 0;
    std::vector<int> ids;
    bool ok = true;
    for (int pos = 0; pos < np + 7; ++pos) {
        int32_t in = pos < np ? prompt[pos] : token;
        if (gemma4_engine_q35_embed(eng, in, emb.data()) != 0 ||
            gemma4_engine_q35_forward_layers(eng, full, pos, 1, 0, L,
                                             emb.data(), logits_full.data(), 1) != 0 ||
            gemma4_engine_q35_forward_layers(eng, split, pos, 1, 0, cut,
                                             emb.data(), frontier.data(), 0) != 0 ||
            gemma4_engine_q35_forward_layers(eng, split, pos, 1, cut, L,
                                             frontier.data(), logits_split.data(), 1) != 0) {
            std::fprintf(stderr, "shard forward failed at pos %d\n", pos);
            ok = false; break;
        }
        if (std::memcmp(logits_full.data(), logits_split.data(), (size_t)V*sizeof(float)) != 0) {
            size_t i = 0; while (i < (size_t)V && logits_full[i] == logits_split[i]) ++i;
            std::fprintf(stderr, "frontier mismatch pos=%d first_logit=%zu full=%g split=%g\n",
                         pos, i, i<(size_t)V?logits_full[i]:0.f, i<(size_t)V?logits_split[i]:0.f);
            ok = false; break;
        }
        if (pos >= np - 1) {
            token = (int32_t)argmax(logits_split);
            ids.push_back(token);
        }
    }
    int oracle = 0;
    for (size_t i = 0; i < ids.size() && i < 8; ++i) oracle += ids[i] == ref[i];
    std::printf("qwen35 shard cut [0,%d)+[%d,%d):", cut, cut, L);
    for (int id : ids) std::printf(" %d", id);
    std::printf("\n%s: %zu/8 byte-identical frontier steps, %d/8 FP8 oracle tokens\n",
                ok && ids.size()==8 && oracle==8 ? "PASS" : "FAIL", ids.size(), oracle);
    ok = ok && ids.size() == 8 && oracle == 8;
    gemma4_engine_seq_remove(eng, full);
    gemma4_engine_seq_remove(eng, split);
    gemma4_engine_destroy(eng);
    return ok ? 0 : 1;
}
