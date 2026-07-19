// ABOUTME: Bounded deterministic telemetry for the existing Qwen3.5 SSD expert streamer.
// ABOUTME: Emits atomic fucina-expert-profile-v1 JSON without CUDA readbacks or timing inputs.
#include "expert_profile.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <new>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint64_t kDefaultMaxEvents = 65536;
constexpr uint64_t kHardMaxEvents = 262144;
// Paired with scripts/expert_residency_plan.py::MAX_TRACE_IDS. At the compact JSON grammar's
// proven worst case this keeps producer output <=46,206,976 bytes, below the 64 MiB parser cap.
constexpr size_t kHardTraceIds = 6 * 1024 * 1024;
constexpr int kMaxLayers = 256;
constexpr int kMaxExperts = 4096;
constexpr size_t kMaxPairs = 65536;
constexpr size_t kMaxPath = 4096;

uint64_t saturated_add(uint64_t a, uint64_t b) {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

uint64_t profile_max_events() {
    const char *value = getenv("FUCINA_EXPERT_PROFILE_MAX_EVENTS");
    if (!value || !*value) return kDefaultMaxEvents;
    if (*value == '-') {
        fprintf(stderr, "fucina: invalid FUCINA_EXPERT_PROFILE_MAX_EVENTS; using %llu\n",
                static_cast<unsigned long long>(kDefaultMaxEvents));
        return kDefaultMaxEvents;
    }
    errno = 0;
    char *end = nullptr;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        fprintf(stderr, "fucina: invalid FUCINA_EXPERT_PROFILE_MAX_EVENTS; using %llu\n",
                static_cast<unsigned long long>(kDefaultMaxEvents));
        return kDefaultMaxEvents;
    }
    if (errno == ERANGE || parsed > kHardMaxEvents) {
        fprintf(stderr, "fucina: FUCINA_EXPERT_PROFILE_MAX_EVENTS clamped to %llu\n",
                static_cast<unsigned long long>(kHardMaxEvents));
        return kHardMaxEvents;
    }
    return static_cast<uint64_t>(parsed);
}

struct PairCounters {
    uint64_t selection_events = 0;
    uint64_t selected_rows = 0;
};

struct LayerCounters {
    uint64_t event_count = 0;
    uint64_t active_uniqueness = 0;
    uint64_t adjacent_intersection = 0;
    uint64_t adjacent_union = 0;
    uint32_t previous_count = 0;
    bool has_previous = false;
};

struct TraceEvent {
    uint32_t layer;
    uint32_t offset;
    uint32_t count;
};

class JsonFdWriter {
public:
    explicit JsonFdWriter(int fd) : fd_(fd) {}

    void text(const char *value) { bytes(value, strlen(value)); }
    void number(uint64_t value) {
        char buffer[32];
        int n = snprintf(buffer, sizeof(buffer), "%llu", static_cast<unsigned long long>(value));
        if (n > 0) bytes(buffer, static_cast<size_t>(n));
        else ok_ = false;
    }
    bool ok() const { return ok_; }

private:
    void bytes(const char *value, size_t size) {
        size_t done = 0;
        while (ok_ && done < size) {
            ssize_t wrote = write(fd_, value + done, size - done);
            if (wrote < 0 && errno == EINTR) continue;
            if (wrote <= 0) { ok_ = false; return; }
            done += static_cast<size_t>(wrote);
        }
    }

    int fd_;
    bool ok_ = true;
};

std::string parent_directory(const std::string &path) {
    size_t slash = path.rfind('/');
    if (slash == std::string::npos) return ".";
    if (slash == 0) return "/";
    return path.substr(0, slash);
}

}  // namespace

struct fucina_expert_profile {
    fucina_expert_profile(const char *out, int layers, int experts, int slots, uint64_t max)
        : path(out), n_layers(layers), n_experts(experts), configured_slots(slots), max_events(max),
          pairs(static_cast<size_t>(layers) * experts), layer(static_cast<size_t>(layers)),
          previous(static_cast<size_t>(layers) * experts, 0),
          ever_active(static_cast<size_t>(layers) * experts, 0) {
        const size_t event_reserve = static_cast<size_t>(std::min<uint64_t>(max_events, 4096));
        events.reserve(event_reserve);
        const size_t id_reserve = std::min(kHardTraceIds,
            event_reserve > SIZE_MAX / 16 ? kHardTraceIds : event_reserve * 16);
        trace_ids.reserve(id_reserve);
    }

    std::string path;
    int n_layers;
    int n_experts;
    int configured_slots;
    uint64_t max_events;
    uint64_t dropped_events = 0;
    std::vector<PairCounters> pairs;
    std::vector<LayerCounters> layer;
    std::vector<uint8_t> previous;
    std::vector<uint8_t> ever_active;
    std::vector<TraceEvent> events;
    std::vector<uint32_t> trace_ids;
};

extern "C" fucina_expert_profile_t *fucina_expert_profile_create(
    int ssd_active, int n_layers, int n_experts, int configured_slots) {
    if (!ssd_active) return nullptr;
    const char *path = getenv("FUCINA_EXPERT_PROFILE_OUT");
    if (!path || !*path) return nullptr;  // Keep default-off allocation-free.
    const size_t path_size = strlen(path);
    if (path_size > kMaxPath) {
        fprintf(stderr, "fucina: expert profile path is too long; profiling disabled\n");
        return nullptr;
    }
    if (n_layers < 1 || n_layers > kMaxLayers || n_experts < 1 || n_experts > kMaxExperts ||
        static_cast<size_t>(n_layers) > kMaxPairs / static_cast<size_t>(n_experts) ||
        configured_slots < 1 || configured_slots > 4096) {
        fprintf(stderr, "fucina: invalid expert profile geometry; profiling disabled\n");
        return nullptr;
    }
    try {
        fucina_expert_profile *profile = new fucina_expert_profile(
            path, n_layers, n_experts, configured_slots, profile_max_events());
        fprintf(stderr, "fucina: observational SSD expert profile ON (max_events=%llu, out=%s)\n",
                static_cast<unsigned long long>(profile->max_events), path);
        return profile;
    } catch (...) {
        fprintf(stderr, "fucina: expert profile allocation failed; inference continues unprofiled\n");
        return nullptr;
    }
}

extern "C" void fucina_expert_profile_record(
    fucina_expert_profile_t *profile, int layer_index, const int *counts, int n_experts) {
    if (!profile || !counts || layer_index < 0 || layer_index >= profile->n_layers ||
        n_experts != profile->n_experts) return;

    const size_t base = static_cast<size_t>(layer_index) * profile->n_experts;
    LayerCounters &layer = profile->layer[static_cast<size_t>(layer_index)];
    uint32_t active_count = 0;
    uint32_t intersection = 0;
    for (int expert = 0; expert < profile->n_experts; ++expert) {
        const int rows = counts[expert];
        if (rows <= 0) continue;
        ++active_count;
        const size_t pair_index = base + static_cast<size_t>(expert);
        PairCounters &pair = profile->pairs[pair_index];
        pair.selection_events = saturated_add(pair.selection_events, 1);
        pair.selected_rows = saturated_add(pair.selected_rows, static_cast<uint64_t>(rows));
        if (!profile->ever_active[pair_index]) {
            profile->ever_active[pair_index] = 1;
            layer.active_uniqueness = saturated_add(layer.active_uniqueness, 1);
        }
        intersection += profile->previous[pair_index] != 0;
    }
    layer.event_count = saturated_add(layer.event_count, 1);
    if (layer.has_previous) {
        layer.adjacent_intersection = saturated_add(layer.adjacent_intersection, intersection);
        const uint64_t union_count = static_cast<uint64_t>(layer.previous_count) + active_count - intersection;
        layer.adjacent_union = saturated_add(layer.adjacent_union, union_count);
    }
    memset(profile->previous.data() + base, 0, static_cast<size_t>(profile->n_experts));
    for (int expert = 0; expert < profile->n_experts; ++expert)
        if (counts[expert] > 0) profile->previous[base + static_cast<size_t>(expert)] = 1;
    layer.previous_count = active_count;
    layer.has_previous = true;

    if (active_count == 0 || profile->events.size() >= profile->max_events ||
        profile->trace_ids.size() > kHardTraceIds - std::min<size_t>(active_count, kHardTraceIds)) {
        profile->dropped_events = saturated_add(profile->dropped_events, 1);
        return;
    }
    const size_t old_ids = profile->trace_ids.size();
    try {
        for (int expert = 0; expert < profile->n_experts; ++expert)
            if (counts[expert] > 0) profile->trace_ids.push_back(static_cast<uint32_t>(expert));
        profile->events.push_back({static_cast<uint32_t>(layer_index), static_cast<uint32_t>(old_ids),
                                   active_count});
    } catch (...) {
        profile->trace_ids.resize(old_ids);
        profile->dropped_events = saturated_add(profile->dropped_events, 1);
    }
}

static int write_profile_json(const fucina_expert_profile *profile,
                              const fucina_expert_stream_stats_t &stats) {
    static std::atomic<unsigned long long> serial{0};
    char suffix[96];
    snprintf(suffix, sizeof(suffix), ".tmp.%ld.%llu", static_cast<long>(getpid()),
             serial.fetch_add(1, std::memory_order_relaxed));
    std::string temp;
    try { temp = profile->path + suffix; }
    catch (const std::bad_alloc &) { return -1; }

    int fd = open(temp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return -1;
    JsonFdWriter out(fd);
    out.text("{\"format\":\"fucina-expert-profile-v1\",\"geometry\":{\"layers\":");
    out.number(static_cast<uint64_t>(profile->n_layers));
    out.text(",\"experts\":"); out.number(static_cast<uint64_t>(profile->n_experts));
    out.text("},\"configured_slots\":"); out.number(static_cast<uint64_t>(profile->configured_slots));
    out.text(",\"max_events\":"); out.number(profile->max_events);
    out.text(",\"events_recorded\":"); out.number(profile->events.size());
    out.text(",\"events_dropped\":"); out.number(profile->dropped_events);
    out.text(",\"layers\":[");
    for (int layer_index = 0; layer_index < profile->n_layers; ++layer_index) {
        if (layer_index) out.text(",");
        const LayerCounters &layer = profile->layer[static_cast<size_t>(layer_index)];
        out.text("{\"layer\":"); out.number(static_cast<uint64_t>(layer_index));
        out.text(",\"event_count\":"); out.number(layer.event_count);
        out.text(",\"active_expert_uniqueness\":"); out.number(layer.active_uniqueness);
        out.text(",\"adjacent_intersection_count\":"); out.number(layer.adjacent_intersection);
        out.text(",\"adjacent_union_count\":"); out.number(layer.adjacent_union);
        out.text(",\"experts\":[");
        const size_t base = static_cast<size_t>(layer_index) * profile->n_experts;
        for (int expert = 0; expert < profile->n_experts; ++expert) {
            if (expert) out.text(",");
            const PairCounters &pair = profile->pairs[base + static_cast<size_t>(expert)];
            out.text("{\"expert\":"); out.number(static_cast<uint64_t>(expert));
            out.text(",\"selection_events\":"); out.number(pair.selection_events);
            out.text(",\"selected_rows\":"); out.number(pair.selected_rows); out.text("}");
        }
        out.text("]}");
    }
    out.text("],\"streamer\":{\"cache_hits\":"); out.number(stats.cache_hits);
    out.text(",\"cache_misses\":"); out.number(stats.cache_misses);
    out.text(",\"ssd_reads\":"); out.number(stats.ssd_reads);
    out.text(",\"ssd_bytes\":"); out.number(stats.ssd_bytes);
    out.text(",\"checksum_failures\":"); out.number(stats.checksum_failures);
    out.text(",\"prefetch_advice\":"); out.number(stats.prefetch_advice);
    out.text("},\"trace\":[");
    for (size_t i = 0; i < profile->events.size(); ++i) {
        if (i) out.text(",");
        const TraceEvent &event = profile->events[i];
        out.text("{\"layer\":"); out.number(event.layer); out.text(",\"experts\":[");
        for (uint32_t j = 0; j < event.count; ++j) {
            if (j) out.text(",");
            out.number(profile->trace_ids[static_cast<size_t>(event.offset) + j]);
        }
        out.text("]}");
    }
    out.text("]}\n");

    int rc = 0;
    if (!out.ok() || fsync(fd) != 0) rc = -1;
    if (close(fd) != 0) rc = -1;
    if (rc == 0 && rename(temp.c_str(), profile->path.c_str()) != 0) rc = -1;
    if (rc == 0) {
        const std::string parent = parent_directory(profile->path);
        int dir_fd = open(parent.c_str(), O_RDONLY | O_DIRECTORY);
        if (dir_fd >= 0) { (void)fsync(dir_fd); close(dir_fd); }
    }
    if (rc != 0) unlink(temp.c_str());
    return rc;
}

extern "C" int fucina_expert_profile_finish(
    fucina_expert_profile_t *profile, const fucina_expert_stream_stats_t *stats) {
    if (!profile) return 0;
    const fucina_expert_stream_stats_t zero{};
    int rc = -1;
    try { rc = write_profile_json(profile, stats ? *stats : zero); }
    catch (...) { rc = -1; }
    delete profile;
    return rc;
}
