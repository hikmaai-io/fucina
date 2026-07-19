// ABOUTME: Host-only selftest for bounded SSD expert telemetry and atomic profile emission.
// ABOUTME: Exercises default-off behavior without CUDA or a model checkpoint.
#include "expert_profile.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>

static int fail(const char *message) {
    fprintf(stderr, "expert_profile_test: %s\n", message);
    return 1;
}

static std::string read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return {};
    std::string value;
    char buffer[4096];
    size_t n;
    while ((n = fread(buffer, 1, sizeof(buffer), file)) != 0) value.append(buffer, n);
    fclose(file);
    return value;
}

int main() {
    char dir[] = "/tmp/fucina-expert-profile-test.XXXXXX";
    if (!mkdtemp(dir)) return fail("mkdtemp failed");
    std::string path = std::string(dir) + "/profile.json";

    unsetenv("FUCINA_EXPERT_PROFILE_OUT");
    unsetenv("FUCINA_EXPERT_PROFILE_MAX_EVENTS");
    if (fucina_expert_profile_create(/*ssd_active=*/1, 2, 4, 2) != nullptr)
        return fail("unset profile path did not stay allocation-free/off");
    if (access(path.c_str(), F_OK) == 0) return fail("default-off created a file");

    setenv("FUCINA_EXPERT_PROFILE_OUT", path.c_str(), 1);
    if (fucina_expert_profile_create(/*ssd_active=*/0, 2, 4, 2) != nullptr)
        return fail("profile enabled without SSD streaming");
    if (access(path.c_str(), F_OK) == 0) return fail("SSD-inactive profile created a file");
    setenv("FUCINA_EXPERT_PROFILE_MAX_EVENTS", "2", 1);
    fucina_expert_profile_t *profile = fucina_expert_profile_create(1, 2, 4, 2);
    if (!profile) return fail("profile create failed");

    const int a[] = {2, 0, 1, 0};  // active [0,2], selected rows [2,1]
    const int b[] = {1, 3, 0, 0};  // active [0,1]
    const int c[] = {0, 0, 4, 1};  // active [2,3], dropped only from bounded trace
    fucina_expert_profile_record(profile, 0, a, 4);
    fucina_expert_profile_record(profile, 0, b, 4);
    fucina_expert_profile_record(profile, 1, c, 4);

    fucina_expert_stream_stats_t stats{};
    stats.cache_hits = 7;
    stats.cache_misses = 5;
    stats.ssd_reads = 20;
    stats.ssd_bytes = 1234;
    stats.checksum_failures = 1;
    stats.prefetch_advice = 9;
    if (fucina_expert_profile_finish(profile, &stats) != 0)
        return fail("atomic profile finish failed");

    std::string json = read_file(path.c_str());
    const std::string first_json = json;
    if (json.find("\"format\":\"fucina-expert-profile-v1\"") == std::string::npos)
        return fail("schema missing");
    if (json.find("\"events_recorded\":2") == std::string::npos ||
        json.find("\"events_dropped\":1") == std::string::npos)
        return fail("trace bounds missing");
    if (json.find("\"selection_events\":2,\"selected_rows\":3") == std::string::npos)
        return fail("per-expert counters wrong");
    if (json.find("\"adjacent_intersection_count\":1,\"adjacent_union_count\":3") == std::string::npos)
        return fail("adjacent overlap counters wrong");
    if (json.find("\"ssd_bytes\":1234") == std::string::npos)
        return fail("stream counters missing");
    if (json.find("\"trace\":[{\"layer\":0,\"experts\":[0,2]},{\"layer\":0,\"experts\":[0,1]}]") == std::string::npos)
        return fail("stable trace wrong");

    // Cross-language contract: the hostile-input validator accepts recorder output and the
    // generated plan retains the exact lexical schema q35_seed_ssd_residency consumes.
    const std::string plan_path = std::string(dir) + "/plan.json";
    const std::string report_path = std::string(dir) + "/report.json";
    const std::string command = "python3 scripts/expert_residency_plan.py " + path +
        " --slots 2 --capacities 1,2 --out-plan " + plan_path + " --out-report " + report_path +
        " >/dev/null";
    if (system(command.c_str()) != 0) return fail("recorder-to-generator compatibility failed");
    const std::string plan = read_file(plan_path.c_str());
    if (plan.find("\"format\": \"fucina-expert-residency-v1\"") == std::string::npos ||
        plan.find("\"layers.0.experts.") == std::string::npos ||
        plan.find("\"tier\": \"vram\"") == std::string::npos ||
        plan.find("\"importance\":") == std::string::npos)
        return fail("generated plan is not loader-compatible");
    unlink(plan_path.c_str());
    unlink(report_path.c_str());

    // Repeating identical observations must produce identical profile bytes.
    profile = fucina_expert_profile_create(1, 2, 4, 2);
    if (!profile) return fail("repeat profile create failed");
    fucina_expert_profile_record(profile, 0, a, 4);
    fucina_expert_profile_record(profile, 0, b, 4);
    fucina_expert_profile_record(profile, 1, c, 4);
    if (fucina_expert_profile_finish(profile, &stats) != 0)
        return fail("repeat profile finish failed");
    if (read_file(path.c_str()) != first_json) return fail("profile bytes are nondeterministic");

    // A later flush must atomically replace an existing target and leave no temp sibling.
    setenv("FUCINA_EXPERT_PROFILE_MAX_EVENTS", "999999999999999999999999", 1);
    profile = fucina_expert_profile_create(1, 1, 2, 1);
    if (!profile) return fail("clamped profile create failed");
    const int d[] = {0, 1};
    fucina_expert_profile_record(profile, 0, d, 2);
    if (fucina_expert_profile_finish(profile, &stats) != 0)
        return fail("atomic replacement failed");
    json = read_file(path.c_str());
    if (json.find("\"layers\":1,\"experts\":2") == std::string::npos)
        return fail("target was not replaced");

    DIR *dp = opendir(dir);
    if (!dp) return fail("opendir failed");
    int files = 0;
    while (dirent *entry = readdir(dp)) {
        if (entry->d_name[0] == '.') continue;
        files++;
        if (strstr(entry->d_name, ".tmp.")) { closedir(dp); return fail("temp file leaked"); }
    }
    closedir(dp);
    if (files != 1) return fail("unexpected files in atomic-write directory");

    // An unwritable destination reports failure but cannot throw or terminate the caller.
    const std::string bad_path = std::string(dir) + "/missing/profile.json";
    setenv("FUCINA_EXPERT_PROFILE_OUT", bad_path.c_str(), 1);
    profile = fucina_expert_profile_create(1, 1, 2, 1);
    if (!profile) return fail("failure-path profile create failed");
    fucina_expert_profile_record(profile, 0, d, 2);
    if (fucina_expert_profile_finish(profile, &stats) != -1)
        return fail("unwritable profile destination did not report failure");

    unlink(path.c_str());
    rmdir(dir);
    puts("PASS expert profile: default-off + bounded counters + atomic deterministic JSON");
    return 0;
}
