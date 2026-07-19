// ABOUTME: CUDA-independent bounded recorder API for observational SSD expert-stream telemetry.
// ABOUTME: Keeps profile state absent unless both SSD streaming and its explicit env gate are on.
#ifndef FUCINA_EXPERT_PROFILE_H
#define FUCINA_EXPERT_PROFILE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fucina_expert_profile fucina_expert_profile_t;

typedef struct fucina_expert_stream_stats {
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t ssd_reads;
    uint64_t ssd_bytes;
    uint64_t checksum_failures;
    uint64_t prefetch_advice;
} fucina_expert_stream_stats_t;

// Returns NULL before allocating when ssd_active is false or FUCINA_EXPERT_PROFILE_OUT is unset.
fucina_expert_profile_t *fucina_expert_profile_create(
    int ssd_active, int n_layers, int n_experts, int configured_slots);

// counts is the existing ascending expert-count D2H snapshot owned by the SSD streaming path.
void fucina_expert_profile_record(
    fucina_expert_profile_t *profile, int layer, const int *counts, int n_experts);

// Atomically writes, destroys profile in every case, and returns 0 on success or -1 on I/O failure.
int fucina_expert_profile_finish(
    fucina_expert_profile_t *profile, const fucina_expert_stream_stats_t *stats);

#ifdef __cplusplus
}
#endif
#endif
