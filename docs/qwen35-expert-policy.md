<!-- ABOUTME: Defines observational SSD expert telemetry and deterministic Qwen3.5 hotlist generation. -->
<!-- ABOUTME: Records DS4 provenance, exact replay semantics, default-off guarantees, and limitations. -->
# Qwen3.5 SSD expert policy

This policy layer extracts more value from fucina's **existing** grouped-NVFP4 SSD expert streamer.
It does not add another streamer, change model arithmetic, change expert order, or select policy from
timing. It is useful only when the transformed expert set does not fit the configured device slots.

The design is the highest-ranked adoption from
`docs/ds4-subsystem-analysis-2026-07-19.md` sections B.2.2, C, and D#1: observe real routing,
replay candidate capacities, and generate a deterministic preload hotlist.

## Gates and overhead

Profile mode is off unless both conditions hold:

1. `FUCINA_EXPERT_STREAM_SSD` enabled the existing SSD streamer successfully; and
2. `FUCINA_EXPERT_PROFILE_OUT` is a non-empty output path.

With `FUCINA_EXPERT_PROFILE_OUT` unset, the full-resident/default decode path has no profile
allocation, counter update, D2H copy, synchronization, or profile branch. Its CUDA graph and
resident decode structure are unchanged. The recorder pointer remains null. SSD streaming already
copies `d_moe_count` to the host and synchronizes before choosing active experts; profile mode uses
that exact host array and adds **no** GPU copy, readback, or synchronization.

Profiling does add bounded host bookkeeping and trace memory to the explicitly selected SSD path.
It is observational: recorded values never choose a kernel, expert order, cache victim, or weight.
Write failures are logged during engine destruction and never turn a successful inference into a
failure.

### Local host-only synthetic microbenchmark

An independent local review microbenchmark measured the recorder and generator without GPU or SSD
serving work. With 200,000 synthetic records at 40×256 geometry and 32 active experts, recording
cost **0.398 µs/event**. Atomically flushing a 65,536-event, **9.1 MiB** JSON profile cost **1.36 s**
at shutdown. Replaying five generator capacities cost **2.39 s** with **76,632 KiB** maximum RSS.
These are host-only synthetic measurements for overhead sizing, not p95/p99 serving evidence and
not a claim about request latency under real inference, SSD contention, or production routing.

## Configuration

| Environment variable | Meaning |
|---|---|
| `FUCINA_EXPERT_PROFILE_OUT=/path/profile.json` | Enables profile mode, but only with active SSD expert streaming. The file is emitted on engine destruction. |
| `FUCINA_EXPERT_PROFILE_MAX_EVENTS=N` | Maximum retained trace events. Default `65536`; `0` retains only aggregates; values above the hard safety bound `262144` are clamped. Invalid values use the default. |
| `FUCINA_EXPERT_STREAM_SSD=/path/expert-store.bin` | Existing streamer backing file; required for profile mode. |
| `FUCINA_EXPERT_STREAM_SLOTS=N` | Existing runtime global slot capacity, recorded in the profile. |
| `FUCINA_EXPERT_RESIDENCY_PLAN=/path/plan.json` | Existing loader input used on the subsequent restart. |

In addition to the event limit, the recorder and parser share a **6,291,456-ID** safety ceiling.
Events that do not fit either bound are counted in `events_dropped`; aggregate counters continue to
cover them. A conservative compact-JSON proof budgets 4,096 fixed bytes, 65,536 pair rows at 96
bytes, 256 layer wrappers at 256 bytes, 262,144 event wrappers at 32 bytes, and 6,291,456 IDs at
five bytes (four-digit ID plus comma). The resulting conservative producer upper bound is
**46,206,976 bytes (44.066 MiB)**, below the parser's 67,108,864-byte (64 MiB) limit by
20,901,888 bytes. The default
65,536-event workload can retain 32 active IDs/event (2,097,152 IDs) without approaching this cap.
The regression test keeps the C++ and Python cap values equal and proves this bound remains below
the consumer limit. Layer/expert geometry, path length, and all allocation products are also
bounded and checked.

## `fucina-expert-profile-v1`

The file is deterministic JSON: object fields, layer rows, expert rows, active IDs, and trace events
have fixed ordering. No timestamp, latency, address, random value, or timing-derived decision is
present. It is written in the destination directory as a temporary file, `fsync`ed, and atomically
renamed over the destination. The destination directory is also best-effort `fsync`ed.

```json
{
  "format": "fucina-expert-profile-v1",
  "geometry": {"layers": 40, "experts": 256},
  "configured_slots": 512,
  "max_events": 65536,
  "events_recorded": 65536,
  "events_dropped": 17,
  "layers": [
    {
      "layer": 0,
      "event_count": 1640,
      "active_expert_uniqueness": 251,
      "adjacent_intersection_count": 1234,
      "adjacent_union_count": 5678,
      "experts": [
        {"expert": 0, "selection_events": 73, "selected_rows": 91}
      ]
    }
  ],
  "streamer": {
    "cache_hits": 100,
    "cache_misses": 20,
    "ssd_reads": 80,
    "ssd_bytes": 123456789,
    "checksum_failures": 0,
    "prefetch_advice": 200
  },
  "trace": [
    {"layer": 0, "experts": [0, 7, 19]}
  ]
}
```

The example abbreviates arrays; real files contain exactly `geometry.layers` layer rows and exactly
`geometry.experts` expert rows per layer.

### Counter semantics

- One **selection event** is one SSD-streamed MoE layer invocation after routing.
- `(layer, expert).selection_events` increments once when that expert has at least one routed row in
  the event.
- `(layer, expert).selected_rows` adds the existing `d_moe_count[expert]`, so it measures selected
  token/expert assignment rows, not distinct requests.
- `event_count` is the number of events observed for that layer, including events dropped from the
  bounded trace.
- `active_expert_uniqueness` is the number of experts observed at least once in that layer.
- For consecutive events of the **same layer**, `adjacent_intersection_count` sums `|A ∩ B|` and
  `adjacent_union_count` sums `|A ∪ B|`. Their ratio is the aggregate adjacent-event Jaccard; the
  profile stores integer numerator and denominator rather than a rounded float.
- `streamer` values are the existing process-lifetime actual SSD cache/read/checksum/prefetch
  counters at destruction. They are not simulated values.
- Every retained trace event is `(layer, sorted active expert IDs)`. Expert IDs preserve the
  streamer's ascending scan. The trace is a prefix of observed events; `events_dropped` reports the
  unavailable suffix. No per-event row counts or timestamps are retained.

All integer counters are unsigned 64-bit values and recorder addition saturates instead of wrapping.

## Simulate capacities and generate a plan

`scripts/expert_residency_plan.py` rejects duplicate JSON keys, oversized files, wrong schemas,
negative/overflowing counters, excessive geometry, inconsistent aggregate counts, truncated event
lists, invalid layer/expert IDs, and unsorted or duplicate active IDs.

```bash
python3 scripts/expert_residency_plan.py /tmp/qwen-agent-profile.json \
  --slots 512 \
  --capacities 64,128,256,512,1024 \
  --out-plan /tmp/qwen-agent-residency.json \
  --out-report /tmp/qwen-agent-capacities.json
```

The report starts each candidate capacity cold and replays the retained global chronological trace.
For events no larger than the candidate capacity, it reproduces the current streamer's global LRU:
all event hits are refreshed in ascending expert order before ascending missing experts evict the
oldest pair. If an event has more active experts than the capacity, it reproduces the existing
chunk fallback: every active pair is a logical miss and retained cache state is cleared. The report
contains integer hits, misses, accesses, and fallback-event counts for every requested capacity.
Dropped events cannot be replayed and remain explicit in the report.

The generated file uses the already supported `fucina-expert-residency-v1` format; there is no
second runtime plan schema. All `(layer, expert)` pairs are ranked lexicographically by:

1. selection events, descending;
2. selected rows, descending;
3. layer ID, ascending;
4. expert ID, ascending.

The first `--slots` pairs receive tier `vram`; all others receive tier `ssd`. A bounded integer
ordinal is written as `importance`, so `q35_seed_ssd_residency` can parse and sort it without large
counter-to-double tie collapse. Repeated runs with identical input path, bytes, and CLI values emit
byte-identical plan/report files.

## Profile → validate → restart workflow

1. Choose an SSD slot budget and profile a representative, diverse workload:

   ```bash
   FUCINA_EXPERT_STREAM_SSD=/fast/qwen-experts.bin \
   FUCINA_EXPERT_STREAM_SLOTS=512 \
   FUCINA_EXPERT_PROFILE_OUT=/tmp/qwen-agent-profile.json \
   ./fucina -m /models/Qwen3.5-35B-A3B-NVFP4 ...
   ```

2. Stop the process normally so engine destruction atomically flushes the profile.
3. Run the generator with several candidate `--capacities`; inspect the optional report and choose
   `--slots` from held-out workload behavior rather than the training trace alone.
4. Restart with the existing loader:

   ```bash
   FUCINA_EXPERT_STREAM_SSD=/fast/qwen-experts.bin \
   FUCINA_EXPERT_STREAM_SLOTS=512 \
   FUCINA_EXPERT_RESIDENCY_PLAN=/tmp/qwen-agent-residency.json \
   ./fucina -m /models/Qwen3.5-35B-A3B-NVFP4 ...
   ```

5. Confirm the startup line reports that the plan seeded the expected number of expert slots, then
   compare cold misses and p95/p99 request latency on a separate held-out trace.

Run the host gates with `make expert-policy-test`; `make go-test` includes that target. The normal
CUDA/model gates remain authoritative for arithmetic and byte identity.

## Limitations

- This can improve policy only in SSD-capacity mode. It makes no resident decode throughput claim.
- A static hotlist can overfit one prompt mix, tenant, language, or agent workflow. Version profiles
  with workload/model identity outside this schema and refresh them deliberately.
- Capacity curves cover only retained events and begin cold; they do not reconstruct dropped events,
  OS page-cache behavior, SSD latency, startup seeding, or concurrent CPU/GPU LPDDR contention.
- Frequency is not latency. Validate p95 and p99 on held-out traces and retain the no-plan fallback.
- Prefetch advice is observed but not used to derive policy. Timing-driven prefetch or capacity
  selection would violate the deterministic design.
