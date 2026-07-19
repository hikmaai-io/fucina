#!/usr/bin/env python3
# ABOUTME: Replays bounded SSD expert traces and emits deterministic Qwen3.5 residency hotlists.
# ABOUTME: Strictly validates fucina-expert-profile-v1 before making loader-consumed policy.
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any


PROFILE_FORMAT = "fucina-expert-profile-v1"
PLAN_FORMAT = "fucina-expert-residency-v1"
REPORT_FORMAT = "fucina-expert-residency-report-v1"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_LAYERS = 256
MAX_EXPERTS = 4096
MAX_PAIRS = 65_536
MAX_EVENTS = 262_144
MAX_TRACE_IDS = 6 * 1024 * 1024
MAX_CAPACITY = 4096
U64_MAX = (1 << 64) - 1


# Conservative compositional proof for expert_profile.cc's compact writer grammar. Bounds include
# list commas: pair row <=96 B, layer wrapper <=256 B, event wrapper <=32 B, expert ID+comma <=5 B,
# and 4096 B for fixed root/streamer syntax. Keep the matching regression test when grammar changes.
def producer_profile_size_upper_bound() -> int:
    return (4096 + MAX_PAIRS * 96 + MAX_LAYERS * 256 +
            MAX_EVENTS * 32 + MAX_TRACE_IDS * 5)


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list")
    return value


def _integer(value: Any, where: str, low: int = 0, high: int = U64_MAX) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low or value > high:
        raise ValueError(f"{where} must be an integer in [{low}, {high}]")
    return value


def _keys(value: dict[str, Any], required: set[str], where: str) -> None:
    missing = required - value.keys()
    if missing:
        raise ValueError(f"{where} missing keys: {', '.join(sorted(missing))}")


def _checked_add(a: int, b: int, where: str) -> int:
    if a > U64_MAX - b:
        raise ValueError(f"{where} overflows uint64")
    return a + b


def _read_bounded(path: Path) -> bytes:
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"cannot read profile: {exc}") from exc
    if not raw or len(raw) > MAX_FILE_BYTES:
        raise ValueError(f"profile size must be in [1, {MAX_FILE_BYTES}] bytes")
    return raw


def load_profile(path: Path, *, raw: bytes | None = None) -> dict[str, Any]:
    """Load and validate one bounded fucina-expert-profile-v1 document."""
    try:
        if raw is None:
            raw = _read_bounded(path)
        elif not raw or len(raw) > MAX_FILE_BYTES:
            raise ValueError(f"profile size must be in [1, {MAX_FILE_BYTES}] bytes")
        profile = json.loads(raw, object_pairs_hook=_object_no_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError,
            RecursionError, MemoryError) as exc:
        raise ValueError(f"invalid profile JSON: {exc}") from exc

    root = _mapping(profile, "profile")
    required = {"format", "geometry", "configured_slots", "max_events", "events_recorded",
                "events_dropped", "layers", "streamer", "trace"}
    _keys(root, required, "profile")
    if root["format"] != PROFILE_FORMAT:
        raise ValueError(f"unsupported profile format: {root['format']!r}")

    geometry = _mapping(root["geometry"], "geometry")
    _keys(geometry, {"layers", "experts"}, "geometry")
    n_layers = _integer(geometry["layers"], "geometry.layers", 1, MAX_LAYERS)
    n_experts = _integer(geometry["experts"], "geometry.experts", 1, MAX_EXPERTS)
    if n_layers > MAX_PAIRS // n_experts:
        raise ValueError("layer/expert geometry exceeds bounded pair count")
    pair_count = n_layers * n_experts

    _integer(root["configured_slots"], "configured_slots", 1, MAX_CAPACITY)
    _integer(root["max_events"], "max_events", 0, MAX_EVENTS)
    events_recorded = _integer(root["events_recorded"], "events_recorded", 0, MAX_EVENTS)
    events_dropped = _integer(root["events_dropped"], "events_dropped")
    if events_recorded > root["max_events"]:
        raise ValueError("events_recorded exceeds max_events")
    events_total = _checked_add(events_recorded, events_dropped, "event count")

    streamer = _mapping(root["streamer"], "streamer")
    stream_keys = {"cache_hits", "cache_misses", "ssd_reads", "ssd_bytes",
                   "checksum_failures", "prefetch_advice"}
    _keys(streamer, stream_keys, "streamer")
    for key in stream_keys:
        _integer(streamer[key], f"streamer.{key}")

    trace = _list(root["trace"], "trace")
    if len(trace) != events_recorded:
        raise ValueError("events_recorded does not match trace length (truncated profile)")
    trace_layer_events = [0] * n_layers
    trace_select = [[0] * n_experts for _ in range(n_layers)]
    trace_id_total = 0
    trace_adj_inter = [0] * n_layers
    trace_adj_union = [0] * n_layers
    previous: list[list[int] | None] = [None] * n_layers
    for event_index, raw_event in enumerate(trace):
        event = _mapping(raw_event, f"trace[{event_index}]")
        _keys(event, {"layer", "experts"}, f"trace[{event_index}]")
        layer = _integer(event["layer"], f"trace[{event_index}].layer", 0, n_layers - 1)
        experts = _list(event["experts"], f"trace[{event_index}].experts")
        if not experts or len(experts) > n_experts:
            raise ValueError(f"trace[{event_index}].experts has invalid length")
        trace_id_total += len(experts)
        if trace_id_total > MAX_TRACE_IDS:
            raise ValueError(f"trace exceeds the {MAX_TRACE_IDS}-ID producer/consumer contract")
        last = -1
        checked: list[int] = []
        for j, expert_value in enumerate(experts):
            expert = _integer(expert_value, f"trace[{event_index}].experts[{j}]", 0, n_experts - 1)
            if expert <= last:
                raise ValueError(f"trace[{event_index}].experts must be strictly ascending")
            last = expert
            checked.append(expert)
            trace_select[layer][expert] += 1
        trace_layer_events[layer] += 1
        if previous[layer] is not None:
            old = previous[layer]
            intersection = len(set(old).intersection(checked))
            trace_adj_inter[layer] += intersection
            trace_adj_union[layer] += len(old) + len(checked) - intersection
        previous[layer] = checked

    layer_rows = _list(root["layers"], "layers")
    if len(layer_rows) != n_layers:
        raise ValueError("layers list does not match geometry.layers")
    layer_event_sum = 0
    for layer, raw_layer in enumerate(layer_rows):
        layer_row = _mapping(raw_layer, f"layers[{layer}]")
        layer_required = {"layer", "event_count", "active_expert_uniqueness",
                          "adjacent_intersection_count", "adjacent_union_count", "experts"}
        _keys(layer_row, layer_required, f"layers[{layer}]")
        if _integer(layer_row["layer"], f"layers[{layer}].layer", 0, n_layers - 1) != layer:
            raise ValueError("layers must be in ascending numeric order")
        event_count = _integer(layer_row["event_count"], f"layers[{layer}].event_count")
        layer_event_sum = _checked_add(layer_event_sum, event_count, "layer event count")
        uniqueness = _integer(layer_row["active_expert_uniqueness"],
                              f"layers[{layer}].active_expert_uniqueness", 0, n_experts)
        adjacent_intersection = _integer(layer_row["adjacent_intersection_count"],
                                         f"layers[{layer}].adjacent_intersection_count")
        adjacent_union = _integer(layer_row["adjacent_union_count"],
                                  f"layers[{layer}].adjacent_union_count")
        if adjacent_intersection > adjacent_union:
            raise ValueError(f"layers[{layer}] adjacent intersection exceeds union")
        if (adjacent_intersection < trace_adj_inter[layer] or
                adjacent_union < trace_adj_union[layer]):
            raise ValueError(f"layers[{layer}] adjacency counters are below held trace")
        adjacent_pairs = event_count - 1 if event_count else 0
        if adjacent_union > adjacent_pairs * n_experts:
            raise ValueError(f"layers[{layer}] adjacent union exceeds geometry bound")
        if event_count < trace_layer_events[layer]:
            raise ValueError(f"layers[{layer}].event_count is below held trace")

        experts = _list(layer_row["experts"], f"layers[{layer}].experts")
        if len(experts) != n_experts:
            raise ValueError(f"layers[{layer}].experts does not match geometry.experts")
        positive = 0
        for expert, raw_expert in enumerate(experts):
            expert_row = _mapping(raw_expert, f"layers[{layer}].experts[{expert}]")
            _keys(expert_row, {"expert", "selection_events", "selected_rows"},
                  f"layers[{layer}].experts[{expert}]")
            if _integer(expert_row["expert"], f"layers[{layer}].experts[{expert}].expert",
                        0, n_experts - 1) != expert:
                raise ValueError("expert rows must be in ascending numeric order")
            selected_events = _integer(expert_row["selection_events"],
                                       f"layers[{layer}].experts[{expert}].selection_events")
            selected_rows = _integer(expert_row["selected_rows"],
                                     f"layers[{layer}].experts[{expert}].selected_rows")
            if selected_events > event_count or selected_rows < selected_events:
                raise ValueError(f"invalid counters for layer {layer}, expert {expert}")
            if selected_events < trace_select[layer][expert]:
                raise ValueError(f"selection counter below held trace for layer {layer}, expert {expert}")
            positive += selected_events != 0
        if uniqueness != positive:
            raise ValueError(f"layers[{layer}] uniqueness disagrees with expert counters")
        if events_dropped == 0:
            if event_count != trace_layer_events[layer]:
                raise ValueError(f"layers[{layer}].event_count disagrees with complete trace")
            if adjacent_intersection != trace_adj_inter[layer] or adjacent_union != trace_adj_union[layer]:
                raise ValueError(f"layers[{layer}] adjacency counters disagree with complete trace")
            for expert, expert_row in enumerate(experts):
                if expert_row["selection_events"] != trace_select[layer][expert]:
                    raise ValueError(f"selection counter disagrees with complete trace at {layer}/{expert}")
    if layer_event_sum != events_total:
        raise ValueError("sum of layer event counts does not match recorded+dropped events")
    if pair_count <= 0:  # defensive: multiplication above is deliberately bounded
        raise ValueError("empty geometry")
    return root


def _validate_capacities(capacities: list[int]) -> None:
    if not isinstance(capacities, list) or not capacities:
        raise ValueError("capacities must be a non-empty list")
    previous = 0
    for i, capacity in enumerate(capacities):
        value = _integer(capacity, f"capacities[{i}]", 1, MAX_CAPACITY)
        if value <= previous:
            raise ValueError("capacities must be strictly increasing and unique")
        previous = value


def _replay(trace: list[dict[str, Any]], capacity: int) -> dict[str, int]:
    # OrderedDict oldest→newest reproduces the streamer's monotonic-age global LRU. The existing
    # >slots deterministic chunk path does not retain cache state, so such an event is all misses
    # and clears residency for the simulated capacity.
    resident: OrderedDict[tuple[int, int], None] = OrderedDict()
    hits = misses = fallback_events = 0
    for event in trace:
        active = [(event["layer"], expert) for expert in event["experts"]]
        if len(active) > capacity:
            misses += len(active)
            fallback_events += 1
            resident.clear()
            continue
        missing = []
        for pair in active:  # streamer scans hits for the entire event before loading misses
            if pair in resident:
                hits += 1
                resident.move_to_end(pair)
            else:
                missing.append(pair)
        for pair in missing:
            misses += 1
            if len(resident) == capacity:
                resident.popitem(last=False)
            resident[pair] = None
    return {"capacity": capacity, "hits": hits, "misses": misses,
            "accesses": hits + misses, "fallback_events": fallback_events}


def build_report(profile: dict[str, Any], capacities: list[int]) -> dict[str, Any]:
    _validate_capacities(capacities)
    return {
        "format": REPORT_FORMAT,
        "profile_format": PROFILE_FORMAT,
        "trace_events": profile["events_recorded"],
        "dropped_events": profile["events_dropped"],
        "capacity_curves": [_replay(profile["trace"], capacity) for capacity in capacities],
    }


def build_plan(profile: dict[str, Any], slots: int, source: str,
               source_sha256: str) -> dict[str, Any]:
    n_layers = profile["geometry"]["layers"]
    n_experts = profile["geometry"]["experts"]
    pair_count = n_layers * n_experts
    slots = _integer(slots, "slots", 1, min(MAX_CAPACITY, pair_count))
    if not isinstance(source, str) or not isinstance(source_sha256, str):
        raise ValueError("source metadata must be strings")

    ranked = []
    for layer_row in profile["layers"]:
        layer = layer_row["layer"]
        for expert_row in layer_row["experts"]:
            ranked.append((layer, expert_row["expert"], expert_row["selection_events"],
                           expert_row["selected_rows"]))
    ranked.sort(key=lambda row: (-row[2], -row[3], row[0], row[1]))
    hot = {(layer, expert) for layer, expert, _, _ in ranked[:slots]}
    rank_of = {(layer, expert): rank for rank, (layer, expert, _, _) in enumerate(ranked)}

    placement: dict[str, Any] = {}
    for layer, expert, selected_events, selected_rows in ranked:
        rank = rank_of[(layer, expert)]
        placement[f"layers.{layer}.experts.{expert}"] = {
            "tier": "vram" if (layer, expert) in hot else "ssd",
            # q35_seed_ssd_residency parses this numeric field and sorts descending. A bounded
            # ordinal avoids floating-point collapse for overflow-sized real-world counters.
            "importance": pair_count - rank,
            "selection_events": selected_events,
            "selected_rows": selected_rows,
        }
    return {
        "format": PLAN_FORMAT,
        "source": source,
        "source_sha256": source_sha256,
        "profile_format": PROFILE_FORMAT,
        "geometry": {"layers": n_layers, "experts": n_experts},
        "configured_slots": slots,
        "ranking": ["selection_events_desc", "selected_rows_desc", "layer_asc", "expert_asc"],
        "expert_counts": {"vram": slots, "host": 0, "ssd": pair_count - slots},
        "trace_events": profile["events_recorded"],
        "dropped_events": profile["events_dropped"],
        "placement": placement,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    parent = path.parent
    temp = parent / f"{path.name}.tmp.{os.getpid()}"
    payload = (json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n").encode()
    fd = -1
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as out:
            fd = -1
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
        try:
            dir_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _capacities(value: str) -> list[int]:
    try:
        result = [int(item, 10) for item in value.split(",") if item != ""]
        _validate_capacities(result)
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a fucina SSD expert profile and generate a deterministic hotlist")
    parser.add_argument("profile", type=Path)
    parser.add_argument("--slots", type=int, required=True,
                        help="number of top-ranked (layer,expert) pairs assigned to vram")
    parser.add_argument("--capacities", type=_capacities, required=True,
                        help="strictly increasing comma-separated global LRU capacities")
    parser.add_argument("--out-plan", type=Path, required=True)
    parser.add_argument("--out-report", type=Path)
    args = parser.parse_args()
    try:
        source_bytes = _read_bounded(args.profile)
        profile = load_profile(args.profile, raw=source_bytes)
        plan = build_plan(profile, args.slots, str(args.profile), hashlib.sha256(source_bytes).hexdigest())
        report = build_report(profile, args.capacities)
        write_json_atomic(args.out_plan, plan)
        if args.out_report:
            write_json_atomic(args.out_report, report)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"wrote {args.out_plan}: slots={args.slots} trace={profile['events_recorded']} "
          f"dropped={profile['events_dropped']}")
    if args.out_report:
        print(f"wrote {args.out_report}: capacities={args.capacities}")


if __name__ == "__main__":
    main()
