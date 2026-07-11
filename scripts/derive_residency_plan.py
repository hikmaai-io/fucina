#!/usr/bin/env python3
# ABOUTME: Converts calibration heat maps into deterministic VRAM/host/SSD expert placement.
# ABOUTME: Starts Phase C without pretending metadata planning already implements weight streaming.
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def derive(sidecar_path: Path, vram_gib: float, host_gib: float, hidden: int = 2048,
           intermediate: int = 512, bits: float = 4.5) -> dict[str, Any]:
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("format") != "fucina-imatrix-v1":
        raise ValueError("unsupported importance sidecar format")
    # Three matrices per expert: gate/up [I,H], down [H,I]. Include FP4 scale overhead in
    # the effective 4.5 bpw, matching fucina's resident grouped-NVFP4 slabs.
    expert_bytes = round(3 * hidden * intermediate * bits / 8)
    experts = []
    total_routes = 0
    for layer in sidecar["expert_heat_map"]:
        assignments = int(layer["assignments"])
        for e in layer["experts"]:
            count = int(e["count"])
            total_routes += count
            experts.append({"layer": int(layer["layer"]), "expert": int(e["expert"]),
                            "route_count": count,
                            "assignment_frequency": float(e["frequency"]),
                            "mean_router_weight": float(e["mean_weight"]),
                            "importance": float(e["importance"]),
                            "bytes": expert_bytes})
    # Heat first, confidence second, stable layer/expert tie-break. Route count is corpus-size
    # dependent but exactly what minimizes expected misses for a fixed-size expert geometry.
    experts.sort(key=lambda x: (-x["route_count"], -x["importance"], x["layer"], x["expert"]))
    budgets = {"vram": int(vram_gib * 1024**3), "host": int(host_gib * 1024**3)}
    used = {"vram": 0, "host": 0, "ssd": 0}
    route_hits = {"vram": 0, "host": 0, "ssd": 0}
    tier_counts = {"vram": 0, "host": 0, "ssd": 0}
    placement = {}
    for e in experts:
        if used["vram"] + expert_bytes <= budgets["vram"]:
            tier = "vram"
        elif used["host"] + expert_bytes <= budgets["host"]:
            tier = "host"
        else:
            tier = "ssd"
        used[tier] += expert_bytes
        route_hits[tier] += e["route_count"]
        tier_counts[tier] += 1
        placement[f"layers.{e['layer']}.experts.{e['expert']}"] = {
            "tier": tier, "bytes": expert_bytes, "route_count": e["route_count"],
            "importance": e["importance"]}
    hit = {k: (route_hits[k] / total_routes if total_routes else 0.0) for k in route_hits}
    return {
        "format": "fucina-expert-residency-v1",
        "source": str(sidecar_path),
        "source_sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
        "model": sidecar.get("model", ""),
        "geometry": {"hidden": hidden, "intermediate": intermediate,
                     "effective_bits": bits, "bytes_per_expert": expert_bytes},
        "budgets": {"expert_vram_bytes": budgets["vram"], "expert_host_bytes": budgets["host"]},
        "occupancy_bytes": used,
        "expert_counts": tier_counts,
        "calibration_route_fraction": hit,
        "placement": placement,
        "runtime_status": "planning-only; C1 store/prefetch must consume this manifest",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sidecar", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expert-vram-gib", type=float, required=True)
    ap.add_argument("--expert-host-gib", type=float, default=0)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--intermediate", type=int, default=512)
    ap.add_argument("--effective-bits", type=float, default=4.5)
    args = ap.parse_args()
    if min(args.expert_vram_gib, args.expert_host_gib) < 0:
        ap.error("budgets must be non-negative")
    plan = derive(args.sidecar, args.expert_vram_gib, args.expert_host_gib,
                  args.hidden, args.intermediate, args.effective_bits)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}: experts={plan['expert_counts']} route_fraction={plan['calibration_route_fraction']}")


if __name__ == "__main__":
    main()
