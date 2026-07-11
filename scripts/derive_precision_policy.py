#!/usr/bin/env python3
# ABOUTME: Converts a fucina importance sidecar into a capability-gated precision policy.
# ABOUTME: Keeps unsupported sub-4-bit codecs out of runnable policies by construction.
"""Derive Phase-B precision tiers from measured routing and activation importance."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def expert_identity(name: str) -> tuple[int, int] | None:
    parts = name.split(".")
    try:
        li, ei = parts.index("layers"), parts.index("experts")
        return int(parts[li + 1]), int(parts[ei + 1])
    except (ValueError, IndexError):
        return None


def derive(sidecar_path: Path, sub4_kernel: bool = False) -> dict[str, Any]:
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("format") != "fucina-imatrix-v1":
        raise ValueError("unsupported importance sidecar format")
    importance = {k: float(v) for k, v in sidecar["tensor_importance"].items()}
    by_expert: dict[tuple[int, int], list[float]] = {}
    for name, score in importance.items():
        ident = expert_identity(name)
        if ident is not None:
            by_expert.setdefault(ident, []).append(score)
    expert_scores = {k: sum(v) / len(v) for k, v in by_expert.items()}
    cold_cut = quantile(list(expert_scores.values()), 0.20)
    hot_cut = quantile(list(expert_scores.values()), 0.80)
    tensor_policy: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for name, score in importance.items():
        ident = expert_identity(name)
        if ident is None:
            codec, tier, reason = "fp8_block", "critical", "attention/deltanet/shared/router/norm"
        else:
            es = expert_scores[ident]
            if es >= hot_cut:
                codec, tier, reason = "nvfp4", "hot", "top importance quintile"
            elif es <= cold_cut:
                codec = "int2" if sub4_kernel else "nvfp4"
                tier = "cold"
                reason = "bottom importance quintile" + ("; sub4 kernel enabled" if sub4_kernel else "; NVFP4 safety floor")
            else:
                codec, tier, reason = "nvfp4", "warm", "middle importance band"
        tensor_policy[name] = {"codec": codec, "tier": tier, "importance": score, "reason": reason}
        counts[codec] = counts.get(codec, 0) + 1
    return {
        "format": "fucina-precision-policy-v1",
        "source": str(sidecar_path),
        "source_sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
        "model": sidecar.get("model", ""),
        "thresholds": {"cold_q20": cold_cut, "hot_q80": hot_cut},
        "capabilities": {"nvfp4": True, "fp8_block": True, "int2": sub4_kernel},
        "codec_tensor_counts": dict(sorted(counts.items())),
        "tensor_policy": tensor_policy,
        "accuracy_gate_required": True,
        "notes": [
            "Critical/shared tensors never inherit routed-expert compression.",
            "Cold experts remain NVFP4 unless --sub4-kernel explicitly asserts a tested kernel.",
            "A policy is declarative until the checkpoint converter/loader applies it and quality gates pass."
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sidecar", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sub4-kernel", action="store_true", help="allow INT2 only after a tested kernel exists")
    args = ap.parse_args()
    policy = derive(args.sidecar, args.sub4_kernel)
    args.out.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {policy['codec_tensor_counts']}")


if __name__ == "__main__":
    main()
