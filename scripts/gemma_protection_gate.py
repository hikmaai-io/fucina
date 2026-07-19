#!/usr/bin/env python3
"""Report-only Gemma protection checker for raw three-start evidence.

Default behavior never turns noisy or incomplete data into a merge gate. Pass
--enforce only after the report says every protected cell has >=3 starts and
CV <= --max-cv; then a regression beyond --floor fails the command.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any


def load(pattern: str) -> list[dict[str, Any]]:
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(pattern))]


def cells(runs: list[dict[str, Any]]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for run in runs:
        for row in run.get("performance", {}).get("concurrency", []):
            if row.get("failures") == 0 and row.get("aggregate_completion_tps") is not None:
                out.setdefault(int(row["N"]), []).append(float(row["aggregate_completion_tps"]))
    return out


def summarize(values: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(values) if values else None
    cv = statistics.stdev(values) / mean if len(values) > 1 and mean else (0.0 if values else None)
    return {"raw": values, "n": len(values), "median": statistics.median(values) if values else None, "cv": cv}


def quality_hashes(runs: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for run in runs:
        for case in run.get("quality", {}).get("cases", []):
            out.setdefault(case["case"], []).append(case["output_sha256"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="glob for baseline raw JSON")
    ap.add_argument("--candidate", required=True, help="glob for candidate raw JSON")
    ap.add_argument("--floor", type=float, default=.05, help="maximum protected median regression")
    ap.add_argument("--max-cv", type=float, default=.05, help="maximum CV for enforceable cells")
    ap.add_argument("--enforce", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    base, cand = load(args.baseline), load(args.candidate)
    bc, cc = cells(base), cells(cand)
    report: dict[str, Any] = {
        "mode": "enforce" if args.enforce else "report-only",
        "baseline_files": len(base), "candidate_files": len(cand), "floor": args.floor,
        "max_cv": args.max_cv, "cells": {}, "failures": [], "not_enforceable": [],
        "quality": {"baseline": quality_hashes(base), "candidate": quality_hashes(cand)},
    }
    for n in sorted(set(bc) | set(cc)):
        bs, cs = summarize(bc.get(n, [])), summarize(cc.get(n, []))
        delta = None
        if bs["median"] and cs["median"] is not None:
            delta = cs["median"] / bs["median"] - 1
        enforceable = (bs["n"] >= 3 and cs["n"] >= 3 and bs["cv"] is not None and
                       cs["cv"] is not None and bs["cv"] <= args.max_cv and cs["cv"] <= args.max_cv)
        report["cells"][str(n)] = {"baseline": bs, "candidate": cs, "relative": delta,
                                    "enforceable": enforceable}
        if not enforceable:
            report["not_enforceable"].append(n)
        elif delta is not None and delta < -args.floor:
            report["failures"].append(f"N={n}: {delta:.2%} < -{args.floor:.2%}")
    # Within each mode, deterministic quality cases must have one hash across starts.
    for side in ("baseline", "candidate"):
        for case, hashes in report["quality"][side].items():
            if len(set(hashes)) > 1:
                report["failures"].append(f"{side} quality hash unstable: {case}")
    report["pass"] = not report["failures"]
    report["enforcement_allowed"] = not report["not_enforceable"] and bool(report["cells"])
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    if args.enforce and (not report["enforcement_allowed"] or report["failures"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
