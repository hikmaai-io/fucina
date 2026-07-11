#!/usr/bin/env python3
# ABOUTME: Compares tool-eval benchmark reports and fails precision-policy quality regressions.
# ABOUTME: Makes the Phase-B accuracy gate reproducible in CI and local GB10 experiments.
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_report(path: Path) -> dict[str, float]:
    text = path.read_text()
    patterns = {
        "final_score": r"\*\*Final Score\*\*: \*\*([0-9.]+)\*\* / 100",
        "quality": r"\*\*Quality\*\*: ([0-9.]+) / 100",
        "deployability": r"\*\*Deployability\*\*: \*\*([0-9.]+)\*\* / 100",
        "error_rate": r"\| Error Rate \| ([0-9.]+) \|",
    }
    result = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"{path}: missing {name}")
        result[name] = float(match.group(1))
    return result


def compare(base: dict[str, float], candidate: dict[str, float], max_quality_drop: float,
            max_score_drop: float, max_error_increase: float) -> tuple[bool, list[str]]:
    failures = []
    if base["quality"] - candidate["quality"] > max_quality_drop:
        failures.append(f"quality drop {base['quality']-candidate['quality']:.2f} > {max_quality_drop:.2f}")
    if base["final_score"] - candidate["final_score"] > max_score_drop:
        failures.append(f"score drop {base['final_score']-candidate['final_score']:.2f} > {max_score_drop:.2f}")
    if candidate["error_rate"] - base["error_rate"] > max_error_increase:
        failures.append(f"error-rate increase {candidate['error_rate']-base['error_rate']:.4f} > {max_error_increase:.4f}")
    return not failures, failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--max-quality-drop", type=float, default=1.0)
    ap.add_argument("--max-score-drop", type=float, default=1.0)
    ap.add_argument("--max-error-increase", type=float, default=0.0)
    args = ap.parse_args()
    base, candidate = parse_report(args.baseline), parse_report(args.candidate)
    passed, failures = compare(base, candidate, args.max_quality_drop, args.max_score_drop,
                               args.max_error_increase)
    print(json.dumps({"pass": passed, "baseline": base, "candidate": candidate,
                      "failures": failures}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
