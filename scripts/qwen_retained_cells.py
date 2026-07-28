#!/usr/bin/env python3
"""Validate the 12 retained Qwen throughput cells for SOL-05.

Candidate values are medians across independently started servers. The eleven
historical winning cells are hard gates at -2%; dense N=32 is reported as the
remaining competitive-gap target rather than being mislabeled a win.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def unwrap_run(value):
    if isinstance(value, list):
        if len(value) != 1 or not isinstance(value[0], dict):
            raise ValueError("raw result array must contain exactly one bench_serving result")
        return value[0]
    if not isinstance(value, dict):
        raise ValueError("candidate must be an object or a one-object raw result array")
    return value


def cells(run):
    return {int(cell["N"]): cell for cell in run.get("concurrency", [])}


def median_runs(paths):
    runs = [cells(unwrap_run(load_json(path))) for path in paths]
    if not runs:
        raise ValueError("at least one candidate run is required")
    ns = set(runs[0])
    if any(set(run) != ns for run in runs[1:]):
        raise ValueError("candidate runs have different concurrency cells")
    out = {}
    for n in sorted(ns):
        keys = set.intersection(*(set(run[n]) for run in runs))
        out[n] = {"N": n}
        for key in keys:
            values = [run[n][key] for run in runs]
            if key != "N" and all(isinstance(v, (int, float)) for v in values):
                out[n][key] = statistics.median(values)
    return out


def validate_config(config):
    expected = set(config["concurrency"])
    if len(expected) != 6 or set(config["models"]) != {"dense", "moe"}:
        raise ValueError("config must define dense+moe and six concurrency cells")
    winning = sum(len(model["winning_cells"]) for model in config["models"].values())
    if winning != 11:
        raise ValueError(f"config must identify exactly 11 winning cells, got {winning}")
    for name, model in config["models"].items():
        if not set(model["winning_cells"]).issubset(expected):
            raise ValueError(f"{name}: winning cell outside concurrency matrix")


def check(config_path, candidates):
    root = Path(config_path).resolve().parents[2]
    config = load_json(config_path)
    validate_config(config)
    tolerance = float(config["max_winning_regression"])
    metric = config["metric"]
    expected = set(config["concurrency"])
    failures = []
    report = {"schema_version": 1, "config": str(config_path), "tolerance": tolerance, "models": {}}

    for name, model in config["models"].items():
        paths = candidates.get(name, [])
        candidate = median_runs(paths)
        if set(candidate) != expected:
            missing = sorted(expected - set(candidate))
            extra = sorted(set(candidate) - expected)
            raise ValueError(f"{name}: expected six retained cells; missing={missing} extra={extra}")
        baseline = cells(load_json(root / model["baseline"]))
        vllm = cells(load_json(root / model["vllm_reference"]))
        rows = []
        for n in sorted(expected):
            cand = float(candidate[n][metric])
            base = float(baseline[n][metric])
            ref = float(vllm[n][metric])
            delta = cand / base - 1.0
            winning = n in model["winning_cells"]
            passed = not winning or delta + 1e-12 >= -tolerance
            if not passed:
                failures.append(
                    f"{name} N={n}: {cand:.3f} < {(1-tolerance)*base:.3f} "
                    f"({delta:.2%} vs retained baseline)"
                )
            rows.append({
                "N": n,
                "candidate": cand,
                "baseline": base,
                "delta": delta,
                "vllm_reference": ref,
                "competitive_gap": cand / ref - 1.0,
                "winning_cell": winning,
                "passed": passed,
                "median_ttft_ms": candidate[n].get("median_ttft_ms"),
                "p95_ttft_ms": candidate[n].get("p95_ttft_ms"),
            })
        report["models"][name] = {"candidate_runs": [str(p) for p in paths], "cells": rows}

    report["passed"] = not failures
    report["failures"] = failures
    return report


def print_report(report):
    tol = report["tolerance"]
    print(f"# SOL-05 retained Qwen gate (winning-cell floor: -{tol:.0%})")
    print(f"{'model':>6} {'N':>3} {'candidate':>11} {'baseline':>10} {'delta':>8} {'vLLM':>10} {'gap':>8} {'gate':>6}")
    for name, model in report["models"].items():
        for row in model["cells"]:
            gate = "PASS" if row["passed"] else "FAIL"
            if not row["winning_cell"]:
                gate = "TARGET"
            print(f"{name:>6} {row['N']:>3} {row['candidate']:>11.2f} {row['baseline']:>10.2f} "
                  f"{row['delta']:>8.2%} {row['vllm_reference']:>10.2f} "
                  f"{row['competitive_gap']:>8.2%} {gate:>6}")
    if report["failures"]:
        print("\nGATE FAIL", file=sys.stderr)
        for failure in report["failures"]:
            print(f"  - {failure}", file=sys.stderr)
    else:
        print("\nGATE PASS — all 11 retained winning cells are within the 2% floor.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check_parser = sub.add_parser("check", help="median candidate runs and enforce the 11-cell floor")
    check_parser.add_argument("--config", required=True)
    check_parser.add_argument("--candidate", action="append", default=[], metavar="MODEL=JSON")
    check_parser.add_argument("--json-out", default="")
    wrap = sub.add_parser("wrap", help="wrap one bench_serving object as qualification raw array")
    wrap.add_argument("--input", required=True)
    wrap.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "wrap":
        value = unwrap_run(load_json(args.input))
        Path(args.output).write_text(json.dumps([value], indent=2) + "\n")
        return 0

    candidates = {"dense": [], "moe": []}
    for item in args.candidate:
        if "=" not in item:
            parser.error("--candidate must be MODEL=JSON")
        model, path = item.split("=", 1)
        if model not in candidates:
            parser.error(f"unknown candidate model {model!r}")
        candidates[model].append(path)
    missing = [name for name, paths in candidates.items() if not paths]
    if missing:
        parser.error(f"missing candidates for: {', '.join(missing)}")
    try:
        report = check(args.config, candidates)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print_report(report)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
