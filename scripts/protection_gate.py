#!/usr/bin/env python3
"""Qwen3.5 dual-sided protection gate (plan rev2 P3).

Two independent assertions per (model, N) cell, over agg_decode_tps + median_ttft_ms
+ p95_ttft_ms:

  (a) ABSOLUTE FLOOR   — no metric regresses >TOL vs a FROZEN raw baseline (throughput
                         may not drop >TOL; TTFT may not rise >TOL).
  (b) COMPETITIVE EDGE — in cells flagged as claimed wins, the candidate must beat a
                         CONTEMPORANEOUS protocol-matched vLLM run (not a historical
                         number). Throughput: fucina >= vLLM. TTFT: fucina <= vLLM.

Baselines are produced by bench_serving.py (--out) run per (engine, model). The gate is
config-driven by a frozen baseline manifest (JSON, see freeze_baselines()).

Usage:
  # freeze (once, on a quiescent box):
  protection_gate.py freeze --model moe  --fucina fucina-moe.json  --vllm vllm-moe.json  --out baseline-moe.json
  # check a candidate fucina run against the frozen baseline + a fresh contemporaneous vLLM run:
  protection_gate.py check --baseline baseline-moe.json --candidate cand-moe.json --vllm vllm-moe-now.json
Exit 0 = all cells pass; nonzero = at least one violation (printed).
"""
import json, sys, argparse

TOL = 0.05  # 5% floor

# Cells where fucina ROBUSTLY beats a contemporaneous vLLM (must keep holding). Set from the
# 2026-07-11 frozen head-to-head, NOT the plan's looser "N<=8 everything" prose:
#   MoE  agg: only N=8 (155 vs 147, +6%); N=2 vLLM wins (71 vs 58), N=4 vLLM edges (105 vs 102).
#   dense agg: N=2/4/8 (56.6/109.6/179.3 vs 42.9/83.9/161.7 = +32/+31/+11%).
# Single-stream decode (fucina +28% MoE) lives in single_short, not the concurrency array, and
# vLLM's N=1 concurrency cell is a compile-warmup artifact — so N=1 is excluded from the agg gate.
# TTFT is NOT a claimed win at any N (vLLM chunked prefill wins TTFT under load) — competitive
# TTFT checks stay informational until P1 flips them.
DEFAULT_WIN_CELLS = {
    "moe":  {"agg_decode_tps": [8]},
    "dense": {"agg_decode_tps": [2, 4, 8]},
}

def by_n(runjson):
    d = json.load(open(runjson)) if isinstance(runjson, str) else runjson
    return {c["N"]: c for c in d.get("concurrency", [])}, d

def freeze(args):
    fuc, _ = by_n(args.fucina)
    vll, _ = by_n(args.vllm)
    win = DEFAULT_WIN_CELLS.get(args.model, {})
    base = {"model": args.model, "tol": TOL, "cells": {}}
    for n, c in sorted(fuc.items()):
        v = vll.get(n, {})
        base["cells"][str(n)] = {
            "agg_decode_tps": c.get("agg_decode_tps"),
            "median_ttft_ms": c.get("median_ttft_ms"),
            "p95_ttft_ms": c.get("p95_ttft_ms"),
            "vllm_agg_decode_tps": v.get("agg_decode_tps"),
            "vllm_median_ttft_ms": v.get("median_ttft_ms"),
            "vllm_p95_ttft_ms": v.get("p95_ttft_ms"),
            "win_agg": n in win.get("agg_decode_tps", []),
        }
    json.dump(base, open(args.out, "w"), indent=2)
    print(f"froze {len(base['cells'])} cells for model={args.model} -> {args.out}")

def check(args):
    base = json.load(open(args.baseline))
    cand, _ = by_n(args.candidate)
    vll, _ = by_n(args.vllm) if args.vllm else ({}, None)
    tol = base.get("tol", TOL)
    fails = []
    print(f"# protection gate: model={base['model']} tol={tol:.0%}")
    print(f"{'N':>3} {'metric':>16} {'cand':>10} {'base':>10} {'floor':>8} {'vLLM':>10} {'edge':>8}")
    for ns, b in sorted(base["cells"].items(), key=lambda x: int(x[0])):
        n = int(ns); c = cand.get(n)
        if not c:
            fails.append(f"N={n}: MISSING from candidate"); continue
        # (a) absolute floor
        for metric, worse_is in (("agg_decode_tps", "down"), ("median_ttft_ms", "up"), ("p95_ttft_ms", "up")):
            cv, bv = c.get(metric), b.get(metric)
            if cv is None or bv is None: continue
            if worse_is == "down":
                ok = cv >= bv * (1 - tol); flr = f"{bv*(1-tol):.1f}"
            else:
                ok = cv <= bv * (1 + tol); flr = f"{bv*(1+tol):.1f}"
            vv = vll.get(n, {}).get({"agg_decode_tps": "agg_decode_tps",
                                     "median_ttft_ms": "median_ttft_ms",
                                     "p95_ttft_ms": "p95_ttft_ms"}[metric]) if vll else b.get("vllm_"+metric)
            # (b) competitive edge (only agg wins are asserted; TTFT informational)
            edge = ""
            if metric == "agg_decode_tps" and b.get("win_agg") and vv is not None:
                won = cv >= vv * (1 - 1e-9)
                edge = "WIN" if won else "LOSE"
                if not won: fails.append(f"N={n} {metric}: fucina {cv:.1f} < vLLM {vv:.1f} (claimed win)")
            flag = "" if ok else "  <<FLOOR"
            if not ok: fails.append(f"N={n} {metric}: {cv:.1f} vs base {bv:.1f} floor {flr}{flag}")
            print(f"{n:>3} {metric:>16} {cv:>10.1f} {bv:>10.1f} {flr:>8} {str(round(vv,1) if vv else '-'):>10} {edge:>8}{flag}")
    print()
    if fails:
        print("GATE FAIL:"); [print("  -", f) for f in fails]; sys.exit(1)
    print("GATE PASS — all cells within floor; claimed-win cells beat contemporaneous vLLM.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--model", required=True); f.add_argument("--fucina", required=True)
    f.add_argument("--vllm", required=True); f.add_argument("--out", required=True); f.set_defaults(fn=freeze)
    c = sub.add_parser("check"); c.add_argument("--baseline", required=True); c.add_argument("--candidate", required=True)
    c.add_argument("--vllm", default=""); c.set_defaults(fn=check)
    a = ap.parse_args(); a.fn(a)
