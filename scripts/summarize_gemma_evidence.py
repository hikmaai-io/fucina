#!/usr/bin/env python3
"""Aggregate raw Gemma evidence JSON without inventing missing cells."""
from __future__ import annotations
import json, math, statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "benchmark-evidence/results/2026-07-20-gemma-gb10/raw"
OUT = ROOT / "benchmark-evidence/results/2026-07-20-gemma-gb10/summary.json"


def summary(xs: list[float]) -> dict[str, Any]:
    ys = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not ys: return {"n": 0, "raw": [], "median": None, "mean": None, "cv": None}
    mean = statistics.fmean(ys)
    return {"n": len(ys), "raw": ys, "median": statistics.median(ys), "mean": mean,
            "cv": statistics.stdev(ys)/mean if len(ys)>1 and mean else 0.0}


def main() -> None:
    groups: dict[str,list[dict[str,Any]]] = defaultdict(list)
    for p in sorted(RAW.glob("*.json")):
        try: r=json.loads(p.read_text())
        except Exception: continue
        groups[r["mode"]].append(r)
    out: dict[str,Any] = {"schema_version":1,"modes":{},"comparisons":{}}
    for mode,runs in groups.items():
        m: dict[str,Any]={"starts":len(runs),
                         "stream_interval_semantics": ("token-exact" if runs[0].get("engine")=="fucina" else "SSE-content-chunk; not guaranteed token ITL"),
                         "startup_ms":summary([r.get("startup_ms") for r in runs]),
                         "rss_at_ready_bytes":summary([r.get("first_ready_rss_bytes") for r in runs]),
                         "physical_available_at_ready_bytes":summary([r.get("physical_available_bytes_at_ready") for r in runs]),
                         "quality":{},"concurrency":{},"long_3500":{},"context":{},"prefix":{},"mixed":{}}
        qh: dict[str,list[str]]=defaultdict(list); qe: dict[str,list[str]]=defaultdict(list); qo: dict[str,list[str]]=defaultdict(list)
        for r in runs:
            for q in r.get("quality",{}).get("cases",[]):
                qh[q["case"]].append(q["output_sha256"]); qo[q["case"]].append(q["output_utf8"])
                if q.get("token_event_sha256"): qe[q["case"]].append(q["token_event_sha256"])
        for case,hashes in qh.items():
            m["quality"][case]={"output_hashes":hashes,"stable_across_starts":len(set(hashes))==1,
                                  "token_event_hashes":qe[case],
                                  "token_events_stable_where_captured":len(set(qe[case]))<=1,"outputs":qo[case]}
        byn: dict[int,list[dict[str,Any]]]=defaultdict(list)
        for r in runs:
            for row in r.get("performance",{}).get("concurrency",[]): byn[int(row["N"])].append(row)
        for n,rows in sorted(byn.items()):
            m["concurrency"][str(n)]={
                "aggregate_completion_tps":summary([x.get("aggregate_completion_tps") for x in rows]),
                "ttft_p50_ms":summary([x.get("ttft_ms",{}).get("p50") for x in rows]),
                "ttft_p95_ms":summary([x.get("ttft_ms",{}).get("p95") for x in rows]),
                "ttft_p99_ms":summary([x.get("ttft_ms",{}).get("p99") for x in rows]),
                "itl_p50_ms":summary([x.get("itl_ms",{}).get("p50") for x in rows]),
                "itl_p95_ms":summary([x.get("itl_ms",{}).get("p95") for x in rows]),
                "itl_p99_ms":summary([x.get("itl_ms",{}).get("p99") for x in rows]),
                "per_stream_decode_tps":summary([x.get("per_stream_decode_tps",{}).get("p50") for x in rows]),
                "failures":sum(int(x.get("failures",0)) for x in rows),
                "all_requests_128":all(all(req.get("completion_tokens")==128 for req in x.get("requests",[]) if req.get("ok")) for x in rows),
            }
        for r in runs:
            for row in r.get("performance",{}).get("long_3500",[]):
                m["long_3500"].setdefault("ttft",[]).append(row.get("ttft_ms",{}).get("p50"))
                m["long_3500"].setdefault("wall",[]).append(row.get("wall_s"))
                usages=[q.get("usage",{}).get("prompt_tokens") for q in row.get("requests",[])]
                m["long_3500"].setdefault("prompt_tokens",[]).extend([u for u in usages if u is not None and u > 0])
            for row in r.get("performance",{}).get("context",[]):
                label=row["label"]
                m["context"].setdefault(label,{"ttft":[],"wall":[],"prompt_tokens":[]})
                m["context"][label]["ttft"].append(row.get("ttft_ms",{}).get("p50"))
                m["context"][label]["wall"].append(row.get("wall_s"))
                m["context"][label]["prompt_tokens"].extend([q.get("usage",{}).get("prompt_tokens") for q in row.get("requests",[]) if (q.get("usage",{}).get("prompt_tokens") or 0) > 0])
            pref=r.get("performance",{}).get("prefix",{})
            for side in ("cold","warm"):
                if side in pref: m["prefix"].setdefault(side,[]).append(pref[side].get("ttft_ms",{}).get("p50"))
            mixed=r.get("performance",{}).get("mixed",{})
            for side in ("active_decode","arriving_prefill"):
                if side in mixed: m["mixed"].setdefault(side,[]).append(mixed[side].get("ttft_ms",{}).get("p50"))
        if m["long_3500"]:
            m["long_3500"]={k:summary(v) for k,v in m["long_3500"].items()}
        m["context"]={lab:{k:summary(v) for k,v in vals.items()} for lab,vals in m["context"].items()}
        m["prefix"]={k:summary(v) for k,v in m["prefix"].items()}
        m["mixed"]={k:summary(v) for k,v in m["mixed"].items()}
        out["modes"][mode]=m
    plain=out["modes"].get("fucina-dense-q4-plain")
    mtp=out["modes"].get("fucina-dense-q4-mtp")
    if plain and mtp:
        cmp={"concurrency":{},"quality_exact":{}}
        for n in sorted(set(plain["concurrency"]) & set(mtp["concurrency"]),key=int):
            p=plain["concurrency"][n]["aggregate_completion_tps"]["median"]
            q=mtp["concurrency"][n]["aggregate_completion_tps"]["median"]
            cmp["concurrency"][n]={"plain":p,"mtp":q,"relative":q/p-1 if p and q is not None else None}
        for case in sorted(set(plain["quality"]) & set(mtp["quality"])):
            cmp["quality_exact"][case]=plain["quality"][case]["output_hashes"]==mtp["quality"][case]["output_hashes"]
        out["comparisons"]["fucina_q4_mtp_vs_plain"]=cmp
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
    print(OUT)
if __name__=="__main__": main()
