#!/usr/bin/env python3
"""
sensitivity_sweep.py — Stage B2: per-tensor sensitivity sweep.

Sweeps EVERY quantizable weight tensor in the 31B Q4_0 GGUF, quantizing each
with the B1-winning 2-bit format (NF2 codebook). For each tensor we compute the
reconstruction error of uniform 2-bit MXFP2 vs the Q4_0/Q4_1/Q4_K reference:
relative Frobenius MSE, RMS error, max-abs error, cosine similarity, SQNR.

We report BOTH encoders:
  - nf2 absmax  : the cheap NVFP4-style absmax-derived block scale (ships as-is)
  - nf2 mse     : per-block MSE-optimal scale search (achievable floor)

Quantizable types present: Q4_0 (most), Q4_1 (7 early ffn_down), Q4_K (token_embd).
The F32 tensors are RMSNorm gains / per-layer scalars / rope_freqs — they are NOT
matmul weights, are tiny (5376 or fewer elems), and are never candidates for 2-bit
quantization, so they are excluded from the sweep (reported in a separate count).

Outputs:
  sensitivity_table.csv   — one row per tensor (machine readable)
  sensitivity_summary.md  — ranked worst tensors, role/layer aggregates, verdict
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict

import numpy as np

import gguf_reader as G
import mxfp2 as M

DEFAULT_GGUF = ("/opt/spark/models/hub/models--unsloth--gemma-4-31B-it-GGUF/"
                "snapshots/8906b3db2e669a0b1d6293c315d3f9fbf934a86d/"
                "gemma-4-31B-it-Q4_0.gguf")

QUANT_TYPES = {G.GGML_TYPE_Q4_0, G.GGML_TYPE_Q4_1, G.GGML_TYPE_Q4_K}


def classify(name):
    """Return (role, layer) for a tensor name. layer = -1 for non-block."""
    m = re.match(r"blk\.(\d+)\.(.+)\.weight", name)
    if m:
        layer = int(m.group(1))
        sub = m.group(2)
        role_map = {
            "attn_q": "attn_q", "attn_k": "attn_k", "attn_v": "attn_v",
            "attn_output": "attn_o",
            "ffn_gate": "ffn_gate", "ffn_up": "ffn_up", "ffn_down": "ffn_down",
        }
        return role_map.get(sub, sub), layer
    if name == "token_embd.weight":
        return "embed", -1
    if name == "output.weight":
        return "output", -1
    return "other", -1


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--variant", default="nf2", choices=["nf2", "e1m0"])
    ap.add_argument("--mse-sample", action="store_true", default=True,
                    help="also report the MSE-optimal floor on ONE tensor per role "
                         "(the full MSE encoder is ~24x slower; absmax runs on all)")
    ap.add_argument("--no-mse", dest="mse_sample", action="store_false")
    ap.add_argument("--limit", type=int, default=0,
                    help="only sweep first N quantizable tensors (debug)")
    ap.add_argument("--outdir", default="scripts/quant2bit")
    args = ap.parse_args(argv[1:])

    r = G.GGUFReader(args.gguf)
    # collect quantizable tensors in file order
    qtensors = [(name, r.tensors[name]) for name in r._order
                if r.tensors[name].ggml_type in QUANT_TYPES]
    n_f32 = sum(1 for n in r._order if r.tensors[n].ggml_type == G.GGML_TYPE_F32)
    if args.limit:
        qtensors = qtensors[:args.limit]

    print("Sweeping %d quantizable weight tensors (variant=%s, mse_sample=%s)."
          % (len(qtensors), args.variant, args.mse_sample), flush=True)
    print("(%d F32 norm/scalar tensors excluded — not matmul weights.)" % n_f32,
          flush=True)

    # one representative tensor per role for the (slow) MSE-optimal floor.
    mse_floor = {}        # role -> {"name","absmax_rel_mse","mse_rel_mse",...}
    seen_roles = set()
    MSE_SAMPLE_CAP = 16_000_000  # cap elems fed to the MSE grid (block-aligned)
    # rel_mse/cos/sqnr of a per-block quantizer are sampling-invariant under
    # block-aligned subsampling; 8M elems = 512k independent 16-elt blocks gives
    # a rock-stable estimate while keeping each tensor's measurement ~1s.
    ABSMAX_CAP = 8_000_000       # cap elems for the absmax error measurement

    rows = []
    t0 = time.time()
    for i, (name, ti) in enumerate(qtensors):
        role, layer = classify(name)
        src = G.GGML_TYPE_NAME.get(ti.ggml_type, ti.ggml_type)
        w = r.dequant(name)
        rms_w = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)))

        # The absmax encoder builds a [nb,16,4] broadcast; for very large
        # tensors (token_embd: 1.4B elems) that is tens of GB and minutes.
        # rel_mse / cos / sqnr of a PER-BLOCK quantizer are sampling-invariant
        # under block-aligned subsampling (each 16-elt block is independent), so
        # we cap the measurement at ABSMAX_CAP block-aligned elements. max_abs
        # may slightly underestimate but the ranking metrics are exact in
        # expectation.
        we = w
        sampled = False
        if we.size > ABSMAX_CAP:
            nb = ABSMAX_CAP // M.BLOCK
            we = w[: nb * M.BLOCK]
            sampled = True
        rec = M.roundtrip(we, args.variant)
        e = M.errors(we, rec)
        row = {
            "name": name, "role": role, "layer": layer, "src_type": src,
            "n_elem": int(ti.n_elem), "n_measured": int(we.size),
            "sampled": sampled, "rms_w": rms_w,
            "absmax_rel_mse": e["rel_mse"], "absmax_rms_err": e["rms_err"],
            "absmax_max_abs_err": e["max_abs_err"], "absmax_cos": e["cos_sim"],
            "absmax_sqnr_db": e["sqnr_db"],
        }
        rows.append(row)

        # MSE floor on the first tensor seen of each role (subsampled if huge)
        if args.mse_sample and role not in seen_roles:
            seen_roles.add(role)
            ws = w
            if ws.size > MSE_SAMPLE_CAP:
                nb = MSE_SAMPLE_CAP // M.BLOCK
                ws = w[: nb * M.BLOCK]
            recm = M.roundtrip_mse(ws, args.variant)
            em = M.errors(ws, recm)
            ea = M.errors(ws, M.roundtrip(ws, args.variant))
            mse_floor[role] = {
                "name": name, "n_used": int(ws.size),
                "absmax_rel_mse": ea["rel_mse"], "mse_rel_mse": em["rel_mse"],
                "absmax_sqnr_db": ea["sqnr_db"], "mse_sqnr_db": em["sqnr_db"],
                "mse_cos": em["cos_sim"],
            }
            print("    [mse-floor %-9s] %-30s absmax=%.4f -> mse=%.4f"
                  % (role, name, ea["rel_mse"], em["rel_mse"]), flush=True)

        del w, rec
        if (i + 1) % 25 == 0 or i == len(qtensors) - 1:
            dt = time.time() - t0
            print("  [%3d/%3d] %-34s rel_mse(absmax)=%.4f  (%.0fs)"
                  % (i + 1, len(qtensors), name, e["rel_mse"], dt), flush=True)

    # ---- write CSV ----
    import os
    csv_path = os.path.join(args.outdir, "sensitivity_table.csv")
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print("wrote %s (%d rows)" % (csv_path, len(rows)))

    # ---- aggregates ----
    # absmax runs on every tensor -> it is the ranking/headline metric.
    key = "absmax_rel_mse"

    by_role = defaultdict(list)
    by_layer_band = defaultdict(list)
    for row in rows:
        by_role[row["role"]].append(row[key])
        L = row["layer"]
        if L < 0:
            band = "global"
        elif L < 5:
            band = "early(0-4)"
        elif L < 55:
            band = "mid(5-54)"
        else:
            band = "late(55-59)"
        by_layer_band[band].append(row[key])

    def stat(xs):
        xs = np.asarray(xs)
        return dict(n=len(xs), mean=float(xs.mean()), median=float(np.median(xs)),
                    p90=float(np.percentile(xs, 90)), max=float(xs.max()),
                    min=float(xs.min()))

    role_stats = {k: stat(v) for k, v in by_role.items()}
    band_stats = {k: stat(v) for k, v in by_layer_band.items()}

    worst = sorted(rows, key=lambda r: r[key], reverse=True)[:20]
    best = sorted(rows, key=lambda r: r[key])[:5]

    # ---- markdown summary ----
    md = []
    md.append("# Stage B2 — Per-tensor 2-bit sensitivity sweep (NF2)\n")
    md.append("Reference = Q4_0/Q4_1/Q4_K GGUF dequant (the 4-bit weights we ship "
              "from). Errors are the **incremental** 4-bit -> ~2.5-bit loss "
              "(Lever F_bytes), not the true-BF16 floor.\n")
    md.append("- Quantizable weight tensors swept: **%d**" % len(rows))
    md.append("- F32 norm/scalar tensors excluded (not matmul weights): **%d**" % n_f32)
    md.append("- Format: NF2 codebook, 16-elt block, 1 E4M3 scale + 16x2-bit = "
              "0.3125 B/elt = 2.5 bit/elt -> 31B = 9.69 GB")
    md.append("- Headline metric below: **nf2 absmax rel_mse** (the shippable "
              "NVFP4-style encoder, run on ALL %d tensors). The per-block "
              "MSE-optimal floor is reported per-role below (run on one "
              "representative tensor per role; ~24x slower).\n" % len(rows))

    allk = np.array([row[key] for row in rows])
    md.append("Overall rel_mse (absmax): mean=%.4f median=%.4f p90=%.4f max=%.4f\n"
              % (allk.mean(), np.median(allk), np.percentile(allk, 90), allk.max()))

    if mse_floor:
        md.append("## MSE-optimal floor by role (one tensor each)\n")
        md.append("| role | tensor | absmax rel_mse | mse rel_mse | mse cos | mse sqnr_dB |")
        md.append("|---|---|---|---|---|---|")
        for role in sorted(mse_floor):
            mf = mse_floor[role]
            md.append("| %s | %s | %.4f | %.4f | %.4f | %.2f |"
                      % (role, mf["name"], mf["absmax_rel_mse"], mf["mse_rel_mse"],
                         mf["mse_cos"], mf["mse_sqnr_db"]))
        md.append("")

    md.append("## Worst 20 tensors by rel_mse\n")
    md.append("| rank | tensor | role | layer | src | rel_mse | cos | sqnr_dB |")
    md.append("|---|---|---|---|---|---|---|---|")
    ck = "absmax_cos"
    sk = "absmax_sqnr_db"
    for i, row in enumerate(worst):
        md.append("| %d | %s | %s | %d | %s | %.4f | %.4f | %.2f |"
                  % (i + 1, row["name"], row["role"], row["layer"], row["src_type"],
                     row[key], row[ck], row[sk]))
    md.append("")
    md.append("## Best 5 tensors (lowest error)\n")
    md.append("| tensor | role | layer | rel_mse |")
    md.append("|---|---|---|---|")
    for row in best:
        md.append("| %s | %s | %d | %.4f |" % (row["name"], row["role"],
                                               row["layer"], row[key]))
    md.append("")

    md.append("## Error by role (rel_mse)\n")
    md.append("| role | n | mean | median | p90 | max |")
    md.append("|---|---|---|---|---|---|")
    for role in sorted(role_stats, key=lambda k: role_stats[k]["mean"], reverse=True):
        s = role_stats[role]
        md.append("| %s | %d | %.4f | %.4f | %.4f | %.4f |"
                  % (role, s["n"], s["mean"], s["median"], s["p90"], s["max"]))
    md.append("")

    md.append("## Error by layer band (rel_mse)\n")
    md.append("| band | n | mean | median | p90 | max |")
    md.append("|---|---|---|---|---|---|")
    order = ["global", "early(0-4)", "mid(5-54)", "late(55-59)"]
    for band in [b for b in order if b in band_stats]:
        s = band_stats[band]
        md.append("| %s | %d | %.4f | %.4f | %.4f | %.4f |"
                  % (band, s["n"], s["mean"], s["median"], s["p90"], s["max"]))
    md.append("")

    md.append("## Per-role per-layer-band mean rel_mse\n")
    rl = defaultdict(list)
    for row in rows:
        L = row["layer"]
        if L < 0:
            band = "global"
        elif L < 5:
            band = "early"
        elif L < 55:
            band = "mid"
        else:
            band = "late"
        rl[(row["role"], band)].append(row[key])
    roles = sorted({row["role"] for row in rows})
    bands = ["early", "mid", "late", "global"]
    md.append("| role | " + " | ".join(bands) + " |")
    md.append("|" + "---|" * (len(bands) + 1))
    for role in roles:
        cells = []
        for b in bands:
            v = rl.get((role, b))
            cells.append("%.4f" % np.mean(v) if v else "-")
        md.append("| %s | %s |" % (role, " | ".join(cells)))
    md.append("")

    md_path = os.path.join(args.outdir, "sensitivity_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print("wrote %s" % md_path)

    # ---- json metrics blob (for the structured return) ----
    blob = {
        "n_swept": len(rows), "n_f32_excluded": n_f32,
        "variant": args.variant, "headline_metric": key,
        "overall": {"mean": float(allk.mean()), "median": float(np.median(allk)),
                    "p90": float(np.percentile(allk, 90)), "max": float(allk.max())},
        "role_stats": role_stats, "band_stats": band_stats,
        "mse_floor": mse_floor,
        "worst20": [{"name": row["name"], "role": row["role"], "layer": row["layer"],
                     "rel_mse": row[key], "cos": row[ck]} for row in worst],
    }
    json_path = os.path.join(args.outdir, "sensitivity_metrics.json")
    with open(json_path, "w") as f:
        json.dump(blob, f, indent=2)
    print("wrote %s" % json_path)
    print("\n=== SUMMARY ===")
    print(json.dumps(blob["overall"], indent=2))
    return blob


if __name__ == "__main__":
    main(sys.argv)
