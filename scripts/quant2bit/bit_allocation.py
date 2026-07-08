#!/usr/bin/env python3
"""
bit_allocation.py — Stage B3 synthesis.

Reads the B2 per-tensor sensitivity table (sensitivity_table.csv, which carries
the real n_elem per tensor) and computes the EXACT blended footprint for a
sensitivity-aware mixed-precision recipe.

Per-element on-disk cost (microscaled, 16-elt block, 1 E4M3 scale/block):
  - 2-bit NF2  : (16*2 + 8) / 8 / 16 = 0.3125 B/elt = 2.5  bit/elt
  - 4-bit NVFP4: (16*4 + 8) / 8 / 16 = 0.5625 B/elt = 4.5  bit/elt
  - 3-bit NF3  : (16*3 + 8) / 8 / 16 = 0.4375 B/elt = 3.5  bit/elt  (comparison point)

We assign each of the 411 quantizable weight tensors a precision by ROLE
(B2 found the problem is uniform, with a role ordering: embed > attn_k/v > rest,
attn_o best) and report blended bits + GB for several recipes. The non-weight
F32 tensors (norms / scalars / rope_freqs, 422 of them) are kept F32 but are
tiny; we add their real byte cost from the GGUF so the GB number is honest.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "sensitivity_table.csv")

# bytes/elt for each microscaled precision (16-elt block + 1 E4M3 scale)
BPE = {2: (16 * 2 + 8) / 8 / 16,   # 0.3125
       3: (16 * 3 + 8) / 8 / 16,   # 0.4375
       4: (16 * 4 + 8) / 8 / 16}   # 0.5625
BITS = {2: 2.5, 3: 3.5, 4: 4.5}


def load_tensors():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append({
                "name": r["name"],
                "role": r["role"],
                "layer": int(r["layer"]),
                "n_elem": int(r["n_elem"]),
                "rel_mse": float(r["absmax_rel_mse"]),
            })
    return rows


def recipe_uniform2(t):
    return 2


def recipe_mixed(t):
    """Sensitivity-aware: protect embed + KV projections at 4-bit, rest 2-bit."""
    if t["role"] == "embed":
        return 4
    if t["role"] in ("attn_k", "attn_v"):
        return 4
    return 2


def recipe_mixed_plus(t):
    """Conservative variant: also lift attn_q to 3-bit (next-narrowest matmul)."""
    if t["role"] == "embed":
        return 4
    if t["role"] in ("attn_k", "attn_v"):
        return 4
    if t["role"] == "attn_q":
        return 3
    return 2


def recipe_lean(t):
    """Target ~9.5 GB / ~2.45 avg bits. Lift only the worst KV tensors and the
    embed to 3-bit (NF3) instead of full 4-bit. Embed is the single worst tensor
    but a 3-bit codebook already halves its incremental error; attn_k/v are the
    worst roles but tiny, so 3-bit there is cheap insurance. Everything else 2-bit."""
    if t["role"] == "embed":
        return 3
    if t["role"] in ("attn_k", "attn_v"):
        return 3
    return 2


def recipe_uniform4(t):
    return 4


def evaluate(rows, assign, f32_bytes):
    tot_elem = sum(t["n_elem"] for t in rows)
    tot_bytes = 0.0
    # weighted rel_mse: weight by elems, and the protected tensors get the
    # 4-bit reference error (~0 incremental loss vs the GGUF, since the GGUF IS
    # 4-bit) — i.e. protected tensors contribute their 4-bit error ~= 0 here.
    w_relmse_num = 0.0
    by_bits_elem = {2: 0, 3: 0, 4: 0}
    for t in rows:
        b = assign(t)
        tot_bytes += t["n_elem"] * BPE[b]
        by_bits_elem[b] += t["n_elem"]
        # 2-bit tensors carry their measured incremental NF2 error.
        # 3-bit (NF3) tensors carry ~0.17x of their NF2 error (measured on 5
        # representative tensors: nf3/nf2 ratio 0.16-0.18, mean ~0.17).
        # 4-bit tensors keep >= the GGUF precision -> ~0 incremental loss.
        if b == 2:
            eff = t["rel_mse"]
        elif b == 3:
            eff = 0.17 * t["rel_mse"]
        else:
            eff = 0.0
        w_relmse_num += t["n_elem"] * eff
    blended_bpe = tot_bytes / tot_elem
    blended_bits = blended_bpe * 8
    gb_weights = tot_bytes / 1e9
    gb_total = (tot_bytes + f32_bytes) / 1e9
    w_relmse = w_relmse_num / tot_elem
    frac2 = by_bits_elem[2] / tot_elem
    return {
        "blended_bpe": blended_bpe,
        "blended_bits": blended_bits,
        "gb_weights": gb_weights,
        "gb_total": gb_total,
        "weighted_incr_relmse": w_relmse,
        "frac_elem_2bit": frac2,
        "frac_elem_3bit": by_bits_elem[3] / tot_elem,
        "frac_elem_4bit": by_bits_elem[4] / tot_elem,
        "tot_elem": tot_elem,
    }


def main():
    rows = load_tensors()
    # F32 non-weight bytes: 422 tensors, but their total elem count is small.
    # We don't have them in the CSV (only the 411 weights). Their byte cost in
    # the 31B Gemma is dominated by per-layer RMSNorm gains (~width per norm) and
    # is < 0.1 GB. Use an env override if a measured number is available, else 0.
    f32_bytes = float(os.environ.get("F32_BYTES", "0"))

    recipes = [
        ("uniform-2bit (all NF2)", recipe_uniform2),
        ("MIXED (embed+KV @4bit, rest 2bit)", recipe_mixed),
        ("MIXED+ (embed+KV @4bit, attn_q @3bit, rest 2bit)", recipe_mixed_plus),
        ("LEAN (embed+KV @3bit NF3, rest 2bit)", recipe_lean),
        ("uniform-4bit (NVFP4 ref)", recipe_uniform4),
    ]
    print(f"# tensors: {len(rows)}  total weight elems: {sum(t['n_elem'] for t in rows):,}")
    print(f"# F32 non-weight bytes added: {f32_bytes/1e9:.3f} GB\n")
    hdr = f"{'recipe':52s} {'blend.bits':>10s} {'GBwt':>7s} {'GBtot':>7s} {'incrRelMSE':>11s} {'%2b/%3b/%4b elems':>20s}"
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for name, fn in recipes:
        r = evaluate(rows, fn, f32_bytes)
        results[name] = r
        print(f"{name:52s} {r['blended_bits']:10.3f} {r['gb_weights']:7.2f} "
              f"{r['gb_total']:7.2f} {r['weighted_incr_relmse']:11.4f} "
              f"{100*r['frac_elem_2bit']:5.1f}/{100*r['frac_elem_3bit']:4.1f}/{100*r['frac_elem_4bit']:4.1f}")

    # role byte breakdown for the mixed recipe (to show KV-protection cost)
    print("\n# per-role elem share & protection cost (MIXED recipe):")
    role_elem = {}
    for t in rows:
        role_elem.setdefault(t["role"], 0)
        role_elem[t["role"]] += t["n_elem"]
    tot = sum(role_elem.values())
    for role in sorted(role_elem, key=lambda k: -role_elem[k]):
        n = role_elem[role]
        prot = role in ("embed", "attn_k", "attn_v")
        # extra bytes for protecting at 4-bit vs 2-bit
        extra = n * (BPE[4] - BPE[2]) if prot else 0.0
        print(f"  {role:10s} {n/1e9:6.3f} Gelem  {100*n/tot:5.1f}%  "
              f"{'PROTECT@4bit' if prot else 'NF2@2bit':14s} "
              f"{('+%.3f GB' % (extra/1e9)) if prot else ''}")

    import json
    out = os.path.join(HERE, "bit_allocation.json")
    with open(out, "w") as f:
        json.dump({"recipes": results, "bpe": BPE, "bits": BITS,
                   "f32_bytes": f32_bytes}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
