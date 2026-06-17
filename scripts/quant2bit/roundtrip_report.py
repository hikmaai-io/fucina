#!/usr/bin/env python3
"""
roundtrip_report.py — Stage B1 verification.

Loads representative tensors from the 31B Q4_0 GGUF (dequantized to fp32 = the
reference), runs the MXFP2 (E1M0 & NF2) round-trip on each, and reports
relative MSE, RMS error, max abs error, cosine similarity, SQNR, and bytes/elt.

Reference precision note: the Q4_0 GGUF is itself a 4-bit quantization of the
original BF16. Reconstruction error here is "2-bit vs the 4-bit reference we
will actually ship from". Absolute floor vs true BF16 would be larger; this
measures the *incremental* loss of going 4-bit -> ~2.5-bit, which is exactly
Lever F_bytes.
"""
import argparse
import sys

import numpy as np

import gguf_reader as G
import mxfp2 as M

DEFAULT_GGUF = ("/opt/spark/models/hub/models--unsloth--gemma-4-31B-it-GGUF/"
                "snapshots/8906b3db2e669a0b1d6293c315d3f9fbf934a86d/"
                "gemma-4-31B-it-Q4_0.gguf")

# (label, tensor name) — one attn_q, one ffn_down, one ffn_gate, the embedding.
DEFAULT_TENSORS = [
    ("attn_q   (blk.0)", "blk.0.attn_q.weight"),
    ("ffn_down (blk.10)", "blk.10.ffn_down.weight"),
    ("ffn_gate (blk.0) ", "blk.0.ffn_gate.weight"),
    ("ffn_up   (blk.0) ", "blk.0.ffn_up.weight"),
    ("attn_output(blk.0)", "blk.0.attn_output.weight"),
    ("token_embd       ", "token_embd.weight"),
]


def fmt_row(label, src_type, e):
    return ("%-19s %-12s rel_mse=%.4e rms=%.4e maxabs=%.4e cos=%.5f sqnr=%5.2fdB"
            % (label, src_type, e["rel_mse"], e["rms_err"], e["max_abs_err"],
               e["cos_sim"], e["sqnr_db"]))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--max-elems", type=int, default=0,
                    help="subsample to at most N elements for speed (0=all)")
    ap.add_argument("--mse", action="store_true",
                    help="also report the MSE-optimal-scale encoder (achievable floor)")
    args = ap.parse_args(argv[1:])

    r = G.GGUFReader(args.gguf)
    print("MXFP2 2-bit round-trip report (reference = Q4_0/Q4_K GGUF dequant)")
    print("bytes/elt = %.4f  (31B -> %.2f GB)"
          % (M.bytes_per_elt(), 31e9 * M.bytes_per_elt() / 1e9))
    print("E1M0 levels = %s" % M.LEVELS_E1M0.tolist())
    print("NF2  levels = %s" % [round(x, 5) for x in M.LEVELS_NF2.tolist()])
    print("=" * 100)

    summary = {"e1m0": [], "nf2": []}
    for label, name in DEFAULT_TENSORS:
        ti = r.tensors[name]
        src = G.GGML_TYPE_NAME.get(ti.ggml_type, ti.ggml_type)
        w = r.dequant(name)
        if args.max_elems and w.size > args.max_elems:
            # deterministic stride subsample, block-aligned
            stride = w.size // args.max_elems
            w = w[::stride]
        print("%s  [%s]  n=%d  rms_w=%.5f" % (label.strip(), src, w.size,
                                              float(np.sqrt(np.mean(w**2)))))
        for v in ("e1m0", "nf2"):
            rec = M.roundtrip(w, v)
            e = M.errors(w, rec)
            summary[v].append((label, e))
            print("    " + fmt_row("", v + " absmax", e))
            if args.mse:
                rec2 = M.roundtrip_mse(w, v)
                e2 = M.errors(w, rec2)
                summary.setdefault(v + "_mse", []).append((label, e2))
                print("    " + fmt_row("", v + " mse   ", e2))
        print("-" * 100)

    print("=" * 100)
    print("MEAN across tensors:")
    for v in [k for k in ("e1m0", "e1m0_mse", "nf2", "nf2_mse") if k in summary and summary[k]]:
        rel = np.mean([e["rel_mse"] for _, e in summary[v]])
        sqnr = np.mean([e["sqnr_db"] for _, e in summary[v]])
        cos = np.mean([e["cos_sim"] for _, e in summary[v]])
        print("  %-5s mean rel_mse=%.4e  mean sqnr=%.2f dB  mean cos=%.5f"
              % (v, rel, sqnr, cos))

    # winner
    e1 = np.mean([e["rel_mse"] for _, e in summary["e1m0"]])
    nf = np.mean([e["rel_mse"] for _, e in summary["nf2"]])
    print("WINNER: %s (lower mean rel_mse)" % ("nf2" if nf < e1 else "e1m0"))


if __name__ == "__main__":
    main(sys.argv)
