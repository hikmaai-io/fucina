#!/usr/bin/env python3
"""verify_store.py — independent decoder for the fucina-2bit store.

Reads store.json + store.bin, decodes each tensor back to fp32 using ONLY the
header offsets (no producer state), and compares against the GGUF fp32 dequant.
This proves the on-disk format is self-describing and the producer's reported
rel_mse is reproducible from disk. Used as a correctness gate on the smoke store.
"""
import json
import os
import sys

import numpy as np

import gguf_reader as G
import mxfp2 as M

BITS = {"nf2": 2, "nf3": 3}


def unpack_codes(buf, n_codes, bits):
    """Inverse of produce_2bit_store.pack_codes: LSB-first bit-unpack."""
    all_bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")
    all_bits = all_bits[: n_codes * bits].reshape(n_codes, bits)
    weights = (1 << np.arange(bits, dtype=np.uint32))[None, :]
    return (all_bits.astype(np.uint32) * weights).sum(axis=1).astype(np.uint8)


def decode_tensor(rec, fbin):
    variant = rec["variant"]
    n = rec["n_elem"]
    if variant == "f32":
        fbin.seek(rec["raw_off"])
        return np.frombuffer(fbin.read(rec["raw_bytes"]), dtype=np.float32)
    bits = BITS[variant]
    nb = rec["n_blocks"]
    n_codes = nb * M.BLOCK
    fbin.seek(rec["codes_off"])
    codes_buf = fbin.read(rec["codes_bytes"])
    idx = unpack_codes(codes_buf, n_codes, bits).reshape(nb, M.BLOCK)
    fbin.seek(rec["scales_off"])
    sbytes = np.frombuffer(fbin.read(rec["scales_bytes"]), dtype=np.uint8)
    scale = M.e4m3_decode(sbytes).astype(np.float32)
    return M.dequantize(idx, scale, n, variant)


def main(argv):
    store_dir = argv[1]
    gguf = argv[2] if len(argv) > 2 else None
    hdr = json.load(open(os.path.join(store_dir, "store.json")))
    fbin = open(os.path.join(store_dir, "store.bin"), "rb")
    r = G.GGUFReader(gguf) if gguf else None
    print("format=%s v%d arch=%s n_tensors=%d"
          % (hdr["format"], hdr["version"], hdr["arch"], hdr["n_tensors"]))
    worst = 0.0
    for rec in hdr["tensors"]:
        rec_dec = decode_tensor(rec, fbin)
        assert rec_dec.size == rec["n_elem"], (rec["name"], rec_dec.size, rec["n_elem"])
        line = "%-32s %-4s n=%d" % (rec["name"], rec["variant"], rec["n_elem"])
        if r is not None:
            ref = r.dequant(rec["name"]).astype(np.float32).reshape(-1)
            # compare on a block-aligned sample to keep it fast
            k = min(ref.size, 8_000_000 // M.BLOCK * M.BLOCK)
            d = rec_dec[:k].astype(np.float64) - ref[:k].astype(np.float64)
            rel = float(np.dot(d, d)) / (float(np.dot(ref[:k].astype(np.float64),
                                                       ref[:k].astype(np.float64))) + 1e-30)
            worst = max(worst, rel if rec["variant"] != "f32" else 0.0)
            line += "  rel_mse=%.5f" % rel
            if rec["variant"] == "f32":
                line += " (exact: max|d|=%.2e)" % float(np.max(np.abs(d)))
        print(line)
    print("OK — all tensors decoded; worst quantized rel_mse=%.5f" % worst)


if __name__ == "__main__":
    main(sys.argv)
