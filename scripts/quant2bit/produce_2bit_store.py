#!/usr/bin/env python3
"""
produce_2bit_store.py — Stage B3: produce the REAL LEAN 2-bit store on disk.

Streams every weight tensor from the 31B Q4_0 GGUF, dequantizes to fp32, and
re-quantizes per the LEAN role recipe using the MSE-optimal MXFP2/MXFP3 encoder
(scripts/quant2bit/mxfp2.py), then writes a single self-describing on-disk store.

LEAN recipe (from the B2 sensitivity sweep / bit_allocation.json):
  embed + attn_k + attn_v        -> NF3 (3-bit, 0.4375 B/elt)   [protect KV + embed]
  attn_q, attn_o, ffn_gate/up/dn -> NF2 (2-bit, 0.3125 B/elt)   [the bulk]
  norms / 1-D scalars / rope     -> raw F32 (not matmul weights)
Expected blended ~2.62 bit/elt, ~10 GB store.

--------------------------------------------------------------------------------
STORE CONTAINER FORMAT (v1) — see STORE_FORMAT.md for the full spec.
--------------------------------------------------------------------------------
A directory containing exactly two files:

  store.json     UTF-8 JSON header. Top-level keys:
                   "format": "fucina-2bit-store"
                   "version": 1
                   "block": 16            (codec block size for nf2/nf3)
                   "arch": "<gguf general.architecture>"
                   "source_gguf": "<path>"
                   "n_tensors": <int>
                   "tensors": [ {tensor record}, ... ]   (in GGUF file order)
                 Each tensor record:
                   "name":        original GGUF tensor name
                   "dims":        list[int] (GGUF dim order, fastest-varying first)
                   "n_elem":      product(dims)
                   "src_type":    GGUF ggml source type name (Q4_0/Q4_1/Q4_K/F32)
                   "variant":     "nf2" | "nf3" | "f32"
                   "role":        embed/attn_k/.../ffn_down/other (informational)
                   "n_blocks":    number of 16-elt codec blocks (0 for f32)
                   "codes_off":   byte offset into store.bin of the packed codes
                   "codes_bytes": length in bytes of the packed codes
                   "scales_off":  byte offset of the E4M3 block-scale array
                   "scales_bytes":length in bytes of the scale array (== n_blocks)
                   "raw_off":     byte offset of raw f32 payload (variant f32 only)
                   "raw_bytes":   length of raw f32 payload (variant f32 only)

  store.bin      Concatenated tensor payloads, each region pointed to by the
                 offsets above. Per quantized tensor:
                   - codes : packed 2-bit (nf2) or 3-bit (nf3) codes, bit-packed
                             LSB-first, block-major then element order. Total
                             bits = n_blocks*16*bits_per_code, rounded up to a
                             whole number of bytes (one tensor = one bitstream).
                   - scales: n_blocks bytes, one E4M3 (uint8) block scale each.
                 Per f32 tensor:
                   - raw   : n_elem little-endian float32, verbatim.

Code packing: nf2 uses 2 bits/code (4 levels), nf3 uses 3 bits/code (8 levels).
Codes are emitted in the SAME order mxfp2.quantize returns them: block 0 elt 0,
block 0 elt 1, ..., block 0 elt 15, block 1 elt 0, ... Decode = level[code]*scale.

Memory is bounded: large tensors are quantized in chunks of whole 16-elt blocks
and the codes are bit-packed + appended incrementally, so peak RAM is ~one chunk.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

import gguf_reader as G
import mxfp2 as M
from sensitivity_sweep import classify

DEFAULT_GGUF = ("/opt/spark/models/hub/models--unsloth--gemma-4-31B-it-GGUF/"
                "snapshots/8906b3db2e669a0b1d6293c315d3f9fbf934a86d/"
                "gemma-4-31B-it-Q4_0.gguf")
DEFAULT_OUT = "/opt/spark/models/fucina-2bit-31b"

QUANT_TYPES = {G.GGML_TYPE_Q4_0, G.GGML_TYPE_Q4_1, G.GGML_TYPE_Q4_K}

# LEAN recipe: role -> codec variant.
NF3_ROLES = {"embed", "attn_k", "attn_v"}
BITS = {"nf2": 2, "nf3": 3}

# Quantize big tensors in chunks of whole blocks to bound RAM.
# 4M blocks * 16 = 64M elems/chunk; the MSE grid peak is ~chunk*16*4 f32 per
# candidate but mxfp2.quantize_mse already sub-chunks internally to <=1M blocks.
CHUNK_BLOCKS = 4_000_000

# Error sampling cap (block-aligned). rel_mse of a per-block quantizer is
# sampling-invariant under block-aligned subsampling (see B2 notes), so we
# measure error on at most this many block-aligned elems per tensor.
ERR_CAP = 8_000_000


def variant_for(role):
    if role in NF3_ROLES:
        return "nf3"
    return "nf2"


def pack_codes(idx, bits):
    """Bit-pack a flat uint8 array of small codes (values < 2**bits) LSB-first
    into a uint8 bitstream. idx is flattened in its given order."""
    idx = np.ascontiguousarray(idx.reshape(-1).astype(np.uint8))
    # expand each code to its `bits` bits (LSB first), then pack 8 bits/byte.
    bitplanes = np.empty((idx.size, bits), dtype=np.uint8)
    for b in range(bits):
        bitplanes[:, b] = (idx >> b) & 1
    flat_bits = bitplanes.reshape(-1)  # length idx.size*bits, LSB-first per code
    return np.packbits(flat_bits, bitorder="little")


# NOTE on byte-alignment that lets us split a tensor across workers:
# every 16-elt block packs to a WHOLE number of bytes — nf2 = 16*2 = 32 bits =
# 4 bytes/block, nf3 = 16*3 = 48 bits = 6 bytes/block. So a block-aligned range
# of a tensor packs independently and its bytes concatenate with neighbours with
# no bit-straddle. We exploit this to parallelize the giant tensors (token_embd,
# ffn_*) across pool workers as (name, blk_start, blk_count) sub-jobs.

def quantize_block_range(w_range, variant, measure_err):
    """Quantize one block-aligned fp32 slice. Returns
    (codes_bytes, scales_bytes, n_blocks, sse, ss_ref, n_err) where the error
    stats cover up to ERR_CAP elems of THIS slice when measure_err is True.
    w_range length must be a multiple of BLOCK except possibly the very last
    slice of the tensor (zero-padded by quantize_mse)."""
    bits = BITS[variant]
    idx, scale, n = M.quantize_mse(w_range, variant)  # idx [nb,BLOCK], scale[nb]
    nb = scale.shape[0]
    codes = pack_codes(idx.reshape(-1).astype(np.uint8), bits).tobytes()
    scales = _e4m3_to_byte(scale).tobytes()
    sse = ss_ref = 0.0
    n_err = 0
    if measure_err:
        take = min(ERR_CAP, w_range.size)
        # snap to whole blocks for a clean per-block sample
        take = (take // M.BLOCK) * M.BLOCK
        if take > 0:
            wref = np.asarray(w_range[:take], dtype=np.float64)
            rec = M.dequantize(idx, scale, n, variant)[:take].astype(np.float64)
            d = rec - wref
            sse = float(np.dot(d, d))
            ss_ref = float(np.dot(wref, wref))
            n_err = int(take)
    return codes, scales, nb, sse, ss_ref, n_err


# ---- E4M3 float -> nearest representable byte (inverse of mxfp2.e4m3_decode) ----
_E4M3_ALL = M.e4m3_decode(np.arange(256, dtype=np.uint8))


def _e4m3_to_byte(scale_f32):
    """Map positive E4M3-valued floats (as produced by quantize_e4m3) back to the
    uint8 code whose decode equals them. The encoder already snapped to E4M3, so
    this is an exact lookup; we match on the positive (sign=0) table."""
    sf = np.asarray(scale_f32, dtype=np.float32)
    pos_codes = np.arange(128, dtype=np.uint8)
    pos_vals = _E4M3_ALL[:128]
    ok = np.isfinite(pos_vals)
    pos_codes = pos_codes[ok]
    pos_vals = pos_vals[ok]
    order = np.argsort(pos_vals)
    pv = pos_vals[order]
    pc = pos_codes[order]
    idx = np.searchsorted(pv, sf)
    idx = np.clip(idx, 1, len(pv) - 1)
    lo = pv[idx - 1]
    hi = pv[idx]
    pick_hi = np.abs(hi - sf) < np.abs(sf - lo)
    return np.where(pick_hi, pc[idx], pc[idx - 1]).astype(np.uint8)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0,
                    help="only process first N tensors (debug)")
    args = ap.parse_args(argv[1:])

    os.makedirs(args.out, exist_ok=True)
    bin_path = os.path.join(args.out, "store.bin")
    json_path = os.path.join(args.out, "store.json")

    r = G.GGUFReader(args.gguf)
    arch = r.kv.get("general.architecture", "unknown")
    names = list(r._order)
    if args.limit:
        names = names[:args.limit]

    records = []
    role_sse = {}     # role -> list of (rel_mse, n_elem, sampled)
    elem_by_variant = {"nf2": 0, "nf3": 0, "f32": 0}

    t0 = time.time()
    off = 0
    print("producing LEAN 2-bit store: %d tensors -> %s" % (len(names), args.out),
          flush=True)

    # Single-process, sequential, memory-bounded. Each tensor is dequantized to
    # fp32 once; quantized tensors are processed in producer-level chunks of whole
    # blocks (CHUNK_BLOCKS) so peak RAM is ~one chunk (+ quantize_mse's internal
    # sub-chunk). store.bin is written strictly in GGUF file order so the byte
    # offsets in store.json are exact.
    with open(bin_path, "wb") as fbin:
        for i, name in enumerate(names):
            ti = r.tensors[name]
            role, _layer = classify(name)
            src = G.GGML_TYPE_NAME.get(ti.ggml_type, ti.ggml_type)
            dims = [int(d) for d in ti.dims]
            n_elem = int(ti.n_elem)

            if ti.ggml_type not in QUANT_TYPES:
                # raw f32 payload (norms / scalars / rope). dequant() upcasts any
                # of F32/F16/BF16 to fp32.
                w = np.ascontiguousarray(r.dequant(name).astype(np.float32))
                raw = w.tobytes()
                fbin.write(raw)
                rb = len(raw)
                records.append({
                    "name": name, "dims": dims, "n_elem": n_elem,
                    "src_type": src, "variant": "f32", "role": role,
                    "n_blocks": 0, "codes_off": 0, "codes_bytes": 0,
                    "scales_off": 0, "scales_bytes": 0,
                    "raw_off": int(off), "raw_bytes": int(rb),
                })
                off += rb
                elem_by_variant["f32"] += n_elem
                variant = "f32"
                rel_mse = 0.0
                del w
            else:
                variant = variant_for(role)
                w = r.dequant(name).astype(np.float32).reshape(-1)
                # process in whole-block chunks; codes/scales bytes concatenate
                # cleanly because each 16-elt block packs to whole bytes.
                codes_off = off
                cb_total = 0
                sb_total = 0
                nb_total = 0
                sse = ss_ref = 0.0
                n_err = 0
                chunk_elems = CHUNK_BLOCKS * M.BLOCK
                pos = 0
                codes_parts = []
                scales_parts = []
                while pos < w.size:
                    end = min(pos + chunk_elems, w.size)
                    # only the LAST chunk may carry a partial trailing block.
                    if end != w.size:
                        end = pos + ((end - pos) // M.BLOCK) * M.BLOCK
                    wslice = w[pos:end]
                    # sample error from the FIRST chunk only (per-block quantizer
                    # error is sampling-invariant); cap at ERR_CAP elems.
                    measure = (pos == 0)
                    c, s, nb, csse, css_ref, cn = quantize_block_range(
                        wslice, variant, measure)
                    codes_parts.append(c)
                    scales_parts.append(s)
                    cb_total += len(c)
                    sb_total += len(s)
                    nb_total += nb
                    sse += csse
                    ss_ref += css_ref
                    n_err += cn
                    pos = end
                fbin.write(b"".join(codes_parts))
                fbin.write(b"".join(scales_parts))
                assert sb_total == nb_total, (name, sb_total, nb_total)
                records.append({
                    "name": name, "dims": dims, "n_elem": n_elem,
                    "src_type": src, "variant": variant, "role": role,
                    "n_blocks": int(nb_total),
                    "codes_off": int(codes_off), "codes_bytes": int(cb_total),
                    "scales_off": int(codes_off + cb_total),
                    "scales_bytes": int(sb_total),
                    "raw_off": 0, "raw_bytes": 0,
                })
                off += cb_total + sb_total
                elem_by_variant[variant] += n_elem
                rel_mse = (sse / ss_ref) if ss_ref > 0 else 0.0
                sampled = (n_err < n_elem)
                role_sse.setdefault(role, []).append((rel_mse, n_elem, sampled))
                del w, codes_parts, scales_parts

            dt = time.time() - t0
            gb = off / (1024 ** 3)
            print("  [%3d/%3d] %-30s %-3s rel_mse=%.4f  store=%.2f GiB (%.0fs)"
                  % (i + 1, len(names), name, variant, rel_mse, gb, dt),
                  flush=True)

    # ---- header ----
    header = {
        "format": "fucina-2bit-store",
        "version": 1,
        "block": M.BLOCK,
        "arch": arch,
        "source_gguf": args.gguf,
        "recipe": "LEAN (embed+attn_k+attn_v -> NF3; rest -> NF2; norms -> F32)",
        "n_tensors": len(records),
        "tensors": records,
    }
    with open(json_path, "w") as f:
        json.dump(header, f, indent=1)

    # ---- summary numbers ----
    store_bytes = off
    store_gib = store_bytes / (1024 ** 3)
    tot_elem = sum(rrec["n_elem"] for rrec in records)
    quant_elem = elem_by_variant["nf2"] + elem_by_variant["nf3"]
    # blended bits over the QUANTIZED matmul weights (the headline; f32 norms are
    # negligible — ~2.2M elems). bits/elt incl. block scale: nf2=2.5, nf3=3.5.
    bits_nf2 = 2.5
    bits_nf3 = 3.5
    blended_bits = ((elem_by_variant["nf2"] * bits_nf2
                     + elem_by_variant["nf3"] * bits_nf3)
                    / max(1, quant_elem))

    print("\n=== STORE SUMMARY ===")
    print("out dir         : %s" % args.out)
    print("store.bin size  : %.3f GiB (%d bytes)" % (store_gib, store_bytes))
    print("total elems     : %d" % tot_elem)
    print("  nf2 (2-bit)   : %d elems (%.1f%%)"
          % (elem_by_variant["nf2"], 100 * elem_by_variant["nf2"] / tot_elem))
    print("  nf3 (3-bit)   : %d elems (%.1f%%)"
          % (elem_by_variant["nf3"], 100 * elem_by_variant["nf3"] / tot_elem))
    print("  f32 (norms)   : %d elems" % elem_by_variant["f32"])
    print("blended bits/elt (over quantized weights): %.4f" % blended_bits)

    # per-role rel_mse (element-weighted mean across tensors of that role)
    print("\n=== PER-ROLE rel_mse (vs fp32 dequant, sampled<=8M/tensor) ===")
    role_table = {}
    print("%-10s %6s %8s %14s %s" % ("role", "variant", "n_tens", "elem-wt rel_mse", "any_sampled"))
    for role in sorted(role_sse):
        items = role_sse[role]
        tot = sum(e for _, e, _ in items)
        wmse = sum(rm * e for rm, e, _ in items) / max(1, tot)
        any_s = any(s for _, _, s in items)
        var = variant_for(role)
        role_table[role] = {"variant": var, "n_tensors": len(items),
                            "elem_weighted_rel_mse": wmse, "any_sampled": any_s}
        print("%-10s %6s %8d %14.4f %s"
              % (role, var, len(items), wmse, any_s))

    summary = {
        "store_dir": args.out, "store_bin_bytes": store_bytes,
        "store_gib": store_gib, "total_elems": tot_elem,
        "elem_by_variant": elem_by_variant,
        "blended_bits_per_elt": blended_bits,
        "runtime_s": time.time() - t0,
        "per_role_rel_mse": role_table,
    }
    print("\nruntime: %.0f s" % summary["runtime_s"])
    # emit the machine-readable summary alongside the store and to the script dir
    with open(os.path.join(args.out, "produce_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    main(sys.argv)
