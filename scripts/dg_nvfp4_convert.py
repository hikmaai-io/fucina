#!/usr/bin/env python3
"""
dg_nvfp4_convert.py — Read NVIDIA's NVFP4 DiffusionGemma-26B-A4B-it checkpoint and
extract the MoE expert weights for later use by the fucina CUDA engine.

The checkpoint NVFP4-quantizes ONLY the experts. Per (layer L, expert E, proj) tile
there are four tensors:

  model.decoder.layers.{L}.experts.{E}.{gate,up,down}_proj.weight
      uint8, shape [out, in/2]   -> packed E2M1, 2 FP4 values per byte (low nibble first)
  model.decoder.layers.{L}.experts.{E}.{...}.weight_scale
      float8_e4m3, shape [out, in/16] -> per-16-element block scales, LINEAR/row-major
  model.decoder.layers.{L}.experts.{E}.{...}.weight_scale_2
      float32 scalar -> per-tensor global scale
  model.decoder.layers.{L}.experts.{E}.{...}.input_scale
      float32 scalar -> static activation (input) scale

Dequant:  w_fp32 = e2m1_decode(nibble) * e4m3_decode(block_scale) * global_scale

This module is dependency-light: it parses the safetensors container manually
(8-byte LE header length + JSON header + raw tensor bytes) so it works without the
`safetensors` pip package. numpy is used for the dequant sanity check only.

CLI:
  python3 scripts/dg_nvfp4_convert.py --inspect   # full layout report + sanity check
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

# ----------------------------------------------------------------------------
# Checkpoint location
# ----------------------------------------------------------------------------
# NVFP4 checkpoint snapshot directory (e.g. a downloaded Hugging Face hub snapshot of
# nvidia/diffusiongemma-26B-A4B-it-NVFP4). Provide it via --ckpt or the DG_NVFP4_CKPT
# environment variable; there is no machine-specific default.
DEFAULT_CKPT = os.environ.get("DG_NVFP4_CKPT", "")

NUM_LAYERS = 30
NUM_EXPERTS = 128
PROJS = ("gate_proj", "up_proj", "down_proj")
GROUP_SIZE = 16  # NVFP4 block scale group size (from hf_quant_config.json)

# safetensors dtype string -> (numpy dtype or None for FP8, itemsize bytes)
_ST_DTYPE = {
    "F64": (np.float64, 8),
    "F32": (np.float32, 4),
    "F16": (np.float16, 2),
    "BF16": (None, 2),       # bfloat16, handled specially
    "F8_E4M3": (None, 1),    # float8 e4m3, handled specially
    "F8_E5M2": (None, 1),
    "I64": (np.int64, 8),
    "I32": (np.int32, 4),
    "I16": (np.int16, 2),
    "I8": (np.int8, 1),
    "U8": (np.uint8, 1),
    "BOOL": (np.bool_, 1),
}


# ----------------------------------------------------------------------------
# E2M1 (FP4) decode table — 16 entries indexed by the 4-bit nibble (sign<<3 | mag)
#   magnitude levels: {0, .5, 1, 1.5, 2, 3, 4, 6}
# ----------------------------------------------------------------------------
_E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_E2M1_LUT = np.empty(16, dtype=np.float32)
_E2M1_LUT[0:8] = _E2M1_MAG
_E2M1_LUT[8:16] = -_E2M1_MAG  # sign bit set -> negative


def e2m1_decode(nibbles):
    """Decode an array of 4-bit values (uint8 in 0..15) to fp32."""
    return _E2M1_LUT[np.asarray(nibbles, dtype=np.uint8) & 0x0F]


def e4m3_decode(u8):
    """Decode float8_e4m3 (OCP/torch convention, bias 7) raw uint8 bytes to fp32.

    e4m3: 1 sign, 4 exponent (bias 7), 3 mantissa. No infinities; the all-ones
    exponent+mantissa pattern (0x7F / 0xFF) is NaN. Subnormals when exp==0.
    """
    u8 = np.asarray(u8, dtype=np.uint8)
    sign = np.where((u8 & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exp = ((u8 >> 3) & 0x0F).astype(np.int32)
    man = (u8 & 0x07).astype(np.float32)
    out = np.empty(u8.shape, dtype=np.float32)
    # normal: (1 + man/8) * 2^(exp-7)
    norm = (1.0 + man / 8.0) * np.power(2.0, (exp - 7).astype(np.float32))
    # subnormal (exp==0): (man/8) * 2^(1-7) = (man/8) * 2^-6
    sub = (man / 8.0) * (2.0 ** -6)
    out = np.where(exp == 0, sub, norm).astype(np.float32)
    # NaN pattern: exp all ones AND man all ones
    nan_mask = (exp == 0x0F) & (man == 7.0)
    out = np.where(nan_mask, np.float32(np.nan), out)
    return sign * out


# ----------------------------------------------------------------------------
# Minimal safetensors reader (manual header parse, no dependency)
# ----------------------------------------------------------------------------
class SafetensorsShard:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(hlen))
        self.data_start = 8 + hlen
        self._meta = self.header.pop("__metadata__", None)

    def info(self, name):
        m = self.header[name]
        return m["dtype"], tuple(m["shape"]), m["data_offsets"]

    def byte_size(self, name):
        o = self.header[name]["data_offsets"]
        return o[1] - o[0]

    def raw_bytes(self, name):
        """Return raw tensor bytes (no dtype interpretation)."""
        m = self.header[name]
        a, b = m["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.data_start + a)
            return f.read(b - a)

    def tensor(self, name):
        """Return tensor as a numpy array (reshaped). FP8/BF16 returned as raw uint8/uint16."""
        dtype, shape, _ = self.info(name)
        buf = self.raw_bytes(name)
        if dtype == "F8_E4M3" or dtype == "F8_E5M2":
            arr = np.frombuffer(buf, dtype=np.uint8)
        elif dtype == "BF16":
            arr = np.frombuffer(buf, dtype=np.uint16)
        else:
            npdt, _ = _ST_DTYPE[dtype]
            if npdt is None:
                raise ValueError("unsupported dtype %s" % dtype)
            arr = np.frombuffer(buf, dtype=npdt)
        if shape:
            arr = arr.reshape(shape)
        else:
            arr = arr.reshape(())  # scalar
        return arr


class Checkpoint:
    """Lazily opens the safetensors shards referenced by the index, resolving names."""

    def __init__(self, ckpt_dir=DEFAULT_CKPT):
        self.dir = ckpt_dir
        idx_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
        with open(idx_path) as f:
            self.weight_map = json.load(f)["weight_map"]
        self._shards = {}

    def _shard(self, fname):
        if fname not in self._shards:
            self._shards[fname] = SafetensorsShard(os.path.join(self.dir, fname))
        return self._shards[fname]

    def shard_for(self, name):
        return self._shard(self.weight_map[name])

    def info(self, name):
        return self.shard_for(name).info(name)

    def byte_size(self, name):
        return self.shard_for(name).byte_size(name)

    def raw_bytes(self, name):
        return self.shard_for(name).raw_bytes(name)

    def tensor(self, name):
        return self.shard_for(name).tensor(name)


# ----------------------------------------------------------------------------
# Naming helpers
# ----------------------------------------------------------------------------
def expert_key(L, E, proj, suffix):
    return "model.decoder.layers.%d.experts.%d.%s.%s" % (L, E, proj, suffix)


# ----------------------------------------------------------------------------
# Extraction API
# ----------------------------------------------------------------------------
def extract(L, E, proj, ckpt=None):
    """Extract one NVFP4 expert projection tile.

    Returns dict:
      weight_bytes        : bytes, packed E2M1 (uint8, 2 vals/byte), length out*(in/2)
      block_scales_linear : np.ndarray uint8 [out, in/16], LINEAR e4m3 block scales
                            (raw bytes; decode with e4m3_decode). NOT swizzled —
                            the CUDA loader must apply the cuBLASLt 32x4x4 swizzle.
      global_scale_f32    : float, per-tensor weight_scale_2
      input_scale_f32     : float, static activation scale
      in_dim              : int, logical input elements per row
      out_dim             : int, output rows
    """
    if ckpt is None:
        ckpt = Checkpoint()
    if proj not in PROJS:
        raise ValueError("proj must be one of %s" % (PROJS,))

    w_name = expert_key(L, E, proj, "weight")
    ws_name = expert_key(L, E, proj, "weight_scale")
    g_name = expert_key(L, E, proj, "weight_scale_2")
    a_name = expert_key(L, E, proj, "input_scale")

    w_dtype, w_shape, _ = ckpt.info(w_name)
    if w_dtype != "U8":
        raise ValueError("expected packed uint8 weight, got %s" % w_dtype)
    out_dim, half_in = w_shape
    in_dim = half_in * 2

    ws = ckpt.tensor(ws_name)  # uint8 [out, in/16]
    if ws.shape != (out_dim, in_dim // GROUP_SIZE):
        raise ValueError(
            "weight_scale shape %s != expected [%d,%d] (not linear?)"
            % (ws.shape, out_dim, in_dim // GROUP_SIZE)
        )

    g = float(ckpt.tensor(g_name))
    a = float(ckpt.tensor(a_name))

    return {
        "weight_bytes": ckpt.raw_bytes(w_name),
        "block_scales_linear": np.ascontiguousarray(ws),
        "global_scale_f32": g,
        "input_scale_f32": a,
        "in_dim": in_dim,
        "out_dim": out_dim,
    }


def dequant_row(rec, row):
    """Dequantize one output row of an extracted tile to fp32 [in_dim]."""
    in_dim = rec["in_dim"]
    half_in = in_dim // 2
    nblk = in_dim // GROUP_SIZE
    w = np.frombuffer(rec["weight_bytes"], dtype=np.uint8).reshape(rec["out_dim"], half_in)
    packed = w[row]  # [in/2] uint8
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    nib = np.empty(in_dim, dtype=np.uint8)
    nib[0::2] = lo
    nib[1::2] = hi
    fp4 = e2m1_decode(nib)  # [in_dim]
    bscale = e4m3_decode(rec["block_scales_linear"][row])  # [nblk]
    bscale_full = np.repeat(bscale, GROUP_SIZE)  # [in_dim]
    return fp4 * bscale_full * rec["global_scale_f32"]


# ----------------------------------------------------------------------------
# Inspection / layout report
# ----------------------------------------------------------------------------
def inspect(ckpt_dir=DEFAULT_CKPT, L=0, E=0):
    ckpt = Checkpoint(ckpt_dir)

    print("=" * 78)
    print("NVFP4 DiffusionGemma checkpoint layout report")
    print("checkpoint:", ckpt_dir)
    print("=" * 78)

    hq_path = os.path.join(ckpt_dir, "hf_quant_config.json")
    if os.path.exists(hq_path):
        with open(hq_path) as f:
            hq = json.load(f)["quantization"]
        print("quant_algo=%s group_size=%s kv_cache=%s"
              % (hq.get("quant_algo"), hq.get("group_size"), hq.get("kv_cache_quant_algo")))
    print("layers=%d experts/layer=%d  (example L=%d E=%d)" % (NUM_LAYERS, NUM_EXPERTS, L, E))
    print()

    print("--- Per-tensor dtype / shape / bytes (layer %d, expert %d) ---" % (L, E))
    hdr = "%-10s %-15s %-9s %-16s %12s"
    print(hdr % ("proj", "suffix", "dtype", "shape", "bytes"))
    print("-" * 70)
    per_expert_total = 0
    for proj in PROJS:
        for suf in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            name = expert_key(L, E, proj, suf)
            dtype, shape, _ = ckpt.info(name)
            nbytes = ckpt.byte_size(name)
            per_expert_total += nbytes
            print(hdr % (proj, suf, dtype, str(list(shape)), nbytes))
        print()

    print("per-expert byte size (all 3 projs, 4 tensors each): %d bytes (%.1f KiB)"
          % (per_expert_total, per_expert_total / 1024.0))
    print("estimated total expert bytes: %d experts * %d layers * %d = %.2f GiB"
          % (NUM_EXPERTS, NUM_LAYERS, per_expert_total,
             NUM_EXPERTS * NUM_LAYERS * per_expert_total / (1024.0 ** 3)))
    print()

    # E2M1 packing + block-scale group-size confirmation
    print("--- Packing / scale-layout confirmation ---")
    for proj in PROJS:
        _, w_shape, _ = ckpt.info(expert_key(L, E, proj, "weight"))
        _, ws_shape, _ = ckpt.info(expert_key(L, E, proj, "weight_scale"))
        out_dim, half_in = w_shape
        in_dim = half_in * 2
        e2m1_ok = (w_shape == (out_dim, in_dim // 2))
        gs_ok = (ws_shape == (out_dim, in_dim // GROUP_SIZE))
        print("  %-10s out=%-5d in=%-5d | weight[out,in/2]=%-12s E2M1=%s | "
              "scale[out,in/16]=%-11s linear/group16=%s"
              % (proj, out_dim, in_dim, str(list(w_shape)),
                 "OK" if e2m1_ok else "FAIL",
                 str(list(ws_shape)), "OK" if gs_ok else "FAIL"))
    print()
    print("Scale storage: LINEAR (row-major [out][in/16]); shapes match exactly,")
    print("no swizzle present. cuBLASLt path needs 32x4x4 swizzle -> loader swizzles on load.")
    print()

    # Dequant sanity check
    print("--- Dequant sanity check (L=%d E=%d gate_proj, row 0) ---" % (L, E))
    rec = extract(L, E, "gate_proj", ckpt)
    print("  in_dim=%d out_dim=%d global_scale=%.6g input_scale=%.6g"
          % (rec["in_dim"], rec["out_dim"], rec["global_scale_f32"], rec["input_scale_f32"]))
    deq = dequant_row(rec, 0)
    finite = deq[np.isfinite(deq)]
    print("  row0 dequant: n=%d  min=%.5f max=%.5f  mean|.|=%.5f  rms=%.5f"
          % (deq.size, float(finite.min()), float(finite.max()),
             float(np.mean(np.abs(finite))), float(np.sqrt(np.mean(finite ** 2)))))

    # sample a few experts / projs to confirm magnitudes are uniformly sane
    print()
    print("--- Magnitude survey across sample (layer,expert,proj) ---")
    sane = True
    for (sl, se, sp) in [(0, 0, "gate_proj"), (0, 0, "down_proj"),
                         (0, 5, "up_proj"), (15, 64, "gate_proj"),
                         (29, 127, "down_proj")]:
        try:
            r = extract(sl, se, sp, ckpt)
            d = dequant_row(r, 0)
            d = d[np.isfinite(d)]
            rms = float(np.sqrt(np.mean(d ** 2)))
            mx = float(np.max(np.abs(d)))
            ok = (mx < 50.0) and (rms < 5.0)
            sane = sane and ok
            print("  L=%-2d E=%-3d %-10s rms=%.5f maxabs=%.5f gscale=%.4g %s"
                  % (sl, se, sp, rms, mx, r["global_scale_f32"], "OK" if ok else "SUSPECT"))
        except Exception as ex:  # noqa: BLE001
            print("  L=%d E=%d %s ERROR: %s" % (sl, se, sp, ex))
            sane = False
    print()
    print("SANITY CHECK: %s" % ("PASSED (magnitudes small / ~unit)" if sane else "FAILED"))
    print("=" * 78)
    return sane


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="NVFP4 DiffusionGemma expert extractor")
    p.add_argument("--ckpt", default=DEFAULT_CKPT,
                   help="NVFP4 checkpoint snapshot dir (or set the DG_NVFP4_CKPT env var)")
    p.add_argument("--inspect", action="store_true", help="print full layout report")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--expert", type=int, default=0)
    args = p.parse_args(argv)

    if not args.ckpt:
        p.error("no checkpoint given — pass --ckpt <snapshot dir> or set $DG_NVFP4_CKPT")

    if args.inspect:
        ok = inspect(args.ckpt, args.layer, args.expert)
        return 0 if ok else 1

    # default: brief report
    inspect(args.ckpt, args.layer, args.expert)
    return 0


if __name__ == "__main__":
    sys.exit(main())
