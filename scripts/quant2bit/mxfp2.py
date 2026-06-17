#!/usr/bin/env python3
"""
mxfp2.py — 2-bit microscaled weight codec (MXFP2 / NF2), mirroring the repo's
NVFP4 codec at 2-bit.

Layout (per 16-element block):
  - 1 FP8 (E4M3) block scale (absmax-derived)
  - 16 x 2-bit codes
  => 16*2 bits codes + 8 bits scale = 40 bits / 16 elts = 2.5 bit/elt
     = 0.3125 bytes/elt.  31B params -> 31e9 * 0.3125 / 1e9 = ~9.7 GB.

Two code variants:
  v1 "E1M0": signed levels {-2, -1, +1, +2}  (no zero; 1 exponent bit, 0 mantissa)
  v2 "NF2" : 4-entry normal-float codebook = quantiles of a standard normal,
             symmetric, scaled so the outermost level maps to absmax.

Scale derivation:
  absmax = max |w| over the block.
  For each variant we pick the scale s so that the largest-magnitude code level
  maps onto absmax (i.e. s = absmax / max_level). s is then rounded to the
  nearest representable E4M3 value (quantize_e4m3), matching NVFP4's FP8 block
  scale storage. Decode = code_level[idx] * s_e4m3.

Encoding picks, per element, the code minimizing |w - level*s|.

This module depends only on numpy and reuses e4m3 decode semantics from
dg_nvfp4_convert (OCP E4M3, bias 7).
"""
import numpy as np

BLOCK = 16  # microscale group size (matches NVFP4 group_size=16)

# ---- variant level tables (unit scale; the block scale stretches them) ----
# v1 E1M0: signed magnitudes {1,2}, no zero.
LEVELS_E1M0 = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)
# v2 NF2: 4 symmetric quantiles of a standard normal. The standard NF
# construction places levels at the inverse-CDF of evenly spaced probabilities
# in the interior of (0,1). For 4 symmetric non-zero levels we use the
# half-quantile offsets {1/8,3/8,5/8,7/8} -> normalized so |outer|=1, then we
# rescale below so outer maps to absmax.
_q = np.array([1.0 / 8, 3.0 / 8, 5.0 / 8, 7.0 / 8], dtype=np.float64)
# inverse normal CDF (probit) via erfinv
from math import sqrt
try:
    from scipy.special import erfinv  # noqa
    _probit = sqrt(2.0) * erfinv(2 * _q - 1)
except Exception:  # no scipy; use a rational approximation of probit
    def _erfinv(y):
        # Winitzki approximation
        a = 0.147
        ln = np.log(1 - y * y)
        t = 2 / (np.pi * a) + ln / 2
        return np.sign(y) * np.sqrt(np.sqrt(t * t - ln / a) - t)
    _probit = sqrt(2.0) * _erfinv(2 * _q - 1)
LEVELS_NF2 = (_probit / _probit[-1]).astype(np.float32)  # normalize outer to 1

VARIANTS = {
    "e1m0": LEVELS_E1M0,
    "nf2": LEVELS_NF2,
}


# ---- E4M3 (OCP, bias 7) encode/decode ----
def e4m3_decode(u8):
    u8 = np.asarray(u8, dtype=np.uint8)
    sign = np.where((u8 & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exp = ((u8 >> 3) & 0x0F).astype(np.int32)
    man = (u8 & 0x07).astype(np.float32)
    norm = (1.0 + man / 8.0) * np.power(2.0, (exp - 7).astype(np.float32))
    sub = (man / 8.0) * (2.0 ** -6)
    out = np.where(exp == 0, sub, norm).astype(np.float32)
    nan_mask = (exp == 0x0F) & (man == 7.0)
    out = np.where(nan_mask, np.float32(np.nan), out)
    return sign * out


# Precompute the full positive E4M3 value table (non-negative, finite) for
# nearest-value rounding of scales.
def _build_e4m3_pos_table():
    codes = np.arange(128, dtype=np.uint8)  # sign=0
    vals = e4m3_decode(codes)
    ok = np.isfinite(vals)
    codes = codes[ok]
    vals = vals[ok]
    order = np.argsort(vals)
    return codes[order], vals[order]


_E4M3_CODES, _E4M3_VALS = _build_e4m3_pos_table()
E4M3_MAX = float(_E4M3_VALS.max())   # 448.0
E4M3_MIN_POS = float(_E4M3_VALS[_E4M3_VALS > 0].min())  # smallest subnormal


def quantize_e4m3(x):
    """Round non-negative scalar(s) x to nearest representable E4M3 magnitude.
    Returns the rounded float value (what the decoder would see)."""
    x = np.asarray(x, dtype=np.float32)
    xc = np.clip(x, 0.0, E4M3_MAX)
    idx = np.searchsorted(_E4M3_VALS, xc)
    idx = np.clip(idx, 1, len(_E4M3_VALS) - 1)
    lo = _E4M3_VALS[idx - 1]
    hi = _E4M3_VALS[idx]
    pick_hi = (np.abs(hi - xc) < np.abs(xc - lo))
    out = np.where(pick_hi, hi, lo).astype(np.float32)
    out = np.where(xc <= 0, 0.0, out)
    return out


def quantize(w, variant):
    """Quantize a 1-D float32 weight array with MXFP2.

    Returns (codes_uint8, scales_e4m3_float32, n_blocks). The encoded weight
    array length must be a multiple of BLOCK; the tail (if any) is padded with
    zeros and the padding is dropped on dequant via the returned `n`.
    """
    levels = VARIANTS[variant]
    max_level = float(np.max(np.abs(levels)))
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    n = w.size
    pad = (-n) % BLOCK
    if pad:
        w = np.concatenate([w, np.zeros(pad, dtype=np.float32)])
    nb = w.size // BLOCK
    wb = w.reshape(nb, BLOCK)
    absmax = np.max(np.abs(wb), axis=1)  # [nb]
    raw_scale = absmax / max_level
    # avoid zero scale on all-zero blocks
    raw_scale = np.where(raw_scale <= 0, E4M3_MIN_POS, raw_scale)
    scale = quantize_e4m3(raw_scale)  # [nb] rounded E4M3 magnitudes
    scale = np.where(scale <= 0, E4M3_MIN_POS, scale).astype(np.float32)
    # decode levels in real units: levels * scale -> [nb, 4]
    lvl_real = levels[None, :] * scale[:, None]  # [nb,4]
    # assign each weight to nearest level
    # dist[nb,BLOCK,4]
    diff = wb[:, :, None] - lvl_real[:, None, :]
    idx = np.argmin(np.abs(diff), axis=2).astype(np.uint8)  # [nb,BLOCK]
    return idx.reshape(-1)[:n] if False else idx, scale, n


def quantize_mse(w, variant, n_scales=24):
    """Like quantize() but searches per-block over a small grid of E4M3 scales
    (around the absmax-derived scale) and keeps the one minimizing block MSE.
    Still stores exactly one E4M3 scale + 2-bit codes per block, so the on-disk
    format and bytes/elt are identical to quantize(); only the encoder is
    smarter. This reports the achievable floor of the format."""
    levels = VARIANTS[variant]
    max_level = float(np.max(np.abs(levels)))
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    n = w.size
    pad = (-n) % BLOCK
    if pad:
        w = np.concatenate([w, np.zeros(pad, dtype=np.float32)])
    nb = w.size // BLOCK
    wb = w.reshape(nb, BLOCK)
    absmax = np.max(np.abs(wb), axis=1)
    base = absmax / max_level
    base = np.where(base <= 0, E4M3_MIN_POS, base)
    # candidate multipliers on the base scale (covers shrinking the scale, which
    # lowers MSE for non-uniform/gaussian-ish weights)
    mults = np.linspace(0.45, 1.10, n_scales).astype(np.float32)
    best_idx = np.zeros((nb, BLOCK), dtype=np.uint8)
    best_scale = np.empty(nb, dtype=np.float32)
    # chunk over blocks to bound peak memory (the [chunk,16,4] broadcast)
    CH = 1 << 20  # blocks per chunk -> ~256 MB peak per candidate
    for s in range(0, nb, CH):
        e = min(s + CH, nb)
        wc = wb[s:e]                       # [c,BLOCK]
        bc = base[s:e]
        best_mse = np.full(e - s, np.inf, dtype=np.float32)
        bidx = np.zeros((e - s, BLOCK), dtype=np.uint8)
        bscl = np.empty(e - s, dtype=np.float32)
        for m in mults:
            scale = quantize_e4m3(bc * m)
            scale = np.where(scale <= 0, E4M3_MIN_POS, scale).astype(np.float32)
            lvl_real = levels[None, :] * scale[:, None]      # [c,4]
            diff = wc[:, :, None] - lvl_real[:, None, :]     # [c,BLOCK,4]
            idx = np.argmin(np.abs(diff), axis=2).astype(np.uint8)
            rec = levels[idx] * scale[:, None]
            mse = np.mean((rec - wc) ** 2, axis=1)
            upd = mse < best_mse
            best_mse = np.where(upd, mse, best_mse)
            bscl = np.where(upd, scale, bscl)
            bidx[upd] = idx[upd]
        best_idx[s:e] = bidx
        best_scale[s:e] = bscl
    return best_idx, best_scale.astype(np.float32), n


def roundtrip_mse(w, variant):
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    idx, scale, n = quantize_mse(w, variant)
    return dequantize(idx, scale, n, variant)


def dequantize(idx, scale, n, variant):
    """Inverse of quantize. idx [nb,BLOCK] uint8, scale [nb] f32."""
    levels = VARIANTS[variant]
    nb = scale.shape[0]
    lvl = levels[idx]                  # [nb,BLOCK]
    out = lvl * scale[:, None]         # [nb,BLOCK]
    return out.reshape(-1)[:n]


def roundtrip(w, variant):
    """Convenience: quantize then dequantize, returning the reconstruction."""
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    idx, scale, n = quantize(w, variant)
    return dequantize(idx, scale, n, variant)


def bytes_per_elt():
    # 2 bits/code + 8 bits scale / 16 elts
    return (BLOCK * 2 + 8) / 8.0 / BLOCK


def errors(ref, rec):
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    rec = np.asarray(rec, dtype=np.float64).reshape(-1)
    err = rec - ref
    mse = float(np.mean(err ** 2))
    denom = float(np.mean(ref ** 2)) + 1e-30
    rel_mse = mse / denom
    max_abs = float(np.max(np.abs(err)))
    # cosine similarity (captures direction preservation for GEMV)
    cos = float(np.dot(ref, rec) / (np.linalg.norm(ref) * np.linalg.norm(rec) + 1e-30))
    return {
        "rel_mse": rel_mse,
        "rms_err": float(np.sqrt(mse)),
        "max_abs_err": max_abs,
        "cos_sim": cos,
        "sqnr_db": float(10.0 * np.log10(1.0 / (rel_mse + 1e-30))),
    }


if __name__ == "__main__":
    print("E1M0 levels:", LEVELS_E1M0)
    print("NF2  levels:", LEVELS_NF2)
    print("bytes/elt = %.4f" % bytes_per_elt())
    rng = np.random.default_rng(0)
    w = rng.standard_normal(1 << 20).astype(np.float32) * 0.015
    for v in ("e1m0", "nf2"):
        rec = roundtrip(w, v)
        print(v, errors(w, rec))
