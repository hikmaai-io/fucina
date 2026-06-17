#!/usr/bin/env python3
"""
gguf_reader.py — Dependency-light GGUF v2/v3 parser + Q4_0 block dequant.

Parses the GGUF container header (magic, version, n_tensors, n_kv), the KV
metadata block, and the tensor-info table (name, n_dims, dims, ggml_type,
offset). Tensor data follows the header, aligned to `general.alignment`
(default 32). Only numpy is required.

Q4_0 block layout (block of 32 weights):
  - 1 fp16 scale `d`
  - 16 bytes = 32 packed 4-bit nibbles
  Each nibble q in 0..15; dequant: w = (q - 8) * d.
  Storage order in llama.cpp: the low nibbles of all 16 bytes give weights
  0..15, the high nibbles give weights 16..31.

This module exposes:
  GGUFReader(path)            — parse header
  .tensors                   — dict name -> TensorInfo
  .read_raw(name)            — raw tensor bytes
  .dequant_q4_0(name)        — fp32 numpy array (reference)
  .dequant_f32/f16/bf16
"""
import json
import struct
import sys

import numpy as np

# ggml type enum (subset we care about)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14
GGML_TYPE_BF16 = 30

GGML_TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0", 12: "Q4_K", 14: "Q6_K",
    30: "BF16",
}

QK_K = 256  # super-block size for k-quants

# GGUF metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

_SCALAR_FMT = {
    GGUF_TYPE_UINT8: ("<B", 1),
    GGUF_TYPE_INT8: ("<b", 1),
    GGUF_TYPE_UINT16: ("<H", 2),
    GGUF_TYPE_INT16: ("<h", 2),
    GGUF_TYPE_UINT32: ("<I", 4),
    GGUF_TYPE_INT32: ("<i", 4),
    GGUF_TYPE_FLOAT32: ("<f", 4),
    GGUF_TYPE_BOOL: ("<?", 1),
    GGUF_TYPE_UINT64: ("<Q", 8),
    GGUF_TYPE_INT64: ("<q", 8),
    GGUF_TYPE_FLOAT64: ("<d", 8),
}

QK4_0 = 32  # Q4_0 block size


class TensorInfo:
    __slots__ = ("name", "dims", "ggml_type", "rel_offset", "abs_offset", "n_elem")

    def __init__(self, name, dims, ggml_type, rel_offset):
        self.name = name
        self.dims = dims
        self.ggml_type = ggml_type
        self.rel_offset = rel_offset
        self.abs_offset = None
        n = 1
        for d in dims:
            n *= d
        self.n_elem = n

    def __repr__(self):
        return "TensorInfo(%s dims=%s type=%s n=%d)" % (
            self.name, self.dims, GGML_TYPE_NAME.get(self.ggml_type, self.ggml_type),
            self.n_elem)


class _Cursor:
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def take(self, n):
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def u32(self):
        (v,) = struct.unpack_from("<I", self.buf, self.pos)
        self.pos += 4
        return v

    def u64(self):
        (v,) = struct.unpack_from("<Q", self.buf, self.pos)
        self.pos += 8
        return v

    def i32(self):
        (v,) = struct.unpack_from("<i", self.buf, self.pos)
        self.pos += 4
        return v

    def string(self):
        n = self.u64()
        b = self.take(n)
        return b.decode("utf-8", "replace")


class GGUFReader:
    def __init__(self, path, header_bytes=64 * 1024 * 1024):
        self.path = path
        with open(path, "rb") as f:
            head = f.read(header_bytes)
        c = _Cursor(head)
        magic = c.take(4)
        if magic != b"GGUF":
            raise ValueError("not a GGUF file: %r" % magic)
        self.version = c.u32()
        if self.version not in (2, 3):
            raise ValueError("unsupported GGUF version %d" % self.version)
        # v2/v3 use u64 counts
        self.n_tensors = c.u64()
        self.n_kv = c.u64()

        self.kv = {}
        for _ in range(self.n_kv):
            key = c.string()
            val = self._read_value(c)
            self.kv[key] = val

        self.alignment = int(self.kv.get("general.alignment", 32))

        self.tensors = {}
        order = []
        for _ in range(self.n_tensors):
            name = c.string()
            n_dims = c.u32()
            dims = [c.u64() for _ in range(n_dims)]
            ggml_type = c.u32()
            rel_off = c.u64()
            ti = TensorInfo(name, dims, ggml_type, rel_off)
            self.tensors[name] = ti
            order.append(name)
        self._order = order

        # data section start, aligned
        data_start = c.pos
        if data_start % self.alignment != 0:
            data_start += self.alignment - (data_start % self.alignment)
        self.data_start = data_start
        for name in order:
            self.tensors[name].abs_offset = data_start + self.tensors[name].rel_offset

    def _read_value(self, c):
        t = c.u32()
        return self._read_typed(c, t)

    def _read_typed(self, c, t):
        if t in _SCALAR_FMT:
            fmt, sz = _SCALAR_FMT[t]
            (v,) = struct.unpack_from(fmt, c.buf, c.pos)
            c.pos += sz
            return v
        if t == GGUF_TYPE_STRING:
            return c.string()
        if t == GGUF_TYPE_ARRAY:
            elem_t = c.u32()
            n = c.u64()
            if elem_t == GGUF_TYPE_STRING:
                return [c.string() for _ in range(n)]
            fmt, sz = _SCALAR_FMT[elem_t]
            out = list(struct.unpack_from("<" + fmt[1] * n, c.buf, c.pos))
            c.pos += sz * n
            return out
        raise ValueError("unknown GGUF value type %d" % t)

    # ------------------------------------------------------------------
    def read_raw(self, name):
        ti = self.tensors[name]
        nbytes = _tensor_nbytes(ti)
        with open(self.path, "rb") as f:
            f.seek(ti.abs_offset)
            return f.read(nbytes)

    def dequant(self, name):
        """Return tensor as fp32 numpy array regardless of stored type."""
        ti = self.tensors[name]
        t = ti.ggml_type
        if t == GGML_TYPE_Q4_0:
            return self.dequant_q4_0(name)
        if t == GGML_TYPE_Q4_1:
            return self.dequant_q4_1(name)
        if t == GGML_TYPE_Q4_K:
            return self.dequant_q4_k(name)
        if t == GGML_TYPE_F32:
            return np.frombuffer(self.read_raw(name), dtype=np.float32).astype(np.float32)
        if t == GGML_TYPE_F16:
            return np.frombuffer(self.read_raw(name), dtype=np.float16).astype(np.float32)
        if t == GGML_TYPE_BF16:
            u = np.frombuffer(self.read_raw(name), dtype=np.uint16).astype(np.uint32)
            return (u << 16).view(np.float32)
        raise ValueError("dequant not implemented for type %s"
                         % GGML_TYPE_NAME.get(t, t))

    def dequant_q4_0(self, name):
        ti = self.tensors[name]
        if ti.ggml_type != GGML_TYPE_Q4_0:
            raise ValueError("%s is not Q4_0" % name)
        n = ti.n_elem
        if n % QK4_0 != 0:
            raise ValueError("n_elem %d not divisible by %d" % (n, QK4_0))
        nblocks = n // QK4_0
        raw = np.frombuffer(self.read_raw(name), dtype=np.uint8)
        # each block = 2 (fp16 scale) + 16 (nibbles) = 18 bytes
        block_bytes = 2 + QK4_0 // 2
        raw = raw[: nblocks * block_bytes].reshape(nblocks, block_bytes)
        scales = raw[:, 0:2].copy().view(np.float16).astype(np.float32)  # [nb,1]
        scales = scales.reshape(nblocks, 1)
        qs = raw[:, 2:]  # [nb, 16] packed nibbles
        lo = (qs & 0x0F).astype(np.int32)        # weights 0..15
        hi = ((qs >> 4) & 0x0F).astype(np.int32)  # weights 16..31
        codes = np.empty((nblocks, QK4_0), dtype=np.int32)
        codes[:, 0:16] = lo
        codes[:, 16:32] = hi
        out = (codes - 8).astype(np.float32) * scales
        return out.reshape(-1)

    def dequant_q4_1(self, name):
        ti = self.tensors[name]
        if ti.ggml_type != GGML_TYPE_Q4_1:
            raise ValueError("%s is not Q4_1" % name)
        n = ti.n_elem
        nblocks = n // QK4_0
        raw = np.frombuffer(self.read_raw(name), dtype=np.uint8)
        block_bytes = 2 + 2 + QK4_0 // 2  # d (f16) + m (f16) + 16 nibbles
        raw = raw[: nblocks * block_bytes].reshape(nblocks, block_bytes)
        d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nblocks, 1)
        m = raw[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(nblocks, 1)
        qs = raw[:, 4:]
        lo = (qs & 0x0F).astype(np.float32)
        hi = ((qs >> 4) & 0x0F).astype(np.float32)
        codes = np.empty((nblocks, QK4_0), dtype=np.float32)
        codes[:, 0:16] = lo
        codes[:, 16:32] = hi
        out = codes * d + m
        return out.reshape(-1)

    def dequant_q4_k(self, name):
        """Q4_K super-block (256 weights): block layout (144 bytes):
          d      : f16  (super-block scale for the 6-bit scales)
          dmin   : f16  (super-block min)
          scales : 12 bytes -> 8 sub-block 6-bit scales + 8 sub-block 6-bit mins
          qs     : 128 bytes -> 256 4-bit quants
        w = d * sc * q - dmin * mn, per 32-element sub-block.
        """
        ti = self.tensors[name]
        if ti.ggml_type != GGML_TYPE_Q4_K:
            raise ValueError("%s is not Q4_K" % name)
        n = ti.n_elem
        nsb = n // QK_K
        raw = np.frombuffer(self.read_raw(name), dtype=np.uint8)
        block_bytes = 2 + 2 + 12 + 128
        raw = raw[: nsb * block_bytes].reshape(nsb, block_bytes)
        d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nsb, 1)
        dmin = raw[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(nsb, 1)
        sc_bytes = raw[:, 4:16]  # [nsb,12]
        qs = raw[:, 16:]         # [nsb,128]

        # Unpack 8 scales (sc) and 8 mins (m), each 6-bit, llama.cpp scheme.
        sc = np.zeros((nsb, 8), dtype=np.uint8)
        mn = np.zeros((nsb, 8), dtype=np.uint8)
        b = sc_bytes.astype(np.uint16)
        for j in range(8):
            if j < 4:
                sc[:, j] = (b[:, j] & 0x3F).astype(np.uint8)
                mn[:, j] = (b[:, j + 4] & 0x3F).astype(np.uint8)
            else:
                sc[:, j] = (((b[:, j + 4] & 0x0F)) | ((b[:, j - 4] >> 6) << 4)).astype(np.uint8)
                mn[:, j] = (((b[:, j + 4] >> 4)) | ((b[:, j] >> 6) << 4)).astype(np.uint8)
        sc = sc.astype(np.float32)
        mn = mn.astype(np.float32)

        # qs: 128 bytes -> for each of 4 groups of 64 weights (two sub-blocks),
        # low nibble = sub-block 2*g, high nibble = sub-block 2*g+1.
        out = np.empty((nsb, QK_K), dtype=np.float32)
        qs = qs.reshape(nsb, 4, 32)  # 4 groups of 32 bytes
        for g in range(4):
            grp = qs[:, g, :]  # [nsb,32]
            lo = (grp & 0x0F).astype(np.float32)   # sub-block 2g  (32 weights)
            hi = ((grp >> 4) & 0x0F).astype(np.float32)  # sub-block 2g+1
            j0 = 2 * g
            j1 = 2 * g + 1
            out[:, j0 * 32:(j0 + 1) * 32] = d * sc[:, j0:j0 + 1] * lo - dmin * mn[:, j0:j0 + 1]
            out[:, j1 * 32:(j1 + 1) * 32] = d * sc[:, j1:j1 + 1] * hi - dmin * mn[:, j1:j1 + 1]
        return out.reshape(-1)


def _tensor_nbytes(ti):
    t = ti.ggml_type
    n = ti.n_elem
    if t == GGML_TYPE_F32:
        return n * 4
    if t == GGML_TYPE_F16 or t == GGML_TYPE_BF16:
        return n * 2
    if t == GGML_TYPE_Q4_0:
        return (n // QK4_0) * (2 + QK4_0 // 2)
    if t == GGML_TYPE_Q4_1:
        return (n // QK4_0) * (2 + 2 + QK4_0 // 2)
    if t == GGML_TYPE_Q8_0:
        return (n // QK4_0) * (2 + QK4_0)
    if t == GGML_TYPE_Q4_K:
        return (n // QK_K) * (2 + 2 + 12 + 128)
    if t == GGML_TYPE_Q6_K:
        return (n // QK_K) * (128 + 64 + 16 + 2)
    raise ValueError("nbytes unknown for type %s" % GGML_TYPE_NAME.get(t, t))


def main(argv):
    path = argv[1]
    r = GGUFReader(path)
    print("GGUF v%d  n_tensors=%d n_kv=%d alignment=%d data_start=%d"
          % (r.version, r.n_tensors, r.n_kv, r.alignment, r.data_start))
    arch = r.kv.get("general.architecture")
    print("arch=%s" % arch)
    # print a few key KVs
    for k in sorted(r.kv):
        if "block_count" in k or "embedding_length" in k or "feed_forward" in k \
                or "head_count" in k or "context_length" in k:
            print("  %s = %s" % (k, r.kv[k]))
    print("--- first 20 tensors ---")
    for i, name in enumerate(r._order[:20]):
        print("  %s" % r.tensors[name])
    # type histogram
    from collections import Counter
    hist = Counter(GGML_TYPE_NAME.get(t.ggml_type, t.ggml_type)
                   for t in r.tensors.values())
    print("type histogram:", dict(hist))


if __name__ == "__main__":
    main(sys.argv)
