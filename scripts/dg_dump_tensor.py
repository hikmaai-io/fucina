#!/usr/bin/env python3
"""Dump raw quantized bytes + reference fp32 dequant for chosen DiffusionGemma tensors.

Used by cuda/test_diffusion_dequant.cu to validate the CUDA dequant kernels bit-for-bit
against the canonical gguf-py / ggml dequantization.

Output per tensor into <outdir>:
  <key>.raw       raw on-disk quantized block bytes
  <key>.ref       reference dequantized float32 (little-endian)
  <key>.meta      one line: "<ggml_type_id> <n_elem> <raw_bytes>"
"""
import os
import sys

# Path to llama.cpp's gguf-py. Only needed if `gguf` is not already importable;
# override with the LLAMA_GGUF_PY environment variable.
_gguf_py = os.environ.get("LLAMA_GGUF_PY")
if _gguf_py:
    sys.path.insert(0, _gguf_py)

import numpy as np  # noqa: E402
import gguf  # noqa: E402
import gguf.quants as gq  # noqa: E402

# DiffusionGemma GGUF to dump from. Override with the DG_GGUF env var.
GGUF_PATH = os.environ.get("DG_GGUF", "./models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf")

# One representative tensor per quant format present in the model.
TENSORS = [
    ("blk.0.attn_q.weight", "q4_k"),   # Q4_K  super-block
    ("self_cond_down.weight", "q5_0"), # Q5_0  legacy 5-bit
    ("blk.0.attn_v.weight", "q6_k"),   # Q6_K  super-block (sliding V)
    ("blk.0.ffn_down.weight", "q8_0"), # Q8_0  already supported (sanity)
]


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dg_dequant"
    os.makedirs(outdir, exist_ok=True)
    r = gguf.GGUFReader(GGUF_PATH)
    by_name = {t.name: t for t in r.tensors}

    for name, tag in TENSORS:
        t = by_name[name]
        raw = np.asarray(t.data).tobytes()          # on-disk quantized block bytes
        n_elem = int(np.prod(t.shape))
        ref = gq.dequantize(np.frombuffer(raw, dtype=np.uint8), t.tensor_type)
        ref = np.asarray(ref, dtype=np.float32).reshape(-1)
        assert ref.size == n_elem, f"{name}: ref {ref.size} != n_elem {n_elem}"

        base = os.path.join(outdir, tag)
        with open(base + ".raw", "wb") as f:
            f.write(raw)
        ref.tofile(base + ".ref")
        with open(base + ".meta", "w") as f:
            f.write(f"{int(t.tensor_type)} {n_elem} {len(raw)}\n")
        print(f"{tag:5s} {name:28s} type={int(t.tensor_type):2d} "
              f"n_elem={n_elem:9d} raw={len(raw):9d}B "
              f"ref[0:3]={ref[:3]}")
    print(f"\nwrote dumps to {outdir}")


if __name__ == "__main__":
    main()
