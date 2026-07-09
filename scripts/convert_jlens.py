#!/usr/bin/env python3
"""Convert an Anthropic ``JacobianLens.save()`` checkpoint to fucina FJSPACE1.

The native CUDA engine intentionally does not embed Python or a pickle reader. This offline tool
uses torch's weights-only loader, validates the matrices, and emits a tiny deterministic format:

    magic[8] = FJSPACE1
    uint32 version, d_model, model_layers, entries, n_prompts
    repeated: int32 layer, uint32 reserved, float16 J[d_model,d_model] row-major

Example:
    python scripts/convert_jlens.py lens.pt lens.fjls --model-layers 32
"""
from __future__ import annotations

import argparse
import os
import struct
import tempfile
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "convert_jlens.py requires PyTorch (run it in the environment used to fit/load the J-Lens)"
    ) from exc


def parse_layers(value: str | None) -> set[int] | None:
    if not value:
        return None
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="JacobianLens .pt checkpoint")
    ap.add_argument("output", type=Path, help="output .fjls file")
    ap.add_argument("--model-layers", type=int, required=True,
                    help="decoder layer count of the exact fucina target model")
    ap.add_argument("--layers", help="optional subset, e.g. 0,4,8,12-20,31")
    args = ap.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=True)
    if "J" not in ckpt or "d_model" not in ckpt:
        raise SystemExit(f"{args.input} is not an Anthropic JacobianLens checkpoint")
    d_model = int(ckpt["d_model"])
    selected = parse_layers(args.layers)
    matrices: list[tuple[int, torch.Tensor]] = []
    for raw_layer, raw_j in ckpt["J"].items():
        layer = int(raw_layer)
        if selected is not None and layer not in selected:
            continue
        if not 0 <= layer < args.model_layers:
            raise SystemExit(f"lens layer {layer} outside target model [0,{args.model_layers})")
        j = raw_j.detach().to(device="cpu", dtype=torch.float16).contiguous()
        if tuple(j.shape) != (d_model, d_model):
            raise SystemExit(f"layer {layer}: expected {(d_model, d_model)}, got {tuple(j.shape)}")
        if not torch.isfinite(j).all():
            raise SystemExit(f"layer {layer}: matrix contains NaN/Inf")
        matrices.append((layer, j))
    matrices.sort(key=lambda item: item[0])
    if not matrices:
        raise SystemExit("no matrices selected")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=args.output.name + ".", dir=args.output.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(b"FJSPACE1")
            f.write(struct.pack("<IIIII", 1, d_model, args.model_layers, len(matrices),
                                int(ckpt.get("n_prompts", 0))))
            for layer, matrix in matrices:
                f.write(struct.pack("<iI", layer, 0))
                f.write(matrix.numpy().tobytes(order="C"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, args.output)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

    gib = args.output.stat().st_size / (1024**3)
    print(f"wrote {args.output}: H={d_model}, layers={[x[0] for x in matrices]}, {gib:.3f} GiB")


if __name__ == "__main__":
    main()
