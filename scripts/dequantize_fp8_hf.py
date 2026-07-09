#!/usr/bin/env python3
"""Materialize a block-FP8 Hugging Face checkpoint as autograd-capable BF16.

Official Qwen3.5 FP8 checkpoints use custom inference-only matmul operators whose PyTorch
implementations do not register backward formulas. Anthropic's Jacobian-lens fitter requires
input gradients. This converter exactly applies each ``weight_scale_inv`` block and writes normal
BF16 linear weights, allowing transformers + autograd to fit a lens. The source is never modified.

This is an offline interpretability artifact (roughly 2x the FP8 checkpoint size), not a fucina
runtime format. Fucina should continue loading the original FP8 checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("destination", type=Path)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ModuleNotFoundError as exc:
        raise SystemExit(f"missing dependency {exc.name}; install torch and safetensors") from exc

    source = args.source.resolve()
    destination = args.destination.resolve()
    index_path = source / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"missing {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    if destination.exists():
        if not args.force:
            raise SystemExit(f"{destination} exists; pass --force to replace it")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".", dir=destination.parent))

    handles = {}
    try:
        for shard in shards:
            handles[shard] = safe_open(source / shard, framework="pt", device="cpu")

        def tensor(name: str):
            return handles[weight_map[name]].get_tensor(name)

        output_map: dict[str, str] = {}
        total_size = 0
        for shard_index, shard in enumerate(shards, 1):
            names = sorted(name for name, location in weight_map.items()
                           if location == shard and not name.endswith("_scale_inv"))
            converted = {}
            quantized = 0
            for name in names:
                value = tensor(name)
                scale_name = name + "_scale_inv"
                if scale_name in weight_map:
                    if value.ndim != 2:
                        raise RuntimeError(f"{name}: block-scaled tensor is not a matrix: {value.shape}")
                    scale = tensor(scale_name).float()
                    rows, cols = value.shape
                    expected = ((rows + 127) // 128, (cols + 127) // 128)
                    if tuple(scale.shape) != expected:
                        raise RuntimeError(f"{scale_name}: shape {tuple(scale.shape)}, expected {expected}")
                    expanded = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
                    value = (value.float() * expanded).to(torch.bfloat16).contiguous()
                    quantized += 1
                else:
                    value = value.contiguous()
                converted[name] = value
                output_map[name] = shard
            out_path = temporary / shard
            print(f"[{shard_index}/{len(shards)}] {shard}: {len(names)} tensors, "
                  f"dequantized {quantized}", flush=True)
            save_file(converted, out_path, metadata={"format": "pt"})
            total_size += out_path.stat().st_size
            del converted

        # Copy all non-weight assets and remove the quantizer declaration from config.json so
        # transformers constructs ordinary nn.Linear modules with registered autograd.
        for item in source.iterdir():
            if item.name.startswith("model.safetensors"):
                continue
            if item.is_file():
                shutil.copy2(item, temporary / item.name)
        config_path = temporary / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.pop("quantization_config", None)
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        out_index = {"metadata": {"total_size": total_size}, "weight_map": output_map}
        (temporary / index_path.name).write_text(json.dumps(out_index, indent=2) + "\n",
                                                  encoding="utf-8")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        handles.clear()

    print(f"wrote autograd-capable BF16 checkpoint: {destination} ({total_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
