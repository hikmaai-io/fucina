#!/usr/bin/env python3
"""Fit an Anthropic Jacobian lens for an exact local Hugging Face checkpoint.

This is the offline companion to fucina's ``--jspace`` mode. It deliberately uses Anthropic's
upstream ``jlens.fit`` implementation rather than approximating the Jacobian in the inference
engine. Fitting is expensive: for hidden width H it performs ceil(H / dim_batch) backwards per
prompt. Start with ``--n-prompts 1`` and a few middle layers, then increase the corpus.

Requirements (in a Python environment separate from the fucina binary):
    pip install torch transformers accelerate
    pip install git+https://github.com/anthropics/jacobian-lens.git

The output .fjls can be loaded directly by fucina. The .pt file remains compatible with the
upstream JacobianLens API.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import struct
import tempfile
import time
from pathlib import Path

DEFAULT_PROMPTS = [
    "The old library stood beside a quiet river, and the evening light reflected from its windows.",
    "A careful engineer checks each assumption before changing a system that other people depend on.",
    "During winter the mountain path is often covered by snow, but the valley remains green.",
    "The committee compared several proposals and recorded both the advantages and the risks.",
    "A musician practiced the difficult passage slowly until every transition sounded natural.",
    "The market opens early on Saturday, when farmers bring vegetables, bread, flowers, and cheese.",
    "Scientists repeated the measurement with a second instrument to rule out calibration error.",
    "After reading the instructions, the student organized the materials and began the experiment.",
    "The train crossed the bridge just before sunrise and arrived at the coastal station on time.",
    "Good documentation explains not only what a function does, but why its constraints matter.",
    "A small wooden boat moved across the lake while clouds gathered above the distant hills.",
    "The doctor listened carefully, asked several questions, and summarized the available options.",
    "When the power returned, the server recovered its journal and verified every committed record.",
    "The museum displayed maps, tools, letters, and photographs from the early polar expeditions.",
    "Before publishing the result, the team tested whether another explanation fit the evidence.",
    "Children watched the baker shape the dough and place each loaf inside the warm stone oven.",
]


def parse_layers(value: str, n_layers: int) -> list[int]:
    out: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            a, b = (int(x) for x in item.split("-", 1))
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(item))
    layers = sorted(out)
    if not layers or layers[0] < 0 or layers[-1] >= n_layers:
        raise ValueError(f"layers must be within [0,{n_layers}); got {layers}")
    return layers


def load_prompts(path: Path | None, n: int) -> list[str]:
    if path:
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.lstrip().startswith("#")]
    else:
        prompts = list(DEFAULT_PROMPTS)
    if n < 1:
        raise ValueError("--n-prompts must be positive")
    if len(prompts) < n:
        raise ValueError(f"requested {n} prompts but corpus contains only {len(prompts)}")
    return prompts[:n]


@contextlib.contextmanager
def transformers_compatible_model_dir(model: Path):
    """Add missing safetensors index metadata without modifying the checkpoint.

    Some official Qwen FP8 snapshots omit the optional metadata object. Fucina's native loader
    accepts that index, while recent transformers requires it. A temporary symlink view keeps the
    source checkpoint immutable and makes both consumers read exactly the same tensor shards.
    """
    index_path = model / "model.safetensors.index.json"
    if not index_path.exists():
        yield model
        return
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if "metadata" in index:
        yield model
        return
    shards = set(index.get("weight_map", {}).values())
    index["metadata"] = {"total_size": sum((model / shard).stat().st_size for shard in shards)}
    with tempfile.TemporaryDirectory(prefix="fucina-jlens-hf-") as tmp:
        view = Path(tmp)
        for source in model.iterdir():
            if source.name != index_path.name:
                os.symlink(source, view / source.name)
        (view / index_path.name).write_text(json.dumps(index), encoding="utf-8")
        yield view


def write_fjspace(lens, path: Path, model_layers: int) -> None:
    matrices = sorted((int(layer), matrix.detach().cpu().to(dtype=__import__("torch").float16)
                       .contiguous()) for layer, matrix in lens.jacobians.items())
    d_model = int(lens.d_model)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(b"FJSPACE1")
            f.write(struct.pack("<IIIII", 1, d_model, model_layers, len(matrices),
                                int(lens.n_prompts)))
            for layer, matrix in matrices:
                f.write(struct.pack("<iI", layer, 0))
                f.write(matrix.numpy().tobytes(order="C"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="exact local HF checkpoint directory")
    ap.add_argument("--output", type=Path, required=True, help="upstream-compatible .pt lens")
    ap.add_argument("--fjls-output", type=Path, help="native fucina output (default: output with .fjls)")
    ap.add_argument("--layers", default="8,12,16,20,24,28",
                    help="source layers/ranges; selected middle layers keep readout cost manageable")
    ap.add_argument("--prompts-file", type=Path, help="one fitting prompt per line")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=64)
    ap.add_argument("--skip-first", type=int, default=8)
    ap.add_argument("--dim-batch", type=int, default=64,
                    help="Jacobian rows per backward; larger is faster but uses more memory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", type=Path,
                    help="resumable fit checkpoint (default: OUTPUT.fit-checkpoint.pt)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    try:
        import jlens
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(f"missing fitting dependency: {exc.name}; see this script's docstring") from exc

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if getattr(config, "quantization_config", None):
        raise SystemExit(
            "this checkpoint still declares inference-only quantization; Jacobian backprop needs "
            "ordinary weights. First run:\n  python scripts/dequantize_fp8_hf.py SOURCE BF16_DEST\n"
            "then pass BF16_DEST to --model (fucina itself should keep using SOURCE)."
        )
    text_config = config.get_text_config()
    n_layers, d_model = int(text_config.num_hidden_layers), int(text_config.hidden_size)
    layers = parse_layers(args.layers, n_layers)
    prompts = load_prompts(args.prompts_file, args.n_prompts)
    if any(len(prompt.split()) <= args.skip_first for prompt in prompts):
        raise SystemExit("a fitting prompt is too short for --skip-first; use longer prompts or lower it")
    passes = math.ceil(d_model / args.dim_batch) * len(prompts)
    checkpoint = args.checkpoint or args.output.with_suffix(".fit-checkpoint.pt")
    fjls_output = args.fjls_output or args.output.with_suffix(".fjls")
    logging.info("model H=%d L=%d; fitting layers=%s prompts=%d, %d backward passes",
                 d_model, n_layers, layers, len(prompts), passes)

    started = time.time()
    with transformers_compatible_model_dir(args.model) as load_path:
        tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            load_path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation="eager", device_map=args.device, trust_remote_code=True,
        )
        model.eval()
        lens_model = jlens.from_hf(model, tokenizer)
        logging.info("model loaded in %.1fs; CUDA allocated %.2f GiB", time.time() - started,
                     torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0)
        lens = jlens.fit(
            lens_model, prompts, source_layers=layers, dim_batch=args.dim_batch,
            max_seq_len=args.max_seq_len, skip_first=args.skip_first,
            checkpoint_path=str(checkpoint), checkpoint_every=1, resume=not args.no_resume,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        lens.save(str(args.output))
        write_fjspace(lens, fjls_output, n_layers)

    logging.info("finished in %.1f minutes", (time.time() - started) / 60)
    logging.info("wrote %s and %s", args.output, fjls_output)


if __name__ == "__main__":
    main()
