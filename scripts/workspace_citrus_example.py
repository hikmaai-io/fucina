#!/usr/bin/env python3
"""Replicate one experiment from Anthropic's 2026 global-workspace paper with fucina.

This reproduces the qualitative directed-modulation example from Figure 9: the model copies
"The old painting hung crookedly on the wall" while either concentrating on citrus fruits or
receiving no side task. It compares J-Lens words at the generated token covering "ook" in
"crookedly". A successful qualitative replication has citrus words (orange, lemon, fruit, ...)
in the focus trace but not the control trace.

The lens must have been fitted on the exact checkpoint and converted with convert_jlens.py.
This is an interpretability experiment, not a production benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

SENTENCE = "The old painting hung crookedly on the wall."
FOCUS_PROMPT = (
    "While you copy the following sentence exactly, concentrate on citrus fruits. "
    "Output only the sentence and do not mention the side task. Sentence: " + SENTENCE
)
CONTROL_PROMPT = "Copy the following sentence exactly and output only that sentence. Sentence: " + SENTENCE
TARGET_WORDS = {"citrus", "orange", "lemon", "lime", "fruit", "grapefruit", "tangerine"}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def run_condition(args: argparse.Namespace, name: str, prompt: str) -> Path:
    trace = args.output_dir / f"{name}.jsonl"
    stdout = args.output_dir / f"{name}.stdout.txt"
    stderr = args.output_dir / f"{name}.stderr.txt"
    cmd = [
        str(args.fucina), "-m", str(args.model), "--ctx", str(args.ctx),
        "--interactive", "--jspace", "--j-lens", str(args.lens),
        "--jspace-out", str(trace), "--jspace-top-k", str(args.top_k),
        "--temp", "0", "--thinking", "off", "--spec=false", "-n", str(args.max_new_tokens),
    ]
    env = os.environ.copy()
    env.setdefault("FUCINA_PAGED_MAXSEQS", "1")
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, input=prompt + "\n/quit\n", text=True, stdout=out, stderr=err,
                              env=env, timeout=args.timeout)
    if proc.returncode:
        raise RuntimeError(f"{name} run failed ({proc.returncode}); inspect {stderr}")
    if not trace.exists() or not trace.stat().st_size:
        raise RuntimeError(f"{name} produced no J-space records; inspect {stderr}")
    return trace


def lens_metadata(path: Path) -> dict[str, int]:
    with path.open("rb") as f:
        magic = f.read(8)
        header = f.read(20)
    if magic != b"FJSPACE1" or len(header) != 20:
        raise ValueError(f"{path} is not an FJSPACE1 lens")
    version, hidden_size, model_layers, fitted_layers, n_prompts = struct.unpack("<IIIII", header)
    if version != 1:
        raise ValueError(f"unsupported FJSPACE version {version}")
    return {"hidden_size": hidden_size, "model_layers": model_layers,
            "fitted_layers": fitted_layers, "n_prompts": n_prompts}


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def probe_record(records: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    # Record 0's source is the final prompt token. Subsequent source tokens are generated text.
    pieces = [str(row.get("source_token", "")) for row in records[1:]]
    text = "".join(pieces)
    low = text.lower()
    start = low.find("crookedly")
    if start < 0:
        sampled = "".join(str(row.get("sampled_token", "")) for row in records)
        raise ValueError(
            "model did not copy 'crookedly' exactly; generated text was: " + repr(sampled)
        )
    # Anthropic probes the token containing "ook". Select the generated source-token span that
    # contains its middle character; this remains valid when Qwen tokenizes the word differently.
    probe_char = start + len("cro") + 1
    cursor = 0
    for row, piece in zip(records[1:], pieces):
        if cursor <= probe_char < cursor + len(piece):
            return row, text
        cursor += len(piece)
    raise ValueError("could not map the 'ook' character back to a source token")


def norm_word(value: str) -> str:
    return value.strip().lower().replace("▁", "")


def is_target_word(value: str) -> bool:
    word = norm_word(value)
    if word in TARGET_WORDS:
        return True
    # Tokenizers may split a concept as "orang" + "e". Accept informative stems of at least four
    # characters, but not tiny fragments such as "it"/"rus" that create generic false positives.
    return len(word) >= 4 and any(target.startswith(word) or word.startswith(target)
                                  for target in TARGET_WORDS)


def summarize(path: Path) -> dict[str, Any]:
    records = load_records(path)
    row, generated = probe_record(records)
    layers = []
    hit_count = 0
    for layer in row["layers"]:
        words = [item["token"] for item in layer["top"]]
        hits = [norm_word(word) for word in words if is_target_word(word)]
        hit_count += bool(hits)
        layers.append({"layer": layer["layer"], "words": words, "target_hits": hits})
    return {
        "trace": str(path),
        "probe_source_position": row["source_position"],
        "probe_source_token": row["source_token"],
        "sampled_next_token": row["sampled_token"],
        "generated_text": generated,
        "layers_with_citrus_word": hit_count,
        "layers": layers,
    }


def print_summary(name: str, result: dict[str, Any]) -> None:
    print(f"\n{name.upper()}: source token {result['probe_source_token']!r} at "
          f"position {result['probe_source_position']}; citrus hits in "
          f"{result['layers_with_citrus_word']} fitted layer(s)")
    for layer in result["layers"]:
        rendered = ", ".join(repr(word) for word in layer["words"])
        marker = "  <-- CITRUS" if layer["target_hits"] else ""
        print(f"  L{layer['layer']:>2}: {rendered}{marker}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fucina", type=Path, default=Path("./fucina"))
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--lens", type=Path, required=True, help="matching FJSPACE1 lens")
    ap.add_argument("--output-dir", type=Path, default=Path("workspace-citrus-run"))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--analyze-only", action="store_true",
                    help="reuse control.jsonl and focus.jsonl in --output-dir")
    args = ap.parse_args()
    args.fucina = args.fucina.resolve()
    args.model = args.model.resolve()
    args.lens = args.lens.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        control = run_condition(args, "control", CONTROL_PROMPT)
        focus = run_condition(args, "focus", FOCUS_PROMPT)
    else:
        control = args.output_dir / "control.jsonl"
        focus = args.output_dir / "focus.jsonl"

    try:
        results = {"paper_example": "directed modulation / citrus while copying",
                   "lens": lens_metadata(args.lens),
                   "control": summarize(control), "focus": summarize(focus)}
    except ValueError as exc:
        raise SystemExit(f"INCONCLUSIVE: {exc}") from exc
    print_summary("control", results["control"])
    print_summary("focus", results["focus"])
    delta = (results["focus"]["layers_with_citrus_word"] -
             results["control"]["layers_with_citrus_word"])
    results["focus_minus_control_layer_hits"] = delta
    if delta > 0 and results["lens"]["n_prompts"] >= 8:
        verdict = "QUALITATIVE REPLICATION"
    elif delta > 0:
        verdict = "POSITIVE SMOKE SIGNAL (lens corpus too small for a replication claim)"
    else:
        verdict = "NO REPLICATION IN THIS RUN"
    results["verdict"] = verdict
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n{verdict}: focus-control citrus layer delta = {delta:+d}")
    print(f"summary: {summary_path}")
    if delta <= 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
