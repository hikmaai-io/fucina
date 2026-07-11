#!/usr/bin/env python3
# ABOUTME: Builds a deterministic, provenance-carrying JSONL calibration corpus from local sources.
# ABOUTME: Enforces the Phase-B coding/red-team category mix without downloading or hiding datasets.
"""Build a weighted, reproducible calibration corpus.

Recipe sources are objects with: path, category, provenance, license (optional). Input may be
plain text (one record per line) or JSONL containing text/prompt/content/messages. The output keeps
the original JSON value under `sample` and adds provenance. fucina-calibrate understands it via
the recursive content extractor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def text_of(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(filter(None, (text_of(x) for x in value)))
    if isinstance(value, dict):
        for key in ("text", "prompt"):
            if isinstance(value.get(key), str):
                return value[key]
        for key in ("messages", "content", "sample"):
            if key in value:
                return text_of(value[key])
    return ""


def load_source(source: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    path = (root / source["path"]).resolve()
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                sample = line
            text = text_of(sample)
            if not text:
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            records.append({
                "category": source["category"],
                "provenance": source["provenance"],
                "license": source.get("license", "unspecified"),
                "source_path": source["path"],
                "source_line": line_no,
                "sha256": digest,
                "sample": sample,
                "_estimated_tokens": max(1, len(text.encode("utf-8")) // 4),
            })
    return records


def allocate(categories: dict[str, float], available: dict[str, list], target: int) -> dict[str, int]:
    weights = {k: float(v) for k, v in categories.items() if v > 0 and available.get(k)}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("recipe has no non-empty weighted categories")
    return {k: max(1, round(target * w / total)) for k, w in weights.items()}


def build(recipe_path: Path, output: Path, manifest: Path | None = None) -> dict[str, Any]:
    recipe = json.loads(recipe_path.read_text())
    root = recipe_path.parent
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for source in recipe.get("sources", []):
        for record in load_source(source, root):
            if record["sha256"] not in seen:
                seen.add(record["sha256"])
                grouped[record["category"]].append(record)
    rng = random.Random(int(recipe.get("seed", 0)))
    for rows in grouped.values():
        rng.shuffle(rows)
    target = int(recipe.get("target_tokens", 3_000_000))
    budget = allocate(recipe["categories"], grouped, target)
    chosen: list[dict[str, Any]] = []
    actual: dict[str, int] = defaultdict(int)
    # Weighted deficit scheduling keeps the running corpus mixed instead of category-blocked.
    cursor = defaultdict(int)
    while sum(actual.values()) < target:
        candidates = [k for k in budget if cursor[k] < len(grouped[k]) and actual[k] < budget[k]]
        if not candidates:
            break
        category = max(candidates, key=lambda k: budget[k] - actual[k])
        row = grouped[category][cursor[category]]
        cursor[category] += 1
        actual[category] += row.pop("_estimated_tokens")
        chosen.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in chosen:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    result = {
        "format": "fucina-calibration-manifest-v1",
        "recipe": str(recipe_path),
        "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        "output": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "records": len(chosen),
        "estimated_tokens": sum(actual.values()),
        "estimated_tokens_by_category": dict(sorted(actual.items())),
        "unique_input_records": len(seen),
        "note": "Token counts are UTF-8/4 estimates; fucina-calibrate enforces the exact tokenizer budget.",
    }
    manifest = manifest or output.with_suffix(output.suffix + ".manifest.json")
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=Path, default=Path("calibration/corpus-recipe.json"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()
    result = build(args.recipe, args.out, args.manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
