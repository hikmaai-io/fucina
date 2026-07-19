#!/usr/bin/env python3
"""Reproducible Gemma GB10 serving evidence over OpenAI chat SSE.

This is a report-only benchmark collector: it retains every request's TTFT,
inter-event intervals, output, usage, status/error and hashes. Run it once per
independent server start; aggregate the resulting JSON files separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DIVERSE_PROMPTS = [
    "What is 137 + 265? Show the addition, then give the final number.",
    "Write a Python function that returns the first n Fibonacci numbers.",
    "Translate 'Good morning, where is the train station?' into French, German, and Japanese.",
    "Explain recursion in programming in exactly three short sentences.",
    "Continue this sequence and explain the rule: 2, 3, 5, 8, 13,",
    "Write a four-line poem about copper rain. Do not repeat a line.",
    "List three causes of the French Revolution with one sentence each.",
    "What is 17 multiplied by 23? Show your work.",
    "Explain the difference between TCP and UDP in two sentences.",
    "Write valid JSON with keys name, primes, and active; use three prime numbers.",
    "In Spanish, explain why leaves appear green in two sentences.",
    "Give a concise proof that the square root of 2 is irrational.",
    "Write a SQL query selecting the five newest paid orders per customer.",
    "Name the largest planet and state two facts about it.",
    "Summarize photosynthesis for a twelve-year-old in three sentences.",
    "Correct this code and explain the bug: for i in range(3) print(i)",
    "Write a Rust function that reverses a UTF-8 string by characters.",
    "Translate 'Knowledge grows when shared' into Arabic and Italian.",
    "Calculate 999 minus 487 and check the result by addition.",
    "Explain hash-table collisions and name two resolution strategies.",
    "Write five different words beginning with 'trans'.",
    "Describe the water cycle without using the word 'water' more than twice.",
    "Give pseudocode for binary search and state its time complexity.",
    "Answer in Chinese: why does the Moon have phases?",
    "What year did World War II end, and name three Allied powers?",
    "Write a JavaScript function that debounces another function.",
    "Explain the difference between encryption and hashing.",
    "List the first six powers of two starting at one.",
    "Write a haiku about a quiet GPU at midnight.",
    "Describe one advantage and one risk of continuous batching.",
    "Solve 3x + 7 = 31 and verify the solution.",
    "Write a shell command that finds .log files larger than 10 MB.",
]

QUALITY_CASES = [
    ("arithmetic", DIVERSE_PROMPTS[0]),
    ("code_python", DIVERSE_PROMPTS[1]),
    ("multilingual", DIVERSE_PROMPTS[2]),
    ("repetition", "Repeat the exact sequence 'red blue green' eight times, one sequence per line."),
    ("code_rust", DIVERSE_PROMPTS[16]),
    ("reasoning", DIVERSE_PROMPTS[11]),
]

FILL = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
        "mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu ")


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - k) + ys[hi] * (k - lo)


def stats(xs: list[float]) -> dict[str, Any]:
    ys = [x for x in xs if x is not None and math.isfinite(x)]
    if not ys:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "mean": None, "cv": None, "raw": []}
    mean = statistics.fmean(ys)
    sd = statistics.stdev(ys) if len(ys) > 1 else 0.0
    return {
        "n": len(ys), "p50": percentile(ys, .50), "p95": percentile(ys, .95),
        "p99": percentile(ys, .99), "mean": mean, "cv": (sd / mean if mean else None),
        "raw": ys,
    }


def long_prompt(target_words: int, nonce: str) -> str:
    words = (FILL * (target_words // 26 + 2)).split()[:target_words]
    return (f"Evidence nonce {nonce}.\n" + " ".join(words) +
            "\nWrite a detailed analysis of at least 300 words. Continue until the output limit.")


def get_json(url: str, timeout: float = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def request_one(base_url: str, model: str, prompt: str, max_tokens: int,
                request_id: str, ignore_eos: bool = True) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": False,
        "thinking": False,
    }
    if ignore_eos:
        body["ignore_eos"] = True
        body["min_tokens"] = max_tokens
    encoded = json.dumps(body).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions", data=encoded,
        headers={"Content-Type": "application/json", "X-Request-Id": request_id})
    t0 = time.perf_counter()
    stamps: list[float] = []
    pieces: list[str] = []
    usage: dict[str, Any] = {}
    status = None
    error = None
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            status = response.status
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    stamps.append(time.perf_counter())
                    pieces.append(piece)
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = exc.read().decode("utf-8", "replace")[:4000]
    except Exception as exc:  # benchmark must retain failures, not abort the matrix
        error = f"{type(exc).__name__}: {exc}"
    t1 = time.perf_counter()
    text = "".join(pieces)
    event_hasher = hashlib.sha256()
    for piece in pieces:
        b = piece.encode()
        event_hasher.update(len(b).to_bytes(8, "little"))
        event_hasher.update(b)
    completion_tokens = int(usage.get("completion_tokens") or len(stamps))
    itl_ms = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
    ttft_ms = (stamps[0] - t0) * 1000 if stamps else None
    decode_s = stamps[-1] - stamps[0] if len(stamps) > 1 else None
    return {
        "request_id": request_id,
        "http_status": status,
        "error": error,
        "ok": error is None and status == 200,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_event_sha256": event_hasher.hexdigest(),
        "token_events_utf8": pieces,
        "output_utf8": text,
        "usage": usage,
        "completion_tokens": completion_tokens,
        "event_count": len(stamps),
        "event_count_matches_completion_tokens": len(stamps) == completion_tokens,
        "ttft_ms": ttft_ms,
        "itl_ms_raw": itl_ms,
        "wall_ms": (t1 - t0) * 1000,
        "decode_tps": ((completion_tokens - 1) / decode_s if decode_s and completion_tokens > 1 else None),
    }


def burst(base_url: str, model: str, prompts: list[str], max_tokens: int,
          label: str, ignore_eos: bool = True, start_delay: float = 0.0) -> dict[str, Any]:
    results: list[dict[str, Any] | None] = [None] * len(prompts)
    gate = threading.Barrier(len(prompts) + 1)
    def worker(i: int) -> None:
        gate.wait()
        if start_delay:
            time.sleep(start_delay)
        results[i] = request_one(base_url, model, prompts[i], max_tokens, f"{label}-{i}", ignore_eos)
    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(len(prompts))]
    for thread in threads:
        thread.start()
    gate.wait()
    t0 = time.perf_counter()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - t0
    rows = [r for r in results if r is not None]
    successful = [r for r in rows if r["ok"]]
    total = sum(r["completion_tokens"] for r in successful)
    ttfts = [r["ttft_ms"] for r in successful if r["ttft_ms"] is not None]
    itls = [x for r in successful for x in r["itl_ms_raw"]]
    per_stream = [r["decode_tps"] for r in successful if r["decode_tps"] is not None]
    return {
        "label": label, "N": len(prompts), "max_tokens": max_tokens,
        "wall_s": wall, "completion_tokens": total,
        "aggregate_completion_tps": (total / wall if wall else None),
        "failures": len(rows) - len(successful),
        "ttft_ms": stats(ttfts), "itl_ms": stats(itls),
        "per_stream_decode_tps": stats(per_stream), "requests": rows,
    }


def read_metrics(base_url: str) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            raw = response.read().decode("utf-8", "replace")
            try:
                return {"kind": "json", "value": json.loads(raw)}
            except json.JSONDecodeError:
                interesting = [line for line in raw.splitlines() if any(k in line.lower() for k in
                    ("prefix", "preempt", "recompute", "spec", "cache", "request")) and not line.startswith("#")]
                return {"kind": "prometheus", "interesting_lines": interesting, "raw_sha256": hashlib.sha256(raw.encode()).hexdigest()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def quality_suite(args: argparse.Namespace) -> dict[str, Any]:
    cases = []
    required = {
        "arithmetic": ("402",),
        "code_python": ("def ", "fibonacci"),
        "multilingual": ("Bonjour", "Guten Morgen"),
        "repetition": ("red blue green",),
        "code_rust": (".chars()",),
        # Fixed 96-token cap can end before the parity step; opening the valid
        # contradiction proof is the deterministic semantic gate at this budget.
        "reasoning": ("contradiction",),
    }
    for name, prompt in QUALITY_CASES:
        r = request_one(args.base_url, args.model, prompt, args.quality_tokens,
                        f"quality-{name}", ignore_eos=False)
        r["case"] = name
        r["prompt"] = prompt
        r["semantic_pass"] = r["ok"] and all(x.lower() in r["output_utf8"].lower() for x in required[name])
        if name == "repetition":
            lines = [x.strip() for x in r["output_utf8"].splitlines() if x.strip()]
            r["semantic_pass"] = r["semantic_pass"] and lines == ["red blue green"] * 8
        cases.append(r)
    # A deterministic long-context retrieval probe.
    marker = "The verification code is COBALT-7319."
    prompt = long_prompt(args.long_words, "quality-long") + "\n" + marker + "\nWhat is the verification code?"
    r = request_one(args.base_url, args.model, prompt, 32, "quality-long-context", ignore_eos=False)
    r["case"] = "long_context_retrieval"
    r["prompt_tail"] = prompt[-200:]
    r["semantic_pass"] = r["ok"] and "COBALT-7319" in r["output_utf8"]
    cases.append(r)
    return {"cases": cases, "all_http_ok": all(r["ok"] for r in cases),
            "all_semantic_pass": all(r["semantic_pass"] for r in cases)}


def performance_suite(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {"warmup": [], "concurrency": [], "long_3500": [], "context": [], "back_to_back": []}
    out["metrics_before"] = read_metrics(args.base_url)
    out["warmup"].append(request_one(args.base_url, args.model, DIVERSE_PROMPTS[0], 16, "warmup-0"))
    out["warmup"].append(request_one(args.base_url, args.model, DIVERSE_PROMPTS[1], 16, "warmup-1"))
    traffic_suffix = " Provide at least 300 words; continue until the output-token limit."
    for n in args.concurrency:
        for rep in range(args.burst_reps):
            prompts = [DIVERSE_PROMPTS[(i + rep) % len(DIVERSE_PROMPTS)] + traffic_suffix for i in range(n)]
            out["concurrency"].append(burst(args.base_url, args.model, prompts, args.max_tokens, f"n{n}-r{rep}"))
    for rep in range(args.long_reps):
        prompt = long_prompt(args.long_words, f"long-{args.start_id}-{rep}")
        out["long_3500"].append(burst(args.base_url, args.model, [prompt], args.max_tokens, f"long-r{rep}"))
    # Calibrated on this tokenizer: filler words map to ~1.36 tokens/word.
    for target_tokens, words in ((4000, 2940), (16000, 11760)):
        prompt = long_prompt(words, f"ctx-{target_tokens}-{args.start_id}")
        out["context"].append(burst(args.base_url, args.model, [prompt], args.max_tokens, f"ctx-{target_tokens}"))
    for rep in range(3):
        prompts = [DIVERSE_PROMPTS[(i + rep * 8) % len(DIVERSE_PROMPTS)] + traffic_suffix for i in range(8)]
        out["back_to_back"].append(burst(args.base_url, args.model, prompts, args.max_tokens, f"b2b-r{rep}"))
    prefix = long_prompt(3500, f"prefix-{args.start_id}")
    out["prefix"] = {
        "cold": burst(args.base_url, args.model, [prefix], args.max_tokens, "prefix-cold"),
        "warm": burst(args.base_url, args.model, [prefix], args.max_tokens, "prefix-warm"),
    }
    # Four active long decodes, then an arriving 3500-word prefill after 250 ms.
    dec_prompts = [p + " Give a long, detailed response." for p in DIVERSE_PROMPTS[:4]]
    mixed: dict[str, Any] = {}
    def decodes() -> None:
        mixed["active_decode"] = burst(args.base_url, args.model, dec_prompts, 256, "mixed-decode")
    thread = threading.Thread(target=decodes, daemon=True)
    thread.start()
    time.sleep(.25)
    mixed["arriving_prefill"] = burst(args.base_url, args.model,
        [long_prompt(args.long_words, f"mixed-{args.start_id}")], args.max_tokens, "mixed-prefill")
    thread.join()
    out["mixed"] = mixed
    out["metrics_after"] = read_metrics(args.base_url)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--start-id", type=int, required=True)
    parser.add_argument("--suite", choices=("quality", "performance", "all"), default="all")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--quality-tokens", type=int, default=96)
    parser.add_argument("--long-words", type=int, default=2570,
                        help="filler words; 2570 measures about 3500 Gemma tokens")
    parser.add_argument("--long-reps", type=int, default=2)
    parser.add_argument("--burst-reps", type=int, default=1)
    parser.add_argument("--startup-ms", type=float)
    parser.add_argument("--first-ready-rss-bytes", type=int)
    parser.add_argument("--physical-available-bytes", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.concurrency = [int(x) for x in args.concurrency.split(",")]
    result: dict[str, Any] = {
        "schema_version": 1, "engine": args.engine, "mode": args.mode,
        "start_id": args.start_id, "base_url": args.base_url, "model": args.model,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "startup_ms": args.startup_ms, "first_ready_rss_bytes": args.first_ready_rss_bytes,
        "physical_available_bytes_at_ready": args.physical_available_bytes,
        "measurement_note": "ITL uses non-empty SSE content-event timestamps. event_count_matches_completion_tokens marks whether those events are token-exact.",
    }
    if args.suite in ("quality", "all"):
        result["quality"] = quality_suite(args)
    if args.suite in ("performance", "all"):
        result["performance"] = performance_suite(args)
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(args.out),
                      "quality_http_ok": result.get("quality", {}).get("all_http_ok"),
                      "quality_semantic_pass": result.get("quality", {}).get("all_semantic_pass")}, indent=2))


if __name__ == "__main__":
    main()
