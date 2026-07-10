#!/usr/bin/env python3
"""Measure uncached turn-1 and hybrid-cache turn-2 TTFT over chat-completions SSE."""
import argparse
import json
import statistics
import time
import urllib.request

PARA = (
    "Large language model inference combines a compute-bound prompt phase with a "
    "memory-bandwidth-bound decoding phase. Continuous batching improves utilization by "
    "combining active sequences while preserving each request's independent state. Hybrid "
    "models also carry recurrent state alongside full-attention key and value caches. "
)


def prompt_for(tag: str, words: int) -> str:
    base = (PARA * (words // len(PARA.split()) + 2)).split()[:words]
    return f"Benchmark nonce {tag}. " + " ".join(base) + "\nSummarize this passage briefly."


def stream_chat(base_url, model, messages, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    content, reasoning = [], []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            obj = json.loads(payload)
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                c = delta.get("content") or ""
                r = delta.get("reasoning_content") or ""
                if first is None and (c or r or delta.get("tool_calls")):
                    first = time.perf_counter()
                content.append(c)
                reasoning.append(r)
    end = time.perf_counter()
    if first is None:
        raise RuntimeError("stream completed without a generated token")
    assistant = {"role": "assistant", "content": "".join(content)}
    if any(reasoning):
        assistant["reasoning_content"] = "".join(reasoning)
    return {
        "ttft_ms": (first - start) * 1000,
        "wall_ms": (end - start) * 1000,
        "assistant": assistant,
    }


def median(values):
    return statistics.median(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prompt-words", type=int, default=1500,
                    help="Approximate 2k Qwen tokens with ordinary English text")
    ap.add_argument("--turn-tokens", type=int, default=16)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.reps < 1:
        ap.error("--reps must be positive")

    stamp = time.time_ns()
    cold, warm = [], []
    for i in range(args.reps):
        user = {"role": "user", "content": prompt_for(f"{stamp}-cold-{i}", args.prompt_words)}
        result = stream_chat(args.base_url, args.model, [user], 1)
        cold.append({k: v for k, v in result.items() if k != "assistant"})

    for i in range(args.reps):
        user = {"role": "user", "content": prompt_for(f"{stamp}-warm-{i}", args.prompt_words)}
        first = stream_chat(args.base_url, args.model, [user], args.turn_tokens)
        followup = {"role": "user", "content": "Continue with one additional short sentence."}
        second = stream_chat(args.base_url, args.model, [user, first["assistant"], followup], 1)
        warm.append({
            "turn1_wall_ms": first["wall_ms"],
            "turn2_ttft_ms": second["ttft_ms"],
            "turn2_wall_ms": second["wall_ms"],
        })

    out = {
        "base_url": args.base_url,
        "model": args.model,
        "reps": args.reps,
        "prompt_words": args.prompt_words,
        "cold_turn1": cold,
        "cold_turn1_ttft_median_ms": median([x["ttft_ms"] for x in cold]),
        "warm_turn2": warm,
        "warm_turn2_ttft_median_ms": median([x["turn2_ttft_ms"] for x in warm]),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
