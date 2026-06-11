#!/usr/bin/env python3
"""Benchmark gem4d server: prefill scaling, decode throughput, agentic tool loop.

Phases (select with --phase, default all):
  prefill   Cold-prefill latency vs context size (decay curve). Each request
            starts with a unique nonce so the prefix cache cannot help.
  decode    Steady-state decode tok/s from SSE inter-token timing, at small
            and large context.
  toolloop  Simulates a coding agent's read loop: system prompt + tools, then
            N iterations of (model emits tool_call -> client appends tool
            result -> resend). Per iteration it reports wall time, prefix
            tokens reused vs newly prefilled (from /metrics deltas), and the
            effective prefill tok/s. This is the "tool_call read is slow even
            on small files" reproduction: if reused_tokens collapses to ~0 at
            some context size, prefix reuse broke (re-render divergence or
            context compaction) and every iteration pays a full re-prefill.

Stdlib only. Usage:
  python3 scripts/benchmark_gem4.py [--url http://localhost:8080] [--phase all]
      [--iters 12] [--result-bytes 400] [--pad-tokens 0] [--max-tokens 0]
  --max-tokens 0 omits max_tokens like pi does (exercises the server's
  ctx/2 completion-reservation + compaction path); pass e.g. 512 to compare.
"""

import argparse
import json
import sys
import time
import urllib.request

ALPHA = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu".split()


def http_json(url, payload=None, timeout=600):
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Server:
    def __init__(self, base):
        self.base = base.rstrip("/")
        info = http_json(self.base + "/v1/models")
        self.model = info["data"][0]["id"]
        m = self.metrics()
        self.ctx = m["context"]["capacity"]

    def metrics(self):
        return http_json(self.base + "/metrics")

    def chat(self, messages, tools=None, max_tokens=0, stream=False,
             temperature=0.0, thinking=False):
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "stream": stream,
                   "enable_thinking": thinking}
        if tools:
            payload["tools"] = tools
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        return payload

    def send(self, payload):
        """Non-streaming request. Returns (response, wall_seconds, metrics_delta)."""
        m0 = self.metrics()
        t0 = time.monotonic()
        resp = http_json(self.base + "/v1/chat/completions", payload)
        wall = time.monotonic() - t0
        m1 = self.metrics()
        d = {
            "reused": m1["prefix_cache"]["reused_tokens"] - m0["prefix_cache"]["reused_tokens"],
            "req_tok": m1["prefix_cache"]["request_tokens"] - m0["prefix_cache"]["request_tokens"],
            "prefilled": m1["totals"]["prefill_tokens"] - m0["totals"]["prefill_tokens"],
            "prefill_tps": m1["throughput_tok_s"]["prefill_last"],
            "decode_tps": m1["throughput_tok_s"]["decode_last"],
            "ctx_used": m1["context"]["used"],
        }
        return resp, wall, d

    def send_stream(self, payload):
        """Streaming request. Returns dict with ttft, decode tok/s, n tokens."""
        payload = dict(payload, stream=True)
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        first = None
        stamps = []
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                ch = chunk.get("choices") or [{}]
                delta = ch[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                    now = time.monotonic()
                    if first is None:
                        first = now
                    stamps.append(now)
        out = {"ttft_s": (first - t0) if first else None, "chunks": len(stamps)}
        if len(stamps) >= 2:
            span = stamps[-1] - stamps[0]
            out["decode_tps"] = (len(stamps) - 1) / span if span > 0 else None
        return out


def pad_text(n_words, seed):
    """Deterministic filler, roughly one token per word."""
    return " ".join(ALPHA[(seed + i) % len(ALPHA)] for i in range(n_words))


# ─── Phase 1: cold prefill scaling ───────────────────────────────────────

def phase_prefill(srv, args):
    print("\n== Phase 1: cold prefill latency vs context size ==")
    print(f"{'target':>8} {'prompt_tok':>10} {'prefilled':>10} {'reused':>7} "
          f"{'wall_s':>7} {'prefill_tok/s':>13}")
    points = [1024, 2048, 4096, 8192, 16384, 32768]
    points = [p for p in points if p < srv.ctx - 256] or [srv.ctx // 2]
    for i, target in enumerate(points):
        # Unique leading nonce defeats prefix reuse -> true cold prefill.
        nonce = f"run-{i}-{int(time.time())}"
        msgs = [{"role": "user",
                 "content": nonce + "\n" + pad_text(target, seed=i * 7) +
                 "\nReply with the single word: done"}]
        payload = srv.chat(msgs, max_tokens=8)
        resp, wall, d = srv.send(payload)
        ptok = resp["usage"]["prompt_tokens"]
        tps = d["prefilled"] / wall if wall > 0 else 0
        print(f"{target:>8} {ptok:>10} {d['prefilled']:>10} {d['reused']:>7} "
              f"{wall:>7.2f} {d['prefill_tps']:>13.1f}")


# ─── Phase 2: decode throughput ──────────────────────────────────────────

def phase_decode(srv, args):
    print("\n== Phase 2: decode throughput (SSE inter-chunk rate) ==")
    for label, ctx_words in [("small ctx (~100 tok)", 100),
                             ("large ctx", min(8192, srv.ctx // 2))]:
        msgs = [{"role": "user",
                 "content": pad_text(ctx_words, seed=99) +
                 "\nCount from 1 to 100, one number per line."}]
        payload = srv.chat(msgs, max_tokens=300)
        r = srv.send_stream(payload)
        tps = r.get("decode_tps")
        print(f"  {label:<22} ttft={r['ttft_s']:.2f}s  chunks={r['chunks']}"
              f"  decode={tps:.1f} chunk/s" if tps else
              f"  {label:<22} ttft={r['ttft_s']}  chunks={r['chunks']} (too few for rate)")


# ─── Phase 3: agentic tool-call loop ─────────────────────────────────────

READ_TOOL = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the repository and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative file path"},
            },
            "required": ["path"],
        },
    },
}]

SYSTEM = ("You are a code-exploration agent. To inspect a file you MUST call "
          "the read_file tool — never guess contents. After each result, read "
          "the next file the user asks for. Keep prose to one short sentence.")


def phase_toolloop(srv, args):
    print("\n== Phase 3: agentic tool-call loop (the 'slow read' reproduction) ==")
    print(f"   tool result size: {args.result_bytes} bytes; "
          f"max_tokens: {'omitted (pi-style)' if args.max_tokens == 0 else args.max_tokens}; "
          f"ctx capacity: {srv.ctx}")
    print(f"{'iter':>4} {'prompt_tok':>10} {'reused':>7} {'prefilled':>10} "
          f"{'reuse%':>7} {'wall_s':>7} {'tool_call':>10} {'ctx_used':>9}")

    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Read the file src/file_000.go"}]
    pad = pad_text(args.pad_tokens, seed=5) if args.pad_tokens else ""

    rows = []
    for it in range(args.iters):
        payload = srv.chat(msgs, tools=READ_TOOL, max_tokens=args.max_tokens,
                           thinking=args.thinking)
        try:
            resp, wall, d = srv.send(payload)
        except Exception as e:
            print(f"  iteration {it}: request failed: {e}")
            break
        choice = resp["choices"][0]
        m = choice["message"]
        ptok = resp["usage"]["prompt_tokens"]
        reuse_pct = 100.0 * d["reused"] / ptok if ptok else 0.0
        calls = m.get("tool_calls") or []
        called = calls[0]["function"]["name"] if calls else "-"
        rows.append((ptok, d["reused"], d["prefilled"], wall))
        print(f"{it:>4} {ptok:>10} {d['reused']:>7} {d['prefilled']:>10} "
              f"{reuse_pct:>6.1f}% {wall:>7.2f} {called:>10} {d['ctx_used']:>9}")

        # Append the assistant turn exactly as received (like an OpenAI client).
        asst = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("reasoning_content"):
            asst["reasoning_content"] = m["reasoning_content"]
        if calls:
            asst["tool_calls"] = calls
        msgs.append(asst)
        if calls:
            for c in calls:
                # Deterministic fake file body; pad to the requested size.
                body = (f"// {json.loads(c['function']['arguments']).get('path', '?')}\n"
                        f"package main\n" + pad)
                body = (body * (args.result_bytes // max(1, len(body)) + 1))[:args.result_bytes]
                msgs.append({"role": "tool", "tool_call_id": c["id"],
                             "name": c["function"]["name"], "content": body})
            msgs.append({"role": "user",
                         "content": f"Now read src/file_{it + 1:03d}.go"})
        else:
            msgs.append({"role": "user",
                         "content": f"You must use read_file. Read src/file_{it + 1:03d}.go"})

        if args.interloper:
            # Unrelated small request between agent turns — the eviction
            # scenario the KV snapshot cache exists for.
            srv.send(srv.chat([{"role": "user",
                                "content": f"interloper {it}: say OK"}],
                              max_tokens=8))

    if len(rows) > 2:
        # Skip the first two iterations: iter 0 has an empty cache and iter 1's
        # reuse fraction is dominated by the small denominator.
        steady = rows[2:]
        full_reprefills = sum(1 for p, r, _, _ in steady if r < p * 0.5)
        print(f"\n   summary: {len(steady)} steady-state iterations, "
              f"{full_reprefills} with <50% prefix reuse "
              f"({'prefix cache is NOT holding in the tool loop' if full_reprefills > 0 else 'prefix cache OK'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--phase", default="all",
                    choices=["all", "prefill", "decode", "toolloop"])
    ap.add_argument("--iters", type=int, default=12,
                    help="tool-loop iterations")
    ap.add_argument("--result-bytes", type=int, default=400,
                    help="size of each fake tool result")
    ap.add_argument("--pad-tokens", type=int, default=0,
                    help="extra filler words inside each tool result (to grow context fast)")
    ap.add_argument("--interloper", action="store_true",
                    help="fire an unrelated small request between tool-loop iterations")
    ap.add_argument("--thinking", action="store_true",
                    help="enable the reasoning channel in the tool loop (pi default)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="0 = omit max_tokens like pi (exercises compaction); else explicit cap")
    args = ap.parse_args()

    try:
        srv = Server(args.url)
    except Exception as e:
        sys.exit(f"cannot reach gem4d at {args.url}: {e}")
    print(f"server: {srv.model}  ctx={srv.ctx}")

    if args.phase in ("all", "prefill"):
        phase_prefill(srv, args)
    if args.phase in ("all", "decode"):
        phase_decode(srv, args)
    if args.phase in ("all", "toolloop"):
        phase_toolloop(srv, args)


if __name__ == "__main__":
    main()
