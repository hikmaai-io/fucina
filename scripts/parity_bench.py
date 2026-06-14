#!/usr/bin/env python3
"""Parity + prefill-throughput harness for gem4d kernel changes.

Drives the server deterministically (temp 0) so output is bit-stable, and
reports the prefill metrics that the dequant work targets. Two workloads:

  toolloop : multi-turn agentic read loop (system + read_file tool, fake file
             results). Turn 1 is a fresh prefill (batched path); every later
             turn is a SUFFIX prefill (flash path) — the agentic hot case.
  fresh    : single large unique prompt (> few k tokens) → fresh flash path.

It prints a per-turn table (new/reused tokens, wall, prefill tok/s) and a
single SHA256 FINGERPRINT over all assistant outputs. Same fingerprint
before/after a change == numerically identical == parity preserved.

Stdlib only.  Usage:
  python3 scripts/parity_bench.py [--url http://127.0.0.1:8080] [--turns 6]
"""
import argparse, hashlib, json, sys, time, urllib.request

def http_json(url, payload=None, timeout=600):
    req = urllib.request.Request(
        url, data=(json.dumps(payload).encode() if payload is not None else None),
        headers={"Content-Type": "application/json"} if payload is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

SYSTEM = ("You are a code-exploration agent analysing a CUDA inference engine. "
          "To inspect a file you MUST call the read_file tool. After reading, "
          "summarise what the file does in two sentences.")
READ_TOOL = [{"type": "function", "function": {
    "name": "read_file",
    "description": "Read a source file from the repository",
    "parameters": {"type": "object",
        "properties": {"path": {"type": "string", "description": "repo-relative path"}},
        "required": ["path"]}}}]

# Deterministic fake file body (~`nbytes` chars), stable across runs.
def fake_file(path, nbytes):
    head = f"// {path}\npackage engine\n"
    filler = ("func step(x int) int { return x*7 + 3 } // dequant rotating scratch\n")
    body = head + filler * (nbytes // len(filler) + 1)
    return body[:nbytes]

class Srv:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.ctx = http_json(self.base + "/metrics")["context"]["capacity"]
    def metrics(self): return http_json(self.base + "/metrics")
    def chat(self, messages, tools=None, thinking=False):
        p = {"model": "gem4d", "messages": messages, "temperature": 0.0,
             "stream": False, "thinking": thinking}
        if tools: p["tools"] = tools
        return p
    def send(self, payload):
        m0 = self.metrics(); t = time.time()
        resp = http_json(self.base + "/v1/chat/completions", payload)
        wall = time.time() - t; m1 = self.metrics()
        d = {"reused": m1["prefix_cache"]["reused_tokens"] - m0["prefix_cache"]["reused_tokens"],
             "prefilled": m1["totals"]["prefill_tokens"] - m0["totals"]["prefill_tokens"],
             "pf_tps": m1["throughput_tok_s"]["prefill_last"],
             "dec_tps": m1["throughput_tok_s"]["decode_last"],
             "ctx_used": m1["context"]["used"]}
        return resp, wall, d

def fingerprint_update(h, msg):
    # Hash the assistant's full deterministic output: reasoning + content + tool calls.
    h.update((msg.get("reasoning_content") or "").encode())
    h.update((msg.get("content") or "").encode())
    for c in (msg.get("tool_calls") or []):
        h.update(c["function"]["name"].encode())
        h.update(c["function"]["arguments"].encode())

def run_toolloop(srv, turns, thinking):
    print(f"\n== toolloop ({turns} turns; turn1=fresh/batched, rest=suffix/flash) ==")
    print(f"{'turn':>4} {'prompt_tok':>10} {'reused':>7} {'new':>6} {'reuse%':>7} "
          f"{'wall_s':>7} {'pf_tok/s':>9} {'dec_tok/s':>9} {'tool':>10}")
    h = hashlib.sha256()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Read the file cuda/gemma4_kernels.cu"}]
    for it in range(turns):
        resp, wall, d = srv.send(srv.chat(msgs, tools=READ_TOOL, thinking=thinking))
        m = resp["choices"][0]["message"]
        ptok = resp["usage"]["prompt_tokens"]
        reuse = 100.0 * d["reused"] / ptok if ptok else 0.0
        calls = m.get("tool_calls") or []
        called = calls[0]["function"]["name"] if calls else "-"
        print(f"{it:>4} {ptok:>10} {d['reused']:>7} {d['prefilled']:>6} {reuse:>6.1f}% "
              f"{wall:>7.2f} {d['pf_tps']:>9.1f} {d['dec_tps']:>9.1f} {called:>10}")
        fingerprint_update(h, m)
        asst = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("reasoning_content"): asst["reasoning_content"] = m["reasoning_content"]
        if calls: asst["tool_calls"] = calls
        msgs.append(asst)
        if calls:
            for c in calls:
                path = json.loads(c["function"]["arguments"]).get("path", "?")
                msgs.append({"role": "tool", "tool_call_id": c["id"],
                             "name": c["function"]["name"], "content": fake_file(path, 1500)})
            msgs.append({"role": "user", "content": f"Now read cuda/file_{it+1:03d}.cu"})
        else:
            msgs.append({"role": "user", "content": f"You must call read_file on cuda/file_{it+1:03d}.cu"})
    return h.hexdigest()

def run_fresh(srv, thinking):
    print(f"\n== fresh (single large unique prompt → fresh flash path) ==")
    lines = [f"Record {i}: value={i*31%97}, hour={i%24}, tag={chr(97+i%26)}" for i in range(1200)]
    prompt = ("Below are log records. Report the single maximum 'hour' value and how "
              "many records share it.\n" + "\n".join(lines) + "\nAnswer concisely.")
    msgs = [{"role": "user", "content": prompt}]
    resp, wall, d = srv.send(srv.chat(msgs, thinking=thinking))
    m = resp["choices"][0]["message"]
    print(f"  prompt_tok={resp['usage']['prompt_tokens']} new={d['prefilled']} "
          f"wall={wall:.2f}s pf={d['pf_tps']:.1f} tok/s dec={d['dec_tps']:.1f} tok/s")
    h = hashlib.sha256(); fingerprint_update(h, m); return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--thinking", default="off")
    a = ap.parse_args()
    th = a.thinking not in ("off", "false", "0", "")
    srv = Srv(a.url)
    print(f"server ctx capacity = {srv.ctx}; thinking={th}")
    fp1 = run_fresh(srv, th)
    fp2 = run_toolloop(srv, a.turns, th)
    combined = hashlib.sha256((fp1 + fp2).encode()).hexdigest()
    print(f"\nFRESH    fingerprint: {fp1}")
    print(f"TOOLLOOP fingerprint: {fp2}")
    print(f"COMBINED fingerprint: {combined}")

if __name__ == "__main__":
    main()
