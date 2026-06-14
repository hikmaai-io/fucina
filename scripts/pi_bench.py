#!/usr/bin/env python3
"""pi-real side-by-side benchmark: gem4d vs any OpenAI-compatible server (llama-server).

The point of this harness (vs parity_bench/benchmark_gem4) is a FAIR head-to-head on
the *actual* pi workload: an agentic tool-loop with thinking ON, at a large context
capacity, driving BOTH servers with a BYTE-IDENTICAL transcript every turn.

Divergence control: each turn we send the same input messages to every server and
time each independently, but we append only ONE canonical assistant output (from the
reference server, default the first one) to the shared history. So the prompt token
counts are identical across servers turn-by-turn — the thing that made the earlier
`pi` comparison apples-to-oranges (outputs drifted, contexts diverged) can't happen.

Per turn, per server we report:
  new      newly-prefilled tokens (gem4d: /metrics delta; llama: response timings.prompt_n)
  pf_t/s   prefill throughput (server self-reported)
  dec_t/s  decode throughput  (server self-reported; falls back to gen_tok/decode_wall)
  wall_s   end-to-end client wall for the request

Determinism: temperature 0. Thinking on by default (the pi case). Stdlib only.

Usage:
  python3 scripts/pi_bench.py \
      --server gem4d=http://127.0.0.1:8080 \
      --server llama=http://127.0.0.1:8000 \
      --turns 8 --result-bytes 6000 --max-tokens 256 --thinking
"""
import argparse, json, sys, time, urllib.request, urllib.error

def http_json(url, payload=None, timeout=900):
    data = json.dumps(payload).encode() if payload is not None else None
    hdr = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=hdr)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

SYSTEM = ("You are a code-exploration agent analysing a CUDA inference engine. "
          "To inspect any file you MUST call the read_file tool exactly once, then "
          "summarise what the file does in two sentences. Always read the file the "
          "user names before answering.")
READ_TOOL = [{"type": "function", "function": {
    "name": "read_file",
    "description": "Read a source file from the repository",
    "parameters": {"type": "object",
        "properties": {"path": {"type": "string", "description": "repo-relative path"}},
        "required": ["path"]}}}]

def fake_file(path, nbytes):
    head = f"// {path}\npackage engine\n"
    filler = "func step(x int) int { return x*7 + 3 } // rotating dequant scratch tile\n"
    body = head + filler * (nbytes // len(filler) + 1)
    return body[:nbytes]


class Server:
    """OpenAI-compatible endpoint. Reads gem4d /metrics if present, else uses the
    `timings` block llama-server attaches to chat-completion responses."""
    def __init__(self, name, base):
        self.name = name
        self.base = base.rstrip("/")
        self.has_metrics = False
        try:
            self._metrics()
            self.has_metrics = True
        except Exception:
            pass
        try:
            self.model = http_json(self.base + "/v1/models")["data"][0]["id"]
        except Exception:
            self.model = name

    def _metrics(self):
        return http_json(self.base + "/metrics")

    def send(self, messages, tools, thinking, max_tokens, temperature):
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "stream": False,
                   # gem4d understands these; llama ignores unknown keys and reasons
                   # per its own chat template (reasoning-budget) by default.
                   "thinking": thinking, "enable_thinking": thinking,
                   "timings_per_token": True}
        if tools:
            payload["tools"] = tools
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens

        m0 = self._metrics() if self.has_metrics else None
        t = time.monotonic()
        resp = http_json(self.base + "/v1/chat/completions", payload)
        wall = time.monotonic() - t
        m1 = self._metrics() if self.has_metrics else None

        msg = resp["choices"][0]["message"]
        usage = resp.get("usage", {}) or {}
        comp = usage.get("completion_tokens", 0)

        new = pf = dec = float("nan")
        if m0 and m1:  # gem4d path — authoritative split from /metrics
            new = m1["totals"]["prefill_tokens"] - m0["totals"]["prefill_tokens"]
            pf = m1["throughput_tok_s"]["prefill_last"]
            dec = m1["throughput_tok_s"]["decode_last"]
        tim = resp.get("timings")
        if tim:  # llama-server path
            if new != new:  # nan
                new = tim.get("prompt_n", float("nan"))
            if pf != pf and tim.get("prompt_ms"):
                pf = 1000.0 * tim.get("prompt_n", 0) / tim["prompt_ms"]
            if dec != dec and tim.get("predicted_ms"):
                dec = 1000.0 * tim.get("predicted_n", 0) / tim["predicted_ms"]
        if dec != dec and comp:  # last resort: client-side decode estimate
            dec = comp / wall
        return {"msg": msg, "prompt_tok": usage.get("prompt_tokens", 0),
                "comp_tok": comp, "new": new, "pf": pf, "dec": dec, "wall": wall}


def build_assistant(msg):
    a = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("reasoning_content"):
        a["reasoning_content"] = msg["reasoning_content"]
    if msg.get("tool_calls"):
        a["tool_calls"] = msg["tool_calls"]
    return a


def run(servers, turns, result_bytes, max_tokens, thinking, temperature):
    ref = servers[0]
    print(f"\n== pi-real tool-loop: {turns} turns, result={result_bytes}B, "
          f"thinking={thinking}, temp={temperature}, max_tok={max_tokens} ==")
    print(f"reference (canonical output) = {ref.name}; "
          f"servers = {', '.join(s.name for s in servers)}\n")

    hdr = f"{'turn':>4} {'prompt':>7}"
    for s in servers:
        hdr += f" | {s.name+' new':>10} {s.name+' pf':>9} {s.name+' dec':>8} {s.name+' wall':>9}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Read the file cuda/gemma4_kernels.cu"}]
    for it in range(turns):
        per = {}
        for s in servers:
            per[s.name] = s.send(msgs, READ_TOOL, thinking, max_tokens, temperature)
        r = per[ref.name]
        line = f"{it:>4} {r['prompt_tok']:>7}"
        for s in servers:
            d = per[s.name]
            line += (f" | {d['new']:>10.0f} {d['pf']:>9.1f} {d['dec']:>8.1f} {d['wall']:>9.2f}")
        print(line)
        rows.append({"turn": it, "prompt_tok": r["prompt_tok"],
                     "per": {n: {k: v for k, v in d.items() if k != "msg"}
                             for n, d in per.items()}})

        # Advance the SHARED transcript using the reference server's output only.
        rmsg = r["msg"]
        msgs.append(build_assistant(rmsg))
        calls = rmsg.get("tool_calls") or []
        if calls:
            for c in calls:
                try:
                    path = json.loads(c["function"]["arguments"]).get("path", "?")
                except Exception:
                    path = "?"
                msgs.append({"role": "tool", "tool_call_id": c["id"],
                             "name": c["function"]["name"],
                             "content": fake_file(path, result_bytes)})
            msgs.append({"role": "user", "content": f"Now read cuda/file_{it+1:03d}.cu"})
        else:
            msgs.append({"role": "user", "content": f"Now read cuda/file_{it+1:03d}.cu"})

    # Aggregate: weighted means over turns where both numbers are finite.
    print("\n== summary (means over turns; prefill weighted by new tokens) ==")
    for s in servers:
        pf_num = pf_den = dec_sum = dec_n = wall_sum = 0.0
        for row in rows:
            d = row["per"][s.name]
            if d["pf"] == d["pf"] and d["new"] == d["new"] and d["new"] > 0:
                pf_num += d["pf"] * d["new"]; pf_den += d["new"]
            if d["dec"] == d["dec"]:
                dec_sum += d["dec"]; dec_n += 1
            wall_sum += d["wall"]
        pf = pf_num / pf_den if pf_den else float("nan")
        dec = dec_sum / dec_n if dec_n else float("nan")
        print(f"  {s.name:>8}: prefill {pf:8.1f} tok/s | decode {dec:6.1f} tok/s "
              f"| total wall {wall_sum:6.2f}s")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", action="append", required=True,
                    help="name=URL (repeatable). First is the canonical/reference.")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--result-bytes", type=int, default=6000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    servers = []
    for spec in args.server:
        name, _, url = spec.partition("=")
        servers.append(Server(name, url))
    for s in servers:
        print(f"server {s.name}: {s.base} (model={s.model}, metrics={s.has_metrics})")

    rows = run(servers, args.turns, args.result_bytes, args.max_tokens,
               args.thinking, args.temp)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
