#!/usr/bin/env python3
"""Apples-to-apples LLM serving bench over OpenAI /v1/completions (streaming).
Measures TTFT and decode tok/s, single-stream short+long, and a concurrency sweep.
Works against both fucina and vLLM (both expose /v1/completions)."""
import sys, json, time, argparse, threading, statistics, urllib.request

def gen_long_prompt(approx_tokens):
    # ~0.75 tok/word for english; build a coherent-ish long doc.
    para = ("In the study of large language model inference, throughput and latency "
            "are governed by memory bandwidth, kernel fusion, and the scheduling of "
            "concurrent sequences. The decode phase is bandwidth bound while the prefill "
            "phase is compute bound. Continuous batching interleaves prefill and decode "
            "to keep the accelerator busy. ")
    words_needed = int(approx_tokens / 0.7)
    s = (para * (words_needed // len(para.split()) + 2))
    words = s.split()[:words_needed]
    return " ".join(words) + "\nSummarize the key ideas above in one paragraph: "

def one_request(base_url, model, prompt, max_tokens, ignore_eos):
    body = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        body["ignore_eos"] = True
        body["min_tokens"] = max_tokens
    data = json.dumps(body).encode()
    req = urllib.request.Request(base_url + "/v1/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = None; t_last = t0
    n_chunks = 0; usage_tok = None; text = ""
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            ch = obj.get("choices") or []
            if ch:
                piece = ch[0].get("text", "")
                if piece:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    n_chunks += 1
                    text += piece
            u = obj.get("usage")
            if u and u.get("completion_tokens"):
                usage_tok = u["completion_tokens"]
    t_end = time.perf_counter()
    n = usage_tok if usage_tok else n_chunks
    ttft = (t_first - t0) if t_first else float("nan")
    decode_window = (t_last - t_first) if (t_first and t_last > t_first) else float("nan")
    dtoks = (n - 1) if n and n > 1 else 0
    tps = (dtoks / decode_window) if decode_window and decode_window > 0 else float("nan")
    return {"ttft": ttft, "decode_tps": tps, "ntok": n, "wall": t_end - t0, "text": text[:200]}

def single(base_url, model, prompt, max_tokens, ignore_eos, reps=3):
    res = [one_request(base_url, model, prompt, max_tokens, ignore_eos) for _ in range(reps)]
    ttfts = [r["ttft"] for r in res]
    tpss = [r["decode_tps"] for r in res]
    return {"ttft_ms": statistics.median(ttfts) * 1000,
            "decode_tps": statistics.median(tpss),
            "ntok": res[-1]["ntok"], "sample": res[-1]["text"]}

def concurrency(base_url, model, prompt, max_tokens, ignore_eos, N):
    results = [None] * N
    def work(i):
        results[i] = one_request(base_url, model, prompt, max_tokens, ignore_eos)
    t0 = time.perf_counter()
    threads = [threading.Thread(target=work, args=(i,)) for i in range(N)]
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0
    total_decode = sum((r["ntok"] - 1) for r in results if r and r["ntok"])
    agg_tps = total_decode / wall
    ttfts = [r["ttft"] * 1000 for r in results if r]
    per_tps = [r["decode_tps"] for r in results if r]
    return {"N": N, "agg_decode_tps": agg_tps, "median_ttft_ms": statistics.median(ttfts),
            "median_per_stream_tps": statistics.median(per_tps), "wall_s": wall}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="server")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--long-tokens", type=int, default=3500)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--conc", default="1,2,4,8")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    short_prompt = "Explain how continuous batching improves LLM serving throughput. Be detailed.\n\nAnswer:"
    long_prompt = gen_long_prompt(args.long_tokens)

    out = {"label": args.label, "base_url": args.base_url, "model": args.model}
    # warm-up (triggers graph capture / compile)
    one_request(args.base_url, args.model, short_prompt, 16, args.ignore_eos)

    print(f"[{args.label}] single short ctx ...", flush=True)
    out["single_short"] = single(args.base_url, args.model, short_prompt, args.max_tokens, args.ignore_eos)
    print(f"[{args.label}] single long ctx (~{args.long_tokens} tok) ...", flush=True)
    out["single_long"] = single(args.base_url, args.model, long_prompt, args.max_tokens, args.ignore_eos, reps=2)
    out["concurrency"] = []
    for N in [int(x) for x in args.conc.split(",")]:
        print(f"[{args.label}] concurrency N={N} ...", flush=True)
        out["concurrency"].append(concurrency(args.base_url, args.model, short_prompt, args.max_tokens, args.ignore_eos, N))

    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
