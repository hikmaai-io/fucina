# Launch copy — fucina

Drafts for the v0.1.0 announcement. Keep the claims as-is: they match the README and the
`scripts/pi_bench.py` methodology. Don't inflate ("parity-to-ahead", not "faster than").

---

## Blog post

### fucina: forging Gemma 4 12B for the NVIDIA DGX Spark

We're open-sourcing **fucina** — a from-scratch Gemma 4 12B inference engine, written in Go and
CUDA C++, built for exactly one machine: the **NVIDIA DGX Spark GB10**.

Most inference engines aim for portability across dozens of GPUs. fucina makes the opposite bet. It
targets a single accelerator — Blackwell `sm_121a`, CUDA 13, 128 GB unified memory — and spends
that constraint on speed: FP8 Tensor-Core attention, an FP8 KV cache, position-independent
CUDA-graph decode, on-GPU sampling, and MTP speculative decoding. The name says it: *fucina* is
Italian for **forge** — the smithy where raw Gemma 4 weights are hammered into a fast engine for one
box.

**Does the bet pay off?** We benchmarked it head-to-head against `llama.cpp` on a fair harness
(identical transcript, temperature 0, the MTP draft head enabled on both engines —
`scripts/pi_bench.py`):

- **Decode: parity-to-ahead overall, and +15–20% at high context (≥5k tokens).** Throughput *rises*
  with context as speculation acceptance climbs and the FP8-KV attention scales — exactly the regime
  that dominates long agentic sessions.
- **Prefill: steady-state tied.** The residual gap is one-time cold turns, not steady throughput.
- **Tool calling** matches on the easy suite and is ahead on hard agentic scenarios.

Every change is validated **bit-exact** — greedy output byte-identical to the reference path,
`compute-sanitizer` clean.

**Where the speed comes from.** Single-token decode on the GB10 is bandwidth-bound on total weight
bytes, so the levers aren't wider loads — they're **eliminating launch bubbles** (the decode step
and the batched speculative-verify forward are captured as position-independent CUDA graphs and
replayed each token) and **raising speculation acceptance** (τ). Both are observable live in
`/metrics`.

**What it is — and isn't.** fucina is an experimental research project from
[hikmaai.io](https://hikmaai.io), released **as-is with no support**, Apache-2.0. It runs **only** on
the DGX Spark GB10 today. If you try it on an RTX 3090 or 4090, it won't build — that's by design,
not a bug. The roadmap: an `sm_120` (RTX 50-series) port, and the experimental DiffusionGemma 26B-A4B
engine that already shares the binary.

If you have a DGX Spark and want the fastest path to a Gemma 4 12B endpoint, give it a spin — and
tell us where it breaks.

→ **github.com/hikmaai-io/fucina**

---

## Show HN

**Title:** Show HN: Fucina – Gemma 4 12B inference for the DGX Spark GB10, at parity with llama.cpp

**Body:**
fucina is a from-scratch Gemma 4 12B inference engine (Go + CUDA C++) built for exactly one GPU: the
NVIDIA DGX Spark GB10 (Blackwell sm_121a). It trades portability for speed — FP8 Tensor-Core
attention, FP8 KV cache, position-independent CUDA-graph decode, MTP speculative decoding, an
OpenAI-compatible server.

On a fair side-by-side vs llama.cpp (same transcript, temp 0, MTP on both): decode is
parity-to-ahead overall and +15–20% at high context; prefill steady-state tied. All bit-exact.

Caveats up front: it's experimental, single-hardware (won't build on a 3090/4090 — by design), and
shipped with no support. Apache-2.0. Roadmap is an sm_120 (RTX 50-series) port.

Happy to answer questions about the CUDA-graph decode, the MTP verify, or the FP8 KV path.

---

## X / Twitter thread

1/ We're open-sourcing **fucina** — a Gemma 4 12B inference engine in Go + CUDA, built for ONE
machine: the NVIDIA DGX Spark GB10. On a fair head-to-head vs llama.cpp, decode is parity-to-ahead,
and +15–20% at high context. 🧵 github.com/hikmaai-io/fucina

2/ The bet: skip portability, spend it on speed. FP8 Tensor-Core attention, FP8 KV cache,
position-independent CUDA-graph decode, on-GPU sampling, MTP speculative decoding. "fucina" =
Italian for *forge* — where raw weights get hammered into a fast engine.

3/ Why it's fast: single-token decode on GB10 is bandwidth-bound on weight bytes. So the wins aren't
wider loads — they're killing launch bubbles (decode + batched verify replayed as CUDA graphs) and
raising speculation acceptance (τ). Both live in /metrics.

4/ Honest caveats: experimental, no support, Apache-2.0, and it runs ONLY on the DGX Spark GB10.
Try it on a 3090 and it won't build — by design. Next: an sm_120 (RTX 50-series) port.

5/ Built at @hikmaai. If you've got a DGX Spark and want a fast Gemma 4 endpoint, kick the tires and
tell us where it breaks. → github.com/hikmaai-io/fucina

---

## r/LocalLLaMA

**Title:** fucina — a Gemma 4 12B engine hand-tuned for the DGX Spark GB10 (parity-to-ahead of
llama.cpp), now open source

We built and open-sourced fucina: a from-scratch Gemma 4 12B inference engine (Go + CUDA C++) that
targets exactly one GPU — the NVIDIA DGX Spark GB10. The whole idea is to see how far you can push a
single Blackwell box by giving up portability: FP8 Tensor-Core attention + FP8 KV cache,
position-independent CUDA-graph decode, MTP speculative decoding (prompt-lookup + the official
draft head), prefix-reuse KV cache, on-GPU sampling.

On a fair side-by-side vs llama.cpp (identical transcript, temp 0, MTP draft head on both): decode
is parity-to-ahead overall and +15–20% at high context; prefill steady-state tied; tool-calling
matches/ahead. Everything validated bit-exact (greedy byte-identical, memcheck clean).

Big caveat, stated loudly: it's **experimental, no-support, and GB10-only** — it won't build on a
3090/4090/5090 today (the build is pinned to sm_121a and the hot paths use GB10 tensor-core
features). Apache-2.0. An sm_120 (RTX 50-series) port is on the roadmap.

Repo + full writeup: github.com/hikmaai-io/fucina — feedback welcome, especially from anyone else
running a DGX Spark.
