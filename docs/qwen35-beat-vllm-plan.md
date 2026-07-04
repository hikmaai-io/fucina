# Qwen3.5-MoE-35B — Finalized "beat vLLM on agentic coding" plan

_Finalized Jul 4 2026 (branch `qwen35-hybrid`), after the decode-kernel push. Backed by
measurement + adversarial design review; see memory `qwen35-decode-opt-findings.md`._

## Where we stand (measured)

| Regime | fucina | vLLM | Status |
|---|---|---|---|
| decode single-stream (short) | 56–58 tok/s | 53.4 | ✅ WON |
| turn-2+ TTFT @2k (state cache) | 0.107 s | (weak APC) | ✅ WON |
| **long-ctx single-stream decode** | **49.5 @3.5k / 45.5 @6k** | — | ✅ FIXED (flash-decoding) |
| aggregate @conc-16 | ~203 served / 253–274 engine | 449* | ⚠️ (*bench artifact) |
| cold turn-1 TTFT @2k | 2.6 s | 1.19 s | ⚠️ 2.2× |
| structured output (json_schema) | **absent on this branch** | guided decode | ❌ capability gap |

\*449 = identical-prompt convergent-routing bench artifact; diverse-traffic floor ≈ 51 ms/step
makes it physically unreachable (see `moe35b-vllm-headtohead.md`). Real engine target ≈ 290.

**The two things a decode-kernel push can win here are WON** (single-stream decode + turn-2 TTFT).
Node-traced B=16 step ≈ 58 ms: grouped experts ~13.6–20 ms (biggest), mixer Q4_K 13.2 ms (**floored**),
GDN 9.2 ms, LM head 6.3 ms, act-quant 5.9 ms, shared-expert 4.3 ms. The decode kernels are at/near
their GB10 floors; remaining headroom is **capability**, **grouped-expert efficiency**, and **cold prefill**.

## Shipped this session

dialect + Hermes/XML tool calling · per-conversation state cache (turn-2 TTFT 1.47→0.11 s) ·
tokenizer newline fix · scheduler phase telemetry · fp16 FULL-layer KV (memory-only, −8 GB) ·
**flash-decoding attention (long-ctx decode 33→49 tok/s)** · oracle opt-in for long-ctx gates.
Validated: MoE oracle 8/8, 9B batch-test, llama.cpp long-ctx parity 40/40 @1k+4k.

## Ranked remaining plan

| # | Item | Effort | Risk | Gain | Call |
|---|---|---|---|---|---|
| 1 | **json_schema / response_format** (cross-branch merge of `978d91d`+`6ef429b` from `feat/e4b-gguf`, then extend to json_schema + sampler bind) | M | Low–Med | capability (not tok/s) | **GO — #1** |
| 2 | **Grouped-expert NVFP4 GEMM tuning** (`dg_fp4_moe.cu`; tile/cluster/stage/split-K past 68–78% of BW-ideal) | M–L | Med | ~10–15% aggregate (only lever that scales with concurrency) | GO |
| 3 | **LM head quant** BF16→Q6_K/Q8 (keep BF16 for `want_argmax==0`) | S–M | Low–Med | ~5–7% single decode | GO (cheap) |
| 4 | **GDN bf16 arena** (halve state traffic; FIX the OOB memset byte-length; fp32 shared math) | S | Med | ~3–5% step | GO — needs the ≥2k drift gate (now runnable via the oracle opt-in) |
| 5 | **Full ctx-cap lift** (stream the base>0 prefill-continuation attn off shared-scores + stream the fp32 oracle) → serve >25k | S–M | Low | reach, not speed | GO (measure-first) |
| 6 | Long-prompt **prefill** (GDN-chunk occupancy + TC-attn past base==0) | L | Med–High | cold TTFT 2.6→~1.5 s | **DEFER** (state cache already frees turn-2+) |
| 7 | Shared-expert quant FP8→Q4_K | M | Med–High | ~+3% decode | **DEFER** (tc-prefill reads Q4_K as FP8 → corruption footgun) |
| 8 | **spec-on-MoE** (per-slot GDN/conv chunk-scan SANDBOX; single-token-kernel mechanism is non-lossless) | L | High | 2–4× low-conc, +4% @conc-8 | **DEFER** (gate on measured acceptance first) |
| 9 | Mixer IMMA / tensor-core Q4_K GEMV | — | — | — | **DROP** (measured slower; Q4_K scales force per-weight dequant = the ALU cost) |

## Highest-value next action

**Merge json_schema/response_format across from `feat/e4b-gguf` (#1).** The two things a kernel push
can win here are already won; every remaining perf lever is sub-15% against floors three independent
debunks confirmed (mixer, WMMA, expert-smem), and the flagship 449 aggregate target is a bench
artifact. Meanwhile this branch has **zero** structured-output capability (verified: the grammar core
is unreachable from `qwen35-hybrid`) while vLLM ships guided decoding — a missing capability the
agentic workload depends on has unbounded relative cost. Sequence: ship json_schema (#1), bank the two
cheap decode wins (LM head quant #3 + GDN bf16 #4, ~8–12% at S effort, low risk), then grouped-expert
tuning (#2) as the one perf lever with real concurrency headroom. Defer prefill/shared-expert/spec —
the state cache and the physics floor have blunted their upside.
