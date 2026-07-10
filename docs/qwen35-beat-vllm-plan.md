# Qwen3.5-MoE-35B — Finalized "beat vLLM on agentic coding" plan

## July 10 follow-up: cold-prefill and C32 plan completed

The later `perf/qwen35-cold-ttft-c32` work closed the remaining serving-capacity gap. Under the
reproducible diverse-prompt protocol in [`../benchmark-evidence/PROTOCOL.md`](../benchmark-evidence/PROTOCOL.md):

| Metric | Earlier fucina | Current fucina | Recorded vLLM |
|---|---:|---:|---:|
| Unique ~2k cold TTFT | 3.51 s | 1.49 s median (1.15 s best clean run) | 0.807 s |
| Warm state-cache TTFT | 76.5 ms | 71.8 ms median | 215–233 ms |
| Served tok/s N=16 | 206.0 | 208.0 | 159 |
| Served tok/s N=32 | 212.6 | **291.2** | 269.9 |

The C32 fix was capacity/accounting, not another decode kernel: exact Qwen allocations replace
cold/warm mmap free-memory guesses, Linux `MemAvailable` recognizes reclaimable page cache, slot
admission reserves a configurable 8K typical context while KV grows transactionally, and the
stale CLI cap of 16 was lifted to the real 32-row engine ceiling. N=1–16 throughput remains inside
the protection band; C32 improves 37%. CUDA-event telemetry also shows current 2k prefill at
1.38 s warm (27% expert GEMMs, 27% expert dequant, 4% routing, 2% shared expert, 40% other), so
the old 71.6%-expert prior below is historical. See the raw result and gate report under
`benchmark-evidence/results/2026-07-10-fucina-a43ab6d.*`.

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

## Ranked remaining plan — EXECUTED (Jul 4, this session)

| # | Item | Outcome |
|---|---|---|
| 1 | **json_schema / response_format** | ✅ **SHIPPED** (commits `e9e45d7`+`46612cd`). Ported the JSON grammar core + response_format; extended to real `json_schema` (OpenAI Structured-Outputs subset: typed props, required, additionalProperties:false, arrays, nested, enums, primitives) via a bitmask-cloneable schema automaton. Constrained requests forced onto the host-sampling path; route-guarded off the batch path (501). Full unit + server tests. |
| 4 | **GDN bf16 arena** | ✅ **SHIPPED** (commit `6fd8b5e`) — the one perf lever that survived. Decode **+5.8% @B=16**, +4.3% @B=8 (scales with B); halves the GDN arena memory. Fixed the OOB memset. Gated: MoE oracle 8/8 + graph-on==off + long-ctx **40/40 @1k & 4k** (delta-rule decay bounds bf16 rounding → no drift). |
| 3 | **LM head quant** | ❌ **DEAD (measured).** Single-stream already uses the exact two-pass Q8 head. Extending the two-pass to batched greedy stayed bit-identical (8/8) but **regressed 2.2× @B=16** (275→126 tok/s): the per-row candidate scan (`<<<B,1024>>>`) underutilizes the 48-SM GB10. The BF16 batched GEMV is already the right kernel. Reverted. |
| 2 | **Grouped-expert NVFP4 GEMM tuning** | ❌ **DEAD (measured).** Already ~81% of the GB10's ~273 GB/s LPDDR5X peak — bandwidth-saturated, not 68–78% of a higher ideal. The NVFP4 SF swizzle forces BM/BN to 128-multiples (no small-M decode tile compiles). Full buildable sweep: BK=256 gives ~2–3% on `down` only, ~0% on the dominant `gate\|up` → <1% aggregate + risks the shared ALU-bound prefill. |
| 5 | **Full ctx-cap lift** (stream the base>0 prefill-continuation attn off shared-scores + stream the fp32 oracle) → serve >25k | S–M | Low | reach, not speed | GO (measure-first) |
| 6 | Long-prompt **prefill** (GDN-chunk occupancy + TC-attn past base==0) | L | Med–High | cold TTFT 2.6→~1.5 s | **DEFER** (state cache already frees turn-2+) |
| 7 | Shared-expert quant FP8→Q4_K | M | Med–High | ~+3% decode | **DEFER** (tc-prefill reads Q4_K as FP8 → corruption footgun) |
| 8 | **spec-on-MoE** (per-slot GDN/conv chunk-scan SANDBOX; single-token-kernel mechanism is non-lossless) | L | High | 2–4× low-conc, +4% @conc-8 | **DEFER** (gate on measured acceptance first) |
| 9 | Mixer IMMA / tensor-core Q4_K GEMV | — | — | — | **DROP** (measured slower; Q4_K scales force per-weight dequant = the ALU cost) |

## Outcome (this session)

Roadmap executed. **Shipped: json_schema/response_format (#1) — the capability gap vs vLLM is
closed — and the GDN bf16 arena (#4, decode +5.8% @B=16).** The two perf tuning levers were
measured and **debunked**: LM-head quant (#3) regresses the batched path 2.2× (per-row candidate
scan underutilizes the 48-SM GB10), and grouped-expert GEMM tuning (#2) has no headroom — the
kernel is already ~81% of the GB10's ~273 GB/s LPDDR5X peak and the NVFP4 SF swizzle forbids a
decode-shaped tile. **Net confirmation of the branch's thesis:** GB10 decode is bandwidth-saturated,
so schedule/tile levers are dead; the only wins left are *fewer bytes moved* (#4 halved the GDN
state traffic → the last decode lever) and *capability* (#1). Remaining perf upside is <5% against
physics floors; the two decode-kernel-winnable regimes stay won. Next real gains would need a
different weight quant (symmetric int8-friendly) or the deferred spec-on-MoE — not tuning.
