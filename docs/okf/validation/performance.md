---
type: Benchmark Record
title: Qwen3.5 GB10 performance position
description: Checked-in contemporaneous throughput position against vLLM and the accepted determinism boundary.
tags: [performance, qwen, vllm, determinism, gb10]
status: draft
stale_after: 2026-08-18
generated: { by: openai-codex/gpt-5.6-sol, at: 2026-07-25T10:33:01+02:00 }
snapshot_commit: 39a96dbd4856f394821021efa10ef31848ad2581
sources:
  - id: mission
    resource: ../../sota-gb10-qwen3-mission-plan.md
    title: Concluded Qwen3.5 mission and official position
  - id: evidence
    resource: ../../../benchmark-evidence/PROTOCOL.md
    title: Benchmark protocol
  - id: d32
    resource: ../../qwen35-d32b.md
    title: Dense B32 boundary analysis
---

# Checked-in result

The 2026-07-18 contemporaneous sweep records fucina winning 11 of 12 aggregate-throughput cells against vLLM on one GB10: all six Qwen3.5-35B-A3B MoE cells and five of six Qwen3.5-9B dense cells.[^mission]

[^mission]: Concluded Qwen3.5 mission and official position

| Model | N cells won | Remaining loss |
|---|---:|---|
| Qwen3.5-35B-A3B FP8 MoE | 6/6 | Burst TTFT can still trail vLLM at high concurrency even when aggregate throughput wins. |
| Qwen3.5-9B FP8 dense | 5/6 | N=32 aggregate throughput: 438.8 vs 521.8 tok/s in the recorded sweep. |

# Accepted boundary

The dense N=32 gap was attributed to a register-bound, dependency-latency-limited bit-identical `dp4a` kernel class. Closing it with tensor-core reduction would change arithmetic order and violate the project's byte-identical output guarantee.[^d32] The documented decision is to retain determinism rather than claim all 12 cells.

[^d32]: Dense B32 boundary analysis

# 2026-07-25 MoE graph-router qualification

The Qwen3.6-35B-A3B-FP8 agentic incident improved from **12/100 in 1,299 s** to **80/100 in 347 s** after decode router execution was moved onto the captured stream; the dense Qwen3.6-27B-FP8 reference scored **81/100**. The router-fixed MoE engine produced coherent prose and correct tool calls rather than stale-route corruption. A later exact continuation-prefill hardening rerun scored **78/100 in 370.3 s**; this bundle retains 80 as the isolated router-fix before/after and 78 as the final branch measurement.

Decode performance did not regress. Three fresh Qwen3.6 MoE B=1 graph runs measured **55.7/59.4/59.0 tok/s**, median **59.0 tok/s** (16.96 ms/step), above the historical 46–53 tok/s served baseline. Median B=2/4/8 aggregate throughput was **95.4/150.3/219.8 tok/s**. The dense 27B smoke measured 11.90 tok/s at B=1 and produced coherent prose plus the correct Berlin weather call.

# Caveats

* The final branch's structured-output HTTP 501 behavior loses five scenarios; TC-60 sleeper injection remains a safety-critical failure.
* These are historical checked-in measurements, except for the explicitly dated 2026-07-25 agentic/decode qualification above.
* The claim is aggregate throughput, not a win in every TTFT metric.
* Any future public comparison must rerun both engines contemporaneously with the same protocol, checkpoint, prompts, sampling, concurrency, and quiescent hardware.
* DFlash speculative work reached correctness but was shelved as a GB10 performance lever because batching already captures most weight-amortization benefit.
