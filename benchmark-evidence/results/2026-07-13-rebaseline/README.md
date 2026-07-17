# Qwen3.5 re-baseline sweep — 2026-07-13 (main @ 60b109a)

Fresh fucina sweep on merged main (P1+P2+prune+S2+DFlash-substrate+MoE-TTFT all
landed) per canonical PROTOCOL.md: bench_serving.py, diverse prompts, conc
1-32, max-tokens 128, long 3500. Quiescent box (no vLLM resident, 108 GB free).

**vLLM comparison numbers are CARRIED FORWARD from 2026-07-11** (the
hellohal2064/vllm-qwen3.5-gb10 image was not re-run contemporaneously — no
vLLM container present on the box at sweep time). Cells are still clock-fair
(same box, same checkpoints) but the vLLM column is 2 days old.

## Qwen3.5-35B-A3B-FP8 (MoE) — agg tok/s | median/p95 TTFT ms

| N | fucina 07-13 | fucina 07-11 (post-P1) | vLLM 07-11 | verdict |
|---|---|---|---|---|
| 1 | 61.3 \| 62/62 | 59.8 \| 57/57 | 13.4* | WIN |
| 2 | 106.8 \| 91/100 | 99.5 \| 106/117 | 71.1 \| 207 | WIN (+50%) |
| 4 | 161.5 \| 148/148 | 166.0 \| 163/163 | 105.0 \| 417 | WIN (+54%) |
| 8 | 239.3 \| 225/226 | 224.0 \| 291/293 | 146.5 \| 669 | WIN (+63%) |
| 16 | 323.0 \| 367/370 | 287.7 \| 466/532 | 204.8 \| 549 | WIN (+58%) |
| 32 | **450.2 \| 641/647** | 405.1 \| 866/874 | 302.8 \| 664 | **WIN both (agg +49%, TTFT −3.5%)** |

Single-stream 64.0 tok/s (07-11: 62.0); 3500-tok TTFT 4257 ms (07-11: 4306).
MoE-TTFT F1/F2/F3 fully confirmed in serving: N=32 TTFT 866→641 ms, now BELOW
vLLM 664 median AND p95. Every MoE cell won.

## Qwen3.5-9B-FP8 (dense) — agg tok/s | median/p95 TTFT ms

| N | fucina 07-13 | fucina 07-11 (post-P1) | vLLM 07-11 | verdict |
|---|---|---|---|---|
| 1 | 35.1 \| 83/83 | 33.2 \| 90/90 | 10.1* | WIN |
| 2 | 63.2 \| 103/103 | 57.0 \| 122/122 | 42.9 \| 193 | WIN (+47%) |
| 4 | 124.0 \| 147/164 | 112.8 \| 192/210 | 83.9 \| 259 | WIN (+48%) |
| 8 | 211.6 \| 207/208 | 194.3 \| 293/295 | 161.7 \| 286 | WIN (+31%) |
| 16 | 321.7 \| 290/293 | 268.1 \| 460/463 | 296.1 \| 296 | **WIN (+8.6% agg, TTFT ~par)** |
| 32 | 392.4 \| 497/505 | 303.2 \| 866/872 | **501.9** \| 484 | **LOSS agg (−22%), TTFT ~par** |

P2 F1/F2 confirmed in serving: N=16 flipped to a WIN (268→322 vs 296); N=32
improved +29% (303→392) but vLLM 501.9 still leads — the remaining dense gap.

\* vLLM N=1 includes cold-start (their harness artifact), not comparable.

## Scoreboard summary

- **MoE 35B-A3B: fucina wins ALL 6 concurrency cells**, both aggregate and TTFT.
- **Dense 9B: fucina wins N=1..16** (N=16 newly flipped); **N=32 aggregate is the
  single losing cell** (392 vs 502, was 303 vs 502 — gap halved by P2).
- Caveat: vLLM column not contemporaneous (07-11). A fresh vLLM run is needed
  before freezing "wins everything except dense-32" as the official claim.

Raw: `moe-rebaseline.json`, `dense-rebaseline.json`.
