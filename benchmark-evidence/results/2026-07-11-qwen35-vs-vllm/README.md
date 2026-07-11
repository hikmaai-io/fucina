# Qwen3.5 FP8 — fucina vs vLLM head-to-head (2026-07-11)

Reproducible head-to-head behind `docs/qwen35-only-beat-vllm-plan.md`. Establishes the
scoreboard that P1 (fused prefill+decode on the qwen35 M4 engine) must flip.

## Hardware / build

- NVIDIA GB10 unified memory, CUDA 13, `sm_121a`.
- fucina built from `main` @ `41a1b99` (post tensor-refactor-merge `70ee5fd`); code is
  identical to `4e51b91` (docs-only commits on top). Clean worktree, not the conflicted
  checkout.
- GPU clock ~2400 MHz on **both** engines (fucina ran ~2431 uncapped; vLLM image caps via
  `GPU_CLOCK_MAX`, set to 2400 here so the comparison is clock-fair; decode is
  memory-bound so the cap is ~3% per the image author).

## Checkpoints (same for both engines)

- Primary: `Qwen/Qwen3.5-35B-A3B-FP8`, snapshot `0b2752837483aa34b3db6e83e151b150c0e00e49` (MoE hybrid).
- Secondary: `Qwen/Qwen3.5-9B-FP8` (dense hybrid).

## Servers

fucina (canonical PROTOCOL.md launch):
```sh
MODEL=/opt/spark/models/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/0b27528…
./fucina -m "$MODEL" --ctx 25280 --parallel 32 --max-concurrent 64 --gpu-mem-util 0.90 --port 18080
# fusion-off variant: FUCINA_NO_FUSED_PREFILL=1 (a NO-OP on Qwen3.5 — see below)
```

vLLM (only image with Qwen3.5 support; generic vLLM 0.15/0.16 does not register `Qwen3_5*`):
```sh
docker run --gpus all --network host --privileged \
  -e MODEL_PATH=<snapshot-dir> -e PORT=18081 -e MAX_MODEL_LEN=25280 \
  -e GPU_MEMORY_UTIL=0.88 -e ATTENTION_BACKEND=FLASHINFER -e GPU_CLOCK_MAX=2400 \
  -v <repo-root>:/models/repo:ro -v ~/.cache/flashinfer:/root/.cache/flashinfer \
  hellohal2064/vllm-qwen3.5-gb10:latest
# vLLM 0.16.0rc1, Qwen3_5MoeForConditionalGeneration, FLASHINFER_CUTLASS FP8 MoE, chunked prefill.
# NOTE: HF-snapshot dirs symlink config.json → ../../blobs; mount the WHOLE repo, point MODEL_PATH at the snapshot subdir.
```

## Sweep (canonical harness)

```sh
python3 scripts/bench_serving.py --base-url http://127.0.0.1:PORT --model <served-name> \
  --max-tokens 128 --long-tokens 3500 --ignore-eos --conc 1,2,4,8,16,32 --diverse --verify-sample 4 --out result.json
```
`agg_decode_tps = sum(completion_tokens-1)/whole_burst_wall_time` (includes admission+TTFT).
Diverse prompts mandatory (identical prompts hide MoE row-mixing). Bursts are synchronized —
they do NOT exercise fusion (all prefills arrive at t=0); see fusion note.

## Results (agg_decode_tps / median TTFT ms)

**Qwen3.5-35B-A3B-FP8 (MoE):**

| N | fucina | vLLM | fucina TTFT | vLLM TTFT |
|---|---|---|---|---|
| 1 (single) | 60.1 dec | 47.1 dec | 83 | 103 |
| 2 | 58.2 | 71.1 | 143 | 207 |
| 4 | 101.8 | 105.0 | 264 | 417 |
| 8 | 154.7 | 146.5 | 502 | 669 |
| 16 | 208.4 | 204.8 | 974 | 549 |
| 32 | 291.7 | 302.8 | 1923 | 664 |

single 3500-tok-prompt TTFT: fucina 4603 ms vs vLLM 6844 ms (fucina wins).

**Qwen3.5-9B-FP8 (dense), agg_decode_tps @ N=2/4/8/16/32:**
fucina 56.6 / 109.6 / 179.3 / 236.1 / 260.5 · vLLM 42.9 / 83.9 / 161.7 / 296.1 / 501.9.

(vLLM N=1 concurrency entries are compile warmup artifacts — use its `single_short`.)

## Verdicts

- **Won:** single-stream decode (+28% MoE), N≤8 everything, single long-prompt TTFT.
- **Lost (L1):** MoE TTFT under concurrency — 2.9× worse at N=32 (1923 vs 664 ms).
- **Lost (L2):** dense aggregate N≥16 — −93% at N=32.
- **Tie (L3):** MoE N=32 aggregate — −4%, within noise.

## Regression check (tensor-refactor merge 70ee5fd): CLEAN

35B-A3B matches PROTOCOL baselines (32.2/58.2/101.7/154.6/206.0 @ N=1/2/4/8/16) within 1–2%;
N=32 improved (291.7 vs old 212.6). verify-sample correctness passed.

## N=32 baseline provenance (291.7 now vs 212.6 in PROTOCOL.md) — RESOLVED

The `212.6` in `benchmark-evidence/PROTOCOL.md:50` is a **stale pre-optimization baseline**,
explicitly labelled "the old N=32 result". `benchmark-evidence/results/2026-07-10-fucina-a43ab6d.md`
documents the jump: `| 32 | 212.6 | 291.22 | +37.0% |` — commit **a43ab6d** (2026-07-10) raised
N=32 from 212.6 → 291.2. Later runs cluster higher still: 293.7 (`rerun-fa982db`), 317–320
(`tensor-refactor-phase1/5`), 333 (`unsloth-nvfp4`). This run's **291.7** matches the a43ab6d /
rerun-fa982db cohort. So 291.7 is **not** a regression — PROTOCOL.md's N=32 figure is simply
outdated (the frozen gate baselines it lists are N=1..16 only: 32.2/58.2/101.7/154.6/206.0, which
this run matches within 1–2%). CAVEAT: phase5 reported 320.7 vs this run's 291.7 — a ~9% spread
across nominally-identical protocol runs (prompt-mix / admission-order noise), which is why P3
freezes baselines from a fresh contemporaneous quiescent-box run rather than trusting any single
historical number.

## Arrival model + TTFT statistic (review requirement)

All concurrency cells above are **synchronized bursts** — the canonical `bench_serving.py` fires all
N requests simultaneously with 16 cycled SHORT diverse prompts (~10–15 tok each), plus a separate
`single_long` (3500-tok) probe. Consequently the N=32 TTFT of 1923 ms is dominated by *admission /
scheduling* latency, not prefill compute (32×~15 tok ≈ sub-ms of prefill). TTFT here is the
**median**; the archived JSONs store median only (the harness did not retain per-request arrays).
**p95 TTFT baselines are frozen separately under `../` P3 with an enhanced harness** (a synchronized
burst does NOT exercise fusion — see below — so this is an admission/scheduling metric).

## Fusion on/off is a STRUCTURAL NO-OP on Qwen3.5 (root cause of L1)

Byte-identical ON vs OFF on both models AND both burst and staggered-decode workloads
(35B staggered: 137.1 vs 137.0 agg, 3612 vs 3603 ms TTFT). `GEMMA4_IS_QWEN3_FAMILY =
QWEN3||QWEN3MOE` excludes `QWEN3_5`; Qwen3.5 loads the separate M4 batched engine and the
fused ABI returns −2. Stage-18 fusion was never wired into the qwen35 hybrid engine →
this is the P1 lever.
