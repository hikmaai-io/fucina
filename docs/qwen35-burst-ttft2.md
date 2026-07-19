<!-- ABOUTME: Attribution, exactness, and served results for Qwen3.5 short-burst admission. -->
<!-- ABOUTME: Documents the profile-gated clean-prefix GDN keeper and its rollback contract. -->
# Qwen3.5 short-burst TTFT2: exact clean-prefix GDN

Status: **BURST_TTFT2_WIN** on `perf/qwen35-burst-ttft2`, based on `6da987a`.
Hardware: GB10, 48 SM, `sm_121a`, CUDA 13, approximately 273 GB/s. All GPU runs used
`/tmp/fucina_gpu.lock`; `nvidia-smi` was checked before each multi-start run.

## Scheduler answer first

The synchronized N=32 burst is admitted as **one admission containing all 32 rows**, not as
multiple admissions. With `FUCINA_QWEN35_PREFILL_TIMING=1`, the initial attribution probes logged:

| model | admissions | rows | prompt tokens in probe | arrival-to-engine | engine admission | first decode |
|---|---:|---:|---:|---:|---:|---:|
| MoE | 1 | 32 | 96 | 0.52–0.53 ms | 315.6 ms cold probe | 56.9 ms |
| dense | 1 | 32 | 96 | 0.37–0.38 ms | 319.3 ms cold probe | 76.4 ms |

The later smaller admissions in the server logs are the harness's post-burst sample verification,
not splits of the initial N=32 burst. Scheduler/coalescing is therefore not the median-gap lever.
The normal N=32 p95-minus-median spread was also far smaller than the competitor gap.

Raw scheduler logs:
`benchmark-evidence/results/2026-07-19-qwen35-burst-ttft2/{q35moe,q35dense}-attribution-server.log`.

## Phase 0 protocol and fresh baseline

Both checkpoints were run at N=1/2/4/8/16/32 with diverse synchronized prompts, 128 generated
tokens, and the canonical 3,500-token probe. Baseline and candidate each used three isolated server
starts. Raw JSON is under:

- `benchmark-evidence/results/2026-07-19-qwen35-burst-ttft2/baseline/`
- `benchmark-evidence/results/2026-07-19-qwen35-burst-ttft2/candidate/`
- machine-readable medians/deltas: `benchmark-evidence/results/2026-07-19-qwen35-burst-ttft2/summary.json`

### Baseline phase attribution, 15 tokens per sequence

CUDA-event spans are stream ordered. Percentages are shares of admission GPU elapsed time; H2D,
D2H, and embedding were each below 0.02 ms in steady runs. `classified` includes those small
categories and leaves launch/event gaps as unclassified. Steady attribution is 95.6–98.7%, above
the required 90%.

#### MoE 35B-A3B

| M | GPU ms | projection | conv/ring | full attention | cross-seq GDN | FFN | LM head | classified |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54.5 | 14.8 (27.2%) | 0.3 (0.6%) | 0.3 (0.6%) | 7.0 (12.9%) | 25.9 (47.5%) | 4.9 (9.0%) | 97.7% |
| 2 | 72.4 | 15.8 (21.8%) | 0.6 (0.9%) | 0.7 (0.9%) | 14.0 (19.5%) | 34.8 (47.8%) | 4.7 (6.5%) | 97.7% |
| 4 | 100.1 | 17.5 (17.7%) | 1.3 (1.3%) | 1.3 (1.3%) | 21.5 (21.5%) | 51.1 (50.7%) | 4.8 (4.8%) | 97.5% |
| 8 | 151.4 | 20.7 (13.7%) | 2.6 (1.7%) | 2.6 (1.7%) | 43.4 (28.7%) | 72.8 (48.1%) | 4.7 (3.1%) | 97.0% |
| 16 | 243.3 | 25.8 (10.6%) | 6.0 (2.5%) | 5.4 (2.2%) | 80.2 (33.0%) | 112.7 (46.3%) | 4.6 (1.9%) | 96.5% |
| 32 | **449.1** | 38.1 (8.5%) | 12.6 (2.8%) | 10.7 (2.4%) | **161.3 (35.9%)** | 200.3 (44.6%) | 7.8 (1.7%) | 96.1% |

#### Dense 9B

| M | GPU ms | projection | conv/ring | full attention | cross-seq GDN | FFN | LM head | classified |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 85.0 | 22.5 (26.5%) | 0.3 (0.3%) | 0.3 (0.3%) | 5.6 (6.6%) | 45.6 (53.6%) | 9.6 (11.3%) | 98.7% |
| 2 | 95.8 | 23.7 (24.8%) | 0.5 (0.5%) | 0.5 (0.5%) | 11.2 (11.7%) | 49.0 (51.2%) | 9.2 (9.6%) | 98.5% |
| 4 | 108.9 | 26.2 (24.1%) | 1.1 (1.0%) | 1.1 (1.0%) | 17.2 (15.8%) | 51.8 (47.6%) | 9.4 (8.6%) | 98.1% |
| 8 | 138.4 | 30.7 (22.2%) | 2.2 (1.6%) | 2.3 (1.7%) | 35.0 (25.3%) | 55.2 (40.0%) | 9.2 (6.7%) | 97.4% |
| 16 | 197.1 | 40.0 (20.3%) | 4.7 (2.4%) | 4.3 (2.2%) | 65.2 (33.1%) | 66.5 (33.8%) | 9.1 (4.6%) | 96.5% |
| 32 | **331.4** | 55.8 (16.8%) | 10.1 (3.1%) | 9.2 (2.8%) | **130.2 (39.3%)** | 96.1 (28.9%) | 15.6 (4.7%) | 95.6% |

Decision gate: conv/ring and full attention were below 5% at M=32 for both models. Fresh-slot GDN
was 35.9%/39.3%, so only the clean-prefix GDN candidate was implemented. No grouped-attention,
conv fusion, mixer, MoE residency, or scheduler arithmetic change was made.

Host and GPU elapsed are essentially equal in the dedicated admission bench after warm-up. In the
served N=32 baseline, MoE had approximately 449 ms admission GPU versus 640 ms median TTFT; dense
had 331 ms versus 470 ms. The remainder is scheduler/HTTP/first-decode wall time, not hidden H2D or
D2H. The timing mode itself creates events and synchronizes; it is default-off and is not used for
throughput claims.

## Implemented exact specialization

`FUCINA_QWEN35_CLEAN_GDN` controls a metadata-driven clean-prefix kernel. It is default-on after the
measured win; set `FUCINA_QWEN35_CLEAN_GDN=0` for immediate rollback.

Dispatch requires both explicit per-slot facts:

1. `q35_state_is_clean`: the GDN matrix was stream-ordered zeroed by fresh-slot reset;
2. `q35_conv_is_empty`: the causal conv ring was stream-ordered zeroed.

Both bits clear after prefill and on restore. Continuations, snapshots, restored sessions, warm
prefixes, and later calls cannot enter the specialization.

For a fresh first chunk, deterministic classes 16/32/48/64 remove only padded rows. The kernel
omits `kcd @ S` and `qgs @ S` only while `S` is proven +0. It retains the incumbent subtraction,
causal scalar order, triangular solve, intra-chunk GEMM, and final state add. A block-wide finiteness
check falls back to the incumbent WMMA interactions for NaN/Inf, preserving IEEE poisoning;
signed-zero subtraction remains explicit. N=65 executes a full 64-row first chunk and an incumbent
second chunk.

At M=32, 15 tokens/sequence, admission GPU time changed:

| model | baseline | clean | delta | GDN baseline | GDN clean |
|---|---:|---:|---:|---:|---:|
| MoE | 449.1 ms | 326.7 ms | **-27.3%** | 161.3 ms | 34.2 ms |
| dense | 331.4 ms | 232.2 ms | **-29.9%** | 130.2 ms | 28.5 ms |

## Served results: three-start median

| model / N | baseline TTFT med/p95 ms | clean TTFT med/p95 ms | med / p95 improvement | aggregate delta |
|---|---:|---:|---:|---:|
| MoE 1 | 57.9 / 57.9 | 58.7 / 58.7 | -1.2% / -1.2% | +0.4% |
| MoE 2 | 96.2 / 106.0 | 84.1 / 93.8 | +12.5% / +11.5% | -0.7% |
| MoE 4 | 143.0 / 143.6 | 123.8 / 124.2 | +13.4% / +13.5% | +1.5% |
| MoE 8 | 235.3 / 236.6 | 191.9 / 193.5 | +18.4% / +18.2% | -0.1% |
| MoE 16 | 376.5 / 395.7 | 298.3 / 300.0 | +20.8% / +24.2% | -3.1% |
| MoE 32 | **639.7 / 696.3** | **531.7 / 555.6** | **+16.9% / +20.2%** | **+4.3%** |
| dense 1 | 89.7 / 89.7 | 90.7 / 90.7 | -1.2% / -1.2% | +0.5% |
| dense 2 | 109.2 / 109.4 | 100.0 / 100.3 | +8.4% / +8.3% | -1.2% |
| dense 4 | 155.4 / 172.9 | 139.0 / 156.5 | +10.6% / +9.5% | +1.0% |
| dense 8 | 212.9 / 213.7 | 179.7 / 180.4 | +15.6% / +15.6% | +1.1% |
| dense 16 | 290.7 / 295.1 | 237.1 / 241.0 | +18.4% / +18.3% | +1.1% |
| dense 32 | **469.5 / 476.9** | **388.0 / 395.0** | **+17.4% / +17.2%** | **+1.1%** |

Long-prompt TTFT stayed protected: MoE 4356.7 -> 4415.5 ms (+1.35%); dense 3679.4 ->
3703.4 ms (+0.65%). Valid single-short TTFT added no wait: MoE -0.88 ms and dense -0.38 ms.
The MoE N=16 full-sweep aggregate median was -3.1% versus this branch's three-start baseline;
three separately warmed isolated starts measured 305.6/321.2/317.3 tok/s (median 321.2), and the
frozen protection gate passed.

## Exactness and protection gates

- `qwen35-clean-gdn-test`, both checkpoints: PASS. Incumbent and clean dispatch compare first
  token, full logits for M>1, full hybrid state bytes (GDN matrix + conv ring + attention KV), and
  32 continuation tokens/state for every length 1..65 and mixed M=1/2/4/8/16/32.
- `qwen35-clean-gdn-meta-test`: PASS for class boundaries, gaps/overlaps, zero/negative/overflow
  lengths, NaN/Inf omission guards, and signed zero.
- `qwen35-multiseq-prefill-test`: PASS with unchanged bounds, MoE <=0.0946 and dense <=0.0029.
- dense graph on/off, row independence, sampling, self-chain: PASS.
- state save/restore, GDN rollback j=0..6, and chunk parity: PASS.
- dense D32B stream hash: unchanged `c6ab45eab1f2751c`.
- protection gate: PASS for both models; raw reports are `*-protection-gate.log`.
- `go test ./...` and `make lib libdg fucina`: PASS.

The known MoE grouped-GEMM B>1 self-consistency gate still fails while oracle parity is 8/8.
`FUCINA_QWEN35_CLEAN_GDN=0` reproduces the identical failure counts and tokens in
`moe-engine-rollback-control.log`; the new same-engine incumbent-vs-clean exact gate passes MoE,
so this pre-existing decode issue is not used to permit any clean-GDN drift.

## Telemetry controls

- `FUCINA_QWEN35_PREFILL_TIMING=1`: scheduler admission/wait/first-decode logs plus CUDA-event phase
  timing. Default off. The legacy plural spelling remains accepted for the older coarse timer.
- `FUCINA_QWEN35_CLEAN_GDN=0`: force the incumbent GDN scan for rollback.
