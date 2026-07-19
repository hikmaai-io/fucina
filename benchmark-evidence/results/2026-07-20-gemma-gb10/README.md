# Fresh Gemma GB10 evidence — 2026-07-20

**Status: BLOCKED for a competitive winner.** This is the first fresh, raw served matrix in this
tree, but it does **not** establish a dense fucina-vs-vLLM winner. The intended same-artifact
native-NVFP4 cell failed on fucina before readiness; the runnable dense engines use different
weight artifacts/precisions; vLLM has no local semantically matching HF MTP assistant; and current
paged/batch correctness gates fail. Raw throughput is retained below as evidence, not promoted to
a parity claim.

Run directory timestamps begin 2026-07-19 Europe/Rome; the requested publication directory is
2026-07-20. All GPU phases were serialized. No runtime source file was changed.

## Decision matrix

| Cell | Fucina | vLLM | Class | Winner |
|---|---|---|---|---|
| Dense 12B native NVFP4, exact local artifact | load fails at layer 5 projection 1 shape check | candidate loader not run after fucina failed | **MISSING / stopped pre-speed** | **unknown** |
| Dense 12B served plain | Google QAT Q4_0 GGUF; Q4_0 decode + derived NVFP4 prefill | Google BF16 safetensors; BF16 weights + FP8 KV | **FORMAT/SYSTEM**, not kernel parity | **none reported** |
| Dense 12B fucina MTP vs fucina plain | same Q4_0 target + pinned Q8_0 assistant | n/a | same-target within-engine productization probe | final three-start gate reported in `summary.json` / `protection-report.json` |
| Dense 12B vLLM MTP | local assistant is GGUF Q8_0 only | no matching HF assistant | **MISSING** | **unknown** |
| E4B HTTP continuous batch | source capability exists, but fresh batch parity fails 2/8 tokens in one sequence | runnable source/image capability | **stopped pre-speed** | **unknown** |
| E4B single-sequence MTP | unchanged test manually linked: 160 tokens byte-identical, 55.9→167.6 tok/s | no matching local HF assistant | capability-only | n/a |

Exact paths, revisions, SHA-256 values, image digest, source/image commit mismatch, launch commands,
and acquisition requests are in [`manifest.json`](manifest.json).

## Apples-to-apples cross-engine table

| Model/artifact | Fucina | vLLM | Measured parity result | Winner |
|---|---|---|---|---|
| dense RedHat NVFP4, SHA-256 `2a476980…afaf27` | **load FAIL** before ready | not run after peer failed | `MISSING` | unknown |
| dense Google Q4_0 GGUF | supported | audited image has no GGUF model loader | `MISSING` | unknown |
| dense Google BF16 safetensors | not a production-equivalent fucina Q4/NVFP4 cell | supported | `MISSING` | unknown |
| E4B BF16 source artifact | fucina transforms PLE/decode precision and batch parity fails | BF16-capable | stopped pre-speed; not identical runtime arithmetic | unknown |

There is intentionally no numeric row here. Every runnable cross-engine number belongs in the
FORMAT/SYSTEM section below.

## Format/system observations — three-start medians

The following table is deliberately **not** an apples-to-apples leaderboard. It demonstrates the
shape of the two runnable systems and identifies bottlenecks. Both used the same OpenAI chat API,
local tokenizer/chat template, deterministic temperature zero, synchronized diverse prompts, and
128 completion-token standard traffic. The four mixed active-decode streams use 256 tokens so they
remain active when the long prefill arrives 50 ms later; this explicit deviation is isolated to the
mixed probe. Parentheses are aggregate-throughput CV; latency columns are p50/p95/p99.

| N | fucina Q4_0 aggregate tok/s (CV) | fucina TTFT ms | vLLM BF16 aggregate tok/s (CV) | vLLM TTFT ms |
|---:|---:|---:|---:|---:|
| 1 | 17.887 (0.37%) | 265.6 / 265.6 / 265.6 | 7.618 (0.22%) | 295.6 / 295.6 / 295.6 |
| 2 | 33.953 (0.44%) | 402.0 / 516.0 / 526.1 | 18.662 (1.29%) | 252.0 / 252.1 / 252.1 |
| 4 | 62.363 (0.26%) | 654.8 / 996.3 / 1,026.9 | 36.960 (0.55%) | 293.0 / 397.4 / 398.7 |
| 8 | 81.744 (0.29%) | 1,155.4 / 1,949.4 / 2,020.7 | 72.842 (0.41%) | 398.2 / 446.9 / 447.0 |
| 16 | 92.082 (0.23%) | 2,160.7 / 3,860.3 / 4,011.2 | 142.493 (0.42%) | 539.5 / 542.2 / 542.7 |
| 32 | 96.973 (0.20%) | 4,156.4 / 7,663.6 / 7,975.0 | 273.130 (0.66%) | 785.0 / 787.8 / 788.1 |

Fucina's token-exact ITL p50/p95/p99 ranges from 54.5/58.4/59.0 ms at N=1 to
269.3/273.8/276.0 ms at N=32. vLLM's diagnostic SSE-chunk intervals range from
131.0/140.4/141.5 ms to 113.7/124.9/128.5 ms, but are **not published as token ITL**: some
128-token replies contain 127 non-empty content chunks (two tokens coalesced). True per-token vLLM
ITL is `MISSING`; aggregate completion throughput, TTFT and wall time remain usage-count valid.
Different target arithmetic independently prevents naming a cross-engine winner.

### Same-target fucina batched MTP productization probe — three starts

This is the one directly comparable performance probe: the target artifact, tokenizer, traffic,
KV format, and server are identical; only the pinned Q8_0 assistant is added. Throughput variance
is low enough for a non-flaky 5% floor, but the candidate **fails** that floor.

| N | plain median tok/s (CV) | MTP median tok/s (CV) | MTP delta | 5% floor |
|---:|---:|---:|---:|---|
| 1 | 17.887 (0.37%) | 35.707 (0.42%) | **+99.63%** | pass |
| 2 | 33.953 (0.44%) | 46.519 (0.15%) | **+37.01%** | pass |
| 4 | 62.363 (0.26%) | 48.175 (0.08%) | **−22.75%** | **FAIL** |
| 8 | 81.744 (0.29%) | 62.745 (0.08%) | **−23.24%** | **FAIL** |
| 16 | 92.082 (0.23%) | 91.657 (0.12%) | −0.46% | pass |
| 32 | 96.973 (0.20%) | 96.563 (0.14%) | −0.42% | pass |

The +15% target-band requirement is met at N=1–2, but the no-regression requirement is violated at
N=4–8. The 32-row verify budget explains the shape: learned drafting is deep and effective at low
B, expensive at intermediate B, and automatically tapers to plain decode at B≥16. Current source
needs a concurrency/utility gate that declines learned MTP at the lossy intermediate shapes; this
branch does not implement it.

All seven plain/MTP quality outputs match SHA-256 across all three starts. The newer token-event
hash capture matches exactly for every case on starts 2–3; start 1 predates that harness field but
its preserved UTF-8 output hash also matches. This is served losslessness evidence in addition to
the scheduler unit tests. It does not override the separate failing paged/batch GPU gates.

Batch `/metrics` incorrectly reports all speculation counters as zero even while MTP is visibly
active and throughput changes. True drafted/accepted counts and accepted-length distributions are
therefore **MISSING**; they are not inferred from near-simultaneous SSE events. This observability
gap independently blocks productization. `protection-report.json` is enforcement-capable on
variance and correctly fails N=4 and N=8; it remains report-only because broader correctness is
red.

### Prompt, prefix, mixed load, and startup

| Probe | fucina Q4_0 start 1 | vLLM BF16 start 1 |
|---|---:|---:|
| startup to ready | 26.354 s (11.70% CV) | 269.727 s (0.69% CV) |
| ~3,500-token cold TTFT, median of 6 | 37.277 s | 1.314 s |
| 4,002-token TTFT | 43.574 s | 1.468 s |
| 15,877-token TTFT | 262.470 s | 6.472 s |
| 4,751-token prefix cold TTFT | 53.382 s | 1.738 s |
| same prefix warm TTFT | 53.367 s | 0.283 s |
| mixed active-decode TTFT p50 | 0.649 s | 0.269 s (101% CV; not stable) |
| arriving ~3,500-token prefill TTFT | 38.783 s | 1.373 s |

Fucina's cumulative prefix counters remained zero and warm TTFT was unchanged. vLLM reported
4,736 cached prompt tokens per run, zero preemptions, and a sixfold warm-prefix TTFT reduction.
vLLM model load used 22.83 GiB, exposed about 80.95 GiB for a ~504.8k-token FP8 KV cache,
compiled for ~30.5 s, and captured graphs for ~20 s (0.92 GiB on start 1); median process-to-ready
was 269.7 s. Fucina reported a Q4_0 model plus optimized copies/pools and median ready in 26.4 s.
The first post-ready quality request had median TTFT/wall 283 ms/5.38 s plain, 281 ms/2.21 s MTP,
and 1.74 s/13.98 s vLLM; warmed N=1 is reported separately above. Host RSS for vLLM's API PID
alone undercounts the separate EngineCore and is not used as a memory comparison; physical
available memory and the engine's own memory lines are retained raw.

The dominant measured fucina bottleneck is unsupported/long paged prefill: ~3.5k tokens took
~37 s to first token and 15.9k took ~262 s, versus roughly 1.3 s and 6.3 s in vLLM. This is much
larger than any decode scheduling difference and is the first runtime item after correctness and
artifact loading are restored.

## Quality and token-event identity

The fixed greedy corpus covers arithmetic, Python, multilingual text, exact repetition, Rust,
reasoning, and 3.5k-token retrieval (`COBALT-7319`). Full UTF-8 outputs, length-prefixed
per-SSE-content-event SHA-256 hashes, event boundaries, and counts are in each raw JSON. Fucina
emits one content event per reported token in every request; vLLM chunk boundaries are stable for
the quality corpus but are not claimed as token IDs.

- Fucina Q4_0 passes manual semantic review and has stable output hashes across all three starts.
- vLLM BF16 passes manual semantic review and has stable output/token-event hashes across all three
  starts. The initial start-1 automated reasoning heuristic required the word `even`; the correct
  proof was truncated at the fixed 96-token quality cap before that word. The preserved answer
  correctly begins the contradiction proof, and the corrected budget-aware rule passes starts
  2–3. This adjudication does not assert equality with the differently quantized fucina output.
- Plain-vs-MTP equality is a separate same-target gate and uses token-event hashes by case and
  start. It is never inferred from decoded text alone.

## Correctness and tooling gates

| Gate | Result | Detail |
|---|---|---|
| `make lib libdg fucina` + `go test ./... -count=1` | PASS | clean required base/branch |
| dense paged-KV device | **FAIL** | global paged-vs-contiguous max error 0.0112; paged-vs-host 1.19e-07; sliding cases pass |
| dense batch self-test | **FAIL** | `seq_add(batch) failed`, including explicit four-slot rerun |
| legacy `make bench` correctness | **FAIL** | batch/sampling markers fail; one plain-vs-batch greedy text probe matches but does not override gate |
| exact dense NVFP4 load | **FAIL** | L5/P1 packed and scale geometry mismatch; server never ready |
| E4B config/PLE, BF16 load, Q4_0 load sanity, hybrid NVFP4+FP8 | PASS | raw logs retained |
| E4B batch parity | **FAIL** | third sequence differs at 2/8 token positions |
| E4B HF forward/generation oracle | **MISSING** | `/tmp/e4b_ref.bin`, `/tmp/e4b_gen_ref.bin`, and a repository producer are absent |
| E4B assistant Make targets | **BROKEN TARGET** | omit `libdg.a`, undefined `dg_*` symbols |
| unchanged E4B assistant tests manually linked with `libdg.a` | PASS | load, 160-token lossless spec, 160-token lossless server stream |

Because these gates fail, performance rows are observations beside preserved quality, not merge
claims. No tolerance was weakened and no runtime fix was made on this branch.

## Protocol details and raw data

- Hardware: GB10, 48 SM, compute capability 12.1 / `sm_121a`, CUDA 13, driver 580.142,
  128 GiB unified LPDDR.
- `GPU_CLOCK_MAX=2400` was supplied to fucina. The vLLM image does not implement fucina's custom
  environment control; observed clock/performance snapshots are retained per start. This is an
  additional reason not to present the cross-format row as parity.
- Continuous batching was enabled. Fucina plain used `FUCINA_NO_BATCH_SPEC=1`; `--spec=false`
  alone is not a valid plain batch baseline. vLLM used `TRITON_ATTN`, async scheduling, chunked
  prefill, prefix caching, FP8 KV, and no speculative config.
- Every vLLM N=1 run occurred after two explicit warm requests and after model compile/graph
  capture; no cold first-request artifact was averaged into the row.
- Fucina streaming usage currently reports prompt tokens as zero; exact 3,501/4,002/15,877/4,751
  counts are preserved in server logs. vLLM reports those counts in API usage.
- Raw request arrays: [`raw/`](raw/). Aggregation: [`summary.json`](summary.json). Harness/event
  audit: [`harness-audit.json`](harness-audit.json). Quality adjudication:
  [`quality-adjudication.json`](quality-adjudication.json). Report-only floor checker:
  [`protection-report.json`](protection-report.json).

## Missing acquisition/tooling requests

1. **Dense vLLM MTP:** `google/gemma-4-12B-it-assistant@364bd03c9952e5b7da73665ee30c9eccfc408345`;
   `model.safetensors` is 845,719,296 bytes, LFS SHA-256
   `3279c173daddd7186e79d652ad94022415736d3a1370625696c898429b06d6df`, plus config/tokenizer
   metadata. The local Q8_0 GGUF assistant cannot establish same-format parity.
2. **E4B vLLM MTP:** `google/gemma-4-E4B-it-assistant@8d0031ea8c2109e2b1e86bb9368a4539b537f80a`;
   `model.safetensors` is 159,138,208 bytes, LFS SHA-256
   `12875062fc25c51e8fa9b62abd2de7ad48b7d63f8559d5d604fbd5a3d6bcff16`. Local assistant is
   GGUF Q4_0 only. Both assistant records came from the Hugging Face API (`blobs=true`); no weights
   were downloaded.
3. **E4B oracle tooling:** repository-pinned producer and hashes for `e4b_ref.bin` and
   `e4b_gen_ref.bin`.
4. **Dense exact cell runtime tooling:** loader support for the already-local, already-hashed
   RedHat NVFP4 artifact. No replacement checkpoint was downloaded.

## Bottom line

- **Competitive dense winner:** unknown.
- **MTP productization:** **FAIL**—despite +99.6% at N=1 and +37.0% at N=2 with exact outputs,
  N=4 and N=8 regress by 22.8% and 23.2%; GPU correctness and telemetry are also red.
- **E4B multi-tenant readiness:** no; batch parity is red even though single-sequence MTP is
  lossless and fast in its dedicated gate.
- **Top runtime bottleneck after correctness:** dense long-prompt paged prefill fallback.
- **Next brief:** repair exact NVFP4 tensor-geometry loading, root-cause global paged parity and
  self-test slot provisioning, then profile/tile long paged prefill before tuning MTP depth.
