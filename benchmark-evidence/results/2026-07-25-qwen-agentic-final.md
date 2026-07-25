# Qwen3.6 MoE graph-router final qualification — GB10

Date: 2026-07-25

Branch base: local `main` `69bdacb` plus `fix/qwen-agentic-quality`

Hardware: NVIDIA GB10, CUDA 13, `sm_121a`

Serialization: every model run held `/tmp/fucina_gpu.lock`

## Correctness incident

| Measurement | Before | Router-fixed | Final exact-continuation branch |
|---|---:|---:|---:|
| Full 69-scenario tool-eval | 12/100 | **80/100** | **78/100** |
| Wall time | 1,299 s | **347 s** | **370.3 s** |
| B=3 vs B=1 | 6/24 | **24/24** | **24/24** |
| Graph-on vs graph-off | 6/24 | **24/24** | **24/24** |
| Self-chain | FAIL | **PASS** | **PASS** |
| Engine oracle | 8/8 | 8/8 | 8/8 |
| Standalone torch oracle | 8/8 | 8/8 | 8/8 |

The 80/100 report is `tool-eval-bench v2.0.4` run
`2026-07-25T18-58-59.079320Z_179ae609` (110/138, Good). The final branch report is run
`2026-07-25T20-18-37.931234Z_0eb0489d` (107/138, Good). The final branch makes exact scalar
base>0 continuation attention the production default after the old tensor-core candidate measured
2/25 against scalar/one-shot 25/25 under correct router replay. The isolated router-fix comparison
is therefore 12→80; the exact final binary is reported separately as 78.

The historical failure in
`benchmark-evidence/results/2026-07-19-qwen35-burst-ttft2/qwen-gates.log:417` is resolved, not
edited in place.

## Final gate results

- Clean rebuild and cgo relink: PASS.
- `make gpu-gates`: PASS.
- `make qwen35-moe-fp8-engine-test`: PASS; all three ragged rows 24/24, graph parity 24/24,
  self-chain PASS, oracle 8/8.
- `make qwen35-moe-fp8-test`: PASS; torch oracle 8/8.
- `make qwen35-state-test`: PASS; restored continuation 16/16.
- `make qwen35-shard-test`: PASS; frontier 8/8 and oracle 8/8.
- `make qwen35-http-session-restart-test`: PASS; restored 11 cached tokens and prefilled only the
  7-token suffix.
- `go test ./... -count=1`: PASS.
- `make check`: PASS for vet and race; `golangci-lint` was not installed and explicitly skipped.
- `make test`: PASS; its `--test-parser` and `--test-cuda` commands remain placeholders.

## Decode throughput

Official Qwen3.6-35B-A3B-FP8 snapshot `95a723d08a9490559dae23d0cff1d9466213d989`, 128 timed
steps after 32 warm steps, served `step_batch` CUDA graph:

| B | rep 1 | rep 2 | rep 3 | median aggregate tok/s |
|---:|---:|---:|---:|---:|
| 1 | 55.7 | 59.4 | 59.0 | **59.0** |
| 2 | 92.0 | 95.4 | 96.5 | **95.4** |
| 4 | 147.2 | 150.3 | 151.8 | **150.3** |
| 8 | 215.8 | 219.8 | 221.0 | **219.8** |

B=1 median step time was 16.96 ms. This exceeds the historical 46–53 tok/s served baseline; the
captured-stream router correction did not regress decode.

## Dense and Gemma regression checks

Qwen3.6-27B-FP8 B=1 measured 11.90 tok/s over 64 graph steps. HTTP returned
`The sea whispers ancient secrets to the shore.` and a forced `get_weather` call with
`{"location":"Berlin"}`. Its same-day complete quality reference is 81/100.

Gemma results reproduced their established signatures:

- `make paged-kv-device-test`: PASS (global max error 0.000968; sliding exact).
- E4B config/PLE, BF16 load, Q4_0 load sanity, and NVFP4 kernel gates: PASS.
- E4B MTP load, 160-token lossless spec, and 160-token streaming spec: PASS after adding the missing
  `libdg.a` linkage to their Make targets (57.8→168.8 tok/s in the measured spec gate).
- Legacy dense `make bench`: FAIL on missing batch/sampling self-test markers, while its direct
  plain-vs-batch output remains byte-identical. Same historical failure.
- `e4b-batch-test`: FAIL with the same two mismatches in sequence 2 as the prior evidence. No new
  Gemma/E4B failure signature was introduced.
- `e4b-fwd-test` and `e4b-gen-test`: unavailable because `/tmp/e4b_ref.bin` and
  `/tmp/e4b_gen_ref.bin` and a pinned repository producer are absent.

## Remaining known gaps

- Continuous-batch `response_format` returns HTTP 501 and loses TC-64, TC-65, TC-66, TC-67, and
  TC-69.
- TC-60 cross-turn sleeper injection remains a safety-critical failure.
- Exact scalar continuation prefill is slower than the quarantined tensor-core candidate for the
  measured 1,376-token base>0 chunk (about 2.04 s versus 1.09 s); decode throughput is unaffected.
