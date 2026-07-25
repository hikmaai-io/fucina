# Fucina OKF bundle history

## 2026-07-25

* **MoE graph-decode resolution**: Recorded the captured-stream router fix, 6/24→24/24 row/graph parity, self-test FAIL→PASS, dual 8/8 oracle passes, and resolution of the historical `qwen-gates.log:417` failure.
* **Agentic quality**: Recorded Qwen3.6-35B-A3B-FP8 12/100→80/100 (1,299→347 s), final exact-continuation 78/100, and dense 27B parity at 81/100.
* **Performance**: Recorded three MoE B=1 decode runs (55.7/59.4/59.0 tok/s; 59.0 median), above the 46–53 tok/s baseline, plus the dense 27B smoke.
* **Known gaps**: Preserved continuous-batch `response_format` HTTP 501 (five scenarios), TC-60 sleeper injection, quarantined approximate continuation attention, and unchanged historical Gemma/E4B red gates.
* **Phase E execution**: Added the Qwen3.5 CUDA layer-range ABI, cgo shard runner, FCNDIST v2 lifecycle, TCP worker/coordinator CLI, an 8/8 byte-identical GB10 split-layer gate, and a separate-process TCP loopback generation.
* **Creation**: Added the initial OKF v0.2 bundle covering architecture, model support, DS4 implementation phases, test gates, performance evidence, known gaps, and roadmap.
* **Validation**: Recorded `go test ./...` as passing and `make check` as failing under the race detector in `TestBatchedAdmissionCancellation`.
* **Documentation**: Identified legacy Qwen3 claims and stale plan documents that no longer match the Qwen3.5-only detector on `main`.
* **Qwen HTTP sessions**: Documented scheduler-owned `q35-slot` restore/export, exact commit frontiers, strict-prefix and concurrent-writer safety, CPU/race coverage, and the dedicated residual GB10 restart gate.
