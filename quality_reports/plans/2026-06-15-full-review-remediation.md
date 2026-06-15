# Fucina — Full Review & Remediation Plan (2026-06-15)

Reviewed at commit `687e20b` on a clean detached worktree (`/tmp/fucina-review`).
Verification gate baseline (PROVEN, re-run on clean tree):
- `gofmt -l .` clean; `go vet` (pure-Go pkgs) clean.
- `go test` + `-race` PASS on internal/{server,tokenizer,sampler,chat}.
- Coverage: server 74.4%, tokenizer 71.4%, sampler 95.9%, chat 19.4% (artifact — true ~77% via cross-pkg wrappers).
- `golangci-lint`: 13 issues (8 errcheck, 1 staticcheck, 4 unused).
- `go.mod`: zero external dependencies. No SQL/embedded DB anywhere (grep proven).

Threat-model anchor: default bind is `127.0.0.1` (args.go:139); `--host 0.0.0.0` is a documented first-class flag with NO auth code in the repo.

---

## Severity-ranked findings (all independently verified at file:line)

### P0 — CRITICAL (fix before any networked / untrusted-model deployment)

| # | Finding | Evidence | Verified |
|---|---------|----------|----------|
| C1 | Panic in cgo token callback is fatal across the C boundary — one bad request crashes the whole process, dropping all in-flight streams + model. No `recover()` anywhere (grep=0). | `internal/engine/cuda/callback.go:21-28` | Read directly |
| C2 | No top-level/handler recover; panics on the heartbeat goroutine (not owned by net/http) also crash the process. | `internal/server/sse.go:103`, `cmd/fucina/main.go:159-170` | Read directly |
| C3 | Diffusion GGUF loader has ZERO bounds checks: unchecked `fstat`, no magic/version re-check, walks `p` past mmap, `std::string(q,l)` with attacker `l`, `cudaMemcpy(data+offset,nbytes)` unvalidated → segfault / OOB GPU copy on malformed model. | `cuda/diffusion_gemma_engine.cu:139-148` | Read directly |
| C4 | AR loader: per-tensor offsets never validated vs tensor-data region → OOB device reads at inference on corrupt GGUF (silent garbage output). | `cuda/gemma4_kernels.cu:2884` (`gguf_find_tensor`), `:3483-3492` (upload) | Confirmed via reviewer + macro read |
| C5 | Missing required tensor loads silently as offset 0 (calloc'd) — wrong-tensor inference, no abort; `eng->loaded=1` still set. | `cuda/gemma4_kernels.cu:3249-3257` (`LOAD_TENSOR_OFFSET`) | Read directly |
| C6 | No authentication on ANY endpoint (both AR + diffusion servers). Catastrophic the moment `--host 0.0.0.0` is used. | `server.go:462-471`, `diffusion.go:344-465` | Read directly |
| C7 | Unvalidated sampling params flow raw into CUDA C kernels (`top_k`, `temp`, `top_p`, `min_p`). Negative/huge `top_k` or NaN/Inf temp reach C with no clamp on the default spec path. DoS-certain, potential heap corruption. | `server.go:625-639` (no clamp) → `bridge.go:351-352` (`C.int(topK)` raw) | Read directly |

### P1 — MAJOR (fix this cycle)

| # | Finding | Evidence |
|---|---------|----------|
| M1 | No concurrency cap / queue bound / 429. Engine is single-flight (`s.kv.Lock()` whole span); K clients → K−1 goroutines parked, each holding a fully-buffered body + live heartbeat. Unbounded tail latency, no backpressure. | `server.go:716-717`; absence proven (grep semaphore/429=0) |
| M2 | Diffusion server lacks body-size cap AND timeouts the AR server has: bare `json.NewDecoder` + `http.ListenAndServe` (no ReadHeaderTimeout/IdleTimeout). Trivial OOM/slowloris. | `diffusion.go:347, 463` |
| M3 | Uncapped output: `max_tokens` only clamped to `ctx/2` (up to 131072). One request monopolizes the single GPU for minutes → DoS amplifier under single-flight. | `server.go:649-651, 661-663` |
| M4 | 1 MiB logits buffer (`make([]float32, 262144)`) allocated PER Prefill/Decode/token; CPU decode loop churns GBs of GC garbage on the hot path. | `bridge.go:148, 212, 333` |
| M5 | `/metrics` is bespoke JSON, not Prometheus/OpenMetrics — uningestible by standard tooling. | `metrics.go:54-118`, `server.go:493` |
| M6 | No latency histogram, TTFT, error-rate, or queue-depth/in-flight metrics — the SLO + saturation signals are absent. | `server.go:528-538` (duration logged, never aggregated) |
| M7 | `/health` never probes engine/GPU (returns `{"status":"ok"}` always); no `/readyz`. Diffusion health is literal `"ok"`. Orchestrators route to dead GPUs. | `server.go:881-893`, `diffusion.go:460` |
| M8 | Unstructured `log.Printf` everywhere, no request/correlation IDs → failures uncorrelatable under concurrency. | repo-wide; grep slog/request-id=0 |
| M9 | Adversarial/malformed tool-call parser input is completely untested. `parseGemmaArray` 0% covered; unbounded recursion (stack-overflow risk on deep nesting); the most security-sensitive parser, only happy-path tested. | `internal/chat/tools.go:437-566` |
| M10 | CI runs neither lint nor `-race` nor coverage gate, though Makefile defines all. Concurrency regressions + 7 unchecked-error sites ship green. | `.github/workflows/ci.yml:32-43`; `Makefile:129,138,146` |
| M11 | 7 production unchecked-error sites (incl. `json.NewEncoder(w).Encode` response writer, SSE `Fprint`/`Fprintf`). | `server.go:618,619,1593,1650`; `sse.go:131,138,158` |
| M12 | `AbortPrefill` reads `e.ptr` unsynchronized vs `Close()`; 10s Shutdown timeout can let `eng.Close()` race a live watcher → UAF/nil-deref into CUDA at shutdown. | `bridge.go:188-193`, `main.go:155`, `server.go:787` |
| M13 | No request-body read deadline (`ReadTimeout` 0); slow-body client pins a connection and, if it wins the lock, stalls the single-flight queue. | `server.go:563, 1626` |
| M14 | Nil-tokenizer server startup: load failure is non-fatal in server mode only; the referenced "fallback" does not exist → serves with no tokenizer. | `cmd/fucina/main.go:112-116` |
| M15 | GGUF version field never checked in ANY loader → no forward-compat guard; a future format silently corrupts. Unchecked `fstat` in AR loader; possible `tbytes` underflow. | `gemma4_kernels.cu:3144-3146, 80`; `diffusion_*.cu:139` |
| M16 | Full ~12 GB GGUF read into Go heap (twice) via `os.ReadFile` just to extract tokenizer KV → load-time OOM risk; engine also mmaps it separately. | `cmd/fucina/main.go:62, 107` |

### P2 — MINOR (track / opportunistic)

| # | Finding | Evidence |
|---|---------|----------|
| m1 | Debug dump writes full clear-text prompts to world-readable `/tmp/fucina_debug.log` (0644), unbounded, predictable path. Not traversable (const path). | `server.go:460, 612-620` |
| m2 | SSE liveness gap during long suppressed-token generation (reasoning/tool buffer): no wall-clock heartbeat after prefill; stall detected only on visible-token writes. | `server.go:1381, 1410` |
| m3 | Control-token confinement: a user message containing literal `<turn|>`/`<\|tool_call>` tokenizes to real control ids → role-confusion / spoofed tool-call injection. | `template.go:99-160`, `tokenizer.go:451-487` |
| m4 | Dead code: `tokenizer.go:36 byteFallback`, `Score{id,score,str}` (78-80) — byte-fallback BEHAVIOR is implemented + tested; these are vestigial. | lint `unused` |
| m5 | `staticcheck QF1003` tagged-switch; `isEmptyArg` empty-array/object cases untested; `encodeGemmaValue` bool/null/array re-render untested (KV prefix-cache token-exactness). | `chat/tools.go:112, 263, 136` |
| m6 | `KVStateSize` returns 32-bit `int` from C; ~52 GB snapshot would overflow (bounded today by --ctx/16 GB budget). | `bridge.go:416-419` |
| m7 | `Metrics` uses a mutex while hot counters use atomics; `/metrics` "lock-free" guarantee not fully honored (O(1) window). | `metrics.go:9` |

### Confirmed NON-issues (credit / do not "fix")
- KV-cache bookkeeping (`kvcache.go`) is well-engineered: skew-healing invariants, lock ordering doc, 23 targeted tests incl. `-race`. **Leave it.**
- No regex catastrophic backtracking; loop detector + tokenizer scans are bounded.
- Release scripts (`scripts/release/*.sh`) reviewed: strict mode, quoted, no eval/curl|bash, no secrets.
- Tool-call required-param validation (TC-43) is adequate for its purpose (model-output guard, not a security boundary).
- No secrets in Go or scripts. CORS absent = safe (same-origin protects localhost).
- chat coverage "19.4%" is a per-package measurement artifact (~77% via server/tools.go wrappers).
- ABOUTME headers are NOT a fucina convention (global CLAUDE.md only) — do not retrofit.

---

## Remediation plan (phased, TDD where the code is Go)

### Phase 0 — CI safety net FIRST (do before any code change; pure-Go, no GPU)
0.1 Add to `.github/workflows/ci.yml`: a `golangci-lint run $PKGS` step and a `go test -race $PKGS` step.
0.2 Make `Makefile check: lint go-test-race` (currently `vet go-test`).
0.3 Add a coverage floor for the touched packages (informational gate to start).
Rationale: every later fix lands behind a gate that would catch its regressions. Commit: `chore(ci): enforce lint + race + coverage on pure-Go packages`.

### Phase 1 — P0 process-survival + auth + input validation (Go-only, no GPU needed)
1.1 **C1/C2 panic safety** — wrap `fucinaSpecTokenGo` body in `defer recover()` that sets a sticky abort and returns 1 (stop) so the panic never reaches C. Add `defer recover()` in the `logRequest` middleware and inside the heartbeat goroutine. Tests: a handler/closure that panics must not crash the test binary (use a panicking emit stub).
1.2 **C7 param validation** — add `validateParams()` called right after the `req.*` overrides (`server.go:639`): clamp `top_k∈[0,vocab]`, reject NaN/Inf, clamp `temp/top_p/min_p` to finite non-negative; apply to BOTH AR and diffusion. Table-driven test.
1.3 **C6 auth** — bearer-token middleware (constant-time compare vs `--api-key`/env) wrapping all `/v1/*` on both muxes; refuse to start with non-loopback `--host` unless a key is set. `/health` may stay open. Tests: 401 without/with-wrong key, 200 with key.
1.4 **M3 hard max_tokens ceiling** — absolute server-side cap independent of ctx (configurable, default ~4096–8192). Test the clamp.
Commit per item (`fix(server): ...`, `feat(server): bearer auth`).

### Phase 2 — P1 reliability + scalability (Go-only)
2.1 **M1 concurrency cap** — bounded semaphore (`chan struct{}`) acquired with `select` on `r.Context()`; 503 + `Retry-After` when full; start heartbeat only AFTER acquiring the lock. Test queue-full → 503, context-cancel releases slot.
2.2 **M2 diffusion hardening** — `http.MaxBytesReader` + `http.Server{ReadHeaderTimeout, IdleTimeout}` mirroring the AR server.
2.3 **M4 logits scratch buffer** — per-engine reusable buffer (engine is single-flight) instead of `make` per call. Benchmark before/after allocs.
2.4 **M11 unchecked errors** + **m4 dead code** + **m5 tagged-switch** — fix the 7 errcheck sites (handle `json.Encode`/SSE writes), delete `byteFallback`/`Score`. (Boy Scout; lint gate from Phase 0 enforces.)
2.5 **M12 shutdown race** — guard `e.ptr` via `atomic.Pointer` (safe nil read) OR make `Stop()` block on in-flight handlers before `eng.Close()`.
2.6 **M13/M14** — arm body read deadline; make nil-tokenizer fatal in server mode.

### Phase 3 — P1 observability
3.1 **M5/M6** — add `github.com/prometheus/client_golang`, expose `promhttp` at `/metrics` (keep JSON at `/debug/stats`): request-duration histogram{path,status}, outcome counter{status}, TTFT histogram, `inflight_requests` gauge, `queue_wait_seconds` histogram + existing engine gauges re-exported. NOTE: first external dependency — 🔴 needs Mr. Wolf's approval.
3.2 **M7** — split `/healthz` (liveness) from `/readyz` (active engine probe → 503 when not serviceable); apply to diffusion.
3.3 **M8** — migrate to `log/slog` JSON; per-request ID in context + every log line; honor `X-Request-Id`.

### Phase 4 — P0/P1 model-loader integrity (CUDA/C++ — REQUIRES GB10 to build/verify)
4.1 **C3/C4/C5/M15** — thread `end = base+size` through every read in BOTH loaders; reject any offset/len past mmap; validate magic + version (reject != 3); detect missing required tensors and abort (sentinel `UINT64_MAX`, not 0); check `fstat`; honor `general.alignment`; guard `tbytes` underflow. Add a malformed/truncated-GGUF fixture test (standalone `nvcc` harness) — cannot run in CI; run on GB10.
4.2 **M16** — mmap once / stream only the metadata KV region for the tokenizer instead of two full heap copies.
4.3 **m6** — widen `KVStateSize` to `int64`/`size_t`.
Gate: these compile/verify only on the DGX Spark GB10 (CI is CPU-only by design).

### Phase 5 — P1 test hardening
5.1 **M9** — new `internal/chat/tools_parse_test.go`: table-driven ~25 malformed/adversarial inputs (unterminated string/dict/array, top-level array, deep nesting → assert terminates, unicode, partial open-marker, garbage between calls) → assert no-panic + exact `(content, []ToolCall)`. Closes `parseGemmaArray` 0%.
5.2 **m5** — `isEmptyArg` empty-array/object cases; `encodeGemmaValue` bool/null/array byte-exact round-trip; `validate_test` assert `DroppedCall.Param`.

### Phase 6 — P2 hygiene
6.1 **m1** debug dump: `0600`, configurable/private path, size cap, flag-only.
6.2 **m2** wall-clock heartbeat through generation (single-writer-safe via the token-callback path).
6.3 **m3** strip/escape control-marker literals in user/tool content before rendering (defense-in-depth).

---

## Sequencing & gating
- Phase 0 → 1 → 2 → 3 → 5 → 6 are all pure-Go and verifiable here. Phase 4 is CUDA, verifiable only on GB10.
- 🔴 Approval gates (per decision framework): Phase 3.1 (new external dep `prometheus/client_golang`), Phase 1.3 (auth = API/interface change), Phase 4 (touches CUDA core).
- Run `/simplify` after Phases 1, 2, 3 (each >100 lines / >3 files).
- Every phase: Red→Green→Refactor→commit (Conventional Commits), `make check` (now lint+race) must be pristine.

## Recommended immediate scope (if doing one PR now)
Phase 0 + Phase 1 (CI gate, panic safety, param validation, auth, max_tokens cap) — all Go, no GPU, and they neutralize every "one request takes down / corrupts / hangs the box" class. Phase 4 (model-loader bounds) is the other must-do but needs the GB10.
