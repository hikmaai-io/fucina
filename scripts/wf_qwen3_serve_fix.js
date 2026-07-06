export const meta = {
  name: 'qwen3-serve-fix',
  description: 'Fix the broken Qwen3 HTTP-served path (LockOSThread + NVFP4-prefill guard) so Qwen3-8B-Q4_K serves end-to-end; then verify + bench served Qwen3',
  phases: [
    { title: 'Fix' },
    { title: 'Verify' },
  ],
}

const REPO = '/home/mauromedda/hack/fucina-tau'
const QWEN = '/opt/spark/models/Qwen3-8B-abliterated.Q4_K_M.gguf'
const GEMMA = '/opt/spark/models/gemma-4-12b-it-qat-q4_0.gguf'

const COMMON = [
  'Worktree: ' + REPO + ' (branch feat/dense-31b-tau, HEAD 04e13da). The gemma4 engine serves Qwen3-8B-Q4_K_M CORRECTLY at the C/multiseq level (make qwen3-parity-test = 8/8 vs llama.cpp), AND has packed-Q4_K decode + fast Q4_K/Q6_K prefill (paged_prefill_qwen3) + DSpark confidence-scheduled spec + chunked prefill — all committed. BUT the Go HTTP-SERVED path is BROKEN for Qwen3 (works for Gemma). THIS is the only remaining blocker to actually serving Qwen3.',
  'KNOWN FAILURE (reproduce first): serving ' + QWEN + ' via the HTTP server fails two ways:',
  '  (a) single-flight (default server, no --batch): SIGSEGV in gemma4_engine_prefill_batched (stack server.go:1090 -> internal/server/kvcache.go:280 -> internal/engine/cuda/bridge.go:194 cgo).',
  '  (b) --batch (continuous batching): HTTP 200 but completion_tokens:0 / empty content; engine log shows "NVFP4 weight build failed -> BF16 prefill fallback", "batched-prefill CUDA error: invalid device context" (x3 warmup), "batch: AddSeq failed: seq_add failed (no slot / not paged / prefill error)".',
  'ROOT-CAUSE LEADS (verified by inspection):',
  '  1. internal/server/batch/scheduler.go run() is launched via `go s.run()` (line ~345) WITHOUT runtime.LockOSThread(). The CUDA context is bound to main\'s locked OS thread (cmd/fucina/main.go:117). run() is the SOLE engine caller on the batch path but its goroutine migrates OS threads → "invalid device context". FIX: call runtime.LockOSThread() at the top of run() (and matching Unlock on exit). This is the likely fix for failure (b).',
  '  2. For a Qwen3 Q4_K_M checkpoint the engine should NOT build/use the NVFP4-prefill weight copy (that is for NVFP4 checkpoints / the fp4-budget path); Qwen3 prefill must use the new native paged_prefill_qwen3 (fast Q4_K/Q6_K dequant + cuBLAS GEMM) that the C parity test exercises. The "NVFP4 weight build failed -> BF16 prefill fallback" + the single-flight prefill_batched SIGSEGV suggest the server prefill entry routes Qwen3 to an fp4/BF16 prefill path that does not handle Q4_K/Q6_K. Guard the NVFP4-prefill build for non-NVFP4 models, and route the served prefill (both single-flight gemma4_engine_prefill_batched/prefill_flash AND the batch/paged path) to the WORKING Qwen3 prefill (paged_prefill_qwen3 / the chunked seq_prefill_chunk). The batch path is the priority (that is where DSpark + chunked prefill live); make single-flight work too if tractable, else at minimum ensure --batch serves Qwen3.',
  '  3. A warmup pass may also trigger the fp4 prefill build crash (early-session workaround was FUCINA_NO_WARMUP_PASS=1) — guard warmup for Qwen3 so the SERVER works WITHOUT needing that env var.',
  'BUILD env: `export CUT=/home/mauromedda/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass`; nvcc=/usr/local/cuda-13/bin/nvcc -arch=sm_121a. Build: `make lib CUTLASS_DIR=$CUT && make fucina CUTLASS_DIR=$CUT`.',
  'HARD GATES (run ALL; never fake a green gate):',
  '  G1 BUILD clean.',
  '  G2 QWEN3 C PARITY still 8/8: `make qwen3-parity-test CUTLASS_DIR=$CUT` → [12095,13,576,6722,315,15344,374,21718].',
  '  G3 GEMMA still serves (regression guard): start the server on ' + GEMMA + ', POST a /v1/chat/completions, get a non-empty deterministic (temp 0) completion; and `./fucina -m ' + GEMMA + ' --prompt "The capital of France is" --predict 8 --temp 0` byte-identical to before (git-stash A/B if you touch shared code).',
  '  G4 GO TESTS: `go test ./internal/server/... ./internal/server/batch/ ./internal/tokenizer/ ./internal/sampler/` green.',
  '  G5 *** THE DELIVERABLE *** QWEN3 SERVED END-TO-END: start the server on ' + QWEN + ' and get a CORRECT non-empty completion via HTTP — BOTH with --batch AND (if fixable) single-flight. Test: launch `./fucina -m ' + QWEN + ' --batch --port <PORT> --ctx 4096 --gpu-mem-util 0.6 &` (background; wait for the "listening"/ready log; use a non-8080 port), then `curl -s http://127.0.0.1:<PORT>/v1/completions -d \'{"model":"x","prompt":"The capital of France is","max_tokens":12,"temperature":0}\'` (use /v1/completions raw, NOT chat, to avoid chat-template token divergence) → must return non-empty coherent text (temp 0 greedy → should contain "Paris"; compare the first tokens to the C parity decode of [12095,...] which is "Paris..."). Kill the server after. completion_tokens must be > 0 and NOT a crash.',
  'CONSTRAINTS: NO new per-token hot-path observability (error-path logs OK). Gemma path must stay correct (G3). Do NOT regress the C parity (G2) or the just-landed DSpark/chunked-prefill. Keep edits minimal + in-idiom. Commit on green (footer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`); never commit a failing gate.'
].join('\n')

phase('Fix')
const FIX = {
  type: 'object', additionalProperties: false,
  required: ['status','committed','commitSha','rootCause','whatChanged','buildOk','cParity8of8','gemmaServes','goTestOk','qwen3ServedBatch','qwen3ServedSingleFlight','servedSample','notes'],
  properties: {
    status: { type: 'string', enum: ['done','partial','blocked'] },
    committed: { type: 'boolean' }, commitSha: { type: 'string' },
    rootCause: { type: 'string', description: 'the confirmed root cause(s) of each failure mode' },
    whatChanged: { type: 'array', items: { type: 'string' } },
    buildOk: { type: 'boolean' }, cParity8of8: { type: 'boolean' }, gemmaServes: { type: 'boolean' }, goTestOk: { type: 'boolean' },
    qwen3ServedBatch: { type: 'boolean', description: 'Qwen3 served correctly via HTTP with --batch (non-empty correct completion)' },
    qwen3ServedSingleFlight: { type: 'boolean', description: 'Qwen3 served via HTTP single-flight (or honestly false if deferred)' },
    servedSample: { type: 'string', description: 'the actual served completion text for "The capital of France is"' },
    notes: { type: 'string' },
  },
}
const fix = await agent(COMMON +
  '\n\nYOUR TASK (FIX THE QWEN3 SERVED PATH). 1) Reproduce both failure modes on ' + QWEN + ' (single-flight + --batch) and capture the exact errors. 2) Apply fix lead #1 (runtime.LockOSThread in scheduler.run) and re-test the --batch path. 3) Apply leads #2/#3 (guard the NVFP4-prefill build for non-NVFP4 Qwen3; route the served prefill to the working paged_prefill_qwen3 / chunked path; guard warmup) until Qwen3 serves end-to-end. Root-cause each failure before fixing — confirm the fix addresses the actual crash, do not paper over it. Run ALL gates G1-G5. The deliverable is G5: a real non-empty correct Qwen3 completion over HTTP (--batch at minimum; single-flight too if tractable — if you defer single-flight, say so honestly and ensure --batch works). On green, COMMIT.',
  { label: 'fix:served', phase: 'Fix', schema: FIX, effort: 'high' })

phase('Verify')
const VERDICT = {
  type: 'object', additionalProperties: false,
  required: ['allGreen','cParity8of8','gemmaServes','qwen3ServedBatch','goTestOk','committed','headCommit','servedSample','perfSummary','findings','summary'],
  properties: {
    allGreen: { type: 'boolean' },
    cParity8of8: { type: 'boolean' }, gemmaServes: { type: 'boolean' }, qwen3ServedBatch: { type: 'boolean' }, goTestOk: { type: 'boolean' },
    committed: { type: 'boolean' }, headCommit: { type: 'string' },
    servedSample: { type: 'string', description: 'independently reproduced Qwen3 served completion' },
    perfSummary: { type: 'string', description: 'served Qwen3 per-user tok/s + TTFT at concurrency 1/8/16 (DSpark + chunked prefill now ON the Qwen3 served path); vs vLLM if obtainable' },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['severity','title','detail'],
      properties: { severity:{type:'string',enum:['critical','high','medium','low']}, title:{type:'string'}, detail:{type:'string'} } } },
    summary: { type: 'string' },
  },
}
const verify = await agent(COMMON +
  '\n\nFIX stage result:\n' + JSON.stringify(fix) +
  '\n\nADVERSARIAL VERIFY + BENCH (served Qwen3). From a CLEAN build, INDEPENDENTLY reproduce: G1 build; G2 C parity 8/8; G3 Gemma still serves; G4 go tests; G5 Qwen3 served over HTTP (--batch) returns a correct non-empty completion for "The capital of France is" (greedy temp 0 → coherent, contains Paris) — launch the server yourself, curl, kill. Confirm the fix is COMMITTED (git log --oneline -3; head sha). Then BENCH the Qwen3 SERVED path (now that it works): per-user decode tok/s + TTFT at concurrency 1/8/16 with --batch (DSpark + chunked prefill now active on Qwen3) — report the numbers; if a vLLM baseline is obtainable on the same model, compare (this is the original goal: beat vLLM on served Qwen3). Hunt for: any remaining crash under concurrency, a served output that diverges from C greedy, leftover hot-path instrumentation, or a fix that papers over the crash (e.g. disabling batching). Report allGreen + findings + headCommit + perfSummary + the served sample.',
  { label: 'verify:served', phase: 'Verify', schema: VERDICT, effort: 'high' })

return { fix, verify }
