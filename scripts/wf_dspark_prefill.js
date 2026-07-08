export const meta = {
  name: 'dspark-and-chunked-prefill',
  description: 'DSpark confidence-scheduled verify (paper Alg.1, training-free) + chunked prefill & prefix caching on tau — serial, each gated (lossless + parity 8/8 + Gemma byte-identical) and committed',
  phases: [
    { title: 'DSpark' },
    { title: 'Prefill' },
    { title: 'Verify' },
  ],
}

const REPO = '/home/mauromedda/hack/fucina-tau'
const SURV = '/home/mauromedda/hack/fucina'  // e4b-gguf worktree: holds the surviving uncommitted specsched.go
const QWEN = '/opt/spark/models/Qwen3-8B-abliterated.Q4_K_M.gguf'
const GEMMA = '/opt/spark/models/gemma-4-12b-it-qat-q4_0.gguf'

const COMMON = [
  'Worktree: ' + REPO + ' (branch feat/dense-31b-tau, HEAD has the rebuilt Qwen3-dense engine + Q4_K + packed decode + fast prefill + DSpark prompt-lookup spec wiring already committed). Qwen3-8B-Q4_K_M serves at parity 8/8 vs llama.cpp. The continuous-batching scheduler is internal/server/batch/scheduler.go; the spec path is scheduler.stepSpec (prompt-lookup drafter internal/server/batch/drafter.go, per-seq hist, default-on, lossless greedy verify); the cgo bridge internal/engine/cuda/bridge.go (BatchAdapter.StepBatchSpec → C gemma4_engine_step_batch_spec / _ext); the engine cuda/gemma4_kernels.cu.',
  'BUILD env: `export CUT=/home/mauromedda/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass`; nvcc=/usr/local/cuda-13/bin/nvcc -arch=sm_121a. Build: `make lib CUTLASS_DIR=$CUT && make fucina CUTLASS_DIR=$CUT`.',
  'HARD GATES every stage (run ALL; never claim success on a failing gate):',
  '  G1 BUILD clean (make lib + make fucina).',
  '  G2 QWEN3 PARITY 8/8: `make qwen3-parity-test CUTLASS_DIR=$CUT` → 8/8, tokens [12095,13,576,6722,315,15344,374,21718].',
  '  G3 GEMMA byte-identity: `./fucina -m ' + GEMMA + ' --prompt "The capital of France is" --predict 16 --temp 0` equals the pre-change output (git-stash A/B). Gemma path must be untouched. (Note: tau Gemma raw-prompt output is a pre-existing degenerate "011111…"; identity = same string before/after, that is the gate, not the content.)',
  '  G4 GO TESTS: `cd ' + REPO + ' && go test ./internal/server/... ./internal/server/batch/ ./internal/tokenizer/ ./internal/sampler/` green.',
  'LOSSLESS is the cardinal correctness property for BOTH features: any scheduling/caching change must produce EXACTLY the same generated tokens as the non-feature path for greedy (temp=0). Prove it with a test, not by assertion.',
  'CONSTRAINTS: NO new per-token hot-path observability/logging (user directive — only error-path logs allowed; remove any debug prints before gates). Keep Gemma + existing Qwen3 decode/spec correct. Do NOT touch the NVFP4 path. Keep edits minimal + in-idiom.',
  'COMMIT POLICY: after ALL a stage\'s gates pass, COMMIT on feat/dense-31b-tau (progress must never be uncommitted — that is how this engine was lost before). Footer line exactly: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. If a gate fails and you cannot fix it, commit only the parts that DO pass their gates (or nothing) and report status=partial/blocked with exact failing output — never fake a green gate.'
].join('\n')

phase('DSpark')
const DSPARK = {
  type: 'object', additionalProperties: false,
  required: ['status','committed','commitSha','whatChanged','buildOk','parity8of8','gemmaIdentical','goTestOk','lossless','perUserTokS','notes'],
  properties: {
    status: { type: 'string', enum: ['done','partial','blocked'] },
    committed: { type: 'boolean' }, commitSha: { type: 'string' },
    whatChanged: { type: 'array', items: { type: 'string' } },
    buildOk: { type: 'boolean' }, parity8of8: { type: 'boolean' }, gemmaIdentical: { type: 'boolean' }, goTestOk: { type: 'boolean' },
    lossless: { type: 'boolean', description: 'DSpark-scheduled spec greedy output == plain decode output, with dynamic per-request verify lengths' },
    perUserTokS: { type: 'object', additionalProperties: true, description: 'per-user decode tok/s at concurrencies (e.g. {"1":..,"8":..,"16":..}) DSpark-on vs fixed-draftK baseline' },
    notes: { type: 'string' },
  },
}
const dspark = await agent(COMMON +
  '\n\nYOUR TASK (DSpark CONFIDENCE-SCHEDULED VERIFICATION — paper Algorithm 1, training-free portable half). The paper is DeepSeek "Confidence-Scheduled Speculative Decoding". Its SYSTEM win (the high-concurrency differentiator, training-free) is the Hardware-Aware Prefix Scheduler: per step, given each request r a sequence of draft positions j with a per-position SURVIVAL estimate a_{r,j}=∏_{i<=j} c_{r,i} (cumulative accept prob), GREEDILY admit draft tokens across ALL requests in descending survival, B+=1 per admit, τ=Σ_r(1+Σ_{j<=ℓ_r} a_{r,j}), Θ=τ·SPS(B); admit while Θ rises, NON-ANTICIPATING EARLY-STOP when Θ drops (this stop preserves losslessness). It dynamically tailors per-request verify length ℓ*[r] by load — at high concurrency drafts shrink toward 0 (degenerating to plain step_batch), at low concurrency they grow. The semi-AR Markov/RNN drafter half NEEDS TRAINING — OUT OF SCOPE; note it.',
  '\n\nThe POLICY already exists, fully tested, UNCOMMITTED in a different worktree: ' + SURV + '/internal/server/batch/specsched.go + specsched_test.go (NewSpecScheduler/NewSpecSchedulerTable/SPSFromTable; BuildVerifyPlan(conf [][]float64, R int) []int → per-request ℓ*[r] via Alg.1: survival prefix-product, descending sort with total-order tiebreak, greedy Θ=τ·SPS(B) non-anticipating early-stop, prefix invariant asserted; 7 tests incl tail-truncation invariance). COPY specsched.go + specsched_test.go into tau (' + REPO + '/internal/server/batch/), adapt to tau\'s package, `go test` them green.',
  '\n\nThen WIRE it into tau\'s scheduler.stepSpec (which today uses a FIXED draftK=6): (1) SPS(B) — profile steps/sec vs forward batch size B ONCE at engine create (a tiny timing loop over B=1..MAX, or derive from a cheap model; expose a table to Go via bridge, e.g. Engine.SPSTable() []float64). (2) CONFIDENCE PROXY (training-free, since fucina has no trained confidence head) — per draft position give a survival estimate c_{r,i}: use the prompt-lookup drafter\'s n-gram consensus strength AND/OR an ONLINE per-position historical accept-rate EMA (track, per draft slot-position, the fraction of recent steps where that draft position was accepted) AND/OR the MTP draft logit margin when an MTP assistant is loaded. Pick a simple robust proxy; it only chooses WHICH/HOW-MANY tokens to draft — accept/reject stays the EXACT greedy verify, so the result is LOSSLESS regardless of proxy quality. (3) Replace the fixed draftK: build conf[r][j] for each active request from its drafter\'s proposed tokens + proxy, call BuildVerifyPlan to get ℓ*[r], truncate each request\'s drafts to ℓ*[r], call StepBatchSpec, commit accepted runs (existing logic). Budget Σ(1+ℓ*[r]) ≤ MaxVerifyRows still holds.',
  '\n\nGATES: G1-G4 PLUS: (a) LOSSLESS — a Go test (extend the cycle-engine spec_lossless_test) proving DSpark-scheduled spec greedy output == plain decode output, with VARIABLE per-request verify lengths and at ≥2 concurrencies; (b) the new specsched tests + scheduler tests green; (c) MEASURE per-user decode tok/s at concurrency 1/8/16 DSpark-on vs the fixed-draftK baseline (the DSpark claim is HIGHER per-user throughput under concurrency by not wasting verify on low-survival drafts) — report the numbers, keep the change if it does not regress and ideally improves under load. parity G2 + Gemma G3 should be unaffected (Go-side scheduling; if you add a .cu conf sidecar or per-position RNG, gate it and keep greedy bit-identical). On green, COMMIT.',
  { label: 'dspark:sched', phase: 'DSpark', schema: DSPARK, effort: 'high' })

phase('Prefill')
const PREFILL = {
  type: 'object', additionalProperties: false,
  required: ['status','committed','commitSha','whatChanged','buildOk','parity8of8','gemmaIdentical','goTestOk','chunkedLossless','prefixCacheLossless','ttftConcurrent','notes'],
  properties: {
    status: { type: 'string', enum: ['done','partial','blocked'] },
    committed: { type: 'boolean' }, commitSha: { type: 'string' },
    whatChanged: { type: 'array', items: { type: 'string' } },
    buildOk: { type: 'boolean' }, parity8of8: { type: 'boolean' }, gemmaIdentical: { type: 'boolean' }, goTestOk: { type: 'boolean' },
    chunkedLossless: { type: 'boolean', description: 'chunked prefill produces identical tokens to single-shot prefill' },
    prefixCacheLossless: { type: 'boolean', description: 'prefix-cache reuse produces identical tokens to cold prefill' },
    ttftConcurrent: { type: 'object', additionalProperties: true, description: 'TTFT ms at concurrency 1/8/16 before vs after' },
    notes: { type: 'string' },
  },
}
const prefill = await agent(COMMON +
  '\n\nPRIOR STAGE (DSpark) result:\n' + JSON.stringify(dspark) +
  '\n\nYOUR TASK (CHUNKED PREFILL + PREFIX CACHING + PREFILL/DECODE INTERLEAVE — the high-concurrency TTFT lever; vLLM has all three, fucina runs each prefill to completion and blocks the batch so TTFT grows ~linearly with concurrency). Implement in internal/server/batch/scheduler.go + the engine as needed:',
  '\n(1) CHUNKED PREFILL: split a long prompt\'s prefill into token-budget chunks (e.g. 512 tok) and feed them across scheduler steps, INTERLEAVED with ongoing decode of other sequences, instead of one blocking seq_add of the whole prompt. The engine must support incremental prefill: prefill tokens [c0, c0+cn) for a seq, append their KV to its paged blocks, and continue next step from c0+cn — the paged KV + the prefill path (paged_prefill_*) already append position-for-position, so add a chunk loop + per-seq prefill cursor in the scheduler and an engine entrypoint that prefills a CONTIGUOUS RANGE of a seq\'s prompt. The first decode token is produced only after the last chunk. MUST be lossless: chunked prefill output == single-shot prefill output (identical tokens).',
  '\n(2) PREFIX CACHING: when a new request\'s prompt shares a leading prefix with an already-resident sequence, REUSE that prefix\'s KV instead of recomputing it. The paged KV makes this natural via block-table sharing: hash prompt blocks, keep a block→content map, and point the new seq\'s block table at existing blocks for the shared prefix (ref-count; copy-on-write at the first divergent block). MUST be lossless: a request served with a cached prefix produces identical tokens to a cold full prefill. Start simple (exact full-block prefix match, ref-counted shared blocks) — correctness over coverage.',
  '\n(3) INTERLEAVE: the scheduler step should be able to mix prefill chunks (for admitting/continuing seqs) AND decode (for running seqs) so a big prefill never starves decode — a token budget per step split across prefill+decode.',
  '\nGATES: G1-G4 PLUS: (a) chunkedLossless — a test/harness proving chunked prefill (e.g. chunk=64) yields identical greedy tokens to single-shot prefill on the same prompt; (b) prefixCacheLossless — two requests sharing a prefix: the second (cache hit) yields identical tokens to it run cold; (c) MEASURE TTFT at concurrency 1/8/16 before vs after (expect flat/low TTFT under concurrency vs the old linear growth). go tests green incl new ones. parity G2 + Gemma G3 unaffected. On green, COMMIT. If prefix-caching proves too large to finish losslessly this stage, ship CHUNKED PREFILL + INTERLEAVE (gated) and report prefix-caching as partial — do NOT ship an unverified cache.',
  { label: 'prefill:chunked', phase: 'Prefill', schema: PREFILL, effort: 'high' })

phase('Verify')
const VERDICT = {
  type: 'object', additionalProperties: false,
  required: ['allGreen','parity8of8','gemmaIdentical','goTestOk','dsparkCommitted','prefillCommitted','headCommit','losslessConfirmed','perfSummary','findings','summary'],
  properties: {
    allGreen: { type: 'boolean' },
    parity8of8: { type: 'boolean' }, gemmaIdentical: { type: 'boolean' }, goTestOk: { type: 'boolean' },
    dsparkCommitted: { type: 'boolean' }, prefillCommitted: { type: 'boolean' }, headCommit: { type: 'string' },
    losslessConfirmed: { type: 'boolean', description: 'both features re-verified lossless from a clean build' },
    perfSummary: { type: 'string', description: 'per-user decode tok/s + TTFT vs concurrency, DSpark+prefill vs the pre-stage baseline' },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['severity','title','detail'],
      properties: { severity:{type:'string',enum:['critical','high','medium','low']}, title:{type:'string'}, detail:{type:'string'} } } },
    summary: { type: 'string' },
  },
}
const verify = await agent(COMMON +
  '\n\nDSpark stage:\n' + JSON.stringify(dspark) + '\n\nPrefill stage:\n' + JSON.stringify(prefill) +
  '\n\nADVERSARIAL FINAL VERIFY. From a clean build re-run EVERY gate: G1 build; G2 qwen3-parity 8/8 (report tokens); G3 Gemma byte-identity (a 2nd prompt too); G4 go tests. RE-PROVE LOSSLESS for BOTH features yourself (DSpark spec-on==off; chunked==single-shot; prefix-cache-hit==cold). Confirm both stages committed (git log --oneline -4; head sha). Bench: per-user decode tok/s + TTFT at concurrency 1/8/16, DSpark+chunked-prefill vs the baseline before this workflow — state the high-concurrency win plainly (this is the vLLM-gap close). Check the diffs for any leftover hot-path instrumentation. Report allGreen + findings + head commit + perfSummary.',
  { label: 'verify:final', phase: 'Verify', schema: VERDICT, effort: 'high' })

return { dspark, prefill, verify }
