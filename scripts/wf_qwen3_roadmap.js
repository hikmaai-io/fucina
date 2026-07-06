export const meta = {
  name: 'qwen3-roadmap-rest',
  description: 'Finish the Qwen3 roadmap on tau: fast Q4_K/Q6_K prefill, then DSpark spec wiring — serial, each gated (parity 8/8 + Gemma byte-identical) and committed',
  phases: [
    { title: 'Prefill' },
    { title: 'Spec' },
    { title: 'Verify' },
  ],
}

const REPO = '/home/mauromedda/hack/fucina-tau'
const QWEN = '/opt/spark/models/Qwen3-8B-abliterated.Q4_K_M.gguf'
const GEMMA = '/opt/spark/models/gemma-4-12b-it-qat-q4_0.gguf'

const COMMON = [
  'Worktree: ' + REPO + ' (branch feat/dense-31b-tau), GB10 sm_121a, CUDA 13. The gemma4 engine now serves Qwen3-8B-Q4_K_M correctly (parity 8/8 vs llama.cpp) AND the packed-Q4_K decode optimization just landed. cuda/gemma4_kernels.cu (~12200 lines) is the engine; cuda/gemma4_detect.h arch detect; cuda/mmvq* the GEMV kernels; internal/engine/cuda/bridge.go the cgo bridge; internal/server/batch/ the continuous-batching scheduler.',
  'BUILD env: `export CUT=/home/mauromedda/.venv/lib/python3.12/site-packages/flashinfer/data/cutlass`; nvcc=/usr/local/cuda-13/bin/nvcc -arch=sm_121a. Build: `make lib CUTLASS_DIR=$CUT && make fucina CUTLASS_DIR=$CUT` (libdg.a present).',
  'HARD GATES every stage (run ALL; never claim success on a failing gate):',
  '  G1 BUILD clean.',
  '  G2 QWEN3 PARITY 8/8: `make qwen3-parity-test CUTLASS_DIR=$CUT` must print 8/8 match, tokens [12095,13,576,6722,315,15344,374,21718].',
  '  G3 GEMMA byte-identity: `./fucina -m ' + GEMMA + ' --prompt "The capital of France is" --predict 16 --temp 0` must equal the pre-change output (git-stash A/B if unsure). Gemma is arch==GEMMA4 → must be untouched.',
  '  G4 GO TESTS: `cd ' + REPO + ' && go test ./internal/server/... ./internal/server/batch/ ./internal/tokenizer/ ./internal/sampler/ 2>&1 | tail` green.',
  'CONSTRAINTS: every arch/feature change gated so Gemma + the existing Qwen3 decode stay correct. NO committed hot-path observability/logging (user directive: keep the decode/prefill hot path free of instrumentation; any debug print removed before gates). Keep edits minimal + in-idiom. Do NOT touch NVFP4 path.',
  'COMMIT POLICY: each stage, after ALL its gates pass, COMMIT its work on feat/dense-31b-tau (so progress is never lost — the whole reason this engine was rebuilt is that uncommitted work was lost). Commit message footer line exactly: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. If a gate fails and you cannot fix it, do NOT commit; report status=blocked with the exact failing output.'
].join('\n')

phase('Prefill')
const PREFILL = {
  type: 'object', additionalProperties: false,
  required: ['status','committed','commitSha','whatChanged','buildOk','parity8of8','gemmaIdentical','ttftBeforeMs','ttftAfterMs','notes'],
  properties: {
    status: { type: 'string', enum: ['done','partial','blocked'] },
    committed: { type: 'boolean' }, commitSha: { type: 'string' },
    whatChanged: { type: 'array', items: { type: 'string' } },
    buildOk: { type: 'boolean' }, parity8of8: { type: 'boolean' }, gemmaIdentical: { type: 'boolean' },
    ttftBeforeMs: { type: 'number', description: 'prefill time for a ~256-tok prompt BEFORE (token-by-token path)' },
    ttftAfterMs: { type: 'number', description: 'prefill time AFTER (fast GEMM path)' },
    notes: { type: 'string' },
  },
}
const prefill = await agent(COMMON +
  '\n\nYOUR TASK (ROADMAP ITEM: FAST QWEN3 PREFILL). Today Qwen3 prefill falls back to the token-by-token decode path because the BF16 fast-prefill GEMM dequantizes weights via a helper (grep `decode_weight` / `dequant_*_to_bf16` / `paged_prefill_batched` returning -2 for Qwen3) that only knows Q4_0/Q8_0 — NOT the Q4_K/Q6_K bulk weights of a Qwen3 Q4_K_M checkpoint, and the prefill attention is Gemma head_dim. Implement native Q4_K + Q6_K dequant-to-BF16 (mirror the existing `dequant_q4_0_to_bf16_kernel`; the Q4_K superblock=144B [half d; half dmin; u8 scales[12]; u8 qs[128]], value = d*sc_j*q_i - dmin*m_j with the same scale/min unpack the decode kernel uses; Q6_K=210B superblock like the existing mmvq_q6_k decode) and wire the Qwen3 prefill to use the fast GEMM path with per-tensor format + head_dim=128 attention (the decode path already has <32,8,128>; reuse it for prefill attention). Gate G1-G4 PLUS: measure prefill/TTFT for a ~256-token prompt before vs after (the test_qwen3_bench or a small harness prints prefill time; or time the first-token latency) — it must IMPROVE (fast GEMM beats token-by-token) and parity MUST stay 8/8 (the dequant must be numerically faithful — greedy tokens unchanged). If you cannot reach faithful dequant (parity breaks), REVERT and report blocked. On green, COMMIT.',
  { label: 'prefill:q4k', phase: 'Prefill', schema: PREFILL, effort: 'high' })

phase('Spec')
const SPEC = {
  type: 'object', additionalProperties: false,
  required: ['status','committed','commitSha','whatChanged','buildOk','parity8of8','gemmaIdentical','goTestOk','specAcceptsDrafts','lossless','notes'],
  properties: {
    status: { type: 'string', enum: ['done','partial','blocked'] },
    committed: { type: 'boolean' }, commitSha: { type: 'string' },
    whatChanged: { type: 'array', items: { type: 'string' } },
    buildOk: { type: 'boolean' }, parity8of8: { type: 'boolean' }, gemmaIdentical: { type: 'boolean' }, goTestOk: { type: 'boolean' },
    specAcceptsDrafts: { type: 'boolean', description: 'a spec smoke shows multi-token accepted runs on a repetitive prompt' },
    lossless: { type: 'boolean', description: 'spec output == non-spec greedy output (lossless verify)' },
    notes: { type: 'string' },
  },
}
const spec = await agent(COMMON +
  '\n\nPRIOR STAGE (prefill) result:\n' + JSON.stringify(prefill) +
  '\n\nYOUR TASK (ROADMAP ITEM: DSPARK SPEC WIRING into the continuous-batching scheduler). tau already has a COMMITTED per-slot C ABI `gemma4_engine_step_batch_spec(eng, const int*slots, const int32_t*in_tokens, int B, int32_t*out_tokens, int*out_lens)` and `Engine.StepBatchSpec(slots, inputs)` in internal/engine/cuda/bridge.go. FIRST map that ABI exactly (read its impl in cuda/gemma4_kernels.cu + the bridge): how it takes drafts, how out_tokens/out_lens encode the accepted run per slot, the draft-length limit. A prompt-lookup drafter + scheduler integration exist UNCOMMITTED on a different worktree (/home/mauromedda/hack/fucina, branch feat/e4b-gguf) as internal/server/batch/drafter.go + scheduler.stepSpec, but they target a DIFFERENT (ragged) ABI — use drafter.go as a REFERENCE for the prompt-lookup algorithm (n-gram suffix match, strict-majority consensus) but ADAPT the scheduler integration to tau\'s per-slot ABI. Implement: a model-agnostic prompt-lookup drafter (internal/server/batch/drafter.go), seq history tracking, and a scheduler spec path that detects the SpecBatchEngine, drafts per slot, calls StepBatchSpec, and commits the accepted runs — default-on, LOSSLESS (greedy: accept draft iff it equals the target argmax; the C verify already enforces this). Add Go unit tests (drafter + scheduler spec path). Gate G1-G4 PLUS: (a) go tests green incl new ones; (b) a spec smoke (a small Go test or harness) shows multi-token accepted runs on a repetitive prompt; (c) LOSSLESS: spec-on vs spec-off greedy generation identical. parity G2 + Gemma G3 unaffected (no .cu numeric change expected — this is scheduler/bridge Go wiring on top of the existing C verify). On green, COMMIT.',
  { label: 'spec:wiring', phase: 'Spec', schema: SPEC, effort: 'high' })

phase('Verify')
const VERDICT = {
  type: 'object', additionalProperties: false,
  required: ['allGreen','parity8of8','gemmaIdentical','goTestOk','prefillCommitted','specCommitted','headCommit','decodeTokS','findings','summary'],
  properties: {
    allGreen: { type: 'boolean' },
    parity8of8: { type: 'boolean' }, gemmaIdentical: { type: 'boolean' }, goTestOk: { type: 'boolean' },
    prefillCommitted: { type: 'boolean' }, specCommitted: { type: 'boolean' }, headCommit: { type: 'string' },
    decodeTokS: { type: 'number' },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['severity','title','detail'],
      properties: { severity:{type:'string',enum:['critical','high','medium','low']}, title:{type:'string'}, detail:{type:'string'} } } },
    summary: { type: 'string' },
  },
}
const verify = await agent(COMMON +
  '\n\nPrefill stage:\n' + JSON.stringify(prefill) + '\n\nSpec stage:\n' + JSON.stringify(spec) +
  '\n\nADVERSARIAL FINAL VERIFY. From a clean build re-run EVERY gate yourself: G1 build; G2 `make qwen3-parity-test` (report the 8 tokens); G3 Gemma byte-identity (a second different prompt too, to catch silent drift); G4 go tests. Confirm both stages COMMITTED (git log --oneline -3; report the head sha). Re-bench Qwen3 decode tok/s. Check: did the prefill change alter decode numerics? did the spec wiring stay lossless (re-run spec-on vs off)? is the hot path free of leftover instrumentation (git show the diffs)? Report allGreen + any findings + the head commit.',
  { label: 'verify:roadmap', phase: 'Verify', schema: VERDICT, effort: 'high' })

return { prefill, spec, verify }
