export const meta = {
  name: 'gemma4-decode-overhead',
  description: 'Profile gemma4 multiseq single-token decode (the Qwen3 serving path), then cut the per-token OVERHEAD that keeps decode at ~48% of peak BW — beat vLLM single-stream (39 tok/s), keep numerics byte-identical',
  phases: [
    { title: 'Profile' },
    { title: 'Optimize' },
    { title: 'Verify' },
  ],
}

const REPO = '/home/mauromedda/hack/fucina'
const QGGUF = '/opt/spark/models/Qwen3-8B-abliterated.Q4_K_M.gguf'
const GEMMA = '/opt/spark/models/gemma-4-12b-it-qat-q4_0.gguf'
const LLAMA = '/usr/local/llama/bin/llama-bench'

const COMMON = [
  'Repo: ' + REPO + ' (Go+CUDA, GB10 sm_121, CUDA 13). TARGET ENGINE = the gemma4 DENSE engine in cuda/gemma4_kernels.cu (9480 lines) — it is the SERVING engine and also runs Qwen3 dense via the arch descriptor (cuda/model_arch.h). The hot serving decode path is gemma4_engine_step_batch(eng, slots, inputs, B, out) = the MULTISEQ path: B rows, ONE token/row, sampled ON-DEVICE (sample_logits_ms_kernel), token ids kept on device (d_ms_outtok), the whole B-row forward CAPTURED into a per-B CUDA graph (multiseq_graph[B], commit lineage). Go calls it via internal/engine/cuda/bridge.go Engine.StepBatch (line 618): per step it allocs a C.int slot array + a B-int32 out slice, ONE cgo call, reads B tokens back. The continuous-batching scheduler (internal/server/batch/scheduler.go) drives this once per decode tick.',
  'THE MEASURED PROBLEM (this is what you are fixing): single-stream Qwen3-8B-Q4K decode = ~26 tok/s = 26 tok/s x ~5.0GB weights = ~131 GB/s = only ~48% of GB10 ~273 GB/s peak. fucina reads 3x LESS memory than vLLM fp16 (16GB) yet is SLOWER (26 vs vLLM 39 tok/s single-stream). A Q4 model on a bandwidth-bound decode should CRUSH fp16 — it does not, so decode is OVERHEAD-bound, NOT GEMV-bound. The per-token fixed overhead (kernel launches, sampler structural sync, token D2H readback, Go scheduler tick + per-step allocs, cgo boundary, any residual cudaStreamSynchronize / pipeline drain) is eating ~half the step. Batch decode also saturates at B~8 (~80 tok/s) vs vLLM ~200 — the per-step fixed cost does not amortize. The codebase ALREADY has: per-B graph capture, on-device sampling, deferred/lazy decode timing (decode_timing_lap), deferred D2H. So the residual overhead is SUBTLE — you must MEASURE it, not guess.',
  'ORACLES: (1) llama.cpp is the kernel oracle — ' + LLAMA + ' on the SAME ' + QGGUF + ' gives a clean decode tok/s (run: ' + LLAMA + ' -m ' + QGGUF + ' -n 128 -r 3 -p 0; pp=prefill, tg=decode). fucina is reported to already BEAT llama.cpp on this model, so use llama.cpp to find blocks where fucina spends time it does not. (2) vLLM single-stream = 39 tok/s is the bar to beat. (3) GB10 peak BW ~273 GB/s.',
  'GROUND-TRUTH GATES — EVERY optimize stage must pass ALL, else REVERT that change (git checkout the touched hunks):\n  (G1) BUILD: `make lib && make fucina` clean.\n  (G2) QWEN3 PARITY (the gemma4-engine numeric gate on the path you are editing): `make qwen3-parity-test` must print the 8/8 greedy match vs llama.cpp ([12095,13,576,6722,315,15344,374,21718]). ANY mismatch = numerics broken = revert.\n  (G3) GEMMA GOLDEN (the Gemma-arch bit-identity gate): `FUCINA_BATCH_SELFTEST=1 ./fucina -m ' + GEMMA + ' --prompt x --predict 0 --temp 0 2>&1 | grep "GOLDEN token-hash"` must EQUAL the baseline hash captured in the Profile phase (expected 0xf6961c714fa5c0f2). Different hash = numerics broken = revert. (Discover exact fucina flags with `./fucina --help` if --predict 0 is rejected; the self-test fires at engine init regardless of generation.)\n  (G4) SPEC still works: `make qwen3-spec-test` passes.\n  (G5) GO TESTS: `make go-test` green.\n  (G6) SPEED: re-measure decode tok/s (B=1 AND B=8) with the SAME harness/model/ctx, graph ON; only KEEP a change that does NOT regress and ideally improves. Report exact tok/s before/after.',
  'HARD CONSTRAINT FROM THE USER (verbatim: "togli tutti i potenziali loc e se avere observability rallenta togli"): the COMMITTED hot path must carry ZERO new observability/instrumentation. Any cudaEvent / printf / counter / timing you add to MEASURE must be removed (or compiled out behind a profile-only #ifdef that is OFF by default) before you run the gates and report. Do NOT leave logging or metrics in the decode step. Fewer lines of code is better.',
  'Do NOT change model math/results — only fuse/reorganize kernels, cut launches, remove syncs/round-trips, reduce Go/cgo per-step cost. Keep the per-B graph capture working (re-verify it still engages, not silent eager fallback). Build the test bench by compiling cuda/test_qwen3_bench.cu like the parity target: `nvcc -O3 -arch=sm_121 -std=c++17 -Icuda cuda/test_qwen3_bench.cu cuda/libfucina.a cuda/libdg.a -o /tmp/qbench -lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm` then `/tmp/qbench ' + QGGUF + '` (it benches B=1 single-token + B-row via gemma4_engine_step_batch = the exact serving path). NO server is running; if you launch ./fucina use a non-default port. Work in the repo tree (no worktree). Serial edits to gemma4_kernels.cu only — never two writers at once.'
].join('\n')

phase('Profile')
const PROFILE = {
  type: 'object', additionalProperties: false,
  required: ['baselineGoldenHash', 'qwen3ParityPass', 'tokS_B1', 'tokS_B8', 'tokS_llamaCpp', 'graphBenefitPct', 'breakdown', 'topTargets', 'method', 'notes'],
  properties: {
    baselineGoldenHash: { type: 'string', description: 'the FUCINA_BATCH_SELFTEST GOLDEN token-hash on the UNMODIFIED tree — the G3 reference for all later stages' },
    qwen3ParityPass: { type: 'boolean', description: 'did `make qwen3-parity-test` pass 8/8 on the unmodified tree (baseline G2)' },
    tokS_B1: { type: 'number', description: 'baseline single-stream (B=1) decode tok/s, graph ON, via /tmp/qbench' },
    tokS_B8: { type: 'number', description: 'baseline B=8 aggregate decode tok/s, graph ON' },
    tokS_llamaCpp: { type: 'number', description: 'llama-bench tg (decode) tok/s on the same model (0 if unavailable)' },
    graphBenefitPct: { type: 'number', description: 'tok/s delta graph ON vs FUCINA_NO_BATCHED_GRAPH/FUCINA_NO_DECODE_GRAPH — how much the graph already buys; small remaining gap means overhead is OUTSIDE the captured region (Go/cgo/sampler-sync/D2H)' },
    breakdown: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['block','usPerToken','pctOfToken','kind','location'],
      properties: { block:{type:'string',description:'e.g. forward-GEMV / attention / sampler-sync / token-D2H-readback / cgo+Go-tick / kernel-launch-gaps / logits-head'},
        usPerToken:{type:'number'}, pctOfToken:{type:'number'}, kind:{type:'string',enum:['memory','compute','launch','sync','host','mixed']},
        location:{type:'string',description:'file:line or function where this cost lives'} } },
      description: 'per-token decode time split, summing ~to the B=1 step time; SEPARATE the in-graph forward from the OUT-of-graph host/sync/readback overhead — that split is the whole point' },
    topTargets: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['target','whyExpensive','fixIdea','estGainPct','risk'],
      properties: { target:{type:'string'}, whyExpensive:{type:'string'}, fixIdea:{type:'string'}, estGainPct:{type:'number'}, risk:{type:'string',enum:['low','medium','high']} } },
      description: 'ranked highest-leverage OVERHEAD-cutting targets (NOT GEMV micro-opt unless it dominates). e.g. remove a per-step sync, batch the token readback, drop Go per-step allocs, fuse tail kernels into the graph, cut a stream barrier' },
    method: { type: 'string' },
    notes: { type: 'string' },
  },
}
const profile = await agent(COMMON + '\n\nYOUR TASK (PROFILE — do NOT optimize yet). On the UNMODIFIED tree: (1) `make lib && make fucina`. (2) Capture the BASELINE G2/G3 references: run `make qwen3-parity-test` (record pass + the 8 tokens) and capture the FUCINA_BATCH_SELFTEST GOLDEN token-hash (G3 reference). (3) Build /tmp/qbench and record baseline decode tok/s at B=1 and B=8 (graph ON). (4) Run llama-bench on the same model for the decode-tok/s oracle. (5) A/B the graph: re-run /tmp/qbench with FUCINA_NO_BATCHED_GRAPH=1 (and FUCINA_NO_DECODE_GRAPH=1) and report the tok/s delta — this tells you how much overhead is ALREADY inside the captured region vs OUTSIDE it (Go scheduler tick, cgo call, per-step C.int/out allocs in bridge.go StepBatch, sampler structural sync, the B-token D2H readback that ends each step). (6) PROFILE where a B=1 step spends time: use cudaEvent timing around the multiseq forward vs the sampler vs the readback (REMOVE this instrumentation after measuring — do not commit it), and/or nsys (`nsys profile -t cuda -o /tmp/qprof /tmp/qbench ' + QGGUF + '` then nsys stats) to see kernel-launch gaps (host-side bubbles between kernels = launch/sync overhead) and the gap between graph replays. Also read gemma4_engine_step_batch / decode_multiseq_forward / the multiseq sampler + readback in cuda/gemma4_kernels.cu and StepBatch in bridge.go to locate every host-side sync and per-step allocation. Produce: baseline hashes/tok/s, the graph-benefit split, a quantitative per-token breakdown that SEPARATES in-graph forward from out-of-graph host/sync/readback overhead, and a RANKED list of overhead-cutting targets (low-risk first). This drives the optimize phase.', { label: 'profile:decode', phase: 'Profile', schema: PROFILE, effort: 'high' })

phase('Optimize')
const RESULT = {
  type: 'object', additionalProperties: false,
  required: ['status','target','whatChanged','filesTouched','buildOk','qwen3Parity','goldenHashMatch','specOk','goTestOk','noObservabilityLeft','tokS_B1_before','tokS_B1_after','tokS_B8_before','tokS_B8_after','notes'],
  properties: {
    status: { type: 'string', enum: ['done','partial','reverted','blocked'] },
    target: { type: 'string' },
    whatChanged: { type: 'array', items: { type: 'string' } },
    filesTouched: { type: 'array', items: { type: 'string' } },
    buildOk: { type: 'boolean' },
    qwen3Parity: { type: 'string', description: 'G2 result: 8/8 match? the 8 tokens' },
    goldenHashMatch: { type: 'boolean', description: 'G3: golden hash == baseline?' },
    specOk: { type: 'boolean', description: 'G4: qwen3-spec-test passed' },
    goTestOk: { type: 'boolean', description: 'G5: make go-test green' },
    noObservabilityLeft: { type: 'boolean', description: 'confirmed NO new instrumentation/printf/counter left in the committed hot path' },
    tokS_B1_before: { type: 'number' }, tokS_B1_after: { type: 'number' },
    tokS_B8_before: { type: 'number' }, tokS_B8_after: { type: 'number' },
    notes: { type: 'string' },
  },
}
const targets = (profile.topTargets || []).slice(0, 3)
const done = []
for (let i = 0; i < targets.length; i++) {
  const t = targets[i]
  const r = await agent(COMMON + '\n\nPROFILE (ground truth):\n' + JSON.stringify(profile) +
    '\n\nPRIOR OPTIMIZE STAGES (already applied to the tree):\n' + JSON.stringify(done) +
    '\n\nYOUR TASK (OPTIMIZE TARGET ' + (i+1) + '/' + targets.length + '): ' + t.target +
    '\nWhy expensive: ' + t.whyExpensive + '\nFix idea: ' + t.fixIdea + '\nRisk: ' + t.risk +
    '\nImplement THIS ONE overhead-cutting change (in cuda/gemma4_kernels.cu and/or internal/engine/cuda/bridge.go and/or internal/server/batch/scheduler.go as the target dictates). Do NOT alter numerics. Then run ALL gates in order: G1 build, G2 `make qwen3-parity-test` (8/8), G3 golden hash == baseline (' + (profile.baselineGoldenHash || '0xf6961c714fa5c0f2') + '), G4 `make qwen3-spec-test`, G5 `make go-test`, G6 re-bench /tmp/qbench B=1 and B=8 (graph ON) vs the before numbers. CONSTRAINT: leave ZERO new observability in the committed hot path (remove any timing/printf you used). If ANY of G2/G3/G4/G5 fails, or G6 REGRESSES tok/s, REVERT your change (git checkout the touched files/hunks) and report status=reverted with the failing gate. Report exact before/after tok/s and every gate result.',
    { label: 'opt:' + (t.target || ('t'+i)).slice(0,22), phase: 'Optimize', schema: RESULT, effort: 'high' })
  done.push(r)
}

phase('Verify')
const VERDICT = {
  type: 'object', additionalProperties: false,
  required: ['allGatesPass','qwen3Parity','goldenHashMatch','specOk','goTestOk','noObservabilityLeft','tokS_B1_baseline','tokS_B1_final','tokS_B8_baseline','tokS_B8_final','tokS_vLLM','tokS_llamaCpp','beatsVLLMSingleStream','speedupB1','findings','recommendation'],
  properties: {
    allGatesPass: { type: 'boolean' },
    qwen3Parity: { type: 'string' },
    goldenHashMatch: { type: 'boolean' },
    specOk: { type: 'boolean' },
    goTestOk: { type: 'boolean' },
    noObservabilityLeft: { type: 'boolean', description: 'grep the diff: no leftover instrumentation in the committed hot path' },
    tokS_B1_baseline: { type: 'number' }, tokS_B1_final: { type: 'number' },
    tokS_B8_baseline: { type: 'number' }, tokS_B8_final: { type: 'number' },
    tokS_vLLM: { type: 'number', description: 'the 39 tok/s single-stream bar (carry over)' },
    tokS_llamaCpp: { type: 'number' },
    beatsVLLMSingleStream: { type: 'boolean' },
    speedupB1: { type: 'number' },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['severity','title','detail'],
      properties: { severity:{type:'string',enum:['critical','high','medium','low']}, title:{type:'string'}, detail:{type:'string'} } } },
    recommendation: { type: 'string' },
  },
}
const review = await agent(COMMON + '\n\nAll optimize stages:\n' + JSON.stringify(done) +
  '\n\nADVERSARIAL VERIFY + BENCH. Re-run EVERY gate yourself from a clean build: G1 `make lib && make fucina`; G2 `make qwen3-parity-test` (report the 8 tokens); G3 golden hash == ' + (profile.baselineGoldenHash || 'baseline') + '; G4 `make qwen3-spec-test`; G5 `make go-test`. Then `git diff --stat` + read the decode hot-path diff and CONFIRM no new observability/printf/counter/timing was left committed (the user forbade it). Hunt for: (a) a change that silently shifted numerics but the parity prompt did not catch — run a SECOND, different greedy prompt through ./fucina before/after if any kernel math moved; (b) the per-B graph silently falling back to eager (check for a capture-failed path / verify the graph still engages); (c) correctness holding only at short ctx — bench/decode at a longer ctx (e.g. 2k+) too; (d) a B=1 win that REGRESSED B=8 or vice-versa. Then bench decode tok/s (B=1 and B=8, graph ON) and compute the net speedup vs the Profile baseline. State plainly: does fucina now BEAT vLLM single-stream (39 tok/s)? remaining gap + the next overhead target? Verdict: net speedup, still byte-identical (G2+G3), gates all green, hot path clean of observability.', { label: 'verify:decode', phase: 'Verify', schema: VERDICT, effort: 'high' })

return { profile, optimizations: done, review }
