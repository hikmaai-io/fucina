export const meta = {
  name: 'e4b-decode-overhead',
  description: 'Profile E4B single-token decode, then fuse the dominant overhead to close the 3-5x gap to the bandwidth ceiling — keep decode byte-identical',
  phases: [
    { title: 'Profile' },
    { title: 'Optimize' },
    { title: 'Verify' },
  ],
}

const REPO = '/home/mauromedda/hack/fucina'
const GGUF = '/opt/spark/models/hub/models--google--gemma-4-E4B-it-qat-q4_0-gguf/snapshots/bb3b92e6f031fa438b409f898dd9f14f499a0cb0/gemma-4-E4B_q4_0-it.gguf'

const COMMON = [
  'Repo: ' + REPO + ' (Go+CUDA Gemma-4 E4B, GB10 sm_121, CUDA 13). HARD TARGET: llama.cpp (llama-cli) runs this SAME E4B Q4_0 on this SAME GB10 at ~67.3 tok/s decode (llama-bench, clean; plain decode, NO speculation). That is the bar — fucina E4B decode must MATCH or BEAT ~67 tok/s. It proves the gap is fucina kernel inefficiency, NOT a hardware/PLE limit. FACT: E4B single-token decode (e4b_step T==1) is ALREADY CUDA-graph-captured (eng->dec graph, commit 02a7019), yet fucina decode measured anywhere from ~22 tok/s (spec-test baseline) to ~53 tok/s (short-ctx server) — the wide spread is itself a clue (ctx length? graph not engaged in some harness? prefill folded into timing?) and MUST be explained by the profile. fucina baseline is ~50 tok/s = ~26% slower than llama.cpp. The gap to 67 is KERNEL INEFFICIENCY inside the graph: many tiny memory-round-trip kernels (E4B has 5 layernorms/layer + q/k/v head-norms + residual adds + GeGLU + small GEMVs), the Per-Layer-Embedding (PLE) per-layer work (plm_proj + ple_in_gate/ple_proj GEMVs + gate + extra norm), attention KV reads (grow with n_past), AND possibly GEMVs slower than llama.cpp tuned MMVQ. Use llama.cpp as the oracle: where fucina spends time that llama.cpp does not is the target.',
  'GROUND-TRUTH GATES (every optimize stage): (1) CORRECTNESS — single-token decode must stay byte-identical. Build a decode-equivalence check: a fixed prompt greedily decoded N>=128 tokens BEFORE vs AFTER your change must produce identical token ids (the existing make e4b-gen-test / e4b-spec-test compare greedy output; reuse them, and/or a small harness diffing logits argmax). Greedy decode is deterministic. (2) SPEED — measure tok/s before/after with the SAME prompt+ctx; only keep a change that does not regress and ideally improves. Report exact tok/s.',
  'Do NOT change model math/results — only fuse/reorganize kernels for fewer global-memory round-trips and better occupancy. Keep the MTP/drafter path (just optimized by a prior workflow) working: re-run make e4b-spec-test too. Build: make lib then the test. DO NOT disturb the server on port 8080; use another port if launching. Work in the repo tree (no worktree).'
].join('\n')

phase('Profile')
const PROFILE = {
  type: 'object', additionalProperties: false,
  required: ['breakdown', 'topTargets', 'totalPerTokenUs', 'method', 'notes'],
  properties: {
    method: { type: 'string', description: 'how measured (cuda events around blocks / nsys / kernel timeline)' },
    totalPerTokenUs: { type: 'number', description: 'measured per-token decode time in microseconds' },
    breakdown: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['block','usPerToken','pctOfToken','kind'],
      properties: { block:{type:'string',description:'e.g. attention / PLE / FFN-GEMV / layernorms / head / elementwise'},
        usPerToken:{type:'number'}, pctOfToken:{type:'number'}, kind:{type:'string',enum:['memory','compute','launch','mixed']} } },
      description: 'per-token time split across decode blocks, summing ~to totalPerTokenUs' },
    topTargets: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['target','whyExpensive','fusionIdea','estGainPct'],
      properties: { target:{type:'string'}, whyExpensive:{type:'string'}, fusionIdea:{type:'string'}, estGainPct:{type:'number'} } },
      description: 'ranked, highest-leverage fusion/optimization opportunities' },
    notes: { type: 'string' },
  },
}
const profile = await agent(COMMON + '\n\nYOUR TASK (PROFILE — do NOT optimize yet). Measure where E4B single-token decode time goes, per token, on this box. Instrument e4b_step T==1 (or use a focused harness) with cudaEvent timing around the major blocks PER LAYER and globally: (a) embed+PLE lookup/combine, (b) per-layer attention (rope+kv_store+attn_flash_decode), (c) per-layer Q/K/V/O + FFN GEMVs, (d) per-layer the 5 layernorms + q/k/v head-norms + residual adds + GeGLU (the small elementwise/norm kernels), (e) per-layer PLE (ple_in_gate/ple_proj/gate/post_ple_norm), (f) the tied head. Run a real decode (prefill a ~256-tok prompt with ' + GGUF + ', decode 64 tokens, average). Since the decode is graph-captured, to time sub-blocks you may need FUCINA_E4B_NOGRAPH=1 (note the graph-off total will be higher; report BOTH graph-on total tok/s and the graph-off per-block split, and reason about which blocks stay dominant under the graph). Optionally use nsys/compute-sanitizer --tool=... or cudaEventElapsedTime. ALSO establish the llama.cpp ORACLE: locate llama-cli on this box (which llama-cli; or find / -name llama-cli 2>/dev/null; common under ~/llama.cpp/build/bin or /usr/local/bin), run it on the SAME ' + GGUF + ' with the SAME ctx/prompt, and record its decode tok/s (target ~67.3; use llama-bench -n 128 -r 3 for a clean number, NOT a contended llama-cli run) — if available, nsys it too and DIFF its kernel timeline vs fucina (llama.cpp fuses rmsnorm/residual, uses tuned MMVQ; identify which fucina blocks have no cheap llama.cpp counterpart = the gap). If llama-cli is not found, note it and proceed with the bandwidth-ceiling reasoning. Produce the per-token breakdown (us + % per block), the fucina-vs-llama.cpp tok/s + (if profiled) per-block gap, and a RANKED list of the highest-leverage fusion targets (e.g. fuse rmsnorm+residual+scale; fuse GeGLU into the down-proj epilogue; batch the per-layer norms; cut PLE round-trips; faster Q4_0 GEMV). Be quantitative; this drives the optimize phase.', { label: 'profile:decode', phase: 'Profile', schema: PROFILE, effort: 'high' })

phase('Optimize')
const RESULT = {
  type: 'object', additionalProperties: false,
  required: ['status','target','whatChanged','buildOk','correctness','tokSBefore','tokSAfter','notes'],
  properties: {
    status: { type: 'string', enum: ['done','partial','reverted','blocked'] },
    target: { type: 'string' },
    whatChanged: { type: 'array', items: { type: 'string' } },
    buildOk: { type: 'boolean' },
    correctness: { type: 'string', description: 'decode byte-identical? (the gate result)' },
    tokSBefore: { type: 'number' }, tokSAfter: { type: 'number' },
    notes: { type: 'string' },
  },
}
// Take the top 3 ranked targets and implement them sequentially (same file → must be serial),
// each gated; revert any change that breaks byte-identity or regresses speed.
const targets = (profile.topTargets || []).slice(0, 3)
let prev = { profile }
const done = []
for (let i = 0; i < targets.length; i++) {
  const t = targets[i]
  const r = await agent(COMMON + '\n\nPROFILE (ground truth):\n' + JSON.stringify(profile) +
    '\n\nPRIOR OPTIMIZE STAGES:\n' + JSON.stringify(done) +
    '\n\nYOUR TASK (OPTIMIZE TARGET ' + (i+1) + '/' + targets.length + '): ' + t.target +
    '\nWhy it is expensive: ' + t.whyExpensive + '\nFusion idea: ' + t.fusionIdea +
    '\nImplement this ONE optimization in cuda/e4b_engine.cu (fuse kernels / reduce global-memory round-trips / improve occupancy; do NOT alter results). After it, RE-CAPTURE the decode graph still works (graph-on). GATES: build (make lib); decode byte-identical (run make e4b-gen-test or a greedy-equivalence harness vs a saved baseline — identical token ids) AND make e4b-spec-test still byte-identical; measure tok/s before/after (same prompt+ctx, graph ON). If the change breaks byte-identity OR regresses tok/s, REVERT it (git checkout the file hunks) and report status=reverted with why. Report exact tok/s before/after.',
    { label: 'opt:' + (t.target || ('t'+i)).slice(0,24), phase: 'Optimize', schema: RESULT, effort: 'high' })
  done.push(r); prev = r
}

phase('Verify')
const VERDICT = {
  type: 'object', additionalProperties: false,
  required: ['gatePassed','tokSBaseline','tokSFinal','tokSLlamaCpp','speedup','findings','recommendation'],
  properties: {
    gatePassed: { type: 'boolean', description: 'decode + spec still byte-identical (re-run yourself)' },
    tokSBaseline: { type: 'number', description: 'decode tok/s before this workflow' },
    tokSFinal: { type: 'number', description: 'decode tok/s after all kept optimizations' },
    tokSLlamaCpp: { type: 'number', description: 'llama-cli decode tok/s on the same model/ctx (the ~67.3 target; 0 if not found)' },
    speedup: { type: 'number' },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['severity','title','detail'],
      properties: { severity:{type:'string',enum:['critical','high','medium','low']}, title:{type:'string'}, detail:{type:'string'} } } },
    recommendation: { type: 'string' },
  },
}
const review = await agent(COMMON + '\n\nAll optimize stages:\n' + JSON.stringify(done) +
  '\n\nADVERSARIAL VERIFY + BENCH. Re-run the gates yourself: build (make lib), greedy decode equivalence (make e4b-gen-test) AND make e4b-spec-test — BOTH must be byte-identical (report the exact lines). Then bench E4B decode tok/s (graph ON, fixed prompt+ctx) and compute speedup vs the pre-workflow baseline. ALSO run llama-cli on the same model/ctx and report tokSLlamaCpp (the ~67.3 bar, via llama-bench): state plainly whether fucina now MATCHES/BEATS llama.cpp, and if not, the remaining gap + the next target. Hunt for: a fusion that silently changed numerics but the test prompt did not catch (try a 2nd, different prompt); broken graph capture (fell back to eager — check FUCINA_E4B_NOGRAPH path / a capture-failed warning); occupancy regressions at long ctx; correctness only holding at short ctx. Verdict: net decode speedup, vs the ~67.3 tok/s llama.cpp bar, AND still lossless?', { label: 'verify:decode', phase: 'Verify', schema: VERDICT, effort: 'high' })

return { profile, optimizations: done, review }
