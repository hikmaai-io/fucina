# S2 — DFlash graph substrate (MRV2 selective adoption)

Status: ACTIVE (2026-07-12). Branch `feat/s2-graph-substrate` from merged main
`06df4b5`. Implements the S2 section of `sota-gb10-qwen3-mission-plan.md` and the
B.1 MRV2 mapping of `vllm-subsystem-analysis-2026-07-12.md`.

## Mission

Build the *graph-replayable full decode step* that S1 (DFlash spec decode) needs,
by adopting exactly three MRV2 ideas — cheapest first — and rejecting all PCIe
machinery (StagedWriteTensor, UvaBufferPool, pinned-staging) which is moot on
GB10 unified memory.

- **S2b (S)** — CUDA-graph key scheme `(num_tokens, num_reqs, uniform_token_count)`
  with dominance dispatch + decode-first batch ordering.
- **S2a (M)** — GPU-native input splicing: derive next-step `input_ids`/positions
  from persistent slot state ON-GPU so the whole decode step is a replayable graph
  with no host knowledge of the sampled token.
- **S2c (S, conditional)** — permanent-slot request state ONLY IF the engine
  still compacts/reorders batch state on admit/exit.

Correctness invariants (non-negotiable): byte-identical run-to-run determinism;
`qwen35-multiseq-prefill-test` PASS with unchanged bounds; `protection_gate.py`
green both models; `make lib libdg fucina` green. Never leave the tree broken.

## Reconnaissance (findings before touching code)

### Existing decode graph cache (the S2b starting point)

`cuda/qwen35_runtime.cuh`:

- `qwen35_ms_graph_ensure(eng, B)` (:474) captures one FULL decode graph per row
  count `B` into `eng->q35.graph[B]` (`cudaGraphExec_t graph[GEMMA4_MAX_SEQS+1]`,
  `qwen35_state.cuh:113`). The capture wraps `qwen35_decode_multiseq_body(eng, B,
  want_argmax=1, cs)`.
- `qwen35_ms_run(...)` (:530) refreshes the per-step device inputs (`d_sb[0]`
  input tokens, `d_ms_pos` positions, `routing_workspace` row→slot) OUTSIDE the
  capture, then either replays `graph[B]` or falls back to per-kernel launches.
- **The key is a single integer `B`.** Every decode row today is exactly one
  token (`num_tokens == num_reqs == B`), and `uniform_token_count == 1`
  implicitly. There is no representation for the `(1+K)`-token spec-decode batch
  that S1/DFlash will introduce.

`GEMMA4_MAX_SEQS == GEMMA4_SPEC_MAX == 32` (`gemma4_kernels.cu:3516`,
`gemma4_kernels.cuh:50`).

### Host-side per-step input construction (the S2a target)

`qwen35_step_batch(eng, slots, in_tokens, B, out_tokens)`
(`qwen35_runtime.cuh:1728`):

1. HOST builds `in2[]` (= `in_tokens[r]`, the previous step's sampled token as
   handed back by the caller), `positions[]` (= `s->n_tokens`), `rowmap[]`.
2. `qwen35_ms_run` H2D-copies `in2`/`positions`/slot into device buffers.
3. Runs the forward, D2H-copies `out_tokens`, host advances
   `s->n_tokens`/`s->n_sampled`.

The sampled token round-trips HOST→device each step. That host dependency is what
prevents a zero-sync draft loop: replay currently needs the host to know the
previous token. **S2a removes it** by splicing the last sampled/accepted token
device-side from persistent slot state.

### S2c decision — SKIP (engine is already slot-stable)

Checked `q35_slot_state_ensure` (:914) and `gemma4_engine_seq_remove`: a request
holds a fixed slot index for its whole lifetime; `seq_remove` marks the slot free
but RETAINS its state (pooled — "churn never enters cudaMalloc/cudaFree",
:911-913). The scheduler passes explicit `slots[]` into `step_batch`; there is no
tensor-wide compaction or row reordering on admit/exit. Per-slot GDN/conv/KV
arenas are indexed by stable slot id. **fucina already has MRV2's permanent-row
persistent state. S2c is a no-op — nothing to do.**

## Plan (cheapest first, gated each step)

1. **S2b** — generalize the graph key to a small `q35_graph_key`
   `{num_tokens, num_reqs, uniform_token_count}` with dominance dispatch, replace
   the `graph[B]` array with a keyed cache, and add decode-first batch ordering
   for mixed steps. `uniform_token_count` defaults to 1 for pure decode (bitwise
   no-op today), but is now representable for the `(1+K)` spec-decode batch.
2. **S2a** — a GPU input-splice kernel that writes next-step `input_ids` from the
   persistent per-slot last-token state, so the token buffer no longer needs a
   host-authored value inside the replayable region.

Both must land green + gated before push.

## Decisions log

### S2b — keyed CUDA-graph cache (LANDED, `e503175`)

- New `cuda/qwen35_graph_key.cuh` (ABOUTME): pure, host-testable `q35_graph_key
  {num_tokens, num_reqs, uniform_token_count}`, `q35_make_decode_key`,
  `q35_make_spec_key`, `q35_graph_dominates` (dominance dispatch), and
  `q35_sort_batch_decode_first` (stable decode-first ordering, C-style, no STL).
- `qwen35_state.cuh`: replaced `cudaGraphExec_t graph[GEMMA4_MAX_SEQS+1]` with a
  linear-probed `q35_graph_entry graph_cache[Q35_GRAPH_CACHE_CAP=64]` +
  `graph_count`.
- `qwen35_runtime.cuh`: `qwen35_ms_graph_ensure` now takes a key, looks up by
  dominance, captures on miss, and `q35_graph_evict` compacts on replay failure.
  `qwen35_ms_run` builds `q35_make_decode_key(B)` for the plain-decode path.
- **Determinism**: plain decode maps `B → (B, B, 1)`, one distinct key per row
  count, so the captured body and dispatch are bitwise-identical to the old
  `graph[B]` scheme. `uniform_token_count` must match EXACTLY in dominance (it
  changes the per-request query layout), so a decode key never aliases a
  `(1+K)` spec key.
- **Gates**: host unit test `qwen35-graph-key-test` PASS; batch selftest
  `graph-on==off PASS`, `row-independence PASS`, `M3-parity 8/8` (captured log:
  `qwen35 M4 batch graph captured (nt=3 nr=3 utc=1)`); multiseq-prefill PASS with
  **unchanged** bounds (MoE ≤0.0946, dense ≤0.0029). `make lib libdg fucina`
  green.
- **Multi-agent note**: recovered from a cross-worktree stash collision (a
  sibling agent parked this WIP in a shared-object-store stash and a foreign
  `__launch_bounds__` edit leaked into the index). Reset to pristine HEAD,
  re-applied the S2b patch cleanly, verified the `launch_bounds` baseline intact,
  committed immediately.

### Why performance-neutral

The graph body captured is unchanged; only the host-side cache index changed
(array subscript → small linear probe over ≤64 entries, once per distinct shape,
then a pointer replay). No per-token host work added inside the hot loop.

### S2a — GPU-native input splicing (LANDED, `eb850fc`)

- New device kernels (`cuda/qwen35_kernels.cuh`, one thread/row, distinct slots):
  - `qwen35_splice_inputs_kernel` — gather `in_tok[r]=slot_tok[rowslot[r]]`,
    `pos[r]=slot_pos[rowslot[r]]`. Runs INSIDE the captured graph, at the top of
    `qwen35_decode_multiseq_body`, so replay derives inputs with no host token.
  - `qwen35_writeback_slot_state_kernel` — `slot_tok[slot]=out_tok[r]`,
    `slot_pos[slot]+=1`. In-graph on the greedy path (after argmax); post-sampler
    in `qwen35_ms_run` on the sampled path. Skips sentinel `out_tok<0` rows.
  - `qwen35_seed_slot_state_kernel` — scatter host inputs into slot state before
    a step (outside the graph). Makes the in-graph splice reproduce exactly the
    old host-copy `(tok,pos)`.
- `qwen35_state.cuh`: persistent `int32_t *d_slot_tok` / `int *d_slot_pos`
  (capacity-indexed) + `int gpu_splice_enabled` toggle
  (`FUCINA_QWEN35_NO_GPU_SPLICE=1` → legacy host-copy path). Allocated in
  `ensure_q35_scratch`, freed in `gemma4_engine_destroy`.
- `qwen35_runtime.cuh`: `qwen35_decode_multiseq_body` and `qwen35_ms_run` take a
  `splice` flag; `qwen35_ms_graph_ensure` captures WITH splice+writeback so the
  graph is a self-contained `splice(state)→forward→argmax→writeback(state)` step.
  `qwen35_step_batch` seeds slot state each step then runs spliced. The sequential
  prefill paths (`seq_add`, `seq_prefill_chunk`) pass `splice=0` (their
  per-position tokens are not in slot state).
- **Determinism**: with splice on, `step_batch` still computes the host
  `in2[]/positions[]` exactly as before and *seeds* slot state from them; the
  in-graph splice re-derives identical `(tok,pos)`. The writeback mutation is
  authoritatively overwritten by the next step's seed. Every step's inputs are
  therefore byte-identical to the host-copy path — proven by the multiseq-prefill
  gate holding its EXACT prior bounds.
- **Zero-sync primitive proven**: new self-chain gate seeds slot state ONCE, then
  replays the captured greedy graph 24× with NO host token feedback; the graph's
  in-body writeback→splice advances `(token, position)` device-side and the
  produced tokens byte-match the host-fed reference. This is the S1/DFlash
  zero-sync draft-loop prerequisite, validated.
- **Gates**: batch selftest `self-chain=PASS graph-on==off=PASS
  row-independence=PASS M3-parity 8/8 sampling=PASS`; multiseq-prefill PASS with
  **unchanged** bounds (MoE ≤0.0946, dense ≤0.0029); `make lib libdg fucina`
  green.
- **Performance**: neutral by construction — adds two int-only kernels over B≤32
  rows (a gather and a scatter), both captured in the same graph; the ~33 ms/step
  weight traffic is untouched. On the greedy graph path S2a REMOVES two per-step
  host→device copies (input tokens + positions), a small latency win.

## Protection gate

S2 changes no kernel math and adds only negligible int-shuffle work, so serving
throughput is expected neutral. The `scripts/protection_gate.py` sweep requires a
quiescent box (a contended GPU makes throughput numbers invalid). Status recorded
in the completion note below; determinism + parity gates (the correctness
contract) are fully green.

### Protection-gate run log (2026-07-12)

- Attempt 1 (dense 9B, `s2-a134cbf`): **INVALID — multi-agent GPU contention.**
  During the ordered 1→32 sweep a sibling agent's 12–22 GB job was resident on the
  shared 273 GB/s GPU; server telemetry showed `engine 74.7 ms/step` at avgB=2
  (vs the ~30 ms floor). Result: N=2/4/8 throughput ~halved then *recovered* to
  ABOVE baseline by N=32 as contention eased (N=32 336 vs base 261; TTFT better in
  every cell). A byte-identical-math change (determinism-proven) cannot halve N=2
  decode — the shape is a textbook contention artifact, not a regression.
- **Decision**: do NOT weaken the gate and do NOT accept a contention-poisoned
  result. The protection sweep must be re-run in a quiescent window (no other
  inference server active, per `benchmark-evidence/PROTOCOL.md`). The
  correctness-critical gates (byte-identical determinism, unchanged logit bounds,
  24× zero-sync self-chain, parity) are all green and are what guard the change's
  *correctness*; the protection gate guards *throughput*, which S2 does not touch
  (no kernel math changed; +2 int-only kernels over B≤32 rows inside the graph).
- Re-run command (quiescent box):
  `flock /tmp/fucina_gpu.lock -c '/tmp/s2_sweep_dense.sh'` then
  `protection_gate.py check --baseline …/baseline-dense.json --candidate
  /tmp/s2_cand_dense.json` (and the MoE equivalent).

### The S2b dominance-dispatch regression (found + fixed, `0a3bfb0`)

The invalid attempt-1 numbers led to an A/B on the SAME binary
(`FUCINA_QWEN35_NO_GPU_SPLICE`): splice ON = `engine 74 ms/step`, splice OFF =
`30 ms/step`, decode throughput HALVED at N=2/4/8. Instrumentation showed the
regression tracked **graph dispatch**, not the splice kernels: under continuous
batching only two graphs were ever captured (nt=1, nt=31), and every
steady-state decode step (B=2/4/8) matched the **nt=31 admission-peak graph** by
DOMINANCE (`q35_graph_dominates({31,31,1},{4,4,1})==true`). Because fucina runs
exactly `key.num_tokens` rows of REAL work with NO per-step input padding, a
31-row graph replaying a 4-row step processes 31 rows — ~8× waste. Determinism
was never affected (the 4 real rows were always correct; the extra rows were
discarded), which is why the correctness gates passed while throughput tanked.

**Fix**: decode dispatch is now EXACT-match (`q35_graph_exact_match`).
`q35_graph_dominates` is retained in the header as the primitive for the FUTURE
S1/DFlash path, which pads every device buffer to max shapes — only then is
serving a smaller batch from a larger capture correct AND cheap. A host
regression lock in `qwen35-graph-key-test` now asserts a 31-row graph never
matches a 4-row step. This bug was introduced by S2b (the old `graph[B]` array
was implicitly exact-match) and masked until a clean multi-batch server sweep.

### Protection gate — PASS (quiescent box, post-fix)

Both models, `protection_gate.py check` vs the frozen 2026-07-11 baselines
(absolute 5% floor + contemporaneous-vLLM competitive edge). ms/step tracked avgB
cleanly (dense 31→71 ms at B=1→30; MoE 18→57 ms), confirming no contention.

- **dense — GATE PASS**: every cell above floor; claimed-win N=2/4/8 beat vLLM
  (57.0/112.8/193.8 vs vLLM 42.9/83.9/161.7); N=16/32 improved to 304.8/375.2
  (base 235.3/260.6).
- **MoE — GATE PASS**: every cell far above floor (N=1 59.2 vs base 33.9; N=32
  453.2 vs base 293.4); N=8 claimed-win beats vLLM (224.5 vs 146.5).

S2 is throughput-neutral-to-positive and byte-identical deterministic.
</content>
</invoke>
