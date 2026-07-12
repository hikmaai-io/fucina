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

(appended as work lands)
</content>
</invoke>
