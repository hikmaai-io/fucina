# Plan: remove legacy Qwen3 architecture support

Status: proposed (decision made 2026-07-11: fucina targets **Qwen3.5 only**)
Owner: Mauro
Scope: delete the pre-3.5 Qwen3 (dense + qwen3moe) engine paths, tests, and docs

## Decision

Fucina does not need to support the legacy Qwen3 architecture (Qwen3-8B dense,
Qwen3-30B-A3B qwen3moe). The only supported Qwen family going forward is
**Qwen3.5** (hybrid GatedDeltaNet + gated attention, 1 shared + top-8-of-256
MoE), per the qwen35-fucina-plan. Gemma/diffusion/e4b engines are out of scope
for this plan and stay.

## Why remove rather than keep

- Two Qwen attention families double the maintenance surface of every
  refactor: today's tensor-management merge (70ee5fd) had to reconcile
  conflicts in code that exists only to serve both.
- The legacy path drags GGUF-specific machinery (Q4_K/Q5_K expert requant,
  qwen3moe arch-detect) that the Qwen3.5 FP8/NVFP4 safetensors path never
  uses.
- Benchmark/CI time: the qwen3 parity/bench gates run models we no longer
  ship.

## What is legacy (to remove) vs shared (to keep)

**Remove — used only by pre-3.5 Qwen:**

- GGUF arch-detect for `qwen3` / `qwen3moe` (M1 lineage, 9164a2c) and the
  QWEN3-family attention-site wiring for the softmax-only stack.
- Legacy MoE loader pieces: expert slab Q5_K requant, qwen3moe router mapping
  (e89acb1 lineage).
- Tests: `test_qwen3_parity.cu`, `test_qwen3_bench.cu`, `test_qwen3_fused.cu`,
  `test_qwen3_prefix.cu`, `test_qwen3_spec.cu`, `test_qwen3_suffix_prefill.cu`,
  `test_qwen3moe_parity.cu`, `test_qwen3moe_one.cu`, `test_qwen3moe_spec.cu`,
  and their Makefile targets (`qwen3-parity-test`, `qwen3moe-*`,
  `qwen3-fused-test`, `qwen3-prefix-test`, `qwen3-suffix-test`, …).
- Server/scheduler branches keyed on the legacy qwen3 family where they are
  not shared with Qwen3.5 (`IsQwen3Family` call sites need auditing: some
  gate behavior Qwen3.5 also relies on).

**Keep — shared or Qwen3.5-native:**

- Everything under `qwen35_*` (backend, runtime, kernels, state, loaders,
  refs, tests) — this IS the target.
- `dg_fp4_moe.cu` grouped NVFP4 GEMM incl. the `_mapped` slot-pool entry
  points (Phase-C SSD streaming uses them).
- Generic GGUF/safetensors loaders, tokenizer, sampler, server, batch
  scheduler, session persistence, `internal/dist`.
- Q4_K/Q6_K kernels used by Gemma/e4b GGUF models (verify per-kernel before
  deleting anything from mmvq — several are shared).

## Sequencing (do NOT start before)

1. **Qwen3.5 benchmark matrix lands** (in flight on claude-fucina): Stage-18
   fusion A/B numbers must be re-established on Qwen3.5-35B-A3B, because the
   current fused-prefill losslessness gate runs on the legacy models being
   deleted. Port that gate to Qwen3.5 first (fused == standalone,
   byte-identical, on 35B-A3B) — the lever must not lose its regression test.
2. Audit `IsQwen3Family` / arch-detect call sites: split "legacy qwen3" from
   "qwen3.5" behavior flags so removal cannot silently change Qwen3.5 paths.
3. Delete legacy tests + Makefile targets in one commit; delete engine/loader
   code in a second; each commit passes full `make` + remaining gates.
4. Update docs (`sota-program-roadmap`, remaining-plans) — Stage-18 evidence
   references legacy models; re-point at the Qwen3.5 gate.

## Acceptance

- `rg -i 'qwen3moe|qwen3[^.5]'` in cuda/ shows only intentional survivors
  (shared kernels, comments explaining lineage).
- Full `make`, all remaining kernel gates, Go tests green.
- Qwen3.5 fused-prefill losslessness gate green on 35B-A3B.
- No change in Qwen3.5 benchmark numbers (same binary before/after removal).

## Risks

- Shared-code entanglement: the fused prefill+decode path (Stage 18) was
  built and validated on legacy models; the Qwen3.5 port of its gate is the
  load-bearing prerequisite, not the deletion itself.
- `mmvq`/quant kernels look legacy but serve Gemma GGUF models — delete only
  what a call-graph audit proves unreachable.
