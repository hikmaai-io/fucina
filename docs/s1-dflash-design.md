# S1a DFlash design and architecture gate

Status: **P0–P2 + planner delivered green; P3/P4 stop at the real-weights boundary** (2026-07-12)

## Delivered (this branch, byte-identical when DFlash disabled)

- **P0 GDN snapshot/rewind/commit** (`ec7f705`): per-slot GDN/conv recurrent-state
  snapshot + `commit(accepted_len)` that restores the pre-verify snapshot and
  replays exactly the accepted tokens through the standard decode path. Gate
  `qwen35-gdn-rollback-test` proves `commit(j)` is byte-identical to `j`
  sequential single-token decodes for every `j` in `0..K` (rewind `j=0` and
  full-accept `j=K` included), on the real Qwen3.5-9B GGUF.
- **P1 deterministic RNG + rejection sampler** (`c068107`): stateless counter PRF
  keyed `(request_seed, absolute_position, domain)`; greedy + probabilistic
  (Leviathan) rejection with deterministic residual/bonus sampling. Gates
  `qwen35-dflash-rng-test` (host oracle + pinned vectors) and
  `qwen35-dflash-parity-test` (CUDA==CPU bit-identical over 120 RNG triples,
  greedy, and 8 probabilistic seeds).
- **P2 config-derived bounds-checked loader** (`4bcb49f`): symbolic `Geometry`
  from the draft `config.json`; validates every global/per-layer tensor
  rank/shape/dtype and the optional reduced-vocab `d2t` map before any CUDA
  allocation; rejects hostile config/tensors with precise reasons. Gate
  `qwen35-dflash-loader-test`; verified offline against the real public config
  (`H=4096 L=6 NQ=32 NKV=8 HD=128 V=248320 mask=248077 F=8 fc_in=32768
  window=4096`, layers `S S S S S F`).
- **Planner + default-off gate** (`938a72c`): `(1+K)` verify shape, S2 spec graph
  key `(R*(1+K), R, 1+K)` that never aliases a decode key, `K+1` KV lookahead,
  and the `FUCINA_QWEN35_DFLASH=0/1/auto` concurrency gate (default OFF, disable
  at/above a conservative critical batch). OFF schedules no DFlash work; the
  `qwen35-batch` gate (row-independence + graph-on==off + M3-parity + self-chain)
  stays PASS, so decode is byte-identical to current main.

## Real-weights progress (checkpoint downloaded + SHA-verified 2026-07-12)

The draft checkpoint `z-lab/Qwen3.5-9B-DFlash` (2,583,816,465 bytes, SHA-256
`0a42274b32554f48de1faa0d42824e9c2ceda649c30ae0a731cddf410dd698c7`) was
approved, downloaded into the standard hub cache under `/opt/spark/models`, and
its on-disk hash verified byte-for-byte. The entire P3 numerical stack is now
validated against it (device fp32 vs host double reference):

- Real 69-tensor loader gate — exact geometry pinned (`747a7ab`).
- Context-KV precompute: hidden RMSNorm -> fused KV projection -> grouped K-norm
  -> neox RoPE, all 6 layers, 4.3e-7 (`e2840bf`, `23c5885`).
- Non-causal GQA query attention (32 q / 8 kv), 9.0e-8 (`0d622b6`).
- Full DFlash decoder-layer forward composed end to end, 1.3e-6 (`e0e5c8a`).

### Drafting side + engine substrate DONE and validated on real weights

Every drafting-side compute path and the engine substrate are implemented and
gated on the real weights (device fp32 vs host double, signal-relative error):

- Draft residency: 6 layers -> 2.406 GiB BF16 device slab, views byte-match source.
- fc aux-hidden combine (target->draft interface): 1.9e-7.
- Context-KV precompute over residency: 3.9e-7 (K) / 1.8e-7 (V).
- Full query forward (context + self attention, all layers): 3.1e-6.
- Greedy draft sampling (shared LM head argmax + tie rule): exact.
- Single drafting entry point (fc->precompute->query->sample): K in-vocab tokens,
  run-to-run byte-identical.
- Device greedy verify-accept: matches the P1 host oracle for all j in 0..K.
- In-engine resident draft lifecycle: loads + validates the real draft against
  the live target (geometry + vocab matched), gated behind FUCINA_QWEN35_DFLASH,
  freed on destroy.
- Target aux-hidden capture seam: gated decode-body hook, no-op when off.
- DFlash-OFF byte-identity: PASS at every step (qwen35-batch row-independence +
  graph-on==off + M3-parity + self-chain); P0 GDN rollback byte-identical for all
  j in 0..K; full `make lib libdg fucina` + Go tests green.

### Real end-to-end status (2026-07-12): greedy LOSSLESS proven; acceptance = 0 (open)

The integrated `gemma4_engine_q35_dflash_real_step` drives the resident real draft
model through the full loop (draft -> verify -> accept -> commit) and its emitted
stream is BYTE-IDENTICAL to plain greedy decode over 32 steps on the FP8 target
(`qwen35-dflash-real-e2e-test`). This proves the losslessness contract with the
real drafter. However the MEASURED mean accepted drafts/step is **0.000** — the
draft currently proposes nothing the target accepts. This is honest: the loop is
lossless by construction; the zero means the aux CONTENT fed to the draft does not
yet match what the checkpoint was trained on.

Ruled out so far (each fixed, still 0): draft context size (now full accumulated
context, not a 17-row window); context RoPE positions (now absolute per-token, not
[0..n)). The remaining open question is the exact aux SEMANTICS the draft's `fc`
expects: which residual tensor at each `target_layer_id` (pre- vs post-norm, which
of the FP8 target's 32 hybrid layers maps to the id), and whether the draft
requires the target's aux at the QUERY positions too (not just context). Resolving
this needs a per-tensor cross-check of one target-layer aux against a reference,
not more loop iterations. **No speedup is claimed; acceptance is not yet
demonstrated.**

### Remaining for S1A_VALIDATED (the final serving-step orchestration)

- The verify serving step: run the target `(1+K)` forward over [last accepted ++
  K draft] with all-row logit capture + aux capture, call the device verify-
  accept, commit via P0 rollback + P4 assembly, advance zero-host-feedback, N+1
  lookahead, concurrency-gated (this touches the target verify forward for all-row
  logits and a persistent draft KV cache across steps).
- End-to-end gates: greedy DFlash token-identical to greedy baseline (losslessness
  proof); measured acceptance length + tok/s at B=1/2/4 vs the non-spec baseline
  on the same checkpoint (real numbers only).

---

## Original architecture-gate analysis (retained)

## Scope and sources

This design targets only `Qwen/Qwen3.5-9B` with the exact public draft
`z-lab/Qwen3.5-9B-DFlash`. It was derived from:

- repository revision `5fc3b3d474760f18c516db87d84c37edbfd3ede6`;
- its public `config.json`, README, Hugging Face file metadata, and the first
  1 MiB HTTP range of the safetensors file (the complete safetensors JSON header
  is 7,433 bytes); no weight payload was downloaded;
- vLLM at `/tmp/vllm`, especially `qwen3_dflash.py`, the MRV2 DFlash
  speculator/CUDA-graph manager, rejection sampler/Gumbel implementation,
  Mamba-hybrid/GDN state management, and scheduler lookahead tests;
- fucina's current Qwen3.5 runtime and S2 graph substrate.

The public weight artifact is **2,583,816,465 bytes** with SHA-256
`0a42274b32554f48de1faa0d42824e9c2ceda649c30ae0a731cddf410dd698c7`.
It was not downloaded. Any future download requires explicit approval after
restating this size and hash.

## Architecture gate result

The requested S1a/S1b boundary is not implementable losslessly as stated.
`Qwen3.5-9B` is dense in its FFN, but it is **not stateless/full-attention-only**:
its target `text_config.layer_types` has 32 layers in a repeating
`linear_attention, linear_attention, linear_attention, full_attention` cadence.
Thus 24 target layers are stateful Gated DeltaNet (GDN) layers. fucina's
`qwen35_decode_multiseq_body` mutates each slot's GDN matrix and causal-conv ring
in place for every verified token.

A target `(1+K)` verification pass necessarily advances those recurrent states
through rejected proposals. Lossless continuation requires one of:

1. snapshotting candidate recurrent states and selecting the state after the
   last accepted token;
2. snapshot/restore followed by recomputation through accepted tokens; or
3. an equivalent multi-version state-cache scheme.

All three are GDN rewind/commit semantics. vLLM confirms this contract with
per-spec-token state indices and `num_accepted_tokens` in `gdn_attn.py`, fed by
`MambaHybridModelState`. fucina itself documents that its GDN state is a
recurrence at exactly `n_tokens` and cannot be truncated like KV.

The mission asks Phase 4 to serve the dense 9B target while explicitly reserving
"GDN rewind/35B" for S1b and forbidding it from being implemented or faked in
S1a. The problem is model architecture, not model size or MoE: **9B needs the
same correctness primitive.** Proceeding without it would silently corrupt
state after the first rejection and violate losslessness and byte-identical
repeatability. Consequently Phases 1–5 are not started; tests or loader work
would not make the requested serving path correct.

There is a second, non-blocking correction to the prior mission notes: the exact
draft is not all-full-attention. Its six layers are five
`sliding_attention` layers followed by one `full_attention` layer, with a 4096
window. A future implementation must preserve those config-derived per-layer
modes and separate cache groups; it must not substitute six full-attention
layers.

## Exact public draft config

- architecture/model type: `DFlashDraftModel` / `qwen3`
- dtype: BF16; own checkpoint payload is BF16
- hidden/intermediate: 4096 / 12288
- layers: 6
- Q heads / KV heads / head dimension: 32 / 8 / 128
- vocabulary: 248320; mask token: 248077
- maximum positions: 262144; RoPE theta: 10000000, default style
- RMS epsilon: 1e-6; SiLU; no attention bias
- layer modes: five sliding-attention layers, then one full-attention layer
- sliding window: 4096
- target hidden inputs: target layers `[1,5,9,13,17,21,25,29]`
- `fc` input width: `8 * 4096 = 32768`; output width 4096
- advertised training/context: 40k; configured draft block size 16
- no own embedding or LM head in the artifact; both are shared with the target
- no `d2t`/`t2d` tensor in this artifact; target and draft vocabularies match

### Exact safetensors schema

The file contains 69 tensors:

- `fc.weight`: BF16 `[H, target_hidden * num_target_features]`
- `hidden_norm.weight`, `norm.weight`: BF16 `[H]`
- for every layer `l in [0,L)`:
  - `layers.l.input_layernorm.weight`,
    `layers.l.post_attention_layernorm.weight`: BF16 `[H]`
  - `layers.l.self_attn.q_proj.weight`: BF16 `[NQ*HD,H]`
  - `layers.l.self_attn.k_proj.weight`, `v_proj.weight`: BF16 `[NKV*HD,H]`
  - `layers.l.self_attn.o_proj.weight`: BF16 `[H,NQ*HD]`
  - `layers.l.self_attn.q_norm.weight`, `k_norm.weight`: BF16 `[HD]`
  - `layers.l.mlp.gate_proj.weight`, `up_proj.weight`: BF16 `[I,H]`
  - `layers.l.mlp.down_proj.weight`: BF16 `[H,I]`

A generalized loader must derive all symbols (`H`, `I`, `L`, `NQ`, `NKV`,
`HD`, vocab, target feature count) from config, perform checked integer
multiplication, validate every rank/dimension/dtype and the exact required tensor
set before any CUDA allocation, and reject unknown architecture modes. Optional
reduced-vocabulary checkpoints may add `d2t`; its validated shape is
`[draft_vocab]`, integral, and each mapped target id must remain in target-vocab
bounds. Weight storage must use existing `WeightRef`/safetensors and existing
FP8/NVFP4 conversion paths rather than a parallel tensor system.

## Intended generalized pipeline (after scope correction)

1. Retain target hidden features from configured target layer ids; concatenate
   and project through `fc` when `use_aux_hidden_state` is enabled.
2. Eager variable-shape context precompute: hidden RMSNorm; one GEMM against
   stacked per-layer K/V projections; grouped per-layer K RMSNorm; batched RoPE;
   cache insertion into each draft layer's config-derived sliding/full cache.
3. Build exactly `1+K` draft queries per request: accepted/bonus token followed
   by K mask tokens. Run one non-causal fixed-shape draft forward and sample K
   positions.
4. Run one target `(1+K)` verification pass into **versioned GDN state**, apply
   deterministic rejection, then commit recurrent and attention state only
   through the emitted prefix.
5. Splice the emitted final token and absolute position into S2's persistent GPU
   slot state without host feedback.

Variable context precompute stays eager. Fixed query forward, draft sampling,
target verification, rejection, state selection, and S2 splice are captured with
S2 key `(num_tokens,num_reqs,uniform_token_count)`, where
`uniform_token_count=1+K`. Dominance dispatch is legal only with fully padded
buffers; otherwise exact shape matching is required.

## State ownership and ABI

- Engine owns immutable draft config and `WeightRef`s, fused K/V views, graph
  entries, and all device workspaces.
- Stable request slot owns draft KV and the target's committed GDN/conv/KV state.
- One verification step owns candidate/versioned target state until rejection
  chooses a committed prefix.
- Scheduler owns admission and reserves `K+1` lookahead positions, checked
  against model length and cache capacity before enqueue.
- No per-token allocation, STL, or host synchronization belongs in the hot path.

A future public C ABI should expose config validation/load separately from
activation, report a structured incompatibility reason, and leave the existing
plain-decode ABI untouched when disabled.

## Deterministic rejection math

For request seed `s` and absolute predecessor position `p`, RNG is stateless:
`u = CounterRNG(s,p)`, mapped to the open interval `(0,1)`. Draft and verifier
compute the same key independently; no mutable RNG state is advanced.

- Greedy: accept proposal `x_i` iff it equals target `argmax(P_i)` using the
  existing deterministic tie rule. On first mismatch emit target argmax; if all
  K match, emit the target bonus token.
- Probabilistic: accept while `P_i(x_i) > u_i Q_i(x_i)` (equivalently
  `log P_i(x_i) > log(u_i)+log Q_i(x_i)`). At first rejection sample from
  `R_i(x) = max(P_i(x)-Q_i(x),0) / sum_x max(P_i(x)-Q_i(x),0)`. If all proposals
  pass, sample the bonus from the target distribution.

Sampling and rejection uniforms require distinct deterministic counter domains
under the same `(seed, absolute_position)` contract. CPU and CUDA must use the
same integer mixing, float conversion, strict comparison, tie breaking, and
nonzero clamp; tests must include boundary uniforms, infinities, reduced vocab,
and repeated runs.

## Configuration, gating, rollback

Planned controls were:

- `FUCINA_QWEN35_DFLASH=0|1|auto`, default `0` until real-checkpoint acceptance;
- a config-derived `K` bounded by checkpoint/model/cache limits;
- conservative concurrency gate (initially at most 4, never assumed performant
  above 8 without GB10 measurement);
- immediate rollback with `FUCINA_QWEN35_DFLASH=0`, preserving the current S2
  plain-decode path byte-for-byte.

No activation is safe until the scope explicitly includes target GDN state
selection for 9B.

## 128 GiB unified-memory budget

The exact draft artifact is 2,583,816,465 bytes (2.406 GiB) on disk. Runtime
weight residency must be measured for the chosen representation; it must not be
reported as smaller without an implemented conversion.

Draft KV bytes are config-derived. For this checkpoint each cached token per
layer is `2 * NKV * HD * sizeof(BF16) = 4096` bytes. With five 4096-token sliding
windows plus one full layer:

`draft_kv_per_request(ctx) = 5*4096*4096 + min(ctx,262144)*4096` bytes.

That is 236.25 MiB at a 40k full-layer context and 1.078 GiB at the configured
262144 maximum, before allocator alignment and query lookahead. Capacity must be
admitted against fucina's measured target weights, target recurrent/KV state,
draft weights/KV, graph workspaces, and a non-overcommitted system reserve. On
unified memory, reclaimable file cache is not free model capacity; use the
existing physical-availability accounting.

## Acceptance matrix

| Gate | Synthetic before weights | Real checkpoint required | Current result |
|---|---:|---:|---|
| Config/tensor hostile-shape validation before CUDA allocation | yes | no | not run; stopped at architecture gate |
| Planner shape and `K+1` lookahead boundary tests | yes | no | not run |
| Counter-RNG CPU vectors and CUDA bit parity | yes | no | not run |
| Greedy/probabilistic CPU oracle and CUDA parity | yes | no | not run |
| Graph key/padding and S2 zero-host-feedback state splice | yes | no | S2 existing gates only |
| Candidate GDN state selection/commit parity | yes | no | **required but excluded by S1a scope** |
| Existing build, Go, Qwen3.5/Gemma/diffusion/e4b gates | yes | no | unchanged tree |
| DFlash-off byte identity vs baseline | yes | no | vacuous: no code change |
| Draft forward/logit parity | no | yes | unavailable |
| Rejection output-distribution/accept-length parity | no | yes | unavailable |
| Repeated end-to-end determinism | no | yes | unavailable |
| GB10 critical-concurrency and performance acceptance | no | yes | unavailable; no numbers claimed |

## Required scope decision

Unblock by changing one premise: include lossless target GDN candidate-state
selection for **Qwen3.5-9B** in S1a (while leaving MoE-35B enablement and tuning
for S1b), or choose a genuinely stateless target architecture. Implementing the
remaining phases under the present "no GDN rewind in S1a" rule would weaken the
losslessness gate and is therefore rejected.
