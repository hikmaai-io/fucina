# S1a DFlash design and architecture gate

## S1A_COMPLETE (2026-07-12)

**Correctness deliverable DONE + fully gated on the real Qwen3.5-9B checkpoints;
exact-lossless greedy wall-clock ceiling MEASURED + understood (not a speedup).**

What is delivered, in-tree on feat/qwen35-dflash, default-off via
FUCINA_QWEN35_DFLASH (DFlash-off byte-identical to main -- qwen35-batch-test):
- P0 GDN snapshot/rewind/commit: commit(j) byte-identical to j sequential decodes
  for all j (qwen35-gdn-rollback-test), on FP8 + GGUF.
- P1 shared-key counter RNG + greedy/probabilistic rejection: CPU oracle + CUDA
  bit-parity; probabilistic rejection distribution-preserving (TV=0.0015, 200k MC).
- P2 config-derived bounds-checked draft loader: real 69-tensor validation +
  hostile-input rejection.
- P3 draft forward on REAL z-lab weights: precompute / non-causal GQA / query
  forward / backbone / combine / residency parities (<=1e-3 vs host double).
- P4 BOTH serving paths on the real FP8 target + z-lab draft:
  * GREEDY: emitted stream BYTE-IDENTICAL to plain greedy decode on all prompts
    (qwen35-dflash-measure-test), MEASURED accept 3.56 drafts/step.
  * PROBABILISTIC: distribution-preserving, assembled, deterministic per seed
    (qwen35-dflash-prob-step-test).
- P5 gate matrix: 25 gates GREEN (9 host + 8 GPU numerical/verify + 8 GPU e2e/
  regression) + make lib libdg fucina + go test ./... .

MEASURED performance (real FP8 target + z-lab draft, B=1, single-stream): target
decode 29 ms/tok; DFlash greedy step 436 ms emitting 9.2 tok/step = ~47 ms/emitted-
token => **~1.6x SLOWER**. The draft forward was optimized 14x (1059->76 ms,
lossless). The residual gap is an INTRINSIC ceiling: exact-lossless accept requires
per-token decode-body validation of each accepted token because the batched verify
argmax diverges from single-token decode at interior positions (batched GEMM/attn
vs gemv numerics -- MEASURED ~10/40 steps), so the commit replays ~j single-token
decodes. NO speedup is claimed. Routes below the ceiling (future, out of scope):
bit-identical batched forward, statistical losslessness, or a cheaper draft.

Exact remaining real-checkpoint validation (future, NOT blocking correctness):
wall-clock speedup work (needs a bit-identical batched forward), B>1 concurrency
tuning, and the 35B tuning pass on the same rollback primitive.

### DFlash-off byte-identity: STRUCTURAL, not merely tested (verified 2026-07-12)

The P5 "DFlash-off byte-identical to main" requirement holds by CONSTRUCTION, the
strongest possible form: `eng->q35.dflash_mode` is written once from the
FUCINA_QWEN35_DFLASH env (gemma4_kernels.cu:5092) and READ NOWHERE in any decode
path -- its only consumer is the (test-only) planner q35_dflash_gate. The
production hot path gemma4_engine_step_batch (line ~12374) contains ZERO DFlash
references; the entire feature is reachable only through the explicit opt-in APIs
gemma4_engine_q35_dflash_real_step / _real_step_prob, which nothing in the serving
loop calls. So the emitted stream with the flag off/absent is not just empirically
equal to main (qwen35-batch-test, greedy determinism gate) -- the decode path is
literally the same code, making a regression structurally impossible. The greedy
repeated-run determinism gate (qwen35-dflash-determinism-test) additionally proves
the opt-in DFlash path itself is reproducible: 43 emitted tokens + 16 per-step
accept counts byte-identical across two independent runs.

---

Status: **BOTH DFLASH SERVING PATHS COMPLETE on real weights, full gate matrix green**
(2026-07-12). Greedy: lossless (emitted byte-identical to plain greedy decode on
every prompt), measured single-stream acceptance 3.56 drafts/step. Probabilistic:
distribution-preserving (P1 rejection, proven TV=0.0015), assembled + functional
(deterministic per seed on the real FP8 target).

## Honest performance status (2026-07-12) — NOT yet net-faster

MEASURED on the real FP8 target + z-lab draft, single-stream B=1:
- target plain decode: **29.2 ms/tok**.
- DFlash greedy step: **1409 ms/step**, emitting 9.2 tokens/step => **153 ms per
  emitted token** — i.e. DFlash is currently **~5.2x SLOWER per token** despite a
  high 9.2 accept.
- The draft step dominates; a micro-profile of the draft forward (K=16, ctx=32)
  shows the LM-head sampling is the bottleneck (head ~286 ms vs query_fwd 28 ms,
  precompute 9 ms). The draft head does K serial H-dot-products per vocab token in
  a single thread — compute-bound, not bandwidth-bound.

Optimization so far (draft forward, MEASURED, parity preserved): fp64->fp32 warp
matmul + shared-mem head (1059.5->477.8 ms), batched head one-weight-pass
(477.8->327.1 ms), warp-cooperative head (327->76 ms, 14x cumulative). All
lossless (measure gate byte-identical on all prompts).

UPDATED end-to-end (MEASURED, real FP8 target + z-lab draft, B=1, head opt +
SEQUENTIAL lossless commit): target decode 29.1 ms/tok; DFlash step 436 ms
emitting 9.2 tok/step = **47.4 ms/emitted-token => ~1.6x slower** (down from 5.2x).

The remaining bottleneck is `q35_gdn_commit`: it replays the 1+j accepted tokens
as j SEQUENTIAL decode-body steps (MEASURED 293 ms for 10 tokens). A batched-commit
using the CHUNK body was tried and REVERTED: the chunk GDN kernel
(qwen35_b_gdn_chunk_kernel) is NOT bit-identical to the single-token decode kernel
(qwen35_b_gdn_kernel) -- it drifts by ~position 45 on the numeric prompt, breaking
losslessness (this is also why greedy_step must derive the emitted token from the
DECODE body, not the chunk-body argmax).

Lossless fast-commit STATE primitive (DONE, q35_gdn_commit_fast, gated byte-
identical for all j): batches the weight-heavy projections once but runs the
GDN/conv recurrence token-sequentially with the DECODE kernels -> per-slot state
byte-identical to j sequential decodes in ~one weight pass.

BUT it does NOT remove the commit-replay cost, because of a deeper finding
(MEASURED, /tmp/divg): the batched (T-row) verify argmax diverges from single-
token decode at ~10 INTERIOR positions per 40 steps (K=16), NOT just the
correction -- and switching the verify GDN recurrence to the decode kernel does
NOT fix it. The divergence comes from batched GEMM/attention vs single-token
gemv numerics across the 8 FULL-attention + projection layers. CONSEQUENCE: a
lossless accept decision REQUIRES per-token decode-body validation of each
ACCEPTED token; the fast-commit gives lossless STATE but not a trustworthy accept
argmax. So greedy_step keeps the sequential q35_gdn_commit (j decode steps for j
accepted tokens). This is an INTRINSIC performance ceiling for the exact-losslessness
contract with the current batched kernels: at accept ~9.2 the commit replays ~10
single-token decodes (~290 ms), so DFlash sits ~1.6x SLOWER per token.

The ONLY ways below this ceiling (future, out of current scope): (a) a batched
verify forward that is BIT-IDENTICAL to single-token decode (make the batched
GEMM/attention deterministic-equal to gemv -- a deep engine change), or (b) relax
to a *statistical* losslessness (probabilistic acceptance already is), or (c) a
draft/target where the draft is cheap enough that even with replay the amortized
cost wins at higher accept. CORRECTNESS is complete + gated; the exact-lossless
greedy path is ~1.6x slower and the ceiling is now understood + measured, not
hand-waved.

## Certified P5 gate matrix (2026-07-12, ALL PASS)

Host (8): dflash-rng, -loader, -plan, -commit, -pipeline, -pcgeom, -prob-dist
(TV=0.0015), -real-load. GPU numerical parity (real weights): -parity (CPU==CUDA),
-verify (greedy accept), -verify-prob (prob accept, 10 seeds), -sample-prob,
-verify-logits, plus precompute/backbone/attn/ctxkv/residency/forward/query/combine.
GPU end-to-end (real FP8 target + z-lab draft): -gdn-rollback (commit(j) byte-
identical), -measure (GREEDY byte-identical to plain greedy on all prompts, mean
emitted/step 4.556 = accepted 3.556), -prob-step (probabilistic assembly: in-vocab
+ deterministic per seed). Regression: -batch (row-independence + graph-on==off +
M3-parity + self-chain), full make lib libdg fucina + Go tests. DFlash-off byte-
identity preserved throughout.

## Certified gate matrix (2026-07-12, all PASS)

Host: dflash-rng, -loader, -plan, -commit, -pipeline, -pcgeom, -real-load.
GPU numerical (real weights): -parity (CPU==CUDA), -query forward, -draft entry,
-verify-accept, precompute/backbone/attn/ctxkv/residency/forward/combine parities.
GPU end-to-end (real FP8 target + z-lab draft): -gdn-rollback (commit(j) byte-
identical ∀j), -verify-block (per-row argmax == sequential decode, GDN rollback),
-engine-load (real draft resident + validated), **-measure (byte-identical to
greedy on all prompts; MEASURED mean emitted/step 4.556 = accepted 3.556/step)**.
Regression: -batch (row-independence + graph-on==off + M3-parity + self-chain),
full `make lib libdg fucina` + Go tests. DFlash-off byte-identity preserved.

---

(Original phased status below.)


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

### Real speculation WORKING (2026-07-12): measured accept 6.78-7.71/step, one losslessness bug open

After fixing the aux data-flow bugs (query embeds token ids; aux = residual
stream at layer id-1; SWA draft layers causal; **and the two that unblocked it:**
the aux-capture hook was missing from the verify chunk-body path, and the engine
`target_layer_ids` parser broke on the config's NEWLINES so ZERO capture layers
were set), the real draft now PREDICTS the target. Measured on the FP8 target +
real z-lab draft:

- prompt 0 ("The capital of France is"): lossless, **6.78 accepted+emitted/step**.
- prompt 1: lossless, **7.71/step**.
- prompt 2 (numeric ids): **losslessness FAILS at the tail** under many rejections
  — a real bug (`qwen35-dflash-measure-test` returns nonzero, kept honest).

Isolation results (all GPU-verified on the FP8 target):
- verify-block per-row argmax: **lossless, 17/17 == sequential decode at K=16** on
  the failing numeric prompt (so the verify math is correct).
- Q8-vs-BF16 head: Q8 head is exact-by-design.
- draft-model forward corrupting scratch: **RULED OUT** — the failure reproduces
  with SYNTHETIC drafts (no draft-model forward) via greedy_step (`iso.cu`).
- K dependence: **RULED OUT** — fails at the same emit index 45 with K=1 and K=16.
- prompt dependence: **CONFIRMED** — the France prompt is lossless to N=56 (14
  steps); the numeric prompt {100,200,..,1000} fails at emit 45. Plain greedy on
  the numeric prompt is deterministic (56/56 across two runs).
- P0 GDN rollback gate: PASS on FP8 (single snapshot->advance->commit byte-
  identical), so ONE commit is perfect.

ROOT CAUSE FOUND + FIXED: `greedy_step` trusted the verify CHUNK-body per-row
argmax (`am[]`) as both the accept decision AND the emitted correction token. But
the verify chunk body and the standard DECODE body can produce subtly different
logits for the same tokens (different kernels/accumulation order), so on some
sequences the chunk-body argmax disagreed with what the committed decode state
actually predicts — emitting a token that diverges from plain greedy decode. Fix:
the emitted tokens are now AUTHORITATIVELY derived from the decode body
(`q35_gdn_commit`'s per-replay-step argmax `out_next[]`): the chunk-body argmax is
only a fast filter to bound replay length, then acceptance is re-derived by
comparing each draft to the decode body's `out_next[i]`, and the correction is
`out_next[j]`. Emitted tokens == plain greedy decode by construction.

Result (measured, real, single-stream, FP8 target + z-lab draft, all lossless):
prompt 0 6.78/step, prompt 1 7.71/step, prompt 2 2.45/step; AGGREGATE mean
emitted/step = 4.556 (accepted 3.556/step). qwen35-dflash-measure-test PASSES
(byte-identical to greedy on every prompt). P0 rollback + verify-block + batch
gates still green. No speedup claimed yet (reference draft kernels are unoptimized
fp64-accum; wall-clock is not yet favorable and is reported honestly).

### (superseded) earlier status: greedy LOSSLESS proven; acceptance = 0

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
[0..n)); plumbing/structure (a `FUCINA_DFLASH_DIAG` trace shows `draft0` is
deterministic and RESPONDS to the repeating context with a repeating output — the
pipeline runs and is context-sensitive — but the prediction never equals the
target greedy token). The output being context-responsive yet systematically
wrong localizes the bug to **aux CONTENT semantics**: the exact target hidden-
state tensor the draft's `fc` was trained on at each `target_layer_id` (pre- vs
post-layernorm residual; which of the FP8 target's 32 hybrid layers maps to each
id; possibly the aux at query positions too). This is a checkpoint-matching
research problem that requires a Python/transformers reference cross-check of one
target-layer aux tensor against fucina's capture — a distinct investigation, not
more serving-loop iterations. **No speedup is claimed; acceptance is not yet
demonstrated (measured 0).** Greedy losslessness remains fully proven and gated.

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
