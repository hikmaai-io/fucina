# Tree speculative decoding for Gemma-4 MTP (feat/tree-spec)

## Why (measured, not assumed)
Linear MTP spec is near its single-chain ceiling: at temp=1 the draw-and-match accept rule
already captures code 58% / prose 29%, and no verify-rule lever (idea A, the confidence gate)
moves prose tok/s. The bottleneck is structural: **one argmax chain cannot cover a high-entropy
next-token distribution.** The de-risk diagnostic (`FUCINA_SPEC_TOPK_DIAG`, commit dc703a9)
proves the headroom is reachable by WIDTH:

| at a reject, target token in head's… | top-2 | top-4 |
|---|---|---|
| PROSE | 21% | **40%** |
| CODE  | 30% | **63%** |

A width-4 tree recovers 40–63% of the positions where the linear chain dies.

**Economics make it cheap.** The batched verify forwards all K tree nodes in ONE weight pass —
the 17 GB target weights are read once per step no matter how many candidates. On a
bandwidth-bound decode the marginal cost of a tree node is only attention + sampling
(O(K·ctx) + O(K·V)), not bandwidth. Wide/deep trees are nearly free where it counts.

## Distribution invariant (must hold)
Acceptance is still **per-edge draw-and-match**: at a node, sample the target token from that
node's logits; accept a child iff the child's draft token equals it. This preserves the exact
target distribution, just like the linear path. Correctness gate: at temp=0, fixed seed, the
tree path MUST emit byte-identical output to the linear path (the tree only changes which
drafts are *tried*, never which token is *committed*).

## Data structures
Static tree template (tuned offline), described by parent pointers:
```
struct spec_tree {
  int   n;                       // number of nodes (root = committed g at index 0)
  int   parent[TREE_MAX];        // parent[0] = -1; ancestors via chase
  int   depth[TREE_MAX];         // absolute pos offset = depth (root depth 0)
  int32 tok[TREE_MAX];           // draft token at each node (filled by drafter)
  // ancestor mask for verify attention: anc[r] bit c set iff c is on root→r path
  uint32 anc[TREE_MAX];          // TREE_MAX ≤ 32 so a u32 bitmask suffices
};
```
`TREE_MAX ≤ GEMMA4_SPEC_MAX` initially (16). Template examples: trunk depth-6 width-1 (==current
linear) as a sanity baseline; then add top-2 branches at the shallow trunk nodes (where coverage
is highest) — e.g. depth-4 with width-2 at depths 0–1.

## Build milestones
- **T1 — correctness skeleton.** Tree structs + a static template. Drafter fills `tok[]` by
  running the MTP head per node (h flows along the parent edge; each node's forward conditioned
  on parent token + parent h). Tree-aware batched verify forward: node r at absolute pos
  `pos+depth[r]`, writes KV to slot `pos+r`, attention = full prefix `[0,pos)` + ancestor slots
  via `anc[r]`. Tree verify walks root→leaf taking the longest draw-and-match path; commit path
  tokens; compact accepted path KV to `[pos,pos+len)`; rewind. GATE: temp=0 byte-identical to
  linear; correct generation at temp=1. Measure τ/tok-s vs linear.
- **T2 — topology tuning.** Sweep width×depth templates for tok/s on prose+code (the de-risk says
  width helps most at shallow depth). Pick defaults; keep `FUCINA_SPEC_TREE=` override.
- **T3 — adaptive.** Size/shape the tree per-step from head confidence (wide when unsure, deep
  when confident), the dynamic-tree analog of the per-drafter EMA already in GenerateSpec.

## Key architectural finding (reshapes T1)
`mtp_forward` is a **pure recurrence in h**: it consumes (token, h), and its attention reads the
FROZEN target KV at a FIXED position (`pos_ptr = n_tokens`) for every draft step — it never
attends previously-drafted tokens. The recurrence is carried entirely through `d_mtp_h`, which
the head OVERWRITES on each call (line ~8890, `post_proj → d_mtp_h`). Consequences:
- **Tree drafter is trivial / KV-free.** For node N: set `d_mtp_h=h[N]`, `d_mtp_tok=tok[N]`, run
  `mtp_forward` → logits (N's children dist) + `h'[N]`. All of N's children inherit the SAME
  `h'[N]`; they differ only in their token (top-k of N's logits). Just save `h'[N]` per node
  ([n_nodes][H] ≈ 344 KB) and fork. No per-node position math (head pos is fixed at n_tokens).
- **All the hard work is the TARGET verify**, which IS a full causal forward and DOES need the
  tree mask. Isolate the risk there; keep a separate tree-attention kernel variant so the linear
  path is untouched.

## Hard parts / risks
1. **Ancestor-mask attention.** Reuse the existing prefix attention for `[0,pos)` (dense, shared
   by all rows); add a small tree-block pass that, per row, reduces only over its ≤depth ancestor
   slots using `anc[r]`. N small (≤16–32) so this is cheap and simple.
2. **Per-node head forward** breaks the single-chain CUDA graph (h no longer a single recurrence).
   Start per-kernel (no graph) for correctness; re-add a per-template captured graph in T2.
3. **KV compaction** of the accepted scattered path → contiguous. Post-verify per-layer slot
   copies (path length small). Verify rewind math (`gemma4_engine_rewind`) stays exact.
4. Positions: node pos = depth, NOT row index — RoPE and attention bounds must use `depth[r]`.

## Reference
GLM-5.2 IndexShare (indices computed once, reused across draft steps) → the shared-prefix
attention here. SpecInfer/EAGLE topology-aware causal mask = the `anc[]` mask. See
[[mtp-spec-ideas-glm-bebop]].
