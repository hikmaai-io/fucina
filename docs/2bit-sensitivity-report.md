# 2-bit microscaled quantization for Gemma-4-31B — sensitivity & GO/NO-GO

**Milestone:** de-risk M3/M4 (the `F_bytes` lever of the 89 tok/s plan,
`docs/dense-31b-89tok-plan.md`). **Mode:** offline, no engine build.
**Worktree:** `feat/dense-31b-2bit`. **Stages:** B1 (codec) + B2 (sweep) +
B3 (this synthesis).

---

## 1. The format under test (MXFP2 / NF2)

Mirrors the repo's NVFP4 codec at 2-bit. One FP8 (E4M3) block scale per 16
weights, weights stored as 2-bit codes:

| precision | code | layout per 16-elt block | bytes/elt | bit/elt |
|---|---|---|---|---|
| 2-bit | NF2 (4-entry normal-float codebook) | 1×E4M3 scale + 16×2-bit | 0.3125 | **2.5** |
| 3-bit | NF3 (8-entry normal-float codebook) | 1×E4M3 scale + 16×3-bit | 0.4375 | **3.5** |
| 4-bit | NVFP4 ref | 1×E4M3 scale + 16×4-bit | 0.5625 | **4.5** |

Codebook = symmetric quantiles of a standard normal, scaled so the outermost
level maps to the per-block absmax; scale rounded to E4M3. B1 found NF2 beats the
E1M0 `{-2,-1,+1,+2}` variant, so NF2 is the shippable 2-bit code.

**Reference frame:** all errors are measured **against the Q4_0/Q4_1/Q4_K GGUF
dequant**, i.e. they are the *incremental* 4-bit → 2.5-bit loss (exactly the
`F_bytes` lever), **not** the true loss vs BF16. This is a weight-reconstruction
proxy, not an end-task metric — see §5.

---

## 2. Per-tensor sensitivity (Stage B2, 411 weight tensors)

Full data: `scripts/quant2bit/sensitivity_table.csv` (411 rows),
`sensitivity_summary.md`, `sensitivity_metrics.json`.

**Headline: the problem is uniform.** Round-tripping every quantizable tensor
through NF2 gives mean rel_mse **0.1545**, sd ~0.006; 410 of 411 tensors fall in
0.145–0.182. There is **no catastrophic per-tensor cliff** — this is a
uniform-budget problem, not a few-fragile-tensors problem.

### Sensitivity by role (incremental NF2 rel_mse, lower = safer)

| role | n | mean rel_mse | note |
|---|---|---|---|
| **embed** (token_embd) | 1 | **0.1975** | #1 worst, +0.04 over field; already Q4_K in the GGUF |
| **attn_k** | 60 | **0.1584** | worst *role*; narrow KV proj; owns most of top-13 |
| **attn_v** | 50 | **0.1575** | 2nd worst role; feeds the FP8 KV cache |
| ffn_down | 60 | 0.1541 | bulk |
| attn_q | 60 | 0.1540 | bulk |
| ffn_gate | 60 | 0.1538 | bulk |
| ffn_up | 60 | 0.1533 | bulk |
| **attn_o** | 60 | **0.1506** | *best* role — counter to "protect the output proj" |

12 of the top-13 worst tensors are attn_k / attn_v. The KV projections are the
narrowest matmuls (n=2048–4096 cols), so each 16-elt block sees a heavier-tailed
distribution that NF2's 4 levels fit worst.

### Sensitivity by layer position

Barely matters: early(0-4) 0.1523 / mid(5-54) 0.1548 / late(55-59) 0.1530. A
faint mid-layer bump (attn_k/v peak ~layers 22–53), but **no first/last-layer
cliff**. Layer-based protection buys ~nothing; **role-based** does.

### 3-bit (NF3) recovery — new in B3

Lifting a tensor from NF2 to NF3 cuts its incremental rel_mse to **~0.17×**
(measured on 5 representative tensors, ratio 0.16–0.18; ≈ +7.7 dB SQNR):

| tensor | NF2 rel_mse | NF3 rel_mse | NF3/NF2 |
|---|---|---|---|
| token_embd | 0.1975 | 0.0319 | 0.161 |
| blk.41.attn_k | 0.1819 | 0.0305 | 0.168 |
| blk.22.attn_v | 0.1733 | 0.0296 | 0.171 |
| blk.0.attn_q | 0.1518 | 0.0272 | 0.179 |
| blk.9.ffn_down | 0.1655 | 0.0287 | 0.173 |

So a single bit of protection (2→3) on the worst tensors almost eliminates their
incremental loss at only +1.0 bit/elt — much cheaper than going to full 4-bit.

---

## 3. Recommended mixed-precision recipe + footprint

Computed exactly from the real per-tensor `n_elem` (30.696 G weight params; the
422 F32 norm/scalar/rope tensors are 1.33 M elems = **0.005 GB**, negligible).
Tool: `scripts/quant2bit/bit_allocation.py` → `bit_allocation.json`.

| recipe | blended bits | GB (weights) | weighted incr. rel_mse | % elems 2b/3b/4b |
|---|---|---|---|---|
| uniform-2bit (all NF2) | 2.500 | **9.59** | 0.1558 | 100 / 0 / 0 |
| **LEAN** (embed+KV @3-bit NF3, rest 2-bit) | **2.621** | **10.06** | 0.1384 | 87.9 / 12.1 / 0 |
| **MIXED** (embed+KV @4-bit, rest 2-bit) | 2.742 | 10.53 | 0.1348 | 87.9 / 0 / 12.1 |
| MIXED+ (embed+KV @4b, attn_q @3b, rest 2b) | 2.843 | 10.91 | 0.1220 | 77.8 / 10 / 12.1 |
| uniform-4bit (NVFP4 ref) | 4.500 | 17.27 | 0.0000 | 0 / 0 / 100 |

Per-role mass & protection cost (over all-2-bit baseline):

| role | Gelem | % of weights | action | extra GB |
|---|---|---|---|---|
| ffn_down / ffn_gate / ffn_up | 6.94 each | 22.6% each (67.8% total) | NF2 @2-bit | — |
| attn_o | 3.08 | 10.0% | NF2 @2-bit | — |
| attn_q | 3.08 | 10.0% | NF2 @2-bit | — |
| embed | 1.41 | 4.6% | protect | +0.35 (4b) / +0.18 (3b) |
| attn_k | 1.21 | 3.9% | protect | +0.30 (4b) / +0.15 (3b) |
| attn_v | 1.10 | 3.6% | protect | +0.28 (4b) / +0.14 (3b) |

**Recommendation: the LEAN recipe — embed + attn_k + attn_v at 3-bit NF3,
everything else 2-bit NF2.** Blended **2.621 bit/elt = 10.06 GB** weights
(10.06 GB total). It gives the best quality-per-byte: it cuts the
worst-tensors' incremental error ~6× for only **+0.47 GB** over uniform-2-bit,
versus MIXED which spends +0.94 GB for a similar (slightly better) effect.

**Honest note on the ~9.5 GB target.** The task targeted ~9.5 GB / 2.45 bit.
**Only uniform-2-bit (9.59 GB) lands there**, and it leaves the worst tensors
unprotected. Any protection of embed+KV pushes to ≥10 GB, because embed alone is
4.6% of params. ~9.5 GB with protection is **not reachable** at this block size;
the realistic protected footprint is **10.0–10.5 GB (2.62–2.74 bit)**. This is
still a 1.7× shrink vs the 17.3 GB 4-bit NVFP4 and comfortably hits `F_bytes`
(31B in ~10 GB, well under the 11 GB budget).

**Also honest:** the weighted incremental rel_mse barely improves with
protection (0.156 → 0.138), because the 2-bit FFN bulk (68% of mass) dominates
regardless of how well embed+KV are protected. Protection is cheap insurance for
attention fidelity, **not** a fix for the dominant FFN 2-bit error.

---

## 4. Gold-standard perplexity — SKIPPED (infeasible offline)

torch / transformers are **not installed** on this box (`pip list` shows neither;
no `import torch`). The BF16 weights *are* present
(`/opt/spark/models/hub/models--google--gemma-4-31B-it`, ~62 GB), but installing
a multi-GB torch+transformers stack and running a forward pass over a 62 GB model
is out of scope for an offline, no-engine-build de-risk subagent (heavy install,
slow load, risk of hang). **No real perplexity / logit-KL was measured.**

Projection from B2/B3 reconstruction error instead: per-block NF2 at SQNR ~8 dB
(cos ~0.92; ~10 dB / 0.95 with an MSE-optimal scale) is a **large** absolute
weight perturbation. By analogy to published 2-bit weight-only results
(GPTQ/AWQ-class), uniform 2-bit *without* calibration or a learned codebook
typically incurs a meaningful perplexity hit; the NF codebook + microscaling
here is better than naive int2 but the ~0.15 incremental rel_mse on the FFN bulk
is the open risk. **This must be confirmed with a forward-pass logit-KL /
perplexity before committing** — see remaining work.

---

## 5. GO / NO-GO verdict

**CONDITIONAL GO** for ~2.5-bit microscaled (NF2) on Gemma-4-31B, with the LEAN
mixed-precision recipe, **gated on a forward-pass quality check**.

What is de-risked (high confidence, offline):
- The *shape* of the problem: **uniform, no fragile-tensor cliff, no layer
  cliff.** A simple role-based recipe is sufficient; no per-tensor search needed.
- The **byte budget is met**: LEAN = 10.06 GB (2.62 bit), MIXED = 10.53 GB —
  both under the 11 GB `F_bytes` budget and ~1.7× smaller than 4-bit NVFP4.
- Protection is cheap: embed+KV at 3-bit costs only +0.47 GB and erases ~6× of
  their incremental error.

What is **not** de-risked (the gating risk):
- **End-task quality is unproven.** All numbers are incremental
  weight-reconstruction error vs the 4-bit GGUF, at SQNR ~8 dB. No perplexity /
  logit-KL was run (no torch offline). The 2-bit FFN bulk (68% of mass at
  rel_mse ~0.15) is the dominant unknown and protection does not address it.

**Decision rule:** proceed to *implement* the LEAN recipe in the MXFP2 codec
path (it is provably within budget and the right shape), **but do not ship**
until a forward-pass logit-KL vs BF16 (or perplexity on a few hundred calib
tokens) confirms the FFN 2-bit error is tolerable. If that check fails, the
cheapest next lever is lifting the FFN-down (or all FFN) to 3-bit NF3, which the
B3 data shows recovers ~6× of the error per +1 bit — landing ~3.0–3.3 bit /
12–13 GB, still under a 4-bit footprint.

---

## 6. Reproduce

```
# sensitivity sweep (B2)
python3 scripts/quant2bit/sensitivity_sweep.py    # -> sensitivity_*.{csv,md,json}
# bit-allocation synthesis (B3)
F32_BYTES=5327088 python3 scripts/quant2bit/bit_allocation.py   # -> bit_allocation.json
```
Deps: numpy only. GGUF:
`/opt/spark/models/hub/models--unsloth--gemma-4-31B-it-GGUF/.../gemma-4-31B-it-Q4_0.gguf`.
