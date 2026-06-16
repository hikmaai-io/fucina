# KV-cache quantization refinement — exploration & decision (Phase 6)

Status: **EXPLORED.** Tool: `cuda/kv_quant_explore.cc` (`make kv-quant-explore`). This is the
"(opt) KV quant refinement, gate on tool-bench" phase of `docs/continuous-batching.md`. Conclusion
up front: **keep FP8 E4M3 as the default KV codec; TurboQuant is declined; NVFP4-KV is the only
refinement with a real payoff (memory, ~1.78×) and is offered as a flagged, bench-gated follow-up,
not a default.**

## What the engine does today

KV is stored as flat **FP8 E4M3, 8 bit/elem, no scaling** (`kv_t = __nv_fp8_storage_t`). K is
post-RMSNorm + post-RoPE, V is post-RMSNorm — both O(1) magnitude, well inside E4M3's ±448. There
is no per-head / per-block / per-tensor KV scale anywhere; the only clamp is the ±448 saturate.

## Schemes compared

Head-to-head on the same synthetic post-norm K/V (Gaussian coordinates + a few injected
high-variance "outlier" channels, the thing that wrecks *per-tensor* quantization), at the two
head dims (256 sliding / 512 global):

| scheme | what | bit/elem |
|---|---|---|
| `fp8` | current engine: raw E4M3, no scale | 8.00 |
| `fp8_pertok` | E4M3 + one fp16 per-vector amax scale | 8.06 |
| `nvfp4` | E2M1 + per-16 E4M3 block scale (the weight codec) | 4.50 |
| `tq_mse_b4` | **TurboQuant-MSE** (arXiv 2504.19874): randomized Hadamard rotation + per-coord Lloyd-Max normal centroids, 4-bit + fp16 norm | 4.06 |
| `tq_mse_b3` | same, 3-bit | 3.06 |

Metrics: relative MSE `E‖x−x̃‖²/E‖x‖²`, cosine error `1−cos`, and inner-product error vs a random
unit query (`ip_bias` = mean signed error — the bias MSE-optimal quant introduces; `ip_rmse` = RMS
error — what perturbs attention logits).

## Results (head_dim 256; 512 is within noise of these)

```
no outliers (fucina's actual well-conditioned post-norm case):
  scheme        rel_mse    cos_err      ip_rmse   bit/el
  fp8          7.06e-04   3.49e-04     2.67e-02     8.00
  fp8_pertok   6.80e-04   3.37e-04     2.58e-02     8.06
  nvfp4        9.05e-03   4.49e-03     9.34e-02     4.50
  tq_mse_b4    9.57e-03   4.75e-03     9.92e-02     4.06    <- no win over nvfp4
  tq_mse_b3    3.43e-02   1.72e-02     1.87e-01     3.06

heavy outliers (16 channels @ std 12, pathological / activation-like):
  fp8          7.02e-04   3.04e-04     8.17e-02     8.00    <- FP8 unfazed by outliers
  fp8_pertok   5.17e-04   2.38e-04     7.21e-02     8.06
  nvfp4        8.84e-03   4.30e-03     2.97e-01     4.50
  tq_mse_b4    8.72e-03   4.34e-03     3.00e-01     4.06    <- ties nvfp4, still no win
  tq_mse_b3    3.15e-02   1.58e-02     5.57e-01     3.06
```

## Findings

1. **FP8 is already near-lossless for KV, and outlier-robust.** rel-MSE ~7e-4 / cos-error ~3e-4 in
   every regime, *including* heavy outliers. E4M3's 4 exponent bits act as a per-element scale, so
   it needs no calibration and shrugs off outlier channels that destroy per-tensor int8. This
   validates the engine's no-scale FP8 design — there is no accuracy problem to fix.

2. **The accuracy refinement (`fp8_pertok`) is not worth it.** A per-vector scale buys ~25% MSE
   reduction under heavy outliers, but the absolute error is already ~1e-3; it adds a scale store +
   a multiply in the hot read path for no user-visible gain. Declined.

3. **TurboQuant gives no advantage at fucina's operating point.** At a comparable 4-bit budget it
   *ties or slightly trails* NVFP4 (`tq_mse_b4` 9.6e-3 vs `nvfp4` 9.0e-3 at fewer bits), while its
   3-bit point is much worse (3.4e-2). The paper's near-optimality and "3.5-bit quality-neutral"
   claims live at 2.5–3.5 bit against *worst-case adversarial* vectors and rely on the QJL-residual
   `prod` variant for inner-product unbiasedness — neither matches fucina's regime, where KV is
   already well-conditioned post-norm and NVFP4's per-block FP scale already absorbs outliers.
   And TurboQuant is *expensive* here: a Hadamard rotate of every K/V at write, a query rotation at
   read, a per-vector norm store, and an inverse rotation for the V output — all for zero accuracy
   over NVFP4. **Declined.** (The randomized-Hadamard machinery is kept in the harness for the
   record; head dims 256/512 are exact powers of two, so it would have been cheap to wire.)

4. **The only refinement with a real payoff is memory, via NVFP4-KV.** ~4.5 bit halves the KV
   footprint vs FP8 (~1.78×: sliding 4096→2304 B/tok/layer, global 1024→576), reusing the existing
   NVFP4 weight codec (E2M1 + per-block E4M3 scale). But: (a) it is **memory-only** — per
   `nvfp4-decode-bandwidth.md` there is no decode-bandwidth edge at the same effective bit-width, so
   it does not speed up generation; (b) it costs ~1e-2 rel-MSE / 4e-3 cos-error on KV, which *can*
   perturb long-context attention and so **must be gated on a real generation bench** (tool-eval /
   spec-bench), not just this offline distortion test; (c) it is a hot-path rewrite of every KV read
   kernel (split-K sliding/global, contiguous + paged) plus the write path.

## Decision

- **Default stays FP8 E4M3.** It is near-lossless and outlier-robust; no change to the stabilized
  engine.
- **TurboQuant: explored and declined** on the data above.
- **NVFP4-KV: recommended only if the client-RAM target (`deploy-target-client-ram.md`) needs the
  ~1.78× KV reduction**, and only behind a flag (like `FUCINA_PAGED_KV`) with a tool-bench gate
  before it could ever become default. Not started — it is a memory-vs-quality trade the user should
  greenlight, not a free win to slip into a perf branch.

The harness is the deliverable here: it lets that decision be re-run against real dumped K/V (feed
real vectors instead of the synthetic generator) the moment the memory pressure is real.
