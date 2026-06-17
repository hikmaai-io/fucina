# Dense Gemma-4 31B at ~89 tok/s on the GB10 — design & brainstorm

**Branch:** `feat/dense-31b-tau`
**Target:** prefill *and* decode a **dense** Gemma-4-31B-it (Q4_0, 17.34 GB) at the throughput
of a 30B **MoE** (~89 tok/s single-stream), on one DGX Spark GB10 (273 GB/s, ~1 PFLOP FP4).
**Status:** design. No τ/quant code landed yet; this doc is the contract.

---

## 0. The governing equation

Decode on GB10 is **bandwidth-bound on weight bytes**. The only equation that matters:

```
emitted_tok/s  =  ( effective_BW  /  weight_bytes_per_forward )  ×  τ
```

* `effective_BW`  — GB/s actually achieved by the decode GEMV (today ~152–193 of 273).
* `weight_bytes_per_forward` — bytes of weights streamed once per forward pass.
* `τ` — **tokens committed per weight-read** (speculative acceptance length). τ=1 is plain decode.

A 30B MoE hits 89 because it reads only ~3B active params (~1.7 GB). A dense 31B has **no such
escape** — every forward reads all of it.

### The 31B numbers (measured dims, Q4_0 = 17.34 GB)

| effective_BW | raw tok/s (τ=1) | gap to 89 |
|---|---|---|
| 152 (today's NVFP4 GEMV) | 8.8 | 10.1× |
| 273 (hardware ceiling)   | 15.7 | 5.7× |

**Conclusion:** 89 tok/s requires an effective ~**3.07 GB read per *emitted* token** — a **5.7×
reduction** vs reading 17.34 GB once. No single-token decode path reaches it. The 5.7× must come
from the product of three independently-improvable factors.

---

## 1. The budget: spend 5.7× across THREE factors (the key insight)

Instead of forcing τ alone to 5.7 (requires an EAGLE-3-class drafter — hard research), **split the
budget**. Each factor below is individually known/achievable:

```
89 / 15.7  =  5.7×   =   F_bytes  ×  F_bw  ×  F_τ
```

| Factor | Lever | Conservative | Aggressive |
|---|---|---|---|
| **F_bytes** | quantize 4.5-bit → ~2.5-bit (mixed) | 1.4× (3.2-bit) | 2.0× (2.25-bit) |
| **F_bw** | perfect GEMV: 152 → 240+ GB/s | 1.3× | 1.6× |
| **F_τ** | tree spec verify (no EAGLE needed) | 2.5× | 3.5× |
| **product** | | **4.6×** | **11×** |

Two worked recipes that both land ≥ 89:

* **Recipe A (quant-led):** 2.5-bit mixed (≈9.6 GB) · BW 230 · τ 3.0 → **24 t/s raw × 3.0 ≈ 72** …
  push τ to 3.7 → **89**. Needs a good 2-bit kernel + simple trees. *No EAGLE.*
* **Recipe B (τ-led):** stay 4.5-bit (17.34 GB) · BW 250 · τ 6 → **14.4 × 6 ≈ 86**. Needs an
  EAGLE-3-class drafter (τ≈6). Highest quality, hardest drafter.

We pursue **A first** (each factor is de-risked and measurable), keep B's EAGLE work as the
quality-preserving fallback. The two compose: 2.5-bit *and* τ=6 → ~160 t/s headroom for sampling-temp
acceptance loss.

---

## 2. Lever B — F_bytes: the quantization mechanism (define it here)

Goal: cut the 17.34 GB while keeping Gemma-4-31B quality near Q4_0. Decode is a GEMV, so the
**dequant ALU is free** (we're bandwidth-bound) — we can afford an arbitrarily clever on-the-fly
decode of an exotic format. Blackwell GB10 has **no native 2-bit MMA**, so 2-bit lives in a custom
unpack→dp4a/FP GEMV, *not* tensor cores. (Spec-verify's K-row skinny GEMM may stay NVFP4/tensor-core;
see §5.)

### Candidate formats (ranked by quality-per-byte × kernel tractability)

1. **MXFP2 / NF2 microscaled 2-bit (recommended first).** Mirror the existing NVFP4 codec but at
   2-bit: 2-bit mantissa codes (E1M0 signed `{-2,-1,+1,+2}` *or* a 4-entry NF2 codebook) + one
   **E4M3 (FP8) block scale per 16 weights**. Footprint = 2/8 + 8/(16·8) = **0.3125 B/elt ≈ 2.5-bit**.
   31B → **~9.7 GB**. Same swizzle/block-scale machinery as `nvfp4_gemv.cuh`, just narrower codes.
   The decode kernel is the NVFP4 GEMV with a 2-bit unpack LUT.
2. **Sensitivity-aware mixed precision (stack on #1).** Keep the *quality-critical* tensors at 4-bit
   and push the bulk to 2-bit. Empirically critical for transformers: `attn_v`, `ffn_down`, layer 0
   and the last 2 layers, and the (tied) embedding/LM head. Push `ffn_gate/up` + most `attn_q/k` to
   2-bit. Target blended **~2.6–2.9-bit** at near-Q4_0 quality. This is the AWQ/SpQR/SqueezeLLM
   "protect the salient weights" idea, applied per-tensor.
3. **Per-layer optimal bit allocation.** Fix a byte budget (e.g. 9.5 GB ≈ 2.45-bit avg), run a
   one-shot sensitivity sweep (perturb each tensor, measure KL on a calib set), and solve a simple
   knapsack for bits per tensor. Best quality-per-byte; offline, no kernel cost.
4. **Outlier-split (SpQR/SqueezeLLM).** Keep ~0.1–1% outlier columns in FP16 as a sparse side-matrix;
   2-bit the dense rest. Adds a tiny sparse SpMV to decode. Use only if #1–3 leave a quality gap.
5. **Codebook / lattice 2-bit (QuIP# / AQLM / QTIP) — the SOTA, hardest kernel.** Near-FP16 quality
   at ~2-bit via vector-quantized lattice codebooks. Decode = gather from a small (cache-resident)
   codebook by 2-bit indices. Highest quality-per-byte; kernel is a shared-memory gather GEMV.
   Reserve for if linear 2-bit (#1–3) can't hold 31B quality.

### Quality gate (non-negotiable — matches repo's bar)

* Bit-exactness vs Q4_0 is **gone** by construction; replace with: **KL(2-bit ‖ Q4_0) per token**
  under a fixed budget on a calibration set, plus task evals (the repo's `quality_reports/` harness).
* `compute-sanitizer` clean; greedy determinism preserved (same seed → same tokens).
* Stage behind a flag (`FUCINA_WBITS=2`), default stays Q4_0 — same discipline as the packed-KV codec.

---

## 3. Lever A — F_bw: the "perfect" 2-bit decode GEMV kernel

Today `nvfp4_gemv.cuh` reaches 152–193 GB/s (55–70% of 273) with ROWS=4 register-blocking. Spec for
the 2-bit kernel (and a back-port of its wins to the 4-bit path):

* **Warp-per-N-rows, register-blocked** (ROWS=4..8): one weight stream feeds N output rows from
  registers → amortizes the activation read, raises arithmetic intensity off the bandwidth floor.
* **128-bit vectorized loads** (`int4`): one 16-byte load = **64 weights at 2-bit**. Maximal
  coalescing, minimal instruction overhead per byte.
* **`cp.async` double-buffered** weight tiles (Blackwell async copy) → hide global latency behind
  compute, no L2 thrash.
* **2-bit unpack via a 16-entry LUT in registers/`__shfl`**; dequant `code × fp8_block_scale` fused
  into the dp4a/FFMA accumulate. ALU is free here.
* **One mega-CUDA-graph per forward** — capture the *entire* 60-layer decode (and the K-row verify)
  as a single replayed graph; kill per-kernel launch bubbles (the README's stated lever). This alone
  is ~1.3–1.6× on a model this deep (60 layers × many kernels = many launch gaps).
* **Target:** ≥ 230 GB/s effective → 9.7 GB model decodes at **~24 t/s raw**.

---

## 4. Lever C — F_τ: tree speculative verification (build out the `:7984` stub)

`cuda/gemma4_kernels.cu:7984` "Token-tree speculative decode" is an **empty stub**. Today only a
**linear MTP chain** (τ≈2.5, 0.59 accept) + prompt-lookup exist. A chain caps near τ≈3 (one miss
kills the tail). Trees are how τ gets to 3.5–6.

* **Draft a tree, not a chain.** From the MTP head, expand top-2 (or top-k) at each of D positions →
  a ~16–48-node candidate tree (budget already exists: `GEMMA4_SPEC_MAX=16`).
* **Tree attention mask in the batched verify.** The existing "weight-read-once batched verify"
  already forwards K rows in one weight pass; add a **tree causal mask** so each node attends only
  its ancestors. One weight-read verifies the whole tree.
* **Longest-accepted-path commit + exact rewind.** Reuse the existing exact-rewind machinery; walk
  the accepted path, commit its tokens, rewind the rest. Distribution-exact at any temperature
  (preserve the current guarantee).
* **Expected:** linear τ≈2.5 → tree τ≈3.5–4.5 on structured text; less on creative (set
  expectations). This is the single highest-leverage τ item and needs **no new model**.

### Drafter upgrades (raise τ further, ordered by ROI)

1. **Wider/deeper MTP tree** (above) — reuse the shipped 4-layer assistant head. *Needs a 31B
   assistant head;* if Google/Unsloth ship none for 31B, fall back to prompt-lookup + self-spec.
2. **EAGLE-3 feature drafter** (Recipe B): draft in the target's hidden-feature space reusing
   multi-level target activations; reported τ≈4–6. Replaces the recurrent MTP head. Hardest; needs
   training a small head against 31B features.
3. **Self-speculation / LayerSkip** (no extra model): early-exit the target at layer ~20/60 to
   draft, verify with the full 60-layer pass. Draft cost ≈ ⅓ forward, acceptance high (it *is* the
   model). Best **fallback** when no 31B assistant head exists and EAGLE isn't trained yet.
4. **Dual-quant self-spec (edge idea):** draft with the **2-bit** model, verify with a **higher-bit**
   copy. Same architecture → high acceptance; but autoregressive 2-bit drafting reads ~9.7 GB×K,
   which dominates the 17 GB verify unless paired with LayerSkip. Use only layer-skipped.

---

## 5. Cross-cutting: where tensor cores vs GEMV applies

* **batch=1 decode** → pure GEMV; 2-bit unpack wins (fewer bytes, ALU free, no MMA benefit).
* **K-row spec-verify** → skinny GEMM (up to 16 rows). Here Blackwell **NVFP4 tensor-core MMA** may
  beat a 2-bit dp4a GEMV. Option: **hybrid storage** — keep a 2-bit copy for batch-1 decode *and* an
  NVFP4 copy for the verify GEMM (9.7 + 17 = ~27 GB, trivially fits 128 GB). Measure both; pick per
  path. This is a real edge lever: the verify reads NVFP4 once for the whole tree via tensor cores,
  the lone-token decode reads 2-bit.
* **prefill** is compute-bound, not the bottleneck — GB10's FP4 FLOPs eat a 31B prefill fine; the
  batched BF16/FP4 tensor-core prefill path already exists. No work needed for the headline number.

---

## 6. Milestones (sequenced; each ends in a measured number)

| M | Deliverable | Why first | Exit metric |
|---|---|---|---|
| **M0** | **Load & run 31B Q4_0** (generalize dims) | nothing is measurable until 31B runs | raw decode tok/s on 31B |
| M1 | Mega-graph + GEMV back-port (F_bw) | cheap 1.3–1.6×, helps every later step | effective GB/s ↑ |
| M2 | Tree verify on existing MTP/prompt-lookup (F_τ) | biggest τ jump, no new model | τ, tok/s |
| M3 | MXFP2 codec + 2-bit decode GEMV (F_bytes) | the quant mechanism | bytes ↓, quality gate |
| M4 | Sensitivity-aware mixed precision | recover 2-bit quality | KL vs Q4_0, evals |
| M5 | EAGLE-3 / LayerSkip self-spec | push τ to 6 (Recipe B) | τ ≥ 5 |

### M0 blast radius (Gemma-4-31B vs hardcoded 12B)

**Constraint (hard):** fucina is ONE binary that **identifies the model from the checkpoint** — arch
and format read from the GGUF kv / safetensors `config.json` at load time. **No compiler flags, no
env vars** to select model size. So M0 is a *runtime* config, not a compile-time variant.

From the dim survey — ~80 coupled locations. Hard blockers:

* The 12B `#define`s (`gemma4_kernels.cuh`: `MAX_LAYERS 48`, `HIDDEN 3840`, `INTERMEDIATE 15360`,
  `HEADS 16`, `KV_HEADS 8`, `GLOBAL_KV_HEADS 1`) become **fields of `gemma4_model_config_t`**
  (`cuda/gemma4_config.h`) populated by the loader. Static arrays size to `GEMMA4_CAP_*` maxima;
  loops/launches read the runtime counts. Head dims stay constant (`HEAD_DIM 256` / `GLOBAL 512`).
* **Structural:** global attention was specialized for **1 KV head (broadcast)**; 31B global is
  **4-KV-head GQA**. The `global_attn_splitk*` family must handle NKV=4, not just broadcast.
* `[GEMMA4_MAX_LAYERS]` fixed arrays in `gemma4_engine_t` (layers, layer_types,
  global_layer_indices, global_slot, d_fp4_w*, h_out_scale) → size to `GEMMA4_CAP_LAYERS` (64).
* Kernel **template instantiations** on `<HEADS, KV_HEADS, HEAD_DIM>` (6 families): since head_dim
  is constant and only a few `(n_heads, n_kv)` configs exist (12B 16/8/1, 31B 32/16/4), instantiate
  the supported configs and **dispatch on the runtime value** at launch. Static `__shared__` tiles
  size for the max head config (`GEMMA4_CAP_*`); HD 256/512 unchanged → fits.
* Loader: `gemma4_kernels.cu:3535` asserts `n_layers == GEMMA4_MAX_LAYERS`; default layer pattern
  `:3404-3415` hardcodes 48/8-global. Replace both with reads of the GGUF `sliding_window_pattern[]`
  and per-layer `head_count_kv[]` (the parse path at `:3601-3643` already exists) — the 31B GGUF
  carries both, so the engine learns it's a 31B with 10 global / 4-KV-global layers from the file.

**M0 approach (runtime only):** add `gemma4_model_config_t`, populate it in the loader from the
file metadata for *both* GGUF and safetensors, thread it through the engine replacing the `#define`
reads, grow static arrays to `GEMMA4_CAP_*`, add the global-GQA kernel path + runtime head-count
dispatch. One binary auto-selects 12B vs 31B (vs any future size) from the checkpoint alone.

---

## 7. Honest caveats

* τ is workload- and temperature-dependent. **89 is a best-case** (greedy/low-temp/structured)
  number; creative generation will sit lower. Report a distribution, not a point.
* 2-bit on a 31B is a real quality risk; the mixed-precision + sensitivity work (M4) is what makes it
  safe, and the gate must hold before it's a default.
* No 31B MTP/assistant head may exist publicly → M2 may start on prompt-lookup + self-spec (M5.3).
* Memory is not the constraint (128 GB unified); even hybrid 2-bit + NVFP4 storage (~27 GB) + KV fits.
