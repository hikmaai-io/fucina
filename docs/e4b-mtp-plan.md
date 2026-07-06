# E4B MTP / speculative decode — implementation plan

Goal: ~2× E4B decode via the official `gemma4-assistant` draft head, greedy speculation
(lossless: greedy-spec output is **bit-identical** to greedy non-spec). Mirrors the dense
12B/31B MTP (`cuda/gemma4_kernels.cu` `mtp_forward`/`mtp_draft`/spec-verify), ported to the
E4B engine's per-slot **ring** KV.

## Draft head — exact layout (parsed from the GGUF)
`models--unsloth--gemma-4-E4B-it-qat-GGUF/.../MTP/gemma-4-E4B-it-Q4_0-MTP.gguf`
- arch `gemma4-assistant`, 78M, weights **Q4_0**, norms/scales **F32**.
- AH (assistant hidden) = **256**, FF = **2048**, layers = **4**, q_heads = 4, kv_heads = 2.
- layer types from `sliding_window_pattern [1,1,1,0]`: blk.0/1/2 = **sliding** (head_dim 256,
  rope θ=1e4), blk.3 = **global** (head_dim 512, rope θ=1e6). `attn_q_norm` width confirms
  (256 for sliding, 512 for global), same as the dense load heuristic.
- `embedding_length_out = 2560` (= target hidden H), `sliding_window = 512`, rms_eps 1e-6.

Tensors (ggml `[ne0=in, ne1=out]` == HF `[out,in]` row-major):
- `nextn.pre_projection.weight  [5120,256]`  : pre_proj  5120→256  (in = concat(embed,h))
- `nextn.post_projection.weight [256,2560]`  : post_proj 256→2560  (next recurrent h)
- `token_embd.weight            [256,262144]`: assistant **unembed** AH→VOCAB
- per layer blk.l: `attn_norm[256]`, `attn_q[AH→qdim]` (qdim=1024 l<3 / 2048 l=3),
  `attn_q_norm[head_dim]`, `attn_output[qdim→AH]`, `post_attention_norm[256]`,
  `ffn_norm[256]`, `ffn_gate[AH→FF]`, `ffn_up[AH→FF]`, `ffn_down[FF→AH]`,
  `post_ffw_norm[256]`, `layer_output_scale[1]` (F32 out_scale).
- `output_norm[256]`, `rope_freqs[256]`.
- **No K/V projections** (Q-only): the drafter attends the TARGET's K/V.

## Drafter forward (per draft token), porting dense `mtp_forward`
Inputs: token id `tok`, recurrent `h` (2560, the target post-final-norm hidden of the
preceding token), absolute position `pos`.
1. `xh[0:2560] = embed_target(tok)·√2560` (E4B `d_embed` lookup + scale); `xh[2560:5120]=h`.
2. `cur[256] = pre_proj · xh`.
3. for l in 0..3 (is_global = l==3):
   - `t1 = rmsnorm(cur, attn_norm[l])`; `q = attn_q[l]·t1`; `head_rmsnorm(q, q_norm[l], hd)`;
     `rope(q, pos, θ)`.
   - **attention (Q-only)** vs the TARGET slot-0 cache of the provider layer:
     sliding (l<3) → `kc/vc[prov_sliding]`, window=sliding_window-1=511, cap=sliding_cap;
     global (l=3) → `kc/vc[prov_full]`, window=0, cap=max_ctx.
     Reuse `attn_flash_decode_kernel` with n_heads=4, n_kv=2 (GQA group=2), hd, scaling=1,
     reading the device position. (prov_sliding=22, prov_full=23 for the 5:1 / 18-shared E4B.)
   - `t1 = attn_output[l]·attn`; `t2 = rmsnorm(t1, post_attn_norm[l]) + cur`.
   - FFN GeGLU: `t1=rmsnorm(t2,ffn_norm[l])`; `g=gate·t1, u=up·t1`; `geglu`; `t1=down·`;
     `cur = rmsnorm(t1, post_ffw_norm[l]) + t2`; `cur *= layer_output_scale[l]`.
4. `t1 = rmsnorm(cur, output_norm)`; `logits = token_embd·t1` (NO softcap); `h_next = post_proj·t1`.
5. draft = argmax(logits); chain (tok←draft, h←h_next, pos++).

All kernels already exist in `e4b_engine.cu` (rmsnorm_kernel, head_rmsnorm_kernel, rope_kernel,
attn_flash_decode_kernel, geglu_kernel, add_kernel, scale_kernel, `linear`). v1 dequants the
Q4_0 weights → BF16 at load (`e4bgguf::gguf_dequant_to_bf16`, like `up_bf16_gguf`) and uses
cuBLAS `linear` GEMVs (drafter is tiny; optimize to dp4a later).

## Verify (the one mandatory engine change)
`e4b_step` emits only the last row's logits (`xrow=d_norm+(T-1)*H`, e4b_engine.cu:~1367).
Verify needs per-row argmax over `[g, draft0..draftD-1]` (T=D+1). Add an all-rows head:
project `d_norm[0..T-1]` → `[T,V]` logits (reuse the **batched** head already used by
`e4b_step_batch_decode`: `mmvq_q6_k_batched_launch` for use_q40 / batched `linear` else),
softcap, argmax each row on device, D2H the T argmax ids. The candidate K/V are written by the
normal T>1 path (`kv_store_kernel` + `attn_cache_kernel`, ring-correct), so verify == running
e4b_step over the candidates + reading per-row argmax.

Accept (greedy): `a=0; while(a<D && argmax[a]==draft[a]) a++;` emit `g` + accepted drafts
(a+1 tokens). Rewind KV to `n_past_before + (a+1)` via `e4b_engine_rewind` (depth ≤ K ≪
sliding_cap-window → always safe). Recurrent h for the next step = target hidden row `min(a,D)`
of the verify forward (needs the per-row `d_norm` → expose like fin_out for that row).

## Increments (each independently verifiable)
1. **Loader** `e4b_mtp.cuh` + `e4b_engine_load_assistant(path)`: parse the assistant GGUF,
   dequant Q4_0→BF16, upload, store F32 norms/scales/rope, detect per-layer global/sliding from
   q_norm width. Verify: residency log + tensor-count/shape asserts. *(this increment now)*
2. **Drafter forward** `mtp_forward_e4b` + standalone numeric check (one step from a known h/tok).
3. **All-rows verify head** in e4b_step (or a dedicated `e4b_verify`), argmax per row.
4. **Greedy spec loop** `e4b_engine_generate_spec_greedy` (prefill → draft K → verify → accept →
   rewind → repeat). **GATE:** output bit-identical to `e4b_engine_generate_greedy`.
5. **Server wiring**: `--assistant` for E4B; adapter `GenerateSpecStream` drives the C spec loop
   with the emit callback + SpecStats; interacts cleanly with prefix-cache (rewind after accept).

## Decisive correctness gate
`make e4b-spec-test`: same prompt, `e4b_engine_generate_greedy` vs the MTP spec path, **must be
byte-identical** for ≥256 greedy tokens. (Greedy spec is lossless by construction.)

## Recommended C/Go split
Spec loop lives in **C** (`e4b_engine_generate_spec_greedy` + a per-token cgo callback), mirroring
the dense `gemma4_engine_generate_spec_stream`. Go bridge adds `LoadAssistant(path)` + the
streaming spec call; the e4bServer adapter routes `GenerateSpecStream` to it when an assistant is
loaded, else the current plain loop. Lower risk than driving draft/verify across cgo per token.
