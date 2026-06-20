#!/usr/bin/env python3
"""Quality gate for the 2-bit recipe — forward-pass perplexity on the real BF16 31B.

Track B's sensitivity sweep gave a CONDITIONAL GO; the gate it was conditioned on is a
forward-pass number, which the workflow couldn't run (no torch). This script closes that gap:
it loads the on-disk BF16 Gemma-4-31B, applies the MXFP2 fake-quant (NF2/NF3 from mxfp2.py)
per recipe by overwriting each weight with its quantize->dequantize reconstruction, and reports
perplexity for fp16 baseline vs uniform-2bit vs the LEAN recipe on a fixed calib text.

Recipes (role-based, per B2: sensitivity is role- not layer-driven):
  fp16     : no quant (baseline)
  uniform  : every quantizable matmul weight -> NF2 (2-bit)
  lean     : embed + attn_k + attn_v -> NF3 (3-bit); all other matmul weights -> NF2

HF Gemma param name -> role: embed_tokens=embed, k_proj=attn_k, v_proj=attn_v, q_proj=attn_q,
o_proj=attn_o, {gate,up,down}_proj=ffn_*. 1-D tensors (norms) are skipped. lm_head is tied to
embed_tokens, so protecting embed protects the head.

Designed to run as an unattended 5AM batch job: defensive (GPU sanity-checked first), --encoder
absmax (fast, pessimistic) by default so a PASS is a safe lower bound; --encoder mse for the
format's true floor. Run: python perplexity_gate.py --model google/gemma-4-31B-it
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mxfp2

CALIB = (
    "The development of large language models has transformed natural language processing. "
    "Quantization reduces the memory footprint of neural networks by representing weights with "
    "fewer bits, trading a small amount of accuracy for substantial savings in bandwidth and "
    "storage. On memory-bandwidth-bound hardware, decoding speed is governed by how many bytes "
    "of weights must be read for each token produced. A mixture-of-experts model activates only "
    "a fraction of its parameters per token, while a dense model must read every weight. "
    "Speculative decoding amortizes the weight read across several verified tokens, and aggressive "
    "low-bit quantization shrinks the bytes that must be read in the first place. Combining both "
    "lets a dense model approach the throughput that sparsity grants a mixture-of-experts model, "
    "provided the lost precision does not degrade the output distribution beyond a usable bound. "
) * 6  # ~1k tokens

ROLE_OF = [
    ("embed_tokens", "embed"), ("k_proj", "attn_k"), ("v_proj", "attn_v"),
    ("q_proj", "attn_q"), ("o_proj", "attn_o"),
    ("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down"),
]
def role_of(name):
    for sub, role in ROLE_OF:
        if sub in name:
            return role
    return None

def variant_for(role, recipe):
    if recipe == "fp16":
        return None
    if recipe == "uniform":
        return "nf2"
    if recipe == "lean":
        return "nf3" if role in ("embed", "attn_k", "attn_v") else "nf2"
    if recipe == "nf3":          # all-3-bit floor (the viable fallback after 2-bit NO-GO)
        return "nf3"
    if recipe == "lean3":        # FFN@3bit, embed/k/v@3bit too — i.e. 3-bit everywhere but a label for sweeps
        return "nf3"
    raise ValueError(recipe)

def fake_quant_(tensor, variant, encoder):
    import torch
    BLK = 16                      # codec block size; chunk on multiples so blocks never split
    flat = tensor.detach().to(torch.float32).cpu().numpy().reshape(-1)
    n = flat.size
    out = np.empty_like(flat)
    CHUNK = (64 * 1024 * 1024 // BLK) * BLK   # ~64M elems/chunk → bounded transient (~GB, not 20GB)
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        w = flat[s:e]
        if encoder == "mse":
            idx, scale, nn = mxfp2.quantize_mse(w, variant)
            out[s:e] = mxfp2.dequantize(idx, scale, nn, variant)
        else:
            out[s:e] = mxfp2.roundtrip(w, variant)
    del flat
    rec = torch.from_numpy(out.reshape(tensor.shape))
    with torch.no_grad():   # in-place copy_ onto a leaf Parameter (requires_grad) needs no_grad
        tensor.copy_(rec.to(tensor.dtype).to(tensor.device))
    del rec, out

def perplexity(model, input_ids):
    import torch
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
    return float(torch.exp(out.loss).item())

def run_recipe(model_id, recipe, tokenizer, input_ids, encoder, dtype, device, log):
    import torch
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    model.eval()
    nq = 0; ne = 0
    if recipe != "fp16":
        for name, p in model.named_parameters():
            if p.ndim < 2:
                continue
            role = role_of(name)
            if role is None:
                continue
            v = variant_for(role, recipe)
            fake_quant_(p, v, encoder)
            nq += 1; ne += p.numel()
    ppl = perplexity(model, input_ids)
    log(f"  [{recipe}] ppl={ppl:.4f}  quantized {nq} tensors / {ne/1e9:.2f}G elems  "
        f"({time.time()-t0:.0f}s)")
    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    return ppl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--encoder", choices=["absmax", "mse"], default="absmax")
    ap.add_argument("--recipes", default="fp16,uniform,lean")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    def log(m): print(m, flush=True)
    log(f"== 2-bit perplexity gate ==  model={args.model} encoder={args.encoder}")

    try:
        import torch
        from transformers import AutoTokenizer
    except Exception as e:
        log(f"FATAL: torch/transformers import failed: {e}")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  device={device}")
    if device == "cuda":
        try:
            x = torch.randn(64, 64, device="cuda"); _ = (x @ x).sum().item()
            log(f"  GPU sanity matmul OK on {torch.cuda.get_device_name(0)} "
                f"(cap {torch.cuda.get_device_capability(0)})")
        except Exception as e:
            log(f"FATAL: GPU matmul failed (likely sm_121/CUDA13 wheel gap): {e}")
            return 3
    else:
        log("WARNING: no CUDA — 31B on CPU is impractical; aborting to avoid a multi-hour hang.")
        return 4

    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(CALIB, return_tensors="pt").input_ids[:, : args.max_tokens].to(device)
    log(f"calib tokens={ids.shape[1]}")

    results = {}
    for r in args.recipes.split(","):
        try:
            results[r] = run_recipe(args.model, r, tok, ids, args.encoder, dtype, device, log)
        except Exception as e:
            log(f"  [{r}] FAILED: {e}")
            results[r] = None

    base = results.get("fp16")
    log("\n== VERDICT ==")
    for r, p in results.items():
        if p is None:
            log(f"  {r}: FAILED"); continue
        d = f"  (Δppl {p-base:+.4f}, {100*(p-base)/base:+.1f}%)" if base and r != "fp16" else ""
        log(f"  {r:8s} ppl={p:.4f}{d}")
    if base and results.get("lean"):
        rel = (results["lean"] - base) / base
        verdict = "GO" if rel < 0.05 else ("MARGINAL" if rel < 0.12 else "NO-GO")
        log(f"\n  LEAN vs fp16: {100*rel:+.1f}%  ->  {verdict} "
            f"(GO<5%, MARGINAL<12%; encoder={args.encoder} is "
            f"{'a lower bound — true recipe is better' if args.encoder=='absmax' else 'the format floor'})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
