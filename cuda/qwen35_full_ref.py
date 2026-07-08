#!/usr/bin/env python3
# qwen35_full_ref.py — full-model torch reference for the qwen35 hybrid over the fixed prompt.
# Dequantizes the GGUF, runs all layers (FULL output-gated softmax-GQA + GDN linear) over the
# prompt token ids, and prints the greedy next-token argmax after the last prompt token plus a
# per-layer hidden-norm trace. Used to localize the M3 fucina forward divergence.
import sys, numpy as np, torch, torch.nn.functional as F, gguf
from transformers.models.qwen3_next.modeling_qwen3_next import torch_recurrent_gated_delta_rule

GG = sys.argv[1] if len(sys.argv) > 1 else "/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf"
PROMPT = [760, 6511, 314, 9338, 369]
H, HEAD_DIM, NQ, NKV, ROT, THETA = 4096, 256, 16, 4, 64, 1e7
CONV_DIM, KEYD, VALD, NKH, NVH, SD, TSR, CK = 8192, 2048, 4096, 16, 32, 128, 32, 4
EPS = 1e-6
torch.set_default_dtype(torch.float32)

r = gguf.GGUFReader(GG)
_t = {t.name: t for t in r.tensors}
def W(name):
    t = _t[name]
    a = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(a))
def rms(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
def lin(x, w):
    return x @ w.t()

nl = 32
attn_kind = ["FULL" if (i + 1) % 4 == 0 else "LINEAR" for i in range(nl)]

# embedding (token_embd rows for the prompt)
emb = W("token_embd.weight")  # [vocab, H]
N = len(PROMPT)
h = emb[torch.tensor(PROMPT)].clone()  # [N, H]
print("embed norm:", h.norm(dim=-1).tolist())

def full_layer(h, l):
    x = rms(h, W(f"blk.{l}.attn_norm.weight"))
    qg = lin(x, W(f"blk.{l}.attn_q.weight")).view(N, NQ, HEAD_DIM * 2)
    q = qg[:, :, :HEAD_DIM].contiguous()
    gate = qg[:, :, HEAD_DIM:].reshape(N, NQ * HEAD_DIM)
    q = rms(q, W(f"blk.{l}.attn_q_norm.weight"))
    k = lin(x, W(f"blk.{l}.attn_k.weight")).view(N, NKV, HEAD_DIM)
    k = rms(k, W(f"blk.{l}.attn_k_norm.weight"))
    v = lin(x, W(f"blk.{l}.attn_v.weight")).view(N, NKV, HEAD_DIM)
    pos = torch.arange(N).float()
    inv = THETA ** (-torch.arange(0, ROT, 2).float() / ROT)
    ang = torch.outer(pos, inv)
    cos = torch.cat([ang.cos(), ang.cos()], -1); sin = torch.cat([ang.sin(), ang.sin()], -1)
    def rope(t):
        tr = t[..., :ROT]; x1, x2 = tr[..., :ROT // 2], tr[..., ROT // 2:]
        rot = torch.cat([-x2, x1], -1); out = t.clone()
        out[..., :ROT] = tr * cos[:, None, :] + rot * sin[:, None, :]; return out
    q, k = rope(q), rope(k)
    grp = NQ // NKV; scale = 1.0 / (HEAD_DIM ** 0.5)
    out = torch.zeros(N, NQ, HEAD_DIM); cmask = torch.tril(torch.ones(N, N))
    for hd in range(NQ):
        kv = hd // grp
        sc = (q[:, hd, :] @ k[:, kv, :].t()) * scale
        sc = sc.masked_fill(cmask == 0, float("-inf"))
        out[:, hd, :] = torch.softmax(sc, -1) @ v[:, kv, :]
    attn = out.reshape(N, NQ * HEAD_DIM) * torch.sigmoid(gate)
    return lin(attn, W(f"blk.{l}.attn_output.weight"))

def pp(tag, t):  # print first3/last3 of token 0
    v = t.reshape(N, -1)[0]
    print(f"  {tag}: [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}, ..., {v[-3]:.4f}, {v[-2]:.4f}, {v[-1]:.4f}]")

def gdn_layer(h, l):
    x = rms(h, W(f"blk.{l}.attn_norm.weight"))
    if l == 0: pp("attn_norm-0", x)
    qkv = lin(x, W(f"blk.{l}.attn_qkv.weight"))
    z = lin(x, W(f"blk.{l}.attn_gate.weight"))
    b = lin(x, W(f"blk.{l}.ssm_beta.weight")); a = lin(x, W(f"blk.{l}.ssm_alpha.weight"))
    if l == 0: pp("alpha-0", a); pp("z-0", z)
    cw = W(f"blk.{l}.ssm_conv1d.weight")
    xpad = F.pad(qkv.t(), (CK - 1, 0)); conv = torch.zeros(CONV_DIM, N)
    for j in range(CK):
        conv += cw[:, j:j + 1] * xpad[:, j:j + N]
    conv = F.silu(conv).t()
    if l == 0: pp("conv_silu-0(tok0)", conv)
    q = conv[:, :KEYD].reshape(N, NKH, SD); k = conv[:, KEYD:2 * KEYD].reshape(N, NKH, SD)
    v = conv[:, 2 * KEYD:].reshape(N, NVH, SD)
    # q/k (16 heads) expand to the 32 v-heads by TILING (HF repeat, v-head vh ↔ k/q-head
    # vh % NKH) — NOT repeat_interleave. Verified token-for-token vs llama.cpp GATED_DELTA_NET.
    q = q.repeat(1, NVH // NKH, 1); k = k.repeat(1, NVH // NKH, 1)
    beta = torch.sigmoid(b)
    g = W(f"blk.{l}.ssm_a") * F.softplus(a + W(f"blk.{l}.ssm_dt.bias"))
    core, _ = torch_recurrent_gated_delta_rule(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
              g=g.unsqueeze(0), beta=beta.unsqueeze(0), initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=True)
    core = core.squeeze(0)
    if l == 0:
        print(f"  core(tok0,head0): [{core[0,0,0]:.4f}, {core[0,0,1]:.4f}, {core[0,0,2]:.4f}, ..., {core[0,0,-1]:.4f}]")
        print(f"  core(tok0,head1): [{core[0,1,0]:.4f}, {core[0,1,1]:.4f}, {core[0,1,2]:.4f}, ..., {core[0,1,-1]:.4f}]")
    out = rms(core, W(f"blk.{l}.ssm_norm.weight")) * F.silu(z.reshape(N, NVH, SD))
    lao = lin(out.reshape(N, VALD), W(f"blk.{l}.ssm_out.weight"))
    if l == 0: pp("linear_attn_out-0", lao)
    return lao

for l in range(nl):
    mix = full_layer(h, l) if attn_kind[l] == "FULL" else gdn_layer(h, l)
    h = h + mix
    x2 = rms(h, W(f"blk.{l}.post_attention_norm.weight"))
    g = lin(x2, W(f"blk.{l}.ffn_gate.weight")); u = lin(x2, W(f"blk.{l}.ffn_up.weight"))
    h = h + lin(F.silu(g) * u, W(f"blk.{l}.ffn_down.weight"))
    print(f"L{l:2d} {attn_kind[l]:6s} h.norm(last)={h[-1].norm().item():.3f}")

hn = rms(h, W("output_norm.weight"))
logits = lin(hn[-1], W("output.weight"))
top = torch.topk(logits, 5)
print("final top5 ids:", top.indices.tolist())
print("final top5 val:", [round(v, 3) for v in top.values.tolist()])
print("ARGMAX:", int(logits.argmax()))
