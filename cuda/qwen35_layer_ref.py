#!/usr/bin/env python3
# qwen35_layer_ref.py — M2 per-layer torch reference for the Qwen3.5 hybrid (qwen35).
#
# Loads ONE FULL softmax-attention layer (idx 3) and ONE GDN linear layer (idx 0) from the
# qwen35 Q4_K_M GGUF, dequantizes the weights to fp32, runs the exact HF qwen3_next math
# (Qwen3NextAttention gated path + Qwen3NextGatedDeltaNet recurrent AND chunked delta rule),
# and dumps {dequantized weights, a deterministic input hidden state, the reference mixer
# outputs} to a flat little-endian fp32 binary that cuda/test_qwen35_layer_parity.cu reads.
#
# The "mixer output" compared in M2 is the sub-block result that gets added to the residual
# stream: for FULL = o_proj(sigmoid(gate) * GQA(q,k,v)); for GDN = out_proj(gated_rmsnorm(
# delta_rule(...), z)). The pre-mixer RMSNorm (attn_norm) is included so the reference takes
# the raw residual-stream hidden as input, exactly like the kernel under test.
#
# Conventions confirmed against llama.cpp src/models/qwen35.cpp + delta-net-base.cpp:
#   * all RMSNorm gains are applied as x_norm * w  (the +1 is already baked into the GGUF).
#   * ssm_a already stores -exp(A_log); decay g = ssm_a * softplus(alpha + dt_bias).
#   * delta-rule q is scaled by 1/sqrt(head_dim) AFTER l2norm.
#   * partial NEOX RoPE on the first rotary_dim(64) dims, theta 1e7 (mrope collapses to this
#     for a single text position stream).
import sys, struct
import numpy as np
import torch
import torch.nn.functional as F
import gguf
from transformers.models.qwen3_next.modeling_qwen3_next import (
    torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule)

GGUF_PATH = sys.argv[1] if len(sys.argv) > 1 else "/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf"
OUT_PATH  = sys.argv[2] if len(sys.argv) > 2 else "/tmp/qwen35_m2_ref.bin"
FULL_L, GDN_L, N = 3, 0, 80

H, HEAD_DIM, NQ, NKV, ROT, THETA = 4096, 256, 16, 4, 64, 1e7
CONV_DIM, KEYD, VALD, NKH, NVH, SD, TSR, CK, CHUNK = 8192, 2048, 4096, 16, 32, 128, 32, 4, 64
EPS = 1e-6
torch.set_default_dtype(torch.float32)

r = gguf.GGUFReader(GGUF_PATH)
_t = {t.name: t for t in r.tensors}
def W(name):  # returns fp32 torch [out, in] (or [n] for 1-D)
    t = _t[name]
    a = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(a))

def rmsnorm(x, w, eps=EPS):  # x [..., n], w [n]; gain applied directly (no +1)
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * w

def lin(x, w):  # x [N, in], w [out, in] -> [N, out]
    return x @ w.t()

torch.manual_seed(1234)
h_full = torch.randn(N, H) * 2.0
h_gdn  = torch.randn(N, H) * 2.0

# ───────────────────────── FULL softmax-attention layer ─────────────────────────
def full_layer(h):
    x = rmsnorm(h, W(f"blk.{FULL_L}.attn_norm.weight"))
    qg = lin(x, W(f"blk.{FULL_L}.attn_q.weight")).view(N, NQ, HEAD_DIM * 2)
    q  = qg[:, :, :HEAD_DIM].contiguous()        # [N,16,256]
    gate = qg[:, :, HEAD_DIM:].reshape(N, NQ * HEAD_DIM)  # [N,4096]
    q = rmsnorm(q, W(f"blk.{FULL_L}.attn_q_norm.weight"))
    k = lin(x, W(f"blk.{FULL_L}.attn_k.weight")).view(N, NKV, HEAD_DIM)
    k = rmsnorm(k, W(f"blk.{FULL_L}.attn_k_norm.weight"))
    v = lin(x, W(f"blk.{FULL_L}.attn_v.weight")).view(N, NKV, HEAD_DIM)
    # partial NEOX rope, first ROT dims, positions 0..N-1
    pos = torch.arange(N).float()
    inv = THETA ** (-torch.arange(0, ROT, 2).float() / ROT)   # [32]
    ang = torch.outer(pos, inv)                               # [N,32]
    cos = torch.cat([ang.cos(), ang.cos()], -1)              # [N,64]
    sin = torch.cat([ang.sin(), ang.sin()], -1)
    def rope(t):  # t [N, heads, 256]
        tr = t[..., :ROT]
        x1, x2 = tr[..., :ROT // 2], tr[..., ROT // 2:]
        rot = torch.cat([-x2, x1], -1)
        out = t.clone()
        out[..., :ROT] = tr * cos[:, None, :] + rot * sin[:, None, :]
        return out
    q, k = rope(q), rope(k)
    # GQA causal softmax, scale 1/sqrt(256)
    grp = NQ // NKV
    scale = 1.0 / (HEAD_DIM ** 0.5)
    out = torch.zeros(N, NQ, HEAD_DIM)
    cmask = torch.tril(torch.ones(N, N))
    for hd in range(NQ):
        kv = hd // grp
        sc = (q[:, hd, :] @ k[:, kv, :].t()) * scale          # [N,N]
        sc = sc.masked_fill(cmask == 0, float("-inf"))
        p = torch.softmax(sc, -1)
        out[:, hd, :] = p @ v[:, kv, :]
    attn = out.reshape(N, NQ * HEAD_DIM)
    attn = attn * torch.sigmoid(gate)
    return lin(attn, W(f"blk.{FULL_L}.attn_output.weight"))

# ───────────────────────── GDN linear layer ─────────────────────────
def gdn_proj(h):
    x = rmsnorm(h, W(f"blk.{GDN_L}.attn_norm.weight"))
    qkv = lin(x, W(f"blk.{GDN_L}.attn_qkv.weight"))           # [N,8192]
    z   = lin(x, W(f"blk.{GDN_L}.attn_gate.weight"))          # [N,4096]
    b   = lin(x, W(f"blk.{GDN_L}.ssm_beta.weight"))           # [N,32]
    a   = lin(x, W(f"blk.{GDN_L}.ssm_alpha.weight"))          # [N,32]
    # causal depthwise conv1d k=4 over channels, then silu
    cw = W(f"blk.{GDN_L}.ssm_conv1d.weight")                  # [8192,4] channel-major
    xpad = F.pad(qkv.t(), (CK - 1, 0))                        # [8192, N+3]
    conv = torch.zeros(CONV_DIM, N)
    for j in range(CK):
        conv += cw[:, j:j + 1] * xpad[:, j:j + N]
    conv = F.silu(conv).t()                                  # [N,8192]
    q = conv[:, :KEYD].reshape(N, NKH, SD)
    k = conv[:, KEYD:2 * KEYD].reshape(N, NKH, SD)
    v = conv[:, 2 * KEYD:].reshape(N, NVH, SD)
    q = q.repeat_interleave(NVH // NKH, dim=1)               # [N,32,128]
    k = k.repeat_interleave(NVH // NKH, dim=1)
    beta = torch.sigmoid(b)                                  # [N,32]
    g = W(f"blk.{GDN_L}.ssm_a") * F.softplus(a + W(f"blk.{GDN_L}.ssm_dt.bias"))  # [N,32]
    return x, z, q, k, v, beta, g

def gdn_finish(core, z):
    z = z.reshape(N, NVH, SD)
    out = rmsnorm(core, W(f"blk.{GDN_L}.ssm_norm.weight")) * F.silu(z)   # [N,32,128]
    out = out.reshape(N, VALD)
    return lin(out, W(f"blk.{GDN_L}.ssm_out.weight"))

def gdn_layer(h, kernel):
    x, z, q, k, v, beta, g = gdn_proj(h)
    qf = q.unsqueeze(0); kf = k.unsqueeze(0); vf = v.unsqueeze(0)
    gf = g.unsqueeze(0); bf = beta.unsqueeze(0)
    core, _ = kernel(qf, kf, vf, g=gf, beta=bf, initial_state=None,
                     output_final_state=False, use_qk_l2norm_in_kernel=True)
    core = core.squeeze(0)                                   # [N,32,128]
    return gdn_finish(core, z)

ref_full        = full_layer(h_full)
ref_gdn_recur   = gdn_layer(h_gdn, torch_recurrent_gated_delta_rule)
ref_gdn_chunk   = gdn_layer(h_gdn, torch_chunk_gated_delta_rule)
print(f"[ref] torch recurrent-vs-chunk max-abs diff = "
      f"{(ref_gdn_recur - ref_gdn_chunk).abs().max().item():.3e}")

# ───────────────────────── dump binary ─────────────────────────
def f32(t): return np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32))
buf = bytearray()
buf += struct.pack("<iii", 0x51573532, N, H)
def put(t): buf.extend(f32(t).tobytes())
# FULL block
put(W(f"blk.{FULL_L}.attn_norm.weight"))
put(W(f"blk.{FULL_L}.attn_q.weight")); put(W(f"blk.{FULL_L}.attn_k.weight"))
put(W(f"blk.{FULL_L}.attn_v.weight")); put(W(f"blk.{FULL_L}.attn_output.weight"))
put(W(f"blk.{FULL_L}.attn_q_norm.weight")); put(W(f"blk.{FULL_L}.attn_k_norm.weight"))
put(h_full); put(ref_full)
# GDN block
put(W(f"blk.{GDN_L}.attn_norm.weight"))
put(W(f"blk.{GDN_L}.attn_qkv.weight")); put(W(f"blk.{GDN_L}.attn_gate.weight"))
put(W(f"blk.{GDN_L}.ssm_beta.weight")); put(W(f"blk.{GDN_L}.ssm_alpha.weight"))
put(W(f"blk.{GDN_L}.ssm_conv1d.weight"))            # [8192,4] channel-major
put(W(f"blk.{GDN_L}.ssm_a")); put(W(f"blk.{GDN_L}.ssm_dt.bias"))
put(W(f"blk.{GDN_L}.ssm_norm.weight")); put(W(f"blk.{GDN_L}.ssm_out.weight"))
put(h_gdn); put(ref_gdn_recur); put(ref_gdn_chunk)
with open(OUT_PATH, "wb") as fh:
    fh.write(buf)
print(f"[ref] wrote {OUT_PATH} ({len(buf)} bytes), N={N}")
