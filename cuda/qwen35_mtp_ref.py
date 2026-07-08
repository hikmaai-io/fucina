#!/usr/bin/env python3
# qwen35_mtp_ref.py — torch oracle for the Qwen3.5-9B single-MTP draft head (M6).
#
# Extends cuda/qwen35_fp8_ref.py (same FP8 checkpoint, same hybrid backbone math) with the
# 22 mtp.* tensors and the vLLM-authoritative MTP forward (qwen3_5_mtp.py):
#   inputs_embeds = embed(input_ids); inputs_embeds = pre_fc_norm_embedding(inputs_embeds)
#   hidden        = pre_fc_norm_hidden(h_prev)            # h_prev = main model POST-final-norm hidden
#   x   = fc( cat([inputs_embeds, hidden], -1) )          # embedding FIRST -> [2H]->[H], no bias
#   one Qwen3_5DecoderLayer (FULL attention, same shape as a backbone full layer) on x
#   draft = argmax( lm_head( mtp.norm( residual_stream ) ) )
#
# Purpose: (1) lock the exact MTP math the CUDA kernels must match, (2) measure the natural
# greedy accept-rate of the MTP drafts vs the real backbone continuation, for both a FULL-context
# MTP KV cache and a draft-CHAIN-only KV cache (decides the fucina implementation complexity).
import sys, json, struct, glob, os
import numpy as np, torch, torch.nn.functional as F
from transformers.models.qwen3_next.modeling_qwen3_next import torch_recurrent_gated_delta_rule

DIR = sys.argv[1] if len(sys.argv) > 1 else "/opt/spark/models/models--Qwen--Qwen3.5-9B-FP8"
NGEN = int(sys.argv[2]) if len(sys.argv) > 2 else 24
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 4
PROMPT_TXT = "The capital of France is"

H, HEAD_DIM, NQ, NKV, ROT, THETA = 4096, 256, 16, 4, 64, 1e7
CONV_DIM, KEYD, VALD, NKH, NVH, SD, TSR, CK = 8192, 2048, 4096, 16, 32, 128, 32, 4
EPS = 1e-6
LM = "model.language_model."
torch.set_default_dtype(torch.float32)

shards = sorted(glob.glob(os.path.join(DIR, "model.safetensors-*.safetensors")))
HDR = {}
for f in shards:
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    dstart = 8 + n
    for k, v in h.items():
        if k == "__metadata__":
            continue
        HDR[k] = (f, v["dtype"], v["shape"], v["data_offsets"][0], v["data_offsets"][1], dstart)

def raw(name):
    f, dt, shape, b, e, ds = HDR[name]
    with open(f, "rb") as fh:
        fh.seek(ds + b); buf = fh.read(e - b)
    return dt, shape, buf

def as_tensor(name):
    dt, shape, buf = raw(name)
    if dt == "F32":
        a = np.frombuffer(buf, dtype=np.float32).reshape(shape).copy(); return torch.from_numpy(a).float()
    if dt == "BF16":
        a = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32).reshape(shape)
        return torch.from_numpy((a << 16).view(np.float32).copy()).float()
    if dt == "F8_E4M3":
        t = torch.frombuffer(bytearray(buf), dtype=torch.uint8).view(torch.float8_e4m3fn)
        return t.reshape(shape).float()
    raise RuntimeError(f"unhandled dtype {dt} for {name}")

_WC = {}
def W(name):
    if name in _WC: return _WC[name]
    sname = name + "_scale_inv"; w = as_tensor(name)
    if sname in HDR:
        scale = as_tensor(sname); O, I = w.shape
        sf = scale.repeat_interleave(128, 0)[:O].repeat_interleave(128, 1)[:, :I]
        w = w * sf
    _WC[name] = w; return w

def rms(x, w, eps=EPS):  return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)
def rmsg(x, w, eps=EPS): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
def lin(x, w): return x @ w.t()

nl = 32
attn_kind = ["FULL" if (i + 1) % 4 == 0 else "LINEAR" for i in range(nl)]

def rope_cossin(positions):
    pos = positions.float()
    inv = THETA ** (-torch.arange(0, ROT, 2).float() / ROT)
    ang = torch.outer(pos, inv)
    return torch.cat([ang.cos(), ang.cos()], -1), torch.cat([ang.sin(), ang.sin()], -1)

def rope_apply(t, cos, sin):  # t [N, heads, HEAD_DIM]
    tr = t[..., :ROT]; x1, x2 = tr[..., :ROT // 2], tr[..., ROT // 2:]
    rot = torch.cat([-x2, x1], -1); out = t.clone()
    out[..., :ROT] = tr * cos[:, None, :] + rot * sin[:, None, :]; return out

def full_layer(h, l):
    N = h.shape[0]; p = f"{LM}layers.{l}."
    x = rms(h, W(p + "input_layernorm.weight"))
    qg = lin(x, W(p + "self_attn.q_proj.weight")).view(N, NQ, HEAD_DIM * 2)
    q = qg[:, :, :HEAD_DIM].contiguous(); gate = qg[:, :, HEAD_DIM:].reshape(N, NQ * HEAD_DIM)
    q = rms(q, W(p + "self_attn.q_norm.weight"))
    k = lin(x, W(p + "self_attn.k_proj.weight")).view(N, NKV, HEAD_DIM)
    k = rms(k, W(p + "self_attn.k_norm.weight"))
    v = lin(x, W(p + "self_attn.v_proj.weight")).view(N, NKV, HEAD_DIM)
    cos, sin = rope_cossin(torch.arange(N))
    q, k = rope_apply(q, cos, sin), rope_apply(k, cos, sin)
    grp = NQ // NKV; scale = 1.0 / (HEAD_DIM ** 0.5)
    out = torch.zeros(N, NQ, HEAD_DIM); cmask = torch.tril(torch.ones(N, N))
    for hd in range(NQ):
        kv = hd // grp
        sc = (q[:, hd, :] @ k[:, kv, :].t()) * scale
        sc = sc.masked_fill(cmask == 0, float("-inf"))
        out[:, hd, :] = torch.softmax(sc, -1) @ v[:, kv, :]
    attn = out.reshape(N, NQ * HEAD_DIM) * torch.sigmoid(gate)
    return lin(attn, W(p + "self_attn.o_proj.weight"))

def gdn_layer(h, l):
    N = h.shape[0]; p = f"{LM}layers.{l}."
    x = rms(h, W(p + "input_layernorm.weight"))
    qkv = lin(x, W(p + "linear_attn.in_proj_qkv.weight"))
    z   = lin(x, W(p + "linear_attn.in_proj_z.weight"))
    a   = lin(x, W(p + "linear_attn.in_proj_a.weight"))
    b   = lin(x, W(p + "linear_attn.in_proj_b.weight"))
    cw  = W(p + "linear_attn.conv1d.weight").reshape(CONV_DIM, CK)
    xpad = F.pad(qkv.t(), (CK - 1, 0)); conv = torch.zeros(CONV_DIM, N)
    for j in range(CK): conv += cw[:, j:j + 1] * xpad[:, j:j + N]
    conv = F.silu(conv).t()
    q = conv[:, :KEYD].reshape(N, NKH, SD); k = conv[:, KEYD:2 * KEYD].reshape(N, NKH, SD)
    v = conv[:, 2 * KEYD:].reshape(N, NVH, SD)
    q = q.repeat_interleave(NVH // NKH, dim=1); k = k.repeat_interleave(NVH // NKH, dim=1)
    beta = torch.sigmoid(b)
    A_log = W(p + "linear_attn.A_log"); dt = W(p + "linear_attn.dt_bias")
    g = -torch.exp(A_log) * F.softplus(a + dt)
    core, _ = torch_recurrent_gated_delta_rule(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
              g=g.unsqueeze(0), beta=beta.unsqueeze(0), initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=True)
    core = core.squeeze(0)
    out = rmsg(core, W(p + "linear_attn.norm.weight")) * F.silu(z.reshape(N, NVH, SD))
    return lin(out.reshape(N, VALD), W(p + "linear_attn.out_proj.weight"))

EMB = W(LM + "embed_tokens.weight"); ONORM = W(LM + "norm.weight"); LMH = W("lm_head.weight")

def backbone_hidden(ids):
    """Return per-position POST-final-norm hidden hn[N,H] for the full sequence ids."""
    h = EMB[torch.tensor(ids)].clone()
    for l in range(nl):
        mix = full_layer(h, l) if attn_kind[l] == "FULL" else gdn_layer(h, l)
        h = h + mix
        p = f"{LM}layers.{l}."
        x2 = rms(h, W(p + "post_attention_layernorm.weight"))
        gg = lin(x2, W(p + "mlp.gate_proj.weight")); uu = lin(x2, W(p + "mlp.up_proj.weight"))
        h = h + lin(F.silu(gg) * uu, W(p + "mlp.down_proj.weight"))
    return rms(h, ONORM)  # [N,H]

# ── MTP head ──
MP = "mtp."
def mtp_fc(token_id, h_prev):
    e = EMB[torch.tensor([token_id])].clone()                       # [1,H]
    e = rms(e, W(MP + "pre_fc_norm_embedding.weight"))
    hp = rms(h_prev.view(1, H), W(MP + "pre_fc_norm_hidden.weight"))
    x = torch.cat([e, hp], -1)                                      # [1,2H] embedding FIRST
    return lin(x, W(MP + "fc.weight"))                              # [1,H]

def mtp_layer_kv(x_fc):
    """K/V (+ q,gate,residual prologue) for one MTP token from the fc-fused hidden x_fc[1,H]."""
    p = MP + "layers.0."
    xn = rms(x_fc, W(p + "input_layernorm.weight"))
    qg = lin(xn, W(p + "self_attn.q_proj.weight")).view(1, NQ, HEAD_DIM * 2)
    q = qg[:, :, :HEAD_DIM].contiguous(); gate = qg[:, :, HEAD_DIM:].reshape(1, NQ * HEAD_DIM)
    q = rms(q, W(p + "self_attn.q_norm.weight"))
    k = lin(xn, W(p + "self_attn.k_proj.weight")).view(1, NKV, HEAD_DIM)
    k = rms(k, W(p + "self_attn.k_norm.weight"))
    v = lin(xn, W(p + "self_attn.v_proj.weight")).view(1, NKV, HEAD_DIM)
    return q, k, v, gate, x_fc  # residual = x_fc

def mtp_finish(q, gate, residual, Kc, Vc, pos):
    """Given q[1,NQ,HD], the accumulated Kc/Vc lists [pos+1][NKV,HD], finish the MTP layer + head."""
    p = MP + "layers.0."
    cos, sin = rope_cossin(torch.tensor([pos]))
    qr = rope_apply(q, cos, sin)
    K = torch.stack(Kc, 0)  # [T,NKV,HD]
    V = torch.stack(Vc, 0)
    grp = NQ // NKV; scale = 1.0 / (HEAD_DIM ** 0.5)
    out = torch.zeros(1, NQ, HEAD_DIM)
    for hd in range(NQ):
        kv = hd // grp
        sc = (qr[:, hd, :] @ K[:, kv, :].t()) * scale          # [1,T] causal (q is newest)
        out[:, hd, :] = torch.softmax(sc, -1) @ V[:, kv, :]
    attn = out.reshape(1, NQ * HEAD_DIM) * torch.sigmoid(gate)
    h = residual + lin(attn, W(p + "self_attn.o_proj.weight"))
    x2 = rms(h, W(p + "post_attention_layernorm.weight"))
    gg = lin(x2, W(p + "mlp.gate_proj.weight")); uu = lin(x2, W(p + "mlp.up_proj.weight"))
    h = h + lin(F.silu(gg) * uu, W(p + "mlp.down_proj.weight"))
    hn = rms(h, W(MP + "norm.weight"))
    return int(lin(hn, LMH).argmax()), h.view(H)  # draft token + residual stream (re-fed as h_prev)

# tokenize
try:
    from transformers import AutoTokenizer
    prompt_ids = AutoTokenizer.from_pretrained(DIR)(PROMPT_TXT, add_special_tokens=False)["input_ids"]
except Exception as ex:
    print("tok fallback:", ex, file=sys.stderr); prompt_ids = [760, 6511, 314, 9338, 369]
print("prompt ids:", prompt_ids)

# Greedy continuation (the real backbone target) + per-step MTP accept measurement.
ids = list(prompt_ids)
with torch.no_grad():
    # Generate the real continuation first (greedy) so we know the target tokens.
    full = list(ids)
    for _ in range(NGEN + DEPTH + 2):
        hn = backbone_hidden(full)
        full.append(int(lin(hn[-1], LMH).argmax()))
    target = full[len(ids):]
    print("CONT_IDS =", target[:NGEN])

    # For each decode point t (0..NGEN-1), the committed token is full[len(ids)+t-1]... use prefix.
    def build_mtp_kv_full(seq):
        """Persistent MTP KV over the whole sequence seq[0..M-1]; entry j uses input seq[j], h_prev=hn[j-1]."""
        hn = backbone_hidden(seq); Kc = []; Vc = []; res = []; Q = []; Gate = []
        for j in range(len(seq)):
            hp = hn[j - 1] if j >= 1 else torch.zeros(H)
            q, k, v, gate, r = mtp_layer_kv(mtp_fc(seq[j], hp))
            # store ROPE'd K at absolute position j
            cos, sin = rope_cossin(torch.tensor([j])); k = rope_apply(k, cos, sin)
            Kc.append(k[0]); Vc.append(v[0]); res.append(r); Q.append(q); Gate.append(gate)
        return hn, Kc, Vc

    # Measure accept@depth for FULL-context vs CHAIN-only MTP attention.
    for mode in ("full", "chain"):
        acc = np.zeros(DEPTH + 1)  # acc[d] = #points whose draft prefix length >= d
        npts = 0
        for t in range(NGEN):
            seq = list(ids) + target[:t]      # committed prefix; last token = seq[-1]
            M = len(seq)
            hn = backbone_hidden(seq)
            # draft depth-D from the end of seq
            if mode == "full":
                _, Kc, Vc = build_mtp_kv_full(seq)
            else:
                Kc, Vc = [], []
            drafts = []
            hp = hn[-1]                         # main hidden of the last committed token
            tok = seq[-1]
            for d in range(DEPTH):
                pos = M + d                     # MTP predicts token at this position's next
                q, k, v, gate, r = mtp_layer_kv(mtp_fc(tok, hp))
                cos, sin = rope_cossin(torch.tensor([pos])); kk = rope_apply(k, cos, sin)
                Kc2 = Kc + [kk[0]]; Vc2 = Vc + [v[0]]
                dft, mtp_hid = mtp_finish(q, gate, r, Kc2, Vc2, pos)
                drafts.append(dft)
                Kc, Vc = Kc2, Vc2
                tok = dft
                hp = mtp_hid                    # re-feed the MTP residual-stream hidden (vLLM convention)
            # compare drafts to the real target continuation after position M-1
            real = target[t:t + DEPTH]
            a = 0
            while a < DEPTH and a < len(real) and drafts[a] == real[a]: a += 1
            for d in range(a + 1): acc[d] += 1
            npts += 1
        print(f"[{mode}] points={npts} accept@depth:",
              " ".join(f"{d}:{acc[d]/npts:.2f}" for d in range(DEPTH + 1)),
              " mean_accepted=%.3f" % (sum(acc[1:]) / npts))
