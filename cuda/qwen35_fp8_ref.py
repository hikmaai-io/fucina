#!/usr/bin/env python3
# qwen35_fp8_ref.py — full-model torch oracle for the OFFICIAL Qwen3.5-9B FP8 checkpoint.
#
# Mirrors the M3 GGUF reference (qwen35_full_ref.py) but reads the FP8 *safetensors* checkpoint
# directly: DeepSeek-V3 block-fp8 weights (F8_E4M3 [out,in] + weight_scale_inv BF16 [out/128,
# in/128], dequant W = fp8(W) * scale[o//128,i//128]) for the quantized Linears, BF16/F32 for the
# modules_to_not_convert (all norms, embed, lm_head, conv1d, A_log, dt_bias, in_proj_a/b). Runs the
# exact qwen35 hybrid math (24 GDN gated-deltanet linear + 8 FULL output-gated softmax-GQA layers,
# period-4) and greedily generates the continuation. This is the M5 parity oracle — fucina's FP8
# decode forward (fp8_block GEMV) must match these argmax ids 8/8 for the fixed prompt.
#
# Identical mixer math to the M3 oracle that matched llama.cpp 8/8 on the GGUF; only the weights
# (official FP8 vs abliterated Q4_K) and the tensor container differ. Text path only (skip visual).
import sys, json, struct, glob, os
import numpy as np, torch, torch.nn.functional as F
from transformers.models.qwen3_next.modeling_qwen3_next import torch_recurrent_gated_delta_rule

DIR = sys.argv[1] if len(sys.argv) > 1 else "/opt/spark/models/models--Qwen--Qwen3.5-9B-FP8"
NGEN = int(sys.argv[2]) if len(sys.argv) > 2 else 12
PROMPT_TXT = "The capital of France is"

H, HEAD_DIM, NQ, NKV, ROT, THETA = 4096, 256, 16, 4, 64, 1e7
CONV_DIM, KEYD, VALD, NKH, NVH, SD, TSR, CK = 8192, 2048, 4096, 16, 32, 128, 32, 4
EPS = 1e-6
LM = "model.language_model."
torch.set_default_dtype(torch.float32)

# ── safetensors multi-shard reader (header-only; returns raw bytes per tensor) ──
shards = sorted(glob.glob(os.path.join(DIR, "model.safetensors-*.safetensors")))
HDR = {}   # name -> (file, dtype, shape, off_b, off_e, data_start)
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
        fh.seek(ds + b)
        buf = fh.read(e - b)
    return dt, shape, buf

def as_tensor(name):
    dt, shape, buf = raw(name)
    if dt == "F32":
        a = np.frombuffer(buf, dtype=np.float32).reshape(shape).copy()
        return torch.from_numpy(a).float()
    if dt == "BF16":
        a = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32).reshape(shape)
        return torch.from_numpy((a << 16).view(np.float32).copy()).float()
    if dt == "F8_E4M3":
        t = torch.frombuffer(bytearray(buf), dtype=torch.uint8).view(torch.float8_e4m3fn)
        return t.reshape(shape).float()
    raise RuntimeError(f"unhandled dtype {dt} for {name}")

_WC = {}
def W(name):
    # Quantized Linear iff a block-scale sibling exists; else plain BF16/F32 weight.
    if name in _WC:
        return _WC[name]
    sname = name + "_scale_inv"
    w = as_tensor(name)
    if sname in HDR:
        scale = as_tensor(sname)                     # [out/128, in/128]
        O, I = w.shape
        sf = scale.repeat_interleave(128, 0)[:O].repeat_interleave(128, 1)[:, :I]
        w = w * sf
    _WC[name] = w
    return w

# Qwen3_5RMSNorm: output = _norm(x) * (1 + weight)   [Gemma-style +1; the GGUF bakes it in]
def rms(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)
# Qwen3_5RMSNormGated (linear_attn.norm): plain weight * _norm(x), gate applied separately (NO +1)
def rmsg(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
def lin(x, w):
    return x @ w.t()

nl = 32
attn_kind = ["FULL" if (i + 1) % 4 == 0 else "LINEAR" for i in range(nl)]

def full_layer(h, l):
    N = h.shape[0]
    p = f"{LM}layers.{l}."
    x = rms(h, W(p + "input_layernorm.weight"))
    qg = lin(x, W(p + "self_attn.q_proj.weight")).view(N, NQ, HEAD_DIM * 2)
    q = qg[:, :, :HEAD_DIM].contiguous()
    gate = qg[:, :, HEAD_DIM:].reshape(N, NQ * HEAD_DIM)
    q = rms(q, W(p + "self_attn.q_norm.weight"))
    k = lin(x, W(p + "self_attn.k_proj.weight")).view(N, NKV, HEAD_DIM)
    k = rms(k, W(p + "self_attn.k_norm.weight"))
    v = lin(x, W(p + "self_attn.v_proj.weight")).view(N, NKV, HEAD_DIM)
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
    return lin(attn, W(p + "self_attn.o_proj.weight"))

def gdn_layer(h, l):
    N = h.shape[0]
    p = f"{LM}layers.{l}."
    x = rms(h, W(p + "input_layernorm.weight"))
    qkv = lin(x, W(p + "linear_attn.in_proj_qkv.weight"))     # [N, 8192]
    z   = lin(x, W(p + "linear_attn.in_proj_z.weight"))       # [N, 4096]
    a   = lin(x, W(p + "linear_attn.in_proj_a.weight"))       # [N, 32]  -> decay
    b   = lin(x, W(p + "linear_attn.in_proj_b.weight"))       # [N, 32]  -> beta
    cw  = W(p + "linear_attn.conv1d.weight").reshape(CONV_DIM, CK)  # [8192,1,4] -> [8192,4]
    xpad = F.pad(qkv.t(), (CK - 1, 0)); conv = torch.zeros(CONV_DIM, N)
    for j in range(CK):
        conv += cw[:, j:j + 1] * xpad[:, j:j + N]
    conv = F.silu(conv).t()
    q = conv[:, :KEYD].reshape(N, NKH, SD); k = conv[:, KEYD:2 * KEYD].reshape(N, NKH, SD)
    v = conv[:, 2 * KEYD:].reshape(N, NVH, SD)
    q = q.repeat_interleave(NVH // NKH, dim=1); k = k.repeat_interleave(NVH // NKH, dim=1)  # HF interleave 16->32
    beta = torch.sigmoid(b)
    A_log = W(p + "linear_attn.A_log"); dt = W(p + "linear_attn.dt_bias")
    g = -torch.exp(A_log) * F.softplus(a + dt)
    core, _ = torch_recurrent_gated_delta_rule(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
              g=g.unsqueeze(0), beta=beta.unsqueeze(0), initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=True)
    core = core.squeeze(0)
    out = rmsg(core, W(p + "linear_attn.norm.weight")) * F.silu(z.reshape(N, NVH, SD))
    return lin(out.reshape(N, VALD), W(p + "linear_attn.out_proj.weight"))

EMB = W(LM + "embed_tokens.weight")
ONORM = W(LM + "norm.weight")
LMH = W("lm_head.weight")

def forward_logits_last(ids):
    h = EMB[torch.tensor(ids)].clone()
    for l in range(nl):
        mix = full_layer(h, l) if attn_kind[l] == "FULL" else gdn_layer(h, l)
        h = h + mix
        p = f"{LM}layers.{l}."
        x2 = rms(h, W(p + "post_attention_layernorm.weight"))
        g = lin(x2, W(p + "mlp.gate_proj.weight")); u = lin(x2, W(p + "mlp.up_proj.weight"))
        h = h + lin(F.silu(g) * u, W(p + "mlp.down_proj.weight"))
    hn = rms(h, ONORM)
    return lin(hn[-1], LMH)

# ── tokenize the fixed prompt with the official tokenizer ──
prompt_ids = None
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DIR)
    prompt_ids = tok(PROMPT_TXT, add_special_tokens=False)["input_ids"]
except Exception as ex:
    print("tokenizer load failed, falling back to pinned ids:", ex, file=sys.stderr)
    prompt_ids = [760, 6511, 314, 9338, 369]
print("prompt ids:", prompt_ids)

ids = list(prompt_ids)
cont = []
with torch.no_grad():
    for step in range(NGEN):
        logits = forward_logits_last(ids)
        nxt = int(logits.argmax())
        cont.append(nxt); ids.append(nxt)
        print(f"step {step:2d}: argmax={nxt}")
print("PROMPT_IDS =", prompt_ids)
print("CONT_IDS   =", cont)
