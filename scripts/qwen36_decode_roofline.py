#!/usr/bin/env python3
"""Host-only, source-shaped byte ledger for Qwen3.6-35B-A3B B=1 decode."""

import argparse
import json
from pathlib import Path

H = 2048
VOC = 248_320
LAYERS = 40
FULL = 10
GDN = 30
NQ = 16
NKV = 2
HD = 256
NVH = 32
SD = 128
CONVD = 8192
CK = 4
TOPK = 8
EXPERTS = 256
EFFN = 512
Q8HEAD_MAXCAND = 64
MAXCTX = 25_280
BW = 273e9


def load_plan(path: Path):
    plan = json.loads(path.read_text())
    assert plan["finalized"] is True
    return plan


def immutable_ledger(plan):
    out = {
        "q4_mixer": 0,
        "router": 0,
        "shared_gate": 0,
        "shared_expert": 0,
        "shared_scales": 0,
        "mixer_metadata": 0,
    }
    for t in plan["tensors"]:
        name, arena, dst, size = t["logical_name"], t["arena"], t["destination"], t["bytes"]
        if arena == "scales":
            out["shared_scales"] += size
        elif arena != "core_weights":
            continue
        elif dst == "Q4_K":
            out["q4_mixer"] += size
        elif ".mlp.gate.weight" in name:
            out["router"] += size
        elif "shared_expert_gate" in name:
            out["shared_gate"] += size
        elif "shared_expert." in name:
            out["shared_expert"] += size
        else:
            out["mixer_metadata"] += size
    assert sum(out.values()) == plan["totals"]["core_weights"] + plan["totals"]["scales"]
    return out


def fixed_traffic(plan):
    imm = immutable_ledger(plan)
    # Each token's top-k is unique, so exactly TOPK/EXPERTS of each layer's packed expert
    # and scale slabs is selected. The generated plan records all 40 resident slabs exactly.
    expert_resident = sum(t["bytes"] for t in plan["tensors"] if t["arena"] == "expert_slabs")
    # Exclude the 2 F32 global scales per layer; they are included in tiny metadata below.
    expert_globals = LAYERS * 2 * 4
    routed = (expert_resident - expert_globals) * TOPK // EXPERTS
    assert routed == 566_231_040

    q8_head = next(t["bytes"] for t in plan["tensors"] if t["logical_name"] == "lm_head.q8")
    assert q8_head == (VOC * H // 32) * 34
    rescore = Q8HEAD_MAXCAND * H * 2

    # qwen35_b_gdn_kernel loads and stores BF16 S; qwen35_b_conv_kernel loads and stores
    # CK-1 F32 ring values for each channel.
    gdn_state = GDN * 2 * (NVH * SD * SD * 2)
    conv_state = GDN * 2 * (CONVD * (CK - 1) * 4)

    # Every Q4_K projection quantizes its F32 activation to q8 + F32 scale + I32 sum.
    # Count one source read, one scratch write and one compulsory scratch read.
    q4_in_elems = GDN * (H + H + NVH * SD) + FULL * (H + H + H + NQ * HD)
    q4_act_quant = q4_in_elems * (4 + 1 + 4 / 32 + 4 / 32 + 1 + 4 / 32 + 4 / 32)
    assert q4_act_quant.is_integer()

    # NVFP4 routed activation quant, for TOPK assignment rows: row-amax read + quant read,
    # packed E2M1 write/read, and one E4M3 scale per 16 values write/read, for GU and down.
    fp4_act_quant = 0
    for width in (H, EFFN):
        fp4_act_quant += 2 * TOPK * width * 4
        fp4_act_quant += 2 * TOPK * (width // 2)
        fp4_act_quant += 2 * TOPK * (width // 16)
    fp4_act_quant *= LAYERS

    # Approx-logit write and the two candidate scans. Candidate ids/counts are <1 KiB at B=1.
    head_scan_scratch = 3 * VOC * 4
    # BF16 embedding row and the expert global scales are tiny but explicitly charged.
    tiny_static = H * 2 + expert_globals

    # Flash partial writes followed by combine reads. S is capture-stable and based on maxctx.
    splits = (MAXCTX + 511) // 512
    attn_partials = FULL * 2 * NQ * splits * (2 * 4 + HD * 4)

    fixed = sum(imm.values()) + routed + q8_head + rescore + gdn_state + conv_state
    fixed += int(q4_act_quant) + fp4_act_quant + head_scan_scratch + tiny_static + attn_partials
    return {
        **imm,
        "routed_experts": routed,
        "q8_head": q8_head,
        "bf16_rescore_max": rescore,
        "gdn_state": gdn_state,
        "conv_state": conv_state,
        "q4_act_quant": int(q4_act_quant),
        "fp4_act_quant": fp4_act_quant,
        "head_scan_scratch": head_scan_scratch,
        "tiny_static": tiny_static,
        "attn_partials": attn_partials,
        "fixed": fixed,
    }


def kv_bytes(prior_context, gqa_requested=False):
    # Current K/V is written, then attention reads positions [0, prior_context].
    per_position = FULL * 2 * NKV * HD * 2
    read_multiplier = NQ // NKV if gqa_requested else 1
    return per_position + read_multiplier * (prior_context + 1) * per_position


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path,
                    default=Path("benchmark-evidence/tensor-plans/qwen36-unsloth-accurate.json"))
    ap.add_argument("--contexts", default="0,1000,3500,4096,8192,25280")
    args = ap.parse_args()
    plan = load_plan(args.plan)
    t = fixed_traffic(plan)

    print("Qwen3.6-35B-A3B B=1 decode byte ledger (decimal MB/GB)")
    for key in ("q4_mixer", "router", "shared_gate", "shared_expert", "shared_scales",
                "mixer_metadata", "routed_experts", "gdn_state", "conv_state", "q8_head",
                "bf16_rescore_max", "q4_act_quant", "fp4_act_quant", "head_scan_scratch",
                "tiny_static", "attn_partials"):
        print(f"{key:24s} {t[key]:12,d} B  {t[key]/1e6:9.3f} MB")
    print(f"{'fixed before KV':24s} {t['fixed']:12,d} B  {t['fixed']/1e9:9.3f} GB")
    print("\nctx  unique_GB peak_ms peak_tok/s 80%_tok/s  GQA-request_GB peak_tok/s")
    for p in map(int, args.contexts.split(",")):
        unique = t["fixed"] + kv_bytes(p, False)
        requested = t["fixed"] + kv_bytes(p, True)
        print(f"{p:5d} {unique/1e9:9.3f} {unique/BW*1e3:7.3f} {BW/unique:10.1f} "
              f"{0.8*BW/unique:9.1f} {requested/1e9:15.3f} {BW/requested:10.1f}")


if __name__ == "__main__":
    main()
