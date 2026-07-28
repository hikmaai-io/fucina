package model

import (
	"fmt"
	"sort"
	"strings"
)

type tensorChecker struct {
	tensors map[string]TensorInfo
	errs    *[]TensorMismatch
}

func (c tensorChecker) expect(name string, shape []int64, encodings ...string) {
	t, ok := c.tensors[name]
	if !ok {
		mismatch(c.errs, name, MismatchMissing, describe(shape, encodings), "missing")
		return
	}
	if !sameShape(t.Shape, shape) {
		mismatch(c.errs, name, MismatchShape, shapeString(shape), shapeString(t.Shape))
	}
	if len(encodings) > 0 && !encodingIn(t.Encoding, encodings) {
		mismatch(c.errs, name, MismatchEncoding, strings.Join(encodings, "|"), canonicalEncoding(t.Encoding))
	}
}

func (c tensorChecker) expectElements(name string, elements int64, encodings ...string) {
	t, ok := c.tensors[name]
	if !ok {
		mismatch(c.errs, name, MismatchMissing, fmt.Sprintf("%d elements, %s", elements, strings.Join(encodings, "|")), "missing")
		return
	}
	actual := int64(1)
	for _, d := range t.Shape {
		actual *= d
	}
	if actual != elements {
		mismatch(c.errs, name, MismatchShape, fmt.Sprintf("%d elements", elements), fmt.Sprintf("%d elements (%s)", actual, shapeString(t.Shape)))
	}
	if len(encodings) > 0 && !encodingIn(t.Encoding, encodings) {
		mismatch(c.errs, name, MismatchEncoding, strings.Join(encodings, "|"), canonicalEncoding(t.Encoding))
	}
}

func (c tensorChecker) optional(name string, shape []int64, encodings ...string) {
	if _, ok := c.tensors[name]; ok {
		c.expect(name, shape, encodings...)
	}
}

func sameShape(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
func encodingIn(actual string, allowed []string) bool {
	a := canonicalEncoding(actual)
	for _, e := range allowed {
		if a == canonicalEncoding(e) {
			return true
		}
	}
	return false
}
func shapeString(s []int64) string {
	if len(s) == 0 {
		return "scalar"
	}
	return fmt.Sprint(s)
}
func describe(shape []int64, enc []string) string {
	return shapeString(shape) + " " + strings.Join(enc, "|")
}
func i64(v ...int) []int64 {
	o := make([]int64, len(v))
	for i, n := range v {
		o[i] = int64(n)
	}
	return o
}

var ggufWeights = []string{"Q4_0", "Q8_0", "Q4_1", "Q4_K", "Q5_K", "Q6_K", "F16", "BF16"}
var ggufNorms = []string{"F32", "F16", "BF16"}

func preflightGemmaGGUF(s DescriptorSnapshot, t map[string]TensorInfo, errs *[]TensorMismatch) {
	c := tensorChecker{t, errs}
	g := s.Geometry
	c.expect("token_embd.weight", i64(g.HiddenSize, g.VocabSize), append(ggufWeights, "Q6_K")...)
	c.expect("output_norm.weight", i64(g.HiddenSize), ggufNorms...)
	c.optional("output.weight", i64(g.HiddenSize, g.VocabSize), ggufWeights...)
	for l, mix := range s.Mixers {
		hd, nkv := g.HeadDim, g.KVHeads
		if mix == MixerFullAttention {
			hd = g.GlobalHeadDim
			nkv = g.GlobalKVHeads
		}
		p := fmt.Sprintf("blk.%d.", l)
		c.expect(p+"attn_q.weight", i64(g.HiddenSize, g.AttentionHead*hd), ggufWeights...)
		c.expect(p+"attn_k.weight", i64(g.HiddenSize, nkv*hd), ggufWeights...)
		if mix == MixerSlidingAttention {
			c.expect(p+"attn_v.weight", i64(g.HiddenSize, nkv*hd), ggufWeights...)
		}
		c.expect(p+"attn_output.weight", i64(g.AttentionHead*hd, g.HiddenSize), ggufWeights...)
		c.expect(p+"attn_norm.weight", i64(g.HiddenSize), ggufNorms...)
		c.expect(p+"attn_q_norm.weight", i64(hd), ggufNorms...)
		c.expect(p+"attn_k_norm.weight", i64(hd), ggufNorms...)
		c.expect(p+"post_attention_norm.weight", i64(g.HiddenSize), ggufNorms...)
		c.expect(p+"ffn_gate.weight", i64(g.HiddenSize, g.Intermediate), ggufWeights...)
		c.expect(p+"ffn_up.weight", i64(g.HiddenSize, g.Intermediate), ggufWeights...)
		c.expect(p+"ffn_down.weight", i64(g.Intermediate, g.HiddenSize), ggufWeights...)
		c.expect(p+"ffn_norm.weight", i64(g.HiddenSize), ggufNorms...)
		c.expect(p+"post_ffw_norm.weight", i64(g.HiddenSize), ggufNorms...)
		c.expect(p+"layer_output_scale.weight", i64(1), ggufNorms...)
	}
}

func preflightQwenGGUF(s DescriptorSnapshot, t map[string]TensorInfo, errs *[]TensorMismatch) {
	c := tensorChecker{t, errs}
	g := s.Geometry
	d := s.GDN
	c.expect("token_embd.weight", i64(g.HiddenSize, g.VocabSize), ggufWeights...)
	c.expect("output_norm.weight", i64(g.HiddenSize), ggufNorms...)
	c.optional("output.weight", i64(g.HiddenSize, g.VocabSize), ggufWeights...)
	for l, mix := range s.Mixers {
		p := fmt.Sprintf("blk.%d.", l)
		c.expect(p+"attn_norm.weight", i64(g.HiddenSize), ggufNorms...)
		c.expect(p+"post_attention_norm.weight", i64(g.HiddenSize), ggufNorms...)
		c.expect(p+"ffn_gate.weight", i64(g.HiddenSize, g.Intermediate), ggufWeights...)
		c.expect(p+"ffn_up.weight", i64(g.HiddenSize, g.Intermediate), ggufWeights...)
		c.expect(p+"ffn_down.weight", i64(g.Intermediate, g.HiddenSize), ggufWeights...)
		if mix == MixerFullAttention {
			c.expect(p+"attn_q.weight", i64(g.HiddenSize, 2*g.AttentionHead*g.HeadDim), ggufWeights...)
			c.expect(p+"attn_k.weight", i64(g.HiddenSize, g.KVHeads*g.HeadDim), ggufWeights...)
			c.expect(p+"attn_v.weight", i64(g.HiddenSize, g.KVHeads*g.HeadDim), ggufWeights...)
			c.expect(p+"attn_output.weight", i64(g.AttentionHead*g.HeadDim, g.HiddenSize), ggufWeights...)
			c.expect(p+"attn_q_norm.weight", i64(g.HeadDim), ggufNorms...)
			c.expect(p+"attn_k_norm.weight", i64(g.HeadDim), ggufNorms...)
		} else {
			c.expect(p+"attn_qkv.weight", i64(g.HiddenSize, d.ConvDim), ggufWeights...)
			c.expect(p+"attn_gate.weight", i64(g.HiddenSize, d.InnerSize), ggufWeights...)
			c.expect(p+"ssm_alpha.weight", i64(g.HiddenSize, d.TimeStepRank), ggufWeights...)
			c.expect(p+"ssm_beta.weight", i64(g.HiddenSize, d.TimeStepRank), ggufWeights...)
			c.expect(p+"ssm_a", i64(d.TimeStepRank), ggufNorms...)
			c.expect(p+"ssm_dt.bias", i64(d.TimeStepRank), ggufNorms...)
			c.expect(p+"ssm_conv1d.weight", i64(d.ConvKernel, d.ConvDim), ggufNorms...)
			c.expect(p+"ssm_norm.weight", i64(d.StateSize), ggufNorms...)
			c.expect(p+"ssm_out.weight", i64(d.InnerSize, g.HiddenSize), ggufWeights...)
		}
	}
}

func checkpointRoot(t map[string]TensorInfo) string {
	for _, r := range []string{"model.language_model.", "model."} {
		if _, ok := t[r+"layers.0.input_layernorm.weight"]; ok {
			return r
		}
	}
	return "model."
}

func preflightGemmaSafetensors(s DescriptorSnapshot, root map[string]any, t map[string]TensorInfo, errs *[]TensorMismatch) {
	c := tensorChecker{t, errs}
	g := s.Geometry
	r := checkpointRoot(t)
	c.expect(r+"embed_tokens.weight", i64(g.VocabSize, g.HiddenSize), "BF16", "F16")
	c.expect(r+"norm.weight", i64(g.HiddenSize), "BF16", "F32")
	if _, ok := t["lm_head.weight"]; ok {
		c.expect("lm_head.weight", i64(g.VocabSize, g.HiddenSize), "BF16", "F16")
	} else if !configBool(root, "tie_word_embeddings") {
		mismatch(errs, "lm_head.weight", MismatchMissing, "explicit head or tie_word_embeddings=true", "missing")
	}
	for l, mix := range s.Mixers {
		hd, nkv := g.HeadDim, g.KVHeads
		if mix == MixerFullAttention && g.GlobalHeadDim > 0 {
			hd = g.GlobalHeadDim
			nkv = g.GlobalKVHeads
		}
		p := fmt.Sprintf("%slayers.%d.", r, l)
		expectSTLinear(c, p+"self_attn.q_proj.weight", g.AttentionHead*hd, g.HiddenSize, s.SourceQuant)
		expectSTLinear(c, p+"self_attn.k_proj.weight", nkv*hd, g.HiddenSize, s.SourceQuant)
		expectSTLinear(c, p+"self_attn.v_proj.weight", nkv*hd, g.HiddenSize, s.SourceQuant)
		expectSTLinear(c, p+"self_attn.o_proj.weight", g.HiddenSize, g.AttentionHead*hd, s.SourceQuant)
		expectSTLinear(c, p+"mlp.gate_proj.weight", g.Intermediate, g.HiddenSize, s.SourceQuant)
		expectSTLinear(c, p+"mlp.up_proj.weight", g.Intermediate, g.HiddenSize, s.SourceQuant)
		expectSTLinear(c, p+"mlp.down_proj.weight", g.HiddenSize, g.Intermediate, s.SourceQuant)
		for _, n := range []string{"input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"} {
			c.expect(p+n, i64(g.HiddenSize), "BF16", "F32")
		}
		c.expect(p+"self_attn.q_norm.weight", i64(hd), "BF16", "F32")
		c.expect(p+"self_attn.k_norm.weight", i64(hd), "BF16", "F32")
	}
}

func preflightQwenSafetensors(s DescriptorSnapshot, root map[string]any, t map[string]TensorInfo, errs *[]TensorMismatch) {
	c := tensorChecker{t, errs}
	g := s.Geometry
	d := s.GDN
	r := checkpointRoot(t)
	c.expect(r+"embed_tokens.weight", i64(g.VocabSize, g.HiddenSize), "BF16")
	c.expect(r+"norm.weight", i64(g.HiddenSize), "BF16")
	if _, ok := t["lm_head.weight"]; ok {
		expectHead(c, "lm_head.weight", g.VocabSize, g.HiddenSize, s.SourceQuant)
	} else if _, ok := t[r+"lm_head.weight"]; ok {
		expectHead(c, r+"lm_head.weight", g.VocabSize, g.HiddenSize, s.SourceQuant)
	} else {
		mismatch(errs, "lm_head.weight", MismatchMissing, "untied Qwen head", "missing")
	}
	for l, mix := range s.Mixers {
		p := fmt.Sprintf("%slayers.%d.", r, l)
		c.expect(p+"input_layernorm.weight", i64(g.HiddenSize), "BF16")
		c.expect(p+"post_attention_layernorm.weight", i64(g.HiddenSize), "BF16")
		if mix == MixerFullAttention {
			expectSTLinear(c, p+"self_attn.q_proj.weight", 2*g.AttentionHead*g.HeadDim, g.HiddenSize, s.SourceQuant)
			expectSTLinear(c, p+"self_attn.k_proj.weight", g.KVHeads*g.HeadDim, g.HiddenSize, s.SourceQuant)
			expectSTLinear(c, p+"self_attn.v_proj.weight", g.KVHeads*g.HeadDim, g.HiddenSize, s.SourceQuant)
			expectSTLinear(c, p+"self_attn.o_proj.weight", g.HiddenSize, g.AttentionHead*g.HeadDim, s.SourceQuant)
			c.expect(p+"self_attn.q_norm.weight", i64(g.HeadDim), "BF16")
			c.expect(p+"self_attn.k_norm.weight", i64(g.HeadDim), "BF16")
		} else {
			expectSTLinear(c, p+"linear_attn.in_proj_qkv.weight", d.ConvDim, g.HiddenSize, s.SourceQuant)
			expectSTLinear(c, p+"linear_attn.in_proj_z.weight", d.InnerSize, g.HiddenSize, s.SourceQuant)
			expectSTLinear(c, p+"linear_attn.out_proj.weight", g.HiddenSize, d.InnerSize, s.SourceQuant)
			c.expect(p+"linear_attn.in_proj_a.weight", i64(d.TimeStepRank, g.HiddenSize), "BF16")
			c.expect(p+"linear_attn.in_proj_b.weight", i64(d.TimeStepRank, g.HiddenSize), "BF16")
			c.expectElements(p+"linear_attn.conv1d.weight", int64(d.ConvDim*d.ConvKernel), "BF16")
			c.expectElements(p+"linear_attn.A_log", int64(d.TimeStepRank), "BF16", "F32")
			c.expectElements(p+"linear_attn.dt_bias", int64(d.TimeStepRank), "BF16")
			c.expectElements(p+"linear_attn.norm.weight", int64(d.StateSize), "BF16", "F32")
		}
		if s.MoE.Experts > 0 {
			si := s.MoE.SharedExpertIntermediate
			if si == 0 {
				si = g.Intermediate
			}
			c.expect(p+"mlp.shared_expert_gate.weight", i64(g.HiddenSize), "BF16")
			expectSTLinear(c, p+"mlp.shared_expert.gate_proj.weight", si, g.HiddenSize, s.SourceQuant)
			expectSTLinear(c, p+"mlp.shared_expert.up_proj.weight", si, g.HiddenSize, s.SourceQuant)
			expectSTLinear(c, p+"mlp.shared_expert.down_proj.weight", g.HiddenSize, si, s.SourceQuant)
			c.expect(p+"mlp.gate.weight", i64(s.MoE.Experts, g.HiddenSize), "BF16")
			for e := 0; e < s.MoE.Experts; e++ {
				ep := fmt.Sprintf("%smlp.experts.%d.", p, e)
				expectSTLinear(c, ep+"gate_proj.weight", s.MoE.ExpertIntermediate, g.HiddenSize, s.SourceQuant)
				expectSTLinear(c, ep+"up_proj.weight", s.MoE.ExpertIntermediate, g.HiddenSize, s.SourceQuant)
				expectSTLinear(c, ep+"down_proj.weight", g.HiddenSize, s.MoE.ExpertIntermediate, s.SourceQuant)
			}
		} else {
			expectSTLinear(c, p+"mlp.gate_proj.weight", g.Intermediate, g.HiddenSize, s.SourceQuant)
			expectSTLinear(c, p+"mlp.up_proj.weight", g.Intermediate, g.HiddenSize, s.SourceQuant)
			expectSTLinear(c, p+"mlp.down_proj.weight", g.HiddenSize, g.Intermediate, s.SourceQuant)
		}
	}
}

func preflightE4BSafetensors(s DescriptorSnapshot, cfg map[string]any, t map[string]TensorInfo, errs *[]TensorMismatch) {
	c := tensorChecker{t, errs}
	g := s.Geometry
	r := "model.language_model."
	pd, _ := intValue(cfg, "hidden_size_per_layer_input")
	pv, _ := intValue(cfg, "vocab_size_per_layer_input")
	if pv == 0 {
		pv = g.VocabSize
	}
	width := g.Layers * pd
	c.expect(r+"embed_tokens.weight", i64(g.VocabSize, g.HiddenSize), "BF16")
	c.expect(r+"embed_tokens_per_layer.weight", i64(pv, width), "BF16")
	c.expect(r+"per_layer_model_projection.weight", i64(width, g.HiddenSize), "BF16")
	c.expect(r+"per_layer_projection_norm.weight", i64(pd), "BF16")
	c.expect(r+"norm.weight", i64(g.HiddenSize), "BF16")
	for l, mix := range s.Mixers {
		hd := g.HeadDim
		if mix == MixerFullAttention {
			hd = g.GlobalHeadDim
		}
		p := fmt.Sprintf("%slayers.%d.", r, l)
		c.expect(p+"self_attn.q_proj.weight", i64(g.AttentionHead*hd, g.HiddenSize), "BF16")
		c.expect(p+"self_attn.k_proj.weight", i64(g.KVHeads*hd, g.HiddenSize), "BF16")
		c.expect(p+"self_attn.v_proj.weight", i64(g.KVHeads*hd, g.HiddenSize), "BF16")
		c.expect(p+"self_attn.o_proj.weight", i64(g.HiddenSize, g.AttentionHead*hd), "BF16")
		c.expect(p+"self_attn.q_norm.weight", i64(hd), "BF16")
		c.expect(p+"self_attn.k_norm.weight", i64(hd), "BF16")
		c.expect(p+"mlp.gate_proj.weight", i64(g.Intermediate, g.HiddenSize), "BF16")
		c.expect(p+"mlp.up_proj.weight", i64(g.Intermediate, g.HiddenSize), "BF16")
		c.expect(p+"mlp.down_proj.weight", i64(g.HiddenSize, g.Intermediate), "BF16")
		for _, n := range []string{"input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "post_per_layer_input_norm.weight"} {
			c.expect(p+n, i64(g.HiddenSize), "BF16")
		}
		c.expect(p+"per_layer_input_gate.weight", i64(pd, g.HiddenSize), "BF16")
		c.expect(p+"per_layer_projection.weight", i64(g.HiddenSize, pd), "BF16")
		c.optional(p+"layer_scalar", []int64{}, "BF16")
	}
}

func expectHead(c tensorChecker, name string, out, in int, source string) {
	t, ok := c.tensors[name]
	if !ok {
		c.expect(name, i64(out, in), "BF16")
		return
	}
	if canonicalEncoding(t.Encoding) == "BF16" {
		c.expect(name, i64(out, in), "BF16")
		return
	}
	expectSTLinear(c, name, out, in, source)
}

func expectSTLinear(c tensorChecker, name string, out, in int, source string) {
	t, ok := c.tensors[name]
	// Official block-FP8.
	if ok && canonicalEncoding(t.Encoding) == "F8_E4M3" {
		c.expect(name, i64(out, in), "F8_E4M3")
		scale := name + "_scale_inv"
		if _, yes := c.tensors[scale]; yes {
			c.expect(scale, i64((out+127)/128, (in+127)/128), "BF16")
		} else if _, yes := c.tensors[name+"_scale"]; yes {
			c.expectElements(name+"_scale", int64(out), "BF16", "F32")
		} else {
			mismatch(c.errs, scale, MismatchScale, "BF16 128x128 scale grid or per-row scale", "missing")
		}
		return
	}
	base := strings.TrimSuffix(name, "weight")
	packedName := base + "weight_packed"
	if _, yes := c.tensors[packedName]; !yes {
		packedName = name
	}
	if p, yes := c.tensors[packedName]; yes && canonicalEncoding(p.Encoding) == "U8" {
		c.expect(packedName, i64(out, in/2), "U8")
		scale := base + "weight_scale"
		if _, ok := c.tensors[scale]; !ok {
			mismatch(c.errs, scale, MismatchScale, "NVFP4 block scales", "missing")
		} else {
			st := c.tensors[scale]
			if !encodingIn(st.Encoding, []string{"U8", "F8_E4M3", "BF16"}) {
				mismatch(c.errs, scale, MismatchEncoding, "U8|F8_E4M3|BF16", st.Encoding)
			}
			blockShape := i64(out, (in+15)/16)
			rowShape := i64(out, 1)
			if !sameShape(st.Shape, blockShape) && !sameShape(st.Shape, rowShape) {
				mismatch(c.errs, scale, MismatchShape, shapeString(blockShape)+" or "+shapeString(rowShape), shapeString(st.Shape))
			}
		}
		global := base + "weight_global_scale"
		if _, yes := c.tensors[global]; !yes {
			global = name + "_scale_2"
		}
		c.expectElements(global, 1, "F32")
		return
	}
	if strings.EqualFold(source, "BF16") {
		c.expect(name, i64(out, in), "BF16")
		return
	}
	if !ok {
		mismatch(c.errs, name, MismatchMissing, fmt.Sprintf("[%d %d] block-FP8/NVFP4/BF16 weight", out, in), "missing")
		return
	}
	mismatch(c.errs, name, MismatchEncoding, "F8_E4M3 with scales, packed U8 NVFP4 triplet, or BF16", canonicalEncoding(t.Encoding))
}

func configBool(root map[string]any, key string) bool {
	if v, ok := root[key].(bool); ok {
		return v
	}
	if t, ok := root["text_config"].(map[string]any); ok {
		if v, ok := t[key].(bool); ok {
			return v
		}
	}
	return false
}

func sortedTensorNames(t map[string]TensorInfo) []string {
	o := make([]string, 0, len(t))
	for n := range t {
		o = append(o, n)
	}
	sort.Strings(o)
	return o
}

var _ = sortedTensorNames
