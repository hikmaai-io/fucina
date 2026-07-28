package model

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGemma4GGUFQualifiedFormats(t *testing.T) {
	for _, tc := range []struct{ name, bulk, embed, want string }{
		{"Q4_0-QAT", "Q4_0", "Q6_K", "Q4_0-QAT"},
		{"Q8_0", "Q8_0", "Q8_0", "Q8_0"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := gemmaGGUF(tc.bulk, tc.embed)
			d, err := FromGGUF(m)
			if err != nil {
				t.Fatal(err)
			}
			if got := d.SourceQuantization(); got != tc.want {
				t.Fatalf("quant=%q want %q", got, tc.want)
			}
			if d.Qualification() != QualificationValidated {
				t.Fatalf("qualification=%s", d.Qualification())
			}
			caps := d.Capabilities()
			if !caps.BatchMTP || !caps.ContinuousBatchingDefault || caps.LegacyQwen3 {
				t.Fatalf("bad source capabilities: %+v", caps)
			}
			assertFingerprints(t, d)
		})
	}
}

func TestConfigQualifiedAcceptanceMatrix(t *testing.T) {
	cases := []struct {
		name                string
		makeFixture         func(*testing.T) string
		family, arch, quant string
		layers              int
		moe                 bool
	}{
		{"gemma4-12b-nvfp4", fixtureGemmaNVFP4, "gemma-4", "gemma4", "NVFP4", 48, false},
		{"qwen35-9b-dense", func(t *testing.T) string { return fixtureQwenFP8(t, "Qwen3.5-9B-FP8", 4096, 12288, 32, 16, 4, 32, 0) }, "qwen3.5", "qwen35", "FP8_BLOCK_128", 32, false},
		{"qwen36-27b-dense", func(t *testing.T) string { return fixtureQwenFP8(t, "Qwen3.6-27B-FP8", 5120, 17408, 64, 24, 4, 48, 0) }, "qwen3.6", "qwen35", "FP8_BLOCK_128", 64, false},
		{"qwen35-35b-a3b-moe", func(t *testing.T) string {
			return fixtureQwenFP8(t, "Qwen3.5-35B-A3B-FP8", 2048, 512, 40, 16, 2, 32, 256)
		}, "qwen3.5", "qwen35", "FP8_BLOCK_128", 40, true},
		{"gemma4-e4b-bf16", fixtureE4B, "gemma-4", "gemma4-e4b", "BF16", 42, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := tc.makeFixture(t)
			d, err := FromConfigJSON(path)
			if err != nil {
				t.Fatal(err)
			}
			s := d.Snapshot()
			if s.Family != tc.family || s.Architecture != tc.arch || s.SourceQuant != tc.quant || s.Geometry.Layers != tc.layers {
				t.Fatalf("unexpected descriptor: family=%s arch=%s quant=%s layers=%d", s.Family, s.Architecture, s.SourceQuant, s.Geometry.Layers)
			}
			if (s.MoE.Experts > 0) != tc.moe {
				t.Fatalf("moe=%+v", s.MoE)
			}
			if !s.Capabilities.ContinuousBatchingDefault || s.Capabilities.LegacyQwen3 {
				t.Fatalf("bad caps: %+v", s.Capabilities)
			}
			if tc.arch == "gemma4-e4b" && !s.Capabilities.E4BServing {
				t.Fatal("E4B serving not source-derived")
			}
			assertFingerprints(t, d)
		})
	}
}

func TestPreflightReturnsEveryCorruption(t *testing.T) {
	m := gemmaGGUF("Q8_0", "Q8_0")
	delete(m.Tensors, "blk.0.attn_k.weight")
	x := m.Tensors["blk.1.ffn_down.weight"]
	x.Shape = []int64{7, 9}
	m.Tensors["blk.1.ffn_down.weight"] = x
	x = m.Tensors["blk.2.attn_norm.weight"]
	x.Encoding = "I8"
	m.Tensors["blk.2.attn_norm.weight"] = x
	_, err := FromGGUF(m)
	if err == nil {
		t.Fatal("corrupted fixture unexpectedly passed")
	}
	var pe *PreflightError
	if !errors.As(err, &pe) {
		t.Fatalf("error type %T, want *PreflightError", err)
	}
	want := map[string]MismatchKind{"blk.0.attn_k.weight": MismatchMissing, "blk.1.ffn_down.weight": MismatchShape, "blk.2.attn_norm.weight": MismatchEncoding}
	for _, mm := range pe.Mismatches {
		if k, ok := want[mm.Tensor]; ok && k == mm.Kind {
			delete(want, mm.Tensor)
		}
	}
	if len(want) > 0 {
		t.Fatalf("full mismatch list omitted: %v; got %+v", want, pe.Mismatches)
	}
}

func TestPreflightListsShapeEncodingAndScaleTogether(t *testing.T) {
	path := fixtureQwenFP8(t, "Qwen3.5-9B-FP8", 4096, 12288, 32, 16, 4, 32, 0)
	// Rewrite three independent header entries, including a missing scale dependency.
	idx := readFixtureHeader(t, filepath.Join(path, "model.safetensors"))
	idx["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"] = TensorInfo{Shape: []int64{1, 1}, Encoding: "F8_E4M3"}
	idx["model.language_model.layers.1.linear_attn.in_proj_z.weight"] = TensorInfo{Shape: []int64{4096, 4096}, Encoding: "I8"}
	delete(idx, "model.language_model.layers.2.linear_attn.out_proj.weight_scale_inv")
	writeSafetensors(t, path, idx)
	_, err := FromConfigJSON(path)
	var pe *PreflightError
	if !errors.As(err, &pe) {
		t.Fatalf("got %v", err)
	}
	seen := map[MismatchKind]bool{}
	for _, m := range pe.Mismatches {
		seen[m.Kind] = true
	}
	for _, k := range []MismatchKind{MismatchShape, MismatchEncoding, MismatchScale} {
		if !seen[k] {
			t.Fatalf("missing mismatch kind %s: %+v", k, pe.Mismatches)
		}
	}
}

func TestDescriptorAndGetterAreImmutable(t *testing.T) {
	d, err := FromGGUF(gemmaGGUF("Q8_0", "Q8_0"))
	if err != nil {
		t.Fatal(err)
	}
	s := d.Snapshot()
	s.Mixers[0] = MixerGatedDeltaNet
	s.Family = "corrupt"
	if d.Family() == "corrupt" || d.Mixers()[0] == MixerGatedDeltaNet {
		t.Fatal("snapshot mutated descriptor")
	}
	mix := d.Mixers()
	mix[0] = MixerGatedDeltaNet
	if d.Mixers()[0] == MixerGatedDeltaNet {
		t.Fatal("slice getter leaked backing array")
	}
	store := NewStore(d)
	got, ok := store.Get(d.ID())
	if !ok || got != d {
		t.Fatal("getter did not publish descriptor")
	}
	if _, ok := store.Get("other"); ok {
		t.Fatal("getter matched wrong id")
	}
}

func assertFingerprints(t *testing.T, d *ModelDescriptor) {
	t.Helper()
	f := d.Fingerprints()
	for name, v := range map[string]string{"model": f.Model, "tokenizer": f.Tokenizer} {
		if len(v) != 64 {
			t.Fatalf("%s fingerprint length=%d", name, len(v))
		}
		if _, err := fmt.Sscanf(v, "%x", new([]byte)); err != nil {
			t.Fatalf("%s fingerprint not hex: %q", name, v)
		}
	}
}

func gemmaGGUF(bulk, embed string) GGUFMetadata {
	const L, H, I, NH, V = 48, 3840, 15360, 16, 262144
	pat := make([]bool, L)
	kv := make([]int, L)
	for l := 0; l < L; l++ {
		pat[l] = (l+1)%6 != 0
		if pat[l] {
			kv[l] = 8
		} else {
			kv[l] = 1
		}
	}
	m := GGUFMetadata{Values: map[string]any{"general.architecture": "gemma4", "general.name": "Gemma-4-12B-it", "gemma4.block_count": L, "gemma4.embedding_length": H, "gemma4.feed_forward_length": I, "gemma4.attention.head_count": NH, "gemma4.attention.head_count_kv": kv, "gemma4.attention.sliding_window_pattern": pat, "gemma4.vocab_size": V, "gemma4.attention.key_length_swa": 256, "gemma4.attention.key_length": 512, "gemma4.rope.freq_base": 1e6, "gemma4.rope.freq_base_swa": 1e4, "tokenizer.ggml.model": "llama", "tokenizer.ggml.token_count": V, "tokenizer.chat_template": "gemma"}, Tensors: map[string]TensorInfo{}}
	add := func(n, e string, s ...int) { m.Tensors[n] = TensorInfo{Shape: i64(s...), Encoding: e} }
	add("token_embd.weight", embed, H, V)
	add("output_norm.weight", "F32", H)
	for l := 0; l < L; l++ {
		hd, nkv := 256, 8
		if !pat[l] {
			hd = 512
			nkv = 1
		}
		p := fmt.Sprintf("blk.%d.", l)
		add(p+"attn_q.weight", bulk, H, NH*hd)
		add(p+"attn_k.weight", bulk, H, nkv*hd)
		if pat[l] {
			add(p+"attn_v.weight", bulk, H, nkv*hd)
		}
		add(p+"attn_output.weight", bulk, NH*hd, H)
		for _, n := range []string{"attn_norm.weight", "post_attention_norm.weight", "ffn_norm.weight", "post_ffw_norm.weight"} {
			add(p+n, "F32", H)
		}
		add(p+"attn_q_norm.weight", "F32", hd)
		add(p+"attn_k_norm.weight", "F32", hd)
		add(p+"ffn_gate.weight", bulk, H, I)
		add(p+"ffn_up.weight", bulk, H, I)
		add(p+"ffn_down.weight", bulk, I, H)
		add(p+"layer_output_scale.weight", "F32", 1)
	}
	return m
}

func fixtureGemmaNVFP4(t *testing.T) string {
	t.Helper()
	const L, H, I, NH, NKV, V = 48, 3840, 15360, 16, 8, 262144
	dir := fixtureDir(t, "gemma-4-12B-it-NVFP4")
	layers := make([]string, L)
	for i := range layers {
		if (i+1)%6 == 0 {
			layers[i] = "full_attention"
		} else {
			layers[i] = "sliding_attention"
		}
	}
	cfg := map[string]any{"model_type": "gemma4_unified_text", "_name_or_path": "RedHatAI/gemma-4-12B-it-NVFP4", "hidden_size": H, "intermediate_size": I, "num_hidden_layers": L, "num_attention_heads": NH, "num_key_value_heads": NKV, "num_global_key_value_heads": 1, "head_dim": 256, "global_head_dim": 512, "vocab_size": V, "tie_word_embeddings": true, "layer_types": layers, "quantization_config": map[string]any{"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized"}}
	tensors := map[string]TensorInfo{"model.embed_tokens.weight": ti("BF16", V, H), "model.norm.weight": ti("BF16", H)}
	for l := 0; l < L; l++ {
		hd, nkv := 256, NKV
		if layers[l] == "full_attention" {
			hd = 512
			nkv = 1
		}
		p := fmt.Sprintf("model.layers.%d.", l)
		addNV(tensors, p+"self_attn.q_proj.weight", NH*hd, H)
		addNV(tensors, p+"self_attn.k_proj.weight", nkv*hd, H)
		addNV(tensors, p+"self_attn.v_proj.weight", nkv*hd, H)
		addNV(tensors, p+"self_attn.o_proj.weight", H, NH*hd)
		addNV(tensors, p+"mlp.gate_proj.weight", I, H)
		addNV(tensors, p+"mlp.up_proj.weight", I, H)
		addNV(tensors, p+"mlp.down_proj.weight", H, I)
		for _, n := range []string{"input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"} {
			tensors[p+n] = ti("BF16", H)
		}
		tensors[p+"self_attn.q_norm.weight"] = ti("BF16", hd)
		tensors[p+"self_attn.k_norm.weight"] = ti("BF16", hd)
	}
	writeConfigFixture(t, dir, cfg, tensors)
	return dir
}

func fixtureQwenFP8(t *testing.T, name string, H, I, L, NQ, NKV, NVH, E int) string {
	t.Helper()
	dir := fixtureDir(t, name)
	moe := E > 0
	cfgText := map[string]any{"model_type": "qwen3_5_text", "hidden_size": H, "num_hidden_layers": L, "num_attention_heads": NQ, "num_key_value_heads": NKV, "head_dim": 256, "full_attention_interval": 4, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16, "linear_num_value_heads": NVH, "linear_value_head_dim": 128, "vocab_size": 248320, "max_position_embeddings": 262144, "rope_parameters": map[string]any{"rope_theta": 1e7, "partial_rotary_factor": 0.25}}
	if moe {
		cfgText["model_type"] = "qwen3_5_moe_text"
		cfgText["num_experts"] = E
		cfgText["num_experts_per_tok"] = 8
		cfgText["moe_intermediate_size"] = 512
		cfgText["shared_expert_intermediate_size"] = I
		cfgText["num_shared_experts"] = 1
	} else {
		cfgText["intermediate_size"] = I
	}
	cfg := map[string]any{"model_type": "qwen3_5", "text_config": cfgText, "tie_word_embeddings": false, "quantization_config": map[string]any{"quant_method": "fp8", "weight_block_size": []int{128, 128}}}
	r := "model.language_model."
	tensors := map[string]TensorInfo{r + "embed_tokens.weight": ti("BF16", 248320, H), r + "norm.weight": ti("BF16", H), "lm_head.weight": ti("BF16", 248320, H)}
	inner := NVH * 128
	convd := 2*16*128 + inner
	for l := 0; l < L; l++ {
		p := fmt.Sprintf("%slayers.%d.", r, l)
		tensors[p+"input_layernorm.weight"] = ti("BF16", H)
		tensors[p+"post_attention_layernorm.weight"] = ti("BF16", H)
		if (l+1)%4 == 0 {
			addFP8(tensors, p+"self_attn.q_proj.weight", 2*NQ*256, H)
			addFP8(tensors, p+"self_attn.k_proj.weight", NKV*256, H)
			addFP8(tensors, p+"self_attn.v_proj.weight", NKV*256, H)
			addFP8(tensors, p+"self_attn.o_proj.weight", H, NQ*256)
			tensors[p+"self_attn.q_norm.weight"] = ti("BF16", 256)
			tensors[p+"self_attn.k_norm.weight"] = ti("BF16", 256)
		} else {
			addFP8(tensors, p+"linear_attn.in_proj_qkv.weight", convd, H)
			addFP8(tensors, p+"linear_attn.in_proj_z.weight", inner, H)
			addFP8(tensors, p+"linear_attn.out_proj.weight", H, inner)
			tensors[p+"linear_attn.in_proj_a.weight"] = ti("BF16", NVH, H)
			tensors[p+"linear_attn.in_proj_b.weight"] = ti("BF16", NVH, H)
			tensors[p+"linear_attn.conv1d.weight"] = ti("BF16", convd, 1, 4)
			tensors[p+"linear_attn.A_log"] = ti("F32", NVH)
			tensors[p+"linear_attn.dt_bias"] = ti("BF16", NVH)
			tensors[p+"linear_attn.norm.weight"] = ti("F32", 128)
		}
		if moe {
			tensors[p+"mlp.shared_expert_gate.weight"] = ti("BF16", H)
			addFP8(tensors, p+"mlp.shared_expert.gate_proj.weight", I, H)
			addFP8(tensors, p+"mlp.shared_expert.up_proj.weight", I, H)
			addFP8(tensors, p+"mlp.shared_expert.down_proj.weight", H, I)
			tensors[p+"mlp.gate.weight"] = ti("BF16", E, H)
			for e := 0; e < E; e++ {
				ep := fmt.Sprintf("%smlp.experts.%d.", p, e)
				addFP8(tensors, ep+"gate_proj.weight", 512, H)
				addFP8(tensors, ep+"up_proj.weight", 512, H)
				addFP8(tensors, ep+"down_proj.weight", H, 512)
			}
		} else {
			addFP8(tensors, p+"mlp.gate_proj.weight", I, H)
			addFP8(tensors, p+"mlp.up_proj.weight", I, H)
			addFP8(tensors, p+"mlp.down_proj.weight", H, I)
		}
	}
	writeConfigFixture(t, dir, cfg, tensors)
	return dir
}

func fixtureE4B(t *testing.T) string {
	t.Helper()
	const L, H, I, NH, NKV, V, PD = 42, 2560, 10240, 8, 2, 262144, 256
	dir := fixtureDir(t, "gemma-4-E4B")
	layers := make([]string, L)
	for i := range layers {
		if (i+1)%6 == 0 {
			layers[i] = "full_attention"
		} else {
			layers[i] = "sliding_attention"
		}
	}
	text := map[string]any{"model_type": "gemma4_text", "hidden_size": H, "intermediate_size": I, "num_hidden_layers": L, "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": 256, "global_head_dim": 512, "vocab_size": V, "hidden_size_per_layer_input": PD, "vocab_size_per_layer_input": V, "num_kv_shared_layers": 18, "layer_types": layers, "rope_parameters": map[string]any{"sliding_attention": map[string]any{"rope_theta": 1e4}, "full_attention": map[string]any{"rope_theta": 1e6, "partial_rotary_factor": 0.25}}}
	cfg := map[string]any{"model_type": "gemma4", "text_config": text, "tie_word_embeddings": true}
	r := "model.language_model."
	width := L * PD
	tensors := map[string]TensorInfo{r + "embed_tokens.weight": ti("BF16", V, H), r + "embed_tokens_per_layer.weight": ti("BF16", V, width), r + "per_layer_model_projection.weight": ti("BF16", width, H), r + "per_layer_projection_norm.weight": ti("BF16", PD), r + "norm.weight": ti("BF16", H)}
	for l := 0; l < L; l++ {
		hd := 256
		if layers[l] == "full_attention" {
			hd = 512
		}
		p := fmt.Sprintf("%slayers.%d.", r, l)
		tensors[p+"self_attn.q_proj.weight"] = ti("BF16", NH*hd, H)
		tensors[p+"self_attn.k_proj.weight"] = ti("BF16", NKV*hd, H)
		tensors[p+"self_attn.v_proj.weight"] = ti("BF16", NKV*hd, H)
		tensors[p+"self_attn.o_proj.weight"] = ti("BF16", H, NH*hd)
		tensors[p+"self_attn.q_norm.weight"] = ti("BF16", hd)
		tensors[p+"self_attn.k_norm.weight"] = ti("BF16", hd)
		tensors[p+"mlp.gate_proj.weight"] = ti("BF16", I, H)
		tensors[p+"mlp.up_proj.weight"] = ti("BF16", I, H)
		tensors[p+"mlp.down_proj.weight"] = ti("BF16", H, I)
		for _, n := range []string{"input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "post_per_layer_input_norm.weight"} {
			tensors[p+n] = ti("BF16", H)
		}
		tensors[p+"per_layer_input_gate.weight"] = ti("BF16", PD, H)
		tensors[p+"per_layer_projection.weight"] = ti("BF16", H, PD)
	}
	writeConfigFixture(t, dir, cfg, tensors)
	return dir
}

func ti(enc string, dims ...int) TensorInfo { return TensorInfo{Encoding: enc, Shape: i64(dims...)} }
func addFP8(m map[string]TensorInfo, name string, out, in int) {
	m[name] = ti("F8_E4M3", out, in)
	m[name+"_scale_inv"] = ti("BF16", (out+127)/128, (in+127)/128)
}
func addNV(m map[string]TensorInfo, name string, out, in int) {
	base := strings.TrimSuffix(name, "weight")
	m[base+"weight_packed"] = ti("U8", out, in/2)
	m[base+"weight_scale"] = ti("F8_E4M3", out, (in+15)/16)
	m[base+"weight_global_scale"] = ti("F32")
}
func fixtureDir(t *testing.T, name string) string {
	d := filepath.Join(t.TempDir(), name)
	if err := os.MkdirAll(d, 0o755); err != nil {
		t.Fatal(err)
	}
	return d
}
func writeConfigFixture(t *testing.T, dir string, cfg map[string]any, tensors map[string]TensorInfo) {
	t.Helper()
	b, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{"model":{"type":"BPE"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"chat_template":"{% generation %}"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	writeSafetensors(t, dir, tensors)
}
func writeSafetensors(t *testing.T, dir string, tensors map[string]TensorInfo) {
	t.Helper()
	h := map[string]any{"__metadata__": map[string]string{"format": "pt"}}
	names := sortedTensorNames(tensors)
	off := int64(0)
	for _, n := range names {
		x := tensors[n]
		elements := int64(1)
		for _, d := range x.Shape {
			elements *= d
		}
		width, ok := safetensorsDTypeBytes(x.Encoding)
		if !ok {
			width = 1
		}
		end := off + elements*width
		h[n] = map[string]any{"dtype": x.Encoding, "shape": x.Shape, "data_offsets": []int64{off, end}}
		off = end
	}
	b, err := json.Marshal(h)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var p [8]byte
	binary.LittleEndian.PutUint64(p[:], uint64(len(b)))
	if _, err = f.Write(p[:]); err != nil {
		t.Fatal(err)
	}
	if _, err = f.Write(b); err != nil {
		t.Fatal(err)
	}
	// Payloads are sparse: preflight needs the real byte ranges, never the weight contents.
	if err := f.Truncate(8 + int64(len(b)) + off); err != nil {
		t.Fatal(err)
	}
}
func readFixtureHeader(t *testing.T, path string) map[string]TensorInfo {
	t.Helper()
	m, err := readSafetensorsHeader(path)
	if err != nil {
		t.Fatal(err)
	}
	return m
}
