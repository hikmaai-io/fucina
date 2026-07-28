// Package model contains the immutable, source-derived description of a loaded model.
//
// Builders in this package are deliberately host-only. They inspect checkpoint metadata and
// tensor indexes and complete preflight before a caller is allowed to allocate CUDA memory.
package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
)

type SourceFormat string
type Qualification string
type MixerType string

const (
	FormatGGUF        SourceFormat = "gguf"
	FormatSafetensors SourceFormat = "safetensors"

	QualificationValidated    Qualification = "validated"
	QualificationLoads        Qualification = "loads"
	QualificationExperimental Qualification = "experimental"

	MixerSlidingAttention MixerType = "sliding-attention"
	MixerFullAttention    MixerType = "full-attention"
	MixerGatedDeltaNet    MixerType = "gated-deltanet"
)

// TensorInfo is the allocation-free portion of a checkpoint tensor index. GGUF shapes use GGUF
// order ([in,out]); safetensors shapes use row-major HF order ([out,in]).
type TensorInfo struct {
	Shape    []int64 `json:"shape"`
	Encoding string  `json:"encoding"`
	Bytes    int64   `json:"bytes,omitempty"`
}

// GGUFMetadata is the input contract between a GGUF reader and FromGGUF. Values must include the
// architecture metadata used by the CUDA detector; Tensors must contain the complete GGUF tensor
// index. The builder never retains either map.
type GGUFMetadata struct {
	Values  map[string]any
	Tensors map[string]TensorInfo
}

type ServedQuantization struct {
	Attention string `json:"attention"`
	FFN       string `json:"ffn"`
	Experts   string `json:"experts,omitempty"`
	Embedding string `json:"embedding"`
	LMHead    string `json:"lm_head"`
	Norms     string `json:"norms"`
}

type Geometry struct {
	Layers        int `json:"layers"`
	HiddenSize    int `json:"hidden_size"`
	Intermediate  int `json:"intermediate_size"`
	AttentionHead int `json:"attention_heads"`
	KVHeads       int `json:"kv_heads"`
	HeadDim       int `json:"head_dim"`
	GlobalKVHeads int `json:"global_kv_heads,omitempty"`
	GlobalHeadDim int `json:"global_head_dim,omitempty"`
	VocabSize     int `json:"vocab_size"`
}

type MoEGeometry struct {
	Experts                  int `json:"experts"`
	TopK                     int `json:"top_k"`
	ExpertIntermediate       int `json:"expert_intermediate"`
	SharedExperts            int `json:"shared_experts"`
	SharedExpertIntermediate int `json:"shared_expert_intermediate"`
}

type AttentionConfig struct {
	RoPETheta             float64 `json:"rope_theta"`
	SlidingRoPETheta      float64 `json:"sliding_rope_theta,omitempty"`
	RotaryDim             int     `json:"rotary_dim,omitempty"`
	PartialRotaryFactor   float64 `json:"partial_rotary_factor,omitempty"`
	SlidingWindow         int     `json:"sliding_window,omitempty"`
	FullAttentionEvery    int     `json:"full_attention_every,omitempty"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings,omitempty"`
}

type GDNGeometry struct {
	StateSize    int `json:"state_size"`
	ConvKernel   int `json:"conv_kernel"`
	InnerSize    int `json:"inner_size"`
	GroupCount   int `json:"group_count"`
	TimeStepRank int `json:"time_step_rank"`
	KeyDim       int `json:"key_dim"`
	ConvDim      int `json:"conv_dim"`
}

type TokenizerCapabilities struct {
	Kind            string `json:"kind"`
	Available       bool   `json:"available"`
	ChatTemplate    bool   `json:"chat_template"`
	Tools           bool   `json:"tools"`
	ThinkingControl bool   `json:"thinking_control"`
}

type ExecutionCapabilities struct {
	PagedKV                   bool `json:"paged_kv"`
	PrefixCache               bool `json:"prefix_cache"`
	ContinuousBatching        bool `json:"continuous_batching"`
	ContinuousBatchingDefault bool `json:"continuous_batching_default"`
	BatchMTP                  bool `json:"batch_mtp"`
	E4BServing                bool `json:"e4b_serving"`
	LegacyQwen3               bool `json:"legacy_qwen3"`
	StructuredOutput          bool `json:"structured_output"`
	ToolCalling               bool `json:"tool_calling"`
	OpenAIChatCompletions     bool `json:"openai_chat_completions"`
	OpenAICompletions         bool `json:"openai_completions"`
	AnthropicMessages         bool `json:"anthropic_messages"`
}

type Fingerprints struct {
	Model     string `json:"model_sha256"`
	Tokenizer string `json:"tokenizer_sha256"`
}

// DescriptorSnapshot is a detached representation suitable for GET /v1/models/{id}. Mutating a
// snapshot cannot mutate its ModelDescriptor.
type DescriptorSnapshot struct {
	ID            string                `json:"id"`
	Family        string                `json:"family"`
	Architecture  string                `json:"architecture"`
	Variant       string                `json:"variant"`
	SourceFormat  SourceFormat          `json:"source_format"`
	SourceQuant   string                `json:"source_quantization"`
	ServedQuant   ServedQuantization    `json:"served_quantization"`
	Qualification Qualification         `json:"qualification"`
	Geometry      Geometry              `json:"geometry"`
	MoE           MoEGeometry           `json:"moe"`
	Mixers        []MixerType           `json:"mixers"`
	Attention     AttentionConfig       `json:"attention"`
	GDN           GDNGeometry           `json:"gdn"`
	Tokenizer     TokenizerCapabilities `json:"tokenizer"`
	Capabilities  ExecutionCapabilities `json:"capabilities"`
	Fingerprints  Fingerprints          `json:"fingerprints"`
}

// ModelDescriptor is immutable after construction. All state is private; Snapshot and slice
// getters return copies.
type ModelDescriptor struct{ snapshot DescriptorSnapshot }

func (d *ModelDescriptor) ID() string                             { return d.snapshot.ID }
func (d *ModelDescriptor) Family() string                         { return d.snapshot.Family }
func (d *ModelDescriptor) Architecture() string                   { return d.snapshot.Architecture }
func (d *ModelDescriptor) Variant() string                        { return d.snapshot.Variant }
func (d *ModelDescriptor) SourceFormat() SourceFormat             { return d.snapshot.SourceFormat }
func (d *ModelDescriptor) SourceQuantization() string             { return d.snapshot.SourceQuant }
func (d *ModelDescriptor) Qualification() Qualification           { return d.snapshot.Qualification }
func (d *ModelDescriptor) Geometry() Geometry                     { return d.snapshot.Geometry }
func (d *ModelDescriptor) MoEGeometry() MoEGeometry               { return d.snapshot.MoE }
func (d *ModelDescriptor) Attention() AttentionConfig             { return d.snapshot.Attention }
func (d *ModelDescriptor) GDNGeometry() GDNGeometry               { return d.snapshot.GDN }
func (d *ModelDescriptor) Tokenizer() TokenizerCapabilities       { return d.snapshot.Tokenizer }
func (d *ModelDescriptor) Capabilities() ExecutionCapabilities    { return d.snapshot.Capabilities }
func (d *ModelDescriptor) ServedQuantization() ServedQuantization { return d.snapshot.ServedQuant }
func (d *ModelDescriptor) Fingerprints() Fingerprints             { return d.snapshot.Fingerprints }
func (d *ModelDescriptor) Mixers() []MixerType                    { return append([]MixerType(nil), d.snapshot.Mixers...) }
func (d *ModelDescriptor) Snapshot() DescriptorSnapshot           { return cloneSnapshot(d.snapshot) }

func (d *ModelDescriptor) MarshalJSON() ([]byte, error) { return json.Marshal(d.Snapshot()) }

func cloneSnapshot(in DescriptorSnapshot) DescriptorSnapshot {
	out := in
	out.Mixers = append([]MixerType(nil), in.Mixers...)
	return out
}

// Getter is the integration surface for the later GET /v1/models/{id} route.
type Getter interface{ ModelDescriptor() *ModelDescriptor }

// Store publishes an immutable descriptor to readers without locks.
type Store struct {
	ptr atomic.Pointer[ModelDescriptor]
}

func NewStore(d *ModelDescriptor) *Store           { s := &Store{}; s.Set(d); return s }
func (s *Store) Set(d *ModelDescriptor)            { s.ptr.Store(d) }
func (s *Store) ModelDescriptor() *ModelDescriptor { return s.ptr.Load() }
func (s *Store) Get(id string) (*ModelDescriptor, bool) {
	d := s.ptr.Load()
	return d, d != nil && (id == "" || id == d.ID())
}

type MismatchKind string

const (
	MismatchMetadata MismatchKind = "metadata"
	MismatchMissing  MismatchKind = "missing"
	MismatchShape    MismatchKind = "shape"
	MismatchEncoding MismatchKind = "encoding"
	MismatchScale    MismatchKind = "scale_dependency"
)

// TensorMismatch is one independently actionable preflight failure.
type TensorMismatch struct {
	Tensor   string       `json:"tensor"`
	Kind     MismatchKind `json:"kind"`
	Expected string       `json:"expected"`
	Actual   string       `json:"actual"`
}

// PreflightError contains every mismatch found in one complete host-side pass.
type PreflightError struct{ Mismatches []TensorMismatch }

func (e *PreflightError) Error() string {
	if e == nil || len(e.Mismatches) == 0 {
		return "model preflight failed"
	}
	return fmt.Sprintf("model preflight failed with %d mismatch(es): %s", len(e.Mismatches), e.Mismatches[0].Tensor)
}

func (e *PreflightError) Is(target error) bool { _, ok := target.(*PreflightError); return ok }

func mismatch(errs *[]TensorMismatch, tensor string, kind MismatchKind, expected, actual string) {
	*errs = append(*errs, TensorMismatch{Tensor: tensor, Kind: kind, Expected: expected, Actual: actual})
}

func finish(s DescriptorSnapshot, tensors map[string]TensorInfo, metadata any, tokenizerSeed []byte, errs []TensorMismatch) (*ModelDescriptor, error) {
	if len(errs) != 0 {
		sort.SliceStable(errs, func(i, j int) bool {
			if errs[i].Tensor == errs[j].Tensor {
				return errs[i].Kind < errs[j].Kind
			}
			return errs[i].Tensor < errs[j].Tensor
		})
		return nil, &PreflightError{Mismatches: errs}
	}
	s.Fingerprints.Model = fingerprint(metadata, tensors)
	h := sha256.Sum256(tokenizerSeed)
	s.Fingerprints.Tokenizer = hex.EncodeToString(h[:])
	s.ID = descriptorID(s)
	return &ModelDescriptor{snapshot: cloneSnapshot(s)}, nil
}

func descriptorID(s DescriptorSnapshot) string {
	v := strings.ToLower(strings.TrimSpace(s.Variant))
	v = strings.NewReplacer(" ", "-", "/", "-", "_", "-").Replace(v)
	for strings.Contains(v, "--") {
		v = strings.ReplaceAll(v, "--", "-")
	}
	if v == "" {
		v = strings.ToLower(s.Architecture)
	}
	q := strings.ToLower(strings.NewReplacer("_", "-", " ", "-").Replace(s.SourceQuant))
	return strings.Trim(strings.ToLower(s.Family)+"-"+v+"-"+q, "-")
}

func fingerprint(metadata any, tensors map[string]TensorInfo) string {
	h := sha256.New()
	b, _ := json.Marshal(metadata)
	h.Write(b)
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	enc := json.NewEncoder(h)
	for _, name := range names {
		t := tensors[name]
		_ = enc.Encode(struct {
			Name     string  `json:"name"`
			Shape    []int64 `json:"shape"`
			Encoding string  `json:"encoding"`
			Bytes    int64   `json:"bytes"`
		}{name, t.Shape, canonicalEncoding(t.Encoding), t.Bytes})
	}
	return hex.EncodeToString(h.Sum(nil))
}

func canonicalEncoding(s string) string {
	s = strings.ToUpper(strings.TrimSpace(s))
	s = strings.ReplaceAll(s, "FLOAT8_E4M3FN", "F8_E4M3")
	s = strings.ReplaceAll(s, "FP8_E4M3", "F8_E4M3")
	return s
}

func intValue(m map[string]any, keys ...string) (int, bool) {
	for _, key := range keys {
		v, ok := m[key]
		if !ok {
			continue
		}
		switch x := v.(type) {
		case int:
			return x, true
		case int32:
			return int(x), true
		case int64:
			return int(x), true
		case uint32:
			return int(x), true
		case uint64:
			if x <= math.MaxInt {
				return int(x), true
			}
		case float64:
			if x == math.Trunc(x) {
				return int(x), true
			}
		case json.Number:
			n, err := strconv.Atoi(x.String())
			if err == nil {
				return n, true
			}
		}
	}
	return 0, false
}

func floatValue(m map[string]any, keys ...string) (float64, bool) {
	for _, key := range keys {
		v, ok := m[key]
		if !ok {
			continue
		}
		switch x := v.(type) {
		case float64:
			return x, true
		case float32:
			return float64(x), true
		case int:
			return float64(x), true
		case json.Number:
			f, err := x.Float64()
			if err == nil {
				return f, true
			}
		}
	}
	return 0, false
}

func stringValue(m map[string]any, keys ...string) (string, bool) {
	for _, key := range keys {
		if s, ok := m[key].(string); ok {
			return s, true
		}
	}
	return "", false
}

func intSlice(m map[string]any, key string) []int {
	v := m[key]
	switch x := v.(type) {
	case []int:
		return append([]int(nil), x...)
	case []int32:
		out := make([]int, len(x))
		for i, n := range x {
			out[i] = int(n)
		}
		return out
	case []any:
		out := make([]int, 0, len(x))
		for _, v := range x {
			n, ok := intValue(map[string]any{"x": v}, "x")
			if ok {
				out = append(out, n)
			}
		}
		return out
	}
	return nil
}

func boolSlice(m map[string]any, key string) []bool {
	v := m[key]
	switch x := v.(type) {
	case []bool:
		return append([]bool(nil), x...)
	case []any:
		out := make([]bool, 0, len(x))
		for _, v := range x {
			if b, ok := v.(bool); ok {
				out = append(out, b)
			}
		}
		return out
	}
	return nil
}

func requiredInt(m map[string]any, errs *[]TensorMismatch, label string, keys ...string) int {
	if n, ok := intValue(m, keys...); ok && n > 0 {
		return n
	}
	mismatch(errs, "metadata."+label, MismatchMetadata, "positive integer", "missing or invalid")
	return 0
}

// FromGGUF derives and preflights a descriptor from already parsed GGUF metadata and tensor index.
func FromGGUF(in GGUFMetadata) (*ModelDescriptor, error) {
	values := cloneMap(in.Values)
	tensors := cloneTensors(in.Tensors)
	var errs []TensorMismatch
	arch, _ := stringValue(values, "general.architecture")
	arch = strings.ToLower(arch)
	if arch == "" {
		mismatch(&errs, "metadata.general.architecture", MismatchMetadata, "gemma4 or qwen35", "missing")
	}
	var s DescriptorSnapshot
	s.SourceFormat = FormatGGUF
	s.Tokenizer = ggufTokenizer(values)
	s.Capabilities = baseCapabilities()

	switch arch {
	case "gemma4":
		s = buildGemmaGGUF(s, values, tensors, &errs)
	case "qwen35":
		s = buildQwenGGUF(s, values, tensors, &errs)
	default:
		if arch != "" {
			mismatch(&errs, "metadata.general.architecture", MismatchMetadata, "gemma4 or qwen35 (legacy qwen3 is intentionally unsupported)", arch)
		}
	}
	tokSeed, _ := json.Marshal(filterMap(values, "tokenizer."))
	return finish(s, tensors, values, tokSeed, errs)
}

func buildGemmaGGUF(s DescriptorSnapshot, m map[string]any, tensors map[string]TensorInfo, errs *[]TensorMismatch) DescriptorSnapshot {
	s.Family, s.Architecture = "gemma-4", "gemma4"
	g := Geometry{}
	g.Layers = requiredInt(m, errs, "gemma4.block_count", "gemma4.block_count")
	g.HiddenSize = requiredInt(m, errs, "gemma4.embedding_length", "gemma4.embedding_length")
	g.Intermediate = requiredInt(m, errs, "gemma4.feed_forward_length", "gemma4.feed_forward_length")
	g.AttentionHead = requiredInt(m, errs, "gemma4.attention.head_count", "gemma4.attention.head_count")
	g.VocabSize, _ = intValue(m, "gemma4.vocab_size")
	if g.VocabSize <= 0 {
		g.VocabSize, _ = intValue(m, "tokenizer.ggml.token_count")
	}
	if g.VocabSize <= 0 {
		mismatch(errs, "metadata.gemma4.vocab_size", MismatchMetadata, "positive integer or tokenizer token count", "missing")
	}
	g.HeadDim, _ = intValue(m, "gemma4.attention.key_length_swa")
	if g.HeadDim == 0 {
		g.HeadDim = 256
	}
	g.GlobalHeadDim, _ = intValue(m, "gemma4.attention.key_length")
	if g.GlobalHeadDim == 0 {
		g.GlobalHeadDim = 512
	}
	kv := intSlice(m, "gemma4.attention.head_count_kv")
	if len(kv) < g.Layers {
		mismatch(errs, "metadata.gemma4.attention.head_count_kv", MismatchMetadata, fmt.Sprintf("%d entries", g.Layers), fmt.Sprintf("%d entries", len(kv)))
	}
	if len(kv) > 0 {
		g.KVHeads, g.GlobalKVHeads = kv[0], kv[0]
		for _, n := range kv {
			if n > g.KVHeads {
				g.KVHeads = n
			}
			if n < g.GlobalKVHeads {
				g.GlobalKVHeads = n
			}
		}
	}
	pat := boolSlice(m, "gemma4.attention.sliding_window_pattern")
	if len(pat) < g.Layers {
		mismatch(errs, "metadata.gemma4.attention.sliding_window_pattern", MismatchMetadata, fmt.Sprintf("%d entries", g.Layers), fmt.Sprintf("%d entries", len(pat)))
	}
	s.Mixers = make([]MixerType, g.Layers)
	for i := range s.Mixers {
		if i < len(pat) && !pat[i] {
			s.Mixers[i] = MixerFullAttention
		} else {
			s.Mixers[i] = MixerSlidingAttention
		}
	}
	s.Geometry = g
	s.Attention.RoPETheta, _ = floatValue(m, "gemma4.rope.freq_base")
	s.Attention.SlidingRoPETheta, _ = floatValue(m, "gemma4.rope.freq_base_swa")
	s.Attention.SlidingWindow, _ = intValue(m, "gemma4.attention.sliding_window")
	s.SourceQuant = detectGGUFQuant(tensors)
	s.Variant = variantValue(m, fmt.Sprintf("%dB", approximateBillions(g.HiddenSize, g.Layers, g.Intermediate)))
	s.ServedQuant = servedFor("gemma4", s.SourceQuant, false)
	s.Qualification = qualificationFor(s)
	s.Capabilities.BatchMTP = true
	preflightGemmaGGUF(s, tensors, errs)
	return s
}

func buildQwenGGUF(s DescriptorSnapshot, m map[string]any, tensors map[string]TensorInfo, errs *[]TensorMismatch) DescriptorSnapshot {
	s.Family, s.Architecture = "qwen3.5", "qwen35"
	g := Geometry{}
	g.Layers = requiredInt(m, errs, "qwen35.block_count", "qwen35.block_count")
	g.HiddenSize = requiredInt(m, errs, "qwen35.embedding_length", "qwen35.embedding_length")
	g.Intermediate = requiredInt(m, errs, "qwen35.feed_forward_length", "qwen35.feed_forward_length")
	g.AttentionHead = requiredInt(m, errs, "qwen35.attention.head_count", "qwen35.attention.head_count")
	g.KVHeads = requiredInt(m, errs, "qwen35.attention.head_count_kv", "qwen35.attention.head_count_kv")
	g.GlobalKVHeads = g.KVHeads
	g.HeadDim = requiredInt(m, errs, "qwen35.attention.key_length", "qwen35.attention.key_length")
	g.VocabSize, _ = intValue(m, "qwen35.vocab_size", "tokenizer.ggml.token_count")
	if g.VocabSize <= 0 {
		mismatch(errs, "metadata.qwen35.vocab_size", MismatchMetadata, "positive integer", "missing")
	}
	s.Geometry = g
	s.Attention.RoPETheta, _ = floatValue(m, "qwen35.rope.freq_base")
	s.Attention.RotaryDim, _ = intValue(m, "qwen35.rope.dimension_count")
	s.Attention.FullAttentionEvery, _ = intValue(m, "qwen35.full_attention_interval")
	if s.Attention.FullAttentionEvery <= 0 {
		s.Attention.FullAttentionEvery = 4
	}
	s.GDN.StateSize = requiredInt(m, errs, "qwen35.ssm.state_size", "qwen35.ssm.state_size")
	s.GDN.ConvKernel = requiredInt(m, errs, "qwen35.ssm.conv_kernel", "qwen35.ssm.conv_kernel")
	s.GDN.InnerSize = requiredInt(m, errs, "qwen35.ssm.inner_size", "qwen35.ssm.inner_size")
	s.GDN.GroupCount = requiredInt(m, errs, "qwen35.ssm.group_count", "qwen35.ssm.group_count")
	s.GDN.TimeStepRank = requiredInt(m, errs, "qwen35.ssm.time_step_rank", "qwen35.ssm.time_step_rank")
	s.GDN.KeyDim = s.GDN.GroupCount * s.GDN.StateSize
	s.GDN.ConvDim = 2*s.GDN.KeyDim + s.GDN.InnerSize
	s.Mixers = hybridMixers(g.Layers, s.Attention.FullAttentionEvery)
	s.SourceQuant = detectGGUFQuant(tensors)
	s.Variant = variantValue(m, "9B-dense")
	s.ServedQuant = servedFor("qwen35", s.SourceQuant, false)
	s.Qualification = qualificationFor(s)
	s.Capabilities.BatchMTP = false
	preflightQwenGGUF(s, tensors, errs)
	return s
}

func baseCapabilities() ExecutionCapabilities {
	return ExecutionCapabilities{
		PagedKV: true, PrefixCache: true, ContinuousBatching: true,
		ContinuousBatchingDefault: true, LegacyQwen3: false, StructuredOutput: true,
		ToolCalling: true, OpenAIChatCompletions: true, OpenAICompletions: true, AnthropicMessages: true,
	}
}

func ggufTokenizer(m map[string]any) TokenizerCapabilities {
	kind, _ := stringValue(m, "tokenizer.ggml.model")
	_, hasTokens := m["tokenizer.ggml.tokens"]
	_, hasCount := m["tokenizer.ggml.token_count"]
	_, hasTemplate := m["tokenizer.chat_template"]
	return TokenizerCapabilities{Kind: kind, Available: hasTokens || hasCount, ChatTemplate: hasTemplate, Tools: hasTemplate, ThinkingControl: hasTemplate}
}

func variantValue(m map[string]any, fallback string) string {
	if v, ok := stringValue(m, "general.name", "general.basename", "_name_or_path"); ok && v != "" {
		return v
	}
	return fallback
}

func detectGGUFQuant(t map[string]TensorInfo) string {
	for _, name := range []string{"blk.0.attn_q.weight", "blk.0.ffn_gate.weight"} {
		if x, ok := t[name]; ok {
			e := canonicalEncoding(x.Encoding)
			if e == "Q4_0" {
				if emb, ok := t["token_embd.weight"]; ok && canonicalEncoding(emb.Encoding) == "Q6_K" {
					return "Q4_0-QAT"
				}
				return "Q4_0"
			}
			return e
		}
	}
	return "unknown"
}

func servedFor(arch, source string, moe bool) ServedQuantization {
	source = canonicalEncoding(source)
	if arch == "qwen35" && strings.Contains(source, "FP8") {
		q := ServedQuantization{Attention: "Q4_K", FFN: "Q4_K", Embedding: "BF16", LMHead: "BF16", Norms: "F32"}
		if moe {
			q.Experts = "NVFP4"
		}
		return q
	}
	if arch == "qwen35" && strings.Contains(source, "NVFP4") {
		q := ServedQuantization{Attention: "Q4_K", FFN: "Q4_K", Embedding: "BF16", LMHead: "Q8_0+BF16-rescore", Norms: "F32"}
		if moe {
			q.Experts = "NVFP4"
		}
		return q
	}
	if arch == "e4b" {
		return ServedQuantization{Attention: "BF16", FFN: "BF16", Embedding: "BF16+FP8-PLE", LMHead: "BF16", Norms: "BF16"}
	}
	if strings.Contains(source, "Q4_0") {
		return ServedQuantization{Attention: "Q4_0", FFN: "Q4_0", Embedding: "Q8_0", LMHead: "Q6_K", Norms: "F32"}
	}
	if source == "Q8_0" {
		return ServedQuantization{Attention: "Q8_0", FFN: "Q8_0", Embedding: "Q8_0", LMHead: "Q8_0", Norms: "F32"}
	}
	if strings.Contains(source, "NVFP4") {
		return ServedQuantization{Attention: "NVFP4", FFN: "NVFP4", Embedding: "BF16", LMHead: "BF16", Norms: "BF16"}
	}
	return ServedQuantization{Attention: source, FFN: source, Embedding: source, LMHead: source, Norms: "F32"}
}

func qualificationFor(s DescriptorSnapshot) Qualification {
	q := strings.ToUpper(s.SourceQuant)
	v := strings.ToLower(s.Variant)
	switch s.Architecture {
	case "gemma4":
		if strings.Contains(v, "12") && (strings.Contains(q, "Q4_0") || q == "Q8_0" || strings.Contains(q, "NVFP4")) {
			return QualificationValidated
		}
		return QualificationLoads
	case "qwen35":
		if strings.Contains(q, "FP8") && (strings.Contains(v, "9b") || strings.Contains(v, "27b") || strings.Contains(v, "35b")) {
			return QualificationValidated
		}
		if strings.Contains(q, "NVFP4") {
			return QualificationLoads
		}
		return QualificationExperimental
	case "gemma4-e4b":
		return QualificationLoads
	}
	return QualificationExperimental
}

func hybridMixers(layers, interval int) []MixerType {
	out := make([]MixerType, layers)
	if interval <= 0 {
		interval = 4
	}
	for i := range out {
		if (i+1)%interval == 0 {
			out[i] = MixerFullAttention
		} else {
			out[i] = MixerGatedDeltaNet
		}
	}
	return out
}

func approximateBillions(h, l, i int) int {
	if h == 3840 && l == 48 {
		return 12
	}
	if h == 5376 && l == 60 {
		return 31
	}
	if h == 0 || l == 0 || i == 0 {
		return 0
	}
	return int(math.Round(float64(l*(4*h*h+3*h*i)) / 1e9))
}

func cloneMap(in map[string]any) map[string]any {
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
func cloneTensors(in map[string]TensorInfo) map[string]TensorInfo {
	out := make(map[string]TensorInfo, len(in))
	for k, v := range in {
		v.Shape = append([]int64(nil), v.Shape...)
		out[k] = v
	}
	return out
}
func filterMap(in map[string]any, prefix string) map[string]any {
	out := map[string]any{}
	for k, v := range in {
		if strings.HasPrefix(k, prefix) {
			out[k] = v
		}
	}
	return out
}

// FromConfigJSON reads config.json and every safetensors header in its checkpoint directory, then
// runs a complete host-side preflight. path may name config.json or the checkpoint directory.
func FromConfigJSON(path string) (*ModelDescriptor, error) {
	configPath, dir, err := resolveConfigPath(path)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("model config: %w", err)
	}
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.UseNumber()
	var root map[string]any
	if err := dec.Decode(&root); err != nil {
		return nil, fmt.Errorf("model config JSON: %w", err)
	}
	tensors, err := readCheckpointTensorIndex(dir)
	if err != nil {
		return nil, err
	}
	// The generation label (notably Qwen3.6) is not represented by model_type: Qwen3.6
	// checkpoints still declare qwen3_5. Retain the source path only as builder input; the
	// resulting descriptor stores the normalized variant, never the path.
	root["_checkpoint_path"] = dir
	var errs []TensorMismatch
	s := DescriptorSnapshot{SourceFormat: FormatSafetensors, Capabilities: baseCapabilities()}
	cfg := textConfig(root)
	modelType, _ := stringValue(cfg, "model_type")
	if modelType == "" {
		modelType, _ = stringValue(root, "model_type")
	}
	isE4B := intFromNested(cfg, "hidden_size_per_layer_input") > 0 && hasNested(cfg, "num_kv_shared_layers")
	if isE4B {
		s = buildE4BConfig(s, root, cfg, tensors, &errs)
	} else if strings.Contains(strings.ToLower(modelType), "qwen3_5") || strings.Contains(strings.ToLower(modelType), "qwen35") {
		s = buildQwenConfig(s, root, cfg, tensors, &errs)
	} else if strings.Contains(strings.ToLower(modelType), "gemma") {
		s = buildGemmaConfig(s, root, cfg, tensors, &errs)
	} else {
		mismatch(&errs, "metadata.model_type", MismatchMetadata, "gemma4 or qwen3_5", modelType)
	}
	tokSeed, tcaps := readTokenizerSeed(dir)
	s.Tokenizer = tcaps
	return finish(s, tensors, root, tokSeed, errs)
}

func resolveConfigPath(path string) (string, string, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return "", "", fmt.Errorf("model config: %w", err)
	}
	if fi.IsDir() {
		return filepath.Join(path, "config.json"), path, nil
	}
	return path, filepath.Dir(path), nil
}

func textConfig(root map[string]any) map[string]any {
	if v, ok := root["text_config"].(map[string]any); ok {
		return v
	}
	return root
}
func hasNested(m map[string]any, k string) bool    { _, ok := m[k]; return ok }
func intFromNested(m map[string]any, k string) int { n, _ := intValue(m, k); return n }

func configString(root, cfg map[string]any, keys ...string) string {
	if s, ok := stringValue(cfg, keys...); ok {
		return s
	}
	s, _ := stringValue(root, keys...)
	return s
}

func buildGemmaConfig(s DescriptorSnapshot, root, cfg map[string]any, t map[string]TensorInfo, errs *[]TensorMismatch) DescriptorSnapshot {
	s.Family, s.Architecture = "gemma-4", "gemma4"
	s.Geometry = configGeometry(cfg, errs)
	s.Geometry.GlobalHeadDim, _ = intValue(cfg, "global_head_dim")
	if s.Geometry.GlobalHeadDim == 0 {
		s.Geometry.GlobalHeadDim = s.Geometry.HeadDim
	}
	s.Geometry.GlobalKVHeads, _ = intValue(cfg, "num_global_key_value_heads")
	if s.Geometry.GlobalKVHeads == 0 {
		s.Geometry.GlobalKVHeads = s.Geometry.KVHeads
	}
	period, _ := intValue(cfg, "sliding_window_pattern")
	if period <= 0 {
		period = 6
	}
	s.Attention.FullAttentionEvery = period
	layers := stringSlice(cfg["layer_types"])
	s.Mixers = make([]MixerType, s.Geometry.Layers)
	for i := range s.Mixers {
		if (i < len(layers) && layers[i] == "full_attention") || (len(layers) == 0 && (i+1)%period == 0) {
			s.Mixers[i] = MixerFullAttention
		} else {
			s.Mixers[i] = MixerSlidingAttention
		}
	}
	s.Attention.RoPETheta, _ = floatValue(cfg, "rope_theta")
	s.Attention.SlidingRoPETheta, _ = floatValue(cfg, "rope_local_base_freq")
	s.Attention.SlidingWindow, _ = intValue(cfg, "sliding_window")
	s.Attention.MaxPositionEmbeddings, _ = intValue(cfg, "max_position_embeddings")
	s.SourceQuant = detectConfigQuant(root, t)
	s.Variant = configString(root, cfg, "_name_or_path", "name")
	if s.Variant == "" {
		s.Variant = fmt.Sprintf("%dB", approximateBillions(s.Geometry.HiddenSize, s.Geometry.Layers, s.Geometry.Intermediate))
	}
	s.ServedQuant = servedFor("gemma4", s.SourceQuant, false)
	s.Qualification = qualificationFor(s)
	s.Capabilities.BatchMTP = true
	preflightGemmaSafetensors(s, root, t, errs)
	return s
}

func buildQwenConfig(s DescriptorSnapshot, root, cfg map[string]any, t map[string]TensorInfo, errs *[]TensorMismatch) DescriptorSnapshot {
	s.Family, s.Architecture = "qwen3.5", "qwen35"
	s.MoE.Experts, _ = intValue(cfg, "num_experts")
	s.MoE.TopK, _ = intValue(cfg, "num_experts_per_tok")
	s.MoE.ExpertIntermediate, _ = intValue(cfg, "moe_intermediate_size")
	s.MoE.SharedExperts, _ = intValue(cfg, "num_shared_experts")
	s.MoE.SharedExpertIntermediate, _ = intValue(cfg, "shared_expert_intermediate_size")
	// MoE configs intentionally omit dense intermediate_size; the runtime's `intermediate`
	// field carries shared-expert width for this architecture.
	geomCfg := cloneMap(cfg)
	if s.MoE.Experts > 0 {
		if _, ok := intValue(geomCfg, "intermediate_size"); !ok && s.MoE.SharedExpertIntermediate > 0 {
			geomCfg["intermediate_size"] = s.MoE.SharedExpertIntermediate
		}
	}
	s.Geometry = configGeometry(geomCfg, errs)
	s.Geometry.HeadDim = positiveConfigInt(cfg, errs, "head_dim", 256)
	interval, _ := intValue(cfg, "full_attention_interval")
	if interval <= 0 {
		interval = 4
	}
	s.Attention.FullAttentionEvery = interval
	if rope, ok := cfg["rope_parameters"].(map[string]any); ok {
		s.Attention.RoPETheta, _ = floatValue(rope, "rope_theta")
		s.Attention.RotaryDim = int(float64(s.Geometry.HeadDim) * numberOr(rope, "partial_rotary_factor", 0.25))
	} else {
		s.Attention.RoPETheta, _ = floatValue(cfg, "rope_theta")
		s.Attention.RotaryDim = int(float64(s.Geometry.HeadDim) * numberOr(cfg, "partial_rotary_factor", 0.25))
	}
	s.Attention.MaxPositionEmbeddings, _ = intValue(cfg, "max_position_embeddings")
	s.GDN.StateSize = positiveConfigInt(cfg, errs, "linear_key_head_dim", 128)
	s.GDN.ConvKernel = positiveConfigInt(cfg, errs, "linear_conv_kernel_dim", 4)
	s.GDN.InnerSize = positiveConfigInt(cfg, errs, "linear_value_head_dim", 128) * positiveConfigInt(cfg, errs, "linear_num_value_heads", 32)
	s.GDN.GroupCount = positiveConfigInt(cfg, errs, "linear_num_key_heads", 16)
	s.GDN.TimeStepRank = positiveConfigInt(cfg, errs, "linear_num_value_heads", 32)
	s.GDN.KeyDim = s.GDN.GroupCount * s.GDN.StateSize
	s.GDN.ConvDim = 2*s.GDN.KeyDim + s.GDN.InnerSize
	s.Mixers = hybridMixers(s.Geometry.Layers, interval)
	if s.MoE.Experts > 0 && s.MoE.SharedExperts == 0 {
		s.MoE.SharedExperts = 1
	}
	if s.MoE.Experts > 0 && s.MoE.SharedExpertIntermediate == 0 {
		s.MoE.SharedExpertIntermediate = s.Geometry.Intermediate
	}
	deriveQwenGeometryFromTensors(&s, t, errs)
	s.SourceQuant = detectConfigQuant(root, t)
	s.Variant = inferQwenVariant(root, s)
	if strings.Contains(strings.ToLower(s.Variant), "qwen3.6") {
		s.Family = "qwen3.6"
	}
	s.ServedQuant = servedFor("qwen35", s.SourceQuant, s.MoE.Experts > 0)
	s.Qualification = qualificationFor(s)
	s.Capabilities.BatchMTP = false
	preflightQwenSafetensors(s, root, t, errs)
	return s
}

func buildE4BConfig(s DescriptorSnapshot, root, cfg map[string]any, t map[string]TensorInfo, errs *[]TensorMismatch) DescriptorSnapshot {
	s.Family, s.Architecture = "gemma-4", "gemma4-e4b"
	s.Geometry = configGeometry(cfg, errs)
	s.Geometry.GlobalHeadDim = positiveConfigInt(cfg, errs, "global_head_dim", 512)
	s.Attention.SlidingWindow, _ = intValue(cfg, "sliding_window")
	s.Attention.MaxPositionEmbeddings, _ = intValue(cfg, "max_position_embeddings")
	layers := stringSlice(cfg["layer_types"])
	s.Mixers = make([]MixerType, s.Geometry.Layers)
	for i := range s.Mixers {
		if i < len(layers) && layers[i] == "full_attention" {
			s.Mixers[i] = MixerFullAttention
		} else if len(layers) == 0 && (i+1)%6 == 0 {
			s.Mixers[i] = MixerFullAttention
		} else {
			s.Mixers[i] = MixerSlidingAttention
		}
	}
	if rope, ok := cfg["rope_parameters"].(map[string]any); ok {
		if x, ok := rope["sliding_attention"].(map[string]any); ok {
			s.Attention.SlidingRoPETheta, _ = floatValue(x, "rope_theta")
		}
		if x, ok := rope["full_attention"].(map[string]any); ok {
			s.Attention.RoPETheta, _ = floatValue(x, "rope_theta")
			s.Attention.PartialRotaryFactor, _ = floatValue(x, "partial_rotary_factor")
		}
	}
	s.SourceQuant = detectConfigQuant(root, t)
	s.Variant = configString(root, cfg, "_name_or_path", "name")
	if s.Variant == "" {
		s.Variant = "E4B"
	}
	s.ServedQuant = servedFor("e4b", s.SourceQuant, false)
	s.Qualification = qualificationFor(s)
	s.Capabilities.E4BServing = true
	s.Capabilities.BatchMTP = false
	s.Capabilities.StructuredOutput = false
	preflightE4BSafetensors(s, cfg, t, errs)
	return s
}

func deriveQwenGeometryFromTensors(s *DescriptorSnapshot, tensors map[string]TensorInfo, errs *[]TensorMismatch) {
	r := checkpointRoot(tensors)
	embed, ok := tensors[r+"embed_tokens.weight"]
	if ok && len(embed.Shape) == 2 {
		s.Geometry.VocabSize, s.Geometry.HiddenSize = int(embed.Shape[0]), int(embed.Shape[1])
	}
	full := s.Attention.FullAttentionEvery - 1
	q := logicalWeightTensor(tensors, fmt.Sprintf("%slayers.%d.self_attn.q_proj.weight", r, full))
	k := logicalWeightTensor(tensors, fmt.Sprintf("%slayers.%d.self_attn.k_proj.weight", r, full))
	if len(q.Shape) == 2 && s.Geometry.HeadDim > 0 {
		s.Geometry.AttentionHead = int(q.Shape[0]) / (2 * s.Geometry.HeadDim)
	}
	if len(k.Shape) == 2 && s.Geometry.HeadDim > 0 {
		s.Geometry.KVHeads = int(k.Shape[0]) / s.Geometry.HeadDim
		s.Geometry.GlobalKVHeads = s.Geometry.KVHeads
	}
	z := logicalWeightTensor(tensors, r+"layers.0.linear_attn.in_proj_z.weight")
	if len(z.Shape) == 2 {
		s.GDN.InnerSize = int(z.Shape[0])
		if s.GDN.StateSize > 0 {
			s.GDN.TimeStepRank = s.GDN.InnerSize / s.GDN.StateSize
		}
		s.GDN.ConvDim = 2*s.GDN.KeyDim + s.GDN.InnerSize
	}
	mlpName := r + "layers.0.mlp.gate_proj.weight"
	if s.MoE.Experts > 0 {
		mlpName = r + "layers.0.mlp.shared_expert.gate_proj.weight"
	}
	mlp := logicalWeightTensor(tensors, mlpName)
	if len(mlp.Shape) == 2 {
		s.Geometry.Intermediate = int(mlp.Shape[0])
		if s.MoE.Experts > 0 {
			s.MoE.SharedExpertIntermediate = s.Geometry.Intermediate
		}
	}
	if s.Geometry.HiddenSize <= 0 || s.Geometry.AttentionHead <= 0 || s.Geometry.KVHeads <= 0 {
		mismatch(errs, "tensor-derived-geometry", MismatchMetadata, "embed/full-q/full-k tensor geometry", "incomplete")
	}
}

func logicalWeightTensor(t map[string]TensorInfo, name string) TensorInfo {
	if x, ok := t[name]; ok {
		return x
	}
	packed := strings.TrimSuffix(name, "weight") + "weight_packed"
	return t[packed]
}

func inferQwenVariant(root map[string]any, s DescriptorSnapshot) string {
	path, _ := stringValue(root, "_checkpoint_path", "_name_or_path", "name")
	generation := "Qwen3.5"
	if strings.Contains(strings.ToLower(path), "qwen3.6") {
		generation = "Qwen3.6"
	}
	if s.MoE.Experts > 0 && s.Geometry.HiddenSize == 2048 && s.Geometry.Layers == 40 {
		return generation + "-35B-A3B-MoE"
	}
	if s.Geometry.HiddenSize == 5120 && s.Geometry.Layers == 64 {
		return generation + "-27B-dense"
	}
	if s.Geometry.HiddenSize == 4096 && s.Geometry.Layers == 32 {
		return generation + "-9B-dense"
	}
	shape := fmt.Sprintf("H%d-L%d", s.Geometry.HiddenSize, s.Geometry.Layers)
	if s.MoE.Experts > 0 {
		shape += "-MoE"
	} else {
		shape += "-dense"
	}
	return generation + "-" + shape
}

func configGeometry(cfg map[string]any, errs *[]TensorMismatch) Geometry {
	g := Geometry{}
	g.Layers = positiveConfigInt(cfg, errs, "num_hidden_layers", 0)
	g.HiddenSize = positiveConfigInt(cfg, errs, "hidden_size", 0)
	g.Intermediate = positiveConfigInt(cfg, errs, "intermediate_size", 0)
	g.AttentionHead = positiveConfigInt(cfg, errs, "num_attention_heads", 0)
	g.KVHeads = positiveConfigInt(cfg, errs, "num_key_value_heads", 0)
	g.GlobalKVHeads = g.KVHeads
	g.HeadDim = positiveConfigInt(cfg, errs, "head_dim", 0)
	if g.HeadDim == 0 && g.AttentionHead > 0 {
		g.HeadDim = g.HiddenSize / g.AttentionHead
	}
	g.VocabSize = positiveConfigInt(cfg, errs, "vocab_size", 0)
	return g
}

func positiveConfigInt(cfg map[string]any, errs *[]TensorMismatch, key string, fallback int) int {
	if n, ok := intValue(cfg, key); ok && n > 0 {
		return n
	}
	if fallback > 0 {
		return fallback
	}
	mismatch(errs, "metadata."+key, MismatchMetadata, "positive integer", "missing or invalid")
	return 0
}
func numberOr(m map[string]any, k string, d float64) float64 {
	if n, ok := floatValue(m, k); ok {
		return n
	}
	return d
}
func stringSlice(v any) []string {
	switch x := v.(type) {
	case []string:
		return append([]string(nil), x...)
	case []any:
		o := []string{}
		for _, v := range x {
			if s, ok := v.(string); ok {
				o = append(o, s)
			}
		}
		return o
	}
	return nil
}

func detectConfigQuant(root map[string]any, t map[string]TensorInfo) string {
	b, _ := json.Marshal(root)
	s := strings.ToLower(string(b))
	if strings.Contains(s, "nvfp4") || strings.Contains(s, "nvfp4-pack-quantized") {
		return "NVFP4"
	}
	if strings.Contains(s, "\"quant_method\":\"fp8\"") || strings.Contains(s, "\"quant_method\": \"fp8\"") {
		return "FP8_BLOCK_128"
	}
	counts := map[string]int{}
	for _, x := range t {
		counts[canonicalEncoding(x.Encoding)]++
	}
	if counts["U8"] > 0 && counts["F8_E4M3"] > 0 {
		return "NVFP4"
	}
	if counts["F8_E4M3"] > 0 {
		return "FP8_BLOCK_128"
	}
	if counts["BF16"] > 0 {
		return "BF16"
	}
	return "unknown"
}

func readTokenizerSeed(dir string) ([]byte, TokenizerCapabilities) {
	files := []string{"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
	var seed []byte
	caps := TokenizerCapabilities{}
	for _, name := range files {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			continue
		}
		seed = append(seed, []byte(name)...)
		seed = append(seed, b...)
		caps.Available = true
		if name == "tokenizer.json" {
			caps.Kind = "hf-json"
		}
		if name == "tokenizer_config.json" {
			var m map[string]any
			if json.Unmarshal(b, &m) == nil {
				if _, ok := m["chat_template"]; ok {
					caps.ChatTemplate = true
					caps.Tools = true
					caps.ThinkingControl = true
				}
			}
		}
	}
	if caps.Kind == "" {
		caps.Kind = "unknown"
	}
	return seed, caps
}

func canonicalJSON(v any) []byte { b, _ := json.Marshal(v); return b }

var _ Getter = (*Store)(nil)
var _ json.Marshaler = (*ModelDescriptor)(nil)
var _ = errors.As
var _ = canonicalJSON
