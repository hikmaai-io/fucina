package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/hikmaai-io/fucina/internal/model"
)

type tierAFixtures struct {
	TokenizeResponseKeys    []string `json:"tokenize_response_keys"`
	DetokenizeResponseKeys  []string `json:"detokenize_response_keys"`
	VersionResponseKeys     []string `json:"version_response_keys"`
	ModelDetailRequiredKeys []string `json:"model_detail_required_keys"`
	ErrorRequiredKeys       []string `json:"error_required_keys"`
	UsageOnlyChoicesLength  int      `json:"usage_only_choices_length"`
	UsageRequiredKeys       []string `json:"usage_required_keys"`
}

func loadTierAFixtures(t *testing.T) tierAFixtures {
	t.Helper()
	var fixture tierAFixtures
	data, err := osReadFile("testdata/vllm_tier_a_fixtures.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture
}

// osReadFile is a variable solely so fixture loading stays explicit in this
// wire-contract suite and can report the exact provenance file beside it.
var osReadFile = func(name string) ([]byte, error) { return os.ReadFile(name) }

func newContractServer(t *testing.T) *Server {
	t.Helper()
	tk, idx := newServerTokenizer(t)
	eng := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
	return New(eng, tk)
}

func contractMux(s *Server) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/tokenize", s.handleTokenize)
	mux.HandleFunc("/detokenize", s.handleDetokenize)
	mux.HandleFunc("/version", s.handleVersion)
	mux.HandleFunc("/v1/models/", s.handleModelDetail)
	mux.HandleFunc("/v1/embeddings", s.handleEmbeddings)
	return mux
}

func requestJSON(t *testing.T, h http.Handler, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func responseKeys(t *testing.T, data []byte) []string {
	t.Helper()
	var value map[string]json.RawMessage
	if err := json.Unmarshal(data, &value); err != nil {
		t.Fatalf("invalid JSON %q: %v", data, err)
	}
	keys := make([]string, 0, len(value))
	for key := range value {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func assertKeys(t *testing.T, got, want []string) {
	t.Helper()
	sort.Strings(want)
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("keys=%v want %v", got, want)
	}
}

func assertRequiredKeys(t *testing.T, data []byte, want []string) {
	t.Helper()
	var value map[string]json.RawMessage
	if err := json.Unmarshal(data, &value); err != nil {
		t.Fatal(err)
	}
	for _, key := range want {
		if _, ok := value[key]; !ok {
			t.Errorf("missing fixture-required key %q in %s", key, data)
		}
	}
}

func TestOpenAIContractMaxCompletionTokensPrecedence(t *testing.T) {
	maxCompletion := 3
	got, explicit, err := resolveCompletionLimit(99, &maxCompletion)
	if err != nil || !explicit || got != 3 {
		t.Fatalf("resolve=(%d,%v,%v), want (3,true,nil)", got, explicit, err)
	}
	got, explicit, err = resolveCompletionLimit(7, nil)
	if err != nil || !explicit || got != 7 {
		t.Fatalf("alias resolve=(%d,%v,%v), want (7,true,nil)", got, explicit, err)
	}
	if _, _, err := resolveCompletionLimit(7, intPtr(0)); err == nil || !strings.Contains(err.Error(), "max_completion_tokens") {
		t.Fatalf("zero max_completion_tokens error=%v", err)
	}
}

func intPtr(v int) *int { return &v }

func TestOpenAIContractStreamOptionsValidation(t *testing.T) {
	if err := validateStreamOptions(false, &StreamOptions{IncludeUsage: true}); err == nil {
		t.Fatal("stream_options without stream must be rejected like vLLM")
	}
	if err := validateStreamOptions(true, &StreamOptions{IncludeUsage: true}); err != nil {
		t.Fatal(err)
	}
}

func TestOpenAIContractIncludeUsageSSEWire(t *testing.T) {
	fixture := loadTierAFixtures(t)
	for _, tc := range []struct {
		name    string
		include bool
	}{
		{"default_no_usage", false},
		{"include_usage", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			sse, ok := newSSEWriter(rec, false, "fixture-model")
			if !ok {
				t.Fatal("recorder must support flushing")
			}
			sse.setIncludeUsage(tc.include)
			sse.begin()
			usage := &Usage{
				PromptTokens: 10, CompletionTokens: 3, TotalTokens: 13,
				PromptTokensDetails: &PromptTokensDetails{CachedTokens: 4},
			}
			sse.finish("stop", usage)

			var events []map[string]json.RawMessage
			for _, line := range strings.Split(rec.Body.String(), "\n") {
				if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
					continue
				}
				var event map[string]json.RawMessage
				if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
					t.Fatal(err)
				}
				events = append(events, event)
			}
			last := events[len(events)-1]
			_, hasUsage := last["usage"]
			if hasUsage != tc.include {
				t.Fatalf("last event usage=%v body=%s", hasUsage, rec.Body.String())
			}
			if !tc.include {
				for _, event := range events {
					if _, ok := event["usage"]; ok {
						t.Fatalf("default stream unexpectedly contains usage: %s", rec.Body.String())
					}
				}
				return
			}
			for i, event := range events[:len(events)-1] {
				raw, ok := event["usage"]
				if !ok || string(raw) != "null" {
					t.Fatalf("event %d usage=%s, want explicit null: %s", i, raw, rec.Body.String())
				}
			}
			var choices []json.RawMessage
			if err := json.Unmarshal(last["choices"], &choices); err != nil {
				t.Fatal(err)
			}
			if len(choices) != fixture.UsageOnlyChoicesLength {
				t.Fatalf("usage-only choices=%d want %d", len(choices), fixture.UsageOnlyChoicesLength)
			}
			assertRequiredKeys(t, last["usage"], fixture.UsageRequiredKeys)
			var got Usage
			if err := json.Unmarshal(last["usage"], &got); err != nil {
				t.Fatal(err)
			}
			if got.PromptTokensDetails == nil || got.PromptTokensDetails.CachedTokens != 4 {
				t.Fatalf("cached usage=%+v", got)
			}
			if got.PromptTokensDetails.CachedTokens < 0 || got.PromptTokensDetails.CachedTokens > got.PromptTokens {
				t.Fatalf("cached invariant violated: %+v", got)
			}
			if got.PromptTokens-got.PromptTokensDetails.CachedTokens != 6 {
				t.Fatalf("new prefill invariant violated: %+v", got)
			}
			if got.TotalTokens != got.PromptTokens+got.CompletionTokens {
				t.Fatalf("total invariant violated: %+v", got)
			}
		})
	}
}

func TestOpenAIContractTokenizerEndpointsAndVersionFixtures(t *testing.T) {
	fixture := loadTierAFixtures(t)
	srv := newContractServer(t)
	h := contractMux(srv)

	rec := requestJSON(t, h, http.MethodPost, "/tokenize", `{"prompt":"hello world","add_special_tokens":false}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("tokenize status=%d body=%s", rec.Code, rec.Body.String())
	}
	assertKeys(t, responseKeys(t, rec.Body.Bytes()), fixture.TokenizeResponseKeys)
	var tokenized TokenizeResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &tokenized); err != nil {
		t.Fatal(err)
	}
	if tokenized.Count == 0 || tokenized.Count != len(tokenized.Tokens) || tokenized.MaxModelLen != 8192 {
		t.Fatalf("tokenize response=%+v", tokenized)
	}

	encoded, _ := json.Marshal(DetokenizeRequest{Tokens: int32sToInt64s(tokenized.Tokens)})
	rec = requestJSON(t, h, http.MethodPost, "/detokenize", string(encoded))
	if rec.Code != http.StatusOK {
		t.Fatalf("detokenize status=%d body=%s", rec.Code, rec.Body.String())
	}
	assertKeys(t, responseKeys(t, rec.Body.Bytes()), fixture.DetokenizeResponseKeys)
	var detokenized DetokenizeResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &detokenized); err != nil {
		t.Fatal(err)
	}
	if detokenized.Prompt != "hello world" {
		t.Fatalf("prompt=%q want hello world", detokenized.Prompt)
	}

	rec = requestJSON(t, h, http.MethodGet, "/version", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("version status=%d body=%s", rec.Code, rec.Body.String())
	}
	assertKeys(t, responseKeys(t, rec.Body.Bytes()), fixture.VersionResponseKeys)
	if !strings.Contains(rec.Body.String(), `"version":"`) {
		t.Fatalf("empty version: %s", rec.Body.String())
	}
}

func int32sToInt64s(in []int32) []int64 {
	out := make([]int64, len(in))
	for i, value := range in {
		out[i] = int64(value)
	}
	return out
}

func TestOpenAIContractModelDetailFromManifestStore(t *testing.T) {
	fixture := loadTierAFixtures(t)
	descriptor, err := model.FromGGUF(contractGemmaGGUF())
	if err != nil {
		t.Fatal(err)
	}
	srv := newContractServer(t)
	srv.SetModelStore(model.NewStore(descriptor))
	rec := requestJSON(t, contractMux(srv), http.MethodGet, "/v1/models/"+url.PathEscape(descriptor.ID()), "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	assertRequiredKeys(t, rec.Body.Bytes(), fixture.ModelDetailRequiredKeys)
	var got ModelDetailResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.ID != descriptor.ID() || got.Object != "model" || got.Fingerprints.Model == "" {
		t.Fatalf("model detail=%+v", got)
	}

	rec = requestJSON(t, contractMux(srv), http.MethodGet, "/v1/models/missing", "")
	assertOpenAIErrorFixture(t, rec, http.StatusNotFound, "model_not_found")
}

func TestOpenAIContractStrictUnknownAndMultimodalErrors(t *testing.T) {
	srv := newContractServer(t)
	h := contractMux(srv)
	srv.SetOpenAIStrict(true)
	rec := requestJSON(t, h, http.MethodPost, "/tokenize", `{"prompt":"hello","mystery":1}`)
	assertOpenAIErrorFixture(t, rec, http.StatusBadRequest, "invalid_request")
	if !strings.Contains(rec.Body.String(), "mystery") {
		t.Fatalf("strict error did not name field: %s", rec.Body.String())
	}
	var nested struct {
		StreamOptions *StreamOptions `json:"stream_options"`
	}
	if err := decodeOpenAIJSON([]byte(`{"stream_options":{"include_usage":true,"mystery":1}}`), &nested, true); err == nil || !strings.Contains(err.Error(), "stream_options.mystery") {
		t.Fatalf("nested strict error=%v", err)
	}

	srv.SetOpenAIStrict(false)
	var logs bytes.Buffer
	old := log.Writer()
	log.SetOutput(&logs)
	t.Cleanup(func() { log.SetOutput(old) })
	rec = requestJSON(t, h, http.MethodPost, "/tokenize", `{"prompt":"hello","mystery":1}`)
	if rec.Code != http.StatusOK || !strings.Contains(logs.String(), "mystery") {
		t.Fatalf("compat unknown status=%d logs=%q body=%s", rec.Code, logs.String(), rec.Body.String())
	}

	body := []byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"x"}}]}]}`)
	if err := validateSupportedContentParts(body); err == nil || !strings.Contains(err.Error(), "image_url") {
		t.Fatalf("multimodal validation error=%v", err)
	}
}

func TestOpenAIContractEmbeddingErrorFixturePreserved(t *testing.T) {
	srv := newContractServer(t)
	rec := requestJSON(t, contractMux(srv), http.MethodPost, "/v1/embeddings", `{"model":"local","input":"hello"}`)
	assertOpenAIErrorFixture(t, rec, http.StatusNotImplemented, "model_not_embedding")
}

func assertOpenAIErrorFixture(t *testing.T, rec *httptest.ResponseRecorder, status int, code string) {
	t.Helper()
	if rec.Code != status {
		t.Fatalf("status=%d want %d body=%s", rec.Code, status, rec.Body.String())
	}
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(rec.Body.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	raw, ok := envelope["error"]
	if !ok || len(envelope) != 1 {
		t.Fatalf("non-standard envelope: %s", rec.Body.String())
	}
	assertRequiredKeys(t, raw, loadTierAFixtures(t).ErrorRequiredKeys)
	var apiErr OpenAIError
	if err := json.Unmarshal(raw, &apiErr); err != nil {
		t.Fatal(err)
	}
	if fmt.Sprint(apiErr.Code) != code {
		t.Fatalf("code=%v want %s body=%s", apiErr.Code, code, rec.Body.String())
	}
}

func contractGemmaGGUF() model.GGUFMetadata {
	const layers, hidden, intermediate, heads, vocab = 48, 3840, 15360, 16, 262144
	pattern := make([]bool, layers)
	kvHeads := make([]int, layers)
	for layer := 0; layer < layers; layer++ {
		pattern[layer] = (layer+1)%6 != 0
		if pattern[layer] {
			kvHeads[layer] = 8
		} else {
			kvHeads[layer] = 1
		}
	}
	metadata := model.GGUFMetadata{Values: map[string]interface{}{
		"general.architecture": "gemma4", "general.name": "Gemma-4-12B-it",
		"gemma4.block_count": layers, "gemma4.embedding_length": hidden,
		"gemma4.feed_forward_length": intermediate, "gemma4.attention.head_count": heads,
		"gemma4.attention.head_count_kv": kvHeads, "gemma4.attention.sliding_window_pattern": pattern,
		"gemma4.vocab_size": vocab, "gemma4.attention.key_length_swa": 256,
		"gemma4.attention.key_length": 512, "gemma4.rope.freq_base": 1e6,
		"gemma4.rope.freq_base_swa": 1e4, "tokenizer.ggml.model": "llama",
		"tokenizer.ggml.token_count": vocab, "tokenizer.chat_template": "gemma",
	}, Tensors: map[string]model.TensorInfo{}}
	add := func(name, encoding string, dims ...int) {
		shape := make([]int64, len(dims))
		for i, dim := range dims {
			shape[i] = int64(dim)
		}
		metadata.Tensors[name] = model.TensorInfo{Shape: shape, Encoding: encoding}
	}
	add("token_embd.weight", "Q8_0", hidden, vocab)
	add("output_norm.weight", "F32", hidden)
	for layer := 0; layer < layers; layer++ {
		headDim, kv := 256, 8
		if !pattern[layer] {
			headDim, kv = 512, 1
		}
		prefix := fmt.Sprintf("blk.%d.", layer)
		add(prefix+"attn_q.weight", "Q8_0", hidden, heads*headDim)
		add(prefix+"attn_k.weight", "Q8_0", hidden, kv*headDim)
		if pattern[layer] {
			add(prefix+"attn_v.weight", "Q8_0", hidden, kv*headDim)
		}
		add(prefix+"attn_output.weight", "Q8_0", heads*headDim, hidden)
		for _, name := range []string{"attn_norm.weight", "post_attention_norm.weight", "ffn_norm.weight", "post_ffw_norm.weight"} {
			add(prefix+name, "F32", hidden)
		}
		add(prefix+"attn_q_norm.weight", "F32", headDim)
		add(prefix+"attn_k_norm.weight", "F32", headDim)
		add(prefix+"ffn_gate.weight", "Q8_0", hidden, intermediate)
		add(prefix+"ffn_up.weight", "Q8_0", hidden, intermediate)
		add(prefix+"ffn_down.weight", "Q8_0", intermediate, hidden)
		add(prefix+"layer_output_scale.weight", "F32", 1)
	}
	return metadata
}
