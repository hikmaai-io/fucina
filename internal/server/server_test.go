package server

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"io"
	"log"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mauromedda/gem4d/internal/tokenizer"
)

// ─── minimal in-memory GGUF builder (duplicated from the tokenizer test, kept
//     minimal here to avoid a cross-package test helper) ──────────────────────

const (
	ggufTypeUint32  = 4
	ggufTypeFloat32 = 6
	ggufTypeString  = 8
	ggufTypeArray   = 9
)

type ggufWriter struct{ buf []byte }

func (w *ggufWriter) u32(v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	w.buf = append(w.buf, b[:]...)
}
func (w *ggufWriter) u64(v uint64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	w.buf = append(w.buf, b[:]...)
}
func (w *ggufWriter) f32(v float32) { w.u32(math.Float32bits(v)) }
func (w *ggufWriter) str(s string) {
	w.u64(uint64(len(s)))
	w.buf = append(w.buf, s...)
}
func (w *ggufWriter) kvStringArray(key string, vals []string) {
	w.str(key)
	w.u32(ggufTypeArray)
	w.u32(ggufTypeString)
	w.u64(uint64(len(vals)))
	for _, s := range vals {
		w.str(s)
	}
}
func (w *ggufWriter) kvF32Array(key string, vals []float32) {
	w.str(key)
	w.u32(ggufTypeArray)
	w.u32(ggufTypeFloat32)
	w.u64(uint64(len(vals)))
	for _, v := range vals {
		w.f32(v)
	}
}
func (w *ggufWriter) kvU32(key string, v uint32) {
	w.str(key)
	w.u32(ggufTypeUint32)
	w.u32(v)
}

func byteTok(b int) string {
	const hex = "0123456789ABCDEF"
	return "<0x" + string([]byte{hex[(b>>4)&0xF], hex[b&0xF]}) + ">"
}

// newServerTokenizer builds a tiny gemma-4-flavored tokenizer with the control
// tokens the server relies on (<|turn>, <turn|>, <|channel>, <channel|>, tool
// markers, "▁hello", …).
func newServerTokenizer(t *testing.T) (*tokenizer.Tokenizer, map[string]int32) {
	t.Helper()
	tokens := []string{"<pad>", "<eos>", "<bos>", "<unk>"}
	for b := 0; b < 256; b++ {
		tokens = append(tokens, byteTok(b))
	}
	pieces := []string{
		"▁hello", "hello", "▁world", "world", "▁",
		"h", "e", "l", "o",
		"<|turn>", "<turn|>", "<|channel>", "<channel|>",
		"<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
		"<|tool_response>", "<tool_response|>", `<|"|>`, "<|think|>",
	}
	tokens = append(tokens, pieces...)

	scores := make([]float32, len(tokens))
	for i := range scores {
		scores[i] = float32(-i)
	}
	idx := make(map[string]int32, len(tokens))
	for i, s := range tokens {
		if _, ok := idx[s]; !ok {
			idx[s] = int32(i)
		}
	}

	w := &ggufWriter{}
	w.u32(0x46554747) // "GGUF"
	w.u32(3)          // version
	w.u64(0)          // tensor count
	w.u64(5)          // kv count
	w.kvStringArray("tokenizer.ggml.tokens", tokens)
	w.kvF32Array("tokenizer.ggml.scores", scores)
	w.kvU32("tokenizer.ggml.bos_token_id", uint32(idx["<bos>"]))
	w.kvU32("tokenizer.ggml.eos_token_id", uint32(idx["<eos>"]))
	w.kvU32("tokenizer.ggml.padding_token_id", uint32(idx["<pad>"]))

	tk, err := tokenizer.New(w.buf, int64(len(w.buf)))
	if err != nil {
		t.Fatalf("tokenizer.New failed: %v", err)
	}
	return tk, idx
}

// ─── deterministic fake engine ─────────────────────────────────────────────

// fakeServerEngine implements serverEngine without a GPU. Generation is driven
// by a fixed `script` of token ids: Prefill positions the cursor at the start,
// Decode/DecodeNoCopy advance it, and Decode/SampleDevice return logits/ids that
// peak at the scripted token. When the cursor runs past the script the engine
// emits EOS so generation terminates.
type fakeServerEngine struct {
	ctxSize uint32
	vocab   int
	eos     int32

	tokens []int32 // KV-cache contents (for NTokens/Reset/Rewind)
	script []int32 // tokens generation should produce, in order
	cursor int     // index into script for the NEXT token

	// observability
	lastPrefillLen   int
	decodeCalls      int
	decodeNoCopy     int
	sampleDeviceHits int
	specCalls        int
}

func (f *fakeServerEngine) scriptAt(i int) int32 {
	if i < 0 || i >= len(f.script) {
		return f.eos
	}
	return f.script[i]
}

func (f *fakeServerEngine) logitsFor(id int32) []float32 {
	l := make([]float32, f.vocab)
	if id >= 0 && int(id) < f.vocab {
		l[id] = 100.0
	}
	return l
}

func (f *fakeServerEngine) Prefill(tokens []int32) ([]float32, error) {
	f.lastPrefillLen = len(tokens)
	f.tokens = append(f.tokens, tokens...)
	f.cursor = 0
	return f.logitsFor(f.scriptAt(0)), nil
}

func (f *fakeServerEngine) Decode(token int32) ([]float32, error) {
	f.decodeCalls++
	f.tokens = append(f.tokens, token)
	f.cursor++
	return f.logitsFor(f.scriptAt(f.cursor)), nil
}

func (f *fakeServerEngine) DecodeNoCopy(token int32) error {
	f.decodeNoCopy++
	f.tokens = append(f.tokens, token)
	f.cursor++
	return nil
}

func (f *fakeServerEngine) SampleDevice(temp float32, topK int, topP, minP, rnd float32) (int32, error) {
	f.sampleDeviceHits++
	return f.scriptAt(f.cursor), nil
}

func (f *fakeServerEngine) GenerateSpecContinue(history []int32, firstLogits []float32, maxNew int,
	stops []int32, draftK int, temp float32, topK int, topP, minP float32, seed uint64) ([]int32, int, error) {
	return f.GenerateSpecStream(history, firstLogits, maxNew, stops, draftK,
		temp, topK, topP, minP, seed, nil)
}

// GenerateSpecStream mirrors the real engine's contract: every generated token
// is reported to emit (when non-nil) in order; emit returning true stops
// generation after that token, and all generated tokens stay in the returned
// slice (the server reconciles the prefix cache from NTokens).
func (f *fakeServerEngine) GenerateSpecStream(history []int32, firstLogits []float32, maxNew int,
	stops []int32, draftK int, temp float32, topK int, topP, minP float32, seed uint64,
	emit func(int32) bool) ([]int32, int, error) {
	f.specCalls++
	isStop := func(t int32) bool {
		for _, s := range stops {
			if s == t {
				return true
			}
		}
		return false
	}
	// Consume the script from the shared cursor (NOT from the start) so a
	// resumed round — the server's thinking-budget force-close re-enters
	// generation after a Decode — continues where the previous round stopped,
	// like the real engine's KV-continuation does.
	out := make([]int32, 0, len(f.script))
	for len(out) < maxNew && f.cursor < len(f.script) {
		tk := f.script[f.cursor]
		f.cursor++
		out = append(out, tk)
		f.tokens = append(f.tokens, tk)
		if emit != nil && emit(tk) {
			break
		}
		if isStop(tk) {
			break
		}
	}
	return out, len(out), nil
}

func (f *fakeServerEngine) NTokens() int        { return len(f.tokens) }
func (f *fakeServerEngine) Reset()              { f.tokens = f.tokens[:0] }
func (f *fakeServerEngine) ContextSize() uint32 { return f.ctxSize }

func (f *fakeServerEngine) Rewind(nKeep int) bool {
	if nKeep < 0 || nKeep > len(f.tokens) {
		return false
	}
	f.tokens = f.tokens[:nKeep]
	return true
}

// compile-time assertion that the fake satisfies the interface the server uses.
var _ serverEngine = (*fakeServerEngine)(nil)

// ─── test harness ──────────────────────────────────────────────────────────

func newTestServer(t *testing.T, ctxSize uint32, script []int32) (*Server, *fakeServerEngine) {
	t.Helper()
	tk, _ := newServerTokenizer(t)
	f := &fakeServerEngine{
		ctxSize: ctxSize,
		vocab:   tk.NumTokens(),
		eos:     tk.EOS,
		script:  script,
	}
	srv := New(f, tk)
	srv.SetLogLevel("warn") // keep test output quiet
	return srv, f
}

func mux(s *Server) *http.ServeMux {
	m := http.NewServeMux()
	s.RegisterRoutes(m)
	return m
}

// helloWorldScript returns ["▁hello","▁world"] which decode to " hello world".
func helloWorldScript(idx map[string]int32) []int32 {
	return []int32{idx["▁hello"], idx["▁world"]}
}

// ─── tests ─────────────────────────────────────────────────────────────────

func TestHandleHealth(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, httptest.NewRequest("GET", "/health", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200", rec.Code)
	}
	var body map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	if body["status"] != "ok" {
		t.Errorf("status=%v want ok", body["status"])
	}
}

func TestHandleModels(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	srv.SetModelName("gemma-4-12b-q8")
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, httptest.NewRequest("GET", "/v1/models", nil))

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200", rec.Code)
	}
	var resp ModelsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	if len(resp.Data) != 1 || resp.Data[0].ID != "gemma-4-12b-q8" {
		t.Errorf("models data=%+v want id gemma-4-12b-q8", resp.Data)
	}
}

func chatBody(t *testing.T, m map[string]interface{}) io.Reader {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	return strings.NewReader(string(b))
}

func TestChatCompletionsNonStreamSpecPath(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	// No stop string + default repeat penalty → the speculative fast path.
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":    []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens":  16,
		"temperature": 0,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 (body=%s)", rec.Code, rec.Body.String())
	}
	if f.specCalls != 1 {
		t.Errorf("GenerateSpecContinue called %d times, want 1", f.specCalls)
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("choices=%d want 1", len(resp.Choices))
	}
	if got := resp.Choices[0].Message.Content; got != "hello world" {
		t.Errorf("content=%q want %q", got, "hello world")
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish=%q want stop", resp.Choices[0].FinishReason)
	}
	if resp.Usage.CompletionTokens != 2 {
		t.Errorf("completion_tokens=%d want 2", resp.Usage.CompletionTokens)
	}
	if resp.Usage.TotalTokens != resp.Usage.PromptTokens+resp.Usage.CompletionTokens {
		t.Errorf("total=%d != prompt+completion (%d+%d)",
			resp.Usage.TotalTokens, resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	}
}

func TestChatCompletionsStream(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":    []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens":  16,
		"temperature": 0,
		"stream":      true,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("content-type=%q want text/event-stream", ct)
	}
	// Streaming rides the speculative fast path (per-token callback into the SSE
	// state machine) — NOT the per-token GPU-sample loop, which only remains for
	// repeat-penalty requests.
	if f.specCalls != 1 {
		t.Errorf("expected 1 spec stream call, got %d (sampleDevice=%d decodeNoCopy=%d)",
			f.specCalls, f.sampleDeviceHits, f.decodeNoCopy)
	}

	events := parseSSE(t, rec.Body.String())
	if len(events) == 0 {
		t.Fatal("no SSE events")
	}
	if events[len(events)-1] != "[DONE]" {
		t.Errorf("last event=%q want [DONE]", events[len(events)-1])
	}

	var sawRole, sawContent, sawFinish, sawUsage bool
	var content strings.Builder
	for _, e := range events {
		if e == "[DONE]" {
			continue
		}
		var sr StreamResponse
		if err := json.Unmarshal([]byte(e), &sr); err != nil {
			t.Fatalf("bad chunk %q: %v", e, err)
		}
		if len(sr.Choices) == 0 {
			continue
		}
		d := sr.Choices[0].Delta
		if d.Role == "assistant" {
			sawRole = true
		}
		if d.Content != "" {
			sawContent = true
			content.WriteString(d.Content)
		}
		if sr.Choices[0].FinishReason == "stop" {
			sawFinish = true
		}
		if sr.Usage != nil {
			sawUsage = true
			if sr.Usage.CompletionTokens != 2 {
				t.Errorf("usage completion=%d want 2", sr.Usage.CompletionTokens)
			}
		}
	}
	if !sawRole {
		t.Error("missing role delta chunk")
	}
	if !sawContent {
		t.Error("missing content chunk")
	}
	if !sawFinish {
		t.Error("missing finish_reason stop")
	}
	if !sawUsage {
		t.Error("missing final usage")
	}
	if got := strings.TrimSpace(content.String()); got != "hello world" {
		t.Errorf("streamed content=%q want %q", got, "hello world")
	}
}

func TestChatCompletionsInvalidJSON(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader("{not json"))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status=%d want 400", rec.Code)
	}
}

func TestChatCompletionsWrongMethod(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	req := httptest.NewRequest("GET", "/v1/chat/completions", nil)
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status=%d want 405", rec.Code)
	}
}

func TestChatCompletionsStopSequence(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	// Script decodes to " hello world"; the stop string "world" truncates it.
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":    []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens":  16,
		"temperature": 0,
		"stop":        "world",
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 (body=%s)", rec.Code, rec.Body.String())
	}
	// A stop string forces the per-token decode loop, not the spec path.
	if f.specCalls != 0 {
		t.Errorf("spec path used despite stop string (specCalls=%d)", f.specCalls)
	}
	if f.decodeCalls == 0 {
		t.Errorf("expected per-token Decode loop to run")
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	if got := resp.Choices[0].Message.Content; got != "hello" {
		t.Errorf("content=%q want %q (truncated before stop)", got, "hello")
	}
}

func TestChatCompletionsMaxTokensCap(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	// Long script, but max_tokens caps generation. A non-matching stop string
	// forces the per-token loop where the cap is enforced step by step.
	long := []int32{idx["▁hello"], idx["▁world"], idx["▁hello"], idx["▁world"], idx["▁hello"]}
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: long}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":    []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens":  2,
		"temperature": 0,
		"stop":        "ZZZ", // never matches
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200", rec.Code)
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	if resp.Usage.CompletionTokens != 2 {
		t.Errorf("completion_tokens=%d want 2 (max_tokens cap)", resp.Usage.CompletionTokens)
	}
}

func TestChatCompletionsContextCompaction(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	// Tiny context forces trimming of the oldest prompt tokens.
	const ctxSize = 16
	f := &fakeServerEngine{ctxSize: ctxSize, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	longPrompt := strings.Repeat("hello world ", 20)
	maxTokens := 8
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":    []map[string]string{{"role": "user", "content": longPrompt}},
		"max_tokens":  maxTokens,
		"temperature": 0,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 (body=%s)", rec.Code, rec.Body.String())
	}
	budget := ctxSize - maxTokens
	// After compaction the prefilled prompt must fit the budget (BOS may be
	// re-prepended, allowing budget+1).
	if f.lastPrefillLen > budget+1 {
		t.Errorf("prefilled %d tokens, want <= %d (compaction failed)", f.lastPrefillLen, budget+1)
	}
	if f.lastPrefillLen == 0 {
		t.Error("nothing prefilled")
	}
}

// Regression: when a client (pi) omits max_tokens and the prompt approaches the
// context window, the server must RESERVE a completion budget and compact the
// prompt — not collapse max_tokens toward 1 token (which produced dead, empty
// replies and meant compaction never fired). With no max_tokens the reserved
// budget is ctx/2, so the prompt is trimmed to fit ctx/2 (+1 BOS).
func TestChatCompletionsNoMaxTokensCompacts(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	const ctxSize = 16
	f := &fakeServerEngine{ctxSize: ctxSize, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	longPrompt := strings.Repeat("hello world ", 20)
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":    []map[string]string{{"role": "user", "content": longPrompt}},
		"temperature": 0,
		// no max_tokens — the pi path
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 (body=%s)", rec.Code, rec.Body.String())
	}
	// Reserved completion budget is ctx/2, so compaction trims the prompt to
	// ctx-ctx/2 = ctx/2 tokens (a leading BOS may be re-prepended).
	budget := ctxSize - ctxSize/2
	if f.lastPrefillLen > budget+1 {
		t.Errorf("prefilled %d tokens, want <= %d (no-max_tokens compaction failed)", f.lastPrefillLen, budget+1)
	}
	if f.lastPrefillLen == 0 {
		t.Error("nothing prefilled")
	}
}

func TestLegacyCompletions(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	req := httptest.NewRequest("POST", "/v1/completions", chatBody(t, map[string]interface{}{
		"prompt":      "hello",
		"max_tokens":  16,
		"temperature": 0,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 (body=%s)", rec.Code, rec.Body.String())
	}
	var resp CompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	if resp.Object != "text_completion" {
		t.Errorf("object=%q want text_completion", resp.Object)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("choices=%d want 1", len(resp.Choices))
	}
	if resp.Choices[0].Text != "hello world" {
		t.Errorf("text=%q want %q", resp.Choices[0].Text, "hello world")
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish=%q want stop", resp.Choices[0].FinishReason)
	}
}

func TestSetLogLevelSilencesAccessLog(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)

	var sb strings.Builder
	log.SetOutput(&sb)
	defer log.SetOutput(io.Discard)

	srv.SetLogLevel("warn")
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, httptest.NewRequest("GET", "/health", nil))
	if strings.Contains(sb.String(), "GET /health") {
		t.Errorf("access log emitted at warn level: %q", sb.String())
	}

	sb.Reset()
	srv.SetLogLevel("info")
	rec = httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, httptest.NewRequest("GET", "/health", nil))
	if !strings.Contains(sb.String(), "GET /health") {
		t.Errorf("access log missing at info level: %q", sb.String())
	}
}

// parseSSE splits an SSE response body into the JSON payloads of each `data:`
// line (the "[DONE]" sentinel is returned verbatim).
func parseSSE(t *testing.T, body string) []string {
	t.Helper()
	var out []string
	sc := bufio.NewScanner(strings.NewReader(body))
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		out = append(out, strings.TrimPrefix(line, "data: "))
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan SSE: %v", err)
	}
	if len(out) == 0 {
		t.Fatalf("no data lines in:\n%s", body)
	}
	return out
}
