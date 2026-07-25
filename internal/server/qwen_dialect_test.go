package server

// End-to-end tests of the Qwen ChatML dialect through BOTH serving paths:
// the single-flight spec path (runSpec/streamResponse/generateResponse) and
// the continuous-batching path (serveBatch/streamBatch/collectBatch), with a
// scripted engine standing in for the GPU. These lock the agentic contract:
// reasoning_content split, structured tool_calls with schema-coerced JSON
// arguments, streaming tool-call deltas, and finish_reason "tool_calls".

import (
	"bufio"
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/hikmaai-io/fucina/internal/server/batch"
	"github.com/hikmaai-io/fucina/internal/tokenizer"
)

// qwenVocabWords are the regular (non-special) vocab entries the scripted
// generations decode through. Ids are 10+index.
var qwenVocabWords = []string{
	"I should check the weather", // 10
	"\n",                         // 11
	"<function=get_weather>",     // 12
	"<parameter=city>",           // 13
	"Paris",                      // 14
	"</parameter>",               // 15
	"</function>",                // 16
	"Hello",                      // 17
	" world",                     // 18
	"\n\n",                       // 19
	"The answer is 42.",          // 20
	"get_weather>",               // 21 (completion after tool_choice:"required" prefix)
	`{"name":"get_weather","arguments":{"city":"Paris"}}`, // 22 legacy Qwen JSON body
}

// newQwenTokenizer builds a ChatML tokenizer from a synthetic HF
// tokenizer.json (the Qwen3.5 shape: added ChatML specials, no bos).
func newQwenTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	vocab := map[string]int32{}
	for i, w := range qwenVocabWords {
		vocab[w] = int32(10 + i)
	}
	var added []map[string]interface{}
	for i, s := range []string{
		"<|endoftext|>", "<|im_start|>", "<|im_end|>",
		"<think>", "</think>", "<tool_call>", "</tool_call>",
		"<tool_response>", "</tool_response>",
	} {
		added = append(added, map[string]interface{}{"id": 100 + i, "content": s, "special": true})
	}
	doc := map[string]interface{}{
		"added_tokens": added,
		"model": map[string]interface{}{
			"type":   "BPE",
			"vocab":  vocab,
			"merges": []string{},
		},
	}
	data, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	tk, err := tokenizer.NewFromHFJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	return tk
}

const (
	qEOT      = 100 // <|endoftext|>
	qImEnd    = 102 // <|im_end|>
	qThinkEnd = 104 // </think>
	qTCOpen   = 105 // <tool_call>
	qTCEnd    = 106 // </tool_call>
)

// qwenToolCallScript is a thinking-on turn that reasons, then emits one
// get_weather call: reasoning "</think>" "\n\n" <tool_call> "\n<function=
// get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n"
// </tool_call> <|im_end|>.
var qwenToolCallScript = []int32{
	10, qThinkEnd, 19,
	qTCOpen, 11, 12, 11, 13, 11, 14, 11, 15, 11, 16, 11, qTCEnd,
	qImEnd,
}

var qwenWeatherTool = map[string]interface{}{
	"type": "function",
	"function": map[string]interface{}{
		"name":        "get_weather",
		"description": "Get the weather",
		"parameters": map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"city": map[string]interface{}{"type": "string"},
			},
			"required": []string{"city"},
		},
	},
}

func newQwenSingleFlightServer(t *testing.T, script []int32) *Server {
	t.Helper()
	tk := newQwenTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: script}
	srv := New(f, tk)
	srv.SetLogLevel("warn")
	if srv.dialect.Name() != "qwen" {
		t.Fatalf("dialect = %s, want qwen (ChatML vocab auto-detection)", srv.dialect.Name())
	}
	return srv
}

func TestQwenDialect_SingleFlight_ToolCall(t *testing.T) {
	srv := newQwenSingleFlightServer(t, qwenToolCallScript)
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":   []map[string]string{{"role": "user", "content": "weather in Paris?"}},
		"tools":      []interface{}{qwenWeatherTool},
		"thinking":   true,
		"max_tokens": 64,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	msg := resp.Choices[0].Message
	if !strings.Contains(msg.ReasoningContent, "I should check the weather") {
		t.Errorf("reasoning_content = %q", msg.ReasoningContent)
	}
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("tool_calls = %+v, want 1 call", msg.ToolCalls)
	}
	tc := msg.ToolCalls[0]
	if tc.Function.Name != "get_weather" || tc.Function.Arguments != `{"city":"Paris"}` {
		t.Errorf("call = %s(%s)", tc.Function.Name, tc.Function.Arguments)
	}
	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("finish_reason = %q", resp.Choices[0].FinishReason)
	}
	if strings.Contains(msg.Content, "think") || strings.Contains(msg.Content, "tool_call") {
		t.Errorf("markers leaked into content: %q", msg.Content)
	}
}

func TestQwenDialect_SingleFlight_PlainThinking(t *testing.T) {
	// Reason, close the block, answer in prose.
	srv := newQwenSingleFlightServer(t, []int32{10, qThinkEnd, 19, 20, qImEnd})
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":   []map[string]string{{"role": "user", "content": "meaning of life?"}},
		"thinking":   true,
		"max_tokens": 64,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	msg := resp.Choices[0].Message
	if !strings.Contains(msg.ReasoningContent, "I should check the weather") {
		t.Errorf("reasoning_content = %q", msg.ReasoningContent)
	}
	if msg.Content != "The answer is 42." {
		t.Errorf("content = %q", msg.Content)
	}
}

// ─── continuous-batching path ──────────────────────────────────────────────

// scriptedBatchEngine is a deterministic BatchEngine: every admitted sequence
// replays the same script, one token per step.
type scriptedBatchEngine struct {
	mu     sync.Mutex
	script []int32
	pos    map[int]int
	next   int
}

func newScriptedBatchEngine(script []int32) *scriptedBatchEngine {
	return &scriptedBatchEngine{script: script, pos: map[int]int{}}
}

func (e *scriptedBatchEngine) Supported() bool { return true }
func (e *scriptedBatchEngine) Capacity() int   { return 4 }

func (e *scriptedBatchEngine) AddSeq(prompt []int32, _ batch.SeqParams) (int, int32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	slot := e.next
	e.next++
	e.pos[slot] = 1
	if len(e.script) == 0 {
		return slot, qImEnd, nil
	}
	return slot, e.script[0], nil
}

func (e *scriptedBatchEngine) StepBatch(active []int32, inputs []int32) ([][]int32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([][]int32, len(active))
	for i, slot := range active {
		p := e.pos[int(slot)]
		if p >= len(e.script) {
			out[i] = []int32{qImEnd}
			continue
		}
		out[i] = []int32{e.script[p]}
		e.pos[int(slot)] = p + 1
	}
	return out, nil
}

func (e *scriptedBatchEngine) RemoveSeq(slot int) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.pos, slot)
	return nil
}

func newQwenBatchServer(t *testing.T, script []int32) *Server {
	t.Helper()
	tk := newQwenTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS}
	srv := New(f, tk)
	srv.SetLogLevel("warn")
	if !srv.SetBatchEngine(newScriptedBatchEngine(script)) {
		t.Fatal("SetBatchEngine refused the scripted engine")
	}
	t.Cleanup(srv.scheduler.Shutdown)
	return srv
}

func TestQwenDialect_Batch_CollectToolCall(t *testing.T) {
	srv := newQwenBatchServer(t, qwenToolCallScript)
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":   []map[string]string{{"role": "user", "content": "weather in Paris?"}},
		"tools":      []interface{}{qwenWeatherTool},
		"thinking":   true,
		"max_tokens": 64,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	msg := resp.Choices[0].Message
	if !strings.Contains(msg.ReasoningContent, "I should check the weather") {
		t.Errorf("reasoning_content = %q", msg.ReasoningContent)
	}
	if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].Function.Arguments != `{"city":"Paris"}` {
		t.Fatalf("tool_calls = %+v", msg.ToolCalls)
	}
	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("finish_reason = %q", resp.Choices[0].FinishReason)
	}
}

// sseEvents parses an SSE body into its JSON data payloads.
func sseEvents(t *testing.T, body string) []StreamResponse {
	t.Helper()
	var events []StreamResponse
	sc := bufio.NewScanner(strings.NewReader(body))
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var ev StreamResponse
		if err := json.Unmarshal([]byte(line[len("data: "):]), &ev); err != nil {
			t.Fatalf("bad SSE payload %q: %v", line, err)
		}
		events = append(events, ev)
	}
	return events
}

func TestQwenDialect_Batch_StreamToolCall(t *testing.T) {
	srv := newQwenBatchServer(t, qwenToolCallScript)
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":   []map[string]string{{"role": "user", "content": "weather in Paris?"}},
		"tools":      []interface{}{qwenWeatherTool},
		"thinking":   true,
		"stream":     true,
		"max_tokens": 64,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var reasoning, content string
	var calls []DeltaToolCall
	finish := ""
	for _, ev := range sseEvents(t, rec.Body.String()) {
		for _, ch := range ev.Choices {
			reasoning += ch.Delta.ReasoningContent
			content += ch.Delta.Content
			calls = append(calls, ch.Delta.ToolCalls...)
			if ch.FinishReason != "" {
				finish = ch.FinishReason
			}
		}
	}
	if !strings.Contains(reasoning, "I should check the weather") {
		t.Errorf("streamed reasoning = %q", reasoning)
	}
	if strings.TrimSpace(content) != "" {
		t.Errorf("unexpected streamed content: %q", content)
	}
	if len(calls) != 1 {
		t.Fatalf("streamed tool_calls = %+v, want 1", calls)
	}
	if calls[0].Function.Name != "get_weather" || calls[0].Function.Arguments != `{"city":"Paris"}` {
		t.Errorf("streamed call = %s(%s)", calls[0].Function.Name, calls[0].Function.Arguments)
	}
	if finish != "tool_calls" {
		t.Errorf("finish_reason = %q", finish)
	}
}

func TestQwenDialect_Batch_StreamMixedContentAndXMLCall(t *testing.T) {
	call := qwenToolCallScript[3 : len(qwenToolCallScript)-1]
	script := append([]int32{17, 18, 19}, call...)
	script = append(script, qImEnd)
	srv := newQwenBatchServer(t, script)
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages": []map[string]string{{"role": "user", "content": "say and check"}},
		"tools":    []interface{}{qwenWeatherTool}, "stream": true, "max_tokens": 64,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	var content string
	var calls []DeltaToolCall
	for _, ev := range sseEvents(t, rec.Body.String()) {
		for _, ch := range ev.Choices {
			content += ch.Delta.Content
			calls = append(calls, ch.Delta.ToolCalls...)
		}
	}
	if content != "Hello world\n\n" {
		t.Errorf("mixed streamed content=%q", content)
	}
	if len(calls) != 1 || calls[0].Function.Arguments != `{"city":"Paris"}` {
		t.Fatalf("mixed streamed calls=%+v", calls)
	}
}

func TestQwenDialect_Batch_StreamLegacyJSONCall(t *testing.T) {
	srv := newQwenBatchServer(t, []int32{qTCOpen, 22, qTCEnd, qImEnd})
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages": []map[string]string{{"role": "user", "content": "weather"}},
		"tools":    []interface{}{qwenWeatherTool}, "stream": true, "max_tokens": 16,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	var calls []DeltaToolCall
	for _, ev := range sseEvents(t, rec.Body.String()) {
		for _, ch := range ev.Choices {
			calls = append(calls, ch.Delta.ToolCalls...)
		}
	}
	if len(calls) != 1 || calls[0].Function.Arguments != `{"city":"Paris"}` {
		t.Fatalf("legacy JSON stream calls=%+v", calls)
	}
}

func TestQwenDialect_Batch_StreamPlain(t *testing.T) {
	// Thinking off: prompt pre-closes the think block; output is prose only.
	srv := newQwenBatchServer(t, []int32{17, 18, qImEnd})
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":   []map[string]string{{"role": "user", "content": "hi"}},
		"thinking":   false,
		"stream":     true,
		"max_tokens": 16,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	var content, reasoning string
	for _, ev := range sseEvents(t, rec.Body.String()) {
		for _, ch := range ev.Choices {
			content += ch.Delta.Content
			reasoning += ch.Delta.ReasoningContent
		}
	}
	if content != "Hello world" {
		t.Errorf("content = %q", content)
	}
	if reasoning != "" {
		t.Errorf("unexpected reasoning: %q", reasoning)
	}
}

func TestQwenDialect_Batch_StreamMultipleToolCalls(t *testing.T) {
	one := qwenToolCallScript[3 : len(qwenToolCallScript)-1] // open...close, no reasoning/EOS
	script := append(append([]int32{}, one...), one...)
	script = append(script, qImEnd)
	srv := newQwenBatchServer(t, script)
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages": []map[string]string{{"role": "user", "content": "weather twice"}},
		"tools":    []interface{}{qwenWeatherTool}, "stream": true, "max_tokens": 64,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	var calls []DeltaToolCall
	for _, ev := range sseEvents(t, rec.Body.String()) {
		for _, ch := range ev.Choices {
			calls = append(calls, ch.Delta.ToolCalls...)
		}
	}
	if len(calls) != 2 {
		t.Fatalf("streamed tool_calls = %+v, want 2", calls)
	}
	if calls[0].Index != 0 || calls[1].Index != 1 || calls[0].ID == calls[1].ID {
		t.Errorf("bad streaming call indexes/IDs: %+v", calls)
	}
}

func TestQwenDialect_Batch_UnterminatedFallback(t *testing.T) {
	// EOS arrives after a partial function. The fragment must not dispatch, but
	// unlike the old dead-turn behavior it must be emitted as content.
	script := []int32{qTCOpen, 11, 12, 11, 13, 11, 14, qImEnd}
	srv := newQwenBatchServer(t, script)
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages": []map[string]string{{"role": "user", "content": "weather"}},
		"tools":    []interface{}{qwenWeatherTool}, "stream": true, "max_tokens": 32,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	var content string
	var calls []DeltaToolCall
	for _, ev := range sseEvents(t, rec.Body.String()) {
		for _, ch := range ev.Choices {
			content += ch.Delta.Content
			calls = append(calls, ch.Delta.ToolCalls...)
		}
	}
	if len(calls) != 0 {
		t.Fatalf("partial call dispatched: %+v", calls)
	}
	if !strings.Contains(content, "<function=get_weather>") || !strings.Contains(content, "Paris") {
		t.Errorf("buffered fallback missing from content: %q", content)
	}
}

func TestQwenDialect_Batch_CollectUnterminatedFallback(t *testing.T) {
	script := []int32{qTCOpen, 11, 12, 11, 13, 11, 14, qImEnd}
	srv := newQwenBatchServer(t, script)
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages": []map[string]string{{"role": "user", "content": "weather"}},
		"tools":    []interface{}{qwenWeatherTool}, "max_tokens": 32,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	msg := resp.Choices[0].Message
	if len(msg.ToolCalls) != 0 || !strings.Contains(msg.Content, "<function=get_weather>") ||
		!strings.Contains(msg.Content, "Paris") {
		t.Fatalf("unterminated collected response=%+v", resp.Choices[0])
	}
}

func TestQwenDialect_Batch_StreamCapFallback(t *testing.T) {
	script := []int32{qTCOpen}
	for i := 0; i < 2050; i++ {
		script = append(script, 14) // Paris
	}
	script = append(script, qImEnd)
	srv := newQwenBatchServer(t, script)
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages": []map[string]string{{"role": "user", "content": "weather"}},
		"tools":    []interface{}{qwenWeatherTool}, "stream": true, "max_tokens": 4096,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	var content, finish string
	var calls []DeltaToolCall
	for _, ev := range sseEvents(t, rec.Body.String()) {
		for _, ch := range ev.Choices {
			content += ch.Delta.Content
			calls = append(calls, ch.Delta.ToolCalls...)
			if ch.FinishReason != "" {
				finish = ch.FinishReason
			}
		}
	}
	if len(calls) != 0 || finish != "length" {
		t.Fatalf("cap fallback calls=%+v finish=%q", calls, finish)
	}
	if got := strings.Count(content, "Paris"); got != 2049 {
		t.Fatalf("cap fallback preserved %d/2049 buffered values", got)
	}
}

func TestQwenDialect_Batch_UnterminatedOuterRecoversCall(t *testing.T) {
	// Qwen occasionally emits EOS after </function> without </tool_call>. The
	// complete inner call is safe and should still become structured output.
	script := []int32{qTCOpen, 11, 12, 11, 13, 11, 14, 11, 15, 11, 16, qImEnd}
	srv := newQwenBatchServer(t, script)
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages": []map[string]string{{"role": "user", "content": "weather"}},
		"tools":    []interface{}{qwenWeatherTool}, "stream": true, "max_tokens": 32,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	var calls []DeltaToolCall
	for _, ev := range sseEvents(t, rec.Body.String()) {
		for _, ch := range ev.Choices {
			calls = append(calls, ch.Delta.ToolCalls...)
		}
	}
	if len(calls) != 1 || calls[0].Function.Arguments != `{"city":"Paris"}` {
		t.Fatalf("unterminated outer call not recovered: %+v", calls)
	}
}

func TestQwenDialect_RequiredAnyToolChoice(t *testing.T) {
	// "required" opens <function= in the prompt; generation starts with the
	// selected function name and completes the body.
	script := []int32{21, 11, 13, 11, 14, 11, 15, 11, 16, 11, qTCEnd, qImEnd}
	srv := newQwenBatchServer(t, script)
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages": []map[string]string{{"role": "user", "content": "7 times 8"}},
		"tools":    []interface{}{qwenWeatherTool}, "tool_choice": "required", "max_tokens": 64,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Choices[0].Message.ToolCalls) != 1 || resp.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("required tool call not enforced: %+v", resp.Choices[0])
	}
}

func TestQwenDialect_RequiredMalformedIsNotCleanStop(t *testing.T) {
	srv := newQwenBatchServer(t, []int32{21, qImEnd})
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages": []map[string]string{{"role": "user", "content": "weather"}},
		"tools":    []interface{}{qwenWeatherTool}, "tool_choice": "required", "max_tokens": 16,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	choice := resp.Choices[0]
	if choice.FinishReason == "stop" || len(choice.Message.ToolCalls) != 0 {
		t.Fatalf("malformed required call reported as success: %+v", choice)
	}
	if choice.Message.Content == "" {
		t.Fatalf("malformed required call discarded its buffered fallback: %+v", choice)
	}
}

func TestQwenDialect_UnknownForcedToolRejected(t *testing.T) {
	srv := newQwenBatchServer(t, qwenToolCallScript)
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":    []map[string]string{{"role": "user", "content": "weather"}},
		"tools":       []interface{}{qwenWeatherTool},
		"tool_choice": map[string]interface{}{"type": "function", "function": map[string]interface{}{"name": "not_a_tool"}},
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != 400 || !strings.Contains(rec.Body.String(), "unknown function") {
		t.Fatalf("unknown forced tool: status=%d body=%q", rec.Code, rec.Body.String())
	}
}

func TestQwenDialect_ForcedToolChoice(t *testing.T) {
	// tool_choice forces get_weather: the prompt ends with the forced prefix
	// "<tool_call>\n<function=get_weather>\n", so the model only completes the
	// body — "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>".
	script := []int32{13, 11, 14, 11, 15, 11, 16, 11, qTCEnd, qImEnd}
	srv := newQwenBatchServer(t, script)
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":    []map[string]string{{"role": "user", "content": "weather in Paris?"}},
		"tools":       []interface{}{qwenWeatherTool},
		"tool_choice": map[string]interface{}{"type": "function", "function": map[string]interface{}{"name": "get_weather"}},
		"max_tokens":  64,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	msg := resp.Choices[0].Message
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("tool_calls = %+v (forced call not parsed)", msg.ToolCalls)
	}
	if msg.ToolCalls[0].Function.Name != "get_weather" ||
		msg.ToolCalls[0].Function.Arguments != `{"city":"Paris"}` {
		t.Errorf("forced call = %s(%s)", msg.ToolCalls[0].Function.Name, msg.ToolCalls[0].Function.Arguments)
	}
	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("finish_reason = %q", resp.Choices[0].FinishReason)
	}
}
