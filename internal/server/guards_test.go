package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ─── cycle detector ──────────────────────────────────────────────────────────

func TestCycleDetectorPeriod1(t *testing.T) {
	d := &cycleDetector{}
	for i := 1; i <= 256; i++ {
		if d.push(42) {
			t.Fatalf("triggered at push %d, want no trigger before 257 (span %d)", i, cycleMinSpan)
		}
	}
	// Push 257: the trailing 256 tokens are 1-periodic with one period of history.
	if !d.push(42) {
		t.Fatal("no trigger at push 257 on a period-1 cycle")
	}
}

func TestCycleDetectorPeriod3(t *testing.T) {
	d := &cycleDetector{}
	cyc := []int32{7, 8, 9}
	for i := 0; i < 258; i++ {
		if d.push(cyc[i%3]) {
			t.Fatalf("triggered at push %d, want 259 (span %d + period 3)", i+1, cycleMinSpan)
		}
	}
	if !d.push(cyc[258%3]) {
		t.Fatal("no trigger at push 259 on a period-3 cycle")
	}
}

func TestCycleDetectorNoFalsePositives(t *testing.T) {
	// Distinct tokens never trigger.
	d := &cycleDetector{}
	for i := int32(0); i < 4096; i++ {
		if d.push(i) {
			t.Fatalf("false positive on distinct tokens at %d", i)
		}
	}
	// A coding agent re-emitting a long block verbatim looks like one giant
	// repetition with period >> maxPeriod — it must NOT trigger (this is the
	// legitimate ~100%-lookup-acceptance regime).
	d = &cycleDetector{}
	block := make([]int32, 400)
	for i := range block {
		block[i] = int32(1000 + i)
	}
	for rep := 0; rep < 3; rep++ {
		for i, tok := range block {
			if d.push(tok) {
				t.Fatalf("false positive on period-400 verbatim block (rep %d, idx %d)", rep, i)
			}
		}
	}
}

// ─── finish_reason "length" at the MaxTokens cap ────────────────────────────

// capScript yields more non-stop tokens than the request's max_tokens, with no
// short-period cycle (alternation period 2 would need 256+ tokens to trigger
// the detector, far above the cap used here).
func capScript(idx map[string]int32) []int32 {
	a, b := idx["▁hello"], idx["▁world"]
	s := make([]int32, 10)
	for i := range s {
		if i%2 == 0 {
			s[i] = a
		} else {
			s[i] = b
		}
	}
	return s
}

func TestNonStreamFinishLengthAtCap(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: capScript(idx)}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":   []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens": 4,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v (body=%s)", err, rec.Body.String())
	}
	if got := resp.Choices[0].FinishReason; got != "length" {
		t.Errorf("finish=%q want length (truncated at max_tokens must be visible to agents)", got)
	}
	if resp.Usage.CompletionTokens != 4 {
		t.Errorf("completion=%d want 4", resp.Usage.CompletionTokens)
	}
}

func TestStreamFinishLengthAtCap(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: capScript(idx)}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":   []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens": 4,
		"stream":     true,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	finish := ""
	for _, e := range parseSSE(t, rec.Body.String()) {
		if e == "[DONE]" {
			continue
		}
		var sr StreamResponse
		if json.Unmarshal([]byte(e), &sr) == nil && len(sr.Choices) > 0 && sr.Choices[0].FinishReason != "" {
			finish = sr.Choices[0].FinishReason
		}
	}
	if finish != "length" {
		t.Errorf("streamed finish=%q want length", finish)
	}
}

// ─── multiple tool calls per turn ────────────────────────────────────────────

// byteScript encodes plain ASCII as <0xXX> byte tokens (DecodeRaw reassembles
// the bytes), letting tests script arbitrary tool-call bodies.
func byteScript(t *testing.T, idx map[string]int32, s string) []int32 {
	t.Helper()
	out := make([]int32, 0, len(s))
	for i := 0; i < len(s); i++ {
		id, ok := idx[byteTok(int(s[i]))]
		if !ok {
			t.Fatalf("byte token %q missing from test vocab", byteTok(int(s[i])))
		}
		out = append(out, id)
	}
	return out
}

func multiCallScript(t *testing.T, idx map[string]int32) []int32 {
	tcOpen, tcEnd, turnEnd := idx["<|tool_call>"], idx["<tool_call|>"], idx["<turn|>"]
	s := []int32{tcOpen}
	s = append(s, byteScript(t, idx, "call:alpha{}")...)
	s = append(s, tcEnd, tcOpen)
	s = append(s, byteScript(t, idx, "call:beta{}")...)
	s = append(s, tcEnd, turnEnd)
	return s
}

func toolsField() []map[string]interface{} {
	return []map[string]interface{}{
		{"type": "function", "function": map[string]interface{}{
			"name":       "alpha",
			"parameters": map[string]interface{}{"type": "object", "properties": map[string]interface{}{}},
		}},
		{"type": "function", "function": map[string]interface{}{
			"name":       "beta",
			"parameters": map[string]interface{}{"type": "object", "properties": map[string]interface{}{}},
		}},
	}
}

func TestStreamMultiToolCalls(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: multiCallScript(t, idx)}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
		"tools":    toolsField(),
		"stream":   true,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	var names []string
	finish := ""
	for _, e := range parseSSE(t, rec.Body.String()) {
		if e == "[DONE]" {
			continue
		}
		var sr StreamResponse
		if json.Unmarshal([]byte(e), &sr) != nil || len(sr.Choices) == 0 {
			continue
		}
		for _, tc := range sr.Choices[0].Delta.ToolCalls {
			names = append(names, tc.Function.Name)
		}
		if sr.Choices[0].FinishReason != "" {
			finish = sr.Choices[0].FinishReason
		}
	}
	if len(names) != 2 || names[0] != "alpha" || names[1] != "beta" {
		t.Errorf("streamed tool calls=%v want [alpha beta] — generation must NOT stop at the first <tool_call|>", names)
	}
	if finish != "tool_calls" {
		t.Errorf("finish=%q want tool_calls", finish)
	}
}

func TestNonStreamMultiToolCalls(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: multiCallScript(t, idx)}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
		"tools":    toolsField(),
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v (body=%s)", err, rec.Body.String())
	}
	calls := resp.Choices[0].Message.ToolCalls
	if len(calls) != 2 || calls[0].Function.Name != "alpha" || calls[1].Function.Name != "beta" {
		t.Errorf("tool calls=%+v want alpha+beta", calls)
	}
	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("finish=%q want tool_calls", resp.Choices[0].FinishReason)
	}
}

// ─── thinking budget force-close ─────────────────────────────────────────────

func TestThinkBudgetForceClose(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	chOpen, chEnd := idx["<|channel>"], idx["<channel|>"]
	// Script: open thought channel, 3 think tokens (budget), one sacrificial
	// token (consumed by the fake's cursor advancing on the injected
	// Decode(<channel|>)), then the visible answer and EOS.
	script := []int32{chOpen, idx["h"], idx["e"], idx["l"], idx["o"], idx["▁hello"], tk.EOS}
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: script}
	srv := New(f, tk)
	srv.SetLogLevel("warn")
	srv.SetThinkBudget(3)

	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v (body=%s)", err, rec.Body.String())
	}
	// The channel was force-closed: exactly one Decode (the injected <channel|>)
	// and the answer still made it out instead of an empty-content turn.
	if f.decodeCalls != 1 {
		t.Errorf("decodeCalls=%d want 1 (the injected <channel|>)", f.decodeCalls)
	}
	found := false
	for _, tok := range f.tokens {
		if tok == chEnd {
			found = true
		}
	}
	if !found {
		t.Error("injected <channel|> never committed to the engine KV")
	}
	if got := resp.Choices[0].Message.Content; !strings.Contains(got, "hello") {
		t.Errorf("content=%q want the post-budget answer to contain %q", got, "hello")
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish=%q want stop (budget must not kill the turn)", resp.Choices[0].FinishReason)
	}
}

// ─── runaway repetition inside a tool-call span (the Req3 incident shape) ────

func TestStreamRunawayInsideToolCall(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	// An unterminated tool call whose body degenerates into a period-1 cycle:
	// previously this silently buffered everything to the 8192 cap and the turn
	// died with finish "stop" and no output. Now the cycle detector cuts it and
	// the truncation is labeled.
	script := []int32{idx["<|tool_call>"]}
	rep := idx[byteTok('a')]
	for i := 0; i < 600; i++ {
		script = append(script, rep)
	}
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: script}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
		"tools":    toolsField(),
		"stream":   true,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	finish := ""
	completion := 0
	sawToolDelta := false
	for _, e := range parseSSE(t, rec.Body.String()) {
		if e == "[DONE]" {
			continue
		}
		var sr StreamResponse
		if json.Unmarshal([]byte(e), &sr) != nil || len(sr.Choices) == 0 {
			continue
		}
		if len(sr.Choices[0].Delta.ToolCalls) > 0 {
			sawToolDelta = true
		}
		if sr.Choices[0].FinishReason != "" {
			finish = sr.Choices[0].FinishReason
		}
		if sr.Usage != nil {
			completion = sr.Usage.CompletionTokens
		}
	}
	if finish != "length" {
		t.Errorf("finish=%q want length (cycle cut must be visible)", finish)
	}
	if sawToolDelta {
		t.Error("a truncated tool call must not be emitted as a valid tool_calls delta")
	}
	if completion == 0 || completion > 400 {
		t.Errorf("completion=%d want bounded ~257 (cycle detector), not the full script", completion)
	}
}

// ─── cancelled-before-prefill fast path ──────────────────────────────────────

func TestCancelledRequestSkipsPrefill(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // client already gone when the handler runs
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})).WithContext(ctx)
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	if f.lastPrefillLen != 0 {
		t.Errorf("prefill ran (%d tokens) for an already-cancelled request — zombie prefills stack up behind the kv lock", f.lastPrefillLen)
	}
	if f.specCalls != 0 {
		t.Errorf("generation ran for a cancelled request (specCalls=%d)", f.specCalls)
	}
	if rec.Code == http.StatusOK {
		t.Errorf("status=%d want non-200 for a cancelled non-streaming request", rec.Code)
	}
}
