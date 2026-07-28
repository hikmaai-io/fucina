package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func twoTurnCompletionPrompt(t *testing.T, prompt1 string, first CompletionResponse) string {
	t.Helper()
	if len(first.Choices) == 0 {
		t.Fatal("first response missing choices")
	}
	return prompt1 + first.Choices[0].Text + " again"
}

func runLegacyCompletion(t *testing.T, srv *Server, prompt string, stream bool) (CompletionResponse, []string) {
	t.Helper()
	body := map[string]interface{}{
		"prompt":      prompt,
		"max_tokens":  16,
		"temperature": 0,
		"stream":      stream,
	}
	if stream {
		body["stream_options"] = map[string]bool{"include_usage": true}
	}
	req := httptest.NewRequest("POST", "/v1/completions", chatBody(t, body))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if !stream {
		var resp CompletionResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("bad json: %v", err)
		}
		return resp, nil
	}
	events := parseSSE(t, rec.Body.String())
	var final CompletionResponse
	for _, e := range events {
		if e == "[DONE]" {
			continue
		}
		if err := json.Unmarshal([]byte(e), &final); err != nil {
			t.Fatalf("bad sse chunk %q: %v", e, err)
		}
	}
	return final, events
}

func cachedTokens(u Usage) int {
	if u.PromptTokensDetails == nil {
		return 0
	}
	return u.PromptTokensDetails.CachedTokens
}

func assertUsageInvariants(t *testing.T, u Usage) {
	t.Helper()
	cached := cachedTokens(u)
	if cached < 0 || cached > u.PromptTokens {
		t.Fatalf("cached=%d prompt=%d usage=%+v", cached, u.PromptTokens, u)
	}
	if u.TotalTokens != u.PromptTokens+u.CompletionTokens {
		t.Fatalf("total invariant failed: %+v", u)
	}
	if u.PromptTokens-cached < 0 {
		t.Fatalf("new prefill negative: %+v", u)
	}
}

func TestCompletionColdRequestCachedTokensZero(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
	srv := New(f, tk)
	srv.SetLogLevel("warn")
	resp, _ := runLegacyCompletion(t, srv, "hi", false)
	assertUsageInvariants(t, resp.Usage)
	if got := cachedTokens(resp.Usage); got != 0 {
		t.Fatalf("cold cached=%d want 0 usage=%+v", got, resp.Usage)
	}
}

func TestCompletionSecondTurnCachedTokensNonZero(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	first, _ := runLegacyCompletion(t, srv, "hi", false)
	secondPrompt := twoTurnCompletionPrompt(t, "hi", first)
	second, _ := runLegacyCompletion(t, srv, secondPrompt, false)
	assertUsageInvariants(t, second.Usage)
	if second.Usage.PromptTokensDetails == nil || second.Usage.PromptTokensDetails.CachedTokens <= 0 {
		t.Fatalf("second-turn usage=%+v want non-zero cached tokens", second.Usage)
	}
}

func TestCompletionPartialHitReportsExactPhysicalSkip(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
	srv := New(f, tk)
	srv.SetLogLevel("warn")
	first, _ := runLegacyCompletion(t, srv, "hi", false)
	secondPrompt := twoTurnCompletionPrompt(t, "hi", first)
	second, _ := runLegacyCompletion(t, srv, secondPrompt, false)
	assertUsageInvariants(t, second.Usage)
	if got := cachedTokens(second.Usage); got != 3 {
		t.Fatalf("partial-hit cached=%d want exact 3 usage=%+v", got, second.Usage)
	}
	if newPrefill := second.Usage.PromptTokens - cachedTokens(second.Usage); newPrefill != 8 {
		t.Fatalf("new prefill=%d want 8 usage=%+v", newPrefill, second.Usage)
	}
}

func TestCompletionSecondTurnStreamCollectedUsageParity(t *testing.T) {
	mk := func() *Server {
		tk, idx := newServerTokenizer(t)
		f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
		srv := New(f, tk)
		srv.SetLogLevel("warn")
		return srv
	}
	colSrv := mk()
	first, _ := runLegacyCompletion(t, colSrv, "hi", false)
	secondPrompt := twoTurnCompletionPrompt(t, "hi", first)
	col, _ := runLegacyCompletion(t, colSrv, secondPrompt, false)

	streamSrv := mk()
	first2, _ := runLegacyCompletion(t, streamSrv, "hi", false)
	secondPrompt2 := twoTurnCompletionPrompt(t, "hi", first2)
	stream, events := runLegacyCompletion(t, streamSrv, secondPrompt2, true)
	if len(events) == 0 || events[len(events)-1] != "[DONE]" {
		t.Fatalf("missing [DONE]: %v", events)
	}
	assertUsageInvariants(t, col.Usage)
	assertUsageInvariants(t, stream.Usage)
	if col.Usage.PromptTokens != stream.Usage.PromptTokens ||
		col.Usage.CompletionTokens != stream.Usage.CompletionTokens ||
		col.Usage.TotalTokens != stream.Usage.TotalTokens {
		t.Fatalf("usage mismatch collected=%+v streamed=%+v", col.Usage, stream.Usage)
	}
	if (col.Usage.PromptTokensDetails == nil) != (stream.Usage.PromptTokensDetails == nil) {
		t.Fatalf("prompt detail presence mismatch collected=%+v streamed=%+v", col.Usage, stream.Usage)
	}
	if col.Usage.PromptTokensDetails != nil && col.Usage.PromptTokensDetails.CachedTokens != stream.Usage.PromptTokensDetails.CachedTokens {
		t.Fatalf("cached mismatch collected=%+v streamed=%+v", col.Usage, stream.Usage)
	}
}

func TestBatchStreamFinalUsagePromptTokensNonZero(t *testing.T) {
	srv := newQwenBatchServer(t, []int32{17, 18, qImEnd})
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":       []map[string]string{{"role": "user", "content": "hi"}},
		"stream":         true,
		"stream_options": map[string]bool{"include_usage": true},
		"max_tokens":     16,
		"temperature":    0,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var sawUsage bool
	for _, e := range parseSSE(t, rec.Body.String()) {
		if e == "[DONE]" {
			continue
		}
		var chunk StreamResponse
		if err := json.Unmarshal([]byte(e), &chunk); err != nil {
			t.Fatalf("bad chunk: %v", err)
		}
		if chunk.Usage != nil {
			sawUsage = true
			assertUsageInvariants(t, *chunk.Usage)
			if chunk.Usage.PromptTokens <= 0 {
				t.Fatalf("final usage prompt_tokens=%d want >0", chunk.Usage.PromptTokens)
			}
		}
	}
	if !sawUsage {
		t.Fatal("missing final usage chunk")
	}
}

func TestAnthropicCollectedUsageReportsReadAndCreationWithoutDoubleCount(t *testing.T) {
	resp := chatToAnthropic(ChatResponse{
		ID:      "chatcmpl-1",
		Model:   "local",
		Choices: []Choice{{Index: 0, Message: ChatMessage{Role: "assistant", Content: "ok"}, FinishReason: "stop"}},
		Usage: Usage{
			PromptTokens:        100,
			CompletionTokens:    2,
			TotalTokens:         102,
			PromptTokensDetails: &PromptTokensDetails{CachedTokens: 60},
		},
	})
	if resp.Usage.CacheReadInputTokens != 60 || resp.Usage.CacheCreationInputTokens != 40 || resp.Usage.InputTokens != 0 {
		t.Fatalf("usage=%+v", resp.Usage)
	}
	if got := resp.Usage.InputTokens + resp.Usage.CacheReadInputTokens + resp.Usage.CacheCreationInputTokens; got != 100 {
		t.Fatalf("logical input=%d want 100 (%+v)", got, resp.Usage)
	}
}

func TestAnthropicStreamUsageConsistentWithMessageStart(t *testing.T) {
	rec := httptest.NewRecorder()
	aw := newAnthropicResponseWriter(rec, true, 100)
	aw.Header().Set("Content-Type", "text/event-stream")
	aw.WriteHeader(http.StatusOK)
	chunk := StreamResponse{ID: "chatcmpl-abc", Model: "local", Choices: []StreamChoice{{Index: 0, Delta: Delta{}, FinishReason: "stop"}}, Usage: &Usage{PromptTokens: 100, CompletionTokens: 7, TotalTokens: 107, PromptTokensDetails: &PromptTokensDetails{CachedTokens: 60}}}
	data, _ := json.Marshal(chunk)
	if _, err := aw.Write([]byte("data: " + string(data) + "\n\n")); err != nil {
		t.Fatal(err)
	}
	if _, err := aw.Write([]byte("data: [DONE]\n\n")); err != nil {
		t.Fatal(err)
	}
	aw.finish()
	events := parseAnthropicSSE(t, rec.Body.String())
	if len(events) < 3 {
		t.Fatalf("events=%v", eventNames(events))
	}
	var start struct {
		Message struct {
			Usage anthropicUsage `json:"usage"`
		} `json:"message"`
	}
	if err := json.Unmarshal([]byte(events[0].data), &start); err != nil {
		t.Fatal(err)
	}
	var delta struct {
		Usage anthropicUsage `json:"usage"`
	}
	if err := json.Unmarshal([]byte(events[len(events)-2].data), &delta); err != nil {
		t.Fatal(err)
	}
	if start.Message.Usage.InputTokens != delta.Usage.InputTokens+delta.Usage.CacheReadInputTokens+delta.Usage.CacheCreationInputTokens {
		t.Fatalf("start=%+v delta=%+v", start.Message.Usage, delta.Usage)
	}
	if delta.Usage.CacheReadInputTokens != 60 || delta.Usage.CacheCreationInputTokens != 40 {
		t.Fatalf("delta usage=%+v", delta.Usage)
	}
}

func TestCompletionSecondTurnPromptContainsReturnedText(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
	srv := New(f, tk)
	first, _ := runLegacyCompletion(t, srv, "hi", false)
	secondPrompt := twoTurnCompletionPrompt(t, "hi", first)
	if !strings.Contains(secondPrompt, strings.TrimSpace(first.Choices[0].Text)) {
		t.Fatalf("prompt=%q first=%q", secondPrompt, first.Choices[0].Text)
	}
}
