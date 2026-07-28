package server

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAnthropicRequestTranslation(t *testing.T) {
	in := anthropicRequest{
		Model: "local-model", MaxTokens: 128, Stream: true,
		System: json.RawMessage(`[{"type":"text","text":"be concise"}]`),
		Messages: []anthropicInputMessage{
			{Role: "user", Content: json.RawMessage(`"weather?"`)},
			{Role: "assistant", Content: json.RawMessage(`[
				{"type":"thinking","thinking":"need a tool"},
				{"type":"tool_use","id":"toolu_1","name":"weather","input":{"city":"Paris"}}
			]`)},
			{Role: "user", Content: json.RawMessage(`[
				{"type":"tool_result","tool_use_id":"toolu_1","content":"sunny"},
				{"type":"text","text":"summarize it"}
			]`)},
		},
		Tools:      []anthropicTool{{Name: "weather", Description: "weather lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		ToolChoice: &anthropicToolChoice{Type: "tool", Name: "weather"},
		Thinking:   &anthropicThinkingConfig{Type: "enabled", BudgetTokens: 1024},
	}

	got, err := anthropicToChatRequest(in)
	if err != nil {
		t.Fatal(err)
	}
	if got.Model != in.Model || got.MaxTokens != 128 || !got.Stream {
		t.Fatalf("basic translation = %+v", got)
	}
	if len(got.Messages) != 5 {
		t.Fatalf("messages=%d want 5: %+v", len(got.Messages), got.Messages)
	}
	if got.Messages[0].Role != "system" || got.Messages[0].Content != "be concise" {
		t.Errorf("system = %+v", got.Messages[0])
	}
	assistant := got.Messages[2]
	if assistant.ReasoningContent != "need a tool" || len(assistant.ToolCalls) != 1 {
		t.Errorf("assistant = %+v", assistant)
	}
	if got.Messages[3].Role != "tool" || got.Messages[3].Name != "weather" || got.Messages[3].Content != "sunny" {
		t.Errorf("tool result = %+v", got.Messages[3])
	}
	if got.Messages[4].Role != "user" || got.Messages[4].Content != "summarize it" {
		t.Errorf("following user text = %+v", got.Messages[4])
	}
	if len(got.Tools) != 1 || got.Tools[0].Function.Name != "weather" {
		t.Errorf("tools = %+v", got.Tools)
	}
	choice, ok := got.ToolChoice.(map[string]interface{})
	if !ok || choice["type"] != "function" {
		t.Errorf("tool choice = %#v", got.ToolChoice)
	}
	if got.Thinking == nil || !*got.Thinking {
		t.Errorf("thinking = %#v", got.Thinking)
	}
}

func TestAnthropicMessagesNonStream(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
	srv := New(f, tk)
	srv.SetModelName("local-gemma")
	srv.SetLogLevel("warn")

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", chatBody(t, map[string]interface{}{
		"model":       "client-alias",
		"max_tokens":  16,
		"messages":    []map[string]string{{"role": "user", "content": "hi"}},
		"temperature": 0,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var resp anthropicResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Type != "message" || resp.Role != "assistant" || !strings.HasPrefix(resp.ID, "msg_") {
		t.Errorf("response identity = %+v", resp)
	}
	if resp.Model != "local-gemma" || resp.StopReason != "end_turn" {
		t.Errorf("model/stop = %q/%q", resp.Model, resp.StopReason)
	}
	if len(resp.Content) != 1 || resp.Content[0]["type"] != "text" || resp.Content[0]["text"] != "hello world" {
		t.Errorf("content = %#v", resp.Content)
	}
	if resp.Usage.OutputTokens != 2 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if got := resp.Usage.InputTokens + resp.Usage.CacheReadInputTokens + resp.Usage.CacheCreationInputTokens; got == 0 {
		t.Errorf("logical input is zero: %+v", resp.Usage)
	}
}

func TestPromptAccountingToAnthropicUsage(t *testing.T) {
	usage := promptAccountingToAnthropicUsage(PromptAccounting{
		PromptTokens: 1024,
		CachedTokens: 768,
		Source:       CacheSourceDiskSession,
	}, 17, 128)
	if usage.InputTokens != 128 || usage.CacheReadInputTokens != 768 || usage.CacheCreationInputTokens != 128 {
		t.Fatalf("usage=%+v", usage)
	}
	if usage.InputTokens+usage.CacheReadInputTokens+usage.CacheCreationInputTokens != 1024 {
		t.Fatalf("logical input invariant failed: %+v", usage)
	}
}

func TestAnthropicMessagesStream(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: helloWorldScript(idx)}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", chatBody(t, map[string]interface{}{
		"model":       "local",
		"max_tokens":  16,
		"messages":    []map[string]string{{"role": "user", "content": "hi"}},
		"temperature": 0,
		"stream":      true,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("content type=%q", got)
	}
	events := parseAnthropicSSE(t, rec.Body.String())
	wantOrder := []string{"message_start", "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop", "message_delta", "message_stop"}
	if len(events) != len(wantOrder) {
		t.Fatalf("events=%v want=%v\n%s", eventNames(events), wantOrder, rec.Body.String())
	}
	for i, want := range wantOrder {
		if events[i].name != want {
			t.Errorf("event %d=%q want %q", i, events[i].name, want)
		}
	}
	var text strings.Builder
	for _, event := range events {
		if event.name != "content_block_delta" {
			continue
		}
		var value struct {
			Delta struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(event.data), &value); err != nil {
			t.Fatal(err)
		}
		if value.Delta.Type == "text_delta" {
			text.WriteString(value.Delta.Text)
		}
	}
	if strings.TrimSpace(text.String()) != "hello world" {
		t.Errorf("stream text=%q", text.String())
	}
}

func TestAnthropicStreamingToolUseTranslation(t *testing.T) {
	rec := httptest.NewRecorder()
	aw := newAnthropicResponseWriter(rec, true, 12)
	aw.Header().Set("Content-Type", "text/event-stream")
	aw.WriteHeader(http.StatusOK)
	chunks := []StreamResponse{
		{ID: "chatcmpl-abc", Model: "local", Choices: []StreamChoice{{Index: 0, Delta: Delta{Role: "assistant"}}}},
		{ID: "chatcmpl-abc", Model: "local", Choices: []StreamChoice{{Index: 0, Delta: Delta{ToolCalls: []DeltaToolCall{{
			Index: 0, ID: "call_weather", Type: "function",
			Function: ToolCallFunction{Name: "weather", Arguments: `{"city":"Paris"}`},
		}}}}}},
		{ID: "chatcmpl-abc", Model: "local", Choices: []StreamChoice{{Index: 0, Delta: Delta{}, FinishReason: "tool_calls"}}, Usage: &Usage{PromptTokens: 12, CompletionTokens: 7, TotalTokens: 19}},
	}
	for _, chunk := range chunks {
		data, err := json.Marshal(chunk)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := aw.Write([]byte("data: " + string(data) + "\n\n")); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := aw.Write([]byte("data: [DONE]\n\n")); err != nil {
		t.Fatal(err)
	}
	aw.finish()

	events := parseAnthropicSSE(t, rec.Body.String())
	want := []string{"message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"}
	if got := eventNames(events); strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("events=%v want=%v\n%s", got, want, rec.Body.String())
	}
	var start struct {
		ContentBlock struct {
			Type string `json:"type"`
			ID   string `json:"id"`
			Name string `json:"name"`
		} `json:"content_block"`
	}
	if err := json.Unmarshal([]byte(events[1].data), &start); err != nil {
		t.Fatal(err)
	}
	if start.ContentBlock.Type != "tool_use" || start.ContentBlock.ID != "call_weather" || start.ContentBlock.Name != "weather" {
		t.Errorf("tool start=%+v", start.ContentBlock)
	}
	if !strings.Contains(events[2].data, `"partial_json":"{\"city\":\"Paris\"}"`) {
		t.Errorf("tool delta=%s", events[2].data)
	}
	if !strings.Contains(events[4].data, `"stop_reason":"tool_use"`) || !strings.Contains(events[4].data, `"output_tokens":7`) {
		t.Errorf("message delta=%s", events[4].data)
	}
}

func TestAnthropicErrorsAndAPIKeyAuth(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	srv.SetAPIKey("secret")

	body := `{"model":"local","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}`
	for _, tc := range []struct {
		name string
		key  string
		want int
	}{
		{name: "missing", want: http.StatusUnauthorized},
		{name: "anthropic header", key: "secret", want: http.StatusOK},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
			if tc.key != "" {
				req.Header.Set("x-api-key", tc.key)
			}
			rec := httptest.NewRecorder()
			mux(srv).ServeHTTP(rec, req)
			if rec.Code != tc.want {
				t.Fatalf("status=%d want=%d body=%s", rec.Code, tc.want, rec.Body.String())
			}
			if tc.want != http.StatusOK {
				var e struct {
					Type  string `json:"type"`
					Error struct {
						Type string `json:"type"`
					} `json:"error"`
				}
				if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil {
					t.Fatal(err)
				}
				if e.Type != "error" || e.Error.Type != "authentication_error" {
					t.Errorf("error = %+v", e)
				}
			}
		})
	}
}

func TestAnthropicRequiresMaxTokens(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"local","messages":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "max_tokens") || !strings.Contains(rec.Body.String(), "invalid_request_error") {
		t.Errorf("body=%s", rec.Body.String())
	}
}

type namedSSE struct{ name, data string }

func parseAnthropicSSE(t *testing.T, body string) []namedSSE {
	t.Helper()
	var out []namedSSE
	var current namedSSE
	scanner := bufio.NewScanner(strings.NewReader(body))
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "event: "):
			current.name = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			current.data = strings.TrimPrefix(line, "data: ")
		case line == "" && current.name != "":
			out = append(out, current)
			current = namedSSE{}
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return out
}

func eventNames(events []namedSSE) []string {
	out := make([]string, len(events))
	for i := range events {
		out[i] = events[i].name
	}
	return out
}
