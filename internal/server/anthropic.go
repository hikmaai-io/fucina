package server

// Anthropic Messages API compatibility.
//
// The inference core speaks one internal, OpenAI-shaped request/response model.
// This file translates Anthropic /v1/messages requests into that model and
// translates the response back on the wire. Keeping the adapter at the HTTP
// boundary means the single-flight and continuous-batching generation paths
// remain identical for both APIs.

import (
	"bytes"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const anthropicBodyLimit = 64 << 20

type anthropicRequest struct {
	Model         string                   `json:"model"`
	Messages      []anthropicInputMessage  `json:"messages"`
	System        json.RawMessage          `json:"system,omitempty"`
	MaxTokens     int                      `json:"max_tokens"`
	Temperature   *float64                 `json:"temperature,omitempty"`
	TopP          *float64                 `json:"top_p,omitempty"`
	TopK          *int                     `json:"top_k,omitempty"`
	StopSequences []string                 `json:"stop_sequences,omitempty"`
	Stream        bool                     `json:"stream,omitempty"`
	Tools         []anthropicTool          `json:"tools,omitempty"`
	ToolChoice    *anthropicToolChoice     `json:"tool_choice,omitempty"`
	Thinking      *anthropicThinkingConfig `json:"thinking,omitempty"`
}

type anthropicInputMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type anthropicContentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type anthropicToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

type anthropicThinkingConfig struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

type anthropicResponse struct {
	ID           string                   `json:"id"`
	Type         string                   `json:"type"`
	Role         string                   `json:"role"`
	Content      []map[string]interface{} `json:"content"`
	Model        string                   `json:"model"`
	StopReason   string                   `json:"stop_reason"`
	StopSequence interface{}              `json:"stop_sequence"`
	Usage        anthropicUsage           `json:"usage"`
}

func (s *Server) requireAnthropicAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.apiKey == "" {
			next(w, r)
			return
		}
		xKey := r.Header.Get("x-api-key")
		bearer := ""
		const prefix = "Bearer "
		if h := r.Header.Get("Authorization"); strings.HasPrefix(h, prefix) {
			bearer = h[len(prefix):]
		}
		xOK := subtle.ConstantTimeCompare([]byte(xKey), []byte(s.apiKey))
		bearerOK := subtle.ConstantTimeCompare([]byte(bearer), []byte(s.apiKey))
		if xOK|bearerOK != 1 {
			writeAnthropicError(w, http.StatusUnauthorized, "authentication_error", "invalid or missing API key")
			return
		}
		next(w, r)
	}
}

func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeAnthropicError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method not allowed")
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, anthropicBodyLimit)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "request body is too large or unreadable")
		return
	}
	var in anthropicRequest
	if err := json.Unmarshal(body, &in); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON: "+err.Error())
		return
	}
	if strings.TrimSpace(in.Model) == "" {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "model must not be empty")
		return
	}
	if in.MaxTokens <= 0 {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "max_tokens must be greater than zero")
		return
	}
	if len(in.Messages) == 0 {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "messages must not be empty")
		return
	}

	chatReq, err := anthropicToChatRequest(in)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	translated, err := json.Marshal(chatReq)
	if err != nil {
		writeAnthropicError(w, http.StatusInternalServerError, "api_error", "could not translate request")
		return
	}

	// serveCompletions owns body limits, admission, prompt construction, sessions,
	// batching and generation. Give it a cloned request with the translated body.
	r2 := r.Clone(r.Context())
	r2.Body = io.NopCloser(bytes.NewReader(translated))
	r2.ContentLength = int64(len(translated))
	inputTokens := s.estimateAnthropicInputTokens(chatReq)
	aw := newAnthropicResponseWriter(w, in.Stream, inputTokens)
	s.serveCompletions(aw, r2, false)
	aw.finish()
}

func anthropicToChatRequest(in anthropicRequest) (ChatRequest, error) {
	out := ChatRequest{
		Model:       in.Model,
		MaxTokens:   in.MaxTokens,
		Temperature: in.Temperature,
		TopP:        in.TopP,
		TopK:        in.TopK,
		Stream:      in.Stream,
		Stop:        StopField(in.StopSequences),
	}

	if len(in.System) > 0 && string(in.System) != "null" {
		system, err := anthropicTextContent(in.System, false)
		if err != nil {
			return out, fmt.Errorf("system: %w", err)
		}
		if system != "" {
			out.Messages = append(out.Messages, ChatMessage{Role: "system", Content: system})
		}
	}

	// Anthropic tool_result blocks identify the preceding tool by id, while the
	// internal Gemma dialect also benefits from its name. Remember ids as the
	// conversation is translated so results can carry both where available.
	toolNames := make(map[string]string)
	for mi, msg := range in.Messages {
		if msg.Role != "user" && msg.Role != "assistant" {
			return out, fmt.Errorf("messages.%d.role must be user or assistant", mi)
		}
		var plain string
		if err := json.Unmarshal(msg.Content, &plain); err == nil {
			out.Messages = append(out.Messages, ChatMessage{Role: msg.Role, Content: plain})
			continue
		}
		var blocks []anthropicContentBlock
		if err := json.Unmarshal(msg.Content, &blocks); err != nil {
			return out, fmt.Errorf("messages.%d.content must be a string or content-block array", mi)
		}

		if msg.Role == "assistant" {
			cm := ChatMessage{Role: "assistant"}
			for bi, block := range blocks {
				switch block.Type {
				case "text":
					cm.Content += block.Text
				case "thinking":
					cm.ReasoningContent += block.Thinking
				case "redacted_thinking":
					// Redacted provider state cannot be replayed into a local model.
				case "tool_use":
					if block.Name == "" {
						return out, fmt.Errorf("messages.%d.content.%d tool_use is missing name", mi, bi)
					}
					args := strings.TrimSpace(string(block.Input))
					if args == "" || args == "null" {
						args = "{}"
					}
					id := block.ID
					if id == "" {
						id = fmt.Sprintf("toolu_history_%d_%d", mi, bi)
					}
					toolNames[id] = block.Name
					cm.ToolCalls = append(cm.ToolCalls, ToolCall{
						ID: id, Type: "function",
						Function: ToolCallFunction{Name: block.Name, Arguments: args},
					})
				default:
					return out, fmt.Errorf("messages.%d.content.%d has unsupported type %q", mi, bi, block.Type)
				}
			}
			out.Messages = append(out.Messages, cm)
			continue
		}

		// A user message can contain tool results and ordinary text. Preserve block
		// order as separate internal turns; consecutive tool turns are grouped by
		// the Qwen renderer and valid in the Gemma renderer as well.
		var userText strings.Builder
		flushUser := func() {
			if userText.Len() > 0 {
				out.Messages = append(out.Messages, ChatMessage{Role: "user", Content: userText.String()})
				userText.Reset()
			}
		}
		for bi, block := range blocks {
			switch block.Type {
			case "text":
				userText.WriteString(block.Text)
			case "tool_result":
				flushUser()
				result, err := anthropicTextContent(block.Content, true)
				if err != nil {
					return out, fmt.Errorf("messages.%d.content.%d tool_result: %w", mi, bi, err)
				}
				if block.IsError {
					result = "Error: " + result
				}
				out.Messages = append(out.Messages, ChatMessage{
					Role: "tool", Content: result, ToolCallID: block.ToolUseID,
					Name: toolNames[block.ToolUseID],
				})
			default:
				return out, fmt.Errorf("messages.%d.content.%d has unsupported type %q", mi, bi, block.Type)
			}
		}
		flushUser()
	}

	for i, tool := range in.Tools {
		if tool.Name == "" {
			return out, fmt.Errorf("tools.%d.name must not be empty", i)
		}
		schema := tool.InputSchema
		if len(schema) == 0 || string(schema) == "null" {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		out.Tools = append(out.Tools, Tool{Type: "function", Function: ToolFunction{
			Name: tool.Name, Description: tool.Description, Parameters: schema,
		}})
	}
	if in.ToolChoice != nil {
		switch in.ToolChoice.Type {
		case "", "auto":
			out.ToolChoice = "auto"
		case "none":
			out.ToolChoice = "none"
		case "any":
			out.ToolChoice = "required"
		case "tool":
			if in.ToolChoice.Name == "" {
				return out, fmt.Errorf("tool_choice.name is required when type is tool")
			}
			out.ToolChoice = map[string]interface{}{
				"type": "function", "function": map[string]interface{}{"name": in.ToolChoice.Name},
			}
		default:
			return out, fmt.Errorf("unsupported tool_choice.type %q", in.ToolChoice.Type)
		}
	}
	if in.Thinking != nil {
		switch in.Thinking.Type {
		case "enabled", "adaptive":
			out.Thinking = boolPtr(true)
		case "disabled":
			out.Thinking = boolPtr(false)
		default:
			return out, fmt.Errorf("unsupported thinking.type %q", in.Thinking.Type)
		}
	}
	return out, nil
}

// anthropicTextContent extracts either a string or text content blocks. Empty
// tool-result content is valid; system content rejects non-text blocks.
func anthropicTextContent(raw json.RawMessage, allowEmpty bool) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		if allowEmpty {
			return "", nil
		}
		return "", fmt.Errorf("content is empty")
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text, nil
	}
	var blocks []anthropicContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", fmt.Errorf("content must be a string or text-block array")
	}
	var sb strings.Builder
	for i, block := range blocks {
		if block.Type != "text" {
			return "", fmt.Errorf("content block %d has unsupported type %q", i, block.Type)
		}
		sb.WriteString(block.Text)
	}
	return sb.String(), nil
}

func (s *Server) estimateAnthropicInputTokens(req ChatRequest) int {
	thinking := s.thinkingDefault
	if t := req.resolveThinking(); t != nil {
		thinking = *t
	}
	wantTools := len(req.Tools) > 0 && !isToolChoiceNone(req.ToolChoice)
	prefix := ""
	if wantTools {
		if name, forced := forcedToolChoice(req.ToolChoice); forced {
			prefix = s.dialect.ForcedCallPrefix(name)
			if prefix != "" {
				thinking = false
			}
		}
	}
	tokens := s.tokenizer.Encode(s.renderChatTemplate(req.Messages, req.Tools, thinking)+prefix, true, false)
	ctx := int(s.engine.ContextSize())
	maxTokens := req.MaxTokens
	if cap := ctx / 2; maxTokens > cap {
		maxTokens = cap
	}
	if s.maxOutputTokens > 0 && maxTokens > s.maxOutputTokens {
		maxTokens = s.maxOutputTokens
	}
	if budget := ctx - maxTokens; budget > 0 && len(tokens) > budget {
		kept := tokens[len(tokens)-budget:]
		// Mirror serveCompletions' compaction rule: retaining the leading BOS may
		// make the final prompt one token larger than the nominal budget.
		if len(tokens) > 0 && tokens[0] == s.tokenizer.BOS &&
			(len(kept) == 0 || kept[0] != s.tokenizer.BOS) {
			return len(kept) + 1
		}
		return len(kept)
	}
	return len(tokens)
}

func anthropicStopReason(openAI string) string {
	switch openAI {
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	default:
		return "end_turn"
	}
}

func anthropicMessageID(id string) string {
	id = strings.TrimPrefix(id, "chatcmpl-")
	id = strings.TrimPrefix(id, "cmpl-")
	if id == "" {
		id = newRequestID()
	}
	return "msg_" + id
}

func chatToAnthropic(resp ChatResponse) anthropicResponse {
	out := anthropicResponse{
		ID: anthropicMessageID(resp.ID), Type: "message", Role: "assistant",
		Model: resp.Model, StopReason: "end_turn", StopSequence: nil,
		Usage:   anthropicUsage{InputTokens: resp.Usage.PromptTokens, OutputTokens: resp.Usage.CompletionTokens},
		Content: make([]map[string]interface{}, 0),
	}
	if len(resp.Choices) == 0 {
		return out
	}
	choice := resp.Choices[0]
	out.StopReason = anthropicStopReason(choice.FinishReason)
	if choice.Message.ReasoningContent != "" {
		out.Content = append(out.Content, map[string]interface{}{
			"type": "thinking", "thinking": choice.Message.ReasoningContent, "signature": "",
		})
	}
	if choice.Message.Content != "" {
		out.Content = append(out.Content, map[string]interface{}{"type": "text", "text": choice.Message.Content})
	}
	for _, call := range choice.Message.ToolCalls {
		var input interface{}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &input); err != nil {
			input = map[string]interface{}{}
		}
		out.Content = append(out.Content, map[string]interface{}{
			"type": "tool_use", "id": call.ID, "name": call.Function.Name, "input": input,
		})
	}
	return out
}

func writeAnthropicError(w http.ResponseWriter, status int, typ, message string) {
	writeJSON(w, status, map[string]interface{}{
		"type":  "error",
		"error": map[string]string{"type": typ, "message": message},
	})
}

func anthropicErrorType(status int) string {
	switch status {
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusRequestEntityTooLarge:
		return "request_too_large"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusServiceUnavailable:
		return "overloaded_error"
	default:
		if status >= 500 {
			return "api_error"
		}
		return "invalid_request_error"
	}
}

// anthropicResponseWriter adapts both collected JSON and incremental OpenAI SSE
// emitted by serveCompletions. It deliberately implements http.Flusher so the
// core keeps its immediate-header/heartbeat behavior.
type anthropicResponseWriter struct {
	w            http.ResponseWriter
	streamWanted bool
	streaming    bool
	status       int
	wroteHeader  bool
	body         bytes.Buffer
	sseBuffer    string
	inputTokens  int
	started      bool
	stopped      bool
	messageID    string
	model        string
	blockIndex   int
	blockType    string
	outputTokens int
}

func newAnthropicResponseWriter(w http.ResponseWriter, stream bool, inputTokens int) *anthropicResponseWriter {
	return &anthropicResponseWriter{w: w, streamWanted: stream, status: http.StatusOK, inputTokens: inputTokens}
}

func (w *anthropicResponseWriter) Header() http.Header { return w.w.Header() }

func (w *anthropicResponseWriter) WriteHeader(status int) {
	if w.wroteHeader {
		return
	}
	w.wroteHeader = true
	w.status = status
	w.streaming = w.streamWanted && status == http.StatusOK && strings.HasPrefix(w.Header().Get("Content-Type"), "text/event-stream")
	if w.streaming {
		w.w.WriteHeader(status)
	}
}

func (w *anthropicResponseWriter) Write(p []byte) (int, error) {
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	if !w.streaming {
		return w.body.Write(p)
	}
	w.sseBuffer += string(p)
	w.consumeSSE(false)
	return len(p), nil
}

func (w *anthropicResponseWriter) Flush() {
	if w.streaming {
		w.consumeSSE(false)
		if f, ok := w.w.(http.Flusher); ok {
			f.Flush()
		}
	}
}

// FlushError preserves the core SSE writer's stalled-client detection through
// this protocol adapter. ResponseController prefers this over the error-less
// Flush method and follows the outer middleware's Unwrap chain to the socket.
func (w *anthropicResponseWriter) FlushError() error {
	if w.streaming {
		w.consumeSSE(false)
	}
	return http.NewResponseController(w.w).Flush()
}

func (w *anthropicResponseWriter) Unwrap() http.ResponseWriter { return w.w }

func (w *anthropicResponseWriter) finish() {
	if w.streaming {
		w.consumeSSE(true)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if w.status < 200 || w.status >= 300 {
		message := strings.TrimSpace(w.body.String())
		var openAI struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(w.body.Bytes(), &openAI) == nil && openAI.Error.Message != "" {
			message = openAI.Error.Message
		}
		if message == "" {
			message = http.StatusText(w.status)
		}
		writeAnthropicError(w.w, w.status, anthropicErrorType(w.status), message)
		return
	}
	var resp ChatResponse
	if err := json.Unmarshal(w.body.Bytes(), &resp); err != nil {
		writeAnthropicError(w.w, http.StatusInternalServerError, "api_error", "invalid internal completion response")
		return
	}
	writeJSON(w.w, w.status, chatToAnthropic(resp))
}

func (w *anthropicResponseWriter) consumeSSE(final bool) {
	for {
		i := strings.Index(w.sseBuffer, "\n\n")
		if i < 0 {
			break
		}
		record := strings.TrimSuffix(w.sseBuffer[:i], "\r")
		w.sseBuffer = w.sseBuffer[i+2:]
		w.consumeSSERecord(record)
	}
	if final && strings.TrimSpace(w.sseBuffer) != "" {
		w.consumeSSERecord(strings.TrimSpace(w.sseBuffer))
		w.sseBuffer = ""
	}
}

func (w *anthropicResponseWriter) consumeSSERecord(record string) {
	if strings.HasPrefix(record, ":") {
		fmt.Fprint(w.w, record, "\n\n")
		return
	}
	var payload string
	for _, line := range strings.Split(record, "\n") {
		line = strings.TrimSuffix(line, "\r")
		if strings.HasPrefix(line, "data:") {
			payload += strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		}
	}
	if payload == "" || payload == "[DONE]" {
		return
	}
	var raw map[string]json.RawMessage
	if json.Unmarshal([]byte(payload), &raw) != nil {
		return
	}
	if errRaw, ok := raw["error"]; ok {
		var e struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		}
		_ = json.Unmarshal(errRaw, &e)
		if e.Message == "" {
			e.Message = "streaming generation failed"
		}
		w.emitAnthropicSSE("error", map[string]interface{}{
			"type": "error", "error": map[string]string{"type": "api_error", "message": e.Message},
		})
		w.stopped = true
		return
	}
	var chunk StreamResponse
	if json.Unmarshal([]byte(payload), &chunk) != nil || len(chunk.Choices) == 0 {
		return
	}
	if chunk.ID != "" {
		w.messageID = anthropicMessageID(chunk.ID)
	}
	if chunk.Model != "" {
		w.model = chunk.Model
	}
	w.ensureMessageStart()
	choice := chunk.Choices[0]
	if choice.Delta.ReasoningContent != "" {
		w.ensureBlock("thinking", "", "")
		w.emitAnthropicSSE("content_block_delta", map[string]interface{}{
			"type": "content_block_delta", "index": w.blockIndex,
			"delta": map[string]string{"type": "thinking_delta", "thinking": choice.Delta.ReasoningContent},
		})
	}
	if choice.Delta.Content != "" {
		w.ensureBlock("text", "", "")
		w.emitAnthropicSSE("content_block_delta", map[string]interface{}{
			"type": "content_block_delta", "index": w.blockIndex,
			"delta": map[string]string{"type": "text_delta", "text": choice.Delta.Content},
		})
	}
	for _, call := range choice.Delta.ToolCalls {
		w.closeBlock()
		w.ensureBlock("tool_use", call.ID, call.Function.Name)
		args := call.Function.Arguments
		if args == "" {
			args = "{}"
		}
		w.emitAnthropicSSE("content_block_delta", map[string]interface{}{
			"type": "content_block_delta", "index": w.blockIndex,
			"delta": map[string]string{"type": "input_json_delta", "partial_json": args},
		})
		w.closeBlock()
	}
	if chunk.Usage != nil {
		w.outputTokens = chunk.Usage.CompletionTokens
	}
	if choice.FinishReason != "" && !w.stopped {
		w.closeBlock()
		w.emitAnthropicSSE("message_delta", map[string]interface{}{
			"type":  "message_delta",
			"delta": map[string]interface{}{"stop_reason": anthropicStopReason(choice.FinishReason), "stop_sequence": nil},
			"usage": map[string]int{"output_tokens": w.outputTokens},
		})
		w.emitAnthropicSSE("message_stop", map[string]string{"type": "message_stop"})
		w.stopped = true
	}
}

func (w *anthropicResponseWriter) ensureMessageStart() {
	if w.started {
		return
	}
	w.started = true
	if w.messageID == "" {
		w.messageID = anthropicMessageID("")
	}
	w.emitAnthropicSSE("message_start", map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id": w.messageID, "type": "message", "role": "assistant", "content": []interface{}{},
			"model": w.model, "stop_reason": nil, "stop_sequence": nil,
			"usage": anthropicUsage{InputTokens: w.inputTokens, OutputTokens: 0},
		},
	})
}

func (w *anthropicResponseWriter) ensureBlock(typ, id, name string) {
	if w.blockType == typ && typ != "tool_use" {
		return
	}
	w.closeBlock()
	var block map[string]interface{}
	switch typ {
	case "thinking":
		block = map[string]interface{}{"type": "thinking", "thinking": "", "signature": ""}
	case "tool_use":
		block = map[string]interface{}{"type": "tool_use", "id": id, "name": name, "input": map[string]interface{}{}}
	default:
		block = map[string]interface{}{"type": "text", "text": ""}
	}
	w.blockType = typ
	w.emitAnthropicSSE("content_block_start", map[string]interface{}{
		"type": "content_block_start", "index": w.blockIndex, "content_block": block,
	})
}

func (w *anthropicResponseWriter) closeBlock() {
	if w.blockType == "" {
		return
	}
	w.emitAnthropicSSE("content_block_stop", map[string]interface{}{
		"type": "content_block_stop", "index": w.blockIndex,
	})
	w.blockType = ""
	w.blockIndex++
}

func (w *anthropicResponseWriter) emitAnthropicSSE(event string, value interface{}) {
	data, _ := json.Marshal(value)
	fmt.Fprintf(w.w, "event: %s\ndata: %s\n\n", event, data)
}
