package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/mauromedda/gem4d/internal/engine/cuda"
	"github.com/mauromedda/gem4d/internal/tokenizer"
)

// ─── Types ─────────────────────────────────────────────────────────

type Server struct {
	engine    *cuda.Engine
	tokenizer *tokenizer.Tokenizer
	kv              *KVCache
	modelName       string
	genParams       GenerationParams
	thinkingDefault bool // startup default for the gemma-4 reasoning channel
	debug           bool // dump full request bodies + rendered prompts
	metrics         Metrics
	httpServer      *http.Server
}

type GenerationParams struct {
	Temperature      float64 `json:"temperature"`
	TopP             float64 `json:"top_p"`
	TopK             int     `json:"top_k"`
	MinP             float64 `json:"min_p"`
	Seed             int64   `json:"seed"`
	RepeatPenalty    float64 `json:"repeat_penalty"`
	FrequencyPenalty float64 `json:"frequency_penalty"`
	PresencePenalty  float64 `json:"presence_penalty"`
	MaxTokens        int      `json:"max_tokens"`
	Stream           bool     `json:"stream"`
	Stop             []string `json:"stop,omitempty"`
}

type ChatMessage struct {
	Role             string     `json:"role"`
	Content          string     `json:"content"`
	ReasoningContent string     `json:"reasoning_content,omitempty"` // gemma-4 thought channel
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
	Name             string     `json:"name,omitempty"`
}

// contentPart is one element of the OpenAI "content parts" array form, e.g.
// {"type":"text","text":"hello"}. Only text parts contribute to the prompt;
// non-text parts (images, etc.) are ignored.
type contentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// UnmarshalJSON accepts `content` either as a plain string (legacy/OpenAI
// simple form) or as an array of typed parts (OpenAI vision/multipart form,
// which clients such as `pi` send). Array parts are flattened to text so the
// rest of the server can keep treating Content as a string.
func (m *ChatMessage) UnmarshalJSON(data []byte) error {
	// Alias avoids infinite recursion into this method.
	type alias ChatMessage
	aux := &struct {
		Content json.RawMessage `json:"content"`
		*alias
	}{alias: (*alias)(m)}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	if len(aux.Content) == 0 || string(aux.Content) == "null" {
		m.Content = ""
		return nil
	}
	// Try plain string first.
	var s string
	if err := json.Unmarshal(aux.Content, &s); err == nil {
		m.Content = s
		return nil
	}
	// Fall back to the array-of-parts form.
	var parts []contentPart
	if err := json.Unmarshal(aux.Content, &parts); err != nil {
		return fmt.Errorf("content must be a string or an array of content parts: %w", err)
	}
	var sb strings.Builder
	for _, p := range parts {
		if p.Type == "" || p.Type == "text" {
			sb.WriteString(p.Text)
		}
	}
	m.Content = sb.String()
	return nil
}

type ChatRequest struct {
	Model       string          `json:"model"`
	Messages    []ChatMessage   `json:"messages"`
	Prompt      json.RawMessage `json:"prompt,omitempty"` // legacy /v1/completions
	MaxTokens   int             `json:"max_tokens"`
	Temperature *float64        `json:"temperature"`
	TopP        *float64        `json:"top_p"`
	TopK        *int            `json:"top_k"`
	MinP        *float64        `json:"min_p"`
	Seed        *int64          `json:"seed"`
	Stream      bool            `json:"stream"`
	Stop        StopField       `json:"stop,omitempty"`
	Tools       []Tool          `json:"tools,omitempty"`
	ToolChoice  interface{}     `json:"tool_choice,omitempty"`

	// Thinking / reasoning control. gemma-4 gates a reasoning channel: when
	// enabled the model emits a <|channel>thought…<channel|> block before its
	// answer; when disabled the template pre-closes an empty thought channel so
	// the model answers directly. Accepted forms (first non-empty wins):
	//   reasoning_effort: OpenAI standard string — "none"/"minimal" → off,
	//                     "low"/"medium"/"high" → on.
	//   thinking / enable_thinking: explicit bool.
	//   chat_template_kwargs.enable_thinking: bool (HF/vLLM convention, what pi sends).
	ReasoningEffort  string          `json:"reasoning_effort,omitempty"`
	Thinking         *bool           `json:"thinking,omitempty"`
	EnableThinking   *bool           `json:"enable_thinking,omitempty"`
	ChatTemplateArgs json.RawMessage `json:"chat_template_kwargs,omitempty"`
}

// resolveThinking decides whether the gemma-4 reasoning channel is enabled for
// THIS request. Precedence: explicit bools > chat_template_kwargs.enable_thinking
// > reasoning_effort string. Returns nil when the request says nothing about
// thinking, so the caller falls back to the server's startup default.
//
// gemma-4's native reasoning control is binary (enable_thinking on/off), so the
// graded effort levels collapse to on/off: "none"/"minimal"/"off" → off, and
// "low"/"medium"/"high"/"xhigh"/"max"/"on" → on.
func (r *ChatRequest) resolveThinking() *bool {
	if r.Thinking != nil {
		return r.Thinking
	}
	if r.EnableThinking != nil {
		return r.EnableThinking
	}
	if len(r.ChatTemplateArgs) > 0 {
		var kw struct {
			EnableThinking *bool `json:"enable_thinking"`
		}
		if json.Unmarshal(r.ChatTemplateArgs, &kw) == nil && kw.EnableThinking != nil {
			return kw.EnableThinking
		}
	}
	switch strings.ToLower(strings.TrimSpace(r.ReasoningEffort)) {
	case "none", "minimal", "off", "false", "no":
		return boolPtr(false)
	case "low", "medium", "mid", "high", "xhigh", "max", "on", "true", "yes":
		return boolPtr(true)
	}
	return nil // unspecified → use the server default
}

func boolPtr(b bool) *bool { return &b }

// ParseThinkingLevel maps a startup/CLI thinking level to the binary gemma-4
// enable_thinking flag. Accepts off/on plus the graded aliases (which collapse to
// on, since the model's native control is binary). Unknown → off.
func ParseThinkingLevel(level string) bool {
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "", "none", "minimal", "off", "false", "no", "0":
		return false
	case "low", "medium", "mid", "high", "xhigh", "max", "on", "true", "yes", "1":
		return true
	}
	return false
}

// StopField accepts the OpenAI `stop` parameter in either form: a single string
// or an array of strings. Both decode to []string.
type StopField []string

func (s *StopField) UnmarshalJSON(data []byte) error {
	if len(data) == 0 || string(data) == "null" {
		return nil
	}
	var one string
	if err := json.Unmarshal(data, &one); err == nil {
		*s = []string{one}
		return nil
	}
	var many []string
	if err := json.Unmarshal(data, &many); err != nil {
		return fmt.Errorf("stop must be a string or array of strings: %w", err)
	}
	*s = many
	return nil
}

// legacyPrompt extracts the prompt text from the /v1/completions `prompt`
// field, which may be a string or an array of strings (joined with newlines).
func (r *ChatRequest) legacyPrompt() string {
	if len(r.Prompt) == 0 {
		return ""
	}
	var one string
	if json.Unmarshal(r.Prompt, &one) == nil {
		return one
	}
	var many []string
	if json.Unmarshal(r.Prompt, &many) == nil {
		return strings.Join(many, "\n")
	}
	return ""
}

type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type Choice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type Delta struct {
	Role             string          `json:"role,omitempty"`
	Content          string          `json:"content,omitempty"`
	ReasoningContent string          `json:"reasoning_content,omitempty"` // gemma-4 thought channel
	ToolCalls        []DeltaToolCall `json:"tool_calls,omitempty"`
}

// DeltaToolCall is the streaming form of a tool call. It carries the array
// `index` that OpenAI clients use to reassemble streamed tool calls (we emit
// each call whole in a single chunk, so name+arguments arrive together).
type DeltaToolCall struct {
	Index    int              `json:"index"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function ToolCallFunction `json:"function"`
}

type StreamChoice struct {
	Index        int    `json:"index"`
	Delta        Delta  `json:"delta"`
	FinishReason string `json:"finish_reason,omitempty"`
}

type StreamResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
	Usage   *Usage         `json:"usage,omitempty"` // sent on the final chunk for context tracking
}

type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`         // unix seconds (OpenAI clients type this as int)
	OwnedBy string `json:"owned_by"`
}

func DefaultParams() GenerationParams {
	// google/gemma-4-12B model-card standardized sampling config (also embedded in
	// the GGUF general.sampling.*): temperature 1.0, top_p 0.95, top_k 64, no min-p
	// or repeat penalty.
	return GenerationParams{
		Temperature:     1.0,
		TopP:            0.95,
		TopK:            64,
		MinP:            0.0,
		Seed:            0,
		RepeatPenalty:   1.0,
		FrequencyPenalty: 0.0,
		PresencePenalty: 0.0,
		// Generous completion cap for when a client omits max_tokens (pi does): the
		// model stops at EOS/end-of-turn on its own; 512 truncated agent turns
		// (reasoning + answer) mid-output. This is only a safety bound on runaways.
		MaxTokens: 8192,
		Stream:    false,
	}
}

func New(eng *cuda.Engine, tok *tokenizer.Tokenizer) *Server {
	return &Server{
		engine:    eng,
		tokenizer: tok,
		kv:        NewKVCache(eng),
		modelName: "gemma-4-12b-it",
		genParams: DefaultParams(),
	}
}

// SetModelName overrides the id reported by /v1/models and echoed in responses.
// Callers pass a quantization-aware id (e.g. derived from the GGUF filename) so
// clients can tell which build they are talking to.
func (s *Server) SetModelName(name string) {
	if name != "" {
		s.modelName = name
	}
}

// SetThinkingDefault sets the startup default for the gemma-4 reasoning channel.
// Per-request reasoning_effort / thinking / enable_thinking overrides it.
func (s *Server) SetThinkingDefault(on bool) { s.thinkingDefault = on }

// SetDebug enables verbose request logging: each chat request's full body and the
// rendered gemma-4 prompt are appended to debugDumpPath (and the per-request
// summary is logged). Use to inspect exactly what a client like pi sends.
func (s *Server) SetDebug(on bool) { s.debug = on }

// debugDumpPath is where SetDebug(true) appends request/prompt dumps.
const debugDumpPath = "/tmp/gem4d_debug.log"

func (s *Server) RegisterRoutes(mux *http.ServeMux) {
	h := func(f http.HandlerFunc) http.HandlerFunc { return logRequest(f) }
	mux.HandleFunc("/v1/models", h(s.handleModels))
	mux.HandleFunc("/v1/chat/completions", h(s.handleChatCompletions))
	mux.HandleFunc("/v1/completions", h(s.handleCompletions))
	mux.HandleFunc("/v1/embeddings", h(s.handleEmbeddings))
	mux.HandleFunc("/health", h(s.handleHealth))
	mux.HandleFunc("/metrics", h(s.handleMetrics))
	mux.HandleFunc("/", h(s.handleNotFound))
}

// handleMetrics reports live KV/context utilization, prefix-cache hit rate, and
// prefill/decode throughput (cumulative + last request). JSON for easy curl/pi use.
func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	ctxCap := int(s.engine.ContextSize())
	s.kv.Lock()
	used := s.engine.NTokens()
	s.kv.Unlock()
	hits, misses, reused, reqTok := s.kv.DetailedStats()

	// gemma-4 KV memory (FP8, 1 byte/elem; K and V stored separately):
	//   sliding: MAX_LAYERS(48) × KV_HEADS(8) × WINDOW(1024) × HEAD_DIM(256), fixed
	//   global:  GLOBAL_LAYERS(8) × ctxCap × GLOBAL_HEAD_DIM(512), grows with ctx
	const mib = 1024.0 * 1024.0
	slidingMB := float64(48*8*1024*256) * 2 / mib
	globalMB := float64(8*512) * float64(ctxCap) * 2 / mib

	writeJSON(w, http.StatusOK, s.metrics.snapshot(
		s.modelName, used, ctxCap, slidingMB, globalMB, hits, misses, reused, reqTok))
}

// statusRecorder captures the response status so the access log can report it.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) { r.status = code; r.ResponseWriter.WriteHeader(code) }

// Flush/Hijack are needed because the streaming handler type-asserts http.Flusher.
func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// logRequest logs each HTTP request's method, path, status, and duration so that
// client failures (4xx/5xx from clients like pi) are visible — the handlers return
// errors via http.Error without logging, which otherwise leaves silent failures.
func logRequest(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		rec := &statusRecorder{ResponseWriter: w, status: 200}
		start := time.Now()
		next(rec, r)
		log.Printf("gem4d: %s %s -> %d (%.0fms)",
			r.Method, r.URL.Path, rec.status, float64(time.Since(start).Microseconds())/1000.0)
	}
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, ModelsResponse{
		Object: "list",
		Data: []ModelInfo{{
			ID: s.modelName, Object: "model", Created: time.Now().Unix(), OwnedBy: "google",
		}},
	})
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	var req ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, fmt.Sprintf("invalid JSON: %v", err), http.StatusBadRequest)
		return
	}

	// Build the prompt. /v1/completions sends a raw `prompt`; /v1/chat/completions
	// sends `messages` that we render into the gemma-4 chat template.
	wantTools := len(req.Tools) > 0 && !isToolChoiceNone(req.ToolChoice)
	// Thinking: per-request override (reasoning_effort/thinking/enable_thinking)
	// falls back to the server's startup default when the request is silent.
	enableThinking := s.thinkingDefault
	if t := req.resolveThinking(); t != nil {
		enableThinking = *t
	}
	var prompt string
	if legacy := req.legacyPrompt(); legacy != "" {
		prompt = legacy
		wantTools = false // raw-completions mode never emits structured tool calls
	} else {
		prompt = s.renderChatTemplate(req.Messages, req.Tools, enableThinking)
	}
	if prompt == "" {
		http.Error(w, "empty prompt", http.StatusBadRequest)
		return
	}
	tokens := s.tokenizer.Encode(prompt, true, false)
	if len(tokens) == 0 {
		http.Error(w, "tokenization failed", http.StatusInternalServerError)
		return
	}

	// Per-request summary so the client's real footprint is visible (system-prompt
	// size, tool count, thinking, streaming). GEM4D_DEBUG=1 also dumps the full
	// request body + rendered prompt to /tmp for inspecting exactly what a client
	// (e.g. pi) sends.
	sysChars := 0
	if len(req.Messages) > 0 && req.Messages[0].Role == "system" {
		sysChars = len(req.Messages[0].Content)
	}
	log.Printf("gem4d: chat: %d msgs, %d tools, sys=%dch, %d prompt-tok, thinking=%v, stream=%v",
		len(req.Messages), len(req.Tools), sysChars, len(tokens), enableThinking, req.Stream)
	if s.debug || os.Getenv("GEM4D_DEBUG") == "1" {
		dump := fmt.Sprintf("\n========== %s  %d msgs / %d tools / %d tok / thinking=%v / stream=%v ==========\n"+
			"--- REQUEST BODY ---\n%s\n--- RENDERED PROMPT ---\n%s\n",
			time.Now().Format("15:04:05"), len(req.Messages), len(req.Tools), len(tokens),
			enableThinking, req.Stream, string(body), prompt)
		if f, err := os.OpenFile(debugDumpPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644); err == nil {
			f.WriteString(dump)
			f.Close()
		}
	}

	params := s.genParams
	if req.Temperature != nil {
		params.Temperature = *req.Temperature
	}
	if req.TopP != nil {
		params.TopP = *req.TopP
	}
	if req.TopK != nil {
		params.TopK = *req.TopK
	}
	if req.MinP != nil {
		params.MinP = *req.MinP
	}
	if req.Seed != nil {
		params.Seed = *req.Seed
	}
	ctx := int(s.engine.ContextSize())
	if req.MaxTokens > 0 {
		params.MaxTokens = req.MaxTokens // explicit client cap
	} else {
		// No client cap (pi omits max_tokens): generate until EOS or the context
		// fills — matching llama.cpp (n_predict = -1, "no limit") and vLLM (defaults
		// max_tokens to max_model_len - prompt_tokens). Cap to the space left after
		// the prompt; the model stops at end-of-turn on its own well before this.
		params.MaxTokens = ctx - len(tokens)
		if params.MaxTokens < 1 {
			params.MaxTokens = 1
		}
	}
	params.Stream = req.Stream
	params.Stop = req.Stop

	// Context budgeting ("compaction"): the engine holds a fixed KV window
	// (ContextSize). Guarantee room for the prompt AND the requested completion
	// by trimming the OLDEST prompt tokens when the two together would overflow.
	// We keep the most recent tokens (the live turn) which is what matters for a
	// coding agent; a leading BOS is preserved so the sequence stays well-formed.
	if budget := ctx - params.MaxTokens; budget > 0 && len(tokens) > budget {
		dropped := len(tokens) - budget
		kept := tokens[dropped:]
		if len(tokens) > 0 && tokens[0] == s.tokenizer.BOS &&
			(len(kept) == 0 || kept[0] != s.tokenizer.BOS) {
			kept = append([]int32{s.tokenizer.BOS}, kept...)
		}
		log.Printf("gem4d: context compaction: prompt %d + max_tokens %d > ctx %d; dropped %d oldest tokens",
			len(tokens), params.MaxTokens, ctx, dropped)
		tokens = kept
	}

	// Acquire the single physical KV cache for the whole request (prefill +
	// generation). Released inside the response helpers.
	s.kv.Lock()

	// Cache-aware prefill: reuse the longest cached prefix and compute only the
	// divergent suffix. The returned logits are for the final prompt token, so
	// no phantom Decode(0) is needed.
	prefillStart := time.Now()
	pf, err := s.kv.Prefill(tokens)
	if err != nil {
		s.kv.Unlock()
		http.Error(w, fmt.Sprintf("prefill failed: %v", err), http.StatusInternalServerError)
		return
	}
	prefillElapsed := time.Since(prefillStart)
	promptTokens := pf.PromptTokens
	prefillTPS := 0.0
	if prefillElapsed.Seconds() > 0 && pf.NewTokens > 0 {
		prefillTPS = float64(pf.NewTokens) / prefillElapsed.Seconds()
	}
	s.metrics.recordPrefill(pf.NewTokens, prefillElapsed.Seconds())
	used := s.engine.NTokens()
	log.Printf("gem4d: prefill %d tokens (%d cached, %d new) in %.2fs (%.1f tok/s) | ctx %d/%d (%.0f%%)",
		promptTokens, pf.ReusedTokens, pf.NewTokens, prefillElapsed.Seconds(), prefillTPS,
		used, ctx, 100.0*float64(used)/float64(ctx))

	logits := pf.Logits

	if params.Stream {
		s.streamResponse(r.Context(), w, params, promptTokens, logits, wantTools)
	} else {
		s.generateResponse(r.Context(), w, params, promptTokens, logits, wantTools)
	}
}

func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	s.handleChatCompletions(w, r)
}

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"object": "list",
		"data":   []interface{}{},
		"model":  s.modelName,
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	hits, misses, hitRate := s.kv.Stats()
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status": "ok",
		"kv_cache": map[string]interface{}{
			"prefix_hits":     hits,
			"prefix_misses":   misses,
			"token_hit_rate":  hitRate,
			"cached_tokens":   s.engine.NTokens(),
			"context_size":    s.engine.ContextSize(),
		},
	})
}

func (s *Server) handleNotFound(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
}

// modelTurnOpenNoThink opens an assistant turn and pre-fills an empty thought
// channel (thinking OFF), so the model skips reasoning and answers directly.
// modelTurnOpenThink opens the turn WITHOUT closing the thought channel, so the
// model produces its own <|channel>thought…<channel|> reasoning then the answer.
const modelTurnOpenNoThink = "<|turn>model\n<|channel>thought\n<channel|>"
const modelTurnOpenThink = "<|turn>model\n"

// renderChatTemplate builds the gemma-4 prompt. The real vocab uses <|turn> /
// <turn|> delimiters (NOT <start_of_turn>/<end_of_turn>, which are not tokens).
// When tools are present they are declared in the (forced) system turn as
// <|tool>…<tool|> blocks; role:tool messages render as <|tool_response>… blocks.
// enableThinking gates the gemma-4 reasoning channel (see ChatRequest.resolveThinking).
func (s *Server) renderChatTemplate(messages []ChatMessage, tools []Tool, enableThinking bool) string {
	var sb strings.Builder
	modelTurnOpen := modelTurnOpenNoThink
	if enableThinking {
		modelTurnOpen = modelTurnOpenThink
	}

	// System turn: merge an explicit system message (if first) with tool decls.
	// When thinking is on, the gemma-4 template injects a <|think|> marker at the
	// very top of the (forced) system turn.
	sysContent := ""
	start := 0
	if len(messages) > 0 && messages[0].Role == "system" {
		sysContent = messages[0].Content
		start = 1
	}
	if sysContent != "" || len(tools) > 0 || enableThinking {
		sb.WriteString("<|turn>system\n")
		if enableThinking {
			sb.WriteString("<|think|>\n")
		}
		sb.WriteString(sysContent)
		if len(tools) > 0 {
			sb.WriteString(renderToolDeclarations(tools))
		}
		sb.WriteString("<turn|>\n")
	}

	for i := start; i < len(messages); i++ {
		msg := messages[i]
		switch msg.Role {
		case "system":
			fmt.Fprintf(&sb, "<|turn>system\n%s<turn|>\n", msg.Content)
		case "user":
			fmt.Fprintf(&sb, "<|turn>user\n%s<turn|>\n", msg.Content)
		case "tool":
			sb.WriteString(renderToolResponse(msg.Name, msg.Content))
		case "assistant":
			if i == len(messages)-1 && len(msg.ToolCalls) == 0 && msg.Content == "" {
				sb.WriteString(modelTurnOpen)
			} else {
				sb.WriteString("<|turn>model\n")
				sb.WriteString(msg.Content)
				if len(msg.ToolCalls) > 0 {
					sb.WriteString(renderAssistantToolCalls(msg.ToolCalls))
				}
				sb.WriteString("<turn|>\n")
			}
		}
	}
	if len(messages) > 0 && messages[len(messages)-1].Role != "assistant" {
		sb.WriteString(modelTurnOpen)
	}
	return sb.String()
}

// generateResponse runs non-streaming generation. It assumes the caller holds
// s.kv.Lock() (acquired in the handler); it releases it when generation ends.
func (s *Server) generateResponse(ctx context.Context, w http.ResponseWriter, params GenerationParams, promptTokens int, logits []float32, wantTools bool) {
	defer s.kv.Unlock()

	generated := 0
	var toks []int32
	tcEnd := s.tokenizer.ToolCallEnd
	genStart := time.Now()
	finish := "stop"

	// The speculative fast-path is a single blocking engine call: it cannot check
	// text stop-strings or honor mid-flight cancellation. Use it only when neither
	// is needed; otherwise fall back to the per-token CPU loop.
	useSpec := params.RepeatPenalty == 1.0 && len(params.Stop) == 0

	if useSpec {
		// Speculative decoding (default): one weight pass per [g, draft...], same
		// output distribution as plain decode. Continues from the prefilled state.
		stops := []int32{s.tokenizer.EOS, s.tokenizer.EndOfTurn}
		if wantTools && tcEnd >= 0 {
			stops = append(stops, tcEnd)
		}
		seed := uint64(params.Seed)
		if params.Seed < 0 {
			seed = uint64(time.Now().UnixNano())
		}
		history := s.kv.CurrentTokens()
		var err error
		toks, _, err = s.engine.GenerateSpecContinue(history, logits, params.MaxTokens,
			stops, 6, float32(params.Temperature), params.TopK,
			float32(params.TopP), float32(params.MinP), seed)
		if err != nil {
			http.Error(w, fmt.Sprintf("generation failed: %v", err), http.StatusInternalServerError)
			return
		}
		// Sync the prefix cache with the tokens actually committed to the engine
		// KV (a trailing stop token may be emitted but not forwarded).
		committed := s.engine.NTokens() - promptTokens
		for i := 0; i < committed && i < len(toks); i++ {
			s.kv.AppendDecoded(toks[i])
		}
		generated = len(toks)
	} else {
		// Per-token decode loop with the CPU sampler: required for repeat-penalty,
		// text stop-sequences, and cancellation ("steering").
		rng := rand.New(rand.NewSource(params.Seed))
		for generated < params.MaxTokens {
			if logits == nil {
				break
			}
			if ctx.Err() != nil { // client aborted / steered away
				finish = "cancelled"
				break
			}
			token, err := s.sampleToken(logits, params, rng)
			if err != nil || s.tokenizer.IsStop(token) {
				break
			}
			toks = append(toks, token)
			s.kv.AppendDecoded(token)
			generated++
			if wantTools && tcEnd >= 0 && token == tcEnd {
				break
			}
			if hit, trimmed := stopHit(s.tokenizer.Decode(toks), params.Stop); hit {
				toks = s.tokenizer.Encode(trimmed, false, false)
				break
			}
			var err2 error
			logits, err2 = s.engine.Decode(token)
			if err2 != nil {
				break
			}
		}
	}
	s.logGenSpeed(genStart, generated)

	msg := ChatMessage{Role: "assistant"}
	// Separate the gemma-4 thought channel into reasoning_content; the rest is the
	// answer (and any tool-call markers). Done for BOTH paths so reasoning never
	// leaks into content when thinking is enabled.
	reasoning, rest := splitReasoning(s.tokenizer.DecodeRaw(toks))
	msg.ReasoningContent = reasoning
	if wantTools {
		content, calls := parseToolCalls(rest)
		msg.Content = content
		if len(calls) > 0 {
			msg.ToolCalls = calls
			finish = "tool_calls"
		}
	} else {
		// rest still carries control markers as literal strings (DecodeRaw); strip
		// them to plain text the way Decode would.
		msg.Content = strings.TrimSpace(stripMarkers(rest))
	}

	writeJSON(w, http.StatusOK, ChatResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   s.modelName,
		Choices: []Choice{{
			Index: 0, Message: msg, FinishReason: finish,
		}},
		Usage: Usage{
			PromptTokens: promptTokens, CompletionTokens: generated,
			TotalTokens: promptTokens + generated,
		},
	})
}

func (s *Server) streamResponse(ctx context.Context, w http.ResponseWriter, params GenerationParams, promptTokens int, logits []float32, wantTools bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	defer s.kv.Unlock()

	rng := rand.New(rand.NewSource(params.Seed))
	generated := 0
	completionID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()
	genStart := time.Now()

	// Role delta
	writeSSE(w, StreamResponse{
		ID: completionID, Object: "chat.completion.chunk", Created: created, Model: s.modelName,
		Choices: []StreamChoice{{Index: 0, Delta: Delta{Role: "assistant"}}},
	})
	flusher.Flush()

	tcOpen := s.tokenizer.ToolCallOpen
	tcEnd := s.tokenizer.ToolCallEnd
	chOpen := s.tokenizer.ChannelOpen
	chEnd := s.tokenizer.ChannelEnd
	finish := "stop"
	var emitted strings.Builder // accumulated visible text, for stop-string detection
	var rawToks []int32         // ALL generated ids (markers included) for tool parsing
	inTool := false             // currently inside a <|tool_call> … <tool_call|> span
	inChannel := false          // currently inside a <|channel> … <channel|> reasoning span
	channelLabel := false       // still skipping the "thought" label at a channel's start

	// GPU-side sampling (4-byte id back, no 262k logits copy).  Used whenever repeat
	// penalty is off (the GPU sampler cannot apply it).  The device sampler handles
	// temp=0 (argmax) and temp>0 (temperature / top-k / top-p / min-p / multinomial)
	// correctly — the binary-search threshold bug was fixed (sHi instead of sLo).
	gpuSample := params.RepeatPenalty == 1.0
	temp := float32(params.Temperature)
	for generated < params.MaxTokens {
		// Honor client cancellation ("steering"): if pi aborts the request, stop
		// generating immediately and release the KV lock for the next turn.
		if ctx.Err() != nil {
			finish = "cancelled"
			break
		}
		var token int32
		var err error
		if gpuSample {
			token, err = s.engine.SampleDevice(temp, params.TopK,
				float32(params.TopP), float32(params.MinP), float32(rng.Float64()))
		} else {
			if logits == nil {
				break
			}
			token, err = s.sampleToken(logits, params, rng)
		}
		if err != nil || s.tokenizer.IsStop(token) {
			break
		}
		rawToks = append(rawToks, token)
		s.kv.AppendDecoded(token)
		generated++

		// advance steps to the next token (GPU: leave logits on device; CPU: copy).
		advance := func() error {
			if gpuSample {
				return s.engine.DecodeNoCopy(token)
			}
			logits, err = s.engine.Decode(token)
			return err
		}

		// Tool-call handling: when the model opens a call, stop streaming content
		// and buffer until the call closes — we emit it as a structured tool_calls
		// delta after the loop. Mirrors the non-streaming spec path (one call/turn).
		if wantTools && tcOpen >= 0 && token == tcOpen {
			inTool = true
		}
		if inTool {
			if tcEnd >= 0 && token == tcEnd {
				break // complete tool call captured
			}
			if err = advance(); err != nil {
				break
			}
			continue
		}

		// Reasoning channel: stream <|channel>thought … <channel|> as reasoning_content
		// (not content), so the client can show/hide the thinking block separately.
		if chOpen >= 0 && token == chOpen {
			inChannel = true
			channelLabel = true // next text token is the "thought" label — skip it
			if err = advance(); err != nil {
				break
			}
			continue
		}
		if inChannel {
			if chEnd >= 0 && token == chEnd {
				inChannel = false
				if err = advance(); err != nil {
					break
				}
				continue
			}
			rtext := s.tokenizer.Decode([]int32{token})
			if channelLabel { // drop the leading "thought\n" channel label
				if i := strings.IndexByte(rtext, '\n'); i >= 0 {
					rtext = rtext[i+1:]
					channelLabel = false
				} else {
					rtext = "" // still inside the label line
				}
			}
			if rtext != "" {
				writeSSE(w, StreamResponse{
					ID: completionID, Object: "chat.completion.chunk", Created: created, Model: s.modelName,
					Choices: []StreamChoice{{Index: 0, Delta: Delta{ReasoningContent: rtext}}},
				})
				flusher.Flush()
			}
			if err = advance(); err != nil {
				break
			}
			continue
		}

		// Never stream the gemma tool/string markers to the client as visible text.
		if s.tokenizer.IsToolMarker(token) {
			if err = advance(); err != nil {
				break
			}
			continue
		}

		tokenStr := s.tokenizer.Decode([]int32{token})

		// Stop-sequence handling: if appending this piece completes a stop string,
		// emit only the text up to the stop and finish.
		if len(params.Stop) > 0 {
			if hit, trimmed := stopHit(emitted.String()+tokenStr, params.Stop); hit {
				if tail := strings.TrimPrefix(trimmed, emitted.String()); tail != "" {
					writeSSE(w, StreamResponse{
						ID: completionID, Object: "chat.completion.chunk", Created: created, Model: s.modelName,
						Choices: []StreamChoice{{Index: 0, Delta: Delta{Content: tail}}},
					})
					flusher.Flush()
				}
				break
			}
		}

		if tokenStr != "" {
			emitted.WriteString(tokenStr)
			writeSSE(w, StreamResponse{
				ID: completionID, Object: "chat.completion.chunk", Created: created, Model: s.modelName,
				Choices: []StreamChoice{{Index: 0, Delta: Delta{Content: tokenStr}}},
			})
			flusher.Flush()
		}

		if err = advance(); err != nil {
			break
		}
	}

	// Emit any captured tool call(s) as a structured delta before the final chunk.
	if wantTools {
		if _, calls := parseToolCalls(s.tokenizer.DecodeRaw(rawToks)); len(calls) > 0 {
			deltas := make([]DeltaToolCall, len(calls))
			for i, c := range calls {
				deltas[i] = DeltaToolCall{Index: i, ID: c.ID, Type: c.Type, Function: c.Function}
			}
			writeSSE(w, StreamResponse{
				ID: completionID, Object: "chat.completion.chunk", Created: created, Model: s.modelName,
				Choices: []StreamChoice{{Index: 0, Delta: Delta{ToolCalls: deltas}}},
			})
			flusher.Flush()
			finish = "tool_calls"
		}
	}

	s.logGenSpeed(genStart, generated)

	// Final chunk carries finish_reason AND usage so the client can track context
	// consumption (prompt + completion tokens) on streamed responses, matching the
	// non-streaming path. OpenAI sends usage on the terminal chunk.
	writeSSE(w, StreamResponse{
		ID: completionID, Object: "chat.completion.chunk", Created: created, Model: s.modelName,
		Choices: []StreamChoice{{Index: 0, Delta: Delta{}, FinishReason: finish}},
		Usage: &Usage{
			PromptTokens:     promptTokens,
			CompletionTokens: generated,
			TotalTokens:      promptTokens + generated,
		},
	})
	writeDONE(w)
	flusher.Flush()
}

// logGenSpeed logs generation throughput in tokens/second and records it for /metrics.
func (s *Server) logGenSpeed(start time.Time, generated int) {
	elapsed := time.Since(start)
	tps := 0.0
	if elapsed.Seconds() > 0 {
		tps = float64(generated) / elapsed.Seconds()
	}
	s.metrics.recordDecode(generated, elapsed.Seconds())
	log.Printf("gem4d: generated %d tokens in %.2fs (%.1f tok/s)",
		generated, elapsed.Seconds(), tps)
}

// sampleToken applies repeat-penalty → temperature → top-k → top-p → min-p →
// multinomial. Temperature 0 is greedy argmax. The previous implementation used
// two O(n²) bubble sorts over the full 262144-token vocab per token (~7e10 ops);
// this sorts once with sort.Slice (O(n log n)) and truncates.
// topKIndices returns the indices of the k largest values, sorted descending, via
// a bounded size-k min-heap (single O(V) pass). k<=0 or k>=len → all indices sorted.
func topKIndices(vals []float64, k int) []int {
	n := len(vals)
	if k <= 0 || k >= n {
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return vals[idx[a]] > vals[idx[b]] })
		return idx
	}
	h := make([]int, 0, k) // min-heap of indices keyed by vals[idx]
	for i := 0; i < n; i++ {
		if len(h) < k {
			h = append(h, i)
			for j := len(h) - 1; j > 0; {
				p := (j - 1) / 2
				if vals[h[p]] <= vals[h[j]] {
					break
				}
				h[p], h[j] = h[j], h[p]
				j = p
			}
		} else if vals[i] > vals[h[0]] {
			h[0] = i
			for j := 0; ; {
				l, r, sm := 2*j+1, 2*j+2, j
				if l < k && vals[h[l]] < vals[h[sm]] {
					sm = l
				}
				if r < k && vals[h[r]] < vals[h[sm]] {
					sm = r
				}
				if sm == j {
					break
				}
				h[j], h[sm] = h[sm], h[j]
				j = sm
			}
		}
	}
	sort.Slice(h, func(a, b int) bool { return vals[h[a]] > vals[h[b]] })
	return h
}

func (s *Server) sampleToken(logits []float32, params GenerationParams, rng *rand.Rand) (int32, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("empty logits")
	}
	if params.Temperature == 0 {
		return int32(cuda.Argmax(logits)), nil
	}
	n := len(logits)

	scaled := make([]float64, n)
	for i, v := range logits {
		scaled[i] = float64(v) / params.Temperature
	}
	if params.RepeatPenalty != 1.0 {
		// kv lock is held for the whole request, so CurrentTokens is safe.
		for _, past := range s.kv.CurrentTokens() {
			if int(past) >= 0 && int(past) < n {
				if scaled[past] < 0 {
					scaled[past] *= params.RepeatPenalty
				} else {
					scaled[past] /= params.RepeatPenalty
				}
			}
		}
	}

	type cand struct {
		id int32
		p  float64
	}

	// Select the top-k by scaled logit (temperature is monotonic) with a bounded
	// min-heap — one O(V) pass instead of sorting/allocating over all 262k logits
	// every token (~30 ms/token otherwise). Top-k disabled → full sort fallback.
	idx := topKIndices(scaled, params.TopK)
	maxLogit := scaled[idx[0]] // idx is sorted descending by scaled value
	cands := make([]cand, len(idx))
	for i, id := range idx {
		cands[i] = cand{int32(id), math.Exp(scaled[id] - maxLogit)}
	}

	// Normalize the (sorted, truncated) candidates.
	var sum float64
	for _, c := range cands {
		sum += c.p
	}
	if sum <= 0 {
		return int32(cuda.Argmax(logits)), nil
	}

	// Top-p: smallest prefix whose cumulative prob >= TopP.
	if params.TopP > 0 && params.TopP < 1.0 {
		cum, cut := 0.0, len(cands)
		for i := range cands {
			cum += cands[i].p / sum
			if cum >= params.TopP {
				cut = i + 1
				break
			}
		}
		cands = cands[:cut]
	}

	// Min-p: drop candidates below MinP * max_prob (cands[0] is the max).
	if params.MinP > 0 {
		thresh := params.MinP * cands[0].p
		keep := 1
		for i := 1; i < len(cands); i++ {
			if cands[i].p >= thresh {
				keep = i + 1
			} else {
				break
			}
		}
		cands = cands[:keep]
	}

	// Draw from the surviving candidates.
	var z float64
	for _, c := range cands {
		z += c.p
	}
	r := rng.Float64() * z
	var acc float64
	for _, c := range cands {
		acc += c.p
		if r <= acc {
			return c.id, nil
		}
	}
	return cands[len(cands)-1].id, nil
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeSSE(w http.ResponseWriter, v interface{}) {
	data, _ := json.Marshal(v)
	fmt.Fprintf(w, "data: %s\n\n", data)
}

// writeDONE writes the terminal SSE sentinel that OpenAI-style clients expect.
func writeDONE(w http.ResponseWriter) {
	fmt.Fprint(w, "data: [DONE]\n\n")
}

// stopHit reports whether `text` contains any of the stop strings. If so it
// returns the text truncated at the FIRST stop occurrence (the stop string
// itself is removed), matching OpenAI semantics.
func stopHit(text string, stops []string) (bool, string) {
	cut := -1
	for _, s := range stops {
		if s == "" {
			continue
		}
		if i := strings.Index(text, s); i >= 0 && (cut < 0 || i < cut) {
			cut = i
		}
	}
	if cut < 0 {
		return false, text
	}
	return true, text[:cut]
}

func (s *Server) Start(addr string) error {
	mux := http.NewServeMux()
	s.RegisterRoutes(mux)
	s.httpServer = &http.Server{Addr: addr, Handler: mux}
	log.Printf("gem4d: server listening on %s", addr)
	return s.httpServer.ListenAndServe()
}

func (s *Server) Stop() {
	if s.httpServer != nil {
		s.httpServer.Close()
	}
}
