package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/mauromedda/gem4d/internal/chat"
	"github.com/mauromedda/gem4d/internal/sampler"
	"github.com/mauromedda/gem4d/internal/tokenizer"
)

// ─── Engine interface ──────────────────────────────────────────────

// serverEngine is the consumer-side view of the inference engine that the HTTP
// handlers depend on. It is declared here (where it is used) rather than in the
// cuda package, per Go best practice, so the server can be tested with a fake
// engine and no GPU. *cuda.Engine satisfies this interface.
//
// It is a strict superset of kvEngine (kvcache.go), so a serverEngine can be
// passed straight to NewKVCache.
type serverEngine interface {
	Prefill(tokens []int32) ([]float32, error)
	Decode(token int32) ([]float32, error)
	DecodeNoCopy(token int32) error
	SampleDevice(temp float32, topK int, topP, minP, rnd float32) (int32, error)
	GenerateSpecContinue(history []int32, firstLogits []float32, maxNew int, stops []int32, draftK int, temp float32, topK int, topP, minP float32, seed uint64) ([]int32, int, error)
	NTokens() int
	Reset()
	Rewind(nKeep int) bool
	ContextSize() uint32
}

// ─── Types ─────────────────────────────────────────────────────────

type Server struct {
	engine          serverEngine
	tokenizer       *tokenizer.Tokenizer
	kv              *KVCache
	modelName       string
	genParams       GenerationParams
	thinkingDefault bool         // startup default for the gemma-4 reasoning channel
	debug           bool         // dump full request bodies + rendered prompts
	logLevel        atomic.Int32 // gates per-request access logs (see logLevelT)
	metrics         Metrics
	httpServer      *http.Server
}

// logLevelT is a tiny leveled-logging knob for the server package (no external
// deps). Higher = quieter. Per-request access logs are emitted at Info; setting
// the level to Warn silences them while leaving error/warn lines intact.
type logLevelT int32

const (
	logLevelDebug logLevelT = iota
	logLevelInfo
	logLevelWarn
)

// SetLogLevel sets the server's log verbosity from a string: "debug", "info"
// (default), or "warn"/"warning"/"error" (silences per-request access logs).
// Unknown values leave the level unchanged.
func (s *Server) SetLogLevel(level string) {
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "debug":
		s.logLevel.Store(int32(logLevelDebug))
	case "info", "":
		s.logLevel.Store(int32(logLevelInfo))
	case "warn", "warning", "error":
		s.logLevel.Store(int32(logLevelWarn))
	}
}

// logEnabled reports whether a message at the given level should be emitted.
func (s *Server) logEnabled(level logLevelT) bool {
	return int32(level) >= s.logLevel.Load()
}

type GenerationParams struct {
	Temperature      float64  `json:"temperature"`
	TopP             float64  `json:"top_p"`
	TopK             int      `json:"top_k"`
	MinP             float64  `json:"min_p"`
	Seed             int64    `json:"seed"`
	RepeatPenalty    float64  `json:"repeat_penalty"`
	FrequencyPenalty float64  `json:"frequency_penalty"`
	PresencePenalty  float64  `json:"presence_penalty"`
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

// CompletionResponse is the legacy /v1/completions (text_completion) shape:
// choices carry `text` instead of a chat `message`.
type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   Usage              `json:"usage"`
}

type CompletionChoice struct {
	Index        int    `json:"index"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"`
}

// CompletionStreamChoice is the streaming form of a legacy completion choice.
type CompletionStreamChoice struct {
	Index        int    `json:"index"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason,omitempty"`
}

type CompletionStreamResponse struct {
	ID      string                   `json:"id"`
	Object  string                   `json:"object"`
	Created int64                    `json:"created"`
	Model   string                   `json:"model"`
	Choices []CompletionStreamChoice `json:"choices"`
	Usage   *Usage                   `json:"usage,omitempty"`
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
	Created int64  `json:"created"` // unix seconds (OpenAI clients type this as int)
	OwnedBy string `json:"owned_by"`
}

func DefaultParams() GenerationParams {
	// google/gemma-4-12B model-card standardized sampling config (also embedded in
	// the GGUF general.sampling.*): temperature 1.0, top_p 0.95, top_k 64, no min-p
	// or repeat penalty.
	return GenerationParams{
		Temperature:      1.0,
		TopP:             0.95,
		TopK:             64,
		MinP:             0.0,
		Seed:             0,
		RepeatPenalty:    1.0,
		FrequencyPenalty: 0.0,
		PresencePenalty:  0.0,
		// Generous completion cap for when a client omits max_tokens (pi does): the
		// model stops at EOS/end-of-turn on its own; 512 truncated agent turns
		// (reasoning + answer) mid-output. This is only a safety bound on runaways.
		MaxTokens: 8192,
		Stream:    false,
	}
}

func New(eng serverEngine, tok *tokenizer.Tokenizer) *Server {
	s := &Server{
		engine:    eng,
		tokenizer: tok,
		kv:        NewKVCache(eng),
		modelName: "gemma-4-12b-it",
		genParams: DefaultParams(),
	}
	s.logLevel.Store(int32(logLevelInfo))
	return s
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
	h := func(f http.HandlerFunc) http.HandlerFunc { return s.logRequest(f) }
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
func (s *Server) logRequest(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		rec := &statusRecorder{ResponseWriter: w, status: 200}
		start := time.Now()
		next(rec, r)
		if s.logEnabled(logLevelInfo) {
			log.Printf("gem4d: %s %s -> %d (%.0fms)",
				r.Method, r.URL.Path, rec.status, float64(time.Since(start).Microseconds())/1000.0)
		}
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
	s.serveCompletions(w, r, false)
}

// serveCompletions backs both /v1/chat/completions (legacy=false, chat format)
// and /v1/completions (legacy=true, text_completion format). Apart from the
// response shape both paths share the same prompt/prefill/generation machinery.
func (s *Server) serveCompletions(w http.ResponseWriter, r *http.Request, legacy bool) {
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
	if lp := req.legacyPrompt(); lp != "" {
		prompt = lp
		legacy = true     // a `prompt` field forces legacy text_completion output
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
	// generation). The handler is the lock OWNER: it holds the lock for the
	// entire prefill+generation span and releases it here via defer. The
	// response helpers run while this lock is held and must NOT unlock it.
	s.kv.Lock()
	defer s.kv.Unlock()

	// Cache-aware prefill: reuse the longest cached prefix and compute only the
	// divergent suffix. The returned logits are for the final prompt token, so
	// no phantom Decode(0) is needed.
	prefillStart := time.Now()
	pf, err := s.kv.Prefill(tokens)
	if err != nil {
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
		s.streamResponse(r.Context(), w, params, promptTokens, logits, wantTools, legacy)
	} else {
		s.generateResponse(r.Context(), w, params, promptTokens, logits, wantTools, legacy)
	}
}

func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	s.serveCompletions(w, r, true)
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
			"prefix_hits":    hits,
			"prefix_misses":  misses,
			"token_hit_rate": hitRate,
			"cached_tokens":  s.engine.NTokens(),
			"context_size":   s.engine.ContextSize(),
		},
	})
}

func (s *Server) handleNotFound(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
}

// renderChatTemplate builds the gemma-4 prompt. The real vocab uses <|turn> /
// <turn|> delimiters (NOT <start_of_turn>/<end_of_turn>, which are not tokens).
// When tools are present they are declared in the (forced) system turn as
// <|tool>…<tool|> blocks; role:tool messages render as <|tool_response>… blocks.
// enableThinking gates the gemma-4 reasoning channel (see ChatRequest.resolveThinking).
//
// The turn structure lives in internal/chat; this method only injects the
// tool-specific syntax (declarations, tool_calls re-rendering, tool responses)
// through the chat.Renderer hooks, keeping all tool logic in this package.
func (s *Server) renderChatTemplate(messages []ChatMessage, tools []Tool, enableThinking bool) string {
	msgs := make([]chat.Message, len(messages))
	for i, m := range messages {
		msgs[i] = chat.Message{Role: m.Role, Content: m.Content}
	}

	// Tool declarations go inside the forced system turn.
	sysExtra := ""
	if len(tools) > 0 {
		sysExtra = renderToolDeclarations(tools)
	}

	r := chat.Renderer{
		EnableThinking: enableThinking,
		SystemExtra:    sysExtra,
		// Assistant tool_calls are re-rendered inside the model turn.
		TurnExtra: func(i int) string {
			if len(messages[i].ToolCalls) > 0 {
				return renderAssistantToolCalls(messages[i].ToolCalls)
			}
			return ""
		},
		// role:"tool" messages render as <|tool_response>… blocks.
		ToolResponse: func(i int) string {
			return renderToolResponse(messages[i].Name, messages[i].Content)
		},
	}
	return r.Render(msgs)
}

// generateResponse runs non-streaming generation. The caller (handler) is the
// lock OWNER and holds s.kv.Lock() for the whole request; this function runs
// under that lock and must NOT release it. When legacy is true it emits the
// /v1/completions text_completion shape instead of the chat.completion shape.
func (s *Server) generateResponse(ctx context.Context, w http.ResponseWriter, params GenerationParams, promptTokens int, logits []float32, wantTools, legacy bool) {
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
			// Record the token in the prefix cache only AFTER Decode commits it
			// to the engine KV. The tool-call-end and stop-sequence breaks above
			// exit the loop WITHOUT decoding the final token; recording it
			// before the commit left cachedTokens one token ahead of the engine
			// on every stop-sequence request, and the next Prefill had to heal
			// the skew (a warning plus one lost token of reuse).
			s.kv.AppendDecoded(token)
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

	if legacy {
		// Legacy /v1/completions: flatten reasoning + content into the text field.
		text := msg.Content
		if msg.ReasoningContent != "" {
			text = msg.ReasoningContent + text
		}
		writeJSON(w, http.StatusOK, CompletionResponse{
			ID:      fmt.Sprintf("cmpl-%d", time.Now().UnixNano()),
			Object:  "text_completion",
			Created: time.Now().Unix(),
			Model:   s.modelName,
			Choices: []CompletionChoice{{
				Index: 0, Text: text, FinishReason: finish,
			}},
			Usage: Usage{
				PromptTokens: promptTokens, CompletionTokens: generated,
				TotalTokens: promptTokens + generated,
			},
		})
		return
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

// streamResponse runs streaming generation. The caller (handler) is the lock
// OWNER and holds s.kv.Lock() for the whole request; this function runs under
// that lock and must NOT release it. When legacy is true it emits the
// /v1/completions text_completion stream shape (choices[].text) instead of the
// chat.completion.chunk shape (choices[].delta).
func (s *Server) streamResponse(ctx context.Context, w http.ResponseWriter, params GenerationParams, promptTokens int, logits []float32, wantTools, legacy bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	rng := rand.New(rand.NewSource(params.Seed))
	generated := 0
	object := "chat.completion.chunk"
	idPrefix := "chatcmpl"
	if legacy {
		object = "text_completion"
		idPrefix = "cmpl"
	}
	completionID := fmt.Sprintf("%s-%d", idPrefix, time.Now().UnixNano())
	created := time.Now().Unix()
	genStart := time.Now()

	// emitContent streams a piece of visible text in the right wire shape for the
	// active endpoint (chat delta vs legacy text).
	emitContent := func(text string) {
		if legacy {
			writeSSE(w, CompletionStreamResponse{
				ID: completionID, Object: object, Created: created, Model: s.modelName,
				Choices: []CompletionStreamChoice{{Index: 0, Text: text}},
			})
		} else {
			writeSSE(w, StreamResponse{
				ID: completionID, Object: object, Created: created, Model: s.modelName,
				Choices: []StreamChoice{{Index: 0, Delta: Delta{Content: text}}},
			})
		}
		flusher.Flush()
	}

	// Role delta (chat only; legacy text_completion has no role).
	if !legacy {
		writeSSE(w, StreamResponse{
			ID: completionID, Object: object, Created: created, Model: s.modelName,
			Choices: []StreamChoice{{Index: 0, Delta: Delta{Role: "assistant"}}},
		})
		flusher.Flush()
	}

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
		generated++

		// advance steps to the next token (GPU: leave logits on device; CPU:
		// copy) and only then records the token in the prefix cache: the token
		// enters the engine KV during this decode, not when it was sampled. The
		// tool-call-end and stop-sequence breaks below exit the loop WITHOUT
		// advancing, so recording before the commit left cachedTokens one token
		// ahead of the engine on EVERY streaming tool call / stop-string hit
		// (this path has no post-loop `committed` sync like the spec path), and
		// the next request's Prefill had to heal the skew.
		advance := func() error {
			if gpuSample {
				if derr := s.engine.DecodeNoCopy(token); derr != nil {
					return derr
				}
			} else {
				logits, err = s.engine.Decode(token)
				if err != nil {
					return err
				}
			}
			s.kv.AppendDecoded(token)
			return nil
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
				if legacy {
					// Legacy text_completion has no reasoning channel; fold it into text.
					emitContent(rtext)
				} else {
					writeSSE(w, StreamResponse{
						ID: completionID, Object: object, Created: created, Model: s.modelName,
						Choices: []StreamChoice{{Index: 0, Delta: Delta{ReasoningContent: rtext}}},
					})
					flusher.Flush()
				}
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
					emitContent(tail)
				}
				break
			}
		}

		if tokenStr != "" {
			emitted.WriteString(tokenStr)
			emitContent(tokenStr)
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
	usage := &Usage{
		PromptTokens:     promptTokens,
		CompletionTokens: generated,
		TotalTokens:      promptTokens + generated,
	}
	if legacy {
		writeSSE(w, CompletionStreamResponse{
			ID: completionID, Object: object, Created: created, Model: s.modelName,
			Choices: []CompletionStreamChoice{{Index: 0, Text: "", FinishReason: finish}},
			Usage:   usage,
		})
	} else {
		writeSSE(w, StreamResponse{
			ID: completionID, Object: object, Created: created, Model: s.modelName,
			Choices: []StreamChoice{{Index: 0, Delta: Delta{}, FinishReason: finish}},
			Usage:   usage,
		})
	}
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
// multinomial via the shared sampler package. Temperature 0 is greedy argmax.
// The repeat penalty operates on the RAW logits BEFORE temperature (llama.cpp
// semantics); pastTokens are supplied ONLY when a penalty is active. The kv lock
// is held for the whole request, so CurrentTokens is safe to read here.
func (s *Server) sampleToken(logits []float32, params GenerationParams, rng *rand.Rand) (int32, error) {
	var pastTokens []int32
	if params.RepeatPenalty != 1.0 {
		pastTokens = s.kv.CurrentTokens()
	}
	return sampler.Sample(logits, sampler.Params{
		Temperature:   params.Temperature,
		TopK:          params.TopK,
		TopP:          params.TopP,
		MinP:          params.MinP,
		RepeatPenalty: params.RepeatPenalty,
	}, rng, pastTokens)
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

// Stop gracefully shuts the HTTP server down, allowing in-flight requests up to
// ~10s to finish before forcing connections closed. On timeout or shutdown
// error it falls back to an immediate Close.
func (s *Server) Stop() {
	if s.httpServer == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := s.httpServer.Shutdown(ctx); err != nil {
		log.Printf("gem4d: graceful shutdown failed (%v), forcing close", err)
		s.httpServer.Close()
	}
}
