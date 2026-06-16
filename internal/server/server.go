package server

import (
	"context"
	crand "crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/hikmaai-io/fucina/internal/chat"
	"github.com/hikmaai-io/fucina/internal/sampler"
	"github.com/hikmaai-io/fucina/internal/server/batch"
	"github.com/hikmaai-io/fucina/internal/tokenizer"
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
	GenerateSpecContinue(history []int32, firstLogits []float32, maxNew int, stops []int32, draftK int, temp float32, topK int, topP, minP, repeatPenalty float32, seed uint64) ([]int32, int, error)
	GenerateSpecStream(history []int32, firstLogits []float32, maxNew int, stops []int32, draftK int, temp float32, topK int, topP, minP, repeatPenalty float32, seed uint64, emit func(int32) bool) ([]int32, int, error)
	NTokens() int
	Reset()
	Rewind(nKeep int) bool
	ContextSize() uint32
	SpecStats() (steps, drafted, accepted, emitted int64)
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
	draftK          int          // speculative draft length per step (--draft-k)
	thinkBudget     int          // reasoning-channel token budget: 0=auto (MaxTokens/2), <0=off
	lastUsed        atomic.Int64 // engine NTokens mirror so /metrics never blocks on the kv lock
	// Speculative-decode counter mirrors, refreshed at end-of-generation under the kv
	// lock so /metrics reads them lock-free (calling engine.SpecStats() from /metrics
	// would take the engine mutex and block behind a multi-minute generation).
	specSteps    atomic.Int64
	specDrafted  atomic.Int64
	specAccepted atomic.Int64
	specEmitted  atomic.Int64
	metrics      Metrics
	httpServer   *http.Server

	debugDumpFull atomic.Bool // set once the debug dump file hits its size cap

	// apiKey, when non-empty, is required as `Authorization: Bearer <key>` on all
	// /v1/* routes (constant-time compared). Empty = auth disabled (localhost dev).
	apiKey string

	// maxOutputTokens is an absolute ceiling on tokens generated per request,
	// independent of the context window, so one client cannot monopolize the
	// single-flight GPU for a 130k-token generation. 0 = no extra cap.
	maxOutputTokens int

	// inflight bounds concurrent inference. The engine is single-flight, so this
	// is a small buffered channel acting as an admission queue: a full channel
	// means the server is saturated and new requests get 503 instead of piling
	// up unbounded goroutines (each holding a buffered body) behind the kv lock.
	inflight chan struct{}

	// scheduler is the continuous-batching scheduler. It is nil unless batching
	// was enabled at startup (SetBatchEngine, gated on FUCINA_BATCH). When non-nil
	// serveCompletions routes through it (per-step serialization, no per-request
	// kv lock) instead of the single-flight kv path. Its presence is a pure
	// additive opt-in: with it nil the behaviour is exactly as before.
	scheduler *batch.Scheduler
}

// prefillAborter is satisfied by engines that support cooperative prefill
// cancellation (cuda.Engine). Detected by assertion so test fakes stay minimal.
type prefillAborter interface{ AbortPrefill() }

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
	Tools            []Tool   `json:"-"` // request tool schemas, for required-param validation
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

// validateAndClampParams sanitizes sampling parameters before they cross into the
// CUDA C kernels (where there is no bounds checking). It clamps in-range values
// and returns a non-empty message for values that cannot be safely coerced
// (NaN/Inf), so the caller can reject the request with 400. vocab is the
// tokenizer vocabulary size (the upper bound for top_k).
func validateAndClampParams(p *GenerationParams, vocab int) string {
	// NaN/Inf cannot be clamped meaningfully — reject rather than guess. A fixed
	// array (not a map) keeps this stack-allocated on the per-request hot path and
	// makes the reported field deterministic (map iteration order is random).
	for _, f := range [...]struct {
		name string
		v    float64
	}{
		{"temperature", p.Temperature}, {"top_p", p.TopP},
		{"min_p", p.MinP}, {"repeat_penalty", p.RepeatPenalty},
	} {
		if math.IsNaN(f.v) || math.IsInf(f.v, 0) {
			return "invalid " + f.name + ": must be a finite number"
		}
	}
	// top_k: negative is nonsense to the device kernel; 0 means "no top-k". Cap at
	// the vocabulary size so it can never index past the logits buffer.
	if p.TopK < 0 {
		p.TopK = 0
	}
	if vocab > 0 && p.TopK > vocab {
		p.TopK = vocab
	}
	// Probabilities live in [0,1]; temperature must be non-negative (0 = greedy).
	p.TopP = clamp01(p.TopP)
	p.MinP = clamp01(p.MinP)
	if p.Temperature < 0 {
		p.Temperature = 0
	}
	if p.RepeatPenalty < 0 {
		p.RepeatPenalty = 0
	}
	return ""
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// defaultInflight is the admission-queue depth: the engine runs one request at a
// time, so this is 1 in-flight plus a small wait queue. Beyond it, requests get
// 503 instead of stacking unbounded goroutines (each pinning a buffered body)
// behind the kv lock. Tunable via SetMaxConcurrent.
const defaultInflight = 4

func New(eng serverEngine, tok *tokenizer.Tokenizer) *Server {
	s := &Server{
		engine:    eng,
		tokenizer: tok,
		kv:        NewKVCache(eng),
		modelName: "gemma-4-12b-it",
		genParams: DefaultParams(),
		draftK:    6,
		inflight:  make(chan struct{}, defaultInflight),
	}
	s.logLevel.Store(int32(logLevelInfo))
	return s
}

// BatchEngine is the engine the continuous-batching scheduler drives. It is the
// scheduler's contract (batch.BatchEngine) plus Supported(), so the server can
// refuse to enable batching when the engine was not built for it (e.g. paged KV
// off). *cuda.BatchAdapter satisfies it; the server stays GPU-free for tests.
type BatchEngine interface {
	batch.BatchEngine
	// Supported reports whether the engine can actually serve batched requests
	// (paged mode enabled). When false the scheduler must not be started.
	Supported() bool
}

// SetBatchEngine enables continuous batching: it constructs and starts a
// batch.Scheduler over eng so serveCompletions routes through per-step batching
// instead of the per-request kv lock. It is a no-op (returns false) when eng is
// nil or reports !Supported(), leaving the single-flight path untouched. Call it
// once at startup, after the engine is warmed, gated on FUCINA_BATCH.
func (s *Server) SetBatchEngine(eng BatchEngine) bool {
	if eng == nil || !eng.Supported() {
		return false
	}
	// Queue depth mirrors the single-flight admission channel: waiting requests
	// not yet admitted to a slot. The scheduler admits up to engine Capacity()
	// concurrently and queues the rest (surfacing ErrQueueFull as a 503).
	slots := eng.Capacity()
	if slots < 1 {
		slots = 1
	}
	// Queue depth: room for a backlog of waiting requests beyond the live slots.
	depth := slots
	s.scheduler = batch.New(eng, depth)
	s.scheduler.Start()
	// The single-flight inflight bound (default 4) would cap concurrency below
	// the engine's slot budget, since each batched handler holds an inflight slot
	// for its whole request. Grow it so up to Capacity() sequences can run
	// concurrently plus a queue depth of waiters; the scheduler's own ErrQueueFull
	// backpressure still sheds load past that.
	s.inflight = make(chan struct{}, slots+depth)
	return true
}

// SetAPIKey enables bearer-token auth on /v1/* routes. Empty disables auth
// (localhost dev default). The key is compared in constant time.
func (s *Server) SetAPIKey(key string) { s.apiKey = key }

// SetMaxOutputTokens sets an absolute ceiling on generated tokens per request,
// independent of the context window, so one client cannot monopolize the
// single-flight GPU for a 130k-token generation. 0 = no extra cap.
func (s *Server) SetMaxOutputTokens(n int) {
	if n >= 0 {
		s.maxOutputTokens = n
	}
}

// SetMaxConcurrent sets the admission-queue depth (in-flight + waiting requests).
// Values < 1 are ignored. The engine is single-flight, so 2-8 is the useful range.
func (s *Server) SetMaxConcurrent(n int) {
	if n >= 1 {
		s.inflight = make(chan struct{}, n)
	}
}

// SetDraftK sets the speculative draft length used by the server generation
// paths (--draft-k). Out-of-range values are ignored (engine cap is SPEC_MAX-1).
func (s *Server) SetDraftK(k int) {
	if k > 0 && k <= 15 {
		s.draftK = k
	}
}

// SetThinkBudget bounds the gemma-4 reasoning channel per request: after this
// many thought tokens the server force-closes the channel (committing the
// <channel|> token) and lets the model answer. 0 = auto (half of MaxTokens),
// negative = unlimited. Guards against the runaway-thinking failure mode where
// a turn burns the whole MaxTokens cap inside the thought channel and delivers
// an empty-content message.
func (s *Server) SetThinkBudget(n int) { s.thinkBudget = n }

// SetKVSnapshotBudget bounds the host memory for snapshotted KV sequences
// (multi-conversation prefix cache; --kv-snapshot-gb). 0 disables. No effect
// when the engine lacks snapshot support.
func (s *Server) SetKVSnapshotBudget(bytes int64) { s.kv.SetSnapshotBudget(bytes) }

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

// debugDumpPath is where SetDebug(true) appends request/prompt dumps. It is
// created 0600 (owner-only): the dump contains raw prompts, which for a coding
// agent include file contents and pasted secrets — it must not be world-readable.
const debugDumpPath = "/tmp/fucina_debug.log"

// debugDumpMaxBytes caps the dump file so an enabled --debug session cannot fill
// the disk. Once the file reaches this size, writes are skipped (a one-line
// notice is logged on the first skip per process).
const debugDumpMaxBytes = 256 << 20 // 256 MiB

// writeDebugDump appends s to the owner-only debug file, honoring the size cap.
// All errors are logged rather than ignored.
func (s *Server) writeDebugDump(dump string) {
	if fi, err := os.Stat(debugDumpPath); err == nil && fi.Size() >= debugDumpMaxBytes {
		if !s.debugDumpFull.Swap(true) {
			log.Printf("fucina: debug dump %s reached %d bytes — further dumps skipped", debugDumpPath, debugDumpMaxBytes)
		}
		return
	}
	f, err := os.OpenFile(debugDumpPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		log.Printf("fucina: debug dump open failed: %v", err)
		return
	}
	defer func() {
		if cerr := f.Close(); cerr != nil {
			log.Printf("fucina: debug dump close failed: %v", cerr)
		}
	}()
	if _, err := f.WriteString(dump); err != nil {
		log.Printf("fucina: debug dump write failed: %v", err)
	}
}

func (s *Server) RegisterRoutes(mux *http.ServeMux) {
	// /v1/* routes carry auth; health/metrics stay open for probes/scrapers. All
	// routes get the recover + access-log middleware.
	open := func(f http.HandlerFunc) http.HandlerFunc { return s.logRequest(f) }
	authed := func(f http.HandlerFunc) http.HandlerFunc { return s.logRequest(s.requireAuth(f)) }
	mux.HandleFunc("/v1/models", authed(s.handleModels))
	mux.HandleFunc("/v1/chat/completions", authed(s.handleChatCompletions))
	mux.HandleFunc("/v1/completions", authed(s.handleCompletions))
	mux.HandleFunc("/v1/embeddings", authed(s.handleEmbeddings))
	mux.HandleFunc("/health", open(s.handleHealth))
	mux.HandleFunc("/healthz", open(s.handleHealth))
	mux.HandleFunc("/readyz", open(s.handleReady))
	mux.HandleFunc("/metrics", open(s.handleMetrics))
	mux.HandleFunc("/", open(s.handleNotFound))
}

// requireAuth enforces a constant-time bearer-token check when an API key is
// configured. With no key set (the localhost dev default) it is a pass-through.
func (s *Server) requireAuth(next http.HandlerFunc) http.HandlerFunc {
	return BearerAuth(s.apiKey, next)
}

// BearerAuth wraps a handler with a constant-time `Authorization: Bearer <key>`
// check. An empty key disables auth (pass-through). It is exported so the
// separate diffusion-server mux (cmd/fucina) gets the same auth as the dense
// server rather than silently serving unauthenticated when a key is configured.
func BearerAuth(key string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if key == "" {
			next(w, r)
			return
		}
		const prefix = "Bearer "
		h := r.Header.Get("Authorization")
		if !strings.HasPrefix(h, prefix) ||
			subtle.ConstantTimeCompare([]byte(h[len(prefix):]), []byte(key)) != 1 {
			writeJSON(w, http.StatusUnauthorized, map[string]interface{}{
				"error": map[string]string{"message": "invalid or missing API key", "type": "invalid_request_error"},
			})
			return
		}
		next(w, r)
	}
}

// handleMetrics reports live KV/context utilization, prefix-cache hit rate, and
// prefill/decode throughput (cumulative + last request). JSON for easy curl/pi use.
// LOCK-FREE: it must answer instantly even while a request holds the kv lock for
// a multi-second prefill+generation span (s.lastUsed mirrors engine.NTokens).
func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	ctxCap := int(s.engine.ContextSize())
	used := int(s.lastUsed.Load())
	hits, misses, reused, reqTok := s.kv.DetailedStats()

	// gemma-4 KV memory (FP8, 1 byte/elem; K and V stored separately). Since the
	// flat per-position sliding cache (DECODE-30-35 Step 3) BOTH caches scale
	// with the context capacity:
	//   sliding: MAX_LAYERS(48) × KV_HEADS(8) × HEAD_DIM(256) × ctxCap
	//   global:  GLOBAL_LAYERS(8) × GLOBAL_HEAD_DIM(512) × ctxCap
	const mib = 1024.0 * 1024.0
	slidingMB := float64(48*8*256) * float64(ctxCap) * 2 / mib
	globalMB := float64(8*512) * float64(ctxCap) * 2 / mib

	// Lock-free: read the mirrored spec counters (refreshed under the kv lock at
	// end-of-generation), NOT engine.SpecStats() which would take the engine mutex.
	snap := s.metrics.snapshot(
		s.modelName, used, ctxCap, slidingMB, globalMB, hits, misses, reused, reqTok,
		s.specSteps.Load(), s.specDrafted.Load(), s.specAccepted.Load(), s.specEmitted.Load())
	// Saturation gauge: how many admission slots are taken vs the cap. Reading the
	// channel's len/cap is lock-free and shows queueing before it becomes latency.
	snap["saturation"] = map[string]interface{}{
		"in_flight":     len(s.inflight),
		"max_in_flight": cap(s.inflight),
	}
	writeJSON(w, http.StatusOK, snap)
}

// newRequestID returns a short random hex id for request correlation. crypto/rand
// keeps it dependency-free; on the vanishingly rare read error it falls back to a
// timestamp so a request is never left without an id.
func newRequestID() string {
	var b [8]byte
	if _, err := crand.Read(b[:]); err != nil {
		return "req-" + strconv.FormatInt(time.Now().UnixNano(), 36)
	}
	return hex.EncodeToString(b[:])
}

// statusRecorder captures the response status so the access log can report it.
// wroteHeader lets the panic recover decide whether a 500 can still be sent.
type statusRecorder struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.wroteHeader = true
	r.ResponseWriter.WriteHeader(code)
}

// Write marks the header as sent (implicit 200) so the recover knows a body is
// already on the wire and it can only log, not rewrite the status.
func (r *statusRecorder) Write(b []byte) (int, error) {
	r.wroteHeader = true
	return r.ResponseWriter.Write(b)
}

// Flush is needed because the streaming handler type-asserts http.Flusher.
func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Unwrap lets http.NewResponseController reach the underlying writer's
// per-request deadline/flush support through this wrapper.
func (r *statusRecorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

// FlushError propagates flush errors (e.g. an expired write deadline) through
// this wrapper. Without it, ResponseController.Flush matches the error-less
// Flush() above and always reports nil — which silently killed the
// stalled-client cutoff (sseWriter.stalled) in production.
func (r *statusRecorder) FlushError() error {
	return http.NewResponseController(r.ResponseWriter).Flush()
}

// logRequest logs each HTTP request's method, path, status, and duration so that
// client failures (4xx/5xx from clients like pi) are visible — the handlers return
// errors via http.Error without logging, which otherwise leaves silent failures.
func (s *Server) logRequest(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		rec := &statusRecorder{ResponseWriter: w, status: 200}
		start := time.Now()
		// Correlation: honor an inbound X-Request-Id (cross-service tracing) or
		// mint one, echo it back, and tag every log line for this request so a
		// 500 can be tied to its access line under concurrency.
		reqID := r.Header.Get("X-Request-Id")
		if reqID == "" {
			reqID = newRequestID()
		}
		rec.Header().Set("X-Request-Id", reqID)
		// Recover so one request's panic cannot take down the shared process.
		// net/http recovers per-connection, but this also covers panics that
		// escape into helper paths and gives a uniform 500 + access-log line.
		// wroteHeader tracks whether we can still set a 500 (a streaming handler
		// has already sent 200 + bytes; then we can only log).
		defer func() {
			if rv := recover(); rv != nil {
				log.Printf("fucina: [%s] PANIC in %s %s: %v", reqID, r.Method, r.URL.Path, rv)
				if !rec.wroteHeader {
					rec.status = http.StatusInternalServerError
					http.Error(rec, "internal server error", http.StatusInternalServerError)
				}
			}
			dur := time.Since(start)
			// Only the inference endpoints count toward SLO metrics; /metrics and
			// /health self-scrapes would otherwise dominate the averages.
			if strings.HasPrefix(r.URL.Path, "/v1/") {
				s.metrics.recordRequest(rec.status, dur)
			}
			if s.logEnabled(logLevelInfo) {
				log.Printf("fucina: [%s] %s %s -> %d (%.0fms)",
					reqID, r.Method, r.URL.Path, rec.status, float64(dur.Microseconds())/1000.0)
			}
		}()
		next(rec, r)
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
	// Admission control: the engine is single-flight, so without a bound, K
	// concurrent clients spawn K goroutines that each buffer a body and park on
	// the kv lock — unbounded memory + unbounded tail latency. Acquire a slot or
	// shed load with 503 + Retry-After. The slot is released in all exit paths.
	select {
	case s.inflight <- struct{}{}:
		defer func() { <-s.inflight }()
	default:
		w.Header().Set("Retry-After", "1")
		writeJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
			"error": map[string]string{"message": "server busy: too many concurrent requests", "type": "overloaded"},
		})
		return
	}
	// Bound the body read: an agent context is large (hundreds of KB) but finite;
	// without a cap any client could OOM the process that also owns the GPU.
	r.Body = http.MaxBytesReader(w, r.Body, 64<<20)
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
	// size, tool count, thinking, streaming). FUCINA_DEBUG=1 also dumps the full
	// request body + rendered prompt to /tmp for inspecting exactly what a client
	// (e.g. pi) sends.
	sysChars := 0
	if len(req.Messages) > 0 && req.Messages[0].Role == "system" {
		sysChars = len(req.Messages[0].Content)
	}
	log.Printf("fucina: chat: %d msgs, %d tools, sys=%dch, %d prompt-tok, thinking=%v, stream=%v",
		len(req.Messages), len(req.Tools), sysChars, len(tokens), enableThinking, req.Stream)
	if s.debug || os.Getenv("FUCINA_DEBUG") == "1" {
		dump := fmt.Sprintf("\n========== %s  %d msgs / %d tools / %d tok / thinking=%v / stream=%v ==========\n"+
			"--- REQUEST BODY ---\n%s\n--- RENDERED PROMPT ---\n%s\n",
			time.Now().Format("15:04:05"), len(req.Messages), len(req.Tools), len(tokens),
			enableThinking, req.Stream, string(body), prompt)
		s.writeDebugDump(dump)
	}

	params := s.genParams
	params.Tools = req.Tools // for required-parameter validation of emitted calls
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
		// Clamp to the window. A client (pi) sizes max_tokens to its CONFIGURED
		// contextWindow, which may exceed the server's actual --ctx. Left unclamped,
		// budget = ctx-MaxTokens below goes negative, the compaction guard never
		// fires, and an over-long prompt is pushed into a smaller KV cache (window
		// wrap → garbage + full re-prefill every turn). Reserve at least half the
		// window for the prompt so completion can never starve prefill entirely.
		if cap := ctx / 2; params.MaxTokens > cap {
			params.MaxTokens = cap
		}
	} else {
		// No client cap (pi omits max_tokens). DO NOT set MaxTokens = ctx-len(tokens):
		// that collapses the completion budget toward zero as the conversation grows,
		// and at full context clamps to 1 token — so pi gets a single-token (dead)
		// reply and the compaction guard below (budget = ctx-MaxTokens = len(tokens))
		// never even fires. Instead RESERVE a fixed completion budget and let
		// compaction trim the OLDEST prompt tokens to make room. The model stops at
		// EOS/end-of-turn on its own well before this bound; it is only a runaway cap.
		params.MaxTokens = s.genParams.MaxTokens // generous default (DefaultParams: 8192)
		if cap := ctx / 2; params.MaxTokens > cap {
			params.MaxTokens = cap // never starve the prompt of room
		}
		if params.MaxTokens < 1 {
			params.MaxTokens = 1
		}
	}
	params.Stream = req.Stream
	params.Stop = req.Stop

	// Absolute output ceiling (independent of the context window): one client must
	// not be able to monopolize the single-flight GPU for a ctx/2 (up to ~131k)
	// token generation. 0 = no extra cap (the ctx/2 clamp above still applies).
	if s.maxOutputTokens > 0 && params.MaxTokens > s.maxOutputTokens {
		params.MaxTokens = s.maxOutputTokens
	}

	// Sanitize sampling knobs BEFORE they cross into CUDA C kernels. Unvalidated
	// values (negative/huge top_k, NaN/Inf temp) reach the device sampler raw on
	// the spec path; clamp them here so a malformed request cannot crash or
	// corrupt the engine. Rejects the request rather than guessing on NaN/Inf.
	if msg := validateAndClampParams(&params, s.tokenizer.NumTokens()); msg != "" {
		http.Error(w, msg, http.StatusBadRequest)
		return
	}

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
		log.Printf("fucina: context compaction: prompt %d + max_tokens %d > ctx %d; dropped %d oldest tokens",
			len(tokens), params.MaxTokens, ctx, dropped)
		tokens = kept
	}

	// Continuous-batching path (FUCINA_BATCH): route the request through the
	// scheduler so it shares a per-step batched forward with other in-flight
	// sequences, instead of holding the whole-request kv lock. Everything below
	// (the single-flight kv path) is left exactly as before for when batching is
	// off (s.scheduler == nil).
	if s.scheduler != nil {
		s.serveBatch(w, r, params, tokens, wantTools, legacy)
		return
	}

	// Streaming requests get their first bytes BEFORE the prefill: SSE headers +
	// role delta now (~1ms TTFB instead of the full prefill latency), then a
	// ": ping" heartbeat while the prefill runs, so the client (and any proxy)
	// can tell a working server from a hung one. Consequence: any later failure
	// must be reported in-stream — the 200 is already on the wire.
	var sse *sseWriter
	if params.Stream {
		var ok bool
		sse, ok = newSSEWriter(w, legacy, s.modelName)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}
		sse.begin()
	}

	// Acquire the single physical KV cache for the whole request (prefill +
	// generation). The handler is the lock OWNER: it holds the lock for the
	// entire prefill+generation span and releases it here via defer. The
	// response helpers run while this lock is held and must NOT unlock it.
	if sse != nil {
		sse.startHeartbeat(10 * time.Second)
		// Panic safety only: every normal path stops the heartbeat explicitly
		// (and MUST, before any token write). stopHeartbeat is idempotent; this
		// defer prevents a leaked goroutine writing to a dead ResponseWriter if
		// Prefill panics.
		defer sse.stopHeartbeat()
	}
	s.kv.Lock()
	defer s.kv.Unlock()

	// The wait for the lock can be long (another request's prefill+generation).
	// If the client gave up in the meantime, don't burn a prefill for a dead
	// socket — N rapid abort+retry cycles otherwise stack N zombie prefills.
	if r.Context().Err() != nil {
		if sse != nil {
			sse.stopHeartbeat()
			s.finishStream(sse, "cancelled", len(tokens), 0)
		} else {
			http.Error(w, "client closed request", 499)
		}
		log.Printf("fucina: request cancelled while queued; skipping prefill")
		return
	}

	// Cooperative prefill cancellation: a watcher trips the engine's abort flag
	// when the client disconnects, so an ESC during a long prefill stops at the
	// next chunk/layer boundary instead of grinding on with the lock held.
	prefillDone := make(chan struct{})
	watcherDone := make(chan struct{})
	if ab, ok := s.engine.(prefillAborter); ok {
		go func() {
			defer close(watcherDone)
			select {
			case <-r.Context().Done():
				ab.AbortPrefill()
			case <-prefillDone:
			}
		}()
	} else {
		close(watcherDone)
	}

	// Debug: show WHERE the new prompt diverges from the cached sequence, in
	// both token ids and text. This is the tool for diagnosing prefix-cache
	// misses caused by re-render drift (a rendered turn that does not
	// token-match what generation committed to the KV).
	if s.debug || os.Getenv("FUCINA_DEBUG") == "1" {
		cached := s.kv.CurrentTokens()
		lcp := longestCommonPrefix(cached, tokens)
		if lcp < len(cached) && lcp < len(tokens) {
			lo := lcp - 8
			if lo < 0 {
				lo = 0
			}
			hiC, hiP := lcp+24, lcp+24
			if hiC > len(cached) {
				hiC = len(cached)
			}
			if hiP > len(tokens) {
				hiP = len(tokens)
			}
			log.Printf("fucina: prefix diverges at %d/%d cached:\n  cache:  %v %q\n  prompt: %v %q",
				lcp, len(cached),
				cached[lo:hiC], s.tokenizer.DecodeRaw(cached[lo:hiC]),
				tokens[lo:hiP], s.tokenizer.DecodeRaw(tokens[lo:hiP]))
		}
	}

	// Cache-aware prefill: reuse the longest cached prefix and compute only the
	// divergent suffix. The returned logits are for the final prompt token, so
	// no phantom Decode(0) is needed.
	prefillStart := time.Now()
	pf, err := s.kv.Prefill(tokens)
	close(prefillDone)
	// JOIN the watcher before proceeding: a stale AbortPrefill must land while
	// THIS request still holds the kv lock, so the next request's chain-head
	// clear erases it. An un-joined watcher could fire after the next request's
	// prefill started and spuriously abort it (confirmed race).
	<-watcherDone
	if sse != nil {
		sse.stopHeartbeat() // heartbeat must be joined before any token writes
	}
	if err != nil {
		s.lastUsed.Store(int64(s.engine.NTokens())) // keep the /metrics mirror honest
		if r.Context().Err() != nil {
			// Client-initiated abort (the watcher tripped the engine flag) —
			// not a server fault. The prefix cache keeps the shared prefix
			// (kvcache treats aborts as consistent), so the retry re-prefills
			// only the suffix.
			log.Printf("fucina: prefill aborted by client disconnect")
			if sse != nil {
				s.finishStream(sse, "cancelled", len(tokens), 0)
			}
			return
		}
		log.Printf("fucina: prefill failed: %v", err)
		if sse != nil {
			sse.errorEvent(fmt.Sprintf("prefill failed: %v", err))
		} else {
			http.Error(w, fmt.Sprintf("prefill failed: %v", err), http.StatusInternalServerError)
		}
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
	s.lastUsed.Store(int64(used))
	log.Printf("fucina: prefill %d tokens (%d cached, %d new) in %.2fs (%.1f tok/s) | ctx %d/%d (%.0f%%)",
		promptTokens, pf.ReusedTokens, pf.NewTokens, prefillElapsed.Seconds(), prefillTPS,
		used, ctx, 100.0*float64(used)/float64(ctx))

	logits := pf.Logits

	if params.Stream {
		s.streamResponse(r.Context(), sse, params, promptTokens, logits, wantTools)
	} else {
		s.generateResponse(r.Context(), w, params, promptTokens, logits, wantTools, legacy)
	}
	s.lastUsed.Store(int64(s.engine.NTokens()))
	// Refresh the lock-free spec-decode mirror for /metrics (still under the kv lock).
	st, dr, ac, em := s.engine.SpecStats()
	s.specSteps.Store(st)
	s.specDrafted.Store(dr)
	s.specAccepted.Store(ac)
	s.specEmitted.Store(em)
}

// serveBatch handles a completion via the continuous-batching scheduler. It
// submits one batch.Request (prefill prompt, greedy/temperature sampling on the
// device, stop at EOS/end-of-turn) and drives the per-token Emit/Done lifecycle
// onto the wire: streaming deltas for SSE requests, a single collected response
// otherwise. It is the per-request analogue of streamResponse/generateResponse
// for the batched path, but it never touches the kv lock — the scheduler owns
// the engine and serializes per step, not per request.
//
// Sampling note: the current batched C ABI samples greedily on-device regardless
// of SeqParams. The params are still forwarded so the contract is stable for when
// the kernels grow temperature/top-k support.
func (s *Server) serveBatch(w http.ResponseWriter, r *http.Request, params GenerationParams, tokens []int32, wantTools, legacy bool) {
	stops := []int32{s.tokenizer.EOS, s.tokenizer.EndOfTurn}

	seed := uint64(params.Seed)
	if params.Seed < 0 {
		seed = uint64(time.Now().UnixNano())
	}
	sp := batch.SeqParams{
		Temperature:   float32(params.Temperature),
		TopK:          params.TopK,
		TopP:          float32(params.TopP),
		MinP:          float32(params.MinP),
		RepeatPenalty: float32(params.RepeatPenalty),
		Seed:          seed,
	}

	// tokCh carries sampled token ids from the scheduler goroutine to THIS
	// handler goroutine. Emit must not block the shared step loop, so it does a
	// non-blocking send and asks the scheduler to evict (returns false) if the
	// buffer is full — i.e. this client is too slow to keep up. The buffer is
	// generous so a transient write hiccup does not drop the sequence.
	tokCh := make(chan int32, 1024)
	done := make(chan batch.Result, 1)

	req := batch.Request{
		Tokens: tokens,
		Params: sp,
		Stops:  stops,
		MaxNew: params.MaxTokens,
		Ctx:    r.Context(),
		Emit: func(t int32) bool {
			select {
			case tokCh <- t:
				return true
			default:
				return false // client backpressure: drop the sequence
			}
		},
		Done: done,
	}

	if err := s.scheduler.Submit(req); err != nil {
		// Queue full / shutting down: shed load the same way the single-flight
		// path does (503), or report shutdown.
		w.Header().Set("Retry-After", "1")
		writeJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
			"error": map[string]string{"message": "server busy: too many concurrent requests", "type": "overloaded"},
		})
		return
	}

	if params.Stream {
		s.streamBatch(w, r, tokCh, done, legacy)
	} else {
		s.collectBatch(w, r, tokCh, done, wantTools, legacy, len(tokens), params.Tools)
	}
}

// drainTokens reads token ids from tokCh until done fires AND tokCh is empty,
// invoking onTok for each id in order. It returns the terminal Result. Because
// the scheduler delivers the terminal Done only AFTER its last Emit, draining
// tokCh to empty after Done guarantees no in-flight token is lost.
func drainTokens(tokCh <-chan int32, done <-chan batch.Result, onTok func(int32)) batch.Result {
	for {
		select {
		case t := <-tokCh:
			onTok(t)
		case res := <-done:
			// Flush any tokens already queued before the terminal result.
			for {
				select {
				case t := <-tokCh:
					onTok(t)
				default:
					return res
				}
			}
		}
	}
}

// streamBatch streams a batched sequence's tokens to an SSE client. It decodes
// incrementally (decode the whole id slice each step and emit only the new text)
// so multi-byte UTF-8 / SentencePiece pieces are never split mid-character.
func (s *Server) streamBatch(w http.ResponseWriter, r *http.Request, tokCh <-chan int32, done <-chan batch.Result, legacy bool) {
	sse, ok := newSSEWriter(w, legacy, s.modelName)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}
	sse.begin()

	var ids []int32
	var emitted string
	generated := 0
	genStart := time.Now()

	res := drainTokens(tokCh, done, func(t int32) {
		generated++
		if s.tokenizer.IsStop(t) {
			return // never render stop markers
		}
		ids = append(ids, t)
		full := stripMarkers(s.tokenizer.DecodeRaw(ids))
		if len(full) > len(emitted) {
			delta := full[len(emitted):]
			emitted = full
			if legacy {
				sse.event(CompletionStreamResponse{
					ID: sse.id, Object: sse.object, Created: sse.created, Model: s.modelName,
					Choices: []CompletionStreamChoice{{Index: 0, Text: delta}},
				})
			} else {
				sse.event(StreamResponse{
					ID: sse.id, Object: sse.object, Created: sse.created, Model: s.modelName,
					Choices: []StreamChoice{{Index: 0, Delta: Delta{Content: delta}}},
				})
			}
		}
	})

	s.logGenSpeed(genStart, generated)
	finish := batchFinish(res, r.Context())
	s.finishStream(sse, finish, 0, generated)
}

// collectBatch accumulates a non-streaming batched sequence and writes the full
// response (with reasoning split + tool-call parsing, mirroring generateResponse).
func (s *Server) collectBatch(w http.ResponseWriter, r *http.Request, tokCh <-chan int32, done <-chan batch.Result, wantTools, legacy bool, promptTokens int, tools []Tool) {
	var ids []int32
	genStart := time.Now()
	res := drainTokens(tokCh, done, func(t int32) {
		if s.tokenizer.IsStop(t) {
			return
		}
		ids = append(ids, t)
	})
	generated := len(ids)
	s.logGenSpeed(genStart, generated)
	finish := batchFinish(res, r.Context())

	msg := ChatMessage{Role: "assistant"}
	reasoning, rest := splitReasoning(s.tokenizer.DecodeRaw(ids))
	msg.ReasoningContent = reasoning
	if wantTools {
		if finish == "length" || finish == "cancelled" {
			if o := strings.LastIndex(rest, "<|tool_call>"); o > strings.LastIndex(rest, "<tool_call|>") {
				rest = rest[:o]
			}
		}
		content, calls := parseToolCalls(rest)
		if len(calls) > 0 {
			calls, _ = validateToolCalls(calls, tools)
		}
		msg.Content = strings.TrimSpace(stripMarkers(content))
		if len(calls) > 0 {
			msg.ToolCalls = calls
			if finish != "cancelled" {
				finish = "tool_calls"
			}
		}
	} else {
		msg.Content = strings.TrimSpace(stripMarkers(rest))
	}

	if legacy {
		text := msg.Content
		if msg.ReasoningContent != "" {
			text = msg.ReasoningContent + text
		}
		writeJSON(w, http.StatusOK, CompletionResponse{
			ID:      fmt.Sprintf("cmpl-%d", time.Now().UnixNano()),
			Object:  "text_completion",
			Created: time.Now().Unix(),
			Model:   s.modelName,
			Choices: []CompletionChoice{{Index: 0, Text: text, FinishReason: finish}},
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
		Choices: []Choice{{Index: 0, Message: msg, FinishReason: finish}},
		Usage: Usage{
			PromptTokens: promptTokens, CompletionTokens: generated,
			TotalTokens: promptTokens + generated,
		},
	})
}

// batchFinish maps a scheduler Result (and the request context) to the OpenAI
// finish_reason string.
func batchFinish(res batch.Result, _ context.Context) string {
	switch res.Reason {
	case batch.FinishStop:
		return "stop"
	case batch.FinishLength:
		return "length"
	case batch.FinishCancelled:
		return "cancelled"
	case batch.FinishError, batch.FinishShutdown:
		// A truncated turn: make it visible rather than looking like a clean stop.
		return "length"
	default:
		return "stop"
	}
}

// finishStream emits the terminal finish_reason + usage chunk and [DONE] on an
// already-begun SSE stream (used by the early-exit paths that never generate).
func (s *Server) finishStream(sse *sseWriter, finish string, promptTokens, completion int) {
	usage := &Usage{
		PromptTokens:     promptTokens,
		CompletionTokens: completion,
		TotalTokens:      promptTokens + completion,
	}
	if sse.legacy {
		sse.event(CompletionStreamResponse{
			ID: sse.id, Object: sse.object, Created: sse.created, Model: s.modelName,
			Choices: []CompletionStreamChoice{{Index: 0, Text: "", FinishReason: finish}},
			Usage:   usage,
		})
	} else {
		sse.event(StreamResponse{
			ID: sse.id, Object: sse.object, Created: sse.created, Model: s.modelName,
			Choices: []StreamChoice{{Index: 0, Delta: Delta{}, FinishReason: finish}},
			Usage:   usage,
		})
	}
	sse.done()
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

// handleHealth is LOCK-FREE for the same reason as /metrics: a health endpoint
// that blocks behind an in-flight request's kv lock reports "dead" precisely
// while the server is doing its job (engine.NTokens takes the engine mutex,
// held across the whole prefill — use the atomic mirror instead).
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	hits, misses, hitRate := s.kv.Stats()
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status": "ok",
		"kv_cache": map[string]interface{}{
			"prefix_hits":    hits,
			"prefix_misses":  misses,
			"token_hit_rate": hitRate,
			"cached_tokens":  s.lastUsed.Load(),
			"context_size":   s.engine.ContextSize(),
		},
	})
}

// handleReady is a readiness probe: unlike /health (liveness — "the process is
// up"), it confirms the server can actually serve a request. It checks the
// tokenizer is loaded and the engine reports a usable context window. An
// orchestrator routes traffic only when this returns 200; a 503 means "up but
// not serviceable" (e.g. tokenizer init failed and the server is running blind).
func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	var reasons []string
	if s.tokenizer == nil {
		reasons = append(reasons, "tokenizer not loaded")
	}
	if s.engine == nil || s.engine.ContextSize() == 0 {
		reasons = append(reasons, "engine not initialized")
	}
	if len(reasons) > 0 {
		writeJSON(w, http.StatusServiceUnavailable, map[string]interface{}{
			"status": "not_ready", "reasons": reasons,
		})
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":       "ready",
		"context_size": s.engine.ContextSize(),
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
		msgs[i] = chat.Message{Role: m.Role, Content: m.Content, Reasoning: m.ReasoningContent}
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

// runSpec drives the speculative generation engine with the server-side guards
// shared by the streaming and non-streaming paths:
//
//   - client cancellation (checked per token via the engine callback);
//   - repetition-loop detection (true period detection on emitted ids — the
//     Req3 runaway burned 130s/8192 tokens in a cycle the drafter accelerated);
//   - a reasoning-channel token budget: when the model has thought for `budget`
//     tokens without closing the channel, the server stops the engine, COMMITS
//     a <channel|> token itself (Decode — safe here: the engine mutex is free
//     once GenerateSpecStream returns; calling Decode from inside the callback
//     would self-deadlock), and resumes generation so the turn still produces
//     an answer instead of an empty-content message.
//
// emit is the caller's per-token handler (the streaming state machine); nil for
// non-streaming. Returns all generated tokens (markers included, plus any
// injected <channel|>), a finish_reason — "stop", "length", "cancelled", or ""
// when emit asked to stop (the caller knows why) — and the first engine error.
// The prefix cache is synced with engine-committed tokens after every round;
// the caller must hold the kv lock.
func (s *Server) runSpec(ctx context.Context, params GenerationParams, logits []float32,
	stops []int32, emit func(int32) bool) ([]int32, string, error) {

	chOpen, chEnd := s.tokenizer.ChannelOpen, s.tokenizer.ChannelEnd
	tcOpen, tcEnd := s.tokenizer.ToolCallOpen, s.tokenizer.ToolCallEnd
	budget := s.thinkBudget
	if budget == 0 {
		budget = params.MaxTokens / 2 // auto: never let thinking eat the whole turn
	}
	seed := uint64(params.Seed)
	if params.Seed < 0 {
		seed = uint64(time.Now().UnixNano())
	}

	const (
		stopNone = iota
		stopCancelled
		stopLoop
		stopThink
		stopEmit
	)
	var all []int32
	detector := &cycleDetector{}
	thinkToks, inCh, inTC, thinkClosed := 0, false, false, false
	remaining := params.MaxTokens

	for remaining > 0 {
		why := stopNone
		history := s.kv.CurrentTokens()
		baseN := s.engine.NTokens()
		toks, _, err := s.engine.GenerateSpecStream(history, logits, remaining, stops,
			s.draftK, float32(params.Temperature), params.TopK,
			float32(params.TopP), float32(params.MinP), float32(params.RepeatPenalty), seed,
			func(t int32) bool {
				if ctx.Err() != nil {
					why = stopCancelled
					return true
				}
				// Channel/tool tracking for the thinking budget. Tool-call spans
				// are excluded: their tokens are not "thinking", and the budget
				// must never fire inside one (the force-closed <channel|> would
				// be committed into the middle of the buffered call body).
				switch {
				case tcOpen >= 0 && t == tcOpen:
					inTC = true
				case tcEnd >= 0 && t == tcEnd:
					inTC = false
				case chOpen >= 0 && t == chOpen:
					inCh = true
				case chEnd >= 0 && t == chEnd:
					inCh = false
				case inCh && !inTC:
					thinkToks++
				}
				if detector.push(t) {
					why = stopLoop
					return true
				}
				if emit != nil && emit(t) {
					why = stopEmit
					return true
				}
				if inCh && !inTC && !thinkClosed && budget > 0 && thinkToks >= budget {
					why = stopThink
					return true
				}
				return false
			})
		// Sync the prefix cache with the tokens actually committed to the engine
		// KV (a trailing emitted token may not be forwarded) so the next request
		// reuses this response's prefix.
		committed := s.engine.NTokens() - baseN
		appended := committed
		if appended > len(toks) {
			appended = len(toks)
		}
		for i := 0; i < appended; i++ {
			s.kv.AppendDecoded(toks[i])
		}
		all = append(all, toks...)
		remaining -= len(toks)
		seed++ // a resumed round must not replay the same random draws
		if err != nil {
			// Make truncation visible: a mid-stream engine failure must not
			// look like a clean completion to an agent client.
			return all, "length", err
		}

		switch why {
		case stopCancelled:
			return all, "cancelled", nil
		case stopLoop:
			log.Printf("fucina: WARNING: repetition loop detected after %d tokens — cutting generation", len(all))
			return all, "length", nil
		case stopEmit:
			return all, "", nil
		case stopThink:
			if chEnd < 0 || remaining <= 0 {
				return all, "length", nil
			}
			// The verify pass commits the WHOLE accepted run before the per-token
			// emission loop, so when the budget stopped the callback mid-run the
			// engine holds accepted-but-unemitted tokens beyond `appended`. Trim
			// them: the injected <channel|> must land exactly after the last
			// token recorded in cachedTokens, or the bookkeeping and the KV
			// CONTENT diverge silently and the next request reuses a corrupted
			// prefix (confirmed by review).
			if !s.engine.Rewind(baseN + appended) {
				return all, "length", nil
			}
			lg, derr := s.engine.Decode(chEnd) // commit <channel|> into the KV
			if derr != nil {
				return all, "length", derr
			}
			s.kv.AppendDecoded(chEnd)
			all = append(all, chEnd)
			remaining--
			detector.push(chEnd)
			inCh, thinkClosed = false, true
			if emit != nil && emit(chEnd) { // streaming state machine closes its channel
				return all, "", nil // it asked to stop — honor it
			}
			log.Printf("fucina: thinking budget (%d tokens) reached — force-closed the thought channel", budget)
			logits = lg
			continue
		default:
			// The engine stopped on its own: a stop token, or max_new exhausted.
			if n := len(toks); n > 0 {
				last := toks[n-1]
				for _, sid := range stops {
					if last == sid {
						return all, "stop", nil
					}
				}
			}
			if remaining <= 0 {
				return all, "length", nil // hit the cap mid-output: tell the client
			}
			return all, "stop", nil
		}
	}
	return all, "length", nil
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

	// The speculative fast-path streams through a per-token callback, so it
	// supports cancellation, loop detection, the thinking budget (runSpec) and
	// repeat-penalty (applied on-GPU in the engine's spec sampler). Only text
	// stop-strings still need the per-token CPU loop here (the non-stream path
	// trims them post-hoc rather than via a callback).
	useSpec := len(params.Stop) == 0

	if useSpec {
		// Speculative decoding (default): one weight pass per [g, draft...], same
		// output distribution as plain decode. Continues from the prefilled state.
		// Generation runs to end-of-turn (NOT the first tool-call end), so a turn
		// may carry MULTIPLE tool calls; parseToolCalls extracts them all.
		//
		// emit mirrors the streaming path's post-call cutoff: once ≥1 call has
		// closed, only whitespace and further calls may follow. The first prose
		// or tool marker after a call means the model is moving past its calls —
		// usually starting to HALLUCINATE the tool response (observed: fake file
		// contents / a <|tool_response> runaway). Stopping there keeps the turn's
		// KV clean (call ends the cached sequence, so the next request's
		// re-rendered prompt token-matches it) and saves the wasted tokens.
		var emit func(int32) bool
		tcOpen := s.tokenizer.ToolCallOpen
		if wantTools && tcOpen >= 0 {
			inTool, completed := false, 0
			emit = func(t int32) bool {
				if t == tcOpen {
					inTool = true
					return false
				}
				if inTool {
					if tcEnd >= 0 && t == tcEnd {
						inTool = false
						completed++
					}
					return false
				}
				if completed > 0 {
					if s.tokenizer.IsToolMarker(t) {
						return true
					}
					return strings.TrimSpace(s.tokenizer.DecodeRaw([]int32{t})) != ""
				}
				return false
			}
		}
		stops := []int32{s.tokenizer.EOS, s.tokenizer.EndOfTurn}
		var err error
		var specFinish string
		toks, specFinish, err = s.runSpec(ctx, params, logits, stops, emit)
		if err != nil {
			if len(toks) == 0 {
				http.Error(w, fmt.Sprintf("generation failed: %v", err), http.StatusInternalServerError)
				return
			}
			// Partial output: deliver what exists, but make the failure visible.
			log.Printf("fucina: generation error after %d tokens: %v", len(toks), err)
		}
		if specFinish != "" {
			finish = specFinish
		} else if emit != nil {
			// The post-call cutoff fired: the token that triggered it (first
			// prose/marker after the calls) is hallucination, not answer text —
			// drop everything after the last completed call so it can't leak
			// into message.content.
			for i := len(toks) - 1; i >= 0; i-- {
				if toks[i] == tcEnd {
					toks = toks[:i+1]
					break
				}
			}
		}
		generated = len(toks)
	} else {
		// Per-token decode loop with the CPU sampler: required for repeat-penalty
		// and text stop-sequences.
		rng := rand.New(rand.NewSource(params.Seed))
		detector := &cycleDetector{}
		capExit := true
		for generated < params.MaxTokens {
			if logits == nil {
				capExit = false
				break
			}
			if ctx.Err() != nil { // client aborted / steered away
				finish = "cancelled"
				capExit = false
				break
			}
			token, err := s.sampleToken(logits, params, rng)
			if err != nil || s.tokenizer.IsStop(token) {
				capExit = false
				break
			}
			toks = append(toks, token)
			generated++
			if detector.push(token) {
				log.Printf("fucina: WARNING: repetition loop detected after %d tokens — cutting generation", generated)
				finish = "length"
				capExit = false
				break
			}
			if wantTools && tcEnd >= 0 && token == tcEnd {
				capExit = false
				break
			}
			if hit, trimmed := stopHit(s.tokenizer.Decode(toks), params.Stop); hit {
				toks = s.tokenizer.Encode(trimmed, false, false)
				capExit = false
				break
			}
			var err2 error
			logits, err2 = s.engine.Decode(token)
			if err2 != nil {
				capExit = false
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
		if capExit && finish == "stop" {
			finish = "length" // ran into MaxTokens mid-output: tell the client
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
		if finish == "length" || finish == "cancelled" {
			// Truncated turn: never dispatch a trailing UNTERMINATED call whose
			// arguments were cut at an arbitrary token (mirror of the streaming
			// path's guard); complete earlier calls still parse.
			if o := strings.LastIndex(rest, "<|tool_call>"); o > strings.LastIndex(rest, "<tool_call|>") {
				rest = rest[:o]
			}
		}
		content, calls := parseToolCalls(rest)
		// Refuse to dispatch a call that violates its required-parameter schema
		// (e.g. web_search{"query":""}); answer with a clarification instead.
		var clar string
		if len(calls) > 0 {
			calls, clar = validateToolCalls(calls, params.Tools)
		}
		// parseToolCalls leaves non-call text as-is; with generation running to
		// end-of-turn the literal "<turn|>" marker would otherwise leak into
		// message.content (streaming never emits markers — keep parity).
		msg.Content = strings.TrimSpace(stripMarkers(content))
		if msg.Content == "" && clar != "" {
			msg.Content = clar
		}
		if len(calls) > 0 {
			msg.ToolCalls = calls
			if finish != "cancelled" {
				finish = "tool_calls"
			}
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

// streamResponse runs streaming generation over an already-begun SSE session
// (headers + role delta went out before the prefill; see serveCompletions).
// The caller (handler) is the lock OWNER and holds s.kv.Lock() for the whole
// request; this function runs under that lock and must NOT release it.
func (s *Server) streamResponse(ctx context.Context, sse *sseWriter, params GenerationParams, promptTokens int, logits []float32, wantTools bool) {
	legacy := sse.legacy
	generated := 0
	genStart := time.Now()
	ttftRecorded := false
	lastWrite := time.Now() // last time any bytes hit the wire (for the keep-alive)

	// emitContent streams a piece of visible text in the right wire shape for the
	// active endpoint (chat delta vs legacy text).
	emitContent := func(text string) {
		// First visible token marks time-to-first-token (the latency the user
		// actually feels). Recorded once per request.
		if !ttftRecorded {
			s.metrics.recordTTFT(time.Since(genStart))
			ttftRecorded = true
		}
		lastWrite = time.Now()
		if legacy {
			sse.event(CompletionStreamResponse{
				ID: sse.id, Object: sse.object, Created: sse.created, Model: s.modelName,
				Choices: []CompletionStreamChoice{{Index: 0, Text: text}},
			})
		} else {
			sse.event(StreamResponse{
				ID: sse.id, Object: sse.object, Created: sse.created, Model: s.modelName,
				Choices: []StreamChoice{{Index: 0, Delta: Delta{Content: text}}},
			})
		}
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
	completedCalls := 0         // tool calls fully captured this turn (multi-call support)
	toolToks := 0               // tokens buffered inside the CURRENT tool-call span

	// Bounds for the tool-call buffer. The Req3 incident: a repetition loop
	// INSIDE an unterminated tool call silently swallowed ~7950 of 8192 tokens —
	// no wire output, no warning, a dead turn. The cap turns that failure mode
	// into a visible, bounded one (the cycle detector in runSpec usually fires
	// first; this is the backstop for non-cyclic runaways).
	const maxToolToks = 2048
	const maxToolCalls = 8

	// processToken runs the streaming state machine for ONE generated token —
	// tool-call buffering, the reasoning channel, marker suppression, stop
	// sequences, SSE emission — and reports whether generation must stop after
	// it. It is shared by the speculative path (where the engine invokes it as
	// the per-token callback via runSpec) and the repeat-penalty fallback loop
	// below, so both paths stream byte-identical wire output.
	processToken := func(token int32) bool {
		if s.tokenizer.IsStop(token) {
			return true
		}
		rawToks = append(rawToks, token)
		generated++

		// A stalled client (write deadline expired) can't be streamed to; its
		// blocked writes would otherwise wedge the engine callback forever.
		if sse.stalled() {
			finish = "cancelled"
			return true
		}

		// Wall-clock keep-alive: a long run of SUPPRESSED tokens (reasoning-channel
		// labels, a long tool-call buffer) emits no wire bytes, so an idle-timeout
		// proxy could kill the connection mid-generation even though the server is
		// working. Ping on the handler goroutine (single-writer-safe — the heartbeat
		// goroutine was already joined before generation) when the gap grows.
		if time.Since(lastWrite) > sseKeepAlive {
			sse.ping()
			lastWrite = time.Now()
		}

		// Tool-call handling: when the model opens a call, stop streaming content
		// and buffer until the call closes — calls are emitted as structured
		// tool_calls deltas after the loop. A closed call does NOT stop the turn:
		// the model may emit further calls (multi/parallel tool calls); the turn
		// ends at end-of-turn, on the first prose after a call, or at the caps.
		if wantTools && tcOpen >= 0 && token == tcOpen && !inTool {
			// !inTool: a re-emitted open marker inside an unterminated span must
			// NOT reset the counter — that would defeat the maxToolToks backstop
			// for exactly the runaway it exists to bound.
			inTool = true
			toolToks = 0
		}
		if inTool {
			toolToks++
			if tcEnd >= 0 && token == tcEnd {
				inTool = false
				completedCalls++
				return completedCalls >= maxToolCalls
			}
			if toolToks > maxToolToks {
				log.Printf("fucina: WARNING: unterminated tool call exceeded %d buffered tokens — cutting generation (runaway inside a tool-call span)", maxToolToks)
				finish = "length"
				return true
			}
			if toolToks%64 == 0 {
				sse.ping() // liveness during the silently-buffered span
			}
			return false
		}

		// Reasoning channel: stream <|channel>thought … <channel|> as reasoning_content
		// (not content), so the client can show/hide the thinking block separately.
		if chOpen >= 0 && token == chOpen {
			if completedCalls > 0 {
				return true // calls captured and the model is moving on — dispatch them
			}
			inChannel = true
			channelLabel = true // next text token is the "thought" label — skip it
			return false
		}
		if inChannel {
			if chEnd >= 0 && token == chEnd {
				inChannel = false
				return false
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
					sse.event(StreamResponse{
						ID: sse.id, Object: sse.object, Created: sse.created, Model: s.modelName,
						Choices: []StreamChoice{{Index: 0, Delta: Delta{ReasoningContent: rtext}}},
					})
				}
			}
			return false
		}

		// Never stream the gemma tool/string markers to the client as visible text.
		// After ≥1 completed call a tool marker means the model is starting to
		// HALLUCINATE the tool response (observed: a <|tool_response> runaway
		// burning the whole token budget) — the calls are complete, dispatch
		// them. Before any call, just swallow the stray marker.
		if s.tokenizer.IsToolMarker(token) {
			return completedCalls > 0
		}

		tokenStr := s.tokenizer.Decode([]int32{token})

		// After ≥1 completed tool call, whitespace between/after calls is
		// swallowed and the first real prose ends the turn — the captured calls
		// are dispatched together (the gemma turn normally closes right after
		// its call block, so this costs nothing in the common case).
		if completedCalls > 0 {
			return strings.TrimSpace(tokenStr) != ""
		}

		// Stop-sequence handling: if appending this piece completes a stop string,
		// emit only the NOT-YET-STREAMED text up to the stop and finish. When part
		// of the stop string was already streamed in earlier tokens (digits split
		// across tokens, etc.), emitted is LONGER than the trimmed text — the old
		// TrimPrefix re-emitted the whole response in that case (double emission).
		if len(params.Stop) > 0 {
			if hit, trimmed := stopHit(emitted.String()+tokenStr, params.Stop); hit {
				if len(trimmed) > emitted.Len() {
					emitContent(trimmed[emitted.Len():])
				}
				return true
			}
		}

		if tokenStr != "" {
			emitted.WriteString(tokenStr)
			emitContent(tokenStr)
		}
		return false
	}

	{
		// Speculative fast path (now the ONLY streaming path): MTP/prompt-lookup
		// drafting with batched verify — one weight pass per accepted run instead
		// of one per token — every token streaming through processToken via the
		// engine's per-token callback. runSpec adds the shared guards:
		// cancellation, repetition-loop cut-off, and the thinking budget.
		// Repeat-penalty is applied on-GPU inside the engine's spec sampler, so
		// repeat_penalty != 1.0 no longer drops to a per-token CPU decode loop
		// (1 MB logits D2H + host top-k per token, and no drafting).
		stops := []int32{s.tokenizer.EOS, s.tokenizer.EndOfTurn}
		_, specFinish, err := s.runSpec(ctx, params, logits, stops, processToken)
		if err != nil {
			log.Printf("fucina: generation error after %d tokens: %v", generated, err)
		}
		if specFinish != "" {
			finish = specFinish
		}
	}

	// Emit any captured tool call(s) as a structured delta before the final chunk.
	if wantTools {
		raw := s.tokenizer.DecodeRaw(rawToks)
		if inTool && (finish == "length" || finish == "cancelled") {
			// The turn was TRUNCATED inside an unterminated call: its arguments
			// are arbitrarily-cut text (often the repetition cycle itself). The
			// lenient recovery in parseToolCalls must not dispatch it — drop the
			// trailing unterminated span; complete earlier calls still go out.
			if i := strings.LastIndex(raw, "<|tool_call>"); i >= 0 {
				raw = raw[:i]
			}
		}
		if _, calls := parseToolCalls(raw); len(calls) > 0 {
			// Drop calls that violate their required-parameter schema; if every
			// call is dropped, stream a clarification instead of a malformed call.
			calls, clar := validateToolCalls(calls, params.Tools)
			if len(calls) > 0 {
				deltas := make([]DeltaToolCall, len(calls))
				for i, c := range calls {
					deltas[i] = DeltaToolCall{Index: i, ID: c.ID, Type: c.Type, Function: c.Function}
				}
				sse.event(StreamResponse{
					ID: sse.id, Object: "chat.completion.chunk", Created: sse.created, Model: s.modelName,
					Choices: []StreamChoice{{Index: 0, Delta: Delta{ToolCalls: deltas}}},
				})
				if finish != "cancelled" {
					finish = "tool_calls"
				}
			} else if clar != "" {
				sse.event(StreamResponse{
					ID: sse.id, Object: "chat.completion.chunk", Created: sse.created, Model: s.modelName,
					Choices: []StreamChoice{{Index: 0, Delta: Delta{Content: clar}}},
				})
			}
		} else if inTool {
			// The stream ended inside an unterminated tool-call span and nothing
			// parseable was recovered: ~toolToks tokens of model output are being
			// dropped. Without this line the turn dies silently (the Req3 mode).
			log.Printf("fucina: WARNING: generation ended inside an unterminated tool call (%d tokens buffered, finish=%s) — no tool_calls emitted", toolToks, finish)
		}
	}

	s.logGenSpeed(genStart, generated)
	s.finishStream(sse, finish, promptTokens, generated)
}

// logGenSpeed logs generation throughput in tokens/second and records it for /metrics.
func (s *Server) logGenSpeed(start time.Time, generated int) {
	elapsed := time.Since(start)
	tps := 0.0
	if elapsed.Seconds() > 0 {
		tps = float64(generated) / elapsed.Seconds()
	}
	s.metrics.recordDecode(generated, elapsed.Seconds())
	log.Printf("fucina: generated %d tokens in %.2fs (%.1f tok/s)",
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
	// The status line is already written, so a mid-encode socket error can only
	// be logged, not recovered into a different response. Surface it so a
	// truncated /metrics, /health, or completion response isn't silent.
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("fucina: writeJSON encode failed: %v", err)
	}
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
	s.httpServer = &http.Server{
		Addr:    addr,
		Handler: mux,
		// ReadHeaderTimeout closes slowloris sockets; IdleTimeout reaps dead
		// keep-alive connections. ReadTimeout/WriteTimeout stay ZERO on purpose:
		// a WriteTimeout is measured from the end of the header read and would
		// kill long SSE streams (multi-minute generations); per-flush write
		// deadlines (sseWriter.flush) bound stalled clients instead.
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
		// Explicit (matches net/http's 1 MiB default) so the bound is intentional
		// rather than implicit — header size is part of the connection-flood surface.
		MaxHeaderBytes: 1 << 20,
	}
	log.Printf("fucina: server listening on %s", addr)
	// ErrServerClosed is the normal result of a graceful Stop()/Shutdown — report it
	// as a clean return so the caller falls through to its deferred engine teardown
	// (rather than os.Exit-ing mid-CUDA-call).
	if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return err
	}
	return nil
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
		log.Printf("fucina: graceful shutdown failed (%v), forcing close", err)
		if cerr := s.httpServer.Close(); cerr != nil {
			log.Printf("fucina: forced close failed: %v", cerr)
		}
	}
	// Drain the batch scheduler AFTER the HTTP server stops accepting requests,
	// so no in-flight handler is still trying to Submit. Shutdown blocks until
	// the owner goroutine has evicted every sequence (freeing its KV slots).
	if s.scheduler != nil {
		s.scheduler.Shutdown()
	}
}
