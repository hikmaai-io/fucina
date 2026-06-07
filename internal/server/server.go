package server

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/mauromedda/gem4d/internal/engine/cuda"
	"github.com/mauromedda/gem4d/internal/tokenizer"
)

// ─── Types ─────────────────────────────────────────────────────────

type Server struct {
	engine    *cuda.Engine
	tokenizer *tokenizer.Tokenizer
	kv           *KVCache
	modelName    string
	genParams    GenerationParams
	httpServer   *http.Server
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
	MaxTokens        int     `json:"max_tokens"`
	Stream           bool    `json:"stream"`
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature *float64      `json:"temperature"`
	TopP        *float64      `json:"top_p"`
	TopK        *int          `json:"top_k"`
	MinP        *float64      `json:"min_p"`
	Seed        *int64        `json:"seed"`
	Stream      bool          `json:"stream"`
	Stop        []string      `json:"stop"`
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
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
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
}

type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

type ModelInfo struct {
	ID     string `json:"id"`
	Object string `json:"object"`
	Owner  string `json:"created"`
}

func DefaultParams() GenerationParams {
	return GenerationParams{
		Temperature:     0.8,
		TopP:            0.95,
		TopK:            40,
		MinP:            0.05,
		Seed:            0,
		RepeatPenalty:   1.1,
		FrequencyPenalty: 0.0,
		PresencePenalty: 0.0,
		MaxTokens:       512,
		Stream:          false,
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

func (s *Server) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/v1/completions", s.handleCompletions)
	mux.HandleFunc("/v1/embeddings", s.handleEmbeddings)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/", s.handleNotFound)
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, ModelsResponse{
		Object: "list",
		Data:   []ModelInfo{{ID: s.modelName, Object: "model", Owner: "google"}},
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
	prompt := s.renderChatTemplate(req.Messages)
	if prompt == "" {
		http.Error(w, "empty prompt", http.StatusBadRequest)
		return
	}
	tokens := s.tokenizer.Encode(prompt, true, false)
	if len(tokens) == 0 {
		http.Error(w, "tokenization failed", http.StatusInternalServerError)
		return
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
	if req.MaxTokens > 0 {
		params.MaxTokens = req.MaxTokens
	}
	params.Stream = req.Stream

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
	log.Printf("gem4d: prefill %d tokens (%d cached, %d new) in %.2fs (%.1f tok/s)",
		promptTokens, pf.ReusedTokens, pf.NewTokens, prefillElapsed.Seconds(), prefillTPS)

	logits := pf.Logits

	if params.Stream {
		s.streamResponse(w, params, promptTokens, logits)
	} else {
		s.generateResponse(w, params, promptTokens, logits)
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

func (s *Server) renderChatTemplate(messages []ChatMessage) string {
	var sb strings.Builder
	for i, msg := range messages {
		switch msg.Role {
		case "system", "user":
			sb.WriteString(fmt.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n", msg.Content))
		case "assistant":
			if i == len(messages)-1 {
				sb.WriteString("<start_of_turn>model\n")
			} else {
				sb.WriteString(fmt.Sprintf("<start_of_turn>model\n%s<end_of_turn>\n", msg.Content))
			}
		}
	}
	if len(messages) > 0 && messages[len(messages)-1].Role != "assistant" {
		sb.WriteString("<start_of_turn>model\n")
	}
	return sb.String()
}

// generateResponse runs non-streaming generation. It assumes the caller holds
// s.kv.Lock() (acquired in the handler); it releases it when generation ends.
func (s *Server) generateResponse(w http.ResponseWriter, params GenerationParams, promptTokens int, logits []float32) {
	defer s.kv.Unlock()

	generated := 0
	var text strings.Builder
	rng := rand.New(rand.NewSource(params.Seed))

	genStart := time.Now()
	for generated < params.MaxTokens {
		if logits == nil {
			break
		}
		token, err := s.sampleToken(logits, params, rng)
		if err != nil || token == s.tokenizer.EOS {
			break
		}
		tokenStr := s.tokenizer.Decode([]int32{token})
		text.WriteString(tokenStr)
		var err2 error
		logits, err2 = s.engine.Decode(token)
		if err2 != nil {
			break
		}
		// Keep the cache token list in sync so this token (and the whole
		// reply) can be reused as a prefix by the next request.
		s.kv.AppendDecoded(token)
		generated++
	}
	logGenSpeed(genStart, generated)

	writeJSON(w, http.StatusOK, ChatResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   s.modelName,
		Choices: []Choice{{
			Index: 0, Message: ChatMessage{Role: "assistant", Content: text.String()}, FinishReason: "stop",
		}},
		Usage: Usage{
			PromptTokens: promptTokens, CompletionTokens: generated,
			TotalTokens: promptTokens + generated,
		},
	})
}

func (s *Server) streamResponse(w http.ResponseWriter, params GenerationParams, promptTokens int, logits []float32) {
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

	for generated < params.MaxTokens {
		if logits == nil {
			break
		}
		token, err := s.sampleToken(logits, params, rng)
		if err != nil || token == s.tokenizer.EOS {
			break
		}
		tokenStr := s.tokenizer.Decode([]int32{token})
		writeSSE(w, StreamResponse{
			ID: completionID, Object: "chat.completion.chunk", Created: created, Model: s.modelName,
			Choices: []StreamChoice{{Index: 0, Delta: Delta{Content: tokenStr}}},
		})
		flusher.Flush()

		logits, err = s.engine.Decode(token)
		if err != nil {
			break
		}
		s.kv.AppendDecoded(token)
		generated++
	}

	logGenSpeed(genStart, generated)

	writeSSE(w, StreamResponse{
		ID: completionID, Object: "chat.completion.chunk", Created: created, Model: s.modelName,
		Choices: []StreamChoice{{Index: 0, Delta: Delta{}, FinishReason: "stop"}},
	})
	flusher.Flush()
}

// logGenSpeed logs generation throughput in tokens/second.
func logGenSpeed(start time.Time, generated int) {
	elapsed := time.Since(start)
	tps := 0.0
	if elapsed.Seconds() > 0 {
		tps = float64(generated) / elapsed.Seconds()
	}
	log.Printf("gem4d: generated %d tokens in %.2fs (%.1f tok/s)",
		generated, elapsed.Seconds(), tps)
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
		// Penalize tokens in the current sequence (prompt + generated). The kv
		// lock is held for the whole request, so reading CurrentTokens is safe.
		for _, past := range s.kv.CurrentTokens() {
			if int(past) < n {
				if scaled[past] < 0 {
					scaled[past] *= params.RepeatPenalty
				} else {
					scaled[past] /= params.RepeatPenalty
				}
			}
		}
	}
	maxLogit := scaled[0]
	for _, v := range scaled {
		if v > maxLogit {
			maxLogit = v
		}
	}
	probs := make([]float64, n)
	var sum float64
	for i, v := range scaled {
		probs[i] = math.Exp(v - maxLogit)
		sum += probs[i]
	}

	// Top-K
	if params.TopK > 0 && params.TopK < n {
		type pair struct{ idx int; p float64 }
		sorted := make([]pair, n)
		for i := 0; i < n; i++ {
			sorted[i] = pair{i, probs[i]}
		}
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				if sorted[j].p > sorted[i].p {
					sorted[i], sorted[j] = sorted[j], sorted[i]
				}
			}
		}
		threshold := sorted[params.TopK].p
		for i := range probs {
			if probs[i] < threshold {
				probs[i] = 0
			}
		}
		sum = 0
		for _, v := range probs {
			sum += v
		}
	}

	// Top-P
	if params.TopP > 0 && params.TopP < 1.0 {
		type pair struct{ idx int; p float64 }
		sorted := make([]pair, n)
		for i := 0; i < n; i++ {
			sorted[i] = pair{i, probs[i]}
		}
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				if sorted[j].p > sorted[i].p {
					sorted[i], sorted[j] = sorted[j], sorted[i]
				}
			}
		}
		cum := 0.0
		for _, p := range sorted {
			if cum >= params.TopP {
				probs[p.idx] = 0
			} else {
				cum += p.p
			}
		}
		sum = 0
		for _, v := range probs {
			sum += v
		}
	}

	if sum <= 0 {
		return int32(cuda.Argmax(logits)), nil
	}
	r := rng.Float64() * sum
	cum := 0.0
	for i, p := range probs {
		cum += p
		if r <= cum {
			return int32(i), nil
		}
	}
	return int32(cuda.Argmax(logits)), nil
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
