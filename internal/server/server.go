package server

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
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

// modelTurnOpen opens an assistant turn and pre-fills an empty thought channel
// (thinking disabled by default), so the model does not stream its reasoning.
const modelTurnOpen = "<|turn>model\n<|channel>thought\n<channel|>"

// renderChatTemplate builds the gemma-4 prompt. The real vocab uses <|turn> /
// <turn|> delimiters (NOT <start_of_turn>/<end_of_turn>, which are not tokens).
func (s *Server) renderChatTemplate(messages []ChatMessage) string {
	var sb strings.Builder
	for i, msg := range messages {
		switch msg.Role {
		case "system":
			fmt.Fprintf(&sb, "<|turn>system\n%s<turn|>\n", msg.Content)
		case "user":
			fmt.Fprintf(&sb, "<|turn>user\n%s<turn|>\n", msg.Content)
		case "assistant":
			if i == len(messages)-1 {
				sb.WriteString(modelTurnOpen)
			} else {
				fmt.Fprintf(&sb, "<|turn>model\n%s<turn|>\n", msg.Content)
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
		if err != nil || s.tokenizer.IsStop(token) {
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
		if err != nil || s.tokenizer.IsStop(token) {
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
