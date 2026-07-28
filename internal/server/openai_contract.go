package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"reflect"
	"runtime/debug"
	"sort"
	"strings"
	"sync"

	"github.com/hikmaai-io/fucina/internal/model"
)

const openAIRequestBodyLimit = 64 << 20

// BuildVersion is returned by GET /version. Release builds may set it with
// -ldflags; development builds derive it from Go module build information.
var BuildVersion = detectBuildVersion()

// StreamOptions implements the OpenAI/vLLM stream_options subset in Tier A.
// continuous_usage_stats is intentionally not accepted as implemented: that is
// a vLLM extension outside Tier A.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// OpenAIErrorEnvelope is the error shape shared by all OpenAI-compatible routes.
type OpenAIErrorEnvelope struct {
	Error OpenAIError `json:"error"`
}

type OpenAIError struct {
	Message string      `json:"message"`
	Type    string      `json:"type"`
	Param   interface{} `json:"param,omitempty"`
	Code    interface{} `json:"code,omitempty"`
}

type TokenizeRequest struct {
	Model            string `json:"model,omitempty"`
	Prompt           string `json:"prompt"`
	AddSpecialTokens *bool  `json:"add_special_tokens,omitempty"`
}

type TokenizeResponse struct {
	Count       int     `json:"count"`
	MaxModelLen int     `json:"max_model_len"`
	Tokens      []int32 `json:"tokens"`
}

type DetokenizeRequest struct {
	Model  string  `json:"model,omitempty"`
	Tokens []int64 `json:"tokens"`
}

type DetokenizeResponse struct {
	Prompt string `json:"prompt"`
}

type VersionResponse struct {
	Version string `json:"version"`
}

type ModelDetailResponse struct {
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
	model.DescriptorSnapshot
}

var strictOpenAIServers sync.Map // map[*Server]bool; servers live for process lifetime

// SetOpenAIStrict controls unknown-field handling. Strict mode rejects unknown
// top-level request fields; compatibility mode (the default) accepts and logs.
func (s *Server) SetOpenAIStrict(strict bool) { strictOpenAIServers.Store(s, strict) }

func (s *Server) openAIStrict() bool {
	if v, ok := strictOpenAIServers.Load(s); ok {
		return v.(bool)
	}
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FUCINA_OPENAI_STRICT"))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func detectBuildVersion() string {
	if info, ok := debug.ReadBuildInfo(); ok && info.Main.Version != "" && info.Main.Version != "(devel)" {
		return info.Main.Version
	}
	return "dev"
}

// resolveCompletionLimit applies OpenAI/vLLM precedence: the newer
// max_completion_tokens wins whenever it is present, even when max_tokens is
// also present. max_tokens retains the server's historical zero=omitted shape.
func resolveCompletionLimit(maxTokens int, maxCompletionTokens *int) (value int, explicit bool, err error) {
	if maxCompletionTokens != nil {
		if *maxCompletionTokens <= 0 {
			return 0, true, fmt.Errorf("max_completion_tokens must be greater than zero")
		}
		return *maxCompletionTokens, true, nil
	}
	if maxTokens < 0 {
		return 0, true, fmt.Errorf("max_tokens must be greater than zero")
	}
	return maxTokens, maxTokens > 0, nil
}

func validateStreamOptions(stream bool, options *StreamOptions) error {
	if options != nil && !stream {
		return fmt.Errorf("stream_options can only be defined when stream is true")
	}
	return nil
}

// decodeOpenAIJSON detects unknown top-level fields without making compatibility
// mode brittle. It logs in the default mode and returns an actionable error in
// strict mode, then uses the destination's normal JSON unmarshalling behavior.
func decodeOpenAIJSON(body []byte, dst interface{}, strict bool) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(body, &raw); err != nil {
		return err
	}
	allowed := jsonFieldNames(reflect.TypeOf(dst))
	unknown := make([]string, 0)
	for key := range raw {
		if _, ok := allowed[key]; !ok {
			unknown = append(unknown, key)
		}
	}
	if len(unknown) != 0 {
		sort.Strings(unknown)
		if strict {
			return fmt.Errorf("unknown field(s): %s", strings.Join(unknown, ", "))
		}
		log.Printf("fucina: WARNING: OpenAI request ignored unknown field(s): %s", strings.Join(unknown, ", "))
	}
	return json.Unmarshal(body, dst)
}

func jsonFieldNames(t reflect.Type) map[string]struct{} {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	out := make(map[string]struct{})
	if t.Kind() != reflect.Struct {
		return out
	}
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		name := strings.Split(f.Tag.Get("json"), ",")[0]
		if name == "-" {
			continue
		}
		if name == "" {
			if f.Anonymous {
				for nested := range jsonFieldNames(f.Type) {
					out[nested] = struct{}{}
				}
				continue
			}
			name = f.Name
		}
		out[name] = struct{}{}
	}
	return out
}

// validateSupportedContentParts rejects modalities the loaded text-only engines
// cannot execute. This runs on the raw body before ChatMessage flattening so an
// image/audio/file part can never be silently discarded.
func validateSupportedContentParts(body []byte) error {
	var req struct {
		Messages []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil // the main decoder returns the canonical malformed-JSON error
	}
	for i, message := range req.Messages {
		trimmed := bytes.TrimSpace(message.Content)
		if len(trimmed) == 0 || trimmed[0] != '[' {
			continue
		}
		var parts []struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(trimmed, &parts); err != nil {
			continue
		}
		for j, part := range parts {
			if part.Type != "" && part.Type != "text" {
				return fmt.Errorf("messages[%d].content[%d]: unsupported content part type %q", i, j, part.Type)
			}
		}
	}
	return nil
}

func writeOpenAIError(w http.ResponseWriter, status int, typ, message string, param, code interface{}) {
	writeJSON(w, status, OpenAIErrorEnvelope{Error: OpenAIError{
		Message: message,
		Type:    typ,
		Param:   param,
		Code:    code,
	}})
}

func writeOpenAIRequestError(w http.ResponseWriter, message string, param interface{}) {
	writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", message, param, "invalid_request")
}

func requireMethod(w http.ResponseWriter, r *http.Request, method string) bool {
	if r.Method == method {
		return true
	}
	w.Header().Set("Allow", method)
	writeOpenAIError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method not allowed", nil, "method_not_allowed")
	return false
}

func (s *Server) decodeContractRequest(w http.ResponseWriter, r *http.Request, dst interface{}) bool {
	r.Body = http.MaxBytesReader(w, r.Body, openAIRequestBodyLimit)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeOpenAIRequestError(w, "request body is too large or unreadable", nil)
		return false
	}
	if err := decodeOpenAIJSON(body, dst, s.openAIStrict()); err != nil {
		writeOpenAIRequestError(w, "invalid JSON: "+err.Error(), nil)
		return false
	}
	return true
}

func (s *Server) modelAccepted(name string) bool {
	if name == "" || name == s.modelName {
		return true
	}
	return s.modelStore != nil && func() bool {
		_, ok := s.modelStore.Get(name)
		return ok
	}()
}

func (s *Server) handleModelDetail(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	id := strings.TrimPrefix(r.URL.Path, "/v1/models/")
	if id == "" || s.modelStore == nil {
		writeOpenAIError(w, http.StatusNotFound, "invalid_request_error", fmt.Sprintf("model %q not found", id), "model", "model_not_found")
		return
	}
	descriptor, ok := s.modelStore.Get(id)
	if !ok {
		writeOpenAIError(w, http.StatusNotFound, "invalid_request_error", fmt.Sprintf("model %q not found", id), "model", "model_not_found")
		return
	}
	writeJSON(w, http.StatusOK, ModelDetailResponse{
		Object:             "model",
		OwnedBy:            "fucina",
		DescriptorSnapshot: descriptor.Snapshot(),
	})
}

func (s *Server) handleTokenize(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	var req TokenizeRequest
	if !s.decodeContractRequest(w, r, &req) {
		return
	}
	if !s.modelAccepted(req.Model) {
		writeOpenAIError(w, http.StatusNotFound, "invalid_request_error", fmt.Sprintf("model %q not found", req.Model), "model", "model_not_found")
		return
	}
	if s.tokenizer == nil {
		writeOpenAIError(w, http.StatusServiceUnavailable, "server_error", "tokenizer not loaded", nil, "tokenizer_unavailable")
		return
	}
	addSpecial := true
	if req.AddSpecialTokens != nil {
		addSpecial = *req.AddSpecialTokens
	}
	tokens := s.tokenizer.Encode(req.Prompt, addSpecial, false)
	writeJSON(w, http.StatusOK, TokenizeResponse{
		Count:       len(tokens),
		MaxModelLen: int(s.engine.ContextSize()),
		Tokens:      tokens,
	})
}

func (s *Server) handleDetokenize(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	var req DetokenizeRequest
	if !s.decodeContractRequest(w, r, &req) {
		return
	}
	if !s.modelAccepted(req.Model) {
		writeOpenAIError(w, http.StatusNotFound, "invalid_request_error", fmt.Sprintf("model %q not found", req.Model), "model", "model_not_found")
		return
	}
	if s.tokenizer == nil {
		writeOpenAIError(w, http.StatusServiceUnavailable, "server_error", "tokenizer not loaded", nil, "tokenizer_unavailable")
		return
	}
	tokens := make([]int32, len(req.Tokens))
	for i, token := range req.Tokens {
		if token < 0 || token > math.MaxInt32 || token >= int64(s.tokenizer.NumTokens()) {
			writeOpenAIRequestError(w, fmt.Sprintf("tokens[%d] is outside the tokenizer vocabulary", i), fmt.Sprintf("tokens[%d]", i))
			return
		}
		tokens[i] = int32(token)
	}
	writeJSON(w, http.StatusOK, DetokenizeResponse{Prompt: s.tokenizer.Decode(tokens)})
}

func (s *Server) handleVersion(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	writeJSON(w, http.StatusOK, VersionResponse{Version: BuildVersion})
}
