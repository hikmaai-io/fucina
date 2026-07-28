package server

// ignore_eos / min_tokens (vLLM extensions) — host-side generation-length
// contract tests: request validation, single-flight spec-path behavior, and
// the continuous-batching path, including the usage invariant
// total_tokens == prompt_tokens + completion_tokens.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/hikmaai-io/fucina/internal/server/batch"
)

func postChat(t *testing.T, srv *Server, body map[string]interface{}) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, body))
	mux(srv).ServeHTTP(rec, req)
	return rec
}

func decodeErrorParam(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	var envelope struct {
		Error struct {
			Type  string `json:"type"`
			Param string `json:"param"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("bad error json: %v (body=%s)", err, rec.Body.String())
	}
	if envelope.Error.Type != "invalid_request_error" {
		t.Errorf("error.type=%q want invalid_request_error", envelope.Error.Type)
	}
	return envelope.Error.Param
}

// ─── validation ────────────────────────────────────────────────────────────

func TestMinTokensNegativeRejected(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	rec := postChat(t, srv, map[string]interface{}{
		"messages":   []map[string]string{{"role": "user", "content": "hi"}},
		"min_tokens": -1,
	})
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d want 400 (body=%s)", rec.Code, rec.Body.String())
	}
	if p := decodeErrorParam(t, rec); p != "min_tokens" {
		t.Errorf("error.param=%q want min_tokens", p)
	}
}

func TestMinTokensExceedingMaxTokensRejected(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	rec := postChat(t, srv, map[string]interface{}{
		"messages":   []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens": 4,
		"min_tokens": 5,
	})
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d want 400 (body=%s)", rec.Code, rec.Body.String())
	}
	if p := decodeErrorParam(t, rec); p != "min_tokens" {
		t.Errorf("error.param=%q want min_tokens", p)
	}
}

func TestIgnoreEOSWithResponseFormatRejected(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	rec := postChat(t, srv, map[string]interface{}{
		"messages":        []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens":      8,
		"ignore_eos":      true,
		"response_format": map[string]string{"type": "json_object"},
	})
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d want 400 (body=%s)", rec.Code, rec.Body.String())
	}
	if p := decodeErrorParam(t, rec); p != "response_format" {
		t.Errorf("error.param=%q want response_format", p)
	}
}

func TestMinTokensWithResponseFormatRejected(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	rec := postChat(t, srv, map[string]interface{}{
		"messages":        []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens":      8,
		"min_tokens":      2,
		"response_format": map[string]string{"type": "json_object"},
	})
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d want 400 (body=%s)", rec.Code, rec.Body.String())
	}
	if p := decodeErrorParam(t, rec); p != "response_format" {
		t.Errorf("error.param=%q want response_format", p)
	}
}

// ─── single-flight speculative path ────────────────────────────────────────

// TestIgnoreEOSSingleFlightRunsToMaxTokens: the script contains an EOS in the
// middle. Default behavior stops there (completion=2, finish=stop); with
// ignore_eos the engine receives no stop ids, generation runs to max_tokens
// (completion=3, finish=length), and the usage invariant holds.
func TestIgnoreEOSSingleFlightRunsToMaxTokens(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	script := []int32{idx["▁hello"], tk.EOS, idx["▁world"]}

	// Baseline: EOS terminates.
	{
		f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: script}
		srv := New(f, tk)
		srv.SetLogLevel("warn")
		rec := postChat(t, srv, map[string]interface{}{
			"messages":    []map[string]string{{"role": "user", "content": "hi"}},
			"max_tokens":  3,
			"temperature": 0,
		})
		if rec.Code != http.StatusOK {
			t.Fatalf("baseline status=%d (body=%s)", rec.Code, rec.Body.String())
		}
		var resp ChatResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatal(err)
		}
		if resp.Usage.CompletionTokens != 2 || resp.Choices[0].FinishReason != "stop" {
			t.Fatalf("baseline completion=%d finish=%q want 2/stop",
				resp.Usage.CompletionTokens, resp.Choices[0].FinishReason)
		}
	}

	// ignore_eos: the mid-script EOS is an ordinary token; run to max_tokens.
	{
		f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: script}
		srv := New(f, tk)
		srv.SetLogLevel("warn")
		rec := postChat(t, srv, map[string]interface{}{
			"messages":    []map[string]string{{"role": "user", "content": "hi"}},
			"max_tokens":  3,
			"temperature": 0,
			"ignore_eos":  true,
		})
		if rec.Code != http.StatusOK {
			t.Fatalf("status=%d (body=%s)", rec.Code, rec.Body.String())
		}
		var resp ChatResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatal(err)
		}
		if resp.Usage.CompletionTokens != 3 {
			t.Errorf("completion_tokens=%d want 3 (EOS must not terminate)", resp.Usage.CompletionTokens)
		}
		if resp.Choices[0].FinishReason != "length" {
			t.Errorf("finish=%q want length", resp.Choices[0].FinishReason)
		}
		if resp.Usage.TotalTokens != resp.Usage.PromptTokens+resp.Usage.CompletionTokens {
			t.Errorf("total=%d != prompt+completion (%d+%d)",
				resp.Usage.TotalTokens, resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
		}
	}
}

// TestMinTokensSingleFlightSuppressesEarlyEOS: EOS at position 2 must not end
// generation while fewer than min_tokens=3 tokens exist; the EOS at position 4
// (three tokens precede it) terminates with finish=stop.
func TestMinTokensSingleFlightSuppressesEarlyEOS(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	script := []int32{idx["▁hello"], tk.EOS, idx["▁world"], tk.EOS, idx["▁hello"]}
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: script}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	rec := postChat(t, srv, map[string]interface{}{
		"messages":    []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens":  16,
		"temperature": 0,
		"min_tokens":  3,
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d (body=%s)", rec.Code, rec.Body.String())
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Usage.CompletionTokens != 4 {
		t.Errorf("completion_tokens=%d want 4 (early EOS suppressed, second EOS terminal)",
			resp.Usage.CompletionTokens)
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish=%q want stop", resp.Choices[0].FinishReason)
	}
	if resp.Usage.TotalTokens != resp.Usage.PromptTokens+resp.Usage.CompletionTokens {
		t.Errorf("total=%d != prompt+completion (%d+%d)",
			resp.Usage.TotalTokens, resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	}
}

// ─── continuous-batching path ──────────────────────────────────────────────

// eosOnlyBatchEngine is a minimal BatchEngine whose every sampled token is the
// tokenizer EOS — the worst case for a generation-length contract.
type eosOnlyBatchEngine struct {
	mu       sync.Mutex
	eos      int32
	capacity int
	next     int
	free     []int
	live     map[int]bool
}

func newEOSOnlyBatchEngine(eos int32, capacity int) *eosOnlyBatchEngine {
	return &eosOnlyBatchEngine{eos: eos, capacity: capacity, live: make(map[int]bool)}
}

func (e *eosOnlyBatchEngine) AddSeq(prompt []int32, _ batch.SeqParams) (int, int32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	var slot int
	if n := len(e.free); n > 0 {
		slot = e.free[n-1]
		e.free = e.free[:n-1]
	} else {
		slot = e.next
		e.next++
	}
	e.live[slot] = true
	return slot, e.eos, nil
}

func (e *eosOnlyBatchEngine) StepBatch(active []int32, inputs []int32) ([][]int32, error) {
	out := make([][]int32, len(active))
	for i := range active {
		out[i] = []int32{e.eos}
	}
	return out, nil
}

func (e *eosOnlyBatchEngine) RemoveSeq(slot int) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.live, slot)
	e.free = append(e.free, slot)
	return nil
}

func (e *eosOnlyBatchEngine) Capacity() int { return e.capacity }

func (e *eosOnlyBatchEngine) Supported() bool { return true }

// TestIgnoreEOSBatchRunsToMaxTokens: on the scheduler path ignore_eos wires
// Stops: nil, so an engine that emits ONLY EOS still generates exactly
// max_tokens tokens with finish=length and a consistent usage block.
func TestIgnoreEOSBatchRunsToMaxTokens(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	if !srv.SetBatchEngine(newEOSOnlyBatchEngine(srv.tokenizer.EOS, 1)) {
		t.Fatal("SetBatchEngine failed")
	}
	defer srv.Stop()

	rec := postChat(t, srv, map[string]interface{}{
		"messages":    []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens":  5,
		"temperature": 0,
		"ignore_eos":  true,
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d (body=%s)", rec.Code, rec.Body.String())
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Usage.CompletionTokens != 5 {
		t.Errorf("completion_tokens=%d want exactly max_tokens=5", resp.Usage.CompletionTokens)
	}
	if resp.Choices[0].FinishReason != "length" {
		t.Errorf("finish=%q want length", resp.Choices[0].FinishReason)
	}
	if resp.Usage.TotalTokens != resp.Usage.PromptTokens+resp.Usage.CompletionTokens {
		t.Errorf("total=%d != prompt+completion (%d+%d)",
			resp.Usage.TotalTokens, resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	}
	if resp.Usage.PromptTokens <= 0 {
		t.Errorf("prompt_tokens=%d want > 0", resp.Usage.PromptTokens)
	}
}

// TestMinTokensBatchSuppressesEarlyEOS: with min_tokens=3 the first three EOS
// tokens are ordinary output (kept and counted); the fourth (three tokens
// precede it) is the terminal stop and is dropped from the rendered output per
// the existing batch-path convention, so completion_tokens == min_tokens.
func TestMinTokensBatchSuppressesEarlyEOS(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	if !srv.SetBatchEngine(newEOSOnlyBatchEngine(srv.tokenizer.EOS, 1)) {
		t.Fatal("SetBatchEngine failed")
	}
	defer srv.Stop()

	rec := postChat(t, srv, map[string]interface{}{
		"messages":    []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens":  10,
		"temperature": 0,
		"min_tokens":  3,
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d (body=%s)", rec.Code, rec.Body.String())
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Usage.CompletionTokens != 3 {
		t.Errorf("completion_tokens=%d want 3 (min_tokens floor; terminal stop dropped)", resp.Usage.CompletionTokens)
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("finish=%q want stop", resp.Choices[0].FinishReason)
	}
	if resp.Usage.TotalTokens != resp.Usage.PromptTokens+resp.Usage.CompletionTokens {
		t.Errorf("total=%d != prompt+completion (%d+%d)",
			resp.Usage.TotalTokens, resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	}
}
