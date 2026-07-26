package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/hikmaai-io/fucina/internal/server/batch"
)

// TestResponseFormatJSONObject drives a full /v1/chat/completions request with
// response_format {type:"json_object"} through the single-flight constrained path.
// The scripted engine peaks its logits at "{" then "}" (byte tokens); the grammar
// mask keeps those legal and forbids everything else, so the output is a valid JSON
// object. It also asserts the request took the constrained host-sampling path (NOT
// the speculative fast path, which cannot apply a host logit mask).
func TestResponseFormatJSONObject(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	// "{" = 0x7B, "}" = 0x7D — a well-formed empty object.
	script := []int32{idx["<0x7B>"], idx["<0x7D>"]}
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: script}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":        []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens":      16,
		"temperature":     0,
		"response_format": map[string]string{"type": "json_object"},
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 (body=%s)", rec.Code, rec.Body.String())
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	got := resp.Choices[0].Message.Content
	if got != "{}" {
		t.Errorf("content=%q want %q", got, "{}")
	}
	// The content must itself be valid JSON — the whole point of the constraint.
	var any interface{}
	if err := json.Unmarshal([]byte(got), &any); err != nil {
		t.Errorf("constrained output is not valid JSON: %v (%q)", err, got)
	}
	// Constrained requests must NOT use the on-device speculative sampler (it can't
	// take a host mask); they run the per-token host-sampling loop instead.
	if f.specCalls != 0 {
		t.Errorf("spec path used %d times under response_format; want 0 (constrained host path)", f.specCalls)
	}
	if f.decodeCalls < 2 {
		t.Errorf("engine.Decode called %d times; want >=2 (host constrained loop)", f.decodeCalls)
	}
}

// TestResponseFormatForceCloseAtCap verifies that when max_tokens is exhausted before
// the JSON structure completes, the constraint force-closes it so the output stays a
// valid, parseable JSON object rather than a truncated fragment.
func TestResponseFormatForceCloseAtCap(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	// Open an object and a key, then run out of budget mid-structure: "{" then a
	// key string that never closes the object. Close() must append the missing "}".
	script := []int32{idx["<0x7B>"], idx["<0x22>"], idx["<0x61>"], idx["<0x22>"], idx["<0x3A>"], idx["<0x31>"]}
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: script}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":        []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens":      6, // cap hit before the object closes
		"temperature":     0,
		"response_format": map[string]string{"type": "json_object"},
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 (body=%s)", rec.Code, rec.Body.String())
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	got := resp.Choices[0].Message.Content
	var obj map[string]interface{}
	if err := json.Unmarshal([]byte(got), &obj); err != nil {
		t.Fatalf("force-closed output is not a valid JSON object: %v (%q)", err, got)
	}
	if v, ok := obj["a"]; !ok || v != float64(1) {
		t.Errorf("force-closed object=%v want {\"a\":1}", obj)
	}
}

// TestResponseFormatJSONSchema drives response_format {type:"json_schema"} with a schema
// requiring an integer property "a". The scripted engine peaks at the bytes of {"a":1};
// the schema constraint keeps exactly those legal, so the output matches the schema.
func TestResponseFormatJSONSchema(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	b := func(c byte) int32 {
		return idx["<0x"+string([]byte{"0123456789ABCDEF"[(c>>4)&0xF], "0123456789ABCDEF"[c&0xF]})+">"]
	}
	script := []int32{b('{'), b('"'), b('a'), b('"'), b(':'), b('1'), b('}')}
	f := &fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS, script: script}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":    []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens":  32,
		"temperature": 0,
		"response_format": map[string]interface{}{
			"type": "json_schema",
			"json_schema": map[string]interface{}{
				"name":   "answer",
				"strict": true,
				"schema": map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{"a": map[string]interface{}{"type": "integer"}},
					"required":   []string{"a"},
				},
			},
		},
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 (body=%s)", rec.Code, rec.Body.String())
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	var obj struct {
		A int `json:"a"`
	}
	got := resp.Choices[0].Message.Content
	if err := json.Unmarshal([]byte(got), &obj); err != nil {
		t.Fatalf("schema-constrained output not valid JSON: %v (%q)", err, got)
	}
	if obj.A != 1 {
		t.Errorf("content=%q want {\"a\":1}", got)
	}
	if f.specCalls != 0 {
		t.Errorf("spec path used %d times under json_schema; want 0", f.specCalls)
	}
}

// TestResponseFormatBadSchema asserts a malformed json_schema is rejected up front (400)
// rather than silently ignored.
func TestResponseFormatBadSchema(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":   []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens": 16,
		"response_format": map[string]interface{}{
			"type": "json_schema",
			"json_schema": map[string]interface{}{
				"name":   "bad",
				"schema": map[string]interface{}{"type": "object", "required": []string{"missing"}},
			},
		},
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d want 400 (body=%s)", rec.Code, rec.Body.String())
	}
}

// TestResponseFormatRejectedUnderBatching asserts the route-guard remains intact
// for a batch adapter that cannot expose exact logits (the E4B capability shape).
func TestResponseFormatRejectedUnderBatching(t *testing.T) {
	srv := newQwenBatchServer(t, []int32{})

	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":        []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens":      16,
		"response_format": map[string]string{"type": "json_object"},
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)

	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("status=%d want 501 (body=%s)", rec.Code, rec.Body.String())
	}
}

func TestResponseFormatRejectsForcedToolChoice(t *testing.T) {
	srv := newQwenBatchServer(t, nil)
	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":        []map[string]string{{"role": "user", "content": "weather"}},
		"tools":           []interface{}{qwenWeatherTool},
		"tool_choice":     "required",
		"response_format": map[string]string{"type": "json_object"},
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d want 400 for incompatible forced tool + response_format (body=%s)", rec.Code, rec.Body.String())
	}
}

// responseLogitBatch is a CPU fake of the CUDA constrained ABI used to prove
// response_format now traverses the HTTP continuous-batching route end-to-end.
type responseLogitBatch struct {
	mu     sync.Mutex
	script []int32
	pos    int
	last   int32
	vocab  int
}

func (e *responseLogitBatch) Supported() bool { return true }
func (e *responseLogitBatch) Capacity() int   { return 1 }
func (e *responseLogitBatch) AddSeq(_ []int32, _ batch.SeqParams) (int, int32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.pos, e.last = 1, e.script[0]
	return 0, e.last, nil
}
func (e *responseLogitBatch) StepBatch(slots []int32, inputs []int32) ([][]int32, error) {
	out, err := e.StepBatchExact(slots, inputs)
	return [][]int32{{out[0]}}, err
}
func (e *responseLogitBatch) StepBatchExact(_ []int32, _ []int32) ([]int32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.pos >= len(e.script) {
		e.last = -1
	} else {
		e.last = e.script[e.pos]
		e.pos++
	}
	return []int32{e.last}, nil
}
func (e *responseLogitBatch) CopyLogits(_ int, _ bool) ([]float32, int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	lg := make([]float32, e.vocab)
	for i := range lg {
		lg[i] = -100
	}
	if e.last >= 0 {
		lg[e.last] = 100
	}
	return lg, e.vocab, nil
}
func (e *responseLogitBatch) RemoveSeq(int) error { return nil }

func TestResponseFormatJSONSchemaUnderBatching(t *testing.T) {
	tk, idx := newServerTokenizer(t)
	b := func(c byte) int32 {
		const hex = "0123456789ABCDEF"
		return idx["<0x"+string([]byte{hex[c>>4], hex[c&15]})+">"]
	}
	text := `{"rows":[{"kind":"red","n":2}]}`
	script := make([]int32, 0, len(text)+1)
	for i := range []byte(text) {
		script = append(script, b(text[i]))
	}
	script = append(script, tk.EOS)
	eng := &responseLogitBatch{script: script, vocab: tk.NumTokens()}
	srv := New(&fakeServerEngine{ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS}, tk)
	srv.SetLogLevel("warn")
	if !srv.SetBatchEngine(eng) {
		t.Fatal("SetBatchEngine refused constrained-logit engine")
	}
	defer srv.scheduler.Shutdown()

	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":   []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens": 128, "temperature": 0,
		"response_format": map[string]interface{}{
			"type": "json_schema",
			"json_schema": map[string]interface{}{"name": "rows", "strict": true,
				"schema": map[string]interface{}{
					"type": "object", "properties": map[string]interface{}{
						"rows": map[string]interface{}{"type": "array", "items": map[string]interface{}{
							"type": "object", "properties": map[string]interface{}{
								"kind": map[string]interface{}{"enum": []string{"red", "green"}},
								"n":    map[string]interface{}{"type": "integer"},
							}, "required": []string{"kind", "n"},
						}},
					}, "required": []string{"rows"},
				},
			},
		},
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var resp ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	got := resp.Choices[0].Message.Content
	var obj struct {
		Rows []struct {
			Kind string `json:"kind"`
			N    int    `json:"n"`
		} `json:"rows"`
	}
	if err := json.Unmarshal([]byte(got), &obj); err != nil || len(obj.Rows) != 1 || obj.Rows[0].Kind != "red" || obj.Rows[0].N != 2 {
		t.Fatalf("batch schema output invalid: err=%v obj=%+v content=%q", err, obj, got)
	}
}
