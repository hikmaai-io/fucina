package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
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

// TestResponseFormatRejectedUnderBatching asserts the route-guard: response_format
// cannot be honored by the on-device batch sampler, so under continuous batching the
// server returns 501 (rather than silently producing unconstrained output).
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
