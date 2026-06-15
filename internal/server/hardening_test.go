// ABOUTME: Tests for the P0/P1 hardening pass — auth, param validation, output
// ABOUTME: cap, admission control, readiness probe, and panic-recovery middleware.

package server

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// ─── auth (C6) ───────────────────────────────────────────────────────────────

func TestAuthDisabledByDefault(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, httptest.NewRequest("GET", "/v1/models", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("no key set: status=%d want 200 (auth must be off by default)", rec.Code)
	}
}

func TestAuthRejectsMissingAndWrongKey(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	srv.SetAPIKey("s3cret")

	cases := []struct {
		name, header string
		want         int
	}{
		{"missing", "", http.StatusUnauthorized},
		{"wrong", "Bearer nope", http.StatusUnauthorized},
		{"no-bearer-prefix", "s3cret", http.StatusUnauthorized},
		{"correct", "Bearer s3cret", http.StatusOK},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", "/v1/models", nil)
			if tc.header != "" {
				req.Header.Set("Authorization", tc.header)
			}
			rec := httptest.NewRecorder()
			mux(srv).ServeHTTP(rec, req)
			if rec.Code != tc.want {
				t.Fatalf("status=%d want %d", rec.Code, tc.want)
			}
		})
	}
}

func TestAuthDoesNotGateHealthOrMetrics(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	srv.SetAPIKey("s3cret")
	for _, path := range []string{"/health", "/healthz", "/readyz", "/metrics"} {
		rec := httptest.NewRecorder()
		mux(srv).ServeHTTP(rec, httptest.NewRequest("GET", path, nil))
		if rec.Code == http.StatusUnauthorized {
			t.Errorf("%s should not require auth, got 401", path)
		}
	}
}

// ─── param validation / clamp (C7) ───────────────────────────────────────────

func TestValidateAndClampParams(t *testing.T) {
	const vocab = 1000
	tests := []struct {
		name    string
		in      GenerationParams
		wantMsg bool
		check   func(GenerationParams) string // "" = ok
	}{
		{"negative top_k -> 0", GenerationParams{TopK: -5, Temperature: 1, RepeatPenalty: 1}, false,
			func(p GenerationParams) string {
				if p.TopK != 0 {
					return "top_k not clamped to 0"
				}
				return ""
			}},
		{"huge top_k -> vocab", GenerationParams{TopK: 1 << 30, Temperature: 1, RepeatPenalty: 1}, false,
			func(p GenerationParams) string {
				if p.TopK != vocab {
					return "top_k not clamped to vocab"
				}
				return ""
			}},
		{"top_p > 1 -> 1", GenerationParams{TopP: 5, Temperature: 1, RepeatPenalty: 1}, false,
			func(p GenerationParams) string {
				if p.TopP != 1 {
					return "top_p not clamped"
				}
				return ""
			}},
		{"negative min_p -> 0", GenerationParams{MinP: -0.5, Temperature: 1, RepeatPenalty: 1}, false,
			func(p GenerationParams) string {
				if p.MinP != 0 {
					return "min_p not clamped"
				}
				return ""
			}},
		{"negative temp -> 0", GenerationParams{Temperature: -2, RepeatPenalty: 1}, false,
			func(p GenerationParams) string {
				if p.Temperature != 0 {
					return "temp not clamped"
				}
				return ""
			}},
		{"NaN temp rejected", GenerationParams{Temperature: math.NaN(), RepeatPenalty: 1}, true, nil},
		{"Inf top_p rejected", GenerationParams{Temperature: 1, TopP: math.Inf(1), RepeatPenalty: 1}, true, nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := tc.in
			msg := validateAndClampParams(&p, vocab)
			if tc.wantMsg != (msg != "") {
				t.Fatalf("msg=%q wantMsg=%v", msg, tc.wantMsg)
			}
			if tc.check != nil {
				if problem := tc.check(p); problem != "" {
					t.Errorf("%s: %+v", problem, p)
				}
			}
		})
	}
}

func TestChatCompletionsRejectsNaNTemperature(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	// Marshal NaN manually — encoding/json refuses to encode it.
	body := strings.NewReader(`{"messages":[{"role":"user","content":"hi"}],"temperature":1e400}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", body)
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d want 400 for non-finite temperature", rec.Code)
	}
}

// ─── absolute output cap (M3) ────────────────────────────────────────────────

func TestSetMaxOutputTokensCapsRequest(t *testing.T) {
	srv, eng := newTestServer(t, 8192, []int32{1, 2, 3, 4, 5, 6, 7, 8})
	srv.SetMaxOutputTokens(3)
	body := strings.NewReader(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":5000}`)
	req := httptest.NewRequest("POST", "/v1/chat/completions", body)
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if eng.lastMaxNew > 3 {
		t.Errorf("maxNew=%d want <=3 (absolute cap not applied)", eng.lastMaxNew)
	}
}

// ─── admission control (M1) ──────────────────────────────────────────────────

func TestAdmissionControlReturns503WhenSaturated(t *testing.T) {
	srv, eng := newTestServer(t, 8192, []int32{1, 2})
	srv.SetMaxConcurrent(1) // one slot total

	// Block the in-flight request inside generation so the single slot stays held.
	release := make(chan struct{})
	eng.blockGen = release

	var firstStarted sync.WaitGroup
	firstStarted.Add(1)
	eng.onGenStart = firstStarted.Done

	go func() {
		body := strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`)
		mux(srv).ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("POST", "/v1/chat/completions", body))
	}()
	firstStarted.Wait() // first request now holds the slot inside generation

	// Second request must be shed.
	body := strings.NewReader(`{"messages":[{"role":"user","content":"hi"}]}`)
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", body))
	close(release)

	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d want 503 when saturated", rec.Code)
	}
	if rec.Header().Get("Retry-After") == "" {
		t.Errorf("503 should carry a Retry-After header")
	}
}

// ─── readiness probe (M7) ────────────────────────────────────────────────────

func TestReadyzReports200WhenReady(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, httptest.NewRequest("GET", "/readyz", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200", rec.Code)
	}
	var body map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["status"] != "ready" {
		t.Errorf("status=%v want ready", body["status"])
	}
}

func TestReadyz503WhenTokenizerMissing(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	srv.tokenizer = nil // simulate a failed tokenizer load
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, httptest.NewRequest("GET", "/readyz", nil))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("status=%d want 503 when tokenizer missing", rec.Code)
	}
}

// ─── request-id correlation (M8) ─────────────────────────────────────────────

func TestRequestIDGeneratedAndEchoed(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, httptest.NewRequest("GET", "/health", nil))
	if rec.Header().Get("X-Request-Id") == "" {
		t.Error("response missing generated X-Request-Id")
	}
}

func TestRequestIDHonorsInbound(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	req := httptest.NewRequest("GET", "/health", nil)
	req.Header.Set("X-Request-Id", "trace-abc-123")
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if got := rec.Header().Get("X-Request-Id"); got != "trace-abc-123" {
		t.Errorf("X-Request-Id=%q want trace-abc-123 (inbound id not honored)", got)
	}
}

// ─── panic-recovery middleware (C2) ──────────────────────────────────────────

func TestLogRequestRecoversPanicBeforeHeaders(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	panicker := srv.logRequest(func(w http.ResponseWriter, r *http.Request) {
		panic("boom")
	})
	rec := httptest.NewRecorder()
	// Must not propagate the panic (no crash) and must return 500.
	panicker(rec, httptest.NewRequest("GET", "/x", nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status=%d want 500 after recovered panic", rec.Code)
	}
}

func TestLogRequestRecoversPanicAfterHeaders(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	panicker := srv.logRequest(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("partial"))
		panic("boom mid-stream")
	})
	rec := httptest.NewRecorder()
	panicker(rec, httptest.NewRequest("GET", "/x", nil)) // must not crash
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want 200 (headers already sent, cannot rewrite)", rec.Code)
	}
}
