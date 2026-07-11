// ABOUTME: Tests for the "session" chat-request field — resume from a disk
// ABOUTME: snapshot (suffix-only prefill), save-back after generation, and
// ABOUTME: rejection of invalid names, corrupt files, and wrong-model sessions.
package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/hikmaai-io/fucina/internal/session"
)

// fakeSessionServerEngine adds the kvSnapshotter capability to the scripted
// server engine: the "state" is the token ids serialized 4 B each, so restore
// reconstructs the engine token list exactly.
type fakeSessionServerEngine struct {
	fakeServerEngine
	saves, restores int
}

func (f *fakeSessionServerEngine) KVStateSize(n int) int {
	if n <= 0 {
		return 0
	}
	return n * 4
}

func (f *fakeSessionServerEngine) KVSave(buf []byte, n int) error {
	if n > len(f.tokens) {
		return fmt.Errorf("save beyond live sequence (%d > %d)", n, len(f.tokens))
	}
	for i := 0; i < n; i++ {
		v := f.tokens[i]
		buf[i*4], buf[i*4+1], buf[i*4+2], buf[i*4+3] =
			byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	}
	f.saves++
	return nil
}

func (f *fakeSessionServerEngine) KVRestore(buf []byte, n int) error {
	f.tokens = f.tokens[:0]
	for i := 0; i < n; i++ {
		f.tokens = append(f.tokens, int32(buf[i*4])|int32(buf[i*4+1])<<8|
			int32(buf[i*4+2])<<16|int32(buf[i*4+3])<<24)
	}
	f.restores++
	return nil
}

// tokensToState mirrors the fake engine's KVSave layout.
func tokensToState(tokens []int32) []byte {
	b := make([]byte, len(tokens)*4)
	for i, v := range tokens {
		b[i*4], b[i*4+1], b[i*4+2], b[i*4+3] =
			byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	}
	return b
}

// newSessionTestServer builds a session-enabled server on a fake snapshotting
// engine, with a scratch model dir (for the config hash) and session dir.
func newSessionTestServer(t *testing.T, script []int32) (*Server, *fakeSessionServerEngine, string, string) {
	t.Helper()
	tk, _ := newServerTokenizer(t)
	f := &fakeSessionServerEngine{fakeServerEngine: fakeServerEngine{
		ctxSize: 8192,
		vocab:   tk.NumTokens(),
		eos:     tk.EOS,
		script:  script,
	}}
	srv := New(f, tk)
	srv.SetLogLevel("warn")

	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"arch":"test"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	sessDir := t.TempDir()
	if err := srv.SetSessionDir(sessDir, modelDir); err != nil {
		t.Fatalf("SetSessionDir: %v", err)
	}
	return srv, f, sessDir, modelDir
}

func postJSON(t *testing.T, srv *Server, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	mux(srv).ServeHTTP(rec, req)
	return rec
}

// longPrompt is a raw /v1/completions prompt long enough that the session
// prefix clears the tiny-prefix and swap-margin thresholds.
func longPrompt() string {
	return strings.Repeat(" hello world over there", 200)
}

func TestSessionResumeSkipsSavedPrefix(t *testing.T) {
	srv, f, sessDir, modelDir := newSessionTestServer(t, nil)

	prompt := longPrompt()
	full := srv.tokenizer.Encode(prompt, true, false)
	if len(full) < 600 {
		t.Fatalf("test prompt too short: %d tokens", len(full))
	}
	// A previous process saved this session at all-but-32 of the prompt.
	sess := full[:len(full)-32]
	hash, err := session.HashModelConfig(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	meta := session.Meta{
		CreatedAt:  time.Now().UTC(),
		NTokens:    len(sess),
		EngineKind: session.KindFlatKV,
		Model:      session.Identity{Path: modelDir, ConfigHash: hash, StateProbe: 16 * 4},
	}
	if err := session.WriteFile(filepath.Join(sessDir, "alpha.fcsess"), meta, sess, tokensToState(sess)); err != nil {
		t.Fatal(err)
	}

	body := fmt.Sprintf(`{"prompt": %q, "session": "alpha", "max_tokens": 4}`, prompt)
	rec := postJSON(t, srv, "/v1/completions", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if f.restores != 1 {
		t.Errorf("engine restores = %d, want 1 (session restored)", f.restores)
	}
	// The saved prefix must cost ZERO prefill: only the 32 fresh tokens run.
	if f.lastPrefillLen != 32 {
		t.Errorf("prefilled %d tokens, want 32 (session covered the rest)", f.lastPrefillLen)
	}
	// The updated conversation was saved back and grew past the seed.
	snap, err := session.ReadFile(filepath.Join(sessDir, "alpha.fcsess"), 1<<30)
	if err != nil {
		t.Fatalf("re-read saved session: %v", err)
	}
	if snap.Meta.NTokens <= len(sess) {
		t.Errorf("saved-back session has %d tokens, want > %d", snap.Meta.NTokens, len(sess))
	}
}

func TestSessionCreatedWhenMissing(t *testing.T) {
	srv, _, sessDir, _ := newSessionTestServer(t, nil)
	body := fmt.Sprintf(`{"prompt": %q, "session": "fresh", "max_tokens": 2}`, longPrompt())
	rec := postJSON(t, srv, "/v1/completions", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if _, err := os.Stat(filepath.Join(sessDir, "fresh.fcsess")); err != nil {
		t.Errorf("session file not created: %v", err)
	}
}

func TestSessionRejectsBadNames(t *testing.T) {
	srv, _, _, _ := newSessionTestServer(t, nil)
	for _, name := range []string{"../evil", "a/b", "/abs", ".hidden", strings.Repeat("x", 200)} {
		nameJSON, _ := json.Marshal(name)
		body := fmt.Sprintf(`{"prompt": "hi there", "session": %s, "max_tokens": 1}`, nameJSON)
		rec := postJSON(t, srv, "/v1/completions", body)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("session name %q: status=%d, want 400", name, rec.Code)
		}
	}
}

func TestSessionRejectsCorruptFile(t *testing.T) {
	srv, f, sessDir, _ := newSessionTestServer(t, nil)
	if err := os.WriteFile(filepath.Join(sessDir, "bad.fcsess"), []byte("not a session"), 0o600); err != nil {
		t.Fatal(err)
	}
	body := fmt.Sprintf(`{"prompt": %q, "session": "bad", "max_tokens": 1}`, longPrompt())
	rec := postJSON(t, srv, "/v1/completions", body)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("corrupt session: status=%d body=%s, want 400", rec.Code, rec.Body.String())
	}
	if f.restores != 0 {
		t.Error("corrupt session must never reach the engine")
	}
}

func TestSessionRejectsWrongModel(t *testing.T) {
	srv, f, sessDir, _ := newSessionTestServer(t, nil)
	prompt := longPrompt()
	full := srv.tokenizer.Encode(prompt, true, false)
	sess := full[:len(full)-32]
	meta := session.Meta{
		CreatedAt:  time.Now().UTC(),
		NTokens:    len(sess),
		EngineKind: session.KindFlatKV,
		Model:      session.Identity{Path: "/other", ConfigHash: 0x1234, StateProbe: 16 * 4},
	}
	if err := session.WriteFile(filepath.Join(sessDir, "other.fcsess"), meta, sess, tokensToState(sess)); err != nil {
		t.Fatal(err)
	}
	body := fmt.Sprintf(`{"prompt": %q, "session": "other", "max_tokens": 1}`, prompt)
	rec := postJSON(t, srv, "/v1/completions", body)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("wrong-model session: status=%d, want 400", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "config") {
		t.Errorf("error %q does not name the config mismatch", rec.Body.String())
	}
	if f.restores != 0 {
		t.Error("wrong-model session must never reach the engine")
	}
}

func TestSessionDisabledWithoutDir(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil) // plain server, no SetSessionDir
	body := `{"prompt": "hi", "session": "x", "max_tokens": 1}`
	rec := postJSON(t, srv, "/v1/completions", body)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("session without --session-dir: status=%d, want 400", rec.Code)
	}
}
