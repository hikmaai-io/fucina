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
	"sync"
	"testing"
	"time"

	"github.com/hikmaai-io/fucina/internal/server/batch"
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

func TestSessionRejectsSymlink(t *testing.T) {
	srv, f, sessDir, _ := newSessionTestServer(t, nil)
	target := filepath.Join(t.TempDir(), "outside.fcsess")
	if err := os.WriteFile(target, []byte("outside"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, filepath.Join(sessDir, "linked.fcsess")); err != nil {
		t.Fatal(err)
	}
	rec := postJSON(t, srv, "/v1/completions",
		fmt.Sprintf(`{"prompt":%q,"session":"linked","max_tokens":1}`, longPrompt()))
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "symbolic link") {
		t.Fatalf("symlink status=%d body=%q, want precise 400", rec.Code, rec.Body.String())
	}
	if f.restores != 0 {
		t.Fatal("symlink session reached the engine")
	}
}

// fakeQ35SessionServerEngine supplies only the startup geometry probe. Inference
// itself is owned by q35HTTPBatchEngine, mirroring mandatory Qwen batching.
type fakeQ35SessionServerEngine struct{ fakeServerEngine }

func (f *fakeQ35SessionServerEngine) SeqStateSize(n int) int64 {
	if n <= 0 {
		return 0
	}
	return int64(8 + 4*n)
}

type q35HTTPBatchEngine struct {
	mu sync.Mutex

	capacity int
	next     int
	free     []int
	live     map[int][]int32

	restores    int
	exports     int
	prefill     []int32
	failRestore bool
}

func newQ35HTTPBatchEngine(capacity int) *q35HTTPBatchEngine {
	return &q35HTTPBatchEngine{capacity: capacity, live: make(map[int][]int32)}
}

func (e *q35HTTPBatchEngine) Supported() bool { return true }
func (e *q35HTTPBatchEngine) Capacity() int   { return e.capacity }

func (e *q35HTTPBatchEngine) alloc(tokens []int32) int {
	var slot int
	if n := len(e.free); n > 0 {
		slot = e.free[n-1]
		e.free = e.free[:n-1]
	} else {
		slot = e.next
		e.next++
	}
	e.live[slot] = append([]int32(nil), tokens...)
	return slot
}

func (e *q35HTTPBatchEngine) AddSeq(prompt []int32, _ batch.SeqParams) (int, int32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.alloc(prompt), 42, nil
}

func (e *q35HTTPBatchEngine) OpenSeq(_ []int32, _ batch.SeqParams) (int, int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.alloc(nil), 0, nil
}

func (e *q35HTTPBatchEngine) PrefillChunk(slot int, tokens []int32, last bool) (int32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.prefill = append(e.prefill, tokens...)
	e.live[slot] = append(e.live[slot], tokens...)
	if last {
		return 42, nil
	}
	return 0, nil
}

func (e *q35HTTPBatchEngine) StepBatch(slots []int32, inputs []int32) ([][]int32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([][]int32, len(slots))
	for i, slot := range slots {
		e.live[int(slot)] = append(e.live[int(slot)], inputs[i])
		out[i] = []int32{42}
	}
	return out, nil
}

func (e *q35HTTPBatchEngine) StepBatchExact(slots []int32, inputs []int32) ([]int32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]int32, len(slots))
	for i, slot := range slots {
		e.live[int(slot)] = append(e.live[int(slot)], inputs[i])
		out[i] = 42
	}
	return out, nil
}

func (e *q35HTTPBatchEngine) RestoreSession(prompt []int32, _ batch.SeqParams, snap batch.StateSnapshot) (int, int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.failRestore {
		return 0, 0, fmt.Errorf("forced CUDA restore failure")
	}
	n := len(snap.Tokens)
	if n <= 0 || n >= len(prompt) || !tokensArePrefix(snap.Tokens, prompt) {
		return 0, 0, fmt.Errorf("not a strict prefix")
	}
	if want := q35FakeState(snap.Tokens); string(want) != string(snap.State) {
		return 0, 0, fmt.Errorf("bad state")
	}
	e.restores++
	return e.alloc(snap.Tokens), n, nil
}

func (e *q35HTTPBatchEngine) ExportSession(slot int, history []int32) (*batch.StateSnapshot, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.exports++
	committed := append([]int32(nil), e.live[slot]...)
	if len(committed) > len(history) || !tokensArePrefix(committed, history) {
		return nil, fmt.Errorf("committed state/history mismatch")
	}
	return &batch.StateSnapshot{Tokens: committed, State: q35FakeState(committed)}, nil
}

func (e *q35HTTPBatchEngine) RemoveSeq(slot int) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.live, slot)
	e.free = append(e.free, slot)
	return nil
}

func q35FakeState(tokens []int32) []byte {
	state := make([]byte, 8+4*len(tokens))
	copy(state, "Q35STATE")
	for i, tok := range tokens {
		state[8+4*i] = byte(tok)
		state[8+4*i+1] = byte(tok >> 8)
		state[8+4*i+2] = byte(tok >> 16)
		state[8+4*i+3] = byte(tok >> 24)
	}
	return state
}

func newQ35SessionTestServer(t *testing.T, sessionDir, modelDir string) (*Server, *q35HTTPBatchEngine) {
	t.Helper()
	tk, _ := newServerTokenizer(t)
	base := &fakeQ35SessionServerEngine{fakeServerEngine: fakeServerEngine{
		ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS,
	}}
	srv := New(base, tk)
	srv.SetLogLevel("warn")
	if err := srv.SetSessionDir(sessionDir, modelDir); err != nil {
		t.Fatalf("SetSessionDir(Q35): %v", err)
	}
	eng := newQ35HTTPBatchEngine(2)
	if !srv.SetBatchEngine(eng) {
		t.Fatal("SetBatchEngine(Q35) refused stateful engine")
	}
	// Ignore scheduler graph-warmup traffic in request assertions.
	eng.mu.Lock()
	eng.prefill = nil
	eng.restores = 0
	eng.exports = 0
	eng.mu.Unlock()
	return srv, eng
}

func TestQ35HTTPSessionPersistsAcrossSchedulerRestart(t *testing.T) {
	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"arch":"qwen3_5"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	sessionDir := t.TempDir()
	prompt := longPrompt()

	// Process 1: missing file is a cold admission, then the complete Q35 slot
	// frontier is exported and atomically written after generation.
	srv1, eng1 := newQ35SessionTestServer(t, sessionDir, modelDir)
	rec := postJSON(t, srv1, "/v1/completions", fmt.Sprintf(`{"prompt":%q,"session":"agent","max_tokens":1}`, prompt))
	if rec.Code != http.StatusOK {
		t.Fatalf("first request status=%d body=%s", rec.Code, rec.Body.String())
	}
	snap1, err := session.ReadFile(filepath.Join(sessionDir, "agent.fcsess"), 1<<30)
	if err != nil {
		t.Fatalf("read first snapshot: %v", err)
	}
	want1 := srv1.tokenizer.Encode(prompt, true, false)
	if snap1.Meta.EngineKind != session.KindQ35Slot || !tokensArePrefix(snap1.Tokens, want1) || len(snap1.Tokens) != len(want1) {
		t.Fatalf("first snapshot kind/tokens = %s/%d, want q35/%d", snap1.Meta.EngineKind, len(snap1.Tokens), len(want1))
	}
	if eng1.exports != 1 || eng1.restores != 0 {
		t.Errorf("first process exports/restores = %d/%d, want 1/0", eng1.exports, eng1.restores)
	}
	srv1.scheduler.Shutdown()

	// Process 2: load the disk file during HTTP validation, restore it during
	// scheduler admission, and prefill only the newly appended prompt suffix.
	extended := prompt + "\nA genuinely new user turn follows."
	srv2, eng2 := newQ35SessionTestServer(t, sessionDir, modelDir)
	full2 := srv2.tokenizer.Encode(extended, true, false)
	if !tokensArePrefix(snap1.Tokens, full2) || len(snap1.Tokens) >= len(full2) {
		t.Fatalf("test prompt does not strictly extend saved tokens: saved=%d full=%d", len(snap1.Tokens), len(full2))
	}
	rec = postJSON(t, srv2, "/v1/completions", fmt.Sprintf(`{"prompt":%q,"session":"agent","max_tokens":1}`, extended))
	if rec.Code != http.StatusOK {
		t.Fatalf("resume status=%d body=%s", rec.Code, rec.Body.String())
	}
	eng2.mu.Lock()
	restores, exports := eng2.restores, eng2.exports
	prefilled := append([]int32(nil), eng2.prefill...)
	eng2.mu.Unlock()
	if restores != 1 || exports != 1 {
		t.Errorf("resume restores/exports = %d/%d, want 1/1", restores, exports)
	}
	if !tokensArePrefix(prefilled, full2[len(snap1.Tokens):]) || len(prefilled) != len(full2)-len(snap1.Tokens) {
		t.Errorf("resume prefilled %d tokens, want exact %d-token suffix", len(prefilled), len(full2)-len(snap1.Tokens))
	}
	snap2, err := session.ReadFile(filepath.Join(sessionDir, "agent.fcsess"), 1<<30)
	if err != nil {
		t.Fatalf("read updated snapshot: %v", err)
	}
	if len(snap2.Tokens) != len(full2) || string(snap2.State) != string(q35FakeState(full2)) {
		t.Errorf("updated snapshot is not the complete second-request frontier: tokens=%d want=%d", len(snap2.Tokens), len(full2))
	}
	srv2.scheduler.Shutdown()
}

func TestQ35HTTPRestoreFailureIsVisible(t *testing.T) {
	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"arch":"qwen3_5"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	sessionDir := t.TempDir()
	srv, eng := newQ35SessionTestServer(t, sessionDir, modelDir)
	defer srv.scheduler.Shutdown()
	prompt := longPrompt()

	rec := postJSON(t, srv, "/v1/completions",
		fmt.Sprintf(`{"prompt":%q,"session":"restorefail","max_tokens":1}`, prompt))
	if rec.Code != http.StatusOK {
		t.Fatalf("seed status=%d body=%s", rec.Code, rec.Body.String())
	}
	eng.mu.Lock()
	eng.failRestore = true
	eng.mu.Unlock()
	rec = postJSON(t, srv, "/v1/completions",
		fmt.Sprintf(`{"prompt":%q,"session":"restorefail","max_tokens":1}`, prompt+"\nnew suffix"))
	if rec.Code != http.StatusInternalServerError || !strings.Contains(rec.Body.String(), "forced CUDA restore failure") {
		t.Fatalf("restore failure status=%d body=%q, want visible 500", rec.Code, rec.Body.String())
	}
}

func TestQ35HTTPSessionPersistsStreamingResponse(t *testing.T) {
	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"arch":"qwen3_5"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	sessionDir := t.TempDir()
	srv, eng := newQ35SessionTestServer(t, sessionDir, modelDir)
	defer srv.scheduler.Shutdown()

	rec := postJSON(t, srv, "/v1/completions",
		fmt.Sprintf(`{"prompt":%q,"session":"streamed","max_tokens":2,"stream":true}`, longPrompt()))
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "data: [DONE]") {
		t.Fatalf("stream status=%d body=%q", rec.Code, rec.Body.String())
	}
	if _, err := session.ReadFile(filepath.Join(sessionDir, "streamed.fcsess"), 1<<30); err != nil {
		t.Fatalf("streaming request did not persist session: %v", err)
	}
	eng.mu.Lock()
	exports := eng.exports
	eng.mu.Unlock()
	if exports != 1 {
		t.Fatalf("streaming exports=%d, want 1", exports)
	}
}

func TestQ35HTTPSessionRejectsDivergentPrompt(t *testing.T) {
	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"arch":"qwen3_5"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	sessionDir := t.TempDir()
	srv, eng := newQ35SessionTestServer(t, sessionDir, modelDir)
	defer srv.scheduler.Shutdown()

	rec := postJSON(t, srv, "/v1/completions",
		fmt.Sprintf(`{"prompt":%q,"session":"fork","max_tokens":1}`, longPrompt()))
	if rec.Code != http.StatusOK {
		t.Fatalf("seed status=%d body=%s", rec.Code, rec.Body.String())
	}
	rec = postJSON(t, srv, "/v1/completions",
		`{"prompt":"an unrelated conversation","session":"fork","max_tokens":1}`)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "strict prefix") {
		t.Fatalf("divergent status=%d body=%q, want strict-prefix 400", rec.Code, rec.Body.String())
	}
	eng.mu.Lock()
	restores := eng.restores
	eng.mu.Unlock()
	if restores != 0 {
		t.Fatalf("divergent session reached restore %d times", restores)
	}
}

func TestQ35SessionRefusesBatchEngineWithoutStateABI(t *testing.T) {
	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"arch":"qwen3_5"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	tk, _ := newServerTokenizer(t)
	base := &fakeQ35SessionServerEngine{fakeServerEngine: fakeServerEngine{
		ctxSize: 8192, vocab: tk.NumTokens(), eos: tk.EOS,
	}}
	srv := New(base, tk)
	if err := srv.SetSessionDir(t.TempDir(), modelDir); err != nil {
		t.Fatal(err)
	}
	if srv.SetBatchEngine(newScriptedBatchEngine([]int32{42})) {
		t.Fatal("Q35 session accepted a batch engine without restore/export ABI")
	}
}

func TestQ35HTTPSessionRejectsBadGeometryBeforeScheduler(t *testing.T) {
	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{"arch":"qwen3_5"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	sessionDir := t.TempDir()
	srv, eng := newQ35SessionTestServer(t, sessionDir, modelDir)
	defer srv.scheduler.Shutdown()
	prompt := longPrompt()
	full := srv.tokenizer.Encode(prompt, true, false)
	saved := full[:len(full)-4]
	hash, _ := session.HashModelConfig(modelDir)
	meta := session.Meta{
		CreatedAt: time.Now().UTC(), NTokens: len(saved), EngineKind: session.KindQ35Slot,
		Model: session.Identity{Path: modelDir, ConfigHash: hash, StateProbe: 8 + 4*session.ProbeTokens},
	}
	// Checksummed and structurally valid, but one byte shorter than this model's
	// per-token state geometry. It must not reach RestoreSession/CUDA.
	if err := session.WriteFile(filepath.Join(sessionDir, "badshape.fcsess"), meta, saved, q35FakeState(saved)[:len(q35FakeState(saved))-1]); err != nil {
		t.Fatal(err)
	}
	rec := postJSON(t, srv, "/v1/completions", fmt.Sprintf(`{"prompt":%q,"session":"badshape","max_tokens":1}`, prompt))
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "lays out") {
		t.Fatalf("status=%d body=%q, want precise geometry 400", rec.Code, rec.Body.String())
	}
	if eng.restores != 0 {
		t.Errorf("bad geometry reached engine restore %d times", eng.restores)
	}
}

func TestSessionClaimRejectsConcurrentWriter(t *testing.T) {
	s := &Server{sessionActive: make(map[string]struct{})}
	if !s.claimSession("agent") {
		t.Fatal("first claim rejected")
	}
	if s.claimSession("agent") {
		t.Fatal("concurrent claim accepted")
	}
	if !s.claimSession("other") {
		t.Fatal("independent session claim rejected")
	}
	s.releaseSession("agent")
	if !s.claimSession("agent") {
		t.Fatal("released session could not be reclaimed")
	}
}
