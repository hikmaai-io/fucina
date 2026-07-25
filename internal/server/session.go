// ABOUTME: Server-side session persistence — the "session" chat-request field
// ABOUTME: resumes a conversation from a disk snapshot and saves it back after.
package server

import (
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"time"

	"github.com/hikmaai-io/fucina/internal/server/batch"
	"github.com/hikmaai-io/fucina/internal/session"
)

// sessionNameRe is the allowed shape of a client-supplied session name. A
// NAME, never a path: the client must not be able to point the server at (or
// create) files outside the configured session directory.
var sessionNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)

// seqStateSizer is the startup-only capability probe for Qwen3.5/3.6 hybrid
// slot snapshots. Two adjacent sizes derive the exact affine layout
// (fixed recurrent bytes + per-token full-attention K/V) without calling the
// CUDA engine from HTTP goroutines after the scheduler owns it.
type seqStateSizer interface {
	SeqStateSize(nTokens int) int64
}

// SetSessionDir enables disk session persistence. Qwen3.5/3.6 is detected
// first and uses q35-slot state through mandatory continuous batching; other
// engines retain the existing flat-KV single-flight path. modelPath is
// fingerprinted into every file, and adjacent Q35 geometry probes let request
// handlers reject malformed state lengths before scheduler/CUDA admission.
func (s *Server) SetSessionDir(dir, modelPath string) error {
	kind := ""
	var probe, base, step int64
	if q35, ok := s.engine.(seqStateSizer); ok {
		probe = q35.SeqStateSize(session.ProbeTokens)
		next := q35.SeqStateSize(session.ProbeTokens + 1)
		if probe > 0 && next > probe {
			step = next - probe
			base = probe - int64(session.ProbeTokens)*step
			if base < 0 {
				return fmt.Errorf("engine reports invalid Q35 snapshot geometry")
			}
			kind = session.KindQ35Slot
		}
	}
	if kind == "" {
		if !s.kv.SessionSupported() {
			return fmt.Errorf("engine does not support state snapshots; --session-dir unavailable")
		}
		snap, ok := s.engine.(kvSnapshotter)
		if !ok {
			return fmt.Errorf("engine does not support KV snapshots; --session-dir unavailable")
		}
		probe = int64(snap.KVStateSize(session.ProbeTokens))
		if probe <= 0 {
			return fmt.Errorf("engine reports no snapshot geometry; --session-dir unavailable")
		}
		kind = session.KindFlatKV
	}
	hash, err := session.HashModelConfig(modelPath)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("session dir: %w", err)
	}
	s.sessionDir = dir
	s.sessionKind = kind
	s.sessionStateBase = base
	s.sessionStateStep = step
	s.sessionIdent = session.Identity{Path: modelPath, ConfigHash: hash, StateProbe: probe}
	s.sessionActive = make(map[string]struct{})
	return nil
}

// sessionFilePath validates a client-supplied session name and resolves it
// inside the session directory.
func (s *Server) sessionFilePath(name string) (string, error) {
	if s.sessionDir == "" {
		return "", fmt.Errorf("session support is disabled (start the server with --session-dir)")
	}
	if !sessionNameRe.MatchString(name) || name == ".." || filepath.Base(name) != name {
		return "", fmt.Errorf("invalid session name %q: use letters, digits, '.', '_', '-' (max 128 chars)", name)
	}
	return filepath.Join(s.sessionDir, name+".fcsess"), nil
}

func (s *Server) claimSession(name string) bool {
	s.sessionMu.Lock()
	defer s.sessionMu.Unlock()
	if _, exists := s.sessionActive[name]; exists {
		return false
	}
	s.sessionActive[name] = struct{}{}
	return true
}

func (s *Server) releaseSession(name string) {
	s.sessionMu.Lock()
	delete(s.sessionActive, name)
	s.sessionMu.Unlock()
}

// loadBatchSession validates a Q35 disk snapshot entirely on the HTTP
// goroutine, before scheduler admission. A missing file starts a new session.
// Existing state must be an exact strict prefix of this rendered prompt: hybrid
// recurrent state cannot be rewound or applied to a divergent conversation.
func (s *Server) loadBatchSession(path string, prompt []int32) (*batch.StateSnapshot, error) {
	if err := rejectSessionSymlink(path); err != nil {
		return nil, err
	}
	// Bound allocation by the largest state this engine can restore, not the
	// format-wide 64 GiB ceiling. Keep the full context bound (rather than this
	// prompt's length) so a divergent session can still be parsed and rejected
	// with the precise strict-prefix error.
	maxTokens := int64(s.engine.ContextSize())
	if maxTokens < 1 || s.sessionStateBase < 0 || s.sessionStateStep <= 0 {
		return nil, fmt.Errorf("engine reports no restorable session geometry")
	}
	maxState := int64(sessionMaxStateBytes)
	if maxTokens <= (sessionMaxStateBytes-s.sessionStateBase)/s.sessionStateStep {
		maxState = s.sessionStateBase + maxTokens*s.sessionStateStep
	}
	snap, err := session.ReadFile(path, maxState)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	cur := session.Meta{EngineKind: session.KindQ35Slot, Model: s.sessionIdent}
	if err := snap.Meta.Validate(cur); err != nil {
		return nil, err
	}
	n := len(snap.Tokens)
	if n <= 0 || n >= len(prompt) || !tokensArePrefix(snap.Tokens, prompt) {
		return nil, fmt.Errorf("saved token history (%d tokens) is not a strict prefix of the %d-token request prompt", n, len(prompt))
	}
	if n > int(s.engine.ContextSize()) {
		return nil, fmt.Errorf("saved token history %d exceeds engine context %d", n, s.engine.ContextSize())
	}
	want := s.sessionStateBase + int64(n)*s.sessionStateStep
	if want <= 0 || int64(len(snap.State)) != want {
		return nil, fmt.Errorf("session state is %d B but the engine lays out %d tokens as %d B", len(snap.State), n, want)
	}
	return &batch.StateSnapshot{Tokens: snap.Tokens, State: snap.State}, nil
}

func tokensArePrefix(prefix, tokens []int32) bool {
	if len(prefix) > len(tokens) {
		return false
	}
	for i, tok := range prefix {
		if tokens[i] != tok {
			return false
		}
	}
	return true
}

// saveBatchSessionResult writes an exported Q35 snapshot after generation.
// Export happens on the scheduler owner while the slot is live; filesystem I/O
// happens here, outside the scheduler, so another sequence's decode is not
// stalled by fsync. As on the flat path, persistence failure is logged rather
// than rewriting an already-correct model response.
func (s *Server) saveBatchSessionResult(path string, res batch.Result) {
	if path == "" {
		return
	}
	if res.SessionErr != nil {
		log.Printf("fucina: session save %s: %v", filepath.Base(path), res.SessionErr)
		return
	}
	if res.SessionState == nil {
		log.Printf("fucina: session save %s: no state exported (finish=%s)", filepath.Base(path), res.Reason)
		return
	}
	snap := res.SessionState
	meta := session.Meta{
		CreatedAt:  time.Now().UTC(),
		NTokens:    len(snap.Tokens),
		EngineKind: session.KindQ35Slot,
		Model:      s.sessionIdent,
	}
	if err := session.WriteFile(path, meta, snap.Tokens, snap.State); err != nil {
		log.Printf("fucina: session save %s: %v", filepath.Base(path), err)
		return
	}
	log.Printf("fucina: session %s saved (%d tokens, %.1f MB; %d restored)",
		filepath.Base(path), len(snap.Tokens), float64(len(snap.State))/(1<<20), res.ReusedTokens)
}

// loadSessionIntoKV reads a session file and seeds it into the KVCache
// snapshot pool, so the request's Prefill restores it when the prompt
// matches (a full match skips prefilling the restored prefix entirely). A
// missing file is NOT an error — it is a new session that the post-request
// save will create. The caller must hold s.kv.Lock().
func (s *Server) loadSessionIntoKV(path string) error {
	if err := rejectSessionSymlink(path); err != nil {
		return err
	}
	snap, err := session.ReadFile(path, sessionMaxStateBytes)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	cur := session.Meta{EngineKind: session.KindFlatKV, Model: s.sessionIdent}
	if err := snap.Meta.Validate(cur); err != nil {
		return err
	}
	if err := s.kv.SeedSnapshot(snap.Tokens, snap.State); err != nil {
		return err
	}
	log.Printf("fucina: session %s seeded (%d tokens)", filepath.Base(path), len(snap.Tokens))
	return nil
}

// saveSessionFromKV exports the live sequence (this request's conversation,
// prompt + generated reply) and writes it back to the session file. Failures
// are logged, not fatal: the response was already correct, only persistence
// is lost. The caller must hold s.kv.Lock().
func (s *Server) saveSessionFromKV(path string) {
	tokens, state, err := s.kv.ExportSession()
	if err != nil {
		log.Printf("fucina: session save %s: %v", filepath.Base(path), err)
		return
	}
	meta := session.Meta{
		CreatedAt:  time.Now().UTC(),
		NTokens:    len(tokens),
		EngineKind: session.KindFlatKV,
		Model:      s.sessionIdent,
	}
	if err := session.WriteFile(path, meta, tokens, state); err != nil {
		log.Printf("fucina: session save %s: %v", filepath.Base(path), err)
		return
	}
	log.Printf("fucina: session %s saved (%d tokens, %.1f MB)",
		filepath.Base(path), len(tokens), float64(len(state))/(1<<20))
}

// rejectSessionSymlink prevents a client-controlled session name from making
// the server read through a pre-planted symlink outside --session-dir. Missing
// paths are valid new sessions; writes use atomic rename and replace in-dir.
func rejectSessionSymlink(path string) error {
	info, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("session: inspect file: %w", err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("session: refusing symbolic link %s", filepath.Base(path))
	}
	return nil
}

// sessionMaxStateBytes caps the state section a session file may declare
// before the server refuses to allocate for it (hostile-file bound; real
// flat-KV states are ~200 KB/token).
const sessionMaxStateBytes = 64 << 30
