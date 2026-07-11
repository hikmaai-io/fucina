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

	"github.com/hikmaai-io/fucina/internal/session"
)

// sessionNameRe is the allowed shape of a client-supplied session name. A
// NAME, never a path: the client must not be able to point the server at (or
// create) files outside the configured session directory.
var sessionNameRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)

// SetSessionDir enables disk session persistence for the single-flight
// (KVCache) serving path: requests carrying a "session" field load their
// snapshot from dir before prefill and save the updated conversation back
// after generation. modelPath is fingerprinted into every saved session so a
// restart with a different model rejects the file instead of restoring
// garbage. Returns an error when the engine cannot snapshot KV state.
func (s *Server) SetSessionDir(dir, modelPath string) error {
	if !s.kv.SessionSupported() {
		return fmt.Errorf("engine does not support KV snapshots; --session-dir unavailable")
	}
	snap, ok := s.engine.(kvSnapshotter)
	if !ok {
		return fmt.Errorf("engine does not support KV snapshots; --session-dir unavailable")
	}
	probe := int64(snap.KVStateSize(session.ProbeTokens))
	if probe <= 0 {
		return fmt.Errorf("engine reports no snapshot geometry; --session-dir unavailable")
	}
	hash, err := session.HashModelConfig(modelPath)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("session dir: %w", err)
	}
	s.sessionDir = dir
	s.sessionIdent = session.Identity{Path: modelPath, ConfigHash: hash, StateProbe: probe}
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

// loadSessionIntoKV reads a session file and seeds it into the KVCache
// snapshot pool, so the request's Prefill restores it when the prompt
// matches (a full match skips prefilling the restored prefix entirely). A
// missing file is NOT an error — it is a new session that the post-request
// save will create. The caller must hold s.kv.Lock().
func (s *Server) loadSessionIntoKV(path string) error {
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

// sessionMaxStateBytes caps the state section a session file may declare
// before the server refuses to allocate for it (hostile-file bound; real
// flat-KV states are ~200 KB/token).
const sessionMaxStateBytes = 64 << 30
