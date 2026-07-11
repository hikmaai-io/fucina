// ABOUTME: REPL /save and /load — persist a conversation (token history +
// ABOUTME: engine KV/state + chat transcript) to disk and resume across runs.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/hikmaai-io/fucina/internal/chat"
	"github.com/hikmaai-io/fucina/internal/engine/cuda"
	"github.com/hikmaai-io/fucina/internal/server/batch"
	"github.com/hikmaai-io/fucina/internal/session"
	"github.com/hikmaai-io/fucina/internal/tokenizer"

	gemserver "github.com/hikmaai-io/fucina/internal/server"
)

// sessionMaxStateBytes caps how large a state section a session file may
// declare before we refuse to allocate for it. Real snapshots are well under
// this (≈200 KB/token flat, ≈48 KB/token + 75 MiB fixed hybrid).
const sessionMaxStateBytes = 64 << 30

// replClientState is the REPL bookkeeping stored in Meta.Client: everything
// needed to re-render future prompts byte-identically after a restart.
type replClientState struct {
	Thinking string             `json:"thinking,omitempty"`
	Dense    []chat.Message     `json:"dense_history,omitempty"`
	Paged    []chat.RichMessage `json:"paged_history,omitempty"`
}

// parseSessionCommand recognizes "/save FILE" and "/load FILE". ok reports
// the input was one of the two commands (even with bad arity, so the caller
// consumes it as a command rather than chat input); file is empty on misuse.
func parseSessionCommand(input string) (save bool, file string, ok bool) {
	var cmd string
	n, _ := fmt.Sscanf(input, "%s %s", &cmd, &file)
	if cmd != "/save" && cmd != "/load" {
		return false, "", false
	}
	if n != 2 {
		return cmd == "/save", "", true
	}
	return cmd == "/save", file, true
}

// sessionIdentity fingerprints the loaded model for the session header.
// stateProbe must be the engine's snapshot size for session.ProbeTokens
// tokens (KVStateSize or SeqStateSize depending on the path).
func sessionIdentity(modelPath string, stateProbe int64) (session.Identity, error) {
	if stateProbe <= 0 {
		return session.Identity{}, errors.New("this engine does not support state snapshots")
	}
	hash, err := session.HashModelConfig(modelPath)
	if err != nil {
		return session.Identity{}, err
	}
	return session.Identity{Path: modelPath, ConfigHash: hash, StateProbe: stateProbe}, nil
}

// ─── Dense (flat single-sequence) REPL: gemma engines via KVCache ───────────

// denseSessionSave persists the live KVCache sequence.
func denseSessionSave(kv *gemserver.KVCache, eng *cuda.Engine, args CLIArgs,
	history []chat.Message, thinking, file string) error {
	ident, err := sessionIdentity(args.ModelPath, int64(eng.KVStateSize(session.ProbeTokens)))
	if err != nil {
		return err
	}
	kv.Lock()
	tokens, state, err := kv.ExportSession()
	kv.Unlock()
	if err != nil {
		return err
	}
	client, err := json.Marshal(replClientState{Thinking: thinking, Dense: history})
	if err != nil {
		return err
	}
	meta := session.Meta{
		CreatedAt:  time.Now().UTC(),
		NTokens:    len(tokens),
		EngineKind: session.KindFlatKV,
		Model:      ident,
		Client:     client,
	}
	return session.WriteFile(file, meta, tokens, state)
}

// denseSessionLoad restores a saved session as the live sequence and returns
// the saved chat transcript + thinking level for the REPL to adopt.
func denseSessionLoad(kv *gemserver.KVCache, eng *cuda.Engine, args CLIArgs,
	file string) (history []chat.Message, thinking string, nTokens int, err error) {
	ident, err := sessionIdentity(args.ModelPath, int64(eng.KVStateSize(session.ProbeTokens)))
	if err != nil {
		return nil, "", 0, err
	}
	snap, err := session.ReadFile(file, sessionMaxStateBytes)
	if err != nil {
		return nil, "", 0, err
	}
	cur := session.Meta{EngineKind: session.KindFlatKV, Model: ident}
	if err := snap.Meta.Validate(cur); err != nil {
		return nil, "", 0, err
	}
	kv.Lock()
	err = kv.ImportSession(snap.Tokens, snap.State)
	kv.Unlock()
	if err != nil {
		return nil, "", 0, err
	}
	var cs replClientState
	if len(snap.Meta.Client) > 0 {
		if err := json.Unmarshal(snap.Meta.Client, &cs); err != nil {
			return nil, "", 0, fmt.Errorf("session client state is corrupt: %w", err)
		}
	}
	return cs.Dense, cs.Thinking, len(snap.Tokens), nil
}

// ─── Paged (Qwen hybrid) REPL: per-slot SeqState snapshots ──────────────────

// pagedSessionSave persists the conversation for the paged REPL. The paged
// REPL frees its slot after every turn, so the state is rebuilt by prefilling
// the rendered transcript into a scratch slot (full-history prefill — the
// cost of saving, not of every turn), snapshotting it, and freeing the slot.
// The snapshot includes the GDN recurrent state and conv rings (SeqStateSave),
// without which a hybrid session could not resume.
func pagedSessionSave(eng *cuda.Engine, tok *tokenizer.Tokenizer, dialect chat.Dialect,
	history []chat.RichMessage, args CLIArgs, thinking, file string) error {
	ident, err := sessionIdentity(args.ModelPath, eng.SeqStateSize(session.ProbeTokens))
	if err != nil {
		return err
	}
	if len(history) == 0 {
		return errors.New("no conversation to save")
	}
	promptToks := tok.Encode(dialect.Render(history, nil, false), true, false)
	if len(promptToks) == 0 {
		return errors.New("conversation renders to no tokens")
	}
	// Greedy scratch slot; the sampled first token is discarded.
	slot, _, err := eng.SeqAdd(promptToks, batch.SeqParams{})
	if err != nil {
		return fmt.Errorf("prefill for save: %w", err)
	}
	defer eng.SeqRemove(slot)
	n := eng.SeqNTokens(slot)
	if n <= 0 || n > len(promptToks) {
		return fmt.Errorf("engine reports %d committed tokens for a %d-token prompt", n, len(promptToks))
	}
	size := eng.SeqStateSize(n)
	if size <= 0 {
		return errors.New("engine reports no snapshot state for the conversation")
	}
	state := make([]byte, size)
	if err := eng.SeqStateSave(slot, state, n); err != nil {
		return err
	}
	client, err := json.Marshal(replClientState{Thinking: thinking, Paged: history})
	if err != nil {
		return err
	}
	meta := session.Meta{
		CreatedAt:  time.Now().UTC(),
		NTokens:    n,
		EngineKind: session.KindQ35Slot,
		Model:      ident,
		Client:     client,
	}
	return session.WriteFile(file, meta, promptToks[:n], state)
}

// pagedSessionLoad reads + validates a session for the paged REPL. The state
// is NOT pushed into the engine here (there is no live slot between turns);
// the caller keeps the snapshot and admits the next turn through
// pagedSessionAdmit, which restores it into a fresh slot.
func pagedSessionLoad(eng *cuda.Engine, args CLIArgs, file string) (
	snap *session.Snapshot, history []chat.RichMessage, thinking string, err error) {
	ident, err := sessionIdentity(args.ModelPath, eng.SeqStateSize(session.ProbeTokens))
	if err != nil {
		return nil, nil, "", err
	}
	snap, err = session.ReadFile(file, sessionMaxStateBytes)
	if err != nil {
		return nil, nil, "", err
	}
	cur := session.Meta{EngineKind: session.KindQ35Slot, Model: ident}
	if err := snap.Meta.Validate(cur); err != nil {
		return nil, nil, "", err
	}
	if want := eng.SeqStateSize(len(snap.Tokens)); want != int64(len(snap.State)) {
		return nil, nil, "", fmt.Errorf("session state is %d B but the engine lays out %d tokens as %d B",
			len(snap.State), len(snap.Tokens), want)
	}
	var cs replClientState
	if len(snap.Meta.Client) > 0 {
		if err := json.Unmarshal(snap.Meta.Client, &cs); err != nil {
			return nil, nil, "", fmt.Errorf("session client state is corrupt: %w", err)
		}
	}
	return snap, cs.Paged, cs.Thinking, nil
}

// pagedSessionAdmit starts a turn, restoring the pending session when the
// prompt strictly extends it (restore + suffix-only prefill: the restored
// prefix costs zero prefill tokens); otherwise it falls back to a cold
// SeqAdd of the full prompt. restored is the number of prompt tokens served
// by the snapshot (0 on the cold path).
func pagedSessionAdmit(eng *cuda.Engine, snap *session.Snapshot,
	promptToks []int32, params batch.SeqParams) (slot int, first int32, restored int, err error) {
	if snap != nil && len(snap.Tokens) < len(promptToks) && tokensArePrefix(snap.Tokens, promptToks) {
		n := len(snap.Tokens)
		slot, err = eng.SeqOpen(params)
		if err == nil {
			if rerr := eng.SeqStateRestore(slot, snap.State, n); rerr == nil {
				first, err = eng.SeqPrefillChunk(slot, promptToks[n:], true)
				if err == nil {
					return slot, first, n, nil
				}
			} else {
				err = rerr
			}
			// Failed restore/prefill: the slot state is unreliable — free it
			// and fall through to the cold path.
			eng.SeqRemove(slot)
			fmt.Fprintf(os.Stderr, "fucina: session restore failed (%v) — cold prefill\n", err)
		}
	}
	slot, first, err = eng.SeqAdd(promptToks, params)
	return slot, first, 0, err
}

func tokensArePrefix(p, s []int32) bool {
	if len(p) > len(s) {
		return false
	}
	for i, t := range p {
		if s[i] != t {
			return false
		}
	}
	return true
}
