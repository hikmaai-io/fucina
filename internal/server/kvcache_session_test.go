// ABOUTME: Tests for KVCache session export/import/seed — byte-exact round-trip
// ABOUTME: through the disk format and rejection of corrupt/mismatched state.
package server

import (
	"bytes"
	"testing"

	"github.com/hikmaai-io/fucina/internal/session"
)

// TestSessionExportImportRoundTrip drives the full disk path: build a live
// sequence, export it, serialize + reload through the session format, import
// into a FRESH cache/engine pair (the process-restart simulation), and verify
// the next Prefill reuses the whole restored prefix with zero recompute.
func TestSessionExportImportRoundTrip(t *testing.T) {
	f := &fakeSnapEngine{fakeEngine: fakeEngine{window: 1 << 20}}
	kv := NewKVCache(f)
	prompt := seq(100, 2000)
	kv.Lock()
	if _, err := kv.Prefill(prompt); err != nil {
		t.Fatal(err)
	}
	kv.AppendDecoded(9001) // a generated token, committed
	f.tokens = append(f.tokens, 9001)
	tokens, state, err := kv.ExportSession()
	kv.Unlock()
	if err != nil {
		t.Fatalf("ExportSession: %v", err)
	}
	if len(tokens) != 2001 {
		t.Fatalf("exported %d tokens, want 2001", len(tokens))
	}

	// Through the disk format (in memory).
	meta := session.Meta{NTokens: len(tokens), EngineKind: session.KindFlatKV}
	var buf bytes.Buffer
	if err := session.Write(&buf, meta, tokens, state); err != nil {
		t.Fatalf("session.Write: %v", err)
	}
	snap, err := session.Read(&buf, int64(len(state)))
	if err != nil {
		t.Fatalf("session.Read: %v", err)
	}
	if !bytes.Equal(snap.State, state) {
		t.Fatal("state not byte-exact through the disk format")
	}

	// "Restart": fresh engine, fresh cache.
	f2 := &fakeSnapEngine{fakeEngine: fakeEngine{window: 1 << 20}}
	kv2 := NewKVCache(f2)
	kv2.Lock()
	defer kv2.Unlock()
	if err := kv2.ImportSession(snap.Tokens, snap.State); err != nil {
		t.Fatalf("ImportSession: %v", err)
	}
	if f2.restores != 1 {
		t.Errorf("engine restores = %d, want 1", f2.restores)
	}

	// Continue the conversation: prompt extends the restored session.
	next := append(append([]int32(nil), snap.Tokens...), seq(5000, 30)...)
	res, err := kv2.Prefill(next)
	if err != nil {
		t.Fatalf("Prefill after import: %v", err)
	}
	if res.ReusedTokens != len(snap.Tokens) {
		t.Errorf("reused %d tokens, want the full session %d", res.ReusedTokens, len(snap.Tokens))
	}
	if res.NewTokens != 30 {
		t.Errorf("prefilled %d new tokens, want 30", res.NewTokens)
	}
}

func TestSessionExportEmpty(t *testing.T) {
	f := &fakeSnapEngine{fakeEngine: fakeEngine{window: 1 << 20}}
	kv := NewKVCache(f)
	kv.Lock()
	defer kv.Unlock()
	if _, _, err := kv.ExportSession(); err == nil {
		t.Error("ExportSession on an empty cache did not error")
	}
}

func TestSessionImportSizeMismatch(t *testing.T) {
	f := &fakeSnapEngine{fakeEngine: fakeEngine{window: 1 << 20}}
	kv := NewKVCache(f)
	kv.Lock()
	defer kv.Unlock()
	tokens := seq(0, 100)
	// fakeSnapEngine states are 4 B/token; hand it a wrong-size state.
	if err := kv.ImportSession(tokens, make([]byte, 100*4-1)); err == nil {
		t.Error("ImportSession accepted a state whose size does not match the engine layout")
	}
	if f.restores != 0 {
		t.Error("mismatched state must be rejected BEFORE touching the engine")
	}
}

func TestSessionSeedSnapshotRestoresOnMatch(t *testing.T) {
	f := &fakeSnapEngine{fakeEngine: fakeEngine{window: 1 << 20}}
	kv := NewKVCache(f)

	// A session from a previous process: 3000 tokens.
	sessTokens := seq(0, 3000)
	sessState := make([]byte, len(sessTokens)*4)
	for i, v := range sessTokens {
		sessState[i*4], sessState[i*4+1], sessState[i*4+2], sessState[i*4+3] =
			byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
	}

	kv.Lock()
	if err := kv.SeedSnapshot(sessTokens, sessState); err != nil {
		t.Fatalf("SeedSnapshot: %v", err)
	}
	// An unrelated small request runs first — must not disturb the seed.
	if _, err := kv.Prefill(seq(90000, 40)); err != nil {
		t.Fatal(err)
	}
	// Now the session's continuation arrives.
	next := append(append([]int32(nil), sessTokens...), seq(70000, 50)...)
	res, err := kv.Prefill(next)
	kv.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if f.restores != 1 {
		t.Errorf("engine restores = %d, want 1 (seeded snapshot restored)", f.restores)
	}
	if res.ReusedTokens != len(sessTokens) {
		t.Errorf("reused %d, want the full seeded session %d", res.ReusedTokens, len(sessTokens))
	}
	if res.NewTokens != 50 {
		t.Errorf("prefilled %d new tokens, want 50", res.NewTokens)
	}
}

func TestSessionSeedSnapshotValidates(t *testing.T) {
	f := &fakeSnapEngine{fakeEngine: fakeEngine{window: 1 << 20}}
	kv := NewKVCache(f)
	kv.Lock()
	if err := kv.SeedSnapshot(nil, []byte{1}); err == nil {
		t.Error("SeedSnapshot accepted empty tokens")
	}
	if err := kv.SeedSnapshot(seq(0, 8), make([]byte, 3)); err == nil {
		t.Error("SeedSnapshot accepted a state size that does not match the engine layout")
	}
	kv.Unlock()
	kv.SetSnapshotBudget(0) // takes the lock itself
	kv.Lock()
	if err := kv.SeedSnapshot(seq(0, 8), make([]byte, 32)); err == nil {
		t.Error("SeedSnapshot accepted a seed with snapshotting disabled")
	}
	kv.Unlock()
}
