// ABOUTME: Tests for the versioned on-disk session snapshot format — byte-exact
// ABOUTME: round-trip, identity validation, and hostile-file corruption rejection.
package session

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func testMeta() Meta {
	return Meta{
		CreatedAt:  time.Date(2026, 7, 11, 12, 0, 0, 0, time.UTC),
		NTokens:    5,
		EngineKind: KindQ35Slot,
		Model: Identity{
			Path:       "/models/qwen3.5-35b",
			ConfigHash: 0xdeadbeefcafef00d,
			StateProbe: 123456,
		},
	}
}

func testTokens() []int32 { return []int32{760, 6511, 314, 9338, 369} }

func testState() []byte {
	state := make([]byte, 4096)
	for i := range state {
		state[i] = byte(i * 31)
	}
	return state
}

// encode is the test helper: serialize a snapshot to bytes.
func encode(t *testing.T, meta Meta, tokens []int32, state []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := Write(&buf, meta, tokens, state); err != nil {
		t.Fatalf("Write: %v", err)
	}
	return buf.Bytes()
}

func TestRoundTrip(t *testing.T) {
	meta, tokens, state := testMeta(), testTokens(), testState()
	raw := encode(t, meta, tokens, state)

	snap, err := Read(bytes.NewReader(raw), int64(len(state)))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if !snap.Meta.CreatedAt.Equal(meta.CreatedAt) {
		t.Errorf("CreatedAt = %v, want %v", snap.Meta.CreatedAt, meta.CreatedAt)
	}
	if snap.Meta.EngineKind != meta.EngineKind || snap.Meta.NTokens != meta.NTokens {
		t.Errorf("meta mismatch: %+v vs %+v", snap.Meta, meta)
	}
	if snap.Meta.Model != meta.Model {
		t.Errorf("identity mismatch: %+v vs %+v", snap.Meta.Model, meta.Model)
	}
	if len(snap.Tokens) != len(tokens) {
		t.Fatalf("tokens len = %d, want %d", len(snap.Tokens), len(tokens))
	}
	for i, tk := range tokens {
		if snap.Tokens[i] != tk {
			t.Errorf("token[%d] = %d, want %d", i, snap.Tokens[i], tk)
		}
	}
	if !bytes.Equal(snap.State, state) {
		t.Error("state bytes not byte-exact after round-trip")
	}
}

func TestFileRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sess.fcsess")
	meta, tokens, state := testMeta(), testTokens(), testState()
	if err := WriteFile(path, meta, tokens, state); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	snap, err := ReadFile(path, 1<<20)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if !bytes.Equal(snap.State, state) || len(snap.Tokens) != len(tokens) {
		t.Error("file round-trip not byte-exact")
	}
	// Atomic write: no stray tmp files left behind.
	entries, _ := os.ReadDir(filepath.Dir(path))
	if len(entries) != 1 {
		t.Errorf("expected 1 file in dir after WriteFile, found %d", len(entries))
	}
}

func TestEmptyTokensRejected(t *testing.T) {
	var buf bytes.Buffer
	if err := Write(&buf, testMeta(), nil, testState()); err == nil {
		t.Error("Write accepted empty token history")
	}
	if err := Write(&buf, testMeta(), testTokens(), nil); err == nil {
		t.Error("Write accepted empty state")
	}
}

// flip returns a copy of raw with one byte at off flipped.
func flip(raw []byte, off int) []byte {
	c := append([]byte(nil), raw...)
	c[off] ^= 0xff
	return c
}

func TestCorruptionRejected(t *testing.T) {
	meta, tokens, state := testMeta(), testTokens(), testState()
	raw := encode(t, meta, tokens, state)

	// Find section offsets: magic(12) version(4) metaLen(4).
	metaLen := int(binary.LittleEndian.Uint32(raw[16:20]))
	metaOff := 20
	tokOff := metaOff + metaLen + 8 + 4 // +meta fnv +nTokens
	stateOff := tokOff + len(tokens)*4 + 8 + 8

	cases := []struct {
		name string
		off  int
	}{
		{"magic", 0},
		{"version", 12},
		{"meta-json", metaOff},
		{"meta-checksum", metaOff + metaLen},
		{"token-bytes", tokOff},
		{"token-checksum", tokOff + len(tokens)*4},
		{"state-bytes", stateOff},
		{"state-checksum", len(raw) - 8},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Read(bytes.NewReader(flip(raw, tc.off)), int64(len(state))); err == nil {
				t.Errorf("corruption at %s (offset %d) not rejected", tc.name, tc.off)
			}
		})
	}
}

func TestTruncationRejected(t *testing.T) {
	raw := encode(t, testMeta(), testTokens(), testState())
	for _, n := range []int{0, 5, 11, 15, 19, 40, len(raw) / 2, len(raw) - 1} {
		if _, err := Read(bytes.NewReader(raw[:n]), 1<<20); err == nil {
			t.Errorf("truncation to %d bytes not rejected", n)
		}
	}
}

func TestTrailingGarbageRejected(t *testing.T) {
	raw := encode(t, testMeta(), testTokens(), testState())
	if _, err := Read(bytes.NewReader(append(raw, 0xAA)), 1<<20); err == nil {
		t.Error("trailing garbage not rejected")
	}
}

// TestHostileLengths hand-crafts headers with absurd section lengths and checks
// they are rejected BEFORE any large allocation (same standard as the
// safetensors loader: never trust a length field from disk).
func TestHostileLengths(t *testing.T) {
	raw := encode(t, testMeta(), testTokens(), testState())

	huge := append([]byte(nil), raw...)
	binary.LittleEndian.PutUint32(huge[16:20], 1<<30) // metaLen = 1 GiB
	if _, err := Read(bytes.NewReader(huge), 1<<20); err == nil {
		t.Error("1 GiB meta length not rejected")
	}

	metaLen := int(binary.LittleEndian.Uint32(raw[16:20]))
	tokCountOff := 20 + metaLen + 8
	hugeTok := append([]byte(nil), raw...)
	binary.LittleEndian.PutUint32(hugeTok[tokCountOff:], 1<<31-1)
	if _, err := Read(bytes.NewReader(hugeTok), 1<<20); err == nil {
		t.Error("2^31 token count not rejected")
	}

	// stateLen over the caller's cap must be rejected even with a valid header.
	if _, err := Read(bytes.NewReader(raw), int64(len(testState()))-1); err == nil {
		t.Error("state over maxState cap not rejected")
	}
	if _, err := Read(bytes.NewReader(raw), 0); err == nil {
		t.Error("maxState=0 accepted a nonempty state")
	}
}

func TestMetaTokenCountCrossCheck(t *testing.T) {
	meta := testMeta()
	meta.NTokens = 999 // lies about the token section
	var buf bytes.Buffer
	if err := Write(&buf, meta, testTokens(), testState()); err == nil {
		t.Error("Write accepted meta.NTokens != len(tokens)")
	}
}

func TestValidate(t *testing.T) {
	saved := testMeta()

	ok := saved
	if err := saved.Validate(ok); err != nil {
		t.Errorf("identical identity rejected: %v", err)
	}

	// A different absolute path to the SAME model (same config hash) is fine.
	moved := saved
	moved.Model.Path = "/elsewhere/qwen3.5-35b"
	if err := saved.Validate(moved); err != nil {
		t.Errorf("same model at different path rejected: %v", err)
	}

	cases := []struct {
		name    string
		mutate  func(*Meta)
		wantSub string
	}{
		{"config hash", func(m *Meta) { m.Model.ConfigHash++ }, "config"},
		{"engine kind", func(m *Meta) { m.EngineKind = KindFlatKV }, "engine kind"},
		{"state probe", func(m *Meta) { m.Model.StateProbe += 8 }, "geometry"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cur := saved
			tc.mutate(&cur)
			err := saved.Validate(cur)
			if err == nil {
				t.Fatalf("%s mismatch not rejected", tc.name)
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Errorf("error %q does not mention %q", err, tc.wantSub)
			}
		})
	}
}

func TestHashModelConfigDir(t *testing.T) {
	dir := t.TempDir()
	cfg := filepath.Join(dir, "config.json")
	if err := os.WriteFile(cfg, []byte(`{"arch":"qwen3_5"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	h1, err := HashModelConfig(dir)
	if err != nil {
		t.Fatalf("HashModelConfig(dir): %v", err)
	}
	h2, _ := HashModelConfig(dir)
	if h1 != h2 || h1 == 0 {
		t.Errorf("hash not stable/nonzero: %x vs %x", h1, h2)
	}
	if err := os.WriteFile(cfg, []byte(`{"arch":"gemma4"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	h3, _ := HashModelConfig(dir)
	if h3 == h1 {
		t.Error("different config bytes produced the same hash")
	}
}

func TestHashModelConfigGGUF(t *testing.T) {
	dir := t.TempDir()
	gguf := filepath.Join(dir, "model.gguf")
	blob := make([]byte, 128<<10)
	for i := range blob {
		blob[i] = byte(i)
	}
	if err := os.WriteFile(gguf, blob, 0o644); err != nil {
		t.Fatal(err)
	}
	h1, err := HashModelConfig(gguf)
	if err != nil {
		t.Fatalf("HashModelConfig(gguf): %v", err)
	}
	// Change a byte inside the hashed prefix window.
	blob[1024] ^= 0xff
	if err := os.WriteFile(gguf, blob, 0o644); err != nil {
		t.Fatal(err)
	}
	h2, _ := HashModelConfig(gguf)
	if h1 == h2 {
		t.Error("GGUF header change did not change the hash")
	}
}
