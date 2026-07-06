package tokenizer

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

// ─── GGUF v3 serialization helpers ────────────────────────────────

// gw is a tiny little-endian writer for building GGUF bytes in memory.
type gw struct {
	buf []byte
}

func (w *gw) u32(v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	w.buf = append(w.buf, b[:]...)
}

func (w *gw) u64(v uint64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	w.buf = append(w.buf, b[:]...)
}

func (w *gw) f32(v float32) {
	w.u32(math.Float32bits(v))
}

// str writes a GGUF string: u64 length + raw bytes.
func (w *gw) str(s string) {
	w.u64(uint64(len(s)))
	w.buf = append(w.buf, s...)
}

// kvStringArray writes a KV pair whose value is a string array.
func (w *gw) kvStringArray(key string, vals []string) {
	w.str(key)
	w.u32(ggufTypeArray)
	w.u32(ggufTypeString)
	w.u64(uint64(len(vals)))
	for _, s := range vals {
		w.str(s)
	}
}

// kvF32Array writes a KV pair whose value is a float32 array.
func (w *gw) kvF32Array(key string, vals []float32) {
	w.str(key)
	w.u32(ggufTypeArray)
	w.u32(ggufTypeFloat32)
	w.u64(uint64(len(vals)))
	for _, v := range vals {
		w.f32(v)
	}
}

// kvU32 writes a KV pair whose value is a u32 scalar.
func (w *gw) kvU32(key string, v uint32) {
	w.str(key)
	w.u32(ggufTypeUint32)
	w.u32(v)
}

// kvString writes a KV pair whose value is a string scalar.
func (w *gw) kvString(key, val string) {
	w.str(key)
	w.u32(ggufTypeString)
	w.str(val)
}

// kvBool writes a KV pair whose value is a bool scalar (single byte).
func (w *gw) kvBool(key string, v bool) {
	w.str(key)
	w.u32(ggufTypeBool)
	var b byte
	if v {
		b = 1
	}
	w.buf = append(w.buf, b)
}

// buildTestGGUF serializes a minimal valid GGUF v3 binary in memory with the
// metadata the tokenizer parser reads: tokens, scores, and the bos/eos/pad ids.
func buildTestGGUF(tokens []string, scores []float32, bos, eos, pad int32) []byte {
	w := &gw{}

	// header: magic, version, tensor_count(0), metadata_kv_count
	w.u32(0x46554747) // "GGUF"
	w.u32(3)          // version
	w.u64(0)          // tensor_count

	const kvCount = 5
	w.u64(kvCount)

	w.kvStringArray("tokenizer.ggml.tokens", tokens)
	w.kvF32Array("tokenizer.ggml.scores", scores)
	w.kvU32("tokenizer.ggml.bos_token_id", uint32(bos))
	w.kvU32("tokenizer.ggml.eos_token_id", uint32(eos))
	w.kvU32("tokenizer.ggml.padding_token_id", uint32(pad))

	return w.buf
}

// ─── shared test vocab ────────────────────────────────────────────

// makeVocab returns a vocab (with the required special, byte, piece, and
// gemma-4 control tokens), the matching scores slice, and a name→id map.
func makeVocab() (tokens []string, scores []float32, idx map[string]int32) {
	tokens = []string{
		"<pad>", // 0
		"<eos>", // 1
		"<bos>", // 2
		"<unk>", // 3
	}
	// byte tokens for the full range so byte-fallback always resolves.
	for b := 0; b < 256; b++ {
		tokens = append(tokens, byteTokenStr(b))
	}
	// pieces (after the 256 byte tokens, ids 260+)
	pieces := []string{
		"▁hello", "hello", "▁world", "world", "▁",
		"h", "e", "l", "o",
		// boundary-straddling piece: greedy matching would take "o<" over "o"
		// before a control marker (the Encode pre-split regression).
		"o<",
		// gemma-4 control tokens
		"<|turn>", "<turn|>", "<|channel>", "<channel|>",
		"<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
		"<|tool_response>", "<tool_response|>", `<|"|>`,
	}
	tokens = append(tokens, pieces...)

	scores = make([]float32, len(tokens))
	for i := range scores {
		scores[i] = float32(-i)
	}
	idx = make(map[string]int32, len(tokens))
	for i, s := range tokens {
		if _, ok := idx[s]; !ok {
			idx[s] = int32(i)
		}
	}
	return tokens, scores, idx
}

func byteTokenStr(b int) string {
	const hex = "0123456789ABCDEF"
	return "<0x" + string([]byte{hex[(b>>4)&0xF], hex[b&0xF]}) + ">"
}

// newTestTokenizer builds the standard tokenizer used by most tests.
func newTestTokenizer(t *testing.T) (*Tokenizer, map[string]int32) {
	t.Helper()
	tokens, scores, idx := makeVocab()
	data := buildTestGGUF(tokens, scores, idx["<bos>"], idx["<eos>"], idx["<pad>"])
	tk, err := New(data, int64(len(data)))
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	return tk, idx
}

// ─── New() ────────────────────────────────────────────────────────

func TestNew_ParsesHeaderAndVocab(t *testing.T) {
	tk, idx := newTestTokenizer(t)

	if got := tk.NumTokens(); got != len(idx) {
		t.Errorf("NumTokens()=%d, want %d", got, len(idx))
	}
	if tk.BOS != idx["<bos>"] {
		t.Errorf("BOS=%d, want %d", tk.BOS, idx["<bos>"])
	}
	if tk.EOS != idx["<eos>"] {
		t.Errorf("EOS=%d, want %d", tk.EOS, idx["<eos>"])
	}
	if tk.PAD != idx["<pad>"] {
		t.Errorf("PAD=%d, want %d", tk.PAD, idx["<pad>"])
	}
}

func TestNew_ResolvesControlTokensByString(t *testing.T) {
	tk, idx := newTestTokenizer(t)

	cases := []struct {
		name string
		got  int32
		want int32
	}{
		{"StartOfTurn", tk.StartOfTurn, idx["<|turn>"]},
		{"EndOfTurn", tk.EndOfTurn, idx["<turn|>"]},
		{"ChannelOpen", tk.ChannelOpen, idx["<|channel>"]},
		{"ChannelEnd", tk.ChannelEnd, idx["<channel|>"]},
		{"ToolOpen", tk.ToolOpen, idx["<|tool>"]},
		{"ToolEnd", tk.ToolEnd, idx["<tool|>"]},
		{"ToolCallOpen", tk.ToolCallOpen, idx["<|tool_call>"]},
		{"ToolCallEnd", tk.ToolCallEnd, idx["<tool_call|>"]},
		{"ToolRespOpen", tk.ToolRespOpen, idx["<|tool_response>"]},
		{"ToolRespEnd", tk.ToolRespEnd, idx["<tool_response|>"]},
		{"StringDelim", tk.StringDelim, idx[`<|"|>`]},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s=%d, want %d", c.name, c.got, c.want)
		}
	}
}

func TestNew_BadMagic(t *testing.T) {
	tokens, scores, idx := makeVocab()
	data := buildTestGGUF(tokens, scores, idx["<bos>"], idx["<eos>"], idx["<pad>"])
	binary.LittleEndian.PutUint32(data[0:4], 0xDEADBEEF)
	if _, err := New(data, int64(len(data))); err == nil {
		t.Fatal("expected error on bad magic, got nil")
	} else if !strings.Contains(err.Error(), "magic") {
		t.Errorf("expected magic error, got %v", err)
	}
}

func TestNew_TooSmall(t *testing.T) {
	data := []byte{0x47, 0x47, 0x55, 0x46} // just the magic, < 24 bytes
	if _, err := New(data, int64(len(data))); err == nil {
		t.Fatal("expected error on too-small file, got nil")
	}
}

func TestNew_Truncated(t *testing.T) {
	tokens, scores, idx := makeVocab()
	data := buildTestGGUF(tokens, scores, idx["<bos>"], idx["<eos>"], idx["<pad>"])
	// Chop off the back half — should fail mid-parse.
	trunc := data[:len(data)/2]
	if _, err := New(trunc, int64(len(trunc))); err == nil {
		t.Fatal("expected error on truncated file, got nil")
	}
}

func TestNew_MissingTokensKey(t *testing.T) {
	w := &gw{}
	w.u32(0x46554747)
	w.u32(3)
	w.u64(0)
	w.u64(1) // 1 kv pair, but not the tokens key
	w.kvU32("tokenizer.ggml.bos_token_id", 2)
	if _, err := New(w.buf, int64(len(w.buf))); err == nil {
		t.Fatal("expected error when tokens key missing, got nil")
	} else if !strings.Contains(err.Error(), "no tokens") {
		t.Errorf("expected 'no tokens' error, got %v", err)
	}
}

// ─── Encode / Decode round-trip ───────────────────────────────────

func TestEncodeDecode_RoundTrip(t *testing.T) {
	tk, _ := newTestTokenizer(t)

	ids := tk.Encode("hello world", false, false)
	if len(ids) == 0 {
		t.Fatal("Encode returned no tokens")
	}
	got := tk.Decode(ids)
	if got != "hello world" {
		t.Errorf("round-trip = %q, want %q", got, "hello world")
	}
}

func TestEncode_LeadingSpaceMarker(t *testing.T) {
	tk, idx := newTestTokenizer(t)

	// " world" → space normalized to ▁, should match "▁world".
	ids := tk.Encode(" world", false, false)
	if len(ids) != 1 || ids[0] != idx["▁world"] {
		t.Errorf("Encode(%q) = %v, want [%d]", " world", ids, idx["▁world"])
	}
	if got := tk.Decode(ids); got != " world" {
		t.Errorf("Decode = %q, want %q", got, " world")
	}
}

func TestEncode_AddBosEos(t *testing.T) {
	tk, _ := newTestTokenizer(t)

	ids := tk.Encode("hello", true, true)
	if len(ids) < 2 {
		t.Fatalf("expected BOS+content+EOS, got %v", ids)
	}
	if ids[0] != tk.BOS {
		t.Errorf("first token = %d, want BOS %d", ids[0], tk.BOS)
	}
	if ids[len(ids)-1] != tk.EOS {
		t.Errorf("last token = %d, want EOS %d", ids[len(ids)-1], tk.EOS)
	}
}

// ─── Byte fallback ────────────────────────────────────────────────

func TestEncode_ByteFallback(t *testing.T) {
	tk, idx := newTestTokenizer(t)

	// 'z' is not a piece in the vocab → must fall back to its <0x7A> byte token.
	ids := tk.Encode("z", false, false)
	if len(ids) != 1 {
		t.Fatalf("Encode(%q) = %v, want single byte token", "z", ids)
	}
	if ids[0] != idx["<0x7A>"] {
		t.Errorf("byte fallback id = %d, want %d (<0x7A>)", ids[0], idx["<0x7A>"])
	}
	if got := tk.Decode(ids); got != "z" {
		t.Errorf("Decode of byte fallback = %q, want %q", got, "z")
	}
}

func TestEncodeDecode_MultiByteUTF8ViaByteFallback(t *testing.T) {
	tk, _ := newTestTokenizer(t)

	// "é" (U+00E9) is not a piece → 2 byte tokens that must reassemble.
	const s = "é"
	ids := tk.Encode(s, false, false)
	if len(ids) != 2 {
		t.Fatalf("Encode(%q) = %v, want 2 byte tokens", s, ids)
	}
	if got := tk.Decode(ids); got != s {
		t.Errorf("multi-byte round-trip = %q, want %q", got, s)
	}
}

// ─── Decode vs DecodeRaw on control tokens ────────────────────────

func TestDecode_SkipsControlTokens(t *testing.T) {
	tk, idx := newTestTokenizer(t)

	ids := []int32{tk.BOS, idx["hello"], tk.StartOfTurn, idx["▁world"], tk.EndOfTurn, tk.EOS}
	got := tk.Decode(ids)
	if got != "hello world" {
		t.Errorf("Decode = %q, want %q (control tokens skipped)", got, "hello world")
	}
}

func TestDecodeRaw_KeepsControlTokens(t *testing.T) {
	tk, idx := newTestTokenizer(t)

	ids := []int32{idx["hello"], tk.ChannelOpen, idx["<|tool_call>"]}
	got := tk.DecodeRaw(ids)
	want := "hello<|channel><|tool_call>"
	if got != want {
		t.Errorf("DecodeRaw = %q, want %q", got, want)
	}
}

// ─── IsStop ───────────────────────────────────────────────────────

func TestIsStop(t *testing.T) {
	tk, idx := newTestTokenizer(t)

	if !tk.IsStop(tk.EOS) {
		t.Error("IsStop(EOS) = false, want true")
	}
	if !tk.IsStop(tk.EndOfTurn) {
		t.Error("IsStop(EndOfTurn) = false, want true")
	}
	if tk.IsStop(idx["hello"]) {
		t.Error("IsStop(hello) = true, want false")
	}
	if tk.IsStop(tk.StartOfTurn) {
		t.Error("IsStop(StartOfTurn) = true, want false")
	}
}

// ─── IsToolMarker ─────────────────────────────────────────────────

func TestIsToolMarker(t *testing.T) {
	tk, idx := newTestTokenizer(t)

	markers := []string{
		"<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
		"<|tool_response>", "<tool_response|>", `<|"|>`,
	}
	for _, m := range markers {
		if !tk.IsToolMarker(idx[m]) {
			t.Errorf("IsToolMarker(%q id=%d) = false, want true", m, idx[m])
		}
	}

	// Normal ids must not be tool markers.
	for _, m := range []string{"hello", "▁world", "<|turn>", "<|channel>"} {
		if tk.IsToolMarker(idx[m]) {
			t.Errorf("IsToolMarker(%q) = true, want false", m)
		}
	}
}

func TestIsToolMarker_NoToolTokens(t *testing.T) {
	// A vocab without any tool tokens → all tool ids should be -1 and
	// IsToolMarker must always return false (never matching real ids).
	tokens := []string{"<pad>", "<eos>", "<bos>", "<unk>", "hello", "world"}
	scores := make([]float32, len(tokens))
	data := buildTestGGUF(tokens, scores, 2, 1, 0)
	tk, err := New(data, int64(len(data)))
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}

	for _, id := range []int32{tk.ToolOpen, tk.ToolEnd, tk.ToolCallOpen,
		tk.ToolCallEnd, tk.ToolRespOpen, tk.ToolRespEnd, tk.StringDelim} {
		if id != -1 {
			t.Errorf("expected absent tool id to be -1, got %d", id)
		}
	}
	// Check across every real id (including 0..len-1) that nothing matches.
	for id := int32(0); id < int32(tk.NumTokens()); id++ {
		if tk.IsToolMarker(id) {
			t.Errorf("IsToolMarker(%d) = true on tool-less vocab, want false", id)
		}
	}
	// -1 must also be false.
	if tk.IsToolMarker(-1) {
		t.Error("IsToolMarker(-1) = true, want false")
	}
}

// ─── parseByteToken / hexNibble ───────────────────────────────────

func TestParseByteToken_AllValid(t *testing.T) {
	for b := 0; b < 256; b++ {
		// uppercase form
		s := byteTokenStr(b)
		got, ok := parseByteToken(s)
		if !ok || got != byte(b) {
			t.Errorf("parseByteToken(%q) = (%d,%v), want (%d,true)", s, got, ok, b)
		}
		// lowercase hex form is also accepted by hexNibble
		ls := strings.ToLower(s[:3]) + strings.ToLower(s[3:5]) + ">"
		lgot, lok := parseByteToken(ls)
		if !lok || lgot != byte(b) {
			t.Errorf("parseByteToken(%q) = (%d,%v), want (%d,true)", ls, lgot, lok, b)
		}
	}
}

func TestParseByteToken_Invalid(t *testing.T) {
	bad := []string{
		"",        // empty
		"<0x4>",   // too short
		"<0x412>", // too long
		"0x41",    // missing brackets
		"<1x41>",  // wrong prefix digit
		"<0y41>",  // wrong 'x'
		"<0x4G>",  // bad low nibble
		"<0xG1>",  // bad high nibble
		"<0x41]",  // wrong terminator
		"hello",   // not a byte token
		"<0xZZ>",  // both nibbles bad
		"▁hello",  // unicode piece
	}
	for _, s := range bad {
		if _, ok := parseByteToken(s); ok {
			t.Errorf("parseByteToken(%q) = ok, want false", s)
		}
	}
}

func TestHexNibble(t *testing.T) {
	valid := map[byte]byte{
		'0': 0, '9': 9, 'A': 10, 'F': 15, 'a': 10, 'f': 15,
	}
	for c, want := range valid {
		if got, ok := hexNibble(c); !ok || got != want {
			t.Errorf("hexNibble(%q) = (%d,%v), want (%d,true)", c, got, ok, want)
		}
	}
	for _, c := range []byte{'G', 'g', 'z', '/', ':', '@', ' ', '-'} {
		if _, ok := hexNibble(c); ok {
			t.Errorf("hexNibble(%q) = ok, want false", c)
		}
	}
}

// ─── TokenStr / NumTokens ─────────────────────────────────────────

func TestTokenStr(t *testing.T) {
	tk, idx := newTestTokenizer(t)

	if got := tk.TokenStr(idx["hello"]); got != "hello" {
		t.Errorf("TokenStr(hello id) = %q, want %q", got, "hello")
	}
	if got := tk.TokenStr(0); got != "<pad>" {
		t.Errorf("TokenStr(0) = %q, want %q", got, "<pad>")
	}
	// out-of-range → ""
	if got := tk.TokenStr(-1); got != "" {
		t.Errorf("TokenStr(-1) = %q, want \"\"", got)
	}
	if got := tk.TokenStr(int32(tk.NumTokens())); got != "" {
		t.Errorf("TokenStr(out-of-range) = %q, want \"\"", got)
	}
}

func TestNumTokens(t *testing.T) {
	tk, idx := newTestTokenizer(t)
	if got := tk.NumTokens(); got != len(idx) {
		t.Errorf("NumTokens() = %d, want %d", got, len(idx))
	}
}

// ─── Byte-level (gpt2/qwen2) BPE ──────────────────────────────────

// TestByteLevel_ByteToRuneSpace pins the GPT-2 byte↔unicode table's space slot:
// 0x20 must map to U+0120, and the map must round-trip every byte.
func TestByteLevel_ByteToRuneSpace(t *testing.T) {
	if got := byteToRune(' '); got != 'Ġ' {
		t.Errorf("byteToRune(space) = %U, want U+0120", got)
	}
	// printable identity sample
	if got := byteToRune('a'); got != 'a' {
		t.Errorf("byteToRune('a') = %U, want 'a'", got)
	}
	for b := 0; b < 256; b++ {
		r := byteToRune(byte(b))
		back, ok := runeToByte(r)
		if !ok || back != byte(b) {
			t.Errorf("round-trip byte %d via %U = (%d,%v)", b, r, back, ok)
		}
	}
}

// buildByteLevelGGUF serializes a minimal Qwen3-style byte-level GGUF: a full
// 256-byte alphabet plus the merged pieces, tokenizer.ggml.model="gpt2", the
// merges array, and add_bos_token (controllable).
func buildByteLevelGGUF(extraPieces, merges []string, addBOS bool) []byte {
	tokens := []string{"<pad>", "<eos>", "<bos>", "<unk>"}
	for b := 0; b < 256; b++ {
		tokens = append(tokens, string(byteToRune(byte(b))))
	}
	tokens = append(tokens, extraPieces...)

	w := &gw{}
	w.u32(0x46554747) // "GGUF"
	w.u32(3)          // version
	w.u64(0)          // tensor_count
	w.u64(6)          // metadata_kv_count

	w.kvStringArray("tokenizer.ggml.tokens", tokens)
	w.kvString("tokenizer.ggml.model", "gpt2")
	w.kvStringArray("tokenizer.ggml.merges", merges)
	w.kvBool("tokenizer.ggml.add_bos_token", addBOS)
	w.kvU32("tokenizer.ggml.bos_token_id", 2)
	w.kvU32("tokenizer.ggml.eos_token_id", 1)
	return w.buf
}

func TestByteLevelBPE_RoundTripAndFlags(t *testing.T) {
	const sp = "Ġ" // GPT-2 space marker (byteToRune(' '))
	// merges: build "ab" from a+b, and " cd" (sp+c+d) from sp+c then +d.
	merges := []string{
		"a b",
		sp + " c",
		sp + "c d",
	}
	pieces := []string{"ab", sp + "c", sp + "cd"}
	data := buildByteLevelGGUF(pieces, merges, false)

	tk, err := New(data, int64(len(data)))
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	if !tk.byteLevel {
		t.Fatal("byteLevel flag not set for tokenizer.ggml.model=gpt2")
	}
	if !tk.bpe {
		t.Fatal("bpe flag not set for gpt2 model")
	}
	if tk.byteFallback {
		t.Fatal("byteFallback must be off for byte-level BPE")
	}
	if tk.AddBOS {
		t.Fatal("AddBOS must be false (add_bos_token=false)")
	}

	// AddBOS honored: even with addBos=true the caller gets no BOS.
	ids := tk.Encode("ab cd", true, false)
	if len(ids) > 0 && ids[0] == tk.BOS {
		t.Errorf("BOS prepended despite add_bos_token=false: %v", ids)
	}

	wantIDs := []int32{tk.tokenToID["ab"], tk.tokenToID[sp+"cd"]}
	if len(ids) != len(wantIDs) || ids[0] != wantIDs[0] || ids[1] != wantIDs[1] {
		t.Errorf("Encode(%q) = %v, want %v", "ab cd", ids, wantIDs)
	}

	// Round-trip Encode→Decode.
	if got := tk.Decode(ids); got != "ab cd" {
		t.Errorf("round-trip = %q, want %q", got, "ab cd")
	}

	// Space maps through U+0120 inside an encoded piece: the second token's
	// string must start with the GPT-2 space marker.
	if s := tk.TokenStr(ids[1]); !strings.HasPrefix(s, sp) {
		t.Errorf("space token %q does not start with U+0120 marker", s)
	}
}

func TestByteLevelBPE_DefaultAddBOSAndUnmergedBytes(t *testing.T) {
	// No merges, no extra pieces, add_bos_token=true: every byte stays a single
	// symbol and resolves against the 256-byte alphabet (no <0xXX> fallback).
	data := buildByteLevelGGUF(nil, nil, true)
	tk, err := New(data, int64(len(data)))
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	if !tk.AddBOS {
		t.Fatal("AddBOS should be true (add_bos_token=true)")
	}
	ids := tk.Encode("hi!", true, false)
	if len(ids) < 1 || ids[0] != tk.BOS {
		t.Fatalf("expected leading BOS, got %v", ids)
	}
	if got := tk.Decode(ids); got != "hi!" {
		t.Errorf("round-trip = %q, want %q", got, "hi!")
	}
}

// Control markers must encode to their ids even when the preceding text ends
// in a character that forms a vocab piece with '<' ("hello<channel|>" must not
// become ... "o<" "channel" ... — that token mismatch silently broke KV prefix
// reuse for re-rendered reasoning channels ending without a newline).
func TestEncode_MarkerAfterTextWithoutBoundary(t *testing.T) {
	tk, idx := newTestTokenizer(t)
	got := tk.Encode("hello<channel|>", false, false)
	want := []int32{idx["hello"], idx["<channel|>"]}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Errorf("Encode(hello<channel|>) = %v, want %v (o< must not eat the marker)", got, want)
	}
	// And a marker mid-text re-splits cleanly on both sides.
	got = tk.Encode("hello<tool_call|>world", false, false)
	if len(got) != 3 || got[1] != idx["<tool_call|>"] {
		t.Errorf("Encode(hello<tool_call|>world) = %v, want marker id %d in the middle", got, idx["<tool_call|>"])
	}
}
