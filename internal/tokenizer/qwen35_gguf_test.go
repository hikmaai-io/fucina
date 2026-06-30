package tokenizer

import (
	"os"
	"reflect"
	"testing"
)

// qwen35GGUFPath returns the Qwen3.5-9B GGUF to test the served-path tokenizer
// against (env override, else the dev checkpoint), or "" to skip — the 5.6 GB
// file is not a committed fixture.
func qwen35GGUFPath() string {
	if p := os.Getenv("FUCINA_QWEN35_GGUF"); p != "" {
		return p
	}
	const dev = "/opt/spark/models/Qwen3.5-9B-abliterated-Q4_K_M.gguf"
	if _, err := os.Stat(dev); err == nil {
		return dev
	}
	return ""
}

// TestQwen35_ServedTokenizer locks the served-path invariants that the CUDA-level
// parity gates (which use hard-coded ids) cannot catch:
//   - the gpt2 vocab loads (248320 tokens),
//   - Encode(prompt, addBos=true, ...) does NOT prepend a spurious BOS (the GGUF
//     omits add_bos_token, so it must default to false, NOT gemma's true) — the
//     pinned llama-simple ids have no BOS,
//   - EOS is read from the GGUF (248046),
//   - the gemma turn markers do NOT leak in as real byte tokens (EndOfTurn must
//     not be id 106 == "®", or IsStop fires spuriously mid-generation).
func TestQwen35_ServedTokenizer(t *testing.T) {
	path := qwen35GGUFPath()
	if path == "" {
		t.Skip("no Qwen3.5 GGUF available (set FUCINA_QWEN35_GGUF)")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read gguf: %v", err)
	}
	tk, err := New(data, int64(len(data)))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if tk.vocabSize != 248320 {
		t.Errorf("vocabSize=%d, want 248320", tk.vocabSize)
	}
	if !tk.gpt2 {
		t.Error("expected gpt2 byte-level mode")
	}
	if tk.EOS != 248046 {
		t.Errorf("EOS=%d, want 248046", tk.EOS)
	}
	if tk.addBOS {
		t.Error("addBOS=true, want false (GGUF omits add_bos_token -> gpt2 default false)")
	}

	// Pinned parity prompt: "The capital of France is" -> [760,6511,314,9338,369]
	// (no leading BOS), matching llama-simple on this exact GGUF.
	want := []int32{760, 6511, 314, 9338, 369}
	got := tk.Encode("The capital of France is", true, false)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Encode(addBos=true):\n  got  %v\n  want %v", got, want)
	}

	// The gemma turn markers are absent from the gpt2 vocab; they must resolve to
	// -1, NOT the gemma default ids (which collide with byte tokens). Otherwise
	// IsStop(106) would terminate generation on a "®".
	if tk.EndOfTurn == 106 {
		t.Error("EndOfTurn resolved to 106 (== \"®\" byte token) — spurious-stop bug")
	}
	if tk.EndOfTurn != -1 {
		t.Errorf("EndOfTurn=%d, want -1 (marker absent in gpt2 vocab)", tk.EndOfTurn)
	}
	if tk.IsStop(106) {
		t.Error("IsStop(106)=true — would stop generation on a real byte token")
	}
	if !tk.IsStop(tk.EOS) {
		t.Error("IsStop(EOS)=false, want true")
	}

	// Round-trip: decode(encode(s)) reproduces s.
	const s = "Hello, world! 12345"
	if rt := tk.Decode(tk.Encode(s, false, false)); rt != s {
		t.Errorf("round-trip: got %q want %q", rt, s)
	}
}
