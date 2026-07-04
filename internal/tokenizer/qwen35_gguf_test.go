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

	// The gemma turn markers must never fall back to their gemma default ids
	// (which collide with byte tokens — id 106 is "®"); the ChatML mapping
	// resolves the REAL turn markers instead: <|im_start|>/<|im_end|>, the
	// <think>/<tool_call> spans, and <|endoftext|> as the secondary stop.
	if tk.EndOfTurn == 106 {
		t.Error("EndOfTurn resolved to 106 (== \"®\" byte token) — spurious-stop bug")
	}
	if tk.EndOfTurn != 248046 {
		t.Errorf("EndOfTurn=%d, want 248046 (<|im_end|>)", tk.EndOfTurn)
	}
	if tk.StartOfTurn != 248045 {
		t.Errorf("StartOfTurn=%d, want 248045 (<|im_start|>)", tk.StartOfTurn)
	}
	if tk.EOS2 != 248044 {
		t.Errorf("EOS2=%d, want 248044 (<|endoftext|>)", tk.EOS2)
	}
	if tk.ChannelOpen != 248068 || tk.ChannelEnd != 248069 {
		t.Errorf("think span = %d/%d, want 248068/248069", tk.ChannelOpen, tk.ChannelEnd)
	}
	if tk.ToolCallOpen != 248058 || tk.ToolCallEnd != 248059 {
		t.Errorf("tool_call span = %d/%d, want 248058/248059", tk.ToolCallOpen, tk.ToolCallEnd)
	}
	if tk.IsStop(106) {
		t.Error("IsStop(106)=true — would stop generation on a real byte token")
	}
	if !tk.IsStop(tk.EOS) || !tk.IsStop(248044) {
		t.Error("IsStop must fire on <|im_end|> and <|endoftext|>")
	}

	// ChatML markers must encode as single tokens (pre-split registration) —
	// the rendered chat template depends on it.
	mids := tk.Encode("<|im_start|>user\nhi<|im_end|>\n", false, false)
	if len(mids) < 2 || mids[0] != 248045 || mids[len(mids)-2] != 248046 {
		t.Errorf("ChatML markers did not encode as single tokens: %v", mids)
	}

	// Round-trip: decode(encode(s)) reproduces s.
	const s = "Hello, world! 12345"
	if rt := tk.Decode(tk.Encode(s, false, false)); rt != s {
		t.Errorf("round-trip: got %q want %q", rt, s)
	}
}
