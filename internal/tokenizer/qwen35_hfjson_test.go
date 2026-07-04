package tokenizer

import (
	"os"
	"testing"
)

// qwen35JSONPath returns the Qwen3.5 HF tokenizer.json the safetensors serving
// path loads (env override, else the dev checkpoint), or "" to skip.
func qwen35JSONPath() string {
	if p := os.Getenv("FUCINA_QWEN35_TOKENIZER"); p != "" {
		return p
	}
	const dev = "/opt/spark/models/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/0b2752837483aa34b3db6e83e151b150c0e00e49/tokenizer.json"
	if _, err := os.Stat(dev); err == nil {
		return dev
	}
	return ""
}

// TestQwen35_HFJSONServedTokenizer locks the ChatML invariants on the
// tokenizer.json path (what `--tokenizer` serves for the FP8/MoE safetensors
// checkpoints). Before the ChatML mapping this path kept the Gemma defaults:
// EOS=1 (a live byte token, spurious stops), BOS=2 silently prepended to every
// prompt, EndOfTurn=106 ("®").
func TestQwen35_HFJSONServedTokenizer(t *testing.T) {
	path := qwen35JSONPath()
	if path == "" {
		t.Skip("no Qwen3.5 tokenizer.json available (set FUCINA_QWEN35_TOKENIZER)")
	}
	tk, err := NewFromHFJSON(path)
	if err != nil {
		t.Fatalf("NewFromHFJSON: %v", err)
	}

	if tk.EOS != 248046 || tk.EOS2 != 248044 {
		t.Errorf("EOS/EOS2 = %d/%d, want 248046 (<|im_end|>) / 248044 (<|endoftext|>)", tk.EOS, tk.EOS2)
	}
	if tk.BOS != -1 || tk.addBOS {
		t.Errorf("BOS=%d addBOS=%v — ChatML must not prepend a BOS", tk.BOS, tk.addBOS)
	}
	if tk.StartOfTurn != 248045 || tk.EndOfTurn != 248046 {
		t.Errorf("im_start/im_end = %d/%d", tk.StartOfTurn, tk.EndOfTurn)
	}
	if tk.ChannelOpen != 248068 || tk.ChannelEnd != 248069 {
		t.Errorf("think span = %d/%d", tk.ChannelOpen, tk.ChannelEnd)
	}
	if tk.ToolCallOpen != 248058 || tk.ToolCallEnd != 248059 {
		t.Errorf("tool_call span = %d/%d", tk.ToolCallOpen, tk.ToolCallEnd)
	}
	if tk.IsStop(1) || tk.IsStop(106) {
		t.Error("IsStop fires on live byte tokens (the pre-mapping bug)")
	}

	// A rendered ChatML prompt: markers single-token, no BOS even with
	// addBos=true from the caller, round-trip through DecodeRaw.
	prompt := "<|im_start|>user\nWeather in Paris?<|im_end|>\n<|im_start|>assistant\n<think>\n"
	ids := tk.Encode(prompt, true, false)
	if len(ids) == 0 || ids[0] != 248045 {
		t.Fatalf("Encode: first id = %v, want <|im_start|> (no BOS)", ids[:min(3, len(ids))])
	}
	if got := tk.DecodeRaw(ids); got != prompt {
		t.Errorf("DecodeRaw round-trip:\n got %q\nwant %q", got, prompt)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
