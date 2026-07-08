package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// buildChatMLJSON writes a minimal HF tokenizer.json with a ChatML added-token
// set (the Qwen3.5 shape: ByteLevel BPE, no <bos>/<eos>, ChatML specials).
func buildChatMLJSON(t *testing.T) string {
	t.Helper()
	base := map[string]int32{}
	// A tiny byte-level alphabet so encode has something to chew on.
	for i, s := range []string{"h", "i", "Ġ", "e", "l", "o", "hi", "Ġhi"} {
		base[s] = int32(i)
	}
	added := []map[string]interface{}{}
	for i, s := range []string{
		"<|endoftext|>", "<|im_start|>", "<|im_end|>",
		"<think>", "</think>", "<tool_call>", "</tool_call>",
		"<tool_response>", "</tool_response>",
	} {
		added = append(added, map[string]interface{}{
			"id": 100 + i, "content": s, "special": true,
		})
	}
	doc := map[string]interface{}{
		"added_tokens":  added,
		"pre_tokenizer": map[string]interface{}{"type": "ByteLevel"},
		"model": map[string]interface{}{
			"type":   "BPE",
			"vocab":  base,
			"merges": []string{"h i"},
		},
	}
	data, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestHFJSON_ChatMLMapping(t *testing.T) {
	tok, err := NewFromHFJSON(buildChatMLJSON(t))
	if err != nil {
		t.Fatal(err)
	}
	if tok.EOS != 102 { // <|im_end|>
		t.Errorf("EOS = %d, want 102 (<|im_end|>)", tok.EOS)
	}
	if tok.EOS2 != 100 { // <|endoftext|>
		t.Errorf("EOS2 = %d, want 100 (<|endoftext|>)", tok.EOS2)
	}
	if tok.BOS != -1 || tok.addBOS {
		t.Errorf("ChatML must not add BOS (BOS=%d addBOS=%v)", tok.BOS, tok.addBOS)
	}
	if tok.StartOfTurn != 101 || tok.EndOfTurn != 102 {
		t.Errorf("im_start/im_end = %d/%d", tok.StartOfTurn, tok.EndOfTurn)
	}
	if tok.ChannelOpen != 103 || tok.ChannelEnd != 104 {
		t.Errorf("think markers = %d/%d", tok.ChannelOpen, tok.ChannelEnd)
	}
	if tok.ToolCallOpen != 105 || tok.ToolCallEnd != 106 {
		t.Errorf("tool_call markers = %d/%d", tok.ToolCallOpen, tok.ToolCallEnd)
	}
	if tok.ToolRespOpen != 107 || tok.ToolRespEnd != 108 {
		t.Errorf("tool_response markers = %d/%d", tok.ToolRespOpen, tok.ToolRespEnd)
	}
	if !tok.IsStop(102) || !tok.IsStop(100) {
		t.Error("IsStop must fire on <|im_end|> and <|endoftext|>")
	}
	if tok.IsStop(1) || tok.IsStop(106) {
		t.Error("IsStop fires on non-stop ids (the old Gemma-default bug)")
	}
	stops := tok.StopIDs()
	if len(stops) != 2 {
		t.Errorf("StopIDs = %v, want [102 100]", stops)
	}
	if !tok.HasToken("<|im_start|>") || tok.HasToken("<|turn>") {
		t.Error("HasToken misreports the vocab")
	}
	if !tok.IsToolMarker(105) || !tok.IsToolMarker(108) {
		t.Error("IsToolMarker must cover the ChatML tool markers")
	}

	// The ChatML markers must encode as single tokens (pre-split scan).
	ids := tok.Encode("<|im_start|>hi<|im_end|>", true, false)
	want := []int32{101, 6, 102}
	if len(ids) != len(want) {
		t.Fatalf("Encode ids = %v, want %v", ids, want)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("Encode ids = %v, want %v", ids, want)
		}
	}
}
