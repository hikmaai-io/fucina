package tokenizer

// HuggingFace tokenizer.json (BPE) loader — the tokenizer the NVFP4 safetensors Gemma-4
// checkpoints ship instead of a GGUF vocab. Gemma's HF tokenizer is byte-fallback BPE over a
// metaspace alphabet: the normalizer rewrites every space to "▁" (U+2581), then BPE merges
// adjacent symbols by merge RANK (lowest rank first). This disagrees with the GGUF path's
// unigram longest-prefix match, so the model's own tokenizer.json is the only faithful encoder
// for these checkpoints — loading the GGUF vocab and greedy-matching mis-segments and degrades
// output. Decode (id→string, ▁→space, <0xXX> byte reassembly) is algorithm-agnostic and shared
// with the GGUF path. Verified token-for-token against the HF `tokenizers` library in tests.

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"os"
	"sort"
)

// hfTokenizer is the subset of tokenizer.json we consume (BPE model + added tokens).
type hfTokenizer struct {
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
	Model struct {
		Type         string           `json:"type"`
		Vocab        map[string]int32 `json:"vocab"`
		Merges       []hfMerge        `json:"merges"` // ordered [left, right] pairs
		ByteFallback bool             `json:"byte_fallback"`
		UnkToken     string           `json:"unk_token"`
	} `json:"model"`
}

// hfMerge accepts BOTH on-disk merge encodings tokenizers emits: the legacy space-joined
// string "le ft" and the newer ["le","ft"] pair (Qwen3.5 checkpoints use the former, Gemma
// NVFP4 exports the latter). Always normalized to the [left, right] pair.
type hfMerge [2]string

func (m *hfMerge) UnmarshalJSON(b []byte) error {
	if len(b) > 0 && b[0] == '[' {
		var pair []string
		if err := json.Unmarshal(b, &pair); err != nil {
			return err
		}
		if len(pair) != 2 {
			return fmt.Errorf("merge pair has %d elements", len(pair))
		}
		m[0], m[1] = pair[0], pair[1]
		return nil
	}
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	sp := strings.IndexByte(s, ' ')
	if sp < 0 {
		return fmt.Errorf("merge %q has no space separator", s)
	}
	m[0], m[1] = s[:sp], s[sp+1:]
	return nil
}

// NewFromHFJSON builds a BPE tokenizer from a HuggingFace tokenizer.json file (the format the
// NVFP4 checkpoints carry). Defaults match Gemma-4; control-token ids are resolved by string so
// they track the export.
func NewFromHFJSON(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: read %s: %w", path, err)
	}
	var hf hfTokenizer
	if err := json.Unmarshal(data, &hf); err != nil {
		return nil, fmt.Errorf("tokenizer: parse %s: %w", path, err)
	}
	if hf.Model.Type != "BPE" {
		return nil, fmt.Errorf("tokenizer: %s is %q, only BPE tokenizer.json is supported", path, hf.Model.Type)
	}
	if len(hf.Model.Vocab) == 0 {
		return nil, fmt.Errorf("tokenizer: %s has empty vocab", path)
	}

	t := &Tokenizer{BOS: 2, EOS: 1, PAD: 0, bpe: true, byteFallback: hf.Model.ByteFallback,
		addBOS: true, addEOS: true}

	// vocab[id] = token string; ids are dense 0..N-1.
	maxID := int32(-1)
	for _, id := range hf.Model.Vocab {
		if id > maxID {
			maxID = id
		}
	}
	t.vocabSize = int(maxID) + 1
	t.vocab = make([]string, t.vocabSize)
	t.vocabScores = make([]float32, t.vocabSize) // unused in BPE mode; sized for API parity
	t.tokenToID = make(map[string]int32, len(hf.Model.Vocab))
	for tok, id := range hf.Model.Vocab {
		t.vocab[id] = tok
		t.tokenToID[tok] = id
	}
	// Added tokens (specials) are matched verbatim and never BPE-merged; make sure their literal
	// strings resolve, even if absent from model.vocab.
	for _, a := range hf.AddedTokens {
		if int(a.ID) < len(t.vocab) {
			t.vocab[a.ID] = a.Content
		}
		t.tokenToID[a.Content] = a.ID
	}

	// merges → rank table (index in the list = priority; lower merges first).
	t.bpeMerges = make(map[mergePair]int32, len(hf.Model.Merges))
	for rank, m := range hf.Model.Merges {
		// First rank wins if a pair somehow repeats (it shouldn't).
		if _, dup := t.bpeMerges[mergePair{m[0], m[1]}]; !dup {
			t.bpeMerges[mergePair{m[0], m[1]}] = int32(rank)
		}
	}

	// Control-token ids by string (Gemma-4; same lookups as the GGUF path).
	lookup := func(s string, def int32) int32 {
		if id, ok := t.tokenToID[s]; ok {
			return id
		}
		return def
	}
	t.BOS = lookup("<bos>", 2)
	t.EOS = lookup("<eos>", 1)
	t.PAD = lookup("<pad>", 0)
	t.StartOfTurn = lookup("<|turn>", 105)
	t.EndOfTurn = lookup("<turn|>", 106)
	t.ChannelOpen = lookup("<|channel>", 100)
	t.ChannelEnd = lookup("<channel|>", 101)
	t.ToolOpen = lookup("<|tool>", -1)
	t.ToolEnd = lookup("<tool|>", -1)
	t.ToolCallOpen = lookup("<|tool_call>", -1)
	t.ToolCallEnd = lookup("<tool_call|>", -1)
	t.ToolRespOpen = lookup("<|tool_response>", -1)
	t.ToolRespEnd = lookup("<tool_response|>", -1)
	t.StringDelim = lookup(`<|"|>`, -1)

	// Encode pre-split set: every added token (HF always matches these before the model) plus the
	// GGUF-style control markers if present, deduped, longest-first (so no marker shadows a longer
	// one sharing its prefix). This keeps fucina's rendered turn/tool markers single-token.
	seen := make(map[string]bool)
	addSpecial := func(s string) {
		if s == "" || seen[s] {
			return
		}
		if id, ok := t.tokenToID[s]; ok {
			t.specials = append(t.specials, specialToken{str: s, id: id})
			seen[s] = true
		}
	}
	for _, a := range hf.AddedTokens {
		addSpecial(a.Content)
	}
	for _, s := range []string{
		"<|turn>", "<turn|>", "<|channel>", "<channel|>",
		"<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
		"<|tool_response>", "<tool_response|>", `<|"|>`, "<|think|>",
	} {
		addSpecial(s)
	}
	sort.Slice(t.specials, func(i, j int) bool {
		return len(t.specials[i].str) > len(t.specials[j].str)
	})

	return t, nil
}

// bpeEncode encodes one control-marker-free, ▁-normalized segment by greedy rank-ordered BPE:
// start from the segment's Unicode characters and repeatedly merge the adjacent pair with the
// lowest merge rank (leftmost on ties), then map each final symbol to its id — falling back to
// <0xXX> byte tokens for any symbol the vocab lacks. Naive O(L²) over the segment, which is the
// whole text since Gemma's pre-tokenizer Split-on-space is a no-op after ▁ normalization; fine
// for prompt-length inputs.
func (t *Tokenizer) bpeEncode(text string, tokens []int32) []int32 {
	if text == "" {
		return tokens
	}
	// initial symbols: one per Unicode rune
	syms := make([]string, 0, len(text))
	for _, r := range text {
		syms = append(syms, string(r))
	}
	// merge by lowest rank until none applies
	for len(syms) > 1 {
		bestRank := int32(math.MaxInt32)
		bestI := -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.bpeMerges[mergePair{syms[i], syms[i+1]}]; ok && r < bestRank {
				bestRank = r
				bestI = i
			}
		}
		if bestI < 0 {
			break
		}
		syms[bestI] += syms[bestI+1]
		syms = append(syms[:bestI+1], syms[bestI+2:]...)
	}
	// symbols → ids (byte fallback for out-of-vocab symbols)
	for _, s := range syms {
		if id, ok := t.tokenToID[s]; ok {
			tokens = append(tokens, id)
			continue
		}
		if t.byteFallback {
			for i := 0; i < len(s); i++ {
				if id, ok := t.tokenToID[fmt.Sprintf("<0x%02X>", s[i])]; ok {
					tokens = append(tokens, id)
				} else if id, ok := t.tokenToID["<unk>"]; ok {
					tokens = append(tokens, id)
				} else {
					tokens = append(tokens, t.PAD)
				}
			}
		} else if id, ok := t.tokenToID["<unk>"]; ok {
			tokens = append(tokens, id)
		} else {
			tokens = append(tokens, t.PAD)
		}
	}
	return tokens
}
