package tokenizer

// GPT-2 / Qwen2 byte-level BPE for GGUF checkpoints whose tokenizer.ggml.model == "gpt2"
// (Qwen3, Qwen2, and other GPT-2-lineage models). This is a DIFFERENT algorithm from
// Gemma's SentencePiece path: the vocab and merges are expressed over a byte→unicode
// "alphabet" (every raw byte maps to a printable rune; the space byte 0x20 → "Ġ"), the
// text is first split by the GPT-2/Qwen2 regex pre-tokenizer, and BPE then merges symbols
// by merge RANK within each pre-token. Loading such a vocab through the SentencePiece
// greedy/▁ path mis-segments every space into <unk> — which is exactly the garbage the
// Qwen3 HTTP path produced before this. Decode is the inverse byte map.
//
// Verified against llama.cpp's Qwen3 Q4_K_M reference: "The capital of France is" →
// [785, 6722, 315, 9625, 374] (the same ids the C parity test feeds).

import (
	"strings"
	"unicode"
)

// setupGPT2 switches the tokenizer to byte-level BPE mode: builds the byte↔unicode
// alphabet maps, loads the merge ranks, and registers control/user-defined tokens as
// pre-split specials. Called from New() after the vocab is loaded.
func (t *Tokenizer) setupGPT2(mergesRaw []string, tokenTypes []int32) {
	t.gpt2 = true
	t.buildByteMaps()

	// merges "<left> <right>" → rank table (list index = priority, lower first).
	t.bpeMerges = make(map[mergePair]int32, len(mergesRaw))
	for rank, m := range mergesRaw {
		// The separator is a literal ASCII space; the byte-level space is "Ġ", so a
		// SplitN on " " is unambiguous. Skip anything that does not split in two.
		parts := strings.SplitN(m, " ", 2)
		if len(parts) != 2 {
			continue
		}
		pair := mergePair{parts[0], parts[1]}
		if _, dup := t.bpeMerges[pair]; !dup {
			t.bpeMerges[pair] = int32(rank)
		}
	}

	// Register control (3) and user-defined (4) tokens as specials so a prompt that
	// contains e.g. "<|im_start|>" tokenizes it as one id instead of byte-splitting it.
	// (token_type follows llama.cpp: 1 NORMAL, 2 UNKNOWN, 3 CONTROL, 4 USER_DEFINED,
	// 6 BYTE.)
	for id, ty := range tokenTypes {
		if ty != 3 && ty != 4 {
			continue
		}
		if id >= len(t.vocab) {
			break
		}
		s := t.vocab[id]
		if s == "" {
			continue
		}
		t.specials = append(t.specials, specialToken{str: s, id: int32(id)})
	}
}

// buildByteMaps constructs the GPT-2 bytes_to_unicode bijection: printable byte ranges
// map to themselves, every other byte to a private-use rune (256+n), so all 256 bytes
// are representable as single vocab symbols.
func (t *Tokenizer) buildByteMaps() {
	inRange := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}
	t.byteDecoder = make(map[rune]byte, 256)
	n := 0
	for b := 0; b < 256; b++ {
		var r rune
		if inRange(b) {
			r = rune(b)
		} else {
			r = rune(256 + n)
			n++
		}
		t.byteEncoder[b] = r
		t.byteDecoder[r] = byte(b)
	}
}

// gpt2Encode pre-tokenizes a (special-marker-free) text segment with the GPT-2/Qwen2
// rules, byte-encodes each pre-token, then BPE-merges it by rank.
func (t *Tokenizer) gpt2Encode(text string, tokens []int32) []int32 {
	for _, piece := range pretokenizeGPT2(text) {
		tokens = t.bpeEncode(t.byteEncode(piece), tokens)
	}
	return tokens
}

// byteEncode maps a pre-token's raw UTF-8 bytes through the byte→unicode alphabet,
// yielding the symbol string BPE operates on (space → "Ġ", etc.).
func (t *Tokenizer) byteEncode(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		b.WriteRune(t.byteEncoder[s[i]])
	}
	return b.String()
}

// gpt2Decode maps a byte-level vocab string back to its raw bytes.
func (t *Tokenizer) gpt2Decode(s string, buf []byte) []byte {
	for _, r := range s {
		if b, ok := t.byteDecoder[r]; ok {
			buf = append(buf, b)
		} else {
			// Not an alphabet rune (e.g. a special token's literal ASCII content,
			// which round-trips as-is) — append its UTF-8 bytes.
			buf = append(buf, string(r)...)
		}
	}
	return buf
}

// pretokenizeGPT2 splits text into pre-tokens following the GPT-2/Qwen2 regex
//
//	(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// hand-written because Go's RE2 has no lookahead for the `\s+(?!\S)` clause. The
// ordered alternation is replicated: at each position the first matching clause
// consumes. A leading space therefore attaches to the following word (" capital"),
// which is what makes the byte-level vocab's "Ġword" tokens single ids.
func pretokenizeGPT2(text string) []string {
	r := []rune(text)
	n := len(r)
	var out []string
	i := 0
	isL := func(c rune) bool { return unicode.IsLetter(c) }
	isN := func(c rune) bool { return unicode.IsNumber(c) }
	isWS := func(c rune) bool { return unicode.IsSpace(c) }
	for i < n {
		c := r[i]
		// 1. contractions: '(s|t|re|ve|m|ll|d), case-insensitive.
		if c == '\'' && i+1 < n {
			c1 := unicode.ToLower(r[i+1])
			if i+2 < n {
				c2 := unicode.ToLower(r[i+2])
				if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
					out = append(out, string(r[i:i+3]))
					i += 3
					continue
				}
			}
			if c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd' {
				out = append(out, string(r[i:i+2]))
				i += 2
				continue
			}
		}
		// 2. [^\r\n\p{L}\p{N}]? \p{L}+  — letters with one optional non-letter/digit
		//    prefix (typically the leading space, captured as "Ġ" after byte-encode).
		{
			j := i
			if c != '\r' && c != '\n' && !isL(c) && !isN(c) && j+1 < n && isL(r[j+1]) {
				j++ // take the optional prefix; a letter follows
			}
			if j < n && isL(r[j]) {
				k := j
				for k < n && isL(r[k]) {
					k++
				}
				out = append(out, string(r[i:k]))
				i = k
				continue
			}
		}
		// 3. \p{N}  — a single digit/number rune.
		if isN(c) {
			out = append(out, string(r[i:i+1]))
			i++
			continue
		}
		// 4. ' ?[^\s\p{L}\p{N}]+[\r\n]*  — punctuation run with optional leading space.
		{
			j := i
			if c == ' ' && j+1 < n && !isWS(r[j+1]) && !isL(r[j+1]) && !isN(r[j+1]) {
				j++
			}
			if j < n && !isWS(r[j]) && !isL(r[j]) && !isN(r[j]) {
				k := j
				for k < n && !isWS(r[k]) && !isL(r[k]) && !isN(r[k]) {
					k++
				}
				for k < n && (r[k] == '\r' || r[k] == '\n') {
					k++
				}
				out = append(out, string(r[i:k]))
				i = k
				continue
			}
		}
		// 5/6/7. whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+ .
		//
		// The newline clause `\s*[\r\n]+` has PRIORITY and is greedy: a run that
		// contains newlines is consumed through its LAST newline in one piece —
		// newlines never become a following word's prefix (the word clause
		// excludes \r\n from its optional prefix). Getting this wrong split
		// "\n\nWord" into ["\n","\n","Word"] (198,198) instead of ["\n\n","Word"]
		// (271 "ĊĊ"), diverging from HF/llama.cpp on every multi-line prompt and
		// breaking token-exact re-render of ChatML turns (prefix-cache misses).
		//
		// A pure space/tab run followed by a non-space leaves its LAST char for
		// the next clause (the `(?!\S)` lookahead), so it becomes the following
		// word's "Ġ" prefix.
		if isWS(c) {
			k := i
			for k < n && isWS(r[k]) {
				k++
			}
			lastNL := -1
			for j := k - 1; j >= i; j-- {
				if r[j] == '\n' || r[j] == '\r' {
					lastNL = j
					break
				}
			}
			if lastNL >= 0 {
				out = append(out, string(r[i:lastNL+1]))
				i = lastNL + 1
				continue
			}
			if k < n && k-1 > i {
				k-- // leave the last whitespace for the following word/punct
			}
			out = append(out, string(r[i:k]))
			i = k
			continue
		}
		// Fallback (should be unreachable): emit one rune.
		out = append(out, string(r[i:i+1]))
		i++
	}
	return out
}
