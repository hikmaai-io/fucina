package tokenizer

// GPT-2 / Qwen2 byte-level BPE support: the byte↔unicode alphabet and the
// pretokenizer splitter used by Qwen3 GGUFs (tokenizer.ggml.model == "gpt2").
//
// Byte-level BPE never operates on raw bytes; it first maps each of the 256
// possible bytes to a *printable* Unicode code point so the merge alphabet is a
// clean set of single-character strings. The mapping (identical to OpenAI's GPT-2
// `bytes_to_unicode`) keeps the printable ASCII/Latin-1 ranges as themselves and
// shifts every other byte up into the U+0100.. private range, in byte order:
//
//	[0x21..0x7E] ∪ [0xA1..0xAC] ∪ [0xAE..0xFF]  →  identity
//	all remaining bytes (0x00..0x20, 0x7F..0xA0, 0xAD), in ascending byte order,
//	                                              →  U+0100 + n  (n = 0,1,2,...)
//
// In particular the ASCII space 0x20 is the 33rd remapped byte (n=32) and so maps
// to U+0120 ("Ġ"), the leading-space marker Qwen3 token strings carry.

import "unicode"

var (
	// byteToRuneTable[b] is the GPT-2 code point for raw byte b.
	byteToRuneTable [256]rune
	// runeToByteTable inverts byteToRuneTable.
	runeToByteTable map[rune]byte
)

func init() {
	// Printable identity set.
	printable := func(b int) bool {
		return (b >= 0x21 && b <= 0x7E) ||
			(b >= 0xA1 && b <= 0xAC) ||
			(b >= 0xAE && b <= 0xFF)
	}
	for b := 0; b < 256; b++ {
		if printable(b) {
			byteToRuneTable[b] = rune(b)
		}
	}
	// Remaining bytes → U+0100 + n in ascending byte order.
	n := 0
	for b := 0; b < 256; b++ {
		if !printable(b) {
			byteToRuneTable[b] = rune(0x100 + n)
			n++
		}
	}
	runeToByteTable = make(map[rune]byte, 256)
	for b := 0; b < 256; b++ {
		runeToByteTable[byteToRuneTable[b]] = byte(b)
	}
}

// byteToRune maps a raw byte to its GPT-2 byte-level code point.
func byteToRune(b byte) rune { return byteToRuneTable[b] }

// runeToByte inverts byteToRune. ok is false for runes outside the byte alphabet.
func runeToByte(r rune) (byte, bool) {
	b, ok := runeToByteTable[r]
	return b, ok
}

// appendByteLevel decodes a byte-level token string s (a sequence of GPT-2
// code-point runes) back to its raw bytes and appends them to buf. Runes outside
// the byte alphabet are dropped (shouldn't occur for a well-formed vocab).
func appendByteLevel(buf []byte, s string) []byte {
	for _, r := range s {
		if b, ok := runeToByte(r); ok {
			buf = append(buf, b)
		}
	}
	return buf
}

// pretokenizeByteLevel splits text into pretokenizer pieces approximating the
// Qwen2 (GPT-2-derived) regex:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d) | ' ?\p{L}+ | \p{N} | ' ?[^\s\p{L}\p{N}]+ |
//	\s+(?!\S) | \s+
//
// Go's RE2 has no lookahead, so this is a hand-written scanner. Approximations
// vs. the upstream pattern, all documented and harmless for round-tripping:
//   - The optional leading character before letters/symbols is restricted to an
//     ASCII space ' ' (upstream allows any non-letter/non-digit, excluding CR/LF).
//   - Numbers are split one digit at a time (\p{N}), matching Qwen2's single-digit
//     behavior rather than GPT-2's ' ?\p{N}+ runs.
//   - \s+(?!\S) is emulated by: for a whitespace run longer than one that is
//     followed by a non-space character, the final space is left for the next
//     piece (so " word" / " !" keep their leading space); otherwise the whole run
//     is one piece.
//
// Byte-level BPE then byte-maps each piece, so even pieces that are themselves
// just whitespace round-trip exactly.
func pretokenizeByteLevel(text string) []string {
	runes := []rune(text)
	n := len(runes)
	var pieces []string
	isOther := func(r rune) bool {
		return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
	}

	i := 0
	for i < n {
		// 1. Contractions: 's 't 're 've 'm 'll 'd (case-insensitive).
		if runes[i] == '\'' && i+1 < n {
			if i+2 < n {
				two := string([]rune{unicode.ToLower(runes[i+1]), unicode.ToLower(runes[i+2])})
				if two == "re" || two == "ve" || two == "ll" {
					pieces = append(pieces, string(runes[i:i+3]))
					i += 3
					continue
				}
			}
			one := unicode.ToLower(runes[i+1])
			if one == 's' || one == 't' || one == 'm' || one == 'd' {
				pieces = append(pieces, string(runes[i:i+2]))
				i += 2
				continue
			}
		}

		// 2. ' ?\p{L}+ : optional single leading space, then a run of letters.
		{
			j := i
			if runes[j] == ' ' && j+1 < n && unicode.IsLetter(runes[j+1]) {
				j++
			}
			if j < n && unicode.IsLetter(runes[j]) {
				k := j
				for k < n && unicode.IsLetter(runes[k]) {
					k++
				}
				pieces = append(pieces, string(runes[i:k]))
				i = k
				continue
			}
		}

		// 3. \p{N} : a single digit.
		if unicode.IsNumber(runes[i]) {
			pieces = append(pieces, string(runes[i:i+1]))
			i++
			continue
		}

		// 4. ' ?[^\s\p{L}\p{N}]+ : optional leading space, then a run of "other".
		{
			j := i
			if runes[j] == ' ' && j+1 < n && isOther(runes[j+1]) {
				j++
			}
			if j < n && isOther(runes[j]) {
				k := j
				for k < n && isOther(runes[k]) {
					k++
				}
				pieces = append(pieces, string(runes[i:k]))
				i = k
				continue
			}
		}

		// 5. Whitespace runs: \s+(?!\S) | \s+.
		if unicode.IsSpace(runes[i]) {
			k := i
			for k < n && unicode.IsSpace(runes[k]) {
				k++
			}
			// If the run is followed by a non-space char, leave the last space
			// for the following word/symbol piece (only when the run > 1 char).
			if k < n && (k-i) > 1 {
				k--
			}
			pieces = append(pieces, string(runes[i:k]))
			i = k
			continue
		}

		// Fallback: emit a lone rune (should be unreachable).
		pieces = append(pieces, string(runes[i:i+1]))
		i++
	}
	return pieces
}
