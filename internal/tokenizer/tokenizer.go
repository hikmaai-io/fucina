// Package tokenizer provides Gemma 4 12B tokenization using SentencePiece / Unigram.
//
// Gemma 4 uses a SentencePiece tokenizer with vocab_size=262144.
// Special tokens:
//   BOS = 2,  EOS = 1,  PAD = 0
//   BOI (image start) = 255999
//   BOA (audio start) = 256000
//
// The tokenizer is loaded from the GGUF file's tokenizer section.

package tokenizer

import (
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode/utf8"
)

// Tokenizer handles Gemma 4 tokenization using SentencePiece unigram model.
type Tokenizer struct {
	vocab       []string
	vocabScores []float32
	vocabSize   int

	// Byte-fallback: for unknown bytes, each byte becomes \x<hex>
	byteFallback bool

	// Special token IDs
	BOS int32
	EOS int32
	PAD int32
}

// Score represents a token candidate during decoding.
type Score struct {
	id    int32
	score float32
	str   string
}

// ─── GGUF value type tags (v3 spec) ───────────────────────────────

const (
	ggufTypeUint8   = 0
	ggufTypeInt8    = 1
	ggufTypeUint16  = 2
	ggufTypeInt16   = 3
	ggufTypeUint32  = 4
	ggufTypeInt32   = 5
	ggufTypeFloat32 = 6
	ggufTypeBool    = 7
	ggufTypeString  = 8
	ggufTypeArray   = 9
	ggufTypeUint64  = 10
	ggufTypeInt64   = 11
	ggufTypeFloat64 = 12
)

// ggufScalarSize returns the byte width of a fixed-size GGUF scalar type,
// or 0 for variable-length types (string / array).
func ggufScalarSize(t uint32) int64 {
	switch t {
	case ggufTypeUint8, ggufTypeInt8, ggufTypeBool:
		return 1
	case ggufTypeUint16, ggufTypeInt16:
		return 2
	case ggufTypeUint32, ggufTypeInt32, ggufTypeFloat32:
		return 4
	case ggufTypeUint64, ggufTypeInt64, ggufTypeFloat64:
		return 8
	default:
		return 0
	}
}

// ggufReader is a bounds-checked cursor over the mmap'd GGUF bytes.
type ggufReader struct {
	data []byte
	size int64
	pos  int64
}

func (r *ggufReader) readU32() (uint32, bool) {
	if r.pos+4 > r.size {
		return 0, false
	}
	v := binary.LittleEndian.Uint32(r.data[r.pos : r.pos+4])
	r.pos += 4
	return v, true
}

func (r *ggufReader) readU64() (uint64, bool) {
	if r.pos+8 > r.size {
		return 0, false
	}
	v := binary.LittleEndian.Uint64(r.data[r.pos : r.pos+8])
	r.pos += 8
	return v, true
}

// readString reads a GGUF string: uint64 length + raw bytes (no NUL, no pad).
func (r *ggufReader) readString() (string, bool) {
	n, ok := r.readU64()
	if !ok {
		return "", false
	}
	if r.pos+int64(n) > r.size {
		return "", false
	}
	s := string(r.data[r.pos : r.pos+int64(n)])
	r.pos += int64(n)
	return s, true
}

// readArrayHeader reads the element type (u32) and count (u64) of an array.
func (r *ggufReader) readArrayHeader() (uint32, uint64, bool) {
	at, ok := r.readU32()
	if !ok {
		return 0, 0, false
	}
	n, ok := r.readU64()
	if !ok {
		return 0, 0, false
	}
	return at, n, true
}

// readScalarInt reads a u32/i32 (and friends) scalar as int32, advancing pos.
func (r *ggufReader) readScalarInt(valType uint32) (int32, bool) {
	switch valType {
	case ggufTypeUint32, ggufTypeInt32:
		v, ok := r.readU32()
		return int32(v), ok
	case ggufTypeUint16, ggufTypeInt16:
		if r.pos+2 > r.size {
			return 0, false
		}
		v := binary.LittleEndian.Uint16(r.data[r.pos : r.pos+2])
		r.pos += 2
		return int32(int16(v)), true
	case ggufTypeUint8, ggufTypeInt8:
		if r.pos+1 > r.size {
			return 0, false
		}
		v := r.data[r.pos]
		r.pos++
		return int32(int8(v)), true
	default:
		// Not an integer type we handle; skip it to stay in sync.
		r.skipValue(valType)
		return 0, false
	}
}

// skipValue advances past a value of the given type. Returns false on overflow.
func (r *ggufReader) skipValue(valType uint32) bool {
	switch valType {
	case ggufTypeString:
		_, ok := r.readString()
		return ok
	case ggufTypeArray:
		at, n, ok := r.readArrayHeader()
		if !ok {
			return false
		}
		if at == ggufTypeString {
			for i := uint64(0); i < n; i++ {
				if _, ok := r.readString(); !ok {
					return false
				}
			}
			return true
		}
		sz := ggufScalarSize(at)
		if sz == 0 {
			return false // nested arrays are not valid GGUF
		}
		r.pos += sz * int64(n)
		return r.pos <= r.size
	default:
		sz := ggufScalarSize(valType)
		if sz == 0 {
			return false
		}
		r.pos += sz
		return r.pos <= r.size
	}
}

// New creates a tokenizer from GGUF tokenizer data.
//
// It parses the real GGUF v3 binary layout (no null-terminated keys, no
// 32-byte key padding, no per-entry alignment). Strings are stored as
// uint64 length + raw bytes. KV pairs are packed contiguously:
//
//	header = magic(4) version(4) tensor_count(8) metadata_kv_count(8)
//	kv     = key(string) + type(u32) + value
//
// We extract:
//	tokenizer.ggml.tokens        (string array) -> vocab
//	tokenizer.ggml.scores        (float array)  -> scores
//	tokenizer.ggml.bos_token_id  (u32/i32)
//	tokenizer.ggml.eos_token_id  (u32/i32)
//	tokenizer.ggml.padding_token_id (u32/i32)
func New(ggufData []byte, ggufSize int64) (*Tokenizer, error) {
	t := &Tokenizer{
		BOS: 2,
		EOS: 1,
		PAD: 0,
	}

	p := &ggufReader{data: ggufData, size: ggufSize, pos: 0}

	if p.size < 24 {
		return nil, fmt.Errorf("tokenizer: file too small for GGUF header")
	}
	magic := binary.LittleEndian.Uint32(ggufData[0:4])
	if magic != 0x46554747 { // "GGUF"
		return nil, fmt.Errorf("tokenizer: bad GGUF magic 0x%08x", magic)
	}
	kvCount := binary.LittleEndian.Uint64(ggufData[16:24])
	p.pos = 24

	tokensFound := false
	for i := uint64(0); i < kvCount; i++ {
		key, ok := p.readString()
		if !ok {
			return nil, fmt.Errorf("tokenizer: truncated metadata key at kv %d", i)
		}
		valType, ok := p.readU32()
		if !ok {
			return nil, fmt.Errorf("tokenizer: truncated value type for %q", key)
		}

		switch key {
		case "tokenizer.ggml.tokens":
			if valType != ggufTypeArray {
				return nil, fmt.Errorf("tokenizer: tokens is not an array")
			}
			arrType, n, ok := p.readArrayHeader()
			if !ok || arrType != ggufTypeString {
				return nil, fmt.Errorf("tokenizer: tokens array malformed")
			}
			t.vocab = make([]string, n)
			for j := uint64(0); j < n; j++ {
				s, ok := p.readString()
				if !ok {
					return nil, fmt.Errorf("tokenizer: truncated token %d", j)
				}
				t.vocab[j] = s
			}
			tokensFound = true

		case "tokenizer.ggml.scores":
			if valType != ggufTypeArray {
				return nil, fmt.Errorf("tokenizer: scores is not an array")
			}
			arrType, n, ok := p.readArrayHeader()
			if !ok || arrType != ggufTypeFloat32 {
				return nil, fmt.Errorf("tokenizer: scores array malformed")
			}
			t.vocabScores = make([]float32, n)
			for j := uint64(0); j < n; j++ {
				bits, ok := p.readU32()
				if !ok {
					return nil, fmt.Errorf("tokenizer: truncated score %d", j)
				}
				t.vocabScores[j] = math.Float32frombits(bits)
			}

		case "tokenizer.ggml.bos_token_id":
			if v, ok := p.readScalarInt(valType); ok {
				t.BOS = v
			}
		case "tokenizer.ggml.eos_token_id":
			if v, ok := p.readScalarInt(valType); ok {
				t.EOS = v
			}
		case "tokenizer.ggml.padding_token_id":
			if v, ok := p.readScalarInt(valType); ok {
				t.PAD = v
			}

		default:
			if !p.skipValue(valType) {
				return nil, fmt.Errorf("tokenizer: failed to skip value for %q", key)
			}
		}
	}

	if !tokensFound || len(t.vocab) == 0 {
		return nil, fmt.Errorf("tokenizer: no tokens found in GGUF")
	}

	t.vocabSize = len(t.vocab)
	if t.vocabScores == nil {
		t.vocabScores = make([]float32, t.vocabSize)
		for i := range t.vocabScores {
			t.vocabScores[i] = float32(-i) // descending order
		}
	}

	return t, nil
}

// Encode tokenizes a string into token IDs.
// Uses unigram/bpe tokenization: longest prefix match with scores.
func (t *Tokenizer) Encode(text string, addBos bool, addEos bool) []int32 {
	// For SentencePiece, we do unigram tokenization with longest match.
	// This is simplified; production should use the full BPE/Unigram algorithm.
	var tokens []int32

	if addBos {
		tokens = append(tokens, t.BOS)
	}

	// Simple byte-level tokenization using longest prefix match
	// In practice, this should use the full SentencePiece model
	// For now, we tokenize by splitting on whitespace and matching
	// the longest token in the vocab.

	// Build a trie for efficient matching
	// For simplicity, use a sorted vocab and binary search
	// This is NOT the full SentencePiece algorithm but handles common cases

	// First try: use pre-built trie for longest prefix match
	for len(text) > 0 {
		matched := false

		// Try longest match from full text length down to 1 byte
		maxLen := len(text)
		if maxLen > 32 { // Reasonable max token length
			maxLen = 32
		}

		for l := maxLen; l >= 1; l-- {
			prefix := text[:l]
			if id, ok := t.findToken(prefix); ok {
				tokens = append(tokens, id)
				text = text[l:]
				matched = true
				break
			}
		}

		if !matched {
			// Byte fallback: encode as individual bytes
			r, size := utf8.DecodeRuneInString(text)
			if r == utf8.RuneError && size == 1 {
				// Use byte token <0xXX>
				byteToken := fmt.Sprintf("<0x%02X>", text[0])
				if id, ok := t.findToken(byteToken); ok {
					tokens = append(tokens, id)
				} else {
					tokens = append(tokens, t.PAD)
				}
				text = text[1:]
			} else {
				text = text[size:]
			}
		}
	}

	if addEos {
		tokens = append(tokens, t.EOS)
	}

	return tokens
}

// Decode converts token IDs back to a string.
func (t *Tokenizer) Decode(tokens []int32) string {
	var sb strings.Builder
	for _, id := range tokens {
		if id >= 0 && int(id) < len(t.vocab) {
			s := t.vocab[id]
			// Skip special tokens
			if s == "" || id == t.BOS || id == t.EOS || id == t.PAD {
				continue
			}
			// Remove the Ġ space marker (SentencePiece)
			s = strings.ReplaceAll(s, "Ġ", " ")
			sb.WriteString(s)
		}
	}
	return sb.String()
}

// findToken searches for a token string in the vocab using longest prefix.
func (t *Tokenizer) findToken(s string) (int32, bool) {
	// Direct map lookup
	for i, v := range t.vocab {
		if v == s {
			return int32(i), true
		}
	}

	// Check with SentencePiece prefix (Ġ for space)
	if len(s) > 0 && s[0] != ' ' {
		for i, v := range t.vocab {
			if v == "Ġ"+s {
				return int32(i), true
			}
		}
	}

	return 0, false
}

// NumTokens returns the vocabulary size.
func (t *Tokenizer) NumTokens() int {
	return t.vocabSize
}

// TokenStr returns the string representation of a token ID.
func (t *Tokenizer) TokenStr(id int32) string {
	if id >= 0 && int(id) < len(t.vocab) {
		return t.vocab[id]
	}
	return ""
}

// Generate the tokenizer trie for efficient encoding.
// This is a simplified version; production should use
// the full SentencePiece model from the GGUF tokenizer.ggml.model data.
func (t *Tokenizer) buildTrie() {
	// Build a sorted list of token strings for binary search
	// For production: build actual trie data structure
	sort.Slice(t.vocab, func(i, j int) bool {
		// Sort by length descending, then alphabetically
		if len(t.vocab[i]) != len(t.vocab[j]) {
			return len(t.vocab[i]) > len(t.vocab[j])
		}
		return t.vocab[i] < t.vocab[j]
	})
}

// GetVocab returns the vocabulary slice.
func (t *Tokenizer) GetVocab() []string {
	return t.vocab
}
