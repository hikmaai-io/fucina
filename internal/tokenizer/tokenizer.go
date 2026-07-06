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
)

// spaceMarker is the SentencePiece "▁" (U+2581) used by Gemma to encode a
// leading space on a token. It is NOT the GPT-2 BPE "Ġ".
const spaceMarker = "▁"

// Tokenizer handles Gemma 4 tokenization using SentencePiece unigram model.
type Tokenizer struct {
	vocab       []string
	vocabScores []float32
	vocabSize   int

	// tokenToID maps a token string to its id for O(1) Encode lookups
	// (replaces the previous O(vocab) linear scan).
	tokenToID map[string]int32

	// Special token IDs (read from the GGUF vocab by string when present).
	BOS int32
	EOS int32
	PAD int32

	// ChatTemplate is the model's embedded Jinja chat template (GGUF
	// tokenizer.chat_template), empty if the GGUF omits it. Used by the --jinja
	// rendering path (internal/chat) to format messages exactly like llama.cpp.
	ChatTemplate string

	// Gemma-4 turn / channel control tokens. EndOfTurn (<turn|>) terminates an
	// assistant turn and must be treated as a stop token alongside EOS.
	StartOfTurn int32 // <|turn>   = 105
	EndOfTurn   int32 // <turn|>   = 106
	ChannelOpen int32 // <|channel> = 100
	ChannelEnd  int32 // <channel|> = 101

	// Gemma-4 tool-calling tokens (-1 if absent). See internal/server tool support.
	ToolOpen     int32 // <|tool>          (tool declaration open)
	ToolEnd      int32 // <tool|>
	ToolCallOpen int32 // <|tool_call>     (model emits a call)
	ToolCallEnd  int32 // <tool_call|>
	ToolRespOpen int32 // <|tool_response> (tool result back to the model)
	ToolRespEnd  int32 // <tool_response|>
	StringDelim  int32 // <|"|>            (string value delimiter in the dict syntax)

	// specials are the control-marker literals (above) that resolved to vocab
	// ids, longest-first. Encode splits the input at these literals BEFORE the
	// greedy longest-prefix matcher runs: matching from an arbitrary position
	// can otherwise merge a preceding character into the marker's leading '<'
	// (".<channel|>" → ".<" + "channel" + "|>") so the marker id is never
	// produced — re-encoded prompts then token-mismatch the generated sequence
	// and silently break KV prefix reuse.
	specials []specialToken

	// BPE mode (HF tokenizer.json, used by the NVFP4 safetensors checkpoints which
	// ship no GGUF vocab). When set, Encode merges adjacent symbols by merge RANK
	// instead of the unigram longest-prefix match — the two algorithms disagree, so
	// the model's own tokenizer.json is the only faithful encoder. Populated by
	// NewFromHFJSON; Decode is algorithm-agnostic and shared. See hf_bpe.go.
	bpe          bool
	byteFallback bool
	bpeMerges    map[mergePair]int32 // ordered merge → rank (lower rank applied first)

	// byteLevel selects the GPT-2 / Qwen2 byte-level BPE pre-processing used by
	// Qwen3 GGUFs (tokenizer.ggml.model == "gpt2"). When set, Encode runs the
	// qwen2-style pretokenizer, maps every input byte through the GPT-2
	// byte↔unicode table (bytelevel.go) before BPE merging, and skips the
	// SentencePiece U+2581 space normalization. Decode reverses the byte map.
	// Mutually exclusive with the SentencePiece path; byteFallback is forced off
	// (the byte alphabet already covers all 256 bytes as base vocab tokens).
	byteLevel bool

	// AddBOS mirrors tokenizer.ggml.add_bos_token (default true; Qwen3 sets it
	// false). Encode only prepends BOS when both the caller asks for it AND this
	// flag is set, so a checkpoint that must not start with BOS never gets one.
	AddBOS bool

	// pre is tokenizer.ggml.pre (the pretokenizer name, e.g. "qwen2"). Stored for
	// diagnostics; the byte-level splitter currently approximates the qwen2 regex.
	pre string
}

// mergePair keys the BPE merge table: the adjacent (left,right) symbol pair.
type mergePair struct{ a, b string }

// specialToken pairs a control-marker literal with its vocab id for Encode's
// pre-split scan.
type specialToken struct {
	str string
	id  int32
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

// readBool reads a single-byte GGUF bool (type 7): nonzero ⇒ true.
func (r *ggufReader) readBool() (bool, bool) {
	if r.pos+1 > r.size {
		return false, false
	}
	v := r.data[r.pos]
	r.pos++
	return v != 0, true
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
//
//	tokenizer.ggml.tokens        (string array) -> vocab
//	tokenizer.ggml.scores        (float array)  -> scores
//	tokenizer.ggml.bos_token_id  (u32/i32)
//	tokenizer.ggml.eos_token_id  (u32/i32)
//	tokenizer.ggml.padding_token_id (u32/i32)
func New(ggufData []byte, ggufSize int64) (*Tokenizer, error) {
	t := &Tokenizer{
		BOS:    2,
		EOS:    1,
		PAD:    0,
		AddBOS: true, // GGUF default; tokenizer.ggml.add_bos_token overrides (Qwen3=false).
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
		case "tokenizer.chat_template":
			if valType == ggufTypeString {
				if s, ok := p.readString(); ok {
					t.ChatTemplate = s
				}
			} else if !p.skipValue(valType) {
				return nil, fmt.Errorf("tokenizer: failed to read chat_template")
			}

		case "tokenizer.ggml.model":
			if valType == ggufTypeString {
				if s, ok := p.readString(); ok && s == "gpt2" {
					// GPT-2 / Qwen2 byte-level BPE (Qwen3 GGUFs). Merge-by-rank
					// over a byte-mapped alphabet; no SentencePiece, no <0xXX>
					// byte fallback (the 256 byte runes are all base vocab).
					t.bpe = true
					t.byteLevel = true
					t.byteFallback = false
				}
			} else if !p.skipValue(valType) {
				return nil, fmt.Errorf("tokenizer: failed to read tokenizer model")
			}

		case "tokenizer.ggml.pre":
			if valType == ggufTypeString {
				if s, ok := p.readString(); ok {
					t.pre = s
				}
			} else if !p.skipValue(valType) {
				return nil, fmt.Errorf("tokenizer: failed to read tokenizer pre")
			}

		case "tokenizer.ggml.merges":
			if valType != ggufTypeArray {
				return nil, fmt.Errorf("tokenizer: merges is not an array")
			}
			arrType, n, ok := p.readArrayHeader()
			if !ok || arrType != ggufTypeString {
				return nil, fmt.Errorf("tokenizer: merges array malformed")
			}
			t.bpeMerges = make(map[mergePair]int32, n)
			for j := uint64(0); j < n; j++ {
				s, ok := p.readString()
				if !ok {
					return nil, fmt.Errorf("tokenizer: truncated merge %d", j)
				}
				// "LEFT RIGHT" split on the FIRST space; list index = rank
				// (lower rank applied first). First occurrence wins on dupes.
				sp := strings.IndexByte(s, ' ')
				if sp < 0 {
					continue // malformed merge line; skip defensively
				}
				pair := mergePair{a: s[:sp], b: s[sp+1:]}
				if _, dup := t.bpeMerges[pair]; !dup {
					t.bpeMerges[pair] = int32(j)
				}
			}

		case "tokenizer.ggml.add_bos_token":
			if valType == ggufTypeBool {
				if v, ok := p.readBool(); ok {
					t.AddBOS = v
				}
			} else if !p.skipValue(valType) {
				return nil, fmt.Errorf("tokenizer: failed to read add_bos_token")
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

	// Build the string→id map once (id = vocab index).
	t.tokenToID = make(map[string]int32, t.vocabSize)
	for i, s := range t.vocab {
		// First occurrence wins (ids are unique but be defensive).
		if _, exists := t.tokenToID[s]; !exists {
			t.tokenToID[s] = int32(i)
		}
	}

	// Resolve control tokens by string so we never hard-code ids that drift
	// between exports. Defaults match the gemma-4 12B GGUF.
	lookup := func(s string, def int32) int32 {
		if id, ok := t.tokenToID[s]; ok {
			return id
		}
		return def
	}
	// Defaults are -1 (absent), NOT the Gemma ids: in non-Gemma vocabs (Qwen3) ids
	// 100/101/105/106 are REAL tokens, so a 105/106/100/101 fallback would make IsStop
	// spuriously stop and isControl silently DROP those tokens (word-salad). Gemma GGUFs
	// carry the literals so lookup hits and the default is never used → Gemma unchanged.
	t.StartOfTurn = lookup("<|turn>", -1)
	t.EndOfTurn = lookup("<turn|>", -1)
	t.ChannelOpen = lookup("<|channel>", -1)
	t.ChannelEnd = lookup("<channel|>", -1)
	t.ToolOpen = lookup("<|tool>", -1)
	t.ToolEnd = lookup("<tool|>", -1)
	t.ToolCallOpen = lookup("<|tool_call>", -1)
	t.ToolCallEnd = lookup("<tool_call|>", -1)
	t.ToolRespOpen = lookup("<|tool_response>", -1)
	t.ToolRespEnd = lookup("<tool_response|>", -1)
	t.StringDelim = lookup(`<|"|>`, -1)

	// Register the marker literals for Encode's pre-split scan (longest-first
	// so no marker can shadow a longer one sharing its prefix). <|think|> has
	// no dedicated struct field but appears in rendered prompts.
	for _, s := range []string{
		"<|turn>", "<turn|>", "<|channel>", "<channel|>",
		"<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
		"<|tool_response>", "<tool_response|>", `<|"|>`, "<|think|>",
	} {
		if id, ok := t.tokenToID[s]; ok {
			t.specials = append(t.specials, specialToken{str: s, id: id})
		}
	}
	sort.Slice(t.specials, func(i, j int) bool {
		return len(t.specials[i].str) > len(t.specials[j].str)
	})

	return t, nil
}

// DecodeRaw is like Decode but does NOT skip control/special tokens — it emits each
// token's literal vocab string (channel/tool/turn markers stay visible). The server
// uses it so the gemma-4 tool-call structure survives decoding for parsing.
func (t *Tokenizer) DecodeRaw(tokens []int32) string {
	var buf []byte
	for _, id := range tokens {
		if id < 0 || int(id) >= len(t.vocab) {
			continue
		}
		s := t.vocab[id]
		if s == "" {
			continue
		}
		if t.byteLevel {
			buf = appendByteLevel(buf, s)
			continue
		}
		if b, ok := parseByteToken(s); ok {
			buf = append(buf, b)
			continue
		}
		buf = append(buf, strings.ReplaceAll(s, spaceMarker, " ")...)
	}
	return string(buf)
}

// IsStop reports whether a token id terminates generation (EOS or end-of-turn).
func (t *Tokenizer) IsStop(id int32) bool {
	return id == t.EOS || id == t.EndOfTurn
}

// IsToolMarker reports whether an id is one of the gemma-4 tool-protocol markers
// (<|tool>, <tool|>, <|tool_call>, <tool_call|>, <|tool_response>, <tool_response|>,
// or the <|"|> string delimiter). These must never be streamed to the client as
// visible content — the server buffers them and parses structured tool_calls.
func (t *Tokenizer) IsToolMarker(id int32) bool {
	return (id == t.ToolOpen && t.ToolOpen >= 0) ||
		(id == t.ToolEnd && t.ToolEnd >= 0) ||
		(id == t.ToolCallOpen && t.ToolCallOpen >= 0) ||
		(id == t.ToolCallEnd && t.ToolCallEnd >= 0) ||
		(id == t.ToolRespOpen && t.ToolRespOpen >= 0) ||
		(id == t.ToolRespEnd && t.ToolRespEnd >= 0) ||
		(id == t.StringDelim && t.StringDelim >= 0)
}

// isControl reports whether an id is a special/control token that must not be
// rendered into user-visible text.
func (t *Tokenizer) isControl(id int32) bool {
	return id == t.BOS || id == t.EOS || id == t.PAD ||
		id == t.StartOfTurn || id == t.EndOfTurn ||
		id == t.ChannelOpen || id == t.ChannelEnd
}

// Encode tokenizes a string into token IDs using greedy longest-prefix match
// over the SentencePiece vocab. Spaces are first normalized to the "▁" marker
// (add_space_prefix=false, matching this GGUF), and any byte that cannot be
// matched falls back to its <0xXX> byte token rather than being dropped.
//
// NOTE: this is an approximation of true SentencePiece unigram (Viterbi over
// scores). It round-trips spaces and byte content correctly; exact id parity
// with llama.cpp for ambiguous merges is a known follow-up.
func (t *Tokenizer) Encode(text string, addBos bool, addEos bool) []int32 {
	var tokens []int32

	if addBos && t.AddBOS {
		tokens = append(tokens, t.BOS)
	}

	// SentencePiece space normalization: ' ' -> U+2581. Skipped for byte-level
	// BPE (Qwen3), where spaces are handled by the byte↔unicode map instead.
	if !t.byteLevel {
		text = strings.ReplaceAll(text, " ", spaceMarker)
	}

	// Split at control-marker literals FIRST (see Tokenizer.specials): the
	// greedy matcher below must never see a marker, or a piece straddling the
	// marker's leading '<' (e.g. ".<") can win the longest-prefix race and the
	// marker id is never produced.
	start := 0
	for i := 0; i < len(text); i++ {
		if text[i] != '<' {
			continue
		}
		for _, sp := range t.specials {
			if strings.HasPrefix(text[i:], sp.str) {
				tokens = t.encodeSegment(text[start:i], tokens)
				tokens = append(tokens, sp.id)
				i += len(sp.str) - 1 // -1: the loop increment adds it back
				start = i + 1
				break
			}
		}
	}
	tokens = t.encodeSegment(text[start:], tokens)

	if addEos {
		tokens = append(tokens, t.EOS)
	}

	return tokens
}

// encodeSegment encodes one control-marker-free, ▁-normalized text segment with
// whichever model this tokenizer carries: BPE merge-by-rank (HF tokenizer.json) or
// the unigram longest-prefix match (GGUF).
func (t *Tokenizer) encodeSegment(text string, tokens []int32) []int32 {
	if t.byteLevel {
		return t.bpeEncodeByteLevel(text, tokens)
	}
	if t.bpe {
		return t.bpeEncode(text, tokens)
	}
	return t.encodeGreedy(text, tokens)
}

// encodeGreedy runs the longest-prefix-match loop over a text segment that
// contains no control-marker literals, appending to tokens.
func (t *Tokenizer) encodeGreedy(text string, tokens []int32) []int32 {
	for len(text) > 0 {
		matched := false

		// Longest prefix match, bounded to a sane token byte length.
		maxLen := len(text)
		if maxLen > 48 {
			maxLen = 48
		}
		for l := maxLen; l >= 1; l-- {
			if id, ok := t.tokenToID[text[:l]]; ok {
				tokens = append(tokens, id)
				text = text[l:]
				matched = true
				break
			}
		}

		if !matched {
			// Byte fallback: emit the leading byte as its <0xXX> token.
			b := text[0]
			byteToken := fmt.Sprintf("<0x%02X>", b)
			if id, ok := t.tokenToID[byteToken]; ok {
				tokens = append(tokens, id)
			} else if id, ok := t.tokenToID[t.unkOr(b)]; ok {
				tokens = append(tokens, id)
			} else {
				tokens = append(tokens, t.PAD)
			}
			text = text[1:]
		}
	}
	return tokens
}

// unkOr returns the <unk> token string as a last-resort fallback label.
func (t *Tokenizer) unkOr(byte) string { return "<unk>" }

// Decode converts token IDs back to a string. It accumulates raw bytes so that
// <0xXX> byte tokens (which may split a multi-byte UTF-8 rune across several
// tokens) reassemble correctly, converts the SentencePiece "▁" marker to a
// space, and skips control tokens.
func (t *Tokenizer) Decode(tokens []int32) string {
	var buf []byte
	for _, id := range tokens {
		if id < 0 || int(id) >= len(t.vocab) {
			continue
		}
		if t.isControl(id) {
			continue
		}
		s := t.vocab[id]
		if s == "" {
			continue
		}
		if t.byteLevel {
			buf = appendByteLevel(buf, s)
			continue
		}
		// Raw byte token <0xXX> → the single byte it names.
		if b, ok := parseByteToken(s); ok {
			buf = append(buf, b)
			continue
		}
		// Normal piece: U+2581 marker → space, then append its UTF-8 bytes.
		buf = append(buf, strings.ReplaceAll(s, spaceMarker, " ")...)
	}
	return string(buf)
}

// parseByteToken parses a "<0xXX>" SentencePiece byte token into its byte value.
func parseByteToken(s string) (byte, bool) {
	if len(s) == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>' {
		hi, ok1 := hexNibble(s[3])
		lo, ok2 := hexNibble(s[4])
		if ok1 && ok2 {
			return hi<<4 | lo, true
		}
	}
	return 0, false
}

func hexNibble(c byte) (byte, bool) {
	switch {
	case c >= '0' && c <= '9':
		return c - '0', true
	case c >= 'A' && c <= 'F':
		return c - 'A' + 10, true
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10, true
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

// GetVocab returns the vocabulary slice.
func (t *Tokenizer) GetVocab() []string {
	return t.vocab
}
