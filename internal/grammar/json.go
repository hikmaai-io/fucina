// Package grammar provides constrained decoding: a Constraint masks the tokens a
// sampler may pick so the generated text is guaranteed to follow a structure. The
// JSON constraint makes a malformed or unterminated JSON value structurally
// impossible — at every step it forbids any token whose bytes would break JSON, and
// it forbids the end-of-sequence token until the value is syntactically complete.
//
// Design: a byte-level JSON pushdown automaton (step). Token masking simulates
// feeding each candidate token's bytes through a COPY of the live automaton; a token
// is allowed iff every byte is accepted. The live automaton then advances on the
// token actually chosen (Accept). Special/control tokens (those that decode to the
// empty string or contain bytes JSON never allows) are masked out; EOS is allowed
// only at a complete value.
package grammar

import "math"

// Constraint masks invalid next tokens and advances on the accepted one.
type Constraint interface {
	// Mask sets logits[id] = -inf for every token id that is not a legal next token.
	Mask(logits []float32)
	// Accept advances the constraint state by the chosen token.
	Accept(id int32)
	// Done reports whether generation may stop here (the structure is complete).
	Done() bool
	// Close returns the minimal byte sequence that completes the structure from the
	// current state (e.g. close an open string + close open containers). Empty when
	// already complete. Used to finish a truncated structure at the token cap so the
	// output is valid even when generation is cut short.
	Close() []byte
}

// json automaton modes.
const (
	mValue   = iota // expecting the start of a value
	mObjKey1        // right after '{': a string key or '}'
	mObjKey         // after ',' in an object: a string key
	mColon          // after a key: ':'
	mObjNext        // after a value in an object: ',' or '}'
	mArrNext        // after a value in an array: ',' or ']'
	mString         // inside a string (see esc/isKey)
	mNumber         // inside a number
	mLit            // inside a literal true/false/null (see litWord/litPos)
	mEnd            // top-level value complete
)

type jsonState struct {
	stack      []byte // container stack: 'o' object, 'a' array (bottom→top)
	mode       int
	esc        bool   // in mString: previous byte was a backslash
	litWord    string // in mLit: the literal being matched
	litPos     int    // in mLit: next index to match
	postLit    int    // mode to return to after a number/literal completes
	requireObj bool   // top-level value must be an object (OpenAI json_object)
}

func (s *jsonState) clone(dst *jsonState) {
	dst.stack = append(dst.stack[:0], s.stack...)
	dst.mode = s.mode
	dst.esc = s.esc
	dst.litWord = s.litWord
	dst.litPos = s.litPos
	dst.postLit = s.postLit
	dst.requireObj = s.requireObj
}

// afterValue returns the mode entered once a value finishes, given the container stack.
func (s *jsonState) afterValue() int {
	if len(s.stack) == 0 {
		return mEnd
	}
	if s.stack[len(s.stack)-1] == 'o' {
		return mObjNext
	}
	return mArrNext
}

func isWS(b byte) bool   { return b == ' ' || b == '\t' || b == '\n' || b == '\r' }
func isDigit(b byte) bool { return b >= '0' && b <= '9' }
func numCont(b byte) bool {
	return isDigit(b) || b == '.' || b == 'e' || b == 'E' || b == '+' || b == '-'
}

// step feeds one byte; returns false (and leaves the state unspecified) if the byte
// is not legal. Numbers and literals end on the first non-continuation byte, which is
// then re-processed in the post-value mode (the loop).
func (s *jsonState) step(b byte) bool {
	for {
		switch s.mode {
		case mString:
			if s.esc {
				s.esc = false
				return true // any byte may follow a backslash (rough but safe for masking)
			}
			if b == '\\' {
				s.esc = true
				return true
			}
			if b == '"' { // string closes
				s.mode = s.postLit
				return true
			}
			return b >= 0x20 // control bytes are illegal in JSON strings
		case mNumber:
			if numCont(b) {
				return true
			}
			s.mode = s.postLit // number ends; re-process this byte
			continue
		case mLit:
			if s.litPos < len(s.litWord) && b == s.litWord[s.litPos] {
				s.litPos++
				if s.litPos == len(s.litWord) {
					s.mode = s.postLit
				}
				return true
			}
			return false
		}
		// Top-level start under json_object: force '{' immediately. Leading whitespace IS
		// valid JSON, but the model (greedy) can loop on newlines/spaces forever and never
		// start the object; disallowing it here keeps the output valid and unblocks it.
		if s.requireObj && len(s.stack) == 0 && s.mode == mValue {
			return s.beginValue(b)
		}
		if isWS(b) {
			return true // whitespace is legal between tokens (not inside strings/numbers, handled above)
		}
		switch s.mode {
		case mValue:
			return s.beginValue(b)
		case mObjKey1:
			if b == '}' {
				s.popContainer()
				return true
			}
			if b == '"' {
				s.mode, s.postLit = mString, mColon
				return true
			}
			return false
		case mObjKey:
			if b == '"' {
				s.mode, s.postLit = mString, mColon
				return true
			}
			return false
		case mColon:
			if b == ':' {
				s.mode = mValue
				return true
			}
			return false
		case mObjNext:
			if b == ',' {
				s.mode = mObjKey
				return true
			}
			if b == '}' {
				s.popContainer()
				return true
			}
			return false
		case mArrNext:
			if b == ',' {
				s.mode = mValue
				return true
			}
			if b == ']' {
				s.popContainer()
				return true
			}
			return false
		case mEnd:
			return false // nothing may follow a complete top-level value (except whitespace, handled)
		}
		return false
	}
}

func (s *jsonState) beginValue(b byte) bool {
	// json_object: the TOP-level value must be an object (sidesteps bare-number
	// termination too — completion is unambiguously the closing '}').
	if s.requireObj && len(s.stack) == 0 {
		if b == '{' {
			s.stack = append(s.stack, 'o')
			s.mode = mObjKey1
			return true
		}
		return false
	}
	switch {
	case b == '{':
		s.stack = append(s.stack, 'o')
		s.mode = mObjKey1
		return true
	case b == '[':
		s.stack = append(s.stack, 'a')
		s.mode = mValue // array may be empty: also allow ']' below
		// allow immediate close of an empty array
		return true
	case b == '"':
		s.mode, s.postLit = mString, s.afterValue()
		return true
	case b == '-' || isDigit(b):
		s.mode, s.postLit = mNumber, s.afterValue()
		return true
	case b == 't':
		s.mode, s.litWord, s.litPos, s.postLit = mLit, "true", 1, s.afterValue()
		return true
	case b == 'f':
		s.mode, s.litWord, s.litPos, s.postLit = mLit, "false", 1, s.afterValue()
		return true
	case b == 'n':
		s.mode, s.litWord, s.litPos, s.postLit = mLit, "null", 1, s.afterValue()
		return true
	case b == ']':
		// empty array close: only legal if the enclosing container is an array we just opened
		if len(s.stack) > 0 && s.stack[len(s.stack)-1] == 'a' {
			s.popContainer()
			return true
		}
		return false
	}
	return false
}

func (s *jsonState) popContainer() {
	if len(s.stack) > 0 {
		s.stack = s.stack[:len(s.stack)-1]
	}
	s.mode = s.afterValue()
}

// complete reports whether the automaton is at a point where the value is finished.
func (s *jsonState) complete() bool { return s.mode == mEnd }

// JSON is a Constraint that forces well-formed JSON output.
type JSON struct {
	pieces [][]byte // decoded UTF-8 bytes per token id ("" for control tokens)
	eos    int32
	st     jsonState
	scr    jsonState // reused scratch for per-token simulation
}

// NewJSON builds a JSON constraint. pieces[id] is the literal text token id emits
// (the tokenizer's decoded piece); control/special tokens should map to empty. eos is
// the end-of-sequence id, allowed only at a complete value.
// NewJSON builds a json_object constraint (top-level value must be an object).
func NewJSON(pieces [][]byte, eos int32) *JSON {
	return &JSON{pieces: pieces, eos: eos, st: jsonState{mode: mValue, requireObj: true}}
}

func (j *JSON) Done() bool { return j.st.complete() }

// Close completes a truncated JSON value: close an in-progress string/value, then close
// every open container (top of stack first). Approximate for rare mid-token states but
// yields parseable JSON for the common truncation cases (mid-string, mid-value, open
// objects/arrays).
func (j *JSON) Close() []byte {
	s := &j.st
	if s.complete() {
		return nil
	}
	var b []byte
	switch s.mode {
	case mString:
		if s.esc {
			b = append(b, '\\') // dangling backslash → escape itself
		}
		b = append(b, '"') // close the string
	case mColon:
		b = append(b, ':', 'n', 'u', 'l', 'l') // a key with no value yet
	case mValue:
		b = append(b, 'n', 'u', 'l', 'l') // expected a value
	}
	for i := len(s.stack) - 1; i >= 0; i-- {
		if s.stack[i] == 'o' {
			b = append(b, '}')
		} else {
			b = append(b, ']')
		}
	}
	return b
}

func (j *JSON) allows(piece []byte) bool {
	if len(piece) == 0 {
		return false // control tokens are not JSON text
	}
	j.st.clone(&j.scr)
	for _, b := range piece {
		if !j.scr.step(b) {
			return false
		}
	}
	return true
}

// Mask sets the logits of illegal tokens to -inf. EOS is legal only when complete.
func (j *JSON) Mask(logits []float32) {
	neg := float32(math.Inf(-1))
	for id := 0; id < len(logits) && id < len(j.pieces); id++ {
		if int32(id) == j.eos {
			if !j.st.complete() {
				logits[id] = neg
			}
			continue
		}
		if !j.allows(j.pieces[id]) {
			logits[id] = neg
		}
	}
	// ids past len(pieces) (shouldn't happen) are forbidden.
	for id := len(j.pieces); id < len(logits); id++ {
		logits[id] = neg
	}
}

// Accept advances the live automaton by the chosen token's bytes.
func (j *JSON) Accept(id int32) {
	if id == j.eos || int(id) >= len(j.pieces) {
		return
	}
	for _, b := range j.pieces[id] {
		if !j.st.step(b) {
			return // should not happen if Mask was honored
		}
	}
}
