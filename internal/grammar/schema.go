package grammar

// schema.go extends the JSON constraint to a JSON *Schema* constraint (the OpenAI
// Structured-Outputs subset): objects with typed properties + required keys +
// additionalProperties:false, arrays with a typed item schema, nested objects,
// enums, and the primitive types string / number / integer / boolean / null.
//
// Like json.go it is a byte-level pushdown automaton whose token masking clones the
// live state and simulates each candidate token's bytes. The traversal state is a
// stack of value/container frames; the schema tree itself is immutable and shared by
// pointer, so cloning a frame is a plain struct copy (no maps, no allocation) — which
// keeps Mask cheap even though it runs the whole vocabulary through the automaton.
//
// Bitmask limits (per object / per enum): at most 64 declared properties per object
// and 64 enum alternatives, so the candidate sets fit in a uint64. Schemas past those
// bounds are rejected at construction.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
)

// JSON type bits (a node may allow several, e.g. ["string","null"]).
const (
	tObject = 1 << iota
	tArray
	tString
	tNumber
	tInteger
	tBoolean
	tNull
)

const maxProps = 64 // object property / enum-alternative cap (uint64 candidate masks)

// schemaNode is one immutable, parsed schema node. Shared by pointer across cloned
// automaton states; never mutated after parseSchema returns.
type schemaNode struct {
	typeMask int
	propName []string      // object: property names, index == bit position
	propNode []*schemaNode // object: property value schemas, parallel to propName
	reqMask  uint64        // object: required properties (bit per property index)
	items    *schemaNode   // array: item schema (nil ⇒ any)
	enumVals []string      // enum: compacted JSON serializations of each alternative
}

func (n *schemaNode) has(bit int) bool { return n.typeMask&bit != 0 }

// rawSchema is the subset of JSON Schema keywords we decode.
type rawSchema struct {
	Type       json.RawMessage            `json:"type"`
	Properties map[string]json.RawMessage `json:"properties"`
	Required   []string                   `json:"required"`
	Items      json.RawMessage            `json:"items"`
	Enum       []json.RawMessage          `json:"enum"`
}

func parseSchema(raw json.RawMessage) (*schemaNode, error) {
	var rs rawSchema
	if err := json.Unmarshal(raw, &rs); err != nil {
		return nil, fmt.Errorf("schema: %w", err)
	}
	n := &schemaNode{}

	// type: a string, an array of strings, or absent (inferred below).
	if len(rs.Type) > 0 {
		var one string
		if err := json.Unmarshal(rs.Type, &one); err == nil {
			n.typeMask = typeBit(one)
		} else {
			var many []string
			if err := json.Unmarshal(rs.Type, &many); err != nil {
				return nil, fmt.Errorf("schema: bad \"type\": %s", rs.Type)
			}
			for _, t := range many {
				n.typeMask |= typeBit(t)
			}
		}
		if n.typeMask == 0 {
			return nil, fmt.Errorf("schema: unknown \"type\": %s", rs.Type)
		}
	}

	// enum: compact each alternative to the exact bytes we will force.
	if len(rs.Enum) > 0 {
		if len(rs.Enum) > maxProps {
			return nil, fmt.Errorf("schema: enum has %d alternatives (max %d)", len(rs.Enum), maxProps)
		}
		for _, e := range rs.Enum {
			var buf bytes.Buffer
			if err := json.Compact(&buf, e); err != nil {
				return nil, fmt.Errorf("schema: bad enum value: %w", err)
			}
			n.enumVals = append(n.enumVals, buf.String())
		}
	}

	// object properties (order is stable within this parse; bit indices derive from it).
	if len(rs.Properties) > 0 {
		if len(rs.Properties) > maxProps {
			return nil, fmt.Errorf("schema: object has %d properties (max %d)", len(rs.Properties), maxProps)
		}
		for name, sub := range rs.Properties {
			child, err := parseSchema(sub)
			if err != nil {
				return nil, err
			}
			n.propName = append(n.propName, name)
			n.propNode = append(n.propNode, child)
		}
	}
	// required — validated against the declared properties even when properties is absent.
	for _, r := range rs.Required {
		i := indexOf(n.propName, r)
		if i < 0 {
			return nil, fmt.Errorf("schema: required %q is not a declared property", r)
		}
		n.reqMask |= 1 << uint(i)
	}

	// array items.
	if len(rs.Items) > 0 {
		child, err := parseSchema(rs.Items)
		if err != nil {
			return nil, err
		}
		n.items = child
	}

	// Infer a type when omitted, from the shape that was declared.
	if n.typeMask == 0 && len(n.enumVals) == 0 {
		switch {
		case len(n.propName) > 0:
			n.typeMask = tObject
		case n.items != nil:
			n.typeMask = tArray
		default:
			n.typeMask = tObject | tArray | tString | tNumber | tInteger | tBoolean | tNull
		}
	}
	return n, nil
}

func indexOf(names []string, s string) int {
	for i, n := range names {
		if n == s {
			return i
		}
	}
	return -1
}

func typeBit(t string) int {
	switch t {
	case "object":
		return tObject
	case "array":
		return tArray
	case "string":
		return tString
	case "number":
		return tNumber
	case "integer":
		return tInteger
	case "boolean":
		return tBoolean
	case "null":
		return tNull
	}
	return 0
}

// schema automaton frame modes.
const (
	smValue    = iota // at the start of a value (skip ws, dispatch on first non-ws byte)
	smObjFirst        // after '{': first key or '}'
	smObjKey          // after ',': a key string is required
	smKeyStr          // inside an object key string
	smColon           // after a key: ':'
	smObjNext         // after a member value: ',' or '}'
	smArrFirst        // after '[': first element or ']'
	smArrNext         // after an element: ',' or ']'
	smStr             // inside a string value
	smNum             // inside a number value
	smLit             // inside a literal / enum value
)

// sframe is one traversal frame. It is copied by value when the state is cloned, so it
// holds no owned heap state: node/lits are shared read-only pointers/slices; the rest
// are scalars (candidate sets are uint64 bitmasks).
type sframe struct {
	node *schemaNode
	mode uint8

	used    uint64 // object: properties already consumed (bit per index)
	keyCand uint64 // key string: declared properties still prefix-matching
	keyPos  int    // key string: bytes matched so far
	curProp int    // object: property index selected for the pending value

	esc      bool // string: previous byte was a backslash
	numIsInt bool // number: integer-only (no '.'/'e')
	numSeen  bool // number: at least one digit consumed

	lits    []string // literal/enum: candidate serializations (shared, read-only)
	litMask uint64   // literal/enum: candidates still matching
	litPos  int      // literal/enum: bytes matched so far
}

var (
	litsTrue  = []string{"true"}
	litsFalse = []string{"false"}
	litsNull  = []string{"null"}
)

type schemaState struct {
	stack []sframe
	root  *schemaNode
}

func (s *schemaState) init(root *schemaNode) {
	s.root = root
	s.stack = s.stack[:0]
	s.push(root)
}

func (s *schemaState) push(node *schemaNode) {
	s.stack = append(s.stack, sframe{node: node, mode: smValue, curProp: -1})
}

func (s *schemaState) pop() { s.stack = s.stack[:len(s.stack)-1] }

func (s *schemaState) clone(dst *schemaState) {
	dst.root = s.root
	dst.stack = append(dst.stack[:0], s.stack...)
}

func (s *schemaState) complete() bool { return len(s.stack) == 0 }

// step feeds one byte; returns false if the byte is not legal under the schema.
func (s *schemaState) step(b byte) bool {
	for {
		if len(s.stack) == 0 {
			return isWS(b) // a complete top-level value: only trailing whitespace may follow
		}
		top := &s.stack[len(s.stack)-1]
		switch top.mode {
		case smStr:
			if top.esc {
				top.esc = false
				return true
			}
			if b == '\\' {
				top.esc = true
				return true
			}
			if b == '"' { // string value closes → the value is complete
				s.pop()
				return true
			}
			return b >= 0x20
		case smKeyStr:
			if b == '"' { // key closes: a candidate whose name length == keyPos must exist
				idx := s.resolveKey(top)
				if idx < 0 {
					return false
				}
				top.curProp = idx
				top.used |= 1 << uint(idx)
				top.mode = smColon
				return true
			}
			return s.narrowKey(top, b)
		case smNum:
			if top.numIsInt {
				if isDigit(b) {
					top.numSeen = true
					return true
				}
			} else if numCont(b) {
				if isDigit(b) {
					top.numSeen = true
				}
				return true
			}
			if !top.numSeen { // a bare '-' is not a number
				return false
			}
			s.pop() // number ends on this non-continuation byte; re-process it in the parent
			continue
		case smLit:
			done, ok := s.stepLit(top, b)
			if !ok {
				return false
			}
			if done {
				s.pop()
			}
			return true
		}
		// Root value start forbids leading whitespace (mirrors json_object): a greedy
		// model can otherwise loop on newlines and never begin the value.
		if len(s.stack) == 1 && top.mode == smValue {
			return s.beginValue(top, b)
		}
		// Structural modes: whitespace is legal between tokens.
		if isWS(b) {
			return true
		}
		switch top.mode {
		case smValue:
			return s.beginValue(top, b)
		case smObjFirst:
			if b == '}' {
				if top.reqSatisfied() {
					s.pop()
					return true
				}
				return false
			}
			if b == '"' && top.hasUnused() {
				s.startKey(top)
				return true
			}
			return false
		case smObjKey:
			if b == '"' && top.hasUnused() {
				s.startKey(top)
				return true
			}
			return false
		case smColon:
			if b == ':' {
				child := top.node.propNode[top.curProp]
				top.mode = smObjNext // return here once the value frame pops
				s.push(child)
				return true
			}
			return false
		case smObjNext:
			if b == ',' && top.hasUnused() { // only continue if a declared key remains
				top.mode = smObjKey
				return true
			}
			if b == '}' {
				if top.reqSatisfied() {
					s.pop()
					return true
				}
				return false
			}
			return false
		case smArrFirst:
			if b == ']' { // empty array
				s.pop()
				return true
			}
			top.mode = smArrNext // return here once each element frame pops
			s.push(top.node.items)
			continue // re-process b as the first byte of the element
		case smArrNext:
			if b == ',' {
				s.push(top.node.items)
				return true
			}
			if b == ']' {
				s.pop()
				return true
			}
			return false
		}
		return false
	}
}

func (f *sframe) reqSatisfied() bool { return f.used&f.node.reqMask == f.node.reqMask }

// fullMask is the bitmask of all declared properties (additionalProperties:false ⇒ the
// legal key set). hasUnused reports whether any declared property is not yet consumed.
func (f *sframe) fullMask() uint64 {
	np := len(f.node.propName)
	if np == 0 {
		return 0
	}
	if np >= 64 {
		return ^uint64(0)
	}
	return (uint64(1) << uint(np)) - 1
}

func (f *sframe) hasUnused() bool { return f.fullMask()&^f.used != 0 }

// startKey begins an object key string: candidates are the declared properties not yet
// used (additionalProperties:false — undeclared keys are never legal).
func (s *schemaState) startKey(f *sframe) {
	f.mode = smKeyStr
	f.keyPos = 0
	f.keyCand = f.fullMask() &^ f.used
}

// narrowKey keeps only candidates whose name has byte b at position keyPos.
func (s *schemaState) narrowKey(f *sframe, b byte) bool {
	next := uint64(0)
	m := f.keyCand
	for m != 0 {
		i := bits_TrailingZeros(m)
		m &= m - 1
		name := f.node.propName[i]
		if f.keyPos < len(name) && name[f.keyPos] == b {
			next |= 1 << uint(i)
		}
	}
	if next == 0 {
		return false
	}
	f.keyCand = next
	f.keyPos++
	return true
}

// resolveKey returns the candidate whose name is exactly keyPos bytes long (the key just
// closed), or -1 if none.
func (s *schemaState) resolveKey(f *sframe) int {
	m := f.keyCand
	for m != 0 {
		i := bits_TrailingZeros(m)
		m &= m - 1
		if len(f.node.propName[i]) == f.keyPos {
			return i
		}
	}
	return -1
}

// beginValue dispatches the first non-whitespace byte of a value against the node's
// allowed types (enum, if present, takes precedence).
func (s *schemaState) beginValue(f *sframe, b byte) bool {
	if len(f.node.enumVals) > 0 {
		return s.enterLit(f, f.node.enumVals, b)
	}
	n := f.node
	switch {
	case b == '{' && n.has(tObject):
		f.mode = smObjFirst
		f.used = 0
		return true
	case b == '[' && n.has(tArray):
		f.mode = smArrFirst
		return true
	case b == '"' && n.has(tString):
		f.mode = smStr
		f.esc = false
		return true
	case (b == '-' || isDigit(b)) && (n.has(tNumber) || n.has(tInteger)):
		f.mode = smNum
		f.numIsInt = n.has(tInteger) && !n.has(tNumber)
		f.numSeen = isDigit(b)
		return true
	case b == 't' && n.has(tBoolean):
		return s.enterLit(f, litsTrue, b)
	case b == 'f' && n.has(tBoolean):
		return s.enterLit(f, litsFalse, b)
	case b == 'n' && n.has(tNull):
		return s.enterLit(f, litsNull, b)
	}
	return false
}

// enterLit starts matching a literal/enum value; b is its first byte.
func (s *schemaState) enterLit(f *sframe, lits []string, b byte) bool {
	mask := uint64(0)
	for i, l := range lits {
		if len(l) > 0 && l[0] == b {
			mask |= 1 << uint(i)
		}
	}
	if mask == 0 {
		return false
	}
	f.lits = lits
	f.litMask = mask
	f.litPos = 1
	f.mode = smLit
	// A 1-byte literal would already be complete; none exist here (true/false/null and
	// quoted enum strings are all ≥3 bytes), but stay correct if that ever changes.
	if done, _ := s.litCompletes(f); done {
		s.pop()
	}
	return true
}

// stepLit advances the literal/enum match by byte b. Returns (done, ok): ok=false when
// no candidate matches; done=true when a candidate is fully consumed on this byte.
func (s *schemaState) stepLit(f *sframe, b byte) (done, ok bool) {
	next := uint64(0)
	m := f.litMask
	for m != 0 {
		i := bits_TrailingZeros(m)
		m &= m - 1
		l := f.lits[i]
		if f.litPos < len(l) && l[f.litPos] == b {
			next |= 1 << uint(i)
		}
	}
	if next == 0 {
		return false, false
	}
	f.litMask = next
	f.litPos++
	d, _ := s.litCompletes(f)
	return d, true
}

// litCompletes reports whether some still-matching candidate is exactly litPos bytes
// long (fully consumed).
func (s *schemaState) litCompletes(f *sframe) (bool, int) {
	m := f.litMask
	for m != 0 {
		i := bits_TrailingZeros(m)
		m &= m - 1
		if len(f.lits[i]) == f.litPos {
			return true, i
		}
	}
	return false, -1
}

// bits_TrailingZeros is math/bits.TrailingZeros64 without the import churn; m must be nonzero.
func bits_TrailingZeros(m uint64) int {
	n := 0
	for m&1 == 0 {
		m >>= 1
		n++
	}
	return n
}

// JSONSchema is a Constraint that forces output matching a JSON Schema (subset).
type JSONSchema struct {
	pieces [][]byte
	eos    int32
	st     schemaState
	scr    schemaState // reused scratch for per-token simulation
}

// NewJSONSchema builds a schema constraint over the token pieces. Returns an error if
// the schema is malformed or exceeds the property/enum bounds.
func NewJSONSchema(pieces [][]byte, eos int32, schema json.RawMessage) (*JSONSchema, error) {
	root, err := parseSchema(schema)
	if err != nil {
		return nil, err
	}
	j := &JSONSchema{pieces: pieces, eos: eos}
	j.st.init(root)
	return j, nil
}

func (j *JSONSchema) Done() bool { return j.st.complete() }

func (j *JSONSchema) allows(piece []byte) bool {
	if len(piece) == 0 {
		return false
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
func (j *JSONSchema) Mask(logits []float32) {
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
	for id := len(j.pieces); id < len(logits); id++ {
		logits[id] = neg
	}
}

// Accept advances the live automaton by the chosen token's bytes.
func (j *JSONSchema) Accept(id int32) {
	if id == j.eos || int(id) >= len(j.pieces) {
		return
	}
	for _, b := range j.pieces[id] {
		if !j.st.step(b) {
			return
		}
	}
}

// Close completes a truncated value into parseable (not necessarily schema-valid) JSON:
// finish any in-progress scalar, then close every open container from the top down.
func (j *JSONSchema) Close() []byte {
	st := &j.st
	if st.complete() {
		return nil
	}
	var b []byte
	for i := len(st.stack) - 1; i >= 0; i-- {
		f := &st.stack[i]
		switch f.mode {
		case smStr:
			if f.esc {
				b = append(b, '\\')
			}
			b = append(b, '"')
		case smKeyStr:
			if f.esc {
				b = append(b, '\\')
			}
			b = append(b, '"', ':', 'n', 'u', 'l', 'l') // key with no value yet
		case smColon:
			b = append(b, ':', 'n', 'u', 'l', 'l')
		case smValue:
			b = append(b, 'n', 'u', 'l', 'l')
		case smNum:
			if !f.numSeen {
				b = append(b, '0') // dangling '-' → -0
			}
		case smLit:
			if _, idx := firstBit(f.litMask); idx >= 0 { // finish the chosen literal/enum
				b = append(b, f.lits[idx][f.litPos:]...)
			}
		}
		switch f.mode {
		case smObjFirst, smObjKey, smKeyStr, smColon, smObjNext:
			b = append(b, '}')
		case smArrFirst, smArrNext:
			b = append(b, ']')
		}
	}
	return b
}

func firstBit(m uint64) (uint64, int) {
	if m == 0 {
		return 0, -1
	}
	return m, bits_TrailingZeros(m)
}
