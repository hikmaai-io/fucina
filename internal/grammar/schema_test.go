package grammar

import (
	"encoding/json"
	"math"
	"testing"
)

// byteVocab builds a vocabulary where every ASCII byte is its own token, plus an EOS.
// This lets tests drive the automaton one character at a time.
func byteVocab() ([][]byte, map[string]int32, int32) {
	strs := make([]string, 0, 130)
	for c := 0; c < 128; c++ {
		strs = append(strs, string([]byte{byte(c)}))
	}
	strs = append(strs, "EOS")
	pieces := make([][]byte, len(strs))
	idx := map[string]int32{}
	for i, s := range strs {
		pieces[i] = []byte(s)
		idx[s] = int32(i)
	}
	eos := idx["EOS"]
	pieces[eos] = nil
	return pieces, idx, eos
}

func newSchema(t *testing.T, schemaJSON string) (*JSONSchema, int) {
	t.Helper()
	pieces, _, eos := byteVocab()
	j, err := NewJSONSchema(pieces, eos, json.RawMessage(schemaJSON))
	if err != nil {
		t.Fatalf("NewJSONSchema(%s): %v", schemaJSON, err)
	}
	return j, len(pieces)
}

func maskAllowed(c Constraint, n int) map[byte]bool {
	lg := make([]float32, n)
	c.Mask(lg)
	out := map[byte]bool{}
	for i := 0; i < 128; i++ {
		if !math.IsInf(float64(lg[i]), -1) {
			out[byte(i)] = true
		}
	}
	return out
}

// drive feeds s byte-by-byte, asserting each byte is allowed at its step, then Accepts it.
func drive(t *testing.T, j *JSONSchema, n int, s string) {
	t.Helper()
	for i := 0; i < len(s); i++ {
		al := maskAllowed(j, n)
		if !al[s[i]] {
			t.Fatalf("byte %q at pos %d of %q was masked (illegal)", s[i], i, s)
		}
		j.Accept(int32(s[i]))
	}
}

func TestSchemaObjectRequiredKeyAndType(t *testing.T) {
	j, n := newSchema(t, `{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}`)

	// Start: must be '{'.
	al := maskAllowed(j, n)
	if !al['{'] || al['['] || al['"'] || al[' '] {
		t.Fatalf("start: only '{' should be allowed, got %v", al)
	}
	j.Accept('{')

	// After '{': the key "a" ('"') is allowed; '}' is NOT (a is required).
	al = maskAllowed(j, n)
	if !al['"'] {
		t.Fatal(`after '{': a key '"' must be allowed`)
	}
	if al['}'] {
		t.Fatal(`after '{': '}' must be forbidden (required key "a" missing)`)
	}
	j.Accept('"')
	// key byte must be 'a' (only declared property); 'b' is masked.
	al = maskAllowed(j, n)
	if !al['a'] || al['b'] {
		t.Fatalf("key: only 'a' allowed, got a=%v b=%v", al['a'], al['b'])
	}
	drive(t, j, n, `a":`)

	// After ':': integer value — a number start is allowed, a string '"' is NOT.
	al = maskAllowed(j, n)
	if !al['1'] || !al['-'] {
		t.Fatal("after ':': integer value must allow digits / '-'")
	}
	if al['"'] {
		t.Fatal("after ':': a string value must be forbidden for an integer property")
	}
	drive(t, j, n, `1}`)
	if !j.Done() {
		t.Fatalf(`{"a":1} must complete`)
	}
	// EOS allowed only now (complete).
	lg := make([]float32, n)
	j.Mask(lg)
	if math.IsInf(float64(lg[j.eos]), -1) {
		t.Fatal("EOS must be allowed at a complete value")
	}
}

func TestSchemaEnum(t *testing.T) {
	j, n := newSchema(t, `{"enum":["red","green"]}`)
	// Enum of strings: value must begin with '"'.
	al := maskAllowed(j, n)
	if !al['"'] || al['r'] {
		t.Fatalf("enum start: only '\"' allowed, got %v", al)
	}
	j.Accept('"')
	// First char must be 'r' or 'g'; not 'b'.
	al = maskAllowed(j, n)
	if !al['r'] || !al['g'] || al['b'] {
		t.Fatalf("enum first char: r/g allowed, b not; got r=%v g=%v b=%v", al['r'], al['g'], al['b'])
	}
	j.Accept('r')
	// Committed to "red": next must be 'e'.
	al = maskAllowed(j, n)
	if !al['e'] || al['a'] {
		t.Fatalf("enum after 'r': only 'e' allowed, got %v", al)
	}
	drive(t, j, n, `ed"`)
	if !j.Done() {
		t.Fatal(`"red" must complete the enum`)
	}
}

func TestSchemaArrayOfBooleans(t *testing.T) {
	j, n := newSchema(t, `{"type":"array","items":{"type":"boolean"}}`)
	al := maskAllowed(j, n)
	if !al['['] {
		t.Fatal("array must start with '['")
	}
	j.Accept('[')
	// First element: boolean → 't'/'f' allowed, a number is not; ']' closes (empty ok).
	al = maskAllowed(j, n)
	if !al['t'] || !al['f'] || !al[']'] {
		t.Fatal("array element: booleans and ']' must be allowed")
	}
	if al['1'] {
		t.Fatal("array element: a number must be forbidden for a boolean item schema")
	}
	drive(t, j, n, `true,false]`)
	if !j.Done() {
		t.Fatal("[true,false] must complete")
	}
}

func TestSchemaNestedObject(t *testing.T) {
	j, n := newSchema(t, `{"type":"object","properties":{"o":{"type":"object","properties":{"n":{"type":"integer"}},"required":["n"]}},"required":["o"]}`)
	drive(t, j, n, `{"o":{"n":5}}`)
	if !j.Done() {
		t.Fatal(`nested {"o":{"n":5}} must complete`)
	}
}

func TestSchemaAdditionalPropsFalse(t *testing.T) {
	j, n := newSchema(t, `{"type":"object","properties":{"a":{"type":"integer"}}}`)
	drive(t, j, n, `{"a":1`)
	// The only declared property is used: no ',' (would need another key), only '}'.
	al := maskAllowed(j, n)
	if al[','] {
		t.Fatal("',' must be forbidden once every declared property is used (additionalProperties:false)")
	}
	if !al['}'] {
		t.Fatal("'}' must be allowed to close the object")
	}
	j.Accept('}')
	if !j.Done() {
		t.Fatal("object must complete")
	}
}

func TestSchemaNullableString(t *testing.T) {
	j, n := newSchema(t, `{"type":["string","null"]}`)
	al := maskAllowed(j, n)
	if !al['"'] || !al['n'] {
		t.Fatal(`["string","null"]: both '"' and 'n' (null) must be allowed`)
	}
	if al['1'] {
		t.Fatal("a number must be forbidden for a string|null value")
	}
	drive(t, j, n, "null")
	if !j.Done() {
		t.Fatal("null must complete a string|null value")
	}
}

func TestSchemaForceClose(t *testing.T) {
	j, n := newSchema(t, `{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}`)
	drive(t, j, n, `{"a":"hel`) // truncated mid-string value
	closing := j.Close()
	full := `{"a":"hel` + string(closing)
	var obj map[string]interface{}
	if err := json.Unmarshal([]byte(full), &obj); err != nil {
		t.Fatalf("force-closed output is not valid JSON: %v (%q)", err, full)
	}
	if obj["a"] != "hel" {
		t.Errorf("force-closed = %v want {\"a\":\"hel\"}", obj)
	}
}

func TestSchemaParseErrors(t *testing.T) {
	pieces, _, eos := byteVocab()
	cases := []string{
		`{"type":"object","required":["missing"]}`,      // required not a declared property
		`{"type":"nonsense"}`,                            // unknown type
		`not json`,                                       // malformed
	}
	for _, c := range cases {
		if _, err := NewJSONSchema(pieces, eos, json.RawMessage(c)); err == nil {
			t.Errorf("expected an error for schema %q", c)
		}
	}
}
