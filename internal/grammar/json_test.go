package grammar

import (
	"math"
	"testing"
)

func buildVocab(strs []string) ([][]byte, map[string]int32) {
	pieces := make([][]byte, len(strs))
	idx := map[string]int32{}
	for i, s := range strs {
		pieces[i] = []byte(s)
		idx[s] = int32(i)
	}
	return pieces, idx
}

// allowed returns the set of token ids NOT masked to -inf at the current state.
func allowed(j *JSON, n int) map[int]bool {
	lg := make([]float32, n)
	j.Mask(lg)
	out := map[int]bool{}
	for i, v := range lg {
		if !math.IsInf(float64(v), -1) {
			out[i] = true
		}
	}
	return out
}

func TestJSONStartRequiresObject(t *testing.T) {
	strs := []string{"{", "}", "[", ":", ",", `"`, "1", "true", `"a"`, "EOS", " "}
	pieces, idx := buildVocab(strs)
	eos := idx["EOS"]
	pieces[eos] = nil // control token decodes to nothing
	j := NewJSON(pieces, eos)

	al := allowed(j, len(strs))
	if !al[int(idx["{"])] {
		t.Fatal("'{' must be allowed to start a json_object")
	}
	for _, bad := range []string{"}", "[", ":", ",", "1", "true", `"a"`, `"`} {
		if al[int(idx[bad])] {
			t.Fatalf("%q must NOT start a json_object", bad)
		}
	}
	if al[int(eos)] {
		t.Fatal("EOS must be forbidden before any value")
	}
	if al[int(idx[" "])] == false {
		t.Fatal("leading whitespace should be allowed")
	}
}

func TestJSONObjectFlow(t *testing.T) {
	strs := []string{"{", "}", ":", ",", `"a"`, `"b"`, "1", "EOS"}
	pieces, idx := buildVocab(strs)
	eos := idx["EOS"]
	pieces[eos] = nil
	j := NewJSON(pieces, eos)

	j.Accept(idx["{"])
	al := allowed(j, len(strs))
	if !al[int(idx[`"a"`])] || !al[int(idx["}"])] {
		t.Fatal("after '{': a key string or '}' must be allowed")
	}
	if al[int(idx["1"])] || al[int(idx[":"])] || al[int(idx[","])] || al[int(eos)] {
		t.Fatal("after '{': number/':'/','/EOS must be forbidden")
	}

	j.Accept(idx[`"a"`]) // key
	al = allowed(j, len(strs))
	if !al[int(idx[":"])] {
		t.Fatal("after a key, ':' is required")
	}
	if al[int(idx[","])] || al[int(idx["}"])] || al[int(eos)] {
		t.Fatal("after a key, only ':' is legal")
	}

	j.Accept(idx[":"])
	al = allowed(j, len(strs))
	if !al[int(idx["1"])] || !al[int(idx[`"b"`])] {
		t.Fatal("after ':', a value must be allowed")
	}
	if al[int(idx["}"])] || al[int(eos)] {
		t.Fatal("after ':', '}'/EOS must be forbidden (value required)")
	}

	j.Accept(idx["1"]) // number value (still 'open' until a delimiter)
	al = allowed(j, len(strs))
	if !al[int(idx["}"])] || !al[int(idx[","])] {
		t.Fatal("after a number value, ','/'}' must close/continue")
	}
	if al[int(idx[":"])] || al[int(eos)] {
		t.Fatal("mid-object EOS / ':' must be forbidden")
	}

	j.Accept(idx["}"]) // close object
	if !j.Done() {
		t.Fatal(`{"a":1} must be complete`)
	}
	al = allowed(j, len(strs))
	if !al[int(eos)] {
		t.Fatal("EOS must be allowed at a complete object")
	}
	if al[int(idx[","])] || al[int(idx["{"])] {
		t.Fatal("nothing but whitespace/EOS after a complete top-level value")
	}
}

// A multi-byte token that spans several automaton transitions must be validated as a
// whole: "1}" both ends the number and closes the object.
func TestJSONMultiByteToken(t *testing.T) {
	strs := []string{"{", `"a"`, ":", "1}", "1", "EOS"}
	pieces, idx := buildVocab(strs)
	eos := idx["EOS"]
	pieces[eos] = nil
	j := NewJSON(pieces, eos)
	j.Accept(idx["{"])
	j.Accept(idx[`"a"`])
	j.Accept(idx[":"])
	al := allowed(j, len(strs))
	if !al[int(idx["1}"])] {
		t.Fatal(`the token "1}" (number then close) must be allowed as a value`)
	}
	j.Accept(idx["1}"])
	if !j.Done() {
		t.Fatal(`{"a":1} via "1}" token must be complete`)
	}
}

func TestJSONNestedAndString(t *testing.T) {
	strs := []string{"{", "}", "[", "]", ":", ",", `"a"`, `"x\"y"`, "true", "EOS"}
	pieces, idx := buildVocab(strs)
	eos := idx["EOS"]
	pieces[eos] = nil
	j := NewJSON(pieces, eos)
	// {"a":["x\"y",true]}
	for _, s := range []string{"{", `"a"`, ":", "[", `"x\"y"`, ",", "true", "]", "}"} {
		al := allowed(j, len(strs))
		if !al[int(idx[s])] {
			t.Fatalf("step %q should be allowed", s)
		}
		j.Accept(idx[s])
	}
	if !j.Done() {
		t.Fatal("nested object/array/escaped-string must complete")
	}
}
