package server

import (
	"encoding/json"
	"testing"
)

func TestStopFieldUnmarshal(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{`{"stop":"END"}`, []string{"END"}},
		{`{"stop":["a","b"]}`, []string{"a", "b"}},
		{`{"stop":null}`, nil},
		{`{}`, nil},
	}
	for _, c := range cases {
		var req ChatRequest
		if err := json.Unmarshal([]byte(c.in), &req); err != nil {
			t.Fatalf("%s: %v", c.in, err)
		}
		if len(req.Stop) != len(c.want) {
			t.Fatalf("%s: got %v want %v", c.in, req.Stop, c.want)
		}
		for i := range c.want {
			if req.Stop[i] != c.want[i] {
				t.Errorf("%s: got %v want %v", c.in, req.Stop, c.want)
			}
		}
	}
}

func TestLegacyPrompt(t *testing.T) {
	cases := []struct{ in, want string }{
		{`{"prompt":"hello"}`, "hello"},
		{`{"prompt":["a","b"]}`, "a\nb"},
		{`{"messages":[]}`, ""},
	}
	for _, c := range cases {
		var req ChatRequest
		if err := json.Unmarshal([]byte(c.in), &req); err != nil {
			t.Fatalf("%s: %v", c.in, err)
		}
		if got := req.legacyPrompt(); got != c.want {
			t.Errorf("%s: got %q want %q", c.in, got, c.want)
		}
	}
}

func TestStopHit(t *testing.T) {
	cases := []struct {
		text    string
		stops   []string
		hit     bool
		trimmed string
	}{
		{"hello END world", []string{"END"}, true, "hello "},
		{"no stop here", []string{"END"}, false, "no stop here"},
		{"firstSTOPsecondHALT", []string{"HALT", "STOP"}, true, "first"}, // earliest wins
		{"abc", []string{""}, false, "abc"},                              // empty stop ignored
		{"abc", nil, false, "abc"},
	}
	for _, c := range cases {
		hit, trimmed := stopHit(c.text, c.stops)
		if hit != c.hit || trimmed != c.trimmed {
			t.Errorf("stopHit(%q,%v) = (%v,%q) want (%v,%q)",
				c.text, c.stops, hit, trimmed, c.hit, c.trimmed)
		}
	}
}

func TestToolCallUniqueIDs(t *testing.T) {
	raw := `<|tool_call>call:search{q: <|"|>a<|"|>}<tool_call|>` +
		`<|tool_call>call:search{q: <|"|>b<|"|>}<tool_call|>`
	_, calls := parseToolCalls(raw)
	if len(calls) != 2 {
		t.Fatalf("got %d calls", len(calls))
	}
	if calls[0].ID == calls[1].ID {
		t.Errorf("duplicate tool-call ids: %q == %q", calls[0].ID, calls[1].ID)
	}
	if calls[0].ID == "" || calls[1].ID == "" {
		t.Errorf("empty tool-call id: %q %q", calls[0].ID, calls[1].ID)
	}
}
