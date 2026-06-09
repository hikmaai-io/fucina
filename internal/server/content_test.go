package server

import "testing"
import "encoding/json"

func TestContentParts(t *testing.T) {
	cases := []struct{ in, want string }{
		{`{"role":"user","content":"hi"}`, "hi"},
		{`{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}`, "ab"},
		{`{"role":"user","content":null}`, ""},
		{`{"role":"user"}`, ""},
		{`{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}},{"type":"text","text":"c"}]}`, "c"},
	}
	for _, c := range cases {
		var m ChatMessage
		if err := json.Unmarshal([]byte(c.in), &m); err != nil {
			t.Fatalf("%s: %v", c.in, err)
		}
		if m.Content != c.want {
			t.Errorf("%s: got %q want %q", c.in, m.Content, c.want)
		}
	}
}
