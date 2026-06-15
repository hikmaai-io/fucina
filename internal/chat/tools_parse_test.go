// ABOUTME: Adversarial/malformed-input tests for the tool-call parser, which
// ABOUTME: turns attacker-influenceable model output into dispatched tool calls.

package chat

import (
	"strings"
	"testing"
)

// The parser must never panic and must terminate on imperfect model output
// (token-limit truncation, hallucinated syntax). These feed malformed strings
// through the public entry and assert (content, calls) without crashing.
func TestParseToolCallsAdversarial(t *testing.T) {
	cases := []struct {
		name string
		in   string
	}{
		{"unterminated string", `<|tool_call>call:f{q: <|"|>unclosed`},
		{"unterminated json string", `<|tool_call>call:f{q: "unclosed`},
		{"open dict no close", `<|tool_call>call:f{`},
		{"open dict eof mid key", `<|tool_call>call:f{key_without_value`},
		{"double comma", `<|tool_call>call:f{a:1,,b:2}<tool_call|>`},
		{"key then close brace", `<|tool_call>call:f{a}<tool_call|>`},
		{"top-level array body", `<|tool_call>call:f[1,2,3]<tool_call|>`},
		{"array value", `<|tool_call>call:f{tags:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>`},
		{"nested array", `<|tool_call>call:f{m:[[1,2],[3]]}<tool_call|>`},
		{"unterminated array", `<|tool_call>call:f{tags:[1,2`},
		{"unicode in value", `<|tool_call>call:f{q:<|"|>café 日本語 😀<|"|>}<tool_call|>`},
		{"close without open", `call:f{x:1}<tool_call|>`},
		{"garbage between calls", `<|tool_call>call:f{a:1}<tool_call|> noise <|tool_call>call:g{b:2}<tool_call|>`},
		{"no call keyword", `<|tool_call>not a call<tool_call|>`},
		{"empty", ``},
		{"just markers", `<|tool_call><tool_call|>`},
		{"channel then call", `<|channel>thinking<channel|><|tool_call>call:f{x:1}<tool_call|>`},
		{"bool and null values", `<|tool_call>call:f{a:true,b:false,c:null}<tool_call|>`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Must not panic (defer recover surfaces it as a test failure).
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("ParseToolCalls panicked on %q: %v", tc.in, r)
				}
			}()
			content, calls := ParseToolCalls(tc.in)
			_ = content
			_ = calls
		})
	}
}

// Deep nesting must not overflow the stack (the recursive descent is otherwise
// unbounded: value→dict/array→value; ~2M brackets crash with a FATAL,
// unrecoverable stack overflow). The depth cap turns it into a graceful parse
// failure. 3M brackets would crash the process without the guard.
func TestParseToolCallsDeepNestingTerminates(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("deep nesting panicked: %v", r)
		}
	}()
	deep := "<|tool_call>call:f{x:" + strings.Repeat("[", 3_000_000) + "}<tool_call|>"
	_, _ = ParseToolCalls(deep) // must return, not overflow the stack
}

// A well-formed array value must round-trip (parseGemmaArray was previously 0%
// covered).
func TestParseToolCallsArrayValue(t *testing.T) {
	_, calls := ParseToolCalls(`<|tool_call>call:search{tags:[<|"|>go<|"|>,<|"|>rust<|"|>]}<tool_call|>`)
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	if !strings.Contains(calls[0].Function.Arguments, "go") ||
		!strings.Contains(calls[0].Function.Arguments, "rust") {
		t.Errorf("array args not parsed: %s", calls[0].Function.Arguments)
	}
}

// SplitReasoning with an unterminated channel (token limit hit mid-thought).
func TestSplitReasoningUnterminated(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("SplitReasoning panicked: %v", r)
		}
	}()
	reasoning, rest := SplitReasoning("<|channel>still thinking with no close")
	_ = reasoning
	_ = rest
}
