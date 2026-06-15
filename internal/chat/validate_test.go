package chat

import (
	"encoding/json"
	"testing"
)

func tool(name, params string) Tool {
	return Tool{Type: "function", Function: ToolFunction{Name: name, Parameters: json.RawMessage(params)}}
}

func call(name, args string) ToolCall {
	return ToolCall{Type: "function", Function: ToolCallFunction{Name: name, Arguments: args}}
}

func TestValidateToolCalls(t *testing.T) {
	webSearch := tool("web_search", `{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`)
	noReq := tool("get_time", `{"type":"object","properties":{"tz":{"type":"string"}}}`)
	calc := tool("add", `{"type":"object","properties":{"n":{"type":"number"}},"required":["n"]}`)
	tagger := tool("tag", `{"type":"object","properties":{"items":{"type":"array"}},"required":["items"]}`)
	tools := []Tool{webSearch, noReq, calc, tagger}

	tests := []struct {
		name      string
		calls     []ToolCall
		wantValid int
		wantDrop  bool
	}{
		{"empty required string dropped", []ToolCall{call("web_search", `{"query":""}`)}, 0, true},
		{"whitespace required string dropped", []ToolCall{call("web_search", `{"query":"   "}`)}, 0, true},
		{"missing required dropped", []ToolCall{call("web_search", `{}`)}, 0, true},
		{"empty array required dropped", []ToolCall{call("tag", `{"items":[]}`)}, 0, true},
		{"empty object required dropped", []ToolCall{call("tag", `{"items":{}}`)}, 0, true},
		{"non-empty array required kept", []ToolCall{call("tag", `{"items":["a"]}`)}, 1, false},
		{"valid query kept", []ToolCall{call("web_search", `{"query":"cats"}`)}, 1, false},
		{"no required params kept", []ToolCall{call("get_time", `{}`)}, 1, false},
		{"required number zero kept", []ToolCall{call("add", `{"n":0}`)}, 1, false},
		{"unknown tool passes through", []ToolCall{call("mystery", `{}`)}, 1, false},
		{"one valid one invalid", []ToolCall{call("web_search", `{"query":""}`), call("add", `{"n":2}`)}, 1, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			valid, dropped := ValidateToolCalls(tc.calls, tools)
			if len(valid) != tc.wantValid {
				t.Errorf("valid=%d want %d", len(valid), tc.wantValid)
			}
			if (len(dropped) > 0) != tc.wantDrop {
				t.Errorf("dropped=%d wantDrop=%v", len(dropped), tc.wantDrop)
			}
		})
	}
}

// The dropped-call reason must name the CORRECT missing param — it becomes the
// user-facing clarification string, so a wrong name misleads the client.
func TestValidateToolCallsReportsCorrectParam(t *testing.T) {
	webSearch := tool("web_search", `{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`)
	_, dropped := ValidateToolCalls([]ToolCall{call("web_search", `{}`)}, []Tool{webSearch})
	if len(dropped) != 1 {
		t.Fatalf("dropped=%d want 1", len(dropped))
	}
	if dropped[0].Param != "query" {
		t.Errorf("dropped param=%q want query", dropped[0].Param)
	}
}
