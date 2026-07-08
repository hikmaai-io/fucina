package server

import (
	"encoding/json"
	"github.com/hikmaai-io/fucina/internal/chat"
	"strings"
	"testing"
)

func TestRenderToolDeclarations(t *testing.T) {
	tools := []Tool{{
		Type: "function",
		Function: ToolFunction{
			Name:        "get_weather",
			Description: "Get the weather for a city",
			Parameters: json.RawMessage(`{
				"type":"object",
				"properties":{
					"city":{"type":"string","description":"City name"},
					"unit":{"type":"string","enum":["celsius","fahrenheit"]}
				},
				"required":["city"]
			}`),
		},
	}}
	got := chat.RenderToolDeclarations(tools)
	for _, want := range []string{
		"<|tool>declaration:get_weather{",
		`description:<|"|>Get the weather for a city<|"|>`,
		`city:{description:<|"|>City name<|"|>,type:<|"|>STRING<|"|>}`,
		`required:[<|"|>city<|"|>]`,
		`type:<|"|>OBJECT<|"|>`,
		"<tool|>",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("declaration missing %q\n got: %s", want, got)
		}
	}
}

func TestParseToolCalls(t *testing.T) {
	// Simulated raw model output (markers preserved, as DecodeRaw would produce).
	raw := "<|channel>thought\n<channel|>Let me check the weather." +
		`<|tool_call>call:get_weather{city: <|"|>Paris<|"|>, unit: <|"|>celsius<|"|>, days: 3}<tool_call|>`
	content, calls := chat.ParseToolCalls(raw)
	if content != "Let me check the weather." {
		t.Errorf("content = %q, want %q", content, "Let me check the weather.")
	}
	if len(calls) != 1 {
		t.Fatalf("got %d calls, want 1", len(calls))
	}
	c := calls[0]
	if c.Function.Name != "get_weather" {
		t.Errorf("name = %q", c.Function.Name)
	}
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(c.Function.Arguments), &args); err != nil {
		t.Fatalf("arguments not valid JSON: %v (%s)", err, c.Function.Arguments)
	}
	if args["city"] != "Paris" || args["unit"] != "celsius" {
		t.Errorf("args = %v", args)
	}
	if d, ok := args["days"].(float64); !ok || d != 3 {
		t.Errorf("days = %v (%T)", args["days"], args["days"])
	}
}

func TestParseMultipleToolCalls(t *testing.T) {
	raw := `Sure.<|tool_call>call:a{x: 1}<tool_call|><|tool_call>call:b{y: <|"|>hi<|"|>}<tool_call|>`
	content, calls := chat.ParseToolCalls(raw)
	if content != "Sure." {
		t.Errorf("content = %q", content)
	}
	if len(calls) != 2 || calls[0].Function.Name != "a" || calls[1].Function.Name != "b" {
		t.Fatalf("calls = %+v", calls)
	}
}

func TestToolCallRoundTrip(t *testing.T) {
	// encode assistant tool_calls → parse back.
	calls := []ToolCall{{
		Type:     "function",
		Function: ToolCallFunction{Name: "search", Arguments: `{"q":"golang","limit":5}`},
	}}
	rendered := chat.RenderAssistantToolCalls(calls)
	_, got := chat.ParseToolCalls(rendered)
	if len(got) != 1 || got[0].Function.Name != "search" {
		t.Fatalf("round-trip calls = %+v", got)
	}
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(got[0].Function.Arguments), &args); err != nil {
		t.Fatalf("unmarshal args: %v", err)
	}
	if args["q"] != "golang" {
		t.Errorf("q = %v", args["q"])
	}
	if l, _ := args["limit"].(float64); l != 5 {
		t.Errorf("limit = %v", args["limit"])
	}
}

func TestToolResponseRender(t *testing.T) {
	got := chat.RenderToolResponse("get_weather", "18C sunny")
	want := `<|tool_response>response:get_weather{value:<|"|>18C sunny<|"|>}<tool_response|>`
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}
