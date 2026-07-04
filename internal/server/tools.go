package server

// Tool render/parse logic lives in internal/chat behind the chat.Dialect
// interface (gemma-4 turn/tool markers vs Qwen ChatML + XML tool calls). The
// server picks s.dialect at startup from the loaded vocabulary; these thin
// wrappers keep call sites terse.

import "github.com/hikmaai-io/fucina/internal/chat"

type (
	Tool             = chat.Tool
	ToolFunction     = chat.ToolFunction
	ToolCall         = chat.ToolCall
	ToolCallFunction = chat.ToolCallFunction
)

func isToolChoiceNone(tc interface{}) bool { return chat.IsToolChoiceNone(tc) }

// forcedToolChoice interprets the OpenAI tool_choice forms that FORCE a call:
// "required" (any function) and {"type":"function","function":{"name":X}}.
// Returns the forced function name ("" for "required") and whether forcing is
// requested at all. "auto"/"none"/absent return forced=false.
func forcedToolChoice(tc interface{}) (name string, forced bool) {
	switch v := tc.(type) {
	case string:
		return "", v == "required"
	case map[string]interface{}:
		fn, _ := v["function"].(map[string]interface{})
		if fn == nil {
			return "", false
		}
		n, _ := fn["name"].(string)
		return n, n != ""
	}
	return "", false
}

func (s *Server) parseToolCalls(raw string, tools []Tool) (string, []ToolCall) {
	return s.dialect.ParseToolCalls(raw, tools)
}
func (s *Server) splitReasoning(raw string, thinking bool) (string, string) {
	return s.dialect.SplitReasoning(raw, thinking)
}
func (s *Server) stripMarkers(x string) string { return s.dialect.StripMarkers(x) }

// validateToolCalls drops calls that violate their tool's required-parameter
// schema (missing or empty required arg) and returns a clarification string when
// every call was dropped, so the turn answers instead of dispatching a malformed
// call. clar is "" when at least one call is valid (or nothing was dropped).
func validateToolCalls(calls []ToolCall, tools []Tool) (valid []ToolCall, clar string) {
	valid, dropped := chat.ValidateToolCalls(calls, tools)
	if len(valid) == 0 && len(dropped) > 0 {
		d := dropped[0]
		clar = "I can't call " + d.Name + " without the required parameter \"" + d.Param +
			"\". Please provide it."
	}
	return valid, clar
}
