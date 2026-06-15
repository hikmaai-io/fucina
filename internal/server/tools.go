package server

// The gemma-4 tool types + render/parse logic now live in internal/chat so the AR server
// and the diffusion server share one implementation (no duplication). These aliases + thin
// wrappers keep the rest of this package's call sites unchanged.

import "github.com/hikmaai-io/fucina/internal/chat"

type (
	Tool             = chat.Tool
	ToolFunction     = chat.ToolFunction
	ToolCall         = chat.ToolCall
	ToolCallFunction = chat.ToolCallFunction
)

func renderToolDeclarations(t []Tool) string         { return chat.RenderToolDeclarations(t) }
func renderToolResponse(name, content string) string { return chat.RenderToolResponse(name, content) }
func renderAssistantToolCalls(c []ToolCall) string   { return chat.RenderAssistantToolCalls(c) }
func parseToolCalls(raw string) (string, []ToolCall) { return chat.ParseToolCalls(raw) }
func isToolChoiceNone(tc interface{}) bool           { return chat.IsToolChoiceNone(tc) }

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
func splitReasoning(s string) (string, string) { return chat.SplitReasoning(s) }
func stripMarkers(s string) string             { return chat.StripMarkers(s) }
