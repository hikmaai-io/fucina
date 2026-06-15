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
func splitReasoning(s string) (string, string)       { return chat.SplitReasoning(s) }
func stripMarkers(s string) string                   { return chat.StripMarkers(s) }
