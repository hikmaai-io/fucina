package chat

// Dialect abstracts the per-model-family chat wire format: how messages, tool
// declarations, tool calls and reasoning render into the prompt, and how the
// model's raw decoded output parses back into content / reasoning / tool_calls.
// The server selects the dialect at startup from the loaded vocabulary
// (ChatML marker present → Qwen, else Gemma) — never from a flag.
type Dialect interface {
	Name() string

	// Render builds the full prompt (tool declarations included) from the
	// conversation. enableThinking gates the dialect's reasoning block.
	Render(msgs []RichMessage, tools []Tool, enableThinking bool) string

	// ForcedCallPrefix returns the text appended to the rendered prompt to
	// force a tool call (OpenAI tool_choice). fnName=="" means "required"
	// (any function). Empty return = forcing unsupported by this dialect.
	ForcedCallPrefix(fnName string) string

	// ParseToolCalls extracts content + tool calls from RAW decoded output
	// (control markers intact). tools, when non-nil, provides the parameter
	// schemas used to coerce argument types.
	ParseToolCalls(raw string, tools []Tool) (content string, calls []ToolCall)

	// SplitReasoning separates the reasoning block from the answer in RAW
	// decoded output. thinking reports whether the request enabled reasoning —
	// dialects whose reasoning opener lives in the PROMPT (Qwen's <think>\n)
	// need it to classify an output with no reasoning markers at all.
	SplitReasoning(raw string, thinking bool) (reasoning, rest string)

	// StripMarkers removes this dialect's control-marker literals from text.
	StripMarkers(s string) string

	// ToolCallLits returns the open/close tool-call literals as they appear in
	// raw decoded text (used by truncation guards around unterminated calls).
	ToolCallLits() (open, close string)

	// StartsInReasoning reports whether generation begins already inside the
	// reasoning block (Qwen renders the <think>\n opener into the prompt, so
	// the model never emits an open marker; Gemma emits its own <|channel>).
	StartsInReasoning(thinking bool) bool

	// HasReasoningLabel reports whether the reasoning block opens with a label
	// line the streamer must skip (Gemma's "thought\n"); Qwen has none.
	HasReasoningLabel() bool
}

// RichMessage is a full-fidelity chat message: what Dialect.Render needs to
// reproduce a turn byte-exactly (role, content, reasoning echo, tool calls,
// and the tool name for role:"tool" results).
type RichMessage struct {
	Role      string
	Content   string
	Reasoning string
	Name      string
	ToolCalls []ToolCall
}

// ─── Gemma dialect ─────────────────────────────────────────────────────
//
// Thin adapter over the pre-existing package-level gemma-4 renderer/parsers,
// wired exactly the way internal/server used them before dialects existed, so
// rendered prompts stay byte-identical (KV prefix-cache compatibility).

type gemmaDialect struct{}

// Gemma is the gemma-4 chat dialect (<|turn>/<|channel>/<|tool_call> markers).
var Gemma Dialect = gemmaDialect{}

func (gemmaDialect) Name() string { return "gemma" }

func (gemmaDialect) Render(msgs []RichMessage, tools []Tool, enableThinking bool) string {
	plain := make([]Message, len(msgs))
	for i, m := range msgs {
		plain[i] = Message{Role: m.Role, Content: m.Content, Reasoning: m.Reasoning}
	}
	sysExtra := ""
	if len(tools) > 0 {
		sysExtra = RenderToolDeclarations(tools)
	}
	r := Renderer{
		EnableThinking: enableThinking,
		SystemExtra:    sysExtra,
		TurnExtra: func(i int) string {
			if len(msgs[i].ToolCalls) > 0 {
				return RenderAssistantToolCalls(msgs[i].ToolCalls)
			}
			return ""
		},
		ToolResponse: func(i int) string {
			return RenderToolResponse(msgs[i].Name, msgs[i].Content)
		},
	}
	return r.Render(plain)
}

func (gemmaDialect) ForcedCallPrefix(string) string { return "" }

func (gemmaDialect) ParseToolCalls(raw string, _ []Tool) (string, []ToolCall) {
	return ParseToolCalls(raw)
}

func (gemmaDialect) SplitReasoning(raw string, _ bool) (string, string) {
	return SplitReasoning(raw)
}

func (gemmaDialect) StripMarkers(s string) string { return StripMarkers(s) }

func (gemmaDialect) ToolCallLits() (string, string) { return "<|tool_call>", "<tool_call|>" }

func (gemmaDialect) StartsInReasoning(bool) bool { return false }

func (gemmaDialect) HasReasoningLabel() bool { return true }
