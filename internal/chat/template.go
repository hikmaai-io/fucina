// Package chat renders the gemma-4 chat template shared by the OpenAI-style
// HTTP server and the interactive CLI REPL.
//
// gemma-4 does NOT use <start_of_turn>/<end_of_turn> (those are not real vocab
// tokens). Turns are delimited by <|turn> / <turn|>, and the assistant's
// reasoning lives in a <|channel>thought…<channel|> block. The template emitted
// here is:
//
//	<|turn>system\n[<|think|>\n]SYS_CONTENT[SYS_EXTRA]<turn|>\n   (system turn)
//	<|turn>user\nUSER_CONTENT<turn|>\n
//	<|turn>model\nASSISTANT_CONTENT[TURN_EXTRA]<turn|>\n
//	…
//	<|turn>model\n[<|channel>thought\n<channel|>]                 (trailing open)
//
// Thinking control is binary. When thinking is OFF the model turn is opened with
// an already-closed empty thought channel (ModelTurnOpenNoThink) so the model
// answers directly; when ON the turn is opened bare (ModelTurnOpenThink) and the
// system turn carries a leading <|think|> marker so the model emits its own
// reasoning channel first.
//
// The system turn is FORCED (emitted even without an explicit system message)
// whenever there is system content, caller-supplied SystemExtra, or thinking is
// enabled. The final message decides the trailing open: a trailing non-assistant
// message (or a trailing EMPTY assistant message) opens a fresh model turn for
// generation; a trailing assistant message WITH content is closed normally.
//
// Tool-specific syntax (declarations, tool_calls, tool_responses) is NOT owned by
// this package — it is injected by the caller through SystemExtra and the
// TurnExtra / ToolResponse hooks, keeping all tool logic in the server package.
package chat

import (
	"fmt"
	"strings"
)

// Message is the minimal unit the template needs: a role and its text content.
// Tool-specific data (names, tool_calls) stays in the caller, which supplies it
// through the Renderer hooks keyed by message index.
type Message struct {
	Role    string
	Content string
}

// ModelTurnOpenNoThink opens an assistant turn and pre-fills an already-closed
// empty thought channel (thinking OFF) so the model skips reasoning and answers
// directly.
const ModelTurnOpenNoThink = "<|turn>model\n<|channel>thought\n<channel|>"

// ModelTurnOpenThink opens an assistant turn WITHOUT closing the thought channel
// (thinking ON) so the model produces its own <|channel>thought…<channel|>
// reasoning before the answer.
const ModelTurnOpenThink = "<|turn>model\n"

// Renderer renders []Message into the gemma-4 prompt. The hooks let the caller
// inject tool-specific text without this package knowing about tools:
//
//   - SystemExtra is appended inside the (forced) system turn after the system
//     content (the server passes its rendered tool declarations here; the CLI
//     passes "").
//   - TurnExtra(i) returns extra text appended inside message i's model turn,
//     before the closing <turn|> (the server uses it for assistant tool_calls;
//     the CLI passes nil).
//   - ToolResponse(i) fully renders a role:"tool" message (the server wraps the
//     tool output in <|tool_response>… here). When nil, a tool message falls
//     back to emitting its raw Content.
type Renderer struct {
	EnableThinking bool
	SystemExtra    string
	TurnExtra      func(i int) string
	ToolResponse   func(i int) string
}

// modelTurnOpen returns the correct model-turn opener for the thinking mode.
func (r Renderer) modelTurnOpen() string {
	if r.EnableThinking {
		return ModelTurnOpenThink
	}
	return ModelTurnOpenNoThink
}

// turnExtra safely invokes the TurnExtra hook.
func (r Renderer) turnExtra(i int) string {
	if r.TurnExtra == nil {
		return ""
	}
	return r.TurnExtra(i)
}

// Render builds the full gemma-4 prompt from messages.
func (r Renderer) Render(messages []Message) string {
	var sb strings.Builder
	open := r.modelTurnOpen()

	// System turn: merge an explicit leading system message (if any) with the
	// caller's SystemExtra (e.g. tool declarations). Forced when there is system
	// content, SystemExtra, or thinking is enabled.
	sysContent := ""
	start := 0
	if len(messages) > 0 && messages[0].Role == "system" {
		sysContent = messages[0].Content
		start = 1
	}
	if sysContent != "" || r.SystemExtra != "" || r.EnableThinking {
		sb.WriteString("<|turn>system\n")
		if r.EnableThinking {
			sb.WriteString("<|think|>\n")
		}
		sb.WriteString(sysContent)
		sb.WriteString(r.SystemExtra)
		sb.WriteString("<turn|>\n")
	}

	for i := start; i < len(messages); i++ {
		msg := messages[i]
		switch msg.Role {
		case "system":
			fmt.Fprintf(&sb, "<|turn>system\n%s<turn|>\n", msg.Content)
		case "user":
			fmt.Fprintf(&sb, "<|turn>user\n%s<turn|>\n", msg.Content)
		case "tool":
			if r.ToolResponse != nil {
				sb.WriteString(r.ToolResponse(i))
			} else {
				sb.WriteString(msg.Content)
			}
		case "assistant":
			extra := r.turnExtra(i)
			if i == len(messages)-1 && extra == "" && msg.Content == "" {
				// Trailing empty assistant turn → open a fresh model turn for
				// generation instead of closing it.
				sb.WriteString(open)
			} else {
				sb.WriteString("<|turn>model\n")
				sb.WriteString(msg.Content)
				sb.WriteString(extra)
				sb.WriteString("<turn|>\n")
			}
		}
	}
	if len(messages) > 0 && messages[len(messages)-1].Role != "assistant" {
		sb.WriteString(open)
	}
	return sb.String()
}

// Render is the simple entry point used by callers without tool support (the
// CLI). sysExtra is appended inside the forced system turn; turnExtra(i) appends
// extra text inside message i's model turn (pass nil for none). For full control
// (e.g. role:"tool" rendering) construct a Renderer directly.
func Render(messages []Message, enableThinking bool, sysExtra string, turnExtra func(i int) string) string {
	return Renderer{
		EnableThinking: enableThinking,
		SystemExtra:    sysExtra,
		TurnExtra:      turnExtra,
	}.Render(messages)
}
