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
	// Reasoning is the assistant turn's thought-channel payload (the client's
	// reasoning_content echo). Historical assistant turns are re-rendered WITH
	// their channel block — empty when Reasoning is "" — so the rendered prompt
	// token-matches what generation actually committed to the KV cache. Without
	// it, every multi-turn request diverged at the channel opener of the last
	// assistant turn and re-prefilled the whole turn (measured: a constant
	// ~263-token re-prefill per agent tool-loop iteration).
	Reasoning string
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

// controlMarkers are the gemma-4 turn/channel/tool literals that only the SERVER
// may emit. If they appear verbatim in caller-supplied content they tokenize to
// the real control ids, letting a user/tool message spoof a turn boundary or
// fabricate a tool call (role confusion). sanitizeContent neutralizes them in
// untrusted content by inserting a zero-width space after the leading '<' so the
// literal no longer matches; assistant turns are NOT sanitized (their content is
// the model's own output and must stay byte-exact for KV prefix-cache reuse).
var controlMarkers = []string{
	"<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|think|>",
	"<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
	"<|tool_response>", "<tool_response|>", `<|"|>`,
}

// zeroWidthSpace breaks a control-marker literal when inserted after its leading
// '<' (so it no longer tokenizes to the control id) while staying visually
// invisible if the content is ever displayed.
const zeroWidthSpace = "\u200b"

func sanitizeContent(s string) string {
	if !strings.Contains(s, "<") {
		return s // fast path: no marker can be present
	}
	for _, m := range controlMarkers {
		if strings.Contains(s, m) {
			s = strings.ReplaceAll(s, m, "<"+zeroWidthSpace+m[1:])
		}
	}
	return s
}

// Render builds the full gemma-4 prompt from messages. Untrusted content
// (system/user/tool) is run through sanitizeContent to prevent control-marker
// injection; assistant turns are reproduced verbatim for KV prefix-cache match.
func (r Renderer) Render(messages []Message) string {
	var sb strings.Builder
	open := r.modelTurnOpen()

	// System turn: merge an explicit leading system message (if any) with the
	// caller's SystemExtra (e.g. tool declarations). Forced when there is system
	// content, SystemExtra, or thinking is enabled.
	sysContent := ""
	start := 0
	if len(messages) > 0 && messages[0].Role == "system" {
		sysContent = sanitizeContent(messages[0].Content)
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
			fmt.Fprintf(&sb, "<|turn>system\n%s<turn|>\n", sanitizeContent(msg.Content))
		case "user":
			fmt.Fprintf(&sb, "<|turn>user\n%s<turn|>\n", sanitizeContent(msg.Content))
		case "tool":
			if r.ToolResponse != nil {
				sb.WriteString(r.ToolResponse(i))
			} else {
				sb.WriteString(sanitizeContent(msg.Content))
			}
		case "assistant":
			extra := r.turnExtra(i)
			if i == len(messages)-1 && extra == "" && msg.Content == "" {
				// Trailing empty assistant turn → open a fresh model turn for
				// generation instead of closing it.
				sb.WriteString(open)
			} else {
				// Historical model turn: re-render the thought channel the turn
				// was generated with. Thinking OFF commits the pre-closed empty
				// channel from ModelTurnOpenNoThink into the KV; thinking ON
				// commits the model's own reasoning. Reproducing it keeps the
				// rendered prompt token-identical to the cached KV sequence, so
				// the prefix cache survives the turn (see Message.Reasoning).
				sb.WriteString("<|turn>model\n<|channel>thought\n")
				sb.WriteString(msg.Reasoning)
				sb.WriteString("<channel|>")
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
