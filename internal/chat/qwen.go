package chat

// Qwen3.5 / Qwen3 ChatML dialect. The wire format is taken from the
// chat_template.jinja shipped inside the Qwen3.5-35B-A3B checkpoint (which is
// the authority — NOT classic Hermes JSON calls):
//
//	<|im_start|>system
//	[# Tools …<tools>{tool json}…</tools> + fixed instructions]
//	[system content]<|im_end|>
//	<|im_start|>user
//	CONTENT<|im_end|>
//	<|im_start|>assistant
//	[<think>REASONING</think>]CONTENT[<tool_call>…</tool_call>]<|im_end|>
//	<|im_start|>user            (tool results are grouped into a user turn)
//	<tool_response>
//	RESULT
//	</tool_response><|im_end|>
//	…
//	<|im_start|>assistant
//	<think>\n                    (thinking ON — model closes with </think>)
//	<think>\n\n</think>\n\n      (thinking OFF — pre-closed empty block)
//
// Tool calls use the Qwen3-Coder XML form (the <function=/<parameter= literals
// are plain text, only <tool_call>/</tool_call> are vocab tokens):
//
//	<tool_call>
//	<function=NAME>
//	<parameter=KEY>
//	VALUE
//	</parameter>
//	</function>
//	</tool_call>
//
// Reasoning retention follows the template: only assistant turns AFTER the
// last real user query (the current tool loop) re-render their <think> block;
// earlier turns render content only.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

type qwenDialect struct{}

// Qwen is the ChatML dialect used by the Qwen3 / Qwen3.5 families.
var Qwen Dialect = qwenDialect{}

func (qwenDialect) Name() string { return "qwen" }

// qwenToolInstructions is the fixed tool-protocol blurb from the checkpoint's
// chat_template.jinja, byte-for-byte.
const qwenToolInstructions = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
	"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
	"<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n" +
	"<IMPORTANT>\nReminder:\n" +
	"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
	"- Required parameters MUST be specified\n" +
	"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
	"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
	"</IMPORTANT>"

// qwenControlMarkers are the ChatML/Qwen special-token literals neutralized in
// untrusted (system/user/tool) content — same role-confusion defense as the
// gemma controlMarkers list. Assistant content is NOT sanitized (byte-exact
// re-render for KV prefix reuse).
var qwenControlMarkers = []string{
	"<|im_start|>", "<|im_end|>", "<|endoftext|>",
	"<think>", "</think>", "<tool_call>", "</tool_call>",
	"<tool_response>", "</tool_response>",
	"<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
	"<|image_pad|>", "<|video_pad|>", "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>",
	"<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>",
	"<|quad_start|>", "<|quad_end|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
}

func qwenSanitize(s string) string {
	if !strings.Contains(s, "<") {
		return s
	}
	for _, m := range qwenControlMarkers {
		if strings.Contains(s, m) {
			s = strings.ReplaceAll(s, m, "<"+zeroWidthSpace+m[1:])
		}
	}
	return s
}

// qwenToolJSON serializes one tool declaration the way the HF template's
// `tool | tojson` does: compact JSON, no HTML escaping. The client's
// parameters schema is embedded verbatim (compacted), preserving key order.
func qwenToolJSON(t Tool) string {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(t); err != nil {
		return "{}"
	}
	return strings.TrimSuffix(buf.String(), "\n")
}

func (qwenDialect) Render(msgs []RichMessage, tools []Tool, enableThinking bool) string {
	var sb strings.Builder

	// System turn: tools block (with optional system content appended) or the
	// bare system message. Only a LEADING system message is honored, matching
	// the template's must-be-first rule.
	sysContent := ""
	start := 0
	if len(msgs) > 0 && msgs[0].Role == "system" {
		sysContent = strings.TrimSpace(qwenSanitize(msgs[0].Content))
		start = 1
	}
	if len(tools) > 0 {
		sb.WriteString("<|im_start|>system\n")
		sb.WriteString("# Tools\n\nYou have access to the following functions:\n\n<tools>")
		for _, t := range tools {
			sb.WriteString("\n")
			sb.WriteString(qwenToolJSON(t))
		}
		sb.WriteString("\n</tools>")
		sb.WriteString(qwenToolInstructions)
		if sysContent != "" {
			sb.WriteString("\n\n")
			sb.WriteString(sysContent)
		}
		sb.WriteString("<|im_end|>\n")
	} else if sysContent != "" {
		sb.WriteString("<|im_start|>system\n")
		sb.WriteString(sysContent)
		sb.WriteString("<|im_end|>\n")
	}

	// lastQuery = index of the last REAL user query (not a pre-wrapped
	// <tool_response> forwarded as a user message). Assistant turns after it
	// (the live tool loop) re-render their reasoning; earlier ones drop it.
	lastQuery := len(msgs) - 1
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != "user" {
			continue
		}
		c := strings.TrimSpace(msgs[i].Content)
		if !(strings.HasPrefix(c, "<tool_response>") && strings.HasSuffix(c, "</tool_response>")) {
			lastQuery = i
			break
		}
	}

	for i := start; i < len(msgs); i++ {
		msg := msgs[i]
		switch msg.Role {
		case "system":
			// Non-leading system messages have no slot in the Qwen template;
			// render as a user turn so the content is not silently dropped.
			fmt.Fprintf(&sb, "<|im_start|>user\n%s<|im_end|>\n", strings.TrimSpace(qwenSanitize(msg.Content)))
		case "user":
			fmt.Fprintf(&sb, "<|im_start|>user\n%s<|im_end|>\n", strings.TrimSpace(qwenSanitize(msg.Content)))
		case "tool":
			// Tool results group into ONE user turn: open only when the
			// previous message was not a tool result, close only when the next
			// one is not.
			if i == 0 || msgs[i-1].Role != "tool" {
				sb.WriteString("<|im_start|>user")
			}
			sb.WriteString("\n<tool_response>\n")
			sb.WriteString(strings.TrimSpace(qwenSanitize(msg.Content)))
			sb.WriteString("\n</tool_response>")
			if i == len(msgs)-1 || msgs[i+1].Role != "tool" {
				sb.WriteString("<|im_end|>\n")
			}
		case "assistant":
			if i == len(msgs)-1 && msg.Content == "" && len(msg.ToolCalls) == 0 {
				break // trailing empty assistant → generation prompt below
			}
			content := strings.TrimSpace(msg.Content)
			reasoning := msg.Reasoning
			if reasoning == "" {
				// Inline <think> block in content (client echoed raw text).
				if idx := strings.Index(content, "</think>"); idx >= 0 {
					head := strings.TrimRight(content[:idx], "\n")
					if o := strings.LastIndex(head, "<think>"); o >= 0 {
						head = head[o+len("<think>"):]
					}
					reasoning = strings.TrimLeft(head, "\n")
					content = strings.TrimLeft(content[idx+len("</think>"):], "\n")
				}
			}
			if i > lastQuery {
				sb.WriteString("<|im_start|>assistant\n<think>\n")
				sb.WriteString(strings.TrimSpace(reasoning))
				sb.WriteString("\n</think>\n\n")
				sb.WriteString(content)
			} else {
				sb.WriteString("<|im_start|>assistant\n")
				sb.WriteString(content)
			}
			for j, tc := range msg.ToolCalls {
				if j == 0 && strings.TrimSpace(content) == "" {
					sb.WriteString("<tool_call>\n<function=")
				} else if j == 0 {
					sb.WriteString("\n\n<tool_call>\n<function=")
				} else {
					sb.WriteString("\n<tool_call>\n<function=")
				}
				sb.WriteString(tc.Function.Name)
				sb.WriteString(">\n")
				for _, kv := range orderedArgs(tc.Function.Arguments) {
					sb.WriteString("<parameter=")
					sb.WriteString(kv.key)
					sb.WriteString(">\n")
					sb.WriteString(kv.text)
					sb.WriteString("\n</parameter>\n")
				}
				sb.WriteString("</function>\n</tool_call>")
			}
			sb.WriteString("<|im_end|>\n")
		}
	}

	// Generation prompt: open an assistant turn unless the conversation ended
	// with a completed assistant message.
	last := len(msgs) - 1
	if last < 0 || msgs[last].Role != "assistant" ||
		(msgs[last].Content == "" && len(msgs[last].ToolCalls) == 0) {
		sb.WriteString("<|im_start|>assistant\n")
		if enableThinking {
			sb.WriteString("<think>\n")
		} else {
			sb.WriteString("<think>\n\n</think>\n\n")
		}
	}
	return sb.String()
}

// argKV is one tool-call argument rendered to its <parameter> body text.
type argKV struct {
	key  string
	text string
}

// orderedArgs decodes a JSON arguments object PRESERVING key order (Go maps
// randomize it; the re-rendered prompt must token-match what the model
// emitted) and renders each value the way the template does: objects/arrays
// as compact JSON, strings verbatim, numbers as their literal text, booleans
// and null in Python string form ("True"/"False"/"None" — the template runs
// `value | string` through jinja, and that is what the model was trained on).
func orderedArgs(rawArgs string) []argKV {
	dec := json.NewDecoder(strings.NewReader(rawArgs))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil || tok != json.Delim('{') {
		return nil
	}
	var out []argKV
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return out
		}
		key, _ := kt.(string)
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			return out
		}
		out = append(out, argKV{key: key, text: qwenArgText(raw)})
	}
	return out
}

// qwenArgText renders one decoded JSON value into its <parameter> body text.
func qwenArgText(raw json.RawMessage) string {
	s := strings.TrimSpace(string(raw))
	if s == "" {
		return ""
	}
	switch s[0] {
	case '"':
		var str string
		if json.Unmarshal(raw, &str) == nil {
			return str
		}
		return s
	case '{', '[':
		var buf bytes.Buffer
		if json.Compact(&buf, raw) == nil {
			return buf.String()
		}
		return s
	}
	switch s {
	case "true":
		return "True"
	case "false":
		return "False"
	case "null":
		return "None"
	}
	return s // number: keep the literal text ("3" stays "3", not "3.0")
}

func (qwenDialect) ForcedCallPrefix(fnName string) string {
	if fnName == "" {
		return "<tool_call>\n<function=" // tool_choice:"required" — model picks the name
	}
	return "<tool_call>\n<function=" + fnName + ">\n"
}

// ─── Output side ───────────────────────────────────────────────────────

func (qwenDialect) SplitReasoning(raw string, thinking bool) (string, string) {
	if idx := strings.Index(raw, "</think>"); idx >= 0 {
		reasoning := raw[:idx]
		// The opener normally lives in the prompt; strip a re-emitted one.
		if o := strings.LastIndex(reasoning, "<think>"); o >= 0 {
			reasoning = reasoning[o+len("<think>"):]
		}
		rest := strings.TrimLeft(raw[idx+len("</think>"):], "\n")
		return strings.TrimLeft(reasoning, "\n"), rest
	}
	if thinking {
		// Thinking was on (the prompt opened <think>) and the block never
		// closed — the whole truncated output is reasoning.
		return raw, ""
	}
	return "", raw
}

func (qwenDialect) ParseToolCalls(raw string, tools []Tool) (string, []ToolCall) {
	schemas := qwenParamSchemas(tools)
	var calls []ToolCall
	var contentB strings.Builder
	for {
		o := strings.Index(raw, "<tool_call>")
		if o < 0 {
			contentB.WriteString(raw)
			break
		}
		contentB.WriteString(raw[:o])
		body := raw[o+len("<tool_call>"):]
		if e := strings.Index(body, "</tool_call>"); e >= 0 {
			raw = body[e+len("</tool_call>"):]
			body = body[:e]
		} else {
			// Unterminated final block: recover only a complete inner
			// <function=…></function>; else the text is a truncated call
			// and must not be dispatched or leak into content.
			raw = ""
			if !strings.Contains(body, "</function>") {
				break
			}
		}
		if tc, ok := parseQwenCall(body, schemas); ok {
			tc.ID = fmt.Sprintf("call_%s_%d", tc.Function.Name, len(calls))
			calls = append(calls, tc)
		}
	}
	return strings.TrimSpace(contentB.String()), calls
}

// qwenParamSchemas maps function name → parameter name → declared JSON-Schema
// type, for argument type coercion.
func qwenParamSchemas(tools []Tool) map[string]map[string]string {
	m := make(map[string]map[string]string, len(tools))
	for _, t := range tools {
		if t.Function.Name == "" || len(t.Function.Parameters) == 0 {
			continue
		}
		var p struct {
			Properties map[string]struct {
				Type string `json:"type"`
			} `json:"properties"`
		}
		if json.Unmarshal(t.Function.Parameters, &p) != nil {
			continue
		}
		pm := make(map[string]string, len(p.Properties))
		for name, prop := range p.Properties {
			pm[name] = prop.Type
		}
		m[t.Function.Name] = pm
	}
	return m
}

// parseQwenCall parses one <tool_call> body: <function=NAME> then repeated
// <parameter=KEY>\nVALUE\n</parameter>, closed by </function>.
func parseQwenCall(body string, schemas map[string]map[string]string) (ToolCall, bool) {
	f := strings.Index(body, "<function=")
	if f < 0 {
		return ToolCall{}, false
	}
	rest := body[f+len("<function="):]
	nameEnd := strings.IndexAny(rest, ">\n")
	if nameEnd < 0 {
		return ToolCall{}, false
	}
	name := strings.TrimSpace(rest[:nameEnd])
	if name == "" {
		return ToolCall{}, false
	}
	rest = rest[nameEnd:]
	if rest != "" && rest[0] == '>' {
		rest = rest[1:]
	}
	if fe := strings.LastIndex(rest, "</function>"); fe >= 0 {
		rest = rest[:fe]
	}

	// Arguments JSON is built by hand to preserve parameter ORDER (a re-render
	// of this call must token-match the model's emission).
	var args bytes.Buffer
	args.WriteByte('{')
	n := 0
	for {
		p := strings.Index(rest, "<parameter=")
		if p < 0 {
			break
		}
		rest = rest[p+len("<parameter="):]
		ke := strings.Index(rest, ">")
		if ke < 0 {
			break
		}
		key := strings.TrimSpace(rest[:ke])
		rest = rest[ke+1:]
		var val string
		if ve := strings.Index(rest, "</parameter>"); ve >= 0 {
			val = rest[:ve]
			rest = rest[ve+len("</parameter>"):]
		} else {
			val = rest // truncated last parameter: take what's there
			rest = ""
		}
		// The template wraps values in single newlines: <parameter=k>\nVALUE\n</parameter>.
		val = strings.TrimPrefix(val, "\n")
		val = strings.TrimSuffix(val, "\n")
		if n > 0 {
			args.WriteByte(',')
		}
		n++
		keyJSON, _ := json.Marshal(key)
		args.Write(keyJSON)
		args.WriteByte(':')
		args.WriteString(qwenCoerceArg(val, schemas[name][key]))
	}
	args.WriteByte('}')
	return ToolCall{
		ID:   "call_" + name,
		Type: "function",
		Function: ToolCallFunction{
			Name:      name,
			Arguments: args.String(),
		},
	}, true
}

// qwenCoerceArg converts a raw <parameter> body into a JSON value using the
// declared schema type; without a schema entry it falls back to a conservative
// guess (object/array/bool/null literals, else string).
func qwenCoerceArg(val, typ string) string {
	jsonStr := func() string {
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetEscapeHTML(false)
		enc.Encode(val)
		return strings.TrimSuffix(buf.String(), "\n")
	}
	switch typ {
	case "string":
		return jsonStr()
	case "integer":
		if _, err := strconv.ParseInt(strings.TrimSpace(val), 10, 64); err == nil {
			return strings.TrimSpace(val)
		}
		if f, err := strconv.ParseFloat(strings.TrimSpace(val), 64); err == nil {
			return strconv.FormatInt(int64(f), 10)
		}
		return jsonStr()
	case "number":
		if _, err := strconv.ParseFloat(strings.TrimSpace(val), 64); err == nil {
			return strings.TrimSpace(val)
		}
		return jsonStr()
	case "boolean":
		switch strings.ToLower(strings.TrimSpace(val)) {
		case "true":
			return "true"
		case "false":
			return "false"
		}
		return jsonStr()
	case "object", "array":
		if json.Valid([]byte(val)) {
			var buf bytes.Buffer
			if json.Compact(&buf, []byte(val)) == nil {
				return buf.String()
			}
		}
		return jsonStr()
	}
	// No schema: accept structured literals, keep everything else as string.
	t := strings.TrimSpace(val)
	if len(t) > 0 && (t[0] == '{' || t[0] == '[') && json.Valid([]byte(t)) {
		var buf bytes.Buffer
		if json.Compact(&buf, []byte(t)) == nil {
			return buf.String()
		}
	}
	switch strings.ToLower(t) {
	case "true":
		return "true"
	case "false":
		return "false"
	case "null", "none":
		return "null"
	}
	return jsonStr()
}

var qwenMarkerStripper = strings.NewReplacer(
	"<|im_start|>", "", "<|im_end|>", "", "<|endoftext|>", "",
	"<think>", "", "</think>", "",
	"<tool_call>", "", "</tool_call>", "",
	"<tool_response>", "", "</tool_response>", "",
)

func (qwenDialect) StripMarkers(s string) string { return qwenMarkerStripper.Replace(s) }

func (qwenDialect) ToolCallLits() (string, string) { return "<tool_call>", "</tool_call>" }

func (qwenDialect) StartsInReasoning(thinking bool) bool { return thinking }

func (qwenDialect) HasReasoningLabel() bool { return false }
