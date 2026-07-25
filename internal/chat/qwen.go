package chat

// Qwen3.5 / Qwen3.6 ChatML dialect. The wire format is taken from the
// chat_template.jinja shipped inside the Qwen3.5/3.6 dense and MoE checkpoints (which is
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
	"io"
	"math"
	"strconv"
	"strings"
)

type qwenDialect struct{}

// Qwen is the ChatML dialect used by the Qwen3 / Qwen3.5 / Qwen3.6 families.
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
				// DELIBERATE deviation from the checkpoint template (which
				// renders old assistant turns bare): keep the EMPTY think
				// block. A thinking-off generation committed exactly
				// "<think>\n\n</think>\n\n"+content to the KV (the pre-closed
				// opener was part of its prompt), so this re-render stays
				// byte-identical to the committed sequence and the per-
				// conversation state/prefix cache keeps matching across
				// turns. Thinking-ON old turns still break the prefix when
				// their reasoning is dropped — unavoidable (and identical to
				// the official template's behavior). The empty block is
				// distribution-neutral: it is precisely what the model sees
				// in every non-thinking exchange.
				sb.WriteString("<|im_start|>assistant\n<think>\n\n</think>\n\n")
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
		open, bodyStart := findQwenOpenTag(raw, 0, "tool_call")
		if open < 0 {
			contentB.WriteString(raw)
			break
		}
		contentB.WriteString(raw[:open])

		close, afterClose := findQwenCloseTag(raw, bodyStart, "tool_call")
		bodyEnd := len(raw)
		closed := close >= 0
		if closed {
			bodyEnd = close
		}
		body := raw[bodyStart:bodyEnd]
		tc, ok := parseQwenCall(body, schemas)
		if ok {
			tc.ID = fmt.Sprintf("call_%s_%d", tc.Function.Name, len(calls))
			calls = append(calls, tc)
		} else {
			// Parsing failures are output, not absence of output. Preserve the
			// complete original span so the HTTP layer can expose it as content;
			// exact control markers are stripped there, while the useful body is
			// retained. Incomplete calls are never dispatched.
			end := len(raw)
			if closed {
				end = afterClose
			}
			contentB.WriteString(raw[open:end])
		}
		if !closed {
			break
		}
		raw = raw[afterClose:]
	}
	return strings.TrimSpace(contentB.String()), calls
}

// findQwenOpenTag finds an XML-ish bare opening tag. Qwen emits exact tags, but
// accepting horizontal/newline whitespace before '>' makes parsing resilient to
// harmless formatting variants without treating arbitrary attributes as valid.
func findQwenOpenTag(s string, from int, element string) (start, after int) {
	needle := "<" + element
	for from <= len(s)-len(needle) {
		i := strings.Index(s[from:], needle)
		if i < 0 {
			break
		}
		i += from
		j := i + len(needle)
		for j < len(s) && isQwenSpace(s[j]) {
			j++
		}
		if j < len(s) && s[j] == '>' {
			return i, j + 1
		}
		from = i + 1
	}
	return -1, -1
}

func findQwenCloseTag(s string, from int, element string) (start, after int) {
	needle := "</" + element
	for from <= len(s)-len(needle) {
		i := strings.Index(s[from:], needle)
		if i < 0 {
			break
		}
		i += from
		j := i + len(needle)
		for j < len(s) && isQwenSpace(s[j]) {
			j++
		}
		if j < len(s) && s[j] == '>' {
			return i, j + 1
		}
		from = i + 1
	}
	return -1, -1
}

func isQwenSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
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

// parseQwenCall auto-detects the two Qwen dialects inside <tool_call>:
//
//	XML (Qwen3.6 template): <function=NAME><parameter=KEY>VALUE...</parameter></function>
//	JSON (legacy Qwen3/3.5): {"name":"NAME","arguments":{...}}
//
// Dialect selection comes only from emitted bytes; there is no model/quant flag.
func parseQwenCall(body string, schemas map[string]map[string]string) (ToolCall, bool) {
	if strings.HasPrefix(strings.TrimSpace(body), "{") {
		return parseQwenJSONCall(body)
	}
	return parseQwenXMLCall(body, schemas)
}

func parseQwenJSONCall(body string) (ToolCall, bool) {
	type jsonFunction struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	var wire struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
		Function  *jsonFunction   `json:"function"`
	}
	dec := json.NewDecoder(strings.NewReader(body))
	if err := dec.Decode(&wire); err != nil {
		return ToolCall{}, false
	}
	// Reject a valid object followed by model garbage inside the call span.
	var extra interface{}
	if dec.Decode(&extra) != io.EOF {
		return ToolCall{}, false
	}
	name, arguments := wire.Name, wire.Arguments
	if wire.Function != nil {
		name, arguments = wire.Function.Name, wire.Function.Arguments
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return ToolCall{}, false
	}
	args, ok := compactQwenJSONArguments(arguments)
	if !ok {
		return ToolCall{}, false
	}
	return ToolCall{
		ID:       "call_" + name,
		Type:     "function",
		Function: ToolCallFunction{Name: name, Arguments: args},
	}, true
}

func compactQwenJSONArguments(raw json.RawMessage) (string, bool) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return "{}", true
	}
	// Some OpenAI-compatible emitters encode arguments as a JSON string rather
	// than an object. Accept that legacy variant when the string itself is JSON.
	if bytes.HasPrefix(bytes.TrimSpace(raw), []byte{'"'}) {
		var encoded string
		if json.Unmarshal(raw, &encoded) != nil {
			return "", false
		}
		raw = json.RawMessage(encoded)
	}
	t := bytes.TrimSpace(raw)
	if len(t) == 0 || t[0] != '{' || !json.Valid(t) {
		return "", false
	}
	var compact bytes.Buffer
	if json.Compact(&compact, t) != nil {
		return "", false
	}
	return compact.String(), true
}

// parseQwenXMLCall parses one complete function block. Values are delimited only
// by the exact closing parameter tag, so ordinary comparisons, HTML, paths, and
// JSON containing '<' or '>' remain data rather than confusing the scanner.
func parseQwenXMLCall(body string, schemas map[string]map[string]string) (ToolCall, bool) {
	fnStart, name, fnBodyStart, ok := findQwenAttributeTag(body, 0, "function")
	if !ok || strings.TrimSpace(body[:fnStart]) != "" {
		return ToolCall{}, false
	}
	fnClose, fnAfter := findQwenCloseTag(body, fnBodyStart, "function")
	if fnClose < 0 || strings.TrimSpace(body[fnAfter:]) != "" {
		return ToolCall{}, false
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return ToolCall{}, false
	}

	rest := body[fnBodyStart:fnClose]
	// Build JSON by hand to preserve parameter order. Re-rendering assistant
	// calls in a later turn must retain the model's original argument ordering.
	var args bytes.Buffer
	args.WriteByte('{')
	n := 0
	pos := 0
	for {
		p, key, valueStart, found := findQwenAttributeTag(rest, pos, "parameter")
		if !found {
			if strings.TrimSpace(rest[pos:]) != "" {
				return ToolCall{}, false
			}
			break
		}
		if strings.TrimSpace(rest[pos:p]) != "" {
			return ToolCall{}, false
		}
		key = strings.TrimSpace(key)
		if key == "" {
			return ToolCall{}, false
		}
		valueEnd, afterValue := findQwenCloseTag(rest, valueStart, "parameter")
		if valueEnd < 0 { // never dispatch a truncated parameter
			return ToolCall{}, false
		}
		val := trimQwenStructuralNewline(rest[valueStart:valueEnd])
		if n > 0 {
			args.WriteByte(',')
		}
		n++
		keyJSON, _ := json.Marshal(key)
		args.Write(keyJSON)
		args.WriteByte(':')
		args.WriteString(qwenCoerceArg(val, schemas[name][key]))
		pos = afterValue
	}
	args.WriteByte('}')
	return ToolCall{
		ID:       "call_" + name,
		Type:     "function",
		Function: ToolCallFunction{Name: name, Arguments: args.String()},
	}, true
}

// findQwenAttributeTag accepts <element=value> plus whitespace around '='. It
// intentionally does not decode XML entities: this protocol is XML-shaped, not
// general XML, and model values are raw text.
func findQwenAttributeTag(s string, from int, element string) (start int, value string, after int, ok bool) {
	needle := "<" + element
	for from <= len(s)-len(needle) {
		i := strings.Index(s[from:], needle)
		if i < 0 {
			break
		}
		i += from
		j := i + len(needle)
		for j < len(s) && isQwenSpace(s[j]) {
			j++
		}
		if j >= len(s) || s[j] != '=' {
			from = i + 1
			continue
		}
		j++
		for j < len(s) && isQwenSpace(s[j]) {
			j++
		}
		valueStart := j
		for j < len(s) && s[j] != '>' {
			j++
		}
		if j >= len(s) {
			return -1, "", -1, false
		}
		return i, s[valueStart:j], j + 1, true
	}
	return -1, "", -1, false
}

func trimQwenStructuralNewline(s string) string {
	if strings.HasPrefix(s, "\r\n") {
		s = s[2:]
	} else if strings.HasPrefix(s, "\n") || strings.HasPrefix(s, "\r") {
		s = s[1:]
	}
	if strings.HasSuffix(s, "\r\n") {
		s = s[:len(s)-2]
	} else if strings.HasSuffix(s, "\n") || strings.HasSuffix(s, "\r") {
		s = s[:len(s)-1]
	}
	return s
}

// qwenCoerceArg converts a raw <parameter> body into valid JSON. A declared
// schema wins; otherwise unambiguous JSON numbers, booleans, null, objects, and
// arrays retain their types, while every other value remains a string.
func qwenCoerceArg(val, typ string) string {
	jsonStr := func() string {
		var buf bytes.Buffer
		enc := json.NewEncoder(&buf)
		enc.SetEscapeHTML(false)
		_ = enc.Encode(val)
		return strings.TrimSuffix(buf.String(), "\n")
	}
	t := strings.TrimSpace(val)
	compactStructured := func(expected byte) (string, bool) {
		if len(t) == 0 || t[0] != expected || !json.Valid([]byte(t)) {
			return "", false
		}
		var buf bytes.Buffer
		if json.Compact(&buf, []byte(t)) != nil {
			return "", false
		}
		return buf.String(), true
	}
	switch typ {
	case "string":
		return jsonStr()
	case "integer":
		if _, err := strconv.ParseInt(t, 10, 64); err == nil && json.Valid([]byte(t)) {
			return t
		}
		if f, err := strconv.ParseFloat(t, 64); err == nil && !math.IsInf(f, 0) && !math.IsNaN(f) && f == math.Trunc(f) {
			return strconv.FormatFloat(f, 'f', 0, 64)
		}
		return jsonStr()
	case "number":
		if len(t) > 0 && json.Valid([]byte(t)) && (t[0] == '-' || t[0] >= '0' && t[0] <= '9') {
			return t
		}
		return jsonStr()
	case "boolean":
		switch strings.ToLower(t) {
		case "true":
			return "true"
		case "false":
			return "false"
		}
		return jsonStr()
	case "object":
		if compact, ok := compactStructured('{'); ok {
			return compact
		}
		return jsonStr()
	case "array":
		if compact, ok := compactStructured('['); ok {
			return compact
		}
		return jsonStr()
	}
	if compact, ok := compactStructured('{'); ok {
		return compact
	}
	if compact, ok := compactStructured('['); ok {
		return compact
	}
	switch strings.ToLower(t) {
	case "true":
		return "true"
	case "false":
		return "false"
	case "null", "none":
		return "null"
	}
	if len(t) > 0 && json.Valid([]byte(t)) && (t[0] == '-' || t[0] >= '0' && t[0] <= '9') {
		return t
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
