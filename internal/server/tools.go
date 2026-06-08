package server

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// ─── OpenAI-compatible tool types ──────────────────────────────────────

type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON-encoded string, per OpenAI convention
}

// ─── Prompt side: render tool declarations into the gemma-4 syntax ─────
//
// gemma-4 declares tools inside the system turn as:
//   <|tool>declaration:NAME{description:<|"|>DESC<|"|>,parameters:{properties:{
//     KEY:{description:<|"|>...<|"|>,type:<|"|>STRING<|"|>},...},
//     required:[<|"|>k<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>
// (custom `<|"|>` string delimiters; `{k: v}` dicts, NOT JSON.)

const sd = `<|"|>` // string delimiter

func gstr(s string) string { return sd + s + sd }

// renderToolDeclarations returns the concatenated <|tool>…<tool|> blocks.
func renderToolDeclarations(tools []Tool) string {
	var sb strings.Builder
	for _, t := range tools {
		fn := t.Function
		sb.WriteString("<|tool>declaration:")
		sb.WriteString(fn.Name)
		sb.WriteString("{description:")
		sb.WriteString(gstr(fn.Description))
		if len(fn.Parameters) > 0 {
			var params map[string]interface{}
			if json.Unmarshal(fn.Parameters, &params) == nil && len(params) > 0 {
				sb.WriteString(",parameters:")
				sb.WriteString(renderParamSchema(params))
			}
		}
		sb.WriteString("}<tool|>")
	}
	return sb.String()
}

// renderParamSchema renders a JSON-Schema object node into the gemma-4 dict syntax.
func renderParamSchema(schema map[string]interface{}) string {
	var sb strings.Builder
	sb.WriteString("{")
	parts := []string{}
	if props, ok := schema["properties"].(map[string]interface{}); ok && len(props) > 0 {
		var inner []string
		for _, name := range sortedKeys(props) {
			pv, _ := props[name].(map[string]interface{})
			inner = append(inner, name+":"+renderProperty(pv))
		}
		parts = append(parts, "properties:{"+strings.Join(inner, ",")+"}")
	}
	if req, ok := schema["required"].([]interface{}); ok && len(req) > 0 {
		var rs []string
		for _, r := range req {
			if s, ok := r.(string); ok {
				rs = append(rs, gstr(s))
			}
		}
		parts = append(parts, "required:["+strings.Join(rs, ",")+"]")
	}
	typ := "OBJECT"
	if ts, ok := schema["type"].(string); ok {
		typ = strings.ToUpper(ts)
	}
	parts = append(parts, "type:"+gstr(typ))
	sb.WriteString(strings.Join(parts, ","))
	sb.WriteString("}")
	return sb.String()
}

// renderProperty renders a single property schema node.
func renderProperty(p map[string]interface{}) string {
	if p == nil {
		return "{type:" + gstr("STRING") + "}"
	}
	var parts []string
	if d, ok := p["description"].(string); ok && d != "" {
		parts = append(parts, "description:"+gstr(d))
	}
	typ := "STRING"
	if ts, ok := p["type"].(string); ok {
		typ = strings.ToUpper(ts)
	}
	if typ == "STRING" {
		if en, ok := p["enum"].([]interface{}); ok && len(en) > 0 {
			var es []string
			for _, e := range en {
				es = append(es, gstr(fmt.Sprintf("%v", e)))
			}
			parts = append(parts, "enum:["+strings.Join(es, ",")+"]")
		}
	} else if typ == "ARRAY" {
		if items, ok := p["items"].(map[string]interface{}); ok {
			parts = append(parts, "items:"+renderProperty(items))
		}
	} else if typ == "OBJECT" {
		if _, ok := p["properties"]; ok {
			return renderParamSchema(p) // nested object reuses the object renderer
		}
	}
	parts = append(parts, "type:"+gstr(typ))
	return "{" + strings.Join(parts, ",") + "}"
}

// encodeGemmaValue encodes a JSON value into the gemma-4 value syntax (inverse of
// parseGemmaValue) — used to re-render assistant tool_calls into the prompt on
// multi-turn requests so the context matches what the model produced.
func encodeGemmaValue(v interface{}) string {
	switch x := v.(type) {
	case nil:
		return "null"
	case bool:
		if x {
			return "true"
		}
		return "false"
	case float64:
		return strconv.FormatFloat(x, 'g', -1, 64)
	case json.Number:
		return x.String()
	case string:
		return gstr(x)
	case map[string]interface{}:
		var parts []string
		for _, k := range sortedKeys(x) {
			parts = append(parts, k+": "+encodeGemmaValue(x[k]))
		}
		return "{" + strings.Join(parts, ", ") + "}"
	case []interface{}:
		var parts []string
		for _, e := range x {
			parts = append(parts, encodeGemmaValue(e))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	default:
		return gstr(fmt.Sprintf("%v", x))
	}
}

// renderAssistantToolCalls re-renders an assistant message's tool_calls into the
// gemma-4 <|tool_call>call:NAME{…}<tool_call|> syntax.
func renderAssistantToolCalls(calls []ToolCall) string {
	var sb strings.Builder
	for _, c := range calls {
		sb.WriteString("<|tool_call>call:")
		sb.WriteString(c.Function.Name)
		var args map[string]interface{}
		if json.Unmarshal([]byte(c.Function.Arguments), &args) == nil {
			sb.WriteString(encodeGemmaValue(args))
		} else {
			sb.WriteString("{}")
		}
		sb.WriteString("<tool_call|>")
	}
	return sb.String()
}

// renderToolResponse renders a tool result message back to the model.
func renderToolResponse(name, content string) string {
	if name == "" {
		name = "tool"
	}
	// content is the raw tool output; wrap as {value:<|"|>...<|"|>}.
	return "<|tool_response>response:" + name + "{value:" + gstr(content) + "}<tool_response|>"
}

// isToolChoiceNone reports whether the request explicitly disabled tools.
func isToolChoiceNone(tc interface{}) bool {
	s, ok := tc.(string)
	return ok && s == "none"
}

func sortedKeys(m map[string]interface{}) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	// simple insertion sort (small maps)
	for i := 1; i < len(ks); i++ {
		for j := i; j > 0 && ks[j-1] > ks[j]; j-- {
			ks[j-1], ks[j] = ks[j], ks[j-1]
		}
	}
	return ks
}

// ─── Output side: parse <|tool_call>call:NAME{dict}<tool_call|> ─────────

// parseToolCalls extracts assistant content and tool calls from the RAW decoded
// output (which still contains the channel/tool markers). Returns the cleaned
// content (thought channels + tool calls removed) and any tool calls found.
func parseToolCalls(raw string) (string, []ToolCall) {
	// Strip complete reasoning channels: <|channel>…<channel|>
	content := stripChannels(raw)

	var calls []ToolCall
	rest := content
	var contentB strings.Builder
	for {
		idx := strings.Index(rest, "<|tool_call>")
		if idx < 0 {
			contentB.WriteString(rest)
			break
		}
		contentB.WriteString(rest[:idx])
		rest = rest[idx+len("<|tool_call>"):]
		// expect "call:NAME{...}<tool_call|>"
		end := strings.Index(rest, "<tool_call|>")
		var body string
		if end < 0 {
			body = rest // unterminated (e.g. stopped on the close token)
			rest = ""
		} else {
			body = rest[:end]
			rest = rest[end+len("<tool_call|>"):]
		}
		if tc, ok := parseOneCall(body); ok {
			calls = append(calls, tc)
		}
		if end < 0 {
			break
		}
	}
	return strings.TrimSpace(contentB.String()), calls
}

// parseOneCall parses "call:NAME{dict}" into a ToolCall with JSON arguments.
func parseOneCall(body string) (ToolCall, bool) {
	body = strings.TrimSpace(body)
	body = strings.TrimPrefix(body, "call:")
	brace := strings.Index(body, "{")
	if brace < 0 {
		return ToolCall{}, false
	}
	name := strings.TrimSpace(body[:brace])
	val, _, ok := parseGemmaValue(body, brace)
	if !ok {
		return ToolCall{}, false
	}
	argsJSON, err := json.Marshal(val)
	if err != nil {
		argsJSON = []byte("{}")
	}
	return ToolCall{
		ID:       "call_" + name,
		Type:     "function",
		Function: ToolCallFunction{Name: name, Arguments: string(argsJSON)},
	}, true
}

// stripChannels removes complete <|channel>…<channel|> blocks and any stray
// <channel|> markers from the text.
func stripChannels(s string) string {
	for {
		o := strings.Index(s, "<|channel>")
		if o < 0 {
			break
		}
		e := strings.Index(s[o:], "<channel|>")
		if e < 0 {
			s = s[:o] // unterminated channel → drop the rest
			break
		}
		s = s[:o] + s[o+e+len("<channel|>"):]
	}
	return strings.ReplaceAll(s, "<channel|>", "")
}

// ─── gemma-4 value parser (dict/array/string/number/bool/null) ─────────

func parseGemmaValue(s string, i int) (interface{}, int, bool) {
	i = skipSpace(s, i)
	if i >= len(s) {
		return nil, i, false
	}
	if strings.HasPrefix(s[i:], sd) { // string: <|"|>…<|"|>
		i += len(sd)
		end := strings.Index(s[i:], sd)
		if end < 0 {
			return s[i:], len(s), true
		}
		return s[i : i+end], i + end + len(sd), true
	}
	switch s[i] {
	case '{':
		return parseGemmaDict(s, i)
	case '[':
		return parseGemmaArray(s, i)
	}
	if strings.HasPrefix(s[i:], "true") {
		return true, i + 4, true
	}
	if strings.HasPrefix(s[i:], "false") {
		return false, i + 5, true
	}
	if strings.HasPrefix(s[i:], "null") {
		return nil, i + 4, true
	}
	// number: read until a delimiter
	j := i
	for j < len(s) && !strings.ContainsRune(",}]", rune(s[j])) && s[j] != ' ' {
		j++
	}
	tok := s[i:j]
	if n, err := strconv.ParseFloat(tok, 64); err == nil {
		return n, j, true
	}
	return tok, j, true // fall back to bare string
}

func parseGemmaDict(s string, i int) (interface{}, int, bool) {
	m := map[string]interface{}{}
	i++ // consume '{'
	i = skipSpace(s, i)
	if i < len(s) && s[i] == '}' {
		return m, i + 1, true
	}
	for i < len(s) {
		i = skipSpace(s, i)
		// key: bare text up to ':'
		k := i
		for k < len(s) && s[k] != ':' && s[k] != '}' {
			k++
		}
		if k >= len(s) || s[k] == '}' {
			return m, k, true
		}
		key := strings.TrimSpace(s[i:k])
		i = k + 1 // consume ':'
		v, ni, ok := parseGemmaValue(s, i)
		if !ok {
			return m, ni, false
		}
		m[key] = v
		i = skipSpace(s, ni)
		if i < len(s) && s[i] == ',' {
			i++
			continue
		}
		if i < len(s) && s[i] == '}' {
			return m, i + 1, true
		}
		break
	}
	return m, i, true
}

func parseGemmaArray(s string, i int) (interface{}, int, bool) {
	var arr []interface{}
	i++ // consume '['
	i = skipSpace(s, i)
	if i < len(s) && s[i] == ']' {
		return arr, i + 1, true
	}
	for i < len(s) {
		v, ni, ok := parseGemmaValue(s, i)
		if !ok {
			return arr, ni, false
		}
		arr = append(arr, v)
		i = skipSpace(s, ni)
		if i < len(s) && s[i] == ',' {
			i++
			continue
		}
		if i < len(s) && s[i] == ']' {
			return arr, i + 1, true
		}
		break
	}
	return arr, i, true
}

func skipSpace(s string, i int) int {
	for i < len(s) && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r') {
		i++
	}
	return i
}
