package chat

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
func RenderToolDeclarations(tools []Tool) string {
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
	switch typ {
	case "STRING":
		if en, ok := p["enum"].([]interface{}); ok && len(en) > 0 {
			var es []string
			for _, e := range en {
				es = append(es, gstr(fmt.Sprintf("%v", e)))
			}
			parts = append(parts, "enum:["+strings.Join(es, ",")+"]")
		}
	case "ARRAY":
		if items, ok := p["items"].(map[string]interface{}); ok {
			parts = append(parts, "items:"+renderProperty(items))
		}
	case "OBJECT":
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
		// Plain JSON-style quotes, NOT the <|"|> delimiter: the model itself
		// emits `path: "value"` in its calls (observed at temp 0), and the
		// re-render must token-match that emission or the prefix cache
		// diverges inside the previous turn on every multi-turn request. The
		// <|"|> form remains the DECLARATION syntax (gstr) and is still
		// accepted by the parser.
		return strconv.Quote(x)
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
func RenderAssistantToolCalls(calls []ToolCall) string {
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
func RenderToolResponse(name, content string) string {
	if name == "" {
		name = "tool"
	}
	// content is the raw tool output; wrap as {value:<|"|>...<|"|>}.
	return "<|tool_response>response:" + name + "{value:" + gstr(content) + "}<tool_response|>"
}

// isToolChoiceNone reports whether the request explicitly disabled tools.
func IsToolChoiceNone(tc interface{}) bool {
	s, ok := tc.(string)
	return ok && s == "none"
}

// DroppedCall records a tool call rejected by ValidateToolCalls because a
// required parameter was missing or empty.
type DroppedCall struct {
	Name  string // function name
	Param string // the missing/empty required parameter
}

// ValidateToolCalls splits calls into those that satisfy their tool's declared
// `required` parameters and those that do not. A call is dropped when a required
// property is absent, or present but "empty" (empty string / empty array /
// empty object / null) — e.g. web_search{"query":""}. Calls whose tool is
// unknown or declares no required params pass through unchanged.
//
// The engine uses this to refuse to dispatch a schema-violating tool call
// (returning a clarification instead), rather than forwarding a malformed call.
func ValidateToolCalls(calls []ToolCall, tools []Tool) (valid []ToolCall, dropped []DroppedCall) {
	type spec struct{ required []string }
	byName := make(map[string]spec, len(tools))
	for _, t := range tools {
		if t.Function.Name == "" || len(t.Function.Parameters) == 0 {
			continue
		}
		var p struct {
			Required []string `json:"required"`
		}
		if json.Unmarshal(t.Function.Parameters, &p) == nil && len(p.Required) > 0 {
			byName[t.Function.Name] = spec{required: p.Required}
		}
	}
	for _, c := range calls {
		sc, ok := byName[c.Function.Name]
		if !ok {
			valid = append(valid, c)
			continue
		}
		var args map[string]interface{}
		_ = json.Unmarshal([]byte(c.Function.Arguments), &args)
		missing := ""
		for _, req := range sc.required {
			if v, present := args[req]; !present || isEmptyArg(v) {
				missing = req
				break
			}
		}
		if missing != "" {
			dropped = append(dropped, DroppedCall{Name: c.Function.Name, Param: missing})
		} else {
			valid = append(valid, c)
		}
	}
	return valid, dropped
}

// isEmptyArg reports whether a decoded JSON argument value counts as "empty" for
// a required parameter. Numbers and booleans (including 0 and false) are NEVER
// empty — only null, "", [], and {}.
func isEmptyArg(v interface{}) bool {
	switch x := v.(type) {
	case nil:
		return true
	case string:
		return strings.TrimSpace(x) == ""
	case []interface{}:
		return len(x) == 0
	case map[string]interface{}:
		return len(x) == 0
	default:
		return false
	}
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
func ParseToolCalls(raw string) (string, []ToolCall) {
	// Strip complete reasoning channels: <|channel>…<channel|>
	content := stripChannels(raw)

	// A call is delimited by <|tool_call>…<tool_call|> (AR models) OR just terminated by
	// <tool_call|> with the OPEN token omitted (observed with DiffusionGemma). Splitting on
	// the CLOSE token handles both: each segment may carry an optional leading <|tool_call>
	// and a "call:NAME{…}" body; the final segment is trailing content.
	var calls []ToolCall
	var contentB strings.Builder
	segs := strings.Split(content, "<tool_call|>")
	for k, seg := range segs {
		last := k == len(segs)-1
		if idx := strings.Index(seg, "<|tool_call>"); idx >= 0 {
			contentB.WriteString(seg[:idx]) // text before the open marker is content
			seg = seg[idx+len("<|tool_call>"):]
		} else if last {
			contentB.WriteString(seg) // trailing text after the last call (no open, no close)
			continue
		}
		// seg is a candidate call body: text before "call:" is content, the rest is the call.
		ci := strings.Index(seg, "call:")
		if ci < 0 {
			contentB.WriteString(seg)
			continue
		}
		contentB.WriteString(seg[:ci])
		if tc, ok := parseOneCall(seg[ci:]); ok {
			// Unique id per call so clients can map multiple tool results back to the right
			// call (two calls to the same function must not collide).
			tc.ID = fmt.Sprintf("call_%s_%d", tc.Function.Name, len(calls))
			calls = append(calls, tc)
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
		ID:       "call_" + name, // overwritten with a unique id by parseToolCalls
		Type:     "function",
		Function: ToolCallFunction{Name: name, Arguments: string(argsJSON)},
	}, true
}

// markerStripper removes the gemma-4 control-marker literals that DecodeRaw emits
// (it keeps them so tool/channel structure survives parsing). Applied to plain
// answer text after reasoning + tool calls have been extracted.
var markerStripper = strings.NewReplacer(
	"<|turn>", "", "<turn|>", "",
	"<|channel>", "", "<channel|>", "",
	"<|tool>", "", "<tool|>", "",
	"<|tool_call>", "", "<tool_call|>", "",
	"<|tool_response>", "", "<tool_response|>", "",
	"<|think|>", "", `<|"|>`, "",
)

func StripMarkers(s string) string { return markerStripper.Replace(s) }

// splitReasoning separates the gemma-4 thought channel from the answer in RAW
// decoded output (markers intact). It returns (reasoning, rest): reasoning is the
// concatenated <|channel>thought…<channel|> payloads with the leading channel
// label ("thought\n") stripped; rest is everything outside the channels (the
// answer, still containing any tool-call markers for the caller to parse). An
// unterminated channel (generation hit the token limit mid-thought) contributes
// its partial payload to reasoning and leaves rest empty.
func SplitReasoning(s string) (reasoning, rest string) {
	var rsb, osb strings.Builder
	for {
		o := strings.Index(s, "<|channel>")
		if o < 0 {
			osb.WriteString(s)
			break
		}
		osb.WriteString(s[:o])
		s = s[o+len("<|channel>"):]
		e := strings.Index(s, "<channel|>")
		var payload string
		if e < 0 { // unterminated: rest of output is reasoning
			payload = s
			s = ""
		} else {
			payload = s[:e]
			s = s[e+len("<channel|>"):]
		}
		// Drop the channel label (e.g. "thought\n") — the text up to the first newline.
		if nl := strings.IndexByte(payload, '\n'); nl >= 0 {
			payload = payload[nl+1:]
		}
		rsb.WriteString(payload)
		if e < 0 {
			break
		}
	}
	// Reasoning is returned EXACTLY as generated (no trimming): the client
	// echoes it back as reasoning_content, and the chat template re-renders it
	// inside the thought channel. Any whitespace skew breaks the token match
	// with the cached KV — the model typically ends its reasoning with "\n",
	// and without it the re-encoded ".<channel|>" boundary even BPE-merges
	// instead of hitting the <channel|> special token (observed). Streaming
	// clients already receive the exact payload, delta by delta.
	return rsb.String(), osb.String()
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
	if s[i] == '"' { // string: plain JSON-style quotes (what the model emits)
		var sb strings.Builder
		j := i + 1
		for j < len(s) {
			c := s[j]
			if c == '\\' && j+1 < len(s) {
				// JSON escapes; unknown sequences keep the escaped char as-is.
				switch s[j+1] {
				case 'n':
					sb.WriteByte('\n')
				case 't':
					sb.WriteByte('\t')
				case 'r':
					sb.WriteByte('\r')
				default:
					sb.WriteByte(s[j+1])
				}
				j += 2
				continue
			}
			if c == '"' {
				return sb.String(), j + 1, true
			}
			sb.WriteByte(c)
			j++
		}
		return sb.String(), j, true // unterminated: rest of input
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
