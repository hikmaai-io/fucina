// Diffusion-gemma execution path. Routed from main() when the GGUF's architecture is
// "diffusion-gemma". DiffusionGemma denoises whole 256-token blocks, so generation is
// block-oriented (no token streaming): oneshot, interactive, and a minimal non-streaming
// OpenAI-compatible chat server.

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/hikmaai-io/fucina/internal/chat"
	"github.com/hikmaai-io/fucina/internal/engine/diffusion"
	gemserver "github.com/hikmaai-io/fucina/internal/server"
	"github.com/hikmaai-io/fucina/internal/tokenizer"
)

// runDiffusion drives the diffusion-gemma engine for the chosen mode and returns.
func runDiffusion(args CLIArgs, tok *tokenizer.Tokenizer) {
	if tok == nil {
		log.Fatalf("fucina: diffusion mode requires a working tokenizer")
	}
	// DiffusionGemma declares a 262144-token context (GGUF diffusion-gemma.context_length).
	// Each prompt token costs ~1 MB of scratch + KV cache and the naive O(N²) canvas attention
	// makes long prompts slow, so the ceiling is the model's declared max but the DEFAULT is a
	// modest base; --ctx raises the window toward the ceiling, and dg_engine_create then caps
	// allocation to whatever GPU memory actually allows (so a big --ctx degrades, not OOMs).
	const dgCtxBase, dgCtxMax = 8192, 262144
	ctx := args.ContextSize
	if ctx >= dgCtxMax { // global --ctx default (262144) → use the diffusion base, not the full ceiling
		ctx = dgCtxBase + 256
	}
	maxPrompt := ctx - 256
	if maxPrompt > dgCtxMax {
		maxPrompt = dgCtxMax
	}
	if maxPrompt < 256 {
		maxPrompt = 1024
	}
	log.Printf("fucina: loading diffusion-gemma model %s ...", args.ModelPath)
	t0 := time.Now()
	eng, err := diffusion.NewEngine(args.ModelPath, maxPrompt, args.FP4MoE)
	if err != nil {
		log.Fatalf("fucina: %v", err)
	}
	defer eng.Close()
	log.Printf("fucina: diffusion engine ready in %.1fs (canvas=%d, max_prompt=%d)",
		time.Since(t0).Seconds(), eng.CanvasLength(), eng.MaxPrompt())

	params := diffusion.DefaultParams()
	params.EOTID = int(tok.EndOfTurn) // stop block chaining at the model's end-of-turn marker
	if args.DenoiseSteps > 0 {
		params.MaxSteps = args.DenoiseSteps // quality/speed knob: fewer steps/block = faster, lower quality
	}
	if args.Seed >= 0 {
		params.Seed = uint64(args.Seed) + 1
	}

	// answer renders the history, denoises one block, and decodes the committed canvas
	// (dropping the model's <|channel>thought…<channel|> block). Also returns timing stats.
	answer := func(history []gemserver.ChatMessage) (string, diffusion.Stats, error) {
		ids := tok.Encode(buildDiffusionPrompt(args.System, history, nil), true, false)
		out, st, err := eng.Generate(ids, params)
		if err != nil {
			return "", diffusion.Stats{}, err
		}
		return extractAnswer(tok, out), st, nil
	}

	// ── one-shot ──
	if args.Prompt != "" || args.PromptFile != "" {
		text := args.Prompt
		if args.PromptFile != "" {
			b, err := os.ReadFile(args.PromptFile)
			if err != nil {
				log.Fatalf("fucina: %v", err)
			}
			text = string(b)
		}
		out, st, err := answer([]gemserver.ChatMessage{{Role: "user", Content: text}})
		if err != nil {
			log.Fatalf("fucina: %v", err)
		}
		fmt.Println(out)
		logDiffusionRate(st)                                // prefill + generation tok/s (like dense)
		fmt.Fprintf(os.Stderr, "[%s]\n", tokPerSecLine(st)) // compact generation readout
		if args.Timings {
			printDiffusionStats(st)
		}
		return
	}

	// ── interactive REPL ──
	if args.Interactive {
		runDiffusionREPL(args, tok, eng, answer)
		return
	}

	// ── server (minimal, non-streaming OpenAI-compatible chat) ──
	runDiffusionServer(args, tok, eng, params)
}

// runDiffusionREPL is the interactive chat loop for DiffusionGemma. It mirrors the dense model's
// REPL (runInteractive) — same banner, green "> " prompt, /reset|/clear, /stats, /quit|/exit|/q,
// Ctrl-D handling, near-context warning, blue "Assistant:" prefix, and dim per-turn stat lines —
// differing only where the model forces it: block diffusion denoises a whole block at once and
// cannot stream tokens, so it shows a "denoising" progress indicator instead.
func runDiffusionREPL(args CLIArgs, tok *tokenizer.Tokenizer, eng *diffusion.Engine, answer func([]gemserver.ChatMessage) (string, diffusion.Stats, error)) {
	var history []gemserver.ChatMessage
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 0, 1<<20), 1<<20)
	ctxTokens := eng.MaxPrompt()

	fmt.Fprintf(os.Stderr,
		"fucina: interactive mode (DiffusionGemma 26B-A4B) — ctx=%d\n%s\n",
		ctxTokens, diffusionCommandsHelp)

	for {
		fmt.Fprint(os.Stderr, "\033[1;32m> \033[0m") // green prompt
		if !scanner.Scan() {
			fmt.Fprintln(os.Stderr, "\nfucina: bye") // Ctrl-D / EOF
			return
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		// Slash commands
		switch input {
		case "/quit", "/exit", "/q":
			fmt.Fprintln(os.Stderr, "fucina: bye")
			return
		case "/help", "/h", "/?", "/commands":
			fmt.Fprintf(os.Stderr, "fucina: commands —\n%s", diffusionCommandsHelp)
			continue
		case "/reset", "/clear":
			history = history[:0]
			fmt.Fprintln(os.Stderr, "fucina: conversation cleared")
			continue
		case "/stats":
			// Block diffusion re-prefills each turn (no cross-turn KV reuse like the dense
			// KVCache), so report the prompt window + block size rather than a cache hit rate.
			fmt.Fprintf(os.Stderr, "fucina: context — max_prompt=%d  canvas=%d tokens\n", eng.MaxPrompt(), eng.CanvasLength())
			continue
		}

		// Unknown leading-slash command (the diffusion REPL has no /thinking): report
		// it rather than sending "/foo" to the model as chat input.
		if looksLikeCommand(input) {
			fmt.Fprintf(os.Stderr, "fucina: unknown command %q — type /help for the list\n",
				strings.Fields(input)[0])
			continue
		}

		history = append(history, gemserver.ChatMessage{Role: "user", Content: input})

		// Warn if we are close to the context limit (mirrors the dense REPL).
		promptToks := tok.Encode(buildDiffusionPrompt(args.System, history, nil), true, false)
		if len(promptToks) > ctxTokens-64 {
			fmt.Fprintf(os.Stderr,
				"\033[33mfucina: warning: prompt is %d tokens, near context limit %d\033[0m\n",
				len(promptToks), ctxTokens)
		}

		// Live indicator: a block takes several seconds and can't stream, so emit dots on stderr
		// until it returns, then clear the line and print the reply behind the blue prefix.
		done := make(chan struct{})
		go func() {
			fmt.Fprint(os.Stderr, "\033[2mdenoising\033[0m")
			t := time.NewTicker(500 * time.Millisecond)
			defer t.Stop()
			for {
				select {
				case <-done:
					return
				case <-t.C:
					fmt.Fprint(os.Stderr, "\033[2m.\033[0m")
				}
			}
		}()

		out, st, err := answer(history)
		close(done)
		fmt.Fprint(os.Stderr, "\r\033[K") // clear the indicator line

		if err != nil {
			fmt.Fprintf(os.Stderr, "fucina: %v\n", err)
			history = history[:len(history)-1] // undo user turn
			continue
		}
		fmt.Fprintf(os.Stdout, "\033[1;34mAssistant:\033[0m %s\n", out)
		// Dim per-turn stats to stderr, matching the dense REPL's prefill/generated lines.
		fmt.Fprintf(os.Stderr,
			"\033[2mfucina: prefill %d tokens %.2fs %.1f tok/s | generated %d tokens %.2fs %.1f tok/s (%d steps, %.0f canvas tok/s)\033[0m\n\n",
			st.NPrompt, st.PrefillMS/1000, st.PrefillTokPerSec(),
			st.NOut, st.DenoiseMS/1000, st.GenTokPerSec(), st.Steps, st.CanvasTokPerSec())
		if args.Timings {
			printDiffusionStats(st)
		}
		history = append(history, gemserver.ChatMessage{Role: "assistant", Content: out})
	}
}

// printDiffusionStats writes prefill/generation timing for one denoised block to stderr (so
// piped stdout stays clean). Generation tok/s = committed tokens over the denoise loop; the
// prompt is prefilled once (KV-cached) and not recomputed per denoising step.
func printDiffusionStats(st diffusion.Stats) {
	fmt.Fprintf(os.Stderr,
		"timings: prefill %d tok in %.2fs (%.0f tok/s) | generate %d tok over %d steps in %.2fs (%.1f tok/s delivered, %.0f canvas tok/s, %.0f ms/step)\n",
		st.NPrompt, st.PrefillMS/1000, st.PrefillTokPerSec(),
		st.NOut, st.Steps, st.DenoiseMS/1000, st.GenTokPerSec(), st.CanvasTokPerSec(),
		stepMS(st))
}

// logDiffusionRate logs prefill + generation tok/s per request, mirroring the dense server's
// "prefill … | generated …" log lines. Generation reports both rates: "delivered" (committed
// answer tokens / time, what the caller gets) and "canvas" (all CanvasLen slots refined per step
// / time, the engine's true denoising throughput — the honest number for short answers).
func logDiffusionRate(st diffusion.Stats) {
	log.Printf("fucina: prefill %d tokens in %.2fs (%.1f tok/s) | generate %d tokens in %.2fs (%.1f tok/s delivered, %.0f canvas tok/s, %d steps, %.0f ms/step)",
		st.NPrompt, st.PrefillMS/1000, st.PrefillTokPerSec(),
		st.NOut, st.DenoiseMS/1000, st.GenTokPerSec(), st.CanvasTokPerSec(), st.Steps, stepMS(st))
}

// tokPerSecLine is the compact generation readout shown in the UI (REPL/oneshot/server).
func tokPerSecLine(st diffusion.Stats) string {
	return fmt.Sprintf("%d tok · %d steps · %.1f tok/s (%.0f canvas tok/s) · %.0f ms/step",
		st.NOut, st.Steps, st.GenTokPerSec(), st.CanvasTokPerSec(), stepMS(st))
}

// stepMS is the average denoise time per step (0 if no steps ran).
func stepMS(st diffusion.Stats) float64 {
	if st.Steps == 0 {
		return 0
	}
	return st.DenoiseMS / float64(st.Steps)
}

// extractAnswer drops the model's thought channel from a committed canvas: the answer is the
// text after the (last) <channel|> marker, up to the first <turn|>/EOS. Falls back to the whole
// block when there is no thought channel.
func extractAnswer(tok *tokenizer.Tokenizer, out []int32) string {
	start := 0
	for i, id := range out {
		if id == tok.ChannelEnd {
			start = i + 1
		}
	}
	end := len(out)
	for i := start; i < len(out); i++ {
		if out[i] == tok.EndOfTurn || out[i] == tok.EOS {
			end = i
			break
		}
	}
	return strings.TrimSpace(tok.Decode(out[start:end]))
}

// buildDiffusionPrompt formats the gemma-4 chat prompt for DiffusionGemma and opens the model
// turn in thinking style (`<|turn>model\n`) so the model emits its own <|channel>thought…
// <channel|> block followed by the answer — the validated-working diffusion format. Tool
// declarations + assistant tool_calls + tool results are rendered via the SHARED gemma-4
// helpers in internal/chat (same logic the AR server uses), so only the diffusion-specific
// turn skeleton lives here.
func buildDiffusionPrompt(system string, msgs []gemserver.ChatMessage, tools []gemserver.Tool) string {
	var sb strings.Builder
	sysContent := system
	i := 0
	if len(msgs) > 0 && msgs[0].Role == "system" {
		if sysContent == "" {
			sysContent = msgs[0].Content
		}
		i = 1
	}
	toolDecl := ""
	if len(tools) > 0 {
		toolDecl = chat.RenderToolDeclarations(tools)
	}
	if sysContent != "" || toolDecl != "" {
		sb.WriteString("<|turn>system\n")
		sb.WriteString(sysContent)
		sb.WriteString(toolDecl)
		sb.WriteString("<turn|>\n")
	}
	for ; i < len(msgs); i++ {
		m := msgs[i]
		switch m.Role {
		case "user":
			sb.WriteString("<|turn>user\n")
			sb.WriteString(m.Content)
			sb.WriteString("<turn|>\n")
		case "assistant":
			sb.WriteString("<|turn>model\n<|channel>thought\n<channel|>")
			sb.WriteString(m.Content)
			if len(m.ToolCalls) > 0 {
				sb.WriteString(chat.RenderAssistantToolCalls(m.ToolCalls))
			}
			sb.WriteString("<turn|>\n")
		case "tool":
			sb.WriteString(chat.RenderToolResponse(m.Name, m.Content))
		case "system":
			sb.WriteString("<|turn>system\n")
			sb.WriteString(m.Content)
			sb.WriteString("<turn|>\n")
		}
	}
	sb.WriteString("<|turn>model\n")
	return sb.String()
}

// decodeDiffusionTools decodes the committed canvas into the answer text + any tool calls,
// using the SHARED gemma-4 parsers. DecodeRaw keeps the tool/channel markers so SplitReasoning
// can drop the thought channel and ParseToolCalls can extract <|tool_call>…<tool_call|> blocks.
func decodeDiffusionTools(tok *tokenizer.Tokenizer, out []int32) (string, []gemserver.ToolCall) {
	end := len(out)
	for i, id := range out {
		if id == tok.EOS || id == tok.EndOfTurn {
			end = i
			break
		}
	}
	raw := tok.DecodeRaw(out[:end])
	if os.Getenv("DG_TOOL_DEBUG") == "1" {
		fmt.Fprintf(os.Stderr, "[DG_TOOL_DEBUG] raw=%q\n", raw)
	}
	_, rest := chat.SplitReasoning(raw)
	content, calls := chat.ParseToolCalls(rest)
	return strings.TrimSpace(chat.StripMarkers(content)), calls
}

// runDiffusionServer serves the OpenAI chat API for DiffusionGemma. It REUSES the AR server's
// OpenAI wire types (gemserver.ChatRequest/ChatMessage — which already accept `content` as a
// string or an array of parts — and ModelInfo/ModelsResponse) so the request/response contract
// matches the AR server exactly. The generation LOOP can't be shared (block diffusion vs the
// AR token-by-token serverEngine interface), so for a streaming request the whole denoised
// block is emitted as a single SSE delta.
func runDiffusionServer(args CLIArgs, tok *tokenizer.Tokenizer, eng *diffusion.Engine,
	params diffusion.Params) {
	addr := fmt.Sprintf("%s:%d", args.Host, args.Port)
	// Canonical id reported by /v1/models — derived from the GGUF filename (not hardcoded).
	// Requests echo back whatever `model` the client sends, so any pi/OpenAI provider id works.
	modelID := deriveModelID(args.ModelPath)

	// Auth, matching the dense server: flag wins, else FUCINA_API_KEY. /v1/* routes
	// are wrapped with the same bearer check so a configured key is NOT silently
	// ignored on the diffusion path (an unauthenticated GPU otherwise).
	apiKey := args.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("FUCINA_API_KEY")
	}
	authed := func(f http.HandlerFunc) http.HandlerFunc { return gemserver.BearerAuth(apiKey, f) }
	if apiKey != "" {
		log.Printf("fucina: API-key auth ENABLED on /v1/*")
	} else if args.Host != "127.0.0.1" && args.Host != "localhost" {
		log.Printf("fucina: WARNING — binding %s with NO API key; /v1/* is unauthenticated. Set --api-key or FUCINA_API_KEY.", args.Host)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", authed(func(w http.ResponseWriter, r *http.Request) {
		// Bound the body: without this any client could OOM the process that owns
		// the GPU (the dense server caps at the same 64 MiB).
		r.Body = http.MaxBytesReader(w, r.Body, 64<<20)
		var req gemserver.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Messages) == 0 {
			http.Error(w, `{"error":{"message":"invalid request","type":"invalid_request_error"}}`, http.StatusBadRequest)
			return
		}
		// tool_choice "none" disables tools entirely; otherwise declare req.Tools.
		var tools []gemserver.Tool
		if !chat.IsToolChoiceNone(req.ToolChoice) {
			tools = req.Tools
		}
		ids := tok.Encode(buildDiffusionPrompt(args.System, req.Messages, tools), true, false)
		// DiffusionGemma's prompt window (max_prompt, ~4096) is far smaller than a typical
		// agent's configured context window. When the conversation exceeds it, return the
		// OpenAI-standard context_length_exceeded error so the client (pi, etc.) triggers
		// compaction and retries — rather than the engine's opaque "exceeds max_prompt" 500.
		if len(ids) > eng.MaxPrompt() {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
				"message": fmt.Sprintf("This model's maximum context length is %d tokens, but the messages resulted in %d tokens. Reduce the message history (compact) and retry.", eng.MaxPrompt(), len(ids)),
				"type":    "invalid_request_error",
				"param":   "messages",
				"code":    "context_length_exceeded",
			}})
			return
		}
		// Per-request output cap: honor the client's max_tokens (rounded up to whole blocks),
		// else fall back to the default. Block chaining stops early on EOS/end-of-turn anyway.
		rp := params
		if req.MaxTokens > 0 {
			rp.MaxNewTokens = req.MaxTokens
		}
		out, st, err := eng.Generate(ids, rp)
		if err != nil {
			http.Error(w, fmt.Sprintf(`{"error":{"message":%q}}`, err.Error()), http.StatusInternalServerError)
			return
		}
		logDiffusionRate(st) // per-request prefill + generation tok/s, like the dense server
		if args.Timings {
			printDiffusionStats(st)
		}
		content, calls := decodeDiffusionTools(tok, out)
		respModel := req.Model
		if respModel == "" {
			respModel = modelID
		}
		finish := "stop"
		if len(calls) > 0 {
			finish = "tool_calls"
		}
		id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
		created := time.Now().Unix()
		usage := map[string]any{
			"prompt_tokens": st.NPrompt, "completion_tokens": st.NOut, "total_tokens": st.NPrompt + st.NOut,
			"denoise_steps": st.Steps, "tokens_per_second": st.GenTokPerSec(),
			"canvas_tokens_per_second": st.CanvasTokPerSec(), "ms_per_step": stepMS(st),
		}
		// assistant message/delta: content (null when tool_calls present) + tool_calls.
		msgFields := func() map[string]any {
			m := map[string]any{"role": "assistant"}
			if len(calls) > 0 {
				m["content"] = nil
				m["tool_calls"] = calls
			} else {
				m["content"] = content
			}
			return m
		}

		if req.Stream {
			fl, ok := w.(http.Flusher)
			if !ok {
				http.Error(w, `{"error":{"message":"streaming unsupported"}}`, http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			send := func(ev map[string]any) {
				b, _ := json.Marshal(ev)
				fmt.Fprintf(w, "data: %s\n\n", b)
				fl.Flush()
			}
			base := func(choices []any) map[string]any {
				return map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": respModel, "choices": choices}
			}
			send(base([]any{map[string]any{"index": 0, "delta": msgFields(), "finish_reason": nil}})) // whole block, one delta
			send(base([]any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": finish}}))
			usageEv := base([]any{}) // final usage chunk (OpenAI include_usage form)
			usageEv["usage"] = usage
			send(usageEv)
			fmt.Fprint(w, "data: [DONE]\n\n")
			fl.Flush()
			return
		}

		resp := map[string]any{
			"id": id, "object": "chat.completion", "created": created, "model": respModel,
			"choices": []any{map[string]any{"index": 0, "message": msgFields(), "finish_reason": finish}},
			"usage":   usage,
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	// OpenAI model discovery (shared gemserver types). Any id under /v1/models/{id} → the served model.
	modelObj := gemserver.ModelInfo{ID: modelID, Object: "model", Created: time.Now().Unix(), OwnedBy: "fucina"}
	mux.HandleFunc("/v1/models", authed(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(gemserver.ModelsResponse{Object: "list", Data: []gemserver.ModelInfo{modelObj}})
	}))
	mux.HandleFunc("/v1/models/", authed(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(modelObj)
	}))
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "ok") })
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "ok") })

	log.Printf("fucina: diffusion server on http://%s (OpenAI: POST /v1/chat/completions [+stream], GET /v1/models)", addr)
	// Hardened server (mirrors the dense server): a header read timeout defeats
	// slowloris and an idle timeout reaps dead keep-alive connections. No
	// WriteTimeout: generations stream for many seconds and a write deadline
	// would kill long responses.
	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("fucina: server: %v", err)
	}
}
