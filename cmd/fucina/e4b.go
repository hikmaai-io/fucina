package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/hikmaai-io/fucina/internal/chat"
	"github.com/hikmaai-io/fucina/internal/engine/e4b"
	"github.com/hikmaai-io/fucina/internal/sampler"
	"github.com/hikmaai-io/fucina/internal/tokenizer"
)

// ─── Gemma-4-E4B path ─────────────────────────────────────────────────
//
// E4B runs through the standalone e4b engine (Per-Layer Embeddings, KV-sharing,
// runtime dims), which is separate from the dense gemma4 engine. It has no MTP /
// speculative path yet, so generation is a plain prefill + per-token decode loop
// with the host sampler. Server mode is not wired for E4B yet.

// e4bCommandsHelp mirrors denseCommandsHelp (think.go) so the two REPLs expose the
// SAME command surface. E4B has no cross-turn KV prefix cache, so /stats reports
// context use + last-turn speed rather than a cache hit rate.
const e4bCommandsHelp = "  /thinking LEVEL  set reasoning: off|on|low|medium|high|xhigh\n" +
	"  /reset           clear conversation\n" +
	"  /stats           show context use + last-turn speed\n" +
	"  /help            show this help\n" +
	"  /quit            exit (or Ctrl-D)\n"

// runE4B is the CLI entry for an E4B checkpoint (already detected by main()).
func runE4B(args CLIArgs) {
	log.Printf("fucina: loading E4B checkpoint %s (ctx=%d, device=%d)...",
		args.ModelPath, args.ContextSize, args.DeviceID)
	eng, err := e4b.New(args.ModelPath, uint32(args.ContextSize), args.DeviceID)
	if err != nil {
		log.Fatalf("fucina: %v", err)
	}
	defer eng.Close()

	tok, err := loadTokenizer(args.ModelPath, args.TokenizerPath)
	if err != nil {
		log.Fatalf("fucina: tokenizer init failed: %v", err)
	}

	switch {
	case args.Interactive:
		runE4BInteractive(eng, tok, args)
	case args.Prompt != "" || args.PromptFile != "":
		runE4BOneShot(eng, tok, args)
	default:
		runE4BServer(eng, tok, args) // OpenAI-compatible HTTP server (no spec/MTP for E4B)
	}
}

// runE4BOneShot generates a single reply for -p/-f. E4B is an instruction-tuned
// checkpoint, so the prompt is wrapped in the Gemma chat template (it would
// ramble on a bare prompt).
func runE4BOneShot(eng *e4b.Engine, tok *tokenizer.Tokenizer, args CLIArgs) {
	prompt := args.Prompt
	if args.PromptFile != "" {
		data, err := os.ReadFile(args.PromptFile)
		if err != nil {
			log.Fatalf("fucina: cannot read prompt file: %v", err)
		}
		prompt = string(data)
	}
	if prompt == "" {
		log.Fatalf("fucina: empty prompt. Use -p or -f")
	}

	var history []chat.Message
	if args.System != "" {
		history = append(history, chat.Message{Role: "system", Content: args.System})
	}
	history = append(history, chat.Message{Role: "user", Content: prompt})
	// Honor --thinking: open (or pre-close) the model's thought channel in the template,
	// same as the dense -p path.
	thinkOn, _ := thinkSetting(args.Thinking, args.Predict)
	promptToks := tok.Encode(chat.Render(history, thinkOn, "", nil), true, false)
	log.Printf("fucina: prompt has %d tokens", len(promptToks))

	rng := newRNG(args.Seed)
	nToGenerate := args.Predict
	if nToGenerate < 0 {
		nToGenerate = 512
	}

	pfStart := time.Now()
	logits, err := eng.Prefill(promptToks)
	if err != nil {
		log.Fatalf("fucina: prefill failed: %v", err)
	}
	pfElapsed := time.Since(pfStart)
	log.Printf("fucina: prefill %d tokens in %.2fs (%.1f tok/s)",
		len(promptToks), pfElapsed.Seconds(), tps(len(promptToks), pfElapsed))

	fmt.Print(prompt)
	past := append([]int32(nil), promptToks...)
	genStart := time.Now()
	generated := 0
	for i := 0; i < nToGenerate; i++ {
		if logits == nil {
			break
		}
		token, err := sampler.Sample(logits, samplerParams(args), rng, past)
		if err != nil || tok.IsStop(token) {
			break
		}
		fmt.Print(tok.Decode([]int32{token}))
		logits, err = eng.Decode(token)
		if err != nil {
			break
		}
		past = append(past, token)
		generated++
	}
	genElapsed := time.Since(genStart)
	fmt.Println()
	log.Printf("fucina: generated %d tokens in %.2fs (%.1f tok/s)",
		generated, genElapsed.Seconds(), tps(generated, genElapsed))
}

// runE4BInteractive is a multi-turn chat REPL. E4B prefill resets to a FRESH KV
// cache, so each turn re-prefills the whole rendered conversation (correctness
// first; at ~2.5k tok/s prefill this is cheap for chat-length contexts).
// Incremental prefix reuse is a later optimization.
func runE4BInteractive(eng *e4b.Engine, tok *tokenizer.Tokenizer, args CLIArgs) {
	var history []chat.Message
	if args.System != "" {
		history = append(history, chat.Message{Role: "system", Content: args.System})
	}
	rng := newRNG(args.Seed)
	nToGenerate := args.Predict
	if nToGenerate <= 0 {
		nToGenerate = 1 << 20
	}

	// Thinking level is mutable at runtime via /thinking; it seeds from --thinking —
	// same semantics as the dense REPL (see thinkSetting). The level decides whether the
	// model turn opens with the thought channel OPEN (model reasons) or pre-closed
	// (answers directly); the budget force-closes runaway reasoning.
	thinkLevel := args.Thinking
	if thinkLevel == "" {
		thinkLevel = "off"
	}

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1<<20), 1<<20) // allow long pasted prompts
	ctxTokens := int(eng.ContextSize())

	fmt.Fprintf(os.Stderr,
		"fucina: interactive mode (Gemma-4-E4B) — ctx=%d, %d layers, hidden %d, %.2f GB resident, thinking=%s\n%s",
		ctxTokens, eng.NLayers(), eng.HiddenSize(), float64(eng.DeviceBytes())/1e9, thinkLevel, e4bCommandsHelp)

	var lastPFTPS, lastGenTPS float64
	lastHistToks := 0

	for {
		fmt.Fprint(os.Stderr, "\033[1;32m> \033[0m")
		if !scanner.Scan() {
			fmt.Fprintln(os.Stderr, "\nfucina: bye")
			return
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		switch input {
		case "/quit", "/exit", "/q":
			fmt.Fprintln(os.Stderr, "fucina: bye")
			return
		case "/help", "/h", "/?", "/commands":
			fmt.Fprintf(os.Stderr, "fucina: commands —\n%s", e4bCommandsHelp)
			continue
		case "/reset", "/clear":
			history = history[:0]
			if args.System != "" {
				history = append(history, chat.Message{Role: "system", Content: args.System})
			}
			eng.Reset()
			lastHistToks = 0
			fmt.Fprintln(os.Stderr, "fucina: conversation cleared")
			continue
		case "/stats":
			fmt.Fprintf(os.Stderr,
				"fucina: context %d/%d tokens; last turn prefill %.1f tok/s, decode %.1f tok/s "+
					"(E4B re-prefills each turn — no KV prefix reuse)\n",
				lastHistToks, ctxTokens, lastPFTPS, lastGenTPS)
			continue
		}
		// /thinking LEVEL — set the reasoning channel level for subsequent turns.
		if strings.HasPrefix(input, "/thinking") || strings.HasPrefix(input, "/think") {
			parts := strings.Fields(input)
			if len(parts) < 2 {
				on, budget := thinkSetting(thinkLevel, nToGenerate)
				fmt.Fprintf(os.Stderr, "fucina: thinking=%s (on=%v, budget=%d). "+
					"Usage: /thinking off|on|low|medium|high|xhigh\n", thinkLevel, on, budget)
				continue
			}
			lvl := strings.ToLower(parts[1])
			if !validThinkLevel(lvl) {
				fmt.Fprintf(os.Stderr, "fucina: unknown thinking level %q — "+
					"use off|on|low|medium|high|xhigh\n", parts[1])
				continue
			}
			thinkLevel = lvl
			on, budget := thinkSetting(thinkLevel, nToGenerate)
			fmt.Fprintf(os.Stderr, "fucina: thinking=%s (on=%v, budget=%d tokens)\n", thinkLevel, on, budget)
			continue
		}
		// A leading-slash token that matched no command is a typo, not chat input.
		if looksLikeCommand(input) {
			fmt.Fprintf(os.Stderr, "fucina: unknown command %q — type /help for the list\n",
				strings.Fields(input)[0])
			continue
		}

		thinkOn, thinkBudget := thinkSetting(thinkLevel, nToGenerate)
		history = append(history, chat.Message{Role: "user", Content: input})
		promptToks := tok.Encode(chat.Render(history, thinkOn, "", nil), true, false)
		lastHistToks = len(promptToks)
		if len(promptToks) > ctxTokens-64 {
			fmt.Fprintf(os.Stderr,
				"\033[33mfucina: warning: prompt is %d tokens, near context limit %d\033[0m\n",
				len(promptToks), ctxTokens)
		}

		eng.Reset()
		pfStart := time.Now()
		logits, err := eng.Prefill(promptToks)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fucina: prefill error: %v\n", err)
			history = history[:len(history)-1] // undo user turn
			continue
		}
		pfElapsed := time.Since(pfStart)
		lastPFTPS = tps(len(promptToks), pfElapsed)
		fmt.Fprintf(os.Stderr,
			"\033[2mfucina: prefill %d tokens %.2fs %.1f tok/s\033[0m\n",
			len(promptToks), pfElapsed.Seconds(), lastPFTPS)

		fmt.Fprint(os.Stdout, "\033[1;34mAssistant:\033[0m ")
		var replyBuf strings.Builder
		past := append([]int32(nil), promptToks...)
		genStart := time.Now()
		generated := 0
		inCh, thinkClosed, thinkToks := false, false, 0
		for i := 0; i < nToGenerate; i++ {
			if logits == nil {
				break
			}
			token, err := sampler.Sample(logits, samplerParams(args), rng, past)
			if err != nil || tok.IsStop(token) {
				break
			}
			switch {
			case tok.ChannelOpen >= 0 && token == tok.ChannelOpen:
				inCh = true
			case tok.ChannelEnd >= 0 && token == tok.ChannelEnd:
				inCh = false
			default:
				piece := tok.Decode([]int32{token})
				if inCh {
					fmt.Printf("\033[2m%s\033[0m", piece) // dim reasoning channel
					thinkToks++
				} else {
					fmt.Print(piece)
				}
				replyBuf.WriteString(piece)
			}
			logits, err = eng.Decode(token)
			if err != nil {
				break
			}
			past = append(past, token)
			generated++

			// Thinking budget: force-close the thought channel so the model answers,
			// mirroring the dense path (genWithThinking). Inject <channel|>, decode it,
			// and stop counting — render nothing for the marker.
			if inCh && !thinkClosed && thinkBudget > 0 && thinkToks >= thinkBudget && tok.ChannelEnd >= 0 {
				logits, err = eng.Decode(tok.ChannelEnd)
				if err != nil {
					break
				}
				past = append(past, tok.ChannelEnd)
				generated++
				inCh, thinkClosed = false, true
			}
		}
		genElapsed := time.Since(genStart)
		lastGenTPS = tps(generated, genElapsed)
		fmt.Println()
		fmt.Fprintf(os.Stderr,
			"\033[2mfucina: generated %d tokens %.2fs %.1f tok/s\033[0m\n\n",
			generated, genElapsed.Seconds(), tps(generated, genElapsed))

		_, reply := chat.SplitReasoning(replyBuf.String())
		history = append(history, chat.Message{Role: "assistant", Content: strings.TrimSpace(reply)})
	}
}

// tps is tokens/second, guarding the zero-duration case.
func tps(n int, d time.Duration) float64 {
	if d.Seconds() <= 0 {
		return 0
	}
	return float64(n) / d.Seconds()
}
