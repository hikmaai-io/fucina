package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/hikmaai-io/fucina/internal/chat"
	"github.com/hikmaai-io/fucina/internal/engine/cuda"
	"github.com/hikmaai-io/fucina/internal/server/batch"
	"github.com/hikmaai-io/fucina/internal/session"
	"github.com/hikmaai-io/fucina/internal/tokenizer"
)

// pagedCommandsHelp is the command list for the paged REPL. Unlike the dense
// (gemma single-flight) REPL there is no cross-turn KV prefix cache to report,
// so /stats shows the running context/turn counts instead of a hit rate.
const pagedCommandsHelp = "  /thinking LEVEL  set reasoning: off|on|low|medium|high|xhigh\n" +
	"  /reset           clear conversation\n" +
	"  /save FILE       save the conversation + engine state to disk\n" +
	"  /load FILE       resume a saved conversation (restores on the next turn)\n" +
	"  /stats           show context usage\n" +
	"  /help            show this help\n" +
	"  /quit            exit (or Ctrl-D)\n"

// selectDialect picks the chat wire format from the tokenizer vocabulary: a
// ChatML vocab (<|im_start|> present) means the Qwen template + XML tool calls,
// otherwise gemma-4. This mirrors the server's startup selection (server.go) so
// REPL prompts are byte-identical to served prompts — never chosen from a flag.
func selectDialect(tok *tokenizer.Tokenizer) chat.Dialect {
	if tok != nil && tok.HasToken("<|im_start|>") {
		return chat.Qwen
	}
	return chat.Gemma
}

// runInteractivePaged is the REPL for engines that are served ONLY through the
// paged multi-sequence path — the Qwen3 family and the Qwen3.5 hybrid. Their
// single-flight prefill entry points deliberately decline (they are gemma-layout
// only; running them on Qwen weights corrupts the CUDA context), so the dense
// REPL's kv.Prefill / eng.Decode loop fails at "prefill failed". Here each turn
// runs as one continuous-batching request instead: SeqAdd prefills the rendered
// conversation into a fresh paged slot and returns the first sampled token, then
// StepBatch advances that single slot one token at a time. Sampling is on-device
// per the slot's SeqParams (temp/top_k/top_p/min_p/seed), exactly like a served
// request. The chat template is chosen from the vocab (Qwen ChatML vs gemma).
//
// There is no cross-turn KV reuse (the hybrid geometry has no effective prefix
// cache): every turn re-prefills the full history, which is correct and fast
// enough for interactive use. Repeat-penalty is not applied on the paged path
// (the on-device sampler does not take it); temperature<=0 is exact greedy.
func runInteractivePaged(eng *cuda.Engine, tok *tokenizer.Tokenizer, args CLIArgs) {
	dialect := selectDialect(tok)
	var history []chat.RichMessage
	// pendingSession is a /load-ed disk snapshot awaiting its first matching
	// turn; every turn whose prompt extends it restores it into a fresh slot
	// and prefills only the suffix (the restored prefix costs zero prefill).
	var pendingSession *session.Snapshot
	jtrace, err := newJSpaceTracer(eng, tok, args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fucina: %v\n", err)
		return
	}
	if jtrace != nil {
		defer func() {
			if err := jtrace.Close(); err != nil {
				fmt.Fprintf(os.Stderr, "fucina: close J-space trace: %v\n", err)
			}
		}()
	}

	nToGenerate := args.Predict
	if nToGenerate <= 0 {
		nToGenerate = 1 << 20
	}

	thinkLevel := args.Thinking
	if thinkLevel == "" {
		thinkLevel = "off"
	}

	if args.System != "" {
		history = append(history, chat.RichMessage{Role: "system", Content: args.System})
	}

	scanner := bufio.NewScanner(os.Stdin)
	// Long pasted prompts overflow bufio.Scanner's default 64 KiB line cap.
	scanner.Buffer(make([]byte, 0, 1024*1024), 8*1024*1024)
	ctxTokens := int(eng.ContextSize())

	commandsHelp := pagedCommandsHelp
	if jtrace != nil {
		commandsHelp += "  /jdump           print the latest J-space readout\n"
		if args.JSpaceDebug {
			commandsHelp += "  /jsteer \" TOKEN\" STRENGTH [LAYERS]  steer future forwards\n" +
				"  /jclear          clear J-space steering\n"
		}
	}
	fmt.Fprintf(os.Stderr,
		"fucina: interactive mode (%s dialect, paged multi-seq) — ctx=%d, thinking=%s\n%s\n",
		dialect.Name(), ctxTokens, thinkLevel, commandsHelp)

	turn := uint64(0)

	for {
		fmt.Fprint(os.Stderr, "\033[1;32m> \033[0m") // green prompt
		if !scanner.Scan() {
			fmt.Fprintln(os.Stderr, "\nfucina: bye")
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		// Slash commands
		if handled, jerr := jtrace.handleCommand(input); handled {
			if jerr != nil {
				fmt.Fprintf(os.Stderr, "fucina: J-space: %v\n", jerr)
			}
			continue
		}
		switch input {
		case "/quit", "/exit", "/q":
			fmt.Fprintln(os.Stderr, "fucina: bye")
			return
		case "/help", "/h", "/?", "/commands":
			fmt.Fprintf(os.Stderr, "fucina: commands —\n%s", commandsHelp)
			continue
		case "/reset", "/clear":
			history = history[:0]
			pendingSession = nil
			if args.System != "" {
				history = append(history, chat.RichMessage{Role: "system", Content: args.System})
			}
			fmt.Fprintln(os.Stderr, "fucina: conversation cleared")
			continue
		case "/stats":
			promptToks := tok.Encode(dialect.Render(history, nil, false), true, false)
			fmt.Fprintf(os.Stderr,
				"fucina: context — %d/%d tokens, %d messages in history\n",
				len(promptToks), ctxTokens, len(history))
			continue
		}
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
			fmt.Fprintf(os.Stderr, "fucina: thinking=%s (on=%v, budget=%d tokens)\n",
				thinkLevel, on, budget)
			continue
		}
		// /save FILE | /load FILE — session persistence across process restarts.
		if save, file, ok := parseSessionCommand(input); ok {
			if file == "" {
				fmt.Fprintln(os.Stderr, "fucina: usage: /save <file> | /load <file>")
				continue
			}
			if save {
				if err := pagedSessionSave(eng, tok, dialect, history, args, thinkLevel, file); err != nil {
					fmt.Fprintf(os.Stderr, "fucina: save: %v\n", err)
				} else {
					fmt.Fprintf(os.Stderr, "fucina: session saved to %s\n", file)
				}
			} else if snap, hist, think, err := pagedSessionLoad(eng, args, file); err != nil {
				fmt.Fprintf(os.Stderr, "fucina: load: %v\n", err)
			} else {
				pendingSession = snap
				history = hist
				if think != "" {
					thinkLevel = think
				}
				fmt.Fprintf(os.Stderr,
					"fucina: session loaded from %s — %d tokens, %d messages; restores on the next turn\n",
					file, len(snap.Tokens), len(history))
			}
			continue
		}

		if looksLikeCommand(input) {
			fmt.Fprintf(os.Stderr, "fucina: unknown command %q — type /help for the list\n",
				strings.Fields(input)[0])
			continue
		}

		thinkOn, thinkBudget := thinkSetting(thinkLevel, nToGenerate)
		history = append(history, chat.RichMessage{Role: "user", Content: input})
		promptStr := dialect.Render(history, nil, thinkOn)
		promptToks := tok.Encode(promptStr, true, false)

		if len(promptToks) > ctxTokens-64 {
			fmt.Fprintf(os.Stderr,
				"\033[33mfucina: warning: prompt is %d tokens, near context limit %d\033[0m\n",
				len(promptToks), ctxTokens)
		}

		// Per-turn seed: distinct per turn so a fixed --seed does not replay the
		// same sampling stream every turn (splitmix64 golden-ratio gamma stride).
		seed := uint64(time.Now().UnixNano())
		if args.Seed >= 0 {
			seed = uint64(args.Seed) + turn*0x9e3779b97f4a7c15
		}
		turn++

		params := batch.SeqParams{
			Temperature: float32(args.Temperature),
			TopK:        args.TopK,
			TopP:        float32(args.TopP),
			MinP:        float32(args.MinP),
			Seed:        seed,
		}

		pfStart := time.Now()
		slot, first, restored, err := pagedSessionAdmit(eng, pendingSession, promptToks, params)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fucina: prefill error: %v\n", err)
			history = history[:len(history)-1] // undo user turn
			continue
		}
		pfElapsed := time.Since(pfStart)
		pfNew := len(promptToks) - restored
		pfTPS := 0.0
		if pfElapsed.Seconds() > 0 {
			pfTPS = float64(pfNew) / pfElapsed.Seconds()
		}
		fmt.Fprintf(os.Stderr,
			"\033[2mfucina: prefill %d tokens (%d from session, %d new) %.2fs %.1f tok/s\033[0m\n",
			len(promptToks), restored, pfNew, pfElapsed.Seconds(), pfTPS)

		fmt.Fprint(os.Stdout, "\033[1;34mAssistant:\033[0m ")

		genStart := time.Now()
		reply, generated, gerr := generatePaged(eng, tok, dialect, slot, first,
			nToGenerate, thinkOn, thinkBudget, jtrace, turn-1,
			promptToks[len(promptToks)-1], len(promptToks)-1)
		eng.SeqRemove(slot)
		fmt.Println() // newline after the streamed reply

		if gerr != nil {
			fmt.Fprintf(os.Stderr, "fucina: generate error: %v\n", gerr)
			history = history[:len(history)-1] // undo user turn (no clean reply to keep)
			continue
		}

		genElapsed := time.Since(genStart)
		genTPS := 0.0
		if genElapsed.Seconds() > 0 {
			genTPS = float64(generated) / genElapsed.Seconds()
		}
		fmt.Fprintf(os.Stderr,
			"\033[2mfucina: generated %d tokens %.2fs %.1f tok/s\033[0m\n\n",
			generated, genElapsed.Seconds(), genTPS)

		history = append(history, chat.RichMessage{Role: "assistant", Content: strings.TrimSpace(reply)})
	}
}

// generatePaged decodes one assistant turn on a single paged slot. It streams
// tokens to stdout as they are sampled — the reasoning span (dialect channel
// markers) is dimmed, the answer is shown normally — and returns the answer
// text (reasoning stripped) for the history. Text is decoded incrementally with
// the server's whole-slice-decode/emit-new-suffix trick so multi-byte UTF-8 is
// never split across tokens. When thinking runs past thinkBudget reasoning
// tokens the channel is force-closed by feeding the close marker, so the model
// stops thinking and answers (mirrors the server's runSpec budget).
func generatePaged(eng *cuda.Engine, tok *tokenizer.Tokenizer, dialect chat.Dialect,
	slot int, first int32, nToGenerate int, thinkOn bool, thinkBudget int,
	jtrace *jspaceTracer, turn uint64, sourceToken int32,
	sourcePosition int) (reply string, generated int, err error) {

	chOpen, chEnd := tok.ChannelOpen, tok.ChannelEnd
	// Qwen renders the reasoning opener (<think>\n) INTO the prompt, so with
	// thinking on generation starts already inside the reasoning block and never
	// emits an open marker; gemma opens its channel with a generated token.
	inChannel := dialect.StartsInReasoning(thinkOn)
	labelPending := inChannel && dialect.HasReasoningLabel() // gemma's "thought\n" label line
	thinkClosed := false
	reasonToks := 0

	var contentIDs, reasonIDs []int32
	emittedContent, emittedReason := "", ""

	token := first
	for generated < nToGenerate {
		// Force-close an over-budget reasoning block: replace the next fed token
		// with the channel-close marker so the model leaves the thought channel
		// and answers instead of thinking forever.
		if inChannel && thinkOn && !thinkClosed && thinkBudget > 0 &&
			reasonToks >= thinkBudget && chEnd >= 0 {
			token = chEnd
		}

		if tok.IsStop(token) {
			break
		}
		if err := jtrace.Record(turn, generated, sourcePosition, sourceToken, token); err != nil {
			return emittedContent, generated, err
		}

		switch {
		case chOpen >= 0 && token == chOpen:
			inChannel = true
			labelPending = dialect.HasReasoningLabel()
		case chEnd >= 0 && token == chEnd:
			inChannel = false
			thinkClosed = true
		case inChannel:
			reasonToks++
			reasonIDs = append(reasonIDs, token)
			full := tok.Decode(reasonIDs)
			if labelPending { // skip gemma's "thought\n" label line — never rendered
				if nl := strings.IndexByte(full, '\n'); nl >= 0 {
					labelPending = false
					emittedReason = full[:nl+1]
				}
			}
			if !labelPending && len(full) > len(emittedReason) {
				fmt.Printf("\033[2m%s\033[0m", full[len(emittedReason):]) // dim reasoning
				emittedReason = full
			}
		default:
			// Swallow leading whitespace (e.g. the "\n\n" the Qwen template puts
			// between </think> and the answer) so the reply starts clean.
			if len(contentIDs) == 0 && strings.TrimSpace(tok.Decode([]int32{token})) == "" {
				break
			}
			contentIDs = append(contentIDs, token)
			full := tok.Decode(contentIDs)
			if len(full) > len(emittedContent) {
				fmt.Print(full[len(emittedContent):])
				emittedContent = full
			}
		}

		out, serr := eng.StepBatch([]int32{int32(slot)}, []int32{token})
		if serr != nil {
			return emittedContent, generated, serr
		}
		generated++
		sourceToken = token
		sourcePosition++
		token = out[0]
	}

	return emittedContent, generated, nil
}
