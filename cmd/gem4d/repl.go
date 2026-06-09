package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/mauromedda/gem4d/internal/chat"
	"github.com/mauromedda/gem4d/internal/engine/cuda"
	"github.com/mauromedda/gem4d/internal/sampler"
	gemserver "github.com/mauromedda/gem4d/internal/server"
	"github.com/mauromedda/gem4d/internal/tokenizer"
)

// ─── Interactive REPL ────────────────────────────────────────────

// runInteractive runs a multi-turn chat REPL with prefix-reuse KV caching.
// Each turn:
//  1. Build the full Gemma chat-template prompt from conversation history.
//  2. Ask KVCache.Prefill — which reuses the shared prefix from last turn
//     (the entire prior conversation) and only computes the new user message.
//  3. Stream tokens to stdout as they are sampled.
//  4. Append the completed reply to history so the NEXT turn can reuse it.
func runInteractive(eng *cuda.Engine, tok *tokenizer.Tokenizer, args CLIArgs) {
	var history []chat.Message

	kv := gemserver.NewKVCache(eng)
	rng := newRNG(args.Seed)

	nToGenerate := args.Predict
	if nToGenerate <= 0 {
		nToGenerate = 1 << 20
	}

	// The gemma-4 chat template lives in internal/chat. The REPL always runs with
	// thinking OFF, so chat.Render opens each model turn with an already-closed
	// empty thought channel (<|turn>model\n<|channel>thought\n<channel|>) — turns
	// are delimited by <|turn>/<turn|> (the vocab has NO <start_of_turn>).

	// Optional system prompt injected as the first turn (role "system").
	if args.System != "" {
		history = append(history, chat.Message{Role: "system", Content: args.System})
	}

	scanner := bufio.NewScanner(os.Stdin)
	ctxTokens := int(eng.ContextSize())

	fmt.Fprintf(os.Stderr,
		"gem4d: interactive mode (Gemma 4 12B-IT) — ctx=%d\n"+
			"  /reset  clear conversation\n"+
			"  /stats  show KV cache hit rate\n"+
			"  /quit   exit (or Ctrl-D)\n\n",
		ctxTokens)

	for {
		fmt.Fprint(os.Stderr, "\033[1;32m> \033[0m") // green prompt
		if !scanner.Scan() {
			// Ctrl-D / EOF
			fmt.Fprintln(os.Stderr, "\ngem4d: bye")
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		// Slash commands
		switch input {
		case "/quit", "/exit", "/q":
			fmt.Fprintln(os.Stderr, "gem4d: bye")
			return
		case "/reset", "/clear":
			history = history[:0]
			if args.System != "" {
				history = append(history, chat.Message{Role: "system", Content: args.System})
			}
			kv.Lock()
			eng.Reset()
			kv.Unlock()
			fmt.Fprintln(os.Stderr, "gem4d: conversation cleared")
			continue
		case "/stats":
			hits, misses, rate := kv.Stats()
			fmt.Fprintf(os.Stderr,
				"gem4d: KV cache — hits=%d misses=%d token_hit_rate=%.1f%% cached=%d/%d\n",
				hits, misses, rate*100, eng.NTokens(), ctxTokens)
			continue
		}

		// Add user turn and build the full prompt.
		history = append(history, chat.Message{Role: "user", Content: input})
		promptStr := chat.Render(history, false, "", nil)
		promptToks := tok.Encode(promptStr, true, false)

		// Warn if we are close to the context limit.
		if len(promptToks) > ctxTokens-64 {
			fmt.Fprintf(os.Stderr,
				"\033[33mgem4d: warning: prompt is %d tokens, near context limit %d\033[0m\n",
				len(promptToks), ctxTokens)
		}

		// Cache-aware prefill — holds kv lock through generation.
		kv.Lock()
		pfStart := time.Now()
		pf, err := kv.Prefill(promptToks)
		if err != nil {
			kv.Unlock()
			fmt.Fprintf(os.Stderr, "gem4d: prefill error: %v\n", err)
			history = history[:len(history)-1] // undo user turn
			continue
		}
		pfElapsed := time.Since(pfStart)
		pfTPS := 0.0
		if pfElapsed.Seconds() > 0 && pf.NewTokens > 0 {
			pfTPS = float64(pf.NewTokens) / pfElapsed.Seconds()
		}
		fmt.Fprintf(os.Stderr,
			"\033[2mgem4d: prefill %d tokens (%d cached, %d new) %.2fs %.1f tok/s\033[0m\n",
			pf.PromptTokens, pf.ReusedTokens, pf.NewTokens, pfElapsed.Seconds(), pfTPS)

		// Stream generation token-by-token.
		var replyBuf strings.Builder
		logits := pf.Logits
		genStart := time.Now()
		generated := 0

		// Print assistant prefix so the reply is visually distinct.
		fmt.Fprint(os.Stdout, "\033[1;34mAssistant:\033[0m ")

		for i := 0; i < nToGenerate; i++ {
			if logits == nil {
				break
			}
			token, err := sampler.Sample(logits, samplerParams(args), rng, nil)
			// Stop on EOS or end-of-turn (<turn|>); these are control tokens
			// and must not be rendered.
			if err != nil || tok.IsStop(token) {
				break
			}
			piece := tok.Decode([]int32{token})
			fmt.Print(piece)
			replyBuf.WriteString(piece)
			kv.AppendDecoded(token)

			logits, err = eng.Decode(token)
			if err != nil {
				break
			}
			generated++
		}
		kv.Unlock()

		genElapsed := time.Since(genStart)
		genTPS := 0.0
		if genElapsed.Seconds() > 0 {
			genTPS = float64(generated) / genElapsed.Seconds()
		}
		fmt.Println() // newline after reply
		fmt.Fprintf(os.Stderr,
			"\033[2mgem4d: generated %d tokens %.2fs %.1f tok/s\033[0m\n\n",
			generated, genElapsed.Seconds(), genTPS)

		// Add completed assistant reply to history so the NEXT turn's prompt
		// includes it verbatim and the KVCache can reuse the whole sequence.
		history = append(history, chat.Message{Role: "assistant", Content: strings.TrimSpace(replyBuf.String())})
	}
}
