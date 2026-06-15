package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/hikmaai-io/fucina/internal/chat"
	"github.com/hikmaai-io/fucina/internal/engine/cuda"
	"github.com/hikmaai-io/fucina/internal/sampler"
	gemserver "github.com/hikmaai-io/fucina/internal/server"
	"github.com/hikmaai-io/fucina/internal/tokenizer"
)

// ─── Interactive REPL ────────────────────────────────────────────

// runInteractive runs a multi-turn chat REPL with prefix-reuse KV caching.
// Each turn:
//  1. Build the full Gemma chat-template prompt from conversation history.
//  2. Ask KVCache.Prefill — which reuses the shared prefix from last turn
//     (the entire prior conversation) and only computes the new user message.
//  3. Generate: speculative decoding (MTP/prompt-lookup, one blocking engine
//     call, reply rendered as a block) when eligible, else stream tokens
//     one-by-one with the host sampler.
//  4. Append the completed reply to history so the NEXT turn can reuse it.
func runInteractive(eng *cuda.Engine, tok *tokenizer.Tokenizer, args CLIArgs) {
	var history []chat.Message

	kv := gemserver.NewKVCache(eng)
	rng := newRNG(args.Seed)
	specTurn := uint64(0) // per-session turn counter for spec seed derivation

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
		"fucina: interactive mode (Gemma 4 12B-IT) — ctx=%d\n"+
			"  /reset  clear conversation\n"+
			"  /stats  show KV cache hit rate\n"+
			"  /quit   exit (or Ctrl-D)\n\n",
		ctxTokens)

	for {
		fmt.Fprint(os.Stderr, "\033[1;32m> \033[0m") // green prompt
		if !scanner.Scan() {
			// Ctrl-D / EOF
			fmt.Fprintln(os.Stderr, "\nfucina: bye")
			break
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
		case "/reset", "/clear":
			history = history[:0]
			if args.System != "" {
				history = append(history, chat.Message{Role: "system", Content: args.System})
			}
			// Reset through the KVCache so the engine KV and the cache's prefix
			// bookkeeping are cleared in LOCKSTEP. Calling eng.Reset() directly
			// here is exactly the bug this replaces: cachedTokens kept the old
			// conversation, the next turn's Prefill "reused" the shared
			// chat-template prefix that no longer existed in the empty engine,
			// prefilled only the suffix at wrong positions, and the model
			// replied with word salad.
			kv.Lock()
			kv.Reset()
			kv.Unlock()
			fmt.Fprintln(os.Stderr, "fucina: conversation cleared")
			continue
		case "/stats":
			hits, misses, rate := kv.Stats()
			fmt.Fprintf(os.Stderr,
				"fucina: KV cache — hits=%d misses=%d token_hit_rate=%.1f%% cached=%d/%d\n",
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
				"\033[33mfucina: warning: prompt is %d tokens, near context limit %d\033[0m\n",
				len(promptToks), ctxTokens)
		}

		// Cache-aware prefill — holds kv lock through generation.
		kv.Lock()
		pfStart := time.Now()
		pf, err := kv.Prefill(promptToks)
		if err != nil {
			kv.Unlock()
			fmt.Fprintf(os.Stderr, "fucina: prefill error: %v\n", err)
			history = history[:len(history)-1] // undo user turn
			continue
		}
		pfElapsed := time.Since(pfStart)
		pfTPS := 0.0
		if pfElapsed.Seconds() > 0 && pf.NewTokens > 0 {
			pfTPS = float64(pf.NewTokens) / pfElapsed.Seconds()
		}
		fmt.Fprintf(os.Stderr,
			"\033[2mfucina: prefill %d tokens (%d cached, %d new) %.2fs %.1f tok/s\033[0m\n",
			pf.PromptTokens, pf.ReusedTokens, pf.NewTokens, pfElapsed.Seconds(), pfTPS)

		// The speculative fast-path is a single blocking engine call: one weight
		// pass per [g, draft...] with MTP/prompt-lookup drafting on-device, same
		// output distribution as plain decode, and no 1 MiB logits D2H copy +
		// O(262k) host top-k per token. The REPL has no text stop-strings and
		// needs no mid-flight cancellation, so eligibility is just spec-on + no
		// (repeat-penalty included — the engine applies it on-GPU). Otherwise
		// fall back to the per-token loop.
		useSpec := args.Spec

		var reply string
		genStart := time.Now()
		generated := 0
		specStats := "" // extra spec diagnostics appended to the stats line

		// Print assistant prefix so the reply is visually distinct.
		fmt.Fprint(os.Stdout, "\033[1;34mAssistant:\033[0m ")

		if useSpec {
			stops := []int32{tok.EOS, tok.EndOfTurn}
			// Per-turn seed: random when --seed < 0; otherwise derived from
			// --seed but DISTINCT per turn (reusing the seed verbatim would
			// replay the same sampling stream every turn). The golden-ratio
			// gamma is the splitmix64 stream increment — deterministic,
			// well-separated streams, reproducible from --seed.
			seed := uint64(time.Now().UnixNano())
			if args.Seed >= 0 {
				seed = uint64(args.Seed) + specTurn*0x9e3779b97f4a7c15
			}
			specTurn++

			cacheToks := kv.CurrentTokens() // full cached sequence, under kv lock
			// Stream each token as the spec loop emits it (between verify steps),
			// so the reply renders incrementally at full speculative speed.
			// EOS/<turn|> are control tokens: stop without rendering them.
			var replyBuf strings.Builder
			toks, nAccepted, err := eng.GenerateSpecStream(cacheToks, pf.Logits,
				nToGenerate, stops, args.DraftK, float32(args.Temperature),
				args.TopK, float32(args.TopP), float32(args.MinP),
				float32(args.RepeatPenalty), seed,
				func(t int32) bool {
					if tok.IsStop(t) {
						return true
					}
					piece := tok.Decode([]int32{t})
					fmt.Print(piece)
					replyBuf.WriteString(piece)
					return false
				})
			if err != nil {
				kv.Unlock()
				fmt.Println()
				fmt.Fprintf(os.Stderr, "fucina: spec generate error: %v\n", err)
				history = history[:len(history)-1] // undo user turn
				continue
			}
			// Sync the prefix cache with the tokens actually committed to the
			// engine KV (a trailing stop token may be emitted but not forwarded)
			// so the NEXT turn's prefill reuses this whole turn.
			committed := eng.NTokens() - pf.PromptTokens
			for i := 0; i < committed && i < len(toks); i++ {
				kv.AppendDecoded(toks[i])
			}
			generated = len(toks)
			reply = replyBuf.String()
			specStats = fmt.Sprintf(", %d drafts accepted (avg %.2f tokens/step, draft-k=%d)",
				nAccepted, float64(len(toks))/float64(max(1, len(toks)-nAccepted)), args.DraftK)
		} else {
			// Per-token decode loop with the host sampler: required for
			// repeat-penalty (it edits logits using the token history) or when
			// --spec=false. Streams tokens to stdout as they are sampled.
			var replyBuf strings.Builder
			logits := pf.Logits
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

				logits, err = eng.Decode(token)
				if err != nil {
					break
				}
				// Record the token in the prefix cache only AFTER Decode commits
				// it to the engine KV (AppendDecoded's contract): if Decode fails,
				// the token never made it into the engine and recording it first
				// would leave the bookkeeping one token ahead of the engine.
				kv.AppendDecoded(token)
				generated++
			}
			reply = replyBuf.String()
		}
		kv.Unlock()

		genElapsed := time.Since(genStart)
		genTPS := 0.0
		if genElapsed.Seconds() > 0 {
			genTPS = float64(generated) / genElapsed.Seconds()
		}
		fmt.Println() // newline after reply
		fmt.Fprintf(os.Stderr,
			"\033[2mfucina: generated %d tokens %.2fs %.1f tok/s%s\033[0m\n\n",
			generated, genElapsed.Seconds(), genTPS, specStats)

		// Add completed assistant reply to history so the NEXT turn's prompt
		// includes it verbatim and the KVCache can reuse the whole sequence.
		history = append(history, chat.Message{Role: "assistant", Content: strings.TrimSpace(reply)})
	}
}
