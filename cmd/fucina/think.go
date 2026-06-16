package main

import (
	"strings"

	"github.com/hikmaai-io/fucina/internal/engine/cuda"
	gemserver "github.com/hikmaai-io/fucina/internal/server"
	"github.com/hikmaai-io/fucina/internal/tokenizer"
)

// thinkSetting resolves a thinking level (off|on|low|medium|high|xhigh) to whether
// the gemma-4 reasoning channel is opened and the per-turn token budget that force-
// closes it. Mirrors the server's level→enable + think-budget intent so the CLI
// behaves like /v1/chat/completions. budget 0 means "no cap" (only meaningful when
// on=false). "on" derives an auto budget of half the generation cap.
func thinkSetting(level string, maxNew int) (on bool, budget int) {
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "", "off", "false", "no", "0":
		return false, 0
	case "low":
		return true, 256
	case "medium", "mid":
		return true, 1024
	case "high":
		return true, 4096
	case "xhigh", "max":
		return true, 16384
	default: // "on", "true", "yes", "1"
		b := maxNew / 2
		if b < 512 {
			b = 512
		}
		return true, b
	}
}

// validThinkLevel reports whether s is an accepted /thinking argument.
func validThinkLevel(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "on", "off", "low", "medium", "mid", "high", "xhigh", "max":
		return true
	}
	return false
}

// genWithThinking runs speculative decoding with a thinking-token budget, mirroring
// the server's runSpec (server.go): it tracks the <|channel>thought…<channel|> span
// and, when reasoning exceeds `budget` tokens, force-closes the channel — Rewind to
// the last emitted token, then Decode the <channel|> token — so the model stops
// thinking and answers. The KV prefix cache is kept in lockstep with the engine
// exactly as runSpec does (trim accepted-but-unemitted tokens before the injected
// <channel|>, or the next turn reuses a corrupted prefix). Streams via emit(piece,
// inThought); the channel marker tokens and stop tokens are not rendered. The caller
// owns the kv lock for the whole call. budget<=0 disables the cap (free thinking).
func genWithThinking(eng *cuda.Engine, kv *gemserver.KVCache, tok *tokenizer.Tokenizer,
	firstLogits []float32, maxNew int, stops []int32, draftK int,
	temp float32, topK int, topP, minP, repPen float32, seed uint64, budget int,
	emit func(piece string, inThought bool)) (all []int32, nAccepted int, err error) {

	chOpen, chEnd := tok.ChannelOpen, tok.ChannelEnd
	inCh, thinkClosed := false, false
	thinkToks := 0
	remaining := maxNew
	logits := firstLogits

	for remaining > 0 {
		hitBudget := false
		history := kv.CurrentTokens()
		baseN := eng.NTokens()
		toks, acc, gerr := eng.GenerateSpecStream(history, logits, remaining, stops,
			draftK, temp, topK, topP, minP, repPen, seed,
			func(t int32) bool {
				switch {
				case chOpen >= 0 && t == chOpen:
					inCh = true
				case chEnd >= 0 && t == chEnd:
					inCh = false
				case inCh:
					thinkToks++
				}
				if tok.IsStop(t) {
					return true // control token: stop, do not render
				}
				if t != chOpen && t != chEnd {
					emit(tok.Decode([]int32{t}), inCh)
				}
				if inCh && !thinkClosed && budget > 0 && thinkToks >= budget {
					hitBudget = true
					return true
				}
				return false
			})
		nAccepted += acc

		// Sync the prefix cache with the tokens actually committed to the engine KV
		// (a verify pass commits the whole accepted run before per-token emission, so
		// when the budget stops the callback mid-run the engine holds accepted-but-
		// unemitted tokens beyond what we render). appended==committed here.
		committed := eng.NTokens() - baseN
		appended := committed
		if appended > len(toks) {
			appended = len(toks)
		}
		for i := 0; i < appended; i++ {
			kv.AppendDecoded(toks[i])
		}
		all = append(all, toks...)
		remaining -= len(toks)
		seed++ // a resumed round must not replay the same draws
		if gerr != nil {
			return all, nAccepted, gerr
		}

		if hitBudget && chEnd >= 0 && remaining > 0 {
			// Force-close the thought channel so the model answers. Trim the accepted-
			// but-unemitted tail first: the injected <channel|> must land exactly after
			// the last recorded token, or cachedTokens and the KV CONTENT diverge and
			// the next turn reuses a corrupted prefix (see server.runSpec).
			if !eng.Rewind(baseN + appended) {
				return all, nAccepted, nil
			}
			lg, derr := eng.Decode(chEnd)
			if derr != nil {
				return all, nAccepted, derr
			}
			kv.AppendDecoded(chEnd)
			all = append(all, chEnd)
			remaining--
			inCh, thinkClosed = false, true
			logits = lg
			continue
		}
		// The engine stopped on its own (stop token or max_new exhausted).
		return all, nAccepted, nil
	}
	return all, nAccepted, nil
}
