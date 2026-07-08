package main

import (
	"fmt"
	"log"
	"os"
	"time"

	"github.com/hikmaai-io/fucina/internal/engine/cuda"
	"github.com/hikmaai-io/fucina/internal/sampler"
	"github.com/hikmaai-io/fucina/internal/server/batch"
	"github.com/hikmaai-io/fucina/internal/tokenizer"
)

// ─── One-shot prompt ───────────────────────────────────────────────

// runOneShot runs the single-prompt generation path: it resolves the prompt
// (from -p or -f, optionally prefixed with the system prompt), tokenizes it,
// and either runs prompt-lookup speculative decode or a plain prefill+decode
// loop, streaming the result to stdout.
func runOneShot(eng *cuda.Engine, tok *tokenizer.Tokenizer, args CLIArgs) {
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
	if args.System != "" {
		prompt = fmt.Sprintf("System: %s\n\n%s", args.System, prompt)
	}

	// Tokenize
	tokens := tok.Encode(prompt, true, false)
	log.Printf("fucina: prompt has %d tokens", len(tokens))

	// Paged-only engines (Qwen3 family, Qwen3.5 hybrid): the single-flight
	// prefill/spec entry points decline for these archs, so run the one-shot as
	// a single continuous-batching request — SeqAdd prefills the raw prompt into
	// a paged slot, StepBatch advances it token-by-token with on-device sampling.
	// Same raw-completion semantics as the dense path (no chat template).
	if eng.SeqFreeCapacity() > 0 {
		runOneShotPaged(eng, tok, args, prompt, tokens)
		return
	}

	// Spec enabled → prompt-lookup/MTP speculative decode. Works for greedy AND
	// sampling — including repeat-penalty, which the engine applies on-GPU — with
	// an output distribution identical to plain decode at the same settings.
	if args.Spec {
		nToGen := args.Predict
		if nToGen < 0 {
			nToGen = 512
		}
		seed := uint64(args.Seed)
		if args.Seed < 0 {
			seed = uint64(time.Now().UnixNano())
		}
		stops := []int32{tok.EOS, tok.EndOfTurn}
		genStart := time.Now()
		out, nAccepted, err := eng.GenerateSpec(tokens, nToGen, stops, args.DraftK,
			float32(args.Temperature), args.TopK, float32(args.TopP), float32(args.MinP),
			float32(args.RepeatPenalty), seed)
		if err != nil {
			log.Fatalf("fucina: spec generate failed: %v", err)
		}
		genElapsed := time.Since(genStart)
		fmt.Print(prompt)
		for _, t := range out {
			if tok.IsStop(t) {
				break
			}
			fmt.Print(tok.Decode([]int32{t}))
		}
		fmt.Println()
		genTPS := float64(len(out)) / genElapsed.Seconds()
		log.Printf("fucina: [spec] generated %d tokens in %.2fs (%.1f tok/s), "+
			"%d drafts accepted (avg %.2f tokens/step, draft-k=%d)",
			len(out), genElapsed.Seconds(), genTPS, nAccepted,
			float64(len(out))/float64(max(1, len(out)-nAccepted)), args.DraftK)
		return
	}

	// Prefill directly (bypass KVCache to isolate any cache bugs)
	prefillStart := time.Now()
	logits, err := eng.Prefill(tokens)
	if err != nil {
		log.Fatalf("fucina: prefill failed: %v", err)
	}
	prefillElapsed := time.Since(prefillStart)
	prefillTPS := float64(len(tokens)) / prefillElapsed.Seconds()
	log.Printf("fucina: prefill %d tokens in %.2fs (%.1f tok/s)",
		len(tokens), prefillElapsed.Seconds(), prefillTPS)

	rng := newRNG(args.Seed)
	nToGenerate := args.Predict
	if nToGenerate < 0 {
		nToGenerate = 1 << 20
	}

	// GPU-side sampling: when no repeat penalty is configured, select each token
	// on the device and decode without copying the 262k logits to host (4-byte id
	// instead of 1 MB per token, and no CPU sampling round-trip). Repeat penalty
	// still needs the host sampler (it edits logits using the token history).
	gpuSample := args.RepeatPenalty == 1.0
	temp := float32(args.Temperature)

	fmt.Print(prompt)
	genStart := time.Now()
	generated := 0
	// Repeat penalty (host sampler path) edits logits using the token history;
	// `past` carries prompt + generated tokens so the penalty actually applies
	// (passing nil here was a silent no-op).
	past := append([]int32(nil), tokens...)
	for i := 0; i < nToGenerate; i++ {
		var token int32
		var err error
		if gpuSample {
			token, err = eng.SampleDevice(temp, args.TopK,
				float32(args.TopP), float32(args.MinP), float32(rng.Float64()))
		} else {
			if logits == nil {
				break
			}
			token, err = sampler.Sample(logits, samplerParams(args), rng, past)
		}
		if err != nil || tok.IsStop(token) {
			break
		}
		fmt.Print(tok.Decode([]int32{token}))
		if gpuSample {
			err = eng.DecodeNoCopy(token)
		} else {
			logits, err = eng.Decode(token)
		}
		if err != nil {
			break
		}
		past = append(past, token)
		generated++
	}
	genElapsed := time.Since(genStart)
	fmt.Println()

	genTPS := 0.0
	if genElapsed.Seconds() > 0 {
		genTPS = float64(generated) / genElapsed.Seconds()
	}
	log.Printf("fucina: generated %d tokens in %.2fs (%.1f tok/s)",
		generated, genElapsed.Seconds(), genTPS)

	if args.Timings {
		eng.PrintTiming()
		ts := eng.Timing()
		log.Printf("fucina: [GPU] prefill %.1f tok/s, decode %.1f tok/s",
			ts.PrefillTokensPerSec(), ts.DecodeTokensPerSec())
	}
}

// runOneShotPaged runs the one-shot prompt on a paged multi-sequence engine:
// one SeqAdd prefill, then a StepBatch loop on that slot. Sampling is on-device
// per SeqParams, exactly like a served request. Output streams incrementally
// with the whole-slice-decode/emit-new-suffix trick so multi-byte UTF-8 is
// never split across tokens. Repeat-penalty is not applied on this path (the
// on-device sampler does not take it).
func runOneShotPaged(eng *cuda.Engine, tok *tokenizer.Tokenizer, args CLIArgs,
	prompt string, tokens []int32) {

	nToGenerate := args.Predict
	if nToGenerate < 0 {
		nToGenerate = 512
	}
	seed := uint64(args.Seed)
	if args.Seed < 0 {
		seed = uint64(time.Now().UnixNano())
	}
	params := batch.SeqParams{
		Temperature: float32(args.Temperature),
		TopK:        args.TopK,
		TopP:        float32(args.TopP),
		MinP:        float32(args.MinP),
		Seed:        seed,
	}

	prefillStart := time.Now()
	slot, token, err := eng.SeqAdd(tokens, params)
	if err != nil {
		log.Fatalf("fucina: prefill failed: %v", err)
	}
	defer eng.SeqRemove(slot)
	prefillElapsed := time.Since(prefillStart)
	log.Printf("fucina: prefill %d tokens in %.2fs (%.1f tok/s)",
		len(tokens), prefillElapsed.Seconds(),
		float64(len(tokens))/prefillElapsed.Seconds())

	fmt.Print(prompt)
	genStart := time.Now()
	generated := 0
	var outIDs []int32
	emitted := ""
	for generated < nToGenerate && !tok.IsStop(token) {
		outIDs = append(outIDs, token)
		if full := tok.Decode(outIDs); len(full) > len(emitted) {
			fmt.Print(full[len(emitted):])
			emitted = full
		}
		out, serr := eng.StepBatch([]int32{int32(slot)}, []int32{token})
		if serr != nil {
			log.Printf("fucina: decode error: %v", serr)
			break
		}
		generated++
		token = out[0]
	}
	genElapsed := time.Since(genStart)
	fmt.Println()

	genTPS := 0.0
	if genElapsed.Seconds() > 0 {
		genTPS = float64(generated) / genElapsed.Seconds()
	}
	log.Printf("fucina: generated %d tokens in %.2fs (%.1f tok/s)",
		generated, genElapsed.Seconds(), genTPS)
}
