package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"syscall"

	"github.com/hikmaai-io/fucina/internal/engine/e4b"
	"github.com/hikmaai-io/fucina/internal/sampler"
	gemserver "github.com/hikmaai-io/fucina/internal/server"
	"github.com/hikmaai-io/fucina/internal/tokenizer"
)

// e4bServer adapts *e4b.Engine to the OpenAI server's serverEngine interface.
//
// E4B has no speculative decode (GenerateSpecStream is a plain decode loop, 0 accepted
// drafts), but it now supports the server KVCache prefix-reuse contract:
//   - NTokens() reports the real n_past, so the KVCache tracks the live KV length;
//   - Rewind(nKeep) rewinds slot 0 to the shared prefix (false if the sliding ring
//     already dropped that window → KVCache falls back to a full reset+re-prefill);
//   - Prefill(suffix) APPENDS at the current n_past, so the KVCache re-prefills only
//     the divergent suffix after reusing a shared prefix — turning each request's full
//     re-prefill of a large shared system prompt into suffix-only.
type e4bServer struct {
	eng  *e4b.Engine
	last []float32 // last logits, for SampleDevice / DecodeNoCopy emulation
}

func (a *e4bServer) Prefill(tokens []int32) ([]float32, error) {
	// The KVCache hands us only the divergent suffix and has already Rewound the engine
	// to the shared-prefix length, so append at the current n_past (not a fresh reset).
	lg, err := a.eng.PrefillAppend(tokens)
	a.last = lg
	return lg, err
}

func (a *e4bServer) Decode(token int32) ([]float32, error) {
	lg, err := a.eng.Decode(token)
	a.last = lg
	return lg, err
}

func (a *e4bServer) DecodeNoCopy(token int32) error {
	lg, err := a.eng.Decode(token) // E4B has no logits-free decode; cache for SampleDevice
	a.last = lg
	return err
}

// SampleDevice host-samples from the cached logits (E4B has no on-GPU sampler). Not on
// the server's hot path — generation goes through GenerateSpecStream — but kept correct.
func (a *e4bServer) SampleDevice(temp float32, topK int, topP, minP, rnd float32) (int32, error) {
	rng := rand.New(rand.NewSource(int64(rnd*1e9) + 1))
	return sampler.Sample(a.last, sampler.Params{
		Temperature: float64(temp), TopK: topK, TopP: float64(topP), MinP: float64(minP), RepeatPenalty: 1.0,
	}, rng, nil)
}

// GenerateSpecStream generates the response, streaming each committed token via emit
// (return true = stop) and stopping at a stop token or max_new.
//
// When an MTP assistant is loaded AND the request is greedy (temp<=0), it drives the C
// greedy speculative-decode loop (~2x decode, byte-identical to greedy). Otherwise (temp>0,
// or no assistant) it falls back to the plain host-sampled prefill-then-decode loop so
// temperature/top-k/top-p/min-p still apply. Returns the committed tokens + accepted drafts.
func (a *e4bServer) GenerateSpecStream(history []int32, firstLogits []float32, maxNew int,
	stops []int32, draftK int, temp float32, topK int, topP, minP, repeatPenalty float32,
	seed uint64, emit func(int32) bool) ([]int32, int, error) {

	// Greedy + assistant → the lossless C spec loop (continues from the live KV the server
	// already prefilled; emits per token; advances n_past by the committed count).
	if temp <= 0 && a.eng.HasAssistant() {
		_, _, accBefore, _ := a.eng.SpecStats()
		toks, err := a.eng.SpecStream(history, firstLogits, maxNew, stops, emit)
		_, _, accAfter, _ := a.eng.SpecStats()
		return toks, int(accAfter - accBefore), err
	}

	p := sampler.Params{
		Temperature: float64(temp), TopK: topK, TopP: float64(topP),
		MinP: float64(minP), RepeatPenalty: float64(repeatPenalty),
	}
	rng := rand.New(rand.NewSource(int64(seed)))
	logits := firstLogits
	past := append([]int32(nil), history...)
	var all []int32
	for n := 0; n < maxNew; n++ {
		if logits == nil {
			break
		}
		t, err := sampler.Sample(logits, p, rng, past)
		if err != nil {
			return all, 0, err
		}
		all = append(all, t)
		past = append(past, t)
		if (emit != nil && emit(t)) || isStopToken(t, stops) {
			break
		}
		if logits, err = a.eng.Decode(t); err != nil {
			return all, 0, err
		}
	}
	return all, 0, nil
}

func (a *e4bServer) GenerateSpecContinue(history []int32, firstLogits []float32, maxNew int,
	stops []int32, draftK int, temp float32, topK int, topP, minP, repeatPenalty float32,
	seed uint64) ([]int32, int, error) {
	return a.GenerateSpecStream(history, firstLogits, maxNew, stops, draftK, temp, topK, topP, minP, repeatPenalty, seed, nil)
}

func (a *e4bServer) NTokens() int                                 { return a.eng.NPast() }
func (a *e4bServer) Reset()                                       { a.eng.Reset() }
func (a *e4bServer) Rewind(nKeep int) bool                        { return a.eng.Rewind(nKeep) }
func (a *e4bServer) ContextSize() uint32                          { return a.eng.ContextSize() }
func (a *e4bServer) SpecStats() (steps, drafted, accepted, emitted int64) { return a.eng.SpecStats() }

func isStopToken(t int32, stops []int32) bool {
	for _, s := range stops {
		if s == t {
			return true
		}
	}
	return false
}

// runE4BServer serves an E4B checkpoint over the OpenAI-compatible HTTP API, mirroring
// the dense server setup in main.go (no spec / MTP / continuous batching for E4B).
func runE4BServer(eng *e4b.Engine, tok *tokenizer.Tokenizer, args CLIArgs) {
	srv := gemserver.New(&e4bServer{eng: eng}, tok)
	srv.SetModelName(deriveModelID(args.ModelPath))
	srv.SetThinkingDefault(gemserver.ParseThinkingLevel(args.Thinking))
	srv.SetThinkBudget(args.ThinkBudget)
	apiKey := args.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("FUCINA_API_KEY")
	}
	srv.SetAPIKey(apiKey)
	if apiKey != "" {
		log.Printf("fucina: API-key auth ENABLED on /v1/*")
	} else if args.Host != "127.0.0.1" && args.Host != "localhost" {
		log.Printf("fucina: WARNING — binding %s with NO API key; /v1/* is unauthenticated.", args.Host)
	}
	srv.SetMaxConcurrent(args.MaxConcurrent)
	srv.SetMaxOutputTokens(args.MaxOutputToks)
	if args.Debug {
		srv.SetDebug(true)
	}
	// Continuous batching (--parallel / --cont-batching): drive E4B's slot-based
	// step_batch through the scheduler so concurrent requests share one weight pass.
	// Greedy per sequence (the E4B batched kernel argmaxes), so per-request temp/top-p
	// are not applied in batched mode.
	if args.ContBatching {
		if srv.SetBatchEngine(e4b.NewBatchAdapter(eng)) {
			log.Printf("fucina: E4B continuous batching ENABLED (up to %d concurrent sequences, greedy)", eng.SeqCapacity())
		} else {
			log.Printf("fucina: WARNING — --cont-batching requested but no free E4B slots; single-flight")
		}
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() { <-sigCh; log.Println("fucina: shutting down..."); srv.Stop() }()

	addr := fmt.Sprintf("%s:%d", args.Host, args.Port)
	specNote := "no spec/MTP"
	if eng.HasAssistant() {
		specNote = "greedy spec/MTP (temp=0)"
	}
	log.Printf("fucina: server starting on %s (Gemma-4-E4B; %s)", addr, specNote)
	// Report the effective ctx (the engine may have auto-shrunk it to fit memory),
	// not the raw --ctx request, so admission limits and this log agree.
	log.Printf("fucina: model=%s ctx=%d", args.ModelPath, eng.ContextSize())
	if err := srv.Start(addr); err != nil {
		log.Printf("fucina: server stopped: %v", err)
	}
}
