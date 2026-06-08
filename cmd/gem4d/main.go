// gem4d - Gemma 4 12B inference engine for DGX Spark GB10
//
// CLI matches llama.cpp style flags for compatibility.
// Supports server mode (default) and one-shot prompt mode.
//
// Usage:
//   gem4d -m model.gguf --ctx 32768 --host 0.0.0.0 --port 8080
//   gem4d -m model.gguf -p "Hello" -n 100
//   gem4d -m model.gguf --lora-scaled lora.gguf --prompt "Test" --predict 32

package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"runtime"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/mauromedda/gem4d/internal/engine/cuda"
	gemserver "github.com/mauromedda/gem4d/internal/server"
	"github.com/mauromedda/gem4d/internal/tokenizer"
)

// newRNG builds the sampler RNG, mapping seed -1 to a time-based seed.
func newRNG(seed int64) *rand.Rand {
	if seed < 0 {
		seed = time.Now().UnixNano()
	}
	return rand.New(rand.NewSource(seed))
}

// ─── Command-line flags (llama.cpp style) ─────────────────────────

type CLIArgs struct {
	// Model
	ModelPath string
	LoraPath  string
	LoraScale float64
	Format    string // "fp8" or "q8_0"

	// Inference
	ContextSize int
	BatchSize   int
	Threads     int
	FlashAttn   bool

	// Sampling
	Temperature     float64
	TopK            int
	TopP            float64
	MinP            float64
	Seed            int64
	RepeatPenalty   float64
	FrequencyPenalty float64
	PresencePenalty float64

	// Generation
	Prompt      string
	PromptFile  string
	Predict     int
	Keep        int
	NoDisplay   bool
	Spec        bool
	DraftK      int

	// Server
	Host    string
	Port    int
	Timeout int
	Slots   int

	// System
	System   string
	Verbose  bool
	Timings  bool
	DeviceID int
	Memory   string

	// Mode
	Interactive bool
	Multiline   bool
	Color       bool
	Help        bool
}

func parseFlags() CLIArgs {
	var a CLIArgs

	// Defaults matching llama.cpp
	flag.StringVar(&a.ModelPath, "m", "", "Path to GGUF model file")
	flag.StringVar(&a.ModelPath, "model", "", "Path to GGUF model file")
	flag.StringVar(&a.LoraPath, "lora-scaled", "", "Path to LoRA scaled adapter GGUF")
	flag.Float64Var(&a.LoraScale, "lora-scale", 1.0, "LoRA scale multiplier")
	flag.StringVar(&a.Format, "memory-format", "fp8", "Memory format: fp8 or q8_0")
	flag.StringVar(&a.Format, "format", "fp8", "Memory format: fp8 or q8_0")

	flag.IntVar(&a.ContextSize, "ctx", 4096, "Context size in tokens")
	flag.IntVar(&a.ContextSize, "c", 4096, "Context size (short)")
	flag.IntVar(&a.BatchSize, "batch-size", 2048, "Logical maximum batch size")
	flag.IntVar(&a.Threads, "threads", 8, "Number of CPU threads for preprocessing")
	flag.BoolVar(&a.FlashAttn, "flash-attn", true, "Use Flash Attention")

	flag.BoolVar(&a.Spec, "spec", true, "Prompt-lookup speculative decoding (greedy/temp=0 only)")
	flag.IntVar(&a.DraftK, "draft-k", 6, "Max speculative draft length per step")
	flag.Float64Var(&a.Temperature, "temp", 0.8, "Sampling temperature")
	flag.IntVar(&a.TopK, "top-k", 40, "Top-K sampling")
	flag.Float64Var(&a.TopP, "top-p", 0.95, "Top-P sampling")
	flag.Float64Var(&a.MinP, "min-p", 0.05, "Min-P sampling")
	flag.Int64Var(&a.Seed, "seed", -1, "Random seed (-1 = random)")
	flag.Float64Var(&a.RepeatPenalty, "repeat-penalty", 1.1, "Repeat penalty")
	flag.Float64Var(&a.FrequencyPenalty, "frequency-penalty", 0.0, "Frequency penalty")
	flag.Float64Var(&a.PresencePenalty, "presence-penalty", 0.0, "Presence penalty")

	flag.StringVar(&a.Prompt, "p", "", "Prompt string")
	flag.StringVar(&a.Prompt, "prompt", "", "Prompt string")
	flag.StringVar(&a.PromptFile, "f", "", "Prompt file")
	flag.StringVar(&a.PromptFile, "file", "", "Prompt file")
	flag.IntVar(&a.Predict, "n", 512, "Number of tokens to predict (-1 = infinite)")
	flag.IntVar(&a.Predict, "predict", 512, "Number of tokens to predict")

	flag.StringVar(&a.Host, "host", "127.0.0.1", "Server listen address")
	flag.IntVar(&a.Port, "port", 8080, "Server port")
	flag.IntVar(&a.Timeout, "timeout", 600, "Request timeout in seconds")
	flag.IntVar(&a.Slots, "n-slots", 1, "Number of processing slots")

	flag.StringVar(&a.System, "s", "", "System prompt")
	flag.StringVar(&a.System, "system", "", "System prompt")
	flag.BoolVar(&a.Verbose, "v", false, "Verbose output")
	flag.BoolVar(&a.Verbose, "verbose", false, "Verbose output")
	flag.BoolVar(&a.Timings, "timings", false, "Show timing information")
	flag.IntVar(&a.DeviceID, "cuda-device", 0, "CUDA device ID")
	flag.StringVar(&a.Memory, "mlock", "", "mlock model in memory (unused on CUDA)")

	flag.BoolVar(&a.Interactive, "interactive-first", false, "Force interactive mode")
	flag.BoolVar(&a.Interactive, "interactive", false, "Force interactive mode")
	flag.BoolVar(&a.Color, "color", false, "Color output")

	flag.BoolVar(&a.Help, "h", false, "Show help")
	flag.BoolVar(&a.Help, "help", false, "Show help")

	// Test flags
	testParser := flag.Bool("test-parser", false, "Run tokenizer tests and exit")
	testCUDA := flag.Bool("test-cuda", false, "Run CUDA tests and exit")
	testVectors := flag.String("test-vectors", "", "Run test vectors from file")

	flag.Parse()

	if a.Help {
		printUsage()
		os.Exit(0)
	}

	// Model is required unless running tests
	if a.ModelPath == "" && !*testParser && !*testCUDA && *testVectors == "" {
		// Check default locations
		for _, p := range []string{"./gemma-4-12b-it.gguf", "./model.gguf", "./gguf/model.gguf"} {
			if _, err := os.Stat(p); err == nil {
				a.ModelPath = p
				break
			}
		}
		if a.ModelPath == "" {
			fmt.Fprintf(os.Stderr, "error: no model specified. Use -m <model.gguf>\n\n")
			printUsage()
			os.Exit(1)
		}
	}

	// Handle test flags
	if *testParser {
		os.Exit(runTestParser())
	}
	if *testCUDA {
		os.Exit(runTestCUDA())
	}
	if *testVectors != "" {
		os.Exit(runTestVectors(*testVectors))
	}

	return a
}

// ─── Main ─────────────────────────────────────────────────────────

func main() {
	runtime.LockOSThread() // CUDA requires thread affinity

	args := parseFlags()

	// Determine format
	var fmtType cuda.TensorFormat
	switch strings.ToLower(args.Format) {
	case "fp8":
		fmtType = cuda.FormatFP8
	case "q8_0":
		fmtType = cuda.FormatQ8_0
	default:
		log.Fatalf("unsupported format: %s (use fp8 or q8_0)", args.Format)
	}

	// Initialize engine
	log.Printf("gem4d: loading model %s (format=%s, ctx=%d, device=%d)...",
		args.ModelPath, args.Format, args.ContextSize, args.DeviceID)

	eng, err := cuda.NewEngine(cuda.Config{
		ModelPath:   args.ModelPath,
		LoraPath:    args.LoraPath,
		LoraScale:   args.LoraScale,
		Format:      fmtType,
		ContextSize: uint32(args.ContextSize),
		DeviceID:    args.DeviceID,
	})
	if err != nil {
		log.Fatalf("gem4d: engine init failed: %v", err)
	}
	defer eng.Close()

	if args.Verbose {
		eng.PrintInfo()
	}

	// Load GGUF for tokenizer
	ggufData, err := os.ReadFile(args.ModelPath)
	if err != nil {
		log.Fatalf("gem4d: cannot read GGUF for tokenizer: %v", err)
	}

	tok, err := tokenizer.New(ggufData, int64(len(ggufData)))
	if err != nil {
		log.Printf("gem4d: warning: tokenizer init (will use fallback): %v", err)
		tok = nil
	}

	if args.Verbose && tok != nil {
		log.Printf("gem4d: tokenizer loaded (%d tokens)", tok.NumTokens())
	}

	// Determine mode: server, interactive REPL, or one-shot
	isServer := args.Prompt == "" && args.PromptFile == "" && !args.Interactive

	if isServer {
		// Server mode
		addr := fmt.Sprintf("%s:%d", args.Host, args.Port)
		srv := gemserver.New(eng, tok)

		// Handle graceful shutdown
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

		go func() {
			<-sigCh
			log.Println("gem4d: shutting down...")
			srv.Stop()
			os.Exit(0)
		}()

		log.Printf("gem4d: server starting on %s", addr)
		log.Printf("gem4d: model=%s ctx=%d format=%s", args.ModelPath, args.ContextSize, args.Format)
		if args.LoraPath != "" {
			log.Printf("gem4d: lora=%s scale=%.2f", args.LoraPath, args.LoraScale)
		}

		if err := srv.Start(addr); err != nil {
			log.Printf("gem4d: server stopped: %v", err)
		}
	} else if args.Interactive {
		// ── Interactive REPL ─────────────────────────────────────────────
		if tok == nil {
			log.Fatalf("gem4d: tokenizer not available")
		}
		runInteractive(eng, tok, args)
	} else {
		// ── One-shot prompt ───────────────────────────────────────────────
		prompt := args.Prompt
		if args.PromptFile != "" {
			data, err := os.ReadFile(args.PromptFile)
			if err != nil {
				log.Fatalf("gem4d: cannot read prompt file: %v", err)
			}
			prompt = string(data)
		}
		if prompt == "" {
			log.Fatalf("gem4d: empty prompt. Use -p or -f")
		}
		if args.System != "" {
			prompt = fmt.Sprintf("System: %s\n\n%s", args.System, prompt)
		}
		if tok == nil {
			log.Fatalf("gem4d: tokenizer not available")
		}

		// Tokenize
		tokens := tok.Encode(prompt, true, false)
		log.Printf("gem4d: prompt has %d tokens", len(tokens))

		// Greedy + spec enabled → prompt-lookup speculative decode (one weight pass
		// per [g, draft...]; produces the exact same tokens as plain greedy decode).
		if args.Spec && args.Temperature <= 0 {
			nToGen := args.Predict
			if nToGen < 0 {
				nToGen = 512
			}
			stops := []int32{tok.EOS, tok.EndOfTurn}
			genStart := time.Now()
			out, nAccepted, err := eng.GenerateSpec(tokens, nToGen, stops, args.DraftK)
			if err != nil {
				log.Fatalf("gem4d: spec generate failed: %v", err)
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
			log.Printf("gem4d: [spec] generated %d tokens in %.2fs (%.1f tok/s), "+
				"%d drafts accepted (avg %.2f tokens/step, draft-k=%d)",
				len(out), genElapsed.Seconds(), genTPS, nAccepted,
				float64(len(out))/float64(max(1, len(out)-nAccepted)), args.DraftK)
			return
		}

		// Prefill directly (bypass KVCache to isolate any cache bugs)
		prefillStart := time.Now()
		logits, err := eng.Prefill(tokens)
		if err != nil {
			log.Fatalf("gem4d: prefill failed: %v", err)
		}
		prefillElapsed := time.Since(prefillStart)
		prefillTPS := float64(len(tokens)) / prefillElapsed.Seconds()
		log.Printf("gem4d: prefill %d tokens in %.2fs (%.1f tok/s)",
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
				token, err = sample(logits, args, rng, nil)
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
			generated++
		}
		genElapsed := time.Since(genStart)
		fmt.Println()

		genTPS := 0.0
		if genElapsed.Seconds() > 0 {
			genTPS = float64(generated) / genElapsed.Seconds()
		}
		log.Printf("gem4d: generated %d tokens in %.2fs (%.1f tok/s)",
			generated, genElapsed.Seconds(), genTPS)

		if args.Timings {
			eng.PrintTiming()
			ts := eng.Timing()
			log.Printf("gem4d: [GPU] prefill %.1f tok/s, decode %.1f tok/s",
				ts.PrefillTokensPerSec(), ts.DecodeTokensPerSec())
		}
	}
}

// ─── Interactive REPL ────────────────────────────────────────────

// runInteractive runs a multi-turn chat REPL with prefix-reuse KV caching.
// Each turn:
//  1. Build the full Gemma chat-template prompt from conversation history.
//  2. Ask KVCache.Prefill — which reuses the shared prefix from last turn
//     (the entire prior conversation) and only computes the new user message.
//  3. Stream tokens to stdout as they are sampled.
//  4. Append the completed reply to history so the NEXT turn can reuse it.
func runInteractive(eng *cuda.Engine, tok *tokenizer.Tokenizer, args CLIArgs) {
	type turn struct{ role, content string }
	var history []turn

	kv  := gemserver.NewKVCache(eng)
	rng := newRNG(args.Seed)

	nToGenerate := args.Predict
	if nToGenerate <= 0 {
		nToGenerate = 1 << 20
	}

	// Gemma 4 chat template. The real gemma-4 vocab has NO <start_of_turn>/
	// <end_of_turn> tokens — turns are delimited by <|turn> (105) / <turn|>
	// (106), and an empty thought channel <|channel>thought\n<channel|> is
	// pre-filled after the model turn opener when thinking is disabled (default),
	// otherwise the model opens a thought channel and leaks its reasoning.
	const modelOpen = "<|turn>model\n<|channel>thought\n<channel|>"
	buildPrompt := func(turns []turn) string {
		var sb strings.Builder
		for i, t := range turns {
			switch t.role {
			case "system":
				fmt.Fprintf(&sb, "<|turn>system\n%s<turn|>\n", t.content)
			case "user":
				fmt.Fprintf(&sb, "<|turn>user\n%s<turn|>\n", t.content)
			case "assistant":
				if i == len(turns)-1 {
					sb.WriteString(modelOpen) // open turn — model fills this in
				} else {
					fmt.Fprintf(&sb, "<|turn>model\n%s<turn|>\n", t.content)
				}
			}
		}
		if len(turns) == 0 || turns[len(turns)-1].role != "assistant" {
			sb.WriteString(modelOpen)
		}
		return sb.String()
	}

	// Optional system prompt injected as the first user turn.
	if args.System != "" {
		history = append(history, turn{"system", args.System})
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
				history = append(history, turn{"system", args.System})
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
		history = append(history, turn{"user", input})
		promptStr := buildPrompt(history)
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
		logits  := pf.Logits
		genStart := time.Now()
		generated := 0

		// Print assistant prefix so the reply is visually distinct.
		fmt.Fprint(os.Stdout, "\033[1;34mAssistant:\033[0m ")

		for i := 0; i < nToGenerate; i++ {
			if logits == nil {
				break
			}
			token, err := sample(logits, args, rng, nil)
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
		history = append(history, turn{"assistant", strings.TrimSpace(replyBuf.String())})
	}
}

// ─── Prompt sampling ──────────────────────────────────────────────

type cand struct {
	id int32
	lg float32
}

// topKCandidates returns the k highest-logit candidates sorted descending, using a
// bounded size-k min-heap — a single O(V) pass instead of sorting all 262k logits
// (the previous full sort + 262k-element allocation cost ~30 ms per token). k<=0 or
// k>=len falls back to a full sort (top-k disabled → top-p/min-p need the full order).
func topKCandidates(logits []float32, k int) []cand {
	n := len(logits)
	if k <= 0 || k >= n {
		cands := make([]cand, n)
		for i := range logits {
			cands[i] = cand{int32(i), logits[i]}
		}
		sort.Slice(cands, func(a, b int) bool { return cands[a].lg > cands[b].lg })
		return cands
	}
	h := make([]cand, 0, k) // min-heap keyed by lg: h[0] is the smallest kept
	for i := 0; i < n; i++ {
		lg := logits[i]
		if len(h) < k {
			h = append(h, cand{int32(i), lg})
			for j := len(h) - 1; j > 0; { // sift up
				p := (j - 1) / 2
				if h[p].lg <= h[j].lg {
					break
				}
				h[p], h[j] = h[j], h[p]
				j = p
			}
		} else if lg > h[0].lg {
			h[0] = cand{int32(i), lg}
			for j := 0; ; { // sift down
				l, r, small := 2*j+1, 2*j+2, j
				if l < k && h[l].lg < h[small].lg {
					small = l
				}
				if r < k && h[r].lg < h[small].lg {
					small = r
				}
				if small == j {
					break
				}
				h[j], h[small] = h[small], h[j]
				j = small
			}
		}
	}
	sort.Slice(h, func(a, b int) bool { return h[a].lg > h[b].lg })
	return h
}

// sample draws the next token from logits applying the configured pipeline:
// repeat-penalty → temperature → top-k → top-p → min-p → multinomial.
// Temperature <= 0 forces greedy argmax (deterministic).
func sample(logits []float32, args CLIArgs, rng *rand.Rand, pastTokens []int32) (int32, error) {
	if args.Temperature <= 0 {
		return int32(cuda.Argmax(logits)), nil
	}

	// Repeat penalty, applied in place (logits is regenerated each decode step, so
	// no defensive copy is needed). llama.cpp applies it before temperature.
	if args.RepeatPenalty != 1.0 && len(pastTokens) > 0 {
		rp := float32(args.RepeatPenalty)
		for _, id := range pastTokens {
			if id < 0 || int(id) >= len(logits) {
				continue
			}
			if logits[id] > 0 {
				logits[id] /= rp
			} else {
				logits[id] *= rp
			}
		}
	}

	// Candidate selection. Temperature is monotonic so it does not change the
	// ordering — select the top-k on the raw (penalized) logits FIRST, then apply
	// temperature/softmax to just those k. This avoids sorting and allocating over
	// the whole 262k vocab every token (the dominant per-token CPU cost otherwise).
	cands := topKCandidates(logits, args.TopK)

	// Softmax over the (sorted) candidates, with temperature folded in.
	invT := float64(1.0 / args.Temperature)
	maxLg := float64(cands[0].lg)
	var sum float64
	probs := make([]float64, len(cands))
	for i, c := range cands {
		p := math.Exp((float64(c.lg) - maxLg) * invT)
		probs[i] = p
		sum += p
	}
	for i := range probs {
		probs[i] /= sum
	}

	// Top-p (nucleus): keep the smallest prefix whose cumulative prob >= TopP.
	if args.TopP > 0 && args.TopP < 1.0 {
		var cum float64
		cut := len(probs)
		for i := range probs {
			cum += probs[i]
			if cum >= args.TopP {
				cut = i + 1
				break
			}
		}
		cands = cands[:cut]
		probs = probs[:cut]
	}

	// Min-p: drop candidates below MinP * max_prob.
	if args.MinP > 0 {
		thresh := args.MinP * probs[0]
		keep := 0
		for i := range probs {
			if probs[i] >= thresh {
				keep = i + 1
			} else {
				break
			}
		}
		if keep > 0 {
			cands = cands[:keep]
			probs = probs[:keep]
		}
	}

	// Renormalize and draw.
	var z float64
	for _, p := range probs {
		z += p
	}
	r := rng.Float64() * z
	var acc float64
	for i, p := range probs {
		acc += p
		if r <= acc {
			return cands[i].id, nil
		}
	}
	return cands[len(cands)-1].id, nil
}

// ─── Test stubs ───────────────────────────────────────────────────

func runTestParser() int {
	fmt.Println("gem4d: tokenizer parser test - not yet implemented")
	return 0
}

func runTestCUDA() int {
	fmt.Println("gem4d: CUDA test - not yet implemented")
	return 0
}

func runTestVectors(path string) int {
	fmt.Printf("gem4d: test vectors from %s - not yet implemented\n", path)
	return 0
}

// ─── Usage ────────────────────────────────────────────────────────

func printUsage() {
	fmt.Print(`gem4d - Gemma 4 12B inference engine for DGX Spark GB10

Usage:
  gem4d -m model.gguf [options]                  # Server mode (default)
  gem4d -m model.gguf -p "Hello" -n 100          # One-shot prompt
  gem4d -m model.gguf --lora-scaled lora.gguf \  
         --prompt "Test" --predict 32             # With LoRA adapter

Model options:
  -m, --model FILE           Path to GGUF model file
  --lora-scaled FILE         Path to LoRA scaled adapter GGUF
  --lora-scale F             LoRA scale multiplier (default: 1.0)
  --memory-format FORMAT     fp8 or q8_0 (default: fp8)

Inference options:
  --ctx N                    Context size (default: 4096, max: 262144)
  --batch-size N             Maximum batch size (default: 2048)
  --threads N                CPU threads (default: 8)
  --flash-attn               Use Flash Attention (default: true)

Sampling options:
  --temp F                   Temperature (default: 0.8)
  --top-k N                  Top-K sampling (default: 40)
  --top-p F                  Top-P sampling (default: 0.95)
  --min-p F                  Min-P sampling (default: 0.05)
  --seed N                   Seed (-1 = random)
  --repeat-penalty F         Repeat penalty (default: 1.1)

Generation options:
  -p, --prompt PROMPT        Prompt string
  -f, --file FILE            Prompt file
  -n, --predict N            Tokens to generate (default: 512)
  -s, --system TEXT          System prompt

Server options:
  --host ADDR                Server address (default: 127.0.0.1)
  --port N                   Server port (default: 8080)
  --timeout N                Request timeout (default: 600s)
  --n-slots N                Number of slots (default: 1)

Other options:
  --cuda-device N            CUDA device ID (default: 0)
  -v, --verbose              Verbose output
  --timings                  Show detailed GPU timing (prefill/decode tok/s)
  -h, --help                 Show this help

Formats:
  fp8    (default) - FP8 E4M3, native Blackwell Tensor Core
  q8_0             - GGML Q8_0 blocks (fallback)

Examples:
  gem4d -m gemma-4-12b-it.gguf --ctx 32768 &
  curl http://localhost:8080/v1/chat/completions \\
    -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
`)
}

func init() {
	// Ensure we use all CPU cores for preprocessing
	runtime.GOMAXPROCS(runtime.NumCPU())
}
