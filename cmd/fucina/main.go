// fucina - Gemma 4 12B inference engine for DGX Spark GB10
//
// CLI matches llama.cpp style flags for compatibility.
// Supports server mode (default) and one-shot prompt mode.
//
// Usage:
//   fucina -m model.gguf --ctx 32768 --host 0.0.0.0 --port 8080
//   fucina -m model.gguf -p "Hello" -n 100
//   fucina -m model.gguf --prompt "Test" --predict 32

package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/hikmaai-io/fucina/internal/engine/cuda"
	"github.com/hikmaai-io/fucina/internal/engine/diffusion"
	"github.com/hikmaai-io/fucina/internal/sampler"
	gemserver "github.com/hikmaai-io/fucina/internal/server"
	"github.com/hikmaai-io/fucina/internal/tokenizer"
)

// newRNG builds the sampler RNG, mapping seed -1 to a time-based seed.
func newRNG(seed int64) *rand.Rand {
	if seed < 0 {
		seed = time.Now().UnixNano()
	}
	return rand.New(rand.NewSource(seed))
}

// deriveModelID maps a model path to a client-facing model id. For a GGUF or single
// .safetensors file it strips the extension; for an NVFP4 directory it uses the dir
// name verbatim (so a checkpoint dir yields its base name, not a mangled string).
func deriveModelID(modelPath string) string {
	base := filepath.Base(modelPath)
	if fi, err := os.Stat(modelPath); err == nil && fi.IsDir() {
		return base
	}
	base = strings.TrimSuffix(base, ".gguf")
	base = strings.TrimSuffix(base, ".safetensors")
	return base
}

// samplerParams maps the CLI flags onto the shared sampler.Params.
func samplerParams(args CLIArgs) sampler.Params {
	return sampler.Params{
		Temperature:   args.Temperature,
		TopK:          args.TopK,
		TopP:          args.TopP,
		MinP:          args.MinP,
		RepeatPenalty: args.RepeatPenalty,
	}
}

// ─── Main ─────────────────────────────────────────────────────────

func main() {
	runtime.LockOSThread() // CUDA requires thread affinity

	args := parseFlags()

	// DiffusionGemma is a separate architecture (block text-diffusion MoE) with its own
	// engine. Detect it from the GGUF and route to the diffusion path before the
	// autoregressive engine is created.
	if diffusion.IsDiffusion(args.ModelPath) {
		ggufData, err := os.ReadFile(args.ModelPath)
		if err != nil {
			log.Fatalf("fucina: cannot read GGUF for tokenizer: %v", err)
		}
		tok, err := tokenizer.New(ggufData, int64(len(ggufData)))
		if err != nil {
			log.Fatalf("fucina: tokenizer init failed: %v", err)
		}
		runDiffusion(args, tok)
		return
	}

	// Initialize engine (weight format Q4_0-QAT/Q8_0 auto-detected from the GGUF)
	log.Printf("fucina: loading model %s (ctx=%d, device=%d)...",
		args.ModelPath, args.ContextSize, args.DeviceID)

	eng, err := cuda.NewEngine(cuda.Config{
		ModelPath:   args.ModelPath,
		ContextSize: uint32(args.ContextSize),
		DeviceID:    args.DeviceID,
	})
	if err != nil {
		log.Fatalf("fucina: engine init failed: %v", err)
	}
	defer eng.Close()

	// Optional MTP assistant (the official Gemma-4 draft head): drafts novel text in
	// the speculative loop; prompt-lookup still covers repeated/structured text.
	if args.AssistantPath != "" {
		if err := eng.LoadAssistant(args.AssistantPath); err != nil {
			log.Fatalf("fucina: %v", err)
		}
	}

	// CUDA graphs: off by default, --cuda-graphs enables persistent scratch.
	if args.CudaGraphs {
		eng.SetGraphMode(1)
		log.Printf("fucina: CUDA graph mode = prefill")
	}

	if args.Verbose {
		eng.PrintInfo()
	}

	// Load the tokenizer vocab. It lives in a GGUF; an NVFP4 safetensors checkpoint carries
	// no GGUF vocab, so --tokenizer must point at a Gemma-4 GGUF (same vocab) in that case.
	tokSrc := args.ModelPath
	if args.TokenizerPath != "" {
		tokSrc = args.TokenizerPath
	}
	ggufData, err := os.ReadFile(tokSrc)
	if err != nil {
		log.Fatalf("fucina: cannot read GGUF for tokenizer (%s): %v "+
			"(for an NVFP4 checkpoint pass --tokenizer <gemma-4.gguf>)", tokSrc, err)
	}

	tok, err := tokenizer.New(ggufData, int64(len(ggufData)))
	if err != nil {
		// There is no tokenizer fallback: every mode (server, REPL, one-shot)
		// needs Encode/Decode. A nil tokenizer in server mode would serve every
		// request blind, so fail fast and loud rather than start broken.
		log.Fatalf("fucina: tokenizer init failed: %v", err)
	}

	if args.Verbose {
		log.Printf("fucina: tokenizer loaded (%d tokens)", tok.NumTokens())
	}

	// Determine mode: server, interactive REPL, or one-shot
	isServer := args.Prompt == "" && args.PromptFile == "" && !args.Interactive

	if isServer {
		// Server mode
		addr := fmt.Sprintf("%s:%d", args.Host, args.Port)
		// Eagerly run the lazy first-prefill setup (persistent prefill scratch +
		// BF16 dequant scratch) AND one real batched prefill pass, so request #1's
		// prefill timer measures prefill — not ~0.5-2.1s of one-time cudaMallocs,
		// cuBLAS library init, or CUDA lazy module loading. Closes the cold-start
		// gap (first prefill was ~1385 tok/s vs ~1714 warm).
		warmStart := time.Now()
		eng.Warmup()
		log.Printf("fucina: prefill scratch warmed in %.2fs", time.Since(warmStart).Seconds())
		srv := gemserver.New(eng, tok)
		// Report a quantization-aware model id (GGUF basename minus extension), e.g.
		// gemma-4-12b-it-qat-q4_0, so clients can see which build/quant they hit.
		srv.SetModelName(deriveModelID(args.ModelPath))
		// Startup default for the reasoning channel; per-request reasoning_effort wins.
		srv.SetThinkingDefault(gemserver.ParseThinkingLevel(args.Thinking))
		srv.SetDraftK(args.DraftK)
		srv.SetThinkBudget(args.ThinkBudget)
		srv.SetKVSnapshotBudget(int64(args.KVSnapshotGB * (1 << 30)))
		// Auth: flag wins, else FUCINA_API_KEY. Empty leaves auth disabled.
		apiKey := args.APIKey
		if apiKey == "" {
			apiKey = os.Getenv("FUCINA_API_KEY")
		}
		srv.SetAPIKey(apiKey)
		if apiKey != "" {
			log.Printf("fucina: API-key auth ENABLED on /v1/*")
		} else if args.Host != "127.0.0.1" && args.Host != "localhost" {
			log.Printf("fucina: WARNING — binding %s with NO API key; /v1/* is unauthenticated. Set --api-key or FUCINA_API_KEY.", args.Host)
		}
		srv.SetMaxConcurrent(args.MaxConcurrent) // 0 ignored (keeps default)
		srv.SetMaxOutputTokens(args.MaxOutputToks)
		// Continuous batching (experimental): FUCINA_BATCH routes requests through
		// the per-step scheduler over the paged multi-sequence engine instead of the
		// per-request kv lock. Requires the engine to be in paged mode
		// (FUCINA_PAGED_KV=1); SetBatchEngine is a no-op otherwise, so the
		// single-flight path stays intact.
		if os.Getenv("FUCINA_BATCH") != "" {
			if srv.SetBatchEngine(cuda.NewBatchAdapter(eng)) {
				log.Printf("fucina: continuous batching ENABLED (paged KV multi-sequence)")
			} else {
				log.Printf("fucina: WARNING — FUCINA_BATCH set but engine not in paged mode (FUCINA_PAGED_KV); batching disabled")
			}
		}
		// Debug request dumping: --debug or --log-level debug.
		if args.Debug || strings.EqualFold(args.LogLevel, "debug") {
			srv.SetDebug(true)
			log.Printf("fucina: debug logging ON — request dumps -> /tmp/fucina_debug.log")
		}

		// Handle graceful shutdown
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

		// On signal, gracefully stop the HTTP server (drains in-flight requests).
		// srv.Start() then returns and main() falls through to `defer eng.Close()`
		// for a clean CUDA teardown — vs the old os.Exit(0), which skipped the
		// deferred Close and could exit mid-CUDA-call if Shutdown timed out.
		go func() {
			<-sigCh
			log.Println("fucina: shutting down...")
			srv.Stop()
		}()

		log.Printf("fucina: server starting on %s", addr)
		log.Printf("fucina: model=%s ctx=%d", args.ModelPath, args.ContextSize)

		if err := srv.Start(addr); err != nil {
			log.Printf("fucina: server stopped: %v", err)
		}
	} else if args.Interactive {
		// ── Interactive REPL ─────────────────────────────────────────────
		runInteractive(eng, tok, args)
	} else {
		// ── One-shot prompt ───────────────────────────────────────────────
		runOneShot(eng, tok, args)
	}
}

func init() {
	// Ensure we use all CPU cores for preprocessing
	runtime.GOMAXPROCS(runtime.NumCPU())
}
