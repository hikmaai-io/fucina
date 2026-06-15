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

	// Load GGUF for tokenizer
	ggufData, err := os.ReadFile(args.ModelPath)
	if err != nil {
		log.Fatalf("fucina: cannot read GGUF for tokenizer: %v", err)
	}

	tok, err := tokenizer.New(ggufData, int64(len(ggufData)))
	if err != nil {
		log.Printf("fucina: warning: tokenizer init (will use fallback): %v", err)
		tok = nil
	}

	if args.Verbose && tok != nil {
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
		srv.SetModelName(strings.TrimSuffix(filepath.Base(args.ModelPath), ".gguf"))
		// Startup default for the reasoning channel; per-request reasoning_effort wins.
		srv.SetThinkingDefault(gemserver.ParseThinkingLevel(args.Thinking))
		srv.SetDraftK(args.DraftK)
		srv.SetThinkBudget(args.ThinkBudget)
		srv.SetKVSnapshotBudget(int64(args.KVSnapshotGB * (1 << 30)))
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
		if tok == nil {
			log.Fatalf("fucina: tokenizer not available")
		}
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
