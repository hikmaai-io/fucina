// gem4d - Gemma 4 12B inference engine for DGX Spark GB10
//
// CLI matches llama.cpp style flags for compatibility.
// Supports server mode (default) and one-shot prompt mode.
//
// Usage:
//   gem4d -m model.gguf --ctx 32768 --host 0.0.0.0 --port 8080
//   gem4d -m model.gguf -p "Hello" -n 100
//   gem4d -m model.gguf --prompt "Test" --predict 32

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

	"github.com/mauromedda/gem4d/internal/engine/cuda"
	"github.com/mauromedda/gem4d/internal/sampler"
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

	// Initialize engine (weight format Q4_0-QAT/Q8_0 auto-detected from the GGUF)
	log.Printf("gem4d: loading model %s (ctx=%d, device=%d)...",
		args.ModelPath, args.ContextSize, args.DeviceID)

	eng, err := cuda.NewEngine(cuda.Config{
		ModelPath:   args.ModelPath,
		ContextSize: uint32(args.ContextSize),
		DeviceID:    args.DeviceID,
	})
	if err != nil {
		log.Fatalf("gem4d: engine init failed: %v", err)
	}
	defer eng.Close()

	// Optional MTP assistant (the official Gemma-4 draft head): drafts novel text in
	// the speculative loop; prompt-lookup still covers repeated/structured text.
	if args.AssistantPath != "" {
		if err := eng.LoadAssistant(args.AssistantPath); err != nil {
			log.Fatalf("gem4d: %v", err)
		}
	}

	// CUDA graphs: off by default, --cuda-graphs enables persistent scratch.
	if args.CudaGraphs {
		eng.SetGraphMode(1)
		log.Printf("gem4d: CUDA graph mode = prefill")
	}

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
		// Report a quantization-aware model id (GGUF basename minus extension), e.g.
		// gemma-4-12b-it-qat-q4_0, so clients can see which build/quant they hit.
		srv.SetModelName(strings.TrimSuffix(filepath.Base(args.ModelPath), ".gguf"))
		// Startup default for the reasoning channel; per-request reasoning_effort wins.
		srv.SetThinkingDefault(gemserver.ParseThinkingLevel(args.Thinking))
		// Debug request dumping: --debug or --log-level debug.
		if args.Debug || strings.EqualFold(args.LogLevel, "debug") {
			srv.SetDebug(true)
			log.Printf("gem4d: debug logging ON — request dumps -> /tmp/gem4d_debug.log")
		}

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
		log.Printf("gem4d: model=%s ctx=%d", args.ModelPath, args.ContextSize)

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
		runOneShot(eng, tok, args)
	}
}

func init() {
	// Ensure we use all CPU cores for preprocessing
	runtime.GOMAXPROCS(runtime.NumCPU())
}
