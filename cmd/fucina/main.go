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
	"github.com/hikmaai-io/fucina/internal/engine/e4b"
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

// loadTokenizer resolves the tokenizer for a model. NVFP4 safetensors checkpoints ship a
// HuggingFace tokenizer.json (BPE) and no GGUF vocab, so we auto-detect one next to / inside the
// model; GGUF models carry their vocab inline. An explicit --tokenizer (override) wins and may be
// either a .json (HF BPE) or a .gguf (vocab). No silent fallback — a missing tokenizer is fatal.
func loadTokenizer(modelPath, override string) (*tokenizer.Tokenizer, error) {
	if override != "" {
		if strings.HasSuffix(override, ".json") {
			return tokenizer.NewFromHFJSON(override)
		}
		data, err := os.ReadFile(override)
		if err != nil {
			return nil, fmt.Errorf("read --tokenizer %s: %w", override, err)
		}
		return tokenizer.New(data, int64(len(data)))
	}
	if j := siblingTokenizerJSON(modelPath); j != "" {
		// NewFromHFJSON validates the file up-front (BPE type, non-empty vocab,
		// merge format) and returns a descriptive error; wrap it so the user can
		// tell the tokenizer.json was AUTO-DETECTED next to the model (and can pass
		// an explicit --tokenizer to override a bad one).
		tok, err := tokenizer.NewFromHFJSON(j)
		if err != nil {
			return nil, fmt.Errorf("auto-detected tokenizer %s is not usable: %w "+
				"(pass --tokenizer <gemma-4.gguf|tokenizer.json> to override)", j, err)
		}
		return tok, nil
	}
	data, err := os.ReadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("cannot read tokenizer source %s: %w "+
			"(an NVFP4 dir needs a tokenizer.json, or pass --tokenizer <gemma-4.gguf|tokenizer.json>)",
			modelPath, err)
	}
	return tokenizer.New(data, int64(len(data)))
}

// siblingTokenizerJSON returns a tokenizer.json located in the model directory (when modelPath
// is a dir) or alongside the model file (NVFP4 single-file .safetensors), or "" if none.
func siblingTokenizerJSON(modelPath string) string {
	dir := modelPath
	if fi, err := os.Stat(modelPath); err != nil || !fi.IsDir() {
		dir = filepath.Dir(modelPath)
	}
	cand := filepath.Join(dir, "tokenizer.json")
	if _, err := os.Stat(cand); err == nil {
		return cand
	}
	return ""
}

func main() {
	runtime.LockOSThread() // CUDA requires thread affinity

	args := parseFlags()
	if args.JSpace {
		if args.JLENSPath == "" {
			log.Fatal("fucina: --jspace requires --j-lens <lens.fjls> (convert with scripts/convert_jlens.py)")
		}
		if !args.Interactive {
			log.Fatal("fucina: --jspace is a debug-only interactive feature; add --interactive or use --jspace-debug")
		}
		if args.JSpaceTopK < 1 || args.JSpaceTopK > 32 {
			log.Fatal("fucina: --jspace-top-k must be between 1 and 32")
		}
	}

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

	// Gemma-4-E4B is the on-device family member (Per-Layer Embeddings, KV-sharing,
	// runtime dims) and runs through its own engine, not the dense gemma4 path. Detect
	// it from the safetensors config.json / GGUF and route before the dense engine is
	// created. The (mmap-ing) detector is guarded by a cheap suffix/dir pre-filter so a
	// multi-GB non-E4B GGUF never reaches it.
	if e4b.LooksLikeCheckpoint(args.ModelPath) && e4b.IsE4B(args.ModelPath) {
		runE4B(args)
		return
	}

	// Paged multi-sequence KV pools are gated inside the C engine on the FUCINA_PAGED_KV
	// env var, read at engine init. Promote the --paged-kv/--batch flags to that env var
	// BEFORE NewEngine so the flag and the legacy env path converge on one code path.
	if args.PagedKV && os.Getenv("FUCINA_PAGED_KV") == "" {
		os.Setenv("FUCINA_PAGED_KV", "1")
	}
	// Size per-sequence CUDA state to the concurrency the process can actually admit. Qwen3.5's
	// recurrent arenas are large (GiBs at 16 slots), so always allocating the compile-time maximum
	// wasted memory on the default 4-request server and on one-shot/interactive runs. An explicit
	// env value remains the low-level override used by benchmarks and embedders.
	if os.Getenv("FUCINA_PAGED_MAXSEQS") == "" {
		maxSeqs := 4 // server's default admission depth
		if args.MaxConcurrent > 0 {
			maxSeqs = args.MaxConcurrent
		}
		if args.Parallel > 0 {
			maxSeqs = args.Parallel
		}
		if args.Prompt != "" || args.Interactive {
			maxSeqs = 1
		}
		if maxSeqs < 1 {
			maxSeqs = 1
		} else if maxSeqs > 16 {
			maxSeqs = 16
		}
		os.Setenv("FUCINA_PAGED_MAXSEQS", fmt.Sprintf("%d", maxSeqs))
	}

	// Initialize engine (weight format Q4_0-QAT/Q8_0 auto-detected from the GGUF)
	log.Printf("fucina: loading model %s (ctx=%d, device=%d)...",
		args.ModelPath, args.ContextSize, args.DeviceID)

	eng, err := cuda.NewEngine(cuda.Config{
		ModelPath:   args.ModelPath,
		ContextSize: uint32(args.ContextSize),
		DeviceID:    args.DeviceID,
		GPUMemUtil:  args.GPUMemUtil,
	})
	if err != nil {
		log.Fatalf("fucina: engine init failed: %v", err)
	}
	defer eng.Close()

	if args.JSpace {
		if err := eng.LoadJSpace(args.JLENSPath, args.JSpaceTopK); err != nil {
			log.Fatalf("fucina: J-space initialization failed: %v", err)
		}
		log.Printf("fucina: J-space JSONL -> %s", args.JSpaceOut)
	}

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

	// Load the tokenizer. NVFP4 safetensors checkpoints ship a HuggingFace tokenizer.json
	// (BPE, no GGUF vocab) which loadTokenizer resolves from the model dir/sibling automatically
	// — so an NVFP4 model is self-contained; GGUF models carry their vocab inline. --tokenizer
	// overrides with either a .json or a .gguf. There is no tokenizer fallback: every mode
	// (server, REPL, one-shot) needs Encode/Decode, so fail fast and loud rather than start blind.
	tok, err := loadTokenizer(args.ModelPath, args.TokenizerPath)
	if err != nil {
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
		// Continuous batching is the DEFAULT: requests share a per-step scheduler over
		// the paged multi-sequence engine. SetBatchEngine self-disables (returns false)
		// when the engine has no paged pool (alloc failed → contiguous → seq_capacity 0),
		// so the single-flight path stays intact as the fallback. The route-guard
		// (batchIneligible) keeps grammar/repeat_penalty/custom-stop requests off the
		// batch path. Opt out with --no-cont-batching (Gemma only; Qwen3 needs paging).
		if !args.NoContBatching {
			if srv.SetBatchEngine(cuda.NewBatchAdapter(eng)) {
				log.Printf("fucina: continuous batching ENABLED (paged KV multi-sequence)")
			} else {
				log.Printf("fucina: paged pool unavailable — running single-flight")
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
		// The Qwen3 family and Qwen3.5 hybrid are served ONLY through the paged
		// multi-sequence path — their single-flight prefill entry points decline
		// (gemma-layout only), so the dense REPL's kv.Prefill fails with "prefill
		// failed". SeqFreeCapacity()>0 means the engine is in paged mode; route
		// those to the paged REPL (SeqAdd + StepBatch on one slot). Gemma/dense
		// single-flight stays on the original path.
		if eng.SeqFreeCapacity() > 0 {
			runInteractivePaged(eng, tok, args)
		} else {
			runInteractive(eng, tok, args)
		}
	} else {
		// ── One-shot prompt ───────────────────────────────────────────────
		runOneShot(eng, tok, args)
	}
}

func init() {
	// Ensure we use all CPU cores for preprocessing
	runtime.GOMAXPROCS(runtime.NumCPU())
}
