package main

import (
	"flag"
	"fmt"
	"os"
)

// ─── Command-line flags (llama.cpp style) ─────────────────────────

type CLIArgs struct {
	// Model
	ModelPath     string
	TokenizerPath string // --tokenizer: GGUF to source the tokenizer vocab from (required when
	//                       -m is an NVFP4 safetensors checkpoint, which carries no GGUF vocab)
	AssistantPath string // Gemma-4 MTP assistant GGUF (official draft head)
	DiffModelPath string // -dm: diffusion model path; also enables the NVFP4 MoE experts
	FP4MoE        bool   // --fp4-moe / implied by -dm: enable DiffusionGemma NVFP4 MoE experts
	DenoiseSteps  int    // --denoise-steps: diffusion denoise-step cap per block (0 = default 48); lower = faster, lower quality

	// Inference
	ContextSize int
	BatchSize   int
	Threads     int

	// Sampling
	Temperature      float64
	TopK             int
	TopP             float64
	MinP             float64
	Seed             int64
	RepeatPenalty    float64
	FrequencyPenalty float64
	PresencePenalty  float64

	// Generation
	Prompt       string
	PromptFile   string
	Predict      int
	Keep         int
	NoDisplay    bool
	Spec         bool
	DraftK       int
	ThinkBudget  int     // server: reasoning-channel token budget (0=auto, <0=off)
	KVSnapshotGB float64 // server: host-memory budget for saved KV sequences (0 = off)
	CudaGraphs   bool    // --cuda-graphs (experimental, off by default)

	// Server
	Host          string
	Port          int
	Timeout       int
	Slots         int
	Thinking      string // gemma-4 reasoning channel default: off/on/low/mid/high/xhigh
	APIKey        string // optional bearer token required on /v1/* (empty = auth off)
	MaxConcurrent int    // admission-queue depth (in-flight + waiting); 0 = default
	MaxOutputToks int    // absolute per-request output-token ceiling; 0 = no extra cap
	PagedKV       bool   // --paged-kv: allocate the paged multi-sequence KV pools (sets FUCINA_PAGED_KV)
	Batch         bool   // --batch: route /v1/* through the continuous-batching scheduler (implies --paged-kv)

	// System
	System     string
	Verbose    bool
	Timings    bool
	DeviceID   int
	GPUMemUtil float64 // --gpu-mem-util: fraction of total device mem the engine may use (vLLM-style)
	Memory     string
	Debug      bool
	LogLevel   string

	// Mode
	Interactive bool
	Multiline   bool
	Color       bool
	Help        bool
}

// testFlags holds the test-only dispatch flags (--test-parser, --test-cuda,
// --test-vectors). They live outside CLIArgs because they trigger immediate
// exit-and-run behavior rather than configuring a normal run.
type testFlags struct {
	parser  bool
	cuda    bool
	vectors string
}

// parseArgs is the testable core of flag parsing: it registers every flag on
// the provided FlagSet and parses argv. It does NOT call os.Exit, does NOT do
// default-model lookup, and does NOT dispatch test commands — those side
// effects live in the parseFlags() wrapper. This lets tests drive a fresh
// FlagSet (e.g. with ContinueOnError) and inspect the result.
func parseArgs(fs *flag.FlagSet, argv []string) (CLIArgs, testFlags, error) {
	var a CLIArgs
	var t testFlags

	// Defaults matching llama.cpp
	fs.StringVar(&a.ModelPath, "m", "", "Path to GGUF model file")
	fs.StringVar(&a.ModelPath, "model", "", "Path to GGUF model file")
	fs.StringVar(&a.TokenizerPath, "tokenizer", "",
		"GGUF to load the tokenizer vocab from (required for an NVFP4 safetensors checkpoint)")
	fs.StringVar(&a.AssistantPath, "assistant", "",
		"Path to the Gemma-4 MTP assistant GGUF (official draft head; enables draft-mtp speculation)")
	// -dm: like -m but for DiffusionGemma — sets the model path AND enables the NVFP4 MoE
	// experts (CUTLASS grouped FP4 tensor cores, ~1.9× the dp4a denoise step).
	fs.StringVar(&a.DiffModelPath, "dm", "", "Diffusion model GGUF path (implies --fp4-moe)")
	fs.StringVar(&a.DiffModelPath, "diffusion-model", "", "Diffusion model GGUF path (implies --fp4-moe)")
	fs.BoolVar(&a.FP4MoE, "fp4-moe", false, "Enable DiffusionGemma NVFP4 MoE experts (works with -m too)")
	fs.IntVar(&a.DenoiseSteps, "denoise-steps", 0, "DiffusionGemma denoise steps per block (0=default 48; lower=faster, lower quality)")

	// Default to the model's maximum trained context (262144). A coding agent
	// injects large file contents via Read tool responses; a small window forces
	// constant compaction, which trims the oldest tokens and collapses the prefix
	// cache (longestCommonPrefix → ~1), re-prefilling the whole window every turn.
	// MEMORY COST at full context (FP8 KV, 1 B/elem, K+V): the global cache still
	// scales with ctx (8 layers × 512 head-dim × ctx ≈ 2.1 GiB @131k), but the
	// sliding cache is now a capped RING (default 8192 slots, FUCINA_SLIDING_RING):
	// 48 × 8 × 256 × 8192 × 2 ≈ 1.5 GiB, ctx-independent — was ~21 GiB flat @131k.
	// Sliding-window attention only reads the last 1024 positions, so the ring loses
	// nothing for same-conversation turns and speculation; only a prefix-reuse rewind
	// deeper than (ring-1024) tokens degrades to a full re-prefill. Engine clamps to 262144.
	fs.IntVar(&a.ContextSize, "ctx", 262144, "Context size in tokens")
	fs.IntVar(&a.ContextSize, "c", 262144, "Context size (short)")
	fs.IntVar(&a.BatchSize, "batch-size", 2048, "Logical maximum batch size")
	fs.IntVar(&a.Threads, "threads", 8, "Number of CPU threads for preprocessing")

	fs.BoolVar(&a.Spec, "spec", true, "Prompt-lookup speculative decoding (works at any temperature; Gemma-4 also gets MTP with --assistant)")
	fs.IntVar(&a.DraftK, "draft-k", 6, "Max speculative draft length per step")
	fs.IntVar(&a.ThinkBudget, "think-budget", 0,
		"Server: max reasoning-channel tokens per turn before the thought channel is force-closed (0 = auto: half of max_tokens; negative = unlimited)")
	fs.Float64Var(&a.KVSnapshotGB, "kv-snapshot-gb", 16,
		"Server: host-memory budget (GiB) for snapshotted KV sequences so one client cannot evict another's cached conversation (0 = off)")
	fs.BoolVar(&a.CudaGraphs, "cuda-graphs", false, "Enable CUDA graph support (experimental, allocates persistent prefill scratch)")
	// Defaults follow the google/gemma-4-12B model card's standardized sampling
	// configuration (temperature 1.0, top_p 0.95, top_k 64; no min-p). The GGUF
	// embeds the same values in general.sampling.{temp,top_k,top_p}.
	fs.Float64Var(&a.Temperature, "temp", 1.0, "Sampling temperature (gemma-4 default 1.0)")
	fs.IntVar(&a.TopK, "top-k", 64, "Top-K sampling (gemma-4 default 64)")
	fs.Float64Var(&a.TopP, "top-p", 0.95, "Top-P sampling (gemma-4 default 0.95)")
	fs.Float64Var(&a.MinP, "min-p", 0.0, "Min-P sampling (gemma-4 default off)")
	fs.Int64Var(&a.Seed, "seed", -1, "Random seed (-1 = random)")
	fs.Float64Var(&a.RepeatPenalty, "repeat-penalty", 1.0, "Repeat penalty (gemma-4 default: off)")
	fs.Float64Var(&a.FrequencyPenalty, "frequency-penalty", 0.0, "Frequency penalty")
	fs.Float64Var(&a.PresencePenalty, "presence-penalty", 0.0, "Presence penalty")

	fs.StringVar(&a.Prompt, "p", "", "Prompt string")
	fs.StringVar(&a.Prompt, "prompt", "", "Prompt string")
	fs.StringVar(&a.PromptFile, "f", "", "Prompt file")
	fs.StringVar(&a.PromptFile, "file", "", "Prompt file")
	fs.IntVar(&a.Predict, "n", 512, "Number of tokens to predict (-1 = infinite)")
	fs.IntVar(&a.Predict, "predict", 512, "Number of tokens to predict")

	fs.StringVar(&a.Host, "host", "127.0.0.1", "Server listen address")
	fs.IntVar(&a.Port, "port", 8080, "Server port")
	fs.IntVar(&a.Timeout, "timeout", 600, "Request timeout in seconds")
	fs.IntVar(&a.Slots, "n-slots", 1, "Number of processing slots")
	fs.StringVar(&a.Thinking, "thinking", "off",
		"Default gemma-4 reasoning channel: off|on|low|mid|high|xhigh (per-request reasoning_effort overrides)")
	fs.StringVar(&a.APIKey, "api-key", "",
		"Bearer token required on /v1/* (constant-time check). Empty = auth disabled (localhost dev). Reads FUCINA_API_KEY if unset.")
	fs.IntVar(&a.MaxConcurrent, "max-concurrent", 0,
		"Admission-queue depth (in-flight + waiting requests); excess requests get 503. 0 = default (4).")
	fs.IntVar(&a.MaxOutputToks, "max-output-tokens", 0,
		"Absolute per-request output-token ceiling (independent of context window). 0 = no extra cap.")
	// Continuous batching: serve concurrent requests in one batched forward pass via the
	// per-step scheduler over a paged multi-sequence KV cache, instead of the per-request
	// kv lock that serializes prefill+generation (TTFT scales linearly with clients).
	// OFF by default: the batch path has no MTP spec decode yet and pays a ~10% split-K
	// single-stream tax, so it is a deliberate opt-in for genuinely concurrent deployments.
	// Equivalent to the legacy FUCINA_PAGED_KV=1 FUCINA_BATCH=1 env pair.
	fs.BoolVar(&a.PagedKV, "paged-kv", false,
		"Allocate the paged multi-sequence KV pools (prerequisite for --batch; equivalent to FUCINA_PAGED_KV=1)")
	fs.BoolVar(&a.Batch, "batch", false,
		"Continuous batching: serve concurrent requests through the per-step scheduler (implies --paged-kv). OFF by default; no MTP spec decode in this path.")

	fs.StringVar(&a.System, "s", "", "System prompt")
	fs.StringVar(&a.System, "system", "", "System prompt")
	fs.BoolVar(&a.Verbose, "v", false, "Verbose output")
	fs.BoolVar(&a.Verbose, "verbose", false, "Verbose output")
	fs.BoolVar(&a.Timings, "timings", false, "Show timing information")
	fs.BoolVar(&a.Debug, "debug", false, "Dump full request bodies + rendered prompts to "+"/tmp/fucina_debug.log")
	fs.StringVar(&a.LogLevel, "log-level", "info", "Log level: info|debug (debug also dumps requests)")
	fs.IntVar(&a.DeviceID, "cuda-device", 0, "CUDA device ID")
	// GPU-memory budget (vLLM-style). The engine fits weights + KV + scratch +
	// (optionally) the packed-Q4_0 decode copy under this fraction of total device
	// memory, auto-capping ctx and dropping the packed copy as needed to satisfy it.
	// Default 0.90 is behavior-preserving where everything already fits (e.g. the
	// 128 GB GB10); lower it to share the GPU or fit a smaller host.
	fs.Float64Var(&a.GPUMemUtil, "gpu-mem-util", 0.90,
		"Fraction of total GPU memory the engine may use (0<F<=1); caps ctx / drops the packed-Q4_0 copy to fit")
	fs.StringVar(&a.Memory, "mlock", "", "mlock model in memory (unused on CUDA)")

	fs.BoolVar(&a.Interactive, "interactive-first", false, "Force interactive mode")
	fs.BoolVar(&a.Interactive, "interactive", false, "Force interactive mode")
	fs.BoolVar(&a.Color, "color", false, "Color output")

	fs.BoolVar(&a.Help, "h", false, "Show help")
	fs.BoolVar(&a.Help, "help", false, "Show help")

	// Test flags
	fs.BoolVar(&t.parser, "test-parser", false, "Run tokenizer tests and exit")
	fs.BoolVar(&t.cuda, "test-cuda", false, "Run CUDA tests and exit")
	fs.StringVar(&t.vectors, "test-vectors", "", "Run test vectors from file")

	if err := fs.Parse(argv); err != nil {
		return a, t, err
	}
	// -dm <model> is sugar for: -m <model> + --fp4-moe (DiffusionGemma NVFP4 MoE).
	if a.DiffModelPath != "" {
		a.ModelPath = a.DiffModelPath
		a.FP4MoE = true
	}
	// --batch needs the paged multi-sequence engine; the scheduler is a no-op without it.
	if a.Batch {
		a.PagedKV = true
	}
	return a, t, nil
}

// parseFlags preserves the exact original behavior: it parses os.Args using the
// global flag.CommandLine (ExitOnError), then performs --help exit, default
// model lookup, and test-flag dispatch — all of which may call os.Exit.
func parseFlags() CLIArgs {
	a, t, err := parseArgs(flag.CommandLine, os.Args[1:])
	if err != nil {
		// flag.CommandLine uses ExitOnError, so a parse error already exited;
		// this guard only fires if that policy ever changes.
		os.Exit(2)
	}

	if a.Help {
		printUsage()
		os.Exit(0)
	}

	// Model is required unless running tests
	if a.ModelPath == "" && !t.parser && !t.cuda && t.vectors == "" {
		// Check default locations: a GGUF file, or an NVFP4 safetensors checkpoint
		// (a single .safetensors file or a directory the C loader resolves via its
		// .index.json / single-shard handling).
		for _, p := range []string{
			"./gemma-4-12b-it.gguf", "./model.gguf", "./gguf/model.gguf",
			"./model.safetensors", "./gemma-4-12B-it-NVFP4",
		} {
			if _, err := os.Stat(p); err == nil {
				a.ModelPath = p
				break
			}
		}
		if a.ModelPath == "" {
			fmt.Fprintf(os.Stderr, "error: no model specified. "+
				"Use -m <model.gguf | nvfp4-dir | model.safetensors>\n\n")
			printUsage()
			os.Exit(1)
		}
	}

	// Handle test flags
	if t.parser {
		os.Exit(runTestParser())
	}
	if t.cuda {
		os.Exit(runTestCUDA())
	}
	if t.vectors != "" {
		os.Exit(runTestVectors(t.vectors))
	}

	return a
}
