package main

import "fmt"

// ─── Usage ────────────────────────────────────────────────────────

func printUsage() {
	fmt.Print(`fucina - Gemma 4 12B inference engine for DGX Spark GB10

Usage:
  fucina -m model.gguf [options]                  # Server mode (default)
  fucina -m model.gguf -p "Hello" -n 100          # One-shot prompt
  fucina -m ./gemma-4-12B-it-NVFP4 -p "Hi" -n 64  # NVFP4 safetensors checkpoint (dir)

Model options:
  -m, --model PATH           GGUF file, or an NVFP4 safetensors checkpoint (a directory,
                             an .index.json, or a single .safetensors). Auto-detected
                             from the file header (Q4_0/Q8_0/NVFP4).
  -dm, --diffusion-model FILE  DiffusionGemma GGUF; like -m but also enables the NVFP4 MoE
                             experts (CUTLASS grouped FP4 tensor cores, ~1.9x denoise)
  --fp4-moe                  Enable DiffusionGemma NVFP4 MoE experts (use with -m)
  --assistant FILE           Gemma-4 MTP assistant GGUF (official draft head; ~2x decode)

Inference options:
  --ctx N                    Context size (default: 262144, max: 262144)
  --batch-size N             Maximum batch size (default: 2048)
  --threads N                CPU threads (default: 8)

Sampling options:
  --temp F                   Temperature (gemma-4 default: 1.0)
  --top-k N                  Top-K sampling (gemma-4 default: 64)
  --top-p F                  Top-P sampling (gemma-4 default: 0.95)
  --min-p F                  Min-P sampling (gemma-4 default: 0.0/off)
  --seed N                   Seed (-1 = random)
  --repeat-penalty F         Repeat penalty (gemma-4 default: 1.0/off)

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
  --draft-k N                Speculative draft length per step (default: 6)
  --kv-snapshot-gb F         Host-memory budget (GiB) for snapshotted KV sequences
                             (multi-conversation prefix cache; 0 = off, default 16)
  --think-budget N           Max reasoning tokens per turn before the thought
                             channel is force-closed (0 = auto: max_tokens/2;
                             negative = unlimited)
  --max-concurrent N         Admission-queue depth (in-flight + waiting); excess
                             requests get 503 (0 = default 4)
  --paged-kv                 Allocate the paged multi-sequence KV pools
                             (prerequisite for --batch)
  --batch                    Continuous batching: serve concurrent requests in one
                             batched forward pass (implies --paged-kv). OFF by
                             default — no MTP spec decode in this path, ~10%
                             single-stream tax; opt in for concurrent serving.

Other options:
  --cuda-device N            CUDA device ID (default: 0)
  --gpu-mem-util F           Fraction of total GPU memory the engine may use
                             (0<F<=1, default 0.90; caps ctx / drops the packed
                             Q4_0 decode copy to fit the budget, vLLM-style)
  -v, --verbose              Verbose output
  --timings                  Show detailed GPU timing (prefill/decode tok/s)
  -h, --help                 Show this help

Examples:
  fucina -m gemma-4-12b-it.gguf --ctx 32768 &
  curl http://localhost:8080/v1/chat/completions \\
    -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
`)
}
