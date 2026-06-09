package main

import "fmt"

// ─── Usage ────────────────────────────────────────────────────────

func printUsage() {
	fmt.Print(`gem4d - Gemma 4 12B inference engine for DGX Spark GB10

Usage:
  gem4d -m model.gguf [options]                  # Server mode (default)
  gem4d -m model.gguf -p "Hello" -n 100          # One-shot prompt

Model options:
  -m, --model FILE           Path to GGUF model file (Q4_0-QAT or Q8_0; auto-detected)
  --assistant FILE           Gemma-4 MTP assistant GGUF (official draft head; ~2x decode)

Inference options:
  --ctx N                    Context size (default: 4096, max: 262144)
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

Other options:
  --cuda-device N            CUDA device ID (default: 0)
  -v, --verbose              Verbose output
  --timings                  Show detailed GPU timing (prefill/decode tok/s)
  -h, --help                 Show this help

Examples:
  gem4d -m gemma-4-12b-it.gguf --ctx 32768 &
  curl http://localhost:8080/v1/chat/completions \\
    -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'
`)
}
