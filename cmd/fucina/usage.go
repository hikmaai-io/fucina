package main

import "fmt"

// ─── Usage ────────────────────────────────────────────────────────

func printUsage() {
	fmt.Print(`fucina - Gemma 4 and Qwen3/Qwen3.5/Qwen3.6 inference engine for DGX Spark GB10

Usage:
  fucina -m model.gguf [options]                     # Server mode (default)
  fucina -m model.gguf -p "Hello" -n 100             # One-shot prompt
  fucina -m ./gemma-4-12B-it-NVFP4 -p "Hi" -n 64     # Gemma-4 NVFP4 safetensors checkpoint (dir)
  fucina -m ./Qwen3.6-35B-A3B-FP8 --host 0.0.0.0     # Qwen checkpoint; batching auto-enabled

Model detection: architecture (Gemma-4 / Qwen3 dense / Qwen3 MoE / Qwen3.5-3.6 hybrid dense or
MoE) and weight format (Q4_0/Q8_0/Q4_K GGUF, official FP8-block safetensors, NVFP4 generic or
NVIDIA ModelOpt mixed) are auto-detected from the file itself. There is no --model-type flag.

Model options:
  -m, --model PATH           GGUF file, or a safetensors checkpoint (a directory, an
                             .index.json, or a single .safetensors) — Gemma-4 NVFP4,
                             Qwen3.5/3.6 official FP8-block, or NVIDIA ModelOpt NVFP4/FP8.
  --tokenizer PATH           tokenizer.json (HF BPE) or .gguf to source vocab from.
                             Only needed if auto-discovery (a tokenizer.json sibling of
                             -m) fails — e.g. -m points at a HF hub-cache repo root
                             instead of its snapshots/<hash> dir. GGUF models never need
                             this; their vocab is inline.
  -dm, --diffusion-model FILE  DiffusionGemma GGUF; like -m but also enables the NVFP4 MoE
                             experts (CUTLASS grouped FP4 tensor cores, ~1.9x denoise)
  --fp4-moe                  Enable DiffusionGemma NVFP4 MoE experts (use with -m)
  --assistant FILE           Gemma-4 MTP assistant GGUF (official draft head; ~2x decode).
                             Gemma-4 only; has no effect on Qwen checkpoints.

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
                             (prerequisite for --batch). Auto-forced on for any
                             Qwen3/3.5/3.6 checkpoint; no-op to pass explicitly there.
  --batch                    Continuous batching: serve concurrent requests in one
                             batched forward pass (implies --paged-kv). OFF by
                             default for Gemma-4 — no MTP spec decode in this path,
                             ~10% single-stream tax; opt in for concurrent Gemma-4
                             serving. Auto-forced ON for every Qwen3/3.5/3.6
                             checkpoint (no single-flight path exists for Qwen).

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
  fucina -m ./Qwen3.6-35B-A3B-FP8 --host 0.0.0.0 --port 8080 &
  curl http://localhost:8080/v1/chat/completions \\
    -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'

See llms.txt / README.md for the full model support matrix, checkpoint download commands,
tool-calling examples, and diagnostics.
`)
}
