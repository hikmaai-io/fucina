---
type: Architecture
title: Serving and scheduling architecture
description: OpenAI-compatible HTTP routing, continuous batching, state reuse, and session behavior.
tags: [server, scheduler, batching, paged-kv, sessions]
status: draft
stale_after: 2026-08-25
generated: { by: openai-codex/gpt-5.6-sol, at: 2026-07-25T10:33:01+02:00 }
snapshot_commit: 39a96dbd4856f394821021efa10ef31848ad2581
sources:
  - id: server
    resource: ../../../internal/server/server.go
    title: HTTP server implementation
  - id: scheduler
    resource: ../../../internal/server/batch/scheduler.go
    title: Continuous-batching scheduler
  - id: batching-doc
    resource: ../../continuous-batching.md
    title: Continuous batching design
  - id: sessions
    resource: ../../session-persistence.md
    title: Session persistence design
---

# Request paths

* **Gemma single-flight:** default path with the broadest feature set, including host constrained decoding, prefix KV reuse, and HTTP disk sessions.
* **Continuous batching:** mandatory for Qwen3.5/3.6 and optional for Gemma. One scheduler goroutine owns admission, chunked prefill, decode steps, cancellation, and eviction.
* **E4B batching:** separate adapter and CUDA kernel; currently greedy in batch mode.
* **DiffusionGemma:** separate engine and server wiring with block-level generation behavior.

# State models

Gemma single-flight uses a physical KV cache plus snapshots and prefix matching. Qwen hybrid batching uses independent slots containing full-attention KV plus Gated-DeltaNet recurrent and convolution state. Burst coalescing and chunked admission are optimized for shared decode progress rather than one request holding the engine for its full lifetime.[^batching-doc]

[^batching-doc]: Continuous batching design

# Feature asymmetry

| Feature | Gemma single-flight | Continuous batch / Qwen | E4B batch |
|---|---:|---:|---:|
| Concurrent independent sequences | No | Yes | Yes |
| Per-request sampling parameters | Yes | Yes for main CUDA adapter | No; greedy only |
| JSON constrained decoding | Yes | No; HTTP 501 | No batch integration |
| Disk session through HTTP | Yes | No | No |
| Prompt-lookup speculation | Yes | Dense batch rows only | Separate E4B paths |
| Gemma MTP assistant | Yes | Not implemented per slot | E4B-specific MTP path exists |

# API safety

The primary server includes request limits, readiness/liveness endpoints, metrics, optional bearer authentication, and request IDs. These are lab-grade interfaces rather than a portability or support commitment.
