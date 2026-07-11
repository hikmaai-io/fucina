# Phase E: multi-Spark distributed inference

Phase E splits the model's layers across N nodes (layer sharding / pipeline
parallelism) so a Qwen3.5 MoE too large for one GB10 runs across two or more,
and a model that fits one node can still gain throughput from the pipeline.

## Topology

One **coordinator** + N-1 **workers**. The coordinator owns the tokenizer,
sampler, chat template, and the OpenAI-compatible server surface — workers are
headless layer executors. Every node loads only its layer range's weights
(the loader already maps tensors by name, so a range filter is cheap).

At each shard boundary the only tensor crossing the wire is the residual
hidden activation: `n_tokens x hidden` (hidden 2048 for 35B-A3B, BF16 =
4 KiB/token/hop). Decode is 1 token/step, so per-hop latency — not bandwidth —
dominates; the transport must be a persistent connection with zero
per-message allocation in the steady state.

Transport tiers, best-first (the DS4 pattern):

1. **RDMA / ConnectX 200GbE** — the Spark's fast fabric (follow-up; the wire
   protocol is transport-agnostic so this slots under the same framing).
2. **TCP** with `TCP_NODELAY` — the portable baseline implemented first.

## Wire protocol (E1) — `internal/dist`

Versioned magic `FCNDIST1`, exchanged once per connection in a JSON `Hello`
handshake that pins: protocol version, model config hash (same FNV-1a64 the
session format uses), layer range, hidden size, and activation dtype. A
mismatch on any of these refuses the connection — a shard computing with the
wrong model or the wrong layer split must fail loudly at connect time, never
silently at token time.

After the handshake, framed binary messages:

    type u32 | payloadLen u32 | payload | fnv1a64(payload) u64

Every length read off the wire is bounds-checked against a hard cap before
allocation (the safetensors-loader / session-format standard). Message types:

- `Activation` — seq id, position, token count, dtype + raw hidden bytes.
- `Logits` — final-shard result returned to the coordinator.
- `SeqReset` — drop a sequence's KV/recurrent state on every shard.
- `Ping`/`Pong` — liveness.

## Pipeline (E2)

The coordinator drives: embed+shard-0 locally, forward the activation through
each worker in layer order, receive logits from the last shard, sample, loop.
Each worker holds its own KV cache + GDN recurrent state for its layer range;
`SeqReset` fans out on sequence end. The `ShardRunner` interface isolates the
CUDA engine so the pipeline is unit-testable over `net.Pipe` with a fake
runner (same reason the residency controller tests run without a checkpoint).

Engine boundary (CUDA side, follow-up): a partial-forward entry point
`layers [lo,hi) : hidden in -> hidden out` on the existing engine — the layer
loop already exists; the cut points are the residual stream between layers.

## Bench gates (exit criteria)

- 2-node pipeline runs a model that fits one node with measured speedup
  recorded (DS4 saw 1.4-1.85x on 2 nodes; decode is latency-bound so treat
  parity as acceptable, throughput-under-batch as the win).
- A Qwen3.5 MoE too large for one GB10 runs across two at all.
- Committed-token parity vs single-node for a fixed greedy prompt set.
