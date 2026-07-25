# Phase E: multi-Spark distributed inference

**Status (2026-07-25): integrated experimental boundary; not multi-node complete.**

Phase E splits Qwen3.5 layers across a coordinator and workers. This increment
lands the exact layer-cut engine boundary, cgo runner, hardened TCP protocol,
and a one-shot coordinator/worker CLI. It deliberately does **not** claim that
a model too large for one GB10 can run yet: every process still loads the full
checkpoint because range-filtered weight loading/residency is not implemented.
The OpenAI HTTP scheduler is also still single-node.

## Implemented topology

The coordinator owns tokenization and sampling, embeds one token locally, runs
its `[0,k)` layer range, and sends the fp32 residual through workers in layer
order. The final worker applies output norm + the LM head and returns logits.
Every shard owns its layer-local FULL-attention KV and GDN recurrent state.

The integrated CUDA ABI is:

- `gemma4_engine_q35_embed`: token id to fp32 residual;
- `gemma4_engine_q35_forward_layers`: exact `[lo,hi)` residual-to-residual
  forward, or logits when the final range ends at `n_layers`;
- `internal/engine/cuda.CUDAShardRunner`: maps protocol sequence ids to engine
  slots and validates monotonic positions and reset/reuse.

Only `ntokens=1` is accepted. Prompt prefill is therefore token-sequential. This
is intentionally slow but correct for Qwen3.5's ordered GDN recurrence and
causal attention. Treating a prompt's rows as independent would be corruption,
so the boundary rejects it rather than pretending batched distributed prefill
exists. The ordinary single-node forward still calls the original whole-model
wrapper with the same layer order, kernels, graph path, and arithmetic.

## FCNDIST1 TCP protocol

`internal/dist` uses a persistent `TCP_NODELAY` connection. Both sides exchange
versioned magic `FCNDIST1` and a JSON `Hello` pinning:

- protocol version;
- FNV-1a64 model-config identity (`session.HashModelConfig`);
- contiguous layer range;
- hidden width;
- activation dtype (fp32 is the integrated CUDA mode).

Frames are `type u32 | payloadLen u32 | payload | fnv1a64(payload) u64`.
Lengths are capped before allocation, short writes are completed, checksums are
verified, and activation byte counts must exactly equal
`n_tokens * hidden * dtype_bytes`. Replies must echo sequence/position/token
identity. A worker rejects gaps, stale/out-of-order positions, dtype/model
mismatches, and malformed frames before running CUDA.

`SeqReset` is synchronous: the coordinator waits for `Ack` after worker state is
dropped. Disconnect resets every sequence still owned by that connection, so a
reconnect cannot silently inherit stale KV/GDN state. `Ping/Pong` gates route
startup. TCP keepalive is enabled; RDMA remains a future transport under the
same logical boundary.

## CLI (experimental one-shot only)

Example two-process split of a 32-layer checkpoint:

```bash
# worker / node 2
./fucina -m /models/Qwen3.5-9B-FP8 \
  --dist-listen 0.0.0.0:9091 --dist-layers 16:32 --dist-final

# coordinator / node 1
./fucina -m /models/Qwen3.5-9B-FP8 \
  --dist-layers 0:16 --dist-workers node2:9091 \
  -p 'The capital of France is' -n 8 --temp 0
```

More workers may be supplied comma-separated in contiguous layer order. Startup
fails on any model hash, hidden size, dtype, or layer-range mismatch. Distributed
mode currently requires `-p`/`-f`; it does not expose the HTTP server, continuous
batching, speculative decode, sessions, or prefix caching.

## OKF: objective, key results, falsification gates

**Objective.** Establish an exact and lifecycle-safe Qwen3.5 layer boundary that
can become a real multi-GB10 backend without changing single-node behavior.

**Delivered key results.**

1. Real CUDA `[lo,hi)` execution uses the production Qwen3.5 mixer/FFN kernels
   and per-slot FULL KV/GDN state; final projection returns real logits.
2. The cgo runner and FCNDIST1 worker reject unsupported batching, topology,
   geometry, stale positions, and sequence reuse races.
3. The host-only protocol/pipeline suite covers hostile lengths/checksums,
   partial writes, handshakes, exact geometry, reset ACK, liveness, contiguous
   routes, and stale positions over `net.Pipe`.
4. `make qwen35-shard-test` crosses a real host fp32 frontier at layer 16 on the
   Qwen3.5-9B-FP8 engine and requires all final-logit bytes plus the 8-token FP8
   oracle stream to equal an unsplit `[0,32)` boundary.

**Falsification / hardware gates.** Phase E is not complete until all of these
are recorded on the target topology:

- **Single GB10 integrated cut (available and passed 2026-07-25):**
  `make qwen35-shard-test`. Observed cut `[0,16)+[16,32)`, byte-identical logits
  on 8 steps, oracle ids
  `11751 13 198 760 6511 314 9338 369` (8/8).
- **Local TCP orchestration (passed 2026-07-25, not a multi-node gate):** two
  processes on one GB10, ranges `0:16` and `16:32`, full FP8 checkpoint loaded
  by both, `FCNDIST1` over loopback. The five-token prompt prefetched at 15.5
  tok/s and generated ` Paris.` at 30.8 tok/s. This proves CLI/TCP/cgo wiring,
  not distributed hardware, capacity, or speedup.
- **Single-node regression:** `make gpu-gates` and the normal Go/cgo tests; no
  existing generated-token or state byte may change.
- **Two physical GB10s, same fitting checkpoint (NOT RUN):** run the CLI above,
  compare a fixed greedy prompt corpus token-for-token with single-node, inject
  reset/reconnect/wrong-hash/wrong-position failures, and record TTFT/decode.
- **Range-filtered residency (NOT IMPLEMENTED):** each process must prove that
  only its layer weights plus required embed/head tensors are resident. Until
  then Phase E cannot run a checkpoint that fails to fit one node.
- **Two-node capacity gate (NOT RUN / BLOCKED by residency):** run a Qwen3.5 MoE
  that cannot fit one GB10 and complete generation without host/SSD weight
  fallback masquerading as layer sharding.
- **Performance gate (NOT RUN):** record same-hardware 1-node vs 2-node prompt
  prefill, decode latency, and batched throughput. No speedup is claimed; the
  current token-sequential TCP path is expected to be latency-bound.

## Remaining work

1. Filter safetensors/GGUF tensor upload and Qwen workspace/state allocation by
   layer range; coordinator alone owns embedding and final shard alone owns head.
2. Add exact ordered multi-token prefill (bounded chunks/credits) with frontier
   and recurrent-state identity gates before enabling any pipeline overlap.
3. Integrate a distributed backend behind the server batch/session interfaces,
   including cancellation, state snapshots, and topology-neutral recovery.
4. Run and archive the physical two-node correctness, failure-injection,
   capacity, and performance gates above; only then update status to complete.
