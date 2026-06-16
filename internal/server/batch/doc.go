// Package batch implements the continuous-batching scheduler for the fucina
// inference engine. It is Phase 4 of the paged-KV + continuous-batching rework:
// a single owner goroutine drives the engine and folds multiple in-flight
// sequences into shared per-step forward passes.
//
// # Why this exists
//
// Today the HTTP server (internal/server/server.go, serveCompletions) is
// single-flight: it takes s.kv.Lock() for the ENTIRE request — prefill through
// the last generated token — so concurrent requests serialize behind one lock.
// Time-to-first-token therefore scales linearly with the number of connected
// clients: the second client waits for the first client's whole generation
// before its prefill even starts.
//
// Continuous batching removes that serialization. Instead of each handler
// owning the engine for its whole request, ONE scheduler goroutine owns the
// engine and runs a step loop. Each step feeds one input token per active
// sequence through a single batched forward pass and scatters the sampled
// tokens back to the per-request emit callbacks. New sequences are admitted
// (prefilled and given a slot) as capacity frees up, and finished/cancelled
// sequences are evicted between steps, immediately freeing their slot for a
// queued request. Many clients then share decode steps and see TTFT bounded by
// queue depth and prefill cost, not by other clients' total generation length.
//
// # The model
//
//	            ┌─────────────────────────────────────────────┐
//	submit ───▶ │ Scheduler.run (single owner goroutine)      │
//	(channel)   │                                             │
//	            │  1. admit waiting reqs while a slot is free  │
//	            │     → BatchEngine.AddSeq (prefill prompt)    │
//	            │  2. build decode batch from active slots     │
//	            │     → BatchEngine.StepBatch(active, inputs)  │
//	            │  3. scatter one sampled token per slot to    │
//	            │     each sequence's emit callback            │
//	            │  4. advance budgets; mark stop-token /       │
//	            │     maxNew / ctx-cancel sequences finished   │
//	            │  5. evict finished → BatchEngine.RemoveSeq   │
//	            │     (slot freed for a queued request)        │
//	            └─────────────────────────────────────────────┘
//
// Exactly one goroutine ever calls the BatchEngine, so the engine itself needs
// no internal locking for the step loop — the concurrency boundary is the
// submit channel, not a mutex held across a multi-second request.
//
// # Backpressure to slow clients
//
// A batched step advances ALL active sequences together, so one slow consumer
// must not stall the shared step loop. Each sequence delivers tokens through a
// non-blocking emit callback (see Request.Emit): the scheduler never blocks the
// step loop on a client write. The reference wiring uses a buffered per-sequence
// channel whose sender drops (and cancels the sequence) when the buffer is full,
// so a wedged client is evicted rather than allowed to back up the GPU. The emit
// callback returns false to ask the scheduler to stop and evict the sequence.
//
// # How this will wire into server.go later (a separate stream owns that file)
//
// serveCompletions currently does, under s.kv.Lock():
//
//	pf, _ := s.kv.Prefill(tokens)
//	s.streamResponse(ctx, sse, params, ..., pf.Logits, ...)
//
// With the scheduler it instead submits the request and ranges over emitted
// tokens, holding no engine lock:
//
//	done := make(chan batch.Result, 1)
//	req := batch.Request{
//	    Tokens: tokens,
//	    Params: batch.SeqParams{Temperature: ..., TopK: ..., TopP: ..., MinP: ..., Seed: ...},
//	    Stops:  []int32{tok.EOS, tok.EndOfTurn},
//	    MaxNew: params.MaxTokens,
//	    Ctx:    r.Context(),
//	    Emit:   func(t int32) bool { /* SSE-write t; false on client backpressure */ return true },
//	    Done:   done,
//	}
//	if err := sched.Submit(req); err != nil { /* 503 busy */ }
//	res := <-done // finish reason + counts, after the scheduler evicts the slot
//
// The scheduler replaces the per-request engine lock entirely: admission
// control (today's s.inflight channel + 503) becomes the scheduler's bounded
// queue, and the prefill/decode split moves behind the BatchEngine interface so
// the CUDA side can batch the decode step across slots.
//
// # Scope of this phase
//
// This package is pure Go with no cgo and no GPU dependency. It is built and
// tested against the BatchEngine interface with a deterministic mock
// (scheduler_test.go); the CUDA side implements BatchEngine for real. The
// interface defined here is the contract between the two streams.
package batch
