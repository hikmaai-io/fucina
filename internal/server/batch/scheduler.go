package batch

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// ─── Engine contract ───────────────────────────────────────────────
//
// BatchEngine is the consumer-side view of the future batched engine that the
// scheduler drives. It is declared HERE (where it is consumed), per Go best
// practice, so the scheduler is tested with a mock and no GPU. The CUDA side
// implements it for real.
//
// The interface refines the original design sketch in three ways, each forced
// by correctness of the step loop:
//
//  1. AddSeq returns a *first sampled token*, not raw logits. Sampling is
//     on-device per row (the design says so for StepBatch), so prefill must
//     sample on-device too — otherwise the host would need the logits buffer
//     and a CPU sampler, defeating the point. Returning the first token also
//     makes prefill and decode symmetric: prefill yields token[0], each step
//     yields token[n+1]. The token is what the scheduler emits and what it
//     feeds back as the next step's input for that slot.
//
//  2. StepBatch takes (active, inputs) of EQUAL length and the caller
//     guarantees inputs[i] is the token to advance slot active[i]. The engine
//     does one forward pass over exactly those rows and returns one sampled
//     token per row, out[i] for slot active[i]. Keeping active explicit (rather
//     than "all slots") lets the scheduler shrink the batch the instant a
//     sequence is evicted, without the engine tracking liveness — the scheduler
//     already owns that bookkeeping.
//
//  3. Capacity() is queried every admission pass (dynamic), so the engine may
//     grow/shrink the slot count (e.g. paged-KV pressure) and the scheduler
//     adapts without a restart.
//
// All methods are called from the SINGLE scheduler goroutine, so the engine
// needs no internal locking for the step loop.
type BatchEngine interface {
	// AddSeq admits a new sequence: it prefills prompt into a fresh slot and
	// returns the slot handle plus the FIRST sampled token for that sequence
	// (the token the scheduler emits first and feeds back as the next input).
	// An error means admission failed (e.g. out of KV blocks); no slot is
	// consumed in that case.
	AddSeq(prompt []int32, params SeqParams) (slot int, first int32, err error)

	// StepBatch runs ONE batched decode step. active is the list of slot ids to
	// advance and inputs is the matching input token per slot (len(inputs) ==
	// len(active), inputs[i] advances slot active[i]). It returns a RUN of freshly
	// sampled tokens per slot: out[i] is the ordered list of tokens produced for
	// slot active[i] this step. A plain (non-speculative) step returns a run of
	// length 1 per slot; a per-sequence speculative-decode step may return a
	// VARIABLE number of accepted tokens per slot (>=1 normally). A run beginning
	// with the -1 sentinel (out[i] == []int32{-1}) signals that slot can no longer
	// grow its KV and must be stopped gracefully. An empty run (len(out[i]) == 0)
	// is treated as "no progress this step" and the slot is left active.
	StepBatch(active []int32, inputs []int32) (out [][]int32, err error)

	// RemoveSeq frees a slot's KV blocks when its sequence finishes or is
	// cancelled. After it returns the slot id may be reused by a later AddSeq.
	RemoveSeq(slot int) error

	// Capacity reports the current maximum number of concurrent slots. It is
	// queried every admission pass, so the engine may report a dynamic value.
	Capacity() int
}

// SeqParams carries the per-sequence sampling configuration the engine applies
// on-device. It mirrors the knobs the server already validates and clamps
// (see server.GenerationParams) but is decoupled from the HTTP types so the
// CUDA side does not import the server package.
type SeqParams struct {
	Temperature   float32
	TopK          int
	TopP          float32
	MinP          float32
	RepeatPenalty float32
	Seed          uint64
}

// SpecReq is one row of a speculative batched-decode step (see SpecBatchEngine):
// feed Anchor to Slot and verify Drafts behind it. The engine commits the accepted
// prefix of Drafts plus one bonus token, losslessly.
type SpecReq struct {
	Slot   int32
	Anchor int32
	Drafts []int32
}

// SpecBatchEngine is the optional speculative extension of BatchEngine. When the
// concrete engine implements it, the scheduler drafts tokens per slot (prompt-lookup
// over each sequence's history) and verifies them in ONE batched forward via
// StepBatchSpec — committing the accepted run per slot (one weight pass, multiple
// tokens) instead of one token per slot. out[i] is the committed run for reqs[i].Slot,
// in request order; a run beginning with the -1 sentinel signals that slot must stop
// (KV could not grow), exactly like StepBatch. The verify is LOSSLESS: the committed
// run is byte-identical to what StepBatch would emit for that slot.
type SpecBatchEngine interface {
	BatchEngine
	StepBatchSpec(reqs []SpecReq) (out [][]int32, err error)
}

// SpecGater is the optional per-model gate on speculative decoding. An engine that
// implements it can decline speculation for models where a batched verify does not
// pay for itself — sparse/MoE models are the canonical case: every drafted token
// routes to its OWN top-k experts, so the dominant expert weight reads scale with
// the draft length instead of amortizing like a dense model's single weight pass.
// Engines that don't implement SpecGater keep speculation default-on.
type SpecGater interface {
	SpecWorthwhile() bool
}

// PrefillChunkHinter is the optional engine-declared prefill-chunking preference.
// Returns (chunkSize, chunkMin); size <= 0 keeps the scheduler defaults. Sparse/MoE
// FP8 engines return wide values: each prefill pass pays a per-layer expert-slab
// dequant, so wide passes amortize it (a 2k prompt in 256-token chunks paid it 8x).
type PrefillChunkHinter interface {
	PrefillChunkHint() (size, min int)
}

// ChunkPrefillEngine is the optional CHUNKED-PREFILL extension of BatchEngine. When
// the concrete engine implements it, the scheduler prefills LONG prompts in bounded
// chunks INTERLEAVED with decode steps of the already-active sequences, instead of
// running each prefill to completion (which blocks the whole batch and makes TTFT grow
// ~linearly with concurrency). Short prompts keep the one-shot AddSeq fast path.
//
// The lifecycle is: OpenSeq reserves a slot with empty KV → PrefillChunk is called
// repeatedly to append the prompt in pieces (the LAST call samples and returns the
// sequence's first generated token) → the slot then joins the normal decode batch via
// StepBatch. RemoveSeq frees it at any stage.
//
// LOSSLESSNESS (cardinal): OpenSeq + PrefillChunk over the whole prompt MUST produce a
// KV cache and a first token byte-identical to AddSeq(prompt). The chunk boundary is
// invisible to the model — each token is forwarded at the same absolute position with
// the same per-token computation regardless of how the prompt is split — so chunking
// changes only WHEN work happens, never WHAT is generated.
type ChunkPrefillEngine interface {
	BatchEngine
	// OpenSeq reserves a fresh slot, adopts the longest cached prefix of prompt into
	// its KV (cross-request prefix cache), and reports nShared = the number of prompt
	// tokens already satisfied by that adopted prefix, so the caller chunk-prefills
	// only prompt[nShared:]. It does NOT prefill the suffix. err means no slot was
	// available (no slot is consumed in that case). nShared is 0 on a cache miss.
	OpenSeq(prompt []int32, params SeqParams) (slot int, nShared int, err error)
	// PrefillChunk appends chunk to slot's KV at the slot's current position. When
	// last is true it additionally samples the sequence's FIRST generated token and
	// returns it in first (the token the scheduler emits and feeds back as the next
	// step's input); when last is false first is unused. A non-nil error means the
	// prefill failed (e.g. KV could not grow) and the caller frees the slot.
	PrefillChunk(slot int, chunk []int32, last bool) (first int32, err error)
}

// PrefixCacheStatsEngine is the optional observability hook for the cross-request
// prefix cache: an engine that can report cumulative reuse counters. The scheduler
// snapshots them lock-free for /metrics.
type PrefixCacheStatsEngine interface {
	// PrefixCacheStats returns cumulative counters: lookups (prefix probes at admit),
	// hitBlocks (KV blocks reused instead of re-prefilled), cachedBlocks (live tree
	// size), evictions. hitBlocks*blockTokens is the prefill work saved.
	PrefixCacheStats() (lookups, hitBlocks, cachedBlocks, evictions int64)
}

// PrefixCommitEngine is the optional decode-time registration hook: an engine that
// can register a slot's completed full blocks (prompt + generated text) from its
// committed token history, so a later request extending this sequence reuses the
// generated KV. Idempotent; the scheduler calls it as a sequence crosses 256-token
// boundaries.
type PrefixCommitEngine interface {
	PrefixCommit(slot int, history []int32)
}

// defaultPrefillChunk is the prompt-token budget committed per scheduler pass for a
// chunked (interleaved) prefill. Each pass commits at most one chunk for ONE prefilling
// sequence AND runs one decode step, so decode of the active sequences keeps flowing
// while a long prompt prefills. The engine's chunked prefill is token-by-token (one
// weight pass per ≤GEMMA4_MAX_SEQS=16 tokens via the batched suffix-prefill), so the
// chunk size trades the new sequence's prefill latency against the active sequences'
// per-step decode latency; 256 = ~16 batched weight passes per chunk, still interleaving
// a decode step between chunks while keeping per-chunk overhead amortized.
const defaultPrefillChunk = 256

// defaultPrefillChunkMin is the prompt length AT OR BELOW which prefill stays on the
// one-shot AddSeq path (no chunking). It sits at the engine's one-shot fast-prefill cap
// (4096 tokens for both the Gemma and Qwen3 paged-prefill paths): at/below it the fast
// single-pass prefill is used, so chunking never regresses it; ABOVE it the engine
// already falls back to a blocking token-by-token prefill — exactly the case chunking
// turns into an interleaved, non-batch-blocking one.
const defaultPrefillChunkMin = 256

// maxOneShotAdmitsPerPass caps blocking one-shot AddSeq admissions per scheduler pass.
// A one-shot AddSeq prefills synchronously before any decode step, so admitting a whole
// burst in one pass serializes all their prefills ahead of decode (head-of-line TTFT
// blowup at concurrency). Capping to 1 forces a decode step between admissions; longer
// prompts use the chunked interleave path which is not capped (opening is cheap).
const maxOneShotAdmitsPerPass = 1

// MaxVerifyRows is the engine's per-step verify-row budget (GEMMA4_MAX_SEQS in the CUDA
// engine): the speculative step flattens to Σ(1+len(drafts)) forward rows, which must not
// exceed it. With R active slots, R rows are anchors and the rest is the draft budget — so
// speculation naturally tapers as concurrency rises (anchors crowd out drafts), which
// matches where spec pays off (low concurrency / single-stream latency).
const MaxVerifyRows = 16

// ─── Public request type ───────────────────────────────────────────

// Request is one unit of work submitted to the scheduler. It is the full
// per-request lifecycle input: the prompt, sampling params, stop ids, the
// generation budget, a cancellation context, the per-token emit sink, and a
// Done channel that receives the terminal Result.
type Request struct {
	// Tokens is the already-tokenized prompt to prefill.
	Tokens []int32
	// Params is the on-device sampling configuration for this sequence.
	Params SeqParams
	// Stops are token ids that end generation when sampled (e.g. EOS,
	// end-of-turn). The stop token itself is delivered to Emit before eviction.
	Stops []int32
	// MaxNew bounds generated tokens for this sequence (a runaway cap). Values
	// < 1 are treated as 1 so every admitted sequence makes progress.
	MaxNew int
	// Ctx cancels the sequence: once Ctx.Done() fires the scheduler evicts it
	// (freeing its slot) before the next step.
	Ctx context.Context
	// Emit receives each sampled token for this sequence, called from the
	// scheduler goroutine. It MUST NOT block: a slow client must not stall the
	// shared step loop. Return false to ask the scheduler to stop and evict
	// this sequence (e.g. client backpressure / disconnect detected by the
	// emit sink). A typical implementation does a non-blocking send to a
	// buffered per-sequence channel and returns false when that channel is full.
	Emit func(token int32) bool
	// Done receives the terminal Result exactly once when the sequence is
	// evicted, for any reason. It MUST be buffered (cap >= 1) so the scheduler
	// never blocks delivering it. A nil Done is allowed (fire-and-forget).
	Done chan<- Result
}

// FinishReason explains why a sequence was evicted.
type FinishReason string

const (
	// FinishStop: a stop token was sampled.
	FinishStop FinishReason = "stop"
	// FinishLength: the MaxNew budget was exhausted.
	FinishLength FinishReason = "length"
	// FinishCancelled: the request context was cancelled, or Emit returned false.
	FinishCancelled FinishReason = "cancelled"
	// FinishError: an engine error aborted the sequence (see Result.Err).
	FinishError FinishReason = "error"
	// FinishShutdown: the scheduler shut down before the sequence completed.
	FinishShutdown FinishReason = "shutdown"
)

// Result is delivered on Request.Done when a sequence is evicted.
type Result struct {
	// Reason is why the sequence ended.
	Reason FinishReason
	// Generated is the number of tokens emitted for this sequence.
	Generated int
	// Err is the underlying engine error when Reason is FinishError, else nil.
	Err error
}

// ─── Errors ────────────────────────────────────────────────────────

// ErrShutdown is returned by Submit after the scheduler has been shut down.
var ErrShutdown = errors.New("batch: scheduler shut down")

// ErrQueueFull is returned by Submit when the bounded submit queue is full —
// the caller should shed load (e.g. respond 503), exactly as the current
// server's inflight admission channel does today.
var ErrQueueFull = errors.New("batch: submit queue full")

// ─── Per-sequence scheduler state ──────────────────────────────────

// seq is the scheduler's private bookkeeping for one admitted sequence. It is
// touched only by the scheduler goroutine, so it needs no synchronization.
type seq struct {
	req       Request
	slot      int   // engine slot id (valid once admitted)
	next      int32 // the input token to feed this slot on the next StepBatch
	remaining int   // generation budget left (counts down from MaxNew)
	generated int   // tokens emitted so far
	stops     map[int32]struct{}
	// prefillPos counts how many of req.Tokens have already been committed to the
	// slot's KV during a chunked prefill. It is meaningful only while the sequence is
	// in the prefill backlog (phasePrefill); once prefillPos == len(req.Tokens) the
	// prompt is fully prefilled and the sequence joins the active decode set.
	prefillPos int
	// hist is the sequence's full token history (prompt + every committed token),
	// the corpus the prompt-lookup drafter scans for repeated n-grams. Populated on
	// the speculative path AND when the engine has a prefix cache (so generated text
	// can be registered for cross-request reuse); nil otherwise so the plain path
	// pays nothing.
	hist []int32
	// regBlocks is how many full 256-token blocks of hist have been registered with
	// the prefix cache; used to call PrefixCommit only when a new block completes.
	regBlocks int
}

// stopHit reports whether t is one of this sequence's stop tokens.
func (s *seq) stopHit(t int32) bool {
	_, ok := s.stops[t]
	return ok
}

// ─── Scheduler ─────────────────────────────────────────────────────

// Scheduler owns a BatchEngine on a single goroutine and folds multiple
// in-flight sequences into shared per-step forward passes. Construct it with
// New, start it with Start, feed it with Submit, and stop it with Shutdown.
type Scheduler struct {
	engine BatchEngine

	// spec is the speculative-decode fast path: non-nil when engine implements
	// SpecBatchEngine. When set, step() drafts per-slot tokens (prompt-lookup over
	// each sequence's history) and verifies them in one batched forward, committing
	// the accepted run per slot — one weight pass commits multiple tokens. nil →
	// the plain one-token-per-slot path. draftK bounds the per-slot draft length.
	spec   SpecBatchEngine
	draftK int

	// chunk is the chunked-prefill fast path: non-nil when engine implements
	// ChunkPrefillEngine. When set, a prompt longer than chunkMin is prefilled in
	// chunkSize-token chunks INTERLEAVED with decode steps (one chunk + one decode
	// step per scheduler pass), so a long prefill no longer blocks the whole batch.
	// Shorter prompts use the one-shot AddSeq fast path. nil → every prompt uses
	// the one-shot path (the original behavior).
	chunk     ChunkPrefillEngine
	chunkSize int
	chunkMin  int

	// prefixStats is the cross-request prefix-cache observability hook: non-nil when
	// the engine implements PrefixCacheStatsEngine. The run loop (the sole engine
	// caller) snapshots the engine counters into the atomics below each pass so
	// /metrics can read them lock-free without touching the engine mutex.
	prefixStats PrefixCacheStatsEngine
	pcLookups   atomic.Int64
	pcHitBlocks atomic.Int64
	pcCached    atomic.Int64
	pcEvictions atomic.Int64

	// prefixCommit, when non-nil, registers generated text for cross-request reuse:
	// the scheduler tracks each sequence's committed history and calls it as the
	// sequence crosses 256-token block boundaries.
	prefixCommit PrefixCommitEngine

	// submit carries new requests to the owner goroutine. It is bounded; a full
	// channel is backpressure (Submit returns ErrQueueFull) rather than
	// unbounded goroutine/memory growth.
	submit chan Request

	// quit asks the run loop to stop; closed once by Shutdown.
	quit     chan struct{}
	quitOnce sync.Once

	// done is closed by the run loop when it has exited and drained all
	// sequences, so Shutdown can block until the goroutine is truly gone (no
	// goroutine leak).
	done chan struct{}

	// started guards against a double Start.
	started bool
	mu      sync.Mutex
}

// New constructs a Scheduler over engine. queueDepth bounds the submit queue
// (waiting requests not yet admitted to a slot); values < 1 default to a small
// queue. The scheduler is not running until Start is called.
func New(engine BatchEngine, queueDepth int) *Scheduler {
	if queueDepth < 1 {
		queueDepth = 1
	}
	s := &Scheduler{
		engine: engine,
		submit: make(chan Request, queueDepth),
		quit:   make(chan struct{}),
		done:   make(chan struct{}),
	}
	// Speculative decoding is a DEFAULT-on capability for DENSE models: if the engine
	// can verify a batched draft step, use it. The drafter is model-agnostic
	// (prompt-lookup), so it works for every dense arch (Qwen3, Gemma) with no extra
	// weights. draftK is the max per-slot draft length, kept under the verify-row
	// budget so a full draft plus its anchor fits. SPARSE/MoE engines decline via
	// SpecGater (expert reads scale with draft length — spec doesn't bring value there).
	specOK := true
	if g, ok := engine.(SpecGater); ok {
		specOK = g.SpecWorthwhile()
	}
	if se, ok := engine.(SpecBatchEngine); ok && specOK && os.Getenv("FUCINA_NO_BATCH_SPEC") == "" {
		s.spec = se
		s.draftK = 6
	}
	// Chunked prefill is a DEFAULT-on capability: if the engine can prefill a slot
	// in pieces, long prompts are prefilled interleaved with decode so they never
	// block the batch. No effect on engines that don't implement it.
	if ce, ok := engine.(ChunkPrefillEngine); ok {
		s.chunk = ce
		s.chunkSize = defaultPrefillChunk
		s.chunkMin = defaultPrefillChunkMin
		// Engine-declared chunking preference (batch.PrefillChunkHinter): sparse/MoE FP8
		// engines dequant per-layer expert slabs per prefill pass, so FEWER, WIDER passes
		// amortize that fixed cost (256-token chunks repeated it 8x on a 2k prompt).
		if h, ok := engine.(PrefillChunkHinter); ok {
			if size, min := h.PrefillChunkHint(); size > 0 {
				s.chunkSize, s.chunkMin = size, min
			}
		}
		// Overrides: some engines (e.g. the Qwen3.5 hybrid, whose prefill GEMM dequantizes
		// weights per tile) prefer FEWER, WIDER prefill passes — a large chunkMin routes a
		// long prompt through the one-shot prefill (one dequant) instead of many 256-token
		// chunks (one dequant each). Trades decode-interleaving for TTFT.
		if v := os.Getenv("FUCINA_PREFILL_CHUNK"); v != "" {
			if n, err := strconv.Atoi(v); err == nil && n > 0 {
				s.chunkSize = n
			}
		}
		if v := os.Getenv("FUCINA_PREFILL_CHUNK_MIN"); v != "" {
			if n, err := strconv.Atoi(v); err == nil && n > 0 {
				s.chunkMin = n
			}
		}
	}
	if ps, ok := engine.(PrefixCacheStatsEngine); ok {
		s.prefixStats = ps
	}
	if pc, ok := engine.(PrefixCommitEngine); ok {
		s.prefixCommit = pc
	}
	return s
}

// Start launches the owner goroutine. It is safe to call once; subsequent calls
// are no-ops.
func (s *Scheduler) Start() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.started {
		return
	}
	s.started = true
	go s.run()
}

// Submit enqueues a request. It returns ErrShutdown if the scheduler is
// stopping, or ErrQueueFull if the bounded queue is full (the caller should
// shed load). It never blocks waiting for a slot — admission happens inside the
// run loop as capacity frees up.
func (s *Scheduler) Submit(req Request) error {
	if req.MaxNew < 1 {
		req.MaxNew = 1
	}
	// Prefer a shutdown signal over enqueuing, so Submit after Shutdown is
	// deterministic even when the queue has room.
	select {
	case <-s.quit:
		return ErrShutdown
	default:
	}
	select {
	case s.submit <- req:
		return nil
	case <-s.quit:
		return ErrShutdown
	default:
		return ErrQueueFull
	}
}

// Shutdown stops the scheduler and blocks until the owner goroutine has exited
// and every in-flight and queued sequence has been evicted (each receiving a
// FinishShutdown Result). It is idempotent.
func (s *Scheduler) Shutdown() {
	s.quitOnce.Do(func() { close(s.quit) })
	<-s.done
}

// ─── Owner goroutine ───────────────────────────────────────────────

// run is the single owner goroutine. It is the ONLY caller of the engine, so
// the engine needs no internal locking for the step loop.
func (s *Scheduler) run() {
	// The CUDA context is bound to a single OS thread (the engine's creating
	// thread calls runtime.LockOSThread in main). run() is the SOLE engine caller
	// on the batch path — every AddSeq/StepBatch/RemoveSeq cgo call happens here —
	// so this goroutine MUST stay pinned to one OS thread for the lifetime of the
	// loop. Without this, the Go scheduler migrates the goroutine across OS threads
	// between engine calls, and CUDA rejects the foreign thread with "invalid
	// device context" (the batched-prefill / seq_add failures on Qwen3). Lock for
	// the whole loop; do NOT Unlock on exit (a thread that has ever held the CUDA
	// context is best left to die with the goroutine rather than returned to the
	// runtime's reusable pool).
	runtime.LockOSThread()

	// active maps engine slot id → sequence. The scheduler holds at most
	// Capacity() entries here at once.
	active := make(map[int]*seq)
	// prefill is the round-robin backlog of sequences whose (long) prompt is being
	// prefilled in chunks, interleaved with decode. A sequence holds an engine slot
	// the whole time it is here; when its prompt is fully prefilled it moves to
	// active. Empty unless the engine supports chunked prefill (s.chunk != nil).
	var prefill []*seq
	// waiting is the queue of accepted-but-not-yet-admitted requests, drained
	// from the submit channel and admitted as slots free up.
	var waiting []Request

	defer close(s.done)
	defer func() {
		// Drain everything on the way out so no client is left hanging and no
		// slot is leaked: evict every active and prefilling sequence and reply to
		// every queued request with FinishShutdown.
		for slot, sq := range active {
			if err := s.engine.RemoveSeq(slot); err != nil {
				log.Printf("batch: RemoveSeq(%d) on shutdown: %v", slot, err)
			}
			reply(sq.req, Result{Reason: FinishShutdown, Generated: sq.generated})
		}
		for _, sq := range prefill {
			if err := s.engine.RemoveSeq(sq.slot); err != nil {
				log.Printf("batch: RemoveSeq(%d) on shutdown: %v", sq.slot, err)
			}
			reply(sq.req, Result{Reason: FinishShutdown, Generated: sq.generated})
		}
		for _, req := range waiting {
			reply(req, Result{Reason: FinishShutdown})
		}
		// Drain any requests still in the channel buffer.
		for {
			select {
			case req := <-s.submit:
				reply(req, Result{Reason: FinishShutdown})
			default:
				return
			}
		}
	}()

	for {
		// 1. Pull newly submitted requests into the waiting backlog without
		//    blocking, so admission can consider them this pass. The backlog is
		//    bounded: overflow stays in the channel buffer, so Submit observes a
		//    full queue (ErrQueueFull) instead of the scheduler growing an
		//    unbounded slice.
		s.drainSubmit(&waiting)

		// 2. Admit waiting requests while slots are free. Capacity is queried
		//    each pass so a dynamic engine can grow/shrink. A short prompt is
		//    prefilled one-shot and joins active; a long prompt (chunked engine)
		//    is opened and enters the prefill backlog. admitted reports whether
		//    at least one sequence was slotted this pass.
		admitted := s.admit(active, &prefill, &waiting)

		// 3. Evict any active OR prefilling sequence whose context was cancelled
		//    BEFORE the step, so a cancelled client's slot is freed promptly (and
		//    never burns another forward pass / prefill chunk).
		s.evictCancelled(active)
		s.sweepCancelledPrefill(&prefill)

		// 4. Advance ONE chunk of one prefilling sequence (round-robin). Paired
		//    with the decode step below, this interleaves prefill and decode so a
		//    long prompt's prefill never blocks the active sequences. A sequence
		//    whose prompt finished prefilling is promoted to active here.
		s.advancePrefill(active, &prefill)

		// If there is nothing to do, block instead of spinning. The scheduler is
		// idle only when there is no active decode, no prefill in flight, and we
		// admitted nothing this pass:
		//   - nothing waiting: block until a new request or shutdown.
		//   - waiting but un-admittable this pass (engine at capacity / AddSeq
		//     stuck) AND we admitted nothing: block on a new request or a short
		//     timer, so a capacity that frees externally is retried without a
		//     100%-CPU spin.
		if len(active) == 0 && len(prefill) == 0 && !admitted {
			select {
			case <-s.quit:
				return
			case req := <-s.submit:
				waiting = append(waiting, req)
			case <-retryTimer(len(waiting) > 0):
			}
			continue
		}
		if len(active) == 0 {
			// No active decode this pass: either everything self-evicted (e.g.
			// one-token stop sequences) or work is still in the prefill backlog.
			// Loop to advance the next prefill chunk / admit more waiters without
			// parking (prefill in flight is real progress, not a spin).
			continue
		}

		// 5. Build the decode batch from the active slots and run ONE step that
		//    advances ALL of them together.
		if !s.step(active) {
			// Fatal engine error already handled inside step (all sequences
			// evicted); fall through to re-check shutdown / new work.
		}

		// 6. Publish prefix-cache counters lock-free for /metrics. Done once per pass
		//    (not just on admit) so decode-time changes — generated-block registration
		//    via PrefixCommit and LRU evictions — are reflected. The engine read is a
		//    cheap uncontended counter fetch on the goroutine that owns the engine.
		if s.prefixStats != nil {
			lk, hb, cb, ev := s.prefixStats.PrefixCacheStats()
			s.pcLookups.Store(lk)
			s.pcHitBlocks.Store(hb)
			s.pcCached.Store(cb)
			s.pcEvictions.Store(ev)
		}

		// 7. Honor a shutdown request between steps.
		select {
		case <-s.quit:
			return
		default:
		}
	}
}

// maxBacklog bounds the in-loop waiting slice so a saturated engine cannot make
// the scheduler grow memory without limit: overflow stays in the bounded submit
// channel, where Submit observes it as ErrQueueFull. It is sized off the channel
// capacity so the total accepted-but-unadmitted budget tracks New's queueDepth.
func (s *Scheduler) maxBacklog() int { return cap(s.submit) }

// PrefixCacheStats returns the last-published cross-request prefix-cache counters
// (lock-free). Zero when the engine has no prefix cache (e.g. Gemma). The reuse rate
// is hitBlocks/lookups; hitBlocks*256 tokens of prefill were skipped.
func (s *Scheduler) PrefixCacheStats() (lookups, hitBlocks, cachedBlocks, evictions int64) {
	if s.prefixStats == nil {
		return 0, 0, 0, 0
	}
	return s.pcLookups.Load(), s.pcHitBlocks.Load(), s.pcCached.Load(), s.pcEvictions.Load()
}

// drainSubmit moves currently-queued submissions into waiting without blocking,
// up to the backlog bound. Leaving the rest in the channel is what surfaces
// backpressure to Submit.
func (s *Scheduler) drainSubmit(waiting *[]Request) {
	limit := s.maxBacklog()
	for len(*waiting) < limit {
		select {
		case req := <-s.submit:
			*waiting = append(*waiting, req)
		default:
			return
		}
	}
}

// retryPoll is how long the loop waits before re-checking engine capacity when
// it holds waiters it cannot yet admit (engine momentarily at capacity). Short
// enough to be responsive, long enough to avoid a busy spin.
const retryPoll = 5 * time.Millisecond

// retryTimer returns a channel that fires after retryPoll when there are stuck
// waiters to retry, or nil (blocks forever) when there is nothing to retry — so
// a truly idle scheduler parks on the submit channel with no wakeups.
func retryTimer(haveWaiters bool) <-chan time.Time {
	if !haveWaiters {
		return nil
	}
	return time.After(retryPoll)
}

// admit prefills and slots waiting requests while engine capacity allows. A
// request whose context is already cancelled is dropped without consuming a
// slot. An AddSeq error fails just that request; others keep their place. It
// returns true if at least one sequence was slotted (used by the run loop to
// decide whether to retry or park).
func (s *Scheduler) admit(active map[int]*seq, prefill *[]*seq, waiting *[]Request) bool {
	admitted := false
	oneShotAdmits := 0 // blocking AddSeq admissions this pass (capped to interleave decode)
	w := *waiting
	// A prefilling sequence also holds a slot, so it counts against capacity.
	held := func() int { return len(active) + len(*prefill) }
	for len(w) > 0 && held() < s.engine.Capacity() {
		req := w[0]

		// Skip a request that gave up while queued — don't burn a prefill for a
		// dead socket.
		if req.Ctx != nil && req.Ctx.Err() != nil {
			reply(req, Result{Reason: FinishCancelled})
			w = w[1:]
			continue
		}

		// Chunked path: a prompt on a chunk-capable engine is opened now (adopting any
		// cached prefix) and prefilled in pieces interleaved with decode, so it never
		// blocks the batch. Opening is cheap (no prefill), so it is NOT subject to the
		// one-shot admit cap below. Very short prompts fall through to the one-shot path.
		if s.chunk != nil && len(req.Tokens) > s.chunkMin {
			slot, nShared, err := s.chunk.OpenSeq(req.Tokens, req.Params)
			if err != nil {
				log.Printf("batch: OpenSeq failed: %v", err)
				reply(req, Result{Reason: FinishError, Err: fmt.Errorf("prefill open: %w", err)})
				w = w[1:]
				continue
			}
			sq := s.newSeq(req, slot)
			sq.prefillPos = nShared // adopted prefix already in KV; chunk-prefill only the suffix
			*prefill = append(*prefill, sq)
			w = w[1:]
			admitted = true
			continue
		}

		// One-shot AddSeq runs a BLOCKING suffix-prefill to completion before any decode
		// step. Cap it to maxOneShotAdmitsPerPass so a long burst of admissions cannot
		// starve decode (head-of-line TTFT blowup): the rest wait for the next pass,
		// which runs a decode step in between.
		if oneShotAdmits >= maxOneShotAdmitsPerPass {
			break
		}

		slot, first, err := s.engine.AddSeq(req.Tokens, req.Params)
		if err != nil {
			log.Printf("batch: AddSeq failed: %v", err)
			reply(req, Result{Reason: FinishError, Err: fmt.Errorf("prefill: %w", err)})
			w = w[1:]
			continue
		}

		sq := s.newSeq(req, slot)
		w = w[1:]
		admitted = true // a slot was consumed (progress), even if it self-evicts below
		oneShotAdmits++

		// Deliver the first (prefill-sampled) token immediately and apply the
		// same stop/budget bookkeeping a decode step would, so a one-token
		// sequence (prompt that immediately samples a stop) is evicted here
		// rather than entering the step loop.
		if s.deliver(active, sq, first) {
			active[slot] = sq
		}
	}
	*waiting = w
	return admitted
}

// newSeq builds the scheduler's per-sequence state for an admitted request bound to
// slot, including the drafter corpus seed (prompt tokens) when speculating. Shared by
// both admission paths (one-shot AddSeq and chunked OpenSeq) so they stay in lockstep.
func (s *Scheduler) newSeq(req Request, slot int) *seq {
	sq := &seq{
		req:       req,
		slot:      slot,
		remaining: req.MaxNew,
		stops:     make(map[int32]struct{}, len(req.Stops)),
	}
	// Track full history for the drafter (spec) and/or decode-time prefix-cache
	// registration. The prompt's full blocks are already registered by AddSeq, so
	// start regBlocks past them — PrefixCommit only fires for generated blocks.
	if s.spec != nil || s.prefixCommit != nil {
		sq.hist = append(make([]int32, 0, len(req.Tokens)+req.MaxNew), req.Tokens...)
		sq.regBlocks = len(req.Tokens) / 256
	}
	for _, st := range req.Stops {
		sq.stops[st] = struct{}{}
	}
	return sq
}

// advancePrefill commits ONE prefill chunk for the sequence at the front of the
// prefill backlog and rotates it to the back (round-robin fairness across concurrent
// prefills). Run once per scheduler pass alongside one decode step, this is what
// interleaves prefill with decode. When a sequence's prompt is fully prefilled it is
// removed from the backlog, its first token delivered, and (if it survives) promoted
// to the active decode set. A prefill error or KV exhaustion evicts just that
// sequence. It is a no-op when the backlog is empty.
func (s *Scheduler) advancePrefill(active map[int]*seq, prefill *[]*seq) {
	p := *prefill
	if len(p) == 0 {
		return
	}
	sq := p[0]
	toks := sq.req.Tokens
	lo := sq.prefillPos
	hi := lo + s.chunkSize
	if hi > len(toks) {
		hi = len(toks)
	}
	last := hi == len(toks)

	first, err := s.chunk.PrefillChunk(sq.slot, toks[lo:hi], last)
	if err != nil {
		log.Printf("batch: PrefillChunk(slot %d) failed: %v", sq.slot, err)
		*prefill = p[1:]
		s.evict(active, sq, Result{Reason: FinishError, Generated: sq.generated, Err: err})
		return
	}
	sq.prefillPos = hi

	if !last {
		// More prompt to go: rotate to the back so other prefills make progress too.
		p = p[1:]
		p = append(p, sq)
		*prefill = p
		return
	}

	// Prompt fully prefilled: leave the backlog and behave exactly like the one-shot
	// admit path — deliver the first sampled token and join active if it survives.
	*prefill = p[1:]
	// Register the prompt's full blocks so concurrent/later requests reuse them (the
	// one-shot AddSeq path registers inside the engine; the chunked path does it here
	// from the committed prompt). Idempotent; adopted blocks are skipped.
	if s.prefixCommit != nil {
		s.prefixCommit.PrefixCommit(sq.slot, sq.req.Tokens)
	}
	if s.deliver(active, sq, first) {
		active[sq.slot] = sq
	}
}

// sweepCancelledPrefill evicts every backlog sequence whose context has been
// cancelled, so a client that gave up during a long prefill frees its slot promptly
// instead of waiting to reach the front of the round-robin.
func (s *Scheduler) sweepCancelledPrefill(prefill *[]*seq) {
	p := *prefill
	if len(p) == 0 {
		return
	}
	kept := p[:0]
	for _, sq := range p {
		if sq.req.Ctx != nil && sq.req.Ctx.Err() != nil {
			if err := s.engine.RemoveSeq(sq.slot); err != nil {
				log.Printf("batch: RemoveSeq(%d) on cancel: %v", sq.slot, err)
			}
			reply(sq.req, Result{Reason: FinishCancelled, Generated: sq.generated})
			continue
		}
		kept = append(kept, sq)
	}
	*prefill = kept
}

// deliver emits one sampled token to a sequence and advances its bookkeeping.
// It returns true if the sequence should remain active for the next step, or
// false if it was evicted here (stop token, budget exhausted, emit backpressure,
// or context cancellation). On false it has already freed the slot and replied.
func (s *Scheduler) deliver(active map[int]*seq, sq *seq, token int32) bool {
	// A negative token is the engine signalling this sequence can no longer grow
	// its KV (block pool exhausted / context limit). Stop it GRACEFULLY — evict
	// only this sequence, never the whole batch — and do not emit the sentinel.
	if token < 0 {
		s.evict(active, sq, Result{Reason: FinishLength, Generated: sq.generated})
		return false
	}

	// Context cancelled after admission but before we emit: evict now.
	if sq.req.Ctx != nil && sq.req.Ctx.Err() != nil {
		s.evict(active, sq, Result{Reason: FinishCancelled, Generated: sq.generated})
		return false
	}

	// Emit the token. A false return is the client asking to stop (backpressure
	// / disconnect): honor it by evicting, so one slow client never stalls the
	// shared step loop.
	if sq.req.Emit != nil && !sq.req.Emit(token) {
		s.evict(active, sq, Result{Reason: FinishCancelled, Generated: sq.generated})
		return false
	}
	sq.generated++
	sq.remaining--

	// Record the committed token in the history corpus (drafter and/or prefix cache).
	if sq.hist != nil {
		sq.hist = append(sq.hist, token)
		// Register a generated block as soon as it completes, so a concurrent/later
		// request whose prompt extends this sequence reuses the generated KV. One
		// cgo call per 256 generated tokens (idempotent on the engine side).
		if s.prefixCommit != nil && len(sq.hist)/256 > sq.regBlocks {
			s.prefixCommit.PrefixCommit(sq.slot, sq.hist)
			sq.regBlocks = len(sq.hist) / 256
		}
	}

	// Stop token: deliver it (done above), then evict.
	if sq.stopHit(token) {
		s.evict(active, sq, Result{Reason: FinishStop, Generated: sq.generated})
		return false
	}
	// Budget exhausted.
	if sq.remaining <= 0 {
		s.evict(active, sq, Result{Reason: FinishLength, Generated: sq.generated})
		return false
	}
	// Survives: this token is the input that advances its slot next step.
	sq.next = token
	return true
}

// evictCancelled removes every active sequence whose context has been cancelled,
// before a step is built, so a cancelled slot is never fed into another forward
// pass.
func (s *Scheduler) evictCancelled(active map[int]*seq) {
	for slot, sq := range active {
		if sq.req.Ctx != nil && sq.req.Ctx.Err() != nil {
			s.evict(active, sq, Result{Reason: FinishCancelled, Generated: sq.generated})
			_ = slot
		}
	}
}

// step advances every active slot by one shared forward pass. When the engine is a
// SpecBatchEngine it takes the speculative path (draft per slot + one batched verify,
// committing the accepted run); otherwise it runs a plain one-token-per-slot decode.
// It returns false if a fatal engine error evicted everything.
func (s *Scheduler) step(active map[int]*seq) bool {
	if len(active) == 0 {
		return true
	}
	if s.spec != nil {
		return s.stepSpec(active)
	}
	return s.stepPlain(active)
}

// stepSpec is the speculative analogue of step: per active slot it drafts tokens
// (prompt-lookup over that sequence's history) and verifies them all in ONE batched
// forward via the engine's StepBatchSpec, then scatters the committed RUN (accepted
// drafts + bonus token) back to each sequence. The accepted run is what makes spec pay
// off — one weight pass commits multiple tokens.
//
// How many of each slot's candidate drafts to actually verify is decided by the DSpark
// Hardware-Aware Prefix Scheduler (scheduleConfidence): each draft token carries a
// consensus confidence, the scheduler ranks all requests' draft positions by survival
// probability and admits them greedily while expected throughput Θ=τ·SPS(B) rises,
// stopping (non-anticipating) the instant it would drop. So speculation grows at low
// concurrency (free verify rows) and tapers toward 0 as concurrency rises (anchors crowd
// out drafts and the throughput bar climbs), the engine's Σ(1+drafts) ≤ MaxVerifyRows cap
// being the hard ceiling. It is LOSSLESS: the scheduler only sets draft *lengths*; the
// committed run is what a plain greedy step would have emitted (the engine's verify
// enforces it), so any length — including zero — yields identical tokens.
func (s *Scheduler) stepSpec(active map[int]*seq) bool {
	// Stable order: build parallel arrays so reqs[i] ↔ slots[i] ↔ out[i] by index.
	slots := make([]int, 0, len(active))
	for slot := range active {
		slots = append(slots, slot)
	}

	// Draft a full candidate run per slot with per-position confidence, then fold each
	// confidence sequence into cumulative survival a_{j}=∏_{i≤j} c_i for the scheduler.
	drafts := make([][]int32, len(slots))
	surv := make([][]float32, len(slots))
	for i, slot := range slots {
		sq := active[slot]
		d, c := promptLookupDraftConf(sq.hist, s.draftK, 2, s.draftK)
		drafts[i] = d
		if len(c) > 0 {
			a := make([]float32, len(c))
			p := float32(1)
			for j, cj := range c {
				p *= cj
				a[j] = p
			}
			surv[i] = a
		}
	}

	// Hardware-aware prefix scheduler: choose the per-slot verify length under the global
	// row budget and the decode throughput model (DSpark Alg.1, training-free half).
	admit := scheduleConfidence(surv, MaxVerifyRows)

	reqs := make([]SpecReq, len(slots))
	anyDraft := false
	for i, slot := range slots {
		sq := active[slot]
		k := admit[i]
		if k > len(drafts[i]) {
			k = len(drafts[i])
		}
		var d []int32
		if k > 0 {
			d = drafts[i][:k]
			anyDraft = true
		}
		reqs[i] = SpecReq{Slot: int32(slot), Anchor: sq.next, Drafts: d}
	}

	// No slot was admitted a draft (all novel prose, or concurrency saturated the row
	// budget): the speculative verify would be pure overhead vs a plain one-token step.
	// Fall back so we never pay to "spec" zero drafts.
	if !anyDraft {
		return s.stepPlain(active)
	}

	out, err := s.spec.StepBatchSpec(reqs)
	if err != nil {
		log.Printf("batch: StepBatchSpec failed (%d active): %v", len(slots), err)
		for _, slot := range slots {
			if sq := active[slot]; sq != nil {
				s.evict(active, sq, Result{Reason: FinishError, Generated: sq.generated, Err: err})
			}
		}
		return false
	}
	if len(out) != len(slots) {
		err := fmt.Errorf("StepBatchSpec returned %d runs for %d slots", len(out), len(slots))
		log.Printf("batch: %v", err)
		for _, slot := range slots {
			if sq := active[slot]; sq != nil {
				s.evict(active, sq, Result{Reason: FinishError, Generated: sq.generated, Err: err})
			}
		}
		return false
	}

	// Scatter each committed run to its slot; deliver walks the run token-by-token and
	// stops emitting the moment a token evicts the sequence (stop/budget/cancel or the
	// -1 KV-exhausted sentinel) — the rest of that run is then dropped.
	for i, slot := range slots {
		sq := active[slot]
		if sq == nil {
			continue // evicted earlier this pass (defensive)
		}
		for _, tok := range out[i] {
			if !s.deliver(active, sq, tok) {
				break
			}
		}
	}
	return true
}

// stepPlain runs one NON-speculative batched decode over all active slots, scattering the
// sampled run per slot back to each sequence (the original step body; stepSpec falls back
// to it when no slot has a draft).
func (s *Scheduler) stepPlain(active map[int]*seq) bool {
	// Build parallel (slot, input) arrays. Iteration order over the map is
	// irrelevant: out[i] corresponds to slots[i] by index.
	slots := make([]int32, 0, len(active))
	inputs := make([]int32, 0, len(active))
	for slot, sq := range active {
		slots = append(slots, int32(slot))
		inputs = append(inputs, sq.next)
	}

	out, err := s.engine.StepBatch(slots, inputs)
	if err != nil {
		// A batched step failed for the whole batch: there is no per-row error
		// signal, so fail every active sequence rather than silently dropping
		// tokens. The slots are freed individually.
		log.Printf("batch: StepBatch failed (%d active): %v", len(active), err)
		for _, slot := range slots {
			sq := active[int(slot)]
			if sq == nil {
				continue
			}
			s.evict(active, sq, Result{Reason: FinishError, Generated: sq.generated, Err: err})
		}
		return false
	}
	if len(out) != len(slots) {
		err := fmt.Errorf("StepBatch returned %d token-runs for %d slots", len(out), len(slots))
		log.Printf("batch: %v", err)
		for _, slot := range slots {
			sq := active[int(slot)]
			if sq == nil {
				continue
			}
			s.evict(active, sq, Result{Reason: FinishError, Generated: sq.generated, Err: err})
		}
		return false
	}

	// Scatter: deliver out[i] (a RUN of tokens) to the sequence in slots[i].
	// deliver handles stop/budget/cancel eviction per token, so we walk the run
	// in order and STOP emitting the moment a token in the run evicts the
	// sequence (deliver returns false) — the remaining drafted tokens of that
	// run are past the stop/budget boundary and must not be emitted. A finished
	// sequence frees its slot now, making room for a queued request on the next
	// admission pass.
	for i, slot := range slots {
		sq := active[int(slot)]
		if sq == nil {
			continue // already evicted this pass (defensive)
		}
		for _, tok := range out[i] {
			if !s.deliver(active, sq, tok) {
				break // evicted mid-run: drop the rest of this row's run
			}
		}
	}
	return true
}

// evict frees a sequence's engine slot, removes it from the active set, and
// delivers its terminal Result. It is safe to call at most once per sequence.
func (s *Scheduler) evict(active map[int]*seq, sq *seq, res Result) {
	if err := s.engine.RemoveSeq(sq.slot); err != nil {
		log.Printf("batch: RemoveSeq(%d): %v", sq.slot, err)
	}
	delete(active, sq.slot)
	reply(sq.req, res)
}

// reply delivers a terminal Result on the request's Done channel without
// blocking. Done is required to be buffered (cap >= 1); a nil Done is a
// fire-and-forget request. The non-blocking send is belt-and-suspenders: it
// guarantees the scheduler goroutine can never wedge on a caller that failed to
// drain its own channel.
func reply(req Request, res Result) {
	if req.Done == nil {
		return
	}
	select {
	case req.Done <- res:
	default:
		log.Printf("batch: dropped Result(%s) — Done channel full or unbuffered", res.Reason)
	}
}
