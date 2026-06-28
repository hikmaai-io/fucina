package batch

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
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
	// hist is the sequence's full token history (prompt + every committed token),
	// the corpus the prompt-lookup drafter scans for repeated n-grams. Populated
	// only on the speculative path (nil otherwise, so the plain path pays nothing).
	hist []int32
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
	// Speculative decoding is a DEFAULT-on capability: if the engine can verify a
	// batched draft step, use it. The drafter is model-agnostic (prompt-lookup), so
	// it works for every arch (Qwen3, Gemma) with no extra weights. draftK is the
	// max per-slot draft length, kept under the verify-row budget so a full draft plus
	// its anchor fits.
	if se, ok := engine.(SpecBatchEngine); ok {
		s.spec = se
		s.draftK = 6
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
	// active maps engine slot id → sequence. The scheduler holds at most
	// Capacity() entries here at once.
	active := make(map[int]*seq)
	// waiting is the queue of accepted-but-not-yet-admitted requests, drained
	// from the submit channel and admitted as slots free up.
	var waiting []Request

	defer close(s.done)
	defer func() {
		// Drain everything on the way out so no client is left hanging and no
		// slot is leaked: evict every active sequence and reply to every queued
		// request with FinishShutdown.
		for slot, sq := range active {
			if err := s.engine.RemoveSeq(slot); err != nil {
				log.Printf("batch: RemoveSeq(%d) on shutdown: %v", slot, err)
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
		//    each pass so a dynamic engine can grow/shrink. admitted reports
		//    whether at least one sequence was slotted this pass.
		admitted := s.admit(active, &waiting)

		// 3. Evict any active sequence whose context was cancelled BEFORE the
		//    step, so a cancelled client's slot is freed promptly (and never
		//    burns another forward pass).
		s.evictCancelled(active)

		// If there is nothing to step, block instead of spinning. Two idle
		// shapes:
		//   - nothing waiting: block until a new request or shutdown.
		//   - waiting but un-admittable this pass (engine at capacity / AddSeq
		//     stuck) AND we admitted nothing: block on a new request or a short
		//     timer, so a capacity that frees externally is retried without a
		//     100%-CPU spin.
		if len(active) == 0 && !admitted {
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
			// admitted>0 a moment ago but everything was evicted in the same
			// pass (e.g. one-token stop sequences); loop to admit more waiters.
			continue
		}

		// 4. Build the decode batch from the active slots and run ONE step that
		//    advances ALL of them together.
		if !s.step(active) {
			// Fatal engine error already handled inside step (all sequences
			// evicted); fall through to re-check shutdown / new work.
		}

		// 5. Honor a shutdown request between steps.
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
func (s *Scheduler) admit(active map[int]*seq, waiting *[]Request) bool {
	admitted := false
	w := *waiting
	for len(w) > 0 && len(active) < s.engine.Capacity() {
		req := w[0]

		// Skip a request that gave up while queued — don't burn a prefill for a
		// dead socket.
		if req.Ctx != nil && req.Ctx.Err() != nil {
			reply(req, Result{Reason: FinishCancelled})
			w = w[1:]
			continue
		}

		slot, first, err := s.engine.AddSeq(req.Tokens, req.Params)
		if err != nil {
			log.Printf("batch: AddSeq failed: %v", err)
			reply(req, Result{Reason: FinishError, Err: fmt.Errorf("prefill: %w", err)})
			w = w[1:]
			continue
		}

		sq := &seq{
			req:       req,
			slot:      slot,
			remaining: req.MaxNew,
			stops:     make(map[int32]struct{}, len(req.Stops)),
		}
		// Seed the drafter corpus with the prompt tokens (only when speculating).
		if s.spec != nil {
			sq.hist = append(make([]int32, 0, len(req.Tokens)+req.MaxNew), req.Tokens...)
		}
		for _, st := range req.Stops {
			sq.stops[st] = struct{}{}
		}
		w = w[1:]
		admitted = true // a slot was consumed (progress), even if it self-evicts below

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

	// Record the committed token in the drafter corpus (spec path only).
	if s.spec != nil {
		sq.hist = append(sq.hist, token)
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
