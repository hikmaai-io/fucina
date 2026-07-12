package batch

import (
	"context"
	"runtime"
	"sync"
	"testing"
	"time"
)

// ─── Deterministic mock engine ─────────────────────────────────────

// mockEngine is a deterministic, GPU-free BatchEngine for testing the
// scheduler. Every slot counts up from a per-slot base: AddSeq returns the
// first token and each StepBatch returns input+1 for every active row. A
// configurable stopAt makes a slot sample a sentinel stop token after a fixed
// number of tokens, so eviction paths are exercised deterministically.
//
// It also records a call log: one entry per StepBatch with the slots advanced,
// so a test can assert that a step advanced ALL active slots at once (not one
// at a time).
type mockEngine struct {
	mu sync.Mutex

	capacity int

	// nextSlot is the lowest free slot id. freed slots are pushed onto free.
	nextSlot int
	free     []int
	live     map[int]bool // currently allocated slots

	// stepLog records, per StepBatch call, the set of slot ids advanced.
	stepLog [][]int32

	// stopToken, when set (non-zero), is sampled for a slot once that slot has
	// produced stopAfter tokens, so the scheduler sees a stop and evicts.
	stopToken int32
	stopAfter int
	produced  map[int]int // tokens produced per slot

	// addErr / stepErr force error paths when set.
	addErr  error
	stepErr error

	// runLen, when > 1, makes StepBatch return a RUN of that many tokens per slot
	// per step (a per-sequence speculative-decode step), so the scheduler's
	// run-walking deliver loop is exercised. Default 0/1 keeps the one-token path.
	runLen int

	// addCalls / removeCalls count admissions and evictions for leak checks.
	addCalls    int
	removeCalls int

	// multiseqCalls counts batched-admission (SeqAddMultiseq) calls.
	multiseqCalls int
}

func newMockEngine(capacity int) *mockEngine {
	return &mockEngine{
		capacity: capacity,
		live:     make(map[int]bool),
		produced: make(map[int]int),
	}
}

func (m *mockEngine) Capacity() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.capacity
}

func (m *mockEngine) AddSeq(prompt []int32, _ SeqParams) (int, int32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.addErr != nil {
		return 0, 0, m.addErr
	}
	m.addCalls++
	var slot int
	if n := len(m.free); n > 0 {
		slot = m.free[n-1]
		m.free = m.free[:n-1]
	} else {
		slot = m.nextSlot
		m.nextSlot++
	}
	m.live[slot] = true
	m.produced[slot] = 0
	// First token: derive from the prompt so different sequences differ, but
	// keep it deterministic. Use a base of slot*1000 + 1.
	first := int32(slot*1000 + 1)
	return slot, m.tokenFor(slot, first), nil
}

// SeqAddMultiseq mirrors AddSeq for M prompts in one call (P1 batched admission path).
func (m *mockEngine) SeqAddMultiseq(prompts [][]int32, _ []SeqParams) ([]int, []int32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.addErr != nil {
		return nil, nil, m.addErr
	}
	m.multiseqCalls++
	M := len(prompts)
	slots := make([]int, M)
	firsts := make([]int32, M)
	for i := 0; i < M; i++ {
		m.addCalls++
		var slot int
		if n := len(m.free); n > 0 {
			slot = m.free[n-1]
			m.free = m.free[:n-1]
		} else {
			slot = m.nextSlot
			m.nextSlot++
		}
		m.live[slot] = true
		m.produced[slot] = 0
		slots[i] = slot
		firsts[i] = m.tokenFor(slot, int32(slot*1000+1))
	}
	return slots, firsts, nil
}

// tokenFor applies the stop-token policy to a freshly produced token for slot.
func (m *mockEngine) tokenFor(slot int, tok int32) int32 {
	m.produced[slot]++
	if m.stopToken != 0 && m.produced[slot] >= m.stopAfter {
		return m.stopToken
	}
	return tok
}

func (m *mockEngine) StepBatch(active []int32, inputs []int32) ([][]int32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.stepErr != nil {
		return nil, m.stepErr
	}
	// Record which slots were advanced this step (a copy, the caller reuses its
	// slice across steps).
	logged := make([]int32, len(active))
	copy(logged, active)
	m.stepLog = append(m.stepLog, logged)

	out := make([][]int32, len(active))
	for i, slot := range active {
		if !m.live[int(slot)] {
			// The scheduler should never feed a freed slot; treat as a test bug
			// signal by echoing input (the assertion lives in the test).
			out[i] = []int32{inputs[i]}
			continue
		}
		n := m.runLen
		if n < 1 {
			n = 1
		}
		// A run of n sequential tokens for this slot. Each call to tokenFor
		// advances the slot's produced counter and applies the stop-token policy,
		// so a stop landing MID-RUN truncates the emitted run (the scheduler must
		// not emit tokens past the stop — that is what this models).
		run := make([]int32, n)
		tok := inputs[i]
		for k := 0; k < n; k++ {
			tok = m.tokenFor(int(slot), tok+1)
			run[k] = tok
		}
		out[i] = run
	}
	return out, nil
}

func (m *mockEngine) RemoveSeq(slot int) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.removeCalls++
	delete(m.live, slot)
	delete(m.produced, slot)
	m.free = append(m.free, slot)
	return nil
}

// liveCount reports how many slots are currently allocated (leak check).
func (m *mockEngine) liveCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.live)
}

func (m *mockEngine) steps() [][]int32 {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([][]int32, len(m.stepLog))
	copy(out, m.stepLog)
	return out
}

func (m *mockEngine) counts() (add, remove int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.addCalls, m.removeCalls
}

// ─── Test helpers ──────────────────────────────────────────────────

// collector drains a sequence's emitted tokens into a slice via a buffered
// channel, modeling the non-blocking emit contract.
type collector struct {
	mu     sync.Mutex
	tokens []int32
}

func (c *collector) emit(t int32) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tokens = append(c.tokens, t)
	return true
}

func (c *collector) got() []int32 {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]int32, len(c.tokens))
	copy(out, c.tokens)
	return out
}

// waitResult waits for a Result with a generous timeout so a stuck scheduler
// fails the test instead of hanging the suite.
func waitResult(t *testing.T, done <-chan Result) Result {
	t.Helper()
	select {
	case r := <-done:
		return r
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for Result")
		return Result{}
	}
}

// ─── Tests ─────────────────────────────────────────────────────────

// (a) N concurrent requests all complete and interleave: each step advances
// ALL active slots, asserted via the mock's step log.
func TestConcurrentRequestsBatchTogether(t *testing.T) {
	const n = 4
	const maxNew = 5
	eng := newMockEngine(n) // enough slots for all at once
	sched := New(eng, 16)
	sched.Start()
	defer sched.Shutdown()

	cols := make([]*collector, n)
	dones := make([]chan Result, n)
	for i := 0; i < n; i++ {
		cols[i] = &collector{}
		dones[i] = make(chan Result, 1)
		if err := sched.Submit(Request{
			Tokens: []int32{int32(i)},
			MaxNew: maxNew,
			Ctx:    context.Background(),
			Emit:   cols[i].emit,
			Done:   dones[i],
		}); err != nil {
			t.Fatalf("submit %d: %v", i, err)
		}
	}

	for i := 0; i < n; i++ {
		res := waitResult(t, dones[i])
		if res.Reason != FinishLength {
			t.Errorf("seq %d: reason = %q want %q", i, res.Reason, FinishLength)
		}
		if res.Generated != maxNew {
			t.Errorf("seq %d: generated = %d want %d", i, res.Generated, maxNew)
		}
		if got := len(cols[i].got()); got != maxNew {
			t.Errorf("seq %d: emitted %d tokens want %d", i, got, maxNew)
		}
	}

	// Assert at least one step advanced all n slots together — proof of true
	// batching rather than one-sequence-at-a-time serialization.
	maxBatch := 0
	for _, st := range eng.steps() {
		if len(st) > maxBatch {
			maxBatch = len(st)
		}
	}
	if maxBatch < n {
		t.Errorf("max slots advanced in a single step = %d, want %d "+
			"(sequences were not batched together)", maxBatch, n)
	}
}

// A burst of short requests hitting an IDLE scheduler is admitted in ONE pass
// (uncapped idle one-shot admission), so the very first decode step already
// advances every row — the rows start in lockstep instead of a one-per-pass
// staggered ramp. Submitting before Start() makes the burst deterministic.
func TestIdleBurstAdmitsInOnePass(t *testing.T) {
	const n = 4
	const maxNew = 5
	eng := newMockEngine(n)
	sched := New(eng, 16)

	cols := make([]*collector, n)
	dones := make([]chan Result, n)
	for i := 0; i < n; i++ {
		cols[i] = &collector{}
		dones[i] = make(chan Result, 1)
		if err := sched.Submit(Request{
			Tokens: []int32{int32(i)},
			MaxNew: maxNew,
			Ctx:    context.Background(),
			Emit:   cols[i].emit,
			Done:   dones[i],
		}); err != nil {
			t.Fatalf("submit %d: %v", i, err)
		}
	}
	sched.Start()
	defer sched.Shutdown()

	for i := 0; i < n; i++ {
		res := waitResult(t, dones[i])
		if res.Reason != FinishLength {
			t.Errorf("seq %d: reason = %q want %q", i, res.Reason, FinishLength)
		}
	}
	steps := eng.steps()
	if len(steps) == 0 {
		t.Fatal("no StepBatch calls recorded")
	}
	if got := len(steps[0]); got != n {
		t.Errorf("first StepBatch advanced %d slots, want %d (burst was not admitted in one pass)", got, n)
	}
}

// TestBatchedAdmissionCancellation (P1, rev-2 correctness matrix — cancellation mid-admit):
// a burst where one request's ctx is already cancelled must NOT corrupt the batched admission
// of the others. admitBatched stops the batch at the cancelled request (leaving it to the serial
// reply-cancel), so the leading run admits via SeqAddMultiseq, the cancelled one finishes
// FinishCancelled, and the tail admits serially — no slot leak, no wrong reasons.
func TestBatchedAdmissionCancellation(t *testing.T) {
	const n = 4
	const cancelIdx = 2
	eng := newMockEngine(n)
	sched := New(eng, 16)

	cols := make([]*collector, n)
	dones := make([]chan Result, n)
	for i := 0; i < n; i++ {
		cols[i] = &collector{}
		dones[i] = make(chan Result, 1)
		ctx := context.Background()
		if i == cancelIdx {
			c, cancel := context.WithCancel(context.Background())
			cancel()
			ctx = c
		}
		if err := sched.Submit(Request{
			Tokens: []int32{int32(i)},
			MaxNew: 3,
			Ctx:    ctx,
			Emit:   cols[i].emit,
			Done:   dones[i],
		}); err != nil {
			t.Fatalf("submit %d: %v", i, err)
		}
	}
	sched.Start()
	defer sched.Shutdown()

	for i := 0; i < n; i++ {
		res := waitResult(t, dones[i])
		if i == cancelIdx {
			if res.Reason != FinishCancelled {
				t.Errorf("cancelled seq %d: reason = %q, want %q", i, res.Reason, FinishCancelled)
			}
		} else if res.Reason != FinishLength {
			t.Errorf("seq %d: reason = %q, want %q", i, res.Reason, FinishLength)
		}
	}
	if eng.multiseqCalls == 0 {
		t.Error("batched admission (SeqAddMultiseq) never fired for the burst")
	}
	if eng.addCalls != eng.removeCalls {
		t.Errorf("slot leak: %d admitted (add), %d freed (remove)", eng.addCalls, eng.removeCalls)
	}
}

// (b) A request hitting a stop token is evicted and frees its slot for a queued
// request.
func TestStopTokenEvictsAndFreesSlot(t *testing.T) {
	eng := newMockEngine(1) // ONE slot: the queued request can only run after eviction
	eng.stopToken = 99
	eng.stopAfter = 3 // each sequence stops after producing 3 tokens
	sched := New(eng, 8)
	sched.Start()
	defer sched.Shutdown()

	colA, colB := &collector{}, &collector{}
	doneA := make(chan Result, 1)
	doneB := make(chan Result, 1)

	// Submit two requests; only one slot exists, so B waits for A's stop.
	if err := sched.Submit(Request{
		Tokens: []int32{1}, MaxNew: 100, Ctx: context.Background(),
		Emit: colA.emit, Done: doneA, Stops: []int32{99},
	}); err != nil {
		t.Fatal(err)
	}
	if err := sched.Submit(Request{
		Tokens: []int32{2}, MaxNew: 100, Ctx: context.Background(),
		Emit: colB.emit, Done: doneB, Stops: []int32{99},
	}); err != nil {
		t.Fatal(err)
	}

	resA := waitResult(t, doneA)
	if resA.Reason != FinishStop {
		t.Errorf("A: reason = %q want %q", resA.Reason, FinishStop)
	}
	resB := waitResult(t, doneB)
	if resB.Reason != FinishStop {
		t.Errorf("B: reason = %q want %q", resB.Reason, FinishStop)
	}

	// Both stopped after exactly stopAfter tokens, and the last emitted token
	// is the stop token (it is delivered before eviction).
	for name, col := range map[string]*collector{"A": colA, "B": colB} {
		got := col.got()
		if len(got) != eng.stopAfter {
			t.Errorf("%s: emitted %d tokens want %d", name, len(got), eng.stopAfter)
		}
		if len(got) > 0 && got[len(got)-1] != 99 {
			t.Errorf("%s: last token = %d want stop token 99", name, got[len(got)-1])
		}
	}

	if live := eng.liveCount(); live != 0 {
		t.Errorf("live slots after both done = %d want 0", live)
	}
}

// (c) Context cancellation mid-generation evicts promptly.
func TestContextCancelEvicts(t *testing.T) {
	eng := newMockEngine(2)
	sched := New(eng, 8)
	sched.Start()
	defer sched.Shutdown()

	ctx, cancel := context.WithCancel(context.Background())
	col := &collector{}
	done := make(chan Result, 1)

	// A long-running sequence (no stop, huge budget) so it would run forever
	// without cancellation.
	if err := sched.Submit(Request{
		Tokens: []int32{1}, MaxNew: 1_000_000, Ctx: ctx,
		Emit: col.emit, Done: done,
	}); err != nil {
		t.Fatal(err)
	}

	// Let it produce a few tokens, then cancel.
	deadline := time.After(2 * time.Second)
	for {
		if len(col.got()) >= 1 {
			break
		}
		select {
		case <-deadline:
			t.Fatal("sequence never produced a token")
		default:
			runtime.Gosched()
		}
	}
	cancel()

	res := waitResult(t, done)
	if res.Reason != FinishCancelled {
		t.Errorf("reason = %q want %q", res.Reason, FinishCancelled)
	}
	if live := eng.liveCount(); live != 0 {
		t.Errorf("live slots after cancel = %d want 0", live)
	}
}

// (d) Capacity backpressure: more requests than slots → queued, admitted as
// slots free.
func TestCapacityBackpressureQueues(t *testing.T) {
	const slots = 2
	const reqs = 6
	const maxNew = 4
	eng := newMockEngine(slots)
	sched := New(eng, reqs) // queue can hold all requests
	sched.Start()
	defer sched.Shutdown()

	dones := make([]chan Result, reqs)
	for i := 0; i < reqs; i++ {
		dones[i] = make(chan Result, 1)
		col := &collector{}
		if err := sched.Submit(Request{
			Tokens: []int32{int32(i)}, MaxNew: maxNew, Ctx: context.Background(),
			Emit: col.emit, Done: dones[i],
		}); err != nil {
			t.Fatalf("submit %d: %v", i, err)
		}
	}

	for i := 0; i < reqs; i++ {
		res := waitResult(t, dones[i])
		if res.Reason != FinishLength || res.Generated != maxNew {
			t.Errorf("seq %d: %+v want length/%d", i, res, maxNew)
		}
	}

	// Never exceeded capacity: no single step advanced more than `slots` slots.
	for i, st := range eng.steps() {
		if len(st) > slots {
			t.Errorf("step %d advanced %d slots, exceeds capacity %d", i, len(st), slots)
		}
	}
	add, remove := eng.counts()
	if add != reqs || remove != reqs {
		t.Errorf("add=%d remove=%d want %d/%d (every request admitted and freed)",
			add, remove, reqs, reqs)
	}
	if live := eng.liveCount(); live != 0 {
		t.Errorf("live slots at end = %d want 0", live)
	}
}

// TestSubmitQueueFull asserts the bounded-queue backpressure error.
func TestSubmitQueueFull(t *testing.T) {
	// Engine with zero capacity never admits, so the queue fills and stays full.
	eng := newMockEngine(0)
	sched := New(eng, 2)
	sched.Start()
	defer sched.Shutdown()

	// Fill the queue (depth 2). Some may be pulled into the run loop's local
	// waiting slice, so submit until we observe ErrQueueFull.
	var sawFull bool
	for i := 0; i < 50; i++ {
		err := sched.Submit(Request{
			Tokens: []int32{int32(i)}, MaxNew: 1, Ctx: context.Background(),
			Done: make(chan Result, 1),
		})
		if err == ErrQueueFull {
			sawFull = true
			break
		}
		if err != nil {
			t.Fatalf("submit %d: unexpected err %v", i, err)
		}
	}
	if !sawFull {
		t.Error("never saw ErrQueueFull despite a saturated zero-capacity engine")
	}
}

// (e) No goroutine leak / clean Shutdown: Shutdown returns, in-flight and
// queued sequences get a FinishShutdown Result, and the owner goroutine exits.
func TestShutdownNoLeak(t *testing.T) {
	before := runtime.NumGoroutine()

	eng := newMockEngine(1) // 1 slot so some requests stay queued
	sched := New(eng, 8)
	sched.Start()

	// Submit several long-running requests; one runs, the rest queue.
	const reqs = 4
	dones := make([]chan Result, reqs)
	for i := 0; i < reqs; i++ {
		dones[i] = make(chan Result, 1)
		col := &collector{}
		if err := sched.Submit(Request{
			Tokens: []int32{int32(i)}, MaxNew: 1_000_000, Ctx: context.Background(),
			Emit: col.emit, Done: dones[i],
		}); err != nil {
			t.Fatalf("submit %d: %v", i, err)
		}
	}

	// Give the loop a moment to admit one and start stepping.
	time.Sleep(50 * time.Millisecond)

	sched.Shutdown() // blocks until the owner goroutine has exited and drained

	// Every request must have received a terminal Result (either it finished or
	// got FinishShutdown). None may be left hanging.
	for i := 0; i < reqs; i++ {
		res := waitResult(t, dones[i])
		switch res.Reason {
		case FinishShutdown, FinishLength, FinishStop, FinishCancelled:
			// acceptable terminal states
		default:
			t.Errorf("seq %d: unexpected reason %q", i, res.Reason)
		}
	}

	if live := eng.liveCount(); live != 0 {
		t.Errorf("live slots after shutdown = %d want 0", live)
	}

	// Submit after shutdown is rejected.
	if err := sched.Submit(Request{Tokens: []int32{1}, Ctx: context.Background()}); err != ErrShutdown {
		t.Errorf("submit after shutdown: err = %v want ErrShutdown", err)
	}

	// Idempotent shutdown.
	sched.Shutdown()

	// Goroutine count should return to baseline (allow a small slack for the
	// runtime/test harness). Poll briefly since teardown is asynchronous.
	leaked := true
	for i := 0; i < 50; i++ {
		if runtime.NumGoroutine() <= before+1 {
			leaked = false
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if leaked {
		t.Errorf("goroutine leak: before=%d after=%d", before, runtime.NumGoroutine())
	}
}

// TestEmitBackpressureEvicts asserts that an emit callback returning false (the
// slow-client backpressure signal) evicts the sequence so it cannot stall the
// shared step loop.
func TestEmitBackpressureEvicts(t *testing.T) {
	eng := newMockEngine(1)
	sched := New(eng, 4)
	sched.Start()
	defer sched.Shutdown()

	done := make(chan Result, 1)
	var emitted int
	var mu sync.Mutex
	// Reject after the 2nd token, simulating a full per-seq buffer.
	emit := func(int32) bool {
		mu.Lock()
		defer mu.Unlock()
		emitted++
		return emitted <= 2
	}
	if err := sched.Submit(Request{
		Tokens: []int32{1}, MaxNew: 1_000_000, Ctx: context.Background(),
		Emit: emit, Done: done,
	}); err != nil {
		t.Fatal(err)
	}

	res := waitResult(t, done)
	if res.Reason != FinishCancelled {
		t.Errorf("reason = %q want %q (backpressure should evict)", res.Reason, FinishCancelled)
	}
	if live := eng.liveCount(); live != 0 {
		t.Errorf("live slots = %d want 0", live)
	}
}

// TestAddSeqErrorRejects asserts a prefill error fails just that request.
func TestAddSeqErrorRejects(t *testing.T) {
	eng := newMockEngine(2)
	eng.addErr = context.DeadlineExceeded // any non-nil error
	sched := New(eng, 4)
	sched.Start()
	defer sched.Shutdown()

	done := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: []int32{1}, MaxNew: 5, Ctx: context.Background(), Done: done,
	}); err != nil {
		t.Fatal(err)
	}
	res := waitResult(t, done)
	if res.Reason != FinishError {
		t.Errorf("reason = %q want %q", res.Reason, FinishError)
	}
	if res.Err == nil {
		t.Error("expected a non-nil Err on FinishError")
	}
}

// (f) Speculative runs: StepBatch returns a RUN of tokens per slot per step. The
// scheduler must emit every token in a row's run, in order, and count each one
// against the sequence's budget — i.e. a row generates runLen tokens per step,
// so MaxNew is reached in MaxNew/runLen steps with every token delivered.
func TestSpeculativeRunsDeliverEveryToken(t *testing.T) {
	const runLen = 3
	const maxNew = 9 // exact multiple of runLen so the budget lands on a run boundary
	eng := newMockEngine(1)
	eng.runLen = runLen
	sched := New(eng, 4)
	sched.Start()
	defer sched.Shutdown()

	col := &collector{}
	done := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: []int32{1}, MaxNew: maxNew, Ctx: context.Background(),
		Emit: col.emit, Done: done,
	}); err != nil {
		t.Fatal(err)
	}
	res := waitResult(t, done)
	if res.Reason != FinishLength || res.Generated != maxNew {
		t.Fatalf("result = %+v want length/%d", res, maxNew)
	}
	if got := len(col.got()); got != maxNew {
		t.Errorf("emitted %d tokens want %d (every run token must be delivered)", got, maxNew)
	}
	// maxNew tokens at runLen per step => the first token comes from AddSeq, then
	// the remaining maxNew-1 are produced by ceil((maxNew-1)/runLen) steps.
	steps := len(eng.steps())
	wantSteps := (maxNew - 1 + runLen - 1) / runLen
	if steps != wantSteps {
		t.Errorf("ran %d steps want %d (runLen=%d amortizes the budget)", steps, wantSteps, runLen)
	}
}

// (g) Stop mid-run truncates: when a stop token lands in the MIDDLE of a row's
// speculative run, the scheduler must emit the stop token and then STOP — the
// drafted tokens after the stop are past the boundary and must not be delivered.
func TestSpeculativeRunStopsMidRun(t *testing.T) {
	const runLen = 4
	eng := newMockEngine(1)
	eng.runLen = runLen
	eng.stopToken = 99
	// The first token is produced by AddSeq (produced=1). With stopAfter=3 the
	// stop lands on the 3rd produced token, which is the 2nd token of the FIRST
	// StepBatch run (run produces tokens #2,#3,#4,#5) — i.e. strictly mid-run.
	eng.stopAfter = 3
	sched := New(eng, 4)
	sched.Start()
	defer sched.Shutdown()

	col := &collector{}
	done := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: []int32{1}, MaxNew: 100, Ctx: context.Background(),
		Emit: col.emit, Done: done, Stops: []int32{99},
	}); err != nil {
		t.Fatal(err)
	}
	res := waitResult(t, done)
	if res.Reason != FinishStop {
		t.Fatalf("reason = %q want %q", res.Reason, FinishStop)
	}
	got := col.got()
	// Emitted: AddSeq first token, then run token #2, then the stop (#3). The run
	// also drafted #4 and #5 AFTER the stop; those must be dropped.
	if len(got) != 3 {
		t.Fatalf("emitted %v (%d tokens) want 3 — tokens past the stop must be dropped", got, len(got))
	}
	if got[len(got)-1] != 99 {
		t.Errorf("last emitted = %d want stop token 99", got[len(got)-1])
	}
	if res.Generated != 3 {
		t.Errorf("Generated = %d want 3", res.Generated)
	}
	if live := eng.liveCount(); live != 0 {
		t.Errorf("live slots after stop = %d want 0 (slot freed mid-run)", live)
	}
}
