package batch

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/hikmaai-io/fucina/internal/metrics"
)

type policyChunkEngine struct{ *mockEngine }

func (e *policyChunkEngine) OpenSeq(_ []int32, p SeqParams) (int, int, error) {
	slot, _, err := e.AddSeq(nil, p)
	return slot, 0, err
}
func (e *policyChunkEngine) PrefillChunk(_ int, _ []int32, last bool) (int32, error) {
	if last {
		return 1, nil
	}
	return 0, nil
}

type recordingChunkEngine struct {
	*chunkMock
	muLens sync.Mutex
	lens   []int
}

func (e *recordingChunkEngine) PrefillChunk(slot int, chunk []int32, last bool) (int32, error) {
	e.muLens.Lock()
	e.lens = append(e.lens, len(chunk))
	e.muLens.Unlock()
	return e.chunkMock.PrefillChunk(slot, chunk, last)
}

type phaseRecorder struct {
	mu     sync.Mutex
	phases map[metrics.Phase]int
}

func newPhaseRecorder() *phaseRecorder                  { return &phaseRecorder{phases: make(map[metrics.Phase]int)} }
func (r *phaseRecorder) SetQueueDepth(int)              {}
func (r *phaseRecorder) ObserveQueueWait(time.Duration) {}
func (r *phaseRecorder) ObservePhase(p metrics.Phase, _ time.Duration) {
	r.mu.Lock()
	r.phases[p]++
	r.mu.Unlock()
}
func (r *phaseRecorder) ObserveBatchSize(int)  {}
func (r *phaseRecorder) AddPreemptions(uint64) {}
func (r *phaseRecorder) count(p metrics.Phase) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.phases[p]
}

func TestIterationBudgetAccountsDecodePrefillAndVerification(t *testing.T) {
	b := newIterationBudget(12)
	if !b.consume(4, false) { // decode anchors
		t.Fatal("decode reservation rejected")
	}
	if !b.consume(6, false) { // prefill tokens
		t.Fatal("prefill reservation rejected")
	}
	if !b.consume(2, false) { // verification rows
		t.Fatal("verification reservation rejected")
	}
	if b.remaining() != 0 || b.consume(1, false) {
		t.Fatalf("budget overrun: used=%d remaining=%d", b.used, b.remaining())
	}

	over := newIterationBudget(4)
	if !over.consume(9, true) || over.used != 9 {
		t.Fatalf("idle oversized progress grant = %d, want 9", over.used)
	}
	if over.consume(1, true) {
		t.Fatal("second oversized action was incorrectly admitted")
	}
}

func TestShortBatchScansPastIneligibleAndReservesHead(t *testing.T) {
	eng := &policyChunkEngine{mockEngine: newMockEngine(4)}
	sched := New(eng, 16)
	sched.chunkMin = 4
	active := make(map[int]*seq)
	long := Request{Tokens: iota1(8), MaxNew: 2, Ctx: context.Background(), Emit: (&collector{}).emit}
	waiting := []Request{
		long,
		{Tokens: []int32{11}, MaxNew: 2, Ctx: context.Background(), Emit: (&collector{}).emit},
		{Tokens: []int32{12}, MaxNew: 2, Ctx: context.Background(), Emit: (&collector{}).emit},
		{Tokens: []int32{13}, MaxNew: 2, Ctx: context.Background(), Emit: (&collector{}).emit},
	}
	budget := newIterationBudget(64)
	if !sched.admitBatched(active, &waiting, eng, 4, nil, budget) {
		t.Fatal("non-leading eligible shorts were not batched")
	}
	if len(active) != 3 || len(waiting) != 1 || len(waiting[0].Tokens) != len(long.Tokens) {
		t.Fatalf("after skip-scan: active=%d waiting=%v", len(active), waiting)
	}
	if eng.liveCount() != 3 {
		t.Fatalf("skip-scan consumed %d slots, want capacity-1", eng.liveCount())
	}

	// The reserved final slot admits the bypassed head through its chunk path before
	// any younger wave can take it.
	var prefill []*seq
	if !sched.admit(active, &prefill, &waiting, budget) {
		t.Fatal("bypassed queue head was not admitted on its reserved slot")
	}
	if len(waiting) != 0 || len(prefill) != 1 || eng.liveCount() != 4 {
		t.Fatalf("fairness reservation failed: waiting=%d prefill=%d live=%d", len(waiting), len(prefill), eng.liveCount())
	}
}

func TestTinyIterationBudgetPartiallyChunksLosslessly(t *testing.T) {
	eng := &recordingChunkEngine{chunkMock: newChunkMock(1)}
	sched := New(eng, 4)
	sched.iterationTokens = 5
	sched.chunkMin = 1
	sched.chunkSize = 256
	prompt := iota1(12)
	col := &collector{}
	done := make(chan Result, 1)
	if err := sched.Submit(Request{Tokens: prompt, MaxNew: 1, Ctx: context.Background(), Emit: col.emit, Done: done}); err != nil {
		t.Fatal(err)
	}
	sched.Start()
	defer sched.Shutdown()
	res := waitResult(t, done)
	if res.Reason != FinishLength || res.Generated != 1 {
		t.Fatalf("result = %+v", res)
	}
	if got := col.got(); len(got) != 1 || got[0] != sumTokens(prompt)+1 {
		t.Fatalf("lossless partial chunks emitted %v", got)
	}
	eng.muLens.Lock()
	defer eng.muLens.Unlock()
	if len(eng.lens) < 3 {
		t.Fatalf("prefill chunks = %v, want multiple budget-bounded chunks", eng.lens)
	}
	for _, n := range eng.lens {
		if n > sched.iterationTokens {
			t.Fatalf("chunk %d exceeds iteration budget %d", n, sched.iterationTokens)
		}
	}
	// Usage invariants for this cold request.
	cached, promptTokens, completion := res.ReusedTokens, len(prompt), res.Generated
	if cached < 0 || cached > promptTokens || promptTokens-cached != len(prompt) || promptTokens+completion != 13 {
		t.Fatalf("usage invariant: cached=%d prompt=%d completion=%d", cached, promptTokens, completion)
	}
}

func TestIterationBudgetTapersSpeculationToDecodeOnly(t *testing.T) {
	eng := &specEngine{mockEngine: newMockEngine(1), accept: 3}
	sched := New(eng, 4)
	sched.iterationTokens = 1 // exactly one active decode anchor, no verify rows
	col := &collector{}
	done := make(chan Result, 1)
	if err := sched.Submit(Request{Tokens: repeatingPrompt(), MaxNew: 6, Ctx: context.Background(), Emit: col.emit, Done: done}); err != nil {
		t.Fatal(err)
	}
	sched.Start()
	defer sched.Shutdown()
	if res := waitResult(t, done); res.Reason != FinishLength {
		t.Fatalf("result = %+v", res)
	}
	if eng.sawAnyDraft() {
		t.Fatal("speculation consumed verification rows after decode exhausted the iteration budget")
	}
}

func TestAdaptiveCoalescerIsBoundedAndLearnsLoneTraffic(t *testing.T) {
	var c burstCoalescer
	maxProbe, maxQuiet, maxHard := 3*time.Millisecond, 12*time.Millisecond, 150*time.Millisecond
	initial := c.plan(maxProbe, maxQuiet, maxHard, 32)
	if initial.probe > maxProbe || initial.quiet > maxQuiet || initial.hard > maxHard {
		t.Fatalf("initial plan exceeds bounds: %+v", initial)
	}
	for i := 0; i < 6; i++ {
		c.observe(1, initial.probe)
	}
	learned := c.plan(maxProbe, maxQuiet, maxHard, 32)
	if learned.probe >= initial.probe {
		t.Fatalf("lone-traffic probe did not adapt down: initial=%s learned=%s", initial.probe, learned.probe)
	}
}

func TestSchedulerEmitsAllSOL10aPhaseHistograms(t *testing.T) {
	rec := newPhaseRecorder()
	eng := newMockEngine(1)
	sched := New(eng, 4, rec)
	sched.Start()
	defer sched.Shutdown()
	// Let the owner goroutine enter its idle receive so this request exercises
	// adaptive coalescing rather than being drained before the idle boundary.
	time.Sleep(10 * time.Millisecond)
	col := &collector{}
	done := make(chan Result, 1)
	if err := sched.Submit(Request{Tokens: []int32{1, 2}, MaxNew: 2, Ctx: context.Background(), Emit: col.emit, Done: done}); err != nil {
		t.Fatal(err)
	}
	if res := waitResult(t, done); res.Reason != FinishLength {
		t.Fatalf("result = %+v", res)
	}
	for _, phase := range []metrics.Phase{
		metrics.PhaseQueue,
		metrics.PhaseCoalesce,
		metrics.PhaseAdmission,
		metrics.PhasePrefill,
		metrics.PhaseFirstDecode,
		metrics.PhaseDecode,
	} {
		if rec.count(phase) == 0 {
			t.Errorf("phase %q was not observed", phase)
		}
	}
}
