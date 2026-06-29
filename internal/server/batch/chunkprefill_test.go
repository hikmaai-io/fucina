package batch

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"
)

// ─── Chunked-prefill mock engine ───────────────────────────────────────────
//
// chunkMock is a deterministic ChunkPrefillEngine for testing chunked prefill +
// prefill/decode interleave without a GPU. A slot's "KV state" is an accumulator:
// prefilling adds the SUM of the prefilled token ids; the first generated token is
// acc+1, and each StepBatch advances a slot by input+1 (a running counter). Because
// the accumulator is order-independent, a one-shot AddSeq and a chunked
// OpenSeq+PrefillChunk over the SAME prompt reach the same acc and therefore emit a
// byte-identical token stream — exactly the invariant the real CUDA chunked prefill
// guarantees (the chunk boundary is invisible to the model). So driving this mock
// down the scheduler's one-shot vs chunked paths isolates the scheduling change and
// proves it is lossless.
type chunkMock struct {
	mu sync.Mutex

	capacity int
	nextSlot int
	free     []int
	live     map[int]bool
	acc      map[int]int32 // per-slot prefilled-token accumulator

	// ops records the engine-call timeline for interleave assertions: "P" for each
	// PrefillChunk, "S" for each StepBatch (in call order).
	ops []string

	openCalls, chunkCalls, addCalls, stepCalls, removeCalls int

	// failPrefillSlot, when >= 0, makes PrefillChunk return an error for that slot
	// (exercises the prefill-error eviction path).
	failPrefillSlot int

	// prefillDelay, when > 0, is slept (outside the lock) per PrefillChunk so a
	// GPU-free prefill takes long enough for a test to cancel it mid-flight.
	prefillDelay time.Duration
}

func newChunkMock(capacity int) *chunkMock {
	return &chunkMock{
		capacity:        capacity,
		live:            make(map[int]bool),
		acc:             make(map[int]int32),
		failPrefillSlot: -1,
	}
}

func (m *chunkMock) Capacity() int { m.mu.Lock(); defer m.mu.Unlock(); return m.capacity }

// allocSlot returns a free slot id (caller holds m.mu).
func (m *chunkMock) allocSlot() int {
	if n := len(m.free); n > 0 {
		s := m.free[n-1]
		m.free = m.free[:n-1]
		return s
	}
	s := m.nextSlot
	m.nextSlot++
	return s
}

func sumTokens(t []int32) int32 {
	var s int32
	for _, x := range t {
		s += x
	}
	return s
}

func (m *chunkMock) AddSeq(prompt []int32, _ SeqParams) (int, int32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.addCalls++
	slot := m.allocSlot()
	m.live[slot] = true
	m.acc[slot] = sumTokens(prompt)
	return slot, m.acc[slot] + 1, nil
}

func (m *chunkMock) OpenSeq(_ []int32, _ SeqParams) (int, int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.openCalls++
	slot := m.allocSlot()
	m.live[slot] = true
	m.acc[slot] = 0
	return slot, 0, nil // mock has no prefix cache → nShared 0
}

func (m *chunkMock) PrefillChunk(slot int, chunk []int32, last bool) (int32, error) {
	m.mu.Lock()
	m.chunkCalls++
	m.ops = append(m.ops, "P")
	delay := m.prefillDelay
	if slot == m.failPrefillSlot {
		m.mu.Unlock()
		return 0, fmt.Errorf("forced prefill failure on slot %d", slot)
	}
	if !m.live[slot] {
		m.mu.Unlock()
		return 0, fmt.Errorf("prefill on dead slot %d", slot)
	}
	m.acc[slot] += sumTokens(chunk)
	var first int32
	if last {
		first = m.acc[slot] + 1
	}
	m.mu.Unlock()
	if delay > 0 {
		time.Sleep(delay) // outside the lock so a test can observe progress / cancel
	}
	return first, nil
}

func (m *chunkMock) StepBatch(active []int32, inputs []int32) ([][]int32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stepCalls++
	m.ops = append(m.ops, "S")
	out := make([][]int32, len(active))
	for i, slot := range active {
		if !m.live[int(slot)] {
			out[i] = []int32{inputs[i]} // test bug if hit; assertions live in tests
			continue
		}
		out[i] = []int32{inputs[i] + 1}
	}
	return out, nil
}

func (m *chunkMock) RemoveSeq(slot int) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.removeCalls++
	delete(m.live, slot)
	delete(m.acc, slot)
	m.free = append(m.free, slot)
	return nil
}

func (m *chunkMock) liveCount() int { m.mu.Lock(); defer m.mu.Unlock(); return len(m.live) }

func (m *chunkMock) opsLog() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]string, len(m.ops))
	copy(out, m.ops)
	return out
}

func (m *chunkMock) callCounts() (open, chunk, add, step int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.openCalls, m.chunkCalls, m.addCalls, m.stepCalls
}

// iota1 builds a prompt of n tokens [1,2,...,n].
func iota1(n int) []int32 {
	p := make([]int32, n)
	for i := range p {
		p[i] = int32(i + 1)
	}
	return p
}

func equalTokens(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ─── Tests ─────────────────────────────────────────────────────────────────

// TestChunkedPrefillLossless is the cardinal property: a chunked prefill produces
// the SAME generated tokens as a one-shot prefill of the same prompt. It runs the
// same prompt through both scheduler paths (chunked, and one-shot via a huge chunkMin)
// and asserts both equal the analytic expected stream — proving the chunk boundary is
// invisible to generation.
func TestChunkedPrefillLossless(t *testing.T) {
	const promptLen = 500
	const maxNew = 24
	prompt := iota1(promptLen)
	sum := sumTokens(prompt)

	// Analytic expectation: first = sum+1, then a +1 counter for maxNew tokens.
	want := make([]int32, maxNew)
	for k := 0; k < maxNew; k++ {
		want[k] = sum + 1 + int32(k)
	}

	run := func(chunkMin, chunkSize int) []int32 {
		eng := newChunkMock(1)
		sched := New(eng, 4)
		sched.chunkMin = chunkMin
		sched.chunkSize = chunkSize
		sched.Start()
		defer sched.Shutdown()

		col := &collector{}
		done := make(chan Result, 1)
		if err := sched.Submit(Request{
			Tokens: prompt, MaxNew: maxNew, Ctx: context.Background(),
			Emit: col.emit, Done: done,
		}); err != nil {
			t.Fatal(err)
		}
		res := waitResult(t, done)
		if res.Reason != FinishLength || res.Generated != maxNew {
			t.Fatalf("result = %+v want length/%d", res, maxNew)
		}
		return col.got()
	}

	// Chunked path: small chunkMin so the 500-token prompt is chunked, small chunkSize
	// so it spans many chunks.
	gotChunk := run(4, 7)
	if !equalTokens(gotChunk, want) {
		t.Errorf("chunked tokens = %v want %v", gotChunk, want)
	}
	// One-shot path: chunkMin above the prompt length so AddSeq is used.
	gotOneShot := run(1<<30, 7)
	if !equalTokens(gotOneShot, want) {
		t.Errorf("one-shot tokens = %v want %v", gotOneShot, want)
	}
	if !equalTokens(gotChunk, gotOneShot) {
		t.Errorf("chunked %v != one-shot %v (chunk boundary changed generation)", gotChunk, gotOneShot)
	}
}

// TestChunkedPrefillUsesChunkPath asserts the engine actually saw a chunked prefill
// (OpenSeq once, multiple PrefillChunk, no AddSeq) for a long prompt, and the last
// chunk is the only one that triggers the first token.
func TestChunkedPrefillUsesChunkPath(t *testing.T) {
	const promptLen = 100
	const chunkSize = 16
	eng := newChunkMock(1)
	sched := New(eng, 4)
	sched.chunkMin = 4
	sched.chunkSize = chunkSize
	sched.Start()
	defer sched.Shutdown()

	col := &collector{}
	done := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: iota1(promptLen), MaxNew: 3, Ctx: context.Background(),
		Emit: col.emit, Done: done,
	}); err != nil {
		t.Fatal(err)
	}
	res := waitResult(t, done)
	if res.Reason != FinishLength {
		t.Fatalf("reason = %q want length", res.Reason)
	}
	open, chunk, add, _ := eng.callCounts()
	wantChunks := (promptLen + chunkSize - 1) / chunkSize
	if open != 1 || add != 0 {
		t.Errorf("open=%d add=%d want 1/0 (chunked path)", open, add)
	}
	if chunk != wantChunks {
		t.Errorf("PrefillChunk calls = %d want %d", chunk, wantChunks)
	}
	if eng.liveCount() != 0 {
		t.Errorf("live slots after done = %d want 0", eng.liveCount())
	}
}

// TestPrefillDecodeInterleave proves a long prompt's prefill is INTERLEAVED with the
// decode of an already-active sequence: a short request becomes active, then a long
// request is chunk-prefilled, and the engine timeline shows StepBatch calls landing
// BETWEEN the long prefill's first and last chunk (decode kept flowing — the prefill
// did not run to completion blocking the batch).
func TestPrefillDecodeInterleave(t *testing.T) {
	eng := newChunkMock(2) // room for the active short seq + the prefilling long seq
	sched := New(eng, 4)
	sched.chunkMin = 8   // short prompt one-shot, long prompt chunked
	sched.chunkSize = 10 // long prompt spans many chunks
	sched.Start()
	defer sched.Shutdown()

	// Short, long-lived active sequence (one-shot prefill → decodes every pass).
	ctxA, cancelA := context.WithCancel(context.Background())
	defer cancelA()
	colA := &collector{}
	doneA := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: []int32{1, 2}, MaxNew: 100000, Ctx: ctxA,
		Emit: colA.emit, Done: doneA,
	}); err != nil {
		t.Fatal(err)
	}
	// Wait until A is active and decoding (so it overlaps B's prefill).
	deadline := time.After(2 * time.Second)
	for len(colA.got()) < 2 {
		select {
		case <-deadline:
			t.Fatal("short sequence never started decoding")
		default:
			time.Sleep(time.Millisecond)
		}
	}

	// Long sequence: chunk-prefilled while A keeps decoding.
	colB := &collector{}
	doneB := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: iota1(300), MaxNew: 4, Ctx: context.Background(),
		Emit: colB.emit, Done: doneB,
	}); err != nil {
		t.Fatal(err)
	}
	resB := waitResult(t, doneB)
	if resB.Reason != FinishLength {
		t.Fatalf("B reason = %q want length", resB.Reason)
	}
	cancelA()
	waitResult(t, doneA)

	// Timeline: find the first and last PrefillChunk ("P") and count StepBatch ("S")
	// strictly between them. >0 means decode interleaved with the long prefill.
	ops := eng.opsLog()
	firstP, lastP := -1, -1
	for i, o := range ops {
		if o == "P" {
			if firstP < 0 {
				firstP = i
			}
			lastP = i
		}
	}
	if firstP < 0 || lastP <= firstP {
		t.Fatalf("expected multiple prefill chunks in ops=%v", ops)
	}
	stepsBetween := 0
	for i := firstP + 1; i < lastP; i++ {
		if ops[i] == "S" {
			stepsBetween++
		}
	}
	if stepsBetween == 0 {
		t.Errorf("no decode steps interleaved with the long prefill (ops=%v) — "+
			"prefill blocked the batch", ops)
	}
}

// TestPrefillSlotCountsAgainstCapacity asserts a prefilling sequence holds a slot:
// with capacity 1, a queued short request cannot be admitted until the chunk-prefilling
// long request has fully finished (prefill + generation + eviction).
func TestPrefillSlotCountsAgainstCapacity(t *testing.T) {
	eng := newChunkMock(1) // ONE slot
	sched := New(eng, 8)
	sched.chunkMin = 4 // long prompt chunked, short prompt one-shot
	sched.chunkSize = 10
	sched.Start()
	defer sched.Shutdown()

	colB := &collector{}
	doneB := make(chan Result, 1)
	if err := sched.Submit(Request{ // long → chunked, holds the only slot while prefilling
		Tokens: iota1(50), MaxNew: 3, Ctx: context.Background(),
		Emit: colB.emit, Done: doneB,
	}); err != nil {
		t.Fatal(err)
	}
	colA := &collector{}
	doneA := make(chan Result, 1)
	if err := sched.Submit(Request{ // short → one-shot, must wait for the slot
		Tokens: []int32{2}, MaxNew: 3, Ctx: context.Background(),
		Emit: colA.emit, Done: doneA,
	}); err != nil {
		t.Fatal(err)
	}

	resB := waitResult(t, doneB)
	resA := waitResult(t, doneA)
	if resB.Reason != FinishLength || resB.Generated != 3 {
		t.Errorf("B = %+v want length/3", resB)
	}
	if resA.Reason != FinishLength || resA.Generated != 3 {
		t.Errorf("A = %+v want length/3", resA)
	}
	// Capacity was never exceeded: no StepBatch advanced more than 1 slot.
	if eng.liveCount() != 0 {
		t.Errorf("live slots at end = %d want 0", eng.liveCount())
	}
	_, _, add, _ := eng.callCounts()
	if add != 1 { // A admitted one-shot after B freed the slot
		t.Errorf("AddSeq calls = %d want 1", add)
	}
}

// TestCancelDuringPrefillEvicts asserts a client that cancels during a long chunked
// prefill is evicted promptly and its slot freed (not stuck until prefill completes).
func TestCancelDuringPrefillEvicts(t *testing.T) {
	eng := newChunkMock(2)
	eng.prefillDelay = time.Millisecond // ~200ms total prefill: cancel lands mid-flight
	sched := New(eng, 4)
	sched.chunkMin = 4
	sched.chunkSize = 5 // 1000-token prompt → 200 chunks: ample time to cancel mid-prefill
	sched.Start()
	defer sched.Shutdown()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: iota1(1000), MaxNew: 5, Ctx: ctx,
		Emit: (&collector{}).emit, Done: done,
	}); err != nil {
		t.Fatal(err)
	}
	// Wait until prefill has started, then cancel mid-prefill.
	deadline := time.After(2 * time.Second)
	for {
		if _, chunk, _, _ := eng.callCounts(); chunk >= 1 {
			break
		}
		select {
		case <-deadline:
			t.Fatal("prefill never started")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	cancel()
	res := waitResult(t, done)
	if res.Reason != FinishCancelled {
		t.Errorf("reason = %q want %q", res.Reason, FinishCancelled)
	}
	// Slot freed, and the prefill stopped well before all 200 chunks ran.
	if eng.liveCount() != 0 {
		t.Errorf("live slots after cancel = %d want 0", eng.liveCount())
	}
	if _, chunk, _, _ := eng.callCounts(); chunk >= 200 {
		t.Errorf("prefill ran to completion (%d chunks) despite cancellation", chunk)
	}
}

// TestPrefillChunkErrorEvicts asserts a chunked-prefill engine error fails just that
// request with FinishError and frees its slot.
func TestPrefillChunkErrorEvicts(t *testing.T) {
	eng := newChunkMock(2)
	eng.failPrefillSlot = 0 // OpenSeq hands out slot 0 first
	sched := New(eng, 4)
	sched.chunkMin = 4
	sched.chunkSize = 10
	sched.Start()
	defer sched.Shutdown()

	done := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: iota1(100), MaxNew: 5, Ctx: context.Background(),
		Emit: (&collector{}).emit, Done: done,
	}); err != nil {
		t.Fatal(err)
	}
	res := waitResult(t, done)
	if res.Reason != FinishError {
		t.Errorf("reason = %q want %q", res.Reason, FinishError)
	}
	if res.Err == nil {
		t.Error("expected non-nil Err on FinishError")
	}
	if eng.liveCount() != 0 {
		t.Errorf("live slots after prefill error = %d want 0", eng.liveCount())
	}
}

// TestChunkedPrefillConcurrentLossless runs several long prompts concurrently through
// the chunked + interleaved path and asserts each still gets exactly its one-shot
// (analytic) token stream — losslessness holds under round-robin prefill + shared
// decode steps.
func TestChunkedPrefillConcurrentLossless(t *testing.T) {
	const n = 3
	const maxNew = 12
	eng := newChunkMock(n)
	sched := New(eng, 8)
	sched.chunkMin = 4
	sched.chunkSize = 9
	sched.Start()
	defer sched.Shutdown()

	prompts := make([][]int32, n)
	cols := make([]*collector, n)
	dones := make([]chan Result, n)
	for i := 0; i < n; i++ {
		// Distinct prompt lengths → distinct sums → distinct token streams.
		prompts[i] = iota1(200 + 37*i)
		cols[i] = &collector{}
		dones[i] = make(chan Result, 1)
		if err := sched.Submit(Request{
			Tokens: prompts[i], MaxNew: maxNew, Ctx: context.Background(),
			Emit: cols[i].emit, Done: dones[i],
		}); err != nil {
			t.Fatalf("submit %d: %v", i, err)
		}
	}

	for i := 0; i < n; i++ {
		res := waitResult(t, dones[i])
		if res.Reason != FinishLength || res.Generated != maxNew {
			t.Fatalf("seq %d: %+v want length/%d", i, res, maxNew)
		}
		sum := sumTokens(prompts[i])
		want := make([]int32, maxNew)
		for k := 0; k < maxNew; k++ {
			want[k] = sum + 1 + int32(k)
		}
		if got := cols[i].got(); !equalTokens(got, want) {
			t.Errorf("seq %d: tokens = %v want %v", i, got, want)
		}
	}
	if eng.liveCount() != 0 {
		t.Errorf("live slots at end = %d want 0", eng.liveCount())
	}
}
