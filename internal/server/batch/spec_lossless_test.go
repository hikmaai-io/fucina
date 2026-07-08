package batch

import (
	"context"
	"reflect"
	"sync"
	"testing"
)

// ─── A FAITHFUL, lossless engine over a fixed token cycle ───────────
//
// cycleEngine models a deterministic target whose next token is a pure function of the
// current token: succCycle (10→11→12→10…). Plain decode therefore emits the cycle. Its
// speculative sibling, cycleSpecEngine, verifies drafts against that SAME successor, so the
// committed run is always a prefix of the true continuation — i.e. byte-identical to a plain
// greedy step regardless of how many drafts were accepted. This is the testable face of the
// C verify's lossless guarantee, with prompt-lookup actually accepting multi-token runs
// (the cycle repeats, so the drafter proposes the exact continuation).

func succCycle(t int32) int32 {
	switch t {
	case 10:
		return 11
	case 11:
		return 12
	case 12:
		return 10
	default:
		return 10 // any other token enters the cycle
	}
}

type cycleEngine struct {
	mu       sync.Mutex
	capacity int
	nextSlot int
	free     []int
	live     map[int]bool
}

func newCycleEngine(capacity int) *cycleEngine {
	return &cycleEngine{capacity: capacity, live: make(map[int]bool)}
}

func (m *cycleEngine) Capacity() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.capacity
}

func (m *cycleEngine) AddSeq(prompt []int32, _ SeqParams) (int, int32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var slot int
	if n := len(m.free); n > 0 {
		slot = m.free[n-1]
		m.free = m.free[:n-1]
	} else {
		slot = m.nextSlot
		m.nextSlot++
	}
	m.live[slot] = true
	return slot, 10, nil // every sequence's first token enters the cycle at 10
}

func (m *cycleEngine) StepBatch(active []int32, inputs []int32) ([][]int32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([][]int32, len(active))
	for i := range active {
		out[i] = []int32{succCycle(inputs[i])}
	}
	return out, nil
}

func (m *cycleEngine) RemoveSeq(slot int) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.live, slot)
	m.free = append(m.free, slot)
	return nil
}

// cycleSpecEngine adds a lossless verify over the same successor: it accepts the longest
// draft prefix that matches the true continuation and appends one bonus token.
type cycleSpecEngine struct {
	*cycleEngine
	rmu        sync.Mutex
	maxRun     int
	acceptSeen int // total accepted draft tokens across the run
}

func (m *cycleSpecEngine) StepBatchSpec(reqs []SpecReq) ([][]int32, error) {
	out := make([][]int32, len(reqs))
	maxRun, accepted := 0, 0
	for i, r := range reqs {
		t := succCycle(r.Anchor) // target for the anchor row
		run := []int32{t}
		for j := 0; j < len(r.Drafts); j++ {
			if r.Drafts[j] != t { // draft must equal the target to be accepted
				break
			}
			accepted++
			t = succCycle(r.Drafts[j]) // next target after committing the accepted draft
			run = append(run, t)
		}
		if len(run) > maxRun {
			maxRun = len(run)
		}
		out[i] = run
	}
	m.rmu.Lock()
	if maxRun > m.maxRun {
		m.maxRun = maxRun
	}
	m.acceptSeen += accepted
	m.rmu.Unlock()
	return out, nil
}

func (m *cycleSpecEngine) stats() (maxRun, accepted int) {
	m.rmu.Lock()
	defer m.rmu.Unlock()
	return m.maxRun, m.acceptSeen
}

// cyclePrompt repeats the cycle so the prompt-lookup drafter immediately finds the
// continuation and proposes correct (acceptable) drafts.
func cyclePrompt() []int32 { return []int32{10, 11, 12, 10, 11, 12} }

func runOne(t *testing.T, eng BatchEngine, prompt []int32, maxNew int) ([]int32, Result) {
	t.Helper()
	sched := New(eng, 8)
	sched.Start()
	defer sched.Shutdown()
	col := &collector{}
	done := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: prompt,
		MaxNew: maxNew,
		Ctx:    context.Background(),
		Emit:   col.emit,
		Done:   done,
	}); err != nil {
		t.Fatalf("submit: %v", err)
	}
	res := waitResult(t, done)
	return col.got(), res
}

// LOSSLESS: speculative greedy generation is byte-identical to non-speculative greedy
// generation over the same deterministic target.
func TestSpecLosslessVsPlain(t *testing.T) {
	const maxNew = 30
	plainToks, plainRes := runOne(t, newCycleEngine(1), cyclePrompt(), maxNew)

	spec := &cycleSpecEngine{cycleEngine: newCycleEngine(1)}
	specToks, specRes := runOne(t, spec, cyclePrompt(), maxNew)

	if !reflect.DeepEqual(plainToks, specToks) {
		t.Fatalf("spec output != plain output (LOSSLESS violated)\n  plain = %v\n  spec  = %v", plainToks, specToks)
	}
	if plainRes.Generated != maxNew || specRes.Generated != maxNew {
		t.Errorf("generated: plain=%d spec=%d want %d", plainRes.Generated, specRes.Generated, maxNew)
	}
	// The speculative path must actually have accepted drafts (multi-token runs), otherwise
	// "lossless" is trivially satisfied by never speculating.
	maxRun, accepted := spec.stats()
	if maxRun <= 1 {
		t.Errorf("max committed run = %d, want > 1 (spec never accepted a draft)", maxRun)
	}
	if accepted == 0 {
		t.Errorf("no drafts accepted across the run — spec did no real work")
	}
}
