package batch

import (
	"context"
	"sync"
	"testing"
)

// specEngine wraps mockEngine and adds StepBatchSpec, so the scheduler auto-detects
// SpecBatchEngine and takes the speculative path. It records the reqs it was given
// (to assert drafts were proposed) and commits `accept` drafts + one bonus token
// per req — modelling a verify that accepts a prefix of the draft.
type specEngine struct {
	*mockEngine
	smu      sync.Mutex
	specReqs [][]SpecReq
	accept   int
}

func (s *specEngine) StepBatchSpec(reqs []SpecReq) ([][]int32, error) {
	s.smu.Lock()
	cp := make([]SpecReq, len(reqs))
	for i, r := range reqs {
		d := make([]int32, len(r.Drafts))
		copy(d, r.Drafts)
		cp[i] = SpecReq{Slot: r.Slot, Anchor: r.Anchor, Drafts: d}
	}
	s.specReqs = append(s.specReqs, cp)
	s.smu.Unlock()

	out := make([][]int32, len(reqs))
	for i, r := range reqs {
		na := s.accept
		if na > len(r.Drafts) {
			na = len(r.Drafts)
		}
		run := make([]int32, 0, na+1)
		run = append(run, r.Drafts[:na]...) // accepted draft prefix
		run = append(run, 700000+r.Anchor)  // distinct bonus token
		out[i] = run
	}
	return out, nil
}

func (s *specEngine) maxVerifyRows() int {
	s.smu.Lock()
	defer s.smu.Unlock()
	max := 0
	for _, reqs := range s.specReqs {
		total := 0
		for _, r := range reqs {
			total += 1 + len(r.Drafts)
		}
		if total > max {
			max = total
		}
	}
	return max
}

func (s *specEngine) sawAnyDraft() bool {
	s.smu.Lock()
	defer s.smu.Unlock()
	for _, reqs := range s.specReqs {
		for _, r := range reqs {
			if len(r.Drafts) > 0 {
				return true
			}
		}
	}
	return false
}

func (s *specEngine) maxRunLen() int {
	s.smu.Lock()
	defer s.smu.Unlock()
	max := 0
	for _, reqs := range s.specReqs {
		for _, r := range reqs {
			na := s.accept
			if na > len(r.Drafts) {
				na = len(r.Drafts)
			}
			if rl := na + 1; rl > max {
				max = rl
			}
		}
	}
	return max
}

// A prompt crafted so that, once mockEngine's first sampled token for slot 0 (=1) is
// appended, the history tail [3 1] recurs earlier — so prompt-lookup proposes a draft on
// the very first step. (mockEngine.AddSeq returns slot*1000+1; slot 0 → 1.)
func repeatingPrompt() []int32 {
	return []int32{9, 1, 2, 3, 1, 2, 3}
}

// The scheduler auto-enables speculation when the engine is a SpecBatchEngine, drafts
// from the sequence history, and commits multi-token accepted runs.
func TestSpecPathDraftsAndCommitsRuns(t *testing.T) {
	eng := &specEngine{mockEngine: newMockEngine(1), accept: 3}
	sched := New(eng, 16)
	if sched.spec == nil {
		t.Fatal("scheduler did not detect SpecBatchEngine (spec path off)")
	}
	sched.Start()
	defer sched.Shutdown()

	col := &collector{}
	done := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: repeatingPrompt(),
		MaxNew: 12,
		Ctx:    context.Background(),
		Emit:   col.emit,
		Done:   done,
	}); err != nil {
		t.Fatalf("submit: %v", err)
	}
	res := waitResult(t, done)
	if res.Reason != FinishLength {
		t.Errorf("reason = %q want %q", res.Reason, FinishLength)
	}
	if res.Generated != 12 {
		t.Errorf("generated = %d want 12", res.Generated)
	}
	if !eng.sawAnyDraft() {
		t.Error("no drafts were proposed — prompt-lookup did not fire on a repeating prompt")
	}
	// Smoke: a committed run must exceed one token (drafts were accepted), which is the
	// whole point of speculation — one weight pass committing multiple tokens.
	if got := eng.maxRunLen(); got <= 1 {
		t.Errorf("max committed run length = %d, want > 1 (no multi-token accepted runs)", got)
	}
}

// Σ(1+drafts) must never exceed the engine's verify-row budget, at any concurrency.
func TestSpecBudgetNeverExceedsVerifyRows(t *testing.T) {
	const n = 8
	eng := &specEngine{mockEngine: newMockEngine(n), accept: 2}
	sched := New(eng, 32)
	sched.Start()
	defer sched.Shutdown()

	dones := make([]chan Result, n)
	for i := 0; i < n; i++ {
		dones[i] = make(chan Result, 1)
		col := &collector{}
		if err := sched.Submit(Request{
			Tokens: repeatingPrompt(),
			MaxNew: 6,
			Ctx:    context.Background(),
			Emit:   col.emit,
			Done:   dones[i],
		}); err != nil {
			t.Fatalf("submit %d: %v", i, err)
		}
	}
	for i := 0; i < n; i++ {
		waitResult(t, dones[i])
	}
	if got := eng.maxVerifyRows(); got > MaxVerifyRows {
		t.Errorf("max verify rows in a step = %d, exceeds budget %d", got, MaxVerifyRows)
	}
}

// sparseSpecEngine is a SpecBatchEngine that declines speculation via SpecGater —
// the sparse/MoE case (expert reads scale with draft length, spec brings no value).
type sparseSpecEngine struct{ *specEngine }

func (s *sparseSpecEngine) SpecWorthwhile() bool { return false }

// A sparse (MoE) engine implements SpecBatchEngine but declines via SpecGater:
// the scheduler must leave the spec path OFF and serve plain batched decode.
func TestSpecGaterDisablesSpecForSparse(t *testing.T) {
	eng := &sparseSpecEngine{specEngine: &specEngine{mockEngine: newMockEngine(1), accept: 3}}
	sched := New(eng, 16)
	if sched.spec != nil {
		t.Fatal("scheduler enabled speculation for an engine whose SpecGater declined")
	}
	sched.Start()
	defer sched.Shutdown()

	col := &collector{}
	done := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: repeatingPrompt(),
		MaxNew: 8,
		Ctx:    context.Background(),
		Emit:   col.emit,
		Done:   done,
	}); err != nil {
		t.Fatalf("submit: %v", err)
	}
	res := waitResult(t, done)
	if res.Reason != FinishLength {
		t.Errorf("reason = %q want %q", res.Reason, FinishLength)
	}
	if res.Generated != 8 {
		t.Errorf("generated = %d want 8", res.Generated)
	}
	if eng.sawAnyDraft() {
		t.Error("drafts were verified on a sparse-gated engine — spec path should be off")
	}
}

// The cross-request suffix corpus drafts continuations for a NEW sequence whose own
// history has no repeats, by matching its tail against a finished sequence's tokens.
func TestCorpusDraftCrossRequest(t *testing.T) {
	corpus := []int32{7, 8, 9, 1, 2, 3, 4, 5, 6, 42}
	hist := []int32{100, 101, 1, 2, 3} // tail [1 2 3] occurs in corpus, followed by 4 5 6 42
	d, c := promptLookupDraftInConf(corpus, hist, 4, 3, 4)
	if len(d) == 0 {
		t.Fatal("corpus lookup proposed nothing")
	}
	want := []int32{4, 5, 6}
	for i := range want {
		if i >= len(d) || d[i] != want[i] {
			t.Fatalf("draft = %v want prefix %v", d, want)
		}
	}
	if len(c) != len(d) {
		t.Fatalf("conf len %d != draft len %d", len(c), len(d))
	}
	// own-history lookup on the same hist proposes nothing (no internal repeat)
	if d2, _ := promptLookupDraftConf(hist, 4, 2, 4); len(d2) != 0 {
		t.Fatalf("own-history draft should be empty, got %v", d2)
	}
}
