package batch

// MinNew (vLLM min_tokens) scheduler contract: a stop id sampled before MinNew
// generated tokens is delivered like an ordinary token and generation
// continues; the first stop id with at least MinNew tokens BEFORE it evicts
// with FinishStop; MinNew is clamped to MaxNew so MinNew >= MaxNew yields an
// exact-length FinishLength generation. Host-side only — no engine/kernel
// involvement beyond the tokens it already samples.

import (
	"context"
	"testing"
)

// TestMinNewSuppressesStopUntilThreshold: the engine emits the stop token from
// the 3rd produced token onward. With MinNew=6 the stops at positions 3..6 must
// be suppressed (delivered + committed), and the stop at position 7 (six tokens
// precede it) terminates with FinishStop.
func TestMinNewSuppressesStopUntilThreshold(t *testing.T) {
	eng := newMockEngine(1)
	eng.stopToken = 99
	eng.stopAfter = 3
	sched := New(eng, 8)
	sched.Start()
	defer sched.Shutdown()

	col := &collector{}
	done := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: []int32{1}, MaxNew: 100, MinNew: 6, Ctx: context.Background(),
		Emit: col.emit, Done: done, Stops: []int32{99},
	}); err != nil {
		t.Fatal(err)
	}
	res := waitResult(t, done)
	if res.Reason != FinishStop {
		t.Fatalf("reason = %q want %q", res.Reason, FinishStop)
	}
	got := col.got()
	if len(got) != 7 || res.Generated != 7 {
		t.Fatalf("emitted %v (%d tokens), Generated=%d; want 7 (MinNew=6 + terminal stop)", got, len(got), res.Generated)
	}
	// Positions 3..7 are all the stop token: 3..6 suppressed by MinNew, 7 terminal.
	for i := 2; i < 7; i++ {
		if got[i] != 99 {
			t.Errorf("token %d = %d want 99 (suppressed/terminal stop id)", i+1, got[i])
		}
	}
	if got[0] == 99 || got[1] == 99 {
		t.Errorf("tokens 1-2 should be ordinary, got %v", got[:2])
	}
}

// TestMinNewClampedToMaxNewEndsAtLength: MinNew > MaxNew is clamped to MaxNew,
// so with an engine that emits ONLY stop tokens the sequence still generates
// exactly MaxNew tokens and finishes with FinishLength — the exact-length
// (ignore_eos-equivalent) generation contract.
func TestMinNewClampedToMaxNewEndsAtLength(t *testing.T) {
	eng := newMockEngine(1)
	eng.stopToken = 99
	eng.stopAfter = 1 // every produced token is the stop id
	sched := New(eng, 8)
	sched.Start()
	defer sched.Shutdown()

	col := &collector{}
	done := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: []int32{1}, MaxNew: 5, MinNew: 100, Ctx: context.Background(),
		Emit: col.emit, Done: done, Stops: []int32{99},
	}); err != nil {
		t.Fatal(err)
	}
	res := waitResult(t, done)
	if res.Reason != FinishLength {
		t.Fatalf("reason = %q want %q (MinNew clamped to MaxNew must exhaust the budget)", res.Reason, FinishLength)
	}
	if got := col.got(); len(got) != 5 || res.Generated != 5 {
		t.Fatalf("emitted %d tokens, Generated=%d; want exactly MaxNew=5", len(got), res.Generated)
	}
}

// TestEmptyStopsRunsToMaxNew (ignore_eos wiring contract): with no Stops at
// all, the same all-stop-token engine runs to MaxNew and finishes with
// FinishLength — the server implements ignore_eos by passing Stops: nil.
func TestEmptyStopsRunsToMaxNew(t *testing.T) {
	eng := newMockEngine(1)
	eng.stopToken = 99
	eng.stopAfter = 1
	sched := New(eng, 8)
	sched.Start()
	defer sched.Shutdown()

	col := &collector{}
	done := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: []int32{1}, MaxNew: 4, Ctx: context.Background(),
		Emit: col.emit, Done: done, // no Stops
	}); err != nil {
		t.Fatal(err)
	}
	res := waitResult(t, done)
	if res.Reason != FinishLength {
		t.Fatalf("reason = %q want %q", res.Reason, FinishLength)
	}
	if got := col.got(); len(got) != 4 || res.Generated != 4 {
		t.Fatalf("emitted %d tokens, Generated=%d; want exactly MaxNew=4", len(got), res.Generated)
	}
}
