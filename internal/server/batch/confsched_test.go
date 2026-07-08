package batch

import (
	"context"
	"reflect"
	"sync"
	"testing"
)

// cumulative folds raw per-position confidences into the non-increasing survival
// sequence a_j=∏_{i≤j} c_i the scheduler consumes (mirrors stepSpec).
func cumulative(c []float32) []float32 {
	if len(c) == 0 {
		return nil
	}
	a := make([]float32, len(c))
	p := float32(1)
	for j, cj := range c {
		p *= cj
		a[j] = p
	}
	return a
}

func sumInts(xs []int) int {
	t := 0
	for _, x := range xs {
		t += x
	}
	return t
}

// No requests → no admissions, no panic.
func TestScheduleConfidenceEmpty(t *testing.T) {
	if got := scheduleConfidence(nil, MaxVerifyRows); len(got) != 0 {
		t.Fatalf("admit = %v, want empty", got)
	}
}

// A lone request with certain drafts fills the free verify rows: all its candidate
// drafts are admitted (low concurrency → speculation grows).
func TestScheduleConfidenceSingleStreamGrows(t *testing.T) {
	surv := [][]float32{cumulative([]float32{1, 1, 1, 1, 1, 1})}
	admit := scheduleConfidence(surv, MaxVerifyRows)
	if admit[0] != 6 {
		t.Fatalf("admit = %v, want all 6 drafts admitted for a single high-confidence stream", admit)
	}
}

// The total verify rows Σ(1+admit_r) must never exceed the engine's hard cap, even when
// every request offers a long, fully-confident candidate draft. R ranges over the
// realizable concurrency (≤ MaxVerifyRows: each active slot owns one anchor row, so the
// scheduler is never handed more requests than the row budget).
func TestScheduleConfidenceNeverExceedsBudget(t *testing.T) {
	for R := 1; R <= MaxVerifyRows; R++ {
		surv := make([][]float32, R)
		long := cumulative([]float32{1, 1, 1, 1, 1, 1, 1, 1})
		for r := range surv {
			surv[r] = long
		}
		admit := scheduleConfidence(surv, MaxVerifyRows)
		if rows := R + sumInts(admit); rows > MaxVerifyRows {
			t.Fatalf("R=%d: total verify rows = %d, exceeds cap %d (admit=%v)", R, rows, MaxVerifyRows, admit)
		}
	}
}

// At concurrency ≥ the row cap every row is an anchor, so no draft can be admitted and the
// step degenerates to a plain batched decode (the paper's high-load behavior).
func TestScheduleConfidenceSaturatedAdmitsZero(t *testing.T) {
	const R = MaxVerifyRows
	surv := make([][]float32, R)
	for r := range surv {
		surv[r] = cumulative([]float32{1, 1, 1})
	}
	admit := scheduleConfidence(surv, MaxVerifyRows)
	if s := sumInts(admit); s != 0 {
		t.Fatalf("admit sum = %d, want 0 at saturation (admit=%v)", s, admit)
	}
}

// Admission is by SURVIVAL, not request index: when the budget is tight, the
// higher-survival request wins the scarce verify row regardless of its position.
func TestScheduleConfidencePrefersHigherSurvival(t *testing.T) {
	// R just below saturation so exactly one extra draft row clears the throughput bar.
	const R = specSaturationRows - 1
	surv := make([][]float32, R)
	for r := range surv {
		surv[r] = nil
	}
	surv[0] = cumulative([]float32{0.5}) // lower survival, earlier index
	surv[1] = cumulative([]float32{1.0}) // higher survival, later index
	admit := scheduleConfidence(surv, MaxVerifyRows)
	if admit[1] != 1 || admit[0] != 0 {
		t.Fatalf("admit = %v, want the higher-survival request (index 1) to get the row, not index 0", admit)
	}
}

// Load adaptivity: a request gets a LONGER verify length at low concurrency than at high
// concurrency — drafts taper as the batch fills (the prefix scheduler's whole point).
func TestScheduleConfidenceTapersWithConcurrency(t *testing.T) {
	mk := func(R int) []int {
		surv := make([][]float32, R)
		long := cumulative([]float32{1, 1, 1, 1, 1, 1})
		for r := range surv {
			surv[r] = long
		}
		return scheduleConfidence(surv, MaxVerifyRows)
	}
	lo := mk(1)[0]               // per-request drafts at concurrency 1
	hi := mk(specSaturationRows) // per-request drafts at saturation
	for _, k := range hi {
		if k >= lo {
			t.Fatalf("per-request drafts did not taper: low-concurrency=%d high-concurrency=%v", lo, hi)
		}
	}
}

// ─── End-to-end losslessness under concurrency ─────────────────────
//
// The confidence scheduler only sets draft LENGTHS; the verify is lossless, so concurrent
// speculative generation must be byte-identical to plain greedy generation for every
// sequence. This drives the real scheduler through the multi-request stepSpec path (where
// scheduleConfidence distributes the row budget) and compares against the plain reference.
func TestConfSchedLosslessUnderConcurrency(t *testing.T) {
	const (
		N      = 3 // below saturation, so drafts are actually admitted
		maxNew = 40
	)
	// Plain single-stream reference (deterministic cycle).
	want, _ := runOne(t, newCycleEngine(1), cyclePrompt(), maxNew)

	spec := &cycleSpecEngine{cycleEngine: newCycleEngine(N)}
	sched := New(spec, 16)
	if sched.spec == nil {
		t.Fatal("spec path not detected")
	}
	sched.Start()
	defer sched.Shutdown()

	cols := make([]*collector, N)
	dones := make([]chan Result, N)
	var wg sync.WaitGroup
	for i := 0; i < N; i++ {
		cols[i] = &collector{}
		dones[i] = make(chan Result, 1)
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if err := sched.Submit(Request{
				Tokens: cyclePrompt(),
				MaxNew: maxNew,
				Ctx:    context.Background(),
				Emit:   cols[i].emit,
				Done:   dones[i],
			}); err != nil {
				t.Errorf("submit %d: %v", i, err)
			}
		}(i)
	}
	wg.Wait()
	for i := 0; i < N; i++ {
		res := waitResult(t, dones[i])
		if res.Reason != FinishLength || res.Generated != maxNew {
			t.Errorf("seq %d: reason=%q generated=%d, want length/%d", i, res.Reason, res.Generated, maxNew)
		}
		if got := cols[i].got(); !reflect.DeepEqual(got, want) {
			t.Fatalf("seq %d not lossless vs plain\n  want = %v\n  got  = %v", i, want, got)
		}
	}
	// Speculation must have done real work (accepted multi-token runs), else "lossless" is
	// trivially true by never speculating.
	if maxRun, accepted := spec.stats(); maxRun <= 1 || accepted == 0 {
		t.Errorf("spec did no work under concurrency: maxRun=%d accepted=%d", maxRun, accepted)
	}
}
