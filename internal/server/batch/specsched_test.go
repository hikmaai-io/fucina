package batch

import (
	"math"
	"reflect"
	"testing"
)

// flatSPS is a forgiving cost model: steps-per-second is independent of the
// number of forward rows, so Θ = τ*·const grows with every admitted draft and
// the greedy never breaks (admits every positive-survival position).
func flatSPS(int) float64 { return 100.0 }

// sharpSPS decays steeply in B: each extra forward row roughly halves the
// achievable steps-per-second. Under contention (high R, hence high base B)
// this makes even the first draft a net loss, so the plan admits ~0 drafts.
func sharpSPS(b int) float64 {
	if b < 0 {
		b = 0
	}
	return 1000.0 / math.Pow(2, float64(b))
}

// ─── (1) monotone survival ⇒ admitted lengths are valid prefixes ────
//
// The plan length is the only thing this module returns, but a length ℓ
// implicitly means "positions 1..ℓ admitted". So the prefix property reduces
// to: ℓ[r] is a contiguous count in [0, γ_r], never exceeding the available
// confidences. The greedy admitting position j only after j-1 (asserted
// internally) is what guarantees there is never a gap. We also verify the
// internal assertion does not trip on well-formed monotone input.
func TestBuildVerifyPlan_PrefixLengths(t *testing.T) {
	cases := []struct {
		name string
		conf [][]float64
		R    int
	}{
		{
			name: "decaying confidences",
			conf: [][]float64{
				{0.9, 0.8, 0.7, 0.6},
				{0.95, 0.5, 0.2},
				{0.99},
			},
			R: 3,
		},
		{
			name: "flat-1 confidences (survival ties)",
			conf: [][]float64{
				{1, 1, 1},
				{1, 1},
			},
			R: 2,
		},
		{
			name: "early zero terminates tail",
			conf: [][]float64{
				{0.9, 0.0, 0.9, 0.9}, // a zero confidence cuts the tail
			},
			R: 1,
		},
	}

	s := NewSpecScheduler(flatSPS)
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan := s.BuildVerifyPlan(tc.conf, tc.R)
			if len(plan) != len(tc.conf) {
				t.Fatalf("plan length = %d, want %d", len(plan), len(tc.conf))
			}
			for r, l := range plan {
				if l < 0 {
					t.Errorf("request %d: negative verify length %d", r, l)
				}
				// A length must be a true prefix: it cannot exceed the number
				// of available positions, and (early-zero case) cannot extend
				// past a zero confidence.
				maxLen := prefixUntilZero(tc.conf[r])
				if l > maxLen {
					t.Errorf("request %d: length %d exceeds admissible prefix %d", r, l, maxLen)
				}
			}
		})
	}
}

// prefixUntilZero is the longest admissible prefix: it stops at the first
// non-positive confidence (survival 0 from there on).
func prefixUntilZero(c []float64) int {
	for i, v := range c {
		if v <= 0 {
			return i
		}
	}
	return len(c)
}

// ─── (2) Θ early-stop: contention vs. forgiving cost models ─────────
//
// A sharply-decaying SPS under high R (heavy contention) should admit ~0
// drafts, because each extra forward row costs more throughput than the draft
// returns. A flat/forgiving SPS should admit (almost) everything.
func TestBuildVerifyPlan_EarlyStopContention(t *testing.T) {
	// 6 requests, each offering a couple of high-confidence drafts.
	conf := [][]float64{
		{0.9, 0.85},
		{0.9, 0.85},
		{0.9, 0.85},
		{0.9, 0.85},
		{0.9, 0.85},
		{0.9, 0.85},
	}
	R := len(conf)

	sharp := NewSpecScheduler(sharpSPS)
	sharpPlan := sharp.BuildVerifyPlan(conf, R)
	if got := sumPlan(sharpPlan); got != 0 {
		t.Errorf("sharp SPS under contention: admitted %d drafts, want 0 (plan=%v)", got, sharpPlan)
	}

	flat := NewSpecScheduler(flatSPS)
	flatPlan := flat.BuildVerifyPlan(conf, R)
	wantFlat := totalPositions(conf) // flat SPS never breaks: admit all positions
	if got := sumPlan(flatPlan); got != wantFlat {
		t.Errorf("flat SPS: admitted %d drafts, want %d (plan=%v)", got, wantFlat, flatPlan)
	}
	if sumPlan(flatPlan) <= sumPlan(sharpPlan) {
		t.Errorf("forgiving SPS should admit strictly more than contended SPS: flat=%d sharp=%d",
			sumPlan(flatPlan), sumPlan(sharpPlan))
	}
}

// A gentle SPS sits between the two extremes: it admits some but not all
// drafts, and the early stop must land exactly where Θ peaks.
func TestBuildVerifyPlan_EarlyStopPeak(t *testing.T) {
	// Linear-ish gentle decay so a few high-survival drafts pay off but a long
	// low-survival tail does not.
	gentle := func(b int) float64 {
		v := 100.0 - float64(b)
		if v < 1 {
			v = 1
		}
		return v
	}
	conf := [][]float64{
		{0.99, 0.98, 0.5, 0.1, 0.05}, // strong head, weak tail
	}
	s := NewSpecScheduler(gentle)
	plan := s.BuildVerifyPlan(conf, 1)

	// Brute-force the optimum over all prefix lengths for this single request
	// and confirm the greedy matches it.
	wantLen := bruteForceSingle(conf[0], 1, gentle)
	if plan[0] != wantLen {
		t.Errorf("plan length = %d, want %d (brute force)", plan[0], wantLen)
	}
}

// ─── (3) NON-ANTICIPATING / LOSSLESS proxy: tail-truncation invariance
//
// The plan depends only on (conf, SPS). Appending EXTRA low-survival drafts to
// the END of each request's confidence sequence must not change the already
// selected prefix: those candidates sort after the chosen ones and lie past
// the early-stop break. This is the testable face of "non-anticipating /
// lossless": deferring a low-survival tail never alters the decision.
func TestBuildVerifyPlan_TailTruncationInvariance(t *testing.T) {
	base := [][]float64{
		{0.9, 0.7, 0.4},
		{0.95, 0.6},
		{0.8},
	}
	R := len(base)

	// A cost model that admits a strict subset (so there IS a chosen prefix to
	// be invariant about, not "admit everything").
	gentle := func(b int) float64 {
		v := 50.0 - 3*float64(b)
		if v < 1 {
			v = 1
		}
		return v
	}
	s := NewSpecScheduler(gentle)

	want := s.BuildVerifyPlan(base, R)
	if sumPlan(want) == 0 || sumPlan(want) == totalPositions(base) {
		t.Fatalf("test misconfigured: need a partial plan, got %v", want)
	}

	// Extend every request with a long, very-low-survival tail.
	extended := make([][]float64, len(base))
	for r := range base {
		extended[r] = append(append([]float64{}, base[r]...), 0.3, 0.2, 0.1, 0.05)
	}

	got := s.BuildVerifyPlan(extended, R)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("tail-truncation NOT invariant:\n  base plan     = %v\n  extended plan = %v", want, got)
	}
}

// Determinism: identical inputs (including survival ties) yield identical plans
// across repeated calls. Guards the stable-sort + deterministic-tiebreak claim.
func TestBuildVerifyPlan_Deterministic(t *testing.T) {
	conf := [][]float64{
		{0.8, 0.8, 0.8}, // ties in survival progression across requests
		{0.8, 0.8, 0.8},
		{0.8, 0.8, 0.8},
	}
	gentle := func(b int) float64 { return 30.0 - float64(b) }
	s := NewSpecScheduler(gentle)

	first := s.BuildVerifyPlan(conf, 3)
	for i := 0; i < 20; i++ {
		if got := s.BuildVerifyPlan(conf, 3); !reflect.DeepEqual(got, first) {
			t.Fatalf("non-deterministic plan on run %d: %v != %v", i, got, first)
		}
	}
}

// ─── (4) edge cases: must not crash ─────────────────────────────────

func TestBuildVerifyPlan_Edges(t *testing.T) {
	s := NewSpecScheduler(flatSPS)

	if got := s.BuildVerifyPlan(nil, 0); len(got) != 0 {
		t.Errorf("nil conf, R=0: want empty plan, got %v", got)
	}
	if got := s.BuildVerifyPlan([][]float64{}, 0); len(got) != 0 {
		t.Errorf("empty conf: want empty plan, got %v", got)
	}
	// R=0 with non-empty conf: no active requests ⇒ zero plan, no crash.
	if got := s.BuildVerifyPlan([][]float64{{0.9}}, 0); !reflect.DeepEqual(got, []int{0}) {
		t.Errorf("R=0 with conf: want [0], got %v", got)
	}
	// Request with no draft positions.
	if got := s.BuildVerifyPlan([][]float64{{}}, 1); !reflect.DeepEqual(got, []int{0}) {
		t.Errorf("empty confidence sequence: want [0], got %v", got)
	}
	// Single request, single position, forgiving cost ⇒ admit it.
	if got := s.BuildVerifyPlan([][]float64{{0.9}}, 1); !reflect.DeepEqual(got, []int{1}) {
		t.Errorf("single request single draft: want [1], got %v", got)
	}
	// Nil SPS falls back to constant cost (admit all), no crash.
	sn := NewSpecScheduler(nil)
	if got := sn.BuildVerifyPlan([][]float64{{0.9, 0.8}}, 1); !reflect.DeepEqual(got, []int{2}) {
		t.Errorf("nil SPS: want [2], got %v", got)
	}
}

// SPSFromTable clamps out-of-range B to the table ends.
func TestSPSFromTable(t *testing.T) {
	f := SPSFromTable([]float64{10, 8, 5})
	checks := []struct {
		b    int
		want float64
	}{
		{-5, 10}, {0, 10}, {1, 8}, {2, 5}, {3, 5}, {100, 5},
	}
	for _, c := range checks {
		if got := f(c.b); got != c.want {
			t.Errorf("SPSFromTable(%d) = %v, want %v", c.b, got, c.want)
		}
	}
	// Empty table ⇒ constant unit cost.
	if got := SPSFromTable(nil)(7); got != 1 {
		t.Errorf("empty table: want 1, got %v", got)
	}
}

// ─── helpers ────────────────────────────────────────────────────────

func sumPlan(plan []int) int {
	n := 0
	for _, l := range plan {
		n += l
	}
	return n
}

// bruteForceSingle finds the Θ-maximizing prefix length for a single request,
// independently of the greedy, by evaluating every prefix length directly.
func bruteForceSingle(c []float64, R int, sps func(int) float64) int {
	bestLen := 0
	bestTheta := float64(R) * sps(R) // empty plan
	surv := 1.0
	tau := float64(R)
	B := R
	for j := 1; j <= len(c); j++ {
		if c[j-1] <= 0 {
			break
		}
		cv := c[j-1]
		if cv > 1 {
			cv = 1
		}
		surv *= cv
		tau += surv
		B++
		theta := tau * sps(B)
		if theta > bestTheta {
			bestTheta = theta
			bestLen = j
		}
		// Note: this brute force keeps scanning (it does not early-stop), so it
		// is a genuine independent check that the greedy's early stop lands on
		// the global optimum for a unimodal Θ.
	}
	return bestLen
}
