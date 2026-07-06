package batch

import "sort"

// ─── DSpark hardware-aware prefix scheduler ────────────────────────
//
// SpecScheduler implements DSpark Algorithm 1 (Hardware-Aware Prefix
// Scheduler) as a PURE policy module: given the drafter's calibrated
// per-position acceptance confidences and a profiled step-cost table, it
// decides how many drafted tokens each active request should put up for the
// engine's exact ragged rejection-sampling verify on the next step.
//
// It has NO engine or CUDA dependency. It operates on plain inputs so it can
// be unit-tested with mocks, and is wired into the run loop later (the
// ragged-verify engine ABI it needs does not exist yet).
//
// ── The cost model ──────────────────────────────────────────────────
//
//	a[r][j] = Π_{i<=j} c[r][i]        survival (joint acceptance) of the
//	                                  first j drafted positions of request r,
//	                                  with c[r][i] ∈ (0,1] the drafter's
//	                                  per-position acceptance probability.
//
//	τ*      = R + Σ_r Σ_{j<=ℓ[r]} a[r][j]   expected accepted tokens this step
//	                                        (R anchor tokens always accepted +
//	                                        the survival mass of admitted drafts)
//
//	B       = R + Σ_r ℓ[r]            total forward rows submitted (R anchors +
//	                                  one row per admitted draft position)
//
//	Θ(ℓ)    = τ* · SPS(B)             expected accepted TOKENS PER SECOND,
//	                                  the quantity we maximize. SPS(B) is the
//	                                  profiled steps-per-second at B forward
//	                                  rows (monotone non-increasing).
//
// ── Why the greedy is optimal ───────────────────────────────────────
//
// Two structural facts make Algorithm 1 a single descending pass with an
// early stop:
//
//  1. PREFIX-FREE SORTING. a[r][j] is monotone non-increasing in j (each
//     factor c ∈ (0,1]). Sorting the global candidate set E = {(r,j)} by
//     DESCENDING survival therefore AUTOMATICALLY respects the per-request
//     prefix constraint: position j of request r can only be reached after
//     j-1 of the same request has already been admitted. The implementation
//     RELIES on this and ASSERTS it (see the ell[r] == j-1 check below).
//
//  2. UNIMODAL Θ ⇒ STEPWISE EARLY STOP. Because SPS(B) is monotone
//     non-increasing, Θ as drafts are added in descending-survival order is
//     unimodal: each admitted draft adds a[r][j] to τ* (a shrinking gain,
//     since survival is sorted descending) while SPS(B) only falls. Once Θ
//     stops rising it never rises again, so BREAKING on the first Θ drop
//     finds the global maximum.
//
// ── Why this is LOSSLESS / NON-ANTICIPATING ─────────────────────────
//
// The break is what makes the policy non-anticipating: ℓ*[r] is a function
// of confidences + SPS ONLY (past / draft-side information), never of the
// target token that is about to be sampled. Truncating a low-survival draft
// tail merely DEFERS those positions to the next step's ordinary decode; it
// can NEVER change which tokens are ultimately accepted, because acceptance
// is decided by the engine's exact rejection-sampling verify OUTSIDE this
// module. Concretely this shows up as TAIL-TRUNCATION INVARIANCE: appending
// extra low-survival candidates past the chosen prefix does not change the
// chosen prefix (see the test of the same name).
type SpecScheduler struct {
	// sps maps total forward rows B → steps-per-second. It MUST be monotone
	// non-increasing for the unimodality argument (and hence the early stop)
	// to hold. Never nil after construction.
	sps func(b int) float64
}

// NewSpecScheduler constructs a scheduler over an explicit SPS(B) function.
// If sps is nil a constant unit step-cost is used (every plan then admits all
// positive-survival drafts), so the scheduler is always safe to call.
func NewSpecScheduler(sps func(b int) float64) *SpecScheduler {
	s := &SpecScheduler{}
	s.SetSPS(sps)
	return s
}

// NewSpecSchedulerTable constructs a scheduler from a profiled SPS lookup
// slice indexed by total forward rows B (table[B] = steps-per-second at B
// rows). Out-of-range B is clamped to the table ends. The table SHOULD be
// monotone non-increasing.
func NewSpecSchedulerTable(table []float64) *SpecScheduler {
	return NewSpecScheduler(SPSFromTable(table))
}

// SetSPS replaces the step-cost model. A nil sps installs a constant unit
// cost. Not safe to call concurrently with BuildVerifyPlan.
func (s *SpecScheduler) SetSPS(sps func(b int) float64) {
	if sps == nil {
		sps = func(int) float64 { return 1 }
	}
	s.sps = sps
}

// SPSFromTable turns a profiled lookup slice indexed by total forward rows
// into an SPS(B) function, clamping B into [0, len-1]. An empty table yields a
// constant unit cost.
func SPSFromTable(table []float64) func(int) float64 {
	if len(table) == 0 {
		return func(int) float64 { return 1 }
	}
	return func(b int) float64 {
		if b < 0 {
			b = 0
		}
		if b >= len(table) {
			b = len(table) - 1
		}
		return table[b]
	}
}

// candidate is one admissible draft position (r, j) with its survival mass.
type candidate struct {
	r    int     // request index
	j    int     // 1-based draft position within request r
	surv float64 // a[r][j] = Π_{i<=j} c[r][i]
}

// BuildVerifyPlan runs DSpark Algorithm 1 and returns the per-request verify
// length ℓ*[r]: how many drafted tokens request r should submit for verify on
// the next step. The result has length len(conf); entry r is in [0, len(conf[r])].
//
// conf[r] is the drafter's per-position acceptance confidence sequence
// c[r][1..γ_r] for request r, each value in (0,1]. R is the number of active
// requests (anchor rows). Values are clamped defensively: a confidence > 1 is
// treated as 1 (keeps survival monotone) and a confidence <= 0 terminates that
// request's draft tail (its survival, and everything past it, is 0).
//
// The pass is allocation-light (two reused int slices plus the candidate set)
// and deterministic: the candidate sort has a total order (survival desc, then
// request index asc, then position asc), so the chosen plan is reproducible and
// the prefix constraint is preserved even when survivals tie (c == 1).
func (s *SpecScheduler) BuildVerifyPlan(conf [][]float64, R int) []int {
	best := make([]int, len(conf))
	if R <= 0 || len(conf) == 0 {
		return best
	}

	// Build the candidate set E and the survival values. Survival is computed
	// as a running prefix product, so it is monotone non-increasing by
	// construction; we stop a request's tail as soon as survival hits 0.
	E := make([]candidate, 0, totalPositions(conf))
	for r := range conf {
		surv := 1.0
		for j := 1; j <= len(conf[r]); j++ {
			c := conf[r][j-1]
			if c > 1 {
				c = 1 // clamp: keep survival monotone non-increasing
			}
			if c <= 0 {
				break // survival is 0 from here on; no more admissible positions
			}
			surv *= c
			if surv <= 0 {
				break // underflow to 0: deeper positions are also 0
			}
			E = append(E, candidate{r: r, j: j, surv: surv})
		}
	}

	// Sort DESCENDING by survival. The tiebreak (request asc, position asc)
	// gives a total order, which (a) makes the plan deterministic and (b)
	// guarantees that for a fixed request lower positions are visited first —
	// the property the prefix-constraint assertion below relies on, even when
	// c == 1 makes consecutive survivals equal.
	sort.SliceStable(E, func(a, b int) bool {
		ea, eb := E[a], E[b]
		if ea.surv != eb.surv {
			return ea.surv > eb.surv
		}
		if ea.r != eb.r {
			return ea.r < eb.r
		}
		return ea.j < eb.j
	})

	// Greedy with non-anticipating early stop (Algorithm 1).
	ell := make([]int, len(conf)) // current working plan ℓ[r]
	B := R                        // total forward rows: R anchors + admitted drafts
	tau := float64(R)             // expected accepted tokens: R anchors + survival mass
	thetaBest := tau * s.sps(B)   // Θ at the empty plan (no drafts)

	for _, e := range E {
		// Prefix-constraint invariant: because survival is monotone and the
		// sort visits lower positions of a request first, admitting (r, j)
		// must find (r, j-1) already admitted. This is what lets a single
		// global sort respect every per-request prefix without bookkeeping.
		if ell[e.r] != e.j-1 {
			panic("batch: specsched prefix invariant violated (non-monotone survival)")
		}

		ell[e.r] = e.j
		B++
		tau += e.surv
		theta := tau * s.sps(B)

		if theta > thetaBest {
			thetaBest = theta
			copy(best, ell) // commit the improved plan (no allocation)
		} else {
			// First Θ drop: Θ is unimodal, so it never rises again. Stopping
			// here is the non-anticipating / lossless choice — the truncated
			// tail simply defers to next step's ordinary decode.
			break
		}
	}

	return best
}

// totalPositions sums γ_r across all requests, to size the candidate slice
// once and avoid repeated growth.
func totalPositions(conf [][]float64) int {
	n := 0
	for r := range conf {
		n += len(conf[r])
	}
	return n
}
