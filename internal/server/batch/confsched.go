package batch

// confsched.go — the training-free half of DSpark "Confidence-Scheduled Speculative
// Decoding" (DeepSeek, Algorithm 1): the Hardware-Aware Prefix Scheduler.
//
// Plain spec decoding gives every request a fixed draft length, which is wrong under load:
// at high concurrency the verify batch overflows and each extra draft row steals throughput
// from real work, while at low concurrency a timid fixed length leaves free rows (and free
// accepted tokens) on the table. The prefix scheduler fixes this by choosing, per step, HOW
// MANY drafts to verify for EACH request as a function of (a) each draft token's survival
// probability and (b) a hardware cost model of the decode step — so it maximises expected
// accepted-tokens-per-second across the whole batch.
//
// It is LOSSLESS by construction: it only decides draft *lengths*. The engine's verify
// commits the same tokens a plain greedy step would, for any draft length (including zero),
// so changing how many we draft can never change the output — only the speed. The semi-AR
// Markov/RNN drafter (the paper's other half) NEEDS TRAINING and is OUT OF SCOPE here; we
// feed the scheduler the model-agnostic prompt-lookup drafter's consensus confidence instead.

// specSaturationRows is the verify-row count up to which a decode step is weight-bandwidth
// bound: below it the step reads the full weight set once and extra rows ride along at ~zero
// marginal cost (free rows); past it the step turns row-bound and each extra row costs time.
// It is the single shape parameter of the sps cost model and a THROUGHPUT tuning knob only —
// it never affects correctness (verify is lossless at any draft length). Coupled to the
// engine's hard verify-row cap so the "free" region is exactly the first half of the budget.
const specSaturationRows = MaxVerifyRows / 2

// sps models a decode step's steps-per-second as a function of the verify-row count b, up to
// a positive multiplicative constant (which cancels in the throughput RATIO the scheduler
// compares, so the absolute scale is irrelevant). Decode is weight-bound while b ≤
// specSaturationRows (flat SPS — rows are free) and row-bound past it (SPS ∝ 1/b — each row
// costs). This is the "hardware-aware" half of the prefix scheduler: at low concurrency
// drafts grow into the free rows, and as concurrency drives b past saturation only
// high-survival drafts clear the rising throughput bar, so per-request drafts taper toward 0.
func sps(b int) float64 {
	d := b
	if d < specSaturationRows {
		d = specSaturationRows
	}
	return 1.0 / float64(d)
}

// scheduleConfidence is the prefix scheduler's admission loop (DSpark Algorithm 1). Given,
// per request r, the cumulative survival a_{r,j}=∏_{i≤j} c_{r,i} of its candidate draft
// positions (surv[r], which MUST be non-increasing — a cumulative product of accept probs in
// [0,1]) and the engine's hard verify-row budget maxRows, it greedily admits draft tokens
// across ALL requests in descending survival, growing the batch row count B by one per admit,
// and stops the instant expected throughput Θ = τ·SPS(B) would drop. It returns admit[r] =
// the number of leading drafts to keep (verify) for request r this step.
//
//   - τ = Σ_r (1 + Σ_{j≤ℓ_r} a_{r,j}): expected accepted tokens this step (each request
//     contributes 1 guaranteed bonus token plus the survival of each admitted draft).
//   - B = Σ_r (1 + ℓ_r): total verify rows (one anchor per request plus admitted drafts).
//   - Θ = τ·SPS(B): expected accepted-tokens-per-second — the quantity maximised.
//
// Because survival is non-increasing within a request, the globally highest-survival
// remaining candidate is always some request's NEXT un-admitted position; admitting in that
// order both respects per-request draft order and makes the early stop NON-ANTICIPATING: if
// the best remaining candidate cannot raise Θ, no worse one can either, so we stop globally.
// This is exactly what preserves losslessness — we never need to look ahead. At B ≥ maxRows
// (e.g. concurrency ≥ maxRows: all rows are anchors) the loop admits nothing and the step
// degenerates to a plain batched decode, as the paper describes.
func scheduleConfidence(surv [][]float32, maxRows int) []int {
	R := len(surv)
	admit := make([]int, R)
	if R == 0 {
		return admit
	}

	B := R            // one anchor row per request, mandatory
	tau := float64(R) // each request's guaranteed bonus token
	theta := tau * sps(B)

	for B < maxRows {
		// Pick the request whose next un-admitted draft has the highest survival.
		// (Per-request survival is non-increasing, so this is the global max over all
		// remaining candidates — descending-survival order.)
		best := -1
		var bestSurv float32
		for r := 0; r < R; r++ {
			j := admit[r]
			if j >= len(surv[r]) {
				continue // this request has no more candidates
			}
			if best < 0 || surv[r][j] > bestSurv {
				best = r
				bestSurv = surv[r][j]
			}
		}
		if best < 0 {
			break // no candidates left across any request
		}

		// Tentatively admit it: +1 verify row, +survival expected tokens.
		nTau := tau + float64(bestSurv)
		nB := B + 1
		nTheta := nTau * sps(nB)
		if nTheta <= theta {
			break // throughput would not rise → non-anticipating early stop
		}
		admit[best]++
		tau, B, theta = nTau, nB, nTheta
	}
	return admit
}
