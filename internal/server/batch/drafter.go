package batch

// promptLookupDraft is the model-agnostic, training-free speculative drafter used
// by the continuous-batching scheduler. It mirrors the C engine's prompt_lookup_draft
// (cuda/gemma4_kernels.cu): take the most-recent suffix of hist, find earlier
// occurrences of that suffix in hist, and propose what followed them — gated by a
// strict-majority consensus so a single spurious match can't draft a long bogus run.
//
// It wins on repetitive / structured / quoted output (code, JSON, agentic file and
// diff re-emission, RAG that echoes context) and proposes nothing on novel prose —
// where it costs ~zero (a short suffix scan) and the step falls back to a plain
// batched decode. Returns up to maxD draft tokens (nil when nothing clears the gate).
//
// Params match the C call site: minNG=2, maxNG=draftK. occ is capped so a token that
// recurs everywhere doesn't blow up the inner agreement scan.
func promptLookupDraft(hist []int32, maxD, minNG, maxNG int) []int32 {
	if maxD <= 0 {
		return nil
	}
	n := len(hist)
	const maxOcc = 16
	var occ [maxOcc]int
	for ng := maxNG; ng >= minNG; ng-- {
		if n < ng+1 {
			continue
		}
		suf := hist[n-ng:]
		// Collect up to maxOcc most-recent EARLIER occurrences of the suffix.
		nocc := 0
		for i := n - ng - 1; i >= 0 && nocc < maxOcc; i-- {
			match := true
			for j := 0; j < ng; j++ {
				if hist[i+j] != suf[j] {
					match = false
					break
				}
			}
			if match {
				occ[nocc] = i + ng // position right after the matched context
				nocc++
			}
		}
		if nocc == 0 {
			continue
		}
		// Strict majority: at least one OTHER occurrence (beyond occ[0], which
		// trivially agrees with itself) must corroborate each drafted token.
		thresh := nocc/2 + 1
		// Confidence cap: 1–2 occurrences are trusted only as far ahead as the
		// matched context is long; 3+ agreeing occurrences earn the full budget.
		cap := maxD
		if nocc <= 2 && cap > ng {
			cap = ng
		}
		var draft []int32
		for dd := 0; dd < cap; dd++ {
			p0 := occ[0] + dd // most-recent occurrence's continuation
			if p0 >= n {
				break
			}
			cand := hist[p0]
			agree := 0
			for k := 0; k < nocc; k++ {
				if p := occ[k] + dd; p < n && hist[p] == cand {
					agree++
				}
			}
			if agree < thresh {
				break // consensus broke → stop drafting here
			}
			draft = append(draft, cand)
		}
		if len(draft) > 0 {
			return draft
		}
	}
	return nil
}
