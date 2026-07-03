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
	d, _ := promptLookupDraftConf(hist, maxD, minNG, maxNG)
	return d
}

// promptLookupDraftConf is promptLookupDraft plus a per-position CONFIDENCE estimate for
// every drafted token, consumed by the confidence-scheduled verifier (DSpark Algorithm 1).
// conf[j] ∈ (0,1] is the consensus fraction at draft position j: of the earlier suffix
// occurrences scanned, the fraction that agree on the proposed token. A unanimous (or
// single-occurrence) match scores 1.0; a bare strict-majority scores ~0.5. It is the
// training-free, model-agnostic stand-in for the drafter's accept probability c_{r,j} that
// the paper's learned drafter would emit. len(conf) == len(draft). The scheduler turns these
// into per-request survival a_{r,j}=∏_{i≤j} conf[i] and ranks draft positions across ALL
// requests by survival to tailor each request's verify length to the current load.
func promptLookupDraftConf(hist []int32, maxD, minNG, maxNG int) ([]int32, []float32) {
	if maxD <= 0 {
		return nil, nil
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
		var conf []float32
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
			conf = append(conf, float32(agree)/float32(nocc))
		}
		if len(draft) > 0 {
			return draft, conf
		}
	}
	return nil, nil
}

// promptLookupDraftInConf is promptLookupDraftConf searching an EXTERNAL corpus: the last
// ngram of hist is matched against `corpus` (the server-global ring of recently finished
// sequences' tokens) and the corpus continuations are proposed. This is the cross-request
// half of suffix decoding (Arctic-Inference-style): agentic loops re-emit each other's
// prompts and outputs across requests, which a per-sequence history can never see. Same
// consensus gating; lossless regardless (the engine verify commits only correct tokens).
func promptLookupDraftInConf(corpus, hist []int32, maxD, minNG, maxNG int) ([]int32, []float32) {
	if maxD <= 0 || len(corpus) == 0 {
		return nil, nil
	}
	n := len(hist)
	cn := len(corpus)
	const maxOcc = 16
	var occ [maxOcc]int
	for ng := maxNG; ng >= minNG; ng-- {
		if n < ng || cn < ng+1 {
			continue
		}
		suf := hist[n-ng:]
		nocc := 0
		for i := cn - ng - 1; i >= 0 && nocc < maxOcc; i-- {
			match := true
			for j := 0; j < ng; j++ {
				if corpus[i+j] != suf[j] {
					match = false
					break
				}
			}
			if match {
				occ[nocc] = i + ng
				nocc++
			}
		}
		if nocc == 0 {
			continue
		}
		thresh := nocc/2 + 1
		cap := maxD
		if nocc <= 2 && cap > ng {
			cap = ng
		}
		var draft []int32
		var conf []float32
		for dd := 0; dd < cap; dd++ {
			p0 := occ[0] + dd
			if p0 >= cn {
				break
			}
			cand := corpus[p0]
			agree := 0
			for k := 0; k < nocc; k++ {
				if p := occ[k] + dd; p < cn && corpus[p] == cand {
					agree++
				}
			}
			if agree < thresh {
				break
			}
			draft = append(draft, cand)
			conf = append(conf, float32(agree)/float32(nocc))
		}
		if len(draft) > 0 {
			return draft, conf
		}
	}
	return nil, nil
}
