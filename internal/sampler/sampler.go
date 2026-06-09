// Package sampler implements the CPU token-sampling pipeline shared by the CLI
// (one-shot / interactive) and the HTTP server. It is deliberately free of any
// cgo / CUDA dependency so it can be unit-tested in isolation and reused by any
// caller holding a host-side logits slice.
//
// Sampling pipeline (order matters, matches llama.cpp):
//
//  1. repeat-penalty  — applied IN PLACE on the RAW logits over pastTokens,
//     BEFORE temperature (positive logit /= rp, negative *= rp).
//  2. top-k           — bounded size-k min-heap selection over the penalized
//     logits (temperature is monotonic so it does not change
//     the ordering; selecting first avoids sorting all 262k).
//  3. softmax         — over the k candidates with temperature folded in, using
//     float64 accumulators for numerical stability.
//  4. top-p (nucleus) — keep the smallest prefix whose cumulative prob >= TopP.
//  5. min-p           — drop candidates below MinP * max_prob.
//  6. multinomial     — weighted random draw from the survivors.
//
// Temperature <= 0 short-circuits to a deterministic greedy argmax.
package sampler

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// Params holds the sampling hyper-parameters. Callers translate their own config
// structs (CLIArgs, GenerationParams) into this neutral form.
type Params struct {
	Temperature   float64
	TopK          int
	TopP          float64
	MinP          float64
	RepeatPenalty float64
}

// cand is a (token id, logit) pair used during top-k selection.
type cand struct {
	id int32
	lg float32
}

// Argmax returns the index of the maximum logit. Pure Go (no cgo) so this
// package never pulls in the CUDA static lib. Returns 0 for empty input.
func Argmax(logits []float32) int {
	best := 0
	var bestV float32
	first := true
	for i, v := range logits {
		if first || v > bestV {
			bestV = v
			best = i
			first = false
		}
	}
	return best
}

// topKCandidates returns the k highest-logit candidates sorted descending, using
// a bounded size-k min-heap — a single O(V) pass instead of sorting all 262k
// logits (a full sort + 262k-element allocation costs ~30 ms per token). k<=0 or
// k>=len falls back to a full sort (top-k disabled → top-p/min-p need full order).
func topKCandidates(logits []float32, k int) []cand {
	n := len(logits)
	if k <= 0 || k >= n {
		cands := make([]cand, n)
		for i := range logits {
			cands[i] = cand{int32(i), logits[i]}
		}
		sort.Slice(cands, func(a, b int) bool { return cands[a].lg > cands[b].lg })
		return cands
	}
	h := make([]cand, 0, k) // min-heap keyed by lg: h[0] is the smallest kept
	for i := 0; i < n; i++ {
		lg := logits[i]
		if len(h) < k {
			h = append(h, cand{int32(i), lg})
			for j := len(h) - 1; j > 0; { // sift up
				p := (j - 1) / 2
				if h[p].lg <= h[j].lg {
					break
				}
				h[p], h[j] = h[j], h[p]
				j = p
			}
		} else if lg > h[0].lg {
			h[0] = cand{int32(i), lg}
			for j := 0; ; { // sift down
				l, r, small := 2*j+1, 2*j+2, j
				if l < k && h[l].lg < h[small].lg {
					small = l
				}
				if r < k && h[r].lg < h[small].lg {
					small = r
				}
				if small == j {
					break
				}
				h[j], h[small] = h[small], h[j]
				j = small
			}
		}
	}
	sort.Slice(h, func(a, b int) bool { return h[a].lg > h[b].lg })
	return h
}

// Sample draws the next token from logits following the package pipeline:
// repeat-penalty → top-k → softmax(temperature) → top-p → min-p → multinomial.
// Temperature <= 0 forces a deterministic greedy argmax.
//
// NOTE: the repeat penalty mutates logits IN PLACE. Callers regenerate logits
// every decode step, so no defensive copy is made (matching the original
// implementations). pastTokens may be nil/empty to disable the penalty.
func Sample(logits []float32, p Params, rng *rand.Rand, pastTokens []int32) (int32, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("sampler: empty logits")
	}
	if p.Temperature <= 0 {
		return int32(Argmax(logits)), nil
	}

	// 1. Repeat penalty on the RAW logits, BEFORE temperature (llama.cpp order).
	//    A positive logit is divided by the penalty (pushed toward zero); a
	//    negative logit is multiplied (pushed further negative). Both reduce the
	//    token's probability.
	if p.RepeatPenalty != 1.0 && len(pastTokens) > 0 {
		rp := float32(p.RepeatPenalty)
		for _, id := range pastTokens {
			if id < 0 || int(id) >= len(logits) {
				continue
			}
			if logits[id] > 0 {
				logits[id] /= rp
			} else {
				logits[id] *= rp
			}
		}
	}

	// 2. Candidate selection. Temperature is monotonic so it does not change the
	//    ordering — select the top-k on the raw (penalized) logits FIRST, then
	//    apply temperature/softmax to just those k.
	cands := topKCandidates(logits, p.TopK)

	// 3. Softmax over the (sorted) candidates with temperature folded in.
	//    float64 accumulators keep this stable for extreme logits.
	invT := 1.0 / p.Temperature
	maxLg := float64(cands[0].lg)
	var sum float64
	probs := make([]float64, len(cands))
	for i, c := range cands {
		v := math.Exp((float64(c.lg) - maxLg) * invT)
		probs[i] = v
		sum += v
	}
	if sum <= 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
		// Degenerate distribution (should not happen with the max-shift, but be
		// safe against pathological inputs): fall back to greedy.
		return cands[0].id, nil
	}
	for i := range probs {
		probs[i] /= sum
	}

	// 4. Top-p (nucleus): keep the smallest prefix whose cumulative prob >= TopP.
	if p.TopP > 0 && p.TopP < 1.0 {
		var cum float64
		cut := len(probs)
		for i := range probs {
			cum += probs[i]
			if cum >= p.TopP {
				cut = i + 1
				break
			}
		}
		cands = cands[:cut]
		probs = probs[:cut]
	}

	// 5. Min-p: drop candidates below MinP * max_prob (probs[0] is the max).
	if p.MinP > 0 {
		thresh := p.MinP * probs[0]
		keep := 0
		for i := range probs {
			if probs[i] >= thresh {
				keep = i + 1
			} else {
				break
			}
		}
		if keep > 0 {
			cands = cands[:keep]
			probs = probs[:keep]
		}
	}

	// 6. Renormalize and draw.
	var z float64
	for _, v := range probs {
		z += v
	}
	r := rng.Float64() * z
	var acc float64
	for i, v := range probs {
		acc += v
		if r <= acc {
			return cands[i].id, nil
		}
	}
	return cands[len(cands)-1].id, nil
}
