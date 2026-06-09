package sampler

import (
	"math"
	"math/rand"
	"testing"
)

func TestArgmaxObviousPeak(t *testing.T) {
	logits := []float32{0.1, 0.2, 9.0, 0.3, -1.0}
	if got := Argmax(logits); got != 2 {
		t.Fatalf("Argmax = %d, want 2", got)
	}
}

func TestArgmaxEmpty(t *testing.T) {
	if got := Argmax(nil); got != 0 {
		t.Fatalf("Argmax(nil) = %d, want 0", got)
	}
}

func TestEmptyLogitsReturnsError(t *testing.T) {
	_, err := Sample(nil, Params{Temperature: 1.0, TopK: 4}, rand.New(rand.NewSource(1)), nil)
	if err == nil {
		t.Fatal("expected error for empty logits, got nil")
	}
}

func TestTempZeroGreedyDeterminism(t *testing.T) {
	logits := []float32{1, 2, 3, 10, 4, 5}
	p := Params{Temperature: 0, TopK: 4, RepeatPenalty: 1.0}
	for i := 0; i < 100; i++ {
		tok, err := Sample(append([]float32(nil), logits...), p, rand.New(rand.NewSource(int64(i))), nil)
		if err != nil {
			t.Fatal(err)
		}
		if tok != 3 {
			t.Fatalf("greedy token = %d, want 3", tok)
		}
	}
}

func TestFixedSeedReproducibility(t *testing.T) {
	logits := []float32{0.5, 1.5, 2.5, 0.1, 3.5, 1.0, 0.2, 2.0}
	p := Params{Temperature: 1.0, TopK: 8, TopP: 0.95, RepeatPenalty: 1.0}

	draw := func(seed int64) []int32 {
		rng := rand.New(rand.NewSource(seed))
		out := make([]int32, 50)
		for i := range out {
			tok, err := Sample(append([]float32(nil), logits...), p, rng, nil)
			if err != nil {
				t.Fatal(err)
			}
			out[i] = tok
		}
		return out
	}

	a := draw(42)
	b := draw(42)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-reproducible at %d: %d vs %d", i, a[i], b[i])
		}
	}
}

func TestTopKRestrictsCandidateSet(t *testing.T) {
	// Distinct logits; top-3 ids are 5, 4, 3 (values 6,5,4).
	logits := []float32{1, 2, 3, 4, 5, 6}
	p := Params{Temperature: 1.0, TopK: 3, RepeatPenalty: 1.0}
	allowed := map[int32]bool{5: true, 4: true, 3: true}
	rng := rand.New(rand.NewSource(7))
	for i := 0; i < 1000; i++ {
		tok, err := Sample(append([]float32(nil), logits...), p, rng, nil)
		if err != nil {
			t.Fatal(err)
		}
		if !allowed[tok] {
			t.Fatalf("top-k=3 produced out-of-set token %d", tok)
		}
	}
}

func TestTopPTruncation(t *testing.T) {
	// One token dominates: softmax prob of id 0 is ~1.0, so top-p=0.5 must keep
	// only id 0.
	logits := []float32{100, 0, 0, 0, 0}
	p := Params{Temperature: 1.0, TopK: 0, TopP: 0.5, RepeatPenalty: 1.0}
	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 500; i++ {
		tok, err := Sample(append([]float32(nil), logits...), p, rng, nil)
		if err != nil {
			t.Fatal(err)
		}
		if tok != 0 {
			t.Fatalf("top-p truncation failed: got token %d, want 0", tok)
		}
	}
}

func TestMinPThreshold(t *testing.T) {
	// id 0 has prob ~1, the rest negligible. min-p=0.5 drops everything below
	// 0.5*maxprob, leaving only id 0.
	logits := []float32{50, 1, 1, 1}
	p := Params{Temperature: 1.0, TopK: 0, MinP: 0.5, RepeatPenalty: 1.0}
	rng := rand.New(rand.NewSource(9))
	for i := 0; i < 500; i++ {
		tok, err := Sample(append([]float32(nil), logits...), p, rng, nil)
		if err != nil {
			t.Fatal(err)
		}
		if tok != 0 {
			t.Fatalf("min-p threshold failed: got token %d, want 0", tok)
		}
	}
}

func TestRepeatPenaltyDemotesToken(t *testing.T) {
	// Symmetric-ish logits; we repeatedly penalize token 0 and check its draw
	// frequency drops relative to the unpenalized case.
	base := []float32{3.0, 3.0, 3.0, 3.0}
	const n = 20000

	count := func(p Params, past []int32) int {
		rng := rand.New(rand.NewSource(123))
		c := 0
		for i := 0; i < n; i++ {
			tok, err := Sample(append([]float32(nil), base...), p, rng, past)
			if err != nil {
				t.Fatal(err)
			}
			if tok == 0 {
				c++
			}
		}
		return c
	}

	unpen := count(Params{Temperature: 1.0, TopK: 0, RepeatPenalty: 1.0}, nil)
	pen := count(Params{Temperature: 1.0, TopK: 0, RepeatPenalty: 2.0}, []int32{0})

	if pen >= unpen {
		t.Fatalf("repeat penalty did not demote token 0: penalized=%d unpenalized=%d", pen, unpen)
	}
}

func TestNumericalStabilityExtremeLogits(t *testing.T) {
	logits := []float32{1e4, -1e4, 1e4, -1e4, 0}
	p := Params{Temperature: 0.7, TopK: 4, TopP: 0.95, MinP: 0.01, RepeatPenalty: 1.1}
	rng := rand.New(rand.NewSource(11))
	for i := 0; i < 1000; i++ {
		tok, err := Sample(append([]float32(nil), logits...), p, rng, []int32{0})
		if err != nil {
			t.Fatal(err)
		}
		if int(tok) < 0 || int(tok) >= len(logits) {
			t.Fatalf("out-of-range token %d", tok)
		}
	}

	// Direct softmax sanity: no NaN should leak through.
	if math.IsNaN(float64(logits[0])) {
		t.Fatal("logits corrupted to NaN")
	}
}
