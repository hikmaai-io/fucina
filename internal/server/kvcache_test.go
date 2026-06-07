package server

import "testing"

func TestLongestCommonPrefix(t *testing.T) {
	cases := []struct {
		a, b []int32
		want int
	}{
		{nil, nil, 0},
		{[]int32{1, 2, 3}, nil, 0},
		{[]int32{1, 2, 3}, []int32{1, 2, 3}, 3},
		{[]int32{1, 2, 3}, []int32{1, 2, 4}, 2},
		{[]int32{1, 2, 3}, []int32{1, 2, 3, 4, 5}, 3},
		{[]int32{1, 2, 3, 4, 5}, []int32{1, 2, 3}, 3},
		{[]int32{9, 9}, []int32{1, 2}, 0},
	}
	for i, c := range cases {
		if got := longestCommonPrefix(c.a, c.b); got != c.want {
			t.Errorf("case %d: lcp=%d want %d", i, got, c.want)
		}
	}
}

// fakeEngine mimics the engine's append-only KV cache cursor so we can test the
// KVCache reuse logic without a GPU.
type fakeEngine struct {
	tokens     []int32 // tokens currently in the "KV cache"
	window     int     // sliding window size (for rewind-safety simulation)
	prefillLog [][]int32
}

func (f *fakeEngine) NTokens() int { return len(f.tokens) }

func (f *fakeEngine) Reset() { f.tokens = f.tokens[:0] }

func (f *fakeEngine) Rewind(nKeep int) bool {
	if len(f.tokens) > f.window {
		return false // wrapped: unsafe, matches engine semantics
	}
	if nKeep < 0 || nKeep > len(f.tokens) {
		return false
	}
	f.tokens = f.tokens[:nKeep]
	return true
}

func (f *fakeEngine) Prefill(suffix []int32) ([]float32, error) {
	f.prefillLog = append(f.prefillLog, append([]int32(nil), suffix...))
	f.tokens = append(f.tokens, suffix...)
	// return a dummy logits slice
	return make([]float32, 4), nil
}

// kvLike is the subset of *cuda.Engine that KVCache uses; we re-implement the
// reuse algorithm against fakeEngine to validate the decision logic directly.
func simulate(f *fakeEngine, cached *[]int32, prompt []int32) (reused, fresh int) {
	lcp := longestCommonPrefix(*cached, prompt)
	if lcp >= len(prompt) {
		lcp = len(prompt) - 1
	}
	engineTokens := f.NTokens()
	if lcp == 0 {
		f.Reset()
		*cached = (*cached)[:0]
	} else if lcp < engineTokens {
		if !f.Rewind(lcp) {
			f.Reset()
			*cached = (*cached)[:0]
			lcp = 0
		} else {
			*cached = (*cached)[:lcp]
		}
	}
	suffix := prompt[lcp:]
	f.Prefill(suffix)
	*cached = append(*cached, suffix...)
	return lcp, len(suffix)
}

func TestPrefixReuseMultiTurn(t *testing.T) {
	f := &fakeEngine{window: 1024}
	var cached []int32

	// Turn 1: fresh prompt.
	p1 := []int32{10, 11, 12, 13}
	reused, fresh := simulate(f, &cached, p1)
	if reused != 0 || fresh != 4 {
		t.Fatalf("turn1: reused=%d fresh=%d want 0,4", reused, fresh)
	}

	// Simulate generation appending two reply tokens to the cache.
	cached = append(cached, 20, 21)
	f.tokens = append(f.tokens, 20, 21)

	// Turn 2: prompt = full turn1 context + reply + new user message.
	p2 := []int32{10, 11, 12, 13, 20, 21, 30, 31}
	reused, fresh = simulate(f, &cached, p2)
	if reused != 6 || fresh != 2 {
		t.Fatalf("turn2: reused=%d fresh=%d want 6,2", reused, fresh)
	}
	// Only the 2 new tokens should have been prefilled.
	last := f.prefillLog[len(f.prefillLog)-1]
	if len(last) != 2 || last[0] != 30 || last[1] != 31 {
		t.Fatalf("turn2 prefill suffix = %v, want [30 31]", last)
	}
}

func TestPrefixDivergence(t *testing.T) {
	f := &fakeEngine{window: 1024}
	var cached []int32

	simulate(f, &cached, []int32{1, 2, 3, 4, 5})

	// New prompt shares prefix [1 2 3] then diverges.
	reused, fresh := simulate(f, &cached, []int32{1, 2, 3, 9, 9})
	if reused != 3 || fresh != 2 {
		t.Fatalf("divergence: reused=%d fresh=%d want 3,2", reused, fresh)
	}
	if got := f.NTokens(); got != 5 {
		t.Fatalf("cache len=%d want 5", got)
	}
	// cached must now equal the new prompt exactly.
	want := []int32{1, 2, 3, 9, 9}
	for i := range want {
		if cached[i] != want[i] {
			t.Fatalf("cached=%v want %v", cached, want)
		}
	}
}

func TestIdenticalPromptKeepsLastTokenFresh(t *testing.T) {
	f := &fakeEngine{window: 1024}
	var cached []int32
	simulate(f, &cached, []int32{1, 2, 3})

	// Identical prompt again: must re-run the final token to get fresh logits.
	reused, fresh := simulate(f, &cached, []int32{1, 2, 3})
	if reused != 2 || fresh != 1 {
		t.Fatalf("identical: reused=%d fresh=%d want 2,1", reused, fresh)
	}
}

func TestRewindUnsafeFallsBackToReset(t *testing.T) {
	f := &fakeEngine{window: 4} // tiny window to force wrap
	var cached []int32
	simulate(f, &cached, []int32{1, 2, 3, 4, 5, 6}) // len 6 > window 4 => wrapped

	// Shares prefix [1 2 3] but rewind is unsafe -> full reset, reused=0.
	reused, fresh := simulate(f, &cached, []int32{1, 2, 3, 7})
	if reused != 0 || fresh != 4 {
		t.Fatalf("unsafe rewind: reused=%d fresh=%d want 0,4", reused, fresh)
	}
}
