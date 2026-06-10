package server

import (
	"errors"
	"testing"
)

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
// real KVCache reuse logic without a GPU. It implements the kvEngine interface.
type fakeEngine struct {
	tokens     []int32 // tokens currently in the "KV cache"
	window     int     // sliding window size (for rewind-safety simulation)
	prefillLog [][]int32

	// failNext, when set, makes the next Prefill return an error (and clears).
	failNext bool
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
	if f.failNext {
		f.failNext = false
		return nil, errors.New("fakeEngine: forced prefill failure")
	}
	f.prefillLog = append(f.prefillLog, append([]int32(nil), suffix...))
	f.tokens = append(f.tokens, suffix...)
	// return a dummy logits slice
	return make([]float32, 4), nil
}

// prefillReal drives the REAL KVCache.Prefill with proper locking.
func prefillReal(t *testing.T, kv *KVCache, prompt []int32) *PrefillResult {
	t.Helper()
	kv.Lock()
	defer kv.Unlock()
	res, err := kv.Prefill(prompt)
	if err != nil {
		t.Fatalf("Prefill(%v) error: %v", prompt, err)
	}
	return res
}

func TestPrefixReuseMultiTurn(t *testing.T) {
	f := &fakeEngine{window: 1024}
	kv := NewKVCache(f)

	// Turn 1: fresh prompt.
	p1 := []int32{10, 11, 12, 13}
	res := prefillReal(t, kv, p1)
	if res.ReusedTokens != 0 || res.NewTokens != 4 {
		t.Fatalf("turn1: reused=%d new=%d want 0,4", res.ReusedTokens, res.NewTokens)
	}
	if res.Logits == nil {
		t.Fatal("turn1: logits nil")
	}

	// Simulate generation appending two reply tokens to the cache. AppendDecoded
	// keeps cachedTokens in sync with the engine.
	kv.Lock()
	kv.AppendDecoded(20)
	kv.AppendDecoded(21)
	kv.Unlock()
	f.tokens = append(f.tokens, 20, 21)

	// Turn 2: prompt = full turn1 context + reply + new user message.
	p2 := []int32{10, 11, 12, 13, 20, 21, 30, 31}
	res = prefillReal(t, kv, p2)
	if res.ReusedTokens != 6 || res.NewTokens != 2 {
		t.Fatalf("turn2: reused=%d new=%d want 6,2", res.ReusedTokens, res.NewTokens)
	}
	if res.Logits == nil {
		t.Fatal("turn2: logits nil")
	}
	// Only the 2 new tokens should have been prefilled.
	last := f.prefillLog[len(f.prefillLog)-1]
	if len(last) != 2 || last[0] != 30 || last[1] != 31 {
		t.Fatalf("turn2 prefill suffix = %v, want [30 31]", last)
	}
}

func TestPrefixDivergence(t *testing.T) {
	f := &fakeEngine{window: 1024}
	kv := NewKVCache(f)

	prefillReal(t, kv, []int32{1, 2, 3, 4, 5})

	// New prompt shares prefix [1 2 3] then diverges.
	res := prefillReal(t, kv, []int32{1, 2, 3, 9, 9})
	if res.ReusedTokens != 3 || res.NewTokens != 2 {
		t.Fatalf("divergence: reused=%d new=%d want 3,2", res.ReusedTokens, res.NewTokens)
	}
	if got := f.NTokens(); got != 5 {
		t.Fatalf("cache len=%d want 5", got)
	}
	// CurrentTokens must now equal the new prompt exactly.
	kv.Lock()
	cached := kv.CurrentTokens()
	kv.Unlock()
	want := []int32{1, 2, 3, 9, 9}
	if len(cached) != len(want) {
		t.Fatalf("cached=%v want %v", cached, want)
	}
	for i := range want {
		if cached[i] != want[i] {
			t.Fatalf("cached=%v want %v", cached, want)
		}
	}
}

func TestIdenticalPromptKeepsLastTokenFresh(t *testing.T) {
	f := &fakeEngine{window: 1024}
	kv := NewKVCache(f)
	prefillReal(t, kv, []int32{1, 2, 3})

	// Identical prompt again: must re-run the final token to get fresh logits.
	res := prefillReal(t, kv, []int32{1, 2, 3})
	if res.ReusedTokens != 2 || res.NewTokens != 1 {
		t.Fatalf("identical: reused=%d new=%d want 2,1", res.ReusedTokens, res.NewTokens)
	}
	if res.Logits == nil {
		t.Fatal("identical: logits nil")
	}
}

func TestRewindUnsafeFallsBackToReset(t *testing.T) {
	f := &fakeEngine{window: 4} // tiny window to force wrap
	kv := NewKVCache(f)
	prefillReal(t, kv, []int32{1, 2, 3, 4, 5, 6}) // len 6 > window 4 => wrapped

	// Shares prefix [1 2 3] but rewind is unsafe -> full reset, reused=0.
	res := prefillReal(t, kv, []int32{1, 2, 3, 7})
	if res.ReusedTokens != 0 || res.NewTokens != 4 {
		t.Fatalf("unsafe rewind: reused=%d new=%d want 0,4", res.ReusedTokens, res.NewTokens)
	}
}

func TestPrefillErrorResetsState(t *testing.T) {
	f := &fakeEngine{window: 1024}
	kv := NewKVCache(f)

	// Seed the cache with a successful prefill.
	prefillReal(t, kv, []int32{1, 2, 3})

	// Force the next Prefill to fail.
	f.failNext = true
	kv.Lock()
	_, err := kv.Prefill([]int32{1, 2, 3, 4})
	kv.Unlock()
	if err == nil {
		t.Fatal("expected error from forced prefill failure")
	}

	// State must have been reset: engine empty, cachedTokens empty.
	if got := f.NTokens(); got != 0 {
		t.Fatalf("engine tokens=%d want 0 after error reset", got)
	}
	kv.Lock()
	cached := kv.CurrentTokens()
	kv.Unlock()
	if len(cached) != 0 {
		t.Fatalf("cachedTokens=%v want empty after error reset", cached)
	}

	// Next Prefill must work from scratch (no stale prefix reuse).
	res := prefillReal(t, kv, []int32{1, 2, 3, 4})
	if res.ReusedTokens != 0 || res.NewTokens != 4 {
		t.Fatalf("post-error: reused=%d new=%d want 0,4", res.ReusedTokens, res.NewTokens)
	}
}

func TestAppendDecodedEnablesNextTurnReuse(t *testing.T) {
	f := &fakeEngine{window: 1024}
	kv := NewKVCache(f)

	// Prefill a prompt, then "decode" two tokens, appending them to the engine
	// and to the cache via AppendDecoded.
	prefillReal(t, kv, []int32{5, 6, 7})
	kv.Lock()
	kv.AppendDecoded(8)
	kv.AppendDecoded(9)
	kv.Unlock()
	f.tokens = append(f.tokens, 8, 9)

	// Next turn includes the generated tokens as part of its prompt; they must
	// be reused thanks to AppendDecoded keeping cachedTokens in sync.
	res := prefillReal(t, kv, []int32{5, 6, 7, 8, 9, 100})
	if res.ReusedTokens != 5 || res.NewTokens != 1 {
		t.Fatalf("reuse: reused=%d new=%d want 5,1", res.ReusedTokens, res.NewTokens)
	}
	last := f.prefillLog[len(f.prefillLog)-1]
	if len(last) != 1 || last[0] != 100 {
		t.Fatalf("prefill suffix=%v want [100]", last)
	}
}

// TestExternalEngineResetNotReused covers the defense-in-depth desync check:
// if a caller resets the engine directly (bypassing KVCache.Reset), the stale
// cachedTokens must NOT yield any prefix reuse — Prefill must detect that the
// engine is behind the bookkeeping and run the FULL prompt from a clean
// engine. This is the exact REPL /reset bug: stale bookkeeping matched the
// shared chat-template prefix, only the suffix was prefilled into an EMPTY
// engine at wrong positions, and the model produced word salad.
func TestExternalEngineResetNotReused(t *testing.T) {
	f := &fakeEngine{window: 1024}
	kv := NewKVCache(f)

	prefillReal(t, kv, []int32{1, 2, 3, 4, 5})

	// Reset the engine BEHIND the cache's back (the bug scenario).
	f.Reset()

	// New prompt shares prefix [1 2 3] with the STALE bookkeeping.
	res := prefillReal(t, kv, []int32{1, 2, 3, 9})
	if res.ReusedTokens != 0 || res.NewTokens != 4 {
		t.Fatalf("desync: reused=%d new=%d want 0,4", res.ReusedTokens, res.NewTokens)
	}
	// The FULL prompt must have been prefilled, not just the suffix.
	last := f.prefillLog[len(f.prefillLog)-1]
	if len(last) != 4 || last[0] != 1 || last[1] != 2 || last[2] != 3 || last[3] != 9 {
		t.Fatalf("desync prefill = %v, want [1 2 3 9]", last)
	}
	// Bookkeeping and engine must agree again.
	kv.Lock()
	cached := kv.CurrentTokens()
	kv.Unlock()
	if len(cached) != 4 || f.NTokens() != 4 {
		t.Fatalf("post-heal: cached=%v engineTokens=%d want 4 tokens each", cached, f.NTokens())
	}
}

// TestCachedAheadOfEngineKeepsCommittedPrefix pins the heal for the COMMON
// direction of skew: cachedTokens recorded a token (via AppendDecoded) that
// was never committed to the engine. The server's generation loops used to
// produce exactly this on every stop-sequence / streaming tool-call request
// (record-then-break-without-decode); a decode error after recording can
// still produce it. The first n engine tokens remain a faithful prefix of the
// bookkeeping, so Prefill must truncate cachedTokens to n and KEEP the
// n-token hit — dropping all bookkeeping here would full-re-prefill the
// entire conversation on every agentic round trip.
func TestCachedAheadOfEngineKeepsCommittedPrefix(t *testing.T) {
	f := &fakeEngine{window: 1024}
	kv := NewKVCache(f)

	prefillReal(t, kv, []int32{1, 2, 3, 4})

	// Record a token in the bookkeeping WITHOUT mirroring it into the fake
	// engine: the engine never decoded it (cachedTokens = engine + 1).
	kv.Lock()
	kv.AppendDecoded(50)
	kv.Unlock()

	// Next turn extends the committed context (the orphaned token 50 reappears
	// in the prompt text, as an assistant reply does, but it is NOT in the
	// engine KV). Exactly the 4 committed tokens must be reused; 50 and 60 are
	// prefilled fresh.
	res := prefillReal(t, kv, []int32{1, 2, 3, 4, 50, 60})
	if res.ReusedTokens != 4 || res.NewTokens != 2 {
		t.Fatalf("cached-ahead: reused=%d new=%d want 4,2", res.ReusedTokens, res.NewTokens)
	}
	last := f.prefillLog[len(f.prefillLog)-1]
	if len(last) != 2 || last[0] != 50 || last[1] != 60 {
		t.Fatalf("cached-ahead prefill suffix=%v want [50 60]", last)
	}
	// Bookkeeping and engine must agree again, holding the full new prompt.
	if got := f.NTokens(); got != 6 {
		t.Fatalf("engine tokens=%d want 6", got)
	}
	kv.Lock()
	cached := kv.CurrentTokens()
	kv.Unlock()
	if len(cached) != 6 {
		t.Fatalf("cached=%v want the 6-token prompt", cached)
	}
}

// TestResetClearsBothSides exercises KVCache.Reset: engine KV and cachedTokens
// must be cleared together so the next Prefill is a clean full miss. This is
// what the REPL /reset handler relies on. Cumulative stats must survive the
// reset (they describe the process lifetime, not one conversation).
func TestResetClearsBothSides(t *testing.T) {
	f := &fakeEngine{window: 1024}
	kv := NewKVCache(f)

	prefillReal(t, kv, []int32{1, 2, 3})
	kv.Lock()
	kv.AppendDecoded(4)
	kv.Unlock()
	f.tokens = append(f.tokens, 4)

	hitsBefore, missesBefore, _ := kv.Stats()

	kv.Lock()
	kv.Reset()
	kv.Unlock()

	if got := f.NTokens(); got != 0 {
		t.Fatalf("engine tokens=%d want 0 after Reset", got)
	}
	kv.Lock()
	cached := kv.CurrentTokens()
	kv.Unlock()
	if len(cached) != 0 {
		t.Fatalf("cachedTokens=%v want empty after Reset", cached)
	}

	// Next Prefill of a prompt sharing the old prefix must be a FULL miss.
	res := prefillReal(t, kv, []int32{1, 2, 3, 4, 5})
	if res.ReusedTokens != 0 || res.NewTokens != 5 {
		t.Fatalf("post-reset: reused=%d new=%d want 0,5", res.ReusedTokens, res.NewTokens)
	}
	last := f.prefillLog[len(f.prefillLog)-1]
	if len(last) != 5 {
		t.Fatalf("post-reset prefill = %v, want the full 5-token prompt", last)
	}

	// Stats stay sane: counters are cumulative; the post-reset prefill
	// registers as one more miss and no new hit.
	hits, misses, _ := kv.Stats()
	if hits != hitsBefore || misses != missesBefore+1 {
		t.Fatalf("stats after reset: hits=%d misses=%d want %d,%d",
			hits, misses, hitsBefore, missesBefore+1)
	}
}

// TestEngineAheadOfCacheStillReusesPrefix documents why the desync check is
// one-directional: speculative decoding can COMMIT accepted draft tokens to
// the engine KV past the emitted stop token (see the `committed` sync in
// server.generateResponse), leaving the engine AHEAD of cachedTokens. The
// cached prefix is still genuine, so Prefill must keep reusing it — rewinding
// the extra token away — rather than full-resetting and losing the hit.
func TestEngineAheadOfCacheStillReusesPrefix(t *testing.T) {
	f := &fakeEngine{window: 1024}
	kv := NewKVCache(f)

	prefillReal(t, kv, []int32{1, 2, 3})
	// Engine committed an extra token the cache never saw.
	f.tokens = append(f.tokens, 99)

	res := prefillReal(t, kv, []int32{1, 2, 3, 4})
	if res.ReusedTokens != 3 || res.NewTokens != 1 {
		t.Fatalf("engine-ahead: reused=%d new=%d want 3,1", res.ReusedTokens, res.NewTokens)
	}
	last := f.prefillLog[len(f.prefillLog)-1]
	if len(last) != 1 || last[0] != 4 {
		t.Fatalf("engine-ahead prefill suffix=%v want [4]", last)
	}
	if got := f.NTokens(); got != 4 {
		t.Fatalf("engine tokens=%d want 4 (extra token rewound away)", got)
	}
}

func TestStatsCounters(t *testing.T) {
	f := &fakeEngine{window: 1024}
	kv := NewKVCache(f)

	// Miss: fresh prompt, no reuse.
	prefillReal(t, kv, []int32{1, 2, 3})
	// Hit: shares prefix [1 2] (last token re-run for fresh logits).
	prefillReal(t, kv, []int32{1, 2, 3})

	hits, misses, hitRate := kv.Stats()
	if hits != 1 || misses != 1 {
		t.Fatalf("Stats: hits=%d misses=%d want 1,1", hits, misses)
	}
	if hitRate <= 0 || hitRate >= 1 {
		t.Fatalf("Stats: hitRate=%v want in (0,1)", hitRate)
	}

	dHits, dMisses, reused, reqTokens := kv.DetailedStats()
	if dHits != 1 || dMisses != 1 {
		t.Fatalf("DetailedStats: hits=%d misses=%d want 1,1", dHits, dMisses)
	}
	// First prompt: 3 tokens, 0 reused. Second: 3 tokens, 2 reused.
	if reqTokens != 6 {
		t.Fatalf("DetailedStats: reqTokens=%d want 6", reqTokens)
	}
	if reused != 2 {
		t.Fatalf("DetailedStats: reusedTokens=%d want 2", reused)
	}
}
