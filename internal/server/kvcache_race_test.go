package server

import (
	"sync"
	"testing"
)

// TestKVCacheConcurrentStats exercises the lock discipline: a writer goroutine
// drives the full prefill+decode span under Lock()/Unlock() while a reader
// goroutine concurrently reads Stats()/DetailedStats() (which take the mutex
// internally). Run with -race to detect data races.
func TestKVCacheConcurrentStats(t *testing.T) {
	f := &fakeEngine{window: 1 << 30}
	kv := NewKVCache(f)

	const iters = 200
	var wg sync.WaitGroup
	wg.Add(2)

	// Writer: prefill + append decoded tokens, holding Lock for the span.
	go func() {
		defer wg.Done()
		prompt := []int32{1, 2, 3, 4}
		for i := 0; i < iters; i++ {
			kv.Lock()
			if _, err := kv.Prefill(prompt); err != nil {
				kv.Unlock()
				t.Errorf("Prefill error: %v", err)
				return
			}
			kv.AppendDecoded(int32(i))
			kv.Unlock()
		}
	}()

	// Reader: hammer the stats accessors concurrently.
	go func() {
		defer wg.Done()
		for i := 0; i < iters; i++ {
			_, _, _ = kv.Stats()
			_, _, _, _ = kv.DetailedStats()
		}
	}()

	wg.Wait()
}

// TestCurrentTokensIsCopy asserts that mutating the slice returned by
// CurrentTokens() does not corrupt the KVCache's internal state.
func TestCurrentTokensIsCopy(t *testing.T) {
	f := &fakeEngine{window: 1024}
	kv := NewKVCache(f)

	prefillReal(t, kv, []int32{1, 2, 3, 4})

	kv.Lock()
	snap := kv.CurrentTokens()
	kv.Unlock()

	if len(snap) != 4 {
		t.Fatalf("CurrentTokens len=%d want 4", len(snap))
	}

	// Mutate the returned slice; internal state must be unaffected.
	for i := range snap {
		snap[i] = -1
	}

	kv.Lock()
	again := kv.CurrentTokens()
	kv.Unlock()

	want := []int32{1, 2, 3, 4}
	for i := range want {
		if again[i] != want[i] {
			t.Fatalf("internal tokens corrupted: got %v want %v", again, want)
		}
	}
}
