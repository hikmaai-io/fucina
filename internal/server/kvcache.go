package server

import (
	"log"
	"sync"

	"github.com/mauromedda/gem4d/internal/engine/cuda"
)

// KVCache implements prefix-reuse on top of the engine's append-only KV cache.
//
// Motivation: llama.cpp's server re-prefills the entire prompt on every request,
// which is wasteful for multi-turn chat where each turn's prompt is the previous
// conversation plus a new message. vLLM avoids this with automatic prefix
// caching. This manager brings the same idea to gem4d.
//
// Model: the engine holds ONE physical KV cache for ONE logical sequence. We
// track the exact token sequence currently materialized in that cache
// (cachedTokens). When a new request arrives we:
//
//  1. Compute the longest common prefix (LCP) between cachedTokens and the new
//     prompt.
//  2. Rewind the engine's KV cache to the LCP length (cheap; no recompute).
//  3. Prefill only the divergent suffix.
//
// The first LCP tokens are a genuine cache hit and are never recomputed.
//
// Because there is a single physical cache, concurrent requests must be
// serialized — the manager holds a mutex for the whole prefill+generate span.
// This matches the engine's single-slot design (--n-slots 1) and keeps the KV
// state consistent. Multi-slot paged attention is future work; the bookkeeping
// here is structured so it can be extended to multiple sequences later.
type KVCache struct {
	engine *cuda.Engine

	mu           sync.Mutex
	cachedTokens []int32 // exact tokens currently in the engine KV cache

	// stats
	hits      int
	misses    int
	hitTokens int64
	reqTokens int64
}

// NewKVCache creates a prefix-reuse manager bound to an engine.
func NewKVCache(engine *cuda.Engine) *KVCache {
	return &KVCache{engine: engine}
}

// PrefillResult reports what happened during a cache-aware prefill.
type PrefillResult struct {
	PromptTokens int       // total prompt length
	ReusedTokens int       // prefix tokens served from cache (no recompute)
	NewTokens    int       // suffix tokens that were actually prefilled
	Logits       []float32 // logits for the last prompt token
}

// Lock acquires exclusive access to the single physical KV cache for the full
// duration of a request (prefill + generation). Callers must pair it with
// Unlock, typically via defer.
func (c *KVCache) Lock()   { c.mu.Lock() }
func (c *KVCache) Unlock() { c.mu.Unlock() }

// Prefill prepares the engine KV cache to hold exactly `prompt`, reusing the
// longest cached prefix and computing only the divergent suffix. The caller
// must already hold Lock().
//
// It returns the logits for the final prompt token, ready for sampling.
func (c *KVCache) Prefill(prompt []int32) (*PrefillResult, error) {
	res := &PrefillResult{PromptTokens: len(prompt)}
	if len(prompt) == 0 {
		return res, nil
	}

	lcp := longestCommonPrefix(c.cachedTokens, prompt)

	// Never reuse the entire prompt: we must run at least the final token
	// through the model to obtain its logits for sampling. Cap the reusable
	// prefix at len(prompt)-1.
	if lcp >= len(prompt) {
		lcp = len(prompt) - 1
	}

	// Try to rewind the engine to the shared prefix. If the engine cannot
	// safely rewind (e.g. sliding window wrapped), fall back to a full reset.
	engineTokens := c.engine.NTokens()
	if lcp == 0 {
		c.engine.Reset()
		c.cachedTokens = c.cachedTokens[:0]
	} else if lcp < engineTokens {
		if !c.engine.Rewind(lcp) {
			log.Printf("gem4d: kvcache: rewind to %d unsafe, full reset", lcp)
			c.engine.Reset()
			c.cachedTokens = c.cachedTokens[:0]
			lcp = 0
		} else {
			c.cachedTokens = c.cachedTokens[:lcp]
		}
	}
	// If lcp == engineTokens == len(cachedTokens), the cache already holds the
	// shared prefix exactly; nothing to rewind.

	suffix := prompt[lcp:]
	res.ReusedTokens = lcp
	res.NewTokens = len(suffix)

	logits, err := c.engine.Prefill(suffix)
	if err != nil {
		// On failure, the engine KV state is undefined: reset to be safe.
		c.engine.Reset()
		c.cachedTokens = c.cachedTokens[:0]
		return nil, err
	}

	c.cachedTokens = append(c.cachedTokens, suffix...)
	res.Logits = logits

	// stats
	if lcp > 0 {
		c.hits++
	} else {
		c.misses++
	}
	c.hitTokens += int64(lcp)
	c.reqTokens += int64(len(prompt))

	return res, nil
}

// CurrentTokens returns the token sequence currently held in the engine KV
// cache (prompt + tokens generated so far this request). The caller must hold
// Lock(); the returned slice must not be retained past Unlock.
func (c *KVCache) CurrentTokens() []int32 {
	return c.cachedTokens
}

// AppendDecoded records a token that was produced by Decode and therefore is
// now present in the engine KV cache. Keeping cachedTokens in sync with the
// engine is what lets the NEXT request reuse this generation as a prefix
// (important for multi-turn chat where the assistant reply becomes context).
// The caller must hold Lock().
func (c *KVCache) AppendDecoded(token int32) {
	c.cachedTokens = append(c.cachedTokens, token)
}

// Stats returns cumulative prefix-cache statistics.
func (c *KVCache) Stats() (hits, misses int, hitRate float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	hits, misses = c.hits, c.misses
	if c.reqTokens > 0 {
		hitRate = float64(c.hitTokens) / float64(c.reqTokens)
	}
	return
}

// DetailedStats returns the raw prefix-cache counters for /metrics.
func (c *KVCache) DetailedStats() (hits, misses int, reusedTokens, reqTokens int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.hits, c.misses, c.hitTokens, c.reqTokens
}

// longestCommonPrefix returns the length of the longest shared prefix of a, b.
func longestCommonPrefix(a, b []int32) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}
