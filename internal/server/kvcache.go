package server

import (
	"log"
	"sync"
)

// kvEngine is the consumer-side view of the inference engine that KVCache
// depends on. It is declared here (where it is used) rather than in the cuda
// package, per Go best practice. *cuda.Engine satisfies this interface.
type kvEngine interface {
	Prefill(tokens []int32) ([]float32, error)
	NTokens() int
	Reset()
	Rewind(nKeep int) bool
}

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
//
// Lock ordering: KVCache.mu MUST be acquired BEFORE Engine.mu. Engine methods
// lock Engine.mu internally, so a KVCache method that calls the engine while
// holding KVCache.mu observes the order KVCache.mu → Engine.mu. To keep this
// ordering total and deadlock-free, never call KVCache methods from inside
// engine code (the engine must not reach back up into the cache).
type KVCache struct {
	engine kvEngine

	mu           sync.Mutex
	cachedTokens []int32 // exact tokens currently in the engine KV cache

	// stats
	hits      int
	misses    int
	hitTokens int64
	reqTokens int64
}

// NewKVCache creates a prefix-reuse manager bound to an engine.
func NewKVCache(engine kvEngine) *KVCache {
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

// Reset clears the engine KV cache AND cachedTokens together, keeping the two
// in lockstep. This is the ONLY correct way to start a fresh conversation on a
// live KVCache: calling engine.Reset() directly leaves cachedTokens claiming
// tokens the engine no longer holds, and the next Prefill would then "reuse" a
// prefix that does not exist in the KV cache — the suffix gets prefilled at
// wrong positions and attention over the missing prefix produces garbage
// (observed as word-salad replies after the REPL's /reset). Cumulative stats
// are intentionally preserved: they describe the process lifetime, not one
// conversation. The caller must hold Lock().
func (c *KVCache) Reset() {
	c.engine.Reset()
	c.cachedTokens = c.cachedTokens[:0]
}

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

	// Defense in depth: cachedTokens must only ever claim tokens that actually
	// exist in the engine KV cache. The bookkeeping can end up AHEAD of the
	// engine in two ways: a caller reset or rewound the engine directly
	// (bypassing KVCache.Reset/this Prefill), or a token was recorded via
	// AppendDecoded but its Decode never happened or failed (the generation
	// loops record only after a successful commit precisely to avoid this; see
	// AppendDecoded). Any prefix "reuse" computed from the stale tail would
	// attend over tokens that are not there — the suffix gets prefilled at
	// wrong positions and the model emits garbage.
	//
	// Heal by truncating the bookkeeping to the engine's length, NOT by
	// dropping it entirely. The engine KV is append-only and its first n
	// tokens were always placed from cachedTokens' own record (Prefill
	// suffixes plus committed AppendDecoded tokens), so cachedTokens[:n] is
	// still a faithful description of the engine contents for every producer
	// of this skew: an external Reset gives n == 0 (truncate == drop, full
	// re-prefill slow path); an external Rewind keeps the surviving prefix;
	// an AppendDecoded overrun keeps everything but the uncommitted tail.
	// Truncation therefore preserves the genuine n-token prefix hit — dropping
	// it all would force a full re-prefill of the entire conversation, the
	// exact multi-second cost at large context that this cache exists to
	// avoid.
	//
	// The OPPOSITE skew — engine holding MORE tokens than cachedTokens — is a
	// normal post-generation state (speculative decoding can commit accepted
	// draft tokens past the emitted stop token; see the `committed` sync in
	// server.generateResponse) and is already healed by the branch chain
	// below: lcp ≤ len(cachedTokens) < engineTokens always takes the
	// rewind-or-reset path. So we deliberately do not warn or truncate for it,
	// preserving prefix reuse across such requests.
	if n := c.engine.NTokens(); len(c.cachedTokens) > n {
		log.Printf("gem4d: kvcache: engine KV (%d tokens) is behind cache bookkeeping (%d); truncating bookkeeping to the engine's length", n, len(c.cachedTokens))
		c.cachedTokens = c.cachedTokens[:n]
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

// CurrentTokens returns a COPY of the token sequence currently held in the
// engine KV cache (prompt + tokens generated so far this request). The caller
// must hold Lock(). A copy is returned so the returned slice is safe to retain
// or read after Unlock without aliasing the cache's internal backing array.
func (c *KVCache) CurrentTokens() []int32 {
	out := make([]int32, len(c.cachedTokens))
	copy(out, c.cachedTokens)
	return out
}

// AppendDecoded records a token that the engine has COMMITTED to its KV cache
// (i.e. a successful Decode/DecodeNoCopy of that token has completed). Keeping
// cachedTokens in sync with the engine is what lets the NEXT request reuse
// this generation as a prefix (important for multi-turn chat where the
// assistant reply becomes context).
//
// Contract: call this only AFTER the commit, never before. The generation
// loops break on tool-call-end / stop-sequence tokens WITHOUT decoding them;
// recording such a token first would leave cachedTokens one ahead of the
// engine on every such request, and the next Prefill would have to heal the
// skew (see the truncation guard there). The caller must hold Lock().
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
