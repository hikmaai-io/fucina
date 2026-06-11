package server

import (
	"errors"
	"log"
	"sync"
	"sync/atomic"
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

// kvSnapshotter is the OPTIONAL engine capability behind the multi-sequence
// prefix cache (see KVCache.saved). Detected by type assertion so the test
// fakes and any engine without it keep working — the cache then simply never
// snapshots. *cuda.Engine satisfies it.
type kvSnapshotter interface {
	KVStateSize(nTokens int) int
	KVSave(buf []byte, nTokens int) error
	KVRestore(buf []byte, nTokens int) error
}

// savedSeq is one snapshotted conversation: the exact token sequence and the
// engine KV bytes for it. state is sized to the tokens actually used
// (~200 KB/token), not the full context capacity.
type savedSeq struct {
	tokens []int32
	state  []byte
	used   int64 // LRU clock stamp
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

	// Multi-sequence prefix cache (nil snap = disabled). The engine holds ONE
	// live KV sequence; an unrelated request (another client, a /metrics
	// scraper that chats, a second conversation) used to evict a large agent
	// conversation and force a full re-prefill on its next turn — 100+ s at
	// 20k context. Instead, when a request would destroy a large cached
	// prefix, the live sequence is snapshotted to host memory first; when a
	// request matches a snapshot better than the live sequence, the snapshot
	// is restored (a memcpy at unified-memory bandwidth) and only the suffix
	// is prefilled. Snapshots are LRU-evicted to stay within snapBudget bytes.
	snap       kvSnapshotter
	saved      []*savedSeq
	snapBudget int64
	snapBytes  int64 // current total across saved states
	snapClock  int64 // LRU stamp source

	// stats — atomics so /metrics and /health can read them WITHOUT taking mu,
	// which is held for the entire prefill+generate span of an in-flight request
	// (a health endpoint that blocks ~100s behind a long prefill reports "dead"
	// exactly while the server is doing its job). Writers still run under mu.
	hits      atomic.Int64
	misses    atomic.Int64
	hitTokens atomic.Int64
	reqTokens atomic.Int64
}

// Snapshot tuning. swapMarginTokens is the minimum prefix-length advantage a
// snapshot must have over the live sequence to be worth a restore (a restore
// is tens of ms; 128 tokens of prefill is comparable, below that keep it
// simple and prefill). saveMinLossTokens is the minimum number of cached
// tokens a request must be about to destroy before the live sequence is worth
// snapshotting (~200 KB/token of host memory).
const (
	swapMarginTokens  = 128
	saveMinLossTokens = 1024
)

// NewKVCache creates a prefix-reuse manager bound to an engine. When the
// engine supports KV snapshots the multi-sequence cache is enabled with a
// default 16 GiB host budget (use SetSnapshotBudget to tune; 0 disables).
func NewKVCache(engine kvEngine) *KVCache {
	c := &KVCache{engine: engine}
	if s, ok := engine.(kvSnapshotter); ok {
		c.snap = s
		c.snapBudget = 16 << 30
	}
	return c
}

// SetSnapshotBudget bounds the total host memory used for saved KV sequences.
// 0 disables snapshotting entirely. Existing snapshots over the new budget are
// evicted on the next save.
func (c *KVCache) SetSnapshotBudget(bytes int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.snapBudget = bytes
	if bytes == 0 {
		c.saved, c.snapBytes = nil, 0
	}
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

	// Multi-sequence cache: if a snapshot matches this prompt much better
	// than the live sequence, save the live one (if losing it would be
	// expensive) and restore the snapshot; if no snapshot helps but this
	// request is about to destroy a large live prefix, save it first so the
	// owning conversation's next turn can come back cheaply.
	if c.snap != nil && c.snapBudget > 0 {
		lcp = c.maybeSwap(prompt, lcp)
	}

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
		// Cooperative abort (client disconnected mid-prefill): the engine state
		// is CONSISTENT — committed chunks sit at their correct absolute
		// positions and unaccounted writes are never read (write-before-advance
		// invariant of the flat KV). Keep the shared prefix: cachedTokens
		// already holds exactly the lcp prefix and the engine may only be
		// AHEAD, which the next request's rewind heals. Dropping everything
		// here made every ESC cost a full re-prefill of the conversation, so
		// repeated abort+retry never converged. Detected structurally (an
		// error exposing Aborted() true) so this package needs no cgo import.
		var ab interface{ Aborted() bool }
		if errors.As(err, &ab) && ab.Aborted() {
			return nil, err
		}
		// On real failure, the engine KV state is undefined: reset to be safe.
		c.engine.Reset()
		c.cachedTokens = c.cachedTokens[:0]
		return nil, err
	}

	c.cachedTokens = append(c.cachedTokens, suffix...)
	res.Logits = logits

	// stats
	if lcp > 0 {
		c.hits.Add(1)
	} else {
		c.misses.Add(1)
	}
	c.hitTokens.Add(int64(lcp))
	c.reqTokens.Add(int64(len(prompt)))

	return res, nil
}

// maybeSwap implements the multi-sequence policy for one incoming prompt.
// liveLCP is the prompt's match against the live sequence; it returns the
// (possibly improved) LCP to continue with. The caller must hold Lock() and
// have healed the bookkeeping/engine skew (cachedTokens fully materialized).
func (c *KVCache) maybeSwap(prompt []int32, liveLCP int) int {
	// Best snapshot for this prompt.
	best, bestLCP := -1, 0
	for i, s := range c.saved {
		if l := longestCommonPrefix(s.tokens, prompt); l > bestLCP {
			best, bestLCP = i, l
		}
	}

	// Pin the snapshot we intend to restore by removing it from the pool NOW:
	// saveLive below evicts by LRU and could otherwise free exactly this entry
	// (and would invalidate the index either way). The pool also must not keep
	// a stale twin of the soon-to-be-live sequence — generation will diverge
	// it, so the snapshot would hold ~200 KB/token hostage for a prefix the
	// live sequence then serves.
	var pinned *savedSeq
	if best >= 0 && bestLCP > liveLCP+swapMarginTokens {
		pinned = c.saved[best]
		c.saved = append(c.saved[:best], c.saved[best+1:]...)
		c.snapBytes -= int64(len(pinned.state))
	}

	// Save the live sequence when this request is about to destroy a costly
	// prefix: a restore overwrites it entirely; otherwise the normal flow
	// rewinds it to liveLCP (the interloper case when liveLCP is small).
	lost := len(c.cachedTokens) - liveLCP
	if pinned != nil {
		lost = len(c.cachedTokens)
	}
	if lost >= saveMinLossTokens {
		c.saveLive()
	}

	if pinned == nil {
		return liveLCP
	}
	if err := c.snap.KVRestore(pinned.state, len(pinned.tokens)); err != nil {
		// A failed restore may have partially overwritten the live KV: the
		// engine state is undefined, so fall back to a clean full prefill.
		log.Printf("gem4d: kvcache: snapshot restore failed: %v", err)
		c.engine.Reset()
		c.cachedTokens = c.cachedTokens[:0]
		return 0
	}
	c.cachedTokens = append(c.cachedTokens[:0], pinned.tokens...)
	log.Printf("gem4d: kvcache: restored snapshot (%d tokens, lcp %d vs live %d)",
		len(pinned.tokens), bestLCP, liveLCP)
	return bestLCP
}

// saveLive snapshots the live sequence into the pool, LRU-evicting until it
// fits the budget. No-ops (with a log line on failure) when the sequence is
// over budget by itself or the engine copy fails. Caller holds Lock().
func (c *KVCache) saveLive() {
	n := len(c.cachedTokens)
	size := int64(c.snap.KVStateSize(n))
	if size <= 0 || size > c.snapBudget {
		return
	}
	// A snapshot holding this exact sequence (or a longer one it prefixes)
	// makes a new save pointless.
	for _, s := range c.saved {
		if len(s.tokens) >= n && longestCommonPrefix(s.tokens, c.cachedTokens) == n {
			return
		}
	}
	for c.snapBytes+size > c.snapBudget && len(c.saved) > 0 {
		lru := 0
		for i, s := range c.saved {
			if s.used < c.saved[lru].used {
				lru = i
			}
		}
		c.snapBytes -= int64(len(c.saved[lru].state))
		log.Printf("gem4d: kvcache: evicted LRU snapshot (%d tokens)", len(c.saved[lru].tokens))
		c.saved = append(c.saved[:lru], c.saved[lru+1:]...)
	}
	buf := make([]byte, size)
	if err := c.snap.KVSave(buf, n); err != nil {
		log.Printf("gem4d: kvcache: snapshot save failed: %v", err)
		return
	}
	c.snapClock++
	toks := make([]int32, n)
	copy(toks, c.cachedTokens)
	c.saved = append(c.saved, &savedSeq{tokens: toks, state: buf, used: c.snapClock})
	c.snapBytes += size
	log.Printf("gem4d: kvcache: saved live sequence (%d tokens, %.0f MB; pool %d seqs / %.0f MB)",
		n, float64(size)/(1<<20), len(c.saved), float64(c.snapBytes)/(1<<20))
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

// Stats returns cumulative prefix-cache statistics. Lock-free: safe to call
// while another request holds the KV lock (the /health use case).
func (c *KVCache) Stats() (hits, misses int, hitRate float64) {
	hits, misses = int(c.hits.Load()), int(c.misses.Load())
	// Load hitTokens BEFORE reqTokens (the writer adds hitTokens first): a
	// concurrent update then transiently UNDER-reports the rate instead of
	// pairing a new hitTokens with an old reqTokens (rate > 1.0).
	hitTok := c.hitTokens.Load()
	if req := c.reqTokens.Load(); req > 0 {
		hitRate = float64(hitTok) / float64(req)
	}
	return
}

// DetailedStats returns the raw prefix-cache counters for /metrics. Lock-free.
func (c *KVCache) DetailedStats() (hits, misses int, reusedTokens, reqTokens int64) {
	return int(c.hits.Load()), int(c.misses.Load()), c.hitTokens.Load(), c.reqTokens.Load()
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
