package server

import (
	"sync"
	"sync/atomic"
	"time"
)

// Metrics accumulates server-wide throughput so /metrics can report cache and
// speed health over time (cumulative averages plus the most recent request).
// The throughput fields are mutex-guarded; the request-outcome/latency/TTFT
// fields use atomics so /metrics and the per-request access-log path never
// contend with the engine lock. Updates are O(1) per request.
type Metrics struct {
	mu sync.Mutex

	requests int64

	// prefill
	prefillTokens  int64
	prefillSec     float64
	lastPrefillTPS float64

	// decode
	decodeTokens  int64
	decodeSec     float64
	lastDecodeTPS float64

	// Request-outcome + latency (atomics: written from logRequest, read lock-free
	// by /metrics). These are the SLO signals — error rate, latency, TTFT — that
	// the throughput counters above did not cover.
	requestsTotal atomic.Int64
	requestErrors atomic.Int64 // responses with status >= 500
	requestNanos  atomic.Int64 // cumulative request duration (for the average)
	ttftCount     atomic.Int64
	ttftNanos     atomic.Int64 // cumulative time-to-first-token
}

// recordRequest tallies one completed request's outcome and total latency.
func (m *Metrics) recordRequest(status int, d time.Duration) {
	m.requestsTotal.Add(1)
	if status >= 500 {
		m.requestErrors.Add(1)
	}
	m.requestNanos.Add(int64(d))
}

// recordTTFT records one request's time-to-first-token (streaming path).
func (m *Metrics) recordTTFT(d time.Duration) {
	m.ttftCount.Add(1)
	m.ttftNanos.Add(int64(d))
}

func (m *Metrics) avgRequestMs() float64 {
	n := m.requestsTotal.Load()
	if n == 0 {
		return 0
	}
	return float64(m.requestNanos.Load()) / float64(n) / 1e6
}

func (m *Metrics) avgTTFTMs() float64 {
	n := m.ttftCount.Load()
	if n == 0 {
		return 0
	}
	return float64(m.ttftNanos.Load()) / float64(n) / 1e6
}

func (m *Metrics) recordPrefill(newTokens int, sec float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests++
	m.prefillTokens += int64(newTokens)
	m.prefillSec += sec
	if sec > 0 {
		m.lastPrefillTPS = float64(newTokens) / sec
	}
}

func (m *Metrics) recordDecode(genTokens int, sec float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.decodeTokens += int64(genTokens)
	m.decodeSec += sec
	if sec > 0 {
		m.lastDecodeTPS = float64(genTokens) / sec
	}
}

func avg(tok int64, sec float64) float64 {
	if sec <= 0 {
		return 0
	}
	return float64(tok) / sec
}

// snapshot returns the metrics as a JSON-friendly map, combined with live
// context-utilization and prefix-cache stats supplied by the caller.
func (m *Metrics) snapshot(model string, ctxUsed, ctxCap int,
	kvSlidingMB, kvGlobalMB float64,
	hits, misses int, reusedTok, reqTok int64,
	specSteps, specDrafted, specAccepted, specEmitted int64) map[string]interface{} {

	m.mu.Lock()
	defer m.mu.Unlock()

	usedPct := 0.0
	if ctxCap > 0 {
		usedPct = 100.0 * float64(ctxUsed) / float64(ctxCap)
	}
	hitRate := 0.0
	if reqTok > 0 {
		hitRate = float64(reusedTok) / float64(reqTok)
	}
	// Speculative-decode health: τ = tokens emitted per target forward (higher = the
	// drafter is carrying more of the decode); acceptance = matched drafts / proposed.
	tokensPerForward := 0.0
	if specSteps > 0 {
		tokensPerForward = float64(specEmitted) / float64(specSteps)
	}
	acceptRate := 0.0
	if specDrafted > 0 {
		acceptRate = float64(specAccepted) / float64(specDrafted)
	}
	return map[string]interface{}{
		"model":    model,
		"requests": m.requests,
		"context": map[string]interface{}{
			"used":     ctxUsed,
			"capacity": ctxCap,
			"used_pct": round1(usedPct),
		},
		"kv_cache_mb": map[string]interface{}{
			"sliding": round1(kvSlidingMB),
			"global":  round1(kvGlobalMB),
			"total":   round1(kvSlidingMB + kvGlobalMB),
		},
		"prefix_cache": map[string]interface{}{
			"hits":           hits,
			"misses":         misses,
			"hit_rate":       round3(hitRate),
			"reused_tokens":  reusedTok,
			"request_tokens": reqTok,
		},
		"throughput_tok_s": map[string]interface{}{
			"prefill_last": round1(m.lastPrefillTPS),
			"prefill_avg":  round1(avg(m.prefillTokens, m.prefillSec)),
			"decode_last":  round1(m.lastDecodeTPS),
			"decode_avg":   round1(avg(m.decodeTokens, m.decodeSec)),
		},
		"totals": map[string]interface{}{
			"prefill_tokens": m.prefillTokens,
			"decode_tokens":  m.decodeTokens,
		},
		// SLO + saturation signals (lock-free atomics). in_flight/queue_depth are
		// supplied by the caller (Server) which owns the admission channel.
		"requests_detail": map[string]interface{}{
			"total":          m.requestsTotal.Load(),
			"errors":         m.requestErrors.Load(),
			"avg_latency_ms": round1(m.avgRequestMs()),
			"avg_ttft_ms":    round1(m.avgTTFTMs()),
		},
		"speculation": map[string]interface{}{
			"verify_forwards":    specSteps,
			"drafted":            specDrafted,
			"accepted":           specAccepted,
			"emitted":            specEmitted,
			"tokens_per_forward": round3(tokensPerForward),
			"accept_rate":        round3(acceptRate),
		},
	}
}

func round1(f float64) float64 { return float64(int64(f*10+0.5)) / 10 }
func round3(f float64) float64 { return float64(int64(f*1000+0.5)) / 1000 }
