package server

import (
	"net/http"
	"strings"
	"time"

	prommetrics "github.com/hikmaai-io/fucina/internal/metrics"
)

// prometheusMetrics is process-scoped, matching the lifetime and semantics of
// Prometheus counters. Server tests may replace it before exercising a scrape.
var prometheusMetrics = prommetrics.New()

// handleMetricsDispatch keeps the historical JSON representation at the same
// endpoint through HTTP content negotiation while making Prometheus the
// default representation expected by curl and ordinary scrapers.
func (s *Server) handleMetricsDispatch(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Query().Get("format") {
	case "json":
		s.handleMetrics(w, r)
		return
	case "prometheus", "prom":
		s.handlePrometheusMetrics(w, r)
		return
	}
	if strings.Contains(strings.ToLower(r.Header.Get("Accept")), "application/json") {
		s.handleMetrics(w, r)
		return
	}
	s.handlePrometheusMetrics(w, r)
}

func (s *Server) handlePrometheusMetrics(w http.ResponseWriter, r *http.Request) {
	s.refreshPrometheusMetrics()
	prometheusMetrics.ServeHTTP(w, r)
}

// refreshPrometheusMetrics imports only lock-free server mirrors. It must not
// call Engine.NTokens or Engine.SpecStats: those can block behind inference.
func (s *Server) refreshPrometheusMetrics() {
	capacity := uint64(s.engine.ContextSize())
	prometheusMetrics.SetKVUtilization(nonNegative(s.lastUsed.Load()), capacity)
	prometheusMetrics.SetQueueDepth(len(s.inflight))

	hits, misses, _, _ := s.kv.DetailedStats()
	prometheusMetrics.UpdateCacheTotals(
		prommetrics.CacheFlat,
		uint64(hits+misses),
		uint64(hits),
		0,
	)
	if s.scheduler != nil {
		lookups, hitBlocks, _, evictions := s.scheduler.PrefixCacheStats()
		prometheusMetrics.UpdateCacheTotals(
			prommetrics.CachePaged,
			nonNegative(lookups),
			nonNegative(hitBlocks),
			nonNegative(evictions),
		)
	}
	prometheusMetrics.UpdateSpeculationTotals(
		"server",
		nonNegative(s.specDrafted.Load()),
		nonNegative(s.specAccepted.Load()),
	)
}

func nonNegative(value int64) uint64 {
	if value < 0 {
		return 0
	}
	return uint64(value)
}

func observePrometheusQueueWait(elapsed time.Duration) {
	prometheusMetrics.ObserveQueueWait(elapsed)
	prometheusMetrics.ObservePhase(prommetrics.PhaseQueue, elapsed)
}

func observePrometheusPrefill(elapsed time.Duration) {
	prometheusMetrics.ObservePhase(prommetrics.PhasePrefill, elapsed)
}

func observePrometheusDecode(elapsed time.Duration) {
	prometheusMetrics.ObservePhase(prommetrics.PhaseDecode, elapsed)
}

func prometheusRequestOutcome(status int) prommetrics.RequestOutcome {
	switch {
	case status == 499:
		return prommetrics.OutcomeCancelled
	case status >= 500:
		return prommetrics.OutcomeError
	case status >= 400:
		return prommetrics.OutcomeRejected
	default:
		return prommetrics.OutcomeSuccess
	}
}
