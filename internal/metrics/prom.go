// Package metrics provides dependency-free Prometheus instrumentation for the
// serving stack. It deliberately imports no Fucina runtime packages, allowing
// server, scheduler, cache, and engine adapters to emit telemetry without
// introducing import cycles.
package metrics

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"sync"
	"time"
)

const prometheusContentType = "text/plain; version=0.0.4; charset=utf-8"

// RequestOutcome bounds the request lifecycle label cardinality.
type RequestOutcome string

const (
	OutcomeSuccess   RequestOutcome = "success"
	OutcomeError     RequestOutcome = "error"
	OutcomeCancelled RequestOutcome = "cancelled"
	OutcomeRejected  RequestOutcome = "rejected"
)

// Phase identifies scheduler and inference phases whose latency is measured.
type Phase string

const (
	PhaseQueue       Phase = "queue"
	PhaseCoalesce    Phase = "coalesce"
	PhaseAdmission   Phase = "admission"
	PhasePrefill     Phase = "prefill"
	PhaseFirstDecode Phase = "first_decode"
	PhaseDecode      Phase = "decode"
)

// CacheSource identifies independently-monotonic cache counter producers.
type CacheSource string

const (
	CacheFlat  CacheSource = "flat"
	CachePaged CacheSource = "paged"
	CacheHost  CacheSource = "host_snapshot"
	CacheDisk  CacheSource = "disk_session"
)

// RequestRecorder is the narrow interface used by HTTP serving paths.
type RequestRecorder interface {
	RequestStarted()
	RequestFinished(RequestOutcome, time.Duration)
	ObserveTTFT(time.Duration)
	ObserveITL(time.Duration)
	Cancellation()
}

// SchedulerRecorder is the narrow interface used by admission and batch code.
type SchedulerRecorder interface {
	SetQueueDepth(int)
	ObserveQueueWait(time.Duration)
	ObservePhase(Phase, time.Duration)
	ObserveBatchSize(int)
	AddPreemptions(uint64)
}

// CacheRecorder accepts event deltas and snapshots from cache implementations.
// UpdateCacheTotals converts a producer's cumulative values to deltas, so
// polling the same snapshot repeatedly never double counts it.
type CacheRecorder interface {
	AddCacheActivity(CacheSource, uint64, uint64, uint64)
	UpdateCacheTotals(CacheSource, uint64, uint64, uint64)
	SetKVUtilization(uint64, uint64)
}

// DecodeRecorder covers speculative decoding and expert residency telemetry.
type DecodeRecorder interface {
	AddSpeculation(uint64, uint64)
	UpdateSpeculationTotals(string, uint64, uint64)
	SetExpertResidency(uint64, uint64)
}

// Recorder is implemented by Collector. Consumers should generally depend on
// one of the narrower interfaces above.
type Recorder interface {
	RequestRecorder
	SchedulerRecorder
	CacheRecorder
	DecodeRecorder
}

var _ Recorder = (*Collector)(nil)

var (
	durationBuckets = []float64{0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60}
	itlBuckets      = []float64{0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2.5}
	batchBuckets    = []float64{1, 2, 4, 8, 16, 32, 64, 128, 256}
)

type histogram struct {
	buckets []float64
	counts  []uint64 // non-cumulative counts; exposition makes them cumulative
	count   uint64
	sum     float64
}

func newHistogram(buckets []float64) histogram {
	return histogram{buckets: append([]float64(nil), buckets...), counts: make([]uint64, len(buckets))}
}

func (h *histogram) observe(v float64) {
	if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
		return
	}
	h.count++
	h.sum += v
	for i, upper := range h.buckets {
		if v <= upper {
			h.counts[i]++
			break
		}
	}
}

type cacheCounters struct {
	lookups, hits, evictions             uint64
	lastLookups, lastHits, lastEvictions uint64
	initialized                          bool
}

type speculationCounters struct {
	proposed, accepted         uint64
	lastProposed, lastAccepted uint64
	initialized                bool
}

// Collector is a concurrency-safe in-process Prometheus collector.
type Collector struct {
	mu sync.Mutex

	lifecycle       map[RequestOutcome]uint64
	requestDuration histogram
	queueDepth      int
	queueWait       histogram
	ttft            histogram
	itl             histogram
	phases          map[Phase]*histogram
	batchSize       histogram

	speculation                   map[string]*speculationCounters
	cache                         map[CacheSource]*cacheCounters
	kvUsed, kvCapacity            uint64
	cancellations                 uint64
	preemptions                   uint64
	expertsResident, expertsTotal uint64
}

// New returns an initialized collector. New collectors expose every required
// series immediately, including the intentionally-zero preemption counter.
func New() *Collector {
	c := &Collector{
		lifecycle:       make(map[RequestOutcome]uint64),
		requestDuration: newHistogram(durationBuckets),
		queueWait:       newHistogram(durationBuckets),
		ttft:            newHistogram(durationBuckets),
		itl:             newHistogram(itlBuckets),
		phases:          make(map[Phase]*histogram),
		batchSize:       newHistogram(batchBuckets),
		speculation:     make(map[string]*speculationCounters),
		cache:           make(map[CacheSource]*cacheCounters),
	}
	for _, outcome := range []RequestOutcome{OutcomeSuccess, OutcomeError, OutcomeCancelled, OutcomeRejected} {
		c.lifecycle[outcome] = 0
	}
	for _, phase := range []Phase{PhaseQueue, PhaseCoalesce, PhaseAdmission, PhasePrefill, PhaseFirstDecode, PhaseDecode} {
		h := newHistogram(durationBuckets)
		c.phases[phase] = &h
	}
	for _, source := range []CacheSource{CacheFlat, CachePaged, CacheHost, CacheDisk} {
		c.cache[source] = &cacheCounters{}
	}
	c.speculation["default"] = &speculationCounters{}
	return c
}

func normalizeOutcome(v RequestOutcome) RequestOutcome {
	switch v {
	case OutcomeSuccess, OutcomeError, OutcomeCancelled, OutcomeRejected:
		return v
	default:
		return OutcomeError
	}
}

func normalizePhase(v Phase) Phase {
	switch v {
	case PhaseQueue, PhaseCoalesce, PhaseAdmission, PhasePrefill, PhaseFirstDecode, PhaseDecode:
		return v
	default:
		return PhaseDecode
	}
}

func normalizeCacheSource(v CacheSource) CacheSource {
	switch v {
	case CacheFlat, CachePaged, CacheHost, CacheDisk:
		return v
	default:
		return CacheFlat
	}
}

// RequestStarted records admission into the HTTP request lifecycle.
func (c *Collector) RequestStarted() {
	c.mu.Lock()
	c.lifecycle[RequestOutcome("started")]++
	c.mu.Unlock()
}

// RequestFinished records a terminal lifecycle outcome and request latency.
func (c *Collector) RequestFinished(outcome RequestOutcome, elapsed time.Duration) {
	c.mu.Lock()
	c.lifecycle[normalizeOutcome(outcome)]++
	c.requestDuration.observe(elapsed.Seconds())
	c.mu.Unlock()
}

func (c *Collector) ObserveTTFT(elapsed time.Duration) {
	c.mu.Lock()
	c.ttft.observe(elapsed.Seconds())
	c.mu.Unlock()
}

func (c *Collector) ObserveITL(elapsed time.Duration) {
	c.mu.Lock()
	c.itl.observe(elapsed.Seconds())
	c.mu.Unlock()
}

func (c *Collector) SetQueueDepth(depth int) {
	if depth < 0 {
		depth = 0
	}
	c.mu.Lock()
	c.queueDepth = depth
	c.mu.Unlock()
}

func (c *Collector) ObserveQueueWait(elapsed time.Duration) {
	c.mu.Lock()
	c.queueWait.observe(elapsed.Seconds())
	c.mu.Unlock()
}

func (c *Collector) ObservePhase(phase Phase, elapsed time.Duration) {
	c.mu.Lock()
	c.phases[normalizePhase(phase)].observe(elapsed.Seconds())
	c.mu.Unlock()
}

func (c *Collector) ObserveBatchSize(size int) {
	if size < 0 {
		return
	}
	c.mu.Lock()
	c.batchSize.observe(float64(size))
	c.mu.Unlock()
}

func (c *Collector) AddSpeculation(proposed, accepted uint64) {
	if accepted > proposed {
		accepted = proposed
	}
	c.mu.Lock()
	s := c.speculation["default"]
	s.proposed += proposed
	s.accepted += accepted
	c.mu.Unlock()
}

// UpdateSpeculationTotals imports cumulative producer counters. A producer
// reset is treated as a new epoch while the exported process counter remains
// monotonic.
func (c *Collector) UpdateSpeculationTotals(source string, proposed, accepted uint64) {
	if source == "" {
		source = "default"
	}
	if accepted > proposed {
		accepted = proposed
	}
	c.mu.Lock()
	s := c.speculation[source]
	if s == nil {
		s = &speculationCounters{}
		c.speculation[source] = s
	}
	if !s.initialized {
		s.proposed += proposed
		s.accepted += accepted
		s.initialized = true
	} else {
		s.proposed += counterDelta(s.lastProposed, proposed)
		s.accepted += counterDelta(s.lastAccepted, accepted)
	}
	s.lastProposed, s.lastAccepted = proposed, accepted
	c.mu.Unlock()
}

func (c *Collector) AddCacheActivity(source CacheSource, lookups, hits, evictions uint64) {
	if hits > lookups {
		hits = lookups
	}
	c.mu.Lock()
	s := c.cache[normalizeCacheSource(source)]
	s.lookups += lookups
	s.hits += hits
	s.evictions += evictions
	c.mu.Unlock()
}

// UpdateCacheTotals imports one cache producer's cumulative counters without
// double counting repeated snapshots. Producer resets start a new epoch.
func (c *Collector) UpdateCacheTotals(source CacheSource, lookups, hits, evictions uint64) {
	if hits > lookups {
		hits = lookups
	}
	c.mu.Lock()
	s := c.cache[normalizeCacheSource(source)]
	if !s.initialized {
		s.lookups += lookups
		s.hits += hits
		s.evictions += evictions
		s.initialized = true
	} else {
		s.lookups += counterDelta(s.lastLookups, lookups)
		s.hits += counterDelta(s.lastHits, hits)
		s.evictions += counterDelta(s.lastEvictions, evictions)
	}
	s.lastLookups, s.lastHits, s.lastEvictions = lookups, hits, evictions
	c.mu.Unlock()
}

func counterDelta(previous, current uint64) uint64 {
	if current >= previous {
		return current - previous
	}
	return current
}

func (c *Collector) SetKVUtilization(used, capacity uint64) {
	if capacity > 0 && used > capacity {
		used = capacity
	}
	c.mu.Lock()
	c.kvUsed, c.kvCapacity = used, capacity
	c.mu.Unlock()
}

func (c *Collector) Cancellation() {
	c.mu.Lock()
	c.cancellations++
	c.mu.Unlock()
}

func (c *Collector) AddPreemptions(n uint64) {
	c.mu.Lock()
	c.preemptions += n
	c.mu.Unlock()
}

func (c *Collector) SetExpertResidency(resident, total uint64) {
	if total > 0 && resident > total {
		resident = total
	}
	c.mu.Lock()
	c.expertsResident, c.expertsTotal = resident, total
	c.mu.Unlock()
}

// Handler returns the Prometheus HTTP handler.
func (c *Collector) Handler() http.Handler { return c }

// ServeHTTP writes Prometheus text exposition. Metrics are snapshotted before
// network writes, so a slow scraper never blocks inference instrumentation.
func (c *Collector) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	payload := c.Gather()
	w.Header().Set("Content-Type", prometheusContentType)
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	if r.Method == http.MethodHead {
		return
	}
	_, _ = w.Write(payload)
}

// Gather returns one internally consistent Prometheus text snapshot.
func (c *Collector) Gather() []byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	var b bytes.Buffer
	c.writePrometheusLocked(&b)
	return b.Bytes()
}

func helpType(w io.Writer, name, help, typ string) {
	fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, typ)
}

func writeCounter(w io.Writer, name string, labels string, value uint64) {
	fmt.Fprintf(w, "%s%s %d\n", name, labels, value)
}

func writeGauge(w io.Writer, name string, value float64) {
	fmt.Fprintf(w, "%s %s\n", name, strconv.FormatFloat(value, 'g', -1, 64))
}

func writeHistogram(w io.Writer, name string, labels string, h *histogram) {
	cumulative := uint64(0)
	for i, upper := range h.buckets {
		cumulative += h.counts[i]
		fmt.Fprintf(w, "%s_bucket%s %d\n", name, mergeLabel(labels, "le", strconv.FormatFloat(upper, 'g', -1, 64)), cumulative)
	}
	fmt.Fprintf(w, "%s_bucket%s %d\n", name, mergeLabel(labels, "le", "+Inf"), h.count)
	fmt.Fprintf(w, "%s_sum%s %s\n", name, labels, strconv.FormatFloat(h.sum, 'g', -1, 64))
	fmt.Fprintf(w, "%s_count%s %d\n", name, labels, h.count)
}

func mergeLabel(labels, key, value string) string {
	pair := key + "=\"" + escapeLabel(value) + "\""
	if labels == "" {
		return "{" + pair + "}"
	}
	return labels[:len(labels)-1] + "," + pair + "}"
}

func label(key, value string) string { return "{" + key + "=\"" + escapeLabel(value) + "\"}" }

func escapeLabel(s string) string {
	var b bytes.Buffer
	for _, r := range s {
		switch r {
		case '\\':
			b.WriteString("\\\\")
		case '\n':
			b.WriteString("\\n")
		case '"':
			b.WriteString("\\\"")
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

func (c *Collector) writePrometheusLocked(w io.Writer) {
	helpType(w, "fucina_request_lifecycle_total", "Requests observed by lifecycle state.", "counter")
	writeCounter(w, "fucina_request_lifecycle_total", label("state", "started"), c.lifecycle[RequestOutcome("started")])
	for _, state := range []RequestOutcome{OutcomeSuccess, OutcomeError, OutcomeCancelled, OutcomeRejected} {
		writeCounter(w, "fucina_request_lifecycle_total", label("state", string(state)), c.lifecycle[state])
	}
	helpType(w, "fucina_request_duration_seconds", "End-to-end request latency.", "histogram")
	writeHistogram(w, "fucina_request_duration_seconds", "", &c.requestDuration)

	helpType(w, "fucina_queue_depth", "Current scheduler queue depth.", "gauge")
	writeGauge(w, "fucina_queue_depth", float64(c.queueDepth))
	helpType(w, "fucina_queue_wait_seconds", "Time requests spend waiting for admission.", "histogram")
	writeHistogram(w, "fucina_queue_wait_seconds", "", &c.queueWait)
	helpType(w, "fucina_ttft_seconds", "Time from request start to first visible token.", "histogram")
	writeHistogram(w, "fucina_ttft_seconds", "", &c.ttft)
	helpType(w, "fucina_itl_seconds", "Inter-token latency between visible output tokens.", "histogram")
	writeHistogram(w, "fucina_itl_seconds", "", &c.itl)
	helpType(w, "fucina_phase_duration_seconds", "Latency split by scheduler and inference phase.", "histogram")
	for _, phase := range []Phase{PhaseQueue, PhaseCoalesce, PhaseAdmission, PhasePrefill, PhaseFirstDecode, PhaseDecode} {
		writeHistogram(w, "fucina_phase_duration_seconds", label("phase", string(phase)), c.phases[phase])
	}
	helpType(w, "fucina_batch_size", "Number of sequences in an execution batch.", "histogram")
	writeHistogram(w, "fucina_batch_size", "", &c.batchSize)

	helpType(w, "fucina_speculation_proposed_total", "Draft tokens proposed for speculative verification.", "counter")
	helpType(w, "fucina_speculation_accepted_total", "Draft tokens accepted by target verification.", "counter")
	var proposed, accepted uint64
	for _, s := range c.speculation {
		proposed += s.proposed
		accepted += s.accepted
	}
	writeCounter(w, "fucina_speculation_proposed_total", "", proposed)
	writeCounter(w, "fucina_speculation_accepted_total", "", accepted)

	var lookups, hits, evictions uint64
	for _, s := range c.cache {
		lookups += s.lookups
		hits += s.hits
		evictions += s.evictions
	}
	helpType(w, "fucina_cache_lookups_total", "Prefix cache lookup attempts.", "counter")
	writeCounter(w, "fucina_cache_lookups_total", "", lookups)
	helpType(w, "fucina_cache_hits_total", "Prefix cache lookups that physically skipped work.", "counter")
	writeCounter(w, "fucina_cache_hits_total", "", hits)
	helpType(w, "fucina_cache_evictions_total", "Prefix cache entries or blocks evicted.", "counter")
	writeCounter(w, "fucina_cache_evictions_total", "", evictions)

	ratio := float64(0)
	if c.kvCapacity > 0 {
		ratio = float64(c.kvUsed) / float64(c.kvCapacity)
	}
	helpType(w, "fucina_kv_utilization_ratio", "KV capacity currently utilized, from zero to one.", "gauge")
	writeGauge(w, "fucina_kv_utilization_ratio", ratio)
	helpType(w, "fucina_kv_tokens", "KV token slots by state.", "gauge")
	fmt.Fprintf(w, "fucina_kv_tokens%s %d\n", label("state", "used"), c.kvUsed)
	fmt.Fprintf(w, "fucina_kv_tokens%s %d\n", label("state", "capacity"), c.kvCapacity)

	helpType(w, "fucina_cancellations_total", "Requests cancelled by clients or deadlines.", "counter")
	writeCounter(w, "fucina_cancellations_total", "", c.cancellations)
	helpType(w, "fucina_preemptions_total", "Scheduler preemptions; expected to remain zero until preemption is implemented.", "counter")
	writeCounter(w, "fucina_preemptions_total", "", c.preemptions)

	expertRatio := float64(0)
	if c.expertsTotal > 0 {
		expertRatio = float64(c.expertsResident) / float64(c.expertsTotal)
	}
	helpType(w, "fucina_expert_residency_ratio", "Fraction of model experts currently resident.", "gauge")
	writeGauge(w, "fucina_expert_residency_ratio", expertRatio)
	helpType(w, "fucina_experts", "Expert count by residency state.", "gauge")
	fmt.Fprintf(w, "fucina_experts%s %d\n", label("state", "resident"), c.expertsResident)
	fmt.Fprintf(w, "fucina_experts%s %d\n", label("state", "total"), c.expertsTotal)
}
