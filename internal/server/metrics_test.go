// ABOUTME: Tests for the observability metrics added in the hardening pass —
// ABOUTME: TTFT, request latency, error counts, and in-flight/saturation gauges.

package server

import (
	"sync"
	"testing"
	"time"
)

func TestMetricsRecordsLatencyAndErrors(t *testing.T) {
	var m Metrics
	m.recordRequest(200, 100*time.Millisecond)
	m.recordRequest(200, 300*time.Millisecond)
	m.recordRequest(500, 50*time.Millisecond)

	if got := m.requestErrors.Load(); got != 1 {
		t.Errorf("requestErrors=%d want 1 (one 5xx)", got)
	}
	if got := m.requestsTotal.Load(); got != 3 {
		t.Errorf("requestsTotal=%d want 3", got)
	}
	// Average latency = (100+300+50)/3 = 150ms.
	avgMs := m.avgRequestMs()
	if avgMs < 149 || avgMs > 151 {
		t.Errorf("avgRequestMs=%.1f want ~150", avgMs)
	}
}

func TestMetricsRecordsTTFT(t *testing.T) {
	var m Metrics
	m.recordTTFT(20 * time.Millisecond)
	m.recordTTFT(40 * time.Millisecond)
	if got := m.avgTTFTMs(); got < 29 || got > 31 {
		t.Errorf("avgTTFTMs=%.1f want ~30", got)
	}
}

func TestMetricsSnapshotIncludesNewFields(t *testing.T) {
	var m Metrics
	m.recordRequest(200, 10*time.Millisecond)
	m.recordRequest(503, 1*time.Millisecond)
	m.recordTTFT(5 * time.Millisecond)

	snap := m.snapshot("model", 0, 8192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
	req, ok := snap["requests_detail"].(map[string]interface{})
	if !ok {
		t.Fatalf("snapshot missing requests_detail: %+v", snap)
	}
	for _, k := range []string{"total", "errors", "avg_latency_ms", "avg_ttft_ms"} {
		if _, present := req[k]; !present {
			t.Errorf("requests_detail missing %q", k)
		}
	}
}

// recordRequest/recordTTFT must be race-free under concurrency (they use atomics,
// not the Metrics mutex, so /metrics stays lock-free). Run with -race.
func TestMetricsConcurrentRecord(t *testing.T) {
	var m Metrics
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			status := 200
			if n%10 == 0 {
				status = 500
			}
			m.recordRequest(status, time.Duration(n)*time.Millisecond)
			m.recordTTFT(time.Duration(n) * time.Millisecond)
		}(i)
	}
	wg.Wait()
	if got := m.requestsTotal.Load(); got != 50 {
		t.Errorf("requestsTotal=%d want 50", got)
	}
	if got := m.requestErrors.Load(); got != 5 {
		t.Errorf("requestErrors=%d want 5", got)
	}
}
