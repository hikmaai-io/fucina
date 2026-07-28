package metrics

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestSmokeScrapeIncludesRequestCacheAndTTFT(t *testing.T) {
	c := New()
	c.RequestStarted()
	c.RequestFinished(OutcomeSuccess, 120*time.Millisecond)
	c.ObserveTTFT(30 * time.Millisecond)
	c.UpdateCacheTotals(CachePaged, 4, 3, 1)

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rr := httptest.NewRecorder()
	c.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("GET /metrics status = %d, want 200", rr.Code)
	}
	if got := rr.Header().Get("Content-Type"); got != prometheusContentType {
		t.Fatalf("Content-Type = %q, want %q", got, prometheusContentType)
	}
	body := rr.Body.String()
	for _, series := range []string{
		`fucina_request_lifecycle_total{state="started"} 1`,
		`fucina_request_lifecycle_total{state="success"} 1`,
		"fucina_request_duration_seconds_count 1",
		"fucina_cache_lookups_total 4",
		"fucina_cache_hits_total 3",
		"fucina_cache_evictions_total 1",
		"fucina_ttft_seconds_count 1",
	} {
		if !strings.Contains(body, series) {
			t.Errorf("scrape missing %q", series)
		}
	}
}

func TestRequiredSeriesExistBeforeTraffic(t *testing.T) {
	body := string(New().Gather())
	for _, name := range []string{
		"fucina_request_lifecycle_total",
		"fucina_request_duration_seconds_bucket",
		"fucina_queue_depth",
		"fucina_queue_wait_seconds_bucket",
		"fucina_ttft_seconds_bucket",
		"fucina_itl_seconds_bucket",
		`fucina_phase_duration_seconds_bucket{phase="prefill"`,
		`fucina_phase_duration_seconds_bucket{phase="decode"`,
		"fucina_batch_size_bucket",
		"fucina_speculation_proposed_total",
		"fucina_speculation_accepted_total",
		"fucina_cache_lookups_total",
		"fucina_cache_hits_total",
		"fucina_cache_evictions_total",
		"fucina_kv_utilization_ratio",
		"fucina_cancellations_total",
		"fucina_preemptions_total 0",
		"fucina_expert_residency_ratio",
	} {
		if !strings.Contains(body, name) {
			t.Errorf("empty scrape missing required series %q", name)
		}
	}
}

func TestCumulativeSnapshotsDoNotDoubleCountAndSurviveReset(t *testing.T) {
	c := New()
	c.UpdateCacheTotals(CachePaged, 10, 7, 2)
	c.UpdateCacheTotals(CachePaged, 10, 7, 2)
	c.UpdateCacheTotals(CachePaged, 14, 9, 3)
	// A producer restart resets its source totals; exported counters stay monotonic.
	c.UpdateCacheTotals(CachePaged, 2, 1, 0)

	c.UpdateSpeculationTotals("engine", 20, 12)
	c.UpdateSpeculationTotals("engine", 20, 12)
	c.UpdateSpeculationTotals("engine", 25, 15)
	c.UpdateSpeculationTotals("engine", 1, 1)

	body := string(c.Gather())
	for _, want := range []string{
		"fucina_cache_lookups_total 16",
		"fucina_cache_hits_total 10",
		"fucina_cache_evictions_total 3",
		"fucina_speculation_proposed_total 26",
		"fucina_speculation_accepted_total 16",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("scrape missing %q\n%s", want, body)
		}
	}
}

func TestInputInvariantsClampInvalidRatios(t *testing.T) {
	c := New()
	c.AddCacheActivity(CacheFlat, 2, 9, 0)
	c.AddSpeculation(3, 8)
	c.SetKVUtilization(200, 100)
	c.SetExpertResidency(12, 4)
	body := string(c.Gather())
	for _, want := range []string{
		"fucina_cache_hits_total 9",
		"fucina_speculation_accepted_total 3",
		"fucina_kv_utilization_ratio 1",
		"fucina_expert_residency_ratio 1",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("scrape missing clamped value %q", want)
		}
	}
}

func TestHistogramBucketsAreCumulative(t *testing.T) {
	c := New()
	c.ObserveTTFT(3 * time.Millisecond)
	c.ObserveTTFT(40 * time.Millisecond)
	body := string(c.Gather())
	previous := uint64(0)
	seen := 0
	for _, line := range strings.Split(body, "\n") {
		if !strings.HasPrefix(line, "fucina_ttft_seconds_bucket") {
			continue
		}
		fields := strings.Fields(line)
		got, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			t.Fatalf("parse %q: %v", line, err)
		}
		if got < previous {
			t.Fatalf("non-cumulative histogram: %d after %d", got, previous)
		}
		previous = got
		seen++
	}
	if seen == 0 || previous != 2 {
		t.Fatalf("TTFT buckets seen=%d final=%d, want final=2", seen, previous)
	}
}

func TestCollectorConcurrentRecordAndScrape(t *testing.T) {
	c := New()
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			c.RequestStarted()
			c.RequestFinished(OutcomeSuccess, time.Duration(n)*time.Millisecond)
			c.ObserveTTFT(time.Duration(n) * time.Millisecond)
			c.ObserveITL(time.Millisecond)
			c.SetQueueDepth(n)
			c.AddCacheActivity(CacheFlat, 1, 1, 0)
			_ = c.Gather()
		}(i)
	}
	wg.Wait()
	body := string(c.Gather())
	for _, want := range []string{
		`fucina_request_lifecycle_total{state="started"} 50`,
		`fucina_request_lifecycle_total{state="success"} 50`,
		"fucina_cache_hits_total 50",
	} {
		if !strings.Contains(body, want) {
			t.Error(fmt.Sprintf("concurrent scrape missing %q", want))
		}
	}
}

func TestHandlerMethodAndHead(t *testing.T) {
	c := New()
	for _, tc := range []struct {
		method string
		status int
		body   bool
	}{
		{method: http.MethodHead, status: http.StatusOK, body: false},
		{method: http.MethodPost, status: http.StatusMethodNotAllowed, body: true},
	} {
		t.Run(tc.method, func(t *testing.T) {
			rr := httptest.NewRecorder()
			c.ServeHTTP(rr, httptest.NewRequest(tc.method, "/metrics", nil))
			if rr.Code != tc.status {
				t.Fatalf("status=%d want=%d", rr.Code, tc.status)
			}
			if (rr.Body.Len() > 0) != tc.body {
				t.Fatalf("body len=%d, want body=%v", rr.Body.Len(), tc.body)
			}
		})
	}
}
