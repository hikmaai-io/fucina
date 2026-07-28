package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	prommetrics "github.com/hikmaai-io/fucina/internal/metrics"
)

func TestMetricsDispatchDefaultsToPrometheusSmokeSeries(t *testing.T) {
	prometheusMetrics = prommetrics.New()
	prometheusMetrics.RequestStarted()
	prometheusMetrics.RequestFinished(prommetrics.OutcomeSuccess, 10*time.Millisecond)
	prometheusMetrics.ObserveTTFT(2 * time.Millisecond)
	prometheusMetrics.UpdateCacheTotals(prommetrics.CacheFlat, 2, 1, 0)

	srv, _ := newTestServer(t, 8192, nil)
	rec := httptest.NewRecorder()
	srv.handleMetricsDispatch(rec, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d want=200", rec.Code)
	}
	if got := rec.Header().Get("Content-Type"); !strings.HasPrefix(got, "text/plain") {
		t.Fatalf("Content-Type=%q, want Prometheus text", got)
	}
	for _, series := range []string{
		"fucina_request_lifecycle_total",
		"fucina_cache_lookups_total",
		"fucina_cache_hits_total",
		"fucina_ttft_seconds_count 1",
	} {
		if !strings.Contains(rec.Body.String(), series) {
			t.Errorf("scrape missing %q", series)
		}
	}
}

func TestBatchRequestEmitsLiveTTFTProducer(t *testing.T) {
	prometheusMetrics = prommetrics.New()
	srv := newQwenBatchServer(t, []int32{17, qImEnd})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":    []map[string]string{{"role": "user", "content": "hi"}},
		"temperature": 0, "max_tokens": 2,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}

	scrape := httptest.NewRecorder()
	srv.handleMetricsDispatch(scrape, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if !strings.Contains(scrape.Body.String(), "fucina_ttft_seconds_count 1") {
		t.Fatalf("batch TTFT producer did not emit:\n%s", scrape.Body.String())
	}
}

func TestMetricsDispatchPreservesJSONAtMetricsPath(t *testing.T) {
	srv, _ := newTestServer(t, 8192, nil)
	for _, request := range []*http.Request{
		httptest.NewRequest(http.MethodGet, "/metrics?format=json", nil),
		httptest.NewRequest(http.MethodGet, "/metrics", nil),
	} {
		request.Header.Set("Accept", "application/json")
		rec := httptest.NewRecorder()
		srv.handleMetricsDispatch(rec, request)
		if got := rec.Header().Get("Content-Type"); got != "application/json" {
			t.Fatalf("Content-Type=%q, want historical JSON", got)
		}
		var body map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
			t.Fatalf("decode historical metrics JSON: %v", err)
		}
		if _, ok := body["context"]; !ok {
			t.Fatalf("historical JSON missing context: %v", body)
		}
	}
}

func TestPrometheusRequestOutcome(t *testing.T) {
	for _, tc := range []struct {
		status int
		want   prommetrics.RequestOutcome
	}{
		{200, prommetrics.OutcomeSuccess},
		{400, prommetrics.OutcomeRejected},
		{499, prommetrics.OutcomeCancelled},
		{500, prommetrics.OutcomeError},
	} {
		if got := prometheusRequestOutcome(tc.status); got != tc.want {
			t.Errorf("status %d: outcome=%q want=%q", tc.status, got, tc.want)
		}
	}
}
