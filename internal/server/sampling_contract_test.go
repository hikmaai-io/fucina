package server

import (
	"encoding/json"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/hikmaai-io/fucina/internal/server/batch"
)

func ptrFloat(v float64) *float64 { return &v }

func TestUnsupportedPenaltyMatrix(t *testing.T) {
	modes := []string{"single-flight", "batch", "speculative", "grammar-constrained"}
	penalties := []struct {
		name      string
		frequency *float64
		presence  *float64
	}{
		{"frequency-penalty", ptrFloat(.5), nil},
		{"presence-penalty", nil, ptrFloat(.5)},
	}
	for _, penalty := range penalties {
		for _, mode := range modes {
			t.Run(penalty.name+"/"+mode, func(t *testing.T) {
				if got := unsupportedPenaltyParameter(penalty.frequency, penalty.presence); got == "" {
					t.Fatal("non-zero unsupported penalty was silently accepted")
				}
			})
		}
	}
	if got := unsupportedPenaltyParameter(ptrFloat(0), ptrFloat(0)); got != "" {
		t.Fatalf("zero no-op penalties rejected as %q", got)
	}
}

func TestUnsupportedParameterEnvelope(t *testing.T) {
	rr := httptest.NewRecorder()
	writeUnsupportedParameter(rr, "frequency_penalty")
	if rr.Code != 400 {
		t.Fatalf("status = %d", rr.Code)
	}
	var body struct {
		Error struct {
			Code  string `json:"code"`
			Param string `json:"param"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Error.Code != "unsupported_parameter" || body.Error.Param != "frequency_penalty" {
		t.Fatalf("error = %+v", body.Error)
	}
}

func TestE4BBatchSamplingEnvelope(t *testing.T) {
	if e4bBatchSamplingUnsupported(batch.GreedyOnlySamplingSupport, GenerationParams{Temperature: 0, RepeatPenalty: 1}) {
		t.Fatal("explicit greedy request rejected")
	}
	if !e4bBatchSamplingUnsupported(batch.GreedyOnlySamplingSupport, GenerationParams{Temperature: .7, RepeatPenalty: 1}) {
		t.Fatal("non-greedy E4B batch request accepted")
	}
	if !repeatPenaltyUnsupported(batch.SamplingSupport{Stochastic: true}, GenerationParams{Temperature: .7, RepeatPenalty: 1.2}) {
		t.Fatal("adapter without repeat support accepted repeat penalty")
	}
	if repeatPenaltyUnsupported(batch.SamplingSupport{Stochastic: true}, GenerationParams{Temperature: 0, RepeatPenalty: 1.2}) {
		t.Fatal("strict greedy's inert repeat penalty was rejected")
	}
	rr := httptest.NewRecorder()
	writeE4BBatchSamplingUnsupported(rr)
	if rr.Code != 400 {
		t.Fatalf("status = %d", rr.Code)
	}
	var body struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Error.Code != "sampling_unsupported_e4b_batch" {
		t.Fatalf("code = %q", body.Error.Code)
	}
}

func TestHTTPRejectsUnsupportedPenaltiesBeforeInference(t *testing.T) {
	if _, ok := reflect.TypeOf(ChatRequest{}).FieldByName("FrequencyPenalty"); !ok {
		t.Skip("requires fucina-swarm/handoff/sol-03/server.patch")
	}
	for _, parameter := range []string{"frequency_penalty", "presence_penalty"} {
		t.Run(parameter, func(t *testing.T) {
			srv, engine := newTestServer(t, 8192, nil)
			req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
				"messages": []map[string]string{{"role": "user", "content": "hi"}},
				parameter:  0.5,
			}))
			rec := httptest.NewRecorder()
			mux(srv).ServeHTTP(rec, req)
			if rec.Code != 400 {
				t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
			}
			if engine.lastPrefillLen != 0 || len(engine.tokens) != 0 {
				t.Fatalf("engine started inference: prefill len=%d tokens=%d", engine.lastPrefillLen, len(engine.tokens))
			}
			var body struct {
				Error struct {
					Code  string `json:"code"`
					Param string `json:"param"`
				} `json:"error"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
				t.Fatal(err)
			}
			if body.Error.Code != "unsupported_parameter" || body.Error.Param != parameter {
				t.Fatalf("error=%+v", body.Error)
			}
		})
	}
}

type greedyOnlyHTTPBatch struct{ *scriptedBatchEngine }

func (e *greedyOnlyHTTPBatch) SamplingSupport() batch.SamplingSupport {
	return batch.GreedyOnlySamplingSupport
}

func TestE4BNonGreedyHTTPErrorBeforeBatchInference(t *testing.T) {
	if _, ok := reflect.TypeOf(Server{}).FieldByName("batchSamplingSupport"); !ok {
		t.Skip("requires fucina-swarm/handoff/sol-03/server.patch")
	}
	srv, _ := newTestServer(t, 8192, nil)
	eng := &greedyOnlyHTTPBatch{newScriptedBatchEngine(nil)}
	if !srv.SetBatchEngine(eng) {
		t.Fatal("SetBatchEngine refused fake greedy-only engine")
	}
	t.Cleanup(srv.scheduler.Shutdown)
	eng.mu.Lock()
	before := eng.next // graph warmup admissions are expected
	eng.mu.Unlock()

	req := httptest.NewRequest("POST", "/v1/chat/completions", chatBody(t, map[string]interface{}{
		"messages":    []map[string]string{{"role": "user", "content": "hi"}},
		"temperature": 0.7,
	}))
	rec := httptest.NewRecorder()
	mux(srv).ServeHTTP(rec, req)
	if rec.Code != 400 {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var body struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body.Error.Code != "sampling_unsupported_e4b_batch" {
		t.Fatalf("code=%q", body.Error.Code)
	}
	eng.mu.Lock()
	after := eng.next
	eng.mu.Unlock()
	if after != before {
		t.Fatalf("request reached batch prefill: admissions before=%d after=%d", before, after)
	}
}
