// fucina-calibrate parser and sidecar policy unit tests.
package main

import (
	"strings"
	"testing"

	"github.com/hikmaai-io/fucina/internal/engine/cuda"
)

func TestCorpusText(t *testing.T) {
	cases := []struct{ in, want string }{
		{"plain source code", "plain source code"},
		{`{"text":"hello"}`, "hello"},
		{`{"prompt":"audit this"}`, "audit this"},
		{`{"messages":[{"role":"user","content":"find bug"},{"role":"assistant","content":"checking"}]}`, "find bug\nchecking"},
	}
	for _, tc := range cases {
		if got := corpusText([]byte(tc.in)); got != tc.want {
			t.Errorf("corpusText(%q)=%q want %q", tc.in, got, tc.want)
		}
	}
}

func TestBuildSidecarImportanceAndTensorKeys(t *testing.T) {
	p := cuda.MoEProfile{Layers: 1, Experts: 2, TopK: 1, Counts: []uint64{9, 1}, WeightSums: []float64{4.5, 0.9}}
	o := buildSidecar("m", "c", 1, 10, p)
	if o.Format != "fucina-imatrix-v1" || len(o.HeatMap) != 1 {
		t.Fatalf("bad metadata: %+v", o)
	}
	if o.HeatMap[0].Experts[0].Expert != 0 {
		t.Fatalf("heat map not sorted hot-first: %+v", o.HeatMap[0])
	}
	hot := o.TensorImportance["layers.0.mlp.experts.0.gate_proj.weight"]
	cold := o.TensorImportance["layers.0.mlp.experts.1.gate_proj.weight"]
	if hot <= cold {
		t.Fatalf("hot importance %f <= cold %f", hot, cold)
	}
	for k, v := range o.TensorImportance {
		if strings.Contains(k, "shared_expert") && v != 1 {
			t.Fatalf("shared expert %s importance=%f", k, v)
		}
	}
}
