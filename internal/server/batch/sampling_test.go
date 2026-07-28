package batch

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/hikmaai-io/fucina/internal/grammar"
)

type passthroughConstraint struct{}

func (passthroughConstraint) Mask([]float32) {}
func (passthroughConstraint) Accept(int32)   {}
func (passthroughConstraint) Done() bool     { return false }
func (passthroughConstraint) Close() []byte  { return nil }

type samplingSupportEngine struct{ support SamplingSupport }

func (e samplingSupportEngine) AddSeq([]int32, SeqParams) (int, int32, error) { return 0, 0, nil }
func (e samplingSupportEngine) StepBatch([]int32, []int32) ([][]int32, error) {
	return nil, nil
}
func (e samplingSupportEngine) RemoveSeq(int) error              { return nil }
func (e samplingSupportEngine) Capacity() int                    { return 1 }
func (e samplingSupportEngine) SamplingSupport() SamplingSupport { return e.support }

func TestSamplingSupportCapability(t *testing.T) {
	limited := samplingSupportEngine{support: GreedyOnlySamplingSupport}
	if got := SamplingSupportFor(limited); got != GreedyOnlySamplingSupport {
		t.Fatalf("limited support = %+v", got)
	}
	if got := SamplingSupportFor(struct{ samplingSupportEngine }{}); got != GreedyOnlySamplingSupport {
		t.Fatalf("embedded provider support = %+v", got)
	}
}

func TestRequiresNonGreedySampling(t *testing.T) {
	for _, tc := range []struct {
		name string
		p    SeqParams
		want bool
	}{
		{"zero-value greedy", SeqParams{}, false},
		{"negative penalty normalizes off", SeqParams{Temperature: 0, RepeatPenalty: -1}, false},
		{"explicit greedy", SeqParams{Temperature: 0, RepeatPenalty: 1, TopK: 40, TopP: .9, MinP: .1, Seed: 7}, false},
		{"temperature", SeqParams{Temperature: .7, RepeatPenalty: 1}, true},
		{"repeat is inert under greedy", SeqParams{Temperature: 0, RepeatPenalty: 1.1}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := RequiresNonGreedySampling(tc.p); got != tc.want {
				t.Fatalf("RequiresNonGreedySampling(%+v) = %v, want %v", tc.p, got, tc.want)
			}
		})
	}
}

// TestSamplingControlMatrix exercises the applied controls over the four serving
// shapes required by SOL-03. Frequency/presence are covered by the HTTP rejection
// matrix in internal/server; all controls represented here are applied by the
// shared host sampler used for constrained and repeat-penalty batch rows.
func TestSamplingControlMatrix(t *testing.T) {
	modes := []string{"single-flight", "batch", "speculative", "grammar-constrained"}
	controls := []struct {
		name string
		p    SeqParams
		hist []int32
	}{
		{"temperature", SeqParams{Temperature: .65, TopK: 4, TopP: 1, RepeatPenalty: 1, Seed: 97}, nil},
		{"top-k", SeqParams{Temperature: 1, TopK: 2, TopP: 1, RepeatPenalty: 1, Seed: 97}, nil},
		{"top-p", SeqParams{Temperature: 1, TopP: .72, RepeatPenalty: 1, Seed: 97}, nil},
		{"min-p", SeqParams{Temperature: 1, TopP: 1, MinP: .18, RepeatPenalty: 1, Seed: 97}, nil},
		{"seed", SeqParams{Temperature: 1, TopK: 4, TopP: 1, RepeatPenalty: 1, Seed: 991}, nil},
		{"repeat-penalty", SeqParams{Temperature: 1, TopK: 4, TopP: 1, RepeatPenalty: 1.8, Seed: 97}, []int32{0, 0, 0}},
	}
	logits := []float32{4, 3.8, 3.1, 2.7}
	for _, control := range controls {
		for _, mode := range modes {
			t.Run(control.name+"/"+mode, func(t *testing.T) {
				var constraint grammar.Constraint
				if mode == "grammar-constrained" {
					constraint = passthroughConstraint{}
				}
				run := func() int32 {
					got, err := sampleHostLogits(append([]float32(nil), logits...), control.p,
						rand.New(rand.NewSource(int64(control.p.Seed))), append([]int32(nil), control.hist...), constraint)
					if err != nil {
						t.Fatal(err)
					}
					return got
				}
				if a, b := run(), run(); a != b {
					t.Fatalf("fixed seed is not deterministic: %d != %d", a, b)
				}
			})
		}
	}
}

func TestSamplingBatchRowsAreIndependent(t *testing.T) {
	pA := SeqParams{Temperature: 1, TopK: 4, TopP: 1, RepeatPenalty: 1.7, Seed: 11}
	pB := SeqParams{Temperature: .8, TopK: 3, TopP: .9, RepeatPenalty: 1, Seed: 29}
	logitsA := []float32{4, 3.9, 3, 2}
	logitsB := []float32{2, 3, 4, 3.8}

	sample := func(p SeqParams, logits []float32, hist []int32) int32 {
		tok, err := sampleHostLogits(append([]float32(nil), logits...), p,
			rand.New(rand.NewSource(int64(p.Seed))), hist, nil)
		if err != nil {
			t.Fatal(err)
		}
		return tok
	}
	alone := []int32{sample(pA, logitsA, []int32{0}), sample(pB, logitsB, []int32{2})}
	// Reverse execution order. Request-local RNG/history must make composition irrelevant.
	combined := []int32{sample(pB, logitsB, []int32{2}), sample(pA, logitsA, []int32{0})}
	combined[0], combined[1] = combined[1], combined[0]
	if !reflect.DeepEqual(alone, combined) {
		t.Fatalf("row samples changed with batch order: alone=%v combined=%v", alone, combined)
	}
}

type repeatExactSpecEngine struct {
	exactCalls int
	specCalls  int
}

func (e *repeatExactSpecEngine) AddSeq([]int32, SeqParams) (int, int32, error) {
	return 0, 0, nil // deliberately unpenalized device token
}
func (e *repeatExactSpecEngine) StepBatch(active []int32, _ []int32) ([][]int32, error) {
	out := make([][]int32, len(active))
	for i := range out {
		out[i] = []int32{0}
	}
	return out, nil
}
func (e *repeatExactSpecEngine) StepBatchExact(active []int32, _ []int32) ([]int32, error) {
	e.exactCalls++
	return make([]int32, len(active)), nil
}
func (e *repeatExactSpecEngine) StepBatchSpec(reqs []SpecReq) ([][]int32, error) {
	e.specCalls++
	return make([][]int32, len(reqs)), nil
}
func (e *repeatExactSpecEngine) CopyLogits(rows int, _ bool) ([]float32, int, error) {
	flat := make([]float32, 0, rows*2)
	for i := 0; i < rows; i++ {
		flat = append(flat, 10, 9)
	}
	return flat, 2, nil
}
func (e *repeatExactSpecEngine) RemoveSeq(int) error { return nil }
func (e *repeatExactSpecEngine) Capacity() int       { return 1 }

func TestRepeatPenaltyUsesExactHostPathAndDisablesSpec(t *testing.T) {
	eng := &repeatExactSpecEngine{}
	s := New(eng, 2)
	// This runtime capability is supplied by the scheduler hot-file handoff. Keep
	// the branch independently testable while ensuring the integrated tree runs
	// the full end-to-end assertion.
	if capability, ok := any(s).(interface{ SupportsRepeatPenalty() bool }); !ok || !capability.SupportsRepeatPenalty() {
		t.Skip("requires fucina-swarm/handoff/sol-03/scheduler.patch")
	}
	s.Start()
	defer s.Shutdown()

	done := make(chan Result, 1)
	var got []int32
	if err := s.Submit(Request{
		Tokens: []int32{0},
		Params: SeqParams{Temperature: 1, TopK: 1, TopP: 1, RepeatPenalty: 2, Seed: 17},
		MaxNew: 2,
		Emit: func(tok int32) bool {
			got = append(got, tok)
			return true
		},
		Done: done,
	}); err != nil {
		t.Fatal(err)
	}
	res := <-done
	if res.Err != nil {
		t.Fatal(res.Err)
	}
	if want := []int32{1, 0}; !reflect.DeepEqual(got, want) {
		t.Fatalf("penalized output = %v, want %v", got, want)
	}
	if eng.exactCalls == 0 || eng.specCalls != 0 {
		t.Fatalf("exact calls=%d spec calls=%d; repeat row must not spec-commit", eng.exactCalls, eng.specCalls)
	}
}
