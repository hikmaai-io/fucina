package server

import "testing"

// TestBatchIneligible verifies the continuous-batching route-guard: requests using
// features serveBatch cannot honor (grammar/JSON constraint, custom stop strings,
// repeat_penalty) are flagged ineligible; plain sampling requests are eligible.
func TestBatchIneligible(t *testing.T) {
	cases := []struct {
		name string
		p    GenerationParams
		want bool // true == ineligible (reason non-empty)
	}{
		{"plain greedy", GenerationParams{RepeatPenalty: 1.0}, false},
		{"plain temp", GenerationParams{Temperature: 0.8, TopK: 40, TopP: 0.9, RepeatPenalty: 1.0}, false},
		{"repeat_penalty default 0", GenerationParams{RepeatPenalty: 0}, false},
		{"repeat_penalty active", GenerationParams{RepeatPenalty: 1.3}, true},
		{"custom stop", GenerationParams{RepeatPenalty: 1.0, Stop: []string{"\n\n"}}, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := batchIneligible(c.p) != ""
			if got != c.want {
				t.Fatalf("batchIneligible(%+v) ineligible=%v, want %v (reason=%q)",
					c.p, got, c.want, batchIneligible(c.p))
			}
		})
	}
}
