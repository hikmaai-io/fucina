package batch

import (
	"errors"
	"math"
	"math/rand"

	"github.com/hikmaai-io/fucina/internal/grammar"
	"github.com/hikmaai-io/fucina/internal/sampler"
)

// SamplingSupport describes controls a concrete batch adapter actually applies.
// The default is the main Gemma/Qwen adapter. Limited adapters (currently E4B)
// opt out explicitly through SamplingSupportProvider.
type SamplingSupport struct {
	Stochastic    bool
	RepeatPenalty bool
}

// MainSamplingSupport is the production Gemma/Qwen contract. Repeat penalty is
// host-sampled from exact per-row logits when it is non-default, so it does not
// require a decode-kernel change.
var MainSamplingSupport = SamplingSupport{Stochastic: true, RepeatPenalty: true}

// GreedyOnlySamplingSupport is the temporary E4B batch contract until SOL-09.
var GreedyOnlySamplingSupport = SamplingSupport{}

// SamplingSupportProvider lets adapters publish limitations without coupling
// the scheduler or HTTP server to a model package.
type SamplingSupportProvider interface {
	SamplingSupport() SamplingSupport
}

// SamplingSupportFor returns the adapter's declared contract. Existing main
// adapters predate the capability interface and therefore retain the complete
// main-engine contract by default; limited adapters must opt out explicitly.
func SamplingSupportFor(engine BatchEngine) SamplingSupport {
	if provider, ok := engine.(SamplingSupportProvider); ok {
		return provider.SamplingSupport()
	}
	return MainSamplingSupport
}

// HasRepeatPenalty reports whether p requests a real repeat penalty. A zero
// value is treated as the historical/default 1.0 so zero-value SeqParams used
// by scheduler warmups and tests stay greedy-compatible.
func HasRepeatPenalty(p SeqParams) bool {
	return p.RepeatPenalty > 0 && p.RepeatPenalty != 1
}

// RequiresNonGreedySampling reports whether a greedy-only adapter would drop a
// requested control. Temperature <= 0 follows the shared sampler's strict argmax
// short-circuit, so top-k/top-p/min-p, seed, and penalties are intentionally inert.
func RequiresNonGreedySampling(p SeqParams) bool {
	return p.Temperature > 0
}

// requiresHostSampling is the main-engine fallback for controls that cannot be
// applied by the paged device sampler without a kernel change. Grammar masking
// already uses the same exact-logit path.
func requiresHostSampling(req Request) bool {
	return req.Constraint != nil || HasRepeatPenalty(req.Params)
}

// sampleHostLogits applies the complete host-side sampling contract to one row.
// It is shared by grammar-constrained and repeat-penalty batch rows. logits is
// request-local (CopyLogits returns a fresh slice), so masking it in place is safe.
func sampleHostLogits(logits []float32, params SeqParams, rng *rand.Rand, history []int32, constraint grammar.Constraint) (int32, error) {
	if constraint != nil {
		constraint.Mask(logits)
	}
	legal := false
	for _, v := range logits {
		if !math.IsInf(float64(v), -1) && !math.IsNaN(float64(v)) {
			legal = true
			break
		}
	}
	if !legal {
		return 0, errors.New("sampling mask removed every token")
	}
	repeat := float64(params.RepeatPenalty)
	if repeat == 0 {
		repeat = 1
	}
	return sampler.Sample(logits, sampler.Params{
		Temperature:   float64(params.Temperature),
		TopK:          params.TopK,
		TopP:          float64(params.TopP),
		MinP:          float64(params.MinP),
		RepeatPenalty: repeat,
	}, rng, history)
}
