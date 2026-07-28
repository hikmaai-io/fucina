package server

import (
	"fmt"
	"net/http"

	"github.com/hikmaai-io/fucina/internal/server/batch"
)

// unsupportedPenaltyParameter implements the Milestone-1 decision for OpenAI
// frequency/presence penalties: the fields are accepted in the request schema,
// but non-zero values are rejected explicitly because no engine applies them
// yet. Zero is a supported no-op and preserves client compatibility.
func unsupportedPenaltyParameter(frequency, presence *float64) string {
	if frequency != nil && *frequency != 0 {
		return "frequency_penalty"
	}
	if presence != nil && *presence != 0 {
		return "presence_penalty"
	}
	return ""
}

func writeUnsupportedParameter(w http.ResponseWriter, parameter string) {
	writeJSON(w, http.StatusBadRequest, map[string]interface{}{
		"error": map[string]interface{}{
			"message": fmt.Sprintf("parameter %q is not supported by this engine", parameter),
			"type":    "invalid_request_error",
			"param":   parameter,
			"code":    "unsupported_parameter",
		},
	})
}

func writeE4BBatchSamplingUnsupported(w http.ResponseWriter) {
	writeJSON(w, http.StatusBadRequest, map[string]interface{}{
		"error": map[string]interface{}{
			"message": "non-greedy sampling is not supported by the E4B batch engine",
			"type":    "invalid_request_error",
			"code":    "sampling_unsupported_e4b_batch",
		},
	})
}

func e4bBatchSamplingUnsupported(support batch.SamplingSupport, params GenerationParams) bool {
	if support.Stochastic {
		return false
	}
	return batch.RequiresNonGreedySampling(batch.SeqParams{
		Temperature:   float32(params.Temperature),
		TopK:          params.TopK,
		TopP:          float32(params.TopP),
		MinP:          float32(params.MinP),
		RepeatPenalty: float32(params.RepeatPenalty),
		Seed:          uint64(params.Seed),
	})
}

func repeatPenaltyUnsupported(support batch.SamplingSupport, params GenerationParams) bool {
	if support.RepeatPenalty || params.Temperature <= 0 {
		return false
	}
	return batch.HasRepeatPenalty(batch.SeqParams{RepeatPenalty: float32(params.RepeatPenalty)})
}
