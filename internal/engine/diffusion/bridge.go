// Package diffusion provides the Go interface to the DiffusionGemma block-diffusion
// CUDA engine (cuda/libdg.a), separate from the autoregressive gemma4 engine.
//
// DiffusionGemma generates whole 256-token blocks by iterative denoising, so the API
// is block-oriented (Generate returns committed token ids), not token-streaming.
package diffusion

// #cgo CFLAGS: -I/usr/local/cuda-13/include -I${SRCDIR}/../../../cuda
// #cgo LDFLAGS: -L${SRCDIR}/../../../cuda -L/usr/local/cuda/lib64 -ldg -lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
// #include "diffusion_gemma_engine.h"
// #include <stdlib.h>
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// IsDiffusion reports whether the GGUF at path is a diffusion-gemma model.
func IsDiffusion(path string) bool {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	return C.dg_gguf_is_diffusion(cp) == 1
}

// Params controls the diffusion denoising loop.
type Params struct {
	MaxSteps     int     // denoising step cap per block (0 → engine default 48)
	TMin         float32 // linear temperature schedule lower bound (final step)
	TMax         float32 // linear temperature schedule upper bound (first step)
	EntropyBound float32 // entropy-bound sampler acceptance budget
	Seed         uint64  // RNG seed
	MaxNewTokens int     // total output-token cap across chained blocks (0 → DefaultMaxNewTokens)
	EOTID        int     // end-of-turn token id that stops block chaining (besides EOS); 0 to disable
}

// DefaultMaxNewTokens caps multi-block generation when the caller gives no explicit limit (the
// model chains 256-token blocks until EOS, so an unbounded answer could fill the whole context).
const DefaultMaxNewTokens = 2048

// DefaultParams mirrors the HF reference generation defaults.
func DefaultParams() Params {
	return Params{MaxSteps: 48, TMin: 0.4, TMax: 0.8, EntropyBound: 0.1, Seed: 0, MaxNewTokens: DefaultMaxNewTokens}
}

// Engine wraps the C diffusion engine.
type Engine struct {
	mu        sync.Mutex
	ptr       *C.dg_engine_t
	canvasLen int
	maxPrompt int
}

// Stats reports timing for the most recent Generate call.
type Stats struct {
	NPrompt   int     // prompt tokens prefilled (cached once)
	NOut      int     // committed output tokens (trimmed at first EOS)
	CanvasLen int     // canvas slots refined every step (the block size, 256)
	Steps     int     // denoising steps actually run
	PrefillMS float64 // prompt-prefill wall time
	DenoiseMS float64 // denoise-loop wall time
}

// PrefillTokPerSec is the prompt prefill rate (0 if no prompt/time).
func (s Stats) PrefillTokPerSec() float64 {
	if s.PrefillMS <= 0 {
		return 0
	}
	return float64(s.NPrompt) / (s.PrefillMS / 1000)
}

// GenTokPerSec is the delivered-token rate: committed answer tokens over the denoise loop.
// This is what a caller experiences, but it under-reports the engine for short answers — each
// step refines the whole canvas regardless of how many tokens land before EOS (see CanvasTokPerSec).
func (s Stats) GenTokPerSec() float64 {
	if s.DenoiseMS <= 0 {
		return 0
	}
	return float64(s.NOut) / (s.DenoiseMS / 1000)
}

// CanvasTokPerSec is the raw denoising throughput: every step the model refines all CanvasLen
// slots, so the engine's true generation rate is CanvasLen×Steps over the denoise loop. For a
// full block this matches GenTokPerSec; for short answers it shows the work actually done.
func (s Stats) CanvasTokPerSec() float64 {
	if s.DenoiseMS <= 0 {
		return 0
	}
	return float64(s.CanvasLen*s.Steps) / (s.DenoiseMS / 1000)
}

// NewEngine loads the GGUF and allocates scratch for prompts up to maxPrompt tokens.
func NewEngine(path string, maxPrompt int, fp4MoE bool) (*Engine, error) {
	if maxPrompt <= 0 {
		maxPrompt = 1024
	}
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	fp4 := C.int(0)
	if fp4MoE {
		fp4 = C.int(1)
	}
	ptr := C.dg_engine_create(cp, C.int(maxPrompt), fp4)
	if ptr == nil {
		return nil, fmt.Errorf("diffusion: failed to create engine from %s", path)
	}
	// Warm up at load time (NVFP4 MoE build + cuBLAS algos + kernel loads) so the first request
	// doesn't stall mid-answer.
	C.dg_engine_warmup(ptr)
	// The engine may cap max_prompt to fit GPU memory; read the ACTUAL value back.
	return &Engine{ptr: ptr, canvasLen: int(C.dg_engine_canvas_length(ptr)), maxPrompt: int(C.dg_engine_max_prompt(ptr))}, nil
}

// CanvasLength is the diffusion block size (256).
func (e *Engine) CanvasLength() int { return e.canvasLen }

// MaxPrompt is the largest prompt length the engine's scratch is sized for.
func (e *Engine) MaxPrompt() int { return e.maxPrompt }

// Generate denoises one block conditioned on prompt and returns the committed
// argmax token ids (trimmed at the first EOS) plus timing stats. Up to CanvasLength tokens.
func (e *Engine) Generate(prompt []int32, p Params) ([]int32, Stats, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.ptr == nil {
		return nil, Stats{}, fmt.Errorf("diffusion: engine closed")
	}
	if len(prompt) == 0 {
		return nil, Stats{}, fmt.Errorf("diffusion: empty prompt")
	}
	if len(prompt) > e.maxPrompt {
		return nil, Stats{}, fmt.Errorf("diffusion: prompt %d exceeds max_prompt %d", len(prompt), e.maxPrompt)
	}
	// Size the output buffer for multi-block generation: at least one canvas, capped by the
	// caller's MaxNewTokens and the room left in the context (cache can't exceed maxPrompt).
	maxOut := p.MaxNewTokens
	if maxOut <= 0 {
		maxOut = DefaultMaxNewTokens
	}
	if room := e.maxPrompt - len(prompt); maxOut > room {
		maxOut = room
	}
	if maxOut < e.canvasLen {
		maxOut = e.canvasLen // always allow at least one full block
	}
	out := make([]int32, maxOut)
	n := C.dg_engine_generate(e.ptr,
		(*C.int32_t)(unsafe.Pointer(&prompt[0])), C.int(len(prompt)),
		C.int(p.MaxSteps), C.float(p.TMin), C.float(p.TMax), C.float(p.EntropyBound),
		C.uint64_t(p.Seed), C.int(p.EOTID),
		(*C.int32_t)(unsafe.Pointer(&out[0])), C.int(len(out)))
	if n < 0 {
		return nil, Stats{}, fmt.Errorf("diffusion: generate failed (rc=%d)", int(n))
	}
	var pf, dn C.float
	var steps C.int
	C.dg_engine_last_stats(e.ptr, &pf, &dn, &steps)
	st := Stats{
		NPrompt: len(prompt), NOut: int(n), CanvasLen: e.canvasLen, Steps: int(steps),
		PrefillMS: float64(pf), DenoiseMS: float64(dn),
	}
	return out[:int(n)], st, nil
}

// Close frees the engine.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.ptr != nil {
		C.dg_engine_free(e.ptr)
		e.ptr = nil
	}
}
