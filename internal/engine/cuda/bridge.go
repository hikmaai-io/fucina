// Package cuda provides the Go interface to the CUDA inference engine.
//
// This package uses CGO to call the C CUDA kernels compiled in libgem4d.a.
// The engine is specific to Gemma 4 12B on DGX Spark GB10 (sm_121, CUDA 13.0).

package cuda

// #cgo LDFLAGS: -L${SRCDIR}/../../../cuda -lgem4d -lcudart -lcublas -lcuda -lpthread -lstdc++ -lm
// #cgo CFLAGS: -I/usr/local/cuda-13/include -I${SRCDIR}/../../../cuda
//
// #include "gemma4_kernels.cuh"
// #include <stdlib.h>
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// TensorFormat represents the weight storage format (C enum).
type TensorFormat int

const (
	FormatFP8  TensorFormat = 0 // C.FORMAT_FP8
	FormatQ8_0 TensorFormat = 1 // C.FORMAT_Q8_0
)

// Engine wraps the C inference engine.
type Engine struct {
	mu   sync.Mutex
	ptr  *C.gemma4_engine_t
	ctx  uint32
	path string
	fmt  TensorFormat
}

// Config holds engine configuration matching llama.cpp-style CLI flags.
type Config struct {
	ModelPath   string       // -m, --model
	LoraPath    string       // --lora-scaled
	LoraScale   float64      // optional scale for LoRA
	Format      TensorFormat // --memory-format (fp8 or q8_0)
	ContextSize uint32       // --ctx
	DeviceID    int          // --cuda-device
}

// NewEngine creates and initializes the CUDA inference engine.
func NewEngine(cfg Config) (*Engine, error) {
	cPath := C.CString(cfg.ModelPath)
	defer C.free(unsafe.Pointer(cPath))

	// Default context to model max if not set
	ctxSize := cfg.ContextSize
	if ctxSize == 0 {
		ctxSize = 8192 // sensible default
	}
	if ctxSize > 262144 {
		ctxSize = 262144
	}

	ptr := C.gemma4_engine_create(
		cPath,
		(C.tensor_format_t)(cfg.Format), // cast tensor_format_t enum
		C.uint32_t(ctxSize),
		C.int(cfg.DeviceID),
	)
	if ptr == nil {
		return nil, fmt.Errorf("gem4d: engine creation failed for %s", cfg.ModelPath)
	}

	eng := &Engine{
		ptr:  ptr,
		ctx:  ctxSize,
		path: cfg.ModelPath,
		fmt:  cfg.Format,
	}

	// Load LoRA if specified
	if cfg.LoraPath != "" {
		scale := float32(cfg.LoraScale)
		if scale == 0 {
			scale = 1.0
		}
		cLora := C.CString(cfg.LoraPath)
		ret := C.gemma4_engine_load_lora(ptr, cLora, C.float(scale))
		C.free(unsafe.Pointer(cLora))
		if ret != 0 {
			eng.Close()
			return nil, fmt.Errorf("gem4d: LoRA loading failed for %s", cfg.LoraPath)
		}
	}

	return eng, nil
}

// Close destroys the engine and frees all GPU resources.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.ptr != nil {
		C.gemma4_engine_destroy(e.ptr)
		e.ptr = nil
	}
}

// Info returns a formatted string with engine information.
func (e *Engine) Info() string {
	return fmt.Sprintf("%s (%d ctx, format=%d, device=%d)",
		e.path, e.ctx, e.fmt, 0)
}

// PrintInfo prints engine configuration to stderr.
func (e *Engine) PrintInfo() {
	e.mu.Lock()
	defer e.mu.Unlock()
	C.gemma4_engine_print_info(e.ptr)
}

// PrintTiming prints accumulated timing statistics.
func (e *Engine) PrintTiming() {
	e.mu.Lock()
	defer e.mu.Unlock()
	C.gemma4_engine_print_timing(e.ptr)
}

// HasLoRA reports whether the engine has loaded LoRA adapters.
func (e *Engine) HasLoRA() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return C.gemma4_engine_has_lora(e.ptr) != 0
}

// Prefill processes a batch of tokens (sequential) and fills the KV cache.
func (e *Engine) Prefill(tokens []int32) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if len(tokens) == 0 {
		return nil, nil
	}

	// Only get logits for the last token
	logits := make([]float32, 262144)

	// Fast path: batched BF16 tensor-core prefill (one weight pass for the whole
	// prompt). Returns -2 when not applicable (e.g. the KV cache is not empty), in
	// which case we fall back to the proven token-by-token path. Both produce the
	// same last-token logits (parity-verified).
	ret := C.gemma4_engine_prefill_batched(
		e.ptr,
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		C.int(len(tokens)),
		(*C.float)(unsafe.Pointer(&logits[0])),
	)
	if ret == -2 {
		ret = C.gemma4_engine_prefill(
			e.ptr,
			(*C.int32_t)(unsafe.Pointer(&tokens[0])),
			C.int(len(tokens)),
			(*C.float)(unsafe.Pointer(&logits[0])),
		)
	}
	if ret != 0 {
		return nil, fmt.Errorf("gem4d: prefill failed")
	}

	return logits, nil
}

// Decode processes a single token and returns logits.
func (e *Engine) Decode(token int32) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	logits := make([]float32, 262144)

	ret := C.gemma4_engine_decode(
		e.ptr,
		C.int32_t(token),
		(*C.float)(unsafe.Pointer(&logits[0])),
	)
	if ret != 0 {
		return nil, fmt.Errorf("gem4d: decode failed")
	}

	return logits, nil
}

// GenerateSpec runs greedy generation with prompt-lookup speculative decoding.
// It prefills `prompt` internally and generates up to maxNew tokens, stopping at
// any id in stops. Returns the generated tokens and the total number of drafts
// accepted (for measuring the acceptance rate). Greedy/argmax only — produces the
// exact same tokens as a plain greedy decode, just faster on context-reusing text.
func (e *Engine) GenerateSpec(prompt []int32, maxNew int, stops []int32, draftK int,
	temp float32, topK int, topP, minP float32, seed uint64) ([]int32, int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	out := make([]int32, maxNew)
	var nacc C.int
	var stopsPtr *C.int32_t
	if len(stops) > 0 {
		stopsPtr = (*C.int32_t)(unsafe.Pointer(&stops[0]))
	}
	ng := C.gemma4_engine_generate_spec(
		e.ptr,
		(*C.int32_t)(unsafe.Pointer(&prompt[0])), C.int(len(prompt)),
		(*C.int32_t)(unsafe.Pointer(&out[0])), C.int(maxNew),
		stopsPtr, C.int(len(stops)),
		C.int(draftK),
		C.float(temp), C.int(topK), C.float(topP), C.float(minP), C.uint64_t(seed),
		&nacc,
	)
	if ng < 0 {
		return nil, 0, fmt.Errorf("gem4d: generate_spec failed")
	}
	return out[:ng], int(nacc), nil
}

// DecodeNoCopy decodes a single token but leaves the logits on the GPU (no 262k
// D2H). Pair it with SampleDevice, which selects the next token on-device.
func (e *Engine) DecodeNoCopy(token int32) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if C.gemma4_engine_decode(e.ptr, C.int32_t(token), nil) != 0 {
		return fmt.Errorf("gem4d: decode failed")
	}
	return nil
}

// SampleDevice selects the next token from the engine's resident logits entirely on
// the GPU (temp<=0 → argmax, else temperature/top-k/top-p/min-p/multinomial), so only
// the 4-byte id crosses to host. rnd must be in [0,1). Repeat penalty is not applied
// here — callers using it must fall back to the host sampler.
func (e *Engine) SampleDevice(temp float32, topK int, topP, minP, rnd float32) (int32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	id := C.gemma4_engine_sample_device(e.ptr, C.float(temp), C.int(topK),
		C.float(topP), C.float(minP), C.float(rnd))
	if id < 0 {
		return 0, fmt.Errorf("gem4d: device sample failed")
	}
	return int32(id), nil
}

// Argmax returns the index of the highest value in logits.
func Argmax(logits []float32) int {
	return int(C.gemma4_sample_argmax(
		(*C.float)(unsafe.Pointer(&logits[0])),
		C.int(len(logits)),
	))
}

// ContextSize returns the engine's configured context window.
func (e *Engine) ContextSize() uint32 {
	return e.ctx
}

// NTokens returns how many tokens are currently materialized in the KV cache.
// This is the position at which the next Prefill/Decode will append.
func (e *Engine) NTokens() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return int(C.gemma4_engine_n_tokens(e.ptr))
}

// Reset rewinds the KV cache to empty so the next Prefill starts a fresh
// sequence at position 0. It does not free device memory.
func (e *Engine) Reset() {
	e.mu.Lock()
	defer e.mu.Unlock()
	C.gemma4_engine_reset(e.ptr)
}

// Rewind keeps only the first nKeep tokens of the KV cache, discarding the
// rest, so a shared prefix can be reused and only the divergent suffix
// re-prefilled. Returns true on success; false means the rewind is unsafe
// (e.g. the sliding window wrapped past the kept prefix) and the caller should
// Reset() and re-prefill from scratch.
func (e *Engine) Rewind(nKeep int) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return C.gemma4_engine_rewind(e.ptr, C.int(nKeep)) == 0
}

// TimingStats holds accumulated prefill/decode timing for speed reporting.
type TimingStats struct {
	PrefillTokens int
	PrefillMS     float64
	DecodeTokens  int
	DecodeMS      float64
}

// PrefillTokensPerSec returns prefill throughput in tokens/second (0 if none).
func (t TimingStats) PrefillTokensPerSec() float64 {
	if t.PrefillMS <= 0 {
		return 0
	}
	return float64(t.PrefillTokens) / (t.PrefillMS / 1000.0)
}

// DecodeTokensPerSec returns generation throughput in tokens/second (0 if none).
func (t TimingStats) DecodeTokensPerSec() float64 {
	if t.DecodeMS <= 0 {
		return 0
	}
	return float64(t.DecodeTokens) / (t.DecodeMS / 1000.0)
}

// Timing returns accumulated prefill/decode timing statistics from the engine.
func (e *Engine) Timing() TimingStats {
	e.mu.Lock()
	defer e.mu.Unlock()
	return TimingStats{
		PrefillTokens: int(C.gemma4_engine_prefill_tokens(e.ptr)),
		PrefillMS:     float64(C.gemma4_engine_prefill_ms(e.ptr)),
		DecodeTokens:  int(C.gemma4_engine_decode_tokens(e.ptr)),
		DecodeMS:      float64(C.gemma4_engine_decode_ms(e.ptr)),
	}
}

// ensure CGO runs on the main thread for CUDA compatibility
func init() {
	runtime.LockOSThread()
}
