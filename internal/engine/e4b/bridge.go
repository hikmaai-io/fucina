// Package e4b is the Go bridge to the standalone Gemma-4-E4B CUDA engine.
//
// E4B is the on-device member of the Gemma-4 family (Per-Layer Embeddings,
// KV-cache sharing, runtime dims). It has its own C ABI (cuda/e4b_engine.h),
// distinct from the dense gemma4 engine in internal/engine/cuda, so it gets its
// own bridge package. The two share libfucina.a but no symbols.
//
// Bring-up scope: BF16 weights resident, FP8 Per-Layer-Embedding index, single
// stream. Prefill/Decode return the last-token softcapped logits to the host so
// the existing Go sampler (internal/sampler) drives token selection — giving the
// E4B path full temp/top-k/top-p/min-p/repeat-penalty parity without any
// engine-side sampler. No MTP/speculation here yet.
package e4b

// #cgo CFLAGS: -I/usr/local/cuda-13/include -I${SRCDIR}/../../../cuda
// #cgo LDFLAGS: -L${SRCDIR}/../../../cuda -L/usr/local/cuda/lib64 -lfucina -lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
//
// #include "e4b_engine.h"
// #include <stdlib.h>
//
// // Per-token streaming bridge: fucinaE4BSpecTokenGo is the cgo-exported Go callback
// // (defined in callback.go); the spec loop invokes it once per committed token. The
// // uintptr is a runtime/cgo.Handle resolving to the request's emit closure.
// extern int fucinaE4BSpecTokenGo(int32_t tok, void *ud);
// static inline int _e4b_spec_stream(e4b_engine_t *eng, const int32_t *hist, int n_hist,
//     const float *first_logits, int32_t *out, int max_new, const int32_t *stops, int n_stop,
//     uintptr_t handle) {
//     return e4b_engine_spec_stream(eng, hist, n_hist, first_logits, out, max_new,
//         stops, n_stop, fucinaE4BSpecTokenGo, (void *)handle);
// }
import "C"

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime/cgo"
	"strings"
	"sync"
	"unsafe"
)

// LooksLikeCheckpoint is a cheap, cgo-free pre-filter: a directory, a single
// .safetensors, an .index.json, or a .gguf could be an E4B checkpoint. For
// safetensors/dir it lets the caller avoid handing arbitrary files to IsE4B
// (which mmaps through the safetensors loader); for .gguf the C detector now
// does a lightweight mmap+header scan (no full parse), so a .gguf is allowed
// through here — IsE4B then returns false for a dense (non-PLE) gemma4 GGUF.
func LooksLikeCheckpoint(path string) bool {
	if fi, err := os.Stat(path); err == nil && fi.IsDir() {
		return true
	}
	p := strings.ToLower(path)
	return strings.HasSuffix(p, ".safetensors") ||
		strings.HasSuffix(p, ".index.json") ||
		strings.HasSuffix(p, ".gguf")
}

// IsE4B reports whether the checkpoint at path is a Gemma-4-E4B text model. For
// safetensors it parses config.json (Per-Layer-Embedding + KV-sharing markers);
// for a .gguf it scans the header for general.architecture=gemma4 PLUS the
// per_layer_token_embd.weight tensor and gemma4.attention.shared_kv_layers — an
// unambiguous E4B-vs-dense discriminator. Guard with LooksLikeCheckpoint first.
func IsE4B(path string) bool {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	return C.e4b_is_e4b_checkpoint(cp) == 1
}

// Engine wraps the C e4b_engine. All calls are serialized: the engine is a
// single-stream, single-sequence decoder in this bring-up.
type Engine struct {
	mu     sync.Mutex
	ptr    *C.e4b_engine_t
	vocab  int
	ctx    uint32
	logits []float32 // reusable last-token logits buffer (vocab floats)
}

// New creates an E4B engine from a safetensors checkpoint: parse config, upload
// language_model weights to device as BF16, quantize the PLE table to FP8.
//
// maxSeqs is the desired concurrent-sequence count for continuous batching
// (clamped engine-side to [1,8]); pass 1 for single-stream. The engine queries
// free device memory after loading and may AUTO-SHRINK ctx and/or maxSeqs to fit
// (also honoring FUCINA_MEM_BUDGET_GB) — ContextSize() reports the value actually
// provisioned, not the request.
func New(path string, ctxSize uint32, maxSeqs int, deviceID int) (*Engine, error) {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	ptr := C.e4b_engine_create(cp, C.uint32_t(ctxSize), C.int(maxSeqs), C.int(deviceID))
	if ptr == nil {
		return nil, fmt.Errorf("e4b: engine create failed for %s", path)
	}
	v := int(C.e4b_engine_vocab_size(ptr))
	if v <= 0 {
		C.e4b_engine_destroy(ptr)
		return nil, fmt.Errorf("e4b: bad vocab size %d", v)
	}
	// The engine may have auto-shrunk ctx to fit device memory; report the real cap
	// so the server never admits a prompt longer than the KV cache can hold.
	actualCtx := uint32(C.e4b_engine_max_ctx(ptr))
	if actualCtx == 0 {
		actualCtx = ctxSize
	}
	return &Engine{ptr: ptr, vocab: v, ctx: actualCtx, logits: make([]float32, v)}, nil
}

// Close destroys the engine and frees device memory. Idempotent.
func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.ptr != nil {
		C.e4b_engine_destroy(e.ptr)
		e.ptr = nil
	}
}

func (e *Engine) VocabSize() int    { return e.vocab }
func (e *Engine) ContextSize() uint32 { return e.ctx }
func (e *Engine) NLayers() int      { return int(C.e4b_engine_n_layers(e.ptr)) }
func (e *Engine) HiddenSize() int   { return int(C.e4b_engine_hidden_size(e.ptr)) }
func (e *Engine) DeviceBytes() uint64 { return uint64(C.e4b_engine_device_bytes(e.ptr)) }

// NPast returns the number of tokens currently in the KV cache.
func (e *Engine) NPast() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return int(C.e4b_engine_n_past(e.ptr))
}

// Reset rewinds the KV cache to empty (n_past = 0).
func (e *Engine) Reset() {
	e.mu.Lock()
	defer e.mu.Unlock()
	C.e4b_engine_reset(e.ptr)
}

// Prefill processes the whole prompt into a FRESH KV cache and returns the last
// token's softcapped logits. The slice is valid only until the next
// Prefill/Decode call (it is a reused buffer).
func (e *Engine) Prefill(tokens []int32) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("e4b: empty prompt")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	rc := C.e4b_engine_prefill(e.ptr,
		(*C.int32_t)(unsafe.Pointer(&tokens[0])), C.int(len(tokens)),
		(*C.float)(unsafe.Pointer(&e.logits[0])))
	if rc != 0 {
		return nil, fmt.Errorf("e4b: prefill failed (rc=%d)", int(rc))
	}
	return e.logits, nil
}

// PrefillAppend prefills `suffix` at the engine's CURRENT n_past (set by a prior
// Rewind) instead of resetting, returning the last token's logits. The server KVCache
// uses this to prefill only the divergent suffix after reusing a shared prefix.
func (e *Engine) PrefillAppend(suffix []int32) ([]float32, error) {
	if len(suffix) == 0 {
		return nil, fmt.Errorf("e4b: empty suffix")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	rc := C.e4b_engine_prefill_append(e.ptr,
		(*C.int32_t)(unsafe.Pointer(&suffix[0])), C.int(len(suffix)),
		(*C.float)(unsafe.Pointer(&e.logits[0])))
	if rc != 0 {
		return nil, fmt.Errorf("e4b: prefill_append failed (rc=%d)", int(rc))
	}
	return e.logits, nil
}

// Rewind drops slot 0's KV back to nKeep tokens for prefix reuse; returns false if the
// sliding ring already overwrote the window for nKeep (caller resets + full re-prefills).
func (e *Engine) Rewind(nKeep int) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return C.e4b_engine_rewind(e.ptr, C.int(nKeep)) == 1
}

// Decode advances the cache by one token and returns the next-token logits. The
// slice is valid only until the next Prefill/Decode call.
func (e *Engine) Decode(token int32) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	rc := C.e4b_engine_decode(e.ptr, C.int32_t(token),
		(*C.float)(unsafe.Pointer(&e.logits[0])))
	if rc != 0 {
		return nil, fmt.Errorf("e4b: decode failed (rc=%d)", int(rc))
	}
	return e.logits, nil
}

// LoadAssistant loads the gemma4-assistant MTP draft head GGUF (~78M, 4 Q-only
// layers) so the single-sequence path can run greedy speculative decode (~2x decode,
// lossless). Returns an error on failure; the engine stays usable on plain decode.
func (e *Engine) LoadAssistant(path string) error {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	e.mu.Lock()
	defer e.mu.Unlock()
	if C.e4b_engine_load_assistant(e.ptr, cp) != 0 {
		return fmt.Errorf("e4b: load assistant failed for %s", path)
	}
	return nil
}

// HasAssistant reports whether an MTP draft head is loaded (spec decode is available).
func (e *Engine) HasAssistant() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return C.e4b_engine_has_assistant(e.ptr) == 1
}

// SpecStream drives the C greedy speculative-decode loop, continuing from the engine's
// CURRENT KV (history must already be prefilled; len(history) == NPast) and the
// last-token logits in firstLogits. It invokes emit(token) for every committed token in
// order; emit returning true stops generation after that token. It stops at any stop id
// or after maxNew tokens. Returns ALL committed tokens (including any the callback
// declined and any accepted-but-unemitted tail of the final round) so the caller can
// reconcile its prefix cache with the engine KV, and the number of accepted drafts.
// Greedy/lossless: byte-identical to plain greedy decode.
func (e *Engine) SpecStream(history []int32, firstLogits []float32, maxNew int,
	stops []int32, emit func(int32) bool) ([]int32, error) {
	if maxNew <= 0 || len(firstLogits) == 0 {
		return nil, fmt.Errorf("e4b: spec_stream bad args (maxNew=%d, logits=%d)", maxNew, len(firstLogits))
	}
	if emit == nil {
		emit = func(int32) bool { return false } // non-streaming: never stop early
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]int32, maxNew)
	var histPtr *C.int32_t
	if len(history) > 0 {
		histPtr = (*C.int32_t)(unsafe.Pointer(&history[0]))
	}
	var stopsPtr *C.int32_t
	if len(stops) > 0 {
		stopsPtr = (*C.int32_t)(unsafe.Pointer(&stops[0]))
	}
	h := cgo.NewHandle(emit)
	defer h.Delete()
	n := C._e4b_spec_stream(e.ptr,
		histPtr, C.int(len(history)),
		(*C.float)(unsafe.Pointer(&firstLogits[0])),
		(*C.int32_t)(unsafe.Pointer(&out[0])), C.int(maxNew),
		stopsPtr, C.int(len(stops)),
		C.uintptr_t(h))
	if n < 0 {
		return nil, fmt.Errorf("e4b: spec_stream failed (rc=%d)", int(n))
	}
	return out[:int(n)], nil
}

// SpecStats returns cumulative speculative-decode counters for /metrics:
// steps (verify rounds), drafted, accepted, emitted (committed). τ = emitted/steps.
func (e *Engine) SpecStats() (steps, drafted, accepted, emitted int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return int64(C.e4b_engine_spec_steps(e.ptr)),
		int64(C.e4b_engine_spec_drafted(e.ptr)),
		int64(C.e4b_engine_spec_accepted(e.ptr)),
		int64(C.e4b_engine_spec_emitted(e.ptr))
}

// ─── Continuous-batching ABI ───────────────────────────────────────
// The E4B engine decodes multiple sequences in ONE weight pass (step_batch). These
// expose its slot API so the server's scheduler can drive continuous batching. Greedy
// only (the batched kernel argmaxes per slot), so per-sequence sampling params are not
// applied in batched mode.

// SeqAdd prefills `prompt` into a fresh slot and returns the slot id (>=1) plus the
// first greedy token. Error if no slot is free or prefill fails.
func (e *Engine) SeqAdd(prompt []int32) (slot int, first int32, err error) {
	if len(prompt) == 0 {
		return -1, 0, fmt.Errorf("e4b: empty prompt")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	var ft C.int32_t
	sid := C.e4b_engine_seq_add(e.ptr,
		(*C.int32_t)(unsafe.Pointer(&prompt[0])), C.int(len(prompt)), &ft)
	if sid < 0 {
		return -1, 0, fmt.Errorf("e4b: seq_add failed (no free slot or prefill error)")
	}
	return int(sid), int32(ft), nil
}

// StepBatch advances each slots[i] by inTokens[i] in ONE batched forward and returns
// the next greedy token per slot (len == len(slots)).
func (e *Engine) StepBatch(slots []int32, inTokens []int32) ([]int32, error) {
	b := len(slots)
	if b == 0 || len(inTokens) != b {
		return nil, fmt.Errorf("e4b: step_batch size mismatch (%d slots, %d tokens)", b, len(inTokens))
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	cslots := make([]C.int, b)
	for i, s := range slots {
		cslots[i] = C.int(s)
	}
	out := make([]int32, b)
	rc := C.e4b_engine_step_batch(e.ptr, &cslots[0],
		(*C.int32_t)(unsafe.Pointer(&inTokens[0])), C.int(b),
		(*C.int32_t)(unsafe.Pointer(&out[0])))
	if rc != 0 {
		return nil, fmt.Errorf("e4b: step_batch failed (rc=%d)", int(rc))
	}
	return out, nil
}

// SeqRemove releases a slot (its KV caches are kept for reuse by a later SeqAdd).
func (e *Engine) SeqRemove(slot int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	C.e4b_engine_seq_remove(e.ptr, C.int(slot))
}

// SeqCapacity reports the number of currently-free sequence slots.
func (e *Engine) SeqCapacity() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return int(C.e4b_engine_seq_capacity(e.ptr))
}

// ConfigDir returns the directory holding the checkpoint (for sibling lookups
// like tokenizer.json). For a file path it is the parent dir.
func ConfigDir(path string) string {
	if fi, err := os.Stat(path); err == nil && fi.IsDir() {
		return path
	}
	return filepath.Dir(path)
}
