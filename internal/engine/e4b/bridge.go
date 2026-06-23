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
import "C"

import (
	"fmt"
	"os"
	"path/filepath"
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
func New(path string, ctxSize uint32, deviceID int) (*Engine, error) {
	cp := C.CString(path)
	defer C.free(unsafe.Pointer(cp))
	ptr := C.e4b_engine_create(cp, C.uint32_t(ctxSize), C.int(deviceID))
	if ptr == nil {
		return nil, fmt.Errorf("e4b: engine create failed for %s", path)
	}
	v := int(C.e4b_engine_vocab_size(ptr))
	if v <= 0 {
		C.e4b_engine_destroy(ptr)
		return nil, fmt.Errorf("e4b: bad vocab size %d", v)
	}
	return &Engine{ptr: ptr, vocab: v, ctx: ctxSize, logits: make([]float32, v)}, nil
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

// ConfigDir returns the directory holding the checkpoint (for sibling lookups
// like tokenizer.json). For a file path it is the parent dir.
func ConfigDir(path string) string {
	if fi, err := os.Stat(path); err == nil && fi.IsDir() {
		return path
	}
	return filepath.Dir(path)
}
