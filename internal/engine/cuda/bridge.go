// Package cuda provides the Go interface to the CUDA inference engine.
//
// This package uses CGO to call the C CUDA kernels compiled in libfucina.a.
// The engine is specific to Gemma 4 12B on DGX Spark GB10 (sm_121, CUDA 13.0).

package cuda

// #cgo LDFLAGS: -L${SRCDIR}/../../../cuda -L/usr/local/cuda/lib64 -lfucina -lcudart -lcublas -lcublasLt -lcuda -lpthread -lstdc++ -lm
// #cgo CFLAGS: -I/usr/local/cuda-13/include -I${SRCDIR}/../../../cuda
//
// #include "gemma4_kernels.cuh"
// #include <stdlib.h>
//
// // Thin C wrappers for CGO compatibility with graph API
// static inline void _fucina_set_graph_mode(struct gemma4_engine *eng, int mode) {
//     gemma4_engine_set_graph_mode(eng, mode);
// }
// static inline void _fucina_graph_stats(const struct gemma4_engine *eng,
//     int *hits, int *misses, int *captures, int *launches) {
//     gemma4_engine_graph_stats(eng, hits, misses, captures, launches);
// }
//
// // Per-token streaming bridge: fucinaSpecTokenGo is the cgo-exported Go callback
// // (defined in callback.go); the engine invokes it once per emitted token. The
// // uintptr is a runtime/cgo.Handle resolving to the request's emit closure.
// extern int fucinaSpecTokenGo(int32_t tok, void *ud);
// static inline int _fucina_generate_spec_stream(gemma4_engine_t *eng,
//     const int32_t *hist, int n_hist, const float *first_logits,
//     int32_t *out, int max_new, const int32_t *stops, int n_stop, int draft_k,
//     float temp, int top_k, float top_p, float min_p, float repeat_penalty,
//     uint64_t seed, int *n_accepted, uintptr_t handle) {
//     return gemma4_engine_generate_spec_stream(eng, hist, n_hist, first_logits,
//         out, max_new, stops, n_stop, draft_k, temp, top_k, top_p, min_p,
//         repeat_penalty, seed, n_accepted, fucinaSpecTokenGo, (void *)handle);
// }
import "C"

import (
	"fmt"
	"runtime"
	"runtime/cgo"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/hikmaai-io/fucina/internal/server/batch"
)

// vocabSize is the gemma-4 logits width (the device writes exactly this many
// floats per Prefill/Decode). Kept as a const so the scratch buffer and the C
// side agree.
const vocabSize = 262144

// Engine wraps the C inference engine.
type Engine struct {
	mu   sync.Mutex
	ptr  *C.gemma4_engine_t
	ctx  uint32
	path string

	// closing is set true BEFORE Close() destroys e.ptr, so AbortPrefill (which
	// reads e.ptr lock-free to avoid deadlocking the prefilling goroutine that
	// holds e.mu) can detect a shutdown-in-progress and skip the C call rather
	// than dereferencing a pointer Close() is about to / has freed.
	closing atomic.Bool

	// logitsBuf is a single reusable last-token logits buffer (vocabSize floats,
	// 1 MiB). Prefill/Decode previously allocated a fresh 1 MiB slice per call; on
	// the token-by-token decode path that churned GBs of GC garbage per request.
	// Reuse is safe because e.mu serializes every engine call, so only one logits
	// result is ever live at a time. CONTRACT: the returned slice is valid only
	// until the next Prefill/Decode call — callers must sample/consume it before
	// re-entering the engine (which the single-flight server does).
	logitsBuf []float32
}

// Config holds engine configuration. The weight format (Q4_0-QAT or Q8_0) is
// auto-detected from the GGUF tensor table; other formats are rejected at load.
type Config struct {
	ModelPath   string // -m, --model
	ContextSize uint32 // --ctx
	DeviceID    int    // --cuda-device
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
		C.FORMAT_Q8_0, // placeholder — the engine auto-detects Q4_0/Q8_0 from the GGUF
		C.uint32_t(ctxSize),
		C.int(cfg.DeviceID),
	)
	if ptr == nil {
		return nil, fmt.Errorf("fucina: engine creation failed for %s", cfg.ModelPath)
	}

	eng := &Engine{
		ptr:       ptr,
		ctx:       ctxSize,
		path:      cfg.ModelPath,
		logitsBuf: make([]float32, vocabSize),
	}

	return eng, nil
}

// LoadAssistant loads the official Gemma-4 MTP assistant GGUF (~423M draft head).
// When loaded, speculative decoding drafts novel text with it (multi-token prediction
// over the shared target KV cache) — the llama.cpp --spec-type draft-mtp equivalent.
func (e *Engine) LoadAssistant(path string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	if C.gemma4_engine_load_assistant(e.ptr, cPath) != 0 {
		return fmt.Errorf("fucina: assistant load failed for %s", path)
	}
	return nil
}

// Close destroys the engine and frees all GPU resources.
func (e *Engine) Close() {
	// Signal shutdown BEFORE taking the lock so a concurrent lock-free
	// AbortPrefill sees closing=true and skips its C call (see Engine.closing).
	e.closing.Store(true)
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.ptr != nil {
		C.gemma4_engine_destroy(e.ptr)
		e.ptr = nil
	}
}

// Info returns a formatted string with engine information.
func (e *Engine) Info() string {
	return fmt.Sprintf("%s (%d ctx, device=%d)", e.path, e.ctx, 0)
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

// Prefill processes a batch of tokens (sequential) and fills the KV cache.
func (e *Engine) Prefill(tokens []int32) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if len(tokens) == 0 {
		return nil, nil
	}

	// Only get logits for the last token. Reuse the per-engine scratch (valid only
	// until the next engine call — see Engine.logitsBuf).
	logits := e.logitsBuf

	// Fast path: batched BF16 tensor-core prefill (one weight pass for the whole
	// prompt). It defers (-2) for very large prompts (the [HEADS][N×N] score buffer
	// would OOM) → chunked FLASH prefill (O(chunk+KV) memory, 256k+ capable) → and
	// finally the proven token-by-token path. All produce the same last-token logits.
	tokPtr := (*C.int32_t)(unsafe.Pointer(&tokens[0]))
	logPtr := (*C.float)(unsafe.Pointer(&logits[0]))
	n := C.int(len(tokens))
	ret := C.gemma4_engine_prefill_batched(e.ptr, tokPtr, n, logPtr)
	if ret == -2 {
		ret = C.gemma4_engine_prefill_flash(e.ptr, tokPtr, n, logPtr)
	}
	if ret == -2 {
		ret = C.gemma4_engine_prefill(e.ptr, tokPtr, n, logPtr)
	}
	if ret == -3 {
		return nil, prefillAborted{}
	}
	if ret != 0 {
		return nil, fmt.Errorf("fucina: prefill failed")
	}

	return logits, nil
}

// prefillAborted marks a cooperative cancellation (AbortPrefill): the engine
// state is consistent (committed chunks at correct positions; unaccounted
// writes never read), so the prefix cache may retain the shared prefix. The
// server's kvcache detects it structurally via the anonymous
// interface{ Aborted() bool } — no cgo import needed there.
type prefillAborted struct{}

func (prefillAborted) Error() string { return "fucina: prefill aborted" }
func (prefillAborted) Aborted() bool { return true }

// AbortPrefill asks an in-flight Prefill (on another goroutine) to stop at the
// next chunk/layer boundary. It deliberately does NOT take e.mu — the
// prefilling goroutine holds it — and only writes an advisory flag engine-side.
// The caller (server) joins its watcher goroutine while still holding the kv
// lock, so a stale call can never outlive its request; the residual e.ptr read
// vs Close() race is confined to process shutdown.
func (e *Engine) AbortPrefill() {
	// closing is checked first (and set before Close() frees e.ptr) so a shutdown
	// race cannot pass a freed pointer to C. The non-atomic e.ptr read remains, but
	// it is only reached when closing==false, i.e. before Close() runs.
	if e.closing.Load() {
		return
	}
	if e.ptr != nil {
		C.gemma4_engine_abort_prefill(e.ptr)
	}
}

// Warmup eagerly runs the engine's lazy first-prefill setup (persistent prefill
// scratch + BF16 dequant scratch, ~0.5-2.1s of cudaMallocs) so the first
// request's prefill timer measures prefill rather than setup. Call once at
// server startup. Errors are non-fatal (the prefill paths keep their lazy
// fallbacks) so none are returned.
func (e *Engine) Warmup() {
	e.mu.Lock()
	defer e.mu.Unlock()
	C.gemma4_engine_warmup(e.ptr)
}

// Decode processes a single token and returns logits. The returned slice is the
// per-engine scratch (Engine.logitsBuf) and is valid only until the next engine
// call — the single-flight server samples it before re-entering the engine.
func (e *Engine) Decode(token int32) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	logits := e.logitsBuf

	ret := C.gemma4_engine_decode(
		e.ptr,
		C.int32_t(token),
		(*C.float)(unsafe.Pointer(&logits[0])),
	)
	if ret != 0 {
		return nil, fmt.Errorf("fucina: decode failed")
	}

	return logits, nil
}

// GenerateSpec runs greedy generation with prompt-lookup speculative decoding.
// It prefills `prompt` internally and generates up to maxNew tokens, stopping at
// any id in stops. Returns the generated tokens and the total number of drafts
// accepted (for measuring the acceptance rate). Greedy/argmax only — produces the
// exact same tokens as a plain greedy decode, just faster on context-reusing text.
func (e *Engine) GenerateSpec(prompt []int32, maxNew int, stops []int32, draftK int,
	temp float32, topK int, topP, minP, repeatPenalty float32, seed uint64) ([]int32, int, error) {
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
		C.float(temp), C.int(topK), C.float(topP), C.float(minP), C.float(repeatPenalty),
		C.uint64_t(seed),
		&nacc,
	)
	if ng < 0 {
		return nil, 0, fmt.Errorf("fucina: generate_spec failed")
	}
	return out[:ng], int(nacc), nil
}

// DecodeNoCopy decodes a single token but leaves the logits on the GPU (no 262k
// D2H). Pair it with SampleDevice, which selects the next token on-device.
func (e *Engine) DecodeNoCopy(token int32) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if C.gemma4_engine_decode(e.ptr, C.int32_t(token), nil) != 0 {
		return fmt.Errorf("fucina: decode failed")
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
		return 0, fmt.Errorf("fucina: device sample failed")
	}
	return int32(id), nil
}

// GenerateSpecContinue runs speculative generation continuing from the engine's
// already-prefilled state (server path). history = prompt tokens in the cache,
// firstLogits = post-prefill logits. Returns generated tokens + drafts accepted.
func (e *Engine) GenerateSpecContinue(history []int32, firstLogits []float32,
	maxNew int, stops []int32, draftK int,
	temp float32, topK int, topP, minP, repeatPenalty float32, seed uint64) ([]int32, int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	out := make([]int32, maxNew)
	var nacc C.int
	var histPtr *C.int32_t
	if len(history) > 0 {
		histPtr = (*C.int32_t)(unsafe.Pointer(&history[0]))
	}
	var stopsPtr *C.int32_t
	if len(stops) > 0 {
		stopsPtr = (*C.int32_t)(unsafe.Pointer(&stops[0]))
	}
	ng := C.gemma4_engine_generate_spec_continue(
		e.ptr,
		histPtr, C.int(len(history)),
		(*C.float)(unsafe.Pointer(&firstLogits[0])),
		(*C.int32_t)(unsafe.Pointer(&out[0])), C.int(maxNew),
		stopsPtr, C.int(len(stops)),
		C.int(draftK),
		C.float(temp), C.int(topK), C.float(topP), C.float(minP), C.float(repeatPenalty),
		C.uint64_t(seed),
		&nacc,
	)
	if ng < 0 {
		return nil, 0, fmt.Errorf("fucina: generate_spec_continue failed")
	}
	return out[:ng], int(nacc), nil
}

// GenerateSpecStream is GenerateSpecContinue with a per-token emit callback: the
// engine invokes emit(token) for every generated token, in order, between verify
// steps, so the caller can stream while keeping the speculative fast path. emit
// returning true stops generation after that token. The returned slice still
// carries ALL generated tokens (including any the callback declined to render),
// which callers need to reconcile the prefix cache with the engine KV.
func (e *Engine) GenerateSpecStream(history []int32, firstLogits []float32,
	maxNew int, stops []int32, draftK int,
	temp float32, topK int, topP, minP, repeatPenalty float32, seed uint64,
	emit func(int32) bool) ([]int32, int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	out := make([]int32, maxNew)
	var nacc C.int
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
	ng := C._fucina_generate_spec_stream(
		e.ptr,
		histPtr, C.int(len(history)),
		(*C.float)(unsafe.Pointer(&firstLogits[0])),
		(*C.int32_t)(unsafe.Pointer(&out[0])), C.int(maxNew),
		stopsPtr, C.int(len(stops)),
		C.int(draftK),
		C.float(temp), C.int(topK), C.float(topP), C.float(minP), C.float(repeatPenalty),
		C.uint64_t(seed),
		&nacc, C.uintptr_t(h),
	)
	if ng < 0 {
		return nil, 0, fmt.Errorf("fucina: generate_spec_stream failed")
	}
	return out[:ng], int(nacc), nil
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

// SpecStats returns cumulative speculative-decode acceptance counters for /metrics:
// steps (verify forwards), drafted (proposed draft tokens), accepted (matched drafts),
// emitted (accepted + bonus committed per step). τ = emitted/steps; acceptance = accepted/drafted.
func (e *Engine) SpecStats() (steps, drafted, accepted, emitted int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return int64(C.gemma4_engine_spec_steps(e.ptr)),
		int64(C.gemma4_engine_spec_drafted(e.ptr)),
		int64(C.gemma4_engine_spec_accepted(e.ptr)),
		int64(C.gemma4_engine_spec_emitted(e.ptr))
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

// KVStateSize returns the host-buffer size in bytes needed to snapshot the
// first nTokens of the KV cache (0 if nTokens is out of range).
func (e *Engine) KVStateSize(nTokens int) int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return int(C.gemma4_engine_kv_state_size(e.ptr, C.int(nTokens)))
}

// KVSave snapshots the first nTokens of the live KV sequence into buf (sized
// via KVStateSize). The live sequence is untouched. ~200 KB/token, copied at
// unified-memory bandwidth — tens of ms for a 20k-token conversation, versus
// the ~100 s full re-prefill it replaces.
func (e *Engine) KVSave(buf []byte, nTokens int) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(buf) == 0 {
		return fmt.Errorf("fucina: kv save: empty buffer")
	}
	if C.gemma4_engine_kv_save(e.ptr, unsafe.Pointer(&buf[0]), C.int(nTokens)) != 0 {
		return fmt.Errorf("fucina: kv save failed (%d tokens)", nTokens)
	}
	return nil
}

// KVRestore overwrites the live KV sequence with a snapshot taken by KVSave
// and sets the engine token count to nTokens.
func (e *Engine) KVRestore(buf []byte, nTokens int) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(buf) == 0 {
		return fmt.Errorf("fucina: kv restore: empty buffer")
	}
	if C.gemma4_engine_kv_restore(e.ptr, unsafe.Pointer(&buf[0]), C.int(nTokens)) != 0 {
		return fmt.Errorf("fucina: kv restore failed (%d tokens)", nTokens)
	}
	return nil
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

// SetGraphMode enables (1) or disables (0) CUDA graph support.
// When enabled, allocates persistent prefill scratch buffers and
// prepares for future graph capture of prefill_batched.
func (e *Engine) SetGraphMode(mode int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	C._fucina_set_graph_mode(e.ptr, C.int(mode))
}

// GraphStats returns CUDA graph statistics.
func (e *Engine) GraphStats() (hits, misses, captures, launches int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	var h, m, c, l C.int
	C._fucina_graph_stats(e.ptr, &h, &m, &c, &l)
	return int(h), int(m), int(c), int(l)
}

// ─── Continuous-batching ABI (FUCINA_BATCH) ────────────────────────
//
// These wrap the paged multi-sequence C ABI (gemma4_engine_seq_add /
// _step_batch / _seq_remove / _seq_capacity). They require the engine to have
// been created with FUCINA_PAGED_KV=1; otherwise every call returns an error
// (the C side checks eng->paged_enabled). Sampling is on-device per row: each
// sequence's SeqParams (temperature/top_k/top_p/min_p/seed) are stored on its
// slot at SeqAdd and applied to every token (temp<=0 ⇒ exact greedy argmax).
//
// maxBatchSeqs mirrors GEMMA4_MAX_SEQS (== GEMMA4_SPEC_MAX) in the kernels: the
// fixed number of concurrent paged slots. seq_capacity() reports only the FREE
// slots, so the adapter reconstructs the *total* (free+used) for the scheduler's
// Capacity() contract by tracking how many slots it currently holds.
const maxBatchSeqs = 16

// SeqAdd prefills prompt into a fresh paged slot and returns the slot id and the
// first sampled token. The per-sequence sampling params are stored on the slot
// and applied on-device to every token of this sequence (temp<=0 ⇒ greedy). err
// is non-nil when no slot is free, the engine is not in paged mode, or prefill
// failed.
func (e *Engine) SeqAdd(prompt []int32, p batch.SeqParams) (slot int, first int32, err error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(prompt) == 0 {
		return 0, 0, fmt.Errorf("fucina: seq_add: empty prompt")
	}
	var firstTok C.int32_t
	id := C.gemma4_engine_seq_add(
		e.ptr,
		(*C.int32_t)(unsafe.Pointer(&prompt[0])), C.int(len(prompt)),
		&firstTok,
		C.float(p.Temperature), C.int(p.TopK), C.float(p.TopP), C.float(p.MinP),
		C.uint64_t(p.Seed),
	)
	if id < 0 {
		return 0, 0, fmt.Errorf("fucina: seq_add failed (no slot / not paged / prefill error)")
	}
	return int(id), int32(firstTok), nil
}

// StepBatch advances each slot in slots by one token (feeding inputs[i] to
// slots[i]) in a single batched forward, returning one freshly sampled token per
// slot. len(inputs) must equal len(slots).
func (e *Engine) StepBatch(slots []int32, inputs []int32) ([]int32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(slots) != len(inputs) {
		return nil, fmt.Errorf("fucina: step_batch: %d slots vs %d inputs", len(slots), len(inputs))
	}
	b := len(slots)
	if b == 0 {
		return nil, nil
	}
	if b > maxBatchSeqs {
		return nil, fmt.Errorf("fucina: step_batch: batch %d exceeds max %d", b, maxBatchSeqs)
	}
	// The C ABI takes `const int *slots`; Go int32 is not C int, so marshal into
	// a C-int slot array.
	cslots := make([]C.int, b)
	for i, s := range slots {
		cslots[i] = C.int(s)
	}
	out := make([]int32, b)
	ret := C.gemma4_engine_step_batch(
		e.ptr,
		&cslots[0],
		(*C.int32_t)(unsafe.Pointer(&inputs[0])),
		C.int(b),
		(*C.int32_t)(unsafe.Pointer(&out[0])),
	)
	if ret != 0 {
		return nil, fmt.Errorf("fucina: step_batch failed")
	}
	return out, nil
}

// SeqRemove frees a slot's paged KV back to the pool and marks it reusable.
func (e *Engine) SeqRemove(slot int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	C.gemma4_engine_seq_remove(e.ptr, C.int(slot))
}

// SeqFreeCapacity reports the number of currently FREE paged slots.
func (e *Engine) SeqFreeCapacity() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return int(C.gemma4_engine_seq_capacity(e.ptr))
}

// BatchAdapter adapts *Engine to batch.BatchEngine so the continuous-batching
// scheduler can drive the paged multi-sequence engine. It is the only consumer
// of the SeqAdd/StepBatch/SeqRemove/SeqFreeCapacity wrappers.
//
// The scheduler calls every method from its single owner goroutine, so the
// adapter needs no locking of its own beyond the engine mutex the wrappers take.
// It tracks the live slot count so Capacity() can report the engine's TOTAL slot
// budget (free + used) — the C seq_capacity() returns only free slots, but the
// scheduler's admission test (len(active) < Capacity()) needs the total.
type BatchAdapter struct {
	eng    *Engine
	active int // slots currently held by the scheduler (incremented in AddSeq, decremented in RemoveSeq)
}

// NewBatchAdapter wraps eng for the batch scheduler. The engine must have been
// created with FUCINA_PAGED_KV=1; Supported() reports whether batching is usable.
func NewBatchAdapter(eng *Engine) *BatchAdapter { return &BatchAdapter{eng: eng} }

// Supported reports whether the engine can serve batched requests: a free-slot
// count > 0 means paged mode is enabled (seq_capacity returns 0 when it is not).
func (a *BatchAdapter) Supported() bool { return a.eng.SeqFreeCapacity() > 0 }

// AddSeq admits a new sequence (prefill + first token sampled with params). On
// success it records the slot so Capacity() stays accurate.
func (a *BatchAdapter) AddSeq(prompt []int32, params batch.SeqParams) (int, int32, error) {
	slot, first, err := a.eng.SeqAdd(prompt, params)
	if err != nil {
		return 0, 0, err
	}
	a.active++
	return slot, first, nil
}

// StepBatch runs one batched decode step over active slots and returns a token
// RUN per slot (batch.BatchEngine contract). The current C ABI
// (gemma4_engine_step_batch) is the non-speculative path: it samples exactly one
// token per slot, so each row's run has length 1 — including the -1 KV-exhausted
// sentinel, which is delivered as []int32{-1} so the scheduler stops just that
// row. When the C engine grows a per-sequence speculative step that emits a
// variable number of accepted tokens per slot, this is the boundary that widens
// each row's run; the scheduler already walks runs token-by-token.
func (a *BatchAdapter) StepBatch(active []int32, inputs []int32) ([][]int32, error) {
	toks, err := a.eng.StepBatch(active, inputs)
	if err != nil {
		return nil, err
	}
	out := make([][]int32, len(toks))
	for i, t := range toks {
		out[i] = []int32{t}
	}
	return out, nil
}

// RemoveSeq frees a slot and updates the live-slot count.
func (a *BatchAdapter) RemoveSeq(slot int) error {
	a.eng.SeqRemove(slot)
	if a.active > 0 {
		a.active--
	}
	return nil
}

// Capacity reports the engine's total concurrent-slot budget (free + currently
// held), so the scheduler's admission test len(active) < Capacity() is correct.
func (a *BatchAdapter) Capacity() int { return a.eng.SeqFreeCapacity() + a.active }

// ensure CGO runs on the main thread for CUDA compatibility
func init() {
	runtime.LockOSThread()
}
