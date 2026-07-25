package cuda

// Phase-E's cgo ShardRunner. The ordinary whole-model Engine methods remain
// untouched; this adapter is constructed only by the explicit distributed CLI.

// #include "gemma4_kernels.cuh"
import "C"

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/hikmaai-io/fucina/internal/dist"
)

// ShardInfo is the runtime geometry pinned by the FCNDIST1 handshake.
type ShardInfo struct {
	Layers int
	Hidden int
	Vocab  int
}

// ShardInfo returns model geometry reported by the loaded CUDA engine.
func (e *Engine) ShardInfo() ShardInfo {
	e.mu.Lock()
	defer e.mu.Unlock()
	return ShardInfo{
		Layers: int(C.gemma4_engine_get_n_layers(e.ptr)),
		Hidden: int(C.gemma4_engine_get_hidden_size(e.ptr)),
		Vocab:  int(C.gemma4_engine_get_vocab_size(e.ptr)),
	}
}

// EmbedShardToken produces the exact fp32 residual row consumed by layer zero.
func (e *Engine) EmbedShardToken(token int32) ([]byte, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	hidden := int(C.gemma4_engine_get_hidden_size(e.ptr))
	if hidden <= 0 {
		return nil, fmt.Errorf("fucina: invalid shard hidden size %d", hidden)
	}
	out := make([]byte, hidden*4)
	ret := C.gemma4_engine_q35_embed(e.ptr, C.int32_t(token),
		(*C.float)(unsafe.Pointer(&out[0])))
	runtime.KeepAlive(out)
	if ret != 0 {
		return nil, fmt.Errorf("fucina: qwen3.5 shard embed failed (code %d)", int(ret))
	}
	return out, nil
}

// CUDAShardRunner owns the mapping from coordinator sequence ids to this
// engine's opaque slots. Calls are serialized with the engine just like every
// other CUDA entry point.
type CUDAShardRunner struct {
	eng       *Engine
	lo, hi    int
	final     bool
	hidden    int
	vocab     int
	mu        sync.Mutex
	sequences map[uint32]shardSequence
}

type shardSequence struct {
	slot int
	next uint32
}

// NewCUDAShardRunner validates a strict Qwen3.5 layer range before accepting a
// connection. final is legal only for the range ending at the model's last layer.
func NewCUDAShardRunner(e *Engine, lo, hi int, final bool) (*CUDAShardRunner, error) {
	if e == nil {
		return nil, fmt.Errorf("fucina: nil shard engine")
	}
	info := e.ShardInfo()
	e.mu.Lock()
	supported := C.gemma4_engine_supports_q35_shards(e.ptr) == 1
	e.mu.Unlock()
	if !supported {
		return nil, fmt.Errorf("fucina: Phase-E CUDA sharding supports Qwen3.5 hybrid checkpoints only")
	}
	if lo < 0 || hi <= lo || hi > info.Layers {
		return nil, fmt.Errorf("fucina: invalid shard layers [%d,%d), model has %d", lo, hi, info.Layers)
	}
	if final && hi != info.Layers {
		return nil, fmt.Errorf("fucina: final shard [%d,%d) does not end at layer %d", lo, hi, info.Layers)
	}
	if info.Hidden <= 0 || info.Vocab <= 0 {
		return nil, fmt.Errorf("fucina: invalid shard geometry hidden=%d vocab=%d", info.Hidden, info.Vocab)
	}
	return &CUDAShardRunner{
		eng: e, lo: lo, hi: hi, final: final, hidden: info.Hidden, vocab: info.Vocab,
		sequences: make(map[uint32]shardSequence),
	}, nil
}

// Forward implements dist.ShardRunner. The integrated engine boundary currently
// accepts exactly one token so recurrent GDN and causal attention advance in the
// same order as single-node decode.
func (r *CUDAShardRunner) Forward(seq, pos uint32, ntokens int, in []byte) ([]byte, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if ntokens != 1 {
		return nil, fmt.Errorf("fucina: shard forward supports ntokens=1, got %d", ntokens)
	}
	if len(in) != r.hidden*4 {
		return nil, fmt.Errorf("fucina: shard input is %d bytes, want %d", len(in), r.hidden*4)
	}
	if uint64(pos) > uint64(^uint(0)>>1) {
		return nil, fmt.Errorf("fucina: shard position %d exceeds host int", pos)
	}

	state, ok := r.sequences[seq]
	if !ok {
		if pos != 0 {
			return nil, fmt.Errorf("fucina: new shard sequence %d starts at %d, want 0", seq, pos)
		}
		r.eng.mu.Lock()
		slot := int(C.gemma4_engine_seq_open(r.eng.ptr, 0, 0, 0, 0, 0))
		r.eng.mu.Unlock()
		if slot < 0 {
			return nil, fmt.Errorf("fucina: no CUDA shard sequence slot available")
		}
		state = shardSequence{slot: slot}
		r.sequences[seq] = state
	}
	if pos != state.next {
		return nil, fmt.Errorf("fucina: shard sequence %d position %d, want %d", seq, pos, state.next)
	}

	nout := r.hidden
	if r.final {
		nout = r.vocab
	}
	out := make([]byte, nout*4)
	r.eng.mu.Lock()
	ret := C.gemma4_engine_q35_forward_layers(
		r.eng.ptr, C.int(state.slot), C.int(pos), C.int(ntokens),
		C.int(r.lo), C.int(r.hi),
		(*C.float)(unsafe.Pointer(&in[0])),
		(*C.float)(unsafe.Pointer(&out[0])), C.int(boolInt(r.final)),
	)
	r.eng.mu.Unlock()
	runtime.KeepAlive(in)
	runtime.KeepAlive(out)
	if ret != 0 {
		return nil, fmt.Errorf("fucina: shard layers [%d,%d) failed at seq=%d pos=%d (code %d)",
			r.lo, r.hi, seq, pos, int(ret))
	}
	state.next++
	r.sequences[seq] = state
	return out, nil
}

// Reset releases this worker's sequence state. It is idempotent so disconnect
// cleanup can safely follow a coordinator-issued reset.
func (r *CUDAShardRunner) Reset(seq uint32) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	state, ok := r.sequences[seq]
	if !ok {
		return nil
	}
	r.eng.mu.Lock()
	C.gemma4_engine_seq_remove(r.eng.ptr, C.int(state.slot))
	r.eng.mu.Unlock()
	delete(r.sequences, seq)
	return nil
}

func boolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

// BytesToFloat32 decodes little-endian protocol logits without unsafe aliasing.
func BytesToFloat32(b []byte) ([]float32, error) {
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("fucina: float32 payload has %d bytes", len(b))
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out, nil
}

var _ dist.ShardRunner = (*CUDAShardRunner)(nil)
