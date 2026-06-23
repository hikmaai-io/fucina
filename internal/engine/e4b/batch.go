package e4b

import "github.com/hikmaai-io/fucina/internal/server/batch"

// BatchAdapter wraps *Engine to satisfy the server's batch.BatchEngine interface so the
// continuous-batching scheduler can drive E4B: AddSeq prefills a slot, StepBatch advances
// all active slots in ONE weight pass, RemoveSeq frees a slot. The E4B batched kernel is
// greedy (argmax per slot), so per-sequence sampling params are not applied in batched
// mode. Mirrors internal/engine/cuda.BatchAdapter; the scheduler calls every method from
// its single owner goroutine, so `active` needs no extra locking beyond the engine mutex.
type BatchAdapter struct {
	eng    *Engine
	active int // slots currently held (AddSeq++ / RemoveSeq--), so Capacity() reports total
}

// NewBatchAdapter wraps eng for the batch scheduler. Supported() reports usability.
func NewBatchAdapter(eng *Engine) *BatchAdapter { return &BatchAdapter{eng: eng} }

// Supported reports whether batched serving is usable (a free slot exists).
func (a *BatchAdapter) Supported() bool { return a.eng.SeqCapacity() > 0 }

// AddSeq prefills the prompt into a fresh slot and returns the slot + first greedy token.
func (a *BatchAdapter) AddSeq(prompt []int32, params batch.SeqParams) (int, int32, error) {
	slot, first, err := a.eng.SeqAdd(prompt) // greedy; params not applied in batched mode
	if err != nil {
		return 0, 0, err
	}
	a.active++
	return slot, first, nil
}

// StepBatch advances active slots one token each (weights read once for the batch) and
// returns a length-1 token run per slot (the non-speculative batch.BatchEngine contract).
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

// Capacity reports the total concurrent-slot budget (free + held) for the admission test.
func (a *BatchAdapter) Capacity() int { return a.eng.SeqCapacity() + a.active }
