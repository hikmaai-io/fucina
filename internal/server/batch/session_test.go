package batch

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"
)

// sessionMock is a CPU-only model of the Q35 slot-state ABI. A slot records
// every committed input token. Prefill commits the prompt/suffix, generation
// samples one token ahead, and ExportSession snapshots only committed tokens.
type sessionMock struct {
	mu       sync.Mutex
	capacity int
	next     int
	free     []int
	live     map[int][]int32

	restoreCalls int
	exportCalls  int
	exactCalls   int
	stepCalls    int
	addCalls     int
	removeCalls  int
	prefilled    []int32
	failRestore  bool
	failExport   bool
}

func newSessionMock(capacity int) *sessionMock {
	return &sessionMock{capacity: capacity, live: make(map[int][]int32)}
}

func (m *sessionMock) alloc(tokens []int32) int {
	var slot int
	if n := len(m.free); n > 0 {
		slot = m.free[n-1]
		m.free = m.free[:n-1]
	} else {
		slot = m.next
		m.next++
	}
	m.live[slot] = append([]int32(nil), tokens...)
	return slot
}

func (m *sessionMock) Capacity() int { return m.capacity }

func (m *sessionMock) AddSeq(prompt []int32, _ SeqParams) (int, int32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.addCalls++
	slot := m.alloc(prompt)
	return slot, prompt[len(prompt)-1] + 1, nil
}

func (m *sessionMock) OpenSeq(_ []int32, _ SeqParams) (int, int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.alloc(nil), 0, nil
}

func (m *sessionMock) PrefillChunk(slot int, chunk []int32, last bool) (int32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.live[slot]; !ok {
		return 0, fmt.Errorf("dead slot %d", slot)
	}
	m.prefilled = append(m.prefilled, chunk...)
	m.live[slot] = append(m.live[slot], chunk...)
	if last {
		return chunk[len(chunk)-1] + 1, nil
	}
	return 0, nil
}

func (m *sessionMock) RestoreSession(prompt []int32, _ SeqParams, snap StateSnapshot) (int, int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.restoreCalls++
	if m.failRestore {
		return 0, 0, errors.New("forced restore failure")
	}
	if len(snap.Tokens) == 0 || len(snap.Tokens) >= len(prompt) || !reflect.DeepEqual(snap.Tokens, prompt[:len(snap.Tokens)]) {
		return 0, 0, errors.New("snapshot is not a strict prefix")
	}
	if !reflect.DeepEqual(snap.State, encodeState(snap.Tokens)) {
		return 0, 0, errors.New("bad state bytes")
	}
	return m.alloc(snap.Tokens), len(snap.Tokens), nil
}

func (m *sessionMock) ExportSession(slot int, history []int32) (*StateSnapshot, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.exportCalls++
	if m.failExport {
		return nil, errors.New("forced export failure")
	}
	committed := m.live[slot]
	if len(committed) > len(history) || !reflect.DeepEqual(committed, history[:len(committed)]) {
		return nil, errors.New("history does not describe committed state")
	}
	tokens := append([]int32(nil), committed...)
	return &StateSnapshot{Tokens: tokens, State: encodeState(tokens)}, nil
}

// StepBatch deliberately returns a multi-token run. A persistent request must
// never call it: the scheduler must select StepBatchExact instead.
func (m *sessionMock) StepBatch(active []int32, inputs []int32) ([][]int32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stepCalls++
	out := make([][]int32, len(active))
	for i, slot := range active {
		m.live[int(slot)] = append(m.live[int(slot)], inputs[i])
		out[i] = []int32{inputs[i] + 1, inputs[i] + 2, inputs[i] + 3}
	}
	return out, nil
}

func (m *sessionMock) StepBatchExact(active []int32, inputs []int32) ([]int32, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.exactCalls++
	out := make([]int32, len(active))
	for i, slot := range active {
		m.live[int(slot)] = append(m.live[int(slot)], inputs[i])
		out[i] = inputs[i] + 1
	}
	return out, nil
}

func (m *sessionMock) RemoveSeq(slot int) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.removeCalls++
	delete(m.live, slot)
	m.free = append(m.free, slot)
	return nil
}

type sessionMockStats struct {
	restores, exports, exact, steps, adds, removes, live int
	prefilled                                            []int32
}

func (m *sessionMock) stats() sessionMockStats {
	m.mu.Lock()
	defer m.mu.Unlock()
	return sessionMockStats{
		restores: m.restoreCalls, exports: m.exportCalls,
		exact: m.exactCalls, steps: m.stepCalls,
		adds: m.addCalls, removes: m.removeCalls, live: len(m.live),
		prefilled: append([]int32(nil), m.prefilled...),
	}
}

func encodeState(tokens []int32) []byte {
	state := make([]byte, 4*len(tokens))
	for i, tok := range tokens {
		binary.LittleEndian.PutUint32(state[4*i:], uint32(tok))
	}
	return state
}

func TestDiskSessionRestoresSuffixAndExportsUpdatedState(t *testing.T) {
	eng := newSessionMock(1)
	sched := New(eng, 4)
	sched.chunkSize = 2 // prove the restored suffix can span normal prefill chunks
	sched.Start()
	defer sched.Shutdown()

	prompt := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	saved := []int32{1, 2, 3, 4, 5}
	done := make(chan Result, 1)
	var got []int32
	if err := sched.Submit(Request{
		Tokens:         prompt,
		MaxNew:         3,
		Ctx:            context.Background(),
		SessionState:   &StateSnapshot{Tokens: saved, State: encodeState(saved)},
		PersistSession: true,
		Emit: func(tok int32) bool {
			got = append(got, tok)
			return true
		},
		Done: done,
	}); err != nil {
		t.Fatal(err)
	}
	res := waitResult(t, done)
	if res.Reason != FinishLength || res.Generated != 3 {
		t.Fatalf("result = %+v, want length/3", res)
	}
	if res.ReusedTokens != len(saved) {
		t.Errorf("reused = %d, want %d", res.ReusedTokens, len(saved))
	}
	stats := eng.stats()
	if !reflect.DeepEqual(stats.prefilled, prompt[len(saved):]) {
		t.Errorf("prefilled = %v, want suffix %v", stats.prefilled, prompt[len(saved):])
	}
	if stats.adds != 0 || stats.restores != 1 {
		t.Errorf("add=%d restore=%d, want 0/1 (no cold prefill)", stats.adds, stats.restores)
	}
	if stats.steps != 0 || stats.exact == 0 {
		t.Errorf("ordinary steps=%d exact steps=%d; persistent generation must use exact steps", stats.steps, stats.exact)
	}
	if !reflect.DeepEqual(got, []int32{9, 10, 11}) {
		t.Errorf("generated = %v, want [9 10 11]", got)
	}
	// The final sampled token (11) is intentionally one ahead of engine state.
	// The complete restorable frontier is prompt + committed generated [9,10].
	wantSaved := append(append([]int32(nil), prompt...), 9, 10)
	if res.SessionErr != nil || res.SessionState == nil {
		t.Fatalf("session export = %#v, err=%v", res.SessionState, res.SessionErr)
	}
	if !reflect.DeepEqual(res.SessionState.Tokens, wantSaved) ||
		!reflect.DeepEqual(res.SessionState.State, encodeState(wantSaved)) {
		t.Errorf("exported snapshot tokens/state do not match committed frontier: %+v", res.SessionState.Tokens)
	}
	stats = eng.stats()
	if stats.live != 0 || stats.removes != 1 {
		t.Errorf("slot leak: live=%d removes=%d", stats.live, stats.removes)
	}
}

func TestDiskSessionRestoreFailureNeverFallsBackCold(t *testing.T) {
	eng := newSessionMock(1)
	eng.failRestore = true
	sched := New(eng, 4)
	sched.Start()
	defer sched.Shutdown()

	done := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens:         []int32{1, 2, 3},
		MaxNew:         1,
		Ctx:            context.Background(),
		SessionState:   &StateSnapshot{Tokens: []int32{1, 2}, State: encodeState([]int32{1, 2})},
		PersistSession: true,
		Done:           done,
	}); err != nil {
		t.Fatal(err)
	}
	res := waitResult(t, done)
	if res.Reason != FinishError || res.Err == nil {
		t.Fatalf("result = %+v, want restore error", res)
	}
	if adds := eng.stats().adds; adds != 0 {
		t.Errorf("restore failure made %d cold AddSeq calls; want none", adds)
	}
}

func TestDiskSessionUnsupportedEngineFailsBeforeAdmission(t *testing.T) {
	eng := newMockEngine(1) // BatchEngine, but no SessionStateEngine
	sched := New(eng, 4)
	sched.Start()
	defer sched.Shutdown()

	done := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: []int32{1, 2, 3}, MaxNew: 1, Ctx: context.Background(),
		PersistSession: true, Done: done,
	}); err != nil {
		t.Fatal(err)
	}
	res := waitResult(t, done)
	if res.Reason != FinishError || res.Err == nil {
		t.Fatalf("result = %+v, want unsupported persistence error", res)
	}
	add, remove := eng.counts()
	if add != 0 || remove != 0 {
		t.Fatalf("unsupported persistence touched engine: add=%d remove=%d", add, remove)
	}
}

func TestNewDiskSessionExportsAndReportsSaveFailure(t *testing.T) {
	for _, fail := range []bool{false, true} {
		t.Run(fmt.Sprintf("fail=%v", fail), func(t *testing.T) {
			eng := newSessionMock(1)
			eng.failExport = fail
			sched := New(eng, 4)
			sched.Start()
			defer sched.Shutdown()

			done := make(chan Result, 1)
			if err := sched.Submit(Request{
				Tokens: []int32{4, 5, 6}, MaxNew: 1, Ctx: context.Background(),
				PersistSession: true, Done: done,
			}); err != nil {
				t.Fatal(err)
			}
			res := waitResult(t, done)
			if res.Reason != FinishLength {
				t.Fatalf("reason = %s, want length", res.Reason)
			}
			if fail {
				if res.SessionErr == nil || res.SessionState != nil {
					t.Fatalf("failed export state=%v err=%v", res.SessionState, res.SessionErr)
				}
			} else if res.SessionErr != nil || res.SessionState == nil {
				t.Fatalf("successful export state=%v err=%v", res.SessionState, res.SessionErr)
			}
			if live := eng.stats().live; live != 0 {
				t.Errorf("slot leaked after export: %d", live)
			}
		})
	}
}
