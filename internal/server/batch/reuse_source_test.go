package batch

import (
	"context"
	"testing"
	"time"
)

type reuseInfoEngine struct {
	addReuse  ReuseInfo
	openReuse ReuseInfo
	chunkN    int
	nextSlot  int
}

func (e *reuseInfoEngine) Capacity() int { return 4 }
func (e *reuseInfoEngine) AddSeq(_ []int32, _ SeqParams) (int, int32, error) {
	s := e.nextSlot
	e.nextSlot++
	return s, 1, nil
}
func (e *reuseInfoEngine) StepBatch(active []int32, _ []int32) ([][]int32, error) {
	out := make([][]int32, len(active))
	for i := range active {
		out[i] = []int32{0}
	}
	return out, nil
}
func (e *reuseInfoEngine) RemoveSeq(int) error { return nil }
func (e *reuseInfoEngine) OpenSeq(_ []int32, _ SeqParams) (int, int, error) {
	s := e.nextSlot
	e.nextSlot++
	return s, e.chunkN, nil
}
func (e *reuseInfoEngine) PrefillChunk(_ int, _ []int32, last bool) (int32, error) {
	if last {
		return 1, nil
	}
	return 0, nil
}
func (e *reuseInfoEngine) LastAddSeqReuse() ReuseInfo  { return e.addReuse }
func (e *reuseInfoEngine) LastOpenSeqReuse() ReuseInfo { return e.openReuse }

type queuedReuseEngine struct {
	reuseInfoEngine
	addQueue []ReuseInfo
}

func (e *queuedReuseEngine) AddSeq(_ []int32, _ SeqParams) (int, int32, error) {
	s := e.nextSlot
	e.nextSlot++
	e.addReuse = ReuseInfo{}
	if len(e.addQueue) > 0 {
		e.addReuse = e.addQueue[0]
		e.addQueue = e.addQueue[1:]
	}
	return s, 1, nil
}

func TestSchedulerCarriesAddSeqReuseSource(t *testing.T) {
	eng := &reuseInfoEngine{addReuse: ReuseInfo{ReusedTokens: 512, Source: ReuseSourceGPUPagedBlock}}
	s := New(eng, 8)
	s.Start()
	defer s.Shutdown()
	done := make(chan Result, 1)
	if err := s.Submit(Request{Tokens: []int32{1, 2, 3}, MaxNew: 1, Ctx: context.Background(), Emit: func(int32) bool { return true }, Done: done}); err != nil {
		t.Fatal(err)
	}
	res := waitResult(t, done)
	if res.ReusedTokens != 512 || res.Source != ReuseSourceGPUPagedBlock {
		t.Fatalf("result=%+v", res)
	}
}

func TestSchedulerCarriesOpenSeqReuseSource(t *testing.T) {
	eng := &reuseInfoEngine{chunkN: 300, openReuse: ReuseInfo{ReusedTokens: 300, Source: ReuseSourceHostSnapshot}}
	s := New(eng, 8)
	s.chunk = eng
	s.chunkMin = 1
	s.chunkAdaptive = false
	s.Start()
	defer s.Shutdown()
	done := make(chan Result, 1)
	toks := make([]int32, 301)
	for i := range toks {
		toks[i] = int32(i + 1)
	}
	if err := s.Submit(Request{Tokens: toks, MaxNew: 1, Ctx: context.Background(), Emit: func(int32) bool { return true }, Done: done}); err != nil {
		t.Fatal(err)
	}
	res := waitResult(t, done)
	if res.ReusedTokens != 300 || res.Source != ReuseSourceHostSnapshot {
		t.Fatalf("result=%+v", res)
	}
}

func TestSchedulerKeepsPagedReuseBlockAligned(t *testing.T) {
	eng := &reuseInfoEngine{chunkN: 256, openReuse: ReuseInfo{ReusedTokens: 256, Source: ReuseSourceGPUPagedBlock}}
	s := New(eng, 8)
	s.chunk = eng
	s.chunkMin = 1
	s.chunkAdaptive = false
	s.Start()
	defer s.Shutdown()
	done := make(chan Result, 1)
	toks := make([]int32, 301)
	for i := range toks {
		toks[i] = int32(i + 1)
	}
	if err := s.Submit(Request{Tokens: toks, MaxNew: 1, Ctx: context.Background(), Emit: func(int32) bool { return true }, Done: done}); err != nil {
		t.Fatal(err)
	}
	res := waitResult(t, done)
	if res.ReusedTokens != 256 || res.Source != ReuseSourceGPUPagedBlock {
		t.Fatalf("result=%+v", res)
	}
}

func TestSchedulerConcurrentIsolationKeepsReusePerRequest(t *testing.T) {
	eng := &queuedReuseEngine{addQueue: []ReuseInfo{{ReusedTokens: 256, Source: ReuseSourceGPUPagedBlock}, {}}}
	s := New(eng, 8)
	s.Start()
	defer s.Shutdown()
	d1 := make(chan Result, 1)
	d2 := make(chan Result, 1)
	if err := s.Submit(Request{Tokens: []int32{1, 2, 3}, MaxNew: 1, Ctx: context.Background(), Emit: func(int32) bool { return true }, Done: d1}); err != nil {
		t.Fatal(err)
	}
	if err := s.Submit(Request{Tokens: []int32{4, 5, 6}, MaxNew: 1, Ctx: context.Background(), Emit: func(int32) bool { return true }, Done: d2}); err != nil {
		t.Fatal(err)
	}
	r1 := waitResult(t, d1)
	r2 := waitResult(t, d2)
	if r1.ReusedTokens != 256 || r1.Source != ReuseSourceGPUPagedBlock {
		t.Fatalf("first result=%+v", r1)
	}
	if r2.ReusedTokens != 0 || r2.Source != ReuseSourceNone {
		t.Fatalf("second result leaked reuse=%+v", r2)
	}
}

func TestCancelDuringPrefillDoesNotLeakReuseAccounting(t *testing.T) {
	eng := newChunkMock(2)
	eng.prefillDelay = time.Millisecond
	s := New(eng, 4)
	s.chunkMin = 4
	s.chunkSize = 5
	s.Start()
	defer s.Shutdown()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan Result, 1)
	if err := s.Submit(Request{Tokens: iota1(1000), MaxNew: 5, Ctx: ctx, Emit: (&collector{}).emit, Done: done}); err != nil {
		t.Fatal(err)
	}
	deadline := time.After(2 * time.Second)
	for {
		if _, chunk, _, _ := eng.callCounts(); chunk >= 1 {
			break
		}
		select {
		case <-deadline:
			t.Fatal("prefill never started")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	cancel()
	res := waitResult(t, done)
	if res.Reason != FinishCancelled {
		t.Fatalf("cancel result=%+v", res)
	}
	if res.ReusedTokens != 0 || res.Source != ReuseSourceNone {
		t.Fatalf("cancel leaked reuse=%+v", res)
	}
	follow := make(chan Result, 1)
	if err := s.Submit(Request{Tokens: []int32{9, 9, 9, 9, 9}, MaxNew: 1, Ctx: context.Background(), Emit: (&collector{}).emit, Done: follow}); err != nil {
		t.Fatal(err)
	}
	res2 := waitResult(t, follow)
	if res2.ReusedTokens != 0 || res2.Source != ReuseSourceNone {
		t.Fatalf("follow request leaked reuse=%+v", res2)
	}
}
