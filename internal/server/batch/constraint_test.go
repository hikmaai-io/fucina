package batch

import (
	"context"
	"encoding/json"
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/hikmaai-io/fucina/internal/grammar"
)

// grammarEngine is a CPU-only model of the exact-logit batch ABI. Each prompt's
// first token selects a byte script. Device samples are deliberately discarded by
// the scheduler; CopyLogits exposes the scripted distribution for grammar masking.
type grammarEngine struct {
	mu sync.Mutex

	scripts map[int32][]byte
	pos     map[int]int
	script  map[int][]byte
	next    int
	lastAdd int
	last    []int32

	exactCalls   int
	specCalls    int
	restoreCalls int
	exportCalls  int
	exported     []int32
}

func newGrammarEngine(scripts map[int32]string) *grammarEngine {
	b := make(map[int32][]byte, len(scripts))
	for k, v := range scripts {
		b[k] = []byte(v)
	}
	return &grammarEngine{scripts: b, pos: map[int]int{}, script: map[int][]byte{}}
}

func (e *grammarEngine) Capacity() int { return 8 }
func (e *grammarEngine) AddSeq(prompt []int32, _ SeqParams) (int, int32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	slot := e.next
	e.next++
	e.script[slot] = e.scripts[prompt[0]]
	e.pos[slot] = 1
	e.lastAdd = slot
	return slot, int32(e.script[slot][0]), nil
}
func (e *grammarEngine) StepBatch(active []int32, inputs []int32) ([][]int32, error) {
	out, err := e.StepBatchExact(active, inputs)
	runs := make([][]int32, len(out))
	for i, tok := range out {
		runs[i] = []int32{tok}
	}
	return runs, err
}
func (e *grammarEngine) StepBatchExact(active []int32, _ []int32) ([]int32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.exactCalls++
	e.last = make([]int32, len(active))
	for i, slot32 := range active {
		slot := int(slot32)
		p := e.pos[slot]
		if p >= len(e.script[slot]) {
			e.last[i] = 128 // EOS
		} else {
			e.last[i] = int32(e.script[slot][p])
			e.pos[slot] = p + 1
		}
	}
	return append([]int32(nil), e.last...), nil
}
func (e *grammarEngine) StepBatchSpec(reqs []SpecReq) ([][]int32, error) {
	e.mu.Lock()
	e.specCalls++
	e.mu.Unlock()
	out := make([][]int32, len(reqs))
	for i := range out {
		out[i] = []int32{'!'}
	}
	return out, nil
}
func (e *grammarEngine) CopyLogits(rows int, batched bool) ([]float32, int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	ids := e.last
	if !batched {
		ids = []int32{int32(e.script[e.lastAdd][0])}
	}
	logits := make([]float32, rows*129)
	for i := range logits {
		logits[i] = -100
	}
	for r := 0; r < rows; r++ {
		logits[r*129+int(ids[r])] = 100
	}
	return logits, 129, nil
}
func (e *grammarEngine) OpenSeq(prompt []int32, _ SeqParams) (int, int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	slot := e.next
	e.next++
	e.script[slot] = e.scripts[prompt[0]]
	e.pos[slot] = 0
	return slot, 0, nil
}
func (e *grammarEngine) PrefillChunk(slot int, _ []int32, last bool) (int32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !last {
		return 0, nil
	}
	e.pos[slot] = 1
	e.lastAdd = slot
	e.last = []int32{int32(e.script[slot][0])}
	return e.last[0], nil
}
func (e *grammarEngine) RestoreSession(prompt []int32, _ SeqParams, snap StateSnapshot) (int, int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.restoreCalls++
	slot := e.next
	e.next++
	e.script[slot] = e.scripts[prompt[0]]
	e.pos[slot] = 0
	return slot, len(snap.Tokens), nil
}
func (e *grammarEngine) ExportSession(_ int, tokens []int32) (*StateSnapshot, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.exportCalls++
	e.exported = append([]int32(nil), tokens...)
	return &StateSnapshot{Tokens: append([]int32(nil), tokens...), State: []byte("ok")}, nil
}
func (e *grammarEngine) RemoveSeq(slot int) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.pos, slot)
	delete(e.script, slot)
	return nil
}

func byteConstraint(t *testing.T, schema string) grammar.Constraint {
	t.Helper()
	pieces := make([][]byte, 129)
	for i := 0; i < 128; i++ {
		pieces[i] = []byte{byte(i)}
	}
	c, err := grammar.NewJSONSchema(pieces, 128, json.RawMessage(schema))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func submitGrammar(t *testing.T, sched *Scheduler, selector int32, schema string, max int) (*collector, <-chan Result) {
	t.Helper()
	col := &collector{}
	done := make(chan Result, 1)
	if err := sched.Submit(Request{
		Tokens: []int32{selector}, Params: SeqParams{RepeatPenalty: 1},
		Stops: []int32{128}, MaxNew: max, Ctx: context.Background(),
		Constraint: byteConstraint(t, schema),
		CloseConstraint: func(p []byte) []int32 {
			out := make([]int32, len(p))
			for i, b := range p {
				out[i] = int32(b)
			}
			return out
		},
		Emit: col.emit, Done: done,
	}); err != nil {
		t.Fatal(err)
	}
	return col, done
}

func bytesOf(ids []int32) string {
	b := make([]byte, 0, len(ids))
	for _, id := range ids {
		if id >= 0 && id < 128 {
			b = append(b, byte(id))
		}
	}
	return string(b)
}

func TestConstraintPerSlotIsolationDifferentSchemas(t *testing.T) {
	eng := newGrammarEngine(map[int32]string{1: `{"a":1}`, 2: `{"b":"red"}`})
	s := New(eng, 8)
	if !s.SupportsConstraints() {
		t.Fatal("exact/logit engine was not recognized")
	}
	s.Start()
	defer s.Shutdown()

	ca, da := submitGrammar(t, s, 1,
		`{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}`, 32)
	cb, db := submitGrammar(t, s, 2,
		`{"type":"object","properties":{"b":{"enum":["red","green"]}},"required":["b"]}`, 32)
	if r := waitResult(t, da); r.Reason != FinishStop {
		t.Fatalf("schema a finish=%+v", r)
	}
	if r := waitResult(t, db); r.Reason != FinishStop {
		t.Fatalf("schema b finish=%+v", r)
	}
	if got := bytesOf(ca.got()); got != `{"a":1}` {
		t.Errorf("slot a=%q", got)
	}
	if got := bytesOf(cb.got()); got != `{"b":"red"}` {
		t.Errorf("slot b=%q", got)
	}
	if eng.exactCalls == 0 || eng.specCalls != 0 {
		t.Errorf("exact=%d spec=%d; constrained slots must cleanly disable spec", eng.exactCalls, eng.specCalls)
	}
}

func TestConstraintNestedSchemaAndArrayOfObjects(t *testing.T) {
	schema := `{"type":"object","properties":{"groups":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"items":{"type":"array","items":{"type":"integer"}}},"required":["name","items"]}}},"required":["groups"]}`
	want := `{"groups":[{"name":"x","items":[1,2]},{"name":"y","items":[]}]}`
	eng := newGrammarEngine(map[int32]string{1: want})
	s := New(eng, 4)
	s.Start()
	defer s.Shutdown()
	col, done := submitGrammar(t, s, 1, schema, 256)
	if r := waitResult(t, done); r.Reason != FinishStop {
		t.Fatalf("finish=%+v", r)
	}
	got := bytesOf(col.got())
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
	var v map[string][]struct {
		Name  string `json:"name"`
		Items []int  `json:"items"`
	}
	if err := json.Unmarshal([]byte(got), &v); err != nil || len(v["groups"]) != 2 {
		t.Fatalf("nested array output invalid: %v %#v", err, v)
	}
}

func TestConstraintEnumMasksCompetingValue(t *testing.T) {
	// The script's illegal "blue" peak is masked at 'b'; after that the remaining
	// equal logits deterministically select the first legal enum continuation.
	schema := `{"type":"object","properties":{"color":{"enum":["red","green"]}},"required":["color"]}`
	eng := newGrammarEngine(map[int32]string{1: `{"color":"blue"}`})
	s := New(eng, 4)
	s.Start()
	defer s.Shutdown()
	col, done := submitGrammar(t, s, 1, schema, 64)
	waitResult(t, done)
	got := bytesOf(col.got())
	var obj map[string]string
	if err := json.Unmarshal([]byte(got), &obj); err != nil {
		t.Fatalf("enum output invalid: %v (%q)", err, got)
	}
	if !reflect.DeepEqual(obj, map[string]string{"color": "green"}) &&
		!reflect.DeepEqual(obj, map[string]string{"color": "red"}) {
		t.Fatalf("enum escaped constraint: %q", got)
	}
}

func TestConstraintSessionRestoreExportUsesExactGrammarPath(t *testing.T) {
	schema := `{"type":"object","properties":{"answer":{"enum":["yes","no"]}},"required":["answer"]}`
	eng := newGrammarEngine(map[int32]string{1: `{"answer":"yes"}`})
	s := New(eng, 4)
	s.Start()
	defer s.Shutdown()
	col := &collector{}
	done := make(chan Result, 1)
	if err := s.Submit(Request{
		Tokens: []int32{1, 99}, Params: SeqParams{RepeatPenalty: 1}, Stops: []int32{128},
		MaxNew: 64, Ctx: context.Background(), Constraint: byteConstraint(t, schema),
		CloseConstraint: func(p []byte) []int32 {
			ids := make([]int32, len(p))
			for i, b := range p {
				ids[i] = int32(b)
			}
			return ids
		},
		SessionState:   &StateSnapshot{Tokens: []int32{1}, State: []byte("old")},
		PersistSession: true, Emit: col.emit, Done: done,
	}); err != nil {
		t.Fatal(err)
	}
	res := waitResult(t, done)
	if got := bytesOf(col.got()); got != `{"answer":"yes"}` {
		t.Fatalf("restored constrained output=%q", got)
	}
	if res.SessionErr != nil || res.SessionState == nil || eng.restoreCalls != 1 || eng.exportCalls != 1 {
		t.Fatalf("session result=%+v restore=%d export=%d", res, eng.restoreCalls, eng.exportCalls)
	}
	if eng.exactCalls == 0 || eng.specCalls != 0 {
		t.Fatalf("session grammar path exact=%d spec=%d", eng.exactCalls, eng.specCalls)
	}
}

func TestConstraintMaxTokenForceCloseConforms(t *testing.T) {
	schema := `{"type":"object","properties":{"rows":{"type":"array","items":{"type":"object","properties":{"kind":{"enum":["red","green"]},"n":{"type":"integer"}},"required":["kind","n"]}}},"required":["rows"]}`
	prefix := `{"rows":[{"kind":"r`
	eng := newGrammarEngine(map[int32]string{1: prefix})
	s := New(eng, 4)
	s.Start()
	defer s.Shutdown()
	col, done := submitGrammar(t, s, 1, schema, len(prefix))
	res := waitResult(t, done)
	if res.Reason != FinishLength {
		t.Fatalf("finish=%+v want length", res)
	}
	got := bytesOf(col.got())
	var obj struct {
		Rows []struct {
			Kind string `json:"kind"`
			N    int    `json:"n"`
		} `json:"rows"`
	}
	if err := json.Unmarshal([]byte(got), &obj); err != nil {
		t.Fatalf("force-close emitted unterminated JSON: %v (%q)", err, got)
	}
	if len(obj.Rows) != 1 || obj.Rows[0].Kind != "red" || obj.Rows[0].N != 0 {
		t.Fatalf("force-close did not satisfy nested required/enum schema: %#v (%q)", obj, got)
	}
	if math.IsNaN(float64(res.Generated)) || res.Generated <= len(prefix) {
		t.Fatalf("closing suffix was not emitted: result=%+v output=%q", res, got)
	}
}
