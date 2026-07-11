// ABOUTME: Tests for the Phase-E wire protocol and layer-shard pipeline:
// ABOUTME: framing/bounds/checksum hostility, handshake gating, net.Pipe pipeline.
package dist

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"net"
	"strings"
	"testing"
)

// --- protocol framing ---

func TestHelloRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	in := Hello{Version: Version, ConfigHash: 0xdeadbeef, LayerLo: 0, LayerHi: 20, Hidden: 2048, DType: DTypeF32}
	if err := WriteHello(&buf, in); err != nil {
		t.Fatal(err)
	}
	out, err := ReadHello(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if out != in {
		t.Fatalf("hello round trip: got %+v want %+v", out, in)
	}
}

func TestHelloRejectsBadMagic(t *testing.T) {
	buf := bytes.NewBufferString("NOTMAGIC\x00\x00\x00\x00")
	if _, err := ReadHello(buf); err == nil || !strings.Contains(err.Error(), "magic") {
		t.Fatalf("want magic error, got %v", err)
	}
}

func TestHelloRejectsWrongVersion(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteHello(&buf, Hello{Version: 99}); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadHello(&buf); err == nil || !strings.Contains(err.Error(), "version") {
		t.Fatalf("want version error, got %v", err)
	}
}

func TestHelloRejectsOversizedLength(t *testing.T) {
	var buf bytes.Buffer
	buf.WriteString(Magic)
	binary.Write(&buf, binary.LittleEndian, uint32(maxHelloBytes+1))
	if _, err := ReadHello(&buf); err == nil || !strings.Contains(err.Error(), "length") {
		t.Fatalf("want length error, got %v", err)
	}
}

func TestMsgRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	payload := []byte{1, 2, 3, 4, 5}
	if err := WriteMsg(&buf, MsgActivation, payload); err != nil {
		t.Fatal(err)
	}
	typ, got, err := ReadMsg(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if typ != MsgActivation || !bytes.Equal(got, payload) {
		t.Fatalf("round trip mismatch: typ=%d payload=%v", typ, got)
	}
}

func TestMsgEmptyPayload(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteMsg(&buf, MsgPing, nil); err != nil {
		t.Fatal(err)
	}
	typ, got, err := ReadMsg(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if typ != MsgPing || len(got) != 0 {
		t.Fatalf("empty payload mismatch: typ=%d len=%d", typ, len(got))
	}
}

func TestMsgRejectsCorruptChecksum(t *testing.T) {
	var buf bytes.Buffer
	if err := WriteMsg(&buf, MsgActivation, []byte("hello")); err != nil {
		t.Fatal(err)
	}
	b := buf.Bytes()
	b[8] ^= 0xff // flip a payload byte; checksum now mismatches
	if _, _, err := ReadMsg(bytes.NewReader(b)); err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Fatalf("want checksum error, got %v", err)
	}
}

func TestMsgRejectsHostileLength(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(MsgActivation))
	binary.Write(&buf, binary.LittleEndian, uint32(math.MaxUint32)) // hostile
	if _, _, err := ReadMsg(&buf); err == nil || !strings.Contains(err.Error(), "cap") {
		t.Fatalf("want cap error, got %v", err)
	}
}

func TestActivationRoundTrip(t *testing.T) {
	h := ActivationHeader{SeqID: 7, Pos: 42, NTokens: 3, DType: DTypeBF16}
	data := []byte{9, 8, 7, 6}
	p, err := EncodeActivation(h, data)
	if err != nil {
		t.Fatal(err)
	}
	gh, gd, err := DecodeActivation(p)
	if err != nil {
		t.Fatal(err)
	}
	if gh != h || !bytes.Equal(gd, data) {
		t.Fatalf("activation round trip: %+v %v", gh, gd)
	}
}

func TestDecodeActivationRejectsShortPayload(t *testing.T) {
	if _, _, err := DecodeActivation([]byte{1, 2, 3}); err == nil {
		t.Fatal("want error on short payload")
	}
}

// --- handshake gating ---

func TestCheckPeerGates(t *testing.T) {
	mine := Hello{Version: Version, ConfigHash: 1, LayerLo: 0, LayerHi: 20, Hidden: 2048, DType: DTypeF32}
	good := Hello{Version: Version, ConfigHash: 1, LayerLo: 20, LayerHi: 40, Hidden: 2048, DType: DTypeF32}
	if err := CheckPeer(mine, good); err != nil {
		t.Fatalf("good peer rejected: %v", err)
	}
	cases := []struct {
		name string
		mut  func(*Hello)
	}{
		{"config", func(h *Hello) { h.ConfigHash = 2 }},
		{"hidden", func(h *Hello) { h.Hidden = 4096 }},
		{"dtype", func(h *Hello) { h.DType = DTypeBF16 }},
		{"gap", func(h *Hello) { h.LayerLo = 21 }},
	}
	for _, c := range cases {
		bad := good
		c.mut(&bad)
		if err := CheckPeer(mine, bad); err == nil {
			t.Errorf("%s mismatch not rejected", c.name)
		}
	}
}

// --- pipeline over net.Pipe with fake runners ---

// addRunner adds a constant to every f32 activation and records resets. The
// final shard doubles instead, standing in for the logits head.
type addRunner struct {
	add    float32
	double bool
	resets []uint32
}

func (r *addRunner) Forward(seq, pos uint32, ntokens int, in []byte) ([]byte, error) {
	out := make([]byte, len(in))
	for i := 0; i+4 <= len(in); i += 4 {
		v := math.Float32frombits(binary.LittleEndian.Uint32(in[i:]))
		if r.double {
			v *= 2
		} else {
			v += r.add
		}
		binary.LittleEndian.PutUint32(out[i:], math.Float32bits(v))
	}
	return out, nil
}

func (r *addRunner) Reset(seq uint32) error {
	r.resets = append(r.resets, seq)
	return nil
}

func f32bytes(vals ...float32) []byte {
	b := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

// startWorker serves hello/r on one end of a net.Pipe and returns the other.
// The worker end is closed when Serve returns so a rejected handshake
// unblocks the dialer instead of deadlocking the synchronous pipe.
func startWorker(t *testing.T, hello Hello, r ShardRunner, final, wantServeOK bool) io.ReadWriteCloser {
	t.Helper()
	c1, c2 := net.Pipe()
	w := &Worker{Hello: hello, Runner: r, Final: final}
	go func() {
		err := w.Serve(c2)
		c2.Close()
		if wantServeOK && err != nil {
			t.Errorf("worker serve: %v", err)
		}
	}()
	t.Cleanup(func() { c1.Close() })
	return c1
}

func TestTwoShardPipeline(t *testing.T) {
	const hash = 0xabc
	// coordinator local shard: layers [0,20), +1.0
	local := &addRunner{add: 1}
	// worker shard: layers [20,40), final (doubles)
	wr := &addRunner{double: true}
	conn := startWorker(t, Hello{Version: Version, ConfigHash: hash, LayerLo: 20, LayerHi: 40, Hidden: 2, DType: DTypeF32}, wr, true, true)

	hop, err := DialHop(conn, Hello{Version: Version, ConfigHash: hash, LayerLo: 0, LayerHi: 20, Hidden: 2, DType: DTypeF32})
	if err != nil {
		t.Fatal(err)
	}
	p := &Pipeline{Local: local, Hops: []*Hop{hop}}

	out, err := p.Forward(1, 0, 1, f32bytes(3, 5))
	if err != nil {
		t.Fatal(err)
	}
	// (3+1)*2=8, (5+1)*2=12
	want := f32bytes(8, 12)
	if !bytes.Equal(out, want) {
		t.Fatalf("pipeline output %v want %v", out, want)
	}

	if err := p.Reset(1); err != nil {
		t.Fatal(err)
	}
	if len(local.resets) != 1 || local.resets[0] != 1 {
		t.Fatalf("local resets %v", local.resets)
	}
	// The worker handles the reset asynchronously to the write; force a sync
	// point by pushing another forward through.
	if _, err := p.Forward(2, 0, 1, f32bytes(1, 1)); err != nil {
		t.Fatal(err)
	}
	if len(wr.resets) != 1 || wr.resets[0] != 1 {
		t.Fatalf("worker resets %v", wr.resets)
	}
}

func TestDialHopRejectsWrongModel(t *testing.T) {
	conn := startWorker(t, Hello{Version: Version, ConfigHash: 999, LayerLo: 20, LayerHi: 40, Hidden: 2, DType: DTypeF32}, &addRunner{}, false, false)
	_, err := DialHop(conn, Hello{Version: Version, ConfigHash: 1, LayerLo: 0, LayerHi: 20, Hidden: 2, DType: DTypeF32})
	if err == nil {
		t.Fatal("want handshake rejection for wrong model")
	}
}

func TestWorkerRejectsNonContiguousLayers(t *testing.T) {
	conn := startWorker(t, Hello{Version: Version, ConfigHash: 1, LayerLo: 20, LayerHi: 40, Hidden: 2, DType: DTypeF32}, &addRunner{}, false, false)
	// coordinator claims [0,10): gap before worker's 20
	_, err := DialHop(conn, Hello{Version: Version, ConfigHash: 1, LayerLo: 0, LayerHi: 10, Hidden: 2, DType: DTypeF32})
	if err == nil {
		t.Fatal("want rejection for non-contiguous layer ranges")
	}
}
