// ABOUTME: Phase-E distributed inference wire protocol (magic FCNDIST1):
// ABOUTME: JSON hello handshake + framed, checksummed binary messages.
//
// The protocol carries the residual hidden activation between layer shards.
// It is deliberately transport-agnostic: anything satisfying io.ReadWriter
// (TCP, net.Pipe in tests, later an RDMA rendezvous) can carry it.
//
// Framing (all integers little-endian):
//
//	handshake: magic 8 B "FCNDIST1" | helloLen u32 | helloLen B JSON (Hello)
//	message:   type u32 | payloadLen u32 | payload | u64 FNV-1a64(payload)
//
// Load-side rules match the session/safetensors standard: no length read off
// the wire is trusted — every one is bounds-checked against a hard cap before
// allocation, and every payload checksum is verified before use.
package dist

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
)

const (
	// Magic identifies a fucina distributed-inference connection.
	Magic = "FCNDIST1"
	// Version is the current protocol version; Handshake rejects any other.
	Version = 1

	maxHelloBytes   = 1 << 16 // JSON handshake cap
	maxPayloadBytes = 1 << 30 // 1 GiB ≫ any activation frame (hidden×tokens×4)
)

// Message types.
const (
	// MsgActivation carries the residual hidden activation into a shard.
	MsgActivation uint32 = 1
	// MsgLogits carries the final-shard logits back to the coordinator.
	MsgLogits uint32 = 2
	// MsgSeqReset tells a shard to drop one sequence's KV/recurrent state.
	MsgSeqReset uint32 = 3
	// MsgPing/MsgPong are liveness probes.
	MsgPing uint32 = 4
	MsgPong uint32 = 5
)

// Activation dtypes on the wire.
const (
	DTypeF32  uint32 = 0
	DTypeBF16 uint32 = 1
)

// Hello is the one-time JSON handshake. Both ends send one; each validates
// the peer's against its own expectations. Everything that would make a
// shard silently compute garbage — wrong model, wrong split, wrong dtype —
// must be pinned here so it fails at connect time instead.
type Hello struct {
	Version    int    `json:"version"`
	ConfigHash uint64 `json:"config_hash"` // FNV-1a64 of model config bytes
	LayerLo    int    `json:"layer_lo"`    // shard layer range [lo, hi)
	LayerHi    int    `json:"layer_hi"`
	Hidden     int    `json:"hidden"` // residual width
	DType      uint32 `json:"dtype"`  // activation dtype on the wire
}

// ActivationHeader prefixes a MsgActivation / MsgLogits payload.
// Fixed 16 bytes, followed by the raw tensor bytes.
type ActivationHeader struct {
	SeqID   uint32 // coordinator-assigned sequence id
	Pos     uint32 // position of the first token in this frame
	NTokens uint32 // token count in this frame
	DType   uint32
}

const actHeaderBytes = 16

func fnv64(b []byte) uint64 {
	h := fnv.New64a()
	h.Write(b)
	return h.Sum64()
}

// WriteHello sends the magic and this end's Hello.
func WriteHello(w io.Writer, h Hello) error {
	j, err := json.Marshal(h)
	if err != nil {
		return fmt.Errorf("dist: marshal hello: %w", err)
	}
	if len(j) > maxHelloBytes {
		return fmt.Errorf("dist: hello %d bytes exceeds cap %d", len(j), maxHelloBytes)
	}
	buf := make([]byte, 0, len(Magic)+4+len(j))
	buf = append(buf, Magic...)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(j)))
	buf = append(buf, j...)
	_, err = w.Write(buf)
	return err
}

// ReadHello reads and validates the peer's magic + Hello.
func ReadHello(r io.Reader) (Hello, error) {
	var h Hello
	magic := make([]byte, len(Magic))
	if _, err := io.ReadFull(r, magic); err != nil {
		return h, fmt.Errorf("dist: read magic: %w", err)
	}
	if string(magic) != Magic {
		return h, fmt.Errorf("dist: bad magic %q", magic)
	}
	var n uint32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return h, fmt.Errorf("dist: read hello length: %w", err)
	}
	if n == 0 || n > maxHelloBytes {
		return h, fmt.Errorf("dist: hello length %d outside (0, %d]", n, maxHelloBytes)
	}
	j := make([]byte, n)
	if _, err := io.ReadFull(r, j); err != nil {
		return h, fmt.Errorf("dist: read hello body: %w", err)
	}
	if err := json.Unmarshal(j, &h); err != nil {
		return h, fmt.Errorf("dist: parse hello: %w", err)
	}
	if h.Version != Version {
		return h, fmt.Errorf("dist: peer protocol version %d, want %d", h.Version, Version)
	}
	return h, nil
}

// CheckPeer validates that a peer's Hello is compatible with ours for one
// pipeline hop: same model, same hidden width, same dtype, and the peer's
// layer range must begin where ours ends (contiguous pipeline).
func CheckPeer(mine, peer Hello) error {
	if peer.ConfigHash != mine.ConfigHash {
		return fmt.Errorf("dist: peer model config hash %016x != ours %016x", peer.ConfigHash, mine.ConfigHash)
	}
	if peer.Hidden != mine.Hidden {
		return fmt.Errorf("dist: peer hidden %d != ours %d", peer.Hidden, mine.Hidden)
	}
	if peer.DType != mine.DType {
		return fmt.Errorf("dist: peer dtype %d != ours %d", peer.DType, mine.DType)
	}
	if peer.LayerLo != mine.LayerHi {
		return fmt.Errorf("dist: peer layers [%d,%d) not contiguous with ours [%d,%d)",
			peer.LayerLo, peer.LayerHi, mine.LayerLo, mine.LayerHi)
	}
	return nil
}

// WriteMsg sends one framed, checksummed message.
func WriteMsg(w io.Writer, typ uint32, payload []byte) error {
	if len(payload) > maxPayloadBytes {
		return fmt.Errorf("dist: payload %d bytes exceeds cap %d", len(payload), maxPayloadBytes)
	}
	hdr := make([]byte, 8)
	binary.LittleEndian.PutUint32(hdr[0:], typ)
	binary.LittleEndian.PutUint32(hdr[4:], uint32(len(payload)))
	if _, err := w.Write(hdr); err != nil {
		return err
	}
	if len(payload) > 0 {
		if _, err := w.Write(payload); err != nil {
			return err
		}
	}
	var sum [8]byte
	binary.LittleEndian.PutUint64(sum[:], fnv64(payload))
	_, err := w.Write(sum[:])
	return err
}

// ReadMsg reads one framed message, verifying length bounds and checksum.
func ReadMsg(r io.Reader) (typ uint32, payload []byte, err error) {
	hdr := make([]byte, 8)
	if _, err = io.ReadFull(r, hdr); err != nil {
		return 0, nil, fmt.Errorf("dist: read frame header: %w", err)
	}
	typ = binary.LittleEndian.Uint32(hdr[0:])
	n := binary.LittleEndian.Uint32(hdr[4:])
	if n > maxPayloadBytes {
		return 0, nil, fmt.Errorf("dist: payload length %d exceeds cap %d", n, maxPayloadBytes)
	}
	payload = make([]byte, n)
	if _, err = io.ReadFull(r, payload); err != nil {
		return 0, nil, fmt.Errorf("dist: read payload: %w", err)
	}
	var sum [8]byte
	if _, err = io.ReadFull(r, sum[:]); err != nil {
		return 0, nil, fmt.Errorf("dist: read checksum: %w", err)
	}
	if got, want := fnv64(payload), binary.LittleEndian.Uint64(sum[:]); got != want {
		return 0, nil, fmt.Errorf("dist: payload checksum %016x != %016x", got, want)
	}
	return typ, payload, nil
}

// EncodeActivation packs a header + raw tensor bytes into one payload.
func EncodeActivation(h ActivationHeader, data []byte) ([]byte, error) {
	if len(data) > maxPayloadBytes-actHeaderBytes {
		return nil, fmt.Errorf("dist: activation %d bytes exceeds cap", len(data))
	}
	buf := make([]byte, actHeaderBytes+len(data))
	binary.LittleEndian.PutUint32(buf[0:], h.SeqID)
	binary.LittleEndian.PutUint32(buf[4:], h.Pos)
	binary.LittleEndian.PutUint32(buf[8:], h.NTokens)
	binary.LittleEndian.PutUint32(buf[12:], h.DType)
	copy(buf[actHeaderBytes:], data)
	return buf, nil
}

// DecodeActivation unpacks a MsgActivation/MsgLogits payload. The returned
// data slice aliases payload (no copy) — callers must not retain it past the
// payload's lifetime.
func DecodeActivation(payload []byte) (ActivationHeader, []byte, error) {
	var h ActivationHeader
	if len(payload) < actHeaderBytes {
		return h, nil, fmt.Errorf("dist: activation payload %d bytes < header %d", len(payload), actHeaderBytes)
	}
	h.SeqID = binary.LittleEndian.Uint32(payload[0:])
	h.Pos = binary.LittleEndian.Uint32(payload[4:])
	h.NTokens = binary.LittleEndian.Uint32(payload[8:])
	h.DType = binary.LittleEndian.Uint32(payload[12:])
	return h, payload[actHeaderBytes:], nil
}
