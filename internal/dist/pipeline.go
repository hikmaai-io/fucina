// ABOUTME: Phase-E layer-shard pipeline — a headless Worker serving one layer
// ABOUTME: range over the wire protocol, and a coordinator-side Hop client.
//
// The CUDA engine is isolated behind ShardRunner so the pipeline logic is
// unit-testable over net.Pipe with a fake runner (the residency-controller
// pattern: policy proven without a checkpoint). A real runner wraps the
// engine's partial-forward entry point for layers [lo, hi).
package dist

import (
	"fmt"
	"io"
	"log"
	"net"
	"sync"
)

// ShardRunner executes this node's layer range on one activation frame.
// Implementations own their KV/recurrent state per sequence.
type ShardRunner interface {
	// Forward runs layers [lo,hi) over ntokens activations starting at pos
	// for sequence seq, returning the output activation (or logits when this
	// is the final shard). The returned slice must not alias in.
	Forward(seq uint32, pos uint32, ntokens int, in []byte) ([]byte, error)
	// Reset drops all state for a sequence.
	Reset(seq uint32) error
}

// Worker serves one shard over a single upstream connection: it answers the
// handshake, then loops Activation→Forward→reply until EOF.
type Worker struct {
	Hello  Hello
	Runner ShardRunner
	// Final marks the last shard: replies carry MsgLogits instead of
	// MsgActivation so the coordinator knows the pipeline is complete.
	Final bool
}

// Serve handles one connection until EOF or error. EOF is a clean shutdown.
func (w *Worker) Serve(conn io.ReadWriter) error {
	peer, err := ReadHello(conn)
	if err != nil {
		return err
	}
	// The upstream's range must end where ours begins, on the same model.
	if peer.ConfigHash != w.Hello.ConfigHash {
		return fmt.Errorf("dist: worker: peer config hash %016x != ours %016x", peer.ConfigHash, w.Hello.ConfigHash)
	}
	if peer.Hidden != w.Hello.Hidden || peer.DType != w.Hello.DType {
		return fmt.Errorf("dist: worker: peer geometry mismatch (hidden %d dtype %d vs %d %d)",
			peer.Hidden, peer.DType, w.Hello.Hidden, w.Hello.DType)
	}
	if peer.LayerHi != w.Hello.LayerLo {
		return fmt.Errorf("dist: worker: peer layers [%d,%d) not contiguous with ours [%d,%d)",
			peer.LayerLo, peer.LayerHi, w.Hello.LayerLo, w.Hello.LayerHi)
	}
	if err := WriteHello(conn, w.Hello); err != nil {
		return err
	}
	for {
		typ, payload, err := ReadMsg(conn)
		if err != nil {
			if err == io.EOF || errIsEOF(err) {
				return nil
			}
			return err
		}
		switch typ {
		case MsgActivation:
			h, data, err := DecodeActivation(payload)
			if err != nil {
				return err
			}
			out, err := w.Runner.Forward(h.SeqID, h.Pos, int(h.NTokens), data)
			if err != nil {
				return fmt.Errorf("dist: worker forward: %w", err)
			}
			reply, err := EncodeActivation(h, out)
			if err != nil {
				return err
			}
			rt := MsgActivation
			if w.Final {
				rt = MsgLogits
			}
			if err := WriteMsg(conn, rt, reply); err != nil {
				return err
			}
		case MsgSeqReset:
			h, _, err := DecodeActivation(payload)
			if err != nil {
				return err
			}
			if err := w.Runner.Reset(h.SeqID); err != nil {
				return fmt.Errorf("dist: worker reset: %w", err)
			}
		case MsgPing:
			if err := WriteMsg(conn, MsgPong, nil); err != nil {
				return err
			}
		default:
			return fmt.Errorf("dist: worker: unexpected message type %d", typ)
		}
	}
}

func errIsEOF(err error) bool {
	return err != nil && (err == io.EOF ||
		// framed reads wrap EOF; unwrapping via string is brittle, so check
		// the common io errors explicitly
		errUnwrapIs(err, io.EOF) || errUnwrapIs(err, io.ErrUnexpectedEOF))
}

func errUnwrapIs(err, target error) bool {
	for err != nil {
		if err == target {
			return true
		}
		u, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}

// Hop is the coordinator's client for one downstream shard.
type Hop struct {
	mu   sync.Mutex
	conn io.ReadWriter
}

// Dial connects to a worker over conn: sends our Hello, validates the
// worker's reply against CheckPeer.
func DialHop(conn io.ReadWriter, mine Hello) (*Hop, error) {
	if err := WriteHello(conn, mine); err != nil {
		return nil, err
	}
	peer, err := ReadHello(conn)
	if err != nil {
		return nil, err
	}
	if err := CheckPeer(mine, peer); err != nil {
		return nil, err
	}
	return &Hop{conn: conn}, nil
}

// Forward sends one activation frame and waits for the shard's reply.
// It returns the reply payload's tensor bytes and whether they are logits.
func (h *Hop) Forward(hdr ActivationHeader, data []byte) ([]byte, bool, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	payload, err := EncodeActivation(hdr, data)
	if err != nil {
		return nil, false, err
	}
	if err := WriteMsg(h.conn, MsgActivation, payload); err != nil {
		return nil, false, err
	}
	typ, reply, err := ReadMsg(h.conn)
	if err != nil {
		return nil, false, err
	}
	if typ != MsgActivation && typ != MsgLogits {
		return nil, false, fmt.Errorf("dist: hop: unexpected reply type %d", typ)
	}
	_, out, err := DecodeActivation(reply)
	if err != nil {
		return nil, false, err
	}
	// Copy out of the frame buffer so callers may retain the result.
	cp := make([]byte, len(out))
	copy(cp, out)
	return cp, typ == MsgLogits, nil
}

// Reset tells the shard to drop a sequence's state.
func (h *Hop) Reset(seq uint32) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	payload, err := EncodeActivation(ActivationHeader{SeqID: seq}, nil)
	if err != nil {
		return err
	}
	return WriteMsg(h.conn, MsgSeqReset, payload)
}

// Pipeline chains hops in layer order. The coordinator runs its own local
// shard (if any) first, then forwards through each hop.
type Pipeline struct {
	Local ShardRunner // optional shard 0 on the coordinator; may be nil
	Hops  []*Hop      // downstream shards in layer order
}

// Forward pushes one frame through local shard then every hop, returning the
// final shard's output (logits when the last worker is Final).
func (p *Pipeline) Forward(seq, pos uint32, ntokens int, act []byte) ([]byte, error) {
	cur := act
	var err error
	if p.Local != nil {
		cur, err = p.Local.Forward(seq, pos, ntokens, cur)
		if err != nil {
			return nil, fmt.Errorf("dist: local shard: %w", err)
		}
	}
	for i, hop := range p.Hops {
		out, _, err := hop.Forward(ActivationHeader{SeqID: seq, Pos: pos, NTokens: uint32(ntokens), DType: DTypeF32}, cur)
		if err != nil {
			return nil, fmt.Errorf("dist: hop %d: %w", i, err)
		}
		cur = out
	}
	return cur, nil
}

// Reset fans a sequence reset out to the local shard and every hop.
func (p *Pipeline) Reset(seq uint32) error {
	if p.Local != nil {
		if err := p.Local.Reset(seq); err != nil {
			return err
		}
	}
	for i, hop := range p.Hops {
		if err := hop.Reset(seq); err != nil {
			return fmt.Errorf("dist: hop %d reset: %w", i, err)
		}
	}
	return nil
}

// ListenAndServe runs a worker on a TCP listener, one connection at a time
// (a shard serves exactly one coordinator).
func ListenAndServe(addr string, w *Worker) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	defer ln.Close()
	for {
		conn, err := ln.Accept()
		if err != nil {
			return err
		}
		if tc, ok := conn.(*net.TCPConn); ok {
			tc.SetNoDelay(true) // decode is latency-bound: never Nagle a token
		}
		if err := w.Serve(conn); err != nil {
			log.Printf("dist: worker connection ended: %v", err)
		}
		conn.Close()
	}
}
