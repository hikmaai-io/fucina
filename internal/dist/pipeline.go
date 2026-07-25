// ABOUTME: Phase-E layer-shard pipeline — a headless Worker serving one layer
// ABOUTME: range over the wire protocol, and a coordinator-side Hop client.
//
// The CUDA engine is isolated behind ShardRunner so the pipeline logic is
// unit-testable over net.Pipe with a fake runner (the residency-controller
// pattern: policy proven without a checkpoint). A real runner wraps the
// engine's partial-forward entry point for layers [lo, hi).
package dist

import (
	"context"
	"errors"
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
	if w == nil || w.Runner == nil {
		return fmt.Errorf("dist: worker has no shard runner")
	}
	if w.Final != w.Hello.Final {
		return fmt.Errorf("dist: worker final=%v disagrees with hello final=%v", w.Final, w.Hello.Final)
	}
	if err := ValidateHello(w.Hello); err != nil {
		return fmt.Errorf("dist: worker hello: %w", err)
	}
	peer, err := ReadHello(conn)
	if err != nil {
		return err
	}
	// The upstream's range must end where ours begins, on the same model.
	if peer.Final {
		return fmt.Errorf("dist: worker: upstream range [%d,%d) is already final", peer.LayerLo, peer.LayerHi)
	}
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
	// Sequence lifecycle is connection-scoped. A reconnect cannot inherit stale
	// worker state: every still-live sequence is reset when this connection ends.
	next := make(map[uint32]uint32)
	defer func() {
		for seq := range next {
			_ = w.Runner.Reset(seq)
		}
	}()
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
			if err := ValidateActivation(h, data, w.Hello.Hidden, w.Hello.DType); err != nil {
				return err
			}
			if h.NTokens > ^uint32(0)-h.Pos {
				return fmt.Errorf("dist: worker sequence %d position overflow", h.SeqID)
			}
			want, exists := next[h.SeqID]
			if !exists {
				want = 0
			}
			if h.Pos != want {
				return fmt.Errorf("dist: worker sequence %d position %d, want %d", h.SeqID, h.Pos, want)
			}
			out, err := w.Runner.Forward(h.SeqID, h.Pos, int(h.NTokens), data)
			if err != nil {
				return fmt.Errorf("dist: worker forward: %w", err)
			}
			next[h.SeqID] = h.Pos + h.NTokens
			if !w.Final {
				if err := ValidateActivation(h, out, w.Hello.Hidden, w.Hello.DType); err != nil {
					return fmt.Errorf("dist: worker output: %w", err)
				}
			} else if len(out) == 0 || len(out)%4 != 0 {
				return fmt.Errorf("dist: worker logits byte length %d is invalid", len(out))
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
			delete(next, h.SeqID)
			if err := WriteMsg(conn, MsgAck, nil); err != nil {
				return err
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
	mine Hello
	peer Hello
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
	return &Hop{conn: conn, mine: mine, peer: peer}, nil
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
	if gotFinal := typ == MsgLogits; gotFinal != h.peer.Final {
		return nil, false, fmt.Errorf("dist: hop: reply final=%v, handshake promised %v", gotFinal, h.peer.Final)
	}
	rh, out, err := DecodeActivation(reply)
	if err != nil {
		return nil, false, err
	}
	if rh != hdr {
		return nil, false, fmt.Errorf("dist: hop: reply identity %+v != request %+v", rh, hdr)
	}
	final := typ == MsgLogits
	if !final {
		if err := ValidateActivation(rh, out, h.peer.Hidden, h.peer.DType); err != nil {
			return nil, false, err
		}
	} else if len(out) == 0 || len(out)%4 != 0 {
		return nil, false, fmt.Errorf("dist: hop: invalid logits byte length %d", len(out))
	}
	// Copy out of the frame buffer so callers may retain the result.
	cp := make([]byte, len(out))
	copy(cp, out)
	return cp, final, nil
}

// Reset tells the shard to drop a sequence's state.
func (h *Hop) Reset(seq uint32) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	payload, err := EncodeActivation(ActivationHeader{SeqID: seq}, nil)
	if err != nil {
		return err
	}
	if err := WriteMsg(h.conn, MsgSeqReset, payload); err != nil {
		return err
	}
	typ, body, err := ReadMsg(h.conn)
	if err != nil {
		return err
	}
	if typ != MsgAck || len(body) != 0 {
		return fmt.Errorf("dist: hop: reset reply type=%d bytes=%d, want empty ack", typ, len(body))
	}
	return nil
}

// Ping performs a synchronous liveness round-trip.
func (h *Hop) Ping() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if err := WriteMsg(h.conn, MsgPing, nil); err != nil {
		return err
	}
	typ, body, err := ReadMsg(h.conn)
	if err != nil {
		return err
	}
	if typ != MsgPong || len(body) != 0 {
		return fmt.Errorf("dist: hop: ping reply type=%d bytes=%d", typ, len(body))
	}
	return nil
}

// Peer returns the immutable worker handshake.
func (h *Hop) Peer() Hello { return h.peer }

// Close closes the transport when it supports io.Closer.
func (h *Hop) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if c, ok := h.conn.(io.Closer); ok {
		return c.Close()
	}
	return nil
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
		out, final, err := hop.Forward(ActivationHeader{SeqID: seq, Pos: pos, NTokens: uint32(ntokens), DType: hop.mine.DType}, cur)
		if err != nil {
			return nil, fmt.Errorf("dist: hop %d: %w", i, err)
		}
		last := i == len(p.Hops)-1
		if final != last {
			return nil, fmt.Errorf("dist: hop %d final=%v, want %v", i, final, last)
		}
		cur = out
	}
	return cur, nil
}

// Reset fans a sequence reset out to the local shard and every hop.
func (p *Pipeline) Reset(seq uint32) error {
	var errs []error
	if p.Local != nil {
		if err := p.Local.Reset(seq); err != nil {
			errs = append(errs, fmt.Errorf("dist: local reset: %w", err))
		}
	}
	// Reset every hop even when an earlier one failed: retaining state on the
	// remaining workers would make sequence-id reuse unsafe.
	for i, hop := range p.Hops {
		if err := hop.Reset(seq); err != nil {
			errs = append(errs, fmt.Errorf("dist: hop %d reset: %w", i, err))
		}
	}
	return errors.Join(errs...)
}

// ListenAndServe runs a worker on a TCP listener, one connection at a time
// (a shard serves exactly one coordinator).
func ListenAndServe(addr string, w *Worker) error {
	return ListenAndServeContext(context.Background(), addr, w)
}

// ListenAndServeContext runs a single-coordinator worker until cancellation.
func ListenAndServeContext(ctx context.Context, addr string, w *Worker) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	defer ln.Close()
	go func() {
		<-ctx.Done()
		_ = ln.Close()
	}()
	for {
		conn, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil || errors.Is(err, net.ErrClosed) {
				return ctx.Err()
			}
			return err
		}
		if tc, ok := conn.(*net.TCPConn); ok {
			_ = tc.SetNoDelay(true) // decode is latency-bound: never Nagle a token
			_ = tc.SetKeepAlive(true)
		}
		if err := w.Serve(conn); err != nil {
			log.Printf("dist: worker connection ended: %v", err)
		}
		_ = conn.Close()
	}
}

// DialTCP establishes a persistent low-latency hop and performs the handshake.
func DialTCP(ctx context.Context, addr string, mine Hello) (*Hop, error) {
	d := net.Dialer{KeepAlive: 30 * 1e9}
	conn, err := d.DialContext(ctx, "tcp", addr)
	if err != nil {
		return nil, err
	}
	if tc, ok := conn.(*net.TCPConn); ok {
		_ = tc.SetNoDelay(true)
	}
	hop, err := DialHop(conn, mine)
	if err != nil {
		_ = conn.Close()
		return nil, err
	}
	return hop, nil
}
