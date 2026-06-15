package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sync/atomic"
	"time"
)

// sseWriter owns the SSE wire protocol for one streaming request.
//
// It exists to fix the worst perceived-latency property the server had: NO
// bytes (not even the status line) reached the client until the entire prefill
// finished — a 19-100s suffix prefill was indistinguishable, at the socket
// level, from a hung server, and intermediaries with header timeouts killed the
// request. begin() puts the headers + role delta on the wire within ~1ms of
// validation, and a ": ping" comment heartbeat keeps the connection visibly
// alive through the prefill (SSE comments are invisible to spec-compliant
// parsers, so clients see liveness, not content).
//
// Every flush arms a per-request write deadline (via http.ResponseController):
// a client that stalls without disconnecting (SIGSTOP, laptop sleep) would
// otherwise block conn.Write inside the engine's token callback FOREVER while
// holding the KV lock — there is no server WriteTimeout (deliberately: one
// would kill long streams).
//
// Concurrency: the heartbeat goroutine is the ONLY writer between begin() and
// stopHeartbeat() (the handler is parked inside the blocking prefill); after
// stopHeartbeat() returns (it joins), the handler goroutine is the only writer.
// No further synchronization is needed — but the discipline is load-bearing.
type sseWriter struct {
	w        http.ResponseWriter
	flusher  http.Flusher
	rc       *http.ResponseController
	legacy   bool
	model    string
	id       string
	object   string
	created  int64
	began    bool
	writeErr atomic.Bool // a write deadline expired: client is stuck/gone
	hbStop   chan struct{}
	hbDone   chan struct{}
}

const sseWriteTimeout = 30 * time.Second

// newSSEWriter prepares (but does not start) an SSE session. ok=false when the
// ResponseWriter cannot stream.
func newSSEWriter(w http.ResponseWriter, legacy bool, model string) (*sseWriter, bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, false
	}
	object, idPrefix := "chat.completion.chunk", "chatcmpl"
	if legacy {
		object, idPrefix = "text_completion", "cmpl"
	}
	return &sseWriter{
		w:       w,
		flusher: flusher,
		rc:      http.NewResponseController(w),
		legacy:  legacy,
		model:   model,
		id:      fmt.Sprintf("%s-%d", idPrefix, time.Now().UnixNano()),
		object:  object,
		created: time.Now().Unix(),
	}, true
}

// begin writes the SSE headers, the 200 status line, and (for chat) the role
// delta — the client's proof of life — and flushes. Call BEFORE the kv lock /
// prefill. After begin, errors must be reported in-stream (errorEvent), not via
// http.Error: the status is already on the wire.
func (e *sseWriter) begin() {
	if e.began {
		return
	}
	e.began = true
	h := e.w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no") // defeat proxy buffering (nginx et al.)
	e.w.WriteHeader(http.StatusOK)
	if !e.legacy {
		e.writeEvent(StreamResponse{
			ID: e.id, Object: e.object, Created: e.created, Model: e.model,
			Choices: []StreamChoice{{Index: 0, Delta: Delta{Role: "assistant"}}},
		})
	}
	e.flush()
}

// startHeartbeat emits ": ping" comments every interval until stopHeartbeat.
// Start it right before the blocking prefill; the handler must not write until
// stopHeartbeat has returned.
func (e *sseWriter) startHeartbeat(interval time.Duration) {
	e.hbStop = make(chan struct{})
	e.hbDone = make(chan struct{})
	go func() {
		defer close(e.hbDone)
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-e.hbStop:
				return
			case <-t.C:
				e.ping()
			}
		}
	}()
}

// stopHeartbeat stops the heartbeat goroutine and JOINS it, guaranteeing no
// concurrent writes once it returns. Idempotent.
func (e *sseWriter) stopHeartbeat() {
	if e.hbStop == nil {
		return
	}
	close(e.hbStop)
	<-e.hbDone
	e.hbStop = nil
}

// ping writes an SSE comment line (invisible to SSE parsers) and flushes. A
// raw-write error (broken pipe before any flush deadline fires) marks the writer
// dead so generation can stop instead of writing into a closed socket.
func (e *sseWriter) ping() {
	if _, err := fmt.Fprint(e.w, ": ping\n\n"); err != nil {
		e.writeErr.Store(true)
	}
	e.flush()
}

// writeEvent marshals v as one `data:` event (no flush — pair with flush()).
func (e *sseWriter) writeEvent(v interface{}) {
	data, _ := json.Marshal(v)
	if _, err := fmt.Fprintf(e.w, "data: %s\n\n", data); err != nil {
		e.writeErr.Store(true)
	}
}

// event writes + flushes one data event.
func (e *sseWriter) event(v interface{}) {
	e.writeEvent(v)
	e.flush()
}

// errorEvent reports a server-side failure in-stream (the OpenAI streaming
// convention once the 200 is out) followed by [DONE].
func (e *sseWriter) errorEvent(msg string) {
	e.event(map[string]interface{}{
		"error": map[string]string{"message": msg, "type": "server_error"},
	})
	e.done()
}

// done writes the terminal [DONE] sentinel and flushes.
func (e *sseWriter) done() {
	if _, err := fmt.Fprint(e.w, "data: [DONE]\n\n"); err != nil {
		e.writeErr.Store(true)
	}
	e.flush()
}

// flush pushes buffered bytes to the socket under a fresh write deadline. A
// deadline error marks the writer dead (stalled client) so generation can stop;
// ErrNotSupported (test recorders) downgrades to a plain Flush.
//
// Toolchain assumption (verified on go1.26): net/http clears the connection
// write deadline after every request, so these per-flush deadlines cannot leak
// into the next request on a reused keep-alive connection (true since go1.21).
func (e *sseWriter) flush() {
	if err := e.rc.SetWriteDeadline(time.Now().Add(sseWriteTimeout)); err != nil {
		if errors.Is(err, http.ErrNotSupported) {
			e.flusher.Flush()
			return
		}
	}
	if err := e.rc.Flush(); err != nil {
		if errors.Is(err, http.ErrNotSupported) {
			e.flusher.Flush()
			return
		}
		e.writeErr.Store(true)
	}
}

// stalled reports whether a write deadline expired (client stopped reading).
func (e *sseWriter) stalled() bool { return e.writeErr.Load() }
