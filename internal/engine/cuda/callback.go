// cgo-exported callbacks live in their own file: a file containing //export
// directives may not define functions in its C preamble (only declarations),
// so the static wrappers stay in bridge.go.

package cuda

// #include <stdint.h>
import "C"

import (
	"log"
	"runtime/cgo"
	"unsafe"
)

// fucinaSpecTokenGo is the per-token bridge the speculative engine calls during
// GenerateSpecStream. ud is a runtime/cgo.Handle whose value is the request's
// emit closure (func(int32) bool); returning 1 tells the engine to stop
// generation after this token.
//
// A panic here would unwind THROUGH the C frame that invoked us, which the Go
// runtime treats as fatal and unrecoverable — taking down the whole shared
// inference process for one request's bad token. The deferred recover keeps the
// panic on the Go side and returns 1 (stop generation) so the engine unwinds
// cleanly and the handler sees a short stream instead of a process crash. The
// recover MUST live in this frame (the one C calls directly); recovering higher
// up is too late, the fatal unwind through C has already happened.
//
//export fucinaSpecTokenGo
func fucinaSpecTokenGo(tok C.int32_t, ud unsafe.Pointer) (ret C.int) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("fucina: PANIC in token callback (token %d): %v", int32(tok), r)
			ret = 1 // stop generation; never let the panic reach C
		}
	}()
	h := cgo.Handle(uintptr(ud))
	emit := h.Value().(func(int32) bool)
	if emit(int32(tok)) {
		return 1
	}
	return 0
}
