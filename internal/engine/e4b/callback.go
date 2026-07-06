// cgo-exported callbacks live in their own file: a file containing //export
// directives may not define functions in its C preamble (only declarations), so
// the static spec-stream wrapper stays in bridge.go.

package e4b

// #include <stdint.h>
import "C"

import (
	"log"
	"runtime/cgo"
	"unsafe"
)

// fucinaE4BSpecTokenGo is the per-token bridge the E4B spec loop calls during
// SpecStream. ud is a runtime/cgo.Handle whose value is the request's emit
// closure (func(int32) bool); returning 1 tells the engine to stop generation
// after this token.
//
// A panic here would unwind THROUGH the C frame that invoked us, which the Go
// runtime treats as fatal — taking down the whole inference process for one
// request's bad token. The deferred recover keeps the panic on the Go side and
// returns 1 (stop) so the engine unwinds cleanly. The recover MUST live in this
// frame (the one C calls directly).
//
//export fucinaE4BSpecTokenGo
func fucinaE4BSpecTokenGo(tok C.int32_t, ud unsafe.Pointer) (ret C.int) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("fucina: PANIC in E4B token callback (token %d): %v", int32(tok), r)
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
