// cgo-exported callbacks live in their own file: a file containing //export
// directives may not define functions in its C preamble (only declarations),
// so the static wrappers stay in bridge.go.

package cuda

// #include <stdint.h>
import "C"

import (
	"runtime/cgo"
	"unsafe"
)

// fucinaSpecTokenGo is the per-token bridge the speculative engine calls during
// GenerateSpecStream. ud is a runtime/cgo.Handle whose value is the request's
// emit closure (func(int32) bool); returning 1 tells the engine to stop
// generation after this token.
//
//export fucinaSpecTokenGo
func fucinaSpecTokenGo(tok C.int32_t, ud unsafe.Pointer) C.int {
	h := cgo.Handle(uintptr(ud))
	emit := h.Value().(func(int32) bool)
	if emit(int32(tok)) {
		return 1
	}
	return 0
}
