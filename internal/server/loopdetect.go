package server

// cycleDetector detects degenerate repetition loops in a generated token
// stream. The Req3 incident generated 8192 tokens (130s of GPU, the full
// runaway cap) of which ~5300 were a verbatim cycle the speculative drafter
// happily accelerated — with no detector, the only bound on a loop is
// max_tokens.
//
// Detection is true period detection on the emitted token ids: trigger when the
// trailing `max(minRepeats*p, minSpan)` tokens are p-periodic for some period
// p ≤ maxPeriod. This deliberately does NOT use the engine's lookup-acceptance
// signal: a coding agent legitimately re-emits file contents verbatim, which
// also yields ~100% lookup acceptance but is not cyclic.
//
// Cost: one ring append plus, for each candidate period whose quick
// single-token probe matches, an O(span) scan — microseconds against a
// multi-millisecond generation step.
type cycleDetector struct {
	buf []int32
}

const (
	cycleMaxPeriod  = 256 // longest repeating unit considered
	cycleMinRepeats = 4   // unit must repeat at least this many times...
	cycleMinSpan    = 256 // ...and the periodic tail must span at least this many tokens
	// Ring capacity: must hold the largest checked span (minRepeats*maxPeriod)
	// plus one extra period for the comparison offset.
	cycleBufCap = cycleMinRepeats*cycleMaxPeriod + cycleMaxPeriod
)

// push appends a token and reports whether the stream has entered a cycle.
func (d *cycleDetector) push(t int32) bool {
	if d.buf == nil {
		d.buf = make([]int32, 0, 2*cycleBufCap)
	}
	d.buf = append(d.buf, t)
	if len(d.buf) > 2*cycleBufCap { // amortized O(1) trim
		d.buf = append(d.buf[:0], d.buf[len(d.buf)-cycleBufCap:]...)
	}
	n := len(d.buf)
	for p := 1; p <= cycleMaxPeriod; p++ {
		span := cycleMinRepeats * p
		if span < cycleMinSpan {
			span = cycleMinSpan
		}
		if n < span+p {
			break // longer periods need even more history — none can match
		}
		if t != d.buf[n-1-p] {
			continue // quick probe: last token must repeat at distance p
		}
		periodic := true
		for i := 2; i <= span; i++ { // i=1 is the probe above
			if d.buf[n-i] != d.buf[n-i-p] {
				periodic = false
				break
			}
		}
		if periodic {
			return true
		}
	}
	return false
}
