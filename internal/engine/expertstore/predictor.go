// ABOUTME: Learns bounded per-layer expert transitions for lookahead prefetch.
package expertstore

import (
	"sort"
	"sync"
)

// Predictor records transitions from every expert selected at token t to every expert selected
// at token t+1 in the same layer. It is deliberately small and online; no prompt text is retained.
type Predictor struct {
	mu            sync.RWMutex
	counts        map[Key]map[Key]uint32
	maxSuccessors int
}

func NewPredictor(maxSuccessors int) *Predictor {
	if maxSuccessors < 1 {
		maxSuccessors = 32
	}
	return &Predictor{counts: make(map[Key]map[Key]uint32), maxSuccessors: maxSuccessors}
}

func (p *Predictor) Observe(layer int, previous, current []int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, a := range previous {
		from := Key{Layer: layer, Expert: a}
		m := p.counts[from]
		if m == nil {
			m = make(map[Key]uint32)
			p.counts[from] = m
		}
		for _, b := range current {
			to := Key{Layer: layer, Expert: b}
			if m[to] < ^uint32(0) {
				m[to]++
			}
		}
		if len(m) > p.maxSuccessors*2 {
			prune(m, p.maxSuccessors)
		}
	}
}

type scoredKey struct {
	key   Key
	count uint32
}

func prune(m map[Key]uint32, n int) {
	x := make([]scoredKey, 0, len(m))
	for k, c := range m {
		x = append(x, scoredKey{k, c})
	}
	sort.Slice(x, func(i, j int) bool {
		if x[i].count != x[j].count {
			return x[i].count > x[j].count
		}
		return x[i].key.Expert < x[j].key.Expert
	})
	for _, v := range x[n:] {
		delete(m, v.key)
	}
}

// Predict returns the most frequent union of successors for the current top-k set.
func (p *Predictor) Predict(layer int, current []int, limit int) []Key {
	if limit < 1 {
		return nil
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	total := make(map[Key]uint64)
	for _, e := range current {
		for k, c := range p.counts[Key{Layer: layer, Expert: e}] {
			total[k] += uint64(c)
		}
	}
	x := make([]struct {
		k Key
		c uint64
	}, 0, len(total))
	for k, c := range total {
		x = append(x, struct {
			k Key
			c uint64
		}{k, c})
	}
	sort.Slice(x, func(i, j int) bool {
		if x[i].c != x[j].c {
			return x[i].c > x[j].c
		}
		return x[i].k.Expert < x[j].k.Expert
	})
	if len(x) > limit {
		x = x[:limit]
	}
	out := make([]Key, len(x))
	for i := range x {
		out[i] = x[i].k
	}
	return out
}
