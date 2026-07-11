// ABOUTME: Adapts prefetch and admission limits when SSD latency or miss pressure rises.
package expertstore

import (
	"sync"
	"time"
)

type Limits struct{ PrefetchDepth, Workers, MaxBatch int }
type PressureSample struct {
	SSDLatency     time.Duration
	PendingMisses  int
	PrefetchUseful bool
}

// Controller uses conservative hysteresis: two pressure samples degrade immediately; four
// healthy samples recover one step. It never changes model math, only lookahead and admission.
type Controller struct {
	mu                sync.Mutex
	limits, min, max  Limits
	highLatency       time.Duration
	pressure, healthy int
}

func NewController(initial, min, max Limits, highLatency time.Duration) *Controller {
	if highLatency <= 0 {
		highLatency = 5 * time.Millisecond
	}
	return &Controller{limits: initial, min: min, max: max, highLatency: highLatency}
}

func (c *Controller) Observe(s PressureSample) Limits {
	c.mu.Lock()
	defer c.mu.Unlock()
	stressed := s.SSDLatency > c.highLatency || s.PendingMisses > 2*c.limits.Workers
	if stressed {
		c.pressure++
		c.healthy = 0
	} else {
		c.healthy++
		c.pressure = 0
	}
	if c.pressure >= 2 {
		c.limits.PrefetchDepth = maxInt(c.min.PrefetchDepth, c.limits.PrefetchDepth-1)
		c.limits.Workers = maxInt(c.min.Workers, c.limits.Workers-1)
		c.limits.MaxBatch = maxInt(c.min.MaxBatch, c.limits.MaxBatch/2)
		c.pressure = 0
	} else if c.healthy >= 4 {
		if s.PrefetchUseful {
			c.limits.PrefetchDepth = minInt(c.max.PrefetchDepth, c.limits.PrefetchDepth+1)
		}
		c.limits.Workers = minInt(c.max.Workers, c.limits.Workers+1)
		c.limits.MaxBatch = minInt(c.max.MaxBatch, c.limits.MaxBatch+1)
		c.healthy = 0
	}
	return c.limits
}
func (c *Controller) Limits() Limits { c.mu.Lock(); defer c.mu.Unlock(); return c.limits }
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
