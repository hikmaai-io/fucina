// ABOUTME: Provides the host orchestration for a budgeted VRAM/host/SSD expert cache.
// ABOUTME: Keeps storage, checksums, LRU eviction, and prefetch independent of CUDA upload kernels.
package expertstore

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"sync"
)

// Key identifies one complete expert payload (all projections in the store's chosen codec).
type Key struct{ Layer, Expert int }

// Record points into the immutable SSD expert blob.
type Record struct {
	Offset int64  `json:"offset"`
	Length int    `json:"length"`
	SHA256 string `json:"sha256"`
}

// Uploader owns actual device allocations/copies. Store calls it while promoting or evicting.
type Uploader interface {
	Upload(Key, []byte) error
	Evict(Key) error
}

type cacheEntry struct {
	key   Key
	data  []byte
	bytes int
}
type deviceEntry struct {
	key   Key
	bytes int
}

type Metrics struct {
	VRAMHits, HostHits, SSDReads, ChecksumFailures uint64
	Promotions, Evictions, BytesRead               uint64
	VRAMBytes, HostBytes                           int64
}

// Store is concurrency-safe. The first implementation serializes a miss through ReaderAt;
// Prefetch overlaps it with model work by running in caller-selected worker goroutines.
type Store struct {
	mu                     sync.Mutex
	reader                 io.ReaderAt
	index                  map[Key]Record
	uploader               Uploader
	vramBudget, hostBudget int64
	vram                   map[Key]*list.Element
	host                   map[Key]*list.Element
	vramLRU, hostLRU       list.List
	metrics                Metrics
}

func New(reader io.ReaderAt, index map[Key]Record, uploader Uploader, vramBudget, hostBudget int64) (*Store, error) {
	if reader == nil || uploader == nil {
		return nil, fmt.Errorf("expertstore: reader and uploader are required")
	}
	if vramBudget < 0 || hostBudget < 0 {
		return nil, fmt.Errorf("expertstore: negative budget")
	}
	return &Store{reader: reader, index: index, uploader: uploader, vramBudget: vramBudget, hostBudget: hostBudget,
		vram: make(map[Key]*list.Element), host: make(map[Key]*list.Element)}, nil
}

// Ensure makes key device-resident, promoting from host or SSD and enforcing both budgets.
func (s *Store) Ensure(key Key) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.vram[key]; e != nil {
		s.vramLRU.MoveToFront(e)
		s.metrics.VRAMHits++
		return nil
	}
	rec, ok := s.index[key]
	if !ok {
		return fmt.Errorf("expertstore: no record for layer %d expert %d", key.Layer, key.Expert)
	}
	var data []byte
	if e := s.host[key]; e != nil {
		s.hostLRU.MoveToFront(e)
		s.metrics.HostHits++
		data = e.Value.(*cacheEntry).data
	} else {
		data = make([]byte, rec.Length)
		n, err := s.reader.ReadAt(data, rec.Offset)
		if err != nil && err != io.EOF {
			return fmt.Errorf("expertstore: read %v: %w", key, err)
		}
		if n != rec.Length {
			return fmt.Errorf("expertstore: short read %v: %d/%d", key, n, rec.Length)
		}
		s.metrics.SSDReads++
		s.metrics.BytesRead += uint64(n)
		if rec.SHA256 != "" {
			sum := sha256.Sum256(data)
			if hex.EncodeToString(sum[:]) != rec.SHA256 {
				s.metrics.ChecksumFailures++
				return fmt.Errorf("expertstore: checksum mismatch for %v", key)
			}
		}
		s.addHost(key, data)
	}
	if int64(len(data)) > s.vramBudget {
		return fmt.Errorf("expertstore: expert %v (%d bytes) exceeds VRAM budget", key, len(data))
	}
	for s.metrics.VRAMBytes+int64(len(data)) > s.vramBudget {
		tail := s.vramLRU.Back()
		if tail == nil {
			break
		}
		old := tail.Value.(*deviceEntry)
		if err := s.uploader.Evict(old.key); err != nil {
			return fmt.Errorf("expertstore: evict %v: %w", old.key, err)
		}
		delete(s.vram, old.key)
		s.vramLRU.Remove(tail)
		s.metrics.VRAMBytes -= int64(old.bytes)
		s.metrics.Evictions++
	}
	if err := s.uploader.Upload(key, data); err != nil {
		return fmt.Errorf("expertstore: upload %v: %w", key, err)
	}
	e := s.vramLRU.PushFront(&deviceEntry{key: key, bytes: len(data)})
	s.vram[key] = e
	s.metrics.VRAMBytes += int64(len(data))
	s.metrics.Promotions++
	return nil
}

func (s *Store) addHost(key Key, data []byte) {
	if int64(len(data)) > s.hostBudget || s.hostBudget == 0 {
		return
	}
	for s.metrics.HostBytes+int64(len(data)) > s.hostBudget {
		tail := s.hostLRU.Back()
		if tail == nil {
			break
		}
		old := tail.Value.(*cacheEntry)
		delete(s.host, old.key)
		s.hostLRU.Remove(tail)
		s.metrics.HostBytes -= int64(old.bytes)
	}
	copyData := append([]byte(nil), data...)
	e := s.hostLRU.PushFront(&cacheEntry{key: key, data: copyData, bytes: len(copyData)})
	s.host[key] = e
	s.metrics.HostBytes += int64(len(copyData))
}

// Prefetch promotes keys concurrently. Cancellation stops scheduling new work, not an in-flight pread.
func (s *Store) Prefetch(done <-chan struct{}, keys []Key, workers int) <-chan error {
	if workers < 1 {
		workers = 1
	}
	jobs := make(chan Key)
	errs := make(chan error, 1)
	var wg sync.WaitGroup
	worker := func() {
		defer wg.Done()
		for key := range jobs {
			if err := s.Ensure(key); err != nil {
				select {
				case errs <- err:
				default:
				}
			}
		}
	}
	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go worker()
	}
	go func() {
		defer close(jobs)
		for _, key := range keys {
			select {
			case <-done:
				return
			case jobs <- key:
			}
		}
	}()
	go func() { wg.Wait(); close(errs) }()
	return errs
}

func (s *Store) Metrics() Metrics { s.mu.Lock(); defer s.mu.Unlock(); return s.metrics }
