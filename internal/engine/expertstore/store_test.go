// ABOUTME: Verifies expert-store tier hits, LRU caps, checksums, and prefetch behavior.
package expertstore

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"testing"
)

type fakeUploader struct {
	mu      sync.Mutex
	live    map[Key][]byte
	evicted []Key
}

func (u *fakeUploader) Upload(k Key, b []byte) error {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.live[k] = append([]byte(nil), b...)
	return nil
}
func (u *fakeUploader) Evict(k Key) error {
	u.mu.Lock()
	defer u.mu.Unlock()
	delete(u.live, k)
	u.evicted = append(u.evicted, k)
	return nil
}

func testStore(t *testing.T, vram, host int64) (*Store, *fakeUploader) {
	t.Helper()
	a, b := []byte("expert-A"), []byte("expert-B")
	blob := append(append([]byte{}, a...), b...)
	hash := func(x []byte) string { s := sha256.Sum256(x); return hex.EncodeToString(s[:]) }
	idx := map[Key]Record{{0, 0}: {Offset: 0, Length: len(a), SHA256: hash(a)},
		{0, 1}: {Offset: int64(len(a)), Length: len(b), SHA256: hash(b)}}
	u := &fakeUploader{live: make(map[Key][]byte)}
	s, err := New(bytes.NewReader(blob), idx, u, vram, host)
	if err != nil {
		t.Fatal(err)
	}
	return s, u
}

func TestStoreLRUAndHostPromotion(t *testing.T) {
	s, u := testStore(t, 8, 16)
	if err := s.Ensure(Key{0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := s.Ensure(Key{0, 1}); err != nil {
		t.Fatal(err)
	}
	if len(u.live) != 1 || len(u.evicted) != 1 {
		t.Fatalf("live=%d evicted=%v", len(u.live), u.evicted)
	}
	if err := s.Ensure(Key{0, 0}); err != nil {
		t.Fatal(err)
	}
	m := s.Metrics()
	if m.SSDReads != 2 || m.HostHits != 1 || m.Evictions != 2 {
		t.Fatalf("metrics=%+v", m)
	}
	if m.VRAMBytes > 8 || m.HostBytes > 16 {
		t.Fatalf("budgets exceeded: %+v", m)
	}
}

func TestStoreChecksumFailure(t *testing.T) {
	s, _ := testStore(t, 8, 0)
	rec := s.index[Key{0, 0}]
	rec.SHA256 = "deadbeef"
	s.index[Key{0, 0}] = rec
	if err := s.Ensure(Key{0, 0}); err == nil {
		t.Fatal("expected checksum failure")
	}
	if s.Metrics().ChecksumFailures != 1 {
		t.Fatal("checksum metric not incremented")
	}
}

func TestPrefetch(t *testing.T) {
	s, u := testStore(t, 16, 0)
	done := make(chan struct{})
	for err := range s.Prefetch(done, []Key{{0, 0}, {0, 1}}, 2) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(u.live) != 2 || s.Metrics().Promotions != 2 || s.Metrics().Prefetches != 2 {
		t.Fatalf("live=%d metrics=%+v", len(u.live), s.Metrics())
	}
	if err := s.Ensure(Key{0, 0}); err != nil {
		t.Fatal(err)
	}
	if s.Metrics().PrefetchHits != 1 {
		t.Fatalf("prefetch hit not recorded: %+v", s.Metrics())
	}
}

func TestPredictorLearnsHotSuccessor(t *testing.T) {
	p := NewPredictor(4)
	for i := 0; i < 5; i++ {
		p.Observe(3, []int{1, 2}, []int{7, 8})
	}
	p.Observe(3, []int{1}, []int{9})
	got := p.Predict(3, []int{1}, 2)
	if len(got) != 2 || got[0] != (Key{3, 7}) || got[1] != (Key{3, 8}) {
		t.Fatalf("prediction=%v", got)
	}
	if other := p.Predict(4, []int{1}, 2); len(other) != 0 {
		t.Fatalf("cross-layer prediction=%v", other)
	}
}
