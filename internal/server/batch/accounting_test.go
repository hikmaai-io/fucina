package batch

import "testing"

func TestReuseInfoNormalized(t *testing.T) {
	got := (ReuseInfo{ReusedTokens: -3, Source: ReuseSourceHostSnapshot}).Normalized()
	if got.ReusedTokens != 0 || got.Source != ReuseSourceNone {
		t.Fatalf("normalized=%+v", got)
	}
}

func TestAdoptedCachedTokens(t *testing.T) {
	for _, tc := range []struct {
		shared int
		want   int
	}{
		{shared: -1, want: 0},
		{shared: 0, want: 0},
		{shared: 1, want: 0},
		{shared: 255, want: 0},
		{shared: 256, want: 256},
		{shared: 511, want: 256},
		{shared: 512, want: 512},
		{shared: 767, want: 512},
	} {
		if got := AdoptedCachedTokens(tc.shared); got != tc.want {
			t.Fatalf("shared=%d got=%d want=%d", tc.shared, got, tc.want)
		}
	}
}
